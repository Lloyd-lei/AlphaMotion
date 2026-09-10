"""0831 lloyd 问题: "不丢弃位置头效果如何" —— 特别是 LAFAN GROUND/趴地片段。

解码器有两个头: hrot (旋转) 和 hpos (位置)。渲染只读 hrot, hpos 在推理时被丢弃;
但 w_pos=5.0 的位置项训练的正是 hpos (0812 审计: 占梯度 85%)。--fk_loss 改成对
预测旋转做可微 FK, 于是 hpos 完全失去监督。

本探针只做推理, 不训练:
  1  hpos 直出位置 vs FK-of-hrot vs GT 的逐关节位置误差 (cm), 分组 mean/p95
  2  两个头的自洽差 |hpos - FK(hrot)| (同帧)
  3  (--ik) 从 hpos 预测位置反解关节角 (manifold_project.lm_fit, 位置项) 的残差
     与最终 vs GT 误差, 对比 hrot 直解角度的路径

窗口三卷: FIXED (判官 v2_windows.json, val 片段, 无 ground) / GROUND (13 条
fall|ground|crawl|getUp 片段) / CTRL (非 ground 的训练片段, 与 GROUND 同为训练行,
用来把"内容差异"和"训练/留出差异"分开 —— 没有这条对照就读不出 ground 的锅)。

用法:
  python hpos_keep_probe.py --ckpts PT_c1dec_reenc_0812,P0V2_base_0830,P0V2_fk_0830 \
      --out /tmp/hpos_keep.json [--ik]
"""
from __future__ import annotations
import os, sys, json, argparse, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import rot6d_to_matrix                                    # noqa: E402
from dual_common import build_model, geom, fk_pos_torch, _depth       # noqa: E402
import v2_judge as VJ                                                 # noqa: E402

RUNS = VJ.RUNS
LAFAN = VJ.LAFAN
SRC, TGTS, T, N_WIN = VJ.SRC, VJ.TGTS, VJ.T, VJ.N_WIN
GROUND_RE = re.compile(r"ground|fall|getUp|crawl", re.I)


# ---------------------------------------------------------------- 关节分组
def groups(spec, device):
    """root / hand(拓扑叶且在臂链) / arm(其余臂) / leg / torso。
    hand 用拓扑判定 (parents 里没出现过 = 末端叶), 不认名字里有没有 'hand'。"""
    from dual_common import GRP_ARM_KW, GRP_LEG_KW
    nl = [n.lower() for n in spec.joint_names]
    J = spec.J
    par = list(spec.parents)
    is_parent = np.zeros(J, bool)
    for p in par:
        if p >= 0:
            is_parent[int(p)] = True
    leaf = ~is_parent
    arm = np.array([any(k in n for k in GRP_ARM_KW) for n in nl])
    leg = np.array([any(k in n for k in GRP_LEG_KW) for n in nl]) & ~arm
    root = np.array([int(p) < 0 for p in par])
    # 关键词表按名字判组, 但末端连杆常叫 end_effector / rubber_hand, 不含臂关键词
    # (gr3 的 *_end_effector_link 就是), 所以叶子沿树上溯继承最近祖先的组别。
    # 拓扑判定, 不靠命名运气 —— 否则 gr3 的"手"格子会是空的。
    arm0, leg0 = arm.copy(), leg.copy()
    for j in range(J):
        if not leaf[j] or arm0[j] or leg0[j]:
            continue
        k = int(par[j])
        while k >= 0 and not (arm0[k] or leg0[k]):
            k = int(par[k])
        if k >= 0:
            (arm if arm0[k] else leg)[j] = True
    hand = leaf & arm & ~root
    out = {"root": root, "hand": hand, "arm": arm & ~hand & ~root,
           "leg": leg & ~root, "torso": ~(arm | leg | root)}
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


# ---------------------------------------------------------------- 窗口三卷
def window_sets(seed=123):
    seqs = json.load(open(f"{LAFAN}/sequences.json"))
    fixed = json.load(open(VJ.WIN_FILE))
    val_clips = set(fixed["val_clips"])
    ground = [s for s in seqs if GROUND_RE.search(s["name"]) and s["len"] >= T + 2]
    ctrl = [s for s in seqs if not GROUND_RE.search(s["name"])
            and s["name"] not in val_clips and s["len"] >= T + 2]
    def sample(pool, rng):
        w = []
        for _ in range(N_WIN):
            s = pool[int(rng.integers(len(pool)))]
            w.append({"clip": s["name"],
                      "start": int(rng.integers(s["start"], s["start"] + s["len"] - T))})
        return w
    g = np.random.default_rng(seed)
    return {"FIXED": fixed["windows"],
            "GROUND": sample(ground, np.random.default_rng(seed)),
            "CTRL": sample(ctrl, np.random.default_rng(seed + 1))}, \
           {"ground_clips": sorted({s["name"] for s in ground}),
            "ctrl_pool": len(ctrl), "val_clips": sorted(val_clips)}


def stats(e, m):
    """e [R,J] cm, m [J] bool -> (mean, p95)"""
    v = e[:, m.cpu()].reshape(-1)
    if v.numel() == 0:
        return None, None
    return round(float(v.mean()), 2), round(float(v.quantile(.95)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="PT_c1dec_reenc_0812,P0V2_base_0830,P0V2_fk_0830")
    ap.add_argument("--out", default="/tmp/hpos_keep.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ik", action="store_true", help="第 3 部分: 位置->角度反解 (慢)")
    ap.add_argument("--ik_frames", type=int, default=180)
    ap.add_argument("--ik_ckpt", default="PT_c1dec_reenc_0812")
    ap.add_argument("--ik_set", default="GROUND")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    wsets, meta = window_sets()
    print(f"[windows] GROUND 片段池: {meta['ground_clips']}")
    print(f"[windows] CTRL 池 {meta['ctrl_pool']} 段 (非 ground 且非 val 片段)")
    all_rows, seg = [], {}
    off = 0
    for k, ws in wsets.items():
        r = torch.cat([torch.arange(w["start"], w["start"] + T) for w in ws])
        seg[k] = slice(off, off + len(r)); off += len(r)
        all_rows.append(r)
    rows = torch.cat(all_rows)
    print(f"[rows] 共 {len(rows)} 行 = 3 卷 x {N_WIN} 窗 x {T} 帧")

    res = {"meta": meta, "sets": {k: wsets[k] for k in wsets}, "arms": {}}
    for ckname in [c for c in args.ckpts.split(",") if c]:
        ck = ckname if os.path.isdir(ckname) else f"{RUNS}/{ckname}"
        cfg = json.load(open(f"{ck}/config.json"))
        caches = [c for c in str(cfg.get("cache", LAFAN)).split(",") if c]
        assert "cache_xwbc_lafan" in os.path.basename(caches[0].rstrip("/"))
        from semantics import build_name_embeddings
        names_all = []
        for n in [SRC] + TGTS:
            d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
            names_all += [str(x) for x in d["joint_names"]]
        lut = build_name_embeddings(names_all, device=dev,
                                    encoder=cfg.get("sem_encoder", "siglip"))
        data = {n: VJ.load_emb(n, rows, caches, cfg, lut, dev) for n in [SRC] + TGTS}
        model = build_model(cfg, dev)
        model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
        model.eval()
        arm_res = {"ckpt": ck, "fk_loss": bool(cfg.get("fk_loss", False)),
                   "rot_only": bool(cfg.get("rot_only", False)),
                   "w_pos": cfg.get("w_pos"), "dirs": {}}
        R = len(rows)
        with torch.no_grad():
            for tgt in TGTS:
                spec, reach = data[tgt]["spec"], data[tgt]["reach"]
                gm = groups(spec, dev)
                e_hp, e_fk, e_gap, e_hpfk = [], [], [], []
                for i in range(0, R, 512):
                    sl = torch.arange(i, min(i + 512, R)); B = len(sl)
                    prot, ppos = model.decode(*model.encode(data[SRC]["pose"][sl],
                                                            geom(data, SRC, B)),
                                              geom(data, tgt, B))
                    gt = data[tgt]["pose"][sl]
                    gtp = gt[..., 6:9] * reach                       # cm, 根在原点
                    hp = ppos * reach                                # hpos 直出 (cm)
                    fk = fk_pos_torch(prot, spec)                    # 渲染路径 (cm)
                    e_hp.append((hp - gtp).norm(dim=-1).cpu())
                    e_fk.append((fk - gtp).norm(dim=-1).cpu())
                    e_gap.append((hp - fk).norm(dim=-1).cpu())
                    # 自检: FK(GT 全局旋转) 应与缓存位置逐位一致
                    e_hpfk.append((fk_pos_torch(gt[..., :6], spec) - gtp).norm(dim=-1).cpu())
                E = {"hpos": torch.cat(e_hp), "fkrot": torch.cat(e_fk),
                     "gap": torch.cat(e_gap), "selfcheck": torch.cat(e_hpfk)}
                d_out = {}
                for wk, s in seg.items():
                    row = {}
                    for metric, ten in E.items():
                        for gname, m in gm.items():
                            mu, p95 = stats(ten[s], m)
                            row[f"{metric}_{gname}_mean"] = mu
                            row[f"{metric}_{gname}_p95"] = p95
                        mu, p95 = stats(ten[s], torch.ones_like(gm["leg"]))
                        row[f"{metric}_all_mean"] = mu; row[f"{metric}_all_p95"] = p95
                    d_out[wk] = row
                arm_res["dirs"][tgt] = {"n_joints": {k: int(v.sum()) for k, v in gm.items()},
                                        "reach_cm": round(float(reach), 2), "sets": d_out}
                sc = float(E["selfcheck"].mean())
                print(f"[{ckname}|{tgt}] selfcheck FK(GT)-cache = {sc:.4f} cm (应 ~0)")
                for wk in seg:
                    r = d_out[wk]
                    print(f"  {wk:7s} all: hpos {r['hpos_all_mean']:6.2f}/{r['hpos_all_p95']:6.2f}  "
                          f"fk(rot) {r['fkrot_all_mean']:6.2f}/{r['fkrot_all_p95']:6.2f}  "
                          f"gap {r['gap_all_mean']:6.2f}/{r['gap_all_p95']:6.2f}  (mean/p95 cm)")
        res["arms"][ckname] = arm_res
        del model, data
        torch.cuda.empty_cache()

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")

    # ---------------------------------------------------------- 3 反解可行性
    if args.ik:
        ik_report(args, wsets, seg, rows, dev, res)
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"-> {args.out} (含 IK)")


def ik_report(args, wsets, seg, rows, dev, res):
    """从 hpos 预测位置反解关节角, 和从 hrot 直解角度对比。
    两条路都用 manifold_project 的整链 LM (lm_fit), 唯一区别是目标项:
      ROT 路: w_rot=1, 目标 = 预测的全局旋转 (= 现行渲染路径 pilot_decode_viz)
      POS 路: w_pos=1, w_rot=0, 目标 = hpos 预测位置 (根旋转仍取预测旋转, 因为根
              是自由关节, 位置反解不出根朝向 —— 这一点如实计入"混合路的代价")
    """
    from manifold_project import fit_angles, lm_fit, fk_from_angles, load_dof
    from common import geodesic_deg
    ck = args.ik_ckpt if os.path.isdir(args.ik_ckpt) else f"{RUNS}/{args.ik_ckpt}"
    cfg = json.load(open(f"{ck}/config.json"))
    caches = [c for c in str(cfg["cache"]).split(",") if c]
    tgt = "unitree_g1_29dof"
    s = seg[args.ik_set]
    sub = torch.arange(s.start, s.stop)[:args.ik_frames]
    from semantics import build_name_embeddings
    names_all = []
    for n in [SRC, tgt]:
        d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=dev, encoder=cfg.get("sem_encoder", "siglip"))
    data = {n: VJ.load_emb(n, rows[sub], caches, cfg, lut, dev) for n in [SRC, tgt]}
    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
    model.eval()
    spec, reach = data[tgt]["spec"], data[tgt]["reach"]
    B = len(sub)
    with torch.no_grad():
        prot, ppos = model.decode(*model.encode(data[SRC]["pose"], geom(data, SRC, B)),
                                  geom(data, tgt, B))
    gt = data[tgt]["pose"]
    gtp = gt[..., 6:9] * reach
    hp, fk = ppos * reach, fk_pos_torch(prot, spec)
    dof, rest = load_dof(caches[0], tgt, device=dev)
    Rg = rot6d_to_matrix(prot)
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])

    out = {"ckpt": ck, "tgt": tgt, "set": args.ik_set, "frames": int(B)}
    # ---- ROT 路 (现行): 角度拟合预测旋转
    q_rot, active = fit_angles(prot, spec, dof, rest=rest, clamp=True, method="global")
    gR_r, gp_r = fk_from_angles(q_rot, spec, dof, rest=rest, root_R=Rg[:, root])
    # ---- POS 路: 同一个 LM, 目标换成 hpos 位置
    q0, _ = fit_angles(prot, spec, dof, rest=rest, clamp=True, method="greedy")
    q_pos = lm_fit(Rg, spec, dof, q0, rest=rest, root_R=Rg[:, root], target_p=hp,
                   w_rot=0.0, w_pos=1.0, clamp=True, iters=60)
    gR_p, gp_p = fk_from_angles(q_pos, spec, dof, rest=rest, root_R=Rg[:, root])
    gm = groups(spec, dev)
    def rep(name, gp, gR):
        d = {}
        e_gt = (gp - gtp).norm(dim=-1).cpu()
        e_hp = (gp - hp).norm(dim=-1).cpu()
        e_fk = (gp - fk).norm(dim=-1).cpu()
        for gname, m in gm.items():
            d[f"vs_gt_{gname}_mean"], d[f"vs_gt_{gname}_p95"] = stats(e_gt, m)
        d["vs_gt_all_mean"], d["vs_gt_all_p95"] = stats(e_gt, torch.ones_like(gm["leg"]))
        d["resid_vs_hpos_mean"] = round(float(e_hp.mean()), 2)
        d["resid_vs_hpos_p95"] = round(float(e_hp.reshape(-1).quantile(.95)), 2)
        d["resid_vs_fkrot_mean"] = round(float(e_fk.mean()), 2)
        d["rot_geo_vs_gt_mean_deg"] = round(float(geodesic_deg(
            gR, rot6d_to_matrix(gt[..., :6])).mean()), 2)
        out[name] = d
        print(f"[IK {name}] vs GT all {d['vs_gt_all_mean']}/{d['vs_gt_all_p95']} cm; "
              f"手 {d['vs_gt_hand_mean']}/{d['vs_gt_hand_p95']}; "
              f"残差 vs hpos {d['resid_vs_hpos_mean']} cm; "
              f"旋转 geo {d['rot_geo_vs_gt_mean_deg']} deg")
    rep("path_rot", gp_r, gR_r)
    rep("path_pos", gp_p, gR_p)
    # 参照: 未经角度投影的两个头本身
    out["raw_hpos_vs_gt_all_mean"] = round(float((hp - gtp).norm(dim=-1).mean()), 2)
    out["raw_fkrot_vs_gt_all_mean"] = round(float((fk - gtp).norm(dim=-1).mean()), 2)
    res["ik"] = out


if __name__ == "__main__":
    main()
