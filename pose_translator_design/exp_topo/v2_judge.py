"""Greenwich-v2 P0 判官 — 一张表, 四项预注册指标 (spec §6 判据, 0830 定)。

    v2_judge.py --ckpt <train_dual run 目录或 runs/ 下的名字> [--tag base]

  a  幅度比      解码/GT 逐关节时间 std 之比 (all/arm/leg) —— 复用 dual_common.
                 amplitude_report, 0813 punch 判决用的那把尺子, 只是窗口固定。
  b  上肢误差    肩/肘/腕链 FK 位置误差 vs GT, mean/p95 (cm)。走 FK-of-rotations
                 而不是 hpos —— 渲染看的是旋转头, hpos 出画不了图 (0812 审计)。
  c  腿角        髋/膝全局旋转 geodesic 误差 vs GT, mean/p95 (deg)。
                 预注册的回归护栏: lloyd 当前的抱怨就是腿; 不许退化。
  d  方言逐维    human vs robot 同一动作行的 FSQ 逐维格码一致率 + 量化前余弦
                 (编码套路照抄 zspace_battery: proj_out(tanh(features)).mean(1))。

为什么单独立判官而不复用 evaluate(): P0 消融的判据必须"所有臂同一批固定窗口"。
evaluate() 用随机 val 行(无时间结构, 量不了幅度), amplitude_report 自己抽窗口
(pools 一换卷子就换)。这里把窗口列表落盘 (v2_windows.json, 固定种子), 谁来考
都是同一份卷子; 生产 checkpoint 和小规模 pilot 都能考 (config.json 驱动建模)。

窗口的"留出"口径 (如实记录, 不装): train_dual 的 val split 是打散的单行, 拼不出
连续窗口, 所以完全行级留出做不到。这里按仓库的片段级惯例 (default_rng(123) 取
10% 片段, 同 bridge_a.CodeData) 在 LAFAN 77 片段里取 val 片段抽窗; 所有臂在行级
见过这些帧的程度相同, 且 human->g1 方向对每条臂 (含生产) 都是留出具身 = 真零样本。

reach 口径: 编码器输入是 gp/reach, reach = 训练缓存全量行的 max —— 判官按 ckpt
config 里的 cache 列表逐缓存算 max 再取最大 (逐帧独立, 与训练逐位同值), 结果
落盘 v2_reach_cache.json, 只算一次。喂错 reach = 输入整体缩放 = 冤枉模型。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CACHE_ROOT, SkeletonSpec, fk_global, rot6d_to_matrix, geodesic_deg   # noqa: E402
from train_ae import static_cond                                                        # noqa: E402
from dual_common import (build_model, geom, fk_pos_torch, amplitude_report,             # noqa: E402
                         FINGER_KW)

RUNS = f"{CACHE_ROOT}/runs"
LAFAN = f"{CACHE_ROOT}/cache_xwbc_lafan"
WIN_FILE = os.path.join(HERE, "v2_windows.json")
WIN_FILE_GROUND = os.path.join(HERE, "v2_windows_ground.json")
REACH_FILE = os.path.join(HERE, "v2_reach_cache.json")
SRC = "human_smpl"
# g1 = 部署目标且在所有臂的 holdout 列表里 (真零样本护栏); gr3 = 训练内具身对照。
TGTS = ["unitree_g1_29dof", "fourier_gr3"]
T, N_WIN = 90, 24                       # 与 amplitude_report/amp_duel 的 0813 口径一致
# 关节关键词: 各家命名不同 (gr3 的肩肘腕叫 upper_arm/lower_arm/hand, 髋膝叫
# thigh/shank), 关键词表按 spec 命名实测覆盖 (0830 核对过 g1/gr3/human)。
UL_KW = ("shoulder", "elbow", "wrist", "upper_arm", "lower_arm", "hand", "clavicle")
HIP_KW = ("hip", "thigh")
KNEE_KW = ("knee", "shank")


def kw_mask(spec, kws, device):
    nl = [n.lower() for n in spec.joint_names]
    fin = [any(k.lower() in n for k in FINGER_KW) for n in nl]
    return torch.tensor([any(k in n for k in kws) and not f
                         for n, f in zip(nl, fin)], device=device)


def cache_reach(cache_dir, name, spec, device):
    """单个缓存里某具身的 reach (全量行 FK 位置模长最大值), 结果落盘复用。
    为什么逐块 max 而不是 build_global: 4.7M 行的全量 gr6/gp 要 6GB 内存,
    这里只要一个标量; 逐帧独立 => 数值与训练的 build_global 完全一致。"""
    rc = json.load(open(REACH_FILE)) if os.path.exists(REACH_FILE) else {}
    key = f"{cache_dir}::{name}"
    if key in rc:
        return rc[key]
    l6 = torch.load(f"{cache_dir}/{name}_paired_local6d.pt", mmap=True)
    mx = 0.0
    with torch.no_grad():
        for i in range(0, l6.shape[0], 8192):
            _, gp = fk_global(l6[i:i + 8192].to(device).float(), spec)
            mx = max(mx, float(gp.norm(dim=-1).max()))
    rc[key] = max(mx, 1e-6)
    json.dump(rc, open(REACH_FILE, "w"), indent=1)
    print(f"[reach] {name} @ {os.path.basename(cache_dir)}: {rc[key]:.2f}cm", flush=True)
    return rc[key]


def make_windows_ground():
    """地面卷子: 接触活儿的真考场。FIXED 那份卷子的片段是 aiming/dance/fight/sprint/
    walk, 地面片段为零 = 对接触工作完全盲测。

    诚实声明: LAFAN 的 16 个 ground/fall 片段**一个都不在 val 划分里** (帧级 10% 随机
    划分下它们的帧约 10% 未参与训练, 但相邻帧会泄漏)。所以这份卷子是**训练集内考试**,
    只能回答"机制有没有起作用", 不能回答"能不能泛化"。结果 json 里带 heldout=false,
    读表时必须连着这一条一起读。"""
    from contact_label import family
    seqs = json.load(open(f"{LAFAN}/sequences.json"))
    pool = [q for q in seqs if family(q["name"]) in ("ground", "fall", "getup", "crawl")
            and q["len"] >= T + 2]
    assert pool, "LAFAN 里找不到地面家族片段"
    g = np.random.default_rng(7)
    wins = [{"clip": (q := pool[int(g.integers(len(pool)))])["name"],
             "start": int(g.integers(q["start"], q["start"] + q["len"] - T))}
            for _ in range(N_WIN)]
    out = {"cache": LAFAN, "T": T, "n_win": N_WIN, "heldout": False,
           "note": "训练集内考试: LAFAN 地面片段不在 val 划分里",
           "val_clips": sorted({w["clip"] for w in wins}), "windows": wins}
    json.dump(out, open(WIN_FILE_GROUND, "w"), indent=1)
    print(f"[windows] 新建 {WIN_FILE_GROUND}: {N_WIN} 窗 x {T} 帧, "
          f"片段 {len(out['val_clips'])} 个 (训练集内)")
    return out


def make_windows():
    """固定窗口卷子: LAFAN 片段级 val split (default_rng(123) 取 10%, 与
    bridge_a.CodeData 同惯例) 里抽 N_WIN 个 T 帧窗, 种子固定, 落盘。"""
    seqs = json.load(open(f"{LAFAN}/sequences.json"))
    rng = np.random.default_rng(123)
    val = sorted(rng.permutation(len(seqs))[: len(seqs) // 10].tolist())
    pool = [seqs[i] for i in val if seqs[i]["len"] >= T + 2]
    g = np.random.default_rng(0)
    wins = []
    for _ in range(N_WIN):
        s = pool[int(g.integers(len(pool)))]
        wins.append({"clip": s["name"], "start": int(g.integers(s["start"], s["start"] + s["len"] - T))})
    out = {"cache": LAFAN, "T": T, "n_win": N_WIN, "val_clips": [seqs[i]["name"] for i in val],
           "windows": wins}
    json.dump(out, open(WIN_FILE, "w"), indent=1)
    print(f"[windows] 新建 {WIN_FILE}: {N_WIN} 窗 x {T} 帧, val 片段 {len(val)} 个")
    return out


def load_emb(name, rows, caches, cfg, lut, device):
    """判官版子集加载: 只加载需要的具身、只 FK 需要的窗口行 (2160 行 << 全量),
    但 reach 用训练口径 (见模块注释)。字段与 load_paired_data 的 entry 对齐,
    geom()/amplitude_report 原样复用。"""
    from semantics import embed_matrix
    d = np.load(f"{caches[0]}/{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
    reach = max(cache_reach(c, name, spec, device) for c in caches)
    l6 = torch.load(f"{LAFAN}/{name}_paired_local6d.pt", mmap=True)[rows]
    with torch.no_grad():
        gR, gp = fk_global(l6.to(device).float(), spec)
        from common import matrix_to_rot6d
        pose = torch.cat([matrix_to_rot6d(gR), gp / reach], -1)      # [R,J,9]
    entry = {"spec": spec, "st": static_cond(spec, device), "pose": pose,
             "reach": reach, "sem": embed_matrix(spec.joint_names, lut).to(device)}
    # 接触通道 (lloyd 0831 "距离进、置信出"): 有 contact_chan 的模型考试时**必须喂**
    # 符号距离 —— 接触规格是一等输入, 不喂等于把模型扔进没见过的配置里考。
    cf = f"{caches[0]}/{name}_contact.npz"
    if os.path.exists(cf):
        z_ = np.load(cf, allow_pickle=True)
        hdr_ = json.loads(str(z_["header"]))
        thr_, D_ = float(hdr_["thr_m"]), float(hdr_["D_used_m"])
        s_ = thr_ - z_["gap"][np.asarray(rows)].astype(np.float32)
        entry["sd"] = torch.as_tensor(
            np.sign(s_) * np.log1p(np.abs(s_) / thr_) / np.log1p(D_ / thr_),
            dtype=torch.float32, device=device)
        del z_
    dof_path = f"{caches[0]}/{name}_dof.npz"
    if os.path.exists(dof_path):
        entry["dof"] = torch.as_tensor(np.load(dof_path)["dof"], dtype=torch.float32, device=device)
        from dof_head import build_kin
        entry["kin"] = build_kin(caches[0], name, spec, device=device)
    print(f"[{name}] J={spec.J} rows={len(rows)} reach={reach:.1f}cm (训练口径)")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train_dual run 目录 (或 runs/ 下的名字)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--append", default=f"{RUNS}/P0V2_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wset", default="FIXED", choices=["FIXED", "GROUND"],
                    help="FIXED=与 P0 可比的老尺子; GROUND=地面/摔倒片段 (训练集内考试)")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    ck = args.ckpt if os.path.isdir(args.ckpt) else f"{RUNS}/{args.ckpt}"
    tag = args.tag or os.path.basename(ck.rstrip("/"))
    # 0831: 两份卷子的成绩必须分开存。原来 all_r[tag]=res 会让同一条臂的 GROUND
    # 直接覆盖掉它的 FIXED, 读表时看到的是最后跑的那份, 而且完全没有提示。
    if str(getattr(args, "wset", "FIXED")).upper() == "GROUND":
        tag = f"{tag}@GROUND"
    cfg = json.load(open(f"{ck}/config.json"))
    caches = [c for c in str(cfg.get("cache", LAFAN)).split(",") if c]
    # 窗口按 LAFAN 行号定义; 拼接缓存把 LAFAN 放在第一段时行号不变 (0809 起的约定),
    # 其他排布下同一行号指向不同帧 => 直接拒考, 不出错误的数字。
    # 0831: 按 sequences.json 的 realpath 判同源, 不按目录名。cache_lafan_v2 是同一份
    # sequences.json 的符号链farm (只换了 *_contact.npz), 行号逐位相同; 按名字判会把
    # 合法的标签升级挡在门外, 让四条臂一条成绩都出不来。
    assert os.path.realpath(os.path.join(caches[0].rstrip("/"), "sequences.json")) == \
           os.path.realpath(os.path.join(LAFAN, "sequences.json")), \
        f"判官窗口定义在 LAFAN 行号上, 但 ckpt 的第一个缓存是 {caches[0]} " \
        f"(它的 sequences.json 与 LAFAN 的不是同一份)"

    wset = getattr(args, "wset", "FIXED")
    if str(wset).upper() == "GROUND":
        win = (json.load(open(WIN_FILE_GROUND)) if os.path.exists(WIN_FILE_GROUND)
               else make_windows_ground())
    else:
        win = json.load(open(WIN_FILE)) if os.path.exists(WIN_FILE) else make_windows()
    assert win["T"] == T and win["n_win"] == N_WIN
    starts_abs = [w["start"] for w in win["windows"]]
    rows = torch.cat([torch.arange(s, s + T) for s in starts_abs])       # 绝对行号
    starts_cpt = list(range(0, len(rows), T))                            # 紧凑张量里的起点

    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
    model.eval()

    from semantics import build_name_embeddings
    names_all = []
    for n in [SRC] + TGTS:
        d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=dev,
                                encoder=cfg.get("sem_encoder", "siglip"))
    data = {n: load_emb(n, rows, caches, cfg, lut, dev) for n in [SRC] + TGTS}

    res = {"ckpt": ck, "tag": tag,
           "windows": WIN_FILE_GROUND if str(getattr(args, "wset", "FIXED")).upper() == "GROUND" else WIN_FILE,
           "wset": str(getattr(args, "wset", "FIXED")).upper(),
           "heldout": bool(win.get("heldout", True)),
           "dirs": {}, "dialect": {}}
    R = len(rows)
    with torch.no_grad():
        # ---- a 幅度比: 复用 amplitude_report, 窗口传死 (starts= 紧凑起点) ----
        amp = amplitude_report(model, data, [(SRC, t) for t in TGTS], pools=None,
                               n_win=N_WIN, T=T, starts=starts_cpt)
        # ---- b/c 上肢误差 + 腿角: 同一批窗口帧, 分块前向 ----
        for tgt in TGTS:
            spec = data[tgt]["spec"]
            ul = kw_mask(spec, UL_KW, dev)
            hip, knee = kw_mask(spec, HIP_KW, dev), kw_mask(spec, KNEE_KW, dev)
            leg = hip | knee
            es, gs = [], []
            for i in range(0, R, 512):
                sl = torch.arange(i, min(i + 512, R))
                B = len(sl)
                _con = (data[SRC]["sd"][sl] if getattr(model, "contact_chan", False)
                        and "sd" in data[SRC] else None)
                prot, _ = model.decode(*model.encode(data[SRC]["pose"][sl], geom(data, SRC, B),
                                                     con=_con),
                                       geom(data, tgt, B))
                gt6 = data[tgt]["pose"][sl][..., :6]
                es.append((fk_pos_torch(prot, spec) -
                           fk_pos_torch(gt6, spec)).norm(dim=-1).cpu())          # [B,J] cm
                gs.append(geodesic_deg(rot6d_to_matrix(prot),
                                       rot6d_to_matrix(gt6)).cpu())              # [B,J] deg
            e, gdeg = torch.cat(es), torch.cat(gs)
            ulcpu, legc, hipc, kneec = ul.cpu(), leg.cpu(), hip.cpu(), knee.cpu()
            k = f"{SRC}->{tgt}"
            res["dirs"][k] = dict(
                amp_all=amp[k]["all"], amp_arm=amp[k]["arm"], amp_leg=amp[k]["leg"],
                ul_mean_cm=round(float(e[:, ulcpu].mean()), 2),
                ul_p95_cm=round(float(e[:, ulcpu].reshape(-1).quantile(.95)), 2),
                leg_mean_deg=round(float(gdeg[:, legc].mean()), 2),
                leg_p95_deg=round(float(gdeg[:, legc].reshape(-1).quantile(.95)), 2),
                hip_mean_deg=round(float(gdeg[:, hipc].mean()), 2),
                knee_mean_deg=round(float(gdeg[:, kneec].mean()), 2))
        # ---- d 方言逐维: 配对行上 human vs robot 的码距 + 量化前余弦 ----
        for tgt in TGTS:
            agp, agr, l1p, l1r, cos = [], [], [], [], []
            for i in range(0, R, 512):
                sl = torch.arange(i, min(i + 512, R))
                B = len(sl)
                fh = model.encode_features(data[SRC]["pose"][sl], geom(data, SRC, B))
                fr = model.encode_features(data[tgt]["pose"][sl], geom(data, tgt, B))
                ih, ir = model.ints_from_features(*fh), model.ints_from_features(*fr)
                agp.append((ih[0] == ir[0]).float().mean(dim=(0, 1, 2)).cpu())   # [f] 逐维
                agr.append((ih[1] == ir[1]).float().mean(dim=(0, 1, 2)).cpu())
                l1p.append((ih[0] - ir[0]).abs().float().mean().cpu())
                l1r.append((ih[1] - ir[1]).abs().float().mean().cpu())
                zh = torch.cat([model.fsqp.proj_out(torch.tanh(fh[0])).mean(1),
                                model.fsqr.proj_out(torch.tanh(fh[1])).mean(1)], -1)
                zr = torch.cat([model.fsqp.proj_out(torch.tanh(fr[0])).mean(1),
                                model.fsqr.proj_out(torch.tanh(fr[1])).mean(1)], -1)
                cos.append(torch.nn.functional.cosine_similarity(zh, zr, -1).cpu())
            ap_, ar_ = torch.stack(agp).mean(0), torch.stack(agr).mean(0)
            cs = torch.cat(cos)
            res["dialect"][tgt] = dict(
                agree_pose=round(float(ap_.mean()), 3), agree_pose_min_dim=round(float(ap_.min()), 3),
                agree_rot=round(float(ar_.mean()), 3), agree_rot_min_dim=round(float(ar_.min()), 3),
                grid_l1_pose=round(float(np.mean(l1p)), 3), grid_l1_rot=round(float(np.mean(l1r)), 3),
                prequant_cos_med=round(float(cs.median()), 3),
                prequant_cos_p10=round(float(cs.quantile(.10)), 3))

    # ---- 一张表 ----
    hdr = (f"{'direction':32s} {'amp_all':>7s} {'amp_arm':>7s} {'amp_leg':>7s} "
           f"{'ul_mean':>7s} {'ul_p95':>7s} {'leg_mean':>8s} {'leg_p95':>7s} "
           f"{'agree_p':>7s} {'agree_r':>7s} {'cos_med':>7s}")
    print(f"\n== v2_judge [{tag}] {ck}")
    print(hdr); print("-" * len(hdr))
    for tgt in TGTS:
        d_, dl = res["dirs"][f"{SRC}->{tgt}"], res["dialect"][tgt]
        print(f"{SRC+'->'+tgt:32s} {d_['amp_all']:7.3f} {d_['amp_arm']:7.3f} {d_['amp_leg']:7.3f} "
              f"{d_['ul_mean_cm']:7.2f} {d_['ul_p95_cm']:7.2f} {d_['leg_mean_deg']:8.2f} "
              f"{d_['leg_p95_deg']:7.2f} {dl['agree_pose']:7.3f} {dl['agree_rot']:7.3f} "
              f"{dl['prequant_cos_med']:7.3f}")
    json.dump(res, open(f"{ck}/v2_judge.json", "w"), indent=1)
    all_r = json.load(open(args.append)) if os.path.exists(args.append) else {}
    all_r[tag] = res
    json.dump(all_r, open(args.append, "w"), indent=1)
    print(f"-> {ck}/v2_judge.json ; 汇总表 {args.append}")


if __name__ == "__main__":
    main()
