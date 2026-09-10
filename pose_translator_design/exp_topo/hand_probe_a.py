"""探针 A —— 编码器对手臂盲不盲 (0831)。

主实验用"子树移植"造严格单变量帧对, 不靠检索:
  human_smpl 的 local6d 里, 手臂子树 = {L/R_Thorax, L/R_Shoulder, L/R_Elbow, L/R_Wrist}
  (8 个关节), 腿子树 = {L/R_Hip, L/R_Knee, L/R_Ankle, L/R_Toe} (8 个关节)。
  把 A 帧的手臂 local 旋转换成 B 帧的 -> 躯干和腿的全局旋转/位置逐位不变, 只有
  6 个关节的位置 + 8 个关节的旋转变了。腿侧同理, 结构完全对称。
  => 条件组位移严格 = 0, 不用"近似匹配", 没有检索偏差。

对每对量:
  码距    FSQ 整数格码逐维不一致率 (hamming) + 平均 |Δlevel| (pose/rot 两条流)
  量化前  proj_out(tanh(features)).mean(1) 余弦距离 (照抄 zspace_battery 特征路径)
  解码响应 解码到目标具身后该组的位移 / GT 目标该组的位移
判读: 同样位移量级下 ARM 的码距 << LEG 的码距 => 编码器丢手臂 (病 B)。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hand_probe_common as H
from common import fk_global, matrix_to_rot6d
from dual_common import fk_pos_torch
import v2_judge as VJ

ARM_SUB = [13, 14, 16, 17, 18, 19, 20, 21]      # L/R_Thorax,Shoulder,Elbow,Wrist
LEG_SUB = [1, 2, 4, 5, 7, 8, 10, 11]            # L/R_Hip,Knee,Ankle,Toe
ARM_MOVED = [16, 17, 18, 19, 20, 21]            # 位置真的会动的 6 个
LEG_MOVED = [4, 5, 7, 8, 10, 11]
BINS = [(2, 5), (5, 10), (10, 20), (20, 40), (40, 80)]    # 变化组逐关节平均位移 (cm)
LEG_TGT_KW = VJ.HIP_KW + VJ.KNEE_KW + ("ankle", "foot", "toe")


def pose9_from_l6(l6, spec, reach):
    gR, gp = fk_global(l6, spec)
    return torch.cat([matrix_to_rot6d(gR), gp / reach], -1)


@torch.no_grad()
def one_kind(model, src, tgts, kind, rows_a, rows_b, dev, bs=256):
    sub = ARM_SUB if kind == "arm" else LEG_SUB
    moved = ARM_MOVED if kind == "arm" else LEG_MOVED
    other = LEG_MOVED if kind == "arm" else ARM_MOVED
    tgt_kw = VJ.UL_KW if kind == "arm" else LEG_TGT_KW
    gm = {t.name: VJ.kw_mask(t.spec, tgt_kw, dev) for t in tgts}
    acc = {k: [] for k in ("vary_cm", "other_cm", "ham_p", "ham_r", "l1_p", "l1_r", "cosd")}
    for t in tgts:
        acc[f"dec_{t.name}"] = []; acc[f"gt_{t.name}"] = []
    for i in range(0, len(rows_a), bs):
        ra, rb = rows_a[i:i + bs], rows_b[i:i + bs]
        B = len(ra)
        la = src.L[ra].to(dev).float()
        lb = src.L[rb].to(dev).float()
        lm = la.clone(); lm[:, sub] = lb[:, sub]              # 子树移植
        pa = pose9_from_l6(la, src.spec, src.reach)
        pm = pose9_from_l6(lm, src.spec, src.reach)
        fa_pos = fk_pos_torch(pa[..., :6], src.spec)
        fm_pos = fk_pos_torch(pm[..., :6], src.spec)
        d = (fa_pos - fm_pos).norm(dim=-1)                    # [B,22] cm
        acc["vary_cm"].append(d[:, moved].mean(-1).cpu())
        acc["other_cm"].append(d[:, other].mean(-1).cpu())
        ga = src.geom(B)
        fa = model.encode_features(pa, ga); fm = model.encode_features(pm, ga)
        ia, im = model.ints_from_features(*fa), model.ints_from_features(*fm)
        acc["ham_p"].append((ia[0] != im[0]).float().mean(dim=(0, 2, 3)).cpu())
        acc["ham_r"].append((ia[1] != im[1]).float().mean(dim=(0, 2, 3)).cpu())
        acc["l1_p"].append((ia[0] - im[0]).abs().float().mean(dim=(0, 2, 3)).cpu())
        acc["l1_r"].append((ia[1] - im[1]).abs().float().mean(dim=(0, 2, 3)).cpu())
        za = torch.cat([model.fsqp.proj_out(torch.tanh(fa[0])).mean(1),
                        model.fsqr.proj_out(torch.tanh(fa[1])).mean(1)], -1)
        zm = torch.cat([model.fsqp.proj_out(torch.tanh(fm[0])).mean(1),
                        model.fsqr.proj_out(torch.tanh(fm[1])).mean(1)], -1)
        acc["cosd"].append(1 - torch.nn.functional.cosine_similarity(za, zm, -1).cpu())
        zpa, zra = model.encode(pa, ga); zpm, zrm = model.encode(pm, ga)
        for t in tgts:
            gt_ = t.geom(B)
            da, _ = model.decode(zpa, zra, gt_)
            dm, _ = model.decode(zpm, zrm, gt_)
            fda, fdm = fk_pos_torch(da, t.spec), fk_pos_torch(dm, t.spec)
            acc[f"dec_{t.name}"].append((fda - fdm)[:, gm[t.name]].norm(dim=-1).mean(-1).cpu())
            # GT 侧: 目标具身在"同一次移植"下应有的位移 —— 用配对缓存的目标 local6d
            # 做同样的子树移植不可行 (关节不同构), 所以用配对行的 GT 差作为参考量级:
            # GT_a(行 ra) vs GT_b(行 rb) 在该组的位移, 再乘以人体侧移植/换帧的位移比。
            tla = t.L[ra].to(dev).float(); tlb = t.L[rb].to(dev).float()
            pta = pose9_from_l6(tla, t.spec, t.reach)
            ptb = pose9_from_l6(tlb, t.spec, t.reach)
            fta, ftb = fk_pos_torch(pta[..., :6], t.spec), fk_pos_torch(ptb[..., :6], t.spec)
            acc[f"gt_{t.name}"].append((fta - ftb)[:, gm[t.name]].norm(dim=-1).mean(-1).cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=6000)   # 每类抽多少对
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0); np.random.seed(0)
    ck, cfg, model, Bd = H.load_ckpt(a.ckpt, dev)
    tag = a.tag or os.path.basename(ck.rstrip("/"))
    src = Bd[H.SRC]; tgts = [Bd[t] for t in H.TGTS]
    N = src.L.shape[0]
    clip, _ = H.clip_of_row(N)
    g = np.random.default_rng(7)
    ra = np.sort(g.choice(N, a.n, replace=False))
    rb = g.choice(N, a.n, replace=False)
    keep = clip[ra] != clip[rb]                       # 供体来自别的片段
    ra, rb = ra[keep], rb[keep]
    ra = torch.as_tensor(ra); rb = torch.as_tensor(rb)
    print(f"[A] {len(ra)} 对 (跨片段供体)", flush=True)

    res = {"ckpt": ck, "tag": tag, "n_pairs": int(len(ra)), "design": "subtree_transplant",
           "bins": {}}
    for kind in ("arm", "leg"):
        r = one_kind(model, src, tgts, kind, ra, rb, dev)
        v = r["vary_cm"].numpy()
        res["bins"][kind] = {}
        res[f"{kind}_other_group_move_cm"] = round(float(r["other_cm"].mean()), 6)  # 必须 0
        for lo, hi in BINS:
            s = np.flatnonzero((v >= lo) & (v < hi))
            if len(s) < 30:
                res["bins"][kind][f"{lo}-{hi}"] = {"n": int(len(s)), "note": "样本不足"}
                continue
            row = {"n": int(len(s)),
                   "vary_cm": round(float(v[s].mean()), 2),
                   "ham_pose": round(float(r["ham_p"][s].mean()), 4),
                   "ham_rot": round(float(r["ham_r"][s].mean()), 4),
                   "l1_pose": round(float(r["l1_p"][s].mean()), 4),
                   "l1_rot": round(float(r["l1_r"][s].mean()), 4),
                   "cos_dist": round(float(r["cosd"][s].mean()), 5)}
            row["cos_per_cm"] = round(row["cos_dist"] / max(row["vary_cm"], 1e-9), 6)
            row["ham_rot_per_cm"] = round(row["ham_rot"] / max(row["vary_cm"], 1e-9), 6)
            for t in tgts:
                dd = float(r[f"dec_{t.name}"][s].mean()); gg = float(r[f"gt_{t.name}"][s].mean())
                row[f"decmove_{t.name}_cm"] = round(dd, 2)
                row[f"gtmove_{t.name}_cm"] = round(gg, 2)
                row[f"resp_{t.name}"] = round(dd / max(gg, 1e-9), 3)
                # 更干净的一把尺子: 解码位移 / 人体侧同组位移。arm 和 leg 用同一
                # 具身、同一 bin 的人体位移做分母, 具身尺度因子对两者完全相同,
                # 所以 arm-vs-leg 的比值可以直接比 (resp_ 那个分母是两独立行的
                # GT 差, 只作量级参考)。
                row[f"resph_{t.name}"] = round(dd / max(float(v[s].mean()), 1e-9), 3)
            res["bins"][kind][f"{lo}-{hi}"] = row
            print(f"[A] {kind:3s} {lo:2d}-{hi:2d}cm n={row['n']:5d} vary={row['vary_cm']:5.2f} "
                  f"ham_p={row['ham_pose']:.4f} ham_r={row['ham_rot']:.4f} "
                  f"cos_d={row['cos_dist']:.5f} cos/cm={row['cos_per_cm']:.5f} "
                  f"resp_g1={row['resp_unitree_g1_29dof']:.3f} "
                  f"resp_gr3={row['resp_fourier_gr3']:.3f}", flush=True)

    out = a.out or f"{H.RUNS}/HANDPROBE_A_{tag}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("->", out)


if __name__ == "__main__":
    main()
