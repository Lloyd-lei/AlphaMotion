"""xwbc AMASS 配对缓存时间轴修复 (0901).

xwbc 的机器人流 (GMR retarget) 对非 120fps 源按每 4 帧取 1 采样 (KIT 100fps -> 快 1.2x,
TotalCapture/CMU-60fps -> 快 2x), 人体流 human_motion 是对的。prep_xwbc_cache 按最短截断只对齐首帧,
片段中段人体第 t 帧配到机器人第 t/r 帧 (r = 人体帧数/机器人帧数)。

修法: 机器人流按人体时间轴重采样 new[t] = robot[t / r] (线性插值, rot6d 重正交), 人体流不动,
行布局 (sequences.json) 不动。只改 {emb}_paired_local6d.pt / {emb}_root_world.pt, 其余文件软链。
接触标签是派生量, 修完后用 contact_label.py 对新缓存重算。

用法: fix_timebase.py <src_cache> <out_cache> <lengths.json>   (lengths: name -> {nh, ng})
"""
import os, sys, json, glob, numpy as np, torch
SRC, OUT, LEN = sys.argv[1:4]
TOL = 0.05
os.makedirs(OUT, exist_ok=True)
seqs = json.load(open(f"{SRC}/sequences.json"))
L = json.load(open(LEN))
robots = sorted({os.path.basename(p)[:-len("_paired_local6d.pt")] for p in glob.glob(f"{SRC}/*_paired_local6d.pt")} - {"human_smpl"})
# 软链所有非机器人逐帧文件
for p in sorted(glob.glob(f"{SRC}/*")):
    b = os.path.basename(p)
    if any(b == f"{r}_{k}" for r in robots for k in ("paired_local6d.pt", "root_world.pt")) or b.startswith("_"):
        continue
    dst = f"{OUT}/{b}"
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(p), dst)

mis = [(q, L[q["name"]]["nh"] / L[q["name"]]["ng"]) for q in seqs if q["name"] in L and abs(L[q["name"]]["nh"] / L[q["name"]]["ng"] - 1) >= TOL]
print(f"[fix] 错位片段 {len(mis)}/{len(seqs)}  行 {sum(q['len'] for q, _ in mis)}  机器人 {len(robots)} 个", flush=True)

def orth6d(x):
    a = x[..., :3]; b = x[..., 3:]
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.cat([a, b], -1)

def resample(X, s, n, r, is6d):
    """X[s:s+n] 按 j = t/r 重采样 (源行也在 s..s+n 内, j <= (n-1)/r < n)."""
    j = torch.arange(n, dtype=torch.float64) / r
    j0 = j.floor().long().clamp(max=n - 1); j1 = (j0 + 1).clamp(max=n - 1); w = (j - j0).float()
    seg = X[s:s + n].float()
    shp = (n,) + (1,) * (seg.dim() - 1)
    y = seg[j0] * (1 - w.view(shp)) + seg[j1] * w.view(shp)
    return orth6d(y) if is6d else y

for emb in robots:
    for key, is6d in (("paired_local6d", True), ("root_world", False)):
        src = torch.load(f"{SRC}/{emb}_{key}.pt", mmap=True)
        X = src.clone()
        for q, r in mis:
            X[q["start"]:q["start"] + q["len"]] = resample(src, q["start"], q["len"], r, is6d).to(X.dtype)
        torch.save(X, f"{OUT}/{emb}_{key}.pt"); del X, src
    print(f"[fix] {emb} done", flush=True)

json.dump({"src": SRC, "lengths": LEN, "tol": TOL, "rule": "robot_new[t] = robot[t / (nh/ng)] (linear, rot6d re-orthonormalized); human untouched",
           "ratio_source": "unitree_g1_29dof npz frame counts (all robots share GMR sampling)",
           "n_misaligned_clips": len(mis), "misaligned_rows": int(sum(q["len"] for q, _ in mis)),
           "by_dataset": {k: sum(1 for q, _ in mis if q["name"].startswith(k + "__")) for k in sorted({q["name"].split("__")[0] for q, _ in mis})},
           "note": "接触标签需用 contact_label.py 对本缓存重算"}, open(f"{OUT}/TIMEBASE_FIX_meta.json", "w"), indent=1)
print("[fix] meta written", flush=True)
