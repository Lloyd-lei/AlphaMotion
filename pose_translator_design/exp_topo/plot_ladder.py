#!/usr/bin/env python
"""0903 尺寸阶梯画图: loss_log.jsonl (每 50 步) + eval_log.json (每 1000 步) + model.pt 参数量
   用法: plot_ladder.py <run_dir>... --out <prefix>
   产物: <prefix>_loss.png (各项 loss vs step, 每档一条线, 对数轴)
         <prefix>_scaling.png (末端 loss / eval rot vs 参数量, 对数-对数)
         <prefix>_chinchilla.png (approach-1: 第一段 backbone 的 loss vs 算力 C=6·N·D, 每档一条, 取下包络)
         <prefix>.md (表)"""
import json, os, sys, argparse, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--tail", type=int, default=500, help="末端 loss = 最后 N 步均值")
a = ap.parse_args()
TERMS = ["rot", "pos", "total_main", "reenc", "hpair", "slip_w", "contrast", "judge", "rootpos", "cgeo"]

def n_params(d):
    p = f"{d}/model.pt"
    if not os.path.exists(p): return None
    import torch; sd = torch.load(p, map_location="cpu")
    return sum(v.numel() for v in sd.values())

runs = []
for d in a.runs:
    lp = f"{d}/loss_log.jsonl"
    if not os.path.exists(lp):
        if os.path.exists(f"{d}_s1/loss_log.jsonl"): d = f"{d}_s1"; lp = f"{d}/loss_log.jsonl"; print(f"[note] 第二段还没开始, 用 {d}")
        else: print(f"[skip] {d}: 无 loss_log.jsonl"); continue
    rows = [json.loads(l) for l in open(lp) if l.strip()]
    ev = json.load(open(f"{d}/eval_log.json")) if os.path.exists(f"{d}/eval_log.json") else []
    # 两段血统: <run>_s1 (F32 段) 接在前面, 第二段 step 加上第一段总步数 (图上一条连续曲线, 8000 处有配方切换)
    s1 = f"{d.rstrip('/')}_s1"
    if os.path.exists(f"{s1}/loss_log.jsonl"):
        r1 = [json.loads(l) for l in open(f"{s1}/loss_log.jsonl") if l.strip()]
        off = r1[-1]["step"] if r1 else 0
        for w in rows: w["step"] += off
        for e in ev: e["step"] += off
        e1 = json.load(open(f"{s1}/eval_log.json")) if os.path.exists(f"{s1}/eval_log.json") else []
        rows = r1 + rows; ev = e1 + ev
    else:
        r1, e1 = [], []
    cfg = json.load(open(f"{d}/config.json")) if os.path.exists(f"{d}/config.json") else {}
    npar = n_params(d) or n_params(s1)
    if not r1 and os.path.basename(d.rstrip("/")).endswith("_s1"): r1, e1 = rows, ev   # 只传了 _s1 目录
    runs.append(dict(name=os.path.basename(d.rstrip("/")), rows=rows, ev=ev, np=npar, cfg=cfg, rows1=r1, ev1=e1))
if not runs: sys.exit("没有可画的 run")

# ---- 图 1: loss vs step ----
terms = [t for t in TERMS if any(t in r["rows"][-1] for r in runs)]
nc = 5; nr = math.ceil(len(terms) / nc)
fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr)); axs = np.atleast_1d(axs).ravel()
for ax, t in zip(axs, terms):
    for r in runs:
        xs = [w["step"] for w in r["rows"] if t in w]; ys = [w[t] for w in r["rows"] if t in w]
        lab = f'{r["name"]} ({r["np"]/1e6:.1f}M)' if r["np"] else r["name"]
        ax.plot(xs, ys, lw=1.2, label=lab)
    ax.set_title(t); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.3, which="both")
for ax in axs[len(terms):]: ax.axis("off")
axs[0].legend(fontsize=7); fig.suptitle("FZ5c 配方尺寸阶梯: 训练 loss vs step (log-log)"); fig.tight_layout()
fig.savefig(f"{a.out}_loss.png", dpi=110); print("WROTE", f"{a.out}_loss.png")

# ---- 图 2: 末端 vs 参数量 ----
def tail_mean(r, t):
    v = [w[t] for w in r["rows"] if t in w and w["step"] > r["rows"][-1]["step"] - a.tail]
    return float(np.mean(v)) if v else float("nan")
def ev_last(r, key):
    for e in reversed(r["ev"]):
        if key in e: return e[key]["rot"] if isinstance(e[key], dict) else e[key]
    return float("nan")
EVK = [k for k in ("human_smpl->human_smpl", "unitree_g1_29dof->unitree_g1_29dof", "human_smpl->unitree_g1_29dof", "human_smpl->fourier_gr3", "human_smpl->pnd_adam_lite") if runs[0]["ev"] and k in runs[0]["ev"][-1]]
sized = [r for r in runs if r["np"]]
if sized:
  fig, axs = plt.subplots(1, 2, figsize=(10, 4))
  for t in ["rot", "pos", "total_main", "reenc"]:
      if all(t in r["rows"][-1] for r in sized):
          axs[0].plot([r["np"] for r in sized], [tail_mean(r, t) for r in sized], "o-", label=t)
  axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("params"); axs[0].set_title(f"末端训练 loss (最后 {a.tail} 步均值)"); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3, which="both")
  for k in EVK[:6]:
      axs[1].plot([r["np"] for r in sized], [ev_last(r, k) for r in sized], "o-", label=k)
  axs[1].set_xscale("log"); axs[1].set_xlabel("params"); axs[1].set_title("eval 旋转误差 (°, 最后一次 eval)"); axs[1].legend(fontsize=6); axs[1].grid(alpha=.3, which="both")
  for ax in axs:
      for r in sized: ax.annotate(r["name"], (r["np"], ax.get_ylim()[0]), fontsize=6, rotation=90, va="bottom")
  fig.tight_layout(); fig.savefig(f"{a.out}_scaling.png", dpi=110); print("WROTE", f"{a.out}_scaling.png")

# ---- 图 3: Chinchilla approach-1, 第一段 backbone: loss vs 算力 ----
# D = 帧×具身对 = step·bs·combo_sample (每步 256 帧 × 8 对); C = 6·N·D. 单位是"帧对"不是 token,
# 三档每步 token 数相同, 所以曲线间的相对位置与 token 定义无关, 只有横轴标度差一个常数.
def smooth(y, k=9):
    y = np.asarray(y, float); return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y
c1 = [r for r in runs if r["np"] and r["rows1"]]
if len(c1) >= 1:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    env = []
    for r in c1:
        bs = r["cfg"].get("bs", 256) or 256; cs = r["cfg"].get("combo_sample", 8) or 8
        xs = np.array([w["step"] for w in r["rows1"] if "rot" in w]) * bs * cs * 6 * r["np"]
        ys = np.array([w["rot"] for w in r["rows1"] if "rot" in w])
        ys_s = smooth(ys); xs_s = xs[len(xs) - len(ys_s):]
        lab = f'{r["name"]} ({r["np"]/1e6:.1f}M, {r["rows1"][-1]["step"]} 步)'
        axs[0].plot(xs_s, ys_s, lw=1.3, label=lab); env += list(zip(xs_s, ys_s))
        ex = [e["step"] * bs * cs * 6 * r["np"] for e in r["ev1"] if "human_smpl->human_smpl" in e]
        ey = [e["human_smpl->human_smpl"]["rot"] for e in r["ev1"] if "human_smpl->human_smpl" in e]
        axs[1].plot(ex, ey, "o-", ms=3, lw=1.2, label=lab)
    if env:
        env.sort(); best = []; m = float("inf")
        for x, y in env:
            if y < m: m = y; best.append((x, y))
        axs[0].plot(*zip(*best), "k--", lw=0.8, alpha=.6, label="下包络 (每算力最优)")
    for ax, t in zip(axs, ["第一段训练 rot loss (9 点平滑)", "第一段 eval h→h 旋转误差 (°)"]):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("C = 6·N·D  (D = 帧×具身对)"); ax.set_title(t); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    fig.suptitle("Chinchilla approach-1: backbone 段, 同数据同 lr, 哪个尺寸在给定算力下 loss 最低"); fig.tight_layout()
    fig.savefig(f"{a.out}_chinchilla.png", dpi=110); print("WROTE", f"{a.out}_chinchilla.png")

# ---- 表 ----
with open(f"{a.out}.md", "w") as f:
    hdr = ["run", "params", "steps", "s/step", "wall_h"] + [f"L_{t}" for t in ["rot", "pos", "total_main", "reenc"]] + EVK[:4]
    f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for r in runs:
        last = r["rows"][-1]
        row = [r["name"], f'{r["np"]/1e6:.1f}M' if r["np"] else "?", str(last["step"]), f'{last.get("s_per_step", float("nan")):.2f}',
               f'{last.get("wall_s", 0)/3600:.2f}'] + [f'{tail_mean(r, t):.3f}' for t in ["rot", "pos", "total_main", "reenc"]] + [f'{ev_last(r, k):.2f}' for k in EVK[:4]]
        f.write("| " + " | ".join(row) + " |\n")
print(open(f"{a.out}.md").read())
