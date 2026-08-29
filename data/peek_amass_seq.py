"""Eyeball the rebuilt full-length AMASS: distributions + skeleton anims.

Outputs to data/peek/:
  overview.png            — length / fps / subset / boundaries-per-seq
  <name>_filmstrip.png    — 8 pose snapshots across time + action label
  <name>_timeline.png     — BABEL act_cat segments + change-point boundaries
  <name>.gif              — skeleton animation (side view, Y up)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path("/home/arenalabs/Desktop/\"be water, robot\"")
SEQ = REPO / "data" / "amass_seq"
OUT = REPO / "data" / "peek"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "notebook" / "01_resources" / "scripts"))
from soma_skeleton import build_parent_index                       # noqa: E402

PAR = build_parent_index()
BONES = [(j, PAR[j]) for j in range(len(PAR)) if PAR[j] >= 0]


def per_frame_cat(fl, T, fps):
    """Dominant act_cat[0] per frame (shortest covering segment wins)."""
    lab = [None] * T
    for s in sorted(fl, key=lambda s: -(s["end_t"] - s["start_t"])):
        ac = s.get("act_cat")
        c = ac[0] if isinstance(ac, list) and ac else None
        a = max(0, min(int(round(s["start_t"] * fps)), T - 1))
        b = max(0, min(int(round(s["end_t"] * fps)), T - 1))
        for i in range(a, b + 1):
            lab[i] = c
    return lab


def draw(ax, P):
    ax.clear()
    ax.scatter(P[:, 0], P[:, 1], s=4, c="tab:blue")
    for a, b in BONES:
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]],
                c="0.4", lw=1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def overview(man):
    T = np.array([m["T"] for m in man])
    dur = np.array([m["dur_s"] for m in man])
    fps = np.array([round(m["fps"]) for m in man])
    nb = np.array([m["n_frame_labels"] for m in man])
    sub = {}
    for m in man:
        sub[m["subset"]] = sub.get(m["subset"], 0) + 1

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].hist(np.clip(dur, 0, 60), bins=60)
    ax[0, 0].set_title(f"sequence duration (s), clipped@60  "
                       f"median={np.median(dur):.1f}s")
    vals, cnts = np.unique(fps, return_counts=True)
    ax[0, 1].bar([str(v) for v in vals], cnts)
    ax[0, 1].set_title("native fps (NOT downsampled to 30)")
    ax[1, 0].bar(list(sub.keys()), list(sub.values()))
    ax[1, 0].set_title("sequences per AMASS subset")
    ax[1, 0].tick_params(axis="x", rotation=90, labelsize=7)
    ax[1, 1].hist(np.clip(nb[nb > 0], 0, 40), bins=40)
    ax[1, 1].set_title(f"BABEL frame-segments / seq (only labelled)  "
                       f"n={int((nb>0).sum())}")
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=130)
    plt.close(fig)


def peek(m):
    d = np.load(m["file"], allow_pickle=True)
    P = d["joints_world"].astype(np.float32)               # [T,77,3] Y-up
    T = P.shape[0]
    fps = float(d["fps"])
    fl = list(d["babel_frame_labels"])
    cats = per_frame_cat(fl, T, fps)
    name = m["sample_name"]

    # filmstrip: 8 snapshots
    cols = 8
    fr = np.linspace(0, T - 1, cols).astype(int)
    fig, axs = plt.subplots(1, cols, figsize=(2.0 * cols, 3.2))
    for k, f in enumerate(fr):
        draw(axs[k], P[f])
        axs[k].set_title(f"{f/fps:4.1f}s\n{cats[f] or '-'}", fontsize=7)
    fig.suptitle(f"{name}  ({m['subset']}, {T}f @ {fps:.0f}fps, "
                 f"{m['dur_s']}s)  — Y-up side view", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_filmstrip.png", dpi=120)
    plt.close(fig)

    # action timeline + change-point boundaries
    ids = {None: 0}
    arr = np.zeros(T, int)
    for i, c in enumerate(cats):
        if c not in ids:
            ids[c] = len(ids)
        arr[i] = ids[c]
    chg = np.flatnonzero(np.diff(arr) != 0) + 1
    t = np.arange(T) / fps
    fig, ax = plt.subplots(figsize=(12, 1.6))
    ax.imshow(arr[None, :], aspect="auto", cmap="tab20",
              extent=[0, T / fps, 0, 1])
    for c in chg:
        ax.axvline(c / fps, color="k", lw=0.8)
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    ax.set_title(f"{name} — BABEL act_cat segments (color) + "
                 f"change-points (lines) = §3 boundary truth", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_timeline.png", dpi=120)
    plt.close(fig)

    # gif: ~80 frames, side view
    step = max(1, T // 80)
    idx = list(range(0, T, step))
    lo = P[:, :, :2].min(0).min(0)
    hi = P[:, :, :2].max(0).max(0)
    fig, ax = plt.subplots(figsize=(4, 4.5))

    def upd(f):
        draw(ax, P[f])
        ax.set_xlim(lo[0] - .2, hi[0] + .2)
        ax.set_ylim(lo[1] - .2, hi[1] + .2)
        ax.set_title(f"{name}  {f/fps:4.1f}s\n{cats[f] or '-'}", fontsize=8)

    an = FuncAnimation(fig, upd, frames=idx, interval=60)
    an.save(OUT / f"{name}.gif", writer=PillowWriter(fps=15))
    plt.close(fig)
    return name


def main():
    man = json.loads((SEQ / "manifest.json").read_text())
    overview(man)

    # pick diverse, multi-action samples (>=6 BABEL segments), 1 per subset
    picked, seen = [], set()
    for m in sorted(man, key=lambda m: -m["n_frame_labels"]):
        if m["subset"] in seen or m["n_frame_labels"] < 6:
            continue
        if m["T"] < 200:
            continue
        seen.add(m["subset"])
        picked.append(m)
        if len(picked) >= 5:
            break

    print(f"overview -> {OUT/'overview.png'}")
    for m in picked:
        n = peek(m)
        print(f"  {n:32s} {m['subset']:18s} {m['T']}f "
              f"{m['dur_s']}s  segs={m['n_frame_labels']}")
    print(f"\nfilmstrip/timeline/gif -> {OUT}")


if __name__ == "__main__":
    main()
