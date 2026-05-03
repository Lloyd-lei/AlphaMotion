"""Render a single sample at high resolution with GT vs prediction overlaid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from soma_skeleton import build_parent_index, JOINT_NAMES


def plot_skel(ax, joints, parents, color, alpha=1.0, label=None):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=25, c=color, alpha=alpha, label=label)
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([joints[j, 0], joints[p, 0]],
                    [joints[j, 1], joints[p, 1]],
                    [joints[j, 2], joints[p, 2]],
                    c=color, alpha=alpha * 0.7, linewidth=1.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="Path to a sample*.npz from visualize_reconstruction.py")
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="Frame indices to render (default: 6 equally spaced).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    gt = data["gt"]            # [T, J, 3]
    pred = data["composite"]   # [T, J, 3]  (gt where unmasked, pred where masked)
    mask = data["mask"]        # [T, J]
    prompt = str(data["prompt"])

    T, J, _ = gt.shape
    if args.frames is None:
        args.frames = list(np.linspace(0, T - 1, 6).astype(int))

    parents = build_parent_index()

    fig = plt.figure(figsize=(18, 12))
    all_coords = np.concatenate([gt, pred], axis=0).reshape(-1, 3)
    ctr = 0.5 * (all_coords.min(axis=0) + all_coords.max(axis=0))
    span = (all_coords.max(axis=0) - all_coords.min(axis=0)).max() * 0.55 + 0.1

    for col, frame in enumerate(args.frames):
        ax = fig.add_subplot(2, len(args.frames), col + 1, projection="3d")
        plot_skel(ax, gt[frame], parents, "tab:green", 0.9, "ground truth" if col == 0 else None)
        ax.set_title(f"GT  t={frame}", fontsize=11)
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)
        ax.view_init(elev=15, azim=-70)
        ax.set_box_aspect((1, 1, 1))

        ax2 = fig.add_subplot(2, len(args.frames), len(args.frames) + col + 1, projection="3d")
        plot_skel(ax2, pred[frame], parents, "tab:blue", 0.9, "reconstruction" if col == 0 else None)
        # Mark masked joints
        m = mask[frame]
        ax2.scatter(pred[frame, m, 0], pred[frame, m, 1], pred[frame, m, 2],
                    s=50, c="red", alpha=0.7, marker="o", facecolor="none", linewidth=2,
                    label="was masked" if col == 0 else None)
        ax2.set_title(f"Pred  t={frame}  (mask={m.sum()})", fontsize=11)
        ax2.set_xlim(ctr[0] - span, ctr[0] + span)
        ax2.set_ylim(ctr[1] - span, ctr[1] + span)
        ax2.set_zlim(ctr[2] - span, ctr[2] + span)
        ax2.view_init(elev=15, azim=-70)
        ax2.set_box_aspect((1, 1, 1))

    fig.suptitle(f"{prompt[:90]}\n(red circles = joints that were masked and filled in by the model)",
                 fontsize=12)
    fig.tight_layout()
    out = args.out or str(Path(args.npz).with_suffix("")) + "_zoom.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
