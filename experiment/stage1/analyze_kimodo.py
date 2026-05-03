"""
Post-hoc analysis for Stage 1 on Kimodo data.

Reads runs/baseline/history.json and runs/motionformer/history.json, then
produces:
    runs/curves.png            — train/val/val_kc loss curves, sample-efficiency annotations
    runs/sample_efficiency.png — bar chart of steps-to-reach-threshold for each model

No ground-truth synergy recovery here (Kimodo data has no known synergies).
This is purely a performance comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(__file__).parent / "runs"


def load(name: str) -> dict:
    return json.load(open(RUNS / name / "history.json"))


def plot_curves():
    b = load("baseline")
    m = load("motionformer")
    h_b = b["history"]
    h_m = m["history"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: train + val loss
    ax = axes[0]
    ax.plot([e["step"] for e in h_b], [e["train_loss"] for e in h_b],
            color="tab:blue", alpha=0.4, label="baseline train")
    ax.plot([e["step"] for e in h_b], [e["val_loss"] for e in h_b],
            color="tab:blue", linewidth=2, label="baseline val")
    ax.plot([e["step"] for e in h_m], [e["train_loss"] for e in h_m],
            color="tab:orange", alpha=0.4, label="motionformer train")
    ax.plot([e["step"] for e in h_m], [e["val_loss"] for e in h_m],
            color="tab:orange", linewidth=2, label="motionformer val")
    for thresh in (0.5, 0.3, 0.2):
        ax.axhline(thresh, linestyle=":", alpha=0.3, color="gray")
    ax.set_xlabel("training step")
    ax.set_ylabel("masked MSE (mixed mask)")
    ax.set_title("Train / val loss over steps")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: val_kc (kinematic-chain failure OOD)
    ax = axes[1]
    ax.plot([e["step"] for e in h_b], [e["val_kc_loss"] for e in h_b],
            color="tab:blue", linewidth=2, label="baseline val_kc")
    ax.plot([e["step"] for e in h_m], [e["val_kc_loss"] for e in h_m],
            color="tab:orange", linewidth=2, label="motionformer val_kc")
    ax.set_xlabel("training step")
    ax.set_ylabel("masked MSE (joint-failure OOD)")
    ax.set_title("Kinematic-chain mask generalisation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: sample efficiency bars
    ax = axes[2]
    thresholds = [1.0, 0.5, 0.3, 0.2, 0.15]
    cap = max(e["step"] for e in h_b + h_m) + 10  # "never" = past end
    def get_steps(runs: dict, t: float):
        v = runs["target_steps"].get(str(t))
        if v is None:
            return cap
        return v
    b_steps = [get_steps(b, t) for t in thresholds]
    m_steps = [get_steps(m, t) for t in thresholds]
    x = np.arange(len(thresholds))
    width = 0.35
    bars_b = ax.bar(x - width / 2, b_steps, width, label="baseline", color="tab:blue")
    bars_m = ax.bar(x + width / 2, m_steps, width, label="motionformer", color="tab:orange")
    for bar, steps in list(zip(bars_b, b_steps)) + list(zip(bars_m, m_steps)):
        label = "never" if steps >= cap else f"{steps}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"≤{t}" for t in thresholds])
    ax.set_xlabel("val_loss threshold")
    ax.set_ylabel("first step to reach (lower=better)")
    ax.set_title("Sample efficiency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RUNS / "curves.png", dpi=110)
    plt.close(fig)
    print(f"Saved {RUNS / 'curves.png'}")


def plot_pair_vis():
    """Visualise MotionFormer's learned pair structure (no ground truth to compare)."""
    import torch
    from data import KimodoMotionDataset
    from motionformer import MotionFormerConfig, MotionFormer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(RUNS / "motionformer" / "final.pt", map_location=device)
    ds = KimodoMotionDataset(RUNS / "kimodo_cache.npz", crop_frames=90)
    cfg = MotionFormerConfig(T=ds.T, J=ds.J, C=ds.C, hidden=128, pair_hidden=32,
                             depth=6, heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16)
    model = MotionFormer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch = torch.stack([ds[i] for i in range(4)]).to(device)
    zero_mask = torch.zeros(batch.shape[:3], dtype=torch.bool, device=device)
    with torch.no_grad():
        _ = model(batch, zero_mask)
    pair = model.extract_pair_structure().detach().cpu().numpy()

    from soma_skeleton import JOINT_NAMES
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(pair, cmap="viridis")
    ax.set_title("MotionFormer learned pair structure (L2 norm of pair tensor)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # Annotate every 8th joint to avoid crowding
    ticks = list(range(0, len(JOINT_NAMES), 8))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([JOINT_NAMES[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([JOINT_NAMES[i] for i in ticks], fontsize=7)
    fig.tight_layout()
    fig.savefig(RUNS / "pair_structure.png", dpi=110)
    plt.close(fig)
    print(f"Saved {RUNS / 'pair_structure.png'}")


def print_summary():
    b = load("baseline")
    m = load("motionformer")
    b_final = b["history"][-1]
    m_final = m["history"][-1]

    print()
    print("=" * 66)
    print(f"{'metric':<22} {'baseline':>14} {'motionformer':>16}  delta")
    print("-" * 66)
    for key in ("train_loss", "val_loss", "val_kc_loss"):
        bv = b_final[key]
        mv = m_final[key]
        d = mv - bv
        arrow = "↑" if d > 0 else "↓"
        print(f"{key:<22} {bv:>14.4f} {mv:>16.4f}  {arrow} {d:+.4f}")
    print("-" * 66)
    thresholds = [1.0, 0.5, 0.3, 0.2, 0.15]
    for t in thresholds:
        bs = b["target_steps"].get(str(t))
        ms = m["target_steps"].get(str(t))
        bs_str = str(bs) if bs is not None else "never"
        ms_str = str(ms) if ms is not None else "never"
        if bs is not None and ms is not None:
            ratio = f"{bs / ms:.1f}×"
        else:
            ratio = "—"
        print(f"val_loss ≤ {t:<10} {bs_str:>14} {ms_str:>16}  speedup: {ratio}")
    print("=" * 66)
    print(f"n_params  baseline: {b['n_params']/1e6:.2f}M   motionformer: {m['n_params']/1e6:.2f}M")


if __name__ == "__main__":
    plot_curves()
    plot_pair_vis()
    print_summary()
