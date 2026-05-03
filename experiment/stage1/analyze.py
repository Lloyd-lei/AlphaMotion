"""
Post-hoc analysis of Stage 1 results.

Reads runs/baseline/history.json and runs/motionformer/history.json, then
produces:
    - runs/curves.png      — train/val loss + subspace_sim + frob_align over epochs
    - runs/pair_vis.png    — [J, J] ground-truth synergy affinity vs each model's
                              learned pair structure

Run after run_experiment.sh has finished.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import SynergyConfig, SyntheticSynergyDataset
from baseline import BaselineConfig, BaselineTransformer
from motionformer import MotionFormerConfig, MotionFormer
from eval import ground_truth_affinity, evaluate_model


RUNS = Path(__file__).parent / "runs"


def load_history(name: str):
    return json.load(open(RUNS / name / "history.json"))


def load_model(name: str, device: str = "cuda"):
    state = torch.load(RUNS / name / "final.pt", map_location=device)
    if name == "baseline":
        cfg = BaselineConfig(T=64, J=20, C=3, hidden=128, depth=12, heads=4, ffn_mult=4)
        model = BaselineTransformer(cfg).to(device)
    else:
        cfg = MotionFormerConfig(T=64, J=20, C=3, hidden=128, pair_hidden=64, depth=6, heads=4)
        model = MotionFormer(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def plot_curves():
    h_b = load_history("baseline")
    h_m = load_history("motionformer")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    ax = axes[0, 0]
    ax.plot([e["epoch"] for e in h_b], [e["train_loss"] for e in h_b], label="baseline train")
    ax.plot([e["epoch"] for e in h_b], [e["val_loss"] for e in h_b], label="baseline val", linestyle="--")
    ax.plot([e["epoch"] for e in h_m], [e["train_loss"] for e in h_m], label="motionformer train")
    ax.plot([e["epoch"] for e in h_m], [e["val_loss"] for e in h_m], label="motionformer val", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Masked motion modeling loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (masked positions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Subspace similarity
    ax = axes[0, 1]
    eps_b = [e for e in h_b if "subspace_sim" in e]
    eps_m = [e for e in h_m if "subspace_sim" in e]
    ax.plot([e["epoch"] for e in eps_b], [e["subspace_sim"] for e in eps_b], "o-", label="baseline")
    ax.plot([e["epoch"] for e in eps_m], [e["subspace_sim"] for e in eps_m], "s-", label="motionformer")
    ax.axhline(0.4, linestyle=":", color="gray", label="chance (K/J)")
    ax.set_title("Top-K subspace similarity to ground-truth synergies")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean squared canonical corr.")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Frobenius alignment
    ax = axes[1, 0]
    ax.plot([e["epoch"] for e in eps_b], [e["frobenius_align"] for e in eps_b], "o-", label="baseline")
    ax.plot([e["epoch"] for e in eps_m], [e["frobenius_align"] for e in eps_m], "s-", label="motionformer")
    ax.set_title("Frobenius alignment of pair structure to ground truth")
    ax.set_xlabel("epoch"); ax.set_ylabel("⟨S_gt, S_model⟩ / ||·||")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Spectrum correlation
    ax = axes[1, 1]
    ax.plot([e["epoch"] for e in eps_b], [e["spectrum_corr"] for e in eps_b], "o-", label="baseline")
    ax.plot([e["epoch"] for e in eps_m], [e["spectrum_corr"] for e in eps_m], "s-", label="motionformer")
    ax.set_title("Eigen-spectrum correlation")
    ax.set_xlabel("epoch"); ax.set_ylabel("cosine of top-K eigenvalue spectra")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RUNS / "curves.png", dpi=110)
    plt.close(fig)
    print(f"Saved {RUNS / 'curves.png'}")


def plot_pair_structures():
    cfg = SynergyConfig()
    ds = SyntheticSynergyDataset(cfg)
    S_gt = ground_truth_affinity(ds.temporal, ds.spatial)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_b = load_model("baseline", device=device)
    model_m = load_model("motionformer", device=device)

    # One forward pass to populate caches
    batch = torch.stack([ds[i] for i in range(4)]).to(device)
    zero_mask = torch.zeros(batch.shape[:3], dtype=torch.bool, device=device)
    _ = model_b(batch, zero_mask)
    _ = model_m(batch, zero_mask)

    S_b = model_b.extract_pair_structure().detach().cpu().numpy()
    S_b /= np.linalg.norm(S_b) + 1e-8
    S_m = model_m.extract_pair_structure().detach().cpu().numpy()
    S_m /= np.linalg.norm(S_m) + 1e-8

    S_gt_n = S_gt / (np.linalg.norm(S_gt) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    im = axes[0].imshow(S_gt_n, cmap="RdBu_r", vmin=-np.max(np.abs(S_gt_n)), vmax=np.max(np.abs(S_gt_n)))
    axes[0].set_title("Ground truth synergy affinity")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    vmax = max(np.max(np.abs(S_b)), np.max(np.abs(S_m)))
    im = axes[1].imshow(S_b, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Baseline: joint-PE correlation")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    im = axes[2].imshow(S_m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("MotionFormer: pair-tensor norm")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("joint j"); ax.set_ylabel("joint i")

    fig.tight_layout()
    fig.savefig(RUNS / "pair_vis.png", dpi=110)
    plt.close(fig)
    print(f"Saved {RUNS / 'pair_vis.png'}")


def print_summary():
    h_b = load_history("baseline")
    h_m = load_history("motionformer")
    b_final = [e for e in h_b if "subspace_sim" in e][-1]
    m_final = [e for e in h_m if "subspace_sim" in e][-1]
    print()
    print("=" * 60)
    print(f"{'metric':<25} {'baseline':>12} {'motionformer':>14}  delta")
    print("-" * 60)
    for key in ("val_loss", "subspace_sim", "frobenius_align", "spectrum_corr"):
        b = b_final[key]
        m = m_final[key]
        d = m - b
        arrow = "↑" if d > 0 else "↓"
        print(f"{key:<25} {b:>12.4f} {m:>14.4f}  {arrow} {d:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    plot_curves()
    plot_pair_structures()
    print_summary()
