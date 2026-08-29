"""AlphaMotion Foundation Evaluation — 动作界的 DINO 需要证明什么？

Six label-free evaluation dimensions that test whether AlphaMotion's trunk
(specifically triangle attention) learns universal temporal-dynamic representations.

Usage:
    python notebook/eval_temporal_foundation.py                 # all 6 variants, all dims
    python notebook/eval_temporal_foundation.py --variants full opm_only  # subset
    python notebook/eval_temporal_foundation.py --dims 1 2 4    # specific dimensions
    python notebook/eval_temporal_foundation.py --step 30000    # checkpoint step

Dimensions:
    1  Multi-noise Denoising Curve
    2  Joint Velocity Covariance Alignment
    3  Temporal Autocorrelation of Pair Tensor
    4  Future Frame Prediction (Inpainting)
    5  Frequency Spectrum Fidelity
    6  Inpainting Robustness (joint masking)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
NOTEBOOK = Path(__file__).resolve().parent
REPO = NOTEBOOK.parent
SCRIPTS_02 = NOTEBOOK / "02_resources" / "scripts"
SCRIPTS_01 = NOTEBOOK / "01_resources" / "scripts"

for p in [str(SCRIPTS_02), str(SCRIPTS_01)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from _eval_common import (
    load_eval_model, eval_loader, fm_forward_at_t,
    fm_one_step_denoise, fm_sample_conditional_inpaint,
    DATA, VARS, ENV,
)
from data_loader import load_kimodo_ood

RESULTS_DIR = NOTEBOOK / "eval_temporal_results"
CKPT_ROOT = NOTEBOOK / "02_resources" / "checkpoints"
SEED = ENV["seed"]
T_FRAMES = DATA["sample"]["T"]  # 90
N_JOINTS = DATA["skeleton"]["n_joints"]  # 77


# ===========================================================================
# Helpers
# ===========================================================================

def _find_checkpoint(variant: str, step: int) -> Path:
    p = CKPT_ROOT / variant / f"step_{step}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _random_subset_indices(subset, n_clips: int, seed: int):
    """Return a new Subset with randomly sampled indices."""
    from torch.utils.data import Subset as TSubset
    rng = np.random.default_rng(seed)
    src = np.asarray(subset.indices)
    n = min(n_clips, len(src))
    picked = rng.choice(src, size=n, replace=False)
    return TSubset(subset.dataset, picked.tolist())


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Dim 1: Multi-noise Denoising Curve
# ===========================================================================

@torch.no_grad()
def eval_denoise_curve(
    ckpt_path: Path, subset, *,
    t_values: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 0.95),
    n_clips: int = 200, batch_size: int = 32,
) -> Dict[float, float]:
    """MPJPE at multiple noise levels. Harder t reveals temporal understanding."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset_indices(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    results = {}
    for t_val in t_values:
        all_err = []
        for batch in loader:
            x_pos_1 = batch["pos"].to(dev)
            x_rot_1 = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)

            B = x_pos_1.shape[0]
            t = torch.full((B,), t_val, device=dev)
            t_expand = t.view(B, 1, 1, 1)

            noise = torch.randn_like(x_pos_1)
            x_pos_t = (1 - t_expand) * noise + t_expand * x_pos_1

            x_rot_noise = torch.randn_like(x_rot_1)
            x_root_noise = torch.randn_like(x_root_1)
            x_rot_t = (1 - t.view(B, 1, 1, 1)) * x_rot_noise + t.view(B, 1, 1, 1) * x_rot_1
            x_root_t = (1 - t.view(B, 1, 1)) * x_root_noise + t.view(B, 1, 1) * x_root_1

            out = model(x_pos_t, x_rot_t, x_root_t, t)
            x_pos_hat = x_pos_t + (1 - t_expand) * out["pos"]

            err = (x_pos_hat - x_pos_1).norm(dim=-1).mean(dim=(1, 2))  # [B]
            all_err.append(err.cpu())

        mpjpe = torch.cat(all_err).mean().item()
        results[t_val] = mpjpe
    return results


# ===========================================================================
# Dim 2: Joint Velocity Covariance Alignment
# ===========================================================================

@torch.no_grad()
def eval_velocity_covariance_alignment(
    ckpt_path: Path, subset, *,
    n_clips: int = 200, batch_size: int = 32,
) -> Dict[str, float]:
    """Pearson correlation between pair tensor and GT velocity covariance."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    has_pair = getattr(cfg, "use_pair", False)

    if not has_pair:
        return {"pearson_r": float("nan"), "has_pair": False}

    sub = _random_subset_indices(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    all_vel_cov = []
    all_pair = []

    for batch in loader:
        x_pos_1 = batch["pos"].to(dev)    # [B, T, J, 3]
        x_rot_1 = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)

        # GT velocity: diff along time axis
        vel = x_pos_1[:, 1:, :, :] - x_pos_1[:, :-1, :, :]  # [B, T-1, J, 3]
        # Flatten xyz for covariance: [B, T-1, J*3]
        B, Tm1, J, _ = vel.shape
        vel_flat = vel.reshape(B, Tm1, J * 3)
        # Per-clip velocity covariance over joints: corr between joint i and j
        # We compute correlation of velocity magnitudes: [B, J]
        vel_mag = vel.norm(dim=-1)  # [B, T-1, J]
        # Pairwise correlation: [B, J, J]
        vel_centered = vel_mag - vel_mag.mean(dim=1, keepdim=True)
        vel_std = vel_centered.std(dim=1, keepdim=True).clamp(min=1e-8)
        vel_norm = vel_centered / vel_std  # [B, T-1, J]
        cov_matrix = torch.bmm(vel_norm.transpose(1, 2), vel_norm) / Tm1  # [B, J, J]
        all_vel_cov.append(cov_matrix.cpu())

        # Forward to get pair tensor
        _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=1.0)
        pair = model._last_pair  # [B, J, J, H_pair]
        pair_aff = pair.norm(dim=-1)  # [B, J, J]
        pair_sym = 0.5 * (pair_aff + pair_aff.transpose(1, 2))
        all_pair.append(pair_sym.cpu())

    vel_cov = torch.cat(all_vel_cov, dim=0).mean(dim=0).numpy()  # [J, J]
    pair_mean = torch.cat(all_pair, dim=0).mean(dim=0).numpy()    # [J, J]

    # Flatten upper triangle (exclude diagonal)
    mask = np.triu(np.ones((N_JOINTS, N_JOINTS), dtype=bool), k=1)
    v_flat = vel_cov[mask]
    p_flat = pair_mean[mask]

    # Pearson correlation
    if v_flat.std() < 1e-10 or p_flat.std() < 1e-10:
        r = 0.0
    else:
        r = float(np.corrcoef(v_flat, p_flat)[0, 1])

    return {"pearson_r": r, "has_pair": True}


# ===========================================================================
# Dim 3: Temporal Autocorrelation of Pair Tensor
# ===========================================================================

@torch.no_grad()
def eval_temporal_autocorrelation(
    ckpt_path: Path, subset, *,
    n_clips: int = 100, batch_size: int = 16,
) -> Dict[str, float]:
    """Measure how smoothly the pair tensor evolves across consecutive clips.

    Since each clip is T=90 frames and pair tensor is static per-clip (no time dim),
    we compare pair tensors across different clips to measure representation stability.
    For variants with meaningful pair, consecutive clips from the same sequence should
    produce similar pair tensors.
    """
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    has_pair = getattr(cfg, "use_pair", False)

    if not has_pair:
        return {"mean_autocorr": float("nan"), "std_autocorr": float("nan"),
                "pair_variance": float("nan"), "has_pair": False}

    # Use sequential clips (not shuffled) to measure temporal consistency
    from torch.utils.data import Subset as TSubset
    src_indices = np.asarray(subset.indices)
    n = min(n_clips, len(src_indices))
    # Take consecutive indices to maximize chance of temporal adjacency
    start = 0
    seq_indices = src_indices[start:start + n].tolist()
    sub = TSubset(subset.dataset, seq_indices)
    loader = eval_loader(sub, batch=batch_size)

    pair_sequence = []
    for batch in loader:
        x_pos_1 = batch["pos"].to(dev)
        x_rot_1 = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)

        _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=1.0)
        pair = model._last_pair  # [B, J, J, H_pair]
        pair_flat = pair.reshape(pair.shape[0], -1).cpu()  # [B, J*J*H]
        pair_sequence.append(pair_flat)

    pairs = torch.cat(pair_sequence, dim=0)  # [N, D]
    N = pairs.shape[0]

    # Cosine similarity between consecutive pairs
    norms = pairs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pairs_normed = pairs / norms

    if N < 2:
        return {"mean_autocorr": float("nan"), "std_autocorr": float("nan"),
                "pair_variance": float("nan"), "has_pair": True}

    cosines = (pairs_normed[:-1] * pairs_normed[1:]).sum(dim=-1)  # [N-1]

    # Pair variance: how much does pair differ across clips
    pair_var = pairs.var(dim=0).sum().item()
    log_var = float(np.log10(max(pair_var, 1e-30)))

    return {
        "mean_autocorr": float(cosines.mean()),
        "std_autocorr": float(cosines.std()),
        "pair_variance_log10": log_var,
        "has_pair": True,
    }


# ===========================================================================
# Dim 4: Future Frame Prediction
# ===========================================================================

@torch.no_grad()
def eval_future_prediction(
    ckpt_path: Path, subset, *,
    horizons: Tuple[int, ...] = (15, 30, 45),
    n_clips: int = 100, batch_size: int = 16, n_steps: int = 16,
) -> Dict[int, float]:
    """Mask last N frames and use inpainting to predict them. Report MPJPE."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset_indices(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    results = {}
    for horizon in horizons:
        all_err = []
        for batch in loader:
            x_pos_1 = batch["pos"].to(dev)      # [B, T, J, 3]
            x_rot_1 = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            B, T, J, _ = x_pos_1.shape

            # Mask: last `horizon` frames, all joints
            mask = torch.zeros(B, T, J, dtype=torch.bool, device=dev)
            mask[:, -horizon:, :] = True

            # Inpainting
            x_pos_pred, _, _ = fm_sample_conditional_inpaint(
                model, x_pos_1, x_rot_1, x_root_1, mask, n_steps=n_steps,
            )

            # MPJPE on masked region only
            masked_pred = x_pos_pred[:, -horizon:, :, :]
            masked_gt = x_pos_1[:, -horizon:, :, :]
            err = (masked_pred - masked_gt).norm(dim=-1).mean(dim=(1, 2))  # [B]
            all_err.append(err.cpu())

        mpjpe = torch.cat(all_err).mean().item()
        results[horizon] = mpjpe
    return results


# ===========================================================================
# Dim 5: Frequency Spectrum Fidelity
# ===========================================================================

@torch.no_grad()
def eval_frequency_fidelity(
    ckpt_path: Path, subset, *,
    t_denoise: float = 0.7, n_clips: int = 200, batch_size: int = 32,
) -> Dict[str, float]:
    """Compare FFT power spectrum of GT vs reconstructed motion.

    Reports: total spectral distance, low-freq distance, high-freq distance.
    """
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset_indices(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    all_gt_spectrum = []
    all_pred_spectrum = []

    for batch in loader:
        x_pos_1 = batch["pos"].to(dev)    # [B, T, J, 3]
        x_rot_1 = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)
        B, T, J, C = x_pos_1.shape

        # 1-step denoise from t_denoise
        t = torch.full((B,), t_denoise, device=dev)
        t_expand = t.view(B, 1, 1, 1)
        noise = torch.randn_like(x_pos_1)
        x_pos_t = (1 - t_expand) * noise + t_expand * x_pos_1
        x_rot_noise = torch.randn_like(x_rot_1)
        x_root_noise = torch.randn_like(x_root_1)
        x_rot_t = (1 - t.view(B, 1, 1, 1)) * x_rot_noise + t.view(B, 1, 1, 1) * x_rot_1
        x_root_t = (1 - t.view(B, 1, 1)) * x_root_noise + t.view(B, 1, 1) * x_root_1

        out = model(x_pos_t, x_rot_t, x_root_t, t)
        x_pos_hat = x_pos_t + (1 - t_expand) * out["pos"]

        # FFT along time axis: [B, T, J, 3] -> magnitude
        # Reshape to [B, T, J*3] for per-joint-axis spectrum
        gt_flat = x_pos_1.reshape(B, T, -1)         # [B, T, J*3]
        pred_flat = x_pos_hat.reshape(B, T, -1)

        gt_fft = torch.fft.rfft(gt_flat, dim=1).abs()      # [B, T//2+1, J*3]
        pred_fft = torch.fft.rfft(pred_flat, dim=1).abs()

        # Log power spectrum (avoid log(0))
        gt_log = torch.log10(gt_fft.clamp(min=1e-10))
        pred_log = torch.log10(pred_fft.clamp(min=1e-10))

        all_gt_spectrum.append(gt_log.cpu())
        all_pred_spectrum.append(pred_log.cpu())

    gt_spec = torch.cat(all_gt_spectrum, dim=0)    # [N, F, J*3]
    pred_spec = torch.cat(all_pred_spectrum, dim=0)

    # Spectral distance per frequency bin
    spec_dist = (gt_spec - pred_spec).pow(2).mean(dim=(0, 2))  # [F]
    n_freq = spec_dist.shape[0]
    mid = n_freq // 2

    total_dist = float(spec_dist.mean())
    low_freq_dist = float(spec_dist[:mid].mean())
    high_freq_dist = float(spec_dist[mid:].mean())

    return {
        "spectral_dist_total": total_dist,
        "spectral_dist_low_freq": low_freq_dist,
        "spectral_dist_high_freq": high_freq_dist,
        "n_freq_bins": n_freq,
        "t_denoise": t_denoise,
    }


# ===========================================================================
# Dim 6: Inpainting Robustness (Joint Masking)
# ===========================================================================

@torch.no_grad()
def eval_inpainting_robustness(
    ckpt_path: Path, subset, *,
    mask_ratios: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    n_clips: int = 100, batch_size: int = 16, n_steps: int = 16,
) -> Dict[float, float]:
    """Random joint masking at various ratios. Report MPJPE on masked joints."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset_indices(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    rng = np.random.default_rng(SEED)
    results = {}

    for ratio in mask_ratios:
        all_err = []
        for batch in loader:
            x_pos_1 = batch["pos"].to(dev)
            x_rot_1 = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            B, T, J, _ = x_pos_1.shape

            # Random joint mask: same joints masked across all frames per clip
            n_masked = max(1, int(J * ratio))
            mask = torch.zeros(B, T, J, dtype=torch.bool, device=dev)
            for b in range(B):
                masked_joints = rng.choice(J, size=n_masked, replace=False)
                mask[b, :, masked_joints] = True

            x_pos_pred, _, _ = fm_sample_conditional_inpaint(
                model, x_pos_1, x_rot_1, x_root_1, mask, n_steps=n_steps,
            )

            # MPJPE only on masked positions
            masked_pred = x_pos_pred[mask.unsqueeze(-1).expand_as(x_pos_pred)]
            masked_gt = x_pos_1[mask.unsqueeze(-1).expand_as(x_pos_1)]
            masked_pred = masked_pred.reshape(-1, 3)
            masked_gt = masked_gt.reshape(-1, 3)
            err = (masked_pred - masked_gt).norm(dim=-1).mean()
            all_err.append(err.cpu())

        mpjpe = torch.stack(all_err).mean().item()
        results[ratio] = mpjpe
    return results


# ===========================================================================
# Visualization
# ===========================================================================

def plot_denoise_curve(all_results: Dict[str, Dict[float, float]], save_path: Path):
    """Dim 1: MPJPE vs noise level, one line per variant."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    for i, (variant, data) in enumerate(sorted(all_results.items())):
        ts = sorted(data.keys())
        vals = [data[t] for t in ts]
        ax.plot(ts, vals, marker="o", color=cmap(i), label=variant, linewidth=2)
    ax.set_xlabel("Noise level t (lower = harder)")
    ax.set_ylabel("MPJPE (normalized)")
    ax.set_title("Dim 1: Multi-noise Denoising Curve (K-eval OOD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_velocity_alignment(all_results: Dict[str, Dict], save_path: Path):
    """Dim 2: Bar chart of Pearson r per variant."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    variants = sorted(all_results.keys())
    rs = [all_results[v]["pearson_r"] for v in variants]
    colors = ["#2ca02c" if all_results[v].get("has_pair") else "#aaa" for v in variants]
    bars = ax.bar(variants, rs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Pearson r (pair tensor vs velocity covariance)")
    ax.set_title("Dim 2: Joint Velocity Covariance Alignment")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, rs):
        if not np.isnan(r):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{r:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                    "N/A", ha="center", va="bottom", fontsize=9, color="#888")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_temporal_autocorr(all_results: Dict[str, Dict], save_path: Path):
    """Dim 3: Autocorrelation + pair variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    variants = sorted(all_results.keys())

    # Autocorrelation
    vals = [all_results[v]["mean_autocorr"] for v in variants]
    colors = ["#2ca02c" if all_results[v].get("has_pair") else "#aaa" for v in variants]
    ax1.bar(variants, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Mean cosine similarity (consecutive clips)")
    ax1.set_title("Pair Tensor Temporal Autocorrelation")
    ax1.grid(axis="y", alpha=0.3)

    # Pair variance
    vals2 = [all_results[v].get("pair_variance_log10", float("nan")) for v in variants]
    ax2.bar(variants, vals2, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("log10(pair variance)")
    ax2.set_title("Pair Tensor Diversity Across Clips")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_future_prediction(all_results: Dict[str, Dict[int, float]], save_path: Path):
    """Dim 4: MPJPE vs horizon, one line per variant."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    for i, (variant, data) in enumerate(sorted(all_results.items())):
        horizons = sorted(data.keys())
        vals = [data[h] for h in horizons]
        ax.plot(horizons, vals, marker="s", color=cmap(i), label=variant, linewidth=2)
    ax.set_xlabel("Prediction horizon (frames masked)")
    ax.set_ylabel("MPJPE (masked region)")
    ax.set_title("Dim 4: Future Frame Prediction (K-eval OOD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_frequency_fidelity(all_results: Dict[str, Dict], save_path: Path):
    """Dim 5: Grouped bar chart (low/high freq distance)."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    variants = sorted(all_results.keys())
    x = np.arange(len(variants))
    w = 0.35

    low = [all_results[v]["spectral_dist_low_freq"] for v in variants]
    high = [all_results[v]["spectral_dist_high_freq"] for v in variants]

    ax.bar(x - w / 2, low, w, label="Low freq", color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, high, w, label="High freq", color="#dd8452", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Spectral distance (log-power MSE)")
    ax.set_title("Dim 5: Frequency Spectrum Fidelity (t=0.7 denoise)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_inpainting_robustness(all_results: Dict[str, Dict[float, float]], save_path: Path):
    """Dim 6: MPJPE vs mask ratio, one line per variant."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    for i, (variant, data) in enumerate(sorted(all_results.items())):
        ratios = sorted(data.keys())
        vals = [data[r] for r in ratios]
        ax.plot([f"{int(r*100)}%" for r in ratios], vals,
                marker="D", color=cmap(i), label=variant, linewidth=2)
    ax.set_xlabel("Joint mask ratio")
    ax.set_ylabel("MPJPE (masked joints)")
    ax.set_title("Dim 6: Inpainting Robustness (K-eval OOD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Summary Table
# ===========================================================================

def print_summary_table(summary: Dict):
    """Print a formatted comparison table to terminal."""
    all_variant_sets = [set(v.keys()) for v in summary.values() if isinstance(v, dict)]
    variants = sorted(set().union(*all_variant_sets)) if all_variant_sets else []
    if not variants:
        print("[WARNING] No results to display.")
        return

    print("\n" + "=" * 80)
    print("  AlphaMotion Foundation Evaluation — Summary")
    print("=" * 80)

    # Dim 1
    if "dim1_denoise" in summary:
        print("\n--- Dim 1: Multi-noise Denoising MPJPE ---")
        t_vals = sorted(next(iter(summary["dim1_denoise"].values())).keys())
        header = f"{'variant':<14}" + "".join(f"{'t='+str(t):<10}" for t in t_vals)
        print(header)
        for v in variants:
            row = f"{v:<14}" + "".join(f"{summary['dim1_denoise'][v][t]:<10.4f}" for t in t_vals)
            print(row)

    # Dim 2
    if "dim2_vel_cov" in summary:
        print("\n--- Dim 2: Velocity Covariance Alignment (Pearson r) ---")
        for v in variants:
            r = summary["dim2_vel_cov"][v]["pearson_r"]
            tag = "" if summary["dim2_vel_cov"][v].get("has_pair") else " [no pair]"
            print(f"  {v:<14} r = {r:+.4f}{tag}")

    # Dim 3
    if "dim3_autocorr" in summary:
        print("\n--- Dim 3: Temporal Autocorrelation ---")
        for v in variants:
            d = summary["dim3_autocorr"][v]
            ac = d["mean_autocorr"]
            pv = d.get("pair_variance_log10", float("nan"))
            tag = "" if d.get("has_pair") else " [no pair]"
            print(f"  {v:<14} autocorr = {ac:.4f}  pair_var(log10) = {pv:.2f}{tag}")

    # Dim 4
    if "dim4_future" in summary:
        print("\n--- Dim 4: Future Frame Prediction MPJPE ---")
        horizons = sorted(next(iter(summary["dim4_future"].values())).keys())
        header = f"{'variant':<14}" + "".join(f"{'h='+str(h):<10}" for h in horizons)
        print(header)
        for v in variants:
            row = f"{v:<14}" + "".join(f"{summary['dim4_future'][v][h]:<10.4f}" for h in horizons)
            print(row)

    # Dim 5
    if "dim5_freq" in summary:
        print("\n--- Dim 5: Frequency Spectrum Fidelity ---")
        print(f"  {'variant':<14} {'total':<10} {'low_freq':<10} {'high_freq':<10}")
        for v in variants:
            d = summary["dim5_freq"][v]
            print(f"  {v:<14} {d['spectral_dist_total']:<10.4f} "
                  f"{d['spectral_dist_low_freq']:<10.4f} {d['spectral_dist_high_freq']:<10.4f}")

    # Dim 6
    if "dim6_inpaint" in summary:
        print("\n--- Dim 6: Inpainting Robustness MPJPE ---")
        ratios = sorted(next(iter(summary["dim6_inpaint"].values())).keys())
        header = f"{'variant':<14}" + "".join(f"{str(int(r*100))+'%':<10}" for r in ratios)
        print(header)
        for v in variants:
            row = f"{v:<14}" + "".join(f"{summary['dim6_inpaint'][v][r]:<10.4f}" for r in ratios)
            print(row)

    print("\n" + "=" * 80)


# ===========================================================================
# Main Runner
# ===========================================================================

def run_all(
    variants: Optional[List[str]] = None,
    dims: Optional[List[int]] = None,
    step: int = 30000,
) -> Dict:
    """Run all (or selected) evaluation dimensions on all (or selected) variants."""
    if variants is None:
        variants = list(VARS.keys())
    if dims is None:
        dims = [1, 2, 3, 4, 5, 6]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Loading K-eval (Kimodo OOD) dataset...")
    k_eval = load_kimodo_ood(DATA)
    print(f"[eval] K-eval: {len(k_eval)} clips")
    print(f"[eval] Variants: {variants}")
    print(f"[eval] Dimensions: {dims}")
    print(f"[eval] Checkpoint step: {step}")
    print()

    summary = {}

    # --- Dim 1 ---
    if 1 in dims:
        print("=" * 60)
        print("  Dim 1: Multi-noise Denoising Curve")
        print("=" * 60)
        summary["dim1_denoise"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = eval_denoise_curve(ckpt, k_eval)
            summary["dim1_denoise"][v] = result
            elapsed = time.time() - t0
            print(f"  [{v}] {elapsed:.1f}s  MPJPE: " +
                  " | ".join(f"t={t}:{m:.4f}" for t, m in sorted(result.items())))
        plot_denoise_curve(summary["dim1_denoise"], RESULTS_DIR / "dim1_denoise_curve.png")
        print()

    # --- Dim 2 ---
    if 2 in dims:
        print("=" * 60)
        print("  Dim 2: Joint Velocity Covariance Alignment")
        print("=" * 60)
        summary["dim2_vel_cov"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = eval_velocity_covariance_alignment(ckpt, k_eval)
            summary["dim2_vel_cov"][v] = result
            elapsed = time.time() - t0
            r = result["pearson_r"]
            tag = "" if result["has_pair"] else " [no pair tensor]"
            print(f"  [{v}] {elapsed:.1f}s  Pearson r = {r:+.4f}{tag}")
        plot_velocity_alignment(summary["dim2_vel_cov"], RESULTS_DIR / "dim2_velocity_alignment.png")
        print()

    # --- Dim 3 ---
    if 3 in dims:
        print("=" * 60)
        print("  Dim 3: Temporal Autocorrelation of Pair Tensor")
        print("=" * 60)
        summary["dim3_autocorr"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = eval_temporal_autocorrelation(ckpt, k_eval)
            summary["dim3_autocorr"][v] = result
            elapsed = time.time() - t0
            ac = result["mean_autocorr"]
            pv = result.get("pair_variance_log10", float("nan"))
            tag = "" if result["has_pair"] else " [no pair]"
            print(f"  [{v}] {elapsed:.1f}s  autocorr = {ac:.4f}  "
                  f"pair_var(log10) = {pv:.2f}{tag}")
        plot_temporal_autocorr(summary["dim3_autocorr"], RESULTS_DIR / "dim3_temporal_autocorr.png")
        print()

    # --- Dim 4 ---
    if 4 in dims:
        print("=" * 60)
        print("  Dim 4: Future Frame Prediction")
        print("=" * 60)
        summary["dim4_future"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = eval_future_prediction(ckpt, k_eval)
            summary["dim4_future"][v] = result
            elapsed = time.time() - t0
            print(f"  [{v}] {elapsed:.1f}s  MPJPE: " +
                  " | ".join(f"h={h}:{m:.4f}" for h, m in sorted(result.items())))
        plot_future_prediction(summary["dim4_future"], RESULTS_DIR / "dim4_future_prediction.png")
        print()

    # --- Dim 5 ---
    if 5 in dims:
        print("=" * 60)
        print("  Dim 5: Frequency Spectrum Fidelity")
        print("=" * 60)
        summary["dim5_freq"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = eval_frequency_fidelity(ckpt, k_eval)
            summary["dim5_freq"][v] = result
            elapsed = time.time() - t0
            print(f"  [{v}] {elapsed:.1f}s  total={result['spectral_dist_total']:.4f}  "
                  f"low={result['spectral_dist_low_freq']:.4f}  "
                  f"high={result['spectral_dist_high_freq']:.4f}")
        plot_frequency_fidelity(summary["dim5_freq"], RESULTS_DIR / "dim5_frequency_fidelity.png")
        print()

    # --- Dim 6 ---
    if 6 in dims:
        print("=" * 60)
        print("  Dim 6: Inpainting Robustness (Joint Masking)")
        print("=" * 60)
        summary["dim6_inpaint"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = eval_inpainting_robustness(ckpt, k_eval)
            summary["dim6_inpaint"][v] = result
            elapsed = time.time() - t0
            print(f"  [{v}] {elapsed:.1f}s  MPJPE: " +
                  " | ".join(f"{int(r*100)}%:{m:.4f}" for r, m in sorted(result.items())))
        plot_inpainting_robustness(summary["dim6_inpaint"], RESULTS_DIR / "dim6_inpainting_robustness.png")
        print()

    # Save summary
    def _serialize(obj):
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Results saved to {RESULTS_DIR}/")
    print(f"[eval] Summary JSON: {summary_path}")

    print_summary_table(summary)
    return summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AlphaMotion Foundation Evaluation — 6 label-free temporal dimensions"
    )
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variants to evaluate (default: all 6)")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="Which dimensions to run (1-6, default: all)")
    parser.add_argument("--step", type=int, default=30000,
                        help="Checkpoint step to evaluate (default: 30000)")
    args = parser.parse_args()

    if args.variants:
        for v in args.variants:
            if v not in VARS:
                parser.error(f"Unknown variant '{v}'. Available: {list(VARS.keys())}")

    run_all(variants=args.variants, dims=args.dims, step=args.step)


if __name__ == "__main__":
    main()
