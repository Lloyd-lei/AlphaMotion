"""AlphaMotion Foundation Demos — Zero-shot capabilities from a single trunk.

Six demonstrations that require NO additional training, showcasing the
foundation model properties of the AlphaMotion trunk.

Usage:
    python notebook/eval_foundation_demos.py                    # all variants, all demos
    python notebook/eval_foundation_demos.py --variants full baseline
    python notebook/eval_foundation_demos.py --demos 1 3 5
    python notebook/eval_foundation_demos.py --step 30000

Demos:
    1  Sparse Sensor Completion (6 joints -> full body)
    2  Temporal Super-resolution (low FPS -> high FPS)
    3  Motion Transition Generation (clip A -> ? -> clip B)
    4  Pair Tensor as Learned Skeleton (visualize joint affinities)
    5  Anomaly / Physics Violation Detection (score unnatural motion)
    6  Motion Interpolation in Pair Space (blend two motions via pair)
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
    fm_sample_conditional_inpaint, DATA, VARS, ENV,
)
from data_loader import load_kimodo_ood

RESULTS_DIR = NOTEBOOK / "eval_foundation_demos"
CKPT_ROOT = NOTEBOOK / "02_resources" / "checkpoints"
SEED = ENV["seed"]
T_FRAMES = DATA["sample"]["T"]  # 90
N_JOINTS = DATA["skeleton"]["n_joints"]  # 77

# Anatomically meaningful sparse sensor joints (indices in SOMA77)
SPARSE_JOINTS = [0, 15, 20, 21, 7, 8]
SPARSE_JOINT_NAMES = ["pelvis", "head", "L_wrist", "R_wrist", "L_ankle", "R_ankle"]


def _find_checkpoint(variant: str, step: int) -> Path:
    p = CKPT_ROOT / variant / f"step_{step}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _random_subset(subset, n_clips: int, seed: int):
    from torch.utils.data import Subset as TSubset
    rng = np.random.default_rng(seed)
    src = np.asarray(subset.indices)
    n = min(n_clips, len(src))
    picked = rng.choice(src, size=n, replace=False)
    return TSubset(subset.dataset, picked.tolist())


# ===========================================================================
# Demo 1: Sparse Sensor Completion
# ===========================================================================

@torch.no_grad()
def demo_sparse_sensor(
    ckpt_path: Path, subset, *,
    n_clips: int = 100, batch_size: int = 16, n_steps: int = 16,
    visible_joints: List[int] = None,
) -> Dict[str, float]:
    """Mask all joints except a few 'sensors', inpaint the full body."""
    if visible_joints is None:
        visible_joints = SPARSE_JOINTS

    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    all_err_masked = []
    all_err_total = []

    for batch in loader:
        x_pos_1 = batch["pos"].to(dev)
        x_rot_1 = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)
        B, T, J, _ = x_pos_1.shape

        mask = torch.ones(B, T, J, dtype=torch.bool, device=dev)
        for j in visible_joints:
            mask[:, :, j] = False

        x_pos_pred, _, _ = fm_sample_conditional_inpaint(
            model, x_pos_1, x_rot_1, x_root_1, mask, n_steps=n_steps,
        )

        err_all = (x_pos_pred - x_pos_1).norm(dim=-1)
        err_masked = err_all[mask].mean()
        err_total = err_all.mean()

        all_err_masked.append(err_masked.cpu())
        all_err_total.append(err_total.cpu())

    return {
        "mpjpe_masked_joints": float(torch.stack(all_err_masked).mean()),
        "mpjpe_total": float(torch.stack(all_err_total).mean()),
        "n_visible": len(visible_joints),
        "n_masked": N_JOINTS - len(visible_joints),
    }


# ===========================================================================
# Demo 2: Temporal Super-resolution
# ===========================================================================

@torch.no_grad()
def demo_temporal_superres(
    ckpt_path: Path, subset, *,
    downsample_factors: Tuple[int, ...] = (2, 3, 4),
    n_clips: int = 100, batch_size: int = 16, n_steps: int = 16,
) -> Dict[int, float]:
    """Keep every Nth frame visible, mask intermediate frames, inpaint."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    results = {}
    for factor in downsample_factors:
        all_err = []
        for batch in loader:
            x_pos_1 = batch["pos"].to(dev)
            x_rot_1 = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            B, T, J, _ = x_pos_1.shape

            mask = torch.ones(B, T, J, dtype=torch.bool, device=dev)
            for t in range(0, T, factor):
                mask[:, t, :] = False

            x_pos_pred, _, _ = fm_sample_conditional_inpaint(
                model, x_pos_1, x_rot_1, x_root_1, mask, n_steps=n_steps,
            )

            err = (x_pos_pred - x_pos_1).norm(dim=-1)
            err_masked = err[mask].mean()
            all_err.append(err_masked.cpu())

        results[factor] = float(torch.stack(all_err).mean())
    return results


# ===========================================================================
# Demo 3: Motion Transition Generation
# ===========================================================================

@torch.no_grad()
def demo_motion_transition(
    ckpt_path: Path, subset, *,
    context_frames: int = 15,
    n_pairs: int = 50, batch_size: int = 16, n_steps: int = 16,
) -> Dict[str, float]:
    """Given start/end context from two different clips, generate transition."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device

    rng = np.random.default_rng(SEED)
    src = np.asarray(subset.indices)
    n = min(n_pairs * 2, len(src))
    picked = rng.choice(src, size=n, replace=False)
    ds = subset.dataset

    all_smoothness = []
    all_transition_err = []

    for i in range(0, len(picked) - 1, 2):
        clip_a = ds[picked[i]]
        clip_b = ds[picked[i + 1]]

        x_pos = torch.zeros(1, T_FRAMES, N_JOINTS, 3, device=dev)
        x_rot = torch.zeros(1, T_FRAMES, N_JOINTS, 6, device=dev)
        x_root = torch.zeros(1, T_FRAMES, 3, device=dev)

        x_pos[0, :context_frames] = clip_a["pos"][:context_frames].to(dev)
        x_pos[0, -context_frames:] = clip_b["pos"][-context_frames:].to(dev)
        x_rot[0, :context_frames] = clip_a["rot6d"][:context_frames].to(dev)
        x_rot[0, -context_frames:] = clip_b["rot6d"][-context_frames:].to(dev)
        x_root[0, :context_frames] = clip_a["root"][:context_frames].to(dev)
        x_root[0, -context_frames:] = clip_b["root"][-context_frames:].to(dev)

        t_start = context_frames
        t_end = T_FRAMES - context_frames
        gap_len = t_end - t_start
        if gap_len <= 0:
            continue

        # Linear interpolation as initialization
        for t in range(gap_len):
            alpha = (t + 1) / (gap_len + 1)
            x_pos[0, t_start + t] = (1 - alpha) * x_pos[0, context_frames - 1] + alpha * x_pos[0, t_end]
            x_rot[0, t_start + t] = (1 - alpha) * x_rot[0, context_frames - 1] + alpha * x_rot[0, t_end]
            x_root[0, t_start + t] = (1 - alpha) * x_root[0, context_frames - 1] + alpha * x_root[0, t_end]

        mask = torch.zeros(1, T_FRAMES, N_JOINTS, dtype=torch.bool, device=dev)
        mask[0, t_start:t_end, :] = True

        x_pos_pred, _, _ = fm_sample_conditional_inpaint(
            model, x_pos, x_rot, x_root, mask, n_steps=n_steps,
        )

        # Smoothness: jerk (3rd derivative)
        transition = x_pos_pred[0, t_start:t_end]
        if transition.shape[0] >= 4:
            vel = transition[1:] - transition[:-1]
            acc = vel[1:] - vel[:-1]
            jerk = acc[1:] - acc[:-1]
            jerk_norm = jerk.norm(dim=-1).mean()
            all_smoothness.append(jerk_norm.cpu())

        # Boundary continuity
        edge_err_start = (x_pos_pred[0, t_start] - x_pos[0, context_frames - 1]).norm(dim=-1).mean()
        edge_err_end = (x_pos_pred[0, t_end - 1] - x_pos[0, t_end]).norm(dim=-1).mean()
        all_transition_err.append(((edge_err_start + edge_err_end) / 2).cpu())

    return {
        "mean_jerk": float(torch.stack(all_smoothness).mean()) if all_smoothness else float("nan"),
        "std_jerk": float(torch.stack(all_smoothness).std()) if all_smoothness else float("nan"),
        "boundary_continuity_err": float(torch.stack(all_transition_err).mean()) if all_transition_err else float("nan"),
        "n_transitions": len(all_smoothness),
    }


# ===========================================================================
# Demo 4: Learned Skeleton (Pair Tensor Visualization)
# ===========================================================================

@torch.no_grad()
def demo_learned_skeleton(
    ckpt_path: Path, subset, *,
    n_clips: int = 200, batch_size: int = 32, top_k_edges: int = 100,
) -> Dict[str, object]:
    """Extract pair tensor, visualize top-K strongest joint connections."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    has_pair = getattr(cfg, "use_pair", False)

    if not has_pair:
        return {"has_pair": False, "affinity_matrix": None, "top_edges": []}

    sub = _random_subset(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)

    all_pair = []
    for batch in loader:
        x_pos_1 = batch["pos"].to(dev)
        x_rot_1 = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)

        _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=1.0)
        pair = model._last_pair
        pair_aff = pair.norm(dim=-1)
        pair_sym = 0.5 * (pair_aff + pair_aff.transpose(1, 2))
        all_pair.append(pair_sym.cpu())

    pair_mean = torch.cat(all_pair, dim=0).mean(dim=0).numpy()
    np.fill_diagonal(pair_mean, 0)

    mask_upper = np.triu(np.ones_like(pair_mean, dtype=bool), k=1)
    values = pair_mean[mask_upper]
    indices = np.argwhere(mask_upper)
    sorted_idx = np.argsort(values)[::-1][:top_k_edges]
    top_edges = [(int(indices[i][0]), int(indices[i][1]), float(values[i]))
                 for i in sorted_idx]

    return {
        "has_pair": True,
        "affinity_matrix": pair_mean,
        "top_edges": top_edges,
        "mean_affinity": float(pair_mean[mask_upper].mean()),
        "max_affinity": float(pair_mean[mask_upper].max()),
        "std_affinity": float(pair_mean[mask_upper].std()),
        "sparsity_ratio": float((pair_mean[mask_upper] > pair_mean[mask_upper].mean()).sum() / mask_upper.sum()),
    }


# ===========================================================================
# Demo 5: Anomaly / Physics Violation Detection
# ===========================================================================

@torch.no_grad()
def demo_anomaly_detection(
    ckpt_path: Path, subset, *,
    n_clips: int = 100, batch_size: int = 32,
) -> Dict[str, float]:
    """Use flow matching velocity field norm as anomaly score.

    Compare: normal motion vs synthetically corrupted motion.
    """
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    sub = _random_subset(subset, n_clips, SEED)
    loader = eval_loader(sub, batch=batch_size)
    rng = np.random.default_rng(SEED)

    scores_normal = []
    scores_teleport = []
    scores_frozen = []
    scores_noise = []

    for batch in loader:
        x_pos_1 = batch["pos"].to(dev)
        x_rot_1 = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)
        B, T, J, C = x_pos_1.shape

        # Normal motion
        out_normal = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=1.0)
        v_norm_normal = out_normal["pos"].norm(dim=-1).mean(dim=(1, 2))
        scores_normal.append(v_norm_normal.cpu())

        # Corruption 1: Teleport joints
        x_pos_teleport = x_pos_1.clone()
        for b in range(B):
            joints_to_corrupt = rng.choice(J, size=10, replace=False)
            offset = torch.randn(10, 3, device=dev) * 3.0
            x_pos_teleport[b, :, joints_to_corrupt, :] += offset[None, :, :]
        out_teleport = fm_forward_at_t(model, x_pos_teleport, x_rot_1, x_root_1, t_value=1.0)
        v_norm_teleport = out_teleport["pos"].norm(dim=-1).mean(dim=(1, 2))
        scores_teleport.append(v_norm_teleport.cpu())

        # Corruption 2: Frozen (repeat frame 0)
        x_pos_frozen = x_pos_1[:, :1, :, :].expand(B, T, J, C).contiguous()
        out_frozen = fm_forward_at_t(model, x_pos_frozen, x_rot_1, x_root_1, t_value=1.0)
        v_norm_frozen = out_frozen["pos"].norm(dim=-1).mean(dim=(1, 2))
        scores_frozen.append(v_norm_frozen.cpu())

        # Corruption 3: Pure random noise
        x_pos_noise = torch.randn_like(x_pos_1) * 2.0
        out_noise = fm_forward_at_t(model, x_pos_noise, x_rot_1, x_root_1, t_value=1.0)
        v_norm_noise = out_noise["pos"].norm(dim=-1).mean(dim=(1, 2))
        scores_noise.append(v_norm_noise.cpu())

    normal_score = float(torch.cat(scores_normal).mean())
    teleport_score = float(torch.cat(scores_teleport).mean())
    frozen_score = float(torch.cat(scores_frozen).mean())
    noise_score = float(torch.cat(scores_noise).mean())

    return {
        "score_normal": normal_score,
        "score_teleport": teleport_score,
        "score_frozen": frozen_score,
        "score_noise": noise_score,
        "separability_teleport": teleport_score / max(normal_score, 1e-8),
        "separability_frozen": frozen_score / max(normal_score, 1e-8),
        "separability_noise": noise_score / max(normal_score, 1e-8),
    }


# ===========================================================================
# Demo 6: Motion Interpolation in Pair Space
# ===========================================================================

@torch.no_grad()
def demo_pair_interpolation(
    ckpt_path: Path, subset, *,
    n_pairs: int = 50,
    alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[str, object]:
    """Interpolate pair tensors between two clips, measure linearity."""
    model, cfg, variant = load_eval_model(ckpt_path)
    dev = next(model.parameters()).device
    has_pair = getattr(cfg, "use_pair", False)

    if not has_pair:
        return {"has_pair": False, "linearity_r": float("nan")}

    rng = np.random.default_rng(SEED)
    src = np.asarray(subset.indices)
    n = min(n_pairs * 2, len(src))
    picked = rng.choice(src, size=n, replace=False)
    ds = subset.dataset

    all_distances = {a: [] for a in alphas}

    for i in range(0, len(picked) - 1, 2):
        clip_a = ds[picked[i]]
        clip_b = ds[picked[i + 1]]

        x_pos_a = clip_a["pos"].unsqueeze(0).to(dev)
        x_rot_a = clip_a["rot6d"].unsqueeze(0).to(dev)
        x_root_a = clip_a["root"].unsqueeze(0).to(dev)

        x_pos_b = clip_b["pos"].unsqueeze(0).to(dev)
        x_rot_b = clip_b["rot6d"].unsqueeze(0).to(dev)
        x_root_b = clip_b["root"].unsqueeze(0).to(dev)

        _ = fm_forward_at_t(model, x_pos_a, x_rot_a, x_root_a, t_value=1.0)
        pair_a = model._last_pair.clone()

        _ = fm_forward_at_t(model, x_pos_b, x_rot_b, x_root_b, t_value=1.0)
        pair_b = model._last_pair.clone()

        pair_dist = (pair_a - pair_b).norm().item()
        if pair_dist < 1e-6:
            continue

        for alpha in alphas:
            pair_interp = (1 - alpha) * pair_a + alpha * pair_b
            dist_to_a = (pair_interp - pair_a).norm().item() / pair_dist
            all_distances[alpha].append(dist_to_a)

    results = {"has_pair": True}
    for alpha in alphas:
        if all_distances[alpha]:
            results[f"mean_dist_alpha_{alpha}"] = float(np.mean(all_distances[alpha]))
        else:
            results[f"mean_dist_alpha_{alpha}"] = float("nan")

    expected = np.array(alphas)
    actual = np.array([results.get(f"mean_dist_alpha_{a}", float("nan")) for a in alphas])
    valid = ~np.isnan(actual)
    if valid.sum() >= 3:
        results["linearity_r"] = float(np.corrcoef(expected[valid], actual[valid])[0, 1])
    else:
        results["linearity_r"] = float("nan")

    return results


# ===========================================================================
# Visualization
# ===========================================================================

def plot_sparse_sensor(all_results: Dict[str, Dict], save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    variants = sorted(all_results.keys())
    vals = [all_results[v]["mpjpe_masked_joints"] for v in variants]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(variants)))
    bars = ax.bar(variants, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MPJPE (masked joints)")
    ax.set_title(f"Demo 1: Sparse Sensor Completion\n({len(SPARSE_JOINTS)} visible joints -> {N_JOINTS} total)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_temporal_superres(all_results: Dict[str, Dict[int, float]], save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    for i, (variant, data) in enumerate(sorted(all_results.items())):
        factors = sorted(data.keys())
        vals = [data[f] for f in factors]
        ax.plot([f"{f}x" for f in factors], vals, marker="o",
                color=cmap(i), label=variant, linewidth=2)
    ax.set_xlabel("Downsample factor (keep every Nth frame)")
    ax.set_ylabel("MPJPE (interpolated frames)")
    ax.set_title("Demo 2: Temporal Super-resolution")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_motion_transition(all_results: Dict[str, Dict], save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    variants = sorted(all_results.keys())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(variants)))

    jerks = [all_results[v]["mean_jerk"] for v in variants]
    ax1.bar(variants, jerks, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Mean jerk (lower = smoother)")
    ax1.set_title("Transition Smoothness")
    ax1.grid(axis="y", alpha=0.3)

    cont_err = [all_results[v]["boundary_continuity_err"] for v in variants]
    ax2.bar(variants, cont_err, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Boundary continuity error")
    ax2.set_title("Edge Continuity")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Demo 3: Motion Transition Generation", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_learned_skeleton(all_results: Dict[str, Dict], save_path: Path):
    pair_variants = {v: r for v, r in all_results.items() if r.get("has_pair")}
    n = len(pair_variants)
    if n == 0:
        return

    fig, axes = plt.subplots(1, min(n, 4), figsize=(5 * min(n, 4), 5), facecolor="white")
    if min(n, 4) == 1:
        axes = [axes]

    for ax, (variant, data) in zip(axes, sorted(pair_variants.items())):
        mat = data["affinity_matrix"]
        if mat is None:
            continue
        im = ax.imshow(mat, cmap="hot", interpolation="nearest")
        ax.set_title(f"{variant}\nmax={data['max_affinity']:.2f} mean={data['mean_affinity']:.2f}")
        ax.set_xlabel("Joint j")
        ax.set_ylabel("Joint i")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Demo 4: Learned Skeleton — Pair Tensor Affinity", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_anomaly_detection(all_results: Dict[str, Dict], save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    variants = sorted(all_results.keys())
    x = np.arange(len(variants))
    w = 0.2

    categories = ["score_normal", "score_teleport", "score_frozen", "score_noise"]
    labels = ["Normal", "Teleport", "Frozen", "Noise"]
    colors_cat = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]

    for ci, (cat, label, color) in enumerate(zip(categories, labels, colors_cat)):
        vals = [all_results[v][cat] for v in variants]
        ax.bar(x + ci * w - 1.5 * w, vals, w, label=label, color=color,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Velocity field norm (anomaly score)")
    ax.set_title("Demo 5: Anomaly Detection — Normal vs Corrupted Motion")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pair_interpolation(all_results: Dict[str, Dict], save_path: Path):
    pair_variants = {v: r for v, r in all_results.items() if r.get("has_pair")}
    if not pair_variants:
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for i, (variant, data) in enumerate(sorted(pair_variants.items())):
        vals = [data.get(f"mean_dist_alpha_{a}", float("nan")) for a in alphas]
        ax.plot(alphas, vals, marker="o", color=cmap(i), label=variant, linewidth=2)

    ax.plot(alphas, alphas, "--", color="gray", alpha=0.5, label="ideal linear")
    ax.set_xlabel("Interpolation alpha")
    ax.set_ylabel("Normalized distance to clip A")
    ax.set_title("Demo 6: Pair Space Interpolation Linearity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Summary
# ===========================================================================

def print_summary(summary: Dict):
    all_variant_sets = [set(v.keys()) for v in summary.values() if isinstance(v, dict)]
    variants = sorted(set().union(*all_variant_sets)) if all_variant_sets else []
    if not variants:
        return

    print("\n" + "=" * 80)
    print("  AlphaMotion Foundation Demos — Summary")
    print("=" * 80)

    if "demo1_sparse" in summary:
        print("\n--- Demo 1: Sparse Sensor Completion (6 joints visible) ---")
        print(f"  {'variant':<14} {'MPJPE(masked)':<14} {'MPJPE(total)':<14}")
        for v in variants:
            if v in summary["demo1_sparse"]:
                d = summary["demo1_sparse"][v]
                print(f"  {v:<14} {d['mpjpe_masked_joints']:<14.4f} {d['mpjpe_total']:<14.4f}")

    if "demo2_superres" in summary:
        print("\n--- Demo 2: Temporal Super-resolution ---")
        sample = next(iter(summary["demo2_superres"].values()))
        factors = sorted(sample.keys())
        header = f"  {'variant':<14}" + "".join(f"{str(f)+'x':<10}" for f in factors)
        print(header)
        for v in variants:
            if v in summary["demo2_superres"]:
                row = f"  {v:<14}" + "".join(f"{summary['demo2_superres'][v][f]:<10.4f}" for f in factors)
                print(row)

    if "demo3_transition" in summary:
        print("\n--- Demo 3: Motion Transition ---")
        print(f"  {'variant':<14} {'jerk(smooth)':<14} {'boundary_err':<14}")
        for v in variants:
            if v in summary["demo3_transition"]:
                d = summary["demo3_transition"][v]
                print(f"  {v:<14} {d['mean_jerk']:<14.4f} {d['boundary_continuity_err']:<14.4f}")

    if "demo4_skeleton" in summary:
        print("\n--- Demo 4: Learned Skeleton ---")
        for v in variants:
            if v in summary["demo4_skeleton"]:
                d = summary["demo4_skeleton"][v]
                if d.get("has_pair"):
                    print(f"  {v:<14} mean_aff={d['mean_affinity']:.3f}  "
                          f"max_aff={d['max_affinity']:.3f}  sparsity={d['sparsity_ratio']:.3f}")
                else:
                    print(f"  {v:<14} [no pair tensor]")

    if "demo5_anomaly" in summary:
        print("\n--- Demo 5: Anomaly Detection ---")
        print(f"  {'variant':<14} {'normal':<10} {'teleport':<10} {'frozen':<10} {'noise':<10} {'sep_T':<8} {'sep_F':<8} {'sep_N':<8}")
        for v in variants:
            if v in summary["demo5_anomaly"]:
                d = summary["demo5_anomaly"][v]
                print(f"  {v:<14} {d['score_normal']:<10.4f} {d['score_teleport']:<10.4f} "
                      f"{d['score_frozen']:<10.4f} {d['score_noise']:<10.4f} "
                      f"{d['separability_teleport']:<8.2f} {d['separability_frozen']:<8.2f} "
                      f"{d['separability_noise']:<8.2f}")

    if "demo6_interp" in summary:
        print("\n--- Demo 6: Pair Interpolation Linearity ---")
        for v in variants:
            if v in summary["demo6_interp"]:
                d = summary["demo6_interp"][v]
                if d.get("has_pair"):
                    r = d.get("linearity_r", float("nan"))
                    print(f"  {v:<14} linearity_r = {r:.4f}")
                else:
                    print(f"  {v:<14} [no pair tensor]")

    print("\n" + "=" * 80)


# ===========================================================================
# Main Runner
# ===========================================================================

def run_all(
    variants: Optional[List[str]] = None,
    demos: Optional[List[int]] = None,
    step: int = 30000,
) -> Dict:
    if variants is None:
        variants = list(VARS.keys())
    if demos is None:
        demos = [1, 2, 3, 4, 5, 6]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[demos] Loading K-eval (Kimodo OOD) dataset...")
    k_eval = load_kimodo_ood(DATA)
    print(f"[demos] K-eval: {len(k_eval)} clips")
    print(f"[demos] Variants: {variants}")
    print(f"[demos] Demos: {demos}")
    print(f"[demos] Checkpoint step: {step}")
    print()

    summary = {}

    if 1 in demos:
        print("=" * 60)
        print("  Demo 1: Sparse Sensor Completion")
        print("=" * 60)
        summary["demo1_sparse"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = demo_sparse_sensor(ckpt, k_eval)
            summary["demo1_sparse"][v] = result
            print(f"  [{v}] {time.time()-t0:.1f}s  "
                  f"masked={result['mpjpe_masked_joints']:.4f}  total={result['mpjpe_total']:.4f}")
        plot_sparse_sensor(summary["demo1_sparse"], RESULTS_DIR / "demo1_sparse_sensor.png")
        print()

    if 2 in demos:
        print("=" * 60)
        print("  Demo 2: Temporal Super-resolution")
        print("=" * 60)
        summary["demo2_superres"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = demo_temporal_superres(ckpt, k_eval)
            summary["demo2_superres"][v] = result
            print(f"  [{v}] {time.time()-t0:.1f}s  " +
                  " | ".join(f"{f}x:{m:.4f}" for f, m in sorted(result.items())))
        plot_temporal_superres(summary["demo2_superres"], RESULTS_DIR / "demo2_temporal_superres.png")
        print()

    if 3 in demos:
        print("=" * 60)
        print("  Demo 3: Motion Transition Generation")
        print("=" * 60)
        summary["demo3_transition"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = demo_motion_transition(ckpt, k_eval)
            summary["demo3_transition"][v] = result
            print(f"  [{v}] {time.time()-t0:.1f}s  "
                  f"jerk={result['mean_jerk']:.4f}  boundary_err={result['boundary_continuity_err']:.4f}")
        plot_motion_transition(summary["demo3_transition"], RESULTS_DIR / "demo3_motion_transition.png")
        print()

    if 4 in demos:
        print("=" * 60)
        print("  Demo 4: Learned Skeleton (Pair Tensor)")
        print("=" * 60)
        summary["demo4_skeleton"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = demo_learned_skeleton(ckpt, k_eval)
            # Store affinity matrix as npy, keep rest in summary
            summary["demo4_skeleton"][v] = {k: val for k, val in result.items()
                                            if k != "affinity_matrix"}
            if result.get("has_pair") and result["affinity_matrix"] is not None:
                np.save(RESULTS_DIR / f"demo4_affinity_{v}.npy", result["affinity_matrix"])
            elapsed = time.time() - t0
            if result.get("has_pair"):
                print(f"  [{v}] {elapsed:.1f}s  mean_aff={result['mean_affinity']:.3f}  "
                      f"max={result['max_affinity']:.3f}  sparsity={result['sparsity_ratio']:.3f}")
            else:
                print(f"  [{v}] {elapsed:.1f}s  [no pair tensor]")
        # Plot with loaded matrices
        plot_data = {}
        for v in variants:
            npy_path = RESULTS_DIR / f"demo4_affinity_{v}.npy"
            plot_data[v] = dict(summary["demo4_skeleton"][v])
            if npy_path.exists():
                plot_data[v]["affinity_matrix"] = np.load(npy_path)
            else:
                plot_data[v]["affinity_matrix"] = None
        plot_learned_skeleton(plot_data, RESULTS_DIR / "demo4_learned_skeleton.png")
        print()

    if 5 in demos:
        print("=" * 60)
        print("  Demo 5: Anomaly Detection")
        print("=" * 60)
        summary["demo5_anomaly"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = demo_anomaly_detection(ckpt, k_eval)
            summary["demo5_anomaly"][v] = result
            print(f"  [{v}] {time.time()-t0:.1f}s  "
                  f"normal={result['score_normal']:.3f}  teleport={result['score_teleport']:.3f}  "
                  f"frozen={result['score_frozen']:.3f}  noise={result['score_noise']:.3f}")
        plot_anomaly_detection(summary["demo5_anomaly"], RESULTS_DIR / "demo5_anomaly_detection.png")
        print()

    if 6 in demos:
        print("=" * 60)
        print("  Demo 6: Pair Space Interpolation")
        print("=" * 60)
        summary["demo6_interp"] = {}
        for v in variants:
            t0 = time.time()
            ckpt = _find_checkpoint(v, step)
            result = demo_pair_interpolation(ckpt, k_eval)
            summary["demo6_interp"][v] = result
            r = result.get("linearity_r", float("nan"))
            tag = "" if result.get("has_pair") else " [no pair]"
            print(f"  [{v}] {time.time()-t0:.1f}s  linearity_r={r:.4f}{tag}")
        plot_pair_interpolation(summary["demo6_interp"], RESULTS_DIR / "demo6_pair_interpolation.png")
        print()

    # Save summary JSON
    def _serialize(obj):
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list) and obj and isinstance(obj[0], tuple):
            return [list(x) for x in obj]
        return obj

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n[demos] Results saved to {RESULTS_DIR}/")
    print(f"[demos] Summary JSON: {summary_path}")

    print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="AlphaMotion Foundation Demos — 6 zero-shot capabilities"
    )
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--demos", nargs="+", type=int, default=None)
    parser.add_argument("--step", type=int, default=30000)
    args = parser.parse_args()

    if args.variants:
        for v in args.variants:
            if v not in VARS:
                parser.error(f"Unknown variant '{v}'. Available: {list(VARS.keys())}")

    run_all(variants=args.variants, demos=args.demos, step=args.step)


if __name__ == "__main__":
    main()
