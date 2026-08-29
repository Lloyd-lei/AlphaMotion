"""AlphaMotion Foundation Demos — GIF Visualization.

Generates side-by-side mesh GIFs for each demo capability,
comparing all 6 variants on the same clip.

Usage:
    python notebook/viz_foundation_demos.py                     # all demos
    python notebook/viz_foundation_demos.py --demos 1 2 3
    python notebook/viz_foundation_demos.py --clip 500          # specific clip index

Outputs to: notebook/eval_foundation_demos/gifs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

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
    load_eval_model, fm_forward_at_t, fm_one_step_denoise,
    fm_sample_conditional_inpaint, DATA, VARS, ENV,
)
from data_loader import load_mixed, load_kimodo_ood
from soma_skeleton import rot6d_to_matrix, forward_kinematics_torch
from lbs import load_skin, lbs_sequence
from mesh_render import render_mesh_gif

GIF_DIR = NOTEBOOK / "eval_foundation_demos" / "gifs"
CKPT_ROOT = NOTEBOOK / "02_resources" / "checkpoints"
SEED = ENV["seed"]
T_FRAMES = DATA["sample"]["T"]
N_JOINTS = DATA["skeleton"]["n_joints"]
SPARSE_JOINTS = [0, 15, 20, 21, 7, 8]

PALETTE = {
    "gt":            (110, 175, 195),
    "full":          ( 95, 145, 200),
    "opm_only":      (110, 175, 110),
    "triangle_only": (210, 110, 110),
    "pair_static":   (165, 130, 195),
    "axial_only":    (170, 170, 170),
    "baseline":      (165, 115,  85),
}

VARIANTS = list(VARS.keys())


def _find_checkpoint(variant: str, step: int = 30000) -> Path:
    p = CKPT_ROOT / variant / f"step_{step}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _load_dataset_and_skin():
    ds = load_mixed(DATA)
    skin = load_skin(NOTEBOOK / "01_resources" / "assets" / "skin_standard.npz")
    stats = {
        "pos_mean": ds.mean[0],
        "pos_std": ds.std[0],
        "root_mean": ds.root_mean[0],
        "root_std": ds.root_std[0],
        "bone_offsets": ds.bone_offsets,
        "parents": ds.parents,
    }
    return ds, skin, stats


@torch.no_grad()
def _to_mesh_verts(rot6d, root, stats, skin, device="cuda"):
    """Convert normalized (rot6d, root) -> mesh vertices [T, V, 3]."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if rot6d.dim() == 3:
        rot6d = rot6d.unsqueeze(0)
        root = root.unsqueeze(0)
    rot6d = rot6d.to(dev)
    root = root.to(dev)

    root_std = torch.from_numpy(stats["root_std"]).to(dev)
    root_mean = torch.from_numpy(stats["root_mean"]).to(dev)
    bone_offsets = torch.from_numpy(stats["bone_offsets"]).float().to(dev)
    parents = torch.from_numpy(stats["parents"]).to(dev)

    root_un = root * root_std + root_mean
    R_local = rot6d_to_matrix(rot6d)
    global_pos, global_rot = forward_kinematics_torch(R_local, root_un, bone_offsets, parents)

    grot = global_rot[0].cpu().numpy().astype(np.float32)
    jts = global_pos[0].cpu().numpy().astype(np.float32)
    verts = lbs_sequence(skin, grot, jts)
    return verts


def _render(verts, skin, out_path, label="", stride=2, fps=15):
    """Render mesh verts to GIF."""
    render_mesh_gif(
        verts[::stride], skin["faces"], out_path,
        viewport=(360, 360), fps=fps,
        mesh_color=PALETTE.get(label, (210, 178, 145)),
    )


# ===========================================================================
# Demo 1 GIF: Sparse Sensor Completion
# ===========================================================================

@torch.no_grad()
def viz_sparse_sensor(clip_idx: int, ds, skin, stats, out_dir: Path, step: int = 30000):
    """GT vs each variant's sparse-sensor completion."""
    out_dir = out_dir / "demo1_sparse_sensor"
    out_dir.mkdir(parents=True, exist_ok=True)

    item = ds[clip_idx]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GT
    verts_gt = _to_mesh_verts(item["rot6d"], item["root"], stats, skin)
    _render(verts_gt, skin, out_dir / "gt.gif", "gt")
    print(f"  [gt] -> gt.gif")

    # Per variant
    for variant in VARIANTS:
        model, cfg, _ = load_eval_model(_find_checkpoint(variant, step))
        x_pos = item["pos"].unsqueeze(0).to(dev)
        x_rot = item["rot6d"].unsqueeze(0).to(dev)
        x_root = item["root"].unsqueeze(0).to(dev)
        B, T, J, _ = x_pos.shape

        mask = torch.ones(B, T, J, dtype=torch.bool, device=dev)
        for j in SPARSE_JOINTS:
            mask[:, :, j] = False

        _, x_rot_pred, x_root_pred = fm_sample_conditional_inpaint(
            model, x_pos, x_rot, x_root, mask, n_steps=16,
        )

        # Use predicted rot6d for masked joints, GT for visible
        rot_out = item["rot6d"].unsqueeze(0).to(dev).clone()
        rot_out[mask.unsqueeze(-1).expand_as(rot_out)] = x_rot_pred[mask.unsqueeze(-1).expand_as(x_rot_pred)]

        verts = _to_mesh_verts(rot_out[0].cpu(), x_root_pred[0].cpu(), stats, skin)
        _render(verts, skin, out_dir / f"{variant}.gif", variant)
        print(f"  [{variant}] -> {variant}.gif")


# ===========================================================================
# Demo 2 GIF: Temporal Super-resolution
# ===========================================================================

@torch.no_grad()
def viz_temporal_superres(clip_idx: int, ds, skin, stats, out_dir: Path, step: int = 30000, factor: int = 3):
    """GT vs each variant's temporal interpolation (keep every Nth frame)."""
    out_dir = out_dir / "demo2_temporal_superres"
    out_dir.mkdir(parents=True, exist_ok=True)

    item = ds[clip_idx]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verts_gt = _to_mesh_verts(item["rot6d"], item["root"], stats, skin)
    _render(verts_gt, skin, out_dir / "gt.gif", "gt")
    print(f"  [gt] -> gt.gif")

    for variant in VARIANTS:
        model, cfg, _ = load_eval_model(_find_checkpoint(variant, step))
        x_pos = item["pos"].unsqueeze(0).to(dev)
        x_rot = item["rot6d"].unsqueeze(0).to(dev)
        x_root = item["root"].unsqueeze(0).to(dev)
        B, T, J, _ = x_pos.shape

        mask = torch.ones(B, T, J, dtype=torch.bool, device=dev)
        for t in range(0, T, factor):
            mask[:, t, :] = False

        _, x_rot_pred, x_root_pred = fm_sample_conditional_inpaint(
            model, x_pos, x_rot, x_root, mask, n_steps=16,
        )

        verts = _to_mesh_verts(x_rot_pred[0].cpu(), x_root_pred[0].cpu(), stats, skin)
        _render(verts, skin, out_dir / f"{variant}.gif", variant)
        print(f"  [{variant}] -> {variant}.gif")


# ===========================================================================
# Demo 3 GIF: Motion Transition
# ===========================================================================

@torch.no_grad()
def viz_motion_transition(clip_idx_a: int, clip_idx_b: int, ds, skin, stats, out_dir: Path, step: int = 30000):
    """Show transition between two different clips."""
    out_dir = out_dir / "demo3_transition"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_a = ds[clip_idx_a]
    item_b = ds[clip_idx_b]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = 15

    # GT clip A
    verts_a = _to_mesh_verts(item_a["rot6d"], item_a["root"], stats, skin)
    _render(verts_a, skin, out_dir / "gt_clip_a.gif", "gt")
    # GT clip B
    verts_b = _to_mesh_verts(item_b["rot6d"], item_b["root"], stats, skin)
    _render(verts_b, skin, out_dir / "gt_clip_b.gif", "gt")
    print(f"  [gt] clip_a + clip_b")

    for variant in VARIANTS:
        model, cfg, _ = load_eval_model(_find_checkpoint(variant, step))

        x_pos = torch.zeros(1, T_FRAMES, N_JOINTS, 3, device=dev)
        x_rot = torch.zeros(1, T_FRAMES, N_JOINTS, 6, device=dev)
        x_root = torch.zeros(1, T_FRAMES, 3, device=dev)

        x_pos[0, :context] = item_a["pos"][:context].to(dev)
        x_pos[0, -context:] = item_b["pos"][-context:].to(dev)
        x_rot[0, :context] = item_a["rot6d"][:context].to(dev)
        x_rot[0, -context:] = item_b["rot6d"][-context:].to(dev)
        x_root[0, :context] = item_a["root"][:context].to(dev)
        x_root[0, -context:] = item_b["root"][-context:].to(dev)

        t_start = context
        t_end = T_FRAMES - context
        gap_len = t_end - t_start

        for t in range(gap_len):
            alpha = (t + 1) / (gap_len + 1)
            x_pos[0, t_start + t] = (1 - alpha) * x_pos[0, context - 1] + alpha * x_pos[0, t_end]
            x_rot[0, t_start + t] = (1 - alpha) * x_rot[0, context - 1] + alpha * x_rot[0, t_end]
            x_root[0, t_start + t] = (1 - alpha) * x_root[0, context - 1] + alpha * x_root[0, t_end]

        mask = torch.zeros(1, T_FRAMES, N_JOINTS, dtype=torch.bool, device=dev)
        mask[0, t_start:t_end, :] = True

        _, x_rot_pred, x_root_pred = fm_sample_conditional_inpaint(
            model, x_pos, x_rot, x_root, mask, n_steps=16,
        )

        verts = _to_mesh_verts(x_rot_pred[0].cpu(), x_root_pred[0].cpu(), stats, skin)
        _render(verts, skin, out_dir / f"{variant}.gif", variant)
        print(f"  [{variant}] -> {variant}.gif")


# ===========================================================================
# Demo 4 GIF: Future Prediction
# ===========================================================================

@torch.no_grad()
def viz_future_prediction(clip_idx: int, ds, skin, stats, out_dir: Path, step: int = 30000, horizon: int = 45):
    """Mask last N frames, predict future, show full sequence."""
    out_dir = out_dir / "demo4_future_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    item = ds[clip_idx]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verts_gt = _to_mesh_verts(item["rot6d"], item["root"], stats, skin)
    _render(verts_gt, skin, out_dir / "gt.gif", "gt")
    print(f"  [gt] -> gt.gif")

    for variant in VARIANTS:
        model, cfg, _ = load_eval_model(_find_checkpoint(variant, step))
        x_pos = item["pos"].unsqueeze(0).to(dev)
        x_rot = item["rot6d"].unsqueeze(0).to(dev)
        x_root = item["root"].unsqueeze(0).to(dev)
        B, T, J, _ = x_pos.shape

        mask = torch.zeros(B, T, J, dtype=torch.bool, device=dev)
        mask[:, -horizon:, :] = True

        _, x_rot_pred, x_root_pred = fm_sample_conditional_inpaint(
            model, x_pos, x_rot, x_root, mask, n_steps=16,
        )

        verts = _to_mesh_verts(x_rot_pred[0].cpu(), x_root_pred[0].cpu(), stats, skin)
        _render(verts, skin, out_dir / f"{variant}.gif", variant)
        print(f"  [{variant}] -> {variant}.gif")


# ===========================================================================
# Demo 5 GIF: Anomaly (show what corrupted motion looks like when "fixed")
# ===========================================================================

@torch.no_grad()
def viz_anomaly_correction(clip_idx: int, ds, skin, stats, out_dir: Path, step: int = 30000):
    """Show: GT, teleported input, model's 1-step denoise of teleported input."""
    out_dir = out_dir / "demo5_anomaly_correction"
    out_dir.mkdir(parents=True, exist_ok=True)

    item = ds[clip_idx]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    verts_gt = _to_mesh_verts(item["rot6d"], item["root"], stats, skin)
    _render(verts_gt, skin, out_dir / "gt.gif", "gt")
    print(f"  [gt] -> gt.gif")

    # Create teleported version
    x_pos_1 = item["pos"].unsqueeze(0).to(dev)
    x_rot_1 = item["rot6d"].unsqueeze(0).to(dev)
    x_root_1 = item["root"].unsqueeze(0).to(dev)
    B, T, J, _ = x_pos_1.shape

    x_pos_corrupt = x_pos_1.clone()
    joints_to_corrupt = rng.choice(J, size=15, replace=False)
    offset = torch.randn(15, 3, device=dev) * 2.0
    x_pos_corrupt[0, :, joints_to_corrupt, :] += offset[None, :, :]

    for variant in VARIANTS:
        model, cfg, _ = load_eval_model(_find_checkpoint(variant, step))

        # 1-step denoise from corrupted input (treat as t=0.8)
        t_val = 0.8
        t = torch.full((1,), t_val, device=dev)
        out = model(x_pos_corrupt, x_rot_1, x_root_1, t)
        x_rot_hat = x_rot_1 + (1 - t.view(1, 1, 1, 1)) * out["rot6d"]
        x_root_hat = x_root_1 + (1 - t.view(1, 1, 1)) * out["root"]

        verts = _to_mesh_verts(x_rot_hat[0].cpu(), x_root_hat[0].cpu(), stats, skin)
        _render(verts, skin, out_dir / f"{variant}.gif", variant)
        print(f"  [{variant}] -> {variant}.gif")


# ===========================================================================
# Demo 6 GIF: Unconditional Generation (pure noise -> motion)
# ===========================================================================

@torch.no_grad()
def viz_unconditional_generation(ds, skin, stats, out_dir: Path, step: int = 30000, n_steps: int = 32):
    """Generate motion from pure noise using Euler integration."""
    from _eval_common import fm_sample
    out_dir = out_dir / "demo6_unconditional_generation"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for variant in VARIANTS:
        model, cfg, _ = load_eval_model(_find_checkpoint(variant, step))
        T = T_FRAMES
        J = N_JOINTS

        _, x_rot_gen, x_root_gen = fm_sample(model, 1, T, J, n_steps=n_steps, device="cuda")

        verts = _to_mesh_verts(x_rot_gen[0].cpu(), x_root_gen[0].cpu(), stats, skin)
        _render(verts, skin, out_dir / f"{variant}.gif", variant)
        print(f"  [{variant}] -> {variant}.gif")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate GIF visualizations for foundation demos")
    parser.add_argument("--demos", nargs="+", type=int, default=None,
                        help="Which demos to visualize (1-6, default: all)")
    parser.add_argument("--clip", type=int, default=None,
                        help="Global clip index to use (default: auto-select from K-eval)")
    parser.add_argument("--clip-b", type=int, default=None,
                        help="Second clip for transition demo (default: auto)")
    parser.add_argument("--step", type=int, default=30000)
    args = parser.parse_args()

    demos = args.demos or [1, 2, 3, 4, 5, 6]
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    print("[viz] Loading dataset and skin...")
    ds, skin, stats = _load_dataset_and_skin()
    k_eval = load_kimodo_ood(DATA)

    # Select representative clips from K-eval
    rng = np.random.default_rng(SEED + 42)
    k_indices = np.asarray(k_eval.indices)

    if args.clip is not None:
        clip_idx = args.clip
    else:
        clip_idx = int(rng.choice(k_indices))

    clip_idx_b = args.clip_b if args.clip_b is not None else int(rng.choice(k_indices))

    print(f"[viz] Using clip {clip_idx} (and {clip_idx_b} for transition)")
    print(f"[viz] Output: {GIF_DIR}/")
    print()

    if 1 in demos:
        print("=" * 50)
        print("  Demo 1: Sparse Sensor Completion")
        print("=" * 50)
        viz_sparse_sensor(clip_idx, ds, skin, stats, GIF_DIR, args.step)
        print()

    if 2 in demos:
        print("=" * 50)
        print("  Demo 2: Temporal Super-resolution (3x)")
        print("=" * 50)
        viz_temporal_superres(clip_idx, ds, skin, stats, GIF_DIR, args.step, factor=3)
        print()

    if 3 in demos:
        print("=" * 50)
        print("  Demo 3: Motion Transition")
        print("=" * 50)
        viz_motion_transition(clip_idx, clip_idx_b, ds, skin, stats, GIF_DIR, args.step)
        print()

    if 4 in demos:
        print("=" * 50)
        print("  Demo 4: Future Prediction (45 frames)")
        print("=" * 50)
        viz_future_prediction(clip_idx, ds, skin, stats, GIF_DIR, args.step, horizon=45)
        print()

    if 5 in demos:
        print("=" * 50)
        print("  Demo 5: Anomaly Correction")
        print("=" * 50)
        viz_anomaly_correction(clip_idx, ds, skin, stats, GIF_DIR, args.step)
        print()

    if 6 in demos:
        print("=" * 50)
        print("  Demo 6: Unconditional Generation")
        print("=" * 50)
        viz_unconditional_generation(ds, skin, stats, GIF_DIR, args.step)
        print()

    print(f"\n[viz] Done! GIFs saved to {GIF_DIR}/")


if __name__ == "__main__":
    main()
