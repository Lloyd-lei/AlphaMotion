"""Render FM-reconstruction GIFs for inspection.

Given a clip from the dataset, we render TWO meshes side-by-side as a single
GIF per source: ground truth and predicted (FM 1-step denoise from t=0.95).

Public API:
    render_recon_gif_all(clip_idx, out_dir, ckpt_paths) -> list[Path]
        Renders one GIF per (gt + each ckpt). Names: gt.gif, full.gif, ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from _eval_common import (
    REPO, DATA, load_eval_model, fm_one_step_denoise,
)

import sys
SCRIPTS_01 = REPO / "notebook/01_resources/scripts"
if str(SCRIPTS_01) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_01))

from soma_skeleton import rot6d_to_matrix, forward_kinematics_torch
from lbs           import load_skin, lbs_sequence
from mesh_render   import render_mesh_gif


def _load_dataset_stats():
    """Cache-friendly lazy loader returning (ds, stats_dict)."""
    from data_loader import load_mixed
    ds = load_mixed(DATA)
    return ds, {
        "pos_mean":     ds.mean[0],
        "pos_std":      ds.std[0],
        "root_mean":    ds.root_mean[0],
        "root_std":     ds.root_std[0],
        "bone_offsets": ds.bone_offsets,
        "parents":      ds.parents,
    }


@torch.no_grad()
def _predict_recon_for_one_clip(model, item, stats, *, t_value: float = 0.95, device: str = "cuda"):
    """Run FM 1-step denoise → unnormalize → FK → return (global_rot, global_pos) numpy."""
    dev = next(model.parameters()).device
    x_pos_1  = item["pos"].unsqueeze(0).to(dev)
    x_rot_1  = item["rot6d"].unsqueeze(0).to(dev)
    x_root_1 = item["root"].unsqueeze(0).to(dev)

    pred_pos, pred_rot, pred_root = fm_one_step_denoise(
        model, x_pos_1, x_rot_1, x_root_1, t_value=t_value,
    )

    pos_std  = torch.from_numpy(stats["pos_std"]).to(dev)
    pos_mean = torch.from_numpy(stats["pos_mean"]).to(dev)
    root_std  = torch.from_numpy(stats["root_std"]).to(dev)
    root_mean = torch.from_numpy(stats["root_mean"]).to(dev)
    bone_offsets = torch.from_numpy(stats["bone_offsets"]).float().to(dev)
    parents      = torch.from_numpy(stats["parents"]).to(dev)

    # Unnormalize root to root-t0-relative world frame (no abs offset; LBS doesn't need it)
    pred_root_un = pred_root * root_std + root_mean   # [1, T, 3]

    # rot6d → local rotation matrices
    R_local = rot6d_to_matrix(pred_rot)                 # [1, T, J, 3, 3]

    # FK → (global_pos, global_rot)
    global_pos, global_rot = forward_kinematics_torch(
        R_local, pred_root_un, bone_offsets, parents,
    )
    return (global_rot[0].cpu().numpy().astype(np.float32),
            global_pos[0].cpu().numpy().astype(np.float32))


def render_recon_gif_all(
    clip_global_idx: int,
    out_dir: Path,
    ckpt_paths: Dict[str, Path],
    *,
    stride: int = 3,
    fps: int = 10,
    viewport=(360, 360),
    t_value: float = 0.95,
    device: str = "cuda",
) -> List[Path]:
    """Render GT + per-variant reconstruction GIFs side-by-side.

    Args:
        clip_global_idx: index into the full mixed dataset (not Subset-relative)
        out_dir:         where to write {gt.gif, <variant>.gif, ...}
        ckpt_paths:      dict {variant_name: Path to latest.pt}

    Returns list of written paths.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds, stats = _load_dataset_stats()
    skin = load_skin(REPO / "notebook/01_resources/assets/skin_standard.npz")
    written: List[Path] = []

    # Per-variant accent colors (RGB 0-255). GT uses calm teal so it stands out
    # immediately as "the reference"; predictions use warmer tints.
    PALETTE = {
        "gt":            (110, 175, 195),  # teal — reference
        "full":          ( 95, 145, 200),  # blue — main thesis variant
        "opm_only":      (110, 175, 110),  # green
        "triangle_only": (210, 110, 110),  # red
        "pair_static":   (165, 130, 195),  # purple
        "axial_only":    (170, 170, 170),  # gray
        "baseline":      (165, 115,  85),  # brown
    }

    # ----- GT mesh GIF -----
    raw = np.load(DATA["amass"]["npz"], allow_pickle=True)
    grot_gt = raw["global_rot_mats"][clip_global_idx][::stride]    # [T', J, 3, 3]
    jts_gt  = raw["joints_world"][clip_global_idx][::stride]       # [T', J, 3]
    jts_gt = jts_gt - raw["joints_world"][clip_global_idx][0:1, 0:1]
    verts_gt = lbs_sequence(skin, grot_gt, jts_gt)
    gt_path = out_dir / "gt.gif"
    render_mesh_gif(verts_gt, skin["faces"], gt_path,
                     viewport=viewport, fps=fps,
                     mesh_color=PALETTE["gt"])
    written.append(gt_path)
    print(f"  ✓ GT  → {gt_path.name}")

    # ----- Per-variant predicted GIF -----
    item = ds[clip_global_idx]
    for variant, ckpt in ckpt_paths.items():
        try:
            model, _, _ = load_eval_model(str(ckpt), device=device)
            grot_pred_full, jts_pred_full = _predict_recon_for_one_clip(
                model, item, stats, t_value=t_value, device=device,
            )
            grot_pred = grot_pred_full[::stride]
            jts_pred  = jts_pred_full[::stride]
            verts = lbs_sequence(skin, grot_pred, jts_pred)
            out = out_dir / f"{variant}.gif"
            render_mesh_gif(verts, skin["faces"], out,
                             viewport=viewport, fps=fps,
                             mesh_color=PALETTE.get(variant, (210, 178, 145)))
            written.append(out)
            print(f"  ✓ {variant:<14} → {out.name}")
        except Exception as e:
            print(f"  ✗ {variant:<14} FAILED: {type(e).__name__}: {e}")

    return written
