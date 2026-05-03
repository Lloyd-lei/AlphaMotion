"""Export MotionFormer predictions as Kimodo-compatible NPZ so the motion can
be loaded in Kimodo's viser demo (SMPL-mesh quality rendering).

Usage:
    python export_to_kimodo.py --model opm_only --sample 100 --mask-mode random
    python export_to_kimodo.py --model opm_only --sample 100 --mask-mode none
    python export_to_kimodo.py --export-gt --sample 100

Output: runs/kimodo_export/<tag>.npz  -- load via 7860 viser "Load Path" box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import KimodoMotionDataset, make_masked_batch
from motionformer import MotionFormerConfig, MotionFormer
from baseline import BaselineConfig, BaselineTransformer
from soma_skeleton import (
    rot6d_to_matrix, forward_kinematics_torch, build_parent_index,
)
from train import MF_VARIANTS


def load_model(variant: str, device: str, cfg_args: dict):
    if variant == "baseline":
        cfg = BaselineConfig(**cfg_args, hidden=128, depth=12, heads=4, ffn_mult=4)
        model = BaselineTransformer(cfg).to(device)
    else:
        variant_kw = MF_VARIANTS[variant]
        cfg = MotionFormerConfig(**cfg_args, hidden=128, pair_hidden=32, depth=6,
                                 heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16,
                                 **variant_kw)
        model = MotionFormer(cfg).to(device)
    state = torch.load(Path("runs") / variant / "final.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def export_from_gt(sample_idx: int, out_path: Path, data_cache: str):
    """Dump ground-truth Kimodo sample — no model involved, as a visual baseline."""
    d = np.load(data_cache, allow_pickle=True)
    save = dict(
        posed_joints   = d["joints_world"][sample_idx].astype(np.float32),
        local_rot_mats = d["local_rot_mats"][sample_idx].astype(np.float32),
        global_rot_mats= d["global_rot_mats"][sample_idx].astype(np.float32),
        root_positions = d["root_positions_world"][sample_idx].astype(np.float32),
    )
    if "foot_contacts" in d:
        save["foot_contacts"] = d["foot_contacts"][sample_idx]
    np.savez(out_path, **save)
    print(f"Saved GT  {out_path}")
    print(f"  prompt: '{d['prompts'][sample_idx]}'")


def export_from_model(model_name: str, sample_idx: int, mask_mode: str,
                      mask_ratio: float, out_path: Path, data_cache: str,
                      seed: int = 42):
    """Run model inference, de-normalise, FK, write Kimodo-format NPZ."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = KimodoMotionDataset(data_cache, crop_frames=90)
    cfg_args = dict(T=ds.T, J=ds.J, C=ds.C)

    model, state = load_model(model_name, device, cfg_args)

    # Pull sample & shared mask
    sample = ds[sample_idx]
    pos_nrm = sample["pos"].unsqueeze(0).to(device)         # [1, T, J, 3]
    rot6d_gt = sample["rot6d"].unsqueeze(0).to(device)
    root_nrm = sample["root"].unsqueeze(0).to(device)

    if mask_mode == "none":
        mask = torch.zeros(1, ds.T, ds.J, dtype=torch.bool, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        _, mask, _ = make_masked_batch(pos_nrm, mask_ratio=mask_ratio,
                                        mask_mode=mask_mode, rng=g)

    # Model input: positions with masked entries zeroed
    masked_pos = pos_nrm.clone()
    masked_pos[mask] = 0.0

    with torch.no_grad():
        pred = model(masked_pos, mask)

    # De-normalise everything to physical units
    pos_mean = torch.from_numpy(ds.mean[0]).to(device)        # [J, 3]
    pos_std  = torch.from_numpy(ds.std[0]).to(device)
    root_mean = torch.from_numpy(ds.root_mean[0]).to(device)  # [3]
    root_std  = torch.from_numpy(ds.root_std[0]).to(device)

    pos_phys  = pred["pos"] * pos_std.view(1, 1, ds.J, 3) + pos_mean.view(1, 1, ds.J, 3)
    root_phys = pred["root"] * root_std.view(1, 1, 3) + root_mean.view(1, 1, 3)

    # Local rotations: 6D -> matrix
    local_rot_mats = rot6d_to_matrix(pred["rot6d"])             # [1, T, J, 3, 3]

    # Global FK: predicted rotations + predicted root -> global positions
    bone_offsets = torch.from_numpy(ds.bone_offsets).to(device)
    parents = torch.from_numpy(ds.parents).to(device)
    fk_pos, global_rot = forward_kinematics_torch(
        local_rot_mats, root_phys, bone_offsets, parents,
    )                                                            # [1, T, J, 3] / ...

    # Replace masked joint positions with FK-consistent predictions for a cleaner export.
    # For unmasked joints, we show the direct predicted position (which for unmasked
    # will equal GT since the model was given that info).
    # Final posed_joints: use FK positions (guaranteed kinematically consistent).
    posed_joints = fk_pos[0].cpu().numpy()                      # [T, J, 3]
    local_rot_np = local_rot_mats[0].cpu().numpy()
    global_rot_np = global_rot[0].cpu().numpy()
    root_np = root_phys[0].cpu().numpy()

    np.savez(
        out_path,
        posed_joints=posed_joints.astype(np.float32),
        local_rot_mats=local_rot_np.astype(np.float32),
        global_rot_mats=global_rot_np.astype(np.float32),
        root_positions=root_np.astype(np.float32),
    )

    # Report reconstruction error on masked joints (world-frame)
    if mask_mode != "none":
        gt_world = ds.raw[sample_idx]   # [T, J, 3] root-relative physical
        m = mask[0].cpu().numpy()
        err = float(np.sqrt(((posed_joints - gt_world) ** 2 * m[..., None]).sum()
                             / (m.sum() * 3 + 1e-8)))
        print(f"  masked joints L2 (world): {err*1000:.1f} mm")

    prompts = np.load(data_cache, allow_pickle=True)["prompts"]
    print(f"Saved {out_path}")
    print(f"  prompt: '{prompts[sample_idx]}'  mask_mode={mask_mode}  masked={mask.float().mean()*100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="opm_only",
                   choices=["full", "opm_only", "triangle_only", "pair_static",
                            "axial_only", "baseline"])
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--mask-mode", default="none",
                   choices=["none", "random", "joint", "time", "keyframe", "kinematic_chain"])
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="runs/kimodo_cache.npz")
    p.add_argument("--out-dir", default="runs/kimodo_export")
    p.add_argument("--export-gt", action="store_true",
                   help="Just export the ground-truth sample, no model inference.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_gt:
        out = out_dir / f"gt_sample{args.sample}.npz"
        export_from_gt(args.sample, out, args.data)
    else:
        tag = f"{args.model}_sample{args.sample}_{args.mask_mode}"
        out = out_dir / f"{tag}.npz"
        export_from_model(args.model, args.sample, args.mask_mode,
                          args.mask_ratio, out, args.data, args.seed)


if __name__ == "__main__":
    main()
