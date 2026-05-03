"""Evaluate a trained MotionFormer on a LARGE held-out AMASS slice that was
never seen during training.

The mixed dataset only took 1500 clips per AMASS subset (10315 total). The
other ~32k AMASS clips live in the per-subset npz files under
experiment/dataset/amass/*_soma77.npz and are pristine — use them for a
statistically meaningful OOD evaluation.

Usage:
    python eval_on_amass_heldout.py \
        --checkpoint runs/opm_only__mixed_full30/final.pt \
        --subsets hdm05 kit totalcapture \
        --max-per-subset 500 \
        --model opm_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from baseline import BaselineConfig, BaselineTransformer
from motionformer import MotionFormerConfig, MotionFormer
from soma_skeleton import (
    build_parent_index, compute_bone_offsets, matrix_to_rot6d,
    rot6d_to_matrix, forward_kinematics_torch,
)
from train import MF_VARIANTS, compute_losses, geodesic_rotation_loss, masked_mse
from data import make_masked_batch


class AmassClipDataset(Dataset):
    """Load AMASS clips from multiple per-subset npz files with optional
    uniform sub-sampling. Returns the same dict schema as MixedMotionDataset."""

    def __init__(self, subset_files: list[Path], max_per_subset: int | None,
                 pos_mean, pos_std, root_mean, root_std,
                 skip_indices_by_subset: dict | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        all_j, all_l, all_g, all_r, all_meta = [], [], [], [], []
        for fp in subset_files:
            d = np.load(fp, allow_pickle=True)
            N = d["joints_world"].shape[0]
            skip = set(skip_indices_by_subset.get(fp.stem.replace("_soma77", ""), [])) \
                   if skip_indices_by_subset else set()
            eligible = np.array([i for i in range(N) if i not in skip])
            if max_per_subset is not None and len(eligible) > max_per_subset:
                pick = rng.choice(eligible, max_per_subset, replace=False)
            else:
                pick = eligible
            all_j.append(d["joints_world"][pick])
            all_l.append(d["local_rot_mats"][pick])
            all_g.append(d["global_rot_mats"][pick])
            all_r.append(d["root_positions"][pick])
            meta = d["meta"][pick]
            all_meta.extend(meta.tolist())
            print(f"  {fp.stem}: took {len(pick)} / {N}")

        joints_world = np.concatenate(all_j, axis=0).astype(np.float32)
        local_rot    = np.concatenate(all_l, axis=0).astype(np.float32)
        global_rot   = np.concatenate(all_g, axis=0).astype(np.float32)
        root_world   = np.concatenate(all_r, axis=0).astype(np.float32)

        # Apply the training-time centering (subtract per-sample root-at-t0)
        root_t0 = root_world[:, :1, :]
        joints_rel = joints_world - root_t0[:, :, None, :]
        root_rel = root_world - root_t0

        # Normalise with the training dataset stats (very important — do NOT
        # re-compute, or we'd be giving the model an unfair leg-up).
        self.pos_tensor = torch.from_numpy(
            (joints_rel - pos_mean) / pos_std
        ).float()
        self.root_tensor = torch.from_numpy(
            (root_rel - root_mean) / root_std
        ).float()
        rot_t = torch.from_numpy(local_rot).float()
        self.rot_tensor = matrix_to_rot6d(rot_t)
        self.rotmat_tensor = rot_t
        self.meta = all_meta
        self.N, self.T, self.J, self.C = joints_rel.shape

    def __len__(self): return self.N

    def __getitem__(self, idx):
        return {
            "pos":            self.pos_tensor[idx],
            "rot6d":          self.rot_tensor[idx],
            "root":           self.root_tensor[idx],
            "local_rot_mats": self.rotmat_tensor[idx],
        }


def build_eval_model(model_name: str, state, cfg_args):
    if model_name == "baseline":
        cfg = BaselineConfig(**cfg_args, hidden=128, depth=12, heads=4, ffn_mult=4)
        model = BaselineTransformer(cfg)
    else:
        variant_kw = MF_VARIANTS[model_name]
        cfg = MotionFormerConfig(**cfg_args, hidden=128, pair_hidden=32, depth=6,
                                  heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16,
                                  **variant_kw)
        model = MotionFormer(cfg)
    model.load_state_dict(state["model_state"])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True,
                   choices=list(MF_VARIANTS.keys()) + ["baseline"])
    p.add_argument("--amass-dir",
                   default="/home/arenalabs/Desktop/\"be water, robot\"/experiment/dataset/amass")
    p.add_argument("--subsets", nargs="+", default=["hdm05", "kit", "totalcapture"],
                   help="Which AMASS subset stems to pull extra samples from.")
    p.add_argument("--max-per-subset", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--mask-mode", default="mixed")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    pos_mean = state["pos_mean"][0]    # [J, 3]
    pos_std  = state["pos_std"][0]
    root_mean = state["root_mean"][0]  # [3]
    root_std  = state["root_std"][0]
    bone_offsets_np = state["bone_offsets"]

    T, J, C = 90, 77, 3
    cfg_args = dict(T=T, J=J, C=C)
    model = build_eval_model(args.model, state, cfg_args).to(device).eval()

    amass_dir = Path(args.amass_dir)
    subset_files = [amass_dir / f"{s}_soma77.npz" for s in args.subsets]
    print("Loading held-out AMASS:")
    ds = AmassClipDataset(subset_files, args.max_per_subset,
                          pos_mean, pos_std, root_mean, root_std)
    print(f"  TOTAL: {len(ds)} clips from {len(args.subsets)} subset(s)")

    def collate(batch_list):
        return {k: torch.stack([b[k] for b in batch_list]) for k in batch_list[0]}

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, pin_memory=True, collate_fn=collate)

    pm = torch.from_numpy(pos_mean).to(device)
    ps = torch.from_numpy(pos_std).to(device)
    rm = torch.from_numpy(root_mean).to(device)
    rs = torch.from_numpy(root_std).to(device)
    bo = torch.from_numpy(bone_offsets_np).to(device)
    parents = torch.from_numpy(np.array(build_parent_index(), dtype=np.int64)).to(device)

    lambdas = state.get("lambdas", dict(pos=1.0, rot=5.0, root=1.0, fk=0.5))

    total = {k: 0.0 for k in ("pos", "rot", "root", "fk", "total")}
    n = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            from data import mixed_mode_masking
            if args.mask_mode == "mixed":
                _, mask, _ = mixed_mode_masking(batch["pos"], mask_ratio=args.mask_ratio)
            else:
                _, mask, _ = make_masked_batch(batch["pos"],
                                                 mask_ratio=args.mask_ratio,
                                                 mask_mode=args.mask_mode)
            masked_pos = batch["pos"].clone(); masked_pos[mask] = 0.0
            pred = model(masked_pos, mask)
            losses = compute_losses(pred, batch, mask, pm, ps, rm, rs, bo, parents, lambdas)
            for k in total:
                total[k] += float(losses[k].item())
            n += 1

    res = {k: v / n for k, v in total.items()}
    res["rot_degrees"] = (res["rot"] ** 0.5) * (180.0 / 3.14159265)
    res["n_clips"] = len(ds)
    res["n_batches"] = n
    res["checkpoint"] = str(args.checkpoint)
    res["subsets"] = args.subsets

    print(f"\n=== HELD-OUT AMASS eval ({len(ds)} clips) ===")
    print(f"  pos={res['pos']:.4f}  rot={res['rot']:.4f} rad²  "
          f"(≈{res['rot_degrees']:.1f}°)  fk={res['fk']:.4f}  root={res['root']:.4f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
