"""
Stage 1 training: MotionFormer / baseline predict joint positions + local
rotations + root position simultaneously, then FK is applied during the loss
computation so rotations and positions stay kinematically consistent.

Loss = λ_pos · MSE(pos, gt_pos)  on masked joints
     + λ_rot · MSE(rot6d, gt_rot6d)  on masked joints
     + λ_root · MSE(root, gt_root)
     + λ_fk · MSE(FK(pred_rot, pred_root), gt_pos)  over all joints

This makes the output directly loadable into Kimodo's viser demo via the
recovered global_rot_mats + posed_joints.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data import (
    KimodoMotionDataset, MixedMotionDataset,
    make_masked_batch, mixed_mode_masking,
)
from baseline import BaselineConfig, BaselineTransformer
from motionformer import MotionFormerConfig, MotionFormer
from soma_skeleton import (
    rot6d_to_matrix, forward_kinematics_torch, build_parent_index,
)


MF_VARIANTS = {
    "full":          dict(use_pair=True,  use_opm=True,  use_triangle=True),
    "opm_only":      dict(use_pair=True,  use_opm=True,  use_triangle=False),
    "triangle_only": dict(use_pair=True,  use_opm=False, use_triangle=True),
    "pair_static":   dict(use_pair=True,  use_opm=False, use_triangle=False),
    "axial_only":    dict(use_pair=False, use_opm=False, use_triangle=False),
}


def build_model(name: str, T: int, J: int, C: int):
    if name == "baseline":
        cfg = BaselineConfig(T=T, J=J, C=C, hidden=128, depth=12, heads=4, ffn_mult=4)
        return BaselineTransformer(cfg), cfg
    if name in MF_VARIANTS:
        cfg = MotionFormerConfig(
            T=T, J=J, C=C,
            hidden=128, pair_hidden=32, depth=6, heads=4, pair_heads=4,
            opm_chunk=16, tri_hidden=16,
            **MF_VARIANTS[name],
        )
        return MotionFormer(cfg), cfg
    if name == "motionformer":
        return build_model("full", T=T, J=J, C=C)
    raise ValueError(name)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred, target: [B, T, J, D]; mask: [B, T, J]
    diff = (pred - target) ** 2
    w = mask.unsqueeze(-1).float()
    return (diff * w).sum() / (w.sum() * pred.size(-1) + 1e-8)


def geodesic_rotation_loss(
    R_pred: torch.Tensor,  # [B, T, J, 3, 3]
    R_gt:   torch.Tensor,
    mask:   torch.Tensor,  # [B, T, J]
) -> torch.Tensor:
    """Squared geodesic angle between two rotation matrices, masked-averaged.

    angle = acos((trace(R_pred @ R_gt^T) - 1) / 2), numerically clamped.
    The returned loss is in (radians)^2.
    """
    R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))     # [B, T, J, 3, 3]
    tr = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)             # [B, T, J]
    cos_angle = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos_angle)                              # radians
    sq = angle ** 2
    w = mask.float()
    return (sq * w).sum() / (w.sum() + 1e-8)


def compute_losses(
    pred: dict,           # {pos, rot6d, root} predicted (normalised space)
    gt: dict,             # {pos, rot6d, root, local_rot_mats}
    mask: torch.Tensor,   # [B, T, J]
    pos_mean: torch.Tensor, pos_std: torch.Tensor,
    root_mean: torch.Tensor, root_std: torch.Tensor,
    bone_offsets: torch.Tensor, parents: torch.Tensor,
    lambdas: dict,
):
    """Return dict of scalar losses + the total."""
    out = {}
    out["pos"] = masked_mse(pred["pos"], gt["pos"], mask)

    # Rotation: geodesic distance on SO(3) (radians squared). Far more
    # appropriate for rotation matrices than MSE on 6D coordinates.
    pred_rot_mats = rot6d_to_matrix(pred["rot6d"])
    out["rot"] = geodesic_rotation_loss(pred_rot_mats, gt["local_rot_mats"], mask)

    # Root: simple MSE per frame
    out["root"] = torch.mean((pred["root"] - gt["root"]) ** 2)

    # FK consistency: predicted rotations + predicted root should yield
    # the ground-truth (de-normalised) joint positions.
    root_phys = pred["root"] * root_std.view(1, 1, 3) + root_mean.view(1, 1, 3)
    fk_pos, _ = forward_kinematics_torch(
        pred_rot_mats, root_phys, bone_offsets, parents,
    )
    gt_pos_world = (
        gt["pos"] * pos_std.view(1, 1, pos_std.shape[0], 3)
        + pos_mean.view(1, 1, pos_std.shape[0], 3)
    )
    out["fk"] = torch.mean((fk_pos - gt_pos_world) ** 2)

    total = (
        lambdas["pos"]  * out["pos"]
        + lambdas["rot"]  * out["rot"]
        + lambdas["root"] * out["root"]
        + lambdas["fk"]   * out["fk"]
    )
    out["total"] = total
    return out


def train(
    model_name: str,
    data_cache: str = "runs/kimodo_cache.npz",
    mixed_data: str | None = None,
    splits_file: str | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    warmup_epochs: int = 0,
    mask_ratio: float = 0.3,
    mask_mode: str = "mixed",
    seed: int = 0,
    outdir: str = "runs",
    T_crop: int | None = 90,
    run_tag: str | None = None,
    lambdas: dict | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lambdas is None:
        lambdas = dict(pos=1.0, rot=5.0, root=1.0, fk=0.5)

    def collate(batch_list):
        out = {}
        for k in batch_list[0]:
            if isinstance(batch_list[0][k], int):
                out[k] = torch.tensor([b[k] for b in batch_list])
            else:
                out[k] = torch.stack([b[k] for b in batch_list])
        return out

    # ------------------------------------------------------------
    # Dataset selection: mixed (AMASS+Kimodo+HY) vs single-source (Kimodo only)
    # ------------------------------------------------------------
    val_loaders = {}  # name -> DataLoader
    if mixed_data is not None and splits_file is not None:
        import json
        ds = MixedMotionDataset(mixed_data, crop_frames=T_crop)
        with open(splits_file) as f:
            splits = json.load(f)
        from torch.utils.data import Subset
        ds_train = Subset(ds, splits["train"])
        ds_val_id = Subset(ds, splits["val_id"])
        ds_val_subject = Subset(ds, splits["val_subject"])
        ds_val_source = Subset(ds, splits["val_source"])
        print(f"Mixed: train={len(ds_train)}  val_id={len(ds_val_id)}  "
              f"val_subject={len(ds_val_subject)}  val_source={len(ds_val_source)}")
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, pin_memory=True,
                              collate_fn=collate)
        for name, subset in [("val_id", ds_val_id), ("val_subject", ds_val_subject),
                              ("val_source", ds_val_source)]:
            if len(subset) > 0:
                val_loaders[name] = DataLoader(subset, batch_size=batch_size,
                                                shuffle=False, num_workers=2,
                                                pin_memory=True, collate_fn=collate)
    else:
        ds = KimodoMotionDataset(data_cache, crop_frames=T_crop)
        N = len(ds)
        N_val = max(16, N // 5)
        N_train = N - N_val
        ds_train, ds_val = random_split(
            ds, [N_train, N_val], generator=torch.Generator().manual_seed(seed)
        )
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, pin_memory=True,
                              collate_fn=collate)
        val_loaders["val"] = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                         num_workers=2, pin_memory=True,
                                         collate_fn=collate)

    model, _ = build_model(model_name, T=ds.T, J=ds.J, C=ds.C)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Constants to GPU
    pos_mean = torch.from_numpy(ds.mean[0]).to(device)       # [J, 3]
    pos_std  = torch.from_numpy(ds.std[0]).to(device)
    root_mean = torch.from_numpy(ds.root_mean[0]).to(device)  # [3]
    root_std  = torch.from_numpy(ds.root_std[0]).to(device)
    bone_offsets = torch.from_numpy(ds.bone_offsets).to(device)
    parents = torch.from_numpy(ds.parents).to(device)

    print(f"[{model_name}] {n_params / 1e6:.2f}M params  "
          f"train={len(dl_train.dataset)}  val_splits={list(val_loaders.keys())}  "
          f"T={ds.T} J={ds.J}  lambdas={lambdas}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1 / max(warmup_epochs, 1), total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(epochs - warmup_epochs, 1),
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    run_dir = model_name if run_tag is None else f"{model_name}__{run_tag}"
    out_root = Path(outdir) / run_dir
    out_root.mkdir(parents=True, exist_ok=True)

    def sample_mask(batch):
        if mask_mode == "mixed":
            _, m, _ = mixed_mode_masking(batch["pos"], mask_ratio=mask_ratio)
        else:
            _, m, _ = make_masked_batch(batch["pos"], mask_ratio=mask_ratio, mask_mode=mask_mode)
        return m

    def mask_input(batch, mask):
        masked_pos = batch["pos"].clone()
        masked_pos[mask] = 0.0
        return masked_pos

    history = []
    start = time.time()
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tally = {k: 0.0 for k in ("total", "pos", "rot", "root", "fk")}
        n_batches = 0
        for batch in dl_train:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            mask = sample_mask(batch)
            masked_pos = mask_input(batch, mask)
            pred = model(masked_pos, mask)
            losses = compute_losses(pred, batch, mask,
                                     pos_mean, pos_std, root_mean, root_std,
                                     bone_offsets, parents, lambdas)
            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            for k in tally:
                tally[k] += float(losses[k].item())
            n_batches += 1
            step += 1
        sched.step()
        tr = {k: v / n_batches for k, v in tally.items()}

        # Validation: evaluate every val split, report separately.
        model.eval()
        val = {}
        with torch.no_grad():
            for split_name, dl in val_loaders.items():
                v_tally = {k: 0.0 for k in ("total", "pos", "rot", "root", "fk")}
                n_v = 0
                for batch in dl:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    mask = sample_mask(batch)
                    masked_pos = mask_input(batch, mask)
                    pred = model(masked_pos, mask)
                    losses = compute_losses(pred, batch, mask,
                                             pos_mean, pos_std, root_mean, root_std,
                                             bone_offsets, parents, lambdas)
                    for k in v_tally:
                        v_tally[k] += float(losses[k].item())
                    n_v += 1
                if n_v > 0:
                    val[split_name] = {k: v / n_v for k, v in v_tally.items()}

        entry = dict(epoch=epoch, step=step, train=tr, val=val)
        history.append(entry)
        elapsed = time.time() - start
        val_str_parts = []
        for split_name, v in val.items():
            val_str_parts.append(
                f"{split_name}[pos={v['pos']:.3f} rot={v['rot']:.3f} fk={v['fk']:.3f}]"
            )
        val_str = "  ".join(val_str_parts)
        print(
            f"[{model_name}] ep {epoch:3d}/{epochs}  "
            f"train[tot={tr['total']:.3f} pos={tr['pos']:.3f} rot={tr['rot']:.3f} fk={tr['fk']:.3f}]  "
            f"{val_str}  t={elapsed:.0f}s",
            flush=True,
        )

    # Save
    with open(out_root / "history.json", "w") as f:
        json.dump({"history": history, "n_params": n_params,
                   "model_name": model_name, "lambdas": lambdas}, f, indent=2)
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "n_params": n_params,
        "model_name": model_name,
        "lambdas": lambdas,
        "bone_offsets": ds.bone_offsets,
        "pos_mean": ds.mean, "pos_std": ds.std,
        "root_mean": ds.root_mean, "root_std": ds.root_std,
    }, out_root / "final.pt")
    print(f"\n[{model_name}] DONE in {time.time() - start:.0f}s  saved to {out_root}")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["baseline", "motionformer", "full", "opm_only", "triangle_only",
                 "pair_static", "axial_only"],
        required=True,
    )
    p.add_argument("--data", default="runs/kimodo_cache.npz")
    p.add_argument("--mixed-data", default=None,
                   help="Path to mixed_soma77.npz for multi-source training. "
                        "When set together with --splits, overrides --data.")
    p.add_argument("--splits", default=None,
                   help="Path to splits.json (train/val_id/val_subject/val_source).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument(
        "--mask-mode",
        default="mixed",
        choices=["random", "joint", "time", "keyframe", "kinematic_chain", "mixed"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs")
    p.add_argument("--run-tag", default=None)
    args = p.parse_args()

    train(
        model_name=args.model,
        data_cache=args.data,
        mixed_data=args.mixed_data,
        splits_file=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        seed=args.seed,
        outdir=args.outdir,
        run_tag=args.run_tag,
    )
