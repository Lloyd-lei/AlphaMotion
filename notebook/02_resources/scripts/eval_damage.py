"""Joint-mask damage robustness (FM regime, Stage-1 v2).

Uses **conditional inpainting sampling**: visible joints stay clean GT,
masked joints integrate from noise via the velocity field. Returns MPJPE
on the masked joints only — measures "given partial body state, can the
trunk reconstruct the rest via FM?".

Public API (unchanged):
    joint_mask_eval(ckpt_path, subset, *, k_values, n_per_k, batch_size, seed)
        -> {variant, k_values, mpjpe_per_k, std_per_k, n_eval_clips}
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Subset

from _eval_common import load_eval_model, eval_loader, fm_sample_conditional_inpaint


@torch.no_grad()
def joint_mask_eval(
    ckpt_path: str,
    subset: Subset,
    *,
    k_values:   Sequence[int] = (0, 1, 2, 3),
    n_per_k:    int = 3,
    n_steps:    int = 16,           # FM sampling steps for inpainting
    batch_size: int = 32,
    seed:       int = 0,
    device:     str = "cuda",
) -> Dict:
    model, _, variant = load_eval_model(ckpt_path, device=device)
    dev = next(model.parameters()).device

    base_ds = subset.dataset
    pos_mean = torch.from_numpy(base_ds.mean[0]).to(dev)
    pos_std  = torch.from_numpy(base_ds.std[0]).to(dev)

    rng = np.random.default_rng(seed)
    cached = []
    for batch in eval_loader(subset, batch=batch_size):
        cached.append((batch["pos"].to(dev),
                       batch["rot6d"].to(dev),
                       batch["root"].to(dev)))
    if not cached:
        raise RuntimeError("empty subset")
    J = cached[0][0].shape[2]

    mpjpe_per_k, std_per_k = [], []
    for K in k_values:
        per_run = []
        for _ in range(n_per_k):
            errs = []
            for x_pos_1, x_rot_1, x_root_1 in cached:
                B, T, J_, _ = x_pos_1.shape
                if K == 0:
                    # No masking — measure inpainting fidelity at "full visibility":
                    # mask is empty (all visible), so FM sample returns x_1 unchanged.
                    # Use 1-step denoise as a proxy MPJPE.
                    from _eval_common import fm_one_step_denoise
                    pred_pos, _, _ = fm_one_step_denoise(
                        model, x_pos_1, x_rot_1, x_root_1, t_value=0.95,
                    )
                    err_mat = ((pred_pos - x_pos_1) * pos_std).norm(dim=-1)  # [B,T,J]
                    errs.append(err_mat.mean().item())
                else:
                    bad = np.zeros((B, J_), dtype=bool)
                    for b in range(B):
                        bad[b, rng.choice(J_, size=K, replace=False)] = True
                    bad_t  = torch.from_numpy(bad).to(dev)
                    bad_bt = bad_t.unsqueeze(1).expand(B, T, J_)             # [B,T,J]

                    # Conditional inpainting via FM
                    pred_pos, _, _ = fm_sample_conditional_inpaint(
                        model, x_pos_1, x_rot_1, x_root_1, mask=bad_bt,
                        n_steps=n_steps,
                    )

                    # MPJPE on masked positions only, in meters
                    pred_un = pred_pos * pos_std + pos_mean
                    gt_un   = x_pos_1  * pos_std + pos_mean
                    per_frame_per_joint = (pred_un - gt_un).norm(dim=-1)     # [B,T,J]
                    e = per_frame_per_joint[bad_bt].mean().item()
                    errs.append(e)
            per_run.append(float(np.mean(errs)))
        mpjpe_per_k.append(float(np.mean(per_run)))
        std_per_k.append(float(np.std(per_run)))

    return {"variant": variant, "k_values": list(k_values),
            "mpjpe_per_k": mpjpe_per_k, "std_per_k": std_per_k,
            "n_per_k": n_per_k, "n_sample_steps": n_steps,
            "n_eval_clips": int(sum(c[0].shape[0] for c in cached))}
