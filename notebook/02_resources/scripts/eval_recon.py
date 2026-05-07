"""Reconstruction MPJPE evaluation (FM regime, Stage-1 v2).

For FM models we cannot do a single deterministic forward — we evaluate via
**1-step denoise from t=0.95**: take ground truth + tiny noise, predict
velocity, extrapolate endpoint, compare to GT. Cheap and reasonable as a
'how well does the velocity field point home from near the manifold' probe.

For paper-grade sampling-based MPJPE you'd integrate from t=0 with N=16+
Euler steps; reserved for headline run.

Public API (unchanged):
    mpjpe_eval(ckpt_path, subset, *, batch_size, device) -> dict
    macro_mpjpe(per_family, *, n_per_family, min_n) -> float
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Subset

from _eval_common import load_eval_model, eval_loader, fm_one_step_denoise


@torch.no_grad()
def mpjpe_eval(
    ckpt_path: str,
    subset: Subset,
    *,
    batch_size: int = 32,
    device:     str = "cuda",
    t_value:    float = 0.95,    # how close to the data manifold we probe
) -> dict:
    model, _, variant = load_eval_model(ckpt_path, device=device)
    dev = next(model.parameters()).device

    base_ds = subset.dataset
    pos_mean = torch.from_numpy(base_ds.mean[0]).to(dev)
    pos_std  = torch.from_numpy(base_ds.std[0]).to(dev)
    action_labels = np.asarray([str(l) for l in base_ds.action_label])

    loader = eval_loader(subset, batch=batch_size)
    err_sum_per_joint = None
    err_count_per_joint = 0
    per_family_sum:   Dict[str, float] = {}
    per_family_count: Dict[str, int]   = {}
    n_total = 0

    for bi, batch in enumerate(loader):
        x_pos_1  = batch["pos"].to(dev)
        x_rot_1  = batch["rot6d"].to(dev)
        x_root_1 = batch["root"].to(dev)
        B, T, J, _ = x_pos_1.shape

        # 1-step denoise from t=t_value
        x_pos_hat, _, _ = fm_one_step_denoise(
            model, x_pos_1, x_rot_1, x_root_1, t_value=t_value,
        )

        # Un-normalize → MPJPE in meters
        pred = x_pos_hat * pos_std + pos_mean
        gt   = x_pos_1   * pos_std + pos_mean
        per_frame_per_joint = (pred - gt).norm(dim=-1)   # [B, T, J]

        contribution = per_frame_per_joint.sum(dim=(0, 1)).cpu().numpy()
        err_sum_per_joint = contribution.astype(np.float64) if err_sum_per_joint is None \
                            else err_sum_per_joint + contribution
        err_count_per_joint += B * T
        n_total += B

        per_clip = per_frame_per_joint.mean(dim=(1, 2)).cpu().numpy()
        for i in range(B):
            fam = action_labels[subset.indices[bi*batch_size + i]]
            per_family_sum.setdefault(fam, 0.0)
            per_family_count.setdefault(fam, 0)
            per_family_sum[fam]   += float(per_clip[i])
            per_family_count[fam] += 1

    mpjpe_per_joint = (err_sum_per_joint / err_count_per_joint).astype(np.float32)
    return {
        "variant":          variant,
        "regime":           "fm_1step_denoise",
        "t_eval":           t_value,
        "n":                n_total,
        "mpjpe_overall":    float(mpjpe_per_joint.mean()),
        "mpjpe_per_joint":  mpjpe_per_joint,
        "mpjpe_per_family": {f: per_family_sum[f]/max(1, per_family_count[f])
                              for f in per_family_sum},
        "n_per_family":     per_family_count,
    }


def macro_mpjpe(per_family: Dict[str, float], *,
                 n_per_family: Dict[str, int] | None = None,
                 min_n: int = 10) -> float:
    keys = list(per_family.keys()) if n_per_family is None else \
           [f for f in per_family if n_per_family.get(f, 0) >= min_n]
    return float(np.mean([per_family[f] for f in keys])) if keys else float("nan")
