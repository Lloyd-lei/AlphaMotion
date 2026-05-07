"""Training helpers for NB1.

Only smoke_run is needed by NB1 — proves env + data + model produce a valid
loss descent in 100 steps under the **conditional flow matching (FM)** regime
(Stage 1 v2). Full training (CLI) is in 02_resources/scripts/train_runner.py.

For pedagogical clarity the loss is **single-component**: only the FM velocity
loss on the `pos` stream (rot / root / FK omitted to keep the smoke fast and
the descent signal clean). The full 4-component FM loss with FK consistency
lives in train_runner.py.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model_builder import build_from_spec


def smoke_run(
    variant_spec: dict,
    train_default: dict,
    train_subset: Subset,
    n_steps: int = 100,
    *,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[List[float], dict]:
    """Tiny FM training loop. Returns (loss_curve, info_dict).

    Per step:
        sample x_0 ~ N(0, I) and t ~ U(0, 1)  (per-stream)
        x_t = (1 - t)·x_0 + t·x_1
        v_target = x_1 - x_0
        v_pred = model(x_pos_t, x_rot_t, x_root_t, t)
        loss = MSE(v_pred.pos, v_target_pos)         (smoke: pos only)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")

    sample = train_subset[0]
    T, J, C = sample["pos"].shape
    model, _ = build_from_spec(variant_spec["model"], T=T, J=J, C=C)
    model = model.to(dev).train()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_default["optim"]["lr"]),
        betas=tuple(train_default["optim"]["betas"]),
        weight_decay=float(train_default["optim"]["weight_decay"]),
    )

    bs = min(int(train_default["batch"]["size"]), len(train_subset))
    loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                        num_workers=0, drop_last=True)

    losses: List[float] = []
    step = 0
    t0 = time.time()
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            x_pos_1  = batch["pos"].to(dev)
            x_rot_1  = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            B = x_pos_1.shape[0]

            # FM step (linear interpolation, constant target velocity)
            x_pos_0  = torch.randn_like(x_pos_1)
            x_rot_0  = torch.randn_like(x_rot_1)
            x_root_0 = torch.randn_like(x_root_1)

            t = torch.rand(B, device=dev)
            t_pos  = t.view(B, 1, 1, 1)
            t_rot  = t.view(B, 1, 1, 1)
            t_root = t.view(B, 1, 1)

            x_pos_t  = (1 - t_pos)  * x_pos_0  + t_pos  * x_pos_1
            x_rot_t  = (1 - t_rot)  * x_rot_0  + t_rot  * x_rot_1
            x_root_t = (1 - t_root) * x_root_0 + t_root * x_root_1
            v_pos_target = x_pos_1 - x_pos_0

            out = model(x_pos_t, x_rot_t, x_root_t, t)
            v_pos_pred = out["pos"]

            mse = F.mse_loss(v_pos_pred, v_pos_target)   # smoke: pos-only FM loss
            opt.zero_grad(set_to_none=True)
            mse.backward()
            opt.step()
            losses.append(float(mse.item()))
            step += 1

    info = {
        "device":       str(dev),
        "regime":       "fm_pos_only",
        "T": T, "J": J, "C": C,
        "n_steps":      len(losses),
        "wall_seconds": round(time.time() - t0, 2),
        "loss_first":   losses[0] if losses else None,
        "loss_last":    losses[-1] if losses else None,
    }
    return losses, info
