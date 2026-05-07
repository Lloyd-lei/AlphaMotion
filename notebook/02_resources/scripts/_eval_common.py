"""Shared utilities for the eval_*.py scripts.

  - load_eval_model(ckpt_path) — build correct variant + load weights, cuda eval
  - eval_loader(subset, batch)  — DataLoader with sensible defaults for eval
  - REPO / CFG_DIR / VARS / DATA — convenience handles
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset


REPO       = Path(__file__).resolve().parent.parent.parent.parent
CFG_DIR    = REPO / "notebook/01_resources/configs"
SCRIPTS_01 = REPO / "notebook/01_resources/scripts"

# Make 01_resources/scripts importable for any eval script that imports this.
if str(SCRIPTS_01) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_01))

from model_builder import build_from_spec   # noqa: E402

VARS          = json.loads((CFG_DIR / "variants.json").read_text())
DATA          = json.loads((CFG_DIR / "data.json").read_text())
ENV           = json.loads((CFG_DIR / "env.json").read_text())
TRAIN_DEFAULT = json.loads((CFG_DIR / "train_default.json").read_text())


def load_eval_model(ckpt_path: str | Path, *, device: str = "cuda") -> Tuple[torch.nn.Module, object, str]:
    """Load a trained model from `ckpt_path` (parent dir name = variant).

    Strips the `_orig_mod.` prefix that `torch.compile` adds to state-dict keys
    so checkpoints saved from a compiled model load into an uncompiled model.

    Returns (model.eval(), cfg_obj, variant_name).
    """
    ckpt_path = Path(ckpt_path)
    variant   = ckpt_path.parent.name
    if variant not in VARS:
        raise ValueError(f"variant '{variant}' not in configs/variants.json")
    spec = VARS[variant]["model"]
    T    = DATA["sample"]["T"]
    J    = DATA["skeleton"]["n_joints"]
    C    = 3

    model, cfg_obj = build_from_spec(spec, T=T, J=J, C=C)

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")

    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sd = ck["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    return model, cfg_obj, variant


def eval_loader(subset: Subset, batch: int = 32, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        subset, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


# ---------- FM evaluation helpers (Stage-1 v2) ------------------------------

@torch.no_grad()
def fm_forward_at_t(
    model, x_pos_1, x_rot_1, x_root_1, *, t_value: float = 1.0, mask=None,
):
    """Forward the FM model at a single t value with the GIVEN clean data.

    For t_value=1.0, x_t equals the clean data (since x_1 = data, no noise).
    Use this to extract trunk activations (pair tensor / MSA hidden) at the
    'data-state' end of the velocity-field schedule.

    Returns the velocity-field output dict.
    """
    B = x_pos_1.shape[0]
    dev = x_pos_1.device
    t = torch.full((B,), float(t_value), device=dev)
    return model(x_pos_1, x_rot_1, x_root_1, t, mask=mask)


@torch.no_grad()
def fm_one_step_denoise(
    model, x_pos_1, x_rot_1, x_root_1, *, t_value: float = 0.95,
):
    """One-step denoise from a slightly-noisy state near the data manifold.

    Build x_t at t=t_value by mixing in a tiny amount of noise:
        x_t = (1 - t) * noise + t * x_1     (so for t=0.95, mostly x_1)
    Predict v, extrapolate endpoint x̂_1 = x_t + (1 - t) * v.

    Returns (x_pos_hat, x_rot_hat, x_root_hat) — all in normalized space.
    Used as a cheap reconstruction proxy in smoke / sanity checks.
    """
    B = x_pos_1.shape[0]
    dev = x_pos_1.device
    t = torch.full((B,), float(t_value), device=dev)
    t_pos  = t.view(B, 1, 1, 1)
    t_rot  = t.view(B, 1, 1, 1)
    t_root = t.view(B, 1, 1)

    x_pos_0  = torch.randn_like(x_pos_1)
    x_rot_0  = torch.randn_like(x_rot_1)
    x_root_0 = torch.randn_like(x_root_1)

    x_pos_t  = (1 - t_pos)  * x_pos_0  + t_pos  * x_pos_1
    x_rot_t  = (1 - t_rot)  * x_rot_0  + t_rot  * x_rot_1
    x_root_t = (1 - t_root) * x_root_0 + t_root * x_root_1

    out = model(x_pos_t, x_rot_t, x_root_t, t)
    x_pos_hat  = x_pos_t  + (1 - t_pos)  * out["pos"]
    x_rot_hat  = x_rot_t  + (1 - t_rot)  * out["rot6d"]
    x_root_hat = x_root_t + (1 - t_root) * out["root"]
    return x_pos_hat, x_rot_hat, x_root_hat


@torch.no_grad()
def fm_sample(
    model, B, T, J, *,
    n_steps: int = 16, device: str = "cuda",
):
    """Unconditional FM sampling via Euler integration from t=0 to t=1.

    Returns (x_pos_1, x_rot_1, x_root_1) in normalized space.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    x_pos  = torch.randn(B, T, J, 3, device=dev)
    x_rot  = torch.randn(B, T, J, 6, device=dev)
    x_root = torch.randn(B, T, 3,    device=dev)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=dev)
        out = model(x_pos, x_rot, x_root, t)
        x_pos  = x_pos  + dt * out["pos"]
        x_rot  = x_rot  + dt * out["rot6d"]
        x_root = x_root + dt * out["root"]
    return x_pos, x_rot, x_root


@torch.no_grad()
def fm_sample_conditional_inpaint(
    model, x_pos_1, x_rot_1, x_root_1, mask,
    *, n_steps: int = 16,
):
    """Inpainting sampling: visible joints stay clean, masked joints integrate from noise.

    `mask`: [B, T, J] bool — True at masked (unknown) positions to be filled.
    Returns the same 3-tuple as fm_sample, with visible positions equal to GT.
    """
    dev = x_pos_1.device
    B, T, J, _ = x_pos_1.shape
    # Initialize masked positions with noise; visible with GT
    x_pos  = torch.where(mask.unsqueeze(-1), torch.randn_like(x_pos_1),  x_pos_1)
    x_rot  = torch.where(mask.unsqueeze(-1), torch.randn_like(x_rot_1),  x_rot_1)
    x_root = x_root_1.clone()                                             # root not masked

    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=dev)
        out = model(x_pos, x_rot, x_root, t, mask=mask)
        # Update only the masked positions
        new_pos = x_pos + dt * out["pos"]
        new_rot = x_rot + dt * out["rot6d"]
        x_pos  = torch.where(mask.unsqueeze(-1), new_pos, x_pos_1)
        x_rot  = torch.where(mask.unsqueeze(-1), new_rot, x_rot_1)
    return x_pos, x_rot, x_root
