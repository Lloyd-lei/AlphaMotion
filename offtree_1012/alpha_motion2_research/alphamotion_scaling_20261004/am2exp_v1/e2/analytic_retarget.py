"""AN: analytic opposition retarget Adroit -> Franka gripper (reference arm, zero training).

c_F = clip(c_A, 0, 7.8 cm) with c_A the FROZEN opposition descriptor of the source hand pose.
Fingers are set symmetrically: q_L = q_R = (c_F + 0.2)/2 cm (in [0, 4] cm), slide_normalized = q/4.
Wrist/arm SE(3) is not mapped (no hand-to-arm calibration in this round; see final_plan.md c7).
"""
from __future__ import annotations
import numpy as np
import torch

from . import opposition as op

SLIDE_UPPER_CM = 4.0


def analytic_clearance(P, G, variant=None, tau_cm=None):
    """P [B,25,3] cm, G [B,25,3,3] (torch) -> dict(c_A, c_F, q_left, q_right, slide_normalized) in cm."""
    with torch.no_grad():
        desc = op.adroit_descriptor(P, G, variant=variant, tau_cm=tau_cm)
    c_A = desc['c_A']
    c_F = c_A.clamp(0.0, op.FRANKA_MAX_CLEARANCE_CM)
    q = ((c_F + op.FRANKA_CLEARANCE_OFFSET_CM) / 2).clamp(0.0, SLIDE_UPPER_CM)
    return {'c_A': c_A, 'c_F': c_F, 'q_left': q, 'q_right': q, 'slide_normalized': q / SLIDE_UPPER_CM, 'variant': desc['variant']}


def analytic_from_pose9(pose9, reach_cm, variant=None, tau_cm=None, chunk=4096):
    """numpy/torch pose9 [N,25,9] -> c_F [N] cm (numpy)."""
    from ._sealed_math import rot6d_to_matrix
    x = torch.as_tensor(np.asarray(pose9), dtype=torch.float32)
    out = []
    for s in range(0, len(x), chunk):
        xb = x[s:s + chunk]
        G = rot6d_to_matrix(xb[..., :6])
        P = xb[..., 6:] * float(reach_cm)
        out.append(analytic_clearance(P, G, variant, tau_cm)['c_F'].numpy())
    return np.concatenate(out) if out else np.zeros(0)
