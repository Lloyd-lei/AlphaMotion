"""Thin adapter to the one FK convention shared by both candidate architectures.

The caller supplies base.pilot_data.fk and common.rot6d_to_matrix. This module
contains no second FK implementation and never treats raw model hpos as XYZ.
"""
from __future__ import annotations
import torch


def global_rot6_decoder(parents,offsets,*,fk_fn,rotation_matrix_fn,prismatic_joints=()):
    if prismatic_joints:raise ValueError('global rotation/fixed-offset decoder cannot represent prismatic joints')
    J=len(parents)
    if J<1 or parents[0]!=-1 or any(not 0<=p<j for j,p in enumerate(parents[1:],1)):raise ValueError('parent-first single root required')
    def decode(rot6):
        if rot6.ndim!=3 or rot6.shape[1:]!=(J,6):raise ValueError('continuous global rot6 state [T,J,6]')
        # Both supplied functions use the same Gram-Schmidt/global convention.
        return fk_fn(rot6,parents,offsets),rotation_matrix_fn(rot6.float())
    return decode


def prediction_from_model_output(outputs,*,parents,offsets,fk_fn,rotation_matrix_fn):
    """Ignore raw auxiliary head for Cartesian metrics; derive all XYZ by shared FK."""
    from eval_engine import PosePrediction
    rot6=outputs[0]
    with torch.no_grad():
        p=fk_fn(rot6.float(),parents,offsets);R=rotation_matrix_fn(rot6.float())
    return PosePrediction(p.detach().cpu().numpy(),R.detach().cpu().numpy())
