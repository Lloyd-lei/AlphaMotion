"""Linear-blend skinning for SOMA-77 skeleton.

Loads `skin_standard.npz` (copied from kimodo assets) and applies LBS to a
sequence of (joint global rotations + joint world positions) to produce a
mesh per frame.

Standalone — does not import kimodo. Pure numpy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def load_skin(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """Load SOMA-77 skin assets.

    Returns dict with:
      bind_vertices       [V=18056, 3]     rest-pose mesh vertices
      faces               [F=36108, 3]     triangle indices
      bind_rig_transform  [J=77, 4, 4]     rest-pose joint global transforms
      bind_rig_inv        [J=77, 4, 4]     pre-computed inverse
      lbs_indices         [V, 8]           per-vertex bound joint indices
      lbs_weights         [V, 8]           per-vertex bound joint weights
      rig_joint_names     [J,]             joint names (debug only)
    """
    d = np.load(str(npz_path), allow_pickle=True)
    bind_rig = d["bind_rig_transform"].astype(np.float32)
    return {
        "bind_vertices":      d["bind_vertices"].astype(np.float32),
        "faces":              d["faces"].astype(np.int64),
        "bind_rig_transform": bind_rig,
        "bind_rig_inv":       np.linalg.inv(bind_rig).astype(np.float32),
        "lbs_indices":        d["lbs_indices"].astype(np.int64),
        "lbs_weights":        d["lbs_weights"].astype(np.float32),
        "rig_joint_names":    d["rig_joint_names"],
    }


def lbs_apply(
    skin: Dict[str, np.ndarray],
    joint_global_rot: np.ndarray,        # [J, 3, 3]
    joint_world_pos:  np.ndarray,        # [J, 3]
) -> np.ndarray:                         # [V, 3]
    """Skin the mesh under one frame of joint state.

    For each joint j: M_j = T_current_j @ T_bind_j^{-1}
        T_current_j = [R_global[j] | t_world[j]; 0 1]
    For each vertex v: v' = sum_k w[v,k] · M_{idx[v,k]} · v_bind_h

    Args:
        joint_global_rot: per-joint global rotation matrix in world frame.
        joint_world_pos:  per-joint world position.
    """
    V = skin["bind_vertices"].shape[0]
    J = joint_global_rot.shape[0]

    # Current global transform per joint: [J, 4, 4]
    cur_T = np.zeros((J, 4, 4), dtype=np.float32)
    cur_T[:, :3, :3] = joint_global_rot
    cur_T[:, :3,  3] = joint_world_pos
    cur_T[:,  3,  3] = 1.0

    # Skinning matrix: cur @ bind^{-1}      [J, 4, 4]
    skin_T = cur_T @ skin["bind_rig_inv"]

    # Bind vertices in homogeneous coords: [V, 4]
    bv_h = np.concatenate(
        [skin["bind_vertices"], np.ones((V, 1), dtype=np.float32)], axis=1
    )

    # For each vertex: gather its 8 joint matrices, transform bind vertex, weight, sum
    gathered    = skin_T[skin["lbs_indices"]]                   # [V, 8, 4, 4]
    transformed = np.einsum("vkij,vj->vki", gathered, bv_h)     # [V, 8, 4]
    weighted    = (transformed * skin["lbs_weights"][:, :, None]).sum(axis=1)
    return weighted[:, :3]


def lbs_sequence(
    skin: Dict[str, np.ndarray],
    global_rot_mats: np.ndarray,         # [T, J, 3, 3]
    joints_world:    np.ndarray,         # [T, J, 3]
) -> np.ndarray:                         # [T, V, 3]
    """Apply LBS to every frame in a sequence."""
    return np.stack([
        lbs_apply(skin, global_rot_mats[t], joints_world[t])
        for t in range(global_rot_mats.shape[0])
    ])
