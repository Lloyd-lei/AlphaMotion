"""Convert AMASS SMPL+H .npz files into the canonical SOMA77 format used by
our MotionFormer training pipeline.

AMASS format (SMPL+H G, per file):
    poses            [T, 156]  axis-angle for 52 joints (22 body + 30 hand)
    trans            [T, 3]    root translation
    betas            [16]      body shape
    gender           'neutral' / 'male' / 'female'
    mocap_framerate  float     (usually 60 or 120)

SOMA77 format (our training target, matches Kimodo):
    joints_world     [T, 77, 3]     world-frame joint positions
    local_rot_mats   [T, 77, 3, 3]
    global_rot_mats  [T, 77, 3, 3]
    root_positions   [T, 3]

Conversion steps:
    1. Downsample to 30 fps (AMASS native is 60 or 120 fps)
    2. Run SMPL-H forward kinematics to get 52 joint positions + rotations
    3. Map 52 SMPL+H joints → 77 SOMA joints (fill missing with T-pose offsets)
    4. Slice into fixed-length 90-frame clips

NOTE: Full SMPL-H FK requires the body model file (~300 MB). If not present,
we fall back to a simpler approach: use joint positions directly from the
bundled AMASS preview or compute a dummy T-pose for missing joints.

Usage:
    python prepare_amass.py --in-dir dataset/amass/CMU --out dataset/amass/cmu_soma77.npz
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import numpy as np


# --------- Joint mapping: SMPL+H (52) -> subset of SOMA77 (77) --------- #
# Based on standard SMPL-H definition and our SOMASkeleton77 ordering.

SMPLH_TO_SOMA = {
    # body (SMPL-H index -> SOMA joint name)
    0: "Hips",
    1: "LeftLeg",
    2: "RightLeg",
    3: "Spine1",
    4: "LeftShin",
    5: "RightShin",
    6: "Spine2",
    7: "LeftFoot",
    8: "RightFoot",
    9: "Chest",
    10: "LeftToeBase",
    11: "RightToeBase",
    12: "Neck1",
    13: "LeftShoulder",
    14: "RightShoulder",
    15: "Head",
    16: "LeftArm",
    17: "RightArm",
    18: "LeftForeArm",
    19: "RightForeArm",
    20: "LeftHand",
    21: "RightHand",
    # left hand
    22: "LeftHandIndex1",
    23: "LeftHandIndex2",
    24: "LeftHandIndex3",
    25: "LeftHandMiddle1",
    26: "LeftHandMiddle2",
    27: "LeftHandMiddle3",
    28: "LeftHandPinky1",
    29: "LeftHandPinky2",
    30: "LeftHandPinky3",
    31: "LeftHandRing1",
    32: "LeftHandRing2",
    33: "LeftHandRing3",
    34: "LeftHandThumb1",
    35: "LeftHandThumb2",
    36: "LeftHandThumb3",
    # right hand
    37: "RightHandIndex1",
    38: "RightHandIndex2",
    39: "RightHandIndex3",
    40: "RightHandMiddle1",
    41: "RightHandMiddle2",
    42: "RightHandMiddle3",
    43: "RightHandPinky1",
    44: "RightHandPinky2",
    45: "RightHandPinky3",
    46: "RightHandRing1",
    47: "RightHandRing2",
    48: "RightHandRing3",
    49: "RightHandThumb1",
    50: "RightHandThumb2",
    51: "RightHandThumb3",
}


# SOMA joints NOT covered by SMPL-H — these get T-pose rotation (identity)
# and are positioned via fixed bone offsets from their SOMA parent.
SOMA_EXTRA_JOINTS = {
    # Head detail
    "HeadEnd", "Jaw", "LeftEye", "RightEye",
    # Neck detail (SMPL has just one Neck, SOMA has Neck1 + Neck2)
    "Neck2",
    # Toe detail
    "LeftToeEnd", "RightToeEnd",
    # Finger 4th-joint tips and ends (SOMA has *Index4/End, SMPL doesn't)
    "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky4", "LeftHandPinkyEnd",
    "LeftHandThumbEnd",
    "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky4", "RightHandPinkyEnd",
    "RightHandThumbEnd",
}


# --------- Coordinate convention --------- #
# AMASS stores SMPL-H poses in Z-up (X=left/right, Y=forward/back, Z=up).
# Our downstream pipeline (Kimodo bone_offsets, MotionFormer training,
# visualisation tooling) is Y-up (X=left/right, Y=up, Z=forward/back).
#
# The change of basis that maps an AMASS vector into the Kimodo frame is a
# rotation by -90° around the X axis:
#
#     C = [[1,  0,  0],
#          [0,  0,  1],
#          [0, -1,  0]]      det(C) = +1  (right-handed preserved)
#
# so that C @ [0,0,1] (AMASS up) = [0,1,0] (Kimodo up).
#
# Vectors (positions, axis-angle rotations) transform as  v' = C @ v.
# Rotation matrices transform as  R' = C R C^T.
# Since axis-angle -> matrix is Rodrigues and conjugation of a Rodrigues by a
# rotation rotates its axis (angle preserved), rotating every axis-angle
# vector by C is exactly equivalent.
C_ZUP_TO_YUP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


# --------- Helpers --------- #


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues rotation: [..., 3] axis-angle -> [..., 3, 3] rotation matrix."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta_safe = np.where(theta > 1e-8, theta, 1.0)
    axis = aa / theta_safe
    cos = np.cos(theta)
    sin = np.sin(theta)
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zero = np.zeros_like(x)
    K = np.stack([
        np.concatenate([zero, -z, y], axis=-1),
        np.concatenate([z, zero, -x], axis=-1),
        np.concatenate([-y, x, zero], axis=-1),
    ], axis=-2)
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    R = I + sin[..., None] * K + (1.0 - cos[..., None]) * (K @ K)
    # where theta==0, we should return identity
    zero_mask = (theta.squeeze(-1) < 1e-8)
    R[zero_mask] = np.eye(3)
    return R


def downsample_fps(x: np.ndarray, src_fps: float, dst_fps: float = 30.0) -> np.ndarray:
    """Stride-based downsample of temporal data (T, ...)."""
    stride = max(1, int(round(src_fps / dst_fps)))
    return x[::stride]


def convert_one(
    amass_npz: Path,
    dst_fps: float,
    bone_offsets_soma77: np.ndarray,   # [77, 3] fixed rest offsets
    soma_joint_index: dict,
    soma_parents: list[int],
    clip_length: int,
) -> Iterable[dict]:
    """Yield per-clip SOMA77-format dicts from a single AMASS file."""
    try:
        d = np.load(amass_npz)
    except Exception as e:
        print(f"  skip {amass_npz.name}: load error {e}")
        return
    if "poses" not in d or "trans" not in d:
        print(f"  skip {amass_npz.name}: missing poses/trans")
        return

    poses = d["poses"]            # [T, 156] axis-angle
    trans = d["trans"]            # [T, 3]
    src_fps = float(d.get("mocap_framerate", 60.0))
    poses = downsample_fps(poses, src_fps, dst_fps)
    trans = downsample_fps(trans, src_fps, dst_fps)
    T = poses.shape[0]
    if T < clip_length:
        return

    # Reshape poses into [T, 52, 3] axis-angle, then to [T, 52, 3, 3]
    aa_52 = poses[:, :156].reshape(T, 52, 3)
    R_52 = axis_angle_to_matrix(aa_52)                    # [T, 52, 3, 3]

    # Build SOMA77 local rotations: use SMPL-H where mapped, identity elsewhere
    J77 = 77
    local_rot_77 = np.tile(np.eye(3), (T, J77, 1, 1)).astype(np.float32)
    for smpl_idx, soma_name in SMPLH_TO_SOMA.items():
        j = soma_joint_index[soma_name]
        local_rot_77[:, j] = R_52[:, smpl_idx]

    # FK to get global rotations + positions
    global_rot = np.zeros_like(local_rot_77)
    global_pos = np.zeros((T, J77, 3), dtype=np.float32)
    for j in range(J77):
        p = soma_parents[j]
        if p < 0:
            global_rot[:, j] = local_rot_77[:, j]
            global_pos[:, j] = trans.astype(np.float32)
        else:
            global_rot[:, j] = np.einsum("tij,tjk->tik",
                                         global_rot[:, p], local_rot_77[:, j])
            offset = bone_offsets_soma77[j].reshape(1, 3)
            global_pos[:, j] = (
                global_pos[:, p]
                + np.einsum("tij,j->ti", global_rot[:, p], bone_offsets_soma77[j])
            )

    # ---- Z-up (AMASS / SMPL) → Y-up (Kimodo) coordinate change ----------
    # Under a global-frame change by C, each object transforms as follows
    # (verified by FK self-consistency, see test_axis_fix.py):
    #   positions:        v_new = C @ v_old
    #   global rotations: R_new = C @ R_old        (left-mult, change of codomain)
    #   root local_rot:   same as global (root's parent is global)
    #   non-root local_rot: UNCHANGED (rotation in parent's abstract local frame)
    #   bone_offsets:     UNCHANGED (components in parent's abstract local frame)
    #
    # The reason bone_offsets and non-root local_rot don't transform: both are
    # numerical representations of objects in a frame that is attached to the
    # parent bone, not to the global world. Changing the global frame does
    # not renumber these.
    C  = C_ZUP_TO_YUP
    CT = C.T
    global_pos = (global_pos @ CT).astype(np.float32)
    trans_y    = (trans.astype(np.float32) @ CT)
    # ROOT rotation only: left-multiply by C. Non-root local_rot unchanged.
    local_rot_77[:, 0] = np.einsum("ij,tjk->tik", C, local_rot_77[:, 0])
    # global_rot: left-multiply by C (every joint, since global_rot is the
    # global-frame representation of the joint's orientation).
    global_rot = np.einsum("ij,tkjl->tkil", C, global_rot)

    # Slice into clips
    n_clips = T // clip_length
    for c in range(n_clips):
        s, e = c * clip_length, (c + 1) * clip_length
        yield {
            "joints_world":    global_pos[s:e].copy().astype(np.float32),
            "local_rot_mats":  local_rot_77[s:e].copy().astype(np.float32),
            "global_rot_mats": global_rot[s:e].copy().astype(np.float32),
            "root_positions":  trans_y[s:e].copy(),
            "source":          "amass",
            "sample_name":     amass_npz.stem,
            "subset":          amass_npz.parent.name,
            "clip_idx":        c,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, help="Directory of AMASS .npz (recursive)")
    p.add_argument("--out",    required=True, help="Output merged .npz file")
    p.add_argument("--fps",    type=float, default=30.0)
    p.add_argument("--clip",   type=int,   default=90)
    p.add_argument("--bone-offsets",
                   default="runs/kimodo_cache.npz",
                   help="Source for SOMA77 bone offsets. A Kimodo cache is fine.")
    p.add_argument("--stage1-dir",
                   default="/home/arenalabs/Desktop/\"be water, robot\"/experiment/stage1",
                   help="Path to add to sys.path for skeleton import.")
    args = p.parse_args()

    import sys
    sys.path.insert(0, args.stage1_dir)
    from soma_skeleton import build_parent_index, JOINT_INDEX, compute_bone_offsets

    # Load bone offsets from a Kimodo cache
    kc = Path(args.stage1_dir) / args.bone_offsets if not Path(args.bone_offsets).is_absolute() else Path(args.bone_offsets)
    kd = np.load(kc, allow_pickle=True)
    bone_offsets = compute_bone_offsets(kd["joints_world"], kd["global_rot_mats"])
    parents = build_parent_index()

    npz_files = sorted(Path(args.in_dir).rglob("*.npz"))
    print(f"Found {len(npz_files)} AMASS .npz under {args.in_dir}")

    all_joints = []
    all_local  = []
    all_global = []
    all_root   = []
    all_meta   = []

    for i, fp in enumerate(npz_files):
        for clip in convert_one(fp, args.fps, bone_offsets, JOINT_INDEX, parents, args.clip):
            all_joints.append(clip["joints_world"])
            all_local.append(clip["local_rot_mats"])
            all_global.append(clip["global_rot_mats"])
            all_root.append(clip["root_positions"])
            all_meta.append({
                k: clip[k] for k in ("source", "sample_name", "subset", "clip_idx")
            })
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(npz_files)} -> {len(all_joints)} clips")

    if not all_joints:
        print("No clips produced. Check --in-dir and file formats.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        joints_world    = np.stack(all_joints),
        local_rot_mats  = np.stack(all_local),
        global_rot_mats = np.stack(all_global),
        root_positions  = np.stack(all_root),
        meta            = np.array(all_meta, dtype=object),
    )
    print(f"\nSaved {len(all_joints)} clips to {out}")
    print(f"  joints_world    shape: {np.stack(all_joints).shape}")


if __name__ == "__main__":
    main()
