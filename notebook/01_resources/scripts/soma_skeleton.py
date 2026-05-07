"""
SOMA77 kinematic tree extracted from Kimodo's skeleton definition.

Used for:
    - Parent index lookup
    - Descendant masking (for kinematic chain masking mode)

We reproduce the skeleton list here so Stage 1 can run without
importing Kimodo's full dependency stack (torch modules, peft, etc).
"""

from __future__ import annotations

from typing import List, Tuple


# (joint_name, parent_name)  — parent=None for the root joint
SOMA77_BONES: List[Tuple[str, str | None]] = [
    ("Hips", None),
    ("Spine1", "Hips"),
    ("Spine2", "Spine1"),
    ("Chest", "Spine2"),
    ("Neck1", "Chest"),
    ("Neck2", "Neck1"),
    ("Head", "Neck2"),
    ("HeadEnd", "Head"),
    ("Jaw", "Head"),
    ("LeftEye", "Head"),
    ("RightEye", "Head"),
    ("LeftShoulder", "Chest"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("LeftHandThumb1", "LeftHand"),
    ("LeftHandThumb2", "LeftHandThumb1"),
    ("LeftHandThumb3", "LeftHandThumb2"),
    ("LeftHandThumbEnd", "LeftHandThumb3"),
    ("LeftHandIndex1", "LeftHand"),
    ("LeftHandIndex2", "LeftHandIndex1"),
    ("LeftHandIndex3", "LeftHandIndex2"),
    ("LeftHandIndex4", "LeftHandIndex3"),
    ("LeftHandIndexEnd", "LeftHandIndex4"),
    ("LeftHandMiddle1", "LeftHand"),
    ("LeftHandMiddle2", "LeftHandMiddle1"),
    ("LeftHandMiddle3", "LeftHandMiddle2"),
    ("LeftHandMiddle4", "LeftHandMiddle3"),
    ("LeftHandMiddleEnd", "LeftHandMiddle4"),
    ("LeftHandRing1", "LeftHand"),
    ("LeftHandRing2", "LeftHandRing1"),
    ("LeftHandRing3", "LeftHandRing2"),
    ("LeftHandRing4", "LeftHandRing3"),
    ("LeftHandRingEnd", "LeftHandRing4"),
    ("LeftHandPinky1", "LeftHand"),
    ("LeftHandPinky2", "LeftHandPinky1"),
    ("LeftHandPinky3", "LeftHandPinky2"),
    ("LeftHandPinky4", "LeftHandPinky3"),
    ("LeftHandPinkyEnd", "LeftHandPinky4"),
    ("RightShoulder", "Chest"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("RightHandThumb1", "RightHand"),
    ("RightHandThumb2", "RightHandThumb1"),
    ("RightHandThumb3", "RightHandThumb2"),
    ("RightHandThumbEnd", "RightHandThumb3"),
    ("RightHandIndex1", "RightHand"),
    ("RightHandIndex2", "RightHandIndex1"),
    ("RightHandIndex3", "RightHandIndex2"),
    ("RightHandIndex4", "RightHandIndex3"),
    ("RightHandIndexEnd", "RightHandIndex4"),
    ("RightHandMiddle1", "RightHand"),
    ("RightHandMiddle2", "RightHandMiddle1"),
    ("RightHandMiddle3", "RightHandMiddle2"),
    ("RightHandMiddle4", "RightHandMiddle3"),
    ("RightHandMiddleEnd", "RightHandMiddle4"),
    ("RightHandRing1", "RightHand"),
    ("RightHandRing2", "RightHandRing1"),
    ("RightHandRing3", "RightHandRing2"),
    ("RightHandRing4", "RightHandRing3"),
    ("RightHandRingEnd", "RightHandRing4"),
    ("RightHandPinky1", "RightHand"),
    ("RightHandPinky2", "RightHandPinky1"),
    ("RightHandPinky3", "RightHandPinky2"),
    ("RightHandPinky4", "RightHandPinky3"),
    ("RightHandPinkyEnd", "RightHandPinky4"),
    ("LeftLeg", "Hips"),
    ("LeftShin", "LeftLeg"),
    ("LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftFoot"),
    ("LeftToeEnd", "LeftToeBase"),
    ("RightLeg", "Hips"),
    ("RightShin", "RightLeg"),
    ("RightFoot", "RightShin"),
    ("RightToeBase", "RightFoot"),
    ("RightToeEnd", "RightToeBase"),
]

JOINT_NAMES: List[str] = [name for name, _ in SOMA77_BONES]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


def build_parent_index() -> List[int]:
    """Return a list of length 77 giving the parent joint index (or -1 for root)."""
    out = []
    for _, parent in SOMA77_BONES:
        out.append(-1 if parent is None else JOINT_INDEX[parent])
    return out


def build_descendants() -> List[List[int]]:
    """For each joint i, the list of joint indices that are descendants of i
    (inclusive of i itself). Used for kinematic chain masking."""
    parents = build_parent_index()
    n = len(parents)
    children: List[List[int]] = [[] for _ in range(n)]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    descendants = [[] for _ in range(n)]
    for i in range(n):
        stack = [i]
        subtree = []
        while stack:
            v = stack.pop()
            subtree.append(v)
            stack.extend(children[v])
        descendants[i] = sorted(subtree)
    return descendants


# Coarse joint groups — useful for mask-mode stats and visualisation
JOINT_GROUPS = {
    "spine_head": [
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2",
        "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye",
    ],
    "left_arm": [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    ],
    "right_arm": [
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ],
    "left_fingers": [n for n in JOINT_NAMES if n.startswith("LeftHand") and n not in ("LeftHand",)],
    "right_fingers": [n for n in JOINT_NAMES if n.startswith("RightHand") and n not in ("RightHand",)],
    "left_leg": ["LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd"],
    "right_leg": ["RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd"],
}


def compute_bone_offsets(
    joints_world: np.ndarray,        # [N, T, J, 3]
    global_rot_mats: np.ndarray,     # [N, T, J, 3, 3]
) -> np.ndarray:
    """Recover per-joint rest-pose bone offsets in the parent's local frame.

    For each (non-root) joint j:
        bone_offsets[j] = R_parent^T @ (pos_j - pos_parent)
    The bone offset should be (approximately) constant over time and samples.
    We take the median over all (N, T) observations for robustness.
    """
    import numpy as _np
    parents = build_parent_index()
    N, T, J, _ = joints_world.shape
    offsets = _np.zeros((J, 3), dtype=_np.float32)
    for j in range(J):
        p = parents[j]
        if p < 0:
            continue  # root, offset stays 0
        pos_diff = joints_world[:, :, j, :] - joints_world[:, :, p, :]   # [N, T, 3]
        R_parent = global_rot_mats[:, :, p, :, :]                         # [N, T, 3, 3]
        # local offset = R_parent^T @ pos_diff
        local = _np.einsum("...ij,...i->...j", R_parent, pos_diff)        # [N, T, 3]
        offsets[j] = _np.median(local.reshape(-1, 3), axis=0)
    return offsets


def forward_kinematics_torch(
    local_rot_mats,                 # [B, T, J, 3, 3] torch.Tensor
    root_positions,                 # [B, T, 3]
    bone_offsets,                   # [J, 3] torch.Tensor, constant
    parents_tensor,                 # [J] torch.LongTensor, -1 for root
):
    """Autograd-safe FK: sequential over joints in topological order, but we
    accumulate into Python lists and stack at the end to avoid in-place ops.

    SOMA77's joint order (JOINT_NAMES) is already topological, so a single
    forward pass over j = 0..J-1 is correct.
    """
    import torch
    B, T, J, _, _ = local_rot_mats.shape

    global_rot_list = [None] * J
    global_pos_list = [None] * J
    parents_cpu = parents_tensor.detach().cpu().tolist()
    for j in range(J):
        p = parents_cpu[j]
        if p < 0:
            global_rot_list[j] = local_rot_mats[:, :, j]
            global_pos_list[j] = root_positions
        else:
            global_rot_list[j] = global_rot_list[p] @ local_rot_mats[:, :, j]
            offset = bone_offsets[j].view(1, 1, 3).expand(B, T, 3)
            global_pos_list[j] = (
                global_pos_list[p]
                + torch.einsum("btij,btj->bti", global_rot_list[p], offset)
            )
    global_rot = torch.stack(global_rot_list, dim=2)     # [B, T, J, 3, 3]
    global_pos = torch.stack(global_pos_list, dim=2)     # [B, T, J, 3]
    return global_pos, global_rot


def rot6d_to_matrix(rot6d):
    """Convert 6D rotation representation (Zhou et al. 2019) to 3x3 rotation.

    rot6d : [..., 6]  -> first 3 dims = first column, next 3 = second column.
    Returns: [..., 3, 3] proper rotation via Gram-Schmidt.
    """
    import torch
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    e1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    a2_proj = a2 - (a2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = a2_proj / (torch.norm(a2_proj, dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)   # [..., 3, 3]


def matrix_to_rot6d(R):
    """Convert 3x3 rotation to 6D (Zhou et al. 2019). Keeps first two columns."""
    import torch
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # [..., 6]


if __name__ == "__main__":
    parents = build_parent_index()
    desc = build_descendants()
    print(f"Total joints: {len(JOINT_NAMES)}")
    print(f"Root(s): {[JOINT_NAMES[i] for i, p in enumerate(parents) if p == -1]}")
    for key, names in JOINT_GROUPS.items():
        print(f"  {key}: {len(names)} joints")
    print()
    # Examples of kinematic chain mask targets
    for jname in ["LeftArm", "LeftShin", "Chest", "Hips"]:
        idx = JOINT_INDEX[jname]
        sub = desc[idx]
        print(f"Mask '{jname}' → descendants: {len(sub)} joints "
              f"({', '.join(JOINT_NAMES[i] for i in sub[:8])}...)")
