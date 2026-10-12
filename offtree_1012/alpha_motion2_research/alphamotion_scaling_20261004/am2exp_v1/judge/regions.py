"""Body-region vocabulary for judge J1 (15 whole-body regions, index order fixed by interfaces.md).

Robots: a collision geom's region comes from the vendor-MJCF body that owns it (name rules below,
plus a few per-robot overrides that were checked against rest-pose geometry, see README).
Human: region of each SMPL-22 joint centre (used only for the secondary CSE_human metric, which is
a documented under-count because labels are joint-centre heights).
"""
from __future__ import annotations
import re

REGIONS = ['pelvis', 'trunk', 'head', 'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm',
           'L_hand', 'R_hand', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot']
R = {n: i for i, n in enumerate(REGIONS)}
N_REGIONS = len(REGIONS)
FOOT_REGIONS = (R['L_foot'], R['R_foot'])
NONFOOT_REGIONS = tuple(i for i in range(N_REGIONS) if i not in FOOT_REGIONS)

ROBOTS_TRAIN = ['booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi', 'pnd_adam_lite',
                'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2']
ROBOTS_HELDOUT = ['fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1']
ROBOTS = ROBOTS_TRAIN + ROBOTS_HELDOUT  # the 15 XWBC robots judged by J1

# Human SMPL-22 (LAFAN names) joint -> region. Hip centres sit inside the pelvis, knees are the
# proximal end of the shin, the shoulder centre is the top of the upper arm, the elbow centre the
# proximal forearm, the wrist the hand.
HUMAN_JOINT_REGION = {
    'Pelvis': 'pelvis', 'L_Hip': 'pelvis', 'R_Hip': 'pelvis',
    'Torso': 'trunk', 'Spine': 'trunk', 'Chest': 'trunk', 'L_Thorax': 'trunk', 'R_Thorax': 'trunk',
    'Neck': 'head', 'Head': 'head',
    'L_Shoulder': 'L_upper_arm', 'R_Shoulder': 'R_upper_arm',
    'L_Elbow': 'L_forearm', 'R_Elbow': 'R_forearm',
    'L_Wrist': 'L_hand', 'R_Wrist': 'R_hand',
    'L_Knee': 'L_shin', 'R_Knee': 'R_shin',
    'L_Ankle': 'L_foot', 'R_Ankle': 'R_foot', 'L_Toe': 'L_foot', 'R_Toe': 'R_foot',
}
HUMAN_FOOT_JOINTS = ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe')

# Per-robot overrides (lower-case MJCF body name -> region), decided from the rest-pose geometry
# table printed by `python -m am2exp_v1.judge.pose_mjcf --regions` (README "Regions").
OVERRIDES = {
    # Booster K1: Arm_1 shoulder pitch, Arm_2 shoulder roll (upper arm shell), Arm_3 elbow (forearm).
    'booster_k1': {'left_arm_2': 'L_upper_arm', 'right_arm_2': 'R_upper_arm',
                   'left_arm_3': 'L_forearm', 'right_arm_3': 'R_forearm'},
    # Booster T1: AL1/AL2 shoulder, AL3 upper-arm shell, hand link carries the forearm+hand capsule.
    'booster_t1': {'al3': 'L_upper_arm', 'ar3': 'R_upper_arm'},
    # Booster T1 29dof: AL3 upper arm, AL4 forearm (elbow), AL5/AL6 wrist.
    'booster_t1_29dof': {'al3': 'L_upper_arm', 'ar3': 'R_upper_arm', 'al4': 'L_forearm', 'ar4': 'R_forearm',
                         'al5': 'L_forearm', 'ar5': 'R_forearm', 'al6': 'L_hand', 'ar6': 'R_hand'},
    # Hightorque Hi: the root base_link is the chest; waist_link (child of root, parent of the hips) is the pelvis.
    'hightorque_hi': {'base_link': 'trunk', 'waist_link': 'pelvis'},
    # Unitree R1: ankle A/B crank links sit at shin height (-47..-56 cm at qpos0); the wrist_roll link is the
    # distal end-effector link (no wrist pitch/yaw/hand geoms follow it).
    'unitree_r1': {'left_ankle_a_link': 'L_shin', 'right_ankle_a_link': 'R_shin',
                   'left_ankle_b_link': 'L_shin', 'right_ankle_b_link': 'R_shin',
                   'left_wrist_roll_link': 'L_hand', 'right_wrist_roll_link': 'R_hand'},
    # Fourier: hand_yaw link is the forearm shell (lower_arm_pitch has no collision geom).
    'fourier_n1': {'left_hand_yaw_link': 'L_forearm', 'right_hand_yaw_link': 'R_forearm'},
    'fourier_gr3': {'left_hand_yaw_link': 'L_forearm', 'right_hand_yaw_link': 'R_forearm'},
}


def side_of(n: str):
    n = n.lower()
    if re.search(r'(^|_)(left|l)(_|$)', n) or 'left' in n or re.match(r'(leg_l|zarm_l|al)\d', n):
        return 'L'
    if re.search(r'(^|_)(right|r)(_|$)', n) or 'right' in n or re.match(r'(leg_r|zarm_r|ar)\d', n):
        return 'R'
    return None


def part_of(n: str):
    n = n.lower()
    m = re.match(r'leg_[lr](\d)', n)
    if m:
        k = int(m.group(1))
        return 'thigh' if k <= 3 else 'shin' if k == 4 else 'foot'
    m = re.match(r'zarm_[lr](\d)', n)
    if m:
        k = int(m.group(1))
        return 'upper_arm' if k <= 3 else 'forearm' if k <= 5 else 'hand'
    if n.startswith('zhead') or n in ('h1', 'h2') or any(k in n for k in ('head', 'neck', 'camera')):
        return 'head'
    if any(k in n for k in ('ankle', 'foot', 'toe', 'sole')):
        return 'foot'
    if any(k in n for k in ('knee', 'shank', 'calf', 'shin')):
        return 'shin'
    if any(k in n for k in ('hip', 'thigh')):
        return 'thigh'
    if 'wrist_roll' in n:
        return 'forearm'
    if any(k in n for k in ('hand', 'wrist', 'end_effector', 'rubber', 'palm', 'finger')):
        return 'hand'
    if any(k in n for k in ('elbow', 'lower_arm', 'forearm')):
        return 'forearm'
    if any(k in n for k in ('shoulder', 'upper_arm', 'arm')) or re.match(r'a[lr]\d', n):
        return 'upper_arm'
    if any(k in n for k in ('torso', 'waist', 'chest', 'spine', 'trunk', 'imu_in_torso')):
        return 'trunk'
    if any(k in n for k in ('pelvis', 'base', 'imu_in_pelvis')):
        return 'pelvis'
    return None


ROOT_TRUNK_SPLIT_CM = 5.0


def region_of(robot: str, body_name: str, is_root: bool, centre_y_cm: float = 0.0) -> int:
    """Region index for a vendor-MJCF body.  Root-link geoms: 'trunk' when the geom centre sits at least
    ROOT_TRUNK_SPLIT_CM above the root origin at qpos0 (Booster/Kuavo roots carry the torso shell),
    else 'pelvis'.  Per-robot OVERRIDES win over every rule."""
    n = body_name.lower()
    o = OVERRIDES.get(robot, {})
    if n in o:
        return R[o[n]]
    if is_root:
        return R['trunk'] if centre_y_cm >= ROOT_TRUNK_SPLIT_CM else R['pelvis']
    part = part_of(n)
    if part is None:
        raise ValueError(f'{robot}: no region rule for body {body_name!r}')
    if part in ('pelvis', 'trunk', 'head'):
        return R[part]
    s = side_of(n)
    if s is None:
        raise ValueError(f'{robot}: limb body {body_name!r} has no side')
    return R[f'{s}_{part}']
