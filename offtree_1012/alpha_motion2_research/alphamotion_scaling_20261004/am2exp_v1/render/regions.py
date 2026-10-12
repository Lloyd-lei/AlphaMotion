"""Body-region tagging (15 regions, interfaces.md order) for humanoid joint trees.

Robots: structural rule on the kinematic tree (feet = lowest leaf per side at the MJCF standing pose,
hands = most distal non-leg leaf per side, knee/ankle and elbow/wrist = name keywords, else the two
longest bones of the limb chain) so it also works for anonymous / unusual names (kuavo leg_l1..l5,
booster AL1..AL3).  Human (SMPL-22, LAFAN names): fixed table by bone child joint.

The tag of a joint is the region of the rigid geometry hanging on that joint's frame (robots) or of the
bone that ends at that joint (human).  This is a RENDER helper (colouring, per-region minimum height);
the independent judge (CODE/judge) owns its own region definition.
"""
from __future__ import annotations

import re

import numpy as np

REGIONS = ['pelvis', 'trunk', 'head', 'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand',
           'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot']
R = {n: i for i, n in enumerate(REGIONS)}

HUMAN_BONE_REGION = {  # region of the bone that ENDS at this joint (root = pelvis)
    'Pelvis': 'pelvis', 'L_Hip': 'pelvis', 'R_Hip': 'pelvis', 'Torso': 'trunk', 'Spine': 'trunk', 'Chest': 'trunk',
    'Neck': 'trunk', 'Head': 'head', 'L_Thorax': 'trunk', 'R_Thorax': 'trunk', 'L_Shoulder': 'trunk',
    'R_Shoulder': 'trunk', 'L_Elbow': 'L_upper_arm', 'R_Elbow': 'R_upper_arm', 'L_Wrist': 'L_forearm',
    'R_Wrist': 'R_forearm', 'L_Knee': 'L_thigh', 'R_Knee': 'R_thigh', 'L_Ankle': 'L_shin', 'R_Ankle': 'R_shin',
    'L_Toe': 'L_foot', 'R_Toe': 'R_foot'}
# region of geometry ATTACHED to a human joint frame (used for leaf stubs: wrist -> hand, toe -> foot)
HUMAN_JOINT_REGION = dict(HUMAN_BONE_REGION, L_Wrist='L_hand', R_Wrist='R_hand', L_Ankle='L_foot', R_Ankle='R_foot',
                          Head='head', L_Hip='L_thigh', R_Hip='R_thigh', L_Knee='L_shin', R_Knee='R_shin',
                          L_Shoulder='L_upper_arm', R_Shoulder='R_upper_arm', L_Elbow='L_forearm', R_Elbow='R_forearm')

_KNEE = re.compile(r'knee|shank|shin|calf', re.I)
_ANKLE = re.compile(r'ankle|foot|toe|sole', re.I)
_ELBOW = re.compile(r'elbow|lower_arm|forearm', re.I)
_WRIST = re.compile(r'wrist|hand|end_effector|palm', re.I)


def human_bone_regions(names) -> np.ndarray:
    return np.array([R[HUMAN_BONE_REGION[n]] for n in names], np.int64)


def human_joint_regions(names) -> np.ndarray:
    return np.array([R[HUMAN_JOINT_REGION[n]] for n in names], np.int64)


def _children(parents):
    ch = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            ch[p].append(j)
    return ch


def _path_to_root(parents, j):
    out = []
    while j >= 0:
        out.append(j)
        j = int(parents[j])
    return out  # j ... root


def _lca(parents, a, b):
    pa = _path_to_root(parents, a)
    sb = set(_path_to_root(parents, b))
    return next(x for x in pa if x in sb)


def _split(chain, names, offsets, first_re, second_re):
    """Indices (into chain) of the two segment starts (knee/ankle or elbow/wrist)."""
    n = len(chain)
    lens = np.array([np.linalg.norm(offsets[c]) for c in chain])
    lens[0] = -1.0  # the chain-start offset is the trunk->limb offset, never a limb bone
    i1 = next((i for i in range(1, n) if names is not None and first_re.search(names[chain[i]])), None)
    i2 = next((i for i in range(1, n) if names is not None and second_re.search(names[chain[i]])
               and (i1 is None or i > i1)), None)
    if i1 is None and i2 is None:
        order = [i for i in np.argsort(-lens) if i > 0][:2]
        if len(order) == 2:
            i1, i2 = sorted(order)
        elif len(order) == 1:
            i1, i2 = order[0], n
        else:
            i1, i2 = n, n
    elif i1 is None:
        cand = [i for i in range(1, i2)]
        i1 = max(cand, key=lambda i: lens[i]) if cand else i2
    elif i2 is None:
        cand = [i for i in range(i1 + 1, n)]
        i2 = max(cand, key=lambda i: lens[i]) if cand else n
    return int(i1), int(i2)


def robot_joint_regions(parents, rest_pos_c, offsets_c, names=None):
    """parents [J]; rest_pos_c [J,3] canonical (Y-up, +X = robot left) standing-pose joint positions;
    offsets_c [J,3] (bone vectors, any frame -- only lengths are used).  Returns (region[J] int, info)."""
    parents = np.asarray(parents)
    J = len(parents)
    pos = np.asarray(rest_pos_c, np.float64)
    ch = _children(parents)
    leaves = [j for j in range(J) if not ch[j]]
    rootx = pos[0, 0]
    side_l = [j for j in leaves if pos[j, 0] - rootx > 1.0]
    side_r = [j for j in leaves if pos[j, 0] - rootx < -1.0]
    if not side_l or not side_r:
        raise ValueError('cannot find left/right leaves')
    foot_l = min(side_l, key=lambda j: pos[j, 1])
    foot_r = min(side_r, key=lambda j: pos[j, 1])
    leg_lca = _lca(parents, foot_l, foot_r)

    def chain(lca, leaf):
        p = _path_to_root(parents, leaf)
        return p[:p.index(lca)][::-1]  # child of lca ... leaf

    leg_l, leg_r = chain(leg_lca, foot_l), chain(leg_lca, foot_r)

    def subtree(j):
        out, st = set(), [j]
        while st:
            x = st.pop()
            out.add(x)
            st.extend(ch[x])
        return out

    leg_sub = subtree(leg_l[0]) | subtree(leg_r[0])

    def reach(j):
        return sum(np.linalg.norm(offsets_c[x]) for x in _path_to_root(parents, j)[:-1])

    cand_l = [j for j in side_l if j not in leg_sub]
    cand_r = [j for j in side_r if j not in leg_sub]
    if not cand_l or not cand_r:
        raise ValueError('cannot find hand leaves')
    hand_l = max(cand_l, key=reach)
    hand_r = max(cand_r, key=reach)
    arm_lca = _lca(parents, hand_l, hand_r)
    arm_l, arm_r = chain(arm_lca, hand_l), chain(arm_lca, hand_r)
    reg = np.full(J, -1, np.int64)
    for side, leg, arm in (('L', leg_l, arm_l), ('R', leg_r, arm_r)):
        k, a = _split(leg, names, offsets_c, _KNEE, _ANKLE)
        for i, j in enumerate(leg):
            reg[j] = R[f'{side}_thigh'] if i < k else (R[f'{side}_shin'] if i < a else R[f'{side}_foot'])
        e, w = _split(arm, names, offsets_c, _ELBOW, _WRIST)
        for i, j in enumerate(arm):
            reg[j] = R[f'{side}_upper_arm'] if i < e else (R[f'{side}_forearm'] if i < w else R[f'{side}_hand'])
    # torso: legLCA -> pelvis; path legLCA..armLCA (excl. legLCA) -> trunk
    reg[leg_lca] = R['pelvis']
    p_leg = _path_to_root(parents, leg_lca)
    p_arm = _path_to_root(parents, arm_lca)
    common = next(x for x in p_leg if x in set(p_arm))
    trunk_path = p_arm[:p_arm.index(common) + 1] + p_leg[:p_leg.index(common)]
    for j in trunk_path:
        if j != leg_lca and reg[j] < 0:
            reg[j] = R['trunk']
    shoulder_y = min(pos[arm_l[0], 1], pos[arm_r[0], 1])
    for j in range(J):  # topological (parents precede children)
        if reg[j] >= 0:
            continue
        p = int(parents[j])
        if p < 0:
            reg[j] = R['pelvis']
        elif reg[p] in (R['trunk'], R['pelvis'], R['head']) and j not in leg_sub:
            reg[j] = R['head'] if (pos[j, 1] > shoulder_y + 1.0 or reg[p] == R['head']) else (
                R['trunk'] if reg[p] != R['pelvis'] or pos[j, 1] > pos[leg_lca, 1] + 5.0 else R['pelvis'])
        else:
            reg[j] = reg[p]  # side branches (e.g. R1 ankle rods) inherit the parent's region
    info = {'foot_leaves': [int(foot_l), int(foot_r)], 'hand_leaves': [int(hand_l), int(hand_r)],
            'leg_lca': int(leg_lca), 'arm_lca': int(arm_lca), 'leg_chains': [leg_l, leg_r], 'arm_chains': [arm_l, arm_r]}
    return reg, info
