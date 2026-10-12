"""Region tagging (interfaces.md vocabulary, 15 body regions + digit/base_or_link/palm) for dense surface points.

Humanoid robots (rule = topology + names + rest geometry, identical code for all 15 robots, no per-robot table):
  1. Feet leaves = the lowest leaf joint (AM2 rest pose) on each side (x>0 left, x<0 right).  Pelvis joint = lowest
     common ancestor of the two foot leaves (root for most robots; hightorque_hi's root is the chest, its waist_link
     is the LCA).  Leg chain = path pelvis-joint -> foot leaf.
  2. Hand leaves = among leaves outside the leg subtrees, the one farthest (path length) from the pelvis joint on
     each side whose subtree root branch is not the head; arm chain = path from the branch point to the hand leaf.
     Head = highest remaining leaf; its branch (first joint after the trunk that does not contain a hand) = head.
  3. Knee / ankle / elbow / wrist joints on the chains: first name match (knee|shank|shin|calf, ankle|foot,
     elbow|lower_arm|forearm, wrist), otherwise geometric arc-length fractions (knee 0.47, ankle 0.85 of the
     hip->sole length; elbow 0.45 of the shoulder->arm-tip length, wrist = elbow + 0.6 of the remaining length).
  4. Each surface point anchored in a limb subtree is projected on that limb's rest polyline (chain joints + the
     geometric tip); its arc position vs the knee/ankle (elbow/wrist) arc positions decides thigh/shin/foot
     (upper_arm/forearm/hand).  So regions follow anatomical joint centres, not how the vendor split links
     (e.g. unitree_r1 ankle crank links at shin height become shin).
  5. Remaining points: head if anchored in the head branch or above y_shoulder + 0.25*(y_shoulder - y_hip) within
     the central column (|x| < max(0.8*shoulder half-width, 8 cm); robots like G1 carry the head mesh on the
     torso link), else pelvis if below
     y_hip + 0.30*(y_shoulder - y_hip) (iliac-crest level), else trunk.
Human (bone_revolution): region of the bone (see human_surface.HUMAN_BONE_REGION).
adroit_pen / relay_franka_full_v2: explicit tables below (fixed-base manipulators).
"""
from __future__ import annotations

import re

import numpy as np

from ._env import RID

KNEE_RE = re.compile(r'knee|shank|shin|calf', re.I)
ANKLE_RE = re.compile(r'ankle|foot', re.I)
ELBOW_RE = re.compile(r'elbow|lower_arm|forearm', re.I)
WRIST_RE = re.compile(r'wrist', re.I)


def children_of(parents):
    ch = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            ch[p].append(j)
    return ch


def path_to_root(parents, j):
    out = []
    while j >= 0:
        out.append(int(j)); j = int(parents[j])
    return out


def subtree(children, j):
    out, st = [], [j]
    while st:
        k = st.pop(); out.append(k); st.extend(children[k])
    return out


def lca(parents, a, b):
    pb = set(path_to_root(parents, b))
    return next(x for x in path_to_root(parents, a) if x in pb)


def polyline_arc(points, poly):
    """Arc-length parameter of each point projected on a polyline (first segment extended backwards, last forwards)."""
    poly = np.asarray(poly, np.float64)
    seg = poly[1:] - poly[:-1]
    L = np.linalg.norm(seg, axis=1)
    keep = L > 1e-6
    poly2 = np.concatenate([poly[:1], poly[1:][keep]])
    seg = poly2[1:] - poly2[:-1]; L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best_d = np.full(len(points), np.inf); best_s = np.zeros(len(points))
    for i in range(len(seg)):
        u = seg[i] / L[i]
        t = (points - poly2[i]) @ u
        lo = -np.inf if i == 0 else 0.0
        hi = np.inf if i == len(seg) - 1 else L[i]
        tc = np.clip(t, lo, hi)
        d = np.linalg.norm(points - (poly2[i] + tc[:, None] * u), axis=1)
        better = d < best_d
        best_d[better] = d[better]; best_s[better] = cum[i] + tc[better]
    return best_s, cum, poly2


def _arc_of_joint(Prest, chain, j):
    cum = 0.0
    for a, b in zip(chain[:-1], chain[1:]):
        if a == j:
            return cum
        cum += float(np.linalg.norm(Prest[b] - Prest[a]))
        if b == j:
            return cum
    return 0.0 if chain[0] == j else None


def humanoid_regions(names, parents, Prest, points, anchor):
    """Per-point region ids for a humanoid robot + an audit/table dict (see module docstring)."""
    names = list(names); parents = np.asarray(parents); J = len(parents)
    ch = children_of(parents)
    leaves = [j for j in range(J) if not ch[j]]
    feet = {}
    for side, sgn in (('L', 1), ('R', -1)):
        cand = [j for j in leaves if sgn * Prest[j, 0] > 1e-3]
        feet[side] = min(cand, key=lambda j: (Prest[j, 1], -abs(Prest[j, 0])))
    pelvis_j = lca(parents, feet['L'], feet['R'])
    leg_root = {}
    for s in 'LR':
        path = path_to_root(parents, feet[s])[::-1]
        leg_root[s] = path[path.index(pelvis_j) + 1]
    leg_nodes = {s: set(subtree(ch, leg_root[s])) for s in 'LR'}
    nonleg = [j for j in range(J) if j not in leg_nodes['L'] | leg_nodes['R']]
    # path length from the pelvis joint (through the tree) for every non-leg leaf
    def plen(j):
        pa = path_to_root(parents, j); pb = path_to_root(parents, pelvis_j)
        c = lca(parents, j, pelvis_j)
        tot = 0.0
        for x in (pa, pb):
            for k in x:
                if k == c:
                    break
                tot += float(np.linalg.norm(Prest[k] - Prest[parents[k]]))
        return tot
    up_leaves = [j for j in leaves if j in nonleg]
    hands = {}
    for side, sgn in (('L', 1), ('R', -1)):
        cand = [j for j in up_leaves if sgn * Prest[j, 0] > 2.0]
        hands[side] = max(cand, key=lambda j: plen(j))
    shoulder_branch = lca(parents, hands['L'], hands['R'])
    arm_root = {}
    for s in 'LR':
        path = path_to_root(parents, hands[s])[::-1]
        arm_root[s] = path[path.index(shoulder_branch) + 1]
    arm_nodes = {s: set(subtree(ch, arm_root[s])) for s in 'LR'}
    rest = [j for j in up_leaves if j not in arm_nodes['L'] | arm_nodes['R']]
    head_leaf = max(rest, key=lambda j: Prest[j, 1]) if rest else None
    head_nodes = set()
    if head_leaf is not None and Prest[head_leaf, 1] > Prest[shoulder_branch, 1]:
        path = path_to_root(parents, head_leaf)[::-1]
        cut = lca(parents, head_leaf, hands['L'])
        hb = path[path.index(cut) + 1]
        head_nodes = set(subtree(ch, hb))
    table = {'pelvis_joint': names[pelvis_j], 'feet': {s: names[feet[s]] for s in 'LR'},
             'hands': {s: names[hands[s]] for s in 'LR'}, 'head_leaf': None if head_leaf is None else names[head_leaf],
             'head_branch': sorted(names[j] for j in head_nodes)}
    region = np.full(len(points), -1, np.int64)
    # ---- legs
    y_hip = np.mean([Prest[leg_root[s], 1] for s in 'LR'])
    for s in 'LR':
        chain = path_to_root(parents, feet[s])[::-1]
        chain = chain[chain.index(leg_root[s]):]
        sel = np.isin(anchor, list(leg_nodes[s]))
        if not sel.any():
            continue
        tip = points[sel][np.argmin(points[sel][:, 1])]  # lowest leg point = sole
        poly = [Prest[j] for j in chain] + [np.array([Prest[chain[-1], 0], tip[1], Prest[chain[-1], 2]])]
        arc, cum, _ = polyline_arc(points[sel], poly)
        Ltot = cum[-1]
        knee = next((j for j in chain[1:] if KNEE_RE.search(names[j])), None)
        knee_how = 'name'
        if knee is None:
            knee = min(chain[1:], key=lambda j: abs(_arc_of_joint(Prest, chain, j) - 0.47 * Ltot)); knee_how = 'arc0.47'
        ka = _arc_of_joint(Prest, chain, knee)
        ank = next((j for j in chain[chain.index(knee) + 1:] if ANKLE_RE.search(names[j])
                    and _arc_of_joint(Prest, chain, j) >= 0.75 * Ltot), None)
        ank_how = 'name'
        if ank is None:
            later = [j for j in chain[chain.index(knee) + 1:] if _arc_of_joint(Prest, chain, j) >= 0.85 * Ltot]
            ank = later[0] if later else chain[-1]; ank_how = 'arc0.85'
        aa = _arc_of_joint(Prest, chain, ank)
        r = np.where(arc < ka, RID[s + '_thigh'], np.where(arc < aa, RID[s + '_shin'], RID[s + '_foot']))
        region[np.flatnonzero(sel)] = r
        table[s + '_leg'] = {'chain': [names[j] for j in chain], 'knee': names[knee], 'knee_rule': knee_how,
                             'ankle': names[ank], 'ankle_rule': ank_how, 'length_cm': float(Ltot)}
    # ---- arms
    y_sh = np.mean([Prest[arm_root[s], 1] for s in 'LR'])
    for s in 'LR':
        chain = path_to_root(parents, hands[s])[::-1]
        chain = chain[chain.index(arm_root[s]):]
        sel = np.isin(anchor, list(arm_nodes[s]))
        if not sel.any():
            continue
        dist = np.linalg.norm(points[sel] - Prest[chain[0]], axis=1)
        tip = points[sel][np.argmax(dist)]
        poly = [Prest[j] for j in chain] + [tip]
        arc, cum, _ = polyline_arc(points[sel], poly)
        Ltot = cum[-1]
        elb = next((j for j in chain[1:] if ELBOW_RE.search(names[j])), None)
        elb_how = 'name'
        if elb is None:
            elb = min(chain[1:], key=lambda j: abs(_arc_of_joint(Prest, chain, j) - 0.45 * Ltot)); elb_how = 'arc0.45'
        ea = _arc_of_joint(Prest, chain, elb)
        wr = [j for j in chain[chain.index(elb) + 1:] if WRIST_RE.search(names[j])]
        if wr:
            wa = _arc_of_joint(Prest, chain, wr[-1]); wr_name = names[wr[-1]]; wr_how = 'name(last wrist joint)'
        else:
            wa = ea + 0.6 * (Ltot - ea); wr_name = None; wr_how = 'arc elbow+0.6*rest'
        r = np.where(arc < ea, RID[s + '_upper_arm'], np.where(arc < wa, RID[s + '_forearm'], RID[s + '_hand']))
        region[np.flatnonzero(sel)] = r
        table[s + '_arm'] = {'chain': [names[j] for j in chain], 'elbow': names[elb], 'elbow_rule': elb_how,
                             'wrist': wr_name, 'wrist_rule': wr_how, 'wrist_arc_cm': float(wa), 'length_cm': float(Ltot)}
    # ---- head / pelvis / trunk
    rest_sel = region < 0
    y_split = y_hip + 0.30 * (y_sh - y_hip)
    y_head = y_sh + 0.25 * (y_sh - y_hip)
    half_w = max(0.8 * np.mean([abs(Prest[arm_root[s], 0]) for s in 'LR']), 8.0)
    in_head = np.isin(anchor, list(head_nodes)) | ((points[:, 1] > y_head) & (np.abs(points[:, 0]) < half_w))
    region[rest_sel & in_head] = RID['head']
    other = rest_sel & ~in_head
    region[other] = np.where(points[other][:, 1] < y_split, RID['pelvis'], RID['trunk'])
    table['y_pelvis_trunk_split_cm'] = float(y_split)
    table['head_rule'] = f'anchor in head branch OR (y > {y_head:.1f} cm and |x| < {half_w:.1f} cm)'
    jt = {}
    for j in range(J):
        sel = anchor == j
        if sel.any():
            vals, cnt = np.unique(region[sel], return_counts=True)
            jt[names[j]] = {REGION_NAME(int(v)): int(c) for v, c in zip(vals, cnt)}
    table['joint_to_region_counts'] = jt
    return region, table


def REGION_NAME(i):
    from ._env import REGIONS
    return REGIONS[i]


ADROIT_REGION = {'forearm': 'base_or_link', 'wrist': 'R_hand', 'palm': 'palm', 'lfmetacarpal': 'palm'}


def adroit_regions(names, anchor):
    reg = np.array([RID[ADROIT_REGION.get(names[a], 'digit')] for a in anchor], np.int64)
    table = {n: ADROIT_REGION.get(n, 'digit') for n in names}
    return reg, table


def franka_regions(names, anchor, mesh_names):
    out = []
    for a, mn in zip(anchor, mesh_names):
        n = names[a]
        if 'finger' in n:
            out.append(RID['digit'])
        elif mn is not None and 'hand' in mn:
            out.append(RID['palm'])
        else:
            out.append(RID['base_or_link'])
    table = {'panda0_link0..7': 'base_or_link', 'hand_viz mesh (on link7)': 'palm', 'left/right finger': 'digit'}
    return np.asarray(out, np.int64), table
