"""Robot staging for UMR.

For every core robot we write a *standalone* MJCF under DATA/robots/<name>/<name>.xml:
  * includes flattened, mesh/texture paths absolute (UMR writes <out>.floating_mjcf.xml and temp object MJCFs
    next to the robot XML, so it must live in a directory we own -- never inside GMR / vendor trees);
  * keyframes removed (they break when joints are removed);
  * optional joint removals (unitree_g1_23dof = 29-DoF tree minus waist roll/pitch and wrist pitch/yaw);
  * <contact><exclude> for collision-geom body pairs that overlap in the rest/T-pose AND are kinematically
    adjacent (<= 2 tree hops) or always touching over random in-limit poses.  The list is returned for docs.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from surface_retargeting.config import CFG as SRCFG
from surface_retargeting.paths import INVENTORY, DRAFT_CONFIGS, ROBOTS, add_umr_to_path

LOCKED_JOINTS = {
    'unitree_g1_23dof': ['waist_roll_joint', 'waist_pitch_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                         'right_wrist_pitch_joint', 'right_wrist_yaw_joint'],
}


# Pairs that touch through a closed kinematic loop that the MJCF models as an open chain (judgement, documented).
MANUAL_EXCLUSIONS = {
    'agibot_a2': [(f'{s}_toe_roll', f'{s}_toe_{r}_rod',
                   'ankle parallel-linkage rod end attaches to the toe plate in the physical closed chain '
                   '(open-chain MJCF); overlaps 6.2-6.4 mm at rest')
                  for s in ('left', 'right') for r in ('A', 'B')],
}


def load_inventory(tier='A_core20'):
    d = SRCFG.load_json(INVENTORY)
    return [r for r in d['robots'] if tier is None or r['tier'] == tier]


def mid(m, t, name):
    return mujoco.mj_name2id(m, t, name)


# ------------------------------------------------------------------------------------------------- MJCF flattening
def _localize(path: Path, dest: Path, used: dict) -> str:
    """Hard-link (same fs) or copy an asset into dest; UMR's correspondence builder rewrites any absolute path that
    contains '/assets/' to its own tree and chokes on quotes, so staged MJCFs must reference files under DATA."""
    import os, shutil
    path = path.resolve()
    if str(path) in used:
        return used[str(path)]
    dest.mkdir(parents=True, exist_ok=True)
    name = f'{len(used):03d}_{path.name}'
    out = dest / name
    if not out.exists():
        try:
            os.link(path, out)
        except OSError:
            shutil.copy2(path, out)
    used[str(path)] = str(out)
    return str(out)


def _abs_assets(spec: mujoco.MjSpec, src: Path, dest: Path):
    base = src.parent
    mdir = Path(spec.meshdir) if spec.meshdir else Path('')
    mdir = mdir if mdir.is_absolute() else base / mdir
    tdir = Path(spec.texturedir) if spec.texturedir else Path('')
    tdir = tdir if tdir.is_absolute() else base / tdir
    used = {}
    for m in spec.meshes:
        if m.file:
            p = Path(m.file)
            if not m.name:                       # implicit name = file stem; keep it when the file is renamed
                m.name = p.stem
            m.file = _localize(p if p.is_absolute() else mdir / p, dest, used)
    for t in spec.textures:
        if t.file:
            p = Path(t.file)
            if not t.name:
                t.name = p.stem
            t.file = _localize(p if p.is_absolute() else tdir / p, dest, used)
    for h in spec.hfields:
        if h.file:
            p = Path(h.file)
            h.file = _localize(p if p.is_absolute() else base / p, dest, used)
    spec.meshdir = ''
    spec.texturedir = ''


def _remove_joints(spec: mujoco.MjSpec, names):
    names = set(names)
    for a in list(spec.actuators):
        if a.target in names:
            spec.delete(a)
    for s in list(spec.sensors):
        if getattr(s, 'objname', '') in names:
            spec.delete(s)
    for j in list(spec.joints):
        if j.name in names:
            spec.delete(j)


def stage_spec(src: str, name: str, excludes=()):
    spec = mujoco.MjSpec.from_file(str(src))
    _abs_assets(spec, Path(src), ROBOTS / name / 'mesh')
    for k in list(spec.keys):
        spec.delete(k)
    if name in LOCKED_JOINTS:
        _remove_joints(spec, LOCKED_JOINTS[name])
    for b1, b2 in excludes:
        e = spec.add_exclude()
        e.bodyname1 = b1
        e.bodyname2 = b2
    return spec


def write_staged(src, name, excludes=()):
    out_dir = ROBOTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = stage_spec(src, name, excludes)
    spec.compile()
    xml = spec.to_xml()
    p = out_dir / f'{name}.xml'
    p.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(p))      # must load standalone
    return p, m


# ------------------------------------------------------------------------------------------------- model queries
def robot_geoms(m):
    """Collision-enabled geoms not on the world body."""
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]


def body_names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f'body{b}' for b in range(m.nbody)]


def tree_hops(m, b1, b2):
    def anc(b):
        out = [b]
        while b != 0:
            b = int(m.body_parentid[b]); out.append(b)
        return out
    a1, a2 = anc(b1), anc(b2)
    common = next(x for x in a1 if x in a2)
    return a1.index(common) + a2.index(common)


def set_tpose(m, d, tpose: dict, root_z=2.0):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[0:3] = [0, 0, root_z]
        d.qpos[3:7] = [1, 0, 0, 0]
    for jn, v in tpose.items():
        j = mid(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, d)


def overlapping_pairs(m, d, qpos, depth=-1e-4):
    """Robot collision-geom pairs (by body) with signed distance < depth at qpos (MuJoCo filters already apply)."""
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    rg = set(robot_geoms(m))
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in rg and c.geom2 in rg and c.dist < depth:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            k = (min(b1, b2), max(b1, b2))
            out[k] = min(out.get(k, 0.0), float(c.dist))
    return out


def random_limit_poses(m, n, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.repeat(m.qpos0[None], n, 0).astype(np.float64)
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi))
            Q[:, m.jnt_qposadr[j]] = rng.uniform(lo, hi, n)
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        Q[:, 0:3] = [0, 0, 2.0]; Q[:, 3:7] = [1, 0, 0, 0]
    return Q


def rest_overlap_exclusions(m, tpose, n_random=300, manual=()):
    """Return (exclude_pairs [(b1,b2,reason)], flagged_pairs) using qpos0 and T-pose overlaps."""
    d = mujoco.MjData(m)
    names = body_names(m)
    q0 = m.qpos0.copy()
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        q0[0:3] = [0, 0, 2.0]; q0[3:7] = [1, 0, 0, 0]
    set_tpose(m, d, tpose)
    qt = d.qpos.copy()
    rest = overlapping_pairs(m, d, q0)
    tp = overlapping_pairs(m, d, qt)
    cand = {}
    for k, v in list(rest.items()) + list(tp.items()):
        cand[k] = min(cand.get(k, 0.0), v)
    # always-touching test
    Q = random_limit_poses(m, n_random)
    hits = {k: 0 for k in cand}
    for q in Q:
        ov = overlapping_pairs(m, d, q)
        for k in cand:
            if k in ov:
                hits[k] += 1
    excl, flagged = [], []
    man = {frozenset((a, b)): r for a, b, r in manual}
    for k, dist in sorted(cand.items()):
        hops = tree_hops(m, *k)
        frac = hits[k] / len(Q)
        rec = dict(body1=names[k[0]], body2=names[k[1]], rest_depth_mm=round(-dist * 1000, 2), tree_hops=hops,
                   overlap_frac_random_poses=round(frac, 3))
        if frozenset((names[k[0]], names[k[1]])) in man:
            rec['reason'] = 'manual: ' + man[frozenset((names[k[0]], names[k[1]]))]
            excl.append(rec)
        elif hops <= 2:
            rec['reason'] = f'rest/T-pose overlap {rec["rest_depth_mm"]} mm, kinematically adjacent ({hops} hops)'
            excl.append(rec)
        elif frac >= 0.95:
            rec['reason'] = f'rest/T-pose overlap, always touching ({frac:.0%} of {len(Q)} random in-limit poses)'
            excl.append(rec)
        else:
            rec['reason'] = 'rest overlap between NON-adjacent links: NOT excluded (counted as penetration)'
            flagged.append(rec)
    return excl, flagged


# ------------------------------------------------------------------------------------------------- T-pose check/fix
def _surface_geoms(m):
    add_umr_to_path()
    from mujoco_geom_surface import surface_geom_ids
    return [int(g) for g in surface_geom_ids(m, 'auto')]


def subtree_bodies(m, root):
    out = [root]
    for b in range(m.nbody):
        p = b
        while p != 0:
            p = int(m.body_parentid[p])
            if p == root:
                out.append(b); break
    return sorted(set(out))


def arm_landmarks(m, d, side_rec, side_sign, geoms):
    """Shoulder (first arm joint anchor), elbow (elbow joint anchor), wrist (hand body), tip (most lateral
    visual vertex of the hand subtree)."""
    J = side_rec['arm_chain_joints']
    jid = [mid(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in J]
    jid = [j for j in jid if j >= 0]
    S = d.xanchor[jid[0]].copy()
    el = [j for j in jid if 'elbow' in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower()]
    E = d.xanchor[el[0]].copy() if el else d.xanchor[jid[len(jid) // 2]].copy()
    hb = mid(m, mujoco.mjtObj.mjOBJ_BODY, side_rec['hand_body'])
    sub = set(subtree_bodies(m, hb))
    # wrist = anchor of the first joint after the elbow (if any) else hand body origin
    after = [j for j in jid if el and j > el[0]]
    W = d.xanchor[after[0]].copy() if after else d.xpos[hb].copy()
    pts = []
    for g in geoms:
        if int(m.geom_bodyid[g]) in sub and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    if pts:
        P = np.concatenate(pts)
        H = P[np.argmax(side_sign * P[:, 1])]
    else:
        H = d.xpos[hb].copy()
    return S, E, W, H


def elbow_axis_z(m, d, side_rec):
    """|z| of the elbow flexion axis in world (1 = vertical, as in a palms-down human T-pose)."""
    for j in side_rec['arm_chain_joints']:
        if 'elbow' in j.lower():
            jid = mid(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                return float(abs(d.xaxis[jid][2]))
    return 1.0


def tpose_metrics(m, tpose, inv):
    d = mujoco.MjData(m)
    set_tpose(m, d, tpose)
    geoms = _surface_geoms(m)
    out = {}
    for side, sgn in (('L', 1.0), ('R', -1.0)):
        S, E, W, H = arm_landmarks(m, d, inv['hands'][side], sgn, geoms)
        ez = elbow_axis_z(m, d, inv['hands'][side])
        lat = np.array([0, sgn, 0.0])

        def ang(a, b):
            v = b - a
            n = np.linalg.norm(v)
            return float(np.degrees(np.arccos(np.clip(v @ lat / max(n, 1e-9), -1, 1)))) if n > 0.03 else 0.0
        out[side] = dict(upper_arm_deg=ang(S, E), forearm_deg=ang(E, W), hand_deg=ang(W, H),
                         tip_minus_shoulder_cm=[round(float(x), 1) for x in (H - S) * 100],
                         elbow_axis_vertical=round(ez, 3))
    # pass = arms level and lateral. elbow_axis_vertical (palms-down proxy) is optimised but only reported:
    # Booster K1/T1 cannot reach it within limits and UMR's own K1 T-pose has it at 0.
    ok = all(max(v['upper_arm_deg'], v['forearm_deg']) <= 15 and abs(v['tip_minus_shoulder_cm'][2]) <= 7
             and abs(v['tip_minus_shoulder_cm'][0]) <= 7 for v in out.values())
    out['pass'] = bool(ok)
    return out


def solve_tpose(m, inv, init: dict, n_starts=24):
    """Least-squares T-pose of each arm chain: upper arm, forearm, hand all lateral (+y left, -y right)."""
    from scipy.optimize import least_squares
    d = mujoco.MjData(m)
    geoms = _surface_geoms(m)
    result = dict(init)
    for side, sgn in (('L', 1.0), ('R', -1.0)):
        rec = inv['hands'][side]
        J = [j for j in rec['arm_chain_joints'] if mid(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0]
        jid = [mid(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in J]
        lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jid])
        hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jid])
        lat = np.array([0, sgn, 0.0])

        def res(x):
            tp = dict(result); tp.update({j: float(v) for j, v in zip(J, x)})
            set_tpose(m, d, tp)
            S, E, W, H = arm_landmarks(m, d, rec, sgn, geoms)
            u = (E - S) / max(np.linalg.norm(E - S), 1e-9)
            f = (W - E) / max(np.linalg.norm(W - E), 1e-9)
            g = (H - W) / max(np.linalg.norm(H - W), 1e-9)
            ez = elbow_axis_z(m, d, rec)
            return np.concatenate([u - lat, f - lat, 0.5 * (g - lat), [(H - S)[2] * 2, (H - S)[0] * 2, 0.5 * (1 - ez)],
                                   0.02 * x])
        x0 = np.clip(np.array([init.get(j, 0.0) for j in J]), lo, hi)
        best = None
        for s in range(n_starts):
            xi = x0 if s == 0 else np.random.default_rng(s).uniform(lo, hi)
            r = least_squares(res, xi, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
        result.update({j: round(float(v), 5) for j, v in zip(J, best.x)})
    return result


def bbox_center_ratio(m, tpose, center_body):
    """UMR robot point_cloud_center height ratio inside the T-pose visual bbox (method of scout; reproduces K1/H2)."""
    d = mujoco.MjData(m)
    set_tpose(m, d, tpose)
    geoms = _surface_geoms(m)
    zs = []
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            zs.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2])
    z = np.concatenate(zs)
    b = mid(m, mujoco.mjtObj.mjOBJ_BODY, center_body.split(':', 1)[1])
    return float((d.xpos[b][2] - z.min()) / (z.max() - z.min()))
