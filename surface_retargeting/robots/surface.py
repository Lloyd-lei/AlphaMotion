"""Dense robot surface (>= 4096 exterior points from VISUAL meshes, UMR's first-hit sampler) bound to MuJoCo geoms,
exact ground gauge (convex-hull vertices of every visual geom), and body -> region map.

Why not surface_v2: those assets are bound to the xwbc cache MJCFs (GMR) and missing for 3 of the 21 robots; ours are
bound to the *staged* MJCF geoms that UMR and the post-pass actually pose, for all 21 robots with one sampler.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from surface_retargeting.paths import ROBOTS, add_umr_to_path
from surface_retargeting.robots import robots as R

REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
RID = {r: i for i, r in enumerate(REGIONS)}

# SMPL-X segm part (joint index) -> region
SMPLX_PART_REGION = {0: 'trunk', 3: 'trunk', 6: 'trunk', 9: 'trunk', 12: 'trunk', 13: 'trunk', 14: 'trunk', 15: 'trunk',
                     22: 'trunk', 23: 'trunk', 24: 'trunk',
                     1: 'L_thigh', 2: 'R_thigh', 4: 'L_shin', 5: 'R_shin', 7: 'L_foot', 10: 'L_foot', 8: 'R_foot',
                     11: 'R_foot', 16: 'L_upper_arm', 17: 'R_upper_arm', 18: 'L_forearm', 19: 'R_forearm',
                     20: 'L_hand', 21: 'R_hand'}
for _j in range(25, 40):
    SMPLX_PART_REGION[_j] = 'L_hand'
for _j in range(40, 55):
    SMPLX_PART_REGION[_j] = 'R_hand'


def _joint_body(m, jname):
    j = R.mid(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return int(m.jnt_bodyid[j]) if j >= 0 else -1


def _ancestors(m, b):
    out = []
    while b != 0:
        out.append(b); b = int(m.body_parentid[b])
    return out


def body_regions(m, inv):
    """Region per body from the kinematic tree: foot = subtree of topmost 'ankle'/'foot' body above the foot,
    shin = knee subtree minus foot, thigh = first leg-chain body subtree minus shin/foot; same for arms
    (upper_arm / forearm (elbow) / hand (first wrist joint, else hand body)); everything else = trunk."""
    reg = np.full(m.nbody, RID['trunk'], np.int16)
    names = R.body_names(m)
    for s in 'LR':
        # ---- leg
        fb = R.mid(m, mujoco.mjtObj.mjOBJ_BODY, inv['feet'][s]['body'])
        chain = _ancestors(m, fb)[::-1]                         # root-child ... foot
        # leg chain starts at the first body whose subtree does not contain the other foot
        other = R.mid(m, mujoco.mjtObj.mjOBJ_BODY, inv['feet']['R' if s == 'L' else 'L']['body'])
        oth_anc = set(_ancestors(m, other))
        leg = [b for b in chain if b not in oth_anc]
        knee = [b for b in leg if 'knee' in names[b].lower() or 'calf' in names[b].lower() or 'shank' in names[b].lower()
                or 'tibia' in names[b].lower()]
        ankle = [b for b in leg if any(k in names[b].lower() for k in ('ankle', 'foot', 'toe', 'tarsus', 'sole'))]
        if not knee:      # fall back: 4th body of a 6-dof leg
            knee = [leg[min(3, len(leg) - 1)]]
        if not ankle:
            ankle = [fb]
        thigh_root, shin_root = leg[0], knee[0]
        foot_root = next(b for b in leg if b in ankle and leg.index(b) > leg.index(shin_root)) \
            if any(leg.index(b) > leg.index(shin_root) for b in ankle) else fb
        for b in R.subtree_bodies(m, thigh_root):
            reg[b] = RID[f'{s}_thigh']
        for b in R.subtree_bodies(m, shin_root):
            reg[b] = RID[f'{s}_shin']
        for b in R.subtree_bodies(m, foot_root):
            reg[b] = RID[f'{s}_foot']
        # ---- arm
        J = inv['hands'][s]['arm_chain_joints']
        jb = [_joint_body(m, j) for j in J]
        jb = [b for b in jb if b >= 0]
        el = [_joint_body(m, j) for j in J if 'elbow' in j.lower()]
        wr = [_joint_body(m, j) for j in J if 'wrist' in j.lower() and _joint_body(m, j) >= 0]
        hb = R.mid(m, mujoco.mjtObj.mjOBJ_BODY, inv['hands'][s]['hand_body'])
        ua_root = jb[0]
        fa_root = el[0] if el and el[0] >= 0 else jb[len(jb) // 2]
        hand_root = wr[0] if wr else hb
        for b in R.subtree_bodies(m, ua_root):
            reg[b] = RID[f'{s}_upper_arm']
        for b in R.subtree_bodies(m, fa_root):
            reg[b] = RID[f'{s}_forearm']
        for b in R.subtree_bodies(m, hand_root):
            reg[b] = RID[f'{s}_hand']
    return reg


def visual_geoms(m):
    add_umr_to_path()
    from mujoco_geom_surface import surface_geom_ids
    return [int(g) for g in surface_geom_ids(m, 'auto') if m.geom_bodyid[g] != 0]


def build_surface(name, inv, n_points=4096, seed=0):
    """Sample exterior points at the T-pose, bind them to geoms. Saves ROBOTS/<name>/surface.npz."""
    add_umr_to_path()
    import surface_sampling as ss
    from mujoco_geom_surface import geom_local_mesh
    st = json.load(open(ROBOTS / name / 'staging.json'))
    m = mujoco.MjModel.from_xml_path(st['staged_mjcf']); d = mujoco.MjData(m)
    R.set_tpose(m, d, st['tpose'])
    gids = visual_geoms(m)
    V, F, GF, hullv, hullg = [], [], [], [], []
    off = 0
    for g in gids:
        v, f = geom_local_mesh(m, g)
        v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
        if len(f) == 0 or len(v) < 4:
            continue
        try:
            h = ConvexHull(v)
            hv = v[h.vertices]
        except Exception:                                      # degenerate (flat) mesh: keep all vertices
            hv = v
        hullv.append(hv); hullg.append(np.full(len(hv), g))
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        V.append(w); F.append(f + off); GF.append(np.full(len(f), g)); off += len(v)
    V = np.concatenate(V); F = np.concatenate(F); GF = np.concatenate(GF)
    tri = V[F]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.flatnonzero(areas > 1e-12)
    rng = np.random.default_rng(seed)
    cand_n = n_points * 8 * 12
    pick = valid[rng.choice(len(valid), size=cand_n, p=areas[valid] / areas[valid].sum())]
    pts = ss._sample_points_on_triangles(tri[pick], rng)
    ext = ss._first_hit_mask(V, F, pick, ray_offset=1e-4, ray_distance=0.0, min_visible_views=2, chunk_size=4096)
    pts, pick = pts[ext], pick[ext]
    sel = ss._farthest_point_indices(pts, n_points, seed)
    pts, pick = pts[sel], pick[sel]
    g_of = GF[pick]
    loc = np.einsum('nji,nj->ni', d.geom_xmat[g_of].reshape(-1, 3, 3), pts - d.geom_xpos[g_of])  # R^T (p - x)
    n_w = np.cross(tri[pick, 1] - tri[pick, 0], tri[pick, 2] - tri[pick, 0])
    n_w /= np.maximum(np.linalg.norm(n_w, axis=1, keepdims=True), 1e-12)
    nloc = np.einsum('nji,nj->ni', d.geom_xmat[g_of].reshape(-1, 3, 3), n_w)
    breg = body_regions(m, inv)
    body = m.geom_bodyid[g_of]
    region = breg[body].astype(np.int16)
    split_note = {}
    for sd, sgn in (('L', 1.0), ('R', -1.0)):
        fa, hd = RID[f'{sd}_forearm'], RID[f'{sd}_hand']
        if (region == fa).sum() == 0 or (region == hd).sum() == 0:
            # hand mesh lives on the elbow/forearm link (no wrist joint): split by lateral distance from the elbow
            idx = np.flatnonzero((region == fa) | (region == hd))
            ej = [j for j in inv['hands'][sd]['arm_chain_joints'] if 'elbow' in j.lower()]
            jid = R.mid(m, mujoco.mjtObj.mjOBJ_JOINT, ej[0])
            lat = sgn * (pts[idx, 1] - d.xanchor[jid][1])
            cut = 0.65 * lat.max()
            region[idx] = np.where(lat > cut, hd, fa)
            split_note[sd] = f'geometric split at {cut*100:.1f} cm lateral from elbow anchor (hand mesh on forearm link)'
    out = dict(points_local=loc.astype(np.float32), normals_local=nloc.astype(np.float32), geom=g_of.astype(np.int32),
               body=body.astype(np.int32), region=region, body_region=breg, split_note=np.array(json.dumps(split_note)),
               hull_local=np.concatenate(hullv).astype(np.float32), hull_geom=np.concatenate(hullg).astype(np.int32),
               regions=np.array(REGIONS), n_exterior_candidates=np.int32(len(ext)), tpose_points=pts.astype(np.float32))
    np.savez_compressed(ROBOTS / name / 'surface.npz', **out)
    return out


def load_surface(name):
    z = np.load(ROBOTS / name / 'surface.npz')
    return {k: z[k] for k in z.files}


def posed(d, geom, local):
    """World positions of geom-local points: [N,3]."""
    Rm = d.geom_xmat[geom].reshape(-1, 3, 3)
    return np.einsum('nij,nj->ni', Rm, local) + d.geom_xpos[geom]
