"""Acceptance numbers for surface_v2 (writes OUT/assets/surface_v2/validation/*.json).

    PY assets/validate_assets.py [--human] [--robots] [--schema]

human: chamfer of the posed bone-revolution points to the shaped SMPL-X mesh (calibration T-pose, a standing
       A-pose and a sitting-like pose, all posed by plain LBS from joint POSITIONS only on our side), and the stance
       check on LAFAN walk payload rows (lowest surface point is a foot, and its height after beta).
robots: FK consistency vs MuJoCo at 3 fresh random q (seed 7): every point posed by surface_v2.pose_surface from the
       AM2 (R, P) of that state vs the same point carried by its MuJoCo geom.
schema: interfaces.md keys/dtypes/shapes for all 18 assets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'assets'

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import _env as E  # noqa: E402
from . import surface_v2 as S  # noqa: E402

VAL_DIR = E.ASSET_DIR / 'validation'

SCHEMA = {  # key: (dtype kind, shape spec)  P=4096
    'points_local': ('f', ('P', 3)), 'normals_local': ('f', ('P', 3)), 'anchor': ('i', ('P',)), 'region': ('i', ('P',)),
    'pad': ('i', ('P',)), 'area': ('f', ('P',)), 'group_centre': ('i', (256,)), 'group_members': ('i', (256, 16)),
    'group_anchor': ('i', (256,)), 'group_region': ('i', (256,)), 'group128_centre': ('i', (128,)),
    'group128_members': ('i', (128, 32)), 'group128_anchor': ('i', (128,)), 'group128_region': ('i', (128,)),
    'sub1024': ('i', (1024,)), 'allowed_anchor_pairs': ('b', ('J', 'J')), 'rest_points': ('f', ('P', 3)),
    'binding': ('U', (1,)), 'beta_vis_cm': ('f', ()), 'delta_col_cm': ('f', ()),
}
HUMAN_EXTRA = {'bone_parent': ('i', ('P',)), 'bone_child': ('i', ('P',)), 't': ('f', ('P',)), 'radial': ('f', ('P',)),
               'theta': ('f', ('P',)), 'leaf_len': ('f', ('P',))}


def check_schema(body, J):
    z = np.load(E.ASSET_DIR / f'{body}.npz')
    errs = []
    spec = dict(SCHEMA, **(HUMAN_EXTRA if body == 'human_smpl' else {}))
    for k, (kind, shape) in spec.items():
        if k not in z.files:
            errs.append(f'missing {k}'); continue
        a = z[k]
        if a.dtype.kind != kind:
            errs.append(f'{k} dtype {a.dtype}')
        want = tuple(4096 if s == 'P' else J if s == 'J' else s for s in shape)
        if a.shape != want:
            errs.append(f'{k} shape {a.shape} != {want}')
    if not errs:
        if z['points_local'].dtype != np.float32:
            errs.append('points not float32')
        if not (z['anchor'].min() >= 0 and z['anchor'].max() < J):
            errs.append('anchor range')
        if not np.allclose(np.linalg.norm(z['normals_local'], axis=1), 1, atol=1e-4):
            errs.append('normals not unit')
        if not (z['region'].min() >= 0 and z['region'].max() < len(E.REGIONS)):
            errs.append('region range')
        if not (z['group_members'][:, 0] == z['group_centre']).all():
            errs.append('group member 0 != centre')
        if (z['allowed_anchor_pairs'] != z['allowed_anchor_pairs'].T).any() or z['allowed_anchor_pairs'].diagonal().any():
            errs.append('allowed pairs not symmetric / diagonal set')
        if len(np.unique(z['sub1024'])) != 1024:
            errs.append('sub1024 not unique')
    return errs


def schema_all():
    dump = E.read_json(E.CACHE_DIR / 'am2_bodies.json')['bodies']
    out = {b: check_schema(b, dump[b]['J']) for b in E.ALL_BODIES}
    return {'passed': all(not v for v in out.values()), 'errors': {k: v for k, v in out.items() if v}}


# ---------------------------------------------------------------------------------------------------------------
def robot_fk(body, n_q=3, seed=7):
    from . import robot_binding as RB
    kin = RB.load_kinematics(body)
    asset = S.load_surface(body)
    z = asset.extra
    g = z['point_geom'].astype(int); v = z['point_geom_vertex_cm'].astype(np.float64); rigid = z['point_rigid'].astype(bool)
    bound = z['point_residual_cm'].astype(np.float64)
    states = kin.random_states(n_q, seed=seed)[1:]
    errs = []
    for q in states:
        R, P, _, gp, gr = kin.forward(q)
        mj = np.einsum('pij,pj->pi', gr[g], v) + gp[g]
        with torch.no_grad():
            pts, nrm = S.pose_surface(asset, torch.as_tensor(R[None], dtype=torch.float32), torch.as_tensor(P[None], dtype=torch.float32))
        errs.append(np.linalg.norm(pts[0].numpy() - mj, axis=1))
    e = np.max(errs, 0)
    return {'states': len(states), 'seed': seed, 'n_rigid_points': int(rigid.sum()), 'n_approx_points': int((~rigid).sum()),
            'rigid_max_err_mm': float(e[rigid].max() * 10) if rigid.any() else None,
            'approx_max_err_cm': float(e[~rigid].max()) if (~rigid).any() else None,
            'approx_p95_err_cm': float(np.percentile(e[~rigid], 95)) if (~rigid).any() else None,
            'approx_points_within_1.5x_documented_residual': float((e[~rigid] <= 1.5 * bound[~rigid] + 0.05).mean()) if (~rigid).any() else None,
            'passed_rigid_0.5mm': bool(e[rigid].max() * 10 < 0.5) if rigid.any() else True}


# ---------------------------------------------------------------------------------------------------------------
def human_poses():
    """Named SMPL-X local rotvecs (radians) for the chamfer test (SMPL-X frame: x left, y up, z forward)."""
    return {
        'tpose_calibration': {},
        'standing_apose': {16: [0, 0, -1.0], 17: [0, 0, 1.0], 18: [0, -0.3, 0], 19: [0, 0.3, 0]},
        'sit_reach': {1: [-1.3, 0, 0], 2: [-1.3, 0, 0], 4: [1.4, 0, 0], 5: [1.4, 0, 0], 16: [0, 0.9, -0.4], 17: [0, -0.9, 0.4],
                      18: [0, -0.8, 0], 19: [0, 0.8, 0], 3: [0.2, 0, 0]},
    }


def human_validation():
    from . import human_surface as H
    asset = S.load_surface('human_smpl')
    sm = H.load_smplx()
    betas = asset.extra['smplx_betas'].astype(np.float64)
    out = {'chamfer': {}}
    for name, rv in human_poses().items():
        Vp, Jp = H.lbs_pose(sm, betas, {k: np.asarray(v, float) for k, v in rv.items()})
        root = Jp[0].copy(); Vp = Vp - root; Jp = Jp - root
        with torch.no_grad():
            pts, _ = S.pose_surface(asset, torch.zeros(1, 22, 3, 3), torch.as_tensor(Jp[None, :22], dtype=torch.float32))
        pts = pts[0].numpy().astype(np.float64)
        d_pm = H.point_to_mesh(pts, Vp, sm['faces'])
        from scipy.spatial import cKDTree
        hand = np.isin(H.vertex_bones(sm), H.HAND_BONES)
        d_mp = cKDTree(pts).query(Vp[~hand])[0]
        body_pts = ~np.isin(asset.bone_parent.numpy(), H.HAND_BONES)
        out['chamfer'][name] = {'points_to_mesh_cm': {'median': float(np.median(d_pm)), 'p95': float(np.percentile(d_pm, 95)),
                                                      'median_without_hand_ellipsoids': float(np.median(d_pm[body_pts]))},
                                'mesh_vertices_to_points_cm': {'median': float(np.median(d_mp)), 'p95': float(np.percentile(d_mp, 95)),
                                                               'note': 'hand vertices excluded (hands are ellipsoids)'},
                                'symmetric_median_cm': float(0.5 * (np.median(d_pm) + np.median(d_mp))),
                                'passed_median_lt_1.5cm': bool(np.median(d_pm) < 1.5 and np.median(d_mp) < 1.5)}
        if name == 'standing_apose':
            np.savez(VAL_DIR / 'human_apose_overlay.npz', points=pts.astype(np.float32), mesh_vertices=Vp.astype(np.float32),
                     faces=sm['faces'].astype(np.int32), joints=Jp[:22].astype(np.float32))
    out['stance'] = human_stance()
    return out


def human_stance():
    """LAFAN walk payload rows with a foot contact label: is the lowest dense point a foot, and how high is it?"""
    gid = [g['group_id'] for g in E.read_json(E.ROOT / 'data' / 'data_manifest_v2.json')['xwbc']['groups']]
    walk = np.array([i for i, g in enumerate(gid) if g.startswith('lafan::walk')])
    asset = S.load_surface('human_smpl')
    from .build_surface_v2 import rot6_to_matrix
    with np.load(E.PILOT_XWBC / 'payloads' / 'human_smpl.npz') as z:
        foot_j = [7, 8, 10, 11]
        m = (np.isin(z['group_index'], walk) & z['root_height_valid'] & z['ground_valid'] & z['contact_trust']
             & z['contact_hard'][:, foot_j].any(1))
        idx = np.flatnonzero(m)
        P = z['position_cm'][idx].astype(np.float32); rh = z['root_height_m'][idx].astype(np.float32)
    with torch.no_grad():
        pts, nrm = S.pose_surface(asset, torch.zeros(len(idx), 22, 3, 3), torch.as_tensor(P))
        h = S.ground_clearance(pts, torch.as_tensor(rh), asset.beta_vis_cm).numpy()
    reg = asset.region.numpy()
    feet = np.isin(reg, [E.RID['L_foot'], E.RID['R_foot']])
    low = h.argmin(1)
    sole = feet[None] & (nrm.numpy()[..., 1] < -0.7)
    h_sole = np.where(sole, h, np.inf).min(1)
    h_nonfoot = np.where(~feet[None], h, np.inf).min(1)
    return {'rows': int(len(idx)), 'clips': 'lafan::walk*', 'lowest_point_is_foot_frac': float(feet[low].mean()),
            'lowest_point_is_downward_sole_frac': float(sole[np.arange(len(idx)), low].mean()),
            'lowest_sole_height_after_beta_cm': {'median': float(np.median(h_sole)), 'p10': float(np.percentile(h_sole, 10)),
                                                 'p90': float(np.percentile(h_sole, 90))},
            'lowest_nonfoot_height_cm_median': float(np.median(h_nonfoot)),
            'beta_vis_cm': float(asset.beta_vis_cm)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', action='store_true'); ap.add_argument('--robots', action='store_true')
    ap.add_argument('--schema', action='store_true')
    a = ap.parse_args(argv)
    if not (a.human or a.robots or a.schema):
        a.human = a.robots = a.schema = True
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    if a.schema:
        r = schema_all(); E.write_json(VAL_DIR / 'schema.json', r); print('schema', json.dumps(r))
    if a.robots:
        r = {b: robot_fk(b) for b in E.ROBOTS}
        E.write_json(VAL_DIR / 'robot_fk.json', r)
        for b, v in r.items():
            print('fk', b, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}))
    if a.human:
        r = human_validation(); E.write_json(VAL_DIR / 'human.json', r); print('human', json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
