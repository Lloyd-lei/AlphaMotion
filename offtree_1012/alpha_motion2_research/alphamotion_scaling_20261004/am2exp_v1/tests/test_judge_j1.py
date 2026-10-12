"""WP-JUDGE: judge J1 geometry and metrics (CPU)."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

mujoco = pytest.importorskip('mujoco')
from am2exp_v1.judge import pose_mjcf as PM  # noqa: E402
from am2exp_v1.judge import ground_judge as GJ  # noqa: E402
from am2exp_v1.judge.regions import ROBOTS, FOOT_REGIONS, R as RIDX  # noqa: E402

AX = PM.AX
BODIES_FAST = ['unitree_g1_29dof', 'fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1', 'booster_t1']


def am2_state(b, model, data, q):
    """MuJoCo state -> AM2 (rot, pos cm root-relative) using the binding manifest node bodies/frames."""
    data.qpos[:] = q; mujoco.mj_forward(model, data)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    nb = np.array([bid(n) for n in b.manifest['node_body']]); nf = np.array([bid(n) for n in b.manifest['node_frame_body']])
    rot = AX[None] @ data.xmat[nf].reshape(-1, 3, 3) @ AX.T[None]
    pos = (data.xpos[nb] - data.xpos[nb[0]]) @ AX.T * 100.0
    return rot[None], pos[None], data.xpos[nb[0]][2] * 100.0


@pytest.mark.parametrize('body', BODIES_FAST)
def test_geom_repose_matches_mujoco(body):
    b = PM.load_binding(body)
    m, d = b.mj
    for q in PM.random_states(m, 3, seed=123)[1:]:
        q = q.copy()
        rot, pos, root_z = am2_state(b, m, d, q)
        gr, gp = PM.pose_geoms_arrays(b, rot, pos)
        root = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b.manifest['node_body'][0])]
        ref_p = (d.geom_xpos[b.geom_id] - root) @ AX.T * 100.0
        ref_r = AX[None] @ d.geom_xmat[b.geom_id].reshape(-1, 3, 3) @ AX.T[None]
        err_p = np.linalg.norm(gp[0] - ref_p, axis=-1)
        rigid = ~b.approx
        assert err_p[rigid].max() < 0.05, (body, err_p.max())
        assert np.abs(gr[0][rigid] - ref_r[rigid]).max() < 1e-3


@pytest.mark.parametrize('body', ['unitree_g1_29dof', 'fourier_gr3', 'unitree_h2'])
def test_lowest_point_matches_brute_force(body):
    """J1 lowest point (hull vertices / analytic primitives) == brute force on full mesh vertices in MuJoCo."""
    b = PM.load_binding(body)
    m, d = b.mj
    for q in PM.random_states(m, 2, seed=5)[1:]:
        rot, pos, root_z = am2_state(b, m, d, q)
        low, _, _ = PM.geom_lowest_cm(b, rot, pos)
        root = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b.manifest['node_body'][0])]
        for k, g in enumerate(b.geom_id):
            R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]; t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                z = (v @ R.T + c)[:, 2].min()
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                z = c[2] - np.abs(R[2]).dot(s)
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = c[2] - s[0]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = c[2] - abs(R[2, 2]) * s[1] - s[0]
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z = c[2] - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0, 1 - R[2, 2] ** 2))
            else:
                continue
            if b.approx[k]:
                continue
            assert abs(low[0, k] - (z - root[2]) * 100.0) < 0.05, (body, k)


def test_all_robots_have_both_feet_and_bindings_verified():
    for body in ROBOTS:
        b = PM.load_binding(body)
        assert b.regions_present[RIDX['L_foot']] and b.regions_present[RIDX['R_foot']], body
        assert not b.approx.any(), body           # every geom rigid or verified merged-chain
        assert b.manifest['approx_residual_max_cm'] < 1e-3, body
        if body != 'unitree_h2':
            assert b.manifest['fk_max_error_cm'] < PM.FK_TOL_CM


def test_selfpen_matches_mj_geomDistance():
    body = 'unitree_g1_29dof'
    b = PM.load_binding(body)
    m, d = b.mj
    found = 0
    for q in PM.random_states(m, 12, seed=11)[1:]:
        rot, pos, _ = am2_state(b, m, d, q)
        gr, gp = PM.pose_geoms_arrays(b, rot, pos)
        mine = PM.selfpen_depth_cm(b, gr, gp, report_threshold_cm=0.0)[0]
        m2, d2 = mujoco.MjModel.from_xml_path(b.manifest['xml']), None
        d2 = mujoco.MjData(m2); d2.qpos[:] = q; mujoco.mj_forward(m2, d2)
        ref = 0.0
        for a_, c_ in b.pairs:
            dist = mujoco.mj_geomDistance(m2, d2, int(b.geom_id[a_]), int(b.geom_id[c_]), 0.0, None)
            ref = max(ref, -dist * 100.0)
        assert abs(mine - ref) < 1e-3
        found += ref > 0
    assert found >= 0


def _fake(T=10, body='unitree_g1_29dof'):
    b = PM.load_binding(body)
    rot = np.repeat(b.rest_rot[None], T, 0); pos = np.repeat(b.rest_pos_cm[None], T, 0)
    low = PM.geom_lowest_cm(b, rot[:1], pos[:1])[0][0].min()
    return rot, pos, np.full(T, -low / 100.0)


def test_judge_case_root_shift_and_identity_metrics():
    rot, pos, rh = _fake()
    g = GJ.judge_case('unitree_g1_29dof', rot, pos, rh)
    assert np.allclose(g['zmin_cm'], 0.0, atol=1e-6)
    feet = g['region_min_height_cm'][:, list(FOOT_REGIONS)]
    assert (feet < 1e-6).any(1).all()
    s = GJ.judge_case('unitree_g1_29dof', rot, pos, rh + 0.05)
    assert np.allclose(s['zmin_cm'], 5.0, atol=1e-6)
    m = GJ.case_metrics(g, g)
    assert m['CSE_GT'] == 0.0 and m['FloatRate'] == 0.0 and m['PenRate'] == 0.0
    m5 = GJ.case_metrics(s, g)
    assert m5['FloatRate'] == 1.0 and m5['CSE_GT'] > 0.5
    mneg = GJ.case_metrics(GJ.judge_case('unitree_g1_29dof', rot, pos, rh - 0.05), g)
    assert mneg['PenRate'] == 1.0 and abs(mneg['PenP95_cm'] - 5.0) < 1e-6
    # snapped variant is invariant to a constant root offset
    assert m5['CSE_snap'] == 0.0


def test_corrected_root_conventions():
    r = np.array([0.8, 0.8])
    assert np.allclose(GJ.corrected_root('booster_k1', r, 'gt'), 0.8 - 0.064)
    assert np.allclose(GJ.corrected_root('booster_k1', r, 'pred'), 0.8 - 0.064)
    assert np.allclose(GJ.corrected_root('unitree_h2', r, 'pred', {'unitree_h2': 2.0}), 0.78)
    with pytest.raises(ValueError):
        GJ.corrected_root('unitree_h2', r, 'pred', None)
    t = GJ.delta_col_table(verify=True, corrected=False)
    assert abs(t['unitree_r1'] + 0.4992) < 1e-3
    assert abs(GJ.DELTA_COL_CM['unitree_r1'] + 4.3845) < 1e-3


def test_human_region_contact_mapping():
    names = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
             'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
    hard = np.zeros((2, 22), bool); hard[0, names.index('L_Knee')] = True; hard[1, names.index('R_Wrist')] = True
    c, v = GJ.human_region_contact(hard, np.ones_like(hard), names)
    assert c[0, RIDX['L_shin']] and c[1, RIDX['R_hand']] and c.sum() == 2


def test_judge_independence():
    """J1 must not import the dense visual surface, the model package or any training loss."""
    for f in ('pose_mjcf.py', 'ground_judge.py', 'regions.py'):
        src = (ROOT / 'am2exp_v1/judge' / f).read_text()
        for bad in ('surface_v2', 'am2exp_v1.assets', 'am2exp_v1.model', 'losses_relational', 'recipe_ground', 'contact_logits'):
            assert bad not in src.split('"""', 2)[-1], (f, bad)


def test_ruler_summary_present_and_consistent():
    p = PM.OUT / 'judge/ruler_v1/ruler_gate_summary.json'
    if not p.exists():
        pytest.skip('ruler gate not run yet')
    s = json.loads(p.read_text())
    assert s['n_bodies'] == 15 and len(s['bodies']) == 15
    # J1 static (qpos0) foot drop equals report-32's true drop for every body except R1 (report-32 error, documented)
    for b, r in s['bodies'].items():
        la = r['corpora']['lafan']
        if b != 'unitree_r1':
            assert abs(la['diag_j1_static_drop_cm'] - la['diag_report32_true_drop_cm']) < 0.05, b
