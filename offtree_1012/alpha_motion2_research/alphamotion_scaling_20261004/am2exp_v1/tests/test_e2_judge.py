"""WP-E2 closure judge J2 + analytic retarget tests (CPU)."""
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from e2 import closure_judge as cj  # noqa: E402
from e2 import analytic_retarget as an  # noqa: E402
from e2 import opposition as op  # noqa: E402
from e2._env import E2_DATA, PEN_XML, ROOT  # noqa: E402

needs_data = pytest.mark.skipif(not (E2_DATA / 'hand_hammer.npz').exists(), reason='run e2/hand_data.py all first')


# --------------------------------------------------------------------------------------------
# independence (rule 4: judge never uses training-loss code)
# --------------------------------------------------------------------------------------------
def test_judge_imports_no_training_or_descriptor_code():
    src = (CODE / 'e2' / 'closure_judge.py').read_text()
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(('.' * node.level) + (node.module or ''))
    allowed = {'__future__', 'numpy', 'mujoco', '._env'}
    assert mods <= allowed, mods - allowed
    # and importing it in a fresh interpreter pulls in none of them
    code = ('import sys; sys.path.insert(0, %r); import e2.closure_judge; '
            'bad=[m for m in sys.modules if m.startswith(("e2.opposition","e2.xbody_loss","e2.analytic","pilot_model","model","torch"))]; '
            'print(bad); assert not bad' % str(CODE))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120,
                       env={'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'})
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------------------------
# bands, BA, floors
# --------------------------------------------------------------------------------------------
def test_gripper_state_band_edges():
    d = 50.0
    c = np.array([46.99, 47.0, 50.0, 58.0, 58.01, 78.0, -2.0])
    s = cj.gripper_state(c, d)
    assert s.tolist() == [cj.CRUSH, cj.HOLD, cj.HOLD, cj.HOLD, cj.OPEN, cj.OPEN, cj.CRUSH]


def test_balanced_accuracy_floors_exact():
    rng = np.random.default_rng(0)
    lab = rng.choice([1, 0, -1], size=1000, p=[.3, .5, .2]).astype(np.int8)
    d = 50.0
    open_ = cj.balanced_accuracy(cj.gripper_state(np.full(1000, 78.0), d), lab)
    assert open_['BA'] == 0.5 and open_['P_hold_given_grasp'] == 0.0 and open_['P_open_given_free'] == 1.0
    hold = cj.balanced_accuracy(cj.gripper_state(np.full(1000, 50.0), d), lab)
    assert hold['BA'] == 0.5
    closed = cj.balanced_accuracy(cj.gripper_state(np.full(1000, -2.0), d), lab)
    assert closed['BA'] == 0.0 and closed['crush_rate_given_grasp'] == 1.0
    oracle = np.where(lab == 1, 50.0, 78.0)
    assert cj.balanced_accuracy(cj.gripper_state(oracle, d), lab)['BA'] == 1.0
    # relocate: 70 mm ball, 78 mm stroke -> open is unreachable for a clipped gripper (documented)
    assert cj.gripper_state(np.array([78.0]), 70.0)[0] == cj.HOLD


def test_episode_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(1)
    ep = np.repeat(np.arange(10), 50)
    lab = rng.choice([1, 0], size=500).astype(np.int8)
    st = np.where(rng.random(500) < 0.8, np.where(lab == 1, cj.HOLD, cj.OPEN), cj.CRUSH).astype(np.int8)
    ba = cj.balanced_accuracy(st, lab)['BA']
    ci = cj.episode_bootstrap(st, lab, ep, n_boot=300, seed=0)
    assert ci['lo'] <= ba <= ci['hi'] and ci['n'] == 300


# --------------------------------------------------------------------------------------------
# geometry: re-posing from AM2 rotations/positions reproduces MuJoCo kinematics
# --------------------------------------------------------------------------------------------
def _decoded(n=20, seed=0):
    sys.path.insert(0, str(ROOT / 'data'))
    from mujoco_dataset_adapter import MujocoMotionAdapter
    a = MujocoMotionAdapter(PEN_XML, 'forearm')
    m = a.model
    lo, hi = m.jnt_range[:24, 0], m.jnt_range[:24, 1]
    rng = np.random.default_rng(seed)
    q = np.repeat(m.qpos0[None], n, 0)
    q[:, :24] = lo + rng.random((n, 24)) * (hi - lo)
    dec = a.decode_q(q)
    return q, dec['position_native'] * 100, dec['global_matrices']


def test_repose_hand_matches_mujoco_kinematics():
    sc = cj.JudgeScene()
    q, P, G = _decoded(20)
    worst_geom, worst_site = 0.0, 0.0
    hand_geoms = [g for g in range(sc.m.ngeom) if int(sc.m.geom_bodyid[g]) in sc.body_index]
    for i in range(len(q)):
        sc.d.qpos[:] = sc.m.qpos0
        sc.d.qpos[:24] = q[i, :24]
        sc.kinematics()
        ref_g = sc.d.geom_xpos[hand_geoms].copy()
        ref_R = sc.d.geom_xmat[hand_geoms].copy()
        ref_s = sc.d.site_xpos[list(sc.sites.values())].copy()
        sc.d.geom_xpos[:] = 0
        sc.repose_hand(P[i], G[i])
        worst_geom = max(worst_geom, float(np.abs(sc.d.geom_xpos[hand_geoms] - ref_g).max()),
                         float(np.abs(sc.d.geom_xmat[hand_geoms] - ref_R).max()))
        worst_site = max(worst_site, float(np.abs(sc.d.site_xpos[list(sc.sites.values())] - ref_s).max()))
    assert worst_geom < 1e-6 and worst_site < 1e-6, (worst_geom, worst_site)
    assert len(sc.body_ids) == 25


def test_object_placement_canonical_roundtrip():
    sc = cj.JudgeScene()
    sc.set_object_size('capsule', 2.5, 5.0)
    R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    g = sc.set_object('capsule', np.array([3.0, -4.0, 12.0]), geom_R_can=R)
    sc.kinematics()
    p_can = cj.AX @ (sc.d.geom_xpos[g] - sc.P_fa) * 100
    R_can = cj.AX @ sc.d.geom_xmat[g].reshape(3, 3)
    assert np.allclose(p_can, [3.0, -4.0, 12.0], atol=1e-9)
    assert np.allclose(R_can, R, atol=1e-6)


# --------------------------------------------------------------------------------------------
# grasp labels on real data: moving the object away flips grasp -> free
# --------------------------------------------------------------------------------------------
@needs_data
def test_grasp_labels_respond_to_object_position():
    from e2 import hand_data as hd
    D = hd.load_hand('hammer')
    sc = cj.JudgeScene()
    lab, dist, _ = cj.grasp_labels({k: D[k][:3000] if np.ndim(D[k]) and len(D[k]) == len(D['hand_qpos']) else D[k] for k in D}, sc)
    g1 = np.flatnonzero(lab == 1)[:50]
    assert len(g1) == 50
    assert (dist[g1, 0] < cj.GRASP_NEAR_MM).all()
    far = {k: (v[g1] if np.ndim(v) and len(v) == len(D['hand_qpos']) else v) for k, v in D.items()}
    far['obj_centre_cm'] = far['obj_centre_cm'] + np.array([0., 0., 40.])      # 40 cm away
    lab2, dist2, _ = cj.grasp_labels(far, sc)
    # mj_geomDistance saturates at distmax (100 mm): far objects read exactly 100 mm
    assert (lab2 == 0).all() and (dist2 >= 100.0 - 1e-9).all()


# --------------------------------------------------------------------------------------------
# F->A closure test: validated on TRUE hammer grasps (the hand really holds a 50 mm handle)
# --------------------------------------------------------------------------------------------
@needs_data
def test_hand_closure_test_on_true_hammer_grasps():
    from e2 import hand_data as hd
    from e2._sealed_math import rot6d_to_matrix
    D = hd.load_hand('hammer')
    sc = cj.JudgeScene()
    lab, _, _ = cj.grasp_labels({k: (v[D['split'] == 'dev'] if np.ndim(v) and len(v) == len(D['hand_qpos']) else v) for k, v in D.items()}, sc)
    dev = np.flatnonzero(D['split'] == 'dev')
    rng = np.random.default_rng(0)
    g1 = dev[rng.choice(np.flatnonzero(lab == 1), 25, replace=False)]
    g0 = dev[rng.choice(np.flatnonzero(lab == 0), 25, replace=False)]

    def run(idx, c):
        P = D['position_cm'][idx]
        G = rot6d_to_matrix(torch.as_tensor(D['global_rot6'][idx])).numpy()
        return cj.hand_closure_test(P, G, np.full(len(idx), c), sc)['pass'].mean()
    true_g1 = run(g1, 5.0)
    big_g1 = run(g1, 8.0)
    true_g0 = run(g0, 5.0)
    assert true_g1 >= 0.85, true_g1          # the real handle size is holdable by the real grasping hand
    assert big_g1 <= 0.2, big_g1             # a 30 mm larger cylinder is not
    assert true_g0 <= 0.5, true_g0           # a free (open / away) hand mostly does not hold the handle size
    P = D['position_cm'][g1]
    G = rot6d_to_matrix(torch.as_tensor(D['global_rot6'][g1])).numpy()
    fit = cj.hand_fit_test(P, G, np.full(len(g1), 5.0), sc)
    assert np.array_equal(fit['fit'], fit['hold'] & ~fit['hold_plus_margin'])
    assert fit['fit'].mean() >= 0.75, fit['fit'].mean()


def test_hand_closure_rejects_flat_open_hand():
    """Rest pose (flat open hand, thumb far from the fingertips): a 5 cm cylinder cannot be pinched."""
    sc = cj.JudgeScene()
    q, P, G = _decoded(1, seed=0)
    sys.path.insert(0, str(ROOT / 'data'))
    from mujoco_dataset_adapter import MujocoMotionAdapter
    a = MujocoMotionAdapter(PEN_XML, 'forearm')
    dec = a.decode_q(a.model.qpos0[None].copy())
    r = cj.hand_closure_test(dec['position_native'] * 100, dec['global_matrices'], np.array([5.0]), sc)
    assert not r['pass'][0] and r['violation_mm'][0] > 0


# --------------------------------------------------------------------------------------------
# AN (analytic retarget)
# --------------------------------------------------------------------------------------------
def test_analytic_retarget_formula():
    q, P, G = _decoded(64, seed=4)
    P = torch.as_tensor(P, dtype=torch.float32)
    G = torch.as_tensor(G, dtype=torch.float32)
    out = an.analytic_clearance(P, G)
    c_A = op.adroit_descriptor(P, G)['c_A']
    assert torch.allclose(out['c_A'], c_A)
    assert torch.allclose(out['c_F'], c_A.clamp(0, 7.8))
    assert torch.allclose(out['q_left'], (out['c_F'] + 0.2) / 2) and torch.equal(out['q_left'], out['q_right'])
    assert float(out['slide_normalized'].max()) <= 1.0 and float(out['slide_normalized'].min()) >= 0.0
    # round trip through the Franka clearance definition
    slide = torch.zeros(64, op.FRANKA_J)
    slide[:, op.FRANKA_LEFT] = out['q_left']
    slide[:, op.FRANKA_RIGHT] = out['q_right']
    assert torch.allclose(op.franka_clearance(slide), out['c_F'], atol=1e-5)


@needs_data
def test_analytic_from_pose9_matches_tensor_path():
    from e2 import hand_data as hd
    D = hd.load_hand('hammer')
    x = D['pose9'][:300]
    c = an.analytic_from_pose9(x, float(D['reach_cm']), chunk=128)
    P = torch.as_tensor(D['position_cm'][:300])
    from e2._sealed_math import rot6d_to_matrix
    G = rot6d_to_matrix(torch.as_tensor(D['global_rot6'][:300]))
    ref = an.analytic_clearance(P, G)['c_F'].numpy()
    assert np.abs(c - ref).max() < 0.05      # pose9 positions are float32 /reach -> tiny rounding
