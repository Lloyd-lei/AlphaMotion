"""WP-E2 data tests (CPU).  Need OUT/data/e2 produced by `e2/hand_data.py all`."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2 import hand_data as hd  # noqa: E402
from e2._env import E2_DATA, ADROIT_PAYLOAD  # noqa: E402

needs_data = pytest.mark.skipif(not (E2_DATA / 'hand_data_manifest.json').exists(), reason='run e2/hand_data.py all first')


def test_hand_joint_order_identical_across_xmls():
    r = hd.verify_hand_joint_order()
    assert r['passed'], r
    assert r['layouts']['pen']['names'] == hd.HAND_JOINTS
    assert r['layouts']['hammer']['qposadr'][0] == 2 and r['layouts']['relocate']['qposadr'][0] == 6 and r['layouts']['pen']['qposadr'][0] == 0
    assert r['hand_subtree_forearm_frame_max_abs_diff'] < 1e-9


def test_split_seeded_10_dev_15_test():
    for ds in ['hammer', 'relocate']:
        s1 = hd.episode_split(ds, 25)
        s2 = hd.episode_split(ds, 25)
        assert (s1 == s2).all()
        assert (s1 == 'dev').sum() == 10 and (s1 == 'test').sum() == 15
    assert not (hd.episode_split('hammer', 25) == hd.episode_split('relocate', 25)).all()


def test_euler_roundtrip_and_sync_helpers():
    rng = np.random.default_rng(0)
    E = rng.uniform(-1.4, 1.4, (200, 3))
    R = hd.euler2mat_gym(E)
    assert np.allclose(np.einsum('nij,nkj->nik', R, R), np.eye(3), atol=1e-12)
    assert np.allclose(hd.mat2euler_gym(R), E, atol=1e-10)
    from scipy.spatial.transform import Rotation
    assert np.allclose(R, Rotation.from_euler('XYZ', E).as_matrix(), atol=1e-12)  # gym convention == intrinsic XYZ
    q = np.arange(10, dtype=float)[:, None] * np.ones((1, 3))
    ep = np.array([0] * 5 + [1] * 5)
    s = hd.sync_to_obs_time(q, ep, 0.2)
    assert s[0, 0] == 0 and s[5, 0] == 5            # episode starts untouched
    assert np.isclose(s[1, 0], 0.8) and np.isclose(s[9, 0], 8.8)
    x = hd.extrapolate_from_obs_time(q, ep, 0.2)
    assert np.isclose(x[1, 0], 1.2) and x[5, 0] == 5


@needs_data
def test_manifest_acceptance_numbers():
    m = json.loads((E2_DATA / 'hand_data_manifest.json').read_text())
    rel = m['datasets']['relocate']['checks']
    assert rel['palm_minus_obj_vs_obs_max_mm'] < 1.0          # acceptance: relocate palm - object vector within 1 mm
    assert rel['init_object_pos_max_mm'] < 1e-6
    assert rel['obs_internal_consistency_max_mm'] < 1e-6
    ham = m['datasets']['hammer']['checks']
    assert ham['init_object_pos_max_mm'] < 1e-6 and ham['init_object_rot_max_abs'] < 1e-9
    assert ham['palm_site_vs_obs_max_mm'] < 2.0
    for ds in ['hammer', 'relocate']:
        d = m['datasets'][ds]
        assert d['checks']['palm_pen_model_vs_transformed_source_max_mm'] < 1e-6
        assert d['fk_max_err_cm'] < 1e-6
        assert len(d['episode_split']['dev']) == 10 and len(d['episode_split']['test']) == 15
    assert m['pen']['payload_qpos_max_abs_diff'] == 0.0
    assert m['joint_order_check']['passed']


@needs_data
def test_converted_arrays_consistent():
    from e2._sealed_math import rot6d_to_matrix
    import torch
    for ds in ['hammer', 'relocate']:
        D = hd.load_hand(ds)
        N = len(D['hand_qpos'])
        assert D['pose9'].shape == (N, 25, 9) and D['obj_centre_cm'].shape == (N, 3)
        # pose9 positions = FK(rot6, offsets) / reach
        G = rot6d_to_matrix(torch.as_tensor(D['global_rot6'][:500])).numpy().astype(np.float64)
        off = D['rest_offsets_cm'].astype(np.float64)
        P = np.zeros((500, 25, 3))
        for j, p in enumerate(D['parents']):
            if p >= 0:
                P[:, j] = P[:, p] + np.einsum('nij,j->ni', G[:, p], off[j])
        assert np.abs(P - D['pose9'][:500, :, 6:] * float(D['reach_cm'])).max() < 1e-3
        assert np.allclose(D['pose9'][..., :6], D['global_rot6'])
        # forearm root fixed at the origin with identity rotation
        assert np.abs(D['position_cm'][:, 0]).max() == 0
        R0 = rot6d_to_matrix(torch.as_tensor(D['global_rot6'][:, 0])).numpy()
        assert np.abs(R0 - np.eye(3)).max() < 1e-5
        # object geometry
        seg = D['obj_segment_cm']
        L = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        assert np.allclose(L, 2 * float(D['obj_half_length_cm']), atol=1e-3)


@needs_data
def test_relocate_palm_object_vector_direct():
    """Recompute palm(FK, pen model) - ball(canonical) and compare with obs[30:33] rotated to canonical."""
    import mujoco
    D = hd.load_hand('relocate')
    raw = dict(np.load(E2_DATA / 'raw' / 'relocate_raw.npz'))
    sc = hd.SourceScene('pen')
    sc.forward(sc.m.qpos0)
    P_fa = sc.d.xpos[sc.fa].copy()
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(D['hand_qpos']), 400, replace=False))
    errs = []
    for i in idx:
        q = sc.m.qpos0.copy()
        q[:24] = D['hand_qpos'][i]
        d = sc.forward(q)
        palm_can = hd.AX @ (d.site_xpos[sc.palm_site] - P_fa) * 100
        v_can = palm_can - D['obj_body_cm'][i]
        # obs vector is in source world; map with the same forearm-relative rotation
        Rfa = D['src_forearm_R_w'][i].astype(np.float64)
        T = hd.AX @ sc.d.xmat[sc.fa].reshape(3, 3) @ Rfa.T
        v_obs = T @ raw['observations'][i, 30:33] * 100
        errs.append(np.linalg.norm(v_can - v_obs) * 10)
    assert max(errs) < 1.0, max(errs)


@needs_data
def test_pen_side_file_matches_payload():
    side = dict(np.load(E2_DATA / 'pen_object.npz'))
    with np.load(ADROIT_PAYLOAD) as z:
        q = z['recorded_robot_qpos']
        split = z['split']
    assert np.array_equal(side['hand_qpos'], q)
    assert np.array_equal(side['payload_split'], split)
    ax = side['obj_axis_can']
    assert np.allclose(np.linalg.norm(ax, axis=1), 1, atol=1e-5)


@needs_data
def test_hammer_euler_convention_on_nail_impacts():
    """Hammer head must land on the nail when the nail touch sensor fires (gym euler), not with other orders."""
    from scipy.spatial.transform import Rotation
    raw = dict(np.load(E2_DATA / 'raw' / 'hammer_raw.npz'))
    obs = raw['observations']
    imp = obs[:, 45] > 0
    p, e, tgt = obs[:, 36:39], obs[:, 39:42], obs[:, 42:45]

    def head_dist(R):
        head = p + np.einsum('nij,j->ni', R, [-0.24, 0, 0])
        ax = R[:, :, 2]
        a, b = head - 0.04 * ax, head + 0.04 * ax
        ab = b - a
        t = np.clip(((tgt - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        return np.linalg.norm(tgt - (a + t[:, None] * ab), axis=1) - 0.02
    good = np.median(head_dist(hd.euler2mat_gym(e))[imp])
    bad = np.median(head_dist(Rotation.from_euler('xyz', e).as_matrix())[imp])
    assert good < 0.01 and bad > 0.1
