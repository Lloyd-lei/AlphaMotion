"""WP-RENDER E2 triptych tests: hand posing exactness, gripper slide geometry, J2 state rule, identity, short render."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from render import triptych_e2 as E2  # noqa: E402

needs_data = pytest.mark.skipif(not (E2.E2_DATA / 'hand_hammer.npz').is_file(), reason='E2 hand data not built')


def same_image(a, b, max_level=2, max_frac=2e-3):
    """Pixel identity up to GPU rasterisation noise: the same scene rendered twice in ONE EGL context differs by
    1 grey level on ~0.05% of pixels (multisample/shadow-map resolve), so exact equality is not a valid test."""
    a = np.asarray(a, np.int16)
    b = np.asarray(b, np.int16)
    if a.shape != b.shape:
        return False
    d = np.abs(a - b).max(-1)
    return int(d.max()) <= max_level and float((d > 0).mean()) <= max_frac


def test_state_rule_equals_closure_judge():
    cj = E2.e2mod('closure_judge')
    assert (cj.HOLD_LOW_MM, cj.HOLD_HIGH_MM) == (E2.HOLD_LOW_MM, E2.HOLD_HIGH_MM)
    c = np.linspace(-10, 100, 1101)
    for d in (30.0, 50.0, 70.0):
        assert np.array_equal(E2.gripper_state(c, d), cj.gripper_state(c, d))
    op = E2.e2mod('opposition')
    assert op.FRANKA_CLEARANCE_OFFSET_CM == E2.FRANKA_OFFSET_CM


def test_camera_from_forward_roundtrip():
    for f in ([1, 0, 0], [0, 1, -1], [-1, -1, -0.3]):
        az, el = E2.cam_from_forward(f)
        v = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                      np.sin(np.radians(el))])
        np.testing.assert_allclose(v, np.asarray(f) / np.linalg.norm(f), atol=1e-9)


@needs_data
def test_hand_posing_is_exact_and_gripper_is_slide_exact():
    cc, lab, _ = E2.load_arm_cF('constant_median', 'hammer') if (E2.E2_ARMS / 'constant_median').is_dir() else (None, None, None)
    h = E2.load_hand_episode('hammer', 1, stride=8, labels=lab)
    desc = E2.hand_descriptor(h)
    T = len(h.times)
    synth = np.linspace(-5.0, 90.0, T)   # sweeps crush -> hold -> open -> beyond stroke
    tri = E2.TriptychE2(h, desc, synth, synth, ('a', 'a'), 'test')
    try:
        assert max(tri.drawers[0].body_pos_check(t) for t in (0, T // 2, T - 1)) < 1e-3   # cm
        dr = tri.drawers[1]
        offs = np.array([dr.clearance_mm(t) - (2 * dr.q_cm[t] - E2.FRANKA_OFFSET_CM) * 10 for t in range(T)])
        assert offs.max() - offs.min() < 1e-6          # rendered gap follows the slide value exactly
        # fingertip pads touch-gap at c_F = 0 (geometry fact reported in the README)
        assert abs(dr.tip_gap_mm_at(0.0) - 3.0) < 0.05
        assert dr.clipped[-1] and not dr.clipped[T // 3]
        assert set(np.unique(dr.state)) == {-1, 0, 1}
        # fingers point down on screen: approach axis ~ world -z
        assert dr.a[2] < -0.95
        # identical control/test series -> identical gripper panels
        for t in (0, T // 2):
            a, b = tri.view(1, t), tri.view(2, t)
            assert same_image(a, b) and a.std() > 5
        img = np.asarray(tri.frame(T // 2))
        assert img.shape == (1080, 1920, 3)
        assert same_image(img[92:1040, 640:1280], img[92:1040, 1280:1920])  # footer spans all panels
    finally:
        tri.close()


@needs_data
def test_load_arm_alignment_check(tmp_path):
    with np.load(E2.E2_DATA / 'hand_hammer.npz') as z:
        n = len(z['episode_index'])
        ep = z['episode_index'].copy()
    np.savez(tmp_path / 'hammer.npz', c_F_mm=np.zeros(n, np.float32), episode=ep)
    c, lab, p = E2.load_arm_cF(str(tmp_path), 'hammer')
    assert len(c) == n and lab is None
    bad = ep.copy()
    bad[0] += 1
    np.savez(tmp_path / 'hammer.npz', c_F_mm=np.zeros(n, np.float32), episode=bad)
    with pytest.raises(ValueError):
        E2.load_arm_cF(str(tmp_path), 'hammer')


@needs_data
def test_short_full_render(tmp_path):
    h = E2.load_hand_episode('hammer', 1, stride=10, frames=(0, 120))
    c1 = np.full(len(h.times), 52.0)
    c2 = np.full(len(h.times), 75.0)
    man = E2.render_e2(h, c1, c2, tmp_path / 'e2', ('hold', 'open'), 'test')
    assert man['frames'] == len(h.times) == 12
    assert man['state_fractions']['control']['hold'] == 1.0 and man['state_fractions']['test']['open'] == 1.0
    assert man['time']['encoded_frames'] == 12
    assert abs(man['time']['duration_seconds'] - 12 * 0.1) < 0.01   # stride 10 at 100 Hz = 0.1 s per frame
    assert man['checks']['hand_body_pos_max_err_cm'] < 1e-3
    for f in ('triptych.mp4', 'grid12.png', 'poster.jpg', 'manifest.json'):
        assert (tmp_path / 'e2' / f).is_file()
    json.loads((tmp_path / 'e2' / 'manifest.json').read_text())
