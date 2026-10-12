"""WP-RENDER E1 triptych tests: case npz round trip, GT|GT|GT identity, synthetic predictions, short full render."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from render import _env  # noqa: E402
from render import triptych_e1 as E1  # noqa: E402
from render.case_io import Window, load_case, load_gt_window, window_to_case_npz  # noqa: E402

GROUP, S0 = 'lafan::fallAndGetUp1_subject4', 4136
T = 24


def _have_caches():
    try:
        _env.xwbc_body_entry('fourier_gr3')
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_caches(), reason='XWBC GT caches unavailable')


def same_image(a, b, max_level=2, max_frac=2e-3):
    """Pixel identity up to GPU rasterisation noise: the same scene rendered twice in ONE EGL context differs by
    1 grey level on ~0.05% of pixels (multisample/shadow-map resolve), so exact equality is not a valid test."""
    a = np.asarray(a, np.int16)
    b = np.asarray(b, np.int16)
    if a.shape != b.shape:
        return False
    d = np.abs(a - b).max(-1)
    return int(d.max()) <= max_level and float((d > 0).mean()) <= max_frac


def _rand_rot(rng, n, deg):
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True) * np.radians(deg) * rng.uniform(0, 1, (n, 1))
    out = []
    for w in v:
        th = np.linalg.norm(w)
        k = w / max(th, 1e-12)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        out.append(np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
    return np.stack(out)


@pytest.fixture(scope='module')
def gt_case(tmp_path_factory):
    d = tmp_path_factory.mktemp('e1')
    src = load_gt_window(GROUP, 'human_smpl', S0, S0 + T)
    gt = load_gt_window(GROUP, 'fourier_gr3', S0, S0 + T)
    meta = {'group_id': GROUP, 'clip': 'fallAndGetUp1_subject4', 'start': S0, 'end': S0 + T, 'fps': 30.0, 'arm': 'GT',
            'mode': 'sensor_on'}
    p = d / 'gt_case.npz'
    window_to_case_npz(p, src, gt, None, meta)
    # synthetic "prediction": GT rotations perturbed by up to 15 deg per joint, FK-consistent positions, root -3 cm
    rng = np.random.default_rng(0)
    names, parents, off, reach = _env.static_spec('fourier_gr3')
    R = gt.R @ _rand_rot(rng, T * len(names), 15).reshape(T, len(names), 3, 3)
    P = _env.fk_positions(R, parents, off)
    pred = Window('fourier_gr3', R, P, gt.root_h_m - 0.03, names=names, parents=parents, reach=reach)
    q = d / 'synthetic_case.npz'
    window_to_case_npz(q, src, gt, pred, dict(meta, arm='synthetic'))
    return dict(dir=d, gt=p, syn=q, src=src, win=gt)


def test_case_roundtrip(gt_case):
    c = load_case(gt_case['gt'])
    assert c['source_body'] == 'human_smpl' and c['target_body'] == 'fourier_gr3'
    assert c['meta']['start'] == S0
    np.testing.assert_allclose(c['pred'].R, gt_case['win'].R, atol=1e-5)
    np.testing.assert_allclose(c['gt'].P, gt_case['win'].P, atol=1e-3)
    np.testing.assert_allclose(c['source'].P, gt_case['src'].P, atol=1e-3)
    # source placement re-read from the frozen cache by meta (root height + world trajectory for the follow cam)
    np.testing.assert_allclose(c['source'].root_h_m, gt_case['src'].root_h_m)
    assert c['source'].world_root_cm is not None
    # GT positions are the FK of GT rotations (canonical convention sanity)
    names, parents, off, _ = _env.static_spec('fourier_gr3')
    assert np.abs(_env.fk_positions(c['gt'].R, parents, off) - c['gt'].P).max() < 0.05


def test_gt_self_render_panels_identical(gt_case):
    """[human | GT | GT]: the two arm panels must be pixel-identical (same pose, root, colours, camera, trace)."""
    c = load_case(gt_case['gt'])
    panels, src = E1.panels_from_cases(c, c, 'GT', 'GT', control_is_gt=True, test_is_gt=True)
    tri = E1.Triptych(panels, src, c['times'], 'test')
    try:
        for t in (0, T // 2, T - 1):
            a, b = tri.view(1, t), tri.view(2, t)
            assert a.shape == (E1.VIEW_H, E1.PANEL_W, 3)
            assert same_image(a, b)
            assert a.std() > 5   # not blank
            img = np.asarray(tri.frame(t))
            assert img.shape == (1080, 1920, 3)
            assert same_image(img[92:1040, 640:1280], img[92:1040, 1280:1920])  # footer spans all panels
        assert np.array_equal(panels[1].state, panels[2].state)
        assert np.array_equal(tri.drawers[1].pos, tri.drawers[2].pos) and np.array_equal(tri.drawers[1].mat, tri.drawers[2].mat)
    finally:
        tri.close()


def test_gt_cache_equals_case_path_render(gt_case):
    """Panel built straight from the cache window == panel built from the case npz (float32 round trip)."""
    c = load_case(gt_case['gt'])
    o, _ = E1.default_root_offset('fourier_gr3', 'gt')
    p_case = E1.PanelSpec('robot', 'x', c['gt'], o)
    p_cache = E1.PanelSpec('robot', 'x', gt_case['win'], o)
    E1.prepare([p_case, p_cache], c['source'], 30.0)
    assert np.abs(p_case.root_y - p_cache.root_y).max() < 1e-4
    assert np.nanmax(np.abs(p_case.region_h - p_cache.region_h)) < 0.01


def test_synthetic_prediction_differs_and_ghost_drawn(gt_case):
    c = load_case(gt_case['syn'])
    panels, src = E1.panels_from_cases(c, c, 'GT', 'synthetic', control_is_gt=True)
    tri = E1.Triptych(panels, src, c['times'], 'test')
    try:
        a, b = tri.view(1, T // 2), tri.view(2, T // 2)
        assert not same_image(a, b)
        assert panels[2].ghost is not None and panels[2].ghost_zmin is not None
        # prediction root 3 cm lower -> its trace sits below the GT ghost on average (rotations also differ)
        assert np.isfinite(panels[2].zmin).all()
    finally:
        tri.close()


def test_short_full_render_mp4(gt_case, tmp_path):
    out = tmp_path / 'tri'
    r = subprocess.run([sys.executable, '-m', 'render.triptych_e1', '--stage0', str(gt_case['syn']), '--out', str(out),
                        '--max-seconds', '600'], cwd=str(CODE), capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl'))
    assert r.returncode == 0, r.stderr[-3000:]
    man = json.loads((out / 'manifest.json').read_text())
    assert man['frames'] == T
    assert man['time']['encoded_frames'] == T and man['time']['max_pts_error_seconds'] < 1e-5
    assert abs(man['time']['duration_seconds'] - T / 30.0) < 0.01
    assert (out / 'grid12.png').is_file() and (out / 'poster.jpg').is_file()
    from render.grid12 import probe
    info = probe(out / 'triptych.mp4')
    assert (info['width'], info['height'], info['frames']) == (1920, 1080, T)
    # evidence is never overwritten
    r2 = subprocess.run([sys.executable, '-m', 'render.triptych_e1', '--stage0', str(gt_case['syn']), '--out', str(out)],
                        cwd=str(CODE), capture_output=True, text=True, timeout=600,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl'))
    assert r2.returncode != 0 and 'exists' in r2.stderr
