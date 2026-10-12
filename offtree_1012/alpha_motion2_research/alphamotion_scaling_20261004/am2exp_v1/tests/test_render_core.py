"""WP-RENDER core tests: 12-frame grid, contact states, robot binding, judge/local agreement (EGL / CPU only)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from render import _env  # noqa: E402
from render import contact as C  # noqa: E402
from render import grid12 as G  # noqa: E402
from render.case_io import load_gt_window  # noqa: E402

DEMO = ('lafan::fallAndGetUp1_subject4', 4136, 4264)


def _have_caches():
    try:
        _env.xwbc_body_entry('unitree_g1_29dof')
        return True
    except Exception:
        return False


needs_cache = pytest.mark.skipif(not _have_caches(), reason='XWBC GT caches unavailable')


# ------------------------------------------------------------------------------------------- grid
@pytest.mark.parametrize('T', [1, 5, 12, 13, 128, 451])
def test_grid_indices_span_whole_duration(T):
    idx = G.grid_indices(T, 12)
    assert idx[0] == 0 and idx[-1] == T - 1
    assert np.all(np.diff(idx) > 0)
    assert len(idx) == min(T, 12)


def test_grid_from_video_picks_exact_frames():
    """Encode 40 synthetic 1920x1080 frames (each a distinct flat grey level) through rollout_tools/encode (VFR) and
    check that every grid tile is the frame its label says."""
    from PIL import Image
    from render.triptych_e1 import _encode
    T = 40
    levels = np.linspace(10, 245, T).round().astype(int)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
        fr = Path(td) / 'f'
        fr.mkdir()
        for t in range(T):
            Image.new('RGB', (1920, 1080), (int(levels[t]),) * 3).save(fr / f'{t:06d}.png')
        times = np.arange(T) / 30.0
        timing = _encode(fr, times, Path(td) / 'v.mp4')
        assert timing['encoded_frames'] == T and timing['max_pts_error_seconds'] < 1e-5
        info = G.probe(Path(td) / 'v.mp4')
        assert (info['width'], info['height'], info['frames']) == (1920, 1080, T)
        idx = G.grid_indices(T)
        tiles = G.extract_frames(Path(td) / 'v.mp4', idx, 96, 54)
        got = np.array([np.asarray(t, np.float64).mean() for t in tiles])
        assert np.abs(got - levels[idx]).max() < 3.0
        out = G.grid_from_video(Path(td) / 'v.mp4', Path(td) / 'g.png', times=times, title='test')
        assert out['indices'] == [int(i) for i in idx]
        assert Path(out['out']).is_file()


def test_grid_from_frames_in_memory(tmp_path):
    frames = [np.full((108, 192, 3), i * 10, np.uint8) for i in range(20)]
    out = G.grid_from_frames(frames, tmp_path / 'g.png', times=np.arange(20) / 10.0)
    assert out['indices'][0] == 0 and out['indices'][-1] == 19


# ------------------------------------------------------------------------------------------- contact
def test_states_thresholds():
    h = np.array([[10.0, 2.49, 2.5, -0.99, -1.0, -1.01, np.nan]])
    assert C.states(h).tolist() == [[0, 1, 0, 1, 1, 2, 0]]
    assert C.TOUCH_CM == 2.5 and C.PEN_CM == 1.0


def test_human_fallback_heights_follow_root():
    from render.case_io import Window
    names, parents, off, reach = _env.static_spec('human_smpl')
    R = np.tile(np.eye(3), (3, len(names), 1, 1))
    P = _env.fk_positions(R, parents, off)
    w = Window('human_smpl', R, P, np.array([1.0, 1.0, 1.0]), names=names, parents=parents, reach=reach)
    root = np.array([100.0, 105.0, 90.0])
    rh, zmin, st, src = C.human_region_heights(w, root, None)
    assert 'nominal' in src
    np.testing.assert_allclose(zmin - zmin[0], root - root[0], atol=1e-9)


# ------------------------------------------------------------------------------------------- robots
@needs_cache
@pytest.mark.parametrize('body', _env.ROBOTS_XWBC)
def test_robot_binding_matches_cache_skeleton(body):
    from render.robot_model import load_robot
    rm = load_robot(body)
    names, parents, off, reach = _env.static_spec(body)
    assert list(rm.names) == list(names)
    assert np.array_equal(rm.parents, parents)
    assert len(rm.visual) > 5 and rm.collision is not None and len(rm.collision) > 5
    a = rm.audit
    if body == 'unitree_h2':
        # DATA_TREE body: anonymous cache skeleton mapped to the vendor MJCF by the judge's explicit node map;
        # the cache offsets differ from the vendor MJCF by up to ~6 cm (documented in render/README.md)
        assert a['mode'] == 'data_tree_explicit_node_map'
        assert a['random_q_fk_max_err_cm'] < 8.0
    else:
        assert a['spec_offset_max_err_cm'] < 1e-3
        assert a['random_q_fk_max_err_cm'] < 0.1


@needs_cache
@pytest.mark.parametrize('body', ['unitree_g1_29dof', 'fourier_gr3'])
def test_local_lowest_point_equals_judge(body):
    """The renderer's own collision lowest-point (fallback) and judge J1 give the same region heights."""
    from render.robot_model import load_robot
    gj = _env.judge_module('ground_judge')
    if gj is None:
        pytest.skip('judge package unavailable')
    w = load_gt_window(DEMO[0], body, DEMO[1], DEMO[1] + 32)
    rm = load_robot(body)
    ry = 100 * w.root_h_m - _env.root_offset_cm(body, 'gt')[0]
    rhL, zL, srcL = C.robot_region_heights(rm, w.R, w.P, ry, 'local')
    rhJ, zJ, srcJ = C.robot_region_heights(rm, w.R, w.P, ry, 'judge')
    assert 'judge' in srcJ and 'local' in srcL
    np.testing.assert_array_equal(np.isnan(rhL), np.isnan(rhJ))
    assert np.nanmax(np.abs(rhL - rhJ)) < 1e-3
    assert np.abs(zL - zJ).max() < 1e-3
    # a +5 cm root shift moves every region by +5 cm
    rh5, _, _ = C.robot_region_heights(rm, w.R, w.P, ry + 5.0, 'local')
    assert np.nanmax(np.abs(rh5 - rhL - 5.0)) < 1e-9


@needs_cache
def test_root_offset_convention_equals_judge():
    gj = _env.judge_module('ground_judge')
    if gj is None:
        pytest.skip('judge package unavailable')
    for b in _env.ROBOTS_XWBC:
        v, tag, _ = _env.root_offset_cm(b, 'gt')
        r = gj.corrected_root(b, np.array([1.0]), 'gt')
        assert abs((1.0 - r[0]) * 100 - v) < 1e-9, b


def test_render_never_imports_dense_asset_for_robots():
    src = (CODE / 'render' / 'contact.py').read_text()
    # dense surface is used only for the HUMAN picture/colouring; robot colours come from collision geoms
    assert "robot_region_heights" in src and "surface_v2" in src
    assert src.index('def robot_region_heights') < src.index('def _surface_module')
    body = src[src.index('def robot_region_heights'):src.index('def _surface_module')]
    assert 'surface' not in body


def test_cli_help_runs():
    for mod in ('render.triptych_e1', 'render.triptych_e2', 'render.grid12'):
        r = subprocess.run([sys.executable, '-m', mod, '--help'], cwd=str(CODE), capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl'))
        assert r.returncode == 0, r.stderr


def test_human_dense_region_heights_synthetic():
    """Dense-surface path (used when WP-ASSETS human_smpl.npz exists): region min over its points + root."""
    from render.case_io import Window
    T, Pn = 4, 300
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(T, Pn, 3)) * 20
    reg = rng.integers(0, 18, Pn)
    root = np.array([90.0, 95.0, 100.0, 80.0])
    w = Window('human_smpl', None, np.zeros((T, 3, 3)), None, names=['a', 'b', 'c'], parents=np.array([-1, 0, 1]))
    rh, zmin, st, src = C.human_region_heights(w, root, (pts, reg))
    assert st is None and 'dense' in src
    for r in range(15):
        m = reg == r
        np.testing.assert_allclose(rh[:, r], pts[:, m, 1].min(-1) + root)
