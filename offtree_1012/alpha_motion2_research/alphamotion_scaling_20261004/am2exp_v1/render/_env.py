"""Paths, read-only imports of sealed / external helper modules, and small shared math.

The render package never registers the module name ``common`` (sealed code imports
``from common import ...`` = base/legacy/common.py); CODE/common/paths.py is loaded by
file location under a private name, exactly like e2/_env.py does.

Everything imported from outside CODE is imported READ-ONLY (PYTHONDONTWRITEBYTECODE
is forced so no __pycache__ appears next to sealed sources).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

_CODE = Path(__file__).resolve().parents[1]


def _load_paths():
    for name in ('common.paths', 'am2exp_common_paths'):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, 'ROOT'):
            return mod
    spec = importlib.util.spec_from_file_location('am2exp_common_paths', _CODE / 'common' / 'paths.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['am2exp_common_paths'] = mod
    spec.loader.exec_module(mod)
    return mod


_P = _load_paths()
ROOT: Path = Path(_P.ROOT)
CODE: Path = Path(_P.CODE)
OUT: Path = Path(_P.OUT)
VIDEOS: Path = OUT / 'videos'
ASSETS_SURFACE_V2: Path = OUT / 'assets' / 'surface_v2'
EXP_TOPO = Path('/home/arenalabs/Desktop') / '"be water, robot"' / 'pose_translator_design' / 'exp_topo'
REPORT32 = Path('/home/arenalabs/Desktop/alphamotion video/resource/report/32_ground_gauge_audit')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
DATA_MANIFEST = ROOT / 'data' / 'data_manifest_v2.json'

# Canonical (Y-up, cm) <- MuJoCo world (Z-up, m):  c = AX @ w * 100 ;  w = AX.T @ c / 100
AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], np.float64)

HUMAN = 'human_smpl'
BODIES_XWBC_TRAIN = ['human_smpl', 'booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi',
                     'pnd_adam_lite', 'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2']
BODIES_XWBC_HELDOUT = ['fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1']
ROBOTS_XWBC = [b for b in BODIES_XWBC_TRAIN + BODIES_XWBC_HELDOUT if b != HUMAN]

# Report-32 collision-geometry root bias (label sole drop - vendor MJCF collision drop), cm.
# README line 71 (H2 is listed there as -5.4: label has no sole offset, ankle centre used as sole).
# Used only when the per-body asset manifest (WP-ASSETS delta_col_cm) is absent.
REPORT32_DELTA_COL_FALLBACK = {'unitree_h2': -5.4}


def load_module(name: str, path: Path):
    """Import a python file read-only under a private module name."""
    key = f'_am2render_ext_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=8)
def judge_module(name: str):
    """CODE/judge/<name>.py imported as ``am2exp_v1.judge.<name>`` (the judge's own absolute-import convention),
    or None when the judge package is absent / broken.  ROOT is APPENDED to sys.path so nothing is shadowed."""
    if not (CODE / 'judge' / f'{name}.py').is_file():
        return None
    root = str(ROOT)
    if root not in sys.path:
        sys.path.append(root)
    try:
        return importlib.import_module(f'am2exp_v1.judge.{name}')
    except Exception:  # pragma: no cover - depends on the sibling package state
        return None


@lru_cache(maxsize=1)
def trajectory_reader():
    """ROOT/data/trajectory_reader.py (sealed, read-only)."""
    return load_module('trajectory_reader', ROOT / 'data' / 'trajectory_reader.py')


@lru_cache(maxsize=1)
def exp_topo():
    """(prep_xwbc_cache, prep_gmr_cache) from pose_translator_design/exp_topo, read-only.
    Used only for MJCF resolution (resolve_xml) and the cached-frame merge rule (build_merge,
    build_spec) that produced the XWBC cache skeletons."""
    p = str(EXP_TOPO)
    if p not in sys.path:
        sys.path.append(p)  # appended (not prepended) so nothing in CODE is shadowed
    import prep_gmr_cache as PG  # noqa: E402
    import prep_xwbc_cache as PX  # noqa: E402
    return PX, PG


@lru_cache(maxsize=1)
def data_manifest() -> dict:
    return json.loads(DATA_MANIFEST.read_text())


def xwbc_body_entry(body: str) -> dict:
    return next(b for b in data_manifest()['xwbc']['bodies'] if b['body'] == body)


@lru_cache(maxsize=32)
def static_spec(body: str):
    """(joint_names list, parents int64[J], rest_offsets float32[J,3] cm, reach cm) of an XWBC body,
    exactly as trajectory_reader / load_scaling_bodies read them."""
    b = xwbc_body_entry(body)
    with np.load(b['static_spec'], allow_pickle=True) as z:
        names = [str(x) for x in z['joint_names']]
        parents = np.asarray(z['parents'], np.int64)
        off = np.asarray(z['rest_offsets'], np.float32)
    reach = trajectory_reader().static_reach(parents, off)
    return names, parents, off, float(reach)


def rot6_to_matrix(x) -> np.ndarray:
    """[...,6] -> [...,3,3] Gram-Schmidt, columns (b1,b2,b3); identical to base/legacy/common.rot6d_to_matrix
    and data/trajectory_reader.rot6matrix."""
    x = np.asarray(x, np.float64)
    a = x[..., :3]
    b = x[..., 3:6]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def fk_positions(R: np.ndarray, parents, offsets) -> np.ndarray:
    """Own-joint global rotations [T,J,3,3] -> root-relative positions [T,J,3] (base/pilot_data.fk)."""
    R = np.asarray(R, np.float64)
    off = np.asarray(offsets, np.float64)
    P = np.zeros(R.shape[:-1], np.float64)
    for j, p in enumerate(parents):
        if p >= 0:
            P[:, j] = P[:, p] + np.einsum('tab,b->ta', R[:, p], off[j])
    return P


@lru_cache(maxsize=1)
def report32_sole_audit() -> dict:
    p = REPORT32 / 'sole_audit.json'
    return json.loads(p.read_text()) if p.is_file() else {}


def asset_manifest(body: str) -> dict | None:
    p = ASSETS_SURFACE_V2 / f'{body}.json'
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def root_offset_cm(body: str, kind: str = 'gt') -> tuple[float, str, str]:
    """Root-height convention used by the renderer = judge J1's ``corrected_root`` (final_plan.md "Root convention"),
    expressed as a constant subtracted from 100*root_height_m.  kind: 'gt' | 'pred' | 'raw'.
    Returns (offset_cm, plain-word tag, provenance).
      human                          -> 0 (label root over the joint-centre floor)
      GT (any robot), pred on a training robot -> judge DELTA_COL_CM[body] (report-32 collision bias, R1 corrected)
      pred on a held-out robot       -> judge heldout constant c_b (A0 stance frames) when measured, else the GT
                                        convention, flagged provisional
    Without CODE/judge the report-32 table (``delta_col_cm``) is used and flagged."""
    if kind == 'raw':
        return 0.0, 'raw', 'no correction requested'
    if body == HUMAN:
        return 0.0, 'human', 'human: label root over the joint-centre floor, no correction'
    gj = judge_module('ground_judge')
    if gj is not None and body in getattr(gj, 'DELTA_COL_CM', {}):
        if kind == 'pred' and body in BODIES_XWBC_HELDOUT:
            try:
                hb = gj.load_heldout_bias()
            except Exception:
                hb = None
            if hb is not None and body in hb:
                return float(hb[body]), 'heldout_c', 'judge heldout_root_bias_v1.json (A0 stance frames, TRAIN windows)'
            return float(gj.DELTA_COL_CM[body]), 'heldout_provisional', \
                'judge heldout constant not measured yet -> GT convention (provisional)'
        return float(gj.DELTA_COL_CM[body]), 'delta_col', 'judge J1 DELTA_COL_CM (report-32 predicted_bias_cm, R1 J1-corrected)'
    v, src = delta_col_cm(body)
    return v, 'delta_col_fallback', src + ' (judge unavailable)'


def delta_col_cm(body: str) -> tuple[float, str]:
    """Collision-geometry label bias in cm and where it came from.
    Order: WP-ASSETS manifest delta_col_cm -> report-32 sole_audit predicted_bias_cm -> README table -> 0."""
    m = asset_manifest(body)
    if m is not None and m.get('delta_col_cm') is not None:
        try:
            v = float(m['delta_col_cm'])
            if np.isfinite(v):
                return v, 'assets/surface_v2 manifest delta_col_cm'
        except (TypeError, ValueError):
            pass
    if body == HUMAN:
        return 0.0, 'human: joint-centre label floor, no correction'
    row = report32_sole_audit().get(body, {})
    v = row.get('predicted_bias_cm') if isinstance(row, dict) else None
    if v is not None:
        return float(v), 'report-32 sole_audit.json predicted_bias_cm'
    if body in REPORT32_DELTA_COL_FALLBACK:
        return REPORT32_DELTA_COL_FALLBACK[body], 'report-32 README table (H2: missing sole offset)'
    return 0.0, 'unknown -> 0'
