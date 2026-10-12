"""Path/env helper for WP-ASSETS.

Loads CODE/common/paths.py by file location under a private module name (same pattern as e2/_env.py) so this
package never registers the top-level name ``common``.  The build needs two sealed import worlds that must not
meet in one process: (1) surface/build_verified_patches.py -> pose_translator_design/exp_topo (its own
``common.py``), and (2) orchestration/scaling_data -> base/legacy/common.py.  World (2) is only ever used in a
subprocess (dump_am2_bodies.py) whose JSON output the build reads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

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
PY: str = str(_P.PY)

ASSET_DIR = OUT / 'assets' / 'surface_v2'
CACHE_DIR = ASSET_DIR / '_cache'
PNG_DIR = ASSET_DIR / 'png'
CONFIG_PATH = ROOT / 'orchestration' / 'configs' / 'A_S_seed0.json'
SMPLX_NPZ = Path('/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz')
REPORT32 = Path('/home/arenalabs/Desktop/alphamotion video/resource/report/32_ground_gauge_audit')
SOLE_AUDIT = REPORT32 / 'sole_audit.json'
PILOT_XWBC = ROOT.parent / 'megatron_scale_pilot_20261004' / 'data_xwbc'
H2_XML = Path('/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml')
XWBC_EXTRACTED = Path('/var/tmp/alphamotion/xwbc_lafan/extracted')

BODIES_XWBC_TRAIN = ['human_smpl', 'booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi',
                     'pnd_adam_lite', 'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2']
BODIES_XWBC_HELDOUT = ['fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1']
BODIES_EXTERNAL = ['adroit_pen', 'relay_franka_full_v2']
ALL_BODIES = BODIES_XWBC_TRAIN + BODIES_XWBC_HELDOUT + BODIES_EXTERNAL
ROBOTS = [b for b in ALL_BODIES if b != 'human_smpl']

REGIONS = ['pelvis', 'trunk', 'head', 'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand',
           'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot', 'digit', 'base_or_link', 'palm']
RID = {n: i for i, n in enumerate(REGIONS)}


def sealed_surface_path():
    """Make surface/build_verified_patches.py importable (read-only)."""
    sys.dont_write_bytecode = True
    p = str(ROOT / 'surface')
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=_jsonable, allow_nan=True) + '\n')
    tmp.replace(path)


def _jsonable(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def cpu_env():
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    return env
