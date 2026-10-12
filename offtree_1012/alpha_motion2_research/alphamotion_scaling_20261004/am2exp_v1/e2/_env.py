"""Path/env helper for e2 (works in both the ardy env and the h5py-only rot env).

Loads CODE/common/paths.py by file location under a private module name so this
package never registers the name ``common`` (the sealed tree has base/legacy/common.py,
imported as ``from common import ...``).  ``sealed()`` puts the sealed folders on
sys.path ahead of CODE.
"""
from __future__ import annotations
import importlib.util
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
PYROT: str = str(_P.PYROT)

E2_DATA = OUT / 'data' / 'e2'
E2_EVAL = OUT / 'eval' / 'e2'

ADROIT_ASSETS = ROOT / 'data/models/Gymnasium-Robotics/gymnasium_robotics/envs/assets/adroit_hand'
PEN_XML = ADROIT_ASSETS / 'adroit_pen.xml'
HAMMER_XML = ADROIT_ASSETS / 'adroit_hammer.xml'
RELOCATE_XML = ADROIT_ASSETS / 'adroit_relocate.xml'
FRANKA_XML = ROOT / 'data/models/relay_robot_only_derived.xml'
PEN_HDF5 = ROOT / 'data/minari/pen/human-v2/data/main_data.hdf5'
ADROIT_PAYLOAD = ROOT / 'data/payloads/adroit_pen.npz'
ADROIT_PAYLOAD_JSON = ROOT / 'data/payloads/adroit_pen.json'
FRANKA_PAYLOAD = ROOT / 'data/payloads/relay_franka_full_v2.npz'
FRANKA_PAYLOAD_JSON = ROOT / 'data/payloads/relay_franka_full_v2.json'
SURFACE_V2 = OUT / 'assets' / 'surface_v2'


def sealed():
    """Make sealed modules importable (read-only); sealed legacy ``common`` wins over CODE."""
    sys.dont_write_bytecode = True
    _P.add_sealed_paths()
    legacy = str(ROOT / 'base' / 'legacy')
    # Defensive: whatever add_sealed_paths did, legacy/common.py must resolve before CODE/common.
    for entry in (str(CODE), str(CODE) + '/'):
        if entry in sys.path and sys.path.index(entry) < sys.path.index(legacy):
            sys.path.remove(entry)
            sys.path.append(entry)
    mod = sys.modules.get('common')
    if mod is not None and getattr(mod, '__path__', None) is not None:
        del sys.modules['common']
    return True
