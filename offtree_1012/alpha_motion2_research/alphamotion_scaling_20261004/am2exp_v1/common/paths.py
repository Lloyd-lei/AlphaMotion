"""Shared paths for am2exp_v1 (interfaces.md contract). First writer wins; others import.

NAME COLLISION WARNING: the sealed tree has top-level modules ``base/legacy/common.py`` and
``base/legacy/model.py`` (``from common import SkeletonSpec``, ``from model import hop_distance``
in pilot_data/model_dual/...). This package is also called ``common`` and WP-MODEL's package is
``model``.  Two safe ways to import am2exp_v1 code:

1. Preferred (WP-OPS code): put ROOT (not CODE) on sys.path and import with the package prefix,
   ``from am2exp_v1.common.paths import ROOT`` / ``am2exp_v1.model.slide_fix`` (``am2exp_v1`` is a
   namespace package; it never shadows a sealed module name).
2. Legacy (first writer's convention): load this file by location under a private name
   (see e2/_env.py), or import ``common.paths`` with CODE on sys.path and then call
   ``add_sealed_paths()``, which puts the sealed folders at the FRONT of sys.path and drops this
   package's ``common`` entry from sys.modules so sealed ``from common import ...`` resolves to
   base/legacy/common.py.  Import order that works:
   ``from common.paths import ...; add_sealed_paths(); import pilot_data``.
"""
from pathlib import Path
import sys

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
CODE = ROOT / 'am2exp_v1'
OUT = Path('/var/tmp/alphamotion/am2exp')
PY = '/home/arenalabs/miniconda3/envs/ardy/bin/python'
PYROT = '/home/arenalabs/miniconda3/envs/rot/bin/python'

SEALED_PATHS = [ROOT / 'orchestration', ROOT / 'base', ROOT / 'base' / 'legacy', ROOT, ROOT / 'prismatic']

# --- WP-OPS additions (additive only; nothing above changed meaning) -------------------------
# Folders under ROOT that must never be written (interfaces.md hard rules).
SEALED_DIRS = ['base', 'surface', 'prismatic', 'orchestration', 'evaluation', 'evaluation_p1',
               'evaluation_phase2_v2', 'continuation_v2', 'continuation_v3', 'runs', 'logs', 'data',
               'qwen', 'rollout_tools', 'reports']
# Single shared GPU lease (flock). Never delete or recreate: flock is bound to the inode.
GPU_LOCK = ROOT / 'gpu_queue.lock'
GPU_LOCK_INODE = 55451429
# Combined download plan identity checked by base/remote_state.capacity().
PLAN_IDENTITY_BYTES = 200_505_865_539
# Output sub-folders (all on the root partition, never on /home).
OUT_LOGS = OUT / 'logs'
OUT_RUNS = OUT / 'runs'
APPROVALS = OUT_LOGS / 'approvals'
# Environment every am2exp process should run with.
BASE_ENV = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false'}


def add_sealed_paths():
    """Insert sealed source folders read-only at the front of sys.path (idempotent)."""
    sys.dont_write_bytecode = True
    for p in reversed(SEALED_PATHS):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    mod = sys.modules.get('common')
    if mod is not None and getattr(mod, '__path__', None) is not None:
        if any(str(CODE) in str(x) for x in mod.__path__):
            del sys.modules['common']  # let the sealed legacy common.py win
    return [str(p) for p in SEALED_PATHS]


def sealed_source_hashes():
    """sha256 of every *.py under the sealed folders (train_scaling.source_hashes() superset).

    Used by WP-OPS acceptance: identical before and after a run proves nothing sealed was edited
    and no bytecode/py files were added there.
    """
    import hashlib
    out = {}
    for folder in SEALED_DIRS:
        base = ROOT / folder
        if not base.exists():
            continue
        for p in sorted(base.rglob('*.py')):
            if '__pycache__' in p.parts:
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p.relative_to(ROOT))] = h
    return out
