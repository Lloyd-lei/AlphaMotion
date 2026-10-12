"""Read-only access to the sealed AM2 modules.

IMPORT NAME WARNING: the sealed tree has a top-level module ``base/legacy/model.py``
(``from model import EncBlock`` inside model_dual.py).  This package is therefore
imported as ``am2exp_v1.model`` (ROOT on sys.path, ``am2exp_v1`` is a namespace
package), never as a bare ``model``.  All intra-package imports are relative.
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'am2exp_v1'
OUT = Path('/var/tmp/alphamotion/am2exp')
SEALED_PATHS = [ROOT / 'orchestration', ROOT / 'base', ROOT / 'base' / 'legacy', ROOT, ROOT / 'prismatic']


def add_sealed_paths():
    """Put the sealed folders at the front of sys.path (idempotent, same order as common.paths)."""
    try:  # prefer the shared helper (first-writer-wins contract)
        from am2exp_v1.common.paths import add_sealed_paths as shared
        return shared()
    except Exception:  # pragma: no cover - fallback with identical semantics
        for p in reversed(SEALED_PATHS):
            s = str(p)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
        return [str(p) for p in SEALED_PATHS]


add_sealed_paths()
_legacy_model = sys.modules.get('model')
if _legacy_model is not None and getattr(_legacy_model, '__path__', None) is not None:
    # someone imported CODE/model as a bare top-level package: the sealed code needs the
    # legacy module under that name.
    raise ImportError('import this package as am2exp_v1.model; bare "model" must stay base/legacy/model.py')

import torch  # noqa: E402
import pilot_model  # noqa: E402  (sealed, read-only)
from pilot_model import ConditionalMegatron, model_parameter_counts  # noqa: E402
from model_dual import DualTranslator  # noqa: E402
from common import rot6d_to_matrix, matrix_to_rot6d  # noqa: E402  (base/legacy/common.py)
from kinematics import (joint_metadata, derive_joint_rotations, mixed_forward_kinematics,  # noqa: E402
                        geometry_float32, UNKNOWN, FIXED, REVOLUTE, PRISMATIC, ROOT as ROOT_TYPE)
from surface.surface_model import AttachmentCrossAttention, SurfaceMegatron  # noqa: E402

__all__ = ['ROOT', 'CODE', 'OUT', 'add_sealed_paths', 'pilot_model', 'ConditionalMegatron', 'DualTranslator',
           'model_parameter_counts', 'rot6d_to_matrix', 'matrix_to_rot6d', 'joint_metadata',
           'derive_joint_rotations', 'mixed_forward_kinematics', 'geometry_float32', 'AttachmentCrossAttention',
           'SurfaceMegatron', 'torch']
