"""Read-only access to the sealed rot6 helpers (base/legacy/common.py) used by the model."""
from ._env import sealed

sealed()
import common as _legacy_common  # noqa: E402  (base/legacy/common.py)

if not hasattr(_legacy_common, 'rot6d_to_matrix'):
    raise ImportError('expected sealed base/legacy/common.py, got ' + repr(getattr(_legacy_common, '__file__', None)))
rot6d_to_matrix = _legacy_common.rot6d_to_matrix
matrix_to_rot6d = _legacy_common.matrix_to_rot6d
