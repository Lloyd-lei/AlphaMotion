"""WP-MODEL: slide fix (SF), dense-surface group tokens (S), relational inputs (I), relational
objective (O), geometry wrapper (incl. S_wrong derangement).

Import as ``am2exp_v1.model`` (ROOT on sys.path).  A bare ``import model`` resolves to the
sealed base/legacy/model.py and must stay that way.
"""
from ._sealed import add_sealed_paths  # noqa: F401  (also installs sealed sys.path entries)
from .slide_fix import (SlideFixHead, SlideFixMixin, SFMegatron, SFSurfaceMegatron, slide_fix_loss,  # noqa: F401
                        make_slide_fix_objective, clamp_slide_native, decode_slide_margin)
from .dense_surface import DenseSurfaceMegatron, arm_parameter_counts  # noqa: F401
from .relational import RelationalInputMegatron, relational_features  # noqa: F401
from .losses_relational import relational_objective, make_objective  # noqa: F401
from .geometry import attach_dense_geometry, detach_dense_geometry, default_derangement, build_dense_static  # noqa: F401
from .ground import (excluded_ground_groups, excluded_ground_groups_from_bodies, set_excluded_ground_groups,  # noqa: F401
                     source_root_height_sensor, row_ground_ok)
from .registry import MODEL_CLASSES, ARMS, build_model, build_arm  # noqa: F401
