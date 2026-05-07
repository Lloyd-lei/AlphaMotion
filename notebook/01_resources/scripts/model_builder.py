"""Build a model from a `model` spec dict (NB1 §0.5 VARIANTS).

Notebook owns the spec dict; this module just turns it into a torch module.
"""
from __future__ import annotations
from typing import Tuple

from baseline    import BaselineConfig, BaselineTransformer
from motionformer import MotionFormerConfig, MotionFormer


def build_from_spec(spec: dict, T: int, J: int, C: int) -> Tuple[object, object]:
    arch = spec["arch"]
    if arch == "baseline":
        cfg = BaselineConfig(
            T=T, J=J, C=C,
            hidden=spec["hidden"], depth=spec["depth"],
            heads=spec["heads"], ffn_mult=spec["ffn_mult"],
        )
        return BaselineTransformer(cfg), cfg

    if arch == "motionformer":
        cfg = MotionFormerConfig(
            T=T, J=J, C=C,
            hidden=spec["hidden"], pair_hidden=spec["pair_hidden"],
            depth=spec["depth"], heads=spec["heads"], pair_heads=spec["pair_heads"],
            opm_chunk=spec["opm_chunk"], tri_hidden=spec["tri_hidden"],
            use_pair=spec["use_pair"], use_opm=spec["use_opm"],
            use_triangle=spec["use_triangle"],
        )
        return MotionFormer(cfg), cfg

    raise ValueError(f"unknown arch: {arch!r}")
