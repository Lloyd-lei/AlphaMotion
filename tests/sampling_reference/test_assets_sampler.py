"""WP-ASSETS: the ported UMR exterior sampler is bit-identical to the upstream reference (CPU, < 30 s)."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip('trimesh')
pytest.importorskip('embreex')
from surface_retargeting.robots.sampling_reference import umr_reference as REF  # noqa: E402
from surface_retargeting.robots.sampling_reference import umr_sampler as U  # noqa: E402


def _two_boxes():
    """A big box with a small box hidden inside it, plus a separate visible box."""
    big = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    inner = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    side = trimesh.creation.box(extents=(0.4, 0.4, 0.4)); side.apply_translation((1.5, 0, 0))
    big = big.subdivide().subdivide(); side = side.subdivide()
    parts = [big, inner, side]
    V = np.concatenate([p.vertices for p in parts]); F = []
    off = 0
    owner = []
    for i, p in enumerate(parts):
        F.append(p.faces + off); off += len(p.vertices); owner.append(np.full(len(p.faces), i))
    return V, np.concatenate(F), np.concatenate(owner)


def test_bit_identical_to_reference():
    V, F, _ = _two_boxes()
    a = U.sample_first_hit_surface_points(V, F, 300, 0)
    b = REF.sample_first_hit_surface_points(V, F, 300, 0)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b)
    a2 = U.sample_first_hit_surface_points(V, F, 300, 5, oversample_ratio=4, candidate_multiplier=3)
    b2 = REF.sample_first_hit_surface_points(V, F, 300, 5, oversample_ratio=4, candidate_multiplier=3)
    assert np.array_equal(a2, b2)


def test_fps_identical_and_hidden_faces_rejected():
    V, F, owner = _two_boxes()
    s = U.sample_first_hit_surface(V, F, 400, 0)
    assert len(s.points) == 400 and s.exterior_count < s.candidate_count
    assert not np.any(owner[s.face_ids] == 1), 'a point came from the hidden inner box'
    assert np.all(s.visible_views >= 2)
    pts = np.random.default_rng(1).normal(size=(2000, 3))
    assert np.array_equal(U.farthest_point_indices(pts, 64, 3), REF._farthest_point_indices(pts, 64, 3))
    # outward hint: the view-direction sum points away from the box centre on the big box
    big = owner[s.face_ids] == 0
    assert np.mean((s.points[big] * s.visible_dir_sum[big]).sum(1) > 0) > 0.99


def test_directions_are_26_unit():
    d = U.first_hit_directions()
    assert d.shape == (26, 3) and np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.array_equal(d, REF._first_hit_directions())
