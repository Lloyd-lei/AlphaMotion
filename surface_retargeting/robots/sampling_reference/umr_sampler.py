"""Faithful port of UMR's exterior surface sampler (scripts/surface_sampling.py, MIT, see umr_reference.py).

Algorithm (unchanged): area-weighted candidate faces (count * oversample_ratio * candidate_multiplier, with
replacement), uniform barycentric points on them, first-hit visibility of every unique candidate face centre
from the 26 axis/diagonal directions (a face is "visible" from a direction when a ray shot from far outside
along that direction hits that face first), keep candidates on faces visible from >= min_visible_views
directions, then farthest-point sampling (FPS) down to `count` with the same seeded start index.

The only additions are bookkeeping that the reference throws away and that the dense-surface asset needs:
which face each kept point came from (-> geom / link / bone binding, triangle normal), how many views saw that
face, and the sum of the directions it was seen from (used to orient normals outward independently of mesh
winding).  RNG consumption order is identical, so `sample_first_hit_surface_points` returns bit-identical
points to the reference (tests/test_assets_sampler.py checks this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

SAMPLER_VERSION = 'umr_first_hit_v1(8c4db7db)+meta'


def _sample_points_on_triangles(triangles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r1 = np.sqrt(rng.random(len(triangles)))
    r2 = rng.random(len(triangles))
    return ((1.0 - r1)[:, None] * triangles[:, 0]
            + (r1 * (1.0 - r2))[:, None] * triangles[:, 1]
            + (r1 * r2)[:, None] * triangles[:, 2])


def farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """UMR `_farthest_point_indices` (identical arithmetic)."""
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int32)
    rng = np.random.default_rng(int(seed))
    selected = np.empty(int(count), dtype=np.int32)
    selected[0] = int(rng.integers(len(points)))
    delta = points - points[selected[0]]
    min_dist2 = np.einsum('ij,ij->i', delta, delta)
    for index in range(1, int(count)):
        selected[index] = int(np.argmax(min_dist2))
        delta = points - points[selected[index]]
        dist2 = np.einsum('ij,ij->i', delta, delta)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


def first_hit_directions() -> np.ndarray:
    directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue
                direction = np.asarray([x, y, z], dtype=np.float64)
                directions.append(direction / np.linalg.norm(direction))
    return np.stack(directions, axis=0)


def _embree_intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(mesh)
    except Exception as exc:  # pragma: no cover - environment guard, same message as upstream
        raise RuntimeError('first_hit object surface sampling requires trimesh\'s Embree backend. '
                           'Install embreex in the active Python environment.') from exc


def first_hit_visibility(vertices, faces, face_ids, ray_offset, ray_distance, chunk_size):
    """Per unique face id: (unique_ids, visible_counts [U], visible_dir_sum [U,3]); same rays as UMR."""
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    outside_distance = float(ray_distance) if ray_distance > 0.0 else max(2.0 * bbox_diag, 1.0)
    intersector = _embree_intersector(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    unique_face_ids, inverse = np.unique(np.asarray(face_ids).astype(np.int64), return_inverse=True)
    face_centers = vertices[faces[unique_face_ids]].mean(axis=1)
    visible_counts = np.zeros(len(unique_face_ids), dtype=np.int32)
    dir_sum = np.zeros((len(unique_face_ids), 3), dtype=np.float64)
    for view_direction in first_hit_directions():
        for start in range(0, len(unique_face_ids), int(chunk_size)):
            end = min(start + int(chunk_size), len(unique_face_ids))
            ray_directions = np.repeat((-view_direction)[None, :], end - start, axis=0)
            origins = face_centers[start:end] + view_direction[None, :] * outside_distance
            first_faces = intersector.intersects_first(origins + ray_directions * float(ray_offset), ray_directions)
            hit = first_faces == unique_face_ids[start:end]
            visible_counts[start:end] += hit
            dir_sum[start:end] += hit[:, None] * view_direction[None, :]
    return unique_face_ids, inverse, visible_counts, dir_sum


@dataclass
class FirstHitSample:
    points: np.ndarray            # [count,3] float64 (same values as the reference, before its float32 cast)
    face_ids: np.ndarray          # [count] int64 source face of each kept point
    visible_views: np.ndarray     # [count] int32 number of the 26 views that saw that face
    visible_dir_sum: np.ndarray   # [count,3] sum of view directions that saw that face (outward hint)
    exterior_points: np.ndarray   # [E,3] all exterior candidates (for Voronoi area weights)
    exterior_face_ids: np.ndarray
    candidate_count: int
    exterior_count: int
    total_area: float
    fps_indices: np.ndarray       # indices into exterior_points


def sample_first_hit_surface(vertices, faces, count, seed, *, oversample_ratio=8, candidate_multiplier=12,
                             ray_offset=1e-4, ray_distance=0.0, min_visible_views=2, chunk_size=4096) -> FirstHitSample:
    """UMR sample_first_hit_surface_points with metadata (identical RNG stream and selection)."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    count = int(count)
    if len(vertices) == 0 or count <= 0 or len(faces) == 0:
        raise ValueError('empty mesh or count')
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        raise ValueError('Object mesh has no valid triangles for first_hit surface sampling.')
    valid_face_ids = np.flatnonzero(valid)
    valid_areas = areas[valid]
    candidate_count = max(count * int(oversample_ratio) * int(candidate_multiplier), count)
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(len(valid_face_ids), size=candidate_count, replace=True, p=valid_areas / valid_areas.sum())
    candidate_face_ids = valid_face_ids[picked]
    candidate_points = _sample_points_on_triangles(triangles[candidate_face_ids], rng)
    uniq, inverse, counts, dsum = first_hit_visibility(vertices, faces, candidate_face_ids, ray_offset, ray_distance,
                                                       chunk_size)
    exterior = (counts >= int(min_visible_views))[inverse]
    exterior_points = candidate_points[exterior]
    if len(exterior_points) < count:
        raise RuntimeError(f'first_hit kept {len(exterior_points)}/{candidate_count} object candidates, less than '
                           f'the requested {count}; refusing to sample hidden surfaces.')
    keep = farthest_point_indices(exterior_points, count, seed)
    ext_faces = candidate_face_ids[exterior]
    ext_inv = inverse[exterior]
    return FirstHitSample(points=exterior_points[keep], face_ids=ext_faces[keep], visible_views=counts[ext_inv[keep]],
                          visible_dir_sum=dsum[ext_inv[keep]], exterior_points=exterior_points,
                          exterior_face_ids=ext_faces, candidate_count=int(candidate_count),
                          exterior_count=int(len(exterior_points)), total_area=float(valid_areas.sum()),
                          fps_indices=np.asarray(keep, np.int64))


def sample_first_hit_surface_points(vertices, faces, count, seed, **kw) -> np.ndarray:
    """Drop-in equivalent of the UMR function (returns float32 points only)."""
    return sample_first_hit_surface(vertices, faces, count, seed, **kw).points.astype(np.float32)
