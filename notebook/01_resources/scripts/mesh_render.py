"""Offscreen mesh rendering → animated GIF (pyrender / EGL).

Used by NB1 + NB2 to render SMPL-X mesh playback. EGL backend works headless.

Visual defaults aim for low eye-strain:
    - background: soft warm gray  (not pure white — avoids blinding flash)
    - mesh color: skin tone        (warm beige, easy to read body shape)
    - ground plane: light gray     (gives the figure a visual anchor)
    - lighting:    key + fill      (better depth perception than single source)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path
from typing import Tuple

import numpy as np
import imageio.v2 as imageio
import pyrender
import trimesh


# ----- camera helper --------------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0., 1., 0.])):
    """OpenGL-convention camera pose looking from eye toward target."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = s
    M[:3, 1] = u
    M[:3, 2] = -f                # camera looks toward -Z
    M[:3, 3] = eye
    return M


def _make_ground_plane(centroid: np.ndarray, radius: float, color=(220, 220, 225)):
    """Flat horizontal plane at the lowest Y of the body, ~6× radius wide."""
    extent = max(radius * 6.0, 1.5)
    plane = trimesh.creation.box(extents=[extent, 0.01, extent])
    plane.apply_translation([centroid[0], 0.0, centroid[2]])
    plane.visual.vertex_colors = list(color) + [255]
    return plane


# ----- main entry point ----------------------------------------------------

def render_mesh_gif(
    verts_per_frame: np.ndarray,        # [T, V, 3]
    faces:           np.ndarray,        # [F, 3]
    out_path:        str | Path,
    *,
    viewport:        Tuple[int, int] = (480, 480),
    fps:             int   = 20,
    mesh_color:      tuple = (210, 178, 145),    # warm skin tone
    bg_color:        tuple = (0.93, 0.93, 0.95, 1.0),
    ground_plane:    bool  = True,
    cam_dist_mult:   float = 2.5,
    cam_height_mult: float = 0.4,
) -> Path:
    """Render a mesh sequence to an animated GIF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_verts = verts_per_frame.reshape(-1, 3)
    centroid  = all_verts.mean(axis=0)
    extent    = all_verts.max(axis=0) - all_verts.min(axis=0)
    radius    = float(np.linalg.norm(extent)) * 0.5 + 1e-6
    cam_dist  = radius * cam_dist_mult
    cam_eye   = centroid + np.array([0., cam_height_mult * radius, cam_dist])
    cam_pose  = _look_at(cam_eye, centroid)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3,
                                      aspectRatio=viewport[0] / viewport[1])

    # Two lights: key (camera-side) + fill (opposite, dimmer) for nicer shading
    key_light  = pyrender.DirectionalLight(color=np.ones(3),       intensity=4.0)
    fill_light = pyrender.DirectionalLight(color=np.array([1, 0.95, 0.9]), intensity=2.0)
    key_pose   = _look_at(cam_eye + np.array([0.,  0.5*radius, 0.]), centroid)
    fill_pose  = _look_at(cam_eye + np.array([0., -1.0*radius, -2*cam_dist]), centroid)

    # Optional ground plane (computed once, reused per frame)
    ground = _make_ground_plane(centroid, radius) if ground_plane else None

    renderer = pyrender.OffscreenRenderer(
        viewport_width=viewport[0], viewport_height=viewport[1],
    )

    frames = []
    try:
        for verts in verts_per_frame:
            scene = pyrender.Scene(bg_color=list(bg_color),
                                    ambient_light=[0.45, 0.45, 0.45])
            mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.visual.vertex_colors = list(mesh_color) + [255]
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            if ground is not None:
                # Place ground at the lowest Y of THIS frame
                g = ground.copy()
                g.apply_translation([0., float(verts[:, 1].min()) - 0.01, 0.])
                scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
            scene.add(cam,        pose=cam_pose)
            scene.add(key_light,  pose=key_pose)
            scene.add(fill_light, pose=fill_pose)
            color, _ = renderer.render(scene)
            frames.append(color)
    finally:
        renderer.delete()

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path
