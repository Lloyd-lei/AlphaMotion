"""MuJoCo EGL panel renderer: one renderer per panel, a checker floor, a shadow-casting light that follows
the subject, a skybox, rigidly re-posed vendor visual meshes and free-form overlay primitives.

No mj_step, no physics: geoms are copied out of mjv_updateScene and their pose/colour overwritten
(the rollout_tools/g1.py pattern).  The floor, ghost skeletons, stick figures and markers are scene-only
geoms, so the robot model's own geom/body ids are never changed (asserted when the model is rebuilt with
the extra texture/material/light through MjSpec).
"""
from __future__ import annotations

import numpy as np

from . import _env  # noqa: F401  (MUJOCO_GL=egl)
from ._env import AX

import mujoco  # noqa: E402

FLOOR_TEX = 'am2r_floor_tex'
FLOOR_MAT = 'am2r_floor_mat'
SKY_TEX = 'am2r_sky'
LIGHT = 'am2r_key_light'

_COPY_FIELDS = ('type', 'dataid', 'matid', 'objtype', 'objid', 'category', 'texcoord', 'segid', 'emission',
                'specular', 'shininess', 'reflectance', 'camdist', 'modelrbound', 'transparent')
_COPY_ARRAYS = ('size', 'pos', 'mat', 'rgba')


SKY1, SKY2 = [0.96, 0.97, 0.98], [0.80, 0.85, 0.90]


def _decorate_spec(spec):
    spec.add_texture(name=FLOOR_TEX, type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.86, 0.87, 0.88], rgb2=[0.74, 0.76, 0.78], width=512, height=512,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.62, 0.64, 0.67])
    mat = spec.add_material(name=FLOOR_MAT, texrepeat=[2, 2], texuniform=True, reflectance=0.0)
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = FLOOR_TEX
    skies = [t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]
    for t in skies:  # the robot asset's own skybox (MuJoCo uses the first one) gets the same neutral gradient
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        t.file = ''
        t.rgb1, t.rgb2 = SKY1, SKY2
        t.width, t.height = 256, 256
    if not skies:
        spec.add_texture(name=SKY_TEX, type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=SKY1, rgb2=SKY2, width=256, height=256)
    spec.worldbody.add_light(name=LIGHT, type=mujoco.mjtLightType.mjLIGHT_SPOT, castshadow=True,
                             pos=[0, 0, 5], dir=[0, 0, -1.0], cutoff=35.0, exponent=1.0, diffuse=[0.6, 0.6, 0.6],
                             ambient=[0.0, 0.0, 0.0], specular=[0.15, 0.15, 0.15])


def build_render_model(xml: str | None, reference_model=None):
    """Compile `xml` (or an empty world) with floor texture/material, skybox and key light added.
    When `reference_model` is given (the plain from_xml_path compile used for bindings) the body/geom
    tables must be identical."""
    spec = mujoco.MjSpec.from_file(xml) if xml else mujoco.MjSpec()
    _decorate_spec(spec)
    m = spec.compile()
    if reference_model is not None:
        ok = (m.ngeom == reference_model.ngeom and m.nbody == reference_model.nbody
              and np.array_equal(m.geom_bodyid, reference_model.geom_bodyid)
              and np.array_equal(m.geom_dataid, reference_model.geom_dataid)
              and np.allclose(m.geom_size, reference_model.geom_size))
        if not ok:
            raise ValueError('decorated render model changed the geom/body tables')
    lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, LIGHT)
    for i in range(m.nlight):
        m.light_active[i] = (i == lid)
    m.vis.headlight.ambient[:] = [0.42, 0.42, 0.42]
    m.vis.headlight.diffuse[:] = [0.42, 0.42, 0.42]
    m.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
    m.vis.quality.shadowsize = 4096
    m.vis.quality.offsamples = 8
    m.vis.map.znear = 0.005
    m.vis.map.zfar = 60.0
    m.stat.extent = 3.0  # shadow / clip extents in metres, independent of the robot asset
    m.vis.map.shadowclip = 0.6
    return m, lid


class Panel3D:
    """One offscreen renderer.  Coordinates handed to the draw_* helpers are MuJoCo WORLD metres (Z-up)."""

    def __init__(self, width: int, height: int, xml: str | None = None, reference_model=None):
        self.m, self.light_id = build_render_model(xml, reference_model)
        self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), width)
        self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), height)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.w, self.h = width, height
        self.r = mujoco.Renderer(self.m, height=height, width=width, max_geom=20000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.floor_mat = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, FLOOR_MAT)
        self.scene = None

    # ------------------------------------------------------------------ frame lifecycle
    def begin(self, lookat, distance, azimuth, elevation, keep_gids=None):
        """Start a frame.  keep_gids: model geom ids to keep (re-posed later via set_geom); others dropped."""
        self.cam.lookat[:] = lookat
        self.cam.distance = float(distance)
        self.cam.azimuth = float(azimuth)
        self.cam.elevation = float(elevation)
        self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        s = self.r.scene
        keep = []
        index = {}
        want = set(int(g) for g in keep_gids) if keep_gids is not None else set()
        for k in range(s.ngeom):
            g = s.geoms[k]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and int(g.objid) in want:
                keep.append(k)
        for tgt, src in enumerate(keep):
            if tgt != src:
                x, y = s.geoms[src], s.geoms[tgt]
                for f in _COPY_FIELDS:
                    setattr(y, f, getattr(x, f))
                for f in _COPY_ARRAYS:
                    getattr(y, f)[:] = getattr(x, f)
            index[int(s.geoms[tgt].objid)] = tgt
        s.ngeom = len(keep)
        self.scene = s
        self.geom_slot = index
        # key light follows the subject
        for i in range(s.nlight):
            L = s.lights[i]
            if int(L.id) == self.light_id:
                L.pos[:] = np.asarray(lookat) * np.array([1, 1, 0]) + np.array([-0.8, -0.6, 4.5])
                v = np.asarray(lookat) * np.array([1, 1, 0]) - L.pos
                L.dir[:] = v / np.linalg.norm(v)
        return self

    def set_geom(self, gid, pos, mat, rgba=None, plain=False):
        k = self.geom_slot.get(int(gid))
        if k is None:
            return
        g = self.scene.geoms[k]
        g.pos[:] = pos
        g.mat[:] = np.asarray(mat, np.float64).reshape(3, 3)
        if rgba is not None:
            g.rgba[:] = rgba
            if plain:
                g.matid = -1
        g.transparent = 1 if (rgba is not None and rgba[3] < 0.999) else 0

    def _new(self):
        s = self.scene
        if s.ngeom >= s.maxgeom:
            raise RuntimeError('scene geom buffer full')
        g = s.geoms[s.ngeom]
        s.ngeom += 1
        return g

    def floor(self, center_xy, half=12.0, period=1.0):
        """Infinite-looking checker plane at z=0; centre snapped to the texture period so the pattern
        stays fixed in the world while the camera follows the subject."""
        cx = np.round(center_xy[0] / period) * period
        cy = np.round(center_xy[1] / period) * period
        g = self._new()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_PLANE, np.array([half, half, 0.05]), np.array([cx, cy, 0.0]),
                            np.eye(3).ravel(), np.array([1, 1, 1, 1], np.float32))
        g.matid = self.floor_mat
        g.category = mujoco.mjtCatBit.mjCAT_STATIC
        return g

    def capsule(self, a, b, radius, rgba):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        if np.linalg.norm(a - b) < 1e-7:
            return self.sphere(a, radius, rgba)
        g = self._new()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(radius), a, b)
        g.transparent = 1 if rgba[3] < 0.999 else 0
        return g

    def sphere(self, p, radius, rgba):
        g = self._new()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, radius, radius]), np.asarray(p, np.float64),
                            np.eye(3).ravel(), np.asarray(rgba, np.float32))
        g.transparent = 1 if rgba[3] < 0.999 else 0
        return g

    def spheres(self, P, radius, rgba):
        for p in P:
            self.sphere(p, radius, rgba)

    def box(self, p, half, mat, rgba):
        g = self._new()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, np.float64), np.asarray(p, np.float64),
                            np.asarray(mat, np.float64).ravel(), np.asarray(rgba, np.float32))
        g.transparent = 1 if rgba[3] < 0.999 else 0
        return g

    def cylinder_between(self, a, b, radius, rgba):
        g = self._new()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, float(radius), np.asarray(a, np.float64), np.asarray(b, np.float64))
        g.transparent = 1 if rgba[3] < 0.999 else 0
        return g

    def arrow(self, a, b, radius, rgba):
        g = self._new()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(radius), np.asarray(a, np.float64), np.asarray(b, np.float64))
        return g

    def render(self) -> np.ndarray:
        return self.r.render().copy()

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass


def canon_to_world(p_c_cm):
    """canonical Y-up cm [...,3] -> MuJoCo world Z-up metres."""
    return np.asarray(p_c_cm, np.float64) @ AX / 100.0


def canon_rot_to_world(R_c):
    """canonical rotation [...,3,3] -> world rotation:  AX.T @ R @ AX."""
    return AX.T @ np.asarray(R_c, np.float64) @ AX
