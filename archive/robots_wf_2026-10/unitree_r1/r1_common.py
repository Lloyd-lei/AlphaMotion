"""unitree_r1 dedicated pipeline -- shared helpers (robot-local, independent of umrf measurement code).

Everything here is written for unitree_r1 only.  Robot-side measurement uses only: the staged MJCF, surface.npz
(points bound to geoms) and MuJoCo (FK, mj_geomDistance).  No umrf.geometry / umrf.labels code is used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROBOT = 'unitree_r1'
RDIR = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT
PDIR = RDIR / 'pipeline'
QADIR = Path('/home/arenalabs/umr_factory_data/qa') / ROBOT
XML = RDIR / f'{ROBOT}.xml'
SURF = RDIR / 'surface.npz'

REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
RID = {r: i for i, r in enumerate(REGIONS)}
REG_COL = np.array([[150, 150, 150], [230, 120, 40], [40, 120, 230], [250, 180, 60], [60, 180, 250],
                    [200, 30, 30], [30, 30, 200], [140, 60, 170], [60, 170, 140], [200, 90, 200],
                    [90, 200, 120], [250, 60, 150], [20, 200, 200]], np.uint8)

try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    FONT_S = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
except Exception:  # pragma: no cover
    FONT = FONT_S = ImageFont.load_default()


def load_model(xml=XML):
    return mujoco.MjModel.from_xml_path(str(xml))


def load_surface():
    z = np.load(SURF, allow_pickle=True)
    return {k: z[k] for k in z.files}


def name(m, t, i):
    return mujoco.mj_id2name(m, t, int(i)) or str(int(i))


def bname(m, b):
    return name(m, mujoco.mjtObj.mjOBJ_BODY, b)


def jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def set_named(m, d, qdict):
    for k, v in qdict.items():
        j = jid(m, k)
        d.qpos[m.jnt_qposadr[j]] = v


def collision_geoms(m):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]


def is_parent_child(m, b1, b2):
    return m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1


def self_pairs_all(m, exclusions=()):
    """Every collision-geom pair MuJoCo would test: different bodies, not parent/child (MuJoCo filterparent),
    not in the exclusion list.  Independent enumeration (does not use mj_collision broadphase)."""
    cg = collision_geoms(m)
    ex = {tuple(sorted(e)) for e in exclusions}
    out = []
    for i, a in enumerate(cg):
        for b in cg[i + 1:]:
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if ba == bb or is_parent_child(m, ba, bb):
                continue
            if tuple(sorted((bname(m, ba), bname(m, bb)))) in ex:
                continue
            out.append((a, b))
    return out


class Poser:
    """Dense surface points posed by MuJoCo FK (geom frames)."""

    def __init__(self, m, surf):
        self.m = m
        self.d = mujoco.MjData(m)
        self.pl = surf['points_local'].astype(np.float64)
        self.pg = surf['geom'].astype(np.int64)
        self.reg = surf['region'].astype(np.int64)
        self.groups = [(int(g), np.flatnonzero(self.pg == g)) for g in np.unique(self.pg)]

    def set(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)

    def points(self):
        out = np.empty((len(self.pl), 3))
        for g, k in self.groups:
            out[k] = self.pl[k] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
        return out


def ortho(X, view, scale, cx, cy, W, H):
    """view 'front' (camera in front of the robot, which faces +x: screen-right = +y = robot LEFT), 'side'
    (camera at -y looking along +y: screen-right = +x = robot forward), 'back'.  Returns u, v, depth (larger = closer to camera)."""
    if view == 'front':                      # camera at +x looking along -x: screen-right = +y (robot's left)
        u = cx + X[:, 1] * scale; depth = X[:, 0]
    elif view == 'side':
        u = cx + X[:, 0] * scale; depth = -X[:, 1]
    elif view == 'back':                     # camera at -x looking along +x: screen-right = -y
        u = cx - X[:, 1] * scale; depth = -X[:, 0]
    else:
        raise ValueError(view)
    v = cy - X[:, 2] * scale
    return u, v, depth


def draw_points(img, X, cols, view, scale, cx, cy, r=2, order=None):
    W, H = img.size
    u, v, dep = ortho(X, view, scale, cx, cy, W, H)
    dr = ImageDraw.Draw(img)
    idx = np.argsort(dep) if order is None else order
    for i in idx:
        c = tuple(int(x) for x in cols[i])
        dr.ellipse([u[i] - r, v[i] - r, u[i] + r, v[i] + r], fill=c)
    return img


def text(img, xy, s, fill=(0, 0, 0), small=False):
    ImageDraw.Draw(img).text(xy, s, fill=fill, font=FONT_S if small else FONT)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def dump(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))
