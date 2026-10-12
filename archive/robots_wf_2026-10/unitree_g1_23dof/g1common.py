"""unitree_g1_23dof dedicated pipeline -- shared helpers (per-robot file; does not touch umr_factory_code).

Model / dense surface loading, FK posing of the 4096 dense points, contact-patch definitions, MuJoCo EGL rendering
with coloured point overlays.  All lengths in metres.
"""
from __future__ import annotations

import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROBOT = 'unitree_g1_23dof'
RDIR = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT
PDIR = RDIR / 'pipeline'
QADIR = Path('/home/arenalabs/umr_factory_data/qa') / ROBOT
XML = RDIR / f'{ROBOT}.xml'
SURF = RDIR / 'surface.npz'

REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
RID = {r: i for i, r in enumerate(REGIONS)}
# left = warm hues, right = cool hues, trunk grey
REGION_RGB = {
    'trunk': (150, 150, 150), 'L_thigh': (220, 40, 40), 'R_thigh': (40, 80, 220), 'L_shin': (255, 140, 0),
    'R_shin': (0, 190, 220), 'L_foot': (240, 220, 0), 'R_foot': (20, 30, 120), 'L_upper_arm': (230, 60, 160),
    'R_upper_arm': (40, 170, 70), 'L_forearm': (150, 60, 20), 'R_forearm': (120, 60, 200), 'L_hand': (255, 170, 190),
    'R_hand': (150, 230, 150)}


def sha256(p, n=16):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:n] if n else h.hexdigest()


def load_model(xml=XML):
    return mujoco.MjModel.from_xml_path(str(xml))


def load_surface():
    z = np.load(SURF, allow_pickle=True)
    return {k: z[k] for k in z.files}


def name(m, t, i):
    return mujoco.mj_id2name(m, t, int(i))


def posed_points(d, geom, local):
    Rm = d.geom_xmat[geom].reshape(-1, 3, 3)
    return np.einsum('nij,nj->ni', Rm, local) + d.geom_xpos[geom]


class Poser:
    """Fast FK of dense (and optionally hull) points: one matmul per geom."""

    def __init__(self, m, local, geom):
        self.m, self.d = m, mujoco.MjData(m)
        self.local = np.asarray(local, np.float64); self.geom = np.asarray(geom, np.int64)
        self.groups = [(int(g), np.flatnonzero(self.geom == g)) for g in np.unique(self.geom)]

    def __call__(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        out = np.empty((len(self.local), 3))
        for g, k in self.groups:
            out[k] = self.local[k] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
        return out


def robot_collision_geoms(m):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]


def mujoco_filter_reason(m, a, b):
    """MuJoCo's own pair filter (engine_collision_driver filterBodyPair + exclude + contype/conaffinity), replicated:
    None if MuJoCo checks the pair, else the reason it is skipped."""
    ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
    wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
    if wa == wb:
        return 'same weld body (rigidly attached: removed/locked joint or same link)'
    pa, pb = int(m.body_weldid[m.body_parentid[wa]]), int(m.body_weldid[m.body_parentid[wb]])
    if not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT) and wa != 0 and wb != 0 and (wa == pb or wb == pa):
        return 'parent-child across one joint (MuJoCo filterparent)'
    sig = (min(ba, bb) << 16) + max(ba, bb)
    for i in range(m.nexclude):
        if int(m.exclude_signature[i]) == sig:
            return 'explicit <exclude>'
    if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
        return 'contype/conaffinity mask'
    return None


def mujoco_considered_pairs(m):
    cg = robot_collision_geoms(m)
    return {(a, b) for i, a in enumerate(cg) for b in cg[i + 1:] if mujoco_filter_reason(m, a, b) is None}


# ------------------------------------------------------------------------------------------------ contact patches
def contact_patches(m, surf):
    """Which dense points may carry contact, by patch.  Defined at the rest pose (qpos0: feet flat on z~0)."""
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    X = posed_points(d, surf['geom'], surf['points_local'])
    reg = surf['region'].astype(int)
    body = surf['body'].astype(int)
    bn = [name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    P = {}
    for s, side in (('L', 'left'), ('R', 'right')):
        foot = reg == RID[f'{s}_foot']
        zsole = X[foot, 2].min()
        # sole + its rim/edges: lower 20 mm of the foot (point / edge contact allowed, ankle top is not)
        P[f'{s}_sole'] = foot & (X[:, 2] <= zsole + 0.020)
        kj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_knee_joint')
        ka = d.xanchor[kj]
        leg = (reg == RID[f'{s}_shin']) | (reg == RID[f'{s}_thigh'])
        P[f'{s}_knee'] = leg & (np.linalg.norm(X - ka, axis=1) <= 0.08)
        # shin front (kneeling: shin lies on the floor) = shin points in front of the knee-ankle line
        aj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_ankle_pitch_joint')
        aa = d.xanchor[aj]
        shin = reg == RID[f'{s}_shin']
        P[f'{s}_shin_front'] = shin & (X[:, 0] > 0.5 * (ka[0] + aa[0]) + 0.01)
        ej = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_elbow_joint')
        ea = d.xanchor[ej]
        arm = (reg == RID[f'{s}_upper_arm']) | (reg == RID[f'{s}_forearm'])
        P[f'{s}_elbow'] = arm & (np.linalg.norm(X - ea, axis=1) <= 0.06)
        P[f'{s}_hand'] = reg == RID[f'{s}_hand']
    trunk = reg == RID['trunk']
    pel = np.isin(body, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('pelvis', 'pelvis_contour_link')])
    P['pelvis'] = trunk & pel
    torso_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    # head mesh is merged into torso_link (geom with head_link mesh)
    head_g = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and
              name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) == 'head_link']
    head = np.isin(surf['geom'], head_g)
    P['head'] = head
    P['torso'] = trunk & ~pel & ~head
    return P, X


# ------------------------------------------------------------------------------------------------ rendering
def add_points(scn, X, rgba, size):
    """Append points as spheres to an mjvScene (respects maxgeom)."""
    X = np.asarray(X, np.float64)
    rgba = np.asarray(rgba, np.float32)
    if rgba.ndim == 1:
        rgba = np.broadcast_to(rgba, (len(X), 4))
    sz = np.array([size, 0, 0], np.float64)
    eye = np.eye(3).reshape(-1)
    for i in range(len(X)):
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, sz, X[i], eye, rgba[i])
        scn.ngeom += 1


def make_cam(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
    return cam


def vis_option(show_collision=False, show_floor=True):
    opt = mujoco.MjvOption()
    for gi in range(6):
        opt.geomgroup[gi] = 0
    opt.geomgroup[1] = 1                       # visual meshes
    if show_collision or show_floor:
        opt.geomgroup[0] = 1                   # floor + collision geoms live in group 0
    return opt
