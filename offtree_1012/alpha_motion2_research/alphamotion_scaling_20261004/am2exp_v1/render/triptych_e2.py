"""E2 triptych renderer: [Adroit hand + object, pads, opposition axis | control gripper | test gripper].

1920x1080 H.264 at the REAL data timestamps (D4RL Adroit = 100 Hz; default stride 2 -> 50 frames/s, still real
time; rollout_tools/encode.encode, VFR, PTS verified).  One episode of hand_<ds>.npz (CODE/e2/hand_data.py output).

Hand panel   vendor Adroit MJCF VISUAL meshes (V_* geoms of adroit_pen.xml) posed by MuJoCo from the dataset hand
             qpos (forearm fixed; hand_data positions equal MuJoCo body positions to 0.000 cm, asserted in tests);
             object = dataset object geometry (hammer handle capsule d=50 mm / relocate ball d=70 mm / pen d=30 mm)
             in the same forearm frame; pads (e2.opposition Adroit pad table, frozen variant) as spheres, thumb
             orange, virtual-finger fingers purple (size ~ softmax weight), opposition axis thumb -> virtual finger.
Gripper      vendor Franka MJCF (relay_robot_only_derived.xml) VISUAL meshes of the hand + both fingers, arm parked at
panels       a fixed home pose; camera rigid in the EE frame (looks across the finger gap).  Finger slides
             qL = qR = (c_F + 0.2 cm)/2 clipped to the 0..4 cm stroke (c_F = qL + qR - 0.2 cm, e2.opposition).
             The object is drawn centred between the pads with the dataset diameter; an orange line across the
             gap = object diameter.  Finger colour = judge J2 state from c_F alone (closure_judge.gripper_state):
             green hold (d-3 <= c_F <= d+8 mm), grey open, red crush.
Traces       per panel: pad clearance (mm) vs time; dashed orange = object diameter; green band = J2 hold band;
             grey line in gripper panels = the hand's own opposition gap c_A; bottom strip = J2 grasp label
             (green hand holds, grey hand free, white not scored).

Inputs for the gripper panels: an E2 arm directory (OUT/eval/e2/arms/<name>/<ds>.npz with c_F_mm aligned row by
row with hand_<ds>.npz -- checked through episode/frame_index) or a raw array (tests / synthetic).

CLI
  python -m render.triptych_e2 --dataset hammer --episode 1 --control AN --test A0_S_at_3000 --out DIR
  python -m render.triptych_e2 --dataset hammer --episode 1 --control AN --test AN --out DIR   # identity check
  (--control/--test: arm name under OUT/eval/e2/arms or a path to a dir / npz; --labels 'G0,G1')
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'render'

from . import _env  # noqa: E402
from ._env import AX  # noqa: E402
from . import layout as L  # noqa: E402
from .scene3d import Panel3D  # noqa: E402

import mujoco  # noqa: E402

PANEL_W, VIEW_H = 640, 600
TRACE_H = 230
E2_ARMS = _env.OUT / 'eval' / 'e2' / 'arms'
E2_DATA = _env.OUT / 'data' / 'e2'
HOLD_LOW_MM, HOLD_HIGH_MM = 3.0, 8.0          # = e2.closure_judge (asserted equal in tests)
FRANKA_OFFSET_CM, FRANKA_STROKE_CM = 0.20, 4.0
FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
DS_LABEL = {'hammer': '锤子把手', 'relocate': '小球', 'pen': '笔'}

OBJ_RGBA = np.array([0.25, 0.50, 0.85, 0.45], np.float32)
THUMB_RGBA = np.array([0.96, 0.55, 0.10, 1.0], np.float32)
FINGER_RGBA = np.array([0.55, 0.30, 0.80, 1.0], np.float32)
PAD_OTHER = np.array([0.55, 0.58, 0.62, 1.0], np.float32)
AXIS_RGBA = np.array([0.96, 0.55, 0.10, 1.0], np.float32)
DIAM_RGBA = np.array([0.96, 0.55, 0.10, 1.0], np.float32)
STATE_RGBA = {1: np.array([0.16, 0.72, 0.32, 1.0], np.float32),    # hold
              0: np.array([0.80, 0.82, 0.85, 1.0], np.float32),    # open
              -1: np.array([0.90, 0.16, 0.16, 1.0], np.float32)}   # crush
STATE_WORD = {1: '夹住（间隙在合格带里）', 0: '张开', -1: '夹太紧（会压进物体）'}
LABEL_WORD = {1: '手：正抓着物体', 0: '手：离物体很远', -1: '手：中间状态（不计分）'}


# ------------------------------------------------------------------------------------------------ e2 imports
@lru_cache(maxsize=None)
def e2mod(name: str):
    """CODE/e2/<name>.py imported as package module ``e2.<name>`` (read-only; CODE appended to sys.path)."""
    code = str(_env.CODE)
    if code not in sys.path:
        sys.path.append(code)
    return importlib.import_module(f'e2.{name}')


def e2_xml(which: str) -> str:
    env = e2mod('_env')
    return str(env.PEN_XML if which == 'pen' else env.FRANKA_XML)


# ------------------------------------------------------------------------------------------------ data
@dataclass
class HandEpisode:
    dataset: str
    episode: int
    rows: np.ndarray           # row indices into hand_<ds>.npz (frame order)
    qpos: np.ndarray           # [T,24]
    P: np.ndarray              # [T,25,3] canonical cm (forearm-relative)
    G: np.ndarray              # [T,25,3,3]
    obj_type: str
    obj_d_mm: float
    obj_radius_cm: float
    obj_segment_cm: np.ndarray  # [T,2,3]
    obj_centre_cm: np.ndarray   # [T,3]
    times: np.ndarray
    split: str = ''
    labels: np.ndarray | None = None   # J2 grasp label per frame (1/0/-1)
    joint_names: list = field(default_factory=list)


def load_hand_episode(dataset: str, episode: int, stride: int = 2, frames=None, labels=None) -> HandEpisode:
    with np.load(E2_DATA / f'hand_{dataset}.npz', allow_pickle=False) as z:
        ep = z['episode_index']
        rows = np.flatnonzero(ep == int(episode))
        if not len(rows):
            raise ValueError(f'{dataset}: no episode {episode}')
        rows = rows[np.argsort(z['frame_index'][rows], kind='stable')]
        if frames is not None:
            a, b = frames
            rows = rows[a:b]
        rows = rows[::max(int(stride), 1)]
        G = _rot6(z['global_rot6'][rows])
        fps = 100.0
        fi = z['frame_index'][rows].astype(np.float64)
        h = HandEpisode(dataset=dataset, episode=int(episode), rows=rows, qpos=z['hand_qpos'][rows].astype(np.float64),
                        P=z['position_cm'][rows].astype(np.float64), G=G, obj_type=str(z['obj_type']),
                        obj_d_mm=float(z['obj_diameter_mm']), obj_radius_cm=float(z['obj_radius_cm']),
                        obj_segment_cm=z['obj_segment_cm'][rows].astype(np.float64),
                        obj_centre_cm=z['obj_centre_cm'][rows].astype(np.float64),
                        times=(fi - fi[0]) / fps, split=str(z['split'][rows[0]]),
                        joint_names=[str(x) for x in z['joint_names']])
    if labels is not None:
        h.labels = np.asarray(labels)[rows]
    return h


def _rot6(x):
    return _env.rot6_to_matrix(x)


def load_arm_cF(arm: str, dataset: str, hand: HandEpisode | None = None):
    """-> (c_F_mm for ALL rows of hand_<ds>.npz, labels or None, provenance).  arm = name under OUT/eval/e2/arms,
    a directory holding <ds>.npz, or an npz path.  Row alignment with the hand file is verified."""
    p = Path(arm)
    if not p.exists():
        p = E2_ARMS / arm.replace('@', '_at_')
    if p.is_dir():
        p = p / f'{dataset}.npz'
    if not p.is_file():
        raise FileNotFoundError(f'no E2 arm output for {arm!r} ({p})')
    with np.load(p, allow_pickle=False) as z:
        c = z['c_F_mm'].astype(np.float64)
        lab = z['labels'] if 'labels' in z.files else None
        ep = z['episode'] if 'episode' in z.files else None
        fi = z['frame_index'] if 'frame_index' in z.files else None
    with np.load(E2_DATA / f'hand_{dataset}.npz', allow_pickle=False) as h:
        if len(c) != len(h['episode_index']):
            raise ValueError(f'{p}: {len(c)} rows, hand file has {len(h["episode_index"])}')
        if ep is not None and not np.array_equal(ep, h['episode_index']):
            raise ValueError(f'{p}: episode column does not match hand_{dataset}.npz')
        if fi is not None and not np.array_equal(fi, h['frame_index']):
            raise ValueError(f'{p}: frame_index column does not match hand_{dataset}.npz')
    return c, lab, str(p)


def hand_descriptor(h: HandEpisode):
    """Frozen opposition descriptor (e2.opposition) on the dataset hand -> dict of numpy arrays (cm)."""
    import torch
    op = e2mod('opposition')
    with torch.no_grad():
        out = op.adroit_descriptor(torch.as_tensor(h.P, dtype=torch.float32), torch.as_tensor(h.G, dtype=torch.float32))
    res = {k: (v.cpu().numpy().astype(np.float64) if hasattr(v, 'cpu') else v) for k, v in out.items()}
    fz = op.load_frozen()
    res['variant_name'] = fz.get('variant', out.get('variant'))
    return res


def gripper_state(c_mm, d_mm):
    c = np.asarray(c_mm, np.float64)
    s = np.ones(c.shape, np.int8)
    s[c > d_mm + HOLD_HIGH_MM] = 0
    s[c < d_mm - HOLD_LOW_MM] = -1
    return s


# ------------------------------------------------------------------------------------------------ camera
def cam_from_forward(f):
    """MuJoCo free camera (azimuth, elevation) in degrees looking along world direction f."""
    f = np.asarray(f, np.float64)
    f = f / np.linalg.norm(f)
    return float(np.degrees(np.arctan2(f[1], f[0]))), float(np.degrees(np.arcsin(np.clip(f[2], -1, 1))))


def _rot_about(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _side_view(o, n, tilt_deg=25.0, max_el=35.0):
    """Camera forward perpendicular to the opposition axis o and to n, tilted about n; elevation limited."""
    f = np.cross(o, n)
    if np.linalg.norm(f) < 1e-6:
        f = np.cross(o, [0, 0, 1.0])
    f = f / np.linalg.norm(f)
    if f[2] > 0:   # prefer looking slightly down
        f = -f
    f = _rot_about(n, tilt_deg) @ f
    az, el = cam_from_forward(f)
    return az, float(np.clip(el, -max_el, max_el))


# ------------------------------------------------------------------------------------------------ drawers
class HandDrawer:
    def __init__(self, h: HandEpisode, desc: dict):
        xml = e2_xml('pen')
        self.ref = mujoco.MjModel.from_xml_path(xml)
        self.p3 = Panel3D(PANEL_W, VIEW_H, xml, reference_model=self.ref)
        m = self.p3.m
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in e2mod('hand_data').HAND_JOINTS])
        fa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'forearm')
        self.gids = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('V_')]
        mujoco.mj_forward(m, self.p3.d)
        self.fa_pos = self.p3.d.xpos[fa].copy()
        self.h, self.desc = h, desc
        w = lambda c: self.fa_pos + np.asarray(c, np.float64) @ AX / 100.0  # noqa: E731 canonical cm -> world m
        self.w = w
        self.pads = w(desc['pads'])
        self.p_th, self.p_vf = w(desc['p_th']), w(desc['p_vf'])
        self.seg = w(h.obj_segment_cm)
        self.ctr = w(h.obj_centre_cm)
        o = (self.p_vf - self.p_th).mean(0)
        palm = w(h.P[:, 2])
        n = (0.5 * (self.p_th + self.p_vf) - palm).mean(0)
        self.az, self.el = _side_view(o / max(np.linalg.norm(o), 1e-9), n / max(np.linalg.norm(n), 1e-9))
        look = 0.5 * (self.p_th + self.p_vf)
        k = 25
        self.look = np.stack([look[max(0, t - k):t + k + 1].mean(0) for t in range(len(look))])
        self.dist = 0.42

    def pose(self, t):
        d = self.p3.d
        d.qpos[self.qadr] = self.h.qpos[t]
        mujoco.mj_forward(self.p3.m, d)
        return d

    def body_pos_check(self, t):
        """max |MuJoCo body position - hand_data position_cm| (cm) at frame t (hand joints in AM2 order)."""
        d = self.pose(t)
        m = self.p3.m
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.h.joint_names]
        return float(np.abs((d.xpos[ids] - self.w(self.h.P[t])) * 100).max())

    def draw(self, t):
        self.pose(t)
        p3 = self.p3.begin(self.look[t], self.dist, self.az, self.el, keep_gids=self.gids)
        h = self.h
        if h.obj_type == 'sphere':
            p3.sphere(self.ctr[t], h.obj_radius_cm / 100.0, OBJ_RGBA)
        else:
            p3.capsule(self.seg[t, 0], self.seg[t, 1], h.obj_radius_cm / 100.0, OBJ_RGBA)
        wts = self.desc['weights'][t]
        fingers = [k for k in range(1, 5)]
        vf_ids = e2_vf_ids(self.desc)
        for k in fingers:
            if k in vf_ids:
                wk = float(wts[vf_ids.index(k)])
                p3.sphere(self.pads[t, k], 0.004 + 0.004 * wk, FINGER_RGBA)
            else:
                p3.sphere(self.pads[t, k], 0.0035, PAD_OTHER)
        p3.sphere(self.pads[t, 0], 0.007, THUMB_RGBA)
        p3.sphere(self.p_vf[t], 0.005, FINGER_RGBA)
        p3.capsule(self.p_th[t], self.p_vf[t], 0.0022, AXIS_RGBA)
        return p3.render()

    def close(self):
        self.p3.close()


def e2_vf_ids(desc):
    """Digit ids (DIGITS order th=0,ff=1,mf=2,rf=3,lf=4) of the virtual-finger members of the frozen variant."""
    op = e2mod('opposition')
    spec, mode = op.VARIANTS[desc['variant']]
    if mode == 'thumb_facing':
        return [1, 2, 3, 4]
    return [op.DIGITS.index(k) for k in spec]


@lru_cache(maxsize=2)
def franka_home(xml: str, n: int = 4000, seed: int = 0):
    """Arm pose (7 joints, inside the MJCF ranges) whose fingers point straight down (world -z), so the free camera
    (screen-up = world +z) shows the gripper upright.  Deterministic seeded search; result cached."""
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'panda0_joint{i}')] for i in range(1, 8)]
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'panda0_joint{i}') for i in range(1, 8)]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    bh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'panda0_link7')
    bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'panda0_leftfinger')
    rng = np.random.default_rng(seed)
    best, bq = 9.0, None
    for q in [FRANKA_HOME] + list(rng.uniform(lo, hi, size=(n, 7))):
        d.qpos[adr] = np.clip(q, lo, hi)
        mujoco.mj_kinematics(m, d)
        a = d.xpos[bl] - d.xpos[bh]
        a = a / np.linalg.norm(a)
        e = a[2] + 1.0
        if e < best:
            best, bq = e, np.clip(q, lo, hi).copy()
    return bq


class GripperDrawer:
    def __init__(self, c_mm: np.ndarray, d_mm: float, obj_type: str, obj_half_len_cm: float):
        xml = e2_xml('franka')
        self.ref = mujoco.MjModel.from_xml_path(xml)
        self.p3 = Panel3D(PANEL_W, VIEW_H, xml, reference_model=self.ref)
        m, d = self.p3.m, self.p3.d
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.arm_adr = np.array([m.jnt_qposadr[jid(f'panda0_joint{i}')] for i in range(1, 8)])
        self.home = franka_home(e2_xml('franka'))
        self.fl, self.fr = m.jnt_qposadr[jid('panda0_finger_joint1')], m.jnt_qposadr[jid('panda0_finger_joint2')]
        self.jl, self.jr = jid('panda0_finger_joint1'), jid('panda0_finger_joint2')
        self.bl, self.br, self.bh = bid('panda0_leftfinger'), bid('panda0_rightfinger'), bid('panda0_link7')
        keep_bodies = {self.bl, self.br, self.bh}
        self.gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] in keep_bodies and m.geom_group[g] == 0]
        self.finger_gids = [g for g in self.gids if m.geom_bodyid[g] in (self.bl, self.br)]
        self.caps = {b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
                     for b in (self.bl, self.br)}
        self.c_raw = np.asarray(c_mm, np.float64)
        self.q_cm = np.clip((self.c_raw / 10.0 + FRANKA_OFFSET_CM) / 2.0, 0.0, FRANKA_STROKE_CM)
        self.clipped = (self.q_cm <= 0.0) | (self.q_cm >= FRANKA_STROKE_CM)
        self.d_mm, self.obj_type, self.half_len = d_mm, obj_type, obj_half_len_cm
        self.state = gripper_state(self.c_raw, d_mm)
        # EE frame at mid opening
        self._set(2.0)
        pl, pr = self.pads()
        self.centre = 0.5 * (pl + pr)
        u = (pl - pr) / np.linalg.norm(pl - pr)
        a = self.centre - d.xpos[self.bh]
        a = a - (a @ u) * u
        a = a / np.linalg.norm(a)
        self.u, self.a = u, a
        self.obj_axis = np.cross(u, a)
        # look straight across the gap along the object axis (gap and object diameter undistorted on screen), tipped
        # 12 deg about the opening axis u so the view is slightly from above; u stays horizontal on screen
        f = self.obj_axis if self.obj_axis[2] <= 0 else -self.obj_axis
        f = _rot_about(u, 12.0) @ f
        if f[2] > 0:
            f = _rot_about(u, -24.0) @ f
        self.az, self.el = cam_from_forward(f)
        self.look = self.centre - 0.035 * a     # a bit toward the palm so the whole finger shows
        self.dist = 0.40

    def tip_gap_mm_at(self, c_mm):
        q = float(np.clip((c_mm / 10.0 + FRANKA_OFFSET_CM) / 2.0, 0.0, FRANKA_STROKE_CM))
        self._set(q)
        m, d = self.p3.m, self.p3.d
        gl, gr = self.caps[self.bl][0], self.caps[self.br][0]
        return float((np.linalg.norm(d.geom_xpos[gl] - d.geom_xpos[gr]) - m.geom_size[gl][0] - m.geom_size[gr][0]) * 1000.0)

    def tip_gap_mm(self, t):
        """Gap between the two fingertip capsules' inner surfaces (mm) at frame t (geometry fact, for the README)."""
        self._set(self.q_cm[t])
        m, d = self.p3.m, self.p3.d
        gl = self.caps[self.bl][0]
        gr = self.caps[self.br][0]
        return float((np.linalg.norm(d.geom_xpos[gl] - d.geom_xpos[gr]) - m.geom_size[gl][0] - m.geom_size[gr][0]) * 1000.0)

    def _set(self, q_cm):
        d = self.p3.d
        d.qpos[self.arm_adr] = self.home
        d.qpos[self.fl] = q_cm / 100.0
        d.qpos[self.fr] = q_cm / 100.0
        mujoco.mj_forward(self.p3.m, d)

    def pads(self):
        """Inner-pad centres (world m) = mean over the finger's capsules of (axis point - radius * opening dir),
        the e2.opposition.franka_pad_table construction."""
        m, d = self.p3.m, self.p3.d
        out = []
        for b, j in ((self.bl, self.jl), (self.br, self.jr)):
            opening = d.xmat[b].reshape(3, 3) @ m.jnt_axis[j]
            inner = -opening / np.linalg.norm(opening)
            out.append(np.mean([d.geom_xpos[g] + m.geom_size[g][0] * inner for g in self.caps[b]], 0))
        return out

    def clearance_mm(self, t):
        """Rendered pad-to-pad distance (mm) at frame t (equals clip(c_F) by construction; checked in tests)."""
        self._set(self.q_cm[t])
        pl, pr = self.pads()
        return float(np.linalg.norm(pl - pr) * 1000.0)

    def draw(self, t):
        self._set(self.q_cm[t])
        p3 = self.p3.begin(self.look, self.dist, self.az, self.el, keep_gids=self.gids)
        col = STATE_RGBA[int(self.state[t])]
        for g in self.finger_gids:
            k = p3.geom_slot.get(g)
            if k is not None:
                p3.scene.geoms[k].rgba[:] = col
                p3.scene.geoms[k].matid = -1
        r = self.d_mm / 2000.0
        if self.obj_type == 'sphere':
            p3.sphere(self.centre, r, OBJ_RGBA)
        else:
            hl = float(np.clip(self.half_len, 2.0, 3.0)) / 100.0  # drawn short so it does not hide the fingers
            p3.capsule(self.centre - hl * self.obj_axis, self.centre + hl * self.obj_axis, r, OBJ_RGBA)
        p3.capsule(self.centre - r * self.u, self.centre + r * self.u, 0.0016, DIAM_RGBA)
        return p3.render()

    def close(self):
        self.p3.close()


# ------------------------------------------------------------------------------------------------ compositor
@dataclass
class E2Panel:
    kind: str          # 'hand' | 'gripper'
    title: str
    c_mm: np.ndarray   # clearance series shown (hand: c_A ; gripper: c_F raw)
    note: str = ''


class TriptychE2:
    def __init__(self, h: HandEpisode, desc: dict, c_ctrl_mm, c_test_mm, labels=('对照组', '实验组'), header='', footer=None):
        self.h, self.desc = h, desc
        self.T = len(h.times)
        self.times = h.times
        self.cA = desc['c_A'] * 10.0
        d = h.obj_d_mm
        self.header = header
        self.footer = footer or (f'橙色虚线 = 物体直径 {d:.0f} mm   绿色带 = 合格夹持间隙（直径 −3 到 +8 mm）   '
                                 '底部色条：绿 = 手正抓着 / 灰 = 手离物体远   灰线 = 人手（灵巧手）自己的间隙')
        self.drawers = [HandDrawer(h, desc),
                        GripperDrawer(c_ctrl_mm, d, h.obj_type, float(np.linalg.norm(h.obj_segment_cm[0, 1] - h.obj_segment_cm[0, 0])) / 2),
                        GripperDrawer(c_test_mm, d, h.obj_type, float(np.linalg.norm(h.obj_segment_cm[0, 1] - h.obj_segment_cm[0, 0])) / 2)]
        vlab = desc.get('variant_name') or desc.get('variant')
        self.panels = [E2Panel('hand', f'灵巧手 + {DS_LABEL.get(h.dataset, h.dataset)}（输入）', self.cA,
                               f'橙线 = 拇指到“虚拟手指”的对握轴（{vlab}）'),
                       E2Panel('gripper', f'{labels[0]} · Franka 夹爪', np.asarray(c_ctrl_mm, np.float64)),
                       E2Panel('gripper', f'{labels[1]} · Franka 夹爪', np.asarray(c_test_mm, np.float64))]
        hi = float(max(90.0, np.ceil(np.nanmax(np.concatenate([self.cA, self.panels[1].c_mm, self.panels[2].c_mm, [d + 10]])) / 10) * 10 + 5))
        hi = min(hi, 160.0)
        lo = float(min(-10.0, np.floor(np.nanmin(np.concatenate([self.cA, self.panels[1].c_mm, self.panels[2].c_mm])) / 10) * 10 - 5))
        lo = max(lo, -60.0)
        self.traces = []
        for k, p in enumerate(self.panels):
            series = [p.c_mm] if k == 0 else [p.c_mm, self.cA]
            cols = [L.INK] if k == 0 else [L.INK, L.GREY]
            labs = ['人手间隙'] if k == 0 else ['夹爪间隙', '人手间隙']
            tr = L.Trace(PANEL_W - 24, TRACE_H, series, cols, self.times, lo=lo, hi=hi, title='手指间隙（mm）', labels=labs,
                         ground=False, bands=[(d - HOLD_LOW_MM, d + HOLD_HIGH_MM, (222, 242, 226))],
                         hlines=[(d, (236, 140, 26))], unit='mm', step=20)
            self._label_strip(tr)
            self.traces.append(tr)

    def _label_strip(self, tr):
        if self.h.labels is None:
            return
        d = ImageDraw.Draw(tr.base)
        y0 = tr.h - tr.pad_b - 6
        for i, g in enumerate(self.h.labels):
            col = {1: (38, 166, 76), 0: (170, 176, 182)}.get(int(g))
            if col is None:
                continue
            x0, x1 = tr.x(i), tr.x(min(i + 1, len(self.h.labels) - 1))
            d.rectangle((x0, y0, max(x1, x0 + 1), y0 + 5), fill=col)

    def view(self, k, t):
        return self.drawers[k].draw(t)

    def frame(self, t) -> Image.Image:
        img = Image.new('RGB', (L.W, L.H), L.BG)
        d = ImageDraw.Draw(img)
        L.text(d, (24, 10), self.header, L.fit_text(d, self.header, 30, L.W - 48, True), L.INK, bold=True)
        sub = (f't = {self.times[t]:5.2f} s   第 {t + 1}/{self.T} 帧   近景，镜头固定在手 / 夹爪自己的坐标系里')
        L.text(d, (24, 54), sub, 21, L.SUB)
        for k, p in enumerate(self.panels):
            x0, y0 = k * PANEL_W, 92
            d.rectangle((x0 + 2, y0, x0 + PANEL_W - 3, y0 + 36), fill=(222, 230, 234))
            L.text(d, (x0 + 14, y0 + 3), p.title, L.fit_text(d, p.title, 24, PANEL_W - 28, True), L.INK, bold=True)
            img.paste(Image.fromarray(self.view(k, t)), (x0, y0 + 38))
            d = ImageDraw.Draw(img)
            yb = y0 + 38 + VIEW_H + 4
            if p.kind == 'hand':
                g = int(self.h.labels[t]) if self.h.labels is not None else None
                s = (LABEL_WORD[g] + '   ' if g is not None else '') + p.note
                L.text(d, (x0 + 14, yb), s, L.fit_text(d, s, 18, PANEL_W - 28), L.SUB)
            else:
                dr = self.drawers[k]
                st = int(dr.state[t])
                s = f'夹爪：{STATE_WORD[st]}   间隙 {p.c_mm[t]:.0f} mm'
                if dr.clipped[t]:
                    s += '（超出行程，画面按行程截断）'
                col = {1: L.GREEN, 0: L.SUB, -1: L.RED}[st]
                L.text(d, (x0 + 14, yb), s, L.fit_text(d, s, 18, PANEL_W - 28, True), col, bold=True)
            img.paste(self.traces[k].frame(t), (x0 + 12, yb + 30))
            d = ImageDraw.Draw(img)
        L.text(d, (24, L.H - 34), self.footer, L.fit_text(d, self.footer, 20, L.W - 48), L.MUTED)
        return img

    def close(self):
        for dr in self.drawers:
            dr.close()


def _offset_stats(drawers, T):
    v = np.array([dr.clearance_mm(t) - (2 * dr.q_cm[t] - FRANKA_OFFSET_CM) * 10 for dr in drawers
                  for t in range(0, T, max(T // 16, 1))])
    return {'mean': float(v.mean()), 'spread': float(v.max() - v.min())}


def _encode(frames_dir, times, out_mp4):
    enc = _env.load_module('rollout_encode', _env.ROOT / 'rollout_tools' / 'encode.py')
    return enc.encode(frames_dir, times, out_mp4)


def render_e2(h: HandEpisode, c_ctrl_mm, c_test_mm, out_dir, labels=('对照组', '实验组'), header='', overwrite=False,
              grid=True, meta=None, max_seconds=1800):
    """Render to out_dir/{triptych.mp4, grid12.png, poster.jpg, manifest.json} (atomic, never overwrites by default).
    c_ctrl_mm / c_test_mm: per-frame clearance of THIS episode (len == len(h.times))."""
    t0 = time.time()
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f'{out_dir} exists (pass overwrite=True)')
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    c_ctrl_mm = np.asarray(c_ctrl_mm, np.float64)
    c_test_mm = np.asarray(c_test_mm, np.float64)
    if len(c_ctrl_mm) != len(h.times) or len(c_test_mm) != len(h.times):
        raise ValueError('clearance series length != episode length')
    tmp = Path(tempfile.mkdtemp(prefix='.' + out_dir.name + '-', dir=out_dir.parent))
    frames_dir = tmp / 'frames'
    frames_dir.mkdir()
    tri = None
    try:
        desc = hand_descriptor(h)
        tri = TriptychE2(h, desc, c_ctrl_mm, c_test_mm, labels, header)
        for t in range(tri.T):
            if time.time() - t0 > max_seconds:
                raise TimeoutError('render exceeded its lifetime')
            img = tri.frame(t)
            img.save(frames_dir / f'{t:06d}.png')
            if t == tri.T // 2:
                img.save(tmp / 'poster.jpg', quality=92)
        timing = _encode(frames_dir, tri.times, tmp / 'triptych.mp4')
        shutil.rmtree(frames_dir)
        if grid:
            from .grid12 import grid_from_video
            grid_from_video(tmp / 'triptych.mp4', tmp / 'grid12.png', times=tri.times, title=header)
        checks = {'gripper_tip_gap_at_cF0_mm': float(tri.drawers[1].tip_gap_mm_at(0.0)),
                  'hand_body_pos_max_err_cm': max(tri.drawers[0].body_pos_check(t) for t in (0, tri.T // 2, tri.T - 1)),
                  # rendered mean-pad gap minus the drawn clearance 2q-0.2 cm: a constant (finger geometry offset of the
                  # c_F convention); its spread over frames must be ~0 (the rendering is exactly the slide value)
                  'gripper_padmean_gap_minus_cF_mm': _offset_stats(tri.drawers[1:], tri.T)}
        sm = lambda c: {k: float(np.mean(gripper_state(c, h.obj_d_mm) == v)) for k, v in (('hold', 1), ('open', 0), ('crush', -1))}  # noqa: E731
        man = {'schema': 'am2exp-render-e2-v1', 'header': header, 'dataset': h.dataset, 'episode': h.episode, 'split': h.split,
               'frames': tri.T, 'rows': [int(h.rows[0]), int(h.rows[-1])], 'stride_rows': int(h.rows[1] - h.rows[0]) if len(h.rows) > 1 else 1,
               'object': {'type': h.obj_type, 'diameter_mm': h.obj_d_mm},
               'descriptor': {'variant': desc.get('variant'), 'tau_cm': desc.get('tau_cm')},
               'time': {k: v for k, v in timing.items() if k != 'frame_pts_seconds'},
               'state_fractions': {'control': sm(c_ctrl_mm), 'test': sm(c_test_mm)},
               'labels_present': h.labels is not None, 'checks': checks, 'meta': meta or {},
               'render_seconds': round(time.time() - t0, 2), 'mujoco_gl': os.environ.get('MUJOCO_GL'),
               'code': str(Path(__file__).resolve())}
        (tmp / 'manifest.json').write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
        tmp.rename(out_dir)
        os.chmod(out_dir, 0o775)  # mkdtemp creates 0700
        return man
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        if tri is not None:
            tri.close()


ARM_WORD = {'AN': '解析映射（对照）', 'constant_open': '一直张开（地板）', 'constant_closed': '一直闭合（地板）',
            'constant_median': '固定中位间隙（地板）'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='hammer', choices=['hammer', 'relocate'])
    ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--control', required=True, help='arm name under OUT/eval/e2/arms, dir or npz')
    ap.add_argument('--test', required=True)
    ap.add_argument('--labels', default=None, help="'control,test' panel labels")
    ap.add_argument('--stride', type=int, default=2, help='row stride (100 Hz data; 2 -> 50 frames/s, real time)')
    ap.add_argument('--frames', default=None, help='a,b slice of the episode frames before striding')
    ap.add_argument('--out', required=True)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=1800)
    a = ap.parse_args(argv)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('lifetime exceeded')))
    signal.alarm(int(a.max_seconds) + 120)
    cc, lab_c, pc = load_arm_cF(a.control, a.dataset)
    ct, lab_t, pt = load_arm_cF(a.test, a.dataset)
    labels = lab_t if lab_t is not None else lab_c
    fr = tuple(int(x) for x in a.frames.split(',')) if a.frames else None
    h = load_hand_episode(a.dataset, a.episode, a.stride, fr, labels=labels)
    if a.labels:
        lc, lt = a.labels.split(',')
    else:
        lc, lt = ARM_WORD.get(a.control, a.control), ARM_WORD.get(a.test, a.test)
    header = (f'{a.dataset}-human 第 {a.episode} 段（{h.split}）· {h.times[-1]:.1f} 秒 · 灵巧手 → Franka 夹爪 · '
              f'物体直径 {h.obj_d_mm:.0f} mm')
    man = render_e2(h, cc[h.rows], ct[h.rows], a.out, (lc, lt), header, overwrite=a.overwrite,
                    meta={'control': pc, 'test': pt, 'dataset': a.dataset, 'episode': a.episode, 'stride': a.stride},
                    max_seconds=a.max_seconds)
    signal.alarm(0)
    print(json.dumps({'out': a.out, 'frames': man['frames'], 'duration_s': man['time']['duration_seconds'],
                      'checks': man['checks'], 'render_seconds': man['render_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
