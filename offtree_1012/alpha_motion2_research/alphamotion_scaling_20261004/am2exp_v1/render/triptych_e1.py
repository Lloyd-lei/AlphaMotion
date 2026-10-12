"""E1 triptych renderer: [human source | control arm | test arm]  (stage 0: [human | GT | A0]).

1920x1080 H.264, one video frame per source frame at the real timestamps (rollout_tools/encode.encode, VFR,
PTS verified).  Robots = vendor MJCF VISUAL meshes rigidly re-posed from the given global rotations/positions
(render/robot_model.py), placed at the given root height (root convention printed under every panel), checker
floor, contact colouring per body region (green = touching < 2.5 cm, red = penetrating > 1 cm; collision-geom
lowest point, see render/contact.py), thin grey GT ghost in arm panels, per-panel lowest-point trace.

Follow camera: positions are root-relative (the model predicts no horizontal root), so every panel is moved
horizontally along the SOURCE root trajectory (displacement from the first window frame), scaled by
reach(panel body)/reach(source body); each camera follows its own panel's (0.5 s moving-average) root.
The floor checker is world-fixed (1 m tiles of 0.5 m squares), so sliding feet are visible.

CLI
  python -m render.triptych_e1 --control CTRL.npz --test TEST.npz --out DIR [--labels A0,S]
  python -m render.triptych_e1 --stage0 CASE.npz --out DIR                  # [human | GT | arm]
  python -m render.triptych_e1 --gt-self CASE.npz --out DIR                 # [human | GT | GT] identity check
  python -m render.triptych_e1 --demo --out DIR                             # GT demo, fallAndGetUp1_s4 [4136,4264)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'render'

from . import _env  # noqa: E402
from ._env import AX  # noqa: E402
from . import contact as C  # noqa: E402
from . import layout as L  # noqa: E402
from .case_io import Window, load_case, load_gt_window  # noqa: E402
from .regions import human_bone_regions, human_joint_regions  # noqa: E402
from .robot_model import load_robot  # noqa: E402
from .scene3d import Panel3D, canon_to_world  # noqa: E402

PANEL_W, VIEW_H = 640, 690
ELEVATION = -18.0
NEUTRAL = np.array([0.80, 0.82, 0.85, 1.0], np.float32)
GREEN = np.array([0.16, 0.72, 0.32, 1.0], np.float32)
RED = np.array([0.90, 0.16, 0.16, 1.0], np.float32)
HUMAN_BONE = np.array([0.22, 0.42, 0.72, 1.0], np.float32)
GHOST = np.array([0.30, 0.30, 0.34, 0.55], np.float32)
STATE_RGBA = {0: None, 1: GREEN, 2: RED}


@dataclass
class PanelSpec:
    kind: str                       # 'human' | 'robot'
    title: str
    win: Window
    root_offset_cm: float = 0.0     # subtracted from 100*root_h_m
    root_note: str = ''
    ghost: Window | None = None
    ghost_offset_cm: float = 0.0
    contact_source: str = 'auto'
    # filled by prepare()
    root_y: np.ndarray = None       # [T] cm
    xz: np.ndarray = None           # [T,2] cm
    region_h: np.ndarray = None
    zmin: np.ndarray = None
    state: np.ndarray = None
    contact_used: str = ''
    ghost_root_y: np.ndarray = None
    ghost_zmin: np.ndarray = None
    extra: dict = field(default_factory=dict)


def default_root_offset(body: str, kind: str = 'gt'):
    """Root convention identical to judge J1 corrected_root (kind 'gt' | 'pred'); see _env.root_offset_cm."""
    v, tag, src = _env.root_offset_cm(body, kind)
    return float(v), src


def root_note(offset_cm: float, src: str, predicted: bool) -> str:
    who = '模型预测根高' if predicted else '标签根高'
    if abs(offset_cm) < 0.05:
        return f'{who}（无修正）'
    sign = '−' if offset_cm > 0 else '+'
    if 'heldout_root_bias' in src:
        why = '没见过的机器人：按 A0 站立帧量出的常数修正，各组一样'
    elif 'provisional' in src:
        why = '没见过的机器人：常数还没量，暂用参考修正'
    else:
        why = '报告32 鞋底修正'
    return f'{who} {sign} {abs(offset_cm):.1f} cm（{why}，与评分一致）'


def _root_y(win: Window, offset_cm: float, lowest_rel=None):
    """[T] canonical root y in cm.  Unknown root height -> lowest point snapped to the floor (p5 over window)."""
    if win.root_h_m is not None and np.isfinite(win.root_h_m).any():
        h = np.asarray(win.root_h_m, np.float64).copy()
        if not np.isfinite(h).all():  # fill gaps by interpolation
            idx = np.arange(len(h))
            ok = np.isfinite(h)
            h[~ok] = np.interp(idx[~ok], idx[ok], h[ok])
        return 100.0 * h - offset_cm, False
    low = lowest_rel if lowest_rel is not None else win.P[..., 1].min(-1)
    return np.full(len(win.P), -np.percentile(low, 5)), True


def _facing_azimuth(src: Window) -> float:
    """Camera azimuth (deg, MuJoCo world) giving a front-left 3/4 view of the source's mean facing direction."""
    names = list(src.names)
    try:
        if 'L_Hip' in names:
            lh, rh = names.index('L_Hip'), names.index('R_Hip')
        else:
            rm = load_robot(src.body)
            lh, rh = rm.region_info['leg_chains'][0][0], rm.region_info['leg_chains'][1][0]
        lat = (src.P[:, lh] - src.P[:, rh]).mean(0)
        fwd_c = np.cross(lat, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(fwd_c[[0, 2]]) < 1e-6:
            raise ValueError
        fw = AX.T @ fwd_c
        phi = np.degrees(np.arctan2(fw[1], fw[0]))
        return float(phi + 180.0 - 35.0)
    except Exception:
        return 135.0


def _smooth(x, n):
    if n <= 1 or len(x) < 2:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, ((pad, n - 1 - pad), (0, 0)), mode='edge')
    return np.stack([np.convolve(xp[:, i], k, mode='valid') for i in range(x.shape[1])], -1)


def prepare(panels, src: Window, fps: float, follow: bool = True):
    T = len(src.P)
    if follow and src.world_root_cm is not None:
        disp = src.world_root_cm[:, [0, 2]] - src.world_root_cm[:1, [0, 2]]
    else:
        disp = np.zeros((T, 2))
    for p in panels:
        scale = p.win.reach / src.reach if src.reach > 0 else 1.0
        p.xz = disp * scale
        if p.kind == 'robot':
            rm = load_robot(p.win.body)
            low = rm.lowest_y(rm.collision if rm.collision is not None else rm.visual, p.win.R, p.win.P).min(-1)
            p.root_y, snapped = _root_y(p.win, p.root_offset_cm, low)
            p.region_h, p.zmin, p.contact_used = C.robot_region_heights(rm, p.win.R, p.win.P, p.root_y, p.contact_source)
            p.state = C.states(p.region_h)
            if p.ghost is not None:
                glow = rm.lowest_y(rm.collision if rm.collision is not None else rm.visual, p.ghost.R, p.ghost.P).min(-1)
                p.ghost_root_y, _ = _root_y(p.ghost, p.ghost_offset_cm, glow)
                _, p.ghost_zmin, _ = C.robot_region_heights(rm, p.ghost.R, p.ghost.P, p.ghost_root_y, p.contact_source)
        else:
            p.root_y, snapped = _root_y(p.win, p.root_offset_cm)
            dense = C.human_dense_points(p.win.R, p.win.P) if p.win.R is not None else None
            p.extra['dense'] = dense
            p.region_h, p.zmin, st, p.contact_used = C.human_region_heights(p.win, p.root_y, dense)
            p.state = st if st is not None else C.states(p.region_h)
        if snapped:
            p.root_note = '根高未知：按最低点贴地'
    return panels


class RobotDrawer:
    def __init__(self, body):
        self.rm = load_robot(body)
        self.p3 = Panel3D(PANEL_W, VIEW_H, self.rm.xml, reference_model=self.rm.model)

    def precompute(self, p: PanelSpec):
        rm = self.rm
        Gc, c = rm.geom_poses(rm.visual, p.win.R, p.win.P)
        root = np.stack([p.xz[:, 0], p.root_y, p.xz[:, 1]], -1)
        self.pos = canon_to_world(c + root[:, None])
        self.mat = (AX.T[None, None] @ Gc @ AX[None, None]).reshape(len(c), -1, 9)
        if p.ghost is not None:
            groot = np.stack([p.xz[:, 0], p.ghost_root_y, p.xz[:, 1]], -1)
            self.ghost = canon_to_world(p.ghost.P + groot[:, None])
        else:
            self.ghost = None

    def draw(self, p: PanelSpec, t, cam):
        rm = self.rm
        self.p3.begin(*cam, keep_gids=rm.visual.gid)
        self.p3.floor(cam[0][:2])
        st = p.state[t]
        for k, gid in enumerate(rm.visual.gid):
            col = STATE_RGBA[int(st[rm.visual.region[k]])] if rm.visual.region[k] >= 0 else None
            self.p3.set_geom(gid, self.pos[t, k], self.mat[t, k], NEUTRAL if col is None else col, plain=True)
        if self.ghost is not None:
            for j, pa in enumerate(rm.parents):
                if pa >= 0:
                    self.p3.capsule(self.ghost[t, pa], self.ghost[t, j], 0.006, GHOST)
            for j in range(len(rm.parents)):
                self.p3.sphere(self.ghost[t, j], 0.009, GHOST)
        return self.p3.render()

    def close(self):
        self.p3.close()


class HumanDrawer:
    def __init__(self, names, parents):
        self.names = list(names)
        self.parents = np.asarray(parents)
        self.bone_reg = human_bone_regions(self.names)
        self.joint_reg = human_joint_regions(self.names)
        self.p3 = Panel3D(PANEL_W, VIEW_H, None)

    def precompute(self, p: PanelSpec):
        root = np.stack([p.xz[:, 0], p.root_y, p.xz[:, 1]], -1)
        self.P = canon_to_world(p.win.P + root[:, None])
        dense = p.extra.get('dense')
        if dense is not None:
            pts, reg = dense
            self.dense = canon_to_world(pts + root[:, None])
            self.dense_reg = reg
        else:
            self.dense = None
        n = self.names
        self.head = n.index('Head') if 'Head' in n else None
        self.neck = n.index('Neck') if 'Neck' in n else None
        self.wrists = [(n.index(w), n.index(e)) for w, e in (('L_Wrist', 'L_Elbow'), ('R_Wrist', 'R_Elbow')) if w in n and e in n]

    def _col(self, st, r):
        c = STATE_RGBA[int(st[r])]
        return HUMAN_BONE if c is None else c

    def draw(self, p: PanelSpec, t, cam):
        self.p3.begin(*cam, keep_gids=[])
        self.p3.floor(cam[0][:2])
        st = p.state[t]
        P = self.P[t]
        if self.dense is not None:
            for k in range(0, self.dense.shape[1]):
                c = STATE_RGBA[int(st[self.dense_reg[k]])] if self.dense_reg[k] < 15 else None
                self.p3.sphere(self.dense[t, k], 0.007, np.array([0.55, 0.62, 0.72, 0.9], np.float32) if c is None else c)
        for j, pa in enumerate(self.parents):
            if pa >= 0:
                self.p3.capsule(P[pa], P[j], 0.016, self._col(st, self.bone_reg[j]))
        for j in range(len(self.parents)):
            self.p3.sphere(P[j], 0.024, self._col(st, self.joint_reg[j]))
        if self.head is not None and self.neck is not None:
            d = P[self.head] - P[self.neck]
            self.p3.sphere(P[self.head] + 0.6 * d, 0.085, self._col(st, self.joint_reg[self.head]))
        for w, e in self.wrists:
            d = P[w] - P[e]
            d = d / max(np.linalg.norm(d), 1e-6)
            self.p3.capsule(P[w], P[w] + 0.08 * d, 0.022, self._col(st, self.joint_reg[w]))
        return self.p3.render()

    def close(self):
        self.p3.close()


def _cams(p: PanelSpec, az: float, fps: float):
    names, parents, off, reach = _env.static_spec(p.win.body)
    pelvis_h = float(abs(off[0][1])) if abs(off[0][1]) > 20 else 0.75 * reach
    look = _smooth(p.xz, max(int(round(fps * 0.5)), 1))
    look_c = np.stack([look[:, 0], np.full(len(look), 0.42 * pelvis_h), look[:, 1]], -1)
    look_w = canon_to_world(look_c)
    dist = 3.0 * pelvis_h / 100.0
    return [(look_w[t], dist, az, ELEVATION) for t in range(len(look_w))]


def _encode(frames_dir, times, out_mp4):
    enc = _env.load_module('rollout_encode', _env.ROOT / 'rollout_tools' / 'encode.py')
    return enc.encode(frames_dir, times, out_mp4)


class Triptych:
    """Frame-by-frame compositor (also used directly by tests and previews)."""

    def __init__(self, panels, src: Window, times, header, subheader='', footer=None, follow=True):
        self.panels, self.src = panels, src
        self.times = np.asarray(times, np.float64)
        self.T = len(self.times)
        self.fps = 1.0 / float(np.median(np.diff(self.times))) if self.T > 1 else 30.0
        self.header, self.subheader, self.follow = header, subheader, follow
        self.footer = footer or ('绿色 = 贴地（离地 < 2.5 cm）   红色 = 穿进地面（> 1 cm）   灰色细线 = GMR 参考姿态   '
                                 '颜色和曲线按碰撞外形算，与评分同一把尺子')
        prepare(panels, src, self.fps, follow)
        self.az = _facing_azimuth(src)
        self.drawers = []
        for p in panels:
            dr = RobotDrawer(p.win.body) if p.kind == 'robot' else HumanDrawer(p.win.names, p.win.parents)
            dr.precompute(p)
            self.drawers.append(dr)
        self.cams = [_cams(p, self.az, self.fps) for p in panels]
        self.traces = []
        for p in panels:
            series, cols, labs = [p.zmin], [L.INK], ['本栏']
            if p.ghost_zmin is not None:
                series.append(p.ghost_zmin)
                cols.append(L.GREY)
                labs.append('参考')
            self.traces.append(L.Trace(PANEL_W - 24, 150, series, cols, self.times, labels=labs))

    def view(self, k, t) -> np.ndarray:
        """Raw 3D view of panel k at frame t (uint8 [VIEW_H, PANEL_W, 3])."""
        return self.drawers[k].draw(self.panels[k], t, self.cams[k][t])

    def frame(self, t) -> Image.Image:
        T, times = self.T, self.times
        img = Image.new('RGB', (L.W, L.H), L.BG)
        d = ImageDraw.Draw(img)
        L.text(d, (24, 10), self.header, L.fit_text(d, self.header, 30, L.W - 48, True), L.INK, bold=True)
        sub = f't = {times[t] - times[0]:5.2f} s   第 {t + 1}/{T} 帧   {self.subheader}'
        L.text(d, (24, 54), sub, 21, L.SUB)
        for k, p in enumerate(self.panels):
            x0, y0 = k * PANEL_W, 92
            d.rectangle((x0 + 2, y0, x0 + PANEL_W - 3, y0 + 36), fill=(222, 230, 234))
            L.text(d, (x0 + 14, y0 + 3), p.title, L.fit_text(d, p.title, 24, PANEL_W - 28, True), L.INK, bold=True)
            img.paste(Image.fromarray(self.view(k, t)), (x0, y0 + 38))
            d = ImageDraw.Draw(img)
            yb = y0 + 38 + VIEW_H + 4
            L.text(d, (x0 + 14, yb), p.root_note, L.fit_text(d, p.root_note, 17, PANEL_W - 28), L.SUB)
            img.paste(self.traces[k].frame(t), (x0 + 12, yb + 28))
            d = ImageDraw.Draw(img)
        L.text(d, (24, L.H - 34), self.footer, L.fit_text(d, self.footer, 20, L.W - 48), L.MUTED)
        return img

    def close(self):
        for dr in self.drawers:
            dr.close()


def render_triptych(panels, src: Window, times, out_dir, header, subheader='', footer=None, follow=True,
                    overwrite=False, grid=True, meta=None, max_seconds=1800):
    """Render panels (list of PanelSpec) to out_dir/{triptych.mp4, grid12.png, manifest.json, poster.jpg}.
    The directory is written atomically (temp dir + rename); an existing out_dir is never overwritten unless asked."""
    t0 = time.time()
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f'{out_dir} exists (evidence is never overwritten; pass overwrite=True)')
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.' + out_dir.name + '-', dir=out_dir.parent))
    frames_dir = tmp / 'frames'
    frames_dir.mkdir()
    tri = None
    try:
        tri = Triptych(panels, src, times, header, subheader, footer, follow)
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
        man = {'schema': 'am2exp-render-e1-v1', 'header': header, 'frames': tri.T, 'fps_nominal': tri.fps,
               'time': {k: v for k, v in timing.items() if k != 'frame_pts_seconds'},
               'camera': {'azimuth_deg': tri.az, 'elevation_deg': ELEVATION,
                          'follow': 'source root xz displacement x reach ratio, 0.5 s moving average' if follow else 'fixed'},
               'panels': [{'kind': p.kind, 'title': p.title, 'body': p.win.body, 'root_offset_cm': p.root_offset_cm,
                           'root_note': p.root_note, 'contact_source': p.contact_used, 'ghost': p.ghost is not None,
                           'zmin_cm_p50': float(np.nanmedian(p.zmin)),
                           'touch_frac_by_region': np.round((p.state == 1).mean(0), 3).tolist(),
                           'pen_frac_by_region': np.round((p.state == 2).mean(0), 3).tolist(),
                           'robot_audit': load_robot(p.win.body).audit if p.kind == 'robot' else None}
                          for p in panels],
               'meta': meta or {}, 'render_seconds': round(time.time() - t0, 2),
               'mujoco_gl': os.environ.get('MUJOCO_GL'), 'code': str(Path(__file__).resolve())}
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


# ---------------------------------------------------------------------------------------------- builders
ARM_LABEL = {'gt': 'GMR 参考（真值）'}


def panels_from_cases(control_case: dict, test_case: dict, control_label: str, test_label: str,
                      control_is_gt=False, test_is_gt=False, contact_source='auto'):
    src = test_case['source']
    tb = test_case['target_body']
    off, srcnote = default_root_offset(tb, 'gt')
    poff, psrcnote = default_root_offset(tb, 'pred')
    sb = test_case['source_body']
    if sb == _env.HUMAN:
        sp = PanelSpec('human', f'人体动作（输入）', src, 0.0, '人体：标签根高（关节中心地面）')
    else:
        so, ssrc = default_root_offset(sb, 'gt')
        sp = PanelSpec('robot', f'输入动作 · {L.body_label(sb)}', src, so, root_note(so, ssrc, False))

    def arm(case, label, is_gt):
        gt = case['gt']
        if is_gt:
            o = case['offsets'].get('gt_root_offset_cm', off)
            return PanelSpec('robot', f'{label} · {L.body_label(tb)}', gt, o, root_note(o, srcnote, False),
                             ghost=gt, ghost_offset_cm=o, contact_source=contact_source)
        o = case['offsets'].get('pred_root_offset_cm', poff)
        go = case['offsets'].get('gt_root_offset_cm', off)
        note = psrcnote if 'pred_root_offset_cm' not in case['offsets'] else 'case npz pred_root_offset_cm'
        return PanelSpec('robot', f'{label} · {L.body_label(tb)}', case['pred'], o, root_note(o, note, True),
                         ghost=gt, ghost_offset_cm=go, contact_source=contact_source)
    return [sp, arm(control_case, control_label, control_is_gt), arm(test_case, test_label, test_is_gt)], src


def demo_panels(group='lafan::fallAndGetUp1_subject4', start=4136, end=4264, bodies=('unitree_g1_29dof', 'fourier_gr3'),
                contact_source='auto'):
    src = load_gt_window(group, _env.HUMAN, start, end, with_contact=True)
    panels = [PanelSpec('human', '人体动作（输入）', src, 0.0, '人体：标签根高（关节中心地面）')]
    for b in bodies:
        w = load_gt_window(group, b, start, end, with_contact=True)
        o, s = default_root_offset(b)
        panels.append(PanelSpec('robot', f'GMR 参考 · {L.body_label(b)}', w, o, root_note(o, s, False),
                                ghost=None, contact_source=contact_source))
    return panels, src


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--test', help='test-arm case npz (with --control)')
    g.add_argument('--stage0', help='case npz: render [human | GT | arm]')
    g.add_argument('--gt-self', help='case npz: render [human | GT | GT] (identity check)')
    g.add_argument('--demo', action='store_true', help='GT demo from the frozen caches')
    ap.add_argument('--control', help='control-arm case npz')
    ap.add_argument('--labels', default='对照组,实验组', help='control,test panel labels')
    ap.add_argument('--out', required=True)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-follow', action='store_true')
    ap.add_argument('--contact-source', default='auto', choices=['auto', 'judge', 'local'])
    ap.add_argument('--max-seconds', type=float, default=1800)
    ap.add_argument('--demo-group', default='lafan::fallAndGetUp1_subject4')
    ap.add_argument('--demo-window', default='4136,4264')
    ap.add_argument('--demo-bodies', default='unitree_g1_29dof,fourier_gr3')
    a = ap.parse_args(argv)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('lifetime exceeded')))
    signal.alarm(int(a.max_seconds) + 120)
    if a.demo:
        s, e = (int(x) for x in a.demo_window.split(','))
        panels, src = demo_panels(a.demo_group, s, e, tuple(a.demo_bodies.split(',')), a.contact_source)
        clip = a.demo_group.split('::')[-1]
        header = f'{clip}  帧 {s}–{e}（{e - s} 帧，{(e - s) / 30:.2f} 秒）· 人体 → GMR 参考（未训练模型，仅真值）'
        man = render_triptych(panels, src, src.times, a.out, header, subheader='相机跟随人体根部水平轨迹（按身高比例）',
                              footer=None, follow=not a.no_follow, overwrite=a.overwrite,
                              meta={'group_id': a.demo_group, 'start': s, 'end': e, 'kind': 'gt_demo'}, max_seconds=a.max_seconds)
    else:
        if a.test:
            if not a.control:
                ap.error('--test requires --control')
            cc, tc = load_case(a.control), load_case(a.test)
            if cc['target_body'] != tc['target_body'] or cc['source_body'] != tc['source_body']:
                raise ValueError('control/test cases differ in source/target body')
            cl, tl = a.labels.split(',')
            panels, src = panels_from_cases(cc, tc, cl, tl, contact_source=a.contact_source)
            meta = tc['meta']
        elif a.stage0:
            tc = load_case(a.stage0)
            arm = tc['meta'].get('arm', '模型')
            panels, src = panels_from_cases(tc, tc, ARM_LABEL['gt'], f'{arm} 模型', control_is_gt=True, contact_source=a.contact_source)
            meta = tc['meta']
        else:
            tc = load_case(a.gt_self)
            panels, src = panels_from_cases(tc, tc, ARM_LABEL['gt'], ARM_LABEL['gt'], control_is_gt=True, test_is_gt=True,
                                            contact_source=a.contact_source)
            meta = tc['meta']
        clip = meta.get('clip') or str(meta.get('group_id', '')).split('::')[-1]
        s = meta.get('start')
        T = len(tc['times'])
        win = f'帧 {s}–{s + T}' if s is not None else f'{T} 帧'
        mode = {'sensor_on': '接触传感器开', 'sensor_off': '接触传感器关'}.get(meta.get('mode', ''), meta.get('mode', ''))
        header = f'{clip}  {win} · {L.body_label(tc["source_body"])} → {L.body_label(tc["target_body"])}  {mode}'.strip()
        man = render_triptych(panels, src, tc['times'], a.out, header, subheader='相机跟随输入动作的根部水平轨迹（按身高比例）',
                              follow=not a.no_follow, overwrite=a.overwrite, meta=meta, max_seconds=a.max_seconds)
    signal.alarm(0)
    print(json.dumps({'out': a.out, 'frames': man['frames'], 'duration_s': man['time']['duration_seconds'],
                      'render_seconds': man['render_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
