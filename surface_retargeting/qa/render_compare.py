"""Side-by-side comparison video (1920x1080, real-time speed): [human SMPL-X surface points + object + floor |
CPU reference | GPU strict], robots drawn as their dense surface points (4096).
Colours (robots, judged per point with one ruler): green = within [-0.5, 1] mm of the ground (contact),
orange = within [-0.5, 1] mm of the object, red = penetrating > 0.5 mm (ground or object), grey otherwise.
Human: green = point of a region with a human ground-contact label, orange = within 1 cm of the object.
Text: frame index, GPU flags, per-frame max penetration (ground / object / self, from eval/judge.py) per column.
With --cols umr the middle column is the GPU UMR solution before the strict pass ("GPU UMR raw") and every robot
column also prints the root step of this frame (cm/frame at 30 fps) and, for the strict column, its root offset from
UMR and the step excess over UMR (the temporal-continuity gate quantity).
Usage: render_compare.py <clip> <robot> [--out-fps 30] [--frames a:b] [--cols cpu|umr] [--slow k] [--tag name]
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, '/home/arenalabs/umr_gpu'); sys.path.insert(0, '/home/arenalabs/umr_gpu/eval')
from surface_retargeting.gpu import labels as L
from judge import floor_model

OUT = Path('/var/tmp/alphamotion/umr_factory/umr_gpu')
W, H = 640, 1080
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def cam(az=-35.0, el=12.0):
    a, e = np.deg2rad(az), np.deg2rad(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz          # world -> camera (x right, y depth, z up)


class RobotPts:
    def __init__(self, robot):
        self.m, _, gmap = floor_model(OUT / 'robots' / robot / f'{robot}.xml')
        self.d = mujoco.MjData(self.m)
        s = np.load(OUT / 'robots' / robot / 'surface.npz')
        self.pl = s['points_local'].astype(np.float64); self.pg = gmap[s['geom'].astype(np.int64)]

    def __call__(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d)
        R = self.d.geom_xmat[self.pg].reshape(-1, 3, 3)
        return np.einsum('nij,nj->ni', R, self.pl) + self.d.geom_xpos[self.pg]


def draw_points(img, P, col, C, center, scale, y0):
    Pc = (P - center) @ C.T
    u = (W / 2 + Pc[:, 0] * scale).astype(int); v = (y0 - Pc[:, 2] * scale).astype(int)
    order = np.argsort(-Pc[:, 1])                       # far first
    a = np.array(img)
    hi = (col[:, 0] > 200) | (col[:, 1] > 160) & (col[:, 0] < 60)      # red / orange / green: drawn larger, on top
    for sel, rad in ((~hi, 1), (hi, 3)):
        o = order[sel[order]]
        for du in range(rad + 1):
            for dv in range(rad + 1):
                uu = u[o] + du - rad // 2; vv = v[o] + dv - rad // 2
                ok = (uu >= 0) & (uu < a.shape[1]) & (vv >= 0) & (vv < a.shape[0])
                a[vv[ok], uu[ok]] = col[o][ok]
    return Image.fromarray(a)


def floor_grid(d, C, center, scale, y0, half=1.5, step=0.25):
    for x in np.arange(-half, half + 1e-9, step):
        for seg in (np.array([[x, -half, 0], [x, half, 0]]), np.array([[-half, x, 0], [half, x, 0]])):
            P = seg + [center[0], center[1], 0]
            Pc = (P - center) @ C.T
            pts = [(W / 2 + p[0] * scale, y0 - p[2] * scale) for p in Pc]
            d.line(pts, fill=(205, 205, 205), width=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('clip'); ap.add_argument('robot'); ap.add_argument('--out-fps', type=float, default=30.0)
    ap.add_argument('--frames', default=''); ap.add_argument('--cols', default='cpu', choices=['cpu', 'umr'])
    ap.add_argument('--slow', type=int, default=1); ap.add_argument('--tag', default='')
    ap.add_argument('--outdir', default='')
    ap.add_argument('--results', default='', help='parity dir (default results/parity; e.g. results/export_slip_v3/parity)')
    a = ap.parse_args()
    PD = Path(a.results) if a.results else OUT / 'results' / 'parity'
    S = np.load(PD / f'{a.clip}__{a.robot}.series.npz', allow_pickle=True)
    res = json.load(open(PD / f'{a.clip}__{a.robot}.json'))
    ex_npz = PD.parent / f'{a.clip}__{a.robot}.npz' if a.results else None
    EXZ = np.load(ex_npz, allow_pickle=True) if ex_npz is not None and ex_npz.exists() else None
    P = np.load(OUT / 'problems' / a.clip / f'{a.robot}.stock.cpu.npz', allow_pickle=True)
    floor = float(S['floor']); fps = float(S['fps'])
    Hs = P['src'].astype(np.float64); Hs[..., 2] -= floor
    hreg = L.part_to_region(P['part_ids'])
    hc = S['human_contact']
    flags = S['flags']
    has_fac = 'q_cpu_factory_strict' in S.files
    cpu_name = 'cpu_factory_strict' if has_fac else 'cpu_umr'
    Qc = (S['q_cpu_factory_strict'] if has_fac else S['q_cpu_umr']).copy(); Qc[:, 2] -= floor
    if a.cols == 'umr':
        cpu_name = 'gpu_umr'
        Qc = S['q_gpu_umr'].copy(); Qc[:, 2] -= floor
    Qg = S['q_gpu_strict'].copy(); Qg[:, 2] -= floor
    s30 = fps / 30.0
    step_c = np.r_[0.0, np.linalg.norm(np.diff(Qc[:, :3], axis=0), axis=1)] * s30 * 100
    step_g = np.r_[0.0, np.linalg.norm(np.diff(Qg[:, :3], axis=0), axis=1)] * s30 * 100
    exc_g = np.r_[0.0, np.linalg.norm(np.diff(Qg[:, :3], axis=0) - np.diff(Qc[:, :3], axis=0), axis=1)] * s30 * 100
    off_g = np.linalg.norm(Qg[:, :3] - Qc[:, :3], axis=1) * 100
    T = len(Qg)
    pose = S['obj_pose'] if S['obj_pose'].size else None
    obj = None
    if pose is not None:
        sys.path.insert(0, '/home/arenalabs/umr_gpu')
        from surface_retargeting.gpu.geometry import object_from_objpen
        V, F, _ = object_from_objpen(str(P['objpen_mjb']), P['objpen_object_geoms'])
        c = V.mean(0); r = np.linalg.norm(V - c, axis=1).mean()
        dirs = np.random.default_rng(0).normal(size=(1500, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        obj = (c, r, c + dirs * r)
    rp = RobotPts(a.robot)
    C = cam()
    font = ImageFont.truetype(FONT, 24); small = ImageFont.truetype(FONT, 19)
    ser = lambda k: S[k] if k in S.files else None

    def obj_world(t):
        Rm = Rot.from_quat([pose[t, 4], pose[t, 5], pose[t, 6], pose[t, 3]]).as_matrix()
        return obj[2] @ Rm.T + pose[t, :3], Rm, pose[t, :3]

    def obj_dist(X, t):
        Rm = Rot.from_quat([pose[t, 4], pose[t, 5], pose[t, 6], pose[t, 3]]).as_matrix()
        return np.linalg.norm((X - pose[t, :3]) @ Rm - obj[0], axis=1) - obj[1]

    def robot_col(X, t):
        col = np.tile(np.array([150, 150, 160], np.uint8), (len(X), 1))
        z = X[:, 2]
        col[(z >= -0.0005) & (z <= 0.001)] = [20, 170, 40]
        if obj is not None:
            dd = obj_dist(X, t)
            col[(dd >= -0.0005) & (dd <= 0.001)] = [255, 140, 0]
            col[dd < -0.0005] = [230, 20, 20]
        col[z < -0.0005] = [230, 20, 20]
        return col

    def frame(t):
        img = Image.new('RGB', (3 * W, H), (252, 252, 252))
        d = ImageDraw.Draw(img)
        center = np.array([Qg[t, 0], Qg[t, 1], 0.0])
        hcen = np.array([Hs[t][:, 0].mean(), Hs[t][:, 1].mean(), 0.0])
        scale = 500.0; y0 = H - 130
        cols = [('human (SMPL-X surface)', None), ('GPU UMR raw (before strict)' if a.cols == 'umr' else cpu_name.replace('_', ' '), Qc),
                ('GPU strict (umr_gpu)', Qg)]
        for k, (title, Q) in enumerate(cols):
            sub = Image.new('RGB', (W, H), (252, 252, 252)); sd = ImageDraw.Draw(sub)
            cc = hcen if Q is None else center
            floor_grid(sd, C, cc, scale, y0)
            if obj is not None:
                Po, _, _ = obj_world(t)
                sub = draw_points(sub, Po, np.tile(np.array([90, 120, 220], np.uint8), (len(Po), 1)), C, cc, scale, y0)
            if Q is None:
                X = Hs[t]
                col = np.tile(np.array([170, 150, 140], np.uint8), (len(X), 1))
                col[hc[t][hreg]] = [20, 170, 40]
                if obj is not None:
                    col[obj_dist(X, t) < 0.01] = [255, 140, 0]
                sub = draw_points(sub, X, col, C, cc, scale, y0)
                lines = [f'human contact regions: {", ".join(L.REGIONS[r] for r in np.flatnonzero(hc[t])) or "-"}']
            else:
                X = rp(Q[t])
                sub = draw_points(sub, X, robot_col(X, t), C, cc, scale, y0)
                name = cpu_name if k == 1 else 'gpu_strict'
                gm = ser(f'{name}__ground_min'); sm = ser(f'{name}__self_min'); om = ser(f'{name}__obj_min')
                pen = lambda v: max(0.0, -float(v) * 1000) if v is not None and np.isfinite(v) else 0.0
                lines = [f'max pen  ground {pen(gm[t]) if gm is not None else 0:.2f} mm   self {pen(sm[t]) if sm is not None else 0:.2f} mm'
                         + (f'   object {pen(om[t]):.2f} mm' if om is not None and len(om) else ''),
                         f'lowest dense point {X[:, 2].min() * 1000:+.2f} mm   contact pts {int(((X[:, 2] >= -0.0005) & (X[:, 2] <= 0.001)).sum())}']
                if a.cols == 'umr':
                    lines.append(f'root step {(step_c if k == 1 else step_g)[t]:5.2f} cm/frame (30 fps)'
                                 + (f'   offset vs UMR {off_g[t]:5.2f} cm   excess {exc_g[t]:5.2f} cm' if k == 2 else ''))
                if k == 2:
                    drop = int(flags[t]) & 255
                    lines.append(f'GPU flags {int(flags[t])}' + ('  (EXCLUDED)' if drop else '')
                                 + ('  256 FOLLOW HUMAN SLIP (kept)' if int(flags[t]) & 256 else ''))
                    if EXZ is not None and 'ground_slip_speed' in EXZ.files:
                        rsp = EXZ['ground_slip_speed'][t].astype(float)
                        hsp = EXZ['human_ground_slip_speed'][t].astype(float) if 'human_ground_slip_speed' in EXZ.files else np.full(L.NR, np.nan)
                        fmt = lambda v: '  -  ' if not np.isfinite(v) else f'{v * 100:5.1f}'
                        lines.append('slip cm/s robot|human  L_foot ' + fmt(rsp[L.RID['L_foot']]) + '|' + fmt(hsp[L.RID['L_foot']])
                                     + '   R_foot ' + fmt(rsp[L.RID['R_foot']]) + '|' + fmt(hsp[L.RID['R_foot']]))
            sd = ImageDraw.Draw(sub)
            sd.text((14, 12), title, fill=(0, 0, 0), font=font)
            for i, ln in enumerate(lines):
                sd.text((14, 50 + 28 * i), ln, fill=(160, 0, 0) if 'EXCLUDED' in ln else (20, 20, 20), font=small)
            img.paste(sub, (k * W, 0))
        d = ImageDraw.Draw(img)
        for k in (1, 2):
            d.line([(k * W, 0), (k * W, H)], fill=(120, 120, 120), width=2)
        d.text((14, H - 60), f'{a.clip}  ->  {a.robot}    frame {t}/{T - 1}  ({t / fps:6.2f} s @ {fps:.0f} fps)'
               + (f'   playback {a.slow}x slower' if a.slow > 1 else ''), fill=(0, 0, 0), font=small)
        d.text((14, H - 32), 'green: ground contact [-0.5,1] mm   orange: object contact   red: penetration > 0.5 mm   (same point-cloud ruler for both robots)',
               fill=(60, 60, 60), font=small)
        return img

    step = max(1, int(round(fps / a.out_fps)))
    rng = range(0, T, step)
    if a.frames:
        s0, s1 = (int(x) for x in a.frames.split(':')); rng = range(s0, min(s1, T), step)
    vd = Path(a.outdir) if a.outdir else OUT / 'videos'; vd.mkdir(parents=True, exist_ok=True)
    stem = f'{a.clip}__{a.robot}' + (f'__{a.tag}' if a.tag else '')
    outp = vd / f'{stem}.mp4'
    w = imageio.get_writer(str(outp), fps=fps / step / a.slow, codec='libx264', quality=5, macro_block_size=8)
    for t in rng:
        w.append_data(np.asarray(frame(t)))
    w.close()
    fr = np.linspace(rng.start, rng.stop - 1, 12).astype(int)
    sheet = Image.new('RGB', (3 * (3 * W // 3), 4 * (H // 3)), (255, 255, 255))
    for i, t in enumerate(fr):
        im = frame(t).resize((3 * W // 3, H // 3))
        sheet.paste(im, ((i % 3) * (3 * W // 3), (i // 3) * (H // 3)))
    sheet.save(vd / f'{stem}_grid12.png')
    print('saved', outp)


if __name__ == '__main__':
    main()
