"""Step 4 -- comparison video + 12-frame grid for one unitree_r1 QA clip: [human source points | robot dense points].
Robot colours (from measure_r1.py state): green = contact point (clearance in [-0.5, 1] mm to floor/object),
red = penetration (< -0.5 mm), orange = hovering 1-20 mm while the human region is in contact, grey = free.
Human: green = contact points (same ruler as the factory, <= 1 cm in a contact region), blue-grey = free.
Object points dark.  Orthographic 3/4 view, each panel centred on its own body, floor z = 0 grid.
Usage: MUJOCO_GL=egl timeout 1800 <umr env python> render_r1.py <qa_id> [--video] [--grid]"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import r1_common as C

COL_R = {0: (165, 165, 170), 1: (0, 190, 0), 2: (230, 0, 0), 3: (255, 140, 0)}
COL_H = {0: (150, 170, 205), 1: (0, 190, 0)}
COL_OBJ = (70, 70, 70)
AZ = np.deg2rad(35.0); EL = np.deg2rad(12.0)          # camera azimuth from the +x (front) axis, elevation


def cam_basis(az=AZ, el=EL):
    f = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])   # viewing direction (into scene)
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return f, right, up


F, RGT, UP = cam_basis()


def splat(arr, X, cols, center, scale, ox, oy, r=1):
    """Draw points X [N,3] with colours [N,3] into arr (H,W,3) using painter's order (far first)."""
    H, W, _ = arr.shape
    Y = X - center
    u = (ox + (Y @ RGT) * scale).astype(int); v = (oy - (Y @ UP) * scale).astype(int)
    dep = -(Y @ F)                                         # larger = closer
    o = np.argsort(dep)
    u, v, c = u[o], v[o], cols[o]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            arr[vv[ok], uu[ok]] = c[ok]


def floor_grid(img, center, scale, ox, oy, ext=1.5, step=0.25):
    dr = ImageDraw.Draw(img)
    cx, cy = np.round(center[0] / step) * step, np.round(center[1] / step) * step
    for k in np.arange(-ext, ext + 1e-9, step):
        for a, b in (((cx + k, cy - ext, 0), (cx + k, cy + ext, 0)), ((cx - ext, cy + k, 0), (cx + ext, cy + k, 0))):
            p = [np.array(a) - center, np.array(b) - center]
            uv = [(ox + q @ RGT * scale, oy - q @ UP * scale) for q in p]
            dr.line(uv, fill=(222, 222, 222), width=1)


class Scene:
    def __init__(self, qa_id):
        self.qa_id = qa_id
        man = [json.loads(l) for l in open(C.PDIR / 'out' / 'manifest.jsonl')]
        row = [r for r in man if r['qa_id'] == qa_id and r['status'].startswith('ok')][-1]
        self.o = np.load(row['npz'], allow_pickle=True)
        prov = json.loads(str(self.o['provenance']))
        self.Q = self.o['qpos'].astype(float); self.T = len(self.Q)
        floor = float(self.o['floor_offset_removed'])
        U = np.load(prov['umr_out'], allow_pickle=True)
        Hs = np.load(prov['human_slots'])['source_slots']
        if len(Hs) != self.T:
            Hs = Hs[U['frame_ids']]
        self.Hs = (Hs - np.array([0, 0, floor], np.float32)).astype(np.float32)
        qd = C.QADIR / qa_id
        self.S = np.load(qd / 'state.npz')
        self.meas = json.load(open(qd / 'measure.json'))
        self.has_obj = 'obj_pts_local' in self.S.files
        self.m = C.load_model(); self.P = C.Poser(self.m, C.load_surface())
        self.clip = row['clip']

    def robot_pts(self, t):
        self.P.set(self.Q[t]); return self.P.points()

    def obj_pts(self, t):
        if not self.has_obj:
            return None
        return self.S['obj_pts_local'] @ self.S['orot'][t].T + self.S['opos'][t]

    def panel(self, t, which, W, H, scale):
        arr = np.full((H, W, 3), 252, np.uint8)
        img = Image.fromarray(arr)
        if which == 'human':
            X = self.Hs[t].astype(float); st = self.S['hstate'][t]; cmap = COL_H
        else:
            X = self.robot_pts(t); st = self.S['state'][t]; cmap = COL_R
        center = np.array([X[:, 0].mean(), X[:, 1].mean(), 0.0])
        ox, oy = W / 2, H - 0.08 * H
        floor_grid(img, center, scale, ox, oy)
        arr = np.asarray(img).copy()
        op = self.obj_pts(t)
        if op is not None:
            splat(arr, op, np.tile(np.array(COL_OBJ, np.uint8), (len(op), 1)), center, scale, ox, oy, r=1)
        cols = np.array([cmap[int(s)] for s in range(4) if s in cmap], np.uint8)
        lut = np.zeros((4, 3), np.uint8)
        for k, c in cmap.items():
            lut[k] = c
        free = st == 0
        splat(arr, X[free], lut[st[free]], center, scale, ox, oy, r=1)
        hot = ~free
        if hot.any():
            splat(arr, X[hot], lut[st[hot]], center, scale, ox, oy, r=2)
        return Image.fromarray(arr)

    def frame(self, t, W=1280, H=720, scale=300, small=False):
        img = Image.new('RGB', (W, H), (252, 252, 252))
        img.paste(self.panel(t, 'human', W // 2, H, scale), (0, 0))
        img.paste(self.panel(t, 'robot', W // 2, H, scale), (W // 2, 0))
        dr = ImageDraw.Draw(img)
        dr.line([(W // 2, 0), (W // 2, H)], fill=(120, 120, 120), width=2)
        S = self.S
        fails = [n for n, k in (('pen', 'fail_pen'), ('not-solid', 'fail_nsolid'), ('slip>=1cm/s', 'fail_slip')) if S[k][t]]
        sm = float(S['self_min'][t]) * 1000
        C.text(img, (10, 8), f'human source (UMR slots, scaled)   t={t}/{self.T - 1}', small=small)
        C.text(img, (W // 2 + 10, 8), f'unitree_r1 output   flag={int(self.o["flags"][t])}  self_min={sm:.1f}mm', small=small)
        C.text(img, (W // 2 + 10, 30 if not small else 24), ('FAIL: ' + ', '.join(fails)) if fails else 'ok',
               fill=(200, 0, 0) if fails else (0, 140, 0), small=small)
        if not small:
            C.text(img, (10, H - 26), f'{self.clip}   green=contact  red=penetration  orange=hover 1-20mm while human in contact', small=True)
        return img


def main():
    qa_id = sys.argv[1]
    sc = Scene(qa_id)
    qd = C.QADIR / qa_id
    if '--grid' in sys.argv:
        fr = np.linspace(0, sc.T - 1, 12).round().astype(int)
        cw, ch = 760, 440
        sheet = Image.new('RGB', (4 * cw, 3 * ch + 30), (255, 255, 255))
        for i, t in enumerate(fr):
            sheet.paste(sc.frame(int(t), cw, ch, scale=185, small=True), ((i % 4) * cw, (i // 4) * ch))
        for i in range(1, 4):
            ImageDraw.Draw(sheet).line([(i * cw, 0), (i * cw, 3 * ch)], fill=(0, 0, 0), width=3)
        for j in range(1, 3):
            ImageDraw.Draw(sheet).line([(0, j * ch), (4 * cw, j * ch)], fill=(0, 0, 0), width=3)
        C.text(sheet, (10, 3 * ch + 6), f'{sc.clip} -> unitree_r1   each cell: human | robot   green=contact red=penetration '
                                        'orange=hover(1-20mm, human in contact)   12 frames evenly over the full window')
        p = qd / f'{qa_id}_grid12.png'; sheet.save(p); print('saved', p)
    if '--video' in sys.argv:
        import imageio.v2 as imageio
        p = qd / f'{qa_id}_human_vs_r1.mp4'
        w = imageio.get_writer(p, fps=float(sc.o['fps']), codec='libx264', quality=7, macro_block_size=8)
        for t in range(sc.T):
            w.append_data(np.asarray(sc.frame(t)))
        w.close(); print('saved', p)


if __name__ == '__main__':
    main()
