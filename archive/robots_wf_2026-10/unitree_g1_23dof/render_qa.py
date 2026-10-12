"""Step 4 -- visual QA for unitree_g1_23dof: [human points | robot] side by side, same camera, same floor (z = 0).

Robot panel (post-pass output = the dataset pose): dense points coloured
  green  = touching (clearance in [-0.5, 1] mm, ground or object)
  red    = penetrating (< -0.5 mm)
  orange = human region in contact but robot region hovering 1-20 mm (region's lowest points)
  purple = human region in contact but robot region > 20 mm away (missing contact)
Human panel: UMR's 4096 human source slots (robot scale), green = in a contacting region and < 1 cm from the surface.
Usage: python render_qa.py <out npz> <tag> [--mp4] [--grid] [--w 1280 --h 720]
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import g1common as C
import g1measure as M

ap = argparse.ArgumentParser()
ap.add_argument('npz'); ap.add_argument('tag'); ap.add_argument('--mp4', action='store_true'); ap.add_argument('--grid', action='store_true')
ap.add_argument('--w', type=int, default=1280); ap.add_argument('--h', type=int, default=720)
ap.add_argument('--az', type=float, default=135.0); ap.add_argument('--el', type=float, default=-14.0)
ap.add_argument('--dist', type=float, default=2.3)
a = ap.parse_args()
t0 = time.time()
res = M.measure(a.npz)
T = res['T']; fps = res['fps']; reg = res['reg']; hreg = res['hreg']
S = M.SPEC
cg = M.contact_stats(res, 'ground')
co = M.contact_stats(res, 'object') if res['has_obj'] else None
X = res['X'].astype(np.float64); H = res['H'].astype(np.float64)
Q = res['npz']['qpos'].astype(np.float64)

# ---------------- per point colour (robot)
GREEN = np.array([0.1, 0.85, 0.1, 1]); RED = np.array([0.95, 0.05, 0.05, 1]); ORANGE = np.array([1.0, 0.55, 0.0, 1])
PURPLE = np.array([0.6, 0.1, 0.85, 1]); HGREY = np.array([0.55, 0.55, 0.6, 1]); HGREEN = np.array([0.1, 0.75, 0.1, 1])


def robot_colours(t):
    cols = {}
    dz = X[t, :, 2]
    for D, cs in ((dz, cg),) + (((res['Dobj'][t].astype(np.float64), co),) if co else ()):
        mk = cs['_masks']
        rc = np.full(len(C.REGIONS), np.inf)
        np.minimum.at(rc, reg, np.where(np.isfinite(D), D, np.inf))
        for r in range(len(C.REGIONS)):
            if mk['hcon'][t, r] and rc[r] > S['contact_max'] and np.isfinite(rc[r]):
                low = np.flatnonzero((reg == r) & (D <= rc[r] + 0.003))
                c = ORANGE if rc[r] <= S['hover_max'] else PURPLE
                for i in low:
                    cols[int(i)] = c
            if mk['hcon'][t, r] and not np.isfinite(rc[r]) and D is not dz and res['has_obj']:
                ob = res['obj']; c_w = ob.pos[t] + ob.rot[t] @ ob.center
                k = np.flatnonzero(reg == r); dist = np.linalg.norm(X[t, k] - c_w, axis=1)
                for i in k[dist <= dist.min() + 0.01]:
                    cols[int(i)] = PURPLE
        touch = np.flatnonzero((D >= -S['pen_tol']) & (D <= S['contact_max']))
        for i in touch:
            cols[int(i)] = GREEN
        for i in np.flatnonzero(D < -S['pen_tol']):
            cols[int(i)] = RED
    idx = np.array(sorted(cols), int)
    return idx, np.array([cols[i] for i in idx]) if len(idx) else np.zeros((0, 4))


def human_colours(t):
    col = np.tile(HGREY, (len(hreg), 1))
    hz = H[t, :, 2]
    on = cg['_masks']['hcon'][t][hreg] & (hz < S['human_enter'])
    if co is not None:
        on |= co['_masks']['hcon'][t][hreg] & (res['Dh'][t] < S['human_enter'])
    col[on] = HGREEN
    return col


# ---------------- render model: robot (+ object visual) ; collision geoms hidden
spec = mujoco.MjSpec.from_file(str(C.XML))
if res['has_obj']:
    M.add_object_to_spec(spec, res['obj'])
m = spec.compile()
for g in range(m.ngeom):
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    if bname == '__qa_obj':
        m.geom_group[g] = 1; m.geom_rgba[g] = [0.35, 0.55, 0.95, 0.55]
    elif m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g]):
        m.geom_rgba[g][3] = 0.0
m.vis.global_.offwidth = max(2000, a.w); m.vis.global_.offheight = max(2000, a.h)
d = mujoco.MjData(m)
robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and
               mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) != '__qa_obj']
alpha0 = m.geom_rgba[robot_geoms, 3].copy()
if res['has_obj']:
    mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, '__qa_obj')])
from scipy.ndimage import gaussian_filter1d
look = gaussian_filter1d(Q[:, :2], 8, axis=0)
opt = C.vis_option(show_floor=True)
PW, PH = a.w // 2, a.h
ren = mujoco.Renderer(m, PH, PW, max_geom=12000)
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)
except Exception:
    FONT = ImageFont.load_default()


def set_frame(t):
    d.qpos[:] = Q[t]
    if res['has_obj']:
        q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, res['obj'].rot[t].reshape(-1))
        d.mocap_pos[mocap] = res['obj'].pos[t]; d.mocap_quat[mocap] = q4
    mujoco.mj_forward(m, d)


def panel(t, which, pw=PW, ph=PH, r=None):
    r = r or ren
    set_frame(t)
    cam = C.make_cam([look[t, 0], look[t, 1], 0.55], a.dist, a.az, a.el)
    m.geom_rgba[robot_geoms, 3] = 0.0 if which == 'human' else alpha0
    r.update_scene(d, cam, opt)
    if which == 'human':
        C.add_points(r.scene, H[t], human_colours(t), 0.008)
    else:
        idx, cols = robot_colours(t)
        if len(idx):
            C.add_points(r.scene, X[t, idx], cols, 0.016 if pw < 600 else 0.012)
    m.geom_rgba[robot_geoms, 3] = alpha0
    return Image.fromarray(r.render())


def frame_text(t):
    mk = cg['_masks']
    regs = [C.RID[k] for k in ('L_foot', 'R_foot', 'L_shin', 'R_shin', 'L_hand', 'R_hand', 'trunk')]
    s = []
    for rr in regs:
        if mk['hcon'][t, rr] or mk['solid'][t, rr]:
            s.append(f"{C.REGIONS[rr]}:H{int(mk['hcon'][t, rr])}R{int(mk['solid'][t, rr])}")
    if co is not None:
        mo = co['_masks']
        for rr in range(len(C.REGIONS)):
            if mo['hcon'][t, rr] or mo['solid'][t, rr]:
                s.append(f"obj {C.REGIONS[rr]}:H{int(mo['hcon'][t, rr])}R{int(mo['solid'][t, rr])}")
    return ' '.join(s)


def compose(t, pw=PW, ph=PH, r=None, small=False):
    L = panel(t, 'human', pw, ph, r); R_ = panel(t, 'robot', pw, ph, r)
    im = Image.new('RGB', (2 * pw, ph)); im.paste(L, (0, 0)); im.paste(R_, (pw, 0))
    dr = ImageDraw.Draw(im)
    dr.line([(pw, 0), (pw, ph)], fill=(255, 255, 255), width=2)
    f = FONT if not small else ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 12)
    src_f = int(res['npz']['qpos'].shape[0])
    dr.text((8, 6), f'human (UMR source slots)  t={t}/{T}', fill=(0, 0, 0), font=f)
    dr.text((pw + 8, 6), f'{C.ROBOT} post-pass  flag={int(res["npz"]["flags"][t])}', fill=(0, 0, 0), font=f)
    dr.text((8, ph - 22), frame_text(t)[:150], fill=(0, 0, 0), font=f)
    if not small:
        dr.text((pw + 8, 26), 'green touch [-0.5,1]mm  red pen>0.5mm  orange hover 1-20mm (human touching)  '
                'purple >20mm (human touching)', fill=(0, 0, 0), font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 11))
    return im


outd = C.QADIR / 'renders'; outd.mkdir(parents=True, exist_ok=True)
if a.grid:
    fr = np.linspace(0, T - 1, 12).round().astype(int)
    tw, th = 440, 380
    rg = mujoco.Renderer(m, th, tw, max_geom=12000)
    tiles = [compose(int(t), tw, th, rg, small=True) for t in fr]
    cols = 3
    sheet = Image.new('RGB', (cols * 2 * tw, 4 * th + 30), (255, 255, 255))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 2 * tw, (i // cols) * th))
    ImageDraw.Draw(sheet).text((8, 4 * th + 6), f'{a.tag}: 12 frames over full window ({T} frames @ {fps:.0f} fps). '
                               'robot: green touch, red pen, orange hover 1-20mm / purple >20mm while human touches; '
                               'human: green = contacting points', fill=(0, 0, 0), font=FONT)
    gp = outd / f'{a.tag}_grid12.png'; sheet.save(gp); print('saved', gp, sheet.size)
    rg.close()
if a.mp4:
    import imageio.v2 as imageio
    vp = outd / f'{a.tag}_human_vs_robot_{a.w}x{a.h}.mp4'
    w = imageio.get_writer(vp, fps=fps, codec='libx264', quality=7, macro_block_size=8)
    for t in range(T):
        w.append_data(np.asarray(compose(t)))
    w.close(); print('saved', vp)
ren.close()
print('seconds', round(time.time() - t0, 1))
