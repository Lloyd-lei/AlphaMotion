"""unitree_h1 dedicated pipeline -- step 4: comparison video [human points | robot] + 12-frame grid (whole window).
Robot panel: MuJoCo mesh + dense points coloured  green = touching (clearance in [-0.5, 1] mm, ground or object),
red = penetrating (< -0.5 mm), orange = region hovering 1-20 mm while the human region is in contact (lowest points of
that region).  Human panel: UMR source slots (grey), green = human contact slots (region in contact, slot < 1 cm).
Uses the independent measurement (qa/unitree_h1/measure_<clip>.npz) for human labels.
Usage: MUJOCO_GL=egl <umr py> render_compare.py <clip_id> [--mp4]"""
import json, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, '/home/arenalabs/umr_factory_code')
import measure_independent as MI

ROBOT = MI.ROBOT; PD = MI.PD; QA = MI.QA; REG = MI.REG
cid = sys.argv[1]; want_mp4 = '--mp4' in sys.argv
o = np.load(PD / 'out' / cid / f'{ROBOT}.npz', allow_pickle=True)
M = np.load(QA / f'measure_{cid}.npz')
prov = json.loads(str(o['provenance']))
Q = o['qpos'].astype(float); T = len(Q); fps = float(o['fps']); floor = float(o['floor_offset_removed'])
has_obj = 'object_pos' in o.files
obj = MI.TrueObject(str(o['object_xml']), float(o['human_scale']), o['object_rot'], o['object_pos']) if has_obj else None
m = MI.build_model(obj); d = mujoco.MjData(m)
for g in range(m.ngeom):
    if m.geom_bodyid[g] != 0 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g).startswith('__objg'):
        m.geom_group[g] = 1; m.geom_rgba[g] = [0.55, 0.45, 0.3, 0.55]
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != T:
    Hs = Hs[umr['frame_ids']]
Hs = Hs.astype(float) - [0, 0, floor]
from umrf import pipeline as P
hreg, _ = P.human_slot_regions(ROBOT, prov['source_kind'], prov['source_data'], prov['seq_key'],
                               str(Path(prov['umr_out']).with_suffix('.umr.log')), prov['source_kind'] == 'hoi')
h_g, h_o, g_reg, o_reg = M['h_g'], M['h_o'], M['g_reg'], M['o_reg']
preg, pgeom, ploc = MI.preg, MI.pgeom, MI.ploc
W, H = 640, 720
rend = mujoco.Renderer(m, H, W, max_geom=9000)
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
except Exception:
    FONT = ImageFont.load_default()
GREEN, RED, ORANGE, GREY, HGREY = (0.05, 0.85, 0.1, 1), (0.95, 0.05, 0.05, 1), (1.0, 0.55, 0.0, 1), (0.6, 0.6, 0.6, 1), (0.45, 0.55, 0.75, 1)
root_xy = Q[:, :2]
# smooth camera target
k = 15; ker = np.ones(k) / k
cam_xy = np.stack([np.convolve(np.pad(root_xy[:, i], k // 2, mode='edge'), ker, 'valid') for i in range(2)], 1)[:T]
hum_xy = Hs.mean(1)[:, :2]
hcam_xy = np.stack([np.convolve(np.pad(hum_xy[:, i], k // 2, mode='edge'), ker, 'valid') for i in range(2)], 1)[:T]


def add_sph(sc, p, r, rgba):
    if sc.ngeom >= sc.maxgeom:
        return
    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float),
                        np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
    sc.ngeom += 1


def frame(t, az=135.0):
    d.qpos[:] = Q[t]
    if obj is not None:
        d.mocap_pos[0] = obj.pos[t]; q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, obj.rot[t].reshape(-1)); d.mocap_quat[0] = q4
    mujoco.mj_forward(m, d)
    X = np.einsum('nij,nj->ni', d.geom_xmat[pgeom].reshape(-1, 3, 3), ploc) + d.geom_xpos[pgeom]
    clr = X[:, 2].copy()
    if obj is not None:
        clr = np.minimum(clr, obj.sdf(X, t))
    cam = mujoco.MjvCamera(); cam.distance = 3.3; cam.azimuth = az; cam.elevation = -14
    # ---- robot panel
    cam.lookat[:] = [cam_xy[t, 0], cam_xy[t, 1], 0.8]
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    rend.update_scene(d, camera=cam, scene_option=opt); sc = rend.scene
    n_c = n_p = n_h = 0
    for i in np.flatnonzero(clr < -MI.PEN):
        add_sph(sc, X[i], 0.010, RED); n_p += 1
    for i in np.flatnonzero((clr >= -MI.PEN) & (clr <= MI.C_HI)):
        add_sph(sc, X[i], 0.010, GREEN); n_c += 1
    hov_regs = []
    for r in range(len(REG)):
        for hc, rc in ((h_g, g_reg), (h_o, o_reg)):
            if hc[t, r] and MI.C_HI < rc[t, r] <= 0.02:
                idx = np.flatnonzero(preg == r)
                low = idx[clr[idx] <= clr[idx].min() + 0.003]
                for i in low:
                    add_sph(sc, X[i], 0.013, ORANGE); n_h += 1
                hov_regs.append(f'{REG[r]} {rc[t, r] * 1000:.0f}mm')
    rimg = Image.fromarray(rend.render())
    # ---- human panel
    cam.lookat[:] = [hcam_xy[t, 0], hcam_xy[t, 1], 0.8]
    opt2 = mujoco.MjvOption(); opt2.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    rend.update_scene(d, camera=cam, scene_option=opt2); sc = rend.scene
    hp = Hs[t]
    hcl = hp[:, 2].copy()
    if obj is not None:
        hcl = np.minimum(hcl, obj.sdf(hp, t))
    hcon = (h_g[t][hreg] | h_o[t][hreg]) & (hcl < 0.01)
    for i in range(len(hp)):
        add_sph(sc, hp[i], 0.010 if hcon[i] else 0.006, GREEN if hcon[i] else HGREY)
    himg = Image.fromarray(rend.render())
    img = Image.new('RGB', (2 * W, H)); img.paste(himg, (0, 0)); img.paste(rimg, (W, 0))
    dr = ImageDraw.Draw(img)
    hl = [REG[r] for r in range(len(REG)) if h_g[t, r]] + [REG[r] + '(obj)' for r in range(len(REG)) if h_o[t, r]]
    dr.text((10, 8), f'{cid}  t={t}/{T - 1}', fill=(0, 0, 0), font=FONT)
    dr.text((10, 28), 'HUMAN (UMR source slots)  contact: ' + (', '.join(hl) or '-'), fill=(0, 0, 0), font=FONT)
    dr.text((W + 10, 28), f'ROBOT unitree_h1  green={n_c} red={n_p} orange={n_h}  flag={int(o["flags"][t])}', fill=(0, 0, 0), font=FONT)
    if hov_regs:
        dr.text((W + 10, 48), 'hover while human contact: ' + ', '.join(hov_regs), fill=(200, 90, 0), font=FONT)
    dr.line([(W, 0), (W, H)], fill=(0, 0, 0), width=2)
    return img


fr = np.linspace(0, T - 1, 12).astype(int)
cells = [frame(t).resize((640, 360)) for t in fr]
sheet = Image.new('RGB', (4 * 640, 3 * 360), (255, 255, 255))
for i, c in enumerate(cells):
    sheet.paste(c, ((i % 4) * 640, (i // 4) * 360))
gp = QA / f'grid_{cid}.png'; sheet.save(gp); print('saved', gp)
if want_mp4:
    import imageio.v2 as imageio
    vp = QA / f'compare_{cid}.mp4'
    w = imageio.get_writer(vp, fps=fps, codec='libx264', quality=7, macro_block_size=8)
    for t in range(T):
        w.append_data(np.asarray(frame(t)))
    w.close(); print('saved', vp)
rend.close()
