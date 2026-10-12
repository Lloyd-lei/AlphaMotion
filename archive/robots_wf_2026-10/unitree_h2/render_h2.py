"""unitree_h2 dedicated pipeline -- visual QA: [human points | robot] per frame (MuJoCo EGL).
Robot dense points: green = touching (clearance in [-0.5, 1] mm, ground or object), red = penetrating (< -0.5 mm),
orange = hovering 1-20 mm while the human's corresponding region is in contact, magenta = > 20 mm while the human is
in contact; grey = other.  Human slots: green = within 1 cm of floor/object in a human-contact region.
Object = translucent blue (collision pieces).  12-frame grid over the full window; optional mp4.
Usage: MUJOCO_GL=egl <umr python> render_h2.py <cid> [--mp4] [--size 1280x720]"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import mujoco, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2common as H2

ap = argparse.ArgumentParser(); ap.add_argument('cid'); ap.add_argument('--mp4', action='store_true')
ap.add_argument('--size', default='1280x720'); ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=-1)
a = ap.parse_args()
W, H = map(int, a.size.split('x'))
ROBOT = H2.ROBOT; PD = H2.PD; QD = H2.QD
o = np.load(PD / 'out' / a.cid / f'{ROBOT}.npz', allow_pickle=True)
prov = json.loads(str(o['provenance']))
pf = np.load(QD / f'{a.cid}.perframe.npz')
cfg = json.load(open(PD / 'pipeline.json'))
surf = dict(np.load(H2.RD / 'surface.npz', allow_pickle=True))
cp = dict(np.load(PD / 'contact_patches.npz', allow_pickle=True))
preg = cp['patch_region'].astype(int); REG = [str(r) for r in surf['regions']]
Q = o['qpos'].astype(np.float64); T = len(Q); fps = float(o['fps']); floor = float(o['floor_offset_removed'])
has_obj = 'object_pos' in o.files
# human slots + regions (same as measure)
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != T:
    fid = umr['frame_ids']; Hs = Hs[fid - (fid[0] if fid.max() >= len(Hs) else 0)]
Hs = Hs.astype(np.float64) - np.array([0, 0, floor])
log = str(Path(prov['umr_out']).with_suffix('.umr.log'))
hreg, _ = P.human_slot_regions(ROBOT, prov['source_kind'], prov['source_data'], prov['seq_key'], log,
                               shared=('beta0' in open(log).read()))
# render model: robot + checker floor + light (+ object pieces as mocap)
spec = mujoco.MjSpec.from_file(str(H2.RD / f'{ROBOT}.xml'))
tex = spec.add_texture(); tex.name = 'grid'; tex.type = mujoco.mjtTexture.mjTEXTURE_2D; tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
tex.rgb1 = [0.82, 0.82, 0.8]; tex.rgb2 = [0.72, 0.72, 0.7]; tex.width = 512; tex.height = 512
mat = spec.add_material(); mat.name = 'grid'; mat.texrepeat = [8, 8]
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
sky = spec.add_texture(); sky.name = 'sky'; sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT; sky.rgb1 = [1, 1, 1]; sky.rgb2 = [0.75, 0.82, 0.9]; sky.width = 256; sky.height = 256
fl = spec.worldbody.add_geom(); fl.type = mujoco.mjtGeom.mjGEOM_PLANE; fl.size = [8, 8, 0.05]; fl.material = 'grid'
fl.contype = 0; fl.conaffinity = 0; fl.group = 2
li = spec.worldbody.add_light(); li.pos = [0, 0, 6]; li.dir = [0, 0, -1]; li.diffuse = [0.6, 0.6, 0.6]
pieces = []
if has_obj:
    pieces, _ = H2.load_object_pieces(str(o['object_xml']), float(o['human_scale']))
    ob = spec.worldbody.add_body(); ob.name = '__obj'; ob.mocap = True
    for i, p in enumerate(pieces):
        me = spec.add_mesh(); me.name = f'__objm{i}'; me.uservert = p['v'].astype(np.float32).reshape(-1)
        g = ob.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = me.name; g.contype = 0; g.conaffinity = 0
        g.rgba = [0.25, 0.45, 0.95, 0.45]; g.group = 2
    OPOS = o['object_pos'].astype(np.float64); OROT = o['object_rot'].astype(np.float64)
m = spec.compile(); d = mujoco.MjData(m)
m.vis.global_.offwidth = 1920; m.vis.global_.offheight = 1080
m.vis.headlight.ambient = [0.45, 0.45, 0.45]; m.vis.headlight.diffuse = [0.5, 0.5, 0.5]
m0 = mujoco.MjModel.from_xml_path(str(H2.RD / f'{ROBOT}.xml'))
GM = H2.geom_map(m0, m)
robot_vis = [GM[g] for g in range(m0.ngeom) if m0.geom_group[g] == 1 and m0.geom_bodyid[g] > 0]
PW = W // 2
rend = mujoco.Renderer(m, H, PW, max_geom=m.ngeom + 4096 * 2 + 50)
pl = surf['points_local'].astype(np.float64); pg = np.array([GM[int(g)] for g in surf['geom']])
GREY, GREEN, RED, ORANGE, MAG, HGREY = ([.55, .55, .58, 1], [0.1, .85, .2, 1], [.95, .1, .1, 1], [1, .55, 0, 1],
                                        [.85, .1, .85, 1], [.45, .5, .65, 1])
cg = pf['cg_patch']; H_g = pf['H_g']
co = pf['co_patch'] if has_obj else None; H_o = pf['H_o'] if has_obj else None
# smoothed camera targets
def smooth(x, k=15):
    ker = np.ones(k) / k
    return np.stack([np.convolve(np.pad(x[:, i], (k // 2, k // 2), mode='edge'), ker, mode='valid') for i in range(3)], 1)


rootxy = smooth(Q[:, :3].copy()); hum_c = smooth(Hs.mean(1))
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except Exception:
    font = ImageFont.load_default()


def robot_points(t):
    d.qpos[:] = Q[t]
    if has_obj:
        d.mocap_pos[0] = OPOS[t]; qq = np.zeros(4); mujoco.mju_mat2Quat(qq, OROT[t].reshape(-1)); d.mocap_quat[0] = qq
    mujoco.mj_forward(m, d)
    return np.einsum('nij,nj->ni', d.geom_xmat[pg].reshape(-1, 3, 3), pl) + d.geom_xpos[pg]


def colors_robot(t, X):
    c = np.tile(GREY, (len(X), 1))
    gz = X[:, 2]
    od_ = H2.obj_sdf(pieces, OPOS[t], OROT[t], X)[0] if has_obj else np.full(len(X), np.inf)
    dmin = np.minimum(gz, od_)
    touch = (dmin >= -0.0005) & (dmin <= 0.001) & (preg >= 0)
    c[touch] = GREEN
    stats = dict(hover=[], far=[])
    for kind, cl, Hc, dist in (('g', cg, H_g, gz), ('o', co, H_o, od_)):
        if cl is None:
            continue
        for r in range(len(REG)):
            if not Hc[t, r]:
                continue
            idx = np.flatnonzero(preg == r)
            if not len(idx):
                continue
            v = cl[t, r]
            if 0.001 < v <= 0.02:
                low = idx[dist[idx] <= v + 0.005]; c[low] = ORANGE; stats['hover'].append(f'{kind}:{REG[r]} {v*1000:.0f}mm')
            elif v > 0.02:
                low = idx[dist[idx] <= v + 0.005]; c[low] = MAG; stats['far'].append(f'{kind}:{REG[r]} {v*100:.1f}cm')
    c[dmin < -0.0005] = RED
    return c, stats


def colors_human(t):
    c = np.tile(HGREY, (len(Hs[t]), 1))
    z = Hs[t][:, 2]
    on = H_g[t][hreg] & (z <= 0.01)
    if has_obj:
        od_ = H2.obj_sdf(pieces, OPOS[t], OROT[t], Hs[t], cull=0.05)[0]
        on |= H_o[t][hreg] & (od_ <= 0.01)
    c[on] = GREEN
    return c


def panel(t, X, cols, center, show_robot):
    cam = mujoco.MjvCamera(); cam.lookat[:] = [center[0], center[1], 0.82]; cam.distance = 2.9; cam.azimuth = 135; cam.elevation = -10
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [0, 1 if show_robot else 0, 1, 0, 0, 0]   # robot visual = group 1
    rgba0 = m.geom_rgba.copy()
    m.geom_rgba[robot_vis, 3] = 0.35
    rend.update_scene(d, cam, opt)
    sc = rend.scene
    order = np.argsort([0 if tuple(cc) in (tuple(GREY), tuple(HGREY)) else 1 for cc in cols])  # coloured last
    for i in order:
        if sc.ngeom >= sc.maxgeom:
            break
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.009 if tuple(cols[i]) not in (tuple(GREY), tuple(HGREY)) else 0.0042, 0, 0]),
                            X[i], np.eye(3).reshape(-1), np.asarray(cols[i], np.float32))
        sc.ngeom += 1
    img = rend.render().copy(); m.geom_rgba[:] = rgba0
    return img


def frame(t):
    X = robot_points(t)
    cr, stt = colors_robot(t, X)
    ch = colors_human(t)
    left = panel(t, Hs[t], ch, hum_c[t], False)
    right = panel(t, X, cr, rootxy[t], True)
    img = Image.fromarray(np.concatenate([left, right], 1)); dr = ImageDraw.Draw(img)
    hl = [REG[r] for r in range(len(REG)) if H_g[t, r]] + ([f'obj:{REG[r]}' for r in range(len(REG)) if H_o[t, r]] if has_obj else [])
    LAST['stt'] = stt; LAST['hl'] = ' '.join(hl)
    dr.text((8, 6), f'HUMAN (SMPL-X slots)  t={t}/{T}  contact: {" ".join(hl)[:70]}', fill=(0, 0, 0), font=font)
    dr.text((PW + 8, 6), f'unitree_h2 (CPU post-pass output)  flag={int(o["flags"][t])}', fill=(0, 0, 0), font=font)
    dr.text((PW + 8, 26), ('hover: ' + ', '.join(stt['hover']))[:80], fill=(220, 110, 0), font=font)
    dr.text((PW + 8, 46), ('far: ' + ', '.join(stt['far']))[:80], fill=(170, 0, 170), font=font)
    dr.text((PW + 8, H - 22), 'green=touch [-0.5,1]mm  red=pen<-0.5mm  orange=hover 1-20mm  magenta=>20mm (human in contact)',
            fill=(0, 0, 0), font=font)
    dr.line([(PW, 0), (PW, H)], fill=(0, 0, 0), width=2)
    return img


tag = a.cid
LAST = {}
idx = np.linspace(0, T - 1, 12).astype(int)
try:
    gfont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
except Exception:
    gfont = font
LAST = {}
def gcell(t):
    im = frame(t).resize((W // 2, H // 2)); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W // 2, 22], fill=(255, 255, 255))
    st = LAST['stt']; hl = LAST['hl']
    dr.text((4, 2), f't={t}  human:{hl[:28]}  hover:{len(st["hover"])} far:{len(st["far"])} flag:{int(o["flags"][t])}', fill=(0, 0, 0), font=gfont)
    return im
cells = [gcell(t) for t in idx]
sheet = Image.new('RGB', (4 * (W // 2), 3 * (H // 2)), (255, 255, 255))
for i, c in enumerate(cells):
    sheet.paste(c, ((i % 4) * (W // 2), (i // 4) * (H // 2)))
gp = QD / f'{tag}_grid12.png'; sheet.save(gp); print('saved', gp)
if a.mp4:
    import imageio.v2 as imageio
    e = T if a.end < 0 else min(a.end, T)
    vp = QD / f'{tag}.mp4'
    w = imageio.get_writer(str(vp), fps=fps, codec='libx264', quality=7, macro_block_size=8)
    for t in range(a.start, e):
        w.append_data(np.asarray(frame(t)))
    w.close(); print('saved', vp)
