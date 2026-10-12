"""unitree_g1_29dof -- QA renders: [human points | robot] side by side, MuJoCo EGL.
Robot panel: meshes + dense points coloured by the independent measurement (measure.py status):
  green = contacting (ground or object, clearance in [-0.5, 1] mm), red = penetration > 0.5 mm,
  orange = hovering 1-20 mm while the human region is in contact.  Object (if any) translucent blue.
Human panel: UMR's 4096 human source slots (retarget frame, floor-shifted); green = human region in contact and slot
within 2 cm of the floor / object.
Usage: <umr env python> render_qa.py <engine npz> <clip> [--mp4] [--grid]
"""
import argparse, json, os, sys, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROBOT = 'unitree_g1_29dof'
DATA = Path('/home/arenalabs/umr_factory_data')
PD = DATA / 'robots' / ROBOT / 'pipeline'
QA = DATA / 'qa' / ROBOT
T_ = mujoco.mjtGeom
ap = argparse.ArgumentParser(); ap.add_argument('npz'); ap.add_argument('clip')
ap.add_argument('--mp4', action='store_true'); ap.add_argument('--grid', action='store_true')
ap.add_argument('--w', type=int, default=1280); ap.add_argument('--h', type=int, default=720)
a = ap.parse_args()
o = np.load(a.npz, allow_pickle=True)
prov = json.loads(str(o['provenance']))
st = np.load(PD / 'out' / f'{a.clip}_status.npz')
S = np.load(DATA / 'robots' / ROBOT / 'surface.npz')
pl, pg = S['points_local'].astype(float), S['geom'].astype(int)
Q = o['qpos'].astype(float); T = len(Q)
has_obj = 'object_pos' in o.files
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != T:
    Hs = Hs[umr['frame_ids']]
Hs = Hs.astype(float) - np.array([0, 0, float(o['floor_offset_removed'])])
meas = json.load(open(QA / f'measure_{a.clip}.json'))

spec = mujoco.MjSpec.from_file(str(o['robot_xml']))
if has_obj:
    sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
    import retarget_smpl_to_humanoid_surface_vector as rs
    oxml = str(o['object_xml']); root = ET.parse(oxml).getroot()
    rs.make_mjcf_mesh_paths_absolute(root, Path(oxml)); rs.scale_mjcf_mesh_assets(root, float(o['human_scale']))
    with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False, dir=str(PD / 'out')) as f:
        f.write(ET.tostring(root, encoding='unicode')); tmp = f.name
    om = mujoco.MjModel.from_xml_path(tmp); Path(tmp).unlink()
    od = mujoco.MjData(om); mujoco.mj_forward(om, od)
    pieces = []
    for g in range(om.ngeom):
        if om.geom_type[g] == T_.mjGEOM_MESH and (om.geom_contype[g] or om.geom_conaffinity[g]):
            mi = om.geom_dataid[g]
            v = om.mesh_vert[om.mesh_vertadr[mi]:om.mesh_vertadr[mi] + om.mesh_vertnum[mi]].astype(float)
            pieces.append(v @ od.geom_xmat[g].reshape(3, 3).T + od.geom_xpos[g])
    b = spec.worldbody.add_body(); b.name = 'qa_object'; b.mocap = True
    for i, v in enumerate(pieces):
        ms = spec.add_mesh(); ms.name = f'qa_obj_m{i}'; ms.uservert = v.astype(np.float32).ravel()
        g = b.add_geom(); g.type = T_.mjGEOM_MESH; g.meshname = ms.name; g.contype = 0; g.conaffinity = 0
        g.rgba = [0.25, 0.45, 0.95, 0.55]; g.group = 1
m = spec.compile(); d = mujoco.MjData(m)
obody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'qa_object')
robot_geom = np.array([m.geom_bodyid[g] not in (0, obody) for g in range(m.ngeom)])
base_rgba = m.geom_rgba.copy()
base_rgba[robot_geom & (m.geom_group == 1), :] = [0.78, 0.78, 0.80, 1.0]
PW, PH = a.w // 2, a.h
rr = mujoco.Renderer(m, PH, PW, max_geom=4096 + 600)
opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = 1; opt.geomgroup[0] = 1
COL = {1: (0.05, 0.85, 0.15), 2: (0.05, 0.85, 0.15), 3: (0.95, 0.05, 0.05), 4: (1.0, 0.55, 0.0)}
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
except Exception:
    FONT = None


def set_frame(t):
    d.qpos[:] = Q[t]
    if has_obj:
        q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, o['object_rot'][t].astype(float).ravel())
        d.mocap_pos[int(m.body_mocapid[obody])] = o['object_pos'][t]; d.mocap_quat[int(m.body_mocapid[obody])] = q4
    mujoco.mj_forward(m, d)


def add_sphere(sc, p, rad, rgb, alpha=1.0):
    if sc.ngeom >= sc.maxgeom:
        return
    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], T_.mjGEOM_SPHERE, np.array([rad, 0, 0]), p, np.eye(3).ravel(),
                        np.array([*rgb, alpha], np.float32))
    sc.ngeom += 1


def cam_for(center, az):
    c = mujoco.MjvCamera(); c.lookat[:] = [center[0], center[1], 0.55]; c.distance = 2.7; c.azimuth = az
    c.elevation = -14; return c


def panel_robot(t, az):
    set_frame(t)
    m.geom_rgba[:] = base_rgba
    rr.update_scene(d, cam_for(d.qpos[:2], az), opt); sc = rr.scene
    s = st['robot'][t]
    k = np.flatnonzero(s > 0)
    if len(k):
        Rm = d.geom_xmat[pg[k]].reshape(-1, 3, 3)
        X = np.einsum('nij,nj->ni', Rm, pl[k]) + d.geom_xpos[pg[k]]
        for p, c in zip(X, s[k]):
            add_sphere(sc, p, 0.011, COL[int(c)])
    return Image.fromarray(rr.render())


def panel_human(t, az):
    set_frame(t)
    m.geom_rgba[:] = base_rgba; m.geom_rgba[robot_geom, 3] = 0.0
    hc = Hs[t].mean(0)
    rr.update_scene(d, cam_for(hc, az), opt); sc = rr.scene
    hs = st['human'][t]
    for p, c in zip(Hs[t], hs):
        add_sphere(sc, p, 0.0075 if c else 0.006, (0.05, 0.85, 0.15) if c else (0.55, 0.6, 0.72))
    return Image.fromarray(rr.render())


fl = o['flags']; bad = st['bad']


def frame_img(t, az=135.0, scale=1.0):
    L = panel_human(t, az); Rimg = panel_robot(t, az)
    img = Image.new('RGB', (2 * PW, PH), (255, 255, 255)); img.paste(L, (0, 0)); img.paste(Rimg, (PW, 0))
    dr = ImageDraw.Draw(img)
    dr.line([(PW, 0), (PW, PH)], fill=(40, 40, 40), width=2)
    s = st['robot'][t]
    txt = (f'{a.clip}  t={t}/{T - 1}  engine_flag={int(fl[t])}  v2_fail={int(bad[t])}  '
           f'contact={int((s == 1).sum() + (s == 2).sum())} pen={int((s == 3).sum())} hover={int((s == 4).sum())}')
    dr.text((10, 8), 'HUMAN (UMR source slots, green = in contact)', fill=(0, 0, 0), font=FONT)
    dr.text((PW + 10, 8), 'G1 29dof  green=contact  red=pen>0.5mm  orange=hover 1-20mm (human in contact)',
            fill=(0, 0, 0), font=FONT)
    dr.text((10, PH - 28), txt, fill=(0, 0, 0), font=FONT)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


QA.mkdir(parents=True, exist_ok=True)
if a.grid:
    fr = np.linspace(0, T - 1, 12).astype(int)
    tiles = [frame_img(int(t), scale=0.5) for t in fr]
    tw, th = tiles[0].size
    sheet = Image.new('RGB', (3 * tw, 4 * th), (255, 255, 255))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 3) * tw, (i // 3) * th))
    p = QA / f'{a.clip}_grid12.png'; sheet.save(p); print('saved', p)
if a.mp4:
    import imageio.v2 as imageio
    p = QA / f'{a.clip}_human_vs_g1.mp4'
    w = imageio.get_writer(p, fps=float(o['fps']), codec='libx264', quality=7, macro_block_size=8)
    for t in range(T):
        w.append_data(np.asarray(frame_img(t)))
    w.close(); print('saved', p)
rr.close()
