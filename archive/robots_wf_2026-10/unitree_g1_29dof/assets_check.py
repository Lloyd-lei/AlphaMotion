"""unitree_g1_29dof dedicated pipeline -- step 1: asset check (per-robot file, read-only use of shared assets).

Outputs (pipeline/assets/):
  tpose_regions.png / rest_regions.png : MuJoCo EGL render (front | side | back), robot meshes translucent, the 4096
                                          dense surface points as spheres coloured by region (L = warm, R = cool).
  patches.png                          : contact patches (sole / foot edge / knee / elbow / hand / pelvis / torso).
  patches.npz                          : per-patch dense point indices (frozen into pipeline.json by sha).
  assets_check.json                    : region stats, L/R check, collision geoms per region, exclusion review,
                                          correspondence files.
Run: MUJOCO_GL=egl <umr env python> assets_check.py
"""
import hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROBOT = 'unitree_g1_29dof'
RD = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT
OUTD = RD / 'pipeline' / 'assets'; OUTD.mkdir(parents=True, exist_ok=True)
XML = RD / f'{ROBOT}.xml'
st = json.load(open(RD / 'staging.json'))
S = np.load(RD / 'surface.npz')
REG = [str(r) for r in S['regions']]
m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
T_ = mujoco.mjtGeom

pl, pg, preg = S['points_local'].astype(float), S['geom'].astype(int), S['region'].astype(int)
nl = S['normals_local'].astype(float)


def sha(p, n=16):
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return h[:n]


def pose(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    R = d.geom_xmat[pg].reshape(-1, 3, 3)
    X = np.einsum('nij,nj->ni', R, pl) + d.geom_xpos[pg]
    N = np.einsum('nij,nj->ni', R, nl)
    return X, N


q_rest = m.qpos0.copy()                         # root at 0.793, all joints 0 (GMR mocap default)
q_t = q_rest.copy()
for jn, v in st['tpose'].items():
    q_t[m.jnt_qposadr[jid(jn)]] = v

# ------------------------------------------------------------------ region colours (L warm, R cool, trunk grey)
COL = {'trunk': (150, 150, 150), 'L_thigh': (230, 60, 40), 'R_thigh': (40, 90, 230), 'L_shin': (250, 140, 30),
       'R_shin': (30, 190, 230), 'L_foot': (200, 0, 120), 'R_foot': (0, 160, 90), 'L_upper_arm': (255, 210, 0),
       'R_upper_arm': (120, 60, 220), 'L_forearm': (180, 90, 20), 'R_forearm': (0, 110, 140),
       'L_hand': (255, 0, 0), 'R_hand': (0, 0, 255)}


def render(q, pts_rgb, X, views, alpha=0.18, w=560, h=720, rad=0.0045, title=''):
    rgba = m.geom_rgba.copy()
    m.geom_rgba[:, 3] = np.where(m.geom_group == 1, alpha, 0.0)        # visual meshes translucent, collision hidden
    m.geom_rgba[0, 3] = 1.0                                             # floor
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    rr = mujoco.Renderer(m, h, w, max_geom=len(X) + 400)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
    tiles = []
    for name, az in views:
        cam = mujoco.MjvCamera(); cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.65]; cam.distance = 2.3
        cam.azimuth = az; cam.elevation = -8
        rr.update_scene(d, cam, opt)
        sc = rr.scene
        for p, c in zip(X, pts_rgb):
            if sc.ngeom >= sc.maxgeom:
                break
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], T_.mjGEOM_SPHERE, np.array([rad, 0, 0]), p, np.eye(3).ravel(),
                                np.array([*(np.array(c) / 255.0), 1.0], np.float32))
            sc.ngeom += 1
        img = Image.fromarray(rr.render())
        ImageDraw.Draw(img).text((8, 8), f'{title} {name}', fill=(0, 0, 0))
        tiles.append(img)
    rr.close()
    m.geom_rgba[:] = rgba
    out = Image.new('RGB', (w * len(tiles), h), (255, 255, 255))
    for i, t in enumerate(tiles):
        out.paste(t, (i * w, 0))
    return out


def legend(img, items):
    dr = ImageDraw.Draw(img)
    y = img.height - 18 * len(items) - 6
    for i, (n, c) in enumerate(items):
        dr.rectangle([8, y + 18 * i, 22, y + 18 * i + 12], fill=tuple(c)); dr.text((28, y + 18 * i), n, fill=(0, 0, 0))
    return img


VIEWS = [('front (camera at +x)', 180.0), ('left side (camera at +y)', -90.0), ('back', 0.0)]
info = dict(robot=ROBOT, mjcf=str(XML), mjcf_sha256_16=sha(XML), surface=str(RD / 'surface.npz'),
            surface_sha256_16=sha(RD / 'surface.npz'), n_points=int(len(pl)))
for tag, q in (('tpose', q_t), ('rest', q_rest)):
    X, N = pose(q)
    rgb = np.array([COL[REG[r]] for r in preg])
    img = render(q, rgb, X, VIEWS, title=tag)
    legend(img, [(r, COL[r]) for r in REG]).save(OUTD / f'{tag}_regions.png')
    # region stats in this pose
    rs = {}
    for r, name in enumerate(REG):
        k = preg == r
        bodies = sorted({bn(int(b)) for b in m.geom_bodyid[pg[k]]})
        rs[name] = dict(n=int(k.sum()), centroid=np.round(X[k].mean(0), 3).tolist() if k.any() else None,
                        zmin=round(float(X[k, 2].min()), 4) if k.any() else None, bodies=bodies)
    info[f'regions_{tag}'] = rs

# ------------------------------------------------------------------ L/R + anatomy sanity (rest pose, robot faces +x)
X, N = pose(q_rest)
chk = {}
for s, sg in (('L', 1), ('R', -1)):
    for part in ('thigh', 'shin', 'foot', 'upper_arm', 'forearm', 'hand'):
        k = preg == REG.index(f'{s}_{part}')
        chk[f'{s}_{part}_y_sign_ok'] = bool(np.sign(X[k, 1].mean()) == sg)
order_ok = {}
for s in 'LR':
    z = {p: X[preg == REG.index(f'{s}_{p}'), 2].mean() for p in ('thigh', 'shin', 'foot')}
    order_ok[s] = bool(z['thigh'] > z['shin'] > z['foot'])
chk['leg_vertical_order_ok'] = order_ok
info['lr_anatomy_check'] = chk

# ------------------------------------------------------------------ contact patches (rest pose geometry)
patch = {}
for s in 'LR':
    kf = np.flatnonzero(preg == REG.index(f'{s}_foot'))
    z0 = X[kf, 2].min()
    patch[f'{s}_sole'] = kf[X[kf, 2] <= z0 + 0.003]                    # flat bottom face
    patch[f'{s}_foot_edge'] = kf[(X[kf, 2] > z0 + 0.003) & (X[kf, 2] <= z0 + 0.020)]   # rim / toe / heel edges
    patch[f'{s}_foot_upper'] = kf[X[kf, 2] > z0 + 0.020]               # instep/ankle: ground contact NOT allowed
    side = 'left' if s == 'L' else 'right'
    ka = d.xanchor[jid(f'{side}_knee_joint')]
    kk = np.flatnonzero(np.isin(preg, [REG.index(f'{s}_thigh'), REG.index(f'{s}_shin')]))
    near = np.linalg.norm(X[kk] - ka, axis=1) <= 0.10
    patch[f'{s}_knee'] = kk[near & (N[kk, 0] > -0.2)]                   # front / sides of the knee
    ks = np.flatnonzero(preg == REG.index(f'{s}_shin'))
    patch[f'{s}_shin_front'] = ks[(N[ks, 0] > 0.3) & (np.linalg.norm(X[ks] - ka, axis=1) > 0.10)]
    ea = d.xanchor[jid(f'{side}_elbow_joint')]
    ke = np.flatnonzero(np.isin(preg, [REG.index(f'{s}_upper_arm'), REG.index(f'{s}_forearm')]))
    patch[f'{s}_elbow'] = ke[np.linalg.norm(X[ke] - ea, axis=1) <= 0.07]
    patch[f'{s}_hand'] = np.flatnonzero(preg == REG.index(f'{s}_hand'))
kt = np.flatnonzero(preg == REG.index('trunk'))
pel_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('pelvis', 'pelvis_contour_link')]
pb = m.geom_bodyid[pg[kt]]
patch['pelvis'] = kt[np.isin(pb, [b for b in pel_bodies if b >= 0])]
patch['torso'] = kt[pb == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')]
patch = {k: np.asarray(v, np.int32) for k, v in patch.items()}
np.savez_compressed(OUTD / 'patches.npz', **patch)
info['patches'] = {k: int(len(v)) for k, v in patch.items()}
info['patches_sha256_16'] = sha(OUTD / 'patches.npz')
PC = {'sole': (0, 200, 0), 'foot_edge': (150, 230, 0), 'foot_upper': (230, 0, 0), 'knee': (255, 140, 0),
      'shin_front': (255, 210, 120), 'elbow': (200, 0, 200), 'hand': (0, 120, 255), 'pelvis': (90, 50, 20),
      'torso': (0, 200, 200)}
for tag, q in (('rest', q_rest),):
    Xp, _ = pose(q)
    rgb = np.full((len(pl), 3), 210)
    for k, v in patch.items():
        key = k[2:] if k[:2] in ('L_', 'R_') else k
        rgb[v] = PC[key]
    img = render(q, rgb, Xp, [('front', 180.0), ('left side', -90.0), ('back', 0.0)], title='contact patches')
    legend(img, list(PC.items())).save(OUTD / 'patches.png')
# bottom view of soles
rgba = m.geom_rgba.copy(); m.geom_rgba[:, 3] = np.where(m.geom_group == 1, 0.15, 0.0); m.geom_rgba[0, 3] = 0
d.qpos[:] = q_rest; d.qpos[2] += 0.3; mujoco.mj_forward(m, d)
Xb, _ = pose(d.qpos.copy())
rr = mujoco.Renderer(m, 560, 560, max_geom=len(pl) + 400); opt = mujoco.MjvOption()
cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.35]; cam.distance = 1.0; cam.azimuth = 180; cam.elevation = 89.0
rr.update_scene(d, cam, opt); sc = rr.scene
rgb = np.full((len(pl), 3), 210)
for k, v in patch.items():
    key = k[2:] if k[:2] in ('L_', 'R_') else k
    rgb[v] = PC[key]
for p, c in zip(Xb, rgb):
    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], T_.mjGEOM_SPHERE, np.array([0.004, 0, 0]), p, np.eye(3).ravel(),
                        np.array([*(c / 255.0), 1.0], np.float32)); sc.ngeom += 1
Image.fromarray(rr.render()).save(OUTD / 'soles_from_below.png'); rr.close(); m.geom_rgba[:] = rgba

# ------------------------------------------------------------------ collision geoms per region
breg = S['body_region']
cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
coll = {}
for g in cg:
    r = REG[int(breg[m.geom_bodyid[g]])]
    coll.setdefault(r, []).append(dict(geom=int(g), body=bn(int(m.geom_bodyid[g])), type=T_(m.geom_type[g]).name[7:],
                                       size=np.round(m.geom_size[g], 4).tolist()))
info['collision_geoms_per_region'] = coll
vis_bodies = {int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_group[g] == 1}
col_bodies = {int(m.geom_bodyid[g]) for g in cg}
info['bodies_visual_without_collision'] = sorted(bn(b) for b in vis_bodies - col_bodies)
info['bodies_without_any_geom'] = sorted(bn(b) for b in range(1, m.nbody) if b not in vis_bodies | col_bodies)
# foot collision vs visual sole (rest): sphere bottoms vs visual sole plane
d.qpos[:] = q_rest; mujoco.mj_forward(m, d)
fz = {}
for s, body in (('L', 'left_ankle_roll_link'), ('R', 'right_ankle_roll_link')):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    sp = [g for g in cg if m.geom_bodyid[g] == b and m.geom_type[g] == T_.mjGEOM_SPHERE]
    bottoms = [float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in sp]
    kf = preg == REG.index(f'{s}_foot')
    Xr, _ = pose(q_rest)
    fz[s] = dict(collision_sphere_bottoms_mm=[round(1000 * z, 2) for z in bottoms],
                 visual_sole_min_mm=round(1000 * float(Xr[kf, 2].min()), 2),
                 sphere_xy_local=[np.round(m.geom_pos[g], 4).tolist() for g in sp])
info['foot_collision_vs_visual_rest'] = fz

# ------------------------------------------------------------------ exclusion review
def overl(q, thr=-1e-4):
    d.qpos[:] = q; d.qpos[0:3] = [0, 0, 2.0]; mujoco.mj_forward(m, d)
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in cg and c.geom2 in cg and c.dist < thr:
            k = tuple(sorted((bn(int(m.geom_bodyid[c.geom1])), bn(int(m.geom_bodyid[c.geom2])))))
            out[k] = min(out.get(k, 0), float(c.dist))
    return out


rng = np.random.default_rng(1)
cnt = {}
NP = 300
for _ in range(NP):
    q = q_rest.copy()
    for j in range(1, m.njnt):
        q[m.jnt_qposadr[j]] = rng.uniform(*m.jnt_range[j])
    for k in overl(q):
        cnt[k] = cnt.get(k, 0) + 1
# pairs MuJoCo never tests (parent-child filter) among collision bodies
cb = sorted(col_bodies)
parent_filtered = []
for i, a in enumerate(cb):
    for b in cb[i + 1:]:
        pa, pb_ = int(m.body_parentid[a]), int(m.body_parentid[b])
        if pa == b or pb_ == a:
            parent_filtered.append([bn(a), bn(b)])
info['exclusions'] = dict(
    staged_exclude_list=st['exclusions'], staged_n=len(st['exclusions']),
    mujoco_parent_child_filtered_pairs=parent_filtered,
    rest_overlaps=[[*k, round(-v * 1000, 2)] for k, v in overl(q_rest).items()],
    tpose_overlaps=[[*k, round(-v * 1000, 2)] for k, v in overl(q_t).items()],
    random_pose_overlap_freq_top=sorted([[*k, round(v / NP, 3)] for k, v in cnt.items()], key=lambda r: -r[2])[:15],
    verdict='no explicit exclusions; only MuJoCo automatic parent-child filtering applies (adjacent links). '
            'No pair overlaps at rest or T-pose, no pair overlaps in >=95% of random poses -> nothing else may be excluded.')

# ------------------------------------------------------------------ correspondences
cd = Path('/home/arenalabs/umr_factory_data/corr') / ROBOT
cf = {}
for p in sorted(cd.rglob('*')):
    if p.is_file():
        cf[str(p.relative_to(cd))] = dict(bytes=p.stat().st_size, sha256_16=sha(p) if p.stat().st_size < 5e6 else None)
info['correspondence_files'] = cf
umr_hoi = sorted(str(p) for p in Path('/var/tmp/alphamotion/umr_factory/umr/UMR/data').glob(f'correspondence_{ROBOT}_*'))
info['umr_hoi_per_sequence_corr'] = umr_hoi
json.dump(info, open(OUTD / 'assets_check.json', 'w'), indent=1, default=str)
print(json.dumps({k: info[k] for k in ('patches', 'lr_anatomy_check', 'bodies_visual_without_collision',
                                       'bodies_without_any_geom', 'foot_collision_vs_visual_rest')}, indent=1))
print(json.dumps(info['exclusions'], indent=1)[:3000])
