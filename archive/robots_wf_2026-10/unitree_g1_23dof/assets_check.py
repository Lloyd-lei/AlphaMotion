"""Step 1 -- asset check for unitree_g1_23dof.

Renders the T-pose and rest pose (front / back / left side / bottom) with the 4096 dense points coloured by region,
a contact-patch sheet, lists collision geoms per region, reviews self-collision exclusions (staged list + MuJoCo's own
implicit filter) with overlap statistics over rest / T-pose / 300 random in-limit poses, checks left/right placement
numerically and the correspondence caches.  Output: QADIR/assets/*.png + assets_check.json
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import g1common as C

t0 = time.time()
OUTD = C.QADIR / 'assets'; OUTD.mkdir(parents=True, exist_ok=True)
m = C.load_model(); d = mujoco.MjData(m)
surf = C.load_surface()
st = json.load(open(C.RDIR / 'staging.json'))
reg = surf['region'].astype(int)
rep = dict(robot=C.ROBOT, mjcf=str(C.XML), mjcf_sha256=C.sha256(C.XML), surface_sha256=C.sha256(C.SURF),
           nq=m.nq, nv=m.nv, nbody=m.nbody, ngeom=m.ngeom, n_dense=int(len(reg)), n_hull_vertices=int(len(surf['hull_local'])))

# ------------------------------------------------------------------ poses
def tpose_q():
    q = m.qpos0.copy()
    for jn, v in st['tpose'].items():
        q[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
    return q

poses = {'tpose': tpose_q(), 'rest': m.qpos0.copy()}
pose_pts = {}
for k, q in poses.items():
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    X = C.posed_points(d, surf['geom'], surf['points_local'])
    H = C.posed_points(d, surf['hull_geom'], surf['hull_local'])
    dz = min(X[:, 2].min(), H[:, 2].min())
    q = q.copy(); q[2] -= dz; poses[k] = q                    # lowest point on z = 0
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    pose_pts[k] = C.posed_points(d, surf['geom'], surf['points_local'])

# ------------------------------------------------------------------ region sanity (numbers)
regs = {}
Xt = pose_pts['tpose']; Xr = pose_pts['rest']
bn = [C.name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
for i, r in enumerate(C.REGIONS):
    k = reg == i
    bodies = sorted({bn[b] for b in surf['body'][k]})
    regs[r] = dict(n_points=int(k.sum()), bodies=bodies,
                   tpose_mean_xyz=np.round(Xt[k].mean(0), 3).tolist(), rest_mean_xyz=np.round(Xr[k].mean(0), 3).tolist(),
                   rest_zmin=round(float(Xr[k, 2].min()), 4), rest_zmax=round(float(Xr[k, 2].max()), 4))
checks = {}
for r in C.REGIONS:
    if r.startswith('L_'):
        checks[f'{r}_on_left(+y)'] = bool(regs[r]['tpose_mean_xyz'][1] > 0.03)
    if r.startswith('R_'):
        checks[f'{r}_on_right(-y)'] = bool(regs[r]['tpose_mean_xyz'][1] < -0.03)
# vertical order at rest: foot < shin < thigh < trunk centre; hands at the arm tips in T-pose
for s in 'LR':
    checks[f'{s}_foot<shin<thigh (rest z)'] = bool(regs[f'{s}_foot']['rest_mean_xyz'][2] < regs[f'{s}_shin']['rest_mean_xyz'][2]
                                                  < regs[f'{s}_thigh']['rest_mean_xyz'][2])
    sg = 1 if s == 'L' else -1
    checks[f'{s}_upper_arm<forearm<hand (tpose |y|)'] = bool(
        sg * regs[f'{s}_upper_arm']['tpose_mean_xyz'][1] < sg * regs[f'{s}_forearm']['tpose_mean_xyz'][1] < sg * regs[f'{s}_hand']['tpose_mean_xyz'][1])
    checks[f'{s}_foot_is_lowest_at_rest'] = bool(regs[f'{s}_foot']['rest_zmin'] < 0.002)
checks['robot_faces_+x (toes at +x)'] = bool(Xr[reg == C.RID['L_foot'], 0].max() > Xr[reg == C.RID['L_foot'], 0].min() + 0.15
                                             and np.abs(Xr[reg == C.RID['L_foot'], 0].max()) > np.abs(Xr[reg == C.RID['L_foot'], 0].min()))
rep['regions'] = regs; rep['region_checks'] = checks; rep['region_checks_all_pass'] = all(checks.values())

# ------------------------------------------------------------------ patches
P, Xrest0 = C.contact_patches(m, surf)
rep['patches'] = {k: dict(n_points=int(v.sum()), regions=sorted({C.REGIONS[i] for i in np.unique(reg[v])})) for k, v in P.items()}
np.savez_compressed(C.PDIR / 'contact_patches.npz', **{k: np.flatnonzero(v).astype(np.int32) for k, v in P.items()})

# ------------------------------------------------------------------ collision geoms per region
TN = {int(getattr(mujoco.mjtGeom, k)): k.replace('mjGEOM_', '') for k in dir(mujoco.mjtGeom) if k.startswith('mjGEOM_')}
breg = surf['body_region']
cg = C.robot_collision_geoms(m)
per_reg = {r: [] for r in C.REGIONS}
for g in cg:
    b = int(m.geom_bodyid[g])
    mesh = C.name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None
    per_reg[C.REGIONS[breg[b]]].append(dict(geom=int(g), body=bn[b], type=TN[int(m.geom_type[g])],
                                            size=np.round(m.geom_size[g], 4).tolist(), mesh=mesh))
rep['collision_geoms_per_region'] = per_reg
# bodies whose visual surface has no collision geom
vis_bodies = sorted({bn[b] for b in np.unique(surf['body'])})
col_bodies = sorted({bn[m.geom_bodyid[g]] for g in cg})
rep['visual_bodies_without_collision'] = [b for b in vis_bodies if b not in col_bodies]
# foot: collision spheres vs visual sole
d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
foot = {}
for s, side in (('L', 'left'), ('R', 'right')):
    fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{side}_ankle_roll_link')
    sph = [g for g in cg if m.geom_bodyid[g] == fb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    Xf = C.posed_points(d, surf['geom'], surf['points_local'])[reg == C.RID[f'{s}_foot']]
    Hf = C.posed_points(d, surf['hull_geom'], surf['hull_local'])[breg[m.geom_bodyid[surf['hull_geom']]] == C.RID[f'{s}_foot']]
    sb = [float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in sph]
    foot[s] = dict(n_sole_spheres=len(sph), sphere_radius_m=float(m.geom_size[sph[0]][0]) if sph else None,
                   sphere_bottom_minus_visual_sole_mm=round((min(sb) - Hf[:, 2].min()) * 1000, 2),
                   sphere_x_range=[round(float(min(d.geom_xpos[g][0] for g in sph)), 3), round(float(max(d.geom_xpos[g][0] for g in sph)), 3)],
                   visual_sole_x_range=[round(float(Hf[:, 0].min()), 3), round(float(Hf[:, 0].max()), 3)],
                   heel_overhang_cm=round(float(min(d.geom_xpos[g][0] for g in sph) - m.geom_size[sph[0]][0] - Hf[:, 0].min()) * 100, 1),
                   toe_overhang_cm=round(float(Hf[:, 0].max() - max(d.geom_xpos[g][0] for g in sph) - m.geom_size[sph[0]][0]) * 100, 1))
rep['foot_collision_vs_visual'] = foot

# ------------------------------------------------------------------ exclusions / self-collision filtering
considered = C.mujoco_considered_pairs(m)
allpairs = {(a, b) for i, a in enumerate(cg) for b in cg[i + 1:]}
filtered = sorted(allpairs - considered)
def pair_reason(a, b):
    return C.mujoco_filter_reason(m, a, b)
filt_body = {}
for a, b in filtered:
    k = tuple(sorted((bn[m.geom_bodyid[a]], bn[m.geom_bodyid[b]])))
    filt_body.setdefault(k, pair_reason(a, b))
# overlap stats for considered pairs over rest, T-pose and random in-limit poses
rng = np.random.default_rng(0)
lo = m.jnt_range[:, 0]; hi = m.jnt_range[:, 1]
qs = [poses['rest'], poses['tpose']]
for _ in range(300):
    q = m.qpos0.copy()
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            q[m.jnt_qposadr[j]] = rng.uniform(lo[j], hi[j])
    qs.append(q)
pen_count = {}; pen_rest = {}; pen_t = {}
cons = sorted(considered)
for qi, q in enumerate(qs):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    for a, b in cons:
        dist = mujoco.mj_geomDistance(m, d, a, b, 0.01, None)
        if dist < -0.0005:
            k = tuple(sorted((bn[m.geom_bodyid[a]], bn[m.geom_bodyid[b]])))
            if qi == 0:
                pen_rest[k] = min(pen_rest.get(k, 0), dist)
            elif qi == 1:
                pen_t[k] = min(pen_t.get(k, 0), dist)
            else:
                pen_count[k] = pen_count.get(k, 0) + 1
rep['exclusions'] = dict(
    staged_exclude_list=st['exclusions'], mjcf_nexclude=int(m.nexclude),
    filterparent_enabled=not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT),
    n_collision_geoms=len(cg), n_geom_pairs_total=len(allpairs), n_geom_pairs_checked_by_mujoco=len(considered),
    implicitly_filtered_body_pairs=[dict(bodies=list(k), reason=v) for k, v in sorted(filt_body.items())],
    nonfiltered_pairs_penetrating_at_rest=[dict(bodies=list(k), depth_mm=round(-v * 1000, 2)) for k, v in pen_rest.items()],
    nonfiltered_pairs_penetrating_at_tpose=[dict(bodies=list(k), depth_mm=round(-v * 1000, 2)) for k, v in pen_t.items()],
    nonfiltered_pairs_penetrating_random_poses=[dict(bodies=list(k), frac_of_300=round(v / 300, 3))
                                                for k, v in sorted(pen_count.items(), key=lambda x: -x[1])][:20],
    always_overlapping_candidates=[list(k) for k, v in pen_count.items() if v >= 0.95 * 300],
)
# non-adjacent filtered pairs (should be none): reason not parent/weld
rep['exclusions']['nonadjacent_filtered'] = [dict(bodies=list(k), reason=v) for k, v in filt_body.items()
                                            if not v.startswith(('same weld', 'parent-child'))]

# ------------------------------------------------------------------ correspondence
cr = {}
for tag in ('smplx', 'beta0'):
    dd = Path('/home/arenalabs/umr_factory_data/corr') / C.ROBOT / tag
    sl = dd / 'train' / 'correspondence_slots_final.npz'
    ds = dd / 'correspondence_dataset.npz'
    e = dict(dir=str(dd), dataset_exists=ds.exists(), slots_exists=sl.exists())
    if sl.exists():
        z = np.load(sl, allow_pickle=True)
        e['slots_names'] = [str(x) for x in z['names']] if 'names' in z.files else None
        e['slots_shape'] = list(z['reconstructed_slots'].shape) if 'reconstructed_slots' in z.files else None
        e['slots_sha256'] = C.sha256(sl)
    if ds.exists():
        z = np.load(ds, allow_pickle=True)
        e['dataset_points_shape'] = list(z['points'].shape) if 'points' in z.files else None
    cr[tag] = e
rep['correspondence'] = cr

# ------------------------------------------------------------------ renders
mr = C.load_model()
for g in cg:
    mr.geom_rgba[g][3] = 0.0                 # hide collision geoms, keep floor
mr.vis.global_.offwidth = 2000; mr.vis.global_.offheight = 2000
dr = mujoco.MjData(mr)
W, Hh = 420, 520
rgba_reg = np.array([list(C.REGION_RGB[C.REGIONS[r]]) + [255] for r in reg], np.float32) / 255
opt = C.vis_option(show_floor=True)
views = [('front', 180, -5), ('back', 0, -5), ('right side (camera at -y)', 90, -5), ('bottom', 180, 85)]


def render_pose(q, rgba, label, pts_size=0.006, alpha_robot=1.0):
    dr.qpos[:] = q; mujoco.mj_forward(mr, dr)
    X = C.posed_points(dr, surf['geom'], surf['points_local'])
    c = X.mean(0)
    tiles = []
    with mujoco.Renderer(mr, Hh, W) as ren:
        for vn, az, el in views:
            cam = C.make_cam([c[0], c[1], 0.62 if vn != 'bottom' else 0.3], 2.6, az, el if vn != 'bottom' else -89.0)
            o2 = C.vis_option(show_floor=(vn != 'bottom'))
            if vn == 'bottom':
                cam.elevation = 89.0                    # look up from below (floor hidden)
                cam.lookat[:] = [c[0], c[1], 0.0]; cam.distance = 1.5
            ren.update_scene(dr, cam, o2)
            C.add_points(ren.scene, X, rgba, pts_size)
            im = Image.fromarray(ren.render())
            ImageDraw.Draw(im).text((6, 6), f'{label} | {vn}', fill=(0, 0, 0))
            tiles.append(im)
    row = Image.new('RGB', (W * len(tiles), Hh), (255, 255, 255))
    for i, t in enumerate(tiles):
        row.paste(t, (i * W, 0))
    return row


rows = [render_pose(poses['tpose'], rgba_reg, 'T-pose'), render_pose(poses['rest'], rgba_reg, 'rest (qpos0)')]
leg_h = 60
sheet = Image.new('RGB', (rows[0].width, sum(r.height for r in rows) + leg_h), (255, 255, 255))
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
dd = ImageDraw.Draw(sheet)
x = 8
for i, rn in enumerate(C.REGIONS):
    col = C.REGION_RGB[rn]
    dd.rectangle([x, y + 10, x + 14, y + 24], fill=col); dd.text((x + 18, y + 11), rn, fill=(0, 0, 0)); x += 125
    if x > sheet.width - 120:
        x = 8; y += 24
dd.text((8, sheet.height - 16), f'{C.ROBOT}: 4096 dense points by region (left=warm, right=cool). Robot faces +x; '
        'front view: robot LEFT appears on image RIGHT.', fill=(0, 0, 0))
sheet.save(OUTD / 'regions_tpose_rest.png')

# patch sheet (rest pose): patches coloured, rest grey
pc = {'L_sole': (240, 220, 0), 'R_sole': (20, 30, 120), 'L_knee': (220, 40, 40), 'R_knee': (40, 80, 220),
      'L_shin_front': (255, 140, 0), 'R_shin_front': (0, 190, 220), 'L_elbow': (230, 60, 160), 'R_elbow': (40, 170, 70),
      'L_hand': (255, 170, 190), 'R_hand': (150, 230, 150), 'pelvis': (120, 60, 20), 'torso': (120, 60, 200),
      'head': (0, 0, 0)}
rgba_p = np.full((len(reg), 4), [0.85, 0.85, 0.85, 0.35], np.float32)
for k in ['torso', 'pelvis', 'head', 'L_shin_front', 'R_shin_front', 'L_knee', 'R_knee', 'L_elbow', 'R_elbow', 'L_hand',
          'R_hand', 'L_sole', 'R_sole']:
    rgba_p[P[k]] = np.array(list(pc[k]) + [255], np.float32) / 255
prow = [render_pose(poses['rest'], rgba_p, 'contact patches (rest)'), render_pose(poses['tpose'], rgba_p, 'contact patches (T)')]
ps = Image.new('RGB', (prow[0].width, 2 * Hh + 40), (255, 255, 255))
ps.paste(prow[0], (0, 0)); ps.paste(prow[1], (0, Hh))
dd = ImageDraw.Draw(ps); x = 8
for k, col in pc.items():
    dd.rectangle([x, 2 * Hh + 12, x + 14, 2 * Hh + 26], fill=col); dd.text((x + 18, 2 * Hh + 13), k, fill=(0, 0, 0)); x += 128
ps.save(OUTD / 'contact_patches.png')

# collision geometry render (rest + T-pose), collision geoms visible, visual hidden
mc = C.load_model(); mc.vis.global_.offwidth = 2000; mc.vis.global_.offheight = 2000
dc = mujoco.MjData(mc)
for g in cg:
    b = int(mc.geom_bodyid[g])
    mc.geom_rgba[g] = np.array(list(C.REGION_RGB[C.REGIONS[breg[b]]]) + [255]) / 255
crow = []
for lab, q in (('collision geoms rest', poses['rest']), ('collision geoms T', poses['tpose'])):
    dc.qpos[:] = q; mujoco.mj_forward(mc, dc)
    o3 = mujoco.MjvOption()
    for gi in range(6):
        o3.geomgroup[gi] = 0
    o3.geomgroup[0] = 1
    tiles = []
    with mujoco.Renderer(mc, Hh, W) as ren:
        for vn, az, el in views[:3]:
            cam = C.make_cam([0, 0, 0.62], 2.6, az, el)
            ren.update_scene(dc, cam, o3); im = Image.fromarray(ren.render())
            ImageDraw.Draw(im).text((6, 6), f'{lab} | {vn}', fill=(0, 0, 0)); tiles.append(im)
    row = Image.new('RGB', (W * 3, Hh)); [row.paste(t, (i * W, 0)) for i, t in enumerate(tiles)]; crow.append(row)
cs = Image.new('RGB', (W * 3, 2 * Hh)); cs.paste(crow[0], (0, 0)); cs.paste(crow[1], (0, Hh))
cs.save(OUTD / 'collision_geoms.png')

rep['renders'] = [str(OUTD / 'regions_tpose_rest.png'), str(OUTD / 'contact_patches.png'), str(OUTD / 'collision_geoms.png')]
rep['seconds'] = round(time.time() - t0, 1)
json.dump(rep, open(OUTD / 'assets_check.json', 'w'), indent=1)
print(json.dumps({k: rep[k] for k in ('region_checks_all_pass', 'patches', 'visual_bodies_without_collision',
                                      'foot_collision_vs_visual', 'seconds')}, indent=1))
print(json.dumps(rep['exclusions'], indent=1)[:4000])
print(json.dumps(rep['correspondence'], indent=1))
