"""unitree_h2 dedicated pipeline -- step 1: asset check (read-only on shared assets).

Renders T-pose and rest (qpos0, feet on z=0) with the 4096 dense surface points coloured by region (front + side),
checks region placement / left-right, lists collision geoms per region, audits self-collision exclusions (MuJoCo
parent-child filter + staged <exclude>), and checks the correspondence caches.
Writes: pipeline/assets_check.json, pipeline/assets_tpose_regions.png, pipeline/assets_rest_regions.png
Run:  MUJOCO_GL=egl timeout 600 <umr python> assets_check.py
"""
import hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco, numpy as np
from PIL import Image, ImageDraw

ROBOT = 'unitree_h2'
RD = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT
PD = RD / 'pipeline'
CORR = Path('/home/arenalabs/umr_factory_data/corr') / ROBOT
XML = RD / f'{ROBOT}.xml'
st = json.load(open(RD / 'staging.json'))
surf = dict(np.load(RD / 'surface.npz', allow_pickle=True))
REG = [str(r) for r in surf['regions']]
m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
m.vis.global_.offwidth = 1920; m.vis.global_.offheight = 1080
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
GT = {0: 'plane', 2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'}

COL = {'trunk': (150, 150, 150), 'L_thigh': (230, 25, 75), 'R_thigh': (0, 130, 200), 'L_shin': (245, 130, 48),
       'R_shin': (70, 240, 240), 'L_foot': (128, 0, 0), 'R_foot': (0, 0, 128), 'L_upper_arm': (255, 225, 25),
       'R_upper_arm': (60, 180, 75), 'L_forearm': (240, 50, 230), 'R_forearm': (145, 30, 180),
       'L_hand': (170, 110, 40), 'R_hand': (0, 128, 128)}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def posed_points():
    g = surf['geom'].astype(int); R = d.geom_xmat[g].reshape(-1, 3, 3)
    return np.einsum('nij,nj->ni', R, surf['points_local'].astype(float)) + d.geom_xpos[g]


def set_pose(tpose):
    mujoco.mj_resetData(m, d); d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]
    if tpose:
        for jn, v in st['tpose'].items():
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
    mujoco.mj_forward(m, d)
    X = posed_points(); d.qpos[2] -= X[:, 2].min(); mujoco.mj_forward(m, d)   # lowest dense point on z = 0
    return posed_points()


def render(X, reg, azim, title):
    W = H = 900
    r = mujoco.Renderer(m, H, W, max_geom=m.ngeom + len(X) + 10)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [0, 1, 0, 0, 0, 0]     # visual group 1 only (collision hidden)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [X[:, 0].mean(), X[:, 1].mean(), X[:, 2].mean()]
    cam.distance = 2.9; cam.azimuth = azim; cam.elevation = -8
    rgba0 = m.geom_rgba.copy(); m.geom_rgba[:, 3] = 0.18
    r.update_scene(d, cam, opt)
    sc = r.scene
    for p, k in zip(X, reg):
        if sc.ngeom >= sc.maxgeom:
            break
        c = np.array(COL[REG[k]] + (255,), float) / 255
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]), p,
                            np.eye(3).reshape(-1), c.astype(np.float32))
        sc.ngeom += 1
    img = Image.fromarray(r.render()); r.close(); m.geom_rgba[:] = rgba0
    ImageDraw.Draw(img).text((10, 10), title, fill=(0, 0, 0))
    return img


def legend(w):
    img = Image.new('RGB', (w, 60), (255, 255, 255)); dr = ImageDraw.Draw(img)
    for i, k in enumerate(REG):
        x = 10 + (i % 7) * (w // 7); y = 8 + (i // 7) * 26
        dr.rectangle([x, y, x + 16, y + 16], fill=COL[k]); dr.text((x + 22, y + 3), f'{k} ({(surf["region"] == i).sum()})', fill=(0, 0, 0))
    return img


reg = surf['region'].astype(int)
out = dict(robot=ROBOT, mjcf=str(XML), mjcf_sha256=sha(XML), surface=str(RD / 'surface.npz'),
           surface_sha256=sha(RD / 'surface.npz'), n_points=int(len(reg)),
           points_per_region={k: int((reg == i).sum()) for i, k in enumerate(REG)})
checks = {}
for name, tp in (('tpose', True), ('rest', False)):
    X = set_pose(tp)
    cen = {k: X[reg == i].mean(0) for i, k in enumerate(REG)}
    zr = {k: [float(X[reg == i][:, 2].min()), float(X[reg == i][:, 2].max())] for i, k in enumerate(REG)}
    c = {}
    for part in ('thigh', 'shin', 'foot', 'upper_arm', 'forearm', 'hand'):
        c[f'{part}_L_is_plus_y'] = bool(cen[f'L_{part}'][1] > 0.02 and cen[f'R_{part}'][1] < -0.02)
    c['feet_lowest'] = bool(min(zr['L_foot'][0], zr['R_foot'][0]) <= min(v[0] for v in zr.values()) + 1e-6)
    c['leg_order_z'] = bool(cen['L_thigh'][2] > cen['L_shin'][2] > cen['L_foot'][2] and cen['R_thigh'][2] > cen['R_shin'][2] > cen['R_foot'][2])
    if tp:
        c['tpose_hand_outermost'] = bool(cen['L_hand'][1] > cen['L_forearm'][1] > cen['L_upper_arm'][1] and
                                         cen['R_hand'][1] < cen['R_forearm'][1] < cen['R_upper_arm'][1])
    else:
        c['rest_hand_below_elbow'] = bool(cen['L_hand'][2] < cen['L_forearm'][2] < cen['L_upper_arm'][2] and
                                          cen['R_hand'][2] < cen['R_forearm'][2] < cen['R_upper_arm'][2])
    # sole: foot points within 5 mm of the foot minimum at rest = sole patch
    checks[name] = dict(checks=c, centroid_m={k: np.round(v, 3).tolist() for k, v in cen.items()},
                        z_range_m={k: np.round(v, 3).tolist() for k, v in zr.items()})
    imgs = [render(X, reg, 180, f'{ROBOT} {name}: front (robot faces +x)'), render(X, reg, 90, f'{name}: left side (+y toward camera)')]
    sheet = Image.new('RGB', (1800, 960), (255, 255, 255)); sheet.paste(imgs[0], (0, 0)); sheet.paste(imgs[1], (900, 0))
    sheet.paste(legend(1800), (0, 900))
    sheet.save(PD / f'assets_{name}_regions.png')
    if not tp:
        out['rest_root_height_m'] = float(d.qpos[2])
        # foot sole patch statistics at rest (points within 5 mm of the floor)
        sole = {}
        for s in 'LR':
            i = REG.index(f'{s}_foot'); idx = np.flatnonzero(reg == i)
            z = X[idx, 2]; sole[s] = dict(n_foot=int(len(idx)), n_within_2mm=int((z < 0.002).sum()), n_within_5mm=int((z < 0.005).sum()),
                                         sole_xy_extent_m=np.round(np.ptp(X[idx[z < 0.005]][:, :2], 0), 3).tolist() if (z < 0.005).any() else None)
        out['rest_sole_patch'] = sole
out['region_checks'] = checks
# ---------------------------------------------------------------- collision geoms per region
bodyreg = surf['body_region'].astype(int)
cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
per = {k: [] for k in REG}
for g in cg:
    b = int(m.geom_bodyid[g])
    per[REG[bodyreg[b]]].append(dict(geom=int(g), body=bn(b), type=GT.get(int(m.geom_type[g]), int(m.geom_type[g])),
                                     size=np.round(m.geom_size[g], 4).tolist()))
out['collision_geoms_per_region'] = per
out['n_collision_geoms'] = len(cg)
vis_bodies = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_group[g] == 1 and m.geom_bodyid[g] != 0})
col_bodies = sorted({int(m.geom_bodyid[g]) for g in cg})
out['bodies_with_visual_but_no_collision'] = [bn(b) for b in vis_bodies if b not in col_bodies]
# dense points on bodies without collision geom (not covered by the MuJoCo self/object cross-check)
nb = np.isin(surf['body'], [b for b in vis_bodies if b not in col_bodies])
out['dense_points_on_bodies_without_collision'] = {REG[i]: int((nb & (reg == i)).sum()) for i in range(len(REG)) if (nb & (reg == i)).any()}
# visual vs collision mismatch: how far the visual dense points of each collision-body stick out of / sit inside its collision geoms at rest
# ---------------------------------------------------------------- self-collision exclusion audit
excl_xml = [(e.get('body1'), e.get('body2')) for e in []]
spec_txt = open(XML).read()
out['staged_exclude_elements_in_mjcf'] = spec_txt.count('<exclude')
out['staging_exclusions'] = st['exclusions']
out['staging_flagged_nonadjacent_rest_overlaps'] = st['flagged_nonadjacent_rest_overlaps']


def pair_dists(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    res = {}
    ft = np.zeros(6)
    for i, g1 in enumerate(cg):
        for g2 in cg[i + 1:]:
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                continue                                  # MuJoCo parent-child filter (adjacent): never checked
            dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, ft)
            k = (bn(min(b1, b2)), bn(max(b1, b2)))
            res[k] = min(res.get(k, 1.0), dist)
    return res


set_pose(True); qt = d.qpos.copy(); set_pose(False); q0 = d.qpos.copy()
rest_pairs = pair_dists(q0); tp_pairs = pair_dists(qt)
rng = np.random.default_rng(0)
Q = np.repeat(q0[None], 300, 0)
for j in range(1, m.njnt):
    lo, hi = m.jnt_range[j]; Q[:, m.jnt_qposadr[j]] = rng.uniform(lo, hi, 300)
hits = {}
for q in Q:
    for k, v in pair_dists(q).items():
        if v < -1e-4:
            hits[k] = hits.get(k, 0) + 1
out['nonadjacent_pairs_checked'] = len(rest_pairs)
out['rest_overlaps_lt_0p1mm'] = {f'{a}|{b}': round(v * 1000, 2) for (a, b), v in rest_pairs.items() if v < -1e-4}
out['tpose_overlaps_lt_0p1mm'] = {f'{a}|{b}': round(v * 1000, 2) for (a, b), v in tp_pairs.items() if v < -1e-4}
out['rest_min_gap_mm_closest5'] = sorted([(f'{a}|{b}', round(v * 1000, 1)) for (a, b), v in rest_pairs.items()], key=lambda x: x[1])[:5]
out['random_pose_overlap_frac_top'] = sorted([(f'{a}|{b}', round(h / 300, 3)) for (a, b), h in hits.items()], key=lambda x: -x[1])[:10]
out['adjacent_parent_child_pairs_filtered_by_mujoco'] = sorted({f'{bn(int(m.body_parentid[b]))}|{bn(b)}' for b in range(2, m.nbody)
                                                              if int(m.body_parentid[b]) != 0})
# ---------------------------------------------------------------- correspondence
cc = {}
for tag in ('smplx', 'beta0'):
    ds = CORR / tag / 'correspondence_dataset.npz'; sl = CORR / tag / 'train' / 'correspondence_slots_final.npz'
    rec = dict(dataset=str(ds), dataset_exists=ds.exists(), slots=str(sl), slots_exists=sl.exists())
    if sl.exists():
        z = np.load(sl, allow_pickle=True)
        rec['slots_names'] = [str(x) for x in z['names']] if 'names' in z.files else None
        rec['n_slots'] = int(z['reconstructed_slots'].shape[1]) if 'reconstructed_slots' in z.files else None
        rec['slots_sha256'] = sha(sl)
    if (CORR / tag / 'smplx_surface_binding.npz').exists():
        rec['binding'] = str(CORR / tag / 'smplx_surface_binding.npz')
    cc[tag] = rec
cc['other_dirs'] = sorted(p.name for p in CORR.iterdir() if p.name not in ('smplx', 'beta0'))
out['correspondence'] = cc
json.dump(out, open(PD / 'assets_check.json', 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in ('points_per_region', 'n_collision_geoms', 'bodies_with_visual_but_no_collision',
                                      'dense_points_on_bodies_without_collision', 'rest_overlaps_lt_0p1mm',
                                      'tpose_overlaps_lt_0p1mm', 'rest_min_gap_mm_closest5', 'random_pose_overlap_frac_top',
                                      'rest_sole_patch')}, indent=1, default=str))
print(json.dumps({k: v['checks'] for k, v in checks.items()}, indent=1))
print(json.dumps(cc, indent=1, default=str)[:2000])
