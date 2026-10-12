"""unitree_h1 dedicated pipeline -- step 1: asset check.
Renders T-pose / rest with the 4096 surface points coloured by region (MuJoCo EGL), lists collision geoms per region,
audits self-collision exclusions (staged list + MuJoCo's implicit parent/child filter + overlap scan over random
in-limit poses) and checks the correspondence caches.  Writes pipeline/assets_check.json + qa/unitree_h1/assets_*.png
Usage: timeout 900 <umr py> assets_check.py"""
import json, os, sys, hashlib
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw

ROBOT = 'unitree_h1'
RD = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT
QA = Path('/home/arenalabs/umr_factory_data/qa') / ROBOT
CORR = Path('/home/arenalabs/umr_factory_data/corr') / ROBOT
QA.mkdir(parents=True, exist_ok=True)
st = json.load(open(RD / 'staging.json'))
m = mujoco.MjModel.from_xml_path(st['staged_mjcf']); d = mujoco.MjData(m)
S = np.load(RD / 'surface.npz')
REG = [str(x) for x in S['regions']]
COL = {'trunk': (150, 150, 150), 'L_thigh': (255, 160, 160), 'R_thigh': (160, 160, 255), 'L_shin': (225, 50, 50),
       'R_shin': (50, 50, 225), 'L_foot': (130, 0, 0), 'R_foot': (0, 0, 130), 'L_upper_arm': (255, 205, 110),
       'R_upper_arm': (110, 225, 205), 'L_forearm': (255, 130, 0), 'R_forearm': (0, 165, 165), 'L_hand': (240, 220, 0),
       'R_hand': (0, 200, 0)}
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
reg = S['region'].astype(int); geom = S['geom'].astype(int); loc = S['points_local'].astype(float)
nloc = S['normals_local'].astype(float)


def set_pose(tpose):
    mujoco.mj_resetData(m, d); d.qpos[:] = m.qpos0
    d.qpos[0:3] = [0, 0, 1.1]; d.qpos[3:7] = [1, 0, 0, 0]
    for jn, v in (tpose or {}).items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
    mujoco.mj_forward(m, d)
    # put the lowest visual point on z=0
    X = posed()
    d.qpos[2] -= X[:, 2].min(); mujoco.mj_forward(m, d)


def posed(off=0.0):
    Rm = d.geom_xmat[geom].reshape(-1, 3, 3)
    return np.einsum('nij,nj->ni', Rm, loc + off * nloc) + d.geom_xpos[geom]


def render(title, az, el=-10, W=900, H=1000):
    r = mujoco.Renderer(m, H, W, max_geom=6000)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.9]; cam.distance = 2.9; cam.azimuth = az; cam.elevation = el
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]             # visual meshes only
    r.update_scene(d, camera=cam, scene_option=opt)
    X = posed(0.002)
    sc = r.scene
    for i in range(len(X)):
        if sc.ngeom >= sc.maxgeom:
            break
        c = np.array(COL[REG[reg[i]]]) / 255.0
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0045, 0, 0]), X[i],
                            np.eye(3).reshape(-1), np.r_[c, 1.0].astype(np.float32))
        sc.ngeom += 1
    img = Image.fromarray(r.render()); r.close()
    dr = ImageDraw.Draw(img); dr.text((10, 10), title, fill=(0, 0, 0))
    return img


def legend(W):
    img = Image.new('RGB', (W, 60), (255, 255, 255)); dr = ImageDraw.Draw(img)
    for k, rn in enumerate(REG):
        x = 10 + (k % 7) * (W // 7); y = 8 + (k // 7) * 25
        dr.rectangle([x, y, x + 16, y + 16], fill=COL[rn]); dr.text((x + 22, y + 3), f'{rn} ({(reg == k).sum()})', fill=(0, 0, 0))
    return img


rep = dict(robot=ROBOT, mjcf=st['staged_mjcf'], mjcf_sha256=hashlib.sha256(open(st['staged_mjcf'], 'rb').read()).hexdigest(),
           surface_sha256=hashlib.sha256(open(RD / 'surface.npz', 'rb').read()).hexdigest(), n_points=int(len(reg)),
           points_per_region={REG[k]: int((reg == k).sum()) for k in range(len(REG))},
           split_note=json.loads(str(S['split_note'])))
# ---------------- images + left/right + placement checks
checks = {}
for name, tp in (('tpose', st['tpose']), ('rest', {})):
    set_pose(tp)
    X = posed()
    ck = {}
    for k, rn in enumerate(REG):
        P = X[reg == k]
        if len(P):
            ck[rn] = dict(mean_xyz_cm=np.round(P.mean(0) * 100, 1).tolist(), z_min_cm=round(float(P[:, 2].min() * 100), 1),
                          z_max_cm=round(float(P[:, 2].max() * 100), 1),
                          bodies=sorted({bn(m.geom_bodyid[g]) for g in np.unique(geom[reg == k])}))
    lr = {}
    for part in ('thigh', 'shin', 'foot', 'upper_arm', 'forearm', 'hand'):
        yl, yr = ck[f'L_{part}']['mean_xyz_cm'][1], ck[f'R_{part}']['mean_xyz_cm'][1]
        lr[part] = dict(L_y_cm=yl, R_y_cm=yr, ok=bool(yl > 0 > yr))        # robot faces +x -> left = +y
    # soles: foot points within 5 mm of the foot minimum (rest pose, standing)
    sole = {}
    for s in 'LR':
        P = X[reg == REG.index(f'{s}_foot')]
        sole[s] = dict(n_within_5mm_of_foot_min=int((P[:, 2] - P[:, 2].min() < 0.005).sum()),
                       n_within_1mm=int((P[:, 2] - P[:, 2].min() < 0.001).sum()),
                       foot_len_x_cm=round(float(np.ptp(P[:, 0]) * 100), 1), foot_w_y_cm=round(float(np.ptp(P[:, 1]) * 100), 1),
                       foot_min_is_lowest_of_robot=bool(abs(P[:, 2].min() - X[:, 2].min()) < 1e-6))
    checks[name] = dict(regions=ck, left_right=lr, all_lr_ok=all(v['ok'] for v in lr.values()), soles=sole)
    front = render(f'{ROBOT} {name}: front (robot left = image right)', 180)
    side = render(f'{ROBOT} {name}: left side', -90)
    sheet = Image.new('RGB', (front.width * 2, front.height + 60), (255, 255, 255))
    sheet.paste(front, (0, 0)); sheet.paste(side, (front.width, 0)); sheet.paste(legend(front.width * 2), (0, front.height))
    sheet.save(QA / f'assets_{name}_regions.png')
rep['checks'] = checks
# ---------------- collision geoms per region
br = S['body_region']
cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
per = {}
for g in cg:
    rn = REG[int(br[m.geom_bodyid[g]])]
    per.setdefault(rn, []).append(dict(geom=g, body=bn(m.geom_bodyid[g]), type=mujoco.mjtGeom(m.geom_type[g]).name.replace('mjGEOM_', ''),
                                      mesh=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if m.geom_type[g] == 7 else None))
rep['collision_geoms_per_region'] = per
rep['regions_without_collision'] = [rn for rn in REG if rn not in per]
rep['note_hand_collision'] = 'L/R_hand and L/R_forearm share ONE collision mesh (elbow_link, MuJoCo uses its convex hull); ' \
                             'hand/forearm split exists only for the dense points'
vis_nocoll = [dict(geom=g, body=bn(m.geom_bodyid[g]), mesh=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]))
              for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_type[g] == 7 and not (m.geom_contype[g] or m.geom_conaffinity[g])]
bodies_coll = {int(m.geom_bodyid[g]) for g in cg}
vis_bodies = {v['body'] for v in vis_nocoll}
rep['visual_meshes_without_matching_collision'] = [v for v in vis_nocoll if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, v['body']) not in bodies_coll
                                                   or v['mesh'] not in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) for g in cg if m.geom_type[g] == 7]]
# ---------------- exclusions
rep['staged_exclusions'] = st['exclusions']
rep['mjcf_contact_exclude'] = int(m.nexclude)
filt = bool(not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
par = []
for b in range(1, m.nbody):
    p = int(m.body_parentid[b])
    if p != 0 and b in bodies_coll and p in bodies_coll:
        par.append([bn(p), bn(b)])
rep['implicit_parent_child_filter'] = dict(enabled=filt, pairs=par,
                                           reason='MuJoCo filterparent: joint-adjacent bodies never collide (always overlapping at the joint)')


def hops(a, b):
    def anc(x):
        o = [x]
        while x != 0:
            x = int(m.body_parentid[x]); o.append(x)
        return o
    A, B = anc(a), anc(b)
    common = next(x for x in A if x in B)
    return A.index(common) + B.index(common)


def overlaps(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    out = {}
    for i, g1 in enumerate(cg):
        for g2 in cg[i + 1:]:
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                continue
            ft = np.zeros(6)
            dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, ft)
            if dist < -1e-4:
                k = (bn(min(b1, b2)), bn(max(b1, b2)))
                out[k] = min(out.get(k, 0), dist)
    return out


rng = np.random.default_rng(0)
set_pose({}); q_rest = d.qpos.copy(); o_rest = overlaps(q_rest)
set_pose(st['tpose']); o_t = overlaps(d.qpos.copy())
cnt = {}
N = 300
for _ in range(N):
    q = q_rest.copy()
    for j in range(1, m.njnt):
        lo, hi = m.jnt_range[j]
        q[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
    for k in overlaps(q):
        cnt[k] = cnt.get(k, 0) + 1
rep['non_adjacent_overlap_rest'] = {f'{a}|{b}': round(v * 1000, 2) for (a, b), v in o_rest.items()}
rep['non_adjacent_overlap_tpose'] = {f'{a}|{b}': round(v * 1000, 2) for (a, b), v in o_t.items()}
rep['non_adjacent_overlap_random_poses'] = {f'{a}|{b}': dict(frac=round(v / N, 3), hops=hops(mujoco.mj_name2id(m, 1, a), mujoco.mj_name2id(m, 1, b)))
                                            for (a, b), v in sorted(cnt.items(), key=lambda x: -x[1])}
rep['exclusion_candidates_(>=95%_random_or_rest_overlap_and_<=2hops)'] = [
    k for k, v in rep['non_adjacent_overlap_random_poses'].items() if v['frac'] >= 0.95 or (k in rep['non_adjacent_overlap_rest'] and v['hops'] <= 2)]
# ---------------- correspondence
cr = {}
for tag in ('smplx', 'beta0'):
    p = CORR / tag
    ds, sl = p / 'correspondence_dataset.npz', p / 'train' / 'correspondence_slots_final.npz'
    e = dict(dataset=str(ds), slots=str(sl), exists=ds.exists() and sl.exists())
    if e['exists']:
        z = np.load(sl, allow_pickle=True)
        e['slot_names'] = [str(x) for x in z['names']] if 'names' in z.files else None
        e['keys'] = list(z.files)
        if 'reconstructed_slots' in z.files:
            e['reconstructed_slots_shape'] = list(z['reconstructed_slots'].shape)
        e['dataset_sha256_16'] = hashlib.sha256(open(ds, 'rb').read()).hexdigest()[:16]
        e['slots_sha256_16'] = hashlib.sha256(open(sl, 'rb').read()).hexdigest()[:16]
    cr[tag] = e
rep['correspondence'] = cr
json.dump(rep, open(RD / 'pipeline' / 'assets_check.json', 'w'), indent=1, default=str)
print(json.dumps({k: rep[k] for k in ('points_per_region', 'regions_without_collision', 'staged_exclusions',
                                      'non_adjacent_overlap_rest', 'non_adjacent_overlap_tpose', 'non_adjacent_overlap_random_poses',
                                      'exclusion_candidates_(>=95%_random_or_rest_overlap_and_<=2hops)', 'visual_meshes_without_matching_collision')}, indent=1, default=str))
for nm in ('tpose', 'rest'):
    print(nm, 'LR ok', checks[nm]['all_lr_ok'], json.dumps(checks[nm]['soles']))
print(json.dumps(cr, indent=1, default=str)[:1500])
