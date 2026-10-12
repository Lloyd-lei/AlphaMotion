"""Step 1 -- unitree_r1 asset check: MJCF, dense surface regions, contact patches, collision geoms, exclusions,
correspondence.  Writes qa/unitree_r1/assets/{assets_check.json, rest_regions.png, tpose_regions.png, patches.png}.
Run:  MUJOCO_GL=egl timeout 900 <umr env python> assets_check.py"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from PIL import Image

import r1_common as C

OUT = C.QADIR / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
m = C.load_model()
d = mujoco.MjData(m)
surf = C.load_surface()
st = json.load(open(C.RDIR / 'staging.json'))
P = C.Poser(m, surf)
reg = P.reg
nm = lambda b: C.bname(m, b)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def pose(kind):
    q = np.zeros(m.nq); q[3] = 1.0
    d.qpos[:] = q
    if kind == 'tpose':
        C.set_named(m, d, st['tpose'])
    q = d.qpos.copy()
    P.set(q)
    X = P.points()
    q[2] -= X[:, 2].min()                       # lowest dense point on z = 0 for display
    P.set(q)
    return q, P.points()


# ------------------------------------------------------------------ contact patches (robot-specific definitions)
def joint_anchor(q, jn):
    P.set(q)
    return P.d.xanchor[C.jid(m, jn)].copy()


q_rest, X_rest = pose('rest')
q_t, X_t = pose('tpose')
P.set(q_rest)
body = surf['body'].astype(int)
patch = {}
for s, side in (('L', 'left'), ('R', 'right')):
    ka = joint_anchor(q_rest, f'{side}_knee_joint'); P.set(q_rest); Xr = P.points()
    dk = np.linalg.norm(Xr - ka, axis=1)
    patch[f'{s}_knee'] = np.flatnonzero(((reg == C.RID[f'{s}_shin']) | (reg == C.RID[f'{s}_thigh'])) & (dk < 0.08))
    ea = joint_anchor(q_t, f'{side}_elbow_joint'); P.set(q_t); Xt = P.points()
    de = np.linalg.norm(Xt - ea, axis=1)
    patch[f'{s}_elbow'] = np.flatnonzero(((reg == C.RID[f'{s}_upper_arm']) | (reg == C.RID[f'{s}_forearm'])) & (de < 0.06))
    patch[f'{s}_hand'] = np.flatnonzero(reg == C.RID[f'{s}_hand'])
    fi = np.flatnonzero(reg == C.RID[f'{s}_foot'])
    patch[f'{s}_foot_all'] = fi
    # sole = foot points within 1.5 cm of the foot's lowest point at rest with a downward-facing normal
    P.set(q_rest); Xr = P.points()
    g = surf['geom'][fi]
    nw = np.einsum('nij,nj->ni', P.d.geom_xmat[g].reshape(-1, 3, 3), surf['normals_local'][fi])
    zmin = Xr[fi, 2].min()
    patch[f'{s}_sole'] = fi[(Xr[fi, 2] < zmin + 0.015) & (nw[:, 2] < -0.5)]
pel = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('pelvis_link', 'imu_in_pelvis_link')]
patch['pelvis'] = np.flatnonzero(np.isin(body, pel))
patch['torso_back'] = np.flatnonzero(np.isin(body, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'waist_yaw_link'),
                                                     mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'waist_roll_link')]))
np.savez_compressed(C.PDIR / 'patches.npz', **{k: v.astype(np.int32) for k, v in patch.items()})

# ------------------------------------------------------------------ checks
checks = {}
counts = {r: int((reg == i).sum()) for i, r in enumerate(C.REGIONS)}
checks['n_points'] = int(len(reg)); checks['region_counts'] = counts
breg = surf['body_region']
checks['region_bodies'] = {r: [nm(b) for b in range(1, m.nbody) if breg[b] == i] for i, r in enumerate(C.REGIONS)}
checks['point_bodies_per_region'] = {r: sorted({nm(b) for b in body[reg == i]}) for i, r in enumerate(C.REGIONS)}
lr = {}
for nmpose, X in (('rest', X_rest), ('tpose', X_t)):
    lr[nmpose] = {r: round(float(X[reg == i, 1].mean()), 4) for i, r in enumerate(C.REGIONS)}
checks['mean_y_per_region_m'] = lr
checks['left_is_plus_y'] = all(lr[p][r] > 0 for p in lr for r in C.REGIONS if r.startswith('L_')) and \
    all(lr[p][r] < 0 for p in lr for r in C.REGIONS if r.startswith('R_'))
# soles are the lowest part at rest
zmin_all = X_rest[:, 2].min()
checks['rest_lowest_region'] = C.REGIONS[int(reg[np.argmin(X_rest[:, 2])])]
checks['rest_foot_min_z_mm'] = {s: round(float(X_rest[reg == C.RID[f'{s}_foot'], 2].min() * 1000), 2) for s in 'LR'}
# hands are the arm extremity in the T-pose
checks['tpose_mean_abs_y_m'] = {k: round(float(np.abs(X_t[reg == C.RID[k], 1]).mean()), 3)
                                for k in ('L_upper_arm', 'L_forearm', 'L_hand', 'R_upper_arm', 'R_forearm', 'R_hand')}
checks['arm_order_ok'] = all(checks['tpose_mean_abs_y_m'][f'{s}_upper_arm'] < checks['tpose_mean_abs_y_m'][f'{s}_forearm'] <
                             checks['tpose_mean_abs_y_m'][f'{s}_hand'] for s in 'LR')
checks['knee_height_rest_m'] = {s: round(float(joint_anchor(q_rest, f'{sd}_knee_joint')[2]), 3) for s, sd in (('L', 'left'), ('R', 'right'))}
checks['patch_counts'] = {k: int(len(v)) for k, v in patch.items()}
# hull vertices vs dense points: how much lower can the true foot go than the dense samples
P.set(q_rest)
hl = surf['hull_local'].astype(float); hg = surf['hull_geom'].astype(int)
Hw = np.einsum('nij,nj->ni', P.d.geom_xmat[hg].reshape(-1, 3, 3), hl) + P.d.geom_xpos[hg]
hreg = breg[m.geom_bodyid[hg]]
Xr = P.points()
checks['rest_foot_dense_minus_hull_min_mm'] = {s: round(float((Xr[reg == C.RID[f'{s}_foot'], 2].min() - Hw[hreg == C.RID[f'{s}_foot'], 2].min()) * 1000), 3) for s in 'LR'}
# spacing of dense points (nearest-neighbour) per contact region
from scipy.spatial import cKDTree
nn = {}
for r in ('L_foot', 'R_foot', 'L_hand', 'R_hand', 'L_shin', 'trunk'):
    Y = X_t[reg == C.RID[r]]
    dd, _ = cKDTree(Y).query(Y, k=2)
    nn[r] = dict(p50_mm=round(float(np.median(dd[:, 1]) * 1000), 2), p95_mm=round(float(np.percentile(dd[:, 1], 95) * 1000), 2))
checks['dense_nn_spacing'] = nn

# collision geoms per region + visual-only bodies
cg = C.collision_geoms(m)
cg_reg = {}
for g in cg:
    r = C.REGIONS[int(breg[m.geom_bodyid[g]])]
    cg_reg.setdefault(r, []).append(dict(geom=int(g), body=nm(m.geom_bodyid[g]), type=int(m.geom_type[g]),
                                         mesh=C.name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if m.geom_type[g] == 7 else None))
checks['collision_geoms_per_region'] = cg_reg
checks['n_collision_geoms'] = len(cg)
checks['bodies_without_collision'] = [nm(b) for b in range(1, m.nbody) if not any(m.geom_bodyid[g] == b for g in cg)]
checks['bodies_without_joint'] = [nm(b) for b in range(1, m.nbody) if m.body_jntnum[b] == 0]

# ------------------------------------------------------------------ self-collision: rest, T-pose, random poses
pairs = C.self_pairs_all(m, [tuple(e['bodies']) if isinstance(e, dict) else tuple(e) for e in st.get('exclusions', [])])
ft = np.zeros(6)


def pair_dists(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    return np.array([mujoco.mj_geomDistance(m, d, a, b, 0.05, ft) for a, b in pairs])


pname = lambda a, b: f'{nm(m.geom_bodyid[a])}|{nm(m.geom_bodyid[b])}'
dr, dt = pair_dists(q_rest), pair_dists(q_t)
checks['n_self_pairs_tested'] = len(pairs)
checks['self_overlap_rest'] = {pname(*pairs[i]): round(float(dr[i] * 1000), 2) for i in np.flatnonzero(dr < 0)}
checks['self_overlap_tpose'] = {pname(*pairs[i]): round(float(dt[i] * 1000), 2) for i in np.flatnonzero(dt < 0)}
rng = np.random.default_rng(0)
hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
NR = 300
ov = np.zeros(len(pairs))
for _ in range(NR):
    q = q_rest.copy()
    for j in hinges:
        lo, hi = m.jnt_range[j]
        q[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
    ov += pair_dists(q) < 0
frac = ov / NR
checks['random_pose_overlap_frac'] = {pname(*pairs[i]): round(float(frac[i]), 3) for i in np.argsort(-frac)[:25] if frac[i] > 0}
checks['always_overlapping_pairs_ge95pct'] = [pname(*pairs[i]) for i in np.flatnonzero(frac >= 0.95)]
# rigidly attached (no joint between them) collision body pairs that MuJoCo would still test
def chain_has_joint(b1, b2):
    a1 = [b1]; x = b1
    while x: x = int(m.body_parentid[x]); a1.append(x)
    a2 = [b2]; x = b2
    while x: x = int(m.body_parentid[x]); a2.append(x)
    lca = next(x for x in a1 if x in a2)
    path = a1[:a1.index(lca)] + a2[:a2.index(lca)]
    return any(m.body_jntnum[b] > 0 for b in path)
checks['rigid_pairs_tested'] = [pname(a, b) for a, b in pairs if not chain_has_joint(int(m.geom_bodyid[a]), int(m.geom_bodyid[b]))]
checks['staging_exclusions'] = st.get('exclusions', [])
checks['staging_flagged_nonadjacent_rest_overlaps'] = st.get('flagged_nonadjacent_rest_overlaps', [])

# ------------------------------------------------------------------ correspondence
corr = {}
for tag in ('smplx', 'beta0'):
    base = Path('/home/arenalabs/umr_factory_data/corr') / C.ROBOT / tag
    sl = base / 'train' / 'correspondence_slots_final.npz'
    ds = base / 'correspondence_dataset.npz'
    e = dict(dataset=str(ds), dataset_exists=ds.exists(), slots=str(sl), slots_exists=sl.exists())
    if sl.exists():
        z = np.load(sl, allow_pickle=True)
        e['slots_keys'] = list(z.files)
        if 'reconstructed_slots' in z.files:
            e['reconstructed_slots_shape'] = list(z['reconstructed_slots'].shape)
        if 'names' in z.files:
            e['names'] = [str(x) for x in z['names']]
        e['slots_sha256'] = sha(sl)[:16]
    corr[tag] = e
checks['correspondence'] = corr

# ------------------------------------------------------------------ renders
def region_sheet(X, title, q):
    W, H = 1500, 820
    img = Image.new('RGB', (W, H), (252, 252, 252))
    scale = 520
    cols = C.REG_COL[reg]
    C.draw_points(img, X, cols, 'front', scale, 300, 760, r=2)
    C.draw_points(img, X, cols, 'side', scale, 800, 760, r=2)
    # mujoco mesh render (front-left 3/4 view)
    try:
        rend = mujoco.Renderer(m, height=420, width=560)
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, X[:, 2].mean()]; cam.distance = 2.0; cam.azimuth = 150; cam.elevation = -10
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
        rend.update_scene(d, cam, opt)
        img.paste(Image.fromarray(rend.render()), (W - 570, 60))
        rend.close()
    except Exception as e:  # pragma: no cover
        C.text(img, (W - 560, 80), f'mujoco render failed: {e!r}'[:80])
    C.text(img, (20, 10), title)
    C.text(img, (150, 40), 'front view (robot faces viewer; robot LEFT on screen RIGHT)', small=True)
    C.text(img, (650, 40), 'side view (robot faces screen right)', small=True)
    for i, r in enumerate(C.REGIONS):
        y = 500 + i * 22
        img.paste(tuple(int(c) for c in C.REG_COL[i]), (W - 560, y, W - 540, y + 16))
        C.text(img, (W - 530, y), f'{r} ({counts[r]})', small=True)
    return img


region_sheet(X_rest, 'unitree_r1 REST (qpos0) -- 4096 dense points coloured by region', q_rest).save(OUT / 'rest_regions.png')
region_sheet(X_t, 'unitree_r1 T-POSE (staging.json) -- 4096 dense points coloured by region', q_t).save(OUT / 'tpose_regions.png')
# patches
W, H = 1500, 820
img = Image.new('RGB', (W, H), (252, 252, 252))
cols = np.full((len(reg), 3), 200, np.uint8)
pc = dict(L_sole=(200, 0, 0), R_sole=(0, 0, 200), L_foot_all=(255, 160, 160), R_foot_all=(160, 160, 255),
          L_knee=(230, 120, 0), R_knee=(0, 150, 150), L_elbow=(160, 0, 160), R_elbow=(0, 160, 0),
          L_hand=(255, 40, 150), R_hand=(0, 200, 220), pelvis=(90, 60, 30), torso_back=(130, 130, 60))
for k in ('L_foot_all', 'R_foot_all', 'L_sole', 'R_sole', 'L_knee', 'R_knee', 'L_elbow', 'R_elbow', 'L_hand', 'R_hand', 'pelvis', 'torso_back'):
    cols[patch[k]] = pc[k]
C.draw_points(img, X_t, cols, 'front', 520, 300, 760, r=2)
C.draw_points(img, X_t, cols, 'side', 520, 800, 760, r=2)
C.draw_points(img, X_t, cols, 'back', 520, 1250, 760, r=2)
C.text(img, (20, 10), 'unitree_r1 contact patches (T-pose): front | side | back')
for i, (k, c) in enumerate(pc.items()):
    y = 60 + i * 22
    img.paste(c, (1100, y, 1120, y + 16)); C.text(img, (1130, y), f'{k} ({len(patch[k])})', small=True)
img.save(OUT / 'patches.png')

checks['seconds'] = round(time.time() - t0, 1)
checks['mjcf_sha256'] = sha(C.XML); checks['surface_sha256'] = sha(C.SURF)
C.dump(OUT / 'assets_check.json', checks)
print(json.dumps(C.jsonable({k: checks[k] for k in ('region_counts', 'left_is_plus_y', 'rest_lowest_region', 'arm_order_ok',
                                                      'patch_counts', 'self_overlap_rest', 'self_overlap_tpose',
                                                      'always_overlapping_pairs_ge95pct', 'rigid_pairs_tested',
                                                      'bodies_without_collision', 'rest_foot_dense_minus_hull_min_mm',
                                                      'dense_nn_spacing', 'seconds')}), indent=1))
