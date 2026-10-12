"""G1->H2 vs SOMA->H2 through Megatron: latent (code) similarity + observation space.
G1 = BoneSeed official G1 retarget of a SOMA clip (mjlab CSV, 120 fps -> 30 fps).
S  = teammate's SOMA->H2 output (AlphaMotion Prime), 43-node H2 skeleton from the dataset metadata.
H  = Megatron: encode(G1) -> decode on the same 43-node H2 skeleton -> feasibility projection.
"""
import os, sys, json, time, numpy as np, torch, mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
from scipy.spatial.transform import Rotation as Rot
from alphamotion.embodiment import registry
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import fk_from_angles, project
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d, rot6d_to_matrix, geodesic_deg
from arena.eval.visual import RobotRenderer
from arena.refiner.robots import profiles

AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
OUT = os.path.dirname(os.path.abspath(__file__))
G1_CSV = '/var/tmp/arena/raw/bones_seed/mjlab_csv'
G1_MJLAB_XML = '/home/arenalabs/alphamotion_ctrl/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
H2_META = '/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json'
H2_NPZ = '/var/tmp/arena/h2_bm/src/motions'
CLIPS = [('getup', 'crutch_rise_up_R_001__A519', 'crutch_rise_up_R_001__A519'),
         ('sit', 'sit_croos_legged_stop_001__A024', 'sit_croos_legged_stop_001__A024'),
         ('injured', 'injured_R_leg_walk_ff_loop_360_006__A072', 'injured_R_leg_walk_ff_loop_360_006__A072'),
         ('dance', 'dance_basic_turn_v2_360_R_loop_fast_002__A323', 'dance_basic_turn_v2_360_R_loop_fast_002__A323')]
RENDER = os.environ.get('RENDER', '1') == '1'
dev = 'cuda'
t0 = time.time()
g = Greenwich.load(model='megatron', device=dev)
print('megatron loaded', round(time.time() - t0, 1), 's; slots', g.n_slots, flush=True)

# ---------------- G1 descriptor + joint map
eg = registry.load('unitree_g1_29dof'); spec_g, dof_g, rest_g = eg.spec, torch.as_tensor(eg.dof).float(), torch.as_tensor(eg.rest).float()
_, _, _, qn_g, _ = build_from_mjcf(eg.xml, 'g1x')           # verified identical spec/dof/rest to the bundled cache
gmap = {n: (k, s) for k, names in enumerate(qn_g) for s, n in enumerate(names)}
mg = mujoco.MjModel.from_xml_path(G1_MJLAB_XML)
g1_hinges = [mg.joint(i).name for i in range(mg.njnt) if mg.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
assert all(h in gmap for h in g1_hinges), [h for h in g1_hinges if h not in gmap]

# ---------------- H2 teammate descriptor (43 nodes) from dataset metadata
meta = json.load(open(H2_META)); ms = meta['merged_skeleton']
spec_h = SkeletonSpec('h2_soma43', np.array(ms['parents']), np.array(ms['rest_offsets_cm'], np.float32), list(ms['joint_names']))
dof_h = torch.as_tensor(np.array(ms['dof_features'], np.float32)); rest_h = torch.as_tensor(np.array(ms['rest_rotations'], np.float32))
hslot = {x['native_joint_name']: (x['node_index'], x['slot_index']) for x in meta['native_slots']}
mh = mujoco.MjModel.from_xml_path(H2_XML); h2_hinges = [mh.joint(i).name for i in range(1, mh.njnt)]
assert all(h in hslot for h in h2_hinges)
Jg, Jh = spec_g.J, spec_h.J
node = {n: i for i, n in enumerate(spec_h.joint_names)}
gnode = {n: i for i, n in enumerate(spec_g.joint_names)}
# homologous end effectors (node indices): G1 vs H2-43
EE = {'l_hand': ('left_wrist_yaw_link', 'left_wrist_yaw_link'), 'r_hand': ('right_wrist_yaw_link', 'right_wrist_yaw_link'),
      'l_foot': ('left_ankle_roll_link', 'left_ankle_pitch_link'), 'r_foot': ('right_ankle_roll_link', 'right_ankle_pitch_link'),
      'head': ('head_link', 'head_yaw_link')}
P = profiles(); leg_g = P['unitree_g1_29dof']['left']['leg_len_cm']; leg_h = 0.5 * (P['unitree_h2']['left']['leg_len_cm'] + P['unitree_h2']['right']['leg_len_cm'])
LEG_RATIO = leg_h / leg_g
print(f'leg length G1 {leg_g} cm, H2 {leg_h:.1f} cm, ratio {LEG_RATIO:.3f}', flush=True)

def rot6d_np(x):
    a, b = x[:, :3], x[:, 3:]; a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (a * b).sum(1, keepdims=True) * a; b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], 2)

def load_g1(stem):
    rows = np.loadtxt(f'{G1_CSV}/{stem}.csv', delimiter=',')[::4]        # 120 -> 30 fps
    pos_m, quat, qj = rows[:, :3], rows[:, 3:7], rows[:, 7:]
    Rn = Rot.from_quat(quat).as_matrix(); Rc = AX @ Rn @ AX.T
    root_cm = pos_m * 100 @ AX.T
    q = np.zeros((len(rows), Jg, 3), np.float32)
    for i, h in enumerate(g1_hinges):
        k, s = gmap[h]; q[:, k, s] = qj[:, i]
    return q, Rc.astype(np.float32), root_cm.astype(np.float32)

def load_h2(stem):
    z = np.load(f'{H2_NPZ}/{stem}.npz', allow_pickle=True)
    return z['h2__q'].astype(np.float32), rot6d_np(z['h2__root_rot6d'].astype(np.float64)).astype(np.float32), z['h2__root_t_cm'].astype(np.float32), float(z['fps'])

def fk(q, spec, dof, rest, root_R):
    gR, gp = fk_from_angles(torch.as_tensor(q), spec, dof, rest=rest, root_R=torch.as_tensor(root_R))
    return gR, gp                                                  # root-relative cm, canonical frame

def encode(gR, spec, dof):
    pose, reach = g.pose9(matrix_to_rot6d(gR).float(), spec, is_global=True)
    return g.encode(pose, spec, dof).cpu(), reach

def zcat(codes):
    Pn = g.n_slots[0]
    zp = g.model.fsqp.from_ints(codes[None, :, :Pn].to(dev)); zr = g.model.fsqr.from_ints(codes[None, :, Pn:].to(dev))
    return torch.cat([zp.flatten(1), zr.flatten(1)], 1).cpu()

def latent(a, b):
    T = min(len(a), len(b)); a, b = a[:T], b[:T]
    Pn = g.n_slots[0]
    agree = (a == b).float(); l1 = (a - b).abs().float()
    za, zb = zcat(a), zcat(b)
    cos = torch.nn.functional.cosine_similarity(za, zb, dim=1)
    return dict(agree=agree.mean().item(), agree_pos=agree[:, :Pn].mean().item(), agree_rot=agree[:, Pn:].mean().item(),
                l1_levels=l1.mean().item(), cos=cos.mean().item(), cos_p05=cos.quantile(0.05).item())

def mujoco_soles(qdict_series, root_R, root_cm, m, feet):
    """H2 official MJCF FK: lowest collision-mesh vertex of each foot (cm), per frame."""
    d = mujoco.MjData(m); T = len(root_cm); sole = np.zeros((T, 2)); names = [m.joint(i).name for i in range(1, m.njnt)]
    for t in range(T):
        d.qpos[:3] = (root_cm[t] @ AX) / 100; d.qpos[3:7] = Rot.from_matrix(AX.T @ root_R[t] @ AX).as_quat(scalar_first=True)
        d.qpos[7:] = [qdict_series[n][t] for n in names]; mujoco.mj_kinematics(m, d)
        for k, (gid, v) in enumerate(feet):
            w = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]; sole[t, k] = w[:, 2].min() * 100
    return sole

feet = []
for side in ('left', 'right'):
    b = mh.body(f'{side}_ankle_pitch_link').id
    gid = [i for i in range(mh.ngeom) if mh.geom_bodyid[i] == b and mh.geom_contype[i]][0]; mid = mh.geom_dataid[gid]
    feet.append((gid, mh.mesh_vert[mh.mesh_vertadr[mid]:mh.mesh_vertadr[mid] + mh.mesh_vertnum[mid]]))

def h2_qdict(q43):
    return {n: q43[:, k, s] for n, (k, s) in hslot.items()}

def q_for_renderer(qdict, r):
    q = np.zeros((len(next(iter(qdict.values()))), len(r.qnames), 3), np.float32)
    for k, names in enumerate(r.qnames):
        for s, n in enumerate(names):
            if n in qdict: q[:, k, s] = qdict[n]
    return q

results = {}; store = {}
rend_g = RobotRenderer('unitree_g1_29dof', width=960, height=540) if RENDER else None
rend_h = RobotRenderer('unitree_h2', width=960, height=540, xml=H2_XML) if RENDER else None
for key, gstem, hstem in CLIPS:
    print(f'\n=== {key}', flush=True)
    qg, Rg, rootg = load_g1(gstem); qs, Rs, roots, fps = load_h2(hstem)
    T = min(len(qg), len(qs)); qg, Rg, rootg, qs, Rs, roots = qg[:T], Rg[:T], rootg[:T], qs[:T], Rs[:T], roots[:T]

    print(f'frames G1 {len(np.loadtxt(f"{G1_CSV}/{gstem}.csv", delimiter=",", max_rows=None))}/4 vs H2 {len(np.load(f"{H2_NPZ}/{hstem}.npz")["h2__q"])} -> T={T}', flush=True)
    # --- gauge check: our FK vs MuJoCo FK of the same CSV rows (G1), root-relative body positions
    gRg, gpg = fk(qg, spec_g, dof_g, rest_g, Rg)
    d = mujoco.MjData(mg); rows = np.loadtxt(f'{G1_CSV}/{gstem}.csv', delimiter=',')[::4][:T]; errs = []
    for t in range(0, T, max(1, T // 8)):
        d.qpos[:3] = rows[t, :3]; d.qpos[3:7] = rows[t, 3:7][[3, 0, 1, 2]]; d.qpos[7:] = rows[t, 7:]; mujoco.mj_kinematics(mg, d)
        for n in ('left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link'):
            pm = (d.xpos[mg.body(n).id] - d.xpos[mg.body('pelvis').id]) * 100 @ AX.T        # native Z-up m -> canonical cm
            errs.append(np.linalg.norm(pm - gpg[t, gnode[n]].numpy()))
    print(f'  gauge G1: our FK vs MuJoCo FK, ee pos err max {max(errs):.2f} cm', flush=True)
    # --- heading gauge: yaw (about canonical +Y) of the root forward axis; the codec is NOT yaw invariant, so align G1 to S at frame 0
    def yaw(R): return np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2]))          # forward axis = 3rd column; components x (left), z (forward)
    dyaw = (yaw(Rg) - yaw(Rs) + 180) % 360 - 180
    delta = float(dyaw[0]); c, s_ = np.cos(np.radians(-delta)), np.sin(np.radians(-delta))
    Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]], np.float32)
    print(f'  heading G1-S at frame0 {delta:+.1f} deg, drift over clip std {dyaw.std():.1f} deg, mean {dyaw.mean():+.1f} -> rotating G1 by {-delta:+.1f} deg about Y', flush=True)
    Rg = np.einsum('ab,tbc->tac', Ry, Rg); rootg = rootg @ Ry.T
    gRg, gpg = fk(qg, spec_g, dof_g, rest_g, Rg)                       # recompute G1 global rotations in the aligned heading
    # --- encode both
    cG, reach_g = encode(gRg, spec_g, dof_g)
    gRs, gps = fk(qs, spec_h, dof_h, rest_h, Rs)
    cS, reach_s = encode(gRs, spec_h, dof_h)
    # --- G1 -> H2 (decode on the 43-node H2 skeleton) + projection to hinges
    r6, pos, con = g.decode_full(cG.to(dev), spec_h, dof_h, contact=True)
    r6p, gph, qh = project(r6.cpu(), spec_h, dof_h, rest=rest_h)
    qh = qh.numpy(); Rh = rot6d_to_matrix(r6[:, 0].cpu()).numpy()
    rooth = np.stack([rootg[:, 0] * LEG_RATIO, pos[:, 0, 0].cpu().numpy() * 100, rootg[:, 2] * LEG_RATIO], 1).astype(np.float32)
    rooth[:, 0] -= rooth[0, 0] - roots[0, 0]; rooth[:, 2] -= rooth[0, 2] - roots[0, 2]   # same start xz as the teammate clip (display only)
    gRh, gph = fk(qh, spec_h, dof_h, rest_h, Rh)
    cH, _ = encode(gRh, spec_h, dof_h)                                   # re-encode the H2 result (round trip)
    # --- G1 self round trip (pipeline sanity)
    r6g, posg = g.decode_full(cG.to(dev), spec_g, dof_g)
    _, _, qgg = project(r6g.cpu(), spec_g, dof_g, rest=rest_g)
    gRgg, gpgg = fk(qgg.numpy(), spec_g, dof_g, rest_g, Rg)
    self_rot = geodesic_deg(gRg, gRgg).mean().item(); self_pos = (gpg - gpgg).norm(dim=-1).mean().item()
    cGG, _ = encode(gRgg, spec_g, dof_g)
    # --- S self round trip on H2 (teammate output through Megatron and back)
    r6s, poss = g.decode_full(cS.to(dev), spec_h, dof_h)
    _, _, qss = project(r6s.cpu(), spec_h, dof_h, rest=rest_h)
    gRss, gpss = fk(qss.numpy(), spec_h, dof_h, rest_h, Rs)
    s_self_rot = geodesic_deg(gRs, gRss).mean().item(); s_self_pos = (gps - gpss).norm(dim=-1).mean().item()
    cSS, _ = encode(gRss, spec_h, dof_h)
    # --- latent
    shift = int(fps)                                                     # 1 s time-shift floor
    lat = {'G_vs_S': latent(cG, cS), 'G_vs_H(G->H2 re-encoded)': latent(cG, cH), 'G_vs_G_selfRT': latent(cG, cGG),
           'S_vs_S_selfRT': latent(cS, cSS), 'floor_G_vs_S_shift1s': latent(cG[shift:], cS[:-shift]),
           'H_vs_S': latent(cH, cS)}
    # --- observation space on H2 (43 nodes): H (G1->H2) vs S (teammate)
    qd_h, qd_s = h2_qdict(qh), h2_qdict(qs)
    ang = {n: np.degrees(np.abs(qd_h[n] - qd_s[n])) for n in h2_hinges}
    grp = lambda ws: np.concatenate([ang[n] for n in h2_hinges if any(w in n for w in ws)])
    obs = {'joint_deg_mean': float(np.mean(np.stack(list(ang.values())))), 'joint_deg_p95': float(np.percentile(np.stack(list(ang.values())), 95)),
           'legs_deg_mean': float(grp(('hip', 'knee', 'ankle')).mean()), 'arms_deg_mean': float(grp(('shoulder', 'elbow', 'wrist')).mean()),
           'waist_head_deg_mean': float(grp(('waist', 'head')).mean()),
           'root_rot_deg_mean': float(geodesic_deg(torch.as_tensor(Rh)[:, None], torch.as_tensor(Rs)[:, None]).mean()),
           'root_height_diff_cm_mean': float(np.abs(rooth[:, 1] - roots[:, 1]).mean()),
           'root_height_H_min_max': [float(rooth[:, 1].min()), float(rooth[:, 1].max())], 'root_height_S_min_max': [float(roots[:, 1].min()), float(roots[:, 1].max())]}
    for nm, (gn, hn) in EE.items():
        obs[f'ee_{nm}_H_vs_S_cm'] = float((gph[:, node[hn]] - gps[:, node[hn]]).norm(dim=-1).mean())
    # each H2 version vs the official G1, root-relative positions normalised by leg length (reported on the H2 scale)
    for tag, gp_ in (('S', gps), ('H', gph)):
        e = [((gpg[:, gnode[gn]] / leg_g - gp_[:, node[hn]] / leg_h).norm(dim=-1) * leg_h).mean().item() for gn, hn in EE.values()]
        obs[f'ee_vs_G1gold_{tag}_cm'] = float(np.mean(e))
    solh = mujoco_soles(qd_h, Rh, rooth, mh, feet); sols = mujoco_soles(qd_s, Rs, roots, mh, feet)
    obs['sole_min_cm_H'] = float(solh.min()); obs['sole_min_cm_S'] = float(sols.min())
    obs['floor_pen_frames_H'] = float((solh.min(1) < -0.5).mean()); obs['floor_pen_frames_S'] = float((sols.min(1) < -0.5).mean())
    lift = np.clip(-solh.min(1), 0, None); rooth_disp = rooth.copy(); rooth_disp[:, 1] += lift
    obs['H_render_root_lift_cm_mean'] = float(lift.mean()); obs['H_render_root_lift_cm_max'] = float(lift.max())
    obs['self_roundtrip_G1_rot_deg'] = self_rot; obs['self_roundtrip_G1_pos_cm'] = self_pos
    obs['self_roundtrip_S_rot_deg'] = s_self_rot; obs['self_roundtrip_S_pos_cm'] = s_self_pos
    results[key] = {'T': T, 'latent': lat, 'obs': obs}
    print(json.dumps(results[key], indent=1), flush=True)
    np.savez(f'{OUT}/{key}_codes.npz', cG=cG.numpy(), cS=cS.numpy(), cH=cH.numpy(), qh=qh, qs=qs, qg=qg, Rh=Rh, rooth=rooth, Rs=Rs, roots=roots)
    if RENDER:
        qd_g = {h: qg[:, gmap[h][0], gmap[h][1]] for h in g1_hinges}
        rend_g.render(q_for_renderer(qd_g, rend_g), Rg, rootg, f'{OUT}/{key}/g1', fps=30, title=f'{key}: BoneSeed official G1 (source)')
        rend_h.render(q_for_renderer(qd_s, rend_h), Rs, roots, f'{OUT}/{key}/soma_h2', fps=30, title=f'{key}: SOMA->H2 (teammate, Prime)')
        rend_h.render(q_for_renderer(qd_h, rend_h), Rh, rooth_disp, f'{OUT}/{key}/g1_h2', fps=30, title=f'{key}: G1->H2 (Megatron; root lifted off floor for display)')
        print('  rendered', flush=True)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
print('done', round(time.time() - t0, 1), 's')
