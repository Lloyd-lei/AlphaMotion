"""Teammate's refined SOMA->H2 clip -> Megatron codes -> decoded on G1 and on the human (SMPL) skeleton.
Panels: teammate H2 (source) | BoneSeed official G1 (gold) | H2->G1 (Megatron) | H2->human (Megatron, stick)."""
import os, sys, json, time, numpy as np, torch, mujoco, subprocess
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from alphamotion.embodiment import registry
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import fk_from_angles, project
from alphamotion.engine.spatial import fk_pos
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d, rot6d_to_matrix, geodesic_deg
from arena.eval.visual import RobotRenderer
from arena.refiner.robots import profiles
import imageio.v2 as iio
from PIL import Image, ImageDraw

AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
OUT = os.path.dirname(os.path.abspath(__file__))
G1_CSV = '/var/tmp/arena/raw/bones_seed/mjlab_csv'
G1_MJLAB_XML = '/home/arenalabs/alphamotion_ctrl/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
H2_META = '/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json'
H2_NPZ = '/var/tmp/arena/h2_bm/src/motions'
CLIPS = [('getup', 'crutch_rise_up_R_001__A519'), ('sit', 'sit_croos_legged_stop_001__A024'),
         ('injured', 'injured_R_leg_walk_ff_loop_360_006__A072'), ('dance', 'dance_basic_turn_v2_360_R_loop_fast_002__A323')]
dev = 'cuda'; t0 = time.time()
g = Greenwich.load(model='megatron', device=dev)
eg = registry.load('unitree_g1_29dof'); spec_g, dof_g, rest_g = eg.spec, torch.as_tensor(eg.dof).float(), torch.as_tensor(eg.rest).float()
_, _, _, qn_g, _ = build_from_mjcf(eg.xml, 'g1x'); gmap = {n: (k, s) for k, names in enumerate(qn_g) for s, n in enumerate(names)}
mg = mujoco.MjModel.from_xml_path(G1_MJLAB_XML); g1_hinges = [mg.joint(i).name for i in range(mg.njnt) if mg.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
meta = json.load(open(H2_META)); ms = meta['merged_skeleton']
spec_h = SkeletonSpec('h2_soma43', np.array(ms['parents']), np.array(ms['rest_offsets_cm'], np.float32), list(ms['joint_names']))
dof_h = torch.as_tensor(np.array(ms['dof_features'], np.float32)); rest_h = torch.as_tensor(np.array(ms['rest_rotations'], np.float32))
hslot = {x['native_joint_name']: (x['node_index'], x['slot_index']) for x in meta['native_slots']}
spec_u, dof_u, rest_u = g.embodiment('human_smpl'); dof_u = torch.as_tensor(dof_u).float(); rest_u = torch.as_tensor(rest_u).float()
Jg = spec_g.J; gnode = {n: i for i, n in enumerate(spec_g.joint_names)}
P = profiles(); leg_g = P['unitree_g1_29dof']['left']['leg_len_cm']; leg_h = 0.5 * (P['unitree_h2']['left']['leg_len_cm'] + P['unitree_h2']['right']['leg_len_cm'])
EEG = {'l_hand': 'left_wrist_yaw_link', 'r_hand': 'right_wrist_yaw_link', 'l_foot': 'left_ankle_roll_link', 'r_foot': 'right_ankle_roll_link', 'head': 'head_link'}

def rot6d_np(x):
    a, b = x[:, :3], x[:, 3:]; a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (a * b).sum(1, keepdims=True) * a; b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], 2)
def load_g1(stem):
    rows = np.loadtxt(f'{G1_CSV}/{stem}.csv', delimiter=',')[::4]
    Rc = AX @ Rot.from_quat(rows[:, 3:7]).as_matrix() @ AX.T; root_cm = rows[:, :3] * 100 @ AX.T
    q = np.zeros((len(rows), Jg, 3), np.float32)
    for i, h in enumerate(g1_hinges): k, s = gmap[h]; q[:, k, s] = rows[:, 7 + i]
    return q, Rc.astype(np.float32), root_cm.astype(np.float32)
def load_h2(stem):
    z = np.load(f'{H2_NPZ}/{stem}.npz', allow_pickle=True)
    return z['h2__q'].astype(np.float32), rot6d_np(z['h2__root_rot6d'].astype(np.float64)).astype(np.float32), z['h2__root_t_cm'].astype(np.float32)
def fk(q, spec, dof, rest, root_R):
    return fk_from_angles(torch.as_tensor(q), spec, dof, rest=rest, root_R=torch.as_tensor(root_R))
def encode(gR, spec, dof):
    pose, _ = g.pose9(matrix_to_rot6d(gR).float(), spec, is_global=True); return g.encode(pose, spec, dof).cpu()
def zcat(c):
    Pn = g.n_slots[0]; zp = g.model.fsqp.from_ints(c[None, :, :Pn].to(dev)); zr = g.model.fsqr.from_ints(c[None, :, Pn:].to(dev))
    return torch.cat([zp.flatten(1), zr.flatten(1)], 1).cpu()
def latent(a, b):
    T = min(len(a), len(b)); a, b = a[:T], b[:T]
    return dict(agree=(a == b).float().mean().item(), cos=torch.nn.functional.cosine_similarity(zcat(a), zcat(b), dim=1).mean().item())
def yaw(R): return np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2]))
def q_for_renderer(qdict, r):
    q = np.zeros((len(next(iter(qdict.values()))), len(r.qnames), 3), np.float32)
    for k, names in enumerate(r.qnames):
        for s, n in enumerate(names):
            if n in qdict: q[:, k, s] = qdict[n]
    return q
def low_z(r):
    """lowest point of any collision geom of the renderer's current pose (m, MuJoCo Z-up)"""
    m, d = r.model, r.data; zmin = 1e9
    for gi in range(m.ngeom):
        if not (m.geom_contype[gi] or m.geom_conaffinity[gi]) or m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE: continue
        c = d.geom_xpos[gi]; s = m.geom_size[gi]; R = d.geom_xmat[gi].reshape(3, 3); t = m.geom_type[gi]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE: z = c[2] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE: z = c[2] - abs(R[2, 2]) * s[1] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX: z = c[2] - np.abs(R[2]).dot(s)
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER: z = c[2] - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0, 1 - R[2, 2] ** 2))
        else: z = c[2] - m.geom_rbound[gi]
        zmin = min(zmin, z)
    return zmin
def stick_video(pos_world_cm, parents, path, title, fps=30, W=960, H=540):
    """pos [T,J,3] canonical (x left, y up, z forward) cm -> mp4 of a stick figure, 3/4 view, camera follows the root."""
    par = np.asarray(parents); T = len(pos_world_cm); w = iio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=1)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = fig.add_subplot(111, projection='3d')
    for t in range(T):
        ax.cla(); p = pos_world_cm[t] / 100.0; X, Y, Z = p[:, 2], p[:, 0], p[:, 1]   # plot: x=forward, y=left, z=up
        cx, cy = X[0], Y[0]
        gx, gy = np.meshgrid(np.arange(-3, 4) + round(cx), np.arange(-3, 4) + round(cy))
        ax.plot_wireframe(gx, gy, np.zeros_like(gx), color='0.85', linewidth=0.5)
        for j in range(len(par)):
            k = par[j]
            if k >= 0: ax.plot([X[k], X[j]], [Y[k], Y[j]], [Z[k], Z[j]], color='#d0522a', linewidth=3)
        ax.scatter(X, Y, Z, color='#222', s=8)
        ax.set_xlim(cx - 1.4, cx + 1.4); ax.set_ylim(cy - 1.4, cy + 1.4); ax.set_zlim(0, 2.8); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=14, azim=-135); ax.set_axis_off(); ax.set_title(f'{title}   frame {t}/{T}', fontsize=9, loc='left')
        fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        w.append_data(np.ascontiguousarray(img))
    w.close(); plt.close(fig)

rend_g = RobotRenderer('unitree_g1_29dof', width=960, height=540); rend_h = RobotRenderer('unitree_h2', width=960, height=540, xml=H2_XML)
results = {}
for key, stem in CLIPS:
    qg, Rg, rootg = load_g1(stem); qs, Rs, roots = load_h2(stem)
    T = min(len(qg), len(qs)); qg, Rg, rootg, qs, Rs, roots = qg[:T], Rg[:T], rootg[:T], qs[:T], Rs[:T], roots[:T]
    # align the teammate clip's heading to the official G1 at frame 0 (the codec is not yaw invariant)
    delta = float(((yaw(Rs) - yaw(Rg) + 180) % 360 - 180)[0]); c, s_ = np.cos(np.radians(-delta)), np.sin(np.radians(-delta))
    Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]], np.float32); Rs = np.einsum('ab,tbc->tac', Ry, Rs); roots = roots @ Ry.T
    gRg, gpg = fk(qg, spec_g, dof_g, rest_g, Rg); gRs, gps = fk(qs, spec_h, dof_h, rest_h, Rs)
    cG = encode(gRg, spec_g, dof_g); cS = encode(gRs, spec_h, dof_h)
    # ---- H2 -> G1
    r6, pos = g.decode_full(cS.to(dev), spec_g, dof_g); _, _, qsg = project(r6.cpu(), spec_g, dof_g, rest=rest_g); qsg = qsg.numpy()
    Rsg = rot6d_to_matrix(r6[:, 0].cpu()).numpy(); gRsg, gpsg = fk(qsg, spec_g, dof_g, rest_g, Rsg); cSG = encode(gRsg, spec_g, dof_g)
    rootsg = np.stack([roots[:, 0] / (leg_h / leg_g), pos[:, 0, 0].cpu().numpy() * 100, roots[:, 2] / (leg_h / leg_g)], 1).astype(np.float32)
    rootsg[:, [0, 2]] += rootg[0, [0, 2]] - rootsg[0, [0, 2]]
    # ---- H2 -> human
    r6u, posu = g.decode_full(cS.to(dev), spec_u, dof_u); pu = fk_pos(r6u.cpu(), spec_u)                       # root-relative cm
    rootu = np.stack([roots[:, 0], posu[:, 0, 0].cpu().numpy() * 100, roots[:, 2]], 1); pu_w = pu + rootu[:, None]
    cSU = encode(rot6d_to_matrix(r6u.cpu()), spec_u, dof_u)
    # ---- numbers: H2->G1 vs official G1
    ang = np.degrees(np.abs(np.stack([qsg[:, gmap[h][0], gmap[h][1]] - qg[:, gmap[h][0], gmap[h][1]] for h in g1_hinges], 1)))
    grp = lambda ws: ang[:, [i for i, h in enumerate(g1_hinges) if any(w in h for w in ws)]].mean()
    obs = {'joint_deg_mean': float(ang.mean()), 'joint_deg_p95': float(np.percentile(ang, 95)), 'legs_deg': float(grp(('hip', 'knee', 'ankle'))),
           'arms_deg': float(grp(('shoulder', 'elbow', 'wrist'))), 'waist_deg': float(grp(('waist',))),
           'root_rot_deg': float(geodesic_deg(torch.as_tensor(Rsg)[:, None], torch.as_tensor(Rg)[:, None]).mean()),
           'root_height_diff_cm': float(np.abs(rootsg[:, 1] - rootg[:, 1]).mean()), 'root_h_G1': [float(rootg[:, 1].min()), float(rootg[:, 1].max())],
           'root_h_S2G1': [float(rootsg[:, 1].min()), float(rootsg[:, 1].max())], 'human_root_h': [float(rootu[:, 1].min()), float(rootu[:, 1].max())]}
    for nm, jn in EEG.items(): obs[f'ee_{nm}_cm'] = float((gpsg[:, gnode[jn]] - gpg[:, gnode[jn]]).norm(dim=-1).mean())
    lat = {'S_vs_G1': latent(cS, cG), 'S2G1_reenc_vs_G1': latent(cSG, cG), 'S2G1_reenc_vs_S': latent(cSG, cS), 'S2human_reenc_vs_S': latent(cSU, cS)}
    # floor penetration of the decoded G1 (lowest collision geom, official G1 model) + display lift
    qd_sg = {h: qsg[:, gmap[h][0], gmap[h][1]] for h in g1_hinges}; qr = q_for_renderer(qd_sg, rend_g); lz = np.zeros(T)
    for t in range(T): rend_g._set_frame(qr[t], Rsg[t], rootsg[t], None, follow=False); lz[t] = low_z(rend_g) * 100
    obs['S2G1_lowest_geom_cm_min'] = float(lz.min()); obs['S2G1_pen_frames'] = float((lz < -0.5).mean())
    lift = np.clip(-lz, 0, None); rootsg_d = rootsg.copy(); rootsg_d[:, 1] += lift; obs['S2G1_display_lift_cm_mean'] = float(lift.mean())
    hu_low = pu_w[:, :, 1].min(1); obs['human_lowest_joint_cm_min'] = float(hu_low.min()); liftu = np.clip(-hu_low, 0, None); pu_d = pu_w.copy(); pu_d[:, :, 1] += liftu[:, None]
    results[key] = {'T': T, 'heading_delta_deg': delta, 'latent': lat, 'obs': obs}; print(key, json.dumps(results[key]), flush=True)
    d = f'{OUT}/rev_{key}'; os.makedirs(d, exist_ok=True)
    rend_h.render(q_for_renderer({n: qs[:, k, s] for n, (k, s) in hslot.items()}, rend_h), Rs, roots, f'{d}/h2', fps=30, title=f'{key}: teammate SOMA->H2 (refined, source)')
    rend_g.render(q_for_renderer({h: qg[:, gmap[h][0], gmap[h][1]] for h in g1_hinges}, rend_g), Rg, rootg, f'{d}/g1', fps=30, title=f'{key}: BoneSeed official G1 (gold)')
    rend_g.render(qr, Rsg, rootsg_d, f'{d}/h2_g1', fps=30, title=f'{key}: H2->G1 (Megatron; root lifted off floor for display)')
    stick_video(pu_d, spec_u.parents, f'{d}/h2_human.mp4', f'{key}: H2->human SMPL (Megatron)')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f'{d}/h2/review.mp4', '-i', f'{d}/g1/review.mp4', '-i', f'{d}/h2_g1/review.mp4', '-i', f'{d}/h2_human.mp4',
                    '-filter_complex', '[0:v][1:v][2:v][3:v]hstack=inputs=4[v]', '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', f'{d}/quad.mp4'], check=True)
    rd = iio.get_reader(f'{d}/quad.mp4'); n = rd.count_frames(); idx = np.linspace(0, n - 1, 12).round().astype(int)
    sheet = Image.new('RGB', (1920, 270 * 12), (5, 5, 5))
    for r_, i in enumerate(idx): sheet.paste(Image.fromarray(rd.get_data(int(i))).resize((1920, 270)), (0, r_ * 270))
    rd.close(); sheet.save(f'{d}/grid12.jpg', quality=85); print('  rendered', d, flush=True)
json.dump(results, open(f'{OUT}/results_reverse.json', 'w'), indent=1); print('done', round(time.time() - t0, 1))
