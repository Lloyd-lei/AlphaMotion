"""ARDY (G1 skeleton) -> Megatron codes -> G1 / H2 / GR3 joint angles -> locomotion.v1 exam -> 4-panel video + 12-frame grid."""
import os, sys, json, glob, subprocess, time, numpy as np, torch, mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ardy_in import load_ardy, SPEC as SPEC_G, DOF as DOF_G, REST as REST_G, QN as QN_G, NAMES as NAMES_G, eg as EG
from alphamotion.engine.greenwich import Greenwich
from alphamotion.embodiment import registry
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import project, fk_from_angles
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d, rot6d_to_matrix, geodesic_deg
from arena.eval.visual import RobotRenderer
from arena.eval.locomotion import evaluate_clip
from arena.refiner.robots import load_robot, profiles, RobotModel
import imageio.v2 as iio
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__)) + '/runs'; os.makedirs(OUT, exist_ok=True)
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
H2_META = '/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json'
CLIPS = sys.argv[1:] or sorted(glob.glob('/home/arenalabs/Desktop/mingrun/outputs_ardy/original/motions/*.npz'))
dev = 'cuda'; g = Greenwich.load(model='megatron', device=dev)

# ---------------- target bodies
def robot_pack(name):
    e = registry.load(name); spec, dof, rest = e.spec, torch.as_tensor(e.dof).float(), torch.as_tensor(e.rest).float()
    _, _, _, qn, _ = build_from_mjcf(e.xml, name + 'x'); rob = load_robot(name)
    return dict(name=name, spec=spec, dof=dof, rest=rest, qn=qn, robot=rob, renderer=RobotRenderer(name, width=960, height=540, xml=e.xml if 'g1' in name else None))
G1 = robot_pack('unitree_g1_29dof'); GR3 = robot_pack('fourier_gr3')
meta = json.load(open(H2_META)); ms = meta['merged_skeleton']
spec_h = SkeletonSpec('h2_soma43', np.array(ms['parents']), np.array(ms['rest_offsets_cm'], np.float32), list(ms['joint_names']))
dof_h = torch.as_tensor(np.array(ms['dof_features'], np.float32)); rest_h = torch.as_tensor(np.array(ms['rest_rotations'], np.float32))
hslot = {x['native_joint_name']: (x['node_index'], x['slot_index']) for x in meta['native_slots']}
hn = {n: i for i, n in enumerate(spec_h.joint_names)}
prof_h = json.loads(json.dumps(profiles()['unitree_h2']))
prof_h['left'].update(hand=hn['left_wrist_yaw_link'], foot=hn['left_ankle_pitch_link'], shoulder=hn['left_shoulder_pitch_link'], hip=hn['left_hip_pitch_link'])
prof_h['right'].update(hand=hn['right_wrist_yaw_link'], foot=hn['right_ankle_pitch_link'], shoulder=hn['right_shoulder_pitch_link'], hip=hn['right_hip_pitch_link'])
_, p0 = fk_from_angles(torch.zeros(1, spec_h.J, 3, dtype=torch.float64), spec_h, dof_h.double(), rest=rest_h.double())
rob_h = RobotModel(name='unitree_h2', spec=spec_h, dof=dof_h.double(), rest=rest_h.double(), profile=prof_h, source='meta', rest_pos=p0[0].numpy())
H2 = dict(name='unitree_h2', spec=spec_h, dof=dof_h, rest=rest_h, qn=None, robot=rob_h, renderer=RobotRenderer('unitree_h2', width=960, height=540, xml=H2_XML))
leg = {k: 0.5 * (profiles()[k]['left']['leg_len_cm'] + profiles()[k]['right']['leg_len_cm']) for k in ('unitree_g1_29dof', 'unitree_h2', 'fourier_gr3')}
print('leg lengths', leg, flush=True)

def q_for_renderer(qdict, r):
    q = np.zeros((len(next(iter(qdict.values()))), len(r.qnames), 3), np.float32)
    for k, names in enumerate(r.qnames):
        for s, n in enumerate(names):
            if n in qdict: q[:, k, s] = qdict[n]
    return q
def qdict_from(q, qn):
    return {n: q[:, k, s] for k, names in enumerate(qn) for s, n in enumerate(names)}
def low_z(r):
    m, d = r.model, r.data; zmin = 1e9
    for gi in range(m.ngeom):
        if not (m.geom_contype[gi] or m.geom_conaffinity[gi]) or m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE: continue
        c = d.geom_xpos[gi]; s = m.geom_size[gi]; R = d.geom_xmat[gi].reshape(3, 3); t = m.geom_type[gi]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE: z = c[2] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE: z = c[2] - abs(R[2, 2]) * s[1] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX: z = c[2] - np.abs(R[2]).dot(s)
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER: z = c[2] - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0, 1 - R[2, 2] ** 2))
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[gi]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; z = (v @ R.T + c)[:, 2].min()
        else: z = c[2] - m.geom_rbound[gi]
        zmin = min(zmin, z)
    return zmin
def floor_scan(pack, qr, root_R, root_cm):
    r = pack['renderer']; lz = np.zeros(len(qr))
    for t in range(len(qr)): r._set_frame(qr[t], root_R[t], root_cm[t], None, follow=False); lz[t] = low_z(r) * 100
    return lz
def contact_columns(a):
    """ARDY foot_contacts [T,4] column -> (side) by correlating each column with -height of the candidate ARDY foot nodes."""
    P = a['ardy_P']; an = a['ardy_names']; cand = {'L': ['left_toe_base', 'left_ankle_roll_skel'], 'R': ['right_toe_base', 'right_ankle_roll_skel']}
    side = []
    for c in range(a['contacts'].shape[1]):
        best = None
        for s, nodes in cand.items():
            for n in nodes:
                h = P[:, an.index(n), 1]; x = a['contacts'][:, c]
                if x.std() < 1e-6 or h.std() < 1e-6: continue
                r = np.corrcoef(x, -h)[0, 1]
                if best is None or r > best[0]: best = (r, s, n)
        side.append(best)
    return side

results = {}
for path in CLIPS:
    key = os.path.basename(path).replace('.npz', ''); t0 = time.time(); d = f'{OUT}/{key}'; os.makedirs(d, exist_ok=True)
    a = load_ardy(path); T = len(a['q']); fps = a['fps']
    cols = [('L', 'heel'), ('L', 'toe'), ('R', 'heel'), ('R', 'toe')]      # ARDY feet.py: [left heel, left toe, right heel, right toe]
    stance = np.zeros((T, 2), bool)
    for c, (side, _) in enumerate(cols): stance[:, 0 if side == 'L' else 1] |= a['contacts'][:, c] > 0.5
    corr = contact_columns(a)
    print(f'\n=== {key}  T={T} fps={fps}  text="{a["text"]}"  stance frac L/R {stance.mean(0).round(2)}  (corr check {[(b[1], round(b[0], 2)) if b else None for b in corr]})', flush=True)
    pose, reach = g.pose9(matrix_to_rot6d(a['gR']).float(), SPEC_G, is_global=True)
    con = torch.zeros(T, SPEC_G.J)
    for side, col in (('left', 0), ('right', 1)):
        for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, NAMES_G.index(n)] = torch.as_tensor(stance[:, col]).float()
    codes = {'nocon': g.encode(pose, SPEC_G, DOF_G), 'con': g.encode(pose, SPEC_G, DOF_G, contact=con.to(dev))}
    root_g_src = a['root_cm'].copy(); root_g_src[:, [0, 2]] -= root_g_src[:1, [0, 2]]
    thr_src = 0.04 * G1['robot'].standing_root_height_cm / 100.0
    ev_src = evaluate_clip(G1['robot'], a['q'].astype(np.float64), a['root_R'].astype(np.float64), root_g_src.astype(np.float64), fps, stance, thr_src)
    res = {'T': T, 'fps': fps, 'text': a['text'], 'ardy_fk_gauge_cm_mean': float(np.mean(list(a['fk_err_cm'].values()))), 'stance_frac': stance.mean(0).tolist(),
           'source_g1': {'metrics': {k: ev_src['metrics'][k] for k in ('slip_v_p95_cm_s', 'drift_p95_cm', 'pen_frac', 'pen_p95_cm', 'float_frac', 'contact_f1', 'rooth_jump_frac', 'joint_jit_p95_deg_s2') if k in ev_src['metrics']}, 'verdict': ev_src['verdict'], 'failed': ev_src['failed']}, 'robots': {}}
    print('  source G1 (ARDY):', json.dumps(res['source_g1']), flush=True)
    panels = []
    qd_src = qdict_from(a['q'], QN_G); panels.append((G1['renderer'], q_for_renderer(qd_src, G1['renderer']), a['root_R'], root_g_src, f'{key[:28]} | ARDY G1 source'))
    for pack in (G1, H2, GR3):
        name = pack['name']; spec, dof, rest, rob = pack['spec'], pack['dof'], pack['rest'], pack['robot']; root = rob.root
        per_arm = {}
        for arm, c in codes.items():
            r6, pos, conl = g.decode_full(c.to(dev), spec, dof, contact=True)
            _, _, q = project(r6.cpu(), spec, dof, rest=rest); q = q.numpy(); root_R = rot6d_to_matrix(r6[:, root].cpu()).numpy()
            ratio = leg[name] / leg['unitree_g1_29dof']
            root_w = np.stack([root_g_src[:, 0] * ratio, pos[:, root, 0].cpu().numpy() * 100, root_g_src[:, 2] * ratio], 1).astype(np.float64)
            gRf, gpf = fk_from_angles(torch.as_tensor(q), spec, dof, rest=rest, root_R=torch.as_tensor(root_R))
            m = {}
            if name == 'unitree_g1_29dof':
                m['self_rot_deg'] = float(geodesic_deg(gRf, a['gR']).mean())
                for nm_, jn in (('hand', 'left_wrist_yaw_link'), ('foot', 'left_ankle_roll_link')):
                    j = NAMES_G.index(jn); m[f'self_{nm_}_cm'] = float((gpf[:, j] - a['gp'][:, j]).norm(dim=-1).mean())
            thr = 0.04 * rob.standing_root_height_cm / 100.0
            ev = evaluate_clip(rob, q.astype(np.float64), root_R.astype(np.float64), root_w, fps, stance, thr)
            if name == 'unitree_h2': qr = q_for_renderer({n: q[:, k, s] for n, (k, s) in hslot.items()}, pack['renderer'])
            else: qr = q_for_renderer(qdict_from(q, pack['qn']), pack['renderer'])
            lz = floor_scan(pack, qr, root_R, root_w); lift = np.clip(-lz, 0, None)
            m.update(dict(lowest_geom_cm_min=float(lz.min()), geom_pen_frac=float((lz < -0.5).mean()), display_lift_cm_mean=float(lift.mean()), display_lift_cm_max=float(lift.max()),
                          root_h_cm=[float(root_w[:, 1].min()), float(root_w[:, 1].max())], contact_thr_cm=thr * 100))
            m.update({k: ev['metrics'][k] for k in ('slip_v_p95_cm_s', 'drift_p95_cm', 'pen_frac', 'pen_p95_cm', 'float_frac', 'contact_f1', 'onset_abs_med_frames', 'rooth_jump_frac', 'joint_jit_p95_deg_s2', 'joint_v_p95_deg_s') if k in ev['metrics']})
            per_arm[arm] = dict(metrics=m, verdict=ev['verdict'], failed=ev['failed'], not_evaluated=ev['not_evaluated'], q=q, root_R=root_R, root_w=root_w, qr=qr, lift=lift)
            print(f'  {name:18s} [{arm:5s}] verdict {ev["verdict"]:11s} failed {ev["failed"]}  ' + ' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in m.items() if k not in ('root_h_cm',)) + f' root_h {np.round(m["root_h_cm"], 1)}', flush=True)
        res['robots'][name] = {arm: {k: v for k, v in pa.items() if k in ('metrics', 'verdict', 'failed', 'not_evaluated')} for arm, pa in per_arm.items()}
        main = per_arm['con']; np.savez(f'{d}/{name}.npz', q=main['q'], root_R=main['root_R'], root_world_cm=main['root_w'], stance=stance, fps=fps)
        rw = main['root_w'].copy(); rw[:, 1] += main['lift']
        panels.append((pack['renderer'], main['qr'], main['root_R'], rw, f'{name} <- Megatron (contact arm; root lifted {main["lift"].mean():.1f} cm for display)'))
    vids = []
    for i, (r, qr, R, rw, title) in enumerate(panels):
        r.render(qr, R, rw, f'{d}/p{i}', fps=fps, title=title); vids.append(f'{d}/p{i}/review.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + sum([['-i', v] for v in vids], []) + ['-filter_complex', ''.join(f'[{i}:v]' for i in range(4)) + 'hstack=inputs=4[v]', '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', f'{d}/quad.mp4'], check=True)
    rd = iio.get_reader(f'{d}/quad.mp4'); n = rd.count_frames(); idx = np.linspace(0, n - 1, 12).round().astype(int)
    sheet = Image.new('RGB', (1920, 270 * 12), (5, 5, 5))
    for r_, i in enumerate(idx): sheet.paste(Image.fromarray(rd.get_data(int(i))).resize((1920, 270)), (0, r_ * 270))
    rd.close(); sheet.save(f'{d}/grid12.jpg', quality=85)
    results[key] = res; json.dump(res, open(f'{d}/result.json', 'w'), indent=1); print(f'  done {key} in {time.time() - t0:.0f}s', flush=True)
json.dump(results, open(f'{OUT}/results_all.json', 'w'), indent=1); print('ALL DONE')
