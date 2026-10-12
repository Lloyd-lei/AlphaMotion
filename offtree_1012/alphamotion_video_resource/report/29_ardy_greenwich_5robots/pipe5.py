"""ARDY (G1) npz -> Megatron codes -> G1 / H2 / A3 / GR3 / Adam -> exam -> 6-panel video + grid, with stage timing.
Usage: python pipe5.py <ardy_npz_dir> <out_dir> [keys...]"""
import os, sys, json, glob, subprocess, time, statistics, numpy as np, torch, mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ardy_in import load_ardy, SPEC as SPEC_G, DOF as DOF_G, REST as REST_G, QN as QN_G, NAMES as NAMES_G
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import project, fk_from_angles
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d, rot6d_to_matrix, geodesic_deg
from arena.eval.visual import RobotRenderer, resolve_xml
from arena.eval.locomotion import evaluate_clip
from arena.refiner.robots import load_robot, profiles, RobotModel
import imageio.v2 as iio
from PIL import Image
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
KEYS = sys.argv[3:] or sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{IN}/*.npz'))
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
H2_META = '/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json'
dev = 'cuda'; t0 = time.perf_counter(); g = Greenwich.load(model='megatron', device=dev); torch.cuda.synchronize(); LOAD_S = time.perf_counter() - t0
def sync(): torch.cuda.synchronize()

# ---------------- target bodies (decode spec = the descriptor the codec was trained with; renderer slots via the MJCF hinge names)
def pack_bundled(name):
    spec, dof, rest = g.embodiment(name); dof, rest = torch.as_tensor(dof).float(), torch.as_tensor(rest).float()
    xml = resolve_xml(name); sx, dx, _, qn, _ = build_from_mjcf(xml, name + 'x')
    assert list(sx.joint_names) == list(spec.joint_names) and float(np.abs(dx - dof.numpy()).max()) == 0.0, name
    return dict(name=name, spec=spec, dof=dof, rest=rest, qn=qn, robot=load_robot(name), renderer=RobotRenderer(name, width=960, height=540, xml=xml), xml=xml)
def pack_profile(name):                                             # A3: research profile cache == MJCF build (checked 0.0 cm)
    rob = load_robot(name); xml = resolve_xml(name); sx, dx, rx, qn, _ = build_from_mjcf(xml, name)
    assert list(sx.joint_names) == list(rob.spec.joint_names)
    return dict(name=name, spec=rob.spec, dof=torch.as_tensor(dx).float(), rest=torch.as_tensor(rx).float(), qn=qn, robot=rob, renderer=RobotRenderer(name, width=960, height=540, xml=xml), xml=xml)
G1 = pack_bundled('unitree_g1_29dof'); GR3 = pack_bundled('fourier_gr3'); ADAM = pack_bundled('pnd_adam_lite'); A3 = pack_profile('agibot_a3_t3d0')
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
H2 = dict(name='unitree_h2', spec=spec_h, dof=dof_h, rest=rest_h, qn=None, robot=rob_h, renderer=RobotRenderer('unitree_h2', width=960, height=540, xml=H2_XML), xml=H2_XML)
PACKS = [G1, H2, A3, GR3, ADAM]
leg = {p['name']: 0.5 * (profiles()[p['name']]['left']['leg_len_cm'] + profiles()[p['name']]['right']['leg_len_cm']) for p in PACKS}
print('load', round(LOAD_S, 1), 's; leg lengths', {k: round(v, 1) for k, v in leg.items()}, flush=True)
for p in PACKS: g.geom(p['spec'], p['dof'])                          # descriptor + SigLIP LUT warm (one-time)

def q_for_renderer(qdict, r):
    q = np.zeros((len(next(iter(qdict.values()))), len(r.qnames), 3), np.float32)
    for k, names in enumerate(r.qnames):
        for s, n in enumerate(names):
            if n in qdict: q[:, k, s] = qdict[n]
    return q
def qdict_from(q, qn): return {n: q[:, k, s] for k, names in enumerate(qn) for s, n in enumerate(names)}
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
def foot_bodies(pack):
    prof = pack['robot'].profile; return [prof['left']['foot_name'], prof['right']['foot_name']]
def stance_slip(pack, qr, root_R, root_cm, fps):
    """stance = lowest collision point < 1 cm, eroded 2 frames each side; slip = foot body origin xy speed (cm/s)"""
    from scipy.ndimage import minimum_filter1d
    r = pack['renderer']; m, d = r.model, r.data; T = len(qr); feet = foot_bodies(pack); bid = []
    for f in feet:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'rb_' + f); bid.append(i)
    sole = np.zeros((T, 2)); xy = np.zeros((T, 2, 2))
    for t in range(T):
        r._set_frame(qr[t], root_R[t], root_cm[t], None, follow=False)
        for k, b in enumerate(bid):
            zmin = 1e9
            for gi in range(m.ngeom):
                if m.geom_bodyid[gi] != b or not (m.geom_contype[gi] or m.geom_conaffinity[gi]): continue
                c = d.geom_xpos[gi]; s = m.geom_size[gi]; R = d.geom_xmat[gi].reshape(3, 3); ty = m.geom_type[gi]
                if ty == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[gi]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; z = (v @ R.T + c)[:, 2].min()
                elif ty == mujoco.mjtGeom.mjGEOM_BOX: z = c[2] - np.abs(R[2]).dot(s)
                elif ty == mujoco.mjtGeom.mjGEOM_SPHERE: z = c[2] - s[0]
                elif ty == mujoco.mjtGeom.mjGEOM_CAPSULE: z = c[2] - abs(R[2, 2]) * s[1] - s[0]
                else: z = c[2] - m.geom_rbound[gi]
                zmin = min(zmin, z)
            sole[t, k] = zmin * 100 if zmin < 1e8 else np.nan; xy[t, k] = d.xpos[b][:2] * 100
    if np.isnan(sole).all(): return None
    st = minimum_filter1d((sole < 1.0).astype(np.uint8), size=5, axis=0, mode='constant').astype(bool)
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps; mask = st[1:] & st[:-1]
    return dict(slip_p95_cm_s=float(np.percentile(sp[mask], 95)) if mask.any() else 0.0, slip_p50_cm_s=float(np.percentile(sp[mask], 50)) if mask.any() else 0.0, stance_frac=float(st.any(1).mean()), foot_sole_min_cm=float(np.nanmin(sole)))
def contact_f1(pred_stance, src_stance):
    tp = (pred_stance & src_stance).sum(); fp = (pred_stance & ~src_stance).sum(); fn = (~pred_stance & src_stance).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))

results, timing = {}, {'megatron_load_s': LOAD_S, 'gpu': os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip(), 'prompts': {}}
for key in KEYS:
    path = f'{IN}/{key}.npz'; tstart = time.time(); d = f'{OUT}/{key}'; os.makedirs(d, exist_ok=True)
    a = load_ardy(path); T = len(a['q']); fps = a['fps']
    stance = np.zeros((T, 2), bool)
    for c, side in enumerate(['L', 'L', 'R', 'R']): stance[:, 0 if side == 'L' else 1] |= a['contacts'][:, c] > 0.5   # ARDY feet order: L heel, L toe, R heel, R toe
    print(f'\n=== {key}  T={T} fps={fps}  "{a["text"][:70]}..."  stance L/R {stance.mean(0).round(2)}', flush=True)
    gR6 = matrix_to_rot6d(a['gR']).float()
    con = torch.zeros(T, SPEC_G.J)
    for side, col in (('left', 0), ('right', 1)):
        for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, NAMES_G.index(n)] = torch.as_tensor(stance[:, col]).float()
    # ---- timing (c): pose9 + encode, whole clip: warm-up + 3 runs
    tm = {'frames': T, 'fps': fps, 'seconds': T / fps}
    def timed(fn, n=3):
        fn(); sync(); xs = []
        for _ in range(n): sync(); s = time.perf_counter(); r = fn(); sync(); xs.append(time.perf_counter() - s)
        return r, 1000 * statistics.median(xs)
    def enc():
        pose, _ = g.pose9(gR6, SPEC_G, is_global=True); return g.encode(pose, SPEC_G, DOF_G, contact=con.to(dev))
    codes, tm['encode_ms'] = timed(enc)
    root_g_src = a['root_cm'].copy(); root_g_src[:, [0, 2]] -= root_g_src[:1, [0, 2]]
    thr_src = 0.04 * G1['robot'].standing_root_height_cm / 100.0
    ev_src = evaluate_clip(G1['robot'], a['q'].astype(np.float64), a['root_R'].astype(np.float64), root_g_src.astype(np.float64), fps, stance, thr_src)
    res = {'T': T, 'fps': fps, 'text': a['text'], 'ardy_fk_gauge_cm_mean': float(np.mean(list(a['fk_err_cm'].values()))), 'ardy_rot_fit_deg': a['rot_fit_deg'], 'stance_frac': stance.mean(0).tolist(),
           'source_g1': {'metrics': {k: ev_src['metrics'][k] for k in ('slip_v_p95_cm_s', 'drift_p95_cm', 'pen_frac', 'pen_p95_cm', 'float_frac', 'contact_f1', 'rooth_jump_frac', 'joint_jit_p95_deg_s2') if k in ev_src['metrics']}, 'verdict': ev_src['verdict'], 'failed': ev_src['failed']}, 'robots': {}}
    panels = [(G1['renderer'], q_for_renderer(qdict_from(a['q'], QN_G), G1['renderer']), a['root_R'], root_g_src, f'{key} | ARDY G1 source')]
    tm['robots'] = {}
    for pack in PACKS:
        name = pack['name']; spec, dof, rest, rob = pack['spec'], pack['dof'], pack['rest'], pack['robot']; root = rob.root
        # ---- timing (d) decode_full, (e) project, whole clip
        (r6, pos, conl), dec_ms = timed(lambda: g.decode_full(codes.to(dev), spec, dof, contact=True))
        (_, _, q), proj_ms = timed(lambda: project(r6.cpu(), spec, dof, rest=rest))
        # ---- timing (f) streaming: one frame at a time, 100 frames (pose9 + encode + decode + project)
        Tn = min(100, T); xs = []
        for t in range(Tn):
            sync(); s = time.perf_counter()
            pose_t, _ = g.pose9(gR6[t:t + 1], SPEC_G, is_global=True); c_t = g.encode(pose_t, SPEC_G, DOF_G, contact=con[t:t + 1].to(dev))
            r6_t, pos_t = g.decode_full(c_t, spec, dof); project(r6_t.cpu(), spec, dof, rest=rest); sync(); xs.append(time.perf_counter() - s)
        xs = xs[5:]                                                  # first frames include allocator warm-up
        tm['robots'][name] = dict(decode_ms=dec_ms, project_ms=proj_ms, stream_frame_ms_median=1000 * statistics.median(xs), stream_frame_ms_p95=1000 * float(np.percentile(xs, 95)))
        q = q.numpy(); root_R = rot6d_to_matrix(r6[:, root].cpu()).numpy(); ratio = leg[name] / leg['unitree_g1_29dof']
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
        ss = stance_slip(pack, qr, root_R, root_w, fps) or {}
        pred_st = (torch.sigmoid(conl[:, [rob.profile['left']['foot'], rob.profile['right']['foot']]]).cpu().numpy() > 0.5)
        from alphamotion.engine.constraints import _limits
        lo_, hi_, lim_ = _limits(dof, 1.0); qt = torch.as_tensor(q); ex = torch.where(lim_.bool()[None], torch.maximum(lo_[None] - qt, qt - hi_[None]).clamp(min=0), torch.zeros_like(qt))
        m['limit_excess_max_deg'] = float(np.degrees(ex.max())); m['limit_frames_frac'] = float((ex.amax((1, 2)) > 1e-3).float().mean())
        m.update(dict(lowest_geom_cm_min=float(lz.min()), geom_pen_frac=float((lz < -0.5).mean()), display_lift_cm_mean=float(lift.mean()), display_lift_cm_max=float(lift.max()),
                      root_h_cm=[float(root_w[:, 1].min()), float(root_w[:, 1].max())], contact_f1_decoded_vs_ardy=contact_f1(pred_st, stance), **ss))
        m.update({k: ev['metrics'][k] for k in ('slip_v_p95_cm_s', 'drift_p95_cm', 'pen_frac', 'pen_p95_cm', 'float_frac', 'contact_f1', 'onset_abs_med_frames', 'rooth_jump_frac', 'joint_jit_p95_deg_s2', 'joint_v_p95_deg_s', 'limit_excess_p95_deg', 'limit_frac') if k in ev['metrics']})
        res['robots'][name] = dict(metrics=m, verdict=ev['verdict'], failed=ev['failed'], not_evaluated=ev['not_evaluated'], timing=tm['robots'][name])
        np.savez(f'{d}/{name}.npz', q=q, root_R=root_R, root_world_cm=root_w, stance=stance, fps=fps)
        print(f'  {name:18s} verdict {ev["verdict"]:11s} failed {ev["failed"]}  ' + ' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in m.items() if k != 'root_h_cm') + f' root_h {np.round(m["root_h_cm"], 1)} | dec {dec_ms:.1f}ms proj {proj_ms:.0f}ms stream {tm["robots"][name]["stream_frame_ms_median"]:.1f}ms/frame', flush=True)
        rw = root_w.copy(); rw[:, 1] += lift
        panels.append((pack['renderer'], qr, root_R, rw, f'{name} <- Megatron (root lifted {lift.mean():.1f} cm for display)'))
    timing['prompts'][key] = tm
    vids = []
    for i, (r, qr, R, rw, title) in enumerate(panels):
        r.render(qr, R, rw, f'{d}/p{i}', fps=fps, title=title); vids.append(f'{d}/p{i}/review.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + sum([['-i', v] for v in vids], []) + ['-filter_complex', ''.join(f'[{i}:v]' for i in range(len(vids))) + f'hstack=inputs={len(vids)}[v];[v]scale=3840:-2[o]', '-map', '[o]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', f'{d}/six.mp4'], check=True)
    rd = iio.get_reader(f'{d}/six.mp4'); n = rd.count_frames(); idx = np.linspace(0, n - 1, 12).round().astype(int)
    sheet = Image.new('RGB', (1920, 180 * 12), (5, 5, 5))
    for r_, i in enumerate(idx): sheet.paste(Image.fromarray(rd.get_data(int(i))).resize((1920, 180)), (0, r_ * 180))
    rd.close(); sheet.save(f'{d}/grid12.jpg', quality=85)
    results[key] = res; json.dump(res, open(f'{d}/result.json', 'w'), indent=1); print(f'  done {key} in {time.time() - tstart:.0f}s', flush=True)
json.dump(results, open(f'{OUT}/results_all.json', 'w'), indent=1); json.dump(timing, open(f'{OUT}/greenwich_timing.json', 'w'), indent=1); print('ALL DONE')
