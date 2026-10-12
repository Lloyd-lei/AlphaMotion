"""Label-free decoder floor gauge: held-out AMASS human clips -> Megatron -> each embodiment -> project -> vendor MJCF lowest collision point.
Also 'as-trained' gauge: cache-spec FK foot joint + labeler sole offset, i.e. what the decoder was asked to hit."""
import os; os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np, torch, json, mujoco, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sole_audit import lowest_point_of_body_tree
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import build_from_cache, build_from_mjcf
from alphamotion.engine.constraints import project, fk_from_angles
from alphamotion.engine.spatial import build_global
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from arena.eval.visual import resolve_xml
AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
HELD = '/var/tmp/alphamotion/topo_exp/cache_amass_heldout'; CACHE = '/var/tmp/alphamotion/topo_exp/cache_lafan_v3'
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
g = Greenwich.load(model='megatron', device='cuda')
seqs = json.load(open(f'{HELD}/sequences.json'))
pick = [s for s in seqs if re.search(r'walk|stand|jog|run', s['name'], re.I) and s['len'] >= 60][:N_CLIPS]
print('held-out clips', len(pick), [s['name'][:40] for s in pick[:5]], flush=True)
spec_h, dof_h, rest_h, _, _ = build_from_cache(HELD, 'human_smpl'); l6 = torch.load(f'{HELD}/human_smpl_paired_local6d.pt', mmap=True)
man = json.load(open(f'{CACHE}/manifest.json'))['embodiments']
sole_json = json.load(open('/home/arenalabs/tmp/sole_offsets3.json'))
out = {}
# encode all clips once
codes = []
for s in pick:
    gr6, gp, reach = build_global(l6[s['start']:s['start'] + s['len']].float(), spec_h, 'cpu')
    pose = torch.cat([gr6, gp / reach], -1).cuda(); codes.append(g.encode(pose, spec_h, dof_h).cpu())
for emb in man:
    if emb in ('pal_talos', 'engineai_pm01', 'human_smpl'): continue
    try:
        spec, dof, rest = g.embodiment(emb); dof_t, rest_t = torch.as_tensor(dof).float(), torch.as_tensor(rest).float()
        xml = H2_XML if emb == 'unitree_h2' else resolve_xml(emb); m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
        sv, dv, rv, qn, _ = build_from_mjcf(xml, emb)
        hdr = json.loads(str(np.load(f'{CACHE}/{emb}_contact.npz', allow_pickle=True)['header']))
    except Exception as e: out[emb] = {'err': str(e)[:150]}; print(emb, 'ERR', str(e)[:150], flush=True); continue
    # slot mapping cache-spec -> vendor build: by names when equal, else by rest-offset nearest match (H2 generic names)
    if list(spec.joint_names) == list(sv.joint_names): node_map = list(range(spec.J))
    else:
        oc, ov = np.asarray(spec.rest_offsets), np.asarray(sv.rest_offsets); node_map = [int(np.argmin(np.linalg.norm(ov - oc[i], axis=1))) for i in range(spec.J)]
    slot2joint = {(node_map[k], s): n for k, names in enumerate(qn) for s, n in enumerate(names)} if list(spec.joint_names) == list(sv.joint_names) else {}
    if not slot2joint:  # H2: vendor qnames indexed by vendor node; map vendor node -> cache node
        inv = {v: k for k, v in enumerate(node_map)}
        slot2joint = {(inv[k], s): n for k, names in enumerate(qn) for s, n in enumerate(names) if k in inv}
    jadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in slot2joint.values() if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0}
    foot_bodies = []
    for n in (hdr['sole_joint_names'] or [hdr['j_floor_name'], hdr['j_low_name']]):
        k = list(spec.joint_names).index(n) if n in spec.joint_names else None
        vn = sv.joint_names[node_map[k]] if k is not None else n; bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, vn)
        if bid >= 0: foot_bodies.append(bid)
    # labeler sole offsets in the cache spec (ardy frame, m) for the 'as-trained' gauge
    # same rule as contact_label.sole_offsets(): per side, walk the MJCF ancestor chain and take the first body present in the spec
    ALIAS = {'unitree_g1_29dof': 'unitree_g1', 'unitree_g1_23dof': 'unitree_g1'}
    ent = sole_json.get(ALIAS.get(emb, emb)) or {}; soff = {}
    for side in ('a', 'b'):
        for c in (ent.get(side) or []):
            if c['body'] in spec.joint_names: soff[list(spec.joint_names).index(c['body'])] = np.asarray(c['off_ardy'], np.float32); break
    lows, trained, roots = [], [], []
    for c in codes:
        r6, pos = g.decode_full(c.cuda(), spec, dof_t); r6p, gp, q = project(r6.cpu(), spec, dof_t, rest=rest_t)
        R = rot6d_to_matrix(r6[:, 0].cpu()).numpy(); root_y = pos[:, 0, 0].cpu().numpy() * 100; q = q.numpy(); T = len(q)
        gR, gpos = fk_from_angles(torch.as_tensor(q), spec, dof_t, rest=rest_t, root_R=torch.as_tensor(R)); gR, gpos = gR.numpy(), gpos.numpy()
        # as-trained: lowest (foot joint + labeler sole offset) height, cm
        if soff:
            ys = [gpos[:, k, 1] + root_y + (gR[:, k] @ (v * 100))[:, 1] for k, v in soff.items()]; trained.append(np.min(np.stack(ys, 1), 1))
        else:
            fj = [list(spec.joint_names).index(n) for n in [hdr['j_floor_name'], hdr['j_low_name']] if n in spec.joint_names]; trained.append(np.min(gpos[:, fj, 1], 1) + root_y)
        # vendor MJCF lowest collision point of the foot bodies
        low = np.zeros(T)
        for t in range(T):
            d.qpos[:] = m.qpos0; d.qpos[:3] = np.array([0, 0, root_y[t] / 100]); Rm = AX.T @ R[t] @ AX
            from scipy.spatial.transform import Rotation as Rot
            d.qpos[3:7] = Rot.from_matrix(Rm).as_quat(scalar_first=True)
            for (k, s), n in slot2joint.items():
                if n in jadr: d.qpos[jadr[n]] = q[t, k, s]
            mujoco.mj_kinematics(m, d); low[t] = min(lowest_point_of_body_tree(m, d, b) for b in foot_bodies) * 100
        lows.append(low); roots.append(root_y)
    lows, trained = np.concatenate(lows), np.concatenate(trained)
    out[emb] = {'n_frames': int(len(lows)), 'decoder_lowest_cm': {'p10': float(np.percentile(lows, 10)), 'p50': float(np.percentile(lows, 50)), 'p90': float(np.percentile(lows, 90))},
                'as_trained_lowest_cm_p50': float(np.percentile(trained, 50)), 'pen_frac_lt_-0.5': float((lows < -0.5).mean()), 'float_frac_gt_2': float((lows > 2).mean()),
                'root_y_cm_p50': float(np.median(np.concatenate(roots))), 'foot_bodies': [m.body(b).name for b in foot_bodies], 'label_sole_drop_cm': hdr['sole_drop_cm']}
    print(f"{emb:18s} decoder lowest p50 {out[emb]['decoder_lowest_cm']['p50']:6.1f} (p10 {out[emb]['decoder_lowest_cm']['p10']:6.1f} p90 {out[emb]['decoder_lowest_cm']['p90']:6.1f})  as-trained p50 {out[emb]['as_trained_lowest_cm_p50']:6.1f}  pen {out[emb]['pen_frac_lt_-0.5']*100:5.1f}%  float {out[emb]['float_frac_gt_2']*100:5.1f}%  feet {out[emb]['foot_bodies']}", flush=True)
json.dump(out, open('decoder_gauge.json', 'w'), indent=1); print('done')
