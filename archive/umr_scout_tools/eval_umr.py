"""Point-cloud contact / penetration / slip evaluation of a UMR retarget result.
python eval_umr.py --result R.npz --src-slots S.npz [--smplx-data lafan.npz] --out out.json
All contacts are decided from surface point clouds (robot: area-sampled visual-mesh points; human: UMR's 4096 exterior
source slots, scaled to robot size, ground z=0), never from joint-centre heights.
"""
import argparse, json, sys, numpy as np, mujoco, trimesh
sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
from mujoco_geom_surface import surface_geom_ids, geom_local_mesh

ap = argparse.ArgumentParser()
ap.add_argument('--result', required=True); ap.add_argument('--src-slots', default='')
ap.add_argument('--n-robot-pts', type=int, default=8192); ap.add_argument('--thr', type=float, default=0.01)
ap.add_argument('--out', required=True); ap.add_argument('--save-labels', default='')
a = ap.parse_args()

R = np.load(a.result, allow_pickle=True)
qpos = R['qpos'].astype(np.float64); fps = float(R['fps'][0]); dt = 1.0 / fps
m = mujoco.MjModel.from_xml_path(str(R['robot_xml'])); d = mujoco.MjData(m)
rng = np.random.default_rng(0)
# ---- robot surface point cloud (area-weighted samples, fixed per geom-local frame)
gids = [int(g) for g in surface_geom_ids(m, 'auto')]
meshes = []
for g in gids:
    v, f = geom_local_mesh(m, g); tm = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False); meshes.append(tm)
areas = np.array([tm.area for tm in meshes]); cnt = np.maximum(1, np.round(a.n_robot_pts * areas / areas.sum())).astype(int)
loc, gof = [], []
for g, tm, c in zip(gids, meshes, cnt):
    p, _ = trimesh.sample.sample_surface(tm, int(c), seed=int(rng.integers(1 << 30))); loc.append(p); gof.append(np.full(len(p), g))
loc = np.concatenate(loc); gof = np.concatenate(gof)
body_of_pt = m.geom_bodyid[gof]
bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(m.nbody)]
def side_feet(side):
    keys = ('ankle', 'foot', 'toe', 'sole')
    return [b for b in range(m.nbody) if bname[b].lower().startswith(side) and any(k in bname[b].lower() for k in keys)]
feet = {'L': side_feet('left'), 'R': side_feet('right')}
foot_mask = {s: np.isin(body_of_pt, feet[s]) for s in feet}
# collision geoms for self-penetration
coll = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0]
# pairs already overlapping (>1cm) at the model rest pose (qpos0) are MJCF artifacts -> excluded
mujoco.mj_resetData(m, d); d.qpos[:] = m.qpos0; d.qpos[2] = 2.0; mujoco.mj_forward(m, d)
rest_pairs = set()
for i in range(d.ncon):
    c = d.contact[i]
    if c.dist < -0.005: rest_pairs.add((min(c.geom1, c.geom2), max(c.geom1, c.geom2)))
T = len(qpos); P = np.zeros((T, len(loc), 3), np.float32)
selfpen = np.zeros(T); selfpairs = []
for t in range(T):
    d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
    P[t] = (np.einsum('nij,nj->ni', d.geom_xmat[gof].reshape(-1, 3, 3), loc) + d.geom_xpos[gof]).astype(np.float32)
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in coll and c.geom2 in coll and m.geom_bodyid[c.geom1] != 0 and m.geom_bodyid[c.geom2] != 0 and (min(c.geom1, c.geom2), max(c.geom1, c.geom2)) not in rest_pairs:
            worst = min(worst, c.dist)
            if c.dist < -0.01: selfpairs.append((bname[m.geom_bodyid[c.geom1]], bname[m.geom_bodyid[c.geom2]]))
    selfpen[t] = -worst

def cloud_stats(P, thr, tag):
    z = P[..., 2]; minz = z.min(1)
    con = z < thr
    both = con[1:] & con[:-1]
    v = np.linalg.norm(P[1:, :, :2] - P[:-1, :, :2], axis=-1) / dt
    sp = v[both]
    out = {f'{tag}_frames': int(len(P)),
           f'{tag}_pen_depth_cm_p50': float(np.percentile(np.maximum(0, -minz), 50) * 100),
           f'{tag}_pen_depth_cm_p95': float(np.percentile(np.maximum(0, -minz), 95) * 100),
           f'{tag}_pen_depth_cm_max': float(np.maximum(0, -minz).max() * 100),
           f'{tag}_frac_frames_pen_gt_1cm': float((minz < -0.01).mean()),
           f'{tag}_frac_frames_airborne_gt_2cm': float((minz > 0.02).mean()),
           f'{tag}_frac_frames_any_contact': float(con.any(1).mean()),
           f'{tag}_contact_pt_slip_cm_s_p50': float(np.percentile(sp, 50) * 100) if sp.size else None,
           f'{tag}_contact_pt_slip_cm_s_p95': float(np.percentile(sp, 95) * 100) if sp.size else None,
           f'{tag}_frac_contact_pts_slip_gt_10cm_s': float((sp > 0.10).mean()) if sp.size else None,
           f'{tag}_frac_contact_pts_slip_gt_25cm_s': float((sp > 0.25).mean()) if sp.size else None}
    return out, con, minz

res = {'result': a.result, 'robot_xml': str(R['robot_xml']), 'fps': fps, 'contact_thr_m': a.thr,
       'robot_pts': int(len(loc)), 'feet_bodies': {s: [bname[b] for b in feet[s]] for s in feet}}
o, rcon, rminz = cloud_stats(P, a.thr, 'robot'); res.update(o)
res['robot_selfpen_cm_p95'] = float(np.percentile(selfpen, 95) * 100); res['robot_selfpen_cm_max'] = float(selfpen.max() * 100)
res['robot_frac_frames_selfpen_gt_1cm'] = float((selfpen > 0.01).mean())
from collections import Counter
res['robot_selfpen_top_pairs'] = Counter(selfpairs).most_common(5)
res['rest_overlap_pairs_excluded'] = [(bname[m.geom_bodyid[g1]], bname[m.geom_bodyid[g2]]) for g1, g2 in rest_pairs]
rfoot = {s: rcon[:, foot_mask[s]].any(1) for s in feet}
res['robot_frac_contact_pts_on_feet'] = float((rcon & (foot_mask['L'] | foot_mask['R'])[None]).sum() / max(1, rcon.sum()))
if a.src_slots:
    S = np.load(a.src_slots)['source_slots'][R['frame_ids']] if 'frame_ids' in R.files else np.load(a.src_slots)['source_slots']
    o, hcon, hminz = cloud_stats(S, a.thr, 'human'); res.update(o)
    # contact-frame agreement: human any-contact vs robot any-contact
    h_any, r_any = hcon.any(1), rcon.any(1)
    res['agree_any_contact'] = float((h_any == r_any).mean())
    res['human_contact_robot_airborne_frac'] = float((h_any & ~r_any).mean())
    # per-foot: assign human contact points to the robot foot with nearest contact point centroid (same frame)
    agree = []
    for t in range(len(S)):
        hc = S[t][hcon[t]]
        hs = {}
        for s in feet:
            fc = P[t][foot_mask[s]]
            hs[s] = fc.mean(0)
        hl = hr = False
        if len(hc):
            dl = np.linalg.norm(hc[:, :2] - hs['L'][:2], axis=1); dr = np.linalg.norm(hc[:, :2] - hs['R'][:2], axis=1)
            near = np.minimum(dl, dr) < 0.25
            hl = bool(((dl < dr) & near).any()); hr = bool(((dr <= dl) & near).any())
        agree.append([hl == rfoot['L'][t], hr == rfoot['R'][t], hl, hr])
    agree = np.array(agree)
    res['per_foot_contact_agreement_L'] = float(agree[:, 0].mean()); res['per_foot_contact_agreement_R'] = float(agree[:, 1].mean())
    res['human_foot_contact_frac_L'] = float(agree[:, 2].mean()); res['robot_foot_contact_frac_L'] = float(rfoot['L'].mean())
    res['human_foot_contact_frac_R'] = float(agree[:, 3].mean()); res['robot_foot_contact_frac_R'] = float(rfoot['R'].mean())
if a.save_labels:
    np.savez_compressed(a.save_labels, robot_foot_contact_L=rfoot['L'], robot_foot_contact_R=rfoot['R'], robot_min_z=rminz,
                        robot_selfpen_m=selfpen.astype(np.float32))
json.dump(res, open(a.out, 'w'), indent=1, default=str)
for k, v in res.items():
    if not isinstance(v, (list, dict)): print(f'{k}: {v}')
print('robot_selfpen_top_pairs', res['robot_selfpen_top_pairs'])
