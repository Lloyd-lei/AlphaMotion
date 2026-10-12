"""Object-contact check for an HSI/HOI UMR result (sphere-like object: signed distance from fitted centre/radius)."""
import sys, json, numpy as np, mujoco, trimesh
sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts'); sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR')
import retarget_smpl_to_humanoid_surface_vector as rs
from mujoco_geom_surface import surface_geom_ids, geom_local_mesh
from pathlib import Path
res_p, src_p, xml_p, out_p = sys.argv[1:5]
R = np.load(res_p, allow_pickle=True); S = np.load(src_p)['source_slots']; sc = float(R['smpl_scale'][0]); fids = R['frame_ids']
xml = Path(xml_p); prop = rs.resolve_prop_trajectory(xml)
mot = rs.read_prop_motion(prop, fids, 'y', True, False, 0.0, 0.0, sc)
loc = rs.object_points_from_mjcf(xml, 2048, 0, sc)
c0 = loc.mean(0); rad = np.linalg.norm(loc - c0, axis=1).mean()
m = mujoco.MjModel.from_xml_path(str(R['robot_xml'])); d = mujoco.MjData(m)
gids = [int(g) for g in surface_geom_ids(m, 'auto')]; L, G = [], []
for g in gids:
    v, f = geom_local_mesh(m, g); tm = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False)
    p, _ = trimesh.sample.sample_surface(tm, max(1, int(8192 * tm.area / 1.2)), seed=0); L.append(p); G.append(np.full(len(p), g))
L = np.concatenate(L); G = np.concatenate(G); bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
hd, rd, rb = [], [], []
for t in range(len(fids)):
    Rm = np.asarray(mot['rot_mats'][t]); pos = np.asarray(mot['positions'][t]); c = c0 @ Rm.T + pos
    hd.append(np.linalg.norm(S[t] - c, axis=1).min() - rad)
    d.qpos[:] = R['qpos'][t]; mujoco.mj_kinematics(m, d)
    P = np.einsum('nij,nj->ni', d.geom_xmat[G].reshape(-1, 3, 3), L) + d.geom_xpos[G]
    dist = np.linalg.norm(P - c, axis=1) - rad; i = dist.argmin(); rd.append(dist[i]); rb.append(bn[m.geom_bodyid[G[i]]])
hd, rd = np.array(hd), np.array(rd)
out = {'ball_radius_scaled_m': float(rad), 'frames': int(len(fids))}
for thr in (0.01, 0.02, 0.05):
    hc, rc = hd < thr, rd < thr
    out[f'human_contact_frames_lt{int(thr*100)}cm'] = int(hc.sum()); out[f'robot_contact_frames_lt{int(thr*100)}cm'] = int(rc.sum())
    out[f'contact_iou_lt{int(thr*100)}cm'] = float((hc & rc).sum() / max(1, (hc | rc).sum()))
out['human_min_signed_dist_cm'] = float(hd.min() * 100); out['robot_min_signed_dist_cm'] = float(rd.min() * 100)
out['robot_frames_penetrating_gt_1cm'] = int((rd < -0.01).sum())
t0 = int(hd.argmin()); out['closest_human_frame'] = t0; out['robot_dist_at_that_frame_cm'] = float(rd[t0] * 100); out['robot_body_at_that_frame'] = rb[t0]
from collections import Counter
out['robot_bodies_in_contact_lt2cm'] = Counter([b for b, x in zip(rb, rd) if x < 0.02]).most_common(4)
json.dump(out, open(out_p, 'w'), indent=1); print(json.dumps(out, indent=1))
