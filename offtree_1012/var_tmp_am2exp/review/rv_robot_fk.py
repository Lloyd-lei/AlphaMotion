"""Reviewer: independent robot FK check.  (1) random q: posed surface points vs MuJoCo visual meshes posed by
mj_forward (point-to-mesh distance over ALL visual geoms, not the stored point_geom). (2) real data: payload
global_rot6/position_cm rows matched to xwbc extracted MuJoCo states, compare payload R vs adapter R and the posed
surface vs MuJoCo meshes using the PAYLOAD (R,P) -- i.e. what training actually feeds pose_surface."""
import sys, json, os
import numpy as np, torch, trimesh, mujoco
CODE = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1'
sys.path.insert(0, CODE)
from assets import robot_binding as RB, surface_v2 as S, _env as E
AX = RB.AX
PAY = E.PILOT_XWBC / 'payloads'

def rot6_to_mat(r6):
    a, b = r6[..., 0:3], r6[..., 3:6]
    # repository convention: first two matrix COLUMNS
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)

def world_mesh(kin, d, root):
    geoms, _, _ = RB.visual_geoms(kin)
    Vs, Fs, off = [], [], 0
    m = kin.model
    for g in geoms:
        V, F = RB.mesh_arrays(kin, g)
        Vw = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        Vw = (Vw - root) @ AX.T * kin.unit
        Vs.append(Vw); Fs.append(F + off); off += len(V)
    return trimesh.Trimesh(np.concatenate(Vs), np.concatenate(Fs), process=False)

def dist_to_mesh(mesh, pts):
    _, d, _ = trimesh.proximity.closest_point(mesh, pts)
    return d

def pose(asset, R, P):
    with torch.no_grad():
        p, n = S.pose_surface(asset, torch.as_tensor(R[None], dtype=torch.float32), torch.as_tensor(P[None], dtype=torch.float32))
    return p[0].numpy().astype(np.float64), n[0].numpy().astype(np.float64)

def random_q(m, rng):
    q = m.qpos0.copy()
    for j in range(m.njnt):
        adr = int(m.jnt_qposadr[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-1, 1)
            q[adr] = rng.uniform(lo, hi)
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qq = rng.normal(size=4); q[adr + 3:adr + 7] = qq / np.linalg.norm(qq)
    return q

def main(body, seed=123):
    kin = RB.load_kinematics(body)
    asset = S.load_surface(body)
    trust = asset.extra['point_data_residual_p95_cm'] <= 3 if 'point_data_residual_p95_cm' in asset.extra else np.ones(4096, bool)
    rigid = asset.extra['point_rigid'].astype(bool)
    m = kin.model; d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    out = {'body': body, 'random_q': []}
    for i in range(3):
        q = random_q(m, rng)
        R, Pfk, Pmj, _, _ = kin.forward(q)
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        root = d.xpos[kin.keep[0]].copy()
        mesh = world_mesh(kin, d, root)
        p, n = pose(asset, R, Pfk)
        dist = dist_to_mesh(mesh, p)
        # normal check: outward = moving along normal 1cm should not get closer to mesh interior ... use sign via
        # agreement with nearest face normal is winding dependent; skip, report point->mesh only.
        out['random_q'].append(dict(fk_vs_mj_joint_cm=float(np.abs(Pfk - Pmj).max()),
            dist_all_p50_mm=float(np.median(dist) * 10), dist_all_p99_mm=float(np.percentile(dist, 99) * 10),
            dist_all_max_cm=float(dist.max()), dist_rigid_max_mm=float(dist[rigid].max() * 10) if rigid.any() else None,
            dist_trusted_max_cm=float(dist[trust].max()), n_gt_1cm=int((dist > 1).sum()), n_gt_3cm=int((dist > 3).sum())))
    # ---- real data rows ----
    zp = np.load(PAY / f'{body}.npz')
    pos = zp['position_cm'].astype(np.float64); R6 = zp['global_rot6'].astype(np.float64)
    J = pos.shape[1]
    names = E.read_json('/var/tmp/alphamotion/xwbc_joint_names.json').get(body, {}).get('joint_names')
    res = []
    for clip in ('walk1_subject1', 'fallAndGetUp1_subject1', 'ground1_subject1'):
        f = E.XWBC_EXTRACTED / body / f'{clip}.npz'
        if not f.exists() or not names:
            continue
        z = np.load(f)
        jn = [str(x) for x in z['joint_names']] if 'joint_names' in z.files else list(names)
        qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) >= 0 else -1 for nm in jn]
        T = len(z['joint_pos'])
        frames = np.arange(0, T, 3)
        Pm_all, Rk_all, qs = [], [], []
        for t in frames:
            q = m.qpos0.copy()
            if m.njnt and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
                q[0:3] = z['body_pos_w'][t, 0]; q[3:7] = z['body_quat_w'][t, 0]
            for i, a in enumerate(qadr):
                if a >= 0: q[a] = z['joint_pos'][t, i]
            R, Pfk, Pmj, _, _ = kin.forward(q)
            Pm_all.append(Pfk); Rk_all.append(R); qs.append(q)
        Pm_all = np.stack(Pm_all)
        # nearest payload row per frame (positions, root-relative)
        A = Pm_all.reshape(len(frames), -1); Bp = pos.reshape(len(pos), -1)
        best = []
        for k in range(0, len(A), 256):
            dd = ((A[k:k + 256, None] - Bp[None]) ** 2).mean(-1)
            best.append(np.stack([dd.argmin(1), np.sqrt(dd.min(1))], 1))
        best = np.concatenate(best)
        ok = best[:, 1] < 0.01
        for fi in np.flatnonzero(ok)[:40]:
            r = int(best[fi, 0])
            Rd = rot6_to_mat(R6[r]); Rk = Rk_all[fi]
            ang = np.degrees(np.arccos(np.clip((np.einsum('jab,jab->j', Rd, Rk) - 1) / 2, -1, 1)))
            # pose surface with PAYLOAD (R,P) and compare to MuJoCo meshes
            q = qs[fi]; d.qpos[:] = q; mujoco.mj_forward(m, d); root = d.xpos[kin.keep[0]].copy()
            mesh = world_mesh(kin, d, root)
            p, _ = pose(asset, Rd, pos[r])
            dist = dist_to_mesh(mesh, p)
            res.append(dict(clip=clip, frame=int(frames[fi]), row=r, pos_rms_cm=float(best[fi, 1]), rot_err_max_deg=float(ang.max()),
                            rot_err_argmax=kin.names[int(ang.argmax())], dist_p50_mm=float(np.median(dist) * 10),
                            dist_trusted_max_cm=float(dist[trust].max()), dist_max_cm=float(dist.max())))
        out.setdefault('data_match', {})[clip] = dict(frames=len(frames), matched=int(ok.sum()), best_rms_p50=float(np.median(best[:, 1])))
    out['data_rows'] = res
    if res:
        out['data_summary'] = dict(n=len(res), rot_err_max_deg=max(r['rot_err_max_deg'] for r in res),
                                   dist_trusted_max_cm=max(r['dist_trusted_max_cm'] for r in res),
                                   dist_p50_mm_med=float(np.median([r['dist_p50_mm'] for r in res])))
    return out

if __name__ == '__main__':
    body = sys.argv[1]
    o = main(body)
    json.dump(o, open(f'/var/tmp/alphamotion/am2exp/review/fk_{body}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in o.items() if k != 'data_rows'}, indent=1))
