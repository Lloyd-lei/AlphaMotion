"""GPU self-collision (point-vs-SDF) vs MuJoCo mj_geomDistance on UMR output poses."""
import sys, time, numpy as np, torch, mujoco
sys.path.insert(0, '/home/arenalabs/umr_gpu')
from surface_retargeting.gpu.kin import KinTree
from surface_retargeting.gpu.geometry import SelfCollision
z = np.load(sys.argv[1]); m = mujoco.MjModel.from_binary_path(str(z['mjb'])); d = mujoco.MjData(m)
kt = KinTree(m, 'cuda', torch.float32)
t0 = time.time(); sc = SelfCollision(kt); print('build', round(time.time() - t0, 1), 's  pairs', len(sc.pairs_geom), 'mesh hull gap mm p50/max', np.round(np.median(sc.shapes.gap[sc.shapes.gap > 0]) * 1e3, 3) if (sc.shapes.gap > 0).any() else 0, np.round(sc.shapes.gap.max() * 1e3, 3))
Q = z['q_cpu'][::5]
K = kt.fk(torch.tensor(Q, device='cuda'))
r = sc.query(K, 0.03)
b = r['b'].cpu().numpy(); pr = r['pair'].cpu().numpy(); dg = r['d'].cpu().numpy()
sel = dg < 0.03
err = []; pen_cpu = []; pen_gpu = []
ft = np.zeros(6)
for i in np.flatnonzero(sel):
    d.qpos[:] = Q[b[i]]; mujoco.mj_kinematics(m, d)
    g1, g2 = sc.pairs_geom[pr[i]]
    dc = mujoco.mj_geomDistance(m, d, int(g1), int(g2), 0.05, ft)
    err.append(dg[i] - dc); pen_cpu.append(dc); pen_gpu.append(dg[i])
err = np.array(err); pc = np.array(pen_cpu); pg = np.array(pen_gpu)
print('entries within 3cm', len(err), 'gpu-mj err mm: p5 %.2f p50 %.2f p95 %.2f min %.2f max %.2f' % tuple(np.percentile(err * 1e3, [5, 50, 95]).tolist() + [err.min() * 1e3, err.max() * 1e3]))
near = pc < 0.005
print('near (mj<5mm):', near.sum(), 'err mm p50 %.2f p95 %.2f max %.2f min %.2f' % (np.median(err[near])*1e3, np.percentile(err[near],95)*1e3, err[near].max()*1e3, err[near].min()*1e3))
pen = pc < -0.0005
print('mj penetrating(<-0.5mm):', pen.sum(), ' detected by gpu (<-0.5mm):', (pg[pen] < -0.0005).sum(), ' gpu false alarms:', ((pg < -0.0005) & ~pen).sum())
# full check of pairs NOT reported by gpu broadphase (should be > 3 cm)
miss = 0
for bi in range(0, len(Q), 37):
    d.qpos[:] = Q[bi]; mujoco.mj_kinematics(m, d)
    rep = set(pr[b == bi].tolist())
    for pi, (g1, g2) in enumerate(sc.pairs_geom):
        if pi in rep: continue
        dd = mujoco.mj_geomDistance(m, d, int(g1), int(g2), 0.05, ft)
        if dd < 0.02:
            miss += 1; print('   missed', bi, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g1)), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g2)), 'mj', round(dd*1000, 2), 'mm  |fromto|', round(float(np.linalg.norm(ft[:3]-ft[3:]))*1000, 2), 'mm')
print('pairs < 2cm missed by broadphase:', miss)
