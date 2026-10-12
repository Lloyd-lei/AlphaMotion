"""KinTree vs MuJoCo: FK, geom poses, point Jacobians, integratePos/differentiatePos (fp64)."""
import sys, json
import mujoco, numpy as np, torch
from surface_retargeting.gpu.kin import KinTree

def run(xml):
    m = mujoco.MjModel.from_xml_path(xml) if xml.endswith('.xml') else mujoco.MjModel.from_binary_path(xml)
    d = mujoco.MjData(m)
    kt = KinTree(m, 'cuda', torch.float64)
    rng = np.random.default_rng(0)
    B = 8
    Q = np.zeros((B, m.nq))
    for b in range(B):
        q = m.qpos0.copy(); q[:3] = rng.normal(size=3); qq = rng.normal(size=4); q[3:7] = qq / np.linalg.norm(qq)
        q[7:] = rng.uniform(-1, 1, size=m.nq - 7); Q[b] = q
    K = kt.fk(torch.tensor(Q, device='cuda'))
    errs = dict(xpos=0, xmat=0, gpos=0, jacp=0, jacr=0, integ=0, diff=0)
    gids = torch.arange(m.ngeom, device='cuda')
    gp, gR = kt.geom_pose(K, gids)
    for b in range(B):
        d.qpos[:] = Q[b]; mujoco.mj_forward(m, d)
        errs['xpos'] = max(errs['xpos'], np.abs(K['xpos'][b].cpu().numpy() - d.xpos).max())
        errs['xmat'] = max(errs['xmat'], np.abs(K['xmat'][b].cpu().numpy().reshape(-1, 9) - d.xmat).max())
        errs['gpos'] = max(errs['gpos'], np.abs(gp[b].cpu().numpy() - d.geom_xpos).max(), np.abs(gR[b].cpu().numpy().reshape(-1, 9) - d.geom_xmat).max())
        bodies = np.arange(1, m.nbody)
        P = d.xpos[bodies] + rng.normal(size=(len(bodies), 3)) * 0.1
        Jp, Jr = kt.jac(K, torch.tensor(P, device='cuda')[None].expand(B, -1, -1), torch.tensor(bodies, device='cuda'), rot=True)
        for i, bb in enumerate(bodies):
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, jr, P[i], int(bb))
            errs['jacp'] = max(errs['jacp'], np.abs(Jp[b, i].cpu().numpy() - jp).max())
            errs['jacr'] = max(errs['jacr'], np.abs(Jr[b, i].cpu().numpy() - jr).max())
    dq = rng.normal(size=(B, m.nv)) * 0.3
    qi = kt.integrate(torch.tensor(Q, device='cuda'), torch.tensor(dq, device='cuda')).cpu().numpy()
    for b in range(B):
        q = Q[b].copy(); mujoco.mj_integratePos(m, q, dq[b], 1.0)
        errs['integ'] = max(errs['integ'], np.abs(q - qi[b]).max())
    dd = kt.difference(torch.tensor(Q, device='cuda'), torch.tensor(qi, device='cuda')).cpu().numpy()
    errs['diff'] = np.abs(dd - dq).max()
    return {k: float(v) for k, v in errs.items()}

if __name__ == '__main__':
    for x in sys.argv[1:]:
        print(x.split('/')[-1], json.dumps(run(x)))
