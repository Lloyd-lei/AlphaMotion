"""CollisionGround (analytic / support-vertex lowest points) vs brute force over dense shape samples + mesh vertices."""
import sys, numpy as np, torch, mujoco
from surface_retargeting.gpu.kin import KinTree
from surface_retargeting.gpu.geometry import CollisionGround, ShapeSet, robot_collision_geoms, mesh_verts
z = np.load(sys.argv[1]); m = mujoco.MjModel.from_binary_path(str(z['mjb']))
kt = KinTree(m, 'cuda', torch.float32); cg = CollisionGround(kt)
Q = torch.tensor(z['q_cpu'][::50], device='cuda')
K = kt.fk(Q); fl = torch.zeros(len(Q), device='cuda')
c, P = cg.lowest(K, fl)
gids = robot_collision_geoms(m)
ss = ShapeSet(m, gids, 'cuda', n_samples=4000)
gp, gR = kt.geom_pose(K, torch.as_tensor(gids, device='cuda'))
S = ss.samples
bf = ((gR[:, :, None].float() @ S[None, ..., None])[..., 0] + gp[:, :, None].float())[..., 2].amin(-1)
for i, g in enumerate(gids):
    if m.geom_type[g] == 7:
        V, _ = mesh_verts(m, g); Vt = torch.tensor(V, device='cuda', dtype=torch.float32)
        bf[:, i] = torch.minimum(bf[:, i], ((gR[:, i, None].float() @ Vt[None, ..., None])[..., 0] + gp[:, i, None].float())[..., 2].amin(-1))
d = (c - bf) * 1000
print(sys.argv[1].split('/')[-1], 'analytic - bruteforce mm: max %.3f min %.3f (negative = analytic lower, i.e. conservative)' % (float(d.max()), float(d.min())))
