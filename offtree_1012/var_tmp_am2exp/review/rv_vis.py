"""Per-point UMR-style first-hit visibility at qpos0 on the MuJoCo-posed visual mesh (closest-point projected)."""
import sys, json
import numpy as np, trimesh, mujoco
sys.path.insert(0, '/var/tmp/alphamotion/am2exp/review')
from rv_robot_fk import RB, S, world_mesh, pose
from rv_exterior import dirs26
from trimesh.ray.ray_pyembree import RayMeshIntersector
res = {}
for body in sys.argv[1:]:
    kin = RB.load_kinematics(body); asset = S.load_surface(body)
    m = kin.model; d = mujoco.MjData(m)
    q = kin.open_slides_qpos() if body == 'relay_franka_full_v2' else m.qpos0.copy()
    R, P, _, _, _ = kin.forward(q); d.qpos[:] = q; mujoco.mj_forward(m, d)
    mesh = world_mesh(kin, d, d.xpos[kin.keep[0]].copy())
    p, n = pose(asset, R, P)
    cp, dist, tri = trimesh.proximity.closest_point(mesh, p)
    it = RayMeshIntersector(mesh)
    far = 2 * np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    views = np.zeros(len(p), int)
    for v in dirs26():
        loc, ri, _ = it.intersects_location(cp + v * far, np.repeat(-v[None], len(p), 0), multiple_hits=False)
        hitd = np.full(len(p), np.inf)
        hitd[ri] = np.linalg.norm(loc - cp[ri], axis=1)
        views += hitd < 0.3   # first hit within 3 mm of the point
    reg = asset.region.numpy()
    occl = views == 0
    by_reg = {S.REGIONS[r]: int((occl & (reg == r)).sum()) for r in np.unique(reg[occl])}
    res[body] = dict(proj_dist_p50_mm=float(np.median(dist) * 10), proj_dist_max_cm=float(dist.max()),
                     views0=int((views == 0).sum()), views1=int((views == 1).sum()), views_ge2=int((views >= 2).sum()),
                     occluded_by_region=by_reg)
    np.save(f'vis_{body}.npy', views)
    print(body, json.dumps(res[body]), flush=True)
json.dump(res, open('visibility_' + sys.argv[1] + '.json', 'w'), indent=1)
