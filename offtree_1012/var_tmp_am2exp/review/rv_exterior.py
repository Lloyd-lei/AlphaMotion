"""Reviewer: exterior/interior + coverage audit at the vendor neutral pose qpos0 for robots.
For each posed point: distance to the assembled visual mesh, number of 26 directions (hemisphere-agnostic) whose
ray from p + 1mm*n escapes to infinity, escape along +n vs -n (normal orientation), region counts, sole/palm counts."""
import sys, json
import numpy as np, torch, trimesh, mujoco
sys.path.insert(0, '/var/tmp/alphamotion/am2exp/review')
from rv_robot_fk import RB, S, E, AX, world_mesh, pose

def dirs26():
    d = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if (x, y, z) != (0, 0, 0):
                    v = np.array([x, y, z], float); d.append(v / np.linalg.norm(v))
    return np.array(d)

def main(body):
    kin = RB.load_kinematics(body)
    asset = S.load_surface(body)
    m = kin.model; d = mujoco.MjData(m)
    q = kin.open_slides_qpos() if body == 'relay_franka_full_v2' else m.qpos0.copy()
    R, Pfk, _, _, _ = kin.forward(q)
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    root = d.xpos[kin.keep[0]].copy()
    mesh = world_mesh(kin, d, root)
    p, n = pose(asset, R, Pfk)
    _, dist, tri = trimesh.proximity.closest_point(mesh, p)
    ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh) if trimesh.ray.has_embree else mesh.ray
    D = dirs26()
    o = p + 0.1 * n                         # 1 mm out along the stored normal
    esc = np.zeros(len(p), int)
    for v in D:
        hit = ray.intersects_any(o, np.repeat(v[None], len(p), 0))
        esc += ~hit
    hit_pn = ray.intersects_any(o, n)
    o2 = p - 0.1 * n
    hit_mn = ray.intersects_any(o2, -n)
    reg = asset.region.numpy(); names = S.REGIONS
    counts = {names[r]: int((reg == r).sum()) for r in np.unique(reg)}
    # soles: foot-region points whose normal points down (y<-0.7) and within 2 cm of the lowest foot point
    foot = np.isin(reg, [names.index('L_foot'), names.index('R_foot')])
    ylow = p[foot, 1].min() if foot.any() else np.nan
    sole = foot & (n[:, 1] < -0.7) & (p[:, 1] < ylow + 2)
    sole_L = int((sole & (reg == names.index('L_foot'))).sum()); sole_R = int((sole & (reg == names.index('R_foot'))).sum())
    # hands: points per hand region and spread of normals (palm present => some normals facing medial/forward)
    hands = {}
    for h in ('L_hand', 'R_hand'):
        mk = reg == names.index(h)
        if mk.any():
            nn = n[mk]
            hands[h] = dict(n=int(mk.sum()), normal_dir_coverage=int(len(np.unique(np.argmax(nn @ D.T, 1)))))
    # region sanity: L_* regions on +x side? (canonical: Y up; check left/right sign consistency)
    side = {}
    for rn in ('L_foot', 'R_foot', 'L_hand', 'R_hand', 'L_thigh', 'R_thigh'):
        mk = reg == names.index(rn)
        if mk.any():
            side[rn] = [round(float(x), 1) for x in p[mk].mean(0)]
    out = dict(body=body, pose='qpos0', dist_to_mesh_max_cm=float(dist[asset.extra['point_rigid'].astype(bool)].max()) if 'point_rigid' in asset.extra else float(dist.max()),
               escape_hist={int(k): int((esc == k).sum()) for k in range(0, 4)}, n_escape_le1=int((esc <= 1).sum()),
               n_escape_0=int((esc == 0).sum()),
               normal_plus_blocked=int(hit_pn.sum()), normal_minus_free=int((~hit_mn).sum()),
               normal_wrong_suspect=int((hit_pn & ~hit_mn).sum()),
               region_counts=counts, sole_points=(sole_L, sole_R), foot_points=int(foot.sum()), hands=hands,
               region_centroids=side)
    np.savez(f'/var/tmp/alphamotion/am2exp/review/ext_{body}.npz', p=p, n=n, esc=esc, hit_pn=hit_pn, hit_mn=hit_mn, reg=reg)
    return out

if __name__ == '__main__':
    res = {}
    for b in sys.argv[1:]:
        try:
            res[b] = main(b)
        except Exception as e:
            res[b] = {'error': repr(e)}
        print(json.dumps(res[b]), flush=True)
    json.dump(res, open('/var/tmp/alphamotion/am2exp/review/exterior_' + ('all' if len(sys.argv) > 3 else sys.argv[1]) + '.json', 'w'), indent=1)
