"""Lowest point of robot COLLISION geoms vs lowest point of VISUAL meshes (hull vertices), qpos0 + 40 random poses.
Analytic per primitive: sphere z-r; capsule/cylinder from axis endpoints; box corners; mesh (convex hull) vertices."""
import sys, json, numpy as np, mujoco
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import robots as R, surface as S, geometry as G
from umrf.paths import ROBOTS
T = mujoco.mjtGeom

def coll_lowest(m, d, g):
    p = d.geom_xpos[g]; Rm = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]; t = m.geom_type[g]
    if t == T.mjGEOM_SPHERE: return p[2] - s[0]
    if t == T.mjGEOM_CAPSULE:
        a = Rm[:, 2] * s[1]; return min(p[2] + a[2], p[2] - a[2]) - s[0]
    if t == T.mjGEOM_CYLINDER:
        a = Rm[:, 2] * s[1]; ax = Rm[:, 2]; rad = s[0] * np.sqrt(max(0.0, 1 - ax[2] ** 2))
        return min(p[2] + a[2], p[2] - a[2]) - rad
    if t == T.mjGEOM_BOX:
        c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
        return (c @ Rm.T + p)[:, 2].min()
    if t == T.mjGEOM_ELLIPSOID:
        # support function in -z: |diag(s) R^T e_z|
        return p[2] - np.linalg.norm(s * Rm[2, :])
    if t == T.mjGEOM_MESH:
        mi = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
        return (v @ Rm.T + p)[:, 2].min()
    return np.inf

out = {}
for rec in R.load_inventory():
    n = rec['name']
    st = json.load(open(ROBOTS / n / 'staging.json'))
    m = mujoco.MjModel.from_xml_path(st['staged_mjcf'])
    geo = G.RobotGeometry(m, S.load_surface(n))
    cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    rng = np.random.default_rng(0); res = []; worst = None
    for k in range(41):
        q = m.qpos0.copy()
        if k:
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
                    lo, hi = m.jnt_range[j]; q[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
            qq = rng.normal(size=4); q[3:7] = qq / np.linalg.norm(qq)
        q[2] = 1.5
        geo.set(q); z, X, _ = geo.ground(); vis = z.min()
        cz = [(coll_lowest(m, geo.d, g), g) for g in cg]; cmin, gmin = min(cz)
        res.append(cmin - vis)
        if worst is None or cmin - vis < worst[0]:
            worst = (cmin - vis, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gmin) or f'geom{gmin}',
                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gmin])))
    r = np.array(res) * 1000
    out[n] = dict(qpos0_coll_minus_vis_mm=round(float(r[0]), 2), random_min_mm=round(float(r[1:].min()), 2),
                  random_p50_mm=round(float(np.median(r[1:])), 2), worst_geom=worst[1], worst_body=worst[2])
    print(n, out[n], flush=True)
json.dump(out, open('/home/arenalabs/umr_factory_data/pilot/diag/coll_vs_vis.json', 'w'), indent=1)
