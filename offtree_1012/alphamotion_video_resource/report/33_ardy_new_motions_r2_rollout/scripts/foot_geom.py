"""Collision-geometry foot hulls in the descriptor foot-node frame (same transform as the teammate's visual-mesh sole model), taken over the
whole MJCF subtree below the foot node (GR3's left sole is on the roll child, A3's on the roll child, Adam's on the toe child). Also a subtree-aware
lowest-point scan for the ruler."""
import numpy as np, mujoco
from scipy.spatial import ConvexHull
from greenwich_umi_proof.sole_geometry import _neutral_descriptor_fk, _MUJOCO_TO_GREENWICH
_DIRS = None
def dirs():
    global _DIRS
    if _DIRS is None:
        d = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float); _DIRS = d / np.linalg.norm(d, axis=1, keepdims=True)
    return _DIRS
def subtree(model, body):
    out = [body]; i = 0
    while i < len(out):
        out += [b for b in range(model.nbody) if model.body_parentid[b] == out[i] and b != out[i]]; i += 1
    return out
def geom_points_world(model, data, g):
    """points on the surface of collision geom g in world metres (mesh vertices, box corners, sphere/capsule/cylinder extreme samples)"""
    R = data.geom_xmat[g].reshape(3, 3); c = data.geom_xpos[g]; s = model.geom_size[g]; t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]; v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]; return v @ R.T + c
    if t == mujoco.mjtGeom.mjGEOM_BOX: return np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])]) @ R.T + c
    if t == mujoco.mjtGeom.mjGEOM_SPHERE: return c + s[0] * dirs()
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ax = R[:, 2] * s[1]; return np.concatenate([c + ax + s[0] * dirs(), c - ax + s[0] * dirs()])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        ang = np.linspace(0, 2 * np.pi, 36, endpoint=False); rim = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1) * s[0]
        return np.concatenate([(rim + [0, 0, s[1]]) @ R.T + c, (rim - [0, 0, s[1]]) @ R.T + c])
    return c[None] + model.geom_rbound[g] * dirs()
def collision_geoms(model, bodies):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
def collision_hulls(pk):
    """replace pk.hull with collision-geometry hulls of each foot subtree, in the descriptor foot frame (cm); returns a report dict"""
    import torch
    model = mujoco.MjModel.from_xml_path(pk.xml); data = mujoco.MjData(model); data.qpos[:] = model.qpos0; mujoco.mj_forward(model, data)
    root_cm = _MUJOCO_TO_GREENWICH @ np.asarray(data.qpos[:3], np.float64) * 100.0 if model.nq >= 7 else np.zeros(3)
    npos, nrot = _neutral_descriptor_fk(pk.spec, pk.rest.numpy()); rep = {}
    for i, j in enumerate(pk.feet):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pk.spec.joint_names[j]); assert b >= 0, pk.spec.joint_names[j]
        gs = collision_geoms(model, subtree(model, b)); assert gs, f'no collision geoms under {pk.spec.joint_names[j]}'
        w = np.concatenate([geom_points_world(model, data, g) for g in gs]) @ _MUJOCO_TO_GREENWICH.T * 100.0
        local = (w - (root_cm + npos[j])) @ nrot[j]
        try: local = local[ConvexHull(local).vertices]
        except Exception: pass
        pk.hull[i] = torch.as_tensor(local, dtype=torch.float32)
        rep[pk.spec.joint_names[j]] = dict(bodies=[model.body(x).name for x in subtree(model, b) if collision_geoms(model, [x])], geoms=len(gs), hull_points=int(len(local)), neutral_drop_cm=float(-local[:, 1].min()))
    return rep
def foot_lowest_points_subtree(m, d, robot, FEET):
    out = []
    for side in FEET[robot]:
        b = m.body(side).id; gs = collision_geoms(m, subtree(m, b)); zmin, xy = 1e9, None
        for g in gs:
            w = geom_points_world(m, d, g); k = w[:, 2].argmin()
            if w[k, 2] < zmin: zmin, xy = w[k, 2], w[k, :2]
        out.append((zmin, xy))
    return out
