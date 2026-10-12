"""CPU-only MuJoCo forward kinematics and compiled source visuals.

Mesh vertices are MuJoCo compiled geom-local coordinates. geom_xmat therefore
already contains the compiler's mesh-frame transform; never apply it twice.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


GEOM_NAMES = {int(getattr(mujoco.mjtGeom, 'mjGEOM_' + n.upper())): n
              for n in ('plane', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box', 'mesh')}


def xml_dependencies(path):
    """Read-only inventory of source XML/includes/assets, used for snapshot binding."""
    result, seen = [], set()
    def visit(p):
        p = Path(p).resolve()
        if p in seen:
            return
        seen.add(p); result.append(p)
        root = ET.parse(p).getroot()
        compiler = root.find('compiler')
        for e in root.iter():
            f = e.get('file')
            if not f:
                continue
            if e.tag == 'include':
                visit(p.parent / f)
            elif e.tag in ('mesh', 'texture', 'hfield'):
                sub = '' if compiler is None else compiler.get('meshdir' if e.tag == 'mesh' else 'texturedir', '')
                q = Path(f) if Path(f).is_absolute() else p.parent / sub / f
                q = q.resolve()
                if q not in seen:
                    seen.add(q); result.append(q)
    visit(path)
    return result


def load_object_model(path, scale):
    """Match upstream UMR: scale mesh assets, retaining geom offsets/rotations."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    meshbase = path.parent / (compiler.get('meshdir', '') if compiler is not None else '')
    for mesh in root.findall('./asset/mesh'):
        f = mesh.get('file')
        if f:
            mesh.set('file', str((meshbase / f).resolve()))
        old = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        if old.shape != (3,):
            raise ValueError('invalid source object mesh scale')
        mesh.set('scale', ' '.join(f'{x:.12g}' for x in old * float(scale)))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def static_scene(model, prefix='robot', object_model=False):
    arrays, geoms = {}, []
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    candidates = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
    visual = [i for i in candidates if (model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0)
              or model.geom_group[i] == 2]
    chosen = visual or candidates
    for i in chosen:
        type_name = GEOM_NAMES.get(int(model.geom_type[i]))
        if type_name is None:
            raise ValueError(f'unsupported source geom type {int(model.geom_type[i])}')
        mesh_key = None
        if type_name == 'mesh':
            mi = int(model.geom_dataid[i]); mesh_key = f'{prefix}_mesh_{mi}'
            va, vn = int(model.mesh_vertadr[mi]), int(model.mesh_vertnum[mi])
            fa, fn = int(model.mesh_faceadr[mi]), int(model.mesh_facenum[mi])
            arrays[mesh_key + '_vertices'] = np.asarray(model.mesh_vert[va:va+vn], dtype=np.float32).copy()
            arrays[mesh_key + '_faces'] = np.asarray(model.mesh_face[fa:fa+fn], dtype=np.uint32).copy()
        color = model.geom_rgba[i].copy()
        mat = int(model.geom_matid[i])
        if mat >= 0:
            color = model.mat_rgba[mat].copy()
        g = dict(id=i, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f'geom_{i}',
                 type=type_name, mesh=mesh_key, size=model.geom_size[i].tolist(), rgba=color.tolist(),
                 body=int(model.geom_bodyid[i]), group=int(model.geom_group[i]))
        if object_model:
            g.update(local_pos=data.geom_xpos[i].tolist(), local_rot=data.geom_xmat[i].reshape(3, 3).tolist())
        geoms.append(g)
    return geoms, arrays


def forward(model, qpos):
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError('qpos shape or finite-value validation failed')
    c = len(qpos)
    result = {'geom_pos': np.empty((c, model.ngeom, 3), np.float32),
              'geom_rot': np.empty((c, model.ngeom, 3, 3), np.float32),
              'joint_pos': np.empty((c, model.njnt, 3), np.float32),
              'body_pos': np.empty((c, model.nbody, 3), np.float32)}
    data = mujoco.MjData(model)
    for j, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        result['geom_pos'][j] = data.geom_xpos
        result['geom_rot'][j] = data.geom_xmat.reshape(-1, 3, 3)
        result['joint_pos'][j] = data.xanchor
        result['body_pos'][j] = data.xpos
    return result


def world_points(poses, points, geom):
    return (np.einsum('cpij,pj->cpi', poses['geom_rot'][:, geom], points, optimize=True)
            + poses['geom_pos'][:, geom])


def object_diagnostic(geoms, arrays):
    """Independent visual-mesh gauge. Never inherit an upstream object classifier.

All sampled ball vertices must support a sphere fit; 8 equidistant box corners
alone cannot establish a sphere. Otherwise use convex supporting-plane distance,
explicitly labeled as an approximate, non-convex-unaware UI diagnostic.
"""
    verts = []
    for g in geoms:
        if g['type'] == 'mesh':
            v = arrays[g['mesh'] + '_vertices'].astype(np.float64)
            r = np.array(g['local_rot']); p = np.array(g['local_pos'])
            verts.append(v @ r.T + p)
    if not verts:
        return None
    v = np.unique(np.concatenate(verts), axis=0)
    A = np.column_stack([2*v, np.ones(len(v))])
    fit, *_ = np.linalg.lstsq(A, np.sum(v*v, axis=1), rcond=None)
    center = fit[:3]; radius = float(np.sqrt(max(0, fit[3] + center @ center)))
    residual = np.abs(np.linalg.norm(v-center, axis=1)-radius)
    if len(v) >= 40 and residual.max() < 0.00005:
        return {'kind': 'sphere_fit', 'center': center, 'radius': radius,
                'fit_max_error_m': float(residual.max()),
                'description': '界面现算：源网格球面拟合距离（误差已测），非数据标签'}
    from scipy.spatial import ConvexHull
    equations = ConvexHull(v).equations
    equations = np.unique(np.round(equations, 12), axis=0)
    return {'kind': 'convex_planes', 'equations': equations,
            'description': '界面现算：源网格凸包支撑面距离；凸外距离近似，不代表凹物体精确SDF'}


def object_distance(points, pos, rot, diagnostic):
    if diagnostic is None:
        return None
    local = np.einsum('cpi,cij->cpj', points - pos[:, None, :], rot, optimize=True)
    if diagnostic['kind'] == 'sphere_fit':
        return (np.linalg.norm(local - diagnostic['center'], axis=-1) - diagnostic['radius']).astype(np.float32)
    eq = diagnostic['equations']; result = np.empty(local.shape[:2], np.float32)
    for i in range(len(local)):
        for j in range(0, local.shape[1], 256):
            result[i, j:j+256] = np.max(local[i, j:j+256] @ eq[:, :3].T + eq[:, 3], axis=1)
    return result
