"""unitree_h2 pipeline helpers shared by measure_h2.py and render_h2.py (independent of the engine's gauges)."""
import json, os
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco, numpy as np, trimesh
from scipy.spatial import ConvexHull

ROBOT = 'unitree_h2'
DATA = Path('/home/arenalabs/umr_factory_data')
RD = DATA / 'robots' / ROBOT; PD = RD / 'pipeline'; QD = DATA / 'qa' / ROBOT


def load_object_pieces(object_xml, hscale):
    """Object collision geoms (convex pieces) in the object body frame, meshes scaled by smpl_scale like UMR."""
    oxml = Path(object_xml)
    root = ET.parse(oxml).getroot()
    md = next((c.get('meshdir') for c in root.iter('compiler') if c.get('meshdir')), None)
    for c in root.iter('compiler'):
        c.attrib.pop('meshdir', None)
    for mesh in root.iter('mesh'):
        f = mesh.get('file')
        if f and not os.path.isabs(f):
            mesh.set('file', str((oxml.parent / (md or '') / f).resolve()))
        sc = np.array([float(x) for x in (mesh.get('scale') or '1 1 1').split()]) * hscale
        mesh.set('scale', ' '.join(f'{x:.9g}' for x in sc))
    om = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    od = mujoco.MjData(om); mujoco.mj_forward(om, od)
    pieces, visual = [], []
    for g in range(om.ngeom):
        if om.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = om.geom_dataid[g]
        v = om.mesh_vert[om.mesh_vertadr[mi]:om.mesh_vertadr[mi] + om.mesh_vertnum[mi]].astype(np.float64)
        f = om.mesh_face[om.mesh_faceadr[mi]:om.mesh_faceadr[mi] + om.mesh_facenum[mi]].astype(np.int64)
        v = v @ od.geom_xmat[g].reshape(3, 3).T + od.geom_xpos[g]
        if om.geom_contype[g] or om.geom_conaffinity[g]:
            h = ConvexHull(v)
            vv = v[h.vertices]
            pieces.append(dict(v=vv, A=h.equations[:, :3], c=h.equations[:, 3], hull=trimesh.convex.convex_hull(vv),
                               ctr=vv.mean(0), rad=float(np.linalg.norm(vv - vv.mean(0), axis=1).max())))
        else:
            visual.append((v, f))
    if not pieces:
        raise RuntimeError('object has no collision geoms')
    return pieces, visual


def obj_sdf(pieces, pos, rot, Xw, cull=0.03):
    """Signed distance (m) of world points to the union of convex pieces posed by (pos, rot: world<-object);
    +inf beyond cull.  Returns d [N], normal in object frame [N,3], object-frame coords [N,3]."""
    q = (Xw - pos) @ rot
    N = len(q); dd = np.full(N, np.inf); nn = np.zeros((N, 3))
    for p in pieces:
        k = np.flatnonzero(np.linalg.norm(q - p['ctr'], axis=1) < p['rad'] + cull)
        if not len(k):
            continue
        s = q[k] @ p['A'].T + p['c']; j = s.argmax(1); smax = s[np.arange(len(k)), j]
        ins = smax <= 0
        val = np.where(ins, smax, np.inf); nrm = p['A'][j].copy()
        out = np.flatnonzero(~ins & (smax < cull))
        if len(out):
            cpt, dist, tri = trimesh.proximity.closest_point(p['hull'], q[k[out]])
            val[out] = dist
            v = q[k[out]] - cpt; nv = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            small = dist < 1e-7; nv[small] = p['hull'].face_normals[tri[small]]
            nrm[out] = nv
        better = val < dd[k]
        dd[k[better]] = val[better]; nn[k[better]] = nrm[better]
    return dd, nn, q


def geom_map(m0, m):
    """m0 (staged robot) geom id -> geom id in a model m built from the same MJCF plus extra world/body geoms
    (a geom added to the worldbody shifts every id).  Extra geoms must carry names starting with '__' or live in
    the world body / bodies appended after the robot."""
    def by_body(model, keep):
        out = {}
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if 0 < b < m0.nbody and keep(g):
                out.setdefault(b, []).append(g)
        return out
    nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    a, b = by_body(m0, lambda g: True), by_body(m, lambda g: not nm(g).startswith('__'))
    gm = {g: b[k][i] for k, gs in a.items() for i, g in enumerate(gs)}
    assert all(m.geom_type[v] == m0.geom_type[k] for k, v in gm.items())
    return gm
