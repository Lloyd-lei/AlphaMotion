"""Posed rigid objects with a signed distance function (outside-positive, metres).

Shape is detected from the mesh: a sphere (radial residual < 1.5 mm) or a box (OBB face residual < 1.5 mm) gets an
analytic SDF (exact); anything else falls back to trimesh (closest point + winding containment, exact on the mesh,
slower).  The object mesh is in the *retarget* frame and size UMR used (y-up -> z-up, scaled by smpl_scale).
"""
from __future__ import annotations

import numpy as np
import trimesh


class PosedObject:
    def __init__(self, name, mesh_local: trimesh.Trimesh, rot, pos, tol=0.0015):
        self.name = name
        self.mesh = mesh_local
        self.rot = np.asarray(rot, np.float64)        # [T,3,3] world <- object
        self.pos = np.asarray(pos, np.float64)        # [T,3]
        v = np.asarray(mesh_local.vertices)
        c = v.mean(0)
        r = np.linalg.norm(v - c, axis=1)
        self.kind = 'mesh'
        if r.std() < tol and (r.max() - r.min()) < 4 * tol:
            self.kind, self.center, self.radius = 'sphere', c, float(r.mean())
        else:
            obb = mesh_local.bounding_box_oriented
            T = np.asarray(obb.primitive.transform); ext = np.asarray(obb.primitive.extents) / 2
            q = (v - T[:3, 3]) @ T[:3, :3]
            face_res = np.min(np.abs(np.abs(q) - ext[None]), axis=1)
            if face_res.max() < tol:
                self.kind, self.box_T, self.box_half = 'box', T, ext
        self.bound_center = v.mean(0)
        self.bound_radius = float(np.linalg.norm(v - self.bound_center, axis=1).max())
        self._pq = None

    @property
    def T(self):
        return len(self.pos)

    def to_local(self, X, t):
        return (X - self.pos[t]) @ self.rot[t]

    def sdf(self, X, t, cull=0.05):
        """X [N,3] world -> (d [N] (inf where culled), normal_world [N,3], local [N,3])."""
        q = self.to_local(X, t)
        d = np.full(len(X), np.inf); n = np.zeros((len(X), 3))
        near = np.linalg.norm(q - self.bound_center, axis=1) < self.bound_radius + cull
        if not near.any():
            return d, n, q
        qn = q[near]
        if self.kind == 'sphere':
            v = qn - self.center; r = np.linalg.norm(v, axis=1)
            dn = r - self.radius; nl = v / np.maximum(r[:, None], 1e-12)
        elif self.kind == 'box':
            R = self.box_T[:3, :3]; p = (qn - self.box_T[:3, 3]) @ R
            h = self.box_half
            o = np.abs(p) - h
            out = np.linalg.norm(np.maximum(o, 0), axis=1)
            ins = np.minimum(o.max(1), 0)
            dn = out + ins
            # gradient
            g = np.where(o > 0, np.sign(p) * np.maximum(o, 0), 0)
            gi = np.zeros_like(p)
            k = o.argmax(1)
            gi[np.arange(len(p)), k] = np.sign(p[np.arange(len(p)), k])
            g = np.where((out > 0)[:, None], g, gi)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            nl = g @ R.T
        else:
            if self._pq is None:
                self._pq = trimesh.proximity.ProximityQuery(self.mesh)
            cp, dist, tri = self._pq.on_surface(qn)
            inside = self.mesh.contains(qn) if self.mesh.is_watertight else np.zeros(len(qn), bool)
            dn = np.where(inside, -dist, dist)
            v = qn - cp
            nl = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            nl = np.where(inside[:, None], -nl, nl)
            small = dist < 1e-6
            nl[small] = self.mesh.face_normals[tri[small]]
        d[near] = dn
        n[near] = nl @ self.rot[t].T
        return d, n, q

    def surface_points_world(self, t, n=1024, seed=0):
        if not hasattr(self, '_sp'):
            self._sp, _ = trimesh.sample.sample_surface_even(self.mesh, n, seed=seed)
        return self._sp @ self.rot[t].T + self.pos[t]


def load_umr_hoi_object(object_dir, frame_ids, smpl_scale, output_up='y', convert_y_up=True, ground_align=False,
                        floor_y=0.0, ground_offset=0.0):
    """Rebuild exactly the object UMR used in retarget (retarget_object_size='scaled')."""
    from pathlib import Path
    import mujoco
    import xml.etree.ElementTree as ET
    import tempfile
    from surface_retargeting.paths import add_umr_to_path
    add_umr_to_path()
    import retarget_smpl_to_humanoid_surface_vector as rs
    object_dir = Path(object_dir)
    xmls = sorted(object_dir.glob('*.xml'))
    xml = xmls[0]
    prop = rs.resolve_prop_trajectory(xml)
    mot = rs.read_prop_motion(prop, frame_ids, output_up, convert_y_up, ground_align, floor_y, ground_offset, smpl_scale)
    root = ET.parse(xml).getroot()
    rs.make_mjcf_mesh_paths_absolute(root, xml)
    rs.scale_mjcf_mesh_assets(root, smpl_scale)
    with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False, dir=str(object_dir.parent if False else '/tmp')) as f:
        f.write(ET.tostring(root, encoding='unicode')); tmp = f.name
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    V, F, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mi]:m.mesh_faceadr[mi] + m.mesh_facenum[mi]].astype(np.int64)
        V.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]); F.append(f + off); off += len(v)
    mesh = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
    return PosedObject(xml.stem, mesh, mot['rot_mats'], mot['positions']), dict(xml=str(xml), prop=str(prop))
