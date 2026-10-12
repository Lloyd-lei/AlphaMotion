"""Posed rigid objects with a signed distance function (outside-positive, metres).

Shape is detected from the mesh: a sphere (radial residual < 1.5 mm) or a box (OBB face residual < 1.5 mm) gets an
analytic SDF (exact); anything else falls back to trimesh (closest point + winding containment, exact on the mesh,
slower).  The object mesh is in the *retarget* frame and size UMR used (y-up -> z-up, scaled by smpl_scale).
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

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


# ============================================================================================ pilot (strict) loader
class ConvexUnionObject(PosedObject):
    """Solid = union of convex pieces (UMR's CoACD collision pieces for HiPHI; the single convex mesh otherwise).
    Every HiPHI visual OBJ is non-watertight (open shells / UV seams), so 'inside' is undefined on the visual mesh;
    the CoACD union is watertight, is what UMR's own robot-object hard constraint uses, and (pieces = hulls of mesh
    parts) covers the visual solid -> no penetration of the union implies no penetration of the visual object.
    Signed distance: inside = max over face planes of the deepest containing piece (exact per piece); outside =
    exact closest point on the nearest piece hull (trimesh) for points within the cull band."""

    def __init__(self, name, visual_local, pieces_local, rot, pos):
        self.name = name
        self.mesh = visual_local                         # rendering / visual-hover stats only
        self.rot = np.asarray(rot, np.float64); self.pos = np.asarray(pos, np.float64)
        self.kind = 'convex_union'
        from scipy.spatial import ConvexHull
        self.pieces = []
        for v in pieces_local:
            v = np.asarray(v, np.float64)
            h = ConvexHull(v)
            eq = h.equations                                # n.x + c <= 0 inside, |n| = 1
            hm = trimesh.convex.convex_hull(v)
            c = v[h.vertices].mean(0)
            self.pieces.append(dict(A=eq[:, :3], c=eq[:, 3], hull=hm, center=c,
                                    radius=float(np.linalg.norm(v[h.vertices] - c, axis=1).max()),
                                    pq=trimesh.proximity.ProximityQuery(hm), verts=v[h.vertices]))
        allv = np.concatenate([p['verts'] for p in self.pieces])
        self.bound_center = allv.mean(0)
        self.bound_radius = float(np.linalg.norm(allv - self.bound_center, axis=1).max())
        self._pq = None

    def sdf(self, X, t, cull=0.05):
        q = self.to_local(X, t)
        N = len(X)
        d = np.full(N, np.inf); nl = np.zeros((N, 3))
        near = np.linalg.norm(q - self.bound_center, axis=1) < self.bound_radius + cull
        if not near.any():
            return d, nl @ self.rot[t].T, q
        idx = np.flatnonzero(near)
        qn = q[idx]
        best_out = np.full(len(idx), np.inf); best_out_n = np.zeros((len(idx), 3))
        best_in = np.full(len(idx), np.inf); best_in_n = np.zeros((len(idx), 3))     # most negative plane value
        for p in self.pieces:
            k = np.flatnonzero(np.linalg.norm(qn - p['center'], axis=1) < p['radius'] + cull)
            if not len(k):
                continue
            s = qn[k] @ p['A'].T + p['c']                   # [n, F]
            j = s.argmax(1); smax = s[np.arange(len(k)), j]
            ins = smax < 0
            if ins.any():
                ki = k[ins]
                better = smax[ins] < best_in[ki]
                best_in[ki[better]] = smax[ins][better]; best_in_n[ki[better]] = p['A'][j[ins][better]]
            out = ~ins & (smax < cull)
            if out.any():
                ko = k[out]
                cp, dist, tri = p['pq'].on_surface(qn[ko])
                better = dist < best_out[ko]
                v = qn[ko] - cp
                nv = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                small = dist < 1e-7
                nv[small] = p['hull'].face_normals[tri[small]]
                best_out[ko[better]] = dist[better]; best_out_n[ko[better]] = nv[better]
        inside = np.isfinite(best_in)
        dd = np.where(inside, best_in, best_out)
        nn = np.where(inside[:, None], best_in_n, best_out_n)
        d[idx] = dd; nl[idx] = nn
        return d, nl @ self.rot[t].T, q

    def surface_points_world(self, t, n=1024, seed=0):
        if not hasattr(self, '_sp'):
            self._sp, _ = trimesh.sample.sample_surface_even(self.mesh, n, seed=seed)
        return self._sp @ self.rot[t].T + self.pos[t]


def _mjcf_geoms_local(xml, smpl_scale):
    """Load the object MJCF exactly as UMR does (absolute mesh paths, scaled by smpl_scale); return per-geom local
    meshes split into visual (contype=conaffinity=0) and collision geoms, in the object body frame."""
    from pathlib import Path
    import mujoco
    import xml.etree.ElementTree as ET
    import tempfile
    from surface_retargeting.paths import add_umr_to_path
    add_umr_to_path()
    import retarget_smpl_to_humanoid_surface_vector as rs
    root = ET.parse(xml).getroot()
    rs.make_mjcf_mesh_paths_absolute(root, Path(xml))
    rs.scale_mjcf_mesh_assets(root, smpl_scale)
    with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False, dir=f'{SRCFG.data_root}/pilot/tmp') as f:
        f.write(ET.tostring(root, encoding='unicode')); tmp = f.name
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    vis, col = [], []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mi]:m.mesh_faceadr[mi] + m.mesh_facenum[mi]].astype(np.int64)
        vw = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        is_col = bool(m.geom_contype[g] or m.geom_conaffinity[g])
        (col if is_col else vis).append((vw, f))
        if is_col and not any(True for _ in vis) and m.ngeom == 1:
            pass
    if not vis:                                        # single geom used for both (OmniContact box/ball MJCFs)
        vis = list(col)
    return vis, col


def load_object_strict(object_dir, frame_ids, smpl_scale, output_up='y'):
    """Object UMR used in retarget, for OmniContact (prop_<name>.csv + <name>.xml in the sequence dir) and HiPHI
    (metadata.json -> object_meshes/<mesh>.xml CoACD MJCF + object_tracks/<id>.csv).  Returns (obj, info)."""
    from pathlib import Path
    from surface_retargeting.paths import add_umr_to_path
    add_umr_to_path()
    import retarget_smpl_to_humanoid_surface_vector as rs
    import humanoid_retarget_pipeline_hsi_hoi as hp
    items = hp.discover_samp_objects(Path(object_dir))
    if len(items) != 1:
        raise RuntimeError(f'expected exactly one object, got {len(items)} in {object_dir}')
    it = items[0]
    mot = rs.read_prop_motion(Path(it['prop']), frame_ids, output_up, True, False, 0.0, 0.0, smpl_scale)
    vis, col = _mjcf_geoms_local(it['xml'], smpl_scale)
    V, F, off = [], [], 0
    for v, f in vis:
        V.append(v); F.append(f + off); off += len(v)
    vmesh = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
    vmesh.merge_vertices(merge_tex=True, merge_norm=True)
    # analytic sphere / box when the single collision geom is one (exact SDF); else convex union of collision pieces
    if len(col) == 1:
        cm = trimesh.Trimesh(col[0][0], col[0][1], process=True)
        po = PosedObject(it['name'], cm, mot['rot_mats'], mot['positions'])
        if po.kind in ('sphere', 'box'):
            po.mesh_visual = vmesh
            return po, dict(xml=it['xml'], prop=it['prop'], solid=po.kind, n_pieces=1, visual_watertight=bool(vmesh.is_watertight))
    obj = ConvexUnionObject(it['name'], vmesh, [c[0] for c in col], mot['rot_mats'], mot['positions'])
    obj.mesh_visual = vmesh
    return obj, dict(xml=it['xml'], prop=it['prop'], solid='convex_union', n_pieces=len(col),
                     visual_watertight=bool(vmesh.is_watertight))


def mj_solid_geoms(obj):
    """Solid as MuJoCo geoms in the object body frame: list of dicts (type, size, pos, mat, verts)."""
    out = []
    if obj.kind == 'sphere':
        out.append(dict(type='sphere', size=[obj.radius, 0, 0], pos=obj.center, mat=np.eye(3)))
    elif obj.kind == 'box':
        out.append(dict(type='box', size=list(obj.box_half), pos=obj.box_T[:3, 3], mat=obj.box_T[:3, :3]))
    elif obj.kind == 'convex_union':
        for p in obj.pieces:
            out.append(dict(type='mesh', verts=p['verts'], pos=np.zeros(3), mat=np.eye(3)))
    else:
        raise ValueError(f'no MuJoCo solid for object kind {obj.kind}')
    return out
