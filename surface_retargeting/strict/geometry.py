"""Exact-as-possible per-frame geometry of a posed robot.

Gauges (all metres, + = separated, - = penetrating).  The robot SOLID is the union of its visual meshes AND its
MuJoCo collision geoms (pilot change, 2026-10-04: on 12 of the 21 robots collision geoms reach 2-48 mm below / outside
the visual meshes in some poses, so a visual-only gauge can pass while the physics geometry penetrates):
  ground : min over (convex-hull vertices of every visual geom  U  4096 dense exterior points  U  the analytic lowest
           support point of every robot collision geom) of z - floor.  Exact for a plane (min of a linear function
           over a convex set is attained at a hull vertex / the support point).
  object : (a) object SDF (analytic sphere/box, else union of convex collision pieces) at dense points U hull vertices
           near the object -- sampled on the robot side, between-sample sag ~ s^2/(8R);
           (b) mj_geomDistance between every robot collision geom and every object solid geom (object added to the
           model as a mocap body) -- exact convex-convex.
  self   : MuJoCo collision geoms, every pair MuJoCo considers (parent/child filtered, staged <exclude> list applied),
           mj_geomDistance (exact for primitives / convex hulls of collision meshes).
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from surface_retargeting.robots.surface import REGIONS

T_ = mujoco.mjtGeom
OBJ_BODY = '__pilot_object'


def skew_batch(r):
    z = np.zeros(len(r))
    return np.stack([np.stack([z, -r[:, 2], r[:, 1]], 1), np.stack([r[:, 2], z, -r[:, 0]], 1),
                     np.stack([-r[:, 1], r[:, 0], z], 1)], 1)


def build_model(xml_path, obj=None):
    """Staged robot MJCF (+ the object's solid as a mocap body appended last, so robot geom/body/qpos ids are
    unchanged).  Returns MjModel."""
    if obj is None:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    from surface_retargeting.strict.objects import mj_solid_geoms
    m0 = mujoco.MjModel.from_xml_path(str(xml_path))
    spec = mujoco.MjSpec.from_file(str(xml_path))
    b = spec.worldbody.add_body(); b.name = OBJ_BODY; b.mocap = True
    for i, gd in enumerate(mj_solid_geoms(obj)):
        g = b.add_geom()
        g.name = f'{OBJ_BODY}_g{i}'
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(gd['mat'], np.float64).reshape(-1))
        g.pos = np.asarray(gd['pos'], np.float64); g.quat = q
        if gd['type'] == 'sphere':
            g.type = T_.mjGEOM_SPHERE; g.size = np.asarray(gd['size'], np.float64)
        elif gd['type'] == 'box':
            g.type = T_.mjGEOM_BOX; g.size = np.asarray(gd['size'], np.float64)
        else:
            mesh = spec.add_mesh(); mesh.name = f'{OBJ_BODY}_m{i}'
            mesh.uservert = np.asarray(gd['verts'], np.float32).reshape(-1)
            g.type = T_.mjGEOM_MESH; g.meshname = mesh.name
        g.contype = 0x7FFFFFFF; g.conaffinity = 0x7FFFFFFF
        g.rgba = [0.3, 0.5, 0.9, 0.3]; g.group = 3
    m = spec.compile()
    assert m.nq == m0.nq and m.nv == m0.nv
    for gi in range(m0.ngeom):
        assert m.geom_bodyid[gi] == m0.geom_bodyid[gi] and m.geom_type[gi] == m0.geom_type[gi], 'geom ids moved'
    return m


class RobotGeometry:
    def __init__(self, m, surf, floor=0.0, obj=None, self_threshold=0.02, use_collision=True):
        self.m = m
        self.d = mujoco.MjData(m)
        self.floor = float(floor)
        self.obj = obj
        self.self_thr = self_threshold
        self.pl = surf['points_local'].astype(np.float64); self.pg = surf['geom'].astype(np.int64)
        self.preg = surf['region'].astype(np.int64)
        self.hl = surf['hull_local'].astype(np.float64); self.hg = surf['hull_geom'].astype(np.int64)
        breg = surf['body_region']
        if len(breg) < m.nbody:                          # object mocap body appended: give it the trunk id (unused)
            breg = np.concatenate([breg, np.zeros(m.nbody - len(breg), breg.dtype)])
        self.breg = breg
        self.hreg = breg[m.geom_bodyid[self.hg]].astype(np.int64)
        self.pbody = m.geom_bodyid[self.pg].astype(np.int64)
        self.hbody = m.geom_bodyid[self.hg].astype(np.int64)
        self.nreg = len(REGIONS)
        # all candidate points: dense first, then hull
        self.all_local = np.concatenate([self.pl, self.hl]); self.all_geom = np.concatenate([self.pg, self.hg])
        self.all_reg = np.concatenate([self.preg, self.hreg]); self.all_body = np.concatenate([self.pbody, self.hbody])
        self.ndense = len(self.pl)
        self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY)
        robot_body = (lambda b: b != 0 and (self.obj_body < 0 or b != self.obj_body))
        self.rgeoms = np.array([g for g in range(m.ngeom) if robot_body(m.geom_bodyid[g]) and
                                (m.geom_contype[g] or m.geom_conaffinity[g])], int)
        self._rset = set(self.rgeoms.tolist())
        self.ogeoms = np.array([g for g in range(m.ngeom) if self.obj_body >= 0 and m.geom_bodyid[g] == self.obj_body], int)
        self._oset = set(self.ogeoms.tolist())
        self._fromto = np.zeros(6)
        self._jcache = {}
        self._t = None
        # ---- collision geoms for the ground gauge
        self.use_collision = bool(use_collision)
        self.cg = self.rgeoms.copy() if use_collision else np.zeros(0, int)
        self.cg_body = m.geom_bodyid[self.cg].astype(np.int64)
        self.cg_reg = breg[self.cg_body].astype(np.int64) if len(self.cg) else np.zeros(0, np.int64)
        self._cg_mesh_local = {}
        for g in self.cg:
            if m.geom_type[g] == T_.mjGEOM_MESH:
                mi = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(np.float64)
                try:
                    v = v[ConvexHull(v).vertices]
                except Exception:
                    pass
                self._cg_mesh_local[int(g)] = v
        if self.obj_body >= 0:
            self.mocap_id = int(m.body_mocapid[self.obj_body])
            self._oquat = np.zeros((obj.T, 4))
            for t in range(obj.T):
                mujoco.mju_mat2Quat(self._oquat[t], obj.rot[t].reshape(-1))

    # ------------------------------------------------------------------ kinematics
    def set(self, q, t=None):
        self.d.qpos[:] = q
        if self.obj_body >= 0 and t is not None:
            self.d.mocap_pos[self.mocap_id] = self.obj.pos[t]
            self.d.mocap_quat[self.mocap_id] = self._oquat[t]
            self._t = t
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        self._jcache = {}

    def world(self, idx=None):
        if idx is None:                                   # fast path: one matmul per geom
            if not hasattr(self, '_groups'):
                self._groups = [(int(gg), np.flatnonzero(self.all_geom == gg)) for gg in np.unique(self.all_geom)]
            out = np.empty((len(self.all_geom), 3))
            for gg, k in self._groups:
                out[k] = self.all_local[k] @ self.d.geom_xmat[gg].reshape(3, 3).T + self.d.geom_xpos[gg]
            return out
        g = self.all_geom if idx is None else self.all_geom[idx]
        L = self.all_local if idx is None else self.all_local[idx]
        Rm = self.d.geom_xmat[g].reshape(-1, 3, 3)
        return np.einsum('nij,nj->ni', Rm, L) + self.d.geom_xpos[g]

    def dense_world(self):
        return self.world(np.arange(self.ndense))

    def body_jac(self, b):
        if b not in self._jcache:
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, self.d, jp, jr, int(b))
            self._jcache[b] = (jp, jr)
        return self._jcache[b]

    def point_jac(self, bodies, P):
        """[N,3,nv] translational Jacobians of world points P rigidly attached to bodies."""
        out = np.zeros((len(P), 3, self.m.nv))
        for b in np.unique(bodies):
            k = np.flatnonzero(bodies == b)
            jp, jr = self.body_jac(int(b))
            r = P[k] - self.d.xpos[int(b)]
            out[k] = jp[None] - skew_batch(r) @ jr[None]
        return out

    # ------------------------------------------------------------------ gauges
    def coll_lowest(self):
        """Lowest world point (support point in -z) of every robot collision geom: (z - floor [nc], P [nc,3])."""
        m, d = self.m, self.d
        n = len(self.cg)
        P = np.zeros((n, 3))
        for i, g in enumerate(self.cg):
            p = d.geom_xpos[g]; Rm = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]; ty = m.geom_type[g]
            if ty == T_.mjGEOM_SPHERE:
                P[i] = p - [0, 0, s[0]]
            elif ty == T_.mjGEOM_CAPSULE:
                a = Rm[:, 2] * s[1]; e = p - a if a[2] > 0 else p + a
                P[i] = e - [0, 0, s[0]]
            elif ty == T_.mjGEOM_CYLINDER:
                ax = Rm[:, 2]; a = ax * s[1]; e = p - a if a[2] > 0 else p + a
                u = np.array([0, 0, -1.0]); r = u - (u @ ax) * ax; nr = np.linalg.norm(r)
                P[i] = e + (s[0] * r / nr if nr > 1e-9 else 0)
            elif ty == T_.mjGEOM_BOX:
                sg = -np.sign(Rm[2, :]); sg[sg == 0] = 1
                P[i] = p + Rm @ (sg * s)
            elif ty == T_.mjGEOM_ELLIPSOID:
                u = np.array([0, 0, -1.0]); w = s * (Rm.T @ u); P[i] = p + Rm @ (s * w / max(np.linalg.norm(w), 1e-12))
            elif ty == T_.mjGEOM_MESH:
                v = self._cg_mesh_local[int(g)] @ Rm.T + p
                P[i] = v[np.argmin(v[:, 2])]
            else:
                P[i] = p + [0, 0, 1e3]                     # plane/hfield/sdf on a robot: not expected
        return P[:, 2] - self.floor, P

    def ground(self):
        X = self.world()
        z = X[:, 2] - self.floor
        reg_min = np.full(self.nreg, np.inf)
        np.minimum.at(reg_min, self.all_reg, z)
        return z, X, reg_min

    def object(self, X, t):
        if self.obj is None:
            return None, None, np.full(self.nreg, np.inf)
        dist, nrm, _ = self.obj.sdf(X, t)
        reg_min = np.full(self.nreg, np.inf)
        np.minimum.at(reg_min, self.all_reg, dist)
        return dist, nrm, reg_min

    def pairs(self):
        """(self_pairs, obj_pairs): lists of (g1, g2, dist, fromto) within self_thr.  obj pairs are (robot, object)."""
        m, d = self.m, self.d
        saved = m.geom_margin.copy()
        cand_s, cand_o = set(), set()
        try:
            m.geom_margin[:] = np.maximum(saved, self.self_thr)
            mujoco.mj_collision(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                a, b = int(c.geom1), int(c.geom2)
                if a in self._rset and b in self._rset:
                    cand_s.add((min(a, b), max(a, b)))
                elif a in self._rset and b in self._oset:
                    cand_o.add((a, b))
                elif b in self._rset and a in self._oset:
                    cand_o.add((b, a))
        finally:
            m.geom_margin[:] = saved
        out_s, out_o = [], []
        for cand, out in ((cand_s, out_s), (cand_o, out_o)):
            for g1, g2 in sorted(cand):
                ft = np.zeros(6)
                dist = mujoco.mj_geomDistance(m, d, g1, g2, self.self_thr, ft)
                if dist < self.self_thr:
                    out.append((g1, g2, float(dist), ft))
        return out_s, out_o

    def self_pairs(self):
        return self.pairs()[0]

    def measure(self, q, t):
        """Full exact evaluation of one frame -> dict of scalars/arrays."""
        self.set(q, t)
        z, X, gmin = self.ground()
        gmin_vis = gmin.copy()
        if len(self.cg):
            cz, cP = self.coll_lowest()
            np.minimum.at(gmin, self.cg_reg, cz)
        else:
            cz, cP = np.zeros(0), np.zeros((0, 3))
        od, on, omin = self.object(X, t)
        sp, op = self.pairs()
        smin = min([p[2] for p in sp], default=np.inf)
        obj_min_vis = float(np.min(od)) if od is not None else np.inf
        ocmin = min([p[2] for p in op], default=np.inf)
        oreg_coll = np.full(self.nreg, np.inf)
        for g1, g2, dd, _ in op:
            r = self.breg[self.m.geom_bodyid[g1]]
            oreg_coll[r] = min(oreg_coll[r], dd)
        omin_all = np.minimum(omin, oreg_coll)
        return dict(z=z, X=X, cz=cz, cP=cP, ground_reg_min=gmin, ground_reg_min_visual=gmin_vis,
                    ground_min=float(min(z.min(), cz.min() if len(cz) else np.inf)),
                    ground_min_visual=float(z.min()), ground_min_coll=float(cz.min()) if len(cz) else np.inf,
                    obj_d=od, obj_n=on, obj_reg_min=omin_all, obj_reg_min_visual=omin, obj_pairs=op,
                    obj_min=float(min(obj_min_vis, ocmin)), obj_min_visual=obj_min_vis, obj_min_coll=float(ocmin),
                    self_pairs=sp, self_min=smin)

    def _pair_normal(self, g1, g2, dist, ft):
        """Unit direction n (world) such that moving geom1 by +eps*n increases the distance (sign by FD probe:
        MuJoCo's fromto orientation is not consistent for every penetrating convex pair)."""
        p1, p2 = ft[:3], ft[3:]
        dv = p1 - p2
        nn = np.linalg.norm(dv)
        if nn < 1e-9:
            return None
        u = dv / nn
        eps = 1e-4
        save = self.d.geom_xpos[g1].copy()
        self.d.geom_xpos[g1] = save + eps * u
        dp = mujoco.mj_geomDistance(self.m, self.d, g1, g2, self.self_thr, np.zeros(6))
        self.d.geom_xpos[g1] = save
        return u if dp > dist else -u

    def self_rows(self, pairs, target):
        """Linear rows A dq >= b for collision pairs: dist + n.(J1-J2) dq >= target."""
        A, b = [], []
        for g1, g2, dist, ft in pairs:
            n = self._pair_normal(g1, g2, dist, ft)
            if n is None:
                continue
            p1, p2 = ft[:3], ft[3:]
            b1, b2 = int(self.m.geom_bodyid[g1]), int(self.m.geom_bodyid[g2])
            J1 = self.point_jac(np.array([b1]), p1[None])[0]
            J2 = self.point_jac(np.array([b2]), p2[None])[0]
            A.append(n @ (J1 - J2)); b.append(target - dist)
        return A, b

    def obj_pair_rows(self, pairs, target):
        """Rows for (robot geom, object geom) pairs; the object is fixed (only the robot side moves)."""
        A, b, regs, dists = [], [], [], []
        for g1, g2, dist, ft in pairs:
            n = self._pair_normal(g1, g2, dist, ft)
            if n is None:
                continue
            b1 = int(self.m.geom_bodyid[g1])
            J1 = self.point_jac(np.array([b1]), ft[:3][None])[0]
            A.append(n @ J1); b.append(target - dist); regs.append(int(self.breg[b1])); dists.append(dist)
        return A, b, regs, dists
