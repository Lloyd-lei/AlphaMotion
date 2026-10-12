"""Exact-as-possible per-frame geometry of a posed robot.

Gauges (all metres, + = separated, - = penetrating):
  ground : min over (convex-hull vertices of every visual geom  U  4096 dense exterior points) of z - floor.
           The minimum of a linear function over a polytope is attained at a hull vertex, so for the visual meshes
           this is EXACT (not sampled).
  object : object SDF (analytic sphere/box, else mesh) at dense points U hull vertices near the object.  Sampled on
           the robot side: between-sample sag for a ball of radius R with sample spacing s is ~ s^2/(8R).
  self   : MuJoCo collision geoms, every pair MuJoCo considers (parent/child filtered, staged <exclude> list applied),
           mj_geomDistance (exact for primitives / convex hulls of collision meshes).
"""
from __future__ import annotations

import mujoco
import numpy as np

from surface_retargeting.robots.surface import REGIONS


def skew_batch(r):
    z = np.zeros(len(r))
    return np.stack([np.stack([z, -r[:, 2], r[:, 1]], 1), np.stack([r[:, 2], z, -r[:, 0]], 1),
                     np.stack([-r[:, 1], r[:, 0], z], 1)], 1)


class RobotGeometry:
    def __init__(self, m, surf, floor=0.0, obj=None, self_threshold=0.02):
        self.m = m
        self.d = mujoco.MjData(m)
        self.floor = float(floor)
        self.obj = obj
        self.self_thr = self_threshold
        self.pl = surf['points_local'].astype(np.float64); self.pg = surf['geom'].astype(np.int64)
        self.preg = surf['region'].astype(np.int64)
        self.hl = surf['hull_local'].astype(np.float64); self.hg = surf['hull_geom'].astype(np.int64)
        breg = surf['body_region']
        self.hreg = breg[m.geom_bodyid[self.hg]].astype(np.int64)
        self.pbody = m.geom_bodyid[self.pg].astype(np.int64)
        self.hbody = m.geom_bodyid[self.hg].astype(np.int64)
        self.nreg = len(REGIONS)
        # all candidate points: dense first, then hull
        self.all_local = np.concatenate([self.pl, self.hl]); self.all_geom = np.concatenate([self.pg, self.hg])
        self.all_reg = np.concatenate([self.preg, self.hreg]); self.all_body = np.concatenate([self.pbody, self.hbody])
        self.ndense = len(self.pl)
        self.rgeoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])], int)
        self._rset = set(self.rgeoms.tolist())
        self._fromto = np.zeros(6)
        self._jcache = {}

    # ------------------------------------------------------------------ kinematics
    def set(self, q):
        self.d.qpos[:] = q
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

    def self_pairs(self):
        """List of (g1, g2, dist, fromto) for robot collision pairs within self_thr (MuJoCo filters + excludes)."""
        m, d = self.m, self.d
        saved = m.geom_margin.copy()
        cand = set()
        try:
            m.geom_margin[:] = np.maximum(saved, self.self_thr)
            mujoco.mj_collision(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 in self._rset and c.geom2 in self._rset:
                    cand.add((min(c.geom1, c.geom2), max(c.geom1, c.geom2)))
        finally:
            m.geom_margin[:] = saved
        out = []
        for g1, g2 in sorted(cand):
            ft = np.zeros(6)
            dist = mujoco.mj_geomDistance(m, d, g1, g2, self.self_thr, ft)
            if dist < self.self_thr:
                out.append((g1, g2, float(dist), ft))
        return out

    def measure(self, q, t):
        """Full exact evaluation of one frame -> dict of scalars/arrays."""
        self.set(q)
        z, X, gmin = self.ground()
        od, on, omin = self.object(X, t)
        sp = self.self_pairs()
        smin = min([p[2] for p in sp], default=np.inf)
        return dict(z=z, X=X, ground_reg_min=gmin, ground_min=float(z.min()), obj_d=od, obj_n=on, obj_reg_min=omin,
                    obj_min=float(np.min(od)) if od is not None else np.inf, self_pairs=sp, self_min=smin)

    def self_rows(self, pairs, target):
        """Linear rows A dq >= b for collision pairs: dist + n.(J1-J2) dq >= target."""
        A, b = [], []
        for g1, g2, dist, ft in pairs:
            p1, p2 = ft[:3], ft[3:]
            dv = p1 - p2
            nn = np.linalg.norm(dv)
            if nn < 1e-9:
                continue
            u = dv / nn
            # sign by a finite-difference probe (MuJoCo's fromto orientation is not consistent for every
            # penetrating convex pair, e.g. box-cylinder): shift geom1 by +eps*u and see whether dist grows
            eps = 1e-4
            save = self.d.geom_xpos[g1].copy()
            self.d.geom_xpos[g1] = save + eps * u
            dp = mujoco.mj_geomDistance(self.m, self.d, g1, g2, self.self_thr, np.zeros(6))
            self.d.geom_xpos[g1] = save
            n = u if dp > dist else -u
            b1, b2 = int(self.m.geom_bodyid[g1]), int(self.m.geom_bodyid[g2])
            J1 = self.point_jac(np.array([b1]), p1[None])[0]
            J2 = self.point_jac(np.array([b2]), p2[None])[0]
            A.append(n @ (J1 - J2)); b.append(target - dist)
        return A, b
