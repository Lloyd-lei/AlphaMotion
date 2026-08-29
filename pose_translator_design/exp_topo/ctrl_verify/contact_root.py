"""Feet-defined root trajectory (owner rule 0820: root gives heading only; feet define
height AND horizontal travel). Given qpos rows whose root xy is arbitrary, re-derive
root xyz so that the stance foot (lowest sole point) stays planted in the world.
Flight frames (human foot clearance > 0) fall back to the human root velocity."""
import numpy as np, mujoco as mj

class ContactRoot:
    def __init__(self, model, foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link")):
        self.model = model; self.data = mj.MjData(model); self._mesh = {}
        fb = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in foot_bodies]; self.fb = fb
        def under(b, anc):
            while b > 0:
                if b == anc: return True
                b = model.body_parentid[b]
            return False
        self.col = [g for g in range(model.ngeom) if model.geom_contype[g] > 0]
        self.foot_geoms = [[g for g in self.col if under(model.geom_bodyid[g], b)] for b in fb]
        assert all(self.foot_geoms), "foot geoms not found"

    def _low_pt(self, g):
        m = self.model; d = self.data; Rm = d.geom_xmat[g].reshape(3, 3); pm = d.geom_xpos[g]
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            if mid not in self._mesh:
                a0 = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]; self._mesh[mid] = m.mesh_vert[a0:a0 + nv].copy()
            P = self._mesh[mid] @ Rm.T + pm; return P[P[:, 2].argmin()]
        z = pm[2] - np.abs(Rm[2]).dot(m.geom_size[g]); return np.array([pm[0], pm[1], z])

    def probe(self, qp_row):
        """-> (lowest point of whole body xyz, [foot0 lowest xyz, foot1 lowest xyz]) with root at given qpos."""
        self.data.qpos[:] = qp_row; mj.mj_kinematics(self.model, self.data)
        pts = [self._low_pt(g) for g in self.col]; body_low = min(pts, key=lambda p: p[2])
        # xy = ankle body origin (stable under heel-toe roll), z = lowest sole vertex
        feet = []
        for b, gs in zip(self.fb, self.foot_geoms):
            zlo = min(self._low_pt(g)[2] for g in gs); o = self.data.xpos[b]
            feet.append(np.array([o[0], o[1], zlo]))
        return body_low, feet

    @staticmethod
    def human_stance(hfeet_w, fps, base_pct=10, h_tol=0.05, v_tol=0.6):
        """hfeet_w: (T, 2, K, 3) human foot keypoints world (m), index1 = [left,right].
        -> stance (T,) in {0,1,-1}: foot in contact with lowest speed; -1 = flight."""
        z = hfeet_w[..., 1] if False else hfeet_w[..., 2]
        zmin = z.min(2)                                            # (T,2) lowest point per foot
        base = np.percentile(zmin.min(1), base_pct)
        c = np.mean(hfeet_w, 2)                                    # (T,2,3) foot centre
        v = np.zeros(zmin.shape); v[1:] = np.linalg.norm(np.diff(c[..., [0, 1]], axis=0), axis=2) * fps
        contact = (zmin < base + h_tol) & (v < v_tol)
        st = np.full(len(z), -1)
        for t in range(len(z)):
            if contact[t].all(): st[t] = int(v[t].argmin())
            elif contact[t].any(): st[t] = int(contact[t].argmax())
        return st

    @staticmethod
    def infer_stance(feet, clear_m=None, w_h=1.5, switch_pen=0.012, both_bonus=0.004):
        """Viterbi over {L,R,B} using only the robot's own feet (root-at-origin FK).
        Cost = |implied root velocity change| (the true stance foot gives a smooth
        root velocity; the wrong one flips it) + height prior (stance foot lower)
        + switch penalty.  feet: (T,2,3).  -> stance (T,) in {0,1,2}, -1 where clear_m>0."""
        T = feet.shape[0]; fz = feet[:, :, 2]
        disp = -(feet[1:, :, :2] - feet[:-1, :, :2])                     # (T-1,2,2) implied root displacement per foot
        dispB = disp.mean(1, keepdims=True); disp = np.concatenate([disp, dispB], 1)  # (T-1,3,2)
        hmin = fz.min(1, keepdims=True)
        emis = np.zeros((T, 3)); emis[:, :2] = w_h * (fz - hmin); emis[:, 2] = w_h * np.abs(fz[:, 0] - fz[:, 1]) - both_bonus
        INF = 1e9; cost = np.zeros((T, 3)); back = np.zeros((T, 3), dtype=int); cost[0] = emis[0]
        for t in range(1, T):
            for s in range(3):
                best, arg = INF, 0
                for sp in range(3):
                    c = cost[t - 1, sp] + (np.linalg.norm(disp[t - 1, s] - disp[t - 2, sp]) if t >= 2 else 0.0) + (switch_pen if s != sp else 0.0)
                    if c < best: best, arg = c, sp
                cost[t, s] = best + emis[t, s]; back[t, s] = arg
        st = np.zeros(T, dtype=int); st[-1] = int(cost[-1].argmin())
        for t in range(T - 1, 0, -1): st[t - 1] = back[t, st[t]]
        if clear_m is not None: st[np.asarray(clear_m) > 1e-4] = -1
        return st

    def infer_stance_dir(self, qp, clear_m, fps, iters=3, h_tol=0.03, tie=0.25, prior=None):
        """Direction rule (EM): relative to the root, the stance foot moves AGAINST
        the travel direction, the swing foot along it.  Travel direction starts
        as the body's facing, then is re-estimated from the solved root velocity.
        Feet clearly higher than the other (> h_tol) are swing regardless.
        Both feet counted planted (2) when their relative velocities nearly agree.
        prior: optional (T,) labels; entries >= -1 are kept fixed (inherited from
        parent windows), -3 = free.  -> stance (T,) in {0,1,2,-1}."""
        from scipy.spatial.transform import Rotation as Rot
        fixed = None if prior is None else (np.asarray(prior) >= -1)
        T = len(qp); feet = []
        for t in range(T):
            row = qp[t].copy(); row[:3] = 0.0; _, ft = self.probe(row); feet.append(ft)
        feet = np.array(feet); fz = feet[:, :, 2]
        v = np.zeros((T, 2, 2)); v[1:] = (feet[1:, :, :2] - feet[:-1, :, :2]) * fps      # foot velocity relative to root
        R = Rot.from_quat(qp[:, 3:7][:, [1, 2, 3, 0]]).as_matrix(); d = R[:, :2, 0]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        st = None
        for _ in range(iters):
            proj = np.einsum("tfk,tk->tf", v, d)                                  # + along travel = swing
            st = np.where(proj[:, 0] < proj[:, 1], 0, 1)
            hi = fz - fz.min(1, keepdims=True); st[(hi[:, 0] > h_tol)] = 1; st[(hi[:, 1] > h_tol)] = 0
            both = (np.linalg.norm(v[:, 0] - v[:, 1], axis=1) < tie * max(np.linalg.norm(v, axis=2).mean(), 1e-3)) & (hi.max(1) < h_tol)
            st[both] = 2
            if clear_m is not None: st[np.asarray(clear_m) > 1e-4] = -1
            if fixed is not None: st[fixed] = np.asarray(prior)[fixed]
            sol, _ = self.solve(qp, clear_m if clear_m is not None else np.zeros(T), None, fps, stance=st.copy())
            rv = np.gradient(sol[:, :2], axis=0) * fps
            k = 9; pad = k // 2; rp = np.pad(rv, ((pad, pad), (0, 0)), mode="edge")
            rv = np.array([rp[i:i + k].mean(0) for i in range(T)])
            spd = np.linalg.norm(rv, axis=1, keepdims=True)
            d = np.where(spd > 0.05, rv / np.maximum(spd, 1e-6), d)                 # keep facing where (nearly) standing
        return st

    def solve(self, qp, clear_m, human_vel_xy, fps, contact_tol=0.015, stance=None):
        """qp: (T, nq) with hinge angles + root quat set; root xyz ignored.
        clear_m: (T,) human foot clearance above stance baseline (m) -> >0 means flight.
        human_vel_xy: (T, 2) scaled human root velocity (m/s), used in flight.
        Returns qp with root xyz rewritten, plus diagnostics."""
        T = len(qp); qp = qp.copy(); qp[:, :3] = 0.0
        if human_vel_xy is None: human_vel_xy = np.full((T, 2), np.nan)
        lows = []; feet = []
        for t in range(T):
            bl, ft = self.probe(qp[t]); lows.append(bl); feet.append(ft)
        feet = np.array(feet)           # (T, 2, 3) foot low points relative to root-at-origin
        zlow = np.array([b[2] for b in lows])
        # z: plant lowest point (median-filtered) + flight clearance
        zplant = -zlow + 0.002
        k = 5; pad = k // 2; zp = np.pad(zplant, pad, mode="edge")
        zf = np.array([np.median(zp[i:i + k]) for i in range(T)])
        zf = np.maximum(zf, zplant - 0.01)          # smoothing may never push a frame >1 cm into the floor
        # stance foot per frame: lower foot, only if within tol of body-low and not in flight
        fz = feet[:, :, 2]; both = np.zeros(T, dtype=bool)
        if isinstance(stance, str) and stance == "dir":
            stance = self.infer_stance_dir(qp, clear_m, fps); in_contact = stance >= 0
            both = stance == 2; stance[~in_contact] = 0; stance[both] = 0
        elif stance is None or (isinstance(stance, str) and stance == "viterbi"):
            stance = self.infer_stance(feet, clear_m); in_contact = stance >= 0
            both = stance == 2; stance[~in_contact] = 0; stance[both] = 0
        else:
            stance = np.asarray(stance).copy(); in_contact = stance >= 0
            both = stance == 2; stance[~in_contact] = 0; stance[both] = 0
        xy = np.zeros((T, 2)); vel = np.zeros((T, 2))
        for t in range(1, T):
            if in_contact[t] and in_contact[t - 1] and (both[t] or both[t - 1]):
                vel[t] = -(feet[t, :, :2] - feet[t - 1, :, :2]).mean(0) * fps
            elif in_contact[t] and in_contact[t - 1] and stance[t] == stance[t - 1]:
                s = stance[t]; vel[t] = -(feet[t, s, :2] - feet[t - 1, s, :2]) * fps
            elif in_contact[t] and in_contact[t - 1]:
                # stance switch: average of both feet pinned -> use the new stance foot's motion
                # but also the old one (double support): mean keeps continuity
                v_new = -(feet[t, stance[t], :2] - feet[t - 1, stance[t], :2]) * fps
                v_old = -(feet[t, stance[t - 1], :2] - feet[t - 1, stance[t - 1], :2]) * fps
                vel[t] = 0.5 * (v_new + v_old)
            else:
                hv = human_vel_xy[t] if t < len(human_vel_xy) else np.array([np.nan, np.nan])
                vel[t] = vel[t - 1] if np.isnan(hv).any() else hv
        # light smoothing of velocity (continuity > exact pin), then integrate
        xy[1:] = np.cumsum(vel[1:], 0) / fps
        qp[:, 0:2] = xy; qp[:, 2] = zf + clear_m
        # diagnostics: stance-foot world speed while in contact
        fw = feet[:, :, :2] + xy[:, None, :]
        sp = np.linalg.norm(np.diff(fw, axis=0), axis=2) * fps
        st_sp = np.array([sp[t - 1, stance[t]] for t in range(1, T) if in_contact[t] and in_contact[t - 1]])
        diag = dict(contact_frac=float(in_contact.mean()), stance_speed_p50=float(np.median(st_sp)) if len(st_sp) else 0.0,
                    stance_speed_p90=float(np.percentile(st_sp, 90)) if len(st_sp) else 0.0,
                    travel=float(np.linalg.norm(xy[-1])))
        return qp, diag
