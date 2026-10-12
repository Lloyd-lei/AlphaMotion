"""STRICT no-penetration / contact projection on top of UMR qpos.

Per frame t (sequential, warm-started from the previous frame's correction):
  verify(q):  ground clearance >= -PEN_TOL everywhere, object SDF >= -PEN_TOL, self distance >= -PEN_TOL,
              every REQUIRED contact region r: min clearance in [0, CONTACT_TOL]  (ground and/or object).
  If the UMR pose passes, it is kept bit-exact.  Otherwise iterate a QP (Clarabel):
     min  |W (q (+) dq (-) q_umr)|^2 + w_t |q (+) dq (-) q_prev|^2 + w_a |J_xy dq + (p_xy - latch_xy)|^2 + damp |dq|^2
     s.t. linearised clearances >= TARGET_LO (penetration rows, active within ACT),
          required-contact lowest point <= TARGET_HI, joint limits, |dq|_inf <= STEP.
  until verify passes (<= MAX_IT iterations); then fall back to a slack-relaxed QP; if still failing the frame is
  FLAGGED (bitmask) -- never silently passed.
Temporal pass: corrections are Gaussian-blended into neighbouring frames, every touched frame re-verified/re-solved.
Slip latch: inside a labelled ground-stance segment the support points (lowest dense points of the region at t-1,
within 2 mm) keep their previous-frame xy (soft anchor), unless the human source slips in that region.
"""
from __future__ import annotations

import time

import clarabel
import mujoco
import numpy as np
from scipy import sparse

PEN_TOL = 0.0005        # 0.5 mm allowed penetration (spec)
CONTACT_TOL = 0.001     # contact must be within [0, 1 mm]
TARGET_LO = 0.0001      # QP aims slightly above 0 (linearisation margin)
TARGET_HI = 0.0008      # QP aims below 0.8 mm for required contacts
ACT = 0.03              # rows activated within 3 cm
MAX_IT = 30
MAX_GROUND_ROWS = 160
CONV_TOL = 2e-4
# correction limits (relative to the UMR pose) and correction-rate limits (per second)
DEV_ROOT_POS, DEV_ROOT_ROT, DEV_JOINT = 0.06, 0.15, 0.5
RATE_ROOT_POS, RATE_ROOT_ROT, RATE_JOINT = 0.30, 1.0, 2.0       # rad / m: stop refining when the QP step is below this
STEP = 0.15

FLAG_GROUND_PEN, FLAG_OBJ_PEN, FLAG_SELF_PEN, FLAG_GROUND_CONTACT, FLAG_OBJ_CONTACT = 1, 2, 4, 8, 16


class Projector:
    def __init__(self, geo, w_root_pos=3.0, w_root_rot=1.0, w_joint=1.0, w_temporal=3.0, w_anchor=200.0, damping=1e-4,
                 max_rows_per_region=24):
        self.g = geo
        m = geo.m
        self.m = m
        nv = m.nv
        w = np.full(nv, w_joint)
        if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            w[0:3] = w_root_pos; w[3:6] = w_root_rot
        self.w = w
        self.w_t, self.w_a, self.damp = w_temporal, w_anchor, damping
        self.corr_prev = None
        self.set_context(None, 30.0)
        self.max_rows = max_rows_per_region
        # joint limits in dof space
        self.lim = []
        for j in range(m.njnt):
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and m.jnt_limited[j]:
                self.lim.append((int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), float(m.jnt_range[j][0]), float(m.jnt_range[j][1])))

    # ---------------------------------------------------------------- helpers
    def diff(self, q_from, q_to):
        v = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, v, 1.0, q_from, q_to)
        return v

    def integ(self, q, dq):
        q = q.copy()
        mujoco.mj_integratePos(self.m, q, dq, 1.0)
        return q

    def clamp_limits(self, q):
        q = q.copy()
        for qa, _, lo, hi in self.lim:
            q[qa] = min(max(q[qa], lo), hi)
        return q

    def verify(self, q, t, req_ground, req_obj, meas=None):
        g = self.g
        mm = meas if meas is not None else g.measure(q, t)
        flags = 0
        if mm['ground_min'] < -PEN_TOL:
            flags |= FLAG_GROUND_PEN
        if mm['obj_min'] < -PEN_TOL:
            flags |= FLAG_OBJ_PEN
        if mm['self_min'] < -PEN_TOL:
            flags |= FLAG_SELF_PEN
        for r in np.flatnonzero(req_ground):
            v = mm['ground_reg_min'][r]
            if not (-PEN_TOL <= v <= CONTACT_TOL):
                flags |= FLAG_GROUND_CONTACT
        for r in np.flatnonzero(req_obj):
            v = mm['obj_reg_min'][r]
            if not (-PEN_TOL <= v <= CONTACT_TOL):
                flags |= FLAG_OBJ_CONTACT
        return flags, mm

    # ---------------------------------------------------------------- one QP step
    def _rows(self, mm, t, req_ground, req_obj):
        g = self.g
        A, b, kinds = [], [], []
        z = mm['z']; X = mm['X']
        # ground penetration rows: convex-hull vertices within ACT (dense points lie inside the hulls, so hull
        # vertices are sufficient and exact for a plane)
        sel = np.flatnonzero(z[g.ndense:] < ACT) + g.ndense
        if len(sel) > MAX_GROUND_ROWS:            # missed rows are caught by the exact verify and added next iteration
            sel = sel[np.argsort(z[sel])[:MAX_GROUND_ROWS]]
        if len(sel):
            J = g.point_jac(g.all_body[sel], X[sel])[:, 2, :]
            A.extend(J); b.extend(TARGET_LO - z[sel]); kinds += ['g'] * len(sel)
        # required ground contact: lowest point of the region must come down to <= TARGET_HI
        for r in np.flatnonzero(req_ground):
            k = np.flatnonzero(g.all_reg == r)
            i = k[np.argmin(z[k])]
            J = g.point_jac(g.all_body[[i]], X[[i]])[0, 2, :]
            A.append(-J); b.append(z[i] - TARGET_HI); kinds.append('gc')
        # object
        if mm['obj_d'] is not None:
            od, on = mm['obj_d'], mm['obj_n']
            act = np.flatnonzero(od < ACT)
            if len(act):
                sel = []
                geoms = g.all_geom[act]
                for gg in np.unique(geoms):
                    k = act[geoms == gg]
                    sel.extend(k[np.argsort(od[k])[:self.max_rows]].tolist())
                sel = np.array(sel)
                J = g.point_jac(g.all_body[sel], X[sel])
                Jn = np.einsum('ni,niv->nv', on[sel], J)
                A.extend(Jn); b.extend(TARGET_LO - od[sel]); kinds += ['o'] * len(sel)
            for r in np.flatnonzero(req_obj):
                k = np.flatnonzero(g.all_reg == r)
                if not np.isfinite(od[k]).any():
                    continue
                i = k[np.argmin(od[k])]
                J = g.point_jac(g.all_body[[i]], X[[i]])[0]
                A.append(-(on[i] @ J)); b.append(od[i] - TARGET_HI); kinds.append('oc')
        # self
        sa, sb = g.self_rows([p for p in mm['self_pairs'] if p[2] < ACT], TARGET_LO)
        A.extend(sa); b.extend(sb); kinds += ['s'] * len(sa)
        return (np.array(A).reshape(-1, self.m.nv), np.array(b), kinds)

    def _qp(self, q, q_ref, q_prev, anchors, A, b, kinds, soft=False, step=STEP):
        m = self.m; nv = m.nv
        # cost blocks: tracking, temporal, anchors
        Jc = [np.diag(self.w)]; rc = [self.w * self.diff(q_ref, q)]
        if self.corr_prev is not None and self.w_t > 0:
            # temporal term on the CORRECTION (q (-) q_umr), not on the pose: penalises changes of the correction
            # between frames (jitter / drift), never the motion itself; weight scales with fps^2 (per-second rate)
            wt = self.w_t * (self.fps / 30.0) ** 2
            Jc.append(np.sqrt(wt) * np.eye(nv)); rc.append(np.sqrt(wt) * (self.diff(q_ref, q) - self.corr_prev))
        if anchors:
            bodies = np.array([a[0] for a in anchors]); P = np.array([a[1] for a in anchors])
            tgt = np.array([a[2] for a in anchors])
            J = self.g.point_jac(bodies, P)[:, :2, :].reshape(-1, nv)
            Jc.append(np.sqrt(self.w_a) * J); rc.append(np.sqrt(self.w_a) * (P[:, :2] - tgt).reshape(-1))
        Jc = np.vstack(Jc); rc = np.concatenate(rc)
        P_ = Jc.T @ Jc + self.damp * np.eye(nv); qv = Jc.T @ rc
        # constraints: A dq >= b  ->  -A dq <= -b ; step box; joint limits
        G = [-A]; h = [-b]
        lo, hi = self._dq_box(q, q_ref, step)
        G.append(np.eye(nv)); h.append(hi)
        G.append(-np.eye(nv)); h.append(-lo)
        for qa, da, lo, hi in self.lim:
            row = np.zeros(nv); row[da] = 1
            G.append(row[None]); h.append([hi - q[qa]])
            G.append(-row[None]); h.append([q[qa] - lo])
        G = np.vstack(G); h = np.concatenate([np.atleast_1d(x) for x in h])
        nslack = len(b) if soft else 0
        if nslack:
            ntot = nv + nslack
            Pf = np.zeros((ntot, ntot)); Pf[:nv, :nv] = P_; Pf[nv:, nv:] = 1e6 * np.eye(nslack)
            qf = np.concatenate([qv, np.zeros(nslack)])
            Gf = np.zeros((len(G) + nslack, ntot)); Gf[:len(G), :nv] = G
            Gf[:len(b), nv:] = -np.eye(nslack)                    # -A dq - s <= -b
            Gf[len(G):, nv:] = -np.eye(nslack)                    # s >= 0
            hf = np.concatenate([h, np.zeros(nslack)])
            P_, qv, G, h = Pf, qf, Gf, hf
        st = clarabel.DefaultSettings(); st.verbose = False; st.max_iter = 200
        Pc = sparse.csc_matrix(np.triu(P_)); Gc = sparse.csc_matrix(G)
        sol = clarabel.DefaultSolver(Pc, qv, Gc, h, [clarabel.NonnegativeConeT(len(h))], st).solve()
        ok = str(sol.status) in ('Solved', 'AlmostSolved')
        x = np.asarray(sol.x)[:nv] if ok else None
        return x, ok

    def _dq_box(self, q, q_ref, step):
        """Per-dof box on dq: trust region, total deviation from the UMR pose (DEV) and change of the correction
        w.r.t. the previous output frame (RATE, per second).  A required contact that cannot be met inside this
        box stays unmet and the frame is FLAGGED rather than producing a jump."""
        c = self.diff(q_ref, q)
        lo = np.maximum(-step, -self.dev - c); hi = np.minimum(step, self.dev - c)
        if self.corr_prev is not None:
            lo = np.maximum(lo, self.corr_prev - self.rate - c); hi = np.minimum(hi, self.corr_prev + self.rate - c)
        lo = np.minimum(lo, 0.0); hi = np.maximum(hi, 0.0)
        return lo, hi

    def set_context(self, corr_prev, fps):
        self.corr_prev = corr_prev
        self.fps = float(fps)
        nv = self.m.nv
        dev = np.full(nv, DEV_JOINT); rate = np.full(nv, RATE_JOINT / fps)
        if self.m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            dev[0:3] = DEV_ROOT_POS; dev[3:6] = DEV_ROOT_ROT
            rate[0:3] = RATE_ROOT_POS / fps; rate[3:6] = RATE_ROOT_ROT / fps
        self.dev, self.rate = dev, rate

    def solve_frame(self, t, q_ref, q_init, q_prev, req_ground, req_obj, anchors_fn=None):
        """Returns (q, flags, iters, initial_flags)."""
        f0, mm = self.verify(q_ref, t, req_ground, req_obj)
        if f0 == 0:
            return q_ref.copy(), 0, 0, 0
        q = self.clamp_limits(q_init)
        flags, mm = self.verify(q, t, req_ground, req_obj)
        it = 0
        step = STEP
        prev_v = self._violation(mm, req_ground, req_obj)
        min_it = 2 if anchors_fn is not None else 0          # latched stance: always let the anchors act
        v_ref = self._violation(self.verify(q_ref, t, req_ground, req_obj)[1], req_ground, req_obj)
        best_overall = (prev_v, q.copy(), flags) if prev_v < v_ref else (v_ref, q_ref.copy(), f0)
        last_step = np.inf
        while (flags or it < min_it or last_step > CONV_TOL) and it < MAX_IT:
            # keep iterating after feasibility until the QP step vanishes: the converged (unique) optimum varies
            # smoothly from frame to frame, whereas "first feasible iterate" jitters (seen as fake slip at 90 fps)
            anchors = anchors_fn(self.g) if anchors_fn else []
            self.g.set(q)
            A, b, kinds = self._rows(mm, t, req_ground, req_obj)
            dq, ok = self._qp(q, q_ref, q_prev, anchors, A, b, kinds, soft=False, step=step)
            if not ok:                                   # linearised problem infeasible within the step box
                dq, ok = self._qp(q, q_ref, q_prev, anchors, A, b, kinds, soft=True, step=step)
            if not ok:
                break
            best = None
            last_step = float(np.abs(dq).max())
            if not flags:                                # feasible already: accept the step only if still feasible
                for a in (1.0, 0.5):
                    qn = self.clamp_limits(self.integ(q, a * dq))
                    fn, mn = self.verify(qn, t, req_ground, req_obj)
                    if fn == 0:
                        q, mm = qn, mn
                        break
                else:
                    last_step = 0.0                      # cannot improve without leaving the feasible set: stop
                it += 1
                continue
            for a in (1.0, 0.5, 0.25):
                qn = self.clamp_limits(self.integ(q, a * dq))
                fn, mn = self.verify(qn, t, req_ground, req_obj)
                viol = self._violation(mn, req_ground, req_obj)
                if best is None or viol < best[0]:
                    best = (viol, qn, fn, mn)
                if fn == 0:
                    break
            v, q, flags, mm = best
            if best_overall is None or v < best_overall[0]:
                best_overall = (v, q, flags)
            if v >= prev_v * 0.98:                       # no progress -> shrink trust region (stops oscillation)
                step = max(step * 0.5, 0.005)
            prev_v = v
            it += 1
        if flags and best_overall[0] < self._violation(self.verify(q, t, req_ground, req_obj)[1], req_ground, req_obj):
            _, q, flags = best_overall                   # never return something worse than the best seen (incl. UMR)
        return q, flags, it, f0

    def viol(self, q, t, rg, ro):
        return self._violation(self.verify(q, t, rg, ro)[1], rg, ro)

    @staticmethod
    def _violation(mm, req_ground, req_obj):
        v = max(0.0, -mm['ground_min'] - PEN_TOL) + max(0.0, -mm['obj_min'] - PEN_TOL if np.isfinite(mm['obj_min']) else 0.0)
        v += max(0.0, -mm['self_min'] - PEN_TOL) if np.isfinite(mm['self_min']) else 0.0
        for r in np.flatnonzero(req_ground):
            x = mm['ground_reg_min'][r]; v += max(0.0, x - CONTACT_TOL) + max(0.0, -x - PEN_TOL)
        for r in np.flatnonzero(req_obj):
            x = mm['obj_reg_min'][r]
            if np.isfinite(x):
                v += max(0.0, x - CONTACT_TOL) + max(0.0, -x - PEN_TOL)
        return v


def run_projection(proj, Q_ref, req_ground, req_obj, latch_regions=None, latch_strength=None, smooth_sigma=2.0,
                   log_every=0, fps=30.0):
    """Q_ref [T,nq]; req_* [T,R] bool.
    latch_regions [T,R] bool: stance regions whose support points are anchored (xy) frame to frame.
    latch_strength [T,R] in [0,1]: anchor target = out_xy(t-1) + (1-s) * (UMR motion of the same material points);
      s=1 pins the points (removes UMR slip), s=0 reproduces UMR's tangential velocity (adds none).  Any xy offset
      accumulated while pinned is carried through the rest of the stance and only released while the region is
      airborne, so latch transitions never create a slip spike.
    Returns Q_out, flags [T], flags_initial [T], info dict."""
    T = len(Q_ref)
    m = proj.m
    Q = Q_ref.copy()
    flags = np.zeros(T, np.int32); flags0 = np.zeros(T, np.int32); iters = np.zeros(T, np.int32)
    g = proj.g
    t0 = time.time()
    latch = latch_regions if latch_regions is not None else np.zeros_like(req_ground)
    lam = latch_strength if latch_strength is not None else latch.astype(float)
    prev_corr = np.zeros(m.nv)

    def anchors_for(t):
        if t == 0 or not latch[t].any():
            return None
        regs = np.flatnonzero(latch[t] & latch[t - 1])
        if not len(regs):
            return None
        g.set(Q[t - 1])
        z, X, _ = g.ground()
        sel, lam_of = [], []
        for r in regs:
            k = np.flatnonzero(g.all_reg[:g.ndense] == r)
            if not len(k):
                continue
            zk = z[k]
            sup = k[zk <= zk.min() + 0.002][:6]
            sel.extend(sup.tolist()); lam_of.extend([float(lam[t, r])] * len(sup))
        if not sel:
            return None
        sel = np.array(sel); lam_of = np.array(lam_of)
        Xo = X[sel, :2].copy()
        g.set(Q_ref[t - 1]); Xr0 = g.world(sel)[:, :2]
        g.set(Q_ref[t]); Xr1 = g.world(sel)[:, :2]
        tgt = Xo + (1.0 - lam_of)[:, None] * (Xr1 - Xr0)

        def fn(geo):
            Xc = geo.world(sel)
            return [(int(geo.all_body[i]), Xc[j], tgt[j]) for j, i in enumerate(sel)]
        return fn

    for t in range(T):
        q_init = proj.integ(Q_ref[t], prev_corr)
        q_prev = Q[t - 1] if t > 0 else None
        proj.set_context(proj.diff(Q_ref[t - 1], Q[t - 1]) if t > 0 else None, fps)
        q, f, it, f0 = proj.solve_frame(t, Q_ref[t], q_init, q_prev, req_ground[t], req_obj[t], anchors_for(t))
        Q[t] = q; flags[t] = f; flags0[t] = f0; iters[t] = it
        prev_corr = proj.diff(Q_ref[t], q) if it > 0 else prev_corr * 0.5
        if log_every and t % log_every == 0:
            print(f'[postpass] t={t}/{T} fixed={int((flags0[:t+1] > 0).sum())} flagged={int((flags[:t+1] > 0).sum())} '
                  f'{(time.time()-t0)/(t+1)*1000:.1f} ms/frame', flush=True)
    t_seq = time.time() - t0
    # ---------------- temporal blending of corrections into neighbours
    n_resolved = 0
    if smooth_sigma > 0:
        C = np.stack([proj.diff(Q_ref[t], Q[t]) for t in range(T)])
        rad = int(3 * smooth_sigma)
        k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / smooth_sigma) ** 2)
        Cs = np.zeros_like(C)
        for j in range(C.shape[1]):
            num = np.convolve(C[:, j], k, mode='same')
            Cs[:, j] = num / np.convolve(np.ones(T), k, mode='same')
        # keep the larger-magnitude of (own correction, blended) per frame so fixes are not weakened
        for t in range(T):
            if t > 0 and (latch[t] & latch[t - 1]).any():
                # stance with latched support points: re-solve with anchors instead of blind blending
                proj.set_context(proj.diff(Q_ref[t - 1], Q[t - 1]), fps)
                q, f, it, _ = proj.solve_frame(t, Q_ref[t], proj.clamp_limits(proj.integ(Q_ref[t], Cs[t])), Q[t - 1],
                                               req_ground[t], req_obj[t], anchors_for(t))
                if f == 0 or proj.viol(q, t, req_ground[t], req_obj[t]) < proj.viol(Q[t], t, req_ground[t], req_obj[t]):
                    Q[t] = q; flags[t] = f
                n_resolved += 1
                continue
            cand = proj.clamp_limits(proj.integ(Q_ref[t], Cs[t]))
            fc, _ = proj.verify(cand, t, req_ground[t], req_obj[t])
            if fc == 0:
                Q[t] = cand; flags[t] = 0
            elif flags[t] == 0 and iters[t] == 0 and flags0[t] == 0:
                continue                                    # clean original stays
            else:
                proj.set_context(proj.diff(Q_ref[t - 1], Q[t - 1]) if t else None, fps)
                q, f, it, _ = proj.solve_frame(t, Q_ref[t], cand, Q[t - 1] if t else None, req_ground[t], req_obj[t],
                                               anchors_for(t))
                if f == 0 or proj.viol(q, t, req_ground[t], req_obj[t]) < proj.viol(Q[t], t, req_ground[t], req_obj[t]):
                    Q[t] = q; flags[t] = f
                n_resolved += 1
    info = dict(seconds_sequential=t_seq, seconds_total=time.time() - t0, ms_per_frame=(time.time() - t0) / T * 1000,
                frames_needing_fix=int((flags0 > 0).sum()), frames_flagged=int((flags > 0).sum()),
                flags_initial_hist={str(b): int(((flags0 & b) > 0).sum()) for b in (1, 2, 4, 8, 16)},
                flags_final_hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16)},
                iters_mean=float(iters[flags0 > 0].mean()) if (flags0 > 0).any() else 0.0, n_resolved_after_blend=n_resolved)
    return Q, flags, flags0, info
