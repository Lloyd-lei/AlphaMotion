"""Batched GPU re-implementation of UMR's per-frame retarget solve (solve_frame_body_segment_qp) + LQR filter.

Per frame, per sequence (exactly UMR's terms, weights and selection rules):
  residual rows   surface points (sqrt point_cost), transported T-pose normal offsets (sqrt normal_cost),
                  source self-contact map (sqrt(self_cost*w), active when robot pair distance < source distance),
                  ground-contact map (cost*strength on z), ground-contact anchors (cost*strength, xy latched at
                  activation, z=0), object contact vectors (cost*strength), smoothness (sqrt), temporal (sqrt),
                  damping;
  constraints     joint-limit box on dq, l2 trust region on local dofs (global_step_size), hard linearised ground
                  clearance on the 128 lowest of all robot surface slots below threshold, optional robot-object and
                  robot-self rows (point-sampled collision geometry, see geometry.py);
  update          mj_integratePos + joint clamp + quaternion normalisation; warm start = previous frame;
                  pose_init_iters Gauss-Newton steps on a sequence's first frame, `iters` afterwards.
The QP of each Gauss-Newton step is solved by the batched ADMM in qp.py (Clarabel in UMR).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from surface_retargeting.gpu.kin import KinTree, skew
from surface_retargeting.gpu import qp as _admm, ipm as _ipm


def _gather(x, idx):
    """x [B,N,...], idx [B,M] (>=0) -> [B,M,...]."""
    B = x.shape[0]
    return x[torch.arange(B, device=x.device)[:, None], idx]


def topk_mask(key, valid, k):
    """Mask of the (at most) k smallest `key` among `valid` (per row)."""
    k = min(k, key.shape[1]) if k > 0 else key.shape[1]
    big = torch.where(valid, key, torch.full_like(key, float('inf')))
    if k >= key.shape[1]:
        return valid.clone()
    _, idx = torch.topk(big, k, dim=1, largest=False)
    m = torch.zeros_like(valid)
    m.scatter_(1, idx, True)
    return m & valid


class SeqState:
    """Streaming state of a batch of sequences."""

    def __init__(self, prob, start, length, S, device):
        B = prob.shape[0]
        self.prob, self.start, self.length = prob, start, length
        self.q = None; self.q_prev = None; self.q_prev2 = None
        self.anchor_tgt = torch.zeros(B, S, 3, dtype=torch.float64, device=device)
        self.anchor_on = torch.zeros(B, S, dtype=torch.bool, device=device)


class UMRSolver:
    def __init__(self, bank, kin_dtype=torch.float32, geometry='auto', qp_iters=None, qp_eps=None, overrides=None,
                 qp='ipm'):
        self.bank = bank
        self.dev = bank.device
        self.a = dict(bank.args)
        if overrides:
            self.a.update(overrides)
        self.kt = KinTree(bank.model, self.dev, kin_dtype)
        self.kdt = kin_dtype
        self.qp = qp
        self.solve_qp = _ipm.solve_qp if qp == 'ipm' else _admm.solve_qp
        self.qp_iters = qp_iters or (40 if qp == 'ipm' else 400)
        self.qp_eps = qp_eps or (1e-9 if qp == 'ipm' else 1e-7)
        self.qp_iters_graph = 24 if qp == 'ipm' else self.qp_iters   # graph mode: fixed count, converged elements frozen
        kt = self.kt
        if geometry == 'auto':
            from surface_retargeting.gpu.geometry import UMRGeometry
            geometry = UMRGeometry(self.kt, bank, self.dev)
            if geometry.obj is None:
                geometry = None
        elif geometry is not None and not hasattr(geometry, 'umr_rows'):          # an ObjectSDF
            from surface_retargeting.gpu.geometry import UMRGeometry
            geometry = UMRGeometry.from_sdf(self.kt, bank, geometry, self.dev)
        self.geo = geometry
        # slot -> body + body-local position/normal (per problem)
        g = bank.tmpl_geom
        self.slot_body = kt.geom_body[g]                                        # [P,N]
        gR = kt.geom_R[g]; gp = kt.geom_pos[g]
        self.slot_blocal = ((gR @ bank.tmpl_local.to(kin_dtype)[..., None])[..., 0] + gp)
        self.slot_bnormal = (gR @ bank.tmpl_normal.to(kin_dtype)[..., None])[..., 0]
        nv = kt.nv
        self.nv = nv
        jq = torch.as_tensor(np.asarray(bank.problems[0]['joint_qpos_addrs'], np.int64), device=self.dev)
        jd = torch.as_tensor(np.asarray(bank.problems[0]['joint_dof_addrs'], np.int64), device=self.dev)
        self.jq, self.jd = jq, jd
        lo = bank.lim_lo[jq]; hi = bank.lim_hi[jq]
        self.j_lo, self.j_hi = lo, hi
        self.local_dofs = kt.local_dofs
        ball = torch.zeros(nv, dtype=torch.bool, device=self.dev)
        a = self.a
        mode = str(a['step_limit_mode']).lower()
        self.use_box = mode in ('box', 'box_l2'); self.use_l2 = mode in ('l2', 'box_l2')
        if self.use_l2:
            ball[self.local_dofs] = True
        self.ball = ball
        rmode = str(a.get('root_step_limit_mode', 'off')).lower()
        if rmode not in ('off', 'none', 'unlimited'):
            raise NotImplementedError('root_step_limit_mode != off')
        self.timers = {}
        self.debug_nan = False
        self.asm_dtype = torch.float32
        self._capturing = False
        self.qp_kred = 0
        self.qp_dtype = torch.float64
        self._graphs = {}
        self._softidx = {}

    # ------------------------------------------------------------------ kinematics helpers
    def slots_world(self, K, prob, idx=None):
        """World points/normals of slots idx [B,M] (all slots if None) -> P [B,M,3], body [B,M], Rb [B,M,3,3]."""
        body = self.slot_body[prob] if idx is None else torch.gather(self.slot_body[prob], 1, idx)
        bl = self.slot_blocal[prob] if idx is None else _gather(self.slot_blocal[prob], idx)
        Rb = _gather(K['xmat'], body)
        P = (Rb @ bl[..., None])[..., 0] + _gather(K['xpos'], body)
        return P, body, Rb, bl

    def clamp(self, q):
        q = q.clone()
        q[:, self.jq] = torch.minimum(torch.maximum(q[:, self.jq], self.j_lo), self.j_hi)
        if self.kt.free_body >= 0:
            q[:, 3:7] = q[:, 3:7] / q[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return q

    # ------------------------------------------------------------------ inputs of one frame
    def frame_inputs(self, st, fi, data=None):
        b = self.bank if data is None else data; a = self.a
        prob = st.prob
        selv = self.bank.sel_valid[prob]
        inp = dict(fi=fi, src=b.src_sel[fi].to(torch.float64), ntgt=b.ntgt[fi].to(torch.float64), selv=selv)
        thr = float(a['ground_contact_map_threshold']); maxp = int(a['ground_contact_map_max_points'])
        if b.has_ground:
            gd, gw = b.gd[fi], b.gw[fi]
            if float(a['ground_contact_map_cost']) > 0 and thr >= 0:
                m = topk_mask(gw, selv & (gd <= thr), maxp)
                inp['gm_mask'] = m
                inp['gm_cost'] = float(a['ground_contact_map_cost']) * ((thr - gw) / max(thr, 1e-8)).clamp(0.05, 1.0)
                inp['gd'] = gd
            if float(a['ground_contact_anchor_cost']) > 0:
                m = topk_mask(gw, selv & (gd.abs() <= 1e-8), maxp)
                inp['an_mask'] = m
                inp['an_cost'] = float(a['ground_contact_anchor_cost']) * ((thr - gw) / max(thr, 1e-8)).clamp(0.05, 1.0)
        if b.has_sc and float(a['self_contact_map_cost']) > 0:
            inp['sc_pairs'] = b.sc_pairs[fi]; inp['sc_d'] = b.sc_d[fi]; inp['sc_w'] = b.sc_w[fi]; inp['sc_dir'] = b.sc_dir[fi]
        if getattr(b, 'has_obj', False):
            inp['obj_pose'] = b.obj_pose[fi]
            if float(a.get('object_contact_map_cost', 0)) > 0:
                inp['oc_slot'] = b.oc_slot[fi]; inp['oc_d'] = b.oc_d[fi]; inp['oc_vec'] = b.oc_vec[fi]; inp['oc_pt'] = b.oc_pt[fi]
        return inp

    def update_anchors(self, st, q, inp, mask=None):
        if 'an_mask' not in inp:
            return
        K = self.kt.fk(q)
        idx = self.bank.sel[st.prob].clamp_min(0)
        P, _, _, _ = self.slots_world(K, st.prob, idx)
        new = inp['an_mask'] & ~st.anchor_on
        tgt = P.to(torch.float64).clone(); tgt[..., 2] = 0.0
        if mask is None:
            st.anchor_tgt = torch.where(new[..., None], tgt, st.anchor_tgt)
            st.anchor_on = inp['an_mask'].clone()
        else:                                   # in-place, graph-safe
            new = new & mask[:, None]
            st.anchor_tgt.copy_(torch.where(new[..., None], tgt, st.anchor_tgt))
            st.anchor_on.copy_(torch.where(mask[:, None], inp['an_mask'], st.anchor_on))

    # ------------------------------------------------------------------ one Gauss-Newton / QP step
    def gn_step(self, st, q, q_prev, q_prev2, has_prev, has_prev2, inp, live):
        a = self.a; kt = self.kt; nv = self.nv
        B = q.shape[0]
        f64 = torch.float64
        K = kt.fk(q)
        prob = st.prob
        idx = self.bank.sel[prob].clamp_min(0)
        Psel, bsel, Rb, _ = self.slots_world(K, prob, idx)
        Jp, Jr = kt.jac(K, Psel, bsel, rot=True)                      # [B,S,3,nv]
        adt = self.asm_dtype
        Psel = Psel.to(adt); Jp = Jp.to(adt); Jr = Jr.to(adt)
        rowsJ, rowsr = [], []
        g = None

        def add(Jrows, r):          # Jrows [B,R,nv], r [B,R]
            rowsJ.append(Jrows.to(adt)); rowsr.append(r.to(adt))

        S = Psel.shape[1]
        cv = lambda x: x.to(adt)
        # surface points + ground-contact anchors + ground-contact map share the slot Jacobians:
        #   one weighted Gram  sum_s,k W[s,k] J[s,k]^T J[s,k]  with W = w^2 + c_anchor^2 + [k==z] c_map^2
        w2 = cv(self.bank.point_cost[prob].clamp_min(0) * inp['selv'])[..., None].expand(B, S, 3)
        rw = w2 * (Psel - cv(inp['src']))
        if 'an_mask' in inp:
            c2 = cv((inp['an_cost'] * (inp['an_mask'] & st.anchor_on)) ** 2)[..., None]
            w2 = w2 + c2
            rw = rw + c2 * (Psel - cv(st.anchor_tgt))
        if 'gm_mask' in inp:
            cg2 = cv((inp['gm_cost'] * inp['gm_mask']) ** 2)
            zt = cv(torch.where(inp['gm_mask'], inp['gd'], torch.zeros_like(inp['gd'])))
            w2 = torch.cat([w2[..., :2], (w2[..., 2] + cg2)[..., None]], -1)
            rw = torch.cat([rw[..., :2], (rw[..., 2] + cg2 * (Psel[..., 2] - zt))[..., None]], -1)
        sw = torch.sqrt(w2)
        add((Jp * sw[..., None]).reshape(B, S * 3, nv),
            (rw / sw.clamp_min(1e-30) * (sw > 0)).reshape(B, S * 3))
        # surface normals (tpose_offset targets)
        wn = cv(torch.sqrt(self.bank.normal_cost[prob].clamp_min(0)) * inp['selv'])
        nb = (Rb @ _gather(self.slot_bnormal[prob], idx)[..., None])[..., 0].to(adt)
        N = nb / nb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Jn = -(skew(N) @ Jr)
        ntg = cv(inp['ntgt'])
        tg = ntg / ntg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        add((Jn * wn[..., None, None]).reshape(B, S * 3, nv), ((N - tg) * wn[..., None]).reshape(B, S * 3))
        # self-contact map
        if 'sc_pairs' in inp:
            pp = inp['sc_pairs']; valid = pp[..., 0] >= 0
            i0 = pp[..., 0].clamp_min(0); i1 = pp[..., 1].clamp_min(0)
            delta = _gather(Psel, i0) - _gather(Psel, i1)
            dist = delta.norm(dim=-1)
            scale = cv(math.sqrt(float(a['self_contact_map_cost'])) * torch.sqrt(inp['sc_w'].clamp_min(0)))
            scd = cv(inp['sc_d'])
            act = valid & (scale > 0) & (dist < scd)
            dirs = torch.where((dist > 1e-12)[..., None], delta / dist.clamp_min(1e-30)[..., None], cv(inp['sc_dir']))
            Jd = _gather(Jp, i0) - _gather(Jp, i1)                     # [B,K,3,nv]
            rows = (dirs[..., None, :] @ Jd)[..., 0, :] * (scale * act)[..., None]
            add(rows, scale * (dist - scd) * act)
        # object contact vectors
        if 'oc_slot' in inp:
            oc = inp['oc_slot']; ov = oc >= 0
            Po, bo, _, _ = self.slots_world(K, prob, oc.clamp_min(0))
            Jo = kt.jac(K, Po, bo).to(adt)
            thr = float(a['object_contact_map_threshold'])
            c = cv(float(a['object_contact_map_cost']) * ((thr - inp['oc_d']) / max(thr, 1e-8)).clamp(0.5, 1.0) * ov)
            Ko = oc.shape[1]
            add((Jo * c[..., None, None]).reshape(B, Ko * 3, nv),
                ((Po.to(adt) - cv(inp['oc_pt']) - cv(inp['oc_vec'])) * c[..., None]).reshape(B, Ko * 3))
        H = None
        for Jb, rb in zip(rowsJ, rowsr):            # per-block GEMMs (no concatenation of the ~4k x nv row matrix)
            JT = Jb.transpose(1, 2)
            hb = torch.baddbmm(H, JT, Jb) if H is not None else JT @ Jb
            H = hb
            gb = (JT @ rb[..., None])[..., 0]
            g = gb if not isinstance(g, torch.Tensor) or g is None else g + gb
        H = H.to(f64); g = g.to(f64)
        # smoothness / temporal (scalar joints)
        jq, jd = self.jq, self.jd
        s = float(a['smooth_cost']) * has_prev.to(f64)
        ts = float(a['temporal_smooth_cost']) * has_prev2.to(f64)
        if q_prev is not None:
            dqp = q[:, jq] - q_prev[:, jq]
            Hd = torch.zeros(B, nv, dtype=f64, device=self.dev)
            Hd[:, jd] = (s + ts)[:, None]
            H = H + torch.diag_embed(Hd)
            g[:, jd] += s[:, None] * dqp
            if q_prev2 is not None:
                g[:, jd] += ts[:, None] * (dqp - (q_prev[:, jq] - q_prev2[:, jq]))
        H = H + float(a['damping']) * torch.eye(nv, dtype=f64, device=self.dev)
        # ---------------- constraints
        lo = torch.full((B, nv), -float('inf'), dtype=f64, device=self.dev)
        hi = torch.full((B, nv), float('inf'), dtype=f64, device=self.dev)
        if self.use_box:
            md_ = torch.full((B, len(self.local_dofs)), float(a['max_dq']), dtype=f64, device=self.dev)
            lo[:, self.local_dofs] = -md_; hi[:, self.local_dofs] = md_
        jl = self.j_lo - q[:, jq]; jh = self.j_hi - q[:, jq]
        l2 = torch.maximum(lo[:, jd], jl); h2 = torch.minimum(hi[:, jd], jh)
        bad = l2 > h2
        if not self.use_box:
            cen = 0.5 * (jl + jh)
            l2 = torch.where(bad, cen, l2); h2 = torch.where(bad, cen, h2)
        else:
            md = float(a['max_dq'])
            below = q[:, jq] < self.j_lo
            fix = torch.where(below, torch.clamp(jl, max=md), torch.clamp(jh, min=-md))
            l2 = torch.where(bad, fix, l2); h2 = torch.where(bad, fix, h2)
        lo[:, jd] = l2; hi[:, jd] = h2
        Gs, hs, cs = [], [], []
        if bool(a['ground_penetration_hard_constraint']):
            mode = str(a['ground_penetration_hard_constraint_mode'])
            margin = float(a['ground_penetration_margin']); thr = float(a['ground_penetration_threshold'])
            if mode == 'surface_slots_all':
                Pall, _, _, _ = self.slots_world(K, prob)
                z = Pall[..., 2]
                cand = torch.ones_like(z, dtype=torch.bool)
            elif mode == 'surface_slots':
                z = Psel[..., 2].to(self.kdt); cand = inp['selv'].clone()
            else:
                raise NotImplementedError(f'ground mode {mode}')
            if thr >= 0:
                cand = cand & (z < margin + thr)
            k = int(a['ground_penetration_max_points'])
            k = min(k, z.shape[1]) if k > 0 else z.shape[1]
            key = torch.where(cand, z, torch.full_like(z, float('inf')))
            zz, gi = torch.topk(key, k, dim=1, largest=False)
            gv = torch.isfinite(zz)
            if mode == 'surface_slots_all':
                Pg, bg, _, _ = self.slots_world(K, prob, gi)
            else:
                gi_s = torch.gather(idx, 1, gi)
                Pg, bg, _, _ = self.slots_world(K, prob, gi_s)
            Jz = kt.jac(K, Pg, bg)[:, :, 2, :].to(f64)
            Gs.append(-Jz * gv[..., None])
            hs.append(torch.where(gv, Pg[..., 2].to(f64) - margin, torch.full_like(zz, float('inf'), dtype=f64)))
            sc = float(a['ground_penetration_hard_slack_cost']) if bool(a['ground_penetration_hard_slack']) else 0.0
            cs.append(sc)
        if self.geo is not None:
            for G_, h_, c_ in self.geo.umr_rows(self, K, q, inp, st):
                Gs.append(G_); hs.append(h_); cs.append(float(c_))
        G = torch.cat(Gs, 1) if Gs else None
        h = torch.cat(hs, 1) if hs else None
        cc = torch.cat([torch.full_like(h_, c_) for h_, c_ in zip(hs, cs)], 1) if Gs else None
        soft_idx, off = [], 0
        for h_, c_ in zip(hs, cs):
            if c_ > 0:
                soft_idx.extend(range(off, off + h_.shape[1]))
            off += h_.shape[1]
        key = tuple(soft_idx)
        if key not in self._softidx:
            self._softidx[key] = torch.as_tensor(soft_idx, dtype=torch.long, device=self.dev)
        soft_idx = self._softidx[key]
        res = self.solve_qp(H, g, lo, hi, G, h, cc, self.ball if self.use_l2 else None, float(a['global_step_size']),
                            max_iter=(self.qp_iters_graph if self._capturing else self.qp_iters), eps=self.qp_eps, eps_abs=self.qp_eps,
                            sync=not self._capturing, soft_idx=soft_idx, kred=self.qp_kred,
                            dtype=self.qp_dtype)
        dq = res.x * live[:, None].to(f64)
        return dq, res

    # ------------------------------------------------------------------ streaming over time
    def run(self, prob, start, length, teacher=False, log_every=0, store=True):
        """Solve B sequences in lock-step.  prob/start/length [B] (long, on device).
        teacher=True: every frame starts from UMR-CPU's own warm start/history (one-step parity)."""
        b = self.bank; a = self.a
        B = prob.shape[0]
        st = SeqState(prob, start, length, b.S, self.dev)
        Tmax = int(length.max())
        out = torch.zeros(B, Tmax, self.kt.nq, dtype=torch.float64, device=self.dev) if store else None
        pose_init = int(a['pose_init_iters']); it_n = int(a['iters'])
        stats = dict(qp_iters=[], qp_rprim=[])
        t0 = time.time()
        for t in range(Tmax):
            act = t < length
            fi = b.off[prob] + torch.minimum(start + t, b.T[prob] - 1)
            inp = self.frame_inputs(st, fi)
            if teacher:
                q0 = b.q_init[fi]
                hp = b.has_prev[fi]
                hp2 = hp & b.has_prev[(fi - 1).clamp_min(0)] & (t >= 0)
                qp1 = b.q_cpu[(fi - 1).clamp_min(0)]; qp2 = b.q_cpu[(fi - 2).clamp_min(0)]
                n_it = b.iters[fi]
            else:
                q0 = b.q_init[fi] if t == 0 else st.q
                hp = torch.full((B,), t >= 1, device=self.dev); hp2 = torch.full((B,), t >= 2, device=self.dev)
                qp1 = st.q_prev if t >= 1 else q0; qp2 = st.q_prev2 if t >= 2 else q0
                n_it = torch.full((B,), pose_init if (t == 0 and pose_init > 0) else it_n, device=self.dev)
            self.update_anchors(st, q0, inp)
            q = q0.clone()
            for k in range(int(n_it.max())):
                live = act & (k < n_it)
                dq, res = self.gn_step(st, q, qp1, qp2, hp, hp2, inp, live)
                stats['qp_iters'].append(res.iters); stats['qp_rprim'].append(float(res.r_prim.max()))
                q = torch.where(live[:, None], self.clamp(self.kt.integrate(q, dq)), q)
            if self.debug_nan and not bool(torch.isfinite(q).all()):
                bad = torch.nonzero(~torch.isfinite(q).all(-1)).flatten()
                raise FloatingPointError(f'non-finite q at t={t} elements={bad.tolist()[:10]}')
            if not teacher:
                st.q_prev2 = st.q_prev; st.q_prev = q
            st.q = q
            if store:
                out[:, t] = q
            if log_every and t % log_every == 0:
                torch.cuda.synchronize()
                print(f'[umr_gpu] t={t}/{Tmax} {(time.time() - t0) / (t + 1) * 1000:.2f} ms/step B={B}', flush=True)
        torch.cuda.synchronize()
        stats['seconds'] = time.time() - t0
        return out, stats

    # ------------------------------------------------------------------ CUDA-graph streaming (production path)
    def _make_graph(self, B, prob, start, length):
        dev = self.dev; nq = self.kt.nq
        bf = dict(
            fi=torch.zeros(B, dtype=torch.long, device=dev), hp=torch.zeros(B, dtype=torch.bool, device=dev),
            hp2=torch.zeros(B, dtype=torch.bool, device=dev), live=torch.zeros(B, dtype=torch.bool, device=dev),
            first=torch.zeros(B, dtype=torch.bool, device=dev),
            q=torch.zeros(B, nq, dtype=torch.float64, device=dev), qp1=torch.zeros(B, nq, dtype=torch.float64, device=dev),
            qp2=torch.zeros(B, nq, dtype=torch.float64, device=dev))
        st = SeqState(prob, start, length, self.bank.S, dev)

        def body():
            inp = self.frame_inputs(st, bf['fi'])
            self.update_anchors(st, bf['q'], inp, mask=bf['first'] & bf['live'])
            dq, _ = self.gn_step(st, bf['q'], bf['qp1'], bf['qp2'], bf['hp'], bf['hp2'], inp, bf['live'])
            bf['q'].copy_(torch.where(bf['live'][:, None], self.clamp(self.kt.integrate(bf['q'], dq)), bf['q']))

        # warm-up on a side stream with valid data, then capture
        b = self.bank
        bf['fi'].copy_(b.off[prob] + start); bf['q'].copy_(b.q_init[bf['fi']]); bf['qp1'].copy_(bf['q']); bf['qp2'].copy_(bf['q'])
        self._capturing = True
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        self._capturing = False
        return g, bf, st

    def run_graph(self, prob, start, length, log_every=0, store=True):
        """Free-running streaming solve with one CUDA graph replay per Gauss-Newton step."""
        b = self.bank; a = self.a
        B = prob.shape[0]
        g, bf, st = self._make_graph(B, prob, start, length)
        st.anchor_on.zero_(); st.anchor_tgt.zero_()
        Tmax = int(length.max())
        out = torch.zeros(B, Tmax, self.kt.nq, dtype=torch.float64, device=self.dev) if store else None
        pose_init = int(a['pose_init_iters']); it_n = int(a['iters'])
        torch.cuda.synchronize(); t0 = time.time()
        for t in range(Tmax):
            act = t < length
            bf['fi'].copy_(b.off[prob] + torch.minimum(start + t, b.T[prob] - 1))
            if t == 0:
                bf['q'].copy_(b.q_init[bf['fi']]); bf['qp1'].copy_(bf['q']); bf['qp2'].copy_(bf['q'])
            else:
                bf['qp2'].copy_(bf['qp1']); bf['qp1'].copy_(bf['q'])
            bf['hp'].fill_(t >= 1); bf['hp2'].fill_(t >= 2)
            n = pose_init if (t == 0 and pose_init > 0) else it_n
            for k in range(n):
                bf['first'].fill_(k == 0)
                bf['live'].copy_(act)
                g.replay()
            if store:
                out[:, t] = bf['q']
            if log_every and t % log_every == 0:
                torch.cuda.synchronize()
                print(f'[umr_gpu] t={t}/{Tmax} {(time.time() - t0) / (t + 1) * 1000:.2f} ms/step B={B}', flush=True)
        torch.cuda.synchronize()
        sec = time.time() - t0
        del g
        return out, dict(seconds=sec)


# ---------------------------------------------------------------------- LQR trajectory filter (UMR lqr_smooth)
def lqr_system(T, d, v, a, j):
    from scipy import sparse
    M = d * sparse.eye(T, format='csc')
    for order, c in ((1, v), (2, a), (3, j)):
        if c > 0 and T > order:
            coeffs = {1: (-1, 1), 2: (1, -2, 1), 3: (-1, 3, -3, 1)}[order]
            D = sparse.diags([np.full(T - order, cc, float) for cc in coeffs], offsets=np.arange(order + 1),
                             shape=(T - order, T), format='csc')
            M = M + c * (D.T @ D)
    return M.tocsc()


def lqr_filter(Q, lengths, cols, a, clamp_fn):
    """Q [B,T,nq] fp64 (padded), lengths [B].  Same linear system as UMR's lqr_smooth_qpos_sequence (data /
    velocity / acceleration / jerk costs, anchored first/last frames, root translation + scalar joints, joint
    clamp afterwards); solved per distinct length with a banded Cholesky over all columns of all sequences."""
    from scipy.linalg import cholesky_banded, cho_solve_banded
    data_c = max(float(a['trajectory_filter_data_cost']), 1e-12)
    vel = max(float(a['trajectory_filter_velocity_cost']), 0.0)
    acc = max(float(a['trajectory_filter_acceleration_cost']), 0.0)
    jerk = max(float(a['trajectory_filter_jerk_cost']), 0.0)
    if str(a['trajectory_filter_mode']).lower() != 'lqr' or (vel <= 0 and acc <= 0 and jerk <= 0):
        return Q
    cols = sorted(set(([0, 1, 2] if bool(a['trajectory_filter_root_translation']) else []) + [int(c) for c in cols]))
    s0 = max(int(a['trajectory_filter_anchor_start_frames']), 0); e0 = max(int(a['trajectory_filter_anchor_end_frames']), 0)
    out = Q.clone()
    Lc = lengths.cpu().numpy()
    for T in np.unique(Lc):
        T = int(T)
        if T <= 1:
            continue
        bi = np.flatnonzero(Lc == T)
        M = lqr_system(T, data_c, vel, acc, jerk)
        fixed = np.array(sorted(set(list(range(min(s0, T))) + list(range(max(T - e0, 0), T)))), np.int64)
        free = np.setdiff1d(np.arange(T), fixed)
        if len(free) == 0:
            continue
        bidx = torch.as_tensor(bi, device=Q.device)
        X = Q[bidx][:, :T][:, :, cols]
        Xn = X.permute(1, 0, 2).reshape(T, -1).cpu().numpy()
        Mr = M[free]
        rhs = data_c * Xn[free] - (Mr[:, fixed] @ Xn[fixed] if len(fixed) else 0.0)
        A = Mr[:, free].todia()
        bw = 3
        n = len(free)
        ab = np.zeros((bw + 1, n))
        Ad = A.tocsr()
        for k in range(bw + 1):
            ab[bw - k, k:] = Ad.diagonal(k)
        cb = cholesky_banded(ab)
        Xn[free] = cho_solve_banded((cb, False), rhs)
        Xs = torch.as_tensor(Xn.reshape(T, len(bi), len(cols)).transpose(1, 0, 2).copy(), device=Q.device, dtype=Q.dtype)
        sub = out[bidx]
        sub[:, :T, cols] = Xs
        out[bidx] = sub
    B, Tm, nq = out.shape
    return clamp_fn(out.reshape(B * Tm, nq)).reshape(B, Tm, nq)


class LQRGPU:
    """UMR's LQR trajectory filter for fixed-length windows on the GPU: the free block of the (anchored) system is
    inverted once (fp64) and applied to all columns of all windows with one GEMM."""

    def __init__(self, a, T, device='cuda'):
        from scipy import sparse  # noqa: F401
        self.a = a; self.T = T
        d = max(float(a['trajectory_filter_data_cost']), 1e-12)
        v = max(float(a['trajectory_filter_velocity_cost']), 0.0)
        ac = max(float(a['trajectory_filter_acceleration_cost']), 0.0)
        j = max(float(a['trajectory_filter_jerk_cost']), 0.0)
        self.on = str(a['trajectory_filter_mode']).lower() == 'lqr' and (v > 0 or ac > 0 or j > 0) and T > 1
        if not self.on:
            return
        M = lqr_system(T, d, v, ac, j).toarray()
        s0 = max(int(a['trajectory_filter_anchor_start_frames']), 0); e0 = max(int(a['trajectory_filter_anchor_end_frames']), 0)
        fixed = np.array(sorted(set(list(range(min(s0, T))) + list(range(max(T - e0, 0), T)))), np.int64)
        free = np.setdiff1d(np.arange(T), fixed)
        Ainv = np.linalg.inv(M[np.ix_(free, free)])
        W = np.zeros((T, T))
        W[np.ix_(free, free)] = Ainv * d
        if len(fixed):
            W[np.ix_(free, fixed)] = -Ainv @ M[np.ix_(free, fixed)]
            W[fixed, fixed] = 1.0
        self.W = torch.as_tensor(W, device=device, dtype=torch.float64)       # filtered = W @ raw (per column)

    def __call__(self, Q, cols, clamp_fn):
        """Q [B,T,nq] -> filtered (root translation + scalar joint columns), joints clamped."""
        if not self.on:
            return Q
        a = self.a
        cols = sorted(set(([0, 1, 2] if bool(a['trajectory_filter_root_translation']) else []) + [int(c) for c in cols]))
        ci = torch.as_tensor(cols, device=Q.device)
        out = Q.clone()
        X = Q[:, :, ci]                                                     # [B,T,c]
        out[:, :, ci] = torch.einsum('st,btc->bsc', self.W, X)
        B, T, nq = out.shape
        return clamp_fn(out.reshape(B * T, nq)).reshape(B, T, nq)
