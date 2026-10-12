"""Frame-parallel strict pass (throughput mode): every frame of every chain is projected independently and in
parallel (no previous-frame coupling, no slip latch), then corrections are Gaussian-blended within each piece and
re-verified.  Same gauges / rows / tolerances as strict.py (lloyd's spec); frames still failing are FLAGGED.
"""
from __future__ import annotations

import time

import torch

from surface_retargeting.gpu import throttle

FLAG_JUMP = 128      # temporal continuity: output velocity differs from the UMR output's by > 1 cm/frame (root) or
                     # > 0.1 rad/frame (any joint) or > 0.05 rad/frame (root orientation), scaled to 30 fps


def smooth_corrections(C, w, piece, lam):
    """argmin_X sum_t w_t |X_t - C_t|^2 + lam sum_t |X_{t+1} - X_t|^2 per column, no coupling across pieces.
    C [B,T,D], w [B,T] >= 0, piece [B,T] (-1 idle).  Tridiagonal (Thomas) solve, batched over B and D."""
    B, T, D = C.shape
    valid = piece >= 0
    nxt = torch.zeros(B, T, dtype=C.dtype, device=C.device)
    nxt[:, :-1] = ((piece[:, 1:] == piece[:, :-1]) & valid[:, :-1]).to(C.dtype)
    prv = torch.zeros_like(nxt); prv[:, 1:] = nxt[:, :-1]
    w = torch.where(valid, w.to(C.dtype), torch.ones_like(w, dtype=C.dtype)) + 1e-6
    diag = w + lam * (nxt + prv)
    off = -lam * nxt                                   # M[t, t+1]
    rhs = C * w[..., None]
    cp = torch.zeros(B, T, dtype=C.dtype, device=C.device); dp = torch.zeros_like(C)
    cp[:, 0] = off[:, 0] / diag[:, 0]; dp[:, 0] = rhs[:, 0] / diag[:, 0, None]
    for t in range(1, T):
        den = diag[:, t] - off[:, t - 1] * cp[:, t - 1]
        cp[:, t] = off[:, t] / den
        dp[:, t] = (rhs[:, t] - off[:, t - 1, None] * dp[:, t - 1]) / den[:, None]
    X = torch.zeros_like(C)
    X[:, T - 1] = dp[:, T - 1]
    for t in range(T - 2, -1, -1):
        X[:, t] = dp[:, t] - cp[:, t, None] * X[:, t + 1]
    return torch.where(valid[..., None], X, C)


def jump_gate(proj, Q, Qref, piece, fps=30.0, jq=None, tol_p=0.01, tol_r=0.05, tol_j=0.1):
    """[B,T] bool: frame t changes too much relative to the reference trajectory's own change (same piece)."""
    kt = proj.kt
    B, T, nq = Q.shape
    s = 30.0 / float(fps)
    same = torch.zeros(B, T, dtype=torch.bool, device=Q.device)
    same[:, 1:] = (piece[:, 1:] == piece[:, :-1]) & (piece[:, 1:] >= 0)
    dv = kt.difference(Q[:, :-1].reshape(-1, nq), Q[:, 1:].reshape(-1, nq)).reshape(B, T - 1, -1)
    du = kt.difference(Qref[:, :-1].reshape(-1, nq), Qref[:, 1:].reshape(-1, nq)).reshape(B, T - 1, -1)
    e = (dv - du).abs()
    bad = torch.zeros(B, T, dtype=torch.bool, device=Q.device)
    if kt.free_body >= 0:
        f0 = kt.free_dof0
        ep = e[..., f0:f0 + 3].norm(dim=-1); er = e[..., f0 + 3:f0 + 6].norm(dim=-1)
        bad[:, 1:] |= (ep > tol_p * s) | (er > tol_r * s)
    jd = kt.scalar_dadr
    bad[:, 1:] |= e[..., jd].amax(-1) > tol_j * s
    return bad & same

from surface_retargeting.gpu import labels as L
from surface_retargeting.strict.strict import _pick, _merge, PEN_TOL, seg_min
from surface_retargeting.gpu import slip as SL
from surface_retargeting.gpu.slip import DROP_MASK


def _assign(dst, rows, src):
    """Measurement dict dst (n elements) with elements `rows` replaced by src (len(rows) elements)."""
    n = None
    for v in dst.values():
        if torch.is_tensor(v) and v.dim():
            n = v.shape[0]; break
    out = {}
    for k, v in dst.items():
        w = src.get(k)
        if k == 'K':
            out['K'] = {}
            for kk, vv in v.items():
                x = vv.clone(); x[rows] = w[kk]; out['K'][kk] = x
        elif k == 'sent':
            gone = torch.zeros(n, dtype=torch.bool, device=rows.device); gone[rows] = True
            kd = ~gone[v['b']]
            e = {kk: torch.cat([v[kk][kd], w[kk]]) for kk in v if kk != 'b'}
            e['b'] = torch.cat([v['b'][kd], rows[w['b']]])
            out['sent'] = e
        elif torch.is_tensor(v) and v.dim() and v.shape[0] == n and torch.is_tensor(w):
            x = v.clone(); x[rows] = w; out[k] = x
        else:
            out[k] = v
    return out


def slip_gate(proj, Q, floor_bt, piece, fps=30.0, human_speed=None, chunk=2048, **rule):
    """No-slip verification of the OUTPUT (umr_gpu/slip.py ruler, lloyd 10-11 rule).  Per (frame t, region): robot
    slip speed between t-1 and t (same piece) = min tangential speed over the region's dense points within
    [-0.5, 1] mm of the floor at both frames (the region's contact / pivot point: rolling and pivoting about the
    contact point are not slip).  A region moving > 1 cm/s is LEGAL (bit 256) when the same human region slides
    (human_speed [B,T,R], same ruler on the SMPL-X support points) faster than slip.V_HUMAN and the robot is within
    max(slip.V_ABS_TOL, slip.K_REL x human speed); otherwise it is slip (bit 32).
    Returns (bad [B,T], follow [B,T], robot speed [B,T,R], bad region [B,T,R], follow region [B,T,R])."""
    B, T, nq = Q.shape
    dev = Q.device
    same = torch.zeros(B, T, dtype=torch.bool, device=dev)
    same[:, 1:] = (piece[:, 1:] == piece[:, :-1]) & (piece[:, 1:] >= 0)
    rs = torch.full((B, T, L.NR), float('nan'), device=dev)
    for b in range(B):
        if bool(same[b].any()):
            rs[b] = SL.robot_slip_speed(proj, Q[b], floor_bt[b], fps, valid_pair=same[b], chunk=chunk)
    hs = human_speed if human_speed is not None else torch.full_like(rs, float('nan'))
    badr, folr = SL.slip_rule(rs, hs.to(rs.dtype), **rule)
    return badr.any(-1) & same, folr.any(-1) & same, rs, badr, folr


def hysteresis_t(d, enter, leave, reset):
    """d [B,T,R] clearance, reset [B,T] (piece starts) -> contact [B,T,R] (factory rule, restarted at resets)."""
    B, T, R = d.shape
    ev = torch.where(d < enter, 1, torch.where((d > leave) | ~torch.isfinite(d), 0, -1)).to(torch.int8)
    ev = torch.where(reset[..., None], (d < enter).to(torch.int8), ev)
    idx = torch.where(ev >= 0, torch.arange(T, device=d.device)[None, :, None].expand(B, T, R), torch.zeros_like(ev, dtype=torch.long))
    idx = torch.cummax(idx, dim=1).values
    return torch.gather(ev, 1, idx) == 1


def drop_short_t(c, min_on, reset):
    """Remove True runs shorter than min_on (runs never cross a reset)."""
    B, T, R = c.shape
    ar = torch.arange(T, device=c.device)[None, :, None].expand(B, T, R)
    prev = torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], 1) & ~reset[..., None]
    nxt_reset = torch.cat([reset[:, 1:], torch.ones_like(reset[:, :1])], 1)
    nxt = torch.cat([c[:, 1:], torch.zeros_like(c[:, :1])], 1) & ~nxt_reset[..., None]
    start = torch.cummax(torch.where(c & ~prev, ar, torch.zeros_like(ar)), 1).values
    end = torch.flip(torch.cummin(torch.flip(torch.where(c & ~nxt, ar, torch.full_like(ar, T)), [1]), 1).values, [1])
    return c & ((end - start + 1) >= min_on)


def _sub_extras(ex, idx):
    out = {}
    for k, v in ex.items():
        if v is None:
            out[k] = None
        elif isinstance(v, tuple):
            out[k] = tuple(t[idx] for t in v)
        else:
            out[k] = v[idx]
    return out


def project_frames(proj, Qref, floor, pose, req_g, req_o, iters=30, chunk=8192, log=False, Qinit=None, extras=None):
    """Qref [F,nq] -> (Q [F,nq], flags [F], flags_ref [F], iters [F]).
    Round-based work queue over ALL frames: every round processes the frames that still fail, in full chunks (no
    per-chunk tails), with a fresh measurement at the current iterate.  Qinit: optional warm start (used if better
    than the reference).  extras: per-frame keyword tensors for measure/qp_step (latch, yaw, lift_g, lift_o).
    Per round: one QP step, line search alpha 1 then 0.5 (factory rule: accept if it passes or lowers the violation;
    else halve the trust region; stop when it falls below 5 mm)."""
    kt = proj.kt
    F = Qref.shape[0]
    dev = Qref.device
    Q = Qref.clone()
    extras = extras or {}
    flags = torch.zeros(F, dtype=torch.int32, device=dev); f0 = torch.zeros_like(flags)
    its = torch.zeros(F, dtype=torch.long, device=dev)
    viol = torch.zeros(F, dtype=torch.float32, device=dev)
    step = torch.full((F,), proj.step0, dtype=torch.float64, device=dev)
    sub = lambda t, ii: None if t is None else t[ii]

    def meas(q, ii):
        return proj.measure(q, floor[ii], sub(pose, ii), req_g[ii], req_o[ii], **_sub_extras(extras, ii))

    # round 0: reference (and warm start) for every frame
    for s in range(0, F, chunk):
        ii = torch.arange(s, min(F, s + chunk), device=dev)
        mm = meas(Qref[ii], ii)
        f0[ii] = mm['flags']; flags[ii] = mm['flags']; viol[ii] = mm['viol'].float()
        if Qinit is not None:
            jj = ii[mm['flags'] != 0]
            if jj.numel():
                qi = proj._clamp(Qinit[jj])
                mi = meas(qi, jj)
                better = (mi['viol'].float() < viol[jj]) | (mi['flags'] == 0)
                Q[jj] = torch.where(better[:, None], qi, Q[jj])
                flags[jj] = torch.where(better, mi['flags'], flags[jj]); viol[jj] = torch.where(better, mi['viol'].float(), viol[jj])
        throttle.tick()
    active = torch.nonzero(flags != 0).flatten()
    for r in range(iters):
        if active.numel() == 0:
            break
        nxt = []
        for s in range(0, active.numel(), chunk):
            ii = active[s:s + chunk]
            q = Q[ii]
            mm = meas(q, ii)
            ex = _sub_extras(extras, ii)
            dq = proj.qp_step(q, Qref[ii], None, None, mm, req_g[ii], req_o[ii], None, step[ii], **ex)
            del mm
            qn = proj._clamp(kt.integrate(q, dq))
            mn = meas(qn, ii)
            fn = mn['flags']; vn = mn['viol'].float(); del mn
            better = (vn < viol[ii]) | (fn == 0)
            hb = torch.nonzero(~better).flatten()
            if hb.numel():
                ih = ii[hb]
                qh = proj._clamp(kt.integrate(q[hb], 0.5 * dq[hb]))
                mh = meas(qh, ih)
                bh = (mh['viol'].float() < viol[ih]) | (mh['flags'] == 0)
                qn[hb] = torch.where(bh[:, None], qh, qn[hb])
                fn[hb] = torch.where(bh, mh['flags'], fn[hb]); vn[hb] = torch.where(bh, mh['viol'].float(), vn[hb])
                better[hb] = bh
            Q[ii] = torch.where(better[:, None], qn, q)
            flags[ii] = torch.where(better, fn, flags[ii]); viol[ii] = torch.where(better, vn, viol[ii])
            step[ii] = torch.where(better, step[ii], (step[ii] * 0.5).clamp_min(0.005))
            its[ii] += 1
            keep = (flags[ii] != 0) & (step[ii] > 0.0051)
            nxt.append(ii[keep])
            throttle.tick()
        active = torch.cat(nxt)
    return Q, flags, f0, its


def _scatter_rows(m_sub, rows, n, like):
    """Measurement of a row subset -> measurement over n rows (other rows copied from `like`)."""
    out = {}
    for k, v in like.items():
        w = m_sub.get(k)
        if k == 'K':
            out['K'] = {kk: vv.clone().index_copy_(0, rows, w[kk]) for kk, vv in v.items()}
        elif k == 'sent':
            keep = ~torch.isin(v['b'], rows)
            e = {kk: torch.cat([vv[keep], w[kk] if kk != 'b' else rows[w['b']]]) for kk, vv in v.items()}
            out['sent'] = e
        elif torch.is_tensor(v) and v.dim() and v.shape[0] == n and w is not None:
            out[k] = v.clone().index_copy_(0, rows, w)
        else:
            out[k] = v
    return out


def blend_pieces(kt, Qref, Q, piece, sigma=2.0):
    """Gaussian blend (sigma frames) of corrections diff(Qref, Q) within each piece.  [B,T,nq] -> [B,T,nq]."""
    B, T, nq = Q.shape
    C = kt.difference(Qref.reshape(B * T, nq), Q.reshape(B * T, nq)).reshape(B, T, -1)
    rad = int(3 * sigma)
    num = torch.zeros_like(C); den = torch.zeros(B, T, 1, dtype=C.dtype, device=C.device)
    for o in range(-rad, rad + 1):
        w = float(torch.exp(torch.tensor(-0.5 * (o / sigma) ** 2)))
        if o >= 0:
            same = torch.zeros(B, T, dtype=torch.bool, device=C.device)
            same[:, :T - o] = (piece[:, o:] == piece[:, :T - o]) & (piece[:, :T - o] >= 0)
            sh = torch.zeros_like(C); sh[:, :T - o] = C[:, o:]
        else:
            same = torch.zeros(B, T, dtype=torch.bool, device=C.device)
            same[:, -o:] = (piece[:, :T + o] == piece[:, -o:]) & (piece[:, -o:] >= 0)
            sh = torch.zeros_like(C); sh[:, -o:] = C[:, :T + o]
        num += w * sh * same[..., None]; den += w * same[..., None]
    Cs = num / den.clamp_min(1e-12)
    return kt.integrate(Qref.reshape(B * T, nq), Cs.reshape(B * T, -1)).reshape(B, T, nq)


def _segments(mask, piece):
    """mask [B,T,R] bool -> (start, end) frame index of the run containing each True element (runs never cross a
    piece change)."""
    B, T, R = mask.shape
    tt = torch.arange(T, device=mask.device)[None, :, None].expand(B, T, R)
    same_prev = torch.cat([torch.zeros_like(piece[:, :1], dtype=torch.bool), piece[:, 1:] == piece[:, :-1]], 1)[..., None]
    same_next = torch.cat([piece[:, :-1] == piece[:, 1:], torch.zeros_like(piece[:, :1], dtype=torch.bool)], 1)[..., None]
    prevm = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], 1) & same_prev
    nextm = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], 1) & same_next
    st = torch.cummax(torch.where(mask & ~prevm, tt, torch.zeros_like(tt)), 1).values
    en = torch.flip(torch.cummin(torch.flip(torch.where(mask & ~nextm, tt, torch.full_like(tt, T - 1)), [1]), 1).values, [1])
    return st, en


def contact_latches(proj, Qref, req, piece, floor_bt, gam, pose=None, oam=None, req_o=None, chunk=65536):
    """One latched contact point per (region, contact segment): the region's lowest dense point at the segment's
    middle frame (UMR output).  Ground target: its xy there, z = floor + 0.45 mm; object target: the point at
    the middle frame in the object frame, projected to 0.45 mm off the surface, re-posed every frame.
    Returns latch tuple over [B*T, A] (A = R ground + R object)."""
    B, T, nq = Qref.shape
    R = req.shape[-1]
    dev = Qref.device
    Fq = Qref.reshape(B * T, nq)
    st, en = _segments(req, piece)
    mid = ((st + en) // 2)                                                   # [B,T,R]
    bi = torch.arange(B, device=dev)[:, None, None].expand(B, T, R)
    pid = gam[bi, mid, torch.arange(R, device=dev)[None, None]]               # lowest dense point at mid frame
    # positions of the segment's latched points at every frame of the segment -> target = segment mean xy
    # (least total displacement from the UMR output), z = floor + 0.45 mm
    def seg_targets(pp):
        ok = pp >= 0
        P = torch.empty(B, T, R, 3, dtype=torch.float64, device=dev)
        flat_b = bi.reshape(-1); flat_t = torch.arange(T, device=dev)[None, :, None].expand(B, T, R).reshape(-1)
        flat_p = pp.clamp_min(0).reshape(-1)
        for s in range(0, flat_b.numel(), chunk * 4):
            sl = slice(s, s + chunk * 4)
            K = proj.kt.fk(Qref[flat_b[sl], flat_t[sl]])
            P.view(-1, 3)[sl] = proj.pts.world(K, flat_p[sl][:, None])[:, 0].double()
        Pm = torch.where((req & ok)[..., None], P, torch.zeros_like(P))
        cs = torch.cat([torch.zeros_like(Pm[:, :1]), torch.cumsum(Pm, 1)], 1)
        ri = torch.arange(R, device=dev)[None, None]
        tg = (cs[bi, en + 1, ri] - cs[bi, st, ri]) / (en - st + 1).clamp_min(1)[..., None].double()
        tg[..., 2] = floor_bt[..., None].double() + 0.00045
        return tg
    tgt = seg_targets(pid)
    # yaw latch: heading of the latched point's body stays at its segment (circular) mean -> the contact patch
    # cannot spin about the latched point
    body = proj.pts.body[pid]                                                # [B,T,R]
    yaw = torch.empty(B, T, R, dtype=torch.float64, device=dev)
    for s in range(0, B * T, chunk):
        K = proj.kt.fk(Fq[s:s + chunk])
        Rm = K['xmat'][torch.arange(K['xmat'].shape[0], device=dev)[:, None], body.reshape(B * T, R)[s:s + chunk]]
        yaw.view(B * T, R)[s:s + chunk] = torch.atan2(Rm[..., 1, 0], Rm[..., 0, 0]).double()
    cz = torch.where(req, torch.cos(yaw), torch.zeros_like(yaw)); sz = torch.where(req, torch.sin(yaw), torch.zeros_like(yaw))
    ccs = torch.cat([torch.zeros_like(cz[:, :1]), torch.cumsum(cz, 1)], 1); scs = torch.cat([torch.zeros_like(sz[:, :1]), torch.cumsum(sz, 1)], 1)
    ri = torch.arange(R, device=dev)[None, None]
    yaw_t = torch.atan2(scs[bi, en + 1, ri] - scs[bi, st, ri], ccs[bi, en + 1, ri] - ccs[bi, st, ri])
    yaw_latch = (body.reshape(B * T, R), yaw_t.reshape(B * T, R), req.reshape(B * T, R))
    idx = pid; valid = req
    if proj.obj is not None and oam is not None and req_o is not None and pose is not None:
        from surface_retargeting.gpu.kin import quat_to_mat
        sto, eno = _segments(req_o, piece)
        mido = (sto + eno) // 2
        pido = oam[bi, mido, torch.arange(R, device=dev)[None, None]]
        Po = torch.empty(B, T, R, 3, dtype=torch.float32, device=dev)
        fb_, ft_, fp_ = bi.reshape(-1), mido.reshape(-1), pido.reshape(-1)
        for s in range(0, fb_.numel(), chunk * 4):
            sl = slice(s, s + chunk * 4)
            K = proj.kt.fk(Qref[fb_[sl], ft_[sl]])
            Po.view(-1, 3)[sl] = proj.pts.world(K, fp_[sl][:, None])[:, 0]
        PD = pose.shape[-1]
        pm = pose[bi, mido]                                                   # object pose at mid frame [B,T,R,PD]
        d, n = proj.obj.sdf(Po.reshape(-1, 1, 3), pm.reshape(-1, PD))
        Ps = Po.reshape(-1, 3) - (d.reshape(-1, 1) - 0.00045) * n.reshape(-1, 3)
        Rm = quat_to_mat(pm.reshape(-1, PD)[:, 3:7].float())
        loc = ((Ps - pm.reshape(-1, PD)[:, :3].float())[:, None, :] @ Rm)[:, 0, :]          # object-frame point
        pt = pose[:, :, None].expand(B, T, R, PD).reshape(-1, PD)              # pose at the current frame
        Rt = quat_to_mat(pt[:, 3:7].float())
        tgo = ((Rt @ loc[..., None])[..., 0] + pt[:, :3].float()).reshape(B, T, R, 3).double()
        idx = torch.cat([idx, pido], -1); tgt = torch.cat([tgt, tgo], -2); valid = torch.cat([valid, req_o], -1)
    A = idx.shape[-1]
    return (idx.reshape(B * T, A), tgt.reshape(B * T, A, 3), valid.reshape(B * T, A)), yaw_latch


def strict_chains(proj, Qref, piece, tl, floor_bt, human_g, human_o=None, pose=None, iters=30, chunk=8192, log=False,
                  human_clear_g=None, human_clear_o=None, latch=True, sweeps=2, sigma=3.0, fps=30.0, lam=4.0, eps_free=0.01):
    """Throughput-mode strict pass (DATA_SPEC_v2) on chain trajectories, all frames in parallel.
    Qref [B,T,nq] (UMR+LQR), piece/tl [B,T], floor_bt [B,T], human_g/o [B,T,R] human contact labels (same ruler),
    human_clear_g/o [B,T,R] human region clearance (for the lift rule: human > 2 cm off -> robot > 5 mm off)."""
    t0 = time.time()
    B, T, nq = Qref.shape
    dev = Qref.device
    valid = piece >= 0
    reset = valid & (tl == 0)
    Fq = Qref.reshape(B * T, nq); fl = floor_bt.reshape(-1)
    R = L.NR
    zero = torch.zeros(B * T, R, dtype=torch.bool, device=dev)
    PS = None if pose is None else pose.reshape(B * T, -1)
    sel = lambda t, i: None if t is None else t[i]
    # robot "before" clearances (UMR output, dense points) -> robot hysteresis contact
    greg = torch.empty(B * T, R, device=dev); gam = torch.empty(B * T, R, dtype=torch.long, device=dev)
    oreg = torch.full((B * T, R), float('inf'), device=dev); oam = torch.zeros(B * T, R, dtype=torch.long, device=dev)
    for s in range(0, B * T, chunk):
        sl = slice(s, s + chunk)
        mm = proj.measure(Fq[sl], fl[sl], sel(PS, sl), zero[sl], zero[sl], self_act=-1.0)
        greg[sl] = mm['greg']; gam[sl] = mm['gam']
        if proj.obj is not None:
            oreg[sl] = mm['oreg']; oam[sl] = mm['oam']
    greg = greg.reshape(B, T, R); oreg = oreg.reshape(B, T, R); gam = gam.reshape(B, T, R); oam = oam.reshape(B, T, R)
    has = proj.pts.has_reg[None, None] & valid[..., None]
    lift_g = (human_clear_g > 0.02) & has if human_clear_g is not None else torch.zeros_like(human_g)
    gin = torch.where(valid[..., None], greg, torch.full_like(greg, float('inf')))
    rc = drop_short_t(hysteresis_t(gin, 0.001, 0.002, reset), 3, reset)          # spec: enter < 1 mm, leave > 2 mm
    req_g = drop_short_t((rc | human_g) & ~lift_g, 3, reset) & has
    req_o = torch.zeros_like(req_g); lift_o = torch.zeros_like(req_g)
    if proj.obj is not None and human_o is not None:
        lift_o = (human_clear_o > 0.02) & has if human_clear_o is not None else torch.zeros_like(human_o)
        oin = torch.where(valid[..., None], oreg, torch.full_like(oreg, float('inf')))
        rco = drop_short_t(hysteresis_t(oin, 0.001, 0.002, reset), 3, reset)
        req_o = drop_short_t((rco | human_o) & ~lift_o, 3, reset) & has
    ex = dict(lift_g=lift_g.reshape(B * T, R), lift_o=lift_o.reshape(B * T, R) if proj.obj is not None else None)
    if latch:
        ex['latch'], ex['yaw'] = contact_latches(proj, Qref, req_g, piece, floor_bt, gam, pose, oam, req_o if proj.obj is not None else None)
    t1 = time.time()
    # two passes: frames t % stride == 0 first, the others warm-started from interpolated neighbour corrections
    stride = 4
    tt = torch.arange(T, device=dev)[None].expand(B, T)
    pa = ((tt % stride == 0) | (tl == 0)) & valid
    ia = torch.nonzero(pa.reshape(-1)).flatten()
    RG = req_g.reshape(B * T, R); RO = req_o.reshape(B * T, R)
    Qa, fa, f0a, ia_its = project_frames(proj, Fq[ia], fl[ia], sel(PS, ia), RG[ia], RO[ia], iters=iters, chunk=chunk,
                                         extras=_sub_extras(ex, ia))
    Qp = Fq.clone(); Qp[ia] = Qa
    C = proj.kt.difference(Fq, Qp).reshape(B, T, -1)
    pidx = torch.where(pa, tt, torch.full_like(tt, -1)); prev_a = torch.cummax(pidx, 1).values
    nidx = torch.where(pa, tt, torch.full_like(tt, T)); next_a = torch.flip(torch.cummin(torch.flip(nidx, [1]), 1).values, [1])
    pv = prev_a.clamp(0, T - 1); nx = next_a.clamp(0, T - 1)
    okp = (prev_a >= 0) & (piece.gather(1, pv) == piece); okn = (next_a < T) & (piece.gather(1, nx) == piece)
    Cp = torch.gather(C, 1, pv[..., None].expand(-1, -1, C.shape[-1])); Cn = torch.gather(C, 1, nx[..., None].expand(-1, -1, C.shape[-1]))
    w = ((tt - prev_a).float() / (next_a - prev_a).clamp_min(1).float()).double()[..., None]
    Ci = torch.where((okp & okn)[..., None], (1 - w) * Cp + w * Cn, torch.where(okp[..., None], Cp, torch.where(okn[..., None], Cn, torch.zeros_like(Cp))))
    Qi = proj.kt.integrate(Fq, Ci.reshape(B * T, -1))
    ib = torch.nonzero((~pa & valid).reshape(-1)).flatten()
    Qb_, fb_, f0b, ib_its = project_frames(proj, Fq[ib], fl[ib], sel(PS, ib), RG[ib], RO[ib], iters=iters, chunk=chunk,
                                           Qinit=Qi[ib], extras=_sub_extras(ex, ib))
    Qp[ib] = Qb_
    fl1 = torch.zeros(B * T, dtype=torch.int32, device=dev); fl1[ia] = fa; fl1[ib] = fb_
    f0 = torch.zeros_like(fl1); f0[ia] = f0a; f0[ib] = f0b
    its = torch.zeros(B * T, dtype=torch.long, device=dev); its[ia] = ia_its; its[ib] = ib_its
    Qp = Qp.reshape(B, T, nq)
    t2 = time.time()
    # temporal coupling: the correction field C(t) = q(t) - q_UMR(t) is replaced by the minimiser of
    #   sum_t a_t |C_t - C^proj_t|^2 + lam sum_t |C_{t+1} - C_t|^2         (per dof, inside each piece; banded solve)
    # with a_t = 1 where the projection had to correct the frame, eps where the UMR frame already satisfied the spec
    # (it follows its neighbours), 0 where the projection failed (interpolated) -> offsets ramp in and out over
    # ~sqrt(lam/eps) frames instead of snapping back; then every frame is re-projected tracking the smooth trajectory.
    Qcur = Qp
    fl1 = fl1.reshape(B, T) if fl1.dim() == 1 else fl1
    f0bt = f0.reshape(B, T)
    for k in range(sweeps):
        C = proj.kt.difference(Qref.reshape(B * T, nq), Qcur.reshape(B * T, nq)).reshape(B, T, -1)
        w = torch.where(fl1.reshape(B, T) != 0, torch.zeros_like(C[..., 0]),
                        torch.where(f0bt != 0, torch.ones_like(C[..., 0]), torch.full_like(C[..., 0], eps_free)))
        Cs = smooth_corrections(C, w, piece, lam)
        Qt = proj._clamp(proj.kt.integrate(Qref.reshape(B * T, nq), Cs.reshape(B * T, -1)))
        Qn, fn, _, itn = project_frames(proj, Qt, fl, PS, RG, RO, iters=iters, chunk=chunk, extras=ex)
        Qcur = Qn.reshape(B, T, nq)
        fl1 = fn.reshape(B, T)
        its = its + itn
    Qout = Qcur
    flags = fl1.reshape(B, T)
    # temporal-continuity gate (QA): per-frame change of the output vs the UMR output (which follows the human)
    jump = jump_gate(proj, Qout, Qref, piece, fps, jq=proj.jq)
    flags = flags | (jump.int() * FLAG_JUMP)
    flags = torch.where(valid, flags, torch.zeros_like(flags))
    t3 = time.time()
    take = torch.zeros(1)
    info = dict(labels_s=t1 - t0, project_s=t2 - t1, blend_s=t3 - t2, frames=int(valid.sum()),
                fixed=int(((f0.reshape(B, T) != 0) & valid).sum()), flagged=int((flags != 0).sum()), blended=int(take.sum()),
                iters_mean=float(its.reshape(B, T)[valid].float().mean()),
                flags_hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128)},
                jump_frames=int(jump.sum()), sweeps=sweeps)
    return Qout, flags, f0.reshape(B, T), req_g, req_o, dict(info, extras=ex, lift_g=lift_g, lift_o=lift_o)


# ============================================================================================ sequential, time-coupled
def strict_seq(proj, Qref, piece, tl, floor_bt, human_g, human_o=None, pose=None, iters=30, log=False,
               human_clear_g=None, human_clear_o=None, fps=30.0, decay=0.88, w_temporal=1.0, human_slip=None,
               human_speed=None, slide_speed=SL.V_HUMAN, max_dev=0.05, max_yaw_dev=0.26, dbg=None, gate_p=0.01, gate_r=0.05, gate_j=0.1, track_init=True, human_only=True, use_yaw=True, use_full=False,
               cont_p=0.0095, cont_r=0.0475, cont_j=0.095, latch_at='umr', decay_ladder=(), stall_k=8,
               leash_p=0.10, leash_r=0.26, leash_j=0.6, align_floor=True, cache_measure=True, zshift=None, roll_tol=0.002,
               relatch_at_output=True, yaw_follow_umr=True, human_slip_vec=None, slide_release=0.30, **_):
    """Strict pass that walks every chain forward in time (all chains in lock-step on the GPU, failing chains only):
      * warm start  q_init(t) = q_UMR(t) (+) decay * correction(t-1)  -> corrections ramp in/out, never snap back;
        if q_init(t) satisfies the spec it is kept;  otherwise QP iterations track q_init(t) (not q_UMR) with the
        temporal term to the previous output frame;
      * online latches (factory rule): on the first frame of a contact segment the region's lowest dense point is
        solved WITHOUT a latch; its resulting position (ground: xy, z = floor + 0.45 mm; object: in the object frame)
        becomes the target for the rest of the segment, together with that body's heading;
      * same gauges / rows / flags as the frame-parallel pass + the temporal-continuity gate.
    Slip (lloyd 10-11, umr_gpu/slip.py): human_speed [B,T,R] is the human region slip speed on the slip.py ruler.
      * human_slip_vec [B,T,R,2] (m/s, slip.py): while the human region slides (> slide_speed = slip.V_HUMAN) the
        ground latch target MOVES with the human's slip velocity (the robot slides like the human -> bit 256);
        below that it is fixed (jitter is held).  Above slide_release (30 cm/s, a sweep rather than a slide) the
        latch is released and the robot follows UMR.  Without human_slip_vec: latch held only while the human
        region does not slide (> slide_speed releases it);
      * a latch re-taken inside a contact segment (after a human slide, or a rolling switch) is placed where the
        OUTPUT had that point at t-1 (no pull back to UMR's point -> no catch-up slide);
      * rolling: when UMR's lowest point of a latched region moves away from the latched point (latched point
        > roll_tol above the region's lowest point in the UMR pose: heel -> toe), the latch moves to the new lowest
        point (held at its output position) instead of pinning the heel;
      * pivoting: the heading latch follows UMR's heading increment of that body (the human's pivot), instead of
        freezing the heading at touch-down;
      * final bit 32 / 256 come from the output gate (slip_gate) plus object-latch deviations; solver-internal ground
        latch / heading deviations are not slip by themselves (they only drive the QP).
    Returns like strict_chains."""
    from surface_retargeting.gpu.kin import quat_to_mat
    t0 = time.time()
    kt = proj.kt
    s30 = 30.0 / float(fps)
    cbox = torch.full((kt.nv,), cont_j * s30, dtype=torch.float64, device=Qref.device)
    if kt.free_body >= 0:
        f0_ = kt.free_dof0
        cbox[f0_:f0_ + 3] = cont_p * s30 / 3 ** 0.5; cbox[f0_ + 3:f0_ + 6] = cont_r * s30 / 3 ** 0.5
    leash = torch.full((kt.nv,), leash_j, dtype=torch.float64, device=Qref.device)
    if kt.free_body >= 0:
        leash[kt.free_dof0:kt.free_dof0 + 3] = leash_p; leash[kt.free_dof0 + 3:kt.free_dof0 + 6] = leash_r
    w_t_saved = proj.w_t
    proj.w_t = w_temporal
    B, T, nq = Qref.shape
    dev = Qref.device
    valid = piece >= 0
    reset = valid & (tl == 0)
    R = L.NR
    nd = proj.pts.ndense
    # ---- requirements (same rules as strict_chains)
    Fq = Qref.reshape(B * T, nq); fl_all = floor_bt.reshape(-1)
    zero = torch.zeros(B * T, R, dtype=torch.bool, device=dev)
    PS = None if pose is None else pose.reshape(B * T, -1)
    greg = torch.empty(B * T, R, device=dev); oreg = torch.full((B * T, R), float('inf'), device=dev)
    for s in range(0, B * T, 8192):
        sl = slice(s, s + 8192)
        mm = proj.measure(Fq[sl], fl_all[sl], None if PS is None else PS[sl], zero[sl], zero[sl], self_act=-1.0)
        greg[sl] = mm['greg']
        if proj.obj is not None:
            oreg[sl] = mm['oreg']
    greg = greg.reshape(B, T, R); oreg = oreg.reshape(B, T, R)
    # robot floor vs human floor (factory 'applied_shift'): UMR puts the feet on its own ground plane, the strict
    # pass uses the calibrated human floor -> shift every UMR frame of a clip by the median gap of the robot's feet
    # over the human's foot-contact frames (a constant offset instead of a correction re-solved every frame)
    floor_shift = torch.zeros(B, dtype=torch.float64, device=dev)
    if zshift is not None:
        floor_shift = zshift.to(torch.float64)
        Qref = Qref.clone(); Qref[..., 2] += floor_shift[:, None]
        greg = greg + floor_shift[:, None, None].float()
    elif align_floor:
        fi = [L.RID['L_foot'], L.RID['R_foot']]
        hf = human_g[..., fi] & valid[..., None]
        gf = greg[..., fi]
        for b in range(B):
            v_ = gf[b][hf[b] & torch.isfinite(gf[b])]
            if v_.numel() >= 10:
                floor_shift[b] = (-v_.double().median() + 0.0004).clamp(-0.05, 0.05)
        if bool((floor_shift != 0).any()):
            Qref = Qref.clone(); Qref[..., 2] += floor_shift[:, None]
            greg = greg + floor_shift[:, None, None].float()
    has = proj.pts.has_reg[None, None] & valid[..., None]
    lift_g = (human_clear_g > 0.02) & has if human_clear_g is not None else torch.zeros_like(human_g)
    inf = torch.full_like(greg, float('inf'))
    rc = drop_short_t(hysteresis_t(torch.where(valid[..., None], greg, inf), 0.001, 0.002, reset), 3, reset)
    # contact is required where the HUMAN region is in contact (same ruler); robot-only touches (UMR feet scraping)
    # are not required and not latched
    req_g = (drop_short_t(human_g & ~lift_g, 3, reset) & has) if human_only else (drop_short_t((rc | human_g) & ~lift_g, 3, reset) & has)
    req_o = torch.zeros_like(req_g); lift_o = torch.zeros_like(req_g)
    hoi = proj.obj is not None and human_o is not None
    if hoi:
        lift_o = (human_clear_o > 0.02) & has if human_clear_o is not None else torch.zeros_like(human_o)
        rco = drop_short_t(hysteresis_t(torch.where(valid[..., None], oreg, inf), 0.001, 0.002, reset), 3, reset)
        req_o = (drop_short_t(human_o & ~lift_o, 3, reset) & has) if human_only else (drop_short_t((rco | human_o) & ~lift_o, 3, reset) & has)
    t1 = time.time()
    # ---- state
    Q = Qref.clone()
    flags = torch.zeros(B, T, dtype=torch.int32, device=dev); f0 = torch.zeros_like(flags)
    its = torch.zeros(B, T, dtype=torch.long, device=dev)
    corr = torch.zeros(B, kt.nv, dtype=torch.float64, device=dev)
    A = 2 * R if hoi else R
    l_on = torch.zeros(B, A, dtype=torch.bool, device=dev)
    l_idx = torch.zeros(B, A, dtype=torch.long, device=dev)
    l_tgt = torch.zeros(B, A, 3, dtype=torch.float64, device=dev)         # ground: world; object: object frame
    l_start = torch.zeros(B, A, dtype=torch.long, device=dev)
    y_on = torch.zeros(B, R, dtype=torch.bool, device=dev); y_body = torch.zeros(B, R, dtype=torch.long, device=dev)
    y_tgt = torch.zeros(B, R, dtype=torch.float64, device=dev)
    ar = torch.arange(B, device=dev)
    dreg = proj.pts.region[:nd]
    slip = torch.zeros(B, T, dtype=torch.bool, device=dev)
    n_release = [0]
    prev_idx = torch.zeros(B, R, dtype=torch.long, device=dev); prev_pos = torch.zeros(B, R, 3, device=dev)
    prev_req = torch.zeros(B, R, dtype=torch.bool, device=dev)
    prevX = None; prevXm = None                                       # output dense points / body xmat at t-1
    rel_drift = torch.zeros(B, R, dtype=torch.bool, device=dev)      # last release was a drift release (re-latch at UMR)
    y_umr_prev = torch.zeros(B, R, dtype=torch.float64, device=dev)  # UMR heading of the yaw-latched body at t-1
    obj_bad = torch.zeros(B, T, dtype=torch.bool, device=dev)
    n_roll = [0]; n_relatch_out = [0]
    wrap = lambda a: torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
    for t in range(T):
        act = valid[:, t]
        if not bool(act.any()):
            continue
        qr = Qref[:, t]
        corr = torch.where(reset[:, t][:, None], torch.zeros_like(corr), corr)
        q_full = proj._clamp(kt.integrate(qr, corr))                 # previous correction carried (warm start)
        # temporal continuity: prediction = previous output moved by the UMR increment; the frame may deviate from it
        # by at most cbox per dof (root pos / root rot / joints), the warm start relaxes toward UMR within that box
        first = (tl[:, t] == 0) | (t == 0)
        if t > 0:
            q_pred = kt.integrate(Q[:, t - 1], kt.difference(Qref[:, t - 1], qr))
        else:
            q_pred = qr
        q_pred = torch.where(first[:, None], qr, q_pred)
        q_dec = kt.integrate(qr, decay * corr)
        step_dec = kt.difference(q_pred, q_dec).clamp(-cbox, cbox)
        q_init = proj._clamp(kt.integrate(q_pred, step_dec))
        q_init = torch.where(first[:, None], proj._clamp(qr), q_init)
        rg = req_g[:, t]; ro = req_o[:, t]; fl = floor_bt[:, t]
        ps = None if pose is None else pose[:, t]
        # latch unless the human region slides (slip.py ruler > slide_speed): a sliding human region is followed
        rel_sp = slide_release if human_slip_vec is not None else slide_speed
        lat_g = rg & (~(human_speed[:, t] > rel_sp) if human_speed is not None else torch.ones_like(rg))
        # release a ground latch that has drifted > max_dev from where the UMR solution has that point (the robot
        # cannot both hold it and follow the human): the frames get the slip flag from the check below
        if bool(l_on[:, :R].any()):
            Kr = kt.fk(qr)
            if yaw_follow_umr and use_yaw and bool(y_on.any()):
                # heading latch follows UMR's heading increment of the latched body (pivot of the human foot)
                Rb_ = Kr['xmat'][ar[:, None], y_body]
                yn = torch.atan2(Rb_[..., 1, 0], Rb_[..., 0, 0]).double()
                y_tgt = torch.where(y_on, y_tgt + wrap(yn - y_umr_prev), y_tgt)
                y_umr_prev = torch.where(y_on, yn, y_umr_prev)
            if roll_tol is not None and prevX is not None and t > 0:
                # rolling (heel -> toe): UMR's contact moved to another point of the region -> move the latch there,
                # held where the OUTPUT had that point at t-1
                Xr = proj.pts.world(Kr)[:, :nd]
                zr_ = Xr[..., 2] - fl[:, None].float()
                zmin_r, gam_r = seg_min(zr_, dreg)
                zl = zr_.gather(1, l_idx[:, :R])
                roll = l_on[:, :R] & ((zl - zmin_r[:, :R]) > roll_tol) & (gam_r[:, :R] != l_idx[:, :R]) & act[:, None] & (tl[:, t] > 0)[:, None]
                if bool(roll.any()):
                    Pn = prevX[ar[:, None], gam_r[:, :R]].double()
                    tgn = Pn.clone(); tgn[..., 2] = fl[:, None].double() + 0.00045
                    l_idx[:, :R] = torch.where(roll, gam_r[:, :R], l_idx[:, :R])
                    l_tgt[:, :R] = torch.where(roll[..., None], tgn, l_tgt[:, :R])
                    n_roll[0] += int(roll.sum())
                del Xr, zr_
            if max_dev < 1e8 or max_yaw_dev < 1e8:
                Pr = proj.pts.world(Kr, l_idx[:, :R]).double()
                dev_xy = (Pr[..., :2] - l_tgt[:, :R, :2]).norm(dim=-1)
                rel = l_on[:, :R] & (dev_xy > max_dev)
                if use_yaw and max_yaw_dev < 1e8:
                    Rb_ = Kr['xmat'][ar[:, None], y_body]
                    yd = torch.remainder(torch.atan2(Rb_[..., 1, 0], Rb_[..., 0, 0]).double() - y_tgt + torch.pi, 2 * torch.pi) - torch.pi
                    rel |= y_on & (yd.abs() > max_yaw_dev)
                n_release[0] += int(rel.sum())
                rel_drift |= rel
                lat_g = lat_g & ~rel
        need = torch.cat([lat_g, ro], 1) if hoi else lat_g
        l_on &= need; y_on &= lat_g
        # ---- new latches at the first frame of a contact segment: the touch-down point of the UMR solution
        # (UMR follows the human) -> the robot re-aligns with UMR at every touch-down, no drift accumulation
        if hoi:
            Rm = quat_to_mat(ps[:, 3:7].float()).double(); xo = ps[:, :3].double()
            sc = ps[:, 7].double() if ps.shape[-1] > 7 else torch.ones(B, dtype=torch.float64, device=dev)
        newg = lat_g & ~l_on[:, :R] & act[:, None]
        newo = (ro & ~l_on[:, R:] & act[:, None]) if hoi else None
        if bool(newg.any()) or (newo is not None and bool(newo.any())):
            Kr = kt.fk(q_init if latch_at == 'out' else qr)
            X = proj.pts.world(Kr)[:, :nd]
            zg = X[..., 2] - fl[:, None].float()
            _, gam_ = seg_min(zg, dreg)
            P = X[ar[:, None], gam_]
            tg = P.double().clone(); tg[..., 2] = fl[:, None].double() + 0.00045
            # re-latch inside a contact segment (after a human slide / not a drift release): hold the point where
            # the OUTPUT had it at t-1 (no pull back to UMR's point)
            mid = newg & prev_req & ~rel_drift & (tl[:, t] > 0)[:, None] if (relatch_at_output and prevX is not None) \
                else torch.zeros_like(newg)
            if bool(mid.any()):
                Pm = prevX[ar[:, None], gam_].double()
                tg[..., :2] = torch.where(mid[..., None], Pm[..., :2], tg[..., :2])
                n_relatch_out[0] += int(mid.sum())
            rel_drift &= ~newg
            l_idx[:, :R] = torch.where(newg, gam_, l_idx[:, :R]); l_tgt[:, :R] = torch.where(newg[..., None], tg, l_tgt[:, :R])
            l_on[:, :R] |= newg
            l_start[:, :R] = torch.where(newg, torch.full_like(l_start[:, :R], t), l_start[:, :R])
            if dbg is not None:
                Po_ = proj.pts.world(kt.fk(q_init))[:, :nd][ar[:, None], gam_].double()
                dbg.setdefault('td_gap', []).append(((Po_[..., :2] - tg[..., :2]).norm(dim=-1))[newg].cpu())
            body = proj.pts.body[gam_]
            Rb = Kr['xmat'][ar[:, None], body]
            yw = torch.atan2(Rb[..., 1, 0], Rb[..., 0, 0]).double()
            if latch_at != 'out':
                yu = yw
            else:
                Rbu = kt.fk(qr)['xmat'][ar[:, None], body]; yu = torch.atan2(Rbu[..., 1, 0], Rbu[..., 0, 0]).double()
            if bool(mid.any()):
                Rbo = prevXm[ar[:, None], body]
                yw = torch.where(mid, torch.atan2(Rbo[..., 1, 0], Rbo[..., 0, 0]).double(), yw)
            y_body = torch.where(newg, body, y_body); y_tgt = torch.where(newg, yw, y_tgt); y_on |= newg
            y_umr_prev = torch.where(newg, yu, y_umr_prev)
            if newo is not None and bool(newo.any()):
                d, n = proj.obj.sdf(X, ps)
                _, oam_ = seg_min(d, dreg)
                Po = X[ar[:, None], oam_].double(); dd = d[ar[:, None], oam_].double(); nn_ = n[ar[:, None], oam_].double()
                Ps = Po - (dd - 0.00045)[..., None] * nn_
                loc = ((Ps - xo[:, None])[:, :, None, :] @ Rm[:, None])[:, :, 0, :] / sc[:, None, None]
                l_idx[:, R:] = torch.where(newo, oam_, l_idx[:, R:]); l_tgt[:, R:] = torch.where(newo[..., None], loc, l_tgt[:, R:])
                l_on[:, R:] |= newo
        # sliding human region: the latch target moves with the human's slip velocity (not on the latch's first frame)
        if human_slip_vec is not None and human_speed is not None:
            mvh = l_on[:, :R] & ~newg & (human_speed[:, t] > slide_speed) & act[:, None]
            if bool(mvh.any()):
                l_tgt[:, :R, :2] = l_tgt[:, :R, :2] + torch.where(mvh[..., None], human_slip_vec[:, t] / float(fps),
                                                                  torch.zeros_like(human_slip_vec[:, t]))
        # latch rows for latched regions (world targets; object targets re-posed at this frame)
        tgt_w = l_tgt.clone()
        if hoi:
            tgt_w[:, R:] = (Rm[:, None] @ (l_tgt[:, R:] * sc[:, None, None])[..., None])[..., 0] + xo[:, None]
        latch = (l_idx, tgt_w, l_on.clone())
        yaw = (y_body, y_tgt, y_on.clone()) if use_yaw else None
        ex = dict(latch=latch, yaw=yaw, lift_g=lift_g[:, t], lift_o=lift_o[:, t] if hoi else None)
        q_prev = q_pred                         # temporal term: toward the predicted motion (previous output moved by UMR's increment), not toward a standstill
        has_prev = (tl[:, t] > 0)
        mm = proj.measure(q_init, fl, ps, rg, ro, **ex)
        f0[:, t] = mm['flags']
        q = q_init.clone(); f = mm['flags'].clone(); v = mm['viol'].float().clone()
        # decay ladder: the relaxed start fails -> try weaker relaxations (the correction is still needed: keep it
        # instead of re-solving it every frame); the first candidate that passes wins, else the least violating
        for dd in decay_ladder:
            jj = torch.nonzero((f != 0) & act & ~first).flatten()
            if not jj.numel():
                break
            q_d = kt.integrate(qr[jj], dd * corr[jj])
            q_c = proj._clamp(kt.integrate(q_pred[jj], kt.difference(q_pred[jj], q_d).clamp(-cbox, cbox)))
            exj = {kk: (None if vv is None else (tuple(y[jj] for y in vv) if isinstance(vv, tuple) else vv[jj])) for kk, vv in ex.items()}
            mf = proj.measure(q_c, fl[jj], None if ps is None else ps[jj], rg[jj], ro[jj], **exj)
            bt = (mf['viol'].float() < v[jj]) | (mf['flags'] == 0)
            q[jj] = torch.where(bt[:, None], q_c, q[jj]); f[jj] = torch.where(bt, mf['flags'], f[jj])
            v[jj] = torch.where(bt, mf['viol'].float(), v[jj])
            q_init[jj] = torch.where(bt[:, None], q_c, q_init[jj])
            del mf
        step = torch.full((B,), proj.step0, dtype=torch.float64, device=dev)
        idx = torch.nonzero((f != 0) & act).flatten()
        v0 = v.clone()
        # measurement cache: the accepted candidate's measurement is the next iteration's measurement
        mcur = _pick(mm, idx, B) if (cache_measure and not decay_ladder and idx.numel()) else None
        k = 0
        while idx.numel() and k < iters:
            sub = lambda x: None if x is None else (tuple(y[idx] for y in x) if isinstance(x, tuple) else x[idx])
            exi = {kk: sub(vv) for kk, vv in ex.items()}
            mmi = mcur if mcur is not None else proj.measure(q[idx], fl[idx], sub(ps), rg[idx], ro[idx], **exi)
            dev_now = kt.difference(q_pred[idx], q[idx])                 # current deviation from the prediction
            lim = cbox[None].expand_as(dev_now)
            fr = first[idx][:, None]
            clo = torch.where(fr, torch.full_like(dev_now, -1e9), -lim - dev_now)
            chi = torch.where(fr, torch.full_like(dev_now, 1e9), lim - dev_now)
            # leash: total deviation from UMR bounded (an unsatisfiable stretch must not make the output wander off);
            # projected into the continuity box so continuity always wins
            off = kt.difference(qr[idx], q[idx])
            Lh = leash[None].expand_as(off)
            boxlo = torch.minimum(torch.maximum(-Lh - off, clo), chi)
            boxhi = torch.minimum(torch.maximum(Lh - off, clo), chi)
            dq = proj.qp_step(q[idx], (q_init if track_init else qr)[idx], q_prev[idx], has_prev[idx], mmi, rg[idx], ro[idx], None, step[idx],
                              box=(boxlo, boxhi), **exi)
            qn = proj._clamp(kt.integrate(q[idx], dq))
            mn = proj.measure(qn, fl[idx], sub(ps), rg[idx], ro[idx], **exi)
            fn = mn['flags'].clone(); vn = mn['viol'].float().clone()
            better = (vn < v[idx]) | (fn == 0)
            macc = _merge(mmi, mn, better) if mcur is not None else None
            del mn, mmi
            hb = torch.nonzero(~better).flatten()
            if hb.numel():
                ih = idx[hb]
                qh = proj._clamp(kt.integrate(q[ih], 0.5 * dq[hb]))
                exh = {kk: (None if vv is None else (tuple(y[ih] for y in vv) if isinstance(vv, tuple) else vv[ih])) for kk, vv in ex.items()}
                mh = proj.measure(qh, fl[ih], None if ps is None else ps[ih], rg[ih], ro[ih], **exh)
                bh = (mh['viol'].float() < v[ih]) | (mh['flags'] == 0)
                qn[hb] = torch.where(bh[:, None], qh, qn[hb])
                fn[hb] = torch.where(bh, mh['flags'], fn[hb]); vn[hb] = torch.where(bh, mh['viol'].float(), vn[hb])
                better[hb] = bh
                if macc is not None and bool(bh.any()):
                    jb = torch.nonzero(bh).flatten()
                    macc = _assign(macc, hb[jb], _pick(mh, jb, hb.numel()))
                del mh
            q[idx] = torch.where(better[:, None], qn, q[idx])
            f[idx] = torch.where(better, fn, f[idx]); v[idx] = torch.where(better, vn, v[idx])
            step[idx] = torch.where(better, step[idx], (step[idx] * 0.5).clamp_min(0.005))
            its[idx, t] += 1
            keep = (f[idx] != 0) & (step[idx] > 0.0051)
            if k == stall_k:                                   # no progress -> this frame will be flagged anyway
                keep &= v[idx] < 0.7 * v0[idx]
            if macc is not None:
                mcur = _pick(macc, torch.nonzero(keep).flatten(), keep.numel())
            idx = idx[keep]
            k += 1
            throttle.tick()
        q = torch.where(act[:, None], q, Q[:, t])
        if dbg is not None and bool(((f & 32) != 0).any()):
            ii = torch.nonzero((f & 32) != 0).flatten()
            mm_ = proj.measure(q[ii], fl[ii], None if ps is None else ps[ii], rg[ii], ro[ii],
                               **{kk: (None if vv is None else (tuple(y[ii] for y in vv) if isinstance(vv, tuple) else vv[ii])) for kk, vv in ex.items()})
            li, lt, lv = latch
            P_ = proj.pts.world(mm_['K'], li[ii]).double(); dvv = (P_ - lt[ii]).abs()
            tol = proj._latch_tol(dev)
            dbg['latch_bad'] += int((lv[ii] & (dvv > tol).any(-1)).any(-1).sum())
            badm = lv[ii] & (dvv > tol).any(-1)
            dbg.setdefault('bad_dev', []).append(dvv.norm(dim=-1)[badm].cpu())
            dbg.setdefault('bad_age', []).append((t - l_start[ii])[badm].cpu())
            if 'yaw_err' in mm_:
                dbg['yaw_bad'] += int((y_on[ii] & (mm_['yaw_err'].abs() > 0.0015)).any(-1).sum())
            dbg['its_at_slip'] += int(its[ii, t].sum()) if False else 0
            dbg['n'] += int(ii.numel())
        Q[:, t] = q; flags[:, t] = torch.where(act, f, torch.zeros_like(f))
        corr = torch.where(act[:, None], kt.difference(qr, q), corr)
        # output state for the next frame (re-latch / rolling targets); robot slip itself is judged on the output
        # by slip_gate below; object latches (object frame) keep their latch-deviation slip check
        Kq = kt.fk(q)
        Xq = proj.pts.world(Kq)[:, :nd]
        if hoi and bool(l_on[:, R:].any()):
            Po_ = proj.pts.world(Kq, l_idx[:, R:]).double()
            dvo = (Po_ - tgt_w[:, R:]).abs()
            obj_bad[:, t] = (l_on[:, R:] & (dvo > proj._latch_tol(dev)).any(-1)).any(-1) & act
        prevX = Xq; prevXm = Kq['xmat']; prev_req = rg.clone()
        if log and t % 500 == 0:
            print(f'[strict_seq] t={t}/{T} flagged so far {int((flags[:, :t + 1] != 0).sum())}', flush=True)
    proj.w_t = w_t_saved
    # bit 32 is rebuilt from the output: solver-internal ground latch / heading deviations are dropped
    flags = (flags & ~SL.FLAG_SLIP) | (obj_bad.int() * SL.FLAG_SLIP)
    jump = jump_gate(proj, Q, Qref, piece, fps, jq=proj.jq, tol_p=gate_p, tol_r=gate_r, tol_j=gate_j)
    flags = flags | (jump.int() * FLAG_JUMP)
    sg, sfol, _, _, _ = slip_gate(proj, Q, floor_bt, piece, fps, human_speed=human_speed)
    flags = flags | (sg.int() * SL.FLAG_SLIP) | (sfol.int() * SL.FLAG_FOLLOW)
    flags = torch.where(valid, flags, torch.zeros_like(flags))
    t2 = time.time()
    info = dict(labels_s=t1 - t0, project_s=t2 - t1, frames=int(valid.sum()), flagged=int(((flags & DROP_MASK) != 0).sum()),
                iters_mean=float(its[valid].float().mean()), jump_frames=int(jump.sum()), slip_gate_frames=int((sg & valid).sum()), latch_releases=n_release[0], latch_roll_switches=n_roll[0], relatch_at_output=n_relatch_out[0], follow_frames=int((sfol & valid).sum()),
                umr_floor_shift_m=[round(float(x), 4) for x in floor_shift[:8]],
                obj_latch_slip=int((obj_bad & valid).sum()), human_sliding=int((req_g & (human_speed > slide_speed)).any(-1).sum()) if human_speed is not None else 0,
                flags_hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)}, mode='sequential')
    # extras for the final verification: latches are online state, so verification uses flags from the solve
    return Q, flags, f0, req_g, req_o, dict(info, extras={}, lift_g=lift_g, lift_o=lift_o)


def umr_floor_shift(proj, Qref, floor_bt, human_g, valid, pose=None, nmax=512):
    """[B] z shift of the UMR solution so that the robot's feet sit on the calibrated human floor: minus the median
    of the robot feet's region clearance over (up to nmax) frames where the human has a foot on the ground."""
    B = Qref.shape[0]; dev = Qref.device
    fi = [L.RID['L_foot'], L.RID['R_foot']]
    out = torch.zeros(B, dtype=torch.float64, device=dev)
    for b in range(B):
        tt = torch.nonzero((human_g[b][:, fi] & valid[b][:, None]).any(-1)).flatten()
        if tt.numel() < 10:
            continue
        tt = tt[::max(1, tt.numel() // nmax)]
        z0 = torch.zeros(tt.numel(), L.NR, dtype=torch.bool, device=dev)
        mm = proj.measure(Qref[b, tt], floor_bt[b, tt], None if pose is None else pose[b, tt], z0, z0, self_act=-1.0)
        g = mm['greg'][:, fi]; hf = human_g[b, tt][:, fi]
        v_ = g[hf & torch.isfinite(g)]
        if v_.numel() >= 10:
            out[b] = (-v_.double().median() + 0.0004).clamp(-0.05, 0.05)
    return out


def strict_seq_segmented(proj, Qc, vC, flC, hgC, hoC=None, poseC=None, hclC=None, hocC=None, hspC=None, fps=30.0,
                         seg_len=256, overlap=90, iters=30, log=None, hsvC=None, **kw):
    """Time-parallel version of strict_seq for clip-major inputs [C, Tmax, ...] (vC [C, Tmax] valid):
    every clip is cut into segments of seg_len frames, each run as its own row with `overlap` warm-up frames taken
    from before its start (all rows in lock-step -> ~Tmax / seg_len x fewer sequential steps).  Consecutive segments
    are stitched at the frame of the overlap's last two thirds where the two solutions agree best (corrections decay
    and latch targets come from UMR's touch-down poses, so the two runs converge inside the overlap); the continuity
    gate and the output slip gate are then re-evaluated on the stitched clip, so a visible seam is FLAGGED, never
    hidden.  The UMR floor shift is computed per clip (not per segment).
    Returns Q [C,Tmax,nq] (floor-shifted UMR + corrections), flags [C,Tmax], req_g, req_o, info."""
    t0 = time.time()
    kt = proj.kt
    C, Tm, nq = Qc.shape
    dev = Qc.device
    lens = vC.sum(1).tolist()
    zs = umr_floor_shift(proj, Qc, flC, hgC, vC, poseC)
    rows = []
    for c in range(C):
        n = int(lens[c]); st = 0
        while st < n:
            a = max(0, st - overlap) if st else 0
            e = min(n, st + seg_len)
            if n - e < seg_len // 4:                 # do not leave a tiny last segment
                e = n
            rows.append((c, a, e, st)); st = e
    Rn = len(rows)
    rc = torch.tensor([r[0] for r in rows], device=dev); ra = torch.tensor([r[1] for r in rows], device=dev)
    re_ = torch.tensor([r[2] for r in rows], device=dev)
    Lr = int((re_ - ra).max())
    ar = torch.arange(Lr, device=dev)
    fidx = ra[:, None] + ar[None]
    rv = fidx < re_[:, None]
    fidx = fidx.clamp(max=Tm - 1)
    g = lambda x: None if x is None else x[rc[:, None], fidx]
    inf = float('inf')
    piece = torch.where(rv, torch.arange(Rn, device=dev)[:, None].expand(Rn, Lr), torch.full((Rn, Lr), -1, dtype=torch.long, device=dev))
    tl = ar[None].expand(Rn, Lr).contiguous()
    hg_r = g(hgC) & rv[..., None]
    ho_r = None if hoC is None else g(hoC) & rv[..., None]
    hcl_r = None if hclC is None else torch.where(rv[..., None], g(hclC), torch.full_like(g(hclC), inf))
    hoc_r = None if hocC is None else torch.where(rv[..., None], g(hocC), torch.full_like(g(hocC), inf))
    Qr, flags_r, f0, rg_r, ro_r, info = strict_seq(proj, g(Qc), piece, tl, g(flC), hg_r, ho_r, pose=g(poseC), iters=iters,
                                                   human_clear_g=hcl_r, human_clear_o=hoc_r, fps=fps, human_speed=g(hspC), human_slip_vec=g(hsvC),
                                                   zshift=zs[rc], log=bool(log), **kw)
    t1 = time.time()
    # ---- stitch
    Qref_sh = Qc.clone(); Qref_sh[..., 2] += zs[:, None]
    Q = Qref_sh.clone()
    flags = torch.zeros(C, Tm, dtype=torch.int32, device=dev)
    req_g = torch.zeros_like(hgC); req_o = torch.zeros_like(hgC)
    seam_d = []
    by_clip = {}
    for i, r in enumerate(rows):
        by_clip.setdefault(r[0], []).append(i)
    jd = kt.scalar_dadr
    for c, ids in by_clip.items():
        cut = 0                                      # first frame of clip c taken from the current row
        for k_, i in enumerate(ids):
            a, e, st = rows[i][1], rows[i][2], rows[i][3]
            if k_ + 1 < len(ids):
                j = ids[k_ + 1]; aj = rows[j][1]; sj = rows[j][3]
                lo_f = max(aj + overlap // 3, cut + 1)
                fr = torch.arange(lo_f, sj, device=dev)
                if fr.numel():
                    qa = Qr[i, fr - a]; qb = Qr[j, fr - aj]
                    dv = kt.difference(qa, qb)
                    d = dv[:, kt.free_dof0:kt.free_dof0 + 3].norm(dim=-1) + dv[:, jd].abs().amax(-1) * 0.1 \
                        if kt.free_body >= 0 else dv[:, jd].abs().amax(-1)
                    m = int(torch.argmin(d)); seam = int(fr[m]); seam_d.append(float(d[m]))
                else:
                    seam = sj; seam_d.append(float('nan'))
            else:
                seam = e
            sl = slice(cut, seam)
            Q[c, sl] = Qr[i, cut - a:seam - a]; flags[c, sl] = flags_r[i, cut - a:seam - a].to(torch.int32)
            req_g[c, sl] = rg_r[i, cut - a:seam - a]; req_o[c, sl] = ro_r[i, cut - a:seam - a]
            cut = seam
    pieceC = torch.where(vC, torch.arange(C, device=dev)[:, None].expand(C, Tm), torch.full((C, Tm), -1, dtype=torch.long, device=dev))
    jump = jump_gate(proj, Q, Qref_sh, pieceC, fps, jq=proj.jq)
    sg, sfol, _, _, _ = slip_gate(proj, Q, flC, pieceC, fps, human_speed=hspC)
    seam_flag = (jump | sg) & ~(((flags & FLAG_JUMP) != 0) | ((flags & 32) != 0))
    # the stitched clip's output gate decides slip / follow (bit 32 from the rows' object latches is kept)
    flags = torch.where(vC, (flags & ~SL.FLAG_FOLLOW) | (jump.int() * FLAG_JUMP) | (sg.int() * SL.FLAG_SLIP) | (sfol.int() * SL.FLAG_FOLLOW),
                        torch.zeros_like(flags))
    sd = torch.tensor([x for x in seam_d if x == x]) if seam_d else torch.zeros(0)
    info = dict(info, segments=Rn, seg_len=seg_len, overlap=overlap, rows_steps=Lr,
                seam_disc_p50=float(sd.median()) if sd.numel() else None, seam_disc_max=float(sd.max()) if sd.numel() else None,
                new_flags_from_stitch=int((seam_flag & vC).sum()), stitch_s=time.time() - t1, floor_shift_clip=zs.tolist()[:8],
                flagged=int(((flags & DROP_MASK) != 0).sum()))
    return Q, flags, req_g, req_o, info
