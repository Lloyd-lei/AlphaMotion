"""Batched primal-dual interior-point QP solver (Mehrotra predictor-corrector), fp64, torch.

Solves, for every batch element independently (all in lock-step), the exact problem UMR hands to Clarabel:

    min_{x,u} 0.5 x'Px + q'x + sum_{i in soft} 0.5 c_i u_i^2
    s.t.  g_i'x - u_i <= h_i, u_i >= 0  (soft rows = UMR slack rows)       g_i'x <= h_i (hard rows)
          lo <= x <= hi                                                     (joint limits / step box)
          ||x[ball]||_2 <= r                                                (UMR l2 trust region; SOC in Clarabel)

The l2 trust region is the convex quadratic constraint (||x_S||^2 - r^2)/(2r) <= 0 (same feasible set as Clarabel's
second-order cone).  All inequalities (rows, box, ball) live in ONE slack/multiplier vector so an iteration is ~50
kernels.  Infeasible start, Mehrotra centring, fraction-to-boundary 0.99, (n+k)x(n+k) batched Cholesky.
"""
from __future__ import annotations

import torch

INF = float('inf')


class Result:
    def __init__(self, x, iters, r_prim, r_dual, mu):
        self.x, self.iters, self.r_prim, self.r_dual, self.mu = x, iters, r_prim, r_dual, mu


def solve_qp(P, q, lo, hi, G=None, h=None, soft=None, ball_mask=None, ball_r=None, max_iter=40, check_every=4,
             eps=1e-9, sync=True, soft_idx=None, kred=0, red_start=4, dtype=torch.float64, **_):
    """kred > 0: after red_start iterations the KKT matrix is assembled from the kred rows with the largest
    z/s only (inexact Newton; residuals stay exact so the fixed point is unchanged)."""
    dt = dtype
    B, n = q.shape
    dev = q.device
    P = P.to(dt); q = q.to(dt); lo = lo.to(dt); hi = hi.to(dt)
    tight = torch.isfinite(lo) & torch.isfinite(hi) & (hi <= lo)
    cen = 0.5 * (lo + hi)
    lo = torch.where(tight, cen - 1e-12, lo); hi = torch.where(tight, cen + 1e-12, hi)
    osc = torch.diagonal(P, dim1=-2, dim2=-1).abs().mean(-1).clamp_min(1e-8)[:, None]
    P = P / osc[..., None]; q = q / osc
    if G is not None and G.shape[1] > 0:
        G = G.to(dt); h = h.to(dt)
        soft = torch.zeros_like(h) if soft is None else soft.to(dt)
        gn = G.norm(dim=-1)
        rv = (gn > 1e-12) & torch.isfinite(h)
        gs = torch.where(rv, gn, torch.ones_like(gn))
        G = torch.where(rv[..., None], G / gs[..., None], torch.zeros_like(G))
        h = torch.where(rv, h / gs, torch.ones_like(h))
        soft = soft * gs * gs / osc
        sidx = torch.nonzero((soft > 0).any(0)).flatten() if soft_idx is None else soft_idx.to(dev)
    else:
        G = torch.zeros(B, 0, n, dtype=dt, device=dev); h = torch.zeros(B, 0, dtype=dt, device=dev)
        rv = torch.zeros(B, 0, dtype=torch.bool, device=dev); soft = torch.zeros(B, 0, dtype=dt, device=dev)
        sidx = torch.zeros(0, dtype=torch.long, device=dev)
    m = G.shape[1]
    k = int(sidx.numel())
    N = n + k
    use_ball = ball_mask is not None and ball_r is not None and float(ball_r) < INF
    nb = 1 if use_ball else 0
    # general rows: [G (m) | u>=0 (k) | ball (nb)];  box rows on x are handled implicitly (diagonal)
    M = m + k + nb
    ib = m + k
    A = torch.zeros(B, M, N, dtype=dt, device=dev)
    b = torch.zeros(B, M, dtype=dt, device=dev)
    valid = torch.zeros(B, M, dtype=torch.bool, device=dev)
    A[:, :m, :n] = G; b[:, :m] = h; valid[:, :m] = rv
    Pf = torch.zeros(B, N, N, dtype=dt, device=dev)
    Pf[:, :n, :n] = P
    if k:
        ar = torch.arange(k, device=dev)
        neg = torch.full((B, k), -1.0, dtype=dt, device=dev)          # tensor value: graph-capture safe index_put
        A[:, sidx, n + ar] = neg
        A[:, m + ar, n + ar] = neg
        sv = soft[:, sidx]
        valid[:, m:m + k] = rv[:, sidx] & (sv > 0)
        csoft = torch.where(sv > 0, sv, torch.full_like(sv, 1e12))      # hard element on a soft row: u pinned to 0
        Pf[:, n:, n:] = torch.diag_embed(csoft)
    if use_ball:
        bm = ball_mask.to(dt).expand(B, n)
        r = float(ball_r)
        valid[:, ib] = True
        bmN = torch.zeros(B, N, dtype=dt, device=dev); bmN[:, :n] = bm
    else:
        bmN = torch.zeros(B, N, dtype=dt, device=dev); r = 1.0
    bb = torch.cat([torch.where(torch.isfinite(hi), hi, torch.zeros_like(hi)), torch.where(torch.isfinite(lo), -lo, torch.zeros_like(lo))], 1)
    vb = torch.cat([torch.isfinite(hi), torch.isfinite(lo)], 1)
    qf = torch.zeros(B, N, dtype=dt, device=dev); qf[:, :n] = q
    vf = valid.to(dt); vbf = vb.to(dt)
    ncon = (vf.sum(-1) + vbf.sum(-1)).clamp_min(1.0)
    regI = (1e-13 if dt == torch.float64 else 1e-9) * torch.eye(N, dtype=dt, device=dev)
    y = torch.zeros(B, N, dtype=dt, device=dev)
    f = (A @ y[..., None])[..., 0] - b
    s = torch.where(valid, (-f).clamp_min(0) + 1.0, torch.ones_like(f))
    z = vf.clone()
    fb = -bb
    sb = torch.where(vb, (-fb).clamp_min(0) + 1.0, torch.ones_like(fb))
    zb = vbf.clone()
    frozen = torch.zeros(B, dtype=torch.bool, device=dev)
    rp_max = rd_max = mu = None
    it = 0
    step_fn = _ipm_step_compiled if COMPILE else _ipm_step
    while it < max_iter:
        y, s, z, sb, zb, rp_max, rd_max, mu, done = step_fn(y, s, z, sb, zb, frozen, A, b, valid, vf, bb, vb, vbf, Pf, qf,
                                                            ncon, regI, bmN, r, use_ball, eps, ib, n)
        frozen = frozen | done
        it += 1
        if sync and it % check_every == 0 and bool(frozen.all()):
            break
    x = torch.minimum(torch.maximum(y[:, :n], lo), hi)
    return Result(x, it, rp_max, rd_max, mu)


def _ipm_step(y, s, z, sb, zb, frozen, A, b, valid, vf, bb, vb, vbf, Pf, qf, ncon, regI, bmN, r: float, use_ball: bool,
              eps: float, ib: int, n: int):
    """One Mehrotra predictor-corrector iteration (pure function; torch.compile-able).
    General rows A (ball row at ib, re-linearised every iteration); box rows [x <= hi | -x <= -lo] implicit."""
    M = A.shape[1]
    N = A.shape[2]
    f = (A @ y[..., None])[..., 0] - b
    if use_ball:
        x2 = (y * bmN).pow(2).sum(-1)
        f = torch.cat([f[:, :ib], ((x2 - r * r) / (2 * r))[:, None], f[:, ib + 1:]], 1)
        J = torch.cat([A[:, :ib], (y * bmN / r)[:, None], A[:, ib + 1:]], 1)
    else:
        J = A
    Jt = J.transpose(1, 2)
    x = y[:, :n]
    xx = torch.cat([x, -x], 1)
    fbx = xx - bb
    pad = torch.zeros(y.shape[0], N - n, dtype=y.dtype, device=y.device)

    def box_t(t):          # J_box^T t
        return torch.cat([t[:, :n] - t[:, n:], pad], 1)

    rd = (Pf @ y[..., None])[..., 0] + qf + (Jt @ (z * vf)[..., None])[..., 0] + box_t(zb * vbf)
    rp = (f + s) * vf
    rpb = (fbx + sb) * vbf
    sz = s * z * vf
    szb = sb * zb * vbf
    mu = (sz.sum(-1) + szb.sum(-1)) / ncon
    rp_max = torch.maximum(rp.abs().amax(-1), rpb.abs().amax(-1)); rd_max = rd.abs().amax(-1)
    done = (rp_max < eps) & (rd_max < eps) & (mu < eps * 0.1)
    D = z / s * vf
    Db = zb / sb * vbf
    Km = Pf + Jt @ (D[..., None] * J) + regI
    dg = torch.cat([Db[:, :n] + Db[:, n:], pad], 1)
    if use_ball:
        dg = dg + z[:, ib, None] * bmN / r
    Km = Km + torch.diag_embed(dg)
    L, info = torch.linalg.cholesky_ex(Km)

    def direction(rc, rcb):
        t = ((-rc + z * rp) / s) * vf
        tb = ((-rcb + zb * rpb) / sb) * vbf
        rhs = -rd - (Jt @ t[..., None])[..., 0] - box_t(tb)
        dy = torch.cholesky_solve(rhs[..., None], L)[..., 0]
        ds = (-rp - (J @ dy[..., None])[..., 0]) * vf
        dxx = torch.cat([dy[:, :n], -dy[:, :n]], 1)
        dsb = (-rpb - dxx) * vbf
        dz = ((-rc - z * ds) / s) * vf
        dzb = ((-rcb - zb * dsb) / sb) * vbf
        return dy, ds, dz, dsb, dzb

    def step_len(ds, dz, dsb, dzb, frac: float):
        big = torch.full_like(s, INF)
        bigb = torch.full_like(sb, INF)
        r1 = torch.where((ds < 0) & valid, -s / ds.clamp(max=-1e-30), big).amin(-1)
        r2 = torch.where((dz < 0) & valid, -z / dz.clamp(max=-1e-30), big).amin(-1)
        r3 = torch.where((dsb < 0) & vb, -sb / dsb.clamp(max=-1e-30), bigb).amin(-1)
        r4 = torch.where((dzb < 0) & vb, -zb / dzb.clamp(max=-1e-30), bigb).amin(-1)
        return (frac * torch.minimum(torch.minimum(r1, r2), torch.minimum(r3, r4))).clamp(max=1.0)

    dy, ds, dz, dsb, dzb = direction(sz, szb)
    a_aff = step_len(ds, dz, dsb, dzb, 1.0)[:, None]
    mu_aff = (((s + a_aff * ds) * (z + a_aff * dz) * vf).sum(-1) + ((sb + a_aff * dsb) * (zb + a_aff * dzb) * vbf).sum(-1)) / ncon
    sigma = (mu_aff / mu.clamp_min(1e-30)).clamp(0, 1) ** 3
    smu = (sigma * mu)[:, None]
    rc = sz + ds * dz * vf - smu * vf
    rcb = szb + dsb * dzb * vbf - smu * vbf
    dy, ds, dz, dsb, dzb = direction(rc, rcb)
    a = step_len(ds, dz, dsb, dzb, 0.99)
    ok = torch.isfinite(dy).all(-1) & torch.isfinite(a) & (info == 0) & ~(frozen | done)
    a = torch.where(ok, a, torch.zeros_like(a))[:, None]
    y = y + a * torch.nan_to_num(dy)
    s = torch.where(valid, s + a * torch.nan_to_num(ds), s)
    z = torch.where(valid, z + a * torch.nan_to_num(dz), z)
    sb = torch.where(vb, sb + a * torch.nan_to_num(dsb), sb)
    zb = torch.where(vb, zb + a * torch.nan_to_num(dzb), zb)
    return y, s, z, sb, zb, rp_max, rd_max, mu, done


import os as _os
COMPILE = _os.environ.get('UMRG_COMPILE', '1') == '1'
_ipm_step_compiled = torch.compile(_ipm_step, dynamic=True, fullgraph=False) if COMPILE else _ipm_step
