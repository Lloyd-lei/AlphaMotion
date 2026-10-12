"""Batched convex QP solver (ADMM / OSQP-style operator splitting = augmented Lagrangian), fp64, torch.

    min_x 0.5 x'Px + q'x + sum_i 0.5 c_i max(0, g_i'x - h_i)^2           (soft rows, c_i > 0  == UMR slack rows)
    s.t.  lo <= x <= hi                                                    (box: joint limits, step boxes)
          ||x[ball]||_2 <= r                                               (UMR "l2" trust region on local dofs)
          g_i'x <= h_i                                                     (hard rows, c_i == 0)

UMR solves exactly this problem with Clarabel (interior point, SOC cone for the l2 trust region).  Here every batch
element is one QP; all elements iterate in lock-step.  Rows are unit-normalised; rho is adapted per element
(OSQP rule) with a batched Cholesky refactorisation of the n x n KKT matrix (n = nv ~ 30-40).
"""
from __future__ import annotations

import torch


class QPResult:
    def __init__(self, x, iters, r_prim, r_dual):
        self.x, self.iters, self.r_prim, self.r_dual = x, iters, r_prim, r_dual


def solve_qp(P, q, lo, hi, G=None, h=None, soft=None, ball_mask=None, ball_r=None, max_iter=400, check_every=25,
             eps_abs=1e-7, eps_rel=1e-6, alpha=1.6, rho0=None, x0=None, **_):
    B, n = q.shape
    dt = torch.float64
    P = P.to(dt); q = q.to(dt); lo = lo.to(dt); hi = hi.to(dt)
    dev = q.device
    eye = torch.eye(n, dtype=dt, device=dev)
    has_G = G is not None and G.shape[1] > 0
    if has_G:
        G = G.to(dt); h = h.to(dt)
        soft = torch.zeros_like(h) if soft is None else soft.to(dt)
        gn = G.norm(dim=-1)
        ok = gn > 1e-12
        gs = torch.where(ok, gn, torch.ones_like(gn))
        G = G / gs[..., None]
        h = torch.where(ok, h / gs, torch.full_like(h, float('inf')))
        soft = soft * gs * gs
    if ball_mask is not None:
        bm = ball_mask.to(dt).expand(B, n)
        r_ball = torch.as_tensor(ball_r, dtype=dt, device=dev).expand(B)
    else:
        bm = None
    # tight boxes (UMR: lower==upper -> centre +- 1e-12)
    tight = torch.isfinite(lo) & torch.isfinite(hi) & (hi <= lo)
    c = 0.5 * (lo + hi)
    lo = torch.where(tight, c - 1e-12, lo); hi = torch.where(tight, c + 1e-12, hi)
    finite_box = (torch.isfinite(lo) | torch.isfinite(hi)).to(dt)

    diagP = torch.diagonal(P, dim1=-2, dim2=-1)
    if rho0 is None:
        rho = (0.1 * diagP.mean(-1)).clamp(1e-4, 1e4)          # [B]
    else:
        rho = torch.full((B,), float(rho0), dtype=dt, device=dev)
    sigma = 1e-6
    # per-row rho multipliers: box rows w/o finite bounds get 1e-6 (as OSQP for inf bounds)
    w_box = torch.where(finite_box > 0, torch.ones_like(finite_box), torch.full_like(finite_box, 1e-6))

    def factor(rho):
        K = P + sigma * eye + torch.diag_embed(rho[:, None] * w_box)
        if bm is not None:
            K = K + torch.diag_embed(rho[:, None] * bm)
        if has_G:
            K = K + rho[:, None, None] * (G.transpose(1, 2) @ G)
        L, info = torch.linalg.cholesky_ex(K)
        return L

    L = factor(rho)
    x = torch.zeros_like(q) if x0 is None else x0.to(dt).clone()
    zb = x.clamp(lo, hi); yb = torch.zeros_like(q)
    if bm is not None:
        zs = x.clone(); ys = torch.zeros_like(q)
    if has_G:
        zg = torch.minimum(G @ x[..., None], h[..., None])[..., 0]
        zg = torch.where(torch.isfinite(zg), zg, torch.zeros_like(zg)); yg = torch.zeros_like(h)
    it = 0
    r_prim = r_dual = None
    while it < max_iter:
        for _ in range(check_every):
            rb = rho[:, None] * w_box
            rhs = sigma * x - q + rb * zb - yb
            if bm is not None:
                rhs = rhs + (rho[:, None] * zs - ys) * bm
            if has_G:
                rhs = rhs + (G.transpose(1, 2) @ (rho[:, None] * zg - yg)[..., None])[..., 0]
            xt = torch.cholesky_solve(rhs[..., None], L)[..., 0]
            x = alpha * xt + (1 - alpha) * x
            # box
            vb = alpha * xt + (1 - alpha) * zb
            zb_new = (vb + yb / rb).clamp(lo, hi)
            yb = yb + rb * (vb - zb_new); zb = zb_new
            if bm is not None:
                vs = alpha * xt + (1 - alpha) * zs
                u = vs + ys / rho[:, None]
                nrm = (u * bm).norm(dim=-1, keepdim=True)
                scl = torch.clamp(r_ball[:, None] / nrm.clamp_min(1e-30), max=1.0)
                zs_new = torch.where(bm > 0, u * scl, u)
                ys = ys + rho[:, None] * (vs - zs_new) * bm; zs = zs_new
            if has_G:
                gx = (G @ xt[..., None])[..., 0]
                vg = alpha * gx + (1 - alpha) * zg
                u = vg + yg / rho[:, None]
                hard = torch.minimum(u, h)
                sft = torch.where(u <= h, u, (soft * h + rho[:, None] * u) / (soft + rho[:, None]))
                zg_new = torch.where(soft > 0, sft, hard)
                yg = yg + rho[:, None] * (vg - zg_new); zg = zg_new
            it += 1
        # residuals (inf-norms) + rho adaptation
        rp = (x - zb).abs().amax(-1)
        Ax = x.abs().amax(-1); Z = zb.abs().amax(-1)
        Aty = yb.clone()
        if bm is not None:
            rp = torch.maximum(rp, ((x - zs) * bm).abs().amax(-1)); Aty = Aty + ys * bm
        if has_G:
            gx = (G @ x[..., None])[..., 0]
            fin = torch.isfinite(h)
            rg = torch.where(fin | (soft > 0), (gx - zg).abs(), torch.zeros_like(gx))
            rp = torch.maximum(rp, rg.amax(-1))
            Ax = torch.maximum(Ax, gx.abs().amax(-1)); Z = torch.maximum(Z, torch.where(fin, zg.abs(), torch.zeros_like(zg)).amax(-1))
            Aty = Aty + (G.transpose(1, 2) @ yg[..., None])[..., 0]
        Px = (P @ x[..., None])[..., 0]
        rd = (Px + q + Aty).abs().amax(-1)
        sp = torch.maximum(Ax, Z); sd = torch.maximum(torch.maximum(Px.abs().amax(-1), Aty.abs().amax(-1)), q.abs().amax(-1))
        r_prim, r_dual = rp, rd
        done = (rp <= eps_abs + eps_rel * sp) & (rd <= eps_abs + eps_rel * sd)
        if bool(done.all()) or it >= max_iter:
            break
        ratio = torch.sqrt((rp / sp.clamp_min(1e-30)) / (rd / sd.clamp_min(1e-30)).clamp_min(1e-30))
        new_rho = (rho * ratio).clamp(1e-6, 1e6)
        change = (new_rho > 5 * rho) | (new_rho < rho / 5)
        if bool(change.any()):
            rho = torch.where(change & ~done, new_rho, rho)
            L = factor(rho)
    x = torch.minimum(torch.maximum(x, lo), hi)
    res = QPResult(x, it, r_prim, r_dual); res.mu = None
    return res
