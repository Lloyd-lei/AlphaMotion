"""Batched first-frame SMPL-X beta fit on the GPU (float64 damped Newton), same objective as
hiphi2smplx.beta_fitting.fit_betas (joints2smpl/SMPLify weights):

  stage 1 (camera):  min_{global_orient, cam}  sum_{j in (2,1,17,16)} |t_j - (J_j + cam)|^2 + 100^2 |cam - est|^2
  stage 2 (body):    min_{body_pose, betas, global_orient, cam}
                     600^2 sum_j gmof(J_j + cam - t_j) + (4.78*1.5)^2 GMM8(body_pose) + 15.2^2 angle_prior
                     + 5^2 |betas|^2 + 5^2 |body_pose - mean_pose|^2
  (J = 22-joint skeleton FK of the beta-shaped SMPL-X rest joints; t = foot-corrected first-frame BVH targets)

The CPU path minimises this with float32 LBFGS (strong Wolfe, lr 1e-2, 10x100 + 100x100 iterations). In float32
that optimiser stalls on loss-resolution noise: a 1e-5 m perturbation of the targets moves its betas by up to 0.4
(8 mm of vertex displacement), while the float64 minimiser is stable to 1e-4 m perturbations and has a lower loss.
This module returns the float64 minimiser (Newton with exact Hessians from torch.func, Levenberg damping,
monotone acceptance, fixed iteration budget with per-problem freezing at convergence).
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, hessian, jacfwd, vmap

sys.path[:0] = ["/var/tmp/alphamotion/umr_factory/hiphi2smplx/src", "/var/tmp/alphamotion/umr_factory/h2s_shim"]
import hiphi2smplx.beta_fitting as BF  # noqa: E402

MODEL = "/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz"
BETA_DATA = Path("/var/tmp/alphamotion/umr_factory/beta_fit")
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
ALIGN = list(BF._ALIGNMENT_JOINTS)
ANG_IDX = list(BF._ANGLE_PRIOR_INDICES)
ANG_SGN = list(BF._ANGLE_PRIOR_SIGNS)


def _aa2mat(a):
    t2 = (a * a).sum(-1, keepdim=True)
    th = t2.clamp_min(1e-12).sqrt()
    s = torch.where(t2 > 1e-8, torch.sin(th) / th.clamp_min(1e-8), 1.0 - t2 / 6.0)
    c = torch.where(t2 > 1e-8, (1.0 - torch.cos(th)) / t2.clamp_min(1e-8), 0.5 - t2 / 24.0)
    x, y, z = a.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(a.shape[:-1] + (3, 3))
    I = torch.eye(3, dtype=a.dtype, device=a.device).expand(K.shape)
    return I + s.unsqueeze(-1) * K + c.unsqueeze(-1) * (K @ K)


class BatchedBetaFit:
    def __init__(self, device="cuda", dtype=torch.float64, gn=True):
        self.dev, self.dt, self.gn = torch.device(device), dtype, gn
        sk = BF._create_skeleton_model(Path(MODEL), torch.device("cpu"))  # same float32 buffers as the CPU path
        kw = dict(dtype=dtype, device=self.dev)
        self.jt = sk.joint_template.to(**kw)            # (22,3)
        self.jb = sk.joint_shape_basis.to(**kw)         # (22,3,10)
        pr = BF._MaxMixturePrior(BETA_DATA / "gmm_08.pkl")
        self.means = pr.means.to(**kw)                  # (8,69)
        self.prec = pr.precisions.to(**kw)              # (8,69,69)
        self.nllw = pr.nll_weights.to(**kw)[0]          # (8,)
        self.prec_bd = torch.block_diag(*self.prec)     # (552,552): one GEMM under vmap/hessian, no expansion
        self.mean_pose = torch.tensor(np.load(BETA_DATA / "mean_pose.npy").reshape(72), **kw)
        self.ang_sgn = torch.tensor(ANG_SGN, **kw)

    # ---- model ----
    def joints(self, go, bp, betas):
        """single problem: go (3,), bp (69,), betas (10,) -> (22,3) joints (root at rest pelvis)."""
        rest = self.jt + (self.jb * betas).sum(-1)
        pose = torch.cat([go, bp]).reshape(24, 3)
        R = _aa2mat(pose[:22])
        gR = [None] * 22; pos = [None] * 22
        for j, p in enumerate(PARENTS):
            if p < 0:
                gR[j] = R[j]; pos[j] = rest[0]
            else:
                gR[j] = gR[p] @ R[j]
                pos[j] = pos[p] + gR[p] @ (rest[j] - rest[p])
        return torch.stack(pos)

    def gmm(self, bp, comp=None):
        """max-mixture NLL; comp (scalar long tensor) pins one component (used for the per-component restarts)."""
        d = (bp[None] - self.means).reshape(-1)                    # (8*69,)
        prod = self.prec_bd @ d
        nll = 0.5 * (prod * d).reshape(8, 69).sum(-1) - torch.log(self.nllw)
        if comp is None:
            return nll.min()
        return (nll * (torch.arange(8, device=nll.device) == comp).to(nll.dtype)).sum()

    def loss_cam(self, x, tgt, est, bp0):
        go, cam = x[:3], x[3:6]
        J = self.joints(go, bp0, torch.zeros(10, dtype=x.dtype, device=x.device)) + cam
        return ((tgt[ALIGN] - J[ALIGN]) ** 2).sum() + (100.0 ** 2) * ((cam - est) ** 2).sum()

    def loss_body(self, x, tgt, bp0, w, comp=None):
        bp, betas, go, cam = x[:69], x[69:79], x[79:82], x[82:85]
        J = self.joints(go, bp, betas) + cam
        d2 = (J - tgt) ** 2
        jl = (100.0 ** 2 * d2) / (100.0 ** 2 + d2)                   # gmof, sigma 100, confidence 1
        loss = 600.0 ** 2 * jl.sum()
        loss = loss + (4.78 * 1.5) ** 2 * self.gmm(bp, comp)
        loss = loss + 15.2 ** 2 * (torch.exp(bp[ANG_IDX] * self.ang_sgn) ** 2).sum()
        loss = loss + 5.0 ** 2 * (betas ** 2).sum()
        loss = loss + w ** 2 * ((bp - bp0) ** 2).sum()
        return loss

    def hess_body(self, x, tgt, bp0, w):
        """Hessian of loss_body: exact for every prior term (GMM: precision of the active component), and
        gmof''(r) J^T J for the joint term (drops the 3%-sized FK-curvature x residual part). Used only to pick
        the step; acceptance, gradient and convergence use the exact objective."""
        def jf(xx):
            return self.joints(xx[79:82], xx[:69], xx[69:79]) + xx[82:85]
        Jx = jacfwd(jf)(x).reshape(66, 85)
        r = (jf(x) - tgt).reshape(66)
        s2 = 100.0 ** 2
        fpp = 2 * s2 * s2 * (s2 - 3 * r * r) / (s2 + r * r) ** 3
        H = (600.0 ** 2) * (Jx.T * fpp) @ Jx
        bp = x[:69]
        d = (bp[None] - self.means).reshape(-1)
        nll = 0.5 * ((self.prec_bd @ d) * d).reshape(8, 69).sum(-1) - torch.log(self.nllw)
        P = (self.prec * (torch.arange(8, device=x.device) == nll.argmin())[:, None, None].to(x.dtype)).sum(0)
        hb = (4.78 * 1.5) ** 2 * P + 2 * w ** 2 * torch.eye(69, dtype=x.dtype, device=x.device)
        ang = torch.zeros(69, dtype=x.dtype, device=x.device)
        ang = ang.index_add(0, torch.tensor(ANG_IDX, device=x.device), 4 * 15.2 ** 2 * torch.exp(2 * bp[ANG_IDX] * self.ang_sgn))
        hb = hb + torch.diag(ang)
        H = H + torch.block_diag(hb, 2 * 25.0 * torch.eye(10, dtype=x.dtype, device=x.device), torch.zeros(6, 6, dtype=x.dtype, device=x.device))
        return H

    # ---- batched damped Newton ----
    def _newton(self, f, x, args, in_dims, iters, lam0=1e-6, hfun=None):
        F = vmap(f, in_dims=(0,) + in_dims)
        G = vmap(grad(f), in_dims=(0,) + in_dims)
        Hs = vmap(hessian(f) if hfun is None else hfun, in_dims=(0,) + in_dims)
        n, P = x.shape
        lam = lam0.clone() if torch.is_tensor(lam0) else torch.full((n,), lam0, dtype=x.dtype, device=x.device)
        fx = F(x, *args)
        I = torch.eye(P, dtype=x.dtype, device=x.device)
        conv = torch.zeros(n, dtype=torch.bool, device=x.device)
        for it in range(iters):
            g = G(x, *args)
            H = Hs(x, *args)
            dg = torch.diagonal(H, dim1=-2, dim2=-1).abs().clamp_min(1e-9)
            A = H + lam[:, None, None] * torch.diag_embed(dg)
            L, info = torch.linalg.cholesky_ex(A)
            step = -torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
            bad = (info != 0) | ~torch.isfinite(step).all(-1)
            step = torch.where(bad[:, None], torch.zeros_like(step), step)
            xn = x + step
            fn = F(xn, *args)
            acc = (~bad) & torch.isfinite(fn) & (fn < fx) & ~conv
            small = acc & ((fx - fn) <= 1e-13 * fx.abs())
            x = torch.where(acc[:, None], xn, x)
            fx = torch.where(acc, fn, fx)
            lam = torch.where(acc, (lam * 0.2).clamp_min(1e-12), (lam * 10.0).clamp_max(1e12))
            # converged: tiny relative improvement, or damping saturated with no progress possible
            conv = conv | small | (lam >= 1e12)
            if bool(conv.all()):
                break
        g = G(x, *args)
        return x, fx, g, it + 1

    def fit(self, targets_prepared: np.ndarray, cam_iters=60, body_iters=200, starts=(1e-6, 1e2, 1e3, 1e5, 1e6), chunk=256):
        """targets_prepared: (N,22,3) foot-corrected frame-0 targets (beta_fitting._prepare_target_joints).
        Multi-start damped Newton (different initial Levenberg damping = different descent paths through the
        non-convex landscape); the lowest-loss solution per clip is kept."""
        X = np.asarray(targets_prepared, np.float32)
        outs = [self._fit_chunk(X[i:i + chunk], cam_iters, body_iters, starts) for i in range(0, len(X), chunk)]
        return {k: (np.concatenate([o[k] for o in outs]) if isinstance(outs[0][k], np.ndarray) else [o[k] for o in outs]) for k in outs[0]}

    def _fit_chunk(self, X, cam_iters, body_iters, starts):
        kw = dict(dtype=self.dt, device=self.dev)
        tgt = torch.tensor(X, **kw)
        n = tgt.shape[0]
        go0 = self.mean_pose[:3].expand(n, 3)
        bp0 = self.mean_pose[3:]
        J0 = vmap(lambda g: self.joints(g, bp0, torch.zeros(10, **kw)))(go0)
        est = (tgt[:, ALIGN] - J0[:, ALIGN]).mean(1)
        x = torch.cat([go0, est], -1)
        x, fc, gc, ic = self._newton(self.loss_cam, x, (tgt, est, bp0), (0, 0, None), cam_iters)
        xb0 = torch.cat([bp0.expand(n, 69), torch.zeros(n, 10, **kw), x[:, :3], x[:, 3:]], -1)
        f5 = lambda xx, t, b: self.loss_body(xx, t, b, 5.0)
        S = len(starts)
        lam0 = torch.tensor(starts, **kw).repeat(n)                     # problem-major: (n*S,)
        h5 = lambda xx, t, b: self.hess_body(xx, t, b, 5.0)
        xs, fs, gs, ib = self._newton(f5, xb0.repeat_interleave(S, 0), (tgt.repeat_interleave(S, 0), bp0), (0, None), body_iters,
                                      lam0=lam0, hfun=h5 if self.gn else None)
        fs = fs.reshape(n, S)
        best = fs.argmin(1)
        ar = torch.arange(n, device=self.dev)
        xb = xs.reshape(n, S, -1)[ar, best]
        fb = fs[ar, best]
        gb = gs.reshape(n, S, -1)[ar, best]
        f0 = vmap(lambda xx, t: self.loss_body(xx, t, bp0, 0.0))(xb, tgt)
        J = vmap(self.joints)(xb[:, 79:82], xb[:, :69], xb[:, 69:79]) + xb[:, None, 82:85]
        err = torch.linalg.vector_norm(J - tgt, dim=-1).mean(-1)
        return dict(betas=xb[:, 69:79].float().cpu().numpy(), loss_w5=fb.cpu().numpy(), loss_w0=f0.cpu().numpy(),
                    mean_joint_err=err.cpu().numpy(), grad_max=gb.abs().max(-1).values.cpu().numpy(),
                    start_used=best.cpu().numpy(), loss_starts=fs.cpu().numpy(), iters_cam=ic, iters_body=ib)
