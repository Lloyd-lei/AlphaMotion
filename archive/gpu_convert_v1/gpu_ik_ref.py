"""Batched GPU re-implementation of hiphi2smplx's MinkPositionSolver (mink 1.0.0 / MuJoCo 3.4.0 math).

One "lane" = one clip. All lanes advance one source frame per `step()` in lock-step; inside a step every
lane runs the exact per-frame algorithm of `MinkPositionSolver.solve`:
  * warm start = previous frame's float32 solution (or the copied reference pose on frame 0),
  * frame tasks on all 22 sites (SE(3) right-minus error, -jlog @ body Jacobian, per-task LM damping),
  * posture task on the ball joints (free-joint dofs zeroed, as mink 1.0 does),
  * unconstrained QP (ball/free joints have no limits -> ConfigurationLimit is inactive),
    solved with a batched Cholesky (daqp returns the same minimiser),
  * 8-step backtracking line search on the weighted position error, early stop when the improvement
    is <= tolerance**2, max `iterations` iterations; lanes that stopped are frozen (masking),
  * outputs rounded to float32 exactly like the CPU path (pose rotvec, translation, joints, mean error).
Everything is float64 (the CPU path is float64 inside MuJoCo/mink); scipy/MuJoCo formulas are mirrored
(rotvec<->quat, mju_quat2Mat, mju_mat2Quat, mju_quatIntegrate, mju_subQuat) so trajectories agree with
the CPU solver to ~1e-9 m except where a borderline early-stop/line-search decision flips.
"""
from __future__ import annotations

import math

import numpy as np
import torch

torch.backends.cuda.preferred_linalg_library("cusolver")  # MAGMA batched potrs is not CUDA-graph capturable

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
         "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
EXTREMITY = {"left_ankle", "right_ankle", "left_foot", "right_foot", "left_wrist", "right_wrist"}
ARM_PAIRS = ((16, 18), (17, 19), (18, 20), (19, 21))
NJ = 22
NV = 6 + 3 * 21
LEVELS = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17], [18, 19], [20, 21]]
MJMINVAL = 1e-15


def ancestor_mask():
    m = np.zeros((NJ, NJ), bool)
    for j in range(NJ):
        k = j
        while k >= 0:
            m[j, k] = True
            k = PARENTS[k]
    return m


# ----------------------------------------------------------------------------------------------
# scipy (1.15) / MuJoCo-compatible rotation helpers, float64, arbitrary leading dims
# ----------------------------------------------------------------------------------------------
def rotvec_to_quat(rv):
    """scipy Rotation.from_rotvec(rv).as_quat(scalar_first=True) -> wxyz."""
    ang = torch.linalg.vector_norm(rv, dim=-1, keepdim=True)
    a2 = ang * ang
    small = ang <= 1e-3
    scale = torch.where(small, 0.5 - a2 / 48 + a2 * a2 / 3840, torch.sin(ang / 2) / torch.where(small, torch.ones_like(ang), ang))
    return torch.cat([torch.cos(ang / 2), rv * scale], -1)


def quat_to_rotvec(q):
    """scipy Rotation.from_quat(q, scalar_first=True).as_rotvec() (normalises, w>=0)."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    ang = 2 * torch.atan2(torch.linalg.vector_norm(v, dim=-1, keepdim=True), q[..., :1])
    a2 = ang * ang
    small = ang <= 1e-3
    scale = torch.where(small, 2 + a2 / 12 + 7 * a2 * a2 / 2880, ang / torch.sin(torch.where(small, torch.ones_like(ang), ang) / 2))
    return scale * v


def scipy_quat_to_mat(q):
    """scipy as_matrix() of a (unit) wxyz quaternion."""
    w, x, y, z = q.unbind(-1)
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    m = torch.stack([x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),
                     2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw),
                     2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2], -1)
    return m.reshape(q.shape[:-1] + (3, 3))


def mj_quat2mat(q):
    w, x, y, z = q.unbind(-1)
    q00, q01, q02, q03 = w * w, w * x, w * y, w * z
    q11, q12, q13 = x * x, x * y, x * z
    q22, q23, q33 = y * y, y * z, z * z
    m = torch.stack([q00 + q11 - q22 - q33, 2 * (q12 - q03), 2 * (q13 + q02),
                     2 * (q12 + q03), q00 - q11 + q22 - q33, 2 * (q23 - q01),
                     2 * (q13 - q02), 2 * (q23 + q01), q00 - q11 - q22 + q33], -1)
    return m.reshape(q.shape[:-1] + (3, 3))


def mj_mulquat(a, b):
    a0, a1, a2, a3 = a.unbind(-1)
    b0, b1, b2, b3 = b.unbind(-1)
    return torch.stack([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0], -1)


def mj_normalize4(q):
    n = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1
    out = torch.where((n - 1).abs() > MJMINVAL, q / torch.where(n < MJMINVAL, torch.ones_like(n), n), q)
    return torch.where(n < MJMINVAL, ident, out)


def mj_mat2quat(m):
    """mju_mat2Quat (branch on largest component) + normalize."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    one = torch.ones_like(m00)
    tr = m00 + m11 + m22
    c0 = tr > 0
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    c2 = (~c0) & (~c1) & (m11 > m22)
    s0 = 0.5 * torch.sqrt(torch.clamp(1 + tr, min=0))
    s1 = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0))
    s2 = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0))
    s3 = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0))
    d0 = torch.where(c0, s0, one); d1 = torch.where(c1, s1, one)
    d2 = torch.where(c2, s2, one); d3 = torch.where(c0 | c1 | c2, one, s3)
    q0 = torch.stack([s0, 0.25 * (m21 - m12) / d0, 0.25 * (m02 - m20) / d0, 0.25 * (m10 - m01) / d0], -1)
    q1 = torch.stack([0.25 * (m21 - m12) / d1, s1, 0.25 * (m01 + m10) / d1, 0.25 * (m02 + m20) / d1], -1)
    q2 = torch.stack([0.25 * (m02 - m20) / d2, 0.25 * (m01 + m10) / d2, s2, 0.25 * (m12 + m21) / d2], -1)
    q3 = torch.stack([0.25 * (m10 - m01) / d3, 0.25 * (m02 + m20) / d3, 0.25 * (m12 + m21) / d3, s3], -1)
    q = torch.where(c0[..., None], q0, torch.where(c1[..., None], q1, torch.where(c2[..., None], q2, q3)))
    return mj_normalize4(q)


def skew(v):
    z = torch.zeros_like(v[..., 0])
    x, y, w = v.unbind(-1)
    return torch.stack([z, -w, y, w, z, -x, -y, x, z], -1).reshape(v.shape[:-1] + (3, 3))


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def rot_vec_quat(v, q):
    return (mj_quat2mat(q) @ v.unsqueeze(-1)).squeeze(-1)


def so3_log_quat(q):
    """mink SO3.log on a wxyz quaternion (sign-canonicalised by w)."""
    sgn = torch.sign(q[..., :1])
    q = q * sgn
    w, v = q[..., :1], q[..., 1:]
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    vn = v / torch.where(n < MJMINVAL, torch.ones_like(n), n)
    out = 2 * torch.atan2(n, w) * vn
    return torch.where(n < 1e-10, torch.zeros_like(out), out)


def se3_log(q, t):
    """mink SE3.log for (wxyz, xyz) -> (v[3], w[3])."""
    om = so3_log_quat(q)
    th = torch.linalg.vector_norm(om, dim=-1, keepdim=True)[..., None]  # (...,1,1)
    t2 = th * th
    W = skew(om)
    W2 = W @ W
    I = torch.eye(3, dtype=q.dtype, device=q.device).expand(W.shape)
    small = t2 < 1e-10
    ht = 0.5 * th
    ths = torch.where(small, torch.ones_like(th), th)
    coef = (1.0 - 0.5 * ths * torch.cos(0.5 * ths) / torch.sin(0.5 * ths)) / torch.where(small, torch.ones_like(t2), t2)
    vinv = torch.where(small, I - 0.5 * W + W2 / 12.0, I - 0.5 * W + coef * W2)
    del ht
    return torch.cat([(vinv @ t.unsqueeze(-1)).squeeze(-1), om], -1)


def so3_ljacinv(w):
    th = torch.linalg.vector_norm(w, dim=-1, keepdim=True)[..., None]
    t2 = th * th
    small = th < 1e-10
    ths = torch.where(small, torch.ones_like(th), th)
    beta_big = (1.0 / (ths * ths)) * (1.0 - (ths * torch.sin(ths) / (2.0 * (1.0 - torch.cos(ths)))))
    beta_small = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
    beta = torch.where(small, beta_small, beta_big)
    inner = (w * w).sum(-1)[..., None, None]
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    outer = w.unsqueeze(-1) * w.unsqueeze(-2)
    return (outer - inner * I) * beta - 0.5 * skew(w) + I


def se3_getQ(c):
    v, w = c[..., :3], c[..., 3:]
    th = torch.linalg.vector_norm(w, dim=-1)[..., None, None]
    t2 = th * th
    small = t2 < 1e-10
    ths = torch.where(small, torch.ones_like(th), th)
    t4 = ths ** 4
    s, co = torch.sin(ths), torch.cos(ths)
    B = torch.where(small, 1.0 / 6.0 + t2 / 120.0, (ths - s) / (ths * ths * ths))
    C = torch.where(small, -1.0 / 24.0 + t2 / 720.0, (1.0 - 0.5 * ths * ths - co) / t4)
    D = torch.where(small, torch.full_like(th, -1.0 / 60.0), (2.0 * ths - 3.0 * s + ths * co) / (2.0 * t4 * ths))
    V = skew(v)
    Wm = skew(w)
    VW = V @ Wm
    WV = VW.transpose(-1, -2)
    WVW = WV @ Wm
    VWW = VW @ Wm
    return 0.5 * V + B * (WV + VW + WVW) - C * (VWW - VWW.transpose(-1, -2) - 3.0 * WVW) + D * (WVW @ Wm + Wm @ WVW)


def se3_jlog(c_log):
    """mink SE3.jlog(X) = rjacinv(log X) = ljacinv(-log X); input = log X."""
    o = -c_log
    w = o[..., 3:]
    th2 = (w * w).sum(-1)
    Ji = so3_ljacinv(w)
    Q = se3_getQ(o)
    top_right = -Ji @ Q @ Ji
    z = torch.zeros_like(Ji)
    J = torch.cat([torch.cat([Ji, top_right], -1), torch.cat([z, Ji], -1)], -2)
    I6 = torch.eye(6, dtype=c_log.dtype, device=c_log.device).expand(J.shape)
    return torch.where((th2 < 1e-10)[..., None, None], I6, J)


def mj_quat_integrate(q, vel, scale):
    """mju_quatIntegrate(quat, vel, scale)."""
    n = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
    ex = torch.zeros_like(vel); ex[..., 0] = 1
    ax = torch.where(n < MJMINVAL, ex, vel / torch.where(n < MJMINVAL, torch.ones_like(n), n))
    ang = scale * n
    qr = torch.cat([torch.cos(ang * 0.5), ax * torch.sin(ang * 0.5)], -1)
    ident = torch.zeros_like(qr)
    ident[..., 0] = 1
    qr = torch.where(ang == 0, ident, qr)
    return mj_mulquat(mj_normalize4(q), qr)


def mj_sub_quat_vel(qa, qb):
    """mj_differentiatePos for a ball joint: log(qb^-1 qa) via mju_subQuat/mju_quat2Vel (inputs normalised)."""
    qa = mj_normalize4(qa)
    qb = mj_normalize4(qb)
    d = mj_mulquat(quat_conj(qb), qa)
    ax = d[..., 1:]
    s = torch.linalg.vector_norm(ax, dim=-1, keepdim=True)
    ex = torch.zeros_like(ax); ex[..., 0] = 1
    axn = torch.where(s < MJMINVAL, ex, ax / torch.where(s < MJMINVAL, torch.ones_like(s), s))
    speed = 2 * torch.atan2(s, d[..., :1])
    speed = torch.where(speed > math.pi, speed - 2 * math.pi, speed)
    return axn * speed


def align_vectors(src, tgt):
    """hiphi2smplx.kinematics.align_vectors, batched."""
    sn = torch.linalg.vector_norm(src, dim=-1, keepdim=True)
    tn = torch.linalg.vector_norm(tgt, dim=-1, keepdim=True)
    degen = (sn < 1e-8) | (tn < 1e-8)
    s = src / torch.where(degen, torch.ones_like(sn), sn)
    t = tgt / torch.where(degen, torch.ones_like(tn), tn)
    cr = torch.cross(s, t, dim=-1)
    sine = torch.linalg.vector_norm(cr, dim=-1, keepdim=True)
    cosine = torch.clamp((s * t).sum(-1, keepdim=True), -1.0, 1.0)
    par = sine < 1e-8
    axis = cr / torch.where(par, torch.ones_like(sine), sine)
    R = scipy_quat_to_mat(rotvec_to_quat(torch.atan2(sine, cosine) * axis))
    # anti-parallel: pi rotation about s x e_argmin|s|
    basis = (torch.arange(3, device=s.device) == s.abs().argmin(-1, keepdim=True)).to(s.dtype)
    ax2 = torch.cross(s, basis, dim=-1)
    ax2 = ax2 / torch.linalg.vector_norm(ax2, dim=-1, keepdim=True).clamp_min(1e-300)
    Rpi = scipy_quat_to_mat(rotvec_to_quat(math.pi * ax2))
    I = torch.eye(3, dtype=src.dtype, device=src.device).expand(R.shape)
    R = torch.where(par[..., None], torch.where((cosine > 0)[..., None], I, Rpi), R)
    return torch.where(degen[..., None], I, R)


def local_to_global(Rl):
    out = [None] * NJ
    for j, p in enumerate(PARENTS):
        out[j] = Rl[..., j, :, :] if p < 0 else out[p] @ Rl[..., j, :, :]
    return torch.stack(out, -3)


# ----------------------------------------------------------------------------------------------
class BatchedMinkIK:
    """Lanes = clips. Per-lane constants: rest joints (22,3) and MJCF body offsets (22,3)."""

    def __init__(self, rest_joints: np.ndarray, device="cuda", iterations=10, dt=0.01, damping=1e-2,
                 posture_cost=0.1, root_posture_cost=0.1, position_cost=1.0, ankle_wrist_toe_weight=5.0,
                 orientation_cost=0.1, tolerance=1e-5, line_search_steps=8):
        rj = np.asarray(rest_joints, np.float32).astype(np.float64)  # (B,22,3) float32 -> float64 like the CPU
        B = rj.shape[0]
        off = np.zeros_like(rj)
        for j, p in enumerate(PARENTS):
            if p >= 0:
                d = rj[:, j] - rj[:, p]
                off[:, j] = np.vectorize(lambda x: float(f"{x:.9g}"))(d)  # MJCF pos written with %.9g
        f64 = dict(dtype=torch.float64, device=device)
        self.B, self.dev, self.iters, self.dt = B, device, iterations, dt
        self.rest = torch.tensor(rj, **f64)
        self.off = torch.tensor(off, **f64)
        self.damping = damping
        self.tol2 = tolerance ** 2
        self.ls = line_search_steps
        pc = np.array([position_cost * (ankle_wrist_toe_weight if n in EXTREMITY else 1.0) for n in NAMES])
        oc = np.array([root_posture_cost] + [orientation_cost] * 21)
        self.pc = torch.tensor(pc, **f64)
        self.W = torch.tensor(np.concatenate([np.repeat(pc[:, None], 3, 1), np.repeat(oc[:, None], 3, 1)], 1), **f64)  # (22,6)
        wp = np.full(NV, posture_cost); wp[:6] = 0.0  # free-joint dofs zeroed by mink 1.0 PostureTask
        self.Wp = torch.tensor(wp, **f64)
        self.anc = torch.tensor(ancestor_mask(), device=device)
        self.scales = torch.tensor([0.5 ** k for k in range(line_search_steps)], **f64)
        self.eyeNV = torch.eye(NV, **f64)
        self.lev_idx = [torch.tensor(lev, device=device) for lev in LEVELS]

    # ---- kinematics (MuJoCo mj_kinematics on the generated MJCF) ----
    def fk(self, root, q, off=None):
        """root (...,3), q (...,22,4) wxyz (q[...,0] = free-joint quat). Returns xquat, xmat, xpos (...,22,*)."""
        off = self.off if off is None else off
        xq = [None] * NJ; xR = [None] * NJ; xp = [None] * NJ
        xq[0] = mj_normalize4(q[..., 0, :]); xR[0] = mj_quat2mat(xq[0]); xp[0] = root
        for li, lev in enumerate(LEVELS[1:], start=1):
            idx = self.lev_idx[li]
            par = [PARENTS[j] for j in lev]
            pR = torch.stack([xR[p] for p in par], -3)
            pq = torch.stack([xq[p] for p in par], -2)
            pp = torch.stack([xp[p] for p in par], -2)
            o = off.index_select(-2, idx)
            pos = (pR @ o.unsqueeze(-1)).squeeze(-1) + pp
            quat = mj_normalize4(mj_mulquat(pq, mj_normalize4(q.index_select(-2, idx))))
            R = mj_quat2mat(quat)
            for k, j in enumerate(lev):
                xq[j] = quat[..., k, :]; xR[j] = R[..., k, :, :]; xp[j] = pos[..., k, :]
        return torch.stack(xq, -2), torch.stack(xR, -3), torch.stack(xp, -2)

    def werr(self, xp, tgt):
        d = (xp - tgt) * self.pc[:, None]
        return (d * d).sum((-1, -2))

    # ---- one source frame for all lanes ----
    def solve_frame(self, tgt, ref_rv, seed_rv, root0, active=None):
        """tgt (B,22,3) f64 targets; ref_rv (B,22,3) f64 copied rotations; seed_rv (B,22,3) f64 warm start
        (float32-rounded previous solution); root0 (B,3) initial free-joint position.
        Returns pose_rv (B,22,3) f64, root (B,3) f64, joints (B,22,3) f64, mean_err (B,), n_iter (B,)."""
        B = tgt.shape[0]
        q = rotvec_to_quat(seed_rv)
        qref = rotvec_to_quat(ref_rv)
        Rg = local_to_global(scipy_quat_to_mat(qref))
        Rg = Rg.clone()
        for a, c in ARM_PAIRS:
            rest_ray = self.rest[:, c] - self.rest[:, a]
            pred = (Rg[:, a] @ rest_ray.unsqueeze(-1)).squeeze(-1)
            tray = tgt[:, c] - tgt[:, a]
            Rg[:, a] = align_vectors(pred, tray) @ Rg[:, a]
        qt = mj_mat2quat(Rg)                       # target rotations (SO3.from_matrix)
        qt_inv = quat_conj(qt)
        tt_inv = -rot_vec_quat(tgt, qt_inv)        # T_t^-1 translation
        root = root0.clone()
        prev = torch.full((B,), float("inf"), dtype=tgt.dtype, device=tgt.device)
        done = torch.zeros(B, dtype=torch.bool, device=tgt.device) if active is None else ~active
        nit = torch.zeros(B, dtype=torch.int32, device=tgt.device)
        W = self.W
        for _ in range(self.iters):
            xq, xR, xp = self.fk(root, q)
            cur = self.werr(xp, tgt)
            # frame-task error e = log(T_b^-1 T_t), Jacobian -jlog(T_t^-1 T_b) @ J_body
            qb = mj_mat2quat(xR)
            qb_inv = quat_conj(qb)
            q_bt = mj_mulquat(qb_inv, qt)
            t_bt = rot_vec_quat(tgt - xp, qb_inv)
            e = se3_log(q_bt, t_bt)                                   # (B,22,6)
            q_tb = mj_mulquat(qt_inv, qb)
            t_tb = rot_vec_quat(xp, qt_inv) + tt_inv
            jl = se3_jlog(se3_log(q_tb, t_tb))                       # (B,22,6,6)
            RjT = xR.transpose(-1, -2)
            A = RjT.unsqueeze(2) @ xR.unsqueeze(1)                   # (B,22,22,3,3) R_j^T R_b
            dl = (RjT.unsqueeze(2) @ (xp.unsqueeze(2) - xp.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)  # R_j^T (p_j-p_b)
            Jp = -(skew(dl) @ A)
            m = self.anc[None, :, :, None, None].to(A.dtype)
            Jrot = (A * m).transpose(2, 3).reshape(B, NJ, 3, NJ * 3)  # rows 3, cols per block b
            Jpos = (Jp * m).transpose(2, 3).reshape(B, NJ, 3, NJ * 3)
            Jb = torch.cat([torch.cat([RjT, Jpos], -1), torch.cat([torch.zeros_like(RjT), Jrot], -1)], -2)  # (B,22,6,69)
            Jt = -(jl @ Jb)
            WJ = W[None, :, :, None] * Jt
            We = W[None] * e
            WJf = WJ.reshape(B, NJ * 6, NV)
            H = WJf.transpose(1, 2) @ WJf
            mu = self.damping * (We * We).sum((-1, -2))
            c = (WJf.transpose(1, 2) @ We.reshape(B, NJ * 6, 1)).squeeze(-1)
            # posture task on ball joints
            ep = torch.zeros(B, NV, dtype=tgt.dtype, device=tgt.device)
            ep[:, 6:] = mj_sub_quat_vel(q[:, 1:], qref[:, 1:]).reshape(B, 63)
            Wpe = self.Wp * ep
            mup = self.damping * (Wpe * Wpe).sum(-1)
            H = H + torch.diag_embed(self.Wp * self.Wp).expand(B, NV, NV) + (mu + mup + self.damping)[:, None, None] * self.eyeNV
            c = c + self.Wp * Wpe
            L, info = torch.linalg.cholesky_ex(H)
            dq = -torch.cholesky_solve(c.unsqueeze(-1), L).squeeze(-1)
            v = dq / self.dt
            # line search: all scales at once (B,S)
            S = self.ls
            vs = v[:, None, :] * self.scales[None, :, None]           # velocity * step_scale
            root_c = root[:, None, :] + self.dt * vs[..., :3]
            q_c = torch.empty(B, S, NJ, 4, dtype=q.dtype, device=q.device)
            q_c[:, :, 0] = mj_quat_integrate(q[:, None, 0].expand(B, S, 4), vs[..., 3:6], self.dt)
            q_c[:, :, 1:] = mj_quat_integrate(q[:, None, 1:].expand(B, S, 21, 4), vs[..., 6:].reshape(B, S, 21, 3), self.dt)
            _, _, xp_c = self.fk(root_c, q_c, off=self.off[:, None])
            err_c = self.werr(xp_c, tgt[:, None])                     # (B,S)
            ok = err_c < cur[:, None]
            acc = ok.any(1)
            first = torch.argmax(ok.to(torch.int8), 1)
            ar = torch.arange(B, device=q.device)
            new_root = torch.where(acc[:, None], root_c[ar, first], root)
            new_q = torch.where(acc[:, None, None], q_c[ar, first], q)
            err = torch.where(acc, err_c[ar, first], cur)
            live = ~done
            root = torch.where(live[:, None], new_root, root)
            q = torch.where(live[:, None, None], new_q, q)
            nit = nit + live.to(torch.int32)
            stop = (prev - err) <= self.tol2
            prev = torch.where(live, err, prev)
            done = done | stop
        _, _, xp = self.fk(root, q)
        pose = quat_to_rotvec(q)
        mean_err = torch.linalg.vector_norm(xp - tgt, dim=-1).mean(-1)
        return pose, root, xp, mean_err, nit


class GraphRunner:
    """Captures K consecutive frame solves (with the float32 warm-start chaining of the CPU path) into one
    CUDA graph. Host loop: load K frames of targets/reference into static buffers, replay, read K outputs."""

    def __init__(self, ik: BatchedMinkIK, K=4, use_graph=True):
        self.ik, self.K = ik, K
        B, dev = ik.B, ik.dev
        f64 = dict(dtype=torch.float64, device=dev)
        self.tgt = torch.zeros(B, K, NJ, 3, **f64)
        self.ref = torch.zeros(B, K, NJ, 3, **f64)
        self.first = torch.zeros(B, K, dtype=torch.bool, device=dev)   # frame 0 of a clip at this slot
        self.active = torch.zeros(B, K, dtype=torch.bool, device=dev)  # lane has a real frame at this slot
        self.seed_rv = torch.zeros(B, NJ, 3, **f64)
        self.seed_t = torch.zeros(B, 3, **f64)
        self.o_pose = torch.zeros(B, K, NJ, 3, dtype=torch.float32, device=dev)
        self.o_trans = torch.zeros(B, K, 3, dtype=torch.float32, device=dev)
        self.o_joints = torch.zeros(B, K, NJ, 3, dtype=torch.float32, device=dev)
        self.o_loss = torch.zeros(B, K, dtype=torch.float32, device=dev)
        self.o_nit = torch.zeros(B, K, dtype=torch.int32, device=dev)
        self.graph = None
        if use_graph and str(dev).startswith("cuda"):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._body()
            torch.cuda.current_stream().wait_stream(s)
            self.seed_rv.zero_(); self.seed_t.zero_()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._body()

    def _body(self):
        ik = self.ik
        rest0 = ik.rest[:, 0]
        for k in range(self.K):
            tgt, ref = self.tgt[:, k], self.ref[:, k]
            first, act = self.first[:, k], self.active[:, k]
            s_rv = torch.where(first[:, None, None], ref, self.seed_rv)
            root0 = torch.where(first[:, None], tgt[:, 0], self.seed_t + rest0)
            pose, root, xp, me, nit = ik.solve_frame(tgt, ref, s_rv, root0, active=act)
            pose32 = pose.float(); tr32 = (root - rest0).float()
            self.o_pose[:, k] = pose32; self.o_trans[:, k] = tr32
            self.o_joints[:, k] = xp.float(); self.o_loss[:, k] = me.float(); self.o_nit[:, k] = nit
            self.seed_rv.copy_(torch.where(act[:, None, None], pose32.double(), self.seed_rv))
            self.seed_t.copy_(torch.where(act[:, None], tr32.double(), self.seed_t))

    def run(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
