"""E25 — masking the redundant rotational DOF per joint.

Two independent mechanisms, both driven by the SAME per-joint DOF descriptor that
prep_dof_features.py already writes ({emb}_dof.npz: 3 slots x [axis, limited,
half_range/pi, center/pi] + dof_count/3, plus the static rest rotation R_rest):

  arm A  "head"  OUTPUT-side.  The decoder emits per-joint hinge ANGLES in the
         3-slot structure; inactive slots are dropped, limited slots are squashed
         into their limit box by a tanh, and the angles are FK'd through the target
         skeleton (including R_rest) to produce GLOBAL rotations.  The raw output is
         therefore ON the mechanical manifold and INSIDE the limit box by
         construction — the projection layer has nothing left to do.

  arm B  "loss"  LOSS-side.  The decoder keeps its free 6D global-rotation head; the
         TRAINING LOSS first projects the predicted (and target) local rotations onto
         each joint's hinge manifold, so gradients never chase components the
         mechanism cannot realize.  The raw output is unchanged in kind.

Graceful fallback (both arms): a joint with 3 active DOF (SMPL/BVH ball joints,
kuavo_s45's triads) genuinely has 3 rotational DOF, so NOTHING is masked there — the
free 6D local rotation is used verbatim.  The ROOT is a free joint and always passes
through its 6D global rotation.  A joint with 0 DOF (welded frame) is pinned to
R_rest, which is exactly what the mechanism does.

CONVENTION (identical to manifold_project.py / DOF_FEATURES.md):
    R_local(q) = rot(A_0,q_0) rot(A_1,q_1) rot(A_2,q_2) @ R_rest
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import rot6d_to_matrix, matrix_to_rot6d
from manifold_project import _slot_axes, _limits, _rot_axis_angle, _best_angle_about


class DofKin:
    """Static per-skeleton kinematics needed by both arms. Built once at load."""

    def __init__(self, spec, dof, rest, device="cpu", dtype=torch.float32):
        dof = dof.to(device, dtype)
        self.J = int(dof.shape[0])
        self.parents = np.asarray(spec.parents, np.int64)
        self.root = int(np.where(self.parents < 0)[0][0])
        self.off = torch.as_tensor(np.asarray(spec.rest_offsets), dtype=dtype, device=device)
        self.A, self.active = _slot_axes(dof)                   # [J,3,3], [J,3]
        lo, hi, limited = _limits(dof, 1.0)
        self.lo, self.hi, self.limited = lo, hi, limited        # [J,3]
        self.half = (hi - lo) / 2
        self.ctr = (hi + lo) / 2
        self.n_dof = (dof[:, 18] * 3).round().long()            # [J]
        self.rest = rest.to(device, dtype)                      # [J,3,3]
        # joints whose local rotation is NOT reconstructible from masked hinge angles
        free = torch.zeros(self.J, dtype=torch.bool, device=device)
        free[self.root] = True                                  # free/ball root: data, not a hinge
        free |= (self.n_dof >= 3)                               # genuine 3-DOF -> no masking
        self.free = free                                        # [J] bool
        # A skeleton with NO hinge joints at all (human_smpl, soma77) has no redundancy to
        # remove, so both arms short-circuit to the current behaviour EXACTLY: the free 6D
        # head keeps predicting GLOBAL rotations directly, with no FK re-parameterisation.
        self.all_free = bool(free.all())
        self.device, self.dtype = device, dtype
        # how much redundancy is actually being removed (for the report)
        self.masked_slots = int((~free).sum() * 3 - self.active[~free].sum())
        self.free_joints = int(free.sum())

    def to(self, device):
        if str(device) == str(self.device):
            return self
        import copy
        k = copy.copy(self)
        for a in ("off", "A", "active", "lo", "hi", "limited", "half", "ctr", "n_dof", "rest", "free"):
            setattr(k, a, getattr(self, a).to(device))
        k.device = device
        return k

    def summary(self):
        h = [int((self.n_dof == d).sum()) for d in range(4)]
        return {"J": self.J, "dof_hist_0123": h, "free_joints": self.free_joints,
                "hinge_joints": self.J - self.free_joints,
                "slots_masked_out": self.masked_slots,
                "limited_slots": int(self.limited[self.active].sum())}


# --------------------------------------------------------------------------
# arm A: angles -> global rotations
# --------------------------------------------------------------------------

def angles_to_local(raw, kin):
    """raw [B,J,3] unbounded -> (q [B,J,3], L [B,J,3,3]) local rotations of the HINGE
    joints.  Limited slots: q = center + half * tanh(raw)  -> structurally inside the
    box.  Unlimited active slots: q = pi * tanh(raw) (a continuous joint's angle is
    only defined mod 2pi, so a bounded chart loses nothing and keeps the scale sane).
    Inactive slots: q = 0."""
    q = torch.tanh(raw)
    q = torch.where(kin.limited, kin.ctr + kin.half * q, np.pi * q)
    q = torch.where(kin.active, q, torch.zeros_like(q))
    Ax = kin.A[None].expand(q.shape[0], -1, -1, -1)
    L = (_rot_axis_angle(Ax[:, :, 0], q[:, :, 0])
         @ _rot_axis_angle(Ax[:, :, 1], q[:, :, 1])
         @ _rot_axis_angle(Ax[:, :, 2], q[:, :, 2])
         @ kin.rest[None])
    return q, L


def fk_locals(L, kin, root_R):
    """local rotations [B,J,3,3] + root global [B,3,3] -> global rotations [B,J,3,3]."""
    gR = [None] * kin.J
    for j in range(kin.J):
        p = int(kin.parents[j])
        gR[j] = root_R if p < 0 else gR[p] @ L[:, j]
    return torch.stack(gR, 1)


def head_to_global6d(rot6d, ang, kin):
    """arm A decoder head -> GLOBAL rot6d [B,J,6], the same object the free head emits.

    rot6d [B,J,6]: the free 6D branch, used for the root and for genuine 3-DOF joints.
    ang   [B,J,3]: the hinge-angle branch, used for every other joint."""
    if kin.all_free:                     # all-ball skeleton: unchanged global-6D behaviour
        return rot6d
    _, L = angles_to_local(ang, kin)
    Lfree = rot6d_to_matrix(rot6d)                                  # local for free joints
    L = torch.where(kin.free[None, :, None, None], Lfree, L)
    gR = fk_locals(L, kin, Lfree[:, kin.root])
    return matrix_to_rot6d(gR)


# --------------------------------------------------------------------------
# arm B: project global rotations onto the per-joint hinge manifold
# --------------------------------------------------------------------------

def globals_to_locals(Rg, kin):
    """[B,J,3,3] global -> [B,J,3,3] local (root local := global)."""
    par = torch.as_tensor(kin.parents, device=Rg.device)
    Rl = Rg[:, par.clamp_min(0)].transpose(-1, -2) @ Rg
    return torch.where((par < 0)[None, :, None, None], Rg, Rl)


def project_locals(L, kin, iters=4):
    """Nearest local rotation ON the hinge manifold, per joint. Differentiable
    (closed-form atan2 sweeps, no branching on the data). Free joints pass through."""
    M = L @ kin.rest.transpose(-1, -2)[None]                        # target axis product
    B = M.shape[0]
    Ax = kin.A[None].expand(B, -1, -1, -1)
    Rs = [torch.eye(3, device=L.device, dtype=L.dtype).expand(B, kin.J, 3, 3)] * 3
    n_it = 1 if int(kin.n_dof[~kin.free].max().item() if (~kin.free).any() else 0) <= 1 else iters
    for _ in range(n_it):
        for s in range(3):
            pre = Rs[0] if s == 1 else (Rs[0] @ Rs[1] if s == 2 else None)
            post = (Rs[1] @ Rs[2]) if s == 0 else (Rs[2] if s == 1 else None)
            X = M
            if pre is not None:
                X = pre.transpose(-1, -2) @ X
            if post is not None:
                X = X @ post.transpose(-1, -2)
            ang = _best_angle_about(X, Ax[:, :, s])
            ang = torch.where(kin.active[None, :, s], ang, torch.zeros_like(ang))
            Rs[s] = _rot_axis_angle(Ax[:, :, s], ang)
    Lp = Rs[0] @ Rs[1] @ Rs[2] @ kin.rest[None]
    return torch.where(kin.free[None, :, None, None], L, Lp)


def project_global6d(rot6d, kin, iters=4):
    """GLOBAL rot6d (or matrices) -> GLOBAL rotations rebuilt from the hinge-manifold
    projection of every local rotation. Root global preserved exactly."""
    Rg = rot6d_to_matrix(rot6d) if rot6d.shape[-1] == 6 else rot6d
    if kin.all_free:                     # nothing to project; identity, exactly
        return Rg
    Lp = project_locals(globals_to_locals(Rg, kin), kin, iters=iters)
    return fk_locals(Lp, kin, Rg[:, kin.root])


# --------------------------------------------------------------------------
# diagnostics (eval only, no grad)
# --------------------------------------------------------------------------

@torch.no_grad()
def manifold_distance_deg(rot6d, kin, iters=4):
    """Per-joint geodesic between a prediction's LOCAL rotation and its nearest
    hinge-manifold point. Free joints are 0 by construction. [B,J] deg."""
    from common import geodesic_deg
    Rg = rot6d_to_matrix(rot6d) if rot6d.shape[-1] == 6 else rot6d
    if kin.all_free:
        return torch.zeros(Rg.shape[:2], device=Rg.device, dtype=Rg.dtype)
    L = globals_to_locals(Rg, kin)
    return geodesic_deg(L, project_locals(L, kin, iters=iters))


@torch.no_grad()
def limit_violation(rot6d, kin, iters=4):
    """Unclamped hinge-angle fit of a prediction, then how often it lies outside the
    mechanical limit box. -> (dof_pct, frame_pct, worst_overshoot_deg)."""
    Rg = rot6d_to_matrix(rot6d) if rot6d.shape[-1] == 6 else rot6d
    M = globals_to_locals(Rg, kin) @ kin.rest.transpose(-1, -2)[None]
    B = M.shape[0]
    Ax = kin.A[None].expand(B, -1, -1, -1)
    Rs = [torch.eye(3, device=Rg.device, dtype=Rg.dtype).expand(B, kin.J, 3, 3)] * 3
    q = torch.zeros(B, kin.J, 3, device=Rg.device, dtype=Rg.dtype)
    for _ in range(iters):
        for s in range(3):
            pre = Rs[0] if s == 1 else (Rs[0] @ Rs[1] if s == 2 else None)
            post = (Rs[1] @ Rs[2]) if s == 0 else (Rs[2] if s == 1 else None)
            X = M
            if pre is not None:
                X = pre.transpose(-1, -2) @ X
            if post is not None:
                X = X @ post.transpose(-1, -2)
            ang = _best_angle_about(X, Ax[:, :, s])
            ang = torch.where(kin.active[None, :, s], ang, torch.zeros_like(ang))
            q[:, :, s] = ang
            Rs[s] = _rot_axis_angle(Ax[:, :, s], ang)
    chk = kin.limited & kin.active & (~kin.free)[:, None]           # [J,3]
    over = torch.zeros_like(q)
    over = torch.maximum(kin.lo[None] - q, q - kin.hi[None]).clamp_min(0)
    over = torch.where(chk[None], over, torch.zeros_like(over))
    bad = over > 1e-6
    n = int(chk.sum())
    if n == 0:
        return 0.0, 0.0, 0.0
    return (100.0 * float(bad.sum()) / (B * n),
            100.0 * float(bad.flatten(1).any(1).float().mean()),
            float(torch.rad2deg(over.max())))


def build_kin(cache, emb, spec, device="cpu"):
    from manifold_project import load_dof
    dof, rest = load_dof(cache, emb, device=device)
    return DofKin(spec, dof, rest, device=device)
