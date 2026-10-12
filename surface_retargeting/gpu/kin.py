"""Batched rigid-body kinematics on GPU (torch), built from a compiled MjModel.

Reproduces mj_kinematics for trees of hinge/slide joints under one free root (all 21 factory robots), including the
MuJoCo conventions UMR relies on:
  * free joint: body pose = qpos[0:7] (pos, quat wxyz); qvel[0:3] world-frame translation, qvel[3:6] body-frame
    angular velocity (mj_integratePos / mju_quatIntegrate rotate in the local frame);
  * hinge: rotation about jnt_axis by (q - qpos0) at anchor jnt_pos (body frame); slide: translation along axis;
  * Jacobians identical to mj_jac (translational: axis x (p - anchor) / axis; rotational: axis).
Bodies are processed level-by-level (all bodies of equal tree depth in one batched op).
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch


def quat_to_mat(q):
    """q [...,4] wxyz (not necessarily normalised) -> [...,3,3]."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return r.reshape(q.shape[:-1] + (3, 3))


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def axis_angle_mat(axis, ang):
    """axis [...,3] unit, ang [...] -> Rodrigues [...,3,3]."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(axis.shape[:-1] + (3, 3))
    s = torch.sin(ang)[..., None, None]
    c = (1 - torch.cos(ang))[..., None, None]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(K.shape)
    return eye + s * K + c * (K @ K)


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(v.shape[:-1] + (3, 3))


class KinTree:
    def __init__(self, m: mujoco.MjModel, device='cuda', dtype=torch.float32):
        self.m = m
        self.device = torch.device(device)
        self.dtype = dtype
        nb, nv, nq = m.nbody, m.nv, m.nq
        self.nbody, self.nv, self.nq = nb, nv, nq
        f = dict(device=self.device, dtype=dtype)
        i64 = dict(device=self.device, dtype=torch.long)
        parent = m.body_parentid.astype(np.int64)
        depth = np.zeros(nb, np.int64)
        for b in range(1, nb):
            depth[b] = depth[parent[b]] + 1
        self.free_body = -1
        jt = np.full(nb, -1, np.int64)          # joint type per body (-1 none)
        jq = np.zeros(nb, np.int64); jd = np.zeros(nb, np.int64)
        jaxis = np.zeros((nb, 3)); jpos = np.zeros((nb, 3)); jref = np.zeros(nb)
        for b in range(nb):
            n = int(m.body_jntnum[b])
            if n == 0:
                continue
            if n > 1:
                raise NotImplementedError(f'body {b} has {n} joints')
            j = int(m.body_jntadr[b])
            t = int(m.jnt_type[j])
            if t == mujoco.mjtJoint.mjJNT_FREE:
                if self.free_body >= 0 or parent[b] != 0 or int(m.jnt_qposadr[j]) != 0:
                    raise NotImplementedError('only one free root joint at qpos[0:7] supported')
                self.free_body = b
            elif t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise NotImplementedError(f'joint type {t}')
            jt[b] = t; jq[b] = m.jnt_qposadr[j]; jd[b] = m.jnt_dofadr[j]
            jaxis[b] = m.jnt_axis[j]; jpos[b] = m.jnt_pos[j]; jref[b] = m.qpos0[m.jnt_qposadr[j]]
        self.levels = []
        for L in range(1, depth.max() + 1):
            bs = np.flatnonzero(depth == L)
            bs = bs[bs != self.free_body]
            if len(bs) == 0:
                continue
            self.levels.append(dict(
                b=torch.as_tensor(bs, **i64), p=torch.as_tensor(parent[bs], **i64),
                pos=torch.as_tensor(m.body_pos[bs], **f), R=quat_to_mat(torch.as_tensor(m.body_quat[bs], **f)),
                hinge=torch.as_tensor(jt[bs] == mujoco.mjtJoint.mjJNT_HINGE, **f),
                slide=torch.as_tensor(jt[bs] == mujoco.mjtJoint.mjJNT_SLIDE, **f),
                qadr=torch.as_tensor(np.where(jt[bs] >= 0, jq[bs], 0), **i64),
                axis=torch.as_tensor(np.where((jt[bs] >= 0)[:, None], jaxis[bs], [[0, 0, 1.0]]), **f),
                jpos=torch.as_tensor(jpos[bs], **f), ref=torch.as_tensor(jref[bs], **f),
            ))
        # dof table: for each dof its body, kind (0 trans-world, 1 rot-free-local, 2 hinge, 3 slide), local index
        dof_body = m.dof_bodyid.astype(np.int64)
        kind = np.zeros(nv, np.int64); fidx = np.zeros(nv, np.int64)
        for d in range(nv):
            b = dof_body[d]
            t = jt[b]
            if t == mujoco.mjtJoint.mjJNT_FREE:
                k = d - jd[b]
                kind[d] = 0 if k < 3 else 1; fidx[d] = k % 3
            elif t == mujoco.mjtJoint.mjJNT_HINGE:
                kind[d] = 2
            else:
                kind[d] = 3
        self.dof_body = torch.as_tensor(dof_body, **i64)
        self.dof_kind = torch.as_tensor(kind, **i64)
        self.dof_fidx = torch.as_tensor(fidx, **i64)
        self.dof_rot = torch.as_tensor((kind == 1) | (kind == 2), **f)
        # ancestor mask [nbody, nv]: dof d moves body b
        anc = np.zeros((nb, nv), bool)
        for b in range(nb):
            a = b
            while a > 0:
                anc[b, dof_body == a] = True
                a = parent[a]
        self.anc = torch.as_tensor(anc, **f)
        # scalar joints (qpos <-> dof)
        self.scalar_qadr = torch.as_tensor([jq[b] for b in range(nb) if jt[b] in (2, 3)], **i64)
        self.scalar_dadr = torch.as_tensor([jd[b] for b in range(nb) if jt[b] in (2, 3)], **i64)
        order = torch.argsort(self.scalar_dadr)
        self.scalar_qadr, self.scalar_dadr = self.scalar_qadr[order], self.scalar_dadr[order]
        # geoms
        self.geom_body = torch.as_tensor(m.geom_bodyid.astype(np.int64), **i64)
        self.geom_pos = torch.as_tensor(m.geom_pos, **f)
        self.geom_R = quat_to_mat(torch.as_tensor(m.geom_quat, **f))
        self.free_dof0 = int(jd[self.free_body]) if self.free_body >= 0 else -1
        self.local_dofs = torch.as_tensor(np.flatnonzero(~np.isin(kind, [0, 1])), **i64)

    # ------------------------------------------------------------------ forward kinematics
    def fk(self, q):
        """q [B,nq] -> dict(xpos [B,nb,3], xmat [B,nb,3,3], axis [B,nv,3], anchor [B,nv,3])."""
        B = q.shape[0]
        q = q.to(self.dtype)
        xpos = q.new_zeros(B, self.nbody, 3)
        xmat = torch.eye(3, dtype=self.dtype, device=self.device).repeat(B, self.nbody, 1, 1)
        jaxis = q.new_zeros(B, self.nbody, 3)
        janchor = q.new_zeros(B, self.nbody, 3)
        if self.free_body >= 0:
            xpos[:, self.free_body] = q[:, 0:3]
            xmat[:, self.free_body] = quat_to_mat(q[:, 3:7])
        for L in self.levels:
            Rp = xmat[:, L['p']]                                    # [B,n,3,3]
            xp = xpos[:, L['p']]
            Rpre = Rp @ L['R']
            xpre = xp + (Rp @ L['pos'][..., None])[..., 0]
            th = q[:, L['qadr']] - L['ref']                         # [B,n]
            ang = th * L['hinge']
            Rj = axis_angle_mat(L['axis'].expand(B, -1, -1), ang)
            Rb = Rpre @ Rj
            ax_w = (Rpre @ L['axis'][..., None])[..., 0]
            anchor = xpre + (Rpre @ L['jpos'][..., None])[..., 0]
            xb = anchor - (Rb @ L['jpos'][..., None])[..., 0] + ax_w * (th * L['slide'])[..., None]
            xpos[:, L['b']] = xb
            xmat[:, L['b']] = Rb
            jaxis[:, L['b']] = ax_w
            janchor[:, L['b']] = anchor
        # per-dof axis/anchor
        db = self.dof_body
        axis = jaxis[:, db].clone()
        anchor = janchor[:, db].clone()
        if self.free_body >= 0:
            f0 = self.free_dof0
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            axis[:, f0:f0 + 3] = eye
            axis[:, f0 + 3:f0 + 6] = xmat[:, self.free_body].transpose(-1, -2)   # columns of R -> rows
            anchor[:, f0:f0 + 6] = xpos[:, self.free_body, None]
        return dict(xpos=xpos, xmat=xmat, axis=axis, anchor=anchor)

    def geom_pose(self, K, gids):
        """gids [G] -> (pos [B,G,3], R [B,G,3,3])."""
        b = self.geom_body[gids]
        Rb = K['xmat'][:, b]
        pos = K['xpos'][:, b] + (Rb @ self.geom_pos[gids][..., None])[..., 0]
        return pos, Rb @ self.geom_R[gids]

    @staticmethod
    def local_to_world(gpos, gR, gidx, local):
        """gpos/gR indexed by gidx (same leading B), local [N,3] or [B,N,3] -> [B,N,3]."""
        R = gR[:, gidx]
        if local.dim() == 2:
            local = local[None].expand(R.shape[0], -1, -1)
        return (R @ local[..., None])[..., 0] + gpos[:, gidx]

    # ------------------------------------------------------------------ Jacobians
    def jac(self, K, P, body, rot=False):
        """P [B,N,3] world points rigidly attached to bodies body [N] or [B,N].
        Returns Jp [B,N,3,nv] (and Jr [B,N,3,nv] if rot)."""
        axis, anchor = K['axis'], K['anchor']                     # [B,nv,3]
        mask = self.anc[body]                                       # [N,nv] or [B,N,nv]
        if mask.dim() == 2:
            mask = mask[None]
        r = P[:, :, None, :] - anchor[:, None]                     # [B,N,nv,3]
        a = axis[:, None].expand_as(r)
        cr = torch.linalg.cross(a, r, dim=-1)
        rotm = self.dof_rot[None, None, :, None]
        Jp = torch.where(rotm > 0, cr, a) * mask[..., None]
        Jp = Jp.transpose(-1, -2)
        if not rot:
            return Jp
        Jr = (a * rotm * mask[..., None]).transpose(-1, -2)
        return Jp, Jr

    # ------------------------------------------------------------------ configuration space
    def _qid(self, dtype):
        if not hasattr(self, '_qid_cache'):
            self._qid_cache = {}
        if dtype not in self._qid_cache:
            self._qid_cache[dtype] = torch.tensor([1.0, 0, 0, 0], dtype=dtype, device=self.device)
        return self._qid_cache[dtype]

    def integrate(self, q, dq):
        """mj_integratePos with scale 1.  q [B,nq], dq [B,nv]."""
        q = q.clone()
        if self.free_body >= 0:
            f0 = self.free_dof0
            q[:, 0:3] = q[:, 0:3] + dq[:, f0:f0 + 3]
            w = dq[:, f0 + 3:f0 + 6]
            ang = w.norm(dim=-1, keepdim=True)
            axis = w / ang.clamp_min(1e-30)
            half = 0.5 * ang
            qr = torch.cat([torch.cos(half), axis * torch.sin(half)], -1)
            qr = torch.where(ang > 0, qr, self._qid(q.dtype))
            qn = quat_mul(q[:, 3:7], qr)
            q[:, 3:7] = qn / qn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q[:, self.scalar_qadr] = q[:, self.scalar_qadr] + dq[:, self.scalar_dadr]
        return q

    def difference(self, q_from, q_to):
        """mj_differentiatePos (dt=1): dq such that integrate(q_from, dq) = q_to."""
        dq = q_from.new_zeros(q_from.shape[0], self.nv)
        if self.free_body >= 0:
            f0 = self.free_dof0
            dq[:, f0:f0 + 3] = q_to[:, 0:3] - q_from[:, 0:3]
            qa = q_from[:, 3:7] / q_from[:, 3:7].norm(dim=-1, keepdim=True)
            qb = q_to[:, 3:7] / q_to[:, 3:7].norm(dim=-1, keepdim=True)
            qa_inv = qa * torch.tensor([1.0, -1, -1, -1], dtype=qa.dtype, device=qa.device)
            dqq = quat_mul(qa_inv, qb)
            dqq = torch.where(dqq[:, :1] < 0, -dqq, dqq)
            s = dqq[:, 1:].norm(dim=-1, keepdim=True)
            ang = 2 * torch.atan2(s, dqq[:, :1])
            dq[:, f0 + 3:f0 + 6] = torch.where(s > 1e-12, dqq[:, 1:] / s.clamp_min(1e-30) * ang, 2 * dqq[:, 1:])
        dq[:, self.scalar_dadr] = q_to[:, self.scalar_qadr] - q_from[:, self.scalar_qadr]
        return dq
