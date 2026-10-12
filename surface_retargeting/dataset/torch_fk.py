"""GPU-side helpers for a loaded batch (trainer process, after the DataLoader): robot body poses, dense surface points,
per-point ground clearance / contact (what the compact tiers drop), AM2-style pose9, Z-up -> Y-up.

    from surface_retargeting.dataset.torch_fk import BatchFK
    bfk = BatchFK('/home/arenalabs/umr_dataset/data', device='cuda')
    out = bfk.bodies(batch['slot1'])            # body_pos [B,W,NB,3], body_rot6 [B,W,NB,6], body_mask [B,NB]
    pts = bfk.surface(batch['slot1'])           # points [B,W,4096,3], point_region [B,4096]
    pc  = bfk.point_ground_contact(batch['slot1'])   # [B,W,4096] bool: z <= 1 mm  AND  region labelled in contact

Body order is MuJoCo's (index 0 = world, dropped here: body k of the output = MuJoCo body k+1); parents / names per
robot via `bfk.skeleton(robot)`.  All robots of a batch slot are handled together (grouped by body index).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .kinematics import TorchFK

# Z-up (UMR / MuJoCo) <-> Y-up (AM2 caches): p_zup = (p_yup.z, p_yup.x, p_yup.y)  =>  p_yup = (p_zup.y, p_zup.z, p_zup.x)
ZUP_TO_YUP = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], np.float32)


class BatchFK:
    def __init__(self, root, device='cuda', dtype=None):
        import torch
        self.root = Path(root)
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self.cat = json.load(open(self.root / 'catalog_meta.json'))
        self.bodies_names = self.cat['bodies']
        self._fk = {}

    def fk(self, b: int) -> TorchFK:
        f = self._fk.get(b)
        if f is None:
            f = TorchFK.load(self.root / 'robots' / self.bodies_names[b] / 'kin.npz', device=self.device, dtype=self.dtype)
            self._fk[b] = f
        return f

    def skeleton(self, robot: str) -> dict:
        b = self.bodies_names.index(robot)
        f = self.fk(b)
        return dict(parents=[p - 1 for p in f.parent[1:]], nbody=f.nbody - 1,
                    region=np.asarray(f.k['body_region'][1:]).tolist() if 'body_region' in f.k else None)

    def _groups(self, slot):
        body = slot['body'].cpu().numpy()
        for b in np.unique(body):
            if b == 0:
                continue
            yield int(b), np.flatnonzero(body == b)

    def bodies(self, slot, y_up=False):
        """Body world poses (robot frame, floor z = 0). Returns body_pos, body_rot6 (first two rotation columns,
        AM2 convention), body_mask; human rows stay zero / masked."""
        import torch
        q = slot['qpos'].to(self.device, self.dtype)
        B, W = q.shape[:2]
        NB = max(self.fk(b).nbody - 1 for b, _ in self._groups(slot))
        pos = torch.zeros(B, W, NB, 3, device=self.device, dtype=self.dtype)
        rot = torch.zeros(B, W, NB, 3, 3, device=self.device, dtype=self.dtype)
        mask = torch.zeros(B, NB, dtype=torch.bool, device=self.device)
        for b, ii in self._groups(slot):
            f = self.fk(b)
            ii_t = torch.as_tensor(ii, device=self.device)
            X, R = f.bodies(q[ii_t, :, :f.qpos0.shape[0]])
            n = f.nbody - 1
            pos[ii_t, :, :n] = X[..., 1:, :]
            rot[ii_t, :, :n] = R[..., 1:, :, :]
            mask[ii_t, :n] = True
        if y_up:
            P = torch.as_tensor(ZUP_TO_YUP, device=self.device, dtype=self.dtype)
            pos = pos @ P.T
            rot = P @ rot @ P.T
        return dict(body_pos=pos, body_rot6=torch.cat([rot[..., :, 0], rot[..., :, 1]], -1), body_rotmat=rot,
                    body_mask=mask)

    def surface(self, slot):
        """Dense 4096-point surface of every robot row (the factory's ground / contact gauge points)."""
        import torch
        q = slot['qpos'].to(self.device, self.dtype)
        B, W = q.shape[:2]
        P = reg = None
        for b, ii in self._groups(slot):
            f = self.fk(b)
            ii_t = torch.as_tensor(ii, device=self.device)
            X = f.surface(q[ii_t, :, :f.qpos0.shape[0]])
            if P is None:
                P = torch.zeros(B, W, X.shape[-2], 3, device=self.device, dtype=self.dtype)
                reg = torch.full((B, X.shape[-2]), -1, dtype=torch.long, device=self.device)
            P[ii_t] = X
            reg[ii_t] = f.pts_region
        return dict(points=P, point_region=reg)

    def point_ground_contact(self, slot, tol=1e-3):
        """Per-point ground contact recomputed from qpos: point height <= tol (1 mm) AND its region labelled in
        contact.  The factory's stored per-point bit used a 1 cm / 2 cm hysteresis at the native rate; this is the
        strict [.., 1 mm] version; the region labels keep the exact stored semantics."""
        import torch
        s = self.surface(slot)
        z = s['points'][..., 2]
        reg = s['point_region'].clamp(min=0)
        con = slot['contact'].to(self.device)                                  # [B,W,13]
        rc = torch.gather(con, 2, reg[:, None, :].expand(-1, con.shape[1], -1))
        return (z <= tol) & rc
