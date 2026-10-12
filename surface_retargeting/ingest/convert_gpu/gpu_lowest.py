"""Lean SMPL-X lowest-vertex (floor QA) on the GPU: the smplx.lbs math (same buffers, same float32 ops up to the
skinning), but only the Y row of the per-vertex skinning transform is formed for all vertices; the full xyz is
then evaluated for the argmin vertex only. ~= smplx_qa.lowest_reduce(smplx forward) per frame, many clips per
batch (per-frame betas)."""
from __future__ import annotations

import numpy as np
import smplx
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

MODEL = "/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz"


class LeanLowest:
    def __init__(self, device="cuda", chunk=4096):
        self.dev, self.chunk = device, chunk
        m = smplx.create(MODEL, model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True,
                         num_betas=10, batch_size=1).to(device)
        self.vt = m.v_template
        self.sd = torch.cat([m.shapedirs, m.expr_dirs], dim=-1)
        self.pd = m.posedirs
        self.Jr = m.J_regressor
        self.W = m.lbs_weights
        self.parents = m.parents
        self.I = torch.eye(3, device=device)

    @torch.no_grad()
    def lowest_pf(self, poses, trans, betas):
        """poses (N,22,3)|(N,165) axis-angle, trans (N,3), betas (N,10) -> (N,3) float32 lowest-Y vertex per frame."""
        P = torch.as_tensor(np.asarray(poses, np.float32).reshape(len(poses), -1), device=self.dev)
        Tr = torch.as_tensor(np.asarray(trans, np.float32), device=self.dev)
        Bt = torch.as_tensor(np.asarray(betas, np.float32)[:, :10], device=self.dev)
        out = []
        for s in range(0, len(P), self.chunk):
            p = P[s:s + self.chunk]
            n = p.shape[0]
            full = torch.zeros(n, 55 * 3, device=self.dev)
            full[:, :p.shape[1]] = p
            b20 = torch.cat([Bt[s:s + self.chunk], torch.zeros(n, 10, device=self.dev)], 1)
            v_shaped = self.vt + blend_shapes(b20, self.sd)
            J = vertices2joints(self.Jr, v_shaped)
            rot = batch_rodrigues(full.view(-1, 3)).view(n, -1, 3, 3)
            pf = (rot[:, 1:] - self.I).view(n, -1)
            vp = torch.matmul(pf, self.pd).view(n, -1, 3) + v_shaped
            _, A = batch_rigid_transform(rot, J, self.parents, dtype=torch.float32)
            Ay = A[:, :, 1, :]                                        # (n,55,4) Y rows
            Ty = torch.matmul(self.W, Ay)                             # (n,V,4)
            y = Ty[..., 0] * vp[..., 0] + Ty[..., 1] * vp[..., 1] + Ty[..., 2] * vp[..., 2] + Ty[..., 3]
            idx = y.argmin(1)
            ar = torch.arange(n, device=self.dev)
            Ti = torch.matmul(self.W[idx].unsqueeze(1), A.reshape(n, 55, 16)).view(n, 4, 4)
            v1 = vp[ar, idx]
            v = (Ti[:, :3, :3] @ v1.unsqueeze(-1)).squeeze(-1) + Ti[:, :3, 3]
            out.append(v + Tr[s:s + self.chunk])
        return torch.cat(out).cpu().numpy()
