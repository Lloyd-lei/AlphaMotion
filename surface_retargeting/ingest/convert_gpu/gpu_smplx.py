"""SMPL-X (neutral, flat_hand_mean, no PCA, 10 betas - same as smplx_qa / UMR) forward on the GPU, chunked."""
from __future__ import annotations

import numpy as np
import smplx
import torch

MODEL = "/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz"


class GpuSmplx:
    def __init__(self, device="cuda", chunk=4096):
        self.dev, self.chunk = device, chunk
        self.m = smplx.create(MODEL, model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True,
                              num_betas=10, batch_size=1).to(device)
        self.faces = self.m.faces

    @torch.no_grad()
    def _fwd(self, p, t, b):
        n = p.shape[0]
        z = torch.zeros(n, 10, device=self.dev)
        return self.m(betas=b.expand(n, -1), global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69],
                      leye_pose=p[:, 69:72], reye_pose=p[:, 72:75], left_hand_pose=p[:, 75:120],
                      right_hand_pose=p[:, 120:165], transl=t, expression=z)

    @torch.no_grad()
    def lowest(self, poses, trans, betas):
        """smplx_qa.lowest_reduce over all frames: (T,3) float32 lowest-(Y) vertex per frame."""
        P = torch.as_tensor(np.asarray(poses, np.float32).reshape(len(poses), -1), device=self.dev)
        Tr = torch.as_tensor(np.asarray(trans, np.float32), device=self.dev)
        b = torch.as_tensor(np.asarray(betas, np.float32).reshape(1, -1)[:, :10], device=self.dev)
        out = []
        for s in range(0, len(P), self.chunk):
            v = self._fwd(P[s:s + self.chunk], Tr[s:s + self.chunk], b).vertices
            i = v[:, :, 1].argmin(1)
            out.append(v[torch.arange(len(v), device=self.dev), i])
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def lowest_pf(self, poses, trans, betas):
        """lowest() with per-frame betas (N,10): many clips in one batched pass. (N,3) float32."""
        P = torch.as_tensor(np.asarray(poses, np.float32).reshape(len(poses), -1), device=self.dev)
        if P.shape[1] == 66:
            P = torch.cat([P, torch.zeros(len(P), 99, device=self.dev)], 1)
        Tr = torch.as_tensor(np.asarray(trans, np.float32), device=self.dev)
        Bt = torch.as_tensor(np.asarray(betas, np.float32)[:, :10], device=self.dev)
        out = []
        for s in range(0, len(P), self.chunk):
            p = P[s:s + self.chunk]
            n = p.shape[0]
            o = self.m(betas=Bt[s:s + self.chunk], global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69],
                       leye_pose=p[:, 69:72], reye_pose=p[:, 72:75], left_hand_pose=p[:, 75:120],
                       right_hand_pose=p[:, 120:165], transl=Tr[s:s + self.chunk], expression=torch.zeros(n, 10, device=self.dev))
            v = o.vertices
            i = v[:, :, 1].argmin(1)
            out.append(v[torch.arange(n, device=self.dev), i])
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def compare(self, pa, ta, ba, pb, tb, bb):
        """per-frame vertex / joint errors (m) between two SMPL-X sequences: dict of (T,) arrays."""
        res = {k: [] for k in ("v_max", "v_mean", "v_median", "j_max", "j_mean", "j55_max")}
        A = [torch.as_tensor(np.asarray(x, np.float32), device=self.dev) for x in (pa, ta, np.reshape(ba, (1, -1))[:, :10])]
        B = [torch.as_tensor(np.asarray(x, np.float32), device=self.dev) for x in (pb, tb, np.reshape(bb, (1, -1))[:, :10])]
        T = len(A[0])
        for s in range(0, T, self.chunk):
            oa = self._fwd(A[0][s:s + self.chunk].reshape(-1, 165), A[1][s:s + self.chunk], A[2])
            ob = self._fwd(B[0][s:s + self.chunk].reshape(-1, 165), B[1][s:s + self.chunk], B[2])
            dv = torch.linalg.vector_norm(oa.vertices - ob.vertices, dim=-1)
            dj = torch.linalg.vector_norm(oa.joints[:, :22] - ob.joints[:, :22], dim=-1)
            dj55 = torch.linalg.vector_norm(oa.joints[:, :55] - ob.joints[:, :55], dim=-1)
            res["v_max"].append(dv.max(1).values); res["v_mean"].append(dv.mean(1)); res["v_median"].append(dv.median(1).values)
            res["j_max"].append(dj.max(1).values); res["j_mean"].append(dj.mean(1)); res["j55_max"].append(dj55.max(1).values)
        return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}
