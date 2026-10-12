"""Warp kernels for UMR's source self-contact maps (identical selection rules, exact distances, no [F x 83k] tensor).

  clearance mode (source_clearance_body_pair_topk, top-1): per frame and non-adjacent segment pair, the closest slot
      pair (first in UMR's pair order on ties)                    -> thread per (frame, group) looping its pairs
  threshold mode (threshold_global_topk): pairs with d <= thr, the max_pairs closest, ascending
      -> per-frame histogram of in-range distances (atomic), cutoff bin, compaction of candidates, exact sort
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

wp.init()
NB = 512


@wp.kernel
def group_min_kernel(P: wp.array2d(dtype=wp.vec3), ui: wp.array(dtype=wp.int32), uj: wp.array(dtype=wp.int32),
                     gptr: wp.array(dtype=wp.int32), thr: float, gmin: wp.array2d(dtype=float),
                     gam: wp.array2d(dtype=wp.int32), hist: wp.array2d(dtype=wp.int32)):
    f, g = wp.tid()
    best = float(1.0e30)
    bi = int(-1)
    for k in range(gptr[g], gptr[g + 1]):
        d = wp.length(P[f, ui[k]] - P[f, uj[k]])
        if d < best:
            best = d
            bi = k
        if d <= thr:
            b = wp.min(int(d / thr * float(NB)), NB - 1)
            wp.atomic_add(hist, f, b, 1)
    gmin[f, g] = best
    gam[f, g] = bi


@wp.kernel
def compact_kernel(P: wp.array2d(dtype=wp.vec3), ui: wp.array(dtype=wp.int32), uj: wp.array(dtype=wp.int32),
                   gptr: wp.array(dtype=wp.int32), cut: wp.array(dtype=float), cap: int,
                   cnt: wp.array(dtype=wp.int32), od: wp.array2d(dtype=float), op: wp.array2d(dtype=wp.int32)):
    f, g = wp.tid()
    c = cut[f]
    for k in range(gptr[g], gptr[g + 1]):
        d = wp.length(P[f, ui[k]] - P[f, uj[k]])
        if d <= c:
            s = wp.atomic_add(cnt, f, 1)
            if s < cap:
                od[f, s] = d
                op[f, s] = k


class SelfContactMaps:
    def __init__(self, ui, uj, gid, thr, max_pairs, device='cuda', cap=1024):
        """ui/uj [Np] positions in the selected-slot list (UMR pair order), gid [Np] segment-pair group."""
        gid = np.asarray(gid); order = np.argsort(gid, kind='stable')
        self.order = torch.as_tensor(order, device=device)                 # sorted position -> original pair index
        self.ui = wp.from_torch(torch.as_tensor(np.asarray(ui)[order], dtype=torch.int32, device=device))
        self.uj = wp.from_torch(torch.as_tensor(np.asarray(uj)[order], dtype=torch.int32, device=device))
        G = int(gid.max()) + 1
        cnt = np.bincount(gid, minlength=G)
        self.gptr = wp.from_torch(torch.as_tensor(np.concatenate([[0], np.cumsum(cnt)]), dtype=torch.int32, device=device))
        self.G, self.thr, self.maxp, self.cap = G, float(thr), int(max_pairs), int(cap)
        self.dev = torch.device(device)

    def __call__(self, P):
        """P [F,S,3] float32 -> dict: g_min [F,G], g_pair [F,G] (original pair index), t_pair [F,maxp] (original
        pair index, -1 pad), t_d [F,maxp], overflow frames."""
        F = P.shape[0]
        dev = self.dev
        st = wp.stream_from_torch(torch.cuda.current_stream()); wd = 'cuda:%d' % (dev.index or 0)
        Pw = wp.from_torch(P.float().contiguous(), dtype=wp.vec3)
        gmin = torch.empty(F, self.G, dtype=torch.float32, device=dev)
        gam = torch.empty(F, self.G, dtype=torch.int32, device=dev)
        hist = torch.zeros(F, NB, dtype=torch.int32, device=dev)
        wp.launch(group_min_kernel, dim=(F, self.G), inputs=[Pw, self.ui, self.uj, self.gptr, self.thr, wp.from_torch(gmin),
                  wp.from_torch(gam), wp.from_torch(hist)], device=wd, stream=st)
        cum = torch.cumsum(hist, 1)
        tot = cum[:, -1]
        hit = cum >= self.maxp
        b = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(tot, NB - 1).long())
        cut = torch.where(tot > self.maxp, (b + 1).float() / NB * self.thr, torch.full((F,), self.thr, device=dev))
        cnt = torch.zeros(F, dtype=torch.int32, device=dev)
        od = torch.full((F, self.cap), float('inf'), dtype=torch.float32, device=dev)
        op = torch.full((F, self.cap), 2 ** 30, dtype=torch.int32, device=dev)
        wp.launch(compact_kernel, dim=(F, self.G), inputs=[Pw, self.ui, self.uj, self.gptr, wp.from_torch(cut), self.cap,
                  wp.from_torch(cnt), wp.from_torch(od), wp.from_torch(op)], device=wd, stream=st)
        orig = torch.where(op < 2 ** 30, self.order[op.clamp(0, self.order.numel() - 1).long()], torch.full_like(op, 2 ** 30).long())
        # exact order: by distance, ties by original pair index
        o2 = torch.argsort(orig, dim=1, stable=True)
        od2 = torch.gather(od, 1, o2); orig2 = torch.gather(orig, 1, o2)
        o3 = torch.argsort(od2, dim=1, stable=True)
        tsel = torch.gather(orig2, 1, o3)[:, :self.maxp]; tdist = torch.gather(od2, 1, o3)[:, :self.maxp]
        ok = torch.isfinite(tdist)
        return dict(g_min=gmin, g_pair=self.order[gam.long()], t_pair=torch.where(ok, tsel, torch.full_like(tsel, -1)),
                    t_d=torch.where(ok, tdist, torch.zeros_like(tdist)), overflow=int((cnt > self.cap).sum()))
