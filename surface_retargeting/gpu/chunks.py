"""Chunked, streaming production solve: thousands of (clip-chunk) chains in lock-step on one GPU.

Each clip is cut into chunks of Lc frames that overlap by Ov frames.  A chunk starts exactly like a UMR sequence
(UMR's initial pose for its first frame + pose_init_iters Gauss-Newton steps) and then streams with UMR's
sequential warm start.  Per time block of Tb steps the SMPL-X forward + all per-frame solver inputs are computed on
the GPU for the whole batch (source_gpu.SourcePrep) into persistent buffers that the captured CUDA graph reads.
Stitching: chunk k is used up to the middle of its overlap with chunk k+1 (the warm-up transient of chunk k+1 and the
LQR end anchors of both chunks lie outside the used part).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from surface_retargeting.gpu.solver import SeqState, LQRGPU
from surface_retargeting.gpu.source_gpu import SMPLXForward


def make_chunks(T, Lc=512, Ov=32):
    """Chunk starts covering [0,T) with full-length chunks (the last one is shifted back to end at T)."""
    if T <= Lc:
        return [(0, T)]
    st = list(range(0, T - Lc + 1, Lc - Ov))
    if st[-1] + Lc < T:
        st.append(T - Lc)
    return [(s, Lc) for s in st]


def stitch(chunk_list, Qc, T):
    """chunk_list [(start, len)], Qc [n, Lc, nq] -> [T, nq]; switch at the middle of each overlap."""
    nq = Qc.shape[-1]
    out = torch.zeros(T, nq, dtype=Qc.dtype, device=Qc.device)
    for k, (s, L) in enumerate(chunk_list):
        a = s if k == 0 else (s + (chunk_list[k - 1][0] + chunk_list[k - 1][1])) // 2
        b = s + L if k == len(chunk_list) - 1 else (chunk_list[k + 1][0] + s + L) // 2
        out[a:b] = Qc[k, a - s:b - s]
    return out


class ChunkRunner:
    """UMR stage for B chains (chunks of possibly different clips of ONE (subject, robot) static prep)."""

    def __init__(self, solver, prep, B, Lc, Tb=16, smplx_batch=4096):
        self.sv, self.prep = solver, prep
        self.B, self.Lc, self.Tb = B, Lc, Tb
        bank = solver.bank
        dev = bank.device
        self.dev = dev
        n = B * Tb
        S = bank.S
        f = dict(device=dev)
        # persistent per-frame input buffers (the captured graph reads these storages)
        bank.src_sel = torch.zeros(n, S, 3, dtype=torch.float32, **f)
        bank.ntgt = torch.zeros(n, S, 3, dtype=torch.float32, **f)
        bank.gd = torch.zeros(n, S, dtype=torch.float64, **f)
        bank.gw = torch.zeros(n, S, dtype=torch.float64, **f)
        if bank.has_sc:
            K = prep.K
            bank.sc_pairs = torch.full((n, K, 2), -1, dtype=torch.long, **f)
            bank.sc_d = torch.zeros(n, K, dtype=torch.float64, **f)
            bank.sc_w = torch.zeros(n, K, dtype=torch.float64, **f)
            bank.sc_dir = torch.zeros(n, K, 3, dtype=torch.float32, **f)
        bank.q_init = torch.zeros(n, bank.q_init.shape[1], dtype=torch.float64, **f)
        bank.off = torch.zeros(bank.nprob, dtype=torch.long, **f)
        bank.T = torch.full((bank.nprob,), n, dtype=torch.long, **f)
        self.smplx = None
        self.smplx_batch = smplx_batch
        self.timing = {}

    def fill_block(self, pose, trans):
        """pose [B*Tb, P], trans [B*Tb, 3] (frame-major per chain: chain b, local step t at index b*Tb+t)."""
        t0 = time.time()
        V, J = self.smplx(pose, trans)
        torch.cuda.synchronize(); t1 = time.time()
        o = self.prep.frames(V, J, chunk=4096)
        b = self.sv.bank
        b.src_sel.copy_(o['src_sel']); b.ntgt.copy_(o['ntgt']); b.gd.copy_(o['gd']); b.gw.copy_(o['gw'])
        if b.has_sc:
            b.sc_pairs.copy_(o['sc_pairs']); b.sc_d.copy_(o['sc_d']); b.sc_w.copy_(o['sc_w']); b.sc_dir.copy_(o['sc_dir'])
        b.q_init.copy_(o['q_init'])
        torch.cuda.synchronize(); t2 = time.time()
        self.timing['smplx'] = self.timing.get('smplx', 0) + t1 - t0
        self.timing['prep'] = self.timing.get('prep', 0) + t2 - t1
        return o

    def run(self, clip_pose, clip_trans, chain_clip, chain_start, gender='neutral', betas=None, log=False):
        """clip_pose: list of [T_c, P] tensors (GPU), chain_clip/chain_start [B] (long, GPU).  Returns Q [B, Lc, nq]."""
        sv = self.sv; b = sv.bank; a = sv.a
        B, Lc, Tb = self.B, self.Lc, self.Tb
        dev = self.dev
        if self.smplx is None:
            self.smplx = SMPLXForward(gender, betas, batch=self.smplx_batch, device=dev)
        P = clip_pose[0].shape[1]
        # all clips concatenated (on GPU) + chain frame indices
        lens = torch.as_tensor([len(p) for p in clip_pose], device=dev)
        offs = torch.cumsum(lens, 0) - lens
        allpose = torch.cat(clip_pose); alltr = torch.cat(clip_trans)
        prob = torch.zeros(B, dtype=torch.long, device=dev)
        g, bf, st = None, None, None
        out = torch.zeros(B, Lc, b.q_init.shape[1], dtype=torch.float64, device=dev)
        pose_init = int(a['pose_init_iters']); it_n = int(a['iters'])
        tloc = torch.arange(Tb, device=dev)
        t_solve = 0.0
        for t0 in range(0, Lc, Tb):
            nb = min(Tb, Lc - t0)
            fr = offs[chain_clip][:, None] + chain_start[:, None] + t0 + tloc[None]
            fr = torch.minimum(fr, (offs + lens - 1)[chain_clip][:, None])
            self.fill_block(allpose[fr.reshape(-1)], alltr[fr.reshape(-1)])
            ts = time.time()
            if g is None:
                g, bf, st = sv._make_graph(B, prob, torch.zeros(B, dtype=torch.long, device=dev), torch.full((B,), Lc, device=dev))
                st.anchor_on.zero_(); st.anchor_tgt.zero_()
            base = torch.arange(B, device=dev) * Tb
            for tl in range(nb):
                t = t0 + tl
                bf['fi'].copy_(base + tl)
                if t == 0:
                    bf['q'].copy_(b.q_init[bf['fi']]); bf['qp1'].copy_(bf['q']); bf['qp2'].copy_(bf['q'])
                else:
                    bf['qp2'].copy_(bf['qp1']); bf['qp1'].copy_(bf['q'])
                bf['hp'].fill_(t >= 1); bf['hp2'].fill_(t >= 2)
                n = pose_init if (t == 0 and pose_init > 0) else it_n
                for k in range(n):
                    bf['first'].fill_(k == 0); bf['live'].fill_(True)
                    g.replay()
                out[:, t] = bf['q']
            torch.cuda.synchronize(); t_solve += time.time() - ts
        self.timing['solve'] = self.timing.get('solve', 0) + t_solve
        return out
