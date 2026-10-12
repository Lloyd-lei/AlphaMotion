"""Production streaming: many clips packed into B long chains per robot, all robots sharing one SMPL-X pass.

* Clips longer than `max_piece` are split into pieces; a piece is solved from `warmup` frames before its kept range
  (UMR's sequential solve forgets its start state within ~100-200 frames: measured, see README) to `tail` frames after
  it (keeps LQR end anchors out of the kept range).
* Pieces are packed into B chains (longest-first, least-loaded chain).  A chain runs piece after piece; at a piece
  start the chain is reset to UMR's frame-0 state, which is precomputed for ALL pieces in one batched init solve
  (initial pose + pose_init_iters Gauss-Newton steps, anchors latched at the initial pose) so the lock-step CUDA graph
  never needs extra iterations.
* Per block of Tb steps: one SMPL-X forward for the block's frames (shared by all robots), then per robot the GPU
  source preprocessing into the persistent buffers its CUDA graph reads, then Tb graph replays.
"""
from __future__ import annotations

import heapq
import time
from types import SimpleNamespace

import numpy as np
import torch

from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu.solver import SeqState
from surface_retargeting.gpu import throttle


# ---------------------------------------------------------------------------------------------- scheduling
def build_schedule(clip_lengths, B, max_piece=1800, warmup=192, tail=16):
    """Returns dict of numpy arrays [B, Tch]: gframe (global frame, -1 idle), clip, cframe, piece, tl (local step),
    keep; plus the piece table."""
    offs = np.concatenate([[0], np.cumsum(clip_lengths)[:-1]]).astype(np.int64)
    pieces = []                                          # (clip, solve_start, solve_end, keep_start, keep_end)
    for c, T in enumerate(clip_lengths):
        T = int(T)
        if T <= max_piece:
            pieces.append((c, 0, T, 0, T))
            continue
        n = int(np.ceil(T / max_piece))
        Lp = int(np.ceil(T / n))
        for k in range(n):
            ks, ke = k * Lp, min(T, (k + 1) * Lp)
            pieces.append((c, max(0, ks - (warmup if k else 0)), min(T, ke + (tail if ke < T else 0)), ks, ke))
    order = sorted(range(len(pieces)), key=lambda i: -(pieces[i][2] - pieces[i][1]))
    heap = [(0, b) for b in range(B)]
    chains = [[] for _ in range(B)]
    for i in order:
        load, b = heapq.heappop(heap)
        chains[b].append(i)
        heapq.heappush(heap, (load + pieces[i][2] - pieces[i][1], b))
    Tch = max(sum(pieces[i][2] - pieces[i][1] for i in ch) for ch in chains) if pieces else 0
    sh = (B, Tch)
    S = dict(gframe=np.full(sh, -1, np.int64), clip=np.full(sh, -1, np.int64), cframe=np.full(sh, -1, np.int64),
             piece=np.full(sh, -1, np.int64), tl=np.full(sh, -1, np.int64), keep=np.zeros(sh, bool))
    for b, ch in enumerate(chains):
        t = 0
        for i in ch:
            c, s0, s1, k0, k1 = pieces[i]
            n = s1 - s0
            fr = np.arange(s0, s1)
            S['gframe'][b, t:t + n] = offs[c] + fr; S['clip'][b, t:t + n] = c; S['cframe'][b, t:t + n] = fr
            S['piece'][b, t:t + n] = i; S['tl'][b, t:t + n] = np.arange(n); S['keep'][b, t:t + n] = (fr >= k0) & (fr < k1)
            t += n
    S['pieces'] = np.asarray(pieces, np.int64).reshape(-1, 5)
    S['offs'] = offs
    S['useful_frames'] = int(np.sum(clip_lengths)); S['solved_frames'] = int((S['gframe'] >= 0).sum())
    return S


# ---------------------------------------------------------------------------------------------- per-robot runner
class ChainRunner:
    def __init__(self, solver, prep, B, Tb, device='cuda'):
        self.sv, self.prep, self.B, self.Tb = solver, prep, B, Tb
        b = solver.bank
        self.dev = torch.device(device)
        n = B * Tb
        S = b.S
        f = dict(device=self.dev)
        b.src_sel = torch.zeros(n, S, 3, dtype=torch.float32, **f)
        b.ntgt = torch.zeros(n, S, 3, dtype=torch.float32, **f)
        b.gd = torch.zeros(n, S, dtype=torch.float64, **f)
        b.gw = torch.zeros(n, S, dtype=torch.float64, **f)
        if b.has_sc:
            K = prep.K
            b.sc_pairs = torch.full((n, K, 2), -1, dtype=torch.long, **f)
            b.sc_d = torch.zeros(n, K, dtype=torch.float64, **f)
            b.sc_w = torch.zeros(n, K, dtype=torch.float64, **f)
            b.sc_dir = torch.zeros(n, K, 3, dtype=torch.float32, **f)
        b.off = torch.zeros(b.nprob, dtype=torch.long, **f)
        b.T = torch.full((b.nprob,), n, dtype=torch.long, **f)
        self.hreg = torch.as_tensor(L.part_to_region(b.problems[0]['part_ids']), device=self.dev)
        self.timing = dict(prep=0.0, solve=0.0, init=0.0)
        self.g = None

    # ---- batched frame-0 solves of every piece
    def init_pieces(self, V, J):
        """V/J of the pieces' first frames -> q0 [Np,nq], anchor targets/on at frame 0."""
        sv = self.sv; a = sv.a
        t0 = time.time()
        o = self.prep.frames(V, J)
        Np = V.shape[0]
        data = SimpleNamespace(src_sel=o['src_sel'], ntgt=o['ntgt'], gd=o['gd'], gw=o['gw'], has_ground=True,
                               has_sc='sc_pairs' in o, has_obj=False,
                               sc_pairs=o.get('sc_pairs'), sc_d=o.get('sc_d'), sc_w=o.get('sc_w'), sc_dir=o.get('sc_dir'))
        prob = torch.zeros(Np, dtype=torch.long, device=self.dev)
        st = SeqState(prob, prob, prob + 1, sv.bank.S, self.dev)
        fi = torch.arange(Np, device=self.dev)
        out_q, out_t, out_on = [], [], []
        CH = 8192
        for s in range(0, Np, CH):
            sl = slice(s, s + CH)
            sub = SimpleNamespace(**{k: (v[sl] if torch.is_tensor(v) else v) for k, v in vars(data).items()})
            stc = SeqState(prob[sl], prob[sl], prob[sl] + 1, sv.bank.S, self.dev)
            inp = sv.frame_inputs(stc, fi[sl] - s, sub)
            q = o['q_init'][sl].clone()
            sv.update_anchors(stc, q, inp)
            n = q.shape[0]
            f = torch.zeros(n, dtype=torch.bool, device=self.dev)
            live = torch.ones(n, dtype=torch.bool, device=self.dev)
            for k in range(int(a['pose_init_iters'])):
                dq, _ = sv.gn_step(stc, q, q, q, f, f, inp, live)
                q = sv.clamp(sv.kt.integrate(q, dq))
            out_q.append(q); out_t.append(stc.anchor_tgt.clone()); out_on.append(stc.anchor_on.clone())
        torch.cuda.synchronize()
        self.timing['init'] += time.time() - t0
        self.q0 = torch.cat(out_q); self.a_tgt0 = torch.cat(out_t); self.a_on0 = torch.cat(out_on)

    def fill_block(self, V, J):
        """V/J [B*Tb] in chain-major order (chain b, local step t -> b*Tb+t)."""
        t0 = time.time()
        o = self.prep.frames(V, J, chunk=4096)
        b = self.sv.bank
        b.src_sel.copy_(o['src_sel']); b.ntgt.copy_(o['ntgt']); b.gd.copy_(o['gd']); b.gw.copy_(o['gw'])
        if b.has_sc:
            b.sc_pairs.copy_(o['sc_pairs']); b.sc_d.copy_(o['sc_d']); b.sc_w.copy_(o['sc_w']); b.sc_dir.copy_(o['sc_dir'])
        # human region clearance (z, not yet floor-shifted) for the contact labels
        z = o['src'][..., 2]
        hcl = torch.full((z.shape[0], L.NR), float('inf'), device=z.device).scatter_reduce(
            1, self.hreg[None].expand(z.shape[0], -1), z, reduce='amin')
        self.timing['prep'] += time.time() - t0
        return hcl

    def steps(self, sched, t0, nb, out, live_mask):
        """Run steps [t0, t0+nb) of the chain schedule (GPU tensors in `sched`)."""
        sv = self.sv; B, Tb = self.B, self.Tb
        ts = time.time()
        if self.g is None:
            prob = torch.zeros(B, dtype=torch.long, device=self.dev)
            self.g, self.bf, self.st = sv._make_graph(B, prob, prob, prob + 1)
            self.base = torch.arange(B, device=self.dev) * Tb
        bf, st = self.bf, self.st
        for tl_ in range(nb):
            t = t0 + tl_
            tl = sched['tl'][:, t]; pc = sched['piece'][:, t].clamp_min(0)
            valid = tl >= 0
            reset = valid & (tl == 0)
            bf['fi'].copy_(self.base + tl_)
            bf['qp2'].copy_(bf['qp1']); bf['qp1'].copy_(bf['q'])
            bf['q'].copy_(torch.where(reset[:, None], self.q0[pc], bf['q']))
            st.anchor_tgt.copy_(torch.where(reset[:, None, None], self.a_tgt0[pc], st.anchor_tgt))
            st.anchor_on.copy_(torch.where(reset[:, None], self.a_on0[pc], st.anchor_on))
            bf['hp'].copy_(tl >= 1); bf['hp2'].copy_(tl >= 2)
            bf['live'].copy_(valid & ~reset); bf['first'].fill_(True)
            self.g.replay()
            out[:, t] = bf['q']
            if tl_ % 4 == 3:
                throttle.tick()
        torch.cuda.synchronize()
        self.timing['solve'] += time.time() - ts


def smplx_frames(smplx, pose_all, trans_all, gframe):
    """SMPL-X forward for global frame indices gframe [N] (idle = -1 -> frame 0)."""
    g = gframe.clamp_min(0)
    return smplx(pose_all[g], trans_all[g])
