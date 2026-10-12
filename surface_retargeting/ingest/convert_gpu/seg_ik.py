"""Time-segmented ("parallel-in-time") lane packing for the batched Mink IK (gpu_ik.FastMinkIK).

The CPU solver is a per-clip chain: frame t is warm-started from the float32 solution of frame t-1. One lane per
clip (v1) leaves the GPU idle unless there are >= 1000 clips of equal length. Here all frames of all clips of a
wave are concatenated into one stream of N frames, cut into B equal pieces of P = ceil(N/B) frames (B = lanes),
and every piece that starts in the middle of a clip is preceded by a warm-up of W frames of the same clip
(warm-up starts like frame 0 of a clip: seed = copied reference pose, root = target pelvis). Warm-up outputs are
discarded; the piece's own frames are then computed from a chain that has forgotten its artificial start (the IK
is strongly contractive in the seed: LM-damped Gauss-Newton to a unique local minimum every frame). Pieces that
start within W frames of a clip start are warmed up from the real frame 0, i.e. exactly the CPU chain.

Per-lane constants of v1 (rest joints, MJCF body offsets) become per-(lane, slot) inputs, so one lane can run
several (short) clips back to back.
"""
from __future__ import annotations

import numpy as np
import torch

from surface_retargeting.ingest.convert_gpu import gpu_ik
from surface_retargeting.ingest.convert_gpu.gpu_ik import NJ


def build_schedule(Fs, B, W, P_min=64):
    """Fs: frames per clip. Returns dict of numpy arrays (B, S): gidx (global frame, -1 = idle), first, keep;
    plus N, P, S, clip_of (N,), goff (C,)."""
    Fs = np.asarray(Fs, np.int64)
    goff = np.concatenate([[0], np.cumsum(Fs)[:-1]])
    N = int(Fs.sum())
    P = max(int(np.ceil(N / B)), min(P_min, N))
    clip_of = np.repeat(np.arange(len(Fs)), Fs)
    fr_in_clip = np.arange(N) - goff[clip_of]
    lanes = []
    for b in range(B):
        g0, g1 = b * P, min((b + 1) * P, N)
        if g0 >= N:
            lanes.append(None); continue
        o = fr_in_clip[g0]
        w = int(min(W, o))
        lanes.append((g0 - w, g1, w))
    S = max(l[1] - l[0] for l in lanes if l is not None)
    gidx = np.full((B, S), -1, np.int64)
    first = np.zeros((B, S), bool)
    keep = np.zeros((B, S), bool)
    for b, l in enumerate(lanes):
        if l is None:
            continue
        a, e, w = l
        n = e - a
        g = np.arange(a, e)
        gidx[b, :n] = g
        first[b, :n] = fr_in_clip[g] == 0
        first[b, 0] = True
        keep[b, w:n] = True
    return dict(gidx=gidx, first=first, keep=keep, N=N, P=P, S=S, clip_of=clip_of, goff=goff,
                warm_frames=int(sum(l[2] for l in lanes if l is not None)))


class SegRunner:
    """K consecutive frame solves for all B lanes captured in one CUDA graph; per-(lane, slot) targets,
    copied rotations, rest joints and MJCF offsets; float32 warm-start chaining exactly as the CPU path."""

    def __init__(self, ik: gpu_ik.FastMinkIK, K=8, use_graph=True):
        self.ik, self.K = ik, K
        B, dev = ik.B, ik.dev
        self.B, self.dev = B, dev
        f64 = dict(dtype=torch.float64, device=dev)
        self.tgt = torch.zeros(B, K, NJ, 3, **f64)
        self.ref = torch.zeros(B, K, NJ, 3, **f64)
        self.rest = torch.zeros(B, K, NJ, 3, **f64)
        self.off = torch.zeros(B, K, NJ, 3, **f64)
        self.first = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.active = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.seed_rv = torch.zeros(B, NJ, 3, **f64)
        self.seed_t = torch.zeros(B, 3, **f64)
        self.o_pose = torch.zeros(B, K, NJ, 3, dtype=torch.float32, device=dev)
        self.o_trans = torch.zeros(B, K, 3, dtype=torch.float32, device=dev)
        self.o_joints = torch.zeros(B, K, NJ, 3, dtype=torch.float32, device=dev)
        self.o_loss = torch.zeros(B, K, dtype=torch.float32, device=dev)
        self.o_nit = torch.zeros(B, K, dtype=torch.int32, device=dev)
        self.lane_ids = torch.arange(B, dtype=torch.int64, device=dev)[:, None]
        # benign warm-up inputs for capture (identity pose, T-pose targets)
        rj = ik.rest.clone()
        self.rest.copy_(rj[:, None]); self.off.copy_(ik.off[:, None]); self.tgt.copy_(rj[:, None])
        self.first.fill_(True); self.active.fill_(True)
        self.graph = None
        if use_graph and str(dev).startswith("cuda"):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._body()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._body()
        self.seed_rv.zero_(); self.seed_t.zero_()

    def _body(self):
        ik = self.ik
        B = self.B
        for k in range(self.K):
            tgt, ref, rest, off = self.tgt[:, k], self.ref[:, k], self.rest[:, k], self.off[:, k]
            first, act = self.first[:, k], self.active[:, k]
            rest0 = rest[:, 0]
            s_rv = torch.where(first[:, None, None], ref, self.seed_rv)
            root = torch.where(first[:, None], tgt[:, 0], self.seed_t + rest0)
            q, qref, qt, qt_inv, tt_inv = ik._pro(tgt, ref, s_rv, rest)
            prev = torch.full((B,), float("inf"), dtype=tgt.dtype, device=tgt.device)
            done = ~act
            nit = torch.zeros(B, dtype=torch.int32, device=tgt.device)
            for _ in range(ik.iters):
                root, q, prev, done, nit = ik._it(root, q, prev, done, nit, tgt, qt, qt_inv, tt_inv, qref, off)
            pose, xp, me = ik._epi(root, q, tgt, off)
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

    @torch.no_grad()
    def _pass(self, G, Cidx, Fst, Act, D, tgt_all, ref_all, rest_tab, off_tab, outs, seed=None, rec=None, merge=None,
              writer=None):
        """One lock-step pass over K-slot groups. G/Cidx/Fst/Act/D: (B,S) device tensors (global frame, clip id,
        first flag, active, output row (N = dummy)). seed: (lanes mask, pose (B,22,3), trans (B,3)) preset before
        slot 0. rec: (B,S) row map (B = dummy) into a (B+1,...) warm-state store. merge: (B,) bool lanes whose
        pass stops once their new output equals the stored output (deterministic chain => identical afterwards).
        Returns number of K-groups run."""
        B, K = self.B, self.K
        S = G.shape[1]
        self.seed_rv.zero_(); self.seed_t.zero_()
        if seed is not None:
            m, sp, st = seed
            self.seed_rv.copy_(torch.where(m[:, None, None], sp.double(), self.seed_rv))
            self.seed_t.copy_(torch.where(m[:, None], st.double(), self.seed_t))
        pending = merge.clone() if merge is not None else None
        groups = 0
        for s in range(0, S, K):
            g = G[:, s:s + K]
            c = Cidx[:, s:s + K]
            self.tgt.copy_(tgt_all[g]); self.ref.copy_(ref_all[g])
            self.rest.copy_(rest_tab[c]); self.off.copy_(off_tab[c])
            self.first.copy_(Fst[:, s:s + K]); self.active.copy_(Act[:, s:s + K])
            self.run()
            groups += 1
            d = D[:, s:s + K]
            if pending is not None:
                same = (self.o_pose == outs["pose"][d]).flatten(2).all(-1) & (self.o_trans == outs["trans"][d]).all(-1)
                same = same & Act[:, s:s + K] & (d < outs["pose"].shape[0] - 1)
                pending = pending & ~same.any(1)
            dd = d.reshape(-1)
            outs["pose"].index_copy_(0, dd, self.o_pose.reshape(-1, NJ, 3))
            outs["trans"].index_copy_(0, dd, self.o_trans.reshape(-1, 3))
            outs["joints"].index_copy_(0, dd, self.o_joints.reshape(-1, NJ, 3))
            outs["loss"].index_copy_(0, dd, self.o_loss.reshape(-1))
            outs["nit"].index_copy_(0, dd, self.o_nit.reshape(-1))
            if writer is not None:
                writer.index_copy_(0, dd, self.lane_ids.expand(B, K).reshape(-1))
            if rec is not None:
                rmap, store = rec
                r = rmap[:, s:s + K].reshape(-1)
                store["pose"].index_copy_(0, r, self.o_pose.reshape(-1, NJ, 3))
                store["trans"].index_copy_(0, r, self.o_trans.reshape(-1, 3))
            if pending is not None and (s // K) % 4 == 3 and not bool(pending.any()):
                break
        return groups

    @torch.no_grad()
    def solve_wave(self, tgt_all, ref_all, rest_tab, off_tab, Fs, W=32, exact=True, max_passes=6):
        """tgt_all/ref_all (N,22,3) float32 on device (clip-concatenated), rest_tab/off_tab (C,22,3) float64 on
        device. Pass 1: all pieces with warm-up W. exact=True: then re-run every piece whose warm-up end state
        (float32 pose+trans at frame g0-1) differs from the preceding piece's output at g0-1, seeded with that
        exact state, until all piece boundaries agree bitwise (= the unsegmented sequential chain). Returns
        per-frame outputs (N,...) on device + stats."""
        dev, B, K = self.dev, self.B, self.K
        sch = build_schedule(Fs, B, W)
        N, S = sch["N"], sch["S"]
        S_pad = int(np.ceil(S / K) * K)
        gidx = np.full((B, S_pad), -1, np.int64); gidx[:, :S] = sch["gidx"]
        first = np.zeros((B, S_pad), bool); first[:, :S] = sch["first"]
        keep = np.zeros((B, S_pad), bool); keep[:, :S] = sch["keep"]
        active = gidx >= 0
        gsafe = np.where(active, gidx, 0)
        nact = active.sum(1)
        for b in range(B):   # idle slots repeat the lane's last real frame (finite inputs; outputs dropped)
            if 0 < nact[b] < S_pad:
                gsafe[b, nact[b]:] = gidx[b, nact[b] - 1]
        dst = np.where(keep, gidx, N)
        clip_of_np = sch["clip_of"]
        clip_of = torch.as_tensor(clip_of_np, device=dev)
        fr_in_clip = np.arange(N) - sch["goff"][clip_of_np]
        # piece boundaries: g0 = first kept frame, w = warm-up, check = warm-up started mid-clip (artificial seed)
        g0 = np.array([gidx[b, keep[b]].min() if keep[b].any() else -1 for b in range(B)])
        wl = np.array([int(np.argmax(keep[b])) if keep[b].any() else 0 for b in range(B)])
        check = (g0 > 0) & (g0 >= 0)
        check &= np.array([fr_in_clip[g0[b]] > W if g0[b] > 0 else False for b in range(B)])
        rmap = np.full((B, S_pad), B, np.int64)
        for b in np.flatnonzero(check):
            rmap[b, wl[b] - 1] = b
        f32 = dict(dtype=torch.float32, device=dev)
        outs = dict(pose=torch.zeros(N + 1, NJ, 3, **f32), trans=torch.zeros(N + 1, 3, **f32),
                    joints=torch.zeros(N + 1, NJ, 3, **f32), loss=torch.zeros(N + 1, **f32),
                    nit=torch.zeros(N + 1, dtype=torch.int32, device=dev))
        used = dict(pose=torch.zeros(B + 1, NJ, 3, **f32), trans=torch.zeros(B + 1, 3, **f32))
        G = torch.as_tensor(gsafe, device=dev)
        writer = torch.full((N + 1,), -1, dtype=torch.int64, device=dev)
        groups = self._pass(G, clip_of[G], torch.as_tensor(first, device=dev), torch.as_tensor(active, device=dev),
                            torch.as_tensor(dst, device=dev), tgt_all, ref_all, rest_tab, off_tab, outs,
                            rec=(torch.as_tensor(rmap, device=dev), used), writer=writer)
        passes = [dict(lanes=int(B), groups=groups)]
        chk = torch.as_tensor(check, device=dev)
        g0m1 = torch.as_tensor(np.where(check, g0 - 1, N), device=dev)
        g0t = torch.as_tensor(np.where(check, g0, N), device=dev)
        lane = torch.arange(B, device=dev)
        n_bad = 0

        def mismatch():
            pp, pt = outs["pose"][g0m1], outs["trans"][g0m1]
            # a boundary crossed by one continuous re-run chain from the left is consistent by construction
            w0, w1 = writer[g0m1], writer[g0t]
            cont = chk & (w0 == w1) & (w0 >= 0) & (w0 < lane)
            used["pose"][:B] = torch.where(cont[:, None, None], pp, used["pose"][:B])
            used["trans"][:B] = torch.where(cont[:, None], pt, used["trans"][:B])
            bad = chk & ~((pp == used["pose"][:B]).flatten(1).all(1) & (pt == used["trans"][:B]).all(1))
            return bad, pp, pt

        if exact and check.any():
            # pass >= 2: every inconsistent piece is re-run from g0 seeded with the current state at g0-1 and
            # CONTINUES past its piece end (through later pieces) until its chain merges bitwise with the stored
            # outputs (or CAP frames). A continuation from the left reaches any frame later in time than a re-run
            # starting at that frame's piece, so the more up-to-date chain is written last.
            CAP = int(np.ceil(min(N, 2 * sch["P"] + K) / K) * K)
            ar = torch.arange(CAP, device=dev)
            fr_t = torch.as_tensor(fr_in_clip, device=dev)
            Graw = g0t[:, None] + ar[None]
            inside = Graw < N
            G2 = Graw.clamp(max=N - 1)
            C2 = clip_of[G2]
            F2 = fr_t[G2] == 0
            for _ in range(max_passes):
                bad, pred_p, pred_t = mismatch()
                n_bad = int(bad.sum())
                if n_bad == 0:
                    break
                dif = torch.maximum((pred_p - used["pose"][:B]).abs().amax((1, 2)), (pred_t - used["trans"][:B]).abs().amax(1))[bad]
                dq = [float(x) for x in torch.quantile(dif.double(), torch.tensor([0.5, 0.9, 1.0], dtype=torch.float64, device=dev))]
                used["pose"][:B] = torch.where(bad[:, None, None], pred_p, used["pose"][:B])
                used["trans"][:B] = torch.where(bad[:, None], pred_t, used["trans"][:B])
                A2 = inside & bad[:, None]
                gr = self._pass(G2, C2, F2, A2, torch.where(A2, G2, torch.full_like(G2, N)), tgt_all, ref_all,
                                rest_tab, off_tab, outs, seed=(bad, pred_p, pred_t), merge=bad, writer=writer)
                passes.append(dict(lanes=n_bad, groups=gr, seed_diff_p50_p90_max=[float(f"{x:.3g}") for x in dq]))
            else:
                n_bad = int(mismatch()[0].sum())
        stats = dict(N=N, P=sch["P"], steps_pass1=S_pad, warm_frames=sch["warm_frames"], checked_pieces=int(check.sum()),
                     passes=passes, total_groups=int(sum(p["groups"] for p in passes)), pieces_not_exact=n_bad,
                     lane_util=round(N / (B * K * sum(p["groups"] for p in passes)), 4))
        return {k: v[:N] for k, v in outs.items()}, stats


def rest_tables(rest_list, device):
    """per-clip float32 rest joints -> (float64 rest, MJCF-parsed float64 offsets) tables on device."""
    rj, off = gpu_ik.mjcf_rest_offsets(np.stack(rest_list))
    return (torch.as_tensor(rj, dtype=torch.float64, device=device),
            torch.as_tensor(off, dtype=torch.float64, device=device))
