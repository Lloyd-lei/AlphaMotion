"""ARCH V6 temporal layer — the temporal layer rebuilt to be STRUCTURALLY
ISOMORPHIC to the spatial layer (lloyd's insight #3c, 2026-08-07).

    train_v6.py --out RUNDIR [--residual_in] [--frame_repr key|code]

=============================================================================
THE ISOMORPHISM
=============================================================================
Read model_dual.py::_trunk + masking.py next to this file; every line below is the
same line with the JOINT axis replaced by the FRAME axis.

  SPATIAL (model_dual.DualTranslator, masking.sample_joint_mask)
    J joint tokens, one per joint (9 numbers: global rot6d + root-rel position)
    -> a random subset of JOINTS is masked on the ENCODER input; the hidden joint's
       pose embedding is replaced by a learned mask token while its GEOMETRY
       (rest offset, bone length, depth, name semantics) stays visible
    -> self-attention ACROSS JOINTS
    -> Perceiver pooling to K latent registers
    -> FSQ
    -> decoder reconstructs ALL J joints. The loss is on everything, masked or not.
    -> the mask FAMILIES (rand / ee / root / sparse) are the product surface: the
       same weights complete a body from 22 visible joints or from 1.

  TEMPORAL V6 (this file, frame_masking.sample_frame_mask)
    T frame tokens, one per frame (nk key joints x 9 = 54 numbers, --frame_repr key;
       or the frozen 3328-d spatial frame code, --frame_repr code)
    -> a random subset of FRAMES is masked on the ENCODER input; the hidden frame's
       pose embedding is replaced by a learned mask token while its TIME COORDINATE
       (Fourier clock) stays visible
    -> self-attention ACROSS FRAMES
    -> Perceiver pooling to n_path ordered registers
    -> FSQ
    -> decoder reconstructs ALL T frames. The loss is on everything, masked or not.
    -> the mask FAMILIES (none / rand / endpoints / sparse / prefix) are the product
       surface: `endpoints` IS V5's entire conditioning regime, `prefix` is streaming
       rollout, `sparse` is animator keyframes.

=============================================================================
WHAT IS DIFFERENT FROM V5, STATED PLAINLY
=============================================================================
V5's conditioning surface was ONE fixed pair of frames — the current state and a
target state — handed to the decoder as two tokens plus a straight-line baseline.
Nothing else about the window's interior was ever asked for, at any density. With a
fixed and always-present pair of boundary frames, the cheapest solution for the
decoder is the per-frame CONDITIONAL MEAN of trajectories with those endpoints, and
that is measurably what V5c produced: prefix ladder flat to 0.017 cm from 1 register
to 8, energy 0.762, jumping jacks historically 0.2-0.4 (a jack whose interior is
averaged away).

V6 changes the shape of the demand, not the weights of the loss:

 1. THE MASK IS A DISTRIBUTION, NOT A CONSTANT. The decoder's visible set is drawn
    per sample from a family mix. `endpoints` (40%) reproduces V5c exactly, so the
    V5c-comparable eval regime is trained, not extrapolated. The other 60% ask for
    reconstruction from 1-3 keyframes, from a random 20-80% subset, or from a past
    prefix. A per-frame conditional mean is not a solution to that mixture: at
    `sparse` density there is no endpoint pair to interpolate between, so the only
    place the interior can come from is the registers.

 2. THE ENCODER IS A MASKED AUTOENCODER OVER FRAMES. V5's encoder always saw all 60
    frames and additionally ingested the two conditioning tokens — which is exactly
    the encoder/predictor collusion channel V5d's --ema_target was built to sever.
    V6's encoder sees only the frames its own mask leaves visible and NEVER sees the
    conditioning; the registers therefore have to be a description of the motion
    rather than a correction term on top of information the decoder already holds.

 3. PIECEWISE-LINEAR BASELINE THROUGH THE VISIBLE FRAMES. V5's architectural prior
    was "predict the deviation from the straight line between the two endpoints".
    V6 generalizes it to "predict the deviation from the piecewise-linear
    interpolation through whatever frames are visible" — which, when the visible set
    is {first, target}, IS V5's straight line, bit for bit.

Everything else is held fixed on purpose. Same 6 key joints, same 60-frame window,
same V5c loss recipe (warp aug `linear` + Riemann `both` + cdrop .15/.05 +
--cdrop_time_detach + w_vel 0.5), same 8 path registers + 1 time register on the
same 6x5 FSQ grid = the same 62.69 bit/s. V6 differs from V5c in the encoder and the
masking architecture and in nothing else, so the eval table attributes cleanly.

=============================================================================
--residual_in  (the other half of insight 3c)
=============================================================================
"punch vs push differ mainly in acceleration/jerk". Instead of absolute per-frame
states, the encoder's frame token is built from FRAME-TO-FRAME RESIDUALS:

  velocity     body-frame relative rotation R[t-1]^T R[t] (identity subtracted) plus
               the position delta. The angular half is bone-length-free by
               construction — R10_RATE_PROBES.md probe 3 measured it transferring
               across bodies at 7-10% magnitude error vs 15% for positional speed.
  acceleration d/dt of the SG5-SMOOTHED velocity. R10 probe 2: raw second differences
               are 4.5-15 dB SNR, SG5 lifts them to 18-28 dB. The smoothing is
               mandatory, not cosmetic.
  jerk         NOT PRESENT. R10 probe 2 measured raw third differences at -11..-0.3
               dB at 30 fps, and the SG9 window that would rescue them is 0.3 s —
               long enough to erase the punch transient being measured. Red light,
               and V6 obeys it.

Drift is structurally impossible because absolutes are kept at I-FRAMES: the first
visible frame of each contiguous visible run carries its absolute state (frame 0
always does, and it is never maskable), every later frame in the run is a P-frame
carried by rates alone. That is a video codec's I/P structure on the pose stream.

=============================================================================
--frame_repr code
=============================================================================
`key` (default) = the 6 key joints' SE3, 54 numbers per frame — comparable with V5.
`code`          = the frozen spatial layer's full 3328-d frame code (e13_stream),
                  dequantized and projected into the frame token IN ADDITION to the
                  key state. The reconstruction target stays the key-joint stream, so
                  every metric in eval_v5 / n3_eval_temporal keeps its meaning; only
                  the encoder is richer. Streams exist for human_smpl only under
                  E19_K128; eval harnesses do not pass codes, so a code-mode model is
                  evaluated with its code channel zeroed unless the caller supplies it.
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                              # noqa: E402
import frame_masking as FM                                         # noqa: E402
from model import FSQ                                              # noqa: E402
from temporal_common import multires_dyn_loss, XBlock, fourier_time  # noqa: E402
from train_v5 import orthonormalize, to_states                     # noqa: E402


# --------------------------------------------------------------------------- codes
class CodeStream:
    """Frozen spatial-layer frame codes for --frame_repr code.

    e13_stream/<tag>/codes.npy is [n_frames, D] uint8, row-aligned with the paired
    cache (rows.npy is the identity), so a cache row index indexes it directly.
    Dequantization matches FSQ.from_ints: (code - offset) / half -> [-1, 1].
    """
    def __init__(self, root, tag):
        d = f"{root}/{tag}"
        self.meta = json.load(open(f"{d}/meta.json"))
        self.codes = np.load(f"{d}/codes.npy", mmap_mode="r")
        self.D = int(self.meta["D"])
        self.off = float(self.meta["deq_offset"][0])
        self.half = float(self.meta["deq_half"][0])
        assert self.codes.shape[1] == self.D, (self.codes.shape, self.D)

    def rows(self, idx):
        """idx int64 [...] cache rows -> float32 [..., D] in [-1,1]."""
        flat = np.asarray(idx).reshape(-1)
        c = np.asarray(self.codes[flat], dtype=np.float32)
        c = (c - self.off) / self.half
        return torch.from_numpy(c).reshape(*np.asarray(idx).shape, self.D)


# --------------------------------------------------------------------------- model
class V6Tokenizer(nn.Module):
    """Frame-token masked autoencoder -> ordered FSQ registers -> full-window decode.

    Public surface is deliberately identical to V5Tokenizer's (encode / decode /
    forward / baseline / token_ints / budgets / n_path / T / nk / head_kind) so
    eval_v5.py and n3_eval_temporal.py measure it with zero changes.
    """

    def __init__(self, nk=6, T=60, d_model=256, n_heads=8, enc_layers=5,
                 readout_layers=2, dec_layers=3, n_path=8, n_time=1,
                 fsq_levels=(5, 5, 5, 5, 5, 5), n_freq=8, head="reg",
                 curve_pts=8, interp="linear", cond_drop=False,
                 frame_repr="key", code_dim=3328, residual_in=False,
                 residual_keep_abs=False, rate_scale=16.0, acc_scale=35.0,
                 dec_ctx="visible", reg_ar=False, n_bodies=0):
        super().__init__()
        assert head == "reg", "V6 is regression-only; the CE control lives in V5"
        assert interp in ("nearest", "linear")
        assert dec_ctx in ("visible", "none")
        assert frame_repr in ("key", "code")
        self.nk, self.T, self.head_kind = nk, T, head
        self.n_path, self.n_time, self.curve_pts = n_path, n_time, curve_pts
        self.n_reg = n_path + n_time
        self.interp, self.cond_drop, self.dec_ctx = interp, cond_drop, dec_ctx
        self.frame_repr, self.residual_in = frame_repr, residual_in
        self.residual_keep_abs = residual_keep_abs
        self.rate_scale, self.acc_scale = rate_scale, acc_scale
        self.n_freq = n_freq
        d, S = d_model, nk * 9
        self.state_dim = S

        # ---------------- frame tokens (the joint tokens' axis-swapped twin) -----
        self.inp = nn.Linear(S, d)                     # absolute state
        if residual_in:
            self.inp_v = nn.Linear(S, d)               # velocity  (angular + linear)
            self.inp_a = nn.Linear(S, d)               # SG5-smoothed acceleration
            self.anchor_emb = nn.Parameter(torch.randn(d) * 0.02)   # I-frame marker
        if frame_repr == "code":
            self.inp_code = nn.Linear(code_dim, d)
        # the TIME COORDINATE is the frame's "geometry": always visible, never masked
        self.enc_time = nn.Linear(2 * n_freq + 1, d)
        # replaces a HIDDEN frame's pose embedding — model_dual.mask_tok, same init
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                           norm_first=True, activation="gelu",
                                           dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, enc_layers)

        # ---------------- Perceiver pooling -> FSQ -------------------------------
        self.path_q = nn.Parameter(torch.randn(n_path, d) * 0.02)      # ORDERED (OAT)
        self.readout = nn.ModuleList([XBlock(d, n_heads) for _ in range(readout_layers)])
        self.reg_ar = reg_ar
        if reg_ar:
            # N3a's insight #2, ported verbatim from V5Tokenizer. The parallel
            # Perceiver readout factorizes the joint code distribution into per-
            # register INDEPENDENT MARGINALS (the XBlock queries never see each
            # other), so each register regresses to its own conditional mean and the
            # OAT prefix ladder goes flat — register 1 alone already says everything
            # registers 1..8 say. Chaining restores joint coherence: register k's
            # readout query cross-attends [frame features ; the QUANTIZED values of
            # registers < k]. The chain is its own teacher (the prefix a register
            # conditions on IS the produced prefix), so train and inference are the
            # same 8 tiny readout steps.
            #
            # This is a DIFFERENT failure mode from the one the frame masking
            # attacks. Masking attacks mean regression (the decoder's conditioning
            # never varied); reg_ar attacks register redundancy (the readout's
            # queries never saw each other). V6_regar is the first arm to carry both.
            f = len(fsq_levels)
            self.ar_in = nn.Linear(f, d)
            self.ar_pos = nn.Parameter(torch.randn(n_path, d) * 0.02)
        self.fsq_path = FSQ(d, fsq_levels)
        self.fsq_time = FSQ(d, fsq_levels)
        # TIME head: unchanged from V5 — its only input is the window's own path
        # integral, so "how many frames the path takes" stays a separate codebook.
        self.time_mlp = nn.Sequential(nn.Linear(curve_pts + 3, 4 * d), nn.GELU(),
                                      nn.Linear(4 * d, d))
        self.time_q = nn.Parameter(torch.randn(n_time, d) * 0.02)

        # ---------------- decoder ------------------------------------------------
        self.reg_id = nn.Parameter(torch.randn(self.n_reg, d) * 0.02)
        self.dec_time = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(),
                                      nn.Linear(d, d))
        if dec_ctx == "visible":
            # every VISIBLE frame becomes one decoder KV token: its state + its
            # clock. With visible = {first, target} these are exactly V5's two
            # conditioning tokens; the family mix is what makes them a distribution.
            self.ctx_in = nn.Linear(S, d)
            self.ctx_id = nn.Parameter(torch.randn(d) * 0.02)
            if cond_drop:
                self.cond_null = nn.Parameter(torch.randn(2, d) * 0.02)
        if n_bodies:
            self.body_emb = nn.Embedding(n_bodies, d)
            self.body_reg_id = nn.Parameter(torch.randn(d) * 0.02)
        self.dec = nn.ModuleList([XBlock(d, n_heads) for _ in range(dec_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, S))
        with torch.no_grad():                      # start at ZERO deviation (V5's fix)
            self.out[-1].weight.mul_(0.01)
            self.out[-1].bias.zero_()
        self.budgets = V.prefix_budgets(n_path)

    # ------------------------------------------------------------------ helpers
    def _tfeat(self, B, T, device, dtype):
        t = torch.linspace(0.0, 1.0, T, device=device)
        return fourier_time(t, self.n_freq).to(dtype)[None].expand(B, -1, -1)

    def _canon(self, w):
        """Resample the window to the canonical T frames (V5's _canon, verbatim)."""
        T_in = w.shape[1]
        if T_in == self.T:
            return w
        if self.interp == "linear":
            t = torch.linspace(0.0, T_in - 1, self.T, device=w.device)
            i0 = t.floor().long()
            i1 = (i0 + 1).clamp(max=T_in - 1)
            a = (t - i0.to(t.dtype))[None, :, None, None].to(w.dtype)
            return orthonormalize((1 - a) * w[:, i0] + a * w[:, i1])
        i = torch.round(torch.linspace(0, T_in - 1, self.T, device=w.device)).long()
        return w[:, i]

    def _canon_code(self, c):
        """Nearest-frame canonicalization for the discrete code channel."""
        T_in = c.shape[1]
        if T_in == self.T:
            return c
        i = torch.round(torch.linspace(0, T_in - 1, self.T, device=c.device)).long()
        return c[:, i]

    # ------------------------------------------------------------------ encoder
    def frame_tokens(self, wc, enc_mask=None, code=None):
        """[B,T,nk,9] (+ optional [B,T,D] codes) -> [B,T,d] post-attention features.

        This is model_dual._trunk with the joint axis swapped for the frame axis:
        pose -> linear, hidden entries -> learned mask token, always-visible
        conditioning added on top (there: skeleton geometry; here: the clock), then
        self-attention across the axis.
        """
        B, T = wc.shape[0], wc.shape[1]
        x = wc.reshape(B, T, -1)
        if not self.residual_in:
            h = self.inp(x)
        else:
            vel, acc = FM.rate_features(wc)
            h = (self.inp_v(vel.reshape(B, T, -1) * self.rate_scale)
                 + self.inp_a(acc.reshape(B, T, -1) * self.acc_scale))
            if self.residual_keep_abs:
                h = h + self.inp(x)
            else:
                # I-FRAME / P-FRAME: absolutes only where a rate is not observable.
                vis = torch.ones(B, T, dtype=torch.bool, device=wc.device) \
                    if enc_mask is None else ~enc_mask
                anc = FM.anchor_flags(vis).to(h.dtype)[..., None]
                h = h + anc * (self.inp(x) + self.anchor_emb)
        if code is not None and self.frame_repr == "code":
            h = h + self.inp_code(code.to(h.dtype))
        if enc_mask is not None:
            h = torch.where(enc_mask[..., None], self.mask_tok.to(h.dtype).expand_as(h), h)
        h = h + self.enc_time(self._tfeat(B, T, wc.device, h.dtype))
        return self.enc(h)

    def path_features(self, w, enc_mask=None, code=None):
        """-> [B,n_path,f] pre-quantization PATH features (canonical time only).

        NOTE what is absent versus V5: no conditioning tokens are concatenated to the
        encoder input. The encoder cannot see the boundary states, so it cannot
        emit a code that is only meaningful as a correction to them.
        """
        wc = self._canon(w)
        B = wc.shape[0]
        h = self.frame_tokens(wc, enc_mask,
                              None if code is None else self._canon_code(code))
        if getattr(self, "reg_ar", False):
            # sequential register production; quantization inside the loop is the
            # SAME deterministic straight-through rounding encode() re-applies
            # afterwards, so the values the chain conditioned on are exactly the
            # values the decoder later reads.
            feats, prefix = [], []
            for k in range(self.n_path):
                kv = h if not prefix else torch.cat([h] + prefix, 1)
                r = self.path_q[k:k + 1][None].expand(B, -1, -1)
                for b in self.readout:
                    r = b(r, kv)
                f_k = self.fsq_path.features(r)                    # [B,1,f]
                feats.append(f_k)
                q_k = self.fsq_path.quantize(f_k)                  # straight-through
                prefix.append(self.ar_in(q_k) + self.ar_pos[k][None, None])
            return torch.cat(feats, 1)
        r = self.path_q[None].expand(B, -1, -1)
        for b in self.readout:
            r = b(r, h)
        return self.fsq_path.features(r)

    def curve_summary(self, w):
        """V5's, verbatim: [normalized cumulative arc-length shape ; length, frame
        ratio, mean speed]. The TIME codebook's only input."""
        c = V.arclen_curve(w)
        T = c.shape[1]
        end = c[:, -1:].clamp_min(1e-4)
        i = torch.round(torch.linspace(0, T - 1, self.curve_pts,
                                       device=w.device)).long()
        shape = (c / end)[:, i]
        rate = torch.stack([torch.log1p(end[:, 0]),
                            torch.full_like(end[:, 0], math.log2(T / self.T)),
                            end[:, 0] / T], -1)
        return torch.cat([shape, rate], -1)

    def time_features(self, w):
        h = self.time_mlp(self.curve_summary(w).to(self.time_q.dtype))
        return self.fsq_time.features(h[:, None] + self.time_q[None])

    def encode(self, w, cur=None, tgt=None, k_norm=None, drop_cur=None,
               drop_tgt=None, enc_mask=None, code=None):
        """-> dict with straight-through FSQ values and decoder-ready embeddings.

        cur / tgt / k_norm / drop_* are accepted and IGNORED (the encoder is
        conditioning-free by design); they are in the signature only so eval_v5's
        tokens_of and n3_eval_temporal call V5 and V6 with the same code path.
        """
        fp = self.path_features(w, enc_mask, code)
        ft = self.time_features(w)
        qp = self.fsq_path.quantize(fp)
        qt = self.fsq_time.quantize(ft)
        return {"path_f": fp, "time_f": ft, "path_q": qp, "time_q": qt,
                "path_z": self.fsq_path.proj_out(qp),
                "time_z": self.fsq_time.proj_out(qt)}

    def ints(self, q, fsq):
        half = (fsq._levels - 1) / 2
        return torch.round(q * half + torch.ceil(half)).long()

    def token_ints(self, e):
        return (self.ints(e["path_q"], self.fsq_path),
                self.ints(e["time_q"], self.fsq_time))

    # ------------------------------------------------------------------ baselines
    @staticmethod
    def endpoint_ctx(cur, tgt, k_norm, T_out, drop_tgt=None):
        """(cur, tgt, k_norm) -> the V5 conditioning expressed as a VISIBLE-FRAME set.

        states [B,T_out,nk,9] with frame 0 = cur and frame k = tgt, vis marking those
        two. Feeding this to baseline_vis reproduces V5.baseline bit for bit
        (including "hold at the target after frame k" and, under drop_tgt, "hold at
        cur everywhere"), which is what makes the V5c-comparable eval regime exact.
        """
        B = cur.shape[0]
        dev = cur.device
        k = torch.round(k_norm.clamp(0, 1) * (T_out - 1)).long().clamp(0, T_out - 1)
        states = cur[:, None].expand(B, T_out, -1, -1).clone()
        vis = torch.zeros(B, T_out, dtype=torch.bool, device=dev)
        b = torch.arange(B, device=dev)
        states[b, k] = tgt
        vis[:, 0] = True
        vis[b, k] = True
        if drop_tgt is not None and bool(drop_tgt.any()):
            vis[b, k] &= ~drop_tgt
            vis[:, 0] = True
        return states, vis

    @staticmethod
    def baseline_vis(states, vis):
        """Piecewise-linear interpolation through the VISIBLE frames -> [B,T,nk,9].

        The generalization of V5's straight-line prior. Before the first visible
        frame it holds; after the last it holds; between two visible frames it lerps
        and re-orthonormalizes. With vis = {0, k} this is exactly V5's baseline, and
        with vis = {0} it is "hold at cur", which is V5's drop_tgt behaviour.

        Same argument as V5's: the shortest path between fixed anchors is the right
        prior, so make it the architecture and let the network predict only the
        DEVIATION from it. Every token bit then goes to the part that is uncertain.
        """
        B, T = vis.shape
        dev = vis.device
        ar = torch.arange(T, device=dev)[None].expand(B, T)
        prev = torch.cummax(torch.where(vis, ar, torch.full_like(ar, -1)), 1).values
        prev = prev.clamp_min(0)
        nxt = torch.cummin(torch.where(vis, ar, torch.full_like(ar, T)).flip(1),
                           1).values.flip(1)
        nxt = torch.where(nxt >= T, prev, nxt)
        den = (nxt - prev).clamp_min(1).to(states.dtype)
        a = (((ar - prev).to(states.dtype) / den).clamp(0, 1))[..., None, None]
        g = prev[..., None, None].expand(-1, -1, states.shape[2], states.shape[3])
        g2 = nxt[..., None, None].expand_as(g)
        lo = torch.gather(states, 1, g)
        hi = torch.gather(states, 1, g2)
        return orthonormalize((1 - a) * lo + a * hi)

    def baseline(self, cur, tgt, k_norm, T_out, drop_tgt=None):
        """V5Tokenizer.baseline, VERBATIM — the straight line between the two boundary
        states in CONTINUOUS normalized time, held at the target after frame k (and
        held at `cur` when the target is dropped).

        Copied rather than derived from baseline_vis because k_norm*(T_out-1) need
        not land on an integer frame, and eval_v5.stat8_ladder prints this row as the
        zero-bit floor: it has to be numerically the same floor V5c was judged
        against. (When k_norm*(T_out-1) IS an integer — every training step and every
        harness call, which all use k_norm=1 or k/(T-1) — baseline_vis through
        {0, k} agrees with this to 2.4e-07.)
        """
        B = cur.shape[0]
        t = torch.linspace(0.0, 1.0, T_out, device=cur.device, dtype=cur.dtype)
        u = (t[None, :] / k_norm[:, None].clamp_min(1e-3)).clamp(0.0, 1.0)
        if drop_tgt is not None and bool(drop_tgt.any()):
            u = u * (~drop_tgt)[:, None].to(u.dtype)
        u = u[..., None, None]
        return orthonormalize((1 - u) * cur[:, None] + u * tgt[:, None])

    # ------------------------------------------------------------------ decoder
    def _ctx_tokens(self, states, vis, drop_cur=None, drop_tgt=None):
        """visible frames -> ([B,T,d] KV tokens, [B,T] bool padding mask).

        Invisible frames are padded out of the attention, so the decoder literally
        cannot read a frame the mask hid. drop_cur nulls frame 0's token while the
        baseline still anchors on it — V5's exact drop_cur semantics.
        """
        B, T = vis.shape
        c = (self.ctx_in(states.reshape(B, T, -1))
             + self.dec_time(self._tfeat(B, T, states.device, states.dtype))
             + self.ctx_id)
        if drop_cur is not None and bool(drop_cur.any()):
            c[:, 0] = torch.where(drop_cur[:, None], self.cond_null[0].to(c.dtype),
                                  c[:, 0])
        return c, ~vis

    def decode(self, path_z, time_z, cur=None, tgt=None, k_norm=None, T_out=None,
               hidden=None, drop_cur=None, drop_tgt=None, body=None, ctx=None):
        """ctx = (states [B,T_out,nk,9], vis [B,T_out] bool): what the decoder sees.

        ctx=None falls back to the ENDPOINT set built from (cur, tgt, k_norm) — i.e.
        V5c's regime — which is what every existing eval harness ends up asking for.
        `hidden` [B,n_path] True = that path register is invisible (OAT prefix); the
        time register and the context tokens are never hidden.
        """
        B = path_z.shape[0]
        T_out = T_out or self.T
        kv = torch.cat([path_z, time_z], 1) + self.reg_id[None]
        pad = torch.zeros(B, kv.shape[1], dtype=torch.bool, device=kv.device)
        if hidden is not None:
            pad[:, :self.n_path] = hidden
        from_endpoints = ctx is None
        if self.dec_ctx == "visible":
            if ctx is None:
                ctx = self.endpoint_ctx(cur, tgt, k_norm, T_out, drop_tgt)
            states, vis = ctx
            ct, cpad = self._ctx_tokens(states, vis, drop_cur, drop_tgt)
            kv = torch.cat([kv, ct.to(kv.dtype)], 1)
            pad = torch.cat([pad, cpad], 1)
            base = (self.baseline(cur, tgt, k_norm, T_out, drop_tgt)
                    if from_endpoints else self.baseline_vis(states, vis))
        else:
            states = vis = None
            base = self.baseline(cur, tgt, k_norm, T_out, drop_tgt)
        if getattr(self, "body_emb", None) is not None:
            if body is None:
                body = torch.zeros(B, dtype=torch.long, device=path_z.device)
            kv = torch.cat([kv, (self.body_emb(body) + self.body_reg_id)[:, None]], 1)
            pad = torch.cat([pad, torch.zeros(B, 1, dtype=torch.bool,
                                              device=pad.device)], 1)
        q = self.dec_time(self._tfeat(B, T_out, path_z.device, path_z.dtype))
        for b in self.dec:
            q = b(q, kv, key_padding_mask=pad)
        o = self.out(q)
        return orthonormalize(base + o.reshape(B, T_out, self.nk, 9))

    def forward(self, w, cur, tgt, k_norm, hidden=None, T_out=None, drop_cur=None,
                drop_tgt=None, time_detach=None, body=None, enc_mask=None,
                dec_vis=None, code=None):
        """enc_mask [B,T_canon] True = hidden from the ENCODER.
        dec_vis  [B,T_out]   True = visible to the DECODER (None -> endpoints)."""
        e = self.encode(w, enc_mask=enc_mask, code=code)
        tz = e["time_z"]
        if time_detach is not None and bool(time_detach.any()):
            tz = torch.where(time_detach[:, None, None], tz.detach(), tz)
        T_out = T_out or w.shape[1]
        ctx = None
        if dec_vis is not None and self.dec_ctx == "visible":
            ctx = (w.detach(), dec_vis)
        return self.decode(e["path_z"], tz, cur, tgt, k_norm, T_out, hidden,
                           drop_cur, drop_tgt, body=body, ctx=ctx), e


# --------------------------------------------------------------------------- run io
def build_v6(cfg):
    return V6Tokenizer(
        nk=cfg["nk"], T=cfg.get("T", V.T_WIN), d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8), enc_layers=cfg.get("enc_layers", 5),
        readout_layers=cfg.get("readout_layers", 2), dec_layers=cfg.get("dec_layers", 3),
        n_path=cfg.get("n_path", 8), n_time=cfg.get("n_time", 1),
        fsq_levels=tuple(int(x) for x in str(cfg.get("fsq_levels", "5,5,5,5,5,5")).split(",")),
        interp=cfg.get("aug_interp", "linear"),
        cond_drop=bool(float(cfg.get("cond_drop_tgt", 0) or 0) > 0
                       or float(cfg.get("cond_drop_both", 0) or 0) > 0),
        frame_repr=cfg.get("frame_repr", "key"),
        code_dim=int(cfg.get("code_dim", 3328)),
        residual_in=bool(cfg.get("residual_in", False)),
        residual_keep_abs=bool(cfg.get("residual_keep_abs", False)),
        rate_scale=float(cfg.get("rate_scale", 16.0)),
        acc_scale=float(cfg.get("acc_scale", 35.0)),
        dec_ctx=cfg.get("dec_ctx", "visible"),
        reg_ar=bool(cfg.get("reg_ar", False)),
        n_bodies=len(cfg.get("xbody_bodies") or []))


def load_run(run, device="cpu"):
    cfg = json.load(open(f"{run}/config.json"))
    m = build_v6(cfg).to(device)
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location=device))
    m.eval()
    return m, cfg


# --------------------------------------------------------------------------- eval
def sample_hidden(B, n_path, budgets, device, rng):
    n = torch.as_tensor(budgets, device=device)[
        torch.as_tensor(rng.integers(0, len(budgets), B), device=device)]
    return torch.arange(n_path, device=device)[None] >= n[:, None]


def _fixed_vis(kind, B, T, device):
    """Deterministic decoder-visibility sets for the val mask ladder."""
    vis = torch.zeros(B, T, dtype=torch.bool, device=device)
    if kind == "endpoints":
        vis[:, 0] = True; vis[:, -1] = True
    elif kind == "sparse3":
        vis[:, 0] = True
        for i in (T // 4, T // 2, 3 * T // 4):
            vis[:, i] = True
    elif kind == "rand50":
        g = torch.Generator(device="cpu").manual_seed(7)
        vis = (torch.rand(B, T, generator=g).to(device) < 0.5)
        vis[:, 0] = True
    elif kind == "prefix_half":
        vis[:, : T // 2] = True
    elif kind == "full":
        vis[:] = True
    else:
        raise ValueError(kind)
    return vis


@torch.no_grad()
def evaluate(model, kd, embs, T, span, dev, n=192, seed=7):
    """V5's evaluate() + the MASK LADDER, which is the number V6 exists to move:
    key error as a function of how much the decoder was allowed to see."""
    model.eval()
    g = np.random.default_rng(seed)
    starts, _ = kd.sample_starts(n, g, "val", span=span)
    out = {}
    emb = embs[0]
    w = kd.window(emb, starts, T, span).to(dev)
    cur, tgt = w[:, 0], w[:, -1]
    kn = torch.ones(w.shape[0], device=dev)
    for b in model.budgets:
        hid = (torch.arange(model.n_path, device=dev)[None] >= b).expand(w.shape[0], -1)
        p = model(w, cur, tgt, kn, hid)[0].float()
        l, ep = V.pathint_loss(p, w.float())
        pe, re = V.key_se3_err(p, w.float(), kd.reach[emb])
        out[f"n{b}"] = {"pathint": round(float(l), 4),
                        "endpoint_rel_err": round(float(ep), 4),
                        "key_pos_cm": round(pe, 3), "key_rot_deg": round(re, 2),
                        "energy_ratio": round(float(p.std(1).mean()
                                                    / w.float().std(1).mean()), 3)}
    ladder = {}
    for kind in ("endpoints", "sparse3", "rand50", "prefix_half", "full"):
        vis = _fixed_vis(kind, w.shape[0], T, dev)
        p = model(w, cur, tgt, kn, dec_vis=vis)[0].float()
        pe, _ = V.key_se3_err(p, w.float(), kd.reach[emb])
        bl = model.baseline_vis(w, vis).float()
        pb, _ = V.key_se3_err(bl, w.float(), kd.reach[emb])
        ladder[kind] = {"key_pos_cm": round(pe, 3),
                        "interp_only_cm": round(pb, 3),
                        "energy_ratio": round(float(p.std(1).mean()
                                                    / w.float().std(1).mean()), 3)}
    out["mask_ladder"] = ladder
    out["static_anchor"] = round(V.static_pathint_anchor(w.float()), 4)
    bl = model.baseline(cur, tgt, kn, T).float()
    out["straight_line_baseline"] = {
        "pathint": round(float(V.pathint_loss(bl, w.float())[0]), 4),
        "key_pos_cm": round(V.key_se3_err(bl, w.float(), kd.reach[emb])[0], 3),
        "energy_ratio": round(float(bl.std(1).mean() / w.float().std(1).mean()), 3)}
    model.train()
    return out


# --------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--embs", default="human_smpl,unitree_h1,booster_t1,fourier_gr3")
    ap.add_argument("--T", type=int, default=V.T_WIN)
    ap.add_argument("--n_path", type=int, default=8)
    ap.add_argument("--n_time", type=int, default=1)
    ap.add_argument("--fsq_levels", default="5,5,5,5,5,5")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=5)
    ap.add_argument("--readout_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=2500)
    # ---- V6 architecture ----------------------------------------------------
    ap.add_argument("--enc_mix", default=",".join(f"{k}:{v}" for k, v in FM.ENC_MIX.items()),
                    help="frame-mask family mix on the ENCODER input (the MAE half)")
    ap.add_argument("--dec_mix", default=",".join(f"{k}:{v}" for k, v in FM.DEC_MIX.items()),
                    help="frame-mask family mix for the DECODER's visible set (the "
                         "anti-mean-regression half); `endpoints` = V5c's regime")
    ap.add_argument("--tie_masks", action="store_true",
                    help="use ONE draw for both (the pure partial-observation "
                         "deployment regime: the decoder sees exactly what the "
                         "encoder saw). Off = two independent knobs.")
    ap.add_argument("--frame_repr", choices=["key", "code"], default="key")
    ap.add_argument("--code_stream", default="E19_K128_amass_human_smpl")
    ap.add_argument("--residual_in", action="store_true",
                    help="feed frame-to-frame RESIDUALS (angular+linear velocity and "
                         "SG5-smoothed acceleration; no jerk, R10 red light) with "
                         "absolute I-frame anchors instead of absolute states")
    ap.add_argument("--residual_keep_abs", action="store_true",
                    help="--residual_in variant: keep the absolute state on EVERY "
                         "frame and add the rates as extra channels")
    ap.add_argument("--rate_scale", type=float, default=16.0,
                    help="MEASURED on 256 AMASS train windows (2026-08-07): the "
                         "absolute key state has RMS 0.507, velocity 0.0324, "
                         "acceleration 0.0146. These two scales put all three "
                         "channels on the same footing (0.507/0.0324 = 15.6, "
                         "0.507/0.0146 = 34.7) so no channel enters the encoder "
                         "two orders of magnitude quieter than its neighbours.")
    ap.add_argument("--acc_scale", type=float, default=35.0)
    ap.add_argument("--reg_ar", action="store_true",
                    help="N3a's insight #2 inside V6: produce the 8 ordered path "
                         "registers SEQUENTIALLY (register k's quantized value "
                         "conditions k+1..) instead of the parallel readout whose "
                         "per-register marginals each regress to their conditional "
                         "mean. Attacks the FLAT OAT LADDER, which the frame masking "
                         "does not touch — masking attacks mean regression, this "
                         "attacks register redundancy. OAT prefix budgets unchanged "
                         "(prefix-n = the first n of the chain), bit budget "
                         "unchanged. Off = bit-exact prior V6 behavior.")
    ap.add_argument("--dec_ctx", choices=["visible", "none"], default="visible",
                    help="'none' = the purist isomorph (decoder reads ONLY the "
                         "registers + the clock, like the spatial decoder reading "
                         "only the codes + the skeleton)")
    # ---- V5c recipe, carried over unchanged ---------------------------------
    ap.add_argument("--aug_interp", choices=["nearest", "linear"], default="linear")
    ap.add_argument("--timing_loss", choices=["cumulative", "riemann", "both"],
                    default="both")
    ap.add_argument("--w_riemann", type=float, default=1.0)
    ap.add_argument("--riemann_deadzone", type=float, default=0.02)
    ap.add_argument("--cond_drop_tgt", type=float, default=0.15)
    ap.add_argument("--cond_drop_both", type=float, default=0.05)
    ap.add_argument("--cdrop_time_detach", action="store_true", default=True)
    ap.add_argument("--no_cdrop_time_detach", dest="cdrop_time_detach",
                    action="store_false")
    ap.add_argument("--w_pathint", type=float, default=1.0)
    ap.add_argument("--w_frame", type=float, default=6.0)
    ap.add_argument("--w_multires", type=float, default=2.0)
    ap.add_argument("--w_vel", type=float, default=0.5)
    ap.add_argument("--w_nce", type=float, default=0.1)
    ap.add_argument("--w_inv", type=float, default=1.0)
    ap.add_argument("--w_sep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke: 30 steps, tiny model")
    V.add_root_vel_flag(ap)
    args = ap.parse_args()
    if args.tiny:
        args.steps, args.bs, args.d_model, args.enc_layers = 30, 4, 32, 1
        args.readout_layers = args.dec_layers = 1
        args.eval_every, args.device = 20, "cpu"
        args.embs = "human_smpl,unitree_h1"
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev, t0 = args.device, time.time()
    enc_mix = FM.parse_mix(args.enc_mix)
    dec_mix = FM.parse_mix(args.dec_mix)

    embs = [e for e in args.embs.split(",") if e]
    kd = V.KeyData(args.cache, embs, device=dev,
                   max_clips=200 if args.tiny else 0, root_vel=args.root_vel)
    nk = kd.key[embs[0]].shape[1]
    span = args.T
    cs = None
    if args.frame_repr == "code":
        cs = CodeStream(V.STREAM_ROOT, args.code_stream)
        print(f"[code] {args.code_stream} D={cs.D} rows={cs.codes.shape[0]}", flush=True)

    cfg = vars(args) | {"arch": "v6", "nk": nk, "span": span,
                        "key_joints": kd.knames[embs[0]], "cache": args.cache,
                        "frozen_spatial": V.FROZEN, "head": "reg", "aug": True,
                        "code_dim": cs.D if cs else 3328,
                        "enc_mix_parsed": enc_mix, "dec_mix_parsed": dec_mix}
    model = build_v6(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    lv = [int(x) for x in args.fsq_levels.split(",")]
    bits = V.bit_budget(args.n_path, args.n_time, lv, args.T)
    cfg |= {"n_params": n_par, "bit_budget": bits, "budgets": list(model.budgets)}
    print(f"=== V6 (frame-token MAE) residual_in={args.residual_in} "
          f"frame_repr={args.frame_repr} dec_ctx={args.dec_ctx} ===")
    print(f"  key joints ({nk}) = {kd.knames[embs[0]]} -> state {nk*9} dims/frame")
    print(f"  params {n_par/1e6:.2f}M  path registers {args.n_path} (OAT budgets "
          f"{model.budgets})  time registers {args.n_time}")
    print("  BIT BUDGET " + json.dumps(bits))
    assert abs(bits["total_bits_per_s"] - 62.69) < 0.01, \
        f"bit budget {bits['total_bits_per_s']} != V5c's 62.69 — fix levels/registers"
    print(f"  enc_mix {enc_mix}\n  dec_mix {dec_mix}  tie_masks={args.tie_masks}")
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=2, default=str)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup,
                           0.5 * (1 + math.cos(math.pi * s / args.steps))))
    rng = np.random.default_rng(args.seed + 1)
    log, t_loop = [], time.time()
    amp = (dev == "cuda")
    model.train()

    for step in range(args.steps):
        starts, _ = kd.sample_starts(args.bs, rng, "train", span=span)
        T_a = args.T
        if rng.random() < 0.5:
            T_a = int(np.clip(round(args.T * 2.0 ** rng.uniform(-1, 1)),
                              args.T // 2, 2 * args.T))
        frac = (args.aug_interp == "linear") and (T_a != span)
        idx_a = V.uniform_fidx(span, T_a) if frac else V.uniform_idx(span, T_a)
        emb0 = embs[int(rng.integers(len(embs)))]
        wa = (kd.window_fidx(emb0, starts, idx_a, span) if frac
              else kd.window_idx(emb0, starts, idx_a)).to(dev)
        k = int(rng.integers(T_a // 2, T_a))
        cur, tgt = wa[:, 0], wa[:, k]
        kn = torch.full((args.bs,), k / max(T_a - 1, 1), device=dev)
        hid = sample_hidden(args.bs, model.n_path, model.budgets, dev, rng)
        code_a = None
        if cs is not None:
            rows = (np.asarray(starts)[:, None]
                    + np.round(np.asarray(idx_a)).astype(np.int64)[None])
            code_a = cs.rows(rows).to(dev)

        # ---- THE V6 STEP: two frame-mask draws -------------------------------
        # The two knobs live on DIFFERENT grids: the encoder always reads the
        # canonical model.T frames (via _canon), the decoder emits T_a. --tie_masks
        # can therefore only tie them when the warp augmentation left T_a == model.T;
        # on stretched views it falls back to two independent draws, which is stated
        # here rather than hidden because it makes `tie_masks` an approximate switch.
        k_enc = torch.full((args.bs,), min(k, model.T - 1), dtype=torch.long, device=dev)
        k_dec = torch.full((args.bs,), k, dtype=torch.long, device=dev)
        dec_mix_used = enc_mix if args.tie_masks else dec_mix
        dec_vis = ~FM.sample_frame_mask(args.bs, T_a, dec_mix_used, dev, k_idx=k_dec)
        if args.tie_masks and T_a == model.T:
            enc_mask = ~dec_vis
        else:
            enc_mask = FM.sample_frame_mask(args.bs, model.T, enc_mix, dev, k_idx=k_enc)
        # cdrop stays meaningful: it removes the TARGET frame from the visible set
        # (baseline then holds at cur — V5's exact behaviour) and nulls frame 0's token.
        dc = dt = None
        if model.cond_drop:
            r_ = torch.rand(args.bs, device=dev)
            dt = r_ < (args.cond_drop_tgt + args.cond_drop_both)
            dc = r_ < args.cond_drop_both
            dec_vis = dec_vis & ~(dt[:, None] & (torch.arange(T_a, device=dev)[None] == k))
            dec_vis[:, 0] = True
        td = dt if args.cdrop_time_detach else None

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o, e = model(wa, cur, tgt, kn, hid, drop_cur=dc, drop_tgt=dt,
                         time_detach=td, enc_mask=enc_mask, dec_vis=dec_vis,
                         code=code_a)
            parts = {}
            pred, gt = o, wa.float()
            pf = pred.float()
            loss = torch.zeros((), device=dev)
            if args.w_pathint > 0 and args.timing_loss in ("cumulative", "both"):
                lp, ep = V.pathint_loss(pf, gt)
                loss = loss + args.w_pathint * lp
                parts["pathint"] = float(lp.detach()); parts["endpoint"] = float(ep.detach())
            if (args.w_pathint > 0 and args.w_riemann > 0
                    and args.timing_loss in ("riemann", "both")):
                lr_, epr = V.riemann_timing_loss(pf, gt, deadzone=args.riemann_deadzone)
                loss = loss + args.w_riemann * lr_
                parts["riemann"] = float(lr_.detach())
            if args.w_frame > 0:
                lf = F.mse_loss(pf, gt)                   # ALL T frames, masked or not
                loss = loss + args.w_frame * lf
                parts["frame"] = float(lf.detach())
            if args.w_multires > 0 or args.w_vel > 0:
                zh, zg = pf.reshape(args.bs, T_a, -1), gt.reshape(args.bs, T_a, -1)
                if args.w_vel > 0:
                    dh, dg = zh[:, 1:] - zh[:, :-1], zg[:, 1:] - zg[:, :-1]
                    lv_ = ((dh - dg) ** 2).mean() / (dg.pow(2).mean() + 1e-6)
                    loss = loss + args.w_vel * lv_
                    parts["vel"] = float(lv_.detach())
                if args.w_multires > 0:
                    lm = multires_dyn_loss(zh, zg, (1, 2, 4, 8))
                    loss = loss + args.w_multires * lm
                    parts["multires"] = float(lm.detach())
            # ---- paired time-scale view (V5c's separation supervision) --------
            if args.w_inv > 0 or args.w_sep > 0:
                T_b = int(np.clip(round(args.T * 2.0 ** rng.uniform(-1, 1)),
                                  args.T // 2, 2 * args.T))
                if args.aug_interp == "linear":
                    idx_b = V.warp_fidx(span, T_b, rng)
                    wb = kd.window_fidx(emb0, starts, idx_b, span).to(dev)
                else:
                    idx_b = V.warp_idx(span, T_b, rng)
                    wb = kd.window_idx(emb0, starts, idx_b).to(dev)
                code_b = None
                if cs is not None:
                    rb = (np.asarray(starts)[:, None]
                          + np.round(np.asarray(idx_b)).astype(np.int64)[None])
                    code_b = cs.rows(rb).to(dev)
                # SAME enc_mask on both views: the invariance being taught is about
                # the PATH, not about which frames happened to be hidden.
                eb = model.encode(wb, enc_mask=enc_mask, code=code_b)
                if args.w_inv > 0:
                    li = F.mse_loss(eb["path_f"], e["path_f"])
                    loss = loss + args.w_inv * li
                    parts["path_inv"] = float(li.detach())
                if args.w_sep > 0:
                    gap = abs(math.log2(T_a / T_b))
                    d = (eb["time_f"] - e["time_f"]).pow(2).mean().sqrt()
                    ls = F.relu(0.5 * gap - d)
                    loss = loss + args.w_sep * ls
                    parts["time_sep"] = float(ls.detach())
            # ---- InfoNCE across embodiments -----------------------------------
            if args.w_nce > 0 and len(embs) > 1:
                e2 = embs[int(rng.integers(len(embs)))]
                if e2 != emb0:
                    w2 = (kd.window_fidx(e2, starts, idx_a, span) if frac
                          else kd.window_idx(e2, starts, idx_a)).to(dev)
                    ec = model.encode(w2, enc_mask=enc_mask, code=code_a)
                    a = F.normalize(e["path_q"].reshape(args.bs, -1), dim=-1)
                    b = F.normalize(ec["path_q"].reshape(args.bs, -1), dim=-1)
                    lg = a @ b.t() / args.tau
                    lab = torch.arange(args.bs, device=dev)
                    ln = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))
                    loss = loss + args.w_nce * ln
                    parts["nce"] = float(ln.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            sps = 100 / max(time.time() - t_loop, 1e-9) if step else 0.0
            t_loop = time.time()
            print(f"step {step} loss {loss.item():.4f} " +
                  " ".join(f"{k_} {v:.3f}" for k_, v in parts.items()) +
                  f"  dec_keep {float(dec_vis.float().mean()):.2f}"
                  f" enc_keep {float((~enc_mask).float().mean()):.2f}"
                  f"  {sps:.1f} steps/s ({time.time()-t0:.0f}s)", flush=True)
            if step == 200:
                print(f"  PROJECTED WALL TIME: "
                      f"{(args.steps - step)/max(sps,1e-6)/60:.1f} min", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = {"step": step + 1, "loss": round(loss.item(), 4),
                  "val": evaluate(model, kd, embs, args.T, span, dev)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=1)
            torch.save(model.state_dict(), f"{args.out}/model.pt")

    torch.save(model.state_dict(), f"{args.out}/model.pt")
    print(f"DONE {args.out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
