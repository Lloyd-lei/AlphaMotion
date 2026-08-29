"""ARCH V5 temporal layer — train the two-codebook (PATH x TIME) tokenizer.

  train_v5.py --arm {pathint, pathint_noaug, baseline_frameCE} --out RUNDIR

WHAT IS DIFFERENT FROM P1 (temporal_common.py / train_temporal.py), and why

  INPUT   P1 compressed the 3328-d frozen FRAME CODE per frame. V5 compresses the
          SE3 state of 6 KEY JOINTS — root, head, both wrists, both ankles — i.e.
          6 x 9 = 54 numbers, 60x less. The rest of the body is completed per frame
          by the frozen spatial layer at decode time; v5_probe_keyjoints.py is the
          measurement that prices that assumption, and its verdict should be read
          BEFORE these numbers.

  ARCH    bidirectional transformer over the window -> ordered Perceiver registers
          -> TWO SEPARATE FSQ codebooks:
            PATH  n_path ordered registers, 6 dims x 5 levels each (15,625 words)
                  — WHICH WAY the body goes. Reads the window resampled to a
                  CANONICAL frame count, so the raw frame count is structurally
                  invisible to it. This is a design choice, not an emergent
                  property, and eval_v5 labels it as such.
            TIME  1 register, same 6x5 grid — HOW MANY FRAMES the path takes. Its
                  ONLY input is the window's own PATH INTEGRAL: total arc length,
                  frame count, and the normalized shape of the cumulative
                  arc-length curve. No hand-set gears anywhere (ARCH_V5 §3); the
                  FSQ lattice distributes over whatever the data does.
          Bit budget is deliberately IDENTICAL to P1's: (n_path + 1) x 13.93 bit per
          2 s window = 62.7 bit/s at the defaults, printed every run.

  COND    the decoder always sees the CURRENT state and a TARGET state at a random
          future frame — a two-point boundary problem. This is the structural fix
          for the 93.9% copy degeneracy frame-level AR showed: with the endpoint
          pinned, "repeat the last frame" cannot reach the target, so the plain
          solution stops being a solution.

  LOSS    MAIN = cumulative arc-length curve matching (path integral, ARCH_V5 §4):
          accumulate consecutive-frame SE3 geodesic increments into a
          "distance travelled vs time" curve and match it. Endpoint = path length
          (went the right way), shape = timing law (at the right speed), and the
          shortest-path property makes smoothness fall out instead of being
          regularized in. Auxiliaries: per-frame key-joint SE3, multi-resolution
          temporal loss at scales 1/2/4/8 (audio's multi-resolution STFT loss ported
          to motion), the velocity loss that was P1b's ONLY validated positive, and
          InfoNCE across embodiment streams. Per-frame cross-entropy is NOT the main
          loss anywhere except in the control arm — Mathieu 2015 and every audio
          codec ablation say its optimum is the temporal average.
          READ arm_config()'s docstring before quoting the weights: the spec asks for
          the per-frame term to be low-weight, and the measurement says it cannot be,
          because one scalar of arc length per frame does not determine WHERE the
          body went. The numbers behind that are recorded there.

ARMS
  pathint          full recipe. Time-scale augmentation ON: the same raw path is
                   read out at a different frame count AND under a non-uniform
                   monotone time warp, with two paired losses — the path features
                   must MATCH across views, the time features must SEPARATE in
                   proportion to |log2 speed ratio|. That pair is the direct
                   supervision for path/time separation (ARCH_V5 §5).
  pathint_noaug    identical model and identical losses, augmentation removed
                   entirely (every window is read out at exactly T frames, no warp,
                   no paired losses). Note the consequence honestly: with no time
                   variation in the data the time codebook has NOTHING to learn
                   from, so this arm is expected to collapse its time token. That is
                   the ablation's finding, not a bug.
  baseline_frameCE the P1-style control: per-frame CROSS-ENTROPY over binned
                   key-joint state, and nothing else — no path integral, no
                   multi-resolution, no velocity, no augmentation, no InfoNCE. Same
                   architecture, same bits. It is here so the verdict table has an
                   honest floor rather than a rhetorical one.

Typical: ~20-40 min per arm on the GPU at the defaults; the projected wall time is
printed at step 200 so a wrong setting is visible immediately.
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                        # noqa: E402
from model import FSQ                                        # noqa: E402
from temporal_common import multires_dyn_loss, XBlock, fourier_time  # noqa: E402

ARMS = ("pathint", "pathint_noaug", "baseline_frameCE")
CE_LO, CE_HI = -1.25, 1.25          # binning range for the frameCE control


def orthonormalize(s):
    """Project the rotation half of a key-joint state onto SO(3) and re-emit as 6D.

    THIS IS LOAD-BEARING, not cosmetic. Measured 2026-08-06 on the first V5 sanity
    run: with a raw (unconstrained) 6D head the model drove the arc-length loss to
    0.025 — a near-perfect path integral — while its output was STATIC (temporal std
    0.0009 vs GT 0.092). Gram-Schmidt normalizes, so a near-degenerate 6D vector
    wobbling by 1e-3 normalizes into wildly different rotation matrices: the model
    had found a way to spend arc length without moving. Emitting an already
    orthonormal 6D closes that hole and puts the per-frame, velocity and
    multi-resolution losses on exactly the same quantity the arc length is measured
    on. It also makes the endpoint conditioning comparable to the output at all.
    """
    from common import rot6d_to_matrix as _r2m, matrix_to_rot6d as _m2r
    return torch.cat([_m2r(_r2m(s[..., :6])), s[..., 6:9]], -1)


# --------------------------------------------------------------------------- model
class V5Tokenizer(nn.Module):
    def __init__(self, nk=6, T=60, d_model=256, n_heads=8, enc_layers=5,
                 readout_layers=2, dec_layers=3, n_path=8, n_time=1,
                 fsq_levels=(5, 5, 5, 5, 5, 5), n_freq=8, head="reg", ce_bins=17,
                 curve_pts=8, interp="nearest", cond_drop=False,
                 token_only_head=False, ema_target=False, reg_ar=False,
                 n_bodies=0):
        super().__init__()
        assert head in ("reg", "ce")
        assert interp in ("nearest", "linear")
        self.interp = interp
        self.cond_drop = cond_drop
        self.ema_target = ema_target
        self.reg_ar = reg_ar
        self.nk, self.T, self.head_kind, self.ce_bins = nk, T, head, ce_bins
        self.n_path, self.n_time, self.curve_pts = n_path, n_time, curve_pts
        self.n_reg = n_path + n_time
        d, S = d_model, nk * 9
        self.state_dim = S
        self.inp = nn.Linear(S, d)
        self.enc_time = nn.Linear(2 * n_freq + 1, d)
        self.n_freq = n_freq
        # conditioning: current state, target state (+ where in the window it sits)
        self.cond_in = nn.Linear(S, d)
        self.cond_role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.cond_when = nn.Linear(2 * n_freq + 1, d)
        if cond_drop:
            # FIX D (V5b_cdrop): learned null embeddings that replace the current /
            # target conditioning tokens when they are dropped. Created only when the
            # arm uses condition dropout so older checkpoints still load strict.
            self.cond_null = nn.Parameter(torch.randn(2, d) * 0.02)
        if token_only_head:
            # V5d option A (--token_only_head): Delta-JEPA auxiliary demand. This
            # head reads ONLY the quantized path registers — the endpoint
            # conditioning is structurally invisible to it — and predicts the
            # canonical window's per-segment |ds| profile (v5_common.se3_step).
            # CFG guidance measured the decoder's marginal token use at ~1 cm, so
            # inference-side amplification is futile; this creates TRAINING-TIME
            # demand for token content that the main decoder cannot shortcut away.
            self.tok_head = nn.Sequential(
                nn.LayerNorm(n_path * d), nn.Linear(n_path * d, 4 * d), nn.GELU(),
                nn.Linear(4 * d, T - 1))
        if ema_target:
            # V5d option B (--ema_target, BYOL/I-JEPA): the TARGET state
            # conditioning is embedded by a slow EMA copy of the pathway
            # (cond_in / cond_when / cond_role), no gradients through it —
            # severing the encoder-predictor collusion channel. Modules exist
            # only when the flag is on so older checkpoints still load strict.
            import copy
            self.tgt_ema = nn.ModuleDict({"inp": copy.deepcopy(self.cond_in),
                                          "when": copy.deepcopy(self.cond_when)})
            for p in self.tgt_ema.parameters():
                p.requires_grad_(False)
            self.register_buffer("tgt_ema_role", self.cond_role.detach().clone())
        if reg_ar:
            # N3a (--reg_ar, lloyd's insight #2): REGISTER-LEVEL AUTOREGRESSIVE
            # production. The parallel Perceiver readout factorizes the joint code
            # distribution into per-register independent marginals (XBlock queries
            # never see each other), so each register regresses to its conditional
            # mean; chaining restores joint coherence. Register k's readout query
            # cross-attends [window features ; the QUANTIZED values of registers
            # < k], embedded by ar_in + a per-slot position embedding. The chain is
            # its own teacher (the prefix a register conditions on IS the produced
            # prefix, train and inference identical — 8 tiny readout steps).
            # Modules exist only under the flag so old checkpoints load strict.
            f = len(fsq_levels)
            self.ar_in = nn.Linear(f, d)
            self.ar_pos = nn.Parameter(torch.randn(n_path, d) * 0.02)
        if n_bodies:
            # N3b (--xbody_loss, insight #3b): per-body TARGET embedding on the
            # decoder side — one extra KV token telling the decoder WHICH body's
            # key joints to emit, so cross-embodiment supervision has somewhere to
            # put body identity other than the temporal tokens. Index 0 is the
            # primary (human) stream; decode(body=None) uses it, which keeps every
            # eval harness measuring the human stream unchanged.
            self.body_emb = nn.Embedding(n_bodies, d)
            self.body_reg_id = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                           norm_first=True, activation="gelu",
                                           dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, enc_layers)
        self.path_q = nn.Parameter(torch.randn(n_path, d) * 0.02)      # ORDERED (OAT)
        self.readout = nn.ModuleList([XBlock(d, n_heads) for _ in range(readout_layers)])
        self.fsq_path = FSQ(d, fsq_levels)
        self.fsq_time = FSQ(d, fsq_levels)
        # TIME head: its ONLY input is the window's own path integral (ARCH_V5 §3/§4)
        self.time_mlp = nn.Sequential(nn.Linear(curve_pts + 3, 4 * d), nn.GELU(),
                                      nn.Linear(4 * d, d))
        self.time_q = nn.Parameter(torch.randn(n_time, d) * 0.02)
        # decoder: per-frame queries cross-attend the (masked) registers + conditions
        self.reg_id = nn.Parameter(torch.randn(self.n_reg + 2, d) * 0.02)
        self.dec_time = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(),
                                      nn.Linear(d, d))
        self.dec = nn.ModuleList([XBlock(d, n_heads) for _ in range(dec_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d),
                                 nn.Linear(d, S * (ce_bins if head == "ce" else 1)))
        if head == "reg":
            # Start at ZERO DEVIATION, i.e. exactly the straight line between the two
            # boundary states. Without a sane start the first thousand steps emit
            # near-random 6D vectors whose arc length is orders of magnitude above GT
            # and the path integral — the main loss — begins four digits high and
            # drowns every auxiliary. Starting on the line also makes the loss curve
            # readable: it measures how much better than "go straight there" the
            # tokens have made the model.
            with torch.no_grad():
                self.out[-1].weight.mul_(0.01)
                self.out[-1].bias.zero_()
        self.budgets = V.prefix_budgets(n_path)

    # ------------------------------------------------------------------ features
    def _tfeat(self, B, T, device, dtype):
        t = torch.linspace(0.0, 1.0, T, device=device)
        return fourier_time(t, self.n_freq).to(dtype)[None].expand(B, -1, -1)

    def _canon(self, w):
        """Resample the window to the canonical T frames. The path readout only ever
        sees this, so the raw frame count — i.e. the SPEED — is structurally invisible
        to the path codebook.

        interp='nearest' (V5): round to the nearest frame — content is not smoothed,
        but stretching DUPLICATES frames, so two readouts of the same path at
        different lengths canonicalize to genuinely different 60-frame sequences.
        interp='linear' (V5b FIX A): fractional-time lerp + re-orthonormalization,
        matching the interpolated data-side warp; without this the model itself
        would reintroduce the staircase the augmentation just removed.
        """
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

    def _cond_tokens(self, cur, tgt, k_norm, drop_cur=None, drop_tgt=None):
        """cur/tgt [B,nk,9], k_norm [B] in [0,1] -> [B,2,d].

        drop_cur / drop_tgt [B] bool (FIX D): that endpoint's token is replaced by a
        learned null embedding — the decoder must get the motion from the PATH
        registers instead of interpolating conditioning it was handed for free.
        """
        B = cur.shape[0]
        a = self.cond_in(cur.reshape(B, -1)) + self.cond_role[0]
        if self.ema_target:
            # --ema_target: target token from the slow network, gradient-free.
            with torch.no_grad():
                b = self.tgt_ema["inp"](tgt.reshape(B, -1)) + self.tgt_ema_role[1]
                b = b + self.tgt_ema["when"](
                    fourier_time(k_norm, self.n_freq).to(b.dtype))
        else:
            b = self.cond_in(tgt.reshape(B, -1)) + self.cond_role[1]
            b = b + self.cond_when(fourier_time(k_norm, self.n_freq).to(b.dtype))
        if drop_tgt is not None and bool(drop_tgt.any()):
            b = torch.where(drop_tgt[:, None], self.cond_null[1][None].to(b.dtype), b)
        if drop_cur is not None and bool(drop_cur.any()):
            a = torch.where(drop_cur[:, None], self.cond_null[0][None].to(a.dtype), a)
        return torch.stack([a, b], 1)

    @torch.no_grad()
    def ema_update(self, m=0.996):
        """--ema_target: slow-network update, once per optimizer step."""
        for pe, p in zip(self.tgt_ema["inp"].parameters(), self.cond_in.parameters()):
            pe.mul_(m).add_(p.detach(), alpha=1 - m)
        for pe, p in zip(self.tgt_ema["when"].parameters(),
                         self.cond_when.parameters()):
            pe.mul_(m).add_(p.detach(), alpha=1 - m)
        self.tgt_ema_role.mul_(m).add_(self.cond_role.detach(), alpha=1 - m)

    def path_features(self, w, cond):
        """-> [B,n_path,f] pre-quantization PATH features (canonical time only)."""
        wc = self._canon(w)
        B, T = wc.shape[0], wc.shape[1]
        h = self.inp(wc.reshape(B, T, -1))
        h = h + self.enc_time(self._tfeat(B, T, wc.device, h.dtype))
        h = self.enc(torch.cat([h, cond], 1))
        if getattr(self, "reg_ar", False):
            # --reg_ar: sequential register production. Quantization inside the
            # loop is the SAME deterministic straight-through rounding encode()
            # re-applies afterwards, so the values the chain conditioned on are
            # exactly the values the decoder later reads.
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
        """The path integral itself, as the TIME head's only input.

        [total arc length (log1p), frames spent (log2 ratio to T), mean speed] +
        the normalized cumulative arc-length curve resampled to `curve_pts` points.
        The curve's SHAPE is the timing law; its length and frame count are the rate.
        Nothing here is a hand-set gear — the FSQ lattice is what turns it discrete.
        """
        c = V.arclen_curve(w)                                   # [B,T]
        T = c.shape[1]
        end = c[:, -1:].clamp_min(1e-4)
        i = torch.round(torch.linspace(0, T - 1, self.curve_pts,
                                       device=w.device)).long()
        shape = (c / end)[:, i]                                 # [B,curve_pts]
        rate = torch.stack([torch.log1p(end[:, 0]),
                            torch.full_like(end[:, 0], math.log2(T / self.T)),
                            end[:, 0] / T], -1)
        return torch.cat([shape, rate], -1)

    def time_features(self, w):
        """-> [B,n_time,f] pre-quantization TIME features."""
        h = self.time_mlp(self.curve_summary(w).to(self.time_q.dtype))
        return self.fsq_time.features(h[:, None] + self.time_q[None])

    # ------------------------------------------------------------------ encode
    def encode(self, w, cur, tgt, k_norm, drop_cur=None, drop_tgt=None):
        """-> dict with straight-through FSQ values and decoder-ready embeddings."""
        cond = self._cond_tokens(cur, tgt, k_norm, drop_cur, drop_tgt)
        fp = self.path_features(w, cond)
        ft = self.time_features(w)
        qp = self.fsq_path.quantize(fp)
        qt = self.fsq_time.quantize(ft)
        return {"path_f": fp, "time_f": ft, "path_q": qp, "time_q": qt,
                "path_z": self.fsq_path.proj_out(qp),
                "time_z": self.fsq_time.proj_out(qt), "cond": cond}

    def ints(self, q, fsq):
        half = (fsq._levels - 1) / 2
        return torch.round(q * half + torch.ceil(half)).long()

    def token_ints(self, e):
        return (self.ints(e["path_q"], self.fsq_path),
                self.ints(e["time_q"], self.fsq_time))

    # ------------------------------------------------------------------ decode
    def baseline(self, cur, tgt, k_norm, T_out, drop_tgt=None):
        """The STRAIGHT LINE between the two boundary states, [B,T_out,nk,9].

        ARCH_V5 §4's own argument — "给定端点最短路径 = 平滑性天然满足" — says the
        shortest path between fixed endpoints is the right prior. So make it the
        architecture instead of hoping the decoder rediscovers it: the network
        predicts the DEVIATION from this line and nothing else.

        Measured reason this is not optional (2026-08-06, 5k-step sanity run without
        it): the decoder never learned to honour its own conditioning — frame 0 sat
        0.67 away from the given current state on a state whose norm is 1.5, i.e. the
        two-point boundary problem that is supposed to kill the copy degeneracy was
        not actually being solved. With the baseline the endpoints are exact by
        construction and every token bit goes to the part that is actually uncertain.

        Rotations are lerped in 6D and re-orthonormalized (a cheap, monotone stand-in
        for slerp); after the target frame the line holds at the target.
        """
        B = cur.shape[0]
        t = torch.linspace(0.0, 1.0, T_out, device=cur.device, dtype=cur.dtype)
        u = (t[None, :] / k_norm[:, None].clamp_min(1e-3)).clamp(0.0, 1.0)
        if drop_tgt is not None and bool(drop_tgt.any()):
            # FIX D: the endpoint shortcut is NOT just the cond tokens — the output is
            # baseline + deviation, so the straight line itself hands the decoder the
            # interpolation for free. When the target is dropped the baseline HOLDS AT
            # THE CURRENT STATE: every frame of actual motion has to come from the
            # path registers. (cur still leaks through the held pose; dropping it too
            # only nulls its cond token — some anchor pose is unavoidable here.)
            u = u * (~drop_tgt)[:, None].to(u.dtype)
        u = u[..., None, None]
        return orthonormalize((1 - u) * cur[:, None] + u * tgt[:, None])

    def decode(self, path_z, time_z, cur, tgt, k_norm, T_out=None, hidden=None,
               drop_cur=None, drop_tgt=None, body=None):
        """path_z [B,n_path,d], time_z [B,n_time,d] + the two boundary states.

        head='reg' -> states [B,T_out,nk,9]
        head='ce'  -> logits [B,T_out,nk*9,ce_bins] over the RESIDUAL from the
                      baseline (so the control arm sits on the identical
                      architecture and differs only in its loss)
        `hidden` [B,n_path] True = that PATH register is invisible (OAT prefix). The
        time register and the two conditioning tokens are NEVER hidden: the time seat
        is reserved, and the endpoints are the boundary condition, not payload.
        """
        B = path_z.shape[0]
        T_out = T_out or self.T
        cond = self._cond_tokens(cur, tgt, k_norm, drop_cur, drop_tgt)
        kv = torch.cat([path_z, time_z, cond], 1) + self.reg_id[None]
        if getattr(self, "body_emb", None) is not None:
            # --xbody_loss: target-body token. body=None -> index 0, the primary
            # (human) stream, so unmodified eval harnesses stay valid on it.
            if body is None:
                body = torch.zeros(B, dtype=torch.long, device=path_z.device)
            kv = torch.cat([kv, (self.body_emb(body) + self.body_reg_id)[:, None]], 1)
        pad = None
        if hidden is not None:
            pad = torch.zeros(B, kv.shape[1], dtype=torch.bool, device=kv.device)
            pad[:, :self.n_path] = hidden
        q = self.dec_time(self._tfeat(B, T_out, path_z.device, path_z.dtype))
        for b in self.dec:
            q = b(q, kv, key_padding_mask=pad)
        o = self.out(q)
        if self.head_kind == "ce":
            return o.reshape(B, T_out, self.state_dim, self.ce_bins)
        base = self.baseline(cur, tgt, k_norm, T_out, drop_tgt)
        return orthonormalize(base + o.reshape(B, T_out, self.nk, 9))

    def forward(self, w, cur, tgt, k_norm, hidden=None, T_out=None,
                drop_cur=None, drop_tgt=None, time_detach=None, body=None):
        e = self.encode(w, cur, tgt, k_norm, drop_cur, drop_tgt)
        tz = e["time_z"]
        if time_detach is not None and bool(time_detach.any()):
            # V5c (--cdrop_time_detach): DECOUPLE the time head from the cdrop
            # dynamics. When the target conditioning is dropped, the baseline holds
            # at `cur` and the decoder must produce ALL motion from the registers —
            # a gradient regime the time codebook was never meant to train under
            # (V5b_cdrop's time-gear axis flipped, rho -0.72). Per-sample detach:
            # the decoder still READS the time register on dropped samples, but the
            # time head only ever LEARNS from the ordinary (conditioned) regime.
            tz = torch.where(time_detach[:, None, None], tz.detach(), tz)
        return self.decode(e["path_z"], tz, cur, tgt, k_norm,
                           T_out or w.shape[1], hidden, drop_cur, drop_tgt,
                           body=body), e


# --------------------------------------------------------------------------- ce head
def ce_bin(x, L):
    """continuous state -> integer bin in [0,L)."""
    u = (x - CE_LO) / (CE_HI - CE_LO)
    return (u * (L - 1)).round().clamp(0, L - 1).long()


def ce_value(logits):
    """logits [...,L] -> softmax-expected continuous value (P1b's readout)."""
    L = logits.shape[-1]
    lv = torch.arange(L, device=logits.device, dtype=torch.float32)
    return (logits.float().softmax(-1) * lv).sum(-1) / (L - 1) * (CE_HI - CE_LO) + CE_LO


def to_states(out, model, cur=None, tgt=None, k_norm=None):
    """model output -> continuous key states [B,T,nk,9], whichever head is in use.
    The ce head predicts the RESIDUAL from the straight-line baseline, so it needs
    the boundary states back to become a trajectory."""
    if model.head_kind != "ce":
        return out
    B, T = out.shape[0], out.shape[1]
    r = ce_value(out).reshape(B, T, model.nk, 9)
    base = model.baseline(cur, tgt, k_norm, T)
    return orthonormalize(base + r)


# --------------------------------------------------------------------------- run io
def build_v5(cfg):
    return V5Tokenizer(
        nk=cfg["nk"], T=cfg.get("T", V.T_WIN), d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8), enc_layers=cfg.get("enc_layers", 5),
        readout_layers=cfg.get("readout_layers", 2), dec_layers=cfg.get("dec_layers", 3),
        n_path=cfg.get("n_path", 8), n_time=cfg.get("n_time", 1),
        fsq_levels=tuple(int(x) for x in str(cfg.get("fsq_levels", "5,5,5,5,5,5")).split(",")),
        head=cfg.get("head", "reg"), ce_bins=cfg.get("ce_bins", 17),
        interp=cfg.get("aug_interp", "nearest"),
        cond_drop=bool(float(cfg.get("cond_drop_tgt", 0) or 0) > 0
                       or float(cfg.get("cond_drop_both", 0) or 0) > 0),
        token_only_head=bool(cfg.get("token_only_head", False)),
        ema_target=bool(cfg.get("ema_target", False)),
        reg_ar=bool(cfg.get("reg_ar", False)),
        n_bodies=len(cfg.get("xbody_bodies") or [])
        if float(cfg.get("xbody_loss", 0) or 0) > 0 else 0)


def load_run(run, device="cpu"):
    """Loads a V5 run, or transparently a V6 one (config.json's arch == 'v6').

    The dispatch lives here, not in the harnesses: eval_v5.py and
    n3_eval_temporal.py both `from train_v5 import load_run`, and V6Tokenizer's
    public surface is interface-compatible with V5Tokenizer's on purpose, so one
    branch here makes every existing temporal measurement work on V6 unchanged.
    The import is lazy because train_v6 imports orthonormalize/to_states from here.
    """
    cfg = json.load(open(f"{run}/config.json"))
    arch = str(cfg.get("arch", "v5")).lower()
    if arch == "v6":
        from train_v6 import load_run as _lr6
        return _lr6(run, device)
    if arch == "v7":
        from train_v7 import load_run as _lr7
        return _lr7(run, device)
    m = build_v5(cfg).to(device)
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location=device))
    m.eval()
    return m, cfg


# --------------------------------------------------------------------------- training
def sample_hidden(B, n_path, budgets, device, rng):
    n = torch.as_tensor(budgets, device=device)[
        torch.as_tensor(rng.integers(0, len(budgets), B), device=device)]
    return torch.arange(n_path, device=device)[None] >= n[:, None]


def arm_config(arm):
    """One place where the three arms differ. Everything else is identical.

    HONEST NOTE ON THE WEIGHTS. ARCH_V5 §4 calls the per-frame SE3 term an auxiliary
    "权重低". Measured 2026-08-06 at 8k steps, w_frame in {0.25, 2, 6} with everything
    else fixed, against the straight-line-between-endpoints baseline (9.24 cm / 0.587
    energy), key-joint error and motion-energy retention:
        w_frame 0.25 -> 11.6 cm / 0.54   (LOSES to interpolating the endpoints)
        w_frame 2.0  ->  9.9 cm / 0.63
        w_frame 6.0  ->  8.7 cm / 0.66   (beats it on error, energy and path integral)
    The reason is structural, not a tuning accident: the arc-length curve is ONE
    scalar per frame, so it constrains how FAR the body has travelled but not WHERE it
    went. The first V5 run exploited exactly that. The path integral stays the main
    loss — it is what makes the timing law right and what the time codebook reads —
    but the per-frame anchor has to carry real weight or the path tokens end up
    carrying nothing at all. Every weight is overridable from the CLI and the value
    that actually ran is recorded in config.json.
    """
    base = dict(w_pathint=1.0, w_frame=6.0, w_multires=2.0, w_vel=0.5, w_nce=0.1,
                w_inv=1.0, w_sep=0.5, aug=True, head="reg")
    if arm == "pathint":
        return base
    if arm == "pathint_noaug":
        return base | dict(aug=False, w_inv=0.0, w_sep=0.0)
    if arm == "baseline_frameCE":
        return dict(w_pathint=0.0, w_frame=0.0, w_multires=0.0, w_vel=0.0, w_nce=0.0,
                    w_inv=0.0, w_sep=0.0, aug=False, head="ce")
    raise SystemExit(f"unknown arm {arm}")


@torch.no_grad()
def evaluate(model, kd, embs, T, span, dev, n=192, seed=7):
    model.eval()
    g = np.random.default_rng(seed)
    starts, _ = kd.sample_starts(n, g, "val", span=span)
    out = {}
    for emb in embs[:1]:
        w = kd.window(emb, starts, T, span).to(dev)
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(w.shape[0], device=dev)
        for b in model.budgets:
            hid = torch.arange(model.n_path, device=dev)[None] >= b
            hid = hid.expand(w.shape[0], -1)
            o, _ = model(w, cur, tgt, kn, hid)
            p = to_states(o, model, cur, tgt, kn).float()
            l, ep = V.pathint_loss(p, w.float())
            pe, re = V.key_se3_err(p, w.float(), kd.reach[emb])
            # motion energy in the SAME currency P1 reported: temporal std ratio
            en = float(p.std(1).mean() / w.float().std(1).mean())
            out[f"n{b}"] = {"pathint": round(float(l), 4),
                            "endpoint_rel_err": round(float(ep), 4),
                            "key_pos_cm": round(pe, 3), "key_rot_deg": round(re, 2),
                            "energy_ratio": round(en, 3)}
        out["static_anchor"] = round(V.static_pathint_anchor(w.float()), 4)
        bl = model.baseline(cur, tgt, torch.ones(w.shape[0], device=dev), T).float()
        out["straight_line_baseline"] = {
            "pathint": round(float(V.pathint_loss(bl, w.float())[0]), 4),
            "key_pos_cm": round(V.key_se3_err(bl, w.float(), kd.reach[emb])[0], 3),
            "energy_ratio": round(float(bl.std(1).mean() / w.float().std(1).mean()), 3)}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
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
    ap.add_argument("--ce_bins", type=int, default=17)
    ap.add_argument("--steps", type=int, default=60000,
                    help="60k at the measured 44 steps/s = ~23 min for the pathint "
                         "arm and less for the other two; all three run at MATCHED "
                         "steps so the ablation is not confounded by budget")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=2500)
    # ---- V5b fixes, all gated; the defaults reproduce V5 exactly ----------------
    ap.add_argument("--aug_interp", choices=["nearest", "linear"], default="nearest",
                    help="FIX A: 'linear' = continuous fractional-time resampling of "
                         "the speed views (lerp positions, lerp+re-orthonormalize the "
                         "6D rotations) instead of nearest-frame duplication; also "
                         "switches the model's canonical resample to match")
    ap.add_argument("--timing_loss", choices=["cumulative", "riemann", "both"],
                    default="cumulative",
                    help="FIX B: 'riemann' = per-segment ds_t matching with a "
                         "deadzone (cumulative-cancellation-proof) + low-weight "
                         "total-length term; 'both' adds it on top of the "
                         "cumulative-curve loss; 'cumulative' = V5")
    ap.add_argument("--w_riemann", type=float, default=1.0)
    ap.add_argument("--riemann_deadzone", type=float, default=0.02,
                    help="measured p10 of GT |ds| (reach units), see riemann_timing_loss")
    ap.add_argument("--w_pairnce", type=float, default=0.0,
                    help="FIX C: InfoNCE where the SAME path under two different "
                         "time warps are positives for the PATH registers and other "
                         "windows in the batch are negatives — invariance through "
                         "attraction rather than by emptying the token. 0 = off (V5)")
    ap.add_argument("--pair_tau", type=float, default=0.1)
    ap.add_argument("--cond_drop_tgt", type=float, default=0.0,
                    help="FIX D: prob of dropping the TARGET conditioning (null cond "
                         "token AND baseline held at cur) — starves the endpoint "
                         "interpolation shortcut so token gradients reach the decoder")
    ap.add_argument("--cond_drop_both", type=float, default=0.0,
                    help="FIX D: prob of dropping BOTH endpoints' cond tokens "
                         "(baseline still holds at cur; see baseline())")
    ap.add_argument("--cdrop_time_detach", action="store_true",
                    help="V5c: on cond-dropped samples the decoder reads the time "
                         "register DETACHED, so the time codebook never trains under "
                         "the held-at-cur gradient regime that flipped V5b_cdrop's "
                         "gear axis. Off = bit-exact V5b behavior.")
    ap.add_argument("--token_only_head", action="store_true",
                    help="V5d option A (Delta-JEPA aux): head reading ONLY the path "
                         "registers (endpoints structurally invisible) predicts the "
                         "canonical per-segment |ds| profile — training-time demand "
                         "for token content. Off = bit-exact V5c behavior.")
    ap.add_argument("--w_tokhead", type=float, default=0.5)
    ap.add_argument("--ema_target", action="store_true",
                    help="V5d option B (BYOL/I-JEPA): target conditioning embedded "
                         "by a slow EMA copy, no gradients — severs the encoder-"
                         "predictor collusion channel. Second line after the token "
                         "head. Off = bit-exact.")
    ap.add_argument("--ema_momentum", type=float, default=0.996)
    # ---- N3 arms (2026-08-07), all gated; defaults reproduce V5c bit-exact -------
    ap.add_argument("--reg_ar", action="store_true",
                    help="N3a (insight #2): produce the 8 ordered path registers "
                         "SEQUENTIALLY — register k's quantized value conditions "
                         "registers k+1.. — instead of the parallel readout whose "
                         "per-register marginals each regress to their conditional "
                         "mean. OAT prefix budgets unchanged (prefix-n = first n of "
                         "the chain). Off = bit-exact prior behavior.")
    ap.add_argument("--xbody_loss", type=float, default=0.0,
                    help="N3b (insight #3b): weight W>0 enables cross-embodiment "
                         "decode supervision — with prob --xbody_prob per step the "
                         "reconstruction losses are charged (scaled by W) against a "
                         "PAIRED TRAINING ROBOT's GT key joints (its endpoints as "
                         "boundary condition + a per-body decoder-side target "
                         "embedding) instead of the source stream's, forcing the "
                         "temporal tokens to carry only body-invariant content. "
                         "0 = off = bit-exact.")
    ap.add_argument("--xbody_prob", type=float, default=0.5)
    ap.add_argument("--xbody_embs", default="unitree_h1,fourier_gr3,tienkung",
                    help="the paired TRAINING bodies used as xbody decode targets "
                         "(G1 stays holdout)")
    # loss-weight overrides. The arm sets the recipe; these exist so a weight can be
    # retuned without editing arm_config, and so the config.json records what actually
    # ran rather than what the arm nominally implies.
    for _w in ("pathint", "frame", "multires", "vel", "nce", "inv", "sep"):
        ap.add_argument(f"--w_{_w}", type=float, default=None)
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
    A = arm_config(args.arm)
    for k in list(A):
        if k.startswith("w_") and getattr(args, k, None) is not None:
            A[k] = float(getattr(args, k))

    embs = [e for e in args.embs.split(",") if e]
    xbody_embs = [e for e in args.xbody_embs.split(",") if e] \
        if args.xbody_loss > 0 else []
    # KeyData loads the union; the TRAINING streams stay exactly `embs` (emb0
    # sampling, InfoNCE), xbody bodies are decode-supervision targets only.
    kd = V.KeyData(args.cache, embs + [e for e in xbody_embs if e not in embs],
                   device=dev, max_clips=200 if args.tiny else 0,
                   root_vel=args.root_vel)
    nk = kd.key[embs[0]].shape[1]      # 6, or 7 with --root_vel (vel channel last)
    span = args.T                       # raw frames a training window covers (2 s)

    xbody_bodies = list(dict.fromkeys(embs + xbody_embs)) if xbody_embs else []
    body_index = {n: i for i, n in enumerate(xbody_bodies)}
    cfg = vars(args) | A | {"nk": nk, "span": span, "key_joints": kd.knames[embs[0]],
                            "cache": args.cache, "frozen_spatial": V.FROZEN,
                            "xbody_bodies": xbody_bodies}
    model = build_v5(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    lv = [int(x) for x in args.fsq_levels.split(",")]
    bits = V.bit_budget(args.n_path, args.n_time, lv, args.T)
    cfg |= {"n_params": n_par, "bit_budget": bits, "budgets": list(model.budgets)}
    print(f"=== V5 arm={args.arm} head={A['head']} ===")
    print(f"  key joints ({nk}) = {kd.knames[embs[0]]}  -> state {nk*9} dims/frame "
          f"(P1 compressed 3328)")
    print(f"  params {n_par/1e6:.2f}M  path registers {args.n_path} (OAT budgets "
          f"{model.budgets})  time registers {args.n_time}  "
          f"vocab/register {bits['vocab_per_register']}")
    print("  BIT BUDGET " + json.dumps(bits))
    print(f"  losses " + json.dumps({k: v for k, v in A.items() if k.startswith('w_')}) +
          f"  aug={A['aug']}")
    print(f"  V5b: aug_interp={args.aug_interp} timing_loss={args.timing_loss} "
          f"w_riemann={args.w_riemann} deadzone={args.riemann_deadzone} "
          f"w_pairnce={args.w_pairnce} pair_tau={args.pair_tau} "
          f"cond_drop_tgt={args.cond_drop_tgt} cond_drop_both={args.cond_drop_both}")
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
        # ---- view A: the reconstruction target ------------------------------
        T_a = args.T
        if A["aug"] and rng.random() < 0.5:
            T_a = int(np.clip(round(args.T * 2.0 ** rng.uniform(-1, 1)),
                              args.T // 2, 2 * args.T))
        # FIX A: 'linear' reads the span at CONTINUOUS fractional times (interpolated),
        # so a stretched view is the same content, not a frame-duplicating staircase
        frac = (args.aug_interp == "linear") and (T_a != span)
        idx_a = V.uniform_fidx(span, T_a) if frac else V.uniform_idx(span, T_a)
        emb0 = embs[int(rng.integers(len(embs)))]
        wa = (kd.window_fidx(emb0, starts, idx_a, span) if frac
              else kd.window_idx(emb0, starts, idx_a)).to(dev)
        # target state = a RANDOM future frame -> two-point boundary problem
        k = int(rng.integers(T_a // 2, T_a))
        cur, tgt = wa[:, 0], wa[:, k]
        kn = torch.full((args.bs,), k / max(T_a - 1, 1), device=dev)
        hid = sample_hidden(args.bs, model.n_path, model.budgets, dev, rng)
        dc = dt = None
        if model.cond_drop and A["head"] == "reg":
            # FIX D: per-sample condition dropout. drop_tgt covers BOTH events (the
            # target is gone in either), drop_cur only the drop-both event.
            r_ = torch.rand(args.bs, device=dev)
            dt = r_ < (args.cond_drop_tgt + args.cond_drop_both)
            dc = r_ < args.cond_drop_both

        td = dt if (args.cdrop_time_detach and dt is not None) else None
        # ---- N3b xbody step: same tokens, PAIRED ROBOT supervision --------------
        xb = None
        if args.xbody_loss > 0 and A["head"] == "reg":
            cands = [b for b in xbody_embs if b != emb0]
            if cands and rng.random() < args.xbody_prob:
                xb = cands[int(rng.integers(len(cands)))]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            body = None
            if getattr(model, "body_emb", None) is not None:
                body = torch.full((args.bs,), body_index[xb or emb0],
                                  dtype=torch.long, device=dev)
            if xb is None:
                o, e = model(wa, cur, tgt, kn, hid, drop_cur=dc, drop_tgt=dt,
                             time_detach=td, body=body)
                gt_w, w_x = wa, 1.0
            else:
                # encode the SOURCE stream (tokens + its own cond), decode with the
                # PAIRED robot's boundary states + body embedding, charge the
                # reconstruction losses against the robot's GT key joints. The
                # cache rows are body-paired, so (starts, idx_a) is the same motion.
                wx = (kd.window_fidx(xb, starts, idx_a, span) if frac
                      else kd.window_idx(xb, starts, idx_a)).to(dev)
                e = model.encode(wa, cur, tgt, kn, dc, dt)
                tz = e["time_z"]
                if td is not None and bool(td.any()):
                    tz = torch.where(td[:, None, None], tz.detach(), tz)
                o = model.decode(e["path_z"], tz, wx[:, 0], wx[:, k], kn, T_a, hid,
                                 dc, dt, body=body)
                gt_w, w_x = wx, float(args.xbody_loss)
            parts = {}
            if xb is not None:
                parts["xbody"] = 1.0
            if A["head"] == "ce":
                L = model.ce_bins
                base = model.baseline(cur, tgt, kn, T_a)
                tb = ce_bin((wa - base).reshape(args.bs, T_a, -1), L)
                loss = F.cross_entropy(o.float().reshape(-1, L), tb.reshape(-1))
                parts["ce"] = float(loss.detach())
                pred = to_states(o, model, cur, tgt, kn)
            else:
                pred = o
                loss = torch.zeros((), device=dev)
            gt = gt_w.float()          # N3b xbody step: the paired robot's GT window
            pf = pred.float()
            if A["w_pathint"] > 0 and args.timing_loss in ("cumulative", "both"):
                lp, ep = V.pathint_loss(pf, gt)
                loss = loss + w_x * A["w_pathint"] * lp
                parts["pathint"] = float(lp.detach()); parts["endpoint"] = float(ep.detach())
            if (A["w_pathint"] > 0 and args.w_riemann > 0
                    and args.timing_loss in ("riemann", "both")):
                # FIX B: per-segment increments cannot cancel the way the normalized
                # cumulative curve can (fast half + slow half = same endpoint)
                lr_, epr = V.riemann_timing_loss(pf, gt, deadzone=args.riemann_deadzone)
                loss = loss + w_x * args.w_riemann * lr_
                parts["riemann"] = float(lr_.detach())
                parts["end_rel"] = float(epr.detach())
            if args.token_only_head:
                # V5d option A: the head sees ONLY the straight-through path
                # registers. Its target — the CANONICAL readout's per-segment |ds|
                # profile — is readout-speed-invariant, because view A is always a
                # uniform readout of the span; the demand therefore does not fight
                # the warp invariance the augmentation buys.
                with torch.no_grad():
                    gt_ds = V.se3_step(model._canon(wa).float())
                pd_ = model.tok_head(e["path_z"].reshape(args.bs, -1)).float()
                ltk = F.smooth_l1_loss(pd_, gt_ds, beta=0.1)
                loss = loss + args.w_tokhead * ltk
                parts["tokhead"] = float(ltk.detach())
            if A["w_frame"] > 0:
                lf = F.mse_loss(pf, gt)
                loss = loss + w_x * A["w_frame"] * lf
                parts["frame"] = float(lf.detach())
            if A["w_multires"] > 0 or A["w_vel"] > 0:
                zh = pf.reshape(args.bs, T_a, -1)
                zg = gt.reshape(args.bs, T_a, -1)
                if A["w_vel"] > 0:
                    dh, dg = zh[:, 1:] - zh[:, :-1], zg[:, 1:] - zg[:, :-1]
                    lv_ = ((dh - dg) ** 2).mean() / (dg.pow(2).mean() + 1e-6)
                    loss = loss + w_x * A["w_vel"] * lv_
                    parts["vel"] = float(lv_.detach())
                if A["w_multires"] > 0:
                    lm = multires_dyn_loss(zh, zg, (1, 2, 4, 8))
                    loss = loss + w_x * A["w_multires"] * lm
                    parts["multires"] = float(lm.detach())
            # ---- paired time-scale view: the separation supervision ----------
            if A["aug"] and (A["w_inv"] > 0 or A["w_sep"] > 0 or args.w_pairnce > 0):
                T_b = int(np.clip(round(args.T * 2.0 ** rng.uniform(-1, 1)),
                                  args.T // 2, 2 * args.T))
                if args.aug_interp == "linear":         # FIX A, warped view
                    idx_b = V.warp_fidx(span, T_b, rng)
                    wb = kd.window_fidx(emb0, starts, idx_b, span).to(dev)
                else:
                    idx_b = V.warp_idx(span, T_b, rng)
                    wb = kd.window_idx(emb0, starts, idx_b).to(dev)
                eb = model.encode(wb, cur, tgt, kn, dc, dt)
                if A["w_inv"] > 0:                      # SAME path -> same path token
                    li = F.mse_loss(eb["path_f"], e["path_f"])
                    loss = loss + A["w_inv"] * li
                    parts["path_inv"] = float(li.detach())
                if A["w_sep"] > 0:                      # different rate -> apart
                    gap = abs(math.log2(T_a / T_b))
                    d = (eb["time_f"] - e["time_f"]).pow(2).mean().sqrt()
                    ls = F.relu(0.5 * gap - d)
                    loss = loss + A["w_sep"] * ls
                    parts["time_sep"] = float(ls.detach())
                if args.w_pairnce > 0:
                    # FIX C: the SAME path under two different time warps are the
                    # POSITIVE pair for the PATH registers; the rest of the batch is
                    # the negatives. Reconstruction pressure always has the shortcut
                    # of emptying the token to buy invariance ("invariance by
                    # discarding", content share 0.91 -> 0.14); InfoNCE removes that
                    # shortcut, because an emptied token cannot single out its own
                    # positive against 127 negatives.
                    pa = F.normalize(e["path_q"].reshape(args.bs, -1), dim=-1)
                    pb = F.normalize(eb["path_q"].reshape(args.bs, -1), dim=-1)
                    lg2 = pa @ pb.t() / args.pair_tau
                    lab2 = torch.arange(args.bs, device=dev)
                    lpn = 0.5 * (F.cross_entropy(lg2, lab2) +
                                 F.cross_entropy(lg2.t(), lab2))
                    loss = loss + args.w_pairnce * lpn
                    parts["pair_nce"] = float(lpn.detach())
            # ---- InfoNCE across embodiments: same motion, other body ---------
            if A["w_nce"] > 0 and len(embs) > 1:
                e2 = embs[int(rng.integers(len(embs)))]
                if e2 != emb0:
                    w2 = (kd.window_fidx(e2, starts, idx_a, span) if frac
                          else kd.window_idx(e2, starts, idx_a)).to(dev)
                    ec = model.encode(w2, w2[:, 0], w2[:, k], kn)
                    a = F.normalize(e["path_q"].reshape(args.bs, -1), dim=-1)
                    b = F.normalize(ec["path_q"].reshape(args.bs, -1), dim=-1)
                    lg = a @ b.t() / args.tau
                    lab = torch.arange(args.bs, device=dev)
                    ln = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))
                    loss = loss + A["w_nce"] * ln
                    parts["nce"] = float(ln.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if args.ema_target:
            model.ema_update(args.ema_momentum)

        if step % 100 == 0:
            sps = 100 / max(time.time() - t_loop, 1e-9) if step else 0.0
            t_loop = time.time()
            print(f"step {step} loss {loss.item():.4f} " +
                  " ".join(f"{k} {v:.3f}" for k, v in parts.items()) +
                  f"  {sps:.1f} steps/s ({time.time()-t0:.0f}s)", flush=True)
            if step == 200:
                eta = (args.steps - step) / max(sps, 1e-6) / 60
                print(f"  PROJECTED WALL TIME: {eta:.1f} min for {args.steps} steps "
                      f"at {sps:.1f} steps/s", flush=True)
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
