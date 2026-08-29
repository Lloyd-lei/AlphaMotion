"""ARCH V7 — the decisive head-to-head between two temporal designs, OAT ELIMINATED.

    train_v7.py --arm jepa     --out RUNDIR
    train_v7.py --arm maskfsq  --out RUNDIR

=============================================================================
WHY OAT IS GONE
=============================================================================
lloyd, 2026-08-07: "不要什么oat了…oat我已经说了是一个超烂的idea…砍掉oat这个寄存器
鬼玩意儿". The data says the same thing without any interpretation needed. The OAT
prefix ladder — key-position error at 1 register vs at all 8 — has been measured on
every arm ever trained, and it is FLAT:

    V5c_0806          0.017 cm  from 1 register to 8
    N3a_regAR_0807    0.031 cm  (and that arm existed *specifically* to unflatten it)
    V6_mae_0807       0.000 cm
    V6_regar_0807     0.019 cm  (8.037 -> 8.018 cm, step 32500)

Ordered-prefix training buys 0.000-0.031 cm and costs a QUARTER of every optimizer
step, because `sample_hidden` draws uniformly from the budget ladder (1,2,4,8) and so
25% of every batch is trained in a crippled 1-register regime whose only measurable
effect is that it makes registers 2..8 redundant. V7 deletes it outright:

    * no `hidden` / prefix budgets anywhere in the training loop
    * no ordered `path_q` semantics — every latent is always present, always used
    * the ladder is still COMPUTABLE at eval time (model.budgets is kept so
      eval_v5.stat8_ladder runs unchanged) but it is now a pure DIAGNOSTIC of an
      untrained regime, not a trained product surface. Read it as "how badly does
      this model break when you delete latents it was never told to survive without".

=============================================================================
THE TWO ARMS, AND EXACTLY WHAT DIFFERS
=============================================================================
Both arms live in this one file on purpose: everything outside `V7Tokenizer.encode`
is shared code, so "controlled" is a property of the source, not of a promise.

  ARM 1 --arm jepa      "given two endpoints, imagine the path"
    context encoder : frame 0 and frame k (the two boundary states) -> 2 tokens
    target  encoder : the FULL window's frame tokens -> self-attention across frames
    readout         : 8 Perceiver queries cross-attend [target frames ; context]
                      -> 8 latents -> FSQ.  PARALLEL, UNORDERED, ALL PRESENT.
    + 1 TIME register off the window's own arc-length curve (V5's, verbatim)
    predictor       : per-frame clock queries cross-attend [latents ; context]
                      -> the intermediate trajectory, as a deviation from the
                      straight line between the endpoints
    JEPA proper     : the predictor's per-frame latent is additionally trained to
                      match the TARGET ENCODER's per-frame features under stop-grad
                      (--w_jepa). Without this term the arm is a conditional VQ-VAE
                      and the name would be a lie; it is arm-intrinsic and is
                      declared as such in the verdict.
    The latents are CONDITIONAL: the predictor already holds the endpoints, so the
    bits are not spent re-describing them. That is the point of the formulation.

  ARM 2 --arm maskfsq   "no registers, per-frame attention, the code stream IS it"
    lloyd: "没有oa没有寄存器 直接temporal attn上 per frame attn学synergy"
    encoder         : per-frame tokens, a random subset of FRAMES masked on the
                      encoder input (frame_masking.py's families), self-attention
                      across frames — the frames learn the synergy directly
    bottleneck      : NO global Perceiver registers. The canonical 60-frame grid is
                      cut into 9 CONTIGUOUS GROUPS and each group is pooled by a
                      masked-softmax query that can only see its own frames, so the
                      quantized code sequence is TIME-LOCAL and IS the representation.
    decoder         : identical to arm 1's predictor.

  Held identical across the arms (this is the controlled part):
    data, key joints, residual input channels and their measured scales, the loss
    recipe and every weight, the time-scale augmentation, the decoder and its
    conditioning surface (endpoints + straight-line baseline), the optimizer, the
    schedule, the seed, and THE BIT BUDGET.

=============================================================================
THE BIT BUDGET — ASSERTED IN BOTH ARMS
=============================================================================
62.69 bit/s. Both arms carry NINE FSQ registers on the same 6x5 grid
(6 * log2(5) = 13.932 bits each; 9 * 13.932 = 125.38 bits per 2 s window):

    jepa      8 path latents + 1 time register   = 9 registers = 62.69 bit/s
    maskfsq   9 time-local group codes + 0 time  = 9 registers = 62.69 bit/s

and `assert abs(total_bits_per_s - 62.69) < 0.01` runs in BOTH arms before the first
optimizer step, so a mismatched arm physically cannot reach the verdict table. This
also equals V5c's and V6's budget, so the V7 rows sit in the same table as theirs.

WHY 9 GROUPS AND NOT 60 (the honest caveat lloyd asked for).
A literal per-FRAME FSQ at 30 fps costs 60 * 13.932 = 836 bits per window = 418 bit/s,
which is 6.7x arm 1 and would make the comparison meaningless. Matching 125.38 bits
with 60 registers forces 2.09 bits per frame (e.g. f=1 at ~4 levels), i.e. one
near-binary scalar per frame — a degenerate representation that would lose for
reasons having nothing to do with the design being tested. Group size 60/9 is the
FAIREST matched-budget configuration: it is the smallest grouping that keeps the
register geometry (count, dims, levels) bit-identical to arm 1, so the only thing
that differs is WHERE the bits live — 8 global path latents + 1 duration gear, versus
9 latents each tied to a ~0.22 s slice of the window. Groups are 7/6/7/7/6/7/7/6/7
frames (round(linspace(0, 60, 10))). This choice is stated in the verdict.

=============================================================================
INPUT (both arms): 6D rotations PLUS residual channels
=============================================================================
frame_masking.rate_features, reused verbatim from V6 with V6's MEASURED scales
(rate_scale 16.0, acc_scale 35.0 — the absolute key state has RMS 0.507, velocity
0.0324, acceleration 0.0146 over 256 AMASS train windows, so these put all three
channels on the same footing):

  absolute   the 6 key joints' global rot6d + root-relative position, EVERY frame
  velocity   body-frame relative rotation R[t-1]^T R[t] (identity subtracted) + the
             position delta. Bone-length-free on the angular half by construction.
  accel      d/dt of the SG(5,2)-SMOOTHED velocity. R10 probe 2: raw second
             differences are 4.5-15 dB SNR, SG5 lifts them to 18-28 dB.
  jerk       ABSENT. R10 probe 2 measured raw third differences at -11..-0.3 dB at
             30 fps, and the SG9 that would rescue them is a 0.3 s window that erases
             the punch transient it was supposed to measure. Red light, obeyed.

V7 uses the --residual_keep_abs form (absolutes on EVERY frame, rates added on top)
rather than V6's I-frame/P-frame form, because the brief says "6D rotations PLUS
... residual channels". V6_mae_resid led V6_mae at every matched step, so the
residual bandwidth is earned. The rotational residual is the full body-frame relative
rotation (which contains both swing and twist); it is NOT split into explicit
swing/twist components, because the brief says to reuse V6's implementation.

=============================================================================
DATA: ALL 18 EMBODIMENTS EXCEPT THE TWO G1s
=============================================================================
lloyd: "不要只加入smpl了 所有18个具身 除了g1都给我添加进去". 16 bodies train
(human_smpl + 15 robots); unitree_g1_29dof and unitree_g1_23dof are held out for the
zero-shot row and are never sampled.
pal_talos uses the FIXED alias spec (v5_common.load_spec now prefers
`{emb}_spec_fixed.npz`; see runs/DATAFIX_talos_g123.md) — its key joints were
previously resolved entirely by the rest-pose geometric fallback because its joints
were literally called body00..body32.
--root_vel is OFF and cannot be on: only 6 of the 18 embodiments ship a
`_root_world.pt` side channel, so the locomotion channel and the 16-body requirement
are mutually exclusive. The 16-body requirement is the explicit instruction, so all
V7 motion numbers are ARTICULATED motion, root-relative — same convention as V5/V6.

=============================================================================
l-DeTok (2507.15856) — --mask_warmup_steps
=============================================================================
l-DeTok reports a grokking-like pathology SPECIFIC to 1D tokenizers trained with
masking: a delayed phase transition that leaves the tokenizer undertrained, fixed by
about 10 epochs of warm-up with masking DISABLED. Arm 2 is exactly that
configuration, so --mask_warmup_steps (default 6000 = 10% of the 60k schedule) runs
the encoder mix at {none: 1.0} before switching to the family mix. Arm 1 has no
encoder masking and the flag is inert there.
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


# --------------------------------------------------------------------------- constants
V7_BITS_PER_S = 62.69          # ASSERTED in both arms; == V5c's == V6's
# "arclen" (V8, lloyd 2026-08-07): maskfsq VERBATIM, on a different sequence axis.
#   lloyd: "不就是把per frame attn换成 per arc-path attn了吗" — exactly that, and
#   nothing else. The window is RESAMPLED onto 60 equal-PATH-FRACTION nodes (node i
#   sits where the motion has covered i/59 of its total arc length), and then the
#   whole maskfsq machinery runs unchanged on that grid: per-node attention, node
#   masking, 8 contiguous groups. A group is now a fixed slice OF THE PATH, so a
#   time warp cannot move content between groups (measured: 2.8-3.9x more stable
#   than clock slices). NO pooling trick, no Gaussian windows — the first version
#   of this arm pooled arclen-weighted windows over the CLOCK grid and lloyd
#   rejected it ("怎么还在用池化"); this one changes the grid itself.
#   * On the path grid the rate channels become TIME-FREE geometric derivatives:
#     velocity -> direction of travel per unit path, accel -> curvature.
#   * The clock is gone from the content stream by construction, so the SPEED
#     CONTROL token (jepa's arc-curve head, verbatim) becomes mandatory: it is the
#     only way the decoder learns WHEN along the clock each path fraction happens.
#   * 8 content + 1 speed = 9 tokens, budget bit-identical to both parents.
ARMS = ("jepa", "maskfsq", "arclen")

# all 18 embodiments of the paired xwbc caches; the two G1s are the zero-shot holdout
EMBS_18 = ["human_smpl", "booster_k1", "booster_t1", "booster_t1_29dof",
           "engineai_pm01", "fourier_gr3", "fourier_n1", "hightorque_hi",
           "kuavo_s45", "pal_talos", "pnd_adam_lite", "tienkung", "unitree_h1",
           "unitree_h1_2", "unitree_h2", "unitree_r1",
           "unitree_g1_29dof", "unitree_g1_23dof"]
G1_HOLDOUT = ["unitree_g1_29dof", "unitree_g1_23dof"]
EMBS_TRAIN = [e for e in EMBS_18 if e not in G1_HOLDOUT]          # 16 bodies


def group_bounds(T, G):
    """Contiguous group boundaries on the canonical grid: 60/9 -> 7,6,7,7,6,7,7,6,7."""
    b = np.round(np.linspace(0, T, G + 1)).astype(int)
    assert b[0] == 0 and b[-1] == T and (np.diff(b) > 0).all(), (T, G, b)
    return b


# --------------------------------------------------------------------------- model
class V7Tokenizer(nn.Module):
    """Both arms. Public surface is V5Tokenizer's / V6Tokenizer's, so eval_v5.py and
    n3_eval_temporal.py measure V7 with zero changes."""

    def __init__(self, arm="jepa", nk=6, T=60, d_model=256, n_heads=8,
                 enc_layers=5, ctx_layers=2, readout_layers=2, dec_layers=3,
                 n_path=8, n_time=1, n_groups=9,
                 fsq_levels=(5, 5, 5, 5, 5, 5), n_freq=8, curve_pts=8,
                 interp="linear", rate_scale=16.0, acc_scale=35.0,
                 fsq_levels_time=None, arc_alpha=1.0, pure_timing=False,
                 endpoint_clamp=False):
        super().__init__()
        assert arm in ARMS, arm
        assert interp in ("nearest", "linear")
        self.arm = arm
        self.nk, self.T, self.head_kind = nk, T, "reg"
        self.endpoint_clamp = endpoint_clamp
        self.n_path, self.n_time, self.curve_pts = n_path, n_time, curve_pts
        self.n_reg = n_path + n_time
        self.interp = interp
        self.rate_scale, self.acc_scale = rate_scale, acc_scale
        # arc_alpha: 1.0 = pure path grid (V8), 0.0 = pure clock grid (maskfsq's).
        # Measured 2026-08-07, PURE GEOMETRY, 8 interpolation nodes: equal-CLOCK
        # slices reconstruct better than equal-PATH slices on all 6 families
        # (run 5.16 vs 5.56 cm, jacks 3.65 vs 3.82) because equal-path starves the
        # slow segments; but equal-path is what survives a time warp. The optimum is
        # expected strictly between the endpoints, so it is a swept parameter, not a
        # constant.
        self.arc_alpha = float(arc_alpha)
        self.pure_timing = bool(pure_timing)
        self.n_freq = n_freq
        d, S = d_model, nk * 9
        self.state_dim = S
        self.cond_drop = False        # V7 carries no conditioning dropout, in EITHER arm

        # ------------- frame tokens: absolutes PLUS residual rate channels ----------
        self.inp = nn.Linear(S, d)                  # absolute state, every frame
        self.inp_v = nn.Linear(S, d)                # velocity
        self.inp_a = nn.Linear(S, d)                # SG5-smoothed acceleration
        self.enc_time = nn.Linear(2 * n_freq + 1, d)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)     # arm 2 only
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                           norm_first=True, activation="gelu",
                                           dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, enc_layers)

        if arm == "jepa":
            # ---- context encoder: the two boundary states, and where they sit ------
            self.ctx_enc_in = nn.Linear(S, d)
            self.ctx_role = nn.Parameter(torch.randn(2, d) * 0.02)
            self.ctx_when = nn.Linear(2 * n_freq + 1, d)
            cl = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                            norm_first=True, activation="gelu",
                                            dropout=0.0)
            self.ctx_enc = nn.TransformerEncoder(cl, ctx_layers)
            # ---- unordered parallel readout: 8 latents, ALL ALWAYS PRESENT --------
            self.path_q = nn.Parameter(torch.randn(n_path, d) * 0.02)
            self.readout = nn.ModuleList([XBlock(d, n_heads)
                                          for _ in range(readout_layers)])
            # ---- the TIME register (V5's, verbatim) --------------------------------
            self.time_mlp = nn.Sequential(nn.Linear(curve_pts + 3, 4 * d), nn.GELU(),
                                          nn.Linear(4 * d, d))
            self.time_q = nn.Parameter(torch.randn(max(n_time, 1), d) * 0.02)
            self.fsq_time = FSQ(d, fsq_levels_time or fsq_levels)
            self.jepa_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        elif arm == "arclen":
            # ---- V8: maskfsq's machinery, verbatim, on the PATH grid --------------
            # per-arc-node attention + node masking + contiguous groups; the ONLY
            # architectural addition is the mandatory speed token (see ARMS note).
            assert n_time == 1, n_time
            self.n_groups = n_path
            gb = group_bounds(T, n_path)                    # 8 slices of the PATH
            gm = torch.zeros(n_path, T, dtype=torch.bool)
            for g in range(n_path):
                gm[g, gb[g]:gb[g + 1]] = True
            self.register_buffer("group_mask", gm)          # [G, T_arc]
            self.grp_q = nn.Parameter(torch.randn(n_path, d) * 0.02)
            self.grp_pos = nn.Parameter(torch.randn(n_path, d) * 0.02)
            gl = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                            norm_first=True, activation="gelu",
                                            dropout=0.0)
            self.grp_enc = nn.TransformerEncoder(gl, 2)
            # the speed token: jepa's, verbatim
            self.time_mlp = nn.Sequential(nn.Linear(curve_pts + 3, 4 * d), nn.GELU(),
                                          nn.Linear(4 * d, d))
            self.time_q = nn.Parameter(torch.randn(1, d) * 0.02)
            self.fsq_time = FSQ(d, fsq_levels_time or fsq_levels)
        else:
            # ---- arm 2: NO global registers. Time-local group pooling. ------------
            assert n_time == 0 and n_path == n_groups, (n_path, n_time, n_groups)
            self.n_groups = n_groups
            gb = group_bounds(T, n_groups)
            gm = torch.zeros(n_groups, T, dtype=torch.bool)
            for g in range(n_groups):
                gm[g, gb[g]:gb[g + 1]] = True
            self.register_buffer("group_mask", gm)                 # [G,T]
            self.grp_q = nn.Parameter(torch.randn(n_groups, d) * 0.02)
            self.grp_pos = nn.Parameter(torch.randn(n_groups, d) * 0.02)
            gl = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                            norm_first=True, activation="gelu",
                                            dropout=0.0)
            self.grp_enc = nn.TransformerEncoder(gl, 2)
            self.fsq_time = None
        self.fsq_path = FSQ(d, fsq_levels)

        # ------------- decoder / predictor (IDENTICAL in both arms) ----------------
        self.reg_id = nn.Parameter(torch.randn(self.n_reg, d) * 0.02)
        self.dec_time = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(),
                                      nn.Linear(d, d))
        self.ctx_in = nn.Linear(S, d)               # the two conditioning frames
        self.ctx_id = nn.Parameter(torch.randn(2, d) * 0.02)
        self.dec = nn.ModuleList([XBlock(d, n_heads) for _ in range(dec_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, S))
        with torch.no_grad():                       # start at ZERO deviation (V5's fix)
            self.out[-1].weight.mul_(0.01)
            self.out[-1].bias.zero_()
        # kept ONLY so eval_v5.stat8_ladder runs unchanged. Never used in training.
        self.budgets = V.prefix_budgets(n_path)

    # ------------------------------------------------------------------ helpers
    def _tfeat(self, B, T, device, dtype):
        t = torch.linspace(0.0, 1.0, T, device=device)
        return fourier_time(t, self.n_freq).to(dtype)[None].expand(B, -1, -1)

    def _canon(self, w):
        """Resample the window to the canonical T frames (V5/V6's _canon, verbatim)."""
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

    # ------------------------------------------------------------------ encoder
    def _arc_regrid(self, wc):
        """[B,T,nk,9] on the CLOCK grid -> [B,T,nk,9] on the PATH grid.

        Node i sits at the clock time where the motion has covered fraction i/(T-1)
        of its total arc length (lloyd: "per arc-path attn"). Near-static windows
        blend toward the plain clock (beta -> 1 as total path -> 0) so the inverse
        stays well-defined. After this, rate_features' velocity/accel channels are
        TIME-FREE derivatives with respect to path: direction of travel / curvature.
        """
        B, T = wc.shape[:2]
        c = V.arclen_curve(wc.float())                          # [B,T], nondecreasing
        end = c[:, -1:].clamp_min(1e-4)
        tn = torch.linspace(0, 1, T, device=wc.device, dtype=c.dtype)[None]
        beta = 1.0 / (1.0 + end)                               # static -> clock
        arc = (1 - beta) * c / end + beta * tn
        a = self.arc_alpha
        prog = arc if a >= 1.0 else (a * arc + (1 - a) * tn)    # strictly increasing
        s = tn.expand(B, T).contiguous()
        idx = torch.searchsorted(prog.contiguous(), s, right=True).clamp(1, T - 1)
        p0, p1 = prog.gather(1, idx - 1), prog.gather(1, idx)
        u = ((s - p0) / (p1 - p0).clamp_min(1e-8)).clamp(0, 1)
        fid = (idx - 1).to(wc.dtype) + u.to(wc.dtype)           # fractional clock time
        i0 = fid.floor().long()
        i1 = (i0 + 1).clamp(max=T - 1)
        a = (fid - i0.to(fid.dtype))[..., None, None]
        ex = lambda i: i[..., None, None].expand(-1, -1, wc.shape[2], wc.shape[3])
        g0, g1 = wc.gather(1, ex(i0)), wc.gather(1, ex(i1))
        return V.orthonormalize6d((1 - a) * g0 + a * g1)

    def frame_tokens(self, wc, enc_mask=None):
        """[B,T,nk,9] -> [B,T,d] post-attention frame features.

        Absolute state PLUS velocity PLUS SG5 acceleration on EVERY frame; the clock
        is added after masking because a frame's TIME COORDINATE is never hidden (the
        exact analogue of the spatial layer keeping a masked joint's geometry visible).
        """
        B, T = wc.shape[0], wc.shape[1]
        x = wc.reshape(B, T, -1)
        vel, acc = FM.rate_features(wc)
        h = (self.inp(x)
             + self.inp_v(vel.reshape(B, T, -1) * self.rate_scale)
             + self.inp_a(acc.reshape(B, T, -1) * self.acc_scale))
        if enc_mask is not None:
            h = torch.where(enc_mask[..., None],
                            self.mask_tok.to(h.dtype).expand_as(h), h)
        h = h + self.enc_time(self._tfeat(B, T, wc.device, h.dtype))
        return self.enc(h)

    def context_tokens(self, cur, tgt, k_norm):
        """The two boundary states as encoder-side context [B,2,d] (arm 1 only)."""
        B = cur.shape[0]
        s = torch.stack([cur.reshape(B, -1), tgt.reshape(B, -1)], 1)
        when = torch.stack([torch.zeros_like(k_norm), k_norm], 1)
        c = (self.ctx_enc_in(s) + self.ctx_role[None]
             + self.ctx_when(fourier_time(when, self.n_freq).to(s.dtype)))
        return self.ctx_enc(c)

    def path_features(self, w, cur=None, tgt=None, k_norm=None, enc_mask=None):
        """-> [B, n_path, f] pre-quantization features on the canonical time grid."""
        wc = self._canon(w)
        B = wc.shape[0]
        if self.arm == "arclen":
            wc = self._arc_regrid(wc)          # clock grid -> equal-path-fraction grid
        h = self.frame_tokens(wc, enc_mask)
        if self.arm == "jepa":
            if cur is None:
                cur, tgt = wc[:, 0], wc[:, -1]
            if k_norm is None:
                k_norm = torch.ones(B, device=wc.device, dtype=wc.dtype)
            ctx = self.context_tokens(cur, tgt, k_norm)
            kv = torch.cat([h, ctx.to(h.dtype)], 1)
            r = self.path_q[None].expand(B, -1, -1).to(h.dtype)
            for b in self.readout:
                r = b(r, kv)
            return self.fsq_path.features(r), h
        # ---- arm 2 / V8: masked-softmax pooling INSIDE each contiguous group ------
        # (for arclen the re-grid already happened in path_features, so a contiguous
        #  group here is a contiguous slice OF THE PATH)
        q = self.grp_q[None].expand(B, -1, -1).to(h.dtype)          # [B,G,d]
        lg = torch.einsum("bgd,btd->bgt", q, h) / math.sqrt(h.shape[-1])
        lg = lg.masked_fill(~self.group_mask[None], torch.finfo(lg.dtype).min)
        a = lg.softmax(-1)
        g = torch.einsum("bgt,btd->bgd", a.to(h.dtype), h) + self.grp_pos[None].to(h.dtype)
        g = self.grp_enc(g)
        return self.fsq_path.features(g), h

    def curve_summary(self, w):
        """The SPEED-CONTROL token's only input.

        V5/V7 fed [normalized arc-length shape ; log(total path) , log2(T/60) ,
        mean speed]. TWO of those three scalars are AMPLITUDE, not timing, and a
        linear probe on the trained speed token reads total path length back at
        R^2 0.87-0.96 — HIGHER than it reads the timing shape (0.85-0.88). So the
        "speed" token was mostly an amplitude token, which is why (a) decoding with
        the content tokens zeroed still produced a big-amplitude kick, and (b)
        transplanting a speed token moved the PATH, not just the pacing
        (dance: 12.88 cm of shape drift after arc-length alignment).

        --pure_timing drops both amplitude scalars, leaving only the normalized
        progress curve and the duration ratio. Amplitude then has to live in the
        content tokens, which is where it belongs: they see the full pose.
        """
        c = V.arclen_curve(w)
        T = c.shape[1]
        end = c[:, -1:].clamp_min(1e-4)
        i = torch.round(torch.linspace(0, T - 1, self.curve_pts,
                                       device=w.device)).long()
        shape = (c / end)[:, i]
        dur = torch.full_like(end[:, 0], math.log2(T / self.T))
        if self.pure_timing:
            z = torch.zeros_like(end[:, 0])
            rate = torch.stack([z, dur, z], -1)      # width kept so the MLP is identical
        else:
            rate = torch.stack([torch.log1p(end[:, 0]), dur, end[:, 0] / T], -1)
        return torch.cat([shape, rate], -1)

    def time_features(self, w):
        h = self.time_mlp(self.curve_summary(w).to(self.time_q.dtype))
        return self.fsq_time.features(h[:, None] + self.time_q[None])

    def encode(self, w, cur=None, tgt=None, k_norm=None, enc_mask=None, **_ignored):
        """-> dict of straight-through FSQ values + decoder-ready embeddings.

        Arm 2 returns a ZERO-WIDTH time bank ([B,0,f] / [B,0,d]) rather than a dummy
        register, because a dummy would silently add 13.9 bits to its budget. Every
        `torch.cat` downstream handles a 0-length bank, so the harnesses are unaffected.
        """
        fp, hframe = self.path_features(w, cur, tgt, k_norm, enc_mask)
        qp = self.fsq_path.quantize(fp)
        out = {"path_f": fp, "path_q": qp, "path_z": self.fsq_path.proj_out(qp),
               "frame_h": hframe}
        if self.fsq_time is not None:
            ft = self.time_features(w)
            qt = self.fsq_time.quantize(ft)
            out |= {"time_f": ft, "time_q": qt, "time_z": self.fsq_time.proj_out(qt)}
        else:
            B, f = fp.shape[0], fp.shape[-1]
            z = fp.new_zeros(B, 0, f)
            out |= {"time_f": z, "time_q": z,
                    "time_z": fp.new_zeros(B, 0, self.reg_id.shape[-1])}
        return out

    def ints(self, q, fsq):
        half = (fsq._levels - 1) / 2
        return torch.round(q * half + torch.ceil(half)).long()

    def token_ints(self, e):
        t = (self.ints(e["time_q"], self.fsq_time) if self.fsq_time is not None
             else e["time_q"].long())
        return self.ints(e["path_q"], self.fsq_path), t

    # ------------------------------------------------------------------ baseline
    def baseline(self, cur, tgt, k_norm, T_out, drop_tgt=None):
        """V5Tokenizer.baseline, VERBATIM — the straight line between the two boundary
        states in continuous normalized time, held at the target after frame k. It is
        the architectural prior in BOTH arms, so the network only ever predicts the
        DEVIATION from it and every token bit goes to the uncertain part."""
        B = cur.shape[0]
        t = torch.linspace(0.0, 1.0, T_out, device=cur.device, dtype=cur.dtype)
        u = (t[None, :] / k_norm[:, None].clamp_min(1e-3)).clamp(0.0, 1.0)[..., None, None]
        return orthonormalize((1 - u) * cur[:, None] + u * tgt[:, None])

    # ------------------------------------------------------------------ predictor
    def decode(self, path_z, time_z, cur=None, tgt=None, k_norm=None, T_out=None,
               hidden=None, return_h=False, **_ignored):
        """Per-frame clock queries cross-attend [latents ; the two context frames].

        `hidden` [B,n_path] True = that latent is padded out. NOTHING in V7's training
        loop ever passes it; it exists so eval_v5.stat8_ladder's (now purely
        diagnostic) prefix ladder still runs.
        """
        B = path_z.shape[0]
        T_out = T_out or self.T
        kv = torch.cat([path_z, time_z], 1) + self.reg_id[None].to(path_z.dtype)
        pad = torch.zeros(B, kv.shape[1], dtype=torch.bool, device=kv.device)
        if hidden is not None:
            pad[:, :self.n_path] = hidden
        when = torch.stack([torch.zeros_like(k_norm), k_norm], 1)
        c = (self.ctx_in(torch.stack([cur.reshape(B, -1), tgt.reshape(B, -1)], 1))
             + self.ctx_id[None]
             + self.dec_time(fourier_time(when, self.n_freq).to(path_z.dtype)))
        kv = torch.cat([kv, c.to(kv.dtype)], 1)
        pad = torch.cat([pad, torch.zeros(B, 2, dtype=torch.bool, device=pad.device)], 1)
        q = self.dec_time(self._tfeat(B, T_out, path_z.device, path_z.dtype))
        for b in self.dec:
            q = b(q, kv, key_padding_mask=pad)
        o = self.out(q)
        base = self.baseline(cur, tgt, k_norm, T_out)
        o = o.reshape(B, T_out, self.nk, 9)
        if self.endpoint_clamp and T_out > 1:
            # 0810 (probe 1): the junction pop is the decoder RE-GENERATING the two
            # observed endpoint frames through the bottleneck (1-2cm gap, velocity-
            # independent). Ramp the deviation to zero over the first/last 3 frames:
            # endpoints become exact pass-through of cur/tgt, interior untouched —
            # the temporal analogue of the spatial layer's vis_residual.
            u = torch.arange(T_out, device=o.device, dtype=o.dtype)
            env = (torch.minimum(u, T_out - 1 - u) / 3.0).clamp(0.0, 1.0)
            o = o * env[None, :, None, None]
        s = orthonormalize(base + o)
        return (s, q) if return_h else s

    def forward(self, w, cur, tgt, k_norm, hidden=None, T_out=None, enc_mask=None,
                return_h=False, **_ignored):
        e = self.encode(w, cur, tgt, k_norm, enc_mask=enc_mask)
        T_out = T_out or w.shape[1]
        r = self.decode(e["path_z"], e["time_z"], cur, tgt, k_norm, T_out, hidden,
                        return_h=return_h)
        if return_h:
            return r[0], e, r[1]
        return r, e


# --------------------------------------------------------------------------- run io
def build_v7(cfg):
    return V7Tokenizer(
        arm=cfg["arm"], nk=cfg["nk"], T=cfg.get("T", V.T_WIN),
        d_model=cfg.get("d_model", 256), n_heads=cfg.get("n_heads", 8),
        enc_layers=cfg.get("enc_layers", 5), ctx_layers=cfg.get("ctx_layers", 2),
        readout_layers=cfg.get("readout_layers", 2),
        dec_layers=cfg.get("dec_layers", 3),
        n_path=cfg["n_path"], n_time=cfg["n_time"], n_groups=cfg.get("n_groups", 9),
        fsq_levels=tuple(int(x) for x in
                         str(cfg.get("fsq_levels", "5,5,5,5,5,5")).split(",")),
        interp=cfg.get("aug_interp", "linear"),
        rate_scale=float(cfg.get("rate_scale", 16.0)),
        acc_scale=float(cfg.get("acc_scale", 35.0)),
        arc_alpha=float(cfg.get("arc_alpha", 1.0)),
        pure_timing=bool(cfg.get("pure_timing", False)),
        endpoint_clamp=bool(cfg.get("endpoint_clamp", False)),
        fsq_levels_time=(tuple(int(x) for x in str(cfg["fsq_levels_time"]).split(","))
                         if cfg.get("fsq_levels_time") else None))


def load_run(run, device="cpu"):
    cfg = json.load(open(f"{run}/config.json"))
    m = build_v7(cfg).to(device)
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location=device))
    m.eval()
    return m, cfg


def arm_registers(arm, n_groups=9, n_content=8):
    """The ONE place the arms' token geometry is decided. Default lands on 9 total;
    n_content overrides the jepa/arclen content-token count (0809: lloyd pinned the
    next generation at 32 — off-budget, so it must ride with --rate_sweep)."""
    if arm in ("jepa", "arclen"):
        return (n_content, 1)
    return (n_groups, 0)


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, kd, embs, T, span, dev, n=192, seed=7, zero_shot=()):
    """Reconstruction + the two controls that decide whether the bits are real, on the
    human stream; plus a zero-shot row per held-out body."""
    model.eval()
    g = np.random.default_rng(seed)
    starts, _ = kd.sample_starts(n, g, "val", span=span)
    out = {}

    def row(emb, kdx):
        w = kdx.window(emb, starts, T, span).to(dev).float()
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(w.shape[0], device=dev)
        e = model.encode(w, cur, tgt, kn)
        p = model.decode(e["path_z"], e["time_z"], cur, tgt, kn, T).float()
        pe, re = V.key_se3_err(p, w, kdx.reach[emb])
        l, ep = V.pathint_loss(p, w)
        gt_std = float(w.std(1).mean())
        r = {"pathint": round(float(l), 4), "endpoint_rel_err": round(float(ep), 4),
             "key_pos_cm": round(pe, 3), "key_rot_deg": round(re, 2),
             "energy_ratio": round(float(p.std(1).mean()) / gt_std, 3)}
        # zeroed / shuffled: the content controls, computed every eval so the
        # content share is a TRAINING CURVE, not a one-shot number at the end.
        p0 = model.decode(torch.zeros_like(e["path_z"]), e["time_z"],
                          cur, tgt, kn, T).float()
        pe0, _ = V.key_se3_err(p0, w, kdx.reach[emb])
        perm = torch.randperm(w.shape[0], device=dev)
        ps = model.decode(e["path_z"][perm], e["time_z"], cur, tgt, kn, T).float()
        pes, _ = V.key_se3_err(ps, w, kdx.reach[emb])
        gain, content = pe0 - pe, pes - pe
        r |= {"zeroed_cm": round(pe0, 3), "shuffled_cm": round(pes, 3),
              "content_share": round(content / gain, 3) if gain > 1e-6 else None}
        bl = model.baseline(cur, tgt, kn, T).float()
        r["straight_line_cm"] = round(V.key_se3_err(bl, w, kdx.reach[emb])[0], 3)
        return r

    out["human_smpl"] = row(embs[0], kd)
    for emb, kdx in zero_shot:
        try:
            out[f"zeroshot_{emb}"] = row(emb, kdx)
        except Exception as ex:
            out[f"zeroshot_{emb}"] = {"error": str(ex)}
    # the (untrained) OAT ladder, kept as a diagnostic only
    w = kd.window(embs[0], starts, T, span).to(dev).float()
    cur, tgt = w[:, 0], w[:, -1]
    kn = torch.ones(w.shape[0], device=dev)
    lad = {}
    for b in model.budgets:
        hid = (torch.arange(model.n_path, device=dev)[None] >= b).expand(w.shape[0], -1)
        p = model(w, cur, tgt, kn, hid)[0].float()
        pe, _ = V.key_se3_err(p, w, kd.reach[embs[0]])
        lad[f"n{b}"] = round(pe, 3)
    out["untrained_prefix_ladder_cm"] = lad
    out["static_anchor"] = round(V.static_pathint_anchor(w), 4)
    model.train()
    return out


# --------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--embs", default=",".join(EMBS_TRAIN),
                    help="ALL 18 embodiments except the two G1s (lloyd 2026-08-07)")
    ap.add_argument("--zero_shot", default=",".join(G1_HOLDOUT))
    ap.add_argument("--T", type=int, default=V.T_WIN)
    ap.add_argument("--n_groups", type=int, default=9,
                    help="arm 2: contiguous frame groups sharing arm 1's register "
                         "geometry, so the bit budget matches EXACTLY")
    ap.add_argument("--n_content", type=int, default=8,
                    help="jepa/arclen content-token count (default 8 = the 62.69 "
                         "bit/s family; any other value is off-budget and requires "
                         "--rate_sweep)")
    ap.add_argument("--fsq_levels", default="5,5,5,5,5,5")
    ap.add_argument("--pure_timing", action="store_true",
                    help="drop the two AMPLITUDE scalars from the speed token's input "
                         "(log total path, mean speed); keep only the normalized "
                         "progress curve + duration ratio")
    ap.add_argument("--endpoint_clamp", action="store_true",
                    help="0810 probe-1 fix: ramp decoder deviation to zero over the "
                         "first/last 3 frames so the observed endpoints pass through "
                         "exactly (temporal vis_residual)")
    ap.add_argument("--arc_alpha", type=float, default=1.0,
                    help="arclen arm only: 1=pure path grid, 0=pure clock grid")
    ap.add_argument("--fsq_levels_time", default="",
                    help="speed-token codec override, e.g. '7,7,7' (exp2: the timing "
                         "curve is ~2-3 dimensional, 6 dims waste capacity). Empty = "
                         "same grid as the content tokens.")
    ap.add_argument("--rate_sweep", action="store_true",
                    help="permit an off-budget fsq_levels point (rate-distortion sweep)")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=5)
    ap.add_argument("--ctx_layers", type=int, default=2)
    ap.add_argument("--readout_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=2500)
    # ---- arm 2: frame masking + the l-DeTok warm-up --------------------------
    ap.add_argument("--enc_mix", default=",".join(f"{k}:{v}" for k, v in FM.ENC_MIX.items()))
    ap.add_argument("--mask_warmup_steps", type=int, default=6000,
                    help="l-DeTok (2507.15856): 1D tokenizers trained WITH masking "
                         "show a grokking-like delayed phase transition and end up "
                         "undertrained; ~10 epochs of masking-DISABLED warm-up cures "
                         "it. Arm 2 is exactly that configuration. Inert for arm 1.")
    # ---- V5c loss recipe, minus everything OAT-specific ----------------------
    ap.add_argument("--aug_interp", choices=["nearest", "linear"], default="linear")
    ap.add_argument("--timing_loss", choices=["cumulative", "riemann", "both"],
                    default="both")
    ap.add_argument("--w_riemann", type=float, default=1.0)
    ap.add_argument("--riemann_deadzone", type=float, default=0.02)
    ap.add_argument("--w_pathint", type=float, default=1.0)
    ap.add_argument("--w_frame", type=float, default=6.0)
    ap.add_argument("--w_multires", type=float, default=2.0)
    ap.add_argument("--w_vel", type=float, default=0.5)
    ap.add_argument("--w_nce", type=float, default=0.1)
    ap.add_argument("--w_inv", type=float, default=1.0)
    ap.add_argument("--w_sep", type=float, default=0.5)
    ap.add_argument("--w_jepa", type=float, default=0.5,
                    help="arm 1 only, ARM-INTRINSIC: the predictor's per-frame latent "
                         "must match the target encoder's stop-grad frame features. "
                         "This is the term that makes the arm a JEPA rather than a "
                         "conditional VQ-VAE; it is declared in the verdict.")
    ap.add_argument("--inv_mode", choices=["warp", "uniform", "none"], default="",
                    help="what the invariance pair is. jepa -> warp (V5c's "
                         "non-uniform time warp: a GLOBAL path code should survive "
                         "it). maskfsq -> uniform (a pure frame-rate change: a "
                         "TIME-LOCAL code stream can and should survive that, but "
                         "cannot survive a warp that moves content between its own "
                         "groups, so demanding it would only force the codes to throw "
                         "the content away). This is the ONE loss asymmetry and it is "
                         "architecture-forced, not tuned.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke: 30 steps, tiny model")
    V.add_root_vel_flag(ap)
    args = ap.parse_args()
    if not args.inv_mode:
        args.inv_mode = "uniform" if args.arm == "maskfsq" else "warp"
    if args.tiny:
        args.steps, args.bs, args.d_model, args.enc_layers = 30, 4, 32, 1
        args.ctx_layers = args.readout_layers = args.dec_layers = 1
        args.eval_every, args.device = 20, "cpu"
        args.embs = "human_smpl,unitree_h1"
        args.zero_shot = "unitree_g1_29dof"
        args.mask_warmup_steps = 10
    assert not args.root_vel, (
        "--root_vel is impossible here: only 6 of the 18 embodiments ship a "
        "_root_world.pt side channel and the brief requires all 16 non-G1 bodies")
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev, t0 = args.device, time.time()
    enc_mix = FM.parse_mix(args.enc_mix)

    embs = [e for e in args.embs.split(",") if e]
    zs = [e for e in args.zero_shot.split(",") if e]
    assert not (set(embs) & set(G1_HOLDOUT)), "a G1 leaked into training"
    kd = V.KeyData(args.cache, embs, device=dev,
                   max_clips=200 if args.tiny else 0, root_vel=False)
    kd_zs = (V.KeyData(args.cache, zs, device=dev, max_clips=200 if args.tiny else 0)
             if zs else None)
    nk = kd.key[embs[0]].shape[1]
    span = args.T
    n_path, n_time = arm_registers(args.arm, args.n_groups, args.n_content)

    cfg = vars(args) | {"arch": "v7", "nk": nk, "span": span,
                        "n_path": n_path, "n_time": n_time,
                        "key_joints": kd.knames[embs[0]], "cache": args.cache,
                        "frozen_spatial": V.FROZEN, "head": "reg", "aug": True,
                        "embs_train": embs, "zero_shot_bodies": zs,
                        "enc_mix_parsed": enc_mix,
                        "oat": False, "prefix_budgets_trained": False}
    model = build_v7(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    lv = [int(x) for x in args.fsq_levels.split(",")]
    bits = V.bit_budget(n_path, n_time, lv, args.T)
    if args.fsq_levels_time and n_time:
        # the speed token rides its own grid — recompute its share honestly, or a
        # mixed-grid run would sail through the matched-budget assert mislabeled
        import math as _m
        lt = [int(x) for x in args.fsq_levels_time.split(",")]
        tb = n_time * sum(_m.log2(x) for x in lt)
        bits["time_bits_per_window"] = round(tb, 2)
        bits["total_bits_per_window"] = round(bits["path_bits_per_window"] + tb, 2)
        bits["total_bits_per_s"] = round(bits["total_bits_per_window"]
                                         / bits["window_seconds"], 2)
        bits["fsq_levels_time"] = args.fsq_levels_time
    cfg |= {"n_params": n_par, "bit_budget": bits,
            "budgets_diagnostic_only": list(model.budgets)}
    print(f"=== V7 arm={args.arm}  (OAT ELIMINATED: no prefix budgets, no ordered "
          f"registers, every latent always present) ===")
    print(f"  bodies ({len(embs)}) = {embs}")
    print(f"  ZERO-SHOT HOLDOUT   = {zs}")
    print(f"  key joints ({nk}) = {kd.knames[embs[0]]} -> state {nk*9} dims/frame")
    print(f"  input channels: absolute + velocity + SG5 acceleration (no jerk); "
          f"scales {args.rate_scale if hasattr(args,'rate_scale') else 16.0}/"
          f"{35.0}")
    print(f"  params {n_par/1e6:.2f}M   registers: n_path {n_path} + n_time {n_time}"
          + (f"  (groups of {group_bounds(args.T, args.n_groups)} frames)"
             if args.arm == "maskfsq" else ""))
    print("  BIT BUDGET " + json.dumps(bits))
    # ---- THE MATCHED-BUDGET ASSERTION, IN BOTH ARMS --------------------------
    # --rate_sweep is the ONE sanctioned exit (lloyd 2026-08-07: "有没有想过词表太
    # 大" — the vocab-size question is answered by a rate-distortion sweep, which by
    # definition trains off-budget points). Off-budget runs are tagged in config and
    # must never be quoted in a same-budget verdict table.
    if args.rate_sweep:
        cfg["OFF_BUDGET_RATE_SWEEP"] = True
        print(f"  [rate_sweep] OFF-BUDGET point: {bits['total_bits_per_s']} bit/s "
              f"(matched budget is {V7_BITS_PER_S}) — excluded from verdict tables")
    else:
        assert abs(bits["total_bits_per_s"] - V7_BITS_PER_S) < 0.01, (
            f"arm {args.arm}: bit budget {bits['total_bits_per_s']} != "
            f"{V7_BITS_PER_S} — a mismatched arm must not reach the verdict table")
    want = 9 if args.arm == "maskfsq" else args.n_content + 1
    assert n_path + n_time == want, (args.arm, n_path, n_time, want)
    if args.n_content != 8:
        assert args.rate_sweep, "non-default --n_content is off-budget: pass --rate_sweep"
    print(f"  inv_mode {args.inv_mode}  w_inv {args.w_inv}  w_sep {args.w_sep}"
          + (f"  w_jepa {args.w_jepa}" if args.arm == "jepa" else ""))
    if args.arm == "maskfsq":
        print(f"  enc_mix {enc_mix}   mask_warmup_steps {args.mask_warmup_steps} "
              f"(l-DeTok 2507.15856)")
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
        # ---- TIME-SCALE AUGMENTATION (the aug that won the 2c exam at 0.99+) ----
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

        # ---- arm 2's encoder-side frame masking, with the l-DeTok warm-up ------
        enc_mask = None
        if args.arm in ("maskfsq", "arclen") and step >= args.mask_warmup_steps:
            enc_mask = FM.sample_frame_mask(args.bs, model.T, enc_mix, dev)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o, e, qh = model(wa, cur, tgt, kn, enc_mask=enc_mask, return_h=True)
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
                lr_, _ = V.riemann_timing_loss(pf, gt, deadzone=args.riemann_deadzone)
                loss = loss + args.w_riemann * lr_
                parts["riemann"] = float(lr_.detach())
            if args.w_frame > 0:
                lf = F.mse_loss(pf, gt)
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
            # ---- JEPA proper (arm 1): predict the target encoder's frame features
            if args.arm == "jepa" and args.w_jepa > 0 and T_a == model.T:
                tgt_h = e["frame_h"].detach()
                lj = F.mse_loss(model.jepa_proj(qh), tgt_h) / (tgt_h.pow(2).mean() + 1e-6)
                loss = loss + args.w_jepa * lj
                parts["jepa"] = float(lj.detach())
            # ---- time-scale invariance pair (V5c's separation supervision) -----
            if args.inv_mode != "none" and (args.w_inv > 0 or args.w_sep > 0):
                T_b = int(np.clip(round(args.T * 2.0 ** rng.uniform(-1, 1)),
                                  args.T // 2, 2 * args.T))
                if args.inv_mode == "warp":
                    idx_b = V.warp_fidx(span, T_b, rng)
                else:                              # pure frame-rate change
                    idx_b = V.uniform_fidx(span, T_b)
                wb = kd.window_fidx(emb0, starts, idx_b, span).to(dev)
                # SAME enc_mask on both views: the invariance is about the MOTION,
                # not about which frames happened to be hidden.
                eb = model.encode(wb, wb[:, 0], wb[:, -1],
                                  torch.ones(args.bs, device=dev), enc_mask=enc_mask)
                if args.w_inv > 0:
                    li = F.mse_loss(eb["path_f"], e["path_f"])
                    loss = loss + args.w_inv * li
                    parts["path_inv"] = float(li.detach())
                if args.w_sep > 0 and model.fsq_time is not None:
                    gap = abs(math.log2(T_a / T_b))
                    # +eps INSIDE the sqrt: d/dx sqrt(x) is infinite at x=0, and the
                    # two time features can be arbitrarily close. This never fired
                    # before because the only arm with a speed token (jepa) always
                    # trained against WARPS, which bend the arc-length curve's SHAPE
                    # and so keep the pair apart. A pure frame-rate change leaves the
                    # normalized curve identical — only 2 of the 11 summary inputs
                    # move — so d ~ 0 and the first arclen+uniform run went NaN at
                    # step ~600. Bug was latent in V5/V7, not introduced by V8.
                    d = ((eb["time_f"] - e["time_f"]).pow(2).mean() + 1e-8).sqrt()
                    ls = F.relu(0.5 * gap - d)
                    loss = loss + args.w_sep * ls
                    parts["time_sep"] = float(ls.detach())
            # ---- InfoNCE across the 16 embodiments -----------------------------
            if args.w_nce > 0 and len(embs) > 1:
                e2 = embs[int(rng.integers(len(embs)))]
                if e2 != emb0:
                    w2 = (kd.window_fidx(e2, starts, idx_a, span) if frac
                          else kd.window_idx(e2, starts, idx_a)).to(dev)
                    ec = model.encode(w2, w2[:, 0], w2[:, k],
                                      kn, enc_mask=enc_mask)
                    a_ = F.normalize(e["path_q"].reshape(args.bs, -1), dim=-1)
                    b_ = F.normalize(ec["path_q"].reshape(args.bs, -1), dim=-1)
                    lg = a_ @ b_.t() / args.tau
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
            mk = "off" if enc_mask is None else f"{float((~enc_mask).float().mean()):.2f}"
            print(f"step {step} loss {loss.item():.4f} " +
                  " ".join(f"{k_} {v:.3f}" for k_, v in parts.items()) +
                  f"  enc_keep {mk}  {sps:.1f} steps/s ({time.time()-t0:.0f}s)",
                  flush=True)
            if step == 200:
                print(f"  PROJECTED WALL TIME: "
                      f"{(args.steps - step)/max(sps,1e-6)/60:.1f} min", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = {"step": step + 1, "loss": round(loss.item(), 4),
                  "val": evaluate(model, kd, embs, args.T, span, dev,
                                  zero_shot=[(z, kd_zs) for z in zs] if kd_zs else ())}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=1)
            torch.save(model.state_dict(), f"{args.out}/model.pt")

    torch.save(model.state_dict(), f"{args.out}/model.pt")
    print(f"DONE {args.out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
