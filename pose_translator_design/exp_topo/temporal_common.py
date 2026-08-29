"""P1 temporal tokenizer — shared model + stream-window data logic (ARCH_V4 P1 row).

Layer-2 of ARCH_V4: encode 2-second WINDOWS of the frozen layer-1 frame-code stream
(E19_K128_0805, via tokenize_stream.py caches) into N=8 ORDERED Perceiver registers
plus ONE reserved duration seat, all FSQ-quantized (6 dims x 5 levels per register).

Design decisions frozen here (mirrors the ARCH_V4 2026-08-05 深夜修订):
  - OAT prefix training: per-sample budget n in {1,2,4,8}; registers beyond n are
    HIDDEN FROM THE DECODER (key_padding_mask), never from the encoder readout.
  - The duration seat (last register) is NEVER hidden — it is a reserved seat whose
    codebook FSQ learns on its own; no hand-binned scale.
  - Decoder is per-frame expansion: frame-index Fourier features cross-attend the
    visible registers. NO temporal mixing in the decoder (no frame self-attention),
    so every frame is decoded independently from the same register set.
  - Reconstruction target = the frame codes' integer FSQ levels, CROSS-ENTROPY over
    L classes per dim (both streams). Never MSE (discrete target kills the
    mean-regression / amplitude-compression failure mode).
  - Controls share this file: mode="unordered" (no prefix budgets, all registers
    always visible) and mode="chunk" (matched-bits frame-chunk control: content
    register r may only read its own T/8 frame slice; duration seat reads all).

Input arms (--stream_repr, P0-b's verdict picks):
  "z"     dequantized continuous grid values per frame [D] -> Linear
  "codes" integer levels per frame, per-(dim,level) EmbeddingBag sum -> Linear

===========================================================================
P1c (2026-08-06) — --temporal_arch: KEEP THE TIME AXIS instead of pooling it
===========================================================================
Everything above is arch="global" and is the DEFAULT, unchanged, so P1/P1b stay
bit-for-bit reproducible. Two new arms sit behind --temporal_arch. Three
independent lines of evidence measured 2026-08-05/06 motivate them:

  (E1) MEASURED, P1: 8 Perceiver registers cross-attending the WHOLE 60-frame
       window + per-frame CE retain only 22-73% of GT motion energy (walk 0.73 /
       run 0.40 / jacks 0.23 / stand 0.50, prefix-1 vs GT temporal std of
       FK-from-rotation positions). The frame-code floor retains 82-98%, so the
       loss is in the TEMPORAL bottleneck, not in frozen E19. P1b's loss-only
       fixes did not rescue it: motion-weighted CE alone was null-to-negative
       (0.71/0.37/0.17 at w=2), and the best variant (+velocity loss on
       continuous z) reached 0.86/0.55/0.29/0.79 — still under the pre-registered
       bar and jumping jacks still dead.
  (E2) MEASURED, zero-training DCT probe: keeping ONLY the first 8 temporal DCT
       coefficients per dim, requantizing to the FSQ levels and decoding through
       frozen E19, retains 95-100% of walk motion and 84% of the floor for
       jumping jacks. Motion is temporally LOW-FREQUENCY and 8 slots are plenty —
       what the slots ATTEND TO is the problem. (Honest caveat, see
       LIT_TEMPORAL_FSQ.md 8.4/8.7: the probe is NOT bit-matched — 8 DCT coeffs
       per dim is ~26k reals vs our 111 bit. It shows direction, not a win.)
  (E3) LITERATURE (pose_translator_design/LIT_TEMPORAL_FSQ.md): audio neural
       codecs universally use STRIDED CONVOLUTIONAL downsampling and never global
       pooling (SoundStream/EnCodec/DAC, LIT 1.1); no codec trains on per-step
       L1/L2/CE alone and there are ablations (DAC SI-SDR 9.12->1.07, LIT 1.2);
       SNAC uses MULTI-SCALE time resolutions, coarse tokens at a low rate and
       fine tokens at a high rate (LIT 1.6); stable-codec uses an FSQ config
       essentially identical to ours (6 dims x 5 levels) but at 25 tokens/s with
       the time axis preserved (LIT 1.7) — so FSQ is fine, the LAYOUT is not.

  arch="conv"  (ARM C1) strided 1D conv encoder, 60 frames -> n_fine latent TIME
      steps, each keeping its own temporal position, each FSQ-quantized by the
      same 6x5 grid. SNAC-style multi-scale residual quantization: --conv_scales
      "2,6" spends 2 tokens on the whole 2 s window (1 Hz) and 6 tokens on the
      residual at 3 Hz. The ordered-prefix property becomes COARSE-TO-FINE OVER
      SCALE, not over registers, so the OAT budgets are the cumulative scale
      boundaries (2, 8) rather than (1,2,4,8). Decoder = transposed-conv upsample
      + refinement; it still never sees a neighbouring GT frame, only its own
      quantized latents.
  arch="freq"  (ARM C2) registers BOUND TO TEMPORAL FREQUENCIES. A FIXED DCT-II
      basis over normalized window time projects the frame stream into K bands
      BEFORE the encoder; register k carries band k; the decoder writes band k
      back through the SAME fixed basis (inverse DCT) and nothing else. This
      makes dropping motion structurally impossible: the decoder's per-frame
      feature is sum_k phi_k(t) g_k, so if only the DC band survives the feature
      is CONSTANT in t and the output is exactly static. Motion can only live in
      k>=1. Band order low->high frequency is a natural coarse-to-fine prefix, so
      the OAT budgets (1,2,4,8) keep their meaning.

Bit budget is IDENTICAL across all three arms at the defaults (this is the point
of holding n_content=8 and fsq_levels=5,5,5,5,5,5 fixed):
    per register  6 * log2(5)      = 13.93 bit
    8 content registers            = 111.45 bit / 2 s window = 55.7 bit/s
    + 1 duration seat              = 125.38 bit / 2 s window = 62.7 bit/s
Use bit_budget() to print the exact numbers for any configuration.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FSQ                                # noqa: E402  (frozen file, reused not edited)
from tokenize_stream import load_stream              # noqa: E402

BUDGETS = (1, 2, 4, 8)
# the 6 bodies of R7's kNN body-identifiability probe (71.6% frame-level reference)
R7_BODIES = ["human_smpl", "unitree_g1_29dof", "pal_talos",
             "unitree_h1", "booster_t1", "fourier_gr3"]


# --------------------------------------------------------------------------- model
def fourier_time(t, n_freq=8):
    """t [...,] in [0,1] -> [..., 2*n_freq+1] (raw t + sin/cos at octave freqs)."""
    f = (2.0 ** torch.arange(n_freq, device=t.device, dtype=t.dtype)) * math.pi
    ang = t[..., None] * f
    return torch.cat([t[..., None], torch.sin(ang), torch.cos(ang)], -1)


class XBlock(nn.Module):
    """CrossBlock (model.py) + attn_mask passthrough, needed by the chunk control.
    Kept local so the frozen model.py is not touched."""
    def __init__(self, d, h):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.nq = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, q, kv, key_padding_mask=None, attn_mask=None):
        a, _ = self.mha(self.nq(q), self.n1(kv), self.n1(kv),
                        key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        q = q + a
        return q + self.ff(self.n2(q))


class ConvResBlock(nn.Module):
    """SoundStream/EnCodec-style residual unit: dilated conv -> GELU -> 1x1 conv.
    Local receptive field only — this is the point (LIT 1.1: codecs downsample the
    time axis with strided convs, they never pool it away)."""
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.n = nn.GroupNorm(8, ch)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(self.n(x))))


def dct_basis(T, K, device, dtype=torch.float32):
    """Fixed DCT-II basis on NORMALIZED window time -> (analysis [T,K], synthesis [T,K]).

    phi_k(t) = cos(pi * k * (t + 0.5) / T).  Analysis uses the amplitude
    normalization (1/T for k=0, 2/T otherwise) so coefficients are LENGTH-INVARIANT:
    the same motion sampled at 30 or 120 frames gives the same band amplitudes, which
    is what the normalized-time decode contract in this file requires. Synthesis is
    the raw basis, so x_hat(t) = sum_k C_k phi_k(t) is the truncated inverse DCT.
    """
    t = torch.arange(T, device=device, dtype=dtype)[:, None]
    k = torch.arange(K, device=device, dtype=dtype)[None, :]
    phi = torch.cos(math.pi * k * (t + 0.5) / T)
    a = torch.full((1, K), 2.0 / T, device=device, dtype=dtype)
    a[0, 0] = 1.0 / T
    return phi * a, phi


class TemporalTokenizer(nn.Module):
    def __init__(self, D, L, T=60, stream_repr="z", d_model=384, n_heads=8,
                 enc_layers=5, readout_layers=2, dec_layers=3, n_content=8,
                 fsq_levels=(5, 5, 5, 5, 5, 5), mode="oat", code_emb=128, n_freq=8,
                 dur_bottleneck=False, arch="global", conv_scales="2,6", conv_down=3,
                 conv_res=2, conv_attn=True, freq_mix_layers=2):
        super().__init__()
        assert mode in ("oat", "unordered", "chunk")
        assert arch in ("global", "conv", "freq")
        assert not (dur_bottleneck and mode == "chunk"), "bottleneck not wired for chunk"
        assert not (arch != "global" and mode == "chunk"), "chunk control is global-only"
        self.D, self.L, self.T, self.mode = D, L, T, mode
        self.stream_repr, self.n_content = stream_repr, n_content
        self.dur_bottleneck = dur_bottleneck
        self.arch = arch
        self.n_reg = n_content + 1                       # +1 = reserved duration seat (last)
        d = d_model
        if stream_repr == "z":
            self.inp = nn.Linear(D, d)
        else:
            self.bag = nn.EmbeddingBag(D * L, code_emb, mode="sum")
            self.register_buffer("dim_off", torch.arange(D, dtype=torch.long) * L)
            self.inp = nn.Linear(code_emb, d)
        self.enc_time = nn.Linear(2 * n_freq + 1, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                           norm_first=True, activation="gelu", dropout=0.0)
        # arch="global": self.enc runs over the 60 FRAMES (then Perceiver readout pools).
        # arch="conv"  : self.enc runs over the n_fine LATENT TIME STEPS after the strided
        #                conv stack (stable-codec puts its attention after downsampling).
        # arch="freq"  : self.enc runs over the K DCT BANDS.
        self.enc = nn.TransformerEncoder(
            layer, enc_layers if arch != "freq" else max(int(freq_mix_layers), 1))
        if arch == "global":
            self.reg_q = nn.Parameter(torch.randn(self.n_reg, d) * 0.02)  # ORDERED queries
            self.readout = nn.ModuleList([XBlock(d, n_heads)
                                          for _ in range(readout_layers)])
        self.fsq = FSQ(d, fsq_levels)                    # one shared 6x5 grid, per register
        if arch == "global":
            self.reg_id = nn.Parameter(torch.randn(self.n_reg, d) * 0.02)  # decoder seat id
            self.dec_time = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(),
                                          nn.Linear(d, d))
            self.dec = nn.ModuleList([XBlock(d, n_heads) for _ in range(dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D * L))
        self.n_freq = n_freq
        if dur_bottleneck:
            # duration-seat information bottleneck (P1b): the seat's ONLY input is the
            # window's length ratio T_in/T; content registers read a canonical-length
            # resampled view, so sequence length is invisible to them by construction.
            # The seat's pre-quantize FSQ features are a FIXED monotone map of
            # s = 0.5*log2(T_in/T)+0.5 (0.5x..2x -> [0,1]) with staggered gains, so
            # the code is guaranteed to move with speed — normalized-time recon gives
            # no gradient that would make a learned seat spread on its own.
            self.register_buffer("dur_gain",
                                 torch.tensor([2., 4., 6., 8., 16., 32.])[:self.fsq.f])
        # dequantization vectors (filled by set_deq from stream meta)
        self.register_buffer("deq_off", torch.zeros(D))
        self.register_buffer("deq_half", torch.ones(D))

        # ------------------------------------------------------------------ P1c arms
        # Everything above is untouched P1: with arch="global" no module below is
        # constructed and the init RNG draw order is identical, so P1/P1b runs
        # reproduce bit-for-bit.
        self.budgets = reg_budgets(n_content)          # OAT prefix budgets for this arch
        if arch == "conv":
            # ---- ARM C1: strided conv over time, SNAC-style multi-scale residual FSQ
            self.conv_scales = tuple(int(x) for x in str(conv_scales).split(",") if x)
            assert sum(self.conv_scales) == n_content, (
                f"--conv_scales {conv_scales} sums to {sum(self.conv_scales)} but "
                f"--n_content is {n_content}; they must match so the BIT BUDGET is "
                f"identical to P1's")
            assert list(self.conv_scales) == sorted(self.conv_scales), \
                "scales must be coarse-to-fine (increasing token count)"
            n_fine = max(self.conv_scales)
            # conv input length L0 = n_fine * 2**conv_down; grow conv_down until the
            # stack sees at least T frames, so no frames are thrown away by the
            # pre-resample (n_fine=6 -> L0=96 with conv_down=4).
            while n_fine * (2 ** conv_down) < T:
                conv_down += 1
            self.conv_down, self.n_fine = conv_down, n_fine
            self.conv_L0 = n_fine * (2 ** conv_down)
            self.conv_attn = bool(conv_attn)
            enc_b, dec_b = [], []
            for _ in range(conv_down):
                enc_b.append(nn.Sequential(
                    *[ConvResBlock(d, 3 ** i) for i in range(conv_res)],
                    nn.GELU(), nn.Conv1d(d, d, 4, stride=2, padding=1)))
                dec_b.append(nn.Sequential(
                    nn.ConvTranspose1d(d, d, 4, stride=2, padding=1), nn.GELU(),
                    *[ConvResBlock(d, 3 ** i) for i in range(conv_res)]))
            self.conv_enc = nn.ModuleList(enc_b)
            self.conv_dec = nn.ModuleList(dec_b[::-1])
            self.step_pos = nn.Parameter(torch.randn(n_fine, d) * 0.02)  # latent time id
            # per-scale pre/post projections around the SHARED 6x5 FSQ grid (SNAC uses
            # one quantizer per scale; we keep one grid so the bits match P1 exactly)
            self.scale_in = nn.ModuleList([nn.Linear(d, d) for _ in self.conv_scales])
            self.scale_out = nn.ModuleList([nn.Linear(d, d) for _ in self.conv_scales])
            self.seat_read = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            self.seat_write = nn.Linear(d, d)
            self.len_emb = nn.Linear(1, d)             # only used without dur_bottleneck
            # prefix budgets are CUMULATIVE SCALE BOUNDARIES: the ordered-prefix
            # property is coarse-to-fine over SCALE, not over registers.
            self.budgets = tuple(int(x) for x in np.cumsum(self.conv_scales))
        elif arch == "freq":
            # ---- ARM C2: registers bound to fixed temporal DCT bands
            self.n_bands = n_content
            # per-frame (POINTWISE in time) feature map before the DCT — restores the
            # nonlinear capacity the frame transformer had, without mixing frames, so
            # the band interpretation survives: a static window still has only DC.
            self.freq_pre = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                          nn.Linear(2 * d, d))
            self.band_emb = nn.Parameter(torch.randn(n_content, d) * 0.02)   # analysis id
            self.band_id = nn.Parameter(torch.randn(n_content, d) * 0.02)    # synthesis id
            self.band_in = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            self.freq_mix_layers = max(int(freq_mix_layers), 1)
            # ANALYSIS may mix bands (it only has to *compute* good coefficients); the
            # SYNTHESIS side is strictly band-local and the register mask is applied
            # AFTER band_out, so a hidden band contributes EXACTLY zero — that is what
            # makes dropping motion structurally impossible rather than merely unlikely.
            self.band_out = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                          nn.Linear(2 * d, d))
            self.band_scale = nn.Parameter(torch.ones(n_content, 1))
            self.seat_read = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            self.seat_write = nn.Linear(d, d)
            self.len_emb = nn.Linear(1, d)             # only used without dur_bottleneck
            # Band 0 alone is DC, i.e. EXACTLY a static pose — that is the structural
            # guarantee, not a failure — so a prefix budget of 1 is not a meaningful
            # MOTION budget and would burn a quarter of the OAT mass on it. The coarsest
            # motion budget is 2 (DC + fundamental), which is also C1's coarsest scale
            # boundary, so the two arms stay directly comparable at matched bits.
            self.budgets = tuple(b for b in reg_budgets(n_content) if b >= 2)

    def set_deq(self, off, half):
        self.deq_off.copy_(torch.as_tensor(off, dtype=torch.float32))
        self.deq_half.copy_(torch.as_tensor(half, dtype=torch.float32))

    def _t_feat(self, B, T, device, dtype):
        t = torch.linspace(0.0, 1.0, T, device=device)
        return fourier_time(t, self.n_freq).to(dtype)[None].expand(B, -1, -1)

    def _chunk_mask(self, T, device):
        """chunk control: content register r may only read frames [r*T/8,(r+1)*T/8);
        the duration seat reads everything. Built per call — T varies under time aug."""
        m = torch.zeros(self.n_reg, T, dtype=torch.bool, device=device)
        for r in range(self.n_content):
            lo, hi = (r * T) // self.n_content, ((r + 1) * T) // self.n_content
            m[r] = True; m[r, lo:hi] = False
        return m

    def embed_frames(self, codes, add_time=True):
        """uint8 codes [B,T,D] -> [B,T,d]. add_time=False (ARM C2 only) drops the
        frame-index Fourier features: they are not a function of the pose, so their
        DCT would pollute every band with a fixed pattern."""
        B, T, D = codes.shape
        if self.stream_repr == "z":
            q = (codes.to(self.deq_off.dtype) - self.deq_off) / self.deq_half
            x = self.inp(q)
        else:
            idx = codes.long().reshape(B * T, D) + self.dim_off
            x = self.inp(self.bag(idx).reshape(B, T, -1))
        if not add_time:
            return x
        return x + self.enc_time(self._t_feat(B, T, codes.device, x.dtype))

    # ------------------------------------------------------------------ P1c shared
    def _canon(self, codes):
        """Resample a window to the canonical length T (nearest frame) — the P1b
        dur_bottleneck contract: sequence length invisible to the content path."""
        T_in = codes.shape[1]
        if T_in == self.T:
            return codes
        idx = torch.round(torch.linspace(0, T_in - 1, self.T,
                                         device=codes.device)).long()
        return codes[:, idx]

    def _len_feat(self, T_in, B, device, dtype):
        """Speed cue for the conv/freq content path when there is NO duration seat
        bottleneck: one scalar s = 0.5*log2(T_in/T)+0.5, same parameterization the
        bottleneck seat uses. Without this the pre-resample would silently delete the
        duration information that P1's variable-length encoder could still see."""
        s = 0.5 * math.log2(T_in / self.T) + 0.5
        return self.len_emb(torch.full((B, 1, 1), s - 0.5, device=device, dtype=dtype))

    def _seat_q(self, latent, T_in):
        """Reserved duration seat -> (q [B,1,f], zq [B,1,d]). With --dur_bottleneck the
        seat's ONLY input is T_in/T (fixed monotone map, P1b); otherwise it is learned
        from a mean over the latent — the one place a global summary is still allowed,
        because that seat exists precisely to hold a window-level scalar."""
        if self.dur_bottleneck:
            s = 0.5 * math.log2(T_in / self.T) + 0.5
            f = ((s - 0.5) * self.dur_gain).to(latent.dtype)[None, None]
            f = f.expand(latent.shape[0], -1, -1)
        else:
            f = self.fsq.proj_in(self.seat_read(latent.mean(1, keepdim=True)))
        q = self.fsq.quantize(f)
        return q, self.seat_write(self.fsq.proj_out(q))

    def _q_ints(self, q):
        """Straight-through FSQ grid values [...,f] -> integer levels, exactly the
        values FSQ.ints() would give for the same features."""
        half = (self.fsq._levels - 1) / 2
        return torch.round(q * half + torch.ceil(half)).long()

    # ------------------------------------------------------------------ ARM C1 (conv)
    def _conv_latent(self, codes):
        """codes [B,T_in,D] -> latent [B,n_fine,d]; every step keeps its OWN temporal
        position (this is the whole point of the arm). Frames are linearly resampled
        to conv_L0 = n_fine * 2**conv_down so the strided stack is length-agnostic."""
        B, T_in = codes.shape[0], codes.shape[1]
        x = self.embed_frames(self._canon(codes) if self.dur_bottleneck else codes)
        h = x.transpose(1, 2)                                   # [B,d,T]
        if h.shape[-1] != self.conv_L0:
            h = F.interpolate(h, size=self.conv_L0, mode="linear", align_corners=False)
        for blk in self.conv_enc:
            h = blk(h)                                          # halve the time axis
        z = h.transpose(1, 2) + self.step_pos
        if not self.dur_bottleneck:
            z = z + self._len_feat(T_in, B, z.device, z.dtype)
        if self.conv_attn:
            z = self.enc(z)                                     # attention over 6-8 steps
        return z

    def _conv_regs(self, codes):
        """SNAC-style multi-scale RESIDUAL quantization over the latent time axis.
        Scale k pools the running residual to n_k steps, quantizes it with the shared
        6x5 FSQ, and the next scale sees what is left. Registers are laid out
        coarse-to-fine, so an OAT prefix = a coarse-to-fine SCALE prefix."""
        z = self._conv_latent(codes)
        res, qs, zs = z, [], []
        for k, n_k in enumerate(self.conv_scales):
            zk = res if n_k == self.n_fine else \
                F.adaptive_avg_pool1d(res.transpose(1, 2), n_k).transpose(1, 2)
            q = self.fsq.quantize(self.fsq.proj_in(self.scale_in[k](zk)))
            dk = self.scale_out[k](self.fsq.proj_out(q))        # [B,n_k,d]
            up = dk if n_k == self.n_fine else F.interpolate(
                dk.transpose(1, 2), size=self.n_fine, mode="linear",
                align_corners=False).transpose(1, 2)
            res = res - up
            qs.append(q); zs.append(dk)
        qd, zd = self._seat_q(z, codes.shape[1])
        return torch.cat(qs + [qd], 1), torch.cat(zs + [zd], 1)

    def _conv_decode(self, zq, hidden, T_out):
        """Upsample the (masked) multi-scale latents back to T_out frames. The decoder
        NEVER sees a neighbouring GT frame — only its own quantized latents — so the
        VAE 'decoder bypasses the latent' failure mode stays excluded (LIT 7.3)."""
        B = zq.shape[0]
        m = torch.ones(B, self.n_reg, 1, device=zq.device, dtype=zq.dtype) \
            if hidden is None else (~hidden)[..., None].to(zq.dtype)
        zc = zq[:, :self.n_content] * m[:, :self.n_content]
        lat, o = 0.0, 0
        for n_k in self.conv_scales:
            part = zc[:, o:o + n_k]; o += n_k
            if n_k != self.n_fine:
                part = F.interpolate(part.transpose(1, 2), size=self.n_fine,
                                     mode="linear", align_corners=False).transpose(1, 2)
            lat = lat + part
        lat = lat + zq[:, -1:] * m[:, -1:]                      # duration seat: constant
        h = lat.transpose(1, 2)
        for blk in self.conv_dec:
            h = blk(h)                                          # n_fine -> conv_L0
        if h.shape[-1] != T_out:
            h = F.interpolate(h, size=T_out, mode="linear", align_corners=False)
        return self.head(h.transpose(1, 2)).reshape(B, T_out, self.D, self.L)

    # ------------------------------------------------------------------ ARM C2 (freq)
    def _freq_regs(self, codes):
        """Register k = the k-th temporal DCT band of the window. Fixed basis, no
        learned pooling: motion cannot be routed away from the bands it lives in."""
        B, T_in = codes.shape[0], codes.shape[1]
        cd = self._canon(codes) if self.dur_bottleneck else codes
        x = self.freq_pre(self.embed_frames(cd, add_time=False))        # [B,T,d]
        A, _ = dct_basis(x.shape[1], self.n_bands, x.device, torch.float32)
        C = torch.einsum("btd,tk->bkd", x.float(), A).to(x.dtype)       # band coefficients
        h = self.band_in(C) + self.band_emb
        if not self.dur_bottleneck:
            h = h + self._len_feat(T_in, B, h.device, h.dtype)
        h = self.enc(h)                                                 # mix across bands
        q = self.fsq.quantize(self.fsq.proj_in(h))
        qd, zd = self._seat_q(h, T_in)
        return torch.cat([q, qd], 1), torch.cat([self.fsq.proj_out(q), zd], 1)

    def _freq_decode(self, zq, hidden, T_out):
        """Inverse DCT with the SAME fixed basis. The register mask is applied AFTER
        band_out, so a hidden band contributes exactly zero: keep only DC and the
        per-frame feature is constant in t, hence the output is EXACTLY static. Motion
        can therefore only be carried by bands k>=1 — it cannot be silently dropped."""
        B = zq.shape[0]
        m = torch.ones(B, self.n_reg, 1, device=zq.device, dtype=zq.dtype) \
            if hidden is None else (~hidden)[..., None].to(zq.dtype)
        g = self.band_out(zq[:, :self.n_bands] + self.band_id) \
            * self.band_scale * m[:, :self.n_bands]                     # [B,K,d]
        _, S = dct_basis(T_out, self.n_bands, zq.device, torch.float32)
        x = torch.einsum("bkd,tk->btd", g.float(), S).to(zq.dtype)      # [B,T_out,d]
        x = x + zq[:, -1:] * m[:, -1:]                                  # seat: constant
        return self.head(x).reshape(B, T_out, self.D, self.L)

    def _read(self, codes):
        h = self.enc(self.embed_frames(codes))
        r = self.reg_q[None].expand(h.shape[0], -1, -1)
        am = self._chunk_mask(codes.shape[1], codes.device) if self.mode == "chunk" else None
        for b in self.readout:
            r = b(r, h, attn_mask=am)
        return r

    def _features(self, codes):
        """Pre-quantization FSQ features for all registers [B,n_reg,f].
        dur_bottleneck: content registers read the window RESAMPLED to canonical T
        (sequence length invisible to them); the duration seat is a fixed monotone
        function of T_in/T ONLY (its exclusive input feature)."""
        if not self.dur_bottleneck:
            return self.fsq.features(self._read(codes))
        T_in = codes.shape[1]
        if T_in != self.T:                                # canonical-length content view
            idx = torch.round(torch.linspace(0, T_in - 1, self.T,
                                             device=codes.device)).long()
            codes = codes[:, idx]
        h = self.enc(self.embed_frames(codes))
        r = self.reg_q[None, :self.n_content].expand(h.shape[0], -1, -1)
        for b in self.readout:
            r = b(r, h)
        fc = self.fsq.features(r)                         # [B,n_content,f]
        s = 0.5 * math.log2(T_in / self.T) + 0.5          # 0.5x..2x -> [0,1]
        fd = ((s - 0.5) * self.dur_gain).to(fc.dtype)[None, None]
        return torch.cat([fc, fd.expand(fc.shape[0], -1, -1)], 1)

    def encode_regs(self, codes):
        """-> (zq [B,n_reg,d] decoder-ready, q [B,n_reg,f] straight-through grid values)."""
        if self.arch == "conv":
            q, zq = self._conv_regs(codes)
            return zq, q
        if self.arch == "freq":
            q, zq = self._freq_regs(codes)
            return zq, q
        if self.dur_bottleneck:
            q = self.fsq.quantize(self._features(codes))
            return self.fsq.proj_out(q), q
        return self.fsq(self._read(codes))

    def reg_ints(self, codes):
        """Integer register codes [B,n_reg,f] in [0, levels) — the temporal vocabulary."""
        if self.arch != "global":
            return self._q_ints(self.encode_regs(codes)[1])
        return self.fsq.ints(self._features(codes))[0]

    def decode_frames(self, zq, hidden=None, T_out=None):
        """Expand registers back to T_out frames at NORMALIZED time. hidden bool
        [B,n_reg] True = register invisible to the decoder. Returns logits
        [B,T_out,D,L].

        arch="global": per-frame expansion, no temporal mixing — every frame query
          sees only the registers, never other frames.
        arch="conv"/"freq": the decoder DOES mix along time (transposed conv /
          inverse DCT) because the registers are now time-local and must be
          re-synthesized into a continuous stream. It still never sees a GT frame, so
          the 'decoder bypasses the latent' failure mode (LIT 7.3) stays excluded.
        """
        if self.arch == "conv":
            return self._conv_decode(zq, hidden, T_out or self.T)
        if self.arch == "freq":
            return self._freq_decode(zq, hidden, T_out or self.T)
        B = zq.shape[0]
        T_out = T_out or self.T
        kv = zq + self.reg_id[None]
        fq = self.dec_time(self._t_feat(B, T_out, zq.device, zq.dtype))
        for b in self.dec:
            fq = b(fq, kv, key_padding_mask=hidden)
        return self.head(fq).reshape(B, T_out, self.D, self.L)

    def forward(self, codes, hidden=None):
        """Reconstruct the input window (any length) from its registers."""
        zq, q = self.encode_regs(codes)
        return self.decode_frames(zq, hidden, T_out=codes.shape[1]), q


def reg_budgets(n_content):
    """Prefix budgets that exist for this register count: powers of two up to
    n_content, plus n_content itself when it is not a power of two.
    n_content=8 -> (1,2,4,8) and n_content=4 -> (1,2,4), i.e. IDENTICAL to the old
    `BUDGETS capped at n_content` for every P1/P1b configuration; the generalization
    only matters for P1c's higher-rate arms (n_content=16 -> (1,2,4,8,16))."""
    b, out = 1, []
    while b <= n_content:
        out.append(b); b *= 2
    if out[-1] != n_content:
        out.append(n_content)
    return tuple(out)


def model_budgets(model):
    """Prefix budgets to train/evaluate at for this model. arch='conv' overrides them
    with the cumulative SCALE boundaries — its ordered prefix is coarse-to-fine over
    scale, so a budget that cuts a scale in half is meaningless."""
    return tuple(getattr(model, "budgets", None) or reg_budgets(model.n_content))


def sample_hidden(B, n_reg, mode, device, gen=None, budgets=None):
    """OAT prefix budgets: per-sample n ~ U over the model's budget list; content
    registers >= n hidden. The duration seat (index n_reg-1) is NEVER hidden.
    unordered/chunk: nothing hidden."""
    hidden = torch.zeros(B, n_reg, dtype=torch.bool, device=device)
    if mode == "oat":
        buds = budgets or reg_budgets(n_reg - 1)
        n = torch.as_tensor(buds, device=device)[
            torch.randint(len(buds), (B,), device=device, generator=gen)]
        hidden[:, :n_reg - 1] = torch.arange(n_reg - 1, device=device)[None] >= n[:, None]
    return hidden


def bit_budget(n_content, fsq_levels, T=60, fps=30.0, scales=None, stride=1):
    """EXACT bit budget of a temporal-tokenizer configuration. Printed by every run so
    the arms are never compared at different rates by accident.

    Reference points from LIT_TEMPORAL_FSQ.md: stable-codec runs the SAME 6x5 FSQ grid
    at 25 tokens/s (~348 bit/s); the rough rate-distortion estimate for carrying gait
    phase is 128-192 bit/s; P1's default is 55.7 bit/s of content."""
    per = float(sum(math.log2(int(l)) for l in fsq_levels))
    win_s = T * stride / fps          # --frame_stride keeps the wall-clock span fixed
    out = {"bits_per_register": round(per, 3), "n_content": n_content,
           "frames_compressed": T, "frame_stride": stride,
           "content_bits_per_window": round(per * n_content, 2),
           "total_bits_per_window": round(per * (n_content + 1), 2),
           "window_seconds": round(win_s, 3),
           "content_bits_per_s": round(per * n_content / win_s, 2),
           "total_bits_per_s": round(per * (n_content + 1) / win_s, 2),
           "content_tokens_per_s": round(n_content / win_s, 2),
           "temporal_compression_x": round(T / n_content, 2)}
    if scales:
        out["scales"] = list(scales)
        out["scale_token_rates_hz"] = [round(s / win_s, 3) for s in scales]
        out["scale_bits_per_s"] = [round(per * s / win_s, 2) for s in scales]
    return out


def multires_dyn_loss(zh, zg, scales=(1, 2, 4, 8)):
    """Multi-resolution temporal dynamics loss — the motion analogue of the audio
    multi-resolution STFT loss (EnCodec's mel windows 2^5..2^11, DAC's 7 window
    lengths; LIT 7.4 calls it the single largest ablation gap in the whole sweep).

    zh/zg [B,T,D] predicted vs GT dequantized z. At each temporal scale s the sequence
    is mean-pooled by s and the FIRST DIFFERENCE is compared with a RELATIVE MSE, so a
    perfectly static prediction scores exactly 1.0 at every scale — the same readable
    anchor --w_vel uses (scale 1 IS --w_vel). Pooling by 2/4/8 prices the slow
    trajectory, which a per-frame loss can satisfy with high-frequency jitter.
    """
    T = zh.shape[1]
    tot, n = zh.new_zeros(()), 0
    for s in scales:
        if T // s < 2:
            continue
        if s == 1:
            a, b = zh, zg
        else:
            k = (T // s) * s
            a = zh[:, :k].reshape(zh.shape[0], T // s, s, -1).mean(2)
            b = zg[:, :k].reshape(zg.shape[0], T // s, s, -1).mean(2)
        da, db = a[:, 1:] - a[:, :-1], b[:, 1:] - b[:, :-1]
        tot = tot + ((da - db) ** 2).mean() / (db.pow(2).mean() + 1e-4)
        n += 1
    return tot / max(n, 1)


def budget_hidden(B, n_reg, n, device):
    """Fixed evaluation budget: first n content registers + duration seat visible."""
    hidden = torch.zeros(B, n_reg, dtype=torch.bool, device=device)
    hidden[:, :n_reg - 1] = torch.arange(n_reg - 1, device=device)[None] >= n
    return hidden


# --------------------------------------------------------------------------- data
class StreamGroup:
    """One cache-aligned group of tokenize_stream outputs: SAME clip subset across
    embodiments (identical bounds), so window (clip, offset) pairs across bodies are
    the InfoNCE positives. Train/val split by clip, rng(123) — mirrors P0-b."""
    def __init__(self, name, dirs_by_emb, T, in_ram=False, stride=1):
        """stride > 1 (P1c, --frame_stride): a window still spans T*stride RAW frames
        (same wall-clock duration) but only every stride-th frame is kept, so the model
        compresses T frames instead of T*stride. This is the ProMAG observation ported
        to motion (LIT 6.7 / mechanism D): a 4x-compression model on 6 fps video beats a
        16x model on 24 fps video. Positions returned by sample_positions are ALREADY
        strided, so every downstream consumer (windows, GT row lookup) just works."""
        self.name, self.T, self.stride = name, T, stride
        self.streams = {}
        ref = None
        for emb, d in dirs_by_emb.items():
            st = load_stream(d, mmap=not in_ram)
            if in_ram:
                st["codes"] = np.ascontiguousarray(st["codes"])
            self.streams[emb] = st
            if ref is None:
                ref = st
            else:
                assert np.array_equal(ref["bounds"], st["bounds"]), \
                    f"{name}: {emb} bounds differ — streams not clip-aligned"
        self.meta = ref["meta"]
        self.bounds = ref["bounds"]
        C = len(self.bounds) - 1
        rng = np.random.default_rng(123)
        val_clips = set(rng.permutation(C)[: max(1, C // 10)].tolist())
        self.tr_pools, self.va_pools = [], []            # (start, end) absolute, len >= T
        for c in range(C):
            s, e = int(self.bounds[c]), int(self.bounds[c + 1])
            if e - s < T * stride:
                continue
            (self.va_pools if c in val_clips else self.tr_pools).append((s, e))
        self.n_frames = int(self.bounds[-1])

    def embs(self):
        return list(self.streams.keys())

    def sample_positions(self, n, g, split="train", T_in=None):
        """Contiguous windows of length T_in (default self.T) -> pos int64 [n,T_in].

        Time-scale augmentation contract (ARCH_V4 速度三件套): the caller varies T_in
        per STEP (0.5x..2x of T). A longer/shorter contiguous span IS the time-scaled
        view — the decoder reconstructs it at corresponding NORMALIZED time, so the
        content registers must encode the normalized-time shape and only the sequence
        length (= duration) distinguishes slow from fast. That length cue is what the
        reserved duration seat can learn to absorb; nothing is hand-binned."""
        T_in = T_in or self.T
        span = T_in * self.stride                        # RAW frames the window covers
        pools = [(s, e) for s, e in
                 (self.tr_pools if split == "train" else self.va_pools) if e - s >= span]
        assert pools, f"{self.name}: no {split} clip is >= {span} frames"
        ci = g.integers(0, len(pools), n)
        starts = np.array([pools[i][0] + g.integers(0, pools[i][1] - pools[i][0] - span + 1)
                           for i in ci], np.int64)
        return starts[:, None] + np.arange(0, span, self.stride)[None]

    def windows(self, emb, pos):
        """pos [n,T] -> uint8 tensor [n,T,D]."""
        return torch.from_numpy(np.asarray(self.streams[emb]["codes"][pos.reshape(-1)])
                                .reshape(pos.shape[0], pos.shape[1], -1))


def load_groups(stream_root, sets, embs, T, in_ram=False, stride=1):
    """sets/embs comma-lists -> [StreamGroup]. Dir layout: {root}/{set}_{emb}."""
    groups = []
    for s in [x for x in sets.split(",") if x]:
        dirs = {}
        for e in [x for x in embs.split(",") if x]:
            d = f"{stream_root}/{s}_{e}"
            if os.path.isdir(d):
                dirs[e] = d
            else:
                print(f"WARN missing stream {d} — skipped")
        assert dirs, f"set {s}: no streams found under {stream_root}"
        groups.append(StreamGroup(s, dirs, T, in_ram=in_ram, stride=stride))
        g = groups[-1]
        print(f"[{s}] embs={list(dirs)} frames={g.n_frames} "
              f"clips tr/va={len(g.tr_pools)}/{len(g.va_pools)} D={g.meta['D']}")
    return groups


def stream_L(meta):
    """Number of CE classes per code dim. FSQ with even levels saturates one step
    past the nominal grid (round(tanh*3.5) hits ±4), so L = 2*offset+1 (=9 for E19)."""
    return int(2 * max(meta["deq_offset"]) + 1)


# --------------------------------------------------------------------------- metrics
def code_metrics(pred, tgt, f=13):
    """pred/tgt long [...,D] -> level acc, 13-dim tuple acc, exact frame, pos/rot split."""
    m = (pred == tgt)
    D = m.shape[-1]
    flat = m.reshape(-1, D)
    return {
        "level_acc": m.float().mean().item(),
        "tuple13_acc": flat.reshape(flat.shape[0], -1, f).all(-1).float().mean().item(),
        "frame_exact": flat.all(-1).float().mean().item(),
        "pos_level_acc": flat[:, :D // 2].float().mean().item(),
        "rot_level_acc": flat[:, D // 2:].float().mean().item(),
    }


def info_nce(qa, qb, tau=0.1):
    """Symmetric InfoNCE between two embodiments' register grids for the SAME windows.
    qa/qb [B,n_reg,f] straight-through FSQ values (grads flow to both encoders).
    Same machinery as train_dual.py's --w_contrast, on register space."""
    va = F.normalize(qa.reshape(qa.shape[0], -1), dim=-1)
    vb = F.normalize(qb.reshape(qb.shape[0], -1), dim=-1)
    logits = va @ vb.t() / tau
    labels = torch.arange(va.shape[0], device=va.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
