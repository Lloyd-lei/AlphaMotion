"""E13 probe — spatiotemporal factorization of the frozen code stream.

Question (appendix "备用架:E13 时空分解码本"): does motion have reusable words above
the frame level? The probe is a small autoencoder over T=16-frame windows of a frozen
tokenizer's code stream whose bottleneck is split into

    ONE slow token per window   (content: "what motion is this")
  + a per-frame phase           (where in the motion we are)
  [+ an optional tiny per-frame residual, only used to buy matched recon quality]

and whose decoder is a PURE FUNCTION of (slow, phase_t) evaluated frame by frame --
no temporal mixing at all in the decoder. That is the strong form of the claim: a
window is a curve, the slow token picks which curve, the phase says how far along.

Why this is the right probe right now (R7_LATENT/ANALYSIS.md):
  - H(code_t | code_{t-1}) is 77.7% below H(code_t) at 30 fps (86% on rot): three
    quarters of every frame is redundant with the previous frame.
  - 59-65% of integer code tuples are singletons: the frame vocabulary is far too
    fine-grained to be words.
  - Both scaling axes (codebook K=16->128, data 500->4000 clips) are flat, so the
    bottleneck is representational, not capacity or data.
  - E1A's frame-level AR degenerates to 93.9% token copying, because with absolute
    per-frame codes the mode of "next code" IS "copy current". A (slow, phase)
    target has no such trivial solution.

Modes (all share one architecture; --mode selects which bottleneck branches exist):
  factor      slow + phase (+ residual)          the hypothesis
  frame       per-frame code only                the matched-bits control ("just spend
                                                  the same budget per frame")
  slow_only   slow only, decoder is constant     what the phase buys
  phase_only  phase only, no slow token          what the slow token buys

Design choices the appendix left open (recorded here because they are load-bearing):
  * Phase is ONE circular scalar. The encoder emits (a,b) per frame, normalized to the
    unit circle -> theta_t; theta is then snapped to --phase_bins bins with a
    straight-through estimator so it costs exactly log2(bins) bits and is a real
    discrete symbol. The decoder consumes Fourier features [cos(h.theta), sin(h.theta)]
    for h=1..--phase_harm, which makes the circular topology explicit (no wrap
    discontinuity) and lets a shallow decoder express sharp phase-dependent shapes.
  * The slow token is shared by BROADCAST: one vector per window, added identically to
    every frame's decoder input. It is quantized with the same FSQ rule as the
    tokenizer (model.FSQ), with ODD default levels (5) so the nominal bit count is
    exact -- FSQ with even L actually reaches L+1 grid values (R7 fig4), which would
    make an "8 levels = 3 bits" claim a slight lie in the bits accounting.
  * NO phase smoothness / monotonicity regularizer by default (--w_phase_smooth 0).
    Forcing the phase to advance would make eval's "does the phase behave like a
    phase" test a tautology. The knob exists but leaving it at 0 is the honest run.
    A CPU dry run showed the free phase barely moving (|dtheta| ~ 0.02 rad/frame), so
    there is a second, weaker knob --w_phase_spread that penalizes only the circular
    CONCENTRATION of a window's phases: it says "use the phase", never "advance in
    order", so monotonicity and wrapping stay emergent and testable. Both default 0;
    the pipeline trains one extra run with spread on as an anti-collapse fallback.
  * The decoder has NO temporal mixing at all -- it is residual MLP blocks over the
    feature axis, so frame t is reconstructed from (slow, phase_t) alone and nothing
    can leak sideways between frames. Deliberately not configurable: a decoder that
    could attend across the window would weaken the claim to "the window is jointly
    coded", which is not what E13 is testing.

Reconstruction target is q = the tokenizer's own dequantized grid value, i.e. exactly
what its decoder consumes; predictions are re-rounded onto the same grid to recover
integer codes, so eval can push them through the frozen decoder for geodesic error.

Usage (GPU, ~1 h):
  python train_e13_probe.py --stream .../e13_stream/E19_K128_amass_human \
      --out .../runs/E13_factor_0805 --mode factor
CPU smoke test:
  python train_e13_probe.py --stream <tiny> --out /tmp/x --device cpu \
      --steps 20 --bs 4 --d_model 64 --n_enc 1 --n_dec 1
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize_stream import load_stream                      # noqa: E402

CACHE_ROOT = "/var/tmp/alphamotion/topo_exp"
DEFAULT_STREAM = f"{CACHE_ROOT}/e13_stream/E19_K128_amass_human"


# ---------------------------------------------------------------- discrete bottlenecks
class ScalarFSQ(nn.Module):
    """FSQ over an explicit vector of scalars (no in/out projections).

    Same bounding + rounding + straight-through rule as model.FSQ, so an E13 slow
    token is a codebook entry in exactly the sense the rest of the stack uses.
    `bits` is nominal; `bits_exact` counts the grid values actually reachable
    (tanh saturation makes even level counts reach L+1 values -- R7 fig4).
    """

    def __init__(self, levels):
        super().__init__()
        self.levels = [int(l) for l in levels]
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.float32))
        self.bits = float(sum(math.log2(l) for l in self.levels))
        self.bits_exact = float(sum(math.log2(2 * math.ceil((l - 1) / 2) + 1) for l in self.levels))
        self.n_words = int(np.prod([2 * math.ceil((l - 1) / 2) + 1 for l in self.levels]))

    def forward(self, x):                                   # [...,n] -> [...,n] on grid
        half = (self._levels - 1) / 2
        xt = torch.tanh(x) * half
        return (xt + (torch.round(xt) - xt).detach()) / half

    def ints(self, x):                                      # [...,n] -> long in [0, L]
        half = (self._levels - 1) / 2
        return (torch.round(torch.tanh(x) * half) + torch.ceil(half)).long()


def expand_levels(levels, dims):
    """(5,) x 8 dims -> 8 fives; (5,5,5) with dims=3 -> itself. Anything else is a typo."""
    levels = list(levels)
    if len(levels) == 1:
        return levels * dims
    assert len(levels) == dims, f"levels {levels} does not match {dims} dims"
    return levels


def circle_phase(ab, bins):
    """[...,2] -> (theta_quantized, theta_raw). theta is snapped to `bins` bins
    straight-through (bins <= 0 keeps it continuous, which is NOT a discrete code --
    the bits accounting refuses to price it).

    Both are returned because they diagnose different failures: if theta_raw sweeps a
    real arc but theta (quantized) is constant, the phase is alive and merely SUB-BIN,
    and the fix is more bins, not more regularization."""
    ab = F.normalize(ab, dim=-1, eps=1e-6)
    raw = torch.atan2(ab[..., 1], ab[..., 0])               # (-pi, pi]
    theta = raw
    if bins and bins > 0:
        step = 2 * math.pi / bins
        theta = raw + (torch.round(raw / step) * step - raw).detach()
    return theta, raw


def fourier(theta, harm):
    h = torch.arange(1, harm + 1, device=theta.device, dtype=theta.dtype)
    a = theta[..., None] * h
    return torch.cat([torch.cos(a), torch.sin(a)], -1)      # [...,2H]


class MLPBlock(nn.Module):
    """Residual MLP over the feature axis only -- applied to [B,T,d] it is strictly
    per-frame, which is what makes the decoder a pure function of (slow, phase_t)."""

    def __init__(self, d, mult=4):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x):
        return x + self.f(self.n(x))


# ---------------------------------------------------------------- the probe
class E13Probe(nn.Module):
    def __init__(self, D, T=16, d_model=256, n_heads=8, n_enc=3, n_dec=3, mode="factor",
                 slow_levels=(5,) * 6, phase_bins=16, phase_harm=4,
                 res_dims=0, res_levels=(5,) * 3, frame_dims=2, frame_levels=(5,)):
        super().__init__()
        self.D, self.T, self.mode = D, T, mode
        self.use_slow = mode in ("factor", "slow_only")
        self.use_phase = mode in ("factor", "phase_only")
        self.use_res = mode == "factor" and res_dims > 0
        self.use_frame = mode == "frame"
        assert self.use_slow or self.use_phase or self.use_frame, f"empty bottleneck: {mode}"

        self.inp = nn.Linear(D, d_model)
        self.tpos = nn.Parameter(torch.randn(T, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_enc)
        self.enc_norm = nn.LayerNorm(d_model)

        if self.use_slow:
            self.slow_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                           nn.Linear(d_model, len(slow_levels)))
            self.slow_fsq = ScalarFSQ(slow_levels)
            self.slow_up = nn.Linear(len(slow_levels), d_model)
        if self.use_phase:
            self.phase_head = nn.Linear(d_model, 2)
            self.phase_bins, self.phase_harm = phase_bins, phase_harm
            self.phase_up = nn.Linear(2 * phase_harm, d_model)
        if self.use_res:
            self.res_head = nn.Linear(d_model, res_dims)
            self.res_fsq = ScalarFSQ(expand_levels(res_levels, res_dims))
            self.res_up = nn.Linear(res_dims, d_model)
        if self.use_frame:
            self.frame_head = nn.Linear(d_model, frame_dims)
            self.frame_fsq = ScalarFSQ(expand_levels(frame_levels, frame_dims))
            self.frame_up = nn.Linear(frame_dims, d_model)

        self.dec_bias = nn.Parameter(torch.zeros(d_model))
        self.dec = nn.ModuleList([MLPBlock(d_model) for _ in range(n_dec)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, D))

    # -- encode: window -> bottleneck parts (pre-quantization AND quantized)
    def encode(self, x):                                    # x [B,T,D] dequantized q
        h = self.enc_norm(self.enc(self.inp(x) + self.tpos[: x.shape[1]]))
        parts = {}
        if self.use_slow:
            s_pre = self.slow_head(h.mean(1))               # [B,n_slow]  ONE per window
            parts["slow_pre"] = s_pre
            parts["slow_q"] = self.slow_fsq(s_pre)
        if self.use_phase:
            parts["theta"], parts["theta_raw"] = circle_phase(self.phase_head(h), self.phase_bins)
        if self.use_res:
            r_pre = self.res_head(h)
            parts["res_pre"] = r_pre
            parts["res_q"] = self.res_fsq(r_pre)
        if self.use_frame:
            f_pre = self.frame_head(h)
            parts["frame_pre"] = f_pre
            parts["frame_q"] = self.frame_fsq(f_pre)
        return parts

    def decode(self, parts, T):
        B = next(iter(parts.values())).shape[0]
        z = self.dec_bias.expand(B, T, -1).clone()
        if "slow_q" in parts:                               # broadcast: identical every frame
            z = z + self.slow_up(parts["slow_q"])[:, None, :]
        if "theta" in parts:
            z = z + self.phase_up(fourier(parts["theta"], self.phase_harm))
        if "res_q" in parts:
            z = z + self.res_up(parts["res_q"])
        if "frame_q" in parts:
            z = z + self.frame_up(parts["frame_q"])
        for b in self.dec:
            z = b(z)
        return self.out(z)

    def forward(self, x):
        parts = self.encode(x)
        return self.decode(parts, x.shape[1]), parts

    # -- bits accounting (per 16-frame window)
    def bits(self, T=None):
        T = T or self.T
        b = {"slow": 0.0, "phase": 0.0, "residual": 0.0, "frame": 0.0}
        if self.use_slow:
            b["slow"] = self.slow_fsq.bits_exact
        if self.use_phase:
            b["phase"] = (math.log2(self.phase_bins) if self.phase_bins > 0 else float("inf")) * T
        if self.use_res:
            b["residual"] = self.res_fsq.bits_exact * T
        if self.use_frame:
            b["frame"] = self.frame_fsq.bits_exact * T
        b["total_per_window"] = sum(v for k, v in b.items() if k != "total_per_window")
        b["total_per_frame"] = b["total_per_window"] / T
        return b


# ---------------------------------------------------------------- data
class WindowStream:
    """Clip-disjoint train/val windows over a packed code stream.

    Windows never straddle a clip boundary. The split is by CLIP, not by window, so
    the downstream action-family probe cannot cheat by memorizing a clip."""

    def __init__(self, stream, T=16, val_frac=0.15, seed=0, in_ram=True, limit_clips=0):
        self.s = stream
        self.T = T
        self.meta = stream["meta"]
        self.D = self.meta["D"]
        bounds = stream["bounds"]
        C = len(bounds) - 1
        if limit_clips:
            C = min(C, limit_clips)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(C)
        nval = max(1, int(round(val_frac * C)))
        self.val_clips = np.sort(perm[:nval])
        self.train_clips = np.sort(perm[nval:])
        self.starts = {k: self._starts(bounds, cl) for k, cl in
                       (("train", self.train_clips), ("val", self.val_clips))}
        self.codes = np.ascontiguousarray(stream["codes"]) if in_ram else stream["codes"]
        self.off = torch.from_numpy(stream["off"])
        self.half = torch.from_numpy(stream["half"])
        self.n_clips = C

    def _starts(self, bounds, clips):
        out = [np.arange(bounds[c], bounds[c + 1] - self.T + 1) for c in clips
               if bounds[c + 1] - bounds[c] >= self.T]
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def window_codes(self, starts):                          # -> uint8 [B,T,D]
        idx = np.asarray(starts)[:, None] + np.arange(self.T)[None, :]
        return self.codes[idx.reshape(-1)].reshape(len(starts), self.T, self.D)

    def deq(self, codes_u8, device):                         # uint8 -> q float32
        x = torch.from_numpy(np.ascontiguousarray(codes_u8)).to(device).float()
        return (x - self.off.to(device)) / self.half.to(device)

    def batch(self, split, bs, rng, device):
        st = self.starts[split]
        sel = st[rng.integers(0, len(st), bs)]
        return self.deq(self.window_codes(sel), device), sel

    def val_batches(self, bs, device, max_windows=8192, stride=None):
        st = self.starts["val"]
        if stride:
            st = st[::stride]
        if len(st) > max_windows:
            st = st[np.linspace(0, len(st) - 1, max_windows).astype(np.int64)]
        for i in range(0, len(st), bs):
            sel = st[i:i + bs]
            yield self.deq(self.window_codes(sel), device), sel


# ---------------------------------------------------------------- metrics
def grid_ints(q, off, half):
    """Predicted q -> integer levels on the tokenizer's own grid (clamped to the
    reachable range), i.e. the inverse of tokenize_stream's dequantization."""
    k = torch.round(q * half)
    return (k.clamp(-off, off) + off).long()


def recon_stats(pred, tgt, off, half, D):
    """MSE + exact level agreement, overall and per stream (pos | rot)."""
    ip, it = grid_ints(pred, off, half), grid_ints(tgt, off, half)
    acc = (ip == it).float()
    h = D // 2
    return {"mse": float(((pred - tgt) ** 2).mean()),
            "mse_pos": float(((pred[..., :h] - tgt[..., :h]) ** 2).mean()),
            "mse_rot": float(((pred[..., h:] - tgt[..., h:]) ** 2).mean()),
            "level_acc": float(acc.mean()),
            "level_acc_pos": float(acc[..., :h].mean()),
            "level_acc_rot": float(acc[..., h:].mean())}


@torch.no_grad()
def evaluate(model, ws, device, bs=256, max_windows=4096):
    """Window-count-weighted mean of recon_stats over held-out (clip-disjoint) windows."""
    was_training = model.training
    model.eval()
    off, half = ws.off.to(device), ws.half.to(device)
    tot, n_tot = {}, 0
    for x, _ in ws.val_batches(bs, device, max_windows):
        b = x.shape[0]
        st = recon_stats(model(x)[0], x, off, half, ws.D)
        parts = model.encode(x)
        if "theta" in parts:                                 # phase advance per frame (rad)
            d = torch.diff(parts["theta"], dim=1)
            d = torch.atan2(torch.sin(d), torch.cos(d))      # wrap into (-pi, pi]
            st["phase_mean_abs_step"] = float(d.abs().mean())
            st["phase_frac_positive"] = float((d > 0).float().mean())
            dr = torch.diff(parts["theta_raw"], dim=1)
            dr = torch.atan2(torch.sin(dr), torch.cos(dr))
            st["phase_raw_mean_abs_step"] = float(dr.abs().mean())
            # 1 - |mean e^(i.theta)| over the window: 0 = fully collapsed, ~1 = spread
            st["phase_raw_dispersion"] = float(1 - (torch.cos(parts["theta_raw"]).mean(1) ** 2 +
                                                    torch.sin(parts["theta_raw"]).mean(1) ** 2).sqrt().mean())
        for k, v in st.items():
            tot[k] = tot.get(k, 0.0) + v * b
        n_tot += b
    out = {k: v / max(n_tot, 1) for k, v in tot.items()}
    out["n_val_windows"] = n_tot
    if was_training:
        model.train()
    return out


def parse_levels(s):
    return tuple(int(x) for x in str(s).split(","))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=DEFAULT_STREAM)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="factor", choices=["factor", "frame", "slow_only", "phase_only"])
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_enc", type=int, default=3)
    ap.add_argument("--n_dec", type=int, default=3)
    ap.add_argument("--slow_levels", default="5,5,5,5,5,5", help="6 dims x 5 = 15625 words")
    ap.add_argument("--phase_bins", type=int, default=16, help="0 = continuous (unpriceable)")
    ap.add_argument("--phase_harm", type=int, default=4)
    ap.add_argument("--res_dims", type=int, default=0, help=">0 adds a per-frame residual code")
    ap.add_argument("--res_levels", default="5,5,5")
    ap.add_argument("--frame_dims", type=int, default=2, help="--mode frame bottleneck width")
    ap.add_argument("--frame_levels", default="5")
    ap.add_argument("--match_bits", type=float, default=0.0,
                    help="--mode frame: pick frame_dims to match this bits/window budget")
    ap.add_argument("--w_phase_smooth", type=float, default=0.0,
                    help="OFF by default on purpose; >0 makes the phase test a tautology")
    ap.add_argument("--w_phase_spread", type=float, default=0.0,
                    help="penalize circular CONCENTRATION |mean e^(i.theta)|^2 inside a window: "
                         "pushes the model to USE the phase without saying anything about order "
                         "or direction, so monotonicity stays an emergent, testable property. "
                         "Still off by default; the anti-collapse fallback run turns it on.")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--limit_clips", type=int, default=0, help="CPU smoke test")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    stream = load_stream(args.stream, mmap=args.mmap)
    ws = WindowStream(stream, T=args.T, val_frac=args.val_frac, seed=args.seed,
                      in_ram=not args.mmap, limit_clips=args.limit_clips)
    assert stream["meta"]["stages"] == 1, "E13 probe assumes single-stage FSQ (residual FSQ " \
                                          "has no unique per-stage inverse from a predicted q)"
    print(f"stream {args.stream}: D={ws.D} clips train/val={len(ws.train_clips)}/{len(ws.val_clips)} "
          f"windows train/val={len(ws.starts['train'])}/{len(ws.starts['val'])}")

    frame_dims = args.frame_dims
    if args.mode == "frame" and args.match_bits > 0:
        per = math.log2(2 * math.ceil((parse_levels(args.frame_levels)[0] - 1) / 2) + 1)
        frame_dims = max(1, int(round(args.match_bits / args.T / per)))
        print(f"--match_bits {args.match_bits:.1f}/window -> frame_dims={frame_dims}")

    model = E13Probe(ws.D, T=args.T, d_model=args.d_model, n_heads=args.n_heads,
                     n_enc=args.n_enc, n_dec=args.n_dec, mode=args.mode,
                     slow_levels=parse_levels(args.slow_levels), phase_bins=args.phase_bins,
                     phase_harm=args.phase_harm, res_dims=args.res_dims,
                     res_levels=parse_levels(args.res_levels), frame_dims=frame_dims,
                     frame_levels=parse_levels(args.frame_levels)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    bits = model.bits(args.T)
    raw = stream["meta"]["bits_per_frame_nominal"] * args.T
    print(f"params {nparam/1e6:.2f}M   bits/window {bits['total_per_window']:.1f} "
          f"({json.dumps({k: round(v,1) for k,v in bits.items()})})   raw frame codes "
          f"{raw:.0f} bits/window -> compression {raw/max(bits['total_per_window'],1e-9):.0f}x")
    if model.use_slow:
        print(f"slow vocabulary: {model.slow_fsq.n_words} words ({model.slow_fsq.bits_exact:.2f} bits)")

    cfg = vars(args) | {"frame_dims": frame_dims, "D": ws.D, "n_params": nparam,
                        "bits": bits, "raw_bits_per_window": raw,
                        "val_clips": ws.val_clips.tolist(), "train_clips": ws.train_clips.tolist()}
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)) *
        0.5 * (1 + math.cos(math.pi * min(s / max(args.steps, 1), 1.0))))
    rng = np.random.default_rng(args.seed + 1)
    amp = args.amp == "bf16" and dev.startswith("cuda")
    hist, t0 = [], time.time()

    model.train()
    for step in range(1, args.steps + 1):
        x, _ = ws.batch("train", args.bs, rng, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred, parts = model(x)
            loss = F.mse_loss(pred.float(), x)
            if "theta" in parts and (args.w_phase_smooth > 0 or args.w_phase_spread > 0):
                th = parts["theta"]
                if args.w_phase_smooth > 0:      # constant-rate prior (makes test C circular)
                    d = torch.diff(th, dim=1)
                    d = torch.atan2(torch.sin(d), torch.cos(d))
                    loss = loss + args.w_phase_smooth * (d - d.mean(1, keepdim=True)).pow(2).mean()
                if args.w_phase_spread > 0:      # anti-collapse only: order-agnostic
                    # applied to the PRE-quantization angle: exact collapse is a
                    # stationary point of any permutation-symmetric spread term, so the
                    # useful gradient lives in the continuous angle, not the binned one
                    tr_ = parts["theta_raw"]
                    R = (torch.cos(tr_).mean(1) ** 2 + torch.sin(tr_).mean(1) ** 2)
                    loss = loss + args.w_phase_spread * R.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % args.eval_every == 0 or step == args.steps or step == 1:
            ev = evaluate(model, ws, dev, bs=min(args.bs, 256))
            ev |= {"step": step, "train_loss": float(loss), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            hist.append(ev)
            json.dump(hist, open(f"{args.out}/loss_history.json", "w"), indent=1)
            json.dump({"step": step, "steps": args.steps, "last": ev, "mode": args.mode},
                      open(f"{args.out}/status.json", "w"), indent=1)
            torch.save(model.state_dict(), f"{args.out}/model.pt")
            print(f"[{step:6d}/{args.steps}] train {float(loss):.5f}  val mse {ev['mse']:.5f} "
                  f"(pos {ev['mse_pos']:.5f} rot {ev['mse_rot']:.5f})  level_acc {ev['level_acc']:.3f}"
                  + (f"  |dtheta| {ev.get('phase_mean_abs_step', 0):.3f} (raw "
                     f"{ev.get('phase_raw_mean_abs_step', 0):.3f}, disp "
                     f"{ev.get('phase_raw_dispersion', 0):.3f})" if "phase_mean_abs_step" in ev else "")
                  + f"  {ev['elapsed_s']:.0f}s", flush=True)

    torch.save(model.state_dict(), f"{args.out}/model.pt")
    print("DONE", args.out)


if __name__ == "__main__":
    main()
