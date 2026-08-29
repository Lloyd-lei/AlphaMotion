"""FRAME-mask sampling for the V6 temporal layer — the exact axis-swap of
masking.py (which masks JOINTS for the spatial layer).

Mask semantics, mirrored one-for-one from masking.py:
    spatial : True = this JOINT's POSE is hidden from the encoder (replaced by a
              learned mask token). The joint's GEOMETRY/SEMANTICS conditioning stays
              visible — the skeleton is always known at deployment.
    temporal: True = this FRAME's POSE is hidden from the encoder (replaced by a
              learned mask token). The frame's TIME COORDINATE stays visible — the
              clock is always known at deployment; only the pose observation is
              partial.

Families (mirror the product surface, exactly as masking.py's docstring argues):
  none       keep every frame          — the ordinary tokenizer regime (encode a
                                         window you fully observe). masking.py has
                                         no such member because a spatial encode is
                                         never asked for "the whole body plus more";
                                         a temporal encoder IS, so it is in the mix.
  rand       keep a random frame subset (keep ratio ~ U(0.2, 0.8)) — masking.py's
                                         `rand`, verbatim ratios.
  endpoints  keep ONLY the first and the target frame — this is V5's entire
                                         conditioning surface, here demoted to ONE
                                         member of a family. masking.py's `root`
                                         (keep only the anchor) is its analogue.
  sparse     keep frame 0 + 1..3 random frames — the animator's-keyframes /
                                         "single SE3 constraint" product case;
                                         masking.py's `sparse`, same 1..3 draw.
  prefix     keep frames [0, p) only, p ~ U(0.1, 0.9)*T — the STREAMING / rollout
                                         case (you only ever have the past). This
                                         member has no spatial analogue because the
                                         joint axis has no arrow of time; it is the
                                         one place the isomorphism is deliberately
                                         broken, and it is broken on purpose.

FRAME 0 IS NEVER MASKED. Exactly parallel to masking.py's "Root is never masked (it
anchors the global frame)": frame 0 anchors the temporal frame, and with --residual_in
it is the mandatory I-frame that stops rate integration from drifting.
"""
from __future__ import annotations
import numpy as np
import torch

# encoder mix: the MAE pressure. `none` is a big slice so full-window tokenization
# (what eval and the downstream generator actually do) stays in-distribution.
ENC_MIX = {"none": 0.30, "rand": 0.40, "sparse": 0.15, "prefix": 0.15}
# decoder mix: what the reconstruction is CONDITIONED on. `endpoints` keeps a
# plurality so V5c's exact regime is still trained (and so the V5c-comparable eval
# is not out-of-distribution), but 60% of steps ask for a different density — that
# is the whole V6 bet against mean regression.
DEC_MIX = {"endpoints": 0.40, "rand": 0.30, "sparse": 0.15, "prefix": 0.15}
FAMILIES = ("none", "rand", "endpoints", "sparse", "prefix")


def parse_mix(s):
    """'rand:0.4,endpoints:0.3' -> dict (same parser shape as masking.parse_mix)."""
    out = {}
    for part in str(s).split(","):
        if not part.strip():
            continue
        k, v = part.split(":")
        k = k.strip()
        if k not in FAMILIES:
            raise ValueError(f"unknown frame-mask family {k!r}; know {FAMILIES}")
        out[k] = float(v)
    if not out:
        raise ValueError("empty frame-mask mix")
    return out


def sample_frame_mask(B, T, mix=None, device="cuda", generator=None, k_idx=None):
    """[B,T] bool mask (True = this frame's pose is HIDDEN). One family per row.

    k_idx [B] long: the TARGET frame index, i.e. where the two-point boundary
    problem's second endpoint sits. `endpoints` keeps {0, k_idx}; without it the
    family degenerates to "keep frame 0 and the last frame".
    """
    mix = mix or ENC_MIX
    fams = list(mix.keys())
    w = torch.tensor([mix[f] for f in fams], device=device, dtype=torch.float32)
    fam_idx = torch.multinomial(w.expand(B, -1), 1, generator=generator).squeeze(1)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    ar = torch.arange(T, device=device)
    if k_idx is None:
        k_idx = torch.full((B,), T - 1, dtype=torch.long, device=device)
    k_idx = k_idx.to(device)
    for fi, fam in enumerate(fams):
        rows = fam_idx == fi
        if not rows.any():
            continue
        n = int(rows.sum())
        if fam == "none":
            continue                                   # nothing hidden
        if fam == "rand":
            keep = torch.rand(n, 1, device=device, generator=generator) * 0.6 + 0.2
            mask[rows] = torch.rand(n, T, device=device, generator=generator) > keep
        elif fam == "endpoints":
            m = torch.ones(n, T, dtype=torch.bool, device=device)
            m[torch.arange(n, device=device), k_idx[rows]] = False
            mask[rows] = m
        elif fam == "sparse":
            # hide all, reveal 1..3 random frames (frame 0 is restored below) —
            # masking.py's sparse v2, same 1..3 range, same rank-argsort trick.
            k = torch.randint(1, 4, (n, 1), device=device, generator=generator)
            rank = torch.rand(n, T, device=device,
                              generator=generator).argsort(1).argsort(1)
            mask[rows] = rank >= k
        elif fam == "prefix":
            p = (torch.rand(n, 1, device=device, generator=generator) * 0.8 + 0.1) * T
            mask[rows] = ar[None] >= p                  # everything from p on is gone
        else:
            raise ValueError(f"unknown frame-mask family {fam}")
    mask[:, 0] = False                                  # frame 0 anchors the clock
    return mask


def anchor_flags(vis):
    """[B,T] bool visible -> [B,T] bool I-FRAMES: the first visible frame of each
    contiguous visible run.

    This is the residual-input half of insight 3c. A frame whose PREDECESSOR was not
    observed has no observable rate, so it must be handed its ABSOLUTE state (an
    I-frame); every frame after it inside the same run is a P-frame carried by
    velocity + acceleration. Frame 0 is always an I-frame, so no chain of P-frames
    can start unanchored and integration cannot drift.
    """
    prev = torch.zeros_like(vis)
    prev[:, 1:] = vis[:, :-1]
    return vis & ~prev


def sg_smooth(x, dim=1, window=5):
    """Savitzky-Golay smoothing along `dim` (window 5, polyorder 2 by default).

    R10_RATE_PROBES.md probe 2 measured this to be MANDATORY, not cosmetic: raw
    second differences of the 30 fps rot6d channels sit at 4.5-15 dB SNR (dance 4.5,
    strike 5.8), and a 5-frame SG lifts them to 18-28 dB. The same table is why V6
    carries NO jerk channel: raw third differences are -11..-0.3 dB, SG5 leaves the
    fast families at -2..+5 dB, and the SG9 that would fix them is a 0.3 s window that
    erases the punch transient it was supposed to measure.
    """
    if window == 5:                                   # SG(5, 2) smoothing kernel
        k = torch.tensor([-3.0, 12.0, 17.0, 12.0, -3.0], device=x.device) / 35.0
    elif window == 7:
        k = torch.tensor([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], device=x.device) / 21.0
    else:
        raise ValueError("sg_smooth supports window 5 or 7")
    x = x.movedim(dim, -1)
    shp = x.shape
    xf = x.reshape(-1, 1, shp[-1])
    r = (window - 1) // 2
    xf = torch.nn.functional.pad(xf, (r, r), mode="replicate")
    y = torch.nn.functional.conv1d(xf, k.to(xf.dtype)[None, None])
    return y.reshape(shp).movedim(-1, dim)


def rate_features(w, sg=True):
    """Key-state window [B,T,nk,9] -> (vel, acc), each [B,T,nk,9].

    vel[t] = the frame-to-frame residual:
        rotation : the BODY-FRAME relative rotation R[t-1]^T R[t], re-emitted as 6D
                   with the identity subtracted, so a static frame is exactly 0.
                   This is the ANGULAR rate channel R10 probe 3 measured to transfer
                   across bodies at 7-10% magnitude error, versus 15% for positional
                   speed against h1 — it is bone-length-free by construction.
        position : the plain delta (reach units), which is the 15% channel; kept
                   because the rotation channel alone cannot express root translation.
    acc[t] = d/dt of the SG5-SMOOTHED velocity (smooth-then-difference, R10 probe 2).
             Smoothing v and differencing once is the same operation as smoothing the
             signal and differencing twice, up to the non-linearity of the relative-
             rotation map; it is done on v because v is already the geometric residual.
    vel[0] and acc[0..1] are zero (no support); those frames are I-frames anyway.

    NOTE the acausality: SG5 looks 2 frames ahead. That is fine for a bidirectional
    frame encoder, and under the `prefix` family it only costs the last two visible
    frames their acceleration.
    """
    from common import rot6d_to_matrix, matrix_to_rot6d
    B, T, nk, _ = w.shape
    R = rot6d_to_matrix(w[..., :6])                          # [B,T,nk,3,3]
    dR = R[:, :-1].transpose(-1, -2) @ R[:, 1:]              # body-frame relative
    I6 = torch.tensor([1., 0., 0., 0., 1., 0.], device=w.device, dtype=w.dtype)
    v_rot = matrix_to_rot6d(dR).to(w.dtype) - I6
    v_pos = w[:, 1:, :, 6:9] - w[:, :-1, :, 6:9]
    v = torch.cat([v_rot, v_pos], -1)                        # [B,T-1,nk,9]
    v = torch.cat([torch.zeros_like(v[:, :1]), v], 1)        # [B,T,nk,9]
    vs = sg_smooth(v, dim=1) if (sg and T >= 5) else v
    a = torch.cat([torch.zeros_like(vs[:, :1]), vs[:, 1:] - vs[:, :-1]], 1)
    a[:, :2] = 0.0
    return v, a
