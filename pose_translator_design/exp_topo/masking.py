"""Joint-mask sampling for E0b masked pretraining. Mask semantics: True = this
joint's POSE is hidden from the encoder (replaced by a learned mask token). The
joint's geometry/semantics conditioning stays visible — the skeleton is always
known at deployment; only the pose observation is partial.

Families (mirror the product surface):
  rand    keep a random joint subset (keep ratio ~ U(0.2, 0.8))
  ee      keep only root + kinematic-tree leaves (end effectors; topology-agnostic)
  root    keep only the root
  sparse  keep only root + 1..3 random leaves — the "single-wrist SE3 constraint"
          product case. Added 2026-08-03 after the demo showed root+wrist (2.5%
          keep) was outside rand's 20-80% range: the mask family distribution IS
          the product surface.
Root is never masked (it anchors the global frame).
"""
from __future__ import annotations
import numpy as np, torch

DEFAULT_MIX = {"rand": 0.4, "ee": 0.3, "root": 0.3}
SPARSE_MIX = {"rand": 0.3, "ee": 0.25, "root": 0.15, "sparse": 0.3}


def parse_mix(s):
    """'rand:0.4,ee:0.3,root:0.3' -> dict"""
    out = {}
    for part in str(s).split(","):
        k, v = part.split(":")
        out[k.strip()] = float(v)
    return out


def leaf_mask(spec):
    """[J] bool, True at kinematic-tree leaves."""
    J = spec.J
    is_parent = np.zeros(J, bool)
    for p in spec.parents:
        if p >= 0:
            is_parent[p] = True
    return torch.as_tensor(~is_parent)


def sample_joint_mask(spec, B, mix=None, device="cuda", generator=None):
    """[B,J] bool mask (True = pose hidden). Each row draws one family from `mix`."""
    mix = mix or DEFAULT_MIX
    J = spec.J
    root = int(np.where(spec.parents < 0)[0][0])
    fams = list(mix.keys())
    w = torch.tensor([mix[f] for f in fams], device=device)
    fam_idx = torch.multinomial(w.expand(B, -1), 1, generator=generator).squeeze(1)  # [B]
    mask = torch.zeros(B, J, dtype=torch.bool, device=device)
    for fi, fam in enumerate(fams):
        rows = fam_idx == fi
        if not rows.any():
            continue
        n = int(rows.sum())
        if fam == "rand":
            keep = torch.rand(n, 1, device=device, generator=generator) * 0.6 + 0.2  # U(0.2,0.8)
            mask[rows] = torch.rand(n, J, device=device, generator=generator) > keep
        elif fam == "ee":
            hide = ~leaf_mask(spec).to(device)          # hide non-leaves
            mask[rows] = hide[None].expand(n, -1)
        elif fam == "root":
            mask[rows] = True
        elif fam == "sparse":
            # v2: hide all, reveal 1..3 random joints from the WHOLE body (not just leaves).
            # v1 sampled tree leaves only — on soma77 those are fingertips, so a wrist
            # constraint stayed out-of-distribution (demo: fingertip 2.65cm vs wrist 16.7cm).
            # The product contract is "any joint can be a constraint": sample accordingly.
            # Vectorized: rank random scores per row, reveal the k lowest-ranked joints.
            k = torch.randint(1, 4, (n, 1), device=device, generator=generator)
            rank = torch.rand(n, J, device=device, generator=generator).argsort(1).argsort(1)
            mask[rows] = rank >= k
        else:
            raise ValueError(f"unknown mask family {fam}")
    mask[:, root] = False
    return mask
