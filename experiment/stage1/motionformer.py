"""
MotionFormer: Evoformer-style block adapted for motion data.

The backbone maintains two tensors that co-evolve across blocks:
    MSA  tensor  [B, T, J, H]        — per-(time, joint) hidden state
    Pair tensor  [B, J, J, H_pair]   — persistent joint-joint relationship

Each block:
    1. Row attention over joints (with pair tensor as attention bias)
    2. Column attention over time (for temporal consistency within a joint)
    3. FFN over MSA
    4. Outer product mean: MSA → update Pair
    5. Triangle multiplicative update (outgoing) on Pair
    6. Triangle attention (starting node) on Pair
    7. FFN over Pair

Design choices (simplified from full AlphaFold Evoformer):
    - Only one of four triangle variants (outgoing mult + starting attention),
      since J is small (20) the discriminative power is adequate.
    - Shared hidden dim between MSA and Pair for simplicity.
    - No recycling — Stage 1 is a single-pass sanity check.

Adapted with structural reference to OpenFold's Evoformer (aqlaboratory/openfold),
reimplemented from scratch for readability. Not line-by-line copied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class MotionFormerConfig:
    T: int = 64
    J: int = 20
    C: int = 3
    hidden: int = 128         # MSA hidden dim
    pair_hidden: int = 64     # Pair hidden dim (smaller to save memory: [J,J,H_pair])
    depth: int = 6
    heads: int = 4
    pair_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0
    # Projection dims
    opm_chunk: int = 16       # outer product mean inner dim
    tri_hidden: int = 32      # triangle update inner dim
    # ---- Ablation switches (all default-on = full MotionFormer) ----
    use_pair: bool = True     # if False: no pair tensor, no pair-bias on row attn
    use_opm: bool = True      # if False: MSA does not update pair (pair stays static)
    use_triangle: bool = True # if False: no triangle multiplicative / triangle attention

    @property
    def variant_name(self) -> str:
        if not self.use_pair:
            return "axial_only"
        if not self.use_opm and not self.use_triangle:
            return "pair_static"
        if self.use_pair and self.use_opm and self.use_triangle:
            return "full"
        # catch-all for exotic combinations
        return f"custom(pair={self.use_pair},opm={self.use_opm},tri={self.use_triangle})"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


class Transition(nn.Module):
    """Simple gated FFN (AlphaFold style)."""

    def __init__(self, dim: int, ffn_mult: int = 4):
        super().__init__()
        hidden = dim * ffn_mult
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(self.norm(x))))


# ----------------------------------------------------------------------------
# Row attention with pair bias
# ----------------------------------------------------------------------------


class RowAttentionWithPairBias(nn.Module):
    """Attend across joints (J axis), biased by the pair tensor.

    MSA is [B, T, J, H]. For each (batch, time) we compute J-way self-attention
    whose attention logits receive an additive bias from pair [B, J, J, heads].
    """

    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.heads = cfg.heads
        self.head_dim = cfg.hidden // cfg.heads
        self.use_pair_bias = cfg.use_pair

        self.norm_msa = nn.LayerNorm(cfg.hidden)
        self.to_qkv = nn.Linear(cfg.hidden, cfg.hidden * 3, bias=False)
        self.gate = nn.Linear(cfg.hidden, cfg.hidden)
        self.to_out = nn.Linear(cfg.hidden, cfg.hidden)
        # Pair-bias machinery only exists in variants that use the pair tensor
        if self.use_pair_bias:
            self.norm_pair = nn.LayerNorm(cfg.pair_hidden)
            self.pair_bias = nn.Linear(cfg.pair_hidden, cfg.heads, bias=False)
        else:
            self.norm_pair = None
            self.pair_bias = None

    def forward(self, msa: torch.Tensor, pair: torch.Tensor | None) -> torch.Tensor:
        B, T, J, H = msa.shape

        msa_n = self.norm_msa(msa)

        qkv = self.to_qkv(msa_n)                                    # [B, T, J, 3H]
        q, k, v = rearrange(
            qkv, "b t j (three heads d) -> three b t heads j d",
            three=3, heads=self.heads,
        )                                                           # each [B, T, heads, J, d]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bthid,bthjd->bthij", q, k) * scale  # [B, T, heads, J, J]

        if self.use_pair_bias and pair is not None:
            pair_n = self.norm_pair(pair)
            bias = self.pair_bias(pair_n)                           # [B, J, J, heads]
            bias = rearrange(bias, "b i j h -> b h i j")             # [B, heads, J, J]
            attn_logits = attn_logits + bias[:, None, :, :, :]       # broadcast across T

        attn = attn_logits.softmax(dim=-1)
        out = torch.einsum("bthij,bthjd->bthid", attn, v)            # [B, T, heads, J, d]
        out = rearrange(out, "b t h j d -> b t j (h d)")

        gate = torch.sigmoid(self.gate(msa_n))
        return self.to_out(out * gate)


# ----------------------------------------------------------------------------
# Column attention (per-joint temporal self-attention)
# ----------------------------------------------------------------------------


class ColumnAttention(nn.Module):
    """Attend across time (T axis), independently per joint."""

    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.hidden // cfg.heads
        self.norm = nn.LayerNorm(cfg.hidden)
        self.to_qkv = nn.Linear(cfg.hidden, cfg.hidden * 3, bias=False)
        self.gate = nn.Linear(cfg.hidden, cfg.hidden)
        self.to_out = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        B, T, J, H = msa.shape
        msa_n = self.norm(msa)
        qkv = self.to_qkv(msa_n)
        q, k, v = rearrange(
            qkv, "b t j (three heads d) -> three b j heads t d",
            three=3, heads=self.heads,
        )
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bjhtd,bjhsd->bjhts", q, k) * scale
        attn = attn_logits.softmax(dim=-1)
        out = torch.einsum("bjhts,bjhsd->bjhtd", attn, v)
        out = rearrange(out, "b j h t d -> b t j (h d)")
        gate = torch.sigmoid(self.gate(msa_n))
        return self.to_out(out * gate)


# ----------------------------------------------------------------------------
# Outer product mean: MSA -> pair update
# ----------------------------------------------------------------------------


class OuterProductMean(nn.Module):
    """Update pair tensor from MSA via outer product mean over the time axis.

    For each (i, j) joint pair, we compute the average over time of outer
    products of projected MSA features, then project to pair hidden.
    """

    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        self.chunk = cfg.opm_chunk
        self.norm = nn.LayerNorm(cfg.hidden)
        self.proj_a = nn.Linear(cfg.hidden, self.chunk, bias=False)
        self.proj_b = nn.Linear(cfg.hidden, self.chunk, bias=False)
        self.out = nn.Linear(self.chunk * self.chunk, cfg.pair_hidden)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        B, T, J, H = msa.shape
        msa = self.norm(msa)
        a = self.proj_a(msa)  # [B, T, J, chunk]
        b = self.proj_b(msa)  # [B, T, J, chunk]
        # Outer product mean over T axis
        outer = torch.einsum("btic,btjd->bijcd", a, b) / T  # [B, J, J, c, c]
        outer = rearrange(outer, "b i j c d -> b i j (c d)")
        return self.out(outer)


# ----------------------------------------------------------------------------
# Triangle multiplicative update (outgoing)
# ----------------------------------------------------------------------------


class TriangleMultiplicativeOutgoing(nn.Module):
    """Outgoing triangle: pair(i, j) is updated using Σ_k a(i, k) * b(j, k)."""

    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        c = cfg.tri_hidden
        self.norm_in = nn.LayerNorm(cfg.pair_hidden)
        self.linear_a = nn.Linear(cfg.pair_hidden, c)
        self.linear_a_g = nn.Linear(cfg.pair_hidden, c)
        self.linear_b = nn.Linear(cfg.pair_hidden, c)
        self.linear_b_g = nn.Linear(cfg.pair_hidden, c)
        self.norm_out = nn.LayerNorm(c)
        self.linear_out = nn.Linear(c, cfg.pair_hidden)
        self.linear_gate = nn.Linear(cfg.pair_hidden, cfg.pair_hidden)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: [B, J, J, H_pair]
        p = self.norm_in(pair)
        a = self.linear_a(p) * torch.sigmoid(self.linear_a_g(p))
        b = self.linear_b(p) * torch.sigmoid(self.linear_b_g(p))
        # Outgoing: new(i, j) = Σ_k a(i, k) * b(j, k)
        out = torch.einsum("bikc,bjkc->bijc", a, b)
        out = self.norm_out(out)
        gate = torch.sigmoid(self.linear_gate(p))
        return gate * self.linear_out(out)


# ----------------------------------------------------------------------------
# Triangle attention (starting node)
# ----------------------------------------------------------------------------


class TriangleAttentionStarting(nn.Module):
    """Triangle attention around the starting node.

    For each (i, j), attend over k using queries from pair(i, j) and
    keys/values from pair(i, k). The pair tensor itself provides an extra bias.
    """

    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        self.heads = cfg.pair_heads
        self.head_dim = cfg.pair_hidden // cfg.pair_heads
        self.norm = nn.LayerNorm(cfg.pair_hidden)
        self.to_qkv = nn.Linear(cfg.pair_hidden, cfg.pair_hidden * 3, bias=False)
        self.bias = nn.Linear(cfg.pair_hidden, cfg.pair_heads, bias=False)
        self.gate = nn.Linear(cfg.pair_hidden, cfg.pair_hidden)
        self.to_out = nn.Linear(cfg.pair_hidden, cfg.pair_hidden)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: [B, J, J, H_pair]
        p = self.norm(pair)
        qkv = self.to_qkv(p)
        q, k, v = rearrange(
            qkv, "b i j (three heads d) -> three b heads i j d",
            three=3, heads=self.heads,
        )
        # For each i, attend (j -> k) where queries are pair(i, j) and keys are pair(i, k)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bhijd,bhikd->bhijk", q, k) * scale

        bias = self.bias(p)                           # [B, J, J, heads]
        bias = rearrange(bias, "b i k h -> b h i k")  # [B, heads, J, J]
        attn_logits = attn_logits + bias[:, :, :, None, :]  # broadcast over j

        attn = attn_logits.softmax(dim=-1)
        out = torch.einsum("bhijk,bhikd->bhijd", attn, v)
        out = rearrange(out, "b h i j d -> b i j (h d)")
        gate = torch.sigmoid(self.gate(p))
        return self.to_out(out * gate)


# ----------------------------------------------------------------------------
# MotionFormer block
# ----------------------------------------------------------------------------


class MotionFormerBlock(nn.Module):
    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.row_attn = RowAttentionWithPairBias(cfg)
        self.col_attn = ColumnAttention(cfg)
        self.msa_transition = Transition(cfg.hidden, cfg.ffn_mult)

        # Pair-side modules only exist when the pair tensor is present
        self.opm = OuterProductMean(cfg) if (cfg.use_pair and cfg.use_opm) else None
        self.tri_mult = TriangleMultiplicativeOutgoing(cfg) if (cfg.use_pair and cfg.use_triangle) else None
        self.tri_attn = TriangleAttentionStarting(cfg) if (cfg.use_pair and cfg.use_triangle) else None
        # Pair transition runs when pair exists AND at least one pair-update is on
        self.pair_transition = (
            Transition(cfg.pair_hidden, cfg.ffn_mult)
            if (cfg.use_pair and (cfg.use_opm or cfg.use_triangle))
            else None
        )

    def forward(self, msa: torch.Tensor, pair: torch.Tensor | None):
        msa = msa + self.row_attn(msa, pair)
        msa = msa + self.col_attn(msa)
        msa = msa + self.msa_transition(msa)

        if pair is not None:
            if self.opm is not None:
                pair = pair + self.opm(msa)
            if self.tri_mult is not None:
                pair = pair + self.tri_mult(pair)
            if self.tri_attn is not None:
                pair = pair + self.tri_attn(pair)
            if self.pair_transition is not None:
                pair = pair + self.pair_transition(pair)

        return msa, pair


# ----------------------------------------------------------------------------
# Full MotionFormer (matches baseline I/O)
# ----------------------------------------------------------------------------


class MotionFormer(nn.Module):
    def __init__(self, cfg: MotionFormerConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.C, cfg.hidden)
        self.mask_embed = nn.Parameter(torch.randn(cfg.hidden) * 0.02)
        self.time_pe = nn.Parameter(torch.randn(cfg.T, cfg.hidden) * 0.02)
        self.joint_pe = nn.Parameter(torch.randn(cfg.J, cfg.hidden) * 0.02)

        # Pair tensor initialisation from joint-joint relative distance encoding
        # (analogue of AlphaFold's relpos / positional encoding for pairs).
        # Here we learn a per-(i, j) embedding directly since J is small.
        if cfg.use_pair:
            self.pair_init = nn.Parameter(torch.randn(cfg.J, cfg.J, cfg.pair_hidden) * 0.02)
        else:
            self.pair_init = None

        self.blocks = nn.ModuleList([
            MotionFormerBlock(cfg) for _ in range(cfg.depth)
        ])

        self.output_norm = nn.LayerNorm(cfg.hidden)
        # 9-dim output per (t, j): 3 pos + 6D rotation (Zhou et al. 2019)
        self.output_proj = nn.Linear(cfg.hidden, cfg.C + 6)
        # Root prediction head: pool the MSA over joint axis and map to [T, 3]
        self.root_proj = nn.Linear(cfg.hidden, 3)

    def forward(
        self,
        motion: torch.Tensor,          # [B, T, J, C]
        mask: torch.Tensor | None,     # [B, T, J]
    ) -> dict:
        """Returns dict with keys pos, rot6d, root (all normalised output space)."""
        B, T, J, C = motion.shape

        msa = self.input_proj(motion)
        msa = msa + self.time_pe[None, :, None, :] + self.joint_pe[None, None, :, :]
        if mask is not None:
            msa = torch.where(mask.unsqueeze(-1), self.mask_embed, msa)

        if self.pair_init is not None:
            pair = self.pair_init.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        else:
            pair = None

        for block in self.blocks:
            msa, pair = block(msa, pair)

        msa = self.output_norm(msa)
        self._last_pair = pair

        per_joint = self.output_proj(msa)                 # [B, T, J, 3+6]
        pos = per_joint[..., :self.cfg.C]                 # [B, T, J, 3]
        rot6d = per_joint[..., self.cfg.C:]               # [B, T, J, 6]

        # Root: pool MSA over joint axis, then project. Time-varying root.
        msa_pool = msa.mean(dim=2)                        # [B, T, H]
        root = self.root_proj(msa_pool)                   # [B, T, 3]

        return {"pos": pos, "rot6d": rot6d, "root": root}

    def extract_pair_structure(self) -> torch.Tensor:
        """Extract a symmetric [J × J] affinity from the pair tensor.

        Returns a zero matrix for variants without a pair tensor (axial_only).
        """
        if self._last_pair is None:
            return torch.zeros(self.cfg.J, self.cfg.J, device=self.pose_embed_device())
        pair = self._last_pair.mean(dim=0)   # [J, J, H]
        scalar = pair.norm(dim=-1)           # [J, J]
        scalar = 0.5 * (scalar + scalar.t())
        return scalar

    def pose_embed_device(self) -> torch.device:
        return self.input_proj.weight.device


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = MotionFormerConfig()
    model = MotionFormer(cfg)
    dummy = torch.randn(2, cfg.T, cfg.J, cfg.C)
    dummy_mask = torch.zeros(2, cfg.T, cfg.J, dtype=torch.bool)
    dummy_mask[:, ::4, ::3] = True
    out = model(dummy, dummy_mask)
    print(f"MotionFormer params: {count_params(model) / 1e6:.2f}M")
    print(f"Input:  {dummy.shape}")
    for k, v in out.items():
        print(f"  out.{k}: {tuple(v.shape)}")
