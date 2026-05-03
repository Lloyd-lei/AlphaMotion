"""
Baseline: standard Transformer encoder for Masked Motion Modeling.

Treats the motion [T, J, C] as a flattened sequence of T*J tokens, each of which
is an embedding of the (time-index, joint-index, channel-values) triple. A
standard multi-head self-attention stack then processes the whole sequence.

This is the sequence-first baseline we compare against MotionFormer's pair-first
architecture. Both models have matching parameter counts and training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange


@dataclass
class BaselineConfig:
    T: int = 64
    J: int = 20
    C: int = 3
    hidden: int = 128
    depth: int = 6
    heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0


class BaselineTransformer(nn.Module):
    """Flatten [T, J, C] into T*J tokens, run self-attention."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg

        # Per-joint channel projection
        self.input_proj = nn.Linear(cfg.C, cfg.hidden)
        # A learned flag that says "this position was masked" (similar to BERT [MASK])
        self.mask_embed = nn.Parameter(torch.randn(cfg.hidden) * 0.02)
        # Learned positional encodings, one for time one for joint
        self.time_pe = nn.Parameter(torch.randn(cfg.T, cfg.hidden) * 0.02)
        self.joint_pe = nn.Parameter(torch.randn(cfg.J, cfg.hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden * cfg.ffn_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)

        self.output_norm = nn.LayerNorm(cfg.hidden)
        self.output_proj = nn.Linear(cfg.hidden, cfg.C + 6)   # 3 pos + 6D rotation
        self.root_proj = nn.Linear(cfg.hidden, 3)

    def forward(
        self,
        motion: torch.Tensor,         # [B, T, J, C]
        mask: torch.Tensor | None,    # [B, T, J]
    ) -> dict:
        B, T, J, C = motion.shape
        assert (T, J, C) == (self.cfg.T, self.cfg.J, self.cfg.C)

        x = self.input_proj(motion)
        x = x + self.time_pe[None, :, None, :]
        x = x + self.joint_pe[None, None, :, :]
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)

        x = rearrange(x, "b t j h -> b (t j) h")
        x = self.encoder(x)
        x = rearrange(x, "b (t j) h -> b t j h", t=T, j=J)
        x = self.output_norm(x)

        per_joint = self.output_proj(x)                    # [B, T, J, 3+6]
        pos   = per_joint[..., :self.cfg.C]                # [B, T, J, 3]
        rot6d = per_joint[..., self.cfg.C:]                # [B, T, J, 6]
        root  = self.root_proj(x.mean(dim=2))              # [B, T, 3]
        return {"pos": pos, "rot6d": rot6d, "root": root}

    def extract_pair_structure(self) -> torch.Tensor:
        """Extract an implicit [J × J] pair structure from the trained baseline.

        We compute joint-joint correlations through the learned joint position
        embeddings, which is the closest analogue a sequence-Transformer has to
        an explicit pair tensor.
        """
        # [J, H] -> correlation matrix [J, J]
        jpe = self.joint_pe
        jpe_centred = jpe - jpe.mean(dim=0, keepdim=True)
        cov = jpe_centred @ jpe_centred.t()
        std = torch.sqrt(torch.diagonal(cov).clamp_min(1e-8))
        return cov / (std[:, None] * std[None, :] + 1e-8)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = BaselineConfig()
    model = BaselineTransformer(cfg)
    dummy = torch.randn(2, cfg.T, cfg.J, cfg.C)
    dummy_mask = torch.zeros(2, cfg.T, cfg.J, dtype=torch.bool)
    dummy_mask[:, ::4, ::3] = True
    out = model(dummy, dummy_mask)
    print(f"Baseline params: {count_params(model) / 1e6:.2f}M")
    for k, v in out.items():
        print(f"  out.{k}: {tuple(v.shape)}")
    print(f"Implicit pair structure: {model.extract_pair_structure().shape}")
