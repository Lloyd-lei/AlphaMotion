"""
Baseline: standard Transformer encoder for Masked Motion Modeling.

Treats the motion [T, J, C] as a flattened sequence of T*J tokens, each of which
is an embedding of the (time-index, joint-index, channel-values) triple. A
standard multi-head self-attention stack then processes the whole sequence.

This is the sequence-first baseline we compare against MotionFormer's pair-first
architecture. Both models have matching parameter counts and training recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange


def _sinusoidal_t_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


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

        # Stage-1 FM: per-joint vector = pos(3) + rot6d(6) + root_broadcast(3) = 12 channels
        IN_CH_FM = cfg.C + 6 + 3
        self.input_proj = nn.Linear(IN_CH_FM, cfg.hidden)
        self.mask_embed = nn.Parameter(torch.randn(cfg.hidden) * 0.02)
        self.time_pe = nn.Parameter(torch.randn(cfg.T, cfg.hidden) * 0.02)
        self.joint_pe = nn.Parameter(torch.randn(cfg.J, cfg.hidden) * 0.02)

        # Time conditioning (FM)
        self.t_embed_dim = 128
        self.t_mlp = nn.Sequential(
            nn.Linear(self.t_embed_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
        )

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
        x_pos_t:  torch.Tensor,        # [B, T, J, 3]
        x_rot_t:  torch.Tensor,        # [B, T, J, 6]
        x_root_t: torch.Tensor,        # [B, T, 3]
        t:        torch.Tensor,        # [B]   in [0, 1]
        mask:     torch.Tensor | None = None,  # [B, T, J] inpainting
    ) -> dict:
        """Conditional flow-matching baseline. Returns velocity fields."""
        B, T, J, _ = x_pos_t.shape

        x_root_b = x_root_t.unsqueeze(2).expand(B, T, J, 3)
        x_in = torch.cat([x_pos_t, x_rot_t, x_root_b], dim=-1)         # [B, T, J, 12]

        x = self.input_proj(x_in)
        x = x + self.time_pe[None, :, None, :]
        x = x + self.joint_pe[None, None, :, :]

        # Time conditioning
        t_emb  = _sinusoidal_t_embed(t, self.t_embed_dim)               # [B, t_embed_dim]
        t_proj = self.t_mlp(t_emb)                                       # [B, hidden]
        x = x + t_proj.view(B, 1, 1, -1)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)

        x = rearrange(x, "b t j h -> b (t j) h")
        x = self.encoder(x)
        x = rearrange(x, "b (t j) h -> b t j h", t=T, j=J)
        x = self.output_norm(x)

        per_joint = self.output_proj(x)
        v_pos = per_joint[..., :self.cfg.C]
        v_rot = per_joint[..., self.cfg.C:]
        v_root = self.root_proj(x.mean(dim=2))
        return {"pos": v_pos, "rot6d": v_rot, "root": v_root}

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
