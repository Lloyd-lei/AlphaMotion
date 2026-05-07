"""Output heads for AlphaMotion.

Stage 1 uses a deterministic regression head living inside MotionFormer/Baseline
(see `motionformer.py` decoder + output projection).

Stage 2 introduces a flow-matching action head (this file). Currently a stub —
NB03 documents the design and runs an interface dry-run; implementation lands
when M2 starts (see notebook/03_stage2_sketch.ipynb §5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatchingHead(nn.Module):
    """Conditional flow-matching head over per-frame actions.

    Trunk-conditioned, π0-style: given a pair tensor produced by a frozen
    MotionFormer trunk, predict an action velocity field; sample by
    integrating from t=0 to t=1.

    Design (from NB03 §3):
      - Conditional flow matching (rectified-flow), not diffusion.
      - Time embedding fused at every block.
      - Inference: ≤16 sampling steps. Training: random t∈[0,1].
    """

    def __init__(self, pair_dim: int, action_dim: int):
        super().__init__()
        self.pair_dim   = pair_dim
        self.action_dim = action_dim
        # NOTE: layers intentionally not built yet — see NB03 §3.1.

    def forward(
        self,
        pair_tensor: torch.Tensor,        # [B, J, J, pair_dim]
        t_embed:     torch.Tensor,        # [B, time_embed_dim]
        x_t:         torch.Tensor,        # [B, T, action_dim]
    ) -> torch.Tensor:                    # action velocity, same shape as x_t
        raise NotImplementedError("FlowMatchingHead is a Stage-2 stub (NB03 M2)")

    def loss(
        self,
        pair_tensor:   torch.Tensor,
        action_target: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("FlowMatchingHead is a Stage-2 stub (NB03 M2)")

    @torch.no_grad()
    def sample(
        self,
        pair_tensor: torch.Tensor,
        n_steps: int = 16,
    ) -> torch.Tensor:
        raise NotImplementedError("FlowMatchingHead is a Stage-2 stub (NB03 M2)")
