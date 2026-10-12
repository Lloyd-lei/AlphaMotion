# UniMate redraw source map

All neural paths use the `Graph AdaLN` configuration from UniMate commit `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`. These are original source-audited redrawings, not copied paper figures.

| ID | Source | Drawn computation |
|---|---|---|
| U1 | [Default configuration](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/configs/uniml3d_60frames_graph_adaln.json) | T=60, max J=60, d=512, 8 heads, 10 blocks, FLAN-T5-base, Graph AdaLN, flow training. |
| U2 | [InputLayer](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L209-L298) | Separate root/joint motion and T-pose MLPs; parent fusion; prefix construction. |
| U3 | [Spatiotemporal blocks](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L97-L343) | Spatial, temporal, FFN order; pre-RMS; QK RMS; gates and nine-vector per-block modulation. |
| U4 | [FinalLayer](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L460-L540), [forward](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/graph_adaln.py#L62-L108) | Root-query aggregation with K=RMS(joints), V=joints; two heads; prefix removal. |
| U5 | [TimestepEmbedder and TposPool](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L81-L206), [condition additions](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L330-L369) | tau scalar -> sinusoidal 256 -> Linear512 -> SiLU -> Linear512; caption and T-pose pooling; depth/name features. |
| U6 | [Graph bias](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L25-L59), [spectral RoPE](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/rope.py#L197-L414) | Hop/relation bias and SignNet spectral rotation of Q/K. |
| U7 | [RMSNorm and SwiGLU](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L50-L74) | RMS epsilon=1e-6; SwiGLU actual inner width 1365. |
| U8 | [Flow objective](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L218-L308), [ODE sampling](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L476-L508) | Velocity prediction and outer integration loop; these are distinct from the 10 network blocks. |

The main figure abbreviates TposPool as a four-query operator. Its actual internals are learned queries, RMS cross-attention + residual, RMS SwiGLU + residual, then query mean, RMS and linear projection. This abbreviation is not a claim that global pooling is a single linear layer.
