# Architecture figure source map

The five subjects each have a compact paper layout and a detailed poster layout, all original vector redrawings from the audited implementation. They are not adapted screenshots of paper figures. `.svg` files retain editable text; `.pdf` files retain vectors and embedded fonts. `paper_*.png` files are 240 dpi document insertion images; `*_preview.png` files are 90 dpi visual-review previews. `build_figures.py` is the reproducible drawing source.

## Physical figure size

The `paper_*.pdf` files are natively 510 pt wide. Body text is at least 8 pt, with source/footer notes at least 7.1 pt at that physical width. The non-prefixed detailed layouts are 1140 pt wide and are intended for zoomable reference or larger-page appendices; they should not be reduced directly to paper-column width. Insert the compact PDFs at their intended size; shrinking them also shrinks their font sizes. If a publication requires a narrower width, reflow the compact drawing rather than simply scaling the detailed poster.

## Version identity

- AlphaMotion-1 code snapshot: `pose_translator_design/exp_topo` from ArenaLabs research checkout `research-0909`, recorded HEAD `692bf7704ebc0d4e2b12678c1bc0c011d426fe0b`. Exact copied-file SHA-256 values are in the adjacent `source-manifest.json`.
- AlphaMotion-1 figure configuration: live `megatron` deployment / source run `L_M20k_0903`; independently retrieved `verified-megatron-config.json` beside this directory. The figure uses the effective settings, not constructor defaults: width 256, 8 heads, 4 encoder and 4 decoder layers, position slots 4, rotation slots 128, single FSQ stage, 13 scalar dimensions configured at 32 levels, SigLIP, DOF features on, contact channel on, `root_head=false`, `root_pos=true`, `vis_residual=false`, `n_judges=16`.
- UniMate: [Friedrich-M/UniMate, pinned source](https://github.com/Friedrich-M/UniMate/tree/2c5b384715aa63d8639b1ed7eb74bfe614570c7a).
- UMR: [hanyang9/UMR, pinned source](https://github.com/hanyang9/UMR/tree/d1cdaedaf67a827198da71c5fa831f48b2e51f91).
- Figure 5 is proposed research architecture. It has no implementation commit or validated model result.

## AlphaMotion-1 source IDs

Paths below are relative to the audit snapshot `source-snapshot/AlphaMotion1`, not the empty AlphaMotion-2 scaffolding.

| ID | Source | What the figure encodes |
|---|---|---|
| A1 | `model_dual.py:28-119,156-174`; `model.py:167-210`; `semantics.py` | Pose embedding, hidden-pose replacement, contact after masking, geometry 15 + DOF 19 descriptor, frozen SigLIP joint-name features and semantic MLP. |
| A2 | `model.py:213-249` | Per-head hop-distance bias; pre-LN residual attention and 4D FFN. |
| A3 | `model_dual.py:78-95,121-141`; `model.py:43-95` | Learned dual query pools and effective single-stage FSQ implementation, including STE rounding and projection widths. |
| A4 | `model_dual.py:96-119,197-251` | Independent position/rotation decoders; position, rotation and contact output modules. No independent root-height head in the chosen checkpoint. |
| A5 | `model.py:252-264` | CrossBlock normalization, MHA and residual FFN. |
| A6 | `model_dual.py:13-24`; effective config | Training auxiliary JudgeHead: stream mean/max concatenation, 1024 -> 256 -> 16. |
| A7 | `train_dual.py:438-453`; `verified-megatron-config.json` | Actual root_pos loss supervises only root-position coordinate 0 as height. Contact BCE and contact-maintenance weights are zero; the existence of a contact head is not an accuracy claim. |

Important implementation caveats: scalar FSQ levels are shown as the configured values. The figure does not infer total discrete cardinality from even-valued level counts and finite-precision rounding. Training-time scalar noise, when configured, is not an inference layer. Frozen external text towers are drawn at their embedding interfaces; their transformer internals are not new AlphaMotion-1 layers.

## UniMate source IDs

| ID | Pinned source | What the figure encodes |
|---|---|---|
| U1 | [Configuration](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/configs/uniml3d_60frames_graph_adaln.json), [motion feature computation](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/utils/motion_utils.py#L35-L179) | Effective T=60, max J=60, graph/adaLN/flow recipe and 12-channel motion representation. Rotations follow the source's T-pose-relative convention and reordered parent-slot semantics; root channels are special. |
| U2 | [InputLayer](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L209-L298) | Independent root/joint motion and T-pose MLPs, parent T-pose fusion and prefix frame. |
| U3 | [Graph attention and SpatioTemporalBlock](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L97-L343) | Spatial then temporal then SwiGLU, QK normalization, nine-way modulation and gated residuals. |
| U4 | [FinalLayer](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L460-L540), [forward path](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/graph_adaln.py#L62-L108) | Root-query cross-attention, separate heads, prefix removal and velocity layout. |
| U5 | [Timestep and TposPool](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L81-L206), [condition and token additions](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L330-L369) | Time/caption/T-pose condition, depth and joint-name embeddings. |
| U6 | [Graph attention bias](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L25-L59) | Hop and relation embeddings, two projection branches and per-head additive bias. |
| U7 | [RMSNorm and SwiGLU](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L50-L74) | fp32 RMS normalization and gated FFN. Actual hidden width is 1365, from int(2/3*2048). |
| U8 | [Flow training losses](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L218-L308), [ODE sampler](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L476-L508) | Noise-to-data straight path, velocity loss, clean reconstruction and ODE inference. Sampling step settings are not labeled as a measured NFE. |
| U9 | [SignNet](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/rope.py#L197-L244), [SpectralJointRoPE](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/rope.py#L307-L414) | Sign-invariant spectral encoding of eight Laplacian modes and per-head rotation angles. |

## UMR source IDs

| ID | Pinned source | What the figure encodes |
|---|---|---|
| R1 | [Surface sampling](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/build_correspondence_ae_dataset.py#L346-L540), [effective defaults](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/humanoid_retarget_defaults.json#L37-L81) | 4096 exterior surface points and canonical template preparation. |
| R2 | [PointNetTemplateResidualAE and losses](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L244-L397) | Shared Conv1D, BN/ReLU, max pooling, global MLP residual decoder and four objective terms. |
| R3 | [Slot binding](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L657-L712) | Learned source/target surface slots attached to meshes and geometric motion. |
| R4 | [Per-frame pose fitting](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1608-L2117) | FK/Jacobians, weighted geometric residuals, inequalities, integration and limits. |
| R5 | [Python recipe](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/humanoid_retarget_defaults.json#L61-L81), [optimizer](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L644-L679) | Actual pipeline 500 epochs / batch 2, loss weights and AdamW; not standalone CLI defaults or browser recipe. |
| R6 | [Clarabel step](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/smpl_surface_retarget_common.py#L1556-L1751) | Quadratic objective, box and linear inequality constraints, optional second-order-cone step limits. |
| R7 | [Slot compatibility](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L420-L464), [train or reuse](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L890-L942), [pipeline stages](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L1034-L1056) | Compatible slots are reused; new/missing template slots invoke training. |

The browser-demo note follows the separately audited website branch `gh-pages` at `b9d749900805fad4700df7ba9e229d57961be560`, `static/js/browser-umr-runtime-exact-v1.js:138-305`, which invokes the current CPU/WASM correspondence training path for 60 epochs. This is supplementary runtime context, not the neural recipe depicted in Figure 4.

## Figure 5 design status

The following are research proposals rather than extracted implementation: semantic surface patch tokens; bidirectional skeleton-surface fusion; a shared target-conditioned joint/surface decoder; local text alignment and combined geometric objectives; prospective tactile tokens. The thesis is that local semantic anchors plus local geometry may permit surface-only fixed-weight correspondence to unseen embodiments. The required full AlphaMotion-2 model couples this surface path with skeleton information. Neither the thesis nor improved performance over UniMate has been validated. Width, depth, loss weights, FSQ use and flow/diffusion choice are intentionally left open.
