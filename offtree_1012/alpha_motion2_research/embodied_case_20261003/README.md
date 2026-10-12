# Real embodiment case assets

The primary executed case is **Unitree G1 → Unitree H2**, using three frames of the same real XWBC-retargeted LAFAN clip, `fallAndGetUp3_subject1`: frames **1800, 2000, 2600** at 30 fps. G1 images are stored reference motion. H2 images show an actual trained AlphaMotion-1 forward followed by rotation-derived FK and rigid-link rendering. These are three independent single-frame evaluations, not generated temporal motion or a physics/control result.

## Start here

| Material | Location | What actually ran |
|---|---|---|
| G1/H2 mesh, skeleton, closeup and camera projection | `geometry/` | Real mesh assets; reference G1 qpos and predicted H2 per-link transforms |
| Full model inputs, internal tensors and output | `alpha1/trace.json`, `alpha1/trace.npz` | CPU Megatron inference, strict checkpoint load, real source-contact input |
| Three robot joint-name text examples | `alpha1/joint_name_semantics_trace.*` | Offline SigLIP text forward, L2 normalization and trained 768→256 semantic projection |
| Go2 and Panda reference-pose numeric cards | `alpha1/alternative_input_cards_trace.*` | Actual model keyframe/FK and descriptors; no motion-network decode for these two bodies |
| Attention, embedding and FSQ figures | `code_audit/` | Actual captured tensors, with Q/K/V provenance and per-head normalization |
| UniMate native sample preprocessing | `unimate/` | Actual UniML3D sample and official feature processing; see its own execution status |

## Keep one instance connected

G1 skeleton node `left_elbow_link` is input index 23 and corresponds to the physical `left_elbow_joint`. The H2 descriptor uses `left_elbow_link` at index 22. Each `*_projection.json` provides the actual rendered joint anchor and axis endpoint in image pixels, camera parameters and closeup crop coordinates. Use these final JSON files when placing diagram arrows; do not estimate the anchor visually.

The source model input has 9 pose channels: global rotation represented by the first two matrix columns concatenated, then root-relative FK position divided by the training-cache reach. The exact G1 reach is **87.42538452148438 cm**, scanned over both full caches in the actual checkpoint config. Geometry/mechanics contributes 34 features; joint-name text contributes a normalized 768-dimensional SigLIP vector. Their learned projection is 256-dimensional. The signed contact-distance channel is geometric, not measured force or tactile data. All source joints are visible in this case.

The actual model has width 256, 8 heads, 4 encoder layers, 4 decoder layers, 4 position slots and 128 rotation slots. Both FSQ streams have 13 dimensions configured with 32 levels each. The exported attention uses actual trained projections and inputs; algebraically reconstructed outputs match captured module outputs to within **7.2e-7**. Every floating tensor in the exported trace is finite.

## Output boundaries

- H2 uses a descriptor rebuilt from real H2 assets. The older training cache's anonymous `body00…body31` tree is not silently bound to the official mesh. This demonstration does not establish unseen-body generalization.
- Decoder rotation6 is an unconstrained network output. Gram–Schmidt converts it to rotation matrices; target bone offsets then give FK positions. The mesh view uses these per-link transforms without IK or a hinge-limit projection. One ambiguous merged `waist_roll_link` mesh is omitted; the torso and full descriptor skeleton remain. This is not a valid robot command by itself.
- `predicted_position_head_raw[:, 0, 0]` is directly supervised as root height above ground in meters. Root channels 1–2 and nonroot position-head channels have no direct coordinate/velocity target in this configuration; they receive indirect cycle gradients. **Use exported rotation-FK positions for diagram geometry, not these raw head values.**
- Go2 and Franka Emika **Panda**, not FR3, are alternative input cards using their actual model keyframes. They are not the same recorded motion as G1, and the model does not concatenate three bodies as one input. Panda's finger slides are not established as supported by the existing hinge-oriented mechanical descriptor.
- The exact text prompts contain the actual link names, including `the link4 joint of a body`. Neither AlphaMotion-1 nor this UniMate trace has a mesh/surface text encoder. Surface-region semantics remain a proposal, not executed functionality.

## Provenance and reproduction

`case-manifest.json` records file hashes and classifications. The geometry and UniMate sub-manifests record source asset revisions, conversions, licensing and coordinate conventions. Only derived images and small selected-frame traces are copied into this local delivery; private weights and full motion sources remain on ArenaLabs.

The isolated remote research directory is `/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/embodied_case_20261003`. The Alpha1 forward script is `alpha1/build_alpha1_trace.py`, run with the existing `/home/arenalabs/miniconda3/envs/ardy/bin/python`, CPU and two threads. No training, model-source edit, service restart or download-queue change was performed.
