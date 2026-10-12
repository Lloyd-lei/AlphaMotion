# Hand surface and skeleton case assets

This is a **constructed geometric illustration**, using the real **Allegro Hand V3 right hand** and **Franka Emika Panda** models around the same analytic sphere of radius **30 mm**. It is not recorded training data, a model prediction, force closure or a physical grasp.

## Start here

- `allegro_mesh_skeleton.png` and `panda_mesh_skeleton.png`: full hand/parallel gripper with object-frame axes. Corresponding mesh, transparent, skeleton and sampled-surface layers are provided.
- `panda_arm_context_mesh_skeleton.png`: full Panda arm reference home pose with the same constructed finger opening. Its scene orientation differs from the standalone hand view and is documented separately.
- `object_frame_contact_comparison.png/.svg`: both surface patch sets expressed on the same 30 mm sphere in identical object coordinates. It compares coverage; it is not a learned one-to-one map.
- `*_trace.json`: actual camera, joint axes/coordinates/limits/units, samples, pixels, crop coordinates and geometric definitions.
- `*_surface_samples.npz` plus `*_compiled_mesh_binding.npz`: deterministic area-weighted triangle/barycentric samples and exact compiled binding bases.
- `*_contact_N_crop.png`, `selected_surface_examples.json`: selected linked numeric examples and local views.
- `umr_reference/`: actual SMPL-X neutral template hand and G1 fixed rubber hand, as separate constructed references. UMR optimization was not executed.

## What the labels mean

Each surface point records its link, compiled triangle index, barycentric coordinate, link-local position/normal, model-world position, object-frame position and camera pixel. The nearest sphere point is analytic. **d = body surface point − nearest object point**, pointing from object to body. It is a geometric vector, not force.

Signed gap is norm(body point − sphere center) − radius; unsigned distance is its absolute value. `penetration` is signed gap < 0. `valid` means the numeric geometry and triangle are valid. The binary `contact_geometry` proxy is valid AND abs(signed gap) ≤ 2 mm. These labels indicate thresholded proximity, not measured pressure/touch.

Final exact triangle checks give Allegro minimum gap **0.350 mm** over all visual meshes, fingertip minima approximately **0.800 mm**; Panda minimum gap **0.600 mm**. There is no sphere penetration in these constructed poses. Self-collision, friction, torques and grasp stability are untested. Surface and skeleton dots are visible overlays, not visibility/occlusion labels.

Allegro has **J=16**, **D=16**, four fingers, hinge q in radians. Panda has **J=2 prismatic coordinates**, **D=1 symmetric opening**, and one actuator. Its equality enforces `finger_joint1 = finger_joint2`; tendon split is `0.5*q1 + 0.5*q2`. Both q are 0.03053730974845663 m. The joint-coordinate opening is 61.0746195 mm; the actual visual pad aperture measured by ray/triangle intersection at the sphere center is **64.0591623 mm**. These definitions differ because of pad geometry. No actuator command was executed.

## Provenance and limits

Allegro and reused Panda assets are pinned to MuJoCo Menagerie revision `4d038b3feae26ec82b46a4d586379114012a8ac7`. All 15 downloaded Allegro files (836,387 bytes) and 12 reused Panda source files were matched against pinned Git blob hashes. Model-specific BSD-2-Clause and Apache-2.0 licenses are copied. Model conversion provenance remains in the source acquisition records.

SMPL-X remains the existing authorized local model; only derived hand pictures and selected geometry are delivered. G1 has no articulated finger DOFs. See its separate reference manifest for licensing, neutral-template and open local-mesh boundaries.

The isolated reproduction scripts are included. The remote root is `/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/hand_surface_case_20261004`; `derived/` holds the Allegro/Panda delivery and `umr_reference/` its separate subset. No UMR optimization, neural model or training ran for these images. Any later Megatron training is a separate experiment and does not change the status of these geometric illustrations.
