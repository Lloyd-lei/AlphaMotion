# WP-ASSETS: dense surface assets (surface_v2)

This package builds dense surface point clouds at UMR density for all 18 AM2 bodies (16 XWBC, adroit_pen and relay_franka_full_v2). Each body gets 4096 exterior points, bound to AM2 joints. The package also holds the shared torch API that poses them (`surface_v2.py`, per interfaces.md).

Outputs go to `/var/tmp/alphamotion/am2exp/assets/surface_v2/`:
- `<body>.npz` and `<body>.json` for each body
- `manifest.json`: frozen config, sha256 of every file, and a content hash per body
- `validation/`: `schema.json`, `robot_fk.json`, `human.json`, plus the A-pose overlay npz
- `png/`: the acceptance renders
- `logs/`

## How to run (CPU only, about 6 min for everything)
```
cd CODE   # ROOT/am2exp_v1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
$PY assets/build_surface_v2.py --all            # or --body NAME (repeatable)
$PY assets/validate_assets.py                   # schema + robot FK vs MuJoCo + human chamfer/stance
$PY assets/render_assets.py --grids             # png/<body>.png, png/grid_<body>.png, png/human_overlay.png
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= $PY -m pytest -q -p no:cacheprovider tests/test_assets_sampler.py tests/test_assets_surface_v2.py
```
Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=` for pytest. Without it, ROS humble's pytest plugins load from the shell profile and crash.

## Files
| file | role |
|---|---|
| `surface_v2.py` | Runtime API: `load_surface`, `pose_surface`, `pose_surface_subset`, `ground_clearance`, `bone_frames`. Pure torch, batched, differentiable, fp32. It has no sealed imports and no relative imports, so it can be loaded by file path. |
| `build_surface_v2.py` | Build pipeline. Holds the frozen `CONFIG` and writes the manifest. |
| `umr_sampler.py` / `umr_reference.py` | Faithful port of UMR `scripts/surface_sampling.py` (commit 8c4db7db) plus a verbatim copy of the reference. The port adds metadata only: source face, number of views that saw it, view-direction sum. It is tested to be bit-identical to the reference. |
| `robot_binding.py` | Kinematic adapters and binding. XWBC robots use `surface/build_verified_patches.source_adapter` (imported read-only). External bodies use the same adapter with `data/payloads/<body>.json`. H2 is mapped by data joint names. It binds visual geoms to AM2 joints, verified over qpos0 plus 7 seeded random q, and checks the residual on real data joint states. |
| `human_surface.py` | human_smpl bone-revolution binding built from SMPL-X neutral. |
| `regions.py` | Region tagger: topology + names + rest geometry. One rule set for all 15 robots; tables for adroit and franka. |
| `pads.py` | Adroit and Franka pad tags, checked against MuJoCo. |
| `validate_assets.py`, `render_assets.py` | Acceptance numbers and renders. |
| `dump_am2_bodies.py`, `_env.py` | AM2 Body dump (subprocess, `load_scaling_bodies` with config A_S_seed0) and paths. |

## Method
- **Sampler (UMR, unchanged):**
  - 4096 × 8 × 12 area-weighted candidates.
  - A first-hit ray test from 26 directions; a candidate is exterior if at least 2 views see it.
  - FPS down to 4096, seed 0.
  - Visibility is computed on the whole assembled body mesh, not per link.
- **Normals:** outward, taken from the views that saw each face, so they do not depend on mesh winding. This fixes the 7 Adam-lite meshes whose winding is reversed.
- **Area per point:** each point gets the area of its Voronoi share of the exterior candidates.
- **Robots:**
  - Each visual geom is bound to an AM2 joint. A binding is exact ("rigid") when the geom is constant in that joint's frame over qpos0 plus 7 random q (tolerance 1e-6 / 1e-4 cm).
  - Otherwise the binding is **approximate**. Candidates are the nearest kept ancestor node and its parent. Each candidate gets two transforms: one exact at qpos0, and one least-squares fit over the states. The pick is the candidate with the smallest median-over-states error.
  - Per point the npz stores: `point_rigid`, `point_residual_cm` (random-q bound), `point_data_residual_p95_cm` (on LAFAN data joint states), and `point_geom` / `point_geom_vertex_cm` (to re-pose through MuJoCo).
- **human_smpl (`bone_revolution`, positions only):**
  - SMPL-X betas: 10 fitted with regularisation 4.0 to the AM2 bone lengths. Height 187 cm; bone-length RMS 4.6 cm. An exact fit is impossible because LAFAN-dialect proportions differ: neck 25.8 cm vs 17, clavicle 20.7 vs 9.7.
  - Each vertex goes to the bone with its largest LBS weight.
  - Frame rules are frozen and listed in the manifest and in the `human_surface.py` docstring:
    - legs: hip lateral
    - arms: chest forward (+0.3 up)
    - trunk, neck, head: hip or thorax lateral
    - feet: the knee–ankle–toe plane
  - Soles are flattened along a fixed sole normal in the foot frame. The ankle→toe line is tilted about 25°, so the sole normal is not v.
  - Hands are ellipsoids fitted to the SMPL-X hand vertices, along the forearm. Rotations are ignored (tested).
- **Regions:** feet and hands are the lowest and farthest leaves, and the pelvis is their LCA. Knee, ankle, elbow and wrist are found by name, or by arc-length fraction as a fallback. Points are split by arc position along the limb polyline. Head and trunk are split by height. Per-body tables are in `<body>.json["regions_table"]`. All 15 body regions are present on all 16 XWBC bodies (tested).
- **Groups:**
  - G=256 × K=16 and G=128 × 32: FPS centres + kNN on rest_points.
  - `sub1024`: FPS.
  - `allowed_anchor_pairs`: tree hop ≥ 3, minus quasi-rigid pairs (closest surface distance mean < 8 cm and std < 1 cm over 128 payload rows).
- **beta_vis_cm** (visual geometry, at qpos0):
  - Robots with a sole table: label sole drop minus visual drop.
  - Joint-centre-label bodies (human, H2, R1): minus the drop from the lowest joint centre to the lowest visual point.
  - Externals: 0.
- **delta_col_cm:** the report-32 `predicted_bias_cm` (NaN for human and H2). Judge-only.

## Acceptance results (all from files in OUT)
- **Tests:** `tests/test_assets_sampler.py` and `tests/test_assets_surface_v2.py`: **46 passed** (5 s).
- **Schema:** passes for all 18 bodies (`validation/schema.json`).
- **FK vs MuJoCo** (3 fresh random q, seed 7, every point):
  - Rigid points are within **≤ 0.0001 mm** on all 17 robots and externals (limit 0.5 mm).
  - Approximate points stay within 1.5× each body's documented max residual.
- **human_smpl:**
  - **Chamfer:**
    - T-pose calibration: points→mesh median 0.000 cm, mesh→points median 0.99 cm.
    - Standing A-pose posed from positions only: 0.000 / 1.00 cm (p95 1.22 / 1.83).
    - Sit-and-reach: 0.21 / 1.04 cm.
    - Mesh→points is limited by the 2 cm point spacing.
  - **Stance check** (1115 LAFAN walk payload rows with a foot contact label):
    - The lowest point is a foot in 96.7% of rows, and a downward sole in 86.6%.
    - The lowest sole is at −0.92 cm after beta (p10 −2.4, p90 +0.6).
- **beta_vis vs independent data check:** the check is the median lowest foot point with GT root height on contact rows. The two agree within 1.1 cm on all 15 robots and within 1.6 cm on human. See the table.
- **Pads:**
  - **Adroit:** 10–14 points per digit, every one within 2.3 mm of the MuJoCo C_*distal capsule surface. The centroid is 1.5–3.3 mm from the capsule pad point radially out from the Tch site.
  - **Franka:** 9 points per finger (gripper open). Pad depth vs the inner capsule surface (geoms 19–30 / 33–44): p50 1.0 mm, within ±1.6 mm.

| body | binding | spacing p50 cm | exterior area % | max residual cm (8 verif. states) | data p95 cm | pts data-p95>3cm | rigid FK err mm | beta_vis cm | beta data check cm | delta_col cm | quasi-rigid removed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| human_smpl | bone_revolution | 2.05 | 99 | - | - | - | - | -1.58 | -3.18 | - | 0 |
| booster_k1 | rigid | 1.20 | 51 | 1.35 | 0.86 | 0 | 0.0001 | 6.11 | 5.72 | 6.40 | 2 |
| booster_t1 | rigid | 1.41 | 62 | 0.00 | 0.00 | 0 | 0.0001 | 7.05 | 6.44 | 7.15 | 0 |
| booster_t1_29dof | rigid | 1.48 | 62 | 0.00 | 0.00 | 0 | 0.0000 | 9.55 | 8.45 | 9.65 | 0 |
| fourier_n1 | rigid | 1.48 | 27 | 0.88 | 1.03 | 0 | 0.0001 | 0.00 | -0.07 | 0.02 | 4 |
| hightorque_hi | rigid | 1.14 | 54 | 0.00 | 0.00 | 0 | 0.0000 | 0.00 | 0.28 | -0.00 | 0 |
| pnd_adam_lite | rigid | 1.74 | 38 | 0.55 | 0.65 | 0 | 0.0001 | 9.54 | 8.67 | 9.40 | 0 |
| tienkung | rigid | 1.89 | 27 | 1.00 | 1.28 | 0 | 0.0001 | 10.01 | 9.59 | 10.00 | 0 |
| unitree_g1_23dof | rigid | 1.43 | 46 | 0.62 | 0.25 | 0 | 0.0001 | -0.00 | 0.24 | 0.04 | 5 |
| unitree_g1_29dof | rigid | 1.43 | 46 | 0.62 | 0.66 | 0 | 0.0001 | -0.00 | 0.42 | 0.04 | 3 |
| unitree_h1 | rigid | 1.72 | 70 | 0.00 | 0.00 | 0 | 0.0001 | -0.00 | 0.12 | -0.00 | 3 |
| unitree_h1_2 | rigid | 1.82 | 55 | 14.11 | 13.84 | 219 | 0.0001 | -0.00 | 0.36 | 0.00 | 0 |
| fourier_gr3 | rigid | 1.94 | 26 | 3.80 | 2.84 | 0 | 0.0001 | 9.84 | 9.17 | 9.99 | 1 |
| kuavo_s45 | rigid | 1.94 | 21 | 12.20 | 15.77 | 457 | 0.0001 | 4.80 | 4.50 | 4.70 | 0 |
| unitree_h2 | rigid | 1.74 | 48 | 1.66 | 1.54 | 0 | 0.0000 | -5.59 | -5.63 | - | 1 |
| unitree_r1 | rigid | 1.32 | 40 | 0.52 | 0.65 | 0 | 0.0001 | -4.38 | -4.35 | -0.50 | 1 |
| adroit_pen | rigid | 0.48 | 79 | 0.00 | - | - | 0.0000 | 0.00 | - | - | 81 |
| relay_franka_full_v2 | rigid | 1.13 | 82 | 0.00 | - | - | 0.0001 | 0.00 | - | - | 0 |

"Exterior area %" is the share of all visual-mesh area that is visible from outside. Internal duplicate shells (kuavo, GR3, N1, tienkung) are correctly rejected.

## Deviations from the plan (reality checked)
1. **AM2 identity rotations are not MJCF qpos0** for 10 robots. AM2 rotations are raw link frames: the rotation at qpos0 is up to 90° off identity on N1's forearm, 37° on GR3's arms, 35° on Adam's hips, 30° on H2 and 16° on G1. Exterior visibility is therefore sampled at the vendor neutral pose **qpos0** (an upright, non-colliding pose), assembled through the AM2 binding. Local coordinates do not depend on pose. `rest_points` is still the contract's identity-rotation pose, and the "identity pose == rest_points" test passes. Franka is sampled with the gripper open, because at qpos0 the closed fingers hide the inner pads from every view.
2. **H2:** the nearest-rest-offset map from report-32 gives a **wrong** map on 11 nodes, because offsets are degenerate and the data tree differs from the MJCF by up to 5.3 cm. The name map (xwbc_joint_names.json, Isaac BFS order) is used instead. It is verified as follows:
   - It respects the tree.
   - Stored data body rotations match the MJCF to 0.07°.
   - Data-frame posed asset vs MJCF at the same joint angles: vertex error p50 0.28 cm, p95 0.70 cm.

   All H2 geoms are approximate (fitted in AM2-FK frames). Max residual is 1.66 cm on random q and 1.54 cm p95 on data.
3. **Merged-joint intermediate links cannot be represented rigidly.** Examples: H1_2 hip-pitch housings (219 points), kuavo shoulder links l1/l2 and hip l1 (457 points), GR3 wrist pitch links (37 points). They sit between the two MuJoCo bodies of one AM2 joint, and the AM2 rotation is the distal frame. Their residual on LAFAN data reaches p95 of 13.8 cm (H1_2) and 15.8 cm (kuavo zarm_r1). These points are flagged per point (`point_rigid=False`, `point_data_residual_p95_cm`). **Consumers should mask points with `point_data_residual_p95_cm > 3`** (none on the other bodies).
4. **beta for joint-centre-label bodies** (human, H2, R1) is −(lowest joint centre − lowest visual point), not 0. The data check confirms this: H2 −5.63 vs −5.59, R1 −4.35 vs −4.38. Externals use 0 (table at Y = −20 cm and floor at Y = −180 cm are noted in their manifest).
5. **Quasi-rigid pairs** are measured on 128 payload rows for every body (data rows rather than random q).
6. **Adroit pad verification:** each pad point lies within 3 mm of the MuJoCo capsule surface (passes). The pad centroid is 3.08 mm (thumb) and 3.30 mm (mf) from the capsule pad point, so `e2/opposition.surface_vs_mjcf_check` (centroid ≤ 3.0 mm) reports `passed=False` on those two digits by ≤ 0.3 mm. E2's default pad source is `mjcf`, so nothing breaks. If E2 wants `surface_v2` pads, its tolerance should be 3.5 mm.

## Open issues
- **Human hands** (ellipsoid along the forearm, wrist twist and flexion unobservable) sink up to about 20 cm into the floor when the palm is planted. In the LAFAN fall window the wrist joints themselves are at −5 to +2 cm. Hand-region ground clearance for human is unreliable; masking it, or using the wrist joint instead, is advised (see `png/grid_human_smpl.png`).
- Kuavo and H1_2 approximate points (above). Fixing them needs a non-rigid binding to two frames, which is outside the contract.
- The human bone-revolution frame uses a fixed 1e-3 fallback vector. It is not rotation-equivariant to within about 0.15 mm (tested to 0.5 mm).
- **Sole sampling is sparse:** about 40 flattened sole points per human foot, and on robots 74 downward points per two G1 feet. That is the UMR density on soles.
- Adroit renders are cramped (the long forearm squashes them). The pad close-ups are the useful panels.

## Renders
- `png/<body>.png` (18 bodies): sample pose front and side, AM2 rest, feet from below with sole normals, and pad close-ups for adroit and franka.
- `png/human_overlay.png`.
- `png/grid_{human_smpl,unitree_g1_29dof,unitree_h2,fourier_gr3}.png`: 12 frames over `lafan::fallAndGetUp1_subject4` [4136, 4264).
