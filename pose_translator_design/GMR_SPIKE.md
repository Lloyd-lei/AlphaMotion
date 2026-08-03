# GMR Spike: SOMA BVH → 17 robot embodiments

Date: 2026-08-03. Timeboxed feasibility spike. All prototype artifacts under
`/var/tmp/alphamotion/gmr_spike/`.

## Route decision: Route 2 — feed SOMA BVH straight into GMR with our own configs

**Validated end-to-end.** 5 SOMA BVH files retargeted to `unitree_g1` and
`booster_t1_29dof`, 10/10 clean (no NaN, joints within limits, feet on ground),
rendered frames look right (robot jogs/walks with arms swinging at its sides,
not exploded, not T-posed).

Why not the other routes:

- **Route 1 (replicate the pipeline that made `g1/csv/`) — dead end.** Those CSVs
  were shipped inside the BONES-SEED HuggingFace download; BONES studio retargeted
  them, not us. No SOMA ik config or converter exists anywhere on this machine
  (searched Desktop + GMR repo). The only local GMR run
  (`wbc_expert/runs/retarget_compare/20260620_gmr_vs_omniretarget_smoke.md`) used a
  LAFAN CSV, not SOMA.
- **Route 3 (SOMA → SMPLX first) — unnecessary.** It needs a body-model fitting
  step (heavy, no ready tooling on disk for BVH→SMPLX). Route 2 gets the same
  robots without it — see "scaling to all 17" below.

## Why Route 2 works

- SOMA joint names are almost the LAFAN1 names GMR already supports. Only the leg
  chain differs: `LeftUpLeg/LeftLeg` (LAFAN1) → `LeftLeg/LeftShin` (SOMA).
- SOMA has `LeftToeBase`/`RightToeBase`, so GMR's "nokov" trick (foot position +
  toe orientation = `FootMod` target) applies as-is.
- GMR only tracks 14 body targets (pelvis, spine, 2×{hip, knee, foot, shoulder,
  elbow, wrist}). SOMA has all 14. Fingers/head/eyes are ignored by design.

Two problems had to be solved (both solved, both in the spike dir):

1. **GMR's vendored BVH parser can't read SOMA files.** SOMA uses 6 channels on
   Root/Hips and 3 elsewhere; the LAFAN parser assumes uniform channels.
   → wrote `soma_bvh_loader.py` (generic parser + FK + Y-up-cm → Z-up-m, same
   output dict GMR expects). ~60 lines, no GMR modification needed.
2. **The rotation offsets in the LAFAN1 configs are wrong for SOMA.** Both rigs
   point bones along local +X, but most chains differ by a ~140–180° roll about
   the bone axis (first attempt gave a robot jogging with arms over its head).
   → wrote `calibrate_offsets.py`: pick the most neutral standing frame from an
   idle motion (feet flat, hands by thighs, auto-scored), pair it with the robot's
   zero pose, solve `off = q_soma⁻¹ · yaw_align · q_robot` per joint. Validated:
   the two joints whose conventions do match (Hips, Spine2) came out within 6–12°
   of the stock values.

**Key generalization trick:** the per-joint convention delta
`C_j = off_soma · off_lafan⁻¹` is *robot-independent* (it only compares two human
rigs). Computed once from the calibrated G1 pair, it converts **any** existing
GMR config to a SOMA config mechanically. This is exactly how the T1 config was
made — and T1's zero pose is a T-pose, so naive per-robot calibration failed
there while the C_j transfer worked first try.

## Working prototype commands

Env: conda env `genmo-gem` already has mink + mujoco 3.4 + daqp (ardy does not
have mink; nothing was installed anywhere). GMR is used via `sys.path`, repo
untouched.

```bash
# one file, one robot, with sanity stats + 3 rendered PNGs
cd /var/tmp/alphamotion/gmr_spike
MUJOCO_GL=egl /home/arenalabs/miniconda3/envs/genmo-gem/bin/python \
  soma_to_robot.py \
  --bvh_file '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/wbc_expert/data/raw/bones_seed_full/soma_uniform/bvh/230731/walk_180_R_002__A437_M.bvh' \
  --robot unitree_g1 --target_fps 30 --render
```

Outputs per file: `out/<robot>/<name>.npz` (qpos array + fps), `.csv` (one qpos
row per frame, beyondmimic-style), `.stats.json` (NaN / joint-range / ground
checks), `.frame{0,1,2}.png` renders.

Files authored (all in `/var/tmp/alphamotion/gmr_spike/`):

| File | What |
|---|---|
| `soma_bvh_loader.py` | Generic SOMA BVH parser + FK → GMR frame dicts |
| `soma_to_robot.py` | Headless retarget driver (registers configs in-memory) |
| `calibrate_offsets.py` | Neutral-frame offset calibration (G1 anchor) |
| `ik_configs/bvh_soma_to_g1.json` | Working SOMA→G1 config |
| `ik_configs/bvh_soma_to_t1_29dof.json` | Working SOMA→T1 config (via C_j transfer) |

## Prototype results (look at the PNGs yourself)

Renders to eyeball: `/var/tmp/alphamotion/gmr_spike/out/unitree_g1/*.png` and
`/var/tmp/alphamotion/gmr_spike/out/booster_t1_29dof/*.png`.

| Robot | Motion | Frames@30fps | NaN | max\|joint\| (rad) | lowest body z (m) |
|---|---|---|---|---|---|
| unitree_g1 | jog_ff_start_180 | 49 | no | 1.20 | 0.054 |
| unitree_g1 | jump_ff_270 | 50 | no | 1.13 | 0.082 |
| unitree_g1 | jump_ff_180 | 50 | no | 1.21 | 0.064 |
| unitree_g1 | idle_turn_270 | 279 | no | 0.91 | 0.103 |
| unitree_g1 | walk_180 (mirrored) | 180 | no | 1.04 | 0.097 |
| booster_t1_29dof | same 5 motions | 49–279 | no | 1.74–2.27 | 0.008–0.018 |

Root heights are plausible (G1 pelvis 0.79–0.91 m, T1 waist 0.62–0.71 m), feet
never penetrate the ground, walking covers ~3 m as in the source.

## Disk / compute estimates (2000 files × 17 robots = 34,000 retargets)

Measured: IK runs at ~144 output-frames/s on one core (aggregate over the 10
runs, includes mink QP + limits). BVH parse 0.06–0.17 s/file. Average BONES-SEED
motion is ~7.3 s (288 h / 142,220 files) = ~219 frames at 30 fps.

| Quantity | Per unit | 2000 × 17 total |
|---|---|---|
| IK time (1 core) | ~1.5 s/file/robot + ~0.2 s load | **~16 CPU-hours** |
| Wall clock, 16 parallel workers | — | **~1 hour** |
| npz (compressed qpos) | ~55 KB avg (240 B/frame, G1 36-dof qpos) | **~1.9 GB** |
| csv (optional, text) | ~70 KB avg | +2.4 GB |

Fully parallel per (file, robot); a plain `xargs -P16` over a manifest is enough.
No GPU needed.

## Risks and caveats

- **Robot coverage.** Direct config transfer done for G1 + T1-29dof; the other 5
  `bvh_lafan1_*` robots (n1, toddy, pm01, talos, g1_with_hands) are the same
  one-line rename + C_j. The 10 smplx-only robots (h1, h1_2, t1-23dof, kuavo, hi,
  r1pro, bhl, k1, adam, tienkung, gr3) need the analogous delta
  `C'_j = off_soma_g1 · off_smplx_g1⁻¹` with the SMPLX↔SOMA name map
  (pelvis↔Hips, left_hip↔LeftLeg, left_knee↔LeftShin, left_foot↔LeftFootMod,
  spine3↔Spine2, left_shoulder↔LeftArm, left_elbow↔LeftForeArm,
  left_wrist↔LeftHand). Mechanical, but **each robot must get one visual check**
  — that is how the T1 T-pose-arms bug was caught; numbers alone looked fine.
- **Calibration residual.** Offsets were calibrated from one near-neutral human
  frame (score-picked from an idle motion). Hips/Spine validated at 6–12° vs
  ground truth; arms may carry a few degrees of constant bias (slightly bent
  elbows in the calibration frame). If arm poses matter a lot, refine by
  averaging several neutral frames or hand-tuning like GMR's DOC.md describes.
- **Not retargeted:** fingers (SOMA has 60+ hand joints, GMR tracks the wrist
  only), head/neck, jaw, eyes. Fine for whole-body control; not fine if you need
  dexterous hands.
- **fps:** source is uniform 120 fps; the driver strides by integer factor
  (120/30=4). Arbitrary target fps needs interpolation — don't bother, keep 30.
- **`galaxea_r1pro` is wheeled** — expect the legs to be meaningless there; it
  also has no lafan-style config, so it comes via the smplx delta path.
- **Vendor G1 CSVs vs ours:** BONES-SEED's `g1/csv/` came from a different
  (unknown) retargeter. Don't mix them with our GMR outputs in one training set
  without checking conventions; prefer regenerating G1 with GMR so all 17
  embodiments share one retargeting distribution.
- **Repo intact:** everything lives in `/var/tmp/alphamotion/gmr_spike/`; configs
  are injected at runtime (`params.IK_CONFIG_DICT["bvh_soma"] = ...`). For the
  real run, either keep that pattern or commit the configs into
  `alpha_motion2/external/GMR/general_motion_retargeting/ik_configs/` (cleaner).

## Recommended next step

1. Generate the remaining 15 configs with the C_j / C'_j transfer (one script,
   ~an hour of work including per-robot visual spot-checks: render 3 frames of
   the same walk clip per robot, eyeball all 17 in one contact sheet).
2. Pick the 2000-file subset (reuse the mixed-dataset caption labels to balance
   action families), write a manifest of (bvh, robot) pairs.
3. Batch driver = `soma_to_robot.py` minus rendering, one process per worker,
   model loaded once per robot; `xargs -P16`. ~1 h wall, ~2 GB.
4. Store as `npz{qpos[T,nq], fps, robot, src_bvh}` — that plus the source SOMA
   BVH *is* the multi-topology paired cache. Before training anything, render a
   handful of clips per robot and look at them.
