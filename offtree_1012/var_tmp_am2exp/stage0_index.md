# am2exp_v1 Stage-0 index

Generated 2026-10-04 18:08 by `am2exp_v1/integration/index.py` (paths only; numbers are copied from the owning reports).

## Gate verdicts (from the owning reports, not recomputed)

| gate | owner | verdict | source |
|---|---|---|---|
| Ruler gate, strict (>= 13/15 within 1 cm) | WP-JUDGE | **FAIL 6/15** (LAFAN only 11/15; flat-foot frames 13/15) | judge/stage0_report.md s.1 |
| +-5 cm root shifts detected | WP-JUDGE | pass only under the rule revised after seeing the data (v0 rule failed) | judge/stage0_report.md s.1 |
| Shuffled-source floor (>= 0.15 CSE worse than A0) | WP-JUDGE | **FAIL** (-0.006: A0 ignores the source for contact) | judge/stage0_report.md |
| Headroom (numeric) | WP-JUDGE | pass (CSE_GT 0.774, FloatRate 0.50 vs GT 0) | judge/stage0_report.md |
| Headroom (video, lloyd) | lloyd | videos now exist (section 4), **not yet seen by lloyd** | integration/videos/stage0 |
| E2 G-0 (AN dev BA >= 0.75) | WP-E2 | pass on dev (0.765), **fragile**: test 0.539 | eval/e2/stage0_report.md |
| E2 OOD (hammer/pen <= 2x) | WP-E2 | S size pass (1.60x / 1.61x), M size over (2.01x / 2.02x) | eval/e2/stage0_report.md |
| Assets binding (< 0.5 mm exact-bound) | WP-ASSETS | pass; kuavo_s45 457 pts, unitree_h1_2 219 pts badly bound (masked downstream by integration) | assets/README.md |

## 1. Dense surface assets (WP-ASSETS)

- `/var/tmp/alphamotion/am2exp/assets/surface_v2/manifest.json` (14 kB, 10-04 16:33)
- `/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1/assets/README.md` (14 kB, 10-04 16:35)

Per-body renders (front, side, AM2 rest, soles, pads):
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/adroit_pen.png` (348 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/booster_k1.png` (361 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/booster_t1.png` (394 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/booster_t1_29dof.png` (348 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/fourier_gr3.png` (429 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/fourier_n1.png` (405 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/hightorque_hi.png` (406 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/human_smpl.png` (399 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/kuavo_s45.png` (392 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/pnd_adam_lite.png` (368 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/relay_franka_full_v2.png` (355 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/tienkung.png` (401 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/unitree_g1_23dof.png` (386 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/unitree_g1_29dof.png` (385 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/unitree_h1.png` (323 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/unitree_h1_2.png` (364 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/unitree_h2.png` (361 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/unitree_r1.png` (405 kB, 10-04 16:33)

Human overlay and 12-frame grids (fallAndGetUp1_subject4 [4136,4264)):
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/grid_fourier_gr3.png` (550 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/grid_human_smpl.png` (643 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/grid_unitree_g1_29dof.png` (361 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/grid_unitree_h2.png` (500 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/png/human_overlay.png` (351 kB, 10-04 16:33)

Validation:
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/validation/human.json` (2 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/validation/robot_fk.json` (5 kB, 10-04 16:33)
- `/var/tmp/alphamotion/am2exp/assets/surface_v2/validation/schema.json` (0 kB, 10-04 16:33)

## 2. Judge J1 / E1 Stage-0 (WP-JUDGE)

- `/var/tmp/alphamotion/am2exp/judge/stage0_report.md` (16 kB, 10-04 16:56)
- `/var/tmp/alphamotion/am2exp/judge/stage0_summary.json` (12 kB, 10-04 16:56)
- `/var/tmp/alphamotion/am2exp/judge/heldout_root_bias_v1.json` (8 kB, 10-04 16:22)
- `/var/tmp/alphamotion/am2exp/judge/ruler_v1/` (31 files)
- `/var/tmp/alphamotion/am2exp/judge/bindings_v1/` (31 files)
- `/var/tmp/alphamotion/am2exp/eval/cases_e1_v1.json` (309 kB, 10-04 16:04)
- `/var/tmp/alphamotion/am2exp/eval/cases_e1_v1.sha256` (0 kB, 10-04 16:04)
- `/var/tmp/alphamotion/am2exp/eval/cases_rootbias_v1.json` (14 kB, 10-04 16:04)
- `/var/tmp/alphamotion/am2exp/eval/e1/A0_s3000/` (3258 files)
- `/var/tmp/alphamotion/am2exp/eval/e1/B0_s3000/` (2172 files)
- `/var/tmp/alphamotion/am2exp/eval/e1/logs/` (24 files)
- `/var/tmp/alphamotion/am2exp/eval/e1/rootbias/` (96 files)

## 3. E2 Stage-0 (WP-E2)

- `/var/tmp/alphamotion/am2exp/eval/e2/stage0_report.md` (12 kB, 10-04 17:42)
- `/var/tmp/alphamotion/am2exp/eval/e2/stage0_summary.json` (72 kB, 10-04 17:41)
- `/var/tmp/alphamotion/am2exp/eval/e2/arms/` (42 files)
- `/var/tmp/alphamotion/am2exp/eval/e2/labels/` (3 files)
- `/var/tmp/alphamotion/am2exp/data/e2/d4rl/` (4 files)

## 4. Videos (mp4 + 12-frame full-duration grid each)

**Stage-0 [human | GMR GT | A0@3000], heldout GR3 and seen G1, fallAndGetUp1_s4 [4136,4264) (integration, from WP-JUDGE npz)**

- `/var/tmp/alphamotion/am2exp/integration/videos/stage0/human_GT_A0__fourier_gr3__fallAndGetUp1_s4_4136/triptych.mp4` (3.5 MB, 10-04 17:52)
- `/var/tmp/alphamotion/am2exp/integration/videos/stage0/human_GT_A0__fourier_gr3__fallAndGetUp1_s4_4136/grid12.png` (756 kB, 10-04 17:52)
- `/var/tmp/alphamotion/am2exp/integration/videos/stage0/human_GT_A0__fourier_gr3__fallAndGetUp1_s4_4136/manifest.json` (9 kB, 10-04 17:52)
- `/var/tmp/alphamotion/am2exp/integration/videos/stage0/human_GT_A0__unitree_g1_29dof__fallAndGetUp1_s4_4136/triptych.mp4` (3.3 MB, 10-04 17:52)
- `/var/tmp/alphamotion/am2exp/integration/videos/stage0/human_GT_A0__unitree_g1_29dof__fallAndGetUp1_s4_4136/grid12.png` (672 kB, 10-04 17:52)
- `/var/tmp/alphamotion/am2exp/integration/videos/stage0/human_GT_A0__unitree_g1_29dof__fallAndGetUp1_s4_4136/manifest.json` (8 kB, 10-04 17:52)

**Demos (WP-RENDER): GT demo and E2 [hand | AN | A0_S@3000]**

- `/var/tmp/alphamotion/am2exp/videos/demo/e2_hammer_test_ep1__hand_AN_A0S3000/triptych.mp4` (720 kB, 10-04 16:13)
- `/var/tmp/alphamotion/am2exp/videos/demo/e2_hammer_test_ep1__hand_AN_A0S3000/grid12.png` (404 kB, 10-04 16:13)
- `/var/tmp/alphamotion/am2exp/videos/demo/e2_hammer_test_ep1__hand_AN_A0S3000/manifest.json` (2 kB, 10-04 16:13)
- `/var/tmp/alphamotion/am2exp/videos/demo/fallAndGetUp1_subject4_4136-4264__human_gtG1_gtGR3/triptych.mp4` (1.9 MB, 10-04 16:12)
- `/var/tmp/alphamotion/am2exp/videos/demo/fallAndGetUp1_subject4_4136-4264__human_gtG1_gtGR3/grid12.png` (983 kB, 10-04 16:12)
- `/var/tmp/alphamotion/am2exp/videos/demo/fallAndGetUp1_subject4_4136-4264__human_gtG1_gtGR3/manifest.json` (6 kB, 10-04 16:12)

**Integration smoke [human | A0@3000 | S untrained] (pipeline check only; S is random init)**

- `/var/tmp/alphamotion/am2exp/integration/videos/human_smpl__unitree_g1_29dof__fallAndGetUp1_subject4_4136/triptych.mp4` (3.1 MB, 10-04 17:50)
- `/var/tmp/alphamotion/am2exp/integration/videos/human_smpl__unitree_g1_29dof__fallAndGetUp1_subject4_4136/grid12.png` (642 kB, 10-04 17:50)
- `/var/tmp/alphamotion/am2exp/integration/videos/human_smpl__unitree_g1_29dof__fallAndGetUp1_subject4_4136/manifest.json` (8 kB, 10-04 17:50)

## 5. Model / ops probes

- `/var/tmp/alphamotion/am2exp/model/bench_cpu_B16.json` (3 kB, 10-04 12:09)
- `/var/tmp/alphamotion/am2exp/model/gpu_timing_B16.json` (0 kB, 10-04 16:07)
- `/var/tmp/alphamotion/am2exp/runs/_smoke/Arep_12steps_20261004T231142Z/` (14 files)

## 6. Integration smoke (WP-INTEGRATION)

- `/var/tmp/alphamotion/am2exp/integration/dryrun_summary.json` (21 kB, 10-04 17:51)
- `/var/tmp/alphamotion/am2exp/integration/dryrun7_b32/` (57 files)
- `/var/tmp/alphamotion/am2exp/integration/dryrun6_b32/` (41 files)
- `/var/tmp/alphamotion/am2exp/integration/e1_summary.json` (9 kB, 10-04 17:50)
- `/var/tmp/alphamotion/am2exp/integration/render_summary.json` (1 kB, 10-04 17:50)
- `/var/tmp/alphamotion/am2exp/integration/ckpt/S_untrained/checkpoint_step0000.pt` (27.8 MB, 10-04 17:49)
- `/var/tmp/alphamotion/am2exp/integration/pytest_final.log` (1 kB, 10-04 18:07)
- `/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1/integration/README.md` (7 kB, 10-04 18:03)

| arm | status | hashes checked / mismatches (5 UniMate steps) | independent re-check | RSS peak GiB |
|---|---|---|---|---|
| SF | completed | 5 / 0 | True | 1.798 |
| S | completed | 5 / 0 | True | 4.939 |
| O | completed | 5 / 0 | True | 4.862 |
| I | completed | 5 / 0 | True | 6.465 |
| S_wrong | completed | 5 / 0 | True | 5.515 |

Not produced (by design in Stage-0): any training run beyond 12 GPU steps, GPU timing probe at batch 128, lloyd approvals.
