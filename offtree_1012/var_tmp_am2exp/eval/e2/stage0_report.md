# E2 Stage-0 report (CPU only, no training)

Generated 2026-10-05T00:15:39Z. All numbers come from `e2/run_e2_eval.py stage0`; the judge is `e2/closure_judge.py` (MuJoCo collision capsules + dataset object pose; it never calls training-loss code).

## Short version

- **G-0 gate: PASS.** Analytic retarget (AN) on hammer dev: balanced accuracy 0.765 (needs >= 0.75). Constant-open gives exactly 0.5000 (dev) and 0.5000 (test), as it must.
- AN on hammer **test** (never used for any choice): BA 0.539 [0.449, 0.644] (episode bootstrap 95%); P(hold | grasp) falls from 0.529 (dev) to 0.177 (test). **The gate passes on dev but the analytic reference does not carry over to test: treat the G-0 pass as fragile.**
- The planned descriptor (thumb pad to virtual-finger pad, best planned variant `vf_ffmfrf`) does **not** measure object size on these grasps: on hammer dev grasps it reads 29.0 mm for a 50 mm handle, AN BA 0.466. Hammer and relocate are wrap (power) grasps: the palm, finger middle/proximal segments and the thumb side touch the object, the fingertip pads mostly do not face the thumb tip.
- Frozen descriptor: `thface_070` (status: already_frozen_consistent). It measures the gap between a thumb pad and the nearest hand pad that faces it (any finger segment or the palm). It was chosen on dev only, by the pre-set rule "mean over {hammer, relocate} of median |c_A - d_obj| (mm) on DEV grasp frames (g=1), raw c_A". Because the family was widened after the planned one failed, the dev gate is optimistic; read the test number.
- Existing model `A0_S@1000` mapping hand -> gripper directly: hammer test BA 0.500; median gripper clearance 77.6 mm on grasp frames, 77.6 mm on free frames (all hammer frames span 77.5-77.7 mm; median slide fraction 0.995).
- Existing model `A0_S@3000` mapping hand -> gripper directly: hammer test BA 0.500; median gripper clearance 74.1 mm on grasp frames, 74.4 mm on free frames (all hammer frames span 72.4-75.3 mm; median slide fraction 0.953).
- Existing model `A_M@1000` mapping hand -> gripper directly: hammer test BA 0.500; median gripper clearance 78.0 mm on grasp frames, 78.0 mm on free frames (all hammer frames span 78.0-78.0 mm; median slide fraction 1.000).
- Existing model `A_M@3000` mapping hand -> gripper directly: hammer test BA 0.500; median gripper clearance 78.0 mm on grasp frames, 78.0 mm on free frames (all hammer frames span 78.0-78.0 mm; median slide fraction 1.000).
- Gripper -> hand closure, balanced fit rate over squeeze / other frames: `A0_S@1000` 0.559; `A0_S@3000` 0.332; `A_M@1000` 0.418; `A_M@3000` 0.345; `floor_rest_hand` 0.000; `floor_random_pen_train_hand` 0.335. Read against the random-hand floor: a hand that ignores the gripper can still score on one side (a fixed open hand fits the open frames), so only a clear margin above `floor_random_pen_train_hand` and above a fixed-hand score means the hand follows the gripper gap. None of the existing models shows that.
- OOD check `A0_S@1000`: same-body hand reconstruction error hammer test / pen eval = 1.60x (within the 2x limit).
- OOD check `A0_S@3000`: same-body hand reconstruction error hammer test / pen eval = 1.61x (within the 2x limit).
- OOD check `A_M@1000`: same-body hand reconstruction error hammer test / pen eval = 2.01x (OVER the 2x limit).
- OOD check `A_M@3000`: same-body hand reconstruction error hammer test / pen eval = 2.02x (OVER the 2x limit).

## Grasp / non-grasp frame counts (closure judge labels)

| set | episodes | frames | grasp g=1 | free g=0 | excluded |
|---|---|---|---|---|---|
| hammer/dev | 10 | 4296 | 2073 | 437 | 1786 |
| hammer/test | 15 | 7039 | 3013 | 670 | 3356 |
| relocate/dev | 10 | 4034 | 2150 | 1506 | 378 |
| relocate/test | 15 | 5933 | 3132 | 2168 | 633 |
| pen/eval | 5 | 1005 | 0 | 9 | 996 |
| pen/train | 20 | 4020 | 157 | 25 | 3838 |

Pen re-measure (audit said ~0%): thumb pad AND a finger pad within 5 mm of the pen = 0.04% of frames (per digit th/ff/mf/rf/lf: 3.7%, 0.1%, 18.5%, 2.5%, 14.0%). With capsules + opposite-side rule the judge finds g=1 on 3.1% of pen frames.

## Descriptor candidates (DEV only)

| variant | family | hammer median c_A on grasps (mm, d=50) | hammer median abs err | relocate median c_A (d=70) | relocate median abs err | rule value | AN hammer dev BA | AN hammer test BA (diagnostic, after freeze) |
|---|---|---|---|---|---|---|---|---|
| vf_ffmf | planned | 28.7 | 21.3 | 31.8 | 38.2 | 29.75 | 0.465 | 0.452 |
| vf_ffmfrf | planned | 29.0 | 21.0 | 32.1 | 37.9 | 29.43 | 0.466 | 0.449 |
| vf_all4 | planned | 28.3 | 21.7 | 32.1 | 37.9 | 29.82 | 0.466 | 0.449 |
| gap_ffmf | planned | 17.5 | 32.5 | 14.0 | 56.0 | 44.23 | 0.377 | 0.306 |
| thface_050 | thumb_facing | 52.9 | 2.9 | 46.3 | 23.7 | 13.33 | 0.932 | 0.732 |
| thface_060 | thumb_facing | 53.0 | 3.0 | 46.3 | 23.7 | 13.33 | 0.914 | 0.653 |
| thface_070 | thumb_facing | 57.1 | 7.1 | 76.8 | 8.8 | 7.93 | 0.765 | 0.539 |

The last column was computed after the choice was frozen and is shown only to make the dev-to-test drop visible; the frozen choice is not changed by it.

## Arms on hammer (primary set; hold band 47-58 mm)

| arm | kind | dev BA | test BA [95% CI] | test P(hold given grasp) | test P(open given free) | test crush given grasp | test median c_F on grasps (mm) |
|---|---|---|---|---|---|---|---|
| constant_open | floor | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 78.0 |
| constant_closed | floor | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 1.000 | -2.0 |
| constant_median | floor | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 74.8 |
| AN | reference | 0.765 | 0.539 [0.449, 0.644] | 0.177 | 0.901 | 0.000 | 72.4 |
| AN_shuffled | floor | 0.469 | 0.481 [0.416, 0.533] | 0.153 | 0.809 | 0.000 | 73.1 |
| AN_planned_vf_ffmfrf | diagnostic | 0.466 | 0.449 [0.366, 0.519] | 0.100 | 0.799 | 0.791 | 33.2 |
| A0_S@1000 | model_direct | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 77.6 |
| A0_S@1000_shuffled | floor | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 77.6 |
| A0_S@3000 | model_direct | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 74.1 |
| A0_S@3000_shuffled | floor | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 74.2 |
| A_M@1000 | model_direct | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 78.0 |
| A_M@1000_shuffled | floor | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 78.0 |
| A_M@3000 | model_direct | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 78.0 |
| A_M@3000_shuffled | floor | 0.500 | 0.500 [0.500, 0.500] | 0.000 | 1.000 | 0.000 | 78.0 |

BA follows the plan: 0.5 x (P(hold | grasp) + P(open | free)). Under it a constant-closed gripper scores 0 (it crushes and is never open); the alternative "not-hold" form (free frames only need not-hold) gives constant-open = constant-closed = 0.5; it is in the JSON as BA_nothold.

## Relocate (secondary; ball 70 mm sits at the top of the 78 mm stroke, so only grasp hold-rate and crush-rate mean something)

| arm | test P(hold given grasp) | test crush given grasp | test median c_F on grasps (mm) |
|---|---|---|---|
| constant_open | 1.000 | 0.000 | 78.0 |
| constant_closed | 0.000 | 1.000 | -2.0 |
| constant_median | 1.000 | 0.000 | 74.8 |
| AN | 0.614 | 0.386 | 72.1 |
| AN_shuffled | 0.761 | 0.239 | 78.0 |
| AN_planned_vf_ffmfrf | 0.000 | 1.000 | 28.2 |
| A0_S@1000 | 1.000 | 0.000 | 77.6 |
| A0_S@1000_shuffled | 1.000 | 0.000 | 77.6 |
| A0_S@3000 | 1.000 | 0.000 | 73.9 |
| A0_S@3000_shuffled | 1.000 | 0.000 | 74.1 |
| A_M@1000 | 1.000 | 0.000 | 78.0 |
| A_M@1000_shuffled | 1.000 | 0.000 | 78.0 |
| A_M@3000 | 1.000 | 0.000 | 78.0 |
| A_M@3000_shuffled | 1.000 | 0.000 | 78.0 |

## Pen eval (no opposition grasps; open side + rank agreement with the source descriptor)

| arm | free frames | P(open given free) | Spearman(c_F, c_A source) |
|---|---|---|---|
| constant_open | 9 | 1.000 | nan |
| constant_closed | 9 | 0.000 | nan |
| constant_median | 9 | 1.000 | nan |
| AN | 9 | 1.000 | 0.984 |
| AN_shuffled | 9 | 0.778 | 0.311 |
| AN_planned_vf_ffmfrf | 9 | 0.778 | 0.550 |
| A0_S@1000 | 9 | 1.000 | 0.540 |
| A0_S@1000_shuffled | 9 | 1.000 | 0.369 |
| A0_S@3000 | 9 | 1.000 | 0.344 |
| A0_S@3000_shuffled | 9 | 1.000 | 0.341 |
| A_M@1000 | 9 | 1.000 | nan |
| A_M@1000_shuffled | 9 | 1.000 | nan |
| A_M@3000 | 9 | 1.000 | 0.458 |
| A_M@3000_shuffled | 9 | 1.000 | 0.191 |

## OOD check (same-body Adroit reconstruction, mean joint position error)

| model | pen eval (cm) | hammer test (cm) | relocate test (cm) | hammer/pen | gate (<= 2x) |
|---|---|---|---|---|---|
| A0_S@1000 | 2.89 | 4.62 | 5.81 | 1.60 | pass |
| A0_S@3000 | 2.22 | 3.57 | 4.82 | 1.61 | pass |
| A_M@1000 | 2.22 | 4.48 | 5.81 | 2.01 | FAIL |
| A_M@3000 | 1.98 | 4.01 | 5.30 | 2.02 | FAIL |

## Gripper -> hand closure test (Franka eval episodes [9, 12, 15, 17], 799 rows, squeeze share 0.385)

A Franka source frame gives a cylinder of diameter c_F (the gripper pad gap). The decoded hand "holds" it if the closure rule below passes; it "fits" it if it holds c_F but not c_F + 30 mm (one-sided hold cannot tell an open hand from a right one). Squeeze frames: median c_F 45.8 mm; other frames are mostly fully open (~77 mm).

| hand source | hold, squeeze | hold, other | fit, squeeze | fit, other | closure BA (fit) |
|---|---|---|---|---|---|
| A0_S@1000 | 1.000 | 0.978 | 0.218 | 0.900 | 0.559 |
| A0_S@3000 | 0.617 | 0.707 | 0.179 | 0.485 | 0.332 |
| A_M@1000 | 0.945 | 0.198 | 0.662 | 0.173 | 0.418 |
| A_M@3000 | 0.698 | 0.230 | 0.487 | 0.204 | 0.345 |
| floor_rest_hand | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| floor_random_pen_train_hand | 0.685 | 0.281 | 0.506 | 0.163 | 0.335 |

Closure-test sanity on real hammer TEST frames (200 grasp, 200 free): grasping hands hold a cylinder of the real handle size in 0.975 of frames, one 30 mm larger in 0.060, one 30 mm smaller in 0.960 (the test cannot say a closed hand is too closed: a smaller object can always be pinched somewhere). Free hands hold the real size in 0.520 and +30 mm in 0.735. Pass rule: some placement of a cylinder of diameter c_F has the thumb tip capsule and an ff/mf/rf tip capsule within 5 mm of its surface on opposite sides, with no hand capsule pushed in deeper than 5 mm (same 5 mm tolerance as the grasp labels; real D4RL grasps sink 1-4 mm into the handle).


## Pad cross-check: dense-asset pad tags (WP-ASSETS surface_v2) vs MJCF pads (descriptor uses the MJCF pads)

| digit | tagged points | patch centroid to MJCF pad (mm) | nearest tagged point to MJCF pad (mm) | normal cos |
|---|---|---|---|---|
| th | 14 | 3.08 | 1.14 | 0.990 |
| ff | 11 | 2.83 | 1.90 | 0.998 |
| mf | 10 | 3.30 | 2.67 | 0.998 |
| rf | 12 | 1.58 | 2.82 | 0.975 |
| lf | 11 | 1.48 | 1.66 | 0.968 |

Centroid rule (<= 3.0 mm): FAIL; cover rule (nearest tagged point <= 3.0 mm): pass. The tagged patches cover the MJCF pads; their centroids sit 1-1.5 mm under the surface (curved patch) and ~2.5 mm toward the knuckle on th/ff/mf, so two centroids miss 3 mm by a fraction of a millimetre. The frozen descriptor keeps the MJCF pads, so this does not change any E2 number.

## Notes

- Time alignment: D4RL observations hold qpos after the step but body/site positions from one 2 ms physics sub-step earlier. Hammer/relocate frames are therefore taken at observation time (hand qpos moved back 0.2 frame); with that, relocate palm-minus-ball matches obs[30:33] within 0.36 mm (max), hammer palm within 1.19 mm. Pen hand rows stay as trained; the pen pose is moved forward 0.2 frame instead.
- Hammer orientation uses the gymnasium-robotics euler convention (= scipy intrinsic XYZ); checked by the hammer head landing on the nail at impact frames (median 3.5 mm; other conventions 218-450 mm).
- Videos for Stage-0 (hand+object | analytic gripper) belong to the render package; per-frame inputs are in `arms/<arm>/<set>.npz`.

Wall time 1562.4 s.

