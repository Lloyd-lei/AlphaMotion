# E1 Stage-0 (CPU): ruler gate, heldout root bias, headroom

Generated 2026-10-04 16:56 PDT. Case list `/var/tmp/alphamotion/am2exp/eval/cases_e1_v1.json` sha256 `ebb05d74a156c35813691b13afae2c4b2be8de4329d1f9c483535adb9232aad0` (28 windows, 1080 cases, 346 primary). Judge J1 = vendor MJCF collision geoms re-posed from AM2 global rotations/positions (OUT/judge/bindings_v1). All numbers are means of per-window means (window = cluster). No video has been looked at yet: these numbers do not replace the [human | GT | A0] videos.

## In short

* Ruler: J1 agrees with report 32 on flat-foot standing frames (13/15 within 1 cm) but the pre-registered strict gate fails (6/15): on tilted stance frames the collision box corner sits 0.5-2 cm lower than the single label sole point. Root shifts of +-5 cm are seen. R1's report-32 delta was wrong by 3.9 cm (fixed in J1).
* A0 almost never puts a non-foot body part on the ground: 9% of frames vs 77% in the GT (heldout targets, contact-rich windows). Contact-pattern error CSE_GT 0.78; position error 375 mm.
* A0 ignores the source for ground contact: feeding it another clip (shuffled source) gives the same CSE_GT; only the pose error gets worse. A real contact pattern from a different window scores CSE_GT 0.53, i.e. better than A0. So the headroom is large, and the shuffled-source floor gate (>= 0.15 worse than A0) cannot pass for A0 itself.
* The GT retargets themselves sink into the floor on ground motions (GT PenP95 ~13 cm on primary windows); penetration must be judged relative to the GT, not to zero.
* B0 (96-patch surface) floats more than A0 and penetrates less; its contact pattern is not better.
* Nothing here has been checked on video yet.

## 1. Ruler gate (J1 on GMR GT vs report 32)

Reference per body/corpus = report-32 stance gap p50 (label sole point above label floor) + delta_col (label drop - vendor collision drop). J1 = p50 over the same stance frames of the lowest collision point with the label root height.

| body | ref LAFAN | J1-ref LAFAN | J1-ref AMASS | strict pass | flat-foot diff L/A | flat pass | J1 static drop vs r32 true drop | shift detected |
|---|---|---|---|---|---|---|---|---|
| booster_k1 | 6.88 | -0.75 | -1.28 | NO | -0.35 / -0.88 | yes | 2.40 vs 2.40 | yes |
| booster_t1 | 7.61 | -0.87 | -1.57 | NO | -0.39 / -0.84 | yes | 3.00 vs 3.00 | yes |
| booster_t1_29dof | 10.13 | -1.11 | -2.09 | NO | -0.51 / -1.32 | NO | 3.00 vs 3.00 | yes |
| fourier_n1 | 0.70 | -0.79 | -0.86 | yes | -0.33 / -0.39 | yes | 4.45 vs 4.43 | yes |
| hightorque_hi | 1.29 | -0.43 | -0.33 | yes | -0.10 / +0.01 | yes | 4.40 vs 4.40 | yes |
| pnd_adam_lite | 10.20 | -0.97 | -1.49 | NO | -0.34 / -0.96 | yes | 6.50 vs 6.50 | yes |
| tienkung | 12.37 | -1.14 | -1.92 | NO | -0.43 / -1.00 | NO | 5.70 vs 5.70 | yes |
| unitree_g1_23dof | 1.26 | -0.64 | -0.45 | yes | -0.21 / +0.01 | yes | 3.50 vs 3.50 | yes |
| unitree_g1_29dof | 0.99 | -0.53 | -0.16 | yes | -0.19 / +0.57 | yes | 3.50 vs 3.50 | yes |
| unitree_h1 | 1.21 | -0.88 | -0.79 | yes | +0.04 / +0.19 | yes | 7.00 vs 7.00 | yes |
| unitree_h1_2 | 1.07 | -0.68 | -0.84 | yes | -0.01 / -0.07 | yes | 4.50 vs 4.50 | yes |
| fourier_gr3 | 10.41 | -0.93 | -1.42 | NO | -0.41 / -0.99 | yes | 6.81 vs 6.81 | yes |
| kuavo_s45 | 5.39 | -0.83 | -1.15 | NO | -0.23 / -0.46 | yes | 6.05 vs 6.05 | yes |
| unitree_h2 | -4.46 | -2.14 | -1.35 | NO | -0.36 / -0.86 | yes | 5.43 vs 5.43 | yes |
| unitree_r1 | 0.13 | -4.45 | -4.97 | NO | -0.17 / -0.76 | yes | 4.38 vs 0.50 | yes |

* Strict gate (both corpora within 1 cm, report-32 delta): **6/15 -> FAIL** (needs 13). With the R1 delta corrected: 6/15. LAFAN only: 11/15.
* Flat-foot stance frames (lowest-geom node tilt < 3 deg vs qpos0), J1-static-drop reference: **13/15**.
* Synthetic +-5 cm root shifts on stance frames detected for all bodies: **True** (v0 rule, both rates move >= 0.5: False).
* Case-level shifts (GT scored against itself, 346 primary cases): -5: CSE 0.189, PenP95 18.07 cm, FloatRate 0.000; +0: CSE 0.000, PenP95 13.19 cm, FloatRate 0.000; +5: CSE 0.379, PenP95 8.70 cm, FloatRate 0.194 -> detected (v1, revised after seeing v0) **True**; pre-declared v0 rule (FloatRate +0.5): False.

Why the strict gate fails (measured, not assumed): J1 sits 0.5-2 cm BELOW the reference on every body, more on AMASS. The report-32 reference is a single labeller sole point rigidly attached to the ankle; J1 is the lowest corner of the collision box/mesh. On heel-strike / toe-off stance frames (median tilt of the lowest node 5-10 deg) the corner is lower than the sole point. On flat-foot stance frames the two agree within 1 cm for 13/15 bodies, and at qpos0 the J1 static drop equals report 32's true drop to < 0.02 cm on 14/15 bodies. R1 is the exception: report 32 measured R1's drop inside the ankle_constraint subtree and missed the real sole (ankle_roll link, merged into ankle_pitch), so its delta is wrong by 3.9 cm; J1 uses the corrected value (DELTA_COL_CM in ground_judge.py). Consequence for E1: every arm and the GT are measured on the same J1 ruler, so the tilt effect cancels in paired comparisons; absolute PenRate of the GT itself is not zero (see gt_PenRate below).

## 2. Heldout root bias of A0 (TRAIN-split windows)

c_b = median over TRAIN-split windows of the per-window median, over GT stance frames (J1 GT, GT root convention: a foot region < 2.5 cm and no non-foot region < 2.5 cm), of zmin(A0 pred, raw predicted root) - zmin(GT).  Applied to EVERY arm as pred_root - c_b/100 for heldout bodies (sensor_on value for both modes).

| body | applied c_b (cm) | windows | stance frames | IQR of window medians | sensor_off c_b (diag) | pred root p50 (m) | GT root p50 (m) |
|---|---|---|---|---|---|---|---|
| fourier_gr3 | -11.19 | 11 | 1407 | -13.75 .. -9.13 | -12.61 | 0.818 | 0.969 |
| kuavo_s45 | +1.57 | 11 | 1228 | 0.63 .. 4.89 | -0.16 | 0.906 | 0.902 |
| unitree_h2 | -11.44 | 11 | 1401 | -13.02 .. -10.27 | -11.74 | 0.850 | 0.949 |
| unitree_r1 | +0.04 | 11 | 1407 | -0.91 .. 1.05 | -0.56 | 0.683 | 0.673 |

## 3. Headroom: A0 / B0 vs GT (human source, primary cases)

Primary = human_smpl source on split-1 contact-rich windows (listed + mined) for all 15 targets, plus the 4 train-split ground windows for the 4 heldout targets. GT is scored by the same J1; CSE_GT and FloatRate of GT against itself are 0.

| run | mode | targets | windows | cases | CSE_GT | CSE_GT_nonfoot | CSE_snap | FloatRate | PenRate | PenP95_cm | gt_PenP95_cm | SelfPen | gt_SelfPen | pos_err_mm | CSE_human | gt_CSE_human | pred_nonfoot_contact_frac | gt_nonfoot_contact_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | sensor_off | seen | 22 | 242 | 0.766 | 0.901 | 0.717 | 0.517 | 0.302 | 4.56 | 13.27 | 0.418 | 0.315 | 281.09 | 0.795 | 0.393 | 0.096 | 0.832 |
| A0 | sensor_off | heldout | 26 | 104 | 0.781 | 0.918 | 0.751 | 0.499 | 0.316 | 5.36 | 12.88 | 0.450 | 0.141 | 376.02 | 0.796 | 0.387 | 0.102 | 0.768 |
| A0 | sensor_on | seen | 22 | 242 | 0.763 | 0.909 | 0.715 | 0.502 | 0.297 | 4.26 | 13.27 | 0.398 | 0.315 | 280.39 | 0.794 | 0.393 | 0.082 | 0.832 |
| A0 | sensor_on | heldout | 26 | 104 | 0.784 | 0.924 | 0.751 | 0.498 | 0.300 | 4.85 | 12.88 | 0.423 | 0.141 | 374.97 | 0.801 | 0.387 | 0.086 | 0.768 |
| A0 shuffled-source | sensor_on | seen | 22 | 242 | 0.747 | 0.934 | 0.718 | 0.428 | 0.323 | 4.46 | 13.27 | 0.353 | 0.315 | 348.10 | 0.760 | 0.393 | 0.079 | 0.832 |
| A0 shuffled-source | sensor_on | heldout | 26 | 104 | 0.786 | 0.935 | 0.760 | 0.432 | 0.310 | 5.02 | 12.88 | 0.377 | 0.141 | 441.09 | 0.795 | 0.387 | 0.079 | 0.768 |
| B0 | sensor_off | seen | 22 | 242 | 0.828 | 0.940 | 0.734 | 0.634 | 0.219 | 1.88 | 13.27 | 0.404 | 0.315 | 275.07 | 0.852 | 0.393 | 0.041 | 0.832 |
| B0 | sensor_off | heldout | 26 | 104 | 0.855 | 0.956 | 0.753 | 0.626 | 0.163 | 1.39 | 12.88 | 0.576 | 0.141 | 379.73 | 0.872 | 0.387 | 0.007 | 0.768 |
| B0 | sensor_on | seen | 22 | 242 | 0.814 | 0.940 | 0.731 | 0.615 | 0.231 | 1.87 | 13.27 | 0.388 | 0.315 | 274.73 | 0.842 | 0.393 | 0.041 | 0.832 |
| B0 | sensor_on | heldout | 26 | 104 | 0.854 | 0.958 | 0.750 | 0.622 | 0.157 | 1.33 | 12.88 | 0.558 | 0.141 | 378.18 | 0.872 | 0.387 | 0.004 | 0.768 |

Headroom gate (numeric part; the plan also requires lloyd to see it in [human | GT | A0] videos):

* A0_seen: CSE gap 0.763, FloatRate gap 0.502, PenP95 gap -9.01 cm -> numeric PASS
* B0_seen: CSE gap 0.814, FloatRate gap 0.615, PenP95 gap -11.39 cm -> numeric PASS
* A0_heldout: CSE gap 0.784, FloatRate gap 0.498, PenP95 gap -8.04 cm -> numeric PASS
* B0_heldout: CSE gap 0.854, FloatRate gap 0.622, PenP95 gap -11.55 cm -> numeric PASS
* Chance level of the contact pattern (GT of another window, same target, vs the case GT): CSE_GT 0.535, CSE_snap 0.764 (26 windows). A0 - chance = +0.251 (perm p 1.0000).
* Shuffled-source floor: CSE_GT(shuffled) - CSE_GT(A0) = -0.006 over 26 windows (perm p 0.5731) -> ruler requirement >= 0.15: **FAIL**
  * CSE_snap: shuffled - A0 = +0.008
  * pos_err_mm: shuffled - A0 = +64.745

## 4. Paired comparisons (window-cluster tests, human primary cases)

| comparison | metric | mean diff | windows | sign p | perm p | bodies neg/pos |
|---|---|---|---|---|---|---|
| A0: sensor_on - sensor_off | CSE_GT | -0.0013 | 26 | 0.577 | 0.6039 | 9/6 |
| A0: sensor_on - sensor_off | FloatRate | -0.0096 | 26 | 1.000 | 0.0193 | 11/4 |
| A0: sensor_on - sensor_off | PenP95_cm | -0.3183 | 26 | 0.995 | 0.0046 | 13/2 |
| A0: sensor_on - sensor_off | pos_err_mm | -0.8877 | 26 | 0.986 | 0.0132 | 15/0 |
| B0: sensor_on - sensor_off | CSE_GT | -0.0091 | 26 | 1.000 | 0.0000 | 14/1 |
| B0: sensor_on - sensor_off | FloatRate | -0.0133 | 26 | 1.000 | 0.0000 | 13/2 |
| B0: sensor_on - sensor_off | PenP95_cm | -0.0186 | 26 | 0.968 | 0.2080 | 8/7 |
| B0: sensor_on - sensor_off | pos_err_mm | -0.7491 | 26 | 0.962 | 0.0035 | 13/2 |
| B0 - A0 (sensor_on) | CSE_GT | +0.0560 | 26 | 0.084 | 0.0198 | 3/12 |
| B0 - A0 (sensor_on) | FloatRate | +0.1182 | 26 | 0.279 | 0.0266 | 4/11 |
| B0 - A0 (sensor_on) | PenP95_cm | -2.6172 | 26 | 1.000 | 0.0001 | 14/1 |
| B0 - A0 (sensor_on) | pos_err_mm | -1.2155 | 26 | 0.577 | 0.7725 | 13/2 |
| B0 - A0 (sensor_off) | CSE_GT | +0.0638 | 26 | 0.084 | 0.0066 | 2/13 |
| B0 - A0 (sensor_off) | FloatRate | +0.1220 | 26 | 0.163 | 0.0232 | 3/12 |
| B0 - A0 (sensor_off) | PenP95_cm | -2.9169 | 26 | 1.000 | 0.0001 | 15/0 |
| B0 - A0 (sensor_off) | pos_err_mm | -1.3540 | 26 | 0.423 | 0.7514 | 13/2 |

Decision-rule dry run (B0 as if it were a factor arm vs A0; illustrative only): verdict **fail**; dCSE +0.0560 vs threshold -0.030, perm p 0.990; reasons: ['no effect beyond threshold / permutation / body-sign criteria', 'guard regression: FloatRate,SelfPen']; pending: ['seed_floor_missing (SF_s1 not available)', 'twin not run / not provided', 'lloyd video review'].

## 5. A0 sensor_on per window (human source; seen / heldout targets)

| window | kind | human non-foot frac | CSE_GT seen | CSE_GT heldout | FloatRate heldout | PenP95 heldout (cm) | GT PenP95 heldout (cm) | mm heldout |
|---|---|---|---|---|---|---|---|---|
| fallAndGetUp1_subject4@4136 | unseen_motion_listed | 0.92 | 0.851 | 0.941 | 0.801 | 1.35 | 21.64 | 347.9 |
| fallAndGetUp1_subject4@4488 | unseen_motion_listed | 1.00 | 0.810 | 0.874 | 0.656 | 4.72 | 22.69 | 384.6 |
| pushAndStumble1_subject2@5488 | unseen_motion_listed | 0.83 | 0.927 | 0.935 | 0.789 | 2.92 | 24.59 | 441.2 |
| multipleActions1_subject3@136 | unseen_motion_listed | 1.00 | 0.668 | 0.812 | 0.254 | 2.01 | 19.35 | 468.6 |
| push1_subject2@2280 | unseen_motion_listed | 1.00 | 0.696 | 0.673 | 0.326 | 9.66 | 12.23 | 192.5 |
| ACCAD__Male2Walking_c3d__B24_-__walk_to_crouch_stageii@48 | unseen_motion_listed | 0.16 | 0.693 | 0.680 | 0.463 | 1.60 | 2.21 | 126.5 |
| pushAndStumble1_subject5@5384 | guard | 0.00 | 0.521 | 0.430 | 0.209 | 5.45 | 2.48 | 210.9 |
| walk2_subject3@3508 | guard | 0.00 | 0.392 | 0.451 | 0.352 | 2.67 | 1.60 | 179.8 |
| fallAndGetUp1_subject4@936 | unseen_motion_mined | 1.00 | 0.724 | 0.616 | 0.205 | 8.22 | 8.65 | 303.2 |
| fallAndGetUp1_subject4@1344 | unseen_motion_mined | 1.00 | 0.939 | 1.000 | 0.998 | 0.00 | 1.85 | 594.4 |
| fallAndGetUp1_subject4@3536 | unseen_motion_mined | 1.00 | 0.892 | 0.927 | 0.857 | 0.60 | 15.47 | 463.1 |
| fallAndGetUp1_subject4@3952 | unseen_motion_mined | 1.00 | 0.938 | 0.950 | 0.812 | 0.72 | 19.29 | 476.5 |
| multipleActions1_subject3@320 | unseen_motion_mined | 0.74 | 0.516 | 0.602 | 0.203 | 7.95 | 12.69 | 340.9 |
| multipleActions1_subject3@1792 | unseen_motion_mined | 0.59 | 0.627 | 0.586 | 0.368 | 13.67 | 13.61 | 361.6 |
| multipleActions1_subject3@4296 | unseen_motion_mined | 1.00 | 0.839 | 0.863 | 0.414 | 1.28 | 17.51 | 462.3 |
| push1_subject2@1792 | unseen_motion_mined | 1.00 | 0.539 | 0.550 | 0.016 | 6.29 | 10.69 | 185.5 |
| push1_subject2@1920 | unseen_motion_mined | 1.00 | 0.498 | 0.487 | 0.000 | 11.70 | 11.03 | 182.5 |
| push1_subject2@2048 | unseen_motion_mined | 1.00 | 0.687 | 0.664 | 0.262 | 8.35 | 12.13 | 191.4 |
| push1_subject2@2408 | unseen_motion_mined | 1.00 | 0.857 | 0.848 | 0.633 | 2.68 | 14.78 | 305.8 |
| pushAndStumble1_subject2@4488 | unseen_motion_mined | 0.62 | 0.853 | 0.868 | 0.664 | 6.67 | 26.59 | 438.6 |
| walk2_subject3@5344 | unseen_motion_mined | 0.79 | 0.696 | 0.407 | 0.326 | 9.16 | 2.76 | 305.5 |
| walk2_subject3@5928 | unseen_motion_mined | 0.95 | 0.739 | 0.776 | 0.486 | 10.45 | 7.06 | 423.6 |
| BMLmovi__Subject_31_F_MoSh__Subject_31_F_20_stageii@72 | unseen_motion_mined | 1.00 | 0.910 | 0.673 | 0.500 | 3.22 | 6.55 | 363.4 |
| HDM05__tr__HDM_tr_03-03_02_120_stageii@104 | unseen_motion_mined | 0.79 | 0.882 | 1.000 | 0.693 | 1.75 | 0.72 | 424.5 |
| ground1_subject5@3568 | unseen_body | 1.00 | n/a | 0.970 | 0.725 | 1.17 | 11.63 | 436.1 |
| ground2_subject3@5208 | unseen_body | 1.00 | n/a | 0.953 | 0.797 | 2.29 | 9.91 | 500.1 |
| ground1_subject1@4408 | unseen_body | 0.98 | n/a | 0.850 | 0.377 | 7.46 | 14.11 | 475.4 |
| fallAndGetUp2_subject2@3344 | unseen_body | 1.00 | n/a | 0.869 | 0.334 | 0.16 | 15.22 | 553.6 |

## 6. A0 sensor_on per target body (human primary cases)

| target | heldout | root shift used (cm) | CSE_GT | CSE_snap | FloatRate | PenP95 (cm) | GT PenP95 (cm) | SelfPen | GT SelfPen | mm |
|---|---|---|---|---|---|---|---|---|---|---|
| booster_k1 | False | +6.40 | 0.708 | 0.736 | 0.460 | 4.19 | 13.30 | 0.327 | 0.381 | 165.0 |
| booster_t1 | False | +7.15 | 0.806 | 0.669 | 0.551 | 1.84 | 11.36 | 0.239 | 0.839 | 250.0 |
| booster_t1_29dof | False | +9.65 | 0.692 | 0.725 | 0.309 | 6.58 | 17.20 | 0.218 | 0.599 | 278.6 |
| fourier_gr3 | True | -11.19 | 0.707 | 0.758 | 0.397 | 4.91 | 8.64 | 0.491 | 0.126 | 344.7 |
| fourier_n1 | False | +0.02 | 0.641 | 0.601 | 0.348 | 6.39 | 8.74 | 0.390 | 0.026 | 240.4 |
| hightorque_hi | False | -0.00 | 0.695 | 0.807 | 0.220 | 8.39 | 14.52 | 0.972 | 0.884 | 217.7 |
| kuavo_s45 | True | +1.57 | 0.678 | 0.801 | 0.366 | 7.44 | 29.83 | 0.125 | 0.340 | 403.8 |
| pnd_adam_lite | False | +9.40 | 0.574 | 0.609 | 0.294 | 6.75 | 15.03 | 0.001 | 0.000 | 301.4 |
| tienkung | False | +10.00 | 0.778 | 0.785 | 0.491 | 3.84 | 19.24 | 0.006 | 0.037 | 328.3 |
| unitree_g1_23dof | False | +0.04 | 0.844 | 0.823 | 0.697 | 2.23 | 10.05 | 0.565 | 0.203 | 283.4 |
| unitree_g1_29dof | False | +0.04 | 0.845 | 0.815 | 0.701 | 2.23 | 10.32 | 0.565 | 0.098 | 286.6 |
| unitree_h1 | False | -0.00 | 0.873 | 0.744 | 0.648 | 2.27 | 17.14 | 0.916 | 0.399 | 324.6 |
| unitree_h1_2 | False | +0.00 | 0.935 | 0.553 | 0.816 | 2.11 | 9.02 | 0.175 | 0.002 | 408.3 |
| unitree_h2 | True | -11.44 | 0.878 | 0.743 | 0.654 | 4.01 | 8.52 | 0.402 | 0.073 | 418.0 |
| unitree_r1 | True | +0.04 | 0.860 | 0.705 | 0.563 | 3.25 | 5.67 | 0.666 | 0.031 | 333.4 |

## 7. Gate verdicts

* Ruler gate (strict, as pre-registered): **FAIL** (6/15). Diagnostic flat-foot: 13/15.
* Synthetic +-5 cm shifts detected: stance-level True; case-level True.
* Headroom numeric (A0_seen): PASS
* Headroom numeric (B0_seen): PASS
* Headroom numeric (A0_heldout): PASS
* Headroom numeric (B0_heldout): PASS
* Headroom video part: pending (render package; lloyd must see [human | GT | A0]).

Files: case npz + scores under OUT/eval/e1/<run>/<mode>/ (scores_j1.jsonl), heldout bias OUT/judge/heldout_root_bias_v1.json, ruler OUT/judge/ruler_v1/ruler_gate_summary.json, bindings OUT/judge/bindings_v1/.
