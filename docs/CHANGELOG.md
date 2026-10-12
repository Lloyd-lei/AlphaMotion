# GPU engine changelog (from ~/umr_gpu/README.md "Progress log")

- 10-04: UMR solver parity reached (one-step 2e-6 rad; free-run post-LQR p50 1.3e-6, max 5e-4 rad).
- 10-04: CUDA graphs, fp32 assembly, compiled IPM; GPU source prep (SMPL-X + inputs) validated vs CPU capture.
- 10-04/05: spec-v2 strict pass (dense-point contact [0,1] mm, latches, lift, collision-geom cross-check),
  floor-relative outputs, 12 parity/judge runs (3 clips x G1/GR3/H2/A3), 24 videos rendered.
- 10-07 (after reboot): soccer recaptured at 30 fps (stride 3); slip metrics on kept frames; throughput runs.
- 10-07 18:20: throughput (4 robots G1/GR3/H2/A3 per pass, shared GPU, CPU pinned to cores 0-15):
  B=256 duty 1.0: 2707 robot-frames/s, util mean 74 %, power mean 313 W (p95 470 W), 8.65 rf/J;
  B=256 duty 0.7: 1882 rf/s, util 51 %, power 240 W (p95 395 W), 7.8 rf/J.  Next: B=1024 duty 0.6.
- 10-07 18:50: B=1024 duty 0.6 (4 robots): 2235 rf/s, util 50 %, 260 W mean (p95 419 W), 8.6 rf/J, 33 GB peak;
  UMR stage alone 15.2k rf/s (163 s for 2.49 M robot-frames), strict 952 s = bottleneck (per-chunk iteration tails).
  Rewrote project_frames as a round-based global work queue (full chunks every round) -> re-test + re-bench next.
- 10-07 19:20: round-based strict queue: B=1024 duty 0.6 (4 robots): 2331 rf/s, util mean 64 %, 318 W mean
  (p95 432 W), 7.3 rf/J, 32.6 GB peak, 0 network pauses; G1 dance B=1 strict 13.3 s -> 7.7 s, same flags/judge.
- 10-07 19:25: production request (lloyd): all T0 sources are HOI (OmniContact box/ball; HiPHI 30 fps: 3795 clips,
  all is_hoi) and every OmniContact clip has its own betas -> implementing (a) beta-zero shared correspondence
  static prep (UMR's reuse_across_smplx_betas mechanism), (b) per-clip betas/scale/up in the GPU source prep,
  (c) per-clip scaled objects on the GPU (object-contact maps, robot-object rows, strict object gauge, human
  object labels), (d) resumable production runner.
- 10-07 19:38: production runner launched (16 OmniContact jobs x 4 robots-interleaved, pid in _logs/runner.pid).
- 10-07 20:00: STOPPED production (coordinator/lloyd: root jumps).  Root cause: the frame-parallel strict pass
  solves every frame independently -> different minimal corrections in neighbouring frames (root offsets vs UMR up
  to 24 cm, steps up to 27 cm/frame; 822 of 1235 gate hits on LAFAN/G1 are inside contact segments, the rest at
  requirement switches or next to failed frames); segment-mean latch targets also force a correction jump at the
  segment start.  Smoothing the correction field + re-projection did NOT fix it (re-projection re-introduces the
  per-frame minimal corrections).  Fix in progress: `strict_seq` = time-coupled pass (all chains in lock-step,
  forward in time): warm start = UMR (+) 0.88 x previous correction (corrections ramp out, never snap back),
  tracking UMR with a temporal term (w=1) to the previous output frame, online latches (target = the contact point
  where the segment's first frame put it; released only during real human slides > 30 cm/s), a contact-point slip
  check on every required region, and the temporal-continuity gate (flag 128: root velocity differs from UMR's by
  > 1 cm/frame, root rotation > 0.05 rad/frame, any joint > 0.1 rad/frame, at 30 fps).  First production job outputs
  (engine umr_gpu-6350231d1845) moved to production/v1/_invalid.
- 10-07 22:00 - 10-08 00:20: root-jump fix (strict_seq), iterations on LAFAN dance / G1 (3945 f):
  * hard continuity box per frame vs UMR's step -> root step max 4.9 cm/frame (UMR 4.8), continuity gate hits 5
    (was 1235 frame-parallel / 2211 first sequential try);
  * found: the QP temporal term pulled toward q_out(t-1) (standstill) -> every dof not held by a constraint lagged
    behind fast motion (shoulder offsets p95 0.4 rad, root heading off by up to 41-96 deg).  Now toward q_pred:
    root offset vs UMR p50 1.5 / p95 4.1 / max 12 cm, heading p50 3.3 / p95 8.7 deg, joint offset p50 0.12 rad;
  * judge-consistent output slip gate added (any region's lowest dense point in contact at t-1 and t moving
    > 1 cm/s, or median contact-point speed > 1 cm/s -> flag 32): kept-frame slip on the CPU judge 0 %
    (before the gate 16 % of kept contact frames slid, from robot-only contacts and feet spinning about the latch);
  * yaw latch back on; latch release at 5 cm / 15 deg from UMR;
  * kept fraction on this dance clip: 66.8 % (flags mostly slip: 460 frames the human foot itself slides > 30 cm/s,
    the rest held-latch releases and slip-gate frames).  Before the fix 66.9 % (with the gate) / ~93 % (no gate, jumps).
  * production run_job now runs strict_seq on a clip-major layout + human support speed per region on the GPU +
    final per-frame verification; QC json carries `continuity` and `flag_bits`.
  * OmniContact box test (Box_H_1, 3-8 clips, G1): kept 27-40 %; root cause is the DATA: the human SMPL-X hands are
    6-7 cm inside the box mesh (R_hand min-SDF p50 -64 mm, L_hand -4..-75 mm) and the trunk up to 10-12 cm inside;
    UMR follows the human (robot hands 8-12 cm inside).  The CPU factory saw the same (object pen before 120 mm).
- 10-08 04:00: final root-jump fix = strict_seq with continuity box + temporal term toward q_pred + latch release
  (5 cm / 15 deg) + leash (10 cm / 0.26 rad / 0.6 rad) + UMR floor alignment + output slip gate.  The 12 parity
  files were re-exported (results/export/*.npz, 02:29-03:58; old frame-parallel exports kept in
  results/export_old_frameparallel/).  Root steps (cm/frame scaled to 30 fps; soccer is the 90 fps capture):

| clip | robot | frames | UMR root step max cm / >5cm | OLD strict max / >5cm | NEW strict max / >5cm (kept max) | NEW root excess vs UMR p99 / max cm | NEW offset vs UMR max cm | kept OLD (OLD after continuity gate) -> NEW | kept: slip>1cm/s / ground pen / self pen / obj pen frames / human-contact solid |
|---|---|---|---|---|---|---|---|---|---|
| dance1_subject2 (30 fps) | unitree_g1_29dof | 3945 | 4.78 / 0 | 27.2 / 208 | 5.21 / 3 (5.2) | 0.95 / 0.95 | 11.62 | None (None) -> 0.663 | 0.0 / 0 / 0 / None / 1.0 |
| dance1_subject2 (30 fps) | fourier_gr3 | 3945 | 5.42 / 7 | 56.3 / 332 | 5.93 / 13 (5.61) | 0.95 / 0.95 | 11.44 | 0.883 (0.512) -> 0.523 | 0.0 / 0 / 0 / None / 1.0 |
| dance1_subject2 (30 fps) | unitree_h2 | 3945 | 6.14 / 29 | 19.2 / 197 | 6.53 / 34 (6.28) | 0.95 / 0.95 | 8.12 | 0.977 (0.689) -> 0.672 | 0.0 / 0 / 0 / None / 1.0 |
| dance1_subject2 (30 fps) | agibot_a3_t3d0 | 3945 | 6.05 / 32 | 24.1 / 247 | 6.72 / 34 (6.48) | 0.95 / 0.95 | 8.04 | 0.91 (0.566) -> 0.629 | 0.0 / 0 / 1 / None / 1.0 |
| getup0004_neutral_30fp (30 fps) | unitree_g1_29dof | 1823 | 3.13 / 0 | 55.5 / 227 | 3.47 / 0 (3.3) | 0.95 / 0.95 | 11.32 | 0.201 (0.145) -> 0.325 | 0.0 / 0 / 0 / None / 1.0 |
| getup0004_neutral_30fp (30 fps) | fourier_gr3 | 1823 | 5.3 / 3 | 32.6 / 102 | 5.93 / 8 (5.56) | 0.95 / 0.95 | 9.73 | 0.152 (0.143) -> 0.297 | 0.0 / 0 / 0 / None / 1.0 |
| getup0004_neutral_30fp (30 fps) | unitree_h2 | 1823 | 5.33 / 6 | None / None | 5.29 / 6 (5.29) | 0.95 / 0.95 | 8.98 | None (None) -> 0.338 | 0.0 / 0 / 0 / None / 1.0 |
| getup0004_neutral_30fp (30 fps) | agibot_a3_t3d0 | 1823 | 4.46 / 0 | 55.5 / 205 | 4.69 / 0 (4.69) | 0.95 / 0.95 | 14.29 | 0.359 (0.259) -> 0.337 | 0.0 / 0 / 3 / None / 1.0 |
| 20260330000586_1_17752 (90 fps) | unitree_g1_29dof | 5771 | 5.37 / 10 | 242.9 / 889 | 5.91 / 24 (5.7) | 0.95 / 0.95 | 14.19 | 0.83 (0.795) -> 0.444 | 0.0 / 0 / 0 / 0 / 1.0 |
| 20260330000586_1_17752 (90 fps) | fourier_gr3 | 5771 | 6.73 / 138 | 197.0 / 1356 | 7.12 / 181 (6.9) | 0.95 / 0.95 | 11.48 | 0.786 (0.736) -> 0.288 | 0.0 / 0 / 0 / 0 / 1.0 |
| 20260330000586_1_17752 (90 fps) | unitree_h2 | 5771 | 7.55 / 354 | 195.3 / 713 | 7.86 / 399 (7.83) | 0.95 / 0.95 | 14.44 | 0.888 (0.86) -> 0.551 | 0.0 / 0 / 0 / 11 / 1.0 |
| 20260330000586_1_17752 (90 fps) | agibot_a3_t3d0 | 5771 | 7.37 / 172 | 252.0 / 692 | 7.99 / 241 (7.56) | 0.95 / 0.95 | 13.81 | 0.853 (0.817) -> 0.484 | 0.0013 / 0 / 0 / 8 / 1.0 |

  "OLD after continuity gate" = old kept frames that would also pass the new continuity gate (old flags did not
  include the output slip gate either).  Every kept frame of the 12 new files: root step excess over UMR <= 0.95
  cm/frame, joint excess <= 0.096 rad/frame, root offset vs UMR changes <= 0.95 cm/frame (no snap-back), CPU judge:
  kept-frame contact slip > 1 cm/s 0-0.13 %, ground penetration 0 frames, self 0-3 frames, ball 0-11 frames (MuJoCo
  ball hull vs analytic sphere, known), human-contact -> robot solid 100 %.  At lloyd's frames: LAFAN/G1 286-290 root
  offset 2.0-2.6 cm (was pulled 7 cm and snapped back 8 cm), 2234-2236 tilt 23-28 deg = UMR's own 25-28 deg (was 30
  deg then snap).  Videos [human | GPU UMR raw | GPU strict] + 12-frame grids + continuity plots:
  videos/fix_jump/ (full clips + 4x slow-motion windows around the old worst jump of every combo).
- 10-08 04:00: production run_job = strict_seq_segmented + measurement cache.  Soccer G1 32 clips (61,678 frames):
  147.5 robot-frames/s end to end (strict 363 s of 418 s; before segmentation 40 rf/s), kept 69.1 %, 256 segments,
  seam disagreement p50 2.5 mm / max 7.7 cm (20 frames flagged by the re-run gates), GPU mean 150 W / util 67 %
  (shared with one parity run).  Production still STOPPED.
- 10-11: slip rule relaxed (lloyd: "有些动作本身就应该打滑"; DATA_SPEC_v2 不滑 / 1011 lloyd 放宽), see "Slip rule
  (lloyd 10-11)".  New `umr_gpu/slip.py` (one ruler for robot + human), `strict_seq` latch moves with the human's
  slip velocity while the human region slides (> 2 cm/s; released above 30 cm/s), re-latch at the output's
  position, rolling latch switch, heading latch follows UMR's increment; output gate -> 32 / 256; export +
  qc_json fields; run_parity `--outdir/--judge strict`; render_compare `--results`; production hsp on the same
  ruler (+ slip velocity) -- production code changed but NOT run (production stays STOPPED).
  Iterations on the 12 parity files (results/export_slip_v3/, reference results/export/ untouched):
  (1) latch released whenever the human slid > 2 cm/s -> robot followed UMR, whose feet slide 3.4x faster than the
  human (dance G1 kept 0.663 -> 0.496, kept in `_try2_release_policy/`); (2) human ruler without the on-floor
  condition legalised lift-off sweeps (`_smoke_before_onfloor/`); (3) final: latch target moves with the human slip
  velocity -> robot/human slip-speed ratio on bit-256 frames median 0.96-1.00.
  Before (results/export, flags == 0) vs after (flags & 255 == 0); root step cm/frame at 30 fps (UMR's own max:
  dance 4.8-6.1, get-up 3.1-5.3, soccer 5.4-7.6); slip speed = ground_slip_speed max m/s (old gauge -> new ruler,
  all frames (kept frames)):

| clip | robot | frames | kept before | kept after | bit256 share (kept) | bit32 before -> after | root step max cm/f@30 (kept) | slip speed max m/s old -> new (kept) |
|---|---|---|---|---|---|---|---|---|
| case3_kick_right__2026033000 | agibot_a3_t3d0 | 5771 | 0.484 | 0.734 | 0.584 (0.536) | 2980 -> 1187 | 7.41 (7.41) | 7.17 -> 10.81 (4.953) |
| case3_kick_right__2026033000 | fourier_gr3 | 5771 | 0.288 | 0.549 | 0.461 (0.408) | 4110 -> 1468 | 7.39 (6.05) | 8.06 -> 8.36 (4.695) |
| case3_kick_right__2026033000 | unitree_g1_29dof | 5771 | 0.444 | 0.726 | 0.6 (0.546) | 3205 -> 1242 | 5.78 (5.78) | 7.57 -> 6.49 (4.387) |
| case3_kick_right__2026033000 | unitree_h2 | 5771 | 0.551 | 0.781 | 0.619 (0.578) | 2591 -> 1045 | 7.55 (7.15) | 7.57 -> 7.41 (5.918) |
| hiphi_getup__getup0004_neutr | agibot_a3_t3d0 | 1823 | 0.337 | 0.363 | 0.214 (0.054) | 1209 -> 734 | 4.68 (3.97) | 6.07 -> 2.84 (1.304) |
| hiphi_getup__getup0004_neutr | fourier_gr3 | 1823 | 0.297 | 0.35 | 0.139 (0.044) | 1282 -> 655 | 5.93 (5.57) | 3.74 -> 4.45 (1.632) |
| hiphi_getup__getup0004_neutr | unitree_g1_29dof | 1823 | 0.325 | 0.383 | 0.175 (0.069) | 1230 -> 668 | 3.55 (3.55) | 2.8 -> 2.51 (1.624) |
| hiphi_getup__getup0004_neutr | unitree_h2 | 1823 | 0.338 | 0.37 | 0.189 (0.06) | 1206 -> 596 | 5.51 (5.51) | 4.34 -> 3.12 (2.09) |
| lafan1_smplx__dance1_subject | agibot_a3_t3d0 | 3945 | 0.629 | 0.816 | 0.463 (0.438) | 1464 -> 383 | 6.72 (6.48) | 4.79 -> 6.78 (4.066) |
| lafan1_smplx__dance1_subject | fourier_gr3 | 3945 | 0.523 | 0.729 | 0.438 (0.396) | 1877 -> 450 | 5.81 (5.49) | 4.5 -> 5.07 (3.41) |
| lafan1_smplx__dance1_subject | unitree_g1_29dof | 3945 | 0.663 | 0.873 | 0.585 (0.561) | 1329 -> 390 | 5.17 (5.16) | 5.34 -> 5.5 (3.441) |
| lafan1_smplx__dance1_subject | unitree_h2 | 3945 | 0.672 | 0.849 | 0.502 (0.478) | 1289 -> 442 | 6.53 (6.18) | 5.18 -> 5.29 (5.289) |
  Remaining bit 32: robot regions sliding while the human region is still or not on the floor (get-up whole-body
  contacts, UMR feet scraping during swing), and sweeps > 30 cm/s where UMR's foot is > 1.5x the human's.
  Get-up stays at 35-38 % kept because of penetration (1) / contact-solid (8) flags, not slip.  Open point for
  lloyd: legal follows include fast floor-grazing sweeps (human foot <= 1 cm above the floor moving > 1 m/s:
  soccer 146-401 kept frames of 5771, dance 25-44 of 3945); a speed cap on "legal slide" would drop them.
  Videos [human | GPU UMR raw | new strict] (4x slow, bit-256 windows): videos/slip_v3/.
