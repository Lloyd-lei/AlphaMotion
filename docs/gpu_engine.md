# umr_gpu — GPU-batched UMR retargeting + strict post-pass (DATA_SPEC_v2)

Status file — updated as work progresses (network/restarts: continue from "Progress log" at the bottom).

## What it is

A re-implementation of UMR's per-frame retarget solver (`solve_frame_body_segment_qp` in
`UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py`, commit d1cdaed) that runs thousands of (clip, robot)
chains in lock-step on one GPU, followed by lloyd's strict constraints (DATA_SPEC_v2) solved on the GPU for all
frames in parallel, labels, and a spec-v2 export.

```
SMPL-X forward (GPU, shared by all robots)
 -> per robot: source prep on GPU (slots, normal targets, ground maps, self-contact maps, q_init)  source_gpu.py
 -> UMR Gauss-Newton/QP step per frame, CUDA graph, batched IPM                                     solver.py, ipm.py
 -> UMR LQR filter (banded Cholesky per piece)                                                       lqr_band.py
 -> labels + strict pass, all frames in parallel                                                     strict.py, strict_fp.py
 -> export: floor-relative qpos + sparse dense-point contacts + region labels + flags                export.py
```

Code: `/home/arenalabs/umr_gpu`; outputs: `/var/tmp/alphamotion/umr_factory/umr_gpu` (`problems/` captured UMR
problems, `results/parity/` parity + judge, `results/export/` spec-v2 files, `videos/` comparison videos,
`results/bench/` throughput runs, `logs/`).  Env: `/var/tmp/alphamotion/umr_factory/umr/env` (unchanged; UMR checkout
untouched).  The CPU factory code was read, never edited; a frozen copy of its ruler is in
`eval/factory_ruler_snapshot/`.

## Components

| file | content |
|---|---|
| `capture/umr_capture_run.py`, `run_capture.py` | runs stock UMR (robot = factory staged MJCF) and records every per-frame solver input + CPU result (exact parity reference), plus static prep data (binding, T-pose normals) |
| `umr_gpu/kin.py` | torch kinematic tree from the compiled MjModel: FK, point/rotational Jacobians, integratePos/differentiatePos — matches MuJoCo to 1e-15 |
| `umr_gpu/solver.py` | UMR's residuals/weights/selection rules, constraints, warm start, anchors; CUDA-graph streaming; LQR |
| `umr_gpu/ipm.py` | batched primal-dual IPM (Mehrotra) = the Clarabel QP incl. the l2 trust region (as a convex quadratic constraint) and slack rows; torch.compile'd step, fp64 or fp32 |
| `umr_gpu/source_gpu.py`, `selfcontact_warp.py` | SMPL-X forward + all UMR per-frame inputs on GPU (self-contact maps via Warp kernels) |
| `umr_gpu/geometry.py`, `narrow_warp.py` | collision geoms: analytic SDFs (sphere/capsule/cylinder/box), outer-polytope SDF of mesh hulls; OBB broadphase; Warp narrow phase; ground/object point gauges; collision-geometry ground gauge |
| `umr_gpu/strict.py`, `strict_fp.py` | spec-v2 gauges/flags/QP rows; frame-parallel projection with contact latches, yaw latch, lift rule, blending |
| `umr_gpu/chains.py`, `scripts/run_factory.py` | production driver: clips packed into B chains, several robots per pass |
| `umr_gpu/export.py` | DATA_SPEC_v2 per-(clip,robot) output |
| `eval/judge.py` | independent CPU ruler (MuJoCo mj_geomDistance + dense points) |
| `eval/run_parity.py`, `rejudge.py`, `summarize.py`, `render_compare.py` | parity/strictness evaluation, tables, videos |
| `tests/` | unit/parity tests (kinematics, one-step, free-run, self-collision, source prep, LQR, chunks) |

## How UMR's per-frame problem is reproduced (UMR stage)

Inputs per frame are exactly what UMR passes to `solve_frame_body_segment_qp` (captured, or recomputed on the GPU by
`source_gpu.py`, validated against the capture: source slots max 0.0014 mm, normal targets <= 0.26 deg, ground
distances 0.0005 mm, identical snapping; self-contact maps: 2-6 pair differences in 3945 frames (fp32 ties)).

* residual rows: surface points (sqrt point cost per body segment), T-pose-offset normals (sqrt normal cost),
  source self-contact map (two modes merged with max weight, active when robot pair distance < source distance),
  ground-contact map (cost x strength on z), ground anchors (latched xy at activation, z = 0), object contact vectors,
  smoothness + temporal terms on scalar joints, damping 0.01.  Same weights (incl. UMR's non-sqrt row weights), same
  top-k selections.
* constraints: joint-limit box on dq, l2 trust region ||dq_local|| <= 0.4 (Clarabel SOC; here an equivalent convex
  quadratic constraint), hard linearised ground clearance on the 128 lowest of the 4096 robot slots within 5 cm,
  robot-object rows (HOI: slack cost 5000; point-sampled collision geoms vs the object SDF instead of
  mj_geomDistance), integratePos + clamp, 15 (10 for HOI) Gauss-Newton steps on the first frame, 1 afterwards,
  sequential warm start, LQR filter (same banded system).
* QP: batched Mehrotra primal-dual IPM (`ipm.py`); fp64 reproduces Clarabel to 1e-9; production uses fp32 with a
  fixed 24 iterations inside a CUDA graph (converged elements frozen).
* kinematics: torch kinematic tree from the compiled MjModel (`kin.py`), FK/Jacobians/integratePos equal MuJoCo to
  1e-15 (fp64).  MuJoCo-Warp was not installed in the UMR env and not needed.

## Strict pass (DATA_SPEC_v2, `strict.py` + `strict_fp.py::strict_seq`)

Default since 10-07 night: **sequential, time-coupled** (`strict_seq`).  All clips/chains walk forward in time in
lock-step on the GPU (one row per clip; production uses a clip-major layout so pieces of the UMR stage do not cut
the strict pass).  Per frame t:

* prediction  q_pred(t) = q_out(t-1) (+) [q_UMR(t) (-) q_UMR(t-1)]   (previous output moved by UMR's own increment);
  warm start = q_pred moved toward UMR (+) 0.88 x previous correction (at most the continuity box), i.e. a correction
  that is no longer needed decays with a half-life of ~5 frames;
* **continuity box (hard QP bounds)**: q(t) (-) q_pred(t) <= 0.95 cm per frame on the root position (0.55 cm per
  axis), 0.0475 rad on the root rotation, 0.095 rad on every joint (scaled to 30 fps).  The correction relative to
  UMR can therefore only ramp, it can never snap in or out within a frame;
* if the warm start satisfies the spec it is kept, otherwise up to 30 linearised QP steps (tracking the warm start;
  temporal term w=1 toward q_pred - NOT toward q_out(t-1), which would make every unconstrained dof lag behind
  fast motion; trust region, line search 1 / 0.5; a frame whose violation has not dropped by 30 % after 8 steps is
  given up) inside the box; frames that still fail are FLAGGED;
* online latches: at the first frame of a human contact segment the region's lowest dense point of the UMR
  touch-down pose is latched (ground: xy fixed, z = floor + 0.45 mm; object: in the object frame) together with the
  body's heading (yaw +-1 mrad).  A latch is released (its frames then fail the slip checks and are flagged) while the
  human region really slides (> 30 cm/s) or when holding it would pull the robot too far from UMR (latched point
  > 5 cm from where UMR has it, or heading > 15 deg off UMR's) - holding a sliding human foot forever made the robot
  drift (up to 1.4 m in a test) and twist (> 40 deg);
* **leash**: the total deviation from UMR is also bounded in the QP (root position 10 cm per axis, root rotation
  0.26 rad per axis, joints 0.6 rad; projected into the continuity box so continuity always wins).  Without it an
  unsatisfiable stretch (soccer/GR3: the human stands with one foot on the ball and both feet on the floor) made the
  output walk 50 cm away from UMR at the box limit while every frame stayed flagged;
* floor alignment: UMR puts the feet on its own ground plane; every UMR frame of a clip is first shifted in z by the
  median gap between the robot's feet and the calibrated human floor over the human's foot-contact frames (the CPU
  factory's `applied_shift`; G1 dance -1.7 mm, GR3 soccer +18.5 mm), so a constant floor offset is not re-solved
  every frame.

* production uses `strict_seq_segmented`: every clip is cut into 256-frame segments, each run as its own row with
  90 warm-up frames taken from before its start (all rows in lock-step: ~346 sequential steps per job instead of the
  longest clip's length); consecutive segments are stitched at the frame of the overlap's last two thirds where the
  two runs agree best, and the continuity gate + output slip gate are re-run on the stitched clip (a visible seam is
  flagged, never hidden).  LAFAN dance / G1: 16 segments, seam disagreement p50 1.4 / max 4.0 (root m + 0.1 x joint
  rad), 0 frames newly flagged by stitching, kept 66.6 % (full sequential 66.3 %), 173 s instead of 509-733 s.
* the accepted candidate's measurement is reused as the next QP iteration's measurement (bit-identical output on
  LAFAN dance / G1: max |dq| = 0, same flags; ~30 % less strict time).

| rule | gauge (GPU) | QP rows | verify (flag bit) |
|---|---|---|---|
| contact solid | required region: min over its DENSE points (4096 / robot) | lowest point <= 0.8 mm | in [-0.5, 1] mm (8 / 16 object) |
| no penetration | dense points + visual-hull vertices + analytic lowest points of MuJoCo collision primitives; object SDF | 256 lowest points >= 0.1 mm; colliding primitives | >= -0.5 mm (1 / 2) |
| no self penetration | MuJoCo-filtered collision-geom pairs: OBB broadphase, Warp narrow phase | witness rows >= +1.5 mm | GPU distance >= +0.5 mm (4) |
| no slip (10-11 rule, `slip.py`) | latched point + heading per (region, human contact segment); latch held only while the human region does not slide; rolling moves the latch to the new lowest point; heading follows UMR's increment | box +-0.1 mm, heading +-1 mrad | **output slip gate** on the slip.py ruler: region slip speed = min tangential speed over its dense points in [-0.5, 1] mm at t-1 and t (contact / pivot point). > 1 cm/s while the human region slides (> 2 cm/s, same ruler on SMPL-X support points) and robot <= max(2 cm/s, 1.5 x human) -> **256 follow_human_slip (kept)**; otherwise 32. Object latches: deviation > 0.15 mm in the object frame (32) |
| no flicker | hysteresis enter < 1 mm / leave > 2 mm, segments >= 3 frames | - | - |
| human consistency | human region in contact (SMPL-X slots, same calibrated floor, enter 1 cm / leave 2 cm) -> robot region required; human region > 2 cm off -> robot region > 5 mm | lift rows >= 5.5 mm | (64) |
| **temporal continuity** | output step vs UMR step, every frame | continuity box | root position step differs from UMR's by > 1 cm/frame, root rotation > 0.05 rad/frame, any joint > 0.1 rad/frame (at 30 fps) (128) |

Frames failing any rule are FLAGGED (bitmask `flags`, legend in `qc_json.flag_bits`) and must be excluded: a frame is
excluded iff `flags & qc_json.drop_mask` (= 255) != 0.  Bit 256 (follow_human_slip) is informational, never excluded.  The QC
json of every output also carries `continuity` (root step excess over UMR p50/p99/max, joint excess max, root step
max, root offset from UMR max).

### Slip rule (lloyd 10-11, `umr_gpu/slip.py`; DATA_SPEC_v2 "不滑 / 1011 lloyd 放宽")

"Some motions should slide": a robot contact may slide when the same human region slides.
* **One ruler** (`slip.py`) for robot and human: region slip speed between t-1 and t = MIN tangential speed over the
  region's points in contact at both frames (robot: dense points in [-0.5, 1] mm; human: SMPL-X support slots within
  3 mm of the region's lowest slot, region labelled in contact AND its lowest slot <= 1 cm above the floor,
  Savitzky-Golay-5 smoothed positions).  The min is the speed of the contact / pivot point: heel->toe rolling and
  pivoting about the contact point are 0, a real slide moves every contact point.  (Before: the lowest point of
  t-1, or the median of all contact points -> a pivot about any other point counted as slip.)
* per (frame, region), robot speed rs > 1 cm/s:  human speed hs > **2 cm/s** and rs <= **max(2 cm/s, 1.5 x hs)** ->
  bit **256 follow_human_slip** (informational, frame KEPT); else bit **32**.  Drop mask = 255 (bits 1-128).
* thresholds from data (old exports, G1, slip.py ruler): human feet in quasi-static stance (HiPHI get-up) slide p25
  0.1 / p50 0.8 cm/s on the floor -> mocap noise ~1 cm/s, so 2 cm/s (2x the spec) separates jitter from sliding;
  dance / soccer feet on the floor are above 2 cm/s in 85 % / 74 % of contact frames (pivots, glides).  The kept
  fraction moves < 0.5 pp between 1 and 3 cm/s.  Robot/human speed ratio on following frames: median 1.2-1.6 on the
  old output (inflated by latch catch-up); robots are smaller than the human, so 1.5x is a 50 % margin; 2x would
  legalise ~1.5 pp more.  The on-floor (<= 1 cm) condition was added after the first test: without it a region
  lifting off (8-17 mm, still labelled 'in contact' by the 2 cm leave threshold) "slid" at 1-2 m/s and legalised
  robot scrapes of up to 1.8 m/s.
* solver (`strict_seq`): while the human region slides (> 2 cm/s) the ground latch target MOVES with the human's
  slip velocity (`slip.human_slip_speed(..., return_vec=True)`: support points' mean direction, min-speed
  magnitude) so the robot slides like the human (robot/human speed ratio median 0.96-1.00 on bit-256 frames);
  below 2 cm/s the target is fixed; above 30 cm/s (a sweep) the latch is released and the robot follows UMR
  (releasing at 2 cm/s instead made the robot follow UMR's feet, 3.4x faster than the human); a latch re-taken inside a contact segment is held where the OUTPUT had the point at t-1 (no pull back
  to UMR's point = no catch-up slide; drift releases still re-align with UMR); rolling: when UMR's lowest point of a
  latched region moves > 2 mm below the latched point, the latch moves to the new lowest point (heel -> toe); the
  heading latch follows UMR's heading increment of that body (human pivots) instead of freezing at touch-down.
  Bit 32 is rebuilt from the output gate (+ object-latch deviation); solver-internal ground latch / heading
  deviations only drive the QP.
* export: `ground_slip_speed` (robot, new ruler, t-1 -> t), `human_ground_slip_speed`, `ground_slip` (32 regions),
  `ground_slip_follow` (256 regions); qc_json: `kept_frames`, `follow_human_slip_frames`, `ground_follow_frac_of_contact`,
  `ground_slip_speed_kept_max/_p99`, `ground_slip_speed_max`, `slip_rule`, `drop_mask`, `flags_hist` incl. 256.
* the 14-34 m/s `ground_slip_speed` values were in `production/v1/omnicontact/*` from the invalid frame-parallel
  engine (umr_gpu-6350231d1845): root jumps of 15-40 cm/frame (e.g. box carry 169: root z 0.75 -> 0.50 -> 0.76 m
  in two frames) with 1-2 foot points staying inside the [-0.5, 1] mm band, and the old gauge took the forward-
  difference median of whatever (1-2) points were in the band.  Not a dt bug; the speeds were real teleports.  The
  current engine's old exports reached 3-8 m/s from feet sweeping through the band (a point within 1 mm of the floor
  at t and t+1 moving 18 cm/frame).  New gauge: min over in-band points, backward difference aligned with the
  flags; those frames are bit 32.

History: the frame-parallel pass (`strict_chains`, all frames projected independently around UMR, segment-mean
latch targets, Gaussian blending) produced root jumps (independent minimal corrections in neighbouring frames, up to
27 cm/frame on LAFAN/G1, 45 cm/frame on get-up/H2); it is kept only for reference.

Floor convention: every output is floor-relative (calibrated human floor = z 0); the floor value is stored as
`floor_offset_removed` (raw world), `floor_height` = 0.  Earlier `q_out` naming is gone (the main session's eval had
subtracted the floor twice).

## Evaluation protocol

* Clips (all SMPL-X sources available locally): LAFAN `dance1_subject2` (3945 f, 30 fps), HiPHI
  `Getting_up-get_up_0004` (neutral body, 90 -> 30 fps, 1823 f, lying/pushing up: whole-body ground contact),
  OmniContact soccer `case3_kick_right` (ball, 5771 f at 90 fps; also recaptured at 30 fps).
* Robots: G1 29-DoF, Fourier GR3, Unitree H2, AgiBot A3 (factory staged MJCF + surface asset).
* CPU reference: stock UMR (defaults; robot = factory staged MJCF), captured by `capture/` (same inputs for both).
  CPU factory strict output (umr_factory_code) exists only for LAFAN x G1/GR3.
* One ruler for every variant: `eval/judge.py` (CPU, MuJoCo `mj_geomDistance` for collision geometry vs floor /
  self pairs / object, dense points + hull vertices for clearance, dense points for contact, consecutive-frame
  speed of contact points for slip).

* Temporal continuity evidence: `eval/plot_continuity.py` (root step UMR vs strict, step excess, root offset;
  needs matplotlib -> run with the system python3), `eval/summarize_fix.py` (before/after table),
  `eval/render_compare.py --cols umr [--frames a:b --slow 4]` ([human | GPU UMR raw | GPU strict] videos),
  `scripts/parity_queue.sh` (parity + export queue, UMRG_DUTY), `scripts/render_fix.sh` (renders after each run).

## Deviations from UMR (UMR stage) and why

1. QP solver: batched IPM instead of Clarabel; in fp64 identical to 1e-9; production fp32 + 24 fixed iterations
   (joint diff vs fp64 p50 3e-6 rad, p99 2.5e-4 rad, max 1.4e-3 rad on LAFAN/G1).
2. Robot-object penetration (HOI): one row per robot collision geom from point samples vs the object SDF (sphere /
   box analytic, other meshes a 96^3 SDF grid) instead of `mj_geomDistance` on the object hull -> the largest
   one-step differences on soccer frames with the ball near a geom (p99 5e-4 rad, max 1.1e-2 rad).
3. Ties in UMR's top-k selections (ground map / anchors / ground constraint slots) are broken by index on the GPU;
   continuous keys make ties practically absent.  UMR's object-contact selection is reproduced in the capture.
4. Window/chunk starts: a production piece starts from UMR's own frame-0 initialisation (15 GN steps); UMR's
   sequential solve forgets its start state within ~100-200 frames (measured: joint diff 1e-2 rad after 64 frames,
   1e-4 after 192, 3e-5 after 256), so pieces longer than `max_piece` (1800) are split with a 192-frame warm-up.
   Chunk-boundary error (LAFAN/G1, Lc 512, overlap 384, stitched at the overlap middle): max joint diff
   5.1e-4 rad, max surface-point diff 0.13 mm; with a 32-frame overlap the error is 0.16 rad / 21 mm (too short).
5. The source preparation (SMPL-X forward, binding, transported normals, contact maps) runs on the GPU with UMR's
   formulas; the static parts (slot correspondence training, robot slot binding, T-pose normals) still come from a
   one-off UMR CPU run per (body shape, robot) (`capture/`, ~1-2 min).

## Remaining gaps (honest list)

* (10-11: the slip rule was relaxed, see "Slip rule (lloyd 10-11)" and the 10-11 progress-log table; the text
  below describes the 10-08 state.)  Kept fraction after the root-jump fix (sequential strict pass, all spec checks
  incl. the output slip gate and the continuity gate): see the 12-run table in the progress log.  Most flags are slip (32): frames where the HUMAN
  contact region itself slides (> 30 cm/s: 460 of 3945 frames on LAFAN dance; human support speed in 'contact'
  frames has a median of 7-8 cm/s on dance and soccer), latches released because holding them would pull the robot
  > 5 cm / 15 deg away from UMR, and robot contacts that slide.  The spec forbids robot slip during contact and the
  continuity gate forbids snapping, so these frames are flagged rather than fixed.  Heel-to-toe rolling inside one
  human contact segment is a support switch the spec does not allow -> the robot holds the touch-down point.
* Throughput of the sequential strict pass is the new bottleneck: production test (OmniContact soccer, G1, 32 clips,
  61,678 frames): 40 robot-frames/s end to end (strict 1483 s of 1541 s); profile (B=32 / 128 rows): 199 / 327 ms
  per time step, ~10 measurements + ~4 QP solves per step.  Next: time-parallel segments (each clip cut into
  overlapping segments run as extra rows; seam at the frame where both segments agree, checked by the continuity
  gate) and reuse of the accepted candidate's measurement as the next iteration's measurement (~35 % of measure
  calls).
* OmniContact box clips (Box_H_1): the human SMPL-X hands are 6-7 cm inside the box mesh (R_hand min-SDF p50
  -64 mm) and the trunk up to 10-12 cm inside; the UMR solution follows (robot hands 8-12 cm inside) and only
  27-40 % of frames pass (CPU factory: same 120 mm before its projection).  The box assets / poses must be checked
  before box jobs are produced.
* Self collision on the GPU is a conservative point-vs-SDF estimate (verified against `mj_geomDistance`: all
  penetrations > 0.5 mm detected on 3 robots, deepest-point depth <= MuJoCo's EPA depth, mesh hull outer
  approximation gap <= 0.83 mm); a handful of kept frames still exceed 0.5 mm in MuJoCo (H2 dance 7, A3 get-up 6 of
  thousands) -> the final per-frame flags should also be cross-checked on the CPU (`eval/judge.py`) for the
  self term, or the self margin raised.
* Object contact solidity is satisfied on kept frames, but soccer object penetration > 0.5 mm remains on 1-22 kept
  frames (MuJoCo ball hull vs our analytic sphere: the hull is a polyhedron up to ~1 mm outside the sphere).
* Throughput: the strict pass dominates; the UMR stage alone runs at 13-46k robot-frames/s (B 1024-4096).
* Only 3 SMPL-X clips were available locally (no gendered SMPL-X models; OmniContact other cases are BVH only).

## Production (resumable, detached)

**Status 10-08: STOPPED** (`production/v1/_STOP` present; first job's outputs from the frame-parallel engine are in
`production/v1/_invalid`).  `run_job` now uses the sequential strict pass on a clip-major layout (no root jumps), but
(a) its throughput is ~40 robot-frames/s on a 32-clip soccer job and (b) the first jobs in the order are box jobs
whose object assets do not match the human (see Remaining gaps).  Restart only after the job order / box assets and
the throughput plan are decided.

```
production/make_jobs.py   # builds /home/arenalabs/umr_factory_data/production/v1/_jobs/jobs.jsonl
production/start.sh       # starts the runner detached (setsid nohup, taskset -c 0-15, nice 10, OMP=2, UMRG_DUTY=0.6)
production/status.sh      # jobs done / total, robot-hours produced, kept fraction, robot-h per wall-h, last log lines
production/stop.sh        # graceful stop after the current job (touch _STOP);  stop.sh --now kills immediately
```

* Output: `/home/arenalabs/umr_factory_data/production/v1/<source>/<clip_id>/<robot>.npz` (atomic write) with
  `qpos` (floor-relative, 30 fps), `floor_offset_removed`, `fps`, sparse dense-point contacts (`contact_ptr/idx/partner`,
  `near_ptr/idx/partner/dist_q`), region labels (`ground_contact`, `ground_slip`, `ground_pen_mm`,
  `ground_clearance_mm`, `object_contact`, `object_pen_mm`, `object_clearance_mm`), `flags` (exclude frames with `flags & 255 != 0`; bit 256 = follow_human_slip is informational and kept), `ground_slip_speed` / `human_ground_slip_speed` (m/s, slip.py ruler, t-1 -> t), `ground_slip` (bit-32 regions), `ground_slip_follow` (bit-256 regions),
  human labels on SMPL-X slots with the same ruler (`human_ground_contact`, `human_ground_clearance_mm`,
  `human_object_contact`, `human_object_clearance_mm`), `object_pose` (floor-relative pos + quat wxyz), `object_scale`,
  `object_id`, `object_xml`, `human_scale`, `frame_ids_in_source`, `source_fps`, `qc_json`, `provenance` (engine
  hash, spec version, robot xml, surface asset, static prep, correspondence mode, input sha1, job id, time).
  Static dense-surface asset per robot: `production/v1/_assets/<robot>/surface.npz` (points are recomputed from qpos).
* Bookkeeping: `_jobs/<job>.done` (summary + per-clip QA), `_jobs/<job>.failed<k>` (traceback, <= 3 attempts),
  `_jobs/progress.jsonl`, `manifest.jsonl` (one row per clip x robot, appended at job end only), `_logs/runner.log`.
  A job without a `.done` marker is simply redone after a reboot.
* GPU etiquette: exclusive `flock` on `/home/arenalabs/umr_factory_data/logs/gpu_heavy.lock` for each job (released
  between jobs), duty cycle 0.6 (util ~50-65 %, mean power ~260-320 W), pause while `logs/throttle.log` says
  `net=down`.
* Job = (source, object asset, robot, <= 200 clips); order: OmniContact (Box_H_1, ScoccerBall_A_1, Box_K_1) chunk by
  chunk, robots G1 -> GR3 -> H2 -> A3 interleaved.  Sources are read at 30 fps (90 fps sources are strided by 3).
* Correspondence: UMR's beta-zero shared SMPL-X correspondence (`reuse_across_smplx_betas`, as UMR's HiPHI adapter):
  one static prep per robot (`problems/betazero__static90/<robot>.betazero.cpu.npz`), per clip only betas -> scale and
  template frames change (computed on the GPU).  HOI jobs use UMR's HSI/HOI solver defaults and the HOI body-segment
  costs.
* Not yet supported by the GPU engine (not in the job list): GRAIL (non-flat terrain), HiPHI (object OBJ meshes need
  UMR's object-MJCF preparation; chairs/tables are non-convex), gendered SMPL-X (neutral model used for all).

(Progress log moved to CHANGELOG.md.)
