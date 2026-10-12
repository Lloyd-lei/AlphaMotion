# UMR data factory (code)

Human motion (SMPL-X / HOI) -> UMR retarget -> **strict no-penetration post-pass** -> point-cloud contact/slip labels,
for the 21 core humanoids of `/var/tmp/alphamotion/umr_factory/robots/robots.json` (tier `A_core20`).
Training use only. Upstream UMR (`/var/tmp/alphamotion/umr_factory/umr/UMR`, commit d1cdaed) is never modified.
Python: `/var/tmp/alphamotion/umr_factory/umr/env/bin/python` (3.12, mujoco 3.3.7, clarabel).

Data root: `/home/arenalabs/umr_factory_data/{robots,corr,runs,out,logs}`.

## Steps
| step | command | output |
|---|---|---|
| 1 stage robots | `scripts/stage_robots.py` | `robots/<r>/<r>.xml` (standalone MJCF, meshes hard-linked/copied into `robots/<r>/mesh/`), `umr_robot.json`, `staging.json` (T-pose metrics, exclusions) |
| 1b T-pose sheet | `scripts/render_tpose_sheet.py` | `robots/tpose_contact_sheet.png` |
| 1c dense surface | `scripts/build_surfaces.py` | `robots/<r>/surface.npz` (4096 exterior pts, convex-hull vertices, regions) |
| 1d correspondence | `scripts/build_correspondence.py` | `corr/<r>/smplx/{correspondence_dataset.npz,train/}` |
| 2-4 run | `scripts/run_clip.py --kind smplx|hoi --data D --seq K --robots a,b` | `runs/<clip>/<r>.*` (raw UMR), `out/<clip>/<r>.npz` + `.qa.json`, `out/manifest.jsonl` |
| QA | `scripts/qa_summary.py` | `out/manifest.json`, `out/qa_summary.json` |
| visual | `scripts/render_check.py <clip> <r> [--mp4 a b]` | `out/<clip>/<r>_grid.png` / mp4 (human / UMR raw / post-pass) |
| tests | `tests/test_postpass.py <r>` | `out/tests/test_postpass_<r>.json` |

## Robot staging
* MJCF flattened with `MjSpec`, keyframes removed, assets localised (UMR's correspondence builder rewrites any
  absolute path containing `/assets/` and cannot read paths with quotes, and UMR writes `*.floating_mjcf.xml` / temp
  object MJCFs next to the robot XML -> the robot must live in a directory we own).
* `unitree_g1_23dof` = GMR 29-DoF tree with waist roll/pitch and wrist pitch/yaw joints **removed** (nq 30).
* T-pose: draft T-pose kept if arms are level and lateral (upper arm and forearm <= 15 deg from lateral, hand tip within
  7 cm of shoulder height and fore/aft); otherwise least-squares on the arm chain (segments lateral + elbow axis vertical
  = palms-down proxy). T-poses shipped by UMR (G1, H2, K1) are never changed.
* Collision exclusions (`<contact><exclude>`): only body pairs whose collision geoms overlap at qpos0 or the T-pose AND
  are <= 2 tree hops apart or overlap in >= 95 % of 300 random in-limit poses; plus a documented manual list
  (`robots.MANUAL_EXCLUSIONS`: Agibot A2 ankle linkage rods vs toe plate). Overlaps between non-adjacent links
  (Talos gripper vs thigh at qpos0) are NOT excluded and count as penetration. Every pair + reason is in `staging.json`.

## UMR run (`umrf/pipeline.run_umr`)
Ground hard constraint ON, robot-object hard constraint ON (HOI), robot self-penetration hard constraint ON with slack
through the scout wrapper `tools/umr_run.py` (`UMR_QP_FALLBACK=1`: on a Clarabel failure retry with soft rows, then
dq = 0; counts in `runs/<clip>/<r>.qpstats.json`). The wrapper also captures UMR's 4096 human source slots.

## Strict post-pass (`umrf/postpass.py`, `umrf/geometry.py`)
Gauges (metres): ground = min over **convex-hull vertices of every visual geom** (exact for a plane) and the dense
points; object = analytic SDF (sphere/box detected, else trimesh) at dense points + hull vertices; self = MuJoCo
collision geoms, `mj_geomDistance`, staged exclusions applied. Normal sign of each self pair is checked by a
finite-difference probe (MuJoCo's `fromto` orientation was wrong for penetrating box-cylinder pairs on GR3).
Per frame: if the UMR pose already satisfies everything it is kept bit-exact; else Clarabel QP iterations (tracking
to UMR pose, temporal term, xy anchors on support points, joint limits, adaptive trust region) until
penetration <= 0.5 mm and every required contact region's lowest point is in [0, 1 mm]; never returns a pose worse
than the best seen; remaining failures are FLAGGED (`flags` bitmask: 1 ground, 2 object, 4 self pen; 8/16 required
ground/object contact not snapped). Corrections are then Gaussian-blended (sigma 2 frames) into neighbours, every
touched frame re-verified. The output is re-verified independently of the solver (`flags` in the npz).

Required contacts = robot hysteresis label on the UMR output OR (human label AND robot region < 3 cm (ground) /
5 cm (object) AND the robot region not sliding), with touch-down/lift-off frames still in the 1-2 cm hysteresis band
trimmed from the ends of each run, min 3 frames.
Slip latch: in a required ground stance the support points get an xy anchor that reproduces UMR's own tangential
velocity (`LATCH_PIN = 0`), so the projection adds no slip. Full pinning (removing UMR slip) was tried and raised
slip distance on dance1 by 30-60 % through release spikes; it is opt-in only.

## Labels (`umrf/labels.py`)
Region contact: hysteresis enter < 1 cm / leave > 2 cm on the region clearance, runs < 3 frames dropped. Output label
`ground_contact` / `object_contact` = the enforced, verified set (every frame = 1 is touching within [0, 1 mm]).
Slip = median tangential material speed of the support points (in contact t-1..t+1, within 3 mm of the region
minimum), relative to the object for object contact; flag = > 3 cm/s sustained >= 3 frames.
Human: same rules on UMR's 4096 source slots (retarget frame, scaled by `human_scale`), regions from SMPL-X part
segmentation through the correspondence face ids. 13 regions: trunk, L/R thigh, shin, foot, upper_arm, forearm, hand.

Floor: per session (here per clip) from the human source: median lowest foot point over quasi-static stance frames.
Output is shifted so the floor is z = 0 (`floor_offset_removed`).

## Output npz keys
`qpos` (floor at z=0), `qpos_umr` (raw UMR, same shift), `root_pos`, `root_quat_wxyz`, `fps`, `floor_height` (=0),
`floor_offset_removed`, `joint_names`, `regions`, `robot_xml`, `ground_contact[T,13]`, `ground_clearance[T,13]`,
`ground_slip`, `ground_slip_speed`, `ground_contact_required`, `ground_point_contact` (packbits [T,512]),
`pen_ground_m[T]`, `pen_self_m[T]`, `flags`, `flags_initial`, `human_ground_contact`, `human_ground_slip`,
`human_ground_clearance`, `human_scale`; HOI adds `object_contact`, `object_clearance`, `object_slip`,
`object_slip_speed`, `object_contact_required`, `object_point_contact`, `pen_object_m`, `object_pos`, `object_rot`,
`object_name`, `object_xml`, `human_object_contact`, `human_object_slip`; `provenance` (json).

## Known limits
* Robot dense points are sampled (4096); object penetration between samples can exceed the point value by
  ~s^2/(8R) (about 0.3 mm for an 8 cm ball). Ground is exact; self uses collision geoms (pnd_adam_lite hands have
  none -> hand self-contact unmeasured there; GMR G1 head has no mesh).
* HOI correspondence caches are written by UMR to `UMR/data` + `UMR/output` (/var/tmp, ~96 MB each); the HOI
  pipeline ignores our `corr/` paths. Plain SMPL-X caches go to `corr/`.
* surface_v2 assets were not reused (bound to the xwbc GMR MJCFs and missing for 3 robots); our surfaces are bound to
  the staged MJCF geoms for all 21.
