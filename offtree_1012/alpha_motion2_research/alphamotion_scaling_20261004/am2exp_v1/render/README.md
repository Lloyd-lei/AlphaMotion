# render/ — WP-RENDER (videos are lloyd's main evaluation)

Three renderers. Each writes `triptych.mp4` (1920x1080 H.264, real timestamps, VFR; made with `rollout_tools/encode.encode`, which checks the PTS), plus `grid12.png`, `poster.jpg` and `manifest.json`.
Output folders are written atomically through a temp dir and a rename. An existing folder is never overwritten unless `--overwrite` is passed.

| file | what |
|---|---|
| `triptych_e1.py` | E1: [human source \| control arm \| test arm], or stage-0 [human \| GT \| arm], or the GT\|GT\|GT identity check |
| `triptych_e2.py` | E2: [Adroit hand + object, pads, opposition axis \| control gripper \| test gripper], close-up |
| `grid12.py` | 12-frame sheet; frames `round(linspace(0, T-1, 12))` decoded exactly from the mp4 (`ffmpeg select=eq(n,i)`), so the first and last frames are always in it |
| `robot_model.py` | vendor MJCF visual and collision geoms bound rigidly to AM2 joint frames (rebuilt cache skeleton, verified on qpos0 + 7 random q) |
| `scene3d.py` | MuJoCo EGL panel: floor, sky, shadow light, geoms re-posed from mjv_updateScene (no physics), overlay primitives |
| `contact.py` | per-region ground clearance → colours and traces |
| `case_io.py` | per-case npz reader (interfaces.md "Eval output") + GT windows from `data/trajectory_reader.py` |
| `layout.py`, `regions.py`, `_env.py` | text/traces, region fallback, paths and read-only sealed imports |

## Run (PY = ardy env; always with a lifetime)
```
cd CODE ; export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" ; unset PYTHONPATH
timeout 1500 $PY -m render.triptych_e1 --demo --out OUT/videos/demo/<name>                    # GT demo
timeout 1500 $PY -m render.triptych_e1 --control CTRL.npz --test TEST.npz --labels SF,S --out DIR
timeout 1500 $PY -m render.triptych_e1 --stage0 CASE.npz --out DIR                             # [human | GT | arm]
timeout 1500 $PY -m render.triptych_e1 --gt-self CASE.npz --out DIR                            # [human | GT | GT]
timeout 1500 $PY -m render.triptych_e2 --dataset hammer --episode 1 --control G0 --test G1 --out DIR
$PY -m render.grid12 VIDEO.mp4 --out grid.png
```
Every CLI also sets SIGALRM to `--max-seconds` (default 1800) plus 120 s. A 128-frame E1 triptych renders in about 21 s. A 200-frame E2 triptych takes about 19 s. RAM stays under 3 GB.
`unset PYTHONPATH` is needed for pytest only: the ROS humble pytest plugins on PYTHONPATH break collection.

## E1 conventions (all checked in tests)
- **Robots**: vendor MJCF VISUAL meshes, re-posed rigidly from the given own-joint global rotations and root-relative positions. Every robot's rebuilt skeleton equals the frozen static spec (offset error 0.000 cm). FK over 7 random q gives < 0.05 cm for all bodies except H2 (see open issues).
- **Root height** is the judge J1 convention, `_env.root_offset_cm` = `judge.ground_judge.corrected_root`. For GT and for a prediction on a training robot it is `100*h - DELTA_COL_CM[body]`. For a prediction on a held-out robot it is the judge's measured heldout constant (`OUT/judge/heldout_root_bias_v1.json`). While that file does not exist, the GT constant is used and the label says "暂用参考修正". A case npz may override this with `pred_root_offset_cm` / `gt_root_offset_cm`. The rule in use is printed under every panel. If no root height is given, the lowest point is snapped to the floor and the label says so.
- **Contact colours and the min-height trace**: green means the region's lowest point is < 2.5 cm above the floor; red means it is > 1 cm below. Both come from the region minima of the COLLISION geoms, computed by `judge/ground_judge.judge_case` when it can be imported. Otherwise a local lowest-point calculation on the same geoms is used. The two agree to < 2e-5 cm on G1 and GR3 (test). The manifest records which one was used. The dense visual surface is never used for robot colours.
- **Human**: stick figure. When WP-ASSETS `OUT/assets/surface_v2/human_smpl.npz` and `CODE/assets/surface_v2.py` exist, its 4096 posed points are drawn and give the region heights. Otherwise the colours come from the human's own contact labels, which use the joint-centre ruler from the frozen cache. The human asset is not there yet, so the demo uses the labels (manifest `contact_source`).
- **GT ghost**: in each arm panel, thin grey capsules show the GT skeleton at the GT root height. The trace shows "本栏" (this panel) and "参考" (GT) lowest point.
- **Follow camera**: the model predicts no horizontal root. So every panel is shifted horizontally along the SOURCE root trajectory (displacement from the window's first frame), scaled by `reach(panel body)/reach(source)`. Each camera follows its own panel's root, averaged over 0.5 s. The checker floor stays fixed in the world, so feet sliding against it can be seen. The camera azimuth is a 3/4 front view of the source's mean facing direction. `--no-follow` turns this off.
- **Titles** are in plain Chinese. Held-out robots are tagged "训练没见过的机器人".

## E2 conventions
- **Hand**: Adroit `V_*` visual meshes posed by MuJoCo from the dataset hand qpos. The forearm is fixed, so the camera stays fixed in the forearm (EE) frame and aims at the moving opposition centre (smoothed over ±0.25 s). `hand_<ds>.npz` positions equal the MuJoCo body positions to 2e-6 cm (test and manifest).
- **Object and descriptor**: the object uses the dataset geometry: hammer handle capsule d = 50 mm, relocate ball d = 70 mm. Pads come from `e2.opposition.adroit_descriptor` (frozen variant, currently `thface_070`). The thumb pad is orange; the virtual-finger pads are purple, sized by their softmax weight. The orange line is the opposition axis.
- **Gripper**: Franka hand and finger visual meshes. The arm is parked at a seeded-search pose whose fingers point straight down, so the free camera shows the gripper upright. The camera looks straight across the gap, along the object axis, tipped 12° about the opening axis. The slides are set to `qL = qR = (c_F + 0.2 cm)/2`, clipped to the 0..4 cm stroke, and the label says when clipping happens. The finger colour is the J2 state from c_F alone (`closure_judge.gripper_state`, equality tested): green = hold, `d-3 <= c_F <= d+8` mm; grey = open; red = crush. The object of diameter d sits centred between the pads, and an orange line across the gap shows d.
- **Traces**: pad clearance in mm. Dashed orange = d; green band = J2 hold band; grey = the hand's own c_A. The bottom strip is the J2 grasp label: green = hand holds, grey = hand free, nothing = not scored.
- **Arm inputs**: `OUT/eval/e2/arms/<name>/<ds>.npz` (`c_F_mm` row-aligned with `hand_<ds>.npz`). Alignment is verified through the `episode`/`frame_index` columns. **Stride 2** is the default: the data is 100 Hz, so the video has 50 frames/s and still plays in real time.

## Tests — `tests/test_render_core.py`, `test_render_e1.py`, `test_render_e2.py`
`41 passed in 21 s` (EGL; the GPU is used for rendering only). Coverage:
- **Grid**: indices span [0, T-1] for T in {1, 5, 12, 13, 128, 451}. An exact-frame decode test encodes 40 distinct grey frames through the real encoder.
- **Contact and colours**: contact state thresholds; local lowest point == judge (G1, GR3) and a +5 cm root shift gives exactly +5 cm. The render root offset == `judge.corrected_root` for all 15 robots. Robot colours never touch the dense surface.
- **Robot binding**: for all 15 robots, the rebuilt skeleton == static spec.
- **Case npz**: round trip; source placement is re-read from the cache by meta.
- **Identity renders**: **GT|GT|GT self-render**: the two arm panels have identical geometry, pose and state arrays, and identical pixels up to GPU rasterisation noise. One EGL context rendering the same scene twice differs by 1 grey level on ~0.05% of pixels, so the test allows ≤2 levels on ≤0.2% of pixels. The GT window built from the cache == the case-npz path to < 1e-4 cm.
- **Synthetic data**: a synthetic prediction (GT ± 15° per joint, FK positions, root −3 cm) differs from GT and draws the ghost.
- **Full renders**: a full 24-frame E1 CLI render (1920x1080, 24 frames, duration 0.8 s, PTS error < 1e-5 s) that refuses to overwrite. An E2 render of 12 frames at stride 10 lasts 1.2 s.
- **E2 geometry**: E2 hand posing is exact. The rendered gripper gap follows the slide value with a spread of 3e-13 mm. The fingertip gap at c_F = 0 is 3.0 mm. The approach axis points down (a_z < -0.95). Identical control/test give identical panels. The arm-file alignment check rejects a shuffled episode column.

## Demo outputs (now in OUT/videos/demo/)
1. `fallAndGetUp1_subject4_4136-4264__human_gtG1_gtGR3/`: [human source | GMR GT G1 | GMR GT GR3], 128 frames, 4.267 s, 30 fps, judge colours.
   - Roots: G1 offset 0.04 cm, GR3 offset 9.99 cm (report-32).
   - Median lowest point: G1 -6.4 cm, GR3 -4.5 cm.
   - The GMR G1 reference itself sinks into the floor on this fall-and-get-up window: pelvis red 70% of frames, feet 79–88%. It is visible in the video and is a property of the GT, not of the renderer.
2. `e2_hammer_test_ep1__hand_AN_A0S3000/`: [hand + hammer handle | analytic map AN | A0_S@3000], hammer test episode 1, 200 frames, 4.0 s.
   - AN holds 4% of frames and is open 96%.
   - A0 is open 100%: it stays at about 74 mm and never closes.

## Deviations / facts found
- **Human ruler**: the joint-centre label marks forearm and hand regions as "penetrating" (> 1 cm below the floor) when the hands are on the ground. Joint centres sit below a contact surface, so this is the documented undercount and overcount of that ruler. It will change when the dense human asset lands.
- **H2**: the cache skeleton (anonymous body00..31) differs from the vendor H2 MJCF by up to 5.8 cm in rest offsets. The judge's explicit node map is used and the same residual is seen by J1. H2 meshes are therefore approximate (manifest `robot_audit`).
- **Franka c_F convention**: `c_F = qL + qR - 0.2 cm`. In the rendered MJCF geometry, at c_F = 0 the fingertip capsules are 3.0 mm apart, and the mean inner-pad gap is c_F + 4.93 mm. This is reported as `checks` in every E2 manifest. The plan's "within 3 mm" pad check belongs to WP-E2. The renderer draws q exactly.
- **Image identity**: the exact pixel-identity requirement for GT self-render is impossible with MSAA and shadow maps, as explained under Tests. Identity is asserted exactly on the geometry, colour-state and camera arrays, and up to 1–2 grey levels on pixels.
- **Fixed during this session**:
  - The judge was imported by file path, which failed on its absolute `am2exp_v1.judge` imports, so the demo silently used the (equal) local calculation. It now imports through `_env.judge_module`.
  - Model panels now use the prediction root convention (`root_offset_cm(body,'pred')`) instead of the GT one.

## Open issues
- No real E1 case npz exists yet (`eval/run_e1_eval.py` from WP-JUDGE is missing), so E1 control/test is tested on synthetic and GT cases only.
- The dense human asset is not built yet. The dense drawing path is unit-tested on synthetic points only.
- The heldout root constants file does not exist yet. Held-out predictions use the GT constant, and the panel label flags this.

## Integration changes (WP-INTEGRATION, 2026-10-04)
- `contact.human_region_heights` (dense path): L_hand/R_hand heights come from the wrist joint centre minus 3 cm,
  not from the dense hand ellipsoid, which sinks ~20 cm when the palm is planted (WP-ASSETS README).
- Known: the dense human pelvis/thigh surface sits up to ~12 cm under the floor on sitting/lying frames with the
  label root height, so human panels show red pelvis/legs on ground motions (see integration/README.md).
