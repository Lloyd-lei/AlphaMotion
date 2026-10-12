# judge/ — independent E1 judge J1 (WP-JUDGE)

J1 measures ground contact, penetration, floating and self-penetration of a posed humanoid on the **vendor MJCF
collision geometry**. It never reads the dense visual surface (CODE/assets), the sole table, any training loss or the
model's contact head (`test_judge_independence` checks the imports).

## Files

| file | what |
|---|---|
| `pose_mjcf.py` | Build (`--build`) and load the link->AM2-node binding of every collision geom; re-pose geoms from AM2 global rotations/positions; lowest point per geom (mesh hull vertices / analytic box, sphere, capsule, cylinder, ellipsoid); self-penetration through `mujoco.mj_geomDistance` on non-adjacent pairs. |
| `regions.py` | The 15 regions and the MJCF-body -> region rules (+ per-robot overrides); human SMPL-22 joint -> region map. |
| `ground_judge.py` | `judge_case(body, rot, pos, root_height_m)` -> `region_min_height_cm [T,15]`, `zmin_cm`, `selfpen_cm`; root conventions (`corrected_root`); `case_metrics` (PenRate, PenP95, FloatRate, SelfPen, CSE_GT, CSE_snap, CSE_human, nonfoot variants, mm guard). |
| `ruler_gate.py` | Stage-0 ruler gate against report 32 (`--all`, bounded parallel processes). |

Binding outputs: `OUT/judge/bindings_v1/<body>.npz/.json` (+ `build_report.json`). Ruler: `OUT/judge/ruler_v1/`.
Stage-0 report (written by `eval/stage0.py`): `OUT/judge/stage0_report.md`, heldout root bias `OUT/judge/heldout_root_bias_v1.json`.

## How the binding works (verified, not assumed)

* AM2 convention for the XWBC robots, checked against MuJoCo at qpos0 + 15 random q per robot: `R_am2[j] = AX R_world(frame_body[j]) AX^T`
  (own-joint GLOBAL; it is **not** identity at rest for MJCF bodies with tilted frames, unlike the interfaces.md wording),
  `P_am2[j] = 100 AX (x(keep_body[j]) - x_root)`, `AX = [[0,1,0],[0,0,1],[1,0,0]]` (Z-up -> Y-up). keep/frame come from the cache
  builder `exp_topo/prep_gmr_cache.build_merge` (imported read-only at build time only). AM2 static-spec FK reproduces vendor FK to < 2e-6 cm
  (GR3: 0.05 cm).
* A geom is bound to the node whose frame carries it rigidly over all states (rotation < 1e-6, position < 1e-4 cm). Geoms on bodies merged
  into a co-located hinge chain (GR3 right foot, Kuavo leg_*1 / zarm_*1-2, H1_2 2 geoms) are re-posed by solving the chain's hinge angles from
  the parent/child node rotations (`chain_pose`, verify residual 0.0 cm). **0 approximate bindings on all 15 robots.**
* unitree_h2: the AM2 tree (body00..31) comes from the retarget data, not the vendor MJCF. Nodes are mapped to the vendor bodies by topology
  (`H2_NODE_MAP`); vendor geoms ride on the mapped node frames. Data vs vendor bone offsets differ horizontally by up to 5.3 cm (wrists),
  2.7 cm (knee/ankle); heights at rest agree to 1e-4 cm (ankle y -98.452 both), so ground heights are not affected at rest.
* unitree_r1: 38 AM2 joints, 34 collision geoms, all rigid.
* Self-penetration pairs: anchors at hop >= 2, not interpenetrating at qpos0 (> -0.5 cm), and penetrating (< -1 cm) in at most half of 32
  random states (excluded pairs listed in each manifest).
* Missing regions (no collision geom there in the vendor MJCF): e.g. T1 pelvis/forearms, Adam head/forearms/hands/thighs, tienkung trunk/head/hands,
  G1/N1/H1 head. CSE uses only regions present on the body (`regions_present`).

## Root conventions (frozen)

* GT: label root height - `delta_col_cm` (report-32 collision bias) for every body.
* Prediction, training bodies: predicted root (`raw_hpos[:,0,0]`) - `delta_col_cm`.
* Prediction, heldout bodies: predicted root - `c_b`, `c_b` measured once from A0 on TRAIN-split windows (eval/stage0.py) and applied to every arm.
* `raw_*` (no correction) and per-frame ground-snapped (`CSE_snap*`) variants are always reported next to it.
* `delta_col` is report 32's table except **unitree_r1: -4.38 cm instead of -0.50 cm** (report 32 measured R1's drop inside
  ankle_constraint_A/B and missed the real sole, `left/right_ankle_roll_link` merged into the ankle_pitch node; J1's qpos0 drop equals report 32's
  true drop to < 0.02 cm on the 14 other bodies). `delta_col_table(verify=True)` refuses to run if report 32's results.json changes.

## Metrics (per frame -> per case)

region contact = region lowest point < 2.5 cm; PenRate = zmin < -1 cm; PenP95 = p95 of max(0, -zmin);
FloatRate = pred zmin > 3 cm on frames where the GT has any region in contact; SelfPen = deepest allowed-pair penetration > 1 cm;
CSE_GT = 1 - F1(pred region contact vs GT region contact on the same ruler); CSE_snap = same after per-frame ground snap;
CSE_human = vs human joint-centre labels (documented under-count: joint centres sit above the skin).

## Stage-0 acceptance (see OUT/judge/stage0_report.md for the full tables)

* Ruler gate (strict, pre-registered: J1 stance lowest point vs report-32 reference within 1 cm on LAFAN **and** AMASS for >= 13/15):
  **FAIL, 6/15** (LAFAN only 11/15). J1 is 0.2-2.1 cm below the reference on every body. Cause measured: the reference is one rigid
  labeller sole point; J1 is the lowest corner of the collision box/mesh, which is lower on tilted stance frames (median tilt 5-10 deg).
  Flat-foot stance frames (tilt < 3 deg): **13/15 within 1 cm** (fails: booster_t1_29dof AMASS -1.32, tienkung AMASS -1.00).
* Synthetic +-5 cm root shifts on stance frames: detected on 15/15 (v1 rule: PenRate(-5) >= 0.9, FloatRate(+5) >= 0.5, both moved the
  right way). The v0 rule (both rates move by >= 0.5) fails on booster_t1_29dof/H2 because unshifted GT already penetrates on 35-65 % of
  AMASS stance frames.

## Tests

`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider am2exp_v1/tests/test_judge_j1.py`
(re-pose vs MuJoCo at random q for 6 robots, lowest point vs brute-force mesh vertices, self-pen vs `mj_geomDistance`, root shift/identity
metrics, root conventions, human region map, independence, ruler consistency). 16 passed.

## Rebuild

`PYTHONPATH=ROOT $PY -m am2exp_v1.judge.pose_mjcf --build` (all 15, ~2 min) then `--regions` to print the geom/region table;
`PYTHONPATH=ROOT $PY -m am2exp_v1.judge.ruler_gate --all --jobs 8` (~3 min).
