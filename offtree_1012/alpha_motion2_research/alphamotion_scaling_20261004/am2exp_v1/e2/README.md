# WP-E2: hand <-> gripper opposition (data, descriptor, loss, judge, Stage-0)

All CPU. Code in this folder; outputs in `/var/tmp/alphamotion/am2exp/{data,eval}/e2`.
Run with `PY=/home/arenalabs/miniconda3/envs/ardy/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=""`,
from `CODE` (`cd am2exp_v1`). Use `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for pytest (the ROS
python path on this machine breaks pytest plugin autoload).

## Files

| file | what it does |
|---|---|
| `hand_data.py` | `download` (curl, pinned HF revision `a0fd465d`, sha256 verified, writes only under OUT/data/e2/d4rl, never the /home HF cache; HF_HUB_OFFLINE dropped only here) -> `export-raw` (rot env, h5py, PYROT subprocess) -> `convert` (ardy env, mujoco). Hammer/relocate 24 hand qpos decoded on the **pen** model with the pen adapter `MujocoMotionAdapter.decode_q` (forearm fixed); object pose moved into the pen-forearm canonical frame; seeded 10 dev / 15 test episode split; pen object side-file. `load_hand(ds)`, `load_pen()`. |
| `opposition.py` | descriptor. Adroit pads from MJCF (`C_*distal` surface point radially outside `Tch_*tip`), planned virtual-finger variants + the Stage-0 thumb-facing family; frozen choice in `opposition_frozen.json`. Franka `c_F = qL + qR - 0.2 cm`; Franka pad table from the inner capsules. `surface_vs_mjcf_check()` for the dense-asset pad tags. |
| `opposition_frozen.json` | FROZEN descriptor: `thface_070`, tau 0.2 cm, pads from MJCF, frozen 2026-10-04T19:11Z on DEV only, with the full candidate table. Do not edit; `freeze()` refuses to overwrite. |
| `xbody_loss.py` | `xbody_loss(model, bodies, rng, cfg)` for the trainer (also accepts `(model, bodies, cfg, rng)`). Contract in the module docstring: returns `{'total', 'terms', 'sampling'}`; total graph-connected also at `w_xb = 0`; RNG use depends only on batch size (G0 == G1 == G1_roll sampling); `roll` rolls targets by N rows (G1_roll uses 32); never conditions on Franka joints 8/9. Imported by `train/train_arm.py` as `am2exp_v1.e2.xbody_loss:xbody_loss`. |
| `analytic_retarget.py` | AN arm: `c_F = clip(c_A, 0, 7.8)`, symmetric fingers. |
| `closure_judge.py` | judge J2. Imports only numpy, mujoco and `_env` (a test enforces it). Grasp labels from dataset hand + object (MuJoCo collision capsules); hold/open/crush bands; BA; episode bootstrap; F->A `hand_closure_test` / `hand_fit_test`. |
| `run_e2_eval.py` | `stage0` (labels, descriptor table/freeze, floors, AN, A0/A_M direct mapping, OOD, F->A closure, report) and `arm --name G1 --checkpoint ...` for trained arms. |
| `bodies.py` | loads only the two external bodies exactly as `scaling_data.load_scaling_bodies` does (sealed code, read-only) and builds/loads checkpoints. |
| `_env.py`, `_sealed_math.py` | paths (loads `common/paths.py` by file so the sealed `common.py` is never shadowed) and the sealed rot6 helpers. |

## How to run

```
$PY -m e2.hand_data all                # download + export (PYROT) + convert; idempotent, verifies checksums
$PY -m e2.run_e2_eval stage0           # ~20-30 min CPU, writes OUT/eval/e2/stage0_report.md + stage0_summary.json
$PY -m e2.run_e2_eval arm --name G1 --checkpoint OUT/runs/G1/checkpoint_step1500.pt --model-class model.slide_fix:SFMegatron
```

## Tests

`tests/test_e2_hand_data.py`, `test_e2_opposition.py`, `test_e2_judge.py`, `test_e2_xbody_loss.py`
(44 tests, all pass, ~20 s with cached labels). They cover: identical hand joint order in pen/hammer/relocate XMLs; seeded split;
relocate palm-minus-ball vs obs (max 0.36 mm, limit 1 mm); pen side-file vs payload; hammer euler convention on
nail impacts; pad FK vs MuJoCo; Franka `c_F` vs `mj_geomDistance` at 5 q (< 0.1 mm); Franka pad centres vs
capsules (3 mm); surface_v2 pad cross-check code path (synthetic; real file not yet published); judge import
independence; band edges; exact floors (constant-open BA 0.5, constant-closed 0, oracle 1); hand re-posing vs
MuJoCo kinematics (1e-6); closure test on real hammer grasps; AN formula; xbody_loss contract, RNG invariance,
zero-weight graph connection, gradient reach, roll, no finger conditioning.

## Stage-0 results

Full tables: `/var/tmp/alphamotion/am2exp/eval/e2/stage0_report.md` and `stage0_summary.json` (run 3, 2026-10-05 00:15Z, 26 min CPU).

- G-0 gate PASS on dev: AN hammer-dev BA 0.765 (>= 0.75); constant-open BA exactly 0.5000 on dev and test.
- AN hammer TEST BA 0.539 [0.449, 0.644]: the analytic reference does not carry over from dev to test (fragile gate).
- Existing models A0_S@1000/3000 and A_M@1000/3000 map hand -> gripper to a nearly fixed, nearly open gripper
  (72-78 mm on every hammer frame): BA 0.500 = the constant-open floor.
- Gripper -> hand closure (balanced fit rate): A0_S@1000 0.559, A0_S@3000 0.332, A_M@1000 0.418, A_M@3000 0.345;
  random-hand floor 0.335, rest-hand floor 0.000.  No model clearly beats the floor.
- OOD (hammer test / pen eval same-body reconstruction error): A0_S 1.60x / 1.61x (pass), A_M 2.01x / 2.02x (just over 2x).
- Pad tags of surface_v2 cover the MJCF pads (nearest point 1.1-2.8 mm); patch centroids 1.5-3.3 mm (th 3.08, mf 3.30 miss 3 mm).

## Deviations from final_plan.md (and why)

1. **Descriptor family widened.** The planned thumb-pad to virtual-finger-pad descriptor reads ~29 mm on grasps
   of the 50 mm hammer handle (these are wrap grasps; fingertip pads do not face the thumb tip). A second family
   (thumb pad vs any facing phalanx/palm pad) was added and the whole table ranked by the pre-set dev rule; the
   chosen `thface_070` was frozen. The dev G-0 gate is therefore optimistic; the hammer **test** BA of AN is the
   honest number and is printed next to it.
2. **Closure test rewritten (rev 2).** Rev 1 put the cylinder at the thumb/finger Tch-site midpoint; on real hammer
   grasps it passed only ~2% at the true handle size, so it measured nothing. Rev 2 searches cylinder placements
   (anchored tangent on the thumb or ff/mf/rf tip capsule, 18 axes around palm lateral) and uses the same 5 mm
   tolerance as the grasp labels (real D4RL grasps sink 1-4 mm into the handle). Validated on hammer: grasps hold
   the true size ~0.97 and +30 mm ~0.06. Because a closed hand can always pinch something smaller, the F->A
   score is two-sided: **fit** = holds c_F but not c_F + 30 mm; `closure_BA_fit` = mean of fit rates on squeeze
   and other frames. The axis grid was set from hammer DEV grasps; the sanity numbers in the report use TEST.
3. **BA definition** is the plan's (`P(hold|g1)`, `P(open|g0)`), so constant-closed = 0, not 0.5; the "not-hold"
   variant (constant-open = constant-closed = 0.5) is in the JSON as `BA_nothold`.
4. **Relocate** is secondary as in the plan: with a 70 mm ball and 78 mm stroke, open is unreachable for a clipped
   gripper, so only hold-rate and crush-rate are reported.
5. `xbody_loss` accepts both argument orders (the interface said `(model, bodies, cfg, rng)`, the trainer calls
   `(model, bodies, rng, cfg)`).
6. Pads for the descriptor come from the MJCF (`pad_source: mjcf`); `OUT/assets/surface_v2/adroit_pen.npz` did not
   exist when the descriptor was frozen. When WP-ASSETS publishes it, `opposition.surface_vs_mjcf_check()` (and the
   test) checks the pad tags against the MJCF pads within 3 mm; the frozen descriptor stays MJCF.

## Open issues

- AN dev-to-test drop on hammer (see report): the analytic reference is weaker than the dev gate says.
- A_M OOD ratio is right at the 2x line (2.01-2.02x): an E2 failure of an M-size model cannot be blamed on the
  mapping alone.
- No Stage-0 videos from this package (render package owns them; per-frame `c_F` inputs are in
  `OUT/eval/e2/arms/<arm>/<set>.npz`).
