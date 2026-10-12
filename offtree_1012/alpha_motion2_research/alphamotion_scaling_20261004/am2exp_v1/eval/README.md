# eval/ — E1 evaluation (WP-JUDGE)

Frozen cases -> CPU inference -> judge J1 scores -> paired window-cluster statistics -> frozen decision rule.

| file | what |
|---|---|
| `cases.py` | Builds and freezes `OUT/eval/cases_e1_v1.json` (sha256 inside + `.sha256`), and the TRAIN-split calibration list `OUT/eval/cases_rootbias_v1.json`. Refuses to overwrite a frozen file with different content. |
| `run_e1_eval.py` | Model registry + inference: one `<case_id>.npz` per case (interfaces.md keys + `pred_joint_rotation`, `pred_contact_logits`, `pred_root_height_m`, source labels, the exact sensor inputs fed, meta json). `--shard k/n`, `--shuffle-source`, `--mode sensor_on|sensor_off`. |
| `score_e1.py` | J1 scores per case -> `<dir>/scores_j1.jsonl` (+ meta). Bounded process pool (<= 8). Never reads the contact head. |
| `stats.py` | Paired per-case differences, window-cluster sign test and sign-flip permutation test (exact <= 16 windows, else 20k MC), per-body sign counts, seed floor, guards, twin, `decide()` = final_plan.md decision rule as code, `replication_ok()`. |
| `stage0.py` | Stage-0 driver: heldout root bias, A0/B0 inference (on/off + shuffled), scoring, synthetic shifts, chance floor, `OUT/judge/stage0_report.md`. |

## Frozen case list (sha256 `ebb05d74a156c35813691b13afae2c4b2be8de4329d1f9c483535adb9232aad0`)

28 windows of 128 native frames (30 fps): 6 listed split-1 windows, 16 mined split-1 windows, 2 contact-poor guards
(pushAndStumble1_s5 [5384,5512), walk2_s3 centre = [3508,3636)), 4 train-split unseen-body windows (heldout targets only).
Mining rule (frozen in the file): human non-foot contact fraction >= 0.5 (any joint whose labeller limb group is not L_foot/R_foot,
`hard | gap<0` on trusted finite rows), all frames trusted, starts every 8 frames, greedy by (-fraction, start), no overlap with each
other or with listed/guard windows, <= 4 mined per clip, split-1 only, DFaust/PosePrior/non-flat (obstacles, vaults, slopes, handrails,
stairs, parkour, JumpOverObstacle) excluded. Mined windows were spot-checked on labels: kneeling/crawling (walk2_s3 @5344/@5928 knees+wrist,
BMLmovi S31_F_20 knees+wrists, HDM05 tr_03-03 lying, root 0.04 m).
Sources human_smpl (primary), unitree_g1_29dof, fourier_n1; targets all 15 robots (not the source itself), heldout targets only for the
unseen-body windows: **1080 cases, 346 primary** (human source on listed/mined/unseen-body windows). Cluster unit for all tests = window.
Root-bias calibration list (sha `c87c9269...`): centre windows of the 11 split-0 LAFAN `walk*_subject*` clips x 4 heldout robots.

## Runner details

* Registry: A0/B0 (or any checkpoint whose config has group A/B and no `am2exp` block) -> original classes through the sealed
  `run_scaling_eval.model_factory`; am2exp arms -> `cfg['model_kind']` ('module:attr') + `registry_entry.extra.model_kwargs` + the
  geometry hook (dense surfaces) recorded by `train/train_arm.py`; `--model-class NAME` -> `am2exp_v1.model.registry.MODEL_CLASSES`.
* Windows are read with the sealed `run_locomotion_eval.load_window(..., with_contact=True)`, which verifies pose9 against the sealed
  loader rows (< 2e-5) and the recomputed `contact_sdist` (cache-header thr/D, fp16) bit-exactly against the payload rows.
* sensor_on: `source_contact` = that contact_sdist (NaN where invalid); arms whose `forward_kinematic` accepts `source_root_height_m`
  (arm I) get the source root height where trusted & finite & any finite contact, NaN elsewhere. sensor_off: None / all NaN.
* `--shuffle-source`: the source window is replaced by the next window of a different clip in the same family (motion windows / body
  windows), GT unchanged.
* CPU speed: ~0.4 s per 128-frame case with 2-4 threads; RSS ~2 GB per process.

## Run

```
cd ROOT; export PYTHONPATH=ROOT PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
$PY -m am2exp_v1.eval.cases                                # freeze (idempotent; refuses changed content)
$PY -m am2exp_v1.eval.run_e1_eval --arm S --checkpoint OUT/runs/S/checkpoint_step3000.pt --mode sensor_on --out OUT/eval/e1/S_s3000/sensor_on --shard 0/4
$PY -m am2exp_v1.eval.score_e1 --dir OUT/eval/e1/S_s3000/sensor_on --jobs 6
python -c "from am2exp_v1.eval import stats, score_e1 as s; print(stats.decide(s.load_scores(ARM_DIR), s.load_scores(PARENT_DIR), sf_records=..., sf_s1_records=..., twin_records=...))"
$PY -m am2exp_v1.eval.stage0 --jobs 6                      # Stage-0 (A0/B0), ~30 min wall on a busy 32-core box
```

## Stage-0 results (2026-10-04, full tables in OUT/judge/stage0_report.md)

* Heldout root bias c_b (A0, TRAIN windows, sensor_on; cm): GR3 -11.19, Kuavo +1.57, H2 -11.44, R1 +0.04 (IQR across 11 windows ~ +-2 cm).
* Headroom (human source, primary, sensor_on, heldout targets): A0 CSE_GT 0.774, FloatRate 0.498, PenP95 4.85 cm (GT 12.88 cm), mm 375;
  B0 CSE_GT 0.848, FloatRate 0.622. Numeric headroom gate passes on CSE and FloatRate (gap >= 0.10). A0 puts a non-foot part on the
  ground on ~9 % of frames vs ~77 % in the GT.
* Shuffled-source floor: CSE_GT(shuffled) - CSE_GT(A0) = -0.006 -> the ">= 0.15 worse than A0" requirement FAILS because A0 itself is
  at chance on the contact pattern (pose error does get 65 mm worse). See the report for the GT-of-another-window chance level.
* sensor_on vs off: tiny (A0 dCSE -0.001; B0 -0.009, p < 1e-4).

## Tests

`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' $PY -m pytest -q -p no:cacheprovider am2exp_v1/tests/test_eval_e1.py`
(mining rule, frozen list rebuild = identical, shuffle map, exact permutation/sign tests, decision rule incl. floor/guard/twin/replication,
real A0 inference on 2 cases in both modes, sensor == sealed payload, scorer incl. heldout-bias handling).

## Deviations from the plan text

* AM2 rotations are own-joint GLOBAL in the MuJoCo frame convention (not identity at rest for tilted MJCF frames); J1 uses them as they are.
* R1 delta_col corrected (-4.38 instead of -0.50 cm), see judge/README.md.
* Heldout root bias is defined against the GT on TRAIN windows (median of zmin(pred raw) - zmin(GT) on GT stance frames), not as an
  absolute 0 cm target, so seen and heldout bodies share the same reference (GT on the J1 ruler).
* Synthetic-shift detection rules were revised after the v0 rules failed for data reasons (GT penetrates on stance/ground frames); both
  verdicts are reported.
* Videos ([human | GT | A0]) are not produced here (render package).
