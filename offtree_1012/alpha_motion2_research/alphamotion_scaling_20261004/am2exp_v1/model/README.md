# WP-MODEL — `am2exp_v1.model`

Model classes and losses for the AM2 surface experiments (final_plan.md / interfaces.md):
slide fix (SF), dense-surface group tokens (S, S_wrong), relational source inputs (I, I_roll),
target relational objective (O, O_roll).

**Import as `am2exp_v1.model`** (ROOT on `sys.path`). A bare `import model` must stay the sealed
`base/legacy/model.py`; `_sealed.py` raises if this package is imported as bare `model`.

## Files
| file | content |
|---|---|
| `_sealed.py` | read-only sealed imports (pilot_model, model_dual, kinematics, surface.surface_model.AttachmentCrossAttention) |
| `slide_fix.py` | `SlideFixHead`, `SlideFixMixin`, `SFMegatron`, `SFSurfaceMegatron`, `slide_fix_loss`, `make_slide_fix_objective`, `clamp_slide_native` |
| `dense_surface.py` | `DenseSurfaceMegatron` (arm S/O/S_wrong), `arm_parameter_counts` |
| `relational.py` | `RelationalInputMegatron` (arm I/I_roll), `relational_features` |
| `losses_relational.py` | `relational_objective` (arm O/O_roll), `make_objective` (composition helper) |
| `geometry.py` | `attach_dense_geometry` / `detach_dense_geometry` (wraps the `Body.geom` closure that `scaling_data.attach_geometry` installed, never edits it), `build_dense_static`, `default_derangement` |
| `ground.py` | DFaust / PosePrior-trust0 exclusion, `row_ground_ok`, `source_root_height_sensor` (I-arm sensor) |
| `surface_api.py` | adapter to `am2exp_v1.assets.surface_v2` (WP-ASSETS) + rigid fallback + synthetic schema assets for tests |
| `registry.py` | arm -> class map, seeded `build_model` mirroring `train_scaling.build_model` |
| `bench.py` | CPU time / saved-activation micro-benchmark -> `OUT/model/bench_cpu_B<B>.json` |

## How it works
* **SF**: `slide_head` replaced after `super().__init__()` (same name): `logit = Linear(2d,1)([LN(qp);LN(qr)])`,
  w ~ N(0,.02), bias = logit((0.968+m)/(1+2m)), m=.05; `q01 = -m+(1+2m)σ(logit)`; native unclamped
  (`clamp_slide_native` for the clamped report). Loss `SmoothL1_β=.02(q01-clip(t,-m,1+m)) + 1e-3·mean(logit²)`;
  logits exposed in `outputs['aux']['slide_logit']`.
* **S**: per body, G=256 group tokens (alt `grouping='g128'` = 128×32): PointNet-lite over K members
  `[(p-c)/(0.05·reach), normal, region one-hot(18)]` (MLP 24→d→d, max-pool) + MLP(centre/reach) + region emb
  + plain descriptor[group_anchor] + type emb. Encoder: per sealed EncBlock, joints = the unchanged SF
  self-attention + `Z_l(cross-attention joints→groups, hop(j,anchor)+1)` with `Z_l` zero-initialised;
  groups attend over [joints; groups] with the B hop construction (surface_model.py:107-116: +1 joint↔group,
  +2 group↔group, diag 0). FSQ pools read joint tokens only. Decoder descriptor = descriptor +
  `AttachmentCrossAttention`(sealed class, `out` zero-initialised) over target group tokens. Bodies with no
  dense surface run the SF code path.
* **I**: on calls that pass `source_root_height_m`/`roll_relations` (forward / forward_kinematic), per group
  28 features from the posed SOURCE dense surface (no_grad, data-derived): ground `[slog(min h), mean h/50,
  frac(h<3cm), mean n_y of near-ground members]` with `h = y + 100·rh − beta_vis_cm`; self-contact nearest
  allowed visible point in `sub1024` `[ln(1+d/2)/ln26, dir in anchor frame (3), partner region one-hot (18)]`;
  2 validity bits. Point visibility = all joints the point depends on are visible (rigid: anchor; bone_revolution:
  measured numerically through the real poser). Zero-init MLP, `torch.where`-masked by (anchor visible &
  sensor/partner valid). Roll twin: row i gets row (i+B/2)%B features, gated by row i's own validity.
  Bare `encode()/decode()/encode_features()` never compute relations (cycle/contrast/judge).
* **Sensor** (`ground.source_root_height_sensor`): `root_height_valid & ground_valid & contact_trust &
  not DFaust/PosePrior-trust0 & any finite source_contact` (ties to the existing 0.3 contact dropout,
  no RNG). NaN for sparse/unimate/eval paths (the trainer wrapper writes NaN there).
* **O** (`relational_objective(outputs, target_batch, tgt_body, rows, weights, roll=False)`): XWBC targets,
  trusted ground rows minus DFaust / PosePrior-trust0. `h_gt` from GT pose9 posed dense points + GT root −
  beta_vis; `h_pred` from predicted `joint_rotation/joint_position` with the same GT root.
  `L_pen = mean relu(-h_pred - relu(-h_gt))²/2²` (w .5), `L_clear = Huber_1((softmin_0.5(region h_pred) −
  min region h_gt)/2)` for regions 0..14 with GT min < 20 cm (w .2), `L_hcon = BCE(contact_logits_j,
  1[min h_gt over points anchored to j < 2.5 cm])` (w .2). Returns `total`, the three terms,
  `relational_hcon_positive_rate` and `count_relational_{rows,points,regions,hcon_joint_frames}`.
  Roll twin takes every GT-derived quantity from row (i+B/2)%B; rows count only when both are valid.
* **S_wrong**: `attach_dense_geometry(..., derangement={body: donor})`: donor shape scaled by
  reach_body/reach_donor, re-anchored to the receiver's nearest rest joint; static-only (no posing, so it
  cannot be combined with I/O; raises if tried). The trainer uses the explicit map in `train/arms.json`.

## Run
```
cd '<ROOT>'   # .../alphamotion_scaling_20261004
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
  /home/arenalabs/miniconda3/envs/ardy/bin/python -m pytest -p no:cacheprovider -q am2exp_v1/tests/test_model_*.py
# CPU bench
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" timeout 1800 \
  /home/arenalabs/miniconda3/envs/ardy/bin/python -m am2exp_v1.model.bench --batch 16
```
`env -u PYTHONPATH` / `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` are needed: the ROS humble PYTHONPATH loads a
pytest plugin that crashes (missing `lark`).

## Tests (2026-10-04): 63 passed, 2 skipped on CPU (108 s) + 2 GPU tests passed (13 s, lock taken/released)
* `test_model_slide_fix.py` (V0): A vs SF parent state_dict bit-equal; init q=0.968; output change < 1e-4 when
  qp,qr ×100; targets 0/1/1.0185 reachable with |logit| < 4; nonzero grad at 0, .5, 1, 1.0185 and at logit 8;
  SF non-slide outputs == A; objective replaces L1 and restores the count; clamped native within limits.
* `test_model_dense.py`: SF⊂S⊂I state_dicts bit-equal; forward bit-equal at init S vs SF, I vs S, I_roll vs S
  (5 directions incl. human→G1, G1→human, T1→GR3, Franka, Adroit; full + masked); O model == S and objective
  runs; bodies without dense surface == SF; mixed dense/plain; bare encode/decode/encode_features/cycle
  re-encode work and ignore relations; NaN-fill of the whole source and `source_root_height_m` in sparse_only
  gives bit-identical outputs (S and I, G1/human/Franka); hidden joints NaN or garbage → identical (I, gates
  open); root height changes output only when valid; roll deterministic, row i = row (i+B/2)%B, I_roll ctor ==
  roll kwarg; every new branch gets gradient; g128 + joints_only variants identity; hop == B construction;
  S_wrong derangement; per-arm parameter counts; strict checkpoint round trip; bench smoke.
* `test_model_relational.py`: features == brute force; masking; sensor rules; objective zero pen on GT,
  pushes sunk points up; DFaust/PosePrior masks; roll counts. **Real-asset acceptance (standing G1/human sole
  groups min h in [-1,3] cm; fall torso < 5 cm) is SKIPPED: OUT/assets/surface_v2/*.npz do not exist yet.**
* `test_model_real_api.py` (new): schema npz files written to tmp and loaded through the REAL
  `assets.surface_v2.load_surface` — rigid G1 + `bone_revolution` human — identity at init S/SF/I in 6
  direction×mode cells, numeric point-dependency (hip-axis frame refs), hidden-joint invariance on the human
  source, objective on a bone_revolution target.
* `test_model_integration.py`: one full recipe step for SF, S, S_wrong, O, O_roll, I, I_roll; sensor wrapper
  consumes no RNG.
* `test_model_gpu.py`: bf16-autocast bit identity at init SF/S/I on CUDA.

## Numbers
Parameters (size S, d=128): A 6,701,777; SF 6,702,417 (+640); S 6,906,593 (+204,176 vs SF); I 6,926,817
(+20,224 relational MLP); O = S.

CPU micro-benchmark, B=16, 8 threads, one human→G1 full-mode direction, synthetic P=4096/G=256/K=16/S=1024
(`OUT/model/bench_cpu_B16.json`): fwd+bwd s / ratio vs SF / saved activations vs A / est. GPU peak at B=128:
| arm | fwd+bwd s | ×SF | act ×A | est. peak GiB |
|---|---|---|---|---|
| SF | 0.139 | 1.00 | 1.00 | 2.39 |
| S g256 full | 0.471 | 3.38 | 3.18 | 7.56 |
| S g256 joints_only | 0.250 | 1.79 | 2.18 | 5.18 |
| S g128 | 0.229 | 1.64 | 1.89 | 4.49 |
| I g256 | 0.576 | 4.14 | 3.21 | 7.63 |
| O g256 | 0.536 | 3.84 | 3.20 | 7.61 |
GPU B=16 bf16 (`OUT/model/gpu_timing_B16.json`): S/SF 0.88–1.34, I/SF 0.98–1.27 across two runs —
launch-overhead bound, not informative for the 1.6× rule. B_S (96 patches) reference: 6.81 GiB, 0.581 s vs
A_S 0.413 s per XWBC step.

## Deviations from the plan text (and why)
1. **Encoder fusion is not one softmax over [joints; groups] for joint queries.** A shared softmax over 256
   extra keys changes every joint token at init, so S ≠ SF at init. Joint queries keep the sealed SF
   attention and add a zero-gated cross-attention to the groups; group queries use the B concat + hop bias.
   Same hop construction as surface_model.py:107-116 (tested).
2. PointNet input is 24-dim (3+3+18 regions; the plan's "18" assumed 12 regions) and local offsets are
   divided by 0.05·reach (O(1) values) instead of reach.
3. Relational inputs are computed under `no_grad` (data features, no gradient into the source pose), 28 dims;
   partner region enters as a one-hot into the MLP (equivalent to an embedding).
4. `L_clear` softmin = softmax(−h/0.5)-weighted mean (≥ min, bias ≪ 0.5 cm); a −τ·logsumexp softmin sits
   τ·ln(n) ≈ 3 cm low.
5. `h_pred` uses the GT root height (the root head is not supervised by O); hcon label per joint is over points
   anchored to that joint (bone_revolution: the bone's parent joint).
6. `default_derangement` (cyclic shift) is only a fallback; S_wrong uses the explicit map in `train/arms.json`.
7. `surface_api.load_surface` now always uses the real WP-ASSETS loader when it imports (also with an
   explicit `asset_dir`), so `bone_revolution` assets load from any directory.

## Open issues
* **Real assets missing**: `OUT/assets/surface_v2/` holds only `_cache/`. Re-run
  `test_model_relational.py::test_real_asset_standing_soles_and_fall_torso` and the bench (it switches to
  real assets automatically) once WP-ASSETS writes the npz files.
* **1.6× timing rule**: CPU says S g256 = 3.4× SF, g128 = 1.6×. The decision must come from the trainer's GPU
  probe at B=128 (not run here: build-phase GPU limits). If it exceeds 1.6×, set `grouping: g128` for
  S/O/I/twins in `train/arms.json` before any run; `dense_group_attention='joints_only'` is a further option.
* human_smpl `rest_points` are the standing calibration pose, while the AM2 rest (offsets) is the
  non-standing LAFAN dialect: `centre_feat` uses the asset's rest pose; S_wrong re-anchoring onto the human
  uses AM2 rest joints (only matters for the twin).
* I/O cost on GPU at B=128 is not measured (the dense cdist and the extra posing are the main costs).

## Integration changes (WP-INTEGRATION, 2026-10-04)
- `geometry.point_trust_mask` + static key `point_trusted` [P]: points whose WP-ASSETS data residual p95 > 3 cm
  (kuavo_s45 457, unitree_h1_2 219) are excluded from arm O losses (`losses_relational`) and treated as never visible
  in arm I features (`relational`). Static S group tokens are unchanged. Tests: `tests/test_integration_contracts.py`.
