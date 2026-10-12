# integration/ (WP-INTEGRATION)

Checks that the six packages (ops, assets, model, judge+eval, e2, render) work together, runs the CPU end-to-end
smoke, and indexes the Stage-0 artifacts. It launches no training and no GPU timing probe, and takes no GPU lease.

| file | what |
|---|---|
| `smoke_e2e.py` | stages `dryrun` / `untrained` / `e1` / `render` / `index` / `all`; children run with timeouts; outputs go to `OUT/integration/` |
| `index.py` | writes `OUT/stage0_index.md`: paths of every Stage-0 artifact plus the gate verdicts copied from the owning reports |
| `../tests/test_integration_contracts.py` | 6 cross-package tests on the real assets and smoke outputs |

Run (from ROOT): `env -u PYTHONPATH PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl $PY -m am2exp_v1.integration.smoke_e2e all`
Tests: `env -u PYTHONPATH MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' $PY -m pytest -p no:cacheprovider am2exp_v1/tests --deselect am2exp_v1/tests/test_model_gpu.py`
(`MUJOCO_GL=egl` is required; without it `test_render_e1::test_gt_self_render_panels_identical` fails because no GL context exists. The ROS humble `PYTHONPATH` must be removed.)

## Fixes made between packages (each marked `[integration]` in the code)

1. **Badly bound surface points now reach the model.** WP-ASSETS stores `point_data_residual_p95_cm` for each point
   and asks users to drop points above 3 cm (457 on kuavo_s45, 219 on unitree_h1_2). WP-MODEL never read it.
   - New: `model/geometry.point_trust_mask` and a static key `point_trusted`.
   - Arm O losses (`losses_relational`) skip those points, for penetration, clearance and contact labels.
   - Arm I features (`relational`) treat them as never visible, for ground and self-contact.
   - S group tokens (rest shape only) are unchanged, so the S == SF identity at init still holds.
   - Tested: moving every untrusted point 50 cm leaves the O loss and the I features exactly the same; without the
     mask the loss changes.
2. **Sinking hands in the human video panel.** With `human_smpl.npz` now built, the renderer colours the human from
   the dense surface. Its hand is an ellipsoid that sinks about 20 cm when the palm is planted. Hand regions now use
   the wrist joint centre minus 3 cm (`render/contact.py`).

No other mismatches were found:
- Every registry symbol imports.
- The geometry hook loads all 18 real assets through `assets.surface_v2`.
- `run_e1_eval` loads both the original A0 and an am2exp S checkpoint written in `train_arm`'s format.
- The judge and the renderer both read `heldout_root_bias_v1.json`.
- `e2.xbody_loss` binds in both argument orders.

## Smoke results (2026-10-04)

**Dry runs** (`train_arm --dry-run-cpu --max-steps 5`, real dense assets, batch 128):

| arm | status | sampling hashes vs runs/A_S_seed0 | independent re-check | RSS peak |
|---|---|---|---|---|
| SF | completed | 5/5 match | ok | 1.8 GiB |
| S | completed | 5/5 match | ok | 4.9 GiB |
| O | completed | 5/5 match | ok | 4.9 GiB |
| I | completed | 5/5 match | ok | 6.5 GiB |
| S_wrong | completed | 5/5 match | ok | 5.5 GiB |

Steps 1-5 of the seed-0 stream are all UniMate, so these 5 steps never touch a body with a dense surface. To exercise
the new code on XWBC steps:
- **Batch 128:** S hit 19.2 GiB RSS at its first XWBC step on CPU and was killed by the watchdog. **Dense arms cannot
  take an XWBC step on CPU at batch 128 inside the 20 GB rule.**
- **Batch 32, 7 steps** (`dryrun7_b32/`; 3 XWBC steps; hashes not comparable at batch 32): S, O, I and S_wrong all
  complete at about 8.5 GiB and about 11-13 s per XWBC step on CPU.
- **Arm O, step 6** (`dryrun6_b32/O`), relational terms are live: pen 3.83, clear 8.58, hcon 1.64, 29 rows,
  118,784 points.
- **Arm I** (`dryrun6_b32/I_probe`, an instrumented in-process run): features were computed on every XWBC encode.
  - Ground-valid fraction is 0.39-0.75 per direction, and 0 when the contact sensor was dropped.
  - Self-valid fraction is 0.72-0.92.

**E1 smoke** (`OUT/integration/e1/`): `run_e1_eval` in sensor_on mode, human -> G1 and human -> GR3,
fallAndGetUp1_s4@4136.
- A0@3000 gives predictions bit-identical to WP-JUDGE's Stage-0 npz for the same case.
- The untrained S (step-0 seed-0 init, `ckpt/S_untrained/`) loads through the am2exp registry path.

J1 scores:

| case | arm | CSE_GT | FloatRate | PenP95 cm | SelfPen |
|---|---|---|---|---|---|
| G1 | A0 | 0.994 | 0.97 | 0.0 | 0.27 |
| G1 | S untrained | 0.302 | 0.00 | 99.1 | 1.00 |
| GR3 | A0 | 0.938 | 0.81 | 1.0 | 0.30 |
| GR3 | S untrained | 0.546 | 0.00 | 79.3 | 1.00 |

**Render:**
- Smoke triptych [human | A0 | S untrained] for G1, at `videos/human_smpl__unitree_g1_29dof__...`.
- Stage-0 [human | GT | A0] for G1 and GR3, at `videos/stage0/`. These are the videos the headroom gate needs.
- What they show: GT G1 and GR3 sit and crawl on the floor, while A0 stands upright through the whole window.

## Risks found here (not fixed: they need an owner decision)

1. **The human dense surface sinks into the floor on sitting and lying frames.** Measured on fallAndGetUp1_s4@4136
   with the label root height:
   - The pelvis region is below −1 cm on 76% of frames (p50 −12.4 cm). The right thigh is below on 50% and the
     forearms on 45-72%.
   - Likely cause: bone frames are built from joint positions only, so the standing-calibration pelvis and buttock
     shape does not tilt with the pelvis.
   - Effects:
     - Arm I human-source ground features read deep "penetration" where the truth is contact.
     - Arm O human-target labels are self-consistent (pred and GT use the same surface), so that use is less affected.
     - The human video panels show red pelvis and legs.
   - Owner: WP-ASSETS (for example a pelvis frame from both hips plus the spine, or a per-region offset). This
     should be decided before S/O/I training, because it changes what arm I's input means on exactly the
     contact-rich motions E1 targets.
2. **CSE_GT rewards a body that is sunk everywhere.** The untrained S, buried 80-99 cm, scores a better CSE_GT than A0
   on both smoke cases. The pre-registered guards (PenP95 +1 cm, SelfPen) catch this, so the decision rule must never
   be read on CSE alone.
3. **Memory and timing are unmeasured where training will run.** Dense arms on CPU at batch 128 need more than
   19 GiB, and the WP-OPS earlier note (I: 28 GB) applies to S, O and S_wrong too. On the 96 GB GPU this is not a
   blocker. But nothing has measured GPU memory or speed at batch 128 (WP-MODEL's estimate is 7.6 GiB), and the
   1.6x timing rule for choosing g256 or g128 still needs a GPU probe. lloyd must approve that probe; it was not run
   here.
4. Gate failures carried over from Stage-0, unchanged:
   - Strict ruler gate: 6/15.
   - Shuffled-source floor fails, because A0 ignores the source for contact.
   - E2 G-0 passes on dev only (test BA 0.539).
