# train/: arm registry + trainer (WP-OPS)

`train_arm.py --arm NAME [--max-steps N] [--dry-run-cpu] [--lease-fd FD] [--approval-stage S] [--override JSON]`
clones the orchestration/train_scaling.py main loop for one arm. It never calls `original.main()`/`guard()`; it imports
`train_scaling as original` and reuses load_scaling_bodies, backward_training_step, finite_backward, ComputeCounter,
effective_tokens, checkpoint, evaluate, loss_function, forward, autocast, digest, source_hashes.

Identical to the original for seed-0 A-stream arms: build_model seeding (torch/np/random seed before construction),
streams `default_rng(seed)`, `default_rng(4)`, `default_rng(seed*1000+77)`, one uniform draw per step for the domain,
`torch.manual_seed(seed*1e6+step)`, AdamW + CosineAnnealingLR(T_max from the registry), sampling digest
over directions + auxiliary.

Differences (declared in arms.json): model class by name; geometry hook (dense surfaces); init checkpoint (E2);
objective wrapper (slide-fix loss replaces slide L1 which stays logged at weight 0; relational objective on XWBC
targets only, target side); input wrappers for arm I (rebind recipe_training.make_inputs/make_reference_inputs and
original.make_inputs, add `source_root_height_m`/`roll_relations` only, assert the numpy RNG state is unchanged);
mixtures (original / E2 4-way / franka_only for V1); optional side batch on private RNGs under torch.random.fork_rng.

Sampling assert: per step, before `opt.step()`, the hash is compared with `reference_train_jsonl`; first mismatch ->
`SamplingMismatch`, failure.json, exit_status "failed". XWBC hashes include device-side torch.rand (reference masks),
so they are only comparable on the same device type (CPU dry runs mark them "unverifiable"; CUDA runs assert them).
The domain must always agree. SF_s1 / G0 / V1_old record only; G1 and G1_roll assert against G0's train_slim.jsonl;
V1_new against V1_old.

Logging: `train_slim.jsonl` every step (hash, domain, total, grad norm, lr, time, slide diagnostics, e2 terms);
`train.jsonl` full record at step 1, every 10 steps and the last step. Checkpoints weights-only at the registry steps
(1000 [, 3000 when < steps]) via original.checkpoint (no Adam) and a final `checkpoint_stepNNNN.pt` without Adam.
GPU runs > 12 steps need `OUT/logs/approvals/<stage>.ok`.

## E2 cross-body loss contract (WP-E2, e2/xbody_loss.py)
`xbody_loss(model, bodies, batch_rng_state, cfg) -> {'total': tensor (graph-connected, = w_xb * L, also at w_xb 0),
'terms': {...}, 'sampling': {...rows only, never w_xb/roll...}}`. `bodies` is {name: Body}; `batch_rng_state` is the run's
main numpy Generator (rows drawn identically whatever w_xb/roll); cfg carries w_xb, roll, batch_size, device, autocast,
step and an `info` dict the loss fills with rows_A/rows_F. The trainer also binds the (cfg, rng) order by parameter name.
The hashed auxiliary record of a hand_gripper step holds only rows + data-RNG state, so G0 == G1 == G1_roll hashes.

## Known limits
- Arms S/O/I/S_wrong/O_roll/I_roll need OUT/assets/surface_v2 (WP-ASSETS); not present at the time of writing, so
  their real dry runs fail fast at the geometry `require` list. `train/test_hooks.py` (synthetic capsules) is test-only.
- Arm I on CPU at batch 128 peaks ~28 GB RSS (relational features); the RSS sampler aborts it between steps. Run I on
  the GPU, or dry-run with a smaller batch.
- G0/G1 need OUT/runs/SF/checkpoint_step3000.pt (from SF); dry runs override init_checkpoint to null.
