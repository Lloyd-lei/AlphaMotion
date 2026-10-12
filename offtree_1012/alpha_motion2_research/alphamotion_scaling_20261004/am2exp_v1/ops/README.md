# WP-OPS: common/, ops/, train/

## What it does
- `common/paths.py` - ROOT, CODE, OUT, PY, PYROT, `add_sealed_paths()` (interfaces.md) + GPU_LOCK (inode 55451429),
  PLAN_IDENTITY_BYTES, SEALED_DIRS, BASE_ENV. Import am2exp code as `am2exp_v1.<pkg>` with ROOT on sys.path
  (CODE/common and CODE/model clash with sealed base/legacy/common.py and model.py).
- `ops/lease.py` - `GpuLease`: pre-checks /proc/locks for the lock inode, opens WITHOUT O_CREAT (never recreates),
  verifies inode, `flock(LOCK_EX|LOCK_NB)` (second taker -> BlockingIOError). `handoff_fd()` / `GpuLease.inherit(fd)`
  let the queue pass the held lease to a child job (no release/re-acquire race). `python ops/lease.py` = read-only status.
- `ops/guard.py` - two-partition guard: /home free >= downloader remaining + 16 GiB (+CODE headroom; the downloader's own
  required_available_bytes can only tighten it) using sealed `base/remote_state.capacity()` read-only, with the
  200,505,865,539-byte plan identity check; AND / free >= 20 GiB + pending OUT allocation. `python ops/guard.py [--json]`.
- `ops/receipts.py` - identity (pid, start_ticks, boot_id, argv), atomic state.json, exit_status.json in `finally`
  (SIGTERM -> status "terminated"), RSS sampler thread (20 GiB limit flag), `free -g` snapshots.
- `ops/queue.py` - single sequential owner (`--plan PLAN.json`, `--status`). Holds the GPU lease for the whole queue
  plus a per-plan launcher lock; per job: approval file `OUT/logs/approvals/<stage>.ok` -> auto gate (twins) ->
  capacity guard -> child with inherited lease fd, wall-clock lifetime (SIGTERM process group, SIGKILL after grace).
  Any failure/timeout stops the queue; rerun resumes at the first unfinished job; completed arm runs are skipped.
  `ops/plans/core_v1.json` is the ordered plan (stages: pipeline, probes, core_sf, core_e1, core_e2, twins_e1, twins_e2).
- `ops/gates.py` - twin rule `decision_file`: reads frozen `OUT/judge/decisions/<test>_vs_<parent>.json`
  (`beats_parent` bool, optional `lloyd_agrees`); missing -> queue stops (blocked), never guesses.
- `train/train_arm.py` + `train/arms.json` - see train/README.md.

## Run
    PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 $PY ops/guard.py
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= $PY train/train_arm.py --arm Arep --dry-run-cpu
    touch /var/tmp/alphamotion/am2exp/logs/approvals/pipeline.ok   # lloyd only
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 $PY ops/queue.py --plan ops/plans/core_v1.json
(`env -u PYTHONPATH`: the login shell exports ROS humble paths which break pytest plugin autoload; the queue strips
PYTHONPATH from children.)

## Tests (pytest, CPU)
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= $PY -m pytest -q -p no:cacheprovider tests/test_ops_*.py
- test_ops_lease_guard_receipts.py (12): second taker BlockingIOError (same process, other process, and on the real
  lock for < 1 s when free), no recreate, inode check, inherited lease, guard passes today and prints both partitions,
  guard rules with injected capacity, receipts on exception and SIGTERM, RSS limit.
- test_ops_queue.py (10): approval gate/order/resume, lease held elsewhere, duplicate owner, twin gates
  (run/skip/blocked/lloyd overrule), lifetime kill, failed job stops queue, capacity block, registry-driven arm jobs,
  shipped plan stops at the first approval.
- test_ops_train_arm.py (11, ~4 min): registry facts, comparison rules, mixture RNG use, xbody binding, CPU dry runs
  (Arep 5 steps == A_S_seed0 hashes; perturbed RNG aborts at step 3; I-wrappers hash-identical incl. XWBC step 6;
  G0 == G1 hashes over 10 steps with the real e2 xbody_loss). `AM2_SKIP_DRYRUN=1` skips the dry runs.

## Acceptance results (2026-10-04)
- All 33 WP-OPS tests pass. Guard: /home free 171.7 GiB vs required 166.0 GiB (headroom 5.7 GiB), / free 92 GiB.
- CPU Arep 5 steps: 5/5 sampling hashes == runs/A_S_seed0; loss max rel diff 0.28 % (CPU fp32 vs CUDA bf16 ref).
- GPU smoke (one run, lease held 16 s): Arep 12 steps on CUDA -> 12/12 hashes match (8 unimate + 4 XWBC), step-1 loss
  bit-identical (17.580429), max rel loss diff 1.5e-4; 3.4 s train time, 2.4 GiB peak CUDA. Lease released.
  Receipt: OUT/runs/_smoke/Arep_12steps_20261004T231142Z.
