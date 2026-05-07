"""train_all.py — CLI launcher for training all variants under one mode.

Run from project root in a separate terminal (not from the notebook):

    cd notebook
    python 02_resources/scripts/train_all.py --mode smoke
    python 02_resources/scripts/train_all.py --mode headline
    python 02_resources/scripts/train_all.py --mode full

While it's running, in two more terminals:
    python 02_resources/scripts/monitor.py
    tensorboard --logdir 02_resources/tensorboard/

Then the notebook is pure post-training analysis: it reads ckpts + loss_history.json
+ TB events from disk; it does NOT launch training itself.

Implementation:
    - subprocess.Popen per variant
    - N_PARALLEL configurable (default 3 — empirical safe ceiling on RTX PRO 6000 96 GB)
    - Wait + refill queue as variants finish
    - One automatic retry pass for OOM-failed variants (GPU is clean by then)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO   = Path(__file__).resolve().parent.parent.parent.parent
TRAIN  = REPO / "notebook/02_resources/scripts/train_runner.py"
CKPT   = REPO / "notebook/02_resources/checkpoints"
CFG    = REPO / "notebook/01_resources/configs/variants.json"


def _launch(v: str, mode: str, no_resume: bool):
    log_dir = CKPT / v; log_dir.mkdir(parents=True, exist_ok=True)
    f = (log_dir / "stdout.log").open("a")
    cmd = [sys.executable, str(TRAIN), "--variant", v, "--mode", mode]
    if no_resume:
        cmd.append("--no-resume")
    f.write(f"\n=== launched at {time.strftime('%F %T')}: {' '.join(cmd)} ===\n"); f.flush()
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    return p, f


def run(variants: list[str], mode: str, n_parallel: int, no_resume: bool):
    """Launch with parallel queue. Returns dict {variant: returncode}."""
    procs = {}
    queue = list(variants)

    while queue and len(procs) < n_parallel:
        v = queue.pop(0)
        procs[v] = _launch(v, mode, no_resume)
        print(f"launched {v} (PID {procs[v][0].pid})")

    rc_table = {}
    t0 = time.time()
    while procs:
        time.sleep(5)
        for v, (p, f) in list(procs.items()):
            if p.poll() is not None:
                f.close()
                rc_table[v] = p.returncode
                ok = "✓" if p.returncode == 0 else "✗"
                print(f"  {ok} {v}  rc={p.returncode}  ({time.time()-t0:.0f}s)")
                del procs[v]
                if queue:
                    nxt = queue.pop(0)
                    procs[nxt] = _launch(nxt, mode, no_resume)
                    print(f"launched {nxt} (PID {procs[nxt][0].pid})")
    print(f"\nGroup done in {time.time()-t0:.0f}s\n")
    return rc_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="smoke", choices=["smoke", "headline", "full"])
    p.add_argument("--n-parallel", type=int, default=3,
                   help="Concurrent training subprocesses. 3 is the empirical safe ceiling on RTX PRO 6000 96 GB.")
    p.add_argument("--no-resume", action="store_true",
                   help="Force fresh training; ignore latest.pt")
    p.add_argument("--no-retry", action="store_true",
                   help="Skip the automatic OOM-retry pass")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Override variant list (default = all from configs/variants.json)")
    args = p.parse_args()

    all_variants = args.variants or list(json.loads(CFG.read_text()).keys())
    print(f"=== train_all.py  mode={args.mode}  variants={all_variants}  n_parallel={args.n_parallel} ===\n")

    rc = run(all_variants, args.mode, args.n_parallel, args.no_resume)
    failed = [v for v, c in rc.items() if c != 0]

    if failed and not args.no_retry:
        print(f"=== retry pass for failed variants: {failed} ===\n")
        # Wait briefly for GPU to clear
        time.sleep(10)
        rc2 = run(failed, args.mode, args.n_parallel, no_resume=True)
        rc.update(rc2)
        still_failed = [v for v, c in rc.items() if c != 0]
        if still_failed:
            print(f"\n✗ still failed after retry: {still_failed}")
            sys.exit(1)

    print(f"\n✓ all variants done: {[v for v in all_variants if rc.get(v) == 0]}")


if __name__ == "__main__":
    main()
