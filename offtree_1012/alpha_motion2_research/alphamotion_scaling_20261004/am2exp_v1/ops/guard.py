"""Two-partition capacity guard for am2exp (replaces train_scaling.guard(), which needs +31 GiB).

Rule (final_plan.md "Ops"):
  /home : free >= downloader remaining + 16 GiB (+ any not-yet-written /home allocation)
          remaining comes from base/remote_state.capacity() (read-only; it executes the
          downloader's own capacity arithmetic) and the combined plan must be exactly
          200,505,865,539 bytes (identity check, as in every research guard).
  /     : free >= 20 GiB (+ any declared not-yet-written OUT allocation)
All am2exp outputs live on / (/var/tmp/alphamotion/am2exp); /home only holds CODE (< 50 MB),
so the /home allocation defaults to what CODE may still grow by.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if _sys.path and _sys.path[0] and _Path(_sys.path[0]).resolve() == _Path(__file__).resolve().parent:
    _sys.path.pop(0)  # running ops/<x>.py as a script: ops/queue.py must not shadow the stdlib 'queue'
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from am2exp_v1.common.paths import CODE, OUT, PLAN_IDENTITY_BYTES, add_sealed_paths  # noqa: E402

GiB = 1024 ** 3
HOME_RESERVE = 16 * GiB          # 15 GiB downloader reserve + 1 GiB metadata overhead
ROOT_FLOOR = 20 * GiB
CODE_ALLOCATION = 50 * 1024 ** 2  # interfaces.md: CODE < 50 MB


class CapacityError(RuntimeError):
    pass


def _tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except FileNotFoundError:
                pass
    return total


def _remote_capacity():
    add_sealed_paths()
    import remote_state  # sealed base/remote_state.py, read-only
    return remote_state.capacity()


def check(out_allocation_bytes=0, home_allocation_bytes=None, capacity_fn=None, root_path=None, raise_on_fail=True):
    """Return a JSON-safe report of both partitions; raise CapacityError if either fails.

    out_allocation_bytes: total bytes the current plan may still write under OUT (already
        written bytes under OUT are subtracted, so the requirement shrinks as outputs land).
    home_allocation_bytes: bytes still to be written on /home (default: CODE headroom to 50 MB).
    capacity_fn: injectable for tests; defaults to base/remote_state.capacity.
    """
    try:
        cap = (capacity_fn or _remote_capacity)()
    except Exception as exc:  # remote_state.capacity raises ValueError itself when the plan identity changed
        raise CapacityError(f'capacity probe failed (base/remote_state.capacity): {exc!r}') from exc
    problems = []
    if int(cap['expected_unique_bytes']) != PLAN_IDENTITY_BYTES:
        problems.append(f"download plan identity changed: {cap['expected_unique_bytes']} != {PLAN_IDENTITY_BYTES}")
    code_bytes = _tree_bytes(CODE)
    if home_allocation_bytes is None:
        home_allocation_bytes = max(0, CODE_ALLOCATION - code_bytes)
    home_free = int(cap['available_bytes'])
    remaining = int(cap['remaining_bytes'])
    # The downloader's own required_available_bytes (= remaining + plan reserve + metadata overhead)
    # must never exceed ours; take the larger so a plan-side reserve change can only tighten the rule.
    downloader_required = int(cap.get('required_available_bytes', 0))
    home_required = max(remaining + HOME_RESERVE, downloader_required) + int(home_allocation_bytes)
    if home_free < home_required:
        problems.append(f'/home free {home_free} < remaining {remaining} + 16 GiB + allocation {home_allocation_bytes} = {home_required}')
    root_path = Path(root_path or (OUT if OUT.exists() else OUT.parent if OUT.parent.exists() else '/'))
    root_free = shutil.disk_usage(root_path).free
    out_used = _tree_bytes(OUT)
    out_pending = max(0, int(out_allocation_bytes) - out_used)
    root_required = ROOT_FLOOR + out_pending
    if root_free < root_required:
        problems.append(f'/ free {root_free} < 20 GiB + pending OUT allocation {out_pending} = {root_required}')
    report = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed': not problems, 'problems': problems,
        'home': {'partition': '/home', 'free_bytes': home_free, 'free_gib': round(home_free / GiB, 2),
                 'downloader_remaining_bytes': remaining, 'reserve_bytes': HOME_RESERVE,
                 'allocation_bytes': int(home_allocation_bytes), 'required_bytes': home_required,
                 'downloader_required_available_bytes': downloader_required,
                 'headroom_bytes': home_free - home_required, 'headroom_gib': round((home_free - home_required) / GiB, 2),
                 'plan_expected_unique_bytes': int(cap['expected_unique_bytes']),
                 'plan_identity_ok': int(cap['expected_unique_bytes']) == PLAN_IDENTITY_BYTES,
                 'complete_blob_bytes': int(cap.get('complete_blob_bytes', -1)),
                 'partial_blob_bytes': int(cap.get('partial_blob_bytes', -1)), 'code_bytes': code_bytes},
        'root': {'partition': '/', 'path_checked': str(root_path), 'free_bytes': root_free,
                 'free_gib': round(root_free / GiB, 2), 'floor_bytes': ROOT_FLOOR, 'out_used_bytes': out_used,
                 'out_allocation_bytes': int(out_allocation_bytes), 'out_pending_bytes': out_pending,
                 'required_bytes': root_required, 'headroom_bytes': root_free - root_required,
                 'headroom_gib': round((root_free - root_required) / GiB, 2)},
    }
    if problems and raise_on_fail:
        raise CapacityError('; '.join(problems) + ' | ' + json.dumps(report))
    return report


def describe(report):
    h, r = report['home'], report['root']
    return (f"[guard {'PASS' if report['passed'] else 'FAIL'}] "
            f"/home free {h['free_gib']} GiB, required {round(h['required_bytes'] / GiB, 2)} GiB "
            f"(downloader remaining {round(h['downloader_remaining_bytes'] / GiB, 2)} GiB + 16 GiB + alloc "
            f"{round(h['allocation_bytes'] / GiB, 3)} GiB), headroom {h['headroom_gib']} GiB, plan identity "
            f"{'ok' if h['plan_identity_ok'] else 'CHANGED'} | / free {r['free_gib']} GiB, required "
            f"{round(r['required_bytes'] / GiB, 2)} GiB (20 GiB floor + pending {round(r['out_pending_bytes'] / GiB, 2)} GiB), "
            f"headroom {r['headroom_gib']} GiB")


def main():
    import argparse
    ap = argparse.ArgumentParser(description='am2exp two-partition capacity guard (read-only).')
    ap.add_argument('--out-allocation-gib', type=float, default=0.)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    rep = check(out_allocation_bytes=int(args.out_allocation_gib * GiB), raise_on_fail=False)
    print(describe(rep))
    if args.json:
        print(json.dumps(rep, indent=2))
    sys.exit(0 if rep['passed'] else 3)


if __name__ == '__main__':
    main()
