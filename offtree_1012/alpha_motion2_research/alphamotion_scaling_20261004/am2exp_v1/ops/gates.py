"""Auto-gate rules for twins (S_wrong, O_roll, I_roll, G1_roll).

A rule is ``fn(job, params, ctx) -> (decision, reason)`` with decision in
{'run', 'skip', 'blocked'}:
  run      the twin is needed (test arm beat its parent) -> queue runs it
  skip     not needed (test arm did not beat its parent) -> recorded, queue continues
  blocked  cannot decide yet (no frozen decision) -> queue stops, nothing runs
The queue never computes statistics itself. The statistics/judge package writes a frozen,
hashed decision file per contrast; this keeps the gate independent of training losses and
makes the decision auditable:

  OUT/judge/decisions/<test>_vs_<parent>.json
  {"test": "S", "parent": "SF", "beats_parent": true|false, "rule": "...", "evidence": {...},
   "lloyd_agrees": true|false|null, ...}

``beats_parent`` must be a JSON boolean. If the decision file also carries ``lloyd_agrees`` and
it is false, the twin is skipped (lloyd overrules).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from am2exp_v1.common.paths import OUT

DECISIONS = OUT / 'judge' / 'decisions'


def decision_path(test, parent, root=None):
    return Path(root or DECISIONS) / f'{test}_vs_{parent}.json'


def decision_file(job, params, ctx=None):
    path = decision_path(params['test'], params['parent'], (ctx or {}).get('decisions_dir'))
    if not path.exists():
        return 'blocked', f'no frozen decision {path}'
    raw = path.read_bytes()
    d = json.loads(raw)
    if d.get('test') not in (None, params['test']) or d.get('parent') not in (None, params['parent']):
        return 'blocked', f'decision file {path} is for a different contrast'
    if not isinstance(d.get('beats_parent'), bool):
        return 'blocked', f'decision file {path} lacks a boolean beats_parent'
    sha = hashlib.sha256(raw).hexdigest()
    if d.get('lloyd_agrees') is False:
        return 'skip', f'lloyd overruled ({path}, sha256 {sha})'
    if d['beats_parent']:
        return 'run', f'{params["test"]} beats {params["parent"]} ({path}, sha256 {sha})'
    return 'skip', f'{params["test"]} does not beat {params["parent"]} ({path}, sha256 {sha})'


def always(job, params, ctx=None):
    return 'run', 'always'


def never(job, params, ctx=None):
    return 'skip', 'never'
