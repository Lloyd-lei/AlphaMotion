"""Frozen E1 case list (final_plan.md "E1 evaluation").

Builds OUT/eval/cases_e1_v1.json (+ its sha256 inside and in a .sha256 side file) and the TRAIN-split
calibration list OUT/eval/cases_rootbias_v1.json used only to measure the heldout-body root bias of A0.

Windows (128 native frames, 30 fps, contiguous rows of the original caches):
  * unseen_motion_listed : the 6 split-1 windows named in final_plan.md
  * unseen_motion_mined  : FROZEN RULE on split-1 clips -- human non-foot contact fraction >= 0.5 (per frame: any
                           human joint whose labeller limb group is not L_foot/R_foot has hard contact, using the
                           reader's definition hard|(gap<0) on trusted finite rows), every frame trusted, candidate
                           starts every 8 frames, greedy by (-fraction, start), non-overlapping with each other and
                           with the listed windows, <= 4 mined windows per clip, DFaust/PosePrior/non-flat excluded
  * guard                : contact-poor guards (pushAndStumble1_s5 [5384,5512), walk2_s3 centre window)
  * unseen_body          : 4 train-split windows, heldout targets only
Sources: human_smpl (primary), unitree_g1_29dof, fourier_n1 (secondary).  Targets: all 15 robots for
unseen-motion/guard windows (target != source), the 4 heldout robots for unseen-body windows.
Modes (sensor_on / sensor_off) are applied by the runner; the case list is mode-free.

Usage:  python -m am2exp_v1.eval.cases [--out OUT/eval] [--force]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
OUT = Path('/var/tmp/alphamotion/am2exp')
EVAL_DIR = OUT / 'eval'
CASES_FILE = EVAL_DIR / 'cases_e1_v1.json'
ROOTBIAS_FILE = EVAL_DIR / 'cases_rootbias_v1.json'
MANIFEST = ROOT / 'data/data_manifest_v2.json'
WINDOW = 128
SCHEMA = 'am2exp_e1_cases_v1'

ROBOTS_TRAIN = ['booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi', 'pnd_adam_lite',
                'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2']
ROBOTS_HELDOUT = ['fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1']
ROBOTS = ROBOTS_TRAIN + ROBOTS_HELDOUT
SOURCES = ['human_smpl', 'unitree_g1_29dof', 'fourier_n1']
PRIMARY_SOURCE = 'human_smpl'

LISTED = [  # (clip, start) -- split-1 unseen-motion windows from final_plan.md
    ('fallAndGetUp1_subject4', 4136), ('fallAndGetUp1_subject4', 4488), ('pushAndStumble1_subject2', 5488),
    ('multipleActions1_subject3', 136), ('push1_subject2', 2280),
    ('ACCAD__Male2Walking_c3d__B24_-__walk_to_crouch_stageii', 48)]
GUARDS = [('pushAndStumble1_subject5', 5384), ('walk2_subject3', 'centre')]
UNSEEN_BODY = [('ground1_subject5', 3568), ('ground2_subject3', 5208), ('ground1_subject1', 4408),
               ('fallAndGetUp2_subject2', 3344)]
# TRAIN-split calibration (heldout root bias of A0): centre windows of every split-0 LAFAN walk clip.
ROOTBIAS_CLIP_RULE = r'^walk\d_subject\d$'

MINE = dict(min_nonfoot_frac=0.5, start_stride=8, max_per_clip=4, require_all_trusted=True)
EXCLUDE_DATASETS = ('DFaust', 'PosePrior')
NONFLAT_RE = re.compile(r'obstacle|vault|slope|handrail|stair|parkour|JumpOverObstacle|climb|ramp|step_up|bench', re.I)


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _reader():
    sys.dont_write_bytecode = True
    p = str(ROOT / 'data')
    if p not in sys.path:
        sys.path.insert(0, p)
    import trajectory_reader as TR
    return TR


def load_manifest():
    return json.loads(MANIFEST.read_text())


def group_by_clip(manifest):
    out = {}
    for g in manifest['xwbc']['groups']:
        if g['clip'] in out:
            raise ValueError(f'duplicate clip name {g["clip"]}')
        out[g['clip']] = g
    return out


def excluded(g) -> str | None:
    if g['source_dataset'] in EXCLUDE_DATASETS:
        return 'dataset ' + g['source_dataset']
    if NONFLAT_RE.search(g['clip']):
        return 'non-flat ground'
    return None


def human_contact_window(manifest, g, body='human_smpl'):
    """Human labeller contact for a whole clip: (nonfoot_any [L] bool, trusted [L] bool, hard [L,J], valid [L,J], names)."""
    TR = _reader()
    b = next(b for b in manifest['xwbc']['bodies'] if b['body'] == body)
    path = b['caches'][g['cache_index']]['contact_npz']
    with np.load(path, allow_pickle=False) as z:
        hdr = json.loads(str(z['header']))
    s, L = int(g['start']), int(g['length'])
    gap = TR.npz_rows(path, 'gap', s, L).astype(np.float32)
    hard = TR.npz_rows(path, 'contact', s, L).astype(bool)
    trust = TR.npz_rows(path, 'trust', s, L).astype(bool)
    valid = trust[:, None] & np.isfinite(gap)
    hard = (hard | (gap < 0)) & valid
    nonfoot = np.array([grp not in ('L_foot', 'R_foot') for grp in hdr['limb_groups']])
    return hard[:, nonfoot].any(1), trust & np.isfinite(gap).all(1), hard, valid, hdr['joint_names']


def mine_clip(nonfoot_any, trusted, blocked, rule=MINE, window=WINDOW):
    """Frozen greedy miner.  blocked: list of (start, stop) already taken in this clip.  Returns [(start, frac)]."""
    L = len(nonfoot_any)
    cands = []
    for s in range(0, L - window + 1, rule['start_stride']):
        if rule['require_all_trusted'] and not trusted[s:s + window].all():
            continue
        f = float(nonfoot_any[s:s + window].mean())
        if f >= rule['min_nonfoot_frac']:
            cands.append((-f, s))
    cands.sort()
    taken = list(blocked); out = []
    for negf, s in cands:
        if len(out) >= rule['max_per_clip']:
            break
        if any(s < b1 and b0 < s + window for b0, b1 in taken):
            continue
        taken.append((s, s + window)); out.append((s, -negf))
    return sorted(out)


def _window(g, start, kind, extra=None):
    start = int((g['length'] - WINDOW) // 2) if start == 'centre' else int(start)
    if not 0 <= start <= g['length'] - WINDOW:
        raise ValueError(f'{g["clip"]}: window [{start},{start + WINDOW}) outside clip of {g["length"]} frames')
    w = dict(window_id=f'{g["clip"]}@{start}', clip=g['clip'], group_id=g['group_id'], source_dataset=g['source_dataset'],
             cache_index=int(g['cache_index']), clip_start_row=int(g['start']), source_frames=int(g['length']),
             start=start, frames=WINDOW, split=int(g['split']), kind=kind)
    w.update(extra or {})
    return w


def window_stats(manifest, g, start):
    nf, tr, hard, valid, names = human_contact_window(manifest, g)
    sl = slice(start, start + WINDOW)
    return dict(human_nonfoot_contact_frac=float(nf[sl].mean()), human_trusted_frac=float(tr[sl].mean()))


def build_windows(manifest):
    by = group_by_clip(manifest)
    windows = []
    for clip, s in LISTED:
        g = by[clip]
        if g['split'] != 1:
            raise ValueError(f'listed unseen-motion clip {clip} is not split 1')
        w = _window(g, s, 'unseen_motion_listed'); w.update(window_stats(manifest, g, w['start'])); windows.append(w)
    listed_blocks = {}
    for w in windows:
        listed_blocks.setdefault(w['clip'], []).append((w['start'], w['start'] + WINDOW))
    for clip, s in GUARDS:
        g = by[clip]
        if g['split'] != 1:
            raise ValueError(f'guard clip {clip} is not split 1')
        w = _window(g, s, 'guard'); w.update(window_stats(manifest, g, w['start'])); windows.append(w)
        listed_blocks.setdefault(clip, []).append((w['start'], w['start'] + WINDOW))
    mined_log = []
    for g in manifest['xwbc']['groups']:
        if g['split'] != 1:
            continue
        why = excluded(g)
        if why or g['length'] < WINDOW:
            mined_log.append(dict(clip=g['clip'], skipped=why or 'shorter than window'))
            continue
        nf, tr, *_ = human_contact_window(manifest, g)
        picks = mine_clip(nf, tr, listed_blocks.get(g['clip'], []))
        mined_log.append(dict(clip=g['clip'], mined=[s for s, _ in picks], best_frac=float(max(
            [nf[s:s + WINDOW].mean() for s in range(0, len(nf) - WINDOW + 1, MINE['start_stride'])] or [0.0]))))
        for s, f in picks:
            w = _window(g, s, 'unseen_motion_mined'); w.update(window_stats(manifest, g, s)); windows.append(w)
    for clip, s in UNSEEN_BODY:
        g = by[clip]
        if g['split'] != 0:
            raise ValueError(f'unseen-body clip {clip} must be train split')
        w = _window(g, s, 'unseen_body'); w.update(window_stats(manifest, g, w['start'])); windows.append(w)
    ids = [w['window_id'] for w in windows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate window ids')
    return windows, mined_log


def build_cases(windows):
    cases = []
    for w in windows:
        targets = ROBOTS_HELDOUT if w['kind'] == 'unseen_body' else ROBOTS
        for src in SOURCES:
            for tgt in targets:
                if tgt == src:
                    continue
                cases.append(dict(case_id=f'{src}__{tgt}__{w["window_id"]}', window_id=w['window_id'], source=src, target=tgt,
                                  target_heldout=tgt in ROBOTS_HELDOUT, kind=w['kind'],
                                  primary=bool(src == PRIMARY_SOURCE and w['kind'] in ('unseen_motion_listed', 'unseen_motion_mined', 'unseen_body'))))
    return cases


def build_rootbias(manifest):
    windows = []
    for g in manifest['xwbc']['groups']:
        if g['split'] == 0 and g['source_dataset'] == 'LAFAN' and re.match(ROOTBIAS_CLIP_RULE, g['clip']):
            windows.append(_window(g, 'centre', 'rootbias_train'))
    cases = [dict(case_id=f'human_smpl__{t}__{w["window_id"]}', window_id=w['window_id'], source='human_smpl', target=t,
                  target_heldout=True, kind='rootbias_train', primary=False) for w in windows for t in ROBOTS_HELDOUT]
    return windows, cases


def freeze(obj, path, force=False):
    body = dict(obj); body.pop('sha256', None)
    sha = canonical_sha(body)
    body['sha256'] = sha
    path = Path(path)
    if path.exists() and not force:
        old = json.loads(path.read_text())
        if old.get('sha256') != sha:
            raise ValueError(f'{path} already frozen with sha {old.get("sha256")}; rebuilt content differs ({sha}). '
                             'A frozen case list is never overwritten silently (use --force only before any training).')
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    path.with_suffix('.sha256').write_text(sha + '\n')
    return body


def load_cases(path=CASES_FILE, verify=True):
    d = json.loads(Path(path).read_text())
    if verify:
        body = dict(d); sha = body.pop('sha256')
        if canonical_sha(body) != sha:
            raise ValueError(f'{path}: content does not match its recorded sha256')
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(EVAL_DIR)); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    man = load_manifest()
    windows, log = build_windows(man)
    cases = build_cases(windows)
    rule = dict(MINE, window=WINDOW, exclude_datasets=list(EXCLUDE_DATASETS), nonflat_regex=NONFLAT_RE.pattern,
                nonfoot='human joints whose labeller limb group is not L_foot/R_foot; hard|(gap<0) on trusted finite rows',
                order='greedy by (-fraction, start); listed + guard windows block overlap; split-1 clips only')
    obj = dict(schema=SCHEMA, window_frames=WINDOW, fps=30, manifest=str(MANIFEST),
               manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), sources=SOURCES, primary_source=PRIMARY_SOURCE,
               robots=ROBOTS, heldout=ROBOTS_HELDOUT, mining_rule=rule, mining_log=log, windows=windows, cases=cases,
               counts=dict(windows=len(windows), cases=len(cases), primary_cases=sum(c['primary'] for c in cases),
                           by_kind={k: sum(w['kind'] == k for w in windows) for k in sorted({w['kind'] for w in windows})}),
               modes=['sensor_on', 'sensor_off'])
    out = Path(a.out)
    frozen = freeze(obj, out / CASES_FILE.name, a.force)
    rw, rc = build_rootbias(man)
    rb = freeze(dict(schema=SCHEMA + '_rootbias', window_frames=WINDOW, rule='centre window of every split-0 LAFAN clip matching '
                     + ROOTBIAS_CLIP_RULE + '; human_smpl source -> 4 heldout robots; used only for the A0 heldout root bias',
                     windows=rw, cases=rc, manifest_sha256=obj['manifest_sha256']), out / ROOTBIAS_FILE.name, a.force)
    print(json.dumps(dict(cases_sha256=frozen['sha256'], counts=frozen['counts'], rootbias_sha256=rb['sha256'],
                          rootbias_windows=len(rw)), indent=1))
    for w in frozen['windows']:
        print(f"  {w['kind']:22s} {w['window_id']:70s} nonfoot={w['human_nonfoot_contact_frac']:.2f}")


if __name__ == '__main__':
    main()
