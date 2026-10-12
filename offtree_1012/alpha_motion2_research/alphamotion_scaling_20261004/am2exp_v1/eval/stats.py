"""E1 statistics and the frozen decision rule (final_plan.md "Statistics").

Unit of analysis: the case (same case_id in both runs); clusters: windows (window_id).  A window-level
difference is the mean of its paired case differences; tests run over windows, never over cases or frames.

decide(arm, parent, ...) implements, for factor X (arm) vs its parent:
  dCSE_GT <= -max(2*seed_floor, 0.03) on human-source PRIMARY cases (mean of window means),
  window permutation (sign-flip) p < 0.05 (one-sided, towards improvement),
  >= 2/3 of target bodies with the same (negative) sign of their mean paired difference,
  no guard regression: PenP95 +1 cm, FloatRate +3 pts, SelfPen +2 pts, pos_err_mm +3 %,
  twin shows no effect (twin vs parent fails the effect criteria), lloyd agrees after videos (external flag).
seed floor: |SF_s1 - SF| per metric on the same primary cases (mean of window means); before SF_s1 exists the floor
is unknown and the rule uses the 0.03 absolute minimum and flags 'seed_floor_missing'.
"""
from __future__ import annotations
import itertools
import math
from collections import defaultdict

import numpy as np

EFFECT_MIN = 0.03
P_MAX = 0.05
BODY_FRAC = 2.0 / 3.0
GUARDS = {'PenP95_cm': ('abs', 1.0), 'FloatRate': ('abs', 0.03), 'SelfPen': ('abs', 0.02), 'pos_err_mm': ('rel', 0.03)}
EXACT_MAX_CLUSTERS = 16
MC_SAMPLES = 20000


def index(records):
    return {r['case_id']: r for r in records}


def select(records, *, primary=None, source=None, heldout=None, kinds=None):
    out = []
    for r in records:
        if primary is not None and bool(r.get('primary')) != primary:
            continue
        if source is not None and r['source'] != source:
            continue
        if heldout is not None and bool(r['heldout']) != heldout:
            continue
        if kinds is not None and r['kind'] not in kinds:
            continue
        out.append(r)
    return out


def paired(a_records, b_records, metric):
    """[(case_id, window_id, target, a - b)] over cases present and finite in both runs."""
    A, B = index(a_records), index(b_records)
    out = []
    for cid in sorted(set(A) & set(B)):
        x, y = A[cid].get(metric), B[cid].get(metric)
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        out.append((cid, A[cid]['window_id'], A[cid]['target'], float(x) - float(y)))
    return out


def cluster_means(diffs):
    by = defaultdict(list)
    for _, w, _, d in diffs:
        by[w].append(d)
    keys = sorted(by)
    return keys, np.array([np.mean(by[k]) for k in keys])


def sign_test(x, alternative='less'):
    """Exact binomial sign test on cluster means (zeros dropped)."""
    x = np.asarray(x); x = x[x != 0]
    n = len(x)
    if n == 0:
        return 1.0
    k = int((x < 0).sum()) if alternative == 'less' else int((x > 0).sum())
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def permutation_test(x, alternative='less', seed=0):
    """Sign-flip permutation test of the mean of cluster means (exact for <= 16 clusters)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return 1.0
    obs = x.mean()
    if n <= EXACT_MAX_CLUSTERS:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(MC_SAMPLES, n))
    null = (signs * np.abs(x)[None]).mean(1)
    if alternative == 'less':
        return float((null <= obs + 1e-12).mean())
    if alternative == 'greater':
        return float((null >= obs - 1e-12).mean())
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def per_body_signs(diffs):
    by = defaultdict(list)
    for _, _, t, d in diffs:
        by[t].append(d)
    means = {t: float(np.mean(v)) for t, v in sorted(by.items())}
    return dict(means=means, negative=sum(v < 0 for v in means.values()), positive=sum(v > 0 for v in means.values()),
                n=len(means))


def compare(a_records, b_records, metric, *, alternative='less', subset=None):
    """Paired a - b on `subset` (callable filter on records of a)."""
    if subset is not None:
        a_records = [r for r in a_records if subset(r)]
    d = paired(a_records, b_records, metric)
    keys, cm = cluster_means(d)
    return dict(metric=metric, n_cases=len(d), n_windows=len(keys), mean_of_window_means=float(cm.mean()) if len(cm) else float('nan'),
                case_mean=float(np.mean([x[3] for x in d])) if d else float('nan'),
                windows=dict(zip(keys, cm.tolist())), sign_p=sign_test(cm, alternative), perm_p=permutation_test(cm, alternative),
                bodies=per_body_signs(d))


def seed_floor(sf_records, sf_s1_records, metric, subset=None):
    if sf_records is None or sf_s1_records is None:
        return None
    c = compare(sf_s1_records, sf_records, metric, alternative='two-sided', subset=subset)
    return abs(c['mean_of_window_means'])


def primary_human(r):
    return bool(r.get('primary')) and r['source'] == 'human_smpl'


def guard_check(arm_records, parent_records, subset=primary_human):
    out = {}
    for metric, (kind, tol) in GUARDS.items():
        c = compare(arm_records, parent_records, metric, alternative='greater', subset=subset)
        delta = c['mean_of_window_means']
        if kind == 'rel':
            base = np.mean([r[metric] for r in parent_records if subset(r) and r.get(metric) is not None and np.isfinite(r[metric])])
            limit = tol * base
        else:
            limit = tol
        out[metric] = dict(delta=delta, limit=float(limit), regressed=bool(np.isfinite(delta) and delta > limit))
    return out


def effect(arm_records, parent_records, *, floor=None, metric='CSE_GT', subset=primary_human, body_frac=BODY_FRAC):
    thr = -max(2 * floor, EFFECT_MIN) if floor is not None else -EFFECT_MIN
    c = compare(arm_records, parent_records, metric, alternative='less', subset=subset)
    b = c['bodies']
    body_ok = b['n'] > 0 and b['negative'] / b['n'] >= body_frac - 1e-9
    passed = bool(np.isfinite(c['mean_of_window_means']) and c['mean_of_window_means'] <= thr and c['perm_p'] < P_MAX and body_ok)
    return dict(threshold=thr, comparison=c, body_same_sign_ok=body_ok, effect=passed)


def decide(arm_records, parent_records, *, sf_records=None, sf_s1_records=None, twin_records=None, lloyd_video_ok=None,
           metric='CSE_GT', subset=primary_human):
    """Frozen decision rule.  Returns a dict with every criterion and verdict in {'pass','fail','pending'}."""
    floor = seed_floor(sf_records, sf_s1_records, metric, subset)
    eff = effect(arm_records, parent_records, floor=floor, metric=metric, subset=subset)
    guards = guard_check(arm_records, parent_records, subset)
    twin = None
    if twin_records is not None:
        tw = effect(twin_records, parent_records, floor=floor, metric=metric, subset=subset)
        twin = dict(twin_effect=tw['effect'], twin_delta=tw['comparison']['mean_of_window_means'], twin_perm_p=tw['comparison']['perm_p'])
    reasons = []
    if not eff['effect']:
        reasons.append('no effect beyond threshold / permutation / body-sign criteria')
    if any(g['regressed'] for g in guards.values()):
        reasons.append('guard regression: ' + ','.join(k for k, g in guards.items() if g['regressed']))
    if twin is not None and twin['twin_effect']:
        reasons.append('twin shows the same effect (not specific)')
    pending = []
    if floor is None:
        pending.append('seed_floor_missing (SF_s1 not available)')
    if twin is None:
        pending.append('twin not run / not provided')
    if lloyd_video_ok is None:
        pending.append('lloyd video review')
    elif not lloyd_video_ok:
        reasons.append('lloyd did not agree after videos')
    verdict = 'fail' if reasons else ('pending' if pending else 'pass')
    return dict(metric=metric, seed_floor=floor, effect=eff, guards=guards, twin=twin, reasons=reasons, pending=pending,
                verdict=verdict, rule='dCSE_GT <= -max(2*seed_floor,0.03) on human-source primary cases, window perm p<.05, '
                                    '>=2/3 bodies same sign, no guard regression (PenP95 +1cm, FloatRate +3pts, SelfPen +2pts, mm +3%), '
                                    'twin no effect, lloyd agrees after videos; then seed-1 replication >= half magnitude (separate gate)')


def replication_ok(first_delta, replication_delta):
    """Seed-1 replication must reproduce >= half the magnitude with the same sign."""
    return bool(np.sign(first_delta) == np.sign(replication_delta) and abs(replication_delta) >= 0.5 * abs(first_delta))


def summarize(records, metrics, group_keys=('heldout',), subset=None):
    """Mean of window means per group (for tables)."""
    rs = [r for r in records if subset is None or subset(r)]
    out = {}
    groups = defaultdict(list)
    for r in rs:
        groups[tuple(r[k] for k in group_keys)].append(r)
    for g, items in sorted(groups.items(), key=lambda x: repr(x[0])):
        row = {'n_cases': len(items), 'n_windows': len({r['window_id'] for r in items})}
        for m in metrics:
            by = defaultdict(list)
            for r in items:
                v = r.get(m)
                if v is not None and np.isfinite(v):
                    by[r['window_id']].append(v)
            row[m] = float(np.mean([np.mean(v) for v in by.values()])) if by else float('nan')
        out[g] = row
    return out
