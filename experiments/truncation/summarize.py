"""Aggregate cov_all.jsonl -> group tables, control arms, low-coverage list, hours projection (JSON + CSV)."""
import json, os, sys, collections, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

labs = load_labels()
rows = [json.loads(l) for l in open(OUT + '/cov_all%s.jsonl' % SUFFIX)]
for r in rows:
    r['group'] = group_of(labs[r['motion_id']])
by = collections.defaultdict(list)
for r in rows:
    by[r['group']].append(r)


def vals(rs, k):
    return np.array([r[k] for r in rs if r.get(k) is not None], float)


def med(rs, k, q=50):
    v = vals(rs, k)
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def frac_ge(rs, k, t):
    v = vals(rs, k)
    return round(float((v >= t).mean()), 3) if len(v) else None


S = {'n_clips': len(rows), 'hours_measured': round(sum(r['dur_s'] for r in rows) / 3600, 2)}
# ---------- group table ----------
G = []
order = sorted(by, key=lambda g: -len(by[g]))
for g in order + ['ALL']:
    rs = rows if g == 'ALL' else by[g]
    hoi = [r for r in rs if 'cyc_in_20' in r]
    d = dict(group=g, zh=GROUP_ZH.get(g, ''), n=len(rs), hours=round(sum(r['dur_s'] for r in rs) / 3600, 2))
    for c in ('f5', 'f8', 'f3', 'w5'):
        for n in (15, 20, 30):
            d['med_cov_%s_%d' % (c, n)] = med(rs, 'cov_%s_%d' % (c, n))
            d['p10_cov_%s_%d' % (c, n)] = med(rs, 'cov_%s_%d' % (c, n), 10)
            d['share90_%s_%d' % (c, n)] = frac_ge(rs, 'cov_%s_%d' % (c, n), 0.9)
    for k in ('nstar_f5_90', 'nstar_f5_95', 'nstar_w5_90', 'nstar_f8_95'):
        d[k + '_med'] = med(rs, k); d[k + '_p90'] = med(rs, k, 90)
    for n in (15, 20, 30):
        d['rescue_f5_%d_mean_s' % n] = round(float(vals(rs, 'rescue_f5_%d' % n).mean()), 2)
        d['cyc_in_%d' % n] = round(float(np.mean([r['cyc_in_%d' % n] for r in hoi])), 3) if hoi else None
    d['n_hoi'] = len(hoi)
    d['first_cycle_end_med'] = med(hoi, 'first_cycle_end_s')
    d['first_cycle_end_p90'] = med(hoi, 'first_cycle_end_s', 90)
    d['no_cycle_detected'] = round(float(np.mean([r.get('first_cycle_end_s') is None for r in hoi])), 3) if hoi else None
    d['move_period_med'] = med(hoi, 'move_period_s')
    # controls
    for k in ('cov_f5_20_same_span', 'ctrl_self_last20', 'ctrl_samelu_otheractor', 'ctrl_sameactor_othergroup',
              'ctrl_othergroup_otheractor'):
        d[k + '_med'] = med(rs, k)
        d[k + '_mean'] = round(float(vals(rs, k).mean()), 3) if len(vals(rs, k)) else None
    a = np.array([r['nn_self_20'] for r in rs if r.get('nn_self_20')])
    d['nn_self_20_p50_p90'] = a[:, :2].mean(0).round(2).tolist() if len(a) else None
    for k in ('ctrl_samelu_otheractor_nn', 'ctrl_sameactor_othergroup_nn', 'ctrl_othergroup_otheractor_nn'):
        a = np.array([r[k] for r in rs if r.get(k)])
        d[k + '_posep50'] = round(float(np.median(a[:, 3])), 2) if len(a) else None
    a = np.array([r['nn_self_20'] for r in rs if r.get('nn_self_20')])
    d['nn_self_20_posep50'] = round(float(np.median(a[:, 3])), 2) if len(a) else None
    G.append(d)
S['groups'] = G

# ---------- low coverage list ----------
low = sorted([r for r in rows if r.get('cov_f5_20') is not None], key=lambda r: r['cov_f5_20'])[:25]
S['lowest_cov_f5_20'] = [dict(motion_id=r['motion_id'], group=r['group'], obj=r['obj'], dur=r['dur_s'],
                              cov15=r.get('cov_f5_15'), cov20=r['cov_f5_20'], cov30=r.get('cov_f5_30'),
                              cov20_8cm=r.get('cov_f8_20'), nstar90=r.get('nstar_f5_90'),
                              text=r['text']) for r in low]
# distribution of cov_f5_20
v = vals(rows, 'cov_f5_20')
S['cov_f5_20_hist'] = {'<0.5': float((v < 0.5).mean()), '0.5-0.8': float(((v >= 0.5) & (v < 0.8)).mean()),
                       '0.8-0.9': float(((v >= 0.8) & (v < 0.9)).mean()), '0.9-0.95': float(((v >= 0.9) & (v < 0.95)).mean()),
                       '>=0.95': float((v >= 0.95).mean())}


# ---------- hours projection ----------
def kept_s(r, rule):
    T = r['dur_s']
    kind, a = rule
    if kind == 'fixed':
        return min(T, a)
    if kind == 'nstar':
        key, floor_s = a
        return min(T, max(floor_s, r.get(key) or T))
    if kind == 'rescue':
        n, cfg = a
        return min(T, n + (r.get('rescue_%s_%d' % (cfg, n)) or 0.0))
    raise ValueError


RULES = {'fixed15': ('fixed', 15), 'fixed20': ('fixed', 20), 'fixed30': ('fixed', 30),
         'sat_f5_90_min15': ('nstar', ('nstar_f5_90', 15)), 'sat_f5_95_min15': ('nstar', ('nstar_f5_95', 15)),
         'sat_w5_90_min15': ('nstar', ('nstar_w5_90', 15)), 'sat_f5_95_min20': ('nstar', ('nstar_f5_95', 20)),
         'pre20_rescue_f5': ('rescue', (20, 'f5')), 'pre15_rescue_f5': ('rescue', (15, 'f5')),
         'pre20_rescue_w5': ('rescue', (20, 'w5')), 'pre30_rescue_f5': ('rescue', (30, 'f5'))}
frac = {}
for name, rule in RULES.items():
    frac[name] = {}
    for g in by:
        rs = by[g]
        frac[name][g] = sum(kept_s(r, rule) for r in rs) / sum(r['dur_s'] for r in rs)
    frac[name]['ALL'] = sum(kept_s(r, rule) for r in rows) / sum(r['dur_s'] for r in rows)
S['kept_fraction_measured'] = {k: {g: round(x, 3) for g, x in v.items()} for k, v in frac.items()}

# full dataset hours per group
full = collections.defaultdict(list)
for r in labs.values():
    full[group_of(r)].append(float(r['duration_sec']))
full_h = {g: sum(v) / 3600 for g, v in full.items()}
MIN_MEASURED = 40  # groups with fewer measured clips use the fallback
proj = {}
for name, rule in RULES.items():
    out = {}
    for fallback in ('same_as_ALL', 'fixed30', 'no_trunc'):
        tot = 0.0
        for g, durs in full.items():
            if rule[0] == 'fixed':
                tot += sum(min(d, rule[1]) for d in durs) / 3600
            elif g in by and len(by[g]) >= MIN_MEASURED:
                tot += full_h[g] * frac[name][g]
            elif fallback == 'same_as_ALL':
                tot += full_h[g] * frac[name]['ALL']
            elif fallback == 'fixed30':
                tot += sum(min(d, 30) for d in durs) / 3600
            else:
                tot += full_h[g]
        out[fallback] = round(tot, 1)
    proj[name] = out
S['projection_hours_of_308.7'] = proj
S['full_hours_by_group'] = {g: round(h, 2) for g, h in sorted(full_h.items(), key=lambda x: -x[1])}
S['measured_clips_by_group'] = {g: len(by[g]) for g in by}

json.dump(S, open(OUT + '/summary%s.json' % SUFFIX, 'w'), indent=1, ensure_ascii=False)
with open(OUT + '/group_table%s.csv' % SUFFIX, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(G[0].keys()))
    w.writeheader()
    for d in G:
        w.writerow(d)
print(json.dumps({k: S[k] for k in ('n_clips', 'hours_measured', 'cov_f5_20_hist', 'projection_hours_of_308.7')}, indent=1))
for d in G:
    print(d['group'], d['n'], d['hours'], 'f5', [d['med_cov_f5_%d' % n] for n in (15, 20, 30)],
          'p10', [d['p10_cov_f5_%d' % n] for n in (15, 20, 30)], 'w5', [d['med_cov_w5_%d' % n] for n in (15, 20, 30)],
          'f8', d['med_cov_f8_20'], 'n*90/95', d['nstar_f5_90_med'], d['nstar_f5_95_med'], d['nstar_f5_95_p90'],
          'cyc', [d['cyc_in_%d' % n] for n in (15, 20, 30)], 'fc', d['first_cycle_end_med'], d['first_cycle_end_p90'],
          'nocyc', d['no_cycle_detected'])
print()
for d in G:
    print(d['group'], 'ctrl same_span/self_last/samelu/sameactor/othergrp',
          d['cov_f5_20_same_span_med'], d['ctrl_self_last20_med'], d['ctrl_samelu_otheractor_mean'],
          d['ctrl_sameactor_othergroup_mean'], d['ctrl_othergroup_otheractor_mean'],
          'NNpose p50 self/samelu/sameact/other', d['nn_self_20_posep50'], d['ctrl_samelu_otheractor_nn_posep50'],
          d['ctrl_sameactor_othergroup_nn_posep50'], d['ctrl_othergroup_otheractor_nn_posep50'])
