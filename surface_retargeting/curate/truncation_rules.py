"""Compare truncation rules from headtail.jsonl: kept hours vs. content lost; project to full 308.7 h."""
import json, os, sys, collections, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

labs = load_labels()
rows = [json.loads(l) for l in open(OUT + '/headtail%s.jsonl' % SUFFIX)]


def has_cycle(r):
    ce = r.get('contact_eps') or []; me = r.get('move_eps') or []
    Te = int(round(r['dur_s'] * HZ))
    ct = np.zeros(Te + 1, bool)
    for s, e in ce:
        ct[s:e] = True
    # a short complete cycle (<= 15 s) exists somewhere in the full clip
    return bool([1 for s, e in ce if s > 0.2 * HZ and e < Te - 0.2 * HZ and e - s <= 15 * HZ] +
                [1 for s, e in me if e < Te and ct[s:e].any() and e - s <= 15 * HZ])


for r in rows:
    if 'contact_eps' in r:
        r['_cyc_possible'] = has_cycle(r)
for r in rows:
    r['group'] = group_of(labs[r['motion_id']])
by = collections.defaultdict(list)
for r in rows:
    by[r['group']].append(r)
rule_names = [k for k in rows[0] if isinstance(rows[0][k], dict)]


def stats(rs, rule):
    T = np.array([r['dur_s'] for r in rs]); kept = np.array([r[rule]['kept'] for r in rs])
    f5 = np.array([r[rule]['f5'] for r in rs]); w5 = np.array([r[rule]['w5'] for r in rs])
    hoi = [r for r in rs if 'grasp' in r[rule]]
    hoic = [r for r in hoi if r.get('_cyc_possible')]
    def rate(key):
        v = [r[rule][key] for r in hoi if r[rule].get(key) is not None]
        return round(float(np.mean(v)), 3) if v else None
    return dict(n=len(rs), kept_frac=round(float(kept.sum() / T.sum()), 3),
                f5_med=round(float(np.median(f5)), 3), f5_p10=round(float(np.percentile(f5, 10)), 3),
                w5_med=round(float(np.median(w5)), 3), w5_p10=round(float(np.percentile(w5, 10)), 3),
                share_w5_ge90=round(float((w5 >= 0.9).mean()), 3), share_w5_lt70=round(float((w5 < 0.7).mean()), 3),
                share_f5_lt70=round(float((f5 < 0.7).mean()), 3),
                lost_w5_frac_of_rest=round(float(((1 - w5) * (T - kept)).sum() / T.sum()), 4),
                grasp=rate('grasp'), release=rate('release'), cycle=rate('cycle'), contact_any=rate('contact_any'),
                cycle_if_possible=round(float(np.mean([r[rule]['cycle'] for r in hoic])), 3) if hoic else None,
                n_hoi=len(hoi), n_cyc_possible=len(hoic))


res = {}
for rule in rule_names:
    res[rule] = {'ALL': stats(rows, rule)}
    for g in by:
        res[rule][g] = stats(by[g], rule)

# projection to full dataset
full = collections.defaultdict(list)
for r in labs.values():
    full[group_of(r)].append(float(r['duration_sec']))
full_h = {g: sum(v) / 3600 for g, v in full.items()}
TOT = sum(full_h.values())
proj = {}
for rule in rule_names:
    p = {}
    for fb in ('measured_ALL_ratio', 'no_trunc'):
        tot = 0.0
        for g, durs in full.items():
            if rule.startswith('fix_'):
                _, N, K = rule.split('_'); s = int(N) + int(K)
                tot += sum(min(d, s) for d in durs) / 3600
            elif g in by and len(by[g]) >= 40:
                tot += full_h[g] * res[rule][g]['kept_frac']
            elif fb == 'measured_ALL_ratio':
                tot += full_h[g] * res[rule]['ALL']['kept_frac']
            else:
                tot += full_h[g]
        p[fb] = round(tot, 1)
    proj[rule] = p
json.dump(dict(rules=res, projection_h=proj, total_h=round(TOT, 2)), open(OUT + '/rules_summary%s.json' % SUFFIX, 'w'), indent=1)
with open(OUT + '/rules_table%s.csv' % SUFFIX, 'w', newline='') as f:
    w = csv.writer(f)
    keys = list(res[rule_names[0]]['ALL'].keys())
    w.writerow(['rule', 'group'] + keys + ['proj_h_measuredratio', 'proj_h_unmeasured_untruncated'])
    for rule in rule_names:
        for g, d in res[rule].items():
            w.writerow([rule, g] + [d[k] for k in keys] +
                       ([proj[rule]['measured_ALL_ratio'], proj[rule]['no_trunc']] if g == 'ALL' else ['', '']))
print('total h', round(TOT, 1), 'measured clips', len(rows))
print('%-22s %5s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %7s %7s' % (
    'rule', 'keep', 'f5med', 'f5p10', 'w5med', 'w5p10', 'w5>=.9', 'w5<.7', 'grasp', 'releas', 'cycle', 'lost', 'proj_h', 'proj_h2'))
for rule in rule_names:
    d = res[rule]['ALL']
    print('%-22s %5.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6s %6s %6s %6.4f %7.1f %7.1f' % (
        rule, d['kept_frac'], d['f5_med'], d['f5_p10'], d['w5_med'], d['w5_p10'], d['share_w5_ge90'], d['share_w5_lt70'],
        d['grasp'], d['release'], d['cycle_if_possible'], d['lost_w5_frac_of_rest'], proj[rule]['measured_ALL_ratio'], proj[rule]['no_trunc']))

SHOW = ['fix_15_0', 'fix_20_0', 'fix_30_0', 'fix_20_8', 'oa_20_8', 'sat_w5_90_8', 'satcap_w5_90_8']
print()
print('per-group (keep / w5 p10 / share w5<0.7 / release / cycle_if_possible)')
for g in sorted(by, key=lambda g: -len(by[g])):
    print('%-12s n=%4d ' % (g, len(by[g])) + ' | '.join('%s %.2f/%.2f/%.2f/%s/%s' % (
        rule.replace('_0', '').replace('satcap_w5_90', 'scap').replace('sat_w5_90', 'sat'), res[rule][g]['kept_frac'],
        res[rule][g]['w5_p10'], res[rule][g]['share_w5_lt70'], res[rule][g]['release'], res[rule][g]['cycle_if_possible'])
        for rule in SHOW))
print('n_hoi', res['fix_20_0']['ALL']['n_hoi'], 'n_cycle_possible', res['fix_20_0']['ALL']['n_cyc_possible'])
cont = [r.get('continuous_contact') for r in rows if 'contact_eps' in r]
print('continuous-contact share', round(float(np.mean([bool(x) for x in cont])), 3))

# ---------------- final combined rules: non-"repeatedly" clips kept whole ----------------
def is_rep(mid):
    return 'repeatedly' in labs[mid]['text_annotation'].lower()


for r in rows:
    r['_rep'] = is_rep(r['motion_id'])
    for base in ('sat_w5_90_8', 'satcap_w5_90_8', 'fix_20_8', 'fix_30_8', 'oa_20_8', 'sat_f5_95_8'):
        if r['_rep']:
            r['final_' + base] = dict(r[base])
        else:
            r['final_' + base] = dict(kept=r['dur_s'], f5=1.0, w5=1.0, grasp=r[base].get('grasp') if 'grasp' in r[base] else None,
                                      release=True if r[base].get('release') is not None else None, cycle=True,
                                      contact_any=r[base].get('contact_any'))
            if 'grasp' not in r[base]:
                for k in ('grasp', 'release', 'cycle', 'contact_any'):
                    r['final_' + base].pop(k, None)
FINAL = ['final_' + b for b in ('sat_w5_90_8', 'satcap_w5_90_8', 'fix_20_8', 'fix_30_8', 'oa_20_8', 'sat_f5_95_8')]
rep_rows = [r for r in rows if r['_rep']]
nonrep_rows = [r for r in rows if not r['_rep']]
print()
print('measured: repeated clips %d (%.1f h), non-repeated clips %d (%.1f h)' % (
    len(rep_rows), sum(r['dur_s'] for r in rep_rows) / 3600, len(nonrep_rows), sum(r['dur_s'] for r in nonrep_rows) / 3600))
for base in ('fix_20_0', 'sat_w5_90_8'):
    if nonrep_rows:
        d = stats(nonrep_rows, base)
        print('non-repeated clips under %s: keep %.2f  w5 med %.2f p10 %.2f  share<0.7 %.2f' % (base, d['kept_frac'], d['w5_med'], d['w5_p10'], d['share_w5_lt70']))
    d = stats(rep_rows, base)
    print('repeated clips under %s: keep %.2f  w5 med %.2f p10 %.2f  share<0.7 %.2f' % (base, d['kept_frac'], d['w5_med'], d['w5_p10'], d['share_w5_lt70']))
by_rep = collections.defaultdict(list)
for r in rep_rows:
    by_rep[r['group']].append(r)
full_rep = collections.defaultdict(float); full_non = 0.0
for r in labs.values():
    if 'repeatedly' in r['text_annotation'].lower():
        full_rep[group_of(r)] += float(r['duration_sec']) / 3600
    else:
        full_non += float(r['duration_sec']) / 3600
final_proj = {}
for rule in FINAL + ['fix_15_0', 'fix_20_0', 'fix_30_0']:
    base = rule.replace('final_', '')
    allr = stats(rep_rows, base)['kept_frac']
    tot = full_non if rule.startswith('final_') else 0.0
    used = {}
    for g, hh in full_rep.items():
        if base.startswith('fix_'):
            continue
        fr = stats(by_rep[g], base)['kept_frac'] if len(by_rep[g]) >= 40 else allr
        used[g] = round(fr, 3)
        tot += hh * fr
    if base.startswith('fix_'):
        _, N, K = base.split('_'); s = int(N) + int(K)
        tot = sum(min(float(r['duration_sec']), s) if ('repeatedly' in r['text_annotation'].lower() or not rule.startswith('final_'))
                  else float(r['duration_sec']) for r in labs.values()) / 3600
    d = stats(rows, rule)
    final_proj[rule] = dict(proj_h=round(tot, 1), keep_measured=d['kept_frac'], w5_p10=d['w5_p10'], share_w5_lt70=d['share_w5_lt70'],
                            lost=d['lost_w5_frac_of_rest'], release=d['release'], cycle=d['cycle_if_possible'], ratio_used=used)
    print('%-24s proj %.1f h of %.1f (save %.0f%%)  measured keep %.3f  w5 p10 %.3f  <0.7 %.3f  lost %.4f  release %s  cycle %s' % (
        rule, tot, TOT, 100 * (1 - tot / TOT), d['kept_frac'], d['w5_p10'], d['share_w5_lt70'], d['lost_w5_frac_of_rest'], d['release'], d['cycle_if_possible']))
json.dump(dict(final=final_proj, full_nonrep_h=round(full_non, 2), full_rep_h={g: round(v, 2) for g, v in full_rep.items()}),
          open(OUT + '/final_rules%s.json' % SUFFIX, 'w'), indent=1)
