"""sample40 table with both rulers + per-group head-only table (v1/v2) for the report."""
import json, os, sys, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
labs = load_labels()
cv = {s: {json.loads(l)['motion_id']: json.loads(l) for l in open(OUT + '/cov_all%s.jsonl' % s)} for s in ('', '_v2')}
ht = {s: {json.loads(l)['motion_id']: json.loads(l) for l in open(OUT + '/headtail%s.jsonl' % s)} for s in ('', '_v2')}
sample = [r['motion_id'] for r in csv.DictReader(open(OUT + '/sample40.csv'))]
keys = ['motion_id', 'group', 'zh', 'obj', 'dur_s', 'repeatedly',
        'v1_cov15', 'v1_cov20', 'v1_cov30', 'v1_win20', 'v2_cov15', 'v2_cov20', 'v2_cov30', 'v2_win20',
        'v1_adaptive_head_s', 'v2_adaptive_head_s', 'onset_contact_s', 'continuous_hold',
        'release_kept_head20', 'release_kept_head20_tail8', 'cycle_in_head15', 'cycle_in_head20', 'cycle_in_head30', 'text']
out = []
for m in sample:
    a, b = cv[''][m], cv['_v2'].get(m, {})
    h1 = ht[''][m]
    r = dict(motion_id=m, group=group_of(labs[m]), zh=GROUP_ZH.get(group_of(labs[m]), ''), obj=a['obj'], dur_s=a['dur_s'],
             repeatedly=a['repeatedly'],
             v1_cov15=a.get('cov_f5_15'), v1_cov20=a.get('cov_f5_20'), v1_cov30=a.get('cov_f5_30'), v1_win20=a.get('cov_w5_20'),
             v2_cov15=b.get('cov_f5_15'), v2_cov20=b.get('cov_f5_20'), v2_cov30=b.get('cov_f5_30'), v2_win20=b.get('cov_w5_20'),
             v1_adaptive_head_s=max(15, a.get('nstar_w5_90') or a['dur_s']), v2_adaptive_head_s=max(15, b.get('nstar_w5_90') or a['dur_s']),
             onset_contact_s=h1.get('onset_contact_s'), continuous_hold=h1.get('continuous_contact'),
             release_kept_head20=h1['fix_20_0'].get('release'), release_kept_head20_tail8=h1['fix_20_8'].get('release'),
             cycle_in_head15=h1['fix_15_0'].get('cycle'), cycle_in_head20=h1['fix_20_0'].get('cycle'), cycle_in_head30=h1['fix_30_0'].get('cycle'),
             text=a['text'])
    out.append(r)
with open(OUT + '/sample40.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in out:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
# print compact
for r in sorted(out, key=lambda r: r['v1_cov20'] if r['v1_cov20'] is not None else 1):
    print('%-34s %-11s %-9s %5.0fs v1 %3.0f/%3.0f/%3.0f w%3.0f | v2 %3.0f/%3.0f/%3.0f w%3.0f | head* %4.0f/%4.0f | rel20 %s rel20+8 %s | cyc15/20/30 %s/%s/%s | %s' % (
        r['motion_id'], r['group'], r['obj'], r['dur_s'], 100 * r['v1_cov15'], 100 * r['v1_cov20'], 100 * r['v1_cov30'], 100 * r['v1_win20'],
        100 * (r['v2_cov15'] or 0), 100 * (r['v2_cov20'] or 0), 100 * (r['v2_cov30'] or 0), 100 * (r['v2_win20'] or 0),
        r['v1_adaptive_head_s'], r['v2_adaptive_head_s'], str(r['release_kept_head20'])[0], str(r['release_kept_head20_tail8'])[0],
        str(r['cycle_in_head15'])[0], str(r['cycle_in_head20'])[0], str(r['cycle_in_head30'])[0], r['text'][:70]))
# per-group summary for sample
g = collections.defaultdict(list)
for r in out:
    g[r['group']].append(r)
print()
for k, rs in g.items():
    print(k, len(rs), 'v1 cov15/20/30 mean %.2f/%.2f/%.2f' % tuple(np.mean([r['v1_cov%d' % n] for r in rs]) for n in (15, 20, 30)),
          'v2 %.2f/%.2f/%.2f' % tuple(np.mean([r['v2_cov%d' % n] or 0 for r in rs]) for n in (15, 20, 30)))
