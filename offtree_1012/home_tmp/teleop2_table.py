import json, collections
rows=json.load(open('/home/arenalabs/tmp/teleop2/scores.json'))
by=collections.defaultdict(dict)
for r in rows: by[r['motion']][r['tag']]=r
tags=['native_w1','gw_w1','native_w15','gw_w15']
print(f"{'motion':24s} " + ' '.join(f"{t:>11s}" for t in tags) + "   (survival_frac / mpjpe_local mm / fall episodes)")
for m,d in sorted(by.items()):
    cells=[]
    for t in tags:
        r=d.get(t); cells.append(f"{r['survival_frac']:.2f}/{r['mpjpe_l']:.0f}/{r['n_fall_episodes']}" if r else '-')
    print(f"{m:24s} " + ' '.join(f"{c:>11s}" for c in cells))
import numpy as np
for t in tags:
    rs=[r for r in rows if r['tag']==t]
    if rs: print(f"{t:12s} mean survival {np.mean([r['survival_frac'] for r in rs]):.2f}  clips fully survived {sum(r['survival_frac']==1.0 for r in rs)}/{len(rs)}  mpjpe_local mean {np.mean([r['mpjpe_l'] for r in rs]):.0f} mm  emd mean {np.mean([r['emd'] for r in rs if r['emd'] is not None]):.2f}")
