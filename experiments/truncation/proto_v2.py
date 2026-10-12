import json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
labs = load_labels(); idx = clip_index()
rows = {json.loads(l)['motion_id']: json.loads(l) for l in open(OUT + '/cov_all.jsonl')}
picks = ['Placing-shoulder_0010', 'Posture-sit_0005', 'Cause_motion-chuck_0032', 'Bringing-hump_0144', 'Adjusting-adjust_0001',
         'Cause_to_move_in_place-rotate_0202', 'Bringing-lug_0291', 'Cause_motion-punt_0002', 'Change_posture-sit_down_0021']
rng = random.Random(1)
allm = sorted(rows)
picks += rng.sample(allm, 50)
def feats(m):
    info = idx[m]; J, F, h, fl = load_clip(info)
    meta = json.load(open(info['dir'] + '/metadata.json'))
    O = object_state(info, meta, J)
    return J, F, h, O
CFG = {'max10': (0.10, None, None), 'max12': (0.12, None, None), 'max10_o15_t20': (0.10, 0.15, 20), 'max12_o15_t20': (0.12, 0.15, 20),
       'max15_o20_t25': (0.15, 0.20, 25)}
res = []
cache = {}
for m in picks:
    J, F, h, O = feats(m); T = len(F); n = 200
    if T <= 2 * n: continue
    Dm, H, Do, Dt = dist_mats_v2(F[n:], h[n:], F[:n], h[:n], O and (O[0][n:], O[1][n:], O[2]), O and (O[0][:n], O[1][:n], O[2]))
    # same-span control: middle covered by last 20 s
    Dm2, H2, Do2, Dt2 = dist_mats_v2(F[n:T-n], h[n:T-n], F[T-n:], h[T-n:], O and (O[0][n:T-n], O[1][n:T-n], O[2]), O and (O[0][T-n:], O[1][T-n:], O[2]))
    # cross control: same actor, other group
    r = labs[m]; g = group_of(r)
    pool = [x for x in allm if labs[x]['actor_id'] == r['actor_id'] and group_of(labs[x]) != g]
    out = dict(m=m, g=g, v1=rows[m]['cov_f5_20'])
    xm = rng.choice(pool) if pool else None
    if xm:
        _, Fx, hx, Ox = feats(xm)
        okO = O is not None and Ox is not None
        Dmx, Hx, Dox, Dtx = dist_mats_v2(F[n:], h[n:], Fx[:n], hx[:n], O and (O[0][n:], O[1][n:], O[2]) if okO else None, (Ox[0][:n], Ox[1][:n], Ox[2]) if okO else None)
    for k, (tb, to, tt) in CFG.items():
        def cov(Dm, H, Do, Dt):
            Mx = (Dm < tb) & (H < 0.05)
            if to is not None and Do is not None:
                Mx &= (Do < to) & (Dt < tt)
            return float(Mx.any(1).mean())
        out[k] = cov(Dm, H, Do, Dt)
        out[k + '_same'] = cov(Dm2, H2, Do2, Dt2)
        if xm:
            out[k + '_x'] = cov(Dmx, Hx, Dox, Dtx)
    res.append(out)
    print(m[:34].ljust(34), g[:10].ljust(10), 'v1 %.2f' % out['v1'], ' '.join('%s %.2f/%.2f/%s' % (k, out[k], out[k + '_same'], ('%.2f' % out[k + '_x']) if xm else '-') for k in CFG), flush=True)
import collections
print('MEDIANS (self20 / same-span ctrl / same-actor-other-group ctrl)')
for k in ['v1'] + list(CFG):
    a = [o[k] for o in res]
    b = [o.get(k + '_same') for o in res if o.get(k + '_same') is not None]
    c = [o.get(k + '_x') for o in res if o.get(k + '_x') is not None]
    print(k, 'med %.3f p10 %.3f' % (np.median(a), np.percentile(a, 10)), 'same %.3f' % (np.median(b) if b else -1), 'x mean %.3f' % (np.mean(c) if c else -1))
