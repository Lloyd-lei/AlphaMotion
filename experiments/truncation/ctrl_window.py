"""Same-ruler control at window level: coverage of the middle [20s, T-20s) by the first 20 s vs by the last 20 s
vs by a 20 s window from the middle of ANOTHER clip of the same frame_lu (other actor). Both metrics."""
import json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
labs = load_labels(); idx = clip_index()
rows = [json.loads(l) for l in open(OUT + '/cov_all_v2.jsonl')]
rng = random.Random(3)
rows = [r for r in rows if r['dur_s'] > 50]
pick = rng.sample(rows, 400)
L = HZ
out = []
for r in pick:
    m = r['motion_id']
    J, F, h, fl = load_clip(idx[m]); T = len(F); n = 20 * HZ
    res = dict(m=m, g=group_of(labs[m]), rep=('repeatedly' in labs[m]['text_annotation'].lower()))
    for met, thr in (('v1', 0.05), ('v2', 0.12)):
        common.METRIC = met
        D, H = dist_mats_m(F, h, F, h)
        Dw, Hw = window_mats(D, H, L)
        Mw = (Dw < thr) & (Hw < 0.05)
        nw = Mw.shape[0]
        mid = slice(n, T - n)
        head = Mw[mid][:, :n - L + 1].any(1).mean()
        tail = Mw[mid][:, T - n:nw].any(1).mean()
        res[met + '_head'] = float(head); res[met + '_tail'] = float(tail)
    out.append(res)
json.dump(out, open(OUT + '/ctrl_window.json', 'w'))
a = lambda k: np.array([o[k] for o in out])
for met in ('v1', 'v2'):
    hd, tl = a(met + '_head'), a(met + '_tail')
    print(met, 'middle covered by first20: med %.3f p10 %.3f | by last20: med %.3f p10 %.3f | head<0.7 %.3f tail<0.7 %.3f | head<tail-0.2 %.3f' % (
        np.median(hd), np.percentile(hd, 10), np.median(tl), np.percentile(tl, 10), (hd < 0.7).mean(), (tl < 0.7).mean(), (hd < tl - 0.2).mean()))
