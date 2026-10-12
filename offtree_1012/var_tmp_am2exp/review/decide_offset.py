"""Feed a pure -X cm root offset of A0 through the frozen decide() rule (eval/stats.py)."""
import json, os, sys, copy, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/var/tmp/alphamotion/am2exp/review')
import root_offset_exploit as P
ROOT = P.ROOT
from am2exp_v1.eval import stats as ST
if __name__ == '__main__':
    files = sorted(str(f) for f in P.D.glob('human_smpl__*.npz'))
    with ProcessPoolExecutor(8) as ex:
        recs = [r for r in ex.map(P.one, files, chunksize=4) if r]
    parent = [json.loads(l) for l in open(P.D / 'scores_j1.jsonl')]
    pidx = {r['case_id']: r for r in parent}
    out = {}
    for s in P.SH:
        if s == 0: continue
        arm = []
        for r in parent:
            a = copy.deepcopy(r)
            arm.append(a)
        aidx = {r['case_id']: r for r in arm}
        for r in recs:
            a = aidx[r['case_id']]
            # deltas relative to this script's own s=0 recomputation, added to the parent's stored values
            for k in ('CSE_GT', 'PenP95_cm', 'FloatRate'):
                x, y = r[s][k], r[0.0][k]
                if np.isfinite(x) and np.isfinite(y):
                    a[k] = pidx[r['case_id']][k] + (x - y)
        d = ST.decide(arm, parent, lloyd_video_ok=True)
        held = ST.decide(arm, parent, lloyd_video_ok=True, subset=lambda r: ST.primary_human(r) and r['heldout'])
        out[f'{s:+.2f}'] = dict(all=dict(verdict=d['verdict'], reasons=d['reasons'], pending=d['pending'], dCSE=d['effect']['comparison']['mean_of_window_means'],
                                         perm_p=d['effect']['comparison']['perm_p'], bodies_neg=d['effect']['comparison']['bodies']['negative'], bodies_n=d['effect']['comparison']['bodies']['n'],
                                         guards={k: (round(v['delta'], 4), v['regressed']) for k, v in d['guards'].items()}),
                                heldout_only=dict(verdict=held['verdict'], dCSE=held['effect']['comparison']['mean_of_window_means'], perm_p=held['effect']['comparison']['perm_p'],
                                                  guards={k: (round(v['delta'], 4), v['regressed']) for k, v in held['guards'].items()}))
    Path('/var/tmp/alphamotion/am2exp/review/decide_offset.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
