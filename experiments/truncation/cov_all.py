"""Prefix-coverage for every available HiPHI SMPL-X clip (resumable, single process).

Output: cov_all.jsonl (one row per clip) + em/<motion_id>.npz (earliest-match arrays) for plotting.
"""
import json, os, sys, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

NS = [5, 10, 15, 20, 25, 30, 40, 45, 60]
if METRIC == 'v1':
    CFG_F = {'f3': (0.03, 0.05), 'f5': (0.05, 0.05), 'f8': (0.08, 0.05)}
    CFG_W = {'w5': (0.05, 0.05), 'w8': (0.08, 0.05)}
else:  # v2 keys keep the v1 names: f5/w5 = primary (max-joint 12 cm), f3 = 10 cm, f8/w8 = 15 cm
    CFG_F = {'f3': (0.10, 0.05), 'f5': (0.12, 0.05), 'f8': (0.15, 0.05)}
    CFG_W = {'w5': (0.12, 0.05), 'w8': (0.15, 0.05)}
L = HZ  # 1 s windows
QS = [0.8, 0.9, 0.95]
CTRL_N = 20


def cov_curve_frame(em, T, n):
    if n >= T:
        return None
    return float((em[n:] < n).mean())


def cov_curve_win(emw, n):
    # remainder windows start at >= n ; prefix windows must end inside prefix (start <= n-L)
    nw = len(emw)
    if n >= nw or n - L < 0:
        return None
    return float((emw[n:] <= n - L).mean())


def nstar(fun, T, q):
    for s in range(1, int(T / HZ) + 1):
        c = fun(s * HZ)
        if c is None:
            return T / HZ
        if c >= q:
            return float(s)
    return T / HZ


def rescue_extra(em, n, T, pad):
    """seconds of remainder kept if we keep prefix n + every uncovered remainder frame dilated by pad."""
    if n >= T:
        return 0.0
    unc = em[n:] >= n
    if not unc.any():
        return 0.0
    k = np.ones(2 * pad + 1)
    keep = np.convolve(unc.astype(float), k, 'same') > 0
    return float(keep.sum() / HZ)


def nn_stats(D, H, th):
    Dm = np.where(H < th, D, np.inf).min(1)
    Dm = np.where(np.isfinite(Dm), Dm, 1.0)  # no height match -> 100 cm
    return [round(float(np.percentile(Dm, q)) * 100, 2) for q in (50, 90, 99)]


def ctrl_cov(FA, hA, FB, hB, n, thr):
    """coverage of A's remainder (>= n) by B's first n frames, frame level (+ NN pose dist cm p50/p90/p99)."""
    if len(FA) <= n:
        return None, None
    D, H = dist_mats_m(FA[n:], hA[n:], FB[:n], hB[:n])
    return float(((D < thr[0]) & (H < thr[1])).any(1).mean()), nn_stats(D, H, thr[1]) + nn_stats(D, H, 9.0)


def main():
    labs = load_labels()
    idx = clip_index()
    out_path = OUT + '/cov_all%s.jsonl' % SUFFIX
    os.makedirs(OUT + '/em' + SUFFIX, exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)['motion_id'])
            except Exception:
                pass
    mids = sorted(m for m in idx if m in labs)
    # partner pools for controls
    by_lu, by_actor = {}, {}
    for m in mids:
        r = labs[m]
        by_lu.setdefault(r['frame_lu'], []).append(m)
        by_actor.setdefault(r['actor_id'], []).append(m)
    rng = random.Random(0)
    grp = {m: group_of(labs[m]) for m in mids}
    cache = {}

    def feats(m):
        if m not in cache:
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
            J, F, h, fl = load_clip(idx[m])
            cache[m] = (J, F, h)
        return cache[m]

    fo = open(out_path, 'a')
    t0 = time.time()
    todo = [m for m in mids if m not in done]
    if os.environ.get('LIMIT'):
        todo = todo[:int(os.environ['LIMIT'])]
    print('todo', len(todo), 'done', len(done), flush=True)
    for ii, m in enumerate(todo):
        r = labs[m]
        info = idx[m]
        try:
            J, F, h = feats(m)
            T = len(F)
            D, H = dist_mats_m(F, h, F, h)
            row = dict(motion_id=m, group=group_of(r), primary=r['primary'], frame_lu=r['frame_lu'],
                       actor=r['actor_id'], obj=r['object_categories'], dur_s=T / HZ,
                       dur_meta_s=float(r['duration_sec']), src=info['src'],
                       repeatedly=('repeatedly' in r['text_annotation'].lower()), text=r['text_annotation'])
            ems = {}
            for n in (15, 20, 30):
                if n * HZ < T:
                    row['nn_self_%d' % n] = nn_stats(D[n * HZ:, :n * HZ], H[n * HZ:, :n * HZ], 0.05) + \
                        nn_stats(D[n * HZ:, :n * HZ], H[n * HZ:, :n * HZ], 9.0)
            for k, (tp, th) in CFG_F.items():
                em = earliest_match((D < tp) & (H < th))
                ems[k] = em.astype(np.int32)
                for n in NS:
                    row['cov_%s_%d' % (k, n)] = cov_curve_frame(em, T, n * HZ)
                    # spec variant: second half covered by first N
                    if n * HZ < T // 2:
                        row['covhalf_%s_%d' % (k, n)] = float((em[T // 2:] < n * HZ).mean())
                for q in QS:
                    row['nstar_%s_%d' % (k, int(q * 100))] = nstar(lambda nn: cov_curve_frame(em, T, nn), T, q)
                for n in (15, 20, 30):
                    row['rescue_%s_%d' % (k, n)] = rescue_extra(em, n * HZ, T, int(0.5 * HZ))
            Dw, Hw = window_mats(D, H, L)
            for k, (tp, th) in CFG_W.items():
                emw = earliest_match((Dw < tp) & (Hw < th))
                ems[k] = emw.astype(np.int32)
                for n in NS:
                    row['cov_%s_%d' % (k, n)] = cov_curve_win(emw, n * HZ)
                for q in QS:
                    row['nstar_%s_%d' % (k, int(q * 100))] = nstar(lambda nn: cov_curve_win(emw, nn), T, q)
                for n in (15, 20, 30):
                    # rescue on window level: uncovered windows -> frames [i, i+L)
                    nn = n * HZ
                    if nn >= len(emw):
                        row['rescue_%s_%d' % (k, n)] = 0.0
                        continue
                    unc = np.zeros(T, bool)
                    for i in np.flatnonzero(emw[nn:] > nn - L) + nn:
                        unc[i:i + L] = True
                    row['rescue_%s_%d' % (k, n)] = float(unc[nn:].sum() / HZ)
            del D, H, Dw, Hw
            # controls (frame level f5, prefix CTRL_N)
            n = CTRL_N * HZ
            thr = CFG_F['f5']
            same_lu = [x for x in by_lu.get(r['frame_lu'], []) if x != m and labs[x]['actor_id'] != r['actor_id']]
            same_act = [x for x in by_actor.get(r['actor_id'], []) if x != m and grp[x] != row['group']]
            diff_grp = [x for x in mids if grp[x] != row['group'] and labs[x]['actor_id'] != r['actor_id']]
            for name, pool in (('ctrl_samelu_otheractor', same_lu), ('ctrl_sameactor_othergroup', same_act),
                               ('ctrl_othergroup_otheractor', diff_grp)):
                if pool:
                    x = rng.choice(pool)
                    _, Fb, hb = feats(x)
                    row[name], row[name + '_nn'] = ctrl_cov(F, h, Fb, hb, n, thr)
                    row[name + '_id'] = x
            # self control: remainder covered by a *later* window of equal length (last 20 s)
            if T > 2 * n:
                em = ems['f5']
                Dl, Hl = dist_mats_m(F[n:T - n], h[n:T - n], F[T - n:], h[T - n:])
                row['ctrl_self_last20'] = float(((Dl < thr[0]) & (Hl < thr[1])).any(1).mean())
                row['cov_f5_20_same_span'] = float((em[n:T - n] < n).mean())
            # objects
            try:
                meta = json.load(open(info['dir'] + '/metadata.json'))
            except Exception:
                meta = {}
            if meta.get('objects'):
                try:
                    ev = object_events(info, meta, J)
                    row.update(cycle_stats(ev, [15, 20, 30]))
                except Exception as e:
                    row['obj_err'] = repr(e)[:200]
            np.savez_compressed(OUT + '/em%s/%s.npz' % (SUFFIX, m), **ems)
            fo.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
            fo.flush()
        except Exception as e:
            print('ERR', m, repr(e)[:300], flush=True)
        if ii % 100 == 0:
            el = time.time() - t0
            msg = '%d/%d  %.0fs elapsed' % (ii, len(todo), el)
            print(msg, flush=True)
            with open(OUT + '/cov_all%s.status' % SUFFIX, 'w') as fs:
                fs.write(msg + '\n')
    fo.close()
    print('DONE', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
