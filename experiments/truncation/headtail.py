"""Evaluate truncation rules per clip: keep [0, head) + [T-tail, T); measure coverage of the dropped middle
(frame-level 5cm and 1s-window 5cm) and which interaction events survive.

Rules:
  fix_N_K   : head = N s
  oa_N_K    : onset-aligned, head = onset + N s  (onset = first object-contact start for HOI, else motion onset)
  sat_<cfg>_K    : head = max(15, N*) where N* = smallest head whose coverage of the rest >= q (from cov_all)
  satcap_<cfg>_K : same, capped at 30 s
Events (HOI): contact episodes (hands<8cm | pelvis<15cm | feet<6cm of object surface, gap-merge 0.5 s, min 0.3 s);
  grasp = episode start (>0.2 s), release = episode end (< T-0.2 s), cycle = an episode (contact or object motion)
  that lies entirely inside one kept segment.
Output: headtail.jsonl
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

L = HZ


def motion_onset(J):
    """first time the body leaves its initial idle: root-relative joint speed or root speed > thr for >= 0.5 s."""
    F, _ = canon_features(J)
    js = np.r_[0, np.sqrt(((np.diff(F, axis=0)) ** 2).sum(-1)).mean(-1) * HZ]
    rs = np.r_[0, np.linalg.norm(np.diff(J[:, 0, [0, 2]], axis=0), axis=1) * HZ]
    act = (js > 0.15) | (rs > 0.3)
    for s, e in runs(act, 2, int(0.5 * HZ)):
        return int(s)
    return 0


def main():
    labs = load_labels(); idx = clip_index()
    cov = {}
    for l in open(OUT + '/cov_all%s.jsonl' % SUFFIX):
        r = json.loads(l)
        cov[r['motion_id']] = r
    out_path = OUT + '/headtail%s.jsonl' % SUFFIX
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)['motion_id'] for l in open(out_path)}
    mids = sorted(m for m in idx if m in labs and m in cov and m not in done)
    if os.environ.get('LIMIT'):
        mids = mids[:int(os.environ['LIMIT'])]
    fo = open(out_path, 'a'); t0 = time.time()
    for ii, m in enumerate(mids):
        try:
            info = idx[m]; r = labs[m]; c = cov[m]
            J, F, h, fl = load_clip(info)
            T = len(F)
            D, H = dist_mats_m(F, h, F, h)
            M = (D < PRIMARY_THR) & (H < 0.05)
            Dw, Hw = window_mats(D, H, L)
            Mw = (Dw < PRIMARY_THR) & (Hw < 0.05)
            del D, H, Dw, Hw
            nw = Mw.shape[0]
            row = dict(motion_id=m, group=group_of(r), dur_s=T / HZ)
            ev = None
            try:
                meta = json.load(open(info['dir'] + '/metadata.json'))
            except Exception:
                meta = {}
            if meta.get('objects'):
                ev = object_events(info, meta, J)
            on_m = motion_onset(J)
            on_c = ev['contact_ep'][0][0] if (ev is not None and ev['contact_ep']) else None
            onset = on_c if on_c is not None else on_m
            row['onset_motion_s'] = on_m / HZ
            row['onset_contact_s'] = on_c / HZ if on_c is not None else None
            if ev is not None:
                row['contact_eps'] = [(int(s), int(e)) for s, e in ev['contact_ep']]
                row['move_eps'] = [(int(s), int(e)) for s, e in ev['move_ep']]
                row['continuous_contact'] = bool(len(ev['contact_ep']) == 1 and
                                                 (ev['contact_ep'][0][1] - ev['contact_ep'][0][0]) > 0.7 * ev['T'])

            def evaluate(name, n, k):
                n = int(min(n, T)); k = int(k)
                if n + k >= T:
                    row[name] = dict(kept=T / HZ, f5=1.0, w5=1.0)
                else:
                    keep = np.zeros(T, bool); keep[:n] = True
                    if k:
                        keep[T - k:] = True
                    f5 = float(M[n:T - k][:, keep].any(1).mean())
                    keepw = np.zeros(nw, bool); keepw[:max(0, n - L + 1)] = True
                    if k >= L:
                        keepw[T - k:T - L + 1] = True
                    stop = min(T - k, nw)
                    w5 = float(Mw[n:stop][:, keepw].any(1).mean()) if stop > n else 1.0
                    row[name] = dict(kept=(n + k) / HZ, f5=round(f5, 4), w5=round(w5, 4))
                if ev is not None:
                    Te = ev['T']
                    segs = [(0, n)] + ([(Te - k, Te)] if k else [])
                    inside = lambda t: any(a <= t < b for a, b in segs)
                    full_in = lambda s, e: any(a <= s and e <= b for a, b in segs)
                    ce, me = ev['contact_ep'], ev['move_ep']
                    grasps = [s for s, e in ce if s > 0.2 * HZ]
                    rels = [e for s, e in ce if e < Te - 0.2 * HZ]
                    row[name]['grasp'] = any(inside(t) for t in grasps) if grasps else None
                    row[name]['release'] = any(inside(t) for t in rels) if rels else None
                    ctm = np.zeros(Te + 1, bool)
                    for s, e in ce:
                        ctm[s:e] = True
                    row[name]['cycle'] = bool([1 for s, e in ce if s > 0.2 * HZ and e < Te - 0.2 * HZ and full_in(s, e)] +
                                              [1 for s, e in me if e < Te and full_in(s, e) and ctm[s:e].any()])
                    # share of all contact time that is kept
                    ct = np.zeros(Te, bool)
                    for s, e in ce:
                        ct[s:e] = True
                    kp = np.zeros(Te, bool); kp[:n] = True
                    if k:
                        kp[Te - k:] = True
                    row[name]['contact_kept_s'] = float((ct & kp).sum() / HZ)
                    row[name]['contact_any'] = bool((ct & kp).any()) if ct.any() else None

            for N in (15, 20, 30):
                for K in (0, 5, 8, 10):
                    evaluate('fix_%d_%d' % (N, K), N * HZ, K * HZ)
                for K in (0, 8):
                    evaluate('oa_%d_%d' % (N, K), onset + N * HZ, K * HZ)
            for cfg in ('f5_90', 'f5_95', 'w5_90'):
                ns = c.get('nstar_' + cfg) or T / HZ
                for K in (0, 8):
                    evaluate('sat_%s_%d' % (cfg, K), max(15, ns) * HZ, K * HZ)
                    evaluate('satcap_%s_%d' % (cfg, K), min(30, max(15, ns)) * HZ, K * HZ)
            del M, Mw
            fo.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n'); fo.flush()
        except Exception as e:
            print('ERR', m, repr(e)[:300], flush=True)
        if ii % 200 == 0:
            msg = 'headtail %d/%d %.0fs' % (ii, len(mids), time.time() - t0)
            print(msg, flush=True)
            open(OUT + '/headtail%s.status' % SUFFIX, 'w').write(msg + '\n')
    print('DONE', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
