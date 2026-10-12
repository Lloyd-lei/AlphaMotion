"""Random-sample audit: how much of UniML3D reloads as grounded LBS point clouds."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, os, json, csv, random, subprocess, traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from surface_retargeting.ingest.uniml3d.reload import *

HERE = os.path.dirname(os.path.abspath(__file__)); R = f'{SRCFG.factory_root}/uniml3d_reload/remote'  # data, not code
CACHE = os.path.join(HERE, '../sample_cache'); os.makedirs(CACHE, exist_ok=True)
HF = 'https://huggingface.co/datasets'
N_OBJ, N_MIX, MAXF = int(sys.argv[1]), int(sys.argv[2]), 300
random.seed(20261004)
L = dict(json.load(open(f'{R}/file_list.json')))
exp = {p.split('/')[-1][:-4]: p for p in L if p.startswith('export/objaverse/motions/')}
feat = sorted(p.split('/')[-1][:-8] for p in L if p.startswith('features/objaverse/motions/'))
featm = sorted(p.split('/')[-1][:-8] for p in L if p.startswith('features/mixamo/motions/'))
cats = {r['object_type']: r['category'] for r in csv.DictReader(open(f'{R}/skeletons.csv')) if r['dataset'] == 'objaverse'}
trims = {}
for ln in open(f'{R}/export/objaverse/clip_trims.txt'):
    s = ln.split()
    if len(s) >= 2 and not ln.startswith('#'):
        try: trims[s[0]] = int(s[1])
        except ValueError: pass
objs = random.sample(feat, N_OBJ); mixs = random.sample(featm, N_MIX)


def get(repo, path):
    dst = os.path.join(CACHE, repo.split('/')[1], path)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(['curl', '-sfL', '--max-time', '180', '-o', dst + '.part', f'{HF}/{repo}/resolve/main/{path}'], check=True)
        os.rename(dst + '.part', dst)
    return dst


jobs = [('objaverse', c, exp[c], f"glb/{c.split('-', 1)[0]}.glb") for c in objs] + \
       [('mixamo', c, f'export/mixamo/motions/{c}.npz', None) for c in mixs]


def fetch(j):
    try:
        e = get('Linzhan/UniML3D', j[2])
        g = get('Linzhan/Objaverse-XL-Rigged-Animated', j[3]) if j[3] else os.path.join(R, 'glb/Y_Bot.glb')
        return j, e, g, None
    except Exception as ex:
        return j, None, None, repr(ex)


def lbs_chunk(P0, Wp, G, Gr):
    out = []
    for s in range(0, len(G), 50):
        out.append(lbs(P0, Wp, G[s:s + 50], Gr))
    return np.concatenate(out)


rows = []
with ThreadPoolExecutor(1 if os.environ.get('SERIAL') else 8) as ex:
    for j, e, g, err in ex.map(fetch, jobs):
        src, clip = j[0], j[1]
        r = {'src': src, 'clip': clip, 'category': cats.get(clip.split('-', 1)[0], 'bipedal' if src == 'mixamo' else '?')}
        if err:
            r['status'] = 'download_fail:' + err[:80]; rows.append(r); continue
        try:
            m = glb_skinned_mesh(g); d = np.load(e, allow_pickle=True)
            names = list(d['names']); tr = trims.get(clip, 0)
            G, Gr = export_fk(d, tr); G = G[:MAXF]
            common = [n for n in m['names'] if n in names]
            r['V'] = int(len(m['V'])); r['J'] = len(names); r['T'] = int(d['anim_local_pos'].shape[0] - tr)
            if len(common) < 1:
                r['status'] = 'no_joint_name_match'; rows.append(r); continue
            pb = np.array([np.linalg.inv(m['ibm'][m['names'].index(n)])[:3, 3] for n in common])
            pe = np.array([Gr[names.index(n)][:3, 3] for n in common])
            if len(common) >= 3 and np.ptp(pb, 0).max() > 0:
                sc, Rr, t = umeyama(pb, pe)
            else:
                sc, Rr, t = 1.0, np.eye(3), pe.mean(0) - pb.mean(0)
            Ve = (sc * (Rr @ m['V'].T)).T + t
            H = np.ptp(Ve, 0).max()
            res = np.linalg.norm((sc * (Rr @ pb.T)).T + t - pe, axis=1).max() / H
            We, lost, moved = weights_to_export(m, names)
            P0, Wp = sample_surface(Ve, m['F'], We, n_cand=12000, n=2048)
            P = lbs_chunk(P0, Wp, G, Gr)
            # rigidity sanity: rest LBS reproduces bind points
            Prest = lbs(P0, Wp, Gr[None], Gr)[0]
            hgt = np.ptp(Ve[:, 1]); rmin = Ve[:, 1].min()
            pfmin = P[:, :, 1].min(1)
            eps_rel = 0.006 * hgt                          # ~1 cm for a 1.7 m body
            h = P[:, :, 1]; con = h < eps_rel
            vel = np.gradient(P, axis=0) * 30.0
            vt = np.linalg.norm(vel[..., [0, 2]], axis=-1) / hgt   # body heights per s
            frc = con.any(1)
            med = np.array([np.median(vt[i][con[i]]) if con[i].any() else 0 for i in range(len(P))])
            # root travel: horizontal displacement of the point-cloud centroid over the clip, in body heights
            cen = P.mean(1); travel = np.linalg.norm(cen[-1, [0, 2]] - cen[0, [0, 2]]) / hgt
            path = np.linalg.norm(np.diff(cen[:, [0, 2]], axis=0), axis=1).sum() / hgt
            part = Wp.argmax(1)
            ps = part_slip(con, vt, part, 0.06)
            r.update(status='ok', weight_moved_to_ancestor=moved, part_slip_frames_0p06H=float((ps).sum() / max(frc.sum(), 1)), bind_res_rel=float(res), bind_scale=float(sc), weight_lost=lost,
                     rest_lbs_err_rel=float(np.abs(Prest - P0).max() / H), extent_units=float(H), height_units=float(hgt),
                     rest_min_y_rel=float(rmin / hgt), pfmin_p2_rel=float(np.percentile(pfmin, 2) / hgt),
                     pfmin_med_rel=float(np.median(pfmin) / hgt), pen_max_rel=float(max(0, -pfmin.min()) / hgt),
                     contact_frames=float(frc.mean()),
                     slip_frames_0p06H=float((frc & (med > 0.06)).sum() / max(frc.sum(), 1)),  # ~0.10 m/s at 1.7 m
                     travel_H=float(travel), path_H=float(path), nframes_used=int(len(P)))
        except Exception as ex2:
            r['status'] = 'error:' + repr(ex2)[:120]
        rows.append(r)
        print(r.get('status'), clip, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ('bind_res_rel', 'rest_min_y_rel', 'pfmin_med_rel', 'contact_frames', 'slip_frames_0p06H', 'travel_H', 'extent_units')}, flush=True)
json.dump(rows, open(os.path.join(HERE, 'out/audit_rows.json'), 'w'), indent=1)
