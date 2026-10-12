"""End-to-end strictness check of the DECODED dataset with the factory's own gauge (umrf.geometry.RobotGeometry,
imported read-only from /home/arenalabs/umr_factory_code, no bytecode written there).

For every live robot entry: decode qpos from the shards, undo the floor shift, and re-measure every frame that the
dataset marks valid (flags == 0) exactly as the factory verifies its output: ground penetration, self penetration
(MuJoCo collision geoms, staged exclusions), object penetration (UMR's posed object, HOI), and the strict contact
band [-0.5 mm, 1 mm] for every labelled ground / object contact region.  The SOURCE pose of the same frame is measured
too: `introduced_by_decode` (violation after decode but not in the source) must be 0; source-side violations
(e.g. outputs made before the factory switched to the visual+collision gauge) are reported separately.

    PYTHONDONTWRITEBYTECODE=1 python scripts/strict_recheck.py --root data [--workers 8] [--max-frames 0]
"""
import sys
sys.dont_write_bytecode = True
import argparse          # noqa: E402
import json              # noqa: E402
import multiprocessing as mp   # noqa: E402
import time              # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/home/arenalabs/umr_factory_code')

PEN, CT = 5e-4, 1e-3


def check_entry(job):
    root, c, b, every = job
    import mujoco
    from surface_retargeting.dataset.dataset import Reader
    from surface_retargeting.strict import geometry as G
    from surface_retargeting.robots import surface as SF
    from surface_retargeting.strict.objects import load_umr_hoi_object
    rd = Reader(root); cat = rd.cat
    clip, robot = cat.clips[c], cat.bodies[b]
    out = dict(clip=clip, robot=robot)
    t0 = time.time()
    segs = rd.seg[(c, b)]
    e0 = segs[0][4]
    st = json.load(open(f'/home/arenalabs/umr_factory_data/robots/{robot}/staging.json'))
    m = mujoco.MjModel.from_xml_path(st['staged_mjcf'])
    surf = SF.load_surface(robot)
    floor = float(cat.e_floor[e0])
    obj = None
    # provenance -> UMR raw output (frame ids, object dir, scale)
    from surface_retargeting.dataset.catalog import _shard_entries
    man = json.load(open(Path(root) / 'manifest.json'))
    rel = cat.shards[int(cat.e_shard[e0])]
    ent = next(x for x in _shard_entries(Path(root), rel, man['shards'][rel]['sha256'])['entries']
               if x['clip'] == clip and x['body'] == robot)
    stride = int(ent['stride'])
    if ent['has_object']:
        R = np.load(ent['prov']['umr_out'], allow_pickle=True)
        obj, _ = load_umr_hoi_object(str(R['source_object_dir']), R['frame_ids'], float(R['smpl_scale'][0]))
    if hasattr(G, 'build_model'):                         # factory >= 2026-10-04 22:47: union visual+collision gauge
        m = G.build_model(st['staged_mjcf'], obj)
    geo = G.RobotGeometry(m, surf, floor=floor, obj=obj)
    zsrc = np.load(ent['src']['path']) if Path(ent['src']['path']).exists() else None
    n_chk = n_valid = 0
    cats = ('ground_pen', 'self_pen', 'object_pen', 'ground_band', 'object_band')
    viol = {k: 0 for k in cats}; src_viol = {k: 0 for k in cats}; introduced = {k: 0 for k in cats}
    worst = dict(ground_pen_mm=0.0, self_pen_mm=0.0, object_pen_mm=0.0, band_excess_mm=0.0)

    def judge(mm, gc, oc):
        v = dict(ground_pen=mm['ground_min'] < -PEN, self_pen=mm['self_min'] < -PEN, object_pen=mm['obj_min'] < -PEN)
        g = mm['ground_reg_min']
        v['ground_band'] = bool((gc & ~((g >= -PEN) & (g <= CT))).any())
        v['object_band'] = False
        if oc is not None and obj is not None:
            o = mm['obj_reg_min']
            v['object_band'] = bool((oc & ~((o >= -PEN) & (o <= CT))).any())
        return v
    from surface_retargeting.dataset.catalog import src_rows
    for k0, n, sh, r0, e in segs:
        rows = src_rows(rd.entry_meta(c, b, k0), k0, n)
        dec = rd.robot(c, b, k0, n)
        q = dec['qpos'].astype(np.float64)
        q[:, 2] += floor                                  # back to the factory measurement frame
        valid = np.flatnonzero(dec['flags'] == 0)
        n_valid += len(valid)
        for i in valid[::every]:
            t = int(rows[i])                              # factory npz row == UMR output frame index
            gc = dec['contact'][i]; oc = dec['object_contact'][i] if 'object_contact' in dec else None
            mm = geo.measure(q[i], t)
            vd = judge(mm, gc, oc)
            n_chk += 1
            vs = None
            if zsrc is not None:
                qs = zsrc['qpos'][t].astype(np.float64); qs[2] += floor
                vs = judge(geo.measure(qs, t), gc, oc)
            for k in cats:
                viol[k] += int(vd[k])
                if vs is not None:
                    src_viol[k] += int(vs[k]); introduced[k] += int(vd[k] and not vs[k])
            worst['ground_pen_mm'] = max(worst['ground_pen_mm'], -mm['ground_min'] * 1e3)
            if np.isfinite(mm['self_min']):
                worst['self_pen_mm'] = max(worst['self_pen_mm'], -mm['self_min'] * 1e3)
            if np.isfinite(mm['obj_min']):
                worst['object_pen_mm'] = max(worst['object_pen_mm'], -mm['obj_min'] * 1e3)
            g = mm['ground_reg_min']
            if gc.any():
                worst['band_excess_mm'] = max(worst['band_excess_mm'], float(np.max(np.where(gc, np.maximum(g - CT, -PEN - g), -1))) * 1e3)
    out.update(gauge=ent.get('gauge', 'visual'), frames_valid=n_valid, frames_checked=n_chk, violations_decoded=viol,
               violations_source_same_frames=src_viol, introduced_by_decode=introduced,
               worst={k: round(v, 4) for k, v in worst.items()}, seconds=round(time.time() - t0, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/home/arenalabs/umr_dataset/data'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--every', type=int, default=1, help='check every n-th valid frame')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    from surface_retargeting.dataset.catalog import Catalog
    cat = Catalog(a.root)
    jobs = sorted({(str(a.root), int(c), int(b), a.every) for c, b in zip(cat.e_clip, cat.e_body) if b != 0})
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs))) as pool:
        res = pool.map(check_entry, jobs, chunksize=1)
    for r in res:
        print(json.dumps(r))
    agg = lambda f: {k: sum(r[f][k] for r in res) for k in res[0][f]}
    summ = dict(entries=len(res), frames_checked=sum(r['frames_checked'] for r in res),
                factory_gauge_now='union visual+collision' if 'build_model' in open('/home/arenalabs/umr_factory_code/umrf/geometry.py').read() else 'visual',
                violations_decoded=agg('violations_decoded'), violations_source_same_frames=agg('violations_source_same_frames'),
                introduced_by_decode=agg('introduced_by_decode'),
                worst={k: max(r['worst'][k] for r in res) for k in res[0]['worst']})
    print('SUMMARY', json.dumps(summ))
    out = a.out or (a.root / 'qa' / 'strict_recheck.json')
    out.parent.mkdir(exist_ok=True)
    json.dump(dict(summary=summ, entries=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
