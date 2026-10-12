"""Collision decomposition + MJCF finalisation for converted GRAIL clips (Stream B).

Per unique canonical visual mesh (smplx/grail/_meshes/<digest>.obj): UMR's own build_collision_cache
(exact convex components if the mesh already splits into closed convex pieces, else CoACD via obj2mjcf with UMR's
GRAIL defaults threshold 0.03 / max 32 hulls / 200 MCTS iters / resolution 2000) into smplx/grail/_collision_cache/<digest>.
QA per digest (qa.json): hull excess (hull-surface samples -> distance to visual mesh: how far collision sits outside
the true surface) and hull deficit (visual-surface samples outside every hull: how far the visual surface pokes out).
Then every converted clip using that digest gets grail_object_collision_<i>.obj hard links, grail_object.xml (same
format as UMR prepare_grail_object_mjcf.write_mjcf), a MuJoCo compile check, and metadata.json (asset_version 3) last.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
OUT = Path('/home/arenalabs/umr_factory_data/smplx/grail')
MESH = OUT / '_meshes'
COLL = OUT / '_collision_cache'
ROWS = OUT / '_convert_rows.jsonl'
ROWS_V2 = OUT / '_convert_rows_v2.jsonl'
ROWS_V3 = OUT / '_convert_rows_v3.jsonl'
COACD = dict(threshold=0.03, max_convex_hull=32, mcts_iterations=200, resolution=2000)


def hull_qa(visual_obj, parts, n=4000):
    import trimesh
    vis = trimesh.load(visual_obj, force='mesh', process=False)
    hulls = [trimesh.load(p, force='mesh', process=True) for p in parts]
    hs = trimesh.util.concatenate(hulls)
    ph = hs.sample(n, seed=0)
    _, d_ex, _ = trimesh.proximity.closest_point(vis, ph)
    pv = vis.sample(n, seed=1)
    out = np.full(len(pv), np.inf)
    for hm in hulls:
        try:
            ch = hm.convex_hull
            nrm = ch.face_normals
            off = np.einsum('ij,ij->i', nrm, ch.triangles[:, 0])
            sd = (pv @ nrm.T - off).max(1)  # >0 outside (lower bound of euclidean distance)
            out = np.minimum(out, np.maximum(sd, 0.0))
        except Exception:
            pass
    return {'hull_excess_mm_p50_p95_max': [round(float(np.percentile(d_ex, q) * 1000), 2) for q in (50, 95)] + [round(float(d_ex.max() * 1000), 2)],
            'visual_outside_hulls_mm_p95_max': [round(float(np.percentile(out, 95) * 1000), 2), round(float(out.max() * 1000), 2)],
            'visual_outside_hulls_gt0p5mm_frac': round(float((out > 0.0005).mean()), 4)}


def decompose(dg):
    t0 = time.time()
    try:
        from obj2mjcf.cli import CoacdArgs
        from prepare_grail_object_mjcf import build_collision_cache
        vis = MESH / f'{dg}.obj'
        parts, method = build_collision_cache(vis, COLL / dg, CoacdArgs(**COACD))
        qa = {'digest': dg, 'method': method, 'parts': len(parts), 'sec': round(time.time() - t0, 2), 'coacd_args': COACD}
        try:
            qa.update(hull_qa(vis, parts))
        except Exception as e:  # noqa: BLE001
            qa['qa_error'] = repr(e)[:200]
        (COLL / dg / 'qa.json').write_text(json.dumps(qa))
        return qa
    except Exception as e:  # noqa: BLE001
        err = {'digest': dg, 'error': repr(e)[:400], 'sec': round(time.time() - t0, 2)}
        (COLL / f'{dg}.error.json').write_text(json.dumps(err))
        return err


def finalize(row):
    import mujoco
    from prepare_grail_object_mjcf import link_collision_parts, sorted_collision_parts, write_mjcf
    od = Path(row['paths']['object_mjcf'])
    dg = row['object']['mesh_digest']
    qa = json.loads((COLL / dg / 'qa.json').read_text())
    parts = sorted_collision_parts(COLL / dg)
    if len(parts) != qa['parts']:
        raise RuntimeError(f'collision cache mismatch {dg}')
    linked = link_collision_parts(parts, od)
    xml = od / 'grail_object.xml'
    write_mjcf(xml, linked)
    m = mujoco.MjModel.from_xml_path(str(xml))
    ncol = int(sum(int(m.geom_group[i]) == 3 for i in range(m.ngeom)))
    if ncol != len(linked):
        raise RuntimeError(f'MJCF collision geom mismatch {ncol} != {len(linked)}')
    meta = {
        'asset_version': 3, 'sequence_key': row['clip'],
        'source_usd': str(OUT / row['category'] / 'object_usd' / f"{row['clip']}.usd"),
        'geometry_digest': dg, 'decomposition_method': qa['method'], 'collision_parts': len(linked),
        'frames': row['frames'], 'texture': None, 'coacd_args': {**COACD}, 'status': 'converted',
        'umrf': {'obj_scale_source': row['object']['obj_scale_source'], 'obj_scale_effective': row['object']['obj_scale_effective'],
                 'floor_dz_m': row['floor_dz_m'], 'shift_policy': row['shift_policy'],
                 'converter_version': row.get('converter_version', 1), 'terrain_xy_offset_m': row.get('terrain_xy_offset_m', [0.0, 0.0]),
                 'hull_qa': {k: qa[k] for k in qa if k.startswith(('hull_', 'visual_'))}},
    }
    tmp = od / '.metadata.json.tmp'
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, od / 'metadata.json')


def main():
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--poll', type=float, default=60)
    ap.add_argument('--max-hours', type=float, default=10)
    ap.add_argument('--stop-file', default=str(OUT / '_converter_done'))
    a = ap.parse_args()
    COLL.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + a.max_hours * 3600
    ctx = mp.get_context('spawn')
    finalized = set()
    log = open(OUT / '_collision_log.jsonl', 'a')
    with ctx.Pool(a.workers) as pool:
        inflight = {}
        while True:
            rows = []
            for rp in (ROWS, ROWS_V2, ROWS_V3):  # highest converter_version wins in the per-clip dedupe below
                if not rp.exists():
                    continue
                for ln in rp.read_text().splitlines():
                    try:
                        r = json.loads(ln)
                    except Exception:
                        continue
                    if r.get('status') == 'ok':
                        rows.append(r)
            best = {}
            for r in rows:  # highest converter_version per clip; ties -> last
                k_ = (r['category'], r['clip'])
                if k_ not in best or r.get('converter_version', 1) >= best[k_].get('converter_version', 1):
                    best[k_] = r
            rows = list(best.values())
            order = []
            seen = set()
            for r in rows:
                dg = r['object']['mesh_digest']
                if dg not in seen:
                    seen.add(dg); order.append(dg)
            for dg in order:
                if dg in inflight or (COLL / dg / 'qa.json').exists() or (COLL / f'{dg}.error.json').exists():
                    continue
                if len(inflight) >= a.workers * 3:
                    break
                inflight[dg] = pool.apply_async(decompose, (dg,))
            for dg in list(inflight):
                if inflight[dg].ready():
                    res = inflight.pop(dg).get()
                    log.write(json.dumps(res) + '\n'); log.flush()
            nfin = 0
            for r in rows:
                key = (r['category'], r['clip'])
                if key in finalized:
                    continue
                od = Path(r['paths']['object_mjcf'])
                dg = r['object']['mesh_digest']
                if (od / 'metadata.json').exists():
                    try:
                        mv = json.loads((od / 'metadata.json').read_text()).get('umrf', {}).get('converter_version', 1)
                    except Exception:
                        mv = -1
                    if mv >= r.get('converter_version', 1):
                        finalized.add(key); continue
                if (COLL / dg / 'qa.json').exists():
                    try:
                        finalize(r); finalized.add(key); nfin += 1
                    except Exception as e:  # noqa: BLE001
                        log.write(json.dumps({'clip': r['clip'], 'finalize_error': repr(e)[:300]}) + '\n'); log.flush()
                        finalized.add(key)
            ndone = sum(1 for _ in COLL.glob('*/qa.json')) + sum(1 for _ in COLL.glob('*.error.json'))
            print(f'{time.strftime("%H:%M:%S")} digests total {len(order)} done {ndone} inflight {len(inflight)} '
                  f'clips finalized {len(finalized)}/{len(rows)} (+{nfin})', flush=True)
            if time.time() > t_end:
                break
            if not a.loop and not inflight and len(finalized) >= len(rows):
                break
            if a.loop and Path(a.stop_file).exists() and not inflight and len(finalized) >= len(rows) and ndone >= len(order):
                break
            time.sleep(a.poll if not inflight else 10)
    print('collision exit', flush=True)


if __name__ == '__main__':
    main()
