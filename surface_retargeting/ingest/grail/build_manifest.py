"""GRAIL manifest builder (Stream B). Reads _convert_rows.jsonl (v1) + _convert_rows_v2.jsonl (v2 wins per clip),
the release tree and the collision metadata, and writes
  smplx/grail/manifest.jsonl          one row per clip in the release (converted, pending or dropped)
  smplx/grail/manifest.csv            flat view
  smplx/grail/manifest_summary.json   per-category clip/hour/frame totals, drop reasons, flag counts
Hours are at the true 24 fps (source pkl says 30). src_valid_* = frames whose human source passes the v2 support check
(penetration <= 10 mm and lowest non-arm body point within 20 mm of a support). Safe to run while the loops are running."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import collections
import csv
import json
import os
from pathlib import Path

OUT = Path(f'{SRCFG.data_root}/smplx/grail')
RAW = Path(f'{SRCFG.data_root}/raw/grail')
FPS = 24.0
LICENSE = ('CC BY-NC 4.0 (GRAIL, nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL); object USD per GRAIL README: '
           'RoboCasa-derived CC BY 4.0 (pickup_*), Hunyuan3D-2.1 output CC BY-NC 4.0 (sitting, stairs), procedural (curb, slope, stairs)')


def main():
    tree = json.loads((RAW / '_tree.json').read_text())
    rows = {}
    for rp in (OUT / '_convert_rows.jsonl', OUT / '_convert_rows_v2.jsonl', OUT / '_convert_rows_v3.jsonl'):
        if rp.exists():
            for ln in rp.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                k_ = (r['category'], r['clip'])
                if k_ not in rows or r.get('converter_version', 1) >= rows[k_].get('converter_version', 1):
                    rows[k_] = r  # highest converter_version wins
    cats = sorted({k.split('/')[0] for k in tree if not k.startswith('_')})
    out, summ = [], {}
    for c in cats:
        rec = {os.path.basename(p)[:-4] for p, _ in tree[f'{c}/recon']['files']}
        usd = {os.path.basename(p)[:-4] for p, _ in tree[f'{c}/object_usd']['files'] if p.endswith('.usd')}
        meta = {os.path.basename(p)[:-4] for p, _ in tree[f'{c}/meta']['files']}
        allk = sorted(rec | usd | meta)
        S = collections.Counter()
        H = collections.Counter()
        flags = collections.Counter()
        drops = collections.Counter()
        for k in allk:
            r = rows.get((c, k))
            m = {'clip': k, 'category': c, 'license': LICENSE}
            if k not in rec:
                m.update(status='dropped', reason='no SMPL-X recon in the GRAIL release (only meta/USD)')
            elif r is None:
                m.update(status='pending', reason='not downloaded/converted yet')
            elif r['status'] != 'ok':
                m.update(status='dropped', reason=r.get('reason', ''))
            else:
                od = Path(r['paths']['object_mjcf'])
                md = {}
                if (od / 'metadata.json').exists():
                    try:
                        md = json.loads((od / 'metadata.json').read_text())
                    except Exception:
                        md = {}
                q = r.get('qa', {})
                m.update({
                    'status': 'ok', 'converter_version': r.get('converter_version', 1),
                    'frames': r['frames'], 'fps': FPS, 'seconds': r['seconds'], 'hours': r['hours'],
                    'src_valid_frac': r.get('src_valid_frac'), 'src_valid_hours': r.get('src_valid_hours'),
                    'object': {**r['object'], 'collision_ready': bool(md), 'collision_parts': md.get('collision_parts'),
                               'decomposition': md.get('decomposition_method'),
                               'hull_qa': (md.get('umrf') or {}).get('hull_qa')},
                    'floor': {'policy': r.get('shift_policy'), 'dz_median_m': r.get('floor_dz_m'),
                              'dz_v1_const_mm': q.get('floor_dz_v1_const_mm', q.get('floor_dz_mm')),
                              'dz_t_range_mm': q.get('dz_t_range_mm'), 'terrain_xy_offset_m': r.get('terrain_xy_offset_m', [0.0, 0.0]),
                              'xy_registration': r.get('xy_registration')},
                    'qa': q, 'flags': r.get('flags', []), 'grail_eval': r.get('grail_eval'),
                    'interaction_frames': r.get('interaction_frames'), 'table': r.get('table'), 'paths': r['paths'],
                })
            out.append(m)
            S[m['status']] += 1
            if m['status'] == 'ok':
                H['hours'] += m['hours']
                H['frames'] += m['frames']
                if m.get('src_valid_hours') is not None and (m['converter_version'] >= 3 or c not in ('pickup_table', 'pickup_ground')):
                    H['src_valid_hours'] += m['src_valid_hours']
                    H['v2_hours'] += m['hours']
                    S['final_version'] += 1
                    if (m['src_valid_frac'] or 0) < 0.5:
                        S['src_valid_lt_50pct'] += 1
                if m['object']['collision_ready']:
                    S['collision_ready'] += 1
                for f in m['flags']:
                    flags[f.split(':')[0]] += 1
            elif m['status'] == 'dropped':
                drops[m['reason'][:90]] += 1
        summ[c] = {'clips_in_release': len(allk), 'counts': dict(S), 'hours_converted': round(H['hours'], 3),
                   'frames_converted': H['frames'], 'hours_v2': round(H['v2_hours'], 3),
                   'src_valid_hours_v2': round(H['src_valid_hours'], 3),
                   'src_valid_frac_v2': round(H['src_valid_hours'] / H['v2_hours'], 4) if H['v2_hours'] else None,
                   'drop_reasons': dict(drops), 'flags': dict(flags)}
    tot = {'hours_converted': round(sum(v['hours_converted'] for v in summ.values()), 3),
           'hours_v2': round(sum(v['hours_v2'] for v in summ.values()), 3),
           'src_valid_hours_v2': round(sum(v['src_valid_hours_v2'] for v in summ.values()), 3),
           'clips_ok': sum(v['counts'].get('ok', 0) for v in summ.values()),
           'clips_pending': sum(v['counts'].get('pending', 0) for v in summ.values()),
           'clips_dropped': sum(v['counts'].get('dropped', 0) for v in summ.values())}
    tmp = OUT / '.manifest.jsonl.tmp'
    with open(tmp, 'w') as fh:
        for m in out:
            fh.write(json.dumps(m, default=float) + '\n')
    os.replace(tmp, OUT / 'manifest.jsonl')
    cols = ['clip', 'category', 'status', 'reason', 'converter_version', 'frames', 'fps', 'seconds', 'hours',
            'src_valid_frac', 'src_valid_hours', 'object_name', 'mesh_digest', 'object_static', 'collision_ready',
            'collision_parts', 'floor_policy', 'dz_median_m', 'dz_t_range_mm', 'terrain_xy_offset_m', 'flags', 'recon', 'object_mjcf', 'license']
    tmp = OUT / '.manifest.csv.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in out:
            o, f = m.get('object', {}), m.get('floor', {})
            w.writerow([m['clip'], m['category'], m['status'], m.get('reason', ''), m.get('converter_version', ''), m.get('frames', ''),
                        m.get('fps', ''), m.get('seconds', ''), m.get('hours', ''), m.get('src_valid_frac', ''), m.get('src_valid_hours', ''),
                        o.get('name', ''), o.get('mesh_digest', ''), o.get('static', ''), o.get('collision_ready', ''), o.get('collision_parts', ''),
                        f.get('policy', ''), f.get('dz_median_m', ''), f.get('dz_t_range_mm', ''), f.get('terrain_xy_offset_m', ''),
                        '|'.join(m.get('flags', [])), (m.get('paths') or {}).get('recon', ''), (m.get('paths') or {}).get('object_mjcf', ''),
                        m['license']])
    os.replace(tmp, OUT / 'manifest.csv')
    (OUT / 'manifest_summary.json').write_text(json.dumps({'total': tot, 'per_category': summ, 'fps': FPS,
                                                           'license': LICENSE}, indent=1))
    print(json.dumps(tot))
    for c, v in summ.items():
        print(c, v['counts'], 'h', v['hours_converted'], 'v2 h', v['hours_v2'], 'valid h', v['src_valid_hours_v2'], v['src_valid_frac_v2'])


if __name__ == '__main__':
    main()
