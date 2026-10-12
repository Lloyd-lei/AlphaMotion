#!/usr/bin/env python3
"""Rank HiPHI archive parts by hours of original (non-mirror) motions that are
human-object interaction (is_hoi) OR rich ground contact (knees/hands/hips/back on floor).
Writes logs/hiphi_part_priority.csv. Stdlib only."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import csv, re, collections, sys
R = f'{SRCFG.data_root}/raw/hiphi'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{SRCFG.data_root}/logs/hiphi_part_priority.csv'
GC_LU = {'Body_movement-kneel', 'Change_posture-kneel_down', 'Change_posture-kneel_up',
         'Change_posture-lie_down', 'Change_posture-sit_down', 'Change_posture-sit_up',
         'Change_posture-sprawl', 'Change_posture-stand_up', 'Getting_up-get_up',
         'Motion_directional-fall', 'Motion_directional-topple', 'Motion_directional-plunge',
         'Posture-kneel', 'Posture-lie', 'Posture-sprawl',
         'Self_motion-crawl', 'Self_motion-burrow', 'Self_motion-scramble'}
GC_RX = re.compile(r"\b(kneel\w*|on (both |one |your )?knees?|to (both |one |your )?knees?|crawl\w*|"
                   r"lie|lies|lying|sprawl\w*|on all fours|hands and knees|"
                   r"sit(s|ting)? (down )?on the (floor|ground)|seated on the (floor|ground)|"
                   r"(fall|falls|falling|plunge|plunges|drop|drops)( \w+){0,3} (to|onto) the (floor|ground)|"
                   r"push-?ups?|roll\w* (on|across) the (floor|ground)|from lying|tiger leap)\b", re.I)
meta = {r['motion_id']: r for r in csv.DictReader(open(f'{R}/metadata/hiphi_metadata.csv'))}
man = {r['archive_name']: r for r in csv.DictReader(open(f'{R}/data/archive_manifest.csv'))}
agg = collections.defaultdict(lambda: collections.Counter())
for r in csv.DictReader(open(f'{R}/data/motion_to_part.csv')):
    m = meta[r['motion_id']]
    part = r['archive_name'].split('_part_')[1][:4]
    h = float(m['duration_sec']) / 3600
    hoi = m['is_hoi'] == 'true'
    gc = m['frame_lu'] in GC_LU or bool(GC_RX.search(m['text_annotation']))
    a = agg[part]
    a['n'] += 1; a['h'] += h
    if hoi: a['hoi_n'] += 1; a['hoi_h'] += h
    if gc: a['gc_n'] += 1; a['gc_h'] += h
    if hoi or gc: a['prio_h'] += h
rows = []
for part, a in agg.items():
    mf = man[f'HiPHI_data_part_{part}_of_0032.tar.zst']
    rows.append(dict(part=part, prio_h=round(a['prio_h'], 3), hoi_h=round(a['hoi_h'], 3), gc_h=round(a['gc_h'], 3),
                     orig_h=round(a['h'], 3), n_orig=a['n'], hoi_n=a['hoi_n'], gc_n=a['gc_n'],
                     compressed_bytes=int(mf['compressed_bytes']), uncompressed_bytes=int(mf['uncompressed_bytes'])))
rows.sort(key=lambda d: (-d['prio_h'], -d['hoi_h'], d['part']))
for i, d in enumerate(rows): d['rank'] = i + 1
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['rank'] + [k for k in rows[0] if k != 'rank']); w.writeheader(); w.writerows(rows)
tot = collections.Counter()
for d in rows:
    for k in ('prio_h', 'hoi_h', 'gc_h', 'orig_h', 'n_orig', 'compressed_bytes', 'uncompressed_bytes'): tot[k] += d[k]
print(OUT); print({k: round(v, 2) for k, v in tot.items()})
for d in rows: print(d)
