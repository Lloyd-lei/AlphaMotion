# -*- coding: utf-8 -*-
"""Synthesis numbers for REPORT.md (read-only; reads per-source summaries + per-clip CSVs).
Writes charts/synthesis_numbers.json. Run:  taskset -c 0-15 nice -n 10 env OMP_NUM_THREADS=2 python3 compute_numbers.py
"""
import csv, gzip, json, os, re
from collections import defaultdict

D = '/home/arenalabs/umr_factory_data/diversity'
F = '/home/arenalabs/umr_factory_data'
OUT = f'{D}/charts/synthesis_numbers.json'
CLASSES = ['locomotion', 'dynamic', 'ground_contact', 'sitting_on_object', 'upper_body_manipulation',
           'whole_body_object', 'gesture_social', 'idle_posture', 'other']
R = lambda x: round(float(x), 3)

res = {'generated': None, 'notes': []}
import datetime
res['generated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

# ---------------- HiPHI ----------------
hs = json.load(open(f'{D}/hiphi/hiphi_diversity_summary.json'))
rows = list(csv.DictReader(open(f'{D}/hiphi/hiphi_clips_labeled.csv')))
ready = [l.strip() for l in open(f'{F}/logs/hiphi_ready_parts.txt') if l.strip()]
ON_DISK = sorted(set(ready) | {'0002', '0029'})
# converted ids: CPU set (smplx/hiphi) + GPU log
conv = set()
for l in open(f'{F}/smplx_gpu/hiphi/convert_log.jsonl'):
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get('ok'):
        conv.add(r['clip_id'].split('/')[-1])
for root, ds, fs in os.walk(f'{F}/smplx/hiphi/data'):
    for f in fs:
        if f.endswith('.npz'):
            conv.add(os.path.basename(root))
part_gb = {}
for r in csv.DictReader(open(f'{F}/logs/hiphi_part_priority.csv')):
    part_gb[r['part']] = int(r['compressed_bytes']) / 1e9

H = {'on_disk_parts': ON_DISK, 'n_converted_clips': 0}
cls = {c: defaultdict(float) for c in CLASSES}
obj = defaultdict(lambda: defaultdict(float))
part = defaultdict(lambda: defaultdict(float))
contact_union = defaultdict(float)
heights = {}
gender_actor = {}
rep = 0.0
tot = 0.0
conv_h = 0.0
conv_n = 0
for r in rows:
    h = float(r['duration_sec']) / 3600
    tot += h
    p = r['part']
    st = 'converted' if r['motion_id'] in conv else ('on_disk' if p in ON_DISK else 'pending')
    if st == 'converted':
        conv_h += h; conv_n += 1
    prim = r['primary']
    cls[prim]['total'] += h
    cls[prim][st] += h
    if p in ON_DISK or st == 'converted':
        cls[prim]['in_hand'] += h
    o = r['object_categories']
    if r['is_hoi'] == 'True' and o:
        for oc in o.split('|'):
            obj[oc]['total'] += h
            obj[oc][st] += h
    # per-part diversity accounting
    sub = set(r['subtags'].split('|')) if r['subtags'] else set()
    con = set(r['contacts'].split('|')) if r['contacts'] else set()
    part[p]['orig_h'] += h
    part[p]['prim_' + prim] += h
    for s in ('dance', 'martial', 'acrobatic', 'fall', 'getup', 'run', 'jump_hop', 'kick'):
        if s in sub:
            part[p]['sub_' + s] += h
    rare_obj = r['is_hoi'] == 'True' and o and not set(o.split('|')) <= {'chair', 'table', 'box'}
    if rare_obj:
        part[p]['hoi_rare_obj'] += h
    scarce = prim in ('ground_contact', 'dynamic', 'sitting_on_object', 'gesture_social', 'idle_posture') or rare_obj
    if scarce:
        part[p]['scarce_h'] += h
    # contacts (six requested groups)
    noobj = not any(c.startswith('object') for c in con)
    ground_other = con & {'hands_ground', 'knees', 'hips_ground', 'back', 'front', 'side', 'body_ground_unspecified'}
    if con == {'feet_ground'} or (noobj and not ground_other and 'feet_ground' in con):
        contact_union['feet_only'] += h
    if 'hands_ground' in con: contact_union['hands_ground'] += h
    if 'knees' in con: contact_union['knees'] += h
    if con & {'hips_ground', 'back', 'front', 'side', 'body_ground_unspecified', 'object_hips'}:
        contact_union['hips_back'] += h
    if 'object_hands' in con: contact_union['object_hands'] += h
    if 'object_feet' in con: contact_union['object_feet'] += h
    heights[r['actor_id']] = int(r['height_cm'])
    gender_actor[r['actor_id']] = r['gender']
    if re.search(r'repeat', r['text_annotation'], re.I):
        rep += h
H['hours_orig_total'] = R(tot)
H['converted_clips'] = conv_n
H['converted_hours'] = R(conv_h)
H['on_disk_hours'] = R(sum(v['orig_h'] for k, v in part.items() if k in ON_DISK))
H['classes_primary'] = {c: {k: R(v) for k, v in cls[c].items()} for c in CLASSES}
H['objects'] = {k: {kk: R(vv) for kk, vv in v.items()} for k, v in sorted(obj.items(), key=lambda x: -x[1]['total'])}
H['contacts6'] = {k: R(v) for k, v in contact_union.items()}
H['actor_heights_cm'] = sorted(heights.values())
H['actor_gender'] = {g: sum(1 for a in gender_actor.values() if a == g) for g in set(gender_actor.values())}
H['repeat_text_hours'] = R(rep)
# per-part download priority (parts not on disk)
pp = []
for p, v in part.items():
    if p in ON_DISK:
        continue
    gb = part_gb.get(p, 0)
    d = {k: R(x) for k, x in v.items()}
    d['part'] = p
    d['gb'] = R(gb)
    d['scarce_h_per_gb'] = R(v['scarce_h'] / gb) if gb else None
    pp.append(d)
pp.sort(key=lambda d: -(d['scarce_h_per_gb'] or 0))
H['remaining_parts_ranked'] = pp
res['hiphi'] = H

# ---------------- other per-clip CSVs (contacts union, actors) ----------------
def contact6_from_csv(path, hours_col, contacts_col='contacts', filt=None, gz=True):
    op = gzip.open(path, 'rt') if gz else open(path)
    out = defaultdict(float)
    for r in csv.DictReader(op):
        if filt and not filt(r):
            continue
        h = float(r[hours_col])
        con = set(r[contacts_col].split('|')) if r[contacts_col] else set()
        if con == {'feet_only'}: out['feet_only'] += h
        if 'hands_ground' in con: out['hands_ground'] += h
        if 'knees' in con: out['knees'] += h
        if con & {'hips', 'back'}: out['hips_back'] += h
        if 'object_hands' in con: out['object_hands'] += h
        if 'object_feet' in con: out['object_feet'] += h
    return {k: R(v) for k, v in out.items()}

res['bones_contacts6'] = contact6_from_csv(f'{D}/bones_seed/bones_clip_labels.csv.gz', 'hours', filt=lambda r: r['is_mirror'] == 'False')
res['amass_contacts6'] = contact6_from_csv(f'{D}/xwbc_amass_lafan/amass_clip_labels.csv.gz', 'hours')
res['lafan_contacts6'] = contact6_from_csv(f'{D}/xwbc_amass_lafan/lafan_clip_labels.csv', 'hours', gz=False)
res['mixamo_contacts6'] = contact6_from_csv(f'{D}/uniml3d_mixamo/uniml3d_clip_labels.csv.gz', 'hours', filt=lambda r: r['dataset'] == 'mixamo')

bh = {}; bg = {}
for r in csv.DictReader(gzip.open(f'{D}/bones_seed/bones_clip_labels.csv.gz', 'rt')):
    if r['actor'] and r['height_cm']:
        bh[r['actor']] = float(r['height_cm']); bg[r['actor']] = r['gender']
res['bones_actor_heights_cm'] = sorted(bh.values())
res['bones_actor_gender'] = {g: sum(1 for a in bg.values() if a == g) for g in set(bg.values())}

# ---------------- GRAIL current (manifest_summary, live) ----------------
gm = json.load(open(f'{F}/smplx/grail/manifest_summary.json'))
og = json.load(open(f'{D}/omnicontact_grail/summary.json'))
G = {'converted_now': {}, 'human_available_est': {}, 'pending_est': {}}
for k, v in gm['per_category'].items():
    G['converted_now'][k] = v['hours_converted']
    ha = og['grail']['per_category'][k]['hours']['smplx_available_on_hf_est']
    G['human_available_est'][k] = ha
    G['pending_est'][k] = R(max(0.0, ha - v['hours_converted']))
G['converted_total'] = gm['total']['hours_converted']
G['clips_ok'] = gm['total']['clips_ok']; G['clips_pending'] = gm['total']['clips_pending']; G['clips_dropped'] = gm['total']['clips_dropped']
G['pickup_ground_no_human_h'] = R(og['grail']['per_category']['pickup_ground']['hours']['release_est(README帧数/25Hz)'] - og['grail']['per_category']['pickup_ground']['hours']['smplx_available_on_hf_est'])
res['grail'] = G

json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print('wrote', OUT)
print(json.dumps({k: v for k, v in H.items() if k not in ('actor_heights_cm', 'remaining_parts_ranked', 'objects')}, ensure_ascii=False, indent=0))
for d in pp:
    print(d['part'], d['gb'], round(d['orig_h'], 2), 'scarce', d.get('scarce_h'), 'per_gb', d['scarce_h_per_gb'],
          'gc', d.get('prim_ground_contact', 0), 'dyn', d.get('prim_dynamic', 0), 'dance', d.get('sub_dance', 0), 'mart', d.get('sub_martial', 0),
          'acro', d.get('sub_acrobatic', 0), 'fall', d.get('sub_fall', 0), 'getup', d.get('sub_getup', 0), 'gest', d.get('prim_gesture_social', 0),
          'idle', d.get('prim_idle_posture', 0), 'sit', d.get('prim_sitting_on_object', 0), 'rareobj', d.get('hoi_rare_obj', 0))
print('objects', json.dumps(H['objects'], ensure_ascii=False))
for k in ('bones_contacts6', 'amass_contacts6', 'lafan_contacts6', 'mixamo_contacts6', 'bones_actor_gender', 'grail'):
    print(k, json.dumps(res[k], ensure_ascii=False))
print('bones heights n', len(bh), min(bh.values()), max(bh.values()))
