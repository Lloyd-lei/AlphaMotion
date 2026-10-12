"""unitree_h2 dedicated pipeline -- step 5: compile qa/unitree_h2/report.json (+ rule verdicts) from the independent
measurements (and the raw-UMR control arm when present).  README.md is written by hand from this json."""
import hashlib, json, time
from pathlib import Path

ROBOT = 'unitree_h2'
DATA = Path('/home/arenalabs/umr_factory_data')
PD = DATA / 'robots' / ROBOT / 'pipeline'; QD = DATA / 'qa' / ROBOT
cfg = json.load(open(PD / 'pipeline.json')); ac = json.load(open(PD / 'assets_check.json'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rows = [json.loads(l) for l in open(PD / 'runs_manifest.jsonl')]
latest = {}
for r in rows:
    latest[r['clip_id']] = r
clips = {}
for c in cfg['qa_clips']:
    want = f"{ROBOT}_qa_{c['id']}__f{c['start']}-{c['end']}s{c['stride']}"
    r = latest.get(c['id'])
    if r is None or r['cid'] != want or not (QD / f'{want}.measure.json').exists():
        clips[c['id']] = dict(status='missing', cid=want); continue
    mz = json.load(open(QD / f'{want}.measure.json'))
    raw = json.load(open(QD / f'{want}.raw.measure.json')) if (QD / f'{want}.raw.measure.json').exists() else None
    eq = json.load(open(PD / 'out' / want / f'{ROBOT}.qa.json'))
    pen = mz['penetration']
    gc = mz['ground_contact']; oc = mz.get('object_contact')
    rule = {}
    pen_ok = pen['ground_frames_gt0p5mm'] == 0 and pen['self_frames_gt0p5mm'] == 0 and pen.get('object_frames_gt0p5mm', 0) == 0
    rule['no_penetration_(dense+visual+MuJoCo_ground/object/self)'] = 'PASS' if pen_ok else 'FAIL'
    rule['self_visual_hull_diagnostic_(links_without_collision,_knee)'] = 'PASS' if pen['diag_visual_hull_self_frames_gt0p5mm'] == 0 else 'WARN'
    rule['touch_ground_(human_contact->robot_[0,1mm])'] = 'PASS' if gc['human_contact_region_frames'] and gc['solid_strict_0_1mm'] == gc['human_contact_region_frames'] else 'FAIL'
    if oc is not None:
        rule['touch_object_(human_contact->robot_[0,1mm])'] = ('PASS' if oc['human_contact_region_frames'] and oc['solid_strict_0_1mm'] == oc['human_contact_region_frames']
                                                              else ('N/A' if not oc['human_contact_region_frames'] else 'FAIL'))
    sl = mz['ground_slip']
    rule['no_slip_ground_(<1cm/s)'] = 'PASS' if sl.get('region_frames', 0) and sl['frac_region_frames_any_point_ge_1cms'] == 0 else ('N/A' if not sl.get('region_frames') else 'FAIL')
    if oc is not None:
        so = mz['object_slip']
        rule['no_slip_object_(<1cm/s)'] = 'N/A (no robot object-contact frame pairs)' if not so.get('region_frames') else ('PASS' if so['frac_region_frames_any_point_ge_1cms'] == 0 else 'FAIL')
    fl = mz['ground_flicker']['hyst_segments_lt3'] + (mz['object_flicker']['hyst_segments_lt3'] if oc is not None else 0)
    rule['no_flicker_(segments>=3_frames,_1/2mm_hysteresis)'] = 'PASS' if fl == 0 else 'FAIL'
    sep = gc['separation_violation_frames'] + (oc['separation_violation_frames'] if oc is not None else 0)
    rule['human_separated->robot_separated_(>2cm->>5mm)'] = 'PASS' if sep == 0 else 'FAIL'
    rule['floor_relative_outputs'] = 'PASS' if mz['floor']['npz_floor_height'] == 0.0 and abs(mz['floor']['robot_min_z_p01_mm']) < 1.0 else 'FAIL'
    rule['failing_frames_flagged_and_counted'] = 'PASS' if mz['violating_but_not_flagged'] == 0 else 'FAIL'
    clips[c['id']] = dict(
        cid=want, source=dict(kind=c['kind'], data=c['data'], seq=c['seq'], window=[c['start'], c['end'], c['stride']]),
        frames=mz['frames'], fps=mz['fps'], output_npz=r['npz'], engine_seconds=r['seconds'],
        engine_flagged_frames=eq['frames_flagged_final'], engine_flags_hist=eq['flags_final_hist'],
        engine_object_solid=eq.get('object', {}).get('solid'),
        measured=dict(penetration=pen, ground_contact={k: v for k, v in gc.items() if k != 'per_region'},
                      ground_contact_per_region=gc['per_region'], ground_slip=sl, ground_flicker=mz['ground_flicker'],
                      object_contact=({k: v for k, v in oc.items() if k != 'per_region'} if oc else None),
                      object_contact_per_region=(oc['per_region'] if oc else None),
                      object_slip=mz.get('object_slip'), object_flicker=mz.get('object_flicker'),
                      object_geometry=mz.get('object_geometry'), floor=mz['floor'],
                      frames_violating_any_rule=mz['frames_violating_any_rule'], frames_violating_by_rule=mz['frames_violating_by_rule'],
                      violating_but_not_flagged=mz['violating_but_not_flagged'], human_contact_frames=mz['human_contact_frames']),
        raw_umr_control_arm=(dict(penetration=raw['penetration'],
                                  ground_contact={k: v for k, v in raw['ground_contact'].items() if k != 'per_region'},
                                  object_contact=({k: v for k, v in raw['object_contact'].items()} if raw.get('object_contact') else None),
                                  ground_slip=raw['ground_slip']) if raw else None),
        rules=rule,
        visuals=dict(grid=str(QD / f'{want}_grid12.png') if (QD / f'{want}_grid12.png').exists() else None,
                     mp4=str(QD / f'{want}.mp4') if (QD / f'{want}.mp4').exists() else None))
# overall verdict per rule: FAIL if any clip FAIL
allr = {}
for k, v in clips.items():
    for rn, verdict in v.get('rules', {}).items():
        allr.setdefault(rn, []).append(verdict)
overall = {rn: ('FAIL' if 'FAIL' in vs else ('WARN' if 'WARN' in vs else 'PASS')) for rn, vs in allr.items()}
rep = dict(robot=ROBOT, created=time.strftime('%Y-%m-%d %H:%M:%S'), spec=cfg['spec'],
           pipeline_json=str(PD / 'pipeline.json'), pipeline_json_sha256=sha(PD / 'pipeline.json'),
           assets_check=dict(json=str(PD / 'assets_check.json'), renders=[str(PD / 'assets_tpose_regions.png'), str(PD / 'assets_rest_regions.png')],
                             region_checks_all_true=all(all(v['checks'].values()) for v in ac['region_checks'].values()),
                             points_per_region=ac['points_per_region'], n_collision_geoms=ac['n_collision_geoms'],
                             bodies_without_collision=ac['bodies_with_visual_but_no_collision'],
                             exclusions=ac['staging_exclusions'], rest_overlaps=ac['rest_overlaps_lt_0p1mm'],
                             tpose_overlaps=ac['tpose_overlaps_lt_0p1mm'], correspondence_ok=all(
                                 ac['correspondence'][k]['dataset_exists'] and ac['correspondence'][k]['slots_exists'] for k in ('smplx', 'beta0'))),
           overall_rules=overall, clips=clips)
json.dump(rep, open(QD / 'report.json', 'w'), indent=1, default=str)
print(json.dumps(overall, indent=1))
for k, v in clips.items():
    print(k, v.get('rules'))
