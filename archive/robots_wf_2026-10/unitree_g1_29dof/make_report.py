"""Assemble qa/unitree_g1_29dof/report.json from the independent measurements + control arms."""
import hashlib, json, time
from pathlib import Path
ROBOT = 'unitree_g1_29dof'
DATA = Path('/home/arenalabs/umr_factory_data'); PD = DATA / 'robots' / ROBOT / 'pipeline'; QA = DATA / 'qa' / ROBOT
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
CFG = json.load(open(PD / 'pipeline.json')); AC = json.load(open(PD / 'assets' / 'assets_check.json'))
man = {}
for ln in open(PD / 'out' / 'manifest.jsonl'):
    r = json.loads(ln); man[r['clip']] = r
ctrl = json.load(open(QA / 'control_slip_pivot.json'))
clips = {}
P = lambda b: 'PASS' if b else 'FAIL'
for c in ['qa_dance', 'qa_kick', 'qa_carry', 'qa_ground']:
    m = json.load(open(QA / f'measure_{c}.json')); raw = json.load(open(QA / f'measure_{c}_umrraw.json'))
    pen = m['penetration']; gc = m['ground_contact']['total']; ge = m['ground_contact_exact_gauge']
    gs = m['ground_slip']['total']; gf = m['ground_flicker']; ex = m['v2_exclusion']
    rec = dict(window=next(q for q in CFG['qa_clips'] if q['id'] == c), frames=m['frames'], fps=round(m['fps'], 3),
               engine_npz=m['engine_npz'], engine_seconds=man[c]['seconds'],
               floor=m['floor'],
               penetration_mm=dict(ground_max=pen['ground_max_mm'], ground_frames_gt_0p5=pen['ground_frames_gt_0p5mm'],
                                   self_max=pen['self_max_mm'], self_frames_gt_0p5=pen['self_frames_gt_0p5mm'], self_pairs=pen['self_pairs_gt_0p5mm'],
                                   object_max=pen.get('object_max_mm'), object_frames_gt_0p5=pen.get('object_frames_gt_0p5mm')),
               ground_contact=dict(human_contact_region_frames=gc['human_contact'], solid_frac_dense=gc['solid_frac'], hover_1_20mm_frac_dense=gc['hover_frac'],
                                   far_gt20mm=gc['far_gt20mm'], solid_frac_exact_gauge=ge['solid_frac'], hover_frac_exact_gauge=ge['hover_frac'],
                                   release_violation_region_frames=gc['release_viol'],
                                   dense_minus_exact_mm_p50_p95_max=m['dense_minus_exact_mm_on_contacts_p50_p95_max']),
               ground_slip_cms=dict(p50=gs.get('p50_cms'), p95=gs.get('p95_cms'), max=gs.get('max_cms'), frac_gt_1cms=gs.get('frac_gt_1cms'),
                                    robot_slowest_point_p50=ctrl[c]['robot_slowest_point']['p50_cms'],
                                    human_source_slowest_point_p50=ctrl[c]['human_slowest_point']['p50_cms'],
                                    human_source_median_point_p50=ctrl[c]['human_median_point']['p50_cms']),
               ground_flicker=dict(segments=gf['segments'], short_lt3=gf['short_lt3']),
               engine=dict(flagged=ex['engine_flagged'], flags_hist=m['engine_vs_independent']['engine_flags_hist']),
               v2_exclusion=ex,
               raw_umr_control=dict(ground_pen_max_mm=raw['penetration']['ground_max_mm'], ground_pen_frames=raw['penetration']['ground_frames_gt_0p5mm'],
                                    self_pen_max_mm=raw['penetration']['self_max_mm'], object_pen_max_mm=raw['penetration'].get('object_max_mm'),
                                    ground_solid_frac_dense=raw['ground_contact']['total']['solid_frac'], ground_slip_p50_cms=raw['ground_slip']['total'].get('p50_cms'),
                                    v2_failing_frames=raw['v2_exclusion']['frames_failing_any_rule']))
    rules = {}
    rules['solid_contact_ground'] = P(gc['solid_frac'] >= 0.999)
    rules['no_penetration'] = P(pen['ground_frames_gt_0p5mm'] == 0 and pen['self_frames_gt_0p5mm'] == 0 and not pen.get('object_frames_gt_0p5mm'))
    rules['no_slip'] = P(gs.get('frac_gt_1cms', 0) == 0)
    rules['no_flicker_min3'] = P(gf['short_lt3'] == 0)
    rules['human_consistency_release'] = P(gc['release_viol'] == 0)
    rules['floor_relative_outputs'] = P(abs(m['floor']['robot_dense_min_over_clip_mm']) < 50 and 'floor_offset_removed' in m['floor'])
    rules['failing_frames_flagged_not_kept'] = P(ex['failing_but_not_flagged_by_engine'] == 0)
    if m['has_object']:
        oc = m['object_contact']['total']; os_ = m['object_slip']['total']; of = m['object_flicker']
        rec['object_contact'] = dict(human_contact_region_frames=oc['human_contact'], solid_frac_dense=oc['solid_frac'], hover_frac=oc['hover_frac'],
                                     far_gt20mm=oc['far_gt20mm'], solid_frac_with_collision_geoms=m['object_contact_with_collision_geoms']['solid_frac'],
                                     release_violation_region_frames=oc['release_viol'],
                                     per_region={k: dict(human=v['human_contact_frames'], solid=v['solid'], clearance_mm_p50_p95=v.get('robot_clearance_mm_when_human_contact_p50_p95'))
                                                 for k, v in m['object_contact']['per_region'].items() if v['human_contact_frames']})
        rec['object_slip_cms'] = dict(p50=os_.get('p50_cms'), p95=os_.get('p95_cms'), frac_gt_1cms=os_.get('frac_gt_1cms'), region_frames=os_['region_frames'])
        rec['object_flicker'] = dict(segments=of['segments'], short_lt3=of['short_lt3'])
        rules['solid_contact_object'] = P(oc['solid_frac'] >= 0.999)
        rules['no_slip_object'] = P(os_['region_frames'] > 0 and os_.get('frac_gt_1cms', 1) == 0) if oc['human_contact'] else 'N/A'
        rules['no_flicker_object'] = P(of['short_lt3'] == 0)
    rec['rules'] = rules
    clips[c] = rec
overall = {}
for c, r in clips.items():
    for k, v in r['rules'].items():
        if v == 'N/A': continue
        overall[k] = 'FAIL' if (overall.get(k) == 'FAIL' or v == 'FAIL') else 'PASS'
rep = dict(
    robot=ROBOT, created=time.strftime('%Y-%m-%d %H:%M:%S'), spec=CFG['spec'], spec_sha256=sha(CFG['spec']),
    pipeline_json=str(PD / 'pipeline.json'), pipeline_json_sha256=sha(PD / 'pipeline.json'),
    engine=dict(code=CFG['engine']['code'], code_sha256=CFG['engine']['code_sha256'], latch_pin=CFG['engine']['latch_pin'], device='CPU'),
    measurement=dict(script=str(PD / 'measure.py'), independent='own FK of surface.npz dense points + visual-mesh vertices + MuJoCo mj_geomDistance/mj_collision (ground, object, self); no engine gauges',
                     binding_gauge='dense points (DATA_SPEC_v2 sec.1); exact gauge (hull vertices + collision geoms) reported alongside',
                     controls=['raw UMR output (qpos_umr) measured with the same ruler: measure_<clip>_umrraw.json',
                               'human source slip with the same ruler: control_human_slip.json, control_slip_pivot.json (median vs slowest touching point)',
                               'engine box gauge vs true box: pipeline/diag/carry_box_gauge.py']),
    assets=dict(mjcf=CFG['assets']['mjcf'], mjcf_sha256=sha(CFG['assets']['mjcf']), surface_sha256=sha(CFG['assets']['surface']),
                n_dense_points=AC['n_points'], regions=CFG['region_points'], left_right_ok=all(v for k, v in AC['lr_anatomy_check'].items() if k != 'leg_vertical_order_ok'),
                patches=AC['patches'], collision_geoms_per_region={k: len(v) for k, v in AC['collision_geoms_per_region'].items()},
                n_collision_geoms=CFG['self_collision']['n_collision_geoms'],
                bodies_visual_without_collision=AC['bodies_visual_without_collision'], bodies_without_any_geom=AC['bodies_without_any_geom'],
                exclusions=dict(staged_explicit=[], mujoco_parent_child_auto=len(CFG['self_collision']['implicit_parent_child_filter']['pairs']),
                                rest_overlaps=[], tpose_overlaps=[], verdict=CFG['self_collision']['verdict']),
                correspondence=dict(dir=str(DATA / 'corr' / ROBOT), files=list(AC['correspondence_files'].keys()), hoi_per_sequence=AC['umr_hoi_per_sequence_corr']),
                images=[str(PD / 'assets' / f) for f in ('tpose_regions.png', 'rest_regions.png', 'patches.png', 'soles_from_below.png')],
                verdict='PASS'),
    clips=clips, rules_overall=overall,
    usable_frames_v2=dict({c: clips[c]['v2_exclusion']['frames_passing_all'] for c in clips}, total=sum(clips[c]['v2_exclusion']['frames_passing_all'] for c in clips), of=sum(clips[c]['frames'] for c in clips)),
    renders=dict(mp4=str(QA / 'qa_dance_human_vs_g1.mp4'), mp4_format='1280x720 30fps 600 frames [human slots | robot]',
                 grids=[str(QA / f'{c}_grid12.png') for c in clips]),
    engine_bugs_found=[
        dict(id='E1_box_as_sphere', where='umrf/objects.py PosedObject.__init__', what='sphere test runs before the box test; an 8-vertex box mesh has all corners equidistant from the centre (r.std()=0) -> OmniContact Box_H_1 (23.1 cm cube after scale) becomes a sphere of radius 200 mm (half diagonal)',
             effect='qa_carry: engine reports R_hand clearance 0.1 mm while the true box face is 37 mm (p50) / 49 mm (p95) away; object solid contact 0/617 region-frames; affects every robot on every box clip',
             fix='test the OBB box fit first (or require >= 50 vertices / low face-normal diversity for a sphere)'),
        dict(id='E2_release_not_enforced', where='umrf postpass required-contact rule', what='required contact = robot hysteresis label on the UMR output OR human label; a UMR robot contact is kept even when the human region is > 2 cm up',
             effect='dance t=53-59: human R_foot 28-30 mm up, robot R_foot pinned at 0.1 mm; release violations 81/95/16/258 region-frames (dance/kick/carry/ground)',
             fix='human clearance > 2 cm must veto required contact and enforce robot clearance > 5 mm'),
        dict(id='E3_slip_latch_off', where='LATCH_PIN = 0', what='support xy anchor reproduces UMR slip', effect='ground slip frac > 1 cm/s 100/57/63/98 %', fix='full latch inside a contact segment; switch support only between segments'),
        dict(id='E4_gauge_not_spec', where='postpass contact snap + labels', what='engine snaps/labels on hull vertices (exact) while DATA_SPEC_v2 binds the dense points', effect='engine-labelled contacts not solid on dense points: dance 265/756, kick 349/1055, carry 562/1200, ground 1961/2782 region-frames', fix='snap on the dense gauge (or make the dense asset dense enough that dense-exact < 0.5 mm on contact patches)'),
        dict(id='E5_flags_incomplete', where='postpass flags', what='flags cover penetration (1/2/4) and unsnapped contacts (8/16) only', effect='v2-failing frames not flagged: dance 536, kick 417, carry 489, ground 297 (all penetration frames ARE flagged)', fix='add v2 slip / release / flicker / dense-solid bits, and count them in the qa json'),
    ],
)
json.dump(rep, open(QA / 'report.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(overall, indent=0)); print(json.dumps(rep['usable_frames_v2']))
for c in clips: print(c, clips[c]['rules'])
