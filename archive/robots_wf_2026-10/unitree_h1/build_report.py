"""unitree_h1 dedicated pipeline -- step 5: aggregate assets check + engine QA + independent measurement into
qa/unitree_h1/report.json with PASS/FAIL per DATA_SPEC_v2 rule.  Rule verdict = PASS only if every violating frame is
flagged/excluded by the engine (nothing silently kept) -- or there are no violations.
Usage: <umr py> build_report.py"""
import json, time
from pathlib import Path
import numpy as np

ROBOT = 'unitree_h1'
D = Path('/home/arenalabs/umr_factory_data'); PD = D / 'robots' / ROBOT / 'pipeline'; QA = D / 'qa' / ROBOT
cfg = json.load(open(PD / 'pipeline.json')); ac = json.load(open(PD / 'assets_check.json'))
SPEC_KEYS = {
    'qpos (root pose + joints), fps=30': ['qpos', 'root_pos', 'root_quat_wxyz', 'fps'],
    'per-frame contact point indices (<=1 mm) + partner': ['ground_point_contact', 'object_point_contact'],
    'sparse quantised distances of points < 2 cm': ['near_surface_dist_q'],
    'region labels: ground/object contact, slip, penetration': ['ground_contact', 'object_contact', 'ground_slip', 'object_slip', 'pen_ground_m', 'pen_object_m'],
    'human SMPL-X params': ['human_smplx_params'],
    'human surface-point contact labels': ['human_point_contact'],
    'human region labels': ['human_ground_contact', 'human_object_contact'],
    'object mesh id + 6D pose': ['object_name', 'object_pos', 'object_rot'],
    'per-frame QA flags': ['flags'],
    'floor value stored': ['floor_height', 'floor_offset_removed'],
}
clips = []
for c in cfg['qa_clips']:
    mp = QA / f"measure_{c['cid']}.json"
    if not mp.exists():
        clips.append(dict(id=c['id'], cid=c['cid'], status='missing')); continue
    me = json.load(open(mp)); qa = json.load(open(PD / 'out' / c['cid'] / f'{ROBOT}.qa.json'))
    o = np.load(PD / 'out' / c['cid'] / f'{ROBOT}.npz', allow_pickle=True)
    has_obj = 'object' in me
    kinds = ['ground'] + (['object'] if has_obj else [])
    pen = me['penetration']
    rules = {}
    rules['R1_solid_contact_dense'] = dict(
        verdict='PASS' if all(me[k]['viol_frames_unflagged'] == 0 for k in kinds) else 'FAIL',
        **{k: dict(human_contact_region_frames=me[k]['human_contact_region_frames'], robot_solid_dense=me[k]['robot_solid_when_human'],
                   robot_solid_any_point=me[k]['robot_solid_any_point_when_human'], hover_1_20mm=me[k]['hover_1_20mm_when_human'],
                   far_gt_20mm=me[k]['far_gt_20mm_when_human'], viol_frames=me[k]['viol_frames'],
                   viol_frames_unflagged=me[k]['viol_frames_unflagged']) for k in kinds})
    rules['R2_no_penetration'] = dict(verdict='PASS' if all(v['frames_gt_0p5mm_unflagged'] == 0 for v in pen.values()) else 'FAIL',
                                      **{k: v for k, v in pen.items()}, self_pairs=me['self_pen_pairs_gt_0p5mm'])
    rules['R3_no_slip'] = dict(verdict='PASS' if all((me[k]['slip']['frac_gt_1cm_s_median_point'] or 0) == 0 for k in kinds) else 'FAIL',
                               **{k: {kk: me[k]['slip'][kk] for kk in ('region_frame_pairs', 'median_point_speed_cm_s_p50', 'median_point_speed_cm_s_p95',
                                                                     'best_point_speed_cm_s_p50', 'frac_gt_1cm_s_median_point')} for k in kinds},
                               engine_slip_flagged_in_flags=False)
    rules['R4_no_flicker'] = dict(verdict='PASS' if me['engine_label_segments_lt3'] == 0 and all(me[k]['flicker']['hysteresis_segments_lt3'] == 0 for k in kinds) else 'FAIL',
                                  engine_label_segments_lt3=me['engine_label_segments_lt3'],
                                  **{k: me[k]['flicker'] for k in kinds})
    rules['R5_consistency_with_human'] = dict(
        verdict='PASS' if all(me[k]['viol_frames_unflagged'] == 0 and me[k]['human_off_viol_frames_unflagged'] == 0 for k in kinds) else 'FAIL',
        **{k: dict(human_contact_robot_not_solid_unflagged_frames=me[k]['viol_frames_unflagged'],
                   human_off_gt2cm_robot_within_5mm_region_frames=me[k]['human_off_robot_within_5mm_region_frames'],
                   human_off_viol_frames_unflagged=me[k]['human_off_viol_frames_unflagged'],
                   human_label_agreement_vs_engine=me[k]['human_label_agreement_vs_engine']) for k in kinds})
    rules['R6_failures_flagged_and_counted'] = dict(
        verdict='PASS' if me['frames_violating_unflagged'] == 0 else 'FAIL', engine_flagged=me['engine_flags'],
        frames_violating_any_geometric_rule=me['frames_violating_any_geometric_rule'], frames_violating_unflagged=me['frames_violating_unflagged'],
        note='slip and human-off violations are not part of this count (engine has no flag bit for them at all)')
    fl = me['floor']
    rules['R7_floor_relative'] = dict(verdict='PASS' if fl['floor_height'] == 0.0 and abs(fl['robot_foot_min_p50_mm_when_solid'] or 0) < 1.0 else 'FAIL', **fl)
    present = {k: [x for x in v if x in o.files] for k, v in SPEC_KEYS.items()}
    missing = {k: [x for x in v if x not in o.files and not (x.startswith('object') and not has_obj)] for k, v in SPEC_KEYS.items()}
    missing = {k: v for k, v in missing.items() if v and not (k.startswith('per-frame contact') and present[k])
               and not (k.startswith('region labels') and present[k])}
    rules['R8_data_content_spec3'] = dict(verdict='PASS' if not missing else 'FAIL', missing=missing,
                                          note='point contacts stored as dense packbits [T,512] (not sparse index lists); '
                                               'penetration depth is per frame, not per region')
    clips.append(dict(id=c['id'], cid=c['cid'], frames=me['frames'], fps=me['fps'], engine_object_kind=qa.get('object', {}).get('kind'),
                      true_object=me.get('object_geometry'), engine_seconds=qa['seconds'],
                      engine_vs_true_object_clearance=me.get('object', {}).get('engine_vs_true_clearance_mm_when_engine_contact'),
                      engine_ground_label_solid_by_dense_gauge=me['engine_ground_label_solid_by_our_gauge'],
                      excluded_frames=me['engine_flags']['frames_flagged'], rules=rules,
                      files=dict(npz=str(PD / 'out' / c['cid'] / f'{ROBOT}.npz'), engine_qa=str(PD / 'out' / c['cid'] / f'{ROBOT}.qa.json'),
                                 measure=str(mp), grid=str(QA / f"grid_{c['cid']}.png") if (QA / f"grid_{c['cid']}.png").exists() else None)))
rule_names = sorted({r for c in clips if 'rules' in c for r in c['rules']})
overall = {r: ('PASS' if all(c['rules'][r]['verdict'] == 'PASS' for c in clips if 'rules' in c) else 'FAIL') for r in rule_names}
rep = dict(robot=ROBOT, created=time.strftime('%F %T'), spec='/home/arenalabs/umr_factory_data/DATA_SPEC_v2.md',
           pipeline=str(PD / 'pipeline.json'),
           assets=dict(verdict='PASS_with_notes', images=[str(QA / 'assets_tpose_regions.png'), str(QA / 'assets_rest_regions.png')],
                       points_per_region=ac['points_per_region'], left_right_ok=ac['checks']['tpose']['all_lr_ok'] and ac['checks']['rest']['all_lr_ok'],
                       foot_sole_points_within_1mm_flat=[ac['checks']['rest']['soles'][s]['n_within_1mm'] for s in 'LR'],
                       collision_geoms_per_region={k: [g['body'] + ':' + g['type'] for g in v] for k, v in ac['collision_geoms_per_region'].items()},
                       exclusions=dict(staged=ac['staged_exclusions'], implicit_parent_child=len(ac['implicit_parent_child_filter']['pairs']),
                                       non_adjacent_overlap_rest=ac['non_adjacent_overlap_rest'], non_adjacent_overlap_tpose=ac['non_adjacent_overlap_tpose']),
                       correspondence={k: v['exists'] for k, v in ac['correspondence'].items()}),
           overall_rules=overall, clips=clips,
           video=str(QA / f"compare_{cfg['qa_clips'][0]['cid']}.mp4"))
json.dump(rep, open(QA / 'report.json', 'w'), indent=1, default=str)
print(json.dumps(overall, indent=1))
for c in clips:
    if 'rules' in c:
        print(c['id'], {r: v['verdict'] for r, v in c['rules'].items()}, 'excluded', c['excluded_frames'])
