"""Step 5 -- aggregate the unitree_r1 QA: assets check + 4 clips (factory qa.json + independent measure.json) ->
qa/unitree_r1/report.json with PASS/FAIL per DATA_SPEC_v2 rule.  (README.md is written by hand from this file.)"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_common as C

cfg = json.load(open(C.PDIR / 'pipeline.json'))
ck = json.load(open(C.QADIR / 'assets' / 'assets_check.json'))
man = [json.loads(l) for l in open(C.PDIR / 'out' / 'manifest.jsonl')]
IDS = [c['id'] for c in cfg['qa_clips']]
clips = {}
for q in IDS:
    rows = [r for r in man if r['qa_id'] == q]
    if not rows:
        clips[q] = dict(status='not_run'); continue
    row = rows[-1]
    if not row['status'].startswith('ok'):
        clips[q] = dict(status=row['status'], error=row.get('error')); continue
    mp = C.QADIR / q / 'measure.json'
    fq = json.load(open(Path(row['npz']).with_suffix('.qa.json')))
    me = json.load(open(mp)) if mp.exists() else None
    e = dict(status=row['status'], clip=row['clip'], frames=row['frames'], npz=row['npz'],
             factory=dict(flagged=fq['frames_flagged_final'], flags_hist=fq['flags_final_hist'],
                          penetration_after=fq['penetration_after'], projection_frames_needing_fix=fq['projection']['frames_needing_fix'],
                          ground_slip_after=fq.get('ground_slip_after'), ground_slip_human=fq.get('ground_slip_human'),
                          object_slip_after=fq.get('object_slip_after'), object_slip_human=fq.get('object_slip_human'),
                          seconds=fq['seconds']),
             measure=me,
             renders=sorted(str(p) for p in (C.QADIR / q).glob('*.png')) + sorted(str(p) for p in (C.QADIR / q).glob('*.mp4')))
    clips[q] = e


def V(q, f, default=None):
    me = clips[q].get('measure')
    if not me:
        return default
    try:
        return f(me)
    except (KeyError, TypeError, IndexError):
        return default


done = [q for q in IDS if clips[q].get('measure')]
rules = []


def rule(rid, cn, thr, vals, ok_fn, note=''):
    per = {q: dict(value=vals[q], pass_=bool(ok_fn(vals[q]))) for q in vals}
    verdict = 'PASS' if per and all(p['pass_'] for p in per.values()) else 'FAIL'
    rules.append(dict(id=rid, rule=cn, threshold=thr, verdict=verdict, per_clip={q: dict(value=p['value'], ok=p['pass_']) for q, p in per.items()}, note=note))


rule('solid_contact_dense', '真碰到: 输出接触标签的每一帧, 该区域 >=1 个稠密点离地/离物 [0,1mm]',
     'dense_solid_frac == 1.0 (label region-frames)',
     {q: dict(ground=V(q, lambda m: m['factory_label_solidity']['ground'].get('dense_solid_frac')),
              ground_incl_hull=V(q, lambda m: m['factory_label_solidity']['ground'].get('incl_hull_solid_frac')),
              object=V(q, lambda m: m['factory_label_solidity'].get('object', {}).get('dense_solid_frac')),
              dense_clearance_mm_p50_p95_max=V(q, lambda m: m['factory_label_solidity']['ground'].get('dense_clearance_mm_p50_p95_max'))) for q in done},
     lambda v: (v['ground'] in (None, 1.0)) and (v['object'] in (None, 1.0)),
     'hull-vertex gauge (engine) says 100% solid; dense-point gauge (spec) does not: R1 dense spacing ~13 mm')
rule('no_pen_ground', '不穿-地面: 任意点/碰撞几何低于地面 <= 0.5 mm', 'max <= 0.5 mm (dense, hull, MuJoCo plane)',
     {q: V(q, lambda m: max(m['penetration']['ground_dense_max_mm'], m['penetration']['ground_hull_max_mm'], m['penetration']['ground_mujoco_coll_max_mm'])) for q in done},
     lambda v: v is not None and v <= 0.5 + 1e-6)
rule('no_pen_object', '不穿-物体: 进入物体 <= 0.5 mm (稠密点 SDF + MuJoCo 碰撞)', 'max <= 0.5 mm',
     {q: V(q, lambda m: max(m['penetration']['object_dense_max_mm'], m['penetration']['object_mujoco_coll_max_mm'])) for q in done if V(q, lambda m: 'object' in m)},
     lambda v: v is not None and v <= 0.5 + 1e-6)
rule('no_pen_self', '不穿-自身: 非相邻部件 MuJoCo 碰撞几何 <= 0.5 mm (失败帧须被标记剔除)', 'unflagged frames > 0.5 mm == 0',
     {q: dict(max_mm=V(q, lambda m: m['penetration']['self_mujoco_max_mm']), frames=V(q, lambda m: m['penetration']['self_frames_gt_0p5mm']),
              unflagged=V(q, lambda m: m['failing_frames_not_flagged_by_factory']['penetration']),
              pairs=V(q, lambda m: m['penetration']['self_pairs_gt_0p5mm'])) for q in done},
     lambda v: v['unflagged'] == 0, 'frames > 0.5 mm that ARE flagged count as excluded (allowed by spec)')
rule('no_slip', '不滑: 接触期间接触点切向速度 < 1 cm/s (地面世界系; 物体在物体系)', 'frames with best-point speed >= 1 cm/s == 0',
     {q: dict(ground=V(q, lambda m: m['ground']['slip'].get('_all')), object=V(q, lambda m: m.get('object_contact', {}).get('slip', {}).get('_all')),
              frames_failing=V(q, lambda m: m['frames_failing']['slip_ge_1cms'])) for q in done},
     lambda v: v['frames_failing'] == 0, 'CPU reference LATCH_PIN=0 reproduces UMR slip; expected FAIL')
rule('no_flicker', '不闪: 接触段 >= 3 帧 (进入 <1mm / 离开 >2mm 滞回)', 'segments < 3 frames == 0',
     {q: V(q, lambda m: m['frames_failing']['flicker_lt3']) for q in done}, lambda v: v == 0)
rule('human_contact_robot_solid', '和人一致(接触): 人体区域接触 -> 机器人对应区域实接触', 'frames violating == 0',
     {q: dict(frames=V(q, lambda m: m['frames_failing']['human_contact_not_solid']),
              per_region={k: dict(human=e['human_contact_frames'], solid_frac=e.get('robot_solid_frac'), solid_incl_hull_frac=e.get('robot_solid_incl_hull_frac'),
                                  hover_1_20mm=e.get('hover_1_20mm_frames'), far_gt20mm=e.get('far_gt20mm_frames'))
                          for blk in ('ground', 'object_contact') for k, e in (m_.get(blk, {}).get('regions', {}) if (m_ := clips[q]['measure']) else {}).items()
                          if e.get('human_contact_frames')}) for q in done},
     lambda v: v['frames'] == 0)
rule('human_off_robot_off', '和人一致(离开): 人体区域离地/离物 > 2 cm -> 机器人对应区域 > 5 mm', 'frames violating == 0',
     {q: V(q, lambda m: m['frames_failing']['robot_contact_while_human_off']) for q in done}, lambda v: v == 0)
rule('failing_frames_flagged', '做不到的帧: 标记并剔除, 报精确数量 (不允许悄悄保留)', 'failing frames with factory flag == 0 must be 0',
     {q: V(q, lambda m: m['failing_frames_not_flagged_by_factory']) for q in done}, lambda v: all(x == 0 for x in v.values()),
     'factory flags only cover penetration + its own required-contact set')
rule('floor_z0', '地面坐标: 输出以校准地面 = z0, floor 原值另存', 'floor_height == 0, floor_offset_removed stored',
     {q: dict(floor_height_is_0=V(q, lambda m: m['format']['floor_height_is_0']), offset_stored=V(q, lambda m: m['format']['floor_offset_stored']),
              floor_offset_m=V(q, lambda m: m['floor_offset_removed_m']),
              human_foot_contact_lowest_mm_p05_p50_p95=V(q, lambda m: m['human_floor_check_mm']['foot_contact_lowest_p05_p50_p95'])) for q in done},
     lambda v: v['floor_height_is_0'] and v['offset_stored'])
rule('format_sparse_contact_le_1mm', '数据内容: 稀疏接触点索引只存 <= 1 mm 的点', 'ground_point_contact points > 1 mm == 0',
     {q: dict(frac_gt_1mm=V(q, lambda m: m['format']['ground_point_contact_frac_gt_1mm']), clearance_mm_p50_p95=V(q, lambda m: m['format']['ground_point_contact_clearance_mm_p50_p95'])) for q in done},
     lambda v: v['frac_gt_1mm'] in (0, 0.0, None), 'factory point label uses the 1 cm / 2 cm hysteresis, not 1 mm')
rule('format_quantized_dist_lt_2cm', '数据内容: < 2 cm 点距离(量化, 稀疏)', 'present', {q: V(q, lambda m: m['format']['has_quantized_distance_lt_2cm']) for q in done}, lambda v: bool(v))
rule('format_human_smplx_in_output', '数据内容: 人体 SMPL-X 参数 + 人体接触标签', 'SMPL-X params in the output npz',
     {q: V(q, lambda m: m['format']['has_human_smplx_params']) for q in done}, lambda v: bool(v), 'only source path in provenance; human labels present')

report = dict(robot=C.ROBOT, generated=time.strftime('%Y-%m-%d %H:%M:%S'), spec=cfg['spec'], pipeline_json=str(C.PDIR / 'pipeline.json'),
              engine=cfg['engine'],
              assets=dict(check=str(C.QADIR / 'assets' / 'assets_check.json'),
                          renders=[str(C.QADIR / 'assets' / f) for f in ('rest_regions.png', 'tpose_regions.png', 'patches.png')],
                          region_counts=ck['region_counts'], left_is_plus_y=ck['left_is_plus_y'], arm_order_ok=ck['arm_order_ok'],
                          rest_lowest_region=ck['rest_lowest_region'], patch_counts=ck['patch_counts'],
                          exclusions=ck['staging_exclusions'], self_overlap_rest=ck['self_overlap_rest'], self_overlap_tpose=ck['self_overlap_tpose'],
                          always_overlapping_pairs=ck['always_overlapping_pairs_ge95pct'],
                          random_pose_overlap_top=dict(list(ck['random_pose_overlap_frac'].items())[:5]),
                          bodies_without_collision=ck['bodies_without_collision'], dense_nn_spacing=ck['dense_nn_spacing'],
                          correspondence={k: dict(slots_exists=v['slots_exists'], names=v.get('names')) for k, v in ck['correspondence'].items()},
                          quirks=cfg['quirks']),
              clips=clips, rules=rules,
              summary=dict(rules_pass=[r['id'] for r in rules if r['verdict'] == 'PASS'], rules_fail=[r['id'] for r in rules if r['verdict'] == 'FAIL'],
                           frames_total=sum(clips[q].get('frames', 0) for q in done),
                           frames_failing_any=sum(V(q, lambda m: m['frames_failing_any'], 0) for q in done),
                           frames_failing_excl_slip=sum(V(q, lambda m: m['frames_failing_any_excl_slip'], 0) for q in done),
                           frames_flagged_by_factory=sum(clips[q]['factory']['flagged'] for q in done)))
if len(sys.argv) > 1:
    report['fixes'] = json.load(open(sys.argv[1]))
C.dump(C.QADIR / 'report.json', report)
print(json.dumps(C.jsonable(dict(summary=report['summary'], verdicts={r['id']: r['verdict'] for r in rules})), indent=1, ensure_ascii=False))
