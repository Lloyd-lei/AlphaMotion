from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
# Aggregate OmniContact + GRAIL diversity survey -> summary.json + summary.md (Chinese). Read-only on sources.
import json, os, re, glob, collections, time, csv
import numpy as np

D = f'{SRCFG.data_root}/diversity/omnicontact_grail'
F = f'{SRCFG.data_root}'
GM = f'{F}/smplx/grail/manifest.jsonl'
NOW = time.strftime('%Y-%m-%d %H:%M:%S')


def pct(a, q=(5, 50, 95), nd=2):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return [round(float(v), nd) for v in np.percentile(a, q)] if a else None


# ---------------------------------------------------------------- retarget status (UMR, any robot)
def umr_retargets():
    hits = collections.defaultdict(set)
    for p in glob.glob(f'{F}/out/*/*.npz') + glob.glob(f'{F}/robots/*/pipeline/out/*/*.npz'):
        run, robot = p.split('/')[-2], os.path.basename(p)[:-4]
        if robot.endswith('src_slots'):
            continue
        m = re.search(r'(\d{14}_\d_\d{10})', run)
        if not m:  # run name without capture id -> look in the matching UMR log
            for lg in glob.glob(p.replace('/out/', '/runs/').rsplit('/', 1)[0] + '/*.umr.log'):
                with open(lg, errors='ignore') as fh:
                    m = re.search(r'(2026\d{10}_\d_\d{10})', fh.read(200000))
                if m:
                    break
        if m:
            hits[('omnicontact', m.group(1))].add(robot)
        elif re.search(r'grail|curb|slope|stair|sitting|pickup', run):
            hits[('grail', run)].add(robot)
    return hits


RT = umr_retargets()

# ---------------------------------------------------------------- OmniContact
OC_CASE = {
    'box/case1_carry': dict(name='搬箱子', tax={'whole_body_object': ['carry_box'], 'upper_body_manipulation': ['pick', 'place', 'carry'], 'locomotion': ['walk', 'turn']},
                            contact=['feet', 'object-hands', 'object-torso(抱箱贴胸, 人体穿模统计佐证)'], contact_evidence='G1 标签 wrist 接触 65.5%, ankle 0%'),
    'box/case2_push': dict(name='推箱子', tax={'whole_body_object': ['push_box'], 'locomotion': ['walk', 'turn']},
                           contact=['feet', 'object-hands'], contact_evidence='G1 标签 wrist 接触 38.5%, ankle 0%; 躯干前倾 p95≈52°'),
    'box/case3_slide': dict(name='用脚滑/推箱子', tax={'whole_body_object': ['push_box_with_foot'], 'locomotion': ['walk', 'turn']},
                            contact=['feet', 'object-feet'], contact_evidence='G1 标签(104/118段) ankle 接触 L8.2%/R21.2%, wrist 仅1.7%'),
    'soccer/case1_kick_forward': dict(name='向前踢球', tax={'dynamic': ['kick'], 'whole_body_object': ['kick_ball'], 'locomotion': ['walk', 'turn']},
                                      contact=['feet', 'object-feet'], contact_evidence='推断(足球 G1 标签未下载); 人体-球 2cm 内帧比 p50≈2%'),
    'soccer/case2_kick_left': dict(name='向左踢球', tax={'dynamic': ['kick'], 'whole_body_object': ['kick_ball'], 'locomotion': ['walk', 'turn']},
                                   contact=['feet', 'object-feet'], contact_evidence='推断; 人体-球 2cm 内帧比 p50≈2%'),
    'soccer/case3_kick_right': dict(name='向右踢球', tax={'dynamic': ['kick'], 'whole_body_object': ['kick_ball'], 'locomotion': ['walk', 'turn']},
                                    contact=['feet', 'object-feet(含小腿)'], contact_evidence='UMR 试跑: 人体 L_foot 503 帧/L_shin 483 帧触球'),
    'soccer/case4_carry': dict(name='捡球搬运', tax={'upper_body_manipulation': ['pick', 'carry', 'place'], 'locomotion': ['walk', 'turn']},
                               contact=['feet', 'object-hands'], contact_evidence='推断; 人体-球 2cm 内帧比 p50≈68%, 弯腰 p95≈51°'),
    'soccer/case5_pickball_kick': dict(name='捡球再踢', tax={'upper_body_manipulation': ['pick', 'place'], 'dynamic': ['kick'], 'whole_body_object': ['kick_ball'], 'locomotion': ['walk', 'turn']},
                                       contact=['feet', 'object-hands', 'object-feet'], contact_evidence='推断; 人体-球 2cm 内帧比 p50≈62%'),
}
OC_README = {'box/case1_carry': (210, 623), 'box/case2_push': (162, 453), 'box/case3_slide': (118, 190),
             'soccer/case1_kick_forward': (30, 177), 'soccer/case2_kick_left': (30, 170), 'soccer/case3_kick_right': (30, 210),
             'soccer/case4_carry': (51, 199), 'soccer/case5_pickball_kick': (59, 209)}
OC_OBJ = {'Box_H_1': '箱子 30×30×30 cm', 'Box_K_1': '箱子 48.5×40×72.5 cm', 'ScoccerBall_A_1': '足球 直径 21.6 cm', 'Box_I_1': '箱子 40×40×60 cm(资产中有, 转换片段未用)'}


def omnicontact():
    rows = [json.loads(l) for l in open(f'{D}/omnicontact_clip_stats.jsonl')]
    by = collections.defaultdict(list)
    for r in rows:
        by[f"{r['domain']}/{r['case']}"].append(r)
    raw_n = {k: len([x for x in os.listdir(f'{F}/raw/omnicontact/raw_mocap/{k}') if os.path.isdir(f'{F}/raw/omnicontact/raw_mocap/{k}/{x}')]) for k in OC_CASE}
    cats = {}
    for k, meta in OC_CASE.items():
        v = by[k]
        hrs = sum(r['seconds_T'] for r in v) / 3600
        rt = {r['capture_id']: sorted(RT[('omnicontact', r['capture_id'])]) for r in v if ('omnicontact', r['capture_id']) in RT}
        g = [r['g1_obj_contact_frac_lA_rA_lW_rW'] for r in v if r.get('g1_obj_contact_frac_lA_rA_lW_rW')]
        gmean = np.mean(g, 0).round(3).tolist() if g else None
        cats[k] = {
            'name_zh': meta['name'], 'taxonomy': meta['tax'],
            'clips': {'source_captures_release': OC_README[k][0], 'task_clips_release_npz_clips': OC_README[k][1],
                      'on_disk_raw': raw_n[k], 'converted_smplx': len(v), 'converted_ok': sum(1 for r in v if r['ok']),
                      'retargeted_umr_captures': len(rt), 'retargeted_umr_detail': rt},
            'fps_source': 90, 'frames': int(sum(r['T'] for r in v)),
            'hours_original': round(hrs, 3), 'hours_mirrored': 0.0,
            'has_object': True, 'objects': dict(collections.Counter(OC_OBJ.get(o, o) for r in v for o in r['objects'])),
            'contact_types': meta['contact'], 'contact_evidence': meta['contact_evidence'],
            'interaction_frac': {'g1_label_obj_contact_approx(max手+max脚)':(round(max(gmean[2], gmean[3]) + max(gmean[0], gmean[1]), 3) if gmean else None),
                                 'g1_label_frac_lAnkle_rAnkle_lWrist_rWrist_mean': gmean, 'g1_label_captures': len(g),
                                 'human_obj_within_2cm_frac_p50': pct([r.get('obj_within_2cm_frac_sampled') for r in v])[1]},
            'motion': {'pelvis_drop_m_p5_50_95': pct([r['pelvis_drop_from_median_m'] for r in v]),
                       'captures_with_deep_squat_gt45cm_ge1pct_frames': sum(1 for r in v if r['frac_pelvis_drop_gt_0p45'] >= 0.01),
                       'path_m_p5_50_95': pct([r['path_xz_m'] for r in v], nd=1),
                       'speed_p95_mps_p5_50_95': pct([r['speed_p95'] for r in v]),
                       'run_frac_gt2mps_max': round(max(r['frac_speed_gt_2p0'] for r in v), 4),
                       'yaw_total_deg_p50': pct([r['yaw_total_deg'] for r in v], nd=0)[1],
                       'torso_tilt_p95_deg_p50': pct([r['torso_tilt_p95_deg'] for r in v], nd=1)[1]},
            'subjects': {'bvh_skeleton_calibrations': len(set(r.get('skel_hash') for r in v)),
                         'bvh_stature_cm_min_max': [round(min(r['skel_stature_cm'] for r in v), 1), round(max(r['skel_stature_cm'] for r in v), 1)]},
            'split': dict(collections.Counter(r['split'] for r in v)),
            'ik_residual_cm_p50': pct([r['ik_residual_cm_mean'] for r in v])[1],
            'status': 'converted_smplx' + (' (+%d 段 QA 窗口试 retarget)' % len(rt) if rt else ''),
        }
    st = sorted(set(round(r['skel_stature_cm'], 1) for r in rows))
    clusters, cur = [], [st[0]]
    for x in st[1:]:
        if x - cur[-1] > 1.0:
            clusters.append(cur); cur = [x]
        else:
            cur.append(x)
    clusters.append(cur)
    tot_h = sum(c['hours_original'] for c in cats.values())
    return {
        'license': 'CC BY-NC 4.0 (lightcone02/OmniContact-Dataset, gated=auto)',
        'capture': '光学动捕 BVH(omni51 骨架) 90 fps + 物体位姿 CSV; 转 SMPL-X neutral (Y-up) 90 fps',
        'subjects': {'bvh_skeleton_calibrations': len(set(r.get('skel_hash') for r in rows)),
                     'stature_cm_min_max': [st[0], st[-1]],
                     'stature_clusters_cm(间隔>1cm 聚类, 估计演员数)': [[c[0], c[-1]] for c in clusters],
                     'estimated_actors': len(clusters),
                     'gender': '未提供(元数据无性别; 转换用 neutral)',
                     'note': 'BVH 骨架身高=静止姿态 头顶端点到脚趾端点 Y 跨度, 约等于身高; 骨架校准数≠人数'},
        'per_category': cats,
        'totals': {'captures': len(rows), 'hours_original': round(tot_h, 3), 'hours_mirrored': 0.0,
                   'task_clips_release': sum(v[1] for v in OC_README.values()),
                   'retargeted_captures': len({c for (s, c) in RT if s == 'omnicontact'}),
                   'retargeted_robots_union': sorted(set().union(*[v for (s, _), v in RT.items() if s == 'omnicontact']) if RT else [])},
        'remaining_download': {
            'raw_mocap': '全部 690 段已在盘(case2_push 目录 165 个, 比 README 多 3 个)',
            'npz(G1 轨迹+接触标签)': 'g1_labels 仅 box 476/490 段(case3_slide 缺 14); 足球 200 段标签未取; 全量 npz 7.58 GB',
            'npz_clips(2231 任务切片, 含箱子尺寸增广)': '未下载, 6.86 GB; 是 G1 关节轨迹, 不是人体, 对 SMPL-X 管线无新增动作',
        },
    }


# ---------------------------------------------------------------- GRAIL
G_README = {  # category: (objects, motions, seq_len_s, total_frames @25Hz, desc)
    'pickup_table': (685, 2991, 10, 747750, '桌面取物: 从桌上抓起物体, 举起, 搬运'),
    'pickup_ground': (644, 1736, 15, 651000, '地面取物: 跪下抓地上物体再站起'),
    'sitting': (189, 1748, 5, 218500, '坐到椅子类物体上'),
    'slope': (200, 1880, 10, 470000, '斜坡行走'),
    'curb': (200, 1769, 10, 442250, '跨路沿'),
    'stair_p1': (None, 6094, 10, 3047000 / 2, '上下楼梯(生成+真实楼梯资产) 第1部分'),
    'stair_p2': (None, 6094, 10, 3047000 / 2, '上下楼梯 第2部分'),
}


def sit_class(n):
    for kw, lab in [('rock', '摇椅'), ('stool', '凳子'), ('bench', '长凳式'), ('throne', '宝座/礼仪椅'), ('ceremonial', '宝座/礼仪椅'),
                    ('lounge', '躺椅/休闲椅'), ('reclin', '躺椅/休闲椅'), ('deck_chair', '躺椅/休闲椅'), ('hanging', '吊椅'),
                    ('armchair', '扶手椅'), ('captain', '扶手椅'), ('wingback', '扶手椅'), ('elbow', '扶手椅'),
                    ('sculpt', '雕塑/异形座'), ('novelty', '雕塑/异形座'), ('shaped', '雕塑/异形座'), ('seat', '单座/剧场座'),
                    ('dining', '餐椅/侧椅'), ('side_chair', '餐椅/侧椅'), ('kitchen', '餐椅/侧椅'), ('chair', '其他椅子')]:
        if kw in n:
            return lab
    return '其他'


def stair_class(n):
    if n.replace('_updown', '') == 'stairs_real':
        return '真实楼梯资产(stairs_real)'
    if re.match(r'stairs_\d+$', n.replace('_updown', '')):
        return '程序化楼梯 stairs_NNNN'
    if 'practice_stair_block' in n:
        return '小型练习台阶(3-6 级)'
    if 'curve' in n:
        return '弧形楼梯'
    if 'straight' in n:
        return '直跑楼梯(生成, 10-15 级)'
    return '其他楼梯'


def grail():
    man = [json.loads(l) for l in open(GM)]
    man_mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(GM)))
    scan = {r['clip']: r for r in (json.loads(l) for l in open(f'{D}/grail_clip_stats.jsonl')) if 'error' not in r}
    tree = json.load(open(f'{F}/raw/grail/_tree.json'))
    rem = json.load(open(f'{D}/_hf_remaining_tree.json'))
    by = collections.defaultdict(list)
    for r in man:
        by[r['category']].append(r)
    cats = {}
    label_hours = collections.defaultdict(float)
    label_clips = collections.Counter()
    for c, (nobj, nmot, seql, tf, desc) in G_README.items():
        v = by[c]
        ok = [r for r in v if r['status'] == 'ok']
        st = collections.Counter(r['status'] for r in v)
        local_recon = set(os.path.basename(p)[:-4] for p in glob.glob(f'{F}/raw/grail/data/{c}/recon/*.pkl'))
        conv_now = set(os.path.basename(p)[:-4] for p in glob.glob(f'{F}/smplx/grail/{c}/recon/*.pkl'))
        hf_recon = dict((os.path.basename(p)[:-4], s) for p, s in tree[f'{c}/recon']['files'])
        miss = [k for k in hf_recon if k not in local_recon]
        sc = [scan[r['clip']] for r in ok if r['clip'] in scan]
        inter = [(r['interaction_frames'][1] - r['interaction_frames'][0]) / r['frames'] for r in ok if r.get('interaction_frames') and r.get('frames')]
        objn = [r['clip'].split('__')[1] for r in v]
        # object categories
        if c in ('pickup_table', 'pickup_ground'):
            oc = collections.Counter(re.sub(r'_\d+$', '', o) for o in objn)
            objects = {'n_unique_instances': len(set(objn)), 'n_classes': len(oc), 'top_classes(clips)': dict(oc.most_common(25)),
                       'size_max_extent_m_p5_50_95': pct([max(r['object']['extent_m']) for r in ok if r.get('object')], nd=3), 'source': 'RoboCasa 衍生 3D 资产(厨房/食物/餐具小物件)'}
            if c == 'pickup_table':
                objects['table'] = '静态桌 2.0×0.6 m, 桌面高约 0.40 m'
        elif c == 'sitting':
            oc = collections.Counter(sit_class(o) for o in objn)
            objects = {'n_unique_instances': len(set(objn)), 'classes(clips)': dict(oc.most_common()),
                       'extent_m_p50(xyz)': np.percentile([r['object']['extent_m'] for r in ok], 50, axis=0).round(2).tolist(), 'source': 'Hunyuan3D-2.1 生成椅子'}
        elif c in ('stair_p1', 'stair_p2'):
            oc = collections.Counter(stair_class(o) for o in objn)
            steps = [int(m.group(1)) for o in set(objn) for m in [re.search(r'with_(\d+)_', o)] if m]
            objects = {'n_unique_instances': len(set(objn)), 'n_unique_geometry(去 _updown)': len(set(o.replace('_updown', '') for o in objn)),
                       'classes(clips)': dict(oc.most_common()), 'updown_clips': sum('updown' in o for o in objn),
                       'named_step_count_min_p50_max': [min(steps), int(np.median(steps)), max(steps)] if steps else None,
                       'extent_m_p5_50_95(xyz)': np.percentile([r['object']['extent_m'] for r in ok], [5, 50, 95], axis=0).round(2).tolist()}
        else:
            ex = np.array([r['object']['extent_m'] for r in ok])
            objects = {'n_unique_instances': len(set(objn)), 'source': '程序化地形'}
            if c == 'curb':
                objects['curb_height_m_p5_50_95'] = np.percentile(ex[:, 2], [5, 50, 95]).round(3).tolist()
            else:
                objects['slope_rise_m_p5_50_95'] = np.percentile(ex[:, 2], [5, 50, 95]).round(3).tolist()
                objects['slope_rise_over_run_deg_p5_50_95'] = np.degrees(np.arctan(np.percentile(ex[:, 2] / ex[:, 0], [5, 50, 95]))).round(1).tolist()
        # per-clip multi-labels with evidence
        lab_counts = collections.Counter()
        for r in ok:
            s = scan.get(r['clip'])
            labs = set()
            if c in ('curb', 'slope', 'stair_p1', 'stair_p2'):
                labs |= {'locomotion', 'locomotion/walk', 'locomotion/' + {'curb': 'curb', 'slope': 'slope'}.get(c, 'stairs')}
                if s:
                    if s['dz_end_start_m'] > 0.08: labs.add('locomotion/ascend')
                    elif s['dz_end_start_m'] < -0.08: labs.add('locomotion/descend')
                    else: labs.add('locomotion/level_or_up-down')
            elif c == 'sitting':
                labs |= {'sitting_on_object', 'sitting_on_object/chair'}
            elif c == 'pickup_table':
                labs |= {'upper_body_manipulation', 'upper_body_manipulation/pick', 'upper_body_manipulation/lift_in_place'}
            elif c == 'pickup_ground':
                labs |= {'upper_body_manipulation', 'upper_body_manipulation/pick'}
                if s and s['pelvis_drop_m'] > 0.35: labs |= {'ground_contact', 'ground_contact/kneel_or_deep_squat'}
            if s:
                if s['net_xy_m'] > 1.0 and c not in ('curb', 'slope', 'stair_p1', 'stair_p2'): labs |= {'locomotion', 'locomotion/walk'}
                if abs(s['yaw_net_deg']) > 60: labs.add('locomotion/turn')
                if s['speed_p95'] > 2.2: labs.add('locomotion/run')
            lab_counts.update(labs)
            for l in labs:
                label_hours[l] += r['frames'] / 24 / 3600
                label_clips[l] += 1
        contact = {'curb': ['feet only(地形物体上)'], 'slope': ['feet only(地形物体上)'], 'stair_p1': ['feet only(楼梯物体上)'], 'stair_p2': ['feet only(楼梯物体上)'],
                   'sitting': ['feet', 'hips(坐面)', 'back(靠背, 推断未核实)'],
                   'pickup_table': ['feet', 'object-hands'], 'pickup_ground': ['feet', 'knees(跪, README 所述 + 骨盆下降佐证)', 'object-hands']}[c]
        hand_obj = pct([r['qa'].get('hand_obj_within_20mm_frac') for r in ok if r.get('qa') and r['qa'].get('hand_obj_within_20mm_frac') is not None], nd=3)
        rel_h = tf / 25 / 3600
        cats[c] = {
            'desc_zh': desc,
            'taxonomy_label_clip_counts(ok段, 多标签)': dict(sorted(lab_counts.items())),
            'clips': {'release': nmot, 'recon_smplx_on_hf': len(hf_recon), 'on_disk_raw_recon': len(local_recon),
                      'manifest_ok': st.get('ok', 0), 'manifest_pending': st.get('pending', 0), 'manifest_dropped': st.get('dropped', 0),
                      'converted_pkl_on_disk_now': len(conv_now), 'converter_version_of_ok': dict(collections.Counter(r.get('converter_version') for r in ok)),
                      'retargeted_umr': 0},
            'fps': {'recon_true': 24, 'source_pkl_field': 30, 'release_robot_traj_hz': 25},
            'hours': {'release_est(README帧数/25Hz)': round(rel_h, 3),
                      'smplx_available_on_hf_est': round(rel_h * len(hf_recon) / nmot, 3),
                      'converted_ok(24fps)': round(sum(r['frames'] for r in ok) / 24 / 3600, 3),
                      'converted_final_v2v3': round(sum(r['frames'] for r in ok if (r.get('converter_version') or 0) >= 2) / 24 / 3600, 3),
                      'converted_old_v1(待重转)': round(sum(r['frames'] for r in ok if (r.get('converter_version') or 0) < 2) / 24 / 3600, 3),
                      'src_valid_of_final_v2v3': round(sum(r.get('src_valid_hours') or 0 for r in ok if (r.get('converter_version') or 0) >= 2), 3),
                      'mirrored': 0.0},
            'seq_len_s_readme': seql,
            'has_object': True, 'objects': objects,
            'contact_types': contact,
            'interaction_frac(interaction_frames/frames)': {'mean': round(float(np.mean(inter)), 3) if inter else None, 'p5_50_95': pct(inter, nd=3)},
            'hand_obj_within_20mm_frac_p5_50_95': hand_obj,
            'motion(已转换段)': {'n_scanned': len(sc),
                              'pelvis_drop_m_p5_50_95': pct([s['pelvis_drop_m'] for s in sc]),
                              'dz_end_start_m_p5_50_95': pct([s['dz_end_start_m'] for s in sc]),
                              'net_xy_m_p5_50_95': pct([s['net_xy_m'] for s in sc]),
                              'speed_p95_mps_p5_50_95': pct([s['speed_p95'] for s in sc]),
                              'yaw_net_deg_abs_p50_p95': pct([abs(s['yaw_net_deg']) for s in sc], q=(50, 95), nd=0)},
            'subjects': {'type': '合成角色(Blender SMPL-X 角色 + Kling 生成视频 → GEM-SMPL 重建), 无真人',
                         'gender': 'neutral', 'meta_smpl_height_m': (1.66 if c not in ('pickup_table', 'pickup_ground') else None),
                         'recon_pred_height_m_p5_50_95': pct([s['pred_height_m'] for s in sc]),
                         'betas_unique(保留2位)': len(set(tuple(np.round(s['betas'], 2)) for s in sc)),
                         'betas_all_zero': all(not any(s['betas']) for s in sc)},
            'flags_top': dict(collections.Counter(f for r in ok for f in (r.get('flags') or [])).most_common(6)),
            'drop_reasons': dict(collections.Counter(r.get('reason') for r in v if r['status'] == 'dropped')),
            'remaining_download': {'recon_missing_files': len(miss), 'recon_missing_GB': round(sum(hf_recon[k] for k in miss) / 1e9, 3),
                                   'robot_G1_traj_GB(未下, 非人体)': round(rem.get(f'{c}/robot', {}).get('bytes', 0) / 1e9, 3),
                                   'objects_postRL_GB(未下)': round(rem.get(f'{c}/objects', {}).get('bytes', 0) / 1e9, 3),
                                   'video_GB(未下)': round(rem.get(f'{c}/video', {}).get('bytes', 0) / 1e9, 3),
                                   'video_files': rem.get(f'{c}/video', {}).get('n', 0)},
            'license': ('CC BY-NC 4.0; 物体 USD: ' + {'pickup_table': 'RoboCasa 衍生 CC BY 4.0', 'pickup_ground': 'RoboCasa 衍生 CC BY 4.0',
                                                     'sitting': 'Hunyuan3D-2.1 生成, CC BY-NC 4.0', 'stair_p1': 'Hunyuan3D 生成/程序化, CC BY-NC 4.0',
                                                     'stair_p2': '程序化, CC BY-NC 4.0', 'curb': '程序化, CC BY-NC 4.0', 'slope': '程序化, CC BY-NC 4.0'}[c]),
            'status': 'converted_smplx(部分)' if st.get('pending') or st.get('dropped') else 'converted_smplx(全部)',
        }
    tot = {k: round(sum(cats[c]['hours'][k] for c in cats), 3) for k in ['release_est(README帧数/25Hz)', 'smplx_available_on_hf_est', 'converted_ok(24fps)', 'converted_final_v2v3', 'converted_old_v1(待重转)', 'src_valid_of_final_v2v3']}
    return {
        'license': 'CC BY-NC 4.0 (nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL), 非商用',
        'hf_sha': tree.get('_sha'), 'manifest_snapshot': man_mtime,
        'hf_categories_released': list(G_README), 'hf_categories_not_released': 'README: tabletop / ground manipulation(ComAsset/Hunyuan 物体) 计划后续发布; HF data/ 目录目前只有这 7 个',
        'per_category': cats,
        'totals': {'clips_release': sum(v[1] for v in G_README.values()), 'hours': tot,
                   'manifest_status': dict(collections.Counter(r['status'] for r in man)), 'retargeted_umr': 0},
    }, label_hours, label_clips


def main():
    oc = omnicontact()
    gr, g_lab_h, g_lab_c = grail()
    # rollup by top-level taxonomy (original hours; OmniContact per-case labels apply to whole capture)
    roll = collections.defaultdict(lambda: {'omnicontact_hours': 0.0, 'grail_converted_hours': 0.0, 'grail_clips': 0, 'omnicontact_captures': 0, 'sub': set()})
    for k, c in oc['per_category'].items():
        for top, subs in c['taxonomy'].items():
            roll[top]['omnicontact_hours'] += c['hours_original']
            roll[top]['omnicontact_captures'] += c['clips']['converted_smplx']
            roll[top]['sub'] |= set(subs)
    for l, h in g_lab_h.items():
        if '/' not in l:
            roll[l]['grail_converted_hours'] += h
            roll[l]['grail_clips'] += g_lab_c[l]
        else:
            roll[l.split('/')[0]]['sub'].add(l.split('/')[1])
    for top in ['locomotion', 'dynamic', 'ground_contact', 'sitting_on_object', 'upper_body_manipulation', 'whole_body_object', 'gesture_social', 'idle_posture', 'other']:
        roll[top]
    rollup = {k: {**{kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'sub'}, 'sub_labels': sorted(v['sub'])} for k, v in roll.items()}
    gsub = {l: {'clips': g_lab_c[l], 'hours': round(g_lab_h[l], 3)} for l in sorted(g_lab_h)}
    summary = {'generated_at': NOW, 'scope': 'OmniContact + NVIDIA GRAIL, 只读统计; 无 GPU',
               'outputs': {'summary_json': f'{D}/summary.json', 'summary_md': f'{D}/summary.md',
                           'omnicontact_clip_stats': f'{D}/omnicontact_clip_stats.jsonl', 'grail_clip_stats': f'{D}/grail_clip_stats.jsonl',
                           'hf_remaining_tree': f'{D}/_hf_remaining_tree.json'},
               'omnicontact': oc, 'grail': gr, 'taxonomy_rollup(多标签, 小时可重复计)': rollup, 'grail_sub_label_counts(ok段)': gsub}
    json.dump(summary, open(f'{D}/summary.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(json.dumps({'oc_totals': oc['totals'], 'gr_totals': gr['totals'], 'rollup': rollup}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
