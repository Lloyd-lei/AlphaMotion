from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
# summary.json -> summary.md (中文表格)
import json

D = f'{SRCFG.data_root}/diversity/omnicontact_grail'
S = json.load(open(f'{D}/summary.json'))
oc, gr = S['omnicontact'], S['grail']
TAXZH = {'locomotion': '行走类', 'dynamic': '动态', 'ground_contact': '贴地', 'sitting_on_object': '坐物体',
         'upper_body_manipulation': '上肢操作', 'whole_body_object': '全身+物体', 'gesture_social': '手势社交',
         'idle_posture': '站立/调整', 'other': '其他'}


def tax(d):
    return '; '.join(f"{k}({','.join(v)})" for k, v in d.items())


L = []
L.append('# OmniContact + GRAIL 数据丰富度盘点')
L.append('')
L.append(f"生成时间 {S['generated_at']}; GRAIL manifest 快照 {gr['manifest_snapshot']} (转换还在跑, 数字会涨); 只读统计, 没用 GPU。")
L.append('')
L.append('## 一句话')
t = gr['totals']['hours']
L.append(f"- OmniContact: 690 段真人动捕, {oc['totals']['hours_original']} 小时(90 fps), 全部已转 SMPL-X; 内容只有 搬箱/推箱/脚推箱/踢球/捡球 5 种, 每段都带物体。")
L.append(f"- GRAIL: 发布 22,312 段 ≈ {t['release_est(README帧数/25Hz)']} 小时, 其中有人体 SMPL-X 的 ≈ {t['smplx_available_on_hf_est']} 小时; 已转 {gr['totals']['manifest_status'].get('ok')} 段 = {t['converted_ok(24fps)']} 小时(新版 v2/v3 {t['converted_final_v2v3']} h, 旧版 v1 {t['converted_old_v1(待重转)']} h 待重转)。全是合成角色, 体型参数全为 0(只有一个身材)。")
L.append(f"- 两个来源加起来已经 retarget 到机器人的: 只有 OmniContact 2 段 QA 试跑窗口(最多 6 台), GRAIL 为 0。离 21 台全量还差全部。")
L.append('')
L.append('## 分类别明细')
L.append('')
L.append('| 来源/类别 | 统一分类(多标签) | 段数: 发布 / 盘上原始 / 已转SMPL-X / 已retarget | 小时: 原始(已转) / 镜像 | fps | 物体 | 接触类型 | 交互帧比 | 状态 |')
L.append('|---|---|---|---|---|---|---|---|---|')
for k, c in oc['per_category'].items():
    cl = c['clips']
    objs = ', '.join(f'{o}×{n}' for o, n in c['objects'].items())
    it = c['interaction_frac']
    itxt = (f"G1标签≈{it['g1_label_obj_contact_approx(max手+max脚)']}" if it.get('g1_label_obj_contact_approx(max手+max脚)') is not None else '标签未下') + f"; 人-物<2cm {it['human_obj_within_2cm_frac_p50']}"
    rt = cl['retargeted_umr_captures']
    rtxt = f"{rt}(QA窗口, {len(next(iter(cl['retargeted_umr_detail'].values())))}台)" if rt else '0'
    L.append(f"| OmniContact {k} {c['name_zh']} | {tax(c['taxonomy'])} | {cl['source_captures_release']}(切片{cl['task_clips_release_npz_clips']}) / {cl['on_disk_raw']} / {cl['converted_smplx']} / {rtxt} | {c['hours_original']} / 0 | 90 | {objs} | {', '.join(c['contact_types'])} | {itxt} | 已转SMPL-X |")
for k, c in gr['per_category'].items():
    cl, h = c['clips'], c['hours']
    labs = c['taxonomy_label_clip_counts(ok段, 多标签)']
    top = '; '.join(f"{l}:{n}" for l, n in labs.items() if '/' in l)
    o = c['objects']
    if 'n_classes' in o:
        otxt = f"{o['n_unique_instances']} 个物体/{o['n_classes']} 类(最大边 p50 {o['size_max_extent_m_p5_50_95'][1]} m)"
    elif 'classes(clips)' in o and k == 'sitting':
        otxt = f"{o['n_unique_instances']} 把椅子: " + ', '.join(f'{a}{b}' for a, b in list(o['classes(clips)'].items())[:6])
    elif 'classes(clips)' in o:
        otxt = f"{o['n_unique_instances']} 个楼梯: " + ', '.join(f'{a}{b}' for a, b in o['classes(clips)'].items()) + f"; 上下往返 {o['updown_clips']}"
    elif 'curb_height_m_p5_50_95' in o:
        otxt = f"200 个路沿, 高 {o['curb_height_m_p5_50_95'][0]}-{o['curb_height_m_p5_50_95'][2]} m"
    else:
        otxt = f"200 个斜坡, 坡度 p50 {o['slope_rise_over_run_deg_p5_50_95'][1]}° (p95 {o['slope_rise_over_run_deg_p5_50_95'][2]}°)"
    it = c['interaction_frac(interaction_frames/frames)']
    st = f"已转 {cl['manifest_ok']}, 待下/待转 {cl['manifest_pending']}, 丢弃 {cl['manifest_dropped']}"
    L.append(f"| GRAIL {k} {c['desc_zh']} | {top} | {cl['release']} / {cl['on_disk_raw_recon']} / {cl['manifest_ok']} / 0 | 发布≈{h['release_est(README帧数/25Hz)']} (已转 {h['converted_ok(24fps)']}; 新版 {h['converted_final_v2v3']}) / 0 | 24 | {otxt} | {', '.join(c['contact_types'])} | {it['mean']} | {st} |")
L.append('')
L.append('交互帧比: GRAIL 用官方 interaction_frames 区间 / 总帧; OmniContact 用 G1 物体接触标签(只有箱子类已下)和人体-物体 2cm 内帧比。')
L.append('')
L.append('## 按统一分类汇总(多标签, 同一段可算进多类)')
L.append('')
L.append('| 统一分类 | OmniContact 小时 | GRAIL 已转小时 | GRAIL 段数 | 出现的子类 |')
L.append('|---|---|---|---|---|')
R = S['taxonomy_rollup(多标签, 小时可重复计)']
for k in ['locomotion', 'dynamic', 'ground_contact', 'sitting_on_object', 'upper_body_manipulation', 'whole_body_object', 'gesture_social', 'idle_posture', 'other']:
    v = R[k]
    L.append(f"| {k} {TAXZH[k]} | {v['omnicontact_hours']} | {v['grail_converted_hours']} | {v['grail_clips']} | {', '.join(v['sub_labels']) or '无'} |")
L.append('')
g = S['grail_sub_label_counts(ok段)']
L.append(f"GRAIL 行走子类(已转段): 上行 {g['locomotion/ascend']['clips']} 段 {g['locomotion/ascend']['hours']} h, 下行 {g['locomotion/descend']['clips']} 段 {g['locomotion/descend']['hours']} h, 平地或上去再下来 {g['locomotion/level_or_up-down']['clips']} 段, 转向>60° {g['locomotion/turn']['clips']} 段, 跑(速度p95>2.2m/s) 只有 {g['locomotion/run']['clips']} 段。")
L.append('')
L.append('## 人的多样性')
s = oc['subjects']
L.append(f"- OmniContact: 真人, BVH 骨架校准 {s['bvh_skeleton_calibrations']} 套, 身高 {s['stature_cm_min_max'][0]}-{s['stature_cm_min_max'][1]} cm, 按身高聚类约 {s['estimated_actors']} 人(估计, 元数据没有演员 ID); 性别未提供。足球三种踢球各只有 1-2 套骨架。")
L.append('- GRAIL: 合成角色, 无真人。所有段 betas 全为 0(一个中性身材), gender=neutral, meta 身高 1.66 m; 重建估计身高 1.60-1.87 m 只是估计噪声, 不代表体型多样。')
L.append('')
L.append('## 还没下的')
rem = {k: c['remaining_download'] for k, c in gr['per_category'].items()}
miss_n = sum(v['recon_missing_files'] for v in rem.values())
miss_gb = round(sum(v['recon_missing_GB'] for v in rem.values()), 2)
L.append(f"- GRAIL 人体 recon 还差 {miss_n} 个文件 ≈ {miss_gb} GB: " + ', '.join(f"{k} {v['recon_missing_files']} 个/{v['recon_missing_GB']} GB" for k, v in rem.items() if v['recon_missing_files']) + '。下载循环 grail_dl_loop.py 正在跑。下完能多出约 ' + str(round(t['smplx_available_on_hf_est'] - t['converted_ok(24fps)'], 1)) + ' 小时(大部分是楼梯)。')
L.append('- GRAIL pickup_ground 有 1527 段官方就没放 SMPL-X recon(只有 G1 轨迹和物体), 人体管线永远拿不到, 只剩 209 段 0.88 h。')
L.append(f"- GRAIL 其他目录(不是人体, 可选): G1 轨迹 robot/ {round(sum(v['robot_G1_traj_GB(未下, 非人体)'] for v in rem.values()), 2)} GB, 物体 RL 后轨迹 objects/ {round(sum(v['objects_postRL_GB(未下)'] for v in rem.values()), 2)} GB, 视频 video/ {round(sum(v['video_GB(未下)'] for v in rem.values()), 1)} GB。")
L.append('- GRAIL 官方说 桌面操作 / 地面大物体操作 两个类别"计划后续发布", HF 上目前没有; 没有别的类别可下。')
for k, v in oc['remaining_download'].items():
    L.append(f'- OmniContact {k}: {v}')
L.append('')
L.append('## 要注意的')
L.append('- GRAIL 已转的楼梯里: stair_p1 基本是下楼(931/1306 段), stair_p2 全是上楼(1261/1311); 只拿一部分会偏。已转部分是按文件名顺序下的, 未必代表全体。')
L.append('- GRAIL 的 interaction_frames 在 sitting/楼梯/坡/路沿 上多半是固定的片尾 30 帧(比值 0.124), 不是真实接触区间, 只有 pickup 两类有参考价值。')
L.append('- GRAIL pickup_table 人几乎不走(位移 p50 0.27 m), 是原地抓起举起, 不是搬运走路。')
L.append('- GRAIL 已转里有 9.5 h 是旧版 v1 转换, 要等 v2/v3 重转; 新版里源数据有效帧(不穿地不悬空)占 ' + f"{t['src_valid_of_final_v2v3']}/{t['converted_final_v2v3']} h。")
L.append('- OmniContact 的 qa.json 里 seconds 字段是转换耗时, 不是片长; 片长按 帧数/90 算。足球类接触类型是推断(G1 标签没下)。')
L.append('- OmniContact 搬箱子里有 ' + str(oc['per_category']['box/case1_carry']['motion']['captures_with_deep_squat_gt45cm_ge1pct_frames']) + ' 段出现骨盆下降>45 cm 的深蹲(从地上搬箱), 是否跪或手撑地没核实(没跑 FK), 先没算进 ground_contact。')
L.append('- 两个来源都没有: 跑、跳、跳舞、摔倒/起身、爬、躺、手势、站立调整。dynamic 只有踢球。镜像数据为 0。')
L.append('- 许可都是 CC BY-NC 4.0(非商用); GRAIL 的 pickup 物体另有 RoboCasa CC BY 4.0 署名要求。')
L.append('')
L.append('## 文件')
for k, v in S['outputs'].items():
    L.append(f'- {k}: `{v}`')
L.append(f"- 脚本: `{D}/scan_grail.py`, `{D}/scan_omnicontact.py`, `{D}/aggregate.py`, `{D}/make_md.py`")
open(f'{D}/summary.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
