#!/usr/bin/env python3
"""Render the Chinese markdown report from hiphi_diversity_summary.json + CSVs."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import json
import pandas as pd

OUT = f'{SRCFG.data_root}/diversity/hiphi'
s = json.load(open(f"{OUT}/hiphi_diversity_summary.json"))
lu = pd.read_csv(f"{OUT}/hiphi_lu_mapping.csv")
c = pd.read_csv(f"{OUT}/hiphi_clips_labeled.csv", dtype={"part": str})
c["h"] = c.duration_sec / 3600

CN = {"locomotion": "移动步态", "dynamic": "高动态", "ground_contact": "地面接触",
      "sitting_on_object": "坐物体", "upper_body_manipulation": "上肢操作物体",
      "whole_body_object": "全身搬/推/踢物体", "gesture_social": "手势/社交/情绪",
      "idle_posture": "站姿/姿态/拉伸", "other": "其他"}
SUBCN = {"turn_curve": "转弯/绕圈/8字", "in_place_body_isolation": "原地局部身体动作(摇头/甩手/扭胯等)",
         "sidestep": "侧移步", "backward": "倒退走", "jump_hop": "跳/单脚跳/蹦", "run": "跑",
         "stumble": "踉跄/绊脚(扰动步态)", "dance": "具名舞蹈", "kick": "踢(含踢物体)",
         "kick_object": "踢箱子/球", "emote": "情绪(发抖/畏缩)", "getup": "从地上起身",
         "martial": "武术/格斗", "in_place_stepping": "原地踏步", "fall": "摔倒/倒地",
         "acrobatic": "体操/特技(空翻/侧手翻/倒立)", "drop_to_crouch": "跌落成低蹲",
         "platform_step": "踏上/跨过长凳(类台阶)", "step_onto_object": "脚踩物体"}
CONCN = {"feet_ground": "地面只有脚(可能同时手碰物体)", "object_hands": "手-物体", "object_body": "躯干/肩/臂-物体",
         "knees": "膝盖着地", "object_feet": "脚-物体(踢/踩)", "object_hips": "臀-物体(坐)",
         "hips_ground": "臀着地(坐地)", "hands_ground": "手撑地", "back": "仰躺",
         "front": "俯趴", "body_ground_unspecified": "倒地(部位未写明)", "side": "侧躺"}

T = s["totals"]; st = s["status"]
L = []
w = L.append
w("# HiPHI 数据丰富度盘点(只读统计)\n")
w(f"生成时间 {s['generated']}。数据源: {s['source']}。\n")
w("## 一句话结论\n")
gc = s["ground_contact"]
mp = s["classes_multilabel"]; pp = s["classes_primary_partition"]
w(f"- HiPHI 原片 **{T['hours_orig']} h**(加镜像 {T['hours_with_mirror']} h), {T['n_clips_orig']} 条, 132 名演员, 214 个动作词条。")
w(f"- 结构很偏: **走路类 + 搬推物体** 占大头。按主类分: 移动步态 {pp['locomotion']['hours_orig']} h({pp['locomotion']['share_of_orig_hours_pct']}%), "
  f"全身搬/推/踢物体 {pp['whole_body_object']['hours_orig']} h({pp['whole_body_object']['share_of_orig_hours_pct']}%), 上肢操作物体 {pp['upper_body_manipulation']['hours_orig']} h({pp['upper_body_manipulation']['share_of_orig_hours_pct']}%)。")
w(f"- 稀缺的: 地面接触只有 **{gc['hours_orig']} h**(2.6%), 坐物体 {mp['sitting_on_object']['hours_orig']} h, 具名舞蹈 {s['subtags']['dance']['hours_orig']} h, "
  f"武术 {s['subtags']['martial']['hours_orig']} h, 体操特技 {s['subtags']['acrobatic']['hours_orig']} h, 摔倒 {s['subtags']['fall']['hours_orig']} h, 起身 {s['subtags']['getup']['hours_orig']} h。没有楼梯/斜坡(全是平地)。")
w(f"- 物体: 每条 HOI 只有 1 个物体, 12 类, 椅子/桌子/箱子 三类占 HOI 的 {round(100*(s['hoi_by_object']['chair']['hours_orig']+s['hoi_by_object']['table']['hours_orig']+s['hoi_by_object']['box']['hours_orig'])/T['hoi_hours_orig'],1)}%。桌子是 60 cm 左右的小边桌, 不是大桌。")
w(f"- 进度: 32 个压缩包只到手 2 个完整(0002, 0029) + 0001 在下(状态快照 {st['tar_partial'].get('0001','')}); "
  f"已转 SMPL-X **{st['converted_smplx_orig_clips']} 条 / {st['converted_smplx_hours_orig']} h 原片**(占 6.2%), 镜像 0 条, **重定向到机器人 0 条**。")
w(f"- 已转的 19.1 h **全是 HOI**(搬推物体), 纯身体动作(走路/舞蹈/武术/地面/手势)一条都还没有: 下载顺序是按 HOI 优先排的。")
w(f"- 想先拿到多样性: **part 0032 只有 2.6 GB, 却装着几乎全部舞蹈(2.94/3.18 h)、全部武术(1.43/1.43 h)和大部分体操特技(0.55/0.77 h)**; part 0027 + 0030 装着地面接触的 30%(2.41/8.11 h)。建议把这三个包提前。")
w(f"- 许可: 非商业(ModalityNet Open Research License), 禁止再分发和公开镜像, 禁止拿来做替代 HiPHI 的商用动作生成模型。商用要单独找 licensing@modalitynet.com。\n")

w("## 1. 总量\n")
w("| 项 | 数 |\n|---|---:|")
w(f"| 原片时长 | {T['hours_orig']} h ({T['n_clips_orig']} 条) |")
w(f"| 镜像时长 | {T['hours_mirror']} h ({T['n_clips_mirror']} 条, 左右镜像, 每条原片一条) |")
w(f"| 合计 | {T['hours_with_mirror']} h (官方 617.5 h) |")
w(f"| HOI(带物体) 原片 | {T['hoi_hours_orig']} h ({T['hoi_clips_orig']} 条) |")
w(f"| 纯身体 原片 | {T['body_only_hours_orig']} h |")
w(f"| 帧率 | 90 Hz (实测 {s['fps']['measured_min']}–{s['fps']['measured_max']}) |")
w(f"| 单条长度 | 中位 {T['clip_duration_sec']['median']} s, 最短 {T['clip_duration_sec']['min']} s, 最长 {T['clip_duration_sec']['max']} s |")
w(f"| 语义 | 22 个 Frame, 214 个 Frame-LU, {T['unique_text_annotations']} 条不重复的英文描述 |")
w(f"| 格式 | {s['format']} |")
w(f"| 许可 | {s['license']} |\n")

w("## 2. 九大类时长(原片 / 加镜像)\n")
w("多标签 = 一条片子可以同时属于几类(比如边走边搬箱子 = 移动步态 + 全身搬物体), 所以加起来超过 100%。"
  "主类 = 按优先级 地面接触 > 高动态 > 坐物体 > 全身物体 > 上肢物体 > 移动 > 手势 > 姿态 > 其他 只取一个, 加起来正好 308.7 h。\n")
w("| 类 | 多标签 原片 h | 加镜像 h | 占比 | 主类 原片 h | 主类占比 | 条数(多标签) | 演员数 | 身高范围 cm | 男/女 h | 其中 HOI h | 已转 SMPL-X h | 下载中(0001) h | 未下载 h | 重定向 |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
for k in mp:
    v = mp[k]; p = pp[k]
    g = v["hours_by_gender"]
    sh = v["status_hours_orig"]
    w(f"| {k} {CN[k]} | {v['hours_orig']} | {v['hours_with_mirror']} | {v['share_of_orig_hours_pct']}% | {p['hours_orig']} | {p['share_of_orig_hours_pct']}% | "
      f"{v['n_clips_orig']} | {v['n_actors']} | {v['actor_height_cm_range'][0]}–{v['actor_height_cm_range'][1]} | "
      f"{g.get('male',0)}/{g.get('female',0)} | {v['hoi_hours_orig']} | {sh.get('converted_smplx',0)} | {sh.get('downloading',0)} | {sh.get('not_downloaded',0)} | 0 |")
w("")
w("各类最大的词条(多标签口径):\n")
for k in mp:
    w(f"- **{CN[k]}**: " + ", ".join(f"{a} {b}h" for a, b in mp[k]["top_frame_lus_by_hours"][:6]))
w("")

w("## 3. 动作细项(子标签, 原片)\n")
w("| 细项 | 原片 h | 加镜像 h | 条数 |\n|---|---:|---:|---:|")
for k, v in s["subtags"].items():
    w(f"| {SUBCN.get(k,k)} (`{k}`) | {v['hours_orig']} | {v['hours_with_mirror']} | {v['n_clips_orig']} |")
w("\n说明: 跑步只有 5.4 h, 而且多是来回折返跑/8 字跑, 没有长距离直线跑。楼梯/斜坡/路沿 = 0。唯一的'上台阶'类是踏上/跨过长凳 0.34 h。\n")

w("## 4. 接触类型(原片)\n")
w("| 接触 | 原片 h | 加镜像 h | 条数 |\n|---|---:|---:|---:|")
for k, v in s["contacts"].items():
    w(f"| {CONCN.get(k,k)} (`{k}`) | {v['hours_orig']} | {v['hours_with_mirror']} | {v['n_clips_orig']} |")
w(f"| 严格'只有脚着地且不碰物体' | {s['contacts_feet_only_no_object']['hours_orig']} | {round(2*s['contacts_feet_only_no_object']['hours_orig'],2)} | |")
w("\n接触是从文字描述推出来的, 不是从动作数据算的。'深蹲贴地走'这种只算脚着地。\n")

w("## 5. 地面接触明细\n")
w(f"合计 **{gc['hours_orig']} h 原片 / {gc['hours_with_mirror']} h 加镜像**, {gc['n_clips_orig']} 条, {gc['n_actors']} 名演员。已转 SMPL-X 只有 {gc['status_hours_orig'].get('converted_smplx',0)} h。\n")
w("| 着地部位 | 原片 h |\n|---|---:|")
for k, v in gc["by_contact_hours_orig"].items():
    w(f"| {CONCN.get(k,k)} | {v} |")
w("\n来源词条(原片 h): " + ", ".join(f"{a} {b}" for a, b in gc["top_frame_lus"]) + "\n")
w("注意: 下载脚本 hiphi_part_priority.csv 里的 gc_h 用的是另一套口径(数值不同, 例如 part 0001 记 0.124 h, 本表按文字只认出 0.02 h), 以本表为准时请说明口径。\n")

w("## 6. HOI 按物体(原片)\n")
w("| 物体 | 条数 | 原片 h | 加镜像 h | 演员数 | 词条数 | 主类拆分 h | 已转 h |")
w("|---|---:|---:|---:|---:|---:|---|---:|")
for k, v in s["hoi_by_object"].items():
    ph = ", ".join(f"{CN.get(a,a)} {b}" for a, b in v["primary_hours"].items())
    w(f"| {k} | {v['n_clips_orig']} | {v['hours_orig']} | {v['hours_with_mirror']} | {v['n_actors']} | {v['n_frame_lus']} | {ph} | {v['converted_hours_orig']} |")
w("\n物体尺寸(网格包围盒): 球 22–24 cm; 箱子 30–72 cm(H/F 型 30–37 cm, A/I/K 型 58–72 cm); 椅子 40–94 cm; 桌子 ~60×50×60 cm 小边桌; 长凳 61 cm; 踏凳 28–66 cm; 衣架 125–148 cm 高; 拖把 137 cm; 水桶 30 cm; 瓶子 34 cm。全部 40 个网格 + 40 个镜像网格都在本地 object_meshes/。")
w("全身/上肢的分界规则: 大物体(椅/桌/箱/踏凳/长凳/衣架/垃圾桶)被搬、推、拉、拖、翻倒, 或者用脚踢球/箱, 算全身; 原地转、摇、晃、倒空、扔接球、拖把、水桶、瓶子算上肢。\n")

a = s["actors"]
w("## 7. 演员多样性\n")
w(f"- {a['n_actors']} 人: 男 {a['gender']['male']}, 女 {a['gender']['female']}。身高 {a['height_cm']['min']}–{a['height_cm']['max']} cm(中位 {a['height_cm']['median']}, 男中位 {a['height_by_gender_median']['male']}, 女中位 {a['height_by_gender_median']['female']}), 体重 {a['weight_kg']['min']}–{a['weight_kg']['max']} kg(中位 {a['weight_kg']['median']})。")
w(f"- 时长按性别: 男 {a['hours_by_gender_orig']['male']} h, 女 {a['hours_by_gender_orig']['female']} h(女占 {round(100*a['hours_by_gender_orig']['female']/T['hours_orig'],1)}%)。")
w(f"- 每人时长: 最少 {a['hours_per_actor_orig']['min']} h, 中位 {a['hours_per_actor_orig']['median']} h, 最多 {a['hours_per_actor_orig']['max']} h; 前 10 人贡献 {a['top10_actor_share_pct']}%。")
w(f"- 每个词条中位 {a['median_actors_per_frame_lu']} 人; {a['frame_lus_with_ge10_actors']} 个词条 ≥10 人, {a['frame_lus_with_le3_actors']} 个词条 ≤3 人(多是小 HOI 词条, 如 Cause_motion-draw/haul 只有 1 人)。")
w(f"- HOI 类演员明显更少: 全身物体 {mp['whole_body_object']['n_actors']} 人, 上肢物体 {mp['upper_body_manipulation']['n_actors']} 人, 坐物体 {mp['sitting_on_object']['n_actors']} 人; 拖把只有 4 人, 足球 2 人。纯身体类 85–113 人。\n")

w("## 8. 下载 / 转换 / 重定向 状态\n")
ss = st["streamc_snapshot"]
w(f"- 压缩包: 32 个。完整在盘: {', '.join(st['tar_complete_on_disk'])}; 正在下: " + ", ".join(f"{k} ({v})" for k, v in st['tar_partial'].items()) + f"; 已解压: {', '.join(st['extracted_parts'])}(0029 解压目录已清理)。")
w(f"- 下载器快照 {ss['t']}: 速度约 {ss['our_rate_MBps_10m']} MB/s, 还剩 {ss['remaining_download_GB']} GB, 估计 {ss['eta_all_parts_h']} h 下完, 但脚本截止时间只剩 {ss['deadline_in_h']} h —— **按现在速度下不完**。")
w(f"- SMPL-X(CPU 管线): {st['converted_smplx_orig_clips']} 条原片 = {st['converted_smplx_hours_orig']} h, 日志 {st['converted_ok_in_log']} 条全部 ok; 镜像转了 {st['converted_smplx_mirror_clips']} 条。目录 {st['converted_dir']}。")
w(f"- GPU 转换管线在 bench4 上重跑了同样的 {st['gpu_bench4_clips_same_set']} 条, 统计时 parity.py 正在对比 GPU 和 CPU 结果。")
w(f"- 重定向到机器人: **0 条, 0 台**(out/ 与 runs/ 里没有 HiPHI 产物)。\n")
w("状态按原片时长: " + ", ".join(f"{k} {v} h ({st['status_clips_orig'][k]} 条)" for k, v in st["status_hours_orig"].items()) + "\n")
w("### 每个压缩包装了什么(原片 h, 按主类)\n")
piv = c.pivot_table(index="part", columns="primary", values="h", aggfunc="sum", fill_value=0)
prio = {p["part"]: p for p in st["parts"]}
cols = ["locomotion", "dynamic", "ground_contact", "whole_body_object", "upper_body_manipulation", "sitting_on_object", "gesture_social", "idle_posture", "other"]
w("| 下载序 | part | 状态 | 压缩 GB | 原片 h | HOI h | " + " | ".join(CN[x] for x in cols) + " | 已转条数 |")
w("|---:|---|---|---:|---:|---:|" + "---:|" * len(cols) + "---:|")
for p in sorted(prio, key=lambda x: prio[x]["rank"]):
    r = prio[p]
    stt = {"complete": "在盘", "partial": "下载中", "absent": "未下"}[r["tar_on_disk"]]
    if r["converted_orig"]:
        stt += "+已转"
    vals = " | ".join(f"{piv.loc[p, x]:.2f}" if x in piv.columns else "0" for x in cols)
    w(f"| {r['rank']} | {p} | {stt} | {r['tar_GB_total']} | {r['orig_h']} | {r['hoi_h']} | {vals} | {r['converted_orig']} |")
w("")
w("读法: 下载序 1–10 的包几乎 100% 是 HOI; 走路/手势/姿态 集中在 0006–0019(每包 ~10 h 纯身体, 其中 ~7 h 走路); 舞蹈+武术+特技 几乎全在 0032; 地面接触最多的是 0030 和 0027。\n")

w("## 9. 有歧义的词条(分类时要看文字, 不能只看词条名)\n")
w("| 词条 | 说明 |\n|---|---|")
for k, v in s["ambiguous_frame_lus"].items():
    w(f"| {k} | {v} |")
w("")

w("## 10. 214 个词条 → 分类映射(按原片时长排序)\n")
w("主类(占比) = 该词条按时长占比最高的主类。多标签列给出各类覆盖的条数比例。完整字段见 hiphi_lu_mapping.csv。\n")
w("| Frame-LU | 原片 h | 条数 | 演员 | HOI% | 物体 | 主类(占比) | 多标签 | 已转/在盘 |")
w("|---|---:|---:|---:|---:|---|---|---|---:|")
for _, r in lu.iterrows():
    obj = "" if r.objects == f"-:{r.n_orig}" else str(r.objects).replace(";", " ")
    ml = " ".join(x for x in str(r.multi_labels).split(";")[:4])
    w(f"| {r.frame_lu} | {r.hours_orig} | {r.n_orig} | {r.n_actors} | {int(round(100*r.hoi_frac))} | {obj} | {CN[r.primary_main]} ({r.primary_main_frac}) | {ml} | {r.converted_orig}/{r.on_disk_or_converted_orig} |")
w("")

w("## 11. 方法和局限\n")
w("- 只读了元数据(hiphi_metadata.csv / actor / frame_lu_index / motion_to_part / archive_manifest)、下载器状态 json、已转目录的文件名和 40 个 OBJ 网格的包围盒。没碰 GPU, 没下载任何东西。")
w("- 分类 = 词条默认类 + 英文描述关键词规则(脚本 hiphi_diversity.py), 再人工抽查过: 地面接触的全部 27 个来源词条逐条看过; 坐物体、跳跃/踢、全身/上肢 各抽 15–25 条看过, 修掉了'shoulder rolls 被当成滚翻'、'how it sits 被当成坐'、'KPOP_Monster 下划线没匹配上舞蹈'等误判。")
w(f"- 规则有边界误差: 全身 vs 上肢物体 的线是人为定的; '原地局部身体动作' 在给定分类里没有合适的位置, 统一放进'其他'({s['classes_multilabel']['other']['hours_orig']} h), 它们是摇头、甩手、扭胯、耸肩之类, 对上半身表现力有用。")
w("- 接触类型完全从文字推断, 不是从动作数据测的。镜像时长 = 原片时长(逐条一一对应, 已核对总和相等)。")
w(f"\n## 输出文件\n")
for k, v in s["outputs"].items():
    w(f"- {k}: `{v}`")
open(f"{OUT}/hiphi_diversity_report.md", "w").write("\n".join(L) + "\n")
print("written", len(L))
