import json, os
D = "/home/arenalabs/umr_factory_data/diversity"
CN = {"locomotion": "移动", "dynamic": "动态", "ground_contact": "地面接触", "sitting_on_object": "坐物体",
      "upper_body_manipulation": "上身操作", "whole_body_object": "全身+物体", "gesture_social": "手势社交",
      "idle_posture": "站姿/调整", "other": "其他"}
CTN = {"feet_only": "只脚", "hands_ground": "手撑地", "knees": "膝", "hips": "臀", "back": "背",
       "object_hands": "手-物体", "object_feet": "脚-物体/台阶"}
C = list(CN)
f = lambda x: (f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else (f"{x:.2f}" if isinstance(x, float) else str(x)))


def tbl(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        s += "| " + " | ".join(f(x) for x in r) + " |\n"
    return s


def class_rows(t):
    return [[f"{CN[c]} ({c})", t[c]["primary_hours"], t[c]["primary_share_pct"], t[c]["primary_clips"],
             t[c]["multilabel_hours"]] for c in C]


HEAD = ["类别", "主类小时", "占比%", "主类片段数", "多标签小时(会重复)"]

# ---------- BONES ----------
b = json.load(open(f"{D}/bones_seed/bones_summary.json"))
md = f"# BONES-SEED 丰富度盘点\n\n"
md += f"- 片段 {b['clips']['total']:,}（原始 {b['clips']['originals']:,} / 镜像 {b['clips']['mirrors']:,}），{b['fps']} fps\n"
md += f"- 小时：原始 **{b['hours']['originals']} h** / 镜像 {b['hours']['mirrors']} h / 合计 {b['hours']['total']} h；片段中位 {b['clip_seconds_originals']['median']} s, p90 {b['clip_seconds_originals']['p90']} s\n"
a = b["actors"]
md += f"- 演员 {a['count']} 人（{a['gender']}），身高 {a['height_cm']['min']}–{a['height_cm']['max']} cm（p10 {a['height_cm']['p10']} / 中位 {a['height_cm']['median']} / p90 {a['height_cm']['p90']}），体重 {a['weight_kg']['min']}–{a['weight_kg']['max']} kg，年龄 {a['age_yr']['min']}–{a['age_yr']['max']}；职业 {a['profession']}\n"
md += f"- 风格（原始小时）：{b['style_hours_originals']}\n\n"
md += "## 分类（原始，不含镜像；镜像分布几乎一样）\n\n" + tbl(HEAD, class_rows(b["taxonomy_originals"]))
md += "\n## 接触类型（原始小时，一段可多种）\n\n" + tbl(list(CTN.values()), [[b["contacts_originals"]["hours_any_class"][k] for k in CTN]])
md += "\n## 物体\n\n"
o = b["objects"]
md += f"- 有道具的片段 {o['has_prop_hours_originals']} h（其中被手拿/搬的 {o['manipulated_prop_hours_originals']} h）；**没有物体位姿/网格**：{o['note']}\n"
md += "- 道具类别小时：" + ", ".join(f"{k} {v}" for k, v in o["prop_category_hours_originals"].items()) + "\n"
md += "- 前 25 道具：" + ", ".join(f"{k} {v}" for k, v in o["top_props_hours_originals"].items()) + "\n\n"
md += "## 许可 / 状态 / 进工厂\n\n"
md += f"- 许可：{b['license']['training_use']}\n- 本地：{b['status']['on_disk_raw']}\n- 下载器：{b['status']['downloader']}\n"
md += f"- SMPL-X：{b['status']['converted_smplx']}\n- 重定向：{b['status']['retargeted']}\n- 进 UMR：{b['umr_bringup']}\n"
open(f"{D}/bones_seed/bones_summary.md", "w").write(md)

# ---------- AMASS / LAFAN ----------
x = json.load(open(f"{D}/xwbc_amass_lafan/xwbc_summary.json"))
am, la = x["amass"], x["lafan"]
md = "# AMASS / LAFAN（xwbc 人体流）丰富度盘点\n\n## AMASS\n\n"
md += f"- {am['clips']:,} 段，**{am['hours']} h**，{am['fps']} fps，无镜像；tar 内 {am['tar_npz_members']:,} 个 npz\n"
ls = am["label_sources"]
md += f"- 标签来源：BABEL v1.0 动作类 {ls['babel_v1_clips']:,} 段；HumanML3D 描述 {ls['humanml3d_caption_clips']:,} 段；至少一种文字标注 {ls['any_text_annotation_clips']:,} 段 / {ls['any_text_annotation_hours']} h；只靠片段名 {ls['name_only_clips']:,} 段 / {ls['name_only_hours']} h\n"
md += f"- 演员：{am['actors']['note']}；文件夹数 {am['actors']['subject_folders_total']}\n\n"
md += tbl(HEAD, class_rows(am["taxonomy"]))
md += "\n### 子集\n\n" + tbl(["子集", "片段", "小时", "BABEL", "HumanML3D", "人/会话文件夹", "主类分布(h)"],
                         [[k, v["clips"], v["hours"], v["babel"], v["humanml3d"], v["subject_folders"],
                           ", ".join(f"{CN[c]} {h}" for c, h in sorted(v["primary_hours"].items(), key=lambda z: -z[1]))]
                          for k, v in sorted(am["subsets"].items(), key=lambda z: -z[1]["hours"])])
md += "\n### 接触（小时）\n\n" + tbl(list(CTN.values()), [[am["contacts"]["hours_any_class"][k] for k in CTN]])
md += f"\n- 物体：文字推断有物体 {am['objects']['object_implied_hours']} h（含 GRAB {am['objects']['grab_hours']} h）；{am['objects']['note']}\n"
md += f"- 许可：{am['license']['name']}；{am['license']['training_use']}\n- 本地：{am['status']['on_disk_raw']}\n- 重定向：{am['status']['retargeted']}\n- 已知问题：{am['status']['known_issue']}\n- 进 UMR：{am['umr_bringup']}\n\n"
md += f"## LAFAN1\n\n- {la['clips']} 段，**{la['hours']} h**，{la['fps']} fps，{la['subjects']} 名演员，无镜像\n"
md += "- 动作族小时：" + ", ".join(f"{k} {v}" for k, v in la["action_families"].items()) + "\n\n"
md += tbl(HEAD, class_rows(la["taxonomy"]))
md += "\n" + tbl(list(CTN.values()), [[la["contacts"]["hours_any_class"][k] for k in CTN]])
md += f"\n- 许可：{la['license']['name']}（{la['license']['training_use']}）\n- 本地：{la['status']['on_disk_raw']}\n- 重定向：{la['status']['retargeted']}\n- 进 UMR：{la['umr_bringup']}\n"
open(f"{D}/xwbc_amass_lafan/xwbc_summary.md", "w").write(md)

# ---------- UniML3D ----------
u = json.load(open(f"{D}/uniml3d_mixamo/uniml3d_summary.json"))
mx, ob = u["mixamo"], u["objaverse_bipedal"]
md = "# UniML3D / Mixamo 丰富度盘点\n\n"
md += tbl(["来源/体型", "片段", "小时"], [[k, v["clips"], v["hours"]] for k, v in u["all_sources"].items()])
md += f"\n## Mixamo（唯一干净的人形部分）\n\n- {mx['clips']:,} 段，**{mx['hours']} h**，30 fps；UniMate 训练子集 {mx['unimate_train_subset']['clips']:,} 段 / {mx['unimate_train_subset']['minutes']} 分钟\n- {mx['rig']}\n"
md += f"- 名字里带道具（枪/剑/公文包/火把…）{mx['prop_in_name_clips']} 段 / {mx['prop_in_name_hours']} h，但道具不在动作数据里\n\n"
md += tbl(HEAD, class_rows(mx["taxonomy"]))
md += "\n" + tbl(list(CTN.values()), [[mx["contacts"]["hours_any_class"][k] for k in CTN]])
md += f"\n## Objaverse 'bipedal'（未验证是否人形）\n\n- {ob['clips']:,} 段 / {ob['hours']} h / {ob['rigs']:,} 种骨架（关节 {ob['joints']}）；{ob['humanoid_usable']}\n"
md += "- 主类小时：" + ", ".join(f"{CN[c]} {h}" for c, h in ob["taxonomy_primary_hours"].items() if h) + "\n"
md += f"- {u['truebones_bipedal_note']}\n\n"
md += f"- 许可：Mixamo {u['license']['mixamo']}；Objaverse {u['license']['objaverse']}；描述 {u['license']['captions_metadata']}\n- 本地：{u['status']['on_disk_raw']}\n- 进 UMR：{u['umr_bringup']}\n"
open(f"{D}/uniml3d_mixamo/uniml3d_summary.md", "w").write(md)

# ---------- scout ----------
s = json.load(open(f"{D}/scout_others/scout_summary.json"))
md = "# 其他候选源（只看可得性/许可，未在本机核实）\n\n"
md += tbl(["源", "时长", "内容", "物体", "格式", "获取", "许可", "本机", "进 UMR", "预计主类"],
          [[r["id"], r["hours"], r["content"], r["objects"], r["fmt"], r["access"], r["license"], r["on_disk"], r["umr"], r["taxonomy_guess"]] for r in s["sources"]])
open(f"{D}/scout_others/scout_summary.md", "w").write(md)

# ---------- combined ----------
cb = json.load(open(f"{D}/core_sources/summary.json"))
md = "# 人体动作数据丰富度总表（BONES / AMASS / LAFAN / Mixamo + 其他候选）\n\n"
t = cb["totals"]
md += f"原始合计（不含镜像）**{t['originals_sum_4_sources']} h** = BONES {t['BONES']['orig']} + AMASS {t['AMASS_xwbc']} + LAFAN {t['LAFAN_xwbc']} + Mixamo {t['Mixamo']}；另有 BONES 镜像 {t['BONES']['mirror']} h，Objaverse 'bipedal' {t['Objaverse_bipedal_unverified']} h（未验证人形）。\n\n"
md += "## 1. 按类别的主类小时（每段只算一次，可相加）\n\n"
srcs = list(cb["primary_hours_by_source"])
rows = [[CN[c]] + [cb["primary_hours_by_source"][s][c] for s in srcs] + [cb["primary_hours_total_originals_only"][c]] for c in C]
rows.append(["合计"] + [round(sum(cb["primary_hours_by_source"][s].values()), 2) for s in srcs] + [round(sum(cb["primary_hours_total_originals_only"].values()), 2)])
md += tbl(["类别"] + srcs + ["原始合计(不含镜像)"], rows)
md += "\n## 2. 多标签小时（一段可属多类，不能相加）\n\n"
s2 = list(cb["multilabel_hours_by_source"])
md += tbl(["类别"] + s2, [[CN[c]] + [cb["multilabel_hours_by_source"][s][c] for s in s2] for c in C])
md += "\n## 3. 接触类型小时\n\n"
s3 = list(cb["contact_hours_by_source"])
md += tbl(["接触"] + s3, [[CTN[k]] + [cb["contact_hours_by_source"][s][k] for s in s3] for k in CTN])
oh = cb["object_hours"]
md += f"\n## 4. 物体\n\n- BONES 有道具 {oh['BONES_prop_text']} h（手拿/搬 {oh['BONES_manipulated']} h），AMASS 推断 {oh['AMASS_implied']} h（GRAB {oh['AMASS_GRAB']} h），Mixamo 名字带道具 {oh['Mixamo_prop_in_name']} h\n- **带物体位姿/网格的小时 = {oh['with_object_pose']}**：这四个源里物体都只是文字，物体接触只能靠 GRAIL / OmniContact / HiPHI / OMOMO 补\n\n"
md += "## 5. 来源总览\n\n"
md += tbl(["源", "原始 h", "镜像 h", "fps", "演员", "许可(训练)", "状态", "进 UMR 要做什么"], [
    ["BONES-SEED", t["BONES"]["orig"], t["BONES"]["mirror"], 120, "522 人(F253/M269), 145–199 cm", "学术或<100万美元初创内部; 禁再分发; 禁训'可替代数据集的动作生成模型'",
     "元数据+10 样例 BVH; 全量 45 GB 未下(排队, 主下载器卡 25.8 h); 厂商 G1 CSV 在盘; UMR 0 台", "下 soma_uniform+shapes → UMR 原生适配器, 无需转换"],
    ["AMASS(xwbc)", t["AMASS_xwbc"], 0, 30, "~文件夹 " + str(json.load(open(f'{D}/xwbc_amass_lafan/xwbc_summary.json'))['amass']['actors']['subject_folders_total']) + "; 无身高/性别", "非商业科研, 禁分发",
     "只有 22 刚体派生数据; xwbc 重定向 18 台; UMR 0 台", "下官方 SMPL-X 版(推荐) 或 有损拟合"],
    ["LAFAN(xwbc)", t["LAFAN_xwbc"], 0, 30, "5 人", "CC BY-NC-ND", "xwbc 18 台; UMR 1 段→3 台", "下 BVH → lafan_to_smplx → UMR(已通)"],
    ["Mixamo", t["Mixamo"], 0, 30, "未知(单 rig)", "Adobe 条款, 有风险", "仅元数据+129 特征样本; UMR 0", "FBX→BVH → NR 适配器(未验证)"],
])
md += "\n## 6. 其他候选（只看可得性/许可）\n\n" + open(f"{D}/scout_others/scout_summary.md").read().split("\n", 2)[2]
md += "\n## 7. 方法与误差\n\n" + "\n".join(f"- {k}: {v}" for k, v in cb["method"].items()) + "\n"
open(f"{D}/core_sources/summary.md", "w").write(md)
print("ok")
