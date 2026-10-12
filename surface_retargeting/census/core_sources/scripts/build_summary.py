import sys, os, json, re, shutil, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import pyarrow.parquet as pq
import taxo

D = "/home/arenalabs/umr_factory_data/diversity"
C = taxo.CLASSES
CT = taxo.CONTACTS
r2 = lambda x: float(round(float(x), 2))
r3 = lambda x: float(round(float(x), 3))


def class_table(df, h="hours"):
    out = {}
    tot = df[h].sum()
    for c in C:
        p = df[df.primary == c]
        m = df[df.classes.str.split("|").apply(lambda s: c in s)]
        out[c] = dict(primary_hours=r2(p[h].sum()), primary_clips=int(len(p)),
                      primary_share_pct=r2(100 * p[h].sum() / tot if tot else 0),
                      multilabel_hours=r2(m[h].sum()), multilabel_clips=int(len(m)))
    return out


def contact_table(df, h="hours"):
    x = df.assign(ct=df.contacts.str.split("|")).explode("ct")
    tot = x.groupby("ct")[h].sum()
    byc = x.groupby(["primary", "ct"])[h].sum()
    return {"hours_any_class": {k: r2(tot.get(k, 0)) for k in CT},
            "hours_by_primary_class": {c: {k: r2(byc.get((c, k), 0)) for k in CT if byc.get((c, k), 0) > 0}
                                       for c in C if c in df.primary.values}}


def obj_table(df, col="object_cats", h="hours"):
    x = df[df[col].fillna("") != ""].assign(oc=lambda d: d[col].str.split("|")).explode("oc")
    return {k: r2(v) for k, v in x.groupby("oc")[h].sum().sort_values(ascending=False).items()}


# ================= BONES =================
B = pd.read_csv(f"{D}/bones_seed/bones_clip_labels.csv.gz")
meta = pq.read_table("/var/tmp/alphamotion/umr_factory/sources/bones_meta_v004.parquet").to_pandas()
bo, bm = B[~B.is_mirror], B[B.is_mirror]
act = meta[~meta.is_mirror].drop_duplicates("actor_uid")
acq = json.load(open('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/pretraining_foundation/logs/acquire_followup/queue_state.json'))
bones = dict(
    source="BONES-SEED (bones-studio/seed, metadata v004)",
    fps=120,
    clips=dict(total=int(len(B)), originals=int(len(bo)), mirrors=int(len(bm))),
    hours=dict(total=r2(B.hours.sum()), originals=r2(bo.hours.sum()), mirrors=r2(bm.hours.sum())),
    clip_seconds_originals=dict(median=r2(bo.hours.median() * 3600), p90=r2(bo.hours.quantile(.9) * 3600),
                                max=r2(bo.hours.max() * 3600)),
    taxonomy_originals=class_table(bo),
    taxonomy_mirrors_primary_hours={c: r2(bm[bm.primary == c].hours.sum()) for c in C},
    contacts_originals=contact_table(bo),
    objects=dict(
        has_prop_hours_originals=r2(bo[bo.has_object].hours.sum()),
        manipulated_prop_hours_originals=r2(bo[bo.object_manipulated].hours.sum()),
        object_pose_or_mesh_available=False,
        note="道具只在文字字段 content_props 里，没有物体 6DoF 位姿/网格；物体接触标签无法用数据工厂的尺子量",
        prop_category_hours_originals=obj_table(bo),
        top_props_hours_originals={k: r2(v) for k, v in bo[bo.props.fillna("") != ""].assign(
            p=lambda d: d.props.str.split(r",\s*")).explode("p").groupby("p").hours.sum()
            .sort_values(ascending=False).head(25).items()}),
    package_category_hours_originals={f"{p}/{c}": r2(v) for (p, c), v in bo.groupby(["package", "category"]).hours.sum().items()},
    style_hours_originals={k: r2(v) for k, v in bo.groupby("style").hours.sum().sort_values(ascending=False).items()},
    actors=dict(count=int(len(act)), gender=act.actor_gender.value_counts().to_dict(),
                hours_by_gender_originals={k: r2(v) for k, v in bo.groupby("gender").hours.sum().items()},
                height_cm=dict(min=int(act.actor_height_cm.min()), p10=r2(act.actor_height_cm.quantile(.1)),
                               median=r2(act.actor_height_cm.median()), p90=r2(act.actor_height_cm.quantile(.9)),
                               max=int(act.actor_height_cm.max())),
                weight_kg=dict(min=int(act.actor_weight_kg.min()), median=r2(act.actor_weight_kg.median()),
                               max=int(act.actor_weight_kg.max())),
                age_yr=dict(min=int(act.actor_age_yr.min()), median=r2(act.actor_age_yr.median()),
                            max=int(act.actor_age_yr.max())),
                profession=act.actor_profession.value_counts().to_dict()),
    license=dict(name="BONES-SEED Dataset License Agreement",
                 training_use="仅学术用户或年营收<100万美元初创公司内部研发可用；禁止再分发/公开镜像；"
                              "Results 不得能还原原始数据；§6 禁止用来训练'主要用途是生成人体动作/动画数据、可替代本数据集'的生成模型"
                              "（明确允许机器人控制策略/VLA/行为克隆/动作跟踪）。AlphaMotion 若定位为'动作生成基础模型'，此条需法务确认。",
                 open_release_of_retargeted=False),
    status=dict(
        on_disk_raw="仅元数据（parquet/csv/temporal_labels v002）+ 10 条样例 BVH；SOMA BVH 全量包 soma_uniform.tar.gz(45.19 GB) 不在本机",
        downloader=f"soma_uniform.tar.gz 排在 follow-up 队列(PID 3272054, runs=[])，还没开始；主下载器(PID 3258020)卡在 UniMate 模型权重阶段 "
                   f"{acq.get('primary_progress_bytes', 0) / 1e9:.1f}/45.9 GB，已 {acq.get('seconds_without_primary_progress', 0) / 3600:.1f} h 无进展（queue status={acq.get('status')}）",
        converted_smplx="不需要（UMR 直接读 SOMA BVH + MHR 体型）；已转 0 条",
        retargeted="厂商自带 G1 CSV（1 台，/var/tmp/arena/raw/bones_seed/g1.tar.gz 23.5 GB，已解压 4 条）；UMR 0 条/0 台；"
                   "另有 8 月旧缓存 topo_exp/cache_bones_body（500 段 soma_body 关节, 30fps, 非 UMR）",
        umr_robots=0),
    umr_bringup="下载 soma_uniform.tar.gz(45.2 GB)+soma_shapes+SOMA-X 资产 → UMR defaults_bones_seed.json/批处理适配器直接跑，无需格式转换；"
                "镜像是独立 BVH 可直接重定向；29.7 h 道具片段没有物体位姿，只能当'纯身体'数据或补物体标注",
)
os.makedirs(f"{D}/bones_seed", exist_ok=True)
json.dump(bones, open(f"{D}/bones_seed/bones_summary.json", "w"), ensure_ascii=False, indent=1)

# ================= AMASS / LAFAN (xwbc) =================
A = pd.read_csv(f"{D}/xwbc_amass_lafan/amass_clip_labels.csv.gz")
Lf = pd.read_csv(f"{D}/xwbc_amass_lafan/lafan_clip_labels.csv")
sub = A.groupby("subset").agg(clips=("clip", "count"), hours=("hours", "sum"), babel=("babel_labelled", "sum"),
                              humanml3d=("humanml3d_captioned", "sum"), subject_folders=("subject", "nunique"))
sub_primary = A.pivot_table(index="subset", columns="primary", values="hours", aggfunc="sum", fill_value=0)
annotated = (A.babel_labelled | A.humanml3d_captioned)
xw_robots = sorted(f[:-7] for f in os.listdir("/var/tmp/alphamotion/xwbc_full/amass") if f.endswith(".tar.gz") and f != "human_motion.tar.gz")
amass = dict(
    source="AMASS 经 xwbc-retarget-data 的人体流 (TonyZhang1999/xwbc-retarget-data, human_motion.tar.gz)",
    fps=30, clips=int(len(A)), tar_npz_members=13283, hours=r2(A.hours.sum()), mirrors_hours=0.0,
    label_sources=dict(babel_v1_clips=int(A.babel_labelled.sum()), humanml3d_caption_clips=int(A.humanml3d_captioned.sum()),
                       any_text_annotation_clips=int(annotated.sum()), any_text_annotation_hours=r2(A[annotated].hours.sum()),
                       name_only_clips=int((~annotated).sum()), name_only_hours=r2(A[~annotated].hours.sum())),
    taxonomy=class_table(A), contacts=contact_table(A),
    objects=dict(object_implied_hours=r2(A[A.has_object_implied].hours.sum()),
                 grab_hours=r2(A[A.subset == "GRAB"].hours.sum()),
                 object_pose_or_mesh_available=False,
                 note="xwbc 人体流只有 22 个 SMPL 刚体位姿；GRAB 原始数据有物体 6DoF+网格，但这里没有",
                 implied_object_category_hours=obj_table(A[A.has_object_implied])),
    subsets={k: dict(clips=int(v.clips), hours=r2(v.hours), babel=int(v.babel), humanml3d=int(v.humanml3d),
                     subject_folders=int(v.subject_folders),
                     primary_hours={c: r2(sub_primary.loc[k].get(c, 0)) for c in C if sub_primary.loc[k].get(c, 0) > 0.005})
             for k, v in sub.iterrows()},
    actors=dict(subject_folders_total=int(A.subject.nunique()),
                note="xwbc 数据无 betas/性别/身高；subject_folders 是子集内的人/会话文件夹数，不等于独立演员数（ACCAD/DanceDB 偏高）"),
    license=dict(name="AMASS License (非商业科研, 禁止分发) + 各子集原许可; BABEL/HumanML3D 标注另有许可",
                 training_use="仅非商业科研；公司内部研发是否覆盖需法务确认", open_release_of_retargeted=False),
    status=dict(on_disk_raw="否：原始 AMASS(SMPL-X/SMPL+H) 已不在本机；只有 xwbc 派生的 22 刚体 pos+quat (5.1 GB tar, 13,283 npz)",
                converted_smplx="0", retargeted=f"xwbc/GMR 已重定向 {len(xw_robots)} 台(tar 在盘): {', '.join(xw_robots)}；"
                                             f"我们的缓存 cache_amass_lconv_tfix 用其中 17 台(去掉 berkeley_humanoid_lite)；UMR 0 台",
                known_issue="xwbc 机器人流 4,560 段时间轴错位(KIT 3,985/CMU 503/TotalCapture 35…)，tfix 缓存已修；人体流本身对",
                umr_robots=0),
    umr_bringup="A(推荐) lloyd 用 AMASS 账号下官方 SMPL-X G 版 → UMR 直接读，无需转换；"
                "B(不下载) 22 关节四元数 → SMPL-X 拟合（无 betas/手指, 有损）；BABEL 逐帧动作标签已在盘(6,002 段带标签)可直接复用",
)
lafan = dict(
    source="LAFAN1 经 xwbc 人体流 (/var/tmp/alphamotion/xwbc_lafan/extracted/human_motion)",
    fps=int(Lf.fps.iloc[0]), clips=int(len(Lf)), hours=r3(Lf.hours.sum()), mirrors_hours=0.0,
    subjects=int(Lf.subject.nunique()), action_families={k: r3(v) for k, v in Lf.groupby("action").hours.sum().items()},
    taxonomy=class_table(Lf), contacts=contact_table(Lf),
    objects=dict(object_implied_hours=0.0, object_pose_or_mesh_available=False,
                 note="obstacles1-6 有越障/攀爬，但障碍物没有几何"),
    license=dict(name="CC BY-NC-ND 4.0", training_use="非商业；ND 禁止分发改编物(含重定向结果)", open_release_of_retargeted=False),
    status=dict(on_disk_raw="原始 BVH 不在本机；有 xwbc 人体流 77 段 + UMR 样例 lafan1_smplx/dance1_subject2.npz",
                converted_smplx="1 段 (dance1_subject2, UMR 自带样例)",
                retargeted=f"xwbc/GMR 18 台(xwbc_full/lafan/*.tar.gz)；UMR: 1 段 dance1_subject2 → 3 台(fourier_gr3/unitree_g1_29dof/unitree_h2, out/) + QA 窗口(g1_23dof/h1/r1)",
                umr_robots=3),
    umr_bringup="从 GitHub 下 LAFAN1 BVH(很小) → lafan_to_smplx(MIT) → UMR，已有成功样例；只是只有 4.6 h",
)
os.makedirs(f"{D}/xwbc_amass_lafan", exist_ok=True)
json.dump(dict(amass=amass, lafan=lafan), open(f"{D}/xwbc_amass_lafan/xwbc_summary.json", "w"), ensure_ascii=False, indent=1)

# ================= UniML3D =================
U = pd.read_csv(f"{D}/uniml3d_mixamo/uniml3d_clip_labels.csv.gz")
mix = U[U.dataset == "mixamo"]
ob = U[(U.dataset == "objaverse") & (U.category == "bipedal")]
clips_all = pd.read_csv("/var/tmp/alphamotion/umr_factory/uniml3d_reload/remote/clips.csv")
anim = pd.read_csv("/var/tmp/alphamotion/umr_factory/uniml3d_reload/remote/objxl_animations.csv")
uni = dict(
    source="UniML3D (Linzhan/UniML3D) = Mixamo + Objaverse-XL 绑骨动画 + Truebones 动物",
    fps=30,
    all_sources={f"{d}/{c}": dict(clips=int(len(g)), hours=r3(g.duration_sec.sum() / 3600))
                 for (d, c), g in clips_all.groupby(["dataset", "category"])},
    mixamo=dict(clips=int(len(mix)), hours=r3(mix.hours.sum()), unimate_train_subset=dict(clips=2162, minutes=105.04),
                rig="单一 65 关节 Mixamo rig（UniML3D 只保留 22 关节人形核心）；无演员信息",
                taxonomy=class_table(mix), contacts=contact_table(mix),
                prop_in_name_clips=int(mix.prop_in_name.sum()), prop_in_name_hours=r3(mix[mix.prop_in_name].hours.sum()),
                object_pose_or_mesh_available=False, local_feature_npz=int(mix.local_feature_npz.sum())),
    objaverse_bipedal=dict(clips=int(len(ob)), hours=r3(ob.hours.sum()), rigs=int(ob.object_type.nunique()),
                           joints=dict(min=int(ob.num_joints.min()), median=r2(ob.num_joints.median()), max=int(ob.num_joints.max())),
                           mixamo_named_animation_share=r3(anim.clip_name.str.lower().str.contains("mixamo").mean()),
                           taxonomy_primary_hours={c: r3(ob[ob.primary == c].hours.sum()) for c in C},
                           humanoid_usable="未验证：'bipedal' 是 VLM 判的体型类，包含卡通/机器人/怪物；27% 动画名是 mixamo.com（Mixamo 动作套在任意角色上，和 Mixamo 重复）"),
    truebones_bipedal_note="Truebones 'bipedal' 是霸王龙/迅猛龙/鸵鸟等 8 种(0.18 h)，不是人形；原始片段需购买",
    license=dict(mixamo="Adobe Mixamo 条款（动作可用于项目，数据集再分发/训练授权不明确 → 有风险）",
                 objaverse="逐物体继承 Objaverse-XL 上游许可，无统一许可",
                 captions_metadata="ODC-BY 1.0", unimate_models="CC BY-NC 4.0",
                 open_release_of_retargeted="有风险/不建议"),
    status=dict(on_disk_raw="元数据(clips.csv/skeletons.csv/README)+HF 缓存冒烟样本: Mixamo 特征 npz 129/2,162, Objaverse 90；"
                            "reload 导出 2 段 + 1.3 GB GLB 样本；全量数据未下载（主下载器在下 UniMate 模型权重, 非数据）",
                converted_smplx="0", retargeted="0", umr_robots=0),
    umr_bringup="Mixamo FBX → BVH(Blender) → UMR 通用 FBX/BVH(NR) 适配器（未验证）；或用 UniML3D 特征(22 关节全局位置+局部旋转@30fps)做 SMPL-X 拟合；"
                "可用的人形部分只有 Mixamo 2.4 h，Objaverse 'bipedal' 需人工/自动筛人形",
)
os.makedirs(f"{D}/uniml3d_mixamo", exist_ok=True)
json.dump(uni, open(f"{D}/uniml3d_mixamo/uniml3d_summary.json", "w"), ensure_ascii=False, indent=1)

# ================= scout others =================
scout = [
    dict(id="OMOMO", hours="约 10 h（论文口径, 未实测）", content="单人搬运/推拉 15 种物体（箱子/椅子/桌子/拖把/台灯…）, 17 人",
         objects="有：物体 6DoF + 15 个网格", fmt="SMPL-X/SMPL-H + 物体位姿", access="Google Drive 公开链接",
         license="代码 MIT；数据无明示许可 → 按保留所有权利对待，风险高", on_disk="否（只有 UMR 样例 README）",
         umr="有 UMR 适配器 (hsi_hoi)，需整理成 UMR 目录；女性序列需 SMPLX_FEMALE（本机缺）",
         taxonomy_guess="whole_body_object 为主 + upper_body_manipulation"),
    dict(id="GRAB", hours="约 3.8 h（AMASS 内 GRAB 身体 3.76 h 已在 xwbc 数据里）", content="10 人抓取 51 种小物体",
         objects="原版有：物体 6DoF + 网格 + MANO 手；xwbc 版没有", fmt="SMPL-X + MANO + 物体", access="MPI 账号注册",
         license="GRAB 非商业科研许可（MPI）", on_disk="仅身体（xwbc 派生）", umr="下原版 GRAB → hsi_hoi 标准适配（未验证）",
         taxonomy_guess="upper_body_manipulation（站着/坐着抓小物体）"),
    dict(id="ARCTIC", hours="约 1 h 量级（339 段, 未实测）", content="10 人双手操作 11 种带关节物体（剪刀/笔记本/微波炉…）",
         objects="有：带关节物体 + 网格", fmt="SMPL-X + MANO + 物体关节角", access="注册下载",
         license="非商业科研（MPI 风格）", on_disk="否", umr="需写适配器（物体是关节体）",
         taxonomy_guess="upper_body_manipulation（几乎全是上半身）"),
    dict(id="CORE4D", hours="未公布小时；Real 约 1.0K 段 + Synthetic 约 10K 段（未实测）", content="两人协作搬/移物体",
         objects="有：物体网格 + 位姿", fmt="SMPL-X + 物体", access="HF 公开（34 GB 动作 zip）",
         license="HF 标 MIT（未对照论文/代码核实）", on_disk="否", umr="SMPL-X 可读；两人场景需拆成单人 + 物体",
         taxonomy_guess="whole_body_object（双人搬大件）"),
    dict(id="100STYLE", hours="约 4M 帧 @60fps ≈ 18 h（论文口径, 未实测）", content="1 名演员 100 种风格行走/跑/侧移/后退/站立",
         objects="无", fmt="原版 BVH；HF 有 100style_smplx (2.5 GB)", access="Zenodo / HF",
         license="原版 CC BY 4.0（可商用可分发, 需署名）；HF smplx 版无许可标签", on_disk="否（只有 G1 重定向版 any4hdmi-g1-100style）",
         umr="BVH→SMPL-X 或直接用 100style_smplx", taxonomy_guess="locomotion（风格多样性最强, 单演员）"),
    dict(id="Motion-X", hours="约 144 h（81.1K 段, 15.6M 帧, 公布口径）", content="网络视频估计的全身+手+脸 SMPL-X，含大量日常/体育/舞蹈/功夫",
         objects="无", fmt="SMPL-X（视频估计, 抖动/滑步多）", access="申请表 + 部分子集需从原站下",
         license="非商业科研, 各子集许可不同", on_disk="否（data/motionx 为空）", umr="SMPL-X 直接读，但质量需先过物理/接触门",
         taxonomy_guess="dynamic/gesture/idle 多, 物体/地面接触少且不可信"),
]
os.makedirs(f"{D}/scout_others", exist_ok=True)
json.dump(dict(note="只汇总可得性与许可，均未在本机核实小时数；来源: scout_sources.md + sources_inventory.json + 公开资料", sources=scout),
          open(f"{D}/scout_others/scout_summary.json", "w"), ensure_ascii=False, indent=1)

# ================= combined =================
per = collections.OrderedDict()
per["BONES-SEED 原始"] = {c: bones["taxonomy_originals"][c]["primary_hours"] for c in C}
per["BONES-SEED 镜像"] = bones["taxonomy_mirrors_primary_hours"]
per["AMASS(xwbc)"] = {c: amass["taxonomy"][c]["primary_hours"] for c in C}
per["LAFAN(xwbc)"] = {c: lafan["taxonomy"][c]["primary_hours"] for c in C}
per["Mixamo"] = {c: uni["mixamo"]["taxonomy"][c]["primary_hours"] for c in C}
uniq_tot = {c: r2(per["BONES-SEED 原始"][c] + per["AMASS(xwbc)"][c] + per["LAFAN(xwbc)"][c] + per["Mixamo"][c]) for c in C}
ml = collections.OrderedDict()
ml["BONES-SEED 原始"] = {c: bones["taxonomy_originals"][c]["multilabel_hours"] for c in C}
ml["AMASS(xwbc)"] = {c: amass["taxonomy"][c]["multilabel_hours"] for c in C}
ml["LAFAN(xwbc)"] = {c: lafan["taxonomy"][c]["multilabel_hours"] for c in C}
ml["Mixamo"] = {c: uni["mixamo"]["taxonomy"][c]["multilabel_hours"] for c in C}
contacts = collections.OrderedDict()
contacts["BONES-SEED 原始"] = bones["contacts_originals"]["hours_any_class"]
contacts["AMASS(xwbc)"] = amass["contacts"]["hours_any_class"]
contacts["LAFAN(xwbc)"] = lafan["contacts"]["hours_any_class"]
contacts["Mixamo"] = uni["mixamo"]["contacts"]["hours_any_class"]
combined = dict(
    generated_for="lloyd: 下一代 AlphaMotion 的人体动作数据丰富度/种类盘点（只读, 无 GPU）",
    taxonomy=C, contacts=CT,
    method=dict(classifier="正则多标签分类(taxo.py)：BONES 用 content_name+动作类型+非站立体位+道具字段；AMASS 用片段名+HumanML3D 描述+BABEL v1.0 act_cat；"
                           "LAFAN 用动作族名；Mixamo 用片段名+UniML3D 描述",
                primary_rule="主类优先级 whole_body_object > ground_contact > sitting_on_object > dynamic > upper_body_manipulation > locomotion > gesture_social > idle_posture > other；主类小时加起来=总时长",
                spot_check="AMASS 按时长加权抽 30 段人工看：约 24 段主类合理(~80%)；主要误差=推搡恢复/伸手抬臂被判成操作、含糊的'sit'默认判成坐物体",
                caveat="多标签小时会重复计算，不能相加；'物体'在这四个源里都只是文字/推断，没有一个带物体位姿"),
    primary_hours_by_source=per, primary_hours_total_originals_only=uniq_tot,
    multilabel_hours_by_source=ml, contact_hours_by_source=contacts,
    totals=dict(
        BONES=dict(orig=bones["hours"]["originals"], mirror=bones["hours"]["mirrors"]),
        AMASS_xwbc=amass["hours"], LAFAN_xwbc=lafan["hours"], Mixamo=uni["mixamo"]["hours"],
        Objaverse_bipedal_unverified=uni["objaverse_bipedal"]["hours"],
        originals_sum_4_sources=r2(bones["hours"]["originals"] + amass["hours"] + lafan["hours"] + uni["mixamo"]["hours"])),
    object_hours=dict(BONES_prop_text=bones["objects"]["has_prop_hours_originals"],
                      BONES_manipulated=bones["objects"]["manipulated_prop_hours_originals"],
                      AMASS_implied=amass["objects"]["object_implied_hours"], AMASS_GRAB=amass["objects"]["grab_hours"],
                      Mixamo_prop_in_name=uni["mixamo"]["prop_in_name_hours"], with_object_pose=0.0),
    status=dict(BONES=bones["status"], AMASS=amass["status"], LAFAN=lafan["status"], UniML3D=uni["status"]),
    outputs=dict(bones="bones_seed/bones_summary.json|bones_summary.md|bones_clip_labels.csv.gz",
                 xwbc="xwbc_amass_lafan/xwbc_summary.json|xwbc_summary.md|amass_clip_labels.csv.gz|lafan_clip_labels.csv",
                 uniml3d="uniml3d_mixamo/uniml3d_summary.json|uniml3d_summary.md|uniml3d_clip_labels.csv.gz",
                 scout="scout_others/scout_summary.json|scout_summary.md",
                 combined="core_sources/summary.json|summary.md|scripts/"),
)
os.makedirs(f"{D}/core_sources/scripts", exist_ok=True)
json.dump(combined, open(f"{D}/core_sources/summary.json", "w"), ensure_ascii=False, indent=1)
here = os.path.dirname(os.path.abspath(__file__))
for f in ["taxo.py", "run_bones.py", "run_xwbc.py", "run_uniml3d.py", "build_summary.py", "write_md.py"]:
    if os.path.exists(os.path.join(here, f)):
        shutil.copy(os.path.join(here, f), f"{D}/core_sources/scripts/{f}")
print(json.dumps(dict(primary=per, total=uniq_tot, ml=ml, contacts=contacts), ensure_ascii=False, indent=0)[:6000])
