---
name: motion-eval
description: AlphaMotion 模型评估的完整流程与口径纪律（lloyd 定版）。任何 spatial/temporal 模型出结果前必须走这六项：数值、视觉(视频)、统计、信息熵压缩、跨具身泛化、动作泛化。含历次踩坑的强制规避条款。
---

# AlphaMotion 评估流程（六项全套）

评估对象出现新 checkpoint 时，六项全跑；单项探索至少走数值+视觉。
所有判决写入 `/var/tmp/alphamotion/topo_exp/runs/<NAME>_verdict_*.txt`，视频写入
`pose_translator_design/R4_VIZ/mp4/`，**每一份产物打印绝对路径**。

## 1. 数值分析
- 误差唯一口径：`v5_common.key_se3_err`（geodesic deg + cm）。禁止自创口径。
- **spatial 层的位置误差一律用旋转 FK（`v5_common.fk_pos` 刚性骨）**——解码器的
  原始位置头是训练代理("橡皮骨"),数值虚高约 2 倍(0810 实测 13.3 vs 7.9cm),
  不许进判决表。
- **均值和尾部一起报**：mean + p95 + p99（逐关节、逐帧分布的尾部）。只报均值的
  结论一律视为不完整——尾部姿态（如 backfist）在均值里不可见。
- 陷阱：`.norm(-1)` ≠ `.norm(dim=-1)`（第一个位置参数是 p，p=-1 范数恒≈0，
  会伪装成完美模型）。所有 0.000 都先怀疑测量。
- `content_share` 在分母（清零收益）< ~4cm 时作废，先看 `registers buy [cm]` 绝对值。
- 工具：`eval_dual.py --run <dir> --out <json>`（空间"唯一尺"，从 config.json 重建），
  `v7_eval.py --runs ...`（时序五行判决表）。

## 2. 视觉分析（视频，权重高于数字）
- lloyd 纪律原文："别看数字 播放视频"、"禁止火柴人渲染视频 必须加mesh"。
- 一律 robot mesh + 限位投影（`robot_stream`/`MP.project` 链），火柴人只许自己 debug。
- GT 格的定义：**全身关节过 spatial 转换**或**数据集真配对重定向**，
  不许拿 6-anchor 脑补版冒充 GT。有真配对（18 具身 cache 内）优先用真配对。
- 不精修（"做se3的时候 记得别精修"）。
- 变速演示只准**速度码移植**（encode 变速副本取 time_z，内容码不动），
  禁止 T_out/播放速率把戏（"方式一…你有脸给我说吗"）。
- **发给 lloyd 前自己抽帧验**（ffmpeg select 抽 2-3 帧看图）——本项目发生过
  FK 喂错(local 当 global)渲出麻花人还发出去的事故。
- 视频写明路径，结论留给 lloyd 肉眼；不替他下"看起来对"。
- 工具：`exam_c_mesh.py`、`punch_gt_g1.py`（真 GT 五联）、`speed_token_rollout.py`、
  `token_ablation.py`、`greenwich_carnival.py`、`src_motion_render.py`+
  `render_smpl_source.py`（SMPL-X 原动作，genmo-gem 环境）。

## 3. 统计分析
- 按 12 家族分桶（`action_labels.py` 分类器）报误差；AMASS 77% 落 "other"，
  稀有家族（punch/kick 等）单列——尾部问题都藏在稀有桶。
- 检索类评估（kNN/m5）必须**平衡检索库** + **GT 自检索对照**（不平衡库出过
  0.15→0.74 的假差距）。

## 4. 信息熵压缩分析
- FSQ 逐维占用熵（`probe_fsq_usage.py`）：死维(<0.1 bit)该砍、饱和维(>2.9/3.00)
  该加档；20 万 token 抽样不同码计数。加维/加档的决定只认占用证据。
- bit/s 口径：时序臂 62.69 bit/s 为同预算可比线；`--rate_sweep` 豁免的 run
  必须标注 OFF_BUDGET，禁止进同预算判决表。
- 内在维度扫描（决定码本维数时）：必须**时间去相关**（每 clip 稀疏抽帧），
  否则 TwoNN 量出来的是轨迹维度(~2)不是流形维度。

## 5. 跨具身泛化
- holdout 铁律：`unitree_g1_29dof, unitree_g1_23dof, pal_talos` 永不入训，
  作为身体级零样本 + 真配对误差（`eval_newbody.py`）。X2 = 任何数据集都没有的身体。
- 陷阱：xwbc 有两套人体骨架（跨语料评人体必崩）；3DoF 关节欧拉分支会伪造
  100° 假超限（异常数字先怀疑测量）；G1 官方 URDF 的 6.90% 违规是机器人物理 ROM。
- **sem_encoder 必须按 run 的 config.json 透传**（SigLIP 768 vs Qwen 1024）——
  `v5_common.body_entry`/`tokenize_stream.load_body` 默认 SigLIP，指向 Qwen 系
  底座前先补 encoder 参数；`load_tokenizer` strict=False 会静默劣化，
  加载后 grep "WARN state_dict"。
- vis_residual 模型的"自重建"指标死（同骨架 identity 通路 0.1°），
  同骨架**无观测**解码是它的 OOD——两个方向都不许当质量证据。

## 6. 动作泛化
- 零样本分三级，表述必须分开：clip 级（val split 内没训过的条目）、
  家族级（整类动作没见过）、语料级（体系外数据集，如 bones-seed）。
- **训练语料变更时逐把尺子重审污染状态**：例——AMASS 13,281 全喂后，
  AMASS 上不再存在 motion 级零样本，主尺换 bones-seed(`cache_bones_body`)；
  cache 换代时 KeyData val split(seed 123) 会整体重排，跨 cache 比较需注明。
- 双轴评估方法论：物理一致性（无需真值）+ WBC 可跟踪率；真值本身可能超限/
  有偏（历史上真值偏 93° 被否决过），不当唯一尺子。
