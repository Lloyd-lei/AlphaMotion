---
name: spatial-eval
description: spatial 层（Greenwich 系）的标准验收流程——跨具身动作迁移 + 零样本动作表达。五个必考项：无监督家族零样本 rollout、有 GT 偏差+视觉双管、四档 SE3 约束（无精修）、机器人→机器人迁移、全具身嘉年华合集。lloyd 0810 定版。
---

# spatial-eval — 空间层五项验收（lloyd 0810 定版）

任何新的 spatial checkpoint 竞争 Greenwich 头衔或对外交付前，跑完五项。
渲染纪律沿用 /motion-eval（mesh+限位投影、不精修、pos 一律 FK 口径、发前抽帧自检、
路径全打印、好/坏由 lloyd 肉眼判）。

## 1. 零样本家族 rollout（无 GT，纯视觉判）
- 语料：**体系外数据集**（当前 = cache_bones_body；训练语料变更时重审）
- 用 PCA 对 clip 级嵌入聚类出动作家族（家族不是 action_labels 的正则标签,
  是数据自己长出来的簇），每簇抽 3 段动作
- rollout 到 **g1 / x2 / adam / fourier_gr3** 四台（覆盖 留出/无数据集/训练内×2）
- 产出：每段动作一支四机并排 mesh 视频；结论只标 好/坏,由 lloyd 判

## 2. 有跨具身 GT 的动作：偏差 + 视觉双管
- 语料：xwbc 全量配对 cache 的 val clip（18 具身真配对）
- 数值：vs 真配对的 geodesic° + FK-pos cm（均值+p95）,身体选 holdout(g1×2/talos)
- **数字再好也必须过视觉管**：每个 holdout 身体至少一支 GT|重建 并排视频

## 3. SE3 约束四档（严禁精修,constraint 走草图进编码器 = m4 v2 通路）
- 档位：单手腕 / 双手腕 / 双脚踝 / 全部 6 anchor
- 轨迹来源三种：
  a. 格斗数据集（AMASS 武术家族）具身 GT 上截取的真轨迹
  b. 合成:手部画圆
  c. 合成:root 下沉（蹲起）
- 报被约束关节的命中误差（FK）+ 自由关节合理性,每档一支视频
- 已知陷阱：v1"事后覆盖"式约束已证伪,勿回退

## 4. 机器人 → 机器人迁移
- 输入某台机器人的 GT 重定向（如 g1 的配对数据）,编码后解码到另一台
  （g1→adam、adam→x2 等,至少含一个 holdout→trained 和一个 trained→无数据集）
- 有 GT 的方向报偏差,全部出视频

## 5. 全具身嘉年华合集
- 以上各项选代表性 rollout,拼进**一支**多具身大画幅视频
  （greenwich_carnival 机制:BLUE=训练内/YELLOW=留出/RED=无数据集 着色,含 SMPL 源格）
- 这是对外演示用的最终交付物

## 产出物清单
- runs/SPATIAL5_<run>_verdict.txt（数值项汇总,标注每支视频绝对路径）
- R4_VIZ/mp4/spatial5/<run>/ 下全部视频
- 判决行:五项各一行 PASS/FAIL/待肉眼,最终由 lloyd 裁决
