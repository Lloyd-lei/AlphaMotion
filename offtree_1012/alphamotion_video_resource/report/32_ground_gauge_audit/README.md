# 32 · 地面 gauge 审计：Megatron 根高偏置的根因（0924）

**结论：根高偏置是标签问题，不是解码器问题。** 接触标注器用的"关节中心→脚底"偏移表（`/home/arenalabs/tmp/sole_offsets3.json`）在 8 台机器人上比厂商 MJCF 实测大 4.7～10 cm，在 H2、R1 上缺失。解码器忠实学了这张错表：15 台机器人上"标签误差预测的偏置"和"实测的解码器地面偏置"差值在 −0.0 到 −2.7 cm 之间（一个所有具身共有的约 −2 cm 解码器小偏移）。A3 不在训练缓存里，零样本偏置另算。

## 数据与口径

- 训练缓存：`/var/tmp/alphamotion/topo_exp/cache_lafan_v3`（LAFAN，77 段，49.7 万行）+ `cache_amass_lconv_tfix`（AMASS，13281 段，473 万行），`L_M20k_0903/launch_args.txt`。每具身：`<emb>_paired_local6d.pt`（局部旋转 [N,J,6]）、`<emb>_root_world.pt`（世界根 cm，ardy Y-up）、`<emb>_contact.npz`（`gap`=各关节离标注器地面的高度 m、`contact` 硬标签、`trust`、header 里的 `h0_m`、`sole_joint_names`、`sole_drop_cm`）、`<emb>_spec.npz`/`_dof.npz`。缓存的 spec 与产品包 `embodiments_v2` 描述子逐位相等（descriptor_audit.json）。
- 根高监督（train_dual.py 134/446 行）：解码器根槽的目标 = `gap_root = Y_root_world − h0`，h0 = 该缓存该具身"最低脚底点"的每段 p5 中位数（h0_table.json 就是这张表）。所以每个缓存自己的地面偏移在训练时已被扣掉；标签自洽：站立相最低脚底点 gap p50 全部在 0.4～1.3 cm。
- 但"脚底点" = 脚部关节中心 + `sole_drop`。这张偏移表就是病根。
- 解码器地面 gauge（decoder_gauge.py）：30 段训练外 AMASS 走/站/跑片段（`cache_amass_heldout/sequences.json`，训练用 `--exclude_set` 排除），Megatron 编码 → 解到每台 → 投影 → 用厂商 MJCF 碰撞几何量脚部最低点（根高用解码值）。"as-trained" = 同一姿态下按标注器口径（脚关节 + 标签偏移）算的最低点，即解码器被要求命中的目标。

## 主表（cm；正 = 悬空，负 = 穿地）

| 具身 | 标签 sole_drop | 厂商 MJCF 实测 drop（碰撞几何） | 预测偏置 = 标签−实测 | 实测：解码器地面 − 标签口径地面 | 残差 | 解码器最低点 p50 | h0 LAFAN / AMASS |
|---|---|---|---|---|---|---|---|
| booster_k1 | 8.8 | 2.4 | +6.4 | +4.4 | -2.0 | +8.8 | -4.3 / -8.0 |
| booster_t1 | 10.2 | 3.0 (视觉网格 3.1) | +7.1 | +5.3 | -1.9 | +8.6 | -6.2 / -9.4 |
| booster_t1_29dof | 12.7 | 3.0 (视觉网格 3.1) | +9.7 | +7.0 | -2.7 | +9.2 | -8.8 / -8.4 |
| fourier_gr3 | 16.8 | 6.8 (视觉网格 1.8) | +10.0 | +7.9 | -2.1 | +10.4 | -9.8 / -16.2 |
| fourier_n1 | 4.5 | 4.4 | +0.0 | -1.7 | -1.7 | +3.3 | +1.1 / -4.2 |
| hightorque_hi | 4.4 | 4.4 | -0.0 | -1.5 | -1.5 | +1.7 | +2.1 / -4.4 |
| kuavo_s45 | 10.8 | 6.0 | +4.7 | +2.4 | -2.3 | +6.2 | -1.7 / -10.3 |
| pnd_adam_lite | 15.9 | 6.5 (视觉网格 1.6) | +9.4 | +7.5 | -1.9 | +10.7 | -2.2 / -14.9 |
| tienkung | 15.7 | 5.7 (视觉网格 0.9) | +10.0 | +8.1 | -1.9 | +10.8 | -5.7 / -13.3 |
| unitree_g1_23dof | 3.5 | 3.5 | +0.0 | -1.5 | -1.6 | +2.8 | +2.9 / -3.6 |
| unitree_g1_29dof | 3.5 | 3.5 | +0.0 | -1.5 | -1.6 | +2.8 | +1.3 / -2.3 |
| unitree_h1 | 7.0 | 7.0 | -0.0 | -2.3 | -2.3 | +3.8 | +4.7 / -6.5 |
| unitree_h1_2 | 4.5 | 4.5 | +0.0 | -2.2 | -2.2 | +1.1 | -0.6 / -4.2 |
| unitree_h2 | 0.0 | 5.4 | -5.4 | -6.1 | -0.7 | -2.7 | +3.7 / +1.3 |
| unitree_r1 | 0.0 | 0.5 | -0.5 | -0.5 | -0.0 | +2.1 | +1.8 / -0.2 |

读法：预测列和实测列几乎逐台相等（残差 −0.0～−2.7）。G1、H1、H1_2、N1、hightorque 的偏移表是对的，解码器地面也对（±2 cm 内）；K1、T1、T1_29、pm01、GR3、kuavo、Adam、tienkung 的偏移表偏大 4.7～10 cm，解码器就悬空同样的量；H2 没有偏移（把踝关节中心当脚底），解码器就穿地一个踝高（5.4 cm）。R1 偏移缺失但真实只差 0.5 cm，无碍。

视觉网格 vs 碰撞几何：GR3、Adam、tienkung 的碰撞盒比视觉网格的鞋底还低 4～5 cm（`visual drop` 1.0～1.8 vs 碰撞 5.7～6.8）。以视觉网格为准的话这三台的标签误差更大（≈15 cm）。仿真里踩的是碰撞几何，物理门按碰撞算；给买家看的视频按视觉网格算。两套数都要写在数据卡上。

偏移表的来源没有找到生成脚本（`sole_offsets{,2,3}.json` 8 月 31 日～9 月 1 日生成，v1 里 GR3 就已经是 16.8）。盘上所有 GR3/Adam/tienkung/T1 的模型文件都复现不出这些数字（见 sole_check 输出），所以它不是"另一个模型文件"的口径，是量错了。

## 每语料的原始地面（标注器扣 h0 之前，站立相最低脚底点 p50 / 穿地帧占比）

| 具身 | LAFAN raw p50 | LAFAN 穿地% | AMASS raw p50 | AMASS 穿地% | 扣 h0 后 gap p50 L/A |
|---|---|---|---|---|---|
| human_smpl | +2.1 | 0 | +2.6 | 20 | 0.6 / 0.5 |
| booster_k1 | -3.8 | 80 | -7.4 | 96 | 0.5 / 0.6 |
| booster_t1 | -5.8 | 83 | -8.6 | 98 | 0.5 / 0.8 |
| booster_t1_29dof | -8.3 | 85 | -8.0 | 97 | 0.5 / 0.4 |
| engineai_pm01 | -7.6 | 82 | -86.5 | 100 | 0.4 / -1.9 |
| fourier_gr3 | -9.4 | 83 | -15.7 | 97 | 0.4 / 0.5 |
| fourier_n1 | +1.7 | 1 | -3.6 | 89 | 0.7 / 0.6 |
| hightorque_hi | +3.4 | 3 | -3.9 | 90 | 1.3 / 0.5 |
| kuavo_s45 | -1.0 | 49 | -9.8 | 84 | 0.7 / 0.5 |
| pnd_adam_lite | -1.4 | 59 | -14.3 | 98 | 0.8 / 0.7 |
| tienkung | -3.4 | 77 | -11.9 | 94 | 2.4 / 1.4 |
| unitree_g1_23dof | +4.1 | 1 | -2.9 | 89 | 1.2 / 0.7 |
| unitree_g1_29dof | +2.2 | 4 | -1.8 | 74 | 0.9 / 0.5 |
| unitree_h1 | +5.9 | 1 | -5.8 | 74 | 1.2 / 0.7 |
| unitree_h1_2 | +0.5 | 16 | -3.2 | 76 | 1.1 / 1.0 |
| unitree_h2 | +4.7 | 1 | +2.6 | 0 | 1.0 / 1.3 |
| unitree_r1 | +2.4 | 3 | +0.8 | 8 | 0.6 / 1.0 |

人体 `human_smpl`：LAFAN raw p50 +2.1 cm、穿地 0%；AMASS raw p50 +2.6、穿地 19.6%（p10 −3.3，场次地面不齐）。扣 h0 后两边都是 0.5 cm。AMASS 的机器人语料原始地面普遍在 −4～−16 cm（xwbc 重定向把标签脚底当地面，而那个"脚底"本来就是错表算的），LAFAN 只有 −10～+5。这解释了以前"K1/T1 AMASS 穿地 83/50% 但 LAFAN 0.3/0.6%"的观察：是同一张错表在两套语料上的表现。engineai_pm01 的 AMASS 缓存 h0 = −84.6 cm（整台埋地下），本来就被排除。

## 判决（逐台）

- **数据/标签问题（修表即可）**：booster_k1、booster_t1、booster_t1_29dof、engineai_pm01、fourier_gr3、kuavo_s45、pnd_adam_lite、tienkung（偏移表偏大），unitree_h2（缺偏移，且缓存骨架是 body00…body31 泛名，SigLIP 语义全空）。
- **本来就对**：unitree_g1_29dof、unitree_g1_23dof、unitree_h1、unitree_h1_2、fourier_n1、hightorque_hi、unitree_r1。
- **解码器自身**：所有具身共有约 −1.5～−2.5 cm 的偏移（实测比标签口径略低），这是模型的账，量级小。
- **A3（agibot_a3_t3d0）**：不在任何训练缓存里，报告 29/30 里的 −7～−22 cm 是零样本泛化的账，本审计不能归因；要么进训练集，要么靠解码后接触贴地。

## 修法（按代价从低到高）

1. **立刻可用、不重训**：解码后按表校正根高，校正量 = 本表"预测偏置"取反（GR3 −10.0、tienkung −10.0、T1_29 −9.7、Adam −9.4、pm01 −9.2、T1 −7.1、K1 −6.4、kuavo −4.7、H2 +5.4、R1 +0.5、其余 0；单位 cm，全是实测值不是手调）。或者更通用：解码器接触 logit 判哪只脚踩地，脚底贴零，这一步天然免疫偏移表。
2. **修数据**：用厂商 MJCF 碰撞几何重算 `sole_offsets`（sole_audit.py 的 `true_drop_cm` 就是），重跑 contact_label.py 生成 8 台 + H2 的 `<emb>_contact.npz`（h0 和 gap 都会变），H2 先把缓存描述子换成厂商命名的 36 节点骨架（几何已相等，drop −98.45 cm 一致，只差名字）。
3. **重训根头**：只有 rootpos、pen、cmaint、slip 这几项依赖 gap，其余损失不变，可以在现有权重上只微调解码器（和"只训解码器 reenc"同一套路），不用从头来。

文件：results.json（主表）、cache_gauge.json/.py（缓存 gauge，每 8 行采样）、sole_audit.json/.py（偏移表 vs 实测）、decoder_gauge.json/.py（30 段训练外解码）、descriptor_audit.json（缓存 spec vs v2 vs MJCF）、gauge_bias_chart.png。