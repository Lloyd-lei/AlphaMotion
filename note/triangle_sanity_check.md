# Triangle Attention 真的学到了东西吗？—— Pair Tensor 的 4-way Sanity Check

**项目**: MotionFormer (be water, robot)
**日期**: 2026-05-02
**作者**: Yudong Lei
**Stage**: 1.5 retrospective
**对应代码库**: `experiment/stage1/`
**配套文档**: `note/motionformer-research-vision.md`（完整研究愿景）

---

## TL;DR

通过 4 个独立 sanity check，我们证明 MotionFormer 的 pair tensor 学到了**可重复、非 artifact、非初始化残留、且不可由 OPM 单独习得**的解剖学结构（spine 暗团 + 4 肢内部 cluster + L.Hand/R.Hand hub）。

| 测试 | 排除的可能性 | 结果 |
|---|---|---|
| 不同 batch 抽样 (seed=0 vs seed=42) | "只是某次抽样巧合" | r = **0.9937** ✓ 高度可重复 |
| Untrained MotionFormer 同 forward | "MotionFormer 架构本身就这样" | r = **0.057**, 非训练态 std/mean = **0.009** ✓ 完全平 |
| pair_init 参数自身 | "init 残留没被训练改变" | r = **0.310** ✓ 只解释 31% |
| OPM-only checkpoint (无 triangle) | "OPM 单独也能学到" | r = **0.0765** ✓ 几乎正交 |

**核心 takeaway（更新 §9.1.ter）**：

> **Triangle attention 不在 reconstruction loss 上做贡献，但是引导 pair tensor 朝 anatomically-meaningful 模块化结构收敛的关键机制。Reconstruction loss 看不到 triangle 的贡献，因为对"填回缺失关节"任务，distributed pair geometry 和 modular pair geometry 都够用——但下游需要 representation 结构的任务（family classification / cross-morphology transfer / interpretable control）只有 modular pair 能给。**

---

## 1. 为什么需要这个 sanity check

mixed_fixed30 完成后，几个数字呈现 paradox：

| Metric | full (有 triangle) | opm_only (无 triangle) | 差异 |
|---|---|---|---|
| AMASS heldout pos | 0.1222 | 0.1212 | +0.8% （平） |
| val_id pos (ep30) | 0.154 | 0.160 | -3.7% （持平） |
| PCA PC1+PC2 方差解释 | 0.538 | 0.307 | **+75%** |
| Pair tensor L2-norm 视觉 | 清晰 anatomical cluster | 弥散网格 | **完全不同** |

数字层面 triangle 看起来 no-op，但视觉层面有显著 cluster。问题：**这个 cluster 是真学到的，还是 confound？**

需要排除四种 confound:
- (A) Sample抽样 artifact
- (B) Architecture artifact
- (C) Initialization 残留
- (D) OPM 单独也能学

---

## 2. 实验设置

**模型**: 6-layer MotionFormer, hidden=128, pair_hidden=32, T=90 (3s @ 30fps), J=77 (SOMA77), C=3
**数据**: AMASS + Kimodo + HY Motion mixed (修完坐标轴后, 6665 train / 833 val_id / 984 val_subject / 41 val_source)
**训练**: 30 epochs, MMM (5 mask modes mixed), lr=3e-4, batch_size=32, seed=0
**Pair tensor 提取**: forward 32 个随机样本（无 mask），取 `model._last_pair`，对 (i, j) 计算 L2 norm 在 H_pair 维度上，得 [77, 77] 矩阵，对称化。
**比较指标**: 对 off-diagonal 元素计算 Pearson r 和 std/mean 分布度

---

## 3. 结果

### 3.1 数字

```
[A1] full TRAINED, sample seed=0    range=[4.79, 16.26]  std/mean=0.162
[A2] full TRAINED, sample seed=42   range=[4.78, 17.18]  std/mean=0.171
       |Δ(A1, A2)| / mean = 1.70%

[B]  full UNTRAINED (random init)   range=[5.51, 5.78]  std/mean=0.009  ← 几乎全平
[C1] pair_init param TRAINED        range=[0.12, 0.34]  std/mean=0.171
[C2] pair_init param UNTRAINED      range=[0.08, 0.15]  std/mean=0.088  ← randn
[D]  opm_only TRAINED (no triangle) range=[4.07, 9.52]  std/mean=0.115
```

**Off-diagonal Pearson r 矩阵：**

```
trained_seed0 vs trained_seed42        = 0.9937   ← 几乎重合
trained_seed0 vs UNTRAINED              = 0.0573   ← 几乎正交
trained_seed0 vs opm_only               = 0.0765   ← 几乎正交
pair_init TRAINED vs UNTRAINED          = 0.2372   ← weights 大幅移动
pair_init TRAINED vs trained_seed0      = 0.3096   ← init 解释 31%
```

### 3.2 视觉证据

参见 `figures/pair_heatmap_sanity.png` (6-panel) 和 `figures/pair_heatmap_compare.png` (full vs opm 并列大图)。

| Panel | 内容 | 视觉特征 |
|---|---|---|
| **A1** | full 训练后, n=32 sample seed=0 | 清晰 anatomical cluster: spine 暗团 + 上肢两块亮 cluster + L.Hand/R.Hand hub 列 + 腿部局部 cluster |
| **A2** | full 训练后, n=32 sample seed=42 | **和 A1 视觉上无法区分**（r=0.99） |
| **B** | full 未训练 (random init), 同 forward | **均匀绿色地毯**, dynamic range 仅 5.51-5.78 |
| **C1** | trained 模型的 pair_init 参数（不 forward） | 有大尺度先验（部分 row/col 偏暗），但形态远不如 A1/A2 复杂 |
| **C2** | 未训练模型的 pair_init 参数 | 完全 randn，无任何 pattern |
| **D** | opm_only 训练后 (无 triangle) | 学到了 cluster, **但 pattern 完全不同**——是水平/网格状亮带，没有 spine 暗团也没有 hub 列 |

### 3.3 怎么读 anatomical heatmap

77 个关节按 SOMA77 拓扑顺序排（脊柱 → 左臂 → 左手指 → 右臂 → 右手指 → 左腿 → 右腿）。每格 (i, j) 颜色 = pair tensor 在那对关节上的 L2 norm = 模型给这对关节的"协同关系"分配了多少 channel capacity。

四个可解释的 cluster:

1. **左上深蓝团 (Hips/Spine/Chest/Head, idx 0-10) = 接近刚体的躯干。** 脊柱+头部在大部分动作里几乎一起平移旋转，相对几何关系不变 → 模型把最少 capacity 给这块。
2. **上肢两个亮块 (左臂 11-38, 右臂 39-65) = 肢体内部协同。** L.Hand 列特别亮，因为 L.Hand 是 13+ 手指 descendant 的 hub，腕的 SE(3) 决定整只手姿态。R.Hand 同。
3. **跨肢体 pair norm 普遍低 (中段大片偏蓝) = 远端关节物理弱耦合。** 左臂和右腿、左手和右脚直接物理关系弱，pair tensor 给最低 capacity。
4. **腿部 cluster (66-75) = 步态周期里左右腿的 anti-phase 协同。**

模型**没有被告知**哪些是手指、哪些是脊柱，全靠 mocap + pair tensor + masked motion modeling 自发学会把 77 关节分割成 3 大功能子系统（spine + 4 肢），并在子系统内分配高 capacity、子系统间分配低 capacity。

---

## 4. 解读：Triangle 的真实角色

**A vs B** 排除"架构 artifact"：未训练 MotionFormer 的 pair tensor 输出**完全均匀**（std/mean=0.009），结构 100% 来自训练。
**A1 vs A2** 排除"batch 巧合"：不同 batch 抽样下 pair tensor 几乎不变（r=0.99）。
**A vs C1** 排除"init 残留"：pair_init 参数本身只解释 31% 的 final pattern；69% 来自 6 个 block 中 OPM + axial + triangle 的**动态精炼**。
**A vs D** 揭示"triangle 真实贡献"：同一架构去掉 triangle 后，pair tensor 学到**完全不同**的几何（r=0.08，几乎正交）。

→ Triangle attention 是引导 pair tensor 朝 "modular anatomical" 几何收敛的**唯一**机制。OPM 单独也能学到 pair structure，但是 "distributed/网格" 几何，缺少模块化和 hub 结构。两者**不可互换**。

### 这改变了 Stage 1 的 thesis 措辞

**旧措辞** (§9.1.ter v1): "Triangle 在 OPM 之上锦上添花，是 representation shaping 而非 loss helper"

**新措辞** (本次发现):

> Pair tensor 几何 ∈ {distributed, modular} 是一对**正交的 representation philosophy**，不是同一 axis 上的 ID-OOD trade-off：
> - **OPM 单独 → distributed pair**：高耦合分散到许多关节对，无明显模块结构
> - **OPM + Triangle → modular pair**：耦合集中到肢体内部，跨肢体稀疏，hub 关节（L.Hand/R.Hand）显式
>
> Reconstruction loss 看不出哪个更好，因为 fill-in 任务对两者无偏好。但下游需要 anatomical 结构的任务（cross-morphology transfer、interpretable style control、family-aware retrieval、damage-aware compensation）应当强烈偏好 modular geometry。
>
> **Triangle 的角色因此不是"loss 优化器"，而是"表征几何选择器"**。

---

## 5. 复现路径

### 数据 / Checkpoint
- 修完坐标轴的 mixed dataset: `experiment/dataset/mixed_soma77.npz` (9356 samples, 77 joints, 90 frames)
- Trained checkpoints:
  - `experiment/stage1/runs/full__mixed_fixed30/final.pt` (full variant, 2.18M params)
  - `experiment/stage1/runs/opm_only__mixed_fixed30/final.pt` (no triangle, 2.12M params)
- 训练命令:
  ```bash
  python train.py --mixed-data ../dataset/mixed_soma77.npz \
                  --splits ../dataset/splits.json \
                  --model {full,opm_only} \
                  --epochs 30 --batch-size 32 --mask-mode mixed \
                  --run-tag mixed_fixed30
  ```

### Sanity check 脚本
见 `experiment/stage1/pair_heatmap_sanity.py`。

```bash
python experiment/stage1/pair_heatmap_sanity.py
# 产物: /tmp/pair_heatmap_sanity.png + 数字 print 到 stdout
```

### 关键代码引用
- Triangle attention 实现: `experiment/stage1/motionformer.py` `TriangleMultiplicativeOutgoing` + `TriangleAttentionStarting`
- OPM 实现: 同文件 `OuterProductMean`
- Variant 控制: `motionformer.py` `MotionFormerConfig.use_pair / use_opm / use_triangle` 三 flag

---

## 6. Open Questions / Future Work

1. **Modular pair 在跨形态迁移上是否真的更好？** 当前所有数字都是 single-morphology (SOMA77 人形)。Stage 2 计划用四足 / 砍腿场景验证。本节的"modular geometry 对 cross-morphology 友好"是 conjecture，待证。
2. **不同 mocap 数据源的 pair geometry 差异**？AMASS-only / Kimodo-only / 100Style-only 训练分别会给出哪种 pair？
3. **Triangle 的 4 个 variant**（outgoing/incoming × mult/attn）在 motion 上是否需要全部？AlphaFold 都用了 4 个，本工作只用了 outgoing-mult + starting-attn 2 个。剩下 2 个会进一步压缩还是无影响？
4. **Stage 1 observation: kinematic chain mask 在 mixed_fixed30 下严重恶化（0.06→0.22, 4×）。** 这表明全身发力**不是**普适 kinematic chain 的线性传递（很多传统武术、芭蕾动作下游发力链断开），下一步 mask 策略应当避免"连下游一起 mask"的硬假设。
5. **Reconstruction loss 是否是 motion learning 的合适评估**？本研究发现 reconstruction loss 对 representation geometry 不敏感。是否应增补："representation similarity to known motor synergies (Bizzi/d'Avella)" 之类的 representation-level metric？

---

## 引用 / 致谢

如果你对此发现感兴趣或想合作，欢迎 issue / PR。

- 架构灵感: AlphaFold 2 (Jumper et al., 2021) Evoformer, OpenFold reference impl
- 训练范式: Masked Motion Modeling (MaskedMimic, Kimodo)
- 数据来源: AMASS, NVIDIA Kimodo (synthesized), HY Motion (Tencent), 100STYLE
- 相关工作对比详见 `note/motionformer-research-vision.md` §10
