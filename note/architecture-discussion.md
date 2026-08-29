# Architecture Discussion

*AlphaMotion 模型架构与训练策略讨论 —— Stage-2 启动前的内部对齐稿*

---

| 字段 | 值 |
|---|---|
| 任务类型 | 架构讨论 |
| 状态 | 进行中 |
| 优先级 | 高 |
| 负责人 | Lei Yudong · Jiang Yuxin · lu chi · lucas zhuang |
| 协作机构 | ArenaLabs · NVIDIA · UCLA |
| 开始日期 | 2026-06-20 |
| 截止日期 | 2026-06-25 |
| 备注 | 根据初步实验结果调整模型架构和训练策略 |

---

> **AlphaMotion —— 第一个以 Scaling Law 为核心验证目标、以合成数据为驱动、以统一上下肢为边界、以模块化多模态嫁接为出口的 Motion Foundation Model。**

## 目录

- [Scopes for a Synergy-Aware Motion Foundation Model](#scopes-for-a-synergy-aware-motion-foundation-model)
- [架构设计总览](#架构设计总览)
- [0. 什么是 Synergy?](#0-什么是-synergy)
- [1. Representation](#1-representation)
- [2. Attention 怎么设计?](#2-attention-怎么设计)
- [3. 让模型学会捕捉关键时序 —— 学会自己 clip 动作序列](#3-让模型学会捕捉关键时序--学会自己-clip-动作序列)
- [4. 如何学习一个 Synergy Graph?](#4-如何学习一个-synergy-graph)
- [5. Encoder and Decoder](#5-encoder-and-decoder)
- [6. Token 设计](#6-token-设计)
- [7. Loss 设计](#7-loss-设计)
- [8. 涌现非人类动力链动作](#8-涌现非人类动力链动作)
- [9. 架构创新尝试 / Pivot](#9-架构创新尝试--pivot)

---

## Scopes for a Synergy-Aware Motion Foundation Model

1. **验证 Scaling Law**：解锁海量互联网数据；利用互联网视频以及 SMPL 数据集收集全身人体数据，训练一个模型能够 Synergy-Aware、涌现、动作类型泛化、可加入 encoder/decoder 后训练过后的任务迁移能力。
2. 成为任意"大脑"范式（VLA、WAM）的动作嵌入基础。
3. 超长程动作生成，more than a few-second clip。
4. 统一 Manipulation 与 Locomotion，不再孤立机器人上下肢训练。
5. 训练一个"孪生 / 原生适配"的下游通用底座模块，结合少量仿真 RL 后训练即可迁移到任意人形机器人上。

---

## 架构设计总览

模型主要分为 **Synergy Module** 和 **Motion Module**。

- **Synergy Module** 作为真正的 Foundation Model 可以广泛地用于预测生成、分类、重建、语义嵌入等任务。Synergy Module 采用 Axial Attention，分别为 Spatial Attention 和 Temporal Attention。
- **Motion Module** 与 Synergy Module 共享同一个 Temporal Attention。

Synergy Module 显示学习一个 Synergy Graph 直接输出稀疏的动作关键帧骨架，用以扩散引导 Motion Module 生成 Physical Solid 的中间动作，并利用 Synergy Module 训练出来的 Synergy Graph 对不同人形机器人硬件做 retargeting。Motion Module 还可以结合仿真 RL 接额外输出头做成一个通用低层控制器，从而实现动作生成和动作执行的端到端 Action Expert。

第一代 Synergy Module 可以先通过无多模态 Label 的原始 SMPL 数据用以动作数据分类和重建，用以清洗并合成海量高质量的互联网数据（数据飞轮）。再通过我们特质的 AR 眼镜 + Sports Coaching APP 捕获带全身人体动作的 Egocentric 数据，用以训练大脑-小脑端到端的 "究极机器人 AI"。励志成为全人行机器人领域真正的 End Game！！！

---

## 0. 什么是 Synergy?

为什么人类可以扔出一个时速高达上百码的棒球，远超任何灵长类动物？为什么一个瘦小的职业拳击手可以爆发惊人的破坏力，远超体型大小限制？为什么人类可以不眠不休的奔跑整个日夜，远超任何肉身动物？为什么人类可以借助身体和工具，完成许多难以实现动作？这一切藏在人类不同身体关节的协同机制 —— 动力链当中。

在我看来，动力链就像是从运动学和生物物理学的角度出发，去破译人体的解剖密码。从生物学出发，今天所有脊椎动物的共同祖先，都是几亿年前生活在海里的鱼类。然而自然界却只有人类进化出直立行走，并且解放双手使用不同的工具。最早的直立行走大约出现在 700 万年前，到了 200 万年前，人类不光可以指令行走，还进化出了耐力奔跑的适应。虽然人类不是奔跑速度的 GOAT，但在耐力上人类几乎站在了金字塔尖上。

从 MPC 的角度出发，人类行走采用的是倒立摆的机制，让重力势能和动能交替作用，这让行走特别节能。当人类进入跑步模式，倒立摆不再适用了，转而进入了质量弹簧模式，来频繁储存势能和释放动能。为什么同一幅身躯有不同的发力模式？关键在于动力链的旋转节藕。相对于走来说，跑的速度更快了，存在双脚同时离地的腾空期，这不光因为下肢要承受更大的反作用外力，也意味着下肢要承受瞬时加速带来的巨大扭矩，所以人类的头颈、胸廓、髋可以大幅单独旋转，这些旋转可以抵消人类剧烈运动时产生的大幅度扭矩，这更稳定，也更节约能量。而这些部位正是人体发力核心。

人类运动能力的巨大进化并不是依赖某一个关节产生的，而是依赖身体各个关节之间的能量传导和协调。当拳击手在打出惊人的后手拳时，力从地起，传导至大腿、髋部、层层脊椎，最终推动身体前行带动上肢"投掷"。而这种类似的发力方式，广泛出现在标枪、棒球、篮球等运动当中。

1875 年，德国机械工程师 *Franz Reuleaux* 从机械齿轮传导的原理当中提出了动力链的概念。他认为人体关节的协调运动就如同机械，取决于构件之间的几何约束，两个可以相对运动的构建组成一个基本的运动单元，多个运动单元连接成了动力链。如果动力链两条两端被固定，那么施加外力的时候，每个阶段都会将力传到到相近构建，形成链式反应。在 1933 年伦敦的骨科医学年会上，德国骨科医学博士 *汉斯·冯·拜耳* 首次将动力链应用到人体运动分析。他着重分析了肢体件肌肉的协同作用，并将执行远端发生的作用比更肢体近端发生的作用作对比。之后真正将懂力链编成运动学核心的人，是美国爱荷华骨科的创始人 *亚瑟斯坦·德勒*。这位精通七种语言的终身学者在为了研究人体运动，主动在 40 多岁的时候学习重新学习微积分、物理、还有工程学，用来分析人体运动。正如他常说的：*"I would rather be wrong with impartial reason than right without one"*。在他快到 80 岁时，他出版了《人体运动学》。

在这本书中，他把动力链定义为"由几个连续的关节排列组合成的复合运动单元"。在这个基础上，他又根据肢体远端是否遭受外力而影响自由运动，把动力链分为了 **开链（open kinetic chain）** 和 **闭链（closed kinetic chain）**。比如挥手这个动作，肢体的远端可以自由运动，他就是一个开链运动。比如引体向上，肢体的远端遭受了非常大的外界阻力，运动发力受阻，他就是一个闭链运动。后来所有关于动力链开链和闭链的描述都基于这个框架。但单纯的开闭链不足以描述所有的运动。比如游泳和自行车，传统上被视为开链运动，但其实远端承受了负荷，只是远端并没有固定或者限制运动。所以生物力学专家 *Charles · J · Dillman* 认为，分类应该由末端关节是移动还是固定，是否承重来决定。基于这些他把动力链分为可移动无负荷，可移动有负荷，固定无负荷，还有固定有负荷。他的研究证明，如果是在肌肉力量激活水平的层面，两个动作并不取决于是否开闭，而是取决于生物力学条件。

在这里，我给出我个人的看法。动力链形式是为了指导我们分类和监督全身动作而用于训练和康复，帮助我们更好的理解人类站立行走这一特殊生物解剖模型从进化学角度的独特。人的运动系统是一个整体，某一部分功能障碍可以通过特定的链条影响全身。结合许多临床研究，某个部位的损伤或者功能损伤障碍，可能会影响其他部位的运动表现。比如，一个棒球投手的脚受伤了，肩关节的下降影响投球表现。同样出现疼痛的部位可能是由于其他部位的功能缺失导致的。人类会自然偏向动作更省力的发力模式。比如在拳击动作的数学分析里，驱赶产生的动能降低了 20%，就需要手臂增加 30% 左右的速度，或者肩部增加 80% 以上的力量，才能保证机打力度相同。所以一个人类的高效动作存在关节之间的分工。有些关节关节拥有更高的自由度，有些关节拥有更强大的力量，有些关节提供稳定性或者传递力量。这些动力链协同机制本身就存在于人类小脑神经网络当中，使得人类展示出了极强的适应能力，可以让人类适应不同类型但是具有类似动力链的全新运动。

综上所述，假使我们能够量化且学到人类动力链的潜在发力结构，我们便可以将人类的运动泛化迁移能力迁移到任何形态的躯体当中。而当这种泛化性质迁移到机器人硬件设计和算法研究时，我们应该期待看到动作激活能力的涌现，让出现在大语言模型上的零样本迁移能涌现在机器人学习领域上。

---

## 1. Representation

一段动作时序数据：

```
S_{skeleton, t} = { { s_i, z_ij }_t },  where
  s_i  → a joint
  z_ij → coordinate between 2 joints
  t    → time sequence
```

---

## 2. Attention 怎么设计?

采用 Axial Attention 的形式，先做 spatial pass，再做 temporal pass，复杂度为 `O(N²T + NT²)`。N=24 关节，T=100 关键帧时：

```
Axial = 24² × 100 + 24 × 100² = 57,600 + 240,000 ≈ 300K
```

```
时间 →     t1            t2            t3 (待预测)
关节 ↓   [j1 j2 j3]    [j1 j2 j3]    [j1 j2 j3]
```

- **t1 帧内**：j1 ↔ j2 ↔ j3（全连接双向） → **t2 帧内**：j1 ↔ j2 ↔ j3（全连接双向）。Causal（可见过去）。
- **t3**：只能被预测，不能作为 key。

### Axis 1 · Spatial Attention (joint dim within frame) — bidirectional

空间维度不采用 dot-product，直接用 pair representation `z_ij` 作 attention bias：

```
Attn(i, t) = softmax( (Q_i · K_j) / √d  +  z_ij )
```

其中 `z_ij` 是全局关节之间位置三角 addition 偏置。要不然采用 **Outer Product Memory**（OPM；`s_i ⊗ s_j`），要不然采用 **Triangle Multiplicative Update**（TMU；`z_ij = z_ik ⊙ z_jk`, `z'_{l+1}{ij} = z_ij + z_ki ⊙ z_kj`），要不然采用 **Full**：每层先做 OPM，再做 TMU。反正我之前跑 Ablation 实验的时候没有发现什么很大区别。

> **题外话**：一开始我想过用类似于 GMR 那种 `z_ii±1` 的 Kinematic Tree 的贝叶斯先做偏置，但我害怕只关注 parent joint 和 child joint 模型没法学到全身 synergy。比如，当我的手尝试去做一个"需要蹲下才能够得到门把手"的动作的时候，手需要往下够的意图传递不到膝盖上。这个 thesis 我还完全没有做过实验，后面应该做一下。或者把 GMR 当成一个 gate 去更新 `z_ij`。GMR 定义和 GMR 作为先验更新 `z_ij` 的方式如下。

GMR 的起点是对 **输入和输出的联合分布** 建一个 GMM，而不是分别建模：

$$
p(\xi^I, \xi^O) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}\left(
\begin{bmatrix} \xi^I \\ \xi^O \end{bmatrix};
\begin{bmatrix} \mu_k^I \\ \mu_k^O \end{bmatrix},
\begin{bmatrix} \Sigma_k^{II} & \Sigma_k^{IO} \\ \Sigma_k^{OI} & \Sigma_k^{OO} \end{bmatrix}
\right)
$$

- `ξ^I = 上半身关节角度（观测到的）`；`ξ^O = 下半身关节角度（要预测的）`，或者：
- `ξ^I = 人体关节角度`；`ξ^O = 机器人关节角度（retargeting）`。

协方差矩阵分成四块，其中 `Σ_k^IO` **是 synergy 的核心** —— 它直接编码了输入关节和输出关节之间的协同强度。

```python
# 用 GMR 拟合训练数据的全身关节分布，然后相关强度 threshold 决定哪些 z_ij 保留
gmm.fit(all_poses)             # poses: [N_frames, N_joints * D]
Sigma = gmm.covariances_       # [N_joints, N_joints] 的相关强度
threshold = 0.1
mask = (Sigma > threshold)     # 数据驱动的稀疏 synergy 图
```

### Axis 2 · Temporal Attention (keyframe, across frames) — causal

采用 RoPE 时序编码：

```
Attn(i, t → i, t') = softmax( R(t) Q_{i,t} · R(t') K_{i,t'} / √d ) · M_{t,t'}
```

其中 `M_{t,t'} = 1[t' ≤ t]` 是因果 mask。另外，时间信息还会同时喂给 Motion Module（扩散 / Flowmatching 输出头）来重建两个关键帧骨架之间的连续、平滑动作。

---

## 3. 让模型学会捕捉关键时序 — 学会自己 clip 动作序列

给模型一个参数让他自己 clip 出需要被捕捉的关键动作，让模型学会运动相位边界检测。

```
input:  S_{skeleton, t}   = { { s_i, z_ij }_t }
output: S_{skeleton, t+1} = { { s_i, z_ij }_{t+1} }
```

**初步想法**：在 Transformer 的每个位置输出一个额外的标量 `g_t ∈ [0, 1]`，表示"这一帧是否是关键帧"，用一个 sparsity 正则（比如 L1 或者 straight-through estimator 的 hard gate）控制密度。这样关键帧密度完全由模型从数据中学会 —— 快速变化的动作段关键帧密，稳定的动作段关键帧稀。

不采用标准等间距的 pos encoding，使用连续时间 RoPE，不用帧序列号 0, 1, 2, 3...，采用实际时间戳 `t` 作为旋转角度的参数：

```
θ = timestamp_in_seconds · (1 / 10000^(2i/d))
```

这样 Δt = 0.1s 和 Δt = 0.5s 两个关键帧，它们之间的 attention bias 自然不同，模型能感知到"这两帧之间经过了多久"。

---

## 4. 如何学习一个 Synergy Graph?

借鉴 AlphaFold 的做法 —— **single pair representation and pair2pair representation** —— 这也是为什么这个模型要叫 AlphaMotion 的原因。

两条表示流量：

```
s_i  : single joint,  [N, d_s]
z_ij : joint2joint,   [N, N, d_z]
```

两条流互相通过 Triangle Attention 更新：

```
z_ij ← f(z_ik, z_kj)        — 对所有 k 求和
```

如果 `i` 和 `k` 有协同关系、`k` 和 `j` 有协同关系，那么 `i` 和 `j` 的协同关系可以被推断出来：

- 髋-膝 协同 + 膝-踝 协同 → 更新 **髋-踝** 协同
- 肩-肘 协同 + 肘-腕 协同 → 更新 **肩-腕** 协同
- 骨盆-脊柱 + 脊柱-肩带   → 更新 **骨盆-肩带**（步态摆臂的来源）

如果 `z_ij` 被充分学习，那么我们应该能期待从 Synergy Graph 中学会从空间和时间上 Mask 预测的能力。

> **输入**：上半身动作序列 → **输出**：下半身动作序列。给出任何 joint 不完整的动作 Sequence 都应该能补全。

```
input:  S_{skeleton, t}   = { { s_1, s_2, s_3, ...{S_j}..., s_k, z_ij }_t }
output: S_{skeleton, t+1} = { { s_1, s_2, s_3, {S_j},        s_k, z_ij }_{t+1} }
```

带来的收益是巨大的：任何互联网上动作缺失的动作视频数据都可以被补全，用来训练更大的模型。

---

## 5. Encoder and Decoder

由于我们直接学习 Embedding，定位是一个 Foundation Model，因此我们需要 AlphaMotion 同时做判别器和生成器。可以使用 **UniLM** 的做法，用不同的 Mask 切换模式，共享同一个 Latent Space。

**生成模式**：因果 mask（预测下一关键帧）

```
[t1, t2, t3] → 预测 t4
```

**判别模式**：双向 mask（看完整序列做分类）

```
[t1, t2, t3, t4] → [CLS] → 开环 / 闭环动作
```

如果模型预测下一帧的置信度极高，说明运动高度可预测，**开环**。如果预测误差大，说明有外部反馈介入，**闭环**。

- **开环**：`p(frame_{t+1} | frame_{1:t})` → 高置信，低熵
- **闭环**：`p(frame_{t+1} | frame_{1:t})` → 低置信，高熵，且实际 `frame_{t+1}` 是"纠偏"方向

这个作为未来和 VLA / WAM 的运动嵌入 Foundation Model 很重要。大脑可以直接读取这个信号来决定输出开环 embedding 还是闭环 embedding。

---

## 6. Token 设计

```jsonc
{
  "skeleton_encoding": {
    "philosophy": "root 独立编码全局状态, body 关节编码相对于 parent 的局部旋转 (kinematic chain local, 不是相对于 pelvis)",
    "reason": "parent-relative 是 SMPL/BVH 标准, 可以直接做 FK, synergy 学的是这些局部旋转之间的关系",

    "root_token": {
      "content": ["global_position_xyz (3D)", "global_orientation_6d (6D)"],
      "dim": 9,
      "coordinate": "world frame",
      "note": "这是序列里第一个 token, 对应 pelvis"
    },

    "body_token": {
      "content": [
        "joint_rotation_6d_local (6D)",
        "delta_t (1D, 到下一个关键帧的时间间隔, 模型自己预测)",
        "contact_flags (4bit, left_foot/right_foot/left_hand/right_hand)"
      ],
      "dim": "6 + 1 + 4 = 11 per joint",
      "coordinate": "parent joint local frame",
      "note": "23 个 body joint, 不含 pelvis"
    },

    "full_keyframe_token": {
      "dim": "9 (root) + 23×11 (body) = 262D per keyframe",
      "sequence": ["<BOM>", "kf_1", "kf_2", "...", "kf_N", "<EOM>"]
    }
  },

  "special_tokens": {
    "llm_standard": {
      "<BOS>":              "Beginning of Sequence, 序列开始",
      "<EOS>":              "End of Sequence, 序列结束",
      "<PAD>":              "Padding, 填充到固定长度",
      "<UNK>":              "Unknown, 词表外的 token",
      "<MASK>":             "BERT 风格, 被 mask 掉的位置, 用于 masked prediction 训练",
      "<SEP>":              "Separator, 分隔两个片段 (BERT 用于 sentence pair)",
      "<CLS>":              "Classification token, 序列级别的聚合表示, 接 head 做分类",
      "<|endoftext|>":      "GPT 风格的文本结束符",
      "<|im_start|> / <|im_end|>": "ChatML 格式, 标记对话角色边界",
      "<think> / </think>": "o1/R1 风格, 标记 chain-of-thought 推理区域",
      "<tool_call> / <tool_response>": "工具调用边界 token"
    },

    "motion_specific": {
      "<BOM>":         { "full": "Beginning of Motion", "role": "运动序列开始, 类比 BOS, 初始化 synergy graph z_ij 为先验状态", "analogy": "BOS" },
      "<EOM>":         { "full": "End of Motion",       "role": "模型输出这个 token 时停止生成, 自主决定序列长度",                "analogy": "EOS", "note": "这是你实现无限长/自主停止的关键" },
      "<KF>":          { "full": "KeyFrame marker",     "role": "标记当前 token 是关键帧 (而非普通帧), 模型学会在动作相位边界输出", "analogy": "句子中的句号—模型学什么时候该打" },
      "<PHASE>":       { "full": "Phase Transition",    "role": "标记运动相位切换 (swing→stance, 准备→执行→收回), 帮助 diffusion 模型知道两关键帧之间的插值难度", "analogy": "段落分隔符" },
      "<CONTACT>":     { "full": "Contact Event",       "role": "标记接触事件发生帧 (脚落地、手抓物), 是 foot sliding 的直接抑制信号", "analogy": "标点中的感叹号, 强调这帧的物理约束" },
      "<MASK_UPPER>":  { "role": "上半身被 mask, 模型从下半身推断上半身", "use": "训练时随机施加, 增加数据利用率" },
      "<MASK_LOWER>":  { "role": "下半身被 mask, 模型从上半身推断下半身", "use": "处理只有上半身的互联网视频数据" },
      "<STYLE>":       { "role": "运动风格条件 token, 后接 style embedding (老人走路 vs 运动员走路)", "analogy": "<|system|> 在 chat 模型里的角色" },
      "<OPEN>":        { "role": "标记这段运动是开环的 (预规划, ballistic)",            "use": "open/closed loop 判别训练目标" },
      "<CLOSE>":       { "role": "标记这段运动是闭环的 (有反馈, corrective)",            "use": "同上" },
      "<CONSTRAINT>":  { "role": "后接 waypoint 或末端执行器约束, 注入外部条件",         "analogy": "function call token, 告诉模型后面是结构化的外部输入" }
    }
  },

  "self_determination_mechanism": {
    "variable_length": {
      "how": "模型在每步预测时都有机会输出 <EOM>, 训练时从数据里学会动作自然结束的时机",
      "training_signal": "teacher forcing: 数据里动作结束的地方标记 <EOM>, 模型学会预测它"
    },
    "variable_keyframe_density": {
      "how": "每个 body_token 里有 delta_t 字段, 模型预测到下一关键帧的时间间隔",
      "effect": "快速动作 (起跳) → 小 delta_t → 密集关键帧; 慢速动作 (站立) → 大 delta_t → 稀疏关键帧",
      "training_signal": "ground truth delta_t 从数据里的相位边界自动标注"
    },
    "sequence_example": [
      "<BOM>",
      "<STYLE> athletic_walk",
      "<CONSTRAINT> waypoint=[0,0,0]→[5,0,0]",
      "kf_1 {root: ..., joints: ..., delta_t: 0.3, contact: [1,0,0,0]}",
      "<CONTACT>",
      "kf_2 {root: ..., joints: ..., delta_t: 0.5, contact: [0,1,0,0]}",
      "<PHASE>",
      "kf_3 {root: ..., joints: ..., delta_t: 0.3, contact: [1,0,0,0]}",
      "<EOM>"
    ]
  }
}
```

---

## 7. Loss 设计

```
L_total =   λ₁ · L_root_pos     # 全局位移, 单位是米
          + λ₂ · L_root_rot     # pelvis 朝向, 6D rot loss
          + λ₃ · L_body_rot     # 每个关节局部旋转, 6D rot loss
          + λ₄ · L_fk_pos       # FK 推算出的全局关节位置, 单位是米
          + λ₅ · L_delta_t      # 关键帧时间间隔预测
          + λ₆ · L_contact      # 接触 flag, BCE loss
          + λ₇ · L_velocity     # 相邻关键帧之间的速度平滑
```

其中，`delta_t` 是让模型自主决定关键帧密度的核心字段 —— 它不是外部给的，是模型输出的一部分，和关节旋转一起预测。模型从数据里学会"这个动作阶段需要多细的粒度"，不需要你手工设定。

---

## 8. 涌现非人类动力链动作

**直腿走路** —— 要求模型不只是记住人类走路，而是 **理解"在约束 c 下实现目标 g 的最优协同策略"**。

这需要 `z_ij` 不只是记录"人类通常怎么协同"，而是记录"协同的功能原因"。

一个可能的方向：在 `z_ij` 的更新里加入一个 **约束条件输入**，比如"关节 k 被锁死在角度 θ"，让模型在这个约束下重新计算所有其他关节的协同策略。

训练信号可以来自：

1. 主动生成部分关节被约束的合成数据（在 AMASS 里强制某些关节固定，看运动如何重新分配）
2. 仿真环境里带约束的 RL rollout 数据

---

## 9. 架构创新尝试 / Pivot

借鉴 Kaiming He 的 **ELF**，Keyframe 的生成也许也可以甚至更适合用 Flowmatching 计算 embedding，只在最后一步 decode 成离散的 token，反正从 MotionGen 的角度来说没有 KV Cache。

唯一一个问题是，采用 FM 的输出头模型可能没法自己决定输出多长的 Clip 和自监督提取 Keyframe。但也许可以只让 FM 输出 keyframe 上的关节序列 `{S_i (z_ij)}`，整个 Sequence 还是自回归的。
