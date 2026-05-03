# MotionFormer: 基于 Pair-First 归纳偏置的跨实体全身机器人控制器

*从 AlphaFold 到机器人全身协同的架构迁移*

---

**作者**：Yudong Lei
**日期**：2026-04-20
**状态**：研究愿景文档（非评审稿）
**对应代码库**：`rot/kimodo/`（Kimodo 本地镜像）、`be water, robot/papers-docs/`（文献）

---

## 目录

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Background](#2-background)
- [3. Method: MotionFormer 架构](#3-method-motionformer-架构)
- [4. Spatial Perception 模块](#4-spatial-perception-模块)
- [5. Data Pipeline](#5-data-pipeline)
- [6. Training Objective](#6-training-objective)
- [7. Warm-Start from Kimodo](#7-warm-start-from-kimodo)
- [8. Compute Budget](#8-compute-budget)
- [9. Experimental Roadmap](#9-experimental-roadmap)
- [10. Comparison with Concurrent Work](#10-comparison-with-concurrent-work)
- [11. Discussion](#11-discussion)
- [12. Conclusion](#12-conclusion)
- [Appendix A: 论文阅读 Syllabus](#appendix-a-论文阅读-syllabus)
- [Appendix B: Cloud GPU 价格速查](#appendix-b-cloud-gpu-价格速查)
- [Appendix C: Tooling](#appendix-c-tooling)

---

## Abstract

本研究针对**跨实体人形机器人全身控制**这一核心问题，提出一个名为 **MotionFormer** 的架构假设：全身动作的本质是**关节与关节之间的协同（pairwise coordination）**，而不是每个关节独立决策的串行集合；因此，用于建模该问题的 backbone 应当显式地把"关节-关节关系"作为一等公民的 tensor 表示（pair representation），而不是像当前主流 VLA / 扩散策略一样把关节压进每一帧的 feature 向量中依赖 MLP 隐式学习。

该假设直接来源于 AlphaFold2 的 **Evoformer** 架构——它证明了"输出结构本质是 pairwise 的"这一观察可以通过将 pair tensor 提升为持久化 backbone 的方式大幅改善学习效率与归纳偏置的匹配度。我们将 Evoformer 的四个核心创新（MSA / pair 双 tensor、axial attention、triangle attention、recycling）映射到机器人全身动作建模，并提出一套与之匹配的训练范式：**倒装 Physical Intelligence 的训练顺序**——先用自监督 Masked Motion Modeling 训好一个覆盖"动作 style manifold"的 action expert（相当于小脑），然后冻结它，再训练一个通过 cross-attention 与之对话的 VLM-like brain（相当于大脑）。

我们的中心 thesis 是：**pair-first 归纳偏置 + brain/cerebellum 倒装训练，可同时解决（a）全身协同的 sample efficiency，（b）跨异构形态的 zero-shot 泛化**。跨形态的证据来自 AlphaFold 本身处理变长蛋白的能力——pair tensor 在变节点数上是原生支持的，不需要重新定义网络。

本文档的其余部分展开论证该 thesis 的动机、架构、数据 pipeline、训练目标、算力预算、实验路线图，并将之与 Kimodo、Skild、π0、HEX 等并发工作定位对比。我们明确提出一系列**可证伪的预测**，第一个仅需 2-4 周在合成 motor synergy 数据集上即可 go/no-go，避免在错误方向上浪费资源。

---

## 1. Introduction

### 1.1 当前范式的二元对立

当下机器人学习领域存在一个事实上的技术二分：

- **"大脑" 派（VLA）**：以 π0 / π0.5 / HEX / RT-2 为代表。用一个 Vision-Language Model 读图读指令，输出直接的关节动作（或动作 token）。训练需要 `(video, language, action)` 三元组 paired data，瓶颈在数据采集成本。
- **"小脑" 派（Diffusion Policy 系）**：ACT、Diffusion Policy、Flow Matching 等。在关节轨迹空间内做条件去噪，预测一个速度场去最小化期望动作与生成动作的位差。训练只需示教数据，但语义泛化能力弱。

产业上最常见的部署方式是把两者拼起来：**VLA 做高层路径规划 / 关节规划，ResMimic（或类似 MaskedMimic）做底层执行**，中间通过 ROS2 这种消息总线通信，**不存在任何神经网络层面的统一 backbone**。

这种拼接有两个根本性缺陷：

1. **协同信息的丢失**：VLA 输出"去那个位置"的时候，并不知道底层控制器会因为力矩分配不匀而产生的不自然步态；底层控制器也不知道上层为什么选这条路径。两者之间只有 token/轨迹级别的通信，**没有共享的 motor primitive representation**。
2. **数据类型的割裂**：VLA 需要 paired data，小脑需要轨迹数据。人类的 mocap、第一人称 POV 视频、武术教学视频等**非配对但极具 style 信息**的数据源在两侧都无法高效利用。

### 1.2 几何空间约束即 "Style"

我们提出一个观察：**任何全身机械系统（包括人体、机器人、汽车、软体操纵器）的 joint state 都受到强几何空间约束，这些约束本身在高维空间里对应一个抽象的"动作 style"流形**。

- "拿杯子"和"开门"的 embedding 距离，应当比"拿杯子"和"上楼梯"要近——因为前者共享"手部抓握 + 腕关节旋转"的协同模式。
- "水平刺剑"和"水平砍刀"的 embedding 距离，应当比"水平刺剑"和"侧踢"要近——因为前者共享同一套上肢发力链。

**现有 Diffusion Policy 与 VLA 都没有显式学到这种 style manifold**。它们学到的是条件去噪，"输入提示 A 就生成动作 A"，但不知道 A 和 B 在 motor level 的亲缘关系。这是一种巨大的浪费，因为人类神经系统显然学到了这种 manifold——否则无法解释为什么学会锤子的人能迅速上手刀、枪、棍、棒。

### 1.3 神经科学动机

我们的架构假设不是空想，有四条神经科学依据：

**（a）Motor synergies / 运动原语**。Bizzi 与 d'Avella 等在上世纪末至本世纪初的一系列工作证明：哺乳动物脊髓层面就存在一组数十维的"动作 basis"，高层运动规划实际是在这些 basis 上做线性或弱非线性组合。换言之，人类并非对每块肌肉独立发指令，而是调用一组预编译的协同 pattern。这直接支持"存在一个低维 style manifold"的假设。

**（b）Tool extension / 身体图式扩展**。Maravita & Iriki (2004) 的猴子电生理实验显示：当猴子熟练使用耙子等工具后，其体感皮层中"手"的感受野会**物理地扩展到工具末端**。李小龙的"武器是身体的延伸"这一直觉性断言因此不是比喻，而是可测量的神经生理事实。推广到机器人设计：一旦 style encoder 学到"抓握"这一原语，理论上可迁移到任何延伸形态——方向盘、门把手、水杯、甚至章鱼博士的触手。

**（c）Cerebellum vs Cortex 的功能分工**。神经科学对空间认知有两条分离的通路：
- **皮层通路**（parietal cortex + hippocampus）——负责**显式的、可语言化的**几何推理（"我在梧桐树下"）；
- **小脑通路**——负责**隐式的、程序性的**空间整合与平衡（走路不摔、手伸准位置、踩稳地面）。

这一区分由喝醉酒的观察自然佐证：酒精首先打击小脑（共济失调最早出现），其次是海马（记忆断片），皮层耐受最强。因此醉汉可以说出"我在某棵树下呕吐"但找不到回家的路——因为**声明式空间推理完好而程序性空间控制受损**。这个二分映射到机器人就是经典的 brain / cerebellum 双层架构，但关键是**两层应当共享一个 motor primitive 接口**，而不是用 ROS 消息拼接。

**（d）Intent encoding 的离散 / 连续混合**。李小龙截拳道教材中将战斗意图按 `{攻击, 截击, 防御} × {jab, cross, hook, side-kick, front-kick, spinning kick}` 编码，武者通过识别对手攻击的编码模式算出反击。这提示我们：人类可以把 high-level intent 压成较低维的离散或连续表示，然后在 motor primitive space 里 decode 成动作。这正是我们要设计的 brain-cerebellum 接口。

### 1.4 中心假设

综合以上，本研究的中心假设有两个，它们彼此独立但相互增强：

**H1（架构假设）**：把 `[joints × joints]` 的 pair tensor 作为 backbone 的一等公民表示，配合 axial attention + triangle attention 进行迭代精炼，能在**sample efficiency** 和**跨形态 zero-shot 泛化**两个维度上优于当前 sequence-first 的 Transformer diffusion。

**H2（训练范式假设）**：相比 π0 / π0.5 的 "VLM 冻结 → action 训练"，采用 "action expert 先训好冻结 → brain 后训并通过 cross-attention 与 action expert 对话" 的倒装顺序，能更好地利用非配对动作数据（mocap、POV 视频），同时让 brain 的任务回归到 VLM 舒适区（"看图听话→输出结构化 primitive 指令"）。

两个假设的交汇产物就是 **MotionFormer**。以下各节展开架构、训练、数据、验证。

---

## 2. Background

### 2.1 全身人形控制谱系

过去三年涌现一批直面全身（whole-body）控制的工作，大致可分为三个子谱系：

**子谱系 A —— 动作模仿派**：
- **HumanPlus** (Stanford, 2406.10454) — 建立了"人类 mocap → humanoid shadow → autonomous skill"的完整 stack
- **OmniH2O** (CMU, 2406.08858) — 用运动学 pose 作 universal control interface，支持 VR / RGB / MoCap / 语言等多种遥操作输入
- **HOVER** (NVIDIA + CMU, 2410.21229) — 多模式策略蒸馏，统一 navigation / loco-manipulation / tabletop 三类模式

**子谱系 B —— 扩散引导派**：
- **BeyondMimic** (Berkeley + Stanford, 2508.08241) — guided diffusion 零样本泛化到新任务
- **SONIC** (NVIDIA, 2511.07820) — 首次将 motion tracking 当基础任务 scale 到 42M 参数 + 700h 数据，得到 universal token space

**子谱系 C —— 全身 VLA 派**：
- **HEX** (北京人形创新中心, 2604.07993) — **第一个明确定位为 whole-body VLA 的工作**，提出"humanoid-aligned experts"和 MoE unified proprioceptive predictor，在 Tienkung 2.0/3.0 上实机验证
- **UniAct** (Tsinghua, 2501.10105) — 定义 universal action space 作为跨机器人的共享原子行为

这几个谱系的共同局限是：**都把关节当独立 token 或压进 per-frame 特征向量**。HEX 的 MoE proprioceptive predictor 在结构上最接近我们想要的 pair representation，但其"expert"是按**身体部位**（手臂 / 腿 / 躯干）划分的分组，而不是按**关节-关节对偶**的全交互表示。

### 2.2 VLA 的两种训练顺序

Physical Intelligence 公司的 π0（2410.24164）和 π0.5（2504.16054）代表当前 VLA 的主流训练方案：

```
PaliGemma VLM (从 internet-scale 数据预训练, 冻结或部分微调)
    ↓
Flow matching action head (从零训练, 依赖 paired 机器人数据)
    ↓
关节动作序列
```

其成功的前提是：**internet-scale 图文数据赋予 VLM 强大的语义泛化，action head 只需学"把语义映射到动作"的最后一步**。但该方案的两个代价：

1. **VLM 的输出瓶颈**：VLM 学的是 sequence-first 表示，要它吐出全身 N 个关节的高带宽协同动作是非常别扭的任务，本质上是在用一把 language-shaped 的工具做 motor-shaped 的活。
2. **数据依赖**：action head 仍然需要成千上万条 `(image, language, action)` paired 数据，而这种数据最难采集（遥操作质量低、真机耗时）。

我们提出**倒装训练**：

```
Action expert (Masked Motion Modeling 预训练, 可用纯 mocap / POV 视频自监督, 冻结)
    ↓
暴露 motor primitive latent (由 style manifold + goal spec 构成的低维连续空间)
    ↑
Brain (VLM-like, cross-attention 到 action expert latent, 后训)
    ↑
Vision + Language + Proprio
```

倒装的好处：

- **数据端**：action expert 预训练可以用海量非配对数据
- **接口端**：brain 的输出不再是 joint command 而是 motor primitive spec，回归到 VLM 擅长的"结构化结果"模式
- **架构端**：两阶段分别优化，避免一个大网络同时承担语义与 motor 的负担

这个思路在概念上与 Skild AI 的 "one brain every robot" 有相似之处，但 Skild 侧重训练范式（domain randomization + student-teacher），我们侧重架构归纳偏置与接口设计。

### 2.3 AlphaFold / Evoformer 架构核心

Evoformer 的四个核心创新是本研究的直接灵感：

**（i）MSA + Pair 双 tensor**：
- MSA representation: `[seqs × pos × C]` —— 每条同源蛋白在每个位置的特征
- Pair representation: `[pos × pos × C]` —— 任意两位置之间的 pairwise 关系

这两个 tensor **跨 48 层 Evoformer block 协同演化**，互相注入信息，共同收敛到一个稳定的"结构先验"。

**（ii）Axial Attention**：把 2D 的 MSA 问题拆成两个 1D attention：
- Row attention —— 同一条序列内、不同位置互相看
- Column attention —— 同一位置、不同序列互相看

计算上避免 `O(seqs² × pos²)` 的平方爆炸，语义上精确对应"共变关系（coevolution）"的提取方式。

**（iii）Triangle Attention + Triangle Multiplicative Update**：
- 几何上，任意三残基 `(i, j, k)` 的距离必须满足三角不等式。这是硬约束，不是学出来的。
- Evoformer 让 pair `(i, j)` 在更新时必须考虑所有经过 `k` 的三角关系，分四个变体（起始边 / 结束边 × 乘法 / 注意力）。
- 这是 **pairwise 问题里最独特的一种归纳偏置**，LLM Transformer 没有对应。

**（iv）Recycling**：Evoformer + Structure module 跑完一轮后，输出的 3D 结构可反哺到下一轮 pair tensor 初始值，整个网络迭代 3-4 次。类似 Newton iteration 的不动点逼近。

### 2.4 vs LLM Transformer：核心差别

标准 Transformer 的 self-attention 会算一个 `N × N` 的 attention matrix（由 `Q · K^T` 得到），**但它只是计算过程的一个中间量，用完就扔**。下一层重新算一个。

Evoformer 的关键决定是：**把这个矩阵提升为一等公民的持久表示**——有独立的 channel 维（`N × N × C`），独立的更新规则（triangle update），独立的几何约束（triangle inequality），跨层持续累积信息。

一句话总结：**LLM 的 attention matrix 是"动词"；AlphaFold 的 pair tensor 是"名词"**。

其他所有 Evoformer 的新设计都是这个决定的自然后果。

### 2.5 Motion Diffusion 两个 Camp

Motion 领域的生成模型分两个技术谱系，经常被混淆：

**Camp A（Kinematic Animation）**：
- 输入输出都在骨架空间（通常是 SMPL/SMPL-X 或其变体）
- 不考虑物理，脚穿地、手穿身等违规用 post-hoc IK 修
- 典型代表：MDM、MotionDiffuse、Tencent HY Motion、**Kimodo**
- 输出形式：几何动画，可 retarget 到游戏角色或作为 policy 训练的参考动作

**Camp B（Physics-based Character Control）**：
- 状态表示是仿真器可直接驱动的 `(joint angles, joint velocities, root pose, contact forces)`
- 输出是 PD target 或 torque，被送入 Isaac Gym / MuJoCo 物理引擎
- 典型代表：AMP、ASE、CALM、**MaskedMimic**、ProtoMotions
- 天然物理合规，脚就踩在地上

Kimodo 是 Camp A 的当前 SOTA，但其产物设计为可导出到 Camp B 的 ProtoMotions 框架训练 physics policy——这两个 camp 因此形成了一个成熟的 pipeline。

### 2.6 RMA / Damage Adaptation 谱系

Deepak Pathak 实验室的 **RMA (Rapid Motor Adaptation for Legged Robots)** (Kumar, Fu, Pathak, Malik, 2021) 是当前"锯腿即时自适应" demo 背后的原始技术。核心三阶段：

1. **Teacher policy（上帝视角）**：在 Isaac Gym 中开数千个并行仿真，每个仿真的机器人参数（质量、摩擦、关节 damping、是否缺条腿）都被随机化。教师策略看所有**特权信息**训练 PPO，学会"在各种奇葩情况下都能走路"。
2. **Student policy（盲人视角）**：只看本体感知（关节角 / 速度 / IMU / 最近动作），通过行为克隆模仿教师。学生**必须从观测历史里隐式推断**特权信息——比如"发 1Nm 力但关节没动 → 这关节坏了"。
3. **Adaptation module**：小型神经网络把观测历史映射到估计的特权参数 embedding，在部署时持续更新。

**关键洞察**：策略并不是"现场学新动作"，而是**训练时见过这个 case（包括断腿），部署时通过观测识别出来并切换**。

Skild AI 将 RMA 产品化和 scale up，并以 "one brain every robot" 为口号。2026 年 1 月 Skild 完成 $1.4B Series C，估值 $14B，投资方包括 NVIDIA、SoftBank、Bezos。Figure AI 近期的 disable-joint demo（Figure 03 髋/膝失效仍能移动）是同一家族技术的变体。

**这对本研究的含义**：Skild 的自适应是**训练范式级别**的成就，backbone 通常是标准 MLP 或 Transformer。MotionFormer 押的是**架构归纳偏置级别**的贡献。两者正交，可以叠加——Evoformer backbone + RMA 训练范式能同时获得"结构先验"和"部署鲁棒"。

---

## 3. Method: MotionFormer 架构

### 3.1 核心思想：Pair-First

MotionFormer 的核心决定只有一个句子：

> **把 `[joints × joints]` 的 pair tensor 作为 backbone 的一等公民表示，而不是从 `[time × joints]` 的 sequence tensor 中隐式推导。**

所有其他设计决定都是这个核心思想的自然推论。

### 3.2 AlphaFold → Motion 概念映射

```
┌────────────────────────────┬─────────────────────────────────────┐
│ AlphaFold / Evoformer      │ MotionFormer                        │
├────────────────────────────┼─────────────────────────────────────┤
│ MSA representation         │ Motion representation               │
│ [seqs × positions × C]     │ [time × joints × C]                 │
│                            │                                     │
│ Pair representation        │ Pair representation                 │
│ [positions × positions×C]  │ [joints × joints × C]               │
│                            │                                     │
│ Row attention              │ Cross-joint attention (同时刻)      │
│ (同序列、跨位置)           │ 同时刻、所有关节互相看              │
│                            │                                     │
│ Column attention           │ Temporal attention (同关节)         │
│ (同位置、跨序列)           │ 同关节、在时序上互相看              │
│                            │                                     │
│ Triangle attention         │ Tri-joint attention                 │
│ (三残基距离约束)           │ (三关节联动 / 闭环运动链约束)       │
│                            │                                     │
│ Outer product mean         │ 关节×时序外积 → pair 初始化         │
│ (MSA → pair)               │                                     │
│                            │                                     │
│ Pair → MSA bias            │ Pair tensor 作为 attention bias     │
│                            │ 注入 cross-joint attention          │
│                            │                                     │
│ Recycling                  │ 输出动作反哺 pair 初始值迭代精炼    │
│                            │                                     │
│ Structure module (IPA)     │ Flow matching action head           │
│ (SE(3) 等变 3D 输出)       │ (输出关节动作序列)                  │
└────────────────────────────┴─────────────────────────────────────┘
```

这个映射不是比喻而是**结构同构**——两边处理的都是"给定一堆两两相互约束的单元，预测它们能同时成立的整体构型"。蛋白质里的单元是残基，机器人里的单元是关节。

### 3.3 vs 标准 Transformer（以 Kimodo 为对标）

Kimodo 是 Camp A 的 SOTA，架构上是两阶段 Transformer encoder denoiser（16 层 × 8 头 × 1024 hidden，总 282M 参数）。它把整帧 263 维特征作为一个 token，attention 只在时间轴上做。**关节间关系由 per-token MLP 隐式学习**。

与此对比，MotionFormer 的 block 包含：

```
┌──────────────────────────────────────────────────────┐
│ MotionFormer Block                                   │
├──────────────────────────────────────────────────────┤
│ 1. Cross-joint attention (row-style)                 │
│    - 输入: MSA tensor [T × J × C]                    │
│    - 沿 J 轴做 attention, 同时刻的关节互相看         │
│    - pair tensor 作为 attention bias 注入            │
│                                                      │
│ 2. Temporal attention (column-style)                 │
│    - 沿 T 轴做 attention, 同关节的时序演化           │
│                                                      │
│ 3. MSA FFN (per-token MLP)                           │
│                                                      │
│ 4. MSA → Pair: outer product mean                    │
│    - 更新 pair tensor [J × J × C]                    │
│                                                      │
│ 5. Triangle multiplicative update                    │
│    - pair(i,j) 用所有 k 的 pair(i,k), pair(k,j) 更新 │
│                                                      │
│ 6. Triangle attention (start / end edges)            │
│    - pair(i,j) attend pair(i,k) 或 pair(k,j)         │
│                                                      │
│ 7. Pair FFN                                          │
└──────────────────────────────────────────────────────┘
× N layers (推荐 8-16 层, 不需要 48 层, 因为 J 比蛋白质残基数小)
```

与 Kimodo 相比，MotionFormer 多出的成本主要在 pair tensor 的维护（`O(J²)` 内存）和 triangle 操作（`O(J³)` 计算）。但 J 在人形机器人上只有 22-40 级别，远小于蛋白质的 100-1000，所以这些 overhead 可接受。

### 3.4 Brain / Cerebellum 分层

完整系统由两层组成：

```
┌─────────────────────────────────────────────────────────────┐
│ Brain 层（cortex，VLM-like，后训）                          │
│ ┌──────────┬──────────┬──────────┐                          │
│ │ Vision   │ Language │ Proprio  │                          │
│ │ (DINO)   │ (LLM)    │ encoder  │                          │
│ └────┬─────┴────┬─────┴────┬─────┘                          │
│      └──────────┴──────────┘                                │
│                 │                                           │
│          Brain Transformer (12-24 layers)                   │
│                 │                                           │
│                 ▼                                           │
│         Cross-attention Query                               │
└───────────────────┬─────────────────────────────────────────┘
                    │ (cross-attention)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Action Expert 层（cerebellum，MotionFormer，预训冻结）      │
│                                                             │
│  Motor primitive latents (Key / Value)                      │
│      ↑                                                      │
│  Evoformer-style backbone (pair tensor + triangle attn)     │
│      ↑                                                      │
│  Proprioceptive input                                       │
│                                                             │
│  Flow matching head                                         │
│      ↓                                                      │
│  Joint action sequence                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 Cross-Attention 接口设计

Brain 和 Action Expert 之间的通信**不经过离散 token 词表**。我们早期考虑过给 VLM 加 `<joint-arms-manipulate>` 这类 token 的方案，但放弃了，理由有三：

1. **粒度粗**：离散 token 表达不了连续的 style 调整（"轻柔地拿"和"猛地抓"应该有连续变化）
2. **形态不通用**：每种机器人都要重新定义词表，违背跨形态初衷
3. **丢失 VLM 的先验**：VLM 已经有成熟的 cross-attention 机制处理异构信号（见 PaliGemma / LLaVA），硬加离散 token 是在抛弃这部分红利

最终接口形式：

```python
# Brain 侧
brain_query = brain_transformer(vision, language, proprio)   # [B, L_brain, D]

# Action Expert 侧 (冻结)
motor_latents = action_expert.get_motor_primitive_space()    # [B, L_motor, D]

# Cross-attention
command = cross_attn(Q=brain_query, K=motor_latents, V=motor_latents)

# Action Expert 接收 command 并 decode 成动作
action = action_expert.decode(command, current_proprio)
```

这个接口的好处：

- **连续、可微**：cross-attention 是连续操作，全程可微分，brain 和 action expert 之间梯度能流通（如果需要 end-to-end fine-tune）
- **表达能力**：cross-attention 可以学出"给予 primitive i 多少权重、primitive j 多少权重"的连续混合
- **类比明确**：这个模式在 AlphaFold 里就是 template module（用已知蛋白的 pair 模板去 bias 当前预测）

### 3.6 架构总览（ASCII）

```
                  ┌─────────────────────────────────┐
   Vision ─────▶ │                                 │
                 │  Brain (VLM-like, 后训)         │
   Language ───▶ │                                 │
                 │  12-24 layer Transformer        │
   Proprio ───▶  │                                 │
                 └─────────────┬───────────────────┘
                               │ cross-attention query
                               ▼
                  ┌─────────────────────────────────┐
                  │  Motor primitive latents        │
                  │  (来自 Action Expert 冻结权重)  │
                  └─────────────┬───────────────────┘
                                │ decoded command
                                ▼
                  ┌─────────────────────────────────┐
                  │                                 │
                  │  Action Expert (预训+冻结)       │
                  │                                 │
                  │  ┌───────────────────────────┐  │
                  │  │ MSA tensor [T × J × C]    │  │
                  │  │ Pair tensor [J × J × C]   │  │
                  │  │                           │  │
                  │  │ Cross-joint attn          │  │
                  │  │ Temporal attn             │  │
                  │  │ Triangle update/attn      │  │
                  │  │ × N blocks                │  │
                  │  └────────────┬──────────────┘  │
                  │               ▼                 │
                  │  Flow matching head             │
                  └─────────────┬───────────────────┘
                                ▼
                      Joint action [T × J]
                                │
                                ▼
                  Physics simulator (Isaac Gym)
                  or real robot PD controller
```

---

## 4. Spatial Perception 模块

### 4.1 SigLIP vs DINO：为什么选后者

Brain 层的 vision encoder 选型并非无关紧要。当前 VLM 生态里最常见的选择是 **SigLIP**（基于 CLIP 风格的 text-image 对比学习），但我们主张用 **DINO** 系（特别是 DINOv2 或后续版本）作为主要 vision encoder。

**两者的本质差别**：

- **SigLIP** 学的是 **semantic alignment to text**：图像特征被 pull 到文本描述的语义邻域。结果是 SigLIP 的 feature 对"狗 vs 猫"这种**语义类别**敏感，但对"物体之间的 3D 几何关系"不敏感——因为文本描述里很少精确描述几何。
- **DINO** 学的是 **self-supervised scale-invariant geometric features**：通过 multi-crop（同一张图片的全局视图和局部视图）之间的特征一致性，DINO 被迫学出**不依赖 text 的纯几何空间表示**。这个表示在 depth estimation、segmentation、keypoint matching 等下游任务上显著优于 CLIP/SigLIP。

**对我们的意义**：机器人操作是 fundamentally 几何任务。要抓到那个杯子，需要知道它在空间的哪个位置，离手多远，朝向如何。语义知识（"那是杯子"）只是 retrieval key，真正执行 manipulation 需要 geometric grounding。

PaliGemma 这类用 SigLIP 的 VLM 能涌现 segmentation / detection，是因为它们通过 `<loc>` / `<seg>` special token 在**任务特定 supervision** 下训练时，LLM decoder 对 vision patches 做了空间 localization。这说明 SigLIP 的特征**可以被强制用于空间任务**，但几何结构是在 decoder 侧学的，而不是 encoder 侧自带。DINO 把这个负担从 decoder 移到 encoder，让整个 pipeline 的空间先验更强。

### 4.2 Cerebellum 式空间感知 vs Cortex 式语义

对应 §1.3 的神经科学分工：

- **皮层式**：用于任务理解（"把红色杯子放到左边抽屉"）—— 由 language encoder + SigLIP-like semantic feature 提供
- **小脑式**：用于动作执行（"伸手、抓、转身、放"）—— 由 DINO-like geometric feature + proprioception 提供

这两类信号在 Brain 层里并不平等：任务起始时**皮层信号主导**（决定要做什么），执行过程中**小脑信号主导**（持续修正空间偏差）。我们的 cross-attention 接口允许这种动态权重变化——brain 在不同时刻对 vision 的不同层 feature 做不同的 attention。

### 4.3 多模态融合

最终 Brain 层的输入是三类 signal 的拼接：

```
brain_input = concat(
    DINO(image),          # 几何空间 feature (多尺度)
    SigLIP(image),        # 语义类别 feature (可选, 做 retrieval)
    LanguageEmbed(text),  # 任务指令 (LLM2Vec 或 CLIP text encoder)
    ProprioEmbed(state),  # 本体感知 (关节角 / 速度 / IMU)
)
```

这里 DINO 和 SigLIP **同时存在**而不是二选一。DINO 提供几何先验，SigLIP 提供语义检索——两者互补。Brain 的 self-attention 会自动学出在什么任务阶段调用哪种 feature。

---

## 5. Data Pipeline

### 5.1 三层数据金字塔

数据策略是本方案的"隐藏优势"。我们把可用数据按精度和规模分为三层：

```
┌──────────────────────────────────────────────────────────────┐
│ Tier 1: 专业 MoCap                                           │
│   - AMASS, CMU, Bones Rigplay (Kimodo 用的)                  │
│   - ~100-700 小时, mm 级精度, 干净                           │
│   - 用于: fine-tune 阶段, 精度校准                           │
├──────────────────────────────────────────────────────────────┤
│ Tier 2: 实验室视频 + 专业 retarget                           │
│   - EgoBody, Human3.6M, 3DPW                                 │
│   - 数百小时, cm 级精度                                      │
│   - 用于: 第二阶段预训练                                     │
├──────────────────────────────────────────────────────────────┤
│ Tier 3: 互联网 POV / 第三人称视频 + SMPL 重建                │
│   - YouTube / InternVid / 武术教学 / 日常 vlog               │
│   - 数千到几万小时, 5-10cm 精度 + 各种 artifact              │
│   - 用于: 第一阶段自监督预训练 (学 style manifold)           │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Video → SMPL 工具链

Tier 3 数据的关键瓶颈是 video→SMPL 重建。2025-2026 年这个方向的 SOTA 选择：

| 方法 | 适用场景 | 核心性质 |
|---|---|---|
| 4D-Humans / HMR 2.0 | 第三人称单人 | Transformer-based, per-frame 质量高 |
| WHAM | 第三人称单/多人 | 时序一致性强, 有 camera-aware 版本 |
| GVHMR | 野外视频需 global trajectory | 同时输出 camera pose 和 world-frame motion |
| TRAM | 长视频人+相机都在动 | SLAM 融合 |
| SLAHMR | 多人长视频遮挡重识别 | 专攻遮挡 |
| EgoLocate / EgoBody | POV 第一人称 | 专门处理 ego-centric camera |
| NLF (Neural Localizer Fields) | 2025 新 SOTA 单人 | 精度超过 4D-Humans |

对应用场景：

- **武术 / 体操视频** → 4D-Humans 或 WHAM（第三人称）
- **第一人称日常视频**（开门、拿杯子、做饭）→ EgoLocate
- **需要 world-frame 轨迹** → TRAM 或 GVHMR

### 5.3 Video SMPL 的四种 Artifact

Video-reconstructed SMPL **不是 MoCap**。误差级别差一个数量级，四类 artifact 要处理：

1. **Foot skate（脚打滑）**：重建的脚看起来踩地但每帧在微微滑动。Diffusion 模型会把这 artifact 一起学进去，导致生成的动作永远带脚打滑。缓解手段：足接触标签作为显式 supervision 信号，训练时对脚打滑加惩罚。
2. **Jitter（抖动）**：per-frame 方法每帧独立预测，关节位置在 1-2cm 内抖。Temporal smoothing 能减轻但引入 lag。WHAM / GVHMR 原生时序方法好一点。
3. **遮挡幻觉**：手被身体挡住时，模型"猜"一个合理的手位置，通常是训练集 mean pose。要靠多视角或时序上下文消除。
4. **手部垃圾**：SMPL 本体手部是简化 rigid linkage。精细操作必须用 SMPL-X + 专门 hand tracker（HaMeR、HaMuCo 等），和身体 pose 做一致性融合。

**处理策略**：在 loss 里按置信度加权。对每帧每关节估计一个重建 confidence（由重建方法自己输出，或用集成方差估计），在 MSE loss 里用 `1 / (σ² + ε)` 作权重，让高置信度关节主导学习，低置信度关节被 down-weight。

### 5.4 Octopus Doctor 的 SMPL Retargeting

我们的自研机器人 **章鱼博士**（Octopus Doctor）是一个 **7DoF + EE** 人形同构机械臂，通过额外加装 j3、j4 电机，改造为**三段式刚性机械臂**（两个大臂 + 一个小臂）。最终自由度约 9 DoF。

关键观察：**人类手臂（含手掌）本身是三段结构**：
- 段 1：肩 → 肘（上臂 / humerus）
- 段 2：肘 → 腕（前臂 / radius-ulna）
- 段 3：腕 → 指尖（手掌 + 手指作为 rigid body 或多 DoF body）

这与章鱼博士的拓扑**一一对应**。因此我们可以把 SMPL 的手臂运动 retarget 到章鱼博士：

```
SMPL 手臂关键点            →   章鱼博士 9 DoF
┌─────────────────────┐        ┌─────────────────────┐
│ shoulder  [x,y,z]   │ ─映射→ │ base 坐标           │
│ elbow     [x,y,z]   │ ─IK─→  │ 段 1/段 2 交界 joint│
│ wrist     [x,y,z]   │ ─IK─→  │ 段 2/段 3 交界 joint│
│ hand/palm [x,y,z,R] │ ─IK─→  │ EE (target)         │
└─────────────────────┘        └─────────────────────┘
```

IK 求解器（推荐 **PyRoki** 或 **mink**）把 EE 目标 + 中间关键点作为约束，求解 9 DoF 的关节角度。多余的 2 DoF（9 - 7）作为 redundancy 处理，可以附加二次优化目标：
- 最小化关节能耗（避免大转动）
- 避免 kinematic singularity
- 保持"人类 style"（和参考 posture 的距离最小）

**预先处理的坑**：

- **段长差异**：人上臂 ≈ 前臂 ≈ 10-25cm；章鱼博士段长不同。先按长度比 **scale 参考 SMPL 骨架**，再做 IK，否则会"缩着动"或无解。
- **关节限位**：人肩 3 DoF 球窝但有生理限制，章鱼博士是三个串联 revolute。IK 必须加关节限制。
- **冗余消解**：9 DoF 拟合人的 7 DoF 有 2 个冗余维度。这 2 维怎么用决定了章鱼博士的"style"——可加 style regularizer。

### 5.5 训练 Curriculum：从脏到净

标准做法是**从 Tier 3 开始，逐步收敛到 Tier 1**：

```
Stage 0: Tier 3 自监督 (数千小时互联网视频 → SMPL)
         目标: Masked Motion Modeling, 学 style manifold 粗结构
         时长: ~1-2 周
         
Stage 1: Tier 2 混合 (Tier 3 + 实验室视频)
         目标: 精度提升, 空间一致性
         时长: ~1 周
         
Stage 2: Tier 1 fine-tune (纯 MoCap)
         目标: 最终精度校准, 去除 Tier 3 的 artifact
         时长: ~3-5 天
```

这个顺序的关键是**不能反过来**：如果先在 Tier 1 小数据上训，再加入 Tier 3 大数据会造成分布 shift，Tier 1 学到的精细能力被冲掉。"脏→净"的 curriculum 在 NLP（预训练 → fine-tune）和 vision（大规模 noisy → clean refine）都验证过，motion 领域 SONIC 部分在用，但没人做到极致——这里是我们的操作 niche。

---

## 6. Training Objective

### 6.1 Action Expert 预训练：五个候选

Action expert 需要在海量非配对动作数据上自监督预训练。五个候选目标函数，各有代价：

**（a）纯重建（autoencoder）**：
- 优点：简单，无需任何 label
- 缺点：会学出一个 style space 但**不保证 metric 结构**——"拿杯子"和"开门"可能在 latent 里离得很远
- 评价：作为 baseline 可以，但不是主打

**（b）对比学习**：
- 要求：同类动作的 pair 标签
- 优点：直接拉近同类 embedding
- 缺点：互联网 POV 视频没有类别标签；要靠人工或半监督
- 评价：作为辅助 loss 可以加

**（c）时序 self-supervised（邻近帧相似）**：
- 假设：时间上邻近的 motion segment 应该 style 相似
- 优点：完全无监督
- 缺点：只能学"motion continuity"，不保证"semantic similarity"——两段完全不同动作的"相邻过渡帧"会被错误拉近
- 评价：补充信号，不能独立支撑

**（d）Video-text alignment**：
- 用视频描述作监督
- 优点：最强的 semantic signal
- 缺点：引入了语言，和"先 action 再 brain"的 clean 分离有冲突；且 video-text 数据有偏（互联网视频 captioning 质量参差）
- 评价：可用但要小心

**（e）Masked Motion Modeling（MMM）** ← **推荐为主**

这是 MaskedMimic / Kimodo 都在用的目标，原理类似 BERT 的 MLM：

```
完整参考动作 X = [T × J × C]
       ↓
随机 mask 策略
  ├─ mask 整段时间 (inpainting)
  ├─ mask 部分关节 (只给 "手的轨迹", 预测全身)
  ├─ mask 稀疏关键帧 (只给 t=0, 10, 20 的状态)
  ├─ mask 特定 modality (只给 root 轨迹, 不给 joint)
  └─ mask kinematic chain (mask 关节 + 所有下游, 模拟关节故障)  ← 必须有
       ↓
[masked X] + [mask indicator] → MotionFormer → 预测完整 X
```

**为什么 MMM 是最佳选择**：

1. **完全自监督**，不需要 label
2. **训练时见过所有约束子集**（时间 mask、关节 mask、关键帧 mask、modality mask、kinematic chain mask 各种组合），推理时 brain 通过 cross-attention 给部分约束就能让 action expert 补全——和 brain/cerebellum 接口目标完全吻合
3. **Mask 策略多样性 = primitive manifold 覆盖度**，可以直接调参控制
4. **有工业 validation**（MaskedMimic、Kimodo）

**Kinematic chain masking 的独特价值**：

前四种 mask 是"信息缺失"的模拟——随机遮住一部分让模型补全。**Kinematic chain masking 是"物理故障"的模拟**：mask 掉某个关节，同时把它在 kinematic tree 下的所有 descendant 一起 mask（mask 膝关节 → 自动连带 mask 小腿、脚踝、脚掌）。这直接对应 Skild / Figure 的 "disable joint adaptation" 场景。

没有这种 mask，fault-tolerant locomotion 只是**推理时的希望**，不是**训练时的监督信号**。训练时如果从没见过"膝盖断了"这个 case，部署时突然发生就只能靠分布外泛化，成功率是赌博。Kinematic chain mask 是把这个 case 显式写进训练目标，让 action expert 学会"在上游关节补偿下游缺失"的通用策略。

这一点对应我们 §10.2 讨论的 Skild RMA 机制在 MotionFormer 里的实现方式——我们不用 teacher-student 蒸馏 + domain randomization，而是用 kinematic chain masking 作为 MMM 的一个 mask mode 直接训到 action expert 里。

### 6.2 Brain 训练

Brain 训练在 action expert **冻结**后开始。Brain 的 loss 由两部分组成：

**主 loss：任务完成**
- Downstream task 的成功信号（抓起杯子、打开门、折叠衣物）
- 可用行为克隆（BC）或强化学习（RL）

**辅助 loss：接口对齐**
- Brain 生成的 command（cross-attention 输出）应当落在 action expert 的 motor primitive manifold 上
- 用一个 KL-like 约束：`brain_command` 的分布不偏离训练时见过的 motor primitive 分布太多

**具体训练信号来源**：

- **遥操作数据**：`(image, language, motor_primitive_label)` —— 把人类遥操作时调用的 primitive 作为 label
- **SMPL 重建数据**：从 video 反推出 motor primitive，作为 brain 的 target
- **RL finetune**：最后一步用 physics simulator 训 brain 的 long-horizon planning

### 6.3 Style Manifold 的 Auxiliary Loss

为了显式促成"拿杯子近开门、远上楼梯"的 metric 结构，在 MMM 基础上加一个**对比学习 auxiliary head**：

```python
# 从 pair tensor 或 motor primitive latent 投影到 style embedding
style_z = style_projection_head(action_expert.pair_tensor)   # [B, d_style]

# 对比 loss (例如 SimCLR / InfoNCE 风格)
# Positive pair: 来自同一段 motion 的两个子 clip
# Negative pair: 不同 motion 的 clip
contrastive_loss = InfoNCE(style_z_anchor, style_z_positive, style_z_negatives)

total_loss = MMM_loss + λ * contrastive_loss
```

Positive pair 的构造可以用**时序邻近**（同一段动作的两个子 clip）或**video-text 亲缘**（描述相似的两段动作）。这个 auxiliary head 可以训完就丢，或者保留作为下游 retrieval 接口。

---

## 7. Warm-Start from Kimodo

### 7.1 Kimodo 参数分布

Kimodo 开源 checkpoint 是启动 MotionFormer 训练的最佳 warm-start 来源。其 282M 参数大致分布：

| 模块 | 参数占比 | 说明 |
|---|---|---|
| Pose 输入 embedding `[263 → 1024]` | ~0.3% | 把每帧 263 维特征投影到 hidden |
| 文本条件化（LLM2Vec 投影） | ~2% | 4096 维 text embedding → 1024 |
| 位置编码 + register tokens | ~0.1% | 时间轴 positional + 49 空 token |
| **Self-attention 权重** | ~40% | 时间轴上的 attention 机器 |
| **FFN 权重** | ~55% | Per-token MLP 加工 |
| Output head `[1024 → 263]` | ~0.3% | Hidden → pose 特征 |
| LayerNorm 等 | <1% | 归一化 |

关键观察：**Transformer 的大部分参数（~95%）在 self-attention 和 FFN**。这两类参数学到的是"处理一段序列的 token 的通用 know-how"，不是任务特定的知识。因此**可大比例迁移**到 MotionFormer。

### 7.2 可移植 vs 全新模块

MotionFormer 相比 Kimodo **新增**四类模块，约占总参数的 38%：

| 新模块 | 做什么 | 估计参数 |
|---|---|---|
| Pair tensor 初始化（outer product mean） | 从 MSA 造 pair | ~1-2M |
| Column attention（时序轴） | 同关节跨时互看 | ~64M |
| Triangle multiplicative update | pair 的自洽更新 | ~32M |
| Triangle attention | pair 的 attention | ~64M |
| Pair → MSA bias | pair 反哺 MSA | ~8M |

合计新参数 ~170M，加上原 Kimodo 282M，总 ~450M。

**可移植的**（约 60%）：

- Pose 输入 embedding（可能需要调整 token 粒度）
- Text 条件化投影
- FFN 权重（每层的 per-token MLP 可直接 copy）
- Self-attention → Cross-joint attention（语义有偏移但权重结构一致）
- Output head

### 7.3 三阶段 Fine-Tune 策略

**Phase 2a（冷启动新模块，1000 steps）**：

```python
freeze(kimodo_inherited)        # FFN, self-attn, embeddings 全部冻结
train_only(newly_initialized)   # pair tensor 相关模块从零学
optimizer = Adam(newly_initialized, lr=5e-4)
```

目标：让 pair tensor 学出合理的初始结构，不要拿噪声的 pair 去污染 Kimodo 的 inherited 权重。

**Phase 2b（解冻 LayerNorm + bias，2000 steps）**：

```python
unfreeze(layernorms, biases)    # 允许统计量适配新 token 粒度
# 其他 Kimodo inherited core weights 仍冻结
train(layernorms + biases + newly_initialized)
```

目标：LayerNorm 的 scale/shift 参数适配新的 activation 分布，因为 MotionFormer 的 token 粒度（时刻, 关节）和 Kimodo 的（整帧）不同，activation 统计量会漂移。

**Phase 2c（全解冻，10K+ steps，分层学习率）**：

```python
optimizer = Adam([
    {'params': kimodo_inherited, 'lr': 1e-5},    # 小 lr 微调
    {'params': newly_initialized, 'lr': 3e-4},   # 大 lr 持续训
])
```

目标：端到端联合微调，让 Kimodo 继承的权重适应 pair-aware 的上下文。

### 7.4 LoRA 替代方案（最保守）

如果资源极度受限或担心 Phase 2c 破坏 Kimodo 权重，可以走 LoRA 路线：

```python
# Kimodo inherited 参数完全冻结
freeze(kimodo_inherited)

# 在它们外面加 LoRA adapter
row_attn_output = kimodo_attn(x) + LoRA_adapter(x)

# 新模块正常训
train(LoRA_adapters + newly_initialized)
```

优点：省资源，不可能破坏 Kimodo 权重。
缺点：LoRA 的 rank 有限，如果 pair tensor 引入的变化太大，LoRA 适应不过来，最终性能上限低于 full fine-tune。

### 7.5 Failure Modes

Warm-start 不是银弹，三种可能的失败模式：

**（a）Token 粒度不兼容**：Kimodo 的 token = 一整帧（所有关节压在特征维），MotionFormer 的 token = 一个 `(时刻, 关节)` 单元。Pose embedding 层必须做 decomposition，让 Kimodo 的帧级 embedding 能被拆成关节级。如果设计不对，warm-start 白搭。

**（b）LayerNorm 统计漂移**：不同 token 粒度下 activation 分布不同，Kimodo 的 LayerNorm 参数可能需要重新估计（Phase 2b 就是为此设计）。

**（c）Row attention 的语义漂移**：Kimodo 的 time attention 学到的是"帧间依赖"，MotionFormer 的 row attention 是"同时刻跨关节依赖"——**架构形式一样但语义不同**。可能出现负迁移。

**对策**：做对照实验。warm-start vs from-scratch 在合成数据上训 1-2 天看 validation curve。如果 warm-start 明显领先，继续；持平或更差，退回 from-scratch。

---

## 8. Compute Budget

### 8.1 三档场景 + 四家云对比

估算基于 Kimodo 的实测配置（16 × A100 80GB，batch 2048，1M steps，282M 参数）作为基线，按以下系数调整：

- MotionFormer 的 Evoformer overhead: ~1.5×（triangle attention + pair tensor）
- H100 vs A100 有效加速比：~2-2.5×（FP16/BF16 + Flash Attention 环境）

**场景 A：MVP 验证（50-80M 参数，100h 数据子集）**

| 配置 | 所需资源 | 时长 |
|---|---|---|
| **本机 RTX Pro 6000 Blackwell 96GB** | 1 张卡 | 1-2 周 |
| 云上加速 | 4 × H100 | 5 天 |

云成本（480 GPU-hour）：

| 云 | On-demand | Spot / Reserved |
|---|---|---|
| Nebius | $1,416 | ~$960 |
| Lambda Labs | $1,200-1,650 | $864 |
| RunPod | $1,291 | $955 |
| CoreWeave | $2,285-2,952 | Reserved 30% off |

**结论**：这一阶段本机就能跑，不花钱。

**场景 B：Match-Kimodo 规模（280M 参数，700h 数据，1M steps）**

估算：16 × H100 × 10 天 = ~3,840 GPU-hour（实际 3,000-5,000）。

| 云 | On-demand（按 4000 hr） | Reserved（~30% off） |
|---|---|---|
| Nebius | **$11,800** | ~$8,200 |
| Lambda Labs | $9,960-$13,760 | $7,000-9,600 |
| RunPod | $10,760 | $7,960 |
| CoreWeave | $19,040 | $13,000+ |

**合理预算：$8-12K**；用 spot 加 robust checkpointing 可压到 $6-8K。

**场景 C：Scale 到 500M-1B**

估算：32-64 × H100 × 3-4 周 = 20K-40K GPU-hour。

| 云 | 总价（按 30,000 hr） |
|---|---|
| Nebius | $88,500 |
| Lambda Labs | $74K-103K |
| CoreWeave | $143K |
| Reserved（任意家） | $50K-70K |

**合理预算：$50-100K**。

**场景 D：Warm-Start Fine-Tune（推荐作为实际起点）**

基于 Kimodo checkpoint 做 warm-start，只训新模块 + 少量 fine-tune：

- 4-8 × H100 × 3-5 天 = **500-1000 GPU-hour**
- **成本：$1.5-3K**

### 8.2 Cloud 选择

- **Nebius** — 性价比之王。$2.95/hr H100，有 InfiniBand，欧洲节点为主。
- **Lambda Labs** — 界面最友好，但热门 H100 常售罄。
- **RunPod Community Cloud** — 最便宜（$1.99/hr H100），SLA 不保证，适合 MVP 不适合 production。
- **CoreWeave** — 最贵但最稳，企业级 SLA，大客户谈下来 ~$3/hr。
- **NVIDIA Brev** — marketplace 模式，底层是 Lambda / CoreWeave / 其他 provider，价格无优势。
- **AWS / GCP / Azure** — GPU 实例 $8-12/hr H100，贵 3-4 倍，除非已有大合约否则忽略。

### 8.3 省钱技巧

1. **Spot / preemptible + 健壮 checkpointing**：每 5-10 分钟 save，能省 30-50%。
2. **Reserved 合约**：承诺 1-3 个月 → 30-40% off。
3. **数据加载不拖后腿**：从 S3/R2 拉数据到本地 NVMe，否则训练被 I/O 卡死。
4. **FlashAttention 3 + bfloat16 + gradient checkpointing**：省 30-50% 显存。
5. **先在 1 张卡上 profile 清楚单步时间**，再决定租多少卡，避免 bad scaling。

---

## 9. Experimental Roadmap

### 9.1 Stage 1（2-4 周）：合成 Motor Synergy Sanity Check

**目标**：验证 pair tensor + triangle attention 能从数据里 emerge 出已知的 motor synergies，而 sequence baseline 不能。

**实验设置**：

- 构造一个合成数据集：已知 K 个 motor synergy basis（用 d'Avella 模型或手动设计），每个动作样本是这些 basis 的线性组合 + 噪声
- Baseline: 标准 Transformer / ACT 架构
- 我方: MotionFormer（Evoformer-style backbone）
- 同参数量、同数据量、同训练 steps
- 测量：模型能否从 pair tensor 的特征值/奇异值分解里 recover 出原始的 K 个 synergies

**Go / No-Go 双指标**（两个都要过）：

**（1）可解释性指标**——pair tensor 是否学到了结构：

- MotionFormer 的 pair tensor top-K 主成分与 ground-truth synergies 的 cosine similarity ≥ 0.8
- Baseline 的同指标 ≤ 0.5

**（2）性能指标**——这个结构是否真有用：

- MotionFormer 达到与 baseline 相同的 validation reconstruction loss，所用 training steps ≤ baseline 的 80%
- 即至少 20% sample efficiency 提升

**为什么要两个都有**：只有指标 (1) 成立而 (2) 不成立的情况意味着——pair tensor 确实 recover 了 synergy 结构，**但这个结构对实际训练没有任何帮助**——架构拿到正确的归纳偏置却转化不成性能，那也是 fail。两个指标一起才能区分"形似"和"实有"。

**如果 fail**：整个架构 hypothesis 不成立，停止后续所有 stage。这是最便宜的 kill switch。

**Stage 1 还要捎带验证的工程问题**：Kimodo warm-start 的 token 粒度 decomposition。Kimodo token = 一整帧 263 维，MotionFormer token = `(时刻, 关节)` 单元每个关节约 8-12 维。这层拆分如果设计错了，后续 warm-start 就是白给。在 Stage 1 合成数据上用 "Kimodo pose embedding 权重 → MotionFormer pose embedding 权重" 的迁移实验跑一次，看 val loss 是否比 from-scratch 快收敛。如果 Stage 1 就暴露 warm-start 负迁移，提前重新设计 decomposition。

### 9.1.bis Stage 1 Retrospective（2026-04-20 夜）

第一次 Stage 1 实验 **提前跑完了**，结果 **初步通过但还不能定论**。记录真实结果、过度乐观风险和下一步修正。

#### 实际执行 vs 原计划

原计划（§9.1）：合成 motor synergy 数据，K=8 个已知 rank-1 synergy，测 pair tensor 能否 recover。

实际执行：因为合成数据 rank-1 可分解，baseline 的 `joint_pe` 直接学到 `ψψᵀ` 结构，pair tensor 的优势无法在该设计下显现。**Pivot 到用 Kimodo 生成 200 条 SOMA77 动作作为训练数据**，评估方式改为：
- 主指标：masked reconstruction 的 sample efficiency（达到 val loss 阈值所需 steps）
- 辅助指标：kinematic chain masking 的 OOD 泛化 loss
- 放弃：pair structure 的数字 recovery metric（Kimodo 数据没有 ground-truth synergy）

#### 数字结果（30 epoch，N_train=160，N_val=40，同 hyper，seed=0）

| 指标 | Baseline | MotionFormer |
|---|---|---|
| 参数量 | 2.40M | 2.18M |
| 最终 train_loss | 0.386 | 0.191 |
| 最终 val_loss | 0.419 | 0.244 |
| val_kc（joint-failure OOD） | 0.484 | **0.173** |
| 达到 val ≤ 0.5 | 25 步 | **15 步** |
| 达到 val ≤ 0.3 | never | 55 步 |
| 达到 val ≤ 0.2 | never | 120 步 |

**定量事实**：MotionFormer 在所有阈值上 sample efficiency ≥ baseline 1.7×，kinematic chain OOD 上误差低 64%。

**定性观察**：pair tensor L2-norm 热图自发呈现 kinematically-meaningful 聚类（左手指簇、右手指簇、手-腿弱耦合）—— 这是没有 ground truth label 情况下的**观察性证据**。

#### 过度乐观的五个风险（必须下一轮 iteration 破除）

1. **数据源是 Kimodo 自己的 diffusion 采样，不是真实 mocap**。MotionFormer 在 Kimodo 分布上赢不等于真实数据上赢，Kimodo 样本可能含 diffusion artifact 而 axial attention 刚好适配这些 artifact 的结构。**破除**：换 AMASS 重跑。
2. **N=160 太小**。小数据上强归纳偏置天然占便宜，在更大数据集上 baseline 可能追上。**破除**：N=1000 和 N=10000 重跑。
3. **Baseline 没调到最佳**。两模型用同一套 hyper，但 12 层扁平 Transformer 的优化难度明显高于 axial MotionFormer（train_loss 震荡说明优化不稳）。**破除**：给 baseline 专门做 warmup + 更小 lr + 更长 schedule。
4. **val_kc 优势可能来自 axial attention 而非 pair tensor**。Row attention 本来就擅长跨 J 轴聚合信息，pair tensor 的独立贡献没有被隔离。**破除**：**消融实验** —— "MotionFormer minus pair tensor"（只保留 row + col attention，去掉 outer product mean + triangle 操作）。这是**最关键的实验**，直接决定你的核心 thesis 是否成立。
5. **单 seed**。需要至少 3 个 seed 才能报有意义的 delta。

#### 真正的 Stage 1 go 判定

本次只算 **cheap sanity check 通过**，下面五项都过才算真 go：

- [x] 基本 pipeline 能跑，MotionFormer 不比 baseline 差
- [ ] Ablation（去 pair tensor）显示 pair tensor 有独立贡献
- [ ] 3 个 seed 下 delta 稳定
- [ ] AMASS 真实 mocap 上保持优势
- [ ] Baseline 公平调参后仍然落后
- [ ] Kimodo warm-start（用 pose embedding 权重迁移）验证 token 粒度 decomposition 设计正确

五项都过 → 启动 Stage 2（跨形态迁移）。任一失败 → 重新审视 hypothesis。

#### 工程副产物

- `experiment/stage1/data.py`：SOMA77 dataset loader + 5 种 mask 模式（含 kinematic chain masking）
- `experiment/stage1/motionformer.py`：450 行 Evoformer-style block（row + col attention、outer product mean、triangle mult、triangle attention）
- `experiment/stage1/gen_kimodo_data.py`：Python API 批量生成 Kimodo 动作（~2 min/200 样本）
- `experiment/stage1/soma_skeleton.py`：SOMA77 kinematic tree + descendant 查询
- `experiment/stage1/analyze_kimodo.py`：训练曲线 + pair tensor 可视化
- **修复**：MotionCorrection C++ 扩展编译失败（pybind11 2.9.1 + Python 3.12 不兼容，升级到 pybind11 3.0.4 解决）。这个 fix 对所有未来用 Kimodo 做数据源或 warm-start 的实验都必须有。

### 9.1.ter Ablation 6 路对比（2026-04-21 凌晨）

紧接着做了消融实验，目的是拆开 MotionFormer 的每个模块，看到底是**哪个部分**贡献了 OOD 改善。六个变体，同 hyper，seed=0，N=200，30 epochs：

| Model | params | val≤0.3 (steps) | val≤0.2 (steps) | val (ID) | **val_kc (OOD)** | train loss |
|---|---|---|---|---|---|---|
| baseline | 2.40M | never | never | 0.419 | 0.484 | 0.386 |
| axial_only | 1.80M | 55 | 120 | 0.349 | 0.283 | 0.179 |
| pair_static | 1.99M | 55 | 95 | 0.304 | 0.234 | 0.139 |
| triangle_only | 2.10M | 55 | **55** | **0.192 ← ID 最佳** | **0.465 ← OOD 几乎等于 baseline** | 0.156 |
| opm_only | 2.12M | 50 | 120 | 0.296 | 0.184 | 0.224 |
| full | 2.18M | 55 | 120 | 0.246 | **0.165 ← OOD 最佳** | 0.195 |

变体定义：
- `baseline`：扁平 `[T·J]` sequence 的标准 Transformer
- `axial_only`：row + col attention（MSA 上的双轴），无 pair tensor
- `pair_static`：上 + 有 pair tensor 但**不更新**（只作为 row attention 的 bias）
- `triangle_only`：上 + triangle mult + triangle attention（pair 内部自洽精炼），但**没有 OPM**（pair 不从 MSA 接收输入）
- `opm_only`：axial + pair + OPM（MSA→pair 外积均值），**没有 triangle**
- `full`：全部打开

#### 三个机制性结论

**结论 1：`axial_only` 已经拿走了 baseline → full 跳跃的大部分**。val_kc 从 baseline 的 0.484 → axial_only 的 0.283（42% 改善）。这部分收益**不是** pair tensor 带来的，是把 motion 当 `[T × J]` 2D 结构而非 `[T·J]` 扁平序列的结果。该归纳偏置在 axial attention 的文献里已有（Reformer / AlphaFold row-col），**不是我们的 novelty**。

**结论 2：Triangle 单独上是一个陷阱**。triangle_only 得到全组**最低**的 train loss (0.156) 和**最佳**的 val (0.192)，但 val_kc 退化到 0.465，**几乎等于 baseline**。这是典型的过拟合签名——对一个"看不到输入" 的 static pair 做 triangle 精炼，等于让模型学一个超强的训练集 memorizer，OOD 完全崩溃。任何未来的"pair 精炼器"（triangle、kinematic-tree bias、SE(3) equivariant 等）都必须和一个"pair 更新机制"配对，单独上是负向的。

**结论 3：OPM 才是关键 OOD 贡献者**。从 pair_static (val_kc 0.234) → opm_only (0.184)，加一个 OPM 就拿到 21% OOD 改善。OPM 做的事是让 pair tensor **按每个样本的输入更新**，而不是保持 input-agnostic。这才是 "pair-first 归纳偏置" 里真正发力的机制。Triangle 在 OPM 存在的前提下再加 10% 改善（opm_only 0.184 → full 0.165），是锦上添花而非独立贡献。

#### Thesis 精修

**旧措辞**："AlphaFold 的 pair representation + triangle attention 是 motion backbone 的正确归纳偏置。"

**新措辞**：

> **Motion backbone 需要一个 *input-conditioned pair tensor refinement* 机制，它由两部分组成：(a) pair tensor 按每个样本的输入更新的 gateway（当前用 OPM），(b) pair 内部自洽的 refiner（当前用 triangle，但可替换为 kinematic-tree-biased attention、SE(3) equivariant 等 robot-native 方案）。两部分缺一不可——只有 gateway 没有 refiner 会欠拟合 ID，只有 refiner 没有 gateway 会在 OOD 上灾难性过拟合。**

这个新措辞的好处：

1. **更 defensible**：不依赖"triangle attention 是最优解"这一未证明的 claim
2. **更贴机器人场景**：机器人有运动链 / 动力学 / 冗余等约束，triangle 的 3D 距离不等式只覆盖其中最弱一项。新措辞允许用 robot-specific refiner 替换 triangle
3. **核心贡献明确**：不是"把 AlphaFold 搬过来"，是"指出 motion 需要 input-conditioned pair refinement 这个抽象模式"，然后给出一种具体实现

#### 卖点重新定位

| 维度 | 旧卖点 | 新卖点 |
|---|---|---|
| 主打指标 | Sample efficiency | **结构化 OOD 鲁棒性**（关节故障 / 跨形态） |
| 对比 π0 | 倒装训练 | 不直接对比 π0（正交） |
| 对比 Skild | 架构归纳偏置 vs 训练范式 | **同一路径**：Skild 用 DR + student-teacher 达到 damage adaptation，我们用结构化 pair refinement 达到同一目标，**两条路互补而非互斥** |
| Defensible niche | 章鱼博士 40 DoF | 章鱼博士 + 任何**关节缺失 / 跨形态 zero-shot** 场景 |

#### 真正的 Stage 1 go 判定（更新）

之前的 5 个 checkbox 现在变成 7 个：

- [x] 基本 pipeline 能跑，MotionFormer 不比 baseline 差
- [x] Ablation 证明 pair tensor 有独立 OOD 贡献（OPM 是关键，triangle 是辅助）
- [x] Ablation 证明 triangle 不是单独优解（triangle_only 证伪）
- [ ] 3 个 seed 下 delta 稳定（41% OOD 改善是真信号还是 seed 抖动）
- [ ] AMASS 真实 mocap 上保持优势（破除 Kimodo 分布 bias 风险）
- [ ] Baseline 公平调参后仍然落后
- [ ] Kinematic-tree-biased refiner 变体能匹配或超越 triangle（从"借来 AlphaFold"进化到"为机器人设计"）

前三项已通过。后四项是明天起的优先工作。

### 9.2 Stage 2（1-2 个月）：跨形态迁移

**目标**：证明 pair tensor 学出的表示在形态变化下仍保持有效。

**实验设置**：

- Train on 4 腿机器狗（MuJoCo humanoid 或 Unitree Go2）
- Freeze backbone, fine-tune action head on：砍掉一条腿变成 3 腿版本
- Measure: fine-tune 数据需求 / 恢复性能

**Go / No-Go 指标**：

- 用 < 30% baseline 数据量就能恢复 ≥ 80% 性能 → 跨形态 transfer 证据强
- 如果需要全量数据才能恢复，说明 backbone 没学到 morphology-invariant 特征

### 9.3 Stage 3（3-6 个月）：章鱼博士 40 DoF

**目标**：在 sequence Transformer 最不擅长的场景（高 DoF 软体结构）展示 pair representation 的优势。

**实验设置**：

- 建好章鱼博士的 URDF（9 DoF 三段式 × 4 臂 = 36 DoF，加底盘可到 40 DoF）
- 在 Isaac Gym 里做触手的 procedural motion generation（触手探、抓、扭）
- 用 MotionFormer 学这些动作
- 对比 baseline（标准 Transformer）在同任务的表现

**Go / No-Go 指标**：

- MotionFormer 在 40 DoF 上的 sample efficiency 明显优于 baseline（至少 2×）
- Pair tensor 的结构可 interpret（哪些关节互相强耦合，对应物理上的 kinematic 链）

**附加价值**：这是 MotionFormer 的 **defensible niche**——没人愿意或能做这种软体形态。

### 9.4 Stage 4（6-12 个月）：完整人形 Demo

**目标**：在真实人形机器人上做完整的"任务完成" demo，和 π0 / HEX 等并发工作可比较。

**实验设置**：

- 用 OpenArm 双臂 + 底盘，或借用 Unitree G1
- 端到端 pipeline: 图文输入 → brain → cross-attention → action expert → 物理执行
- 任务: 抓杯子 / 开门 / 折叠衣物（对标 π0.5 demo）

**Go / No-Go 指标**：

- 任务成功率 ≥ π0.5 在同等任务上的 reported number
- 部分任务（特别是全身协同的）应明显超越

### 9.5 每阶段的资源门槛

| Stage | 资源 | 时长 | 走不过去怎么办 |
|---|---|---|---|
| 1 | 本机 RTX Pro 6000 | 2-4 周 | **停，hypothesis 错了** |
| 2 | 本机 or 云 $500 | 1-2 月 | 重新审视 pair tensor 设计 |
| 3 | 云 $2-5K | 3-6 月 | 退回到刚性臂 demo |
| 4 | 云 $10-30K | 6-12 月 | 聚焦 octopus niche |

每一关都是 falsifiable gate，避免在错方向上烧超过 $5K 之前就得到 kill 信号。

---

## 10. Comparison with Concurrent Work

### 10.1 vs Kimodo：Sequence-First vs Pair-First

| 维度 | Kimodo | MotionFormer |
|---|---|---|
| 任务 | text-to-motion generation | whole-body control |
| Token 粒度 | 整帧 | `(时刻, 关节)` |
| Attention | 只在时间轴 | 双轴 + pair |
| 关节间关系 | 隐式（per-frame MLP） | 显式（pair tensor） |
| 训练数据 | 专业 MoCap (Bones Rigplay) | 三层金字塔（MoCap + 视频） |
| 主要优势 | 数据规模 + 工程成熟度 | 结构归纳偏置 |

**关系**：Kimodo 是我们的 warm-start 来源和工程参考，不是竞争对手。我们把 Kimodo 当"最好的 sequence baseline"，MotionFormer 的价值是在这个 baseline 之上证明 pair-first 能额外带来 gain。

### 10.2 vs Skild / RMA：训练范式 vs 架构归纳偏置

Skild 的 "锯腿自适应" demo 本质是**训练范式**的成就：
- 大规模 domain randomization
- Teacher-student 蒸馏
- 隐式 system identification via 观测历史

**backbone 通常是标准 MLP 或 Transformer，不是 Evoformer**。

MotionFormer 押的是**架构归纳偏置**：pair representation 是结构先验，不需要靠 domain randomization 覆盖所有形态扰动。

**两者正交且可叠加**。具体：

- 同形态故障自适应（四腿→三腿同拓扑）：Skild 式 DR 就够了
- 跨形态 zero-shot（四腿→章鱼 40 DoF）：DR 覆盖不过来，需要 pair tensor 的结构不变性

**对外定位**：
> Skild 解决的是 "single morphology under perturbation" 的鲁棒性。我们解决的是 "dramatically different morphologies with shared motor primitives" 的迁移性。前者是训练范式问题，后者是架构归纳偏置问题，两者正交且可组合。

### 10.3 vs π0 / Physical Intelligence：训练顺序倒装

π0 / π0.5 的训练范式：
```
VLM (frozen, internet pretraining) → action head (from scratch on paired data)
```

MotionFormer：
```
Action expert (MMM pretraining on motion) → brain (from scratch on top)
```

**关键差别**：

- π0 的 VLM 要把语义直接映射到 joint commands —— 是让 VLM 做它不擅长的事
- MotionFormer 的 brain 只需把语义映射到 motor primitive（低维、结构化）—— 回归 VLM 舒适区
- 数据端 MotionFormer 在 action 预训练阶段可用非配对数据，规模潜力更大

**这不是宣战**，是 **respectable disagreement**——他们押 "VLM semantic prior 是瓶颈的上限"，我们押 "motor primitive prior 是瓶颈的上限"。两种假设可以并存，最终由 downstream 任务性能定胜负。

### 10.4 vs HEX：中国全身 VLA 的先行者

HEX（北京人形创新中心，2604.07993）是**第一个明确定位为 whole-body VLA 的工作**，在 Tienkung 2.0/3.0 真机验证。其核心技术：

- **Humanoid-aligned universal state representation**: 跨异构实体的状态标准化
- **MoE unified proprioceptive predictor**: 按身体部位分组的专家混合
- **残差门控融合 + flow matching action head**

**与 MotionFormer 的相似点**：都要处理跨实体、都重视 proprioceptive 信号。

**关键差别**：

- HEX 的 MoE 按**身体部位**分组（手臂 / 腿 / 躯干），仍是 sequence-first 思路，只是分段处理
- MotionFormer 按**关节-关节对偶**构建 pair tensor，任意两关节的关系都在 backbone 里显式建模

HEX 是比我们更成熟的工程系统（已经真机 demo），但其架构归纳偏置仍有提升空间。MotionFormer 的价值是在 HEX 已验证的 "whole-body VLA 可行性" 基础上，提出更强的 backbone。

### 10.5 vs SONIC：数据规模 vs 结构先验

SONIC (NVIDIA, 2511.07820) 把 motion tracking scale 到 42M 参数 + 700h 数据，证明了"scale 也能在 humanoid 上出效果"。

**与 MotionFormer 的关系**：SONIC 的 motion tracking latent 可作为 MotionFormer 的预训练目标之一（Tier 1 数据），但 SONIC 本身是 sequence-first，没有 pair tensor。

**假设对比**：

- SONIC: "更多数据 + 更大模型 = 更好的 universal controller"
- MotionFormer: "正确的归纳偏置 + 中等数据 > 错误偏置 + 大数据"

如果 MotionFormer 在 1/10 SONIC 数据上达到相当性能，是 pair-first 的强证据。

---

## 11. Discussion

### 11.1 神经科学 vs 工程 Abstraction Level

本研究多处引用神经科学（motor synergies、body schema、小脑/皮层分工）作为动机。这是有意的——不是为了"听起来更聪明"，而是因为：

1. **神经科学给了可测量的 ground truth**：motor synergies 在脊髓层面已被电生理证实有数十维 basis。我们的 style manifold 是人造的但**目标就是 approximate 生物学已知的那个 manifold**。
2. **进化选择过的解未必是最优但通常是 robust 的**：哺乳动物的 cerebellum / cortex 分工是亿年选择的结果，直接 copy 这个架构比自己拍脑袋设计风险小。
3. **架构决策有神经科学类比时更容易和评审 / 合作者沟通**。

但要警惕过度类比——神经科学给的是 inspiration，不是 constraint。我们不要求 MotionFormer 的每一部分都严格对应某个脑区，只要 high-level 架构原则一致即可。

### 11.2 Octopus Doctor 作为 Defensible Niche

章鱼博士 40 DoF 的选择不是为了炫技，是因为它是**现有所有主流架构最难处理的场景**：

- **标准 Transformer**：attention cost `O((T × J)²)` 在高 J 下爆炸
- **MoE by body part**：没有 pre-defined "body part" 概念（触手没有人类的手臂/腿的对应）
- **SMPL retargeting**：SMPL 是人形骨架，章鱼拓扑完全不同
- **Skild 式 DR**：四足狗的 DR 覆盖不到章鱼触手的动力学

而 **MotionFormer 的 pair tensor 是 morphology-agnostic 的**：40 个节点也好、23 个节点也好、甚至异构拓扑也好，只要节点特征里编码好每个关节的类型和 kinematic 位置，backbone 架构不变。

这是**商业和学术上都 defensible 的 niche**——没人愿意做，你做了就是唯一。

### 11.3 Risks

**最大风险：Stage 1 fail**

如果在合成 synergy 数据上 MotionFormer 不比 baseline 强，整个架构 thesis 证伪。这个风险是**真实存在的**，历史上很多"看起来很对"的归纳偏置在实证上并没有 gain（比如早期 Graph Neural Network 对分子性质预测的提升有限于特定 benchmark）。

**缓解**：Stage 1 只需 2-4 周 + 一张卡，成本 < $500 等效。这是最便宜的 kill switch。

**次大风险：Warm-start 负迁移**

Kimodo 的 self-attention 语义与 MotionFormer 的 row attention 不同，warm-start 可能反而是负迁移。

**缓解**：对照实验，如果负迁移退回 from-scratch。

**第三风险：工程复杂度**

Evoformer-style 架构工程实现比标准 Transformer 复杂，triangle attention 的 indexing 容易 bug。且 pair tensor `O(J²)` 内存在 long sequence 上可能溢出。

**缓解**：参考 AlphaFold 开源实现（OpenFold、AlphaFold 2 原版），不从零写。

### 11.4 Open Questions

写作时仍未完全确定的设计问题：

**（Q1）Action Expert API 形式**

三个候选：
- (A) Continuous style latent + goal: `z_style ∈ R^64, goal ∈ SE(3)`
- (B) Discrete motor primitive tokens: `{拾取, 推, 拉, 旋, ...}` + 参数
- (C) 混合：codebook（VQ-VAE 风格）+ 连续微调向量

初步倾向是 (A)，但 (C) 在工程上可能更稳（容易对接 VLM tokenizer）。需要第一阶段实验结果辅助决定。

**（Q2）Style Manifold 的可解释性**

目标是"拿杯子近开门、远上楼梯"，但实际学出来的 manifold 可能聚类方式完全不同（比如按"接触/不接触"分，而不是按"手部/全身"）。需要训完后做 t-SNE / UMAP 可视化验证。

**（Q3）章鱼博士 Sim-to-Real**

40 DoF 软体结构的 sim-to-real gap 比人形机器人更大（非线性动力学难仿真）。可能需要 iterative sim-to-real 或 residual learning on real data。

**（Q4）Brain 的输入选择**

`DINO + SigLIP + Language + Proprio` 全都要，但具体融合方式（early fusion / late fusion / cross-attention / concat）需要实验。

---

## 12. Conclusion

### 12.1 一句话 Thesis

> **全身机器人控制本质上是一个 pairwise 协同问题，最好的归纳偏置是把 `[joints × joints]` 的 pair tensor 提升为 backbone 的一等公民（AlphaFold 的核心洞察），并用 brain / cerebellum 倒装训练架构（action expert 先训冻结，brain 后训通过 cross-attention 对话）充分利用非配对动作数据。**

### 12.2 三个核心 Technical Novelty

1. **Pair-first 架构**：把 AlphaFold Evoformer 的 pair representation + axial/triangle attention 首次系统移植到机器人全身控制
2. **倒装训练范式**：相对 π0 的 VLM-frozen-action-trained，我们 action-pretrained-frozen-brain-trained
3. **Cross-attention 接口**：brain 和 cerebellum 之间用 continuous cross-attention 对话，不是离散 token 词表

### 12.3 下一步具体行动

**本周**：
- 下 Kimodo checkpoint，跑 viser editor，熟悉 SOMA/G1 骨架格式
- 在本机 RTX Pro 6000 Blackwell 上配好 PyTorch + FlashAttention 环境

**下周**：
- 构造合成 motor synergy 数据集（K=8, N=10k 样本）
- 实现 MotionFormer 基础 block（row/col attention + pair tensor + triangle update）
- 开始 Stage 1 实验

**下下周**：
- 对比 Stage 1 实验结果
- 如果 pass，启动 Route B 完整实现；如果 fail，重新审视 hypothesis

---

## Appendix A: 论文阅读 Syllabus

对应 agent.md 里的 syllabus，加上本次讨论后每篇的 key takeaway：

### A.1 Foundation Models for Robotics

| 论文 | arxiv ID | Key Takeaway |
|---|---|---|
| π0 | 2410.24164 | Flow Matching VLA 架构范式；VLM 冻结 + action head 从零训练 |
| π0.5 | 2504.16054 | 开放世界泛化，强调 internet-scale data |
| SONIC | 2511.07820 | 700h motion tracking，证明 scale 在 humanoid 上 work |
| BeyondMimic | 2508.08241 | Guided diffusion 零样本泛化 |
| HEX | 2604.07993 | 第一个 whole-body VLA，MoE proprioceptive predictor |
| UniAct | 2501.10105 | Universal action space 跨机器人共享原子行为 |

### A.2 架构基础

| 论文 | arxiv ID | Key Takeaway |
|---|---|---|
| AlphaFold 2 | Nature 2021 | Evoformer 架构灵感源头；pair representation 一等公民化 |
| ACT | 2304.13705 | Action chunking + cVAE 实现；KL 权重调优 |
| Diffusion Policy | 2303.04137 | Chunk size 选择；连续动作空间建模 |
| Flow Matching | 2210.02747 | Flow matching vs diffusion 原理；最优传输路径 |

### A.3 全身控制专项

| 论文 | arxiv ID | Key Takeaway |
|---|---|---|
| HumanPlus | 2406.10454 | 全身 26+ DoF 控制；shadow + imitation stack |
| OmniH2O | 2406.08858 | 运动学 pose 作 universal 接口；多种遥操作输入 |
| HOVER | 2410.21229 | 多模式策略蒸馏；navigation + loco-manipulation + tabletop 统一 |

### A.4 Motion Diffusion（新增，本次讨论产出）

| 论文 / 项目 | 来源 | Key Takeaway |
|---|---|---|
| Kimodo | NVIDIA 2026-03 | Kinematic camp SOTA；700h Bones Rigplay；sequence-first Transformer；两阶段 denoiser（root + body）；LLM2Vec 文本编码 |
| MaskedMimic | NVIDIA SIGGRAPH 2024 | Physics camp 代表；Masked motion modeling 原理；viser 编辑器的 inpainting 机制 |
| ProtoMotions | NVIDIA | 物理角色控制框架；接收 Kimodo 输出训 policy |

### A.5 Adaptive Control（新增）

| 论文 / 项目 | 来源 | Key Takeaway |
|---|---|---|
| RMA | Kumar, Fu, Pathak, Malik 2021 | Teacher-student + domain randomization 原始论文；student 隐式 system ID |
| Skild AI | Deepak Pathak 2026 | RMA 产品化；"one brain every robot"；$1.4B Series C |
| Figure Helix | Figure AI 2024-2026 | Dual-system VLA；disable-joint adaptation via DR |

### A.6 Vision Encoder（新增）

| 模型 | Key Takeaway |
|---|---|
| SigLIP | Text-image 对比，semantic 强 geometric 弱 |
| DINOv2 | Self-supervised multi-crop，scale-invariant geometric features |
| PaliGemma | SigLIP + LLM decoder；segmentation 靠 task-specific supervision emerge |

### A.7 Video-to-SMPL（新增）

| 方法 | 适用 | 精度 |
|---|---|---|
| 4D-Humans / HMR 2.0 | 第三人称单人 | per-frame 高质量，时序需后处理 |
| WHAM | 第三人称，时序 | 时序一致性强 |
| GVHMR | 野外 global trajectory | 输出 world-frame motion |
| TRAM | 长视频 SLAM 融合 | 相机+人物都在动 |
| EgoLocate / EgoBody | 第一人称 POV | 专攻 ego-centric |
| NLF | 2025 SOTA 单人 | 精度超 4D-Humans |

---

## Appendix B: Cloud GPU 价格速查

价格为 2026-04 on-demand，per GPU per hour（USD）。长期合约可再打 30-40%。

| Provider | H100 80GB | H200 | A100 80GB | 备注 |
|---|---|---|---|---|
| Nebius | $2.95 | $3.50 | ~$2 | 性价比王，欧洲节点 |
| Lambda Labs | $2.49-$3.44 | N/A | ~$1.5-2 | 界面友好，易缺货 |
| RunPod Secure Cloud | $2.69 | $3.59 | ~$1.5 | Community Cloud $1.99（无 SLA）|
| CoreWeave | $4.76-$6.15 | N/A 标价 | $2.21 | 企业 SLA 贵但稳 |
| NVIDIA Brev | ~$2.5-3 | - | - | Marketplace，底层是其他 provider |
| AWS / GCP / Azure | $8-12 | - | - | 不要用 |

**推荐选择**：MVP 阶段 RunPod Community Cloud（$1.99/hr，可承受偶尔 preempt），Production 训练 Nebius reserved（~$2/hr，有 SLA）。

---

## Appendix C: Tooling

### C.1 Kimodo CLI（本地已部署在 `rot/kimodo/`）

```bash
# 激活环境
conda activate rot

# CLI 生成
kimodo_gen "A person walks forward while waving their arms" \
    --model Kimodo-SOMA-RP-v1 \
    --duration 5

# 交互式 Viser 编辑器
kimodo_demo
# 浏览器访问 http://127.0.0.1:7860

# Text embedding server（用于多 prompt 批量生成）
kimodo_textencoder
```

**VRAM 需求**：~17GB（主要是 LLM2Vec text encoder）。RTX Pro 6000 96GB 绰绰有余。

**Checkpoint 下载**：首次运行会自动从 HuggingFace 下（nvidia/Kimodo-SOMA-RP-v1 等），国内可能需要镜像。

### C.2 pdf2zh 翻译流程修补（2026-04-20 踩坑总结）

见 `~/.claude/projects/-home-arenalabs-Desktop--be-water--robot-/memory/feedback_pdf2zh_pipeline.md`。关键四个 patch：

1. `np.fromstring` → `np.frombuffer`（numpy 兼容）
2. `lang_out == "zh"` 要归一化为 `"zh-CN"`（Google Translate 参数）
3. `china-ss` 字体不嵌入 → 用 `pdftocairo -pdf` 扁平化强制嵌入
4. 加术语 + 作者名后处理层（约 90 条规则）

### C.3 ROT 系统

`rot/` 目录是完整的机器人操作终端：
- `rot_core/` 后端
- `rot---robotics-operating-terminal/` 前端 UI
- `docker/` 容器化部署
- `smpl_data_2_eric/` SMPL 数据
- `TripoSR/` 3D 生成

启动：

```bash
./start.sh               # 本地 Editor (port 8012) + REST API (port 8080)
./start.sh --docker      # Docker 全栈
```

### C.4 推荐开发环境

```
Python 3.11+
PyTorch 2.4+ (with Flash Attention 3)
CUDA 12.4+
Isaac Gym / Isaac Lab (MuJoCo MJX 可选替代)
PyRoki (SMPL retargeting)
pymupdf + pdftocairo (PDF 处理)
Kimodo (motion generation)
conda env: rot (已存在)
```

---

## 结语

本文档是一份**活的研究愿景**，不是最终论文。随着 Stage 1-4 实验推进，各章节（特别是 §7 warm-start、§9 roadmap、§11 open questions）应持续更新。建议每完成一个 Stage 后增加一节 "Stage X Retrospective"，记录实际发现与原计划的偏差。

最终的 publishable 论文将从本文档提炼（保留 §1 introduction 的动机、§3 method 的架构、§9 roadmap 中 go-status 的实验结果），但不会直接 copy-paste——那是商业文档而非学术论文的风格。

> "Be water, my friend."
> — Bruce Lee

做研究也该如此：保持核心假设的清晰（thesis 要像水一样凝聚），但对具体实现的形态保持流动（架构细节要能 adapt 到实验反馈）。这份 vision 里每一个数字、每一个架构决策、每一个 go/no-go 门槛，都是对下一次实验结果的假设，实验会告诉我们哪些该保留哪些该放下。
