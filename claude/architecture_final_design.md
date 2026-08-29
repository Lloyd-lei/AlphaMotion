# AlphaMotion 架构设计：跨具身动作生成模型（研究员视角完整设计）

本文档是本会话的终稿整合，取代`architecture_spec.md`中与SAMoR讨论冲突的部分（原文档的"整段T帧压缩"假设已被替换为"单帧spatial + 因果时间层"），并入`math_framework.md`未变的部分（多模态条件注入、约束掩码分布、Riemannian flow matching）。结构：每一节先给"大白话"，再给"Appendix式"形式化描述。

---

# Part 0. 一页纸总览（大白话）

把整个系统想成两个员工分工：

- **员工A："万能翻译官"（Spatial Layer）**——不管来的是人、G1还是Optimus，只要给它一帧姿态，它就能翻译成任意其他构型的对应姿态。它不记事，每一帧都是独立处理的，翻译时会自动"脑补"没被指定的关节。这个员工的核心能力就是本会话反复拷问的"跨拓扑retargeting"，参考了SAMoR的做法但做了改造：不整段压缩、不做量化，保留SE3全精度。
- **员工B："动作导演"（Temporal Layer）**——员工B不直接管骨骼细节，只在"翻译官"消化过的抽象语义空间里工作，负责让动作在时间上连贯、听得懂文字指令、能满足"某几个关节在某段时间必须怎么动"这种约束，并且要能像放录像一样一帧一帧实时吐出来，不能卡顿。

两个员工可以单独工作：只用员工A，就是一个能实时跑的、跨机器人的"姿态翻译器/遥操作接口"；两个一起工作，就是完整的动作生成大模型。

---

# Part 1. Spatial Layer（空间层／"翻译官"）

## 1.1 大白话

给员工A一帧动作（可能是完整的、也可能只给了手腕/脚/头这几个关键点），它要做三件事：
1. 把这一帧"翻译"成一个和具体机器人/人体构型无关的中间语言（8张左右的"摘要卡"，参考SAMoR，但**不压成整数编号**，就是普通的连续向量，保留全部精度）；
2. 中间没被指定的关节，靠这些摘要卡自动脑补出合理的位置；
3. 把摘要卡"翻译"回任意目标构型（比如G1的23个关节，或者Optimus的45个关节）的具体SE3姿态,并且如果目标关节被用户强行指定了坐标，必须原样遵守。

为了不产生"左右分不清"的镜像错误，翻译回具体坐标这一步，借鉴AlphaFold的做法：每个关节维护一个"带朝向的坐标系"（frame），而不是只算一个点的坐标，这样天然不会左右搞反。

为了让不同机器人真正共用同一套"中间语言"（而不是每个机器人各自躲在自己的角落里，摘要卡形同虚设），额外加一个"裁判"：训练时找一个小模型专门猜"这张摘要卡是从哪个机器人来的"，然后故意让翻译官学会"骗过裁判"——逼着它把机器人身份的痕迹从摘要卡里洗干净，只留下动作本身的意思。

## 1.2 Appendix：记号与模块规格

**输入**：单帧姿态 $\{(\boldsymbol p_j, \boldsymbol v_j, \boldsymbol r_j, \ell_j)\}_{j=1}^{J}$，其中 $\boldsymbol p_j\in\mathbb R^3$位置、$\boldsymbol v_j=\boldsymbol p_j-\boldsymbol p_j^{(t-1)}$速度（最小时间窗=2帧,非严格单帧）、$\boldsymbol r_j\in\mathbb R^6$（Zhou et al. 6D旋转表示,仅末端/次末端关节）、$\ell_j$为该关节的CLIP文本embedding（关节名）+ kinematic-tree位置编码。$J$随具身变化（G1:23, Optimus:45, SMPL-X:22, 任意新构型:任意）。

**Encoder**（几何双轨 + functional-group pooling）：
```
h_j = MLP([p_j, v_j, r_j]) + CLIP(name_j) · dropout(p) + PosEnc_tree(j)   # (J, D)
h = SelfAttention(h, bias = hop_dist(j,j') + same_chain(j,j') + edge_type(j,j'))  # 单体轨+配对轨融合
                                                                                    # 逐层堆叠 L 次
Q_part = learnable (K, D)          # K≈8，可学习functional-group query
A = softmax(Q_part @ h^T / sqrt(D))               # (K, J)
z = Q_part + A @ h                                # (K, D)  —— 连续瓶颈，不量化
z = SelfAttention_KxK(z)
```
不同于SAMoR：无`for t in range(T//4)`时间下采样循环（此处严格单帧/双帧输入）；无RVQ/FSQ量化步骤（第1.4节说明理由）。

**Decoder**（frame-based IPA, target-topology-as-query）：
```
T_j^{(0)} = T_pose_frame(target_embodiment, j)      # 初始化，j over target joint set J_target
Q_j = anchor_embed(T_j^{(0)}) + CLIP(name_j^target) + PosEnc_tree(j)
for recycle in range(N_recycle):                    # N_recycle 待消融，建议起点 2-3
    for layer in range(L_dec):
        local_coords = InvariantPointAttention(Q, T, context=z)  # IPA核心：用T_j把z变换到局部系
        (ΔR_j, Δt_j) = MLP(local_coords)
        T_j = compose(T_j, (ΔR_j, Δt_j))
    if soft_prior_enabled:                            # 见1.3
        T_j = gate(λ) · T_j^{(t-1)} + (1-gate(λ)) · T_j
    T_j[is_controlled] = constraint_value             # 硬钉死，RePaint式替换
p_j, r_j = T_j.translation, T_j.rotation
```

**解析可行性层**：`p_j, r_j -> Jacobian/geometric reprojection -> feasible joint command`；超出可行域时抛`GeometryUnreachableError(residual, joint_id)`而非静默裁剪；需验证该层可微性，若不可微则显式声明"三项物理loss只监督到此层输出"。

## 1.3 Soft prior（warm-start，非核心但已敲定）
```
T_j^{decode_input} += λ · Embed(T_j^{(t-1)})    # λ 可学习标量或attention自适应
# 时间层关闭时 T_j^{(t-1)} 不存在，该项跳过 —— 遥操作模式不受影响
```

## 1.4 为什么Spatial层不做任何离散量化

- Spatial层的目标函数是**重建精度**（SE3坐标回归），量化（RVQ/FSQ）的动机是"造一个可复用/可被GPT预测的离散词表"，两个目标不同,不应该为了后者牺牲前者。
- FSQ的核心假设是逐维独立舍入，对SO(3)这种非欧流形表示（如6D旋转的正交性约束）会引入舍入后越界、需靠Gram-Schmidt重新投影的系统性失真，这个失真与量化格点粗细无关，是结构性的。
- 结论：**保持连续瓶颈**；若未来确有必要离散化（如复用GPT式backbone），采用MotionVLA式"低频离散+高频残差连续"的分层方案，而非对SE3坐标直接FSQ。

## 1.5 训练目标（四项损失，替代量化式的隐式正则化）

| Loss | 覆盖范围 | 作用 |
|---|---|---|
| $\mathcal L_\text{direct}$ | 有配对数据（SOMA→G1） | 关节位置L2 + 测地距离旋转loss，直接监督 |
| $\mathcal L_\text{swap}$ | 无配对数据（Optimus/Figure/任意新构型） | source $z$ → decode到B → 重新encode → 要求恢复语义与原始一致（cycle-consistency，无监督） |
| $\mathcal L_\text{phys}$ | 两侧 | 手部接触 + 足部防滑（$\Delta\text{foot}\to0$）+ 骨长一致性 + 质心支撑面平滑 |
| $\mathcal L_\text{adv}$ | 两侧（**本次会话新增**） | 判别器 $D_\text{emb}(z)\to\hat e$（猜测具身身份），梯度反转让encoder最小化其可猜测性——防止$z$退化为"各具身各占一块子空间"的偷懒解，逼真正共享synergy结构 |

$\mathcal L = \mathcal L_\text{direct} + \mathcal L_\text{swap} + \alpha\,\mathcal L_\text{phys} + \beta\,\mathcal L_\text{adv}$（$\alpha,\beta$待消融）。

---

# Part 2. Temporal Layer（时间层／"导演"）

## 2.1 大白话

导演不碰任何具体骨骼，只在翻译官吐出的"摘要卡序列"上工作。它是一个只能往前看（causal）、按滑动窗口处理的模型，输入prompt、输入约束（"这几帧、这个部位必须怎样"），预测下一时刻的摘要卡该是什么样。生成方式采用flow matching（从噪声逐渐"流"向目标），但标准flow matching生成一次要走很多步，太慢——所以要做两件事把它变快：(1) 用一种叫MeanFlow的蒸馏技术，把"走很多步"压缩成一两步；(2) 用一种叫Real-Time Chunking的技巧，让"生成下一段"和"执行当前段"同时进行、无缝衔接，而不是生成完一段再等着生成下一段。

## 2.2 Appendix：形式化描述

**自回归对象**：连续 $z_t\in\mathbb R^{K\times D}$（Spatial Encoder输出，见Part 1），非decode后的关节值（已在architecture_spec.md第三节论证：避免每构型单训一个时间模型/避免每步重复encode）。

**约束掩码统一形式**（沿用math_framework.md第2节）：mask函数 $m(j,t)\in\{0,1\}$，联合采样分布覆盖Kimodo型（temporal-sparse, spatial-complete）、OmniControl型（spatial-sparse, temporal-dense）、以及用户目标模式（spatial-sparse且temporal-continuous占主导，建议采样权重 $\alpha_3\gtrsim0.6$）。

**训练目标**：Flow matching，速度场网络 $v_\theta(z_t^\tau,\tau \mid z_{<t}, c_\text{text}, c_\text{constraint}, c_\text{scene})$，$\tau$为流内部时间（0→1，噪声→目标），$t$为序列时间步：
$$\mathcal L_\text{FM} = \mathbb E_{\tau,z_0,z_1}\left[\|v_\theta(z^\tau,\tau\mid\cdot) - (z_1-z_0)\|^2\right],\quad z^\tau=(1-\tau)z_0+\tau z_1$$
旋转通道（若时间层也需要处理旋转分量）替换为Riemannian flow matching：测地插值 $r^\tau = r_0\cdot\exp[\tau\cdot\log(r_0^\top r_1)]$。

**三层实时化叠加**（三者正交，缺一不算真实时）：
1. **因果分块自回归**：causal attention + KV cache，chunk间维持长期记忆，避免"失忆"。
2. **MeanFlow蒸馏**：学习"平均速度" $u_\theta(z^\tau,\tau,\tau')$ 而非瞬时速度，用解析恒等式训练，单步生成质量可打平多步模型——替代更早的consistency distillation路线。
3. **Real-Time Chunking (RTC)**：相邻chunk重叠区域用inpainting式方式从"正在执行的旧chunk"平滑过渡,使"生成下一chunk"与"执行当前chunk"并行,消除计算延迟造成的执行空档。

**多模态条件融合**（沿用math_framework.md第3节）：三条独立cross-attention分支（约束token / 文本prompt / 环境感知：SLAM点云或video编码），各自独立classifier-free-guidance式dropout。

---

# Part 3. 产品功能 → 架构模块对照表

| 产品功能（用户视角） | 对应架构模块 | 关键机制 |
|---|---|---|
| 跨构型即时转换 | Spatial encoder-decoder | functional-group pooling + target-topology-as-query decode |
| 稀疏输入"脑补"全身 | Spatial encoder的单体轨attention + mask机制 | 未指定关节靠双向attention从已知末端约束扩散信息 |
| 关键帧编辑/局部精修 | Decoder的Is_Controlled钉死机制 | RePaint式硬替换,不影响其余关节的联动生成 |
| 可行性反馈（非静默失败） | 解析层 + GeometryUnreachableError | 显式区分"给定可行输入必可行" vs "超出可行域即报错",两种保证不可混淆 |
| 实时遥操作模式 | 单独运行Spatial层（时间层关闭） | soft prior置零退化为纯逐帧独立求解;延迟预算按遥操而非离线生成场景设计 |
| 素材通用性（一份数据服务多构型） | Spatial层的embodiment-agnostic瓶颈$z$ | 对抗判别器loss确保$z$不携带具身身份信息 |
| 置信度/质量分数 | 解析层残差 + recycling收敛曲线（**待设计，未消融**） | 可行方向：用recycling迭代间的frame变化幅度或解析层残差大小做代理指标,尚未验证 |

---

# Part 4. 仍待消融、非架构分歧的开放数字

1. Recycling轮数上限 + 精度验收阈值（如末端<1cm、<2轮 vs <0.5cm、<3轮）。
2. 遥操作单帧延迟预算（毫秒），反推 pairwise track 最大 $J$、encoder/decoder深度。
3. Sliding window步长、causal attention窗口大小。
4. Part token数量 $K$（SAMoR消融显示中间粒度K=8优于K=1和K=J，但该结论来自动画重建任务，未验证是否直接适用于SE3控制精度场景——需要自己的消融）。
5. $\mathcal L_\text{phys}$、$\mathcal L_\text{adv}$ 的权重 $\alpha,\beta$。
6. 约束掩码采样比例 $\alpha_1:\alpha_2:\alpha_3$（初始建议6:2:2，待训练分布验证）。
7. 置信度/质量分数的具体定义（第3节最后一行，产品需求已明确但技术方案未设计）。

这七项的共同特点：均不改变Part 1/2已敲定的模块结构,只影响可由实验曲线直接读出的层数/维度/权重/阈值,因此推迟到有可跑pipeline之后处理,是合理顺序。
