# 文件2：Round-1 训练规格（Codex据此写代码）

**这份文件是实现依据。** 目标：在单帧跨拓扑retargeting任务上，确定7种瓶颈中哪个最适合做"翻译官"。

**开工前先读 `00_codex_prep_checklist.md`**，那里的 `[BLOCK]` 项没完成时本文档的实验跑不出有效结论。

---

## ⚠️ 不可违反的scope约束（读这份文件的第一件事）

**一次unified universal预训练，泛化到所有拓扑构型。零per-embodiment的encoder / decoder / 输出头 / 微调。**

具体禁止：
- ❌ 任何形如 `head[embodiment]`、`decoder[robot_type]`、`if embodiment == 'g1'` 的分支
- ❌ 任何per-embodiment的可学习参数（包括可学习的具身embedding查找表——它对未见具身无法查表）
- ❌ 任何"接入新机器人需要微调/重训"的设计
- ❌ 任何把关节角作为**网络输出空间**的设计（关节角跨拓扑不可比，必然导致per-embodiment头）

**允许**（这些不违反scope，容易搞混所以列清楚）：
- ✅ **按数据可用性mask loss项**——动物没有旋转GT就把该项权重置零。网络权重逐字节相同，只是不同样本激活不同监督项。**mask一个loss ≠ 加一个头。**
- ✅ **解析/几何算子**——只要它是由目标骨架规格（URDF/BVH层级）**确定性地**决定的、零学习参数，就不算per-embodiment模块。换机器人只换它的规格文件。
- ✅ **把目标骨架规格作为条件输入**——这正是universal的实现方式（SAMoR的pose anchor + 关节名 + 图bias就是这么做的）。

**每写一个模块，问一遍这个判据**：给一个训练时从未见过的机器人，只提供它的URDF，能否直接输出可用结果、不新增参数、不训练任何东西？答不出"能"就是跑偏了。

---

## 0.1 对Codex反证审查的九个决策答复（v6修订，2026-07-30）

Codex的 `codex/round1_preflight_review.md` 提出7条P0：**5条接受、1条部分接受、1条驳回**。接受的部分已在本文档落地。

> **v7（2026-07-30 第二轮）**：Codex对v6的§1.2提出新P0，**两条都对，我实测确认后已作废v6的标量$\varphi$方案**，改为 rest-relative delta rotation $\Delta R_j=R_j(t)R_j(\text{rest})^\top$。详见§1.2.3。另修正v6遗留的4处文本冲突（atan2公式/伪代码未跟上、§5.1编码侧约束、§4.3 root独立通道、T-c退化风险）。

| # | 问题 | 决定 |
|---|---|---|
| 1 | 训练task matrix / 方向比例 / GT来源 | **见§3.0新增表**。Truebones**只做自重建与cycle正则，不作跨拓扑翻译GT** |
| 2 | canonical frame如何对任意URDF/BVH做gauge fixing | **见§1.2.3（v7重写）**：rest-relative delta rotation $\Delta R_j=R_j R_j^{\text{rest}\top}$，实测gauge不变(3.2e-6°)且跨rig可比(3.0e-6°) |
| 3 | decoder输出29个URDF joint frames还是约14个semantic frames | **按目标骨架的关节逐个输出frame**（G1即29个revolute的child link）。语义对应只用于**loss mask**，不改变输出粒度 |
| 4 | global root是旁路、constraint、还是本轮不预测 | **本轮不预测**，见§2.3 |
| 5 | SE(3) constraint是否改纯target-side、位置/旋转分离 | **是**，见§5.0 |
| 6 | 是否把input-conditioned pair refinement固定进共享backbone | **不固定**，理由见§0.2 |
| 7 | `val_topo`对无paired GT的动物测什么 | **只测自重建+cycle-consistency，且明确不称"翻译"**，见§6.2b |
| 8 | matched bitrate还是quality-rate Pareto | **走quality-rate曲线**（v7修订）：每个量化变体扫 {12,16,20,24} bits/token 四点，与率失真理论下界叠图。v6定的"对齐到12 bits"已被实测推翻——12 bits的理论最优末端误差就有14.2mm。见§3.3 |
| 9 | claim是否收窄为kinematic translator | **是，明确收窄**，见§0.3 |

## 0.2 关于"是否沿用旧backbone"（P0 #7，驳回）

Codex要求先固定旧实验的 input-conditioned pair update / OPM backbone，依据是"OPM是OOD-key、triangle决定表征几何"这一"硬证据"。**我核实了原始记录：这批结论以现有证据不成立——不是错，是未经验证。**

实测三seed聚合（`experiment/stage1/runs/{full,opm_only}[__seed{1,2}]/history.json`）：

| 指标 | full (mean±sd) | opm_only (mean±sd) | gap / pooled_sd |
|---|---|---|---|
| val total | 0.9433 ± **0.2223** | 1.2281 ± **0.5581** | 0.67 |
| val pos | 0.2099 ± 0.0528 | **0.1952** ± 0.0449 | **−0.30**（方向相反） |
| val rot | 0.1012 ± 0.0197 | 0.1602 ± 0.0944 | 0.87 |

- `full`的total三seed为 0.689 / 1.038 / 1.103，**60%摆幅**；小于此幅度的差异在n=3下不可分辨。
- **pos上 opm_only 反而更好**，与README结论方向相反。
- 所有 gap/pooled_sd < 1，**无一项统计可分辨**。
- README关键表涉及的 `triangle_only/baseline/pair_static/axial_only` **均无 history.json**（只有`full`/`opm_only`有3 seed）——那张表极可能是单seed。
- Pearson r=0.077 是单次比较、无置换检验、无seed重复，不足以支撑"triangle唯一地选择模块化几何"这类机制性断言。

**Lloyd已明确表态：那批实验是他自己做的、做得不严谨，结论有问题，不作为本轮硬约束。**

**但方法论顾虑成立一半**：backbone与bottleneck同时变会让归因困难。解法不是回到未验证的backbone，而是**backbone在七变体间逐字节冻结**（§6.3b已如此要求）。pair/OPM/triangle值不值得加，是**需要自己多seed实验**的独立消融，列入Round-2。

## 0.25 训练前就能算出来的东西（v7新增）

编解码本质上是压缩，而压缩有**不依赖训练的理论**。以下每一条都是在**没有跑任何训练**的情况下算出来的，都改变了设计。放在这里是为了提醒：**先算能算的，再去炼丹。**

| 已算出的 | 结果 | 改变了什么 |
|---|---|---|
| **1. 位置零空间维数**（Jacobian秩） | 14个语义站点的位置（42维输出）对29个DOF的Jacobian，**数值秩恒为21**，**零空间8维** | 证实"只输出位置"不可行，且**比先前逐DOF扰动测出的4个更严重**。ΔR方案的必要性从"4个twist DOF"升级为"8维不可观测子空间" |
| **2. 码率下界**（率失真反向注水） | 12 bits/token的**理论最优**末端误差就有14.2 mm；要到3 mm需20 bits | **直接推翻了v6定的12 bits**，改为扫{12,16,20,24}四点 |
| **3. 单码本VQ的码率天花板** | 20 bits需 $2^{20}\approx10^6$ 码字，训练不现实 | 给出一个**可证伪的预测**：若目标<5mm，单码本VQVAE出局 |
| **4. 姿态本征维数**（PCA） | 90%方差需14维、95%需18维、99%需25维（共29维） | 说明 $K\times D=8\times256$ 的连续瓶颈**远超**信息需求——AE/VAE的瓶颈不是容量瓶颈，它们之间的差异会来自正则化而非容量 |
| **5. gauge不变性**（代数+2000次随机检验） | $\Delta R$ 偏差3.2e-6°；标量$\varphi$偏差**180°** | 作废了整个v6旋转方案 |
| **6. joint axis与bone的关系**（逐关节几何） | 29个revolute里**23个**的axis不沿bone | 同上，且说明"bone twist"这个直觉对URDF不成立 |
| **7. 数据单位**（对URDF限位核验） | G1 CSV关节列是**度**（29/29列吻合），不是弧度 | 解决prep C.2；单位搞错会让所有角度指标错57倍 |

**还能继续算、但还没算的**（建议在写训练代码前做完，都不需要GPU）：

1. **SOMA→G1映射是不是一个函数**：同一个SOMA姿态（在某个容差内）是否对应唯一的G1姿态？在配对数据里找近邻SOMA帧、看对应G1帧的散布。**若散布大，说明retargeter本身有多解，那么GT噪声就是精度的天花板**，模型再好也无法超过它。这直接决定了"末端误差目标定多少才有意义"。
2. **19→16关节对应的信息损失**：SOMA的19个保留关节里只有16个在G1有对应。用同样的Jacobian秩分析算一下**这16个站点能否唯一确定G1的29维**——如果不能（很可能），那么**跨拓扑翻译在数学上就是欠定的**，缺的信息必须由先验（训练数据分布）补，这是模型的价值所在，但也意味着**存在不可消除的误差下界**。
3. **动物骨架的关节数分布与$K=8$的匹配度**：Truebones关节数范围9–71。$K=8$对9关节的物种是**过参数化**（8个query分9个关节），对71关节是**压缩17倍**。同一个$K$跨这个范围是否合理，可以先用各物种的PCA本征维数估计。
4. **约束满足的可行性**：给定一组稀疏SE3约束，目标骨架**在关节限位内是否存在解**？这是纯IK可行性问题，可以用URDF+优化器离线判定。**若某些约束组合本就无解，那它们不该出现在训练集里**，否则是在教模型拟合不可能的目标。

**这四项都是"算出来就能改设计"的类型，加起来大概一天的CPU工作量，远比先训21个模型再回头查便宜。**

## 0.3 本轮claim的范围（收窄，回答#9）

**本轮只声称一件事：一个kinematic的跨拓扑姿态翻译器。**

明确**不**声称：
- ❌ physical plausibility / robot-ready（所有指标都是运动学的：无动力学、无真机、无接触力）
- ❌ 可直接下发电机（关节限位违反率只是筛查，不等于可执行）
- ❌ "any humanoid"泛化（拓扑覆盖约70种，SAMoR约71k种）

关节限位违反率、FK一致性是**必要条件的筛查**，不是充分条件的证明。**报告措辞必须与此一致。**

---

# 0. 一句话任务定义

给定**源骨架**的一帧姿态（可能只给了部分关节），输出**任意目标骨架**（可以是完全不同的拓扑）对应的一帧姿态，并且用户指定了坐标的关节必须精确遵守。

不涉及时序。不涉及文本prompt。这两项留给后续阶段。

---

# 1. 输出表示：一个通用头输出canonical SE(3) frame，零具身特定参数

这一节先讲清楚，因为它决定了整个IO的形状。

## 1.0 硬约束：一次预训练，零per-embodiment参数

**本项目的scope是一次unified universal预训练，泛化到所有拓扑构型，不为任何单一构型微调或增加专属的encoder/decoder/输出头。** 任何"给G1加一个头"、"给人形加一个分支"的设计都直接违反这条，因为它意味着接入第N+1种构型时要改网络、要新数据、要重训——那就不是universal模型，只是一堆专用模型共享了一个编码器。

**验收这条的判据**：**给一个训练时从未见过的机器人（只提供它的URDF/骨架定义），模型必须能直接输出可用结果，不新增任何参数、不训练任何东西。** 下面每个设计决策都要过这个判据。

## 1.1 SAMoR输出什么，以及为什么我们要比它多输出一层

**先直接回答"SAMoR输出关节角度还是3D frame"**：SAMoR输出的是**per-joint 3D位置**（root-normalized），**既不是关节角、也不是SE(3) frame**。原文：所有motion"converted to a unified format of per-joint positions $p_j^t$ and velocities $v_j^t$"；需要旋转时（给mesh做LBS）靠**IK后处理**从位置反解。

**它为什么不输出关节角**——这个理由完全正确且对我们同样成立：**关节角跨拓扑不可共享**。关节角的语义依赖各自的rest pose定义和轴约定，"肘弯30度"在人体骨架和G1上是两个不同的数学对象，关节数还不同（78 vs 29）。**所以关节角绝对不能作为网络的输出空间**——那会立刻退化成per-embodiment头。

**但纯位置输出不够**，实测原因：G1的29个DOF里，**双踝roll对关节位置的影响精确为0.000 mm**，双腕roll约0.6 mm——它们在位置映射的**零空间**里，从位置**原理上不可恢复**。SAMoR自己在Limitations里承认这点："accurate twist recovery from positions alone remains an open problem"。`ankle_roll`是踝侧倾，直接决定足部接触面姿态与横向平衡，对下真机不能丢。

**关键实测：canonical frame能完整承载位置丢失的信息。** 同样的扰动测试，同时测位置位移和frame朝向变化：

| DOF | 最大位置位移 | 最大frame朝向变化 |
|---|---|---|
| `left/right_ankle_roll` | **0.000 mm** | **11.459°** |
| `left/right_wrist_roll` | 0.599 mm | 11.459° |
| `left_ankle_pitch` | 3.506 mm | 11.459° |
| `left_knee` | 63.408 mm | 11.459° |

（11.459° 正是 0.2 rad，即施加的扰动量。）**twist在位置里不可见，在frame里完全可见。**

## 1.2 正确的解法：构造式gauge + 显式twist通道（v6重写）

> **v5的错误**：v5说"canonical空间的绝对frame朝向跨拓扑可比"。**这句话在没有定义gauge的前提下是错的**，Codex的P0 #2指出得对。
>
> 实测G1 URDF在rest pose（全部$q=0$）下各link frame相对identity的偏差：
>
> | link | rest状态固有旋转 |
> |---|---|
> | `left_shoulder_pitch_link` | **16.00°** |
> | `left_elbow_link` | 0.01° |
> | 多数腿部link | 0.00° |
>
> **URDF的link frame带任意的固定局部旋转，BVH rig有自己另一套**。同一根语义骨在两个rig里的"绝对frame朝向"会差一个常量旋转。v5那个4.29e-12的验证只证明**同一gauge内代数可逆**，不证明跨rig可比。

### 1.2.0 位置零空间实测：8维，不是4维

v6用逐DOF扰动测出4个DOF"从位置观测不到"。**那个测法低估了问题**——它只查了单个DOF的独立影响，没查DOF组合。正确做法是算Jacobian的秩：

对14个语义站点的世界坐标（42维输出）关于29个关节角求Jacobian $J\in\mathbb R^{42\times 29}$，在20个随机姿态下：

$$\mathrm{rank}(J)=21\ \ \text{(恒定)},\qquad \dim\ker(J)=29-21=\mathbf{8}$$

**位置观测丢失的是一个8维子空间**，不只是4个孤立的twist DOF。这让 $\Delta R$ 方案的必要性从"补4个角度"上升为"补一个8维不可观测子空间"——**任何只输出位置的方案，都有8个自由度是自由参数。**

### 1.2.1 构造式triad（gauge fixing的定义）

不使用rig自带的link frame。**从位置确定性地构造**每个关节的参考triad：

$$\hat{\boldsymbol x}_j=\frac{\boldsymbol p_j-\boldsymbol p_{\mathrm{pa}(j)}}{\|\boldsymbol p_j-\boldsymbol p_{\mathrm{pa}(j)}\|},\qquad \boldsymbol y'_j=\boldsymbol r_j-(\boldsymbol r_j\!\cdot\!\hat{\boldsymbol x}_j)\hat{\boldsymbol x}_j,\qquad \hat{\boldsymbol y}_j=\frac{\boldsymbol y'_j}{\|\boldsymbol y'_j\|},\qquad \hat{\boldsymbol z}_j=\hat{\boldsymbol x}_j\times\hat{\boldsymbol y}_j$$

$$E_j=[\hat{\boldsymbol x}_j,\ \hat{\boldsymbol y}_j,\ \hat{\boldsymbol z}_j]\in SO(3),\qquad \det E_j=+1$$

其中 $\boldsymbol r_j$ 是**参考向量**，规则（确定性，对任意骨架都有定义）：
1. 若 $\mathrm{pa}(j)$ 有父节点，取 $\boldsymbol r_j=\boldsymbol p_{\mathrm{pa}(\mathrm{pa}(j))}-\boldsymbol p_{\mathrm{pa}(j)}$（祖父方向）；
2. 否则取根节点指向其**第一个子节点**（按关节名字典序，保证确定性）的方向；
3. 若1/2都与 $\hat{\boldsymbol x}_j$ 共线（$\|\boldsymbol y'_j\|<10^{-6}$），改取该关节**所有兄弟节点位置的均值方向**；
4. 若仍共线，使用全局up轴（+Y）；若连它也共线（骨骼严格竖直），使用全局forward轴（+Z）。

**实测验证**（`re-gauge invariance test`，回答Codex要求的invariance test）：

| 检查 | 结果 |
|---|---|
| 两个rig几何相同、但URDF固有旋转差16° → rig自带frame差异 | 16.0° |
| **同一情形下构造式triad的差异** | **0.0°** |
| $\det E_j$ | +1.0（所有非退化情形） |

**几何相同则triad相同，与rig的frame约定完全无关。** 这就是gauge fixing。

### 1.2.2 退化情形（必须显式处理，不能静默出NaN）

实测 `g1_29dof.urdf` 中存在 **2个zero-length origin**：

```
waist_yaw_joint        type=revolute  |xyz| = 0.0    <- b/||b|| 会 0/0 -> NaN
pelvis_contour_joint   type=fixed     |xyz| = 0.0
```

| 退化情形 | 检出条件 | 处理 |
|---|---|---|
| zero-length bone | $\|\boldsymbol p_j-\boldsymbol p_{\mathrm{pa}(j)}\|<10^{-6}$ | **继承父关节的triad**（$E_j:=E_{\mathrm{pa}(j)}$）。该关节与父节点在同一位置，是串联轴的一部分（G1的`waist_yaw`正是此情形） |
| 参考向量共线 | $\|\boldsymbol y'_j\|<10^{-6}$ | 按上面规则3→4逐级回退 |
| 叶节点 | 无子节点 | triad照常定义（只依赖父/祖父，与子节点无关） |
| 分支节点 | 多个子节点 | triad照常定义（参考向量规则不依赖子节点数） |

**这四种情形必须写成单元测试**（`assert not isnan(E).any()` 覆盖全部78+29+动物关节）。

### 1.2.3 网络输出什么（v7重写——v6的标量$\varphi$方案已作废）

> **v6的错误（Codex第二轮P0，两条都对，我实测确认）**：
>
> **(a) 我只证明了 $E$ 的re-gauge不变性，没证明 $\varphi$ 的。** $E$ 由位置构造，所以re-gauge下当然不变——那个0.0°测试**必然通过，却没测到真正携带旋转信息的量**。而 $\varphi$ 是把rig自带frame投影到 $E$ 上得到的，rig做合法re-gauge $R_j\to R_jG_j$ 时它会变。实测：**随机per-link re-gauge下 $|\varphi-\varphi'|$ 最大 180.0°**——完全不invariant。
>
> **(b) 更致命：joint axis 根本不沿 bone axis。** 我逐关节实测了G1的29个revolute：
>
> | 情形 | 数量 |
> |---|---|
> | axis与bone对齐（$\|\cos\|>0.95$） | **6** |
> | **oblique（0.2–0.95）** | 3 |
> | **PERPENDICULAR（$\|\cos\|<0.2$）** | **19** |
> | zero-length（无bone axis） | 1 |
> | **标量bone-twist无法表示的** | **23 / 29** |
>
> Codex举的 `left_ankle_roll_joint` 是精确反例：bone方向 $(0,0,-1)$、axis $(1,0,0)$，$\|\cos\|=0.000$——**绕bone轴的标量能表示它的 0.0%**。而这恰恰是当初要救回的那个0mm零空间DOF。`waist_yaw` 更糟：zero-length继承父triad，真实axis是+Z，绕 $E.x$ 转根本表示不了yaw。
>
> **根因**：parent→joint的位置向量由**父link**姿态决定；joint $j$ 自己的转角不改变自己的origin，只改变child及后代。把该向量当作child frame的旋转轴，在任意URDF serial-axis chain上不成立。"$\boldsymbol p_j$ + bone-twist标量"适合部分动画长骨，**不是任意URDF link的通用SE(3)表示**。

### 采纳的方案：rest-relative global delta rotation

$$\boxed{\ \text{网络输出}\ \big(\ \boldsymbol p_j\in\mathbb R^3,\ \ \Delta R_j\in SO(3)\ \big),\qquad \Delta R_j(t)\ :=\ R_j(t)\,R_j(\text{rest})^{\top}\ }$$

$\Delta R_j$ 用**6D连续表示**（Zhou et al.）参数化输出，再Gram-Schmidt正交化。

**为什么这个可行——两条性质我都实测了**：

**1. gauge不变（这正是$\varphi$缺的）**：合法re-gauge $R\to RG$ 下
$$(R_jG_j)\,(R_j^{\text{rest}}G_j)^{\top}=R_jG_jG_j^{\top}R_j^{\text{rest}\top}=R_j\,R_j^{\text{rest}\top}=\Delta R_j$$
固定gauge $G_j$ **代数上自动消掉**。实测2000次随机 $(R_t,R_\text{rest},G)$：**最大偏差 3.2e-6 度**。

**2. 跨rig可比（这是我们真正需要的）**：两个rig各有自己的frame约定与rest frame，做**同一个语义动作** $W$ 时，
$$\Delta R^A=(WR^A_0)R_0^{A\top}=W,\qquad \Delta R^B=(WR^B_0)R_0^{B\top}=W$$
两者**都等于 $W$ 本身**，与各自rig怎么定义frame无关。实测2000次：**最大差异 3.0e-6 度**。

剩下的唯一要求是**两个rig的rest pose在语义上对齐**（都是T-pose、朝向/up一致）——**这正是§1.2的up/facing/scale canonicalization已经在做的事**，不需要额外机制。

**3. 它承载twist**：`ankle_roll` 转0.2 rad时，$\Delta R$ 的旋转幅度实测 **11.459°**（正是0.2 rad），与位置的0.000 mm形成对比。**零空间问题由此解决，且不依赖joint axis与bone axis的关系。**

### rest pose从哪来（对任意骨架都可得）

| 骨架类型 | $R_j(\text{rest})$ |
|---|---|
| URDF（G1及任何机器人） | 全部 $q=0$ 时的FK link frame |
| BVH（SOMA、Truebones动物） | HIERARCHY段定义的rest hierarchy（所有channel为0） |
| 无显式rest的资产 | 按§2.2的三级回退取anchor clip的第一帧 |

**这三种来源都不需要per-embodiment参数**，只需读各自的规格文件——满足§1.0的验收判据。

### 构造式triad $E_j$ 还留着吗

**留着，但降级为"anchor几何特征"，不再参与旋转表示。** 它仍然有用（作为§2.2几何anchor的一部分，描述骨架形状），但**旋转完全由 $\Delta R_j$ 承载**。§1.2.1/§1.2.2的退化处理仍然需要（anchor特征也不能出NaN）。

### 1.2.4 关节角：解析投影 + 数值稳定处理（回应P0 #5）

**解析投影，零学习参数。** 给定合成后的 $R_j$ 和目标机器人的URDF规格（关节轴$\hat a_j$、origin的rpy/xyz）：

$$R_j^{\text{rel}} = R_{0,j}^\top\, R_{\text{parent}(j)}^\top R_j,\qquad \boxed{\ \theta_j=\mathrm{atan2}\Big(\boldsymbol\omega\!\cdot\!\hat a_j,\ \ \mathrm{tr}(R_j^{\text{rel}})-1\Big)\ }$$

其中 $R_{0,j}$ 是URDF的origin旋转，$\boldsymbol\omega=\big(R^{\text{rel}}_{j,32}-R^{\text{rel}}_{j,23},\ R^{\text{rel}}_{j,13}-R^{\text{rel}}_{j,31},\ R^{\text{rel}}_{j,21}-R^{\text{rel}}_{j,12}\big)$ 是 $R_j^{\text{rel}}$ 的反对称部分（未归一化的log map轴，模长 $=2\sin\theta$）。

**统一用atan2形式，不用 `sign()·arccos`**（v6的prose说了改但公式和伪代码没跟上，Codex指出的文本残留之一——现已统一）。atan2的两个argument分别正比于 $\sin\theta$ 和 $\cos\theta$，在全域（含 $\theta\to0,\pi$）平滑可导，且不需要 `sign()`。

**已实测验证**：对G1的29个revolute DOF，从canonical frames解析恢复关节角，**最大误差 4.29e-12 rad、平均 6.67e-13 rad（机器精度），使用的学习参数为 0 个。**

这一步的意义：**它是一个可微的、纯几何的算子，不是一个"头"。** 换任何机器人只需换它的URDF，网络一个参数都不用改、不用训。这正好满足§1.0的验收判据。

**但那个4.29e-12是"精确FK的代数逆"，不代表对网络输出也稳定**——Codex这一点说得对。网络输出的 $R_j$ 带噪声、可能off-axis，此时上式有三个失稳点，必须处理：

| 失稳点 | 症状 | 处理 |
|---|---|---|
| $\arccos$ 在 $\mathrm{tr}=\pm 1$（即 $\theta\to 0$ 或 $\pi$） | 导数 $\to\infty$，autograd出inf/nan | 用 `atan2` 形式替代：$\theta=\mathrm{atan2}(\boldsymbol\omega\!\cdot\!\hat a_j,\ \mathrm{tr}(R^{\text{rel}}_{\perp}))$，全域平滑 |
| $R_j^{\text{rel}}$ 的旋转轴不平行于 $\hat a_j$（off-axis） | 单轴关节表达不了的成分被 `sign(dot)` 静默丢弃 | **先做joint-manifold投影**：把 $R_j^{\text{rel}}$ 投影到 $\{\mathrm{Rot}(\hat a_j,\theta)\}$ 这条一维子群上，并**把投影残差作为一个显式loss项** $\mathcal L_\text{axis}=\|R^{\text{rel}}_j - \mathrm{Rot}(\hat a_j,\theta_j)\|_F$ 上报，而不是丢掉 |
| `sign()` 在 0 处不可导 | 梯度为0或跳变 | atan2形式天然避免 |

**另外 $\mathcal L_\text{fk}$ 本身不惩罚joint-limit违反**（Codex说得对，FK对超限角度照样算得出位置）。因此单列一项：
$$\mathcal L_\text{limit}=\sum_j \big[\max(0,\ \theta_j-\theta_j^{\max})^2+\max(0,\ \theta_j^{\min}-\theta_j)^2\big]$$
限位从URDF读取（实测G1的`left_hip_pitch_joint`为$(-2.5307, 2.8798)$、`left_knee_joint`为$(-0.0873, 2.8798)$）。

**URDF关节类型的实际情况**（缩小Codex #5的担心范围）：实测 `g1_29dof.urdf` 共 **10个fixed + 29个revolute**，**0个 mimic、0个 prismatic、0个 continuous**。所以本轮只需处理 fixed（跳过）和 revolute（上式）两类。若Round-2接入带mimic/prismatic的机器人，再补规则。

## 1.3 汇总

| | 实现 | per-embodiment参数 |
|---|---|---|
| 编码器 | 通用graph-transformer | **0** |
| 瓶颈 | 通用（七个变体之一） | **0** |
| 解码器 | 通用，输出 $(\boldsymbol p_j,\ \Delta R_j)$ | **0** |
| 构造式triad $E_j$ | **降级为anchor几何特征**（不再参与旋转表示，§1.2.3末） | **0** |
| 关节frame $R_j=\Delta R_j\,R_j^{\text{rest}}$ | 确定性合成（$R^{\text{rest}}$读自URDF/BVH） | **0** |
| 关节角 | **解析投影**（需目标URDF） | **0** |
| 全局root | **本轮不预测**（§2.3） | — |

**$\Delta R$ 的监督按GT可用性做mask，但那是loss层面的事，与"输出头"无关。** 无旋转数据的rig就mask掉该项——**mask一个loss ≠ 加一个头**。网络结构对所有具身逐字节相同，只是不同样本激活的监督项不同。

## 1.3b 一个容易踩的误读：频域分解（DCT/FAST式）不能替代旋转通道

**结论：不能，而且这是数学上必然的。** 记录在此因为这是个很自然的误读。

pi0-FAST的做法是把一段动作序列（H步×D维）沿**时间轴**做DCT压缩，再用BPE式压缩打包成token——它解决的是**时间维度上的token化效率**问题（相比RT-2/OpenVLA那种逐维256均匀分箱），完全不触及**每一帧内部旋转信息缺失**的问题。

**实测验证**：构造两条轨迹，除`ankle_roll`在一条里扫过0.4 rad（约23°）外完全相同，沿时间轴做DCT-II：

| 量 | 两条轨迹差异 |
|---|---|
| 位置轨迹 | **0.000000 mm** |
| 全部DCT系数 | **0.000e+00** |
| 低频（前8） | 0.000e+00 |
| 高频（后8） | 0.000e+00 |

踝关节转了23°，位置轨迹与**所有**DCT系数逐比特相同。原因：DCT是沿时间的**线性映射**，而twist在**每一帧**的位置里都落在零空间（实测0.000 mm）——**线性变换无法创造每一帧都不存在的信息**。换FFT、wavelet、任何可逆变换，结论相同。

**且本轮是严格单帧，T=1时DCT退化为恒等映射，字面上无物可变换。**

频域那条思路属于**第二阶段（时序层）**的备选：若temporal层走离散自回归，FAST式DCT+BPE优于逐维分箱；或MotionVLA式低频离散+高频残差连续。但按当前定案temporal层优先走连续flow matching，那条路不需要量化。

## 1.4 为什么这个问题在时序阶段会放大（所以必须这一轮解决）

如果只编码位置，时间层看到的输入对那4个twist DOF**完全没有可观测信号**——不是学不好，是没有可学的东西。IK后处理只能**逐帧独立地猜**一个twist，相邻帧的猜测彼此无关，直接表现为**时序抖动**。这是表示缺陷造成的抖动，时间层的平滑loss只能掩盖不能消除（会把它压成缓慢漂移，看起来更"自然"但依然是错的）。

**输出canonical frame从根上消除这个问题**：twist在frame里是可观测的（11.459°），时间层因此有信号可用。

**输入侧**：$r_j$ 作为通道，配 `rot_avail` 标记（动物置零）。注意这是**输入的数据可用性标记**，不是网络分支——所有样本走同一套权重。

---

# 2. 数据IO规格

## 2.0 坐标表示：root-relative，不是世界坐标、也不是逐骨父子相对

**本方案不用世界坐标。** G1骨盆离地78cm、人体髋101cm，同一个"举手"在世界坐标下数值完全不同，跨具身无从对齐；且整人平移一米所有坐标全变，模型要浪费容量学"平移不改变姿态"这件本该免费的事。

**SAMoR用的是root-relative**（原文："we therefore use **root-normalized** joint positions and velocities as the shared cross-rig signal"；"Global root translation is **excluded** because heterogeneous assets have inconsistent reference frames"）。

**我们也用root-relative，理由是误差累积。** 父子相对（bone-vector）表示下，末端位置要靠沿链累加每根骨的误差才能得到。实测链深度：

| 关节 | 深度 | | 关节 | 深度 |
|---|---|---|---|---|
| SOMA `LeftFoot` | 4 | | G1 `ankle_roll_link` | 6 |
| SOMA `Head` | 7 | | G1 `wrist_yaw_link` | 10 |
| SOMA `LeftHand` | 8 | | G1 `rubber_hand` | 11 |
| SOMA 指尖 | 12 | | G1 最大 | **11** |
| SOMA 最大 | **13** | | | |

每骨独立误差$\epsilon$时，深度$d$处末端误差按$\epsilon\sqrt d$增长：$\epsilon=1$mm下，LeftFoot 2.00mm / LeftHand 2.83mm / G1 hand 3.32mm / 指尖 3.46mm；$\epsilon=2$mm下翻倍（4.00 / 5.66 / 6.63 / 6.93mm）。

**而末端位置误差正是第一评估指标**（§7维度A第1项）。父子相对等于把误差最大处放在最在乎的量上。

**但父子相对以loss形式出现**：这正是 $\mathcal L_\text{bone}$（权重2.0）的作用——**表示用root-relative，约束用父子相对**。两者互补不矛盾。

**三条必须配套的归一化**：
1. 按骨架高度$H$归一化尺度；
2. 对齐up-axis与facing direction（见prep清单§C.1）——SAMoR对所有角色做了"normalized by mesh bounding box, up-axis, and facing direction"；
3. **rest-pose几何单独走anchor通道**，那里**才**用父子相对骨向量——anchor描述"骨架长什么样"（父子相对是自然表示），motion描述"这帧摆成什么姿势"（root-relative是自然表示）。**两者用不同表示是有意为之。**

## 2.1 输入（每个关节一个token）

```python
# 源端 token: 只描述"源姿态是什么"。约束不在这里(见§5.0), 因为约束属于 target.
x_j = [
    p_j,                  # (3,) root-relative 位置, cm, SOMA的Y-up右手系
    dR6_j * dR_avail,     # (6,) rest-relative delta rotation 的6D表示 (§1.2.3)
    dR_avail,             # (1,) 该样本是否有旋转数据
]                         # 共 10 维
```

**v7变更**：$\Delta R_j$ 用**6D表示**而非v6的标量 $\varphi_j$——v6的标量方案已作废（§1.2.3详述：$\varphi$ 不gauge-invariant，且G1 29个revolute里有23个的axis不沿bone轴，标量根本表示不了）。`is_ctrl`/`ctrl_val` 仍然不在输入里，走decoder的target-side（§5.0）。

**没有速度通道。** 单帧任务里速度与标签无关（目标姿态由源姿态唯一确定），且速度依赖帧率（BONES-SEED是120fps，接其他帧率时同一动作的$v$差数倍而$p$不变），会污染瓶颈的姿态语义。这与SAMoR不同，因为SAMoR处理的是时序chunk，速度在那里承载真实时序信息。

## 2.3 全局root：本轮不预测（v6修订，回答#4）

> **v5的错误**：v5一边说"全局root平移排除出瓶颈"（输入端删掉），一边说"用一个通用root通道输出"。**这在单帧下不可识别**——Codex的P0 #3说得对：输入是root-relative的，全局root信息已经被移除，网络无从恢复它。这不是精度问题，是信息论上的不可能。

**本轮决定：不预测global root。** 输入输出都是root-relative。

理由与后续安排：
- 单帧root-relative输入**不含**全局位姿信息，要求输出它等于要求网络凭空生成。
- SAMoR同样排除global root（"heterogeneous assets have inconsistent reference frames"），HumanML3D那一路用**独立的轻量VQ分支**单独建模root通道——即它也没让主瓶颈承担这件事。
- **遥操作确实需要全局位姿**，但那属于时序层/旁路：真实遥操作场景里root位姿由外部提供（VR头显6DoF、SLAM定位、或操作者手柄），**不需要从单帧姿态里推断**。
- 若Round-2需要，可作为**旁路通道**（不进瓶颈）直接从输入的全局信息passthrough。

**这一条同时简化了评估**：本轮所有位置指标都在root-relative空间计算，不存在"全局漂移"这个混淆项。

## 2.2 每个关节的身份信息（三部分，都不是可学习的查找表）

**关键约束：不能用可学习的per-joint embedding表**，否则遇到训练时未见过的关节名/新拓扑时无法查表，`val_topo`直接失效。

1. **SigLIP文本embedding（冻结）**：规范化后的关节名 → SigLIP文本塔 → 固定向量。训练时随机dropout（置零），强制模型不能只靠名字、必须用几何信息。这也是应对新拓扑中未见关节名的关键。
2. **拓扑位置编码**：作为attention bias注入——hop距离（Floyd-Warshall算最短路后截断到最大值）、是否同一运动链、边类型。**不是**LLM那种线性位置编码，骨架是树不是序列。
3. **几何anchor（相对量，非绝对坐标）**：
$$\text{anchor}_j=\mathrm{MLP}\big([\,\mathrm{Fourier}(\hat{\boldsymbol b}_j),\ \|\boldsymbol b_j\|/H\,]\big),\qquad \boldsymbol b_j=\boldsymbol p_j^{\text{rest}}-\boldsymbol p_{\mathrm{parent}(j)}^{\text{rest}}$$
$\hat{\boldsymbol b}_j$为单位骨向量（尺度无关），$\|\boldsymbol b_j\|/H$为按骨架总高归一化的骨长。**Fourier编码不可省**：MLP直接吃低维原始坐标有频谱偏置，对几何细节不敏感。
**必须用相对量**：G1骨盆78cm、人体髋101cm，绝对坐标数值域完全不同，无法跨具身泛化。

动物/无rest-pose资产用**第一帧**作为pose anchor——照搬SAMoR，原文特别说明第一帧"more reliable than asset-provided rest poses for heterogeneous sources"（异构资产自带的rest pose常常不可靠）。SOMA/G1有canonical rest pose，用rest pose。

**但这有target leakage风险**（Codex P1 #12，我接受）：若anchor取的是**同一clip**的第一帧，它携带的不只是morphology，还可能携带该clip的姿态信息——而姿态正是我们要预测的东西。

**修正为三级回退**：
1. 有canonical rest pose（SOMA、G1）→ **用rest pose**（无泄漏）；
2. 无rest pose但该物种有**多个clip** → 用**另一个随机clip**的第一帧（morphology相同、姿态无关）；
3. 该物种只有一个clip → 退回用本clip第一帧，**并在数据manifest里标记 `anchor_may_leak=True`**。

**T1必须做一次anchor-leakage消融**：把标记了 `anchor_may_leak` 的样本单独统计误差，若显著低于其他样本，说明泄漏在起作用，需要改用跨clip的shape-only anchor（例如骨长向量而非关节位置）。

**SAMoR有一个反直觉的消融结果，直接影响SigLIP的投入产出比**：去掉关节名条件后性能只下降**13–16%**，原文明确说"confirming SAMoR does not rely on joint-name strings as a shortcut"；且规范关节名的覆盖率在HumanML3D是100%、Objaverse-XL只有69%、Truebones Zoo只有81%。

两个推论：(a) **动物关节名标不全甚至完全不标不会毁掉模型**，几何结构本身承载大部分信息；(b) **SigLIP这一路的收益上限有限，不要过度投入**——先用最简单的方式接上（冻结文本塔+dropout），不要花时间调它。

---

# 3. 训练任务矩阵与七个候选的架构

## 3.0 训练任务矩阵（v6新增，回答#1——这是之前最大的缺口）

> **v5的缺口**：只定义了帧采样和数据切分，**没有定义一个batch里 $(\text{source rig}, \text{source pose}, \text{target rig}, \text{target pose})$ 怎么构造**，也没有各方向的采样比例和GT来源。Codex的P0 #1完全正确——没有这个矩阵，"跨拓扑翻译"这个任务本身是未定义的。

**每条训练路径的定义**：

| 路径 | source → target | GT来源 | 采样权重 | 这条路径能证明什么 |
|---|---|---|---|---|
| **A. SOMA→G1** | SOMA-78 → G1-29 | **BONES-SEED逐帧配对**（真配对，NVIDIA retargeter产出） | **0.30** | 跨拓扑翻译（唯一有真GT的方向） |
| **B. G1→SOMA** | G1-29 → SOMA-78 | 同上，反向 | **0.15** | 反向翻译；检验是否只学会单向 |
| **C. SOMA→剪枝** | SOMA-78 → SOMA-38/24/20 | **确定性剪枝**（子集选取，GT=原关节的位置子集） | **0.20** | 拓扑变化下的翻译，GT无歧义 |
| **D. 同rig自重建** | X → X（X∈{SOMA, G1, 剪枝, 动物}） | 恒等 | **0.20** | 表示容量的下界；**不能证明跨拓扑能力** |
| **E. 动物cycle** | 动物$s_1$ → 动物$s_2$ → 动物$s_1$ | **无直接GT**，只约束cycle闭合 | **0.15** | 拓扑正则化；**不是翻译证据** |

**三条硬规则**：
1. **Truebones动物只出现在路径D和E，绝不出现在"有GT的跨拓扑翻译"里。** 各物种只有自己的motion，没有"同一姿态在另一物种拓扑上的真值"——这类GT在物理上不存在（狗的姿态没有唯一的"猫版本"）。
2. **路径E不产生任何精度数字上报。** 它的作用是给 $\mathcal L_\text{adv}$ 和瓶颈的拓扑无关性提供训练压力，是**正则化项**。cycle误差低只说明编解码自洽，**不说明翻译正确**。
3. **路径C是路径A的重要补充**：剪枝的GT是无歧义的（就是原关节位置的子集），不像A的GT本身是retargeter产物。**若A和C的结论矛盾，以C为准**——因为C的GT更干净。

**关于路径A的GT性质（必须诚实标注）**：SOMA→G1的"GT"是NVIDIA用IK retargeter生成的，不是物理真值。**所以路径A实质是在蒸馏一个retargeter**，在它上面"超过GT"没有意义。真正的价值在C（无歧义GT）和零样本泛化。

## 3.1 完全共享的骨干（七个变体逐行相同）

```python
# ============ Encoder ============
# 输入: 严格单帧, 源骨架 J_src 个关节
h = MLP(x)                                        # (J_src, 10) -> (J_src, D) 见§2.1
h = h + SigLIP_frozen(canonical_name) * name_dropout_mask
h = h + anchor_embed(fourier(bone_dir), bone_len / H)

for _ in range(L_enc):                            # 几何双轨 graph transformer
    h = SelfAttn(h, bias = hop_bias + same_chain_bias + edge_type_bias)
# h: (J_src, D)

# ---- part pooling: 仅变体2-7; 变体1(Direct-Token)跳过 ----
Q_part = Parameter(K, D)                          # K=8 起点
A      = softmax(Q_part @ h.T / sqrt(D))          # (K, J_src)
u      = LayerNorm(Q_part + A @ h)                # (K, D)
# 变体1(Direct-Token): u = h, 形状 (J_src, D)

# ============ 瓶颈：七选一 ============
z = BOTTLENECK(u)

# ============ Decoder ============
# 目标骨架 J_tgt 个关节, J_tgt 可完全不同于 J_src.
# 约束是 TARGET-SIDE 的 (见§5.0): 四个张量都按 J_tgt 索引, 与 J_src 无关.
T = init_frames(target_pose_anchor)               # (J_tgt,) SE3 frames
Q = anchor_embed(target) + SigLIP_frozen(target_names) + tree_bias
Q = Q + ctrl_embed(ctrl_pos_val) * ctrl_pos_mask  # (J_tgt,3) / (J_tgt,1)
Q = Q + ctrl_embed(ctrl_rot_val) * ctrl_rot_mask  # (J_tgt,1) / (J_tgt,1)

for _ in range(N_recycle):
    for _ in range(L_dec):
        loc    = InvariantPointAttention(Q, T, context=z)
        dR, dt = MLP(loc)
        T      = compose(T, (dR, dt))
    T = T.detach() if recycle_detach else T        # AlphaFold式: 中间轮不回传

# ==== 输出: 一个通用头, 零 per-embodiment 参数 ====
p_out, d6 = out_head(T)                           # (J_tgt,3), (J_tgt,6)
dR_out    = gram_schmidt_6d(d6)                   # (J_tgt,3,3) rest-relative delta §1.2.3

# ---- RePaint式硬替换: 位置/旋转分量各自独立 ----
p_out  = where(ctrl_pos_mask, ctrl_pos_val, p_out)
dR_out = where(ctrl_rot_mask, ctrl_rot_val, dR_out)   # ctrl_rot_val 也是 dR (3x3)

# ---- 以下全是确定性几何, 零学习参数 ----
R_out = dR_out @ R_rest[target]                   # 还原绝对frame; R_rest 读自 URDF/BVH
q_out = analytic_angles(R_out, target_urdf)       # §1.2.4, 仅在有URDF时
# 注意: 本轮不输出 global root (§2.3)
```

**没有URDF的骨架怎么办**（这是实现时立刻会撞上的问题）：SOMA的BVH和Truebones的动物骨架**都没有URDF**，没有关节轴定义，因此**无法也不需要做解析角度投影**。

处理方式很简单——**这一步是可选的后处理，不是网络的一部分**：

| 具身 | 有URDF? | `q_out` | $\mathcal L_\text{fk}$ |
|---|---|---|---|
| G1-29（及任何机器人） | ✓ | 解析投影 | 激活 |
| SOMA-78 / 剪枝变体 | ✗ | **不计算**（BVH骨架无转轴定义） | 不激活 |
| Truebones 动物 | ✗ | **不计算** | 不激活 |

**这不违反universal scope**：网络的输出（canonical SE(3) frames）对所有具身完全一致；解析投影只是在"目标恰好是一个有URDF的机器人"时，把frame翻译成电机能吃的角度。**换机器人只换URDF文件；没有URDF的目标就不需要这一步。** 网络一个参数都不因此改变。

**BVH骨架若真需要局部旋转**：BVH自带每关节的欧拉角通道，可以用同样的log-map思路从canonical frame反解出局部旋转（相对父节点），但那是三自由度的完整旋转、没有"绕单轴"的约束，因此不存在twist ambiguity问题，也不需要URDF。本轮不需要这条路径。

**`analytic_angles` 的实现**（纯几何，无参数）：
```python
def analytic_angles(R_abs, urdf):                 # 对任意机器人通用
    q = []
    for j in urdf.revolute_joints:                # 逐关节闭式求解
        R_rel = urdf.R0[j].T @ R_abs[urdf.parent[j]].T @ R_abs[j]
        w     = skew_vec(R_rel - R_rel.T)         # 未归一化 log 轴, |w| = 2 sin(theta)
        # atan2 形式: 全域平滑, 无 sign(), 无 arccos 端点奇异
        q.append(atan2(dot(w, urdf.axis[j]), trace(R_rel) - 1))
    return stack(q)
```

**用IPA（Invariant Point Attention）+ 显式frame，而不是纯pairwise距离反解坐标**：纯距离几何无法唯一确定手性（镜像解），这正是AlphaFold用参考frame而非点云来处理化学手性的原因。骨架同理，左右手会搞反。

## 3.2 七个瓶颈的规格

| # | 变体 | 瓶颈实现 | 额外loss | 关键超参 |
|---|---|---|---|---|
| 1 | **Direct-Token** | **无pooling无量化**：`z = h`，`(J_src, D)`，每关节一token | 无 | 无$K$ |
| 2 | **AE** | pooling到`(K,D)`后 `z = u`（或过一层线性），确定性 | 无 | $K$ |
| 3 | **VAE** | `μ,logσ = Head(u); z = μ + σ·ε` | $+\beta_\text{KL}D_\text{KL}(q\|\mathcal N(0,I))$ | $\beta_\text{KL}$，**必须warmup**（0线性升到目标值）防posterior collapse |
| 4 | **CVAE** | `μ,logσ = Head([u, e_tgt])`，$e_\text{tgt}$为目标骨架embedding | 同VAE | 同上。**注意它的latent契约与其他六个不同，见下方说明** |
| 5 | **VQVAE** | 单层码本$V\times D$，最近邻查找，straight-through | $+\gamma\|u-\mathrm{sg}[z_q]\|^2$（**只保留encoder侧commitment**；码本用EMA更新，见下） | **$V$=4096**，$\gamma$=0.25 |
| 6 | **RVQ** | $R$层残差量化 | 每层commitment之和（同上） | **$R$=2，每层$V$=64** |
| 7 | **FSQ** | 逐维bounded round：`z = round(clip(Linear(u)))` | **无**（FSQ不需commitment也不需码本） | **levels=[4,4,4,4,4,4]** |

**v6修正了两处**（Codex P1 #11）：
- **码本更新规则二选一，不能并用**：v5同时写了"EMA更新"和"码本梯度loss $\|\mathrm{sg}[u]-z_q\|^2$"，那是矛盾的。**本轮统一用EMA更新码本**（decay=0.99），loss里**只保留encoder侧的commitment项** $\gamma\|u-\mathrm{sg}[z_q]\|^2$。
- **$\mathcal L_\text{vq}$ 表里写了"commitment + entropy"但公式没有entropy**。本轮**不加entropy项**（FSQ不需要，VQ/RVQ用EMA已能缓解码本坍缩），改为用**指标11码本利用率**来监控坍缩。若T0发现利用率低于30%，再考虑加entropy正则并记录在案。

**配置数字已按§3.3做码率对齐，不是随手取的**——见下节。

**关于CVAE的latent契约冲突（Codex P1 #10，接受）**：CVAE把 $e_\text{tgt}$ 拼进posterior，所以它的 $z$ **随目标骨架变化**——同一个源动作翻译到G1和翻译到Optimus会得到不同的 $z$。而项目规划要求第二阶段的时间层直接预测canonical $z$（一个target-independent的motion state）。**这两者不能同时成立。**

**本轮的处理**：CVAE**仍然作为对照组参赛**（它可能在单帧翻译精度上更好，这个信息有价值），但**明确标注它的latent不是target-independent的motion state**。

**因此增加一条硬性判据**：若CVAE在T1胜出，**不能直接把它接到第二阶段**——要么改成"target只进decoder、不进posterior"的变体重训，要么第二阶段的latent契约相应改变（那意味着时间层要为每个目标具身各跑一次，违背universal初衷）。**这个取舍必须在T1结果出来时显式记录，不能默认继承。**

其余六个变体的 $z$ 都是target-independent的（目标骨架信息只通过decoder的Query进入），与第二阶段契约兼容。

**Direct-Token为什么必须有**：它是检验"压缩会不会损失SE3精度"这个问题的唯一对照组。如果它SE3精度明显更好，说明压缩有代价；如果它在`val_topo`上明显更差，说明压缩正是泛化的来源。没有它，这个问题只能靠信仰。

**但它不公平，报告时必须标注**：它的瓶颈宽约10倍（$J_\text{src}$=78 vs $K$=8）。**不能简单看谁误差低就宣布谁赢**——见§3.3的两表制。

## 3.3 码率公平性（v6新增，回答#8——比FSQ算错本身严重得多）

> **v5的错误**：FSQ隐式码本我写成640,000，实际 $8^4\times 5^4=4096\times 625=$ **2,560,000**。Codex算对了。
>
> 但顺着查下去暴露一个更要紧的缺陷——**三个量化变体码率差4倍**：

| 变体（v5配置） | bits/token |
|---|---|
| VQVAE (V=1024) | 10.0 |
| FSQ [8,8,8,8,5,5,5,5] | **21.3** |
| RVQ (R=4, V=1024) | **40.0** |

**码率差4倍还比谁精度高，比的是码率不是瓶颈设计。** 这让整个量化侧对比无效。

### v7实测：12 bits/token 太紧，必须提到 20

**这个数不该靠猜。** 率失真理论对高斯源给出闭式的码率-失真下界，我用真实数据算了：从BONES-SEED抽2,668帧G1姿态（29维关节角），做PCA得各主成分方差 $\lambda_i$，再用**反向注水**（reverse water-filling）求最优比特分配下的失真下界 $D(R)$。

**顺带确认了一个prep清单里的待办（C.2）**：G1 CSV的关节列是**度不是弧度**——29/29列都落在URDF限位的度数版本内（`left_knee` 数据最大值恰为 165.000，URDF上限 2.8798 rad = 165.0°），弧度解释则 0/29 成立。下表已按弧度换算。

| bits/frame | bits/token (K=8) | RMS角度误差 | 末端误差 @0.6m力臂 |
|---|---|---|---|
| 48 | 6.0 | 4.284° | 44.9 mm |
| 64 | 8.0 | 2.923° | 30.6 mm |
| **96** | **12.0（v6的选择）** | **1.360°** | **14.2 mm** |
| 128 | 16.0 | 0.633° | 6.6 mm |
| **160** | **20.0** | **0.295°** | **3.1 mm** |
| 200 | 25.0 | 0.113° | 1.2 mm |
| 256 | 32.0 | 0.030° | 0.31 mm |

**这是下界**：假设最优高斯编码、单一具身、零模型误差、无跨拓扑负担。真实编码器只会更差。

**结论：12 bits/token 的理论最优末端误差就有 14.2 mm**——而这个量级已经超出"SE3精确控制"的可用范围（作为参照，SAMoR的跨拓扑重建误差是 $2.75\times10^{-2}$ 归一化MPJPE）。**在12 bits下，即使模型完美，量化本身就把精度锁死了。**

**修正：量化三者对齐到 20 bits/token（160 bits/frame，下界0.295°/3.1mm）**：

| 变体 | v7配置 | bits/token |
|---|---|---|
| **VQVAE** | $V$=1,048,576 → **不现实**，改用 **$V$=4096 + 每token 2个码本**（product quantization, 2×12=24）或降级参赛 | 见下 |
| **RVQ** | **$R$=2，每层 $V$=1024** | **20.0** |
| **FSQ** | **levels=[8,8,8,8,4,4,4,4]**（$8^4\times4^4=1{,}048{,}576$） | **20.0**（精确） |

**这暴露了一个更深的事实，必须写下来**：单码本VQVAE在20 bits需要 $2^{20}\approx 10^6$ 个码字，**训练上不现实**（码本利用率会极低）。所以：

- **RVQ 在高码率区间有结构性优势**（多层残差，码率线性叠加而码本大小不变）；
- **单码本VQVAE在这个任务上有先天码率上限**（$V$=4096即12 bits，再高就训不动）；
- **FSQ的码率由levels连乘决定**，提升靠加维度，比VQ容易但格点会变粗。

**所以"谁最适合当翻译官"这个问题，一部分答案在训练之前就能给出**：如果目标精度要求 <5mm，单码本VQVAE基本出局，不是因为它设计差，而是因为它到不了那个码率。这一点应当在报告里作为**预测**写明，然后用实验验证——若实验结果与此矛盾，说明我的率失真模型（高斯假设）不成立，那本身是有价值的发现。

**给Codex的执行建议**：不要只跑一个码率点。**每个量化变体跑 {12, 16, 20, 24} bits/token 四个点**，画出实际的quality-rate曲线，与上表的理论下界叠在同一张图上。**曲线与下界的垂直距离 = 该瓶颈设计的实际损失**，这比单点排名信息量大得多，而且能直接看出谁在哪个码率区间更优。

**连续变体没有"码率"，只能报浮点容量**：

| 变体 | 瓶颈容量 |
|---|---|
| AE / VAE / CVAE | $K\times D=8\times 256=2048$ floats |
| **Direct-Token** | $J\times D=78\times 256=19968$ floats（**约10倍宽**） |

**因此最终报告必须是两张表，不能只报一个排名**：
1. **matched-bitrate表**：三个量化变体在12 bits下的严格对比——**这个对比是干净的**；
2. **quality-rate Pareto图**：横轴瓶颈容量（量化用bits、连续用floats）、纵轴末端误差，**七个变体全画进去**。Direct-Token会在最右侧——它的优势若只来自10倍容量，Pareto图上一眼可见。

**绝不能只报"谁误差最低"**，那个数字会被容量差异主导。

**$K$的扫描**（$K\in\{4,8,16\}$）只在T1胜出的1-2个变体上做，不要7×3全扫。SAMoR原文做过$K$的消融，$K=8$优于$K=1$（单token全局）和$K=J$（逐关节无压缩），但那是动画重建任务的结论，是否适用于SE3精确控制需自己验证。

---

# 4. Loss

## 4.1 组成

| 项 | 权重 | 说明 |
|---|---|---|
| $\mathcal L_\text{pos}$ | **1.0** | smooth-$\ell_1$ on root-relative 关节位置 |
| $\mathcal L_\text{bone}$ | **2.0** | 骨长一致性（父子距离的平方误差） |
| $\mathcal L_\text{attn}$ | **1.0** | attention supervision（仅有pooling的变体2-7） |
| $\mathcal L_{\Delta R}$ | 1.0 | **旋转测地距离**（见§4.2），按 `dR_avail` mask。**mask一个loss项 ≠ 加一个头**——网络权重对所有具身完全相同 |
| $\mathcal L_\text{ctrl}$ | 1.0 | 约束一致性，见§5.2 |
| $\mathcal L_\text{fk}$ | 1.0 | **FK一致性**：$\|\mathrm{FK}(q_\text{out})-p_\text{out}\|$。仅在有URDF时激活（本轮即G1） |
| $\mathcal L_\text{limit}$ | **1.0** | **关节限位违反**（v6新增，见§1.2.4）。$\mathcal L_\text{fk}$ 不惩罚超限，必须单列 |
| $\mathcal L_\text{axis}$ | **0.1** | **单轴投影残差**（v6新增，见§1.2.4）。上报而非丢弃off-axis成分 |
| $\mathcal L_\text{adv}$ | **0（T0/T1初期）→ 0.1（Truebones接入后）** | **具身判别器对抗**（v6：v5只在§4.4提了却没进这张表）。只有2个具身时判别器近乎平凡，故初期置0 |
| $\mathcal L_\text{vq}$ | 1.0 | 仅变体5/6，**只含encoder侧commitment**（码本走EMA，见§3.2） |
| $\mathcal L_\text{KL}$ | warmup | 仅变体3/4，0→$\beta_\text{KL}^{\max}$ 线性升 |

**关于权重的一个重要修正**（Codex P1 #11最后一条，我接受）：v5说"权重直接照搬SAMoR Table 5"。但**SAMoR的表示、单位、任务、decoder都和我们不同**（它是时序chunk+速度通道+局部6D旋转+VQ，我们是单帧+rest-relative $\Delta R$+无速度），照搬权重不具可比性。

**修正做法**：以SAMoR的权重作**初始值**，但在T0阶段**记录每一项的gradient norm**，若某项的梯度量级比 $\mathcal L_\text{pos}$ 差10倍以上则重新标定。$\mathcal L_\text{bone}$=2.0 这一条仍值得保留（SAMoR对它做过消融，跨拓扑下骨长守恒比命中位置更重要），但要用实测梯度确认它没有压倒位置项。

**去掉的**：$\mathcal L_\text{vel}$（无速度通道）、$\mathcal L_\text{smooth}$（单帧无时序）。

## 4.2 旋转loss（v7：$\Delta R$ 的测地距离）

$$\mathcal L_{\Delta R}=\frac{1}{|\mathcal A|}\sum_{j\in\mathcal A}\Big(1-\tfrac{1}{3}\mathrm{tr}\big(\Delta R_j^{\top}\Delta R_j^{\text{GT}}\big)\Big),\qquad \mathcal A=\{j:\texttt{dR\_avail}_j=1\}$$

用这个**chordal形式**而不是 $\arccos$ 测地距离：两者在小角度下等价，但chordal**处处光滑可导**（$\arccos$ 在 $\mathrm{tr}=\pm1$ 处导数发散，与§1.2.4是同一类问题）。取值范围 $[0,\ 4/3]$，$0$ 表示完全对齐。

**报告时**仍换算成角度（$\theta=\arccos\frac{\mathrm{tr}(\Delta R^\top\Delta R^{\text{GT}})-1}{2}$，单位度）便于解读——**训练用chordal，报告用角度**。

**不要用6D向量的朴素L2**：6D表示的欧氏距离与实际角度差不成正比，且同一个旋转有多个6D表示（Gram-Schmidt前）。

## 4.3 跨拓扑不对称的loss策略

SOMA-78 vs G1-29的实测差异：SOMA丢弃**59个**关节（双手手指48 + 脚趾4 + 头颈7 = 59；78−59=19个有语义对应）。G1**没有任何头/颈/手指/脚趾自由度**。

**精确的对应关系**（已逐关节实测枚举，不要用估计值）。去掉手指48+脚趾4+头颈7=59个后，SOMA剩余**19个**关节：

```
Root, Hips, Spine1, Spine2, Chest,
LeftShoulder,  LeftArm,  LeftForeArm,  LeftHand,
RightShoulder, RightArm, RightForeArm, RightHand,
LeftLeg,  LeftShin,  LeftFoot,
RightLeg, RightShin, RightFoot
```

这19个里，**只有16个在G1侧有语义对应**：

| SOMA关节 | G1对应 | G1的DOF数 |
|---|---|---|
| `Hips` | pelvis | （root通道） |
| `Spine1/Spine2/Chest` | `waist_yaw/roll/pitch` | 3 |
| `LeftArm`/`RightArm` | `shoulder_pitch/roll/yaw` | 3 each |
| `LeftForeArm`/`RightForeArm` | `elbow` | 1 each |
| `LeftHand`/`RightHand` | `wrist_roll/pitch/yaw` | 3 each |
| `LeftLeg`/`RightLeg` | `hip_pitch/roll/yaw` | 3 each |
| `LeftShin`/`RightShin` | `knee` | 1 each |
| `LeftFoot`/`RightFoot` | `ankle_pitch/roll` | 2 each |
| **`LeftShoulder`/`RightShoulder`** | **无——G1没有锁骨自由度** | **0** |
| `Root` | **本轮不预测**（§2.3） | — |

**两个必须注意的实现细节**：
1. **对应关系不是joint-to-joint，是joint-to-DOF-group**。G1的29个DOF只分布在约14个关节位置上（髋/肩/腕/腰各3自由度）。**一个SOMA关节可能对应G1的3个DOF**——这正是为什么输出canonical frame（每关节一个$SE(3)$）比输出角度更自然：frame天然表达"这个关节的完整姿态"，不需要关心它被分解成几个转轴。
2. **`LeftShoulder`/`RightShoulder`是锁骨，G1没有对应自由度**。它们不在16个里。别试图硬映射到`shoulder_pitch`——那是SOMA的`LeftArm`。这是个很容易搞错的地方，因为名字听起来像。

**采用分层策略**：
- **16个有对应的关节**：$\mathcal L_\text{direct}$ 全权重监督；
- SOMA独有关节（手指48/头颈7/脚趾4，以及锁骨2）：decode到G1时**mask掉不计loss**，但在SOMA→SOMA自重建路径上仍计loss。

这样翻译官仍学到手部synergy，只是不强求G1侧有对应输出。**不要**给G1补虚拟关节——那是引入虚假监督信号。

## 4.4 具身判别器对抗loss（$\mathcal L_\text{adv}$）

瓶颈token上接一个小判别器猜"这来自哪个具身"，encoder通过梯度反转最大化判别器损失，防止$z$退化成"每个具身各占一块子空间"的偷懒解。

**注意**：只有2个具身时判别器几乎是平凡任务。Truebones到位（约70种拓扑）后这一项才真正有意义。**T0/T1早期只有SOMA+G1时，这一项权重设0或很小**，Truebones接入后再开。

---

# 5. SE3约束控制

## 5.0 约束契约：纯target-side，位置/旋转分离（v6重写，回答#5）

> **v5的错误**（Codex P0 #4，三处都对）：
> 1. `ctrl_val` 只有3维位置，伪代码却写 `T[is_ctrl] = ctrl_val` 赋给一个SE(3) frame——**维度不匹配**；
> 2. 约束在编码侧被拼进**source token**，但约束语义上属于**target关节**——`J_src ≠ J_tgt` 时**完全无定义**（这是最严重的，78个源关节的mask怎么对应29个目标关节？）；
> 3. 位置约束与旋转约束没有分离，无法表达"只钉住手腕位置、不管手腕朝向"这种最常见的需求。

**修正后的契约——四个张量，全部按 $J_\text{tgt}$ 索引，与 $J_\text{src}$ 无关**：

| 张量 | 形状 | 含义 |
|---|---|---|
| `ctrl_pos_mask` | $(J_\text{tgt}, 1)$ | 该目标关节的**位置**是否被约束 |
| `ctrl_pos_val` | $(J_\text{tgt}, 3)$ | 约束的目标位置（root-relative, cm） |
| `ctrl_rot_mask` | $(J_\text{tgt}, 1)$ | 该目标关节的**朝向**是否被约束 |
| `ctrl_rot_val` | $(J_\text{tgt}, 6)$ | 约束的 $\Delta R$（6D表示） |

**三个要点**：
1. **约束只进decoder的Query，不进encoder。** 源端token只描述"源姿态是什么"，目标端Query描述"目标骨架是什么 + 用户要求什么"。这样 $J_\text{src}\neq J_\text{tgt}$ 天然无冲突。
2. **位置和旋转独立mask。** 可以只约束位置（最常见：拖动手腕到某个点）、只约束twist（少见但合法：要求足底保持水平）、或两者都约束。
3. **硬替换也按分量独立进行**（见§3.1伪代码的两个 `where`），不是替换整个frame。

**这也修掉了v5输入特征里的一处冗余**：`is_ctrl`/`ctrl_val` 不再出现在输入 $x_j$ 中（§2.1已相应更新）。

## 5.1 训练时的掩码采样

每个样本随机采约束掩码，**按§5.0的target-side契约**：在 $J_\text{tgt}$ 上采 `ctrl_pos_mask` 与 `ctrl_rot_mask`（两者独立采样），对应的 `ctrl_pos_val`/`ctrl_rot_val` 取target侧GT。**只注入decoder的Query，并在输出端按分量硬替换——不进encoder。**

（v6此处仍残留"编码侧进`is_ctrl`/`ctrl_val`"的旧说法，与§5.0冲突，Codex指出的文本残留之二，现已改正。）

三种模式的采样比例 **`全约束 : 末端稀疏 : 无约束 = 2 : 6 : 2`**：
- 全约束：所有关节都给（退化成纯retargeting）；
- **末端稀疏**：只给手腕/脚/骨盆/头，其余脑补——**这是"翻译官"最核心的使用场景，所以占比最高**；
- 无约束：什么都不给，纯从源姿态翻译。

## 5.2 一个容易漏的loss：约束一致性

对被约束的目标关节，除了最后硬替换，还要在**硬替换之前**的网络原始输出上算（位置/旋转分量各自独立，与§5.0契约一致）：

$$\mathcal L_\text{ctrl}=\underbrace{\sum_{j:\,\texttt{ctrl\_pos\_mask}_j=1}\big\|\tilde{\boldsymbol p}_j^{\text{pre}}-\boldsymbol p_j^{\text{ctrl}}\big\|_1}_{\text{位置}}\;+\;\underbrace{\sum_{j:\,\texttt{ctrl\_rot\_mask}_j=1} d_{\text{geo}}\big(\Delta\tilde R_j^{\text{pre}},\ \Delta R_j^{\text{ctrl}}\big)}_{\text{旋转（测地距离）}}$$

否则网络会学到"反正会被钉死，这个关节随便输出"，导致被约束关节与周围关节在clamp边界处出现**局部撕裂**。

---

# 6. 训练配置

## 6.1 数据采样

实测数据现状（BONES-SEED）：

| 项目 | 值 |
|---|---|
| motion总数 | 142,220（71,132原始 + 71,088镜像） |
| 总帧数 @120fps | 124,554,605 |
| 每条motion帧数 | min 69 / 中位 684 / 均值 875 / max 21,617 |
| 演员数 | 522 |
| package分布 | Locomotion 74,488(52%) / Communication 21,493 / Interactions 14,643 / Dances 11,006 / Gaming 8,700 / Everyday 5,816 / Sport 3,993 / Other 2,081 |
| body_position分布 | **standing 122,548(86%)** / sitting on floor 4,589 / crouching 2,590 / standing,leaning 2,512 / crawling 1,670 |

三条采样规则：
1. **去镜像**：71,088条镜像是原始的精确镜像。基础池只取71,132条原始motion，镜像仅在训练侧做增广。**绝不能让一条motion和它的镜像分处train/val**——等价于标签泄漏。切分必须**按原始motion group**做（`is_mirror=False`的那条作为group key），不是按行切。
   **另外还要防同源重复版本泄漏**（Codex P1 #12）：metadata里同一 `take_name`/`content_name` 下常有多个变体（不同速度、不同风格的同一动作）。这些**必须整组一起分到同一侧**，否则"held-out演员"里会混进训练集见过的同一段动作。切分键建议用 `(actor_uid, take_name)` 的联合。
2. **时间去冗余**：120fps下相邻帧几乎相同。**每30帧取1**（0.25秒）→ 原始侧约2.08M帧可用池。**但起始offset必须per-motion随机**（$\text{offset}\sim\mathcal U\{0,\dots,29\}$，按motion_id做种子）——固定从第0帧开始会锁定相位，让所有采样点落在动作周期的同一阶段（Codex P1 #12说得对）。
3. **分层过采样**：naive采样下86%是standing、52%是Locomotion，模型基本见不到crouching/crawling/sitting。按 `(package × body_position)` 联合分层，对稀有类过采样，每个bucket至少占1-2%。**这条对SE3精度影响很大**——蹲/坐/爬恰恰是关节角分布最极端、retargeting最易失败的区域。

## 6.2 四重切分

| Split | 构造 | 检验 |
|---|---|---|
| `train` | ~450演员 × 分层采样帧 + ~55个动物物种 | — |
| `val_id` | 同训练演员同motion，held-out帧 | **仅pipeline sanity**，不产生任何泛化结论（Codex P1 #12） |
| `val_actor` | **完全held-out的~50演员**（含极端身材：199cm/145cm/145kg各留几个） | 骨长/体型泛化 |
| `val_topo` | **held-out的~10个动物物种** + 1个合成剪枝拓扑（SOMA-24） | **拓扑零样本泛化——核心卖点** |

动物**按物种切分**，不能按motion切（SAMoR用的是character-level切分）。同物种其他motion泄漏进训练集，`val_topo`就失去意义。

### 6.2b `val_topo` 到底测什么（v6重写，回答#7——这是概念上最容易自欺的地方）

> **v5的问题（Codex P0 #1/P1 #13）**：v5把 `val_topo` 说成"拓扑零样本泛化——核心卖点"，但held-out动物**没有跨拓扑配对GT**。在它们身上能算的只有自重建，而**自重建低误差只证明"能自编码一个新拓扑"，不证明"能把动作翻译到新拓扑"**。把前者说成后者是自欺。

**`val_topo` 拆成三个语义不同的子指标，分别报告，不许混为一谈**：

| 子指标 | 在什么数据上 | 有GT? | 能证明什么 | **不能**证明什么 |
|---|---|---|---|---|
| **T-a 未见拓扑自重建** | held-out ~10动物物种，X→X | ✓（恒等） | 编码器能处理未见拓扑、未见关节数 | **不证明翻译能力** |
| **T-b 未见拓扑cycle一致性** | 动物$s_1$→$s_2$→$s_1$ | ✗（只有闭合约束） | 瓶颈的拓扑无关性 | **不证明中间那步$s_2$的姿态正确** |
| **T-c 未见剪枝拓扑翻译** | SOMA-78 → **held-out的SOMA-24**（训练时只见过38/20） | ✓ **无歧义**（GT=原关节位置子集） | **真正的跨拓扑翻译零样本泛化** | — |

**T-c 是唯一能支撑"翻译"这个词的指标**，因为它的GT无歧义（剪枝是确定性子集选取，不是retargeter产物）。**报告的核心结论必须建立在T-c上**，T-a/T-b作为辅助证据。

### T-c 的退化风险与必须加的对照（v7新增，Codex第二轮指出）

**问题**：剪枝拓扑的关节是源骨架关节的**精确子集**，关节名也完全相同。配合SigLIP的同名embedding，模型可能学到一条捷径——**"按名字把已存在的坐标复制过去"**。这样T-c误差会很低，但它证明的只是"能处理变长$J$的copy"，**不是任意跨拓扑翻译**。

**必须加的三项对照**，缺一不可：

| 对照 | 做法 | 若模型在走捷径会怎样 |
|---|---|---|
| **C-1 匿名化关节名** | T-c评估时**100% name-drop**（关节名embedding全部置零）+ 一版用随机打乱的假名 | 捷径依赖名字匹配 → 误差**显著上升** |
| **C-2 形变后的目标** | 目标骨架在剪枝基础上再做**held-out的骨长/比例变换**（例如把小腿拉长20%、肩宽缩窄15%），GT用变换后的FK重算 | 捷径是复制原坐标 → 在变形目标上**系统性偏移** |
| **C-3 语义copy基线** | 一个不学习的基线：**按关节名直接复制源坐标**（名字对不上的取父关节位置） | 若模型打不过这个基线，说明它**就是**在做copy |

**判据**：只有当模型在 C-1（无名字）和 C-2（变形目标）下**仍显著优于 C-3 基线**时，T-c才支撑"跨拓扑翻译"的结论。否则只能报"变长$J$的坐标复制"。

**这一条同时补上了最近邻基线的作用**（§7"必须有的非学习基线"）——C-3是它的特化版本，专门针对copy捷径。

**这也修掉了v5的一处文档冲突**：`02_stage1_project_plan.md` 承认Go2无配对GT，本文档v5却要求报告Go2末端误差。**修正：零样本新机器人测试（指标6b）改为只报"可行性筛查"**——限位违反率、骨长守恒、$\mathcal L_\text{axis}$ 残差、以及是否产生NaN/退化解。**不报末端位置误差**（没有GT可比）。

`g1_23dof` 作为"未见机器人"的说服力有限（Codex P1 #13说得对，它是G1同家族裁剪）。**改为两级**：`g1_23dof` 作为**同家族拓扑变化**的快速筛查；**Go2四足**作为**真正的跨家族**筛查，但只报可行性、不报精度。

## 6.3 数据量分级

| 阶段 | 帧数 | 目的 | 单变体耗时 |
|---|---|---|---|
| T0 冒烟 | 5k | 跑通pipeline、shape检查、loss下降 | <10分钟 |
| **T1 初见端倪** | **200k** | **区分七个变体的相对优劣** | 0.5-1小时 |
| T2 定论 | 1-2M | 确认T1结论、给最终数字 | 4-8小时 |

**T1取200k的理由**：单帧pose→pose是低内在维度的回归问题（人体姿态流形有效维度约50-80），不是需要海量数据的生成任务；200k帧在分层采样下能覆盖各pose类别各约2-4k样本，足以让差异显现。

**先跑T1，看七个变体的排序跨3个seed是否稳定，稳定了再上T2。不要一上来就烧1-2M。**

## 6.3b 网络与训练超参（具体数值，直接用）

之前这一节缺具体数字，Codex没法照着写代码。以下是起点值，**不是需要消融的量**——除了标注了"待扫"的。目标是让七个变体在完全相同的骨干配置下比较，所以这些数字取什么不那么关键，**关键是七个变体必须完全一致**。

```python
# ---- 骨干（七个变体逐字节相同）----
D            = 256        # model dim
n_heads      = 8
L_enc        = 6          # encoder graph-transformer 层数
L_dec        = 6          # decoder IPA 层数
N_recycle    = 2          # 迭代精修轮数（待扫: {1,2,3}，只在胜出变体上扫）
ffn_mult     = 4
dropout      = 0.1
K            = 8          # part token 数（待扫: {4,8,16}，只在胜出变体上扫）
max_hop      = 8          # hop-distance bias 截断值（Floyd-Warshall后clamp）

# ---- 瓶颈：按 §3.3 的率失真分析，每个变体跑 4 个码率点 ----
# 12 bits 的理论下界末端误差已达 14.2mm, 单点比较会误导. 扫码率, 画曲线.
RATE_POINTS = [12, 16, 20, 24]     # bits/token
# 各变体在每个码率点的配置:
#   VQVAE: V=4096(12) / V=65536(16, 已难训) / >=20 bits 单码本不现实 -> 记为 N/A
#   RVQ  : R=2,V=64(12) / R=2,V=256(16) / R=2,V=1024(20) / R=2,V=4096(24)
#   FSQ  : [4]*6(12) / [4]*8(16) / [8,8,8,8,4,4,4,4](20) / [8]*8(24)
# 以上 12/16/20/24 四点均为精确值 (log2 of level-product), 无近似.
# VQVAE 在 >=20 bits 出局是一个 PREDICTION, 不是排除 —— 见 §3.3
vq_gamma     = 0.25       # encoder侧 commitment 权重
vq_ema_decay = 0.99       # 码本用 EMA 更新(不再同时用码本梯度loss)
beta_KL_max  = 1e-3       # VAE/CVAE，warmup 到此值
beta_warmup  = 10000      # step，0 线性升到 beta_KL_max

# ---- v6新增的 loss 权重 ----
w_limit      = 1.0        # 关节限位违反
w_axis       = 0.1        # 单轴投影残差
w_adv        = 0.0        # 具身判别器: T0/T1初期为0, Truebones接入后 -> 0.1
recycle_detach = True     # 中间 recycle 轮不回传梯度(AlphaFold式)

# ---- 优化 ----
optimizer    = AdamW(lr=3e-4, betas=(0.9,0.95), weight_decay=0.01)
scheduler    = cosine with linear warmup 2000 steps
grad_clip    = 1.0
batch_size   = 256
precision    = bf16
T1_steps     = 60000
seeds        = [0, 1, 2]

# ---- SigLIP ----
siglip_dropout = 0.15     # 训练时随机把关节名 embedding 置零的概率
# 冻结，不训练。只加载文本塔。
```

**参数量估算**：$D{=}256$、12层、8头，约 **10–15M 参数**。这与你此前"动作的本征维度不会比LLM/CV高"的判断一致——不需要大模型。单卡RTX PRO 6000（97.9GB）下batch 256绰绰有余，瓶颈在数据IO不在显存。

**唯一需要注意的显存项**：pairwise/hop-bias 是 $O(J^2)$。$J{=}78$ 时 $78^2{=}6084$，可忽略；但如果未来把手指全展开到更精细的骨架、或接入Truebones里关节数达71的物种，仍在同量级，安全。

## 6.3c $\mathcal L_\text{attn}$ 的精确定义（之前只提了名字没给公式）

attention supervision 的作用是防止$K$个part query塌缩成"都关注全身"的冗余表示（SAMoR原文：naive训练下会"reducing K=8 to the effective equivalent of K=1"）。

给定part-pool的注意力矩阵 $A\in\mathbb R^{K\times J}$（$\alpha_{k,j}$ 为query $k$ 对关节 $j$ 的权重）和每个关节的功能组标签 $y_j\in\{0,\dots,K-1\}\cup\{-1\}$：

> **v5的错误（Codex P1 #9，说得对）**：v5的 $A=\mathrm{softmax}(\cdot,\ \dim=\text{joint})$ 使**每个query在所有joint上和为1**，然后却对 $-\log A[y_j,j]$ 声称是"按joint的交叉熵"。这两件事不兼容：同一功能组有多个joint时，它们**不可能同时**让 $\alpha_{y_j,j}\to 1$（因为该query的权重总和被约束为1），loss会系统性偏向关节数少的组。

**修正：用一个独立的、沿 $K$ 维归一化的分配头**，与pooling用的attention分开：

$$P_{k,j}=\frac{\exp(s_{k,j})}{\sum_{k'=1}^{K}\exp(s_{k',j})}\quad(\text{沿}\,K\,\text{归一化}),\qquad \mathcal L_\text{attn}=-\frac{1}{|\mathcal L|}\sum_{j\in\mathcal L}\log P_{y_j,\,j},\quad \mathcal L=\{j:y_j\ge 0\}$$

其中 $s_{k,j}=(W_Q q_k)^\top(W_K x_j)/\sqrt d$ **与pooling共享打分**，但**归一化方向不同**：
- **pooling** 沿 joint 归一化（每个query聚合哪些关节）——保持SAMoR原样；
- **supervision** 沿 $K$ 归一化（每个关节归属哪个slot）——这才是"按joint的交叉熵"。

这样每个关节独立地被推向自己的功能组，**与组内关节数无关**，消除了小组偏置。

**v5的第二个错误（同样接受）**：human的slot 6/7 是左右手、animal的slot 6/7 是尾/其他——**同一个共享query被要求学两种语义，破坏跨具身一致性**。

**修正：改用一套跨物种统一的功能组定义**，按"运动学角色"而非"人体部位名"划分：

| slot | 统一语义 | 人形 | 四足/鸟 |
|---|---|---|---|
| 0 | 躯干/脊柱核心 | Hips, Spine*, Chest | 脊柱各节 |
| 1 | 头颈 | Neck*, Head | 头颈 |
| 2 | 前肢-左 | 左臂全链（含手） | 左前肢 |
| 3 | 前肢-右 | 右臂全链（含手） | 右前肢 |
| 4 | 后肢-左 | 左腿全链（含脚） | 左后肢 |
| 5 | 后肢-右 | 右腿全链（含脚） | 右后肢 |
| 6 | 尾/尾端附属 | （人形无，全部$y_j=-1$） | 尾 |
| 7 | 其他附属 | 锁骨等 | 翅膀/鳍/触角等 |

**关键点：人形和动物用同一张表**，人形只是在slot 6上没有关节（全部标 $y_j=-1$）。这就是SAMoR兜底机制的正确用法——**缺失的组留空，而不是把它挪去表示别的东西**。

**$y_j=-1$ 的关节完全不进这个求和**——对loss贡献为零，但仍然被编码、由学到的注意力自由路由。这是SAMoR的兜底机制，也是动物骨架标不全仍能work的原因。若某个具身**所有**关节都是$-1$（无规范关节名的机械rig），$K$个slot就退化为纯无监督功能桶，只靠运动学结构驱动。

**功能组划分见§4.4的统一表**（人形与动物共用一张表，人形在slot 6上全部 $y_j=-1$）。标不准的关节一律给 $-1$。

## 6.4 训练参数

- **以step为准不以epoch为准**（不同数据量下epoch含义不同）。T1：batch 256，目标 **60k step**（200k帧≈780 step/epoch → 约77 epoch）。
- **七变体 × 3个seed = 21次训练**。不跑多seed就无法判断差异是真实的还是随机波动。T1单次0.5-1小时 → 全部11-21小时。
- 早停/选点：**按`val_topo`的末端位置误差选best checkpoint**，不用`val_id`（那会偏向过拟合的变体）。
- GPU已被占用24.5GB（总97.9GB），开跑前需协调。

---

# 7. 评估指标

## 维度A：SE3精确控制（在有GT的SOMA→G1上）

1. **末端关节位置误差（mm）**——只统计手腕/脚踝/骨盆/头（若目标有），不是全关节平均。
2. **旋转测地距离误差（度）**——canonical frame朝向与GT的测地距离。
3. **约束满足误差**——应≈0（硬替换）。非0即实现有bug，属正确性检查不是性能指标。
4. **约束邻域连续性**——被约束关节的相邻关节误差是否显著高于全局平均（检验clamp是否造成局部撕裂）。
5. **关节限位违反率**——FK后检查是否超出URDF定义的限位。这是能否下发到真机的硬门槛。
6. **Twist DOF专项误差**——**单独报告** `left/right_ankle_roll`、`left/right_wrist_roll` 这4个从位置完全观测不到的DOF。它们在29维平均里只占4/29，**完全学错也会被平均值掩盖**，但`ankle_roll`决定足部接触姿态。**这一项同时验证了canonical frame输出的必要性**：若模型只输出位置，这4项误差必然接近随机。

6b. **零样本新机器人测试（scope的直接验收）**——取训练时完全没出现的机器人URDF，**不做任何训练/微调/加参数**，直接跑推理。**只报可行性筛查，不报末端位置误差**（没有配对GT可比——这修掉了v5与plan文档的冲突）：
   - 关节限位违反率、骨长守恒误差、$\mathcal L_\text{axis}$ 投影残差、是否出现NaN或退化解（所有关节塌到一点）。
   - **两级**：`g1_23dof.urdf` = 同家族拓扑变化的快速筛查；**Go2四足** = 真正的跨家族筛查。
   
   **这一项不过（出NaN或退化），就说明模型没做到universal，而是学成了一堆专用映射。** 但它过了也只说明"没崩"，不说明"翻译对"——精度结论必须来自§6.2b的T-c。

## 维度B：跨拓扑翻译质量（重点，在`val_topo`上）

7. **零样本拓扑指标——按§6.2b拆成三项分别报告**：T-a 未见物种自重建、T-b cycle一致性、**T-c 未见剪枝拓扑翻译（唯一能支撑"翻译"一词的项，核心结论必须建立在它上面）**。
8. **Swap-consistency**——$z\to$decode到拓扑B$\to$重新encode$\to$与原$z$的距离。**定义必须统一**（Codex P1 #13）：$z$在七个变体里形状不同（$J\times D$ / $K\times D$ / 量化索引），余弦距离无法直接跨变体比较。**统一改为在解码输出空间比较**：$\text{err}=\|\text{decode}_A(\text{encode}_B(\text{decode}_B(\text{encode}_A(x))))-x\|$，即**round-trip重建误差（单位mm）**，对所有变体同定义、同单位。**CVAE的$z$是target-conditioned的**，round-trip定义对它同样成立（只是每步要传对应的target），这也绕开了余弦距离在CVAE上没有意义的问题。
9. **骨长泛化曲线**——按`val_actor`身高分箱（145-160/160-175/175-199cm），看误差随体型偏离训练分布的增长速度。

## 维度C：瓶颈本身的性质

10. **帧间瓶颈跳变**——连续两帧近乎相同的输入（同一pose加极小噪声），测$\|z_t-z_{t-1}\|$和输出差异。这是VQ/FSQ系量化边界跳变风险的直接测量，AE/VAE/Direct-Token应显著更平滑。**这一项直接预测了时序阶段会不会抖。**
11. **码本利用率**（仅VQ/RVQ）——实际用到的码字比例，检测collapse。
12. **Part query塌缩检测**（仅有pooling的变体2-7）——SAMoR明确指出naive训练下$K$个query会"attend broadly and produce redundant representations, reducing K=8 to the effective equivalent of K=1"。测$K\times J$注意力矩阵$A$的各query分布两两余弦相似度、以及每个query的注意力熵。**若普遍塌缩，该变体与Direct-Token的对比失去意义**，需加attention supervision重跑。

## 必须有的非学习基线

没有基线的话，七个变体互相比得再漂亮也回答不了"这套东西有没有超过现成方法"：
- **解析retargeting基线**：现成IK retargeter（GMR或SOMA Retargeter官方pipeline）在同一测试集上的误差；
- **最近邻基线**：训练集里找最相似的源pose，直接返回它对应的目标pose。

**如果七个变体都打不过最近邻，说明模型没学到泛化、只是在记忆。**

## 一个诚实的前提

SOMA→G1的"GT"本身是NVIDIA用IK retargeter生成的，不是物理真值。**本轮实质上是在蒸馏一个retargeter。** 这意味着：(a) 在SOMA→G1上"超过GT"没有意义；(b) 真正的价值在`val_topo`——学习式方法能否泛化到那个retargeter根本没配置过的新拓扑。**评估权重应向维度B倾斜。**

---

# 8. 交付物

1. 七个变体 × 3 seed 的 checkpoint + 训练日志
2. **13项指标**的对比表（§7的1–12含6b；含参数量、瓶颈信息容量两列）+ 两个非学习基线
3. 结论：哪个瓶颈胜出，在哪个维度上胜出，是否稳定跨seed
4. `viser` 可视化demo（见文件3的§4）

**报告时的表述范围限定**：本轮结论只在"SOMA + G1 + 剪枝变体 + Truebones 65物种"这个拓扑集合上成立。SAMoR有约71k种拓扑，我们约70种，**不要外推成"any humanoid"或"any skeleton"**。