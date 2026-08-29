# 架构定案：T型时空解耦的跨具身动作生成模型

本文档汇总本会话经过多轮拷问后敲定的架构设计。与更早的`math_framework.md`（跨形态约束/多模态条件/流形分布的数学推导）是互补关系：`math_framework.md`里关于flow matching、Riemannian分布、mask采样、多模态条件注入的推导，在下面的"空间层"和"时间层"两处仍然适用，只是具体的编解码器结构由本次讨论的T型架构取代了此前第5节里更简单的"IK-based/学习式适配器"方案。

---

## 一、整体结构：T型解耦

- **垂直一竖——空间层（Universal Latent Mapper）**：逐帧、跨具身共享的编解码器。职责：把任意人形拓扑（SMPL-X/G1/Optimus/Figure/任意新构型）的关节状态压缩到统一的latent，并能从稀疏/残缺的末端约束脑补出全身姿态，解码到任意目标构型的SE(3)关节位姿。**不假设隐空间服从正态分布**（撤回此前"VAE vs AE"的提法——这一层的角色是确定性、不预设分布形状的双向latent映射，生成能力留给时间层或未来另外叠加的flow matching生成器）。关掉时间层即可独立作为遥操作器使用。
- **水平一横——时间层（Temporal Sliding Transformer）**：causal、sliding-window自回归，在空间层的latent $z$ 上（而非decode后的具体关节值——见下方"已敲定"第1条）逐帧预测演化，支持prompt条件与全局SE3约束注入。

---

## 二、空间层内部设计（已敲定）

### 1. Token化
- **Joints-as-tokens**：每个关节独立token，跨构型用最大关节数padding+mask对齐（如G1 23个，Optimus 45个）。
- **语义层对齐**：每个token注入轻量语义one-hot（"左手/右臂/左膝/右腿"等），消除不同机器人命名体系的鸿沟。
- **拓扑与限制硬编码**：token内嵌kinematic chain父子关系 + 该电机的物理极限（转角/扭矩限制）。

### 2. Encoder：几何双轨注意力（AlphaFold pair-representation式）
- **单体轨**：全连接双向自注意力，让强控制点（如指定的右手3D坐标）在空间中双向扩散，指导未指定（masked）关节的脑补。
- **配对轨**：计算 $J\times J$ 全对距离/内积矩阵，把目标构型标准状态下的骨长/连杆几何作为bias注入注意力矩阵，从底层赋予刚体不变性约束。这一半设计被保留（pairwise track本身是编码相对结构的有效方式）。

### 3. Decoder：frame-based IPA，而非纯距离反解
- **手性歧义问题**：纯粹的pairwise距离/内积信息无法唯一确定手性（镜像对称姿态的所有pairwise距离不变但姿态错误）——AlphaFold2早期也遇到并解决过这个问题。**结论：decode阶段不采用triangle inequality直接反解坐标**,改为采用Invariant Point Attention（IPA）+ 显式刚体frame表示：
  - 每个joint token维护一个局部frame $T_j=(R_j,\boldsymbol t_j)\in SE(3)$，初始化为该构型T-pose对应的frame。
  - 每层decoder：(a) IPA——用query token的frame把其他token坐标变换到自己局部坐标系下做attention（对全局刚体变换天然不变）；(b) 输出frame更新量 $(\Delta R_j,\Delta\boldsymbol t_j)$作用在当前frame上；(c) 若干轮迭代/recycling后取最终frame的平移分量为该关节SE3位置。
  - 手性歧义靠"锚定参考frame"这一显式几何操作消除，不依赖网络从对称距离矩阵里自行猜对方向。
- **被约束关节的处理**：Is_Controlled=1的token，在每轮迭代后直接把该frame钉死为约束值（Query钉死机制的frame版本）。

### 4. 显式解析层与不可达报警（保留自用户前一版设计，措辞纠正）
- Decoder输出3D空间坐标/frame，送入解析动力学层（Jacobian/几何重投影），强制映射到硬件DoF可行域内。
- 纠正表述：不应称为"100%物理一致性"——准确表述是"给定可行输入时保证输出可行；输入超出可行域时，通过残差阈值判定确定性报错（GeometryUnreachableError）"，二者是不同的保证，混用会掩盖报错阈值和恢复机制未设计清楚的问题。
- 若此解析层不可微，需明确：手部接触/足部防滑/质心平滑三个物理loss只能监督到解析层输出，无法通过它反传梯度到网络本身——这决定了训练闭环是否完整，需要在实现时验证解析层的可微性或改用可微松弛近似。

### 5. Soft prior：帧间连续性的warm-start机制
避免逐帧独立求解IK多解圆导致的帧间跳变，且不破坏"关掉时间层即为纯遥操作器"的性质：
- 把上一帧对应关节的实际解（frame $T_j^{(t-1)}$）作为额外conditioning token拼入当前帧decoder的Query，用可学习门控 $\lambda$（或让attention自行学习权重）决定采纳强度。
- 时间层关闭时，该项置零/丢弃，退化为纯逐帧独立求解——遥操作模式不受影响。
- 解析层（Jacobian/IK求解器）本身若支持warm start，可同时把上一帧解作为初始猜测值传入，形成网络端+解析端双重warm-start。

### 6. 迭代精修（recycling）应对latent-to-SE3的保真度风险
用户提出的担忧成立：若只做一次前向decode，控制精度完全依赖$z\to$SE3映射的局部保真度，在IK多解切换边界附近可能出现"latent平滑但decode后关节轨迹跳变"。采用AlphaFold式迭代精修：decode一版SE3位姿→重新编码回$z$→再decode，重复2-3轮直至收敛，用显式精度指标验收（如末端位置误差<1cm、旋转误差<2°，具体阈值待消融）。

---

## 三、时间层设计（已敲定）

1. **自回归状态选择：在latent $z$ 空间里滑动，而非decode后的具体关节值。** 这是本次讨论中最关键的分叉点之一——若时间层直接吃decode后的actual joint state，会导致要么每个目标构型单独训一个时间transformer（违背universal初衷），要么每步都要重新过一遍encoder换回$z$（推理开销翻倍）。确定使用$z$：时间transformer输出下一帧的$z_{t+1}$，全程留在canonical空间，只在最终需要具体构型输出时才过一次空间decoder。
2. Causal transformer + sliding window（步长待消融），支持prompt条件注入与全局SE3约束（根节点/pelvis整段轨迹约束，作用域跨帧，注入为时间层的条件token，区别于空间层内单帧的稀疏末端约束）。
3. 实时输出动作序列；关闭时间层即整个系统退化为逐帧空间层驱动的遥操作器。

---

## 四、训练信号设计（已敲定，尚未在此前版本讨论过）

跨具身的"换脸VAE"类比里最容易被忽略的部分不是decoder的cross-attention结构，而是训练信号本身如何覆盖没有配对数据的目标构型：

- **有配对数据的一侧**（目前仅SOMA→G1，NVIDIA官方唯一发布）：直接监督，关节角L2或测地距离loss。
- **无配对数据的一侧**（Optimus/Figure/任意新构型）：采用**swap-consistency loss**——源动作的$z$解码到目标构型B得到动作，再把这个B动作重新过一遍空间encoder，要求恢复出的动作语义（如末端相对轨迹形状）与原始一致；这是无监督信号,不需要B的ground truth关节角，可以覆盖任意没有配对数据的新构型。
- 两侧同时叠加物理可行性loss（关节限位、不可达报警残差）兜底。
- 三大物理loss（手部接触/足部防滑/质心平滑）沿用用户原设计，但需先确认解析层可微性（见二.4）。

---

## 五、与`math_framework.md`的分工

- 本文档：空间层/时间层的具体网络结构（token化、双轨attention、IPA decode、soft prior、recycling、swap-consistency训练信号）。
- `math_framework.md`：多模态条件融合（约束/文本/环境感知的cross-attention注入）、约束掩码的联合采样分布（spatial稀疏-temporal连续 vs Kimemo型 vs OmniControl型）、旋转通道的Riemannian flow matching替代方案、少步实时蒸馏路径、长期closed-loop PD输出的exposure bias处理（ReactiveBFM式prefix sampling curriculum）——这些设计在时间层里注入的方式不变，只是时间层自回归的对象现在明确为空间层输出的$z$，而不是原始关节角。

---

## 六、待消融、非现在能拍板的数字（不是遗留的架构分歧，而是超参数搜索问题）

1. **Recycling迭代轮数上限及对应精度验收阈值**（如末端误差<1cm、<2轮 vs <0.5cm、<3轮的取舍）。
2. **遥操作模式下单帧延迟预算**（毫秒数），并由此反推pairwise track能支持的最大关节token数 $J$、encoder/decoder层数与维度。
3. Sliding window步长、causal transformer的attention窗口大小。

这三项的共同特点是：架构形状不再依赖它们的取值（即无论最终数字是多少，上面二、三节的结构不需要改变），只影响层数/维度/轮数这些可以通过实验曲线直接读出的量——因此推迟到有可跑的pipeline、能测出延迟-精度曲线之后再定，是合理的顺序，不是回避设计决策。
