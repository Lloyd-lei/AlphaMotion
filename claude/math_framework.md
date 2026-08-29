# 跨形态多模态动作生成模型：完整数学表示推导

**目标系统**：一个以NVIDIA SOMA-X为统一人形骨架接口，接受关节时序约束（spatial稀疏/temporal连续）+ 文本prompt + 环境感知（点云/视频/任意3D-aware模态）三种条件，输出可适配任意人形构型（SMPL-X人体、Unitree G1、自定义机器人）的动作序列，早期作为WBC模仿学习的teacher生成器，长期直接输出causal PD/控制primitive、实现感知-规划-执行端到端小脑的生成模型。

事实核查结论先行：SOMA-X（`NVlabs/SOMA-X`）确实是NVIDIA刚发布的统一骨架框架，把SMPL/SMPL-X/MHR/ANNY等身份表征统一映射到单一canonical mesh topology + 骨架；但**SOMA Retargeter目前只发布了G1一个机器人的输出端**（"Currently supports SOMA as the input skeleton and Unitree G1 (29 DOF) as the output robot. Additional robot targets are planned"），Kimodo和ARDY也都是按骨架训练独立的模型变体（Core-RP人体 / G1-RP机器人）而非单一跨形态解码器。这意味着"一套编解码器原生输出任意人形"目前是真正的开放问题，第五、六节会给出具体解法。

---

## 0. 记号与统一骨架接口

设SOMA canonical骨架有关节集合 $\mathcal{J}=\{1,\dots,J\}$，其运动学树由父指针 $\mathrm{pa}(j)$ 给出。每个具体形态（embodiment）$e\in\mathcal{E}$（人体SMPL-X、G1、任意humanoid）拥有：

- 骨长/偏移向量 $\boldsymbol{\ell}_e\in\mathbb{R}^{J}$（或每骨骼的3D offset）
- 每关节自由度与关节限位 $\{d_j^{(e)}, [\theta_j^{\min,(e)},\theta_j^{\max,(e)}]\}_{j\in\mathcal{J}}$
- 前向动力学映射 $\mathrm{FK}_e:(\,\boldsymbol{p}_{\text{root}},R_{\text{root}},\{\theta_j\}_{j\in\mathcal{J}}\,)\mapsto \{(\boldsymbol{p}_j,R_j)\}_{j\in\mathcal{J}}$，把根位姿+关节角映射到世界坐标下每个关节的位置与朝向

**跨形态统一的核心设计原则**：生成模型的去噪/流匹配主干**只在SOMA canonical空间中运算**，与具体机器人形态无关；每个embodiment $e$ 通过一个轻量适配器 $D_e$（第五节详述）把canonical轨迹映射到该形态的原生关节角。这样一来，"支持新机器人"变成"写一个新的 $D_e$"，而不需要重新训练生成主干——这正是SOMA/Kimodo生态目前唯一已经工程化的部分（G1的 $D_{G1}$），其余机器人的 $D_e$ 需要你自己按同样的多目标IK+关节限位+落地点稳定的配方去实现，或者用第五节的可学习蒸馏方案替代。

---

## 1. 动作序列编码：末端关节代表性假设的数学刻画

### 1.1 你的假设

设末端关节集合 $\mathcal{K}=\{\text{头/眼}, \text{左右手腕}, \text{盆骨(root)}, \text{左右脚跟}, \text{指尖}\}\subset\mathcal{J}$，中间关节集合 $\mathcal{M}=\mathcal{J}\setminus\mathcal{K}=\{\text{肘},\text{膝},\text{肩},\dots\}$。你的假设是：只编码 $\mathcal{K}$ 上每个关节的位置、速度、6D旋转，$\mathcal{M}$ 不出现在被编码的序列里，指望模型在推理时从末端关节"联想"出中间关节——即把 $\mathcal{M}$ 当作一种**天然的、结构化的MAE掩码**。

### 1.2 这在几何上到底丢失了多少信息

对运动学树上任意一条从根 $r$ 到末端 $k\in\mathcal{K}$ 的链路 $r\to m_1\to m_2\to\cdots\to m_n\to k$（$m_i\in\mathcal{M}$ 为链上的中间关节，各自有旋转自由度 $d_{m_i}$），给定链路两端的完整位姿（根位姿 $6$ 维 + 末端位姿 $6$ 维，共作为边界条件），中间关节角所构成的解空间维度为

$$
r_{\text{链}} \;=\; \Big(\sum_{i=1}^n d_{m_i}\Big) \;-\; 6
$$

（若只约束末端**位置**而不约束末端朝向，则减去 $3$ 而非 $6$）。这就是**逆向运动学冗余度**：例如经典的人体手臂链路 肩(3 DOF 球关节)–肘(1 DOF 铰链)–腕(3 DOF)，若只给定肩位姿与腕位姿（共12维约束，扣掉腕自身姿态就是给定肩到腕的6维相对位姿），肘位置并非唯一解，而是落在一个**圆**上——这是解析IK里经典的"双球面相交"结果：给定肩点 $\boldsymbol p_s$、腕点 $\boldsymbol p_w$、上臂长 $\ell_1$、前臂长 $\ell_2$，肘点满足

$$
\boldsymbol p_{\text{elbow}}(\varphi) = \boldsymbol c + \rho\,\big(\cos\varphi\,\boldsymbol u + \sin\varphi\,\boldsymbol v\big),\qquad \varphi\in[0,2\pi)
$$

其中 $\boldsymbol c,\rho,\boldsymbol u,\boldsymbol v$ 由 $\boldsymbol p_s,\boldsymbol p_w,\ell_1,\ell_2$ 解出（标准两球面相交公式），$\varphi$ 即所谓"肘摆角/swing angle"。这与AvatarPoser论文中提到的问题完全一致——"末端关节位姿沿运动学链传播误差"正是这个冗余方向没被约束导致的：AvatarPoser、AGRoL、SparsePoser等一整类"稀疏traker→全身姿态"文献处理的正是这个反问题，它们的共同结论是**必须靠学到的人体运动先验或显式IK精修来消掉这个冗余**，而不是没有代价的。

**结论**：你的直觉方向是对的（这类反问题在文献里成立且有效），但它不是严格意义上"无损"的MAE掩码，而是**每条链每帧丢失恰好 $r_{\text{链}}$ 维信息**，这部分信息必须由（a）跨帧的时间平滑先验，（b）文本/场景条件，或（c）模型自己学到的隐式姿态先验联合承担。数学上，这相当于把生成模型的条件分布写成

$$
p\big(\{\theta_m\}_{m\in\mathcal M_{\text{链}}} \,\big|\, \boldsymbol x_{\mathcal K}\big) \;=\; \int p\big(\{\theta_m\}\,\big|\,\varphi\big)\,p(\varphi\mid \boldsymbol x_{\mathcal K,\,1:T}, c)\,d\varphi
$$

即模型需要隐式地为每条冗余链学一个"摆角"隐变量 $\varphi$（或等价的flow-matching隐码），并以完整的时序上下文与多模态条件 $c$ 为其提供依据。**建议的工程化落地**：不要完全丢弃中间关节，而是把它们降级为一个**低维隐式token**（每条链一个标量或极低维embedding，而不是完整的6D旋转），在编码阶段仍然输入，但只占极小的通道预算——这等价于显式建模上面 $\varphi$ 这个redundancy维度，代价极小、但避免了训练不稳定或多解坍缩到不合理姿态的风险。肩关节稍有不同：它连接躯干与手臂，若也不显式编码，会额外丢失躯干朝向到手臂根部的相对朝向信息，建议肩、髋作为"次级末端"（sub-endpoint）保留位置+旋转，只把肘、膝这类**单自由度铰链关节**完全交给隐式推断——因为铰链关节的冗余度 $r_{\text{链}}$ 恰好最小（通常只有1维，对应上面的 $\varphi$），是这套假设成立代价最低的情形。

### 1.3 编码表示

对每个末端/次末端关节 $j\in\mathcal K$、时刻 $t$，定义token

$$
\boldsymbol z_{j,t} = \big[\,\boldsymbol p_{j,t}\in\mathbb R^3,\;\; \boldsymbol v_{j,t}=\boldsymbol p_{j,t}-\boldsymbol p_{j,t-1}\in\mathbb R^3,\;\; \boldsymbol r_{j,t}\in\mathbb R^6\,\big]\in\mathbb R^{12}
$$

其中 $\boldsymbol r_{j,t}$ 是Zhou et al. (2019) 的连续6D旋转表示（旋转矩阵去掉第三列，用Gram-Schmidt重新正交化解码），比四元数/欧拉角更适合神经网络回归，避免了四元数的双重覆盖和欧拉角的万向锁；根节点位置与朝向按HumanML3D/Kimodo惯例做root-relative canonicalization（以根节点朝向对齐坐标系、扣除全局平移与朝向漂移），减小生成目标的方差。中间关节链路的冗余度 $\varphi_{\text{链}}$ 作为一个额外的低维隐式token 拼接进同一序列，但**不作为约束通道暴露给用户**——用户只能约束 $\mathcal K$（第2节）。

---

## 2. 时空混合约束：spatial稀疏、temporal连续的条件机制

### 2.1 三种约束范式的统一掩码表示

定义一个三元掩码函数 $m:\mathcal J\times\{1,\dots,T\}\to\{0,1\}$，$m(j,t)=1$ 表示第 $j$ 个关节在第 $t$ 帧的位姿被作为硬约束提供。三种范式只是这个掩码在 $(j,t)$ 平面上的不同采样分布：

- **Kimodo型**（关键帧）：$m(j,t)=1 \iff t\in \mathcal T_{\text{key}}$（少数关键时刻），对**所有** $j$ 同时成立——时间稀疏、空间完整。
- **OmniControl型**：$m(j,t)=1$ 对任意 $(j,t)$ 独立采样——完全自由的联合稀疏。
- **你要的范式**：$m(j,t)=1 \iff j\in \mathcal J_{\text{ctrl}}\ (\text{少数指定关节})\ \text{且}\ t\in[t_1,t_2]$（一段连续时间区间）——空间稀疏、时间连续。

这三者是**同一个掩码空间上不同的采样分布**，因此可以用**单一模型**、只需要在训练时把mask采样分布设计得足够丰富（覆盖以上三种及其组合）即可让模型在推理时泛化到任意组合——这正是AnyBody（"a single latent motion representation any keypoint subset can address"，通过masked self-attention接受推理时任意选取的关键点子集）和MaskControl的做法。训练时对mask的采样建议使用分层采样：

$$
m \sim \alpha_1\cdot\mathrm{Unif}(\text{关键帧型}) + \alpha_2\cdot\mathrm{Unif}(\text{任意joint-time对}) + \alpha_3\cdot\mathrm{Unif}(\text{关节子集}\times\text{连续时间段})
$$

$\alpha_1+\alpha_2+\alpha_3=1$，以保证你要的"少关节、连续时间段"这一模式在训练分布中占有足够权重，而不是像Kimodo一样只在关键帧型上训练。

### 2.2 约束注入机制

对每个满足 $m(j,t)=1$ 的 $(j,t)$，生成一个约束token

$$
\boldsymbol e_{j,t} = \mathrm{Embed}_{\text{joint}}(j) + \mathrm{Embed}_{\text{time}}(t) + W_c\,\boldsymbol c_{j,t}
$$

其中 $\boldsymbol c_{j,t}$ 是提供的约束值（位置和/或6D旋转）。这些token构成一个**长度可变**的集合（不同推理请求里被约束的 $(j,t)$ 数量不同），因此不能像Kimodo那样简单地在fixed-size序列上做掩码拼接，而应采用OmniControl式的**ControlNet可训练分支**：主干去噪Transformer的每一层都设一个交叉注意力子层，Query取自主干的运动token，Key/Value取自约束token集合 $\{\boldsymbol e_{j,t}\}_{m(j,t)=1}$：

$$
\boldsymbol h_l \leftarrow \boldsymbol h_l + \mathrm{CrossAttn}\big(Q=\boldsymbol h_l,\ K=V=\{\boldsymbol e_{j,t}\}\big)
$$

### 2.3 硬约束满足：inpainting替换 vs 解析梯度引导

OmniControl用的"spatial guidance"是在每个采样步对预测的 $\hat{\boldsymbol x}_0$ 做一次梯度下降 $\nabla_{x_t}\|\mathrm{FK}(\hat{\boldsymbol x}_0)_{j,t}-\boldsymbol c_{j,t}\|^2$ 来强制满足约束，但官方文献自己承认这"could lead to longer inference times, especially when controlling multiple joints or in scenarios with dense spatial constraints"——与你要的实时性冲突。对于你的最终实时系统，建议改用**RePaint式硬替换（inpainting）**：每一步反向采样后，直接把约束位置的分量替换为真值对应加噪版本，

$$
\hat{\boldsymbol x}_{j,t}\big|_{m(j,t)=1} \leftarrow \boldsymbol c_{j,t}\quad(\text{加噪到对应时间步后替换})
$$

不需要额外梯度步、零延迟开销，代价是约束的"实现精度"略低于解析梯度引导（因为只在离散采样步上做替换）——但对少步实时采样（第6节）而言这是必要的取舍，且已经是MaskControl等最新工作从diffusion转向掩码模型时选的方案。

---

## 3. 多模态条件融合

三种条件模态：(a) 关节时序约束 $C_{\text{ctrl}}=\{\boldsymbol e_{j,t}\}$（第2节，变长集合，按(joint,time)对齐）；(b) 文本prompt，经冻结的LLM编码器（ARDY用LLM2Vec）得到token序列 $C_{\text{text}}$；(c) 环境感知（点云/视频/任意3D-aware模态），经模态专属tokenizer（点云用稀疏卷积/PointNet++，附3D位置编码；视频用时空ViT，附相机位姿编码）得到 $C_{\text{scene}}$，**这类token不与(joint,time)网格对齐**，应视为独立的空间集合。

三路条件各自通过独立的交叉注意力子层注入主干（Perceiver式多路条件，OMG的omni-modal设计即采用此结构同时接受language、audio、reference motion）：

$$
\boldsymbol h_l \leftarrow \boldsymbol h_l + \sum_{\text{modal}\in\{\text{ctrl,text,scene}\}} \lambda_{\text{modal}}\cdot \mathrm{CrossAttn}\big(Q=\boldsymbol h_l,\ K=V=C_{\text{modal}}\big)
$$

训练时对每个模态独立地做**classifier-free guidance式的模态dropout**（随机丢弃某个模态的全部token，替换为可学习的空token），使模型在任意模态子集（甚至零模态，纯无条件生成）下都能工作——OMG和MotionWAM都用这一训练技巧应对"部分模态缺失"的部署场景。推理时可对每个存在的模态分别做guidance scale加权：

$$
\hat{\boldsymbol v} = \boldsymbol v_\varnothing + \sum_{\text{modal}} w_{\text{modal}}\big(\boldsymbol v_{\text{modal}} - \boldsymbol v_\varnothing\big)
$$

（$\boldsymbol v$ 为流匹配速度场或diffusion的噪声/得分预测，$\boldsymbol v_\varnothing$为无条件预测）。

---

## 4. 分布假设：为什么高斯不是唯一选项，也不是最优选项

### 4.1 问题所在

你的动作token里位置/速度通道 $\boldsymbol p,\boldsymbol v\in\mathbb R^3$ 是欧氏的，高斯扰动天然合适；但旋转通道 $\boldsymbol r\in\mathbb R^6$（本质上表示 $SO(3)$）**不是**欧氏空间——多数现有工作（本次文献检索到的所有工作都不例外）直接对拉平的6D/9D向量加高斯噪声，这在几何上是错的：它会在Gram-Schmidt重新正交化的奇异点附近扭曲度量、并在旋转量较大时浪费模型容量去学习一个本不存在的欧氏结构。

### 4.2 已经存在的、更合适的替代分布/训练目标

这套工具最早在蛋白质结构生成（FrameDiff、RFdiffusion、FoldFlow）里被系统化，可以整套搬到动作生成里：

1. **IGSO(3)（$SO(3)$上的各向同性高斯）**：用群上的热核定义的各向同性分布替代欧氏高斯，作为diffusion的前向加噪过程（FrameDiff采用），有解析形式的密度和采样公式。
2. **Riemannian Flow Matching**（Chen & Lipman 2023；FoldFlow, Bose et al. 2023）：直接在 $SO(3)$ 流形上用测地线插值定义训练目标——给定基础分布采样 $r_0$（如 $SO(3)$ 上的均匀分布，或 $\mathrm{IGSO3}(\sigma)$ 先验）与数据 $r_1$，测地线插值 $r_t = r_0\exp[t\log(r_0^\top r_1)]$，训练切空间速度场 $u_\theta(r,t)$ 使其匹配

$$
\mathcal L(\theta) = \mathbb E_{r_0,r_1,t}\Big\| u_\theta(r_t,t) - \log(r_t^\top r_0)/t \Big\|_{SO(3)}^2
$$

这是**simulation-free**、训练稳定、且天然与欧氏flow matching共享同一套ODE采样框架——即位置通道用普通（欧氏）flow matching，旋转通道用Riemannian flow matching，两者共享同一个去噪Transformer输出的两路向量场，整体构成一个 $SE(3)^{|\mathcal K|}\times\mathbb R^{\dots}$ 积流形上的flow matching模型（这正是蛋白质骨架生成里"SE(3) Flow Matching把平移与旋转解耦处理"的配方，Yim et al. 2023）。这是本次会话检索到的动作生成方向论文中（含本轮和更早的文献调研阶段）都没有采用的设计，若你实现它，本身就构成一个方法学贡献。
3. **离散/类别分布替代方案**：把动作量化进VQ codebook（MuGen、MotionVLA、PHASOR的相位分解+运动语义蒸馏），完全绕开流形问题，用交叉熵训练类别分布——代价是量化误差，MotionVLA的双流频域tokenizer（姿态、速度分开量化）是一个相关的改良。
4. **其他流形分布**：von Mises-Fisher（单位球面，如用于单位四元数或骨骼方向向量）、Bingham分布（四元数双重覆盖下的反对称版本）——多用于需要显式似然（不仅是采样器）的场合。

**建议**：对end-effector集合 $\mathcal K$ 的 $(\boldsymbol p,\boldsymbol v)$ 用欧氏flow matching，对 $\boldsymbol r$（以及第1节提到的链路冗余隐变量 $\varphi$，它本身也是一个圆 $S^1$ 上的角度，可以用同样的Riemannian flow matching框架处理，$S^1$ 是 $SO(3)$ 的一个特例子群）用Riemannian flow matching，两者由同一个Transformer联合预测——这既是几何上正确的选择，也直接为第6节的"少步实时"铺路，因为已有工作（Riemannian Consistency Model, 2025）证明Riemannian flow matching可以被一致性蒸馏到1-4步采样，恰好对应你要的"少步DiT"。

---

## 5. 跨形态解码器：如何让一套编解码输出任意人形的SOMA时序数据

### 5.1 两级设计

**级别1（形态无关）**：去噪/流匹配主干只在SOMA canonical空间（即第0节定义的骨架，含 $\mathcal K$ 的位置/速度/旋转 + 链路冗余隐变量）里运算，与具体机器人无关。这一层的输出是一条SOMA骨架的运动序列。

**级别2（形态特定）**：为每个目标形态 $e$ 训练/构造一个轻量适配器 $D_e:\text{SOMA轨迹}\to\theta_e(t)$（该形态的原生关节角轨迹）。两种实现路径：

- **(a) IK-based retargeter**（NVIDIA目前对G1唯一发布的方案）：多目标IK求解 + 关节限位裁剪 + 落地点稳定，即`NVIDIA/soma-retargeter`已实现的配方。**对新机器人，你需要做的不是重新训练生成模型，而是照着这个配方给新机器人写一份运动学树/骨长/关节限位的配置文件，跑一遍同样的IK pipeline**——这是目前"支持任意人形"在SOMA生态里唯一已验证的路径,但IK求解通常较慢、且不可微，不适合直接塞进实时闭环。
- **(b) 学习式蒸馏解码器**（推荐用于实时部署）：用(a)批量生成的 (SOMA canonical轨迹, 机器人原生关节角) 配对数据，训练一个小型MLP/Transformer $D_e$ 做监督回归（关节角L2 + 正向动力学一致性损失 + 关节限位惩罚），把IK的解"蒸馏"进一个前向、可微、常数时间的网络里，代价是一次性的数据生成+蒸馏训练，收益是推理时零IK开销。

### 5.2 让解码器本身具备"任意"泛化能力：形态嵌入

若想进一步减少"每加一个新机器人都要重新训练一个 $D_e$"的成本，可以引入**形态嵌入向量** $\boldsymbol e_{\text{emb}}$，编码该机器人的结构先验（骨长向量、每关节自由度、关节限位、运动学树邻接关系——邻接关系建议用GNN对URDF运动学图编码），把 $D_e$ 换成一个**单一的、以 $\boldsymbol e_{\text{emb}}$ 为条件的共享解码器** $D(\cdot\,;\boldsymbol e_{\text{emb}})$，通过FiLM/adaLN方式把 $\boldsymbol e_{\text{emb}}$ 注入每一层：

$$
\boldsymbol h_l \leftarrow \gamma(\boldsymbol e_{\text{emb}})\odot \boldsymbol h_l + \beta(\boldsymbol e_{\text{emb}})
$$

这正是Being-H0.5的Mixture-of-Flow"解耦共享motor primitive与形态特定专家"思路的具体实现方式——训练时用多个已知形态（人体、G1、若干仿真中的虚构humanoid变体）的配对数据联合训练 $D(\cdot;\boldsymbol e_{\text{emb}})$，使其学到"形态嵌入→关节角映射"的插值/外推能力，从而对一个**从未见过的新机器人**，只要能算出其 $\boldsymbol e_{\text{emb}}$（即写出其URDF/骨架描述），就有机会零样本或少样本适配，而不必为每个新机器人单独训练。这是目前比NVIDIA已发布方案更进一步的设计，但风险在于对结构差异较大的新机器人（自由度分布、拓扑显著不同）泛化性没有保证，需要用(a)的IK基线做兜底校验。

---

## 6. 实时性：flow matching与少步DiT不是互斥选项，是分层选项

### 6.1 两者的关系

"flow matching"是训练目标/概率路径的设计（直线路径降低了ODE的曲率，因此**同为一阶ODE求解器**时，flow matching天然比传统score-based diffusion需要更少的求解步数，通常10-20步 vs. DDPM的50-100步）；"少步DiT（如4步）"则是在此基础上**再做一次蒸馏**——用一致性蒸馏（consistency distillation，MotionLCM已在动作生成的隐空间里验证过实时可用）或"shortcut models"（ARDY自己在Limitations里明确点名的下一步方向：Lu & Song 2025; Geng et al. 2025，学习一个以步长为条件的速度场，使其可以一步跨出任意大小的间隔）把多步ODE解坍缩为1-4步的一致性函数。

**推荐的分层方案**：

1. 主干训练目标：**flow matching**（本身步数需求低，且在旋转通道上用第4节的Riemannian flow matching，几何上正确）；
2. 蒸馏：在flow matching基础上做**一致性蒸馏**，目标步数压到1-4步——第4节提到的Riemannian Consistency Model（2510.00983）已经证明这条路径在 $SO(3)$ 上是可行的，直接对应你"旋转通道用非高斯分布 + 又要少步实时"的双重需求；
3. 流式化：为了实现真正的感知-规划-执行闭环（而非批量离线采样一整段序列），还需要**因果自回归分块**（ARDY/MotionStreamer/Diffusion Forcing的做法）——把序列切成固定长度的chunk，每个chunk的去噪只依赖历史chunk的隐状态（而非未来），使生成可以随时因新的约束/文本/proprioception输入而重新规划（ARDY的interactive demo里"Replan Buffer Size / Replan Trigger Threshold"机制就是这个思想的具体实现）。

### 6.2 走向"直接输出causal PD"的最后一步

目前ARDY+SONIC的组合（NVIDIA官方文档确认："By combining ARDY's real-time humanoid motion generation with the SONIC physical tracking policy, we enable interactive robot motion control"）仍然是**两阶段**：生成器输出运动学参考轨迹，独立训练的WBC追踪策略负责把它转成扭矩/PD。这正是你计划的"早期阶段"。

要走到"长期"——生成模型本身直接输出causal PD/控制primitive——数学上需要做三件事：

1. **把生成器变成一个策略**：不再是开环生成 $p(\text{轨迹}\mid \text{条件})$，而是**闭环**的 $\pi(\boldsymbol a_t \mid \boldsymbol o_t, c)$，其中 $\boldsymbol o_t$ 包含实时本体感觉反馈（关节角、IMU、接触力），$\boldsymbol a_t$ 是下一步的PD目标/关节力矩。
2. **把"生成速率"降到"控制速率"**：ARDY/Kimodo的生成频率是20-25fps，而PD控制通常要50-1000Hz——需要在每个控制tick跑一次**已蒸馏到1步**的去噪（即上面6.1的一致性函数），以当前本体状态 $\boldsymbol o_t$ 和replan buffer为条件，只输出下一小段 $\Delta t$ 的PD目标，这是一个**receding-horizon**的、每tick重新条件化的流式生成过程——本质上是把ARDY已有的"replan"机制的粒度从"关键帧/文本"下沉到"每个控制tick的本体反馈"。
3. **解决生成器-执行器之间的曝光偏差（exposure bias）**：训练时生成器看到的是干净的ground-truth运动，但部署时它接收到的 $\boldsymbol o_t$ 是下游执行器（真实电机/仿真物理）带着跟踪误差产生的状态——这正是ReactiveBFM（本会话文献调研阶段检索到的论文，2606.30362）明确解决的问题，其"scheduled prefix sampling curriculum"让生成器在训练时就暴露给执行器自身产生的误分布，学习纠错行为而非只学干净轨迹的复现。**建议在你实现第三步"直接输出PD"时，直接采用ReactiveBFM的prefix-sampling curriculum**，而不是从零设计曝光偏差的解决方案。

---

## 7. 汇总：一个统一的数学定义

把以上拼成一个整体的生成模型定义：

**状态空间**：SOMA canonical骨架上末端关节集合 $\mathcal K$ 的 $(\boldsymbol p,\boldsymbol v,\boldsymbol r)\in\mathbb R^6\times SO(3)$，加上中间链路冗余隐变量 $\varphi\in S^1$（每条链一个），序列长度 $T$（或流式chunk长度 $\tau$）。

**条件**：$c=(C_{\text{ctrl}}, C_{\text{text}}, C_{\text{scene}})$，其中 $C_{\text{ctrl}}$ 由第2节的联合掩码分布 $m$ 采样得到（涵盖关键帧型/自由型/你要的稀疏关节-连续时间型）。

**生成过程**：对 $(\boldsymbol p,\boldsymbol v)$ 通道用欧氏flow matching，对 $(\boldsymbol r,\varphi)$ 通道用Riemannian flow matching（$SO(3)$ 与 $S^1$），由**同一个**Transformer联合预测两路速度场 $u_\theta\big((\boldsymbol p,\boldsymbol v,\boldsymbol r,\varphi)_t, t; c\big)$，约束token通过第2节的ControlNet式交叉注意力+RePaint式硬替换注入，三模态条件通过第3节的多路交叉注意力+模态dropout注入。训练完成后按6.1做一致性蒸馏（1-4步）+ 因果分块（流式）。

**解码**：SOMA canonical轨迹通过第5节的形态特定适配器 $D_e$（IK-based或蒸馏学习式，可选形态嵌入 $\boldsymbol e_{\text{emb}}$ 做跨形态共享）映射到目标机器人 $e$ 的关节角轨迹 $\theta_e(t)$。

**应用**：

- 早期：$\theta_e(t)$ 作为WBC模仿学习的参考轨迹/teacher（对标Kimodo/ARDY + ProtoMotions/SONIC pipeline）；
- 长期：按6.2把整个流水线收缩为闭环策略 $\pi(\boldsymbol a_t\mid \boldsymbol o_t, c)$，用ReactiveBFM式prefix-sampling curriculum处理曝光偏差，实现感知-规划-执行端到端小脑。

---

## 附：与现有工作的定位对照

| 你的设计要素 | 现有最接近的工作 | 差异/需要你自己补的部分 |
|---|---|---|
| SOMA canonical骨架 + 跨形态适配器分层 | NVIDIA SOMA-X + SOMA Retargeter（仅G1） | 其他机器人的适配器需自建；可学习蒸馏/形态嵌入共享解码器尚无公开工作 |
| 末端关节代表性 + 中间关节隐式推断 | AvatarPoser/AGRoL/SparsePoser/DragPoser（反问题：稀疏→全身） | 这些是"稀疏输入重建"，你是"稀疏编码+生成"，方向相关但目标不同；本文给出的冗余度公式 $r_{\text{链}}$ 是这些工作背后的几何解释 |
| 空间稀疏、时间连续约束 | OmniControl（联合稀疏）、AnyBody（任意关键点子集）、Kimodo（关键帧型） | 三者是同一掩码空间的不同采样分布，你需要的模式训练时要加大采样权重 |
| 三模态融合（约束/文本/场景） | OMG（语言+音频+参考动作）、MotionWAM（视频条件） | 场景感知（点云/SLAM）作为条件在检索到的文献里较少见，需自行设计tokenizer |
| 旋转通道非高斯分布 | FrameDiff/FoldFlow/Riemannian Consistency Model（均为蛋白质结构生成） | **本会话检索到的动作生成方向论文（含本轮及更早的文献调研阶段）均未采用**，是可迁移的方法学空白 |
| 少步实时 | MotionLCM（一致性蒸馏）、ARDY自陈的shortcut model方向 | ARDY自己承认这块尚未做，是当前前沿而非已解决问题 |
| 闭环PD输出 | ReactiveBFM（曝光偏差）、ARDY+SONIC（当前仍是两阶段） | 直接输出PD的端到端方案在公开文献中仍是空白，是你长期目标的真正创新点所在 |
