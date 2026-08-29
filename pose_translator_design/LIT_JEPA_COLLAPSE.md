# JEPA / 自监督联合嵌入家族的防塌缩方法全谱调研

**日期**: 2026-08-06 · **针对**: 我们的条件 JEPA + FSQ 离散 token + 端点强条件解码器
**核心结论先说**: 教科书上 90% 的防塌缩方法治的是"表征塌缩"(type a)——**我们没得这个病**。我们得的是"解码器抄近路"(type c)，这个病在 VAE 文献里叫 "强条件下的 posterior collapse"，在 JEPA 文献里 2026 年才刚有人碰（Delta-JEPA），**没有统一理论、没有标准度量、没有离散版解法**——这就是论文机会所在。

---

## 0. 我们的病理分类（先定坐标系）

| 类型 | 名字 | 症状 | 我们有没有 |
|---|---|---|---|
| (a) | 经典表征塌缩 | 所有输入映到同一个（或低秩的）嵌入 | **没有**。InfoNCE 到 0.02，127 个负样本里认得出自己的窗口；无关窗口对照 ~0.01 |
| (b) | 码本使用塌缩 | 离散码只用一小撮 | **没有**。FSQ 结构性免疫（下文验证），码使用熵健康 |
| (c) | 解码器捷径塌缩 | token 信息量足够，但解码器在强端点条件下**无视它** | **有，主病**。shuffle 消融 content share 只有 0.11–0.28；condition dropout (p=0.15–0.3) 拉回 0.57 |
| (d) | 输出分布均值回归 | 输出被压向平均动作，幅度被削平 | **有，副病**（空间层 frame-level FSQ tokenizer）：踢腿被压扁 |

关键认知：(a)(b) 发生在**编码器/表征侧**，(c)(d) 发生在**解码器/输出侧**。文献里绝大多数"anti-collapse"武器都对准 (a)，少量对准 (b)；(c) 散落在 VAE posterior collapse 文献里，(d) 散落在动作预测 / 生成模型文献里。**没有人把四类放进同一个框架，尤其没有人在"量化 latent + 强条件"的设定下讨论过。**

---

## 1. 经典六件套（全部对准 type a，对我们基本无直接用，但机制要懂）

| 方法 | 机制（一句话） | 治哪类 | 对 FSQ 离散 + 条件设定的适用性 | 成本 |
|---|---|---|---|---|
| **BYOL** (2020) | 目标网络 = 在线网络的 EMA 慢拷贝 + 在线侧多加一个 predictor，制造不对称，塌缩不再是稳定解 | (a) | 低。假设连续嵌入、无条件；我们表征侧已健康 | 双份前向 + EMA 内存 |
| **SimSiam** (2021) | stop-gradient + predictor，证明连 EMA 都可省 | (a) | 低。同上 | 几乎免费 |
| **Barlow Twins** (2021) | 两视图嵌入的互相关矩阵逼近单位阵：对角→1（不变性），非对角→0（去冗余） | (a) 含维度塌缩 | 低-中。只能作用在**量化前**的连续特征上 | 一个 D×D 矩阵，批次要够大 |
| **VICReg** (2022) | 三项显式正则：方差 hinge（每维 std ≥ γ）+ 协方差非对角→0 + 两视图 MSE 不变性 | (a) 含维度塌缩 | 中（详见 §4：在 FSQ 上的正确翻译是码使用熵） | 三个权重要调 |
| **DINO** (2021) | centering（EMA 均值扣除，防单点塌缩）+ sharpening（教师低温，防均匀塌缩）——两个反方向的力互相制衡 | (a)+(b) | **中**。centering/sharpening 对**离散分布**有直接类比：对 token logits 做使用频率校正 | 低 |
| **SwAV** (2020) | Sinkhorn-Knopp 强制批内样本**均匀分配**到各 prototype | (b) 的连续版 | **中**。"均匀分配码"就是码使用均衡的最直接类比；FSQ 已免疫故收益小 | Sinkhorn 迭代，便宜 |

**小结**: 六件套解决的问题我们温度层已经用 InfoNCE + FSQ 解决了。它们的价值是提供"翻译素材"：DINO 的 centering 和 SwAV 的均匀分配，翻译到离散域就是码使用熵最大化——这正是 FSQ/LFQ 社区已经在做的事（§4）。

---

## 2. JEPA 正统家族

| 方法 | 防塌手段 | 有条件吗 | 与我们的关系 |
|---|---|---|---|
| **I-JEPA** (2023) | EMA 目标编码器 + 掩码预测的不对称性 + predictor 以位置 token 为条件。无对比项、无显式正则 | 位置条件 | 我们的祖型。注意：C-JEPA (NeurIPS 2024) 指出 **EMA 单独有时防不住全塌**，要补 VICReg |
| **V-JEPA / V-JEPA 2** (2024/25) | 完全同款：EMA + stop-grad，**没有任何显式方差/对比正则**。V-JEPA 2 只是放大（ViT-g）+ 加了动作条件 predictor 头 | V-JEPA 2-AC: 机器人动作条件 | 关键情报：V-JEPA 2-AC 的 305M 动作条件 predictor 是**确定性单头**——LeCun 蓝图里"承载多模态未来的 latent 变量 z"**所有官方实现都没做**。随机动力学下它会均值回归（= 我们的 type d 在 JEPA 界的表现） |
| **MC-JEPA** (2023, Bardes) | 光流+内容联合学习，**内嵌 VICReg 正则**防塌 | 无 | 先例：JEPA 里塞 VICReg 是可行的 |
| **A-JEPA** (2024, 音频) | EMA + 课程式掩码 | 无 | 无新机制 |
| **ACT-JEPA** (2025) | IL + JEPA 联训：同时预测动作 chunk 和 latent 观测序列；防塌靠 JEPA 常规 + 动作监督本身 | 动作条件 | 与我们同域（策略/动作），但没碰解码器捷径问题 |
| **C-JEPA** (NeurIPS 2024) | JEPA + VICReg 显式结合，动机就是"EMA 不够" | 无 | 若我们温度层将来出 (a)，这是现成配方 |
| **Var-JEPA** (2026) | 给 predictor 塞随机 latent，变分训练（ELBO+KL），显式处理"predictor 无视 latent"——用 KL 权重调度 + 架构约束 + 信息论正则 | 是 | **首个直面 (c) 的变分 JEPA**。但它假设连续高斯 posterior；FSQ 离散 token 上 KL 没有直接定义 |
| **MoP-JEPA / Branch-JEPA** (2026) | 多 predictor 混合 + **硬指派赢者通吃**：随机动力学的每个 mode 由一个 predictor 接，杜绝对 mode 求平均 | 是 | **对准 (d)**（均值回归的根因之一 = 多模态未来被单头平均）。若空间层压扁源于多模态，这是结构解 |
| **Delta-JEPA** (2026) | ★ 详见 §5.3。诊断出 "endpoint 拼接解码 = 捷径"，改成**只从位移 Δz = z_{t+1} − z_t 解码动作** | 是 | **与我们 type (c) 最直接对口的已发表工作** |
| **Demo-JEPA** (2026) | 跨具身单样本模仿的 JEPA | 是 | 与 lloyd 项目的跨具身方向撞车/可引用，防塌上无新意 |

**小结**: JEPA 正统的防塌配方就一句话——EMA + stop-grad + 掩码不对称，全部对准 (a)。**2026 年出现的三个分支（Var-JEPA、MoP-JEPA、Delta-JEPA）恰好分别对准我们的 (c)、(d)、(c)**，说明社区刚刚撞上同一堵墙，但都还是连续嵌入 + 单步设定，没人做过 FSQ 离散 + 轨迹级。

---

## 3. LeCun 最新: LeJEPA + SIGReg（2025.11, Balestriero & LeCun）

### 它到底做了什么

- **理论**: 证明（线性+非线性探针两种评估下）嵌入的最优分布是**各向同性高斯**——这是唯一让下游预测风险最小的分布。
- **SIGReg** (Sketched Isotropic Gaussian Regularization): 把高维嵌入投到**随机一维方向**上，对每个一维投影用 **Epps–Pulley 特征函数检验**（一种统计学拟合优度检验）度量"离标准高斯多远"，多个方向汇总成正则项。选 Epps–Pulley 是因为它的 loss/梯度/曲率**天然有界**，训练稳；线性时间和内存。
- **损失**: `L = (1−λ)·L_pred + λ·SIGReg`，**只有一个超参 λ**。
- **扔掉了什么**: EMA 教师、stop-gradient、predictor 不对称、whitening、学习率魔法调度——全部不要。
- **规模验证**: 10+ 数据集、60+ 架构、最大 1.8B 参数。

### 后续与批评

- **When Does LeJEPA Learn a World Model?** (2026.05, Klindt & LeCun & Balestriero): 给出可辨识性定理——当世界的隐变量按"平稳 + 加性噪声"转移演化时，LeJEPA 能**线性地**恢复真实隐变量，且高斯是唯一能保证这点的目标分布；近似满足时退化是连续的。用 2D 到 1024 维、含像素级机器人控制验证。
- **批评**（社区评论，如 "The Strange Tyranny of the Gaussian"）: 各向同性高斯目标会抹平数据的**结构性/多模态**先验；可辨识性定理的"平稳加性噪声"前提在真实世界很强。尚无严肃的失败复现报告（发表刚 9 个月）。

### 对我们的判决

| 问题 | 判决 |
|---|---|
| SIGReg 治哪类 | **(a) 专用**。它是"表征分布长什么形状"的正则，与解码器是否使用 token **完全无关** |
| 能否用于 FSQ | **有根本冲突，且没人研究过**：SIGReg 要嵌入呈高斯（中间密、边缘疏），FSQ 要 tanh 网格上**均匀**使用（每个格子等概率）。把 SIGReg 加在量化前特征上，会系统性偏爱网格中心码。这个冲突本身是个可发表的观察（§7 开放问题 3） |
| 值得借的 | ① "随机一维投影 + 有界统计检验"这个**廉价度量高维分布形状**的技巧，可以拿来做我们自己的诊断仪表盘；② "单超参、可证明目标"的论文写法 |

---

## 4. VICReg 深挖 + 量化 latent 里的"方差正则"到底是什么

### VICReg 三项各自治什么

| 项 | 数学 | 治什么 | 失效场景 |
|---|---|---|---|
| Variance | 每维 std 的 hinge：max(0, γ − std) | 防"所有样本一个点"（全塌） | 只保证每维**边缘**方差，挡不住维间冗余 |
| Covariance | 非对角协方差平方和 → 0 | 防维度塌缩（信息挤在低秩子空间） | 只去**线性**相关；非线性冗余不管 |
| Invariance | 两视图 MSE | 学习信号本身 | 需要好的增广对 |

已知问题: 三个权重敏感；批统计量依赖大 batch；**VICRegL** (2022) 是局部特征版（全局+局部特征都做 VICReg），对我们无关键增量。

### 关键问题：有没有人在 VQ/FSQ 量化 latent 上做过方差/协方差正则？

**调研结论：没有直接的。** 最接近的谱系是：

| 工作 | 机制 | 与 VICReg 的对应 |
|---|---|---|
| **MAGVIT-v2 / LFQ** (2023) | 隐式 2^d 码本 + **熵损失**：最小化每 token 熵（用码要果断）+ 最大化批平均码分布熵（全体码要用满） | 双熵项 = variance 项的离散翻译 |
| **LGQ** (2026) | token 级"峰度正则" + 全局"使用均衡正则"，学习离散化几何 | 同上，但网格可学 |
| **VAEVQ** (2025) | 码本嵌入向高斯先验对齐的分布一致性正则 (DCR) | 离散-连续空间一致性，蹭到一点 covariance 的意思 |
| **iFSQ** (2026) | 一行代码改 FSQ 提升生成质量 | 工程修补 |
| **FSQ 原文** (2023) | 有界 tanh + 逐维取整，**结构性防死码**——不需要 commitment loss、码重启、熵惩罚，码使用率接近 100% | 这就是我们 (b) 免疫的出处，已被广泛复现 |

**正确的翻译（我们该写进论文的命题）**: 在有界 tanh 网格上，"方差正则"退化为**每维边缘码使用熵最大化**，"协方差正则"对应**维间码联合分布的独立性/联合熵**（6 维 × 5 级，联合熵上限 6·log5 ≈ 13.9 bits）。我们应该测**联合码熵 vs 边缘码熵之和**的差（= 维间总相关 total correlation）——这是 FSQ 版的"维度塌缩"度量，**文献里没人报告过这个量**。

### 量化 latent 的塌缩文献（超出码本塌缩的部分）

- 码本塌缩 (index collapse) 本身：VQ 需要重启/分裂/熵惩罚，FSQ/LFQ 结构性解决——已闭环。
- **超出码本塌缩**: Deconstructing Generative Diversity (2025.12) 用信息瓶颈分析离散 latent 生成模型（AR/MIM/扩散三种 prior），发现 AR prior 走"压缩优先"路线会出码本塌缩、多样性只剩采样温度——但它分析的是 **prior 侧**，且明确**没碰条件设定**。λ-VAE (2026) 也明确写"离散 latent 和条件模型 left for future work"。**→ 空白确认。**

---

## 5. type (c) 专区：条件生成的"捷径塌缩"（我们的主病）

### 5.1 VAE 文献里的名字：强条件下的 posterior collapse

病理机制（与我们逐字吻合）: **条件信号很强 + 解码器足够能干时，模型只靠条件就能出个"说得过去"的输出，于是 latent 被无视**。经典成因分析：
- "Don't Blame the ELBO" (NeurIPS 2019): 线性 VAE 分析证明塌缩不（只）是 KL 项的锅，跟优化景观和方差结构有关。
- λ-VAE (2026): 两个耦合机制——重建梯度比 KL 恢复力消失得快（梯度失衡）+ 采样噪声淹没编码器信号（信息差）。它的实验里最扎心的一条：**两个 ELBO 几乎相同的模型，解码器分配给 latent 的容量差 6.2 倍**——ELBO/总 loss 完全看不出 (c)，必须直接测容量分配。这佐证了我们用 shuffle-ablation content share 而不是 loss 来诊断是对的。

### 5.2 武器清单（按机制分四族）

**族 1: 给 KL/信息项松绑（"别急着惩罚 latent 携带信息"）**

| 方法 | 机制 | FSQ 可用性 |
|---|---|---|
| KL annealing (Bowman 2016) | KL 权重从 0 慢慢升 | ✗ 我们没有 KL 项 |
| Cyclical annealing (Fu 2019) | 周期性重来，多次逃逸 | ✗ 同上 |
| **Free bits** (Kingma 2016) | KL 低于阈值 κ 就不罚——**保底信息预算** | △ 思想可翻译：给 token 设"最低信息地板"（见 5.4） |
| delta-VAE (Razavi 2019) | 构造上让 posterior 无法等于 prior（承诺速率） | △ 同上，结构性保底 |
| InfoVAE / MMD (Zhao 2017) | 用 MMD 替代 KL + 显式最大化 I(z; x) | △ MI 最大化项可搬：最大化 I(token; 轨迹)——我们的 InfoNCE 已经在编码侧做了，缺的是**解码侧** |

**族 2: 削弱捷径 / 强化 latent 通路（架构手术）**

| 方法 | 机制 | FSQ 可用性 |
|---|---|---|
| Word dropout (Bowman 2016) | 训练时把解码器的自回归输入随机挖掉，逼它看 latent | ★ 这就是我们 condition dropout 的祖型 |
| **Skip-VAE** (Dieng 2019) | latent 通过 skip 连接**直连解码器每一层**；定理：skip 结构提升 I(z; x) 下界 | ★★ 直接可用：我们的 token 现在从哪层进解码器？若只在输入层拼接，按 Skip-VAE 改成逐层注入（加到每层 KV 或 AdaLN）是最便宜的结构增强 |
| 强条件 CVAE (ICLR 2020 投稿) | 诊断"拼接条件与 latent"是病根，改成**让 latent 的采样依赖条件**（耦合而非并列） | ★ 对应我们：token 不该与端点并列拼接，而应编码"给定端点之外的增量信息"（与 5.4 残差方案同源） |
| Lagging inference (He 2019) | 编码器多步激进更新，追上解码器 | △ 训练调度层面，可试但非根治 |

**族 3: 条件通路上加信息瓶颈（"让捷径变窄"）**

| 方法 | 机制 | FSQ 可用性 |
|---|---|---|
| Conditional IB / Variational bandwidth bottleneck | 最小化 I(瓶颈; 特权输入 \| 常规输入)，按需付费地访问强信号 | ★ 翻译：给端点条件通路加噪 / 降维 / 晚注入 / 限带宽，使"只靠端点"变得更贵。与 condition dropout 互补（dropout 是随机断路，IB 是持续限流） |
| Free Transformer (Fleuret 2025.10) | 见 5.4 | ★★ |

**族 4: 条件 dropout + 引导（扩散界的成熟答案）**

- **Classifier-free guidance** (Ho & Salimans 2022): 训练时以 10–20% 概率丢弃条件（同一模型学会有/无条件两种生成），推理时外推 `out = uncond + w·(cond − uncond)` 放大条件的边际贡献。
- 为什么防捷径：被迫在无条件模式下也要产出合理分布 → 模型必须真正学会"条件带来了什么增量"，而不是把条件当拐杖。
- **对我们的直接推论（几乎没人做过的方向）**: 我们已经把 dropout 用在**条件（端点）**上拉回 content share 0.57。CFG 的完整形态还有第二半——**推理时对 token 做引导外推**：`traj = decode(端点, ∅_token) + w·(decode(端点, token) − decode(端点, ∅_token))`，w>1 时在推理侧放大 token 的边际贡献，**不需要重训**（只要训练时 token 也偶尔被 drop）。这是"latent guidance"，扩散文献里只有零星探索，JEPA 文献里没有。

### 5.3 JEPA 界唯一的对口工作：Delta-JEPA (2026)

- **诊断**: 逆动力学解码器吃 `[z_t, z_{t+1}]` 拼接时，会利用目标表征里与动作相关的痕迹**抄近路**，不真正建模转移——和我们"解码器吃端点、无视 token"是同构病。
- **手术**: 解码器**只准看位移 Δz = z_{t+1} − z_t**，不给端点本身。位移不携带信息就解不出动作 → "相邻嵌入不能塌在一起，否则丢动作信息"。损失就两项：预测 MSE + λ·动作重建 MSE。
- **诊断工具**（可直接搬）: 固定历史、只变动作，看响应的 PCA 分离度（action-response PCA）；轨迹分叉可视化；Δz 对物理状态变化量的线性探针。
- **成绩**: 四个连续控制任务 SOTA（OGB-Cube +15.1 分 vs LeWM），消融证明"位移解码"稳定优于"端点拼接解码"（+4.1 ~ +12.6）。
- **翻译到我们**: 加一个辅助解码头，**只吃 token（不给端点）**，要求它重建轨迹的**增量表示**（Riemann 弧长增量序列 / 速度剖面——恰好我们的主损失已经是这些量！）。token 解不出增量剖面 → 直接罚。这把"content share"从事后诊断变成了训练时的显式约束。

### 5.4 Free Transformer (2025.10)：离散 latent + 强条件下的量产级配方

与我们设定惊人地近：decoder-only Transformer（条件=自回归上文，极强）+ **离散** latent Z（每 token 16 bit 二值→one-hot）：
1. **注入位置**: Z 不在输入层拼接，而是注入**中间层**（第 L/2+1 块的 KV）——前半层白给条件，后半层被迫融合 Z；
2. **容量预算**: free bits 阈值 κ，只罚超出 κ 的 KL，典型 **0.5–1 bit/token**——latent 被保底又不被滥用；
3. **离散梯度**: 二值编码 + Binary Mapper 直通梯度。
成绩: 8B 参数 / 1T token，HumanEval+ +11.4%，GSM8K +2.8%，开销仅 ~3%。
**要点**: 这证明"解码器无视离散 latent"在最强条件（LLM 上文）下也能被治好，配方 = **注入深度 + 信息预算**两板斧。我们缺的是 FSQ 版的"信息预算"——KL 对 FSQ 无定义，需要发明离散代理（如：token 置换后的 loss 增量下限、或 I(token; 轨迹\|端点) 的变分下界）→ 开放问题 5。

---

## 6. type (d) 专区：输出均值回归（空间层的病）

病根有两个可能，先分诊再下药：
- **根因 A（损失形状）**: 逐帧 L2 重建对"削峰"不敏感——L2 的最优解就是条件均值，高幅度、高频细节首先被牺牲。图像界的同款病 = VQ-VAE 纯 L2 出糊图，**VQGAN 的答案是加对抗损失**。
- **根因 B（多模态平均）**: 同一条件下有多个合理输出，单头确定性解码器输出它们的均值。V-JEPA 2-AC 的确定性 predictor 被点名有这个病；MoP-JEPA 用硬指派混合 predictor 治。

| 药 | 机制 | 治根因 | 成本/风险 |
|---|---|---|---|
| 对抗判别头（VQGAN/ARNet/TCGAN 谱系） | 判别器罚"不像真动作的平均货"，逼幅度分布对齐 | A | GAN 训练不稳；动作界 ARNet、TCGAN 已验证可行 |
| 扩散/流 refinement（HumanMAC、Diffusion-Refinement、PrediFlow） | 粗预测 + 生成式细化头恢复分布尾部 | A+B | 推理多几步；动作预测界 2023-26 的主流答案 |
| RVQ / 离散+连续残差（DC-Motion、FlowCoMotion） | 离散 token 管结构，连续残差管高频幅度——"量化伪影压制高频细节"是已知病，残差是已知药 | A | 架构改动中等；与我们 FSQ 空间层**最兼容**（FSQ 版 RVQ 现成） |
| 频域/差分加权损失（DCT 高频加权、速度/加速度项） | 直接把"削峰"变成显式罚项 | A | 几乎免费，先试这个 |
| MoP-JEPA 硬指派混合 | 每个 mode 一个头，赢者通吃更新 | B | 头数超参；若空间层单帧重建本质单模态，则无用 |
| λ-VAE 方差均衡 | 重参数噪声 σ^λ 缩放，纠正梯度失衡 | A（VAE 内） | 只适用有连续随机层的地方；作者明说离散/条件未探索 |

**分诊实验（先做再选药）**: 空间 tokenizer 是逐帧重建、条件固定，理论上接近单模态 → 大概率是根因 A。验证：把重建损失换成幅度分位数加权（或只看高速帧的 loss），若削峰缓解则 A 实锤，处方 = 频域/差分加权先行，不够再上 RVQ 残差或对抗头。

---

## 7. 理论与度量：该往仪表盘上加什么

| 度量 | 出处 | 测什么 | 我们该怎么用 |
|---|---|---|---|
| **RankMe**（奇异值分布的熵 = 有效秩） | Garrido, Balestriero, LeCun (ICML 2023) | 维度塌缩；无超参、与下游性能相关 | 加在**量化前**的 register 嵌入上，监控温度层 (a) 的早期信号 |
| 码联合熵 − 边缘熵之和（total correlation） | 我们自己（§4 命题） | FSQ 版维度塌缩：6 维是否在传冗余信息 | **文献空白，直接可写进论文** |
| Shuffle content share | 我们已有 | 解码器对 token 的因果依赖 | 形式化为 **I(token; 轨迹 \| 端点) 的干预式估计量**——这个条件互信息才是 type (c) 的正名度量，文献里没有标准名字 |
| 解码器容量分配 | λ-VAE (2026) | ELBO 相同容量分配可差 6.2× | 佐证"总 loss 看不出 (c)"，写 related work 用 |
| Action-response PCA / 分叉可视化 | Delta-JEPA (2026) | 固定其他、只变 token，看输出响应分离度 | 直接搬：固定端点、遍历 token，看轨迹族的散开程度（顺便满足"可视化优先"的项目铁律） |
| 随机 1D 投影 + Epps–Pulley | LeJEPA (2025) | 高维分布形状的线性成本检验 | 诊断量化前特征分布 |
| IB 三探针（码本子集/argmax 确定性/条件改写） | Deconstructing Generative Diversity (2025) | 离散生成模型的多样性来源分解 | 分析我们 prior/生成阶段的多样性预算 |
| 理论背景 | Jing et al 2022（对比学习维度塌缩机制）；"Don't Blame the ELBO"（塌缩非 KL 独责） | 为什么塌 | related work |

---

## 8. 处方与机会

### (i) type (c)（解码器无视 token）——按"性价比 × 结构性"排序

1. **Condition dropout 保留 + 补上 CFG 的另一半：推理时 token 引导外推**。训练时同时偶尔 drop token，推理时 `decode(cond,∅) + w·(decode(cond,tok) − decode(cond,∅))`，w 扫 1–3。已验证 dropout 拉到 0.57，引导外推零重训成本，可能再往上顶。
2. **Delta-JEPA 式辅助解码头**：一个只吃 token、不吃端点的小头，重建弧长增量/速度剖面（恰是我们已有的损失量），λ 加权。把 content share 从诊断变成训练约束，**结构性堵捷径**，机制已在 2026 连续控制任务上验证 (+4~15 分)。
3. **Free Transformer 两板斧**：token 从输入层拼接改为**中层 KV 注入**（Skip-VAE 定理背书：逐层注入提升 I(z;x)）；并给 token 设"最低信息地板"（free bits 的 FSQ 翻译——需要自己发明，见机会 5）。
4. **残差化解码**：先冻结训练一个"只看端点"的基线解码器（输出 = 端点条件下的均值轨迹/测地线），主解码器 + token 只负责解释**残差**。token 的任务从"可有可无的补充"变成"残差的唯一信息源"——与强条件 CVAE 的"耦合而非拼接"同源，且给了 content share 一个干净的目标（残差方差解释率）。
5. **条件通路限流**（conditional IB）：端点嵌入降维/加噪/晚注入。与 1 互补，风险是伤端点精度，放最后调。

（1+2 可以并行上；3 的注入深度改动与 2 兼容；4 是更大的手术，做消融论文时上。）

### (ii) type (d)（空间层削峰）——排序

1. **先分诊**（§6：高速帧加权 loss 实验，确定根因 A vs B），几小时的事。
2. **频域/差分加权重建损失**（DCT 高频项、速度加速度项加权）——近免费。
3. **FSQ-RVQ 残差级**或离散+连续残差（DC-Motion 路线）——治"量化压制高频"的结构病。
4. **轻量对抗头**（VQGAN 在动作域的移植，ARNet 先例）——2、3 不够再上。
5. 若分诊指向多模态平均：MoP-JEPA 硬指派混合头。

### (iii) 开放问题（论文机会清单）

1. **量化条件模型的塌缩分类学**：四类塌缩 (a)(b)(c)(d) 的统一框架不存在。λ-VAE 和 IB-分析两篇 2025/26 论文都明文把"离散 + 条件"列为 future work。我们手里有完整的测量学（InfoNCE、码熵、content share、削峰）——**分类学 + 度量套件本身就是一篇论文的骨架**。
2. **VICReg 的离散翻译**：有界网格上"方差正则 ≡ 码使用熵、协方差正则 ≡ 维间 total correlation"这个命题没人写过；顺手给出 FSQ 维度塌缩度量（联合熵 − Σ边缘熵）。
3. **SIGReg × FSQ 的分布冲突**：各向同性高斯目标 vs 均匀网格使用，量化前加 SIGReg 会系统偏置码分布——一个干净的负结果/分析短文。
4. **JEPA 强条件下的 predictor/decoder 捷径塌缩**：只有 Delta-JEPA 碰了相邻问题（单步、连续、逆动力学）。轨迹级、FSQ 离散、端点 SE3 条件的版本没人做；正名度量 **I(token; 输出 \| 条件)** 及其干预式估计（shuffle content share）没有标准化。**这是我们主病，也是最大机会。**
5. **离散 latent 的 free bits**：KL 对 FSQ 无定义，"最低信息地板"的离散代理（置换 loss 增量下限 / 变分 MI 下界）是空白，Free Transformer 只做了二值+KL 版。
6. **Latent-side guidance**：CFG 外推用在内部 token 而非外部条件上，作为推理时反捷径手段——扩散界零星、JEPA 界空白。
7. **离散 latent 的可辨识性**：LeJEPA 世界模型定理只覆盖高斯连续隐变量，FSQ 网格版不存在。

---

## 主要来源

- LeJEPA: [arXiv 2511.08544](https://arxiv.org/abs/2511.08544) · [alphaXiv 解读](https://www.alphaxiv.org/overview/2511.08544v1) · [代码](https://github.com/rbalestr-lab/lejepa)
- When Does LeJEPA Learn a World Model?: [arXiv 2605.26379](https://arxiv.org/abs/2605.26379)
- Delta-JEPA: [arXiv 2606.31232](https://arxiv.org/html/2606.31232v1) · MoP/Branch-JEPA: [arXiv 2607.05238](https://arxiv.org/pdf/2607.05238) · Var-JEPA: [arXiv 2603.20111](https://arxiv.org/pdf/2603.20111)
- V-JEPA 2 / AC: [EmergentMind 汇总](https://api.emergentmind.com/topics/v-jepa-2) · [V-JEPA-2-AC](https://www.emergentmind.com/topics/v-jepa-2-ac)
- ACT-JEPA: [arXiv 2501.14622](https://arxiv.org/abs/2501.14622) · Demo-JEPA: [arXiv 2605.20811](https://arxiv.org/html/2605.20811v1) · C-JEPA: [NeurIPS 2024](https://arxiv.org/abs/2410.19560)
- Free Transformer: [arXiv 2510.17558](https://arxiv.org/html/2510.17558v1)
- 强条件 CVAE posterior collapse: [OpenReview rJlHea4Kvr](https://openreview.net/forum?id=rJlHea4Kvr) · Skip-VAE: [arXiv 1807.04863](https://arxiv.org/abs/1807.04863) · λ-VAE: [arXiv 2607.05531](https://arxiv.org/html/2607.05531) · Don't Blame the ELBO: [NeurIPS 2019](http://papers.neurips.cc/paper/9138-dont-blame-the-elbo-a-linear-vae-perspective-on-posterior-collapse.pdf) · Lagging inference: [arXiv 1901.05534](https://arxiv.org/pdf/1901.05534)
- CFG: [Ho & Salimans](https://openreview.net/pdf?id=qw8AKxfYbI) · [Dieleman 博客](https://sander.ai/2022/05/26/guidance.html)
- VICReg: [arXiv 2105.04906](https://arxiv.org/abs/2105.04906) · RankMe: [arXiv 2210.02885](https://arxiv.org/abs/2210.02885)
- FSQ: [arXiv 2309.15505](https://arxiv.org/abs/2309.15505) · LGQ: [arXiv 2602.16086](https://arxiv.org/html/2602.16086) · iFSQ: [arXiv 2601.17124](https://arxiv.org/html/2601.17124v2) · VAEVQ: [arXiv 2511.06863](https://arxiv.org/pdf/2511.06863)
- IB 分析离散生成模型: [arXiv 2512.01831](https://arxiv.org/html/2512.01831v1)
- 动作预测均值回归: [HumanMAC](https://arxiv.org/pdf/2302.03665) · [Diffusion-Refinement](https://arxiv.org/pdf/2210.05976) · [ARNet](https://arxiv.org/pdf/2011.11221) · [DC-Motion](https://arxiv.org/pdf/2606.14721) · [FlowCoMotion](https://arxiv.org/html/2604.11083v1)
