# 文献调研：大脑和小脑共享一个潜空间

调研时间：2026-08
调研对象：端到端人形机器人栈。上层是 VLM/规划器，下层是全身控制器（WBC）。两边共用同一个潜空间。给 VLM 的词表加上动作 token，就能直接驱动一个真能站住的物理控制器。

我们手上有的：
- 跨本体的离散动作 tokenizer（FSQ），对没训过的机器人也能零样本用。
- 稀疏 SE3 约束接口，带一个投影层，保证输出满足关节限位。
- 还没接任何物理控制器。没有任何 trackability 数字。

---

## 先说结论

一句话：**你赌的方向是对的，而且刚刚被 NVIDIA 用数据证明了。但也正因为如此，架构本身已经不新了。**

### 最重要的一条：SONIC v3 里有你要的那个消融实验

NVIDIA SONIC 第三版（arXiv 2511.07820v3）直接做了这个对比：**让 VLA 预测 FSQ token，对比让 VLA 预测 SMPL 连续位姿**。

| 任务 | 预测 FSQ token | 预测 SMPL 位姿 | 差距 |
|---|---|---|---|
| 平均 | **68%** | 27% | **+41 分** |
| 拿胡萝卜 | 75% | 60% | +15 |
| 开垃圾桶 | 70% | 20% | +50 |
| 汽水罐扔垃圾桶 | **60%** | **0%** | +60 |

原文解释：
> 「FSQ token 提供了一个**低维、离散**的动作空间，VLA 从遥操作示教里更容易学。而高维连续的 SMPL 位姿空间会把小的预测误差**放大成大的跟踪失败**。」

**这就是你整套论点的实验证据，而且不是你做的。**「给 VLM 加动作 token 比让它吐连续位姿好」——已经被验证，差距 41 个百分点，有一个任务是 60% 对 0%。

而且 SONIC 最终系统里 **VLA 就是直接吐 token 的**：预测 78 维动作 = 64 维通用运动 token + 14 维手部关节，直接由通用控制策略解码。中间没有手工接口。

（注意：v1/v2 版本里 VLA 还是吐「头+双手腕位姿、腰高、导航指令」这种手工信号。v3 才换成直接吐 token。如果你看的是旧版会看漏。）

### 已经存在的（别重做）

1. **「上层发潜、下层物理控制器」**——LeVERB、Helix、GR00T+SONIC 都是。
2. **「离散 token 当 LLM 词表 + 物理控制器」**——MoConVQ（2023）、Humanoid-LLA（2026）已做，真机 G1 + Booster T1。
3. **「跨本体共享 FSQ token 空间」**——SONIC 做了（FSQ-32-32），开源。
4. **「FSQ 端到端 + PPO 物理奖励训 tokenizer」**——GPC（SIGGRAPH 2026）做了，680 小时数据，跟踪成功率 99.98%。
5. **「离散 token 上用物理仿真奖励做 GRPO」**——RLPF、Humanoid-LLA 都做了，真机部署。
6. **「生成器只吐控制器跟得住的动作」**——2604.17335 冻结生成器闭环微调 tracker。
7. **「潜空间搜索恢复」**——BFM-Zero（CEM + DIAL-MPC 式退火）、BFMTrack（REINFORCE 潜序列优化）、ReLA/LoLA 都做了。
8. **「字典最近邻检索恢复」**——FIRM 做了 keyframe codebook + 余弦相似度检索。

**FSQ vs VQ 这个仗也已经打完了，FSQ 赢：**
- SCHUR/Humanoid-Union：码本从 256 加到 65536，**FSQ 保持 >99% 利用率，VQ 掉到 90% 以下并明显塌缩**。
- GPC：FSQ 打平 VQ 的质量，但不需要任何码本稳定化技巧。
- ARP：LIBERO-Long 上 FSQ 83.6% / VQ 74.9% / LFQ 73.3%。

所以你选 FSQ 是对的，但这不是你的贡献，是既定事实。

### 真正没人做的（你的四个洞）

1. **零样本到训练时完全没见过的机器人。** 这是最硬的一个。
   - SONIC 的 token 空间是「同一台 G1 上的不同输入模态」对齐，不是跨机器人身体。
   - Humanoid-LLA 是人 + G1 + T1，靠**配对数据**联合量化。
   - Any2Any 迁移到新机器人要 LoRA 微调（但只要 1% 的从头训练算力——**这是你必须打败的诚实 baseline**）。
   - PHASOR 做了跨 5 个本体，但**故意用连续表示**（FFT 相位流形），而且只管周期性动作、只在仿真里。
   - **没有任何一篇声称「新机器人不训练直接用」。**

2. **接口层的硬可行性保证。** 现在只有软引导（SafeFlow 的 physics guidance）或事后拒绝（SafeFlow 的三级门）。**没人用投影层做数学上的硬保证。**

3. **闭环的字典恢复。** 拆开看：
   - 「在潜空间里搜」——有（BFM-Zero、BFMTrack、LoLA）。
   - 「从扰动中恢复」——有（BFM-Zero、RecoverFormer），但都是**涌现的，不是检索出来的**。
   - **「用一个在线 OOD 触发器，从当前状态去重新查字典，而且快到能跑在控制频率上」——没有。**
   - BFMTrack 的潜搜索在 RTX 5090 上要**几分钟**，根本没法实时。BFM-Zero 的搜索也是离线 few-shot。

4. **在「解码器本身就是控制器」的 tokenizer 上做物理奖励 RL。**
   - RLPF 和 Humanoid-LLA 都是 RL 一个**运动学**生成器，然后拿一个**冻结的** tracker 当裁判。
   - GPC 有物理原生的 FSQ tokenizer，但它的 GPT **只用交叉熵训，完全没做 RL 后训练**。
   - 这个组合没人占。

**一句话战略：架构别争了，争「零样本跨本体」+「硬约束接口」+「实时字典恢复」这三件事。**

---

## 一、物理人形控制的潜动作空间

这一块是「规划器和控制器共享潜空间」最接近的已有工作。基本都出自图形学圈。

### PHC / PHC+（Perpetual Humanoid Control）
- ICCV 2023，arXiv [2305.06456](https://arxiv.org/abs/2305.06456)，代码 [github.com/ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC)
- 做什么：一个物理模仿器，把 AMASS 里几乎所有动作都学会跟。不用外力扶。
- 关键技术 PMCP：碰到学不会的难动作，就给网络加一块新容量。这样能一直往上堆数据，还不会忘掉旧的。
- **对你最有用的一点**：它专门训了「失败态恢复」。摔倒了能爬起来，离参考动作 5 米以上也能走回去接着跟。这是「不靠上游、控制器自己救自己」的路线。
- 它是后面所有工作的底座。PULSE 就是从 PHC+ 蒸馏出来的。

### PULSE（Universal Humanoid Motion Representations）
- ICLR 2024 spotlight，arXiv [2310.04582](https://arxiv.org/abs/2310.04582)，代码 [github.com/ZhengyiLuo/PULSE](https://github.com/ZhengyiLuo/PULSE)
- 潜空间结构：**连续 VAE，32 维**。encoder 和 prior 都是对角高斯。
- 最重要的设计：**prior 是被本体感觉（proprioception）条件化的**。不是标准 VAE 那个零均值高斯。理由很直白——人站着不动和人空中翻腾，下一步该做什么的分布完全不一样。
- 下游任务怎么训：任务策略输出的是**残差潜码**，加到 prior 的均值上，再送进 decoder。公式是 `a_task = D(π_task(z) + μ_prior)`。这样探索是围绕「本来就合理的行为」附近展开的，不会乱跑到潜空间的荒地里。
- 覆盖率：从 PHC+ 蒸馏后，模仿成功率 99.8%（40 小时 AMASS）。
- 恢复能力：摔倒后可以从 prior 里随机采样爬起来。但论文承认「可能卡在摔倒或站立状态」。
- 承认的缺点：信息瓶颈是**有损压缩**。在 VR tracking 这种专门任务上，比从头训一个专用策略要差。
- **对你的意义**：这是「共享潜空间」的原型，但潜是连续的。连续潜没法当 token 喂给自回归 LLM——这正是你选 FSQ 的理由。残差潜码那个技巧值得直接抄。

### ASE / CALM / C·ASE
- ASE：arXiv [2205.01906](https://arxiv.org/abs/2205.01906)，[项目页](https://xbpeng.github.io/projects/ASE/index.html)。对抗模仿 + 互信息技能发现，学一个可复用技能嵌入。
- C·ASE：SIGGRAPH Asia 2023，[项目页](https://frank-zy-dou.github.io/projects/CASE/index.html)。加条件，动作覆盖比 ASE 和 CALM 好。
- **已知问题**：PULSE 论文直接说，ASE 和 CALM 在 AMASS 这种又大又杂的数据上「抓不住运动技能」。它们只在小而干净的技能集上work。
- 结论：对抗式技能嵌入这条线基本被 PULSE 这种蒸馏式路线取代了。别走这条。

### ControlVAE
- SIGGRAPH Asia 2022，arXiv [2210.06063](https://arxiv.org/abs/2210.06063)，[项目页](https://heyuanyao-pku.github.io/Control-VAE/)
- 特点：训练时用一个**可学的世界模型**来直接监督潜空间和控制策略。model-based，样本效率高。
- 也用了状态条件先验，和 PULSE 一个思路，而且更早。

### NCP（Neural Categorical Priors）
- arXiv [2308.07200](https://arxiv.org/abs/2308.07200)，[项目页](https://tencent-roboticsx.github.io/NCP/)，腾讯 RoboticsX
- **这是第一个把离散瓶颈用在物理角色控制上的**。VQ-VAE，离散信息瓶颈。从训好的类别先验里采码，就能生成高质量动作。
- 还提了个 "prior shifting"，用好奇心驱动的 RL 去调先验分布。
- **对你的意义**：离散潜 + 物理控制这条路 2023 年就通了。你不是第一个。但 NCP 是单本体、图形学角色，不是跨本体机器人。

### MoConVQ ★重点
- ACM TOG 2024，arXiv [2310.10198](https://arxiv.org/abs/2310.10198)，[项目页](https://moconvq.github.io/)，北大 Libin Liu 组
- 做什么：**VQ-VAE 离散表示 + model-based RL，做统一的物理动作控制**。几十小时无结构数据。
- **这是「LLM 和物理控制器共享离散潜空间」的第一个完整 demo**，比 GR00T 早一年多：
  - 通用跟踪控制（视频姿态估计、动作生成模型的输出都能跟）
  - 用 GPT 框架从自然语言生成物理动作
  - **和 LLM 做 in-context learning 集成**，处理抽象任务
- **对你的意义**：你的核心架构主张，图形学圈 2023 年就验证过了。好消息是它证明了可行；坏消息是新颖性不在架构本身。

### GPC ★★★这是和我们 tokenizer 最像的一篇，必读
- **Large-Scale Generative Pretraining for Transferable Motor Control**，SIGGRAPH 2026，arXiv [2606.29148](https://arxiv.org/abs/2606.29148)
- **就是 MoConVQ 的继承者，而且换成了 FSQ。**
- 配置：**40 个 token × 9 个量化级别**，隐式码本大小 9⁴⁰。量化用 `⌊⌊L/2⌋·tanh(z)⌉`，直通梯度。
- **关键做法：encoder + FSQ + decoder 用 PPO 端到端训，奖励是物理仿真器里的动作跟踪奖励。**
  - 也就是说，离散码是为**物理可执行性**优化的，不是为运动学重建优化的。
  - 消融证明这很重要：端到端 RL 训 FSQ encoder 是 **99.98% 成功率**，改成运动学预训练只有 99.03%。
- 上层：GPT 式 transformer 解码器，对**分组 token**（G=5，连续 5 个 L 进制 token 合成一个更大词表的码）做自回归。推理用 nucleus sampling 保多样性。
- 结果：Bones 数据集（680 小时）**99.98% 跟踪成功率，MPJPE 34.90mm**；AMASS（40 小时）99.51%，44.43mm。
- **FSQ vs VQ-VAE**：Bones 上 FSQ 99.98%/34.90mm，VQ 99.94%/37.92mm；AMASS 上 FSQ 99.51%/44.43mm，VQ 99.30%/59.28mm。**质量打平或更好，但完全不需要码本塌缩处理和死码重初始化。**
- **涌现恢复行为**：被扰动时会自动切换到摔倒恢复或翻滚，**没有为这些技能设计任何奖励**。
- CoLA 微调：条件低秩适配 + FiLM 调制，加不到 1% 参数就能适配下游任务。
- 局限：主要是移动类任务，没有人-物交互；只在 Isaac Gym 仿真里；文本条件化留作 future work。

**对你的意义**：这篇几乎就是你想做的 tokenizer，只是单本体。**但注意它的 GPT 只用交叉熵训，完全没做 RL 后训练**——「在物理原生 FSQ tokenizer 上做 RL 后训练」这个组合还空着。

### SCHUR / Humanoid-Union ★FSQ vs VQ 的定论
- arXiv [2511.09241](https://arxiv.org/html/2511.09241v1)
- 专门做了 FSQ 和 VQ 的 scaling 对比。码本从 256 加到 65536：
  - **FSQ 保持 >99% 码本利用率。**
  - **VQ 掉到 90% 以下，超过某个阈值后「明显塌缩」。**
  - FSQ 质量随码本增大单调提升，VQ 不是。
- 260+ 小时数据，MPJPE 提升 37%，FID 提升 25%。
- 局限：下游只是运动学 AMASS tracker，他们自己承认没有平衡先验。

### Hybrid Latent Representation ★对你有风险警示
- Eurographics 2025，arXiv [2503.12814](https://arxiv.org/abs/2503.12814)，[代码](https://github.com/jinseokbae/hybrid_latent_prior)
- 核心发现：**纯离散潜会抖**。他们的方案是离散（残差 VQ）+ 连续残差一起用。离散那部分负责抓住不同的模式、避免后验塌缩；连续残差负责把动作磨平滑。
- **这是对你 FSQ-only 方案的直接风险提示**。如果只有离散 token，输出动作可能有 jitter，物理控制器跟起来会很难受。准备好加一个连续残差通道。

### Spherical Latent Motion Prior
- arXiv [2603.01294](https://arxiv.org/pdf/2603.01294)（2026）
- 针对 PULSE 那种高斯 VAE 潜空间的问题提改进，用球面潜。说明这条线 2026 年还在被优化。

---

## 二、统一 / 多面手全身控制器

### MaskedMimic ★和我们的稀疏约束接口几乎一样
- SIGGRAPH Asia 2024，arXiv [2409.14393](https://arxiv.org/abs/2409.14393)，[项目页](https://xbpeng.github.io/projects/MaskedMimic/index.html)，NVIDIA + Xue Bin Peng
- **把物理角色控制整个重新表述成「动作补全（inpainting）」问题**。这就是我们做的事，但他们有物理。

两阶段：
1. 全约束控制器 π_FC：transformer，RL 训练，跟全身动作。约 300 亿仿真步，两周。
2. 部分约束控制器 π_PC：**条件 VAE**，用行为克隆 + DAgger 从 π_FC 蒸馏。约 100 亿步。

**控制接口（这是重点）**——用户可以任意组合：
- **任意关节任意时刻**：任何关节子集在任何未来时刻的 3D 位置和旋转。显式训了 6 个关键关节：双踝、骨盆、头、双手。
- 文本：XCLIP embedding。
- 物体：角色局部坐标下的 bounding box 八个角 + 物体类别。
- 地形高度图（自动观测，不用指定）。
- 运动原语：头部位置目标（走路径）、方向+速度对（摇杆）。

VAE 三件套：
- prior ρ(z | s, g_partial)：transformer encoder，只看部分约束。
- encoder E(z | s, g_full)：只训练时用，看完整目标。**设计成 prior 均值的残差**，保证 encoder 的嵌入不会跑离 prior 太远。
- decoder D(a | s, z)：全连接，出 PD 目标。

训练技巧（值得抄）：
- **掩码在时间上要连贯**：一个关节在 t 时刻被 mask 掉，接下来几帧要继续 mask。不能每帧重新随机。论文说随机 mask 会造成「跨帧歧义」，泛化会烂掉。
- KL 退火：α 从 0.0001 线性升到 0.01。
- **每个 episode 固定一次重参数化噪声 ε**，不是每步都采。为了时间一致性。

结果：VR 跟踪（只有头+手）好过 PULSE、ASE、CALM。不平地形（楼梯、斜坡、碎石）都稳。能坐没见过的家具。

**承认的失败**：
1. 物理路线比纯运动学方法糙，不够平滑。
2. 多步任务还得手工设计有限状态机。
3. **完全新的动作类型不行**。而且「early termination 阻止了它探索 OOD 状态」——所以它压根没学过怎么从 OOD 里爬回来。
4. 文本控制精度不如显式关节目标。

**对你的意义**：稀疏约束接口这件事 MaskedMimic 已经做完了，而且带物理。你的差异点只能是：跨本体 + 离散 token + 硬可行性保证。第 3 条失败是你可以攻的点。

### HOVER
- arXiv [2410.21229](https://arxiv.org/abs/2410.21229)，[项目页](https://hover-versatile-humanoid.github.io/)，NVIDIA + CMU + Berkeley
- 一个 policy 管所有控制模式。**只有 150 万参数**。
- 统一多模态指令空间，用**掩码指令向量**表示：运动学位置跟踪、局部关节角跟踪、根部速度跟踪，都编码在同一个向量里，用 mask 选择当前模式。
- 支持的模式：头+手位姿（Apple Vision Pro）、全身位姿（MoCap/RGB）、全身关节角（外骨骼）、根速度（摇杆）。
- oracle teacher（有特权信息）→ student 蒸馏。
- **为什么蒸馏比训一堆专家好**：多模态共享表示带来跨任务正则化，隐式知识迁移。单一 policy 反而比每个模式单独训的专家强。
- 缺点：模式边界处快速切换时会脆。

**对你的意义**：掩码指令向量 = 你的稀疏约束接口的机器人版本，已经在真机上跑通了。注意它是**连续的掩码向量**，不是 token。

### SONIC ★★最重要的一篇，直接是我们的下游
- arXiv [2511.07820](https://arxiv.org/abs/2511.07820)，NVIDIA GEAR
- 代码 [github.com/NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)，文档 [nvlabs.github.io/GR00T-WholeBodyControl](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- 2026-02-19 开源（GEAR-SONIC）。2026-06-16 又放了低延迟遥操作 checkpoint（4 帧 SMPL 前瞻）。

规模（scaling 曲线很干净）：
| 参数量 | 1.2M → 16M → 42M |
|---|---|
| 数据帧 | 4M → 10M → 22M → 100M+（611 小时） |
| 算力 | 2K → 9K → 21K GPU 小时（16 → 32 → 128 卡） |
| 成功率 | 98.0% → 99.6%（测试集，OOD 内容） |
| MPJPE | 27.7mm → 23.8mm（真机部署 25.7mm） |

**核心：unified token space —— 用的就是 FSQ**

> "universal token 是一个 D_z 维向量，每维 L_z 个量化级别"，量化器明确用 **FSQ (Mentzer et al., 2023)**。

具体配置：**FSQ-32-32**，32 个量化级别 × 每 token 16 维，**两个 token 拼成 64 维通用 token**。VLA 任务里完整动作是 **78 维 = 64 维 token + 14 维手部关节**。

- 三个 encoder（MLP，隐层 [2048,1024,512,512]）：
  - 机器人动作 encoder
  - 人体动作 encoder E_h（吃 SMPL）
  - 混合 encoder E_m（吃当前帧稀疏上半身关键点：头+双手，加上未来 10 帧的下半身机器人动作）
- 控制 decoder（隐层 [2048,2048,1024,1024,512,512]）→ 29 维电机指令。
- 动作 decoder 做辅助监督重建。

**四个损失一起训**：`L = L_ppo + L_recon + L_token + L_cycle`
- `L_token = ||z_r − z_h||²`，**显式把机器人 token 和人体 token 拉到一起**。这就是跨本体对齐。
- `L_recon` 起到隐式重定向（implicit retargeting）的作用——不用显式 retarget。
- `L_cycle` 循环一致性。
- **tokenizer 和 RL 策略是联合训练的，不是分开预训练。**

**VLA 怎么接进来（版本之间变了，务必看 v3）**：

- **v1/v2**：VLA 输出「三个上半身位姿（头+双手腕）、腰高、导航指令」这种遥操作格式信号，再经 kinematic planner 和 hybrid encoder 编成 token。中间隔着一层手工接口。
- **v3**：**VLA 直接预测 78 维动作 = 64 维通用运动 token + 14 维手部关节，直接由通用控制策略解码。** 手工接口没了。

**而且 v3 给了那个决定性的消融**（见开头「先说结论」）：VLA 预测 FSQ token vs 预测 SMPL 位姿，平均 68% vs 27%，最难的任务 60% vs 0%。原因是连续高维位姿空间会把小误差放大成跟踪失败，离散低维 token 好学。

结果：真机 50 条轨迹 100% 成功率。泛化到没见过的动作。

**局限（原文）**：安全性、柔顺性、能效没有形式化处理；部署时的噪声输入没解决。

**Future work（原文）**：
> "探索 planner、tokenizer 和 policy 的**联合训练**，以缩小模态差距（modality gaps）。"

**对你的意义**：
- 好消息：下游控制器有了，开源，接口本来就是 FSQ token 空间，你的 tokenizer 有地方接。而且「VLM 说 token 比说位姿好」这件事**别人已经替你证明了**，你不用再花力气论证。
- 坏消息：**「VLM 直接吐 token 驱动物理控制器」这个架构已经被实现并验证了。**这不再是你的新颖点。
- 机会：SONIC 的 token 空间是**同一台 G1 上不同输入模态**（机器人动作/人体 SMPL/混合）的对齐，**不是跨机器人身体的对齐**。零样本换机器人这件事他们没做。

### 其他 WBC

- **H2O / OmniH2O**（[omni.human2humanoid.com](https://omni.human2humanoid.com/)）：用运动学位姿当通用控制接口。已知问题：SMPL 虚拟人能做的极端动作真机做不到，得靠过滤，这个 gap 一直在。
- **ExBody2**（arXiv [2412.13196](https://arxiv.org/abs/2412.13196)，[项目页](https://exbody2.github.io/)）：通用全身跟踪，任意参考动作输入。
- **OOD 失败的具体例子**（值得引用来支撑你的动机）：tracker 能跟住的 OOD 动作有 hip-hop、鞠躬、剑术弓步、回旋踢；跟不住的是**僵尸爬（超出机器人运动学极限）**和**盘腿坐（需要持续地面接触，离训练分布太远）**。
- **RGMT / Robust and Generalized Humanoid Motion Tracking**（arXiv [2601.23080](https://arxiv.org/abs/2601.23080)，2026-01，[项目页](https://zeonsunlightyu.github.io/RGMT.github.io/)）：直指「参考动作重定向后有噪声和不一致，闭环执行会把局部缺陷放大，导致漂移或失败」——**这就是你说的『tracker 出分布然后乱挥』的正式表述**。他们的解法：因果时序 encoder + 多头交叉注意力指令 encoder，按当前动力学选择性聚合上下文窗口；再加摔倒恢复课程（随机不稳定初始化 + 退火的向上助力）。只要 3.5 小时数据，单阶段端到端训练，不用蒸馏。
- **AnyBody**（arXiv [2606.29209](https://arxiv.org/pdf/2606.29209)，2026）：任意关键点引导的自由形式全身控制。和 MaskedMimic 同思路的机器人版。
- **BAT**（arXiv [2604.01064](https://arxiv.org/pdf/2604.01064)）：在线策略切换平衡敏捷性和稳定性，长时程全身控制。
- **FIRM / Unified Humanoid Fall-Safety Policy**（arXiv [2511.07407](https://arxiv.org/html/2511.07407)）：见第四节，字典检索恢复。

---

## 三、VLM 到控制器的共享表示

**这一节的核心问题：上层给下层的是潜还是 token？端到端联训还是冻结接口？有没有证据说端到端更好？**

### 先给答案

**没有任何一篇论文的标题是「我们试了端到端联训 VLA+WBC，失败了」。**所有证据都是消融形态的。但证据链非常清楚，而且结论是**两头都不对**：

1. **把一个随机初始化的连续动作头端到端反传进预训练 VLM，会实测破坏 VLM 的语义知识。**
   - VLM2VLA（arXiv [2509.22195](https://arxiv.org/abs/2509.22195)）那张表最狠：

   | 模型 | MMMU | MMStar | TextVQA |
   |---|---|---|---|
   | Prismatic VLM 7B（原始） | 35.0 | 38.8 | 42.5 |
   | **OpenVLA 7B**（动作 token 微调） | 26.3 | **0** | **0** |
   | **ECoT 7B** | 26.6 | **0** | **0** |
   | VLM2VLA 12B（动作写成英文 + LoRA） | 42.7 | 48.0 | 64.9 |

   - InstructVLA（arXiv [2507.17520](https://arxiv.org/abs/2507.17520)）复现同样结果：**OpenVLA 在 MM-Vet / MMStar / TextVQA / DocVQA 上全是 0.0。**
   - ReVLA（arXiv [2409.15250](https://arxiv.org/abs/2409.15250)）给了机制层面的解释：动作微调后 **DINOv2 编码器塌缩**，DPT 深度回归输出变成常数。

2. **但把 VLM 完全冻住同样不行。** Physical Intelligence 的 Knowledge Insulation（arXiv [2505.23705](https://arxiv.org/abs/2505.23705)，NeurIPS 2025）原文：
   > 「**冻结主干不是知识隔离的可行选项**，因为预训练模型里的表示对机器人任务是不够的，会导致低性能。」
   - Fig 4a：**冻结主干在 items-in-drawer 上是 0%。**衬衫折叠也失败。

3. **赢的配方在中间：主干要训，但只用离散 token / 语言形态的损失去训；连续动作专家用 stop-gradient 隔开。**
   - KI 的机制描述：「来自动作专家的梯度导致不利的学习动态，不仅让学习慢很多，还**让 VLM 主干丢失一部分网络规模预训练获得的知识**。」原因是动作专家是随机初始化的，它的 flow matching L2 信号污染预训练权重。
   - 两个修复缺一不可：(a) **stop-gradient**，动作 token 能读主干特征但不能反传进去；(b) **用离散 FAST 动作 token 做下一 token 预测来联合训主干**，给它一个没被污染的机器人学习信号。
   - 代价数字：π₀ 需要 **7.5 倍的训练步数**才能达到可比性能。KI 收敛和 π₀-FAST 一样快，还保住了 π₀ 的快推理。
   - 语言遵循（items in drawer，读图约值）：π₀ 约 20–40%，无 stop-grad 的联训约 30–45%，**KI 约 70–75%**。衬衫折叠：**π₀.₅+KI 约 50% vs π₀ 接近 0%。**

**所以「离散 token」的首要作用不是表示选择，是梯度整形选择。**它让主干的学习信号留在它预训练时的那个模态里。这是你论点里最该强调、但你可能还没意识到的一层。

---

### LeVERB ★最贴近「潜动词词表」，但要看清它的消融
- arXiv [2506.13751](https://arxiv.org/abs/2506.13751)（2025-06），[OpenReview](https://openreview.net/forum?id=dLgFqCaXNg)
- 问题陈述值得直接引用：现有 VLA 假设底层控制器已经很好，用的是**手工设计的动作「词表」**（末端位姿、根部速度）。这把研究限制在准静态任务上，做不了人形需要的敏捷全身行为。
- **接口：256 维连续潜「动词」。不是离散码本。**残差 CVAE，后验均值 `μ = μ_ρ + μ_E`（μ_ρ 来自视觉语言策略，μ_E 来自运动学 encoder）。**进程之间就是一个 ROS2 topic，一根单向的线。**
- 速率和参数：LeVERB-VL（System 2）10Hz，约 102.56M 可训练参数，**冻结的 SigLIP ViT-B/16** 视觉和文本编码器。LeVERB-A（System 1）50Hz，**极小**——2 层 transformer，4 头，隐层 128，输出关节位置目标。机上状态 500Hz。潜和本体感觉作为**两个独立 token** 进 LeVERB-A。G1 机上 CPU 跑 ONNX/C++，VL 侧在外接 4090。
- **训练：严格两阶段，阶段之间接口冻结。**
  - 阶段 1：LeVERB-VL 当 CVAE 训，轨迹重建 + KL（β₁=1e-1）+ 梯度反转的对抗判别器（β₂=5e-4），把「视觉语言数据」和「纯语言数据」的潜对齐。
  - 阶段 2：**冻结潜空间**，用 PPO teacher（动作跟踪奖励）→ DAgger 学生蒸馏训 LeVERB-A。
  - 解耦的理由：避免在并行 RL 仿真里做光真实感渲染，而端到端方法必须那么做。
- **数字**（LeVERB-Bench，150+ 任务 10 类，各 20 次）：总体 **58.5%**；简单视觉导航 80%；背朝带干扰物导航 30%；Sit 100%；Locomotion 100%。
- **消融（这段最重要，别被宣传数字骗了）**：
  - 无判别器 33.0%
  - **无运动学 encoder（≈ 没有潜瓶颈、直接条件化）53.0%**
  - 无 VL 25.5%
  - 无采样（无 CVAE 结构）7.5%
  - 宣传的「7.8 倍」是 58.5 vs 7.5，**是打那个退化变体**。
  - **⚠️ 最像「没有潜瓶颈」的那个消融只差 5.5 分（58.5 vs 53.0）。而且论文里根本没有真正的端到端联训 baseline。**
- **局限（原文 §6）**：「受限于缺少真正的长期历史和规划能力……LeVERB 的时域窗口有限，只预测未来几秒。另外 LeVERB-A 缺少快速视觉反馈回路。」
- **另一个必须知道的坑**：真机 G1 结果是**开环重放仿真里录下的潜动词**，不是闭环视觉在环。

### PhysiFlow ★LeVERB 的继任者，位置最有意思
- arXiv [2603.05410](https://arxiv.org/abs/2603.05410)（2026-03，上交）
- 三层：SigLIP+LoRA → **256 维语义-运动潜 z_vl @10Hz**；Gemma 基座的 flow matching 动作专家吃 z_vl + 38 维 G1 状态 → 50Hz 动作块；**一个预训练冻结的 RL 动作 tracker** 执行。
- **novelty 在这里：把跟踪误差穿过冻结的 tracker 反传回去，微调 flow 模型。**原文：「执行动作和参考动作之间的误差……被反传去微调 flow matching 模型，确保它生成的动作是物理感知的。」
- **也就是：底层冻结，但中层是梯度耦合的。**这是「冻结 vs 端到端」之外的第三条路。
- 数字：总体 **74.9% vs LeVERB 65.0%**（本地复现）；长导航 63.6 vs 31.2；导航+坐 18.1 vs 5.8。LeVERB 在短导航仍赢（78.5 vs 71.7）。真机同样是开环执行。

**⚠️ 对你的意义**：PhysiFlow 这招（穿过冻结控制器反传，让生成器变得物理感知）**是你最该考虑的架构 B 变体**，比硬对齐码本便宜得多。

### GR00T N1 → N1.5 → N1.6 → N1.7 ★NVIDIA 自己的自然实验，结论偏向冻结
- N1：arXiv [2503.14734](https://arxiv.org/abs/2503.14734)（2025-03）。2.2B 总参数，1.34B Eagle-2 VLM。
  - **接口：从 LLM 第 12 层（中间层，不是最后一层）取连续 VL token 嵌入**，由交替的 cross-/self-attention DiT 块消费。
  - 明确消融：「用**中间层而非最后一层** LLM 嵌入，推理更快且下游策略成功率更高。」
  - **部分冻结**：LLM/文本部分在预训练和后训练都**冻结**。System 2 @10Hz，System 1 DiT @120Hz。
  - 结果：真机平均 76.8%（比 Diffusion Policy 高 30.4 分）。
- **N1.5**（2025-05，[页面](https://research.nvidia.com/labs/gear/gr00t-n1_5/)）：**已发表证据里最强的冻结接口数据点。**
  - 原文：「**VLM 模型在预训练和微调期间都是冻结的。**」加上简化的 adapter MLP + layernorm，FLARE 潜未来对齐损失，DreamGen 合成数据。
  - **语言遵循 46.6%（N1）→ 93.3%（N1.5）**，同样的 GR-1 水果任务。few-shot（每任务 30 示教）17.4% → 47.5%。人类视频微调后新物体 0% → 55%。
  - ⚠️ 注意：NVIDIA 把增益归给这一整套组合，**没有发表单独的冻结消融**，所以 93.3% 和 FLARE/DreamGen 是混淆的。
- **N1.6**：Cosmos-2B VLM 变体；**去掉** N1.5 那个 4 层 transformer adapter，**解冻 LLM 顶部 4 层**；DiT 16→32 层。没有公开定量结果或消融。
- **N1.7**（2026-04 早期访问，[代码](https://github.com/NVIDIA/Isaac-GR00T)）：仓库 changelog 原文：
  - VLM 主干 Eagle → `nvidia/Cosmos-Reason2-2B`（Qwen3-VL）；`select_layer` 16 → 12；**`tune_top_llm_layers` 4 → 0——N1.7 把 LLM 又完全冻回去了**，撤销了 N1.6 的部分解冻。
  - DiT 32 → 16 层；`action_horizon` 16 → 40；状态/动作维 29 → 132。
  - **这个来回反转是整组证据里最能说明问题的信号。**
- **GR00T ↔ WBC**（[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)）：
  - 出两个控制器：**Decoupled WBC**（RL 下半身 + IK 上半身，经典手工指令接口）给 N1.5/N1.6 用；**GEAR-SONIC** 给 N1.7 用。
  - **N1.7 的 VLA 预测 78 维动作 = 64 维 SONIC 运动 token + 左右手各 7 维，速率 2.5Hz。SONIC 在 50Hz 把潜解码成全身关节指令。**
  - 仓库自己的说法：VLA「**只需要推理做什么，SONIC 负责怎么做**」。
  - 文档没说 VLA 微调时 SONIC 是否冻结，但 collect→finetune→deploy 工作流里只微调 VLA。

### Figure Helix ★最强的端到端潜向量反例
- [Helix v1](https://www.figure.ai/news/helix)（2025-02），[Helix 02](https://www.figure.ai/news/helix-02)（2026-01）
- **接口：单个连续潜向量。**原文：S2「把所有语义任务相关信息蒸馏成**一个连续潜向量**，传给 S1」。机制上：「投影到 S1 的 token 空间，沿序列维和 S1 视觉主干的视觉特征**拼接**」。
- **⚠️ 潜向量维度从未公开。**任何地方看到具体数字都是编的。
- S2 = 7B 开源权重 VLM @7–9Hz；S1 = **80M** cross-attention transformer @**200Hz**，控制 **35 自由度**上半身。
- **联合端到端训练。**原文：「**梯度从 S1 反传进 S2，穿过那个潜通信向量**，允许两个组件联合优化。」单阶段，单套权重。训练时还**在 S1 和 S2 的输入之间注入时间偏移**，校准到部署时的延迟差。
- **他们自己给的「为什么用潜而不是 token」的理由**：潜「直接输出高维动作空间的连续控制，**避开了之前 VLA 用的复杂动作 token 化方案**，那些方案在高维人形控制上面临扩展挑战。」以及：「**解耦 S1 和 S2 让我们能分别迭代，不受『必须找到统一观测空间或动作表示』的约束。**」
- ⚠️ 注意这个解耦是**部署/架构**选择（S2 异步在后台进程写共享内存潜，S1 在第二块 GPU 上读最新的），**不是训练选择**。
- [Helix logistics](https://www.figure.ai/news/scaling-helix-logistics)（2025-06）：4.05 秒/包（从约 5.0 秒）；条码朝向 ~70% → ~95%；数据从 10 小时到 60 小时：6.84 → 4.31 秒，88.2% → 94.4%。**所有改进都在 S1 内部做的**——「分别迭代」这个主张的实证。
- **Helix 02**（2026-01）：变三层了。
  - S2 出「**一串语义潜**」；S1 还是 transformer，条件于 S2 潜，但**现在输出全身关节目标**；**S0 = 10M 参数学出来的 WBC @1kHz**，在仿真里用 >1000 小时关节级重定向人类动作、>20 万并行环境训出来，**取代了 109,504 行手写 C++**。
  - **所以 S1→S0 是显式的关节目标跟踪接口，不是潜。**S0 是否和 S1 梯度耦合，没说。
  - Demo：一次连续 4 分钟洗碗机装卸，61 个 loco-manipulation 动作，无重置。**没有成功率、没有 baseline、没有消融。**

### π0 / π0.5 / Hi Robot（Physical Intelligence）
- **FAST**（arXiv [2501.09747](https://arxiv.org/abs/2501.09747)）：归一化 → 每个动作维做 DCT → 量化 → **BPE，词表 1024**。每条手臂每 1 秒块约 30 个 token，和控制频率无关。压缩比 1.75×（BridgeV2 5Hz）到 13.2×（叠 T 恤 50Hz）。
  - **π₀-FAST 比扩散 π₀ 少用 5 倍 GPU 小时，桌面清理少 3 倍步数，而且语言遵循更好**——扩散 π₀ 在 DROID 上「经常忽略语言指令」。
- **π₀**（arXiv [2410.24164](https://arxiv.org/abs/2410.24164)）：3.3B = 3B PaliGemma + **300M 动作专家**，一个 transformer 里的 MoE，「权重只通过 transformer 的自注意力层交互」。梯度就是从这里到达 VLM 的——这正是 KI 后来要 stop-gradient 的地方。
- **π₀.₅**（arXiv [2504.16054](https://arxiv.org/abs/2504.16054)）：**分层在一个模型内部**：`π(动作, 子任务 | 观测, 任务) = π(动作 | 观测, 子任务) · π(子任务 | 观测, 任务)`，子任务先以**自然语言文本**发出。
  - 先用离散 FAST token 预训练 280k 步，后训练再挂 flow matching 动作专家。
  - 消融：π₀.₅ ≫ π₀-FAST+Flow（**验证了离散 token 预训练**）；网络数据 91% vs 73%（OOD 物体）。
- **Hi Robot**（arXiv [2502.19417](https://arxiv.org/abs/2502.19417)）：显式两层，**接口就是自然语言**。原文：「训练过程**解耦**了高层和低层模型，**除了通过训练样本，它们互相不知道对方的能力**。」
  - 指令准确率 **76% vs 扁平 VLA 36% vs GPT-4o 30%**；任务进度 81% vs 44% vs 64%。做三明治 83% vs 34%。
  - **干净接口的明码代价**：「解耦架构使高层策略无法观测低层任务完成的成功与否。」

### Gemini Robotics ★接口最干净的一派
- **GR 1.5 + ER 1.5**（arXiv [2510.03342](https://arxiv.org/abs/2510.03342)）
  - **接口就是自然语言，模型卡 §2.1 明说**：动作模型「作为一个专用**工具**提供给 orchestrator，通过**开放词表自然语言**接收指令」。**没有潜通道。**
  - GR 1.5 内部还有一层：动作之前先出**自然语言思考轨迹**。
  - **Motion Transfer** 做跨本体（ALOHA-2、双臂 Franka、Apollo 人形）——**内部机制未公开**，不要假设它是共享潜动作空间。
  - 数字：长时程总失败率 44.5%（Gemini 2.5 Flash orchestrator）→ **22.0%**（GR-ER 1.5）；规划失败 25.5% → 9%。
  - **局限**：「虽然 GR 1.5 展示了新水平的泛化，**它的灵巧度仍和上一代持平**」；MT「在本体差距大得多时可能不那么有效，比如高度不相似的人形和其他机器人形态之间」。
- **Gemini Robotics 2 / ER 2**（2026-07-30，[博客](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)）：ER 2「**把运动执行交给任意给定的低层 VLA 模型**」；开发者「**把低层控制接口——比如 VLA 模型或导航 API——声明为工具**」。
  - 新的 **ASIMOV-Agentic** benchmark 衡量「具身推理 agent **拒绝来自 VLA 的不安全工具调用**的能力」——**安全在接口层强制执行，这只有符号接口才做得到。**
  - 夹爪灵巧度 74.2–89.6%；**多指灵巧度 32–92%，被明确列为开放问题**。没有技术报告或 arXiv 论文。

### Wall-OSS-0.5 ★对「冻结」最响亮的公开异议
- arXiv [2605.30877](https://arxiv.org/abs/2605.30877)（2026-05，自变量机器人）
- 4B VLA 建在 3B VLM 上，>20 个本体，每 epoch >100 万轨迹。
- 原文：「**离散动作预测把 VLM 原生的强梯度路由进主干**……连续 flow matching 充当**部署时的动作接口**。」他们管动作 token 交叉熵叫「**梯度桥（the gradient bridge）**」。
- 反对冻结的论证：「**冻结或截断梯度保住了 VLM 先验，但代价是阻止了精确的动作目标去塑造那个大的预训练主干。**」并点名 π₀ 那一家是 cross-attend 到「一个基本冻结的主干」。
- 他们的 MoT 划分是「**路由分解而非梯度隔离**」；单阶段训练；用学出来的 Vision-Aligned RVQ 在 delta-action 空间里取代 FAST。
- **15 个真实任务平均任务进度 60.5%，比 π₀.₅ 高 17.5%**；具身 grounding 上 +21.8% 且视觉语言能力无退化。

### RDT 一系
- **RDT-1B**（ICLR 2025，arXiv [2410.07864](https://arxiv.org/abs/2410.07864)）：1.2B DiT，**冻结 SigLIP + 冻结 T5-XXL**。「物理可解释的统一动作空间」= **手工设计的 128 维向量**，每个槽位有固定物理含义，按本体补零——**不是学出来的**。68.2% vs 从头训 34.8%、ACT 12.2%、OpenVLA 1.7%。
- **RDT2**（arXiv [2602.03310](https://arxiv.org/abs/2602.03310)，2026-02）★**领域内最明确的「先离散 token 再冻结主干」设计**：
  - 阶段 1：7B Qwen2.5-VL，动作块用时域 CNN 压缩后**残差 VQ** 量化；**把 VLM 词表里 1024 个最低频条目征用为动作 token**；128k 次迭代的普通下一 token 交叉熵。
  - 阶段 2：「我们**冻结阶段 1 预训练的 VLA 主干**，训一个专门的动作专家」——400M 的 RDT-1B 变体，flow matching，「cross-attention 引入冻结 VLA 主干**每一层的潜特征**」。
  - 阶段 3：一步扩散蒸馏，前两阶段都冻结。
  - **在 >10000 小时 UMI 手持夹爪人类数据上训，零机器人数据。**
  - **RVQ 在同等误差下比 FAST 少约 2/3 的 token**；23.0Hz vs baseline 的 2.7/1.6Hz。

### 2026 人形 VLA ↔ WBC
- **WholeBodyVLA**（ICLR 2026，arXiv [2512.11047](https://arxiv.org/abs/2512.11047)，[代码](https://github.com/OpenDriveLab/WholebodyVLA)）★**人形领域控制得最好的冻结接口结果**
  - **接口：离散 VQ-VAE 潜动作码本**，从**无动作的第一视角视频**学（300 小时人类移动 + AgiBot World）。**两个独立码本**（操作 `C^mani`、移动 `C^loco`）——单一共享 LAM 在混合数据上被报告为次优，因为操作数据的图像变化被手臂主导，移动数据是环境相对相机在动，一起训注意力会打架。
  - 轻量 decoder `a_t = f(ĉ^mani, ĉ^loco, s_t)` 把 token 落地成上半身关节角 + **离散移动指令 `{-1,0,1}³ × ℝ`**（不是连续速度，理由是显式启停语义更精确）。
  - **低层冻结**：「LMO 控制器……在仿真里单独训练，在遥操作数据采集和最终部署期间**保持固定**。」VLA ~10Hz，LMO 50Hz 只吃本体感觉。
  - **数字：78.0% vs 模块化设计 64.0% vs OpenVLA-OFT+LMO 56.7% vs GR00T+LMO 42.0%。**
  - **消融：去掉 LAM，78.0 → 39.3%（−38.7 分）**；只有操作 LAM 63.3%；共享单一 LAM 66.0%；用速度式 RL 代替 LMO 54.0%。
- **HALOMI**（arXiv [2606.18772](https://arxiv.org/html/2606.18772v1)，2026-06）：用**冻结的 BFM-Zero**（128 维潜）当低层。高层 VLA（π₀.₅ 主干）输出**左手/右手/头的相对位姿轨迹 + 夹爪信号**——**手工指令接口，不是潜**，单独训，再由冻结的 BFM-Zero 解码。
  - 袋子转移 90%，面包 85%，毛巾 80%，未见位置 60%。没有冻结 vs 联训的消融。
  - **失败：面包-盘子相对布局变化时成功率掉到 0/10。**
- **MotionWAM**（arXiv [2606.09215](https://arxiv.org/abs/2606.09215)）：明确攻击指令接口——「主流的分层范式里，高层操作策略只控制上半身，低层控制器**跟踪粗糙的基座指令**——把上下半身放进了**不一致的动作空间**。」换成统一运动潜 / 全身运动 token，条件于视频世界模型的中间去噪特征。**9 个真机 G1 任务上比 VLA baseline 高 >30 个绝对百分点。**
- **OpenHLM**（arXiv [2606.22174](https://arxiv.org/abs/2606.22174)，2026-06）：「不要语义接口」的立场——高层直接发 **10Hz 的全身参考关节指令**（34 维），低层 50Hz PD 跟踪。打败 GR00T N1.6 和 Ψ₀。
  - 接口消融：关节式全身遥操作 86.7/80.0/85.0% vs 解耦控制 66.7/93.3/不可达 vs VR 三点 40.0/不可达/不可达。关节空间 vs SMPL 重定向 88% vs 75%。
  - **⚠️ 关键负面结果：「动作 MSE 是机器人预训练价值的糟糕代理：两个动作 MSE 相同的模型在真机上表现可能非常不同。」**一步生成的验证 MSE 更低（0.007 vs 0.009），真机进度却差约 20 分。**这条直接决定你该测什么指标。**
- **AgiBot GO-1 / ViLLA**（arXiv [2503.06669](https://arxiv.org/abs/2503.06669)）：阶段 1 学 **k=4 个 VQ-VAE 离散潜动作 token/步**；阶段 2 拿它们当伪标签训潜规划器；**阶段 3「引入动作专家并与潜规划器联合训练」**。>60% 长时程真机任务，比 RDT 高 32%。LeVERB 明确引用它：「AGIbot 证明了**用潜接口能提升性能**。」

### 冻结技能层当低层解码器：更多先例

| 系统 | 接口 | 冻结？ | 关键数字 |
|---|---|---|---|
| **HOVER** | 掩码指令向量，3 种模式，每个 mask 位 ~Bernoulli(0.5) | 从 PPO 模仿 oracle 蒸馏（DAgger） | **在 32/32 个指标和模式上打败端到端训练的多模态 RL 策略** |
| **PULSE** | 32 维连续潜 | **明确冻结**：「冻结的 decoder D 连同仿真，现在可以当成新的动力学系统」 | VR 跟踪 **93.4%** vs ASE 37.6%、CALM 10.1%。消融：无可学先验 93.4→**18.1%**，无残差 60.8% |
| **TokenHSI**（CVPR 2025，arXiv [2503.19901](https://arxiv.org/abs/2503.19901)） | 共享本体感觉 token + 每任务 token，transformer 掩码 | 冻结本体 tokenizer/嵌入/动作头/transformer，只训新任务 tokenizer + 零初始化 adapter | **最大的冻结 vs 从头训差距：Climb+Carry 99.2±0.1%（冻结主干）vs 26.8±37.8%（从头）** |
| **Puppeteer**（ICLR 2025，arXiv [2405.18418](https://arxiv.org/abs/2405.18418)） | **15 维 [−1,1] 指令** = 未来参考动作的相对末端位置（头/手/脚）；低层控 56 自由度 | 「跟踪世界模型**全程冻结**……所有下游任务复用同一个跟踪模型」 | **⚠️ 奖励上端到端 TD-MPC2 打平或更好。负面结果是行为上的：端到端会 reward hack——「学会了滚上楼梯」。97.6% 的人类评测者更喜欢分层策略的动作。** |
| **ULTRA**（arXiv [2603.03279](https://arxiv.org/abs/2603.03279)） | 变分技能瓶颈 `z = z_prior + z_res`（PULSE + HOVER 式掩码） | **故意不冻结**——encoder 和 decoder 通过模仿**联合训练**，另加残差捷径把全身目标传给 decoder | 稠密跟踪 67.30% vs OmniRetarget 51.34%；OOD 70.57 vs 46.71；真机 73% 稠密 / 80–90% 稀疏 |
| **LCB**（IROS 2024，arXiv [2405.04798](https://arxiv.org/abs/2405.04798)） | 给 LLaVA 词表加一个可学的 **`<ACT>` token**，它的末层嵌入线性投影进低层策略的潜条件 | 低层在「冷启动」阶段冻结，然后选择性联合微调 | Language Table 推理 **72%** vs GPT-4V ~55%。论据：「通过用一个独立的 `<ACT>` token，我们不破坏 LLM 的核心语言生成和推理能力」 |

**⚠️ 三个诚实标记（你写 proposal 时别踩）**
1. **PULSE 的从头训 baseline 在成功率上打平、在误差上更好**（80.4 vs 88.6mm；真机 43.3 vs 68.4mm）。PULSE 赢在收敛速度和自然度，不是精度。
2. **LeVERB 的「有潜瓶颈 vs 直接条件化」只差 5.5 分，而且论文里没有端到端 baseline。**
3. **ULTRA（2026）故意联合训练技能瓶颈而不是冻结**，数字也很强。所以「冻结更好」不是定论。

**最被低估的一条**：Puppeteer 的结果之所以重要，是因为它跟奖励无关——端到端在**回报上打平甚至更好**，然后**通过滚上楼梯来 reward hack**，而 **97.6% 的人类更喜欢冻结 tracker 那版的动作**。配合 OpenHLM 的「动作 MSE 是糟糕代理」，规律是：**端到端耦合的代价出现在语义和自然度上，不出现在 loss 和成功率上**——正是最容易漏看、也最难发论文的地方。

### 其他 2026 人形 VLA
- **WholeBodyVLA**（ICLR 2026，arXiv [2512.11047](https://arxiv.org/html/2512.11047v1)，[代码](https://github.com/OpenDriveLab/WholebodyVLA)）：
  - 两个独立的 Latent Action Model（LAM），操作一个、移动一个。都是 VQ-VAE + DINOv2 特征，把帧间视觉变化编成离散 token。**分开训的理由**：操作数据的图像变化被手臂主导，移动数据是环境相对相机在动，一起训注意力会打架。
  - **控制器接口是「离散指令」**：前后、左右、偏航各取 ±1 或 0，加站立高度。不是连续速度。理由是显式的启停语义更精确。
  - VLA 跑 10Hz，LMO RL 策略跑 50Hz。
  - 结果：三个任务平均 78% 成功，GR00T 42%，模块化设计 64%。消融：潜学习贡献 +38.7%，LMO 离散指令比速度控制少 24% 失败。
- **MotionWAM**（arXiv [2606.09215](https://arxiv.org/abs/2606.09215)，2026-06，Mondo Robotics + HKUST）：
  - 批评当前主流：上下半身放在不一致的动作空间里，腿被降级成只管平衡。
  - 做法：**统一动作潜，预测全身 motion token**，一个动作空间同时覆盖移动、躯干、高度调节、脚部交互、手部操作。
  - 视频 DiT 和动作 DiT 通过中间去噪特征耦合，统一 flow matching 目标。
- **MotionVLA**（arXiv [2606.15142](https://arxiv.org/html/2606.15142)）
- **Humanoid-VLA**（arXiv [2502.14795](https://arxiv.org/pdf/2502.14795)）
- **HEX**（arXiv [2604.07993](https://arxiv.org/pdf/2604.07993)）：跨本体全身操作的 humanoid-aligned experts。
- **LangWBC**（RSS 2025，arXiv [2504.21738](https://arxiv.org/pdf/2504.21738)，[项目页](https://langwbc.github.io/)）：语言直接端到端到全身控制，CVAE 结构增加动作多样性和可组合性。真机 G1。是 Humanoid-LLA 的 baseline（76.0%）。
- **From Language to Locomotion**（arXiv [2510.14952](https://arxiv.org/pdf/2510.14952)）：免重定向，语言 → 动作潜 → RL 控制器。局限是 OOD 指令上会退化。

---

## 四、行为基础模型和技能字典

你说的「出了分布就去上游字典里搜」，本质上就是这一块。**答案是：已经存在，而且效果不错，但有明确的天花板。**

### 先讲清楚 FB 表示是什么
- Touati & Ollivier 的前向-后向（FB）表示。核心把长期状态占用做低秩分解：
  `M_z(ds'|s,a) ≈ F(s,a,z)ᵀ B(s') ρ(ds')`
- 这个分解带来一个漂亮性质：**任何线性奖励的 Q 函数都塌缩成一个内积** `Q_r(s,a) = F(s,a,z)ᵀ z`。
- z 住在 ℝᵈ 里半径 √d 的超球面上。奖励只通过 z 进入，所以**一个任务在测试时被「编译」成一个向量，不用规划、不用梯度步**。
- **Does Zero-Shot RL Exist?**（ICLR 2023，arXiv [2209.14935](https://arxiv.org/abs/2209.14935)）：系统比较 FB 和 successor features。结论——能用，但**只达到任务专用 RL 的一部分**。
- **整个家族的承重局限**：零样本推断只有当奖励落在学到的特征张成的**线性空间**里才精确。落在外面的部分就是投影误差。**「需要搜索」恰恰就发生在这里。**

### FB-CPR / Meta Motivo ★
- arXiv [2504.11054](https://arxiv.org/abs/2504.11054)，NeurIPS 2024，[代码](https://github.com/facebookresearch/metamotivo)
- FB-CPR = Forward-Backward representations with Conditional-Policy Regularization。无监督 RL。
- 核心：**把轨迹、状态、奖励、策略全部嵌到同一个潜空间**。用一个**潜条件判别器 D(s,z)**（最优解是 `D* = p_M/(p_π + p_M)`）逼策略去覆盖无标签动捕数据集里的状态。演员损失约 `-E[F(s,a,z)ᵀz + α Q_D(s,a,z)]`。
- **就是这个判别器让 z 变成「行为嵌入」而不只是「任务嵌入」——数据集成了隐式的技能字典。**
- 三种 prompt 都是闭式的，不用优化：
  - 奖励：`z_r = (1/n) Σ r(s_i) B(s_i)`
  - 目标：`z_g = B(s_goal)`
  - 动作跟踪：`z_τ = (1/n) Σ B(s_i)`（参考片段上平均）
- 推理 ≈ 300 步一个 episode 用 12 秒。checkpoint 有 5 个 24.5M 和 1 个 288M。API 就是 `z = model.reward_inference(...)` 然后 `model.act(obs, z)`。
- **数字**：零样本不训练，聚合达到任务专用算法上限的 **73.4%**。奖励优化约 TD3 的 61%；目标到达 0.83 成功率（追平 Goal-TD3）；动作跟踪约 PHC 专家的 88%。**每一类都比 ASE 高 1.4 倍以上。**人类评测：83% 的奖励任务里被判比 TD3 更像人。
- **承认的失败（对你很关键）**：**离动捕数据集远了就急剧退化——论文点名「贴地动作、摔倒、站起来」是最差的区域。**只有本体感觉，没有感知/导航/物体交互。

**⚠️ 这条警告你必须记住：字典最薄的地方，恰恰就是你最需要它的地方——摔倒的时候。**

### BFM-Zero ★★这个最像你的想法
- arXiv [2511.04131](https://arxiv.org/abs/2511.04131)，ICLR 2026 poster，[项目页](https://lecar-lab.github.io/BFM-Zero/)
- **第一个用 off-policy 无监督 RL 训真机人形 BFM 的工作**。真机 Unitree G1。
- 学一个统一潜 z，**把动作、目标、奖励嵌到同一个空间**。前向模型 F(s,a,z) + 后向模型 B(s) + z 条件策略 π_z。
- **三种 prompt 方式全部在推理时算出来，不用重训**：
  - 目标：`z = B(s_g)`
  - 动作：`z_t = Σ λⁿ B(s_{t+n})`（未来动作窗口）
  - 奖励：`z = Σ B(s_i) r(s_i)`（按奖励加权状态）
- 训练时**完全没有任务奖励**，纯无监督 RL。预训练数据是模拟器里的无奖励在线交互 + 离线无标签人类动作（LAFAN1）。
- **恢复能力（对你最关键）**：真机演示了从重推、踢躯干、拽倒地、踢腿中恢复。论文说是「自然而温和的恢复」。剧烈扰动下会**涌现出跑步**这种行为。被拽倒在地能优雅爬起。
- **潜空间可搜索/可插值**：两个潜向量之间做球面线性插值，会产生「语义层面有意义的变化」——比如左横移和右横移之间插值，得到中间的走路模式。
- 还支持 few-shot 优化式适配。

**补充关键细节**：
- 规模 440.5M 参数。三个 critic（F / Q_D / Q_R）各约 135M（6 个残差块，2048 隐层），actor 31.9M，判别器 2.9M，**后向映射 B 只有 0.2M**。
- sim2real 三件套：奖励整形、域随机化、**历史依赖的非对称 actor/critic**（消融证明历史这一项对域随机化下的鲁棒性是必需的）。
- 动作 prompt 用的是**前瞻窗口**：`z_t = Σ_{t'=t}^{t+H} B(s_{t'})`，不是单帧。
- **测试时搜索（这段最重要）**：零样本是闭式的，不搜。few-shot 才真的在 z 上做优化：
  - 单姿态：CEM，从 `z_0 = B(s_g, o_g)` 初始化，20 轮迭代最大化 J(z)。
  - 轨迹：DIAL-MPC 式的双循环退火零阶优化，优化整个序列 `z_{t:t+H-1}`，2048 个粒子，τ₁=0.85，τ₂=0.9，6 轮。
  - **明确说明：没有梯度搜索，也没有最近邻检索。**
- **OOD 证据**：只在 LAFAN1 上训，在 **AMASS（175 个 CMU 动作 + 10 个手挑姿态）**上评估，泛化成功。AMASS 跟踪 1.0342m vs 分布内 1.0789m——**跨数据集几乎没退化**。对有遮挡不连续的视频提取动作也鲁棒。
- **不可行的目标姿态被「优雅处理」**——策略收敛到一个自然可行的构型，而不是发散。**这是一个隐式的「投影回可行流形」，已经在策略内部发生了。**
- few-shot 数字：轨迹优化带来 **29.1% 跟踪误差下降**；负重 4kg 单腿平衡，零样本约 5 秒就失败，CEM 适配后 **>15 秒**。
- 跨本体：**同一套超参数直接迁移到 Booster T1。**
- **承认的失败**：(1) 行为范围被训练动作数据集卡死；(2) sim2real 还需要更好的在线适配；(3) 测试时适配只是初步探索；(4) **稀疏奖励推断很脆——重复跑会得到不同策略，有些非常差**；(5) 奖励推断在域随机化下退化远比跟踪严重。

**⚠️ 第 4 条是对你的直接警告：如果你的字典搜索是被一个稀疏的「我是不是要摔了」信号触发的，那正好落在最脆的区间。**

### 测试时在 z 上搜索——最接近你想法的现成机器

- **Fast Adaptation with Behavioral Foundation Models**（RLC 2025，arXiv [2504.07896](https://arxiv.org/abs/2504.07896)）★
  - 核心发现，说得很直白：**BFM 的策略空间里早就装着比零样本推断找到的好得多的行为。**闭式推断只是没找出来。
  - 两个算法都不动网络权重，只在低维 {π_z} 空间里搜：
    - **ReLA**：off-policy actor-critic，小残差 critic 纠正奖励投影误差，successor features 冻结，z 用策略梯度更新。
    - **LoLA**：on-policy，只有 actor，固定时域 rollout 配 BFM 冻结的终端价值函数，z 参数化成潜空间上的高斯，留一法基线。
  - 收益：**DMC 上比零样本好 10–30%，HumEnv 上 40–50%，只要几个 episode。**
  - 失败：ReLA 在分布漂移下非单调；LoLA 需要特权状态重置；**奖励落在特征线性空间外时两个都失败。**
- **BFMTrack**（arXiv [2606.25056](https://arxiv.org/html/2606.25056v1)）★
  - 在 FB-CPR 上做**潜序列优化**：`μ_t = B(g_t)` 初始化，用**粉红噪声**（β=1）采 N 条潜序列，仿真里 rollout，REINFORCE 式更新配留一法基线，再投影回超球面。
  - MPJPE 3.4cm vs 闭式 ER 的 4.7cm；MMPJPE 32.6 vs 40.6cm。
  - 明确观察到 BFM「即使角色状态显著偏离参考也能产生物理上合理的动作」，包括摔倒后接触丰富的爬起。
  - **⚠️ 致命细节：这个潜搜索在 RTX 5090 上配 128 个环境要几分钟。根本没法机上实时。**
- **Optimistic Task Inference for BFMs**（arXiv [2510.20264](https://arxiv.org/pdf/2510.20264)）：把 z 推断当成潜任务空间上的优化，用 UCB 式乐观探索代替穷举搜索。
- **Zero-Shot Adaptation of BFMs to Unseen Dynamics**（arXiv [2505.13150](https://arxiv.org/html/2505.13150v2)）：直接针对 OOD **动力学**。
  - **Belief-FB**：transformer encoder 把测试时轨迹变成动力学上下文 h，然后 `F(s,a,[h;z])ᵀB(s')`。
  - **Rotation-FB**：从以 h 为中心的 von Mises-Fisher 分布采 z，把策略空间切成动力学专属的簇，防止「破坏性平均」。
  - 约 2 倍零样本回报提升。**明确不是潜搜索，是条件化。**只在玩具环境里；作者承认测试时随机探索在复杂场景会崩。

### 显式「投影回可行流形」的工作
- **Back to the Manifold: Recovering from Out-of-Distribution States**（2022，arXiv [2207.08673](https://arxiv.org/abs/2207.08673)）：你这个想法的经典表述，但是离线操作域不是人形。一个**专门的恢复策略**，密度估计说「你离开训练流形了」就触发，用一个平移对应机器人动作的等变潜映射导航回去。真机操作验证：只有 BC 会失败，恢复+BC 成功。**不是技能字典，是一个学出来的「回到支撑集」控制器。**
- **NMR 的聚类专家物理精修**（见第五节）：文本-动作检索建跨模态潜空间，K-means 切成行为同质的 motif，**每个簇一个 RL 专家，把「运动学合法但物理不可行」的动作投影到机器人的动力学可行流形上**。字面意义的「先聚类再投影」——但这是**离线数据管线，不是在线恢复**。
- **Uncertainty-aware Latent Safety Filters**（arXiv [2505.00779](https://arxiv.org/pdf/2505.00779)）：用世界模型的认知不确定性 + 保形预测**检测** OOD，再用可达性合成潜安全滤波器。检测-然后-干预，不是检索技能。

### 其他 BFM
- **A Survey of Behavior Foundation Model**（arXiv [2506.20487](https://arxiv.org/abs/2506.20487)）：**这才是真正的综述。**按预训练管线分类，有维护中的 GitHub 论文列表。
- **Behavior Foundation Model for Humanoid Robots**（arXiv [2509.13780](https://arxiv.org/abs/2509.13780)）：**不是综述，是方法论文。**CVAE + 掩码在线蒸馏（DAgger），teacher 是在重定向 AMASS 上用 PPO 训的代理。统一接口覆盖根部控制/运动学位置控制/关节角控制，用**二值掩码**激活。打败 HOVER：AMASS-test MPJPE 0.2235 vs 0.3055。潜插值能组合出新行为（回旋踢）；用 classifier-free guidance（λ=0.5）做外插还能提升跟踪。
- **ReactiveBFM**（arXiv [2606.30362](https://arxiv.org/abs/2606.30362)）★：批评现有 BFM 只会执行**预定**动作，不反应。加了生成式动作规划 + 闭环反馈；用**调度式前缀采样课程训练规划器从不完美物理状态中恢复**（修 exposure bias）；异步重规划 + 轨迹分块解决「慢规划器/快控制器」不匹配。**真 G1 上扰动下 93.1% 成功率，比基线高 28.6%**，零样本移动目标到达。
- **Perceptive Behavior Foundation Model**（arXiv [2606.08059](https://arxiv.org/abs/2606.08059)）：批评 BFM 假设人类动作已经和机器人所处地形物理兼容。加了地形贴合参考合成（TCRS）+ 身份门控 Transformer。
- **Stubborn**（arXiv [2606.12814](https://arxiv.org/html/2606.12814v1)）：统一跟踪 + 摔倒恢复。
- **BeyondMimic**（arXiv [2508.08241](https://arxiv.org/html/2508.08241)）：引导式扩散。

### 字典检索恢复：FIRM ★这是最直接的证据和最直接的警告
- Unified Humanoid Fall-Safety Policy from a Few Demonstrations，arXiv [2511.07407](https://arxiv.org/html/2511.07407)
- 做法：
  - 从增广后的示教里预先构建一个 **keyframe 特征 codebook** `F = {f_g1, ..., f_gn}`。
  - 运行时，一个 online adapter 根据观测历史预测一个特征向量，**用余弦相似度从 codebook 里检索最接近的 keyframe**。
  - 检索到的 keyframe 特征去条件化一个扩散策略，生成动作序列。每 5 步更新一次，动态挑新的中间目标。
  - 检索到的位姿当路点，把重叠片段串起来拼成轨迹。
- 结果：不平地形上比 HoST 好 70%。负载变化下也稳。
- **它自己承认的致命局限（你必须读这句）**：
  > "FIRM 依赖 keyframe codebook 里的最近邻匹配，这可能在**高度 OOD 的场景下限制性能**。"
  - 而且：不同示教序列的潜表示「相对不相交，限制了跨动作的组合」。
- 也就是说：**字典检索能覆盖的范围，被字典本身的多样性卡死。**离字典远了，最近邻检索本身就是垃圾进垃圾出。

### 其他恢复工作
- **RecoverFormer**（arXiv [2604.22911](https://arxiv.org/pdf/2604.22911)）：端到端接触感知恢复。学到的潜模态在不同力的量级上分化，一个端到端策略就能做多模态、接触感知的恢复，跨扰动量级和接触几何泛化。
- **Learning Getting-Up Policies**（arXiv [2502.12152](https://arxiv.org/pdf/2502.12152)）：真机爬起来。
- **State-Dependent AMP**（arXiv [2605.18611](https://arxiv.org/pdf/2605.18611)）：走、跑、恢复统一。

---

## 五、在离散动作 token 上做 RL 后训练

你的论点：离散 token 给精确 log-prob，物理给真奖励（不是学出来的奖励模型）。

**结论：这个论点是对的，而且已经有人验证了，用的是 GRPO。**

### RLPF（RL from Physical Feedback）★★
- arXiv [2506.12769](https://arxiv.org/html/2506.12769)，标题：Aligning Large Motion Models with Humanoid Control
- **被微调的是一个离散 motion token 模型**。tokenizer 把动作序列离散化成 token（encoder + decoder + codebook）。主干是 **LLaMA2-7B**，自回归生成 motion token。
- **奖励来自真的物理仿真器（IsaacGym），不是学出来的奖励模型**。三步：
  1. 动作重定向：用梯度优化 SMPL 参数适配机器人形态。
  2. 动作跟踪策略：预训练的 student policy（从 teacher 用 DAgger 式方法蒸馏）去执行重定向后的动作。
  3. **二值成功奖励**：`R_tracking = 𝕀(Succ(π, m_i))`，位置偏差低于 0.5m 且保持平衡才算成功。
- **RL 算法：GRPO**，不用 critic，从组内相对分数估基线。
- 还有一个 alignment verification 模块，用对比训练的 encoder 保证语义不跑偏。两个验证奖励：文本-动作对齐、动作-真值对齐。
- **结果：成功率从 43-48% 提到 90-97%（仿真）。真机 Unitree G1 部署成功。**
- **重要的消融**：`RLPF-w/o align` 跟踪很好但**语义灾难性丢失**。也就是说只用物理奖励，模型会学会生成「好跟但和文本没关系」的动作。**这是你必须防的坑。**
- 局限：受限于那个冻结的动作跟踪策略，它是在有限静态数据集上预训练的。

### Humanoid-LLA ★★★这是最接近你完整愿景的一篇
- [OpenReview PDF](https://openreview.net/pdf/830d6b6b95ad212ad757bba8ac1fac08c6273041)，open-vocabulary 人形控制
- **统一动作词表**：人类动作重定向到人形，配对序列做**联合量化**，保证「同一个离散 token 在两种本体上一致地表示同一个动作原语」。
- **tokenizer：带隐式分块的 VQ-VAE**。潜向量被分解成子块，由多个子码本分别量化。这样不用手工指定身体部位，又能保住稳定执行需要的运动学细节。三个损失：模态内重建、commitment、**跨模态重建**（强制双向一致）。
- **两阶段训练**：
  1. SFT：用 VLM 在渲染序列上生成的 motion chain-of-thought 标注。离散 token 是预测目标。
  2. **RL：GRPO，双奖励**——分布奖励（语义-动作对齐）+ 跟踪奖励（仿真执行质量）。
- **token 到控制器**：一个蒸馏出来的**「词表导向 student policy」（CVAE）**把 token 序列变成物理可执行动作。**用的是隐式词表跟踪，推理时不需要在线重定向。**这个控制器是从一个用 PPO 在稠密参考轨迹上训的特权 tracking teacher 蒸馏来的。
- **真机：Unitree G1 和 Booster T1 都部署了。**
- **结果：执行成功率 87.6%，平均每关节位置误差 56.43mm。**baseline：LangWBC 76.0%，UniAct 82.5%。
- 局限：
  - 只在 AMASS（约 2.7 万序列）上评估，对更多样动作源的泛化未知。
  - 需要基于仿真算奖励，部署复杂度高。
  - 没讨论长时程规划和动态重规划。
  - 三阶段管线的计算成本没细算。

**对你的意义**：**你的核心架构（跨本体离散词表 → LLM 词表扩展 → 物理控制器 → GRPO 物理奖励后训练）已经被完整实现并发表了。**必须正面面对这件事。你的差异只剩：
- 他们是配对数据的联合量化（人+G1+T1），**你是零样本到没训过的机器人**。这是真差异。
- 他们是 VQ-VAE + 隐式分块，你是 FSQ。FSQ 没有码本塌缩问题，实现更简单，但表达力上限要验证。
- 他们没有硬关节限位保证，你有投影层。

### 离散 token VLA 的 RL 微调：一个非常清楚的不对称

| 方法 | 链接 | 奖励 | 关键点 |
|---|---|---|---|
| **RIPT-VLA** | [2505.17016](https://arxiv.org/abs/2505.17016) | **只有稀疏二值成功** | 每上下文 K 次 rollout，留一法优势，PPO 目标，**完全不用 critic**。QueST +21.2%；OpenVLA-OFT 到 LIBERO 97.5%。**结论：有离散 token 就不需要价值函数、不需要塑形奖励。** |
| **SimpleVLA-RL** | [2509.09674](https://arxiv.org/pdf/2509.09674) | 环境 0/1 结果 | GRPO 直接作用在**动作 token 概率**上。**⚠️ 明确指出难点：VLA 的动作解码接近确定性，所以探索/rollout 多样性是瓶颈，和 LLM 的 RL 不一样。设计采样器前必读。** |
| **dVLA-RL** | [2606.23623](https://arxiv.org/pdf/2606.23623) | 仿真任务成功 | 在离散扩散 VLA 的去噪轨迹上做 RL；**明确用精确离散 token log-prob，不需要任何近似**。局限：对 token 粒度敏感，算力重。 |
| **VLA-RL** | [2505.18719](https://arxiv.org/abs/2505.18719) | **学出来的**「机器人过程奖励模型」 | **⚠️ 反面教材：只涨 4.5%（40 个 LIBERO 任务），而且需要课程选择、向量化环境、批量解码、critic 预热才能不发散。** |
| **π_RL** | [2510.25889](https://arxiv.org/abs/2510.25889) | — | flow-based VLA，**需要代理 log-prob**。 |

**规律很清楚，这是你最强的论据**：
> **离散 token 的 VLA，PPO/GRPO 原封不动就能用；flow/diffusion 的 VLA 必须用代理似然（π_RL、FA-GRPO）或可微奖励的技巧（MotionRFT）。**

**真奖励 vs 学出来的奖励**，也是一边倒：
- VLA-RL 的学习式过程奖励模型换来 +4.5%，还得靠 critic 预热和课程才不崩。
- RIPT-VLA / RLPF 的二值仿真成功奖励，两样都不需要。
- **MotionRFT**（arXiv [2603.27185](https://arxiv.org/html/2603.27185)）：扩散/flow 生成器 + **可微奖励最大化**（不是 PPO/GRPO），奖励是纯**学出来的** MotionReward 模型。FID 0.132（−70.7%）。但他们自己说：MotionReward「**无法完全解决面向真实人形机器人的物理合理性问题**」。**这是「学出来的奖励不够」的直接证据。**
- **PhyMotion**（arXiv [2605.14269](https://arxiv.org/abs/2605.14269)）：RL 后训练**视频**生成器，奖励靠从生成视频恢复 SMPL、重定向进 **MuJoCo**、再打运动学合理性/接触平衡/动力学可行性分。盲测人评 +68 Elo。**「拿仿真器当奖励函数」做到最大规模的例子。**
- **Humanoid-R0**（[OpenReview](https://openreview.net/forum?id=agohD5ewsR)）：GRPO 微调动作生成器，但**奖励是手工代理（平滑度、骨架合理性、关键帧一致性），不是仿真器**。可以当对照组引用。

### 其他
- **Commanding Humanoid by Free-form Language**（arXiv [2511.22963](https://arxiv.org/html/2511.22963v3)）：**和 Humanoid-LLA 是同一篇论文的两个名字**（arXiv v1 2025-11-28，v3 2026-05-11）。补充细节：量化是**乘积量化（子码本），不是残差 VQ**。GRPO 三个奖励：二值格式奖励（强制输出串里子码本循环顺序）、分布奖励、**仿真 rollout 的跟踪奖励（位置项 r_pos + 加速度项 r_acc）**。数据约 2.7 万 AMASS 文本标注序列，筛成「G1 可执行子集」约保留 70%。FID 2.626 vs 8.513。
- **PhysMoDPO**（arXiv [2603.13228](https://arxiv.org/html/2603.13228)，2026）：
  - 生成器是**扩散**（MotionStreamer、OmniControl），**不是离散 token**，连续 SMPL 旋转空间。
  - 偏好信号来自**物理仿真器 + 固定跟踪策略（DeepMimic）**：采多个动作 → 每个在仿真里执行 → 在**实际被跟出来的动作**上算奖励 → 按多目标奖励占优构造偏好对。
  - 四个奖励：跟踪、滑步、M2T 文本对齐（TMR embedding）、控制。
  - 标准 DPO + 在获胜样本上加 SFT 损失防漂移。迭代式，每轮重新生成偏好对。
  - 结果：HumanML3D 上 R@3 0.8310→0.8517，Jerk 46.75→43.60。零样本迁移到 G1。
  - 局限：只在平地移动上验证；固定仿真策略会带来评估器偏差。
- **SCRIPT**（arXiv [2605.22894](https://arxiv.org/pdf/2605.22894)）：多阶段训练的可扩展扩散策略，语言驱动物理人形控制。
- **MotionGPT / MotionGPT-2**（[项目页](https://qiqiapink.github.io/MotionGPT/)、arXiv [2410.21747](https://arxiv.org/html/2410.21747)）：把动作 VQ 成 token 后**扩展 LLM 词表**，embedding 矩阵和预测头按 motion token 数量增长。**纯运动学，没有物理。**这是「扩词表」这个具体做法的出处。

### 关于生成器只吐可跟的动作

- **SafeFlow**（arXiv [2603.23983](https://arxiv.org/html/2603.23983)，2026）★问题陈述完全就是你的动机
  - 原文：「纯运动学生成器经常出现**物理幻觉（physical hallucinations）**，产生物理上无法跟踪的动作轨迹」——表现为关节限位违反、自碰撞、失衡。
  - 原文：「对于 **OOD 提示**，生成过程变得不稳定，导致结构性崩溃和不安全、不合理的全身构型。」例子：「悬浮」「双后空翻」。
  - 解法一（软）：physics-guided rectified flow。推理时用加权代价函数把轨迹推向可执行区域。四个约束：关节限位（ReLU 平方障碍）、自碰撞（球体惩罚+安全裕度）、平滑（速度加速度正则）、质心稳定。然后用 Reflow 蒸馏把物理约束内化到网络权重里，函数评估次数从 10 降到 1，达到 67.7Hz。
  - 解法二（硬）：**三级安全门，是拒绝，不是投影**：
    1. 输入级：CLIP embedding 空间的马氏距离检测语义 OOD。90% ID 接受阈值下，AUROC 0.9872（没见过的动词）/0.9715（极端动力学），拒掉 95% 危险输入。
    2. 模型级：**方向敏感度差异度量**，在 ODE 积分过程中监测 Jacobian 各向异性。流响应在各扰动方向上方差大 = 结构不稳定。
    3. 输出级：确定性运动学筛查，违反硬件限位/速度/加速度上限的一律拒绝。
  - **被拒绝时不做任何补偿，直接执行预设的安全 fallback（站直 + 插值）。**
  - 结果：关节限位违反 43.14% → 3.08%，成功率 80.6% → 98.5%，MPJPE 81.42 → 40.89mm。真机 G1。
  - **局限（原文，这就是留给你的口子）**：
    > 「未来工作的一个重要方向是**改进 fallback 行为，让它在高动态动作中更任务感知，实现更平滑的恢复**。」
  - 也就是说：**「摔回站立」是当前 SOTA 的 fallback，作者自己知道这很蠢。你的『去字典里搜一个近邻技能』正好补这个洞。**

- **Learning WBC Locomotion via Motion Generation and Motion Tracking**（arXiv [2604.17335](https://arxiv.org/abs/2604.17335)，2026）★这就是你说的「WBC 只见生成器产的动作」
  - 三阶段：(1) 扩散生成器和 RL tracker 各自在离线数据上预训练；(2) **冻结生成器，用 RL 微调 tracker**；(3) 部署时生成器以 receding-horizon 方式每 0.25 秒更新一次参考。
  - **为什么要这么做（原文逻辑）**：预训练的 tracker 学的是「平滑且经过物理过滤」的参考，但部署时在线生成器产的参考有「时序不连续和动作伪影」。**这个分布错配会拖垮性能。**
  - 微调时是「跑着冻结的生成器」，在随机方向指令和更多样地形上训 tracker。
  - **效果**：tracker 变成了一个「动作滤波器」——跟生成器的参考，同时用外感受地形观测调整执行以抑制不安全行为。
  - 数字：箱体攀爬平均成功率 0.859±0.252（只有 tracker）→ 0.987±0.014（加动作生成）。70-80cm 箱体上，固定参考跟踪崩到 0.23-0.61，完整系统保持 0.96+。
  - 涌现行为：角落箱体攀爬、箱上重定向后下降、连续翻越。
  - 局限：依赖 LiDAR 建图质量；只做了移动，没有 loco-manipulation。

- **NMR / Make Tracking Easy**（arXiv [2603.22201](https://arxiv.org/html/2603.22201)，2026）：
  - 传统优化式重定向的两个毛病：非凸（局部最优导致关节跳变和自碰撞）、把源数据噪声原样传下去且没有物理感知。
  - 三段管线：语义过滤 → 带质量检查的运动学重定向 → **用 RL 策略在仿真里做物理精修**。产出约 3 万条验证过的动作对。
  - **trackability 怎么量**（你之后要用的指标就是这三个）：Success Rate（下游 RL 策略成功跟住的比例）、MPJPE、W-MPJPE。加上物理合理性指标：关节连续性、自碰撞、关节限位违反。
  - 结果：**零关节跳变**，自碰撞比 GMR 少 54%，关节限位违反降到 16.80%。用 NMR 参考训的策略 episode 更长、奖励更高。

---

## 六、跨本体动作 tokenizer

这一节决定你到底有没有护城河。按「离散？物理？跨机器人身体？」三个维度分类：

| 系统 | 表示 | 接物理控制器？ | 跨的是什么 | 关键数字 / 问题 |
|---|---|---|---|---|
| **SONIC** | FSQ-32-32，64 维 | ✅ PPO 策略 | **同一台 G1 上的不同输入模态**（机器人动作 / 人体 SMPL / 混合），**不是跨机器人身体** | `L_token = ‖z_r − z_h‖²`。VLA 预测 token 比预测 SMPL 高 41 分 |
| **Humanoid-LLA** | VQ-VAE 乘积量化 | ✅ CVAE 学生策略 | 人 + G1 + T1，**靠配对数据联合量化** | 87.6% 成功，MPJPE 56.43mm |
| **Any2Any** | **没有 tokenizer** | ✅ | G1 → LimX Oli / Luna | **⚠️ 你必须打败的诚实 baseline**：运动学对齐输入输出空间 + 只在动力学敏感模块插 LoRA。**只要从头训练 1% 的算力和数据。** |
| **PHASOR** | **故意用连续**：可微 FFT 相位流形（幅度/频率/相位，每身体部位 32 维） | 用作 RL 的「相位奖励」 | G1(32DOF) + H1.2(28) + Booster T1(24) + Berkeley Lite(25)，SMPL-H 当冻结锚点 | 人→机器人检索 R@1 **90.3%**，机器人→机器人 84.8%，**明确说打败 VQ**。⚠️ 但只管周期性动作，姿态分支仍是每本体一套，**只在仿真里** |
| **UniT** | RQ-VAE，视觉/动作/融合三支共享码本 | ❌ | 人 ↔ 人形 | RoboCasa 66.7% vs 55.0% |
| **LAPA / IGOR / UniVLA / villa-X** | VQ 码本，从视频帧差学 | ❌ **全都不接物理控制器** | 多本体 | UniVLA 也是**扩 LLM 词表加量化潜动作 token**，每个机器人一个轻量解码头 |
| **Motion-Focused LA** | VQ，**码本只有 16**，每帧对 4 个 token | ❌ | WindowX→Franka、Aloha-Agilex 等 | **自己承认离散码本容量不足以做细粒度精确控制** |
| **Unified Latent Space for Cross-Embodiment**（arXiv [2601.15419](https://arxiv.org/abs/2601.15419)） | CVAE，**连续** | ❌ | 多本体 | **新机器人 = 加一个新的嵌入层**（还是要训） |

**关键判断（这是全篇最重要的一句）**：

> **「离散 token 词表 × 物理控制器 × 跨机器人身体」这三样凑齐的，全世界只有 Humanoid-LLA 一篇。而它覆盖两台机器人，靠重定向配对 + 30% 的可行性过滤。**
>
> **没有任何一篇展示过「一个全新的、没见过的机器人可以零样本被丢进这个 token 词表」。**

所以你的护城河是真的，但很窄，而且有两个直接威胁：
1. **Any2Any** 说「不需要共享词表，LoRA 迁移只要 1% 算力」。你必须正面比。
2. **PHASOR** 说「跨本体不需要离散，连续相位流形更好，而且我们明确打败了 VQ」。你必须解释为什么要离散——**唯一站得住的答案是「离散给精确 log-prob，所以能做 RL 后训练，还能当 LLM 词表」，不是「离散表示更好」。**

---

## 七、三个候选架构

### 架构 A：直接对接 SONIC 的 token 空间（先做这个，一周能出数）
```
我们的 FSQ token → 学一个薄适配器 → SONIC 的 64 维通用 token → SONIC decoder（冻结）→ 电机
```
不改 SONIC 任何权重。只训一个小的映射，把我们的码本翻译成它的 64 维 token。

**支持证据**
- **SONIC 的接口本来就是 token，而且开源了**（GEAR-SONIC，2026-02）。64 维通用 token @50Hz，N1.7 的 VLA 就是以 2.5Hz 喂 78 维给它。你要做的事它已经支持。
- **SONIC v3 的消融替你证明了「喂 token 比喂位姿好」**：68% vs 27%，最难任务 60% vs 0%。你不用再论证这一点。
- **冻结低层的先例极强**：TokenHSI 冻结主干 99.2% vs 从头训 26.8%；HOVER 蒸馏版在 32/32 个指标上打败端到端多模态 RL；WholeBodyVLA 冻结 LMO 拿 78.0%，去掉潜动作模块掉 38.7 分。
- 冻结意味着**你不需要那 2.1 万 GPU 小时**。

**能杀死它的风险**
- **SONIC 的 token 空间是和 PPO 策略联合训出来的**（`L = L_ppo + L_recon + L_token + L_cycle`），形状被 RL 塑造过。你的码本是独立训的，几何可能对不上，薄适配器学不动。
- **SONIC 的 decoder 输出写死 29 维 G1 电机指令。对齐到它 = 绑死在 G1 上，你的零样本跨本体优势当场消失。**这是最致命的一条。
- 有损压缩叠加：PULSE 明说信息瓶颈有损，专用任务打不过从头训。你的 token → SONIC token 又压一次。

**判决**：**这不是最终架构，这是你的测量工具。**用它拿到第一个 trackability 数字，证明我们的 token 能驱动真物理。然后再谈别的。

---

### 架构 B：穿过冻结控制器反传（PhysiFlow 那招）★推荐
```
FSQ token → 稀疏 SE3 + 投影层（硬保证关节限位）→ 冻结的 WBC → 仿真
                    ↑                                          │
                    └────── 跟踪误差梯度反传回来 ────────────────┘
```
控制器冻结，但**把物理执行误差穿过它反传，用来微调我们的 tokenizer/生成器**。让生成器学会「只吐控制器跟得住的东西」。

**支持证据**
- **PhysiFlow（2603.05410）已经验证了这一招**：「执行动作和参考动作之间的误差被反传去微调 flow matching 模型，确保它生成的动作是物理感知的。」总体 74.9% vs LeVERB 65.0%，长导航 63.6 vs 31.2。
- **2604.17335 的闭环版本**：冻结生成器微调 tracker（方向相反但同一思想）。箱体攀爬 0.859±0.252 → **0.987±0.014**；70-80cm 箱体上固定参考崩到 0.23-0.61，闭环系统保 0.96+。**明确诊断出「分布错配会拖垮性能」，正是你的动机。**
- **这直接实现你的原始诉求「WBC 只见生成器产得出的动作」**，而且是双向收敛，不是单向过滤。
- **投影层的硬保证是你真正没人做的东西**。SafeFlow 只有软引导（关节限位违反 43.14% → 3.08%，**还剩 3%**）和事后拒绝。硬投影能做到 0%。
- KI 的教训在这里也适用：梯度要**整形**，不能让随机初始化的下游直接污染上游。冻结 WBC + 只回传误差信号，正好符合。

**能杀死它的风险**
- **梯度穿过一个冻结的 RL 策略 + 不可微仿真，这条链能不能通是个真问题。**PhysiFlow 的 tracker 是神经网络所以可微，但仿真器不可微。MoConVQ 的解法是学一个世界模型来近似仿真，代价是多训一个模型。
- **Puppeteer 的警告**：端到端耦合的代价不出现在 loss 和成功率上，出现在**语义和自然度**上（它端到端回报打平，然后学会滚上楼梯）。你必须做人类偏好评测，不能只看数字。
- **OpenHLM 的警告**：动作 MSE 是糟糕的代理，两个 MSE 相同的模型真机差 20 分。别用重建误差当主指标。
- 纯离散会抖：Hybrid Latent Representation（EG 2025）明确说需要连续残差；π0.5 也是预训练离散、部署换连续头。**准备一路连续残差通道。**

---

### 架构 C：在线 OOD 触发 + 字典检索恢复 ★这是唯一真正无人区
```
正常：VLM → motion token → 投影层 → WBC
监控：一个 OOD 判别器实时打分「当前状态还在字典覆盖里吗」
异常：从当前状态出发，在 FSQ 码本上做摊销式最近邻检索 → 拿到候选 token 序列 → 平滑接回
```

**支持证据（每一条都很强）**
- **Fast Adaptation with BFM（RLC 2025）的核心发现就是你的前提，而且是反过来说的**：**「BFM 的策略空间里早就装着比零样本推断找到的好得多的行为」**——好东西已经在字典里，闭式推断只是没找出来。搜索本身带来 **HumEnv 上 40–50% 的提升**。
- **BFM-Zero 的潜空间足够光滑**：两个 z 之间球面插值零样本就能得到可行的中间技能（走路↔跳舞能组合）。**「在 Z 里靠近」真的等于「附近的可行技能」**——这让「搜最近技能」这件事在数学上成立。
- **BFM-Zero 对不可行目标姿态「优雅处理」**，收敛到自然可行构型而不是发散。**隐式的流形投影已经在策略内部发生。**
- **SafeFlow 明确把「更好的 fallback」列为未解决问题**——当前 SOTA 的 fallback 就是站直，作者自己说这在高动态动作里很糟。
- **FB-CPR / BFM-Zero 里那个判别器 `D(s,z)` 是免费的、没人用的 OOD 信号。**它本来就是训来打分「这个状态在不在行为数据集里」的。**没有任何一篇 BFM 把它当运行时 OOD 监控器用过。**
- **离散码本天生适合检索**。FSQ 码本有限、有几何结构，最近邻是 O(1)。连续潜（PULSE 32 维、LeVERB 256 维、BFM-Zero 128 维）反而只能靠 CEM 采样搜。**这是离散相对连续的一个真实且未被论证过的优势。**

**能杀死它的风险（同样每条都很硬）**
- **⚠️ 速度。BFMTrack 的潜序列优化在 RTX 5090 上要几分钟。BFM-Zero 的 CEM 也是离线 few-shot。**恢复时的搜索必须比这快好几个数量级——**所以必须是摊销式检索（预计算 B(s) 建索引），绝不能是 CEM。**这是可行性的生死线。
- **⚠️ 字典最薄的地方正好是你最需要它的地方。**Meta Motivo 点名「**摔倒、站起来、贴地动作**」是最弱区域。摔倒时去搜最近可行技能，搜的正是流形最稀疏的部分。
- **⚠️ 触发信号本身很脆。**BFM-Zero 承认「**稀疏奖励推断不稳定，重复跑会得到不同策略，有些非常差**」。如果你的触发器是一个稀疏的「我要摔了」信号，正好落在最脆的区间。
- **FIRM 自己承认**最近邻在高度 OOD 时失效，而且不同示教的潜表示「相对不相交，限制跨动作组合」。
- **检索到的技能未必在当前状态可行。**字典里有后空翻，不代表正在前扑时能做后空翻。你需要一个「当前状态 → 可行 token 子集」的判别器。PULSE 的本体感觉条件先验就是干这个的——**你可能需要做一个离散版本**。
- **MaskedMimic 的 early termination 阻止它学 OOD 恢复**——如果下游 WBC 训练时从没进过 OOD 状态，它根本执行不了你检索出来的恢复动作。**恢复需要 WBC 配合，不是纯上游的事。**

**判决**：**这是三个里唯一能写成一篇有分量论文的。**差异化的说法不能是「在潜空间搜索」（BFM-Zero/BFMTrack/LoLA 有了），也不能是「从扰动中恢复」（BFM-Zero/RecoverFormer 有了），必须是：

> **闭环：一个在线 OOD 触发器，从当前状态重新查字典，而且便宜到能跑在控制频率上。**

具体建议：用 FB-CPR 式判别器或 B(s) 空间密度当触发器，用**预计算索引的摊销最近邻**（不是 CEM）拿候选 z，然后在 BFM-Zero 自己的恢复 episode 和 Motivo 自己承认的弱区（摔倒/站起/贴地）上做对比评测。

---

## 八、给 lloyd 的直接建议

### 一、必须放弃的三个「新颖点」

这三件事已经被做完了，再拿它们当卖点会被审稿人当场打掉：

1. **「给 VLM 加动作 token 驱动物理控制器」** —— SONIC v3 + GR00T N1.7 已经在跑，而且有 +41 分的消融。
2. **「跨本体 FSQ 动作词表」** —— SONIC 是 FSQ-32-32，GPC 是 FSQ 40×9，SCHUR 做完了 FSQ vs VQ 的 scaling 对比。
3. **「离散 token 上做物理奖励 RL」** —— RLPF、Humanoid-LLA 都做了，都上了真机。

### 二、可以保留但要重新表述的

- **「离散 token 更好」** —— 不要说表示更好。**要说它是梯度整形选择。**KI 那篇给了机制：随机初始化的连续动作头反传会毁掉 VLM 知识（OpenVLA 的 MMStar 和 TextVQA 直接归零），而离散 token 的交叉熵损失和 VLM 预训练同模态，是「梯度桥」。Wall-OSS 直接用了这个词。
  - 同时记住 KI 的另一半：**完全冻结主干也是 0%。**赢的配方是「主干用离散 token 损失训 + 连续专家 stop-gradient」。
- **「精确 log-prob 所以 RL 干净」** —— 这条完全站得住，而且有整齐的证据链：离散 token VLA 的 PPO/GRPO 原封不动能用（RIPT-VLA 连 critic 都不要），flow/diffusion 必须用代理似然（π_RL）。

### 三、真正属于你的四个洞

1. **零样本到训练时完全没见过的机器人。** 全世界只有 Humanoid-LLA 凑齐了「离散×物理×跨身体」，而它是两台机器人 + 配对数据 + 30% 可行性过滤。**没人做零样本。**
2. **接口层的硬可行性保证。** SafeFlow 的软引导剩 3.08% 关节限位违反；硬投影可以做到 0%。没人做。
3. **在线、控制频率级的字典恢复。** 见架构 C。
4. **在「解码器就是控制器」的 tokenizer 上做 RL 后训练。** GPC 有物理原生 FSQ tokenizer，但它的 GPT **只用交叉熵训，完全没 RL**。RLPF/Humanoid-LLA 是 RL 一个运动学生成器 + 冻结 tracker 当裁判。这个组合空着。

### 四、必须先测的四个数（按顺序）

1. **trackability。** 把我们 tokenizer 解出来的动作喂给一个真 WBC（SONIC 最方便，开源），出 Success Rate + MPJPE + W-MPJPE + 关节限位违反率（NMR 那套指标）。**没这个数，上面全是空谈。**
2. **零样本跨本体到底成不成。** 至少 3-4 个没训过的本体。**必须和 Any2Any 比**——它用 LoRA 迁移只要从头训练 1% 的算力，这是你的诚实 baseline。
3. **纯离散会不会抖。** 测 jerk 和加速度误差。抖就立刻加连续残差通道（Hybrid Latent Representation、π0.5 都是这么干的）。
4. **恢复搜索的延迟。** 如果做不到控制频率（50Hz 级），架构 C 就不成立。BFMTrack 是几分钟，这是反面参照。

### 五、必须防的四个坑

1. **语义灾难性丢失。** RLPF 的消融：只用物理跟踪奖励做 RL，模型会学会生成「好跟但和指令无关」的动作。**GRPO 时必须同时挂语义对齐奖励。**
2. **探索塌缩。** SimpleVLA-RL 明确点名：VLA 的动作解码接近确定性，rollout 多样性是瓶颈，和 LLM 的 RL 不一样。**没有一篇 motion-token 论文报告过这个问题的解法。**设计采样器前先读它。
3. **别只看 loss 和成功率。** Puppeteer：端到端回报打平，然后 reward hack 滚上楼梯，97.6% 人类更喜欢分层版。OpenHLM：动作 MSE 相同的两个模型真机差 20 分。**必须做人类偏好评测和可视化。**
4. **冻结 tracker 当裁判的盲区。** RLPF 自己承认：冻结的跟踪策略在有限静态数据集上训的，**它没法判断 OOD 动作**，奖励会悄悄失效。你如果用同样的设置，会继承这个盲区。

---

## 附：完整链接清单

**潜动作空间 / 物理控制**
- PHC: https://arxiv.org/abs/2305.06456
- PULSE: https://arxiv.org/abs/2310.04582
- ASE: https://arxiv.org/abs/2205.01906
- ControlVAE: https://arxiv.org/abs/2210.06063
- NCP: https://arxiv.org/abs/2308.07200
- MoConVQ: https://arxiv.org/abs/2310.10198
- Hybrid Latent Representation: https://arxiv.org/abs/2503.12814
- Spherical Latent Motion Prior: https://arxiv.org/pdf/2603.01294

**统一 WBC**
- MaskedMimic: https://arxiv.org/abs/2409.14393
- HOVER: https://arxiv.org/abs/2410.21229
- SONIC: https://arxiv.org/abs/2511.07820
- GR00T-WBC: https://github.com/NVlabs/GR00T-WholeBodyControl
- ExBody2: https://arxiv.org/abs/2412.13196
- OmniH2O: https://omni.human2humanoid.com/
- RGMT: https://arxiv.org/abs/2601.23080
- AnyBody: https://arxiv.org/pdf/2606.29209

**VLM → 控制器**
- LeVERB: https://arxiv.org/abs/2506.13751
- GR00T N1: https://arxiv.org/abs/2503.14734
- Helix: https://www.figure.ai/news/helix
- π0.5: https://www.pi.website/download/pi05.pdf
- Knowledge Insulation: https://www.physicalintelligence.company/download/pi05_KI.pdf
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342
- WholeBodyVLA: https://arxiv.org/html/2512.11047v1
- MotionWAM: https://arxiv.org/abs/2606.09215
- LangWBC: https://arxiv.org/pdf/2504.21738

**行为基础模型**
- FB-CPR / Motivo: https://arxiv.org/abs/2504.11054
- Metamotivo 代码: https://github.com/facebookresearch/metamotivo
- BFM-Zero: https://arxiv.org/abs/2511.04131
- BFM 综述: https://arxiv.org/pdf/2509.13780
- FIRM: https://arxiv.org/html/2511.07407
- RecoverFormer: https://arxiv.org/pdf/2604.22911

**离散 token RL**
- RLPF: https://arxiv.org/html/2506.12769
- Humanoid-LLA: https://openreview.net/pdf/830d6b6b95ad212ad757bba8ac1fac08c6273041
- Free-form Language LLA: https://arxiv.org/html/2511.22963v3
- PhysMoDPO: https://arxiv.org/html/2603.13228
- MotionGPT: https://qiqiapink.github.io/MotionGPT/

**可行性 / trackability**
- SafeFlow: https://arxiv.org/html/2603.23983
- Gen+Track 闭环: https://arxiv.org/abs/2604.17335
- NMR: https://arxiv.org/html/2603.22201

**（补）FSQ / 离散表示**
- GPC (SIGGRAPH 2026, FSQ 40×9 + PPO 端到端): https://arxiv.org/abs/2606.29148
- SCHUR / Humanoid-Union (FSQ vs VQ scaling): https://arxiv.org/html/2511.09241v1
- FSQ 原始论文: https://arxiv.org/abs/2309.15505
- ARP (FSQ vs VQ vs LFQ 操作对比): https://arxiv.org/html/2606.22480

**（补）VLM 知识退化 / 梯度整形**
- Knowledge Insulation: https://arxiv.org/abs/2505.23705
- VLM2VLA (OpenVLA MMStar/TextVQA 归零): https://arxiv.org/abs/2509.22195
- InstructVLA: https://arxiv.org/abs/2507.17520
- ReVLA (DINOv2 塌缩): https://arxiv.org/abs/2409.15250
- Wall-OSS-0.5 (反对冻结): https://arxiv.org/abs/2605.30877
- RDT2 (RVQ + 冻结主干): https://arxiv.org/abs/2602.03310
- FAST: https://arxiv.org/abs/2501.09747
- Hi Robot (自然语言接口): https://arxiv.org/abs/2502.19417
- VLA 抗遗忘的反证: https://arxiv.org/abs/2603.03818

**（补）冻结技能层 / 分层**
- PhysiFlow (穿过冻结 tracker 反传): https://arxiv.org/abs/2603.05410
- TokenHSI (冻结 99.2% vs 从头 26.8%): https://arxiv.org/abs/2503.19901
- Puppeteer (端到端 reward hack 滚楼梯): https://arxiv.org/abs/2405.18418
- ULTRA (故意联训): https://arxiv.org/abs/2603.03279
- LCB (`<ACT>` token): https://arxiv.org/abs/2405.04798
- OpenHLM (动作 MSE 是糟糕代理): https://arxiv.org/abs/2606.22174
- HALOMI (冻结 BFM-Zero + VLA): https://arxiv.org/html/2606.18772v1
- AgiBot GO-1 / ViLLA: https://arxiv.org/abs/2503.06669

**（补）潜空间搜索 / 零样本 RL**
- Does Zero-Shot RL Exist?: https://arxiv.org/abs/2209.14935
- Fast Adaptation with BFM (ReLA/LoLA): https://arxiv.org/abs/2504.07896
- BFMTrack (潜序列优化): https://arxiv.org/html/2606.25056v1
- Optimistic Task Inference: https://arxiv.org/pdf/2510.20264
- BFM 适配未见动力学 (Belief-FB): https://arxiv.org/html/2505.13150v2
- Back to the Manifold: https://arxiv.org/abs/2207.08673
- BFM 综述: https://arxiv.org/abs/2506.20487

**（补）离散 token RL 微调**
- RIPT-VLA: https://arxiv.org/abs/2505.17016
- SimpleVLA-RL (探索塌缩): https://arxiv.org/pdf/2509.09674
- dVLA-RL: https://arxiv.org/pdf/2606.23623
- VLA-RL (学习式奖励的反面教材): https://arxiv.org/abs/2505.18719
- MotionRFT (学出来的奖励不够): https://arxiv.org/html/2603.27185
- PhyMotion (仿真器当奖励): https://arxiv.org/abs/2605.14269

**（补）跨本体**
- Any2Any (LoRA 迁移，1% 算力): https://arxiv.org/abs/2605.23733
- PHASOR (连续相位流形，打败 VQ): https://arxiv.org/html/2606.01851v1
- UniT (RQ-VAE 共享码本): https://arxiv.org/html/2604.19734v1
- UniVLA: https://arxiv.org/pdf/2505.06111
- Unified Latent Space for Cross-Embodiment: https://arxiv.org/abs/2601.15419
