# 别的领域怎么做「保住动态」的离散 tokenizer

给 lloyd 的文献扫描。目标：从音频、视频、时间序列、脑电这些更成熟的领域，搬解决方案回来修我们的 static collapse。

## 我们的病历（先摆清楚，方便对号入座）

- 输入：60 帧（2 秒 @30fps）的帧级 FSQ code 流。
- 压缩：Perceiver 式 readout，8 个有序 register 全局 cross-attention 读所有帧，再过 FSQ。
- 训练：OAT prefix，逐帧 cross-entropy 重建。
- 结果：重建只留住 22–73% 的真实运动能量（开合跳只剩 22%），而底层帧级 code 本身能留住 82–98%。
- 症状：输出是「平均姿势 + 动作类别」。步态相位没锁上，膝盖 xcorr 在 lag0 ≈ 0。
- 新证据（零训练探针）：只保留每维前 8 个时间 DCT 频率，走路能留 95–100% 的运动能量，开合跳 84%。学出来的 8 个全局 register 只有 73% / 22%。
- 结论方向：运动在时间上是**低频**的。全局池化才是错的那一步，不是「8 个 token 太少」这么简单。

代码位置：`pose_translator_design/exp_topo/train_temporal.py` + `temporal_common.py`。

关键结构事实（读代码确认）：
- `TemporalTokenizer._read()`：register 对**全部帧**做 cross-attention，没有任何时间掩码（除非开 `--mode chunk`）。这就是全局池化。
- `decode_frames()`：每个帧 query 只看 register，**帧与帧之间完全不交互**（注释原文："No temporal mixing: every frame query sees only the registers, never other frames"）。
- 所以解码器本质上是一个「以归一化时间为输入的基函数合成器」——结构上跟 DCT 合成很像。容量是有的。问题在目标函数和瓶颈形状。
- 比特预算：8 register × 6 个 FSQ 维 × log2(5) ≈ **111 bit / 2 秒 ≈ 56 bit/s**。这个数字后面要用。

---

## 一、音频 / 语音神经编解码器

这是最该抄的领域。它们的共同点非常一致，而且**每一条都跟我们现在的做法相反**。

### 机制 1：时间轴永远保留，靠「卷积下采样」压缩，绝不全局池化

所有主流 codec 的编码器都是**带 stride 的卷积**，把时间轴按固定倍数抽稀，但抽完还是一条时间序列。没有任何一个把整段音频压成「几个无时间索引的 register」。

| 系统 | 年份 | 时间栅格 | 每秒 token 位置 | 深度（每个位置几个 token） | 码率 |
|---|---|---|---|---|---|
| SoundStream ([2107.03312](https://arxiv.org/abs/2107.03312)) | 2021 | stride (2,4,5,8)=320，24kHz | 75 Hz | 8 层 RVQ | 3–18 kbps |
| EnCodec ([2210.13438](https://arxiv.org/abs/2210.13438)) | 2022 | 320 | 75 Hz | 4–8 层 | 3–6 kbps |
| DAC / Improved RVQGAN ([2306.06546](https://arxiv.org/abs/2306.06546)) | 2023 | stride [2,4,8,8]=512，44.1kHz | 86 Hz | 9 层 | 8 kbps |
| WavTokenizer ([2408.16532](https://arxiv.org/abs/2408.16532)) | 2024 | stride 320 或 600 | 40–75 Hz | **1 层** | 0.5–0.9 kbps |
| Mimi / Moshi ([2410.00037](https://arxiv.org/abs/2410.00037)) | 2024 | stride (4,5,6,8)+2 | **12.5 Hz** | 8 层（1 语义 + 7 声学） | 1.1 kbps |
| Stable-codec / TAAE ([2411.19842](https://arxiv.org/abs/2411.19842)) | 2024 | patch 320 + strided conv | **25 Hz** | 1–2（**FSQ**，非 RVQ） | **400 / 700 bps** |
| SNAC (2024) | 2024 | 分层：粗慢细快 | 12 / 24 / 48 / 96 | 4 层不同时间率 | 1.9 kbps |
| **我们** | 2026 | **全局池化，时间轴消失** | **4 Hz（且无时间索引）** | 1（FSQ 5^6） | **约 56 bps** |

要点：**降低 token 率的办法是「拉长时间步长 + 在每个时间步堆深度」，不是「把时间维压没」。**

Mimi 是最激进的例子：12.5 Hz 时间栅格，已经是全领域最低的了，但它在每个时间位置上仍然放 8 个 token。它没有说「我只要 8 个 token 描述这一整秒」。

对比我们：8 个 token / 2 秒 = **4 个 token 位置每秒，且这 8 个 token 根本没有时间索引**（register 是按重要性排序的，不是按时间）。这是 Mimi 的 1/3 时间分辨率，而且完全丢掉了「哪个 token 管哪一段时间」这个信息。

### 机制 2：没有一个 codec 用纯逐步 L1/L2/CE 训练——这是有 ablation 证据的

DAC 的损失栈（[2306.06546](https://arxiv.org/abs/2306.06546)，NeurIPS 2023 spotlight）：

- 多尺度 mel 损失，权重 15.0（**7 个不同窗长**的 mel 谱上做 L1）
- feature matching 损失，权重 2.0（判别器中间层的 L1）
- 对抗损失，权重 1.0（HingeGAN）
- codebook 损失 1.0 + commitment 0.25

**DAC 的关键 ablation（直接打脸我们的做法）**：只用重建损失训练时，bitrate efficiency 从 99% 掉到 **62%**，SI-SDR 从 9.12 崩到 **1.07**。论文原话说模型出现大量 artifact，包括嗡嗡声，"as it has not learned to reconstruct phase"（它没学会重建相位）。

**注意「相位」这两个字**。我们的病症就是相位没锁（膝盖 xcorr lag0 ≈ 0）。音频领域早就诊断过：**逐点回归损失学不出相位，因为相位错开的多个解在逐点误差上会被平均掉，平均的结果就是没相位的、能量偏低的信号。** 这跟我们「输出平均姿势」是同一个病，只是换了个词。

Mimi（[2410.00037](https://arxiv.org/abs/2410.00037)）走得更极端：**纯对抗训练，完全去掉重建损失**，只保留 feature loss 和 discriminator loss。论文说："While removing reconstruction losses majorly degrades objective metrics, we observed during development that the resulting audio sounded much better than expected based on aforementioned metrics."（去掉重建损失后客观指标明显变差，但听感明显更好。）

这句话对我们非常重要：**「逐帧准确率」这个指标本身就会奖励静态解。** 我们现在的 eval 是 `budget_level_acc`（逐帧 argmax 命中率），这个指标跟运动能量保留率是脱钩的——正好解释了为什么训练看起来在收敛但动画是飘的。

SoundStream 的损失也是同一个配方：对抗 hinge loss（λ=1）+ feature matching（λ=**100**，注意权重多大）+ 多尺度谱重建（窗长 s ∈ {2^6 … 2^11}，hop = s/4，λ=1）。

### 机制 3：判别器专门针对「周期性」和「高频」设计

DAC 用两个判别器，各治一种病：

- **Multi-period discriminator（MPD）**，周期 [2,3,5,7,11]：专治周期性和音高 artifact。做法是把一维信号按周期 p reshape 成二维再卷积，强迫模型对齐周期结构。
- **Multi-band multi-scale STFT discriminator**，频带切分 [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]：专治混叠和高频建模，论文说分频带能给生成器"stronger gradient signal"。

MPD 来自 HiFi-GAN（[2010.05646](https://arxiv.org/abs/2010.05646), 2020）。**这个东西可以直接搬到步态上**：走路、跑步、开合跳都是周期信号，我们的相位没锁正是 MPD 要解决的问题。

### 机制 4：语义 token 和声学 token 分开（SpeechTokenizer / Mimi）

Mimi 的 "split RVQ"：第一个 VQ 专门蒸馏 WavLM 的语义表征，另外 7 层 RVQ 并行地管声学细节。论文明确说这样做是为了避免把声学信息"conserved in the residual"（被迫塞进语义量化器的残差里）。

SpeechTokenizer（[2308.16692](https://arxiv.org/abs/2308.16692), 2023）是同一思路的开创版本：RVQ 第一层用 HuBERT 蒸馏出语义，后面几层管说话人音色等声学细节。

**映射到我们**：我们的 8 个 register 现在是「一锅炖」——OAT prefix 让前面的 register 抢走了最省 loss 的东西，也就是**平均姿势（静态内容）**，后面的 register 才轮到动态。这在结构上等价于「语义挤掉了声学」。音频领域的解法是**显式分道**，不是靠排序自然涌现。

### 机制 5：FSQ 在语音 codec 里是能打到 SOTA 的（我们的路线有背书）

Stable-codec / TAAE（[2411.19842](https://arxiv.org/abs/2411.19842), Parker et al., Stability AI, 2024）："by scaling a transformer architecture with large parameter count to this problem, and applying a flexible **Finite Scalar Quantization (FSQ)** based bottleneck, it is possible to reach state-of-the-art speech quality at extremely low bit-rates of **400 or 700 bits-per-second**."

这是好消息：**FSQ + transformer 能做到极低码率的 SOTA 语音**，lloyd 的「codebook 才是未来」在音频侧站得住。FSQ 原论文（[2309.15505](https://arxiv.org/abs/2309.15505), Mentzer et al., 2023）也明确说 FSQ 不会 codebook collapse，不需要 commitment loss、codebook reseeding、code splitting、entropy penalty 这一整套。

但要注意**它们是把 FSQ 放在保留时间轴的序列上**，不是放在全局池化的 register 上。FSQ 本身不是问题，池化才是。

### 音频侧比特率对照（拿来评估我们的 56 bit/s）

- Stable-codec：400–700 bps（业界公认「极低」，还是 SOTA）
- WavTokenizer：500–900 bps
- Mimi：1100 bps
- SoundStream / EnCodec：3000–6000 bps
- DAC：8000 bps
- **我们：约 56 bit/s**

语音的内在带宽当然比人体运动高。但差了 7–12 倍，这个量级值得警觉。后面综合部分会算一次运动的率失真下界。

---

### 机制 6：多尺度 / 分层时间分辨率（SNAC）——粗 token 慢采样，细 token 快采样

SNAC（Multi-Scale Neural Audio Codec, Siuzdak 2024, https://github.com/hubertsiuzdak/snac）对 RVQ 做了一个很小但很关键的改动："coarse tokens are sampled less frequently, covering a broader time span"。

标准 RVQ 的所有层共用同一个时间分辨率。SNAC 让不同层用**不同的时间分辨率**。一个实际输出的形状是：

```
[code.shape[1] for code in codes]  ->  [12, 24, 48, 96]
```

四层，时间密度 1:2:4:8。粗层管长程结构，细层管细节。32kHz 模型在 1.9 kbps。

**这正是我们该要的东西的音频版**：低频（粗）系数少而慢，高频（细）系数多而快。跟我们 DCT 探针发现「运动是低频主导」完全对应——但注意 SNAC 依然给高频留了通道，我们现在一个都没留。

### 机制 7：stable-codec 的 FSQ 配置跟我们一模一样——差别只在「放几个、放在哪」

这是整份文献里最直接的对照组。TAAE / stable-codec（[2411.19842](https://arxiv.org/abs/2411.19842)）：

- **FSQ 维度：6 维**
- **每维 levels：5**（训练时在 17 / 9 / 5 之间随机切换）
- **单 token 变体码本：5^6 = 15,625**

我们的 `--fsq_levels 5,5,5,5,5,5` → 5^6 = 15,625。**完全相同。**

差别在这三点，而且每一点我们都站在错的一侧：

| | stable-codec | 我们 |
|---|---|---|
| token 率 | **25 tok/s**（400bps）/ 50 tok/s（700bps） | **4 tok/s** |
| 时间轴 | patch size 320 + strided conv，**时间轴全程保留** | 全局 cross-attention 池化，时间轴消失 |
| 注意力 | 只用 **sliding window 局部注意力**，明确限制感受野 | 全局，无掩码 |
| 损失 | 判别器 feature matching + 多分辨率复数 STFT 对抗 + WavLM 感知损失 | 逐帧 CE |

论文说 FSQ 优于 RVQ 的理由跟我们的动机一致："RVQs can suffer from poor or inconsistent codebook utilization"，FSQ 则 "almost full codebook utilisation even with very large codebook sizes"。**FSQ 路线没问题，是我们把它装在了错的位置上。**

一句话总结音频侧：**同样大小的 token，人家每秒放 25 个并保留时间索引，我们每秒放 4 个还把时间索引抹掉了。**

---
### 补充：EnCodec 的损失细节和「loss balancer」

EnCodec（[2210.13438](https://arxiv.org/abs/2210.13438), Défossez et al., 2022）：

- 时域 L1（有，但只是其中一项）
- **多尺度 mel 谱损失**，窗长 2^5 到 2^11，同时用 L1 和 L2
- **MS-STFT 判别器**，5 个尺度，STFT 窗长 [2048, 1024, 512, 256, 128]
- feature matching
- VQ commitment
- 编码器 stride (2,4,5,8) → 24kHz 下 75 步/秒，中间还有 2 层 LSTM

**loss balancer**：按梯度的滑动平均幅度做归一化，这样权重 λ 的含义变成「这一项贡献了模型梯度的百分之几」，跟各项 loss 的绝对尺度脱钩。**我们要加运动/速度损失时，这个技巧直接可用**——否则 CE 和速度损失的量纲对不上，权重要瞎调。

### 补充：SpeechTokenizer 的解耦证据（支持「分道」而不是「排序」）

SpeechTokenizer（[2308.16692](https://arxiv.org/abs/2308.16692), 2023）用 HuBERT 蒸馏 RVQ 第一层，做法是**在 D 轴（特征维）上跨所有时间步最大化余弦相似度**，而不是逐时间步做——论文说这样"richer supervision signal"。

硬证据：
- RVQ-1 的 Phone-Normalized MI：SpeechTokenizer **0.71** vs EnCodec **0.28**。
- 换声实验：源的 RVQ-1（内容）+ 参考的 RVQ-2:8（音色）→ WER 2.6，说话人相似度 0.72。真的解耦了。
- mel 谱可视化：EnCodec 的 RVQ-1 还留着音色的「横条纹」，SpeechTokenizer 的 RVQ-1 "essentially loses all of the horizontal stripes"。

**教训**：想让不同 register 管不同的东西，得**显式给监督**，不能指望 OAT 的排序自己长出来。我们现在就是在指望它自己长，结果前几个 register 全被静态姿势占了。

---
## 二、我们自己的比特账（先算清楚，不然后面全是空话）

从 `e13_stream/*/meta.json` 读到的真实数字：

- 帧级流：`D = 3328`（256 个 token × f=13 维），每维 8 个 level = 3 bit。
- 每帧名义比特：`bits_per_frame_nominal = 9984`。
- 一个 2 秒窗口：60 × 9984 = **599,040 bit**。
- 我们的时间瓶颈：8 register × 6 维 × log2(5) ≈ **111 bit**。
- 压缩比：**约 5400 倍**。

作为对照，DAC 说自己是「~90x 压缩」，那已经是音频领域拿 NeurIPS spotlight 的成绩。我们在做 5400 倍。

当然帧级流本身冗余极大（3328 维描述一个姿势，远超必要），所以 5400 这个数字虚高。但**绝对值 111 bit / 2 秒是硬的**，这是瓶颈的真实容量上限。

### DCT 探针要诚实解读（重要）

DCT 探针「每维保留前 8 个频率」，如果是对全部 3328 维做的，那它保留的是 3328 × 8 = **26,624 个实数系数**，信息量比 111 bit 高好几个数量级。

所以探针**没有**证明「111 bit 够用」。它证明的是另一件同样重要的事：

> **运动在时间方向上是低频的。一个 8 阶的时间基（每 2 秒）就能表达 95–100% 的走路运动能量。**

这是关于**时间基函数**的结论，不是关于**比特预算**的结论。两件事要分开说，合起来才是完整的诊断：

1. 时间基：低阶就够 → 我们的解码器结构（帧 query 读 register）本来就是个基函数合成器，容量没问题。**问题不在这里。**
2. 比特预算 + 全局池化：111 bit 且无时间索引 → **问题在这里。**

### 一个粗糙的率失真下界

假设运动内在维度 ~8，DCT 探针说 2 秒内 8 阶就够，即每维带宽约 8 / (2 × 2s) = **2 Hz**。

- 奈奎斯特：2 Hz 带限信号每维每秒需要 4 个采样 → 8 维 × 4 = **32 个实数/秒**。
- 每个实数给 4–6 bit（这已经很省了）→ **128–192 bit/s**。
- 我们现在：**56 bit/s**。

**结论：我们的预算大约差 2.5–3.5 倍。** 不是差 100 倍，但也绝不是「够用只是训练没调好」。**8 个 register 对 2 秒来说确实偏少，但更致命的是这 8 个 register 没有时间索引。**

---
## 三、关于「全局池化」这一步，Perceiver 原文自己就说了

我们的 readout 是 Perceiver 式的：一组可学习的 query（register）对所有帧做 cross-attention。

Perceiver 原文（Jaegle et al., 2021, [2103.03206](https://arxiv.org/abs/2103.03206)）自己写得很清楚：

> "Attention is a permutation-invariant operation, and this property is preserved by the Perceiver and related models... A pure attention model will return the same output regardless of the order of its inputs."

> "But permutation invariance means that the Perceiver's architecture cannot in and of itself exploit spatial relationships in the input data."

也就是说：**Perceiver 默认是把顺序信息丢掉的**，全靠往输入上贴位置编码来补。

我们的代码确实贴了（`embed_frames()` 里加 `enc_time`），所以时间没有被完全抹掉。但有两个残留问题：

1. **register 输出端没有时间索引。** 8 个 register 是按「重要性」排序的（OAT prefix），不是按时间排序的。解码时 `reg_id` 只区分是第几个 register，不表示「管第几段时间」。
2. **readout 没有任何时间局部性约束。** 每个 register 都能读全部 60 帧。要表达「第 0.7 秒膝盖抬起来」，模型必须用注意力权重把这个时刻编码进一个全局向量里——这是绕远路。相比之下 stride 卷积或分块 readout 天生就有时间地址。

顺带一提：代码里的 `--mode chunk` 就是「每个 register 只读 T/8 的一段」的对照组，但它是当作**控制组**（control）写的，不是当作主方案。文献的证据方向是：**chunk 才该是主方案，全局才是该被淘汰的那个。**

Perceiver AR（[2202.07765](https://arxiv.org/abs/2202.07765), 2022）的存在本身也说明问题：它专门做了「cross-attention 压到少量 latent，但全程保持因果掩码」，就是为了在压缩的同时**不丢掉时间顺序**。

---

## 四、只看音频证据能得出的即时结论

> 这是扫描到一半时先落地的一版，只用了音频侧证据。**完整结论见第八节总综合**（那里合并了视频、频域、损失三个方向，并且推翻/加强了这里的部分排序）。

### 文献直接否定我们的哪些选择

| 我们的选择 | 文献怎么说 | 证据 |
|---|---|---|
| **全局 cross-attention 池化掉时间轴** | **强烈否定。** 没有任何一个成熟的时序 tokenizer 这么做。全部用 strided conv / patch，时间轴从头保留到尾。stable-codec 甚至只用 sliding-window 局部注意力 | SoundStream/EnCodec/DAC/Mimi/WavTokenizer/stable-codec 无一例外 |
| **只用逐帧 CE 训练** | **强烈否定，且有 ablation。** DAC：只用重建损失 → SI-SDR 9.12→1.07，bitrate efficiency 99%→62%，"has not learned to reconstruct phase" | DAC [2306.06546](https://arxiv.org/abs/2306.06546) |
| **拿逐帧 argmax 准确率当训练指标** | **否定。** Mimi 明确说去掉重建损失后客观指标变差但听感变好——逐点指标跟感知质量脱钩 | Mimi [2410.00037](https://arxiv.org/abs/2410.00037) |
| **8 token / 2 秒（4 tok/s）** | **偏低但不荒谬。** 最激进的语音 codec（Mimi）也有 12.5 个时间位置/秒；stable-codec 25 tok/s。粗算我们的率失真下界是 128–192 bit/s，现在 56 bit/s | 见第二节 |
| **靠 OAT 排序让 register 自然分工** | **否定。** SpeechTokenizer / Mimi 都是**显式监督**分道，不是靠排序涌现。不给显式监督，静态内容一定抢占前排（它最省 loss） | SpeechTokenizer [2308.16692](https://arxiv.org/abs/2308.16692) |
| **FSQ，6 维 5 level** | **支持。** stable-codec 用的是完全相同的配置（5^6 = 15,625），并且拿到极低码率 SOTA。路线没错 | stable-codec [2411.19842](https://arxiv.org/abs/2411.19842) |

### 目前最该做的四件事（音频证据支持）

**1. 让 register 带时间地址（把 `--mode chunk` 扶正）**
- 依据：所有 codec 都保留时间轴；stable-codec 只用局部注意力。
- 改法：`temporal_common.py` 的 `_chunk_mask` 已经写好了，把它从 control 变成默认。或者更好：改成 SNAC 式的**混合**——前 2 个 register 全局（管姿势/风格），后 6 个各管 1/6 的时间段（管动态）。
- 注意 `_chunk_mask` 现在的语义是 `m[r]=True; m[r,lo:hi]=False`，True 表示被屏蔽，所以 register r 只看得到自己那段。语义是对的。

**2. 把逐帧 CE 换掉，或者至少加上「频域/多尺度」项**
- 依据：DAC 的 ablation 是这份文献里最硬的一条。
- 改法：`train_temporal.py` 已经有 `--w_motion`（对帧间发生变化的位置加权）和 `--w_vel`（速度相对 MSE）两个 flag，但默认都是 0。**先把这两个打开跑对照**，这是零成本实验。
- 再进一步：加一个**多尺度时间损失**——把预测和 GT 的去量化 z 序列做不同窗长的 DCT/STFT，在系数上算 L1。这是 EnCodec 的多尺度 mel 损失搬到运动上的直接类比。

**3. 加周期性判别器（专治相位没锁）**
- 依据：DAC 的 MPD，周期 [2,3,5,7,11]，专门治周期性和音高 artifact；来自 HiFi-GAN [2010.05646](https://arxiv.org/abs/2010.05646)。
- 我们的病症（膝盖 xcorr lag0 ≈ 0）就是相位问题，走路/跑步/开合跳都是周期信号。
- 改法：一个小的时间卷积判别器，输入是去量化的 z 序列（或其一阶差分），按周期 reshape。配 EnCodec 的 loss balancer 来定权重，避免瞎调。

**4. 显式给「静态」和「动态」分道，不要靠 OAT 排序**
- 依据：SpeechTokenizer 的 RVQ-1 语义蒸馏（PNMI 0.71 vs 0.28）；Mimi 的 split RVQ 明确说是为了不让声学信息被挤进语义残差。
- 改法：把 register 显式分成两组。一组只喂「窗口内的平均姿势 / 低频 DCT 系数」，另一组只喂「去掉均值后的残差」。后者的重建损失只算残差，静态解在它上面拿不到分。
- 这跟做法 1 可以叠加：静态组全局池化没关系（它本来就该是全局的），动态组按时间分块。


---

## 五、频域 tokenization：DCT/小波 + 标量量化是所有真实编解码器的主力

这一节直接支持我们的 DCT 探针结果。结论先说：**「变换 + 标量量化」是成熟做法，「DCT + FSQ 用在人体运动上」是两个成熟半边的新组合——属于好的那种「新」。**

### 5.1 FAST（机器人动作，2025）：DCT + 取整 + BPE

**FAST: Efficient Action Tokenization for VLA Models**, Pertsch et al., 2025, [2501.09747](https://arxiv.org/abs/2501.09747)

配方（原文 Algorithm 1）：
1. 每个动作维度对整个 chunk 做 DCT
2. **scale-and-round**：`round(γ · C)`，**γ = 10**
3. 按维度交错展平，**低频在前**
4. BPE，词表 1024

**关键细节：FAST 不做硬截断。** 截断是**隐式**的——用 γ 取整后，小的高频系数直接变成 0，形成稀疏矩阵，BPE 再把连续的 0 压掉。所以旋钮是**量化步长**，不是**系数个数**。这就是 JPEG 量化表的逻辑。

数字：chunk = 1 秒。DROID @15Hz：朴素分箱 105 token/chunk vs FAST 29（3.6×）。叠 T 恤 @50Hz 双臂：朴素 **700** vs FAST **53**（13.2×）。π₀-FAST 用 **5 倍更少的 GPU 小时**追平 diffusion 版 π₀。

**FAST 对我们最重要的一句话**——它诊断了逐时间步目标为什么会崩：

> "When using the naïve per-timestep tokenization scheme, this marginal information *approaches zero* as the control frequency of the training signal increases: for smooth signals, as timesteps get shorter the change per timestep decreases proportionally."

后果原文：用分箱训练的模型在高频数据上"**simply copies the first action**"，因为"highly correlated action tokens *diminish* the effectiveness of the next token prediction objective"。

**「复制第一个动作」和我们的「输出平均姿势」是同一个病。** 这把我们的失败从「Perceiver 没训好」重新定性成「**目标函数本身没信息量**」。

对我们的直接启示：我们的 DCT 探针用固定 K=8，FAST 用固定量化步长（自适应保留带宽）。**开合跳只有 84% 而走路 95–100%，正好是固定 K 吃亏的场景**——跳跃需要更多高频系数。建议对比测「固定 K」vs「固定量化步长」。

FAST 的后续工作（供参考，以下 2026 条目为摘要级可信度）：
- **BEAST**（NeurIPS 2025, [2506.06072](https://arxiv.org/abs/2506.06072)）：批评 FAST 的 token 数不定，改用 B 样条基，token 长度固定、可并行解码、天然平滑。
- **OAT: Ordered Action Tokenization**（2026, arXiv:2602.04215）：**FSQ + transformer + register**，做「左到右因果有序 token 空间」，支持 prefix 解码。这大概就是我们 `--mode oat` 的出处。

### 5.2 经典编解码器：全是变换 + 标量量化

MP3（1993）、AAC（1997）、Opus 的 CELT 层（RFC 6716, 2012）全部是 **MDCT + 系数标量量化 + 熵编码**。JPEG 也一样：8×8 DCT → 量化表 → zigzag → 熵编码。**没有一个把逐样本预测当主力机制。**

FAST 的 scale-and-round 本质就是一张单值 γ 的 JPEG 量化表。

学术锚点：Ballé et al. 2016（[1607.05006](https://arxiv.org/abs/1607.05006)）把学习式压缩重新表述为「非线性变换 + **标量量化**」，明确跟 DCT 对标。这是从 JPEG 到学习式 codec 的桥，而且**它保留了标量量化**——FSQ 就是这条线的后代。

现代在 MDCT 域上做神经量化的工作：MDCTCodec（[2411.00464](https://arxiv.org/abs/2411.00464)）、Residual Scalar-Vector Quantization streamable codec（[2504.06561](https://arxiv.org/abs/2504.06561)）、MDCTNet（[2212.04583](https://arxiv.org/abs/2212.04583)）。

### 5.3 最像我们要做的事的两篇：Cosmos 和 Spectral Codecs

**NVIDIA Cosmos Tokenizer**（Cosmos World Foundation Model Platform, 2025, [2501.03575](https://arxiv.org/abs/2501.03575)）:

- 进编码器**之前**先做 **2 级 Haar 小波变换**，把帧分组 `{x₀, x₁:₄, x₅:₈, …} → {g₀, g₁, g₂, …}`，在 x、y **和 t** 三个方向各降 4 倍。
- 理由原文：小波变换"eliminates redundancies in pixel information"，让后面的层"focus on more semantic compression"。
- 离散 tokenizer 就是 **FSQ，latent 维度 6，levels (8,8,8,5,5,5) = 64,000 词表**。

**2025 年部署最广的视频 tokenizer，本质就是「小波变换 → 学习编码器 → FSQ」。** 而且它的 FSQ 也是 6 维——跟我们和 stable-codec 一样。这一条让「先变换、后 FSQ」变成完全不需要辩护的做法。

**Spectral Codecs for Non-Autoregressive TTS**（NVIDIA, 2024, [2406.05298](https://arxiv.org/abs/2406.05298)）——**这是我们的最近邻**：

- 直接对 **80 维 mel 谱**做 **FSQ**，8 个码本，levels **[8,5,5,5]** = 每个 1000 码，每码管 4 维，80 维 mel → 32 个量化维。
- 结论：用这个 spectral codec 训的非自回归 TTS，音质**显著好于**用 mel 谱或用时域 audio codec token 训的。
- **FSQ 打败 RVQ 的理由原文：RVQ "embeds most content in early codebooks"。**

**最后这句要划重点。** RVQ 的病是「内容全挤在前几个码本里」。我们的 OAT prefix 训练是**故意**制造这个结构（前面的 register 优先），结果就是**静态姿势占了前排，动态被挤到后面甚至挤没了**。这是同一个病理，而且我们是主动把它设计进去的。

### 5.4 时间序列基础模型：频域 tokenizer 用更小的词表赢了时域

- **Chronos**（Amazon 2024, [2403.07815](https://arxiv.org/abs/2403.07815)）：均值缩放 + 在 [-15,15] 上**均匀分箱**，B=4094 箱 ≈ 4096 词表。**逐时间步分箱，没有 patching**——正是 FAST 证明在高控制频率下会崩的那个方案。论文自己承认三个损失：精度损失、越界、以及"theoretically infeasible to model time series with a strong trend"。
- **TimesFM**（Google 2024, [2310.10688](https://arxiv.org/abs/2310.10688)）：**确认是连续的，不离散**。输入 patch 长 32，**输出 patch 长 128**（故意比输入长）。没有词表、没有量化。
- **TOTEM**（2024, [2402.16412](https://arxiv.org/abs/2402.16412)）：VQ-VAE，1D **strided CNN** 编码器，沿时间轴产生**不重叠的 token**。时域，无频率变换。
- **WaveToken**（2024, [2412.05244](https://arxiv.org/abs/2412.05244)）：缩放 → 小波分解 → **对小波系数阈值化并量化** → 自回归预测系数。42 个数据集上打败一众基础模型，而且"using a much smaller vocabulary (**1024 tokens**)"，平均排名第一。论文说小波能轻松抓住"**trends, sparse spikes, and non-stationary time series**"。

  **对照：Chronos 在时域用 4096 词表，WaveToken 在小波域用 1024 词表并且赢了。** 4 倍词表缩减——跟我们「8 个 DCT 系数就够」是同一个现象。

- **PatchTST**（Nie et al. 2022, [2211.14730](https://arxiv.org/abs/2211.14730)）——**反对全局池化的那条主引用**。原文：

  > "a single time step does not have semantic meaning like a word in a sentence, thus extracting **local semantic information** is essential."

  patch 长 16 / stride 8。整个时间序列领域面对「长窗口怎么变成少量 token」的答案是**沿时间做局部 patch**，从来不是全局池化摘要。

  **我们的 Perceiver 对全部 60 帧做全局 cross-attention 池化，正是 patching 被发明出来要避免的那个东西。全局池化没有任何机制保留「事情发生在什么时候」，所以平均姿势坍缩是可预期的结果，不是训练 bug。**

### 5.5 脑电：唯一一个对「重建目标本身」做了 A/B 的

**LaBraM**（Jiang et al., **ICLR 2024**, [2405.18765](https://arxiv.org/abs/2405.18765)）

它的 tokenizer 叫 "vector-quantized neural spectrum prediction"。解码器**不重建原始 EEG 波形**，而是重建**傅里叶幅度和相位**：

- 幅度 `A = √(Re² + Im²)`，相位 `φ = arctan(Im/Re)`
- 损失 = `‖ô_A − A‖₂² + ‖ô_φ − φ‖₂²`
- 码本 8192 × 64，ℓ₂ 归一化（低维 + 归一化是防坍缩的老套路，**FSQ 就是这个 hack 的正统继任者**）
- patch 长 200 采样 = **1 秒**，每通道切 patch。**又是局部 patch，不是全局池化。**

**重建目标的消融（Table 7，下游平衡准确率）：**

| 重建目标 | TUAB | TUEV |
|---|---|---|
| LaBraM（傅里叶谱） | 0.8140 | **0.6409** |
| 原始信号重建 | **0.8261** | 0.5630 |
| 谱目标但不做离散 tokenizer | 0.8166 | 0.5730 |

**要诚实转述**：二分类的 TUAB 上原始信号重建反而略好（0.8261 vs 0.8140）。但在更难的 6 类事件分类 TUEV 上，谱目标赢 **7.8 个点**，而且离散 tokenizer 又贡献了约 +6.8。

所以站得住的说法是：**「谱目标 + 离散化」才让表征带上事件结构，而不是只带粗糙的信号统计量。** 不能说成「原始目标一定失败」。

### 5.6 频域量化比时域自编码更保动态——有消融的那篇

**TimeVQVAE**（Lee et al., AISTATS 2023, [2303.04743](https://arxiv.org/abs/2303.04743)）：在**时频域（STFT）**做 VQ，不在时域。原文：

> "the time-frequency domain provides richer information, results in **better VQ modeling than in the time domain**"

而且它把 VQ **拆成低频和高频两条支路**，理由是"LF components are more predictable and compressible than HF components by nature"，这样能"disentangling HF components, such as spikes, from LF components, such as trends"。

**这个 LF/HF 双支路设计我们可以直接抄**：大部分 FSQ 容量给前几个 DCT bin（低频，管整体轨迹），另开一条便宜的支路管残差（高频，管跳跃/落地的尖峰）。**开合跳只有 84% 正是缺高频支路的症状。**

### 5.7 人体运动用 DCT 的先例（我们 8 系数探针的直接先验）

- **Mao et al., ICCV 2019**（[1908.05436](https://arxiv.org/abs/1908.05436)）：把每个关节的时间轨迹表示成 DCT 基的线性组合。**系数个数（消融原文）**：3D 表示用 **15 个**（短期）/ **30 个**（长期）；角度表示用 20 / 35 个。序列是 10 观测 + 10 或 25 预测帧（共 20–35 帧）。补充材料："we can discard some of the high frequency coefficients to achieve better performance in 3D space… use the first **15** DCT coefficients as input."

  **Table 6 的消融显示误差对系数个数是非单调的——用更多 DCT 系数有时反而让平均误差变大**，因为截断压掉了抖动。这是一条已发表的结论：**激进的 DCT 截断不只是「损失可接受」，而是「主动有益」。**

- **History Repeats Itself**（Mao et al., ECCV 2020, [2007.11755](https://arxiv.org/abs/2007.11755)）：Human3.6M 上 M=50 观测帧，"In practice, we can truncate some high frequencies to avoid predicting jittery motion."
- **siMLPe**（WACV 2023, [2207.01567](https://arxiv.org/abs/2207.01567)）：T=50 输入帧，全部运算在 DCT 空间做完再 IDCT。Human3.6M MPJPE 消融：不用 DCT 9.9/22.4/47.3/110.5 mm vs 用 DCT 9.6/21.7/46.3/109.4（80/160/320/1000ms）。提升不大但一致，长时程最明显。

**反方引用（要预备好被人问）**：Human Motion Capture Data Tailored Transform Coding（2014, [1410.4730](https://arxiv.org/abs/1410.4730)）明确说"directly borrowing image or video compression techniques, such as discrete cosine transform, does not work well"，主张学习**数据相关的正交基**再熵编码量化系数。

反驳：他们的目标是长异构序列的近无损归档压缩，不是给生成模型做 2 秒 token 瓶颈。而且「学习的基」正是一个小编码器**架在 DCT 之上**能给的东西。

### 5.8 一个残酷的对照：T2M-GPT

**T2M-GPT**（CVPR 2023, [2301.06052](https://arxiv.org/abs/2301.06052)）：CNN 运动 VQ-VAE，**时间下采样 l = 4**，码本 512×512。HumanML3D 重建 FID 0.070（真实运动 0.002）。

也就是说，这个领域对 60 帧的标准做法是：**约 15 个局部 token，512 词表，卷积产生**。

**我们在要求 8 个全局 token 做得比 15 个局部 token 更好——token 少约 2 倍，还额外扔掉了时间局部性。**

### 5.9 频域这一节的判决

**「变换 + 标量量化」是成熟配方；「DCT + FSQ 用在人体运动上」是新组合，但两个半边都各自成熟。**

- 变换后量化是所有已部署 codec 的默认（JPEG/MP3/AAC/Opus），也是 2025 最强视频 tokenizer 的做法（Cosmos = Haar 小波 + FSQ，64k 词表）。
- 用在**行为/决策序列**上，2025 年起已经成熟：FAST 是机器人动作的 DCT+取整+BPE，是 2026 年新论文的标准基线。
- **FSQ 直接量化谱系数**已发表：NVIDIA Spectral Codecs，mel + FSQ [8,5,5,5]。
- 「频域量化优于时域自编码」有消融支撑：TimeVQVAE、LaBraM。
- 人体运动 DCT 截断到 15–35 系数（20–35 帧）已发表，并且有「系数更多反而更差」的消融（Mao ICCV 2019）。
- **没查到的**：把 FSQ 用在截断的时间 DCT 系数上、当作全身人体运动的 token 词表。所有零件都在，组装是新的。


---

## 六、视频 tokenizer

### 6.0 先看数字

我们是 **4 token/秒**。视频领域**全部**在 **64–600 token/秒**之间，包括那些卖点就是「极致紧凑」的。

| 系统 | token/秒 |
|---|---|
| MAGVIT-v2 / VideoPoet | ~602（1280 token / 2.125 秒，128×128） |
| LARP | ~64 |
| VideoFlexTok（目前文献地板） | ~67（672 token / 10 秒） |
| **我们** | **4** |

**没有任何一个能用的视频 tokenizer 落在 4 token/秒的 1.5 个数量级以内。** 光这一条就能预测出「平均姿势 + 动作类别」。

### 6.1 根因：Mathieu 2015 早就证明了

**Deep Multi-Scale Video Prediction Beyond Mean Square Error**, Mathieu et al., 2015, [1511.05440](https://arxiv.org/abs/1511.05440)

原文：如果一个像素合理取值可能是 v₁ 或 v₂，那么

> "the value v_avg=(v₁+v₂)/2 minimizes the ℓ₂ loss over the data, **even if v_avg has very low probability**."

**逐帧回归损失在数学上就是奖励「对未来取平均」。我们输出的平均姿势就是 v_avg。**

这个论证**原封不动**适用于我们的逐帧 CE：CE 对每帧 code 的**边缘分布**做优化，最优解是边缘众数；一个 2 秒运动窗口的 code 边缘众数就是平均姿势。**我们观测到的 22–73% 能量保留，就是一个边缘众数解。**

Mathieu 给的三个解法，每个都对我们有用：
1. **对「帧序列」做对抗损失**——因为"the sequence (X, Y_avg) … is not a likely sequence"。**序列级的判别器能否掉平均解，哪怕逐帧损失很喜欢它。**
2. **Gradient Difference Loss**——惩罚「差分的差分」。这是我们 `--w_vel` 的近亲，但更狠。
3. 多尺度。

### 6.2 SweetTok：跟我们的 bug 最像的一篇

**SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization**, 2024, [2412.10443](https://arxiv.org/abs/2412.10443)

它直接把病描述出来了：把视频压成一串 query/token

> "may lead to difficulty in learning intertwined spatial temporal information resulting in low reconstruction performance"

而且解码器很难

> "learn the motion information of the same pixel across consecutive frames."

**它的解法（Decoupled Query AutoEncoder）**：
- 空间编码器**只压第一帧** → 256 个 query
- 时间编码器压**相邻帧之间的残差** → 1024 个 query
- 空间解码器重建第 1 帧 → 广播出去 → 时间解码器把动态加回去
- 再加一个 Motion-enhanced Language Codebook

**消融（最值钱的数字）**：

| 配置 | rFVD |
|---|---|
| 普通联合 query tokenizer（≈我们现在） | 35.15 |
| + 解耦（外观 / 帧间残差） | 24.80 |
| + 运动码本 | **20.46** |

**光是「把外观和帧间差分拆开」就换来约 30% 的 rFVD 提升；再加运动码本又 18%。合计比 LARP-L 好 42.8%。**

也就是说：**「不解耦的 query 式 tokenizer」的代价被量出来了，大约是 +72% rFVD。我们现在就是那个不解耦的版本。**

同族工作（2026 条目为摘要级可信度）：
- **DeRA**（2025, [2512.04483](https://arxiv.org/abs/2512.04483)）：1D 视频 tokenizer，但把编码**拆成外观流和运动流**，各自对齐预训练视觉模型。比 LARP 好 25% rFVD。还要专门加一个 Symmetric Alignment-Conflict Projection，因为两路监督在梯度空间打架。
- **VTok**（2026, arXiv:2602.04202）：保留一个关键帧的完整空间特征，之后**每帧只编成一个残差 token**。
- **ProMAG**（2025, [2501.05442](https://arxiv.org/abs/2501.05442)）：关键帧 embedding + 残差 embedding。

**关键帧 + 残差**这个骨架在四篇里反复出现。这是视频领域对付静态坍缩的标准答案。

### 6.3 TiTok：低 token 数的失败模式，字面上就是我们的症状

**An Image is Worth 32 Tokens**, 2024, [2406.07550](https://arxiv.org/abs/2406.07550)

TiTok 用 ViT 全局注意力把图像压成 1D token（256×256 → 32 个 token）。它记录的低 token 失败模式：

> "the model tends to keep the **high-level layout or salient objects** when the latent representation size is limited."

**语义活下来，低层细节死掉。** 这就是「平均姿势 + 动作类别」的图像域原话。64 token 以下重建明显变抽象，64–128 之后收益饱和。

**还有一条对我们极其重要**：TiTok 说**直接单阶段训练 1D token 是不稳定的**。它的解法是两阶段：先拿现成 VQGAN 的 **proxy code**（token 级目标）热身，再选择性地把解码器 finetune 到像素。**1.70 rFID vs 单阶段朴素训练的 5.15。**

**我们现在做的「对着已有的 FSQ code 流做逐帧 CE」，正是 TiTok 的第一阶段 proxy-code 热身。TiTok 把它当热身，我们把它当终点。**

### 6.4 LARP：唯一成功的「整体 query」视频 tokenizer，而且它加了额外目标

**LARP**, 2024, [2410.21264](https://arxiv.org/abs/2410.21264)：用学习式 query token（Perceiver 那一族，跟我们的 register 同源），16 帧 @128² → 1024 token，时间降 4 倍 → 约 64 token/秒。

**它不靠纯重建**：额外联合训练一个 21.7M 的 AR 先验模型，把 latent 空间推成「自回归可预测」的结构，推理时丢掉。

**消融很扎心**：把 AR 先验去掉，**重建指标 rFVD 反而变好 31→23，但生成指标 gFVD 崩了 107→190**。

**教训：重建指标和「动态有没有用」不是同一个目标。** 我们现在盯的 `budget_level_acc` 属于前者。

### 6.5 时间轴怎么处理：全是「保留时间轴，最后才压、压得最少」

- **MAGVIT**（[2212.05199](https://arxiv.org/abs/2212.05199)）：浅层用 **3D 卷积**，深层才用 2D。**时间在高分辨率下处理，很晚才混合**——跟「先把时间池化掉」正好相反。损失 = 逐帧感知损失 + **膨胀成 3D 的 StyleGAN 判别器** + LeCam 正则，GAN 从第 0 步就开。
- **MAGVIT-v2**（[2310.05737](https://arxiv.org/abs/2310.05737)）：**因果 3D CNN**，时间卷积只往**过去**方向 pad。时间降 4×，空间降 8×8。
- **Phenaki / C-ViViT**（[2210.02399](https://arxiv.org/abs/2210.02399)）：因果时间注意力（动机是变长，副作用是时间从不被全局池化）。**损失 = L_VQ + 0.1·L_Adv + 0.1·L_IP + 1.0·L_VP + 1.0·L₂**，其中 **L_VP 是在 I3D 特征上的「视频」感知损失，权重 1.0，跟 L₂ 一样重**。这是一个「只有把动态做对才能降下去」的损失。收益：**FVD 65.78 vs 逐帧基线 166.6**，token 还少 40%。
- **Cosmos**（[2501.03575](https://arxiv.org/abs/2501.03575)）：小波先分解（见第五节），再用**时空可分解的因果 3D 卷积**。**FSQ，6 维，levels (8,8,8,5,5,5) = 64,000 词表。**
- **VidTok**（[2412.13061](https://arxiv.org/abs/2412.13061)）：**FSQ，码本 4,096 / 32,768 / 262,144，利用率约 100%，而 VQ 只有 0.2%。**
- **Large-Motion Video VAE**（[2412.17805](https://arxiv.org/abs/2412.17805)）：明说"entangling spatial and temporal compression by merely extending the image VAE to a 3D VAE can introduce **motion blur** and detail distortion"。解法：先压空间（kernel (1,3,3)，只动空间），再单独一个**运动压缩模块**。

### 6.6 LFQ 就是 FSQ 的兄弟——我们的路线没有代价

MAGVIT-v2 的 **LFQ（lookup-free quantization）**：干掉码本 embedding，latent 维度 = K，码本 = ⨉ᴷ{−1,+1}，量化就是 `ẑ = sign(z)`，索引就是二进制读数，|C| = 2^K。配一个熵惩罚。

**LFQ 就是 FSQ 取 L=2（二值）再加熵损失的特例。** 两者同属「隐式乘积码本、无查表」这一族。

所以**「FSQ-committed」在视频领域完全不是约束**：MAGVIT-v2 的全部结果用 FSQ 都能达到，而 Cosmos 和 VidTok 干脆直接用 FSQ。

Open-MAGVIT2（[2409.04410](https://arxiv.org/abs/2409.04410)）复现到 **K=18 → 2^18 = 262,144 码，100% 码本利用率**。

**MAGVIT-v2 的关键消融：词表越大，重建和生成单调地越好**——推翻了老 VQ-VAE 时代「码本不能太大」的民间智慧。**这对我们是好消息：如果要加容量，把 FSQ 词表做大是有背书的方向。**

### 6.7 时间压缩率的安全线——硬数字

**ProMAG**（[2501.05442](https://arxiv.org/abs/2501.05442)）给了我们最想要的消融。MCL-JCV，z_dim=8：

| 时间压缩率 | MAGVIT-v2 PSNR | MAGVIT-v2 rFVD |
|---|---|---|
| 4× | 30.99 | **57.91** |
| 8× | 28.47 | **219.36** |
| 16× | 26.15 | **255.51** |

**从 4× 到 8×，rFVD 恶化约 3.8 倍。4× 是安全区，8× 已经很痛，16× "leads to poor reconstruction quality"。**

**我们的时间压缩率：60 帧 → 8 个 register = 7.5×，而且这 8 个还没有时间索引，实际比 7.5× 更糟。**

ProMAG 最有启发的一个对照实验：

> **一个 4× 的模型去重建 6fps（抽帧后）的视频，比一个 16× 的模型重建原始 24fps 的视频效果更好。**

**扔掉帧、然后温柔地压，胜过硬压时间。** 直接可用：**把我们的 60 帧抽成 20 或 30 帧，然后用温和的压缩率，可能直接打败 8-register 瓶颈。**

### 6.8 VidTok 的两个消融，直接决定我们该怎么加容量

- **"Allocating information to channel dimensions yields better performance than temporal dimensions at equivalent total information capacity."** 同样的总信息量，**latent 变宽比 latent 时间步变多更好**。

  → 我们的 register 应该**变胖**（FSQ 维度从 6 加到 8–10，或者 level 从 5 加到 8），而不只是变多。但 8 个时间步对 60 帧来说无论如何都在地板以下，两件事都要做。

- **用 3 FPS 训练明显优于 8 FPS**——训练数据里帧间位移更大，逼得模型不能忽略动态。

  → **数据侧的杠杆：训练时对窗口抽帧，人为放大帧间差分，让「复制上一帧」这条捷径变贵。** 我们的 `--aug_time` 已经在做时间尺度增强，但目的是变长鲁棒性，不是放大 delta。

- 另外：非因果比因果重建更好（能看未来有帮助）。因果是为流式付的代价，不是为质量。**我们不做流式，就不该自找因果约束。**

### 6.9 视频领域保住运动的三个机制

1. **绝不全局池化时间轴；把时间当成有序的轴，压缩率 ≤4×。** 因果 3D 卷积 / 因果时间注意力 / 时空可分解注意力。时间是**最后压、压得最少**的那个维度。
2. **一个逐帧损失给不了的「序列级」训练信号。** 3D/时间判别器（MAGVIT 把 StyleGAN 判别器膨胀成 3D；DVD-GAN 的 D_T）、视频感知损失（Phenaki 的 I3D，权重 1.0）、联合训练的 AR 先验（LARP）、或预测式目标。**每一个正经的视频 tokenizer 至少有一个。**
3. **显式拆开「外观」和「运动」——关键帧 + 差分。** SweetTok、DeRA、VTok、ProMAG、Video-LaVIT 全是这个骨架。

**DVD-GAN**（[1907.06571](https://arxiv.org/abs/1907.06571)）把机制 2 讲得最清楚：判别器拆成 **D_S**（抽 8 个**单独的全分辨率帧**，各自评判）和 **D_T**（看**整段视频**，但空间上 2×2 平均池化）。论文原话：**D_T 提供的是 "the learning signal to generate movement"，因为 D_S 单独无法评估时间一致性。**

**这个设计是「扔掉空间细节，保留全部时间，单独评判」——正好是我们池化方向的反面。**


---

## 七、保住动态的损失函数

### 7.1 「逐点损失 → 平均姿势」在运动领域是有明确原话的

**Ruiz et al. 2018, Human Motion Prediction via Spatio-Temporal Inpainting**, [1812.05478](https://arxiv.org/abs/1812.05478)：

> **"The use of this metric [L2] to train a deep network favors motion predictions that converge to a static mean pose."**

还有 "there is also the tendency to produce very small motions"。**这就是我们的症状，一字不差。** 他们的解法是全序列对抗，**完全不用逐帧 L2**。

同族：
- **Martinez et al. 2017**（[1705.02445](https://arxiv.org/abs/1705.02445)）："a **zero-velocity** prediction is a simple but hard-to-beat baseline that largely outperforms the state of the art"；自己的模型 "produces smooth, continuous predictions in the short term, but **converges to a mean pose**"。
- **Fragkiadaki et al. 2015 (ERD)**（[1508.00271](https://arxiv.org/abs/1508.00271)）："soon converges to the mean pose"。
- **acRNN**（[1707.05363](https://arxiv.org/abs/1707.05363)）："motion generation tends to converge to a mean position over time"。

TTS 侧同一个病：Tacotron 2（[1712.05884](https://arxiv.org/abs/1712.05884)）说预测的谱 "oversmoothed and less detailed than the ground truth – **a consequence of the squared error loss**"。ACL 2022 的 Revisiting Over-Smoothness in TTS（[2202.13066](https://arxiv.org/abs/2202.13066)）说得最清楚：MAE/MSE "implicitly assume that data points … are independent to each other and follow a **unimodal** distribution"，而真实分布是多峰的，所以必然过平滑。

**FastSpeech 2**（[2006.04558](https://arxiv.org/abs/2006.04558)）给的框架最好用：把问题定义成 **"information gap"**——输入不包含预测目标所需的全部信息，于是模型只能对缺失的那部分取平均。解法是把 pitch/energy/duration 显式加进条件。

**翻译到我们**：**如果 8 个全局 token 里没有相位，逐帧 CE 就一定会对相位取平均。唯一的出路是要么把相位放进条件/编码，要么在损失里给相位定价。**

### 7.2 最重要的一个便宜诊断：我们是不是在用 argmax 解码

CE 训练本身学的是**完整条件分布**，不是平均。**坍缩发生在解码那一步。**

Holtzman et al. 2019（[1904.09751](https://arxiv.org/abs/1904.09751)）："using likelihood as a decoding objective leads to text that is **bland and strangely repetitive**"。

**离散版的「平均姿势」就是「边缘众数」，而它一样是静态的。** 对每帧 code 的边缘分布取 argmax ≈ 一条几乎不变的 code 序列 ≈ 一个静态姿势。

**我们的代码确实在 argmax**：`train_temporal.py` 的 `evaluate()` 里是 `logits.float().argmax(-1)`。

> **所以第一个该跑的实验（10 分钟）：把解码从 argmax 换成按 categorical 采样（或 top-k / nucleus），再量一次运动能量保留率。**
>
> - 如果能量明显回来 → 22–73% 里有一部分是**解码 artifact**，不是损失的锅。
> - 如果不回来 → token 里确实没有动态信息，得动结构和损失。
>
> 注意：采样能补回**能量**，但**补不回相位锁定**——相位如果没被编进 token，采样也变不出来。所以这个实验的价值是**干净地把两种失败分开**。

### 7.3 我们可以排除一个嫌疑：解码器绕过 token

VAE 领域的经典问题（Bowman et al. 2016, [1511.06349](https://arxiv.org/abs/1511.06349)）：解码器如果能看到历史，就会「走捷径」直接从历史推，把 latent 晾在一边（posterior collapse）。Chen et al. 2017 的 Variational Lossy Autoencoder（[1611.02731](https://arxiv.org/abs/1611.02731)）说得更绝：**局部自回归解码器能建模的东西，它一定会自己建模，这部分信息永远进不了 latent。**

**但我们的解码器看不到历史。** `decode_frames()` 的注释写明："No temporal mixing: every frame query sees only the registers, never other frames"。

**所以「解码器绕过 token」这个嫌疑可以排除。** 这是个好消息，也是个坏消息：
- 好消息：不用去做 word-dropout 那类干预。
- 坏消息：**token 确实拿到了它该拿的位置，是 token 本身没装下动态。** 病因锁定在「全局池化的瓶颈形状」+「逐帧 CE 的目标」上。

### 7.4 具体的损失项和它们的实测收益

**多分辨率 STFT 损失——整份扫描里最大的单项消融差距**

Parallel WaveGAN（2019, [1910.11480](https://arxiv.org/abs/1910.11480)）：

> **单分辨率 STFT 损失 MOS 1.36 ± 0.07 vs 多分辨率 STFT 损失 MOS 4.06 ± 0.10**

2.7 个 MOS，纯粹来自「用多个时频分辨率」。理由原文：多分辨率 "prevents the generator from being **overfit to a fixed STFT representation**"。

各家的窗长配置：
- Parallel WaveGAN：3 个分辨率，FFT 1024/2048/512
- SoundStream：`s ∈ {2^6 … 2^11}`，hop `s/4`，**没有任何时域项**
- EnCodec：7 个尺度，窗长 `2^5 … 2^11`；**时域 L1 权重 λ_t=0.1，谱项 λ_f=1，对抗和 feature 各 3**——时域项被压了 10–30 倍，只是稳定器
- DAC：窗长 **[32, 64, 128, 256, 512, 1024, 2048]**，mel bin 数**随窗长变** [5,10,20,40,80,160,320]。原文警告："fixing mel bin size leads to **holes in the spectrogram** especially at low filter lengths"

**搬到我们身上**：60 帧 @30fps，用窗长 **[4, 8, 16, 32, 60]** 帧，hop = 窗长/4，频率 bin 数随窗长缩放。**这一项直接给「每个频带的运动能量」定价——正好是我们在掉的那个指标。**

**诚实提醒：arXiv 全文扫描没找到任何把多分辨率 STFT 损失用在音频以外的工作。** 动机很足，但在运动上是未经验证的。

**速度 / 加速度损失——最便宜，但天花板要说清楚**

T2M-GPT（[2301.06052](https://arxiv.org/abs/2301.06052)）：`L = smoothL1(X, X̂) + α·smoothL1(V(X), V(X̂))`，α=0.5。

- **消融：α=0（无速度项）重建 FID 0.112 → α=0.5 时 0.070，降 37%。**
- 损失类型消融（α=0.5）：L1 得 0.144，**smooth-L1 得 0.070，纯 L2 得 0.292**——**纯 L2 比 smooth-L1 差 4 倍。**

MDM（[2209.14916](https://arxiv.org/abs/2209.14916)）用同样的速度项加脚部接触损失，并且诚实报告了一个反直觉现象：**去掉脚接触损失 FID 反而略好（12.81 vs 13.08），但 "without it we witness artifacts such as shakiness and unnatural gestures"。指标可以往反方向动，而观感在变好。**（这条完全对应我们「先看动画再看数字」的原则。）

**天花板警告**：**速度的 L1 损失，其最优解依然是「速度的条件均值」，而对相位没锁的数据来说这个均值接近 0。** 它能修抖动和平滑度，**单靠它修不好相位锁定。** 我们的 `--w_vel` 属于这一类——值得跑，但别指望它是解药。

**周期折叠判别器（MPD）——最该抄的那个**

HiFi-GAN（[2010.05646](https://arxiv.org/abs/2010.05646)）的 MPD 动机原文：

> "As speech audio consists of **sinusoidal signals with various periods**, the diverse periodic patterns underlying in the audio data need to be identified."

周期取**质数 [2,3,5,7,11]**，"to avoid overlaps as much as possible"。机制是把 1D 信号 reshape 成 2D，让每个子判别器只看**等间隔采样**的点。

**消融（MOS）：完整 4.10 → 去掉 MPD **2.28** → 去掉 MSD 3.74 → 去掉 mel 损失 3.25。**

**去掉「周期」判别器是整个 vocoder 文献里单项影响最大的消融，−1.82 MOS。**

**搬到我们身上几乎是原样照抄**：步态是周期的，把 60 帧的关节轨迹按 stride [2,3,5,7,11] 折成 2D 再卷积。**这个判别器就是为「周期错了、相位错了」而生的，而我们的症状正是膝盖 xcorr lag0 ≈ 0。**

配 DVD-GAN（[1907.06571](https://arxiv.org/abs/1907.06571)）的拆分：`D_S` 看单帧姿势合理性，`D_T` 用 1D 时间卷积看整段 60 帧但降低关节分辨率。

**feature matching——加判别器时必须同时加**

SoundStream **λ_feat=100 vs λ_adv=1**（feature 项是主项）；EnCodec λ_feat=3（相对 L1，按特征幅度归一化）；HiFi-GAN λ_fm=2。**它是让对抗训练能训起来的东西，不是锦上添花。判别器一进来就要一起加。**

### 7.5 DeepPhase：让「8 个 token」真的可行的那个结构

这是对我们最贴身的一条线。

**DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds**（Starke, Mason, Komura, **SIGGRAPH 2022**, [dl.acm.org/doi/10.1145/3528223.3530178](https://dl.acm.org/doi/10.1145/3528223.3530178)）

机制：1D 卷积编码器把运动窗口映射到 M 个 latent 通道；**一个可微的 FFT 层**把**每个通道**参数化成 **振幅 A、频率 F、偏置 B、相移 S**；latent 再被**重建成参数化的正弦波**。

**也就是说瓶颈被强制成 `(A, F, B, S)` 每通道，而不是一个自由向量。**

**关键：相位是被瓶颈的「形状」强制出来的，不是靠某个损失项。**

**WalkTheDog**（2024, [2407.18946](https://arxiv.org/abs/2407.18946)）是**向量量化版的周期自编码器**——已发表工作里跟我们要做的东西最像的一个。它用码本量化**振幅**，当作 "a narrow bottleneck to regularize unsupervised learning of semantic motion clusters"。**它的损失只有 `L_rec + λ_vq·L_vq`**——再次说明**相位结构全部来自 FFT 参数化，不来自损失**。

前提原文："An intrinsic property of motion is its periodic structure. Common locomotion such as walking and running can be effectively parameterized by a **linear phase variable**."

后续可挖：Motion In-Betweening with Phase Manifolds（[2308.12751](https://arxiv.org/abs/2308.12751)）、part-wise phase（[2503.08180](https://arxiv.org/abs/2503.08180)）、SMAP 的 VQ-PAE（[2505.19463](https://arxiv.org/abs/2505.19463)）。

### 7.6 token 率的全域对照表（含运动领域）

| 系统 | 年份 | 时间栅格 | 码本 | **token/秒** | 码率 |
|---|---|---|---|---|---|
| DAC @8kbps | 2023 | 86 Hz | 9×1024 | **774** | 8 kbps |
| SoundStream / EnCodec @6kbps | 2021/22 | 75 Hz | 8×1024 | **600** | 6 kbps |
| MAGVIT-v2（视频） | 2023 | — | 2^18 LFQ | **~600** | — |
| EnCodec @1.5kbps | 2022 | 75 Hz | 2×1024 | **150** | 1.5 kbps |
| Mimi / Moshi | 2024 | 12.5 Hz | 8×2048 | **100** | 1.1 kbps |
| SNAC | 2024 | 12/23/47 Hz | 4096 | **82** | 0.98 kbps |
| BigCodec | 2024 | 80 Hz | 1×8192 | **80** | 1.04 kbps |
| WavTokenizer | 2024 | 75 / 40 Hz | 1×4096 | **75 / 40** | 0.9 / 0.48 kbps |
| X-Codec2 / Llasa | 2025 | 50 Hz | 1×65536 **FSQ** | **50** | 0.8 kbps |
| HuBERT units | 2021 | 50 Hz | 100–500 | **50** | — |
| **Stable Codec / TAAE** | 2025 | 25 Hz | **FSQ** | **25** | **0.4 kbps** |
| Single-Codec | 2024 | 23.4 Hz | 1×8192 | **23.4** | **0.304 kbps** |
| **MoMask（运动）** | 2023 | 5 Hz × 6 RVQ | 6×512 | **30** | ~270 bps |
| **T2M-GPT（运动）** | 2023 | 5 Hz | 1×512 | **5** | ~45 bps |
| **我们** | 2026 | **4 Hz（无时间索引）** | FSQ 5^6 | **4** | **~56 bps** |

**运动领域的率失真曲线（唯一一条已发表的）**：

> **MoMask（30 tok/s）重建 FID 0.019 vs T2M-GPT（5 tok/s）0.070**（HumanML3D）；KIT-ML 上 **0.112 vs 0.472**。
>
> **token 率从 5 提到 30，重建 FID 改善 3.7–4.2 倍。**

**质量悬崖的证据**：
- WavTokenizer 报告：**把 DAC 砍到只剩 1 个量化器（~100 tok/s），UTMOS 从 3.9097 崩到 1.4940。** 而专门为低码率设计的 WavTokenizer 在 40–75 tok/s 还能保持 3.6–4.0。**架构决定悬崖在哪，不只是码率。**
- **TAAE 的关键消融：25 Hz 下量化版 SI-SDR 4.73 vs 连续版（不量化）5.08。** 量化的代价几乎免费，**真正的约束是帧率，不是量化器。** 这条对我们很重要：**别再怀疑 FSQ 了，去改帧率/时间结构。**

### 7.7 一个信息率锚点

**Coupé, Oh, Dediu & Pellegrino 2019, Science Advances 5(9):eaaw2594**（[科学网页](https://www.science.org/doi/10.1126/sciadv.aaw2594)）：跨 17 种语言，人类语言的平均信息率是 **39.15 bits/s（SD 5.10）**。各语言用「音节率 × 信息密度」互相补偿，最后都收敛到约 39 bits/s。

对比：**能用的最低语音 codec 在 304–800 bps，是语言信息率的 8–20 倍。** 那多出来的部分买的是说话人身份、韵律、信道——也就是「**是谁、用什么语气说的**」，而不只是「**说了什么词**」。

**我们在约 56 bit/s。**

> **换句话说：我们的瓶颈大致只有「词」的预算，没有「语气」的预算。**
>
> **观测到的「平均姿势 + 动作类别」不是 bug，它就是 56 bit/s 下的率失真最优解。动作类别 = 词；步态相位 = 韵律。而韵律正是真实语音 codec 那 8–20 倍余量买的东西。**

注意：**人体运动没有对应的「39 bits/s」已发表数字**，这个类比是我们自己建的论证，不是文献里的结论。引用时要说清楚。


---

## 八、总综合

### 8.1 一句话诊断

**我们的失败不是 bug，是三个已发表结论的必然后果的叠加：**

1. 逐帧回归/CE 损失的最优解**在数学上就是时间平均**（Mathieu 2015）。
2. 全局注意力池化**没有任何机制保留「事情发生在什么时候」**（Perceiver 原文自陈；PatchTST 的整个动机）。
3. 4 token/秒**低于所有领域已发表地板的 1.5 个数量级**（音频 23–86，视频 64–602）。
4. 外加一个便宜的疑点：**解码用 argmax**，而边缘众数天然是静态的（Holtzman 2019）。

任一条单独出现都会掉能量。四条一起上，输出必然是「平均姿势 + 动作类别」。

**好消息是：量化器（FSQ）、有序 register、解码器结构这三样都没问题**，文献全都支持。要改的是**时间轴的形状**、**目标函数**和**预算**。

### 8.2 文献明确否定我们的哪些选择

| 我们的选择 | 判决 | 最硬的证据 |
|---|---|---|
| **全局 cross-attention 池化掉时间轴** | **否定，无一例外** | 音频全部用 strided conv；视频全部用因果 3D conv / 因果时间注意力；PatchTST："a single time step does not have semantic meaning… extracting local semantic information is essential"；SweetTok 把不解耦 query 的代价量到 **+72% rFVD** |
| **只用逐帧 CE 重建** | **否定，有数学证明 + 多个消融** | Mathieu 2015：v_avg 最小化 ℓ₂ "even if v_avg has very low probability"；DAC：只用重建损失 SI-SDR 9.12→1.07、"has not learned to reconstruct phase"；Phenaki I3D 视频感知损失权重 1.0 |
| **拿逐帧 argmax 准确率当主指标** | **否定** | LARP：去掉 AR 先验后 **rFVD 变好 31→23 但 gFVD 崩 107→190**；Mimi：去掉重建损失客观指标变差但听感更好 |
| **8 token / 2 秒（4 tok/s）** | **远低于地板** | 视频地板 ~64–67 tok/s；语音最激进的 Mimi 12.5 个时间位置/s、stable-codec 25 tok/s；ProMAG：时间压缩 4×→8× 时 rFVD 恶化 3.8 倍，我们是 7.5× 且无时间索引 |
| **靠 OAT 排序自然分工** | **否定** | Spectral Codecs 说 RVQ 输给 FSQ 就是因为 "embeds most content in early codebooks"——**我们是故意把这个病理设计进去的**；SpeechTokenizer / Mimi / SweetTok / DeRA 全是**显式分道** |
| **对着已有 code 流做 proxy-code CE** | **只是热身，不是终点** | TiTok 明确把 proxy-code 阶段当 stage-1 warm-up，之后 finetune 解码器，1.70 vs 5.15 rFID |
| **FSQ，6 维 × 5 level** | **支持，路线正确** | stable-codec 完全相同配置拿到 400bps SOTA；Cosmos FSQ 6 维 (8,8,8,5,5,5)=64k；VidTok FSQ 利用率 ~100% vs VQ 0.2%；LFQ = FSQ 的 L=2 特例 |
| **register + FSQ + 有序 prefix** | **有正当出处** | OAT: Ordered Action Tokenization（arXiv:2602.04215），"transformer with registers, finite scalar quantization, and ordering-inducing training mechanisms" |

**注意最后两行**：**我们的量化器和有序 register 都没问题，问题全在「时间轴」和「目标函数」上。** 不需要推翻 FSQ 路线。

TAAE 还有一条消融替 FSQ 洗清了嫌疑：**25 Hz 下量化版 SI-SDR 4.73 vs 完全不量化的连续版 5.08**——量化的代价几乎免费，**真正的约束是帧率。别再怀疑 FSQ 了，去改时间结构。**

还有一条我们自己代码里的问题没进上表：**`evaluate()` 用 `logits.argmax(-1)` 解码。** 对每帧 code 边缘分布取 argmax 就是一条几乎不变的 code 序列（Holtzman 2019, [1904.09751](https://arxiv.org/abs/1904.09751)）。**这可能贡献了一部分能量损失，而且是最便宜的一个检查。** 见 8.6 第 0 步。

**最贴身的一句原话，来自运动领域自己**（Ruiz et al. 2018, [1812.05478](https://arxiv.org/abs/1812.05478)）：

> **"The use of this metric [L2] to train a deep network favors motion predictions that converge to a static mean pose."**

**用比特来说最直观**（Coupé et al., Science Advances 2019）：人类语言的信息率是 **39 bits/s**；能用的最低语音 codec 在 **304–800 bps**，是它的 8–20 倍，多出来的买的是「谁在说、什么语气」。

**我们在 56 bit/s——大致只有「词」的预算，没有「语气」的预算。**

**「平均姿势 + 动作类别」不是 bug，它就是 56 bit/s 下的率失真最优解。动作类别 = 词，步态相位 = 韵律。**

### 8.3 最可能修好我们的五个机制（A–D 按性价比排序，E 是治本的那个）

---

#### 机制 A：把「静态内容」和「帧间差分」显式分道 —— 最高性价比

**证据线**：SweetTok 的消融是整份文献里最直接可比的数字——普通联合 query tokenizer rFVD **35.15**，改成解耦（外观 query / 帧间残差 query）**24.80**，再加运动码本 **20.46**（[2412.10443](https://arxiv.org/abs/2412.10443)）。DeRA 同思路比 LARP 好 25%（[2512.04483](https://arxiv.org/abs/2512.04483)）。音频侧的同构证据是 SpeechTokenizer 的 PNMI 0.71 vs 0.28（[2308.16692](https://arxiv.org/abs/2308.16692)）。

**改到我们的栈里**：改 `temporal_common.py` 的 `_read()` / `_features()`。

- 把 9 个 register 分成两组，比如 **2 个「姿势 register」+ 6 个「运动 register」+ 1 个 duration seat**。
- 姿势 register：读窗口的**时间平均**（或第一帧），可以继续全局池化——它本来就该是全局的。
- 运动 register：**只读去掉均值之后的残差**（或帧间一阶差分）。
- 解码：姿势分支先复原基准姿势并广播到所有帧，运动分支把残差加回去。
- **重建损失只在残差分支上算「运动项」**——这样静态解在运动项上拿 0 分，它再也偷不到懒。

**为什么这个排第一**：它同时打掉了两个病根（静态抢占前排 + 逐帧 CE 偏爱均值），而且不需要引入判别器，改动局限在 `_read` / `decode_frames` 和 loss，不动数据管线。

---

#### 机制 B：给 register 时间地址 —— 改动最小、当天可跑

**证据线**：PatchTST（[2211.14730](https://arxiv.org/abs/2211.14730)）"local semantic information is essential"；stable-codec 只用 sliding-window 局部注意力（[2411.19842](https://arxiv.org/abs/2411.19842)）；ProMAG 的 4×/8×/16× 表（[2501.05442](https://arxiv.org/abs/2501.05442)）；LaBraM 用 1 秒的局部 patch 而非全局池化（[2405.18765](https://arxiv.org/abs/2405.18765)）。

**改到我们的栈里**：`--mode chunk` 已经写好了，`_chunk_mask()` 的语义也是对的（register r 只看 [rT/8, (r+1)T/8)）。**现在它是 control，文献的证据方向是它该是主方案。**

更好的版本是 SNAC 式混合（[github.com/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)）：
- 前 2 个 register：全局（低频 / 整体轨迹）
- 后 6 个 register：各管 1/6 时间段（高频 / 局部动态）

这同时对应 TimeVQVAE 的 LF/HF 双支路（[2303.04743](https://arxiv.org/abs/2303.04743)）——低频可预测、可压缩，高频（尖峰）单独一条道。**开合跳只有 84%（DCT 探针）/ 22%（当前模型）正是缺高频通道的症状。**

**代价**：几乎为零，改一个默认值 + 一次对照实验。**应该第一个跑。**

---

#### 机制 C：换掉纯逐帧 CE —— 加序列级信号

**证据线**：Mathieu 2015 的 v_avg 证明（[1511.05440](https://arxiv.org/abs/1511.05440)）；DAC 的 SI-SDR 9.12→1.07（[2306.06546](https://arxiv.org/abs/2306.06546)）；Phenaki 的 I3D 视频感知损失权重 1.0，FVD 65.78 vs 逐帧基线 166.6（[2210.02399](https://arxiv.org/abs/2210.02399)）；DVD-GAN 的 D_T 是 "the learning signal to generate movement"（[1907.06571](https://arxiv.org/abs/1907.06571)）。

**改到我们的栈里，按从便宜到贵排**：

1. **先把已有的 flag 打开**。`train_temporal.py` 里 `--w_motion`（对帧间发生变化的 code 位置加权）和 `--w_vel`（去量化 z 的一阶差分相对 MSE，静态解得分 = 1.0）**默认都是 0**。这是零成本对照，今天就能跑。`--w_vel` 的定义尤其好——它天然把「静态解」锚在 1.0，可读性强。
2. **加多尺度时间频域损失**。把预测和 GT 的去量化 z 序列做不同窗长的 DCT，在系数上算 L1。这是 EnCodec 多尺度 mel 损失（窗长 2^5..2^11，[2210.13438](https://arxiv.org/abs/2210.13438)）和 DAC 的 7 窗长 mel 损失搬到运动上。**运动是低频的，所以低频 bin 要加权。**
3. **加 Gradient Difference Loss**（Mathieu 2015 的第二个解法）：惩罚「差分的差分」，比 `--w_vel` 更狠一层。
4. **加一个小的时间判别器**。参考 DVD-GAN 的 D_T：**丢掉细节维度、保留全部时间**，只判断「这段序列的动态像不像真的」。配 EnCodec 的 **loss balancer**（按梯度滑动平均归一化，权重变成「占总梯度的百分之几」）来定权重，不然 CE 和判别器的量纲对不上。

**注意**：这一条要连指标一起换。**`budget_level_acc` 必须让位给「运动能量保留率」当主指标**——LARP 的 rFVD 31→23 但 gFVD 107→190 就是「盯错指标」的教科书案例。

---

#### 机制 D：提高预算，且优先加「宽度」和「时间步」，不是只加个数

**证据线**：ProMAG 的时间压缩表（4× 安全 / 8× 已痛 / 16× 崩）；VidTok 的 **"Allocating information to channel dimensions yields better performance than temporal dimensions at equivalent total information capacity"**（[2412.13061](https://arxiv.org/abs/2412.13061)）；MAGVIT-v2 「词表越大单调越好」（[2310.05737](https://arxiv.org/abs/2310.05737)）；第二节的率失真下界估算 128–192 bit/s vs 现在 56 bit/s。

**改到我们的栈里**：
- **时间步**：60 帧压到 8 步是 7.5×，已经在痛区。目标 **12–15 个时间位置 / 2 秒**（≈4–5× 压缩），对齐 T2M-GPT 的 l=4（60 帧 → 15 个局部 token，[2301.06052](https://arxiv.org/abs/2301.06052)）。
- **宽度**：`--fsq_levels` 从 `5,5,5,5,5,5` 加到 8 维或提高 level，比如对齐 Cosmos 的 `8,8,8,5,5,5`（64,000 词表）。VidTok 说同样总容量下加宽比加时间步更划算——**两个都要加，但如果只能加一个，先加宽。**
- **ProMAG 的意外招**：**一个 4× 模型重建 6fps 抽帧视频，胜过一个 16× 模型重建原始 24fps 视频。** 直接可用——**把 60 帧抽成 20–30 帧再温和压缩**，可能不花任何额外比特就赢过现在的 8-register。这是最便宜的一个实验。
- **数据侧**：VidTok 发现 3 FPS 训练明显优于 8 FPS，因为帧间位移更大。我们的 `--aug_time` 现在是为变长鲁棒性做的；**再加一个「抽帧放大 delta」的增强**，让「复制上一帧」这条捷径变贵。

---

---

#### 机制 E：相位参数化瓶颈 —— 唯一能让「8 个 token」在原理上成立的设计

**证据线**：DeepPhase（SIGGRAPH 2022, [dl.acm.org/doi/10.1145/3528223.3530178](https://dl.acm.org/doi/10.1145/3528223.3530178)）用可微 FFT 层把每个 latent 通道强制成 **(振幅 A, 频率 F, 偏置 B, 相移 S)**；WalkTheDog（2024, [2407.18946](https://arxiv.org/abs/2407.18946)）是它的**向量量化版**，损失只有 `L_rec + λ_vq·L_vq`——**相位结构完全来自瓶颈的形状，不来自任何损失项。**

**为什么这条可能是最重要的**：算一下我们的预算够不够。

- 8 个内在维度、有效带宽到 ~5 Hz → 每秒约 8 × 2 × 5 = 80 个自由度 → **2 秒窗口 160 个数**。
- 8 个 token 要装 160 个数 → 每 token 约 20 个数 → 每个数就算只给 4 bit，也要每 token 80 bit，即 FSQ 词表 2^80。**对一般运动来说这是不可能的。**
- **但如果换成相位参数化**：2 秒的步态不是 160 个独立的数，而是 `(A, F, B, S)` × 8 通道 = **32 个数**。**8 个 token 装得下。**

**所以判决是**：**8 token / 2 秒对「近周期的运动」在相位式编码下是可行的，对「瞬态/非周期运动」则根本不够。**

**我们观测到的 22–73% 区间，很可能就在追踪这个分界：高端（73%，走路）是周期步态，低端（22%，开合跳的起跳落地）是瞬态。**

**改到我们的栈里**：在 `_features()` 出 FSQ 之前，加一个 FFT 参数化层——把 readout 出来的 register 特征约束成每通道的 (A, F, B, S)，再送进 FSQ。这跟 SNAC 的粗/细分层、TimeVQVAE 的 LF/HF 双支路是同一个家族，只是把「低频」显式写成正弦参数。

**这条同时解释了为什么膝盖 xcorr lag0 ≈ 0**：相位现在必须被编码进 token 的**数值**里，而逐帧 CE 完全没有理由去学它。把相位写进瓶颈的**结构**里，就不用指望损失去教了。

---

### 8.4 关于 DCT 探针的定位（诚实版）

DCT 探针**不能**用来论证「111 bit 够用」——它保留的实数系数比 111 bit 多好几个数量级（见第二节）。

它证明的是**时间基**的结论：**运动在时间上是低频的，2 秒内 8 阶时间基就能表达 95–100% 的走路能量。** 这条有强文献支撑：
- Mao et al. ICCV 2019（[1908.05436](https://arxiv.org/abs/1908.05436)）：20–35 帧用 15–35 个 DCT 系数，**Table 6 显示系数更多有时误差更大**。
- FAST（[2501.09747](https://arxiv.org/abs/2501.09747)）：DCT + 取整 + BPE，1 秒 chunk 约 30 token/臂。
- WaveToken（[2412.05244](https://arxiv.org/abs/2412.05244)）：小波域 1024 词表打败时域 Chronos 的 4096 词表。

**所以正确的结论是「低频时间基 + 提高比特预算 + 沿时间分配」三件事一起做，不是「有了 DCT 就不用管预算」。**

还有一条 FAST 的警告要听：FAST **故意用固定量化步长（γ=10）而不是固定系数个数**，因为保留带宽应该随内容自适应——**跳跃需要比走路更多的高频系数**。我们探针的「走路 95–100% / 开合跳 84%」正是固定 K=8 吃亏的证据。**建议对比「固定 K」vs「固定量化步长」。**

### 8.5 一个可以先排除的嫌疑

VAE 领域的经典病是「解码器绕过 latent」（Bowman 2016 / Chen 2017，见 7.3）。

**我们不用查了，代码已经排除了**：`decode_frames()` 明确写着 "No temporal mixing: every frame query sees only the registers, never other frames"。解码器看不到历史帧，也看不到历史 code。

**所以 token 确实拿到了它该有的位置——是 token 本身没装下动态。** 病因锁死在「全局池化的瓶颈形状」+「逐帧 CE 的目标」+「预算太低」这三条上，不用去做 word-dropout 那类干预。

### 8.6 建议的实验顺序（从便宜到贵）

**第 0 步（10 分钟，今天就做）：把 argmax 换成采样。**

`train_temporal.py` 的 `evaluate()` 用的是 `logits.float().argmax(-1)`。Holtzman 2019（[1904.09751](https://arxiv.org/abs/1904.09751)）说最大化似然的解码会产出 "bland and strangely repetitive" 的结果——**对每帧 code 边缘分布取 argmax，就是一条几乎不变的 code 序列，也就是一个静态姿势。**

改成按 categorical 采样（或 top-k / nucleus），重新量运动能量保留率：
- **能量明显回来** → 22–73% 里有一部分是**解码 artifact**，不是损失的锅。
- **能量不回来** → token 里确实没装动态，往下走。

**注意：采样能补回能量，补不回相位锁定。** 这个实验的价值就是**干净地把两种失败分开**，让后面的改动不做无用功。

**然后按顺序：**

1. **打开 `--w_motion` 和 `--w_vel`**（已存在的 flag，默认 0）。零改动。同时把 `eval_temporal.py` 的主指标从 `budget_level_acc` 换成**运动能量保留率 + 膝盖 xcorr**。
   证据：T2M-GPT 速度项 FID 0.112→0.070（−37%）；**但天花板有限，速度 L1 的最优解仍是速度均值，修不好相位。**
2. **`--mode chunk` 转正**（机制 B）。改一个默认值，跑 chunk vs 全局的对照。
3. **抽帧实验**（机制 D）：60 帧抽成 20–30 帧，用温和的压缩率。ProMAG 的对照说「4× 模型压 6fps 视频」胜过「16× 模型压 24fps 视频」——**这可能不花任何额外比特就赢。**
4. **静态/动态分道**（机制 A）：2 个姿势 register + 6 个运动 register，运动分支只看去均值残差，运动项损失只在残差上算。SweetTok 量到的收益是 rFVD 35.15→24.80。
5. **相位参数化瓶颈**（机制 E）：FSQ 之前加 FFT 参数化层，每通道 (A,F,B,S)。**这是唯一能让 8 个 token 在原理上成立的设计，也是唯一直接治相位的。**
6. **加宽 + 加时间步**（机制 D）：FSQ 到 `8,8,8,5,5,5`（对齐 Cosmos），时间位置到 12–15（对齐 T2M-GPT 的 l=4）。VidTok 说同容量下加宽比加时间步划算，但两个都欠。
7. **多尺度时间频域损失 + 周期折叠判别器**（机制 C）：窗长 [4,8,16,32,60] 帧；判别器按 stride [2,3,5,7,11] 折叠。配 EnCodec 的 loss balancer 定权重，判别器一进来就同时加 feature matching。
   证据：Parallel WaveGAN 单分辨率 MOS 1.36 vs 多分辨率 4.06；HiFi-GAN 去掉 MPD −1.82 MOS。

**第 0–3 步都是一天内能跑完的对照，每一步文献都预测会有可见提升。第 5 步是治本的那个。**

### 8.7 三个要说清楚的诚实边界

1. **DCT 探针不是等比特对照。** 它证明「时间基低阶就够」，不证明「111 bit 够用」。见 8.4。
2. **多分辨率 STFT 损失在音频以外没有任何已发表应用。** 动机很足（它直接给每个频带的运动能量定价，正是我们在掉的指标），但在运动上是未经验证的。
3. **人体运动没有已发表的「bit/s 信息率」数字。** 8.2 里拿语音的 39 bits/s（Coupé 2019）做的类比是我们自己建的论证，不是文献结论。对外讲的时候要标明。

另外，MDM 报告过一个反直觉现象值得记住：**去掉脚接触损失 FID 反而略好（12.81 vs 13.08），但动画出现抖动和不自然的姿态。指标可以往反方向动，而观感在变好。** 这正是「先看动画再看数字」的文献版佐证。

