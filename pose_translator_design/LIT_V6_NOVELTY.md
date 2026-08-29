# V6 时序 tokenizer 到底新不新

给 lloyd 的新颖性核查。目标不是证明我们新，是把「谁已经做过什么」摆清楚，包括那些**差一点就撞上**的工作。

调研时间：2026-08-06/07。用 WebSearch/WebFetch 扫的公开文献。

---

## 一句话结论

**(b)+(c)+(d) 三件事合在一起、且发生在时间轴上——目前是空的。**

但要说清楚：这三件事**两两组合都已经有人做了**，只是没人凑齐三件，也没人把它放到时间轴上。所以这不是「开辟新大陆」，是「一个没人填的格子」。

最硬的证据来自 **l-DeTok（NeurIPS 2025）**：同一篇论文里，遮输入 + VQ 有了（2D 网格版），1D latent-query + VQ 也有了，**但作者在附录明说 1D 版「为求简单完全省略了遮罩」**。也就是说，这个格子是被公开点名跳过的。好消息是空缺确凿、有出处可引；坏消息是这属于「显而易见的下一步」，随时可能被人补上。下面逐格核对。

---

## 先把要查的四条摆出来

我们 V6 的结构（代码确认过：`exp_topo/train_v6.py` + `exp_topo/frame_masking.py`）：

| | 属性 | 我们的做法 |
|---|---|---|
| (a) | token 就是**帧**，没有时间卷积下采样 | 每帧一个 token，纯 transformer，无 stride |
| (b) | **在 tokenizer 自己的 encoder 里**随机遮帧 | 遮住的帧，pose 换成 learned mask token，**时间坐标始终可见**；遮罩是个**分布**（none 30% / rand 40% / sparse 15% / prefix 15%） |
| (c) | 少量**可学 latent query**（Perceiver/register 式，8 个）把帧 token 池化成定长瓶颈 | 8 个 path register + 1 个 time register，1D tokenization，不是逐帧 token |
| (d) | 瓶颈用 **FSQ** 量化，decoder 重建**所有**帧（含被遮的） | FSQ 6 维 × 5 级，loss 加在全部帧上，不管遮没遮 |

关键区分（后面反复要用）：
- **「在 tokenizer 里遮」** = 遮 tokenizer 的**原始输入**，量化码必须存下足够信息把缺的补回来。
- **「在生成器里遮」** = 先用一个训好的 tokenizer 把数据变成 token，再遮 token 让下游 transformer 猜。MaskGIT / MUSE / MoMask / MMM / MAGVIT / BAMM 全是这一类，**这个非常成熟，不是我们说的东西**。

---

## 主表：最接近的工作 × 四条属性

✅ = 有；⚠️ = 形似但机制不同；❌ = 没有

| 工作 | 年份/出处 | 领域 | (a) 帧级无下采样 | (b) tokenizer 内部遮输入 | (c) latent query 定长瓶颈 | (d) FSQ + 重建全部 |
|---|---|---|---|---|---|---|
| **l-DeTok（VQ 版）** ([2507.15856](https://arxiv.org/abs/2507.15856)) | NeurIPS 2025 | 图像 | — (空间) | ✅ 遮 70–95% patch，encoder 只看可见的 | ❌ **2D 网格瓶颈** | ⚠️ **VQ**（4096×64）；decoder 重建整图 |
| **l-DeTok（1D 版）** 同上 | NeurIPS 2025 | 图像 | — (空间) | ❌ **1D 版本作者明确关掉了遮罩** | ✅ K∈{32,64,128} 可学 latent token | ⚠️ VQ |
| **MAETok** ([2502.03444](https://arxiv.org/abs/2502.03444)) | ICML 2025 | 图像 | — (空间) | ✅ 遮 40–60% patch，decoder 补 | ✅ 128 个 latent token（TiTok 式） | ❌ **纯连续 AE，无量化**（全文无 VQ/FSQ/LFQ 变体） |
| **Masked VQ-VAE**（语义通信）([2206.04011](https://arxiv.org/abs/2206.04011)) | IEEE TWC 2023 | 图像 | — (空间) | ✅ 遮 patch，重建缺失 patch，**码本与编解码器联合训练** | ❌ 逐 patch token | ⚠️ VQ；遮罩偏向噪声区，非纯随机 |
| **MacTok** ([2603.29634](https://arxiv.org/abs/2603.29634)) | 2026 | 图像 | — (空间) | ✅ mask 比例 ~U(0, 0.7)，是个**分布** | ✅ 可学 latent token | ❌ **连续 + KL，无量化** |
| **LATTE** ([2605.11192](https://arxiv.org/abs/2605.11192)) | 2026 | 语音 | ⚠️ 建在冻结 codec 特征上 | ❌ 没有任何输入遮罩 | ✅ 250 个 slot，定长，**非时间对齐** | ⚠️ **BSQ** 不是 FSQ；decoder 重建全序列 |
| **LG-Tok** ([2602.08337](https://arxiv.org/abs/2602.08337)) | 2026 | **人体动作** | ✅ 纯 transformer 无卷积 | ❌ mask token 只在 **de**tokenizer 当输出 query | ✅ 25/36/49 个 latent query | ❌ 多尺度 **RVQ**，不是 FSQ |
| **MagiCodec** ([2506.00385](https://arxiv.org/abs/2506.00385)) | 2026 | 音频 | ❌ 线性下采样 160–640× | ⚠️ **逐帧 Bernoulli 遮**，但遮住的帧换成**高斯噪声**不是 mask token（定位成正则化，不是 MAE） | ❌ 逐帧 token | ❌ 单层 VQ（码本 131072） |
| **TFM-Tokenizer** ([2502.16060](https://arxiv.org/abs/2502.16060)) | ICLR 2026 | 脑电 | ⚠️ 逐窗 | ✅ 时间+频带遮罩，重建谱图，**确实在 tokenizer 训练阶段** | ❌ 逐窗一个 token，长度随输入变 | ❌ VQ（码本 8192） |
| **ElasticTok** ([2410.08368](https://arxiv.org/abs/2410.08368)) | ICLR 2025 | 视频 | ❌ patch | ⚠️ 遮的是**编码后的 latent 尾巴**，不是输入帧 | ❌ 无 latent query | ✅ **FSQ**，decoder 从截断 latent 重建全部帧 |
| **AdapTok** ([2505.17011](https://arxiv.org/abs/2505.17011)) | 2025 | 视频 | ❌ 时空 patch | ⚠️ block tail-drop，同样在 **latent 侧** | ✅ 拼接可学 latent token | ❌ SVQ（随机 VQ） |
| **VFRTok** ([2505.12053](https://arxiv.org/abs/2505.12053)) | 2025 | 视频 | ❌ patch | ⚠️ **encoder/decoder 帧率不对称**：encoder 只看抽稀后的帧，decoder 重建没见过的帧。**这实质上就是我们的 `sparse` 家族**，但论文不叫它 masking，也不是随机子集 | ✅ 512 或 128 个 latent token，与帧数解耦 | ❌ 论文未用量化器（连续） |
| **MotionBERT** ([2210.06551](https://arxiv.org/abs/2210.06551)) | ICCV 2023 | **人体动作** | ✅ 无下采样 | ✅ 10% 帧级 + 5% 关节级遮罩 | ❌ 无瓶颈 | ❌ **全程连续 512 维，完全没有离散码本** |
| MAMP / SkeletonMAE / MotionMAE | 2022–23 | 骨架 | ✅ | ✅ | ❌ | ❌ 无量化器 |
| Causal Motion Tokenizer ([ICCVW 2025](https://openreview.net/forum?id=WavXPunwzM)) | 2025 | 动作 | ❌ 因果卷积 | ⚠️ 遮的是**量化之后**的 token（喂 decoder 前随机换码），为了鲁棒性 | ❌ | ❌ RVQ |
| ScaMo / Shape My Moves / Go to Zero / AnyMo | 2025–26 | 动作 | ❌ 多为 ×4 下采样 | ❌ | ❌ | ✅ **FSQ / residual FSQ**，但完全不遮 |
| MoMask / MMM / T2M-GPT / MoGenTS / MoSa | 2023–25 | 动作 | ❌ ×4 卷积 | ❌（遮在**生成器**里） | ❌ | ❌ VQ/RVQ |
| MaskGIT / MUSE / MAGVIT / MAGE | 2022–23 | 图像/视频 | — | ❌（遮在**生成器**里，tokenizer 是冻结的） | — | — |

---

## 逐题回答

### Q1：有人把 masking 放进 tokenizer 内部吗？

**有，而且不止一个——但没有一个是在时间轴上 + 带量化 + 带 latent query 的。**

真正做到「遮 tokenizer 自己的输入」的：

- **l-DeTok（NeurIPS 2025）**——图像。**这是最关键的一篇。**它遮输入 patch（遮罩率 max(0, U(−0.1, M))，最佳 M=70–90%），encoder 只处理可见 patch，decoder 拿可见 latent + [MASK] token 重建整图。而且 §5.5 / 附录 A.7 明确做了 **VQ 版**（码本 4096×64），原话大意是「VQ 版刻意做到最简，只用遮罩当解构手段」，扫了 0.7 和 0.95 两个遮罩率，**0.95 最好**。
  **但最要命的一句在附录**：他们也做了 1D latent-query 变体（K∈{32,64,128} 个可学 token，只把这些喂 decoder），可是——「把 l-DeTok 用在 1D tokenizer 上时……**为求简单我们完全省略了遮罩**」。
  **换句话说：我们要填的那个格子，是他们自己点名跳过的格子。**这对我们是好消息（确认空着，而且有权威出处可引），也是坏消息（这是个显而易见的下一步，别人随时会做）。
- **Masked VQ-VAE（IEEE TWC 2023，语义通信）**——图像。遮 patch、重建缺失 patch，**码本和编解码器联合训练**。因为发在通信期刊，视觉 tokenizer 圈子基本没人引它。注意它的动机是抗信道噪声 + 省带宽，不是表示质量；遮罩也偏向「语义噪声出现的地方」，不是纯随机。
- **MAETok（ICML 2025）**——图像。遮 40–60% 的 patch，128 个 TiTok 式 latent token，decoder 补回来。**但它明确是纯 AE，没有量化器**（论文原话大意：连变分约束都不需要，重要的是 latent space 的结构）。核查过 ablation 和附录：**全文没有测试任何 VQ/FSQ/LFQ 变体**。它是为 diffusion 服务的连续 tokenizer。
- **MacTok（2026）**——图像。mask 比例从 U(0, 0.7) 采样，**是个分布**，这点和我们一样。同样是连续 + KL。它引用 MAETok 作为先例。
- **TFM-Tokenizer（ICLR 2026）**——脑电。在 tokenizer 训练阶段做时间+频带遮罩并重建谱图，配 VQ 码本。**但是逐窗一个 token，没有 latent query，瓶颈长度随输入变**。
- **MagiCodec（2026）**——音频。逐帧 Bernoulli 遮，被遮的帧整个换成高斯噪声。**注意它的定位不是 MAE，是正则化**：目的是逼模型用长程上下文、别只看局部。逐帧 VQ token。

所以 Q1 的答案是：**「tokenizer 内部遮输入」这件事本身不新**，2023–2026 已经在图像（l-DeTok、MAETok、MacTok、Masked VQ-VAE）、音频（MagiCodec）、脑电（TFM-Tokenizer）各自出现。我们**不能**把这个当成卖点。

**明确排除掉的（都是「在生成器里遮」，正是我们要划清的那条界）**：

- **MAGE**（CVPR 2023）— 用**预训练冻结的 VQGAN** 先量化，再遮 token（比例 0.5→1）。论文自己就在跟 MAE 对比这一点：MAGE 遮的是语义 token，MAE 遮的是输入 patch。
- **BEiT v2** — VQ-KD tokenizer 看的是**完整未遮**的图去蒸馏 CLIP/DINO teacher，遮罩只在下游 ViT。**PeCo**、**CAE**、**MaskFeat** 同理。
- **MaskBit** — tokenizer 正常训，遮罩在下游 bit-token 生成器。
- **MaskGIT / MUSE / MoMask / MMM / MAGVIT / Paella / MaskGCT / InterMask** — 全是第二阶段对已离散 token 遮。
- **SoundStream / EnCodec / DAC** — 不遮输入。它们的 "quantizer dropout" 丢的是 **RVQ 层数**（瓶颈侧，为了码率可伸缩），是 tail-drop 的音频版，不是输入遮罩。**没找到任何一个神经音频 codec 是靠遮自己的波形/谱输入来训练的**（MagiCodec 是换成噪声，最接近）。
- **wav2vec 2.0** — 遮的是特征编码器**输出**（喂给 context network 前），而且量化的是**未遮**的目标；对比学习，无重建 decoder。量化器和遮罩共存，但量化器的输入从没被遮过。
- **VQ-MAE-S / EnCodecMAE** — MAE 加在**已经训好**的 VQ-VAE / codec 产出的 token 上。不算。

新的是：**没有人在时间轴上、用一个定长的量化 latent-query 瓶颈来做这件事**——而且 l-DeTok 明确说了它在 1D 变体里把遮罩关掉了。

### Q2：有没有「时间版 TiTok」？

**有，而且比预想的多。这是我们最需要小心的一题。**

- **LATTE（2026，语音）** 是最直接的时间版 TiTok：一组固定的可学 latent slot 拼在音频特征序列后面，走双向自注意力，**只保留 query 位置**，量化（BSQ）后 decoder 重建整段特征。论文自己就说是 "TiTok-style bottleneck"、"non-temporally aligned"。**这基本就是我们的 (c)+(d)，只差 (b)。**
- **VFRTok（2025，视频）**：512/128 个 latent token，与帧数解耦；而且 encoder 和 decoder **帧率不对称**——encoder 只看抽稀过的帧，decoder 重建它没见过的帧。这在机制上就是我们 `sparse` 家族的一个特例。但论文既不叫它 masking，也不是随机子集，且没有量化器。
- **AdapTok（2025，视频）**：拼接可学 latent token + block tail-drop + SVQ。tail-drop 在 latent 侧。
- **ElasticTok（ICLR 2025，视频）**：FSQ + 随机丢弃 latent 尾部 token，decoder 从截断 latent 重建全部帧。**没有 latent query**（用的是位置式 latent + 学到的 mask embedding）。
- **LG-Tok（2026，人体动作）**：**我们领域里已经有 latent-query 1D tokenizer 了。**纯 transformer 无卷积，25/36/49 个可学 latent query，多尺度 RVQ。它没有输入遮罩。

结论：**「latent query 池化时间轴」这条不新，(c) 这一格已被占。**在我们自己的动作领域，LG-Tok 已经占了 (a)+(c)。

### Q3：FSQ（不是 VQ）配 masked / latent-query 瓶颈，有谁做过？

- FSQ + latent 侧遮罩：**ElasticTok** 有（FSQ + tail-drop），但没有 latent query，遮的也不是输入。
- FSQ + 动作 tokenizer：**ScaMo（CVPR 2025）、Shape My Moves（CVPR 2025）、Go to Zero、AnyMo（residual FSQ）** 都用了 FSQ，理由都是码本利用率/防塌缩。**没有一个遮输入。**
- FSQ + latent query：**没找到。**LATTE 用的是 BSQ（球面二值量化），同一家族但不是 FSQ；AdapTok 用 SVQ；LG-Tok 用 RVQ。
- **FSQ + 输入遮罩：也没找到。**所有「遮输入」的 tokenizer 要么是 VQ（l-DeTok VQ 版、Masked VQ-VAE、TFM-Tokenizer、MagiCodec），要么干脆连续（MAETok、MacTok、l-DeTok 主版）。**输入遮罩 × FSQ 这个组合本身就没发表过。**
- lloyd 提到的 stable-codec/TAAE 用 FSQ 但是 strided conv + 无遮罩——**核对属实**（见 `LIT_TEMPORAL_FSQ.md` 已有的表）。MAGVIT-v2 的 LFQ 是 L=2 的 FSQ 这个说法也站得住。

所以 **FSQ 配 latent-query 瓶颈** 这一小格本身就没人填，配上遮罩就更空。

### Q4：动作领域，有没有带帧遮罩的**离散** tokenizer？

**没有。这条最干净。**

分成两堆，中间没有交集：

- **会遮帧的**：MotionBERT、MAMP、SkeletonMAE、MotionMAE。它们**全是连续表示，一个码本都没有**。MotionBERT 核实过：随机置零 15% 关节（10% 帧级 + 5% 关节级）+ 加噪，但 DSTformer 全程是 `T×J×512` 的连续特征，没有任何量化器。它是「带损坏输入的回归模型」，不是 tokenizer。
- **有离散码本的**：T2M-GPT、MoMask、MMM、MoGenTS、MoMask++、MoSa、AnyMo、MotionGPT……**遮罩一律在下游生成器**，对着已经量化好的 token 遮。tokenizer 本身永远看干净完整的动作。

唯一需要点名区分的近邻是 **Causal Motion Tokenizer（ICCVW 2025）**：它在 encoder 和 decoder 之间**随机替换量化后的 token**来提升 decoder 鲁棒性。这是「遮量化码」，不是「遮输入帧」——机制不同，但是最接近的一篇，写 related work 时应该主动引用并划清界限，别装作没看见。

（这篇 CVF PDF 抓取被 403 挡了，引用前建议手动读一遍原文核实措辞。）

### Q5：任何模态下，(b)+(c)+(d) 三件齐活的？

**没找到。**最接近的三种残缺组合：

- (b)+(c)，缺 (d)：**MAETok / MacTok** — 图像，连续，无量化。
- (c)+(d)，缺 (b)：**LATTE** — 语音，BSQ，不遮。**l-DeTok 的 1D 版也在这格**（1D latent query + VQ，遮罩被作者主动关掉）。
- (b)+(d)，缺 (c)：**l-DeTok VQ 版 / Masked VQ-VAE**（图像，2D 网格 VQ）、**MagiCodec / TFM-Tokenizer**（逐帧/逐窗 VQ）。

**三件齐活的一个都没有。**而且上述没有一个同时满足「在**时间**轴上」+「遮罩是个**家族分布**」+「FSQ」。

最有说服力的证据是 l-DeTok：同一篇论文里，(b)+(d) 有（VQ 版）、(c)+(d) 有（1D 版），**但作者明确说 1D 版「为求简单省略了遮罩」**——所以 (b)+(c)+(d) 是一个被公开点名跳过的空格，不是我们臆想出来的。

---

## 三个最近的工作（要点名的）

1. **l-DeTok**（arXiv 2507.15856，NeurIPS 2025）— **最近的一篇，必读。**它同时拥有 (b)+(d)（VQ 版）和 (c)+(d)（1D 版），唯独没把两者合起来，而且**白纸黑字说 1D 版省略了遮罩**。这既是我们最强的「此格空缺」证据，也是最大的被抢先风险。
2. **MAETok**（arXiv 2502.03444，ICML 2025）— 「在 latent-query tokenizer 里遮输入」这条轴上最近，(b)+(c) 齐全。差别：图像不是时间，**连续无量化**（全文无任何量化变体）。
3. **LATTE**（arXiv 2605.11192，2026）— 「在**时间**轴上用 latent query 做量化 1D tokenization」这条轴上最近，几乎就是 (c)+(d) 的时间版。差别：不遮，且是 BSQ 不是 FSQ。

**领域内最近的**：**LG-Tok**（arXiv 2602.08337，2026）— 动作领域已经有 latent-query 1D tokenizer 了。差别：不遮输入（它的 mask token 只是 detokenizer 的输出 query），多尺度 RVQ 不是 FSQ。

第二梯队但不能漏：**MacTok**（遮罩率是分布）、**Masked VQ-VAE**（IEEE TWC，输入遮 + VQ 联合训码本）、**VFRTok**（帧率不对称 ≈ sparse 遮罩 + latent query）、**MagiCodec**（逐帧遮 + 时间轴）、**TFM-Tokenizer**（脑电，tokenizer 阶段遮 + VQ）、**ElasticTok**（FSQ + latent 侧遮）、**Causal Motion Tokenizer**（动作领域遮量化码）。

---

## 诚实的贡献声明

**可以说的：**

> 把「输入遮罩」搬进一个**量化的、latent-query 定长瓶颈的、时间轴上的** tokenizer，并且把遮罩做成一个**覆盖产品形态的家族分布**（全见 / 随机 / 只留首尾 / 只留几个关键帧 / 只留过去），据我们所知尚无先例。这样一来，插值、关键帧、流式三种部署形态**共用同一个 tokenizer 和同一份码**，而不是训三个模型。

而且这不是我们自己划的一块地：**l-DeTok 在 1D latent-query 变体里明确写了「为求简单省略遮罩」**——这一格是被人点名跳过的，不是没人想到。我们要做的就是把它填上，并解释为什么在时间轴上它特别值得填。

这个「遮罩家族 = 产品面」的角度可能比结构本身更值钱，因为 `prefix`（只留过去 → 流式）在空间轴上根本没有对应物，是时间轴独有的。

**不能说的（会被审稿人一句话打掉）：**

- ❌ 「masked autoencoding 是新的」— MAE 系已经烂熟。
- ❌ 「在 tokenizer 里遮是新的」— l-DeTok、MAETok、MacTok、Masked VQ-VAE、MagiCodec、TFM-Tokenizer 都做了。
- ❌ 「遮输入 + 离散码本是新的」— **l-DeTok 的 VQ 版和 Masked VQ-VAE 已经做了**（只是 2D 网格瓶颈，不是 latent query）。
- ❌ 「1D latent-query tokenization 是新的」— TiTok 2024，LATTE/VFRTok/AdapTok/LG-Tok 都做了；**动作领域 LG-Tok 已经占了这格**。
- ❌ 「FSQ 用在动作 tokenizer 上是新的」— ScaMo、AnyMo 等一堆。
- ❌ 「mask 比例采样成分布是新的」— MacTok 已经是 U(0, 0.7)。
- ❌ 「我们第一个让 tokenizer 重建没见过的帧」— VFRTok 的帧率不对称训练实质上就在做这件事。

**所以诚实的措辞是「组合新」，不是「机制新」。**每个零件都有出处，值钱的是没人把它们拼在时间轴上、也没人把遮罩家族当成产品面来设计。

---

## 风险 / 待核实

1. **l-DeTok 有一条直接可用的技术警告，别忽略。**他们报告 1D tokenizer 上出现类似 "grokking" 的病：加了去噪/遮罩任务会**推迟相变**，导致 tokenizer 训不熟。他们的解法是**前 10 个 epoch 关掉去噪做 warm-up**。我们正好在做「1D latent query + 遮罩」，这个坑大概率会踩到——如果 V6 早期 loss 卡住不动，先试这个 warm-up，别急着改结构。
2. **l-DeTok 是最大的被抢先风险**。它已经把两个零件都摆在桌上了，补上第三个是顺手的事。如果要发表，动作要快，而且必须主动引用它、明说「我们填的正是你们省略的那格 + 时间轴 + FSQ」。
3. **LG-Tok（2602.08337）必须手动通读**。它是同领域最近的一篇，我的判断（无输入遮罩、RVQ）来自两轮 HTML 摘要，结论一致但码本大小始终没读到。如果它某个 ablation 里其实做了输入遮罩，我们的 (b)+(c) 主张在动作领域就要退让。
4. **Causal Motion Tokenizer（ICCVW 2025）PDF 抓取 403**，「遮量化码」这个描述来自索引文本，引用前手动核实。
5. **LATTE 是 2026 年 5 月的 preprint**，可能还在演化；如果它出个带遮罩的版本，(b)+(c)+(d) 这格就被占了。这是最需要盯的一篇。
6. 2026 年的文献更新很快，`Beyond MoCap`（2606.27547）没抓下来（文件太大），值得手动看一眼它的 tokenizer 结构。

---

## 附：我们自己的谱系

值得注意的是，V6 的时序遮罩是我们**空间层**做法的轴对换——`masking.py` 早就在遮**关节**（rand / ee / root / sparse），V6 的 `frame_masking.py` 是逐条镜像过来的，只多了一个 `prefix`（时间轴独有，因为关节轴没有时间箭头）。

这个「同一个遮罩-latent query-FSQ 配方，先用在关节轴、再用在时间轴」的一致性，本身就是论文里一个好讲的结构论点，而且是我们自己的，不依赖任何外部先例。
