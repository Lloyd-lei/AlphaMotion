# 独立复核：pose_translator_design 的严谨性审计

只读探查，全程未触碰 GPU 上正在跑的 `PROD_c1_91k_dim20_tail_0810`（PID 3364858）。

你说"很多地方我感觉做的不是很严谨"——**我核实了，主要问题有三个，其中两个是数据bug而不是方法问题**。同时也有一处你自己已经抓得很好、比我之前给的文档做得好。

---

## 一、最严重：`unitree_g1_23dof` 是 `unitree_g1_29dof` 的副本

逐字段比对两个 spec 文件（`cache_xwbc_lafan/`）：

| 字段 | 是否相同 |
|---|---|
| `name` | 不同（只有这一项） |
| `parents` | **相同** |
| `rest_offsets` | **相同** |
| `norm_bonelen` | **相同** |
| `joint_names` | **相同** |
| `height` | **相同**（471.62） |
| `_dof.npz` 全部数组 | **相同形状 (36,19)** |

**除了名字字符串，两个"不同的机器人"是同一个身体。** 而且两个都是 J=36、19维dof——23dof那个本该是23维。

**影响**：`--holdout unitree_g1_29dof,unitree_g1_23dof,pal_talos` 里，三个留出具身实际是**两个**，而其中一个被算了两遍。所有"两个G1零样本"的结论都是一个身体打了双份权重。

这个bug **`E27_verdict_v2.txt` 自己已经发现了**（"one body scored twice"），但**缓存没有重建**——正在跑的训练还在用有bug的 `cache_xwbc_lafan`。

## 二、`pal_talos` 的关节名是 `body00..body32`

```
pal_talos          J=33   names: body00,body01,body02,body03 ...
```

**语义编码通道对这个具身完全失效**——它是整套架构里唯一依赖关节名的机制，而Talos喂进去的是无意义占位符。

这解释了我上一轮看到的数字：

| 留出具身 | full_rot |
|---|---|
| unitree_g1_29dof | 16.9° |
| unitree_g1_23dof | 16.3°（同一个身体） |
| **pal_talos** | **47.3°** |

**Talos的47.3°不是模型能力的度量，是它的语义通道被切断的后果。** 它一个把留出均值从16.6拉到26.8。

`E27_verdict_v2` 同样已经指出（"Its 40-57 deg errors are not a model result"），也同样**没有修**。`cache_xwbc_lafan_fixed` 存在，但我核对了——**Talos的名字在 `_fixed` 里依然是 `body00..`**，那个缓存修的是别的东西。

## 三、几乎所有结论都是单种子

115个run目录，**只有4个带种子标记**（`E23_K32_s1/s2`、`E23_K128_s1/s2`）。

E23测出的种子噪声：K32 **0.38°**、K128 **0.25°**（`MODEL_VERSIONS.md` 里写的 ±0.9° 更保守，取自判决书）。

拿这个噪声带去量各个决策的差距：

| 决策 | 差距 | 判定 |
|---|---|---|
| K容量拐点（K16 vs K128） | 1.67° | **> 噪声，可信** |
| 语义编码器（SigLIP vs Qwen） | 0.25° | **≈ 噪声，不可判** |

**编码器那个决策落在噪声带里**，而 `E27_verdict.txt` 自己算得比我更狠：

> E19 vs E27_sig —— **同一个编码器** —— 相差 2.2°，即"配方/run间方差比整个编码器效应还大"

这是我在整份材料里看到的最好的一处推理：**用同编码器的两个run之间的差异，作为判定其他差异是否显著的尺子。** 结论是编码器根本不值得换。

### 但这里有个矛盾需要你处理

| 来源 | 结论 |
|---|---|
| `E27_verdict.txt`（早） | **不值得换，SigLIP更好**（限位违反率：Qwen在G1上1.7倍、X2上14倍） |
| `E27_verdict_v2.txt`（晚） | **推荐Qwen**（"ahead on every metric that reaches the robot, 一旦把pal_talos因故排除"） |
| `RUNS_INDEX.md` | "Qwen 后投影质量赢 **18/18**" |
| `ENCODER_BAKEOFF.md` | 轴向分离**BGE-M3赢**，Qwen第二；跨形态检索SigLIP赢 |
| **正在跑的训练** | 用的是 **Qwen** |

**v2的"18/18"建立在"把pal_talos排除"之上——而Talos之所以差,是因为它的关节名是`body00..`这个bug。** 换句话说：**这个排除本身是被数据bug驱动的**，不是"for cause"的独立理由。

v2自己也承认"The margin is inside the run-to-run band per-metric"，并建议"3 seeds × 2 encoders"。**这个建议是对的，而且现在还没做。**

---

## 四、正在跑的训练：留出集在退化

`PROD_c1_91k_dim20_tail_0810`（Qwen + vis_residual + 20维FSQ + tail_alpha=1.0，目标91k步）：

全部8个eval点（不截取，避免误读趋势）：

| step | 训练内 | 留出 | 差距 |
|---|---|---|---|
| 2000 | 16.61° | 32.57° | 15.96 |
| 4000 | 10.70° | 27.13° | 16.43 |
| 6000 | 10.08° | 27.33° | 17.25 |
| 8000 | 9.87° | 26.57° | **16.70**（回落） |
| 10000 | 8.37° | 26.20° | 17.83 |
| 12000 | 7.77° | **26.03°**（谷底） | 18.26 |
| 14000 | 7.45° | 26.60° | 19.15 |
| 16000 | 7.24° | **26.83°** | **19.59** |

**训练误差单调降；留出误差在12k触底后回升；差距总体上升（15.96→19.59），但不是单调——在step 8000 有一次回落（17.25→16.70）。** 后四个点（10k起）是连续上升。目标91k步——按此趋势后面75k步大概率在过拟合，但**8个点、单种子，这个判断本身还不够硬**（同样受第三节单种子问题的限制）。

对照 Greenwich（`PROD_spatial_0808`，24k步）：训练内 **4.34°**、留出 **25.13°**。**当前run两项都更差**——20维FSQ + tail_alpha 到16k步还没显出优势。

**建议**：按留出误差存 best checkpoint（而不是只存最后一步）。这个不需要重启训练，改eval回调即可，下次生效。

---

## 五、你做得比我好的一处

`grand_eval/docs/CLAIM_MATRIX.md`：17条claim逐条标 supported / partially / not established，每条给出"可辩护的措辞"和"边界"，还写了发表规则：

> full-window reconstruction must not be called prediction，kinematic validity must not be called physics-on execution

而且**大部分条目是诚实的负面结论**："zero-shot action — not established"、"sparse temporal completion — not established"（学习式8帧补全比插值差4.88-5.22cm）、"bidirectional completion — not established"。

这比我之前给你写的那批文档在"区分已验证/未验证"上做得更系统。

`MODEL_VERSIONS.md` 的"已知缺陷"那一节也很好，特别是：
- `content_share` 分母塌陷时会飞出[0,1]（出过1.5/3.2）→ 该指标作废
- `.norm(-1)` ≠ `.norm(dim=-1)`（p=-1范数恒≈0，**会伪装成完美模型**）
- 速度不变性验收线0.95，**没有任何版本过线**（V8最高0.922）

第二条和我们之前反复栽的是同一族问题：**某个量不是它声称的那个东西**。

---

## 六、优先级建议

| # | 事项 | 理由 |
|---|---|---|
| 1 | **重建缓存，修Talos关节名 + 拆开两个G1** | 这两个bug污染了所有留出集结论，包括正在跑的训练 |
| 2 | **给当前run加held-out best-checkpoint** | 留出集已在退化，91k步会白跑 |
| 3 | **3 seeds × 2 encoders 重测E27** | 编码器决策目前落在噪声带内，而生产配方已经采用了Qwen |
| 4 | 留出集分开报（G1类 vs 比例差异大的） | 合成一个26.8°既低估G1也掩盖Talos |
| 5 | 磁盘：`/` 只剩5.3G（100%） | 91k步的checkpoint可能写失败 |

**第1项是根**：修完之后第3项才有意义（现在Qwen"赢"的前提是排除Talos，而Talos的问题是数据bug），第4项也会自动变成两个真实具身而不是三个。