# AlphaMotion Spatial-Temporal 机制与潜空间审计

日期：2026-08-07  
范围：冻结的 Spatial E19、Temporal V7-JEPA、Temporal V7-MaskFSQ  
原则：只解释现有模型，不重新训练，不替两条 Temporal 路线做选型。

## 1. 结论先行

1. **当前真实计算图不是 `Spatial -> Temporal`。** Temporal 直接读取 6 个关键关节的时序状态；Spatial E19 只在 Temporal 输出之后，逐帧把 6 个关键关节补成全身。因此 `frozen_spatial` 在 Temporal 配置里只是记录项，不参与 `train_v7.py` 的 forward 或反向传播。
2. **当前 Temporal 是完整窗口 tokenizer / autoencoder，不是未来预测器。** 训练时目标帧 `k` 随机落在窗口后半段，但 encoder 仍能看到包含 `k` 之后帧的完整窗口。它回答的是“如何压缩和重建一段已知动作”，尚不能证明“只看过去预测未来”。
3. **JEPA 与 MaskFSQ 学出了两种不同的潜空间几何。** JEPA 是分布式全局表征，8 个 path register 没有稳定时间槽位；MaskFSQ 有明显时间对角结构，但局部池化后又经过全局 group Transformer，所以仍有显著跨时间混合。
4. **JEPA 的 time register 不是纯时长 token。** 它由累计弧长曲线、总路程、帧数比例和平均速度共同生成，是一个粗粒度的全局运动摘要。因果打乱它会使位置误差上升约 1.86 cm，但 path bank 的影响更大，打乱后上升约 5.12 cm。
5. **量化不是均匀地损失信息。** JEPA 连续 time feature 可用 89.7% 准确率线性识别 30/45/60/90/120 帧的速率，经过 FSQ 后只剩 23.1%，接近 20% 随机水平。上游学到了速率方向，但当前量化边界几乎把它抹掉了。
6. **跨骨架不变性不是“完全去掉身体信息”。** Temporal body probe 约 23%-25%，接近 20% 随机水平；Spatial 组合码 body probe 为 75.5%。Spatial 同时具有很高的跨身体 CKA 和很强的身体可解码性，说明它更像“共享动作子空间 + 身体签名”，而非纯身体无关 latent。
7. **Spatial 的 rotation stream 比 position stream 稳定。** 只给 6 个关键关节时，rotation code 与全输入 code 的 CKA 为 0.993、余弦为 0.912；position 分别只有 0.862、0.521。当前补全瓶颈主要在位置流，而不是旋转流。
8. **当前系统没有 world translation、contact、torque、IMU 或 WBC 输出。** 默认 `root_vel=False`，所有位置是 root-relative。现有结果只能证明运动表示和运动学重建，不能证明 physically solid 的闭环控制。

## 2. 当前真实计算图

```text
paired motion cache
  [T, full-body joints, global rot6d + root-relative xyz]
        |
        | select(root, head, L/R wrist, L/R ankle)
        v
six-key trajectory [T, 6, 9]
        |
        +-------------------- Temporal V7 --------------------+
        |  full-window encoder -> FSQ latents -> decoder      |
        +-----------------------------------------------------+
        |
        v
predicted six-key trajectory [T, 6, 9]
        |
        | once per frame, mask all non-key joints
        v
frozen Spatial E19 -> target-topology full-body pose [T, J, 9]
        |
        v
kinematic projection / render / downstream bridge
```

对应源码证据：

- Temporal 训练仅构造 `KeyData` 和 6-key window：`train_v7.py:615-620, 678-701`。
- `frozen_spatial` 只写入配置：`train_v7.py:623-629`。
- Spatial 补全发生在 bridge：`v7_bridge_mesh.py:75-86`。
- 补全实现是“关键关节可见，其余关节 mask，再经 E19 编解码”：`v5_common.py:787-814`。

这意味着目前两层是**串联使用、分开训练**，不存在 Temporal loss 穿过 Spatial decoder 的端到端梯度。概念上可以把 Spatial 看成一个姿态流形 prior，把 Temporal 看成关键关节轨迹 codec；但不能把 Temporal latent 称为 Spatial hidden state 的时间演化。

## 3. Spatial E19 到底在做什么

### 3.1 输入与 topology attention

每个关节的动态输入是：

```text
x_j = [global rotation 6D, root-relative position xyz]  # 9D
```

动态 pose embedding 与静态 skeleton feature 相加。静态 feature 包括 rest offset、骨长、拓扑深度、SigLIP 关节名语义和可选 DOF 描述。mask 只替换动态 pose embedding，关节几何和语义仍可见。随后 4 层 graph attention 使用 hop-distance bias 在整棵骨架上交换信息。

源码：`model_dual.py:54-59, 89-112`。

### 3.2 两条空间码流

Spatial 用两组 learned queries 对全身关节特征做 Perceiver 式汇聚：

```text
joint hidden [B, J, 256]
  -> 128 position queries -> FSQ(13 dimensions, 8 levels)
  -> 128 rotation queries -> FSQ(13 dimensions, 8 levels)
```

两条流分别 cross-attend 回目标骨架，再经 graph self-attention 输出目标关节的 root-relative position 和 global rot6d。源码：`model_dual.py:69-80, 89-103, 135-168`。

### 3.3 Spatial latent 的实证结构

审计使用 240 个 AMASS 帧、5 种身体，对 E19 做完全独立于训练的 probe：

| 检查 | Position stream | Rotation stream | 两流合并 |
|---|---:|---:|---:|
| 跨身体平均 off-diagonal CKA | 0.808 | 0.961 | 0.948 |
| 身体分类 balanced accuracy | 66.6% | 88.6% | 75.5% |
| 随机水平 | 20% | 20% | 20% |

这两个结果不矛盾。CKA 高表示不同身体共享主要运动几何；body probe 高表示该空间仍保留可线性读取的身体签名。E19 训练中同时存在跨身体 InfoNCE 和 `judge_mode=grad` 的 embodiment critic，因此“共享动作主空间 + 身体残差”正是训练目标共同作用的合理结果，而不是纯粹的 topology-invariant code。

### 3.4 六关键关节输入后的变化

比较同一帧的 full-pose 编码和 six-key mask 编码：

| 检查 | Position | Rotation | 合并 |
|---|---:|---:|---:|
| CKA | 0.862 | 0.993 | 0.981 |
| 样本余弦中位数 | 0.521 | 0.912 | 0.690 |
| FSQ scalar 完全一致率 | 35.1% | 58.8% | 46.9% |

结论是：**群体层面的流形几何基本保留，但单个样本的 token identity 发生了很大变化，尤其是位置流。** 所以不能把 CKA 高误读成“六点输入与全身输入产生相同 token”。

AMASS 六关键点补全结果：

- 隐藏关节均值：11.13 cm / 24.95 deg。
- 可见六关节均值：14.94 cm / 22.24 deg。

可见关节也有明显误差，是因为当前 E19 `vis_residual=False`，所有关节都必须穿过离散瓶颈后重新生成，并没有 visible-joint copy path。E28 实现了同骨架 residual copy，但当前 V7 bridge 仍固定使用 E19。

## 4. Temporal 的公共机制

### 4.1 输入 token

固定语义顺序为：

```text
[root, head, left wrist, right wrist, left ankle, right ankle]
```

每帧为 `6 x 9 = 54D`，由 absolute state、速度、SG5 平滑加速度和 Fourier time embedding 相加，再进入 5 层双向 TransformerEncoder。源码：`train_v7.py:200-214, 287-304`。

重要边界：

- 这是双向 self-attention，不是 causal attention。
- 输入是完整 60-frame window。
- SG5 acceleration 本身也会利用局部未来帧。
- world root translation 默认不存在。

### 4.2 公共 decoder

两条 arm 使用同一个形式的 decoder：每个输出时刻的 Fourier clock query，cross-attend 到量化 latent 和起止状态。网络不直接从零生成轨迹，而是预测起点到目标点直线插值的 residual：

```text
base(t) = interpolate(current, target, normalized_time)
motion(t) = orthonormalize(base(t) + decoder_residual(t))
```

源码：`train_v7.py:253-265, 389-427`。

因此 endpoint 是强结构先验。我们的 endpoint-only ridge 已能解释运动摘要的中位 `R2=0.174`；评估 latent 时必须报告相对 endpoint 的增量，而不能把全部可解释度归给 latent。

### 4.3 它当前不是“预测未来”

训练随机选择后半段目标帧 `k`，但传给 encoder 的仍是整个 `wa`，包括 `k` 后面的帧；decoder 同时拿到 `cur`、`tgt`、`k_norm` 和 `T_out`。源码：`train_v7.py:678-703`。

因此现阶段被验证的是：

```text
P(discrete motion representation | complete motion window, endpoints)
```

而不是：

```text
P(future motion | observed prefix, current state, intent)
```

## 5. JEPA arm 的机制

### 5.1 Path bank

完整窗口经过 frame encoder 后，与起点/目标点的 context tokens 拼接；8 个 learned queries 对它们做两层 cross-attention，得到 8 个无序全局 path registers。每个 register 经 6 维、每维 5 levels 的 FSQ。源码：`train_v7.py:216-234, 315-330`。

这 8 个 register 没有固定的时间槽位。时间局部回归的 diagonal 增量只有 0.016，off-diagonal 为 0.011，说明其信息是分布式的，而不是“第 n 个 token 管第 n 段动作”。

### 5.2 Time register

time register 的输入不是单一 duration，而是：

```text
8-point normalized cumulative arc-length curve
+ log(total path length)
+ log2(frame count / 60)
+ mean speed
```

源码：`train_v7.py:340-356`。

线性 probe 结果：

- `time_q` 单独解释 33 个全局/局部运动摘要：中位 `R2=0.631`。
- endpoint + `time_q`：`R2=0.671`。
- endpoint + 全部量化 latent：`R2=0.757`。
- endpoint + 全部连续 latent：`R2=0.780`。

所以 time register 是“全局路径进度/能量摘要”，不是简单的秒数标签。

### 5.3 JEPA loss 的准确含义

decoder hidden 经 projection 后匹配同一个 encoder 的 `frame_h.detach()`：`train_v7.py:730-735`。它没有独立 EMA target encoder，也没有只输入 context 去预测不可见 target。准确表述应是 **JEPA-style stop-gradient latent reconstruction/self-distillation**，而不是经典 I-JEPA 式预测目标块。

## 6. MaskFSQ arm 的机制

MaskFSQ 在 6000-step warmup 后对输入 frame token 采用 none/rand/sparse/prefix 混合 masking。60 帧先按固定边界分成 9 组，各组用自己的 query 在组内做 masked-softmax pooling，再把 9 个 group tokens 送进 2 层全局 Transformer，最后量化为 9 个 FSQ registers。源码：`train_v7.py:235-251, 331-338, 694-701`。

关键点是：**局部性只约束第一次 pooling，不约束最终 code。** `grp_enc` 会让 9 个组再次全局交换信息。统计结果验证了这一点：

- 对应时间组的增量 `R2` 均值：0.325。
- 非对应组的增量 `R2` 均值：0.135。
- 对角约为非对角的 2.4 倍，但不是严格局部。

因此 MaskFSQ 的真实描述是“位置锚定、全局上下文化的时间 token”，不是独立的 9 个局部片段。

## 7. 配对统计结果

### 7.1 同一批验证窗口的重建

审计抽取 1494 个不重复窗口，来自 376 个验证 clip，每个 clip 最多 4 个窗口。所有置信区间按 clip 做 paired bootstrap。

| 指标 | JEPA | MaskFSQ | Mask - JEPA, 95% CI |
|---|---:|---:|---:|
| 位置误差 | 6.743 cm | 6.395 cm | -0.358 [-0.432, -0.286] cm |
| 旋转误差 | 10.647 deg | 9.840 deg | -0.811 [-0.943, -0.684] deg |
| 运动能量保持 | 0.899 | 0.939 | +0.039 [+0.032, +0.046] |

这些是可靠的同样本差异，但**不能直接归因于 global-vs-local token geometry**，因为两条 arm 还同时改变了 masking、JEPA loss、time register、encoder endpoint context 和 invariance augmentation。

### 7.2 因果 token 干预

| 模型 | 干预 | 位置误差 | 相对正确 code 的变化 |
|---|---|---:|---:|
| JEPA | 正确 code | 6.743 cm | - |
| JEPA | 跨样本打乱 path | 11.847 cm | +5.117 cm [4.791, 5.447] |
| JEPA | 跨样本打乱 time | 8.598 cm | +1.856 cm [1.709, 2.009] |
| MaskFSQ | 正确 code | 6.395 cm | - |
| MaskFSQ | 跨样本打乱 path | 15.315 cm | +8.925 cm [8.635, 9.211] |

这证明两种 latent 都被 decoder 实际使用，而不只是与动作相关。JEPA path bank 承载主要细节，time register 提供较粗的全局调制；MaskFSQ 的 9 个 token 共同承担全部路径和时间信息。

`zero code` 的误差更大，但它是训练分布外干预，只能作为 stress test；上述 shuffle 才是更可信的因果比较。

### 7.3 FSQ 信息使用

按每个标量维度独立熵求和得到的使用上界：

| Code bank | 实际使用上界 | 名义容量 |
|---|---:|---:|
| JEPA path | 96.00 bits | 111.45 bits |
| JEPA time | 9.67 bits | 13.93 bits |
| MaskFSQ path | 123.72 bits | 125.38 bits |

注意：标量熵求和忽略维度间依赖，会高估 joint entropy，不能直接等同于真实压缩率。

### 7.4 跨骨架表示

在 human、H1、GR3、Talos 和 held-out G1 上：

- 仅 endpoint 的 body classifier：93.1%。
- JEPA path latent：25.3%。
- MaskFSQ path latent：22.9%。
- 随机水平：20%。

Temporal 的 cross-body InfoNCE 确实显著压低了 embodiment identity。held-out G1 没有成为异常点。不过机器人之间 CKA 普遍高于 human-to-robot CKA，部分原因是所有机器人动作来自同一 retargeting pipeline；这不能单独证明自然跨形态泛化。

## 8. 数据与评估协议风险

1. **Spatial 旧 split 有泄漏风险。** `dual_common.split_indices` 用随机 frame-level `randperm`，相邻帧可能跨 train/val。本文 Spatial probe 改用 AMASS，因此不依赖 E19 的 LAFAN frame split，但原始 E19 val 数字仍可能偏乐观。
2. **Temporal split 是 clip-clean，但不做动作类别分层。** 当前 val 只有 punch 1 clip、kick 2、jacks 2、jump 3。任何单类别“泛化结论”都不稳定。
3. **两条 Temporal 不是单变量消融。** 共同 bit budget 只能保证码率一致，不能消除其他训练差异。
4. **family probe 不是模型 OOD 证据。** 本文 family probe 在训练分布 clip 上做 grouped probe split，只说明 latent 中有可线性读取的动作类别结构。
5. **没有 physics-on 指标。** foot contact、滑移、碰撞、torque、稳定恢复和 actuator feasibility 都不在当前训练图中。

## 9. 目前能证明与不能证明的事

### 已有证据支持

- Spatial 能在不同拓扑之间形成高度共享、但仍含身体签名的全身姿态流形。
- Temporal 能把 2 秒、6 个关键关节的轨迹压到约 62.69 bit/s，并保留相当部分路径、非线性和局部时序信息。
- JEPA latent 是全局、分布式、因子部分解耦的；MaskFSQ latent 是时间锚定但全局上下文化的。
- 连续 latent 到 FSQ 的信息损失具有方向性，JEPA 的速率方向是明确案例。
- 两条 Temporal latent 都对 decoder 输出具有因果作用。

### 目前不能宣称

- 不能宣称 Spatial attention 的 hidden state 被 Temporal 直接建模。
- 不能宣称当前 Temporal 可以只看过去生成未来。
- 不能宣称 JEPA 或 MaskFSQ 已经胜出。
- 不能宣称 token 具有固定的人类可读动作语义。
- 不能宣称系统已学到 contact-aware、physics-solid 或闭环 WBC 控制。
- 不能用当前重建误差证明长时生成、组合泛化或真机鲁棒性。

## 10. 为真正比较两条 Temporal 所需的最小后续实验

这不是选型建议，而是让未来结论可归因的实验矩阵：

1. 固定 token geometry，只开/关 frame masking。
2. 固定 token geometry，只开/关 JEPA stop-gradient loss。
3. 固定 encoder context，只开/关独立 time register。
4. 固定 invariance augmentation，统一为 uniform 或 warp。
5. 改成 prefix-only encoder，严格禁止任何未来帧和 acausal SG 特征。
6. 用 family-stratified clip split，并对每个动作族报告 clip 数和置信区间。
7. Temporal 输出后同时测 six-key error 与 E19 full-body completion error，把 Temporal 与 Spatial 的误差贡献拆开。
8. 增加 world root、contact、IMU 和下游 WBC 闭环评估，避免把运动学重建等同于可执行动作。

## 11. 可复现资产

- 分析脚本：`pose_translator_design/exp_topo/audit_spatiotemporal_latent.py`
- 完整机器可读结果：`pose_translator_design/audits/DEEP_AUDIT_0807/audit.json`
- Temporal 增量回归：`temporal_incremental_r2.png`
- JEPA path/time 分工：`jepa_factorization.png`
- Temporal token 局部性：`temporal_locality.png`
- 速率量化损失：`rate_invariance.png`
- 数据 split 支撑度：`data_split_support.png`

所有审计均只读取现有 checkpoint；没有修改模型参数或重新训练。
