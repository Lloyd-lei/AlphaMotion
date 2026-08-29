# 当前代码骨干（Jupyter式伪代码）

不讲算法，只讲**数据从哪进、经过什么、变成什么形状**。所有形状和数字都是从真实代码/真实缓存里读出来的。

代码位置：`~/Desktop/"be water, robot"/any_body/`，2405行Python。

**SigLIP在哪——先直接回答**：它在三个地方进入网络，但**不在模型里**。它是**离线跑一次、结果存成一个107×768的查找表**，训练时只是查表。所以你在模型代码里看不到"SigLIP"这个词，只看到 `batch.source_semantics`——那个张量就是查表的结果。下面§1详述。

---

## 0. 一句话总览

```
BVH文件 / URDF文件
   ↓ 离线解析 → SkeletonSpec（关节名、父子表、rest位置、rest旋转、轴、限位）
   ↓ 离线FK   → 每帧每关节的 (位置, ΔR)
   ↓ 离线缓存 → .npz 分片
   ─────────────────────────────────
训练时:
   源骨架的 J_src 个token  ──encoder──→  8个part token ──瓶颈──→  z
                                                                  ↓
   目标骨架的 J_tgt 个query ←──────── decoder（cross-attend到z）──┘
                                        ↓
                                  每关节 (位置, ΔR)
```

**关键点：源和目标的关节数完全不同，中间的 `z` 与骨架无关。** 这是"翻译官"的全部机制。

---

## 1. SigLIP：离线一次，存成查找表

### 1.1 离线阶段（`semantics.py`，跑一次就完事）

```python
# ---- 第一步：把107个关节名规范化成自然语言 ----
# normalize_joint_name() 做两件事：查别名表，然后拆驼峰/下划线
_ALIASES = {
    "LeftLeg":  "left thigh, upper leg",    # 陷阱：SOMA的"Leg"其实是大腿
    "LeftShin": "left shin, lower leg",
    "Hips":     "pelvis, hips",
    "Chest":    "upper torso, chest",
    ...
}
normalize_joint_name("LeftForeArm")           # -> "left forearm"
normalize_joint_name("left_wrist_roll_joint") # -> "left wrist roll"   (剥掉 _joint 后缀)
normalize_joint_name("LeftHand")              # -> "left hand"

# ---- 第二步：套一个prompt模板，喂给SigLIP文本塔 ----
prompts = [f"an articulated body joint: {normalize_joint_name(n)}" for n in all_names]
#   例: "an articulated body joint: left hand"

model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
model.eval().requires_grad_(False)              # 冻结，永不训练
features = model.get_text_features(**tokenizer(prompts))
features = F.normalize(features, dim=-1)        # (107, 768)

torch.save({"names": all_names, "features": features}, "siglip_joint_semantics.pt")
```

**这个文件已经存在了**：`/var/tmp/alphamotion/any_body/cache/siglip_joint_semantics.pt`，107个名字 × 768维，checkpoint是 `google/siglip-base-patch16-224`。只用**文本塔**，图像塔完全没加载。

### 1.2 训练时（`dataset.py` 的 collate）

```python
# 每个batch，对源骨架和目标骨架各查一次表
source_semantics = semantic_store.encode(source_spec.joint_names)  # (J_src, 768)
target_semantics = semantic_store.encode(target_spec.joint_names)  # (J_tgt, 768)
# 查不到的名字（动物骨架、新机器人）→ 退化成一个hash的3-gram特征，同样768维
```

**所以"不同骨骼的SigLIP标记"就在这里**：SOMA用SOMA的关节名查表，G1用G1的关节名查表，两套名字查的是**同一张表**。这是跨骨架能对齐的语义基础——`LeftHand` 和 `left_wrist_yaw_joint` 会得到相近的向量，模型据此知道"这两个关节功能上是一回事"。

实测确认这个机制生效：

| 关节对 | cosine |
|---|---|
| `LeftHand` ↔ `left_wrist_yaw_joint`（跨骨架，同语义） | **+0.928** |
| `LeftFoot` ↔ `left_ankle_roll_joint`（跨骨架，同语义） | **+0.914** |

### 1.3 进入网络（`network.py` 第119-122、137-140行）

```python
# 源侧
source = self.source(batch.source_features)          # Linear(10 -> 256)
source_semantic = F.dropout(self.semantic(batch.source_semantics), p=0.15)
#                            └─ Linear(768 -> 256, bias=False)，这一层是可训练的
source = source + source_semantic + self.anchor(fourier(batch.source_anchor))
#        └─ 三路相加：姿态数值 + 关节身份 + 骨架几何

# 目标侧（对称）
target_semantic = F.dropout(self.semantic(batch.target_semantics), p=0.15)
target_base = target_semantic + self.anchor(fourier(batch.target_anchor))
#             注意：目标侧没有姿态数值（那正是要预测的东西）
```

**只有那个 `Linear(768→256)` 投影是训练的，SigLIP本体冻结。** 源侧和目标侧**共用同一个投影层**——这是有意的，两边必须落在同一个语义空间里才能对齐。

---

## 2. 一个训练样本长什么样

```python
# ============ 源侧：J_src 个 token，每个 10 维 ============
x_src[j] = [
    p[j],              # (3,)  root-relative 位置, cm
    dR6[j] * avail[j], # (6,)  rest-relative delta rotation 的6D表示
    avail[j],          # (1,)  这个样本有没有旋转GT（动物常常没有）
]                      # 共 10 维  → Linear(10, 256)

# ============ 目标侧：J_tgt 个 query ============
#   语义:   target_semantics  (J_tgt, 768)   ← SigLIP查表
#   几何:   target_anchor     (J_tgt, 4)     ← 父子骨向量(方向3 + 归一化长度1)
#                                              过Fourier编码 → 52维
#   约束:   pos_mask (J_tgt,)   pos_value (J_tgt,3)
#           rot_mask (J_tgt,)   rot_value (J_tgt,6)
#   起点:   target_rest_positions (J_tgt,3)  ← decoder的frame从rest pose起步
#   URDF:   target_joint_axes (J_tgt,3)   target_joint_limits (J_tgt,2)

# ============ 图结构（只源侧用）============
source_hop        (J_src, J_src)  # 最短路径跳数，截断到8
source_same_chain (J_src, J_src)  # 是否同一条运动链
source_edge_type  (J_src, J_src)  # 边类型
```

---

## 3. Encoder：源姿态 → 8个part token

```python
h = source                                    # (J_src, 256)

for layer in encoder:                         # 6层
    # attention logits 上加三个可学习的结构性 bias（Graphormer式）
    logits = q @ k.T / sqrt(d)
    logits += hop_bias[source_hop]             # 拓扑距离
    logits += chain_bias * source_same_chain
    logits += edge_bias[source_edge_type]
    h = h + attn(softmax(logits) @ v) + ffn(h)
# h: (J_src, 256)

# ---- Part Pooling：J_src 个关节 → 固定 8 个 token ----
# 这是"骨架无关"的关键一步：无论 J_src 是 29 还是 71，出来都是 8 个
Q_part = nn.Parameter(randn(8, 256))          # 8个可学习的"摘要员"
A = softmax(Q_part @ h.T / sqrt(256))         # (8, J_src)  ← 注意在关节轴归一化
u = LayerNorm(Q_part + A @ h)                 # (8, 256)

# ---- 瓶颈：七个变体唯一不同的地方 ----
z = bottleneck(u)                             # (8, 256) 或量化后的等价物
```

七个变体：

| 变体 | 做什么 |
|---|---|
| `direct_token` | **跳过pooling**，`z = h`，形状 `(J_src, 256)` |
| `ae` | pooling后 `z = u`，无约束 |
| `vae` | `μ,logσ = Linear(u)`，重参数化采样 + KL |
| `cvae` | 同上，但 posterior 额外吃 `target_summary` |
| `vqvae` | 单码本最近邻查找 + EMA更新 |
| `rvq` | 两层残差量化 |
| `fsq` | 逐维 clip + round 到固定格点 |

---

## 4. Decoder：z + 目标骨架描述 → 目标姿态

```python
query     = target_base                       # (J_tgt, 256) 语义+几何
query    += constraint_pos(cat(pos_value, pos_mask))   # 约束注入Query
query    += constraint_rot(cat(rot_value, rot_mask))
frame_pos = target_rest_positions             # (J_tgt, 3) 从rest pose起步
frame_rot = identity                          # (J_tgt, 6) 从单位旋转起步

for recycle in range(2):                      # AlphaFold式循环精修
    for layer in decoder:                     # 6层
        # 几何bias：当前frame位置的成对距离统计量
        distance = cdist(frame_pos, frame_pos) / 100
        geo      = stack(distance.mean(-1), .amin(-1), .amax(-1), .std(-1))  # (J_tgt,4)
        query    = query + geometry_mlp(geo)

        query    = query + self_attn(query)              # 目标关节之间
        query    = query + cross_attn(query, key=z, value=z)   # ← 唯一读到源信息的地方
        query    = query + ffn(query)

        delta      = Linear(query, 9)
        frame_pos  = frame_pos + delta[:3]
        frame_rot  = frame_rot ∘ delta[3:9]
    if recycle_detach: frame_pos, frame_rot = detach(...)

# ---- 输出：一个通用头，零 per-embodiment 参数 ----
pred      = Linear(query, 9)
p_out_pre = frame_pos + pred[:3]              # (J_tgt, 3)
dR_out_pre= frame_rot ∘ pred[3:9]             # (J_tgt, 6)

# RePaint式硬替换：被约束的分量直接钉死
p_out  = where(pos_mask, pos_value, p_out_pre)
dR_out = where(rot_mask, rot_value, dR_out_pre)
```

**注意 `cross_attn(query, key=z)` 这一行**——这是目标侧唯一能看到源姿态的通道，而 `z` 已经是骨架无关的。**"源骨架第3号关节对应目标骨架第7号关节"这种映射表从来不存在**，是cross-attention自己按内容找。

---

## 5. Loss

```python
L = 1.0 * smooth_l1(p_out, p_gt)                        # 位置
  + 2.0 * bone_length_error(p_out, p_gt, parents)       # 骨长（权重最高，照搬SAMoR）
  + 1.0 * chordal(dR_out, dR_gt)   [按 rot_avail mask]  # 旋转，1 - tr(ΔRᵀΔR_gt)/3
  + 1.0 * attention_supervision(A, functional_slots)    # 防8个part query塌缩
  + 1.0 * constraint_consistency(pre_clamp, ctrl_value) # 硬替换前也要对
  + 1.0 * joint_limit_violation(dR_out, axes, limits)   # 超URDF限位的平方惩罚
  + 0.1 * axis_residual(...)                            # off-axis残差，上报而非丢弃
  + 1.0 * bottleneck_loss                               # KL / commitment / 0
```

后两项（limit / axis）是我今天补的——**config里声明了权重但从来没实现，静默失效**。

---

## 6. 训练循环

```python
for step in range(60000):
    batch = next(loader)      # batch_size=256, 每个样本一个 (源骨架,目标骨架,帧) 三元组
    out   = model(batch)
    loss  = compute_losses(out, batch, weights)
    loss.backward(); clip_grad(1.0); optimizer.step()
    # AdamW lr=3e-4, warmup 2000, bf16
```

每个样本的 (源→目标) 组合按五条路径随机采：`soma_to_g1` 0.30 / `g1_to_soma` 0.15 / `soma_to_pruned` 0.20 / `same_rig` 0.20 / `animal_cycle` 0.15。

---

## 7. 我在写这份文档时发现的两个新问题

读代码画图的过程本身暴露了两处此前没注意到的偏离。**都跟SigLIP有关**，所以正好在这里说。

### 7.1 SigLIP几乎分不开左右

我算了107个关节名两两的cosine：

```
全体 off-diagonal:  min 0.753   中位 0.874   max 0.999   std 0.045
```

分布本来就很挤（都在0.75以上，因为prompt模板前缀相同）。但左右这一对特别糟：

| 关节对 | cosine |
|---|---|
| `LeftHand` ↔ `RightHand` | **+0.985** |
| `LeftFoot` ↔ `RightFoot` | **+0.990** |
| `LeftArm` ↔ `RightArm` | **+0.993** |
| `left_knee_joint` ↔ `right_knee_joint` | **+0.987** |
| — 作为对照，语义无关的 — | |
| `Head` ↔ `LeftFoot` | +0.850 |
| `Chest` ↔ `RightFoot` | +0.826 |

**左右同名关节的相似度(0.989)比整体中位数(0.874)还高0.115。** 意思是：在SigLIP的语义空间里，"左手"和"右手"几乎是同一个词——比"头"和"左脚"的距离近得多，而后者本该是完全不相关的。

**为什么这要紧**：左右手性是这个项目最不能出错的东西（镜像bug会让整个动作翻转），而语义通道基本没在提供这个信息。现在区分左右只能靠两个来源——**几何anchor**（父子骨向量的x分量符号）和**图结构bias**（左右链是不同的子树）。这可能够用，但意味着**如果几何anchor出问题，模型没有第二道防线**。

顺带这也解释了为什么decoder用距离几何（`cdist`统计量）是个隐患：距离矩阵本身镜像不变，而语义又分不开左右，两个通道都对手性不敏感。

**已修，但走的不是我原先的建议——那个想法实测是错的。**

我原打算改prompt强化侧别。实际测了三种模板（同一批12个关节名，看"左右对相似度"与"语义无关对相似度"的差距，**差距越小越好**）：

| prompt模板 | 左右对 | 无关对 | **差距** |
|---|---|---|---|
| `"an articulated body joint: left hand"`（现状） | 0.9885 | 0.8446 | **0.1439** |
| `"the LEFT side of the body, left limb: left hand"` | 0.9632 | 0.7747 | 0.1885 |
| `"left hand"`（裸名字） | 0.9683 | 0.7413 | 0.2269 |

**强化侧别让情况更糟**（0.144 → 0.189）。原因是加长prompt把所有embedding整体拉散了，无关对的相似度掉得比左右对更快，净效果是相对可分性下降。裸名字最差。**现有模板已经是三者里最好的。**

所以改用**几何**信号——它不依赖名字，因此匿名化对照下依然有效，且对动物骨架同样成立：

`bone_anchor` 从4通道扩到5，新增第5维**侧别（laterality）**：rest pose里该关节沿侧向轴的带符号偏移，按骨架侧向半展归一化。

在真实G1上实测：

| | 数量 | 侧别值范围 |
|---|---|---|
| left关节 | 13 | 全部正，`[+0.434, +1.000]` |
| right关节 | 13 | 全部负，`[-1.000, -0.434]` |
| 中线关节 | 3 | 全部 ≈ 0 |

**26/26个有侧别的关节符号完全正确**，中线关节恰好为0。这是个免费的、确定性的手性信号——比SigLIP在这件事上强得多。

### 7.2 name_dropout 是逐维的，不是逐关节的

```python
source_semantic = F.dropout(self.semantic(batch.source_semantics), p=0.15)
```

`F.dropout` 在 `(J, 256)` 张量上**逐元素**丢。实测（J=20）：

| | 丢掉的维度比例 | **整个名字被丢掉的关节数** |
|---|---|---|
| 现在的实现 | 15.7% | **0 / 20** |
| SAMoR式（逐关节） | — | **3 / 20** |

**每个关节永远保留约85%的名字信息，没有任何一个关节是"完全匿名"的。**

而joint-name dropout的**全部目的**就是让模型学会在名字缺失时也能工作——这样才能zero-shot到没见过的关节名。逐维dropout达不到这个效果，它只是普通的正则化。

**直接后果**：T-c的C-1对照（100% name-drop测匿名化）会是**完全的分布外测试**。模型训练时从没见过一个无名关节，评估时突然全部无名，那个数字会很差，但那不能证明"模型依赖名字走捷径"——只能证明"我们没训练这种情况"。**这个对照会失去它的鉴别力。**

**已修**：改成逐关节mask，并按 `1/(1-p)` 缩放保持期望不变（`_drop_joint_names()`）：

```python
keep = (torch.rand(B, J) >= p).to(dtype)       # 注意是 (B,J) 不是 (B,J,D)
return features * keep[..., None] / (1 - p)
```

实测（4×20=80个关节，p=0.15）：**完全匿名的关节从 0/80 变成 10/80**（期望约12）。eval模式下不做任何丢弃，已验证张量逐位相同。

---

## 8. 状态小结

| 部分 | 状态 |
|---|---|
| SOMA BVH / G1 URDF 解析 + FK | 已实现 |
| ΔR 提取（rest-relative） | 已实现 |
| SigLIP离线缓存（107名×768） | **已生成** |
| Graph-transformer encoder + 三个结构bias | 已实现 |
| Part pooling (K=8) | 已实现 |
| 七个瓶颈变体 | 已实现，码率已核验（12/16/20/24精确） |
| Decoder + recycling + 硬替换 | 已实现 |
| 位置/骨长/旋转/attention/约束 loss | 已实现 |
| 限位 / 轴残差 loss | **今天补的**（此前静默失效） |
| 稀疏约束采样 | **今天修的**（此前92%≈全约束） |
| Truebones动物数据 | **未下载** |
| `L_adv` 具身判别器 | 未实现（等动物数据） |
| 完整链路 re-gauge 测试 | 未实现（现有的只验代数恒等式） |
| 左右手性信号 | **已修**：几何侧别通道，26/26关节符号正确（§7.1） |
| 逐关节 name dropout | **已修**：10/80完全匿名（此前0/80）（§7.2） |

---

## 9. 七变体冒烟测试

`tests/smoke_round1.py`，CPU、合成骨架（19关节 → 15关节）、每变体2个优化step、约2秒。

```
[1/3] review fixes (J_src=19 -> J_tgt=15)
  sparse constraint density   33%  (was 92-100% before fix)
  laterality separates L/R    6L vs 6R, sign-consistent
  per-joint name dropout      10/80 joints fully anonymised (was 0/80)
  rate-sweep guard            4 continuous excluded

[2/3] all 7 candidates, 2 optimisation steps each
      variant             z shape    params      loss0      loss1    limit     axis
      direct_token       (19, 64)   190,474    172.841    170.925    1.534    3.146
      ae                  (8, 64)   190,474    176.584    174.816    1.511    3.015
      vae                 (8, 64)   198,794    169.429    166.868    1.763    3.738
      cvae                (8, 64)   206,986    162.000    162.135    2.196    3.669
      vqvae               (8, 64)   190,474    162.059    159.934    1.576    2.384
      rvq                 (8, 64)   190,474    167.225    165.675    2.395    4.115
      fsq                 (8, 64)   191,570    172.679    168.910    2.486    4.231

[3/3] rate sweep on the 3 quantized variants
      vqvae          12b:ok  16b:ok  20b:N/A  24b:N/A
      rvq            12b:ok  16b:ok  20b:ok  24b:ok
      fsq            12b:ok  16b:ok  20b:ok  24b:ok
```

**几点值得注意**：

1. **`direct_token` 的 `z` 是 `(19,64)`，其余是 `(8,64)`** —— 瓶颈宽2.4倍（真实骨架上是~10倍）。这正是它作为"压缩是否损失精度"对照组的意义，也是它不公平的地方，报告必须标注。
2. **`vqvae` 在20/24 bits上是 `N/A`，不是失败** —— 单码本20 bits需 $2^{20}\approx10^6$ 个码字，`make_bottleneck` 主动拒绝。这**验证了spec §3.3的预测**：若目标精度要求高码率，单码本VQVAE有结构性上限。RVQ和FSQ四个码率点全通。
3. **`limit` 和 `axis` 七个变体全部非零** —— 这两项此前静默失效，现在确实参与梯度。
4. **loss数值无意义**（合成随机数据，2步），只用于断言有限性和梯度可达。

测试断言的是**契约**不是质量：七变体IO一致、损失有限、梯度到达参数、part token数=8（`direct_token`除外应为None）、限位/轴项存在、未实现权重会报错。

**回归**：Codex原有9项测试全过，`ruff` 干净。我把 `test_model.py` 里硬编码的anchor宽度改成从 `bone_anchor` 反推，这样以后改通道数不会再静默失配。
