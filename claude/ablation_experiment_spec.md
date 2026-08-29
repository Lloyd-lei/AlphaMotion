# 实验方案：Spatial Layer 瓶颈模块消融

目的：验证`architecture_final_design.md` Part 1.4中"不做量化,保持连续瓶颈"这一结论——目前该结论只有文献理论支持（FSQ逐维量化对SO(3)流形的失真论证），尚无自己的实验数据。本方案设计一次可控消融去正面检验。

---

## 一、共享Backbone（六个变体不变的部分）

**Encoder前半段**（token化 + graph transformer，六个变体完全共享代码）：
```
h_j = MLP([p_j, v_j, r_j]) + CLIP_frozen(name_j) · dropout(p) + PosEnc_tree(j)
h = SelfAttention(h, bias = hop_dist(j,j') + same_chain(j,j') + edge_type(j,j'))   # ×L层
```
CLIP文本塔全程冻结，不参与训练——这一点跟随SAMoR的原始设计，不属于本次消融变量。

**Decoder**（target-topology-as-query + IPA，六个变体完全共享）：
```
Q_j = anchor_embed(T_pose_j^target) + CLIP_frozen(name_j^target) + PosEnc_tree(j)
for recycle in range(N_recycle):
    local_coords = InvariantPointAttention(Q, T, context = bottleneck_output)
    (ΔR_j, Δt_j) = MLP(local_coords)
    T_j = compose(T_j, (ΔR_j, Δt_j))
```

**唯一被替换的部分**：`h -> bottleneck_output` 这一段中间模块，六选一。

---

## 一.5 关节语义/接触特征编码器选型

补充说明：encoder输入除了关节的位置/速度/旋转，还需要覆盖手部/足部contact状态（二值或连续压力值,按肢体token同样的方式接入`h_j`的MLP输入，不单独设编码器）。这里单独讨论的是"关节名字符串→语义向量"这一步的编码器选型（即$CLIP\_frozen(name_j)$这一项具体用什么模型）。

候选：CLIP文本塔 / SigLIP文本塔 / LLaMA原始hidden state / LLM2Vec（对LLaMA做双向化+MNTP+SimCSE对比学习改造后的embedding模型）。

**选型结论：优先SigLIP文本塔,其次CLIP文本塔,不用LLaMA原始hidden state,不引入LLM2Vec改造流程。** 理由：
- LLaMA的causal attention使靠前token拿不到后文信息，直接取其hidden state做语义向量效果通常不佳——这正是LLM2Vec这一整套改造方法存在的原因（若原始hidden state已经够用，不需要这套改造）。LLM2Vec需要额外的双向化+MNTP+对比学习训练流程，对编码"关节名字"这种短字符串的任务而言引入成本过高，收益不确定，本次不纳入正式消融，仅作为已知备选记录。
- SigLIP相对CLIP的具体架构差异：SigLIP文本编码器配置为无因果掩码（token可双向互相看到），CLIP标准文本transformer则使用因果掩码——这使SigLIP的文本塔在架构上更适合做双向语义编码。
- 更重要的长期考量：若未来要把"机器人mesh外观的视觉编码"接入同一个跨模态latent空间（呼应用户"动作表征应同时存在于动作/像素空间"的目标），SigLIP/CLIP的文本塔与其视觉塔天然对齐,可以直接复用；LLaMA/LLM2Vec的embedding空间与任何视觉模型都没有对齐关系，需要额外训练对齐层。这是选型上比短文本编码质量本身更关键的依据。
- 补充说明（澄清此前表述）：SAMoR论文实际选择即为CLIP文本塔编码关节名（这是直接读取该论文方法部分得到的事实），但该论文并未做"文本编码器选型"本身的消融对比，因此不能说CLIP/SigLIP已被该论文"验证为最优"，只能说这是一个已被验证可行、有效的具体先例。

本次消融**固定使用SigLIP文本塔**（六个瓶颈变体统一使用，不作为消融变量本身，避免变量数量失控）；若后续怀疑语义编码器是瓶颈，可再单独做一轮"SigLIP vs CLIP"的小消融。

---

## 二、六个瓶颈变体规格

| 变体 | 机制 | 训练侧loss差异 |
|---|---|---|
| AE | `z = MLP_down(pool(h))`，确定性，无约束 | 仅reconstruction |
| VAE | `z ~ N(μ(h), σ(h))`，reparameterization | + KL散度项 |
| CVAE | `z ~ N(μ(h, e), σ(h, e))`，$e$=骨架/具身embedding作为条件 | + KL散度项，条件化在具身embedding上 |
| VQVAE | `z = codebook[argmin‖pool(h)-codebook‖]`，单层码本 | + commitment loss |
| RVQ | 同VQVAE，但残差多层（R层，逐层量化残差） | + 每层commitment loss |
| FSQ | `z = round(clip(pool(h)))`，逐维独立量化，无codebook | 无额外loss项（FSQ无需commitment） |

`pool(h)`统一使用SAMoR式的K=8 functional-group cross-attention pooling（六个变体这一步相同，只是池化后的连续向量在这里被送入不同的瓶颈处理）。

---

## 三、数据

主数据集：**SOMA→G1**（NVIDIA SOMA-X/soma-retargeter官方发布，SOMA骨架作为输入、Unitree G1作为输出机器人；G1本体29自由度，官方称"其他机器人目标在计划中"——这是本会话检索直接确认过的事实）。**订正**：此前版本在这里写了一个具体的数据集版本名和34/22关节数字，事后核查发现这些细节在本会话任何一次检索或记录里都找不到来源，属于编造，已删除。实现时请以SOMA-X/soma-retargeter官方仓库和文档里实际给出的骨架/关节定义为准，不要依赖本文档给出任何未在官方源里核实过的具体数字。

若要评估"跨本体动作相似度"（见四.3/四.4），额外需要至少一份**无GT配对的第二具身数据**用于跑swap/DTW评估——若Optimus/Figure的数据管线尚未就绪，可暂用SOMA自身按不同骨架比例/关节子集人工构造的"伪异构骨架"作为过渡验证（不追求物理真实，只验证方法本身在J变化时是否work），待真实第二具身数据就绪后替换。

---

## 四、评估指标（两个维度：SE3精确控制 + 跨本体动作相似度）

**维度A：SE3精确控制**（仅SOMA→G1，有GT）

1. **末端关节位置误差**（mm）：只统计末端/次末端关节（手腕/脚踝/头部等，architecture_final_design.md定义的显式token集合），不是全部关节的平均——这是用户实际关心的精度量,不是动画重建质量。
2. **旋转测地距离误差**（度）：$d(R_\text{pred}, R_\text{gt}) = \arccos\left(\frac{\text{tr}(R_\text{pred}^\top R_\text{gt})-1}{2}\right)$，同样只统计末端关节。
3. **帧间瓶颈跳变指标**（新增，SAMoR原论文未做过）：给定连续两帧几乎相同的输入姿态（比如同一个静止pose加极小噪声），测量 $\|z_t - z_{t-1}\|$ 或对应的decode输出差异——直接检验"量化边界跳变"这个理论担忧是否在实践中真实发生，及其量级。

未采用原论文的整体MPJPE，因为SAMoR面向的是动画角色重建质量，和本项目要求的SE3控制精度（末端执行器的绝对误差）不是同一个评估目标。

**维度B：跨本体动作相似度**（不要求GT,可在无配对数据的具身上跑）

4. **归一化末端轨迹DTW距离**：源动作与retarget后动作,各自的末端关节轨迹按肢体长度做尺度归一化,用动态时间规整（DTW）计算距离——不要求两侧关节数相同,天然适合跨构型比较。
5. **Swap-consistency误差**：源$z$ → decode到目标具身B → 重新encode → 与原始$z$的余弦距离（$\mathcal L_\text{swap}$本身也是现成的评估指标,不需要单独设计新机制）。

---

## 五、预期结论（假设，需要实验验证，不是既定结果）

- 连续瓶颈（AE）在指标1、2、3上应全面优于三种量化方案（VQVAE/RVQ/FSQ），尤其在指标3（跳变）上差距应最明显。
- VAE/CVAE因KL约束，预期在指标1、2上略逊于AE，换取的是隐空间可采样性——这个代价是否值得，取决于后续时间层是否真的需要从这个瓶颈直接采样（当前architecture_final_design.md的时间层是在$z$序列上做flow matching，并不需要spatial瓶颈本身可采样，因此**若实验证实AE已经足够，VAE/CVAE的KL约束可能是不必要的额外精度损失**）。
- RVQ相比单层VQVAE预期跳变问题有所缓解（残差量化误差更小），但不改变"量化本身在流形边界处失真"这一结构性问题。

---

## 六、产出物

一份对比表（六行×五列指标）+ 帧间跳变的可视化曲线（哪个变体在哪些姿态区域跳变最严重）。该结果将直接决定`architecture_final_design.md` Part 1.4的结论是维持还是修正。

---

## 七、Loss设计与产品体验（UX）的对应关系

在四项通用loss（$\mathcal L_\text{direct}$、$\mathcal L_\text{swap}$、$\mathcal L_\text{phys}$、$\mathcal L_\text{adv}$，见architecture_final_design.md 1.5节）之外，为对齐此前定义的七项产品功能，需要补充以下训练机制（非新增独立loss项，而是训练数据/监督方式的具体设计）：

- **稀疏输入脑补**：训练时需要显式的masked-reconstruction curriculum——对输入关节做随机比例（建议0%~90%）的mask，loss覆盖被mask掉的关节。不能只依赖架构上"允许缺失输入"这一能力，必须让训练分布里大量出现缺失场景，否则测试时的真实稀疏输入会落在分布外。
- **关键帧编辑/局部精修**：对Is_Controlled=1的关节，即便训练时hard-clamp到约束值，仍需在clamp之前的网络原始输出上额外计一项"预clamp一致性"loss（若无clamp,网络本该收敛到该值），否则网络可能学会"反正会被钉死,可以随意输出"，导致clamp边界处出现可感知的不连续。
- **可行性反馈**：需要专门构造一批"故意不可行"的约束样本（目标坐标设在肢体最大伸展范围之外），训练一个独立的可行性判别头，用二分类loss监督——不能假设解析层的报警阈值不需要训练数据就能设对。
- **实时遥操作（帧间抖动）**：把四.3的"帧间瓶颈跳变指标"直接复用为训练时的正则项——对连续两帧相近输入，惩罚输出frame的跳变幅度。
- **置信度分数**（此前architecture_final_design.md Part 3标注为"待设计"，现给出具体方案）：采用Kendall式不确定性加权，每个输出关节额外预测方差$\sigma_j^2$，训练目标从普通L2替换为 $\frac{\|y-\hat y\|^2}{2\sigma_j^2}+\frac12\log\sigma_j^2$。该头训练完成后$\sigma_j$本身即为逐关节置信度分数，可直接暴露给用户，无需另设机制。

---

## 八、Spatial层不接文字prompt——设计边界确认

Spatial层职责是"给定部分关节约束，求解全身姿态"这一结构明确的（欠定）逆问题，约束本身已构成"提示"；"根据文字联想动作演化"属于开放的语言消歧问题，划归Temporal层处理，避免spatial层职责混乱。

需留意一个边界情况：即便在单帧内，稀疏约束本身有时存在多解歧义（如仅约束手腕+骨盆时，"蹲下"与"坐着"可能同时满足约束但整体姿态差异很大）。这不构成"必须在v1加入文字prompt"的理由（时间层的上一帧上下文通常足以消歧），但建议解析层的可行性反馈机制预留一个"多解歧义警告"分支（区别于"不可行报警"）：当稀疏约束几何上可行但存在多个显著不同的合理解时，显式告知用户系统在此处做了不确定的选择。可作为v2的轻量改进项，不影响v1优先聚焦SE3精度与投影合理性的判断。
