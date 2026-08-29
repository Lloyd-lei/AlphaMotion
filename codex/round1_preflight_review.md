# Round-1 开工前反证审查

日期：2026-07-30  
审查对象：Claude 在群聊明确授权的四份 Round-1 文档，以及仓库现有 Stage-1 代码与实验记录。  
结论：**当前可以继续做 M0 数据/坐标探针，但不应直接实现完整训练器或启动 T0/T1。** 下面的 P0 问题会使实验目标不可识别、指标不可计算，或让“七种瓶颈”比较产生混杂。

## P0：开工前必须改清楚

### 1. 跨拓扑训练样本的监督关系没有定义

`claude/01_round1_spec.md:28-32` 定义了任意源骨架到任意目标骨架，但 `:375-418` 只定义了帧采样和数据切分，没有定义一个 batch 中的 `(source_pose, source_rig, target_rig, target_pose)` 如何构造，也没有各方向采样比例。

- SOMA 和 G1 有逐帧配对，可以做 SOMA→G1 / G1→SOMA。
- Truebones 每个物种只有自己的动作，没有“同一姿态在另一拓扑上的目标真值”。
- held-out 动物上的自重建只能证明新拓扑自编码，不能证明跨拓扑翻译。
- `claude/02_stage1_project_plan.md:151-158` 已承认 Go2 没有配对 GT，但 `01_round1_spec.md:497` 又要求报告 Go2 末端误差，两份文档直接冲突。

需要先给出训练任务矩阵、每条路径的 GT 和采样权重，例如：

`SOMA↔G1 paired`、`SOMA↔synthetic-pruned paired`、`same-rig reconstruction`、`animal reconstruction/cycle only`。Truebones 只能作为拓扑正则或自重建证据，不能直接当跨拓扑翻译 GT。

### 2. “canonical SE(3) frame 跨 rig 可比”尚未被定义

`01_round1_spec.md:65-79` 把 canonical frame 当作核心表示，但没有给出 canonical frame 的构造算法。URDF link frame、BVH joint frame 都是资产作者定义的 gauge；同一个物理骨架可以通过给每个 link 乘固定旋转而保持运动完全相同。仅统一 up/facing/scale 不能消除每根骨的 twist gauge。

因此：

- 从同一个 G1 URDF 做 FK→解析逆变换达到 `4.29e-12 rad`，只证明同一 gauge 内代数可逆；
- 它不证明 SOMA BVH frame、G1 URDF frame、未知机器人 URDF frame 彼此可比；
- 单根 bone direction 只有 2 个方向自由度，绕骨轴的 twist 仍未定义，恰好是方案最关心的信息。

需要先定义对每个节点都确定的 canonical triad，包括 branch、leaf、single-child、左右对称、零长度 joint 和退化共线情况，并验证“对 URDF frame 做任意固定 re-gauge 后，模型输入/输出物理结果不变”。

### 3. root 通道在当前输入下不可识别

`01_round1_spec.md:126-167` 排除了全局 root translation，却要求通用 root 通道输出全局位姿。单帧 root-relative 输入不包含绝对世界位置，模型不可能从中恢复被删除的信息。

必须三选一：

1. root 作为源输入的旁路原样传递；
2. root 完全由 target-side constraint 给定；
3. 本轮只输出 root-relative pose，不声称恢复 global root。

同时 `p_j` 一处写“cm”，另一处要求除以骨架高度 `H`，实际应为无量纲；`ctrl_val` 和评估反归一化的单位也需统一。

### 4. SE(3) constraint 接口目前在形状和语义上都不成立

`01_round1_spec.md:153-163` 的 `ctrl_val` 只有 3 维位置，却称为 SE(3) 约束；`:224` 又写 `T[is_ctrl] = ctrl_val`，把 3-vector 赋给 SE(3) frame。

更根本的问题是：约束属于 **target joints**，但 `is_ctrl/ctrl_val` 被拼进了 **source joint token**。当 `J_src != J_tgt` 时无法对齐，除非重新引入一张语义 correspondence 表。

需要改成 target-query-side contract：

- `pos_mask, pos_value(3)`
- `rot_mask, rot_value(6 or quaternion)`
- 可选 root constraint
- pre-clamp loss 分开定义 position/rotation 的单位和权重
- 不可行约束的检测或投影

### 5. 解析关节角公式只验证了精确 FK 逆，不足以接网络预测

`01_round1_spec.md:249-258` 的 `sign(dot(w,axis))*acos(trace)` 对精确单轴旋转可以工作，但网络输出一般不严格落在该单轴子流形上，并且在 0/π 附近有不连续或梯度奇异。

必须补：

- 将 noisy `SO(3)` 投影到目标 joint axis 的明确定义；
- joint limit clamp 或显式 `L_limit`；
- fixed / continuous / prismatic / mimic joint 处理；
- off-manifold noise、0、接近 π、autograd 的单元测试。

`L_fk = ||FK(q_out)-p_out||` 本身不会“惩罚限位违反”；`01_round1_spec.md:294` 的这一表述错误，必须单独加 limit loss/投影。

### 6. G1 的 serial-axis 节点与 anchor/监督粒度未统一

规格一边说“每个目标 joint 输出一个 frame”，一边在 `01_round1_spec.md:322-343` 用 16 个 SOMA 语义 joint 对应 29 个 G1 DOF group，却没有明确：

- decoder 是对 29 个 revolute intermediate links 输出，还是对约 14 个语义 body sites 输出；
- 如果是 14 个 site，29 个角度不能从 14 个末端 frame 按当前逐 joint 公式直接恢复；
- 如果是 29 个 URDF joints，`L_direct` 到底监督哪 29 个 frame，16 个语义映射如何展开。

我实测官方 `g1_29dof.urdf`：29 个 actuated joints 中 `waist_yaw_joint` 的 origin 位移严格为 `(0,0,0)`。因此 `anchor_j` 中的 `b/||b||` 至少有一个确定的 NaN 输入；短 serial links 也会造成数值不稳。必须定义 zero-length joint 的 axis/origin-based anchor。

### 7. 新 backbone 没有承接已有 Stage-1 的硬实验结论

现有真实实验表明：

- `note/motionformer-research-vision.md:962-976`：input-conditioned pair update / OPM 是主要 OOD 贡献者；
- `note/triangle_sanity_check.md:12-25,33-40,75-83`：full 与 OPM-only 重建误差几乎相同，但 pair geometry 几乎正交，triangle 是表征几何选择器。

新规格 `01_round1_spec.md:188-233` 改成普通 graph self-attention，没有 persistent pair state、OPM gateway 或 triangle/refiner。这样实验同时改变了 backbone 和 bottleneck，无法把差异归因给七种瓶颈，也绕开了已有研究结果。

建议先固定一个支持 variable topology 的 input-conditioned pair backbone：pair 初值由相对几何/拓扑生成，OPM-like gateway 按样本更新，refiner 是否保留由一个小型 2× sanity ablation 决定。旧实现里的 learned `joint_pe` / `pair_init` 固定为 `J×J`，违反 universal scope，机制可复用但代码不能原样搬。

## P1：T1 前必须解决

### 8. 七个瓶颈没有做容量/码率公平控制

`FSQ [8,8,8,8,5,5,5,5]` 的隐式码本是：

`8^4 × 5^4 = 2,560,000`

不是文档写的 640,000。每 token 约 21.29 bits；VQ-1024 是 10 bits；4-layer RVQ 是 40 bits。AE/VAE 是 `K×256` 浮点，Direct-Token 又是 `J×256`。当前比较会把“量化方法、码率、latent 维数、token 数”混在一起。

至少要报告 quality-rate Pareto；更严格则提供 matched-bit variants。

### 9. attention supervision 的归一化方向错误

`A = softmax(..., dim=joint)` 使每个 query 在所有 joints 上和为 1；随后 `-log A[y_j,j]` 却声称是“按 joint 的标准交叉熵”。同一功能组有很多 joints 时，不可能让它们同时都接近 1，loss 会系统性偏向小组。

应改为：

- 对每个 joint 在 K 个 slot 上 softmax 后做 CE；或
- 每个 slot 对所属 joints 的目标分布做 KL，并加入 slot diversity / Sinkhorn assignment。

另一个冲突：human 的 slot 6/7 是左右手，animal 的 slot 6/7 是尾/其他。同一共享 query 被要求学习不同语义，破坏跨具身一致性。

### 10. CVAE 与“通用、target-independent motion latent”冲突

CVAE 把 `e_tgt` 拼进 posterior，得到的 `z` 随目标骨架变化。项目计划又要求第二阶段直接预测 canonical `z`。这两者不能同时成立：同一个动作会因目标机器人不同而有不同 latent。

需要先决定 latent 的契约：

- 若 `z` 是通用 motion state，则 target topology 只能进入 decoder；
- 若允许 target-conditioned `z`，则不能把它称为可跨 embodiment 复用的 temporal latent。

### 11. loss/更新规则仍有文档冲突

- `L_adv` 在 §4.4 出现，但未进入总 loss 表、超参或训练日程；
- `L_vq` 表里写 commitment + entropy，公式没有 entropy；
- VQ 同时写 EMA codebook update 和 codebook gradient loss，更新规则需二选一并明确；
- 直接照搬 SAMoR loss 权重不具可比性：当前表示、单位、任务和 decoder 已改变，应先记录每项 gradient norm 再标定。

### 12. 数据切分和 anchor 有潜在泄漏

- `val_id` 使用同 actor、同 motion 的 held-out frame，只能做 pipeline sanity，不能做泛化结论；
- mirror 必须按原始 motion group 切分，actor split 之外还应按 source sequence family 防重复版本泄漏；
- 固定每 30 帧采样会锁定相位，建议 per-motion 随机 offset；
- 动物用“同一 clip 第一帧”做 target anchor，可能携带动作姿态而不只是 morphology，形成目标泄漏。应使用 canonical rest、跨 clip shape anchor，或做 anchor-leakage ablation。

### 13. 零样本和 swap-consistency 指标尚不可计算

- `g1_23dof` 是 G1 同家族裁剪，不能单独支撑“未见机器人”；
- Go2 没有 paired target GT，不能报告 endpoint error；
- Direct-Token 的 `z` 是 `J×D`，其他是 `K×D`，CVAE 又 target-conditioned，当前没有统一的 swap cosine 定义；
- constraint hard-clamp 后误差约 0 是恒等式，应报告 pre-clamp error、feasibility 和邻域 distortion。

### 14. 仅运动学指标不能支持“可直接用于机器人”的表述

本轮明确不做 WBC/physics，因此结论应限定为“kinematic pose translator”。至少补静态可执行性检查：self-collision、ground penetration、support polygon/contact、joint velocity 不适用但 joint-limit 与 IK residual 必须有。否则“只给 URDF 即输出可用动作”超出了证据。

## P2：工程落地前明确

1. 新项目不要覆盖 `experiment/stage1/`；建议 `experiment/round1_translator/`，旧目录保留为基线证据。
2. mixed variable-J batch 需要 padding mask / graph batch / bucket 策略，所有 attention、pooling、loss 都必须 mask。
3. 现有 `experiment/stage1/data.py:155-163` 使用 per-joint mean/std，不能用于未知拓扑；新管线只能用确定性几何归一化。
4. `01_round1_spec.md` 的 IPA 伪代码缺少 exact tensor contract、cross-attention 方式、residual/LN/FFN 和 padding mask，不能直接实现。
5. 60k steps × batch 256 = 15.36M sample presentations/run；21 runs 的 11–21 小时估算需要先以 T0 profiler 实测，不能用纸面值承诺。
6. `/var/tmp` 适合可重建 cache，不应存唯一 checkpoint/result；需把不可再生结果放持久目录并保留 manifest/hash。
7. Viser 同时加载七个模型应改为 lazy load，否则不必要地占显存。
8. SigLIP 对本轮不是首要阻塞。可以先用 deterministic name features 跑几何 M0/T0；正式 T1 再固定 checkpoint，避免下载/依赖阻塞核心几何验证。

## 建议 Claude 先补的九个明确答案

1. 完整训练 task matrix、方向比例和每条路径的 GT 是什么？
2. canonical frame 如何对任意 URDF/BVH 做 gauge fixing？
3. target decoder 到底输出 29 个 G1 intermediate frames，还是约 14 个 semantic frames？
4. global root 是输入旁路、constraint，还是本轮不预测？
5. SE(3) constraint 是否改为纯 target-side，position/rotation 分离？
6. 是否把 input-conditioned pair refinement 固定进共享 backbone？
7. `val_topo` 对无 paired GT 的动物到底测什么，如何避免把自重建称为翻译？
8. 瓶颈比较按 matched bitrate，还是明确改为 quality-rate Pareto？
9. 本轮 claim 是否收窄为 kinematic translator，而不是 physical/robot-ready output？

在这九项明确前，建议只执行不会固化错误接口的探针：CSV schema/单位/欧拉顺序、坐标系单测、FK、pair manifest、IO benchmark、URDF gauge/zero-length tests。
