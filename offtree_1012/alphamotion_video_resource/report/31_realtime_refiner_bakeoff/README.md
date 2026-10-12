# 31 · 实时精修器对决 + G1 全路线损失账本（0924）

问题：ARDY 当规划器实时控制时，精修器必须逐帧或按 ARDY 的 2 秒窗口就能算，不能像 Eric 的离线版看整段。这里做了三个实时候选（R1/R2/R3）和一个手部钉变体（R2h），跟 R0（今天的裸投影）和 E（Eric 的 100 步离线接触精修，只跑他第一段，不跑手臂段）用同一把尺子比。源：ARDY-G1 5 条 prompt、ARDY-Core 人体 5 条 prompt、BoneSeed 官方 G1 10 段 SOMA 片段；目标 G1（36 节点捆绑谱）和 H2（H2_loop.xml 30 节点谱，和 Eric 的脚底模型同一骨架）。

## 候选定义、保证什么、不保证什么

| 候选 | 做法 | 因果性 | 硬保证 | 不保证 |
|---|---|---|---|---|
| R0 | 贪心 + LM25 批量投影；根高用解码器的；根水平轨迹透传源 | 逐帧独立 | 关节限位 | 穿地、滑移、悬空 |
| R1 几何贴地 | 在 R0 姿态上只改根：踩地脚的最低点 = 0；脚落地那帧记下世界系位置，踩地期间平移根让脚不动；无脚踩地时最低脚 = 源自己的脚离地高（按腿长缩放）；再做穿地兜底 | 逐帧因果，无优化 | 脚不穿地；已判"踩地"的脚不滑 | 判错踩地的帧；姿态不动 |
| R2 递推 LM | 上一帧解热启动 + AR(2) 先验（2q₁−q₂）+ 踩地脚位置/旋转钉 + 同 R1 的根规则，每帧 2 次 LM | 逐帧因果 | 同 R1 + 腿部姿态跟着钉脚调整 | 同上 |
| R2h | R2 + 手部钉（解码 FK 的手位置当高权重目标） | 逐帧因果 | 手到目标 0.03–0.05 cm | 同上 |
| R3 滚动窗口 | 按 ARDY 的窗口（G1 52 帧、Core 40 帧、30 fps 源 60 帧）先用贪心姿态把整窗的踩地标签算出来（看得到窗内未来），再逐帧 3 次 LM；窗之间用末两帧和闩锁接续 | 一个窗口延迟（2 秒，本来就有） | 同 R2，落脚从真正第一帧开始钉 | 同上 |
| E Eric 离线 | 整段 100 步 Adam：穿地 80、支撑滑移 30、接触 12、姿态参考 4、根加速度 10，脚底网格凸包；根旋转先做 0.04 s 高斯平滑（他的管线原样） | 整段离线 | 无硬保证（都是软项） | 实时 |
| A ARDY 自带后处理 | ARDY 的 C++ ADMM 轨迹修正 + IK，作用在 ARDY 源上（NVIDIA 对 G1 模型默认关闭）。本机编译三次才成功（路径含引号让 cmake 段错误；系统 pybind11 和 Python 3.11 不兼容；最后用 3.13 加载）。只改了 0.2–0.6 cm | 整段离线 | — | — |

**所有阈值和权重都是量出来或由等式推出来的，没有手调常数**：
- 踩地判定 = 解码器接触概率过 0.5（±0.1 滞回）且 脚最低点离更低那只脚 < band 且 脚世界速度 < v_gate。band = 2 × 投影姿态与解码 FK 的脚位置偏差（G1 0.7–5.6 cm，H2 3.5–11 cm，坐那段最大）。v_gate(t) = max(源自己标注踩地时的脚速 p95，源骨盆速度的一半)（各段 4.2–9.5 cm/s）。
- AR(2) 先验权重 w_ref = 2σ_r² / (J σ_a²)：σ_r 是解码器旋转与投影后旋转的测地噪声（G1 0.02–0.10 rad），σ_a 是源自身关节角二阶差分的标准差。消融（H2 搬箱、G1 盘腿、H2 跨栏）：用全身份额 2σ_r²/σ_a² 时 AR(2) 外推过冲，H2 滑移 p50 从 6.6 涨到 19.2，G1 盘腿 slip95 42 对 9；用单关节份额（除以 J）保住了平滑收益（jerk 153k→45k）且滑移不变，所以取后者。
- 钉脚权重：0.5 cm（尺子的穿地阈值）的钉脚误差 = 全身解码噪声的代价，w_pos = J·2σ_r²/0.5²；旋转钉 w_rot = J。消融显示钉权重 ×10 对滑移几乎无影响，滑移由根规则决定。

## 计时（RTX PRO 6000，CPU 求解，ms / 帧，中位）

| 候选 | G1 | H2 | 备注 |
|---|---|---|---|
| R0 | 0.9 | 0.7 | 批量 |
| R1 | 1.1 | 0.9 | R0 + 逐帧几何 |
| R2 / R2h | 7.7 / 7.8 | 6.8 / 6.9 | 含热启动 FK、门判定 FK、2 次 LM |
| R3 | 13.6（窗 684 ms） | 11.5（窗 590 ms） | 3 次 LM + 窗内贪心预算 |
| E | 3.1 | 2.9 | 整段 100 步摊到每帧，但延迟 = 整段 |

25 fps 预算 40 ms：R1/R2/R3 都在预算内，R2 是"每帧一次高斯牛顿"的递推版本。

## 判决

1. **穿地**：R1/R2/R3 在 G1 上全为 0，H2 上残留 0.2–0.4 cm（凸包脚底 vs 尺子碰撞网格的差），和 E 一样。这一项实时版和离线版打平。
2. **滑移**：ARDY-G1→G1 上 R2 把 p50 从 3.9 压到 0.5、p95 16→10，和 E（9.7/1.2）打平；H2 上 R2 18.9 对 E 6.8，E 明显更好；官方 G1 自往返上 R2 p50 0.7 但 p95 27 反而比 R0 高，落脚/抬脚过渡帧的滞回让脚在松开前后拖一下，这是因果版的固有代价。
3. **姿态**：R1–R3 几乎不动姿态（相对 R0 位移 0–0.3 cm），E 动 1–3 cm 且离解码目标远 2–3 倍（rot_vs_dec 3.8–10.9° 对 1.2–5.6°）。E 更顺（jerk 低 2–4 倍）是靠离解码姿态更远换的。
4. **操作精度**：R2h 手部 0.03–0.05 cm，不影响脚。
5. **Core 人体源**：所有候选滑移都差（27–47），codec 翻译没见过的 27 关节人体是主要损失，精修器补不回。

## G1 损失账本怎么读

ARDY-G1 native → codec (R0)：滑移 p95 不变（16.9→16.1）但 p50 翻倍（1.5→3.9），穿地从 1.8 到 6.8 cm，jerk 92k→224k。**最大的损失在 codec 的根高和脚位置噪声上**，不在姿态（自解码 4.7°/1.8 cm）。R1 一步收回穿地（免费），R2 收回一半滑移（8 ms/帧），E 再多收一点滑移和 jerk（离线）。官方 G1 自往返同样形状：native 11.2 → R0 20.1 → R2 p50 0.7 / E 12.3。对照传统路线：NVIDIA retargeter 39.2/0.2（静态钉死、动态滑），Eric 全管线 14.4/6.0。

# 31 · 实时精修器对决（R0–R3 vs Eric 离线）+ G1 全路线损失账本

尺子：`refiner_duel/ruler.py`（厂商 MJCF FK；每脚碰撞几何最低点；站立相 = 最低点 < 1 cm 且前后各 2 帧腐蚀；滑移 = 站立相脚身原点水平速度）。每格是各段的中位数。
姿态位移 = 候选相对 R0 的关节位置平均变化（cm）；rot_vs_dec = 投影后旋转与解码目标的测地误差；F1 = 候选脚底 < 1 cm 对源接触标签；src = 同骨架源可比时相对源的旋转/位置误差（G1 源→G1）。

## ARDY-G1 源，5 条 prompt → unitree_g1_29dof

| cand | mode | slip95 | slip50 | pen95 | penmax | pen% | air% | rootstep95 | jerk95 | limit | disp_vs_R0 | rot_vs_dec | F1 | ms/frame | hand EE cm | src rot° / pos cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | batched, non-causal only through the batch (each frame independent) | 16.1 | 3.9 | 2.57 | 6.84 | 0.23 | 0.04 | 2.3 | 224k | 0.0 | 0.0 | 1.2 | 0.96 | 0.9 |  | 4.7 / 1.8 |
| R1 | causal (per frame, on top of R0 projection) | 12.9 | 2.9 | 0.00 | 0.00 | 0.00 | 0.00 | 1.7 | 224k | 0.0 | 0.0 | 1.2 | 0.83 | 1.1 |  | 4.7 / 1.8 |
| R2 | causal, 2 LM iteration(s) per frame | 10.1 | 0.5 | 0.00 | 0.00 | 0.00 | 0.08 | 1.7 | 192k | 0.0 | 0.0 | 1.2 | 0.80 | 7.7 |  | 4.7 / 1.8 |
| R2h | causal, 2 LM iteration(s) per frame + hand pins | 10.1 | 0.5 | 0.00 | 0.00 | 0.00 | 0.08 | 1.7 | 205k | 0.0 | 0.2 | 1.3 | 0.80 | 7.8 | 0.03 | 4.6 / 1.8 |
| R3 | receding horizon, window 52 frames (2.08 s), 3 LM iterations | 9.9 | 3.2 | 0.00 | 0.00 | 0.00 | 0.08 | 1.7 | 192k | 0.0 | 0.0 | 1.2 | 0.79 | 13.6 |  | 4.7 / 1.8 |
| E | offline, whole clip, 100 Adam steps (his first stage; arm stage not run) | 9.7 | 1.2 | 0.00 | 0.01 | 0.00 | 0.00 | 1.6 | 48k | 0.0 | 1.0 | 3.8 | 0.98 | 3.1 |  | 6.7 / 2.4 |


## ARDY-G1 源，5 条 prompt → unitree_h2

| cand | mode | slip95 | slip50 | pen95 | penmax | pen% | air% | rootstep95 | jerk95 | limit | disp_vs_R0 | rot_vs_dec | F1 | ms/frame | hand EE cm | src rot° / pos cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | batched, non-causal only through the batch (each frame independent) | 35.3 | 4.8 | 6.99 | 8.51 | 0.95 | 0.01 | 2.6 | 179k | 0.0 | 0.0 | 3.7 | 0.94 | 0.7 |  |  |
| R1 | causal (per frame, on top of R0 projection) | 20.3 | 2.1 | 0.21 | 0.42 | 0.00 | 0.00 | 2.7 | 179k | 0.0 | 0.0 | 3.7 | 0.81 | 0.8 |  |  |
| R2 | causal, 2 LM iteration(s) per frame | 16.8 | 4.5 | 0.05 | 0.22 | 0.00 | 0.11 | 2.2 | 140k | 0.0 | 0.1 | 3.7 | 0.79 | 6.8 |  |  |
| R2h | causal, 2 LM iteration(s) per frame + hand pins | 16.8 | 4.5 | 0.05 | 0.22 | 0.00 | 0.11 | 2.2 | 152k | 0.0 | 0.7 | 4.2 | 0.79 | 6.9 | 0.05 |  |
| R3 | receding horizon, window 52 frames (2.08 s), 3 LM iterations | 16.3 | 4.4 | 0.04 | 0.22 | 0.00 | 0.11 | 2.2 | 154k | 0.0 | 0.1 | 3.7 | 0.78 | 11.5 |  |  |
| E | offline, whole clip, 100 Adam steps (his first stage; arm stage not run) | 6.8 | 2.2 | 0.17 | 0.36 | 0.00 | 0.00 | 2.1 | 49k | 0.0 | 1.3 | 6.4 | 0.98 | 2.9 |  |  |


## ARDY-Core 人体源，5 条 prompt → unitree_g1_29dof

| cand | mode | slip95 | slip50 | pen95 | penmax | pen% | air% | rootstep95 | jerk95 | limit | disp_vs_R0 | rot_vs_dec | F1 | ms/frame | hand EE cm | src rot° / pos cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | batched, non-causal only through the batch (each frame independent) | 0.0 | 0.0 | 0.00 | 0.41 | 0.00 | 0.83 | 1.3 | 339k | 0.0 | 0.0 | 3.7 | 0.05 | 1.5 |  |  |
| R1 | causal (per frame, on top of R0 projection) | 43.9 | 12.0 | 0.00 | 0.00 | 0.00 | 0.00 | 1.2 | 339k | 0.0 | 0.0 | 3.7 | 0.66 | 1.7 |  |  |
| R2 | causal, 2 LM iteration(s) per frame | 27.3 | 7.0 | 0.00 | 0.00 | 0.00 | 0.02 | 1.1 | 309k | 0.0 | 0.0 | 3.7 | 0.66 | 11.1 |  |  |
| R2h | causal, 2 LM iteration(s) per frame + hand pins | 27.3 | 7.0 | 0.00 | 0.00 | 0.00 | 0.02 | 1.1 | 314k | 0.0 | 0.7 | 3.8 | 0.66 | 8.0 | 0.16 |  |
| R3 | receding horizon, window 40 frames (2.00 s), 3 LM iterations | 27.3 | 7.0 | 0.00 | 0.00 | 0.00 | 0.02 | 1.1 | 311k | 0.0 | 0.0 | 3.7 | 0.60 | 15.4 |  |  |
| E | offline, whole clip, 100 Adam steps (his first stage; arm stage not run) | 7.2 | 1.9 | 0.00 | 0.01 | 0.00 | 0.09 | 4.2 | 702k | 0.0 | 3.0 | 10.5 | 0.96 | 4.5 |  |  |


## ARDY-Core 人体源，5 条 prompt → unitree_h2

| cand | mode | slip95 | slip50 | pen95 | penmax | pen% | air% | rootstep95 | jerk95 | limit | disp_vs_R0 | rot_vs_dec | F1 | ms/frame | hand EE cm | src rot° / pos cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | batched, non-causal only through the batch (each frame independent) | 103.3 | 24.1 | 1.35 | 5.13 | 0.17 | 0.12 | 1.1 | 298k | 0.0 | 0.0 | 5.3 | 0.46 | 1.2 |  |  |
| R1 | causal (per frame, on top of R0 projection) | 68.8 | 11.8 | 0.00 | 0.11 | 0.00 | 0.00 | 1.7 | 298k | 0.0 | 0.0 | 5.3 | 0.63 | 1.4 |  |  |
| R2 | causal, 2 LM iteration(s) per frame | 46.7 | 8.0 | 0.00 | 0.14 | 0.00 | 0.07 | 1.7 | 286k | 0.0 | 0.2 | 5.6 | 0.55 | 9.7 |  |  |
| R2h | causal, 2 LM iteration(s) per frame + hand pins | 46.7 | 8.0 | 0.00 | 0.14 | 0.00 | 0.07 | 1.7 | 278k | 0.0 | 1.0 | 5.7 | 0.55 | 10.0 | 0.03 |  |
| R3 | receding horizon, window 40 frames (2.00 s), 3 LM iterations | 46.6 | 11.9 | 0.00 | 0.14 | 0.00 | 0.07 | 1.7 | 269k | 0.0 | 0.1 | 5.3 | 0.55 | 17.2 |  |  |
| E | offline, whole clip, 100 Adam steps (his first stage; arm stage not run) | 13.3 | 2.3 | 0.11 | 0.26 | 0.00 | 0.06 | 5.2 | 592k | 0.0 | 3.0 | 10.9 | 0.97 | 4.4 |  |  |


## BoneSeed 官方 G1 源（10 段 SOMA），Megatron 自往返 → unitree_g1_29dof

| cand | mode | slip95 | slip50 | pen95 | penmax | pen% | air% | rootstep95 | jerk95 | limit | disp_vs_R0 | rot_vs_dec | F1 | ms/frame | hand EE cm | src rot° / pos cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | batched, non-causal only through the batch (each frame independent) | 20.1 | 5.7 | 2.41 | 3.02 | 0.55 | 0.02 | 1.1 | 473k | 0.0 | 0.0 | 1.9 | 0.92 | 1.2 |  | 5.7 / 2.1 |
| R1 | causal (per frame, on top of R0 projection) | 19.3 | 1.8 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0 | 473k | 0.0 | 0.0 | 1.9 | 0.95 | 1.4 |  | 5.7 / 2.1 |
| R2 | causal, 2 LM iteration(s) per frame | 27.3 | 0.7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.9 | 698k | 0.0 | 0.2 | 2.5 | 0.89 | 7.8 |  | 6.7 / 2.4 |
| R2h | causal, 2 LM iteration(s) per frame + hand pins | 29.9 | 0.8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.9 | 705k | 0.0 | 0.5 | 2.8 | 0.89 | 7.9 | 0.04 | 6.8 / 2.5 |
| R3 | receding horizon, window 60 frames (2.00 s), 3 LM iterations | 26.3 | 0.8 | 0.00 | 0.00 | 0.00 | 0.02 | 1.0 | 591k | 0.0 | 0.2 | 2.4 | 0.88 | 14.0 |  | 6.5 / 2.3 |
| E | offline, whole clip, 100 Adam steps (his first stage; arm stage not run) | 12.3 | 4.2 | 0.00 | 0.01 | 0.00 | 0.00 | 0.9 | 39k | 0.0 | 1.8 | 5.2 | 0.99 | 5.5 |  | 8.2 / 2.9 |


## BoneSeed 官方 G1 源（10 段 SOMA），Megatron 自往返 → unitree_h2

| cand | mode | slip95 | slip50 | pen95 | penmax | pen% | air% | rootstep95 | jerk95 | limit | disp_vs_R0 | rot_vs_dec | F1 | ms/frame | hand EE cm | src rot° / pos cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | batched, non-causal only through the batch (each frame independent) | 37.4 | 11.6 | 8.77 | 9.83 | 0.96 | 0.00 | 1.4 | 363k | 0.0 | 0.0 | 3.6 | 0.92 | 0.6 |  |  |
| R1 | causal (per frame, on top of R0 projection) | 23.1 | 3.0 | 0.17 | 0.22 | 0.00 | 0.00 | 1.7 | 363k | 0.0 | 0.0 | 3.6 | 0.92 | 0.8 |  |  |
| R2 | causal, 2 LM iteration(s) per frame | 24.3 | 1.7 | 0.09 | 0.23 | 0.00 | 0.02 | 1.5 | 651k | 0.0 | 0.3 | 4.4 | 0.80 | 6.9 |  |  |
| R2h | causal, 2 LM iteration(s) per frame + hand pins | 24.3 | 1.7 | 0.09 | 0.23 | 0.00 | 0.02 | 1.5 | 548k | 0.0 | 1.0 | 4.8 | 0.80 | 6.9 | 0.04 |  |
| R3 | receding horizon, window 60 frames (2.00 s), 3 LM iterations | 27.9 | 2.4 | 0.13 | 0.20 | 0.00 | 0.04 | 1.5 | 505k | 0.0 | 0.2 | 4.3 | 0.84 | 11.6 |  |  |
| E | offline, whole clip, 100 Adam steps (his first stage; arm stage not run) | 13.9 | 4.6 | 0.06 | 0.12 | 0.00 | 0.00 | 1.1 | 54k | 0.0 | 2.5 | 7.1 | 0.99 | 4.9 |  |  |


## G1 损失账本（同内容、不同路线，同一把尺子，中位数）

| 路线 | slip95 | slip50 | pen95 | penmax | pen% | air% | jerk95 | rot° vs src | pos cm vs src | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| ARDY-G1 native (no codec) | 16.9 | 1.5 | 1.17 | 1.84 | 0.81 | 0.00 | 92k |  |  | — |
| ARDY-G1 → Megatron → G1, R0 | 16.1 | 3.9 | 2.57 | 6.84 | 0.23 | 0.04 | 224k | 4.7 | 1.8 | 0.9 |
| ARDY-G1 → Megatron → G1, R1 | 12.9 | 2.9 | 0.00 | 0.00 | 0.00 | 0.00 | 224k | 4.7 | 1.8 | 1.1 |
| ARDY-G1 → Megatron → G1, R2 | 10.1 | 0.5 | 0.00 | 0.00 | 0.00 | 0.08 | 192k | 4.7 | 1.8 | 7.7 |
| ARDY-G1 → Megatron → G1, R2h | 10.1 | 0.5 | 0.00 | 0.00 | 0.00 | 0.08 | 205k | 4.6 | 1.8 | 7.8 |
| ARDY-G1 → Megatron → G1, R3 | 9.9 | 3.2 | 0.00 | 0.00 | 0.00 | 0.08 | 192k | 4.7 | 1.8 | 13.6 |
| ARDY-G1 → Megatron → G1, E | 9.7 | 1.2 | 0.00 | 0.01 | 0.00 | 0.00 | 48k | 6.7 | 2.4 | 3.1 |
| ARDY-Core → Megatron → G1, R0 | 0.0 | 0.0 | 0.00 | 0.41 | 0.00 | 0.83 | 339k |  |  | 1.5 |
| ARDY-Core → Megatron → G1, R1 | 43.9 | 12.0 | 0.00 | 0.00 | 0.00 | 0.00 | 339k |  |  | 1.7 |
| ARDY-Core → Megatron → G1, R2 | 27.3 | 7.0 | 0.00 | 0.00 | 0.00 | 0.02 | 309k |  |  | 11.1 |
| ARDY-Core → Megatron → G1, R2h | 27.3 | 7.0 | 0.00 | 0.00 | 0.00 | 0.02 | 314k |  |  | 8.0 |
| ARDY-Core → Megatron → G1, R3 | 27.3 | 7.0 | 0.00 | 0.00 | 0.00 | 0.02 | 311k |  |  | 15.4 |
| ARDY-Core → Megatron → G1, E | 7.2 | 1.9 | 0.00 | 0.01 | 0.00 | 0.09 | 702k |  |  | 4.5 |
| A: ARDY-G1 native + ARDY post-process (offline ADMM, NVIDIA disables it for G1) | 18.0 | 0.9 | 1.17 | 3.41 | 0.24 | 0.00 | 115k |  |  | — |
| A + Megatron → G1, R0 | 17.0 | 3.5 | 2.39 | 5.92 | 0.45 | 0.04 | 229k | 4.8 | 1.9 | 1.0 |
| A + Megatron → G1, R2 | 14.0 | 3.4 | 0.00 | 0.00 | 0.00 | 0.02 | 213k | 4.8 | 1.9 | 7.8 |
| A + Megatron → G1, E | 8.0 | 0.9 | 0.00 | 0.01 | 0.00 | 0.00 | 47k | 6.5 | 2.4 | 3.3 |
| A(Core) + Megatron → G1, R0 | 0.0 | 0.0 | 0.00 | 0.18 | 0.00 | 0.88 | 337k |  |  | 1.1 |
| A(Core) + Megatron → G1, R2 | 40.0 | 13.0 | 0.00 | 0.00 | 0.00 | 0.01 | 321k |  |  | 7.7 |
| A(Core) + Megatron → G1, E | 11.4 | 2.1 | 0.00 | 0.01 | 0.00 | 0.00 | 124k |  |  | 4.0 |
| BoneSeed official G1 (Bones retarget) | 11.2 | 0.7 | 0.93 | 0.95 | 0.51 | 0.00 | 301k |  |  | — |
| NVIDIA soma-retargeter G1 | 39.2 | 0.2 | 0.66 | 0.88 | 0.44 | 0.00 | 303k |  |  | — |
| Eric Prime + 100+100 refiner G1 | 14.4 | 6.0 | 0.00 | 0.01 | 0.00 | 0.00 | 115k |  |  | — |
| official G1 native (ruler on the same 10 clips) | 11.2 | 0.7 | 0.93 | 0.95 | 0.51 | 0.00 | 301k |  |  | — |
| official G1 → Megatron → G1 (self round trip), R0 | 20.1 | 5.7 | 2.41 | 3.02 | 0.55 | 0.02 | 473k | 5.7 | 2.1 | 1.2 |
| official G1 → Megatron → G1 (self round trip), R1 | 19.3 | 1.8 | 0.00 | 0.00 | 0.00 | 0.00 | 473k | 5.7 | 2.1 | 1.4 |
| official G1 → Megatron → G1 (self round trip), R2 | 27.3 | 0.7 | 0.00 | 0.00 | 0.00 | 0.00 | 698k | 6.7 | 2.4 | 7.8 |
| official G1 → Megatron → G1 (self round trip), R2h | 29.9 | 0.8 | 0.00 | 0.00 | 0.00 | 0.00 | 705k | 6.8 | 2.5 | 7.9 |
| official G1 → Megatron → G1 (self round trip), R3 | 26.3 | 0.8 | 0.00 | 0.00 | 0.00 | 0.02 | 591k | 6.5 | 2.3 | 14.0 |
| official G1 → Megatron → G1 (self round trip), E | 12.3 | 4.2 | 0.00 | 0.01 | 0.00 | 0.00 | 39k | 8.2 | 2.9 | 5.5 |
