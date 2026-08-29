# Controller-readiness check — Greenwich on Fourier GR3 (0818)

Question: can Greenwich output feed a motion-tracking controller (BeyondMimic-style)?
Setup: 6 AMASS walk clips (xwbc AMASS, 30 fps; KIT WalkingStraightForwards 01/03/05,
ACCAD Female1 B1 stand_to_walk + B3 walk1, HumanEva S2 Walking_3).
  A = human_smpl -> Greenwich (release codec) -> decode on fourier_gr3 -> golden global
      projection -> hinge angles; root position = data passthrough (fourier_gr3_root_world),
      root orientation = decoded pelvis.
  B = xwbc native GR3 retarget (raw npz, Isaac schema).
Same GR3 MJCF (GMR gr3v2_1_1 + freejoint), same metrics. Scripts:
`exp_topo/ctrl_verify/gr3_tracks.py`, `render_tracks.py`. Outputs (CSV in GMR/BeyondMimic
column order + npz in BeyondMimic schema, both tracks): `/var/tmp/alphamotion/ctrl_verify/tracks/`,
report `/var/tmp/alphamotion/ctrl_verify/report.json`, videos `R4_VIZ/mp4/ctrl_verify/*_A_vs_B.mp4`.

| metric (mean, 6 clips) | A Greenwich | B native |
|---|---|---|
| joint-limit violations | 0 | 0 |
| joint vel p95 (rad/s) | 2.28 | 1.81 |
| joint vel max (rad/s) | 14.8 | 6.7 |
| frames with a >8 rad/s spike | 7 / clip | 0 |
| jerk RMS (rad/s^3) | 451 | 208 |
| feet height difference, mean (cm) | 1.68 | 2.45 |
| lower-foot horizontal speed (skate proxy, cm/s) | 89 | 43 |
| root travel (m) | 3.17 | 3.17 (passthrough) |
| A vs B joint angle: mean / p95 | 8.4 deg / 26 deg | |
| A vs B body positions (cm) | 7.2 | |
| worst joints | wrist yaw (both), knee pitch, shoulder roll, waist yaw | |

Reading: format/limits are controller-ready (BeyondMimic schema written, zero limit
violations, root trajectory exact). Two real gaps vs the native retarget: (1) single-frame
velocity spikes (~7 frames/clip, up to 15-22 rad/s) — a tracking policy will either learn
to ignore or fight these; a 1-frame outlier filter on the *reference* is standard practice
in tracking pipelines (not a pose refiner); (2) stride/lift compression — the decoded gait
lifts the feet ~30% less while the root moves at the true speed, so the reference skates
~2x more than the native retarget. Both are visible in the side-by-side videos.

Note: xwbc native data itself has foot MESHES ~12 cm below z=0 (ground was aligned on body
origins) — common-mode, calibrated out in the metrics; a controller pipeline needs its own
ground alignment for both tracks.

Controller stack on this machine: /var/tmp/alphamotion/beyondmimic is actually GMT (G1-only,
MuJoCo sim2sim + pretrained G1 policy); BeyondMimic proper is at
config_harvest/whole_body_tracking (Isaac Lab, G1 config only). XHugWBC (arXiv 2602.05791):
sim robots = Booster K1/T1, Fourier N1, Unitree G1 23/29, Agibot X2, EngineAI PM01, MagicLab
Gen1, TianGong-1/2, Dobot Atom, Unitree H1-2, LeJu Kuavo; real = Booster T1, Fourier N1, G1
23/29, Agibot X2, Dobot Atom, H1-2. **Fourier GR3 is not supported; Fourier N1 is (sim+real).
Code: "coming soon" (not released).**

## Tracking policies (mjlab = BeyondMimic on MuJoCo-Warp; 4096 envs, 4000 iters, walk1, eval 1024 envs from start)

| | G1 AlphaMotion | G1 Native | GR3 AlphaMotion | GR3 Native |
|---|---|---|---|---|
| success rate | 98.0% | 99.7% | 99.9% | 99.9% |
| mpkpe (cm) | 15.8 | 12.3 | 17.5 | 16.4 |
| root-relative mpkpe (cm) | 7.0 | 5.2 | 8.5 | 7.5 |
| ee pos err (cm) | 9.6 | 6.2 | 11.9 | 10.0 |
| ee ori err (rad) | 0.155 | 0.108 | 0.136 | 0.100 |
| joint vel err (rad/s) | 0.97 | 0.68 | 0.99 | 0.70 |

Naming (lloyd 0818): AlphaMotion policy = reference via Greenwich; Native policy = reference via
xwbc native retarget. Repo: /home/arenalabs/alphamotion_ctrl (mjlab 1.6; GR3 robot cfg = GMR MJCF +
gains scaled from torque limits, first pass). Videos: R4_VIZ/mp4/ctrl_verify/{G1,GR3}_walk1_AlphaMotion_vs_Native_policy.mp4

## Unseen embodiment / unseen motion (human -> Greenwich -> robot vs GMR GT, 12 AMASS clips)

| body | status | rot | pos |
|---|---|---|---|
| unitree_g1_29dof | HOLDOUT | 22.1° | 8.0 cm |
| unitree_g1_23dof | HOLDOUT | 22.0° | 8.3 cm |
| pal_talos | HOLDOUT | 55.5° | 27.1 cm |
| fourier_gr3 / h1 / t1 / tienkung / n1 | seen | 14–21° | 3.6–7.6 cm |
| kuavo_s45 | seen | 32.6° | 19.4 cm |
Unseen motion corpus (bones seed, different actors/mocap) -> holdout G1 vs seed's own G1 retarget: 22.3° / 4.8 cm.

## Token-conditioned WBC (stage 1) + sparse tokens + fall/get-up (0818 night)

G1 walk1, 4000 it, eval 1024/512 envs.  Actor of the TOKEN policy sees only Greenwich latent window
(512-d pooled proj_out x 4 frames) + root anchor (root trajectory passthrough) + proprio; decoded
reference only in reward/critic.

| policy | tokens at eval | success | r_mpkpe | ee_pos |
|---|---|---|---|---|
| AlphaMotion ref-conditioned | (full ref) | 98.0% | 7.0 cm | 9.6 cm |
| AlphaMotion token-conditioned | full | 98.5% | 7.5 cm | 10.6 cm |
| token-conditioned, no aug | rand 25% / 50% / 75% hidden | 80% / 0% / 0% | | |
| token-conditioned, no aug | root+wrists+ankles only | 0% | | |
| token-conditioned + mask aug (8 variants) | full | 97.9% | 7.4 cm | 10.1 cm |
| same | rand 25% / 50% / 75% | 98.6% / 97.9% / 67% | | |
| same | root+wrists+ankles (+head) | 96.9% (98.6%) | 7.6 cm | 10.5 cm |

Greenwich's sparse completion (5 points -> decoded G1 ref within 10 deg / 2.7 cm of full) transfers
to the WBC only with token-mask augmentation during RL; then 5-point tokens drive the whole robot.

Fall/get-up (LAFAN fallAndGetUp1_subject1, 168 s): Native ref learns slowly (train mean ep len 331
steps, reward 15.6); AlphaMotion refs (ref- and token-conditioned) do not (~50 steps) — decoded root
pitch/roll while lying was 21 deg off (fixed by passing the root ORIENTATION through from data too),
but ~10% of frames still put bodies >5 cm below the floor: Greenwich cannot yet emit trackable ground
motion (LAFAN's 45 min of falls is all it saw). Robot state -> token fidelity while lying: 19 deg /
7 cm (upright 12 deg / 4.6 cm) — the closed-loop "robot fell -> token -> generator" signal is usable.

## X2 (zero-shot embodiment) experts vs one policy on all families (0819)

Token+mask-aug policies, X2 (never seen by Greenwich). Experts 4000 it; joint policy 6000 it on the
10-family concat (uniform RSI). Success = track the whole clip from start, 256-512 envs.

| family | expert | joint policy |
|---|---|---|
| walk | 95.1% (8.4 cm) | 82.4% (9.7 cm) |
| dance | 96.3% (4.8 cm) | 96.5% (5.6 cm) |
| kick | 87.7% (7.2 cm) | 0% |
| punch | — | 97.7% (3.7 cm) |
| throw | — | 97.3% |
| stand->skip | — | 87.1% |
| sit | — | 77.0% |
| run / jump / jacks | — | 0% |

Joint training matches experts on smooth families (dance/punch/throw) but collapses on the dynamic
tail (run/jump/jacks/kick) at this budget — the multi-task gradient starves the hard clips. Next
lever: per-clip adaptive sampling weights (mjlab's adaptive RSI is per-bin, currently uniform),
or 2-3x budget, or DAgger from experts (lloyd's original suggestion) for the tail only.

X2 fixes that made all of this work: gains 1.5x torque (0.7x sagged under gravity), data-driven
stance (X2 elbows are NEGATIVE), head collision off (mesh permanently inside torso), uniform RSI
on concatenated multi-clip references.

## Token-pool pretraining v1 (0819) — first real generalist number

Pool: 1500 AMASS windows x 150 f (no family filter, holdouts excluded), 373.5k reference frames,
3 token variants (full / rand-40% / 5-point sparse) fp16; teacher refs built by batch
encode->decode->project (minutes, scales to the full 13k-clip corpus). One policy, 4096 envs,
20k iters, uniform RSI. Exam = 8 UNSEEN motions on the UNSEEN body (X2), full tokens:

waltz 100% | GRAB pass-object 97.7% | backwards-walk 94.5% | walk->run 92.2% | squat 71.1% |
throw 72.7% | back-jump 0% | low-kick 0%   (mean 66%, r_mpkpe 15-21 cm)

Dynamic tail (jump/kick) starves under uniform sampling — same failure as allfam. Next levers:
dynamics-weighted sampling, budget, Equator-generated windows (Laplace-demon augmentation).
Gotcha fixed twice: root-trajectory passthrough is FIRST-FRAME-RELATIVE (z starts at 0) — every
consumer must ground references (lowest collision geom -> floor) exactly like the pool packer.
Videos: R4_VIZ/mp4/ctrl_verify/x2/X2_pool_generalist_unseen.mp4, X2_experts_walk_dance_kick.mp4.

### 0820 午 — lloyd 判定 locomotion 全失败(脚没迈出去)→ 根 xy 改为脚定(v4)
实测 v3 参考:walk 根走 1.32m 但支撑脚中位速度 0.16 m/s(根 0.27)=脚拖地;run 触地 p90 1.07 m/s。
根因:根 xy = 人类轨迹×臂展比 0.664,腿步幅 = 解码关节角 FK(≈人类×腿长比 0.782),两者不一致。
腿本身在摆(踝前后分开 29–38cm ≈ 人类 46cm×0.78),hip 无限位饱和 → 不是 Greenwich 的锅。
修:contact_root.py —— 支撑脚(人类数据判)的踝原点世界系钉住积分根 xy;飞行段人类根速度×0.782;z 最低点植地。
结果(8 hold):支撑脚速度 p50 0 / p90 ≤0.03 m/s;run 位移 2.41m(期望 2.44),walk 1.23m(期望 1.56)。
参考视频:R4_VIZ/mp4/ctrl_verify/x2/ref_feet_v3/。池子 v3 重建中(skate p90 ≈0.05),链 v4_chain.sh:
pack→x2_pool_v4 均匀 + x2_pool_cur4 课程→8 段考卷+视频(pool_v4/)。

### 0820 — RL 算法调研(PPO 成预训练瓶颈)
FlashSAC(RSS26, arXiv 2604.04539): SAC + 极低 UTD(2/1024)+大 batch 2048+2.5M 6 层网络+锁权重/特征/梯度范数+C51 critic;
Isaac Lab G1 平地 20min vs PPO 3h,崎岖 4h vs 20h;没测动作跟踪;官方仓库 404,镜像 wht-robot/FlashSAC。
FastSAC/FastTD3(2512.01996): 4090 15min 行走;BeyondMimic 风格整身跟踪 16384 env,FastSAC 长动作更好;代码 amazon-far/holosoma(Apache-2.0, MJWarp, PPO+FastSAC)。
RSL-RL-SAC(2605.24975): rsl_rl 官方 SAC,回报持平 PPO,wall-clock 无优势。LIFT(ICLR26): SAC 预训练+世界模型微调。EGM: 分桶自适应采样+MoE。
LLM 的 GRPO/DAPO/GSPO 是离散 token critic-free,不搬;可借"组内相对优势"做采样加权。
计划:holosoma FastSAC 训练器接 mjlab env → G1 walk1 对照 → X2 pool v4 同池对比 PPO → 再叠 FlashSAC 范数锁。

### 0820 晚 — 地平线算法(geodecity)代码落地  `alphamotion_ctrl/amctrl/horizon/`
按轮运行: 训 K 步(AM-Horizon 任务,前沿加权采样,每窗成功 EMA 落盘 mastery_live.json)→ 读掌握度取前沿(EMA≥0.7)
→ GPU worker(ardy 环境)用 Equator 生成邻居: homecoming(同端点、钉住 30–70% 自身 token、温度重采样内部)、
rainbow(U 尾 75 帧 + B3 桥 45 帧 + V 头 75 帧,U/V 按 A3 token 相似度配对)、retime(同 token 改时长 0.8/1.25)、
amass 真实邻居(缓存里未入池窗按码重叠度排序)→ realize(Greenwich 解到 X2 + 全局投影 + 脚定根;支撑脚由同一组码解到人类骨架判)
→ gate(支撑脚 p90<0.10 m/s、穿地<1.5cm、关节跳变<0.45 rad/帧、限位占比<25%、末端跳变<12cm、倾倒<70°、B3 似然比≥0.45)
→ 入池 → pack(含 window_ids)→ --agent.resume 续训。根流(码 128:256)A3 不能再生,从父窗时间重采样搬运,由门把关。
文件: store.py(清单/码/实现/掌握度) neighbors.py(码重叠) generate.py(Demon 三算子) realize.py gate.py pack.py gpu_worker.py;
token/curriculum.py 新增 HorizonTokenMotionCommand;scripts/horizon_round.py(一轮) scripts/horizon_loop.sh(循环)。
测试轮(10 个走路窗假装已掌握): 第一版 12/12 被门拒 → 支撑脚改为**继承父窗真值**(只推断 rainbow 的桥段)、
rainbow 根流改为 FSQ 网格上 U 末帧→V 首帧渐变(硬切在桥中点跳 0.4m)→ homecoming 4/4、retime 1/2、AMASS 邻居 3/3、rainbow 4/6 过门;
门拒掉的正是该拒的(关节跳 2.44 rad/帧、似然比 0.06)。Horizon 训练任务冒烟通过,掌握度文件落盘。
视频: R4_VIZ/mp4/ctrl_verify/x2/horizon_test/{homecoming_00000,rainbow_00001,retime_00000}.mp4

### 0820 夜 — v5 10k 判定 + 真正的地平线循环开跑(hz1)
v5(全池均匀,0.3m 全 3D 终止)5k/8.5k/10k in-pool walk 位移比 0.43/0.45/0.42,平线 → lloyd 判定:课程错误,不是阈值;停 v5/cur5/v5_ref。
EMA 掌握度按 lloyd 要求废除 → scripts/horizon_exam.py 每轮真考(每窗从头跑 16 遍,成功且位移比≥0.8=掌握)。
store 加 seed 模式(--seed-ids,其余 1499 窗做储备候选,按码重叠度排队)。
邻居 sim 验证(父窗 expert 零样本):父窗 70%/位移比 0.99;homecoming 跟半程 0.84;rainbow 0.38m;retime 0.16m。
hz1:seed=amass_00001 expert(2000 步,考试 100%/0.98),12 轮 × 1500 步 × 40 新窗。logs/horizon_loop_hz1.log。
**0820 深夜 — 地平线致命 bug**:Greenwich 旋转头完全不看位置流(只换位置码,rot diff 0.0000)→ 旋转流从父窗搬运的生成窗,
解到 X2 的关节角和父窗逐位相同(token 改 70%、参考没变=教策略忽略位置流)。位置头直接 IK 翻支 5 rad 不可用。
修:在人体骨架上"位置流给摆动、父窗旋转流给扭转"合成新旋转,5 帧平滑 + 摆动限 25°,Greenwich 重编码让两流一致,
再金标准投影(realize.reencode_swing_twist)。自回环 X2 关节只变 0.038 rad;生成窗相对父窗均值 0.13 rad、关节跳 0.5、过门。
循环 resume 参数要写 --agent.resume True。hz1 重置重跑。

### 0821 凌晨 — hz1 对照组判决 + hz2 开跑
hz1(单走路起点,actor 512-256-128,每轮 1500 步,40 新窗):掌握 22→39→70→96→111→141→168→173→191→**181**(r10,池 340);
r8 起 lost≥gained(13/9,15/18,**32/12**)= 小网络在 300 窗上边学边忘。跨家族零样本 1/15(只过 8 字走)。停在 r10,省算力给 hz2。
hz2:8 家族 seed(walk/run/jump/kick/sit/dance/throw/other,按动力学特征挑窗)从零训 4000 步 → 真考 → 循环(每轮 2000 步);
actor/critic 1024-1024-512(~3.6M,仍是 MLP,transformer 留到下一步);任务 AM-HorizonL / AM-TokenMaskL。
0821 — hz2 seed 纠错(lloyd 指出"深蹲怕是坐下"):sit seed amass_01191 实为坐到地上(根 0.67→0.18m、髋在脚后 27cm、不再站起),
kick seed amass_00027 脚抬 0.9m > X2 腿长 0.6m,均物理不可行 → 换 amass_00260(HDM05 深蹲 32cm、髋在脚上、站回)和 amass_00492(脚 0.34m)。
gate 加 unsupported_frac(根 z<0.45 且髋离脚中点>0.2m 的帧 >10% 拒);储备窗入池也过 gate(之前跳过了)。
hz2 r2: 45/90,丢 1 学 8(大网络不忘);走/跑/投/舞 seed 掌握;跳/踢(旧)/蹲(旧) 0%。

### 0821 — hz2 12 轮跑完(8 家族起点 + 1024-1024-512 + seed 优先 25× + seed 门)
seed 门在 r4–r10、r12 挡住扩张(起点未到 8/9 不外推),池子停在 133 专心补课;起点掌握 4/9(r3)→8/9(r10):
深蹲 32cm 100%、高踢 0.9m 94%、低踢 100%、舞、投、走、跑、跷跷板全过;只剩跳(88% 成功/位移比 0.71)。
r11 扩到 171,r12 106/171 掌握;真实窗按家族:other 9/12、投 6/9、踢 4/8、走 3/5、蹲 3/7、跑 3/8、舞 2/8、**跳 0/8**。
丢/学每轮 5–13/5–15,大网络不再单边遗忘。视频 horizon_hz2_r1..r12/。
expert 对照:跳 95%、高踢 100%(2k 步单窗)→ generalist 早期 0% 是稀释不是不可学;坐地 0%(不可能,已剔除)。

### 0821 晚 — hz3:HumanoidGPT 路线(transformer actor)
amctrl/rl/token_transformer.py:TokenTransformerModel(rsl_rl MLPModel 子类):4 帧 token(512→256)+本体 token+[ACT] 读出,
pre-LN 4 层 4 头 d=256,读出→MLP(256)→高斯头;3.4M 参数;自带 JIT/ONNX 包装(每次存 ckpt 会导出,不接就崩)。
任务 AM-HorizonT / AM-TokenMaskT(critic 仍 1024-1024-512 MLP)。hz2 终考 holdout 8 段仅跳 78%/踢 34%/蹲 25%,其余 0 → lloyd 判 MLP 容量到顶。
hz3:hz2 的 9 个 seed 重新从零(6000 步)→ 循环 12 轮 × 2000 步 × 40 窗。

### 0821 夜 — hz3sac:FlashSAC vs PPO 对照(lloyd 提议)
scripts/train_flashsac.py:自写精简 FlashSAC 接 mjlab/rsl_rl runner(同一个 TokenTransformer actor,ckpt 格式不变,真考/视频/续训复用):
4096 env 每步 8 次更新 batch 4096、回放 1M(fp16 ~18GB)、C51 双 critic 取平均、LayerNorm+权重行单位范数+梯度裁剪、
回报按运行 std 自适应缩放、固定目标熵 σ=0.15 自动调 α、5 步回报、非对称 critic(特权观测)、超时 bootstrap(近似)。
未实现:Zeta 噪声重复、2.5M 倒残差网络。hz3sac:与 hz3 相同 9 起点池,seed 20000 env 步 → 真考 → 循环(每轮 5000 步)。
hz3(PPO) seed 6000 迭代(=144k env 步×4096)4.5h:奖励 23/回合 380(hz2 MLP 同期 ~10/190)。
hz3sac 结果(0822):v1 无界动作 2 分钟发散;v2 学习 std 自己涨到 0.56、α 塌 0;v3 固定 σ 但 tanh 饱和(熵 -39);v4(软界 ±2 + u² 罚)稳定:
4000 步回合 127,之后 4000–20000 步平线 ~120,终考 **2/9**(PPO+transformer 同 seed 144k env 步 8/9;SAC 20k env 步)。
等样本量(~10k env 步)两者回合长度相当(136 vs 122),之后 PPO 继续爬、SAC 陷"站住"局部解。停 hz3sac,hz3 独占 GPU。
hz3 r1–r3(transformer):26/39 → 29/39(补课轮,起点 8/9)→ 51/76(r3,跳 seed 100%/0.85 首次掌握,9 起点全部掌握过);r4 扩到 114。

## 0828 舞蹈单窗三连败（FS 配方边界测定）
LAFAN dance 单窗（150f，贴地40%，1.61步/s）× BMFS 配方（走路窗全绿的同款）：
v1 地板舞（LAFAN spec 跨缓存误用致选段错误，rest 差 29cm）；v2 站立舞 3000 迭代存活平台 66/249；
v3 +双脚着地出生 1000 迭代平台 24 步/回合，无起飞形态 → 止损。
**判决：单窗从零暴力不是高动态家族的武器；舞蹈归地平线课程（从掌握邻域渐进）——与 hz3 dance 种子始终未过（0.69）一致。**
