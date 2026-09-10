# BFM-Zero 训练配方（代码实锤版）

来源：`/home/arenalabs/reference_code/bfm_zero`（LeCAR-Lab 官方训练代码）。
算法是 FB-CPR（Meta motivo 那套）+ 一个额外的惩罚 critic。下面只讲"怎么训出来的"，不讲架构。

真正生效的配置在 `humanoidverse/train.py:594-724` 的 `train_bfm_zero()` 里，**yaml 里的很多数会被这里覆盖**。看数字以 train.py 为准。

---

## 1. 惩罚表：怎么加、加多少

**关键机制：惩罚不进 FB 奖励，走一个独立的 aux critic。**

- 环境把每项惩罚的**原始值**（不乘权重）塞进 `extras["aux_rewards"]`（`envs/legged_base_task/legged_robot_base.py:569`）。
- agent 自己拿权重加权求和（`agents/fb_cpr_aux/agent.py:156-165`），权重在 `train.py:665`。
- 加权和先过一个**只除标准差、不减均值**的归一化器（`train.py:626`，translate=False, scale=True）。
- 然后训练独立的 aux critic（`fb_cpr_aux/agent.py:212-251`）。
- actor 损失（`fb_cpr_aux/agent.py:277-282`）：

```
actor_loss = -Q_fb - 0.05·|Q_fb|·Q_discriminator - 0.02·|Q_fb|·Q_aux
```

reg_coeff=0.05（判别器项）、reg_coeff_aux=0.02（惩罚项），且都乘 `|Q_fb|` 的均值（scale_reg=True）。**惩罚力度自动跟着任务 Q 的量级走**，不用手调平衡。

实际生效的权重（`train.py:665`）：

| 项 | 权重 | 公式（原始值） | 位置 |
|---|---|---|---|
| penalty_slippage（脚滑） | **-2.0** | Σ‖脚线速度‖·(接触力>1N) | base:825-828 |
| penalty_ankle_roll | **-4.0** | 左右 ankle_roll 角² 之和 | motions:580-585 |
| limits_dof_pos | **-10.0** | 超软限位(0.95)的量，线性 | base:795-807 |
| penalty_feet_ori（脚掌不平） | **-0.4** | 触地脚的重力投影 xy 范数 | base:863-869 |
| penalty_undesired_contact | **-1.0** | 非脚部位接触=1，否则0 | motions:571-578 |
| penalty_action_rate | **-0.1** | Σ(Δaction)² | base:785-787 |
| penalty_torques | **0.0（关了）** | Σtorque² | base:773 |
| limits_torque | **0.0（关了）** | 超软扭矩限位量 | base:818 |

注意：`config/rewards/reward_bfm_zero.yaml` 里那套权重（slippage -1.0、action_rate -0.5 等）只进环境的标量 reward，**FB 训练根本不用那个标量**，也不用 yaml 里的 penalty curriculum。别被 yaml 骗了。

**vs 我们**：我们 foot_slide -3.0、joint_limit -10 直接摞进 PPO 奖励，跟任务奖励硬碰硬。他们脚滑 -2 但走独立 critic + |Q| 自适应缩放，等效强度是"任务价值的 2% 量级"，永远压不死任务项。joint_limit -10 两家一致。他们独有：ankle_roll -4（实际最重的一项）、脚掌姿态、扭矩惩罚干脆置零。

## 2. 终止 / 重置

**没有任何失败终止。全关。**

- `config/env/legged_motions.yaml:11-19`：contact/gravity/low_height/motion_far/limit 终止全 False。
- `config/exp/bfm_zero/bfm_zero.yaml:36-42`：terminate_when_motion_far 也是 False。
- 只有超时截断：`max_episode_length_s: 10`（legged_motions.yaml:10），50Hz 即 500 步。
- 摔了不重置，接着滚，让 FB/判别器把它拉回数据流形——起身能力就是这么白捡的。
- （代码里留了 motion_far 的口子：被推后 2 秒内免终止，motions:148；但整个开关是关的。）

**vs 我们**：我们 anchor_pos 0.3m / ee 0.25m 终止。他们零终止 + 10 秒定长。这是两家最大的哲学差异：我们把"跟丢"当失败剪掉，他们把"跟丢"当训练素材。

## 3. 动作数据采样

- 数据：**LAFAN1**（不是 AMASS），切成 10 秒小段：`lafan_29dof_10s-clipped.pkl`（train.py:675），max_len 450 帧（bfm_zero.yaml:25）。
- RSI：每次重置随机挑一条 motion + 均匀随机相位起点（motions.py:162-171；motion_lib_base.py:573-582）。
- **失败加权采样（EMD 优先级）**：每 960 万步跑一次全量 tracking 评测，拿每条 motion 的 EMD 当优先级：`clamp(EMD, 0.5, 2.0) × 2`，再取 `2^p`（exp 模式）→ 权重范围 2~16，**难的动作最多比简单的多采 8 倍**（train.py:350-379, 96-99, 703-707）。权重同时更新两处：环境的 RSI 选 motion 概率（motion_lib_base.py:140-170）+ expert buffer 的采样概率（train.py:391-394）。
- 没有难度课程，就这一个 EMD 反馈环。
- 另外 50% 的环境在 rollout 时直接执行"专家轨迹编码的 z 序列"（每段 250 步 = 5 秒，fb/agent.py:354-379，train.py:648-650）：把专家未来 8 帧窗口的 B 编码滑动平均当 z，逼着策略在自己的物理里复现专家轨迹。

**vs 我们**：我们均匀采样。他们的 EMD 加权 = 我们一直想要的"哪里差练哪里"，而且是拿真评测指标喂回去，不是拿训练 loss。

## 4. 初始状态随机化 / 摔倒训练

- RSI 噪声：**没有**。`noise_to_initial_level: 0`（legged_motions.yaml:44），所以 dof_pos 0.1 / root_rot 10° 那些 init_noise_scale 全部乘零。跟我们一样干净。
- 但每回合随机化 default_dof_pos 偏置 ±0.02 rad（domain_rand.yaml:46-47，抄 BeyondMimic）。
- **躺倒初始化（起身的全部秘密）**：`lie_down_init=True, prob=0.3`（train.py:683）。实现在 motions.py:466-491：30% 的重置把 root 高度设 0.5m，绕 X 轴转 ±90°（仰躺或俯卧，正负号每批随机一个），关节姿势还是 motion 里采的那帧。**没有专门的起身奖励、没有起身数据**——躺倒起点 + 零终止 + 判别器拉回流形，起身自己涌现。

**vs 我们**：我们没有摔倒态训练。他们的配方就三行改动：30% 躺倒重置、不终止、剩下交给模仿目标。

## 5. 域随机化（`config/domain_rand/domain_rand.yaml`）

| 项 | 开关 | 范围 |
|---|---|---|
| 推机器人 | ✅ | 每 1~3 秒推一次，xy 线速度 ±0.5 m/s，角速度 ±0.5；推完 2 秒"免死"窗口 |
| 摩擦 | ✅ | **0.5 ~ 1.25** |
| base 质心偏移 | ✅ | xyz 各 ±0.02 m |
| link 质量 | ✅ | ×0.95 ~ 1.05 |
| default_dof_pos 偏置 | ✅ | ±0.02 rad |
| PD 增益 | ❌ | （0.75~1.25 备着没开） |
| base 质量 | ❌ | |
| 扭矩 RFI | ❌ | |
| 控制延迟 | ❌ | （0~1 步备着没开） |

观测噪声（`config/obs/bfm_zero_obs.yaml:42-51`）：ang_vel 0.2、projected_gravity 0.05、dof_pos 0.01、dof_vel 0.5，action/特权观测无噪声。

**vs 我们**：摩擦他们 0.5~1.25 比我们 0.7~1.2 宽，尤其下探到 0.5（滑地面）。最狠的是推搡频率：**平均每 2 秒一推**，整个 10 秒回合被推 3~5 次。而且真机部署居然不开 PD 随机和延迟随机——他们赌的是 aux 惩罚 + 高频推搡就够了。

## 6. 优化器 / 算法超参（`train.py:608-724`）

规模与节奏：
- num_envs **1024**，总步数 **3.84 亿** env steps；README 说 0.5~1 亿步后 eval/emd < 0.75。
- 回放 buffer **512 万** transition（轨迹式 buffer，GPU 上），expert buffer 全量 LAFAN 参考帧（纯运动学，动作填零，humanoidverse_isaac.py:80）。
- 前 10240 步纯随机动作探索；之后每 1024 env steps 更新 16 次 = **每 64 个环境步 1 次梯度更新**。
- batch 1024（其中 seq_length 8 用于 z 编码），每次更新同时采 expert batch 1024。

学习率与损失：
- lr：F / actor / critic / aux_critic = **3e-4**；B = **1e-5**；判别器 = **1e-5**。
- discount **0.98**；target 软更新 FB τ=0.01、critic τ=0.005。
- FB 正交惩罚 ortho_coef **100**；判别器 WGAN-GP 梯度惩罚 **10**。
- 悲观系数（双 critic 差值惩罚）actor/critic/aux 全是 **0.5**；q_loss_coef 0（关）。
- 探索噪声 actor_std **0.05**（很小！），采样截断 stddev_clip 0.3。

z 的玩法：
- z_dim 256，单位球归一化。rollout 中每 **100 步（2 秒）换一次 z**。
- 训练 batch 里 80% 的 transition 重标 z（relabel_ratio 0.8）；重标分布 = **60% 专家轨迹编码 + 20% buffer 里随机 next_obs 当 goal 编码 + 20% 均匀随机**（expert_asm_ratio 0.6 / train_goal_ratio 0.2，fb_cpr/agent.py:126-152）。
- rollout 用的 z 从最近 8192 个训练 z 里采（z_buffer，而不是纯随机）。

控制栈：200Hz 物理 / decimation 4 = **50Hz 控制**（isaacsim.yaml:15-16）；action_scale 0.25，action clip ±5；机器人用 **hard_waist** 版 G1（腰三轴 kp=300、kd=5，g1_29dof_hard_waist.yaml:207-226），观测归一化用 BatchNorm（momentum 0.01），不用 running mean。

**vs 我们**：他们 1024 envs（off-policy 回放型，不需要 4096）；我们 4096 envs on-policy PPO。discount 0.98 vs 我们的 PPO γ。他们探索噪声极小（0.05），多样性靠换 z 而不是靠动作噪声。

## 7. 流水线阶段

**就一个阶段。** 在线无监督预训练 384M 步一杆到底，没有 finetune、没有蒸馏、没有 RL 二阶段。部署时全靠推理期算 z（tracking/goal/reward 三种 prompt 推理，README），策略网络一个参数不动，直接导 ONNX 上真机。评测每 960 万步一次（顺带喂 EMD 优先级），checkpoint 同频。README 另提了 MJLab 分布式版 6 小时训完。

**vs 我们**：我们的"预训 codec → 翻译 → WBC/RL"多段管线 vs 他们单段端到端。他们能单段是因为惩罚走独立 critic、目标全在 z 里，不存在阶段间接口对不上的问题。

## 8. 特别的招

1. **惩罚归一化只除方差不减均值**（train.py:626）——保留"惩罚永远是负的"这个符号信息，只把量级压稳。
2. **|Q_fb| 自适应正则**（scale_reg，fb_cpr_aux:277）——模仿项和惩罚项的力度锚在任务价值量级上，训练全程比例恒定。
3. **零终止 + 定长 10 秒回合**——terminated 恒为 False，bootstrap 永不截断（fb_cpr_aux:92 的 discount 掩码形同虚设）；所有回合等长，buffer 均匀。
4. **专家数据纯运动学**——不需要专家动作，判别器只看 (状态, z) 对；所以任何 retarget 结果都能当 expert。
5. **torque 惩罚挂零**——列表里留着、权重 0.0，说明他们试过并放弃了；限扭靠 clip_torques=True 硬剪。
6. **推搡后 2 秒免死窗口**（base:371）——虽然终止关了用不上，但这个设计值得记：DR 扰动不该被终止条件惩罚。
7. **env 标量 reward 全程算了但没人用**——工程上宁可白算也不改环境接口。

---

## 可移植改动 Top-5（按性价比排序）

1. **躺倒初始化 30% + 摔倒不终止**（motions.py:466-491）。三行改动换一个起身技能 + 摔后恢复的分布覆盖。对我们"恢复必须来自潜空间搜索"的路线，这是最便宜的数据来源：先让策略见过躺倒态。
2. **EMD/跟踪误差加权的 motion 采样**（train.py:350-379）：每次评测把每条 motion 的真实跟踪误差 clamp(·,0.5,2)×2 再 2^ 化成采样权重（最差:最好 = 8:1）。我们有现成的评测管线，接上就能用。注意他们用真评测指标，不用训练期 loss——符合我们的对照臂纪律。
3. **惩罚项与任务项解耦 + 量级自适应**：哪怕不换算法，也可以把惩罚和做成独立 value head，并把惩罚系数乘上任务优势的滑动量级。解决我们 -3.0 脚滑和任务奖励打架、每换任务重调权重的老问题。
4. **高频推搡 DR**：每 1~3 秒推一次 ±0.5 m/s + 推后 2 秒免终止豁免。我们如果保留终止条件，豁免窗口是必配件，否则推搡=送终止。摩擦下界从 0.7 放到 0.5。
5. **ankle_roll² 重罚（-4）+ 触地脚掌姿态罚（-0.4），撤掉能量/扭矩罚**：他们真机可行性三件套是 脚滑-2 / 踝滚-4 / 限位-10，扭矩和能量都是 0。我们脚部审计（foot_audit）抓的滑冰问题，他们是拿踝部姿态罚兜住的，值得对照实验一臂。
