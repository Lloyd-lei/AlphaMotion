# BeyondMimic 原版训练配方（对照我们的 X2 mjlab 移植版）

来源：`/home/arenalabs/reference_code/beyondmimic`（HybridRobotics/whole_body_tracking，33 个 py 文件，全部读过）。
下面所有行号都指这个 repo。核心配置文件：
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`（主配置）
- `.../tasks/tracking/mdp/commands.py`（采样 + RSI）
- `.../tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`（PPO）

一句话总结：**原版的配方非常"素"。没有任何脚部奖励。它靠三样东西站住：自适应起点采样、很重的 RSI 噪声、贴身的终止条件。我们的移植版把后两样丢了/改了，然后用脚部奖励去补。**

---

## 1. 奖励表（完整）

出处：`tracking_env_cfg.py:198-250`，函数在 `mdp/rewards.py`。

| 项 | 权重 | std | 备注 |
|---|---|---|---|
| motion_global_anchor_pos | **0.5** | 0.3 | 全局锚点(torso)位置，exp 核 |
| motion_global_anchor_ori | **0.5** | 0.4 | 全局锚点朝向 |
| motion_body_pos | 1.0 | 0.3 | **相对系**：14 个 body，对齐机器人 xy+yaw 后再比（commands.py:284-294） |
| motion_body_ori | 1.0 | 0.4 | 同上，相对系 |
| motion_body_lin_vel | 1.0 | 1.0 | 全局系速度 |
| motion_body_ang_vel | 1.0 | 3.14 | 全局系角速度 |
| action_rate_l2 | -0.1 | — | |
| joint_limit | -10.0 | — | 软限位系数 0.9（g1.py:soft_joint_pos_limit_factor） |
| undesired_contacts | **-0.1** | 阈值 1.0N | 除双踝双腕外任何 body 触地都罚（tracking_env_cfg.py:238-250） |

**脚部项：一个都没有。** `rewards.py:77-82` 有个 `feet_contact_time` 函数，但全 repo 没人用它（grep 确认只有定义）。没有 contact match、没有 slide 罚、没有 swing 项。

所有 exp 核都是 `exp(-误差²均值/std²)`，误差先平方再对 body 取平均（rewards.py:40）。没有奖励归一化。

**vs 我们**：我们 body pos/ori/lin/ang 也是 1.0，一致。但我们 global_root_pos 给 1.0，原版只给 **0.5**——原版故意把全局项压低，让相对系动作项主导，全局漂移主要靠终止条件兜底。我们额外加的 foot_contact_match +1.5 / foot_slide -3.0 / swing_refusal -1.5 是纯自研，原版没有。我们没有 undesired_contacts 的等价物（我们 self_collisions -10 罚的是自碰，不是"身体别的部位碰地"）。

**值得抄？** 全局项降回 0.5：可试（Y，低风险）。undesired_contacts -0.1：Y，便宜且防"用膝盖爬"。脚部项去掉：N，别急着删——原版不滑是因为别的三样东西在（见下），先补齐那些再谈减脚部项。

---

## 2. 终止条件

出处：`tracking_env_cfg.py:253-278`，函数在 `mdp/terminations.py`。

| 项 | 函数 | 阈值 | 关键细节 |
|---|---|---|---|
| time_out | time_out | 10s | time_out=True → rsl_rl 会做超时 bootstrap |
| anchor_pos | **bad_anchor_pos_z_only** | 0.25m | **只看 z 高度差**（terminations.py:23-25），xy 漂移不杀 |
| anchor_ori | bad_anchor_ori | 0.8 | **不是四元数角度**：比较重力向量在两个锚点系里的 z 分量之差（terminations.py:28-38），本质是"倾倒程度差" |
| ee_body_pos | **bad_motion_body_pos_z_only** | 0.25m | 双踝+双腕，**也只看 z**（terminations.py:51-58） |

阈值全是静态写死的，没有课程/自适应。注意 repo 里也有全 3D 版本 `bad_anchor_pos` 和 `bad_motion_body_pos`（terminations.py:18, 41），但 G1 配置用的是 z_only 版。

**vs 我们**：我们 anchor_pos 0.3m、ee_body_pos 0.25m——如果我们是 3D norm（大概率是），那我们对 xy 漂移**更严**，对高度反而更松。原版哲学：xy/yaw 漂移交给 0.5 权重的全局奖励慢慢拉，不当死刑；z 塌了（摔）才杀。这跟我们"位移=数据轨迹透传、连续性优先"的取向其实同路。

**值得抄？** Y。z_only 终止 + 重力投影朝向，两个都值得试。3D 终止会把"轨迹平移了但动作是对的"的 episode 错杀，污染自适应采样的失败统计。

---

## 3. 动作起点采样（这是原版最独特的东西）

出处：`mdp/commands.py:207-241`（_adaptive_sampling）、`commands.py:80-86`（初始化）、`commands.py:296-299`（EMA 更新）。

机制，白话版：
1. 把动作按 1 秒切成 bins（`bin_count = 总帧数/50 + 1`，commands.py:80）。
2. 每次有 env **非超时失败**，记下它死在哪个 bin（commands.py:209-214）。
3. 失败计数走 EMA：`bin_failed = 0.001 * 本轮失败 + 0.999 * 历史`（commands.py:296-298，adaptive_alpha=0.001）。
4. 采样概率 = 失败计数 + **10% 均匀底**（adaptive_uniform_ratio=0.1，commands.py:217）。
5. 可选一个非因果卷积核把失败往"前面几个 bin"扩散（λ=0.8 衰减，但默认 kernel_size=1 = 不扩散，commands.py:368-369，G1 没改）。
6. 按概率抽 bin，bin 内再均匀抽帧（commands.py:227-233）。
7. 还往 wandb 记采样熵/top1 概率/top1 位置（commands.py:236-241）——他们把采样分布当健康指标盯。

episode 长度政策：episode 固定 10s；动作播完了**不重置 env，直接就地重采一个新起点并瞬移**（commands.py:279-282，resampling_time_range=1e9 表示永不按时间重采）。另外 train.py:201 `init_at_random_ep_len=True`，开局打散重置节奏。

**vs 我们**：如果我们是均匀采样起点，这是最大的一块缺失。这个机制等于免费的自动课程：哪里摔就多练哪里，10% 均匀底保证不忘。而且它只统计 terminated（不含 timeout），所以终止条件的质量直接决定这个课程的质量——第 2 节和第 3 节是绑在一起的。

**值得抄？** **Y，第一优先。** 实现小（~30 行），无超参地狱（α=0.001、uniform=0.1 抄来就行）。

---

## 4. RSI 重置噪声（我们丢掉的第二样）

出处：`tracking_env_cfg.py:85-99`（配置），`commands.py:243-277`（施加）。

每次重采起点，参考帧状态加噪后写进仿真：

- 根位置：x ±0.05m，y ±0.05m，z ±0.01m
- 根姿态：roll ±0.1，pitch ±0.1，yaw ±0.2 rad
- 根速度：xy ±0.5 m/s，z ±0.2 m/s，roll/pitch ±0.52，yaw ±0.78 rad/s（VELOCITY_RANGE，tracking_env_cfg.py:30-37）
- 关节角：**全关节 ±0.1 rad**，再按软限位裁剪（tracking_env_cfg.py:98，commands.py:265-272）。（cfg 默认值是 ±0.52，commands.py:366，但主配置覆盖成 ±0.1）

**vs 我们**：我们 pose_range 是空的 = 每次都从精确参考态出发。这就是任务书里说的"SUSPICIOUS"，坐实了：原版从不精确重置。没有 RSI 噪声，策略从没见过"我偏了一点"的状态，只能靠推搡事件补，恢复能力和 sim2real 都吃亏。速度噪声 ±0.5 m/s 相当大——策略开局就在"被踹了半脚"的状态里练。

**值得抄？** **Y，第二优先。** 数值照抄即可。

---

## 5. 域随机化

出处：`tracking_env_cfg.py:153-195`（EventCfg），`mdp/events.py:73-151`。

| 项 | 时机 | 范围 |
|---|---|---|
| 地面/body 摩擦 | startup | 静摩擦 0.3–1.6，动摩擦 0.3–1.2，弹性 0–0.5，64 buckets |
| 关节零位标定误差 | startup | 全关节 add ±0.01 rad（自写函数，events.py:73，会同步改 action 的 offset，events.py:110） |
| torso 质心偏移 | startup | x ±0.025，y ±0.05，z ±0.05 m（events.py:113） |
| push_robot | 每 1–3s | 直接设根速度，范围 = VELOCITY_RANGE（同 RSI 速度噪声） |

没有的：质量缩放、PD 增益随机化、动作延迟（`robots/actuator.py` 写了个 DelayedImplicitActuator，但 G1 配置没用它）、地形随机。清单很克制。

**vs 我们**：push_robot 1-3s 我们有，一致。摩擦下限 0.3 比常见配置低（配合我们溜冰门的 0.7 摩擦修复要小心，别照抄下限）。关节零位误差和 COM 偏移如果我们没有，是两个便宜的 sim2real 项。

**值得抄？** 零位误差 + COM 偏移：Y。摩擦范围：看部署目标，N（先保 0.7 附近）。

---

## 6. PPO 超参数

出处：`config/g1/agents/rsl_rl_ppo_cfg.py:5-31`。

- 4096 envs（tracking_env_cfg.py:298），24 steps/env，5 epochs，4 minibatches
- lr 1e-3，schedule=adaptive，desired_kl 0.01
- gamma 0.99，lam 0.95，clip 0.2，entropy_coef 0.005，value_loss_coef 1.0，clipped value loss，max_grad_norm 1.0
- init_noise_std 1.0，MLP [512,256,128] actor 和 critic，elu
- **empirical_normalization = True**（G1，rsl_rl_ppo_cfg.py:11；humanoid 版是 False）
- max_iterations 30000，save 500
- 非对称 critic：critic 用无噪声特权观测，多 body_pos/body_ori（tracking_env_cfg.py:135-146）
- 控制 50Hz（dt 0.005 × decimation 4，tracking_env_cfg.py:312-315）
- 还有个 Low-Freq 变体：decimation 翻倍时 gamma/lam 按 `x^(1/0.5)` 折算、steps 减半（rsl_rl_ppo_cfg.py:34-43）——频率变了折扣率跟着变，细节讲究。

**vs 我们**：核心数字全部一致（4096/1e-3 adaptive/24/5/4）——我们的移植没丢东西。要核对的只有两个：empirical_normalization 是否开了；critic 是否吃特权观测。

**值得抄？** 数字不用动。查 obs 归一化和非对称 critic 这两个开关：Y。

---

## 7. 蒸馏阶段

**不在这个 repo。** 本 repo 只有 tracking 训练；README:16-24 说 guided-diffusion 控制器是 BeyondMimic 的另一部分，部署在 `motion_tracking_controller` repo。这里连 rollout 收集脚本都没有（grep "distill/diffusion/student" 只命中 README 一行）。想挖学生目标得去抓他们别的 repo / 论文（arxiv 2508.08241），本次任务范围内无法给出数字。

---

## 8. 杂项但重要

- **观测噪声**（tracking_env_cfg.py:118-133）：actor 观测全部加均匀噪声——motion_anchor_pos ±0.25（很大！），anchor_ori ±0.05，lin_vel ±0.5，ang_vel ±0.2，joint_pos ±0.01，joint_vel ±0.5。critic 无噪声。
- **超时 bootstrap**：time_out=True + rsl_rl 标准处理，超时不当失败算 value。自适应采样也只统计 terminated 不统计 timeout（commands.py:208）。
- **奖励无归一化**，观测有（empirical_normalization）。
- **动作缩放**：每关节 `0.25 × 力矩上限 / 刚度`（g1.py:184-195）；刚度本身从电机 armature 按 10Hz 自然频率、阻尼比 2.0 推出来（g1.py:7-24）。不是拍脑袋的统一 0.25。
- **相对系奖励的对齐方式**（commands.py:284-294）：比 body pos/ori 前，把参考整体搬到机器人当前 xy + yaw，但高度用参考的。所以"跟上动作"和"站在正确的世界位置"是两笔账，后者只有 0.5 权重的全局项 + z 终止在管。
- 观测里含参考的完整 joint_pos+joint_vel（commands.py:100-102），以及 anchor 相对位姿。

---

## 移植优先级 Top-5

1. **自适应起点采样**（§3）。失败加权 bins + EMA 0.001 + 10% 均匀底。原版的核心创新，实现小，直接治"长动作难段练不够"。
2. **RSI 重置噪声**（§4）。我们 pose_range 为空是真缺陷。照抄：pos ±0.05/±0.05/±0.01，rpy ±0.1/±0.1/±0.2，vel ±0.5/±0.2/±0.52/±0.78，关节 ±0.1。
3. **终止改 z_only + 重力投影朝向**（§2）。别用 3D norm 杀 xy 漂移；配合 #1 效果相乘（失败统计不被平移错杀污染）。
4. **undesired_contacts -0.1**（§1）+ 顺手核对 empirical_normalization 和非对称 critic 开关（§6）。都便宜。
5. **actor 观测加噪**（§8），尤其 anchor_pos ±0.25 这种大噪声 + critic 干净的组合；再加零位误差 ±0.01 和 torso COM 偏移（§5）。

注意事项：#1–#3 是一套联动系统（松终止喂准失败统计，采样把难段喂给带 RSI 噪声的重置），最好一起上或按 3→2→1 顺序上，逐个用对照臂验证（一次只动一个）。
