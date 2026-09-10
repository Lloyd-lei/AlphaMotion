# SONIC 训练配方挖掘报告

来源：`/home/arenalabs/reference_code/sonic`（NVIDIA GR00T-WholeBodyControl，gear_sonic）
入口配置：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
只看学习配置，不看架构。所有数字都从 yaml + py 源码里核对过，带文件行号。

一句话总评：SONIC 的钱花在三个地方——**按失败率采样数据、起点加噪、松绑终止条件**。奖励表本身平平无奇，跟我们差别不大。

---

## 1. 奖励表（完整）

release 用的组合：`config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml`，各项定义在 `config/manager_env/rewards/terms/*.yaml`，函数在 `envs/manager_env/mdp/rewards.py`。所有 tracking 项都是 `exp(-误差²/std²)`，误差是多身体求均值。

| 项 | 权重 | std/参数 | 函数（rewards.py 行号） | 说明 |
|---|---|---|---|---|
| tracking_anchor_pos | **0.5** | std 0.3 | tracking_anchor_pos_error :60 | 根节点世界系位置（含 XY） |
| tracking_anchor_ori | **0.5** | std 0.4 | tracking_anchor_ori_error :82 | 根节点全姿态角误差 |
| tracking_relative_body_pos | 1.0 | std 0.3 | :435 | 14 个身体点，**先对齐根再比**（去掉全局漂移） |
| tracking_relative_body_ori | 1.0 | std 0.4 | :459 | 同上，姿态 |
| tracking_body_linvel | 1.0 | std 1.0 | :531 | 世界系线速度 |
| tracking_body_angvel | 1.0 | std 3.14 | :554 | 世界系角速度 |
| tracking_vr_5point_local | **2.0** | **std 0.1** | :350 | 根局部系下的关键点（release 改成 3 点：躯干+0.5m 偏移、双腕）。**权重最大、std 最紧**，是主奖励 |
| action_rate_l2 | -0.1 | — | isaaclab 标准 | 同我们 |
| joint_limit | -10.0 | — | joint_pos_limits | 同我们 |
| undesired_contacts | -0.1 | 力>1N | 除脚踝/手腕/肘外任何接触 | 比我们的 -10 温柔 100 倍 |
| anti_shake_ang_vel | -5e-3 | 死区 1.5 rad/s | anti_shake_ang_vel_l2 :577 | 只罚腕+头超过 1.5 rad/s 的部分，防高频抖 |
| feet_acc | **-2.5e-6**（release 覆盖，terms 默认 -2.5e-7） | — | joint_acc_l2，只作用于踝关节 | 防脚部急动 |

v1_1 额外加两项（`rewards/tracking/local_feet_acc_energy_5pt.yaml`）：tracking_vr_2wrists_local_ori（0.4, std 0.1）、energy_consumption（-1e-4）。

**SONIC 没有的东西**：没有脚接触匹配奖励、没有 foot slide 惩罚、没有 swing 惩罚、没有 self-collision 大罚（走 undesired_contacts -0.1）、没有 alive bonus、没有力矩惩罚（release）。

**vs 我们**：我们 body pos/ori/lin/ang 各 1.0 + global_root_pos 1.0 + root_ori 0.5，结构几乎一样；差别是 SONIC 给"根局部系关键点"开 2.0 权重 + 0.1 紧 std 当主菜，把"全局根位置"降到 0.5 当配菜。我们的滑冰三件套（contact_match +1.5 / slide -3 / swing_refusal -1.5）SONIC 完全没有。
**值得抄？部分抄**：抄"局部关键点 2.0/std 0.1 当主奖励"的比重思路（Y）；滑冰三件套保留我们自己的（我们有 X2 滑行前科，SONIC 靠速度跟踪+feet_acc 压滑行，未必够）。

---

## 2. 终止条件（含 adaptive 逻辑，已核实）

组合：`config/manager_env/terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml`（该文件里直接覆盖了 terms 的默认阈值），函数在 `envs/manager_env/mdp/terminations.py`。

| 项 | 阈值（release） | 实际比什么 | 函数行号 |
|---|---|---|---|
| anchor_pos | 0.15，adaptive 时 0.75 | **只比根的 Z 高度差**，不是 3D | exceeded_anchor_height :97 |
| ee_body_pos | 0.15，adaptive 时 0.75 | 双踝+双腕，**逐个只比 Z 高度差**，且参考位置已对齐到机器人根（body_pos_relative_w） | exceeded_body_height :206 |
| foot_pos_xyz | 0.2 | 双踝 **3D** 距离，但也是根对齐后的相对位置 | exceeded_body_pos :182 |
| anchor_ori_full | 0.2（**平方值**） | 根四元数角误差平方 > 0.2 → 角误差 > 0.447 rad ≈ 25.6° | exceeded_anchor_ori :160 |
| time_out | — | 动作片段播完 | tracking_time_out :245 |

**threshold_adaptive 到底干什么**（terminations.py:124-126, 238-240）：
```python
thresh = full(threshold)                       # 默认 0.15
thresh[command.running_ref_root_height < 0.5] = down_threshold  # 0.75
```
`running_ref_root_height` 是**参考动作根高度的 EMA**（commands.py:3225-3230，α=0.1，reset 时置为当前参考高度，commands.py:2992）。也就是：**当参考动作本身是低姿态（蹲、爬、躺，根高 < 0.5m）时，高度容差从 0.15 放宽到 0.75**。这就是他们能训地面动作、翻滚、起身而不被终止条件掐死的机关。

**最关键的隐藏事实**：除 foot_pos_xyz 外全部只看 Z；而 foot 用的是根对齐后的相对位置。所以 **全局 XY 漂移在 SONIC 里永远不会导致终止**——XY 只靠 anchor_pos 奖励（0.5, std 0.3）软拉回来。

**vs 我们**：我们 anchor_pos 0.3m 全 3D + ee 0.25m 全 3D，全局 XY 偏 30cm 就死。SONIC 是"高度错 15cm 才死（趴下时放宽到 75cm），XY 随便漂"。
**值得抄？Y，优先级最高之一**。长片段/难片段上，全 3D 终止会让策略根本活不到难点处；把 XY 从终止里拿掉、改成 Z-only + 低姿态自适应放宽，是他们能吃下 fallen/ground motion 的核心。我们如果要做"摔倒爬起"，没有这条根本训不了。

---

## 3. 动作采样：按失败率的分桶采样（完整实现）

配置：`config/manager_env/commands/terms/motion.yaml`（adaptive_sampling 块）；实现：`utils/motion_lib/motion_lib_base.py`。

参数（release 实际值）：bin_size=50 帧（target_fps=50 → **每桶 1 秒**）、init_num_failures=1、uniform_sampling_rate=0.1、pre_failure_sample_window=200、use_failure_rate_decay=false、adp_samp_failure_rate_max_over_mean=**200**（release 覆盖，默认 50）。

机制（逐条核实）：
1. **建桶**（init_adaptive_sampling :2260-2428）：每条动作切成 50 帧一桶，每桶两个计数器：episodes、failures，都初始化为 1 → 没采过的桶失败率=1.0，天生优先被采。
2. **记账**（update_adaptive_sampling :2464）：每个 sim 步，对**因失败终止**（`reset_terminated`，timeout 不算，commands.py:3215）的环境，往"失败发生时刻所在的桶"记一次失败；所有环境往当前桶记 episode（按桶长归一）。
3. **算概率**（update_adaptive_sampling_probabilities :2564）：
   - failure_rate = failures/episodes，**截断到均值×200**（防一条烂数据独吞采样）；
   - 概率 = 0.9×按失败率 + 0.1×均匀兜底；
   - 再乘 bin 权重：sequence_length_agnostic=true → 每条**序列**（不管长短）总采样权重相同（权重 = 桶长/均长 ÷ 该序列桶数）；
   - 另有 max_prob_per_bin / max_prob_per_motion 上限（release 未开）。
4. **采起点**（sample_motion_ids_and_time_steps :2815-2848）：multinomial 采桶 → 桶内均匀采一帧 → **再往前随机退 0~200 帧（0~4 秒）**，让策略从难点之前开始练，"在失败发生前就要采取动作"（源码注释原话）。
5. **刷新节奏**：概率每个 PPO 迭代重算（ppo_trainer.py:1960-1973），多卡每 200 迭代同步一次；动作库每 **250 迭代**整体重采一批（`config/callbacks/im_resample.yaml`，ImResampleCallback）。失败统计随 checkpoint 保存恢复（get_state_dict :2428）。

**vs 我们**：我们均匀采（geodecity 是课程式，不是失败率驱动；且我们吃过"全池一起灌"的亏）。
**值得抄？Y，第一名**。这就是用户说的"学习配置"本体：算力自动流向还没学会的 1 秒片段，同时 0.1 均匀兜底 + 200 倍上限防灾难性遗忘。pre_failure 回退 4 秒这个细节尤其值钱——直接采失败帧是学不会的，要从失败前开始。

---

## 4. RSI / 初始状态随机化

配置：`config/manager_env/commands/terms/motion.yaml`；代码：commands.py:2994-3056（`if not self.is_evaluating` 块，eval 时全关）。

每次 reset：从上面的采样器拿 (动作, 起始帧)，把参考帧的根位姿、根速度、关节位置、关节速度**全部写进仿真**（含参考速度——不是零速起步），然后叠加噪声：
- pose_range：x/y ±0.05m，z ±0.01m，roll/pitch ±0.1，yaw ±0.2 rad
- velocity_range：x/y ±0.5，z ±0.2 m/s；roll/pitch ±0.52，yaw ±0.78 rad/s（**加在参考速度上**）
- joint_position_range：±0.1 rad（加完 clip 到软限位）；joint_velocity_range：[0,0] 关着

**躺倒/低姿态怎么处理**：没有专门的 fallen-reset 逻辑。躺姿片段就是普通片段——照常 RSI 到躺着的参考帧，靠第 2 节的 adaptive 终止（根高<0.5m 时容差 0.75）让它活下来。

**vs 我们**：我们 pose_range 是空的，零噪声 RSI。
**值得抄？Y，第二名**。起点不加噪，策略只见过"完美贴参考"的状态，被推一下、偏一点就回不来了——这和我们"恢复必须来自潜空间搜索"的目标直接相关。照抄这组数字即可，成本几乎为零。

---

## 5. 域随机化（events）

release 用 `config/manager_env/events/tracking/level0_4.yaml`（覆盖 base），terms 在 `config/manager_env/events/terms/`：

| 项 | 时机 | 范围 |
|---|---|---|
| physics_material | startup | 静摩擦 [0.3, 1.6]，动摩擦 [0.3, 1.2]，弹性 [0, 0.5]，64 桶 |
| add_joint_default_pos | startup | 每关节默认位置加 ±0.01 rad（模拟标定误差），**同时改 action offset**（events.py:33-100） |
| base_com | startup | 躯干质心 x ±0.025，y ±0.05，z ±0.05 m |
| randomize_rigid_body_mass | startup（level0_4） | **双腕+躯干质量 scale [0.8, 2.5]**（模拟负载！）；base 版是 reset 时全身 [0.8, 1.2] |
| push_robot | interval **4-6s**（level0_4；base 是 1-3s） | 速度脉冲 xy ±0.5，z ±0.2 m/s，rpy ±0.52/±0.52/±0.78 rad/s |

执行器延迟：`mdp/actuators.py` 有 DelayedImplicitActuator（每次 reset 随机 lag），但**没有任何配置引用它**——release 没用延迟。动作 clip ±20，隐式 PD（增益在 `envs/manager_env/robots/g1.py`，非随机化）。

**vs 我们**：我们 push 1-3s（比 release 更凶），但没有摩擦/质量/质心/标定随机化（mjlab 下部分待查）。
**值得抄？Y（选摘）**：摩擦 [0.3,1.6] + 躯干质心 ±5cm + 腕/躯干质量 0.8-2.5 倍这三个最实惠；标定误差 ±0.01 那条对 sim2real 关键、对纯 sim 评估无所谓。

---

## 6. PPO 超参

基座：`config/algo/ppo_im_phc.yaml`；release 覆盖在 `sonic_release.yaml` 的 algo.config 块；实现 `trl/trainer/ppo_trainer.py`。

| 参数 | SONIC release | 我们 |
|---|---|---|
| num_envs | 4096（每卡） | 4096 |
| num_steps_per_env | **24**（默认 32，release 覆盖） | 24 |
| epochs / minibatches | 5 / 4 | 5 / 4 |
| gamma / lam / clip | 0.99 / 0.95 / 0.2 | 0.99 / 0.95 |
| actor lr | **2e-5** 起步，adaptive KL（desired_kl 0.01，×÷1.5，钳在 [1e-5, 2e-4]，:2158-2178） | 1e-3 adaptive KL 0.01 |
| critic lr | **1e-3 固定**（actor/critic 分开） | 同 actor |
| entropy_coef | **0.01** | 0.005 |
| max_grad_norm | **0.1**（默认 1.0，release 特意压到 0.1） | ? |
| 动作噪声 std | 可学习 per-dim，**init 0.05**，硬钳 [0.001, 0.5]（actor_critic_modules.py:112-181） | ? |
| value | clipped value loss，vf_coef 1.0 | |
| advantage | 归一化，多卡先 gather 再归一（:2136-2152） | |
| obs 归一化 | **running_mean_std: false**，不做 | |
| 控制频率 | sim 0.005 × decimation 4 = 50Hz；episode 10s | 50Hz, 10s |
| 迭代数 | 100000 | |

**timeout bootstrap**（ppo_trainer.py:994-998）：`reward += gamma × time_out × V(s)` ——片段播完（不是失败）时用价值函数补上剩余回报，不让"动作放完了"被当成"死了"。

**vs 我们**：三处实质差异——actor lr 上限 2e-4（我们 1e-3，可能高 5 倍）、max_grad_norm 0.1（激进小）、std 硬钳 0.5 + init 0.05（探索噪声极小，因为 RSI 已经提供了状态多样性）。
**值得抄？Y**：max_grad_norm 0.1 + std clamp + adaptive lr 钳 [1e-5, 2e-4] 是一套"稳字诀"，大数据量长训不炸的保险；timeout bootstrap 若 mjlab/rsl_rl 没有必须补上，否则每条片段结尾都在教"末态没价值"。

---

## 7. 多阶段管线？

结论：**没有奖励课程、没有阶段切换、没有权重退火**。`config/manager_env/curriculum/` 只有 empty.yaml。四个 exp 变体 diff 过：
- sonic_release vs sonic_h2：只换机器人（g1_model_12_dex → h2）+ critic 加宽；
- sonic_bones_seed：多加 SOMA 编码器和对应数据（多一路模态）；
- sonic_v1_1：obs 换 heading 归一化、加腕姿态增强、奖励换成加 energy + 腕 ori 的版本。

真正的"多头"在 aux losses（`config/aux_losses/universal_token/g1_recon_and_all_latent.yaml`）：g1 重建 0.01、三对模态 latent 对齐各 1.0——多模态蒸馏是**联合训练**，不是分阶段。动态的部分全在数据侧：失败率采样 + 每 250 迭代重采动作批，等于一个自动难度课程。

**vs 我们**：我们也在纠结要不要分阶段。SONIC 的答案：单阶段一锅端，课程交给失败率采样自动生成。
**值得抄？Y（思路）**：别手工设计阶段，把课程做进采样器。

---

## 8. Actor 观测噪声 / 延迟

`config/manager_env/observations/policy/local_dir_hist.yaml`，enable_corruption: True，全部加性均匀噪声：
- gravity_dir ±0.05，base_ang_vel ±0.2 rad/s，joint_pos ±0.01 rad，joint_vel ±0.5 rad/s，actions 无噪声
- 每项带 **10 帧历史**（actor_prop_history_length=10）
- actor **没有根线速度、没有根位置**观测（可部署配置）；critic（privileged_mf_hist）才有 base_lin_vel、全身体位姿、多帧未来参考，且无噪声
- 无观测延迟（延迟类存在但未启用）

**vs 我们**：我们 actor 噪声情况待对表；若没有，这组数字可直接抄。
**值得抄？Y**，跟 RSI 噪声同属"别让策略见过完美世界"一族。

---

## 9. 特殊机关（容易漏看的）

1. **freeze_frame_aug**（motion_lib_base.py:1787-1795，prob 0.1）：10% 的片段在随机帧起**冻结到结尾**——把"定住不动"当技能显式教。注意：这对我们的 E17 冻结问题是反向证据——SONIC 把冻结当需要专门数据才能学会的技能，说明"会停"和"会动"都得喂。
2. **上半身重组增强**（motion_lib_base.py:1813-1870）：走/跑/蹲等导航类片段（按前缀白名单），50% 概率把上半身姿态流整段换成非导航数据集里随机片段（不够长就来回 ping-pong 拼）。**下肢×上肢组合爆炸式扩数据**——这是他们"数据好"的最大人工杠杆。
3. timeout 价值 bootstrap（见 §6）。
4. 多卡同步的 advantage 归一化；无 obs running mean std。
5. 对称性增强代码齐全（_flip_obs/_flip_actions，ppo_trainer.py:1015-1109）但默认关。
6. 没有 AMP、没有 style reward、没有奖励归一化。
7. 失败率统计写进 checkpoint，续训不丢课程。

---

## 可移植改动 Top-5（按预期收益排序）

1. **失败率分桶采样 + 失败前 4 秒回退起点**（§3，motion_lib_base.py:2260-2848）。预期效果：难片段成功率显著抬升、易片段不退化（0.1 均匀兜底+200 倍上限）；等于免费拿到一个自动课程。工程量最大但独立成模块，可直接移植到我们的 motion lib。
2. **终止条件手术：XY 不判死、高度 0.15、低姿态自适应放宽到 0.75**（§2，terminations.py:97-244）。预期效果：长片段和地面动作从"必死"变"可训"；这是解锁"摔倒爬起"四支柱的前置条件。改配置+抄两个函数，一天内能做。
3. **RSI 全套噪声**（§4：pose ±5cm/yaw ±0.2，速度 ±0.5，关节 ±0.1）+ **actor 观测噪声**（§8）。预期效果：抗推、偏离参考后能回来；纯配置改动，零成本先上。注意要和第 2 条一起上——起点加噪但终止很紧 = 白白多死。
4. **PPO 稳字诀**：max_grad_norm 0.1、std 钳 [0.001, 0.5] init 0.05、actor lr 钳 [1e-5, 2e-4]、timeout 价值 bootstrap（§6）。预期效果：不提天花板，但长训不塌；我们 lr 1e-3 上限比他们高 5 倍，大概率是隐患。
5. **上半身重组 + 冻结帧增强**（§9.1-9.2）。预期效果：上下肢解耦泛化、静止姿态可控；数据侧改动，符合"数据好架构随便"的主张，但要先给动作分好导航/非导航两池（我们的 action_labels 分类器可以直接派上用场）。

没抄的、且明确建议不抄的：SONIC 无脚接触/滑移奖励——我们的滑冰门是拿 X2 的血换来的，保留。
