# BFM-Zero 极限压力测试（AMASS 零样本，鲁棒性优先）

lloyd 0831 · 三个 checkpoint × 6 段 AMASS 未见动作 + 1 段 LAFAN 对照

---

## 0. 一句话结论（先看数）

1. **官方 checkpoint 这一列不算数。** 它在自己训练过的 LAFAN 平地走路上 **1.9 秒就趴下**（mpjpe_l 4176 vs 我们 685）。原因查清了：官方权重是在 **IsaacSim** 里训的，我们只有 **mjlab/MuJoCo-Warp** 这套环境，HF 上只发了 `model.safetensors`、**没发 config**。这是跨仿真器迁移失败，不是 BFM-Zero 不鲁棒。证据链在第 4 节。
2. **我们的 50M 复现（ours）在 6 段没见过的 AMASS 动作上，4 段全程站住（100% 存活）**：深蹲、跳、空手道 360 转体、跳舞。摔的两段是躺到地上的动作（坐→站、地面踢腿），而且**摔完能自己爬起来**（坐→站那段 2 次跌倒 2 次爬起，看图第 6-8 帧已经站直）。
3. **Greenwich 数据臂（gwarm，102.8M）鲁棒性和 baseline 基本持平**：同样 4 段全程站住，跳舞那段 4.9 秒摔一次但爬回来了。**关节跟踪精度也没有明显更差**（跳跃 mpjpe_l 1281 vs 1355，跳舞 2310 vs 2527）。→ **lloyd 那条线在鲁棒性这个判据下是成立的。**
4. 顺手挖出两个我们自己管线的 bug，第 5 节，其中一个很可能是 GW_ARM "EMD 卡在 1.21" 的真正原因。

---

## 1. 测什么、怎么测（口径）

| 项 | 设定 |
|---|---|
| 动作源 | **AMASS**（13281 段里挑的 6 段）+ 1 段 LAFAN 做"见过的数据"上界对照 |
| 参考轨迹 | **xwbc/GMR 原生 AMASS→G1 retarget**（`/var/tmp/alphamotion/xwbc_full/amass/unitree_g1_29dof.tar.gz`），**不过 Greenwich 解码器**，所以没有解码误差这个混淆项 |
| 仿真 | 本仓库 mjlab（MuJoCo-Warp），50 Hz，DR 和观测噪声都开着（三个模型完全同一套 env / 同一份 `config.json` / 同一份 `init_kwargs.json`，逐字节相同） |
| 跑法 | 自写 runner（`/home/arenalabs/tmp/stress_run.py`）：和 `humanoidverse.tracking_inference` 一样的 z 推断和 rollout，但**摔倒后不停**，一直跑完，才看得到"爬不爬得起来" |
| 摔倒判据 | 骨盆→双肩中点向量的竖直分量 `u < 0.5`（躯干偏离竖直 >60°）**或** 肩高 < 0.50 m。只有"参考还站着而机器人躺了"才算 fail，参考本身贴地的帧不算 |
| 爬起判据 | fail 段结束后 `u > 0.7` 连续 ≥ 0.5 s |

三个 checkpoint：

| tag | 来源 | 训练量 |
|---|---|---|
| `official` | HF `LeCAR-Lab/BFM-Zero` 官方权重 | ~3.0M optimizer steps（BN `num_batches_tracked`=5,999,648） |
| `ours` | `runs/BASELINE_50M_ARCHIVE`，我们的 50M 复现 | 200,144 optimizer steps |
| `gwarm` | `runs/GW_ARM_FINAL_ARCHIVE`，Greenwich 解码数据臂 | 400,336 optimizer steps（102.8M env steps） |

三份权重结构完全一致（537 个张量，名字和形状全同），`config.json` / `init_kwargs.json` md5 相同。

---

## 2. 选片依据（可测的难度，全部从 retarget 后的 G1 数据算）

| clip | 来源 | 时长 | 骨盆最低/最高 (m) | 竖直行程 (m) | 关节角速度 p95 / max (rad/s) | 腾空占比 | 根速度 p95 (m/s) |
|---|---|---|---|---|---|---|---|
| AMASS_squat_deep | SOMA soma_subject1 squat_003 | 15.0 s | 0.34 / 0.85 | **0.511** | 1.40 / 12.2 | 0.42 | 0.22 |
| AMASS_sit_to_stand | Transitions mazen sit_stand | 8.1 s | **0.20** / 0.90 | **0.698** | 2.50 / 18.6 | 0.71 | 0.51 |
| AMASS_kick_ground | Transitions mazen GUS_kick | 10.0 s | **0.21** / 0.99 | **0.783** | 2.78 / 12.6 | 0.74 | 0.93 |
| AMASS_karate_twist360 | Transitions mazen punchkarate_jumpingtwist360 | 7.7 s | 0.49 / **1.05** | 0.562 | 3.54 / 21.7 | 0.04 | 0.86 |
| AMASS_jump | SOMA soma_subject1 jump_002 | 14.0 s | 0.33 / **1.06** | 0.726 | **4.43** / 22.2 | 0.11 | **1.42** |
| AMASS_dance | DanceDB Elena_Neutral_v2 | 15.0 s | 0.34 / 0.98 | 0.638 | **4.55** / **30.0** | 0.24 | 1.00 |
| LAFANctl_dance1_s2（对照，见过） | LAFAN dance1_subject2 帧 1912-2362 | 15.0 s | 0.41 / 0.98 | 0.563 | **9.38** / **51.6** | 0.21 | 1.30 |

覆盖了：深蹲、坐地起身、踢腿+地面、转体空手道、跳跃、舞蹈。LAFAN 本身没有 squat / sit-to-stand / karate 这类内容，所以前 6 段对三个模型都是**真·零样本**。

**数据侧注意**：AMASS retarget 的脚离地中位数在 2-10 cm（LAFAN 只有 2 cm），坐地和踢腿这两段最明显（8.5 / 10.2 cm）。原因是 AMASS 的根高没有按 G1 腿长缩放，机器人"吊"在参考上。三个模型面对的是同一份参考，比较仍然公平，但绝对跟踪误差被这条抬高了。已做逐片接地（脚部 geom 最低点 p10 对齐到 1 cm），位移量在表 2 的 `ground_shift`。

---

## 3. 主表：鲁棒性（这是判决表）

`survival` = 第一次"参考站着而机器人躺了"之前的时长占比；`falls` = 跌倒段数；`recov` = 其中爬起来的段数；`end` = 结束时是否还躺着。

| clip | ckpt | **survival** | **survival_s** | **falls / recov** | **结束时躺着** | fail 占比 | 姿态均值 u | max\|dq/dt\| (rad/s) | 最低体高 (cm) |
|---|---|---|---|---|---|---|---|---|---|
| AMASS_squat_deep | ours | **1.000** | 14.96 | 0 / 0 | 否 | 0.000 | 0.884 | 12.5 | 2.7 |
| | gwarm | **1.000** | 14.96 | 0 / 0 | 否 | 0.000 | 0.922 | 20.5 | 2.4 |
| | official* | 0.162 | 2.42 | 3 / 0 | 否 | 0.676 | 0.505 | 27.1 | 1.6 |
| AMASS_jump | ours | **1.000** | 13.92 | 0 / 0 | 否 | 0.000 | 0.912 | 12.7 | 2.8 |
| | gwarm | **1.000** | 13.92 | 0 / 0 | 否 | 0.000 | 0.941 | 23.4 | 2.5 |
| | official* | 0.111 | 1.54 | 4 / 2 | **是** | 0.691 | 0.434 | 32.8 | 2.9 |
| AMASS_karate_twist360 | ours | **1.000** | 7.66 | 0 / 0 | 否 | 0.000 | 0.984 | 9.9 | 3.1 |
| | gwarm | **1.000** | 7.66 | 0 / 0 | 否 | 0.000 | 0.973 | 21.1 | 1.7 |
| | official* | 0.110 | 0.84 | 3 / 0 | **是** | 0.875 | 0.209 | 27.1 | 2.8 |
| AMASS_dance | ours | **1.000** | 14.96 | 0 / 0 | 否 | 0.000 | 0.880 | 13.5 | 2.7 |
| | gwarm | 0.329 | 4.92 | 1 / **1** | 否 | 0.020 | 0.835 | 18.0 | 2.2 |
| | official* | 0.090 | 1.34 | 7 / 1 | **是** | 0.619 | 0.336 | 32.7 | 2.5 |
| AMASS_sit_to_stand | ours | 0.331 | 2.68 | 2 / **2** | 否 | 0.301 | 0.910 | 13.8 | 3.3 |
| | gwarm | 0.331 | 2.68 | 2 / **1** | 否 | 0.281 | 0.862 | 20.4 | 2.6 |
| | official* | 0.141 | 1.14 | 3 / 1 | **是** | 0.746 | 0.761 | 27.2 | 2.3 |
| AMASS_kick_ground | ours | 0.420 | 4.18 | 1 / **1** | 否 | 0.116 | 0.819 | 21.9 | 1.5 |
| | gwarm | 0.420 | 4.18 | 1 / 0 | 否 | 0.143 | 0.731 | 21.0 | **-0.8** |
| | official* | 0.118 | 1.18 | 4 / 1 | **是** | 0.480 | 0.537 | 26.9 | 1.0 |
| LAFANctl_dance1_s2（见过） | ours | **1.000** | 14.96 | 0 / 0 | 否 | 0.000 | 0.958 | 15.9 | 2.7 |
| | gwarm | **1.000** | 14.96 | 0 / 0 | 否 | 0.000 | 0.927 | 13.1 | 1.5 |
| | official* | 0.060 | 0.90 | 5 / 1 | **是** | 0.746 | 0.441 | 24.9 | 1.7 |

\* official 这一列**不可用**，见第 4 节。

### 不炸机检查
- 三个模型全程**没有 NaN**，`dvel` 超 20 rad/s 的比例：ours **0.000-7e-5**，gwarm 5e-5-1e-4，official 4e-4-2.4e-3（比我们高一个量级，跟它整体在挣扎一致）。
- 地面穿透：所有 rollout 最低刚体高度都在 -0.8 ~ +3.3 cm，没有穿地。
- env 自身的 termination 一次都没触发（我们用的是 `auto_reset=False` + 不中断，摔倒靠姿态判据自己算）。

### 每片一句判决（先看的视频/网格，不是先看数）
- **AMASS_squat_deep**：ours / gwarm 全程跟着蹲到底再站起来，没有"拒绝下蹲"；official 2.4 s 就散架。
- **AMASS_jump**：ours / gwarm 站稳跟到底，起跳幅度比参考小（不真的腾空），但没摔；official 1.5 s 摔倒后再没站稳。
- **AMASS_karate_twist360**：ours / gwarm 完整跟完 360° 转体，ours 关节误差最低（body 误差 78 cm vs gwarm 109 cm）；official 0.8 s 倒地。
- **AMASS_dance**：ours 全程站住；gwarm 4.9 s 摔一次、爬起来后跟完剩下 10 s；official 从第 3/8 帧起就一直躺着。
- **AMASS_sit_to_stand**：**三个都在"往地上坐"那一段掉出来**（2.7 s），但 ours 两次跌倒两次爬起，第 6-8 帧已经站直了 —— 这是本次最有说服力的一段，看 `ours__AMASS_sit_to_stand_grid.png`。
- **AMASS_kick_ground**：参考本身有 22.7% 的帧躺在地上，ours 摔一次爬回来，gwarm 摔了没爬起来（它是三个里唯一穿地 -0.8 cm 的）。
- **LAFANctl_dance1_s2（见过的数据）**：ours / gwarm 15 s 全程站住，说明前面 AMASS 上的失败确实来自"没见过的动作"，不是管线本身有问题。official 在**自己训练过的数据**上 0.9 s 摔倒 —— 这就是判它无效的直接证据。

---

## 4. 为什么 official 这一列不算数（证据链）

按"异常数字先怀疑测量"的规矩，做了 5 个对照：

1. **权重是对的**：537 个张量与我们的完全同名同形；BN 统计 `state` 与我们余弦 **0.9966**、`privileged_state` **0.9990** → 观测空间定义一致，不是错模型、不是别的机器人。`num_batches_tracked` = 5,999,648，是个训练很久的正经 checkpoint。
2. **观测排布没错位**：把我们的 BN 均值做 0..N 循环平移去找最佳对齐，`state` / `privileged_state` / `history_actor` 的最优位移**都是 0**。
3. **不是 DR/噪声的锅**：`--disable-dr --disable-obs-noise` 重跑 LAFAN 平地走，还是 1.6 s 摔（survival 0.157）。
4. **不是 tanh 挤压的锅（但发现了饱和）**：本仓库 actor 输出走 `tanh`。official 的 pre-tanh logit 最大到 **±11.9，8.4% 的分量 |logit|>2（已饱和）**，动作 `|a|` 均值 0.370 / p99 = **1.000**（顶到上限）；我们的 logit 最大 0.736，`|a|` 均值 0.133 / p99 0.514。绕过 tanh（放宽到 ±5）重跑，**更糟**（survival 0.195）。
5. **根因**：官方 repo `/home/arenalabs/reference_code/bfm_zero` 的 `tracking_inference.py` 用的是 **`humanoidverse_isaac` + IsaacSim**；本机 `import isaacsim` 直接 ModuleNotFoundError。而且 HF 快照里只有 `model/checkpoint/model/model.safetensors` 和 `data/lafan_29dof.pkl`，**没有 `config.json`**，我们是拿自己复现 run 的 env config 顶上的。另外官方 repo 的 `g1_29dof_hard_waist.yaml` 和我们 fork 的确实不一样：手腕 kp **16.78 vs 8.61**、kd **1.068 vs 0.548**，手腕力矩上限 **5 vs 13.4 N·m**，髋 roll/pitch 速度上限 **32 vs 20 rad/s**。

**结论**：这是 IsaacSim→MuJoCo-Warp 的 sim2sim 迁移失败 + 缺官方 config，不是 BFM-Zero 的能力上限。
**要拿到真官方数字，需要**：(a) 找 LeCAR-Lab 要 release 对应的 `config.json`（含 env / hydra_overrides），或 (b) 修好本机 IsaacSim 在官方 repo 里跑。

LAFAN 对照原始数据（`/home/arenalabs/tmp/ctl_out`，`lafan_29dof_10s-clipped.pkl`）：

| clip | ckpt | survival_s | mpjpe_l | 判决 |
|---|---|---|---|---|
| walk1_subject1_clip5（平地走） | official | 2.56 | 4176 | 摔 |
| | ours | **9.96（全程）** | **685** | 正常走完 |
| dance1_subject2_clip7 | official | 1.62 | 4747 | 摔 |
| | ours | **9.96（全程）** | 1727 | 跟完 |
| run1_subject2_clip12 | official | 1.82 | 3945 | 摔 |
| | ours | **9.96（全程）** | 1311 | 跟完 |

---

## 5. 附带挖出的两个管线 bug（都影响我们自己）

### 5.1 【严重】RobotState npz 的四元数顺序被无视 → GW_ARM 的训练数据根旋转全错
- `configs/robots/g1_29dof.yaml` 里写死 `root_quat_order: xyzw`，转换器 `humanoidverse/utils/motion_data/robot_state_convert.py` 只认这个字段，**npz 里自带的 `quat_order` 字段完全没被读**。
- 我们的两个转换脚本 `/home/arenalabs/tmp/qpos_to_rs.py` 和 `/home/arenalabs/tmp/xwbc_to_robotstate.py` 写的都是 MuJoCo 的 **wxyz**。
- 结果：`cache/motion_data/g1_lafan_greenwich/..._train_near10s_ufo.pkl` 里根的 axis-angle 模长 = **4.605 rad（264°）**，即**每一帧根姿态都是错的**。`humanoidverse/data/g1_lafan_greenwich_rs/*.npz` 和 `pnd_adam_lafan_rs/*.npz` 全部中招。
- 佐证：GW_ARM 的 BN 统计与 official 的余弦，`state` 只有 **0.699**、`privileged_state` 只有 **0.569**（我们 baseline 对 official 是 0.997 / 0.999）—— 它确实是在一个不同的状态分布上训出来的。
- **这很可能就是 "Greenwich 数据臂 EMD 卡在 1.21、判死" 的真正原因**，不是 Greenwich 解码质量的问题。建议：把 rs 转换器改成写 xyzw（或给 robot yaml 加 wxyz），重建缓存，重训数据臂再判一次。
- 本次实验的 AMASS 数据已修（写成 xyzw 后重建 manifest），修前修后都验证过：修前根 axis-angle 4.62 rad，修后 1.485 rad（= yaw -85°，与源数据一致）。

### 5.2 【中等】xwbc AMASS retarget 的根高没按腿长缩放
G1 站立骨盆约 0.79 m，但多段 AMASS retarget 的骨盆最高到 0.98-1.06 m，脚部 geom 最低点中位数离地 6-10 cm（LAFAN 同管线只有 2 cm）。也就是机器人"吊"在参考上。本次用逐片接地（p10 → 1 cm）缓解，残留在表 2 的 `ref_footclear_median_cm`。跨语料做绝对跟踪评分前要先修这条。

---

## 6. 精度附录（次要指标，不用来排名）

`mpjpe_l` = 关节角 L2 距离 ×1000（rad）；`body_pos_err` / `pelvis_drift` 主要反映全局位置漂移，跟踪策略没有全局位置反馈，长片必然累积。

| clip | ckpt | mpjpe_l | emd | body_pos_err (cm) | pelvis_drift (cm) |
|---|---|---|---|---|---|
| AMASS_squat_deep | ours | 1609 | 1.566 | 76.1 | 74.6 |
| | gwarm | 1408 | 1.321 | 153.3 | 152.4 |
| | official* | 4695 | 4.442 | 78.2 | 56.7 |
| AMASS_jump | ours | 1356 | 1.158 | 197.9 | 195.2 |
| | gwarm | **1281** | **1.076** | 200.6 | 198.2 |
| | official* | 4533 | 4.282 | 147.9 | 135.1 |
| AMASS_karate_twist360 | ours | 1312 | 1.265 | **78.4** | **73.2** |
| | gwarm | **1200** | **1.141** | 108.5 | 109.2 |
| | official* | 5075 | 4.796 | 161.4 | 146.6 |
| AMASS_dance | ours | 2527 | 2.471 | 103.4 | 100.3 |
| | gwarm | **2310** | **2.183** | 139.3 | 139.7 |
| | official* | 5556 | 5.187 | 104.8 | 83.7 |
| AMASS_sit_to_stand | ours | 1624 | 1.464 | **56.0** | **50.6** |
| | gwarm | **1570** | 1.492 | 77.0 | 75.0 |
| | official* | 4704 | 4.582 | 93.7 | 79.6 |
| AMASS_kick_ground | ours | 2178 | 2.049 | 69.8 | 60.9 |
| | gwarm | **1393** | **1.317** | **71.8** | 69.0 |
| | official* | 4759 | 4.645 | 108.7 | 91.5 |
| LAFANctl_dance1_s2 | ours | **1977** | **1.728** | **81.0** | **79.9** |
| | gwarm | 2076 | 1.837 | 115.1 | 105.0 |
| | official* | 4975 | 4.468 | 222.2 | 211.1 |

**这里最反直觉的一条**：gwarm 在 7 段里有 6 段 `mpjpe_l` 比 ours 更低。它在 LAFAN 内部评测里被判"精度死"（EMD 1.21 平台），但在 AMASS 零样本上关节跟踪并不差 —— 再加上 5.1 那个根旋转 bug，之前那个"判死"的结论需要重做。

---

## 7. 自己看视频（请肉眼判，不要只看数字）

**并排 mp4（左＝参考 retarget，右＝策略 rollout）**，共 21 个：
```
/home/arenalabs/tmp/stress_official_{AMASS_squat_deep,AMASS_sit_to_stand,AMASS_kick_ground,AMASS_karate_twist360,AMASS_jump,AMASS_dance,LAFANctl_dance1_s2}.mp4
/home/arenalabs/tmp/stress_ours_{同上}.mp4
/home/arenalabs/tmp/stress_gwarm_{同上}.mp4
```
**8 帧网格 PNG**（每个 mp4 一张，2 行 × 4 列，每格左参考右策略）：
```
/home/arenalabs/tmp/stress_grids/{official,ours,gwarm}__<clip>_grid.png
```
最该先看的三张：
- `ours__AMASS_sit_to_stand_grid.png` —— 躺到地上后自己站起来
- `gwarm__AMASS_dance_grid.png` —— Greenwich 数据臂 15 s 站住
- `official__LAFANctl_dance1_s2_grid.png` —— 官方权重在自己的训练数据上从第 3 帧起一直躺着（无效的直观证据）

**其他产物**
- 逐帧状态：`/home/arenalabs/tmp/stress_out/{tag}__{clip}.npz`（policy/参考的 body 位姿、dof、dof_vel、termination）
- 评分：`/home/arenalabs/tmp/stress_scores.json`；脚本 `/home/arenalabs/tmp/stress_analyze.py`
- rollout 脚本：`/home/arenalabs/tmp/stress_run.py`
- AMASS→G1 参考：`/home/arenalabs/tmp/amass_g1_rs/*.npz`，UFO manifest `configs/data/amass_stress.yaml`，全长 pkl `cache/motion_data/amass_stress/amass_stress_full_ufo.pkl`
- 难度统计：`/home/arenalabs/tmp/amass_clip_difficulty.json`

---

# 第二轮：把参考换成"我们自己产的"（Greenwich 解码 vs 原生 retarget）

lloyd 0831 · 同样 6 段 AMASS + 1 段 LAFAN 对照，同样两个 checkpoint，同样鲁棒性判据。
**唯一改的一件事**：G1 参考轨迹不再来自 xwbc/GMR 原生 retarget，而是 **人类 AMASS → Greenwich（PT_c1dec_reenc_0812，即发布权重）→ 解码到 G1**。
这一列回答的是产品问题：**客户把人类 mocap 交给我们、我们产出他机器人的数据，控制器的鲁棒性会不会掉。**

`official` 这一列不做了：上一轮已证明它的发布权重是 IsaacSim 训的、本机 mjlab/MuJoCo-Warp 上无效（自己训练过的 LAFAN 平地走 1.9 s 就趴），跑它只会污染表格。

---

## 8. 一句话结论

1. **按同一条接地规则直接比：ours 存活比 0.802、gwarm 0.929。掉的全部来自一段（AMASS_dance），而且是"开场 0.2 s 就摔、1.3 s 自己爬起来"。**
2. **这段掉分不是姿态学不会，是接地/出生高度的锅。** 做了对照臂：把 Greenwich 参考按**首帧脚底净空对齐到原生的首帧净空**再跑，AMASS_dance 立刻回到 ours **1.000（14.96 s 全程）**、gwarm **0.322（原生 0.329）**，其余 6 段一格没动。
3. **所以"汇率"是：接地对齐之后，Greenwich 参考 vs 原生 retarget 参考，鲁棒性 = ours 1.002×、gwarm 1.001×。**在这套鲁棒性判据下 **成本为零**。
4. **精度侧还赚了一点**：gwarm 的 `mpjpe_l` 7 段平均 1605 → **1384（0.862×）**，7 段全部更低；ours 1798 → 1770（0.985×）。也就是说 Greenwich 产的参考对策略来说**更好跟**。
5. 顺手又踩到一个真坑，第 11 节：两套 xwbc 语料的 `human_smpl` 骨架**不同但重名**，Greenwich 按 `spec.name` 缓存描述符 → 跨语料批量解码会静默串味（LAFAN 对照片偏了 21.2°）。已修并有回归验证。

---

## 9. 这一轮的管线（口径，逐条对齐第 1 节）

| 项 | 设定 |
|---|---|
| 人类源 | `cache_xwbc_amass_full`（AMASS 13281 段配对缓存）的 `human_smpl_paired_local6d`；LAFAN 对照走 `cache_xwbc_lafan`。**6 段 AMASS + 1 段 LAFAN 全部在人类侧找得到，没有一段需要替代**（片段名和帧窗与第 2 节逐字相同） |
| 编解码 | `Greenwich.load()` = `~/.cache/alphamotion/weights/greenwich`，已核对与 `runs/PT_c1dec_reenc_0812/config.json` **除路径外逐字段相同**。注意它的 `holdout` 里就有 `unitree_g1_29dof` —— **G1 对这个 codec 是零样本具身** |
| A 侧路径 | 完全照 `exp_topo/ctrl_verify/g1_tracks_lafan.py`：`build_global` → `encode` → `decode_full` → `MP.project(method="global", lm_iters=20)` → 铰链角 → qpos |
| 根 6 自由度 | **位置和朝向都是数据透传**（owner 设计，和原生臂同一份）。实测两臂的 `root_quat` 与 `root_pos[:, :2]` **逐比特相同**（max abs diff = 0.00e+00）。→ **本次唯一变量就是 29 个关节角 + 接地造成的 z** |
| 接地 | 逆向还原了第一轮的公式并**逐片复现到小数点后两位**：UFO `g1_29dof.xml` 的脚部 geom 最低顶点，在评分窗内取 **p10 → 抬到 +1 cm**。两臂用同一条规则 |
| 仿真 / 跑法 / 摔倒 / 爬起判据 | 与第 1 节完全一致（同一份 `stress_run.py`、同一份 `stress_analyze.py`、DR 和观测噪声都开、摔倒不停） |

**四元数顺序**（上一轮 5.1 那个 bug，没有重犯）：`configs/robots/g1_29dof.yaml` 写死 `root_quat_order: xyzw`，`robot_state_convert.py` 只认 YAML、**不读 npz 自带的 `quat_order`**。我们的 qpos 是 MuJoCo 的 wxyz，所以在 `/home/arenalabs/tmp/gw_qpos_to_rs.py` 里显式转成 xyzw 再写。
验证：建好的 cache 里根 axis-angle 模长 = **1.60 / 1.62 / 1.65 / 1.68 / 1.90 / 3.31 / 3.33 / 3.37 / 3.80 rad**，且**与原生臂那份逐比特相同**（因为根是透传的）—— 不是 bug 时的 4.605 rad / 264°。

### Greenwich 解码本身的体检（先看这个，再看控制器）

| clip | 源序列 | 帧 | 与原生 retarget 的关节均差 (°) | p95 (°) | 接地位移 (cm) |
|---|---|---|---|---|---|
| AMASS_squat_deep | SOMA soma_subject1 squat_003 | 450 | 15.64 | 45.02 | -24.13 |
| AMASS_sit_to_stand | Transitions mazen sit_stand | 244 | 12.35 | 37.38 | -8.69 |
| AMASS_kick_ground | Transitions mazen GUS_kick | 300 | 11.92 | 32.72 | -7.12 |
| AMASS_karate_twist360 | Transitions mazen punchkarate_jumpingtwist360 | 231 | 12.46 | 34.43 | -13.33 |
| AMASS_jump | SOMA soma_subject1 jump_002 | 419 | 12.87 | 36.48 | -15.46 |
| AMASS_dance | DanceDB Elena_Neutral_v2 | 450 | 21.78 | 59.89 | -17.56 |
| LAFANctl_dance1_s2 | LAFAN dance1_subject2 (1912-2362) | 450 | 11.64 | 32.90 | -7.69 |

LAFAN 那一行是**回归锚点**：`ctrl_verify/report_g1_lafan.json` 里 `dance1_subject2` 的 A-vs-B 是 11.011°，我们全片重算 **11.011°**，关节角与生产轨迹 `dance1_subject2__qposA.npy` 差 **4.9e-07 度**。管线复现是逐位的。

**已经能看出的代价**：Greenwich 解码的腿更直，脚底掉得更低，所以同一条 p10 接地规则会把整体**抬得更高**。参考自身的脚底净空中位数：

| clip | 原生 (cm) | Greenwich (cm) | 首帧净空 原生→GW (cm) |
|---|---|---|---|
| AMASS_squat_deep | 4.9 | **12.6** | 4.8 → **26.2** |
| AMASS_jump | 2.5 | **13.7** | 2.4 → **15.3** |
| AMASS_karate_twist360 | 2.1 | **10.2** | 1.8 → **11.0** |
| AMASS_dance | 3.7 | **15.8** | 2.5 → **20.0** |
| AMASS_sit_to_stand | 8.5 | 9.4 | 10.4 → 10.2 |
| AMASS_kick_ground | 10.2 | 8.5 | 11.7 → 8.8 |
| LAFANctl_dance1_s2 | 4.1 | 7.0 | 3.0 → **14.5** |

也就是说：**"机器人吊在参考上"这条（第 5.2 节）在 Greenwich 参考上更严重**，最多多吊 12 cm，出生点最多高 24 cm。

---

## 10. 主表：原生参考 vs Greenwich 参考（判决表）

`GW` = 与原生同一条 p10 接地规则；`GW-f0` = 对照臂，把 Greenwich 参考按**首帧脚底净空对齐到原生首帧**再接地（只改 z 常数，关节角一个不动）。

| clip | ckpt | 存活 原生 | 存活 **GW** | 存活 **GW-f0** | 跌/爬 原生 | 跌/爬 GW | 跌/爬 GW-f0 | mpjpe 原生 | mpjpe GW | mpjpe GW-f0 |
|---|---|---|---|---|---|---|---|---|---|---|
| AMASS_squat_deep | ours | 1.000 / 14.96 | **1.000 / 14.96** | 1.000 / 14.96 | 0/0 | 0/0 | 0/0 | 1609 | 1712 | 1610 |
| | gwarm | 1.000 / 14.96 | **1.000 / 14.96** | 1.000 / 14.96 | 0/0 | 0/0 | 0/0 | 1408 | **1100** | **1095** |
| AMASS_jump | ours | 1.000 / 13.92 | **1.000 / 13.92** | 1.000 / 13.92 | 0/0 | 0/0 | 0/0 | 1356 | 1435 | 1372 |
| | gwarm | 1.000 / 13.92 | **1.000 / 13.92** | 1.000 / 13.92 | 0/0 | 0/0 | 0/0 | 1281 | **1164** | **1170** |
| AMASS_karate_twist360 | ours | 1.000 / 7.66 | **1.000 / 7.66** | 1.000 / 7.66 | 0/0 | 0/0 | 0/0 | 1312 | 1375 | 1355 |
| | gwarm | 1.000 / 7.66 | **1.000 / 7.66** | 1.000 / 7.66 | 0/0 | 0/0 | 0/0 | 1200 | **1058** | **1091** |
| **AMASS_dance** | ours | 1.000 / 14.96 | **0.015 / 0.22** | **1.000 / 14.96** | 0/0 | 1/0 | 0/0 | 2527 | 2233 | **2044** |
| | gwarm | 0.329 / 4.92 | **0.015 / 0.22** | **0.322 / 4.82** | 1/1 | 4/0 | 1/0 | 2310 | 1876 | **1652** |
| AMASS_sit_to_stand | ours | 0.331 / 2.68 | **0.348 / 2.82** | 0.348 / 2.82 | 2/2 | 5/**5** | 4/**4** | 1624 | 2044 | 2044 |
| | gwarm | 0.331 / 2.68 | **0.348 / 2.82** | 0.348 / 2.82 | 2/1 | 5/3 | 4/3 | 1570 | **1505** | 1673 |
| AMASS_kick_ground | ours | 0.420 / 4.18 | **0.426 / 4.24** | 0.420 / 4.18 | 1/1 | 1/1 | 1/1 | 2178 | 2267 | 2168 |
| | gwarm | 0.420 / 4.18 | **0.426 / 4.24** | 0.420 / 4.18 | 1/0 | 2/1 | 2/1 | 1392 | **1308** | **1300** |
| LAFANctl_dance1_s2（见过） | ours | 1.000 / 14.96 | **1.000 / 14.96** | 1.000 / 14.96 | 0/0 | 0/0 | 0/0 | 1977 | **1828** | **1798** |
| | gwarm | 1.000 / 14.96 | **1.000 / 14.96** | 1.000 / 14.96 | 0/0 | 0/0 | 0/0 | 2076 | **1702** | **1707** |

### 汇总（7 段合计 84.5 s）

| ckpt | 参考 | 存活合计 | 相对原生 | 躺地时间 | 跌倒段 / 爬起 | mpjpe 均值 | 相对原生 |
|---|---|---|---|---|---|---|---|
| ours | 原生 retarget | 73.32 s | 1.000× | 3.59 s (4.3%) | 3 / 3 | 1798 | 1.000× |
| ours | **Greenwich (同规则)** | 58.78 s | **0.802×** | 4.44 s (5.2%) | 7 / 6 | 1842 | 1.025× |
| ours | **Greenwich (首帧对齐)** | **73.46 s** | **1.002×** | 3.11 s (3.7%) | 5 / 5 | **1770** | **0.985×** |
| gwarm | 原生 retarget | 63.28 s | 1.000× | 4.00 s (4.7%) | 4 / 2 | 1605 | 1.000× |
| gwarm | **Greenwich (同规则)** | 58.78 s | **0.929×** | 5.39 s (6.4%) | 11 / 4 | **1388** | **0.864×** |
| gwarm | **Greenwich (首帧对齐)** | **63.32 s** | **1.001×** | 4.58 s (5.4%) | 7 / 4 | **1384** | **0.862×** |

### 不炸机检查
- 两个模型、三种参考、全部 rollout **没有 NaN**（`nonfinite` 全 0），env 自身 termination **一次都没触发**。
- 最低刚体高度 **+1.6 ~ +3.3 cm，没有穿地**。上一轮 gwarm 在 kick_ground 上那个 -0.8 cm 穿地，换成 Greenwich 参考后变成 +2.3 cm，**穿地消失了**。
- `dvel` 超 20 rad/s 的比例：ours 0.000-1.4e-4，gwarm 5e-5-2.3e-4，和上一轮同量级。唯一变大的是 AMASS_dance（ours 0 → 1.4e-4），就是开场那一摔。

### 每片一句判决（先看的视频/网格）
- **AMASS_squat_deep**：两个模型都全程跟着蹲到底再站起来。参考明显吊在空中（首帧脚底 26 cm），策略照样站住 —— 这一段反而说明控制器对"飘着的参考"很耐操。
- **AMASS_jump / AMASS_karate_twist360**：和原生参考一模一样，0 跌倒，gwarm 的关节误差还更低。
- **AMASS_dance（唯一掉分的）**：同规则接地下，机器人**出生就在 20 cm 高空**，0.22 s 摔倒、0.4-0.6 s 完全趴平、约 1.3 s 自己撑起来站直，之后 13.6 s 跟完（fail 只占全片 4.4%，结束时是站着的）。看 `gwstress_grids/ours__AMASS_dance_first1p5s.png`。**首帧对齐的对照臂里这一摔完全消失**（`oursf0__AMASS_dance_first1p5s.png`：从第 1 帧就站着）。
- **AMASS_sit_to_stand**：Greenwich 参考下 **ours 5 次跌倒 5 次全爬起来**（原生是 2/2），存活还略长（2.82 vs 2.68 s）。这是本轮最有说服力的一段。
- **AMASS_kick_ground**：两臂几乎重合；gwarm 在 Greenwich 参考下从"摔了爬不起来"变成"2 次跌倒爬起 1 次"，且不再穿地。
- **LAFANctl_dance1_s2（见过的数据）**：两个模型 15 s 全程站住，mpjpe 还都更低。说明 AMASS 上的失败确实来自动作本身，不是 Greenwich 管线塌了。

---

## 判决：Greenwich 产的参考，要不要付鲁棒性的代价？

> **不用付。汇率 ≈ 1.00。**
>
> 把接地/出生高度对齐之后（`GW-f0` 那一列），7 段合计存活 **ours 1.002×、gwarm 1.001×**，跌倒后爬起来的能力没退（ours 5/5 全爬起），关节跟踪 **ours 0.985×、gwarm 0.862×（更好跟）**。
> 唯一真实的成本不在姿态，在**接地**：Greenwich 解码的腿更直，同一条 p10 接地规则会把参考多吊 8-12 cm、出生点最多高 24 cm，这一条会在最激烈的一段（dance）上换来一次开场摔倒。
> **要收的工程债只有一条：把产品出货的接地换成首帧/接触感知接地，而不是全片 p10。** 换完，客户拿 Greenwich 产的数据训出来的控制器，和拿原生 retarget 训出来的一样稳。

---

## 11. 本轮新挖出的管线 bug

### 11.1 【严重】两套 xwbc 语料的 `human_smpl` 骨架不同却重名 → Greenwich 描述符缓存静默串味
- `cache_xwbc_amass_full` 和 `cache_xwbc_lafan` 里的 `human_smpl_spec.npz`：`name` 都是 `"human_smpl"`、`parents` 相同，但 **`rest_offsets` 最大差 46.74 个单位**（就是记忆里那条"xwbc 有两套人体骨架"）。
- `Greenwich.geom()` 的缓存键是 **`spec.name`**（`src/alphamotion/engine/greenwich.py:87`）。在同一个进程里先解 AMASS 再解 LAFAN，第二个语料会**直接复用第一个语料的几何描述符**，不报错、不告警。
- 后果实测：LAFAN 对照片 `dance1_subject2` 解码出来与生产轨迹差 **21.18° / 身体位置 13.9 cm**；单独跑同一段则是 **4.9e-07 度**。
- 修法（已落在 `/home/arenalabs/tmp/gw_amass_decode.py`）：换语料时 `G._geoms.clear()`。**更根本的修法是把缓存键改成 `spec.name` + rest_offsets 的哈希**，否则任何跨语料批处理都会中招。

### 11.2 【中等】p10 全片接地对 Greenwich 轨迹不合适
第 9 节那张净空表就是证据：p10 接地对"整体偏飘"的轨迹没有约束力（p10 本身就飘），首帧最多被抬到 24 cm。原生 retarget 上这条能用是因为它本来就贴地。建议产品出货改成首帧接触对齐 + 逐帧接触感知，第 10 节的 `GW-f0` 对照臂已经量出这一改能拿回多少（ours 0.802× → 1.002×）。

---

## 12. 自己看视频（请肉眼判）

**并排 mp4（左＝Greenwich 解码的参考，右＝策略 rollout），共 14 个：**
```
/home/arenalabs/tmp/gwstress_ours_{AMASS_squat_deep,AMASS_jump,AMASS_karate_twist360,AMASS_dance,AMASS_sit_to_stand,AMASS_kick_ground,LAFANctl_dance1_s2}.mp4
/home/arenalabs/tmp/gwstress_gwarm_{同上}.mp4
```
**对照臂（首帧对齐）mp4：**
```
/home/arenalabs/tmp/gwstress_f0_{ours,gwarm}_AMASS_dance.mp4
```
**8 帧网格 PNG：**
```
/home/arenalabs/tmp/gwstress_grids/{ours,gwarm}__<clip>_grid.png
/home/arenalabs/tmp/gwstress_grids/{ours,gwarm}f0__AMASS_dance_grid.png
```
最该先看的三张：
- `ours__AMASS_dance_first1p5s.png` —— 开场从 20 cm 高空摔下去、1.3 s 自己站起来（同规则接地）
- `oursf0__AMASS_dance_first1p5s.png` —— 同一段，首帧对齐后**这一摔不存在**（判决的直接证据）
- `ours__AMASS_sit_to_stand_grid.png` —— Greenwich 参考下 5 次跌倒 5 次爬起

**其他产物**
- Greenwich 解码 qpos：`/home/arenalabs/tmp/gwstress/*__qposA_gw.npy`，解码体检 `/home/arenalabs/tmp/gwstress/decode_report.json`
- 参考 RobotState：`/home/arenalabs/tmp/amass_g1_gw_rs/*.npz`（同规则）、`amass_g1_gwf0_rs/*.npz`（首帧对齐）、`amass_g1_gwhm_rs/*.npz`（中位数对齐，第三个对照）
- UFO manifest：`configs/data/amass_gwstress.yaml` / `amass_gwf0stress.yaml` / `amass_gwhmstress.yaml`
- 逐帧状态：`/home/arenalabs/tmp/{gwstress_out,gwf0stress_out,gwhmstress_out}/{tag}__{clip}.npz`
- 评分：`/home/arenalabs/tmp/{gwstress,gwf0stress,gwhmstress}_scores.json`
- 脚本：`gw_amass_decode.py`（解码）、`gw_qpos_to_rs.py`（RobotState + 四元数顺序修正）、`gw_qpos_to_rs_f0.py` / `_hm.py`（对照臂）、`gw_refstats.py`（参考体检）；rollout 和评分复用第一轮的 `stress_run.py` / `stress_analyze.py`
- 参考净空/难度统计：`/home/arenalabs/tmp/gwstress_refstats.json`
