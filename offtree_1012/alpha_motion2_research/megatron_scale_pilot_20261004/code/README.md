# Megatron 单帧扩容 pilot：本地准备包

**当前未开始真实训练，训练入口需要匹配的真实 CUDA 配方验收记录。** ArenaLabs 从 2026-10-04 04:50 UTC 起被 Tailscale 标为离线，SSH 连接超时。本目录是可审查的训练与评估实现；合成数值检查不能替代真实数据批次、旧权重加载、GPU 训练或最终 checkpoint。

## 选定配置

| 项目 | 候选 |
|---|---|
| 主干 | 一期 DualTranslator，d256、8 heads、encoder 6 层、decoder 每流 6 个 cross+self 块 |
| 参数 | 主模型 25,743,428；另有 266,512 参数的 16 类训练辅助 judge；总计 26,009,940，547 个 state entries。冻结 Qwen 文本塔不加载进训练模型 |
| 离散表示 | position 4 slots / rotation 128 slots；单 stage，13 维，每维 32 levels |
| 文本 | 独立重算的固定版本 Qwen3-Embedding-0.6B；584×1024 冻结名称向量；整个 semantic projection 重新初始化 |
| 目标条件 | 每样本 0–2 个真实当前帧关节的 rot6 + position/reach；共享条件编码，接入两个 FSQ 后的 memory |
| 训练预算 | seed 0；batch 128；最多 3,000 steps；训练计时上限 1,800 秒；少于 1,000 steps 标为不足 |
| 优化器 | 原 AdamW：统一 LR 1e-4、weight decay 1e-4、CosineAnnealingLR；无 warmup；clip 1；GPU BF16，FK/损失 FP32 |
| 任务配比 | XWBC 55%，每步原 8 方向主分支；额外纯 SE3 分支概率 25%。UniMate 45%，同体 full/masked/sparse-only 为 40/35/25%；条件 dropout 35% |
| 保存 | 每 500 steps checkpoint + 小评估；最终完整评估、可回读逐关节误差与配对图 |

完整配置见 `candidate_config.json`。新增层以 residual identity 开始，旧兼容张量 warm-start；新增 Qwen projection 和条件模块使整模型不与旧模型数值等价。拓扑 hop bias、双 FSQ、共享目标解码器和单帧骨干均保留。没有 flow、diffusion、surface、时序或平移 DOF 输出头。

旧配方依据真实 `L_M20k_0903/config.json` 原始读取记录，恢复字节的 SHA 与独立审计相同；不是从 CLI 默认值猜测。恢复后的 XWBC 主分支保留以下单帧项：

- 八个有向组合（含同体）从 16×16 中无放回抽取，共用一批真实 source rows；旋转角/30 + 5×FK/reach 以及有效地面项取方向平均。
- InfoNCE 0.5、tau 0.1、每步 6 体的全部 15 对；重复 source row 只在 InfoNCE 内稳定去重，避免同一动作成为负例。judge 与 cycle 保留原 batch 权重。
- 无条件 decode→reencode 一致性 1.0、概率 0.3、前半批至少 8 行；回编码原 raw hpos，不换成 FK，也不 detach 原 latent。
- 16 类辅助 judge 使用原 one-hot BCE、grad 模式、权重 0.1。旧 label 顺序尚未证实，因此整个辅助头重新初始化，并保存当前已验证 16 体映射；不复用错误类别语义。共享目标 decoder 不新增机器人专用参数。
- 穿透 200、接触几何 1、有效 root height 3；从真实 root gap、contact soft/trust、阈值和旋转后的 sole offset 计算。未补齐足底来源 seal 会被 loader 拒绝，不用 joint center 替代足底。
- 原 mask_prob 0.3 与 rand/ee/root/sparse/cask/cgive 六族。root 保持可见；接触距离是独立已知输入，被遮姿态的 joint 仍可接收它；任意有遮挡的行绕过 contact dropout。额外纯 SE3 分支则遮住全部源 pose 且 contact=None。

新增条件/数据/深度、源组均衡采样、batch 与时长预算仍是显式差异。UniMate 不加入跨体 InfoNCE、身体 judge 或未知 metric 地面监督。旧 world-slip 2 / hpair 0.1 需要连续相邻帧和世界根轨迹，本轮稀疏单帧不具备这些观测，明确关闭。逐项比较见 `recipe_review.json` / `recipe_diff.md`；不能将混合改动单种子结果归因于某个模块。

## 数据与支持边界

- XWBC：选定 30,322 个真实共享 source-frame rows，训练 24,744 / 评估 5,578，映射到 16 个已有身体；同一行表全体共享。16 体展开是 485,152 个 body-frame 实例，不是新增 485,152 段独立动作。断联前只确认 9/16 体抽取完成，当前最终状态未知；loader 不接受部分结果。
- AMASS 评估来自旧训练明确排除的源动作；LAFAN 留出组曾被旧 warm-start 看过，报告单列 pretrain-seen。相同 source group 不跨 train/eval；不能用随机 frame split 装作动作留出。
- UniML3D：218 个原生 NPZ、33,540,276 bytes 已下载并校验，另有 conditioning。10 个选择身体包括 Mixamo、蛇、机械链、上肢、两四足、两双足、鸟翼和蜘蛛。选定数量不等于实际入训数量；全部真实帧 FK 审计、payload 与最终样本数尚待远端执行。
- UniMate 仅同体重建、mask completion 和 sparse completion。caption、帧号或相似动作不构成跨体配对。源名、内容 hash、原生数组 hash 合并同源组；4 个预定整身体留出，重复的蛇文件只有 1 个独立组。未知跨数据集原始来源不能被本契约证明无重叠，报告明确这一限制。
- 输出表示只支持**全局旋转 + 固定 rest offsets 的 FK**。prismatic、动态骨长、mimic 执行与真实夹爪开合不在接口中。adapter 全帧 FK 不通过的文件隔离，manifest 报 included / duplicate / quarantined 数和原因；机械臂类别不是夹爪或灵巧手映射验证。
- XWBC 原始长度是 cm；UniMate 是 canonical body units，不能写成 cm/m。UniMate descriptor 的 ×100 是数值尺度，`metric_length_valid=false`；轴、限位、contact、ground、root-height 未知都保留 false。
- 位置损失和评估只读预测旋转经过 FK 的位置。旧 `hpos` 仅 root slot 第 0 通道监督有效 root height（米）；其余 hpos 槽不当作 XYZ。
- 条件帧是**root-centered、canonical Y-up，保留原 canonical 朝向**；没有默认 world/object anchor，也不是再乘 root 逆旋转的 body frame。
- 有明确手/腕名字和上肢链证据才用相应锚点；其他身体用真正拓扑叶端点并标 generic。匿名 H2 不猜手名。每样本最多两个目标；纯条件任务遮掉所有源 pose（含 root）且 source contact=None。未知/隐藏 target pose 不进入模型。

## 已完成的离线验证

本次配方修订共 101 项离线测试通过，结果与源码 SHA 见 `offline_verification.json`；早期 59 项只验证旧简化接口，不能充当完整旧配方验收。测试包括独立 SciPy FK、动态 offset 隔离、分组/整身体留出、NaN/Inf padding 与梯度、隐藏 GT 扰动不影响输出、383-key 合成旧模型 schema、完整 manifest/哈希拒绝、真实格式 object joint-name/DOF 元数据、同一批评估行与同一选中关节集合。三步小模型 AdamW 检查只用合成数据；没有训练真实 pilot。

旧 G1 本地 trace 的端点识别还核对到 `left_rubber_hand` / `right_rubber_hand`。这只确认骨架关节选择，不证明 palm surface、硬件工具坐标或灵巧手能力。

## 远端恢复后的执行顺序

远端实验目录为：

`/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/megatron_scale_pilot_20261004`

1. 先核对下载、接续队列、现有 XWBC 抽取及本实验训练的 PID、`/proc` start ticks、argv、boot ID 和退出记录。**连接断开不是进程退出证据。** 已有抽取/队列仍在运行就接续观察，不重开同任务；完成的 run 直接验收结果，不启动第二份训练。
2. 将本包源码放到实验的 `code/`。保留原始项目、权重、缓存和已运行的 `data_xwbc/extract_xwbc_pilot.py`；本包的 extractor 是审查副本，不能覆盖可能仍运行的脚本。`legacy/` 三文件必须与已审计远端原源码核对 SHA。
3. 若旧抽取已经完整退出，先验收 `data_xwbc/manifest.json`、`checks.json`、16 个 payload 和 row table。若缺静态 seal，用 `seal_xwbc_static.py` 对照实际旧 `cache_lafan_v3` 的 spec/DOF 后补当前 seal；再运行 `seal_xwbc_static.py --backfill-ground`，传入 `--data-root`、按旧配置顺序的两个 `--source-caches`、原 `--legacy-source-dir` 和 `--sole-json`，核对两个 cache 的 contact header、sole geometry、soft/hard/trust/root gap。补齐会暂存新的 derived NPZ 后原子替换本实验 payload，先按已有 payload 的未压缩总量估算临时空间，并纳入组合剩余下载 + 16 GiB 保留 + 2 GiB pilot 容量预算。补齐真实来源字段后才通过 loader，不重启已有抽取。参数 `--source-manifest-sha256` 应由原审计的 `chosen_split.json` 来源 hash 与实际源 manifest 独立相等核对后填入；新观察 hash 不冒充历史 hash。
4. 确认没有同属本任务的 UniMate adapter 在运行，再用已校验的 218 个文件执行 adapter。无需重新下载。
5. 读取新鲜组合容量，覆盖原主下载 + 接续 BONES 全部固定计划。保留 15 GiB 余量 + 1 GiB 下载开销 + 2 GiB pilot allocation；归档解压另估。训练每 100 steps 重新调用只读 cache 容量计数。
6. 跑真实小批次验证，核对确实加载旧 checkpoint、固定 Qwen cache 和所有数据。检查 CUDA BF16、实际 batch128 容量与吞吐；完整执行 8 主方向、强制一次 cycle、6 体 InfoNCE/judge 和地面损失的有限前后向，再写验收 receipt。训练运行持有 `flock`，精确身份和退出状态落盘。

在远端实验目录，以下是入口（`describe` 不加载真实数据或启动训练）：

```bash
/home/arenalabs/miniconda3/envs/ardy/bin/python code/train_pilot.py --config code/candidate_config.json --action describe
/home/arenalabs/miniconda3/envs/ardy/bin/python code/pilot_adapter.py --root data_audit
/home/arenalabs/miniconda3/envs/ardy/bin/python code/remote_state.py --output preflight/current_remote_state.json
/home/arenalabs/miniconda3/envs/ardy/bin/python code/train_pilot.py --config code/candidate_config.json --root . --capacity-receipt preflight/current_remote_state.json --action validate
```

`validate` 无 optimizer step，成功退出为 `recipe_validated_no_training`。它绑定 config、全部 Python 源码、实际旧 config/checkpoint、Qwen、数据与原生足底契约 hash，并保存真实全损失 CUDA 前后向报告。仅把 config 中 train_allowed 改成 true 无效。

重新核对进程所有权并刷新容量 receipt 后，训练需明确传入匹配且未变更的验收目录：`--action train --recipe-validation-run runs/<已完成的validate目录>`。代码、配置、来源文件或数据改变就拒绝旧 receipt；训练仍会重跑实际批次检查。随后完成 baseline、initial、训练、final、配对分析与图，不以“已派发”当完成。不同 run 写入独立时间戳目录，未实现自动 resume。

## 结果验收

`runs/<timestamp>_train_seed0/` 应包含 config、源码/hash、进程身份、真实 data audit、weight transfer、real batch preflight、reference/initial/final 评估、train.jsonl、checkpoint+SHA、配对统计与图、退出状态。基线使用原 checkpoint + 现有 SigLIP768 cache，在相同 XWBC split/FK 口径比较；它不支持新 target-SE3 接口，未准备 UniMate 新名称 SigLIP，因此不伪造那一项基线。

定量 gate：至少 1,000 steps；合法 sparse-only selected joint 平移与旋转误差相对无 SE3 各改善 ≥10%；旧域 full-body 对原基线平均退化 ≤10%；整新身体对初始化平均退化 ≤10%；随后人工查看配对图。误差保存到逐行/逐关节 NPZ，用有条件样本的同一 selected mask 从无条件结果取对应误差；不能用不同集合的平均数替代。训练完成、资源完整、模型效果、仿真和真机验证分别报告。

新手部图稿 ZIP 和本次断联监控记录仍待网络恢复后归档；具体清单见 `pending_remote_work.json`。原下载主进程和接续队列保持原有所有权，本包不修改或停止它们。
