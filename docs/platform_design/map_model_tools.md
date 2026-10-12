# 模型侧工具盘点：训练 / 实时可视化 / 模型评估 / WBC 控制

> 盘点日期 2026-10-07。只读盘点，没有移动、删除或修改任何东西。大小用 `du` 测得（带超时），日期取文件夹或主文件的最后修改时间。
> 状态：**在用** = 最近一个月有改动，或服务正在跑；**过时** = 已被新版本取代，或依赖已经坏了；**归档** = 只当证据或参考保留，不再改。
> 下文 `RW` = `/home/arenalabs/Desktop/"be water, robot"`（文件夹名里带双引号；`~/alpha-motion-workspace`、`~/.alphamotion_workspace` 和 `~/alphamotion2_runtime` 都是指向它或它子目录的软链接）。
> 「建议位置」用的是第 5 节提出的统一布局 `<ROOT>/…`，根目录叫什么名字由 lloyd 定。

## 1. 一句话结论

- 模型侧工具散在 **9 个顶层位置**：`RW/alpha_motion2/research/…`、`RW/pose_translator_design/exp_topo`、`~/Desktop/AlphaMotion`、`~/Desktop/arena-datagrid`、`~/tmp/greenwich_workbench_0911`、`~/Desktop/greenwich_eval_20260826`、`/var/tmp/alphamotion/{studio,topo_exp,am2exp,beyondmimic,ctrl_verify}`、`~/alphamotion_ctrl`、`~/reference_code/{ufo,bfm_zero,beyondmimic}`。另外 `~/tmp` 下还有几十个一次性输出目录。
- 同一份代码有好几份副本。判官代码有 **3 份**（exp_topo 原版、arena-datagrid 打包版、architecture_audit 快照）；产品仓库有 **3 份**；Greenwich SDK 有 **2 份**；控制仓库有 **2 份**。
- 真正值得升格为统一工具链“一等工具”的有 **7 个**（见第 3 节）。其余的要么并进这 7 个，要么归档。
- 现在没有模型训练在跑。GPU 上跑的是数据工厂的任务。有 3 个服务常驻：motion lab :2334、rollout gallery :8891、greenwich mobile :8899，都监听 `0.0.0.0`。另有 1 个服务在**崩溃重启循环**里（见第 6 节）。

## 2. 总表

### 2.1 训练（Training）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被依赖 | 建议位置 |
|---|---|---|---|---|---|---|---|
| `RW/alpha_motion2/research/alphamotion_scaling_20261004/orchestration/`（`train_scaling.py`、`scaling_data.py`、`encode_features.py`、`six_run_plan.json`、`storage_budget.json`） | AM2 六组从零训练（A/B × S/M/L，seed0）：单一 flock 持有者、证据门、算力记账 | `runs/{A,B}_{S,M,L}_seed0/`（checkpoint_step*.pt、train.jsonl、result.json） | 代码 252K；runs 9.3G；continuation_v3 4.4G | 10-04 | 在用（当前 AM2 主线） | 依赖 megatron_scale_pilot 的 recipe_* 和 pilot_model；被 am2exp_v1/train/train_arm.py import | `<ROOT>/toolchain/06_train/am2/`（代码）；`<ROOT>/runs/am2/`（runs） |
| `RW/alpha_motion2/research/megatron_scale_pilot_20261004/code/` | AM2 训练配方库：recipe_training / recipe_ground / recipe_auxiliary / pilot_model / pilot_data | 预检和审阅 json、pilot runs | 整个目录 1.1G | 10-04 | 在用（库） | 被 train_scaling 和 am2exp_v1/model/_sealed.py 只读 import | `<ROOT>/toolchain/06_train/am2/recipe/`（冻结） |
| `RW/alpha_motion2/research/megatro_scale_pilot_20261004/`（目录名少了个 n） | 只剩一个 qwen_audit | — | 40K | 10-03 | 过时（拼错的重复目录） | 无 | `<ROOT>/archive/` |
| `RW/…/alphamotion_scaling_20261004/am2exp_v1/{train,ops,common}` | 按“臂”训练（`train_arm.py` + `arms.json`），并逐步核对采样哈希。ops 层负责：GPU 租约 `lease.py`、磁盘守卫 `guard.py`、回执 `receipts.py`、顺序队列 `queue.py`、双胞胎门 `gates.py`、人工批准文件 `approvals/<stage>.ok` | `/var/tmp/alphamotion/am2exp/{runs,logs}` | 代码 1.4M（am2exp_v1 全部）；输出 3.0G | 10-04 | 在用 | 依赖 train_scaling（import 为 original）；ops 被 eval、render 和 integration 共用 | ops → `<ROOT>/services/jobs/`（统一作业层的核心）；train_arm → `<ROOT>/toolchain/06_train/am2/` |
| `RW/pose_translator_design/exp_topo/`（`train_dual.py`、`dual_common.py`、`model_dual.py`、`ladder*.sh`、`masking.py`、`contact_label.py`） | Greenwich v1/v2 训练（双 FSQ 跨具身编解码）；Chinchilla 阶梯 S20k/M20k/L20k 产出 v0.2 的 Bumblebee/Megatron/Prime | `/var/tmp/alphamotion/topo_exp/runs/`（823 个 run） | 代码 8.3M；topo_exp 160G（runs 16G，各种 cache 约 140G） | 09-09 | 归档且冻结（v0.2 已发布；要复现时还得用它） | 依赖 cache_amass_lconv_tfix（52G）、cache_xwbc_amass_full（61G）等；被产品仓库 `verify_v2_port.py` 逐比特对齐 | `<ROOT>/toolchain/06_train/greenwich_legacy/`；cache 移到冷存储 |
| `RW/alpha_motion2/{models,training,evaluation,modules,heads,losses,geometry}` | 五月的 AlphaMotion2 脚手架（MAE、关键帧、motiongen 三阶段） | 无现存产物 | 每个都小于 100K | 05-31 | 过时 | 无人引用 | `<ROOT>/archive/am2_scaffold_0531/` |
| `RW/alpha_motion2/research/{architecture_audit_20261003,transformer_review_20261003,training_recipe_audit_20261004,embodied_case_20261003,hand_surface_case_20261004}` | 10 月初的架构、配方审计和案例研究；architecture_audit 里有 AlphaMotion1 源码快照（判官的第 3 份副本） | 审计报告、快照 | 42M / 21M / 60K / 106M / 53M | 10-03~04 | 归档 | 无 | `<ROOT>/archive/audits_2026-10/` |
| `~/Desktop/arena-datagrid/packages/arena/arena/services/gpu_batcher.py` + `workers/` + `db/store.py` + `cli/` | GPU 批推理服务（:8932）、带租约的作业队列和 worker 池。作业种类：infer / refine / render_review / manip_clip / loco_clip / pipeline；eval 和 export 还没实现 | 作业库和产物 | 包 900K | 09-13（最后一次 commit） | 在用（半成品；`arena serve api` 故意拒绝启动） | 依赖产品 `alphamotion` 包和 `arena_research` | 并进 `<ROOT>/services/jobs/` 和 `<ROOT>/services/gpu/`（与 am2exp ops 合并） |

### 2.2 实时可视化与渲染（Live viz / Render）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被依赖 | 建议位置 |
|---|---|---|---|---|---|---|---|
| `~/Desktop/AlphaMotion/src/alphamotion/viz/`（`live.py` LiveViewer、`viewer.py`、`video.py`、`preview.py`）+ `service/app.py`（FastAPI :7860） | 产品级实时查看器：一个常驻 viser 画布，每次生成就地换场景；有 transport、gizmo、对比同步、库预览、timeline、atlas 等约 40 个 API | 浏览器视口、mp4、webp 预览 | 整个仓库 1.5G（大部分是 weights_staging） | 09-09（v0.2） | 在用（产品；目前没在跑） | 依赖 Greenwich 权重、Atlas、库；被 arena-datagrid 和 greenwich_workbench 复制 | **一等工具 V1** → `<ROOT>/toolchain/07_viz/live/`（抽成库）+ `<ROOT>/services/api` 的 viewer 路由 |
| `RW/…/am2exp_v1/render/`（`triptych_e1.py`、`triptych_e2.py`、`grid12.py`、`scene3d.py`、`robot_model.py`、`contact.py`） | 标准评测视频：三联「人体源 \| 对照臂 \| 本臂」，1080p，真实时间戳；12 帧网格逐帧精确解码；按区域着色接触；画厂商 MJCF 网格；原子写入 | `triptych.mp4`、`grid12.png`、`poster.jpg`、`manifest.json`（在 `/var/tmp/alphamotion/am2exp/videos`） | 代码在 am2exp_v1 1.4M 里；videos 4.5M | 10-04 | 在用（有测试） | 依赖 rollout_tools/encode.py（检查 PTS）和 judge 的绑定；被 integration/smoke_e2e 调用 | **一等工具 V2** → `<ROOT>/toolchain/07_viz/render/` |
| `RW/…/alphamotion_scaling_20261004/rollout_tools/`（`prepare.py`、`render.py`、`encode.py`、`contracts.py`、`scene.py`、`g1.py`） | 离线“输入包 → 网格回放”：输入包带版本、可脱机重渲；编码时校验逐帧 PTS；容量守卫 | `reports/rollout_mesh_v1/`（24 段 Adroit/Franka + 1 段 G1、index.html、QA 回执） | 152K；报告在 reports 325M 里 | 10-04 | 在用 | 被 am2exp render 用 encode.py | 并进 V2：`<ROOT>/toolchain/07_viz/render/bundle/` |
| `RW/pose_translator_design/exp_topo/showcase_render.py`（arena_research 里有一份改了路径的副本） | 对外展示：一段人体动作，N 台机器人同场同步；SMPL-X 网格、接触箭头 | 1080p mp4 | 30K | 09-03 | 在用（展示用） | 依赖 AlphaMotion/src、AM_CACHE 等 | `<ROOT>/toolchain/07_viz/showcase/` |
| `exp_topo/crossrig_amass.py`、`slipvid_amass.py`、`pilot_decode_viz*.py`、`greenwich_carnival.py`、`p3_*_viz.py`、`render_v5.py` 等约 20 个渲染脚本 | 各个实验的一次性对比视频 | `~/tmp/crossrig*`、`cmp34_*`、`y*_odo` 等（约 370M）；`RW/pose_translator_design/R4_VIZ`（1.1G） | 各 5~30K | 08-02~09-02 | 归档（功能被 V2 和 showcase 覆盖） | 依赖 topo_exp cache | `<ROOT>/archive/greenwich_renders/`；视频 → `<ROOT>/archive/videos/` |
| `/var/tmp/alphamotion/studio/`（`alphamotion_studio_app.py` :8097 + 6 个 worker）+ `/var/tmp/alphamotion/studio_app.py`（更早的原型） | 0814 产品链：prompt/视频 → GEM → Greenwich → Equator → rollout mp4。worker 协议是“参数进、文件出、末行 JSON” | `results/` 下的 trace npz 和 mp4 | 14M | 08-14（日志到 09-08） | 过时（作业表只在内存里，重启就丢；已被产品 service 取代；没在跑） | 依赖 any_body env、topo_exp/runs/eval_current_0814 | worker 协议可以借鉴；代码 → `<ROOT>/archive/studio_0814/` |
| `RW/alpha_motion2/scripts/alphamotion_motion_lab.py`（systemd `alphamotion-motion-lab` :2334） | ARDY 轨迹的 viser 浏览实验室（CoreSkeleton27/G1/SOMA） | 实时页面 | 15K | 07-21 | 在用（服务常驻），但内容只有 ARDY preflight 20 段 | 依赖 `experiment/ardy`；用 RSS 约 745MB | 并进 V1 后停掉；→ `<ROOT>/archive/` |
| `RW/alpha_motion2/web/verified_rollout_gallery.py`（systemd `alphamotion-rollout-gallery` :8891） | 手机端静态画廊：20 段经 QA 的 G1 rollout | 页面 | 10K（web 整体 147M） | 07-21 | 在用（常驻），内容是 7 月的老东西 | 读 `demo/alphamotion_verified_g1_atlas_20/manifest.json` | 由统一 Web UI 的“画廊”页取代 → 归档 |
| `~/Desktop/greenwich_eval_20260826/`（`mobile_server.py`，systemd `greenwich-mobile` :8899；`run_greenwich_eval.py`、`render_greenwich_rollouts.py`、`qa_videos.py`） | 0826 Greenwich 评测证据画廊（两段视频 + 中文报告），支持字节范围请求 | `artifacts/videos`、`qa` | 17M | 08-26 | 在用（常驻），内容停在 0826 | 无 | 同上，归档；字节范围视频服务的写法并进 `<ROOT>/services/api` |
| `RW/alpha_motion2/web/{app.py,sonic_app.py,sonic_live_app.py,ardy_isaac_live_app.py,investor_video_gallery.py,kimodo_g1_worker.py}` + `RW/alpha_motion2/live/` | 7 月 ARDY/Kimodo + SONIC + Isaac 的实时控制演示、投资人画廊 | 演示 | web 147M；live 844K | 07-15~21 | 过时（`ardy_isaac_live_app` 的依赖已经不在了，见第 6 节） | 依赖 `~/Desktop/GR00T-WholeBodyControl`（已不存在）和 Isaac Lab（已坏） | `<ROOT>/archive/am2_demo_0721/` |
| `exp_topo/viser_mesh_viewer.py`（:8093）、`exp_topo/lab_server.py`（MOTION LAB :8114） | 8 月初的 viser 查看器：G1 网格 GT 和模型对比；时序模型补洞实验室（可以按段选“真实/模型填”） | 实时页面 | 6K / 15K | 08-03 / 08-08 | 归档 | 依赖 topo_exp cache | lab_server 的“分段选补全 + 多臂同场对比”交互要写进 V1 的需求；代码归档 |
| `~/motion_audit_web/`（FastAPI :8765 + React/Vite/Three.js） | 数据侧人工审计工作台：人体表面点、真实 MJCF、物体；UMR 原始和严格修正对比；判决追加到 `umr_factory_data/audit/decisions.jsonl`。已有 `.mcp-venv` | 页面、判决日志 | 3.2G（大部分是 cache/arrays） | 10-07 | 在用（数据侧，归数据盘点） | 依赖 UMR env 和工厂 NPZ | 统一 Web UI 的底座候选 → `<ROOT>/services/web/`；加一条“模型输出”通道就能审模型 |
| `~/Desktop/arena-datagrid/apps/datagrid/`（Electron + Vite） | Data Grid 桌面前端 | — | 62M | 09-11 | 半成品 | 依赖 `arena serve api`（没接上） | 和 motion_audit_web 二选一，交给 Codex 实现统一 Web UI |
| `~/Desktop/mingrun/benchmark_motion_viewer.py`、`~/Desktop/rot/` | 组员的 Kimodo/ARDY 基准查看器；ROT 机器人操作终端（5 月） | — | — | 08-05 / 05-09 | 外围项目 | — | 不并入，只在索引里登记 |

### 2.3 模型评估与判官（Model evaluation / Judges）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被依赖 | 建议位置 |
|---|---|---|---|---|---|---|---|
| `RW/…/am2exp_v1/judge/`（`ground_judge.py`、`pose_mjcf.py`、`regions.py`、`ruler_gate.py`）= **判官 J1** | 独立几何判官：在**厂商 MJCF 碰撞几何**上量 15 区域最低点、穿地、悬空、自穿；不读训练损失和模型的接触头（有测试守着这一点）。15 台机器人绑定 0 近似 | `/var/tmp/alphamotion/am2exp/judge/{bindings_v1,ruler_v1,stage0_report.md}` | 判官输出 4.0M | 10-04 | 在用（有测试） | 依赖 exp_topo/prep_gmr_cache.build_merge（构建时只读）；被 eval/score_e1、render 使用 | **一等工具 E1** → `<ROOT>/toolchain/08_model_eval/geometry_judge/` |
| `~/umr_gpu/eval/`（`judge.py`、`rejudge.py`、`render_compare.py`、`run_parity.py`） | UMR 的独立 CPU 判官：CPU UMR、GPU UMR 和 GPU strict 共用一把尺。4096 稠密点 + 碰撞几何；地面/自身/物体穿透；人→机器人接触一致；抬起一致；接触点滑移 | `results/parity/*.series.npz`、汇总 | 1004K（整个 umr_gpu） | 10-07 | 在用（重定向侧） | 依赖 MuJoCo 和 numpy；被重定向侧对拍用 | 与 J1 合并成同一个“几何判官”包：`<ROOT>/toolchain/08_model_eval/geometry_judge/`（数据和模型共用） |
| `RW/…/am2exp_v1/eval/`（`cases.py`、`run_e1_eval.py`、`score_e1.py`、`stats.py`、`stage0.py`） | 冻结病例 → CPU 推理 → J1 打分 → 按窗口聚类的配对符号检验和置换检验 → `decide()`（把冻结的决策规则写成代码） | `cases_e1_v1.json`（带 sha）、`scores_j1.jsonl`、stage0 报告 | 在 am2exp eval 输出 2.9G 里 | 10-04 | 在用 | 依赖 J1、model registry | **一等工具 E5（统计层）** → `<ROOT>/toolchain/08_model_eval/protocol/` |
| `RW/…/am2exp_v1/e2/closure_judge.py` = **判官 J2**（以及 `run_e2_eval.py`、`opposition.py`） | 手 ↔ 夹爪的抓握闭合判官：握住、张开、压碎三档；BA；按 episode bootstrap | `/var/tmp/alphamotion/am2exp/eval/e2` | 小 | 10-04 | 在用 | 只 import numpy、mujoco | `<ROOT>/toolchain/08_model_eval/manip_judge/` |
| `RW/…/alphamotion_scaling_20261004/evaluation*/`（`eval_engine.py`、`run_scaling_eval.py`、`surface_metrics.py`、`trajectory_refinement.py`、`dof_protocol.py`；p1、phase2_v2） | AM2 六组运动学评测：以片段为单位、和架构无关；P0/P1 关节精修；表面块对比 | `results/`、`reports/*` | 3.7G + 431M + 1.8G | 10-04 | 在用（结果已交付，不要重跑） | 依赖 runs 的 checkpoint | 引擎 → `<ROOT>/toolchain/08_model_eval/kinematic/`；结果 → `<ROOT>/evals/am2_scaling_20261004/` |
| `exp_topo/wbc_ready_audit.py`（+ `wbc_ready_report.py`） | WBC 可训性验收：600 条语料外片段，四张考卷（姿态、脚滑、接触标签、物理自洽），尺子 = GMR 自身的 p95 | json 报告 | 24K | 09-03 | 冻结。已移植到 `arena/eval/locomotion.py` | 依赖 tfix cache 和标签目录 | 研究原版 → `<ROOT>/toolchain/08_model_eval/reference_impl/`；正式实现用 arena 版 |
| `exp_topo/x2_phys.py` | 不需要真值的物理判官：根高抖动、姿态抖动、脚滑、穿地；尺子 = 人体源自己 | `~/tmp/x2phys/*.txt` | 17K | 09-03 | 冻结。已移植到 arena | 同上 | 同上 |
| `exp_topo/slip_audit.py`、`ground_metrics.py`、`contact_qc_v2.py`、`gauge_check.py` | 五条臂同一把尺的脚滑审计；接地四口径（touch/pen/IoU/tuck）；接触质检；gauge 自检 | json、npz | 20K / 4K / 33K / 6K | 09-01~03 | 冻结 | 同上 | 同上；指标并进 locomotion 标准 |
| `~/Desktop/arena-datagrid/packages/arena/arena/eval/locomotion.py` + `configs/standards/locomotion.v1.json` + `eval/visual.py`（RobotRenderer） | **把上面这些研究判官移植成逐片段实现**，配一份冻结标准（gates：指标、尺子、阈值）；缺材料时报 `not_evaluated`，绝不默默算通过 | 每段一行 verdict、12 帧网格和 mp4 | arena 900K；configs 很小 | 09-11~13 | 在用（作业种类 eval 还没接进 worker） | 依赖 arena_research（判官副本）和产品包 | **一等工具 E3** → `<ROOT>/toolchain/08_model_eval/standards/`（标准文件）+ `…/locomotion/` |
| `~/Desktop/arena-datagrid/packages/arena_research/` | 研究判官和公共代码的“可 import 打包版”（只改了 sys.path 那一行） | — | 1008K | 09-11 | 在用（arena 依赖它） | 被 arena.eval import | 和 reference_impl 合成一份，删掉 exp_topo 那份副本的“活跃”身份 |
| `exp_topo/v2_judge.py`（`v2_windows*.json`、`v2_reach_cache.json`） | Greenwich-v2 回归考卷：固定窗口，幅度比、上肢 FK 误差、腿角护栏、方言逐维一致率 | `topo_exp/runs/*/v2_judge.json`（20+ 个） | 18K | 08-31 | 冻结（编解码器回归用） | 依赖 train_dual 的 config | `<ROOT>/toolchain/08_model_eval/codec_regression/` |
| `exp_topo/spatial5.py` + 技能 `RW/.claude/skills/{spatial-eval,motion-eval}/SKILL.md` | Greenwich 五项验收（家族零样本、GT 偏差、四档 SE3、机器人→机器人、嘉年华）；六项评估纪律（数值、视觉、统计、熵、跨具身、动作泛化） | `R4_VIZ/mp4/spatial5/*`、`runs/SPATIAL5_*.txt` | 20K；技能 8K | 09-01 / 08-10 | 协议有效，脚本冻结 | 依赖 topo_exp | 协议写成 `<ROOT>/toolchain/08_model_eval/protocol/acceptance_*.yaml`，让 MCP 的 `run_eval` 照着跑 |
| `exp_topo/refiner_suite/`（`eval_suite.py`、`profiles.py`、`manip_set.json`） | 精修器评测套件（钉手、钉脚各档） | 汇总 json | 小 | 09-09 | 冻结（精修器已叫停，当对照臂用） | 依赖产品 refiner | `<ROOT>/toolchain/08_model_eval/refiner/` |
| `exp_topo/eval_rollout.py`、`physeval.py`、`eval_v5.py`、`n3_eval_*.py`、`exam_*.py`、`bridge_*eval*.py` 等约 40 个 | 8 月时序模型的对决和考试（AR-LM 对扩散、桥接、补洞） | `topo_exp/runs/*` | 各 4~43K | 08-02~14 | 归档 | 依赖旧 tokenizer | `<ROOT>/archive/greenwich_exams_0802_0814/` |
| `~/Desktop/arena-datagrid/packages/greenwich_motion_sdk/src/greenwich_motion_sdk/judge.py`（`~/tmp/greenwich_workbench_0911/greenwich-motion-toolkit` 里有一份一模一样的） | 读 checkpoint 里的潜空间 JudgeHead 打分（打的是精修前的码，**不是**物理有效性） | json | SDK 7.9M | 09-11 | 低优先级 | 依赖 Greenwich 权重 | `<ROOT>/toolchain/08_model_eval/latent_critic/`，标注“非物理指标” |
| `~/Desktop/AlphaMotion/src/alphamotion/service/quality.py`（`release_passed`）+ `scripts/{product_acceptance,browser_acceptance}.py` + `alphamotion eval` → `docs/BENCHMARK.md` | 产品发布门：能不能进公开 Atlas；产品和浏览器验收 | `artifacts/browser_acceptance*`、`product_acceptance.json` | 小 | 08-18~09-09 | 在用（产品） | 产品服务 | `<ROOT>/toolchain/08_model_eval/release_gate/` |
| `~/umr_factory_code/scripts/render_check.py` | 重定向 QA：12 帧网格「人体槽 \| UMR \| 后处理」，带接触和穿透颜色 | png、mp4 | 4K | 10-04 | 在用（重定向侧） | UMR | 和 V2 的 grid12 合并 → `<ROOT>/toolchain/07_viz/render/` |
| `RW/Greenwich_G1_18_Paired_Benchmark_Review_20260920/` | 对外评审包（HTML 报告 + 5 段 mp4；从 Mac 拷来，带 .DS_Store） | 交付物 | 247M | 09-24 | 归档（交付物） | — | `<ROOT>/deliverables/2026-09-20_g1_18_paired/` |

### 2.4 WBC / 控制（用跟踪率评估参考动作）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被依赖 | 建议位置 |
|---|---|---|---|---|---|---|---|
| `~/alphamotion_ctrl/`（`amctrl/` 有 G1 地面、GR3、X2、H2 任务，`horizon/` 地平线课程，`token/` 意图码环境，`rl/` 有 transformer actor；`scripts/` 约 60 个：`evaluate_local.py`、`horizon_render.py`、`csv_to_npz_local.py`、`foot_audit.py`、`h2_*` 等） | mjlab（MuJoCo-Warp）上的 BeyondMimic 跟踪训练和评测。用来回答“这份参考动作能不能训出跟踪控制器” | `logs/rsl_rl/*`、`logs/eval_*.json`、mp4；`motions/*.npz` | 29G（`.git` 14G、motions 8.6G、`.venv` 5.8G、logs 891M、代码 688K） | 08-29（commit）；H2 脚本 09-16 | 在用（但默认不开 RL 训练，受电费预算约束） | 依赖 mjlab==1.6.0；参考 CSV 来自 `exp_topo/ctrl_verify/g1_tracks.py` | **一等工具 E7（控制闭环评测）** → `<ROOT>/toolchain/09_control/mjlab/`；motions 和 logs → `<ROOT>/runs/control/`；`.git` 要瘦身 |
| `RW/pose_translator_design/exp_topo/ctrl_verify/` + `/var/tmp/alphamotion/ctrl_verify/` | 导出参考轨迹（g1/gr3/adam tracks）、x2 pool、refiner_a；渲染 qpos 对比 | CSV、npz、mp4 | 代码小；输出 2.9G | 08-30 / 09-03 | 冻结 | 被 alphamotion_ctrl 依赖（导出链） | 导出器 → `<ROOT>/toolchain/09_control/export/` |
| `/var/tmp/alphamotion/beyondmimic/`（`alphamotion_ctrl` 旧副本 + `gmt` 查看器） | 早期副本 | — | 101M | 08-18 | 过时 | 无 | `<ROOT>/archive/`（或经 lloyd 同意后删除） |
| `~/reference_code/ufo/`（BFM-Zero 的 UFO 实现；本地还没提交的：h2_31dof.yaml、g1_lafan_greenwich、h2_ground、seed_g1 等配置；official_ckpt 3.4G；两套 venv 共 17.5G） | 行为基础模型复刻（Blackwell 配方），H2 接入 | `~/tmp/ufo_*`（08-30）、`~/tmp/h2_bfm`（09-17，H2 10M/20M/30M 跟踪视频）、`~/tmp/h2bm` | 26G | 09-17（本地配置） | 在用（实验） | torch2.7.1 cu128、warp1.14 | 上游保持只读；我们自己的配置 → `<ROOT>/toolchain/09_control/bfm_zero_overlay/` |
| `~/reference_code/bfm_zero`、`~/reference_code/beyondmimic` | 上游原版参考 | — | 485M / 524K | — | 参考 | — | `<ROOT>/third_party/`（只读） |
| `RW/alpha_motion2/wbc_expert/` | 6 月 IsaacLab + RSL-RL 的 BeyondMimic 包装 | runs | 1.5G | 06-20 | 过时（本机 Isaac Lab 已经坏了；mjlab 取代了它） | IsaacLab | `<ROOT>/archive/` |

## 3. 建议升格为“一等工具”的 7 个

**实时可视化（2 个）**

1. **V1 实时模型查看器** = 产品 `alphamotion.viz.live.LiveViewer` + service 里的 viewer、transport、comparison 路由。它是唯一一个“常驻进程、就地换内容”的查看器，而且做过浏览器验收和手机端验收。
   - 要补的东西：①训练中途自动出片（watcher 订阅新 checkpoint，只跑 CPU 推理，然后推到 viewer）；②lab_server 那种“多臂同场 + GT 灰影 + 分段补全”；③人体源一定并排显示（渲染纪律）。
   - motion_lab :2334、gallery :8891、mobile :8899 和 studio 都并进来，然后停掉。
2. **V2 标准评测渲染器** = `am2exp_v1/render`（三联「人体 \| 对照臂 \| 本臂」+ grid12）+ `rollout_tools`（版本化输入包 + PTS 校验编码）+ `render_check.py` 的接触和穿透着色。
   - 所有“发视频给 lloyd”的产出都走这一个入口。showcase_render 留着专门做展示。

**模型评估（5 个）**

3. **E1 几何判官** = J1 `ground_judge`（MJCF 碰撞几何）和 `umr_gpu/eval/judge.py`（4096 稠密点 + 碰撞 + 物体 + 接触一致 + 滑移）合并。数据（UMR 重定向）和模型输出用同一把尺，判官不 import 任何训练代码。
4. **E3 逐片段标准评测** = `arena/eval/locomotion.py` + `configs/standards/*.json`。它是 wbc_ready_audit、x2_phys、slip_audit、ground_metrics 的正式移植版，有冻结标准，缺材料就报 `not_evaluated`。后面再加 manip.v1（并入 J2 closure_judge）和 ground.v1。
5. **E5 统计与协议层** = `am2exp_v1/eval/{cases,stats}.py`：冻结病例（带 sha）、按窗口聚类的配对检验、`decide()`、双胞胎门。所有 A/B 判决都走这一层，不再凭肉眼比数字。
6. **E4 编解码器回归考卷** = `v2_judge.py`（固定窗口）。每个新 checkpoint 自动考一遍，防止腿和手退化。
7. **E7 控制闭环评测（WBC 可跟踪率）** = `alphamotion_ctrl`（mjlab）的 `evaluate_local` 和跟踪任务，加上 BFM-Zero 覆盖层。费 GPU，**只能按需触发，而且要先报时长、等 lloyd 点头**。

**训练**：不另外算一等工具。把 `am2exp_v1/ops`（GPU 租约、磁盘守卫、回执、顺序队列、人工批准文件）和 arena 的作业队列、GPU 批推理合并成统一作业层（`<ROOT>/services/jobs`）。训练、评测、渲染都当作一种作业提交给它。`train_scaling` 和 `train_arm` 是 AM2 的训练入口；`train_dual` 冻结成 Greenwich 的复现入口。

## 4. 重复副本清单（统一后每样只留一份活的）

| 东西 | 副本位置 | 建议 |
|---|---|---|
| 研究判官（wbc_ready_audit、x2_phys、v2_judge、slip_audit、showcase_render、eval_rollout、crossrig_amass） | ① `exp_topo/` 原版 ② `arena-datagrid/packages/arena_research/`（只改 sys.path）③ `alpha_motion2/research/architecture_audit_20261003/…/source-snapshot/AlphaMotion1/` | 留 ②，标为 reference_impl；① 冻结；③ 是审计快照，归档 |
| 产品仓库 AlphaMotion（README 的 md5 完全相同） | `~/Desktop/AlphaMotion`（git，release/v0.2）、`arena-datagrid/packages/alphamotion`（只有 siglip 嵌入文件不同）、`~/tmp/greenwich_workbench_0911/Greenwich` | 留 git 那份；另外两份改成依赖它 |
| Greenwich SDK（judge.py 的 md5 一样） | `arena-datagrid/packages/greenwich_motion_sdk`、`~/tmp/greenwich_workbench_0911/greenwich-motion-toolkit`（还带 venv，共 812M） | 留 arena-datagrid 那份；workbench 归档 |
| 控制仓库 | `~/alphamotion_ctrl`（29G）、`/var/tmp/alphamotion/beyondmimic/alphamotion_ctrl`（旧版） | 留前者 |
| 实时查看器 | 产品 LiveViewer、motion_lab、lab_server、viser_mesh_viewer、studio 的 x2_viewer、mobile_server、rollout_gallery | 只留 V1 |
| 12 帧网格 | am2exp `render/grid12.py`、`render_check.py`、`arena/eval/visual.py`、showcase 抽帧 | 只留 grid12（逐帧精确解码，保证首帧和末帧都在） |

## 5. 统一布局里的模型侧部分（提议）

```
<ROOT>/
  toolchain/
    06_train/
      am2/                 train_scaling.py, train_arm.py, arms.json, recipe/ (megatron_scale_pilot/code 冻结)
      greenwich_legacy/    train_dual.py + dual_common/model_dual/masking/contact_label (只读, 复现 v0.2)
    07_viz/
      live/                LiveViewer 抽库 + watcher(新 ckpt → CPU 推理 → 推 viewer)
      render/              triptych / grid12 / scene3d / robot_model / contact / encode(PTS) / bundle
      showcase/            showcase_render
    08_model_eval/
      geometry_judge/      J1 + umr_gpu judge 合并 (数据与模型共用)
      locomotion/          arena.eval.locomotion
      manip_judge/         J2 closure_judge
      codec_regression/    v2_judge + 固定窗口
      kinematic/           AM2 eval_engine / surface_metrics
      protocol/            cases(冻结+sha) / stats / decide / acceptance_*.yaml (spatial-eval, motion-eval)
      standards/           locomotion.v1.json, manip.v1.json ...
      release_gate/        quality.release_passed + 产品/浏览器验收
      latent_critic/       SDK judge (非物理)
      reference_impl/      wbc_ready_audit, x2_phys, slip_audit, ground_metrics 原版 (冻结)
    09_control/
      mjlab/               amctrl + scripts (evaluate_local, horizon_*)
      export/              ctrl_verify 导出器 (→ CSV/npz)
      bfm_zero_overlay/    我们自己的 ufo 配置 (h2_31dof, greenwich 数据配置)
  services/
    jobs/                  作业层: am2exp ops(lease/guard/receipts/queue/gates) + arena store/worker
    gpu/                   arena gpu_batcher (批推理)
    api/                   FastAPI: /models /runs /evals /renders /viewer /jobs
    mcp/                   MCP: 同一批端点的工具化
    web/                   Codex 做的 Web UI (底座候选 motion_audit_web)
  models/                  模型登记: greenwich, bumblebee, megatron, prime, equator_a/b, AM2 A/B×S/M/L (权重 + model card)
  runs/  evals/  renders/  每条带 manifest + 回执
  third_party/             ufo, bfm_zero, beyondmimic, mjlab (只读)
  archive/                 本表标“归档/过时”的全部
  deliverables/            对外交付包
```

**MCP 和 HTTP 要暴露的模型侧能力**（同一套后端）：
- `list_models`、`get_model_card`、`list_runs`、`get_run_curves`、`latest_checkpoint`（只读）
- `run_eval(model, suite=locomotion.v1|codec_regression|geometry|manip, cases=frozen_id)` → 作业 id；`get_eval(job)` 返回 verdict 表，每行写明尺子和通过与否
- `render_triptych(source, control, test)` / `render_grid12(video)` → 作业 id，产物是 mp4/png 的 URL
- `viewer_load(model, clip, robots[])`：把内容推到常驻 viewer，返回浏览器地址
- `compare(a, b, cases)` → stats.decide() 的判决（带置换检验 p 值和窗口符号数）
- `submit_train(arm, approval_stage)`：一定要有 lloyd 的批准文件；会写电费和时长估计
- `wbc_track_eval(motion, robot)`：GPU 控制闭环评测，默认要人工批准

## 6. 发现的问题（只记录，没有动）

1. **`alphamotion-ardy-isaac.service` 一直在崩溃重启**：NRestarts=352，状态 activating auto-restart。原因是 `~/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml` 不存在了。每次重启都要加载一遍，白白占 CPU。建议 lloyd 执行 `systemctl --user disable --now alphamotion-ardy-isaac`。
2. 3 个常驻服务监听 `0.0.0.0`（2334、8891、8899），内容都停在 7 月或 8 月。motion_lab 占约 745MB 内存。可以合并进 V1，然后停掉。
3. `/var/tmp/alphamotion/topo_exp` 有 160G，在 `/` 分区上（这个分区已用 88%，只剩 76G）。其中 cache 约 140G（cache_xwbc_amass_full 61G、cache_amass_lconv_tfix 52G、bridge_codes 17G）。复现 v0.2 时需要这些 cache，建议迁到冷存储，在原位置留软链接。
4. `~/alphamotion_ctrl/.git` 有 14G，大概率把 motions 提交进去了，需要瘦身（改用 LFS 或从历史里移除）。
5. `~/reference_code/ufo` 里有我们自己写的 H2 和 Greenwich 配置，没有提交，散在上游仓库里，一旦更新上游就会丢。
6. `arena serve api` 还是占位；arena 作业队列里 eval 和 export 两种作业还没实现。所以“统一服务”目前没有能直接用的 HTTP 层。最接近成品的是产品 `service/app.py` 和 `motion_audit_web/backend/app.py`。
7. studio 的作业表只存在内存里，重启就丢。新作业层必须把作业落盘（arena 的 `db/store.py` 已经落盘了）。
