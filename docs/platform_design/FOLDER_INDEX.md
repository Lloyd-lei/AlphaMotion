# 文件夹索引：现在每个文件夹是什么、产出了什么、以后放哪

- 日期：2026-10-07 晚。来源：同目录 4 张地图（`map_factory.md`、`map_model_tools.md`、`map_research_repo.md`、`map_caches_artifacts.md`）加 18:48 的现场核对。只读整理，没动任何东西。
- 配套设计：`FINAL_DESIGN.md`。下文 **`根/`** = 新根目录 `/home/arenalabs/alphamotion/`（名字待 lloyd 定）。
- 大小是地图里用 `du` 量的近似值；"P0–P6" 是 `FINAL_DESIGN.md` E.1 的迁移阶段。

## 0. 怎么读

**状态**：
- **运行中**：此刻有进程在写，或服务在跑。
- **生产**：现役引擎、数据或资产，不能丢。
- **承重**：不再写了，但现役代码、训练集、服务在读，挪走就断。
- **活跃**：最近在用，没有常驻进程。
- **休眠**：暂停的工作线，以后可能恢复。
- **参考**：只读依据，别人在引用。
- **归档**：结论已出，只留着查。
- **过时**：已被别的东西取代，可以清理（要 lloyd 点头）。
- **坏了**：链接指向不存在的东西，或服务起不来。
- **无关**：不属于 AlphaMotion。

**去向（动作）**：
- **登记**：原地不动，只进账本；
- **链入**：根目录下建软链，原地不动；
- **拷贝**：拷一份进根目录，原处留软链（只用于小件）；
- **搬**：同分区 `mv`，原处留反向软链 30 天；
- **进 git**：原地 `git init` 做快照，不改文件；
- **归档**：挪进 `根/archive/`，只读；
- **删除候选**：只提议，lloyd 逐条批；
- **不进根**：和 AlphaMotion 无关，或是商业资料（→ `~/business/`）。

---

## 1. 一眼看全：东西散在哪

| 位置 | 大小 | 里面主要是什么 | 去向 |
|---|---|---|---|
| `/var/tmp/alphamotion` | 352 GB（根分区，已用 88%） | 旧训练缓存和 run（`topo_exp` 160 GB）、xwbc 压缩包 129 GB、工厂共用 Python 环境、GPU 引擎产出、机器人模型源 | 大缓存原地登记；小件生产资产 P5 搬到 /home；不再写新东西 |
| `~/umr_factory_data` | 约 208 GB（/home） | 数据工厂：原始数据、SMPL-X、机器人资产、CPU 输出、质检、普查 | P0 链入 `根/data/`，P5 按子目录搬 |
| `~/Desktop/"be water, robot"` | 85 GB | 研究主仓（带引号的那个才是真的）：AM2 预训练底座和规模实验、Greenwich 代码、7 月演示栈、第三方仓库、商业资料 | 链成 `根/repos/research`，内容分流 |
| `~/tmp` | 56 GB | 1,440 个散文件 + 约 120 个目录：接触标签 v3（承重）、各种实验视频和分数 | 承重的登记或拷贝，其余打包归档 |
| `~/reference_code` | 31 GB | UFO/BFM-Zero、SONIC、BeyondMimic 外部仓库 | 链入 `根/third_party/` |
| `~/alphamotion_ctrl` | 29 GB | WBC 控制仓（mjlab），`.git` 臃肿 14 GB | 链成 `根/repos/ctrl` |
| `~/Desktop` 其它 | 约 20 GB | 产品仓、arena-datagrid、报告素材库、组员脚本、无关项目 | 分别链入或不进根 |
| `~/alphamotion_artifacts` | 13 GB | PT2 实验产出、X2 训练池 | 归档 / 登记 |
| `~/gpu_convert`、`~/motion_audit_web`、`~/umr_gpu`、`~/umr_factory_code`、`~/umr_dataset` | 6.5 GB + 3.2 GB + 小 | 引擎和工具代码（都没有 git） | P1 进 git，链成 `根/repos/*`，生产从 `根/engines/` 跑 |

---

## 2. `~/umr_factory_data`（数据工厂，/home）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `raw/hiphi` | HiPHI 原始动捕（Noitom，原片 308.7 h，90 Hz，每段带 1 个物体） | 分卷 tar、解压后的 part 目录、全量元数据 csv、物体网格、LICENSE | **运行中**：下载器在下，**截止约 10-09 早 6 点**，还差 144.5 GB | 链入 `data/s1_raw/hiphi`（P0），P5 搬；永久保留（过了截止拿不回来） |
| `raw/grail` | NVIDIA GRAIL（62 h，24 Hz，视频重建，带地形和物体） | recon pkl、物体 usd、锁定的 HF 版本号、失败记录 | **运行中**：下载循环（建议先暂停，把网让给 HiPHI） | 链入 `data/s1_raw/grail`（P0），P5 搬 |
| `raw/omnicontact` | OmniContact（690 次采集：搬箱、踢球） | BVH + 道具 csv、G1 标签、bvh 缓存 | 生产（已下完） | 链入 `data/s1_raw/omnicontact`（P0），P5 搬 |
| `smplx/hiphi` | HiPHI 转 SMPL-X，CPU 老路径（1,094 段） | 每段 npz + metadata + qa + 物体穿透审计 | 过时（GPU 转换器接替） | 登记成 `data/s1_human/hiphi/smplx_cpu@v1`；有 GPU 版的标"已取代"，只有 CPU 版的补 30 fps |
| `smplx/omnicontact` | OmniContact 转 SMPL-X（690 段） | 按字段分开的 npy + 物体 + qa | 生产 | 链入 `data/s1_human/omnicontact/smplx@v1` |
| `smplx/grail` | GRAIL 转 SMPL-X，带**全平台唯一一份逐段 license 清单**（22,312 行） | recon pkl、物体 mjcf/usd、manifest、碰撞缓存 | **运行中**：转换、碰撞、清单重建三个循环 | 链入 `data/s1_human/grail/smplx@v3`；清单导入账本，当逐段 license 的样板 |
| `smplx/_test` | 两个测试文件 | `t55.npz`、`t55b.npz` | 过时 | 删除候选 |
| `smplx_gpu/hiphi` | HiPHI 转 SMPL-X，GPU 新路径（3,796 段，90 Hz） | 每段 npz + metadata + qa、转换日志、part 状态 | **运行中**：`run_watch.sh` 每 2 分钟扫一次 | 链入 `data/s1_human/hiphi/smplx@v2/fps90`（可重生成） |
| `smplx_gpu/hiphi_30fps` | 同上，降到 30 fps（训练用的帧率） | 同上（3,795 段） | **运行中** | 链入 `data/s1_human/hiphi/smplx@v2/fps30`（保留） |
| `robots/` | 21 台核心机器人的**权威资产** | 展平 MJCF、网格、`staging.json`、`surface.npz`（4096 外表面点 + 法线 + 区域）、T 姿态联系表 | 生产 | P5 搬到 `data/s2_robots/<机器人>/` |
| `robots/<r>/pipeline/` | 5 台机器人各自的 QA 流水线（g1_23、g1_29、h1、h2、r1），每台一份测量脚本 | 冻结配置、脚本、产物、日志 | **运行中**（robots_wf 代理） | 配置 → `data/s2_robots/<r>/pipeline.json`；脚本被统一判官 `geometry@v1` 取代后归档；产物 → `data/s4_dataset_eval/` |
| `corr/` | 人体到机器人表面对应关系（CPU 引擎用） | 每台 smplx 版 + beta0 版 | 生产；**和 GPU 引擎那份字节不同，待查** | P5 搬到 `data/s2_robots/<r>/corr@factory_20261004` |
| `runs/` | CPU 引擎跑的 7 段 UMR 原始结果 | 原始 qpos、`qpstats.json`、人体 4096 槽位 | 归档（被 `out/` 的来源字段引用） | 搬到 `data/s3_retarget/umr_cpu@d1cdaed/runs_20261005` |
| `out/` | CPU 工厂最终输出（`factory_npz_v1`，11 个文件：7 段 × G1/GR3/H2/G1-23） | npz + `.qa.json` + manifest | 活跃（样本级）；**搬箱片段受"箱子当球"bug 影响** | 登记成 `data/s3_retarget/umr_cpu@<commit>/20261005_pilot`；箱子片段隔离 |
| `qa/` | 5 台机器人的 QA 报告（每台布局不同；今天在这里发现箱子 bug） | json、逐帧 npz、网格图、mp4、进度 md | **运行中** | P5 搬到 `data/s4_dataset_eval/2026-10-07_robots_qa/`（先统一格式） |
| `qa_renders/` | SMPL-X 转换质检截图 | png | 归档 | 归档 `archive/2026-10_conversion_qa` |
| `diversity/` | 预训练数据多样性普查（今天出了 `REPORT.md`：已知 571 h，手上 126 h） | 各源 summary、逐段标签、图、合成数字、REPORT | 活跃 | 脚本 → `code/s4_dataset_eval/census/`；产物 → `data/s4_dataset_eval/2026-10-07_diversity/` |
| `audit/` | 人审接口约定（`SCHEMA.md`、`CODEX_PROMPT.md`）；三个 jsonl 还没生成 | md | 参考 | 链入 `data/s4_audit/`；约定拷到 `docs/contracts/audit_contract_v1.md` |
| `logs/` | 下载、转换、守护进程的日志和 pid；`throttle.log`；HiPHI 状态 | log、json、pid（GRAIL `dl_loop.log` 406 MB 不轮转） | **运行中** | P5 搬到 `services/factory/logs/`，加轮转；过期 pid 文件删除候选 |
| `pilot/` | 早期诊断（视觉量尺 vs 碰撞量尺）+ 临时目录 | 小 | 临时 | `scratch/2026-10-04_pilot/` |
| `hf_cache/` | HuggingFace 元数据缓存 | — | 缓存 | `cache/hf_meta/` |
| `workflows/` | Claude 工作流（`diversity_wf.js`、`robots_wf.js`） | js | **运行中** | `code/workflows/`；`diversity_wf.js` 写死的 `/tmp/claude-1000/...` 改读配置 |
| `scripts/` | 第一代脚本：HiPHI/Omni 下载、CPU 版 BVH→SMPL-X、质检、`throttle_guard.sh` | py、sh | 大部分过时；`convert_omni` 和 `throttle_guard`（运行中）还在用 | P1 进 git；P4 → `code/s1_clean/{download,convert_cpu}/`、`code/ops/` |
| `scripts_grail/` | GRAIL 整条流水线（下载、转换 v3、地面校准、碰撞、清单） | py、sh（旁边有 v1/v2 的 `.bak`） | **运行中** | P1 进 git；P4 → `code/s1_clean/grail/` |
| `scripts_streamc/` | HiPHI 分卷下载器（下载 → 校验 → 解压 → 删 tar，带磁盘水位） | py | **运行中** | P1 进 git；P4 → `code/s1_clean/hiphi/` |
| `DATA_SPEC_v2.md` | 数据规格 v2（实接触、不穿、不滑、不闪、和人一致） | — | 参考（约束一切） | 拷到 `docs/specs/`，原处留软链 |
| `RESOURCE_LIMITS.md` | 资源上限（USB 网卡怕高负载） | — | 参考 | 拷到 `docs/ops/`，原处留软链 |
| `platform_design/` | 这次的 4 张地图、两个候选方案、最终设计、本索引 | md | 新 | 拷到 `docs/maps/`（P0） |

## 3. 家目录下的代码（都没有 git）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `~/umr_factory_code` | **CPU 参考工厂**（`umrf` 包）：暂存机器人 → 表面点 → 对应关系 → 跑 UMR → 严格后处理 → 标签 → QA | 写到工厂的 `robots/`、`corr/`、`runs/`、`out/` | 生产（参考实现和量尺，代码基本冻结）；**已知 bug：`objects.py` 把箱子当成球**；`paths.py` 写死所有路径 | P1 进 git；链成 `repos/umr_cpu`；冻结副本 `engines/umr_cpu@<commit>`；P5 搬进 `repos/` |
| `~/umr_gpu` | **GPU 批量 UMR 重定向 + v2 严格后处理（生产引擎）**；和 CPU UMR 对拍过 | 写到 `/var/tmp/.../umr_factory/umr_gpu/`；导出 `gpu_npz_v2` | **运行中，且作者正在改**（18:50 刚改写严格后处理的调度）；还没有读真实片段的生产入口 | P1 进 git + 重跑对拍；生产只从 `engines/umr_gpu@<commit>` 跑；开发目录链成 `repos/umr_gpu`，P5 搬 |
| `~/gpu_convert` | **GPU 批量 BVH→SMPL-X 转换器**（HiPHI/Omni），和 CPU 版对拍过（`runs/parity40`） | `smplx_gpu/hiphi{,_30fps}`；`runs/` 里是对拍和基准 | **运行中**（`run_watch.sh`，11 GB 显存） | P1 进 git；链成 `repos/gpu_convert`；冻结副本 `engines/smplx_convert_gpu@<commit>` |
| `~/gpu_convert/runs/parity40` | 转换器对拍证据 | json、npz | 参考 | 登记到 `data/s4_dataset_eval/engine_parity/convert_parity40` |
| `~/gpu_convert/runs/bench*` | 吞吐基准（`bench4` 有 5 GB） | npz | 过时 | 删除候选 |
| `~/gpu_convert/.inductor_cache` | torch 编译缓存 921 MB | — | 缓存 | `cache/` |
| `~/umr_dataset` | **训练数据打包器 + 加载器**（UMRS v1 分片，校验和，mmap，人体和 21 台成对取窗口） | `data/`（7 条样本的分片和清单）、往返和严格复验 QA | 生产（代码）/ 空闲；**找错 HiPHI 路径（FileNotFound），只认 `factory_npz_v1`** | P1 进 git；链成 `repos/umr_dataset`；`data/` → `data/s5_datasets/umrs_pilot@v0` |
| `~/motion_audit_web` | **人工审计工作台**（FastAPI + React/Three.js，127.0.0.1:8765）+ **今天新建的 MCP**（`agent_tools/`，15 个工具） | 判决追加写到工厂 `audit/`；验收截图 | 活跃（网页没开；MCP 可用） | P1 进 git；P3 带历史并进 `code/amp/service` 和 `code/web`；旧启动脚本改成转发，工具名不变 |
| `~/motion_audit_web/agent_tools/`、`configs/mcp.json` | MCP 工具和作业注册表（每个工作流写明固定命令、代码哈希、设备、超时、输出上限；重定向、训练、评估、仿真故意关着） | — | 活跃 | 注册表格式沿用为 `config/workflows/*.json` |
| `~/motion_audit_web/agent_state/` | MCP 作业状态（锁、计划、作业记录） | — | 活跃 | `services/am-service/state/` |
| `~/motion_audit_web/cache/arrays` | 解压后的 npy 缓存 3.1 GB | — | 缓存（可重生成） | `cache/audit_arrays` |
| `~/motion_audit_web/.venv`、`.mcp-venv` | 网页和 MCP 的 Python 环境（`.mcp-venv` 用 /var/tmp 的 UMR 解释器） | — | 活跃 | `envs/` 登记 |

## 4. `/var/tmp/alphamotion`（352 GB，根分区）

### 4.1 顶层

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `topo_exp/` | Greenwich/Equator 研究线的缓存和训练根（主仓 `exp_topo` 的工作区） | 训练缓存、823 个 run 条目、码本、文本对齐（细项见 4.3） | 活跃 + 承重 | 拆开登记，大缓存**不搬** |
| `xwbc_full/` | 从 HF 下的 xwbc 重定向数据压缩包（16 台 × AMASS/LAFAN + 人体） | 38 个 tar.gz，137.89 GB 全部校验通过 | **承重**（10-12 起：AMASS 缓存 10-11 被误删后，这是本机唯一的 AMASS 配对数据源；HF `TonyZhang1999/xwbc-retarget-data` 同版本 b4e169f 仍在） | **不许删**（lloyd 10-12 批准保留） |
| `xwbc_lafan/extracted/` | xwbc LAFAN 部分解包（19 台机器人 + 人体） | 每段一个 npz | 承重（现役 LAFAN 缓存的源） | 链入 `data/s3_retarget/xwbc@v1/lafan/`（对照臂） |
| `umr_factory/` | UMR 数据工厂运行区（细项见 4.2） | — | **运行中** | 拆开，见 4.2 |
| `any_body/` | 7 月底"AnyBody"早期阶段：conda 环境、关节名向量、训练张量 | `env/` 12 GB、`hf/` 780 MB、`cache/*.pt` | 归档（环境过时） | `cache/*.pt` 和说明归档到 `archive/2026-07_anybody`；`env/` 删除候选 |
| `hf_cache/` | HF 模型缓存：bge-m3、Qwen3-Embedding（文本层、关节名向量） | 权重 5.4 GB | 承重（可重下） | 链入 `cache/hf_models` |
| `am2exp/` | AM2 实验第 0 阶段（模型训练和评测那一侧）（细项见 4.4） | 判官尺子、E1/E2 评测、三联视频 | 归档（lloyd 还没看三联视频） | 拆开，见 4.4 |
| `ctrl_verify/` | 给 mjlab/BeyondMimic 准备的跟踪参考轨迹（G1/GR3/X2/ADAM）、X2 语料池 | `tracks_*`、`x2_corpus/`、报告 json | 承重 / 休眠 | 链入 `data/s5_datasets/legacy/wbc_tracks@v1` |
| `config_harvest/` | 抓下来的外部重定向/WBC 仓库配置：GMR、PHUMA、ProtoMotions、soma-retargeter、TWIST… | 只读参考 | **承重**（`config_harvest/GMR` 被主仓 30 处、工厂 19 处引用） | 链入 `third_party/GMR@harvest_20260803`、`third_party/harvest/*` |
| `robot_models/` | 机器人模型源：menagerie、unitree_ros、agibot、roboterax、Fourier | URDF/MJCF/网格 | 承重（工厂和主仓都引用） | 链入 `data/s2_robots/_sources/`（生产用的是 s2 里展平后的副本） |
| `omniretarget/` | OmniRetarget 数据集 + holosoma 代码（对比基线） | zip 和解包各一份（重复） | 归档 | 链入 `third_party/holosoma`、`data/s1_raw/omniretarget`；重复的 zip 删除候选 |
| `gmr_production/` | GMR 原生重定向批量产出（9 台机器人） | 每台动作 + summary | 承重（Native policy 对照臂的数据源） | 链入 `data/s3_retarget/gmr@v2/` |
| `gmr_production_v1/` | 上一份的旧版 | 同上，630 MB | 过时（无引用） | 删除候选 |
| `gmr_spike/`、`gmr_upstream/` | GMR 调参（手臂偏置、ik_configs）和 GMR 上游源码 | ik_configs、报告 | 承重（ik_configs） | 链入 `third_party/GMR/{tuned_ik,upstream}` |
| `gem_runtime/`、`gem_runtime_pkgs/`、`gem_text/`、`genmo_bridge/`、`genmo_cache/`、`tmr_bridge/` | 视频/文本 → 动作的桥（Studio 用） | 提示词 pt、样例 npz | 休眠 | 链入 `third_party/gem`；Studio 退役后归档 |
| `studio/`、根目录 `studio_app.py` | 0814 产品前端（prompt/视频 → GEM → Greenwich → Equator → mp4），作业表只在内存里 | 结果 trace、mp4 | 过时（没在跑；已被产品 service 取代） | 归档 `archive/2026-08_studio`；它的 worker 协议写进 `docs/` |
| `beyondmimic/` | 最早的 BeyondMimic 环境（`alphamotion_ctrl` 旧副本 + gmt 查看器） | — | 过时 | 删除候选（或归档） |
| `dex-retargeting/`、`anytop/` | 外部仓库：手部重定向、AnyTop 基线 | 源码 | 归档 | 链入 `third_party/` |
| `hf_home/` | 下 xwbc 时的 HF 元数据缓存 | — | 过时 | 删除候选 |
| `xwbc_clips/`、`xwbc_dl/` | xwbc 早期样例片段、下载脚本 | 样例 223 MB | 过时 | 删除候选 |
| `spatial_recipe_ablation_20260816/` | 空间配方重训消融报告（中文 PDF/HTML/CSV） | 报告 | 归档 | 拷到 `docs/reports/2026-08-16_spatial_recipe_ablation/` |
| `boneseed_spatial_*`（20 个目录） | 08-16 BONES-SEED 空间模型审计和视觉门的各个版本 | json、视频、网格图，共约 100 MB | 归档 | 合成一个 `archive/2026-08-16_boneseed_spatial/` |
| `mesh_viz/`、`ab_test/` | G1 网格对照图；holosoma vs soma 的 A/B 指标 | png/gif、json | 归档 | 归档 |
| `pose_translator_2`（软链） | 指向 `~/alphamotion_artifacts/pose_translator_2` | — | 链接 | 删除这个链接（账本里有记录） |
| 根目录约 70 个散文件 | 训练队列脚本、下载日志、BONES 视觉门 PDF/mp4、`xwbc_joint_names.json` | 脚本、日志、视频 | 归档（`xwbc_joint_names.json` 承重） | 脚本 → `archive/queues/`；视频 → `media/2026-08-16_boneseed/`；`xwbc_joint_names.json` → `data/s1_sources/xwbc/` |

### 4.2 `umr_factory/`（18 GB，**运行中**）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `umr/UMR` | 上游 UMR 代码（commit d1cdaed），多了没跟踪的运行时缓存和配置 | — | 生产 | P5 拷到 `third_party/UMR@d1cdaed`，运行时缓存分到 `cache/umr_runtime`；原处留软链 |
| `umr/env` | **几乎所有工厂程序共用的 Python 环境**（3.12、mujoco 3.3.7、torch），6.7 GB | — | 生产（GRAIL 循环、GPU 引擎、MCP 都在用） | P5 在 /home 按锁文件重建到 `envs/umr`，旧环境留一个月 |
| `umr/tools` | UMR 包装（`umr_run.py` 带 QP 失败回退、评测和渲染） | — | 生产（CPU 工厂依赖 `umr_run.py`） | P1 进 git；P4 → `repos/umr_cpu/tools` |
| `umr/{configs,eval,runs,logs,videos}` | UMR 侦察期实验（G1/GR3 默认参数、无地面、自碰撞对照） | json、npz、mp4，约 0.6 GB | 归档 | 归档 `archive/2026-10-04_umr_scout` |
| `robots/` | **机器人登记表**：`robots.json`（34 台，核心 21 台）、`umr_configs/`（30 份）、清单脚本 | json、md、png | 生产 | P1 拷到 `data/s2_robots/registry.json` + `registry/umr_configs/`，导入账本 |
| `umr_gpu/problems/` | CPU 抓好的逐帧问题（GPU 引擎的输入），5.7 GB | npz | 运行中 | 可重生成；登记，P5 移到 /home 或删除 |
| `umr_gpu/results/` | 对拍、导出、吞吐结果；`parity_tables.md`；`export/` 是 `gpu_npz_v2` 唯一的 12 个样本 | npz、json、md | **运行中**（今天还在写吞吐结果） | `export` → `data/s3_retarget/umr_gpu@<commit>/20261005_samples`；`parity` → `data/s4_dataset_eval/engine_parity/` |
| `umr_gpu/videos/` | 24 个 CPU/GPU 对比 mp4 + 网格图，1.6 GB | mp4 | 归档 | `media/2026-10-05_umr_gpu_parity/` |
| `umr_gpu/robots/` | 4 台机器人资产，和工厂那份逐字节相同 | — | 重复 | 删除候选（改成引用同一个资产） |
| `umr_gpu/corr/` | 4 台对应关系，**和工厂那份字节不同** | — | 生产（对拍用的就是它） | 登记成 `data/s2_robots/<r>/corr@umr_gpu_20261004`，查清差异前不动 |
| `umr_gpu/{runs,sources,logs}` | GPU 引擎的运行记录、源样本、日志 | — | 运行中 | 登记，P5 移到 `services/umr_gpu/logs` |
| `labels/` | 点云接触标签器原型（spec v0.1，`pcl_contact`） | 标签 npz、报告 | 过时（被 `umrf/labels.py` 和 `umr_gpu/strict.py` 接替） | 归档 `archive/2026-10_pcl_contact_proto` |
| `sources/` | 数据源侦察：HF 文件树、README、**license 原文**、`sources_inventory.json`、BONES 元数据 parquet | json、md、parquet | 参考（license 追踪唯一原始记录） | P1 拷到 `data/s1_sources/`，导入账本 |
| `uniml3d_reload/` | UniML3D / Mixamo 重载原型 + 样本缓存 1.3 GB | csv、json、缓存 | 归档 / 探索 | 元数据 → `data/s1_sources/uniml3d`；缓存删除候选 |
| `lloyd_gpu_eval/` | 第一次 GPU 和 CPU 对比（舞蹈/踢球 × G1/GR3） | npz、mp4、统计 | 归档，**但 `run_parity.py` 写死了这里** | 链入 `data/s4_dataset_eval/engine_parity/2026-10-04_first_parity`，原处不动 |
| `hiphi2smplx/`、`h2s_shim/`、`beta_fit/` | HiPHI BVH→SMPL-X 第三方代码（MIT）、h5py 垫片、beta 先验 | — | 生产（两个转换器都要用） | P5 → `third_party/hiphi2smplx`、`code/s1_clean/convert_gpu/h2s_shim`、`data/s1_sources/body_models/beta_fit` |

### 4.3 `topo_exp/`（160 GB）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `cache_xwbc_amass_full/` | AMASS × 16 台配对缓存 v1（有时间轴 bug），61 GB | npz | **承重**（现役缓存链接着它，看着像旧版但不能删） | 登记 `legacy_cache`，链入 `data/s5_datasets/legacy/xwbc_amass@v1` |
| `cache_amass_lconv_tfix/` | **现役 AMASS 训练缓存**（时间轴已修；FZ5c、Greenwich-v2、S/M/L20k 都用它），52 GB | 35 个实体文件 + 61 个链接 | 活跃 / 承重 | 登记，链入 `data/s5_datasets/legacy/amass_tfix@v3`；**内部链接不动** |
| `cache_lafan_v3/` | **现役 LAFAN 训练缓存**（115 个链接拼起来） | 链接 | 活跃 / 承重 | 登记，链入 `data/s5_datasets/legacy/lafan@v3` |
| `cache_xwbc_lafan/` | LAFAN 配对缓存 v1（实体文件），7 GB | npz | 承重 | 登记，链入 `data/s5_datasets/legacy/xwbc_lafan@v1` |
| `cache_amass_lconv/` | AMASS 中间版（时间轴修复前），2.4 GB | npz | 承重（tfix 有 3 个链接指向它） | 登记，链入 legacy |
| `cache_amass_heldout/` | AMASS 留出集（600 段语料外审计用），2.2 GB | npz | 承重 | 登记，作为冻结考卷 `G-heldout600@legacy` 的数据 |
| `cache_amass_v3/` | AMASS v3（链接拼起来，旧标签） | 链接 | 归档 | 原地归档（账本记录） |
| `cache_lafan_v2/` | LAFAN v2 | 17 个链接指向已删除的 `~/tmp/cl_v2` | **坏了** | 删除候选 |
| `cache_amass_4000`（链接） | 指向不存在的 `cache_xwbc_amass` | — | **坏了** | 删除候选 |
| `cache_paired*`、`cache_big/`、`cache/` | 8 月初 Stage-1 的配对缓存 | npz，约 400 MB | 归档（`cache_paired_fixed` 还有引用） | 原地归档；有引用的留链接 |
| `cache_bones_body/` | BONES-SEED 身体缓存 322 MB | npz | 归档 | 登记 `data/s5_datasets/legacy/bones_body@v1` |
| `cache_xwbc_lafan_fixed/` | LAFAN 手腕修复版 | npz | 归档 | 原地归档 |
| `bridge_codes/PROD_c1_91k_dim20_tail_0810/` | C1 码本编出的全库码，17 GB | 码 | **承重**（Equator A/B、Atlas、library 还用 C1 码） | 登记，链入 `data/s5_datasets/legacy/codes_c1@0810` |
| `text_align/` | 文本层：caption 向量、文本码、训练/验证配对，2.2 GB | pt、npz | 承重（文本层 B5） | 登记，链入 `data/s5_datasets/legacy/text_align@v1` |
| `v5_keycache/` | 人体 SMPL 关键帧缓存 1.2 GB | — | 归档 | 原地归档 |
| `runs/` | **823 个训练 run 条目**（C1、BRIDGE_A3/B3、FZ5c、Y5b、Z3、E24、L_S/M/L20k……），16 GB；v0.2 发布权重从这里导出 | checkpoint、配置、日志、`v2_judge.json` | 活跃（发布血统）+ 大量归档 | 导入账本 `runs`；血统上的登记成 `runs/greenwich/...`、`runs/equator/...`（链接）；其余进归档计划 |
| `night2_lib/` | night2 依赖库（有一个软链指回主仓 `grand_eval/src`） | py | 承重（11 处引用） | 登记，原处不动 |
| `encoder_bakeoff/`、`_amass_work/`、`contact_stats/`、`v2_prepatch_0831/` | 编码器比武、AMASS 构建脚本、接触统计、补丁前快照 | 小 | 归档 | 原地归档 |
| `qwen3_names.pt`、`siglip_names.pt` | 关节名语义向量（主版） | pt | 承重 | 拷到 `data/s2_robots/_embeddings/` |
| `*.pt.bak`、`*.clobbered` | 上面两份的备份和被覆盖版 | pt | 过时 | 删除候选 |
| 根目录 `*.py`、`*.log` | 桥状态图谱、生产域评测、X2 查看器 | 小 | 归档 | 归档 |

### 4.4 `am2exp/`（3 GB，AM2 实验第 0 阶段）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `assets/surface_v2` | **另一套**机器人表面资产（绑在 xwbc GMR 的 MJCF 上，18 个身体；kuavo_s45、h1_2 绑定有问题） | manifest、png、验证 | 归档（表面点只留工厂那套） | 归档 `archive/2026-10_surface_v2` |
| `judge/` | 判官 J1 的尺子和绑定、第 0 阶段报告、留出根偏置 | json、md | 归档（代码在主仓 `am2exp_v1/judge`） | `evals/am2exp_stage0/judge/` |
| `eval/`、`data/` | E1/E2 评测用例（带 sha）、A0/B0 推理结果 2.9 GB | json、npz | 归档 | `evals/am2exp_stage0/` |
| `integration/`、`review/`、`runs/`、`model/` | 联调 dryrun、审阅、run 记录、未训练的 checkpoint | — | 归档 | `runs/am2exp/stage0/` |
| `videos/` | 三联视频 [人体 \| GT \| 模型] | mp4 | 归档（**lloyd 还没看**） | `media/2026-10-04_am2exp_triptych/` |
| 顶层 `stage0_index.md` | 第 0 阶段总索引和闸门判决（量尺闸门 FAIL 6/15，E2 dev 0.765 / test 0.539） | md | 参考 | `docs/reports/2026-10-04_am2exp_stage0/` |

## 5. `~/tmp`（56 GB，按主题分组）

| 组 / 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| **`cl_v3/`** | 接触标签 v3，每台机器人一个 `*_contact.npz`（amass 15 GB、amass_lconv_tfix 14 GB、lafan 1.6 GB…），31 GB | 标签 | **承重**（现役训练缓存的 34 个链接指向它） | **不搬**，原地登记成 `data/s3_labels/contact@v3`（链接）；其中 `amass/` 15 GB 只被旧缓存用 → 删除候选（确认后） |
| **`sole_offsets3.json`**（及 `sole_offsets.json`、`sole_offsets2.json`） | 每台机器人脚底偏移表（根高偏置的根源：8 台偏大 4.7–10 cm，缺 H2） | json | **承重**（4 处引用） | P1 拷到 `data/s2_robots/<r>/sole_offset@v3.json`，原处留软链；另出 v4 |
| **`gauge_check_0903.json`**、**`dialect/`** | 数据 gauge 检查；人体骨架"方言" Q 矩阵 | json、pt | **承重** | P1 拷到 `data/s4_dataset_eval/2026-09-03_gauge_check`、`data/s1_sources/human_dialect_Q@v1`，原处留软链 |
| **`refiner_suite/`** | SE(3) 精修器评测套件（eval_suite、profiles、manip_set、out），728 MB | 评测结果 | **承重**（13 处引用；精修器已叫停，当对照臂） | 链入：代码看 `repos/research`；`out/` → `evals/refiner_suite@v1` |
| `cl_v2_qc/`、`contact_qc/` | 接触标签 v2 质检、AMASS 穿地审计 | json、视频、npz | 承重（11 处、7 处引用） | 链入 `evals/contact_qc_legacy/` |
| `ab_sole0/`、`ab_sole1/` | 脚底 A/B | npz | 归档 | 归档 |
| `greenwich_workbench_0911/` | **datagrid 服务的运行态**（sqlite + postgres，09-23 恢复过）+ 产品仓副本 + SDK 副本 + H2 资产链接 + venv，812 MB | 服务状态 + 代码副本 | 承重 / 休眠 | P0 多拷一份数据库到 `services/datagrid/backups/`；运行态 → `services/datagrid/state/`；代码副本归档 |
| `arena_handoff_backup/` | datagrid 数据库**唯一的备份**（.sql + sqlite） | 备份 | 承重 | P0 拷到 `services/datagrid/backups/`（永久保留） |
| `ufo_resume_drill/`、`ufo_h2_smoke/` | UFO/BFM-Zero 续训演练、H2 冒烟训练的 checkpoint，13.5 GB | checkpoint | 过时 | 配置留进 `runs/wbc_ufo/_pilots`；checkpoint 删除候选 |
| `ufo_p1..p5`、`ufo_probe*`、`ufo_smoke_g1*`、`ufo_cache/`、`cache/`、`torchinductor_*` | UFO 试跑配置、编译缓存 | 配置、缓存 | 过时 / 缓存 | 配置 → `runs/wbc_ufo/_pilots/`；缓存删除候选 |
| `bfm_data_clone/` | BFM-Zero 仓库副本（和 `reference_code/bfm_zero` 重复） | 源码 | 过时 | 删除候选 |
| `h2_bfm/`、`h2bm/`、`h2gr*/`、`h2hd/`、`h2mp/`、`g1gr/`、`h2_ground_rs/`、`seed_g1_rs/`、`h2_bfm_watch.*` | H2/G1 的 BeyondMimic 和 BFM-Zero 训练中途视频、地面 4 段、种子参考 | mp4、json、npz | 归档（H2 线 09-18 后暂停） | `media/2026-09-1x_h2_wbc/`、`runs/wbc_mjlab/h2/` |
| `teleop2/`、`teleop_*.py` | 遥操两段式探针（Greenwich w15） | mp4、npz | 归档 | `evals/archive/teleop/` |
| `ee_data/` | EgoDex / UMI 手部数据 | 数据 627 MB | 归档 | `data/s1_raw/egodex/` |
| `ee_fill/`、`ego_fill/`、`source_robot_probe_0905/`、`six_robot_manip_eval_0905/`、`six_robot_foot_ablation_0905/`、`alphamotion_probe_0905/`、`embodiment_normalization_0905/` | 手部 SE(3) 补全全身：钉手钉脚、六台机器人操作和脚部消融 | 视频、帧、json，共约 6 GB | 归档（精修器已叫停） | `evals/archive/2026-09-05_ee_fill/`、`media/` |
| `wbcready_*`（162 个文件）、`wbcready_vid/` | 600 段语料外 WBC 可训性审计，每个 checkpoint × 3 台 | txt、json、视频 | 归档（已复制进报告 02） | `evals/archive/wbc_ready/<ckpt>/` |
| `x2phys/` | X2 零样本、不需真值的物理判官结果 | json 61 MB | 归档（已复制进报告 03） | `evals/archive/x2phys/` |
| `stress*`、`gwstress*`、`gwf0stress*`、`gwhmstress*`、`crossrig*`（12 个）、`slipvid_*`（9 个）、`cmp34*`、`p3_*`、`qc_*`、`r13viz`、`r14viz`、`prog_*`、`mid1000/1500`、`y1*/y3*/y5*`、`landscape/`、`death_slices/`、`abl_render_slices/`、`_smoke_render/`、`sweep/`、`gr3_3p/`、`gr3_mid/` | 8 月底到 9 月初的压力测试、跨骨架、滑步视频、对照、里程计、缩放地形 | 视频、网格图、json，约 1.2 GB | 归档（部分被报告引用） | `evals/archive/2026-08-29..09-04/<主题>/`；被报告引用的单独登记 |
| `amass_g1_*` | AMASS → G1 解码对比小样本 | npz | 归档 | 归档 |
| 根目录 194 个 mp4（703 MB） | G1/GR3/ADAM 对决、起身、GW 跟踪、精修器、试点解码…… | 视频 | 归档（一部分已复制进 `alphamotion video`） | `media/<日期>_<主题>/`，报告引用的优先 |
| 根目录约 1,240 个 json/log/sh/py/marker | 各次评测分数、训练日志、链式脚本、完成标记、一次性诊断脚本 | — | 过时 / 归档；`ablation_table.csv`、`scaling_*.csv`、`stress_run.py`、`qpos_to_rs.py`、`xwbc_to_robotstate.py` 有引用 → 承重 | 有引用的链入 `code/` 或 `evals/`；其余打包 `archive/tmp_2026-08_09.tar.zst` |
| 152 个 `tmp*wandb-*` 目录 | wandb 残留临时目录（基本是空的） | — | 过时 | 删除候选 |

## 6. `~/alphamotion_artifacts`、`~/alphamotion_ctrl`、`~/reference_code`

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `~/alphamotion_artifacts/pose_translator_2/` | Pose Translator 2（独立的空间编解码）实验产出：约 30 个试点 run、冒烟、rollout，11 GB | run、json、mp4 | 归档（08-17 后停） | 原地归档（账本 `archive/2026-08_pt2`），冷存储到位后挪 |
| `~/alphamotion_artifacts/x2_pool_v3/` | X2 WBC 训练池 v3（分片 npz + manifest） | shard npz | 承重 / 休眠 | 链入 `data/s5_datasets/legacy/x2_pool@v3` |
| `~/alphamotion_artifacts/x2_pool/` | X2 池 v2 | 同上 | 归档 | 归档 |
| `~/alphamotion_artifacts/x2_pool_v1_broken/` | X2 池 v1（世界行差一位的 bug） | 同上 | 过时 | 删除候选 |
| `~/alphamotion_artifacts/data_contract_audit_20260817/` | 动作数据合约审计 | md、json、png | 归档 | `evals/archive/2026-08-17_data_contract/` |
| （缺失）`~/alphamotion_artifacts/horizon_x2_hz4` | 主仓 2 处引用，但目录不存在 | — | **坏了** | 账本标"缺失" |
| `~/alphamotion_ctrl/amctrl/` | WBC 控制器任务包：G1 地面、GR3、X2、H2（mjlab）、地平线课程、意图码、transformer actor | 代码 | 休眠（电费预算从 09-03 起叫停 RL） | 链成 `repos/ctrl`（自带 git） |
| `~/alphamotion_ctrl/scripts/` | 约 60 个脚本：`evaluate_local.py`、`csv_to_npz_local.py`、`foot_audit.py`、H2 脚本 | py | 承重 | 随 `repos/ctrl`；`evaluate_local` 包成判官 `wbc_track@v1` |
| `~/alphamotion_ctrl/motions/` | mjlab 跟踪参考（206 个 npz：A_greenwich / B_native / 增强），8.6 GB | npz | 承重 / 休眠 | 链入 `data/s5_datasets/legacy/wbc_motions@v1` |
| `~/alphamotion_ctrl/logs/` | rsl_rl 训练 run 和考试 json/mp4 | checkpoint、json、mp4 | 休眠 | 链入 `runs/wbc_mjlab/`，导入账本 |
| `~/alphamotion_ctrl/.venv/` | mjlab 1.6.0 环境 5.8 GB | — | 承重 | 链入 `envs/ctrl` |
| `~/alphamotion_ctrl/.git/` | 14 GB（散对象 8.45 GB + 中断留下的垃圾 1.18 GB） | — | 臃肿 | 仓库主人 `git gc --prune` 约收回 9 GB（要 lloyd 同意） |
| `~/reference_code/ufo/` | UFO（BFM-Zero 的 Blackwell 配方）+ 官方 checkpoint；**里面有我们没提交的 H2/Greenwich 配置**；两套 venv 共 17.5 GB | — | 承重 / 休眠 | 链入 `third_party/ufo`；P1 先把我们的配置拷进 `code/s7_eval/wbc/bfm_zero_overlay/`；多的一套 venv 删除候选 |
| `~/reference_code/sonic/` | NVIDIA GEAR SONIC，**H2 的 MJCF/网格从这里来** | — | 承重 | 链入 `third_party/sonic`；H2 资产拷一份进 `data/s2_robots/unitree_h2/` |
| `~/reference_code/bfm_zero/`、`beyondmimic/` | 上游原版参考 | — | 参考 | 链入 `third_party/` |

## 7. 研究主仓 `~/Desktop/"be water, robot"`（85 GB，分支 research-0909；git 只管了约 3%）

### 7.1 顶层

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| 整个仓库 | 研究主仓（名字带引号、空格、逗号） | — | 活跃 | 链成 `repos/research`；改名等旧服务停掉以后再说 |
| `alpha_motion2/` | 第二代研究大杂烩（**整个不在 git 里**），53 GB | 见 7.2 | 混合 | 分流，见 7.2 |
| `pose_translator_design/` | Greenwich（空间层）/ Equator（时序层）主研究线 | 见 7.3 | 活跃 | 分流，见 7.3 |
| `GENMO/` | NVlabs GENMO 克隆（从视频估人体动作），被 conda `genmo-gem` 以可编辑方式安装，20 GB | `outputs/` 演示、`inputs/` 权重 | 活跃（依赖项） | 登记成 `third_party/GENMO`，**不能搬**（环境装着它） |
| `experiment/` | 早期实验 + 两个 NV 第三方仓库 | 见 7.4 | 混合 | 分流，见 7.4 |
| `notebook/` | Stage-1 时期 notebook（5 月） | ipynb、PDF | 归档 | 归档 `archive/2026-05_stage1/notebook/` |
| `any_body/` | Round-1"AnyBody"跨拓扑姿态翻译器（7 月底） | runs 1.1 GB | 归档（被 Greenwich 取代） | 归档 `archive/2026-07_anybody/` |
| `papers-docs/` | 论文 PDF（含中文和双语翻译），40 份 | — | 参考 | 链入 `docs/papers/` |
| `Greenwich_G1_18_Paired_Benchmark_Review_20260920/` | G1 上 18 段"Greenwich vs 传统重定向"对照评测交付包（未提交 git，带 .DS_Store） | HTML 报告、csv、18 段对照视频、5 段宣传片、外联名单 | 归档（交付物） | 评测部分 → `evals/archive/2026-09-20_g1_18_paired/`；宣传片和外联名单 → `~/business/` |
| `data/` | 第一代数据准备：`action_labels.py`（12 类正则）、BABEL 标签 | py、json | 半活跃（工厂的 `run_xwbc.py` 读 BABEL） | BABEL → `data/s1_sources/babel`；`action_labels.py` → `code/s4_dataset_eval/taxonomy/legacy_regex.py` |
| `deliverables/` | AMASS 留出集零样本网格验证视频 | mp4、json、截图 | 归档 | `media/2026-08-17_amass_zeroshot_mesh/` |
| `note/` | Stage-1 研究愿景和三角注意力复盘 | md、图 | 归档（路线已作废） | 归档 `archive/2026-05_stage1/notes/` |
| `claude/` | lloyd 和 Claude 的架构设计文档 19 份（`INDEX.md` 写明哪份是现行版） | md | 参考（部分已被取代） | `docs/design_legacy/` |
| `agent_chat/`、`codex/` | Claude/Codex/lloyd 三方群聊 CLI 和记录；Codex 角色说明 | md、py | 基本停用 | 新 MCP 上线后归档 `archive/agent_chat/` |
| `.claude/` | 项目级 Claude 设置和两个技能（`spatial-eval`、`motion-eval`） | SKILL.md | **活跃** | 拷到 `根/.claude/skills/`；协议写成 `code/s7_eval/protocol/*.yaml` |
| `.git/` | 仓库历史（origin = `Lloyd-lei/AlphaMotion`） | — | 活跃 | 保留 |
| 根目录 `*.fbx`（3 个） | 自采动捕原始数据（Take_005/012） | — | 原始数据 | `data/s1_raw/arena_mocap/` |
| 根目录 `README.md`、`agent.md`、`AlphaMotion_infra.md` | 旧版仓库说明、课程大纲、76 字节草稿 | — | 过时（讲的是已作废的 Stage-1 路线） | 归档；重写新 README |
| 根目录 `share_struct.md`、`nvidia_inception.md`、`ArenaLabs_Stanford_Canopy_Presenter_Notes.pdf` | 股权说明、NVIDIA Inception 申请、路演讲稿 | — | 与研究无关 | 不进根 → `~/business/`（`share_struct.md` 还被 git 跟踪，要从仓库里拿掉） |
| `pytest-of-arenalabs/`、`torchinductor_arenalabs/`、`.pytest_cache/`、`.ruff_cache/`、`.vscode/` | 测试和工具缓存（pytest 目录还被 git 跟踪了） | — | 过时 | 删除候选；加进 `.gitignore` |
| `tool-kits/` | 空目录 | — | 空壳 | 删除候选 |
| 根目录 `mon.pid`、`server_8844.pid`、`MUJOCO_LOG.TXT` | 已死进程的 pid、一次仿真日志 | — | 过时 | 删除候选 |
| **孪生目录** `~/Desktop/be water, robot`（不带引号，557 MB） | 某个脚本引号写错误建的；里面有**独一份**的 `SCALING_LAW.md`、另一版 `CTRL_VERIFY_GR3.md`、一个 x2 ctrl_verify 视频 | — | 过时，但内容要先救 | P0 拷出 3 样到 `docs/reports/greenwich/`、`media/2026-08-29_ctrl_verify_x2/`；之后删除候选 |

### 7.2 `alpha_motion2/`（53 GB）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `pretraining_foundation/` | 预训练数据底座：下载器、来源清单、索引、按窗口读的加载器（31 个测试） | `cache/index*.sqlite`（240,444 条序列×机器人）、`manifests/` | **运行中**（`acquire.py` 和 `queue_followup.py` 在下 UniMate 模型，建议先暂停） | 代码 P1 进 git（下载器跑完前不动）；索引导入账本 `legacy_datasets`；下载器 → `code/s1_clean/acquire/`；加载器 → `code/s5_store/foundation_loader/` |
| `research/alphamotion_scaling_20261004/orchestration/` | AM2 六组从零训练的调度（A/B × S/M/L）：单一 flock、证据门、算力记账 | 代码 | 活跃（当前 AM2 主线） | P1 进 git → `code/s6_train/am2/` |
| `research/alphamotion_scaling_20261004/runs/`、`continuation_v3/` | 六组训练产出 9.3 GB + 续训 4.4 GB | checkpoint、train.jsonl、result.json | 归档（README 写明"不要重启"） | 链入 `runs/am2/2026-10-04_scaling_AB_SML/`，导入账本 |
| `research/alphamotion_scaling_20261004/am2exp_v1/{train,ops,common}` | 按"臂"训练 + ops 层（GPU 租约、磁盘守卫、回执、顺序队列、双胞胎门、人工批准文件） | 输出在 `/var/tmp/alphamotion/am2exp` | 活跃 | ops → `code/amp/jobs/`（作业层的核心）；train_arm → `code/s6_train/am2/` |
| `research/alphamotion_scaling_20261004/am2exp_v1/judge/` | **判官 J1**：在厂商 MJCF 碰撞几何上量 15 区域最低点、穿地、悬空、自穿；有"不读训练代码"的测试 | 尺子和绑定 | 活跃 | 并进 `code/s7_eval/judges/geometry/` |
| `research/alphamotion_scaling_20261004/am2exp_v1/eval/` | 冻结病例 → CPU 推理 → J1 打分 → 配对检验 → `decide()` | 病例、分数、报告 | 活跃 | `code/s7_eval/protocol/` |
| `research/alphamotion_scaling_20261004/am2exp_v1/e2/` | **判官 J2**：手和夹爪抓握闭合（握住、张开、压碎） | — | 活跃 | `code/s7_eval/judges/manip/` |
| `research/alphamotion_scaling_20261004/am2exp_v1/render/` | 标准评测视频：三联「人体源 \| 对照臂 \| 本臂」、12 帧网格、接触着色、厂商网格 | mp4、png、manifest | 活跃（有测试） | `code/s7_eval/render/`（唯一的出片入口） |
| `research/alphamotion_scaling_20261004/rollout_tools/` | 离线"输入包 → 网格回放"，编码时校验逐帧时间戳 | `reports/rollout_mesh_v1/` | 活跃 | 并进 `code/s7_eval/render/bundle/` |
| `research/alphamotion_scaling_20261004/evaluation*/` | AM2 六组运动学评测（片段为单位、和架构无关）；P0/P1 关节精修；表面块对比，约 6 GB | 结果、报告 | 活跃（结果已交付，不要重跑） | 引擎 → `code/s7_eval/judges/pose_fidelity/`；结果 → `evals/am2_scaling_20261004/` |
| `research/alphamotion_scaling_20261004/reports/` | PDF、曲线、对照视频、`rollout_mesh_v1/index.html` | — | 归档 | `docs/reports/2026-10-04_am2_scaling/` |
| `research/megatron_scale_pilot_20261004/` | AM2 训练配方库（recipe_*、pilot_model、pilot_data）+ 试训 | `data_xwbc/` 712 MB、runs 338 MB | 活跃（库，被只读 import） | 代码 P1 进 git → `code/s6_train/am2/recipe/`（冻结）；runs → `runs/am2/megatron_pilot` |
| `research/megatro_scale_pilot_20261004/` | 上面那个的**拼写错误副本**，只有一个 `qwen_audit/` | — | 过时 | 并入上一行后删除候选 |
| `research/embodied_case_20261003/` | G1→H2 单帧具身案例（给架构图用的真实张量） | 网格、注意力图、trace | 归档 | `docs/reports/cases/2026-10-03_g1_to_h2/` |
| `research/hand_surface_case_20261004/` | 手部表面接触区域案例，对照 UMR 参考 | 构造姿态、对比脚本 | 归档（和 S2 表面采样直接相关） | `data/s4_dataset_eval/cases/2026-10-04_hand/` |
| `research/architecture_audit_20261003/`、`transformer_review_20261003/` | 架构审计、Transformer 评审交付包（含判官的第 3 份副本） | 压缩包 | 归档 | `docs/reports/reviews/` |
| `research/training_recipe_audit_20261004/` | XWBC 骨架配对审计 | 1 个 json | 归档 | `data/s4_dataset_eval/audits/` |
| `research/bones_samples/`、`research/motionbricks/` | BONES-SEED 样例截图；MotionBricks 论文 | 图、PDF | 参考 | `docs/papers/` |
| `external/GMR` | 通用动作重定向（无 git，版本不明），机器人 MJCF 资产和 `params.py` | — | **承重**（工厂 19 处、arena 6 处、exp_topo 10 处写死它） | 链入 `third_party/GMR@unknown`；P4 改路径前不动 |
| `external/UMR` | UMR 参考实现（10-02 提交） | — | 活跃 | 链入 `third_party/UMR@<commit>` |
| `external/UMR-website` | UMR 项目网页（.git 坏了） | — | 参考 | 归档或删除候选 |
| `external/UniMate` | UniMate（09-30 提交） | — | 活跃（`unimate_adapter.py` 用） | 链入 `third_party/UniMate@<commit>` |
| `external/GRAIL` | NVlabs GRAIL，**本地改了 3,465 个文件** | — | 半活跃 | 先把改动导成补丁 `third_party/patches/GRAIL/`，再链入 `third_party/GRAIL@<commit>` |
| `external/whole_body_tracking` | BeyondMimic 跟踪训练 | — | 参考 | 链入 `third_party/` |
| `external/soma-retargeter` | NV SOMA 重定向器（arena 的一个 manifest 写死了它的 bvh） | — | 半活跃 | 链入 `third_party/` |
| `external/robocasa`、`GR00T-WholeBodyControl-motionbricks`、`nvidia-cosmos`、`cosmos-framework` | 场景、WBC、世界模型第三方仓库（7 月演示栈用过） | — | 归档 | 链入 `third_party/` 或删除候选（需要时再克隆） |
| `runs/` | 7 月实验输出：`synergy/` 662 个机理分析运行（9.8 GB）、improvisation、吞吐基准 | — | 归档 | 归档 `archive/2026-07_ardy_sonic_stack/runs/`；`synergy` 删除候选 |
| `demo/ardy_preflight_20/` | ARDY preflight 20 段动作 | — | **承重**（motion-lab 服务 :2334 在读） | 服务合并并停掉前不动；之后归档 |
| `demo/alphamotion_verified_g1_atlas_20/` | 20 段经 QA 的 G1 rollout | — | **承重**（rollout-gallery 服务 :8891 在读） | 同上 |
| `demo/` 其余（57 个子目录，含 `deprecated/`） | 7 月 ARDY/Isaac/SONIC 演示、投资人视频、精选动作集 | 视频、轨迹 | 归档 | 归档 `archive/2026-07_demos/`；投资人视频 → `~/business/` |
| `wbc_expert/` | 6 月 BeyondMimic 风格 G1 跟踪专家薄封装（Isaac Lab 已坏，被 mjlab 取代） | runs 915 MB、原始数据（LAFAN1-G1、OmniRetarget、BONES 探针） | 过时 | 原始数据 → `data/s1_raw/{lafan1_g1,omniretarget,bones_probe}`；其余归档 `archive/2026-06_wbc_expert/` |
| `web/` | 7 月 FastAPI 小应用：ARDY+Isaac 实时、已验证画廊、SONIC、投资人画廊 | — | `verified_rollout_gallery.py` 在被服务跑；`ardy_isaac_live_app.py` 是**崩溃循环**的那个 | 停服务后归档 `archive/2026-07_ardy_sonic_stack/web/` |
| `scripts/` | 168 个脚本：ARDY/SONIC 因果分析、精选视频、Isaac 渲染、MuJoCo 回放 | — | 归档（`alphamotion_motion_lab.py` 在被服务跑） | 停服务后归档 |
| `synergy/`、`improvisation/`、`agentic_motion/`、`live/`、`scene_planner/`、`isaac_common/`、`isaac_hotel/`、`isaac_reference_smoke/`、`isaac_scene_live/`、`contact_control.py`、`asset_pipeline.py` | 7 月 ARDY/SONIC/Isaac 演示栈和机理分析代码 | 代码 | 归档（依赖的 `~/Desktop/GR00T-WholeBodyControl` 已不在） | 归档 `archive/2026-07_ardy_sonic_stack/` |
| `data/` | GRAIL 候选、地形评估 + 5 个 0 字节 py | 298 MB | 归档 | 归档 `archive/2026-07_grail/`；空 py 删除候选 |
| `analysis/boneseed_locomotion_index/` | BONES-SEED 运动片段分级表（tier0~3、排除表） | csv 104 MB | 参考（对数据评估有用） | 链入 `data/s4_dataset_eval/legacy/boneseed_tiers` |
| `configs/`、`tests/`、`notes/`、`docs/`、`deploy/` | 演示栈配置、测试、早期笔记、服务单元文件（`deploy/alphamotion-ardy-isaac.service` 是崩溃服务的模板） | — | 归档 | 归档；service 文件参考进 `code/ops/systemd/` |
| `evaluation/`、`geometry/`、`heads/`、`losses/`、`models/`、`modules/`、`training/` | 05-31 搭的架子，**全是 0 字节文件** | 无 | 空壳 | 删除候选 |
| 13 个预览目录：`smplx22_preview*`、`smplx_mesh_preview*`、`soma_official_mesh_preview*`、`somax_cleaned_*`、`soma_locomotion_seed_v0_preview`、`fbx_preview` | 6 月底一次性的格式和网格预览 | 帧图、mp4，约 30 MB | 过时 | 最终版归档 `archive/2026-06_format_previews/`，其余删除候选 |
| `presentation/`、`market_research/`、`pitch_deck_20260722/`、`assets/` | 路演、市场调研、品牌素材 | PPT、PDF、图，约 690 MB | 与研究无关 | 不进根 → `~/business/` |
| `__pycache__/`、`.pytest_cache/` | 缓存 | — | 过时 | 删除候选 |

### 7.3 `pose_translator_design/`（1.2 GB，git 跟踪）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `exp_topo/`（训练和模型） | **Greenwich/Equator 全系代码**：`train_dual.py`、`model.py`、`dual_common.py`、`model_dual.py`、规模阶梯脚本、`masking.py`、`contact_label.py`、`dialect/` | 产物全在 `/var/tmp/alphamotion/topo_exp` | 活跃（被 Studio 7 处、arena 4 处引用；冻结用于复现 v0.2） | 留在研究仓，`code/s6_train/greenwich_legacy` 软链过去；不改 |
| `exp_topo/` 判官：`wbc_ready_audit.py`、`x2_phys.py`、`slip_audit.py`、`ground_metrics.py`、`contact_qc_v2.py`、`gauge_check.py`、`v2_judge.py`、`spatial5.py` | WBC 可训性四张考卷、免真值物理判官、脚滑审计、接地四口径、接触质检、gauge 自检、编解码回归考卷、五项验收 | json 报告 | 冻结（已移植到 arena `locomotion.v1`） | 研究原版作参考；正式实现用 arena 移植版 → `code/s7_eval/judges/{locomotion,physics_nogt,codec_regression}` |
| `exp_topo/` 渲染：`showcase_render.py`、`greenwich_carnival.py`、`crossrig_amass.py`、`slipvid_amass.py`、`pilot_decode_viz*.py`、`p3_*_viz.py`、`render_v5.py` 等约 20 个 | 展示视频（一段人体 + N 台机器人同场）和各实验的一次性对比视频 | mp4 | showcase 在用；其余归档 | `showcase_render` → `code/s7_eval/render/showcase/`；其余留原处（冻结） |
| `exp_topo/viser_mesh_viewer.py`（:8093）、`lab_server.py`（:8114） | 8 月初 viser 查看器和补洞实验室 | 实时页面 | 归档 | 交互需求写进新查看器；代码留原处 |
| `exp_topo/refiner_suite/` | 精修器评测套件代码 | — | 冻结（当对照臂） | 留原处；输出在 `~/tmp/refiner_suite` |
| `exp_topo/eval_rollout.py`、`physeval.py`、`eval_v5.py`、`n3_eval_*.py`、`exam_*.py`、`bridge_*eval*.py` 等约 40 个 | 8 月时序模型对决和考试 | `topo_exp/runs/*` | 归档 | 留原处（冻结） |
| `exp_topo/ctrl_verify/` | 导出参考轨迹（g1/gr3/adam）、x2 pool、qpos 对比渲染 | CSV、npz、mp4（在 `/var/tmp/.../ctrl_verify`） | 冻结（mjlab 导出链） | 导出器 → `code/s7_eval/wbc/export/` |
| `verdicts/` | 历次对决和消融的判决记录（108 个文件） | json、txt | 参考（重要） | 链入 `docs/verdicts/`，导入账本 `events` |
| `grand_eval/` | 08-08 全模型总评估：模型台账、白皮书中英文、展示视频；`src/` 被 `night2_lib` 软链引用 | 交付物 | 归档（src 承重） | `docs/reports/2026-08-08_grand_eval/`；src 原处不动 |
| `R4_VIZ/` | 8 月可视化：网格重建、迁移 gif、GMR 质检、新机器人 | 图、mp4，1.1 GB | 归档 | 归档 `archive/2026-08_greenwich/viz/` |
| `R3_REVIEW/`、`R7_LATENT/`、`R8_DIM/`、`R9_QWEN_BASELINE_figs/`、`figures/`、`viz/`、`audits/`、`scripts/gen_figures.py` | 各轮评审、潜空间和维度实验图、审计 | 图、md | 归档 | 归档 `archive/2026-08_greenwich/` |
| `long_demo_20260903/` | 长演示拼接脚本 | 1 个 py | 归档 | 归档 |
| `gmr_production_data`（软链） | 指向 `/var/tmp/alphamotion/gmr_production` | — | 活跃数据入口 | 由账本里的 `gmr@v2` 取代 |
| 约 50 份顶层 md/pdf/tex/html | 方案、配方、文献综述、诊断报告；`MODEL_VERSIONS.md`（模型登记簿）、`RUNS_INDEX.md` 是现成台账 | 文档 | 参考 | 台账导入账本 `models`、`runs`；其余链入 `docs/reports/greenwich/` |

### 7.4 `experiment/`（7.7 GB）

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `stage1/` | MotionFormer Stage-1（三角注意力） | 代码、runs、诊断图 | 归档（路线已作废） | 归档 `archive/2026-05_stage1/` |
| `kimodo/` | nv-tlabs kimodo 克隆 + 自己的 `.venv` 5.3 GB | — | 半活跃（7 月脚本读它的资产） | 链入 `third_party/kimodo@<commit>`；`.venv` 删除候选 |
| `ardy/` | nv-tlabs ARDY 克隆 + 1.5 GB checkpoint；09-23 编译过 MotionCorrection | — | 活跃（ARDY+Greenwich 流水线） | 链入 `third_party/ardy@<commit>`；权重登记 |
| `synergy/` | 机理分析 phase0~5 结果 | json、图 596 MB | 归档 | 归档 `archive/2026-07_ardy_sonic_stack/synergy_results/` |
| `openfold_ref/` | 空目录 | — | 空壳 | 删除候选 |

## 8. `~/Desktop` 上的其它目录

| 文件夹 | 是什么 | 产出了什么 | 状态 | 以后放哪 |
|---|---|---|---|---|
| `AlphaMotion/` | **产品仓** release/v0.2（已推 Greenwich.git，HF 公开）：`LiveViewer` 实时查看器、产品服务 :7860、发布门 `quality.release_passed`、验收脚本 | `weights_staging/`（greenwich、bumblebee、megatron、prime、equator_a/b、atlas、library） | 活跃（产品；服务没在跑） | 链成 `repos/product`；`weights_staging` 登记成 `releases/alphamotion@0.2`（永久）；LiveViewer 包成实时查看器 |
| `arena-datagrid/` | **Arena Data Grid 单仓**：作业队列 + GPU 批推理 :8932 + worker、`locomotion.v1` 标准、研究判官打包版 `arena_research`、Greenwich SDK、Electron 前端 | 代码（运行态在 `~/tmp/greenwich_workbench_0911/datagrid`） | 休眠（`arena serve api` 还是占位） | 链成 `repos/datagrid`；`jobs` 表设计并进 `code/amp/jobs`；`locomotion.py` 和标准文件拷进 `code/s7_eval/judges/locomotion`；Electron 前端不再接着做 |
| `arena-datagrid/packages/alphamotion` | 产品仓的一份副本（只有 siglip 嵌入文件不同） | — | 重复 | 改成依赖 `repos/product` |
| `arena-datagrid/packages/greenwich_motion_sdk` | Greenwich SDK（潜空间 JudgeHead，**不是**物理指标） | — | 低优先级 | `latent_critic` 判官，标明"非物理" |
| `alphamotion video/` | 展示视频 01–08（FZ5c，09-02/03）+ **`resource/report/` 技术报告素材库 00–35 号** + 图 | mp4、md、json、图，1.2 GB | 活跃（文档中心；INDEX 写明"全是复制件"） | `report/` 链入 `docs/reports/`（编号不变）；视频 → `media/2026-09-02_showcase_fz5c/` |
| `pose_translator_2/` | PT2 代码仓（独立的空间编解码） | 产出在 `~/alphamotion_artifacts/pose_translator_2` | 归档 | 链成 `repos/pt2`（只读），归档 |
| `greenwich_eval_20260826/` | 08-26 Greenwich 评估包 + `greenwich-mobile` 服务（:8899，对全网开着）的网页画廊 | 中文评估、rollout 渲染、QA 视频 | 服务在跑，内容停在 08-26 | 新网页上线后停服务，归档 `evals/archive/2026-08-26_greenwich/`；按字节范围发视频的写法并进新服务 |
| `_backup_before_delete_20260814/` | IsaacLab 本地改动的 diff + buffer 压缩包 | 备份 | 归档 | 归档 |
| `mingrun/` | 组员 mingrun 的脚本：BONES/ARDY 基准、量化、标定、动作查看器；`bones_seed` 链接**坏了** | 代码 | 归档（组员工作区） | 链成 `repos/contrib/mingrun`（只读）；坏链在账本标"缺失" |
| `IsaacLab/`、`IsaacLab-Arena/`、`IsaacLab-Arena-2.3/` | Isaac Lab 和 Arena（本机 Isaac Lab 已坏） | 源码 | 无关 / 归档 | 不进根，留原处 |
| `rot/`、`rot-vs-code-plugin/` | ROT 项目（含另一份 kimodo、docker） | 源码 | 无关 | 不进根 |
| `SegDepthFusion/`、`claw-code/`、`droctavius/`、`src/`（openarmx ROS2）、`openarm_description/`、`openarm_hardware/`、`openarmx_description/`、`Manipulation_demo/`、`lfs/` | 其它项目：分割深度融合、OpenArm 机械臂、ROS、操作演示 | — | 无关 | 不进根 |
| 散文件：演示视频（Stealth Demo，09-03，约 93 MB） | 对外演示 | mp4 | 归档 | `media/2026-09-03_stealth_demo/` |
| 散文件：`babel_v1-0_release.zip` | BABEL 标签原始包 | zip | 原始数据 | `data/s1_raw/babel/` |
| 散文件：ArenaLabs 融资/研究 PDF、`openarmx_ws_copy.tar.gz`、`Manipulation_demo.tar.gz` | 商业资料、其它项目备份 | — | 无关 | 不进根 |
| 散文件：`github-recovery-codes.txt` | GitHub 恢复码 | — | — | **不要碰** |

## 9. 家目录杂项、环境、服务

| 东西 | 是什么 | 状态 | 以后放哪 |
|---|---|---|---|
| `~/alphamotion2_runtime`、`~/alpha-motion-workspace`、`~/.alphamotion_workspace` | 3 个指向主仓（或 `alpha_motion2`）的软链；旧服务靠它们启动 | 链接（承重） | 保留当兼容层；旧服务全停后再撤 |
| `~/codex_exports` | 斯坦福 Canopy 演讲 deck、PDF、预览图 124 MB | 归档 | 不进根 → `~/business/` |
| `~/alphamotion_cold` | 08-08 建的空目录（打算做冷存储） | 空壳 | 删除候选，或改成冷存储盘的挂载点 |
| `~/Downloads/SMPLX_NEUTRAL.npz` | SMPL-X 中性人体模型（被 `gpu_convert` 等写死） | 承重 | P1 拷到 `data/s1_sources/body_models/`（永久），原处留软链 |
| `~/.cache/huggingface` | `acquire.py` 下的 BONES、UniMate 等本体 | 活跃 | 登记成存储池 `hf` |
| conda 环境 `ardy`、`rot`、`genmo-gem`；miniconda base | motion-lab 服务、workbench、`gpu_convert`（genmo-gem 以可编辑方式装了 `GENMO/`）、下载器 | 承重 | `envs/` 里登记 + 生成锁文件；不搬 |
| systemd 服务 `alphamotion-ardy-isaac` | ARDY Isaac 实时控制（依赖的 `GR00T-WholeBodyControl` 已不在） | **坏了：崩溃循环，18:48 已重启 642 次** | 立刻停用（等 lloyd 点头） |
| systemd 服务 `alphamotion-motion-lab`（:2334，对全网开着） | ARDY preflight 20 段的 viser 浏览，占约 745 MB 内存 | 运行中，内容停在 7 月 | 新查看器上线后停 |
| systemd 服务 `alphamotion-rollout-gallery`（:8891，对全网开着） | 20 段 G1 rollout 手机画廊 | 运行中，内容停在 7 月 | 新网页"画廊"页上线后停 |
| systemd 服务 `greenwich-mobile`（:8899，对全网开着） | 08-26 评估证据画廊 | 运行中，内容停在 08-26 | 同上 |
| 进程：HiPHI 下载器 + aria2c `-x16` | 下 HiPHI 分卷 | 运行中（截止约 10-09 早 6 点） | 改成 systemd 单元（P3）；截止前不动 |
| 进程：GRAIL 下载循环 + aria2c `-j48` | 下 GRAIL | 运行中（超过连接上限） | 建议暂停，HiPHI 截止后再开，降到 ≤ 8 连接 |
| 进程：`convert_grail.py --loop`（6 worker）、`grail_collision.py --loop`（4 worker）、`grail_finalize_watch.sh` | GRAIL 转换、碰撞、清单 | 运行中（最多 24 h） | 自然结束后改成作业 |
| 进程：`gpu_convert/run_watch.sh` → `convert.py` | HiPHI 新 part 转 SMPL-X（GPU） | 运行中 | 改成低优先级 GPU 作业 |
| 进程：`umr_gpu/scripts/run_factory.py`（B=1024，40.8 GB 显存） | 引擎作者的吞吐测试 | 运行中 | 跑完后由 GPU 租约排队 |
| 进程：`pretraining_foundation/acquire.py` + `queue_followup.py` | 下 UniMate 模型（约 116 GB） | 运行中 | 建议暂停到 HiPHI 截止后 |
| 进程：`scripts/throttle_guard.sh` | 每 20 秒把漏网进程绑核降级，写 `throttle.log` | 运行中（寿命 48 h） | 留作兜底；改成 systemd 单元 |

---

## 10. 删除候选和归档汇总（都只是提议，lloyd 逐条批）

### 10.1 删除候选（约 210 GB，其中 xwbc_full 一项 129 GB；两个候选方案写的 180 GB 没算 synergy、kimodo 环境和 bench）

| 东西 | 大小 | 为什么能删 | 前提 |
|---|---|---|---|
| ~~`/var/tmp/alphamotion/xwbc_full/`~~ | 129 GB | **10-12 撤出删除候选**：AMASS 缓存已丢，它是唯一 AMASS 源 | 不许删 |
| `~/tmp/cl_v3/amass/` | 15 GB | 只被旧缓存 `cache_amass_v3` 用 | 用 `am what` 确认没有现役缓存链接它 |
| `~/tmp/ufo_resume_drill/`、`ufo_h2_smoke/` | 13.5 GB | 演练和冒烟 checkpoint | 配置先存进 `runs/wbc_ufo/_pilots` |
| `/var/tmp/alphamotion/any_body/env/` | 12 GB | 7 月早期阶段的 conda 环境 | `cache/*.pt` 和说明先归档 |
| `~/alphamotion_ctrl/.git` 垃圾（`git gc`） | 约 9 GB | 散对象和中断的 tmp_pack | 仓库主人执行 |
| `~/reference_code/ufo` 多出的一套 venv | 7.5–10 GB | 两套环境留一套 | 确认 H2 BFM-Zero 用的是哪套 |
| `alpha_motion2/runs/synergy/` | 9.8 GB | 7 月机理分析，已归档 | 结论在 `experiment/synergy` |
| `experiment/kimodo/.venv/` | 5.3 GB | conda 里另有 kimodo | 7 月脚本已归档 |
| `~/gpu_convert/runs/bench*` | 约 5 GB | 吞吐基准 | — |
| `/var/tmp/alphamotion/umr_factory/uniml3d_reload/sample_cache` | 1.3 GB | 样本缓存 | 元数据先拷走 |
| `/var/tmp/alphamotion/gmr_production_v1/` | 0.6 GB | 旧版，无引用 | — |
| `~/alphamotion_artifacts/x2_pool_v1_broken/` | 0.7 GB | 有 bug 的旧版 | — |
| `/var/tmp/alphamotion/xwbc_clips/`、`xwbc_dl/`、`hf_home/`、`beyondmimic/` | 约 0.4 GB | 早期样例、下载脚本、元数据缓存、旧副本 | — |
| `~/tmp/bfm_data_clone/` | 0.2 GB | 和 `reference_code/bfm_zero` 重复 | — |
| `/var/tmp/.../umr_gpu/robots/` | 小 | 和工厂那份逐字节相同 | 引擎改成引用同一份资产后 |
| `omniretarget/` 里重复的 zip | 小于 1 GB | 解包版已在 | — |
| 坏链：`topo_exp/cache_lafan_v2`、`cache_amass_4000`、`/var/tmp/alphamotion/pose_translator_2` 链接 | 0 | 指向不存在或已被账本取代 | — |
| 152 个 wandb 临时目录、`topo_exp/*.bak`/`*.clobbered`、`smplx/_test`、过期 pid 文件 | 约 0 | 残留 | — |
| 0 字节架子目录（`alpha_motion2/{evaluation,geometry,heads,losses,models,modules,training}`、`alpha_motion2/data/*.py`）、`tool-kits/`、`experiment/openfold_ref/`、`pytest-of-arenalabs/`、`mon.pid`、`server_8844.pid`、`MUJOCO_LOG.TXT` | 约 0 | 空壳和运行残留 | — |
| 13 个预览目录里的非最终版 | 约 30 MB | 一次性预览 | 最终版先归档 |
| `research/megatro_scale_pilot_20261004/` | 40 KB | 拼错名的重复目录 | 先并进正确目录 |
| 孪生目录 `~/Desktop/be water, robot` | 557 MB | 引号写错误建 | P0 先救出 3 样东西 |

### 10.2 归档（挪进 `根/archive/`，只读，不删）

- 7 月 ARDY/SONIC/Isaac 演示栈：`alpha_motion2/{demo 其余, web, scripts, synergy, improvisation, agentic_motion, live, scene_planner, isaac_*, configs, tests, notes, docs, deploy}`、`experiment/synergy` → `archive/2026-07_ardy_sonic_stack/`
- Stage-1 路线：`notebook/`、`note/`、`experiment/stage1/`、旧 README → `archive/2026-05_stage1/`
- AnyBody：`any_body/`（主仓和 /var/tmp 两处）→ `archive/2026-07_anybody/`
- Greenwich 8 月杂项：`R4_VIZ`、`R3_REVIEW` 等图、`topo_exp` 旧缓存和非血统 run → `archive/2026-08_greenwich/`
- Studio 0814、`greenwich_eval_20260826`、`wbc_expert`、`pose_translator_2`、`surface_v2`、`labels/pcl_contact`、UMR 侦察实验、`boneseed_spatial_*`、`~/tmp` 8–9 月散产出（打成一个 `tmp_2026-08_09.tar.zst`）

### 10.3 永久保留（`keep`，任何清理计划都不会碰）

HiPHI 原始数据、所有 license 原文和 `sources/`、GRAIL 逐段清单、`robots/` 和登记表、两份 `corr`、`SMPLX_NEUTRAL.npz` 和 `beta_fit`、`cl_v3`（除 `amass/` 子目录外）、`cache_amass_lconv_tfix`/`cache_lafan_v3` 及其链接目标（`cache_xwbc_amass_full`、`cache_xwbc_lafan`、`cache_amass_lconv`）、`bridge_codes`、`text_align`、`cache_amass_heldout`、`topo_exp/runs` 里 v0.2 血统上的 run、`AlphaMotion/weights_staging`、datagrid 数据库备份、`verdicts/`、`MODEL_VERSIONS.md`、`RUNS_INDEX.md`、`DATA_SPEC_v2.md`、技术报告 00–35。
