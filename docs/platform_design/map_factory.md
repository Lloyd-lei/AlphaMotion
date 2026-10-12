# 数据工厂 + 引擎文件夹地图（map_factory）

测绘时间：2026-10-07 18:20 左右（开机 33 分钟，重启后）。只读测绘，除本文件外没有改动任何东西。
大小来自 `du`（有超时），"最后修改" = 该目录 4 层以内最新文件的时间。

状态说明：
- **生产**：现役引擎或数据，不能丢。
- **运行中**：此刻有进程在写。
- **活跃**：最近在用，但没有常驻进程。
- **参考**：只读依据，别人在引用。
- **归档**：已完成的实验，留着备查。
- **淘汰**：已被别的东西取代，可以清理。

"建议归宿"里的 `<ROOT>` 指统一根目录（暂定 `/home/arenalabs/am_platform`），具体布局见文末第 4 节。

---

## 1. 总表

### 1.1 `/home/arenalabs/umr_factory_data`（数据根，190 GB）

| 文件夹 | 用途 | 产出（格式/清单） | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `raw/hiphi` | HiPHI 原始数据（Noitom，617 h，90 Hz，BVH+物体） | `data/*.tar.zst`（分卷）+ `archive_manifest.csv` + `motion_to_part.csv`；`extracted/part_XXXX/`（解压，只留原版、去镜像）；`metadata/*.csv`（全量元数据）；`object_meshes/`；`LICENSE.md` | 119 GB（解压 101 GB，tar 18 GB） | 10-07 18:23 | **运行中**（下载器在下 part 0004，还剩 23 卷约 142 GB，预计 99 h） | 写入方：`scripts_streamc/hiphi_staged_dl.py`。读取方：`gpu_convert/convert.py`、`scripts/convert_hiphi.py`、`diversity/hiphi` | `<ROOT>/data/10_raw/hiphi/` |
| `raw/grail` | NVIDIA GRAIL（62 h，24/25 Hz，从视频重建，带地形和物体） | `data/<category>/{recon/*.pkl, object_usd/*.usd, meta}`；`_tree.json`（锁定 HF 版本）；`_dl_failures.json` | 21 GB | 10-07 18:23 | **运行中**（`grail_dl_loop.py`） | 写入方：`scripts_grail/grail_dl_loop.py`。读取方：`scripts_grail/convert_grail.py` | `<ROOT>/data/10_raw/grail/` |
| `raw/omnicontact` | OmniContact（690 次采集，搬箱/踢球） | `raw_mocap/{box,soccer}` BVH + `prop_*.csv`；`g1_labels/`；`bvh_cache/`；`assets/`；`_npz_tmp/` | 16 GB | 10-04 22:03 | 生产（下载已完成） | 写入方：`scripts/dl_omni.py`、`omni_npz_rest.py`。读取方：`scripts/convert_omni.py`（CPU）、`gpu_convert --kind omni`（还没跑） | `<ROOT>/data/10_raw/omnicontact/` |
| `smplx/hiphi` | HiPHI 转成 SMPL-X，CPU 老路径（90 Hz） | 每段一个目录：`motion_actor_smplx.npz` + `metadata.json` + `qa.json` + `obj_pen.json` + `object_tracks/`；共 1,094 段（分卷 0002/0029） | 3.8 GB | 10-04 | **淘汰中**（已被 GPU 转换器接替，CPU 链不再跑） | 写入方：`scripts/convert_hiphi.py`、`bvh2smplx.py`。读取方：`gpu_convert` 按有无 `qa.json` 跳过已转的段；umr_factory_code `run_clip.py` | 跟 `smplx_gpu/hiphi` 合并，放到 `<ROOT>/data/20_smplx/hiphi/` |
| `smplx/omnicontact` | OmniContact 转成 SMPL-X（CPU） | 每次采集一个目录，按字段存 `.npy`（betas/poses/transl/gender/mocap_framerate…）+ 物体 `.obj/.xml/.csv` + `qa.json`；共 690 段 | 3.0 GB | 10-04 | 生产（已转完） | 写入方：`scripts/convert_omni.py`。读取方：factory `run_clip --kind hoi`、per-robot QA | `<ROOT>/data/20_smplx/omnicontact/` |
| `smplx/grail` | GRAIL 转成 SMPL-X | `<cat>/recon/*.pkl` + `object_mjcf/` + `object_usd/`；**`manifest.jsonl/.csv`（22,312 行，逐段带 license、floor 策略、src_valid）**；`manifest_summary.json`（10,943 段 OK / 28.4 h，9,842 待转，1,527 丢弃）；`_collision_cache/`、`_meshes/`（2.1 GB）；`_convert_rows{,_v2,_v3}.jsonl` | 4.0 GB | 10-07 18:16 | **运行中**（`convert_grail.py --loop`、`grail_collision.py --loop`、`grail_finalize_watch.sh`） | 依赖 `raw/grail`、UMR 环境。被 `diversity/omnicontact_grail` 读取 | `<ROOT>/data/20_smplx/grail/`；它的 manifest 是全平台**唯一一份带逐段 license 的清单**，应该当作 catalog 的样板 |
| `smplx/_test` | 两个测试 npz | `t55.npz`、`t55b.npz` | 很小 | 10-04 | 淘汰 | 无 | 删除或移到 `scratch/` |
| `smplx_gpu/hiphi` | HiPHI 转成 SMPL-X，GPU 新路径（90 Hz） | 目录结构和 CPU 版相同（`motion_actor_smplx.npz`+`metadata.json`+`qa.json`）；共 3,796 段；`convert_log.jsonl`、`last_run_summary.json`、`parts_status.json`、`parts_done.txt`、`watch.log/.pid` | 13 GB | 10-07 18:08 | **运行中 / 生产**（`gpu_convert/run_watch.sh` 每 2 分钟扫一次） | 依赖 `raw/hiphi/extracted`、genmo-gem 环境、`~/Downloads/SMPLX_NEUTRAL.npz`、`/var/tmp/.../beta_fit`。读取方：factory、diversity | `<ROOT>/data/20_smplx/hiphi/fps90/` |
| `smplx_gpu/hiphi_30fps` | 同一批 HiPHI 降到 30 fps | 同上（3,795 段） | 4.4 GB | 10-07 | 运行中 | 同上；训练用的帧网格是 30 fps | `<ROOT>/data/20_smplx/hiphi/fps30/`（CPU 那 1,094 段没有 30 fps 版本，要补） |
| `robots/` | 21 台核心人形机器人的工厂暂存资产（**机器人登记的实际落地**） | 每台：`<r>.xml`（展平的 MJCF）+ `mesh/`（硬链接或拷贝）+ `umr_robot.json` + `staging.json`（T-pose 和碰撞排除对）+ **`surface.npz`（4096 个外表面点、法线、geom/body/region）** + `umr_shared_beta0.json`；全局：`staging_summary.json`、`surface_summary.json`、`tpose_contact_sheet.png` | 1.1 GB | 10-07 18:15 | **生产** | 写入方：`umr_factory_code/scripts/stage_robots.py`、`build_surfaces.py`；来源是 `robots.json` 指向的 MJCF（16 台在 Desktop 仓 `external/GMR`，5 台在 `/var/tmp/alphamotion/robot_models`）。读取方：CPU/GPU 引擎、umr_dataset、motion_audit_web | `<ROOT>/data/30_robots/<r>/assets/` |
| `robots/<r>/pipeline/` | 每台机器人自己的 QA 流水线（robots_wf） | `pipeline.json`（冻结配置）、自带的 `measure.py/run_qa.py/render_qa.py/make_report.py`、`out/`、`runs/`、日志 | 每台约 100 MB；目前只有 5/21 台有（g1_23、g1_29、h1、h2、r1） | 10-07 18:15 | **运行中**（robots_wf 代理） | 依赖 factory CPU 代码和 `smplx/`。**每台各写一份测量脚本** | 配置 → `<ROOT>/data/30_robots/<r>/pipeline.json`；脚本合并进统一判官 `<ROOT>/code/judges/`；产物 → `60_qa` |
| `corr/` | 人体到机器人的表面对应关系缓存（UMR correspondence） | `<r>/smplx/{correspondence_dataset.npz, train/}`、`<r>/beta0/`（共享 beta0 版）、哈希目录；`correspondence_summary.json`、`shared_beta0_summary.json` | 4.0 GB | 10-04 23:51 | **生产** | 写入方：`build_correspondence.py`、`build_shared_corr.py`。读取方：CPU 引擎。**GPU 引擎用的是自己那份（`/var/tmp/.../umr_gpu/corr`），字节数不一样** | `<ROOT>/data/30_robots/<r>/corr/` |
| `runs/` | CPU 引擎跑出的 UMR 原始结果（7 段） | `<clip>/<r>.*`（UMR 原始 qpos、`qpstats.json`、人体 4096 槽位） | 1.3 GB | 10-05 00:07 | 归档（开发样本） | 被 `out/*.npz` 的 `provenance.human_slots` 引用，motion_audit_web 读人体点云要用 | `<ROOT>/data/40_retarget/cpu_ref/runs/` |
| `out/` | **CPU 工厂最终输出（`factory_npz_v1`）** | `<clip>/<robot>.npz`（11 个：7 段 × G1/GR3/H2/G1-23）+ `.qa.json`；`manifest.jsonl`（逐条）、`manifest.json`、`qa_summary.json`；`tests/` | 33 MB | 10-05 00:16 | 活跃（样本级，还没量产） | 读取方：`umr_dataset` 转换器、`motion_audit_web`、`audit/SCHEMA.md`。注意：10-04 22:47 之前写的文件只用视觉量尺验过 | `<ROOT>/data/40_retarget/<engine>/<clip>/<robot>.npz`（和 GPU 输出统一布局） |
| `qa/` | 逐台机器人的 QA 报告 | 5 台，**每台布局都不一样**（`measure_*.json`、`*.perframe.npz`、`grid12.png`、mp4、`PROGRESS.md`、`report.json`）；只有 g1_29 有 `report.json` | 75 MB | 10-07 18:15 | **运行中**（robots_wf） | 依赖 `robots/<r>/pipeline`；今天在这里发现引擎 bug（`umrf/objects.py` 把立方体当成外接球） | `<ROOT>/data/60_qa/robots/<r>/`（要先统一格式） |
| `qa_renders/` | SMPL-X 转换质检截图 | 各数据源 PNG + `grail/` | 11 MB | 10-04 23:35 | 归档 | 无 | `<ROOT>/data/60_qa/conversion/` |
| `diversity/` | 预训练数据多样性普查 | 各源 `summary.json/.md`、逐段标签 `*_clip_labels.csv(.gz)`、`hiphi_clips_labeled.csv`、`grail/omnicontact_clip_stats.jsonl`、`charts/*.png` + `synthesis_numbers.json`；**`REPORT.md` 还没生成** | 19 MB | 10-07 18:19 | **运行中**（diversity_wf 合成阶段） | 读取 raw 元数据、`smplx/`，以及 `/var/tmp/.../sources`、`/var/tmp/alphamotion/xwbc_*`、Desktop 仓 `pretraining_foundation` | 脚本 → `<ROOT>/code/diversity/`；产物 → `<ROOT>/data/60_qa/diversity/<日期>/` |
| `audit/` | 人工审计接口约定（Claude 和网页之间） | `SCHEMA.md`（定义 queue/decisions/issues.jsonl 三个文件和三种格式）、`CODEX_PROMPT.md`；**这三个 jsonl 文件一个都还没有** | 20 KB | 10-04 23:09 | 参考 / 约定 | 被 `motion_audit_web` 当成唯一接口 | `<ROOT>/data/60_qa/audit/` + 约定文档放 `<ROOT>/docs/contracts/` |
| `logs/` | 所有下载、转换、守护进程的日志和 pid | `throttle.log`（负载/GPU/网络）、`hiphi_streamc_{status,events,state}.json`、`hiphi_convert.jsonl`、`omni_convert.jsonl`、`grail/*`（**`dl_loop.log` 406 MB**，不轮转）、一堆重启前留下的 pid 文件 | 450 MB | 10-07 18:23 | 运行中 | 下载器和守护进程写；diversity 读 status | `<ROOT>/data/logs/<服务>/`，加日志轮转，过期 pid 要清 |
| `pilot/` | 早期试点诊断 + 临时目录 | `diag/coll_vs_vis.{py,json}`（视觉量尺和碰撞量尺的对比）、`tmp/`（今天还有人写 GRAIL 网格图） | 28 KB | 10-07 18:11 | 临时 | 无 | `<ROOT>/scratch/` |
| `hf_cache/` | HuggingFace 元数据缓存 | `datasets--*`、`grail/xet` | 15 MB | 10-04 | 缓存 | 下载器 | `<ROOT>/cache/hf/` |
| `workflows/` | Claude 工作流脚本 | `diversity_wf.js`、`robots_wf.js`（21 台按 2 台一波跑 QA） | 20 KB | 10-07 18:04 | 运行中 | `diversity_wf.js` **写死了 `/tmp/claude-1000/.../scratchpad/am2exp/scout_sources.md`**（临时目录，会消失） | `<ROOT>/code/workflows/` |
| `scripts/` | 第一代脚本：HiPHI/Omni 下载、CPU 版 BVH→SMPL-X、质检、节流守护 | `dl_*.py`、`convert_hiphi.py`、`convert_omni.py`、`bvh2smplx.py`、`smplx_qa.py`、`obj_pen_audit.py`、`build_manifest.py`、`render_clip.py`、`*_chain.sh`、**`throttle_guard.sh`（运行中，寿命 48 h）** | 120 KB | 10-07 18:03 | 大部分**淘汰**（HiPHI 已换 GPU）；Omni 转换和 `throttle_guard` 还在用 | 依赖 `/var/tmp/.../{hiphi2smplx,h2s_shim,beta_fit}`、`~/Downloads/SMPLX_NEUTRAL.npz` | `<ROOT>/code/ingest/{download,convert}/`、`<ROOT>/code/ops/` |
| `scripts_grail/` | GRAIL 整条流水线 | `grail_dl_loop.py`、`convert_grail.py`（v3，旁边有 v1/v2 的 `.bak`）、`floorcal2.py`、`grail_collision.py`、`build_manifest.py`、`grail_finalize_watch.sh`、`render_grid.py` | 272 KB | 10-04 23:38 | **运行中 / 生产** | 依赖 UMR 环境 | `<ROOT>/code/ingest/grail/` |
| `scripts_streamc/` | HiPHI 分卷下载器（下载 → 校验 sha → 解压 → 删 tar，有磁盘水位保护） | `hiphi_staged_dl.py`（+ v1 的 `.bak`）、`rank_hiphi_parts.py` | 116 KB | 10-04 22:47 | **运行中**（寿命 37 h） | 写 `raw/hiphi`、`logs/hiphi_*` | `<ROOT>/code/ingest/hiphi/` |
| `DATA_SPEC_v2.md` | 数据规格 v2（实接触 / 不穿 / 不滑 / 不闪 / 和人一致） | — | 2 KB | 10-04 | **参考（约束一切）** | 被所有引擎、判官、网页引用 | `<ROOT>/docs/specs/DATA_SPEC_v2.md` |
| `RESOURCE_LIMITS.md` | 资源上限 | — | 1 KB | 10-07 | 参考 | 所有任务 | `<ROOT>/docs/ops/` |

### 1.2 `/home` 下的代码目录（**全部没有 git**）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `~/umr_factory_code` | **CPU 参考工厂**（`umrf` 包）：暂存机器人 → 稠密表面 → 对应关系 → 跑 UMR → 严格后处理（Clarabel QP）→ 标签 → QA | 写到 `umr_factory_data/{robots,corr,runs,out}`；`out/manifest.jsonl`、`qa_summary.json` | 468 KB | 10-04 22:49 | **生产（参考实现 + 量尺）**，代码基本冻结 | `umrf/paths.py` **把所有路径写死**：`/var/tmp/.../umr/UMR`、UMR 环境、`robots.json`、`umr_configs`、`am2exp/surface_v2`（只声明了，没用上）。被依赖：umr_gpu `eval/factory_ruler_snapshot`（冻结拷贝）、umr_dataset `strict_recheck`（`RobotGeometry`）、5 台 per-robot 流水线。**已知 bug：`objects.py` 把 8 顶点的盒子判成球体（所有搬箱片段的物体接触都失效）**；有 `.bak_taskB` 文件 | `<ROOT>/code/retarget/umr_cpu/` |
| `~/umr_gpu` | **GPU 批量 UMR 重定向 + v2 严格处理（生产引擎）** | 写到 `/var/tmp/.../umr_gpu/{problems,results,videos,logs}`；导出格式 = `gpu_npz_v2`（CSR 稀疏接触 `contact_idx/ptr`、单位 mm、`qc_json`） | 996 KB | 10-07 18:12 | **生产 / 运行中**（`run_factory.py` 吞吐测试，4 台 B=256，36 GB 显存，超时 3000 s）。**和 CPU UMR 做过一致性测试**：关节 p50 约 1e-5 rad；独立判官 `eval/judge.py` 结果见 `results/parity_tables.md` | 依赖 UMR 环境、`umr_factory_data/robots`（拷贝到自己的 robots/corr）、CPU 抓取的 problems。被依赖：还没有下游能读它的导出格式（umr_dataset 和网页都只认 v1） | `<ROOT>/code/retarget/umr_gpu/` |
| `~/gpu_convert` | **GPU 批量 BVH→SMPL-X 转换器**（HiPHI/Omni），输出布局和 CPU 版完全相同 | `smplx_gpu/hiphi{,_30fps}`；`runs/` 里是一致性测试和基准（parity40、cpubeta…；`bench4` 有 5 GB） | 6.5 GB（代码 200 KB，`.inductor_cache` 921 MB，runs 5.6 GB） | 10-07 18:05 | **生产 / 运行中**（`run_watch.sh` → `convert.py --lanes 256`，11 GB 显存） | 用 genmo-gem 环境；写死了 `~/Downloads/SMPLX_NEUTRAL.npz`、`/var/tmp/.../{beta_fit,hiphi2smplx,h2s_shim}`；`convert_v1.py`、`gpu_ik_ref.py` 已过时 | `<ROOT>/code/ingest/convert_gpu/`；`runs/bench*` → `scratch/` |
| `~/umr_dataset` | **训练数据打包器 + 加载器**（UMRS v1 分片，有校验和，可 mmap；人体和 21 台机器人成对取窗口） | `data/manifest.json`（逐分片 sha256）、`catalog.npz` + `catalog_meta.json`、`clips.jsonl`、`ingest_index.npz`、`robots/<r>/{spec.json,kin.npz,bounds.json,*.umrs}`、`human/*.umrs`、`objects/`；QA：`qa/roundtrip_7entries.json`、`strict_recheck_7entries.json` | 23 MB（只有 7 条） | 10-05 00:19 | 生产（代码）/ 空闲（watch 没在跑） | 用 UMR 环境。**`sources.py` 去 `UMR/sample_data/hiphi` 找 HiPHI 人体，所以工厂转出来的 HiPHI 段会失败**（`logs/watch.log`：get_up_0004 FileNotFound）；只认 `factory_npz_v1` | 代码 → `<ROOT>/code/dataset/`；`data/` → `<ROOT>/data/50_packed/umrs_v1/` |
| `~/motion_audit_web` | **人工审计工作台**（FastAPI + React/Vite/Three.js，127.0.0.1:8765，闲置 8 小时自动退出） | 判决追加写到 `umr_factory_data/audit/{decisions,issues}.jsonl`；`cache/arrays` 是解压后的 npy 缓存；`qa/` 是验收截图 | 3.2 GB（cache 3.1 GB，frontend 113 MB，.venv 31 MB） | 10-07 18:20 | 活跃（服务现在没开；**18:20 有人在建 `.mcp-venv`**，可能是另一个子任务在做 MCP） | `backend/app.py` 写死 `DATA_ROOT=/home/arenalabs/umr_factory_data`；`adapters.py` 读 `/var/tmp/.../umr/UMR/assets/smplx_parts_segm.pkl`。只支持 `factory_npz_v1`，遇到 `gpu_npz_v2`/`packed_v1` 会报"不支持" | `<ROOT>/code/service/web/`（作为 HTTP API + 网页的起点）；cache → `<ROOT>/cache/` |

### 1.3 `/var/tmp/alphamotion/umr_factory`（18 GB，在根分区上，根分区已用 88%，只剩 76 GB）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `umr/UMR` | 上游 UMR 代码（commit d1cdaed） | 已跟踪文件没改；多了未跟踪的 `data/`、`output/`（HOI 对应缓存，约 640 MB）、`robot_configs/..._umrfactory.json`、`assets/fourier_gr3v2_1_1/`、`sample_data/lafan1_smplx`（LAFAN 样本源） | 1.2 GB | 10-07 | **生产** | 被 CPU/GPU 引擎、umr_dataset、网页读 | `<ROOT>/third_party/UMR`（锁定 commit；运行时缓存挪出去） |
| `umr/env` | **几乎所有程序共用的 Python 环境**（3.12，mujoco 3.3.7，torch） | — | 6.7 GB | — | **生产** | 被 factory、umr_gpu、GRAIL 流水线、umr_dataset 使用 | `<ROOT>/envs/umr/`（另存一份锁版本文件） |
| `umr/tools` | UMR 侦察期写的包装：`umr_run.py`（QP 失败回退）、`umr_pipe.py`、评测和渲染 | — | 很小 | 10-04 | **生产**（`umrf/pipeline.run_umr` 要用 `umr_run.py`） | 被 CPU 工厂依赖 | `<ROOT>/code/retarget/umr_cpu/tools/` |
| `umr/{configs,eval,runs,logs,videos}` | UMR 侦察实验（G1/GR3 的默认参数、无地面、自碰撞硬约束对照） | json/npz/mp4 | 约 0.6 GB | 10-04 | 归档 | 无 | `<ROOT>/archive/2026-10-04_umr_scout/` |
| `robots/` | **机器人登记表**：`robots.json`（34 台，其中 21 台 A_core20，带完整的 MJCF 体检字段）、`robots_raw.json`、`robots_table.md`、`umr_configs/`（30 份 UMR 草稿配置）、`umr_ref/`、`inventory.py`、`finalize.py` | json/md/png | 3.4 MB | 10-04 18:19 | **生产**（就是代码里的 `INVENTORY` 和 `DRAFT_CONFIGS`） | 被 `umrf/robots.py`、`stage_robots.py`、两个工作流读 | `<ROOT>/data/30_robots/registry.json` + `registry/umr_configs/`（要从 /var/tmp 挪走） |
| `umr_gpu/` | GPU 引擎的产物 | `problems/`（CPU 抓取的逐帧问题，5.7 GB）；`results/{parity,export,export_bench,bench,factory_test}` + `parity_tables.md`；`videos/`（24 个对比 mp4 + 网格，1.6 GB）；`robots/`（4 台，和工厂那份逐字节相同）；`corr/`（4 台，**和工厂那份不同**）；`runs/`、`sources/`、`logs/` | 7.7 GB | 10-07 18:23 | **运行中** | 写入方：`~/umr_gpu`。`results/export/*.npz` 是 `gpu_npz_v2` 的唯一样本（12 个） | 导出 → `<ROOT>/data/40_retarget/gpu/`；parity/bench → `60_qa/engine/`；problems → `scratch/`（能重新生成） |
| `labels/` | 点云接触标签器原型（spec v0.1，`pcl_contact/`，ardy 环境） | `out/labels_*.npz`、`report_*.json`、`session_floor.json` | 31 MB | 10-04 18:31 | **淘汰**（已被 `umrf/labels.py` 和 `umr_gpu/strict.py` 接替） | 无 | `<ROOT>/archive/` |
| `sources/` | 数据源侦察：HF 文件树、README、**LICENSE 原文**、`sources_inventory.json`（逐源 license/访问权限/格式/小时数）、`bones_meta_v004.parquet`、UMR 适配器笔记 | json/md/parquet | 33 MB | 10-04 18:20 | **参考（license 追踪的唯一原始记录）** | 被 diversity_wf、umr_dataset `sources.py` 引用 | `<ROOT>/data/00_sources/`（license 进 catalog） |
| `uniml3d_reload/` | UniML3D / Mixamo 重载原型 + 样本缓存（Objaverse-XL） | `proto/`、`remote/*.csv/json`、`sample_cache/`（1.3 GB） | 1.3 GB | 10-04 18:27 | 归档 / 探索 | 被 diversity 读元数据 | 元数据 → `00_sources/uniml3d/`；缓存 → `archive/` 或删 |
| `lloyd_gpu_eval/` | GPU 和 CPU 的第一次对比（舞蹈/踢球 × G1/GR3） | `cpu_*.npz`、`*.gpu.npz`、对比 mp4、`stat_*.json` | 45 MB | 10-04 22:25 | 归档，**但 `umr_gpu/eval/run_parity.py` 还写死了这里的 `cpu_*.npz`** | 被 run_parity 引用 | `<ROOT>/data/60_qa/engine/2026-10-04_first_parity/` |
| `hiphi2smplx/`（git）、`h2s_shim/`、`beta_fit/` | HiPHI BVH→SMPL-X 的第三方代码（MIT）、h5py 垫片、beta 先验（gmm_08.pkl、mean params） | — | 63 MB + 16 KB + 836 KB | 10-04 19:22 | **生产**（GPU 和 CPU 转换器都要用） | 被 `gpu_convert/{convert,gpu_beta,prep,parity}.py`、`scripts/bvh2smplx.py` 依赖 | `<ROOT>/third_party/hiphi2smplx/` + `<ROOT>/assets/body_models/`（和 `~/Downloads/SMPLX_NEUTRAL.npz` 放一起） |

### 1.4 `/var/tmp/alphamotion/am2exp`（3.0 GB）

| 文件夹 | 用途 | 产出 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `am2exp/`（总） | AlphaMotion2 实验 v1 第 0 阶段（**模型训练和评测这一侧**，不属于数据工厂） | `stage0_index.md`（总索引和闸门判决：量尺闸门 FAIL 6/15，打乱源对照 FAIL，E2 dev 0.765 但 test 0.539） | 3.0 GB | 10-04 18:36 | **归档**（第 0 阶段已结束；三联视频 lloyd 还没看） | 代码在 Desktop 仓 `alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1` | `<ROOT>/data/70_models/am2exp_v1/` |
| `am2exp/assets/surface_v2` | **另一套**机器人表面资产（绑定在 xwbc GMR 的 MJCF 上；18 个 body，含 franka、adroit） | `manifest.json`、png、`validation/` | 17 MB | 10-04 16:33 | 归档 | `umrf/paths.SURFACE_V2` 声明了但没用；robots.json 有 `surface_v2_asset` 字段；kuavo_s45、h1_2 绑定有问题 | 和 `30_robots/<r>/surface.npz` 二选一，只保留一套 |
| `am2exp/judge` | 判官 J1：`ruler_v1/`、`bindings_v1/`、`stage0_report.md`、`heldout_root_bias_v1.json` | json/md | 4 MB | 10-04 16:56 | 归档 | — | `<ROOT>/code/judges/`（量尺部分）+ `70_models/.../judge/` |
| `am2exp/eval`、`data`、`integration`、`review`、`runs`、`videos`、`model` | E1/E2 评测用例（`cases_e1_v1.json`+sha）、A0/B0 推理结果（2.9 GB）、联调 dryrun、`ckpt/S_untrained`、三联视频 [人 \| GT \| 模型] | json/npz/mp4/pt | 2.9 GB + 约 100 MB | 10-04 18:36 | 归档 | — | `<ROOT>/data/70_models/am2exp_v1/{eval,ckpt,videos}` |

---

## 2. 重叠和冲突（按严重程度排）

1. **GPU 生产引擎的输出现在没人能读。** `umr_gpu` 导出的 `gpu_npz_v2` 和 CPU 的 `factory_npz_v1` 有这些不同：
   - 键名不同（`ground_clearance_mm` 对 `ground_clearance`）；
   - 单位不同（mm 对 m）；
   - 点级接触存法不同（CSR 稀疏对 packbits）；
   - 缺 `joint_names`、`provenance`、物体字段；
   - 布局不同：平铺的 `<clip>__<robot>.npz` 对 `<clip>/<robot>.npz`。

   `umr_dataset` 和 `motion_audit_web` 都只认 v1；`audit/SCHEMA.md` 里 `gpu_npz_v2` 写着"开发中"，没有字段说明。量产之前必须先统一格式（最好让 GPU 导出就是唯一格式）。
2. **已知引擎 bug 正在污染搬箱数据。** `umrf/objects.py` 先用"顶点到中心等距"判球体，结果把 8 顶点的盒子判成外接球。后处理就把手推到球面上：手离箱面 p50 有 37–44 mm，物体实接触 0%。这影响 `out/` 里所有 box 片段，以及由它们打包出来的东西。GPU 引擎的 `geometry.py` 有自己的 SDF，是否有同样的问题**没有核实**。
3. **HiPHI 的 SMPL-X 分在两处**：`smplx/hiphi`（CPU，1,094 段）和 `smplx_gpu/hiphi`（GPU，3,796 段）。只有 GPU 那份有 30 fps 版本；`object_meshes` 存了 3 份（每份 119 MB）。两个转换器靠"有没有 `qa.json`"互相跳过，没有统一清单。
4. **SMPL-X 有 4 种存法**：
   - HiPHI：`motion_actor_smplx.npz`；
   - Omni：按字段分开的 `.npy`；
   - GRAIL：`recon/*.pkl` + manifest；
   - LAFAN：放在 UMR 的 `sample_data` 里。

   `umr_dataset/sources.py` 要一个个适配，HiPHI 现在已经因为路径错了而失败。
5. **机器人资产有 5 份**：
   - 工厂 `robots/`（21 台，权威）；
   - `umr_gpu/robots`（4 台，逐字节相同）；
   - `umr_dataset/data/robots`（21 台，派生的 FK 表）；
   - `am2exp/surface_v2`（另一套绑定）；
   - 登记表 `robots.json` 在 /var/tmp。

   对应关系 `corr/` 和 `umr_gpu/corr` **字节数不同**，有漂移风险。
6. **判官 / 量尺至少 8 把**：
   - `umr_gpu/eval/judge.py`；
   - `umr_gpu/eval/factory_ruler_snapshot`（umrf 的冻结拷贝）；
   - `umrf` 后处理自己的复验；
   - 5 台机器人各自的 `measure.py`；
   - `motion_audit_web/backend/geometry.py`；
   - `umr_dataset/scripts/strict_recheck.py`；
   - `am2exp/judge/ruler_v1`；
   - `labels/pcl_contact`（已淘汰）。

   阈值也不一致：provenance 里的接触滞回是 1 cm/2 cm、打滑 3 cm/s，而 DATA_SPEC_v2 是 1 mm/2 mm、1 cm/s。CPU 工厂故意保留了 UMR 自己的打滑（`LATCH_PIN=0`），**达不到 v2 的"不滑"**。
7. **QA 输出有 3 处、格式各不相同**：工厂 `out/`、`robots/<r>/pipeline/out/`、`/var/tmp/.../umr_gpu/results/`。`qa/<robot>/` 里 5 台的文件布局也互不相同。
8. **6 个生产代码目录全都没有 git**，包括 umr_factory_code、umr_gpu、gpu_convert、umr_dataset、motion_audit_web、umr_factory_data/scripts*。现在靠 `.bak`、`.bak_taskB`、`convert_v1.py`、`.convert_grail_v2.py.bak` 当版本管理。GPU 引擎一致性测试用的是哪个版本的代码，无法追溯。
9. **Python 环境有 5 个以上**：
   - UMR 环境（/var/tmp，大多数程序在用）；
   - genmo-gem（gpu_convert）；
   - miniconda base（下载器、build_manifest）；
   - motion_audit_web 的 `.venv` 和 `.mcp-venv`；
   - ardy（labels 原型）。

   没有锁版本文件，只有网页项目有 `environment-freeze.txt`。
10. **生产资产放在 /var/tmp**：UMR 代码、共用环境、`robots.json`、GPU 产物、license 原文都在这里，而根分区已经用了 88%。tmpfiles 的 30 天自动清理规则目前是注释掉的，所以暂时不会被自动删，但这个目录本来就是临时目录的意思。
11. **写死的路径**（换根目录前都得改）：
    - `umrf/paths.py`（全部路径）；
    - umr_gpu 里有 22 处指向自己的目录，另有若干处指向 /var/tmp；
    - `gpu_convert`、`scripts/bvh2smplx.py`、`smplx_qa.py` 里的 `~/Downloads/SMPLX_NEUTRAL.npz`；
    - `umr_dataset/sources.py` 里的 UMR `sample_data`；
    - `motion_audit_web/backend/app.py` 里的 `DATA_ROOT`；
    - `umr_gpu/eval/run_parity.py` 里的 `lloyd_gpu_eval/cpu_*.npz`；
    - `diversity_wf.js` 里的 `/tmp/claude-1000/.../scratchpad`；
    - `robots.json` 里的 MJCF 来源（Desktop 仓 `external/GMR` 和 `/var/tmp/alphamotion/robot_models`）。
12. **license 分散在 3 处**：`/var/tmp/.../sources/sources_inventory.json`（逐源）、`smplx/grail/manifest.*`（逐段，只有 GRAIL 有）、`raw/*/LICENSE.md` 和 `README`。这几个数据源全部是非商用：
    - HiPHI 的 ModalityNet 许可写明"支持商业产品的内部研发也算商用"，并且不许再分发；
    - GRAIL 和 OmniContact 是 CC BY-NC 4.0；
    - BONES-SEED 用的是 bones-seed-license。

    以后的 catalog 必须逐段带 license。

## 3. 生产关键清单

**做过一致性测试的引擎（不能动，只能纳入版本管理）**
- `~/umr_gpu`：和 CPU UMR 做过一致性测试，单步 2e-6 rad，自由运行 p50 1.3e-6 rad；有独立判官表 `parity_tables.md`。最新吞吐是 4 台机器人一次跑，B=256 时每秒 2,707 机器人帧。
- `~/umr_factory_code`：CPU 参考实现和量尺，再加上 `/var/tmp/.../umr/{UMR@d1cdaed, env, tools/umr_run.py}`。
- `~/gpu_convert`：和 CPU 转换器做过一致性测试，`runs/parity40`。还依赖 `hiphi2smplx`、`beta_fit`、`h2s_shim` 和 `~/Downloads/SMPLX_NEUTRAL.npz`。
- `~/umr_dataset`：分片格式和加载器，有往返误差和严格复验的测试。
- 资产：`umr_factory_data/robots/`、`corr/`，以及 `/var/tmp/.../robots/robots.json` 和 `umr_configs/`。

**此刻在跑的进程（18:20 快照）**

| 进程 | 工作目录 / 写到哪里 | 寿命 |
|---|---|---|
| `scripts_streamc/hiphi_staged_dl.py` + aria2c `-x16` | `raw/hiphi` | 37 h |
| `scripts_grail/grail_dl_loop.py` + aria2c **`-j48`** | `raw/grail` | 25 h |
| `scripts_grail/convert_grail.py --loop`（6 个 worker） | `smplx/grail` | 25 h |
| `scripts_grail/grail_collision.py --loop`（4 个 worker） | `smplx/grail/_collision_cache` | 25 h |
| `scripts_grail/grail_finalize_watch.sh` | 每 10 分钟重建 GRAIL manifest | 26 h |
| `gpu_convert/run_watch.sh` → `convert.py --lanes 256`（显存 11 GB） | `smplx_gpu/hiphi{,_30fps}` | 4 h 一轮，循环 |
| `umr_gpu/scripts/run_factory.py`（吞吐测试，显存 36 GB） | `/var/tmp/.../umr_gpu/results/export_bench` | 3000 s |
| `scripts/throttle_guard.sh` | `logs/throttle.log` | 48 h |
| robots_wf / diversity_wf 代理 | `qa/`、`robots/<r>/pipeline/`、`diversity/charts` | — |
| （范围外）Desktop 仓 `alpha_motion2/pretraining_foundation/acquire.py` + `queue_followup.py` | BONES-SEED 等的下载 | — |

**违反 RESOURCE_LIMITS 的地方（只记录，没有处理）**
- 两个 aria2c 加起来最多 16 + 48 = 64 个连接，超过上限 16。
- 两个重 GPU 任务同时在跑（`run_factory` 吞吐测试和 `gpu_convert`），GPU 利用率 75–81%，超过目标 70%。
- `throttle.log` 在 18:18:48 记了一次 net=down。

## 4. 建议的统一布局（给汇总任务参考）

```
<ROOT>=/home/arenalabs/am_platform
  code/                    一个 git 仓（monorepo）
    ingest/{hiphi,grail,omnicontact,download,convert_cpu,convert_gpu}   ← scripts*, scripts_grail, scripts_streamc, gpu_convert
    robots/                机器人登记、暂存、表面采样        ← umrf/robots.py, surface.py, stage_robots, build_surfaces, /var/tmp robots/*.py
    retarget/{umr_cpu,umr_gpu}                                ← umr_factory_code, umr_gpu（含 umr/tools）
    judges/                唯一一把量尺 + 对照臂               ← umr_gpu/eval/judge.py 为主，合并 per-robot measure.py、am2exp ruler
    dataset/               打包和加载                         ← umr_dataset/umr_dataset
    diversity/             普查和分类                          ← diversity/*/scripts
    service/{api,mcp,web}  一个服务，同时提供 MCP 和 HTTP        ← motion_audit_web（后端和前端）
    workflows/  ops/       工作流、节流守护、watcher
  third_party/{UMR@d1cdaed, hiphi2smplx}   锁版本，运行时缓存挪出去
  assets/body_models/      SMPLX_NEUTRAL.npz, beta_fit/*
  envs/                    umr 环境 + genmo-gem 的锁文件（实体可以继续放原处，用软链）
  docs/{specs,contracts,ops}   DATA_SPEC_v2, audit SCHEMA, RESOURCE_LIMITS
  data/
    catalog/               唯一事实来源（逐段：源、license、各阶段状态、sha、QA 指标、审计判决）
    00_sources/  10_raw/  20_smplx/<src>/{fps90,fps30}/
    30_robots/<r>/{assets,corr,pipeline.json}  registry.json
    40_retarget/<engine>/<clip>/<robot>.npz    （只用一种格式）
    50_packed/umrs_v1/
    60_qa/{conversion,robots,engine,diversity,audit}
    70_models/<exp>/{ckpt,eval,videos}         ← am2exp
    logs/
  cache/  scratch/  archive/
```

搬迁时要注意：
- 正在跑的下载器和转换器里写死了路径，必须等它们自然结束，或者先改成从配置里读路径（`AM_ROOT`）再搬。
- `/var/tmp` 里的东西最好整个搬到 /home，旧位置留软链，这样不用一次改完所有引用。
