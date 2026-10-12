# AlphaMotion 统一平台设计（方案：包住现有的，少折腾）

> 2026-10-07 起草。依据：同目录下的 4 张地图（`map_research_repo.md`、`map_factory.md`、`map_caches_artifacts.md`、`map_model_tools.md`），外加对 `DATA_SPEC_v2.md`、`audit/SCHEMA.md`、`umr_gpu/README.md`、`umr_dataset/README.md`、`motion_audit_web`（含今天新建的 MCP 雏形 `agent_tools/`）的只读核对。
> 全程只读，没有移动、删除或修改任何东西，只写了本文件。
> 本文的「删除」一律是**提议**，执行前要 lloyd 逐条点头。

---

## 0. 一页纸看懂

**核心思路有 4 条：**

1. **代码搬家，数据挂链。**
   - 代码都很小，不算 node_modules 和环境的话加起来不到 50 MB：`umr_gpu` 1 MB、`umr_factory_code` 0.5 MB、`exp_topo` 8 MB……全部收进一个 git 仓。
   - 大数据（790 GB）先原地不动，用软链挂进新根目录，再在 catalog 里登记路径。
   - 新根目录放在 `/home` 分区。`~/umr_factory_data` 也在 `/home`，所以以后真正搬的时候只是改名，一秒完成，不用拷贝。
2. **做过一致性测试的引擎一行都不改。**
   - 包括 `umr_gpu`、`umr_factory_code`、`gpu_convert`、`umr_dataset`。
   - 外面加一层"适配器"，统一输入输出。
   - 引擎先原地打 git 快照，把"测一致性时用的是哪版代码"钉住。
3. **不发明新文件格式，统一放在读取层做。**
   - GPU 引擎写出来的 `gpu_npz_v2` 原样保存，字节不动，这样一致性测试的信任不会丢。
   - 缺的 `joint_names`、来源信息放进旁边的 `.meta.json` 和 catalog。
   - 一个读取函数 `amp.formats.motion.load()` 同时认 v1 和 v2，统一返回米制、地面为 0 的对象。下游（打包器、审计网页、判官、查看器）都只调它。
   - **模型生成的动作也存成同一种格式**。这样数据用的查看器、判官、审计台可以直接拿来看模型、评模型。整个设计主要省力在这一点。
4. **一个服务，两个入口。**
   - 同一份 Python 核心 `amp/service/core.py`：
     - 对 Claude 和代理开 MCP。底座就是今天已经在 `motion_audit_web/agent_tools/` 里搭起来的 MCP，沿用它"先出计划、再提交、带幂等键"的作业模式。
     - 对 lloyd 和 Codex 开 HTTP API 和网页。底座是 `motion_audit_web` 的 FastAPI 和 React 前端。
   - 作业层由两套现成的东西合成：`am2exp_v1/ops`（GPU 租约、磁盘守卫、回执、人工批准文件）和 `arena-datagrid` 的 `jobs` 表（租约、事件）。

**1–2 天内开始量产数据**：
- 第 1 天：搭骨架、打 git 快照、导入 catalog、写读取层。
- 第 2 天：写重定向适配器，先查一遍 GPU 引擎对箱子的处理，然后跑第一批 GRAIL 地形 + HiPHI（不带箱子的片段）+ LAFAN × 21 台。
- 详见 §E 和 §F。

**要 lloyd 拍板的事（按急迫程度排）**：

| # | 事项 | 建议 |
|---|---|---|
| 1 | `alphamotion-ardy-isaac` 服务在崩溃循环里，已重启 370 多次 | `systemctl --user disable --now alphamotion-ardy-isaac` |
| 2 | 两个 aria2 下载器加起来开了 64 个连接（上限 16），18:18 掉过一次网 | GRAIL 从 `-j48` 降到 `-j8`，HiPHI 从 `-x16` 降到 `-x8` |
| 3 | 两个重 GPU 任务同时在跑（`run_factory` 吞吐测试和 `gpu_convert`），利用率 75–81% | 测试跑完以后，同一时间只让一个重 GPU 作业跑（作业层强制） |
| 4 | 新根目录的名字和位置 | 暂定 `/home/arenalabs/am_platform`，下文写作 `$AM_ROOT` |
| 5 | 根分区已用 88% | 删 `xwbc_full`（129 GB，可以从 HF 重新下载）；买或接一块冷存储盘（建议 ≥ 4 TB） |
| 6 | 许可：HiPHI 写明"为商业产品做内部研发也算商用"，HiPHI 和 BONES 都禁止"训练能替代本数据集的生成模型" | 先找法务。平台层面每个片段都带上许可，训练集和模型自动继承"发布受限"标记 |
| 7 | 机器人表面点有两套：工厂的 `surface.npz` 和 am2exp 的 `surface_v2` | 只留工厂那套（引擎和打包器都在用），`surface_v2` 归档 |
| 8 | 大约 180 GB 删除候选（见 §E.12） | 逐条批 |

---

## A. 一个根目录的布局

### A.1 目录树

```
/home/arenalabs/am_platform/                 ($AM_ROOT，放在 /home 分区，剩 345 GB)
├── README.md            人写的：7 个环节一页纸、从哪里看起、常用命令
├── INDEX.md             机器生成的（amp index）：每个目录是什么、谁产出的、多大、最后更新、状态
├── config/
│   ├── paths.toml       所有外部路径和旧路径的唯一出处（GMR、UMR、各个环境、SMPLX 模型、exp_topo……）
│   ├── stores.toml      存储池：home / sys(=/var/tmp/alphamotion) / legacy_tmp(=~/tmp) / hf / cold
│   ├── gauges/data_spec_v2.toml   接触、穿透、打滑、滞回的阈值，所有判官只从这里读
│   ├── budgets.toml     磁盘水位，GPU/CPU/网络并发上限（照抄 RESOURCE_LIMITS）
│   └── service.toml     端口、MCP、worker 数量
├── docs/
│   ├── specs/           DATA_SPEC_v2.md、FORMAT_am_motion_v2.md、FORMAT_umrs_v1.md、TAXONOMY_v1.md
│   ├── contracts/       audit_SCHEMA.md、openapi_v1.yaml、mcp_tools_v1.md、manifest.schema.json
│   ├── ops/             RESOURCE_LIMITS.md、runbook.md（服务起停、开机检查、掉网怎么办）
│   ├── decisions/       lloyd 拍板记录，只追加：YYYY-MM-DD_<题目>.md
│   ├── maps/            这次的 4 张地图和本设计
│   └── reports/         旧技术报告 00–35（来自 "alphamotion video/resource/report"，编号不变）
├── code/                唯一的 git 仓（monorepo）
│   ├── amp/             平台薄层：paths、catalog、formats、jobs、index、cli、service(api+mcp)
│   ├── s1_clean/        下载、转换、gauge 统一、许可
│   ├── s2_surface/      机器人登记、暂存、表面点、对应关系、脚底偏移
│   ├── s3_retarget/     engines/{umr_cpu,umr_gpu}（原样冻结）+ adapters/ + tests/
│   ├── s4_dataset_eval/ 普查、分类、质量汇总、人审队列
│   ├── s5_store/        打包（umr_dataset）、建数据集、清理计划、磁盘报告
│   ├── s6_train/        models/{am2,greenwich_legacy}、watcher（训练中途出片）、viz/live
│   ├── s7_eval/         judges/、render/、golden/、leaderboard/
│   ├── web/             统一网页（底座：motion_audit_web 的前后端）
│   ├── ops/             throttle_guard、systemd 单元、资源守卫
│   └── workflows/       Claude 工作流脚本（diversity_wf.js、robots_wf.js）
├── repos/               有自己 git 历史、不并进 monorepo 的仓库（软链）
│   ├── research -> ~/Desktop/"be water, robot"   研究主仓（exp_topo 冻结在这里）
│   ├── product  -> ~/Desktop/AlphaMotion         产品仓 v0.2（LiveViewer 在这里）
│   ├── datagrid -> ~/Desktop/arena-datagrid      locomotion.v1 标准、jobs 表设计
│   └── ctrl     -> ~/alphamotion_ctrl            WBC / mjlab
├── third_party/<名字>@<版本>/    外部仓库，只读（先软链，后搬）；我们的改动放 patches/<名字>/*.patch
├── envs/                软链 + 锁文件：umr、genmo-gem、ardy、ctrl、ufo
├── data/
│   ├── s1_sources/      来源清单 + 许可证原文
│   ├── s1_raw/<来源>/   原始下载，只读
│   ├── s1_human/<来源>/smplx@<版本>/   清洗后的人体（统一 gauge）
│   ├── s2_robots/<机器人>/   一台一个目录：mjcf、mesh、surface、corr、umr_config、sole_offset、staging
│   │   └── registry.json    机器人登记表（34 台，核心 21 台）
│   ├── s3_retarget/<引擎>@<版本>/<批次>/   重定向输出（中间格式，打包后可回收）
│   ├── s3_labels/<种类>@<版本>/            旧接触标签（cl_v3 等）
│   ├── s4_dataset_eval/<YYYY-MM-DD>_<主题>/  普查、质量汇总、逐台 QA、引擎一致性
│   ├── s4_audit/        人审交接文件 queue/decisions/issues.jsonl（接口约定不变）
│   └── s5_datasets/<名字>@<版本>/   训练分片（UMRS v1）+ 旧训练缓存（只登记、软链）
├── catalog/             catalog.sqlite（唯一的权威记录）+ backups/ + exports/
├── runs/<模型族>/<run_id>/         s6 训练产物：ckpt、曲线、live/（训练中途的视频）
├── evals/<套件>@<版本>/<被评对象>/  s7 评测产物：逐段结果、判决、三联视频
├── media/<YYYY-MM-DD>_<主题>/       给人看的视频和图（报告引用的都放这里）
├── releases/<产品>@<版本>/          发布物
├── services/<服务>/{state,logs}/    服务运行态：amp-service、viewer、datagrid
├── stores/              存储池挂载点（软链）：sys -> /var/tmp/alphamotion，legacy_tmp -> ~/tmp，cold -> 待定
├── cache/               随时可以删
├── scratch/<YYYY-MM-DD>_<任务>/    临时目录，14 天过期
└── archive/<YYYY-MM>_<主题>/       冻结的旧路线，只读，每个目录带一行说明
```

路演、融资、股权、外联名单这些不放进 `$AM_ROOT`，统一放 `~/business/`。

**同一个编号 = 同一个环节**：`code/s3_retarget` 产出 `data/s3_retarget`，`code/s4_dataset_eval` 产出 `data/s4_dataset_eval`。lloyd 看到 `s3` 就知道是第 3 环节（重定向）。

### A.2 大数据住在哪、怎么被引用

| 存储池 | 物理位置 | 放什么 | 怎么引用 |
|---|---|---|---|
| `home` | `/home/arenalabs/am_platform/...`（/home 分区） | 所有新产出；以后从 `~/umr_factory_data` 改名搬来的数据 | catalog `locations` 表：`store=home, relpath=data/...` |
| `sys` | `/var/tmp/alphamotion`（根分区，已用 88%） | 旧训练缓存 `topo_exp`（160 GB）、`xwbc_lafan`、`config_harvest`、`robot_models` | `stores/sys` 软链，加上 `locations.store=sys`；**平台不再往这里写新东西** |
| `legacy_tmp` | `~/tmp` | 有代码在读的旧件：`cl_v3`、`sole_offsets3.json`、`dialect`、`refiner_suite`…… | 软链，加上 catalog 的 `legacy_paths` 反查表 |
| `hf` | `~/.cache/huggingface/hub` | pretraining_foundation 下载的 BONES、UniMate 等 | `locations.store=hf` |
| `cold` | 待定（建议外接 ≥ 4 TB 盘或 NAS） | `archive/`、可以重新生成的东西、旧缓存 | `locations.store=cold`；不在线时读取报"离线"，不能悄悄当成不存在 |

**解析规则**：
- 代码只通过 `amp.paths.resolve("<catalog id>")` 或 `amp.paths.get("<paths.toml 键名>")` 拿路径，不写死。
- 跨环境变量 `AM_ROOT` 一个就够。
- 旧代码先不改，靠旧位置留的反向软链继续跑。
- `amp paths lint` 在 git 钩子里扫描写死路径，只报警、不拦。地图里已经记了这些写死的地方：工厂 19 处、arena 10 处、Studio 7 处、`umrf/paths.py`、`gpu_convert` 里的 `~/Downloads/SMPLX_NEUTRAL.npz`，等等。

### A.3 命名和版本约定

| 东西 | 规则 | 例子 |
|---|---|---|
| 片段 id | `<来源>/<原生相对路径去扩展名>`，镜像加 `#mirror` | `omnicontact/box/case1_carry/20260330000194_1_1775786896` |
| 机器人 id | 登记表的键 | `unitree_g1_29dof` |
| 重定向 id | `<片段>::<机器人>::<引擎>@<git 短哈希>` | `grail/curb/curb_000__000::fourier_gr3::umr_gpu@3f2a9c1` |
| 数据版本 | `<名字>@v<N>` | `smplx@v3`、`corr@v1`、`am_pretrain@v1` |
| 引擎、判官版本 | 引擎用 git 短哈希，判官用 `@v<N>` | `umr_gpu@3f2a9c1`、`geometry_judge@v1` |
| 批次 | `<YYYYMMDD-HHMM>_<来源>_<条数>` | `20261009-0900_grail_1200` |
| run | `<YYYYMMDD>_<模型族>_<短名>` | `20261012_am2_B_S_seed0` |
| 黄金评测集 | `G-<主题>@v<N>` | `G-loco@v1` |
| 媒体 | `media/<YYYY-MM-DD>_<主题>/<对象>__<种类>.mp4` | `.../C1__triptych.mp4` |

硬规则：
- 路径里不能有空格、引号、逗号（`"be water, robot"` 就是教训），`amp paths lint` 会检查。
- 永不覆盖：新版本一律新建目录。目录的 `MANIFEST.json` 里写了 `frozen: true` 以后，里面就不再改。
- `data/`、`runs/`、`evals/`、`media/`、`releases/` 下的每个产出目录都必须带 `MANIFEST.json`，格式见 `docs/contracts/manifest.schema.json`。内容包括：
  - `produced_by`：作业 id 和命令；
  - `code_commit`；
  - `inputs`：catalog id 加 sha；
  - `outputs`；
  - `retention`：保留级别；
  - `catalog_ids`；
  - `created_at`。

### A.4 README 索引（回答"每个文件夹是干嘛的、产出了什么"）

- `README.md`（人写，不超过 2 页）：
  - 7 个环节各一句话；
  - 每个环节"代码在哪、产出在哪、用哪个命令看状态"；
  - 服务地址；
  - 怎么找 lloyd 拍板记录。
- `INDEX.md`（`amp index` 每晚生成，也可以手动跑），由 catalog 加各目录的 `MANIFEST.json` 拼成：

```
| 目录 | 环节 | 是什么 | 产出了什么 | 谁产出的 | 大小 | 最后更新 | 状态 |
| data/s1_human/hiphi/smplx@v2 | s1 清洗 | HiPHI 转 SMPL-X（GPU 转换器，30 fps） | 3,795 段 / 28.7 h | job 0193（gpu_convert@a1b2c3d） | 4.4 GB | 10-07 | 生产 |
| runs/am2/20261004_A_S_seed0 | s6 训练 | AM2 规模对照 A/S | ckpt×4、train.jsonl、result.json | 旧 orchestration（已登记） | 1.5 GB | 10-04 | 归档 |
```

- 每个顶层目录都有一个 3 行的 `README.md`，写明：这里放什么、谁写进来、能不能删。
- `amp index --lint`：发现没有 MANIFEST 的目录、断掉的软链、catalog 里登记了但盘上不存在的东西，就列出来。

---

## B. 7 个环节

公共约定：
- CLI 统一叫 `amp <环节> <动作>`。
- MCP 工具和 HTTP 接口都调用 `amp/service/core.py` 里的同一个函数。
- 有写操作的接口一律走"先出计划（plan，带估时、估磁盘、估 GPU 小时）→ 再提交（submit，带幂等键）"两步。这是今天 `agent_tools/jobs.py` 已经实现的模式，直接沿用。

### B.1 环节 1：数据清洗（下载 → 转 SMPL-X → 统一 gauge → 许可）

**现成的工具（包住，不重写）**：
- 下载：
  - HiPHI：`scripts_streamc/hiphi_staged_dl.py`
  - GRAIL：`scripts_grail/grail_dl_loop.py`
  - OmniContact：`scripts/dl_omni.py`
  - BONES、UniMate 等：`pretraining_foundation/acquire.py`
- 转换：
  - HiPHI：`gpu_convert/convert.py`（生产版，和 CPU 版对过结果）
  - OmniContact：`scripts/convert_omni.py`（CPU 版）
  - GRAIL：`scripts_grail/convert_grail.py`（v3）+ `floorcal2.py` + `grail_collision.py`
- gauge 检查：`exp_topo/gauge_check.py`、xwbc 时间轴修复、人体方言 Q
- 许可原文和逐源清单：`/var/tmp/.../sources/sources_inventory.json`
- 逐段许可的样板：GRAIL 的 `manifest.jsonl`，22,312 行，带 license、floor 策略、src_valid

**新增的薄层**：
- `amp.s1.sources`：登记来源和许可（licenses 表）。没有许可的片段，状态永远到不了 `eligible`。
- `amp.s1.importers.{hiphi,grail,omni,lafan,bones,xwbc}`：读现有的清单和 `qa.json`，往 catalog 写 `clips` 和 `human_versions`。只读、可以重复跑。
- `amp.formats.human.load(human_version_id)`：一个函数读 4 种 SMPL-X 存法，统一返回 Z 轴朝上、米制、地面为 0（原地面值另存）的结果。4 种存法是：HiPHI 的 `motion_actor_smplx.npz`、Omni 的分字段 `.npy`、GRAIL 的 `recon/*.pkl`、LAFAN 放在 UMR 的 `sample_data` 里。打包器 `umr_dataset/sources.py` 找 HiPHI 找错路径的问题，改成调这个函数就修好了。
- gauge 记录：每个人体版本都存一份 `gauge`，包括朝上的轴、单位、地面策略、地面偏移（mm）、帧率、降采样步长、是否镜像。

**输入 / 输出**：
- 输入：`data/s1_raw/<来源>/`。
- 输出：`data/s1_human/<来源>/smplx@<版本>/`（现有目录登记并软链，不重新转换），以及 catalog 的 `sources`、`licenses`、`clips`、`human_versions`。

**CLI**：
- `amp s1 source list|show <id>`
- `amp s1 import --source grail`
- `amp s1 gauge-check --source hiphi [--sample 200]`
- `amp s1 license report [--use commercial]`
- `amp s1 status`：下载和转换进度，加上 throttle 状态

**MCP**：
- `source_list`、`source_get`
- `clip_search(source, class, status, has_object, text, limit)`、`clip_get`
- `ingest_status`
- `license_check(clip_ids|dataset_id, intended_use)`

**HTTP**：
- `GET /api/v1/sources`、`/sources/{id}`、`/licenses`
- `GET /api/v1/clips?source=&class=&status=&has_object=&q=`、`/clips/{id}`、`/clips/{id}/human?start=&count=`（二进制，沿用审计网页的 wire 协议）
- `GET /api/v1/ingest/status`

**测试**：
- 每个来源一段固定样本，读出来后检查：形状、帧率、首帧最低点离地在 ±2 mm 内、Z 轴朝上。
- 许可不能为空的约束测试。
- 时间轴健全性：带"走路"标签的片段，根速度中位数在 0.5–2 m/s。xwbc 的 KIT 1.2 倍速 bug 就是靠这种检查才能发现。
- 导入器跑两遍，结果一致。

### B.2 环节 2：表面点采样（机器人登记 + 4096 个外表面点 + 区域标签）

**现成的工具**：
- 登记表：`robots.json`（34 台，核心 21 台）。
- 暂存和表面点：`umrf/robots.py`、`umrf/surface.py`、`scripts/stage_robots.py`、`build_surfaces.py`。产出每台的 `surface.npz`：4096 点，带法线、geom、body、区域。
- 对应关系：`build_correspondence.py`、`build_shared_corr.py`。
- 打包器派生的 FK 表：`umr_dataset/data/robots/<r>/kin.npz`。
- 人体侧：UMR 的 4096 个 SMPL-X 槽位、`smplx_parts_segm.pkl`。

**决定**：
- 每台机器人一个权威目录 `data/s2_robots/<r>/`，就是现在工厂的 `robots/<r>/`。它已经是展平后的 MJCF 加网格，不再依赖桌面仓库里的 GMR。
- 登记表 `registry.json` 从 `/var/tmp` 拷过来，`source_mjcf` 字段记录原始来源。
- 每类资产都在 `robot_assets` 表登记，带版本和 sha：`mjcf`、`surface`、`corr`、`umr_config`、`sole_offset`、`kin_tables`。
- **对应关系 corr 有两份，字节不一样**：工厂的 `corr/` 和 GPU 引擎的 `umr_gpu/corr`。先不合并，两份分别登记成 `corr@factory_20261004` 和 `corr@umr_gpu_20261004`。生产用 GPU 那份，因为一致性测试是在它上面做的。另开一个工作包查清楚差在哪（WP-S2a），查清前不动。
- 区域词表冻结成 `regions@v1`（现有 13 个区域），并写清人体区域和机器人区域怎么对应。
- 脚底偏移：现在的 `sole_offsets3.json` 有 8 台偏大 4.7–10 cm，H2 没有。从 MJCF 碰撞几何重新量一份，作为 `sole_offset@v4`，只给旧模型线用。UMR 线以表面点、地面 = 0 为准，不依赖这张表。

**CLI**：
- `amp s2 robot list|show <r>`
- `amp s2 stage <r>`：调 `stage_robots.py`
- `amp s2 surface <r>`：调 `build_surfaces.py`
- `amp s2 corr <r>`
- `amp s2 check <r>`

**MCP**：`robot_list`、`robot_get`（自由度、身高、区域、各资产版本、哪些环节已就绪）、`robot_surface_preview`（返回 T 姿态和表面点 png 的 URL）。

**HTTP**：
- `GET /api/v1/robots`、`/robots/{r}`、`/robots/{r}/assets`
- `GET /robots/{r}/surface`（二进制点云，给网页 3D 用）
- `GET /robots/{r}/tpose.png`

**测试**：
- 恰好 4096 个点，全在外表面（射线可见性），法线是单位长度。
- 必需区域（左右脚、左右手）都有点。
- 最远点采样的点间距统计在阈值内。
- MJCF 的 sha 和登记表一致。
- T 姿态联系表能渲出来。
- 脚底偏移和碰撞几何最低点相差 ≤ 1 mm。

### B.3 环节 3：重定向到机器人（UMR CPU 参考 + GPU 引擎 + 严格后处理）

**现成的引擎（冻结，只包住）**：
- `umr_gpu`：生产引擎，和 CPU 版对过结果，关节误差中位数约 1e-6 到 1e-5 rad。包含严格后处理 `strict.py`（接触锁定、偏航锁定、抬脚规则），导出 `gpu_npz_v2`。还带一个独立的 CPU 判官 `eval/judge.py`（用 MuJoCo 的 mj_geomDistance 加稠密点）。
- `umr_factory_code`：CPU 参考实现和量尺。**已知 bug**：`umrf/objects.py` 把 8 顶点的箱子当成外接球。另外 CPU 版保留了 UMR 自己的打滑（`LATCH_PIN=0`），达不到规格 v2 的"不滑"。所以它只当参考，不用于量产。
- 已有的旧重定向：GMR 生产数据（Native 对照臂）、xwbc（18 台）。只登记，当对照臂用。

**新增的薄层**：
- `s3_retarget/adapters/umr_gpu_job.py`：
  1. `plan`：根据片段查询 × 机器人列表，算出帧数、预计 GPU 时间（按 10-07 实测 2,707 机器人帧/秒，4 台、B=256）、预计磁盘。
  2. `run`：生成引擎要的输入，调用 `umr_gpu/scripts/run_factory.py`（不改它）。
  3. `post`：给每个输出写一个旁挂的 `.meta.json`，内容包括 joint_names（从 MJCF 取）、来源信息（引擎哈希、人体版本、corr 版本、规格版本）、sha。
  4. `register`：写 catalog 的 `retargets` 和 `qa_metrics`（引擎自己的 `qc_json`）。
  5. `enqueue_audit`：按风险和随机两种方式往人审队列放条目。
- `amp.formats.motion.load(retarget_id)`：认 `factory_npz_v1` 和 `gpu_npz_v2`，返回统一对象：
  - 米制，地面 = 0；
  - qpos 和 joint_names；
  - 稀疏点级接触（CSR）；
  - 区域级接触、打滑、穿透、离地距离；
  - 人体区域标签；
  - 物体位姿；
  - flags；
  - 来源信息。

  字段规格写在 `docs/specs/FORMAT_am_motion_v2.md`。v1 的点级接触（packbits）在读取时转成 CSR。
- 阈值统一放在 `config/gauges/data_spec_v2.toml`：
  - 实接触 ≤ 1.0 mm；
  - 穿透 > 0.5 mm 算违规；
  - 打滑 < 1 cm/s；
  - 接触段最短 3 帧；
  - 滞回：进入 1 mm、离开 2 mm；
  - 人体悬空 > 2 cm 时，机器人要离开 > 5 mm。

  UMR 原生标签用的是 1 cm / 2 cm / 3 cm/s，这一套标为 `legacy_umr`，单独显示，不冒充 v2。
- 箱子问题：
  - CPU 工厂已经产出的所有搬箱片段，在 catalog 里标成 `quarantined`（隔离）。
  - 量产前先跑 WP-S3b：用合成的箱子片段，检查 GPU 引擎 `geometry.py` 的箱子 SDF。验收标准：接触段手离箱面 ≤ 1 mm、物体实接触率 > 0。
  - 这条没过之前，OmniContact 的搬箱片段不进量产。

**输入 / 输出**：
- 输入：`human_versions` + `s2_robots/<r>` 的资产。
- 输出：`data/s3_retarget/umr_gpu@<哈希>/<批次>/*.npz` 加 `.meta.json`，再加 `MANIFEST.json`。

**CLI**：
- `amp s3 plan --clips "source=grail AND class=locomotion" --robots core21 --engine umr_gpu` → 打出计划 id 和估算
- `amp s3 submit <plan_id>`
- `amp s3 status [batch]`
- `amp s3 import-legacy`：登记 CPU 的 `out/`（11 个）和 GPU 的 export 样本（12 个）
- `amp s3 show <retarget_id>`

**MCP**：
- 已有的工具保留原名，兼容今天的 MCP：`motion_catalog`、`motion_inspect`、`motion_frame_metrics`、`motion_visualize`
- 新增：
  - `retarget_plan(clip_query, robots, engine)`
  - `retarget_submit(plan_id, idempotency_key)`：GPU 作业。每个引擎版本的第一批要人工批准。
  - `retarget_get(clip, robot)`：返回指标、路径、审计状态。

**HTTP**：
- `POST /api/v1/retarget/plans`
- `GET /api/v1/retargets?clip=&robot=&engine=&status=`、`/retargets/{id}`、`/retargets/{id}/frames?start=&count=`、`/retargets/{id}/qa`

**测试**：
- 一致性回归：在冻结的 4 个用例上跑 `umr_gpu/eval/run_parity.py`，关节 p50 必须 ≤ 1e-5 rad。要用 GPU，所以每次升级引擎版本时才跑。
- 读取层：v1、v2 读出来字段齐全，单位是米。v1 的 packbits 转成 CSR 后，和原文件按位对比一致。
- 规格符合：在黄金重定向集上，用独立判官测"不穿、不滑、不闪、实接触"。
- 合成箱子测试（WP-S3b）。
- 适配器：只读引擎目录，输出全部落在批次目录里，用 `inotify` 或比较前后快照验证。

### B.4 环节 4：整库评估预训练数据（多样性、质量、人审）

**现成的工具**：
- `diversity/`：各来源的 `summary.json`、逐段标签 csv、charts、今天的 `REPORT.md`。结论：已知 571.4 h，手上 126.4 h，地面类、手势类、摔倒起身类基本是 0。
- `action_labels.py`：12 族正则分类，AMASS 有 77% 落在 "other"。
- BONES 分级表。
- 5 台机器人各自的 QA 流水线（`robots/<r>/pipeline`），格式各不相同。
- 人审约定 `audit/SCHEMA.md`，和人审网页 `motion_audit_web`（带 Wilson 置信区间、哈希过期机制）。
- `wbc_ready_audit`（600 段语料外考卷）。

**设计成三层**：
1. **覆盖度**：
   - 分类表 `TAXONOMY_v1` = `REPORT.md` 里的 9 个大类，每段只归一类，小时数可以相加；12 族正则作为副标签。
   - `labels` 表记录分类方法（元数据文字、正则、模型、人工）和置信度。
   - 自动生成"稀缺清单"，反过来决定下载和转换的优先级。比如 HiPHI 0032 包，按"每 GB 带来多少稀缺动作"来排。
2. **质量**：
   - 把逐段 `qa_metrics` 按来源 × 机器人 × 大类汇总，算出规格 v2 的通过率。
   - 5 台各写一份的 `measure.py` 退役，统一用 s7 的几何判官加每台机器人一份配置跑，报告格式统一成 `qa_report_v1.json`。
3. **人审**：
   - 队列构建器按两种方式抽条目：按风险（自动质检越差越靠前）和随机（无偏）。
   - 定时把 `decisions.jsonl` 和 `issues.jsonl` 读进 catalog。哈希不一致的判决标为过期。
   - `needs_fix` 的条目回到重定向队列，带上原因。
   - 系统性问题（某台机器人、某个来源、某个引擎）开成 issue。
   - **批次放行门**：随机抽样合格率的 Wilson 下界 ≥ 阈值（默认 0.90，lloyd 定），整批才能进入打包。

**产出**：`data/s4_dataset_eval/<日期>_snapshot/{report.md, numbers.json, charts/}`，加 catalog 的 `eval_snapshots`。

**CLI**：
- `amp s4 census`
- `amp s4 label --source bones --method metadata`
- `amp s4 quality [--by source,robot,class]`
- `amp s4 audit enqueue --risk 50 --random 50 --batch <id>`
- `amp s4 audit ingest`
- `amp s4 snapshot`

**MCP**：
- `diversity_summary(group_by)`、`coverage_gaps()`
- `quality_summary(source, robot, class)`
- `issues_list(status)`
- `audit_queue_status`、`audit_enqueue(batch, n_risk, n_random)`
- `agent_note`（已有，代理留观察记录；人工判决只有网页能写）

**HTTP**：
- `GET /api/v1/stats/diversity?group_by=`、`/stats/coverage-gaps`、`/stats/quality?group_by=`
- `GET /api/v1/audit/queue`
- `POST /api/v1/audit/decisions`、`POST /api/v1/audit/issues`：网页专用，后端照约定追加写 jsonl
- `GET /api/v1/issues`、`/snapshots/dataset-eval`

**测试**：
- 小时数守恒：按各种维度汇总后的总和 = catalog 总时长。
- 每段恰好一个主类。
- 哈希不一致的判决被标为过期。
- 置信区间计算的单元测试（沿用审计网页的现有测试）。
- 黄金评测集的片段不会出现在任何训练集的抽样里。

### B.5 环节 5：存储（catalog、产物、分片、预算）

设计细节见 §C。这里只列工具。

- `amp s5 catalog init|migrate|backup|check`
- `amp s5 dataset plan --name am_pretrain --select "<查询>" --license research_only`：算出段数、小时、机器人帧数、磁盘。
- `amp s5 dataset build <plan_id>`：调 `umr_dataset` 的 convert，原样复用。读取改为走 `amp.formats.motion.load`，这里要给 `umr_dataset/sources.py` 加一个很小的适配。这是唯一需要改动的下游代码，在 monorepo 的副本上改，原仓库打了 tag 留底。
- `amp s5 verify <dataset>`：调 `umr_dataset/verify.py` 校验分片哈希。
- `amp s5 disk`：各存储池的水位、按目录排的大户、按保留级别分的可回收量。
- `amp s5 gc plan`：只出计划；执行要 lloyd 在本机敲 `amp s5 gc apply <plan_id> --i-am-lloyd`，MCP 不开放这个操作。
- `amp index`、`amp paths resolve|lint`

**MCP**：`catalog_query`（白名单过滤条件，不收原始 SQL）、`path_resolve`、`dataset_list`、`dataset_get`、`dataset_plan`、`disk_report`、`gc_plan`。

**HTTP**：
- `GET /api/v1/datasets`、`/datasets/{id}`
- `POST /api/v1/datasets/plans`
- `GET /api/v1/disk`、`/gc/plan`、`/index`、`/paths/resolve?id=`

**测试**：
- 分片校验和。
- 加载器按帧精确配对：沿用 `umr_dataset` 的现有测试。
- 抽查 catalog 一致性：`locations` 里每条都存在，抽样核对 sha。
- 代码里不许引用 `scratch/` 或 `cache/` 下的路径（lint）。

### B.6 环节 6：训练 + 实时看模型生成的东西

设计细节见 §D。

- 训练入口（包住）：
  - AM2：`train_scaling.py`、`train_arm.py` + `arms.json`，配方 `megatron_scale_pilot/code` 冻结。
  - Greenwich：`exp_topo/train_dual.py`，只用于复现 v0.2，冻结。
- 作业类型 `train`：必须带批准文件，里面写清预计 GPU 小时、kWh、电费。默认不开 RL。0903 那次电费 700 刀以后定的规矩。
- watcher：订阅新 checkpoint，用 CPU 在黄金窗口上推理 → 判官打分 → 出 12 帧网格和短三联视频 → 推到实时查看器，写进 `runs/<id>/live/<step>/`。lloyd 问"训练怎么样了"时，直接给最新一段视频。
- 实时查看器：在产品仓 `alphamotion.viz.live.LiveViewer`（viser 常驻、就地换内容）外面加一个读 am_motion 格式的场景适配器。

**CLI**：
- `amp s6 run list|show <id>`
- `amp s6 train plan <arm>` / `amp s6 train submit <plan_id>`（需要批准）
- `amp s6 watch <run_id>`
- `amp s6 viewer serve|load --model <ckpt> --clip <id> --robots g1,h2`

**MCP**：`model_list`、`run_list`、`run_get`、`run_curves`、`run_live_latest`（返回最新中途视频的 URL）、`train_plan`、`train_submit`（需要批准）、`viewer_load`、`viewer_status`。

**HTTP**：
- `GET /api/v1/models`、`/runs`、`/runs/{id}`、`/runs/{id}/curves`、`/runs/{id}/live`
- `POST /api/v1/viewer/load`、`GET /viewer/status`

**测试**：
- watcher 推理只用 CPU，测试里设 `CUDA_VISIBLE_DEVICES=""` 断言。
- 查看器加载一个 am_motion 文件 → 截图 → 帧号和时间轴一致。
- 训练作业没有批准文件就拒绝提交。

### B.7 环节 7：评估模型重定向或生成出来的动作

设计细节见 §D.3。

- 判官（独立）：几何判官（J1 + umr_gpu judge 合并）、locomotion.v1 标准、J2 抓握闭合、编解码回归 v2_judge、无真值物理判官、姿态保真、WBC 可跟踪率（mjlab，需要批准）。
- 统计层：冻结病例 + 按窗口的配对检验 + `decide()`。
- 渲染：三联（人体源 | 对照臂 | 本臂）和全程 12 帧网格。
- 排行榜。

**CLI**：
- `amp s7 eval plan --subject ckpt:<id> --suite geometry@v1,locomotion@v1 --golden G-loco@v1`
- `amp s7 eval submit`
- `amp s7 compare <evalA> <evalB>`
- `amp s7 render triptych --source <clip> --control <retarget_id> --test <rollout_id>`
- `amp s7 leaderboard --suite locomotion@v1 [--robot g1]`

**MCP**：`golden_list`、`eval_plan`、`eval_submit`、`eval_get`、`eval_items`、`eval_compare`、`render_triptych`、`leaderboard_get`、`video_inspect`（已有）、`wbc_track_eval`（需要批准）。

**HTTP**：
- `GET /api/v1/golden`、`/golden/{id}`
- `POST /api/v1/evals/plans`、`GET /evals`、`/evals/{id}`、`/evals/{id}/items`
- `POST /api/v1/compare`
- `GET /api/v1/leaderboard?suite=&robot=`
- `POST /api/v1/renders/plans`
- `GET /api/v1/media/{id}`（支持 Range，手机能拖进度条）、`/media?subject=`
- `POST /api/v1/reviews`：人看过视频以后的判决，必须写审阅人

**测试**：
- 判官不能 import 任何训练代码：测试扫 import 图（J1 已经有这个测试，推广到所有判官）。
- 在"数据地板"上（参考重定向本身）各判官的结果可以复现。
- 三联视频的时间戳逐帧检查（沿用 rollout_tools 的 encode 检查）。
- 12 帧网格必须包含首帧和末帧，并均匀覆盖全程。

### B.8 一个服务：MCP 和 HTTP 怎么落地

```
            Claude / 代理                      lloyd / Codex 做的网页
                │ stdio MCP                         │ HTTP (127.0.0.1:8765 → ssh 转发)
        amp/service/mcp.py                   amp/service/api.py (FastAPI)
                └──────────────┬────────────────────┘
                     amp/service/core.py   ← 唯一的实现
          ┌──────────┬─────────┼───────────┬───────────────┐
      catalog.sqlite  formats  jobs(租约)   paths/stores   旧审计接口(jsonl)
                                   │
          worker: cpu ×≤6 (taskset 0-15, nice 10, OMP=2) │ gpu ×1（重任务）│ net（全局 ≤16 连接）
```

- **底座**：`~/motion_audit_web`。后端有 `backend/app.py`，前端是 React、Vite、Three.js。今天新建的 `agent_tools/server.py` 已经有 `toolchain_status`、`motion_*`、`evidence_*`、`video_inspect`、`workflow_list/plan/submit`、`job_*`、`agent_note` 这些工具。
  - 做法：把它整体并进 `code/web` 和 `code/amp/service`，工具名全部保留，再按 §B.1–B.7 补上新工具。
  - 注意：此刻还有另一个子任务在写 `agent_tools/`，要等它收尾以后再接手（见 §F 的 WP-08）。
- **端口**：只监听 `127.0.0.1`，统一用 8765。现在对外网开着（0.0.0.0）的 2334、8891、8899 三个旧服务，内容并进网页以后停掉。
- **作业层**：
  - 表结构照搬 arena-datagrid `store.py` 里的 `jobs` 和 `job_events`（带租约、尝试次数、事件）。
  - 行为照搬 am2exp ops：`lease.py`（GPU 锁）、`guard.py`（磁盘守卫）、`receipts.py`（回执）、`approvals/<stage>.ok`（人工批准）。
  - 资源分 4 类：

| 作业类型 | 资源类 | 要不要批准 |
|---|---|---|
| ingest_download | net（全局 ≤ 16 连接） | 下载清单批准过一次就行 |
| convert_gpu、retarget_gpu | gpu_heavy（同时只跑 1 个） | 每个引擎版本的第一批要批准 |
| judge、render、pack、census | cpu（≤ 6 个重进程） | 不用 |
| train、wbc_eval | gpu_heavy | 每次都要（附估时和电费） |
| gc_apply | — | 只能 lloyd 在本机执行 |

  - 掉网和内存保护：worker 每 20 秒读一次 `logs/throttle.log`。看到 `net=down` 就暂停领新作业；`free` 可用内存 < 15 GB 也暂停。
- **安全边界**：
  - MCP 不开放任何删除操作，也不能执行任意 shell。
  - 写操作一律要 `actor` 字段。
  - "加冕"（Greenwich 头衔）只能由人在网页上做，`reviews` 表约束 `reviewer_kind='human'`，这条也照搬 arena。
- **网页（交给 Codex）**：先冻结 `docs/contracts/openapi_v1.yaml`。页面包括：
  - 总览：7 个环节的流水线状态、运行中的作业、磁盘水位、掉网状态；
  - 来源和片段浏览；
  - 机器人登记；
  - 重定向浏览和 3D 审片（已有）；
  - 人审队列（已有）；
  - 数据评估看板；
  - 数据集和分片；
  - 作业；
  - 训练 run（曲线和中途视频）；
  - 评测和排行榜；
  - 媒体画廊。

---

## C. 存储设计

### C.1 catalog（唯一的权威记录）

- **为什么用 SQLite**：
  - 只有一台机器，最多几百万行（现在：GRAIL 2.2 万、foundation 索引 24 万；以后 571 h × 21 台约 30–40 万条重定向）。
  - WAL 模式支持多个进程同时读，单写锁就够。
  - arena-datagrid 已经验证过这套用法。
- **位置**：`$AM_ROOT/catalog/catalog.sqlite`。
  - 每天备份到 `catalog/backups/catalog_<日期>.sqlite`，保留 30 份。
  - 小表每晚导出成 `catalog/exports/*.jsonl`，可以进 git 看变化。
- 只能通过 `amp.catalog` 写，每次写都追加一条 `events`。

```sql
-- 来源与许可
CREATE TABLE licenses (id TEXT PRIMARY KEY, name TEXT, commercial INTEGER, internal_rnd_ok INTEGER,
  redistribution INTEGER, generator_restriction TEXT, text_location TEXT, sha256 TEXT, notes TEXT);
CREATE TABLE sources (id TEXT PRIMARY KEY, kind TEXT,  -- mocap|video_recon|synthetic|legacy_retarget
  fps_native REAL, license_id TEXT REFERENCES licenses, upstream TEXT, revision TEXT, hours_known REAL, payload TEXT);
-- 人体
CREATE TABLE clips (id TEXT PRIMARY KEY, source_id TEXT, native_path TEXT, mirror INTEGER DEFAULT 0,
  duration_s REAL, fps REAL, n_frames INTEGER, subject_id TEXT, has_object INTEGER, object_ids TEXT,
  license_id TEXT,              -- 逐段覆盖（GRAIL 物体许可不同）
  split_group TEXT,             -- 同一演员/场次同组，防泄漏
  status TEXT CHECK (status IN ('registered','converted','eligible','quarantined','excluded')),
  payload TEXT);
CREATE TABLE human_versions (id TEXT PRIMARY KEY, clip_id TEXT, format TEXT, -- hiphi_npz|omni_npy|grail_pkl|lafan_sample|bones|xwbc22
  fps REAL, stride INTEGER, gauge TEXT,  -- {up,units,floor_policy,floor_dz_mm,...}
  converter TEXT, converter_rev TEXT, qa TEXT, location_id TEXT, sha256 TEXT, status TEXT);
CREATE TABLE labels (clip_id TEXT, taxonomy TEXT, primary_class TEXT, tags TEXT, method TEXT, confidence REAL,
  PRIMARY KEY (clip_id, taxonomy));
-- 机器人
CREATE TABLE robots (id TEXT PRIMARY KEY, vendor TEXT, family TEXT, dof INTEGER, height_m REAL,
  tier TEXT, mjcf_sha256 TEXT, license TEXT, status TEXT, payload TEXT);
CREATE TABLE robot_assets (id TEXT PRIMARY KEY, robot_id TEXT, kind TEXT, -- mjcf|surface|corr|umr_config|sole_offset|kin_tables|staging
  version TEXT, location_id TEXT, sha256 TEXT, built_by_job TEXT, active INTEGER);
-- 重定向与质检
CREATE TABLE retargets (id TEXT PRIMARY KEY, clip_id TEXT, human_version_id TEXT, robot_id TEXT,
  engine TEXT, engine_rev TEXT, corr_asset_id TEXT, surface_asset_id TEXT,
  format TEXT,                  -- factory_npz_v1|gpu_npz_v2|model_rollout_v2
  location_id TEXT, sha256 TEXT, n_frames INTEGER, fps REAL, spec_version TEXT, batch_id TEXT, job_id TEXT,
  status TEXT CHECK (status IN ('produced','judged','in_audit','accepted','rejected','needs_fix','quarantined','packed','purged')));
CREATE TABLE qa_metrics (retarget_id TEXT, judge TEXT, judge_version TEXT, metrics TEXT, passed INTEGER,
  flagged_frames INTEGER, excluded_frames INTEGER, PRIMARY KEY (retarget_id, judge, judge_version));
CREATE TABLE audit_decisions (item_id TEXT, npz_sha256 TEXT, verdict TEXT, frame_ranges TEXT, reasons TEXT,
  note TEXT, reviewer TEXT, ts TEXT, stale INTEGER, src_line INTEGER, PRIMARY KEY (item_id, ts));
CREATE TABLE issues (id TEXT PRIMARY KEY, scope TEXT, target TEXT, reasons TEXT, note TEXT, examples TEXT,
  status TEXT, reviewer TEXT, ts TEXT);
CREATE TABLE batches (id TEXT PRIMARY KEY, stage TEXT, engine_rev TEXT, n_items INTEGER, random_pass_lb REAL,
  release_gate TEXT, released INTEGER, payload TEXT);
-- 数据集与分片
CREATE TABLE datasets (id TEXT PRIMARY KEY, kind TEXT,   -- umrs_v1|legacy_cache
  selection TEXT, license_policy TEXT, location_id TEXT, n_clips INTEGER, hours REAL, robot_frames INTEGER,
  built_by_job TEXT, frozen INTEGER, release_restrictions TEXT);
CREATE TABLE dataset_members (dataset_id TEXT, retarget_id TEXT, clip_id TEXT, rows TEXT,
  PRIMARY KEY (dataset_id, retarget_id));
CREATE TABLE shards (id TEXT PRIMARY KEY, dataset_id TEXT, body TEXT, location_id TEXT, bytes INTEGER, sha256 TEXT);
-- 模型侧
CREATE TABLE runs (id TEXT PRIMARY KEY, family TEXT, code_commit TEXT, config TEXT, dataset_ids TEXT,
  status TEXT, gpu_hours REAL, kwh_est REAL, cost_est REAL, approval_ref TEXT, parent_run TEXT, location_id TEXT);
CREATE TABLE checkpoints (id TEXT PRIMARY KEY, run_id TEXT, step INTEGER, sha256 TEXT, location_id TEXT,
  tags TEXT,          -- best|last|milestone|released|lineage
  keep INTEGER);
CREATE TABLE models (id TEXT PRIMARY KEY, family TEXT, checkpoint_id TEXT,
  title_status TEXT CHECK (title_status IN ('candidate','crowned','retired')), crowned_by TEXT, release_restrictions TEXT);
CREATE TABLE golden_sets (id TEXT PRIMARY KEY, description TEXT, items TEXT, sha256 TEXT, frozen_at TEXT);
CREATE TABLE evals (id TEXT PRIMARY KEY, subject_kind TEXT, subject_id TEXT, suite TEXT, suite_version TEXT,
  golden_set_id TEXT, judge_versions TEXT, summary TEXT, verdict TEXT, location_id TEXT, job_id TEXT);
CREATE TABLE eval_items (eval_id TEXT, item TEXT, metrics TEXT, passed INTEGER, not_evaluated_reason TEXT,
  PRIMARY KEY (eval_id, item));
CREATE TABLE reviews (id TEXT PRIMARY KEY, subject_kind TEXT, subject_id TEXT, reviewer TEXT,
  reviewer_kind TEXT CHECK (reviewer_kind = 'human'), verdict TEXT, media_id TEXT, note TEXT, ts TEXT);
CREATE TABLE media (id TEXT PRIMARY KEY, kind TEXT,  -- triptych|grid12|showcase|live|tpose
  subject TEXT, location_id TEXT, sha256 TEXT, duration_s REAL, job_id TEXT);
-- 作业（照搬 arena store.py 的 jobs / job_events，加两列）
CREATE TABLE jobs (id TEXT PRIMARY KEY, request_hash TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
  resource_class TEXT, approval_ref TEXT, owner_pid INTEGER, attempt INTEGER DEFAULT 0, payload TEXT NOT NULL,
  priority INTEGER DEFAULT 0, parent_job TEXT, claimed_by TEXT, lease_until TEXT, stage TEXT, error TEXT,
  result TEXT, started_at TEXT, finished_at TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE job_events (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts TEXT, stage TEXT, payload TEXT);
-- 存储
CREATE TABLE locations (id TEXT PRIMARY KEY, store TEXT, relpath TEXT, kind TEXT, bytes INTEGER, n_files INTEGER,
  sha256 TEXT, retention TEXT CHECK (retention IN ('keep','keep_n','regen','scratch','cache')),
  verified_at TEXT, present INTEGER);
CREATE TABLE legacy_paths (old_path TEXT PRIMARY KEY, location_id TEXT, referenced_by TEXT, note TEXT);
CREATE TABLE disk_snapshots (ts TEXT, store TEXT, used_bytes INTEGER, free_bytes INTEGER, top TEXT);
CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, kind TEXT, payload TEXT);
```

**导入现有台账，不重建**：
- GRAIL `manifest.jsonl` → `clips`、`human_versions`、逐段 license。
- HiPHI：`convert_log.jsonl`、`parts_status.json`、`qa.json`、`metadata/*.csv`。
- OmniContact：`qa.json`。
- `robots.json` → `robots`；`staging_summary.json`、`surface_summary.json` → `robot_assets`。
- 工厂 `out/manifest.jsonl` 和 GPU 的 `results/export` → `retargets`。
- `umr_dataset/data/manifest.json` + `catalog_meta.json` → `datasets`、`shards`。
- `pretraining_foundation/cache/index*.sqlite`（240,444 条）→ 来源为 `legacy_xwbc` 的 `retargets`（format=`xwbc22`），作为对照臂。
- `MODEL_VERSIONS.md`、`RUNS_INDEX.md`、`topo_exp/runs`（823 个）、AM2 六组 runs → `runs`、`checkpoints`、`models`。
- 历次判决 `verdicts/` → `events`（kind=verdict）。

### C.2 产物存储

- 按路径寻址，加上 catalog 里的 sha。不做按内容哈希的对象存储：迁移成本高，lloyd 也看不懂。
- 每个产出目录带 `MANIFEST.json`，写完以后原子改名、只读。
- 引擎原样写出的文件不改动。补充信息一律写进旁挂的 `<文件>.meta.json`。
- 模型推理的输出（rollout）存成 `format=model_rollout_v2`，字段和 am_motion 一样，再加 `model_contact_head`（模型自己预测的接触，只用来对照，判官不读）。

### C.3 训练分片格式

- **直接用 `umr_dataset` 的 UMRS v1**，不另造。它已经验证过：
  - 不可变，带校验和，可以 mmap；
  - 21 台机器人和人体按帧精确配对；
  - 姿态解码误差 ≤ 4.5e-5 rad；
  - 表面点误差 ≤ 0.062 mm；
  - 接触位掩码按位精确；
  - 解码后的严格性守卫（decode_guard）。
- 数据集 = catalog 里一次冻结的选择，构建规则：
  1. 条件：许可策略通过；状态是 `accepted`，或者"自动判官通过、所在批次已放行、没被人工拒绝"；不属于黄金评测集的 `split_group`。
  2. 被 flag 的帧按规格剔除，并在 `datasets.payload` 里报出精确帧数。
  3. 调 `umr_dataset` 的 convert，写到 `data/s5_datasets/<名字>@v<N>/`。
  4. 写 `dataset_members`，冻结。
- 旧训练缓存（`cache_amass_lconv_tfix`、`cache_lafan_v3`、`bridge_codes`、`text_align`、`x2_pool_v3`、`wbc_motions`）只登记成 `kind=legacy_cache`，原地软链，**不改里面的软链结构**。地图里已经查清：它们跨根引用了 `~/tmp/cl_v3`。

### C.4 磁盘预算（按今天的实测估）

**单位换算**：
- GPU 导出的 `gpu_npz_v2`：约 310–330 字节 / 机器人帧（4 个文件实测）。
- UMRS 分片：约 126 字节 / 机器人帧（2.30 MB / 18,207 帧，含人体）。
- GPU 吞吐：2,707 机器人帧 / 秒（4 台，B=256，10-07 实测）。

| 规模 | 机器人帧数 | GPU 纯计算 | 限流后估计（≤ 70% 利用率） | npz 中间文件 | UMRS 分片 |
|---|---|---|---|---|---|
| 手上 126.4 h × 21 台（30 fps） | 2.87 亿 | 29 h | 35–45 h | 92 GB（不全留） | **约 36 GB** |
| 已知 571.4 h × 21 台 | 13.0 亿 | 133 h | 7–8 天 | 415 GB（不可能全留） | **约 163 GB** |

**/home 分区（剩 345 GB）的账**：
- HiPHI 还要下约 142 GB → 剩约 200 GB。
- 126 h 的分片 36 GB → 剩约 165 GB。
- 中间文件封顶 40 GB（见下）→ 剩约 125 GB 余量。
- 571 h 全量的分片（163 GB）加 BONES 全包（45 GB）**放不下**，所以必须先解决下面两件事中的至少一件：
  1. 删 `xwbc_full`（在根分区，可以重新下载），根分区回到约 200 GB 可用，把可重生成的东西（`regen` 级）放过去。
  2. 接冷存储，把 `archive/` 和 `topo_exp` 的 140 GB 旧缓存挪过去。

**水位**（写在 `budgets.toml` 里，作业层强制执行）：
- /home：已用 80% 报警；85% 时停止接新的重写入作业。
- 根分区：平台不往里写新东西；90% 报警。
- `s3_retarget` 中间文件总量封顶 40 GB。超过以后新的重定向作业等打包和回收完成再跑。

### C.5 保留与清理

| 保留级别 | 包括 | 规则 |
|---|---|---|
| `keep` 永久 | 原始数据（许可允许的前提下）、许可原文、catalog 和备份、人工判决、lloyd 拍板记录、发布物、黄金评测集、排行榜、血统里的 checkpoint、报告引用的媒体 | 永不自动删 |
| `keep_n` 留 N 个 | 训练 checkpoint | 每个 run 保留 last、best、里程碑；run 结束 14 天后其余的进清理计划 |
| `regen` 可重生成 | `s3_retarget` 中间 npz、90 fps 的 SMPL-X（训练用 30 fps）、`umr_gpu/problems`（5.7 GB）、审计网页 `cache/arrays`（3.1 GB）、没被引用的渲染 | 打包并且过了 14 天以后进清理计划；永远保留 2% 随机样本，以及所有被人审过或被 flag 过的条目 |
| `scratch` 临时 | `scratch/<日期>_<任务>/` | 14 天过期；每天扫描提醒 |
| `cache` 缓存 | `cache/`、hf 元数据、inductor/triton/warp 编译缓存 | 随时可删 |

- `amp s5 gc plan` 只出清单：每条写明路径、大小、保留级别、谁引用它、能不能重新生成。
- 执行只能由 lloyd 在本机做：`gc apply <plan_id> --i-am-lloyd`。默认不自动删任何东西；lloyd 可以在 `budgets.toml` 里给 `scratch` 和 `cache` 打开自动清理。

---

## D. 模型侧

### D.1 训练 run 登记

- `runs`、`checkpoints`、`models` 三张表（见 §C.1）。新训练一律经作业层提交，自动登记：
  - 代码 commit（monorepo，或 research 仓的 commit）；
  - 配置全文；
  - 用了哪些数据集 id；
  - 预计和实际 GPU 小时、kWh；
  - 批准文件引用。
- 旧 run 一次性导入：
  - `topo_exp/runs`：823 个；按 `MODEL_VERSIONS.md` 标出血统，血统里的标 `lineage`，其余标 `archive`。
  - AM2 六组和 continuation_v3。
  - megatron 试训。
  - `alphamotion_ctrl/logs`（WBC）。
  - `pose_translator_2`（归档）。
- 头衔：`models.title_status` 只能由人在网页上改成 `crowned`，`crowned_by` 必填。对决只产出候选（lloyd 定的规矩：没确认之前不能叫 Greenwich）。
- 发布限制会继承：模型用过的数据集只要带 `release_restrictions`（比如 HiPHI、BONES 的"不许训练替代生成模型"），模型自动标"发布前要过法务"。

### D.2 实时可视化：看模型生成了什么

**目标**：训练中、评测时，随时打开浏览器就能看到某个 checkpoint 在某段动作上的输出，并且和人体源、参考重定向并排对照。

**做法：包住产品仓的 LiveViewer（viser 常驻、就地换内容，做过手机端验收）**，新增一个场景适配器 `code/s6_train/viz/live/scene_am_motion.py`，复用 am2exp 的 `render/robot_model.py`（厂商 MJCF 网格）和 `render/contact.py`（按区域上色）。

画面布局（同一条时间轴）：

```
┌──────────────┬──────────────────┬──────────────────┬──────────────┐
│ ① 人体源     │ ② 参考（对照臂）  │ ③ 模型输出（本臂） │ ④ 可选：别的  │
│ SMPL-X 网格  │ UMR 重定向       │ ckpt X           │ ckpt/机器人   │
│ 或 4096 点   │ MJCF 网格+表面点  │ MJCF 网格+表面点  │ （最多 6 台）  │
│ 人体接触着色 │ 文件里的接触      │ 判官现算的接触    │              │
└──────────────┴──────────────────┴──────────────────┴──────────────┘
  时间轴 ── 每帧判官指标的小曲线（穿透 / 悬空 / 打滑）── 跳到最差帧
  开关：参考绿色鬼影叠在③上 · 表面点开关 · 只看接触点 · 模型接触头 vs 判官接触
```

- 颜色约定和审计网页一致：绿 = 地面接触，橙 = 物体接触，红 = 穿透 > 0.5 mm，黄 = 人体在接触但机器人悬空 1–20 mm，灰 = 其他。
- ①人体源必须显示，这是 lloyd 定的渲染纪律。
- ③上的接触分两种：判官独立现算的（权威），和模型接触头预测的（虚线，只供对照）。
- 数据通路：模型推理 → 写一个 `model_rollout_v2` 文件到 `evals/.../rollouts/` 或 `runs/<id>/live/<step>/` → 在 catalog 登记 → `viewer_load(rollout_id, compare=[source, reference])`。**查看器只读文件，不直接接模型。**这样同一份输出在网页的 3D 审片页、判官、渲染器里看到的完全一样。
- 训练中途：watcher 收到新 checkpoint 后：
  1. 用 CPU 在 3–5 个黄金窗口上推理，不和训练抢 GPU；
  2. 几何判官打分；
  3. 渲染 12 帧网格和 10 秒三联视频；
  4. 推到查看器，登记 `media`。

  `run_live_latest` 返回最新那段视频的 URL。按 lloyd 的规矩，问训练进度时必须附视频。
- 进程纪律：
  - 一个查看器实例，只监听 127.0.0.1；
  - 空闲 8 小时自动退出；
  - 内存封顶 4 GB；
  - 启停用 PID 加启动标识核对，不用 `pkill -f`。
- 网页那边：审计网页（Three.js）加一条"模型输出"通道，读同一种格式，用来浏览和审片。viser 查看器负责实时推送和调试。两边读同一种格式，不会出现两套各说各话。

### D.3 模型重定向评测套件

**判官（都独立：只依赖 numpy、mujoco、torch-FK，不读训练损失，也不读模型接触头）**：

| 判官 | 来源（包住） | 量什么 | 尺子 |
|---|---|---|---|
| `geometry@v1` | 合并 am2exp J1 `ground_judge` 和 `umr_gpu/eval/judge.py`（拷进 monorepo，原件不改） | 地面、物体、自身穿透（mm，p95 和最大值，> 0.5 mm 的帧数）；实接触率；悬空；接触点打滑（p95，> 1 cm/s 的帧数）；接触闪烁（< 3 帧的段）；和人体接触一致（逐区域 F1） | 厂商 MJCF 碰撞几何 + 4096 稠密点；阈值来自 `data_spec_v2.toml` |
| `locomotion@v1` | arena `eval/locomotion.py` + `locomotion.v1.json`（研究判官 wbc_ready_audit、x2_phys、slip_audit 的正式移植版） | 逐段 gate；缺材料时报 `not_evaluated`，不会默默算通过 | 冻结标准文件 |
| `manip@v1` | am2exp J2 `closure_judge` | 抓握：握住、张开、压碎三档 | 按 episode bootstrap |
| `pose_fidelity@v1` | 新写（薄） | 和参考比：关节角用测地线距离（避开 3 自由度欧拉分支伪造的 100° 超限）；手脚末端位置误差（按机器人身高归一）；根轨迹误差。和人体比：肢体方向余弦 | **真值不是唯一的尺子**，必须和几何、物理一起看 |
| `physics_nogt@v1` | `x2_phys` 的逻辑 | 根高抖动、关节速度和加速度超限、关节限位违例 | 尺子 = 人体源自己 |
| `codec_regression@v1` | `v2_judge.py` 固定窗口 | 防腿、手退化：幅度比、上肢 FK、腿角警戒线 | 冻结窗口 |
| `wbc_track@v1` | `alphamotion_ctrl`（mjlab）`evaluate_local` | WBC 能不能跟住 | 吃 GPU，**每次都要批准** |

**黄金评测集（冻结，带 sha，永不进训练集）**：

| 集合 | 内容 | 来源 |
|---|---|---|
| `G-loco@v1` | 走、跑、转身、楼梯、坡（30 个窗口，每个 4–10 秒） | GRAIL 地形、LAFAN、HiPHI |
| `G-ground@v1` | 跪、爬、躺、起身 | 现在手上约 0 h，先用 LAFAN、AMASS 留出集，等 BONES/HiPHI 到了升级成 v2 |
| `G-hoi@v1` | 搬箱、踢球、坐椅子 | OmniContact（箱子检查通过后）、GRAIL 坐、HiPHI 物体 |
| `G-hand@v1` | 手部操作 | HiPHI 上肢操作类 |
| `G-heldout600@v1` | 现成的 600 段语料外审计集 | `cache_amass_heldout` |
| `G-unseen-robot@v1` | 零样本机器人（X2、A3） | 登记表里非训练的机器人 |
| `G-random@v1` | 50 个无偏随机窗口 | 全库抽样 |

**协议（照搬 am2exp E5 统计层，并落实对照臂纪律）**：
- 每次评测都同时评"数据地板"：参考重定向本身，用同一套判官。
- 对照臂：随机方向，或打乱来源。
- 同尺重测。
- A/B 判决只走 `stats.decide()`：按窗口聚类的配对符号检验和置换检验，规则冻结在代码里。不凭肉眼比数字。

**视频三联**：
- 由 am2exp `render/triptych_*.py`、`grid12.py`、rollout_tools 的 `encode.py`（逐帧检查时间戳）合成 `code/s7_eval/render/`。
- 布局固定为：人体源 | 对照臂 | 本臂。1080p，真实时间戳。
- 12 帧网格覆盖全程，必须含首帧和末帧。
- 给 lloyd 的视频一律走这一个入口，登记进 `media`，用 `SendUserFile` 发 mp4。

**排行榜**（`GET /api/v1/leaderboard?suite=&robot=`，网页一页）：

```
| 被评对象 | 判官版本 | 机器人 | 穿透p95 | 实接触 | 悬空 | 打滑p95 | 闪烁 | 人体一致F1 | 姿态误差 | 未评数 | 对数据地板 | 统计判决 | 视频谁看过 | 头衔 |
| 数据地板(umr_gpu@3f2a9c1) | geometry@v1 | g1 | 0.3mm | 0.97 | 0.01 | 0.8 | 0 | 0.94 | — | 0 | — | — | lloyd 10-09 | — |
| ckpt C1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | 2 | +0.2mm | 不劣(p=0.41) | 未看 | 候选 |
```

- 规则：一行在"视频谁看过"填上之前，不能标"通过"。这条落实 lloyd 定的"不用数字做动作评估"：分数只回答"跑没跑完整个窗口"，不回答"动作做没做出来"。
- 底线行永远显示：UMR GPU 严格版、GMR 原生、xwbc。

---

## E. 迁移计划（每个现有文件夹的去向）

### E.0 原则和动手前的检查

- **动作代号**：
  - 【搬】移动（代码进 monorepo，或者同分区内改名）；
  - 【链】原地不动，软链挂进根目录，登记 catalog；
  - 【登】只登记；
  - 【档】进 `archive/`；
  - 【删?】删除候选，要 lloyd 逐条批；
  - 【外】不属于 AlphaMotion。
- **不能动**：
  - 正在写的目录（下载器、GRAIL 循环、`gpu_convert` 定时任务、`umr_gpu` 吞吐测试、`throttle_guard`、两个工作流代理）；
  - 3 个常驻服务读的目录；
  - 4 个做过一致性测试的引擎的代码内容。
- **动手前检查**（脚本 `amp ops precheck <路径>`）：
  - 扫 `/proc/*/cwd` 和 `/proc/*/cmdline`，看有没有进程引用这个路径。不用 `lsof +D`，那个太重。
  - 查 catalog 的 `legacy_paths`，看有没有代码写死了它。
  - 查软链指向它的目录，比如 `cache_lafan_v3` → `~/tmp/cl_v3`。
- **搬的方式**：先软链挂上 → 代码改成从配置读路径 → 进程空闲时 `mv` 改名（同分区，秒级）→ 旧位置留反向软链 30 天 → 再撤。

### E.1 阶段和时间

| 阶段 | 什么时候 | 做什么 | 用时 | 风险 |
|---|---|---|---|---|
| P0 | 今天或明早 | 建骨架、写 README；`stores/` 和 `repos/` 软链；文档复制进 `docs/`（含孪生目录里独一份的 `SCALING_LAW.md`、另一版 `CTRL_VERIFY_GR3.md`、ctrl_verify 视频）；4 张地图放进 `docs/maps`；把拍板单交给 lloyd | 1 h | 零（只新建） |
| P1 | 第 1 天 | 4 个引擎和 `motion_audit_web`、`scripts*` 原地 `git init`，写 `.gitignore` 排除产出，打快照 tag `parity-20261007`（只增加 `.git`，不动文件，需要 lloyd 同意）；`alpha_motion2` 里活跃的代码（`pretraining_foundation`、`am2exp_v1`、`orchestration`、`rollout_tools`）拷进 monorepo 立刻进 git；catalog 建表，所有导入器跑一遍；`paths.toml` 写完；生成第一版 `INDEX.md` | 4–6 h | 低 |
| P2 | 第 1–2 天 | 读取层（v1、v2、人体 4 种存法）；修 `umr_dataset` 的 HiPHI 路径（经读取层）；审计网页加 v2 读取器；箱子检查 WP-S3b；重定向适配器；**首批量产**（见 E.2） | 1 天 | 中（第一次量产） |
| P3 | 第 3–5 天 | 服务上线（API + MCP，并入今天的 MCP）；作业层；判官合并；三联渲染入口；实时查看器读 am_motion；冻结 OpenAPI 交给 Codex；s4 质量汇总和人审队列 | 3 天 | 低 |
| P4 | 第 2 周，下载器和 GRAIL 循环跑完以后 | 代码物理搬家：`git subtree add` 把引擎并进 monorepo，旧位置留软链；逐个替换写死的路径（工厂 19 处、arena 10 处、Studio 7 处、`umrf/paths.py` 等，用 `amp paths lint` 清单打勾） | 2–3 天 | 中（改路径要回归测试） |
| P5 | 第 2–3 周 | /home 内数据改名搬家（`~/umr_factory_data/*` → `data/s*`，秒级），旧顶层留软链；/var/tmp 里的小件生产资产拷到 /home（umr 环境 6.7 GB、UMR 1.2 GB、登记表、许可原文、umr_gpu 产出 7.7 GB），原地留软链；`topo_exp` 的 160 GB 看冷存储决定 | 2 天 | 中 |
| P6 | 第 3 周起 | 归档和删除候选清单（E.12）交 lloyd 逐条批；业务资料挪到 `~/business` | 按批 | 低 |

### E.2 首批量产路线（第 2 天）

1. 等 `umr_gpu` 吞吐测试结束（超时 3000 秒）。和 `gpu_convert` 错开：重定向跑的时候暂停它的定时任务（`run_watch.sh`），保证同时只有一个重 GPU 任务。
2. `amp s3 plan --clips "source IN (grail) AND class IN (locomotion,sit)" --robots core21`，再加 HiPHI 非箱子的 200 段和 LAFAN 样本。估计约 30 h 动作 × 21 台 ≈ 6,800 万机器人帧，GPU 约 7–10 h，中间文件约 22 GB，分片约 9 GB。
3. 第一批先跑 4 台（G1-29、GR3、H2、A3）× 100 段，用独立判官全量复核，在三联视频里抽看 20 段，lloyd 点头以后放开到 21 台。
4. 人审队列放 50 段按风险挑的 + 50 段随机的；随机抽样合格率过了放行门，这批才打包成 `am_pretrain@v1`。
5. 箱子检查通过以后，再排 OmniContact（12 h，约 3 h GPU）和 HiPHI 带箱子的片段。

### E.3 `~/umr_factory_data`（191 GB，/home）

| 目录 | 动作 | 新位置 | 阶段 | 前提 |
|---|---|---|---|---|
| `raw/hiphi` | 链→搬 | `data/s1_raw/hiphi` | P0 链 / P5 搬 | 下载器结束（约 99 h） |
| `raw/grail` | 链→搬 | `data/s1_raw/grail` | P0 / P5 | GRAIL 循环结束 |
| `raw/omnicontact` | 链→搬 | `data/s1_raw/omnicontact` | P0 / P5 | — |
| `smplx/hiphi`（CPU 1,094 段） | 登→搬 | `data/s1_human/hiphi/smplx_cpu@v1`；这批没有 30 fps 版本，补转后标 `regen` | P1 / P5 | — |
| `smplx_gpu/hiphi`、`hiphi_30fps` | 链→搬 | `data/s1_human/hiphi/smplx@v2/{fps90,fps30}`；fps90 标 `regen` | P0 / P5 | 转换定时任务停 |
| `smplx/omnicontact` | 链→搬 | `data/s1_human/omnicontact/smplx@v1` | P0 / P5 | — |
| `smplx/grail` | 链→搬 | `data/s1_human/grail/smplx@v3`（manifest 是逐段 license 的样板） | P0 / P5 | GRAIL 循环结束 |
| `smplx/_test` | 删? | — | P6 | — |
| `robots/` | 链→搬 | `data/s2_robots/<r>/`（+ `registry.json`） | P0 / P5 | robots_wf 代理结束 |
| `robots/<r>/pipeline/` | 登→搬 | 配置 → `data/s2_robots/<r>/pipeline.json`；脚本 → 被 `geometry@v1` 取代后 → `archive/2026-10_robots_wf` | P3 / P5 | — |
| `corr/` | 链→搬 | `data/s2_robots/<r>/corr@factory_20261004` | P0 / P5 | WP-S2a 查清差异 |
| `runs/`（7 段 CPU 原始结果） | 搬 | `data/s3_retarget/umr_cpu@d1cdaed/runs_20261005`（`human_slots` 被引用） | P5 | — |
| `out/`（factory_npz_v1，11 个） | 登→搬 | `data/s3_retarget/umr_cpu@<哈希>/20261005_pilot`；箱子片段标隔离 | P1 / P5 | — |
| `qa/`（5 台，格式各异） | 搬 | `data/s4_dataset_eval/2026-10-07_robots_qa/` | P5 | 代理结束 |
| `qa_renders/` | 档 | `archive/2026-10_conversion_qa` | P6 | — |
| `diversity/` | 搬 | 脚本 → `code/s4_dataset_eval/census/`；产物 → `data/s4_dataset_eval/2026-10-07_diversity/` | P4 / P5 | 代理结束 |
| `audit/` | 链 | `data/s4_audit/`（约定文件路径不变，审计网页写这里） | P0 | — |
| `logs/` | 搬 | `services/factory/logs/`，加轮转（`grail/dl_loop.log` 已 406 MB）；过期 pid 进清理计划 | P5 | 下载器结束 |
| `pilot/` | 搬 | `scratch/2026-10-04_pilot/` | P5 | — |
| `hf_cache/` | 搬 | `cache/hf_meta/` | P5 | — |
| `workflows/` | 搬 | `code/workflows/`；`diversity_wf.js` 写死 `/tmp/claude-1000/...`，改成读配置 | P4 | — |
| `scripts/`、`scripts_grail/`、`scripts_streamc/` | 搬 | `code/s1_clean/{download,convert_cpu,grail,hiphi}/`；`throttle_guard.sh` → `code/ops/` | P1 进 git / P4 搬 | 循环和下载器结束 |
| `DATA_SPEC_v2.md`、`RESOURCE_LIMITS.md` | 搬（原处留软链） | `docs/specs/`、`docs/ops/` | P0 复制 / P5 | — |
| `platform_design/` | 搬 | `docs/maps/` | P0 复制 | — |

### E.4 家目录下的代码

| 目录 | 动作 | 新位置 | 阶段 |
|---|---|---|---|
| `~/umr_gpu` | 登（git 快照）→ 搬 | `code/s3_retarget/engines/umr_gpu`（代码原样）；产出路径照旧，经软链 | P1 / P4 |
| `~/umr_factory_code` | 同上 | `code/s3_retarget/engines/umr_cpu`；`.bak_taskB` 进 git 历史后删 | P1 / P4 |
| `~/gpu_convert` | 同上 | 代码 → `code/s1_clean/convert_gpu`；`runs/parity40` → `data/s4_dataset_eval/engine_parity/convert_parity40`；`runs/bench*`（5 GB）删?；`.inductor_cache` → `cache/` | P1 / P4 |
| `~/umr_dataset` | 同上 | 代码 → `code/s5_store/pack`；`data/` → `data/s5_datasets/umrs_pilot@v0` | P1 / P4 |
| `~/motion_audit_web` | 同上 | `code/web`（前端 + 后端 + agent_tools）；`cache/arrays` → `cache/audit_arrays`；`.venv`、`.mcp-venv` → `envs/` | P1 / P3 |

### E.5 `/var/tmp/alphamotion`（352 GB，根分区）

| 目录 | 动作 | 新位置 | 阶段 |
|---|---|---|---|
| `umr_factory/umr/UMR`（d1cdaed） | 链→拷 | `third_party/UMR@d1cdaed`；运行时数据（data/、output/）分出去，放 `cache/umr_runtime` | P0 / P5 |
| `umr_factory/umr/env` | 链→拷 | `envs/umr`（+ `conda list --explicit` 锁文件）；原处留软链 | P0 / P5（GRAIL 循环结束后） |
| `umr_factory/umr/tools` | 搬 | `code/s3_retarget/engines/umr_cpu/tools`（`umr_run.py` 被依赖） | P4 |
| `umr_factory/umr/{configs,eval,runs,logs,videos}` | 档 | `archive/2026-10-04_umr_scout` | P6 |
| `umr_factory/robots/`（登记表） | 拷 | `data/s2_robots/registry.json` + `registry/umr_configs/` | P1 |
| `umr_factory/umr_gpu/`（产出） | 链→拷 | `results/export` → `data/s3_retarget/umr_gpu@<哈希>/20261005_samples`；`results/parity` → `data/s4_dataset_eval/engine_parity/`；`videos` → `media/2026-10-0x_umr_gpu_parity`；`problems`（5.7 GB）标 `regen`；`robots/` 删?（和工厂那份逐字节相同）；`corr/` 登 | P1 / P5 |
| `umr_factory/labels/` | 档 | `archive/2026-10_pcl_contact_proto` | P6 |
| `umr_factory/sources/` | 拷 | `data/s1_sources/`（许可原文的唯一出处） | P1 |
| `umr_factory/uniml3d_reload/` | 档 | 元数据 → `data/s1_sources/uniml3d`；1.3 GB 样本缓存删? | P6 |
| `umr_factory/lloyd_gpu_eval/` | 链 | `data/s4_dataset_eval/engine_parity/2026-10-04_first_parity`（`run_parity.py` 写死了路径，留软链） | P1 |
| `umr_factory/{hiphi2smplx,h2s_shim,beta_fit}` | 链→拷 | `third_party/hiphi2smplx`、`code/s1_clean/convert_gpu/h2s_shim`、`data/s1_sources/body_models/beta_fit`（和 `~/Downloads/SMPLX_NEUTRAL.npz` 放一起） | P0 / P5 |
| `topo_exp/cache_amass_lconv_tfix`、`cache_lafan_v3`、`cache_xwbc_amass_full`、`cache_xwbc_lafan`、`cache_amass_lconv`、`cache_amass_heldout`、`bridge_codes`、`text_align` | 链（登 `legacy_cache`） | `data/s5_datasets/legacy/<名字>@<版本>` → `stores/sys/...`；**内部软链结构不动** | P1；冷存储到位后再议 |
| `topo_exp/runs`（823 个） | 链→分 | 血统 run → `runs/greenwich/`、`runs/equator/`（拷 ckpt + 配置）；其余 → `archive` 或删? | P1 登 / P6 |
| `topo_exp/cache_lafan_v2`、`cache_amass_4000` | 删?（坏链） | — | P6 |
| `topo_exp/cache_paired*`、`cache_big`、`cache`、`cache_amass_v3`、`cache_xwbc_lafan_fixed`、`v5_keycache` | 档 | `archive/2026-08_greenwich_caches`（`cache_paired_fixed` 还有引用，先留软链） | P6 |
| `topo_exp/night2_lib` | 链 | 有 11 处引用；有一个软链指回研究仓的 `grand_eval` | P1 |
| `topo_exp/{qwen3,siglip}_names.pt` | 拷 | `data/s2_robots/_embeddings/`；`.bak`、`.clobbered` 删? | P5 |
| `xwbc_full/`（129 GB tar） | 删?（HF 可重下） | — | P6（腾根分区的首选） |
| `xwbc_lafan/extracted` | 链 | `data/s3_retarget/xwbc@v1/lafan`（对照臂） | P1 |
| `am2exp/` | 链→拷 | `runs/am2exp/stage0`、`evals/am2exp_e1e2@v1`、`media/2026-10-04_am2exp_triptych`；`assets/surface_v2` → 档 | P1 / P5 |
| `ctrl_verify/` | 链 | `data/s5_datasets/legacy/wbc_tracks@v1` | P1 |
| `config_harvest/`（GMR 等） | 链→拷 | `third_party/GMR@harvest_20260803`、`third_party/harvest/*` | P1 / P5 |
| `robot_models/` | 链 | `third_party/robot_models`（MJCF 的源头；s2 的展平副本才是生产用的） | P1 |
| `gmr_production` | 链 | `data/s3_retarget/gmr@v2`（Native 对照臂） | P1 |
| `gmr_production_v1` | 删? | — | P6 |
| `gmr_spike`、`gmr_upstream` | 链 | `third_party/GMR/{tuned_ik,upstream}` | P1 |
| `any_body/` | 档 + 删?（`env` 12 GB） | `archive/2026-07_anybody`（只留 `cache/*.pt` 和说明） | P6 |
| `hf_cache/` | 链 | `cache/hf_models`（bge-m3、Qwen3 嵌入；文本层要用） | P1 |
| `studio/`、`studio_app.py`、`gem_*`、`genmo_*`、`tmr_bridge` | 档 | `archive/2026-08_studio`；worker 协议写进 `docs/` | P6 |
| `beyondmimic/`、`hf_home/`、`xwbc_clips/`、`xwbc_dl/`、`mesh_viz/`、`ab_test/` | 档 / 删? | — | P6 |
| `omniretarget/` | 链 + 删?（重复的 zip） | `third_party/holosoma`、`data/s1_raw/omniretarget` | P6 |
| `dex-retargeting`、`anytop` | 链 | `third_party/` | P1 |
| `spatial_recipe_ablation_20260816`、`boneseed_spatial_*`（20 个） | 档 | `docs/reports/2026-08-16_spatial_recipe_ablation`、`archive/2026-08-16_boneseed_spatial` | P6 |
| `pose_translator_2`（软链） | 删?（链接本身） | — | P6 |
| 根目录约 70 个散文件 | 档 | 队列脚本 → `archive/queues`；视频 → `media/2026-08-16_boneseed`；`xwbc_joint_names.json` → `data/s1_sources/xwbc/` | P6 |

### E.6 `~/tmp`（56 GB）

| 目录 / 组 | 动作 | 新位置 | 阶段 |
|---|---|---|---|
| `cl_v3/`（31 GB） | 链→搬 | `data/s3_labels/contact@v3`；搬的时候同步改 `cache_lafan_v3`、`cache_amass_lconv_tfix` 里的 34 个链接；`cl_v3/amass`（15 GB，只有旧缓存用）删? | P1 链 / P5 搬 |
| `sole_offsets*.json` | 拷（原处留软链） | `data/s2_robots/<r>/sole_offset@v3.json`（另外出 v4，见 B.2） | P1 |
| `gauge_check_0903.json`、`dialect/` | 拷 + 软链 | `data/s4_dataset_eval/2026-09-03_gauge_check`、`data/s1_sources/human_dialect_Q@v1` | P1 |
| `refiner_suite/` | 链 | 代码 → `repos/research` 引用；`out/` → `evals/refiner_suite@v1`（精修器线已叫停，当对照臂） | P1 |
| `cl_v2_qc/`、`contact_qc/`、`ab_sole0/1/` | 链 / 档 | `evals/contact_qc_legacy/`（有引用，先链） | P1 / P6 |
| `greenwich_workbench_0911/datagrid`、`arena_handoff_backup/` | 拷 | `services/datagrid/{state,backups}`（**唯一的数据库备份，先多拷一份**） | P0 |
| `greenwich_workbench_0911/` 其余（产品和 SDK 副本、venv） | 档 | `archive/2026-09_greenwich_workbench` | P6 |
| `ufo_resume_drill/`、`ufo_h2_smoke/`（13.5 GB） | 删?（配置留进 `archive`） | — | P6 |
| `ufo_p*`、`ufo_probe*`、`ufo_smoke_g1*`、`ufo_cache/`、`cache/`、`torchinductor_*` | 档 / 缓存 | 配置 → `runs/wbc_ufo/_pilots`；缓存 → 删? | P6 |
| `bfm_data_clone/` | 删?（和 `reference_code/bfm_zero` 重复） | — | P6 |
| `h2_*`、`h2bm`、`h2gr*`、`h2hd`、`h2mp`、`g1gr`、`seed_g1_rs` | 档 | `media/2026-09-1x_h2_wbc`、`runs/wbc_mjlab/h2/` | P6 |
| `teleop2/`、`ee_data/`、`ee_fill/`、`ego_fill/`、`*_0905/` | 档 | `data/s1_raw/egodex`（ee_data）、`evals/archive/2026-09-05_ee_fill`、`media/` | P6 |
| `wbcready_*`（162 个）、`wbcready_vid/`、`x2phys/` | 档 | `evals/archive/wbc_ready/<ckpt>/`、`evals/archive/x2phys/` | P6 |
| `stress*`、`crossrig*`、`slipvid_*`、`cmp34*`、`y*` 等约 1.2 GB | 档 | `evals/archive/2026-08-29..09-04/<主题>/` | P6 |
| 根目录 194 个 mp4（703 MB） | 档 | `media/<日期>_<主题>/`（报告引用的优先） | P6 |
| 根目录约 1,240 个 json/log/sh/py | 档 / 链 | 有引用的（`ablation_table.csv`、`scaling_*.csv`、`stress_run.py`、`qpos_to_rs.py`、`xwbc_to_robotstate.py`）→ 链；其余打包成 `archive/tmp_2026-08_09.tar.zst` | P6 |
| 152 个 wandb 临时目录 | 删? | — | P6 |

### E.7 `~/alphamotion_artifacts`、`~/alphamotion_ctrl`、`~/reference_code`

| 目录 | 动作 | 新位置 | 阶段 |
|---|---|---|---|
| `alphamotion_artifacts/pose_translator_2`（11 GB） | 档 | `archive/2026-08_pt2` | P6 |
| `alphamotion_artifacts/x2_pool_v3` | 链 | `data/s5_datasets/legacy/x2_pool@v3` | P1 |
| `alphamotion_artifacts/x2_pool`、`x2_pool_v1_broken` | 档 / 删? | — | P6 |
| `alphamotion_artifacts/data_contract_audit_20260817` | 档 | `evals/archive/2026-08-17_data_contract` | P6 |
| `~/alphamotion_ctrl`（代码） | 链 | `repos/ctrl`（自带 git，不并） | P0 |
| `~/alphamotion_ctrl/motions`（8.6 GB）、`logs/` | 链 | `data/s5_datasets/legacy/wbc_motions@v1`、`runs/wbc_mjlab/` | P1 |
| `~/alphamotion_ctrl/.git`（14 GB） | 瘦身提议 | 仓库主人执行 `git gc --prune`，可收回 9 GB 以上 | P6 |
| `reference_code/ufo` | 链 | `third_party/ufo`；我们自己的配置（h2_31dof、greenwich 等，**还没提交**）先拷进 `code/s7_eval/wbc/bfm_zero_overlay/`；两套 venv 删一套? | P1 拷配置 / P6 |
| `reference_code/{sonic,bfm_zero,beyondmimic}` | 链 | `third_party/`；H2 资产拷一份到 `data/s2_robots/unitree_h2/` | P1 |

### E.8 研究主仓 `~/Desktop/"be water, robot"`（85 GB）

整体链成 `repos/research`，研究仓自己的 git 不变。里面的东西按下表分流（拷贝或登记，不删原件）：

| 子目录 | 动作 | 新位置 |
|---|---|---|
| `alpha_motion2/pretraining_foundation` | 搬代码（P1，进 git） | 代码 → `code/s1_clean/acquire/`（下载器）+ `code/s5_store/foundation_loader/`；索引 sqlite → 导入 catalog；**下载器跑完以后才动** |
| `alpha_motion2/research/alphamotion_scaling_20261004` | 搬代码 + 链产物 | `orchestration`、`am2exp_v1/{train,ops,judge,eval,e2,render}`、`rollout_tools` → `code/s6_train/models/am2`、`code/amp/jobs`（ops）、`code/s7_eval/{judges,render}`；runs、evaluation、reports → `runs/am2/`、`evals/am2_scaling_20261004/`（链） |
| `.../megatron_scale_pilot_20261004` | 代码进 git + 链 | `code/s6_train/models/am2/recipe/`（冻结）；runs → `runs/am2/megatron_pilot` |
| `.../megatro_scale_pilot_20261004`（拼错） | 并入上一行后删? | — |
| `.../embodied_case`、`hand_surface_case`、`architecture_audit`、`transformer_review`、`training_recipe_audit` | 链 / 档 | `docs/reports/cases/`；`hand_surface_case` → `data/s4_dataset_eval/cases/` |
| `alpha_motion2/external/GMR` | 链 | `third_party/GMR@unknown`；产品和工厂 19+6+10 处引用，**P4 改路径前不动** |
| `alpha_motion2/external/{UMR,UniMate,soma-retargeter,whole_body_tracking}` | 链 | `third_party/` |
| `alpha_motion2/external/GRAIL`（本地改了 3,465 个文件） | 链 + 导补丁 | `third_party/GRAIL@<commit>` + `patches/GRAIL/` |
| `alpha_motion2/external/{UMR-website,robocasa,GR00T-...-motionbricks,nvidia-cosmos,cosmos-framework}` | 档 / 删? | — |
| `alpha_motion2/demo/ardy_preflight_20`、`alphamotion_verified_g1_atlas_20` | 链（服务在读） | `services/legacy_demo/`；服务停掉以后再档 |
| `alpha_motion2/demo` 其余、`runs/`（11 GB）、`web/`、`scripts/`、`live/`、`synergy` 等 7 月演示栈 | 档 | `archive/2026-07_ardy_sonic_stack/`（`runs/synergy` 9.8 GB 删?） |
| `alpha_motion2/wbc_expert` | 档 | 原始数据 → `data/s1_raw/{lafan1_g1,omniretarget,bones_probe}`；其余 → `archive/2026-06_wbc_expert` |
| `alpha_motion2/analysis/boneseed_locomotion_index` | 链 | `data/s4_dataset_eval/legacy/boneseed_tiers` |
| `alpha_motion2/{evaluation,geometry,heads,losses,models,modules,training}`、`data/*.py`（0 字节） | 删? | — |
| 13 个预览目录 | 档最终版，其余删? | `archive/2026-06_format_previews` |
| `presentation`、`market_research`、`pitch_deck_20260722`、`assets` | 外 | `~/business/` |
| `pose_translator_design/exp_topo` | 链（冻结） | `code/s6_train/models/greenwich_legacy` → `repos/research/...`（软链）；判官和渲染器的活跃版本在 s7（arena 移植版 + am2exp） |
| `pose_translator_design/verdicts` | 链 | `docs/decisions/legacy_verdicts`（导入 events） |
| `pose_translator_design/{MODEL_VERSIONS,RUNS_INDEX}.md` | 登 | 导入 `models`、`runs` |
| `pose_translator_design/{grand_eval,R4_VIZ,R*_*,figures,viz,audits}` | 档 | `archive/2026-08_greenwich/` |
| `pose_translator_design` 约 50 份文档 | 链 | `docs/reports/greenwich/` |
| `GENMO/`（20 GB） | 链 | `third_party/GENMO`（conda `genmo-gem` 装成了可编辑包，**不能搬**） |
| `experiment/ardy`、`experiment/kimodo` | 链 | `third_party/ardy`、`third_party/kimodo`（`.venv` 5.3 GB 删?） |
| `experiment/{stage1,synergy}`、`notebook/`、`note/`、`any_body/` | 档 | `archive/2026-05_stage1/` 等 |
| `papers-docs/` | 链 | `docs/papers/` |
| `Greenwich_G1_18_Paired_Benchmark_Review_20260920/` | 链 + 外 | 评测部分 → `evals/archive/2026-09-20_g1_18_paired`；宣传片和外联名单 → `~/business` |
| `deliverables/` | 链 | `media/2026-08-17_amass_zeroshot_mesh` |
| `claude/`、`agent_chat/`、`codex/` | 链 / 档 | `docs/design_legacy/`；agent_chat 被新 MCP 取代后档 |
| `.claude/skills/{spatial-eval,motion-eval}` | 拷 | `$AM_ROOT/.claude/skills/`（评测协议照这两份写成 `golden/` 和 `suite` 配置） |
| `data/babel`、`action_labels.py` | 链 | `data/s1_sources/babel`、`code/s4_dataset_eval/taxonomy/legacy_regex.py` |
| 根目录 `*.fbx` | 链 | `data/s1_raw/arena_mocap` |
| `share_struct.md`、`nvidia_inception.md`、Stanford PDF | 外 | `~/business/` |
| `pytest-of-arenalabs/`、`tool-kits/`、`experiment/openfold_ref/`、`mon.pid`、`server_8844.pid`、`MUJOCO_LOG.TXT`、各种 cache | 删? | — |
| **孪生目录** `~/Desktop/be water, robot`（不带引号） | P0 救出 3 样东西 → 删? | `SCALING_LAW.md`、`CTRL_VERIFY_GR3.md`（另一版）→ `docs/reports/greenwich/`；视频 → `media/2026-08-29_ctrl_verify_x2` |

### E.9 桌面上的其它目录

| 目录 | 动作 | 新位置 |
|---|---|---|
| `~/Desktop/AlphaMotion` | 链 | `repos/product`；`weights_staging` → 登 `releases/alphamotion@0.2`（原地） |
| `~/Desktop/arena-datagrid` | 链 | `repos/datagrid`；`locomotion.py` 和标准文件拷进 `code/s7_eval/judges/locomotion`；`jobs` 表设计搬进 `amp/jobs` |
| `~/Desktop/alphamotion video` | 链 | `resource/report` → `docs/reports/`（编号不变）；视频 → `media/2026-09-02_showcase_fz5c` |
| `~/Desktop/pose_translator_2`、`greenwich_eval_20260826`、`_backup_before_delete_20260814` | 档 | `archive/`（greenwich-mobile 服务停掉以后） |
| `~/Desktop/mingrun` | 链（只读） | `repos/contrib_mingrun`；坏链 `bones_seed` 在 catalog 里标缺失 |
| `IsaacLab*`、`rot`、`SegDepthFusion`、`openarm*`、`claw-code` 等 | 外 | 不动 |
| 散文件：演示视频、babel zip、`github-recovery-codes.txt` | 视频 → `media/2026-09-03_stealth_demo`；babel → `data/s1_raw/babel`；**恢复码不动** | — |

### E.10 家目录杂项和服务

| 东西 | 动作 |
|---|---|
| `~/.alphamotion_workspace`、`~/alpha-motion-workspace`、`~/alphamotion2_runtime` | 留作兼容层；服务全换新以后撤 |
| `~/Downloads/SMPLX_NEUTRAL.npz` | 拷到 `data/s1_sources/body_models/`，原处留软链 |
| `~/.cache/huggingface/hub` | 登 `store=hf` |
| `~/alphamotion_cold`（空） | 改作冷存储挂载点，或删? |
| `~/codex_exports` | 外 → `~/business` |
| conda 环境 `ardy`、`rot`、`genmo-gem` | 在 `envs/` 里登记并生成锁文件 |
| 服务 `alphamotion-ardy-isaac` | **停用（拍板 1）** |
| 服务 `alphamotion-motion-lab`（2334）、`alphamotion-rollout-gallery`（8891）、`greenwich-mobile`（8899） | P3 新网页上线以后停，目录归档 |

### E.11 每个阶段的验收

- P0：`$AM_ROOT` 里所有软链都能解析，`INDEX.md` 生成出来，原有进程和服务一个没断（对比前后的 `systemctl --user` 和 `pgrep`）。
- P1：catalog 的条数和地图对得上（GRAIL 22,312、robots 34、foundation 240,444、topo runs 823）；引擎 git tag 打好。
- P2：首批 100 段 × 4 台过独立判官，三联视频抽看 20 段由 lloyd 点头；`umr_dataset` 打包成功，HiPHI 段不再报 FileNotFound。
- P4：`amp paths lint` 的写死清单清零（第三方仓库除外）；引擎一致性回归通过。
- P5：旧路径的反向软链都在；`df` 显示根分区 < 85%。

### E.12 删除候选汇总（约 180 GB，逐条批）

`xwbc_full` 129 GB、`any_body/env` 12 GB、`~/tmp/cl_v3/amass` 15 GB、`ufo_resume_drill` + `ufo_h2_smoke` 13.5 GB、`reference_code/ufo` 多出的一套 venv 7.5–10 GB、`alphamotion_ctrl/.git` gc 9 GB 以上、`alpha_motion2/runs/synergy` 9.8 GB、`experiment/kimodo/.venv` 5.3 GB、`gpu_convert/runs/bench*` 5 GB、`gmr_production_v1` 0.6 GB、`x2_pool_v1_broken` 0.7 GB、`bfm_data_clone` 0.2 GB、坏链（`cache_lafan_v2`、`cache_amass_4000`）、152 个 wandb 临时目录、0 字节模块目录、孪生目录（救出东西以后）、`umr_gpu/robots`（和工厂那份逐字节相同）。

---

## F. 第一批工作包（文件互不重叠，可以并行）

规则：
- 每个包只写自己名下的文件；
- `catalog/schema.sql` 只归 WP-01，别的包要加表或加字段，提给 WP-01；
- 所有包都不改引擎原件，不碰正在运行的目录；
- 重活一律加 `taskset -c 0-15 nice -n 10`；
- 要用 GPU 的包，开跑前先报预计时长。

| 包 | 内容 | 只写这些文件 | 依赖 | 用时 | 验收 |
|---|---|---|---|---|---|
| WP-00 骨架 | 建根目录、软链、README、配置初稿、文档拷贝 | `$AM_ROOT/{README.md,config/*.toml,docs/**,stores/*,repos/*,third_party/*(链),envs/*(链)}` | 拍板 4 | 1 h | E.11 P0 |
| WP-01 catalog | 建表、迁移、备份、`amp.catalog` API | `code/amp/catalog/{schema.sql,db.py,migrate.py,backup.py}`、`code/amp/tests/test_catalog.py` | WP-00 | 3 h | 并发读写测试；事件都记下来 |
| WP-02 路径 | `paths.toml` 解析、`resolve`、`lint`（写死清单）、`precheck` | `code/amp/paths.py`、`code/amp/ops/precheck.py`、`config/paths.toml`（WP-00 之后归它） | WP-00 | 3 h | lint 能报出地图里列的全部写死位置 |
| WP-03 导入器 | 只读导入各现有台账 | `code/amp/importers/*.py`（grail、hiphi、omni、robots、factory_out、gpu_export、umr_dataset、foundation_index、topo_runs、am2_runs、sources） | WP-01 | 5 h | E.11 P1 的条数；跑两遍结果一样 |
| WP-04 读取层 | `formats.motion`（v1、v2、rollout）、`formats.human`（4 种存法）、格式规格文档 | `code/amp/formats/{motion.py,human.py}`、`docs/specs/FORMAT_am_motion_v2.md`、`code/amp/tests/test_formats.py` | — | 5 h | 23 个现有样本读出来字段一致；packbits 和 CSR 互转按位一致 |
| WP-05 重定向适配器 | 计划、运行、补 meta、登记、入人审队列 | `code/s3_retarget/adapters/umr_gpu_job.py`、`code/s3_retarget/tests/test_adapter.py` | WP-01、WP-04 | 5 h | 小批次端到端；引擎目录前后快照不变 |
| WP-06 箱子检查 | 合成箱子片段，测 GPU 引擎和 CPU 判官；写问题单 | `code/s3_retarget/tests/test_box_object.py`、`docs/decisions/2026-10-08_box_sphere_bug.md` | — | 3 h（GPU 小于 10 分钟） | 给出结论：GPU 有没有这个 bug |
| WP-07 corr 差异 | 两份 corr 逐字段对比 | `code/s2_surface/tools/corr_diff.py`、`data/s4_dataset_eval/2026-10-08_corr_diff/` | — | 2 h | 讲清差在哪、影不影响一致性测试 |
| WP-08 作业层 | jobs 表（arena）+ 租约、守卫、回执、批准（am2exp ops）+ 资源类 + 掉网和内存暂停 | `code/amp/jobs/{store.py,worker.py,guards.py,approvals.py}`、`code/amp/tests/test_jobs.py` | WP-01 | 6 h | 两个 GPU 重作业不会同时跑；`net=down` 时暂停 |
| WP-09 服务 | 并入 `agent_tools` 的 MCP（工具名全保留）+ FastAPI 路由 + OpenAPI 冻结 | `code/amp/service/{core.py,api.py,mcp.py}`、`docs/contracts/{openapi_v1.yaml,mcp_tools_v1.md}` | WP-01、04、08；**等今天写 agent_tools 的子任务收尾** | 1.5 天 | 现有 MCP 测试 `test_mcp.py` 全过；HTTP 冒烟 |
| WP-10 判官合并 | `geometry@v1`（J1 + umr_gpu judge 拷贝合并）、阈值读配置、不准 import 训练代码的测试 | `code/s7_eval/judges/geometry/**`、`config/gauges/data_spec_v2.toml` | WP-04 | 1 天 | 在 parity 样本上和原两把尺的结果一致，差异在容差内 |
| WP-11 渲染入口 | triptych、grid12、encode（am2exp + rollout_tools 拷贝） | `code/s7_eval/render/**` | WP-04 | 5 h | 时间戳逐帧检查；12 帧网格含首帧和末帧 |
| WP-12 实时查看器适配 | LiveViewer + am_motion 场景（人体 \| 参考 \| 模型） | `code/s6_train/viz/live/**` | WP-04 | 1 天 | 加载一段，截图帧号和时间轴对得上 |
| WP-13 数据评估汇总 | 普查、`TAXONOMY_v1`、质量汇总、人审入队和读回、放行门 | `code/s4_dataset_eval/{census,taxonomy,quality,audit_queue}/**`、`docs/specs/TAXONOMY_v1.md` | WP-01、03 | 1 天 | 小时数守恒；过期判决标记正确 |
| WP-14 索引 | `amp index`（`INDEX.md` + lint） | `code/amp/index.py` | WP-01、03 | 3 h | 每个顶层目录都有一行，缺 MANIFEST 的能报出来 |
| WP-15 网页（Codex） | 按 OpenAPI 做页面（§B.8 的清单） | `code/web/frontend/**` | WP-09 冻结 OpenAPI | 3–5 天 | 浏览器验收截图；手机能看视频 |

**第 1–2 天的关键路径**：WP-00 → WP-01 →（WP-03、WP-04、WP-06 并行）→ WP-05 → 首批量产。WP-08、09、10、11、12、13、14 在第 3–5 天并行推进。WP-15 在 OpenAPI 冻结以后交给 Codex。

---

## 附：这个方案刻意不做的事

- 不重写任何引擎，不改引擎导出格式；GPU 引擎以后想换成原生 am_motion 导出，可以等格式稳定以后再议。
- 不做按内容哈希的对象存储，也不上 Postgres、Kubernetes 或消息队列。
- 不动旧训练缓存内部的软链结构，不重建 v0.2 时代的数据。
- 不通过 MCP 开放删除或任意 shell。
- 不在根分区上新增写入。
