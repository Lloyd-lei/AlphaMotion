# AlphaMotion 开发平台设计：catalog 优先方案

- 日期：2026-10-07。作者：架构子代理（catalog 优先视角）。状态：**草案，等 lloyd 拍板**。
- 依据：同目录下的四份地图（`map_research_repo.md`、`map_factory.md`、`map_caches_artifacts.md`、`map_model_tools.md`），以及这次只读核对过的几处现场：
  - `DATA_SPEC_v2.md`、`audit/SCHEMA.md`；
  - `umr_dataset` 的分片格式和实测体积；
  - `arena-datagrid` 的作业表结构；
  - `pretraining_foundation` 的索引表结构；
  - `motion_audit_web` 的 MCP 工具；
  - HiPHI 下载器的磁盘水位。
- 本文只是设计。除了写这一个文件，没有移动、删除或修改任何东西，也没用 GPU。

---

## 0. 一页看懂

**一句话**：建一个根目录 `~/arena_motion`。里面放一个 Python 包 `arena_motion`（命令叫 `am`）和一个数据库 `catalog/am.sqlite`，这个数据库给一切东西记账。七个环节各自是一组登记过的"作业类型"。网页和 MCP 都是薄薄一层：查这个数据库，或者往作业表里交一个作业。

**为什么要"先记账"**：现在乱，根子不在文件放哪，而在于"这东西是什么、谁产出的、谁在用、能不能删"只记在人脑和散落的 manifest 里。地图里查到的情况：
- 量尺有 8 把；
- 机器人资产有 5 份；
- SMPL-X 有 4 种存法；
- license 散在 3 处；
- 训练集靠跨分区软链拼起来；
- 6 个生产代码目录没有 git。

先把账建起来，搬家就只是改账本里的一个位置字段，代码不用跟着改。

**七条原则**

1. **一个根，一个包，一个账本，一个服务。** 根是 `~/arena_motion`，放在 /home 分区（剩 345G）；根分区只剩 76G，不再往里写新东西。
2. **每个产物在账本里都有一行**：内容哈希、产出它的作业和代码版本、输入是谁、license、保留级别，再加一句中文说明它是什么。
3. **七个环节全部是作业**。所有作业由一个调度器统一放行，它会照 `RESOURCE_LIMITS.md` 执行：只用 0-15 号核、同时只有 1 个重 GPU 任务、下载连接数合计 ≤ 16、遇到断网就暂停。
4. **数据和模型用同一套判官量。** 判官不 import 任何训练代码，阈值只存一份（DATA_SPEC v2）。
5. **结论必须带视频。** 每条评估判决都挂三联视频（人体源 | 对照臂 | 本臂）和 12 帧全程网格。
6. **先记账，后搬家。** 第一周一个大文件都不挪，旧路径全部照常能用。做过一致性测试（对拍）的引擎原样纳入版本管理，重跑对拍通过才算数。
7. **有几件事只有人能做**：批准 GPU 训练、批准删除、给模型加冕（Greenwich 等头衔）、写人工审计判决。代理只能提议。

**直接回答 lloyd 的问题**：
- `am what <任意旧路径或新路径>`：告诉你这是什么、谁产出的、哪些东西依赖它、现在什么状态、能不能删。
- 根目录的 `README.md` 每天由账本自动重写，一张表列出每个文件夹是干嘛的、产出了什么。

```
 原始数据 ─S1 清洗→ 人体动作(SMPL-X, Z 轴朝上, 单位米, 地面 z=0) ─┐
                                                                ├─S3 重定向(UMR GPU + v2 严格后处理)→ 机器人动作 motion_v3
 机器人 MJCF ─S2 表面采样→ 机器人资产(每台 4096 个外表面点 + 区域) ┘                  │
                                                     S4 整库评估(种类覆盖 / 质量 / 人工审计) ←┤
                                                     S5 存储(选片 → UMRS 训练分片 + 版本)  ←┘
                                                                  │
                                       S6 训练 → checkpoint → 实时查看器(人体源 | 参考 | 模型)
                                                                  │
                                       S7 评估(独立判官 / 冻结考卷 / 三联视频 / 排行榜)
   每个箭头 = 一种作业；每个方框 = catalog 里的一行；网页和 MCP 只读这些行、只提交作业
```

### 0.1 需要 lloyd 拍板的事（按急迫程度）

| # | 事项 | 建议 |
|---|---|---|
| 1 | `alphamotion-ardy-isaac` 服务开机 33 分钟崩溃重启了 374 次，它要的 `~/Desktop/GR00T-WholeBodyControl` 已经不在了 | 执行 `systemctl --user disable --now alphamotion-ardy-isaac` |
| 2 | 下载连接数超限：HiPHI 16 个 + GRAIL 48 个 = 64，上限是 16。另外有两个重 GPU 任务同时在跑 | HiPHI 的下载截止时间还剩约 36 小时，16 个连接先全给它；GRAIL 暂停下载或降到 0，等 HiPHI 截止后再开 |
| 3 | 根目录叫什么名字；研究主仓的名字里有空格、逗号、引号 | 根目录叫 `~/arena_motion`。主仓以后改名 `alphamotion_research`，旧路径留软链，3 个 systemd 服务还靠它启动 |
| 4 | 生产数据格式 | 只认 GPU 引擎导出的统一格式 `motion_v3`。CPU 工厂只当参考实现和量尺，它的输出不算 v2 数据（它保留了 UMR 自己的打滑，达不到"不滑"） |
| 5 | license：HiPHI 写明"商业产品的内部研发也算商用"、不许再分发；HiPHI 和 BONES 都不许训练能替代它们的动作生成模型 | 打包时默认"只做研究、不发布权重"。用 HiPHI 或 BONES 训的模型要发布，必须先过法务，账本里记一条 `legal_review` |
| 6 | 网页底座选哪个 | 用 `~/motion_audit_web`（已经有审计页面和 MCP 雏形），Codex 照 HTTP 接口文件继续写。arena-datagrid 的 Electron 前端不再接着做 |
| 7 | 删除候选，约 180G | 只出清单、等你逐项点头。最大的一项是 `/var/tmp/alphamotion/xwbc_full`，129G，在快满的根分区上，可以从 HF 重新下载 |
| 8 | GPU 训练和 WBC 闭环评测 | 默认都要批准，批准前先报时长和电费估算（沿用电费预算的纪律） |
| 9 | 路演、市场、股权、外联名单这些商业资料 | 挪出研究仓，放到 `~/arena_business/`，不进根目录 |

---

## A. 一个根目录

### A.1 目录树

所有路径都相对 `~/arena_motion/`，物理位置在 /home 分区。"git" 列写"是"的目录进同一个 git 仓库。

```
arena_motion/
├── README.md            自动生成的总索引（am index）；lloyd 只需要看这一个文件。人写的开头放在 docs/README.head.md
├── CLAUDE.md            给 Claude 和代理的规矩：只经 am 交作业；临时文件只放 scratch/；不许写死绝对路径
├── am.toml              唯一写路径的地方：卷表、资源上限、端口、环境位置、磁盘配额
├── pyproject.toml       一个包 arena_motion，命令 am
├── src/arena_motion/    唯一的 Python 包（模块见 A.1.1）
├── tests/               单元测试 / 契约测试 / 金样测试 / 对拍门
├── web/                 Codex 写的网页前端，只调 HTTP API（底座是 motion_audit_web/frontend）
├── engines/             冻结的、对拍过的引擎，原样纳入，只经适配器调用（git subtree）
│   ├── umr_cpu/              ← ~/umr_factory_code（CPU 参考实现 + 量尺）
│   ├── umr_gpu/              ← ~/umr_gpu（生产引擎）
│   ├── smplx_convert_gpu/    ← ~/gpu_convert（BVH → SMPL-X）
│   ├── umrs/                 ← ~/umr_dataset（分片容器 + 加载器）
│   └── greenwich_v02/        ← exp_topo 里的训练和模型代码（只读，用来复现 v0.2）
├── repos/               我们自己的其它 git 仓库（先用软链挂进来，以后再真搬）
│   ├── research  → ~/Desktop/"be water, robot"       （以后改名 alphamotion_research）
│   ├── product   → ~/Desktop/AlphaMotion             （产品仓 release/v0.2）
│   ├── ctrl      → ~/alphamotion_ctrl                 （mjlab WBC）
│   ├── datagrid  → ~/Desktop/arena-datagrid
│   └── contrib/mingrun（组员，只读）
├── third_party/<名字>@<commit>/   外部仓库，只读；本地改动放 third_party/patches/<名字>/*.patch
│                                   UMR、GMR、hiphi2smplx、mjlab、ufo、bfm_zero、beyondmimic、sonic、ardy、kimodo、GENMO、UniMate、GRAIL、holosoma…
├── specs/               规格和契约，改动要评审，进 git
│   ├── DATA_SPEC_v2.md + data_spec_v2.thresholds.json   ← 引擎后处理和判官读同一份阈值
│   ├── formats/   human_smplx_v1.md · motion_v3.md · umrs_v2.md · scene_bundle_v1.md
│   ├── taxonomy/  taxonomy_v1.yaml                       ← 动作分类法（唯一一份）
│   ├── eval/      standards/*.json · protocols/*.yaml · eval_sets/*.json（冻结清单 + sha）
│   ├── api/       openapi.v1.json · mcp_tools.v1.json · audit_contract_v1.md
│   └── ops/       RESOURCE_LIMITS.md
├── catalog/             唯一事实来源（不进 git，每晚快照）
│   ├── am.sqlite (+wal)
│   ├── parquet/         逐帧指标（数据量大，DuckDB 来读）
│   └── snapshots/       am-YYYYMMDD.sqlite.zst（另存一份到另一个分区）
├── store/               内容寻址的产物库（见 C.2）
│   ├── blobs/sha256/ab/cd/<hash>   单文件、不可变、会被多处引用的产物
│   ├── trees/<tree_id>.json        目录型产物的文件清单（内容哈希就是它的身份）
│   ├── open/<job_id>/              作业还在写的产物，写完后封存
│   └── trash/<plan_id>/            清理计划执行后的回收站，7 天后才真删
├── vol/                 卷挂载点：都是软链，指向原地不搬的大数据
│   ├── factory  → ~/umr_factory_data
│   ├── vartmp   → /var/tmp/alphamotion
│   ├── hometmp  → ~/tmp
│   ├── hf       → ~/.cache/huggingface
│   └── cold     → （以后接冷存储盘）
├── data/                给人看的数据目录：按环节编号。新产物是真实目录，旧产物是指向 vol/ 的软链
│   ├── s1_raw/<source>/                    原始下载（只读）+ LICENSE
│   ├── s1_human/<source>@<ver>/            统一后的人体动作（human_smplx_v1）
│   ├── s2_robots/<robot_id>/<asset_hash>/  机器人资产：MJCF、网格、4096 点表面、区域、脚底偏移、UMR 配置、对应关系
│   ├── s3_retarget/<engine>@<ver>/<source>/<clip>/<robot>.npz
│   ├── s4_dseval/<YYYY-MM-DD>/             整库评估报告、图、表
│   ├── s5_packs/<pack>@<ver>/              训练分片（UMRS）+ 选片条件 + 统计卡
│   ├── assets/                             人体模型（SMPLX_NEUTRAL、beta 先验、方言 Q）、关节名向量等
│   ├── wbc/                                给控制器的跟踪参考（ctrl_verify 轨迹、x2_pool、mjlab motions）
│   └── releases/<product>@<ver>/           发布物（weights_staging、HF 清单）
├── runs/<family>/<YYYY-MM-DD>_<slug>/      训练 run（第六环节）
├── evals/<YYYY-MM-DD>_<suite>_<subject>/   评估（第七环节）：判决表、三联视频、12 帧网格
├── docs/                人写的文档（进 git）
│   ├── reports/<YYYY-MM-DD>_<slug>/        报告（沿用 alphamotion video/resource/report 的编号）
│   ├── verdicts/        判决记录，只追加
│   ├── decisions/       架构决定（本文和四份地图搬到这里）
│   └── notes/ · papers/
├── services/            systemd 单元文件、服务状态（端口、pid、boot id）、日志（带轮转）
├── envs/                环境锁文件 + 指向真实环境的软链（conda 环境不能直接挪，见 E.4）
├── scratch/<YYYYMMDD>_<task>/   临时目录，14 天过期，代码不许引用这里的路径
└── archive/<YYYY-MM>_<slug>/    冻结的旧路线（只读，压缩包 + 一行说明 + 账本里一行）
```

**哪些进 git**：`README.md`、`CLAUDE.md`、`am.toml`、`pyproject.toml`、`src/`、`tests/`、`web/`、`engines/`（subtree）、`specs/`、`docs/`、`services/units/`。

**哪些不进 git**：`catalog/`、`store/`、`vol/`、`data/`、`runs/`、`evals/`、`envs/`、`scratch/`、`archive/`、`third_party/`（只在 `third_party/LOCK.toml` 里记 commit）。

#### A.1.1 包内模块（目录名和环节编号一一对应）

```
src/arena_motion/
├── core/         config.py（读 am.toml）· uri.py（am:// 路径解析）· catalog.py + schema/*.sql（账本）
│                 cas.py（产物库）· manifest.py · jobs.py（作业表 + 租约）· governor.py（资源上限）
│                 approvals.py · queries.py（所有只读查询，HTTP 和 MCP 共用）
├── formats/      所有文件格式的唯一读写口：human_smplx_v1 · motion_v3（兼容读 v1/v2）· umrs · scene_bundle_v1
│                 规矩：别处不许自己 np.load 业务文件
├── s1_ingest/    sources/{hiphi,grail,omnicontact,bones_seed,amass_xwbc,lafan,mixamo,uniml3d,babel}.py
│                 gauge.py · license.py · download.py · importers/
├── s2_surface/   registry.py · stage.py · surface.py · regions.py · human_surface.py · corr.py · sole_offset.py
├── s3_retarget/  plan.py · adapters/{umr_cpu,umr_gpu}.py · strict.py（调引擎）· export.py · parity.py
├── s4_dseval/    taxonomy.py · coverage.py · quality.py · audit.py · report.py
├── s5_store/     register.py（旧目录登记）· seal.py · verify.py · pack.py · budget.py · gc.py · move.py
├── s6_train/     runs.py · launch.py · reporter.py（训练代码回调）· watch.py · models.py · importers/
├── s7_eval/      judges/{geom,loco,manip,fid,codec,wbc,latent}/ · suites.py · protocol.py（统计 + decide）
│                 eval_sets.py · leaderboard.py
├── viz/          bundle.py（scene_bundle_v1）· live.py（常驻 viser 查看器）
│                 render/{triptych,grid12,encode,scene3d,showcase}.py
├── service/      http.py（FastAPI）· mcp.py（MCP）· auth.py（人 / 代理两种角色）· files.py（视频按字节范围下发）
├── legacy/       refscan.py（找写死的旧路径）· maps_import.py（把四份地图导进账本）
└── cli.py        自动发现各模块的 cli.py，注册成 am 的子命令
```

### A.2 大数据放哪、怎么引用

**卷（volume）**：`am.toml` 里列出所有物理位置，账本只记"哪个卷 + 卷内相对路径"。

```toml
[root]
path = "/home/arenalabs/arena_motion"

[volumes]
home    = { path = "/home/arenalabs/arena_motion", fs = "/home", tier = "hot" }
factory = { path = "/home/arenalabs/umr_factory_data", fs = "/home", tier = "hot",  legacy = true }
vartmp  = { path = "/var/tmp/alphamotion",            fs = "/",     tier = "warm", legacy = true, read_only_new = true }
hometmp = { path = "/home/arenalabs/tmp",             fs = "/home", tier = "warm", legacy = true }
hf      = { path = "/home/arenalabs/.cache/huggingface", fs = "/home", tier = "cache" }
research= { path = '/home/arenalabs/Desktop/"be water, robot"', fs = "/home", tier = "hot", legacy = true }

[limits]            # 照抄 RESOURCE_LIMITS.md，调度器读这里
cpu_set = "0-15"; nice = 10; omp_threads = 2; max_cpu_heavy = 6; max_gpu_heavy = 1
gpu_util_target = 70; gpu_watt_target = 350; max_net_conns = 16
min_avail_mem_gb = 15; max_tree_mem_gb = 20; net_state_file = "vol:factory/logs/throttle.log"

[disk]
home_min_free_gb = 150; root_min_free_gb = 60

[services]
http = "127.0.0.1:8780"; viewer = "127.0.0.1:8781"; idle_ttl_h = 8
```

**三种引用写法**（代码里只许用这三种，CI 会拦 `/home/arenalabs`、`/var/tmp`、`~/tmp`、`Downloads` 这类字面路径）：

| 写法 | 例子 | 解析 |
|---|---|---|
| 卷路径 | `am://vartmp/topo_exp/cache_lafan_v3` | 卷表 + 相对路径 → 绝对路径 |
| 内容身份 | `am:art/sha256:9f2c…` 或 `am:art/tree:41ab…` | 先查账本拿到当前物理位置，读之前可选校验哈希 |
| 业务 ID | `am:clip/hiphi/xxx`、`am:robot/unitree_h2@3fa2c1d0`、`am:pack/core21_hoi@v1` | 先查账本找到对应产物，再按上一行解析 |

**搬家**：`am store move <art> --to <卷>` 分五步：
1. rsync，带 `nice`/`ionice`、限速；
2. 校验哈希；
3. 改账本里的位置；
4. 旧位置留一个反向软链；
5. 30 天后旧位置进清理计划（要批准）。

跨分区搬家算重活，必须批准。

**大数据默认不搬**：
- `/var/tmp/alphamotion/topo_exp` 的 140G 缓存、`~/tmp/cl_v3` 的 31G 标签、HiPHI 的 101G 解压文件，都在原地登记成 `legacy` 卷里的产物。
- `data/` 下面用软链露出来给人看。
- 只有新产物才写到 /home 的根目录里。

### A.3 命名和版本规矩

**文件夹和 ID 一律**：小写 ASCII、下划线分隔、不带空格、逗号、引号（教训来自 `"be water, robot"` 和它的孪生目录）。日期一律写成 `YYYY-MM-DD`。

| 东西 | ID 格式 | 例子 | 规则 |
|---|---|---|---|
| 数据源 | `<source>` | `hiphi` `grail` `omnicontact` `bones_seed` `amass_xwbc` `lafan` `mixamo` `uniml3d` `babel` | 固定词表，加新源要改 `specs/` |
| 片段 | `<source>/<源内原始路径去扩展名>` | `grail/terrain_curbs__curb_000__000` | 永远不变；镜像版加后缀 `~mirror`，默认不进训练 |
| 人体数据版本 | `human:<source>@<fps>.v<N>` | `human:hiphi@fps30.v1` | 转换器或 gauge 一变，N 就加 1 |
| 机器人 | `<厂商>_<型号>[_<变体>]@<资产哈希前 8 位>` | `unitree_g1_29dof@3fa2c1d0` | 沿用 `robots.json` 的 34 个名字；MJCF、网格、表面点、staging 任何一个变了，哈希就变 |
| 引擎 | `<engine>@<git 短哈希>` | `umr_gpu@a1b2c3d` | 有改动没提交的代码不许跑生产作业 |
| 规格 / 格式 | `<名字>_v<主版本>` | `DATA_SPEC_v2`、`motion_v3`、`umrs_v2` | 读的代码要兼容旧版本；不兼容就升主版本 |
| 训练包 | `<名字>@v<N>` | `core21_hoi@v1` | 选片条件冻结成 yaml，内容哈希写进账本 |
| 训练 run | `<YYYY-MM-DD>_<family>_<slug>` | `2026-10-09_am2_B_M_seed0` | family 取 `am2`、`greenwich`、`equator`、`wbc_mjlab`、`wbc_ufo`… |
| checkpoint | `<run_id>/step_<7 位>` | `…/step_0006000` | |
| 模型 | `<family>-<size>-<ver>` | `greenwich-v2-L20k` | **头衔**（Greenwich、Equator、Bumblebee…）是单独的 `title` 字段，只能由 lloyd 用 `am model crown` 写入 |
| 控制器对照臂 | `alphamotion_policy` / `native_policy` | | 沿用已有约定：参考动作走 Greenwich 的叫 AlphaMotion policy；GMR 原生重定向的叫 Native policy |
| 冻结考卷 | `<名字>@v<N>` | `golden_core@v1` | 成员清单 + sha 冻结后不再改；要改就出 v2 |
| 评估 | `<YYYY-MM-DD>_<suite>@<ver>_<subject>` | `2026-10-12_locomotion@v1_am2-B-M-s6000` | |

**版本铁律**

- 产物封存后不再改写。要修就出新版本，账本里记一条 `supersedes`（新版本取代旧版本）的关系。
- "最新"只是别名：`aliases` 表里写 `pack:core21_hoi@latest → v3`，由人或作业显式切换。
- 每个托管目录都带两个文件：
  - `MANIFEST.json`：ID、类型、版本、产出它的作业、代码的 repo 和 commit、输入 ID 列表、参数哈希、内容哈希、license、规格版本、一句中文说明；
  - `README.md`：由 MANIFEST 生成，一段话说明这个目录。

### A.4 根目录 README 总索引（自动生成）

`am index` 每天 07:00 跑一次，账本有大变动时也跑一次。它重写根目录 `README.md` 和每个托管目录的 `README.md`。人只改 `docs/README.head.md`。样子如下：

```markdown
# arena_motion 总索引（自动生成于 2026-10-09 07:00，别手改）

## 现在在跑
| 作业 | 环节 | 进度 | 预计结束 | 资源 | 谁提交的 |
| ingest.download hiphi part 0028 | S1 | 41% | 10-08 02:10 | 网络 16 连接 | lloyd |

## 磁盘
| 分区 | 剩余 | 底线 | 新产物配额已用 |
| /home | 331G | 150G | retarget 12/150G · packs 3/100G · runs 0/150G |
| /     | 76G  | 60G  | （冻结，不写新东西） |

## 顶层目录：每个干什么
| 目录 | 干什么 | 现有内容 | 最近一次产出 |
| data/s1_human | 统一后的人体动作（Z 轴朝上、单位米、地面 z=0） | 4 个源 · 37,880 段 · 126.4 h | 10-08 ingest.convert hiphi |
| …

## data/ 每个数据集一行
| ID | 是什么 | 大小 | 谁产出（作业 / commit） | 日期 | 状态 | license | 被谁用 |

## 还在旧位置的（迁移账本）
| 旧路径 | 登记成 | 状态 | 计划 |
| ~/tmp/cl_v3 | labels:contact@v3 | 原地登记（承重） | P6 再评估 |

## 没登记的新目录（防止再乱）
| 路径 | 发现时间 | 大小 | 建议 |

## 待你处理
- 3 个批准请求 · 2 个删除计划 · 5 条新的严重问题
```

另外两条命令兜底：
- `am what <路径>`：对任何路径都能说出它是什么。没登记的路径会提示"未登记"，并给出最近的上级产物。
- `am doctor`：查坏链、未登记目录、过期临时目录、配额。

---

## B. 七个环节

**统一写法**：每个环节 = 一个子模块 + 若干作业类型 + 若干账本表 + CLI 子命令 + MCP 工具 + HTTP 端点 + 测试门。

所有作业类型和资源级别见附录 1。环节之间只通过账本里的产物 ID 交接，不通过"约定好的目录"交接。

### B.1 S1 数据清洗（来源、规范化、license）

**目的**：把各个源的原始动作变成同一种人体格式，统一以下几项：
- Z 轴朝上；
- 单位米；
- 按会话校准地面，地面 z=0，原始地面值另存；
- fps 按源原样保存，另出一份 30 fps；
- 去掉镜像；
- 每段都带 license。

| 项 | 内容 |
|---|---|
| 模块 | `s1_ingest/sources/<源>.py` 每个源一个适配器，统一四个函数：`discover()`（扫描已有文件）、`register()`（登记，不复制）、`to_human()`（转成 human_smplx_v1，调引擎）、`gauge()`（检查和修正规范）。另有 `license.py`、`download.py` |
| 作业 | `ingest.register` · `ingest.download` · `ingest.convert` · `ingest.gauge` · `ingest.human_qa` |
| 输入 → 输出 | 原始文件（BVH / pkl / npy / npz）→ `data/s1_human/<源>@<版本>/<clip>.npz`（human_smplx_v1）+ 每段 QA（地面残差、物体穿透、脚滑）|
| 统一人体格式 `human_smplx_v1` | `transl[T,3]`（米）、`global_orient[T,3]`、`body_pose[T,63]`、手和下巴（可选）、`betas`、`gender`、`fps`、`frame_ids`（对应源帧号）、`floor_z_removed`、`up="+z"`、`objects[{object_id, pose[T,7]}]`、`source_ref{path, sha256}`、`license_id` |
| 先适配、后重转 | 现在有 4 种存法：HiPHI 的 `motion_actor_smplx.npz`、Omni 按字段分开的 `.npy`、GRAIL 的 `recon/*.pkl`、LAFAN 放在 UMR 的 `sample_data` 里。先给它们写**读取适配器**，登记进账本就能用，不重新转换。打包器和网页都改成走 `formats.human.read(clip_id)`，这样 `umr_dataset/sources.py` 找错 HiPHI 路径的问题也就没了 |
| HiPHI 重复 | CPU 转了 1,094 段，GPU 转了 3,796 段。账本按规则只选一份：有 GPU 版就用 GPU 版，只有 CPU 版的另补一个 30 fps 作业。"有没有 `qa.json`"这种互相跳过的暗号作废 |
| license | `licenses` 表的种子来自三处：`/var/tmp/.../sources/sources_inventory.json`、各 `raw/*/LICENSE.md`、GRAIL manifest 里逐段的 license。每段默认继承源的 license，GRAIL 逐段覆盖。字段有：能不能商用（`no` / `yes` / `internal_rnd_is_commercial` / `unclear`）、能不能再分发、对模型的限制（原文）、署名要求、法务审查状态。打包、导出和模型卡都自动带上 license 汇总 |
| 下载 | 现有的 HiPHI 下载器和 GRAIL 循环先登记成 `external` 作业（只记账，不重启）。调度器按它们占的连接数算总账。新的下载一律走 `ingest.download`，连接数由调度器分，遇到 `net=down` 自动暂停 |
| 账本表 | `sources` · `licenses` · `clips` · `clip_stages`（raw / human@fps90 / human@fps30 / gauge / human_surface）· `objects` · `clip_objects` · `artifacts` · `issues` |
| CLI | `am ingest register hiphi`（只扫描登记）· `am ingest convert hiphi --engine smplx_convert_gpu --fps 30 --limit 500` · `am ingest gauge grail --check` · `am ingest status` · `am license show hiphi` · `am license check --pack core21_hoi@v1` |
| MCP | `list_sources` · `source_card` · `ingest_status` · `license_check` · `submit_job(kind="ingest.*")` |
| HTTP | `GET /api/v1/sources` · `/sources/{id}` · `/clips?source=&stage=&family=` · `/clips/{clip_id}` · `/licenses` · `POST /api/v1/jobs` |
| 测试门 | ① 每个源 3 段金样：Z 轴朝上、单位米、fps 对、静止站立帧双脚最低点离地 ≤ 2 cm；② AMASS（xwbc）时间轴：KIT 和 TotalCapture 两种机器人流按源 fps 重采样后，和人体流误差 < 1 帧（专防已知的 1.2 倍 / 2 倍错位）；③ Y-up 转 Z-up 用 (z,x,y) 置换，有专门测试，防止转偏 90°；④ GRAIL 逐段 license 能覆盖源级 license；⑤ `register` 连跑两遍不产生重复行；⑥ GPU 转换器 commit 一变，自动重跑 `parity40` 对拍 |
| 现有代码来源 | `umr_factory_data/scripts*`、`scripts_grail`、`scripts_streamc`、`gpu_convert`、`hiphi2smplx`、`beta_fit`、`h2s_shim`、`pretraining_foundation/acquire.py`、`data/action_labels.py` |

### B.2 S2 表面点云采样（机器人登记 + 4096 点 + 区域）

**目的**：每台机器人有且只有一份权威资产。表面点照 UMR 的做法：每台 4096 个外表面点，用射线可见性筛出外表面再做最远点采样；点绑在连杆上，带法线、几何体、身体部位和区域标签。逐帧坐标由正向运动学算出来，不逐帧存。

| 项 | 内容 |
|---|---|
| 模块 | `s2_surface/registry.py`（从 `robots.json` 导入 34 台，其中 21 台是核心）· `stage.py`（T 姿态、碰撞排除对）· `surface.py`（4096 点）· `regions.py` · `sole_offset.py` · `corr.py`（人到机器人的对应关系）· `human_surface.py`（SMPL-X 的 4096 个槽位 + 分区表）|
| 作业 | `robot.import` · `robot.stage` · `robot.surface` · `robot.sole_offset` · `robot.corr` · `robot.contact_sheet` |
| 输出：每台一个资产目录 | `data/s2_robots/<robot_id>/<asset_hash>/` 里有：`mjcf/`（展平后的 MJCF）、`meshes/`、`staging.json`、`surface.npz`（`points_local`、`normals_local`、`geom`、`body`、`region`）、`regions.json`、`sole_offset.json`、`umr_config.json`、`corr/{smplx,beta0}/`、`SOURCE.md`（模型来自 menagerie、unitree_ros、GMR 还是 sonic，以及它的 license）|
| 5 份副本合成 1 份 | 权威 = 工厂的 `robots/`（21 台）。`umr_gpu/robots` 和它逐字节相同，改成引用同一个哈希。`umr_dataset` 的 `kin.npz` 登记成派生产物。`am2exp/assets/surface_v2` 是另一套绑定，归档。`/var/tmp` 里的 `robots.json` 导进账本 |
| 对应关系不一致 | 工厂的 `corr/` 和 GPU 引擎自己那份字节数不同。先跑一个只读作业 `robot.corr_diff`，比较点索引和权重在语义上是否一致，然后才定用哪份。定之前两份都登记，每条重定向结果记下自己用的是哪份 |
| 脚底偏移表 | `~/tmp/sole_offsets3.json` 有 8 台偏大 4.7–10 cm，而且缺 H2。改由作业 `robot.sole_offset` 从 MJCF 碰撞几何重新生成，每台一个带版本的值 |
| 影响追踪 | 资产哈希一变，`am robot impact <robot>` 列出所有受影响的重定向结果、训练包和评估，并把它们标成 `stale`（过期）|
| 账本表 | `robots` · `robot_assets` · `artifacts` · `lineage` |
| CLI | `am robot list --core` · `am robot show unitree_h2` · `am surface build unitree_h2` · `am surface check --all` · `am robot impact unitree_h2` |
| MCP | `list_robots` · `robot_card` · `surface_stats` · `robot_impact` |
| HTTP | `GET /robots` · `/robots/{id}` · `/robots/{id}/surface.bin`（给三维视图用的二进制点）· `/robots/{id}/mjcf` · `/robots/{id}/contact_sheet.png` |
| 测试门 | ① 每台正好 4096 点；② 每个点到碰撞几何表面的距离 ≤ 0.5 mm；③ 没有点藏在别的几何体里面（可见性检查）；④ 必需的区域（左右脚底、脚跟、脚尖、手掌、手指、膝、肘、躯干、骨盆）每个点数都 ≥ 下限；⑤ 法线是单位长度、朝外；⑥ 重跑一遍哈希不变；⑦ T 姿态下脚底最低点和 `sole_offset` 相差 ≤ 1 mm；⑧ 21 台核心机器人全部通过以上各项，才允许进入 S3 |
| 现有代码来源 | `umrf/robots.py`、`surface.py`、`stage_robots.py`、`build_surfaces.py`、`build_correspondence.py`、`build_shared_corr.py`、`/var/tmp/.../robots/{inventory,finalize}.py` |

### B.3 S3 重定向（UMR GPU + v2 严格后处理 + 统一导出）

**目的**：把人体动作批量转成 21 台机器人的动作，并满足 DATA_SPEC v2 的四条要求：
- 实接触：接触点离地或离物体在 0 到 1 mm 之间；
- 不穿：穿透 ≤ 0.5 mm；
- 不滑：接触点切向速度 < 1 cm/s；
- 不闪：接触段至少 3 帧，进入阈值 1 mm、离开阈值 2 mm。

做不到的帧要剔除，并报出精确数量。

| 项 | 内容 |
|---|---|
| 引擎分工 | `engines/umr_gpu` 是生产引擎（和 CPU 对拍过：关节误差中位数约 1e-5 rad；吞吐 2,707 机器人帧/秒，4 台、B=256），后处理用它自带的 v2 严格处理。`engines/umr_cpu` 是参考实现和量尺，保留了 UMR 原生的打滑（`LATCH_PIN=0`），**不算 v2 数据** |
| 统一格式 `motion_v3` | 取代 CPU 的 `factory_npz_v1` 和 GPU 的 `gpu_npz_v2`，每个（片段，机器人）一个 npz，放在 `data/s3_retarget/<engine>@<commit>/<source>/<clip>/<robot>.npz`。<br>• 单位：一律米；只有量化字段用毫米，字段名带 `_mm` 后缀。<br>• `meta`（json）：各种 ID、规格版本、阈值哈希、引擎 commit、机器人资产哈希、对应关系 ID、人体产物 ID、`floor_z`、`fps`、`frame_ids`。<br>• 运动：`qpos[T,nq]` f32、`joint_names`、`root_pos`、`root_quat`。<br>• 点级接触（CSR 稀疏存法）：`contact_ptr[T+1]`、`contact_idx`、`contact_target`（0 = 地面，k = 物体）。<br>• 近场距离（CSR）：离接触面 < 2 cm 的点，`near_ptr`、`near_idx`、`near_dist_q`（uint8 对数量化）。<br>• 区域标签：`region_bits[T,4]`，uint16，四种分别是地面接触、物体接触、打滑、穿透。<br>• 穿透深度、`flags[T]`、`excluded[T]`、物体位姿、`qc`（汇总）|
| 兼容读 | `formats.motion.read()` 能读 v1 / v2 / v3，但只写 v3。`umr_dataset` 和审计网页都改成走它 |
| 已知 bug | `umrf/objects.py` 把 8 顶点的箱子判成外接球，所有搬箱片段的物体实接触都是 0%。在 `issues` 里登记，并设 `blocks_packing=1`：CPU 出的搬箱片段一律不进训练包。GPU 引擎 `geometry.py` 有没有同样的问题还没核实，先补一个"立方体距离"单元测试 |
| 阈值只存一份 | 新建 `specs/data_spec_v2.thresholds.json`（1 mm / 0.5 mm / 1 cm/s / 3 帧 / 1 mm 进 2 mm 出 / 人体离地 2 cm 时机器人要离开 5 mm），引擎后处理和所有判官都读这一份。CI 检查代码里不许再出现重复的常数（现在 provenance 里写的是 1 cm / 2 cm / 3 cm/s，和规格对不上）|
| 作业 | `retarget.plan`（选出某引擎版本下还没做的片段×机器人）· `retarget.capture`（CPU 抓问题）· `retarget.solve`（GPU）· `retarget.export`（写 motion_v3）· `judge.run`（独立判官复核）· `retarget.parity` |
| 调度 | `retarget.solve` 属于重 GPU 任务，同一时间只能有一个。批大小从 64 往上加，GPU 利用率保持在 70% 以下。一个计划估算超过 4 小时就要批准 |
| 账本表 | `engines` · `retargets` · `qa_metrics` · `issues` · `artifacts` · `lineage` |
| CLI | `am retarget plan --source hiphi --robots core21 --engine umr_gpu` · `am retarget run <plan_id>` · `am retarget parity --cases parity40` · `am retarget show <clip> <robot>` · `am retarget status` |
| MCP | `retarget_status` · `retarget_inspect(clip, robot)`（返回指标和预览链接）· `submit_job` |
| HTTP | `GET /retargets?clip=&robot=&engine=&status=` · `/retargets/{id}` · `/retargets/{id}/frames?start=&count=`（分块二进制，给三维视图）· `/retargets/{id}/qa` |
| 测试门 | ① `parity40`：GPU 和 CPU 的关节误差 p50 ≤ 2e-5 rad，`parity_tables.md` 里的判官表在容差内不变；② v1 和 v2 转成 v3 后再读回来，在容差内无损；③ 严格性：用独立判官重量导出的文件，保留下来的帧必须全部过阈值，剔除帧数和文件里记的一模一样；④ 立方体物体单元测试，两个引擎都要过；⑤ 同样的输入和 commit，结果哈希相同（GPU 有不确定性时，在浮点容差内相同）；⑥ 格式校验器 |

### B.4 S4 整库评估（种类够不够、质量过没过、人看过没有）

**目的**：对整个预训练数据回答四个问题：有多少、种类全不全、质量过不过关、人抽查过没有。不是只看小时数。

| 项 | 内容 |
|---|---|
| 分类法 | 合并三套：`action_labels.py` 的 12 类正则、diversity 普查用的类别、BABEL 和文本标注，统一成 `specs/taxonomy/taxonomy_v1.yaml`。每段只有 1 个主类，另外可以打多个标签，记下标签来源（文本正则 / BABEL / 模型 / 人）和置信度。"其他"类的占比本身也是一个指标（AMASS 曾经有 77% 落到"其他"）|
| 种类覆盖 | 按 主类 × 源 × 环节状态 × 机器人 统计小时数。再和目标配额对比，列出稀缺清单：现在躺、爬、翻滚、摔倒起身、上下楼梯、重物这几类都约为 0 h。还要看动作池子的熵、演员数、接触模式（哪些区域碰地或碰物体）、物体种类，以及姿态空间的占用率（按姿态聚类后看格子占了多少）|
| 质量汇总 | 把 `qa_metrics` 按 源 / 机器人 / 主类 汇总成分布：穿透 p99、实接触率、打滑 p95、和人一致率、剔除帧比例。阈值统一取 B.3 那份 |
| 人工审计 | 抽样分两种：一种按风险从高到低挑（自动质检越差越靠前），一种分层随机抽。合格率的置信区间**只用随机抽样、且哈希还对得上的判决**来算。判决只能由人写（`reviewer_kind='human'`）；代理只能写观察笔记。`audit/SCHEMA.md` 里的 jsonl 契约继续保留，作为导入导出格式 |
| 报告 | 用 `am dseval report` 生成中文 `REPORT.md` 和图，放到 `data/s4_dseval/<日期>/`，登记成一个产物，同时记下它用的是账本哪一天的快照，保证能复现 |
| 作业 | `dseval.label` · `dseval.coverage` · `dseval.quality` · `audit.sample` · `dseval.report` |
| 账本表 | `clip_labels` · `qa_metrics` · `audit_items` · `audit_decisions` · `issues` · `artifacts`；逐帧指标放在 `catalog/parquet/` |
| CLI | `am dseval label --taxonomy v1` · `am dseval coverage --by family,source` · `am dseval quality --robot unitree_g1_29dof` · `am audit sample --n 200 --random-frac 0.3` · `am dseval report` |
| MCP | `dataset_overview` · `diversity(by, filters)` · `quality_summary(filters)` · `find_clips(text | family | attrs)` · `get_clip` · `audit_queue` · `audit_stats` · `list_issues` · `add_note` |
| HTTP | `GET /dseval/coverage` · `/dseval/quality` · `/dseval/reports` · `/audit/queue` · `POST /audit/decisions`（只限人）· `POST /audit/issues` |
| 测试门 | ① 小时数守恒：按类加总 = 按源加总 = 总数；② 每段正好 1 个主类；③ 置信区间不含非随机样本和哈希过期的判决；④ 同一个快照重新生成报告，数字完全一样 |
| 现有代码来源 | `umr_factory_data/diversity/*`（今天已经出了 `REPORT.md`）、`motion_audit_web`、`audit/SCHEMA.md`、`analysis/boneseed_locomotion_index`、`training_recipe_audit` |

### B.5 S5 存储（账本、产物库、打包、预算）

账本、产物库、分片格式、预算和清理的细节见 C 节。这里只列入口。

| 项 | 内容 |
|---|---|
| 作业 | `store.register`（旧目录原地登记）· `store.seal` · `store.verify` · `store.snapshot` · `store.pack` · `store.gc_plan` · `store.gc_execute`（要批准）· `store.move`（跨分区要批准）· `store.audit_layout`（每天扫没登记的新目录）|
| CLI | `am store register <路径> --kind --name --ver` · `am store du` · `am store verify --sample 1%` · `am pack build specs/packs/core21_hoi.yaml` · `am pack verify core21_hoi@v1` · `am store gc --plan` · `am store gc --execute <plan> --approval <id>` · `am what <路径>` · `am doctor` |
| MCP | `storage_status` · `artifact_info` · `lineage(id, up|down)` · `what_is(path_or_id)` · `list_packs` · `pack_card` · `gc_plan`（只读；只有 lloyd 能在网页上执行）|
| HTTP | `GET /store/volumes` · `/artifacts/{id}` · `/artifacts/{id}/lineage` · `/what?path=` · `/packs` · `/packs/{id}` · `/gc/plans` · `POST /approvals/{id}`（只限人）|
| 测试门 | ① 同一文件放进产物库两次不产生第二份，且哈希一致；② 改一个字节，`verify` 能发现；③ 打包往返误差：沿用 `umr_dataset` 现有的 `roundtrip_7entries` 和 `strict_recheck`；④ 清理计划里绝不出现 `pinned` 产物或还有人引用的产物；⑤ 配额超线时，调度器拒绝放行会产出大文件的作业 |

### B.6 S6 训练和实时可视化

见 D.1、D.2。CLI `am train …` / `am viz …`；MCP `list_runs` · `run_card` · `viewer_load`；HTTP `/runs` · `/scenes/{id}` · `/viewer/load`。

### B.7 S7 模型重定向和生成动作的评估

见 D.3。CLI `am eval …` / `am render …`；MCP `run_eval` · `eval_result` · `leaderboard` · `get_media`；HTTP `/evals` · `/evals/leaderboard` · `/media/{id}`。

---

## C. 存储设计

### C.1 账本（catalog）

**为什么用 SQLite 记账、用 DuckDB 做统计**

- 作业 worker 是多个进程，要同时写账本。SQLite 开了 WAL 以后支持多进程并发，配合 `BEGIN IMMEDIATE` 短事务，就能做作业租约（arena-datagrid 已经这么用了）。
- DuckDB 同一时间只允许一个进程读写，不适合当主账本。但它统计大量逐帧数据很快，所以用它通过 sqlite 扩展读账本，再读 `catalog/parquet/` 里的逐帧指标。

**规模估计**：
- 片段约 13 万：HiPHI 1.8 万、GRAIL 2.2 万、BONES 7.1 万、AMASS 1.3 万……
- 重定向结果约 270 万行（× 21 台机器人）；
- `qa_metrics` 行数和重定向结果差不多。

SQLite 对付这个量级没有问题。逐帧序列放 parquet，不进 SQLite。

**核心表**（`src/arena_motion/core/schema/0001_init.sql`，下面是节选，列名就是契约）：

```sql
-- 0. 卷、产物、血统
CREATE TABLE volumes(vol_id TEXT PRIMARY KEY, path TEXT NOT NULL, fs_mount TEXT,
  tier TEXT CHECK(tier IN('hot','warm','cold','cache')), legacy INTEGER DEFAULT 0, note_zh TEXT);
CREATE TABLE artifacts(
  art_id TEXT PRIMARY KEY,                 -- 'sha256:…'（文件）| 'tree:…'（目录清单哈希）| 'ext:…'（旧大目录，抽样指纹）
  kind TEXT NOT NULL,                      -- raw|human_motion|robot_asset|surface|corr|retarget_motion|qa_series|pack|shard|
                                           -- checkpoint|run_dir|eval_result|media|report|spec|label|legacy_dir|env
  name TEXT, version TEXT,                 -- 'hiphi' + 'fps30.v1'
  vol_id TEXT REFERENCES volumes, relpath TEXT,   -- 物理位置：搬家只改这两列
  bytes INTEGER, nfiles INTEGER,
  hash_strength TEXT CHECK(hash_strength IN('full','sampled','none')),
  format TEXT, format_version TEXT, spec_version TEXT,
  produced_by_job TEXT, code_repo TEXT, code_commit TEXT,
  status TEXT CHECK(status IN('open','sealed','stale','superseded','missing','corrupt','delete_candidate','trashed')),
  retention TEXT CHECK(retention IN('pinned','canonical','regenerable','scratch','archive')),
  license_rollup TEXT, summary_zh TEXT,    -- 一句中文：这是什么（README 用）
  created_at TEXT, sealed_at TEXT, last_verified_at TEXT);
CREATE INDEX art_kind_idx ON artifacts(kind, name, version);
CREATE TABLE lineage(child TEXT NOT NULL, parent TEXT NOT NULL, role TEXT NOT NULL,   -- input|derived_from|supersedes|uses_asset
  PRIMARY KEY(child, parent, role));
CREATE TABLE aliases(alias TEXT PRIMARY KEY, art_id TEXT NOT NULL, set_by TEXT, set_at TEXT);

-- 1. 来源、license、片段
CREATE TABLE licenses(license_id TEXT PRIMARY KEY, name TEXT,
  commercial TEXT CHECK(commercial IN('no','yes','internal_rnd_is_commercial','unclear')),
  redistribution INTEGER, model_restriction TEXT, attribution TEXT, text_art TEXT,
  legal_review TEXT DEFAULT 'pending');
CREATE TABLE sources(source_id TEXT PRIMARY KEY, title TEXT, revision TEXT, url TEXT,
  license_id TEXT REFERENCES licenses, native_fps REAL, hours_known REAL, access_deadline TEXT, note_zh TEXT);
CREATE TABLE clips(clip_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources, native_ref TEXT,
  raw_art TEXT, frames INTEGER, fps REAL, seconds REAL, actor_id TEXT, gender TEXT, height_m REAL,
  has_object INTEGER, has_terrain INTEGER, is_mirror INTEGER DEFAULT 0,
  license_id TEXT REFERENCES licenses,     -- 逐段覆盖（GRAIL）
  split_group TEXT,                        -- 同演员/同会话一组，防止训练集和考卷泄漏
  split TEXT CHECK(split IN('train','val','test','holdout')), text TEXT);
CREATE TABLE clip_stages(clip_id TEXT NOT NULL, stage TEXT NOT NULL, variant TEXT NOT NULL DEFAULT '',
  status TEXT CHECK(status IN('todo','running','ok','failed','excluded','stale')),
  art_id TEXT, job_id TEXT, metrics TEXT, updated_at TEXT, PRIMARY KEY(clip_id, stage, variant));
CREATE TABLE objects(object_id TEXT PRIMARY KEY, source_id TEXT, kind TEXT, mesh_digest TEXT,
  collision_art TEXT, license_id TEXT);
CREATE TABLE clip_objects(clip_id TEXT, object_id TEXT, PRIMARY KEY(clip_id, object_id));

-- 2. 机器人
CREATE TABLE robots(robot_id TEXT PRIMARY KEY, vendor TEXT, model TEXT, nq INTEGER, height_m REAL,
  tier TEXT CHECK(tier IN('core21','extra','retired')), embodiment_split TEXT CHECK(embodiment_split IN('seen','heldout')),
  mjcf_source TEXT, license TEXT, note_zh TEXT);
CREATE TABLE robot_assets(robot_id TEXT NOT NULL, asset_art TEXT NOT NULL, surface_art TEXT, corr_art TEXT,
  sole_offset_m REAL, umr_config_art TEXT, n_points INTEGER CHECK(n_points = 4096), region_counts TEXT,
  is_current INTEGER DEFAULT 0, PRIMARY KEY(robot_id, asset_art));

-- 3. 引擎和重定向
CREATE TABLE engines(engine_id TEXT PRIMARY KEY, family TEXT, code_commit TEXT,
  parity_status TEXT CHECK(parity_status IN('unverified','verified','failed')), parity_report_art TEXT,
  spec_version TEXT, thresholds_hash TEXT, v2_compliant INTEGER);
CREATE TABLE retargets(retarget_id TEXT PRIMARY KEY, clip_id TEXT NOT NULL, robot_id TEXT NOT NULL,
  engine_id TEXT NOT NULL, robot_asset_art TEXT NOT NULL, corr_art TEXT, human_art TEXT NOT NULL,
  art_id TEXT, format TEXT, frames INTEGER, frames_excluded INTEGER, status TEXT, job_id TEXT, created_at TEXT,
  UNIQUE(clip_id, robot_id, engine_id, robot_asset_art, human_art));

-- 4. 质量（数据和模型共用一张表）
CREATE TABLE judges(judge_id TEXT PRIMARY KEY, family TEXT, code_commit TEXT, thresholds_hash TEXT,
  spec_ref TEXT, imports_training_code INTEGER NOT NULL CHECK(imports_training_code = 0));
CREATE TABLE qa_metrics(subject_kind TEXT CHECK(subject_kind IN('human','retarget','model_output','sim_rollout')),
  subject_id TEXT NOT NULL, judge_id TEXT NOT NULL,
  pen_ground_max_mm REAL, pen_object_max_mm REAL, pen_self_max_mm REAL,
  solid_contact_rate REAL, hover_rate REAL, slip_p95_cm_s REAL, human_agree_rate REAL,
  flagged_frames INTEGER, excluded_frames INTEGER,
  verdict TEXT CHECK(verdict IN('pass','fail','not_evaluated')), series_art TEXT, created_at TEXT,
  PRIMARY KEY(subject_kind, subject_id, judge_id));
CREATE TABLE clip_labels(clip_id TEXT, taxonomy TEXT, family TEXT, tags TEXT, label_source TEXT, confidence REAL,
  PRIMARY KEY(clip_id, taxonomy));
CREATE TABLE audit_items(item_id TEXT PRIMARY KEY, subject_art TEXT, art_sha256 TEXT, priority REAL,
  sample_kind TEXT CHECK(sample_kind IN('risk','random')), batch TEXT, enqueued_at TEXT);
CREATE TABLE audit_decisions(id INTEGER PRIMARY KEY, item_id TEXT, art_sha256 TEXT,
  verdict TEXT CHECK(verdict IN('accept','reject','needs_fix')), frame_ranges TEXT, reasons TEXT, note TEXT,
  reviewer TEXT, reviewer_kind TEXT CHECK(reviewer_kind = 'human'), ts TEXT);
CREATE TABLE issues(issue_id TEXT PRIMARY KEY, scope TEXT, target TEXT, severity TEXT, title TEXT, body TEXT,
  status TEXT CHECK(status IN('open','fixing','fixed','wontfix')), blocks_packing INTEGER DEFAULT 0, created_by TEXT, ts TEXT);

-- 5. 打包
CREATE TABLE packs(pack_id TEXT PRIMARY KEY, selection_art TEXT NOT NULL, spec_version TEXT, format TEXT,
  license_rollup TEXT, release_policy TEXT, stats TEXT, status TEXT, created_by_job TEXT);
CREATE TABLE pack_members(pack_id TEXT, retarget_id TEXT, clip_id TEXT, robot_id TEXT, split TEXT,
  admitted_by TEXT CHECK(admitted_by IN('human_accept','random_batch_pass','auto_gate')), PRIMARY KEY(pack_id, retarget_id));
CREATE TABLE shards(shard_art TEXT PRIMARY KEY, pack_id TEXT, body TEXT, rows INTEGER, bytes INTEGER);
CREATE TABLE legacy_datasets(cache_id TEXT PRIMARY KEY, art_id TEXT, representation_kind TEXT,
  up_axis TEXT, length_unit TEXT, fps REAL, lineage_json TEXT);   -- 从 pretraining_foundation 索引导入

-- 6. 模型和训练
CREATE TABLE models(model_id TEXT PRIMARY KEY, family TEXT, size TEXT, title TEXT, crowned_by TEXT,
  crowned_at TEXT, card_art TEXT, release TEXT);
CREATE TABLE runs(run_id TEXT PRIMARY KEY, family TEXT, model_id TEXT, pack_id TEXT, config_art TEXT,
  code_commit TEXT, status TEXT CHECK(status IN('planned','waiting_approval','running','finished','failed','killed','imported')),
  approval_id TEXT, est_gpu_h REAL, est_kwh REAL, started_at TEXT, ended_at TEXT, dir_art TEXT, note_zh TEXT);
CREATE TABLE checkpoints(ckpt_id TEXT PRIMARY KEY, run_id TEXT, step INTEGER, art_id TEXT,
  keep TEXT CHECK(keep IN('best','last','milestone','release','drop')), metrics TEXT);

-- 7. 评估
CREATE TABLE eval_sets(eval_set_id TEXT PRIMARY KEY, purpose TEXT, members_art TEXT, members_sha TEXT,
  frozen_at TEXT, leakage_checked INTEGER);
CREATE TABLE suites(suite_id TEXT PRIMARY KEY, judges TEXT, standard_art TEXT, protocol_art TEXT);
CREATE TABLE eval_runs(eval_id TEXT PRIMARY KEY, suite_id TEXT, eval_set_id TEXT,
  subject_kind TEXT CHECK(subject_kind IN('ckpt','engine','pack','refiner')), subject_id TEXT, control_id TEXT,
  job_id TEXT, status TEXT, decision TEXT CHECK(decision IN('win','lose','tie','not_evaluated') OR decision IS NULL),
  report_art TEXT, created_at TEXT);
CREATE TABLE eval_results(eval_id TEXT, case_id TEXT, metric TEXT, value REAL, gate TEXT,
  outcome TEXT CHECK(outcome IN('pass','fail','not_evaluated')), media_id TEXT, PRIMARY KEY(eval_id, case_id, metric));
CREATE TABLE media(media_id TEXT PRIMARY KEY, kind TEXT CHECK(kind IN('triptych','grid12','showcase','live_snapshot')),
  art_id TEXT, layout TEXT, subjects TEXT, seconds REAL, viewed_by_lloyd INTEGER DEFAULT 0);

-- 8. 作业、审批、服务
CREATE TABLE jobs(job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, request_hash TEXT UNIQUE NOT NULL, params TEXT NOT NULL,
  resource_class TEXT CHECK(resource_class IN('light','cpu_heavy','gpu_heavy','net_heavy')), net_conns INTEGER DEFAULT 0,
  priority INTEGER DEFAULT 0,
  status TEXT CHECK(status IN('queued','waiting_approval','leased','running','done','failed','cancelled','external')),
  approval_id TEXT, parent_job TEXT, submitted_by TEXT, claimed_by TEXT, pid INTEGER, boot_id TEXT, lease_until TEXT,
  attempt INTEGER DEFAULT 0, est_seconds REAL, est_out_bytes INTEGER, created_at TEXT, started_at TEXT, finished_at TEXT,
  error TEXT, outputs TEXT);
CREATE INDEX jobs_claim_idx ON jobs(status, resource_class, priority, created_at);
CREATE TABLE job_events(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts TEXT, stage TEXT, payload TEXT);
CREATE TABLE approvals(approval_id TEXT PRIMARY KEY,
  scope TEXT CHECK(scope IN('train','wbc_eval','long_gpu','gc_execute','cross_fs_move','crown','license_override')),
  request TEXT, cost_estimate TEXT, decided TEXT CHECK(decided IN('pending','approved','rejected')),
  decided_by TEXT, decided_at TEXT, via TEXT CHECK(via IN('web_human','cli_human')));
CREATE TABLE services(svc_id TEXT PRIMARY KEY, port INTEGER, bind TEXT, pid INTEGER, boot_id TEXT,
  started_at TEXT, idle_ttl_h REAL, status TEXT);

-- 9. 迁移账本
CREATE TABLE legacy_paths(old_path TEXT PRIMARY KEY, art_id TEXT, from_map TEXT, purpose_zh TEXT,
  status TEXT CHECK(status IN('registered','linked','moved','archived','delete_candidate','deleted','rescued')),
  new_uri TEXT, hardcoded_refs INTEGER, phase TEXT, note_zh TEXT);
```

**快照**：每晚 `sqlite3 .backup`，压缩后放到 `catalog/snapshots/`。同时复制一份到另一个分区（`/var/tmp/alphamotion/_catalog_backup/`），保留 14 个日快照和 8 个周快照。账本坏了可以从快照加上 `job_events` 重放恢复。

### C.2 产物库（内容寻址）

产物的身份由内容决定（哈希），但物理上不强行把所有东西拆成散块。分三种：

| 种类 | 什么时候用 | 身份 | 物理存放 |
|---|---|---|---|
| 单文件 | 会被多处引用的不可变文件：机器人资产里的单个文件、规格快照、冻结考卷清单、`.umrs` 分片、checkpoint | `sha256:<hex>` | `store/blobs/sha256/ab/cd/<hex>`，同一分区内用硬链接放进去（不复制），然后去掉写权限 |
| 目录 | 一组文件一起用：一个训练包、一个 run、一个评估、一台机器人的资产目录 | `tree:<hex>` = 排好序的（相对路径，大小，sha256，权限）清单再取哈希，清单存在 `store/trees/` | 文件就留在可读的托管目录里（`data/…`、`runs/…`、`evals/…`），封存后改成只读 |
| 旧大目录 | 不想复制的大块（140G 缓存、101G HiPHI 解压文件）| `ext:<指纹>` = 文件数 + 总字节数 + 最新修改时间 + 抽 64 个文件、每个首尾各 1 MB 的哈希 | 原地不动。空闲时 `store.verify --full` 用 `ionice -c3` 慢慢补算完整哈希，算完升级成 `tree:` |

**生命周期**：作业开始时把产物写进 `store/open/<job_id>/`，账本状态是 `open`。作业成功后封存，分四步：
1. 算哈希；
2. 去掉写权限；
3. 挪到托管路径，同一分区内用 rename；
4. 账本状态改成 `sealed`。

作业失败的话，`open/` 下的东西留 3 天方便排查，然后进清理计划。

**完整性**：`store.verify` 每天抽查 1%，每周把 `pinned` 产物全查一遍。不一致就标 `corrupt`，并自动开一条 issue。

### C.3 训练分片格式

- **沿用 UMRS v1 容器**。它在 7 条样本上实测过：
  - 根位置无损；
  - 关节误差 ≤ 4.5e-5 rad；
  - 4096 个表面点重建误差 ≤ 0.062 mm；
  - 接触和打滑标签逐位无损；
  - SMPL-X 顶点误差 ≤ 0.21 mm；
  - 量化后可能违反严格规则的帧会被打上第 64 位标记。

  它还带 sha、能 mmap、只追加，已经够用。
- **UMRS v2 只改输入端和头部，容器本身不改**：
  1. 改读 `motion_v3`（单位米、CSR 接触）；
  2. 增加可选的"点级接触"数据块，DATA_SPEC §3 要求逐点稀疏接触；虽然理论上能从 qpos 加正向运动学重算，但 1 mm 门限附近 0.06 mm 的重建误差会让少数点翻转，所以直接存；
  3. 每条记录带 `license_id` 和 `pack_id`；
  4. 人体数据一律走 `formats.human.read`，修掉 HiPHI 路径找错的问题。
- **怎么分片**：每片 0.5–2 GB。机器人分片按（机器人，批次）切，人体单独成片。
  - 训练 / 验证 / 测试按 `split_group`（演员或会话）划分，防止同一个人的动作同时出现在训练集和考卷里。
  - 留出机器人沿用 pretraining_foundation 索引的"见过 / 没见过"划分，组合出四种评估分区：见过的身体 × 见过的动作、见过的身体 × 没见过的动作、没见过的身体 × 见过的动作、都没见过。
- **加载器**：`arena_motion.formats.umrs.PairedWindows(pack_id, robots, window, stride)`，来自 `umr_dataset/dataset.py`。
- **旧训练缓存**（`cache_lafan_v3`、`cache_amass_lconv_tfix` 等）不转换。按 pretraining_foundation 索引里已有的 `datasets` 和 `body_assets` 描述，导入 `legacy_datasets` 表，旧训练代码照旧能用。

**实测体积**（这次只读量的）：

| 格式 | 每机器人帧 | 依据 |
|---|---|---|
| `factory_npz_v1` | 约 0.41–0.46 KB | g1_29：2.62 MB / 5,771 帧；LAFAN：1.63 MB / 3,945 帧 |
| `gpu_npz_v2` | 约 0.33 KB | 1.89 MB / 5,771 帧 |
| UMRS 机器人分片 | 约 0.05–0.18 KB | g1_23 只有身体：77.8 KB / 1,500 帧；g1_29 带物体：106 KB / 600 帧 |
| UMRS 人体 | 约 0.2 KB/帧 | 1.97 MB / 9,731 帧 |

### C.4 磁盘预算

**现在**（10-07 18:29）：
- `/`：657G，已用 88%，剩 76G。上面有 `/var/tmp/alphamotion` 352G。
- `/home`：1.2T，已用 70%，剩 345G。
- HiPHI 下载器自带两条水位：工厂目录最多 300G，/home 至少剩 200G。

**量产以后会有多大**（126 h 在手的人体数据 × 30 fps × 21 台 ≈ 2.9 亿机器人帧）：

| 产物 | 126 h × 21 台 | 571 h（已知全部）× 21 台 | 结论 |
|---|---|---|---|
| motion_v3 npz（约 0.33–0.45 KB/帧）| 95–130 GB | 430–590 GB | **存不下全量**。npz 只是中转：打包验证完以后，只留审计样本、被标记的、考卷要用的，其余标"可再生"（重算一遍约 30 GPU 小时）|
| UMRS 训练分片 | 15–50 GB | 65–235 GB | 这是正本，长期保留 |
| 人体（30 fps）| 约 3 GB 分片 / 约 20–30 GB npz | 约 15 GB / 约 100 GB | 90 fps 版本可以从原始数据重算 |

**配额**（写在 `am.toml`，调度器执行）：

| 类别 | 位置 | 配额 | 现在 | 规则 |
|---|---|---|---|---|
| 原始数据 | /home（factory）| 300G（沿用下载器的 BUDGET_GB）| 约 156G | HiPHI 有下载截止时间，过了就拿不到，所以原始数据是 `pinned` |
| 人体 | /home | 80G | 约 28G | 90 fps 版本标"可再生" |
| 机器人资产和对应关系 | /home | 15G | 约 5G | |
| 重定向 npz | /home | 150G 滚动 | <1G | 打包后按 C.5 规则清理 |
| 训练包 | /home | 120G | 约 0 | 当前版本加上一版 |
| 新训练 run | /home | 150G | 0 | 每个 run 只留部分 checkpoint（见 C.5）|
| 评估和视频 | /home | 40G | — | |
| 临时目录 | /home | 30G | — | 14 天过期 |
| `/` 上的旧数据 | / | **不再增加** | 352G | 根分区只读不写；如果 `xwbc_full` 获准删除，能多出约 129G |

**两条线**：
- /home 剩余不到 150G，或 `/` 剩余不到 60G 时，调度器停止放行所有会写大文件的作业，并在 README 顶部报警。
- 作业提交时必须带 `est_out_bytes`（预计输出大小），超过剩余配额就直接拒绝。

### C.5 保留和清理

| 保留级别 | 包括什么 | 规则 |
|---|---|---|
| `pinned` | 规格、license、账本快照；发布过的模型权重以及整条血统（run → 训练包 → 重定向 → 人体 → 原始）；冻结考卷；判决和报告引用的视频；过了截止时间就拿不回来的原始数据（HiPHI）| 永不自动清理 |
| `canonical` | 每个训练包的当前版本和上一版；当前机器人资产；训练包用到的人体数据 | 只在出了新版本、且没人引用以后才进清理计划 |
| `regenerable` | 已经打包的 npz、`umr_gpu/problems`、有 30 fps 版本时的 90 fps 人体、没人引用的渲染图、各种编译缓存 | 超配额时按最近最少使用排序进清理计划；前提是能重算：作业参数还在、输入还在 |
| `scratch` | `scratch/` 下所有东西 | 14 天到期提醒，21 天进清理计划 |
| `archive` | 冻结的旧路线 | 压缩、只读；有冷存储盘以后挪过去 |
| run 的 checkpoint | — | 保留最好的、最后的、每 25% 进度一个里程碑、发布用的；其余 30 天后进清理计划。曲线和日志永远保留（都很小）|

**清理一律分三步，代理只能做第一步**：

1. `am store gc --plan` 生成一个清理计划。它本身也是一个产物，列出每一项的原因、大小，以及"重算要花多少"。
2. lloyd 在网页上逐项勾选批准，或者用 `am approve`。系统会核对来源，只认人的操作（`via` 必须是 `web_human` 或 `cli_human`）。
3. `am store gc --execute`：先挪进同分区的 `store/trash/<plan_id>/`，7 天后才真删。每一步都记进账本。

---

## D. 模型侧

### D.1 训练 run 登记

- **提交前先估算**：`am train plan --family am2 --config specs/train/am2_B_M.yaml --pack core21_hoi@v1`
  1. 在账本里生成一个 run，状态 `planned`；
  2. 估算 GPU 小时、电量和电费；
  3. 生成一个审批请求；
  4. lloyd 批准以后，作业 `train.run` 才进入重 GPU 队列。它和重定向的 GPU 作业共用唯一一个重 GPU 名额，由调度器排队。
- **run 目录** `runs/<family>/<run_id>/`：
  - `RUN.json`：冻结的配置、训练包 ID、代码 commit（代码有没提交的改动就拒绝启动）、随机种子、审批号；
  - `metrics.jsonl`、`events.jsonl`；
  - `ckpt/step_XXXXXXX.pt`；
  - `samples/<step>/`：训练中途自动出的三联视频和 12 帧网格；
  - `evals/`：指向评估 ID 的链接；
  - `README.md`：自动生成。
- **训练代码只需要接两个回调**：`reporter.on_step(metrics)`、`reporter.on_checkpoint(path, step)`。账本就能实时看到新的 checkpoint。没改过的旧训练代码，由文件监视器兜底。
- **旧 run 导入**（`s6_train/importers/`），状态一律记为 `imported`：
  - `topo_exp/runs`：两份地图分别数出 204 和 823 个，口径不同，导入时以实际数为准；
  - `MODEL_VERSIONS.md`、`RUNS_INDEX.md`；
  - AM2 六组 A/B × S/M/L、megatron 试训、am2exp 各臂、mjlab 和 UFO 控制训练；
  - 这些都和 AlphaMotion v0.2 的发布权重（Bumblebee / Megatron / Prime = S / M / L20k）连上血统。
- **头衔**：`models.title` 只能由 `am model crown <model> --title Greenwich` 写入，而且必须带上 lloyd 的审批号。对决只出候选。

### D.2 实时可视化：边训练边看模型生成了什么

**目标**：同一个画面里同步播放人体源、参考（数据里的 UMR 严格重定向结果）、模型输出（可以是多个 checkpoint）、仿真（可选）。每一路都显示骨架、4096 个表面点和接触。

**一种场景包，两种查看器**：

1. **场景包 `scene_bundle_v1`**（`specs/formats/scene_bundle_v1.md`）：
   ```
   { bundle_id, fps, frames, floor_z: 0,
     lanes: [ { lane_id, label, kind: human|reference|model|sim,
                robot: "unitree_h2@3fa2c1d0" | null, smplx: {...} | null,
                qpos[T,nq], contacts_csr{ptr,idx,target}, region_bits[T,4], flags[T],
                per_frame{pen_mm, slip_cm_s, agree} } ],
     objects: [ {object_id, pose[T,7]} ], provenance{ckpt_ids, engine, judge} }
   ```
   包里不存逐帧点坐标。查看器按机器人资产缓存表面点，用正向运动学自己算（和 DATA_SPEC 的思路一样），所以一个包只有几 MB。
2. **谁生成场景包**：作业 `viz.bundle`。给定（片段、机器人、checkpoint 列表），它做三件事：
   - 模型那一路只用 **CPU** 推理，窗口不超过 20 秒，不和 GPU 作业抢；
   - 参考那一路取 S3 的结果；
   - 人体那一路取 S1 的结果。
3. **查看器 A：常驻 viser**（`127.0.0.1:8781`，通过服务的 `/viewer` 转出去）。
   - 底子是产品仓 `alphamotion.viz.live.LiveViewer`：一个常驻进程，换内容不用重启，做过手机端验收。
   - 闲置 8 小时自动退出。全系统只有一个实例。
4. **查看器 B：网页里的 Three.js**（在 motion_audit_web 前端上扩展）读同一个包（`GET /api/v1/scenes/{id}`），用在审计页面和手机上。
5. **跟训练模式**：作业 `train.watch`（轻量）收到新 checkpoint 事件后，做三件事：
   - 在 `golden_quick@v1` 上触发 `viz.bundle`。这套考卷是 12 段动作 × 3 台机器人：当前训练用的 2 台加 1 台留出机器人，默认 G1、H2，留出那台用 X2；
   - 查看器自动换上最新的 checkpoint；
   - 同时出三联视频和 12 帧网格，放进 run 的 `samples/`。

   这样 lloyd 问"训练怎么样"时，总有一段最新的视频可以发。
6. **显示内容**：
   - 几路并排，或者把参考画成灰色影子叠在模型上；
   - 表面点按区域上色；
   - 接触点高亮：地面绿、物体蓝；
   - 接触点速度超过 1 cm/s 画红箭头；
   - 穿透的点标红；
   - 时间轴下面挂逐帧指标条：穿透、打滑、和人一致；
   - 同步播放、拖动、在两个 checkpoint 之间切换；
   - 位移大时相机跟着根节点走；
   - 人体可以显示成 SMPL-X 网格或棍状图。LAFAN 缓存的根旋转不可靠，默认用棍状图。
7. **合并、停掉 6 个旧查看器**：motion-lab（:2334）、rollout-gallery（:8891）、greenwich-mobile（:8899）、studio（:8097）、viser_mesh_viewer（:8093）、lab_server（:8114）。
   - lab_server 那种"分段选补全 + 多臂同场"的交互要写进需求。
   - greenwich-mobile 那种按字节范围发视频的写法并进 `service/files.py`。

### D.3 模型重定向和生成动作的评估套件

**五条铁律**：

1. **判官独立**：判官不 import 训练代码，也不读模型自己的接触头。有一个测试专门检查 import 关系，账本表 `judges.imports_training_code` 上也有约束，必须是 0。
2. **同一把尺**：数据（S3 的参考重定向）和模型输出用同一个判官、同一份阈值。参考数据自己的得分就是"数据地板"，作为对照臂，先量对照再解读。
3. **真值不是唯一的尺子**：两条轴都要量，一条是不需要真值的物理一致性，另一条是 WBC 能不能跟住。
4. **缺材料就报"未评"**（`not_evaluated`），不许默默算通过。
5. **不能光凭数字下结论**：每个判决都要挂三联视频和全程 12 帧网格；要用来"升级模型"的判决，必须在 lloyd 看过视频（`media.viewed_by_lloyd=1`）之后才能定。

**判官一览**（全部在 `s7_eval/judges/`）：

| 判官 | 量什么 | 合并自 | 费用 |
|---|---|---|---|
| J-geom | 穿地、穿物体、自穿；实接触率（0–1 mm）；悬空；接触点打滑；和人一致（DATA_SPEC v2 口径，4096 点 + MuJoCo 碰撞几何复核）| `umr_gpu/eval/judge.py` + am2exp J1 `ground_judge`（厂商 MJCF 碰撞几何、15 个区域）| CPU |
| J-loco | WBC 可训性四张考卷；不需要真值的物理检查（根高抖动、姿态抖动、脚滑、穿地）；脚滑审计；接地四口径 | arena-datagrid `arena/eval/locomotion.py` + `locomotion.v1.json`，原版是 wbc_ready_audit、x2_phys、slip_audit、ground_metrics | CPU |
| J-manip | 手和物体的闭合（握住 / 张开 / 压碎）；手离物体表面的距离 | am2exp J2 `closure_judge` | CPU |
| J-fid | 和参考比的姿态保真：关节角误差；手的世界系位置误差（cm）和腿的角度误差（°）**分开报，不混在一起**；根轨迹 | AM2 `eval_engine`、`surface_metrics` | CPU |
| J-codec | 编解码回归考卷：幅度比、上肢正向运动学、腿角护栏、方言一致率 | `v2_judge.py` | CPU |
| J-wbc | WBC 跟踪率（mjlab BeyondMimic）| `alphamotion_ctrl` 的 `evaluate_local` | **重 GPU，要批准** |
| J-latent | 潜空间打分（标明"不是物理指标"）| Greenwich SDK 的 `judge.py` | 可选 |

**冻结考卷**（`specs/eval/eval_sets/*.json`，成员清单和 sha 冻结；只从不在任何训练包里的数据抽；按 `split_group` 做泄漏测试）：

| 考卷 | 内容 | 用途 |
|---|---|---|
| `golden_quick@v1` | 12 段 × 3 台机器人，覆盖主要动作大类 | 训练中途自动出片和快速打分，CPU 5 分钟以内跑完 |
| `golden_core@v1` | 约 200 段，按 主类 × 源 分层；含物体、地形；含见过和留出的机器人；分成 C.3 说的四种评估分区 | 正式比较 |
| `stress@v1` | 摔倒起身、爬、上下楼梯、搬箱、踢、舞蹈、快跑（来自稀缺清单）| 压力测试 |
| `heldout_robots@v1` | 零样本机器人（X2、A3 等）| 跨具身泛化 |
| `amass_heldout600@legacy` | 已有的 600 段语料外 WBC 审计集（时间轴已重采样修正）| 和历史结果衔接 |
| `parity40` | 引擎对拍用例 | S3 引擎门 |

**协议和统计**（`specs/eval/protocols/`）：
- `motion_eval_v1.yaml` 对应六项纪律：数值、视觉、统计、信息熵、跨具身、动作泛化。
- `spatial_eval_v1.yaml` 对应五项必考。
- 作业 `eval.suite` 跑判官；`eval.compare` 和对照臂做配对检验（按窗口分组的符号检验 + 置换检验），代码来自 am2exp 的 `cases`、`stats`、`decide()`。
- 结论只有四种：`win`、`lose`、`tie`、`not_evaluated`。

**三联视频**（`viz/render/`，合并自 am2exp 的 `render/triptych_*` 和 `grid12`、`rollout_tools` 的 `encode`（校验逐帧时间戳）、`render_check.py` 的接触和穿透上色）：
- 布局固定为 **人体源 | 对照臂 | 本臂**，1080p，带真实时间戳；
- 12 帧网格均匀覆盖全程，首帧和末帧都在；
- 另出 `poster.jpg` 和 `manifest.json`；
- MCP 的 `get_media` 返回本地路径和下载链接，方便 Claude 直接把 mp4 文件发给 lloyd。

**排行榜**（`leaderboard(suite@ver, eval_set@ver)`）：
- 只把同一套件版本、同一考卷版本的结果放在一起比；
- 每行一个 checkpoint，列出：对照臂、和对照臂比的结论、每个门的通过率、穿透 p99、实接触率、打滑 p95、和人一致率、手部 cm 和腿部 °、WBC 跟踪率（跑过才有）、案例数、"未评"数、三联视频链接；
- 固定显示两行基线："数据地板"（参考重定向）和 Native policy（GMR 原生）；
- 排序先看结论，再看指标，不按平均分排。

**发布门**：产品仓的 `quality.release_passed` 和浏览器验收，作为套件 `release@v1` 接进来。

---

## 服务：一个进程，两种入口

```
 浏览器 / Codex 网页 ──HTTP /api/v1──┐
                                      ├──> service（FastAPI，127.0.0.1:8780）──> core.queries（只读）──> catalog
 Claude / 代理 ──MCP（/mcp 或 am mcp）──┘                                └──> core.jobs.submit ──> jobs 表
                                                                                         │
                       am worker --class cpu|gpu|net|light（受调度器管）<───────────────┘
                       查看器子进程 viser :8781（经 /viewer 转出去）
```

- **一个命令** `am serve` 起 HTTP 和 MCP；`am mcp` 是给 Claude Code 本地用的 stdio 版。两者调用同一组函数，没有各自的业务逻辑。
- **只监听本机** `127.0.0.1`，lloyd 用 SSH 转发访问。闲置 8 小时自动退出。进程记录 pid 和 boot id，按 PID 精确停止，不用 `pkill -f`。
- **两种角色**：网页登录是"人"，MCP 客户端是"代理"。代理可以查询、提交作业、写笔记、开 issue；代理**不能**批准、加冕、执行删除、写人工审计判决。
- **提交幂等**：`request_hash` 由（类型，输入 ID，代码 commit，参数）算出，并且唯一。同一个请求交两次，拿到的还是同一个作业。
- **调度器**（`core/governor.py`）照 `RESOURCE_LIMITS.md` 执行：
  - 所有重活一律用 `systemd-run --user --scope -p MemoryMax=20G taskset -c 0-15 nice -n 10` 启动，并设 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`；
  - CPU 重活最多 6 个，重 GPU 最多 1 个；
  - GPU 利用率超过 70% 或功耗超过 350 W 时，暂停放行新的 GPU 作业；
  - 下载连接数按作业分配，合计不超过 16；
  - 可用内存不到 15G 就排队等；
  - 读 `throttle.log`，发现 `net=down` 就暂停所有下载和新的重活；
  - 已经在跑的旧进程登记成 `external` 作业，计入总账。
- **底座**：
  - MCP 沿用 `motion_audit_web/agent_tools/server.py` 已有的路子（`workflow_plan` / `workflow_submit` / `job_status` / `evidence_read` / `video_inspect`）；
  - 作业表沿用 arena-datagrid `db/store.py` 的 `jobs` / `job_events`（带租约和唯一的 `request_hash`）；
  - 回执和批准文件的做法沿用 am2exp 的 ops 层（`lease`、`guard`、`receipts`、`approvals`）。

**MCP 工具清单**（`specs/api/mcp_tools.v1.json`，用快照测试冻结）：

| 组 | 只读 | 写（幂等、受调度器管）|
|---|---|---|
| 总览 | `platform_status` · `what_is` · `search` | |
| S1 | `list_sources` · `source_card` · `ingest_status` · `license_check` | |
| S2 | `list_robots` · `robot_card` · `surface_stats` · `robot_impact` | |
| S3 | `retarget_status` · `retarget_inspect` | |
| S4 | `dataset_overview` · `diversity` · `quality_summary` · `find_clips` · `get_clip` · `audit_queue` · `audit_stats` · `list_issues` | `add_note` · `open_issue` |
| S5 | `storage_status` · `artifact_info` · `lineage` · `list_packs` · `pack_card` · `gc_plan` | |
| S6 | `list_runs` · `run_card` · `list_models` · `model_card` | `viewer_load` |
| S7 | `eval_result` · `leaderboard` · `get_media` | `run_eval` · `render` |
| 作业 | `list_jobs` · `job_status` · `job_logs` | `plan_job`（只估时长、费用和产出大小）· `submit_job` · `cancel_job`（只能取消自己提交的）|

**HTTP**（`specs/api/openapi.v1.json` 冻结，Codex 照它写网页）：
- `/status`、`/what`、`/search`
- `/sources`、`/clips`、`/clips/{id}`
- `/robots`、`/robots/{id}`、`/robots/{id}/surface.bin`
- `/retargets`、`/retargets/{id}`、`/retargets/{id}/frames`
- `/dseval/coverage`、`/dseval/quality`、`/dseval/reports`、`/issues`
- `/audit/queue`、`/audit/decisions`、`/audit/issues`
- `/store/volumes`、`/artifacts/{id}`、`/artifacts/{id}/lineage`、`/packs`、`/packs/{id}`、`/gc/plans`
- `/runs`、`/runs/{id}`、`/runs/{id}/metrics`、`/models`
- `/evals`、`/evals/{id}`、`/evals/leaderboard`、`/scenes/{id}`、`/viewer/load`、`/media/{id}`（支持按字节范围取）
- `/jobs`、`/jobs/{id}`、`/approvals`、`/approvals/{id}`

**网页页面**（交给 Codex）：总览、数据源、片段浏览、机器人、重定向、整库评估、人工审计（现成）、存储和清理计划、训练 run、实时查看、评估和排行榜、作业、待你批准。

---

## E. 迁移计划

### E.1 规矩

1. **不打断在跑的东西**。现在在跑的有：
   - HiPHI 下载器：离截止还有约 36 小时，全部下完预计约 60 小时；
   - GRAIL 的下载、转换、碰撞和清单重建循环：约 25 小时；
   - `gpu_convert` 的定时转换；
   - `umr_gpu` 的吞吐测试；
   - `throttle_guard`；
   - robots_wf 和 diversity_wf 两个工作流代理；
   - pretraining_foundation 的下载器；
   - 3 个常驻服务。

   这些进程写死的路径，在它们自然结束之前一律不动。
2. **对拍过的引擎**：先原地纳入 git，代码一个字节都不改。之后每改一处，都要重跑对拍：`umr_gpu` 的 parity 表、`gpu_convert` 的 parity40、`umr_dataset` 的往返和严格复验。
3. **搬家必留反向软链**，至少保留 30 天。所有动作都记进 `legacy_paths`。
4. **删除只提议**：标成 `delete_candidate`，进清理计划，等 lloyd 批准。本设计不执行任何删除。
5. **conda 环境不能直接挪**：里面的脚本头写死了绝对路径。做法是在 /home 上按锁文件重建一个新环境，切过去，旧环境原地保留一个月。`umr/env` 和 `genmo-gem` 都照这个办；`GENMO/` 是以可编辑方式装进环境的，在新环境里重新装一遍。

### E.2 分阶段

| 阶段 | 做什么 | 代理工时 | 能开始的时间 | 风险 |
|---|---|---|---|---|
| **P0 建账** | 建 `~/arena_motion` 骨架和 `am.toml`，初始化账本。把四份地图的表导入 `legacy_paths`。旧顶层目录逐个登记成 `ext:` 产物（`du` 带超时、低优先级）。导入现成清单：GRAIL manifest 22,312 行、HiPHI 转换日志、pretraining_foundation 索引 240,444 条、`robots.json` 34 台、`umr_dataset` 的 catalog、`MODEL_VERSIONS` 和 `RUNS_INDEX`、sources_inventory 里的 license。生成第一版 README | 0.5 天 | 马上 | 零：只读，只写新目录 |
| **P1 代码进 git** | 原地给 `umr_factory_code`、`umr_gpu`、`gpu_convert`、`umr_dataset`、`motion_audit_web`、工厂的 `scripts*` 以及主仓里 `alpha_motion2` 的代码部分（`pretraining_foundation`、`research/*/code`）做 git 快照，`.bak` 文件也照原样提交。`engines/` 用 subtree 引用这些 commit。GPU 空出来的时候重跑对拍，在账本里记 `parity_status=verified@commit` | 1 天 | 马上（`git init` 不影响正在跑的进程）| 低 |
| **P2 服务骨架** | `am serve` 先上只读的 HTTP 和 MCP；冻结 OpenAPI；网页先接总览、`what`、存储三个页面 | 2 天 | P0 之后 | 低 |
| **P3 路径改成读配置** | 把写死的路径换成 `am.uri` 或环境变量：`umrf/paths.py`、`umr_gpu` 22 处、`gpu_convert` 和 `bvh2smplx` 里的 `~/Downloads/SMPLX_NEUTRAL.npz`、`umr_dataset/sources.py`、`motion_audit_web` 的 `DATA_ROOT`、`run_parity.py` 里的 `lloyd_gpu_eval`、`diversity_wf.js` 里的临时目录、`robots.json` 里的 MJCF 来源、arena-datagrid 的 GMR 路径（已经支持 `ARENA_GMR_PARAMS`）、Studio 的 exp_topo。每改一处就过一次对拍门，旧路径照样能用 | 3 天 | 相关进程结束以后（GRAIL 约 25 小时，HiPHI 约 60 小时）| 中：靠对拍门兜底 |
| **P4 统一格式、阈值，修箱子 bug** | 出 `motion_v3` 的读写和两个转换器；阈值只留一份；补立方体测试并修 `objects.py`；搬箱片段重跑；打包器和审计网页改读 v3 | 4 天 | P1 之后 | 中 |
| **P5 作业层接管** | 新的下载、转换、重定向、评估都改走 `am` 作业；旧循环登记成 `external`，等它们自然结束后退役 | 2 天 | P2、P3 之后 | 低 |
| **P6 搬小而承重的文件** | `~/tmp` 下这些被代码写死路径的文件，搬进 `data/` 并留反向软链：`sole_offsets*.json`、`gauge_check_0903.json`、`dialect/`、`refiner_suite/`、`arena_handoff_backup/`、`greenwich_workbench_0911/datagrid`（先确认没有服务在用）。`cl_v3` 有 31G，且被 34 个链接引用，先原地登记，**不搬** | 1 天 | P3 之后 | 低 |
| **P7 机器人资产合成一份** | 工厂的 `robots/`、`corr/`，`/var/tmp` 的 `robots.json` 和 `umr_configs`，`robot_models`，sonic 的 H2，`sole_offsets` → `data/s2_robots/<r>/<hash>/`；`surface_v2` 归档；`corr` 不一致的问题先查清楚再定 | 2 天 | `umr_gpu` 吞吐测试结束、P3 之后 | 中：有资产哈希门 |
| **P8 合并服务** | 6 个旧查看器和画廊并进 `am serve` 和查看器，然后停掉旧的；处理 ardy-isaac 崩溃循环（要 lloyd 批准）| 1 天 | P2、D.2 完成以后 | 低 |
| **P9 归档和删除清单** | 生成归档计划和删除候选清单（约 180G），交 lloyd 逐项批准 | 0.5 天出清单 | P0 之后随时 | 零（只是清单）|
| **P10 大缓存** | **默认不搬**，原地登记。只有 lloyd 批准删掉 `xwbc_full` 等东西、`/` 空出来以后，才考虑重新安排 | — | 看 lloyd 的决定 | — |

**合计**：代理工作约 17 天。几项可以并行，加上等在跑的作业结束，日历时间约 2.5–3 周。P0 到 P2 第一周就能交付：有账本、有 README、有 `am what`，有只读的网页和 MCP。

### E.3 每个现有文件夹的去向

动作的写法：
- **登记**：原地不动，进账本；
- **链入**：在根目录下建软链；
- **搬**：带反向软链地挪过去；
- **进 git**：原地做 git 快照；
- **归档**：挪进 `archive/`；
- **删候选**：只提议；
- **不进根**：与 AlphaMotion 无关，留在原处。

#### E.3.1 `~/umr_factory_data`（190G，/home）

| 现在的位置 | 一句话 | 动作 | 新位置 | 阶段 | 备注 |
|---|---|---|---|---|---|
| `raw/hiphi` | HiPHI 原始数据（下载中）| 登记 + 链入 | `data/s1_raw/hiphi` | P0 | `pinned`（过了截止就拿不到）；下载器写死了路径，不动 |
| `raw/grail` | GRAIL 原始数据（下载中）| 登记 + 链入 | `data/s1_raw/grail` | P0 | 循环跑完前不动 |
| `raw/omnicontact` | OmniContact 原始数据 | 登记 + 链入 | `data/s1_raw/omnicontact` | P0 | |
| `smplx/hiphi` | HiPHI SMPL-X，CPU 版 1,094 段 | 登记 | `data/s1_human/hiphi@fps90.cpu_v1` | P0 | 有 GPU 版的标 `superseded`；只有 CPU 版的补 30 fps |
| `smplx/omnicontact` | Omni SMPL-X，690 段 | 登记 + 链入 | `data/s1_human/omnicontact@fps90.v1` | P0 | 按字段分开的 npy 靠适配器读 |
| `smplx/grail` | GRAIL SMPL-X + 逐段 license 清单 | 登记 + 链入 | `data/s1_human/grail@fps24.v3` | P0 | manifest 导进账本；它是逐段 license 的样板 |
| `smplx/_test` | 两个测试文件 | 删候选 | — | P9 | |
| `smplx_gpu/hiphi`、`hiphi_30fps` | HiPHI SMPL-X，GPU 版（转换中）| 登记 + 链入 | `data/s1_human/hiphi@fps90.v1`、`@fps30.v1` | P0 | 90 fps 版标"可再生" |
| `robots/` | 21 台核心机器人的资产 | 搬 | `data/s2_robots/<r>/<hash>/` | P7 | 引擎读这里，P3 改完路径以后才搬 |
| `robots/<r>/pipeline/` | 5 台机器人各自的 QA 流水线 | 拆开 | 配置 → 资产目录；脚本 → `s7_eval/judges/geom`；产物 → `evals/` | P7 | robots_wf 在跑 |
| `corr/` | 对应关系缓存 | 搬 | `data/s2_robots/<r>/<hash>/corr/` | P7 | 先查清和 GPU 那份的差异 |
| `runs/` | CPU 跑出的 UMR 原始结果（7 段）| 登记 | `data/s3_retarget/umr_cpu@d1cdaed/_dev` | P0 | 被 `provenance.human_slots` 引用 |
| `out/` | CPU 工厂输出（`factory_npz_v1`，11 个）| 登记 | `data/s3_retarget/umr_cpu_factory@snapshot` | P0 | 搬箱片段受 bug 影响，打包时排除 |
| `qa/` | 5 台机器人的 QA 报告 | 登记 → 以后搬 | `evals/2026-10-07_robot_qa_<r>/` | P5 | 布局要先统一 |
| `qa_renders/` | SMPL-X 转换质检截图 | 归档 | `evals/2026-10-04_smplx_conversion_qc` | P9 | |
| `diversity/` | 多样性普查（`REPORT.md` 已出）| 登记；脚本进 git | `data/s4_dseval/2026-10-07/` · `src/s4_dseval/legacy/` | P0 / P1 | |
| `audit/` | 审计接口约定 | 复制规格 | `specs/api/audit_contract_v1.md` | P2 | 以后有了 jsonl 就导入账本 |
| `logs/` | 下载、转换日志和 pid | 链入 | `services/logs/factory_legacy` | P0 | `dl_loop.log` 有 406MB，建议加轮转；过期的 pid 文件进删候选 |
| `pilot/` | 早期诊断和临时文件 | 链入 | `scratch/2026-10-04_pilot` | P0 | 14 天后进删候选 |
| `hf_cache/` | HF 元数据缓存 | 登记 | `vol:hf` 缓存 | P0 | |
| `workflows/` | Claude 工作流脚本 | 进 git | `src/arena_motion/workflows/` | P1 | 写死了临时目录路径，P3 修 |
| `scripts/`、`scripts_grail/`、`scripts_streamc/` | 第一代、GRAIL、HiPHI 下载脚本（在跑）| 进 git | `src/s1_ingest/legacy_scripts/` | P1 | 在跑的进程照旧用原路径 |
| `DATA_SPEC_v2.md`、`RESOURCE_LIMITS.md` | 规格和资源上限 | 复制 | `specs/`、`specs/ops/` | P0 | 原文件改成指向新位置的软链（P3）|
| `platform_design/` | 本设计和四份地图 | 复制 | `docs/decisions/2026-10-07_platform_design/` | P0 | |

#### E.3.2 家目录下的代码目录（都没有 git）

| 现在的位置 | 一句话 | 动作 | 新位置 | 阶段 | 备注 |
|---|---|---|---|---|---|
| `~/umr_factory_code` | CPU 参考工厂和量尺 | 进 git + subtree | `engines/umr_cpu` | P1 | 代码冻结；`objects.py` 的 bug 在 P4 修 |
| `~/umr_gpu` | GPU 生产引擎 | 进 git + subtree | `engines/umr_gpu` | P1 | 对拍过，不能动 |
| `~/gpu_convert` | GPU 版 BVH → SMPL-X | 进 git + subtree | `engines/smplx_convert_gpu` | P1 | `runs/bench*`（5G）进删候选；`.inductor_cache` 当缓存 |
| `~/umr_dataset` | 打包器和加载器 | 进 git + subtree | `engines/umrs`；`data/` → `data/s5_packs/umrs_dev@v0` | P1 | P4 修 HiPHI 路径 |
| `~/motion_audit_web` | 审计工作台 + MCP 雏形 | 进 git | 前端 → `web/`；后端和 MCP → 并进 `src/service/` | P1 / P2 | `cache/` 3.1G 是可再生缓存 |

#### E.3.3 `/var/tmp/alphamotion`（352G，根分区快满，**不再写新东西**）

| 现在的位置 | 一句话 | 动作 | 新位置 | 阶段 | 备注 |
|---|---|---|---|---|---|
| `umr_factory/umr/UMR` | UMR 上游代码（d1cdaed）| 登记 → 以后搬 | `third_party/UMR@d1cdaed` | P5 后 | 没跟踪的运行时缓存挪到 `data/` |
| `umr_factory/umr/env` | 工厂共用的 Python 环境 | 重建 | `envs/umr`（在 /home 上按锁文件重建）| P5 后 | 旧环境保留一个月 |
| `umr_factory/umr/tools` | `umr_run.py` 等包装 | 进 git | `engines/umr_cpu/tools` | P1 | |
| `umr_factory/umr/{configs,eval,runs,logs,videos}` | UMR 侦察期实验 | 归档 | `archive/2026-10_umr_scout` | P9 | |
| `umr_factory/robots/` | 机器人登记表（34 台）| 导入账本 → 搬 | 账本 `robots` + `data/s2_robots/_registry/` | P0 / P7 | |
| `umr_factory/umr_gpu/` | GPU 引擎产物 | 登记，然后拆开 | `problems/` 可再生；`results/parity` → `evals/engine_parity`；`export` → `data/s3_retarget/umr_gpu@…`；`robots`（重复的那份）→ 删候选；`corr` 待查 | P0 / P7 | 正在写 |
| `umr_factory/labels/` | 点云接触标签器原型 | 归档 | `archive/2026-10_pcl_contact` | P9 | 已被取代 |
| `umr_factory/sources/` | license 和数据源侦察 | 导入账本 + 搬 | `data/s1_raw/_sources/` | P0 / P6 | `pinned` |
| `umr_factory/uniml3d_reload/` | UniML3D 原型和样本缓存 | 元数据搬；缓存删候选 | `data/s1_raw/_sources/uniml3d` | P9 | |
| `umr_factory/lloyd_gpu_eval/` | 第一次 GPU 和 CPU 的对比 | 搬 | `evals/2026-10-04_first_gpu_parity` | P3 | 被 `run_parity.py` 写死 |
| `umr_factory/{hiphi2smplx,h2s_shim,beta_fit}` | HiPHI 转换用的第三方代码和先验 | 搬 | `third_party/hiphi2smplx` · `data/assets/human/` | P3 | 和 `~/Downloads/SMPLX_NEUTRAL.npz` 放在一起 |
| `topo_exp/cache_*`（承重的那几个）| 现役训练缓存，跨目录软链拼起来 | **登记，不搬** | `legacy_datasets` + `data/s5_packs/legacy/` 链入 | P0 | 被 141 处以上引用 |
| `topo_exp/cache_lafan_v2`、`cache_amass_4000` | 坏链接 | 删候选 | — | P9 | |
| `topo_exp/{cache_paired*,cache_big,cache,v5_keycache,cache_xwbc_lafan_fixed,cache_amass_v3}` | 8 月旧缓存 | 原地归档 | `archive/` 账本记录 | P9 | `cache_paired_fixed` 还有引用 |
| `topo_exp/runs/` | 训练 run（v0.2 血统）| 导入 + 登记 | 账本 `runs`（`imported`）| P0 | 不在血统上的进归档计划 |
| `topo_exp/{bridge_codes,text_align}` | C1 码、文本层 | 登记（承重）| `legacy_datasets` | P0 | |
| `topo_exp/{qwen3,siglip}_names.pt` | 关节名向量 | 复制 | `data/assets/embeddings/` | P6 | 名字带 `.bak` 和 `.clobbered` 的进删候选 |
| `topo_exp` 其余（`encoder_bakeoff`、`night2_lib`…）| 杂项 | 原地归档；`night2_lib` 承重 | — | P9 | |
| `xwbc_full/` | 129G 压缩包，可以重新下载 | **删候选** | — | P9 | 先记下 HF 仓库名和 sha |
| `xwbc_lafan/extracted` | LAFAN 解包（承重）| 登记 | `data/s3_retarget/xwbc@v1/lafan` 链入 | P0 | |
| `any_body/` | 7 月早期阶段 | 环境 → 删候选；其余归档 | `archive/2026-07_anybody` | P9 | |
| `hf_cache/`、`hf_home/` | HF 缓存 | 登记；`hf_home` 删候选 | `vol:hf` | P0 / P9 | |
| `am2exp/` | AM2 Stage-0 | 登记 + 导入 | run → `runs`（`imported`）；`eval`、`videos` → `evals/2026-10-04_am2exp_stage0`；`surface_v2` → 归档 | P0 | lloyd 还没看过它的三联视频 |
| `ctrl_verify/` | WBC 跟踪参考轨迹 | 登记 | `data/wbc/ctrl_tracks@v1` | P0 | |
| `config_harvest/` | GMR 等外部配置（承重，30 + 19 处引用）| 登记 + 链入 | `third_party/GMR`、`third_party/harvest/*` | P0 / P3 | |
| `robot_models/` | 机器人模型来源 | 登记 + 链入 | `data/s2_robots/_sources/` | P0 | |
| `omniretarget/` | 对比基线 | 登记；重复的 zip 删候选 | `third_party/holosoma` | P9 | |
| `gmr_production/` | Native policy 对照臂的数据 | 登记 | `data/s3_retarget/gmr@v2` | P0 | |
| `gmr_production_v1/` | 旧版 | 删候选 | — | P9 | |
| `gmr_spike/`、`gmr_upstream/` | GMR 调参和上游代码 | 登记 | `third_party/GMR/{tuned_ik,upstream}` | P0 | |
| `gem_runtime*`、`gem_text`、`genmo_*`、`tmr_bridge` | 视频和文本转动作的桥 | 登记（休眠）| `third_party/gem` | P0 | |
| `studio/`、`studio_app.py` | 0814 产品前端（没在跑）| 归档 | `archive/2026-08_studio` | P8 | 它的 worker 协议可以参考 |
| `beyondmimic/` | 旧副本 | 删候选 | — | P9 | |
| `dex-retargeting/`、`anytop/` | 外部仓库 | 链入 | `third_party/` | P0 | |
| `xwbc_clips/`、`xwbc_dl/` | 早期样例 | 删候选 | — | P9 | |
| `spatial_recipe_ablation_20260816/` | 报告 | 搬 | `docs/reports/2026-08-16_spatial_recipe_ablation` | P6 | |
| `boneseed_spatial_*`（20 个）| 08-16 审计 | 归档（合成 1 个）| `archive/2026-08-16_boneseed_spatial` | P9 | |
| `mesh_viz/`、`ab_test/` | 对照图、A/B 指标 | 归档 | — | P9 | |
| `pose_translator_2`（软链）| → artifacts | 账本取代 | — | P9 | |
| 根目录约 70 个散文件 | 队列脚本、日志、视频 | 归档；`xwbc_joint_names.json` 搬走 | `data/s1_raw/_sources/xwbc/` | P9 | |

#### E.3.4 `~/tmp`（56G）

| 现在的位置 | 一句话 | 动作 | 新位置 | 阶段 | 备注 |
|---|---|---|---|---|---|
| `cl_v3/` | 接触标签 v3（承重，31G）| **登记，不搬** | `labels:contact@v3` | P0 | 其中 `amass/` 15G 只被旧缓存用，等确认后进删候选 |
| `sole_offsets*.json` | 脚底偏移表 | 搬 + 链 | `data/s2_robots/<r>/…/sole_offset.json` | P6 | 再按 B.2 重新生成 |
| `gauge_check_0903.json`、`dialect/` | gauge 检查、人体方言 Q | 搬 + 链 | `data/assets/human/dialect_Q@v1` · `evals/gauge/` | P6 | |
| `refiner_suite/` | 精修器评测套件（13 处引用）| 搬 + 链 | 代码 → `repos/research`；输出 → `evals/refiner/` | P6 | |
| `cl_v2_qc/`、`contact_qc/` | 接触质检（承重）| 登记 | `evals/contact_qc/` 链入 | P0 | |
| `ab_sole0/1` | 脚底 A/B | 归档 | — | P9 | |
| `greenwich_workbench_0911/` | datagrid 服务的运行状态 + 几份代码副本 | 状态搬；代码副本进删候选 | `services/datagrid/state/` | P6 | 先确认没有服务在用 |
| `arena_handoff_backup/` | datagrid 数据库唯一的备份 | 搬 | `services/datagrid/backups/`（`pinned`）| P6 | |
| `ufo_resume_drill/`、`ufo_h2_smoke/` | 演练和冒烟检查点，13.5G | 删候选（配置留到 `runs/wbc_ufo/_pilots`）| — | P9 | |
| `ufo_p*`、`probe*`、`cache/`、`torchinductor_*` | 试跑配置、编译缓存 | 配置归档；缓存删候选 | — | P9 | |
| `bfm_data_clone/` | 重复的仓库副本 | 删候选 | — | P9 | |
| `h2_bfm`、`h2bm`、`h2gr*`、`h2hd`、`h2mp`、`g1gr`、`*_rs` | H2/G1 的 WBC 中途视频 | 搬 | `runs/wbc_mjlab/h2/_media` | P9 | |
| `teleop2/`、`ee_*`、`ego_fill`、`*_0905/` | 遥操、手部 SE(3) 补全 | `ee_data` → `data/s1_raw/egodex`；其余归档 | — | P9 | |
| `wbcready_*`、`x2phys/`、`stress*`、`crossrig*`、`slipvid_*`… | 8 月底到 9 月初的审计和视频 | 归档（合成按日期分的包）| `evals/archive/2026-08-29..09-04/` | P9 | 被报告引用的单独登记 |
| 194 个散 mp4、约 1,240 个散 json/log/sh/py | 散产出 | 有引用的搬走；其余打成一个包归档 | `archive/tmp_2026-08_09.tar.zst` | P9 | `ablation_table.csv`、`scaling_*.csv`、`stress_run.py`、`qpos_to_rs.py`、`xwbc_to_robotstate.py` 有引用 |
| 152 个 wandb 临时目录 | 空的残留 | 删候选 | — | P9 | |

#### E.3.5 其它家目录

| 现在的位置 | 一句话 | 动作 | 新位置 | 阶段 |
|---|---|---|---|---|
| `~/alphamotion_artifacts/pose_translator_2` | PT2 产出，11G | 原地归档 | `archive/2026-08_pt2`（账本）| P9 |
| `~/alphamotion_artifacts/x2_pool_v3` | X2 训练池（承重）| 登记 | `data/wbc/x2_pool@v3` | P0 |
| `~/alphamotion_artifacts/x2_pool` / `x2_pool_v1_broken` | 旧版 / 有 bug 的版本 | 归档 / 删候选 | — | P9 |
| `~/alphamotion_artifacts/data_contract_audit_20260817` | 数据合约审计 | 归档 | `evals/archive/` | P9 |
| （缺失）`horizon_x2_hz4` | 代码引用了，但目录不存在 | 账本标 `missing` | — | P0 |
| `~/alphamotion_ctrl` | mjlab WBC 仓库 | 链入 | `repos/ctrl`；`motions/` → `data/wbc/motions@v1`（登记）；`logs/` → `runs/wbc_mjlab`（导入）；`.venv` → `envs/ctrl` | P0 |
| `~/alphamotion_ctrl/.git`（14G）| 仓库膨胀 | 建议仓库主人做 `git gc`（约收回 9G）| — | 等 lloyd |
| `~/reference_code/{ufo,sonic,bfm_zero,beyondmimic}` | 外部参考 | 链入 | `third_party/` | P0 |
| `~/reference_code/ufo` 里我们自己写的配置 | 没提交，上游一更新就会丢 | **先提交** | `repos/ctrl/bfm_zero_overlay/` | P1 |
| `~/reference_code/ufo` 多出来的那套 venv | 7.5–10G | 删候选 | — | P9 |
| `~/Downloads/SMPLX_NEUTRAL.npz` | 人体模型（被写死）| 复制 + `pinned` | `data/assets/human/` | P3 |
| `~/.cache/huggingface` | pretraining_foundation 下载的数据本体 | 登记 | `vol:hf` | P0 |
| `~/codex_exports` | 路演材料 | 不进根 | `~/arena_business/` | P9 |
| `~/alphamotion_cold` | 空目录 | 删候选，或者以后当冷存储挂载点 | — | P9 |
| `~/alphamotion2_runtime`、`~/alpha-motion-workspace`、`~/.alphamotion_workspace` | 指向主仓的软链 | 保留（兼容，服务靠它们启动）| — | — |
| conda 环境 `ardy`、`rot`、`genmo-gem` | 在用 | 登记 | `envs/`（锁文件）| P0 |

#### E.3.6 `~/Desktop`

| 现在的位置 | 一句话 | 动作 | 新位置 | 阶段 | 备注 |
|---|---|---|---|---|---|
| `"be water, robot"`（带引号，85G）| 研究主仓 | 链入 | `repos/research`（以后改名，旧路径留软链）| P0 | 3 个 systemd 服务靠它启动 |
| 其中 `alpha_motion2/pretraining_foundation` | 预训练底座（下载器在跑）| 代码进 git；索引导入账本 | `legacy_datasets` | P1 | 下载器跑完前不动 |
| 其中 `alpha_motion2/research/*` | 规模实验、试训、审计 | 代码进 git；run 导入；报告搬走 | `runs`（`imported`）· `docs/reports/` | P1 / P6 | `megatro_` 那个拼错的目录合并后进删候选 |
| 其中 `alpha_motion2/external/*` | GMR、UMR、UniMate、GRAIL（有 3,465 处本地改动）等 | 链入 | `third_party/<名字>@<commit>`；GRAIL 的改动导成补丁 | P1 | GMR 被写死了 19 + 6 + 10 处 |
| 其中 `alpha_motion2/demo/` 被服务读的 2 个子目录 | motion-lab 和 gallery 的数据 | 先保持原样 | — | P8 后归档 | |
| 其中 `alpha_motion2/{runs,demo 其余,web,scripts,synergy,live,…}` | 7 月的演示栈 | 归档 | `archive/2026-07_ardy_sonic_stack` | P9 | |
| 其中 `alpha_motion2/{evaluation,geometry,heads,losses,models,modules,training}`、13 个预览目录 | 全是 0 字节的空壳 / 一次性预览 | 删候选 | — | P9 | |
| 其中 `presentation`、`market_research`、`pitch_deck_*`、`assets`、`share_struct.md`、`nvidia_inception.md`、外联名单 | 商业资料 | 不进根 | `~/arena_business/` | P9 | |
| 其中 `pose_translator_design/exp_topo` | Greenwich 全系代码 | subtree | `engines/greenwich_v02`；判官和渲染器并进 `s7_eval`、`viz` | P1 / P4 | 被 Studio 和 arena 写死引用 |
| 其中 `pose_translator_design/verdicts`、`MODEL_VERSIONS.md`、`RUNS_INDEX.md` | 判决、台账 | 导入 | `docs/verdicts/` · 账本 `models` / `runs` | P0 | |
| 其中 `GENMO/` | 第三方，以可编辑方式装在环境里 | 登记，不搬 | `third_party/GENMO` | P0 | |
| 其中 `experiment/{ardy,kimodo}` | 第三方 | 链入 | `third_party/` | P0 | `kimodo/.venv` 5.3G 进删候选 |
| 其中 `notebook`、`note`、`experiment/stage1`、`any_body` | 已作废的路线 | 归档 | `archive/2026-05_stage1` 等 | P9 | |
| 其中 `pytest-of-arenalabs`、`*.pid`、空目录 | 垃圾 | 删候选 | — | P9 | |
| 其中 `.claude/skills` | spatial-eval、motion-eval 两个技能 | 复制 | 根目录 `.claude/skills` | P0 | |
| `be water, robot`（不带引号，557M）| 误建的孪生目录 | 先救出 3 样东西（`SCALING_LAW.md`、另一版 `CTRL_VERIFY_GR3.md`、x2 视频）| `docs/notes/` | P0 救；P9 删候选 | |
| `AlphaMotion/` | 产品仓 | 链入 | `repos/product`；`weights_staging` → `data/releases/alphamotion@0.2`（`pinned`）| P0 | |
| `arena-datagrid/` | 作业层、判官打包版 | 链入；设计并进 `core/jobs` 和 `s7_eval` | `repos/datagrid` | P0 / P2 | |
| `alphamotion video/` | 展示视频 + 00–35 号报告 | 链入 | `docs/reports/`（保留编号）| P0 | |
| `pose_translator_2/` | PT2 代码 | 归档 | `repos/pt2`（只读）| P9 | |
| `greenwich_eval_20260826/` | 0826 评估 + :8899 服务 | 归档 | `evals/archive/2026-08-26_greenwich` | P8 | |
| `_backup_before_delete_20260814/` | IsaacLab 改动备份 | 归档 | — | P9 | |
| `mingrun/` | 组员脚本 | 链入（只读）| `repos/contrib/mingrun` | P0 | `bones_seed` 是坏链，记一条 issue |
| `IsaacLab*`、`rot*`、`SegDepthFusion`、`openarm*`、`claw-code`、`droctavius`、`src`、`Manipulation_demo`、`lfs` | 其它项目 | 不进根 | 原处 | — | |
| 散文件：演示视频、babel 压缩包 | | 搬 / 链入 | `docs/reports/media/` · `data/s1_raw/babel/` | P6 | `github-recovery-codes.txt` 不要碰 |

### E.4 不能出错的几件事

- **`cl_v3` 不搬**。`cache_lafan_v3` 和 `cache_amass_lconv_tfix` 里有 34 个链接指向它。真要搬，就要同时改写这些链接，并用训练加载器读一遍做校验。没有必要冒这个险。
- **旧缓存是承重的**。`cache_xwbc_amass_full`（61G）看起来像旧版，其实被现役缓存链接着，不能删。
- **两个服务端口要先合并再停**：motion-lab 读 `demo/ardy_preflight_20`，gallery 读 `demo/alphamotion_verified_g1_atlas_20`。
- **改路径时被写死的地方**（P3 全部改完，并用 `am legacy refscan` 复查出来是 0）：
  - `umr_factory_data` 里 19 处指向 GMR；
  - arena-datagrid 6 + 4 处；
  - Studio 7 处；
  - 主仓里 141 处以上指向 `/var/tmp/alphamotion/topo_exp`。
- **HiPHI 下载有截止时间**：截止前不要做任何会拖慢它的事，包括大规模搬家和大量哈希计算。

---

## F. 第一批工作包（文件互不重叠，可以并行）

**共同规矩**：
- 只写 `~/arena_motion/` 下的文件；旧位置一律只读；不用 GPU；
- 重命令一律加 `taskset -c 0-15 nice -n 10 timeout …`；
- 每个工作包自带测试，验收以 `pytest -q tests/<模块>` 全绿为准；
- CLI 子命令写在各模块自己的 `cli.py` 里，由 `cli.py` 自动发现，所以不会抢同一个文件；
- 账本迁移文件的编号事先分好，每个工作包用自己那一段。

| WP | 内容 | 只改这些文件 | 依赖 | 工时 | 验收 |
|---|---|---|---|---|---|
| WP-0 | 骨架：配置、URI、CLI 自动发现、lint | `pyproject.toml` · `am.toml` · `CLAUDE.md` · `.gitignore` · `src/arena_motion/{__init__,cli}.py` · `core/{config,uri}.py` · `tests/core/test_uri.py` · `tools/lint_no_abs_paths.py` | 无 | 0.5 天 | `am --help` 能用；三种 URI 都能解析；lint 能拦住写死的路径 |
| WP-1 | 账本：表结构、迁移、只读查询 | `core/catalog.py` · `core/schema/0001_init.sql` · `core/queries.py` · `tests/core/test_catalog.py` | WP-0 | 1 天 | C.1 的表全部建好；约束生效（判官 import 训练代码时拒绝写入；审计判决只能由人写）；查询有单元测试 |
| WP-2 | 旧目录登记、`am what`、README 生成 | `s5_store/{register,fingerprint}.py` · `legacy/maps_import.py` · `index/{readme,templates/*}` · `tests/s5_store/test_register.py` | WP-1 | 1 天 | 四份地图的表全部导入 `legacy_paths`；`am what ~/tmp/cl_v3` 能说清楚；README 生成出来 |
| WP-3 | 作业层和调度器 | `core/{jobs,governor,approvals}.py` · `src/arena_motion/worker.py` · `core/schema/0100_jobs_extra.sql` · `tests/core/test_jobs.py` | WP-1 | 1.5 天 | 租约、幂等、外部作业都能用；`net=down` 会暂停；CPU 重活 6 个、GPU 重活 1 个、连接 16 个的上限有测试；代理不能批准 |
| WP-4 | 现成清单导入器 | `s1_ingest/importers/{grail_manifest,hiphi_logs,omni,lafan,licenses}.py` · `s2_surface/importers/robots_json.py` · `s5_store/importers/{umrs_catalog,pf_index}.py` · `s6_train/importers/{model_versions,topo_runs}.py` · `tests/importers/` | WP-1 | 1.5 天 | GRAIL 导入 22,312 行，34 台机器人，pretraining_foundation 索引行数对得上；导两遍不重复 |
| WP-5 | `motion_v3` 格式（读 v1 / v2，写 v3）| `formats/motion.py` · `specs/formats/motion_v3.md` · `tests/formats/test_motion.py` | WP-0 | 1.5 天 | 只读 11 个 CPU 样本和 12 个 GPU 样本，转换往返在容差内 |
| WP-6 | 统一阈值和 J-geom 判官 | `specs/data_spec_v2.thresholds.json` · `s7_eval/judges/geom/*` · `tests/s7_eval/test_geom.py` · `tests/s7_eval/test_import_graph.py` | WP-5 | 2 天 | 复现 `parity_tables.md` 的数字；import 关系测试通过；立方体物体测试能抓出那个"箱子当球"的 bug |
| WP-7 | 服务骨架（只读 HTTP 和 MCP）| `service/{http,mcp,auth,files}.py` · `specs/api/{openapi.v1,mcp_tools.v1}.json` · `tests/service/` | WP-1（查询接口可以先用桩）| 1.5 天 | 冻结的接口快照测试通过；代理角色批准会被拒；视频能按字节范围取 |
| WP-8 | 场景包和常驻查看器 | `viz/{bundle,live}.py` · `specs/formats/scene_bundle_v1.md` · `tests/viz/test_bundle.py` | WP-5 | 2 天 | 一个包同时显示人体、参考、模型三路；FK 算出的表面点和 `kin.npz` 一致（≤ 0.1 mm）|
| WP-9 | 三联视频和 12 帧网格统一 | `viz/render/{triptych,grid12,encode,scene3d}.py` · `tests/viz/test_render.py` | WP-5 | 1.5 天 | 逐帧时间戳校验通过；12 帧网格包含首帧和末帧；布局固定为人体 \| 对照 \| 本臂 |
| WP-10 | 分类法和覆盖统计 | `specs/taxonomy/taxonomy_v1.yaml` · `s4_dseval/{taxonomy,coverage}.py` · `tests/s4_dseval/` | WP-4 | 1.5 天 | 小时数守恒；每段正好 1 个主类；和今天的 `REPORT.md` 对账，差异能逐条解释 |
| WP-11 | 写死路径扫描器（只读）| `legacy/refscan.py` · `tests/legacy/test_refscan.py` | WP-1 | 0.5 天 | 输出每个旧路径被哪些文件引用、各几处，写进 `legacy_paths.hardcoded_refs` |

**顺序**：
1. 第一天：WP-0；
2. 第二天：WP-1、WP-5 并行；
3. 之后：WP-2、3、4、7、11 并行，WP-6、8、9 跟在 WP-5 后面，WP-10 跟在 WP-4 后面。

**第一周的交付**：
- 有账本，能回答"每个文件夹是什么"；
- 有 README 总索引和 `am what`；
- 有只读的网页接口和 MCP；
- 有统一的数据格式和统一的阈值。

---

## 风险和还没定的事

1. **账本和现实对不上**。用三个办法防：
   - `store.audit_layout` 每天扫一遍；
   - 所有写操作都只能走作业；
   - CI 拦住写死的路径。

   最危险的是人或代理手工往 `~/tmp` 里写东西。所以 `CLAUDE.md` 和技能文件都要写明：临时文件只能放 `scratch/`。
2. **GPU 引擎的不确定性**：同一个输入重跑，哈希可能不同。所以重定向结果的身份用"输入 + 引擎 commit"来定，内容哈希只用来检查文件有没有坏，不用来判断"是不是同一个结果"。
3. **对应关系（`corr`）两份不一致**还没查清。在查清之前，GPU 引擎量产出的结果都要记下自己用的是哪份。
4. **license**：HiPHI 和 BONES 的条款可能卡住发布权重这件事，要法务结论。在那之前，训练包的默认策略是"只做研究"。
5. **SQLite 写入争用**：worker 一多，写账本可能会排队。缓解办法是：逐帧数据不进 SQLite；状态更新按批提交。真不够了再考虑换 Postgres，datagrid 已经有 Postgres 运行态可以借用。
6. **磁盘**：HiPHI 剩下的部分要下完，加上量产重定向，/home 会很紧。所以 npz 只当中转，必须严格执行；`xwbc_full` 能不能删，直接决定根分区还有没有余量。

---

## 附录 1：作业类型总表

| 作业类型 | 环节 | 资源级别 | 要批准 | 调用什么 |
|---|---|---|---|---|
| `ingest.register` | S1 | light | 否 | 源适配器 |
| `ingest.download` | S1 | net_heavy（调度器分连接数）| 否 | `hiphi_staged_dl`、`grail_dl_loop`、`acquire.py` |
| `ingest.convert` | S1 | gpu_heavy / cpu_heavy | 超过 4 小时要 | `smplx_convert_gpu`、`convert_grail`、`convert_omni` |
| `ingest.gauge`、`ingest.human_qa` | S1 | cpu_heavy | 否 | `gauge`、`smplx_qa`、`obj_pen_audit` |
| `robot.import/stage/surface/sole_offset/corr/corr_diff/contact_sheet` | S2 | cpu_heavy | 否 | `umr_cpu` 的 stage、surface、corr |
| `retarget.plan` | S3 | light | 否 | |
| `retarget.capture` | S3 | cpu_heavy | 否 | `umr_cpu` 抓问题 |
| `retarget.solve` | S3 | gpu_heavy | 超过 4 小时要 | `umr_gpu` + v2 严格处理 |
| `retarget.export` | S3 | cpu_heavy | 否 | `formats.motion` |
| `retarget.parity` | S3 | gpu_heavy | 否 | `umr_gpu/eval` |
| `judge.run` | S3 / S7 | cpu_heavy | 否 | `s7_eval/judges/*` |
| `dseval.label/coverage/quality/report` | S4 | cpu_heavy / light | 否 | `s4_dseval` |
| `audit.sample` | S4 | light | 否 | |
| `store.register/seal/verify/snapshot/audit_layout` | S5 | light（`ionice -c3`）| 否 | |
| `store.pack` | S5 | cpu_heavy | 否 | `engines/umrs` |
| `store.gc_plan` | S5 | light | 否 | |
| `store.gc_execute` | S5 | light | **是** | |
| `store.move` | S5 | cpu_heavy（磁盘读写）| 跨分区**要** | |
| `train.run` | S6 | gpu_heavy | **是** | am2、greenwich_v02… |
| `train.watch` | S6 | light | 否 | |
| `viz.bundle` | S6 / S7 | cpu_heavy | 否 | CPU 推理 + 读取 |
| `render.triptych`、`render.grid12` | S7 | cpu_heavy | 否 | `viz/render` |
| `eval.suite`、`eval.compare` | S7 | cpu_heavy | 否 | judges、protocol |
| `eval.wbc_track` | S7 | gpu_heavy | **是** | mjlab |
| `model.crown` | S6 | — | **只有 lloyd 能做** | |

## 附录 2：这次核对过的事实（供复查）

- 磁盘：`/` 剩 76G（88%），`/home` 剩 345G（70%）；可用内存 42G；`throttle.log` 18:29 记录网络正常。
- HiPHI 下载器：截止还有 35.9 小时，剩 145G 没下，全部下完预计 60 小时，按 16 个连接下。水位设置：/home 至少剩 200G，工厂目录最多 300G。
- 体积：
  - `factory_npz_v1` 2.62 MB / 5,771 帧；
  - `gpu_npz_v2` 1.89 MB / 5,771 帧；
  - UMRS 机器人分片 77.8 KB / 1,500 帧、106 KB / 600 帧；
  - UMRS 人体 1.97 MB / 9,731 帧。
- `robots.json`：34 台，其中核心 21 台。
- arena-datagrid 的 `jobs` 表有租约，`request_hash` 唯一；`reviews` 表限定 `reviewer_kind='human'`。
- `motion_audit_web/agent_tools/server.py` 已经有 15 个 MCP 工具，含 `workflow_plan` / `workflow_submit` / `job_status`。
- pretraining_foundation 的索引有 `datasets`、`sequences`、`body_assets` 三张表，以及按"见过 / 留出"划分的 `samples` 视图。
