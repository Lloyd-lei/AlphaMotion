# AlphaMotion 开发平台：最终设计（合并版）

- 日期：2026-10-07 晚。状态：**定稿草案，等 lloyd 拍板第 G 节的事项**。
- 依据：
  - 同目录 4 张地图：`map_factory.md`、`map_model_tools.md`、`map_research_repo.md`、`map_caches_artifacts.md`；
  - 两个候选方案：`design_catalog_first.md`（先记账）、`design_wrap_existing.md`（包住现有的）；
  - 18:48 前后的现场核对（只读）：`df`、`free`、`throttle.log`、进程表、`nvidia-smi`、`systemctl --user`、`hiphi_streamc_status.json`、`~/umr_gpu/README.md` 和 `scripts/run_factory.py`、`~/motion_audit_web/{configs/mcp.json, docs/MCP.md, agent_tools/}`、`diversity/REPORT.md`。
- 本文只是设计。除了写本文件和 `FOLDER_INDEX.md`，没有移动、删除、修改任何东西，没用 GPU。
- 现有每个文件夹是什么、产出了什么、以后放哪：见同目录 **`FOLDER_INDEX.md`**。

---

## 0. 一页看懂

**一句话**：建一个根目录 `~/alphamotion`（名字待定）。里面一个 git 仓放我们自己的代码，一个 SQLite 账本记下每样东西是什么、谁产出的、谁在用、能不能删。七个环节都是登记过的"作业"。同一套后端对 Claude 开 MCP，对 lloyd 和 Codex 开网页。大数据先原地不动，用软链挂进来，代码改成从配置读路径以后再搬。

**lloyd 以后找东西只看三处**：
1. 根目录 `INDEX.md`：每晚自动生成，一张表列出每个文件夹是干嘛的、谁产出的、多大、什么状态、能不能删。
2. 命令 `am what <任意路径>`：对任何旧路径或新路径，说出它是什么、谁产出的、谁在用、能不能删。没登记的会直说"没登记"。
3. 网页首页：七个环节的进度、正在跑的作业、磁盘、网络、等你批准的事。

**七个环节用什么工具**（细节见 B 节）：

| 环节 | 现成的（包住，不重写） | 新加的薄层 | 产出放哪 |
|---|---|---|---|
| S1 数据清洗 | HiPHI/GRAIL/Omni 下载器，`gpu_convert`（HiPHI→SMPL-X），`convert_grail` v3，`convert_omni`，`acquire.py`（BONES 等），`gauge_check`，license 原文 | 一个函数读 4 种 SMPL-X 存法；逐段 license；gauge 记录；时间轴检查 | `data/s1_raw`、`data/s1_human` |
| S2 表面点采样 | `umrf` 的 stage/surface/corr，`robots.json`（34 台，核心 21 台） | 机器人资产带版本和哈希；两份 corr 对比；脚底偏移从碰撞几何重算 | `data/s2_robots/<机器人>/` |
| S3 重定向 | `umr_gpu`（生产，和 CPU UMR 对拍过），`umr_factory_code`（参考和量尺） | 真实片段入口 `run_clips`；冻结版本跑生产；补 `.meta.json`；登记；进人审队列 | `data/s3_retarget/umr_gpu@<commit>/<批次>/` |
| S4 整库评估 | `diversity/`（今天的 `REPORT.md`），`action_labels.py`，`motion_audit_web` 人审台 | 分类法 v1、覆盖和稀缺清单、质量汇总、风险+随机人审、批次放行门 | `data/s4_dataset_eval/<日期>/` |
| S5 存储 | `umr_dataset`（UMRS v1 分片 + 加载器） | 账本、MANIFEST、磁盘水位、清理计划（只出清单） | `catalog/`、`data/s5_datasets/` |
| S6 训练 + 实时看 | `train_scaling`/`train_arm`（AM2），`train_dual`（Greenwich，冻结），产品仓 `LiveViewer` | 训练前估时长和电费、要批准；新 checkpoint 自动 CPU 推理出片；查看器读统一格式 | `runs/<族>/<run_id>/` |
| S7 评估 | J1 `ground_judge`、`umr_gpu/eval/judge.py`、arena `locomotion.v1`、J2 `closure_judge`、`v2_judge`、am2exp 统计层和三联渲染 | 判官合并成 7 个，阈值只读一份；冻结考卷；排行榜固定带"数据地板"基线 | `evals/<套件>@<版本>/<对象>/`、`media/` |

**今天就要 lloyd 拍板的 3 件急事**（其余见 G 节）：
1. **HiPHI 按现在的速度会赶不上截止。** 截止还剩 35.6 小时，还要下 144.5 GB；HiPHI 只分到 0.67 MB/s（整条链路 2.7 MB/s，其余被 GRAIL 的 48 个连接和 `acquire.py` 占着），预计 59.7 小时才下完。建议马上暂停 GRAIL 下载和 `acquire.py`，16 个连接全给 HiPHI（按 2.5 MB/s 估约 16 小时下完）。GRAIL 和 HF 上的东西没有截止，以后再下。
2. **`alphamotion-ardy-isaac` 还在崩溃重启**：18:48 时已重启 642 次，它要的 `~/Desktop/GR00T-WholeBodyControl` 已经不在。执行 `systemctl --user disable --now alphamotion-ardy-isaac`。
3. **GPU 现在同时跑着两个重任务**：`run_factory.py` 吞吐测试（B=1024，40.8 GB 显存）和 `gpu_convert`（11.2 GB），再加 6 个 GRAIL 转换 worker，利用率 92%，超过 70% 的目标。建议吞吐测试跑完前暂停 `gpu_convert` 的定时任务，以后由作业层保证"同一时间只有一个重 GPU 任务"。

---

## 1. 两个方案的评审，以及为什么这样合并

### 1.1 打分

| 维度 | catalog_first（先记账） | wrap_existing（包住现有的） | 最终取哪边 |
|---|---|---|---|
| lloyd 找东西 | 强：`am what` 任意路径都能答；每天扫"没登记的新目录"；README 自动生成 | 中上：人写 README + 机器生成 `INDEX.md` + 每个产出目录带 `MANIFEST.json`；没有对任意路径的反查 | 两边都要：人写 `README.md` + 生成的 `INDEX.md` + `am what` + 每日扫描 |
| 对拍过的引擎、在跑的作业的风险 | 中：新造 `motion_v3` 要写两个转换器、要改引擎导出；内容寻址的 blob 库要给文件做硬链接和去写权限 | 低：引擎一行不改、输出字节不动、旁挂 `.meta.json`；搬家前用 `/proc` 查谁在用 | 取 wrap 的做法；只在模型输出这一侧先写新格式 |
| 多快出第一批生产数据 | 慢：第 2 周以后 | 快：写的是第 2 天 | 实际是第 2 天出试点、第 3 天放开 21 台（见 1.3 第 3、4 条，两边都漏了一步） |
| 七个环节是否齐全 | 齐全，表结构最细（血统、审批、`viewed_by_lloyd`、判官不许 import 训练代码的约束） | 齐全，多了批次放行门、排行榜"没人看过视频不能标通过" | 表结构以 catalog_first 为底，补上 wrap 的批次、放行门、审阅 |
| MCP 和网页 | 工具全部新起名字 | 保留今天已经能用的 15 个 MCP 工具名，沿用"先出计划、再提交"的作业注册表 | 保留旧名字，按环节补工具 |
| 重启和断网 | 有 `boot_id` 和租约，没说服务怎么防崩溃循环 | 有断网和内存暂停，没说重启后作业怎么办 | 都不够，补一节（B.9） |
| 资源上限 | 强：`systemd-run --scope -p MemoryMax=20G` 加绑核 | 漏了"单进程树 < 20 GB" | 用 systemd 把上限交给内核执行 |
| 下载急事 | **对**：16 个连接先全给 HiPHI | **错**：建议 HiPHI 从 16 降到 8，会让它更赶不上截止 | 取 catalog_first |

### 1.2 从两边各拿了什么

**从 catalog_first 拿**：`am what` 和每日扫描；账本里的血统表、别名、审批表、`media.viewed_by_lloyd`、`judges.imports_training_code = 0` 约束、`audit_decisions.reviewer_kind = 'human'` 约束；旧大目录的抽样指纹（`ext:`，不用全量算哈希）；调度器用 systemd 限内存和绑核；场景只存 qpos 和接触、表面点由查看器用正向运动学现算；checkpoint 保留规则；排行榜固定两行基线；下载连接先给 HiPHI；账本快照另存一份到别的分区。

**从 wrap_existing 拿**：引擎不改、输出字节不动、旁挂 `.meta.json`；读取层统一格式；模型输出存成和数据一样的格式，数据用的查看器、判官、审计台直接拿来看模型；路径寻址加哈希（不做 blob 库）；保留现有 MCP 工具名和"先计划、再提交"；动手前用 `/proc` 查有没有进程在用；批次放行门（随机抽样合格率的置信下界）；排行榜一行没人看过视频就不能标"通过"；试点 4 台 × 100 段先给 lloyd 看视频再放开；清理只能 lloyd 在本机执行。

**两边都不要的**：
- 内容寻址 blob 库（`store/blobs/sha256/ab/cd/...`）：一台机器用不上，lloyd 也看不懂。
- 把引擎 `git subtree` 并进 monorepo：引擎保留各自的 git 仓，生产只从冻结版本的 worktree 跑（见 A.1、B.3），更简单，也不打扰还在改引擎的人。
- 把研究主仓改名：3 个服务和很多脚本靠旧路径启动，等服务合并完再说。

### 1.3 两边都漏了或说错的（这次现场核对出来的）

1. **HiPHI 下载会赶不上截止**（见第 0 节急事 1）。两边都只看到"连接数超了"，没算速度。按现在的分配，截止时还差约 60 GB 没下。
2. **对拍时用的引擎代码已经找不回来了。** `umr_gpu` 一直没有 git，而 README 写着 18:50 刚把 `project_frames` 改写成全局工作队列，"下一步重测"。也就是说代码在对拍之后已经改过，wrap 说的"打个 tag 钉住对拍时的版本"做不到。正确做法：现在打快照，**在这个快照上重跑一遍对拍**，通过了才算 `verified@<commit>`；生产只从这个 commit 的只读 worktree 跑，引擎作者在自己的目录继续改，互不影响。
3. **GPU 引擎还没有"读真实片段"的生产入口。** `scripts/run_factory.py` 只是吞吐测试：它读一份事先抓好的 LAFAN 舞蹈问题（`PROB`），再复制成很多"合成片段"；`run_one.py` 也只读抓好的问题文件。两边都写了"第 2 天调 run_factory 量产"，实际上要先补一个真实片段入口 `run_clips`（读账本里的 SMPL-X → 引擎已有的 `source_gpu` 做输入 → 求解 → 严格后处理 → 导出），并用"同一片段，`run_clips` 和 `run_one` 结果一致"做等价测试。还没弄清的一点：每台机器人的静态准备数据（绑定、T 姿态法线）是否能按共享 beta0 只抓一次，试点时实测。
4. **GPU 引擎只按平地处理地面。** README 写明所有输出以"校准后的人体地面 z=0"为准，代码里没有地形（楼梯、路沿、坡）的处理，测试片段也只有 LAFAN 舞蹈、HiPHI 起身、Omni 踢球三段。wrap 的首批量产写的是"GRAIL 地形"，不行。首批只放**平地**片段；地形片段先挂一条问题单，等一个 10 分钟的小测证明能把地形当物体喂进去再说。
5. **箱子问题**：CPU 工厂的 `umrf/objects.py` 把 8 顶点的箱子判成外接球；GPU 引擎的 `geometry.py` 有解析的 box SDF，大概率没这个问题，但没测过。量产搬箱数据前跑一个合成箱子小测（GPU < 10 分钟）。
6. **崩溃循环是服务配置造成的。** 现有 4 个 systemd 用户服务都是 `Restart=on-failure/always` 加几秒重启间隔，没有"多少次以后就停"的上限，所以 ardy-isaac 能重启 642 次。下载器和 GRAIL 循环是手动 `nohup` 起的，今天重启后要人手动拉起来。B.9 统一改成带上限的 systemd 单元。
7. **今天已经有一个能用的 MCP 和作业注册表**：`~/motion_audit_web/agent_tools/`（15 个工具）+ `configs/mcp.json`（每个工作流写明固定命令、代码文件哈希、设备、超时、输出上限、证据边界；重定向、训练、评估、仿真 4 个工作流故意关着，写了原因）。这就是作业类型注册表的现成格式，直接沿用。
8. **阈值统一时不能去改引擎。** 引擎严格后处理里的常数写在 `strict.py` 里。做法是：配置文件 `config/gauges/data_spec_v2.toml` 照抄引擎实际用的数，再加一个测试，断言"引擎里的常数 = 配置里的数"。引擎不动，以后谁改了一边，测试就红。

---

## A. 一个根目录

### A.1 目录树

根目录暂定 `/home/arenalabs/alphamotion`（在 /home 分区，剩 343 GB；根分区只剩 76 GB，不再往上写新东西）。环境变量 `AM_ROOT`，命令行叫 `am`，Python 包叫 `amp`。

```
~/alphamotion/
├── README.md        人写，不超过 2 页：七个环节各一句话、代码在哪、产出在哪、服务地址、常用命令
├── INDEX.md         自动生成（am index，每晚 07:00 和大变动后）：每个目录的用途、谁产出、大小、状态、能不能删
├── CLAUDE.md        给 Claude 和代理的规矩：只经 am 交作业；临时文件只放 scratch/；不许写死绝对路径
├── config/          唯一写路径、阈值、上限的地方（进 git）
│   ├── am.toml           根目录、存储池、端口、资源上限、磁盘水位
│   ├── paths.toml        所有外部和旧路径：GMR、UMR、各环境、SMPLX_NEUTRAL.npz、exp_topo……
│   ├── gauges/data_spec_v2.toml   DATA_SPEC v2 阈值，唯一一份
│   └── workflows/*.json  作业类型注册表（沿用 motion_audit_web/configs/mcp.json 的写法）
├── docs/            人写的文档（进 git）
│   ├── specs/            DATA_SPEC_v2.md、FORMAT_human_smplx_v1.md、FORMAT_am_motion_v3.md、FORMAT_umrs_v1.md、TAXONOMY_v1.md
│   ├── contracts/        audit_contract_v1.md、openapi_v1.yaml、mcp_tools_v1.md、manifest.schema.json
│   ├── ops/              RESOURCE_LIMITS.md、runbook.md（起停、开机检查、掉网怎么办）
│   ├── decisions/        lloyd 拍板记录，只追加：YYYY-MM-DD_<题目>.md
│   ├── maps/             这次 4 张地图、两个候选方案、本文、FOLDER_INDEX.md
│   ├── reports/          技术报告 00–35（来自 "alphamotion video/resource/report"，编号不变）
│   ├── verdicts/         历次判决，只追加
│   └── papers/
├── code/            我们自己的唯一一个 git 仓（monorepo）
│   ├── amp/              平台层：config/paths、catalog、formats、jobs（worker + 调度）、index、service（api + mcp）、cli
│   ├── s1_clean/         下载、转换、gauge、license 的包装
│   ├── s2_surface/       机器人登记、暂存、表面点、区域、对应关系、脚底偏移
│   ├── s3_retarget/      adapters/（run_clips、meta、登记）、tests/
│   ├── s4_dataset_eval/  分类、覆盖、质量汇总、人审队列、放行门
│   ├── s5_store/         打包（调 umr_dataset）、数据集计划、清理计划、磁盘报告
│   ├── s6_train/         训练入口包装、watcher（新 checkpoint 自动出片）、viz/live（查看器适配）
│   ├── s7_eval/          judges/、render/、golden/、protocol/、leaderboard/
│   ├── web/              统一网页（底座是 motion_audit_web 的 React/Three.js 前端，Codex 接着写）
│   ├── ops/              systemd 单元、throttle_guard、precheck
│   └── workflows/        Claude 工作流脚本（diversity_wf.js、robots_wf.js）
├── engines/         冻结的引擎运行副本：git worktree，只读，生产只从这里跑
│   └── umr_gpu@<commit>/ · umr_cpu@<commit>/ · smplx_convert_gpu@<commit>/ · umrs@<commit>/
├── repos/           有自己 git 历史的仓库（先软链，P5 再真搬）
│   ├── research → ~/Desktop/"be water, robot"     研究主仓（exp_topo、pretraining_foundation）
│   ├── product  → ~/Desktop/AlphaMotion           产品仓 v0.2（LiveViewer、发布门）
│   ├── datagrid → ~/Desktop/arena-datagrid        locomotion.v1 标准、作业表设计
│   ├── ctrl     → ~/alphamotion_ctrl              WBC / mjlab
│   ├── umr_gpu  → ~/umr_gpu                       引擎开发目录（作者继续在这里改）
│   ├── umr_cpu  → ~/umr_factory_code
│   ├── gpu_convert → ~/gpu_convert
│   ├── umr_dataset → ~/umr_dataset
│   └── contrib/mingrun → ~/Desktop/mingrun        组员，只读
├── third_party/<名字>@<版本>/   外部仓库，只读；本地改动放 patches/<名字>/*.patch；版本记在 LOCK.toml
├── envs/            锁文件 + 指向真实环境的软链（umr、genmo-gem、ardy、ctrl、ufo、mcp）
├── data/            数据，按环节编号；新产出是真目录，旧的是指向 stores/ 的软链
│   ├── s1_sources/       来源清单、license 原文、人体模型（SMPLX_NEUTRAL、beta_fit、方言 Q）
│   ├── s1_raw/<源>/      原始下载，只读
│   ├── s1_human/<源>/smplx@<版本>/   统一后的人体动作
│   ├── s2_robots/<机器人>/ + registry.json + _sources/
│   ├── s3_retarget/<引擎>@<commit>/<批次>/   重定向结果（中间格式，打包后可回收）
│   ├── s3_labels/<种类>@<版本>/     旧接触标签（cl_v3 等）
│   ├── s4_dataset_eval/<日期>_<主题>/
│   ├── s4_audit/         queue / decisions / issues.jsonl（人审接口约定不变）
│   └── s5_datasets/<名字>@v<N>/ + legacy/<旧训练缓存>（软链）
├── catalog/         catalog.sqlite（唯一的权威记录）+ backups/ + exports/*.jsonl（小表，进 git 看变化）
├── runs/<族>/<run_id>/            S6 训练产物
├── evals/<套件>@<版本>/<对象>/     S7 评估产物
├── media/<YYYY-MM-DD>_<主题>/      给人看的视频和图
├── releases/<产品>@<版本>/         发布物
├── services/<服务>/{state,logs}/   服务运行态（am-service、viewer、datagrid）
├── stores/          存储池挂载点（软链）：factory、sys、legacy_tmp、hf、cold
├── cache/           随时可删
├── scratch/<YYYY-MM-DD>_<任务>/    临时，14 天过期；代码不许引用这里
└── archive/<YYYY-MM>_<主题>/       冻结的旧路线，只读，每个目录带一行说明
```

**同一个编号 = 同一个环节**：`code/s3_retarget` 产出 `data/s3_retarget`；S6 产出在 `runs/`，S7 产出在 `evals/` 和 `media/`。

**不进根目录**：路演、融资、股权、外联名单 → `~/business/`；和 AlphaMotion 无关的项目（IsaacLab、rot、openarm 等）留在原处。

**进 git 的**：`README.md`、`CLAUDE.md`、`config/`、`docs/`、`code/`、`catalog/exports/`。其余（数据、账本本体、run、评估、环境、缓存）不进 git。

### A.2 大数据放哪、怎么引用

| 存储池 | 物理位置 | 放什么 | 规则 |
|---|---|---|---|
| `home` | `~/alphamotion/...`（/home） | 所有新产出 | 默认写这里 |
| `factory` | `~/umr_factory_data`（/home） | 现有工厂数据（208 GB，含 HiPHI 下载） | P0 软链挂进 `data/`；P5 按子目录 `mv`（同分区，秒级），旧位置留反向软链 |
| `sys` | `/var/tmp/alphamotion`（根分区，88%） | 旧训练缓存 `topo_exp`、`xwbc_*`、`config_harvest`、`robot_models`、UMR 环境 | **平台不再往这里写新东西**；小件生产资产 P5 拷到 /home |
| `legacy_tmp` | `~/tmp` | 有代码在读的旧件：`cl_v3`、`sole_offsets3.json`、`dialect/`、`refiner_suite/`…… | 原地登记；小件 P1 拷到 `data/` 并留软链 |
| `hf` | `~/.cache/huggingface` | `acquire.py` 下的 BONES、UniMate 等 | 原地登记 |
| `cold` | 待定（建议外接 ≥ 4 TB） | `archive/`、可重生成的、旧缓存 | 不在线时读取报"离线"，不能悄悄当成不存在 |

**引用规则**：
- 代码只用两种方式拿路径：`amp.paths.get("<paths.toml 键>")`，或 `amp.catalog.resolve("<账本 id>")`。
- 账本只记"哪个存储池 + 池内相对路径"。搬家只改账本，代码不用跟着改。
- `am paths lint` 扫描写死的 `/home/arenalabs`、`/var/tmp`、`~/tmp`、`Downloads`。新代码里出现就拦（CI 失败）；旧代码里的只报警，列进迁移清单。

### A.3 命名和版本

- 文件夹和 ID 一律小写 ASCII、下划线，**不许有空格、逗号、引号**（教训：`"be water, robot"` 和它的孪生目录）。日期一律 `YYYY-MM-DD`。

| 东西 | 格式 | 例子 |
|---|---|---|
| 数据源 | 固定词表 | `hiphi` `grail` `omnicontact` `bones_seed` `amass_xwbc` `lafan` `mixamo` `uniml3d` `babel` `arena_mocap` |
| 片段 | `<源>/<源内原路径去扩展名>`，镜像加 `~mirror`（默认不进训练） | `grail/terrain_curbs__curb_000__000` |
| 人体版本 | `<源>/smplx@v<N>`，带 fps 变体 | `hiphi/smplx@v2/fps30` |
| 机器人 | `robots.json` 的键 | `unitree_g1_29dof` |
| 机器人资产 | `<机器人>@<资产哈希前 8 位>` | `unitree_h2@3fa2c1d0` |
| 引擎 | `<引擎>@<git 短哈希>`；有没提交的改动就不许跑生产 | `umr_gpu@a1b2c3d` |
| 重定向 | `<片段>::<机器人>::<引擎>@<commit>` | `grail/...::fourier_gr3::umr_gpu@a1b2c3d` |
| 批次 | `<YYYYMMDD-HHMM>_<源>_<条数>` | `20261009-0900_hiphi_100` |
| 数据集（训练包） | `<名字>@v<N>`，选片条件冻结成 yaml | `am_pretrain@v1` |
| 训练 run | `<YYYY-MM-DD>_<族>_<短名>` | `2026-10-12_am2_B_M_seed0` |
| 模型 | `<族>-<版本>-<规模>`；**头衔**（Greenwich、Equator、Bumblebee……）是单独字段，只能 lloyd 加冕 | `greenwich-v2-L20k` |
| 控制器对照臂 | `alphamotion_policy` / `native_policy` | 沿用已有约定 |
| 冻结考卷 | `<名字>@v<N>` | `golden_quick@v1` |
| 评估 | `<套件>@<版本>/<对象>` | `geometry@v1/umr_gpu@a1b2c3d` |
| 媒体 | `media/<日期>_<主题>/<对象>__<种类>.mp4` | `.../C1__triptych.mp4` |

- **永不覆盖**：产物封存（`MANIFEST.json` 里 `frozen: true`）以后不再改。要修就出新版本，账本里记 `supersedes`。
- **"最新"只是别名**：`dataset:am_pretrain@latest → v3`，由人或作业显式切换。
- **每个产出目录**（`data/`、`runs/`、`evals/`、`media/`、`releases/` 下）都必须带 `MANIFEST.json`：账本 id、种类、版本、产出它的作业和命令、代码 repo 和 commit、输入 id 和哈希、参数哈希、license 汇总、规格版本、保留级别、一句中文说明。

### A.4 怎么防止再乱

- **`am what <路径>`**：登记过的，答"是什么、谁产出（作业/commit）、输入是谁、谁在用、保留级别、能不能删"；没登记的，答"未登记"，并给出最近的已登记上级目录。
- **`INDEX.md`**（由账本和各目录 `MANIFEST.json` 生成），分几块：
  - 现在在跑：作业、环节、进度、预计结束、资源、谁提交的；
  - 磁盘：各分区剩余、底线、各类配额用量；
  - 顶层目录：每个干什么、现有内容、最近一次产出；
  - `data/` 每个数据集一行：是什么、大小、谁产出、日期、状态、license、被谁用；
  - 还在旧位置的（迁移账本）：旧路径、登记成什么、状态、计划；
  - 没登记的新目录：路径、发现时间、大小、建议；
  - 待你处理：批准请求、清理计划、新的严重问题。
- **每日扫描**（`am doctor`，每天 07:00）：没登记的新目录、坏软链、账本里有但盘上没有的、过期的 `scratch/`、超配额的。结果写进 `INDEX.md` 顶部。
- **CLAUDE.md 和技能文件写明**：临时文件只能放 `scratch/<日期>_<任务>/`；不许往 `~/tmp`、`/var/tmp` 写新东西。最容易让账本和现实对不上的，就是人或代理手工往 `~/tmp` 里写东西。

---

## B. 七个环节

### B.0 公共约定

- 命令行统一 `am <环节> <动作>`（如 `am s3 plan`）。MCP 工具和 HTTP 接口都调 `code/amp/service/core.py` 里同一个函数，三个入口没有各自的业务逻辑。
- **写操作一律两步**：先 `plan`（估时长、磁盘、GPU 小时、电费，返回计划 id），再 `submit(plan_id, 幂等键)`。同一个请求交两次拿到的是同一个作业。这是今天 `agent_tools/jobs.py` 已经实现的做法。
- 每种作业在 `config/workflows/<kind>.json` 登记：固定命令模板、代码文件列表（提交时核哈希，代码变了计划作废）、参数 schema、设备（cpu/gpu/net）、资源级别、超时、输出上限、是否要批准、是否可在重启后续跑、证据边界。**MCP 不提供任意 shell。**
- 环节之间只通过账本 id 交接，不靠"约定好的目录"交接。

### B.1 S1 数据清洗（下载 → SMPL-X → 统一 gauge → license）

**现成的（包住）**：
- 下载：HiPHI `scripts_streamc/hiphi_staged_dl.py`（带磁盘水位、已按"每 GB 带来多少稀缺动作"重排包顺序）；GRAIL `scripts_grail/grail_dl_loop.py`；Omni `scripts/dl_omni.py`；BONES/UniMate 等 `pretraining_foundation/acquire.py`。
- 转换：HiPHI 用 `gpu_convert/convert.py`（生产版，和 CPU 版 `parity40` 对过）；Omni 用 `scripts/convert_omni.py`；GRAIL 用 `convert_grail.py` v3 + `floorcal2.py` + `grail_collision.py`。
- gauge：`exp_topo/gauge_check.py`、xwbc 时间轴修复、人体方言 Q（`~/tmp/dialect/Q_final.pt`）。
- license：`/var/tmp/.../umr_factory/sources/sources_inventory.json`（逐源）、`raw/*/LICENSE.md`、GRAIL `manifest.jsonl`（22,312 行，逐段 license，是全平台唯一一份逐段清单，当样板）。

**新加的薄层**：
- `amp.formats.human.load(human_version_id)`：一个函数读 4 种存法（HiPHI 的 `motion_actor_smplx.npz`、Omni 的分字段 `.npy`、GRAIL 的 `recon/*.pkl`、LAFAN 放在 UMR `sample_data` 里），统一返回 `human_smplx_v1` 对象：Z 轴朝上、米、地面 z=0（原地面值另存 `floor_z_removed`）、`fps`、`frame_ids`（对源帧号）、`betas`、`gender`、物体位姿、`source_ref{path, sha256}`、`license_id`。旧文件**不重新转换**。以后新写的转换器（BONES、AMASS 原始、LAFAN 全量）直接写这个格式的 npz。
- `umr_dataset/sources.py` 找错 HiPHI 路径（`get_up_0004` FileNotFound）的问题，改成调这个函数就好了。这是唯一一处要改的下游代码，在 monorepo 里加适配，原仓库打了快照留底。
- HiPHI 两份：CPU 转的 1,094 段和 GPU 转的 3,796 段。规则：有 GPU 版就用 GPU 版（标 `superseded` 给 CPU 版）；只有 CPU 版的，补一个 30 fps 作业。"有没有 `qa.json`"这种互相跳过的暗号作废，以账本为准。
- 每个人体版本记一份 gauge：朝上的轴、单位、地面策略、地面偏移（mm）、帧率、降采样步长、是否镜像、转换器和 commit。
- license：没有 license 的片段永远到不了 `eligible`。字段：能不能商用（`no` / `yes` / `internal_rnd_is_commercial` / `unclear`）、能不能再分发、对模型的限制（原文）、署名要求、法务状态。打包、导出、模型卡自动带上 license 汇总。

**作业**：`s1.register`、`s1.download`（net）、`s1.convert_gpu`（gpu_heavy，低优先级）、`s1.convert_cpu`、`s1.gauge_check`、`s1.human_qa`。在跑的下载器和 GRAIL 循环先登记成 `external` 作业，只记账、计入资源总账，不重启它们。

**CLI**：`am s1 source list|show <id>` · `am s1 import --source grail` · `am s1 gauge-check --source hiphi --sample 200` · `am s1 license report --use commercial` · `am s1 status`（下载和转换进度 + 网络状态 + HiPHI 截止倒计时）

**MCP**：`source_list` · `source_get` · `clip_search(source, class, status, has_object, has_terrain, text, limit)` · `clip_get` · `ingest_status` · `license_check(clip_ids | dataset_id, intended_use)`

**HTTP**：`GET /api/v1/sources` · `/sources/{id}` · `/licenses` · `/clips?source=&class=&status=&has_object=&has_terrain=&q=` · `/clips/{id}` · `/clips/{id}/human?start=&count=`（二进制，沿用审计网页的协议）· `/ingest/status`

**测试门**：
1. 每个源 3 段金样：Z 轴朝上、单位米、fps 对、静止站立帧双脚最低点离地 ≤ 2 cm。
2. 时间轴：标成"走路"的片段，根速度中位数在 0.5–2 m/s；AMASS（xwbc）的 KIT 和 TotalCapture 按源 fps 重采样后和人体流误差 < 1 帧（专防已知的 1.2 倍、2 倍错位）。
3. Y-up 转 Z-up 用 (z, x, y) 置换，有专门测试，防止转偏 90°。
4. GRAIL 逐段 license 能覆盖源级 license；没有 license 的片段进不了 `eligible`。
5. 导入器连跑两遍，结果一样、不产生重复行。
6. GPU 转换器 commit 一变，自动重跑 `parity40` 对拍。

### B.2 S2 表面点采样（机器人登记 + 4096 个外表面点 + 区域）

**现成的**：`robots.json`（34 台，核心 21 台）；`umrf/robots.py`、`surface.py`、`scripts/stage_robots.py`、`build_surfaces.py` 产出每台 `surface.npz`（4096 点，射线可见性筛外表面 + 最远点采样，带法线、geom、body、区域）；`build_correspondence.py`、`build_shared_corr.py`；`umr_dataset` 派生的 `kin.npz`；人体侧 UMR 的 4096 个 SMPL-X 槽位和 `smplx_parts_segm.pkl`。

**决定**：
- 每台机器人一个权威目录 `data/s2_robots/<机器人>/`，就是现在工厂的 `robots/<机器人>/`（已经是展平的 MJCF + 网格，不再依赖桌面仓库里的 GMR）。登记表 `registry.json` 从 `/var/tmp/.../robots/robots.json` 拷来，`source_mjcf` 字段记原始来源（menagerie、unitree_ros、GMR、sonic……）。
- 资产分类登记：`mjcf`、`surface`、`corr`、`umr_config`、`sole_offset`、`kin_tables`、`staging`，各自带版本和哈希。任何一项变了，资产哈希就变；`am s2 impact <机器人>` 列出所有受影响的重定向、数据集、评估，并标 `stale`。
- **表面点只留一套**：工厂的 `surface.npz`（引擎和打包器都在用）。am2exp 的 `surface_v2`（绑在 xwbc GMR 的 MJCF 上，kuavo_s45、h1_2 绑定有问题）归档。
- **两份对应关系 corr 字节不一样**（工厂 `corr/` 和 `umr_gpu/corr`）：先不合并，分别登记成 `corr@factory_20261004` 和 `corr@umr_gpu_20261004`。生产用 GPU 那份（对拍是在它上面做的）。WP-07 查清差在哪。每条重定向结果都记下用的是哪份。
- 区域词表冻结成 `regions@v1`（现有 13 个区域），写清人体区域和机器人区域怎么对应。
- 脚底偏移：`~/tmp/sole_offsets3.json` 有 8 台偏大 4.7–10 cm、缺 H2。从 MJCF 碰撞几何重算一份 `sole_offset@v4`，只给旧模型线（Greenwich）用；UMR 线以表面点和地面 z=0 为准，不依赖这张表。

**作业**：`s2.stage`、`s2.surface`、`s2.corr`、`s2.corr_diff`、`s2.sole_offset`、`s2.contact_sheet`（都是 cpu_heavy）。

**CLI**：`am s2 robot list --core` · `am s2 robot show unitree_h2` · `am s2 surface <机器人>` · `am s2 check --all` · `am s2 impact <机器人>`

**MCP**：`robot_list` · `robot_get`（自由度、身高、区域点数、各资产版本、哪些环节已就绪）· `robot_surface_preview`（T 姿态 + 表面点图的地址）· `robot_impact`

**HTTP**：`GET /api/v1/robots` · `/robots/{r}` · `/robots/{r}/assets` · `/robots/{r}/surface.bin`（给网页 3D 的二进制点）· `/robots/{r}/mjcf` · `/robots/{r}/tpose.png`

**测试门**：① 每台正好 4096 点；② 每个点到碰撞几何表面 ≤ 0.5 mm，且不藏在别的几何体里；③ 必需区域（左右脚底、脚跟、脚尖、手掌、手指、膝、肘、躯干、骨盆）点数都 ≥ 下限；④ 法线单位长度、朝外；⑤ 重跑哈希不变；⑥ MJCF 哈希和登记表一致；⑦ T 姿态脚底最低点和 `sole_offset@v4` 差 ≤ 1 mm；⑧ 21 台核心机器人全部通过才允许进 S3。

### B.3 S3 重定向（UMR GPU 生产 + 严格后处理）

**现成的引擎（冻结，只包住）**：
- `umr_gpu`：生产引擎。UMR 求解和 CPU 对拍过（单步 2e-6 rad，自由运行 p50 1.3e-6 rad），严格后处理 `strict.py`/`strict_fp.py`（接触锁定、偏航锁定、抬脚规则、逐帧标记），导出 `gpu_npz_v2`，自带独立 CPU 判官 `eval/judge.py`。吞吐（4 台一起跑）：B=256 满速 2,707 机器人帧/秒但利用率 74%；限到 70% 以下时约 1,900–2,200 机器人帧/秒（B=256 duty 0.7 为 1,882；B=1024 duty 0.6 为 2,235，利用率 50%，显存峰值 33 GB）。
- `umr_factory_code`（CPU）：参考实现和量尺。已知箱子当球的 bug；保留 UMR 原生打滑（`LATCH_PIN=0`），达不到 v2 的"不滑"。只当参考，**不用于量产**；但它的 `run_clip.py` 能读真实片段，可以用来抓 GPU 引擎要的静态准备数据。
- 旧重定向：GMR 生产数据（Native 对照臂，9 台）、xwbc（18 台）。只登记，当对照臂。

**冻结和版本**（见 1.3 第 2 条）：
1. P1 在 `~/umr_gpu` 原地 `git init`，把现状（含 `.bak`）提交，打 tag `snap-20261008`。只增加 `.git`，不动文件。
2. 在这个 commit 的只读 worktree（`engines/umr_gpu@<commit>/`）上重跑对拍（`run_parity.py` 的 3 段 × 4 台 + 判官表）。通过了，账本记 `engines.parity_status = verified`。
3. 生产只从这个 worktree 跑。适配器自己设 `sys.path`，不调用引擎里写死路径的脚本；有一个测试断言 `umr_gpu.__file__` 落在 worktree 里面。
4. 引擎作者在 `~/umr_gpu` 继续改。要升级生产版本，就打新 commit → 重跑对拍 → 新版本第一批要 lloyd 批准。

**新加的薄层**（`code/s3_retarget/adapters/`）：
- `run_clips.py`：真实片段入口（见 1.3 第 3 条）。读账本里的人体版本 → 引擎 `source_gpu` 做逐帧输入 → `Retargeter` 求解 + 严格后处理 → 写文件。每台机器人的静态准备数据由 `capture` 一次抓好、登记成资产。**等价测试**：在 3 段对拍片段上，`run_clips` 的输出和 `run_one` 一致（关节差 ≤ 1e-6 rad，flag 位完全相同）。
- `plan`：按片段查询 × 机器人列表，算帧数、预计 GPU 时间（按 2,000 机器人帧/秒）、预计磁盘（按 0.33 KB/机器人帧）。
- `post`：每个输出旁边写 `<文件>.meta.json`：`joint_names`（从 MJCF 取）、引擎 commit、人体版本、corr 版本、资产哈希、规格版本、阈值哈希、sha256。
- `register`：写账本 `retargets`，引擎自己的 `qc_json` 写进 `qa_metrics`（标 `judge=engine_self`）；随后独立判官复核一遍。
- `enqueue_audit`：按风险和随机两种方式放进人审队列。
- 输出一律写 `/home`：`data/s3_retarget/umr_gpu@<commit>/<批次>/<片段>__<机器人>.npz`，不再写 `/var/tmp`。

**格式**：读取层 `amp.formats.motion.load(retarget_id)` 认 `factory_npz_v1` 和 `gpu_npz_v2`（加旁挂 meta），统一返回 `am_motion_v3` 对象（字段见 C.3）。旧文件不重写。

**阈值只存一份**：`config/gauges/data_spec_v2.toml`：
- 实接触：该区域稠密点里最低点离地或离物体在 [0, 1.0] mm，至少 1 个点（引擎核验时下限放到 -0.5 mm，和"不穿"的容差一致）；
- 不穿：任何点低于地面或进入物体不超过 0.5 mm，并用 MuJoCo 碰撞几何复核；
- 不滑：接触点切向速度 < 1 cm/s；
- 不闪：进入 < 1 mm、离开 > 2 mm，接触段 ≥ 3 帧；
- 和人一致：人体区域接触 → 机器人对应区域必须接触；人体区域离地 > 2 cm → 机器人对应区域 > 5 mm。

UMR 原生标签用的是 1 cm / 2 cm / 3 cm/s，这套标成 `legacy_umr` 单独显示，不冒充 v2。引擎常数 = 配置的断言测试见 1.3 第 8 条。

**还没过关、先挡住的**（账本 `issues` 里登记，`blocks_production=1`）：
- `ISSUE-box`：CPU 工厂已产出的所有搬箱片段标 `quarantined`；GPU 引擎跑合成箱子小测（WP-06）。验收：接触段手离箱面 ≤ 1 mm、物体实接触率 > 0。没过之前，OmniContact 搬箱片段不进量产。
- `ISSUE-terrain`：GPU 引擎只按平地 z=0 处理地面。GRAIL 的楼梯、路沿、坡片段（账本 `clips.has_terrain=1`）先不进量产；WP-06 顺带做一个"地形当物体喂进去"的小测。
- `ISSUE-corr`：两份 corr 不一致，查清之前每条结果记下用的哪份。

**作业**：`s3.capture`（cpu_heavy，抓静态准备数据）· `s3.solve`（gpu_heavy）· `s3.post` · `s3.register` · `s3.judge`（cpu）· `s3.parity`（gpu_heavy，小）。新引擎版本的第一批、估算超过 4 小时的计划，要 lloyd 批准。

**CLI**：`am s3 plan --clips "source=hiphi AND has_terrain=0" --robots core21 --engine umr_gpu` · `am s3 submit <plan_id>` · `am s3 status [批次]` · `am s3 show <重定向 id>` · `am s3 parity` · `am s3 import-legacy`（登记 CPU `out/` 11 个、GPU export 样本 12 个、GMR、xwbc）

**MCP**：保留旧名 `motion_catalog` · `motion_inspect` · `motion_frame_metrics` · `motion_visualize`；新增 `retarget_plan(clip_query, robots, engine)` · `retarget_get(clip, robot)`（指标、路径、审计状态、预览图）· `retarget_status`。提交走 `workflow_submit`。

**HTTP**：`POST /api/v1/retarget/plans` · `GET /retargets?clip=&robot=&engine=&status=` · `/retargets/{id}` · `/retargets/{id}/frames?start=&count=`（分块二进制，给 3D）· `/retargets/{id}/qa`

**测试门**：
1. 对拍回归：冻结的 3 段 × 4 台，关节 p50 ≤ 1e-5 rad，判官表在容差内不变（要 GPU，只在升级引擎版本时跑）。
2. `run_clips` 和 `run_one` 等价。
3. 读取层：v1、v2 读出来字段齐全、单位是米；v1 的 packbits 接触转成 CSR 后按位一致。
4. 严格性：独立判官重量导出文件，保留帧全部过阈值，剔除帧数和文件里记的一模一样。
5. 合成箱子测试、地形小测。
6. 适配器只读引擎目录：跑前跑后对 worktree 做快照比较，完全不变。

### B.4 S4 整库评估（有多少、种类全不全、质量过没过、人看过没有）

**现成的**：`diversity/`（各源 `summary.json`、逐段标签、图、今天的 `REPORT.md`：已知 571.4 h 原片，手上 126.4 h 已转 SMPL-X，躺/爬/翻滚、摔倒起身、重物、真人楼梯约 0 h）；`action_labels.py`（12 族正则，AMASS 77% 落 "other"）；BONES 分级表；5 台机器人各自的 QA 流水线（格式各不相同）；`audit/SCHEMA.md` 和 `motion_audit_web`（带 Wilson 置信区间、哈希过期机制）；`wbc_ready_audit`（600 段语料外考卷）。

**三层**：
1. **覆盖**：分类法 `TAXONOMY_v1` = `REPORT.md` 的 9 个大类，每段只归 1 个主类（小时数可以相加），12 族正则和 BABEL 作副标签；`labels` 表记方法（元数据文字 / 正则 / 模型 / 人）和置信度。按 主类 × 源 × 环节状态 × 机器人 统计小时数，和目标配额比，自动出**稀缺清单**，反过来决定下载和转换的优先级（HiPHI 包顺序已经这么排了）。另看演员数、接触模式（哪些区域碰地或碰物体）、物体种类、姿态空间占用率，"other" 的占比本身是一个指标。
2. **质量**：逐段 `qa_metrics` 按 源 × 机器人 × 大类 汇总成分布：穿透 p99、实接触率、打滑 p95、和人一致率、剔除帧比例。阈值统一取 B.3 那份。5 台各写一份的 `measure.py` 退役，统一用 S7 的 `geometry@v1` 加每台一份配置，报告统一成 `qa_report_v1.json`。
3. **人审**：队列按风险（自动质检越差越靠前）和随机（无偏）两种方式抽。定时把 `decisions.jsonl`、`issues.jsonl` 读进账本，哈希对不上的判决标过期。`needs_fix` 的回到重定向队列并带上原因；系统性问题（某台机器人、某个源、某个引擎）开成 issue。**判决只能人写**（`reviewer_kind='human'`），代理只能写观察笔记（`agent_note`）。

**批次放行门**：一批重定向要进训练包，必须同时满足：① 随机抽样合格率的 Wilson 下界 ≥ 0.90（数值 lloyd 定）；② 没有 `blocks_production` 的 issue；③ license 策略通过；④ 黄金考卷的 `split_group` 不在里面。合格率只用"随机抽样、且哈希还对得上的判决"来算。

**产出**：`data/s4_dataset_eval/<日期>_snapshot/{REPORT.md, numbers.json, charts/}`，登记时记下用的是账本哪一天的快照，保证同快照能复现同样的数。

**CLI**：`am s4 census` · `am s4 label --taxonomy v1` · `am s4 coverage --by family,source` · `am s4 quality --by source,robot,class` · `am s4 audit enqueue --risk 50 --random 50 --batch <id>` · `am s4 audit ingest` · `am s4 gate <batch>` · `am s4 report`

**MCP**：`diversity_summary(group_by)` · `coverage_gaps()` · `quality_summary(source, robot, class)` · `issues_list(status)` · `issue_open` · `audit_queue_status` · `audit_enqueue(batch, n_risk, n_random)` · `agent_note`（已有）

**HTTP**：`GET /api/v1/stats/diversity?group_by=` · `/stats/coverage-gaps` · `/stats/quality?group_by=` · `/audit/queue` · `POST /audit/decisions`、`POST /audit/issues`（只限人，后端照约定追加写 jsonl）· `GET /issues` · `/snapshots/dataset-eval`

**测试门**：① 小时数守恒（按类加总 = 按源加总 = 总数）；② 每段正好 1 个主类；③ 置信区间不含非随机样本和哈希过期的判决；④ 同一快照重新生成报告，数字完全一样；⑤ 和今天的 `REPORT.md` 对账，差异能逐条解释。

### B.5 S5 存储（账本、数据集、磁盘、清理）

细节见 C 节。工具：
- `am s5 catalog init|migrate|backup|check`
- `am s5 dataset plan --name am_pretrain --select "<查询>" --license research_only` → 段数、小时、机器人帧数、磁盘
- `am s5 dataset build <plan_id>`：调 `umr_dataset` 的 convert（原样复用，人体读取经 `amp.formats.human`）
- `am s5 verify <数据集>`：调 `umr_dataset/verify.py` 校验分片哈希；`am s5 verify --sample 1%` 抽查账本
- `am s5 disk`：各存储池水位、按目录排的大户、按保留级别分的可回收量
- `am s5 gc plan`：只出清单；执行只能 lloyd 在本机 `am s5 gc apply <plan_id>`（要交互确认）
- `am index`、`am what <路径>`、`am doctor`、`am paths resolve|lint`

**MCP**：`catalog_query`（白名单过滤条件，不收原始 SQL）· `what_is(path_or_id)` · `lineage(id, up|down)` · `dataset_list` · `dataset_get` · `dataset_plan` · `disk_report` · `gc_plan`（只读）

**HTTP**：`GET /api/v1/datasets` · `/datasets/{id}` · `POST /datasets/plans` · `GET /disk` · `/gc/plans` · `/index` · `/what?path=` · `/artifacts/{id}` · `/artifacts/{id}/lineage`

**测试门**：① 分片校验和；② 加载器按帧精确配对（沿用 `umr_dataset` 的 `roundtrip_7entries`、`strict_recheck`）；③ 账本里每条位置都存在，抽样核哈希；④ 清理计划里绝不出现 `keep` 级或还被引用的产物；⑤ 超配额时拒绝放行会产出大文件的作业；⑥ 代码不许引用 `scratch/`、`cache/`。

### B.6 S6 训练 + 实时看模型生成了什么

细节见 D.1、D.2。
- 训练入口（包住）：AM2 `train_scaling.py`、`train_arm.py` + `arms.json`，配方 `megatron_scale_pilot/code` 冻结；Greenwich `exp_topo/train_dual.py` 只用于复现 v0.2，冻结。
- **CLI**：`am s6 run list|show` · `am s6 train plan <arm>` / `am s6 train submit <plan_id>`（要批准）· `am s6 watch <run_id>` · `am s6 viewer serve|load --model <ckpt> --clip <id> --robots g1,h2`
- **MCP**：`model_list` · `model_card` · `run_list` · `run_get` · `run_curves` · `run_live_latest`（最新中途视频的路径和地址）· `train_plan` · `viewer_load` · `viewer_status`
- **HTTP**：`GET /api/v1/models` · `/runs` · `/runs/{id}` · `/runs/{id}/curves` · `/runs/{id}/live` · `POST /viewer/load` · `GET /viewer/status`
- **测试门**：watcher 推理只用 CPU（测试里设 `CUDA_VISIBLE_DEVICES=""` 断言）；查看器加载一段后截图，帧号和时间轴一致；没批准文件的训练作业提交被拒。

### B.7 S7 评估模型重定向或生成出来的动作

细节见 D.3。
- **CLI**：`am s7 eval plan --subject ckpt:<id> --suite geometry@v1,locomotion@v1 --golden golden_core@v1` · `am s7 eval submit` · `am s7 compare <评估A> <评估B>` · `am s7 render triptych --source <片段> --control <id> --test <id>` · `am s7 leaderboard --suite locomotion@v1 --robot g1`
- **MCP**：`golden_list` · `eval_plan` · `eval_get` · `eval_items` · `eval_compare` · `render_triptych` · `leaderboard_get` · `media_get`（返回本地路径，Claude 直接把 mp4 发给 lloyd）· `video_inspect`（已有）· `wbc_track_eval`（要批准）
- **HTTP**：`GET /api/v1/golden` · `/golden/{id}` · `POST /evals/plans` · `GET /evals` · `/evals/{id}` · `/evals/{id}/items` · `POST /compare` · `GET /leaderboard?suite=&robot=` · `POST /renders/plans` · `GET /media/{id}`（支持按字节范围取，手机能拖进度条）· `/media?subject=` · `POST /reviews`（只限人：看过视频后的判决）

### B.8 一个服务：MCP + HTTP + 网页

```
      Claude / 代理                               lloyd / Codex 做的网页
          │ stdio MCP（ssh -T arenalabs …）              │ HTTP 127.0.0.1:8765（lloyd 用 ssh 转发）
    amp/service/mcp.py                           amp/service/api.py（FastAPI）
          └───────────────────┬──────────────────────────┘
                     amp/service/core.py   ← 唯一的实现
       ┌──────────┬───────────┼────────────┬─────────────┬──────────────┐
  catalog.sqlite  formats   jobs（租约）   paths/stores   人审 jsonl 约定   查看器 viser 127.0.0.1:8781（经 /viewer 转出）
                              │
     worker@cpu（≤ 6 个重进程）│ worker@gpu（同时 1 个重任务）│ worker@net（全局 ≤ 16 连接）│ worker@light
```

- **底座**：`~/motion_audit_web`。后端 `backend/app.py`，前端 React + Vite + Three.js，MCP `agent_tools/server.py`（15 个工具，今天能用）。P3 把它带历史并进 monorepo 的 `code/amp/service` 和 `code/web`；旧启动脚本 `~/motion_audit_web/scripts/mcp_stdio.sh` 改成三行转发到 `am mcp`，**工具名全部不变**，Codex 和 Claude 的 MCP 配置都不用改。并入前先确认没人还在写 `agent_tools/`（看修改时间和进程）。
- **作业层**：表结构照搬 arena-datagrid `db/store.py` 的 `jobs`、`job_events`（租约、尝试次数、唯一 `request_hash`）；行为照搬 am2exp `ops`（`lease.py` GPU 锁、`guard.py` 磁盘守卫、`receipts.py` 回执、人工批准）；注册表格式照搬 `configs/mcp.json` 的 workflow 写法。
- **端口**：只监听 `127.0.0.1`。统一服务用 8765（审计网页原端口），查看器 8781。现在对全网开着（`0.0.0.0`）的 2334、8891、8899 三个旧服务，内容并进网页后停掉。
- **两种角色**：MCP 客户端是"代理"，网页登录是"人"。人的登录：lloyd 在本机终端跑 `am web login`，打印一次性链接，浏览器拿到会话 cookie。代理可以查询、出计划、提交（受调度器管）、写观察笔记、开 issue；代理**不能**批准、加冕、执行清理、写人工审计判决、跑任意 shell——MCP 进程里根本不注册这些函数。（单人机器上这是防误操作的护栏，不是防攻击的安全边界；`CLAUDE.md` 同样写明代理不许调 `am approve`。）
- **网页页面**（Codex 照冻结的 `docs/contracts/openapi_v1.yaml` 写）：
  1. 总览：七个环节的进度、正在跑的作业、磁盘水位、网络状态、HiPHI 截止倒计时；
  2. "这是什么"搜索框（`/what`）；
  3. 来源和片段浏览；
  4. 机器人（T 姿态、表面点、各资产版本）；
  5. 重定向浏览和 3D 审片（现成）；
  6. 人审队列（现成）；
  7. 数据评估看板（覆盖、稀缺清单、质量分布）；
  8. 数据集和分片；
  9. 存储和清理计划（勾选批准）；
  10. 训练 run（曲线 + 中途视频）；
  11. 实时查看器（嵌 viser）；
  12. 评估和排行榜；
  13. 媒体画廊；
  14. 作业；
  15. 待你批准。

### B.9 资源上限、重启和断网

**资源上限由内核执行，不靠自觉**（照 `RESOURCE_LIMITS.md`）：
- 所有常驻进程（服务、worker、下载器）做成 systemd 用户单元，放进 `am.slice`：`CPUAffinity=0-15`、`Nice=10`、`Environment=OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。用户 linger 已开（`Linger=yes`），开机自动起。
- worker 启动每个作业用 `systemd-run --user --scope --slice=am.slice -p MemoryMax=20G -p CPUAffinity=0-15 nice -n 10 …`，单进程树超过 20 GB 由内核杀掉，而不是拖垮整机。
- 调度器（`amp/jobs/governor.py`）放行规则：CPU 重活最多 6 个；重 GPU 同时 1 个；GPU 利用率 > 70% 或功耗 > 350 W 时不放新 GPU 作业，批大小从小往上加；下载连接按作业分配，合计 ≤ 16；`free` 可用 < 15 GB 就排队；/home 用到 85% 停止放行重写入作业。
- 在跑的旧进程（下载器、GRAIL 循环、`gpu_convert`、引擎作者的测试）登记成 `external` 作业，计入总账。
- `scripts/throttle_guard.sh` 继续当兜底，不依赖它。

**防崩溃循环**：每个单元都写 `Restart=on-failure`、`RestartSec=30`、`StartLimitIntervalSec=600`、`StartLimitBurst=5`（10 分钟内失败 5 次就停，在总览上报红），服务再加 `RuntimeMaxSec`（闲置 8 小时自动退出，lloyd 的"服务必须有寿命"纪律）。停服务按单元名，不用 `pkill -f`。

**重启以后**：
- 账本 SQLite 开 WAL，每次写一个短事务，断电最多丢最后一个事务。
- 每个作业记 `boot_id`。开机后 worker 发现 `running` 但 `boot_id` 不是本次的 → 改成 `interrupted`。
- 注册表里标了 `resumable: true` 的作业（下载、转换、重定向批次、判官、渲染——都按条目跳过已完成的，且提交幂等）自动重新排队。
- 训练不自动续跑：要 lloyd 批准，批准后从最后一个 checkpoint 接着跑。
- 下载器做成单元以后，aria2 用 `--continue` 自动续传，HiPHI 下载器自带状态文件。

**断网时**：
- 调度器每 20 秒读 `logs/throttle.log`。看到 `net=down`：暂停网络作业（对 aria2 发暂停，恢复时 `--continue`），不再放行新的重活；在跑的 CPU/GPU 作业不打断。
- MCP 走 SSH stdio，掉线时客户端断开，但作业和状态都在盘上，重连后用 `job_status` 接着查（今天的 MCP 已经是这样）。
- 网页同理，经 SSH 转发，重连即可。

**账本备份**：每晚 `sqlite3 .backup` 到 `catalog/backups/`（留 30 份），同时拷一份到根分区 `/var/tmp/alphamotion/_catalog_backup/`（另一个分区，文件很小）；小表导出 `catalog/exports/*.jsonl` 进 git。账本坏了，用快照加 `job_events` 重放恢复。

---

## C. 存储设计

### C.1 账本（catalog）

- **SQLite + WAL**：一台机器、最多几百万行（GRAIL 2.2 万段、foundation 索引 24 万条、全量重定向约 130 万 × 21 台以内的行数），多进程读、短事务写就够；arena-datagrid 已经这么用过。逐帧序列不进 SQLite，放 `catalog/parquet/`，统计时用 DuckDB 读。
- 只能经 `amp.catalog` 写，每次写追加一条 `events`。
- 表结构（`code/amp/catalog/schema/0001_init.sql`，列名就是契约，节选）：

```sql
-- 0. 存储、产物、血统
CREATE TABLE stores(store TEXT PRIMARY KEY, path TEXT NOT NULL, fs_mount TEXT,
  tier TEXT CHECK(tier IN('hot','warm','cold','cache')), legacy INTEGER DEFAULT 0, writable INTEGER DEFAULT 1, note_zh TEXT);
CREATE TABLE artifacts(art_id TEXT PRIMARY KEY,           -- 'a:<uuid>'，与内容无关的稳定 id
  kind TEXT NOT NULL,   -- raw|human_motion|robot_asset|surface|corr|retarget|label|qa_series|dataset|shard|legacy_cache|
                        -- run_dir|checkpoint|eval|media|report|spec|env|legacy_dir
  name TEXT, version TEXT, store TEXT REFERENCES stores, relpath TEXT,   -- 搬家只改这两列
  bytes INTEGER, nfiles INTEGER,
  sha256 TEXT, hash_kind TEXT CHECK(hash_kind IN('file','tree','sampled','none')),   -- sampled = 文件数+字节数+抽样 64 个文件首尾哈希
  format TEXT, format_version TEXT, spec_version TEXT,
  produced_by_job TEXT, code_repo TEXT, code_commit TEXT,
  status TEXT CHECK(status IN('open','sealed','stale','superseded','missing','corrupt','quarantined','delete_candidate','trashed')),
  retention TEXT CHECK(retention IN('keep','keep_n','regen','scratch','cache','archive')),
  license_rollup TEXT, summary_zh TEXT, created_at TEXT, sealed_at TEXT, verified_at TEXT);
CREATE TABLE lineage(child TEXT, parent TEXT, role TEXT CHECK(role IN('input','derived_from','supersedes','uses_asset')),
  PRIMARY KEY(child,parent,role));
CREATE TABLE aliases(alias TEXT PRIMARY KEY, art_id TEXT NOT NULL, set_by TEXT, set_at TEXT);
CREATE TABLE legacy_paths(old_path TEXT PRIMARY KEY, art_id TEXT, from_map TEXT, purpose_zh TEXT,
  status TEXT CHECK(status IN('registered','linked','copied','moved','archived','delete_candidate','deleted','rescued','missing')),
  new_path TEXT, hardcoded_refs INTEGER, referenced_by TEXT, phase TEXT, note_zh TEXT);

-- 1. 来源、license、人体
CREATE TABLE licenses(license_id TEXT PRIMARY KEY, name TEXT,
  commercial TEXT CHECK(commercial IN('no','yes','internal_rnd_is_commercial','unclear')),
  redistribution INTEGER, model_restriction TEXT, attribution TEXT, text_art TEXT, legal_review TEXT DEFAULT 'pending');
CREATE TABLE sources(source_id TEXT PRIMARY KEY, kind TEXT, title TEXT, upstream TEXT, revision TEXT,
  license_id TEXT REFERENCES licenses, native_fps REAL, hours_known REAL, access_deadline TEXT, note_zh TEXT);
CREATE TABLE clips(clip_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources, native_path TEXT,
  frames INTEGER, fps REAL, seconds REAL, actor_id TEXT, gender TEXT, height_m REAL,
  has_object INTEGER, has_terrain INTEGER, is_mirror INTEGER DEFAULT 0,
  license_id TEXT REFERENCES licenses,               -- 逐段覆盖（GRAIL）
  split_group TEXT,                                  -- 同演员/同场次一组，防训练集和考卷泄漏
  split TEXT CHECK(split IN('train','val','test','golden','holdout')), text TEXT,
  status TEXT CHECK(status IN('registered','converted','eligible','quarantined','excluded')));
CREATE TABLE human_versions(hv_id TEXT PRIMARY KEY, clip_id TEXT NOT NULL, storage TEXT,  -- hiphi_npz|omni_npy|grail_pkl|lafan_sample|human_smplx_v1
  fps REAL, stride INTEGER, gauge TEXT, converter TEXT, converter_commit TEXT, qa TEXT, art_id TEXT, status TEXT);
CREATE TABLE labels(clip_id TEXT, taxonomy TEXT, primary_class TEXT, tags TEXT, method TEXT, confidence REAL,
  PRIMARY KEY(clip_id, taxonomy));
CREATE TABLE objects(object_id TEXT PRIMARY KEY, source_id TEXT, kind TEXT, mesh_sha256 TEXT, collision_art TEXT, license_id TEXT);
CREATE TABLE clip_objects(clip_id TEXT, object_id TEXT, PRIMARY KEY(clip_id, object_id));

-- 2. 机器人
CREATE TABLE robots(robot_id TEXT PRIMARY KEY, vendor TEXT, model TEXT, dof INTEGER, height_m REAL,
  tier TEXT CHECK(tier IN('core21','extra','retired')), embodiment_split TEXT CHECK(embodiment_split IN('seen','heldout')),
  mjcf_source TEXT, license TEXT, note_zh TEXT);
CREATE TABLE robot_assets(asset_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL,
  kind TEXT CHECK(kind IN('mjcf','surface','corr','umr_config','sole_offset','kin_tables','staging','prep')),
  version TEXT, art_id TEXT, n_points INTEGER, region_counts TEXT, built_by_job TEXT, active INTEGER DEFAULT 0);

-- 3. 引擎、重定向、质检、人审
CREATE TABLE engines(engine_id TEXT PRIMARY KEY, family TEXT, code_commit TEXT, worktree TEXT,
  parity_status TEXT CHECK(parity_status IN('unverified','verified','failed')), parity_report_art TEXT,
  spec_version TEXT, thresholds_hash TEXT, v2_compliant INTEGER, approved_by TEXT);
CREATE TABLE retargets(retarget_id TEXT PRIMARY KEY, clip_id TEXT NOT NULL, hv_id TEXT NOT NULL, robot_id TEXT NOT NULL,
  engine_id TEXT NOT NULL, corr_asset TEXT, surface_asset TEXT,
  format TEXT CHECK(format IN('factory_npz_v1','gpu_npz_v2','am_motion_v3','xwbc22','gmr_native')),
  art_id TEXT, frames INTEGER, frames_excluded INTEGER, fps REAL, batch_id TEXT, job_id TEXT,
  status TEXT CHECK(status IN('produced','judged','in_audit','accepted','rejected','needs_fix','quarantined','packed','purged')),
  UNIQUE(clip_id, robot_id, engine_id, hv_id));
CREATE TABLE judges(judge_id TEXT PRIMARY KEY, family TEXT, code_commit TEXT, thresholds_hash TEXT, spec_ref TEXT,
  imports_training_code INTEGER NOT NULL CHECK(imports_training_code = 0));
CREATE TABLE qa_metrics(subject_kind TEXT CHECK(subject_kind IN('human','retarget','model_output','sim_rollout')),
  subject_id TEXT NOT NULL, judge_id TEXT NOT NULL,
  pen_ground_max_mm REAL, pen_object_max_mm REAL, pen_self_max_mm REAL, solid_contact_rate REAL, hover_rate REAL,
  slip_p95_cm_s REAL, flicker_segments INTEGER, human_agree_f1 REAL, flagged_frames INTEGER, excluded_frames INTEGER,
  verdict TEXT CHECK(verdict IN('pass','fail','not_evaluated')), series_art TEXT, created_at TEXT,
  PRIMARY KEY(subject_kind, subject_id, judge_id));
CREATE TABLE batches(batch_id TEXT PRIMARY KEY, stage TEXT, engine_id TEXT, n_items INTEGER,
  random_n INTEGER, random_pass INTEGER, wilson_lb REAL, gate_threshold REAL, released INTEGER DEFAULT 0, released_by TEXT);
CREATE TABLE audit_items(item_id TEXT PRIMARY KEY, subject_id TEXT, art_sha256 TEXT, priority REAL,
  sample_kind TEXT CHECK(sample_kind IN('risk','random')), batch_id TEXT, enqueued_at TEXT);
CREATE TABLE audit_decisions(id INTEGER PRIMARY KEY, item_id TEXT, art_sha256 TEXT,
  verdict TEXT CHECK(verdict IN('accept','reject','needs_fix')), frame_ranges TEXT, reasons TEXT, note TEXT,
  reviewer TEXT, reviewer_kind TEXT CHECK(reviewer_kind = 'human'), stale INTEGER DEFAULT 0, ts TEXT);
CREATE TABLE agent_notes(id INTEGER PRIMARY KEY, subject_id TEXT, sha256 TEXT, frames TEXT, observation TEXT, agent TEXT, ts TEXT);
CREATE TABLE issues(issue_id TEXT PRIMARY KEY, scope TEXT, target TEXT, severity TEXT, title TEXT, body TEXT,
  status TEXT CHECK(status IN('open','fixing','fixed','wontfix')), blocks_production INTEGER DEFAULT 0,
  blocks_packing INTEGER DEFAULT 0, created_by TEXT, ts TEXT);

-- 4. 数据集
CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY, kind TEXT CHECK(kind IN('umrs_v1','legacy_cache')),
  selection_art TEXT, license_policy TEXT, release_restrictions TEXT, n_clips INTEGER, hours REAL, robot_frames INTEGER,
  art_id TEXT, built_by_job TEXT, frozen INTEGER DEFAULT 0);
CREATE TABLE dataset_members(dataset_id TEXT, retarget_id TEXT, clip_id TEXT, robot_id TEXT, split TEXT,
  admitted_by TEXT CHECK(admitted_by IN('human_accept','released_batch','legacy')), PRIMARY KEY(dataset_id, retarget_id));
CREATE TABLE shards(shard_art TEXT PRIMARY KEY, dataset_id TEXT, body TEXT, rows INTEGER, bytes INTEGER);
CREATE TABLE legacy_datasets(dataset_id TEXT PRIMARY KEY, representation TEXT, up_axis TEXT, length_unit TEXT, fps REAL,
  links_into TEXT, lineage_json TEXT);    -- cache_lafan_v3 等，记下它的软链指向哪里

-- 5. 模型、训练、评估、媒体
CREATE TABLE runs(run_id TEXT PRIMARY KEY, family TEXT, model_id TEXT, dataset_ids TEXT, config_art TEXT, code_commit TEXT,
  status TEXT CHECK(status IN('planned','waiting_approval','running','interrupted','finished','failed','killed','imported')),
  approval_id TEXT, est_gpu_h REAL, est_kwh REAL, est_cost REAL, gpu_h REAL, started_at TEXT, ended_at TEXT,
  art_id TEXT, lineage_tag TEXT CHECK(lineage_tag IN('release','lineage','archive') OR lineage_tag IS NULL), note_zh TEXT);
CREATE TABLE checkpoints(ckpt_id TEXT PRIMARY KEY, run_id TEXT, step INTEGER, art_id TEXT,
  keep TEXT CHECK(keep IN('best','last','milestone','release','drop')), metrics TEXT);
CREATE TABLE models(model_id TEXT PRIMARY KEY, family TEXT, size TEXT, ckpt_id TEXT,
  title TEXT, title_status TEXT CHECK(title_status IN('candidate','crowned','retired')), crowned_by TEXT, crowned_at TEXT,
  release_restrictions TEXT, card_art TEXT);
CREATE TABLE golden_sets(golden_id TEXT PRIMARY KEY, purpose TEXT, members_art TEXT, members_sha TEXT,
  frozen_at TEXT, leakage_checked INTEGER);
CREATE TABLE suites(suite_id TEXT PRIMARY KEY, judges TEXT, standard_art TEXT, protocol_art TEXT);
CREATE TABLE evals(eval_id TEXT PRIMARY KEY, suite_id TEXT, golden_id TEXT,
  subject_kind TEXT CHECK(subject_kind IN('ckpt','engine','dataset','refiner','retarget_batch')), subject_id TEXT,
  control_id TEXT, judge_versions TEXT, job_id TEXT, status TEXT,
  decision TEXT CHECK(decision IN('win','lose','tie','not_evaluated') OR decision IS NULL), report_art TEXT, created_at TEXT);
CREATE TABLE eval_items(eval_id TEXT, case_id TEXT, metric TEXT, value REAL, gate TEXT,
  outcome TEXT CHECK(outcome IN('pass','fail','not_evaluated')), not_evaluated_reason TEXT, media_id TEXT,
  PRIMARY KEY(eval_id, case_id, metric));
CREATE TABLE media(media_id TEXT PRIMARY KEY, kind TEXT CHECK(kind IN('triptych','grid12','showcase','live','tpose')),
  art_id TEXT, layout TEXT, subjects TEXT, seconds REAL, viewed_by_lloyd INTEGER DEFAULT 0, viewed_at TEXT);
CREATE TABLE reviews(review_id TEXT PRIMARY KEY, subject_kind TEXT, subject_id TEXT, media_id TEXT,
  reviewer TEXT, reviewer_kind TEXT CHECK(reviewer_kind = 'human'), verdict TEXT, note TEXT, ts TEXT);

-- 6. 作业、审批、服务、事件
CREATE TABLE jobs(job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, request_hash TEXT UNIQUE NOT NULL, params TEXT NOT NULL,
  resource_class TEXT CHECK(resource_class IN('light','cpu_heavy','gpu_heavy','net')), net_conns INTEGER DEFAULT 0,
  priority INTEGER DEFAULT 0, resumable INTEGER DEFAULT 0,
  status TEXT CHECK(status IN('planned','queued','waiting_approval','leased','running','interrupted','done','failed','cancelled','external')),
  approval_id TEXT, parent_job TEXT, submitted_by TEXT, claimed_by TEXT, pid INTEGER, unit TEXT, boot_id TEXT,
  lease_until TEXT, attempt INTEGER DEFAULT 0, est_seconds REAL, est_out_bytes INTEGER, est_gpu_h REAL,
  created_at TEXT, started_at TEXT, finished_at TEXT, error TEXT, outputs TEXT);
CREATE INDEX jobs_claim_idx ON jobs(status, resource_class, priority, created_at);
CREATE TABLE job_events(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts TEXT, stage TEXT, payload TEXT);
CREATE TABLE approvals(approval_id TEXT PRIMARY KEY,
  scope TEXT CHECK(scope IN('train','wbc_eval','long_gpu','engine_first_batch','batch_release','gc_apply',
                            'cross_fs_move','crown','license_override','service_stop')),
  request TEXT, cost_estimate TEXT, decided TEXT CHECK(decided IN('pending','approved','rejected')),
  decided_by TEXT, decided_at TEXT, via TEXT CHECK(via IN('web_human','cli_human')));
CREATE TABLE services(svc_id TEXT PRIMARY KEY, unit TEXT, port INTEGER, bind TEXT, pid INTEGER, boot_id TEXT,
  started_at TEXT, runtime_max_h REAL, status TEXT);
CREATE TABLE disk_snapshots(ts TEXT, store TEXT, used_bytes INTEGER, free_bytes INTEGER, top TEXT);
CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, kind TEXT, payload TEXT);
```

**导入现有台账，不重建**（WP-03，只读、可重复跑）：
- GRAIL `manifest.jsonl`（22,312 行）→ `clips`、`human_versions`、逐段 license；
- HiPHI `convert_log.jsonl`、`parts_status.json`、`qa.json`、`metadata/*.csv`；Omni `qa.json`；
- `sources_inventory.json`、各 `LICENSE.md` → `licenses`、`sources`（HiPHI 的 `access_deadline` 写进去）；
- `robots.json`（34 台）、`staging_summary.json`、`surface_summary.json` → `robots`、`robot_assets`；
- 工厂 `out/manifest.jsonl`（11）、GPU `results/export`（12）→ `retargets`；
- `umr_dataset/data/manifest.json` + `catalog_meta.json` → `datasets`、`shards`；
- `pretraining_foundation/cache/index*.sqlite`（240,444 条）→ `legacy_datasets` + 格式为 `xwbc22` 的 `retargets`（对照臂）；
- `MODEL_VERSIONS.md`、`RUNS_INDEX.md`、`topo_exp/runs`（823 个条目）、AM2 六组和 `continuation_v3`、megatron 试训、am2exp 各臂、`alphamotion_ctrl/logs` → `runs`、`checkpoints`、`models`（状态 `imported`；血统上的标 `lineage`，v0.2 发布的 Bumblebee/Megatron/Prime 标 `release`）；
- `verdicts/` → `events`（kind=verdict）；
- 4 张地图的表 → `legacy_paths`（含每个旧路径被写死引用的次数，WP-11 扫出来补上）。

### C.2 产物身份

- **按路径存放 + 账本记哈希**。不做内容寻址的 blob 库。
- 三种哈希：单文件 `file`（sha256）；目录 `tree`（排好序的（相对路径、大小、sha256）清单再取哈希，清单存进 `MANIFEST.json`）；旧大目录 `sampled`（文件数 + 总字节 + 最新修改时间 + 抽 64 个文件、每个首尾各 1 MB 的哈希），空闲时用 `ionice -c3` 慢慢补算成 `tree`。**HiPHI 截止前不做大量哈希计算。**
- **生命周期**：作业先写进 `<目标目录>.open/`，账本状态 `open`；成功后算哈希、写 `MANIFEST.json`、同分区改名成正式目录、去掉写权限、状态改 `sealed`。失败的 `.open/` 留 3 天方便排查，然后进清理计划。
- 引擎原样写出的文件不改；补充信息写旁挂的 `<文件>.meta.json`。
- 重定向结果的身份用"输入 + 引擎 commit"定（GPU 浮点有不确定性），内容哈希只用来查文件坏没坏。
- **完整性**：`am s5 verify` 每天抽查 1%，每周把 `keep` 级全查一遍；对不上就标 `corrupt` 并自动开 issue。

### C.3 文件格式（一共 4 种，不再多）

| 格式 | 用途 | 状态 |
|---|---|---|
| `human_smplx_v1` | 统一的人体动作。读取层把 4 种旧存法都读成它；新写的转换器直接写它。可选附带人体 4096 槽位上的接触标签（CSR，和机器人同一把尺子、同一个校准地面，DATA_SPEC v2 第 3 节要求） | 旧文件不重转 |
| `am_motion_v3` | 统一的机器人动作，数据和模型共用 | 读取层把 `factory_npz_v1`、`gpu_npz_v2` 读成它；**模型推理输出从第一天起就写它**；GPU 引擎的导出要等 P4 "导出等价门"通过、lloyd 同意后才改成直接写它 |
| `umrs_v1` | 训练分片（`umr_dataset` 现成的） | 直接用，不另造 |
| `scene_v1.json` | 查看器场景：几路并排看什么 | 只是一个小清单，引用上面几种文件的 id |

**`am_motion_v3` 字段**（一个 npz，每个（片段，机器人）或（片段，机器人，checkpoint）一个）：
- `meta`（json）：各种 id、`lane`（`reference` / `model` / `sim`）、规格版本、阈值哈希、引擎或 checkpoint、机器人资产哈希、corr id、人体版本 id、`floor_z_removed`、`fps`、`frame_ids`。
- 运动：`qpos[T,nq]` f32（米、地面 z=0）、`joint_names`、`root_pos`、`root_quat`。
- 点级接触（CSR 稀疏，只存离地或离物体 ≤ 1 mm 的点）：`contact_ptr[T+1]`、`contact_idx`、`contact_target`（0 = 地面，k = 第 k 个物体）。
- 近场距离（CSR 稀疏，离接触面 < 2 cm 的点）：`near_ptr[T+1]`、`near_idx`、`near_dist_q`（uint8 对数量化）。这两项是 DATA_SPEC v2 第 3 节要求的。
- 区域标签：`region_bits[T,4]` uint16（地面接触 / 物体接触 / 打滑 / 穿透）。
- 逐帧：`pen_depth_mm`、`flags[T]`、`excluded[T]`；物体位姿 `objects[k].pose[T,7]`；`qc`（汇总）。
- 只有模型输出才有：`model_contact_head`（模型自己预测的接触，只用来对照，**判官不读**）。
- 单位一律米；只有量化字段用毫米，字段名带 `_mm` 后缀。

**`umrs_v1` 训练分片**：已验证不可变、带校验和、可 mmap、人体和 21 台按帧精确配对、姿态误差 ≤ 4.5e-5 rad、表面点 ≤ 0.062 mm、接触位按位无损、量化后可能违规的帧打标记。以后要加的（`license_id`、`dataset_id` 写进每条记录，可选的逐点接触块）只改头部和输入端，记成 `umrs_v1.1`，容器不变。
- 分片：每片 0.5–2 GB，机器人按（机器人，批次）切，人体单独成片。训练/验证/测试按 `split_group`（演员或场次）划分；留出机器人沿用 foundation 索引的"见过 / 没见过"，组合出四种评估分区。
- 旧训练缓存（`cache_amass_lconv_tfix`、`cache_lafan_v3`、`bridge_codes`、`text_align`、`x2_pool_v3`、`wbc_motions`、`ctrl_verify`）不转换，只登记成 `legacy_cache`，原地软链，**里面的软链结构不动**（它们跨分区指向 `~/tmp/cl_v3` 等）。

### C.4 磁盘预算（按今天实测）

**现在**（18:48）：`/` 657 GB，用 88%，剩 76 GB（上面 `/var/tmp/alphamotion` 352 GB）；`/home` 1.2 TB，用 70%，剩 343 GB；可用内存 42 GB。HiPHI 下载器自带水位：工厂目录 ≤ 300 GB、/home ≥ 200 GB。

**换算**：`gpu_npz_v2` 约 0.33 KB/机器人帧；`factory_npz_v1` 约 0.41–0.46 KB；UMRS 约 0.13 KB（含人体）；GPU 限速后约 2,000 机器人帧/秒。

| 规模 | 机器人帧 | GPU 时间（限速到 70% 以下） | 重定向 npz | UMRS 分片 |
|---|---|---|---|---|
| 首批平地片段约 30 h × 21 台 | 0.68 亿 | 约 9–10 h | 约 22 GB | 约 9 GB |
| 手上全部 126.4 h × 21 台 | 2.87 亿 | 约 38–45 h | 95–130 GB（不全留） | 约 36 GB |
| 已知全部 571.4 h × 21 台 | 13.0 亿 | 7–8 天 | 430–590 GB（不可能全留） | 约 165 GB |

（另加每台机器人静态准备数据的 CPU 抓取时间，试点时实测。）

**/home 的账**：343 GB − HiPHI 还要下约 145 GB ≈ 200 GB → 减 126 h 的分片 36 GB → 减重定向中间文件封顶 40 GB → 余约 125 GB。**全量 571 h（分片 165 GB + BONES 全包 45 GB）放不下**，必须先做到下面至少一件：① 删 `xwbc_full`（129 GB，在根分区，可从 HF 重下），把可重生成的东西放根分区；② 接冷存储盘，把 `archive/` 和 `topo_exp` 的约 140 GB 旧缓存挪过去。

**配额**（写在 `config/am.toml`，调度器执行；作业提交必须带 `est_out_bytes`，超额直接拒绝）：

| 类别 | 位置 | 配额 | 现在 |
|---|---|---|---|
| 原始数据 | /home（factory） | 300 GB（沿用下载器） | 约 156 GB，还在涨 |
| 人体 | /home | 80 GB | 约 28 GB |
| 机器人资产和 corr | /home | 15 GB | 约 5 GB |
| 重定向中间 npz | /home | **40 GB 滚动** | < 1 GB |
| 训练分片 | /home | 120 GB | 约 0 |
| 新训练 run | /home | 150 GB | 0 |
| 评估和媒体 | /home | 40 GB | — |
| scratch | /home | 30 GB | — |
| 根分区旧数据 | / | **不再增加** | 352 GB |

**水位**：/home 用到 80% 报警，85% 停止放行重写入作业；根分区 90% 报警。

### C.5 保留和清理

| 级别 | 包括 | 规则 |
|---|---|---|
| `keep` 永久 | 原始数据（HiPHI 过了截止就拿不回来）、license 原文、账本和备份、人工判决、拍板记录、发布物及其整条血统（run → 数据集 → 重定向 → 人体 → 原始）、冻结考卷、排行榜、报告引用的媒体 | 永不自动删 |
| `keep_n` 留 N 个 | checkpoint：最好的、最后的、每 25% 进度一个里程碑、发布用的 | run 结束 14 天后其余的进清理计划；曲线和日志永远保留 |
| `regen` 可重生成 | 已打包的重定向 npz、有 30 fps 版时的 90 fps 人体、`umr_gpu/problems`（5.7 GB）、审计网页 `cache/arrays`（3.1 GB）、没人引用的渲染 | 打包并过 14 天后进清理计划；**永远留 2% 随机样本，以及所有被人审过或被标记过的** |
| `scratch` | `scratch/` 下所有东西 | 14 天提醒，21 天进清理计划 |
| `cache` | `cache/`、HF 元数据、inductor/triton/warp 编译缓存 | 随时可删 |
| `archive` | 冻结的旧路线 | 压缩、只读；有冷存储后挪过去 |

**清理分三步，代理只能做第一步**：
1. `am s5 gc plan` 出清单（它本身也登记成产物）：每条写路径、大小、级别、谁引用、能不能重生成、重算要花多少。
2. lloyd 在网页上逐条勾选，或在本机终端 `am approve <id>`（交互确认，记 `via`）。
3. `am s5 gc apply <plan_id>`：先挪进同分区的 `.trash/<plan_id>/`，7 天后才真删。每一步记进账本。

---

## D. 模型侧

### D.1 训练 run 登记

- **提交前先估**：`am s6 train plan --family am2 --config <yaml> --dataset am_pretrain@v1` → 账本生成一个 `planned` 的 run，估 GPU 小时、kWh、电费（沿用 0903 电费 700 刀以后定的规矩：默认不开 RL，长任务先报时长）→ 生成审批请求 → lloyd 批准后才进重 GPU 队列。训练和重定向共用唯一的重 GPU 名额，由调度器排队。
- **run 目录** `runs/<族>/<run_id>/`：`RUN.json`（冻结的配置、数据集 id、代码 commit——有没提交的改动就拒绝启动、随机种子、审批号）、`metrics.jsonl`、`events.jsonl`、`ckpt/step_XXXXXXX.pt`、`live/<step>/`（中途出片）、`evals/`（指向评估 id）、自动生成的 `README.md`。
- **训练代码只接两个回调**：`reporter.on_step(metrics)`、`reporter.on_checkpoint(path, step)`，账本实时看到新 checkpoint。没改过的旧训练代码，由目录监视器兜底。
- **旧 run 一次性导入**（见 C.1 导入清单）。
- **头衔**：`models.title` 只能由人在网页上或本机 `am model crown <模型> --title Greenwich` 写入，必须带 lloyd 的审批号。对决只出"候选"。
- **license 继承**：模型用过的数据集带 `release_restrictions`（HiPHI、BONES 的"不许训练替代它的动作生成模型"），模型自动标"发布前要过法务"。

### D.2 实时可视化：边训练边看模型生成了什么

**目标**：同一条时间轴上同步播放人体源、参考（数据里的 UMR 严格重定向）、模型输出（可以多个 checkpoint）、仿真（可选）。每一路显示骨架或网格、4096 表面点、接触。

**做法**：包住产品仓 `alphamotion.viz.live.LiveViewer`（viser 常驻、就地换内容、做过手机端验收），加一个读 `am_motion_v3` 的场景适配器 `code/s6_train/viz/live/scene_am_motion.py`，复用 am2exp 的 `render/robot_model.py`（厂商 MJCF 网格）和 `render/contact.py`（按区域上色）。

```
┌──────────────┬──────────────────┬──────────────────┬──────────────┐
│ ① 人体源     │ ② 参考（对照臂）  │ ③ 模型输出（本臂） │ ④ 可选：别的   │
│ SMPL-X 网格  │ UMR 严格重定向    │ checkpoint X     │ ckpt / 机器人 │
│ 或棍状图      │ MJCF 网格+表面点  │ MJCF 网格+表面点  │（最多 6 路）   │
└──────────────┴──────────────────┴──────────────────┴──────────────┘
  时间轴 ── 每帧判官指标小曲线（穿透 / 悬空 / 打滑 / 和人一致）── 一键跳到最差帧
  开关：参考画成灰色影子叠在 ③ 上 · 表面点开关 · 只看接触点 · 模型接触头 vs 判官接触
```

- **数据通路**：模型推理 → 写一个 `am_motion_v3` 文件（`lane=model`）到 `runs/<id>/live/<step>/` 或 `evals/.../rollouts/` → 登记 → `viewer_load(scene)`。**查看器只读文件，不直接接模型**。所以同一份输出在网页 3D 审片、判官、渲染器、查看器里看到的完全一样。表面点不逐帧存，查看器按机器人资产缓存，用正向运动学现算。
- **颜色**（和审计网页一致）：绿 = 地面接触，橙 = 物体接触，红 = 穿透 > 0.5 mm，黄 = 人体在接触但机器人悬空 1–20 mm，灰 = 其他；接触点速度 > 1 cm/s 画红箭头。③ 上的接触分两种：判官现算的（实线，权威）和模型接触头的（虚线，只供对照）。
- **人体源必须显示**（lloyd 定的渲染纪律）。LAFAN 缓存的根旋转不可靠，默认棍状图；长位移时相机跟着根节点走。
- **跟训练模式**：`am s6 watch <run_id>`（light 作业）收到新 checkpoint 后：
  1. 在 `golden_quick@v1`（12 段 × 3 台：G1、H2 + 留出的 X2）上用 **CPU** 推理，窗口不超过 20 秒，不和训练抢 GPU；
  2. `geometry@v1` 打分；
  3. 出 12 帧全程网格和 10 秒三联视频，登记 `media`；
  4. 查看器自动换上最新 checkpoint。

  lloyd 问"训练怎么样了"时，`run_live_latest` 直接给最新那段视频（按纪律必须附视频）。
- **进程纪律**：全系统一个查看器实例，只监听 127.0.0.1:8781，闲置 8 小时自动退出，内存封顶 4 GB，按单元名启停。
- **网页那边**：审计网页的 Three.js 加一条"模型输出"通道，读同一种格式，用来浏览和审片；viser 查看器负责实时推送和调试。
- **合并后停掉的旧查看器**：motion-lab（:2334）、rollout-gallery（:8891）、greenwich-mobile（:8899）、studio（:8097，没在跑）、viser_mesh_viewer（:8093）、lab_server（:8114）。lab_server 的"分段选补全 + 多臂同场"、greenwich-mobile 的按字节范围发视频，写进需求。

### D.3 评估套件：模型重定向和生成动作的质量

**五条铁律**：
1. **判官独立**：只依赖 numpy、mujoco、torch 正向运动学，不 import 训练代码，不读训练损失，不读模型接触头。有 import 关系测试（J1 已有，推广到所有判官），账本 `judges.imports_training_code` 必须是 0。
2. **同一把尺**：数据（S3 的参考重定向）和模型输出用同一个判官、同一份阈值。参考数据自己的分数就是**"数据地板"**，每次评估都同时评它，先看对照再解读。
3. **真值不是唯一的尺子**：两条轴都要量——不需要真值的物理一致性，和 WBC 能不能跟住。
4. **缺材料报"未评"**（`not_evaluated`），不许默默算通过。
5. **不能光凭数字下结论**：每个判决挂三联视频和全程 12 帧网格；排行榜上一行没人看过视频（`media.viewed_by_lloyd = 0`），就不能标"通过"，也不能拿去升级模型。

**判官（7 个 + 1 个非物理）**，代码在 `code/s7_eval/judges/`：

| 判官 | 量什么 | 合并自 | 费用 |
|---|---|---|---|
| `geometry@v1` | 地面、物体、自身穿透（mm，p95、最大值、> 0.5 mm 帧数）；实接触率；悬空；接触点打滑（p95、> 1 cm/s 帧数）；接触闪烁（< 3 帧的段）；和人体接触一致（逐区域 F1） | am2exp J1 `ground_judge`（厂商 MJCF 碰撞几何、15 区域）+ `umr_gpu/eval/judge.py`（`mj_geomDistance` + 4096 稠密点），拷进 monorepo，原件不改 | CPU |
| `locomotion@v1` | WBC 可训性四张考卷；脚滑审计；接地四口径 | arena `eval/locomotion.py` + `locomotion.v1.json`（wbc_ready_audit、x2_phys、slip_audit、ground_metrics 的正式移植版） | CPU |
| `physics_nogt@v1` | 根高抖动、关节速度和加速度超限、关节限位违例；尺子 = 人体源自己 | `x2_phys` 的逻辑 | CPU |
| `pose_fidelity@v1` | 和参考比：关节角用测地线距离（避开 3 自由度欧拉分支伪造的 100° 假超限）；手的世界系位置误差（cm）和腿的角度误差（°）**分开报**；根轨迹；和人体比：肢体方向余弦 | AM2 `eval_engine`、`surface_metrics` + 新写的一层 | CPU |
| `manip@v1` | 抓握：握住、张开、压碎三档；手离物体表面距离 | am2exp J2 `closure_judge` | CPU |
| `codec_regression@v1` | 防腿、手退化：幅度比、上肢正向运动学误差、腿角警戒线、方言一致率 | `v2_judge.py` 固定窗口 | CPU |
| `wbc_track@v1` | WBC 跟踪率（mjlab BeyondMimic） | `alphamotion_ctrl` 的 `evaluate_local` | **重 GPU，每次都要批准** |
| `latent_critic`（非物理） | 潜空间打分，标明"不是物理指标" | Greenwich SDK `judge.py` | 可选 |

**冻结考卷**（`code/s7_eval/golden/*.json`，成员清单 + sha 冻结；只从不在任何训练集里的 `split_group` 抽；有泄漏测试）：

| 考卷 | 内容 | 用途 |
|---|---|---|
| `golden_quick@v1` | 12 段 × 3 台（G1、H2、留出 X2），覆盖主要大类 | 训练中途自动出片，CPU 5 分钟内跑完 |
| `golden_core@v1` | 约 200 段，按 主类 × 源 分层，含物体，含见过和留出的机器人，分成四种评估分区 | 正式比较 |
| `G-loco@v1` | 走、跑、转身（平地）；楼梯和坡等地形问题解决后出 v2 | 移动 |
| `G-ground@v1` | 跪、爬、躺、起身（手上约 0 h，先用 LAFAN 和 AMASS 留出集，HiPHI/BONES 到了升 v2） | 地面动作 |
| `G-hoi@v1` | 搬箱（箱子测试通过后）、踢球、坐椅子 | 人和物体 |
| `G-hand@v1` | HiPHI 上肢操作类 | 手 |
| `G-heldout600@legacy` | 现成的 600 段语料外审计集（时间轴已重采样修正） | 和历史结果衔接 |
| `G-unseen-robot@v1` | 零样本机器人（X2、A3） | 跨具身 |
| `G-random@v1` | 50 个无偏随机窗口 | 防挑片 |
| `parity3@v1` | 引擎对拍用例（LAFAN 舞蹈、HiPHI 起身、Omni 踢球 × 4 台） | S3 引擎门 |

**协议和统计**（照搬 am2exp E5 统计层 `cases`、`stats`、`decide()`）：
- `code/s7_eval/protocol/motion_eval_v1.yaml` 对应 motion-eval 技能的六项纪律（数值、视觉、统计、信息熵、跨具身、动作泛化）；`spatial_eval_v1.yaml` 对应 spatial-eval 技能的五项必考。
- 对照臂纪律：数据地板、随机方向或打乱来源、同尺重测。
- A/B 判决只走 `stats.decide()`：按窗口聚类的配对符号检验 + 置换检验，规则冻结在代码里。结论只有 `win` / `lose` / `tie` / `not_evaluated` 四种。

**三联视频和 12 帧网格**（`code/s7_eval/render/`，合并 am2exp `render/triptych_*`、`grid12.py`、rollout_tools 的 `encode.py`（逐帧时间戳校验）、`render_check.py` 的接触和穿透上色）：
- 布局固定 **人体源 | 对照臂 | 本臂**，1080p，真实时间戳；
- 12 帧网格均匀覆盖全程，首帧和末帧都在；
- 另出 `poster.jpg` 和 `manifest.json`，登记 `media`；
- 给 lloyd 的视频一律走这一个入口，`media_get` 返回本地路径，用 SendUserFile 发 mp4。
- `showcase_render` 留着专门做展示。

**排行榜**（`/api/v1/leaderboard?suite=&robot=`，网页一页）：
- 只把同一套件版本、同一考卷版本的结果放在一起比；
- 固定显示基线行：**数据地板**（`umr_gpu@<commit>` 严格版）、**Native policy**（GMR 原生）、xwbc；
- 每行：被评对象、判官版本、机器人、穿透 p95、实接触率、悬空、打滑 p95、闪烁、人体一致 F1、手 cm / 腿 °、WBC 跟踪率（跑过才有）、案例数、未评数、和数据地板比、统计判决、**视频谁看过**、头衔；
- 排序先看判决，再看指标，不按平均分排。

**发布门**：产品仓 `quality.release_passed` 和浏览器验收，接成套件 `release@v1`。

---

## E. 迁移计划

### E.0 规矩

1. **不打断在跑的东西**：HiPHI 下载器（到截止）、GRAIL 下载/转换/碰撞/清单循环（`--max-hours 24`，约 10-08 晚结束）、`gpu_convert` 定时转换、`umr_gpu` 作者的吞吐测试和改写、`throttle_guard`、robots_wf 和 diversity_wf 代理、`acquire.py` 和 `queue_followup.py`、3 个常驻服务。它们写死的路径，在自然结束前一律不动。
2. **对拍过的引擎**：原地 `git init` 只增加 `.git`，代码一个字节不改（要 lloyd 同意）。之后每改一处都重跑对拍：`umr_gpu` 的 parity 表、`gpu_convert` 的 `parity40`、`umr_dataset` 的往返和严格复验。
3. **动手前检查**（`am ops precheck <路径>`）：扫 `/proc/*/cwd`、`/proc/*/cmdline`、`/proc/*/fd` 看有没有进程在用（不用 `lsof +D`，太重）；查 `legacy_paths` 看有没有代码写死它；查有没有软链指向它（如 `cache_lafan_v3 → ~/tmp/cl_v3`）。
4. **搬的方式**：先软链挂上 → 代码改成读配置 → 进程空闲时同分区 `mv` → 旧位置留反向软链至少 30 天 → 再撤。跨分区搬算重活，要批准。
5. **conda 环境不能直接挪**（脚本头写死了绝对路径）：在 /home 上按锁文件重建，切过去，旧环境原地留一个月。`GENMO/` 是可编辑安装的，在新环境里重装。
6. **删除只提议**：标 `delete_candidate`，进清理计划，等 lloyd 逐条批。本设计不执行任何删除。
7. **HiPHI 截止前**：不做大规模搬家、不做大量哈希计算、不加网络负载。

### E.1 阶段和时间

| 阶段 | 什么时候 | 做什么 | 代理工时 | 风险 |
|---|---|---|---|---|
| **P0 急事 + 骨架** | 今天 | ① lloyd 拍板第 0 节 3 件急事（下载让给 HiPHI、停 ardy-isaac、GPU 只留一个重任务）；② 建 `~/alphamotion` 骨架、`README.md`、`config/` 初稿、`stores/` 和 `repos/` 软链；③ 4 张地图、两个方案、本文、`FOLDER_INDEX.md` 拷进 `docs/maps/`；④ 从孪生目录救出 3 样东西（只拷贝：`SCALING_LAW.md`、另一版 `CTRL_VERIFY_GR3.md`、x2 视频）；⑤ datagrid 唯一的数据库备份 `~/tmp/arena_handoff_backup/` 和 `greenwich_workbench_0911/datagrid` 各多拷一份到 `services/datagrid/backups/` | 1 h | 零：只新建、只拷贝 |
| **P1 记账** | 第 1 天（10-08） | ① 6 个代码目录原地 `git init` + 快照 tag（`umr_factory_code`、`umr_gpu`、`gpu_convert`、`umr_dataset`、`motion_audit_web`、工厂 `scripts*`），`.gitignore` 排除产出；`~/reference_code/ufo` 里我们自己没提交的配置先拷进 `code/s7_eval/wbc/bfm_zero_overlay/`；② 账本建表 + 全部导入器；③ `paths.toml`；④ `am what`、`am index` 生成第一版 `INDEX.md`；⑤ 写死路径扫描（`am paths lint`）把引用次数写进 `legacy_paths`；⑥ 小而承重的旧件拷进 `data/` 并留软链：`sole_offsets*.json`、`gauge_check_0903.json`、`dialect/`、`robots.json` + `umr_configs`、`sources/`、`~/Downloads/SMPLX_NEUTRAL.npz` | 1 天 | 低 |
| **P2 能量产** | 第 1–3 天（10-08 – 10-10） | ① 读取层（人体 4 种 + 动作 v1/v2 + meta）；② 阈值文件 + "引擎常数 = 配置"测试；③ GPU 小测三件（共 ≤ 1 h GPU，排在吞吐测试之后）：快照 commit 重跑对拍、合成箱子、地形当物体；④ `run_clips` 真实片段入口 + 等价测试 + 静态准备数据抓取；⑤ 重定向适配器；⑥ `umr_dataset` 改走读取层（HiPHI 不再 FileNotFound）；⑦ **试点**：4 台（G1-29、GR3、H2、A3）× 100 段平地片段，独立判官全量复核，三联视频抽 20 段给 lloyd | 2 天 | 中：第一次量产，靠等价测试和试点视频兜底 |
| **P3 量产 + 服务** | 第 3–6 天（10-10 – 10-13） | ① lloyd 点头后放开到 21 台：先 HiPHI 平地（通过物体测试的）+ GRAIL 平地 + Omni 踢球，约 30 h 人体，GPU 约 10 h；之后手上其余平地片段，GPU 共约 38–45 h；② 作业层 + 调度器 + systemd 单元（下载器、worker、服务都改成单元）；③ 服务：先只读 HTTP + MCP（并入现有 15 个工具），再开写操作；冻结 OpenAPI 交给 Codex；④ `geometry@v1` 判官合并；⑤ 三联渲染入口；⑥ S4 质量汇总、人审入队、放行门；⑦ 第一个训练包 `am_pretrain@v0.1` | 4 天 | 低到中 |
| **P4 模型侧 + 清路径** | 第 2 周（10-14 – 10-20） | ① 查看器适配 + watcher；② 其余判官、冻结考卷、排行榜；③ 模型和 run 登记导入完；④ 6 个旧查看器内容并进网页后停掉（停服务要 lloyd 批）；⑤ 下载器和 GRAIL 循环结束后，逐个替换写死路径（工厂 19 处、arena 10 处、Studio 7 处、`umrf/paths.py`、`gpu_convert` 等，按 `am paths lint` 清单打勾），每改一处过对拍门；⑥ 可选：GPU 导出改成直接写 `am_motion_v3`（过"导出等价门"、lloyd 同意后） | 4 天 | 中：改路径靠对拍门兜底 |
| **P5 搬家** | 第 2–3 周 | ① /home 内同分区搬：`~/umr_factory_data/*` 按子目录 `mv` 到 `data/s*`，旧位置留反向软链；4 个引擎仓 `mv` 进 `repos/`，`~/umr_gpu` 等留反向软链；② /var/tmp 小件生产资产搬到 /home：UMR 环境按锁文件重建到 `envs/umr`、UMR 克隆 1.2 GB、`umr_gpu` 产出 7.7 GB，原地留软链；③ `cl_v3` 和 `topo_exp` 大缓存**不搬**，等冷存储 | 2 天 | 中 |
| **P6 收尾** | 第 3 周起 | 归档清单和删除候选（约 210 GB）交 lloyd 逐条批；商业资料挪到 `~/business/`；冷存储到位后挪 `archive/` 和旧缓存 | 按批 | 低 |

**合计**：代理工作约 14 天，日历约 3 周（要等下载器和循环结束）。

**里程碑**：
- **第 1 天结束**：lloyd 能看 `INDEX.md`、能用 `am what` 查任何文件夹。
- **第 2–3 天**：第一批试点重定向数据和视频给 lloyd。
- **第 4–5 天**：21 台量产开跑；只读网页和 MCP 上线。
- **第 6 天左右**：第一个训练包 `am_pretrain@v0.1`。
- **第 2 周末**：训练中途自动出片、排行榜、旧查看器停掉。
- **第 3 周末**：数据都在一个根下（大缓存除外），根分区回到 85% 以下。

### E.2 首批量产路线

1. 等 `umr_gpu` 作者这一轮吞吐测试结束；作业层接管前，靠人工约定：重定向跑的时候暂停 `gpu_convert` 的 `run_watch.sh`，保证同时只有一个重 GPU 任务。
2. P2 三个 GPU 小测（对拍、箱子、地形）都有结论。
3. `am s3 plan --clips "has_terrain=0 AND status=eligible AND source IN (hiphi, grail, omnicontact)" --robots g1_29,gr3,h2,a3 --limit 100`，其中排除没过箱子测试的箱子片段。
4. 试点 4 台 × 100 段：独立判官全量复核；三联视频抽 20 段给 lloyd（SendUserFile 发 mp4，不只给路径）。
5. lloyd 点头后放开到 21 台；人审队列放 50 段按风险挑的 + 50 段随机的；随机合格率过放行门，这批才打包。
6. 箱子测试通过后，再排 OmniContact 搬箱（12 h 人体，约 3 h GPU）；地形问题解决后再排 GRAIL 楼梯和路沿。

### E.3 每个现有文件夹的去向

逐个文件夹的说明和去向在 **`FOLDER_INDEX.md`**。动作只有 7 种：
- **登记**：原地不动，进账本；
- **链入**：在根目录下建软链，原地不动；
- **拷贝**：拷一份到根目录，原处留软链（小件）；
- **搬**：同分区 `mv`，原处留反向软链；
- **归档**：挪进 `archive/`，只读；
- **删除候选**：只提议，等 lloyd 批；
- **不进根**：和 AlphaMotion 无关，或是商业资料。

### E.4 不能出错的几件事

- **`cl_v3`（31 GB）不搬**：`cache_lafan_v3` 和 `cache_amass_lconv_tfix` 里有 34 个软链指向它。真要搬，必须同时改写这些软链，并用训练加载器读一遍做校验。
- **`cache_xwbc_amass_full`（61 GB）看起来像旧版，其实是承重的**，被现役缓存链接着，不能删。
- **两个服务目录**：motion-lab 读 `alpha_motion2/demo/ardy_preflight_20`，gallery 读 `demo/alphamotion_verified_g1_atlas_20`。先合并再停服务，停之前不动。
- **3 个指向主仓的软链**（`~/.alphamotion_workspace`、`~/alpha-motion-workspace`、`~/alphamotion2_runtime`）是服务启动用的，服务全换新之前不能删。
- **`~/umr_gpu` 正在被改**：生产只从冻结 worktree 跑，不从开发目录跑。
- **写死路径**（P4 改完，用 `am paths lint` 复查到 0，第三方仓库除外）：`umr_factory_data` 里 19 处指向 GMR；arena-datagrid 6 + 4 处；Studio 7 处；主仓 141 处以上指向 `/var/tmp/alphamotion/topo_exp`；`umrf/paths.py`；`umr_gpu` 22 处；`~/Downloads/SMPLX_NEUTRAL.npz`；`motion_audit_web` 的 `DATA_ROOT` 和 `configs/mcp.json`；`run_parity.py` 里的 `lloyd_gpu_eval`；`diversity_wf.js` 里的临时目录。
- **`github-recovery-codes.txt` 不要碰**。

### E.5 每个阶段的验收

- **P0**：根目录里所有软链都能解析；`README.md` 在；原有进程和服务一个没断（对比前后 `systemctl --user` 和 `pgrep`）；HiPHI 下载速度回到 ≥ 2 MB/s。
- **P1**：账本条数和地图对得上（GRAIL 22,312、机器人 34、foundation 240,444、topo runs 823 个条目）；6 个快照 tag 打好；`am what ~/tmp/cl_v3` 能说清楚；`INDEX.md` 生成。
- **P2**：快照 commit 对拍通过；`run_clips` 等价测试通过；箱子和地形各有结论（写进 `docs/decisions/`）；试点 400 条过独立判官，lloyd 看过 20 段视频；`umr_dataset` 打包成功，HiPHI 段不再 FileNotFound。
- **P3**：21 台量产按计划跑完、无同时两个重 GPU 任务（看 `throttle.log`）；MCP 现有测试 `test_mcp.py` 全过，HTTP 冒烟通过；`am_pretrain@v0.1` 校验通过。
- **P4**：watcher 在一个 run 上自动出片；排行榜有数据地板和 Native policy 两行；`am paths lint` 清零（第三方除外）；引擎对拍回归通过。
- **P5**：旧路径的反向软链都在；根分区 < 85%。

---

## F. 第一批工作包（文件互不重叠，可以并行）

**共同规矩**：只写 `~/alphamotion/` 下自己名下的文件；旧位置一律只读；不改引擎原件；不碰正在写的目录；重命令一律 `taskset -c 0-15 nice -n 10 timeout …`；要用 GPU 的包开跑前先报时长、经 GPU 租约排队；`code/amp/catalog/schema/*.sql` 只归 WP-01，别的包要加表提给它；每个包自带测试，`pytest -q <自己的测试>` 全绿算验收；CLI 子命令写在各模块自己的 `cli.py`，自动发现，不抢同一个文件。

| 包 | 内容 | 只写这些文件 | 依赖 | 工时 | 验收 |
|---|---|---|---|---|---|
| WP-00 骨架 | 根目录、软链、README、`config/*.toml` 初稿、文档拷贝、孪生目录救援、datagrid 备份副本 | `README.md` · `CLAUDE.md` · `config/{am,paths}.toml`（之后 paths 归 WP-02）· `docs/**` · `stores/*` · `repos/*` · `services/datagrid/backups/` | 拍板（根目录名） | 1 h | E.5 P0 |
| WP-01 账本 | 建表、迁移、备份（含拷到第二分区）、`amp.catalog` API、事件 | `code/amp/catalog/{schema/*.sql,db.py,migrate.py,backup.py}` · `code/amp/tests/test_catalog.py` | WP-00 | 4 h | 约束生效（判官 import 训练代码拒写、审计判决只能人写）；多进程并发读写测试 |
| WP-02 路径 | `paths.toml` 解析、`resolve`、`lint`、`precheck`（扫 /proc） | `code/amp/paths.py` · `code/amp/ops/precheck.py` · `config/paths.toml` | WP-00 | 3 h | lint 报出地图里列的全部写死位置；precheck 能发现在跑进程的工作目录 |
| WP-03 导入器 | 只读导入各现有台账（C.1 清单） | `code/amp/importers/*.py` · `code/amp/tests/test_importers.py` | WP-01 | 6 h | E.5 P1 的条数；跑两遍结果一样 |
| WP-04 读取层 | `formats.human`（4 种存法）、`formats.motion`（v1、v2+meta、v3 读写）、格式规格文档 | `code/amp/formats/{human,motion}.py` · `docs/specs/FORMAT_{human_smplx_v1,am_motion_v3}.md` · `code/amp/tests/test_formats.py` | — | 6 h | 23 个现有样本（11 + 12）读出来字段一致、单位米；packbits 和 CSR 互转按位一致 |
| WP-05 阈值 | `data_spec_v2.toml`、"引擎常数 = 配置"测试、重复常数扫描 | `config/gauges/data_spec_v2.toml` · `code/s3_retarget/tests/test_thresholds.py` | — | 2 h | 测试能读出引擎 `strict.py` 里的实际常数并和配置比对 |
| WP-06 引擎三小测 | 快照 commit 重跑对拍、合成箱子、地形当物体（GPU 共 ≤ 1 h） | `code/s3_retarget/tests/{test_parity_snapshot,test_box_object,test_terrain_object}.py` · `docs/decisions/2026-10-08_{engine_parity,box,terrain}.md` | WP-00；GPU 空闲 | 4 h | 三份结论；对拍通过才把 `engines.parity_status` 记成 `verified` |
| WP-07 corr 差异 | 两份 corr 逐字段对比（CPU，只读） | `code/s2_surface/tools/corr_diff.py` · `data/s4_dataset_eval/2026-10-08_corr_diff/` | — | 2 h | 讲清差在哪、影不影响对拍 |
| WP-08 真实片段入口 | `run_clips`（读账本人体 → 引擎求解 → 导出）+ 静态准备数据抓取 + 等价测试 | `code/s3_retarget/adapters/{run_clips,prep}.py` · `code/s3_retarget/tests/test_run_clips_equiv.py` | WP-04、WP-06 | 1 天 | 3 段对拍片段上和 `run_one` 一致；`umr_gpu.__file__` 在 worktree 里 |
| WP-09 重定向适配器 | plan / post（`.meta.json`）/ register / enqueue_audit | `code/s3_retarget/adapters/{plan,post,register,audit}.py` · `code/s3_retarget/tests/test_adapter.py` | WP-01、WP-04、WP-08 | 5 h | 小批次端到端；worktree 前后快照不变 |
| WP-10 作业层 | jobs 表 + 租约 + 守卫 + 回执 + 审批 + 资源类 + 调度器 + boot_id 恢复 + 断网暂停 + systemd 单元模板 | `code/amp/jobs/{store,worker,governor,approvals}.py` · `code/ops/systemd/*.service,*.slice` · `code/amp/tests/test_jobs.py` | WP-01 | 1 天 | 两个重 GPU 作业不会同时跑；`net=down` 时暂停；重启后 resumable 作业自动重排；代理不能批准；单元有 StartLimitBurst |
| WP-11 服务 | 并入 `agent_tools`（工具名全保留）+ FastAPI 路由 + OpenAPI 冻结 + 人登录 | `code/amp/service/{core,api,mcp,auth,files}.py` · `docs/contracts/{openapi_v1.yaml,mcp_tools_v1.md}` | WP-01、04、10；确认没人还在写 `agent_tools/` | 1.5 天 | 现有 `test_mcp.py` 全过；HTTP 冒烟；视频按字节范围取 |
| WP-12 几何判官 | `geometry@v1`（J1 + umr_gpu judge 拷贝合并），阈值读配置，import 关系测试 | `code/s7_eval/judges/geometry/**` · `code/s7_eval/tests/test_geometry.py` | WP-04、WP-05 | 1 天 | 在对拍样本上和原两把尺结果一致（容差内）；能抓出 CPU 的箱子当球 |
| WP-13 渲染入口 | triptych、grid12、encode（am2exp + rollout_tools 拷贝） | `code/s7_eval/render/**` | WP-04 | 5 h | 时间戳逐帧校验；12 帧含首帧末帧；布局固定人体源 \| 对照臂 \| 本臂 |
| WP-14 数据评估 | 普查、`TAXONOMY_v1`、质量汇总、人审入队和读回、放行门 | `code/s4_dataset_eval/**` · `docs/specs/TAXONOMY_v1.md` | WP-01、03 | 1 天 | 小时数守恒；过期判决标记正确；和今天 `REPORT.md` 对账 |
| WP-15 索引 | `am index`（`INDEX.md`）、`am what`、`am doctor` | `code/amp/index.py` · `code/amp/what.py` · `code/amp/doctor.py` | WP-01、03 | 4 h | 每个顶层目录都有一行；`am what` 对地图里任意路径都能答；缺 MANIFEST 能报出 |
| WP-16 查看器适配（第 2 周） | LiveViewer + `am_motion_v3` 场景 + watcher | `code/s6_train/viz/live/**` · `code/s6_train/watch.py` | WP-04、WP-13 | 1.5 天 | 加载一段截图，帧号和时间轴对得上；watcher 只用 CPU |
| WP-17 网页（Codex） | 按 OpenAPI 做 B.8 的 15 个页面 | `code/web/frontend/**` | WP-11 冻结 OpenAPI | 3–5 天 | 浏览器验收截图；手机能拖视频进度 |

**关键路径**：WP-00 → WP-01 →（WP-03、WP-04、WP-05、WP-06、WP-07 并行）→ WP-08 → WP-09 → 试点。WP-10 到 WP-15 在第 3–6 天并行；WP-16、WP-17 第 2 周。

---

## G. 需要 lloyd 拍板的事

**今天（急）**：
1. **下载**：暂停 GRAIL 下载（aria2 `-j48`）和 `pretraining_foundation/acquire.py`，16 个连接全给 HiPHI，直到截止（约 10-09 早上 6 点）。理由：按现在的分配 HiPHI 赶不上截止，GRAIL 和 HF 上的东西没有截止。
2. **停服务**：`systemctl --user disable --now alphamotion-ardy-isaac`（已重启 642 次，依赖已不存在）。
3. **GPU**：同一时间只留一个重 GPU 任务。`umr_gpu` 作者的吞吐测试跑完前，暂停 `gpu_convert` 的 `run_watch.sh`。
4. **代码快照**：同意在 6 个没 git 的代码目录原地 `git init`（只增加 `.git` 目录，不动任何文件）。

**这周**：
5. **根目录名字和位置**：建议 `/home/arenalabs/alphamotion`（命令 `am`）。备选 `am_platform`、`arena_motion`。
6. **生产格式**：S3 生产只认 GPU 引擎（冻结 commit、重新对拍通过）；CPU 工厂只当参考和量尺，它的输出不算 v2 数据。
7. **首批量产范围**：只放平地片段；地形（GRAIL 楼梯、路沿、坡）和搬箱等小测有结论再排。
8. **放行门阈值**：随机抽样合格率的 Wilson 下界，建议 0.90。
9. **license**：HiPHI 写明"为商业产品做内部研发也算商用"、不许再分发；HiPHI 和 BONES 都不许训练能替代它们的动作生成模型。建议先过法务；在那之前数据集默认"只做研究、不发布权重"，用到它们的模型自动标"发布前过法务"。
10. **网页底座**：用 `motion_audit_web`（已有审计页和 MCP），Codex 照 OpenAPI 接着写；arena-datagrid 的 Electron 前端不再接着做。
11. **审批范围**：GPU 训练、WBC 闭环评测、超过 4 小时的 GPU 作业、新引擎版本第一批、批次放行、清理执行、跨分区搬家、加冕、停服务，默认都要你批。
12. **表面点只留一套**：留工厂的 `surface.npz`，`surface_v2` 归档。

**之后（逐条批）**：
13. **根分区**：删 `/var/tmp/alphamotion/xwbc_full`（129 GB，可从 HF 重下），或接一块 ≥ 4 TB 冷存储盘。全量 571 h 数据必须二选一。
14. **删除候选约 210 GB**（清单在 `FOLDER_INDEX.md` 第 10 节）：`xwbc_full` 129 GB、`~/tmp/cl_v3/amass` 15 GB、`ufo_resume_drill` + `ufo_h2_smoke` 13.5 GB、`any_body/env` 12 GB、`alphamotion_ctrl/.git` gc 约 9 GB、`reference_code/ufo` 多的一套 venv 7.5–10 GB、`alpha_motion2/runs/synergy` 9.8 GB、`experiment/kimodo/.venv` 5.3 GB、`gpu_convert/runs/bench*` 约 5 GB，以及一批小件和坏链。
15. **商业资料**（路演、市场、股权、外联名单、投资人视频）挪出研究仓，放 `~/business/`。
16. **研究主仓改名**（去掉空格、逗号、引号）：等 3 个旧服务停掉以后再做，旧路径留软链。

---

## 附：这个方案刻意不做的事

- 不重写任何引擎；不改引擎导出格式（除非过了导出等价门、你同意）。
- 不做内容寻址 blob 库，不上 Postgres、Kubernetes、消息队列。
- 不动旧训练缓存内部的软链结构，不重建 v0.2 时代的数据。
- MCP 不开放删除、批准、加冕、人工判决、任意 shell。
- 不在根分区上新增写入。
- HiPHI 截止前不做任何会抢网络或磁盘的事。
