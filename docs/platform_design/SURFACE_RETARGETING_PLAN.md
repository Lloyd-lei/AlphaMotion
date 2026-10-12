# surface-retargeting 方案（等 lloyd 批）

全程只读，还没动任何东西。下面的数字都来自 8 份盘点，没核实的会标"未核实"。

## 0. 先说结论
- **代码比数据金贵。** UMR 相关的代码散在 8 个目录里，一行都没进过 git，总共不到 5 MB。所以第一步是先把代码收进仓库。
- **大块数据都可以上云重算或重下，不用留在本机。** 真正费时间换来的是手调配置、看图得出的结论、license 记录和验证证据，加起来约 1–3 GB。
- **量产数据其实还没产出来。** production/v1 自己标了作废；CPU 那边只有 11 个样本，GPU 那边只有 12 对。所以不存在"丢了要重算很久"的量产结果。
- **给 LLM 项目让路很简单。** 显卡现在本来就全空。只要停掉一个一直崩溃重启的服务和一个闲着的网页服务。大文件放根分区（还剩 231G）。

---

## 1. surface-retargeting 仓库

### 1.1 上游 UMR 怎么放：子模块，不改它的代码
- 放法：`third_party/UMR`，用 git 子模块固定在 `d1cdaed`。
- 理由：我们**没改过上游任何一个文件**（git diff 是空的），只是加了 3 样东西。所以不需要补丁集，固定版本就够了。
- 加的 3 样东西搬进我们自己仓库：
  - GR3 配置（`humanoid_retarget_fourier_gr3_umrfactory.json`）
  - GR3 v2.1.1 网格（40 MB，license 未核实）。先放私有存储，用一个拉取脚本取。
  - `data/correspondence_*.npz`：几秒就能重算，不收。
- HiPHI 转换器 `hiphi2smplx` 同样用子模块，固定在 `65e5131`（MIT）。
  - GPU 转换器调用了它内部的私有函数，所以必须锁死版本。
  - 本地删掉的几个 webm 预览不管。

### 1.2 目录结构（按 UMR 自己的流程排）
```
surface-retargeting/
├─ third_party/UMR            子模块 @d1cdaed
├─ third_party/hiphi2smplx    子模块 @65e5131
├─ configs/
│  ├─ sr.example.toml         唯一的路径配置（见 1.4）
│  ├─ robots/registry.json    34 台登记表，路径改成相对路径 + sha256
│  ├─ robots/<r>/             staging / umr_robot / umr_shared_beta0 / pipeline.json（小 json）
│  ├─ umr_configs/            30 份 UMR 机器人配置
│  └─ ablations/              10-04 的 5 份对照配置，只留改动项
├─ surface_retargeting/
│  ├─ config.py               读配置和环境变量
│  ├─ ingest/                 ① 下载 + 人体转换
│  │   download/ convert_cpu/ convert_gpu/ grail/ hiphi/ uniml3d/
│  ├─ robots/                 ② 机器人暂存、4096 个表面点、登记工具
│  ├─ correspondence/         ③ 人体↔机器人对应关系
│  ├─ retarget/               ④ CPU 参考版（包一层 UMR）
│  ├─ gpu/                    ④ GPU 引擎（求解器 / IPM / 运动学 / LQR / 碰撞）
│  ├─ strict/                 ⑤ 严格后处理（不穿透、滑移规则）
│  ├─ labels/                 ⑤ 接触标签
│  ├─ qa/                     ⑥ 统一判官（5 台机器人各写的一套合成一个）
│  ├─ dataset/                ⑦ 打包、加载
│  └─ curate/ census/         截断规则、多样性普查
├─ scripts/                   按阶段编号的命令：00_stage … 06_export、capture、render
├─ production/                量产调度（任务清单、启动、状态、停止）
├─ viewer/                    ⑧ 审计网页：backend / frontend / mcp
├─ ops/                       throttle_guard、各下载启动脚本
├─ tests/  experiments/  archive/  docs/
└─ pyproject.toml + environment.lock
```
做成可以安装的包，这样各处手写的 `sys.path.insert` 就都不需要了。

### 1.3 旧路径 → 新路径

| 旧路径 | 新位置 | 备注 |
|---|---|---|
| `~/umr_gpu/umr_gpu/` | `surface_retargeting/gpu/` + `strict/` + `export.py` | 5538 行，核心 |
| `~/umr_gpu/{capture,eval,production,scripts,tests}` | `scripts/capture`、`qa/`、`production/`、`scripts/`、`tests/gpu/` | `factory_ruler_snapshot` 不收，改用 git 标签 `ruler-v0` 代替 |
| `~/umr_gpu/README.md` | `docs/gpu_engine.md` + `docs/CHANGELOG.md` | 38 KB 的实测记录 |
| `~/umr_factory_code/umrf/` | `robots/`、`correspondence/`、`retarget/`、`strict/`、`labels/` | `.bak_taskB` 那一版用作 `ruler-v0` 标签 |
| `~/umr_factory_code/{scripts,tests}` | `scripts/00–04`、`tests/strict/` | |
| `/var/tmp/.../umr/tools/{umr_pipe,umr_run}.py` | `retarget/umr_wrappers/` | CPU 工厂离不开它们，全机只有这一份 |
| `/var/tmp/.../umr/tools/{gr3_tpose,tpose_ratio}.py` | `robots/tools/` | 其余评测小工具进 `archive/` |
| `/var/tmp/.../robots/{robots.json,umr_configs,inventory.py,finalize.py,render_tpose.py}` | `configs/robots/`、`configs/umr_configs/`、`robots/registry/` | 删掉 inventory 里对 topo_exp 的引用 |
| `~/umr_dataset/umr_dataset/` + `scripts/` + `tests/` | `dataset/`、`scripts/dataset/`、`tests/dataset/` | |
| `~/gpu_convert/*.py` | `ingest/convert_gpu/`；对拍和基准脚本放 `tests/convert/` | v1 和 `gpu_ik_ref` 进 `archive/` |
| `~/umr_factory_data/scripts/` | `ingest/convert_cpu/`（bvh2smplx、smplx_qa、convert_omni 是承重的库）、`ingest/download/`、`ops/` | 一次性的 shell 脚本进 `archive/` |
| `~/umr_factory_data/scripts_grail/` | `ingest/grail/` | 两个 `.bak` 不收 |
| `~/umr_factory_data/scripts_streamc/` | `ingest/hiphi/` | |
| `/var/tmp/.../h2s_shim` | 不收，改成环境里直接装 h5py | 这个垫片会把真的 h5py 挡掉 |
| `~/umr_factory_data/robots/<r>/pipeline/*.py`（5 台，约 9000 行，大量重复） | 合成一个 `qa/`；原样快照放 `archive/robots_wf_2026-10/` | 合并之前，各台的数字不要互相比 |
| `~/umr_factory_data/experiments/truncation` | `experiments/truncation/`，规则抽到 `curate/` | 只收代码、`PROGRESS.md`、定稿规则 |
| `~/umr_factory_data/diversity` | `census/` + `docs/census/` | |
| `~/motion_audit_web/` | `viewer/`（backend、frontend 源码和 lock、mcp、tests） | 去掉 mcp.json 里 AlphaMotion 的工作流 |
| `/var/tmp/.../uniml3d_reload/proto` | `ingest/uniml3d/` | |
| `/var/tmp/.../labels`（pcl_contact 原型） | `archive/pcl_contact_proto/` | 已被取代，规格文档留作参考 |
| `/var/tmp/.../lloyd_gpu_eval/*.py` | `archive/` | 已被 `render_compare` 取代 |
| `~/.claude/projects/.../workflows/scripts/*umr*.js`（8 个）+ `umr_factory_data/workflows/` | `docs/agent_workflows/`，只读归档 | 这是"东西怎么做出来的"唯一记录，可能被自动清理 |
| `am2exp_v1/assets/{umr_sampler,umr_reference}.py` + 采样器测试 | `robots/sampling_reference/` | 和上游逐位对拍用 |
| `hand_surface_case_20261004` | `examples/hand_surface/`，只收代码 | |
| `audit/SCHEMA.md`、`CODEX_PROMPT.md`、`platform_design/*.md`、`DATA_SPEC_v2.md`、`RESOURCE_LIMITS.md` | `docs/` | |

### 1.4 写死的路径怎么改
- 已数到至少 136 处（gpu_convert、工厂脚本、viewer 还没数），分布：`umr_gpu` 91 处、`umrf` 全集中在 `paths.py`、`gpu_convert`、工厂脚本、`viewer`、`robots.json` 和 `pipeline.json`。
- 统一换成一个配置文件 `configs/sr.toml`，环境变量可以覆盖。只有 6 个变量：

| 变量 | 本机取值 |
|---|---|
| `SR_DATA_ROOT` | `~/umr_factory_data` |
| `SR_WORK` | `/var/tmp/alphamotion/umr_factory/umr_gpu` |
| `SR_UMR_ROOT` | `third_party/UMR` |
| `SR_BODY_MODELS` | SMPL-X、beta_fit 所在目录 |
| `SR_ROBOT_ASSETS_ROOT` | GMR / robot_models |
| `SR_PYTHON_ENV` | UMR 环境 |

- 本机默认值和现在的路径一样，所以迁移完行为不变。

### 1.5 迁移时别把正在跑的东西弄断
- 现在 UMR 只有一个进程在跑：throttle_guard（10-12 12:30 被别的会话拉起来，寿命 48 小时）。下载、量产、转换都停着。
- 做法是"先复制，再切换"：
  1. 新仓库在 `~/Desktop/surface-retargeting`，代码是**复制**过去的，旧目录一律不动、不删。
  2. 新仓库用本机默认路径跑通对拍以后，旧的入口脚本再改成调用新仓库的薄壳。
  3. 也可以加软链 `~/umr_gpu → 新仓库`。
  4. 旧目录等 lloyd 点头后再删。
- **新仓库在本机只跑 CPU 小测试，GPU 测试留到云上。**

### 1.6 git 历史
- 这些代码从来没有过 git，没有历史可以保留。
- 第 1 个提交：判官冻结版（`.bak_taskB`），打标签 `ruler-v0`。
- 第 2 个提交：当前全部代码（10-11 的状态）。
- 私有仓库 `github.com/Lloyd-lei/surface-retargeting`。纯代码不到 5 MB，WiFi 推得动。

### 1.7 不进 git 的东西

| 类别 | 怎么处理 |
|---|---|
| 各种 venv 和 conda 环境（UMR env 6.7G、genmo-gem 11G、viewer 的 .venv/.mcp-venv） | 先导出 `environment.lock`。版本组合：py3.12、torch 2.14.1、warp 1.18.0、mujoco 3.3.7、clarabel 0.11.1。**一定要在删环境之前导出** |
| 缓存（`.inductor_cache`、viewer 的 `cache/`、`node_modules`、`dist`） | 重新生成 |
| 所有输出（problems、videos、runs、out、production、bench） | 见第 2 节 |
| SMPL-X 模型、`smplx_parts_segm.pkl`、`beta_fit`（gmm_08.pkl 等） | 授权不许再分发。只写下载说明，自己留私有备份 |
| 厂商机器人网格、GMR、GR3 网格 | 用拉取脚本取，记 sha256 |

### 1.8 迁移步骤

| 步 | 内容 | 大约耗时 |
|---|---|---|
| 1 | 导出 UMR / genmo-gem 两个环境的 lock 文件 | 10 分钟 |
| 2 | 建仓库骨架和两个子模块，复制代码，做 `ruler-v0` 和现状两个提交 | 1–2 小时 |
| 3 | 把写死的路径换成 `config.py`，加 pyproject，去掉 `sys.path` | 半天到 1 天 |
| 4 | 本机 CPU 冒烟：判官、标签、打包测试、viewer 测试；缺数据的测试标成跳过 | 1 小时 |
| 5 | 推到私有 GitHub | 几分钟 |
| 6 | 修已知的 bug（见 4.8）、5 套 QA 合成一个、两份对应关系缓存合成一份 | 1–2 天，不急 |
| 7 | 云上：重建数据并跑 GPU 对拍，和现在的对拍表比 | 半天 |

---

## 2. 数据

### 2.1 派生数据怎么处理

| 数据 | 大小 | 当初花了多久 | 不能重算的部分 | 上云重算 | 判决 |
|---|---|---|---|---|---|
| 21 台机器人资产和配置（`umr_factory_data/robots`） | 1.1G | 采样只要几分钟；手调配置花了几天 | T-pose、碰撞排除对、pipeline.json | 配置不能自动重算 | **保留**（每台的 pipeline/runs 约 0.4G 丢掉） |
| 登记表 + 30 份 UMR 配置（`/var/tmp/.../robots`） | 3.4M | 10-04 侦察 | 选哪 21 台、配置草稿 | 不能 | **保留** |
| GRAIL 转成 SMPL-X | 20G | 碰撞缓存约 17 小时 CPU；转换 v1→v3 迭代了几天 | 逐段 license 清单 | pkl 15–30 分钟；CoACD 1–2 小时 | **只留小部分**：manifest、`_collision_cache` 307M、convert_rows，约 0.43G |
| HiPHI GPU 转 SMPL-X | 27G | GPU 实际约 1 小时 | 没有 | 单卡约 1 小时 | **上云重算**（日志和汇总约 1M 留下） |
| HiPHI / Omni CPU 转 SMPL-X | 6.8G | 3.5 + 2 小时 CPU | 没有 | 几分钟 | **上云重算** |
| 对应关系缓存（工厂 4.0G + GPU 0.76G + UMR output 0.64G） | 5.4G | 每台 2–43 秒 | 没有 | 不到 15 分钟 | **上云重算**（两份要合成一份） |
| GPU 引擎 problems | 6.1G | 几分钟一段 | 没有 | 几分钟 | **丢弃**（但 `problems/betazero__static90` 389M 先保留：量产要读它，生成它的脚本还没写） |
| GPU 引擎 sources/robots/runs/logs | 14M | — | static90 输入来源不明 | 不能 | **保留** |
| lloyd_gpu_eval 的 cpu_*.npz、gpu.npz、统计 json | 12M | — | 对拍脚本写死要读 | 不能 | **保留**（mp4 丢） |
| GPU 对比视频 | 3.0G | 渲染 | 没有 | 约 30 分钟 | **丢弃** |
| GPU results（对拍表、吞吐、slip_v3、12 对导出） | 0.19G | 一周多开发 | 引擎可信度的唯一证据 | 证据得原样留 | **保留**（frameparallel 那份作废的丢掉） |
| CPU 工厂 runs | 1.3G | 约 2 小时 CPU | 没有 | 约 30 分钟 | **上云重算** |
| CPU 工厂 out（11 个样本） | 33M | 同上 | 格式参考样本 | 30 分钟 | **保留**（盒子判成球的 bug 要标注） |
| 逐台 QA 结论（`qa/`） | 75M | 3 天代理加看图 | 看图结论、找到的 bug | 结论不能重算 | **保留**（mp4 丢掉） |
| 截断实验 | 136M | 1.3 小时加看图 | 定稿规则 | 20–40 分钟 | **只留小部分**，约 5M |
| 多样性普查 | 20M | 1–2 小时 | 标签映射表 | 10 分钟 | **保留** |
| 数据源 license 快照（`sources/`） | 33M | 侦察 | 当时的条款快照 | 不能 | **保留**（26.6M 的 HF 文件树可丢） |
| beta_fit | 0.8M | 拷贝来的 | 来源 URL 未核实 | 不确定 | **保留**（私有，不进 git） |
| gpu_convert runs | 5.6G | 几小时 GPU | 对拍结论 | 1 小时以内 | **只留小部分**：parity40、omni40 的汇总 |
| 打包样本（`umr_dataset/data`） | 23M | 几分钟 | 没有 | 几分钟 | **上云重算** |
| production/v1 | 195M | 10 分钟 | 已作废 | — | **丢弃**（jobs.jsonl 和作废说明留下） |
| 日志 | 465M | — | 耗时实测的唯一证据 | 不能 | **只留小部分**，约 15M |
| viewer 缓存 | 3.3G | 自动生成 | 没有 | 看一次就重建 | **丢弃** |
| 原始数据：HiPHI 388G、GRAIL 35G、Omni 16G | 439G | HiPHI 墙钟 7.5 天、GRAIL 3 天 | 没有（要用已获批的 token） | 云上 1–2 小时 | **上云重下**（清单、sha、LICENSE 约 7M 留下） |
| uniml3d sample_cache | 1.3G | 下载 | 没有 | 可以重下 | **丢弃**（`remote/` 和 `proto/` 约 60M 留下） |

HiPHI 解包后的大小有两个说法：约 370G（减法估的）和 228.7G（下载器自己记的）。两者对不上，未核实。

### 2.2 小而贵的东西放哪
- **进 git**：所有配置 json、登记表、umr_configs、QA 和截断结论、docs、对拍表。合计 MB 级。
- **进私有 HF 数据集**（暂名 `Lloyd-lei/surface-retargeting-assets`），合计约 1–2G，包括：
  - GRAIL manifest 和碰撞缓存
  - out 样本
  - GPU results 证据
  - license 快照
  - GR3 网格
  - 截断和普查的结果表
- **不上传，只做私有备份**：SMPL-X 和 beta_fit（授权受限）。注意 SMPL-X 可能全机只有 `~/Downloads` 里这一份实体文件（未全盘核实）。
- **AlphaMotion 那边的小件**（不属于 UMR，但只有一份）：`~/tmp/sole_offsets3.json`、`gauge_check_0903.json`、`dialect/`。也建议一起备份。

### 2.3 10-11 那次删除到底弄坏了什么
- **UMR 没坏。** 除了 `inventory.py`（写死了 topo_exp、xwbc_lafan、am2exp、sole 的路径），UMR 代码对 topo_exp 和 cl_v3 都是 0 处引用。唯一的影响是 `inventory.py` 重跑时会悄悄少填几个交叉字段，迁移时顺手删掉这段引用就行。
- **AlphaMotion 产品线没坏。** 发布权重在 `weights_staging`、HF、`~/.cache` 各有一份。
- **AlphaMotion 研究线真正丢掉的：**
  - 没发布的 checkpoint：FZ5c、Y5b、Z3、C1 原始 run 等
  - AMASS 缓存的构建代码（`_amass_work`，没有任何副本）
  - heldout 600 段的名单
  - 另外 `~/alphamotion_ctrl/logs/rsl_rl` 也不在了：08-29 之后的 RL checkpoint 都没了，08-18 到 08-29 的只剩 `.git` 里那份
- **现在要不要重建：不用，等上云再做。**
  - 估计云上计算约半天，工程 1–2 天（AMASS 构建脚本要重写）。
  - 前提是 `xwbc_full`（129G，HF 上同一版本 b4e169f 还在）和 `sole_offsets3.json` 不能丢。
  - 注意 FOLDER_INDEX 里 `xwbc_full` 还写着"可删"，这是过时的：缓存删了以后，它是本机唯一的 AMASS 配对数据源。

---

## 3. 给 LLM 项目让路

### 3.1 现在就该停的（都需要 lloyd 点头）

| 对象 | 为什么停 | 命令 |
|---|---|---|
| `alphamotion-ardy-isaac` 服务 | 01:54 以来崩溃重启 6800 次以上，还在涨，每次瞬间吃 420% CPU。原因是找不到 `g1_29dof.xml` | `systemctl --user stop alphamotion-ardy-isaac.service && systemctl --user disable alphamotion-ardy-isaac.service` |
| `alphamotion-motion-lab` 服务（端口 2334） | 常驻 10.7 小时没人用，占 1.4G 内存 | `systemctl --user stop alphamotion-motion-lab.service`（不想开机自启再加 `disable`） |

### 3.2 留着不动
- rollout-gallery（68M）
- greenwich-mobile（另一个会话的，不碰）
- datagrid 的 Postgres（69M）
- Cursor、Claude 桌面版、LM Studio、另一个 Claude 会话
- 系统服务（tailscale、rustdesk 等）
- **throttle_guard 先留着。** 它几乎不耗资源，还在记 WiFi 掉线。例外：如果 LLM 项目要用 genmo-gem 环境，它会被一起绑到 0–15 核并降优先级，到时候用下面这条停：

  `P=$(cat ~/umr_factory_data/logs/throttle_guard.pid); ps -o args= -p $P | grep -q throttle_guard.sh && kill $P`

- 要不要停的，问 lloyd：openclaw 两个服务（约 1G 内存）。

### 3.3 停完以后能用多少

| 资源 | 可用量 |
|---|---|
| 显卡 | 96G 显存全空（本来就没有任务） |
| 内存 | 82G → 约 83–84G，另外去掉了崩溃循环的 CPU 尖峰 |
| 磁盘 | LLM 的权重和 HF_HOME 放根分区，比如 `/var/tmp/llm`，剩 231G。放进 150G 模型后还剩约 80G。**/home 只剩 76G，留 60G 余量后能用的很少，不要放** |
| 网络 | UMR 下载全停，带宽都给 LLM。仍然要守 RESOURCE_LIMITS：功率 <350W、单任务、绑 0–15 核 |

### 3.4 和下载器的冲突
- lloyd 之前要求 HiPHI 和 BONES 都下完，现在两个都停着。
- HiPHI 只差 part 0004，还要下约 4.2G；BONES 动作本体 45G 一点没下到。
- **建议：两个都不在本机重启。**
  - HiPHI 0004 在 LLM 不下大文件的空档补完，解包到根分区；或者干脆上云下。
  - BONES 改到云上下。本机按 3MB/s 要 4 小时以上，而且会把 /home 压到 60G 余量线以下。

---

## 4. 需要 lloyd 拍板的事（每条一句话就能答）

1. 仓库名用 `surface-retargeting`，建在 `Lloyd-lei` 名下、私有？**建议：是。**
2. 上游 UMR 和 hiphi2smplx 用子模块固定版本，不复制源码进来？**建议：是。**
3. 本机迁移只复制、不删旧目录，等新仓库对拍通过再说删不删？**建议：是。**
4. 停掉 ardy-isaac 崩溃循环和 motion-lab 两个服务？**建议：都停，ardy-isaac 同时取消开机自启。**
5. HiPHI 0004 和 BONES 都不在本机下，改到云上？**建议：是。**
6. LLM 的模型和缓存一律放根分区 `/var/tmp/llm`，不放 /home？**建议：是。**
7. 建一个私有 HF 数据集，存 1–2G 的小件（配置、manifest、碰撞缓存、证据）？**建议：是。**
8. 已知 bug 进仓库时先修，再谈量产？**建议：是。** 包括：
   - 盒子被判成球（box 片段的物体接触标签全部作废）
   - 打包器找不到 HiPHI 文件，遇到空片段会崩
   - 10-11 改的滑移规则从来没在量产里跑过
9. `xwbc_full`（129G）暂时不删，FOLDER_INDEX 里改成"唯一 AMASS 源"？**建议：是。** 根分区够用，不必腾它。
10. /home 上可以腾的地方（全部可以上云重来，删之前先查 FOLDER_INDEX）？**建议：等 LLM 真缺空间时再批，现在不动。**
    - HiPHI 原始数据：解包目录约 229–370G（大小未核实），另有 tar 15G、半截下载 3.5G
    - smplx_gpu 27G 和 smplx 里除 GRAIL 小件（manifest、碰撞缓存，约 0.43G，先搬走）以外的部分
    - viewer 缓存 3.3G
11. 机器人厂商网格和 GR3 网格 license 没核实，先只放私有存储、不公开？**建议：是。**
12. 5 套逐台 QA 脚本合成一个判官，旧版只归档？**建议：是，但排在仓库建好之后。**

---

## 附：核对后补充（挑错助手查出、已核实）
- 映射表补几项：`umr_factory_data/pilot/diag/coll_vs_vis.py` → `tools/diag/`；viewer 的 `check_decisions.py`、`scripts/service.py`、`verify_source_baseline.py`、`docs/`、`qa/acceptance.md`、`environment-freeze.txt` 一起进 `viewer/`；`umr_dataset/qa/*.json`、`_scale/bench_*.json` → `docs/benchmarks/`；`alpha_motion2/wbc_expert/scripts/` 里的 BONES 流式下载脚本 → `ingest/bones_seed/legacy/`（上云下 BONES 用得上）；`umr_ref/` 和上游一样，不收。
- `~/motion_audit_web_backups/20261007_191235.tar.gz`（33M）先和现目录比差异再定。
- genmo-gem 环境是可编辑安装的 GENMO 目录（20G），只导 lock 不够：要记 GENMO 的版本号，或者确认 gpu_convert 根本用不到它。
- 用词：对拍 = 新旧两版跑同一段数据逐数比较；子模块 = 只记上游版本号、不复制代码；冒烟 = 只跑几段快速检查能不能跑通；薄壳 = 旧入口只转调新仓库；ruler-v0 = 判官冻结版的标签。
