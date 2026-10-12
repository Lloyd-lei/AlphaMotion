# 缓存 / 产物 / 散落目录地图（map_caches_artifacts）

- 生成：2026-10-07，只读盘点（没有移动、删除、改动任何东西）。大小用 `du --max-depth=1`，"最后修改"取目录内两层以内最新文件的日期。
- 范围：`/var/tmp/alphamotion`、`~/tmp`、`~/alphamotion_artifacts`、`~/alphamotion_ctrl`、`~/reference_code`、`~/Desktop/AlphaMotion`、`~/Desktop/arena-datagrid`、`~/Desktop/alphamotion video`、`~/Desktop` 下其它目录，以及顺带发现的家目录相关目录。
- 主研究仓 `~/Desktop/"be water, robot"`（85G）和数据工厂 `~/umr_factory_data`（191G）这里只占一行，细节归别的地图。

## 0. 先看结论（10 条）

1. **一共约 790G 分散在 4 个根**：`/var/tmp/alphamotion` 352G、`~/umr_factory_data` 191G、`~/Desktop` 105G（主仓就占 85G）、`~/tmp` 56G、`~/reference_code` 31G、`~/alphamotion_ctrl` 29G、`~/alphamotion_artifacts` 13G，另有 `~/gpu_convert` 6.5G、`~/motion_audit_web` 3.2G。
2. **磁盘**：`/var/tmp` 在根分区，**已用 88%，只剩 76G**；`/home` 分区已用 69%，剩 347G。最大的缓存都放在快满的根分区上。
3. **正在跑的数据工厂，Python 环境放在 `/var/tmp`**：`/var/tmp/alphamotion/umr_factory/umr/env`（6.7G）是 GRAIL 下载/转换/碰撞、`run_factory.py`、`motion_audit_web/.mcp-venv` 共用的解释器。UMR 参考实现源码（`umr/UMR`）和 GPU 引擎的产出（`umr_gpu`）也在这里。这是现在最"活"的一块，却放在名字叫 tmp 的地方。
4. **现役训练集是跨根的软链接拼起来的**：`topo_exp/cache_lafan_v3`（131 项里 115 个链接）和 `topo_exp/cache_amass_lconv_tfix`（96 项里 61 个链接）指向三处：`topo_exp/cache_xwbc_lafan`、`topo_exp/cache_xwbc_amass_full`、**`~/tmp/cl_v3/*`（接触标签）**。所以 61G 的 `cache_xwbc_amass_full` 看起来像旧版，**其实不能删**；`~/tmp` 里的 31G 接触标签也不是临时文件。
5. **`~/tmp` 里压着这些"承重"文件**（代码写死了路径，移动就断）：`cl_v3/`（31G 接触标签 v3）、`sole_offsets3.json`（脚底偏移表，代码里 4 处引用，根高偏置的根源）、`gauge_check_0903.json`、`dialect/Q_final.pt`（人体骨架方言 Q）、`refiner_suite/`（精修器评测套件，13 处引用）、`cl_v2_qc/`、`contact_qc/`、`greenwich_workbench_0911/datagrid/`（datagrid 服务的 sqlite+postgres 状态，09-23 还恢复过）、`arena_handoff_backup/`（datagrid 数据库备份）。
6. **已经坏掉的链接**：`topo_exp/cache_lafan_v2` 有 17 个链接指向已删除的 `~/tmp/cl_v2`；`topo_exp/cache_amass_4000 -> cache_xwbc_amass` 目标不存在；主仓代码引用的 `~/alphamotion_artifacts/horizon_x2_hz4` 不存在。
7. **明显的重复/过时**：`gmr_production_v1`（和 `gmr_production` 同为 630M）、`x2_pool_v1_broken`（693M）、20 个 `boneseed_spatial_*` 目录（08-16 一天产出）、`any_body/`（13G，7 月底早期阶段，含 12G conda 环境）、`~/tmp/ufo_resume_drill` + `ufo_h2_smoke`（13.5G 演练/冒烟检查点）、`reference_code/ufo` 两套 venv（7.5G + 10G）。
8. **可以重新下载的大件**：`xwbc_full/`（129G，38 个 tar.gz，HF 仓 `TonyZhang1999/xwbc-retarget-data`，08-04 校验完成）。LAFAN 部分已解包到 `xwbc_lafan/extracted`（15G）；AMASS 部分已建成 `cache_xwbc_amass_full`。这批压缩包可以冷存或删除，省出 129G（要 lloyd 点头）。
9. **`~/alphamotion_ctrl/.git` 有 14G**：8.45G 散对象 + 3.92G pack + 1.18G 中断留下的 `tmp_pack` 垃圾。`git gc` 大概能收回 9G 以上（仓库主人操作，这次没动）。
10. **名字容易认错的**：Desktop 上有两个主仓同名目录，`"be water, robot"`（带引号，85G，真正的主仓）和 `be water, robot`（不带引号，557M，只有一份 08-29 的 `pose_translator_design` 散副本）；`~/alphamotion2_runtime`、`~/alpha-motion-workspace`、`~/.alphamotion_workspace` 三个都是指向主仓的软链接；`/var/tmp/alphamotion/pose_translator_2` 是指向 `~/alphamotion_artifacts/pose_translator_2` 的软链接；`~/Desktop/pose_translator_2` 是代码（6.5M），`~/alphamotion_artifacts/pose_translator_2` 是它的产出（11G）。

**状态口径**：
- **活跃**：最近一周还在写，或者是正在用的服务、产品。
- **承重**：不再写，但现役代码、训练集、服务在读，挪走就断。
- **休眠**：暂停的工作线，以后可能恢复。
- **归档**：有历史价值，只读冷存。
- **过时**：可以删，或者可以重新下载；删前要 lloyd 确认。
- **无关**：不属于 AlphaMotion。

---

## 1. 统一布局提案（下表"建议新家"列用的代号）

根目录名留给总设计定（下面写作 `<ROOT>`）。物理位置建议放 `/home` 分区：它剩 347G，根分区只剩 76G。

```
<ROOT>/
├─ code/                       只放 git 仓库（代码），产出一律不进仓库
│   research(=be water robot) · product(=AlphaMotion) · datagrid · ctrl · pose_translator_2
│   umr_factory_code · umr_gpu · umr_dataset · motion_audit_web · gpu_convert
├─ third_party/<name>/         外部参考仓库/基线（只读）：ufo · bfm_zero · sonic · beyondmimic · GMR · holosoma · dex-retargeting · anytop · harvest/*
├─ envs/<name>/                conda/venv：umr(现 /var/tmp/.../umr/env) · ctrl · ufo · genmo-gem · ardy …
├─ data/
│   ├─ raw/<source>/           原始下载（只读）+ LICENSE + manifest.json
│   ├─ assets/robots/<robot>/  MJCF/URDF/网格 + surface_v2 表面点 + sole_offsets + umr 配置
│   ├─ assets/human/           SMPL/SMPL-X、beta_fit
│   ├─ retarget/<engine>@<ver>/<source>/<robot>/   xwbc 解包、GMR production、UMR cpu/gpu
│   ├─ labels/<kind>@<ver>/    接触/穿地/滑移标签（cl_v3、pcl_contact）
│   └─ datasets/<name>@<ver>/  训练缓存/分片 + manifest（cache_lafan_v3、cache_amass_lconv_tfix、x2_pool_v3、bridge_codes、text_align）
├─ runs/<family>/<run_id>/     训练产物：greenwich · equator · pt2 · wbc_mjlab · wbc_ufo · am2exp
├─ eval/<suite>/<subject>/     判官/审计产出：judge · wbc_ready · x2phys · stress · refiner · ctrl_verify · audit_web
├─ media/<YYYYMMDD>_<topic>/   展示视频、对比图
├─ releases/<product>@<ver>/   发布物（weights_staging、HF 清单）
├─ services/<svc>/state/       服务运行态（datagrid 的 sqlite/postgres、studio 上传、audit_web 缓存）
├─ docs/                       报告、判决、计划（来自 alphamotion video/resource/report）
├─ cache/                      随时可以清：hf、torchinductor、triton、warp、wandb 临时目录
├─ scratch/<YYYYMMDD>_<task>/  临时目录，7–14 天过期（取代 ~/tmp 的平铺乱放）
└─ archive/<YYYY-MM>/<orig>/   只读冷存（压缩 + 一行说明）
```

---

## 2. `/var/tmp/alphamotion`（352G，根分区）

| 目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|---|
| `topo_exp/` | 研究主线的缓存和训练根目录（主仓 `pose_translator_design/exp_topo/*` 的工作区） | 训练缓存、204 个 run、码本、文本对齐；细项见 §3 | 160G | 09-16 | 活跃+承重 | 被主仓 141+ 处引用；依赖 `xwbc_*`、`~/tmp/cl_v3` | 拆到 `data/datasets/`、`runs/`、`data/labels/` |
| `xwbc_full/` | 从 HF 下载的 xwbc 重定向数据压缩包（16 台机器人 × AMASS/LAFAN + human_motion） | 38 个 tar.gz，`DOWNLOAD_DONE` 记录 137.89GB 全部通过 gzip -t | 129G | 08-04 | 过时（可重新下载） | 被 `cache_xwbc_amass_full` 的构建脚本读过（`_amass_work`） | `data/raw/xwbc/`，建议冷存或删除 |
| `xwbc_lafan/extracted/` | xwbc LAFAN 部分解包后的文件，每台机器人一个目录（19 个，含 human_motion、h2、r1） | 每个片段一个 npz | 15G | 08-12 | 承重 | `cache_xwbc_lafan` / `cache_lafan_v3` 的源；主仓 6 处引用 | `data/retarget/xwbc@v1/lafan/<robot>/` |
| `umr_factory/` | UMR 数据工厂运行区：CPU 参考 UMR 的克隆和 conda 环境、GPU 引擎产出、标签器原型、HiPHI→SMPL-X、UniML3D 重载 | 见 §4 | 18G | **10-07（正在写）** | 活跃 | 被 `~/umr_factory_code`、`~/umr_gpu`、`~/umr_factory_data/scripts*`、`motion_audit_web` 依赖（共 29+14 处引用 + 解释器） | `envs/umr`、`third_party/UMR`、`runs/umr_gpu`、`eval/umr_gpu`、`data/assets/robots` |
| `any_body/` | 7 月底 "any body" 早期阶段：conda 环境、SigLIP/Qwen3 关节名向量、t0/t1 训练张量 | `env/`（12G）、`hf/`（780M）、`cache/*.pt`（357M） | 13G | 08-06 | 归档（环境部分过时） | 主仓 3 处引用 `any_body/env`；SigLIP 名向量已另存到 `topo_exp/siglip_names.pt` | `archive/2026-08/any_body`；环境直接删 |
| `hf_cache/` | HuggingFace 模型缓存：BAAI bge-m3、Qwen3-Embedding-0.6B（文本层、关节名向量） | 模型权重 | 5.4G | 08-05 | 承重（可重新下载） | 主仓 4 处 + datagrid 1 处 `HF_HOME` | `cache/hf/` |
| `am2exp/` | AM2 实验 Stage-0（10-04）：致密表面资产 surface_v2、判官 J1/E1 尺子、E1/E2 评测、集成、审阅视频 | `stage0_index.md`、`assets/surface_v2/`（18 个身体的表面点和渲染）、`judge/stage0_report.md`、`eval/e1`（2.9G） | 3.0G | 10-04 | 活跃 | 代码在主仓 `alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1/`；surface_v2 被工厂 2 处引用 | `runs/am2exp/`、`eval/am2exp/`、`data/assets/robots/*/surface_v2` |
| `ctrl_verify/` | 给 mjlab/BeyondMimic 准备的跟踪参考轨迹（G1/GR3/X2/ADAM），X2 语料池 | `tracks_*`（g1_lafan、g1_pilot0902、gr3_ground0903…）、`x2_corpus/`、`report_g1*.json` | 2.9G | 09-05 | 承重/休眠 | 输出喂给 `~/alphamotion_ctrl/motions` 和 `x2_pool*`；主仓 7 处 + ctrl 3 处引用 | `data/datasets/wbc_tracks@v1/` |
| `config_harvest/` | 抓下来的外部重定向/WBC 仓库配置：GMR、PHUMA、ProtoMotions、soma-retargeter、TWIST、TWIST2、whole_body_tracking、unitree_repos.json | 只读参考 | 2.5G | 08-03 | 承重 | **`config_harvest/GMR` 被主仓 30 处、工厂 19 处引用**（GMR 的 IK 配置和机器人资产） | `third_party/GMR`、`third_party/harvest/*` |
| `robot_models/` | 机器人模型源：menagerie、unitree_ros(+official)、agibot a2/a3/x2、roboterax、Fourier GRx | URDF/MJCF/网格 | 1.4G | 10-03（git） | 承重 | 主仓 21+17 处、工厂 11+7 处引用（agibot_x2_urdf、unitree_ros） | `data/assets/robots/_sources/` |
| `omniretarget/` | OmniRetarget 数据集 + holosoma 代码（对比基线） | 数据集 zip 和解包后的目录各一份（重复） | 1.3G | 08-03 | 归档 | 主仓 3 处引用（holosoma） | `third_party/holosoma`、`data/raw/omniretarget/`（只留一份） |
| `gmr_production/` | GMR 原生重定向批量产出（9 台机器人，`POOL_DONE2`） | 每台机器人的动作 + `summary.json` | 629M | 08-03 | 承重 | "Native policy"（B_native）对照臂的数据源；主仓 4 处引用 | `data/retarget/gmr@v2/` |
| `gmr_production_v1/` | 上一份的旧版 | 同上 | 630M | 08-03 | 过时 | 无引用 | 删除，或放 `archive/` |
| `gmr_spike/` `gmr_upstream/` | GMR 调参（肘部/手臂偏置、ik_configs）和 GMR 上游源码 | `ik_configs/`、`*_report.json` | 6.4M / 1.7M | 08-03 | 承重（ik_configs） | 主仓 4+1 处引用 | `third_party/GMR/{upstream,tuned_ik}` |
| `gem_runtime/` `gem_runtime_pkgs/` `gem_text/` `genmo_bridge/` `genmo_cache/` `tmr_bridge/` | 视频/文本→动作桥：GEM/GENMO 运行库（onnxruntime 等）、文本提示样例、TMR 检索 | `*.pt` 提示词、`real.npz` | 566M + 小件 | 08-14 | 休眠 | studio 的依赖；主仓 4+2+4 处引用 | `third_party/gem`、`services/studio/` |
| `studio/` | AlphaMotion Studio 产品前端（eval+rollout 链，worker 多个） | `alphamotion_studio_app.py`、各 worker、`uploads/`、`results/` | 14M | 09-08（log） | 休眠（现在没在跑） | 依赖 `topo_exp/runs`、gem_*、genmo_*、tmr_bridge | `code/studio`（代码）+ `services/studio/state` |
| `beyondmimic/` | 最早的 BeyondMimic 环境（alphamotion_ctrl + gmt） | — | 101M | 08-18 | 过时（已被 `~/alphamotion_ctrl` 取代） | 主仓 1 处引用 | `archive/` |
| `dex-retargeting/` `anytop/` | 外部仓库：手部重定向、AnyTop 基线 | 源码 | 56M / 14M | 08-03 / 07-31 | 归档 | dex 有 1 处引用 | `third_party/` |
| `hf_home/` | 下载 xwbc 时的 HF 元数据/xet 缓存 | — | 63M | 08-03 | 过时 | — | `cache/` 或删除 |
| `xwbc_clips/` `xwbc_dl/` | xwbc 早期样例片段、下载脚本 | 19 台机器人的样例、`run.sh` | 223M / 136K | 08-03/04 | 过时 | 无现役引用 | 删除，或放 `archive/` |
| `spatial_recipe_ablation_20260816/` | 空间配方重训消融报告（中文 PDF/HTML/CSV） | 报告 | 71M | 08-16 | 归档 | 工厂 1 处引用 | `docs/reports/2026-08-16_spatial_recipe_ablation/` |
| `boneseed_spatial_*`（20 个目录） | 08-16 BONES-SEED 空间模型审计、视觉门（smoke/full/paired/anchors/registered 各版本） | json + 视频 + 网格图 | 共约 100M | 08-16 | 归档 | 无现役引用 | `eval/archive/2026-08-16_boneseed_spatial/`（合成一个目录） |
| `mesh_viz/` `ab_test/` | G1 网格对照图；holosoma vs soma 的 A/B 指标 | png/gif、`metrics_*.json` | 424K / 1M | 08-03 | 归档 | 主仓 1 处引用 ab_test | `archive/` |
| `pose_translator_2`（软链接） | → `~/alphamotion_artifacts/pose_translator_2` | — | — | 08-17 | 链接 | — | 删掉链接，以 catalog 为准 |
| 根目录散文件（约 70 个） | `*_queue.sh`/`overnight_*.sh` 训练队列、下载日志、`studio_app.py`、BoneSeed 视觉门 PDF 和 mp4、`xwbc_joint_names.json` | 脚本/日志/视频 | 约 31M | 07-31~08-16 | 归档（`xwbc_joint_names.json` 承重，1 处引用） | — | 脚本 → `archive/queues/`；视频 → `media/2026-08-16_boneseed/`；joint_names → `data/raw/xwbc/` |

---

## 3. `topo_exp/` 细项（160G）

| 子目录 | 用途 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|
| `cache_xwbc_amass_full/` | AMASS × 16 台机器人的配对缓存 v1（有时间轴 bug，KIT 1.2 倍、TotalCapture 2 倍） | 61G | 08-31 | **承重** | `cache_amass_lconv_tfix`（41 个链接）、`cache_amass_v3`（77）、`cache_amass_lconv`（75）都链接到它；主仓 94 处 + 工厂 7 处引用 | `data/datasets/xwbc_amass@v1/` |
| `cache_amass_lconv_tfix/` | **现役 AMASS 训练缓存**（时间轴已修，FZ5c / Greenwich-v2 / Chinchilla 阶梯 S/M/L20k 用的就是它） | 52G（35 个实体文件 + 61 个链接） | 09-03 | 活跃/承重 | 链接到 `cache_xwbc_amass_full`、`cache_amass_lconv`、`~/tmp/cl_v3/amass_lconv_tfix`；被 ladder.sh、gauge_check、refiner_suite、showcase 引用 | `data/datasets/amass_tfix@v3/` |
| `cache_lafan_v3/` | **现役 LAFAN 训练缓存**（链接拼起来的） | 480K（115 个链接） | 09-16 | 活跃/承重 | → `cache_xwbc_lafan`（98）+ `~/tmp/cl_v3/lafan`（17） | `data/datasets/lafan@v3/` |
| `cache_xwbc_lafan/` | LAFAN 配对缓存 v1（实体文件） | 7.0G | 08-31 | 承重 | 被 `cache_lafan_v3`/`v2` 链接；主仓 44 处引用 | `data/datasets/xwbc_lafan@v1/` |
| `cache_amass_lconv/` | AMASS 中间版（时间轴修复前） | 2.4G | 09-01 | 承重（tfix 有 3 个链接指向它） | — | `data/datasets/amass_lconv@v2/` |
| `cache_amass_heldout/` | AMASS 留出集（600 段语料外审计用） | 2.2G | 08-07 | 承重 | 主仓 7 处 + 工厂 5 处引用 | `data/datasets/amass_heldout@v1/` |
| `cache_amass_v3/` | AMASS v3（链接拼起来的，旧标签） | 324K | 08-31 | 归档 | → `cl_v3/amass` + `amass_full` | `archive/` |
| `cache_lafan_v2/` | LAFAN v2 | 408K | 08-31 | **坏了**（17 个链接指向已删除的 `~/tmp/cl_v2`） | — | 删除 |
| `cache_amass_4000`（链接） | → `cache_xwbc_amass`（不存在） | — | 08-04 | **坏了** | — | 删除 |
| `cache_paired*`（`_fixed`/`_test`）、`cache_big/`、`cache/` | 8 月初 Stage-1 的配对缓存 | 119M+119M+3.6M+157M+2.1M | 08-02~08-04 | 归档（`cache_paired_fixed` 还有 4+3 处引用） | — | `archive/2026-08/` |
| `cache_bones_body/` | BONES-SEED 身体缓存 | 322M | 08-05 | 归档 | 2 处引用 | `data/datasets/bones_body@v1/` |
| `cache_xwbc_lafan_fixed/` | LAFAN 手腕修复版（61 个实体文件） | 268K | 08-07 | 归档 | 4 处引用 | `archive/` |
| `bridge_codes/PROD_c1_91k_dim20_tail_0810/` | C1 生产码本编出来的全库码（Equator/Atlas 还在用 C1 码） | 17G | 08-10 | **承重** | Equator A/B、Atlas、library 依赖 C1 码；主仓 3 处、工厂 2 处引用 | `data/datasets/codes_c1@0810/` |
| `text_align/` | 文本层：caption 向量、文本码、训练/验证配对 | 2.2G | 08-14 | 承重（文本层 B5） | 主仓 10 处引用 | `data/datasets/text_align@v1/` |
| `v5_keycache/` | 人体 SMPL 关键帧缓存（amass_full / lafan / h1） | 1.2G | 08-10 | 归档 | — | `archive/` |
| `runs/` | **204 个训练 run**：C1、BRIDGE_A3/B3、FZ5c、Y5b、Z3、E24、L_S/M/L20k、L_M20k_vn_0916… | 16G | 09-16 | 活跃（发布血统）+ 大量归档 | 主仓 141 处、工厂 12 处引用；AlphaMotion 发布权重从这里导出 | `runs/greenwich/<run_id>/`、`runs/equator/<run_id>/`；非血统 run 移到 `archive/` |
| `encoder_bakeoff/` `night2_lib/` `_amass_work/` `contact_stats/` `v2_prepatch_0831/` | 编码器比武、night2 依赖库、AMASS 构建脚本、接触统计、补丁前快照 | 12M/460K/73M/76K/80K | 08-05~08-31 | 归档（night2_lib 有 11 处引用 → 承重） | — | `code/` 或 `archive/` |
| `qwen3_names.pt` `siglip_names.pt`（+ `.bak`/`.clobbered`） | 关节名语义向量 | 各约 1–1.6M | 08-30 / 09-05 | 承重（主版）；`.bak`/`.clobbered` 过时 | 主仓、工厂都引用 | `data/assets/embeddings/` |
| 根目录 `*.py` `*.log` | 桥状态图谱、生产域评测、viser X2 查看器 | 小 | 08-02~08-14 | 归档 | — | `code/research` |

---

## 4. `umr_factory/` 细项（18G，活跃）

| 子目录 | 用途 | 大小 | 最后修改 | 状态 | 建议新家 |
|---|---|---|---|---|---|
| `umr/env` | **工厂共用的 conda 环境**（GRAIL 循环、`run_factory.py`、audit_web 的 MCP venv 都用它） | 6.7G | 10-07 | 活跃（正在跑） | `envs/umr/` |
| `umr/UMR` | UMR CPU 参考实现的克隆 | 1.2G | 10-07 | 活跃 | `third_party/UMR/` |
| `umr/{runs,videos,eval,tools,configs,logs}` | CPU 参考跑出来的结果和视频 | 572M+19M+小 | 10-04 | 活跃 | `runs/umr_cpu/`、`media/` |
| `umr_gpu/{problems,videos,corr,results,runs,logs,robots,sources}` | GPU 引擎的问题集、对比视频、对应关系、bench 结果（今天 18:20 还在写 `bench_v2_b256_4r_*`） | 5.7G+1.6G+379M+82M+… | 10-07 | 活跃 | `runs/umr_gpu/`、`eval/umr_gpu/` |
| `robots/` | 机器人注册表：`robots.json`、`robots_table.md`、umr_configs、T 姿态渲染 | 3.4M | 10-04 | 承重（5 处引用） | `data/assets/robots/registry/` |
| `labels/` | 点云接触/穿地/滑移标签器原型 `pcl_contact`（规格 v0.1）+ 输出 | 31M | 10-04 | 活跃（原型） | `code/` + `data/labels/pcl_contact@v0.1/` |
| `hiphi2smplx/` | HiPHI→SMPL-X 转换代码 | 63M | 10-04 | 承重 | `code/` 或 `third_party/` |
| `uniml3d_reload/` | UniML3D 重载原型 + 样例缓存 | 1.3G | 10-04 | 活跃 | `data/raw/uniml3d/sample_cache`、`code/` |
| `sources/` | 数据源许可证和元数据（AMASS license、BONES meta parquet、GRAIL、HiPHI、OmniContact、SOMA-X、UniML3D、Mixamo、OMOMO 的 README） | 33M | 10-04 | 承重（许可证追踪的起点） | `data/raw/_licenses/` |
| `beta_fit/` `h2s_shim/` `lloyd_gpu_eval/` | SMPL beta 拟合先验；h5py 垫片；lloyd 要的 CPU/GPU 对比视频 | 836K/16K/45M | 10-04 | 承重 / 承重 / 归档 | `data/assets/human/`、`code/`、`eval/umr_gpu/lloyd_gpu_eval/` |

---

## 5. `~/tmp`（56G，1440 个平铺散文件 + 约 120 个目录 + 152 个 wandb 临时目录）

散文件构成：678 json、230 log、194 mp4（703M）、130 txt、52 png、51 sh、42 py、18 npz、13 marker，合计约 932M。下表按主题分组。

| 组 / 目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|---|
| **`cl_v3/`**（amass 15G、amass_lconv_tfix 14G、lafan 1.6G、amass_lconv 701M、`*_stats.json`） | 接触标签 v3，每台机器人一个 `*_contact.npz` | 标签 | **31G** | 09-03 | **承重**（现役训练集的链接目标） | `cache_lafan_v3`、`cache_amass_lconv_tfix`、`cache_amass_v3`、`cache_amass_lconv`；主仓 5 处引用 | `data/labels/contact@v3/`；`amass/` 子目录（15G）只被旧缓存用 → 归档候选 |
| **`sole_offsets3.json`**（及 `sole_offsets.json`、`sole_offsets2.json`） | 每台机器人的脚底偏移表（根高偏置的根源，8 台偏大 4.7–10cm，缺 H2） | json | 45K | 09-01 | **承重**（4 处引用） | 根高修正、接触标签 | `data/assets/robots/<robot>/sole_offset.json` + 版本号 |
| **`gauge_check_0903.json`**、`dialect/`（`Q_final.pt`、`Qcfg.npy`、拟合脚本） | 数据 gauge 检查；人体骨架"方言"Q 矩阵 | json / pt | 8K / 40K | 09-03 / 09-01 | **承重** | 主仓 2 处、8 处引用 | `data/assets/human/dialect_Q@v1/`、`eval/gauge/` |
| **`refiner_suite/`** | SE(3) 精修器评测套件（eval_suite、profiles、manip_set、out） | 评测结果 | 728M | 09-09 | **承重**（13 处引用） | 依赖 `cache_amass_lconv_tfix` | `code/research`（代码）+ `eval/refiner/`（out） |
| `cl_v2_qc/` `contact_qc/` `ab_sole0/1/` | 接触标签 v2 质检、AMASS 穿地/源审计、脚底 A/B | json、视频、`*_contact.npz` | 354M / 107M / 274M×2 | 08-31 | 归档（cl_v2_qc 11 处、contact_qc 7 处引用 → 承重） | — | `eval/contact_qc/`、`archive/` |
| `greenwich_workbench_0911/` | Greenwich 工作台：**datagrid 服务的运行态**（`datagrid.sqlite3`、`postgres/`、model-service），Greenwich 产品仓副本，`greenwich-motion-toolkit`，H2 资产（链接到 `reference_code/sonic`），venv（→ conda `rot`） | 服务状态 + 代码副本 | 812M | **09-23**（数据库恢复） | **承重/休眠** | 依赖 `~/Desktop/arena-datagrid`、`reference_code/sonic` | `services/datagrid/state/`；代码副本对齐 `code/` 后删除 |
| `arena_handoff_backup/` | datagrid 数据库备份（`.sql` + sqlite） | 备份 | 2.6M | 09-11 | 承重（唯一备份） | — | `services/datagrid/backups/` |
| `ufo_resume_drill/` `ufo_h2_smoke/` | UFO/BFM-Zero 续训演练、H2 冒烟训练（检查点） | checkpoint | 6.9G / 6.6G | 08-30 / 09-16 | 过时 | — | 删除（`archive/` 只留 config） |
| `ufo_p1..p5` `ufo_probe*` `ufo_smoke_g1*` `ufo_cache/` `cache/` `torchinductor_*` | UFO 试跑配置、编译缓存（cuda/triton/warp/inductor） | config、编译缓存 | 约 1M + 425M + 94M | 08-30~09-17 | 过时 / 缓存 | — | 配置 → `runs/wbc_ufo/_pilots/`；缓存 → `cache/` |
| `bfm_data_clone/` | BFM-Zero 仓库副本（带数据） | 源码 | 195M | 08-30 | 过时（和 `reference_code/bfm_zero` 重复） | — | 删除 |
| `h2_bfm/` `h2bm/` `h2gr*/` `h2hd/` `h2mp/` `g1gr/` `h2_ground_rs/` `seed_g1_rs/` `h2_bfm_watch.*` | H2/G1 的 BeyondMimic 和 BFM-Zero 训练中途视频、地面动作 4 段、种子参考 | mp4、json、npz | 13M+9M+约 15M | 09-15~09-18 | 归档（H2 线 0918 后暂停） | 训练代码在 `~/alphamotion_ctrl/amctrl/unitree_h2` | `media/2026-09-1x_h2_wbc/`、`runs/wbc_mjlab/h2/` |
| `teleop2/` `teleop_*.py` | 遥操两段式探针（Greenwich w15） | mp4/npz | 74M | 09-17 | 归档 | 报告 23 | `eval/teleop/` |
| `ee_data/` `ee_fill/` `ego_fill/` `source_robot_probe_0905/` `six_robot_manip_eval_0905/` `six_robot_foot_ablation_0905/` `alphamotion_probe_0905/` `embodiment_normalization_0905/` | 手部 SE(3) 补全全身：EgoDex/UMI 数据、钉手钉脚补全、六台机器人操作/脚部消融 | 视频、帧、json、补丁 | 627M / 1.6G / 3.4G / 919M / 251M / 小 | 09-03~09-05 | 归档（精修器已叫停，当对照臂） | 报告 15 | `data/raw/egodex/`（ee_data）、`eval/ee_fill/`、`media/` |
| `wbcready_*`（162 个文件）、`wbcready_vid/` | 600 段语料外 WBC 可训性审计，每个检查点 × 3 台机器人 | txt/json/log、视频 | 约 5M + 30M | 08-05~09-04 | 归档（已复制到报告 02） | — | `eval/wbc_ready/<ckpt>/` |
| `x2phys/` | X2 零样本、无需真值的物理判官 | json | 61M | 09-04 | 归档（已复制到报告 03） | — | `eval/x2phys/` |
| `stress*` `gwstress*` `gwf0stress*` `gwhmstress*` `crossrig*`（12 个） `slipvid_*`（9 个） `cmp34*` `p3_*` `qc_*` `r13viz` `r14viz` `prog_*` `mid1000/1500` `y1*/y3*/y5*`（odo/contact/fixed/lafan/lconv） `landscape/` `death_slices/` `abl_render_slices/` `_smoke_render/` `sweep/` `gr3_3p/` `gr3_mid/` | 8 月底到 9 月初的各种压力测试、跨骨架、滑步视频、对照、里程计、scaling 地形 | 视频、网格图、json | 合计约 1.2G | 08-29~09-04 | 归档 | 部分被报告引用（stress_*、gwstress、sweep） | `eval/archive/2026-08-29..09-04/<topic>/` |
| `amass_g1_*`（raw/rs/gw_rs/gwf0_rs/gwhm_rs） | AMASS → G1 解码对比的小样本 | npz | 约 6M | 08-31 | 归档 | — | `archive/` |
| 根目录散 mp4（194 个，703M） | G1/GR3/ADAM 对决、BRIDGE 起身、GW 20/50/100M 跟踪、精修器 A、试点解码…… | 视频 | 703M | 08-28~09-02 | 归档（一部分已复制进 `alphamotion video`） | 主仓引用 `pilot_decode_*`、`refinerA_*`、`tri_embod_*`、`bridge_getup_*` | `media/<date>_<topic>/` |
| 根目录散 json/log/sh/py/marker（约 1240 个） | 各次 eval 分数、训练日志、链式脚本、`*_DONE.marker`、一次性诊断脚本 | — | 约 230M | 08-20~09-17 | 过时/归档（`ablation_table.csv`、`scaling_*.csv`、`stress_run.py`、`qpos_to_rs.py`、`xwbc_to_robotstate.py` 有引用 → 承重） | — | 有引用的 → `code/research/tools` 或 `eval/`；其余打包进 `archive/tmp_2026-08_09.tar.zst` |
| `tmp*wandb-media` / `tmp*wandb-artifacts`（152 个） | wandb 残留的临时目录（基本是空的） | — | 约 0 | 08-18~09-18 | 过时 | — | 删除 |

---

## 6. `~/alphamotion_artifacts`（13G）

| 目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|---|
| `pose_translator_2/` | Pose Translator 2（独立的空间编解码，代码在 `~/Desktop/pose_translator_2`）的实验产出：约 30 个 `PT2_*` 试点 run、合约冒烟测试、人体未见 rollout、梯度归因 | run、json、mp4 | 11G | 08-17 | 归档（PT2 线 08-17 后停了） | `/var/tmp/alphamotion/pose_translator_2` 链接到这里；主仓 1 处引用 | `runs/pt2/` → `archive/2026-08/pt2/` |
| `x2_pool_v3/` | X2 WBC 训练池 v3（分片 npz + manifest，接触一致的 z） | shard_*.npz | 693M | 08-20 | 承重/休眠 | `~/alphamotion_ctrl/amctrl`（v3 预训练 + 考试链）；主仓 1 处引用 | `data/datasets/x2_pool@v3/` |
| `x2_pool/` | X2 池 v2 | 同上 | 693M | 08-20 | 归档 | — | `archive/` |
| `x2_pool_v1_broken/` | X2 池 v1（世界行差一位的 bug） | 同上 | 693M | 08-19 | 过时 | — | 删除 |
| `data_contract_audit_20260817/` | 动作数据合约审计（姿态/根/时间合约失败图） | md/json/png | 1.7M | 08-17 | 归档 | — | `eval/archive/2026-08-17_data_contract/` |
| （缺失）`horizon_x2_hz4` | 主仓 2 处引用，但目录不存在 | — | — | — | **断了** | 地平线 hz4 实验 | 在 catalog 里标成缺失 |

## 7. `~/alphamotion_ctrl`（29G，git 仓库）

| 目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|---|
| `amctrl/` | WBC 控制器任务包：G1 地面、GR3、X2、H2（mjlab 任务 `Mjlab-Tracking-Flat-Unitree-H2`）、horizon 课程、token/RL | 代码 | 688K | 09-15 | 休眠（电费预算从 0903 起叫停 RL） | 依赖 mjlab 1.6.0、`x2_pool_v3`、`motions/` | `code/ctrl/amctrl` |
| `motions/` | mjlab 跟踪参考（206 个 npz：A_greenwich / B_native / mask 增强…） | npz，50fps | 8.6G | 09-18 | 承重/休眠 | 来自 `ctrl_verify/tracks_*` + `csv_to_npz_local.py` | `data/datasets/wbc_motions@v1/` |
| `logs/` | rsl_rl 训练 run 和考试 json/mp4（abl_*、dance、pool_hold、g1gr/h2bm/h2gr/h2hd/h2mp） | 检查点、eval json、mp4 | 891M | 10-04 | 休眠 | — | `runs/wbc_mjlab/` |
| `.venv/` | uv venv，mjlab==1.6.0（MuJoCo-Warp）、torch cu12 | 环境 | 5.8G | 08-18 | 承重 | — | `envs/ctrl/` |
| `.git/` | 8.45G 散对象 + 3.92G pack + 1.18G 中断的 `tmp_pack` 垃圾 | — | 14G | 09-18 | 臃肿 | `.gitignore` 已经排除 motions/logs/npz，但历史里可能已经提交过 | 仓库主人做 `git gc --prune` 能收回大约 9G 以上（这次没动） |
| `scripts/` `README.md` `MUJOCO_LOG.TXT` | 转换脚本、说明 | — | 288K | 09-16 | 承重 | 主仓 1 处引用 scripts | `code/ctrl/` |

## 8. `~/reference_code`（31G，外部参考）

| 目录 | 用途 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|
| `ufo/` | UFO（BFM-Zero 的 Blackwell 配方：torch2.7.1cu128 + warp1.14），官方检查点和官方跑分 | 26G（`.venv312` 10G + `.venv` 7.5G + official_ckpt 3.4G + official_run 3.2G + humanoidverse 592M + cache 491M） | 10-04（.git） | 承重/休眠（H2 BFM-Zero 0916 用过） | 主仓 2 处引用；`~/tmp/ufo_*` 的来源 | `third_party/ufo`；两套 venv 留一套 → `envs/ufo`；检查点 → `third_party/ufo/ckpt` |
| `sonic/` | NVIDIA GEAR SONIC（解耦 WBC），**H2 的 MJCF/网格从这里来** | 4.7G | 08-29 | 承重 | `greenwich_workbench_0911/h2_assets` 链接到这里；主仓 1 处引用 | `third_party/sonic`；H2 资产复制一份到 `data/assets/robots/unitree_h2/` |
| `bfm_zero/` | BFM-Zero 官方仓库 | 506M | 08-30 | 归档 | 主仓 2 处引用 | `third_party/bfm_zero` |
| `beyondmimic/` | BeyondMimic 官方仓库 | 524K | 08-29 | 归档 | 主仓 1 处引用 | `third_party/beyondmimic` |

## 9. `~/Desktop` 上的项目目录

| 目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议新家 |
|---|---|---|---|---|---|---|---|
| `"be water, robot"`（带引号） | **主研究仓**（分支 research-0909）：alpha_motion2 53G、GENMO 20G、experiment 7.7G、notebook 1.3G、pose_translator_design 1.2G、any_body 1.2G、papers-docs 597M、未跟踪的 `Greenwich_G1_18_Paired_Benchmark_Review_20260920` 247M | 代码 + 大量产出混在一起 | 85G | 10-05 | 活跃 | 引用上面几乎所有根目录 | `code/research`（产出拆出去；归别的地图） |
| `be water, robot`（不带引号） | 只有一份 `pose_translator_design/`（08-29），像是路径引号写错留下的副本 | — | 557M | 08-29 | 过时（要先和主仓里那份对比再删） | 无 | 对比后删除，或放 `archive/` |
| `AlphaMotion/` | **产品仓** release/v0.2（已推 Greenwich.git；HF 已公开）：src、tests、docs、`weights_staging/`（greenwich 70M、bumblebee 14M、megatron 69M、prime 221M、equator_a 153M、equator_b 50M、atlas 3.5M、library 631M、embodiments） | 发布物 | 1.5G（`.git` 285M） | 09-13（.git）/ 09-09 | 活跃（产品） | 权重从 `topo_exp/runs` 导出；被 `greenwich_workbench_0911/Greenwich` 复制过一份 | `code/product`；`weights_staging` → `releases/alphamotion@0.2/` |
| `arena-datagrid/` | **Arena Data Grid 单仓**（apps/datagrid、packages/{alphamotion,arena,arena_research,greenwich_motion_sdk}、configs/{gauge,robots,standards,constraints,selections}）：队列、GPU 批量推理、worker、精修器 v2、locomotion.v1 标准 | 代码（运行态在 `~/tmp/greenwich_workbench_0911/datagrid`） | 112M | 09-11（.git 09-13） | 休眠（**是"一个服务 + API"最接近的现成底座**） | 运行态和备份在 `~/tmp`；用 `hf_cache` | `code/datagrid`；运行态 → `services/datagrid/state` |
| `alphamotion video/` | 展示视频 01–08（FZ5c 检查点，09-02/03）+ **`resource/report/` 技术报告素材库**（00–35 号：计划、审计、判官、阶梯、白皮书、方法、判决、gauge、WBC 配方、文献、产品文档、SE(3) 精修器、H2、ARDY、精修器对决、实时精修器、地面 gauge、one-pager、IROS 报告）+ `image/` | mp4、md、json、图 | 1.2G（report 572M、image 197M、视频约 420M） | 09-27（report/35） | 活跃（文档中心） | INDEX.md 写明"全是复制件，原件留在原处" | 视频 → `media/2026-09-02_showcase_fz5c/`；`report/` → `docs/reports/`（保留编号） |
| `pose_translator_2/` | PT2 代码仓（独立的空间编解码） | 代码 | 6.5M | 08-17 | 归档 | 产出在 `~/alphamotion_artifacts/pose_translator_2` | `code/pose_translator_2`（归档） |
| `greenwich_eval_20260826/` | 08-26 Greenwich 评估包：中文评估文档、rollout 渲染、移动端服务、QA 视频 | 脚本 + artifacts | 17M | 08-26 | 归档 | — | `eval/archive/2026-08-26_greenwich/` |
| `_backup_before_delete_20260814/` | IsaacLab 本地改动的 diff + 未跟踪的 buffer 压缩包 | 备份 | 16M | 08-14 | 归档 | — | `archive/` |
| `mingrun/` | 组员 mingrun 的脚本：BONES-SEED 基准、ARDY 基准、量化（GPTQ/激活）、标定、边缘速度估计、动作查看器；`bones_seed` 链接到主仓的 raw | 代码 | 228M | 08-05 | 归档（组员工作区） | 链接 → 主仓 `alpha_motion2/wbc_expert/data/raw/bones_seed_full` | `code/contrib/mingrun`（只读） |
| `IsaacLab/` `IsaacLab-Arena/` `IsaacLab-Arena-2.3/` | Isaac Lab 和 Arena（记忆里写着"本机 Isaac Lab 已损"） | 源码 | 197M / 779M / 873M | 10-04 / 07-15 / 07-19 | 无关/归档 | — | `third_party/isaaclab*`，或留在原处 |
| `rot/` `rot-vs-code-plugin/` | ROT 项目（含 kimodo、docker） | 源码 | 8.7G / 208K | 08-03 / 02-05 | 无关 | — | 留在原处（不进 AlphaMotion 根） |
| `SegDepthFusion/` `claw-code/` `droctavius/` `src/`(openarmx ROS2) `openarm_description/` `openarm_hardware/` `openarmx_description/` `Manipulation_demo/` `lfs/` | 其它项目：分割深度融合、OpenArm 机械臂、ROS、操作演示 | — | 2.9G / 13M / 154M / 296M / 155M / 1.7G / 85M / 78M / 12K | 01~09 月 | 无关 | — | 留在原处 |
| 散文件 | 演示视频（Stealth Demo 1080p 和手机版，09-03，约 93M）、ArenaLabs 融资/研究 PDF、`babel_v1-0_release.zip`、`openarmx_ws_copy.tar.gz`（513M）、`Manipulation_demo.tar.gz`、`github-recovery-codes.txt` | — | 约 690M | 05~09 月 | 混合 | — | 演示视频 → `media/2026-09-03_stealth_demo/`；babel → `data/raw/babel/`；其余留在原处（恢复码别动） |

## 10. 顺带发现的家目录相关目录（只给指针）

| 目录 | 用途 | 大小 | 最后修改 | 状态 | 建议新家 |
|---|---|---|---|---|---|
| `~/umr_factory_data` | 数据工厂数据根（DATA_SPEC_v2、raw、smplx(_gpu)、corr、out、diversity、qa、audit、runs、workflows、platform_design） | 191G | 10-07 | 活跃 | 是统一根目录的候选（归别的地图） |
| `~/umr_factory_code` `~/umr_gpu` `~/umr_dataset` | 工厂代码（umrf）、GPU 引擎代码、数据集打包 | 468K / 996K / 23M | 10-04~10-07 | 活跃 | `code/` |
| `~/gpu_convert` | GPU SMPL-X/IK 转换（HiPHI 正在跑 `convert.py --kind hiphi`，用 conda env `genmo-gem`） | 6.5G（大多在 `runs/`） | 10-07 | 活跃 | `code/gpu_convert`；`runs/` → `data/retarget/` 或 `runs/` |
| `~/motion_audit_web` | 动作人工审计工作台（127.0.0.1:8765，人体表面点 vs MJCF；决定按 `audit/SCHEMA.md` 追加给工厂）；正在装 `mcp==2.3.0` | 3.2G（含 cache） | 10-07 | 活跃（**Web UI + MCP 的现成雏形**） | `code/motion_audit_web`；cache → `services/audit_web/state` |
| `~/codex_exports` | 斯坦福 Canopy 演讲 deck、PDF、预览图 | 124M | 07-24 | 归档 | `media/2026-07_stanford_deck/` |
| `~/alphamotion_cold` | 空目录（08-08 建的，打算做冷存储） | 4K | 08-08 | 空 | 删除，或改作 `archive/` |
| `~/alphamotion2_runtime` `~/alpha-motion-workspace` `~/.alphamotion_workspace` | 都是指向主仓（或 alpha_motion2）的软链接 | 0 | 07 月 | 链接 | 用 catalog 取代，留着当兼容层 |
| `/home/arenalabs/miniconda3/envs/{ardy,rot,genmo-gem}` | 在跑的 motion_lab 查看器（端口 2334）、workbench venv、gpu_convert 用到的 conda 环境 | — | — | 承重 | 在 `envs/` 里登记 |

---

## 11. 依赖关系图（谁读谁，挪之前一定要看）

```
xwbc_full(tar,129G) ──解包──> xwbc_lafan/extracted ──> topo_exp/cache_xwbc_lafan ─┐
                  └──构建──> topo_exp/cache_xwbc_amass_full ─────────────┐         │
                                                                  │(链接) │(链接)   │(链接)
~/tmp/cl_v3/{lafan,amass_lconv_tfix} ──(链接)──> topo_exp/cache_lafan_v3 + cache_amass_lconv_tfix
                                                             │
                                                             ▼
                 topo_exp/runs (FZ5c, L_S/M/L20k, …) ──导出──> Desktop/AlphaMotion/weights_staging ──> HF / Greenwich.git
                 topo_exp/bridge_codes(C1) ──> Equator A/B · Atlas · library（还是 C1 码）
                 topo_exp/runs ──> studio / greenwich_workbench / arena-datagrid（推理）
ctrl_verify/tracks_* ──> ~/alphamotion_ctrl/motions ──> mjlab 跟踪 run（logs/）
ctrl_verify/x2_corpus ──> ~/alphamotion_artifacts/x2_pool_v3 ──> amctrl X2 课程
config_harvest/GMR + gmr_spike/ik_configs ──> gmr_production ──> B_native 对照臂
robot_models + umr_factory/robots + am2exp/assets/surface_v2 + ~/tmp/sole_offsets3.json ──> 机器人资产（注册表现在分散在 4 处）
umr_factory/umr/env ──> GRAIL 循环 · run_factory · gpu bench · motion_audit_web(.mcp-venv)
reference_code/sonic ──> H2 MJCF/网格 ──> greenwich_workbench/h2_assets、amctrl/unitree_h2
```

## 12. 迁移顺序建议（这次什么都没执行）

1. **先建 catalog，再搬家**：catalog 里每条记录写上 id、原路径、新路径、大小、哈希或文件数、状态、上游、下游、许可证。新布局先用软链接"挂"上去（逻辑迁移），代码改成从 catalog 取路径以后，再动物理位置；旧路径留反向软链接一个月。
2. **第一批（小但承重，最怕丢）**：`~/tmp/{cl_v3, sole_offsets*.json, gauge_check_0903.json, dialect, refiner_suite, cl_v2_qc, contact_qc, arena_handoff_backup}` 和 `greenwich_workbench_0911/datagrid` 的运行态 → `data/labels`、`data/assets`、`services/datagrid`。`cl_v3` 搬完要同时重写 `cache_lafan_v3` 和 `cache_amass_lconv_tfix` 里的 34 个链接。
3. **第二批（活跃工厂）**：`umr_factory/umr/env` → `envs/umr`，UMR 克隆 → `third_party/UMR`。要等 GRAIL 24 小时循环和 GPU bench 跑完再搬，避免中途断掉；还要更新 `motion_audit_web/.mcp-venv` 的解释器链接。
4. **第三批（统一机器人资产）**：`robot_models` + `umr_factory/robots` + `am2exp/assets/surface_v2` + `sole_offsets3.json` + sonic 的 H2 资产，合成 `data/assets/robots/<robot>/`（每台：mjcf、meshes、surface_v2、sole_offset、umr_config、license）。
5. **第四批（大缓存，只在根分区和 /home 之间挪）**：`topo_exp/cache_*`、`bridge_codes`、`text_align` → `data/datasets/<name>@<ver>/`；`topo_exp/runs` 按血统分到 `runs/greenwich|equator/…`，非血统 run 进 `archive/`。
6. **要 lloyd 确认才能清理的**（预计能省约 180G）：`xwbc_full` 129G（可重新下载）、`any_body/env` 12G、`~/tmp/cl_v3/amass` 15G（只被旧缓存用）、`ufo_resume_drill` + `ufo_h2_smoke` 13.5G、`reference_code/ufo` 多出来的那套 venv 7.5–10G、`alphamotion_ctrl/.git` 用 gc 收回约 9G 以上、`gmr_production_v1` 0.6G、`x2_pool_v1_broken` 0.7G、`bfm_data_clone` 0.2G、`omniretarget` 里重复的 zip、坏链接（`cache_lafan_v2`、`cache_amass_4000`）、152 个 wandb 临时目录、`~/Desktop/be water, robot`（不带引号的副本）。
7. **防止再乱**：`scratch/` 一律用 `<YYYYMMDD>_<task>/` 命名，设 TTL，平台每天扫描，超期就提醒；任何被代码引用的路径不能落在 `scratch/` 或 `cache/`（CI 用 `git grep` 检查）；每个新产出目录必须带一个 `MANIFEST.json`（谁产出的、用了哪个 commit、输入是什么、属于哪个 catalog id）。
