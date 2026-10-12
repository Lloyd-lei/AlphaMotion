# 研究主仓地图:`~/Desktop/"be water, robot"`

> 盘点时间 2026-10-07(只读,没动任何东西)。分支 `research-0909`,和 origin 同步,最后一次提交 09-09。
> 大小用 `du`,"最后修改"取的是目录里最新文件的日期(不是目录本身的日期)。
> 状态:**活跃** = 有进程/服务/别的代码正在用,或者最近在动;**归档** = 结论已出,只留着查;**废弃** = 空壳、重复、过期,可以删或并进别处。

## 0. 先看这几条(最要紧)

1. **路径有两个"孪生"目录。** 真仓库名字里带英文双引号:`/home/arenalabs/Desktop/"be water, robot"`(85 GB)。
   旁边还有一个不带引号的 `/home/arenalabs/Desktop/be water, robot`(557 MB),是某个脚本引号写错后误建的。
   里面有**独一份**的 `pose_translator_design/SCALING_LAW.md`(08-29,horizon 课程数据缩放)、一份不同版本的 `CTRL_VERIFY_GR3.md`、和 `R4_VIZ/mp4/ctrl_verify/x2/`。要先合并进真仓库再删。
2. **git 只管了约 3% 的内容。** `.gitignore` 把 `alpha_motion2/`(53 GB,包括 10-03~10-07 最新的预训练底座和规模实验)、`GENMO/`、`any_body/`、`experiment/ardy`、`experiment/synergy` 全排除了。最新的工作**没有任何版本控制**。被跟踪的主要是 `pose_translator_design`(599 个文件)、`notebook`、`experiment/stage1`、`claude/`。
3. **最重要的代码在这,但它的产物不在这。** `pose_translator_design/exp_topo`(只有 8.3 MB,Greenwich/Equator 全系代码 + 判官 + 渲染器)把 checkpoint、缓存、视频全写到 `/var/tmp/alphamotion/`(352 GB,在仓库外;/var/tmp 的 30 天自动清理是注释掉的,目前不会被清)。
4. **正在跑、依赖这个仓库的东西:**
   - 3 个 systemd 用户服务,通过软链 `~/.alphamotion_workspace`、`~/alpha-motion-workspace`(都指向本仓库)启动:
     - `alphamotion-motion-lab`(端口 2334,读 `alpha_motion2/demo/ardy_preflight_20/motions`)
     - `alphamotion-rollout-gallery`(端口 8891,读 `alpha_motion2/demo/alphamotion_verified_g1_atlas_20`)
     - `alphamotion-ardy-isaac`:**开机 33 分钟里崩溃重启了 374 次**(每 5 秒一次),原因是 `~/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml` 不存在。白白耗 CPU,和 RESOURCE_LIMITS 的要求相冲突,建议停掉或修好(本任务只读,没动)。
   - 下载器 `pretraining_foundation/acquire.py`(PID 43946)+ `queue_followup.py`(PID 43763):正在往 HF 缓存拉 UniMate 全部模型,还剩约 116 GB,碰到过 `Connection reset`。这是重网络任务,和"USB 网卡高负载掉线"直接相关。
   - 另外还有 `~/alphamotion2_runtime` → `alpha_motion2`,和一个**坏掉的**软链 `~/Desktop/mingrun/bones_seed` → `alpha_motion2/wbc_expert/data/raw/bones_seed_full`(目标已被删)。
5. **仓库外代码写死了本仓库路径(搬家前必须先改):**
   - `umr_factory_data/robots/*/staging.json`、`pipeline.json`:19 处指向 `alpha_motion2/external/GMR/assets/...` 的 MJCF;
     `umr_factory_data/diversity/core_sources/scripts/run_xwbc.py` 读 `data/babel/babel_v1.0_release`;`build_summary.py` 读 `alpha_motion2/pretraining_foundation/logs/acquire_followup/queue_state.json`。
   - `arena-datagrid`:6 处写死 `alpha_motion2/external/GMR/general_motion_retargeting/params.py`(showcase_render / x2_phys / slip_audit / wbc_ready_audit / render_rollout / arena/eval/visual.py,最后一个可用环境变量 `ARENA_GMR_PARAMS` 覆盖);4 处写死 `pose_translator_design/exp_topo`;1 处写死 `external/soma-retargeter` 的 bvh。
   - Studio(`/var/tmp/alphamotion/studio`):7 个文件通过 `~/.alphamotion_workspace/pose_translator_design/exp_topo` 加载模型代码;`studio_gem_worker.py` 写死 `GENMO/`。
   - conda 环境 `genmo-gem` 把 `GENMO/` 装成了 editable 包(`gem`)。
   - `/var/tmp/alphamotion/topo_exp/night2_lib/g1x_mesh.py` 是软链,指回 `pose_translator_design/grand_eval/src/.../g1x_mesh_compat.py`。
6. **研究仓里混着很多不是研究的东西:** 融资/路演(`presentation` 431 MB、`market_research` 213 MB、`pitch_deck_20260722` 33 MB、根目录 Stanford PDF、`nvidia_inception.md`)、个人股权笔记 `share_struct.md`(还被 git 跟踪了)、学术外联名单(`Greenwich_*/Academic_Outreach_*`)。建议整体挪到单独的 `business/`。
7. **空壳和垃圾:** `alpha_motion2/{evaluation,geometry,heads,losses,models,modules,training}` 和 `alpha_motion2/data/*.py` 全是 05-31 建的 **0 字节**文件;`tool-kits/`、`experiment/openfold_ref/` 是空目录;`mon.pid`、`server_8844.pid` 指向已死进程;`pytest-of-arenalabs/`(还被 git 跟踪)、`torchinductor_arenalabs/`、`MUJOCO_LOG.TXT` 是运行残留;根目录 `README.md` 讲的是已作废的 Stage-1/三角注意力路线。
8. **可回收的大头:** `GENMO/inputs` 20 GB(权重和 onnx)、`alpha_motion2/research/alphamotion_scaling_20261004` 21 GB(已完结)、`alpha_motion2/runs/synergy` 9.8 GB(7 月,已归档)、`experiment/kimodo/.venv` 5.3 GB、`alpha_motion2/demo` 4.2 GB、`alpha_motion2/external` 13 GB(大部分是第三方仓库副本)。

## 1. 建议的统一骨架(本表"建议归宿"一列用的就是这些名字)

> 只是给总设计的草案,按 lloyd 说的 7 个环节排序,具体根目录名由总设计定,这里写作 `<ROOT>`。

```
<ROOT>/
  01_data/        raw/<来源>/(只读原始数据+许可证)  clean/(统一 gauge/地面/单位)  acquire/(下载计划与日志)
  02_surface/     表面点云采样、区域标签、robots/<机器人id>/(机器人注册表与模型文件)
  03_retarget/    UMR CPU 参考 + GPU 引擎 + 严格后处理
  04_dataset_eval/ 整库评估:多样性/分类覆盖/质量/人审
  05_store/       catalog(唯一真相源)、shards/(训练分片)、版本与磁盘预算
  06_train/       models/<模型族>/code + runs/<日期_名字>/(checkpoint、曲线) + viewers/(实时可视化)
  07_eval/        judges/(独立判官)  reports/<日期_名字>/(结论+三联视频)
  services/       MCP + HTTP API + web;systemd 单元文件
  third_party/    外部仓库,按 <名字>@<commit> 固定版本,不在里面改代码(要改就打补丁)
  research/       notes/(设计文档、文献)  verdicts/(判决记录,只追加)  cases/(专题案例)
  archive/        冻结的旧路线,只读
business/(放在 <ROOT> 外)  路演、市场调研、品牌素材、外联名单
```

## 2. 顶层目录总表

| 目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `alpha_motion2/` | 第二代研究大杂烩:7 月的 ARDY/SONIC/Isaac 演示栈 + 10 月的预训练底座和规模实验 + 第三方仓库 | 见第 3 节 | 53 GB | 10-07 | 混合(子目录分开看) | 被 3 个 systemd 服务、umr_factory_data、arena-datagrid 依赖;**整个目录不在 git 里** | 拆分,见第 3 节 |
| `pose_translator_design/` | Greenwich(空间层)/ Equator(时序层)主研究线,8 月到 9 月 | `exp_topo/` 全部模型/训练/判官/渲染代码;50+ 份报告和文献综述(LIT_*、*_RECIPE、*_REPORT);`verdicts/` 判决;`grand_eval/` 08-08 总评估和白皮书;`R4_VIZ/` 可视化 | 1.2 GB | 09-09 | **活跃**(代码被 Studio、arena-datagrid 引用) | 产物写到 `/var/tmp/alphamotion/topo_exp`;读 `external/GMR`、`~/tmp/*`、`~/Desktop/AlphaMotion` | 拆分,见第 4 节 |
| `GENMO/` | NVlabs GENMO 克隆(从视频估人体动作,包名 `gem`) | `outputs/` 611 MB 演示结果;`inputs/` 20 GB 权重/onnx | 20 GB | 09-09 | 活跃(依赖项) | 被 Studio 的 GEM 适配器、conda 环境 `genmo-gem`(editable 安装)依赖 | `third_party/GENMO@<commit>`;权重挪到 `05_store/weights/genmo/` |
| `experiment/` | 早期实验 + 两个 NV 第三方仓库 | `stage1/` MotionFormer Stage-1;`kimodo/`、`ardy/` 两个克隆;`synergy/` 机理分析结果 | 7.7 GB | 09-23 | 混合 | `alpha_motion2/live`、`scripts` 读 `experiment/ardy`、`experiment/kimodo` 的资产和 venv | 拆分,见第 5 节 |
| `notebook/` | Stage-1 时期的 notebook(5 月) | 3 个 ipynb + PDF、`eval_foundation_demos`、`eval_temporal_results` | 1.3 GB | 05-30 | 归档 | 无 | `archive/2026-05_stage1/notebook/` |
| `any_body/` | Round-1 "AnyBody" 通用单帧跨拓扑姿态翻译器(7 月底) | `runs/` 1.1 GB、`artifacts/` 82 MB | 1.2 GB | 08-01 | 归档(已被 exp_topo/Greenwich 取代) | 依据 `claude/01_round1_spec.md`;`/var/tmp/alphamotion/any_body` 有对应产物 | `archive/2026-07_anybody/` |
| `papers-docs/` | 论文 PDF(含中文/双语翻译版),40 份 | — | 597 MB | 04-21 | 参考资料 | 无 | `research/papers/` |
| `Greenwich_G1_18_Paired_Benchmark_Review_20260920/` | G1 上 18 段动作"Greenwich vs 传统重定向"对照评测的交付包(从 Mac 拷来,带 .DS_Store) | HTML 报告、`data/*.csv`、18 段对照视频、5 段宣传片、学术外联名单 | 247 MB | 09-24 | 归档(未提交 git) | 无 | 评测部分 → `07_eval/reports/2026-09-20_g1_18_paired/`;宣传片和外联名单 → `business/` |
| `data/` | 第一代数据准备脚本 + BABEL 标签 | `action_labels.py`(12 类正则分类)、`splits.json`、`babel/`、`babel_labels.json` | 45 MB | 05-16 | 半活跃 | `umr_factory_data/.../run_xwbc.py` 读 `data/babel/babel_v1.0_release`;`amass/humanml3d/motionx` 子目录已空(原始数据不在这了) | `01_data/raw/babel/`;脚本 → `archive/2026-05_stage1/data_prep/` |
| `.git/` | 仓库历史 | — | 69 MB | 10-05 | 活跃 | origin = `github.com:Lloyd-lei/AlphaMotion` | 保留 |
| `deliverables/` | AMASS 留出集零样本网格验证视频 | 1 个 mp4 + audit.json + 截图 | 11 MB | 08-17 | 归档 | 无 | `07_eval/reports/2026-08-17_amass_zeroshot_mesh/` |
| `note/` | Stage-1 研究愿景和三角注意力复盘 | 4 份 md/ipynb + 图 | 512 KB | 05-16 | 归档(路线已作废) | 无 | `archive/2026-05_stage1/notes/` |
| `claude/` | lloyd 和 Claude 的架构设计文档(T 型架构、数学框架、消融方案、Round-1 规格等) | 19 份 md,`INDEX.md` 写明哪份是现行版 | 368 KB | 08-10 | 参考(部分已被取代) | `any_body` 依据其中 round1 规格 | `research/notes/design_docs/` |
| `agent_chat/` | Claude/Codex/lloyd 三方群聊 CLI | `chat.md`(105 KB 记录)、`agent_chat.py` | 132 KB | 09-12(README 有未提交改动) | 基本停用 | `codex/codex.md` 引用它 | 被新 MCP 服务取代后 → `archive/agent_chat/` |
| `codex/` | Codex 的角色说明和 Round-1 预检评审 | 2 份 md | 20 KB | 09-09 | 归档 | 引用 agent_chat | `archive/agent_chat/` |
| `.claude/` | 项目级 Claude 设置和两个技能(`spatial-eval`、`motion-eval`) | 2 个 SKILL.md | 32 KB | 08-10 | **活跃**(Claude 会话会加载) | — | 保留在新根目录的 `.claude/` |
| `pytest-of-arenalabs/` | pytest 临时目录(还被 git 跟踪了) | — | 152 KB | 07-20 | 废弃 | — | 删,并加进 .gitignore |
| `torchinductor_arenalabs/`、`.pytest_cache/`、`.ruff_cache/`、`.vscode/` | 工具缓存 | — | <100 KB | — | 废弃 | — | 删 / 忽略 |
| `tool-kits/` | 空目录 | — | 4 KB | 04-20 | 废弃 | — | 删 |
| 根目录 `*.fbx`(3 个,Take_005/012) | 自采动捕原始数据 | — | 5 MB | 06-27 | 原始数据 | `alpha_motion2/fbx_preview` 预览过 | `01_data/raw/arena_mocap/` |
| 根目录 `README.md`、`agent.md`、`AlphaMotion_infra.md` | 旧版仓库说明 / 课程大纲 / 一个 76 字节的草稿 | — | 15 KB | 05~08 | 过期 | — | 重写一份新 README;旧的 → `archive/` |
| 根目录 `share_struct.md`、`nvidia_inception.md`、`ArenaLabs_Stanford_Canopy_Presenter_Notes.pdf` | 股权说明、NVIDIA Inception 申请文案、路演讲稿 | — | 4 MB | 05~07 | 与研究无关 | — | `business/` |
| 根目录 `mon.pid`、`server_8844.pid`、`MUJOCO_LOG.TXT` | 已死进程的 PID、一次仿真不稳定的日志 | — | <1 KB | — | 废弃 | — | 删 |
| **仓库外孪生** `~/Desktop/be water, robot/`(无引号) | 引号写错误建的目录 | `SCALING_LAW.md`(独一份)、另一版 `CTRL_VERIFY_GR3.md`、1 个 ctrl_verify 视频 | 557 MB | 08-29 | 废弃(但内容要先救出来) | 无 | 合并进 `research/notes/` 和 `07_eval/reports/` 后删除 |

## 3. `alpha_motion2/` 细分(53 GB,整体不在 git 里)

| 子目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `pretraining_foundation/` | 预训练数据底座:下载器、来源清单、索引、数据加载器 | `cache/index*.sqlite`(240,444 条序列×机器人记录,G1 留出版 171,504 条可训)、`manifests/`(上游版本/哈希/许可证、已有 XWBC 资产)、`loader.py`(按窗口读、带掩码)、31 个测试通过 | 82 MB | 10-07 | **活跃**(下载器 PID 43946/43763 正在跑) | 被 `umr_factory_data/diversity/.../build_summary.py` 读日志;被 `research/alphamotion_scaling_*` 当数据来源;数据本体在 `~/.cache/huggingface/hub` | 索引和清单 → `05_store/catalog/`;下载器和计划 → `01_data/acquire/`;loader → `06_train/loaders/` |
| `research/alphamotion_scaling_20261004/` | 6 个模型 A/B × S/M/L 规模对照(A=共享骨架,B=加表面块和几何损失),各 6000 步 | `runs/`(9.3 GB,6 个 checkpoint 目录)、`continuation_v3/`(4.4 GB)、`evaluation*`(约 6 GB)、`reports/`(PDF、曲线、对照视频、`rollout_mesh_v1/index.html`)、`rollout_tools/`(可复用的渲染 CLI) | 21 GB | 10-04 | 归档(已完结,README 写明"不要重启") | 读 pretraining_foundation 索引;共享 GPU 锁 | 训练 → `06_train/runs/2026-10-04_scaling_AB_SML/`;报告 → `07_eval/reports/`;`rollout_tools/` → `07_eval/judges/` 或 `06_train/viewers/` 候选 |
| `research/megatron_scale_pilot_20261004/` | Megatron 规模试训的准备和试跑 | `code/`、`data_xwbc/`(712 MB)、`runs/`(338 MB)、数据审计、Qwen 审计 | 1.1 GB | 10-04 | 归档 | 两个 0 字节锁文件残留 | `06_train/runs/2026-10-04_megatron_pilot/` |
| `research/megatro_scale_pilot_20261004/` | 上面那个的**拼写错误副本**,只有一个 `qwen_audit/` | — | 40 KB | 10-03 | 废弃 | — | 并进上一行后删 |
| `research/embodied_case_20261003/` | G1→H2 单帧具身案例,给架构示意图用的真实张量和投影 | 网格、注意力/FSQ 图、trace.npz、评审压缩包 | 106 MB | 10-03 | 归档 | — | `research/cases/2026-10-03_g1_to_h2/` |
| `research/hand_surface_case_20261004/` | 手部表面接触区域案例,对照 UMR 参考 | 构造姿态、区域对比脚本、评审压缩包 | 53 MB | 10-04 | 归档(和第 2 环节"表面采样"直接相关) | 用 UMR 参考 | `02_surface/cases/2026-10-04_hand/` |
| `research/architecture_audit_20261003/`、`transformer_review_20261003/` | 架构审计、Transformer 评审的交付包 | 压缩包 + 校验 json | 42 MB + 21 MB | 10-03 | 归档 | — | `research/reviews/` |
| `research/training_recipe_audit_20261004/` | XWBC 骨架配对审计 | 1 个 json | 60 KB | 10-03 | 归档 | — | `04_dataset_eval/audits/` |
| `research/bones_samples/`、`research/motionbricks/` | BONES-SEED 样例截图;MotionBricks 论文 | 图片、PDF/txt | 2.4 MB + 47 MB | 06-03 | 参考 | — | `research/papers/` |
| `external/GMR` | 通用动作重定向(无 git,不知道版本) | 机器人 MJCF 资产、`params.py` | 1.2 GB | 04-02 | **活跃(关键依赖)** | umr_factory_data 19 处、arena-datagrid 6 处、exp_topo 10 处写死它 | `third_party/GMR@<补记版本>`;机器人模型同步一份到 `02_surface/robots/` |
| `external/UMR` | UMR 参考实现(10-02 的提交) | — | 1.3 GB | 10-03 | 活跃(重定向 CPU 参考) | 被 hand_surface_case 用;umr_factory_data 没写死这个路径 | `third_party/UMR@<commit>` |
| `external/UMR-website` | UMR 项目网页(.git 损坏,读不出版本) | 网页素材 | 659 MB | 10-03 | 参考 | — | `third_party/` 或删 |
| `external/UniMate` | UniMate(09-30 提交) | — | 64 MB | 10-03 | 活跃 | `pretraining_foundation/unimate_adapter.py` | `third_party/UniMate@<commit>` |
| `external/GRAIL` | NVlabs GRAIL;**本地改动了 3,465 个文件** | — | 3.3 GB | 07-20 | 半活跃 | `alpha_motion2/data/grail_*` 用它产出 | 先把本地改动导成补丁,再放 `third_party/GRAIL@<commit>` + `patches/` |
| `external/whole_body_tracking` | BeyondMimic 跟踪训练 | — | 1.3 GB | 07-14 | 参考 | `wbc_expert` 包了它一层 | `third_party/` |
| `external/soma-retargeter` | NV SOMA 重定向器 | bvh 示例动作 | 982 MB | 06-28 | 半活跃 | arena-datagrid 的一个 manifest 写死了其中一个 bvh | `third_party/` |
| `external/robocasa`(2 处本地改动)、`GR00T-WholeBodyControl-motionbricks`、`nvidia-cosmos`、`cosmos-framework` | 场景/WBC/世界模型第三方仓库 | — | 3.4 GB + 319 MB + 86 MB + 20 MB | 06~07 | 归档 | 7 月演示栈用过 | `third_party/`(用不到的可删,需要时再克隆) |
| `runs/` | 7 月实验输出:`synergy/` 662 个机理分析运行、improvisation 第 0 阶段、吞吐基准 | — | 11 GB(synergy 9.8 GB) | 07-21 | 归档 | 无 | `archive/2026-07_ardy_sonic_stack/runs/`(可压缩或删) |
| `demo/` | 7 月 ARDY/Isaac/SONIC 演示、投资人视频、精选动作集(59 个子目录,含 `deprecated/`) | 视频、轨迹、画廊数据 | 4.2 GB | 07-30 | 归档,**但有 2 个子目录正被服务读** | `ardy_preflight_20/motions` ← motion-lab 服务(2334);`alphamotion_verified_g1_atlas_20` ← rollout-gallery 服务(8891) | 这 2 个 → `services/data/`;其余 → `archive/2026-07_demos/`;投资人视频 → `business/` |
| `wbc_expert/` | BeyondMimic 风格 G1 跟踪专家的薄封装 | `runs/`(915 MB,含仪表盘/监控 pid 残留)、`data/raw`(LAFAN1-G1、OmniRetarget、BONES 探针)、`data/processed` | 1.5 GB | 07-14 | 归档 | exp_topo 有 3 处引用 `wbc_expert/data`;`~/Desktop/mingrun/bones_seed` 指向这里已删的子目录(坏链) | 原始数据 → `01_data/raw/`;其余 → `archive/2026-06_wbc_expert/` |
| `web/` | 7 月的 FastAPI 小应用:ARDY+Isaac 实时应用、已验证轨迹画廊、SONIC 应用、投资人画廊 | — | 147 MB | 07-21 | 2 个在跑(其一崩溃循环) | `verified_rollout_gallery.py` ← 8891 服务;`ardy_isaac_live_app.py` ← 崩溃中的服务 | 先修/停服务;代码 → `archive/2026-07_ardy_sonic_stack/web/`,好用的界面思路给新 `services/` 参考 |
| `scripts/` | 168 个脚本:ARDY/SONIC 因果分析、精选视频、Isaac 渲染、MuJoCo 回放 | — | 5.7 MB | 07-21 | 归档(`alphamotion_motion_lab.py` 在被服务跑) | 读 `experiment/ardy`、`experiment/kimodo`、`experiment/synergy` | motion_lab 一类实时查看器 → `06_train/viewers/`;其余 → `archive/` |
| `synergy/`、`improvisation/`、`agentic_motion/`、`live/`、`scene_planner/`、`isaac_common/`、`isaac_hotel/`、`isaac_reference_smoke/`、`isaac_scene_live/`、`contact_control.py`、`asset_pipeline.py` | 7 月 ARDY/SONIC/Isaac 演示栈和机理分析代码 | 代码 | 合计约 7 MB | 07-13 ~ 07-21 | 归档(`live/`、`agentic_motion/` 被崩溃中的 ardy-isaac 服务引用) | 依赖 `~/Desktop/GR00T-WholeBodyControl`(已不存在) | `archive/2026-07_ardy_sonic_stack/` |
| `data/` | GRAIL 候选、地形评估 + 5 个 0 字节 py 文件 | — | 298 MB | 07-15 | 归档 | 用 external/GRAIL 产出 | `archive/2026-07_grail/`;空 py 删 |
| `analysis/boneseed_locomotion_index/` | BONES-SEED 运动片段分级表(tier0~3、排除表) | CSV | 104 MB | 06-29 | 参考(对数据评估有用) | — | `04_dataset_eval/legacy/boneseed_tiers/` |
| `configs/`、`tests/`、`notes/`、`docs/`、`deploy/` | 演示栈配置、测试、早期笔记、服务单元文件 | — | 约 3 MB | 05-31 ~ 07-21 | 归档 | `deploy/alphamotion-ardy-isaac.service` 是崩溃服务的模板 | `archive/2026-07_ardy_sonic_stack/`;service 文件 → `services/systemd/` |
| `evaluation/`、`geometry/`、`heads/`、`losses/`、`models/`、`modules/`、`training/` | 05-31 搭的架子,**全部是 0 字节文件** | 无 | 4 KB × 7 | 05-31 | 废弃 | 无 | 删 |
| 13 个预览目录:`smplx22_preview*`、`smplx_mesh_preview*`、`soma_official_mesh_preview*`、`somax_cleaned_*`、`soma_locomotion_seed_v0_preview`、`fbx_preview` | 6 月底一次性的格式/网格预览(含 smoke/fixed 多个版本) | 帧图、mp4、`fbx_preview/server.pid` | 合计约 30 MB | 06-27 ~ 07-01 | 废弃 | 无 | 留最终版到 `archive/2026-06_format_previews/`,其余删 |
| `presentation/`、`market_research/`、`pitch_deck_20260722/`、`assets/`(品牌/字体) | 路演、市场调研、品牌素材 | PPT、PDF、图片 | 431 + 213 + 33 + 15 MB | 07-18 ~ 07-24 | 与研究无关 | 无 | `business/` |
| `__pycache__/`、`.pytest_cache/` | 缓存 | — | <100 KB | — | 废弃 | — | 删 |

## 4. `pose_translator_design/` 细分(1.2 GB,被 git 跟踪)

| 子目录 / 文件 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `exp_topo/` | **Greenwich/Equator 全系代码**:`train_dual.py`、`model.py`、`dual_common.py`、规模阶梯脚本、判官(`wbc_ready_audit.py`、`slip_audit.py`、`x2_phys.py`、`gauge_check.py`)、渲染器(`showcase_render.py`、`greenwich_carnival.py`)、`refiner_suite/` 精修器、`dialect/` 人体方言 Q、`ctrl_verify/` 控制验证 | 代码 243 个文件;产物全在 `/var/tmp/alphamotion/topo_exp` 等处 | 8.3 MB | 09-09 | **活跃** | 被 Studio 7 处、arena-datagrid 4 处引用;读 `external/GMR`、`/var/tmp/alphamotion/*`、`~/tmp/*`、`~/Desktop/AlphaMotion` | 模型和训练 → `06_train/models/greenwich/`;判官 → `07_eval/judges/`;渲染器 → `07_eval/render/`;gauge 检查 → `01_data/clean/`;精修器 → `03_retarget/refiners/` |
| `verdicts/` | 历次对决/消融的判决记录(108 个文件) | json/txt | 724 KB | 08-14 | 参考(重要) | — | `research/verdicts/`(只追加) |
| `grand_eval/` | 08-08 全模型总评估层:模型台账、白皮书中英文、展示视频 | `deliverables/`、`whitepaper/`、`paper/`、`src/` | 101 MB | 08-08 | 归档 | `/var/tmp/alphamotion/topo_exp/night2_lib/g1x_mesh.py` 软链指回它的 src | `07_eval/reports/2026-08-08_grand_eval/`(src 里的工具并入 `07_eval/judges/`) |
| `R4_VIZ/` | 8 月可视化:网格重建/迁移 gif、GMR 质检、新机器人、xwbc 样例 | 图片/mp4(mp4 被 gitignore) | 1.1 GB | 08-11 | 归档 | — | `archive/2026-08_greenwich/viz/` |
| `R3_REVIEW/`、`R7_LATENT/`、`R8_DIM/`、`R9_QWEN_BASELINE_figs/`、`figures/`、`viz/`、`audits/`、`scripts/gen_figures.py` | 各轮评审、潜空间/维度实验图、深度审计 | 图和 md | 合计约 6 MB | 08-02 ~ 08-07 | 归档 | — | `archive/2026-08_greenwich/` |
| `long_demo_20260903/` | 长演示拼接脚本 | 1 个 py | 32 KB | 09-09 | 归档 | — | `06_train/viewers/` 或 `archive/` |
| `gmr_production_data` | 软链 → `/var/tmp/alphamotion/gmr_production` | — | 链接 | 08-03 | 活跃数据入口 | — | `05_store/` 里正式登记这份 GMR 生产数据 |
| 约 50 份顶层 md/pdf/tex/html | 方案、配方、文献综述、诊断报告:`MODEL_VERSIONS.md`(模型登记簿)、`RUNS_INDEX.md`、`GREENWICH_V2_SPEC.md`、`CONTACT_LABEL_DATA.md`、`*_RECIPE.md`(BeyondMimic/BFM-Zero/SONIC)、`LIT_*.md`、`HAND_DIAGNOSIS.md` 等 | 文档 | 约 2 MB | 08-01 ~ 08-31 | 参考(`MODEL_VERSIONS`、`RUNS_INDEX` 是现成的台账) | Studio README 引用 | 台账 → `05_store/registry/`(并进 catalog);其余 → `research/notes/` |

## 5. `experiment/` 细分(7.7 GB)

| 子目录 | 用途 | 产出了什么 | 大小 | 最后修改 | 状态 | 依赖 / 被谁依赖 | 建议归宿 |
|---|---|---|---|---|---|---|---|
| `stage1/` | MotionFormer Stage-1(pair-first 主干、三角注意力) | 代码 + `runs/` 日志和诊断图 | 281 MB | 05-02 | 归档(路线已作废) | git 跟踪 | `archive/2026-05_stage1/` |
| `kimodo/` | nv-tlabs kimodo 克隆,带自己的 `.venv`(5.3 GB) | — | 5.4 GB | 06-01 | 半活跃 | `alpha_motion2/scripts` 有 7 处读它的 g1.xml、皮肤模型和 venv python;conda 的 kimodo 实际装的是 `~/Desktop/rot/kimodo`(另一份) | `third_party/kimodo@<commit>`;`.venv` 可删,统一用 conda |
| `ardy/` | nv-tlabs ARDY 克隆 + 1.5 GB checkpoint(Core-20FPS、G1-25FPS);09-23 编译过 MotionCorrection | — | 1.6 GB | 09-23 | 活跃(ARDY+Greenwich 流水线) | `alpha_motion2/live/ardy_generator.py`、两个 scripts 读它 | `third_party/ardy@<commit>`;权重 → `05_store/weights/ardy/` |
| `synergy/` | 机理分析 phase0~5 的结果 | json、图 | 596 MB | 07-21 | 归档 | `alpha_motion2/scripts` 里的报告脚本读它 | `archive/2026-07_ardy_sonic_stack/synergy_results/` |
| `openfold_ref/` | 空 | — | 4 KB | 04-21 | 废弃 | — | 删 |

## 6. 仓库外但相关(供总盘点交叉核对)

- `/var/tmp/alphamotion/`(352 GB):exp_topo 所有 checkpoint(`topo_exp/`)、机器人模型、配置收割、GMR 生产数据、Studio 前端(`studio/`)。
- `~/Desktop/greenwich_eval_20260826/`:`greenwich-mobile` 服务(端口 8899)的网页画廊,不在本仓库里。
- `~/Desktop/AlphaMotion/`:产品包(conda `ardy` 环境以 editable 方式装的 `alphamotion` 就是它),exp_topo 有 16 处引用。
- `~/tmp/`(`dialect`、`cl_v3`、`sole_offsets3.json`、`y*_odo` 等):exp_topo 读写的临时实验目录。
- `~/.cache/huggingface/hub`:pretraining_foundation 下载的数据和模型本体。

## 7. 搬家顺序建议(给总设计参考)

1. 先救孪生目录的 3 样东西,再删孪生目录。
2. 先修或停掉 `alphamotion-ardy-isaac` 崩溃循环;下载器跑完或暂停后再动 `pretraining_foundation`。
3. 先把写死的路径换成一个配置变量(例如 `ARENA_GMR_ROOT`、`ALPHAMOTION_EXP_TOPO`),确认 umr_factory_data / arena-datagrid / Studio 都读变量,再挪 `external/GMR` 和 `exp_topo`。软链 `~/.alphamotion_workspace` 可以当过渡。
4. 把 `alpha_motion2/pretraining_foundation` 和 `research/*` 的代码与小文件纳入 git(大产物进 `05_store`,只在 catalog 里登记)。
5. 业务资料整体挪出;空壳、pid、缓存直接删;大块归档(synergy runs、scaling runs、GENMO onnx、kimodo .venv)先登记再压缩或删除。
