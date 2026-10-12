# offtree_1012：09-09 以后写在 git 外面的 AlphaMotion 代码

- 收集日期：2026-10-12。分支 `research-1012`，从 `research-0909`（692bf77）分出。
- 规则：只收 2026-09-09 之后改过的代码、配置、文档（py/sh/js/md/html/yaml/toml/csv，小 json）。
  不收 checkpoint、npz/npy/pt/pkl、视频、图片、缓存、环境、数据，也不收评测结果大表。
- 原文件一个没动，这里只是拷贝（保留原修改时间）。下面目录结构跟原路径一一对应。
- 已扫过密钥（hf_/ghp_/github_pat_/sk-/password= 等），没有命中。

## 目录 → 原路径

| 本目录 | 原路径 | 是什么 | 最后修改 |
|---|---|---|---|
| `alpha_motion2_research/alphamotion_scaling_20261004/` | `alpha_motion2/research/alphamotion_scaling_20261004/`（主仓里，整个 `alpha_motion2/` 被 .gitignore 忽略） | AM2 规模实验主线：`orchestration/` 六组训练调度；`am2exp_v1/` 按臂训练 + ops 层 + 判官 J1（`judge/`）、J2（`e2/`）+ 评测协议（`eval/`）+ 三联出片（`render/`）及测试；`base/`、`surface/`、`prismatic/` 模型 A/B 臂代码；`rollout_tools/` 网格回放；`evaluation*/` 评测引擎代码（结果 json 没拷）；`reports/` 报告 md/html | 10-04 |
| `alpha_motion2_research/megatron_scale_pilot_20261004/` | `alpha_motion2/research/megatron_scale_pilot_20261004/` | AM2 训练配方库（`code/` 里 recipe_*、pilot_model、pilot_data 及测试）+ 试训前检、后处理、Qwen 名字编码审计 | 10-04 |
| `alpha_motion2_research/architecture_audit_20261003/` | `alpha_motion2/research/architecture_audit_20261003/` | AM2 架构审计交付包的文档和生成脚本（源码快照副本没拷，是重复的） | 10-03 |
| `alpha_motion2_research/transformer_review_20261003/` | `alpha_motion2/research/transformer_review_20261003/` | Transformer 评审交付包（文档 + 小脚本） | 10-03 |
| `alpha_motion2_research/embodied_case_20261003/` | `alpha_motion2/research/embodied_case_20261003/` | G1→H2 单帧具身案例的脚本和说明（给架构图用） | 10-03 |
| `alpha_motion2_research/training_recipe_audit_20261004/` | `alpha_motion2/research/training_recipe_audit_20261004/` | XWBC 骨架配对审计（1 个 json） | 10-03 |
| `alphamotion_video_resource/` | `~/Desktop/alphamotion video/resource/`（不是 git 仓） | 报告素材库：`INDEX.md` + 报告 16–35 的 md、脚本、小结果 json。含 ARDY+Greenwich 流水线（27/29/30/33）、精修器对决（28、31，`ruler.py` 在 28）、地面 gauge 审计（32）、G1 vs SOMA→H2（26）、IROS 讲稿生成（35）。视频没拷 | 09-09 ~ 09-27 |
| `var_tmp_am2exp/` | `/var/tmp/alphamotion/am2exp/` | AM2 实验第 0 阶段工作区：`stage0_index.md` 总索引、`judge/` 判官 J1 尺子与绑定 + 报告、`eval/` E1/E2 冻结病例 + 报告、`review/` 复查脚本和 findings。没拷 runs、integration、assets、videos、data、model | 10-04 |
| `home_tmp/` | `~/tmp/` 散文件 | 遥操两段式探针（`teleop_*.py`、`teleop2_*`）；H2 BeyondMimic/起身探针（`h2gr_*.py`、`h2_probe_uniform.py`、`h2_bfm_watch.sh`）；`ufo_h2_smoke/` H2 BFM-Zero 冒烟训练配置 | 09-15 ~ 09-17 |

## 同一提交里、但不在本目录的

| 路径 | 是什么 |
|---|---|
| `.agents/skills/motion-eval/SKILL.md`、`.agents/skills/spatial-eval/SKILL.md` | 评估流程技能（之前没进 git） |
| `Greenwich_G1_18_Paired_Benchmark_Review_20260920/` | G1 18 段配对基准报告 html + 两张 csv（视频没拷，html 里视频链接会失效） |

## 故意没收的

- UMR / 表面重定向相关：`umr_gpu`、`umr_factory_*`、`gpu_convert`、`umr_dataset`、`motion_audit_web`、`/var/tmp/alphamotion/umr_factory`、`surface-retargeting`、`alpha_motion2/research/hand_surface_case_20261004`（对照 UMR 参考）。
- 别的会话的 `alpha_motion2/pretraining_foundation/`；`greenwich_eval_20260826`；`alpha_motion2/external/`（第三方）。
- `~/tmp/refiner_suite/` 的代码：跟主仓已跟踪的 `pose_translator_design/exp_topo/refiner_suite/` 逐字节相同。
- `~/tmp/greenwich_workbench_0911/` 的 Greenwich / greenwich-motion-toolkit：是产品仓和 arena-datagrid 的副本，内容相同；其余是服务运行态和数据库。
- `~/tmp/h2*/`、`teleop2/` 里的分数 json：训练中途评测结果，不是代码。
- `Greenwich_G1_18_.../Academic_Outreach_100_Discussion_20260922/`：学者联络名单（含邮箱），不是代码。
- `agent_chat/README.md` 的改动：只多了两个乱码字符，不是有效改动。
- `experiment/ardy/checkpoints/*/config.yaml`：下载来的第三方 checkpoint 自带配置。
- 独立 git 仓（`~/alphamotion_ctrl`、`~/Desktop/arena-datagrid`、`~/Desktop/AlphaMotion`）里没提交的改动：应该在各自仓里提交，这里不拷。
