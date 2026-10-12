# AlphaMotion tech report 素材索引（0903 收集，0904 补 Chinchilla 阶梯 + SE(3) 精修器）

全部是**复制**，原件留在原处（管线还在读它们）。日期 = 产出日期。

## report/

| 目录 | 内容 | 原件位置 |
|---|---|---|
| `00_chinchilla_ladder_0904.md` | **Chinchilla 阶梯 S20k/M20k/L20k 终账**：两段血统 + GPU 时长、三尺子表（eval_dual / 600 条审计 / X2）、第 20k 步"坎"的解释、判决（M 之后打平 = 数据天花板，候选 M） | 本文件汇总 |
| `00_key_metrics_0903.md` | **关键指标汇总**：25 个 ckpt × 3 台机器人的 600 条语料外审计表、三方对照主表、X2 判官要点、阶梯 eval、血统 | 由 `~/tmp/wbcready_*_<robot>.txt` 的 ALL 行自动汇总 |
| `00_session_verdicts_digest.md` | 只存在于工作记录里的判决（GR3 地面对决、接触标签 bug、脚滑真因、WBC 试训…） | 会话记忆 |
| `01_plans/` | 本轮 scaling 计划（含 §8 进度）、PLAN_V3（+git 历史 patch）、LOCKED_PLAN、GREENWICH_V2_SPEC、WBC_TRANSPLANT_LIST | `~/.claude/plans/`, `pose_translator_design/` |
| `02_audit_wbc_ready/` | 主线 ckpt 的 600 条审计原报告（FZ5c / S / S0 / F32 / Y5b / Z3 / J0 / J1 + **S20k / M20k / L20k**，各 json + 3 台 txt） | `~/tmp/wbcready_*` |
| `03_x2_physics_judge/` | X2 零样本免真值物理判官：对照表（FZ5c vs S；**x2_phys_4way_0903.txt 五方**）+ 五份逐片段 json | `~/tmp/x2phys/` |
| `04_ladder_eval/` | 阶梯 eval_dual.json（16×16 转换矩阵）、loss_log.jsonl（**含 S20k/M20k/L20k 两段各一份**）、eval_log、config、ladder_0903.md（三档版）、E24 旧阶梯表 + 原始 log | `runs/L_S*_0903*`, `runs/E24_*` |
| `05_whitepaper_grand_eval/` | 0808 白皮书（EN/CN tex + PDF）、CLAIM_MATRIX（哪些结论敢写）、PROTOCOL、evidence_pack、model_inventory、RUNS_INDEX、MODEL_VERSIONS | `pose_translator_design/grand_eval/` |
| `06_method_architecture/` | APPENDIX_STATE_0814（**最佳入口**）、E1–E18 附录、ARCH_V4/V5、ENC_DEC_DESIGN、DOF_FEATURES、ENCODER_BAKEOFF、V2 接触码、E-BFM、20 秒核心一页、机制审计（tex/pdf/md） | `pose_translator_design/` |
| `07_results_verdicts/` | SCALING_LAW_UPSTREAM（119 run 挖出的 8.03° 地板）、ABLATION_TABLE、R3–R10、BFM 压力测试、手/高度/hpos 诊断、三具身地面、CTRL_VERIFY_GR3、LIMITS_AUDIT、P/HPOS json、`verdicts/` 全部 76 份判决 | `pose_translator_design/` |
| `08_data_gauge/` | CONTACT_LABEL_DATA（86 KB）、GMR_QC、XWBC_ASSESSMENT、PUBLIC_RETARGETS、NEW_ROBOTS、TIMEBASE_FIX_meta、LCONV_meta、gauge_check_0903、h0_table、cl_v3 统计、方言 Q 拟合脚本、heldout 600 段名单 | 缓存目录 / `~/tmp` |
| `09_wbc_sim_recipes/` | BeyondMimic / BFM-Zero / SONIC 配方逐行复原 | `pose_translator_design/` |
| `10_literature/` | LIT_TEMPORAL_FSQ、LIT_MOTION、LIT_JEPA_COLLAPSE、LIT_V6_NOVELTY、LIT_WBC、三需求文献 | `pose_translator_design/`, `claude/` |
| `11_founding_docs/` | motionformer 研究愿景、架构讨论、数学框架、仓库 README | `note/`, `claude/` |
| `12_master_matrix_and_tables/` | spatial_matrix（每 run × 每指标）、LAFAN 地面/尺度表、AMASS 接触审计块、§12 地面修正验收 | scratchpad |
| `13_wbc_rl_scaling_0829/` | 地平线课程 49×38 表、actor 架构消融、Greenwich/Equator 侧 scaling csv、写作稿 | `~/tmp` |
| `14_product_docs/` | AlphaMotion 产品 docs（BENCHMARK/GAUGE/SDK/审计）、47 空间模型视觉门 PDF | `~/Desktop/AlphaMotion/docs` |
| `16_v02_handoff_0909.md` | **v0.2 交接说明**：Bumblebee/Megatron/Prime 传了什么、接口、四条纪律、git/HF 流程 | 本文件 |
| `15_se3_refiner_0904/` | 手部 SE(3) 钉点 + 精修器：README（最佳配置 vs 真值、零姿态对照臂、阶梯不帮软补全、吞吐基准）、sweep_pin.json 48 行、三份 pp0_feetpin report、脚本 | `~/tmp/ee_fill/` |

## image/

| 目录 | 内容 |
|---|---|
| `00_grand_eval_paper_figures/` (+`_dark`) | 0808 白皮书 12 张图（png+**svg 矢量**）：血统、JEPA、MaskFSQ、空间补全、时序泛化、率失真、动作流形、因果表征、速度控制、物理代理、数据审计；manifest.json 图→结论 |
| `01_scaling_ladder/` | 本轮阶梯 loss / scaling / chinchilla 三图（**0904 已覆盖为 S20k/M20k/L20k 三档版，带 eval 面板**）+ **E24 旧阶梯重绘**（只 LAFAN，中→大无收益） |
| `02_grid_S_vs_FZ5c/` | S / S0 同三段 12 帧网格（dance1、KIT 直走、kudo），对照 FZ5c；子目录 `L_{S20k,M20k,L20k}_0903/` 三档终版网格 |
| `03_grid_x2_zeroshot/` | X2 零样本 FZ5c(上) vs S(下) 12 帧网格 |
| `04_radar_ablation/` | 全 run 雷达总图（all / GR3 / Adam / 小版）+ per_run 68 张 + method.md 轴定义 |
| `05_showcase_grids_FZ5c/` | FZ5c 0902 展示网格：AMASS 10 段 + LAFAN 6 段 + 竖版 + 人机成对 |
| `06_tri_embodiment_ground/` | G1/GR3/Adam 地面动作三具身网格、pilot_decode 3×4 矩阵、X2/X3/Adam/GR3 单机网格、hpos 混合 IK、骨架对比 |
| `07_wbc_sim_rollout_grids/` | mjlab 仿真 A/B/native 网格、KIT 三轮、五动作族、脚滑 3 机×3 段、G1 足迹 |
| `08_gr3_getup_duel/` | GR3 起身 3 联 it1500 / final |
| `09_latent_analysis_R7/` | FSQ 潜空间：新颖度、身体 vs 动作 t-SNE、特征谱、PC 扰动、token 统计；R10 速率探针 |
| `10_stat_figures_V5_V7/` | 速度不变性、词表-时间、谱泄漏、内在维度、V7 家族能量/转移 |
| `11_hero_stills_4k/` | 3840×2160 carnival 全具身剧照 ×3、X2 SE3 三联 |
| `12_ee_ego_fill/` | 手部 SE(3) 脑补全身（EgoDex 煎蛋/面包/擦桌、UMI、真机 pick-place ± 钉脚）；`ladder_0904/` 真机 835 帧四格 demo（mp4 11 MB + 网格）+ 三档软补全网格 |
| `13_paper_figure_tex_sources/` | 论文架构图 tex 源 12 份 + appendix fig1–4 pdf |
| `14_misc_grids/` | 消融行、23 具身条带、A4 eval、gauge QC 总表、扰动响应 |

## 视频没有复制（太大）。位置：
`runs/L_{S20k,M20k,L20k}_0903/vid_final`（三档终版，上 FZ5c 下新模型）、`runs/L_S_0903/vid_final`、`runs/L_S0_lafan_0903/vid_final`、`runs/ladder_ref`（FZ5c 参考）、`~/tmp/x2phys/vid`、`~/tmp/gr3_mid`（GR3 41 段）、`~/tmp/wbcready_vid`、scratchpad `show/`、`cmp3/`、`cmp4/`、`vid5_*`、`~/tmp/ego_fill`（3 GB）、`~/tmp/ee_fill`、`deliverables/amass_zero_shot_mesh`。本目录上层 01–08 已是精选视频。

## 还会追加
阶梯三档 `_s1`（段 1 末）checkpoint 的 600 条审计（正在跑，回答"段 2 微调买没买到脚滑 / 该不该用段 1 末的存档"），出来后补进 `00_chinchilla_ladder_0904.md` §3。
