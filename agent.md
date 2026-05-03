"Be water, my friend." --Bruce Lee

通用机器人全身控制器 — 研究 Syllabus

研究目标

构建一个单一模型，输入全身 joint time series（手/腰/腿），输出任意机械构型跨 embodiment 全身运动。


───

Phase 1 — 基础理论（读论文）

1.1 Foundation Models for Robotics
| 论文                              | 重点                                            |
| ------------------------------- | --------------------------------------------- |
| π0.5 (arxiv: 2410.24164)          | Flow Matching VLA 架构、跨 embodiment 训练 recipe   |
| SONIC (arxiv: 2511.07820)       | 运动追踪作为基础任务、universal token space              |
| BeyondMimic (arxiv: 2508.08241) | Diffusion guidance 零样本泛化                      |
| HPT (2024)                      | Shared trunk + embodiment-specific heads      |
| UniAct (2025)                   | Universal action space，跨 embodiment 连续 latent |

1.2 架构基础

| 论文                                | 重点                                           |
| --------------------------------- | -------------------------------------------- |
| AlphaFold2 / Evoformer            | Axial attention、triangle attention → 映射到肢体协同 |
| ACT (Action Chunking Transformer) | cVAE 实现、KL 权重调优                              |
| Diffusion Policy                  | Chunk size 选择、连续空间建模                         |
| Flow Matching (Lipman et al.)     | Flow matching vs diffusion 原理                |

1.3 全身控制专项

| 论文            | 重点                                |
| ------------- | --------------------------------- |
| HumanPlus     | 全身 26+ DOF 控制                     |
| OmniH2O       | 人形机器人全身遥操作                        |
| HOVER / CLoWR | 人形 VLA + whole-body（待补全 arxiv ID） |

───

Phase 2 — 架构设计
核心架构："MotionFormer"

• 输入：全身 joint time series [T × D_joints] + 视觉/语言指令
• Backbone：改造 Evoformer → [时间步 × 肢体] 双轴 attention
  • Row attention：同一时刻，肢体间协同（pair representation = 肢体协同矩阵）
  • Column attention：同一肢体，时序演化
  • Triangle attention：三肢体联动约束
• Action Head：Flow Matching decoder（参考 π0）
• Latent Space：连续 cVAE，dim=64~128，无离散量化

关键设计决策待确定

• [ ] Chunk size：全身控制建议 50~100 steps
• [ ] 是否用 VLM backbone（如 PaliGemma）
• [ ] Embodiment-specific head vs universal head
• [ ] Sim-to-real 策略
───

Phase 3 — 工程实现

硬件

• RTX Pro 6000 Blackwell 96GB VRAM（已分析可行）
• 推荐模型规模：Transformer 12层 hidden=512，BF16，batch ~32

数据

• Mocap 数据：参考 SONIC（700 小时）规模
• 示教数据：类 ACT 规模，500~2000 episodes 起步
• 数据增强：joint time series 的时间 warping + 噪声注入

───
Phase 4 — 实验验证

Baseline：单 embodiment，ACT 复现
消融实验：有无 triangle attention、有无 pair representation
跨 embodiment 迁移：同架构在不同机器人上 fine-tune
最终目标：章鱼博士硬件上的全身控制 demo

