# claude/ — 研究员产出文档索引

这些文档是lloyd和claude在Mac侧对话中产出的架构设计文档，写入时间2026-07-29/30。

- **math_framework.md** — 数学框架推导：SOMA-X embodiment统一、IK冗余度形式化、
  OmniControl式(joint,time)约束掩码、多模态cross-attention、SE(3) flow matching。
  已经过一轮引用措辞修正（v2，version_id ebc15595-cb4f-4b47-94ec-490e6dbc5ba1）。
- **architecture_spec.md** — T型架构第一版定案（早期版本）。
  状态：已被architecture_final_design.md取代，保留仅供追溯设计演变过程，
  不作为当前实现依据。
- **architecture_final_design.md** — 当前最新的T型架构定案：spatial encoder-decoder
  (跨具身retargeting，frame-based IPA decoder) + temporal causal transformer；
  含产品功能到架构模块对照表、待消融数字清单。当前实现应以此文档为准
  （version_id c8d2ae99-47c7-4186-ba76-81956422acfb）。
- **ablation_experiment_spec.md** — spatial层瓶颈模块消融实验方案：
  AE/VAE/CVAE/VQVAE/RVQ/FSQ六变体对比，SE3精确控制+跨本体动作相似度两个评估维度，
  loss设计对齐产品UX的具体方案。这是当前建议的第一个实现任务的规格来源
  （v3，version_id c8a4128d-ab39-42fd-9871-a2ef1359a17c）。

## 已知问题记录（供codex参考，避免重复踩坑）

ablation_experiment_spec.md早期版本中曾有一处编造的具体数字，已被审计发现并修正：

1. 曾错误写成"Kimodo-SOMA-RP-v1.1，G1侧34关节，SMPLX侧22关节"——这是编造的数据集
   版本名和关节数，已从文档中删除。真实核实过的事实只有：SOMA骨架到Unitree G1
   （29自由度），NVIDIA SOMA-X/soma-retargeter官方发布，其他机器人目标"计划中"
   尚未发布。**代码实现请务必以bones_seed_full/g1/csv实际文件的列数
   （36列 = 6自由度root + 29关节dof，已在claude侧实测确认）为准，不要相信文档里
   任何未标注来源的具体关节数字。**

2. SAMoR论文(arXiv 2607.02148)的loss列表、K=8 part-token消融结论、CLIP文本塔
   用法等细节，来自claude对该论文PDF的直接视觉阅读，非凭空编造。如果codex要
   复核这些细节，建议直接重新拉取该论文本身核对。

## 与alpha_motion2现有Stage-1代码的关系（待办）

这几份文档是在不了解alpha_motion2/experiment/stage1现有代码和真实实验结果的
情况下独立推导的。Stage-1的triangle attention消融是真实实验结论（full vs
opm_only在AMASS heldout上reconstruction loss几乎一致0.122 vs 0.121，但
pair-tensor几何r=0.077几乎正交），这应作为硬约束去修正/校验本文件夹里的
纸面设计，尚未完成这一步对照。

