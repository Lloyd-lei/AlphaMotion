# AlphaMotion-2 架构研究与工程审计交付

本包保存 AlphaMotion-1、UniMate、UMR 的源码审计、可编辑图稿、逐层计算附录，以及 AlphaMotion-2 候选模型与生产数据清洗管线方案。它是设计与代码审计交付，不是二代模型的性能复现实验或生产验收报告。

团队文档：https://docs.google.com/document/d/1gH9WaPSgqUcTdrPYeBCNaLkWrVoG0qzNsiJLPVsniBU/edit

## 建议阅读顺序

1. `figures/paper_01_comparison.pdf`：三个已实现系统的总体计算流程。
2. `figures/paper_02_alphamotion1.pdf`、`paper_03_unimate.pdf`、`paper_04_umr.pdf`：各模型逐层主图；相同文件名的 SVG 可编辑。
3. `appendix/architecture_layers.md`：40 个不同计算条目、张量形状、公式、有效配置、源码位置和伪代码。冻结文本塔的内部结构补充在 `tool-audit/text_towers.md`。
4. `figures/paper_05_alphamotion2_proposed.pdf` 与 `alphamotion2_research_design.md`：本轮候选架构和消融设计。
5. `engineering-architecture/工程审计说明.md`：预处理、缓存、训练、推理、评测和故障处理的代码架构。三个 `*_doc.pdf` 为可阅读紧凑工程图，相同名称的 SVG 可编辑；无 `_doc` 的文件保留完整大图。
6. `tool-audit/tools_and_surface_data.md`：可视化、测量、线上 demo 的真实执行范围与可复用模块。
7. `AlphaMotion-2-architecture-and-engineering.docx` 为在线团队文档的可编辑快照；`final-google-docs-review.pdf` 为其版面导出，包含原有页签。

## 研究方向

核心假设是：局部表面区域的文本语义与几何共同编码，可能形成可泛化到未见本体的对应。纯 surface 路径应单独验证；完整 AlphaMotion-2 必须联合 skeleton 和 surface 进行表示学习、跨本体对应与预训练。连续/FSQ/混合表示与 flow/diffusion 生成过程为独立设计轴。

SE(3)、接触、灵巧手和未来触觉接口保留在方案中；不存在的资产或标签用有效性掩码表示。共享运动学表示与本体条件的动力学、控制验证分别记录。

## 实现边界

- AlphaMotion-1 图稿依据实际 megatron 配置与权重形状，不以构造函数默认值替代有效配置。
- UniMate 图稿依据本次固定提交的默认 Graph AdaLN 配方；其他变体在附录单列。
- UMR 的静态表面对应网络与逐动作受约束求解分开绘制。
- AlphaMotion-2 图和生产清洗图为 proposed；宽度、深度、损失权重和生成器选择未定。
- 本轮下载与基础工程由 ArenaLabs 上独立任务维护。已有基础测试不构成生产验收；资源表随下载阶段更新。
- 此包不包含数百 GB 的数据/权重。全量公共资源仍由可恢复下载器和定期检查任务管理。

## 可追溯性

`figures/sources.md` 将图中 A/U/R 编号映射到源文件。`source-manifest.json` 保留源路径与 SHA256，`source-integrity-check.json` 验证 404 个快照文件。大小写冲突的两个文件保存在独立路径，读取时优先使用 manifest 的 `snapshot_rel`；其他记录为 `source-snapshot/<tag>/<rel>`。

`doc-structural-qa.json` 记录在线页签、图片与引用数量；`qa/final_docs_visual_review.json` 记录编辑页的视觉检查。独立矢量图的质量检查保存在各图稿目录。

