# 真实旧 run 与修订候选的逐项比较

依据 `reference_recipe/L_M20k_0903.config.json`：恢复出的原始字节与独立审计 SHA 一致。下表比较的是已恢复代码；真实数据与 CUDA 验收仍未完成，不能把离线测试当作可启动训练或模型效果。

| 项目 | 实际旧 run | 修订候选及有效性边界 |
|---|---|---|
| 旋转 / 位置 | geodesic_deg / 30 + 5 × FK/reach 欧式误差 | 保留；预测与 GT 都从全局旋转 FK，按 batch×joint 取平均；cached XYZ 另作 FK 校验 |
| InfoNCE | 0.5；τ=0.1；选 6 体，对 15 个无向体对做双向 CE 平均 | XWBC 真配对上恢复；仅在此损失内稳定去重相同 source row，避免假负例；UniMate 不适用 |
| 重编码一致性 | 1.0；概率 0.3；前半批至少 8 行；随机一个完整组合 | XWBC 恢复；无 mask/contact/SE3；回编码 predicted rot6 + raw hpos，原 latent 不 detach |
| 辅助 judge | 16 类 one-hot BCE；grad；0.1 | 原结构与模式/权重恢复；旧类别顺序未证实，整个头重置并保存当前验证类序；UniMate 不套用旧类别 |
| 穿透 | 200 × mean(relu(-h)²) | 恢复于可信 metric ground rows；h 用 FK_Y/100 + GT root gap + 旋转后 sole offset |
| 接触几何 | 1 × BCEWithLogits((thr−h)/(thr/3), contact_soft) | 恢复；不把它改成 hard-contact 维持项、不做接触数归一；每来源真实阈值/sole 经 seal 核对 |
| root height | 3 × SmoothL1(raw_hpos[root,0], GT gap_root_m) | 有效有限标签上保留；与接触项开关解耦；UniMate 未知则关闭，无假零标签 |
| 地面/足底输入 | 原 contact headers、sole_offsets、gap、soft/hard/trust | 第一版转存遗漏已补可执行工具；须对照原来源补齐并验收。缺文件拒绝，不猜 foot center；原始空映射仅在函数/头信息均证实时按原语义记录 |
| 姿态 mask | 每行启用概率 0.3；rand .25 / ee .15 / root .10 / sparse .25 / cask .15 / cgive .10 | XWBC 主分支恢复原采样函数；root 可见；sparse 从全体 joint 留 1–3 个再保留 root |
| contact 输入 | signed distance；dropout .3；有任意遮挡的行保留 contact | XWBC 主分支恢复；遮姿态不等于遮独立合法接触观测。纯 SE3 分支无源 contact |
| 主方向与任务 | 16×16 含 self，每步无放回选 8 个方向，共用 sel；主损失平均 | 保留；aux 不除以 8。XWBC 55% 步，另以 .25 概率追加同体 pure-SE3 分支，weight1。UniMate 45% 步仅同体 full/masked/sparse |
| 时间项 | world-slip 2、hpair .1；tdec=0 但用了相邻帧副批 | 明确关闭：稀疏选帧缺完整相邻帧/真实时间间隔/世界根运动；不能用表中相邻行或假 30 fps 顶替 |
| 原先关闭的项 | w_align、dof_loss、var_floor、w_contact、w_cbce、w_root 等为 0/false | 保持关闭；不从 CLI 默认值引入其他项 |
| 优化器 | AdamW lr1e-4，decay1e-4，cosine，clip1，无 warmup | 保留；撤销临时两档 LR 与 warmup |
| 结构 / 文本 | d256、8 heads、enc4/dec4、SigLIP768；双 FSQ | enc6/dec6、固定 Qwen1024 新投影、C≤2 共享 target-SE3 接口、显式 metadata missing；FSQ 与共享 decoder 保留 |
| 数据 / 预算 | 旧两缓存；batch256；8000 steps | 选定 source-group split 与同源去重，batch128，最多3000 steps/1800秒；真实 UniMate 通过全部帧 FK 后才入训 |

主模型 **25,743,428** 参数；训练辅助 judge **266,512**；合计 **26,009,940**。共享主干/FSQ 加载 375 个兼容旧键；4 个旧 semantic projection 键与 4 个旧 judge 键弃用并新建。不能将新增辅助分类头描述成目标 decoder 的机器人专用参数。

精确源码证据与各函数哈希见 `recipe_review.json`，实际测试结果见 `offline_verification.json`。启动必须有本版本、同配置、同真实数据、同运行环境的全损失 CUDA 验收 receipt；只有配置中的“ready”字段无效。训练、下载完整、内容校验、仿真、真机验证分别报告。
