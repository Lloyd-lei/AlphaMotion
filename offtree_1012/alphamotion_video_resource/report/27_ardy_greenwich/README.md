# 27 · ARDY(G1) → Megatron → G1 / H2 / GR3 流水线探针 0923

四联视频：ARDY 生成的 G1 源 | Megatron 自解码 G1 | H2（组员 43 节点骨架）| GR3。H2/GR3 的根：ARDY 根 xz 按腿长比透传，高度用解码槽；显示时按每帧实测最低碰撞几何把根抬离地面（抬升量写在标题和 result.json 里，GR3 没抬，它本来就悬空）。
输入：mingrun 基准的 3 段 ARDY G1 输出 + 本次用 ARDY 文本生成的 4 段（p1 走路左转、p2 右手挥手、p3 蹲下捡东西、p4 向前跳；--model g1 --duration 6 --seed 0）。
Megatron 编码走接触通道（ARDY 的 foot_contacts → ankle_roll/toe 节点）；无接触通道的对照臂数字也在 results_all.json（nocon）。
考卷：arena locomotion.v1（evaluate_clip），零样本口径：站姿标签来自 ARDY，接触阈 0.04×站立根高，无 GT 尺子的门标 not_evaluated。
脚本 ardy_in.py（ARDY 34 关节 → 我方 36 节点 G1，FK 核对 0.66 cm 平均、末端站点除外）、run_pipeline.py。
