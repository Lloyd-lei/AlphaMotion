# 26 · G1→H2 vs SOMA→H2（Megatron，潜空间 + 观测空间）0921

三联视频：左 = BoneSeed 官方 G1（源）；中 = 组员 SOMA→H2（Prime 产物，43 节点 H2 骨架）；右 = G1→H2（Megatron 编码 G1，解码到同一 43 节点 H2 骨架，再投影到关节；根为显示抬离地面，抬升量见 results.json）。
注意：SOMA 源动作本机没有（45 GB 门控包未下载），"SOMA→H2" 用的是组员 Prime 产物再经 Megatron 重编码，模型与源两个变量混在一起。
朝向：组员 H2 与官方 G1 首帧朝向差 63～108°（每段不同），已按首帧对齐后再编码；codec 不是 yaw 不变的。
潜空间：132 槽 × 13 维整数码逐槽一致率 + 连续 z 余弦；地板 = 同段错开 1 s。
数字见 results.json；脚本 compare.py / stack.py。
