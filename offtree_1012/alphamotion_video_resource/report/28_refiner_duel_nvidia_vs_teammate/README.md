# 28 · 精修器对决：NVIDIA soma-retargeter v0.2 vs 组员 alphamotion-locomotion-toolkit（Prime + 100+100 精修）0923

源：NVIDIA 仓库自带的 10 段 BONES-SEED SOMA BVH（走、投球、挥手、拉伸、高跳、两段舞、三段捡东西）。
- NVIDIA 臂：仓库自带的 G1 / H2 参考 CSV（他们自己跑出来的）。
- 组员臂：本机用 sajio1/alphamotion-locomotion-toolkit（2026-09-23 HEAD）+ Prime 权重跑同样 10 段 → G1 / H2。公开快照有三处缺口我打了补丁才跑通：缺 NativeSurfaceContacts 类（只做表面接触上色，stub 掉）、runner 引用清单里不存在的 foot_landmark_heights_cm 字段、roster 必须带 palm_frames（用了组员自己 roster 里的 G1/H2 标定）。
- 对照：BoneSeed 官方 G1（Bones 自己的重定向）。
一把尺子（ruler.py）：厂商 MJCF 上 MuJoCo FK，每只脚碰撞几何最低点；站立相 = 最低点离地 < 1 cm 且前后各 2 帧都成立；滑移 = 站立相脚身原点水平速度；穿地 = 最低点低于 0；关节 jerk、限位超出。30 fps。
数字：ruler_results.json。视频：*_duel.mp4（G1：NVIDIA | 官方 | 组员；H2：NVIDIA | 组员），*_grid.jpg 12 帧网格。组员输出：teammate_outputs/。
