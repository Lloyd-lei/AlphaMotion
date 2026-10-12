# HiPHI 数据丰富度盘点(只读统计)

生成时间 2026-10-05 00:00。数据源: HiPHI (Noitom Robotics, CoRL 2026, arXiv 2608.16222)。

## 一句话结论

- HiPHI 原片 **308.73 h**(加镜像 617.46 h), 17626 条, 132 名演员, 214 个动作词条。
- 结构很偏: **走路类 + 搬推物体** 占大头。按主类分: 移动步态 107.78 h(34.9%), 全身搬/推/踢物体 73.97 h(24.0%), 上肢操作物体 43.61 h(14.1%)。
- 稀缺的: 地面接触只有 **8.11 h**(2.6%), 坐物体 1.98 h, 具名舞蹈 3.18 h, 武术 1.43 h, 体操特技 0.77 h, 摔倒 1.01 h, 起身 1.86 h。没有楼梯/斜坡(全是平地)。
- 物体: 每条 HOI 只有 1 个物体, 12 类, 椅子/桌子/箱子 三类占 HOI 的 85.4%。桌子是 60 cm 左右的小边桌, 不是大桌。
- 进度: 32 个压缩包只到手 2 个完整(0002, 0029) + 0001 在下(状态快照 7.66/7.71 GB); 已转 SMPL-X **1094 条 / 19.14 h 原片**(占 6.2%), 镜像 0 条, **重定向到机器人 0 条**。
- 已转的 19.1 h **全是 HOI**(搬推物体), 纯身体动作(走路/舞蹈/武术/地面/手势)一条都还没有: 下载顺序是按 HOI 优先排的。
- 想先拿到多样性: **part 0032 只有 2.6 GB, 却装着几乎全部舞蹈(2.94/3.18 h)、全部武术(1.43/1.43 h)和大部分体操特技(0.55/0.77 h)**; part 0027 + 0030 装着地面接触的 30%(2.41/8.11 h)。建议把这三个包提前。
- 许可: 非商业(ModalityNet Open Research License), 禁止再分发和公开镜像, 禁止拿来做替代 HiPHI 的商用动作生成模型。商用要单独找 licensing@modalitynet.com。

## 1. 总量

| 项 | 数 |
|---|---:|
| 原片时长 | 308.73 h (17626 条) |
| 镜像时长 | 308.73 h (17626 条, 左右镜像, 每条原片一条) |
| 合计 | 617.46 h (官方 617.5 h) |
| HOI(带物体) 原片 | 122.85 h (7025 条) |
| 纯身体 原片 | 185.88 h |
| 帧率 | 90 Hz (实测 90.0–90.009) |
| 单条长度 | 中位 61.6 s, 最短 14.6 s, 最长 309.1 s |
| 语义 | 22 个 Frame, 214 个 Frame-LU, 3832 条不重复的英文描述 |
| 格式 | 55-joint BVH (Y-up, cm, ZXY Euler) + HOI object tracks (CSV, m, xyzw) + 40 OBJ meshes (+40 mirrored) |
| 许可 | ModalityNet Open Research License v1.0 — NON-COMMERCIAL ONLY; no redistribution/public mirrors; no competing dataset or commercial-substitute motion generator; commercial license via licensing@modalitynet.com |

## 2. 九大类时长(原片 / 加镜像)

多标签 = 一条片子可以同时属于几类(比如边走边搬箱子 = 移动步态 + 全身搬物体), 所以加起来超过 100%。主类 = 按优先级 地面接触 > 高动态 > 坐物体 > 全身物体 > 上肢物体 > 移动 > 手势 > 姿态 > 其他 只取一个, 加起来正好 308.7 h。

| 类 | 多标签 原片 h | 加镜像 h | 占比 | 主类 原片 h | 主类占比 | 条数(多标签) | 演员数 | 身高范围 cm | 男/女 h | 其中 HOI h | 已转 SMPL-X h | 下载中(0001) h | 未下载 h | 重定向 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| locomotion 移动步态 | 174.42 | 348.83 | 56.5% | 107.78 | 34.9% | 10015 | 113 | 155–185 | 114.95/59.47 | 57.14 | 6.95 | 2.72 | 164.74 | 0 |
| dynamic 高动态 | 21.24 | 42.49 | 6.9% | 18.21 | 5.9% | 1112 | 106 | 157–185 | 13.55/7.69 | 3.07 | 0.17 | 0.24 | 20.83 | 0 |
| ground_contact 地面接触 | 8.11 | 16.22 | 2.6% | 8.11 | 2.6% | 478 | 85 | 158–185 | 5.4/2.71 | 0.17 | 0.09 | 0.02 | 8.01 | 0 |
| sitting_on_object 坐物体 | 1.98 | 3.96 | 0.6% | 1.96 | 0.6% | 115 | 15 | 160–183 | 1.12/0.87 | 1.98 | 0.41 | 0.22 | 1.35 | 0 |
| upper_body_manipulation 上肢操作物体 | 44.02 | 88.03 | 14.3% | 43.61 | 14.1% | 2539 | 54 | 158–185 | 31.34/12.68 | 44.02 | 8.12 | 4.84 | 31.06 | 0 |
| whole_body_object 全身搬/推/踢物体 | 77.07 | 154.15 | 25.0% | 73.97 | 24.0% | 4384 | 56 | 158–185 | 51.85/25.22 | 77.07 | 10.63 | 4.56 | 61.88 | 0 |
| gesture_social 手势/社交/情绪 | 16.87 | 33.73 | 5.5% | 12.01 | 3.9% | 969 | 95 | 158–185 | 11.36/5.5 | 0.0 | 0 | 0 | 16.87 | 0 |
| idle_posture 站姿/姿态/拉伸 | 17.38 | 34.75 | 5.6% | 10.41 | 3.4% | 995 | 103 | 155–185 | 10.98/6.39 | 0.0 | 0 | 0 | 17.38 | 0 |
| other 其他 | 32.65 | 65.31 | 10.6% | 32.65 | 10.6% | 1893 | 99 | 158–185 | 20.64/12.02 | 0.0 | 0 | 0 | 32.65 | 0 |

各类最大的词条(多标签口径):

- **移动步态**: Bringing-carry 8.31h, Bringing-lug 6.99h, Cause_motion-move 6.82h, Bringing-take 5.45h, Bringing-hump 4.77h, Cause_motion-pull 3.19h
- **高动态**: Self_motion-sashay 2.94h, Self_motion-caper 2.53h, Self_motion-jump 2.47h, Cause_motion-punt 2.39h, Self_motion-hop 2.03h, Self_motion-frolic 1.45h
- **地面接触**: Posture-kneel 0.93h, Change_posture-kneel_down 0.75h, Posture-shrink 0.69h, Body_movement-kneel 0.65h, Posture-lie 0.64h, Self_motion-vault 0.52h
- **坐物体**: Posture-sit 0.92h, Change_posture-stand_up 0.36h, Change_posture-sit_down 0.26h, Change_posture-lean 0.22h, Change_posture-sit_up 0.17h, Change_posture-sprawl 0.05h
- **上肢操作物体**: Adjusting-tweak 4.73h, Adjusting-adjust 4.06h, Cause_to_move_in_place-rotate 3.2h, Cause_to_move_in_place-shake 2.91h, Cause_to_move_in_place-swing 2.82h, Cause_to_move_in_place-rock 2.73h
- **全身搬/推/踢物体**: Bringing-carry 8.19h, Cause_motion-move 7.91h, Bringing-lug 7.64h, Bringing-take 6.81h, Bringing-hump 5.42h, Cause_motion-pull 4.59h
- **手势/社交/情绪**: Body_movement-clap 2.55h, Body_movement-wave 2.31h, Gesture-gesture 2.22h, Gesture-gesticulate 2.17h, Body_movement-shrug 1.62h, Body_movement-shiver 1.53h
- **站姿/姿态/拉伸**: Body_movement-bend 2.98h, Body_movement-flex 1.33h, Change_posture-hunch_up 1.03h, Body_movement-arch 1.0h, Body_movement-reach 0.95h, Change_posture-crouch_down 0.94h
- **其他**: Body_movement-bob 4.28h, Body_movement-shake 2.04h, Body_movement-swing 2.0h, Manipulation-paw 1.97h, Body_movement-twitch 1.59h, Body_movement-twist 1.53h

## 3. 动作细项(子标签, 原片)

| 细项 | 原片 h | 加镜像 h | 条数 |
|---|---:|---:|---:|
| 转弯/绕圈/8字 (`turn_curve`) | 50.29 | 100.57 | 2885 |
| 原地局部身体动作(摇头/甩手/扭胯等) (`in_place_body_isolation`) | 48.38 | 96.75 | 2798 |
| 侧移步 (`sidestep`) | 36.7 | 73.4 | 2112 |
| 倒退走 (`backward`) | 28.82 | 57.63 | 1654 |
| 跳/单脚跳/蹦 (`jump_hop`) | 10.92 | 21.84 | 635 |
| 跑 (`run`) | 5.4 | 10.8 | 319 |
| 踉跄/绊脚(扰动步态) (`stumble`) | 3.86 | 7.72 | 221 |
| 踢(含踢物体) (`kick`) | 3.47 | 6.94 | 211 |
| 具名舞蹈 (`dance`) | 3.18 | 6.36 | 60 |
| 踢箱子/球 (`kick_object`) | 3.1 | 6.2 | 179 |
| 情绪(发抖/畏缩) (`emote`) | 2.43 | 4.87 | 138 |
| 从地上起身 (`getup`) | 1.86 | 3.72 | 109 |
| 武术/格斗 (`martial`) | 1.43 | 2.86 | 58 |
| 原地踏步 (`in_place_stepping`) | 1.09 | 2.18 | 64 |
| 摔倒/倒地 (`fall`) | 1.01 | 2.02 | 63 |
| 体操/特技(空翻/侧手翻/倒立) (`acrobatic`) | 0.77 | 1.55 | 70 |
| 跌落成低蹲 (`drop_to_crouch`) | 0.48 | 0.96 | 28 |
| 踏上/跨过长凳(类台阶) (`platform_step`) | 0.34 | 0.68 | 20 |
| 脚踩物体 (`step_onto_object`) | 0.34 | 0.68 | 20 |

说明: 跑步只有 5.4 h, 而且多是来回折返跑/8 字跑, 没有长距离直线跑。楼梯/斜坡/路沿 = 0。唯一的'上台阶'类是踏上/跨过长凳 0.34 h。

## 4. 接触类型(原片)

| 接触 | 原片 h | 加镜像 h | 条数 |
|---|---:|---:|---:|
| 地面只有脚(可能同时手碰物体) (`feet_ground`) | 300.62 | 601.24 | 17148 |
| 手-物体 (`object_hands`) | 118.9 | 237.8 | 6796 |
| 躯干/肩/臂-物体 (`object_body`) | 9.36 | 18.72 | 526 |
| 膝盖着地 (`knees`) | 4.02 | 8.03 | 230 |
| 脚-物体(踢/踩) (`object_feet`) | 3.84 | 7.68 | 222 |
| 臀-物体(坐) (`object_hips`) | 1.98 | 3.96 | 115 |
| 臀着地(坐地) (`hips_ground`) | 1.47 | 2.95 | 83 |
| 手撑地 (`hands_ground`) | 1.32 | 2.65 | 95 |
| 仰躺 (`back`) | 0.95 | 1.91 | 53 |
| 俯趴 (`front`) | 0.48 | 0.96 | 29 |
| 倒地(部位未写明) (`body_ground_unspecified`) | 0.41 | 0.82 | 24 |
| 侧躺 (`side`) | 0.2 | 0.4 | 12 |
| 严格'只有脚着地且不碰物体' | 177.94 | 355.88 | |

接触是从文字描述推出来的, 不是从动作数据算的。'深蹲贴地走'这种只算脚着地。

## 5. 地面接触明细

合计 **8.11 h 原片 / 16.22 h 加镜像**, 478 条, 85 名演员。已转 SMPL-X 只有 0.09 h。

| 着地部位 | 原片 h |
|---|---:|
| 膝盖着地 | 4.02 |
| 手撑地 | 1.32 |
| 臀着地(坐地) | 1.47 |
| 仰躺 | 0.95 |
| 俯趴 | 0.48 |
| 侧躺 | 0.2 |
| 倒地(部位未写明) | 0.41 |

来源词条(原片 h): Posture-kneel 0.93, Change_posture-kneel_down 0.75, Posture-shrink 0.69, Body_movement-kneel 0.65, Posture-lie 0.64, Self_motion-vault 0.52, Motion_directional-fall 0.42, Motion_directional-topple 0.34, Self_motion-scramble 0.3, Body_movement-wiggle 0.26, Change_posture-stand_up 0.25, Body_movement-wriggle 0.25, Change_posture-sit_down 0.24, Posture-sprawl 0.24, Getting_up-get_up 0.19, Motion_directional-plunge 0.17, Body_movement-cross 0.17, Self_motion-crawl 0.17, Change_posture-lie_down 0.16, Body_movement-arch 0.15, Change_posture-kneel_up 0.14, Body_movement-flutter 0.12, Self_motion-burrow 0.09, Change_posture-lean 0.09, Change_posture-sprawl 0.08

注意: 下载脚本 hiphi_part_priority.csv 里的 gc_h 用的是另一套口径(数值不同, 例如 part 0001 记 0.124 h, 本表按文字只认出 0.02 h), 以本表为准时请说明口径。

## 6. HOI 按物体(原片)

| 物体 | 条数 | 原片 h | 加镜像 h | 演员数 | 词条数 | 主类拆分 h | 已转 h |
|---|---:|---:|---:|---:|---:|---|---:|
| chair | 2589 | 45.57 | 91.14 | 38 | 34 | 全身搬/推/踢物体 31.72, 上肢操作物体 12.47, 坐物体 1.31, 地面接触 0.07 | 7.26 |
| table | 1701 | 29.89 | 59.79 | 35 | 31 | 全身搬/推/踢物体 20.33, 上肢操作物体 9.5, 坐物体 0.05, 地面接触 0.02 | 5.43 |
| box | 1685 | 29.41 | 58.81 | 45 | 56 | 全身搬/推/踢物体 17.65, 上肢操作物体 9.62, 高动态 1.71, 坐物体 0.38, 地面接触 0.03 | 2.79 |
| ball | 278 | 4.71 | 9.42 | 20 | 17 | 上肢操作物体 3.47, 高动态 0.84, 全身搬/推/踢物体 0.4 | 1.12 |
| bucket | 273 | 4.63 | 9.26 | 20 | 16 | 上肢操作物体 4.61, 地面接触 0.02 | 1.43 |
| stepstool | 117 | 2.03 | 4.06 | 9 | 9 | 全身搬/推/踢物体 1.57, 上肢操作物体 0.45 | 0.24 |
| clothrack | 110 | 1.9 | 3.79 | 12 | 5 | 全身搬/推/踢物体 0.97, 上肢操作物体 0.91, 地面接触 0.02 | 0.53 |
| bench | 94 | 1.62 | 3.23 | 16 | 5 | 全身搬/推/踢物体 0.93, 上肢操作物体 0.38, 坐物体 0.22, 移动步态 0.08 | 0.27 |
| mop | 86 | 1.51 | 3.03 | 4 | 17 | 上肢操作物体 1.51 | 0.04 |
| soccerball | 37 | 0.66 | 1.31 | 2 | 2 | 高动态 0.5, 上肢操作物体 0.16 | 0.0 |
| trashbin | 31 | 0.52 | 1.04 | 6 | 5 | 全身搬/推/踢物体 0.4, 上肢操作物体 0.1, 地面接触 0.02 | 0.0 |
| bottle | 24 | 0.41 | 0.81 | 6 | 2 | 上肢操作物体 0.41 | 0.04 |

物体尺寸(网格包围盒): 球 22–24 cm; 箱子 30–72 cm(H/F 型 30–37 cm, A/I/K 型 58–72 cm); 椅子 40–94 cm; 桌子 ~60×50×60 cm 小边桌; 长凳 61 cm; 踏凳 28–66 cm; 衣架 125–148 cm 高; 拖把 137 cm; 水桶 30 cm; 瓶子 34 cm。全部 40 个网格 + 40 个镜像网格都在本地 object_meshes/。
全身/上肢的分界规则: 大物体(椅/桌/箱/踏凳/长凳/衣架/垃圾桶)被搬、推、拉、拖、翻倒, 或者用脚踢球/箱, 算全身; 原地转、摇、晃、倒空、扔接球、拖把、水桶、瓶子算上肢。

## 7. 演员多样性

- 132 人: 男 76, 女 56。身高 155–185 cm(中位 170.0, 男中位 175.0, 女中位 163.0), 体重 40–82 kg(中位 56.25)。
- 时长按性别: 男 203.14 h, 女 105.59 h(女占 34.2%)。
- 每人时长: 最少 0.02 h, 中位 1.22 h, 最多 15.05 h; 前 10 人贡献 32.5%。
- 每个词条中位 23.5 人; 154 个词条 ≥10 人, 19 个词条 ≤3 人(多是小 HOI 词条, 如 Cause_motion-draw/haul 只有 1 人)。
- HOI 类演员明显更少: 全身物体 56 人, 上肢物体 54 人, 坐物体 15 人; 拖把只有 4 人, 足球 2 人。纯身体类 85–113 人。

## 8. 下载 / 转换 / 重定向 状态

- 压缩包: 32 个。完整在盘: 0002, 0029; 正在下: 0001 (7.66/7.71 GB); 已解压: 0002(0029 解压目录已清理)。
- 下载器快照 2026-10-04 23:59:59: 速度约 0.663 MB/s, 还剩 192.58 GB, 估计 80.74 h 下完, 但脚本截止时间只剩 34.8 h —— **按现在速度下不完**。
- SMPL-X(CPU 管线): 1094 条原片 = 19.14 h, 日志 1094 条全部 ok; 镜像转了 0 条。目录 /home/arenalabs/umr_factory_data/smplx/hiphi (3.8 GB; smplx_gpu/hiphi does not exist; GPU re-run lives in /home/arenalabs/gpu_convert/runs/bench4/hiphi)。
- GPU 转换管线在 bench4 上重跑了同样的 1094 条, 统计时 parity.py 正在对比 GPU 和 CPU 结果。
- 重定向到机器人: **0 条, 0 台**(out/ 与 runs/ 里没有 HiPHI 产物)。

状态按原片时长: converted_smplx 19.14 h (1094 条), downloading 9.58 h (543 条), not_downloaded 280.01 h (15989 条)

### 每个压缩包装了什么(原片 h, 按主类)

| 下载序 | part | 状态 | 压缩 GB | 原片 h | HOI h | 移动步态 | 高动态 | 地面接触 | 全身搬/推/踢物体 | 上肢操作物体 | 坐物体 | 手势/社交/情绪 | 站姿/姿态/拉伸 | 其他 | 已转条数 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0001 | 下载中 | 7.71 | 9.58 | 9.58 | 0.00 | 0.24 | 0.02 | 4.32 | 4.79 | 0.22 | 0.00 | 0.00 | 0.00 | 0 |
| 2 | 0002 | 在盘+已转 | 7.72 | 9.57 | 9.57 | 0.00 | 0.14 | 0.05 | 4.69 | 4.39 | 0.31 | 0.00 | 0.00 | 0.00 | 546 |
| 3 | 0029 | 在盘+已转 | 7.74 | 9.57 | 9.57 | 0.00 | 0.03 | 0.03 | 5.74 | 3.65 | 0.11 | 0.00 | 0.00 | 0.00 | 548 |
| 4 | 0021 | 未下 | 7.71 | 9.56 | 9.56 | 0.05 | 0.00 | 0.00 | 7.83 | 1.57 | 0.12 | 0.00 | 0.00 | 0.00 | 0 |
| 5 | 0003 | 未下 | 7.8 | 9.55 | 9.55 | 0.00 | 0.88 | 0.00 | 3.75 | 4.46 | 0.46 | 0.00 | 0.00 | 0.00 | 0 |
| 6 | 0024 | 未下 | 7.71 | 9.55 | 9.55 | 0.03 | 0.16 | 0.00 | 7.03 | 2.31 | 0.02 | 0.00 | 0.00 | 0.00 | 0 |
| 7 | 0022 | 未下 | 7.73 | 9.55 | 9.55 | 0.00 | 0.04 | 0.00 | 8.37 | 1.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 8 | 0023 | 未下 | 7.74 | 9.54 | 9.54 | 0.00 | 0.18 | 0.00 | 6.49 | 2.76 | 0.12 | 0.00 | 0.00 | 0.00 | 0 |
| 9 | 0025 | 未下 | 7.71 | 9.54 | 9.54 | 0.00 | 0.07 | 0.02 | 6.07 | 3.33 | 0.05 | 0.00 | 0.00 | 0.00 | 0 |
| 10 | 0004 | 未下 | 7.77 | 9.53 | 9.53 | 0.00 | 0.83 | 0.02 | 4.08 | 4.37 | 0.23 | 0.00 | 0.00 | 0.00 | 0 |
| 11 | 0005 | 未下 | 7.39 | 9.65 | 7.25 | 0.60 | 0.41 | 0.07 | 3.09 | 3.61 | 0.15 | 0.52 | 0.29 | 0.90 | 0 |
| 12 | 0020 | 未下 | 7.09 | 9.65 | 6.45 | 1.28 | 0.94 | 0.44 | 4.64 | 1.74 | 0.05 | 0.00 | 0.22 | 0.34 | 0 |
| 13 | 0028 | 未下 | 7.33 | 9.69 | 5.03 | 1.94 | 0.19 | 0.62 | 1.81 | 3.13 | 0.10 | 0.35 | 0.70 | 0.85 | 0 |
| 14 | 0030 | 未下 | 6.82 | 9.8 | 2.97 | 2.47 | 0.22 | 1.31 | 1.40 | 1.45 | 0.04 | 0.68 | 0.73 | 1.50 | 0 |
| 15 | 0026 | 未下 | 7.5 | 9.76 | 2.93 | 3.81 | 0.52 | 0.50 | 2.39 | 0.53 | 0.00 | 0.46 | 0.51 | 1.03 | 0 |
| 16 | 0031 | 未下 | 6.7 | 9.79 | 2.56 | 2.35 | 0.28 | 0.37 | 2.24 | 0.32 | 0.00 | 0.81 | 0.88 | 2.55 | 0 |
| 17 | 0027 | 未下 | 7.37 | 9.82 | 0.0 | 4.50 | 0.92 | 1.10 | 0.00 | 0.00 | 0.00 | 0.48 | 1.34 | 1.46 | 0 |
| 18 | 0010 | 未下 | 6.04 | 9.93 | 0.0 | 6.37 | 0.59 | 0.29 | 0.00 | 0.00 | 0.00 | 0.54 | 0.38 | 1.76 | 0 |
| 19 | 0011 | 未下 | 6.11 | 9.93 | 0.0 | 7.26 | 0.41 | 0.29 | 0.00 | 0.00 | 0.00 | 0.31 | 0.37 | 1.29 | 0 |
| 20 | 0012 | 未下 | 6.07 | 9.95 | 0.0 | 6.81 | 0.64 | 0.26 | 0.00 | 0.00 | 0.00 | 0.37 | 0.33 | 1.54 | 0 |
| 21 | 0008 | 未下 | 6.08 | 9.92 | 0.0 | 5.96 | 0.43 | 0.27 | 0.00 | 0.00 | 0.00 | 0.75 | 0.49 | 2.02 | 0 |
| 22 | 0018 | 未下 | 6.02 | 9.96 | 0.0 | 7.16 | 0.24 | 0.21 | 0.00 | 0.00 | 0.00 | 0.45 | 0.34 | 1.57 | 0 |
| 23 | 0019 | 未下 | 6.05 | 9.94 | 0.05 | 6.43 | 1.32 | 0.24 | 0.02 | 0.02 | 0.00 | 0.15 | 0.43 | 1.33 | 0 |
| 24 | 0014 | 未下 | 6.07 | 9.93 | 0.0 | 7.00 | 0.66 | 0.21 | 0.00 | 0.00 | 0.00 | 0.45 | 0.24 | 1.37 | 0 |
| 25 | 0032 | 未下 | 2.61 | 5.94 | 0.07 | 0.33 | 4.54 | 0.50 | 0.03 | 0.04 | 0.00 | 0.17 | 0.15 | 0.17 | 0 |
| 26 | 0017 | 未下 | 6.06 | 9.95 | 0.0 | 7.37 | 0.42 | 0.26 | 0.00 | 0.00 | 0.00 | 0.48 | 0.27 | 1.16 | 0 |
| 27 | 0015 | 未下 | 6.02 | 9.93 | 0.0 | 7.17 | 0.51 | 0.18 | 0.00 | 0.00 | 0.00 | 0.56 | 0.27 | 1.24 | 0 |
| 28 | 0013 | 未下 | 6.04 | 9.94 | 0.0 | 6.97 | 0.70 | 0.19 | 0.00 | 0.00 | 0.00 | 0.55 | 0.24 | 1.30 | 0 |
| 29 | 0007 | 未下 | 6.1 | 9.94 | 0.0 | 4.88 | 0.41 | 0.24 | 0.00 | 0.00 | 0.00 | 1.29 | 0.51 | 2.61 | 0 |
| 30 | 0006 | 未下 | 6.0 | 9.94 | 0.0 | 3.67 | 0.20 | 0.15 | 0.00 | 0.00 | 0.00 | 1.54 | 0.85 | 3.53 | 0 |
| 31 | 0009 | 未下 | 6.05 | 9.9 | 0.0 | 6.35 | 0.51 | 0.14 | 0.00 | 0.00 | 0.00 | 0.55 | 0.55 | 1.80 | 0 |
| 32 | 0016 | 未下 | 6.05 | 9.95 | 0.0 | 7.01 | 0.58 | 0.14 | 0.00 | 0.00 | 0.00 | 0.56 | 0.31 | 1.35 | 0 |

读法: 下载序 1–10 的包几乎 100% 是 HOI; 走路/手势/姿态 集中在 0006–0019(每包 ~10 h 纯身体, 其中 ~7 h 走路); 舞蹈+武术+特技 几乎全在 0032; 地面接触最多的是 0030 和 0027。

## 9. 有歧义的词条(分类时要看文字, 不能只看词条名)

| 词条 | 说明 |
|---|---|
| Body_movement-swing | 混合(162 条): 43 条是武术/器械套路名(Tornado kick/Tai Chi/剑/枪/棍…, 0.62 h)→dynamic(martial); 116 条是单腿/单臂左右摆(2.0 h)→other(原地局部动作) |
| Self_motion-sashay | 混合(156 条): 46 条是具名舞蹈(Samba/KPOP/古典/民族/爵士…, 2.94 h, 单条最长 309 s)→dynamic(dance); 110 条是扭胯走步(1.95 h)→locomotion |
| Self_motion-vault | 70 条全是体操/特技/武术动作名(后空翻/侧手翻/倒立/滚翻/倒地/旋子): 67 条 acrobatic, 其中 46 条有手撑地/倒地→主类 ground_contact; 3 条单腿平衡→idle_posture |
| Cause_motion-hit | 混合(48 条): 15 条徒手格斗(Boxing/Muay Thai/Sanda/Combat, 无物体, 0.82 h)→dynamic(martial); 15 条用脚背踢箱子→whole_body_object+kick; 18 条手掌/前臂拍箱子→upper_body_manipulation |
| Self_motion-step | 混合(86 条): 20 条踏上/跨过/沿长凳顶面走(物体-脚接触, 类台阶 0.34 h); 19 条原地踏步; 其余推桌/推箱/扶凳走 |
| Self_motion-trip | 绊脚踉跄(扰动步态)→归 locomotion, 子标签 stumble; 不算 fall |
| Self_motion-totter | 摇晃欲倒步态→locomotion+stumble |
| Self_motion-lurch | 前扑踉跄步态→locomotion+stumble |
| Motion_directional-fall | 混合(37 条): 28 条真倒地/跪倒/坐倒/护身倒法(ground_contact+fall); 9 条只是落入低蹲(idle_posture/locomotion) |
| Motion_directional-topple | 混合(39 条): 20 条倒地/跪倒; 19 条踉跄后落入低蹲 |
| Change_posture-sit_down | 混合(29 条): 14 条坐到地上/跪坐(ground_contact) vs 15 条坐到椅/箱/桌(sitting_on_object) |
| Change_posture-stand_up | 混合(36 条): 15 条从地上/跪/趴起身(ground+getup) vs 21 条从椅子/箱子起身(sitting_on_object) |
| Change_posture-sprawl | 混合: 扑倒在地 vs 瘫坐/趴在椅子上 |
| Posture-sit | 56 条: 53 条坐椅/凳/箱(sitting_on_object); 3 条坐地上抱着衣架/垃圾桶/水桶(ground_contact) |
| Manipulation-paw | 框架叫 Manipulation 但无物体: 是手在自己腿上抓挠→other(身体局部) |
| Scouring-rifle | 无物体的'搜寻'哑剧(下蹲伸手)→other/locomotion |
| Scouring-scour | 无物体的环顾/转身搜寻→locomotion/other |
| Scouring-sweep | 无物体的'扫视/伸手扫地面'→other |
| Body_movement-wave | 是手臂来回挥摆, 不一定是社交挥手; 归 gesture_social(wave) |
| Body_movement-kneel | 跪地后做上身动作→ground_contact |
| Body_movement-shiver | 发抖/寒颤(情绪)→gesture_social(emote); 多数边走边抖→也算 locomotion |
| Posture-cower | 蜷缩畏惧姿态→gesture_social(emote)+idle_posture |
| Posture-shrink | 抱膝坐地蜷缩→ground_contact(hips) |
| Attaching-hook | 用手臂弯/脚背/小腿勾衣架: 一半是勾住后全身拉/用脚抬架子(whole_body_object, 物体-脚接触 18 条), 一半是手臂勾挂(upper_body_manipulation) |
| Placing-lean | 把椅子斜靠在髋部→upper_body_manipulation + object-body |
| Change_posture-lean | HOI(32 条): 坐在椅/箱上前倾(sitting_on_object 40%), 全身压/靠在椅子或箱子上(whole_body_object 41%), 蹲/跪在椅边(ground 16%) |
| Cause_motion-punt | 踢箱子/球(object-feet, 140 条): 多标签 whole_body_object(kick ball)+dynamic(kick), 主类按优先级记为 dynamic |
| Cause_motion-force | 推/顶衣架、长凳、垃圾桶、球, 多为全身发力 |
| Cause_motion-shove | 同上 |
| Cause_to_move_in_place-turn | 混合: 77% 原地扭转/旋转物体(upper_body_manipulation), 23% 把长凳/箱子立起翻倒(whole_body_object) |
| Self_motion-press | 'press forward' 是前倾推进步态, 不是按压物体 |
| Self_motion-edge | 'edge' 是沿边缘踮脚走, 不是物体 |
| Self_motion-shrink | 缩身怯步走→locomotion |
| Self_motion-slip | 'slip' 是低身滑步走, 不是滑倒 |
| Self_motion-back | 全是后退步态(walk/march backward) |

## 10. 214 个词条 → 分类映射(按原片时长排序)

主类(占比) = 该词条按时长占比最高的主类。多标签列给出各类覆盖的条数比例。完整字段见 hiphi_lu_mapping.csv。

| Frame-LU | 原片 h | 条数 | 演员 | HOI% | 物体 | 主类(占比) | 多标签 | 已转/在盘 |
|---|---:|---:|---:|---:|---|---|---|---:|
| Bringing-carry | 8.33 | 466 | 43 | 100 | box:225 chair:119 table:91 soccerball:18 stepstool:12 mop:1 | 全身搬/推/踢物体 (0.96) | locomotion:1.00 whole_body_object:0.98 dynamic:0.02 upper_body_manipulation:0.02 | 20/20 |
| Cause_motion-move | 7.96 | 450 | 38 | 100 | chair:271 table:125 box:51 mop:3 | 全身搬/推/踢物体 (0.99) | whole_body_object:0.99 locomotion:0.86 upper_body_manipulation:0.01 | 57/57 |
| Bringing-lug | 7.67 | 436 | 46 | 100 | chair:189 table:128 box:86 stepstool:31 mop:2 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.91 upper_body_manipulation:0.00 | 25/25 |
| Bringing-take | 6.97 | 393 | 39 | 100 | chair:156 table:152 box:52 stepstool:24 mop:9 | 全身搬/推/踢物体 (0.98) | whole_body_object:0.98 locomotion:0.78 upper_body_manipulation:0.02 | 24/24 |
| Adjusting-tweak | 6.29 | 351 | 41 | 100 | chair:167 table:112 box:52 stepstool:19 mop:1 | 上肢操作物体 (0.75) | upper_body_manipulation:0.75 locomotion:0.33 whole_body_object:0.25 | 63/63 |
| Adjusting-adjust | 5.92 | 336 | 38 | 100 | chair:189 box:78 table:48 stepstool:15 mop:6 | 上肢操作物体 (0.69) | upper_body_manipulation:0.68 whole_body_object:0.32 locomotion:0.09 | 53/53 |
| Bringing-hump | 5.53 | 310 | 40 | 100 | chair:132 table:124 box:39 stepstool:9 mop:6 | 全身搬/推/踢物体 (0.98) | whole_body_object:0.98 locomotion:0.86 upper_body_manipulation:0.02 | 17/17 |
| Self_motion-sashay | 4.89 | 156 | 51 | 0 |  | 高动态 (0.6) | locomotion:0.71 dynamic:0.29 | 0/0 |
| Cause_motion-lift | 4.67 | 264 | 34 | 100 | chair:138 table:76 box:43 mop:7 | 全身搬/推/踢物体 (0.97) | whole_body_object:0.97 locomotion:0.26 upper_body_manipulation:0.03 | 48/48 |
| Cause_motion-pull | 4.6 | 261 | 34 | 100 | chair:108 table:106 box:46 mop:1 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.69 upper_body_manipulation:0.00 | 44/44 |
| Body_movement-bob | 4.28 | 249 | 62 | 0 |  | 其他 (1.0) | other:1.00 | 0/0 |
| Cause_to_move_in_place-rock | 4.25 | 246 | 34 | 100 | chair:117 table:69 box:56 mop:4 | 上肢操作物体 (0.64) | upper_body_manipulation:0.64 whole_body_object:0.36 locomotion:0.08 | 55/55 |
| Cause_motion-push | 4.2 | 240 | 38 | 100 | chair:96 table:88 box:48 ball:8 | 全身搬/推/踢物体 (0.98) | whole_body_object:0.98 locomotion:0.70 upper_body_manipulation:0.02 | 50/50 |
| Cause_to_move_in_place-rotate | 3.7 | 212 | 40 | 100 | chair:111 table:64 box:27 mop:10 | 上肢操作物体 (0.87) | upper_body_manipulation:0.86 whole_body_object:0.14 locomotion:0.06 | 39/39 |
| Body_movement-toss | 3.54 | 205 | 52 | 0 |  | 移动步态 (0.5) | locomotion:0.50 other:0.41 dynamic:0.09 | 0/0 |
| Cause_to_move_in_place-shake | 3.41 | 197 | 30 | 100 | chair:98 table:58 box:38 mop:3 | 上肢操作物体 (0.85) | upper_body_manipulation:0.85 whole_body_object:0.15 locomotion:0.10 | 54/54 |
| Cause_to_move_in_place-swing | 3.38 | 193 | 30 | 100 | table:84 chair:57 box:48 mop:4 | 上肢操作物体 (0.83) | upper_body_manipulation:0.83 whole_body_object:0.17 locomotion:0.14 | 20/20 |
| Body_movement-bend | 2.98 | 173 | 58 | 0 |  | 站姿/姿态/拉伸 (1.0) | idle_posture:1.00 | 0/0 |
| Cause_to_move_in_place-turn | 2.95 | 173 | 25 | 100 | table:44 chair:31 bucket:27 bench:19 clothrack:19 trashbin:17 box:16 | 上肢操作物体 (0.77) | upper_body_manipulation:0.77 whole_body_object:0.23 locomotion:0.03 | 38/38 |
| Cause_to_move_in_place-spin | 2.78 | 160 | 29 | 100 | chair:62 table:49 box:38 mop:11 | 上肢操作物体 (0.86) | upper_body_manipulation:0.86 locomotion:0.17 whole_body_object:0.14 | 16/16 |
| Dodging-sidestep | 2.7 | 159 | 44 | 0 |  | 移动步态 (0.81) | locomotion:1.00 dynamic:0.19 | 0/0 |
| Body_movement-swing | 2.64 | 162 | 45 | 0 |  | 其他 (0.76) | other:0.72 dynamic:0.27 idle_posture:0.01 ground_contact:0.01 | 0/0 |
| Body_movement-clap | 2.55 | 148 | 56 | 0 |  | 手势/社交/情绪 (0.79) | gesture_social:1.00 locomotion:0.21 | 0/0 |
| Self_motion-caper | 2.53 | 147 | 42 | 0 |  | 高动态 (1.0) | dynamic:1.00 locomotion:1.00 | 0/0 |
| Self_motion-run | 2.52 | 150 | 24 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-lope | 2.49 | 147 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 idle_posture:0.29 | 0/0 |
| Self_motion-jump | 2.47 | 146 | 38 | 0 |  | 高动态 (1.0) | dynamic:1.00 locomotion:0.57 idle_posture:0.02 | 0/0 |
| Body_movement-shake | 2.47 | 143 | 57 | 0 |  | 其他 (0.82) | other:0.83 gesture_social:0.17 | 0/0 |
| Cause_motion-punt | 2.43 | 140 | 18 | 100 | box:76 ball:45 soccerball:19 | 高动态 (0.99) | dynamic:0.99 whole_body_object:0.99 locomotion:0.81 upper_body_manipulation:0.01 | 3/3 |
| Change_direction-cut | 2.33 | 135 | 51 | 0 |  | 移动步态 (0.73) | locomotion:0.73 other:0.27 | 0/0 |
| Body_movement-wave | 2.31 | 134 | 45 | 0 |  | 手势/社交/情绪 (0.84) | gesture_social:1.00 locomotion:0.16 | 0/0 |
| Cause_motion-force | 2.3 | 135 | 18 | 100 | table:44 chair:28 bucket:21 clothrack:10 bench:10 trashbin:9 ball:7 box:6 | 全身搬/推/踢物体 (0.81) | whole_body_object:0.81 locomotion:0.69 upper_body_manipulation:0.19 | 52/52 |
| Change_direction-turn | 2.26 | 131 | 50 | 0 |  | 移动步态 (0.85) | locomotion:0.85 other:0.15 | 0/0 |
| Gesture-gesture | 2.22 | 126 | 45 | 0 |  | 手势/社交/情绪 (1.0) | gesture_social:1.00 | 0/0 |
| Change_direction-bear | 2.2 | 126 | 42 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Change_direction-veer | 2.19 | 127 | 43 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Gesture-gesticulate | 2.17 | 124 | 46 | 0 |  | 手势/社交/情绪 (1.0) | gesture_social:1.00 | 0/0 |
| Self_motion-nance | 2.11 | 121 | 40 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Patrolling-patrol | 2.09 | 120 | 42 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-back | 2.06 | 120 | 41 | 0 |  | 移动步态 (0.94) | locomotion:1.00 dynamic:0.06 | 0/0 |
| Self_motion-goose-step | 2.05 | 119 | 46 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-walk | 2.05 | 115 | 39 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-wriggle | 2.04 | 116 | 45 | 0 |  | 其他 (0.46) | other:0.47 locomotion:0.41 ground_contact:0.12 | 0/0 |
| Self_motion-hop | 2.03 | 119 | 42 | 0 |  | 高动态 (1.0) | dynamic:1.00 locomotion:0.85 | 0/0 |
| Self_motion-frolic | 2.03 | 118 | 45 | 0 |  | 高动态 (0.71) | locomotion:1.00 dynamic:0.71 | 0/0 |
| Self_motion-stomp | 2.03 | 117 | 40 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-hurry | 2.02 | 116 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-shuffle | 2.01 | 114 | 41 | 0 |  | 移动步态 (0.66) | locomotion:0.66 other:0.34 | 0/0 |
| Self_motion-scurry | 2.01 | 117 | 35 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-amble | 2.01 | 117 | 39 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Taking-seize | 2.01 | 116 | 23 | 100 | chair:41 table:36 box:28 mop:11 | 全身搬/推/踢物体 (0.9) | whole_body_object:0.91 locomotion:0.30 upper_body_manipulation:0.09 | 11/11 |
| Cause_to_move_in_place-vibrate | 1.99 | 116 | 21 | 100 | chair:66 box:38 table:12 | 上肢操作物体 (0.79) | upper_body_manipulation:0.79 whole_body_object:0.21 locomotion:0.03 | 18/18 |
| Body_movement-wiggle | 1.99 | 116 | 45 | 0 |  | 其他 (0.65) | other:0.66 locomotion:0.22 ground_contact:0.13 | 0/0 |
| Self_motion-flounce | 1.99 | 116 | 41 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Manipulation-paw | 1.97 | 113 | 45 | 0 |  | 其他 (1.0) | other:1.00 | 0/0 |
| Body_movement-twitch | 1.97 | 114 | 47 | 0 |  | 其他 (0.81) | other:0.81 locomotion:0.19 | 0/0 |
| Body_movement-cross | 1.95 | 113 | 46 | 0 |  | 其他 (0.5) | other:0.50 locomotion:0.42 ground_contact:0.09 | 0/0 |
| Self_motion-trot | 1.94 | 112 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-mince | 1.93 | 112 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-twist | 1.93 | 112 | 40 | 0 |  | 其他 (0.79) | other:0.79 locomotion:0.21 | 0/0 |
| Body_movement-stamp | 1.92 | 112 | 41 | 0 |  | 移动步态 (0.63) | locomotion:0.63 other:0.37 | 0/0 |
| Self_motion-edge | 1.92 | 112 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-pad | 1.9 | 109 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-clomp | 1.83 | 106 | 42 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Motion_directional-dip | 1.83 | 102 | 35 | 0 |  | 移动步态 (0.61) | locomotion:0.61 other:0.23 idle_posture:0.17 | 0/0 |
| Self_motion-waddle | 1.76 | 100 | 35 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-crane | 1.76 | 102 | 46 | 0 |  | 其他 (0.74) | other:0.75 locomotion:0.25 | 0/0 |
| Cause_motion-shove | 1.75 | 102 | 12 | 100 | table:34 bucket:20 box:11 chair:10 clothrack:10 bench:9 ball:5 trashbin:3 | 全身搬/推/踢物体 (0.77) | whole_body_object:0.76 locomotion:0.46 upper_body_manipulation:0.24 | 48/48 |
| Body_movement-hang | 1.71 | 99 | 42 | 0 |  | 移动步态 (0.66) | locomotion:0.66 other:0.34 | 0/0 |
| Body_movement-flap | 1.7 | 99 | 37 | 0 |  | 其他 (0.66) | other:0.67 locomotion:0.33 | 0/0 |
| Self_motion-shuffle | 1.68 | 96 | 40 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-stride | 1.66 | 94 | 42 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-slip | 1.66 | 97 | 39 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Scouring-sweep | 1.66 | 96 | 42 | 0 |  | 移动步态 (0.58) | locomotion:0.58 other:0.42 | 0/0 |
| Self_motion-bop | 1.65 | 95 | 40 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-swagger | 1.64 | 95 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-drop | 1.64 | 95 | 31 | 0 |  | 其他 (0.52) | other:0.53 gesture_social:0.26 locomotion:0.21 | 0/0 |
| Body_movement-shrug | 1.62 | 92 | 39 | 0 |  | 移动步态 (0.61) | gesture_social:1.00 locomotion:0.60 | 0/0 |
| Self_motion-shrink | 1.61 | 93 | 36 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Taking-take | 1.55 | 88 | 19 | 100 | chair:44 table:33 box:9 mop:2 | 全身搬/推/踢物体 (0.98) | whole_body_object:0.98 locomotion:0.27 upper_body_manipulation:0.02 | 20/20 |
| Scouring-scour | 1.54 | 89 | 40 | 0 |  | 移动步态 (0.7) | locomotion:0.70 other:0.30 | 0/0 |
| Manipulation-touch | 1.54 | 89 | 21 | 100 | chair:43 table:19 ball:16 box:11 | 上肢操作物体 (0.79) | upper_body_manipulation:0.79 whole_body_object:0.21 locomotion:0.21 | 20/20 |
| Self_motion-skim | 1.54 | 89 | 41 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-shiver | 1.53 | 88 | 30 | 0 |  | 移动步态 (0.74) | gesture_social:1.00 locomotion:0.74 | 0/0 |
| Self_motion-prowl | 1.53 | 86 | 33 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-curtsy | 1.51 | 88 | 38 | 0 |  | 手势/社交/情绪 (0.54) | gesture_social:1.00 locomotion:0.45 | 0/0 |
| Self_motion-strut | 1.49 | 86 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-step | 1.46 | 86 | 23 | 66 | bench:43 -:29 box:9 chair:2 table:2 trashbin:1 | 全身搬/推/踢物体 (0.49) | locomotion:1.00 whole_body_object:0.49 upper_body_manipulation:0.12 | 2/2 |
| Self_motion-slouch | 1.45 | 83 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-lift | 1.44 | 84 | 29 | 0 |  | 其他 (1.0) | other:1.00 | 0/0 |
| Self_motion-totter | 1.43 | 81 | 35 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-roll | 1.43 | 83 | 34 | 0 |  | 其他 (0.77) | other:0.77 locomotion:0.23 | 0/0 |
| Body_movement-fling | 1.42 | 83 | 30 | 0 |  | 其他 (0.86) | other:0.86 dynamic:0.14 | 0/0 |
| Body_movement-flutter | 1.4 | 81 | 36 | 0 |  | 其他 (0.71) | other:0.70 locomotion:0.21 ground_contact:0.09 | 0/0 |
| Taking-grab | 1.4 | 81 | 16 | 100 | chair:57 box:19 mop:5 | 全身搬/推/踢物体 (0.94) | whole_body_object:0.94 locomotion:0.73 upper_body_manipulation:0.06 | 16/16 |
| Cause_motion-hit | 1.38 | 48 | 17 | 69 | box:33 -:15 | 高动态 (0.86) | dynamic:0.77 whole_body_object:0.46 upper_body_manipulation:0.23 locomotion:0.10 | 9/9 |
| Self_motion-dart | 1.38 | 81 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Manipulation-nudge | 1.36 | 78 | 20 | 100 | chair:35 table:32 box:6 ball:5 | 上肢操作物体 (0.52) | upper_body_manipulation:0.53 whole_body_object:0.47 locomotion:0.10 | 21/21 |
| Self_motion-bustle | 1.34 | 77 | 37 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-flex | 1.33 | 77 | 36 | 0 |  | 站姿/姿态/拉伸 (0.73) | idle_posture:1.00 locomotion:0.27 | 0/0 |
| Manipulation-grab | 1.32 | 76 | 19 | 100 | chair:50 table:26 | 全身搬/推/踢物体 (0.66) | whole_body_object:0.66 upper_body_manipulation:0.34 locomotion:0.13 | 17/17 |
| Self_motion-march | 1.3 | 75 | 35 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-lunge | 1.28 | 75 | 39 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Attaching-hook | 1.22 | 70 | 10 | 100 | clothrack:70 | 全身搬/推/踢物体 (0.51) | whole_body_object:0.51 upper_body_manipulation:0.49 | 14/14 |
| Self_motion-press | 1.21 | 70 | 35 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-prance | 1.18 | 70 | 29 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-lurch | 1.16 | 67 | 38 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-trip | 1.15 | 66 | 33 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Self_motion-scoot | 1.12 | 64 | 35 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-nod | 1.05 | 61 | 27 | 0 |  | 手势/社交/情绪 (0.51) | gesture_social:1.00 locomotion:0.49 | 0/0 |
| Change_posture-hunch_up | 1.03 | 60 | 28 | 0 |  | 移动步态 (0.74) | idle_posture:1.00 locomotion:0.75 | 0/0 |
| Self_motion-limp | 1.0 | 56 | 33 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-arch | 1.0 | 55 | 21 | 0 |  | 移动步态 (0.47) | idle_posture:1.00 locomotion:0.47 ground_contact:0.15 | 0/0 |
| Motion_directional-rise | 0.99 | 56 | 27 | 0 |  | 移动步态 (0.46) | locomotion:0.46 other:0.39 dynamic:0.14 | 0/0 |
| Self_motion-sidle | 0.98 | 55 | 27 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Motion_directional-drop | 0.98 | 54 | 14 | 0 |  | 其他 (0.52) | other:0.52 dynamic:0.28 idle_posture:0.20 | 0/0 |
| Body_movement-move | 0.98 | 57 | 25 | 0 |  | 其他 (0.86) | other:0.86 gesture_social:0.14 | 0/0 |
| Manipulation-caress | 0.98 | 57 | 15 | 100 | table:36 ball:13 box:8 | 上肢操作物体 (0.95) | upper_body_manipulation:0.95 locomotion:0.25 whole_body_object:0.05 | 17/17 |
| Posture-sit | 0.97 | 56 | 11 | 100 | chair:30 bench:13 box:10 trashbin:1 bucket:1 clothrack:1 | 坐物体 (0.95) | sitting_on_object:0.95 upper_body_manipulation:0.20 ground_contact:0.05 | 10/10 |
| Body_movement-reach | 0.95 | 55 | 23 | 0 |  | 站姿/姿态/拉伸 (1.0) | idle_posture:1.00 | 0/0 |
| Change_posture-crouch_down | 0.94 | 55 | 32 | 0 |  | 移动步态 (0.72) | idle_posture:1.00 locomotion:0.73 | 0/0 |
| Placing-shoulder | 0.94 | 54 | 12 | 100 | chair:54 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.19 | 4/4 |
| Self_motion-barge | 0.93 | 54 | 30 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Posture-kneel | 0.93 | 52 | 26 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 | 0/0 |
| Posture-cower | 0.91 | 50 | 23 | 0 |  | 移动步态 (0.7) | gesture_social:1.00 idle_posture:1.00 locomotion:0.70 | 0/0 |
| Change_posture-bend | 0.91 | 53 | 22 | 0 |  | 站姿/姿态/拉伸 (0.73) | idle_posture:1.00 locomotion:0.26 | 0/0 |
| Self_motion-slalom | 0.89 | 51 | 27 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Body_movement-stretch | 0.88 | 51 | 22 | 0 |  | 站姿/姿态/拉伸 (1.0) | idle_posture:1.00 | 0/0 |
| Body_movement-duck | 0.83 | 48 | 11 | 0 |  | 移动步态 (0.63) | locomotion:0.62 other:0.38 | 0/0 |
| Scouring-rifle | 0.77 | 44 | 18 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Change_posture-kneel_down | 0.75 | 44 | 23 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 dynamic:0.84 locomotion:0.61 | 0/0 |
| Self_motion-vault | 0.75 | 70 | 10 | 0 |  | 地面接触 (0.69) | dynamic:0.96 ground_contact:0.66 locomotion:0.07 idle_posture:0.04 | 0/0 |
| Posture-stand | 0.71 | 41 | 12 | 0 |  | 站姿/姿态/拉伸 (1.0) | idle_posture:1.00 | 0/0 |
| Change_posture-stoop | 0.71 | 42 | 18 | 0 |  | 移动步态 (0.78) | idle_posture:1.00 locomotion:0.79 | 0/0 |
| Posture-shrink | 0.69 | 38 | 15 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 locomotion:0.61 | 0/0 |
| Motion_directional-topple | 0.67 | 39 | 18 | 0 |  | 地面接触 (0.51) | locomotion:0.74 dynamic:0.51 ground_contact:0.51 idle_posture:0.49 | 0/0 |
| Posture-bend | 0.66 | 35 | 12 | 0 |  | 移动步态 (0.63) | idle_posture:1.00 locomotion:0.63 | 0/0 |
| Body_movement-kneel | 0.65 | 38 | 12 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 dynamic:0.26 locomotion:0.24 idle_posture:0.08 | 0/0 |
| Posture-lie | 0.64 | 33 | 15 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 idle_posture:0.21 | 0/0 |
| Self_motion-lumber | 0.63 | 36 | 26 | 0 |  | 移动步态 (1.0) | locomotion:1.00 | 0/0 |
| Change_posture-stand_up | 0.61 | 36 | 13 | 58 | chair:19 -:15 box:2 | 坐物体 (0.58) | sitting_on_object:0.58 dynamic:0.42 ground_contact:0.42 locomotion:0.11 | 8/8 |
| Cause_motion-cast | 0.59 | 35 | 6 | 100 | ball:35 | 上肢操作物体 (0.94) | upper_body_manipulation:0.94 locomotion:0.23 whole_body_object:0.06 | 3/3 |
| Cause_motion-chuck | 0.59 | 35 | 5 | 100 | ball:35 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.60 | 6/6 |
| Motion_directional-fall | 0.58 | 37 | 21 | 0 |  | 地面接触 (0.73) | dynamic:0.76 ground_contact:0.76 locomotion:0.51 idle_posture:0.24 | 0/0 |
| Cause_to_move_in_place-roll | 0.57 | 33 | 8 | 100 | box:33 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 | 0/0 |
| Cause_motion-drop | 0.56 | 33 | 7 | 100 | box:33 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.36 | 0/0 |
| Cause_to_move_in_place-seesaw | 0.55 | 32 | 9 | 100 | box:32 | 上肢操作物体 (0.69) | upper_body_manipulation:0.69 whole_body_object:0.31 | 1/1 |
| Change_posture-lean | 0.55 | 32 | 11 | 100 | chair:21 box:11 | 全身搬/推/踢物体 (0.41) | whole_body_object:0.53 sitting_on_object:0.41 ground_contact:0.16 upper_body_manipulation:0.06 | 8/8 |
| Cause_motion-pitch | 0.55 | 33 | 7 | 100 | ball:33 | 上肢操作物体 (0.94) | upper_body_manipulation:0.94 locomotion:0.39 whole_body_object:0.06 | 7/7 |
| Posture-huddle | 0.55 | 29 | 17 | 0 |  | 移动步态 (0.61) | idle_posture:1.00 locomotion:0.62 | 0/0 |
| Posture-crouch | 0.55 | 30 | 11 | 0 |  | 站姿/姿态/拉伸 (0.8) | idle_posture:1.00 locomotion:0.20 | 0/0 |
| Placing-put | 0.54 | 31 | 10 | 100 | box:31 | 全身搬/推/踢物体 (0.94) | whole_body_object:0.94 upper_body_manipulation:0.06 | 9/9 |
| Cause_to_move_in_place-jiggle | 0.54 | 31 | 10 | 100 | box:31 | 上肢操作物体 (0.87) | upper_body_manipulation:0.87 whole_body_object:0.13 | 3/3 |
| Change_posture-sit_down | 0.5 | 29 | 14 | 55 | -:13 chair:9 table:4 box:3 | 坐物体 (0.52) | sitting_on_object:0.52 ground_contact:0.48 dynamic:0.34 upper_body_manipulation:0.10 | 4/4 |
| Posture-slouch | 0.49 | 26 | 7 | 0 |  | 移动步态 (0.77) | idle_posture:1.00 locomotion:0.77 | 0/0 |
| Posture-squat | 0.48 | 27 | 12 | 0 |  | 站姿/姿态/拉伸 (0.66) | idle_posture:1.00 locomotion:0.33 | 0/0 |
| Placing-lean | 0.48 | 28 | 6 | 100 | chair:24 box:4 | 上肢操作物体 (0.68) | upper_body_manipulation:0.68 locomotion:0.36 whole_body_object:0.32 | 14/14 |
| Cause_to_move_in_place-swirl | 0.47 | 27 | 6 | 100 | box:27 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.85 | 0/0 |
| Filling-splash | 0.46 | 27 | 10 | 100 | bucket:27 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.22 | 7/7 |
| Cause_to_move_in_place-wiggle | 0.43 | 25 | 6 | 100 | box:25 | 上肢操作物体 (0.6) | upper_body_manipulation:0.60 whole_body_object:0.40 | 0/0 |
| Cause_motion-thrust | 0.42 | 23 | 8 | 100 | box:23 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.57 | 4/4 |
| Emptying-purge | 0.38 | 22 | 5 | 100 | box:22 | 上肢操作物体 (0.59) | upper_body_manipulation:0.59 whole_body_object:0.41 locomotion:0.05 | 5/5 |
| Emptying-cleanse | 0.38 | 22 | 6 | 100 | box:22 | 上肢操作物体 (0.78) | upper_body_manipulation:0.77 whole_body_object:0.23 locomotion:0.14 | 4/4 |
| Placing-shower | 0.38 | 22 | 8 | 100 | bucket:22 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.18 | 5/5 |
| Filling-douse | 0.37 | 22 | 8 | 100 | bucket:22 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.36 | 8/8 |
| Emptying-clear | 0.36 | 20 | 5 | 100 | box:20 | 上肢操作物体 (0.74) | upper_body_manipulation:0.75 whole_body_object:0.25 | 3/3 |
| Removing-remove | 0.34 | 20 | 7 | 100 | bucket:20 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 | 5/5 |
| Removing-dislodge | 0.33 | 19 | 7 | 100 | bucket:19 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.63 | 5/5 |
| Removing-dust | 0.32 | 19 | 5 | 100 | bucket:19 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.32 | 5/5 |
| Cause_motion-propel | 0.31 | 18 | 4 | 100 | box:18 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.50 | 6/6 |
| Manipulation-clasp | 0.31 | 18 | 3 | 100 | box:18 | 上肢操作物体 (0.67) | upper_body_manipulation:0.67 whole_body_object:0.33 locomotion:0.22 | 3/3 |
| Emptying-empty | 0.31 | 18 | 5 | 100 | box:18 | 全身搬/推/踢物体 (0.56) | whole_body_object:0.56 upper_body_manipulation:0.44 | 3/3 |
| Self_motion-spring | 0.3 | 16 | 13 | 0 |  | 高动态 (1.0) | dynamic:1.00 | 0/0 |
| Self_motion-scramble | 0.3 | 16 | 10 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 | 0/0 |
| Emptying-strip | 0.29 | 17 | 4 | 100 | box:17 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.12 | 5/5 |
| Emptying-expunge | 0.28 | 16 | 6 | 100 | box:16 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 | 5/5 |
| Change_posture-squat_down | 0.27 | 16 | 3 | 0 |  | 站姿/姿态/拉伸 (1.0) | idle_posture:1.00 | 0/0 |
| Emptying-unload | 0.27 | 16 | 4 | 100 | box:16 | 上肢操作物体 (0.63) | upper_body_manipulation:0.62 whole_body_object:0.38 locomotion:0.31 | 1/1 |
| Removing-take | 0.27 | 16 | 5 | 100 | bucket:16 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.44 | 5/5 |
| Emptying-rid | 0.27 | 16 | 5 | 100 | box:16 | 上肢操作物体 (0.69) | upper_body_manipulation:0.69 whole_body_object:0.31 locomotion:0.06 | 2/2 |
| Scouring-rummage | 0.25 | 15 | 5 | 100 | box:15 | 上肢操作物体 (0.81) | upper_body_manipulation:0.80 whole_body_object:0.20 locomotion:0.20 | 1/1 |
| Removing-extract | 0.25 | 15 | 4 | 100 | bucket:15 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.60 | 5/5 |
| Removing-drain | 0.25 | 15 | 4 | 100 | bucket:15 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.20 | 5/5 |
| Removing-flush | 0.25 | 15 | 4 | 100 | bucket:15 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.60 | 5/5 |
| Placing-drizzle | 0.24 | 14 | 6 | 100 | bottle:14 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.43 | 1/1 |
| Posture-sprawl | 0.24 | 12 | 4 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 idle_posture:0.17 | 0/0 |
| Scouring-ransack | 0.24 | 14 | 2 | 100 | box:14 | 上肢操作物体 (0.58) | upper_body_manipulation:0.64 locomotion:0.36 whole_body_object:0.36 dynamic:0.07 | 0/0 |
| Self_motion-pounce | 0.22 | 12 | 7 | 0 |  | 高动态 (1.0) | dynamic:1.00 locomotion:1.00 | 0/0 |
| Cause_motion-slide | 0.22 | 13 | 5 | 100 | ball:13 | 上肢操作物体 (0.68) | locomotion:0.69 upper_body_manipulation:0.69 dynamic:0.31 whole_body_object:0.31 | 3/3 |
| Getting_up-get_up | 0.19 | 11 | 10 | 0 |  | 地面接触 (1.0) | dynamic:1.00 ground_contact:1.00 | 0/0 |
| Cause_motion-roll | 0.19 | 11 | 3 | 100 | ball:11 | 上肢操作物体 (0.63) | upper_body_manipulation:0.64 locomotion:0.64 whole_body_object:0.36 | 0/0 |
| Cause_motion-slam | 0.19 | 11 | 3 | 100 | ball:11 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.45 | 3/3 |
| Change_posture-sprawl | 0.18 | 11 | 5 | 55 | chair:6 -:5 | 地面接触 (0.45) | upper_body_manipulation:0.55 dynamic:0.45 ground_contact:0.45 sitting_on_object:0.27 | 0/0 |
| Placing-plunge | 0.18 | 10 | 5 | 100 | bucket:10 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.20 | 0/0 |
| Self_motion-burrow | 0.18 | 10 | 9 | 0 |  | 地面接触 (0.51) | locomotion:0.50 ground_contact:0.50 | 0/0 |
| Cause_motion-toss | 0.18 | 11 | 3 | 100 | ball:11 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.36 | 6/6 |
| Cause_motion-throw | 0.17 | 10 | 5 | 100 | ball:10 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.40 | 4/4 |
| Change_posture-sit_up | 0.17 | 10 | 2 | 100 | chair:9 box:1 | 坐物体 (0.9) | sitting_on_object:1.00 upper_body_manipulation:0.50 ground_contact:0.10 | 2/2 |
| Manipulation-squeeze | 0.17 | 10 | 3 | 100 | ball:10 | 全身搬/推/踢物体 (0.5) | upper_body_manipulation:0.50 whole_body_object:0.50 locomotion:0.40 | 3/3 |
| Manipulation-seize | 0.17 | 10 | 3 | 100 | ball:10 | 上肢操作物体 (0.59) | locomotion:0.60 upper_body_manipulation:0.60 whole_body_object:0.40 | 1/1 |
| Self_motion-crawl | 0.17 | 10 | 10 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 | 0/0 |
| Motion_directional-plunge | 0.17 | 10 | 7 | 0 |  | 地面接触 (1.0) | dynamic:1.00 ground_contact:1.00 | 0/0 |
| Placing-bottle | 0.17 | 10 | 3 | 100 | bottle:10 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.30 | 1/1 |
| Change_posture-lie_down | 0.16 | 9 | 8 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 dynamic:0.22 | 0/0 |
| Change_posture-kneel_up | 0.14 | 8 | 4 | 0 |  | 地面接触 (1.0) | ground_contact:1.00 dynamic:0.25 | 0/0 |
| Cause_motion-drag | 0.1 | 6 | 2 | 100 | stepstool:5 table:1 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.50 | 0/0 |
| Manipulation-hold | 0.1 | 6 | 3 | 100 | box:6 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 | 4/4 |
| Placing-pile | 0.07 | 4 | 1 | 100 | box:4 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.25 | 0/0 |
| Removing-strip | 0.07 | 4 | 2 | 100 | bucket:4 | 上肢操作物体 (1.0) | upper_body_manipulation:1.00 locomotion:0.25 | 0/0 |
| Placing-situate | 0.03 | 2 | 1 | 100 | table:2 | 全身搬/推/踢物体 (1.0) | whole_body_object:1.00 locomotion:0.50 | 0/0 |
| Cause_to_move_in_place-twirl | 0.02 | 1 | 1 | 100 | table:1 | 上肢操作物体 (1.0) | locomotion:1.00 upper_body_manipulation:1.00 | 0/0 |
| Cause_motion-haul | 0.02 | 1 | 1 | 100 | stepstool:1 | 全身搬/推/踢物体 (1.0) | locomotion:1.00 whole_body_object:1.00 | 0/0 |
| Cause_motion-draw | 0.02 | 1 | 1 | 100 | stepstool:1 | 全身搬/推/踢物体 (1.0) | locomotion:1.00 whole_body_object:1.00 | 1/1 |
| Manipulation-stroke | 0.02 | 1 | 1 | 100 | table:1 | 上肢操作物体 (1.0) | locomotion:1.00 upper_body_manipulation:1.00 | 0/0 |

## 11. 方法和局限

- 只读了元数据(hiphi_metadata.csv / actor / frame_lu_index / motion_to_part / archive_manifest)、下载器状态 json、已转目录的文件名和 40 个 OBJ 网格的包围盒。没碰 GPU, 没下载任何东西。
- 分类 = 词条默认类 + 英文描述关键词规则(脚本 hiphi_diversity.py), 再人工抽查过: 地面接触的全部 27 个来源词条逐条看过; 坐物体、跳跃/踢、全身/上肢 各抽 15–25 条看过, 修掉了'shoulder rolls 被当成滚翻'、'how it sits 被当成坐'、'KPOP_Monster 下划线没匹配上舞蹈'等误判。
- 规则有边界误差: 全身 vs 上肢物体 的线是人为定的; '原地局部身体动作' 在给定分类里没有合适的位置, 统一放进'其他'(32.65 h), 它们是摇头、甩手、扭胯、耸肩之类, 对上半身表现力有用。
- 接触类型完全从文字推断, 不是从动作数据测的。镜像时长 = 原片时长(逐条一一对应, 已核对总和相等)。

## 输出文件

- summary_json: `/home/arenalabs/umr_factory_data/diversity/hiphi/hiphi_diversity_summary.json`
- report_md: `/home/arenalabs/umr_factory_data/diversity/hiphi/hiphi_diversity_report.md`
- lu_mapping_csv: `/home/arenalabs/umr_factory_data/diversity/hiphi/hiphi_lu_mapping.csv`
- clips_csv: `/home/arenalabs/umr_factory_data/diversity/hiphi/hiphi_clips_labeled.csv`
- script: `/home/arenalabs/umr_factory_data/diversity/hiphi/hiphi_diversity.py`
