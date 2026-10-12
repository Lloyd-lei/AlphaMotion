# OmniContact + GRAIL 数据丰富度盘点

生成时间 2026-10-05 00:03:56; GRAIL manifest 快照 2026-10-05 00:01:42 (转换还在跑, 数字会涨); 只读统计, 没用 GPU。

## 一句话
- OmniContact: 690 段真人动捕, 12.026 小时(90 fps), 全部已转 SMPL-X; 内容只有 搬箱/推箱/脚推箱/踢球/捡球 5 种, 每段都带物体。
- GRAIL: 发布 22,312 段 ≈ 61.961 小时, 其中有人体 SMPL-X 的 ≈ 55.599 小时; 已转 9503 段 = 24.377 小时(新版 v2/v3 14.859 h, 旧版 v1 9.518 h 待重转)。全是合成角色, 体型参数全为 0(只有一个身材)。
- 两个来源加起来已经 retarget 到机器人的: 只有 OmniContact 2 段 QA 试跑窗口(最多 6 台), GRAIL 为 0。离 21 台全量还差全部。

## 分类别明细

| 来源/类别 | 统一分类(多标签) | 段数: 发布 / 盘上原始 / 已转SMPL-X / 已retarget | 小时: 原始(已转) / 镜像 | fps | 物体 | 接触类型 | 交互帧比 | 状态 |
|---|---|---|---|---|---|---|---|---|
| OmniContact box/case1_carry 搬箱子 | whole_body_object(carry_box); upper_body_manipulation(pick,place,carry); locomotion(walk,turn) | 210(切片623) / 210 / 210 / 1(QA窗口, 3台) | 3.484 / 0 | 90 | 箱子 30×30×30 cm×210 | feet, object-hands, object-torso(抱箱贴胸, 人体穿模统计佐证) | G1标签≈0.655; 人-物<2cm 0.62 | 已转SMPL-X |
| OmniContact box/case2_push 推箱子 | whole_body_object(push_box); locomotion(walk,turn) | 162(切片453) / 162 / 162 / 0 | 2.929 / 0 | 90 | 箱子 48.5×40×72.5 cm×162 | feet, object-hands | G1标签≈0.385; 人-物<2cm 0.26 | 已转SMPL-X |
| OmniContact box/case3_slide 用脚滑/推箱子 | whole_body_object(push_box_with_foot); locomotion(walk,turn) | 118(切片190) / 118 / 118 / 0 | 2.064 / 0 | 90 | 箱子 30×30×30 cm×118 | feet, object-feet | G1标签≈0.229; 人-物<2cm 0.12 | 已转SMPL-X |
| OmniContact soccer/case1_kick_forward 向前踢球 | dynamic(kick); whole_body_object(kick_ball); locomotion(walk,turn) | 30(切片177) / 30 / 30 / 0 | 0.535 / 0 | 90 | 足球 直径 21.6 cm×30 | feet, object-feet | 标签未下; 人-物<2cm 0.03 | 已转SMPL-X |
| OmniContact soccer/case2_kick_left 向左踢球 | dynamic(kick); whole_body_object(kick_ball); locomotion(walk,turn) | 30(切片170) / 30 / 30 / 0 | 0.533 / 0 | 90 | 足球 直径 21.6 cm×30 | feet, object-feet | 标签未下; 人-物<2cm 0.03 | 已转SMPL-X |
| OmniContact soccer/case3_kick_right 向右踢球 | dynamic(kick); whole_body_object(kick_ball); locomotion(walk,turn) | 30(切片210) / 30 / 30 / 1(QA窗口, 6台) | 0.523 / 0 | 90 | 足球 直径 21.6 cm×30 | feet, object-feet(含小腿) | 标签未下; 人-物<2cm 0.03 | 已转SMPL-X |
| OmniContact soccer/case4_carry 捡球搬运 | upper_body_manipulation(pick,carry,place); locomotion(walk,turn) | 51(切片199) / 51 / 51 / 0 | 0.907 / 0 | 90 | 足球 直径 21.6 cm×51 | feet, object-hands | 标签未下; 人-物<2cm 0.68 | 已转SMPL-X |
| OmniContact soccer/case5_pickball_kick 捡球再踢 | upper_body_manipulation(pick,place); dynamic(kick); whole_body_object(kick_ball); locomotion(walk,turn) | 59(切片209) / 59 / 59 / 0 | 1.051 / 0 | 90 | 足球 直径 21.6 cm×59 | feet, object-hands, object-feet | 标签未下; 人-物<2cm 0.62 | 已转SMPL-X |
| GRAIL pickup_table 桌面取物: 从桌上抓起物体, 举起, 搬运 | locomotion/turn:1; upper_body_manipulation/lift_in_place:1280; upper_body_manipulation/pick:1280 | 2991 / 1339 / 1280 / 0 | 发布≈8.308 (已转 3.57; 新版 0.826) / 0 | 24 | 685 个物体/75 类(最大边 p50 0.148 m) | feet, object-hands | 0.478 | 已转 1280, 待下/待转 1711, 丢弃 0 |
| GRAIL pickup_ground 地面取物: 跪下抓地上物体再站起 | ground_contact/kneel_or_deep_squat:155; locomotion/walk:38; upper_body_manipulation/pick:209 | 1736 / 209 / 209 / 0 | 发布≈7.233 (已转 0.881; 新版 0.881) / 0 | 24 | 644 个物体/76 类(最大边 p50 0.177 m) | feet, knees(跪, README 所述 + 骨盆下降佐证), object-hands | 0.249 | 已转 209, 待下/待转 0, 丢弃 1527 |
| GRAIL sitting 坐到椅子类物体上 | locomotion/turn:878; locomotion/walk:1742; sitting_on_object/chair:1748 | 1748 / 1748 / 1748 / 0 | 发布≈2.428 (已转 2.448; 新版 2.277) / 0 | 24 | 189 把椅子: 其他椅子586, 餐椅/侧椅320, 扶手椅304, 雕塑/异形座160, 宝座/礼仪椅145, 单座/剧场座137 | feet, hips(坐面), back(靠背, 推断未核实) | 0.125 | 已转 1748, 待下/待转 0, 丢弃 0 |
| GRAIL slope 斜坡行走 | locomotion/ascend:366; locomotion/level_or_up-down:1514; locomotion/slope:1880; locomotion/turn:979; locomotion/walk:1880 | 1880 / 1880 / 1880 / 0 | 发布≈5.222 (已转 5.244; 新版 4.535) / 0 | 24 | 200 个斜坡, 坡度 p50 3.6° (p95 9.8°) | feet only(地形物体上) | 0.181 | 已转 1880, 待下/待转 0, 丢弃 0 |
| GRAIL curb 跨路沿 | locomotion/ascend:289; locomotion/curb:1769; locomotion/descend:1; locomotion/level_or_up-down:1479; locomotion/run:5; locomotion/turn:930; locomotion/walk:1769 | 1769 / 1769 / 1769 / 0 | 发布≈4.914 (已转 4.934; 新版 4.538) / 0 | 24 | 200 个路沿, 高 0.174-0.355 m | feet only(地形物体上) | 0.198 | 已转 1769, 待下/待转 0, 丢弃 0 |
| GRAIL stair_p1 上下楼梯(生成+真实楼梯资产) 第1部分 | locomotion/ascend:11; locomotion/descend:931; locomotion/level_or_up-down:293; locomotion/run:1; locomotion/stairs:1306; locomotion/turn:351; locomotion/walk:1306 | 6094 / 1339 / 1306 / 0 | 发布≈16.928 (已转 3.643; 新版 0.895) / 0 | 24 | 3138 个楼梯: 程序化楼梯 stairs_NNNN4012, 小型练习台阶(3-6 级)2002, 弧形楼梯80; 上下往返 513 | feet only(楼梯物体上) | 0.195 | 已转 1306, 待下/待转 4788, 丢弃 0 |
| GRAIL stair_p2 上下楼梯 第2部分 | locomotion/ascend:1261; locomotion/stairs:1311; locomotion/turn:377; locomotion/walk:1311 | 6094 / 1338 / 1311 / 0 | 发布≈16.928 (已转 3.657; 新版 0.907) / 0 | 24 | 2489 个楼梯: 程序化楼梯 stairs_NNNN3463, 直跑楼梯(生成, 10-15 级)1651, 真实楼梯资产(stairs_real)971, 弧形楼梯9; 上下往返 161 | feet only(楼梯物体上) | 0.125 | 已转 1311, 待下/待转 4783, 丢弃 0 |

交互帧比: GRAIL 用官方 interaction_frames 区间 / 总帧; OmniContact 用 G1 物体接触标签(只有箱子类已下)和人体-物体 2cm 内帧比。

## 按统一分类汇总(多标签, 同一段可算进多类)

| 统一分类 | OmniContact 小时 | GRAIL 已转小时 | GRAIL 段数 | 出现的子类 |
|---|---|---|---|---|
| locomotion 行走类 | 12.026 | 20.078 | 8046 | ascend, curb, descend, level_or_up-down, run, slope, stairs, turn, walk |
| dynamic 动态 | 2.642 | 0.0 | 0 | kick |
| ground_contact 贴地 | 0.0 | 0.653 | 155 | kneel_or_deep_squat |
| sitting_on_object 坐物体 | 0.0 | 2.448 | 1748 | chair |
| upper_body_manipulation 上肢操作 | 5.442 | 4.451 | 1489 | carry, lift_in_place, pick, place |
| whole_body_object 全身+物体 | 11.119 | 0.0 | 0 | carry_box, kick_ball, push_box, push_box_with_foot |
| gesture_social 手势社交 | 0.0 | 0.0 | 0 | 无 |
| idle_posture 站立/调整 | 0.0 | 0.0 | 0 | 无 |
| other 其他 | 0.0 | 0.0 | 0 | 无 |

GRAIL 行走子类(已转段): 上行 1927 段 5.375 h, 下行 932 段 2.6 h, 平地或上去再下来 3286 段, 转向>60° 3516 段, 跑(速度p95>2.2m/s) 只有 6 段。

## 人的多样性
- OmniContact: 真人, BVH 骨架校准 26 套, 身高 171.9-189.0 cm, 按身高聚类约 6 人(估计, 元数据没有演员 ID); 性别未提供。足球三种踢球各只有 1-2 套骨架。
- GRAIL: 合成角色, 无真人。所有段 betas 全为 0(一个中性身材), gender=neutral, meta 身高 1.66 m; 重建估计身高 1.60-1.87 m 只是估计噪声, 不代表体型多样。

## 还没下的
- GRAIL 人体 recon 还差 11163 个文件 ≈ 14.0 GB: pickup_table 1652 个/10.088 GB, stair_p1 4755 个/1.959 GB, stair_p2 4756 个/1.955 GB。下载循环 grail_dl_loop.py 正在跑。下完能多出约 31.2 小时(大部分是楼梯)。
- GRAIL pickup_ground 有 1527 段官方就没放 SMPL-X recon(只有 G1 轨迹和物体), 人体管线永远拿不到, 只剩 209 段 0.88 h。
- GRAIL 其他目录(不是人体, 可选): G1 轨迹 robot/ 4.53 GB, 物体 RL 后轨迹 objects/ 17.57 GB, 视频 video/ 139.4 GB。
- GRAIL 官方说 桌面操作 / 地面大物体操作 两个类别"计划后续发布", HF 上目前没有; 没有别的类别可下。
- OmniContact raw_mocap: 全部 690 段已在盘(case2_push 目录 165 个, 比 README 多 3 个)
- OmniContact npz(G1 轨迹+接触标签): g1_labels 仅 box 476/490 段(case3_slide 缺 14); 足球 200 段标签未取; 全量 npz 7.58 GB
- OmniContact npz_clips(2231 任务切片, 含箱子尺寸增广): 未下载, 6.86 GB; 是 G1 关节轨迹, 不是人体, 对 SMPL-X 管线无新增动作

## 要注意的
- GRAIL 已转的楼梯里: stair_p1 基本是下楼(931/1306 段), stair_p2 全是上楼(1261/1311); 只拿一部分会偏。已转部分是按文件名顺序下的, 未必代表全体。
- GRAIL 的 interaction_frames 在 sitting/楼梯/坡/路沿 上多半是固定的片尾 30 帧(比值 0.124), 不是真实接触区间, 只有 pickup 两类有参考价值。
- GRAIL pickup_table 人几乎不走(位移 p50 0.27 m), 是原地抓起举起, 不是搬运走路。
- GRAIL 已转里有 9.5 h 是旧版 v1 转换, 要等 v2/v3 重转; 新版里源数据有效帧(不穿地不悬空)占 12.919/14.859 h。
- OmniContact 的 qa.json 里 seconds 字段是转换耗时, 不是片长; 片长按 帧数/90 算。足球类接触类型是推断(G1 标签没下)。
- OmniContact 搬箱子里有 31 段出现骨盆下降>45 cm 的深蹲(从地上搬箱), 是否跪或手撑地没核实(没跑 FK), 先没算进 ground_contact。
- 两个来源都没有: 跑、跳、跳舞、摔倒/起身、爬、躺、手势、站立调整。dynamic 只有踢球。镜像数据为 0。
- 许可都是 CC BY-NC 4.0(非商用); GRAIL 的 pickup 物体另有 RoboCasa CC BY 4.0 署名要求。

## 文件
- summary_json: `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/summary.json`
- summary_md: `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/summary.md`
- omnicontact_clip_stats: `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/omnicontact_clip_stats.jsonl`
- grail_clip_stats: `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/grail_clip_stats.jsonl`
- hf_remaining_tree: `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/_hf_remaining_tree.json`
- 脚本: `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/scan_grail.py`, `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/scan_omnicontact.py`, `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/aggregate.py`, `/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/make_md.py`
