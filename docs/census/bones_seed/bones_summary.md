# BONES-SEED 丰富度盘点

- 片段 142,220（原始 71,132 / 镜像 71,088），120 fps
- 小时：原始 **144.22 h** / 镜像 144.11 h / 合计 288.32 h；片段中位 5.71 s, p90 12.62 s
- 演员 522 人（{'M': 269, 'F': 253}），身高 145–199 cm（p10 167.0 / 中位 174.0 / p90 187.0），体重 38–145 kg，年龄 17–71；职业 {'actor': 380, 'general': 64, 'professional': 43, 'dancer': 24, 'stuntman': 11}
- 风格（原始小时）：{'neutral': 130.07, 'injured torso': 6.85, 'injured leg': 6.51, 'hurry': 0.55, 'hurry to neutral': 0.1, 'injured torso to neutral': 0.07, 'injured leg to neutral': 0.05, 'old': 0.02}

## 分类（原始，不含镜像；镜像分布几乎一样）

| 类别 | 主类小时 | 占比% | 主类片段数 | 多标签小时(会重复) |
|---|---|---|---|---|
| 移动 (locomotion) | 42.48 | 29.46 | 21,178 | 57.77 |
| 动态 (dynamic) | 28.30 | 19.62 | 14,838 | 30.03 |
| 地面接触 (ground_contact) | 11.07 | 7.67 | 5,368 | 11.13 |
| 坐物体 (sitting_on_object) | 2.83 | 1.96 | 936 | 3.36 |
| 上身操作 (upper_body_manipulation) | 16.43 | 11.39 | 7,305 | 22.33 |
| 全身+物体 (whole_body_object) | 11.01 | 7.63 | 5,865 | 11.01 |
| 手势社交 (gesture_social) | 17.69 | 12.27 | 10,953 | 22.63 |
| 站姿/调整 (idle_posture) | 14.41 | 9.99 | 4,689 | 41.16 |
| 其他 (other) | 0.00 | 0.00 | 0 | 0.00 |

## 接触类型（原始小时，一段可多种）

| 只脚 | 手撑地 | 膝 | 臀 | 背 | 手-物体 | 脚-物体/台阶 |
|---|---|---|---|---|---|---|
| 103.77 | 4.72 | 6.26 | 8.39 | 2.82 | 24.62 | 3.23 |

## 物体

- 有道具的片段 29.68 h（其中被手拿/搬的 25.72 h）；**没有物体位姿/网格**：道具只在文字字段 content_props 里，没有物体 6DoF 位姿/网格；物体接触标签无法用数据工厂的尺子量
- 道具类别小时：box_crate_generic 13.64, terrain_env 13.27, furniture 9.71, articulated_fixture 3.85, tool 2.82, assistive 1.44, instrument 0.97, handheld_device 0.93, kitchenware_food 0.91, person_or_doll 0.89, ball_sport_gear 0.25
- 前 25 道具：crate 2.42, 50cm box 1.76, door 1.33, chair 1.11, medium big light object; shelves (scaffolding) 1.09, big heavy object; shelves (scaffolding) 1.08, big light object; shelves (scaffolding) 0.96, small heavy object; shelves (scaffolding) 0.96, baby (sandbag) 0.77, medium heavy object; shelves (scaffolding) 0.75, small item 0.75, medium big heavy object; shelves (scaffolding) 0.74, medium big light object 0.68, wall 0.66, small obstacle 0.59, crutch 0.57, flute 0.57, crutches 0.52, small light object; shelves (scaffolding) 0.5, watering can 0.47, cellphone 0.44, medium light object; shelves (scaffolding) 0.41, fridge; small light object 0.41, chair; table 0.4, cigarette 0.35

## 许可 / 状态 / 进工厂

- 许可：仅学术用户或年营收<100万美元初创公司内部研发可用；禁止再分发/公开镜像；Results 不得能还原原始数据；§6 禁止用来训练'主要用途是生成人体动作/动画数据、可替代本数据集'的生成模型（明确允许机器人控制策略/VLA/行为克隆/动作跟踪）。AlphaMotion 若定位为'动作生成基础模型'，此条需法务确认。
- 本地：仅元数据（parquet/csv/temporal_labels v002）+ 10 条样例 BVH；SOMA BVH 全量包 soma_uniform.tar.gz(45.19 GB) 不在本机
- 下载器：soma_uniform.tar.gz 排在 follow-up 队列(PID 3272054, runs=[])，还没开始；主下载器(PID 3258020)卡在 UniMate 模型权重阶段 39.5/45.9 GB，已 26.4 h 无进展（queue status=primary_stall_attention）
- SMPL-X：不需要（UMR 直接读 SOMA BVH + MHR 体型）；已转 0 条
- 重定向：厂商自带 G1 CSV（1 台，/var/tmp/arena/raw/bones_seed/g1.tar.gz 23.5 GB，已解压 4 条）；UMR 0 条/0 台；另有 8 月旧缓存 topo_exp/cache_bones_body（500 段 soma_body 关节, 30fps, 非 UMR）
- 进 UMR：下载 soma_uniform.tar.gz(45.2 GB)+soma_shapes+SOMA-X 资产 → UMR defaults_bones_seed.json/批处理适配器直接跑，无需格式转换；镜像是独立 BVH 可直接重定向；29.7 h 道具片段没有物体位姿，只能当'纯身体'数据或补物体标注
