# UniML3D / Mixamo 丰富度盘点

| 来源/体型 | 片段 | 小时 |
|---|---|---|
| mixamo/bipedal | 2,317 | 2.41 |
| objaverse/articulated_rigid | 172 | 0.14 |
| objaverse/avian | 223 | 0.22 |
| objaverse/bipedal | 9,094 | 13.49 |
| objaverse/insectoid | 159 | 0.16 |
| objaverse/marine | 59 | 0.05 |
| objaverse/quadrupedal | 623 | 1.00 |
| objaverse/serpentine | 13 | 0.01 |
| objaverse/uncertain | 12 | 0.01 |
| truebones/avian | 116 | 0.11 |
| truebones/bipedal | 159 | 0.18 |
| truebones/insectoid | 215 | 0.17 |
| truebones/marine | 19 | 0.01 |
| truebones/quadrupedal | 558 | 0.53 |
| truebones/serpentine | 30 | 0.04 |

## Mixamo（唯一干净的人形部分）

- 2,317 段，**2.408 h**，30 fps；UniMate 训练子集 2,162 段 / 105.04 分钟
- 单一 65 关节 Mixamo rig（UniML3D 只保留 22 关节人形核心）；无演员信息
- 名字里带道具（枪/剑/公文包/火把…）319 段 / 0.241 h，但道具不在动作数据里

| 类别 | 主类小时 | 占比% | 主类片段数 | 多标签小时(会重复) |
|---|---|---|---|---|
| 移动 (locomotion) | 0.34 | 14.04 | 563 | 0.69 |
| 动态 (dynamic) | 0.78 | 32.32 | 701 | 0.92 |
| 地面接触 (ground_contact) | 0.29 | 11.87 | 262 | 0.29 |
| 坐物体 (sitting_on_object) | 0.19 | 7.86 | 123 | 0.19 |
| 上身操作 (upper_body_manipulation) | 0.25 | 10.45 | 209 | 0.46 |
| 全身+物体 (whole_body_object) | 0.04 | 1.56 | 50 | 0.04 |
| 手势社交 (gesture_social) | 0.13 | 5.48 | 98 | 0.33 |
| 站姿/调整 (idle_posture) | 0.34 | 14.01 | 252 | 1.20 |
| 其他 (other) | 0.06 | 2.41 | 59 | 0.06 |

| 只脚 | 手撑地 | 膝 | 臀 | 背 | 手-物体 | 脚-物体/台阶 |
|---|---|---|---|---|---|---|
| 1.52 | 0.09 | 0.07 | 0.32 | 0.10 | 0.48 | 0.03 |

## Objaverse 'bipedal'（未验证是否人形）

- 9,094 段 / 13.486 h / 6,614 种骨架（关节 {'min': 3, 'median': 41.0, 'max': 116}）；未验证：'bipedal' 是 VLM 判的体型类，包含卡通/机器人/怪物；27% 动画名是 mixamo.com（Mixamo 动作套在任意角色上，和 Mixamo 重复）
- 主类小时：移动 2.574, 动态 4.444, 地面接触 0.339, 坐物体 0.166, 上身操作 0.117, 全身+物体 0.003, 手势社交 0.242, 站姿/调整 3.999, 其他 1.601
- Truebones 'bipedal' 是霸王龙/迅猛龙/鸵鸟等 8 种(0.18 h)，不是人形；原始片段需购买

- 许可：Mixamo Adobe Mixamo 条款（动作可用于项目，数据集再分发/训练授权不明确 → 有风险）；Objaverse 逐物体继承 Objaverse-XL 上游许可，无统一许可；描述 ODC-BY 1.0
- 本地：元数据(clips.csv/skeletons.csv/README)+HF 缓存冒烟样本: Mixamo 特征 npz 129/2,162, Objaverse 90；reload 导出 2 段 + 1.3 GB GLB 样本；全量数据未下载（主下载器在下 UniMate 模型权重, 非数据）
- 进 UMR：Mixamo FBX → BVH(Blender) → UMR 通用 FBX/BVH(NR) 适配器（未验证）；或用 UniML3D 特征(22 关节全局位置+局部旋转@30fps)做 SMPL-X 拟合；可用的人形部分只有 Mixamo 2.4 h，Objaverse 'bipedal' 需人工/自动筛人形
