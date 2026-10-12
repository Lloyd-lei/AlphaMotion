# 其他候选源（只看可得性/许可，未在本机核实）

| 源 | 时长 | 内容 | 物体 | 格式 | 获取 | 许可 | 本机 | 进 UMR | 预计主类 |
|---|---|---|---|---|---|---|---|---|---|
| OMOMO | 约 10 h（论文口径, 未实测） | 单人搬运/推拉 15 种物体（箱子/椅子/桌子/拖把/台灯…）, 17 人 | 有：物体 6DoF + 15 个网格 | SMPL-X/SMPL-H + 物体位姿 | Google Drive 公开链接 | 代码 MIT；数据无明示许可 → 按保留所有权利对待，风险高 | 否（只有 UMR 样例 README） | 有 UMR 适配器 (hsi_hoi)，需整理成 UMR 目录；女性序列需 SMPLX_FEMALE（本机缺） | whole_body_object 为主 + upper_body_manipulation |
| GRAB | 约 3.8 h（AMASS 内 GRAB 身体 3.76 h 已在 xwbc 数据里） | 10 人抓取 51 种小物体 | 原版有：物体 6DoF + 网格 + MANO 手；xwbc 版没有 | SMPL-X + MANO + 物体 | MPI 账号注册 | GRAB 非商业科研许可（MPI） | 仅身体（xwbc 派生） | 下原版 GRAB → hsi_hoi 标准适配（未验证） | upper_body_manipulation（站着/坐着抓小物体） |
| ARCTIC | 约 1 h 量级（339 段, 未实测） | 10 人双手操作 11 种带关节物体（剪刀/笔记本/微波炉…） | 有：带关节物体 + 网格 | SMPL-X + MANO + 物体关节角 | 注册下载 | 非商业科研（MPI 风格） | 否 | 需写适配器（物体是关节体） | upper_body_manipulation（几乎全是上半身） |
| CORE4D | 未公布小时；Real 约 1.0K 段 + Synthetic 约 10K 段（未实测） | 两人协作搬/移物体 | 有：物体网格 + 位姿 | SMPL-X + 物体 | HF 公开（34 GB 动作 zip） | HF 标 MIT（未对照论文/代码核实） | 否 | SMPL-X 可读；两人场景需拆成单人 + 物体 | whole_body_object（双人搬大件） |
| 100STYLE | 约 4M 帧 @60fps ≈ 18 h（论文口径, 未实测） | 1 名演员 100 种风格行走/跑/侧移/后退/站立 | 无 | 原版 BVH；HF 有 100style_smplx (2.5 GB) | Zenodo / HF | 原版 CC BY 4.0（可商用可分发, 需署名）；HF smplx 版无许可标签 | 否（只有 G1 重定向版 any4hdmi-g1-100style） | BVH→SMPL-X 或直接用 100style_smplx | locomotion（风格多样性最强, 单演员） |
| Motion-X | 约 144 h（81.1K 段, 15.6M 帧, 公布口径） | 网络视频估计的全身+手+脸 SMPL-X，含大量日常/体育/舞蹈/功夫 | 无 | SMPL-X（视频估计, 抖动/滑步多） | 申请表 + 部分子集需从原站下 | 非商业科研, 各子集许可不同 | 否（data/motionx 为空） | SMPL-X 直接读，但质量需先过物理/接触门 | dynamic/gesture/idle 多, 物体/地面接触少且不可信 |
