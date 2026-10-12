# Atlas 扩展进度（Claude 子代理，2026-10-07 起）

断网/重启后从这里接着干。工程源码开工前备份：`/home/arenalabs/motion_audit_web_backups/20261007_191235.tar.gz`。
开工时各文件 mtime 快照：scratchpad `mtimes_start.txt`（开工 19:12 PDT；之前 Codex 最后改动 18:33）。

## 目标（lloyd 要求）
1. 进度页：来源漏斗（已知总时长→已下载→已转 SMPL-X→已重定向）+ 来源×21 台机器人矩阵；后台 ≤5 分钟重扫一次，只读头。
2. GPU 输出查看器（gpu_npz_v2）：MuJoCo CPU 正向运动学 + 4096 表面点 + 骨架；四种模式（部位语义 / 接触二值 / 距离场 / 法向箭头）；13 区时间轴 + flags 位条 + qc 摘要卡；同 clip 多机器人并排（每页 ≤6），左侧人体源。
3. 人体原始 SMPL-X 浏览：列表筛选 + 骨架 + 表面点 + 13 区上色 + 界面现算地面接触/距离热力。
4. "接触是什么"说明框。

## 设计决定（已核对源码）
- `points_local` 是 **geom 坐标系**（umr_gpu/strict.py RobotPoints 用 geom_quat/geom_pos 转到 body），与现有 factory 适配器一致：world = geom_xmat·p + geom_xpos。
- `contact_idx`/`near_idx` 来自 `idx % ndense`，partner = `idx // ndense`（0 地面、1 物体）；`near_dist_q` 反量化 d = q·0.1 mm − 0.5 mm（export.py Q_STEP/Q_OFF）。
- flags 位（strict.py）：1 地面穿透、2 物体穿透、4 自穿透、8 地面接触不实、16 物体接触不实、32 打滑、64 抬离不一致。
- 人体源（GPU 输出）：problems/<clip>/<robot>.stock.cpu.npz 的 `src`（UMR 捕获的 4096 人体槽位，ZIP_STORED → 直接 mmap，不解压）减 `floor_offset_removed`；区域 = part_ids 经 SEG19_REGION。
- 物体：problems 的 `objpen_mjb`（已按 smpl_scale 缩放的物体）+ `objpen_object_geoms`；位姿 = export 的 `object_pose`（地面系，wxyz）。
- 人体 SMPL-X 前向：numpy LBS（SMPLX_NEUTRAL.npz，flat_hand_mean、手/下巴/眼置零，与 UMR 一致），≤64 帧分块；表面点用 UMR G1 对应关系的 4096 槽位（x_bind_face_ids/x_bind_bary）。

## 进度
- [x] 备份、读代码、数据摸底
- [x] 后端 `backend/npzhead.py`（npz 头读取 / ZIP_STORED 原地 mmap）
- [x] 后端 `backend/progress.py`（ProgressScanner：首次 ~3 s，后台 ≥5 分钟重扫）
- [x] 后端 `backend/gpu_adapter.py`（GpuService + GpuNpzAdapter 挂到 FactoryService）
- [x] 后端 `backend/human_smplx.py`（SmplxIndex + numpy SMPL-X，复现 UMR 捕获 src 误差 0.0005 mm）
- [x] `backend/atlas_routes.py` + app.py 两处接线；`tests/test_atlas.py` 13 项通过
  - 旧测试：49 通过 / 1 失败（`test_all_existing_npz_read...` 断言踢球 90 Hz，但 out/ 里 10-05 新增了 30 fps 的 qa_v2 踢球窗口 → 开工前就失败，数据变化导致，未改）
- 数据发现（写进报告）：GPU 导出根位置逐帧跳变（kick G1 p99 26 cm/帧、最大 81 cm/帧，人体躯干 ≤2 cm/帧；LAFAN G1 最大 27 cm/帧且 flags=0）；ground_slip_speed 最大 14–34 m/s；export.py 文档写了 object_slip 但文件没有；人体区域接触用 1 cm/2 cm 滞回（机器人 1/2 mm）；GRAIL 是 joblib pickle 不反序列化。
- [x] 前端 `frontend/src/atlas/*`（App.jsx 只加了 3 个页签 + 在 atlas 页签时不加载审阅条目/不响应审阅快捷键）
  - 进度页：来源漏斗 + HiPHI 32 卷色块 + GPU 量产运行状态（含流水线自己标记的作废引擎）+ 来源×21 台矩阵
  - GPU 输出页：人体源 + ≤6 台机器人同屏；部位语义 / 接触二值 / 距离场 / 法向箭头；质检卡；13 区时间轴 + flags 位条 + 根跳变条
  - 人体原始页：HiPHI / OmniContact 可看（GRAIL 只列清单）；界面现算地面接触/距离；逐帧拟合残差曲线
  - 地面网格改为分段线（整根长线穿过相机后方时会被 WebGL 整条丢掉，只剩远处一条带；旧审阅页有同样现象，未改旧页）
- [x] 量产 v1（production/v1/<source>/<clip>/<robot>.npz）已兼容：robot_xml/surface_asset 从 provenance 取；人体源按 provenance
  用 numpy SMPL-X 重建（与文件 human_ground_clearance_mm 核对 ≤0.03 mm）；物体用 object_xml × object_scale；
  _invalid/ 里列出的引擎版本标为"已作废"，不计入进度矩阵。
- [x] 截图 QA：qa/atlas_v1/01–07（headless Chromium + SwiftShader）
- [x] 正式模式启动（scripts/service.py start，taskset 0-15 + nice 10）

## 测试
`env -u PYTHONPATH .venv/bin/python -m pytest tests --basetemp=qa/pytest-current -q`（PYTHONPATH 里的 ROS 插件会干扰 pytest，必须去掉）。
结果：tests/test_atlas.py 15 项全过；旧测试 51 过 / 1 失败（开工前就失败，见上）。前端 `node --test` 10/10。

## 已知限制
- GRAIL 的 SMPL-X 是 joblib pickle，后端不反序列化 → 只列清单不可视化。
- GPU 评测导出（results/export）的人体源来自 UMR 捕获 src；没有捕获且不是量产格式的文件不显示人体。
- "法向箭头"只是几何代理；人体法向只有人体原始页有（GPU 页人体源只有槽位点，没有网格法线）。
- 距离场 q=0 表示 ≤ −0.5 mm（量化下限，穿透被截断），按穿透色显示。
