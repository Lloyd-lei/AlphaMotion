# 给 Codex 的任务：本地动作数据在线预览 + 人工审计网页

## 背景
我（lloyd）在做一个跨具身人形机器人动作数据工厂：用 UMR（https://github.com/hanyang9/UMR）把人体动作（HiPHI、OmniContact、GRAIL 等，SMPL-X 格式）重定向到约 21 台人形机器人上，并用稠密表面点云给每帧打严格的接触标签（地面接触、物体接触、打滑、穿透）。数据清洗由另一个 AI 助手（Claude）负责跑流水线。你负责写一个**网页程序**，让我在浏览器里直接读本机的数据，逐条在线预览、检查，并做**进入训练前的人工质检判决**。你的判决要按约定写回文件，Claude 的流水线会读它，决定哪些数据进训练、哪些回炉。

## 必读（开工前先读完）
1. 交接约定：`/home/arenalabs/umr_factory_data/audit/SCHEMA.md`（待审队列、数据格式、判决格式、原因表）。这是你和 Claude 之间唯一的接口，严格按它实现；看到不认识的 `format` 版本要在界面上明确报错，不要猜。
2. 数据标准：`/home/arenalabs/umr_factory_data/DATA_SPEC_v2.md`（什么叫"实接触"、穿透和打滑阈值）。
3. 工厂说明：`/home/arenalabs/umr_factory_code/README.md`（输出 npz 的字段、机器人资产、标签含义）。
4. 实际样本：`/home/arenalabs/umr_factory_data/out/*/*.npz`（目前有跳舞片段和踢球片段 × G1/GR3/H2）。用 `np.load(..., allow_pickle=True).files` 看真实字段，以真实文件为准。

## 数据在哪里（只读）
- 每条数据（片段 × 机器人）：`/home/arenalabs/umr_factory_data/out/<片段目录>/<机器人>.npz`，主要字段：
  - `qpos`（T×nq，严格修正后的骨架：根位置 + 根四元数 wxyz + 关节角）、`qpos_umr`（UMR 原始输出，用于对比）、`fps`、`floor_height`、`floor_offset_removed`、`joint_names`、`robot_xml`；
  - 区域级标签（13 个区域，名字在 `regions`）：`ground_contact`、`ground_clearance`、`ground_slip`、`ground_slip_speed`、`ground_contact_required`，物体同理 `object_*`；
  - 点级接触：`ground_point_contact` / `object_point_contact`（每帧 512 字节 = 4096 个表面点按位打包，第 i 位对应 `surface.npz` 的第 i 个点）；
  - 穿透：`pen_ground_m`、`pen_object_m`、`pen_self_m`；质检：`flags`（位：1 地面穿透，2 物体穿透，4 自身穿透，8 地面应接触未接触，16 物体应接触未接触）、`flags_initial`；
  - 人体：`human_ground_contact`、`human_object_contact`、`human_ground_clearance` 等；`provenance`（JSON 字符串）里的 `human_slots` 指向逐帧人体表面点（`source_slots`，4096 点，要减去 `floor_offset_removed`），`source_data` 指向 SMPL-X 参数；
  - 物体：`object_pos`（T×3）、`object_rot`（T×3×3）、`object_name`、`object_xml`（含网格路径）。
- 机器人静态资产：`/home/arenalabs/umr_factory_data/robots/<robot>/<robot>.xml` + `mesh/`；`surface.npz`（`points_local`、`normals_local` 在所属 geom 局部系，`geom`、`body`、`region`、`regions`），`staging.json`（T-pose、碰撞排除对）。
- 待审队列：`/home/arenalabs/umr_factory_data/audit/queue.jsonl`（Claude 会往里追加；现在可能是空的，你可以先扫描 `out/` 目录自动生成一个本地视图，但要标注"未入队"）。
- 将来还有 GPU 引擎输出（`gpu_npz_v2`）和打包后的训练分片（`/home/arenalabs/umr_dataset`，`packed_v1`）。请把读取做成可插拔的"数据源适配器"，先实现 `factory_npz_v1`，给另外两个留接口。

## 要做的功能
### A. 预览（同步时间轴）
1. 三维视图（three.js / WebGL），可以：
   - 并排显示 **人体（表面点云，可选 SMPL-X 网格）| 机器人（MJCF 视觉网格，由 qpos 正向运动学摆姿态）**，以及可切换的"UMR 原始 vs 严格修正后"对比；
   - 机器人表面 4096 点云叠加显示，按逐帧标签上色：地面接触=绿，物体接触=橙，穿透（离地/离物体 < -0.5 mm）=红，"人在接触但机器人悬空 1–20 mm"=黄，其它=灰；打滑的接触点画速度箭头；
   - 骨架连线（关节点 + 父子连线）；物体网格按逐帧位姿摆放；地面网格以校准后地面为 z=0；
   - 视角：自由环绕、正视/侧视/俯视、跟随根节点、脚部特写。
2. 时间轴面板：播放/暂停/逐帧/变速/循环区间；下方画 13 个区域的接触条带（人体 vs 机器人，地面/物体分开），区域间隙曲线、打滑速度曲线、每帧最深穿透、flags 标记；点任意位置跳帧，和三维视图同步。
3. 多机器人对照：同一片段在所有机器人上的网格并排（例如 4×6 宫格），同步播放，方便发现某台机器人的系统性问题。
4. 数据浏览：按来源 / 机器人 / 动作类别 / 自动质检指标筛选和排序；显示每条的自动质检摘要（`auto_qa`）。

### B. 人工审计（核心）
1. 审计模式：按 `queue.jsonl` 的 `priority` 从高到低逐条推送，也能切到"随机抽样"队列；我看完后用快捷键判决：通过 / 拒绝 / 需返修，可多选原因（原因表见 SCHEMA.md），可写备注，可用 I/O 键标出有问题的帧段（只拒绝这一段）。
2. 判决写入 `/home/arenalabs/umr_factory_data/audit/decisions.jsonl`，格式严格按 SCHEMA.md，**必须带当时看到的 npz 文件的 sha256**；只追加，不改写。
3. 系统性问题（"这台机器人所有片段左手都穿模"）可以一键提交到 `issues.jsonl`，附上例子。
4. 进度看板：已审/待审数量，每台机器人、每个来源的通过率，随机抽样的合格率（带置信区间），常见拒绝原因排行。
5. 已判决条目：如果 npz 的哈希变了（被重新生成），界面要提示"判决已过期，需要重审"。

## 硬性约束
- **只读**：除了 `audit/decisions.jsonl`、`audit/issues.jsonl` 和你自己的项目目录，不准写、改、删任何数据文件；不准动 `out/ runs/ robots/ corr/ smplx/ raw/` 和任何正在运行的进程。
- 你的代码放在 `/home/arenalabs/motion_audit_web/`。后端建议 Python（FastAPI），读 npz 和 MJCF；Python 环境用 `/var/tmp/alphamotion/umr_factory/umr/env/bin/python`（已有 numpy、mujoco 3.3.7、trimesh；如需新包请建独立 venv，不要改这个环境）。正向运动学可以在后端用 MuJoCo 算出每帧各视觉 geom 的世界位姿发给前端，前端用 `surface.npz` 的局部点 × geom 位姿得到点云，不要逐帧传 4096×3 坐标。
- 性能：大片段（几千到上万帧）要分块按需加载、二进制传输（float16/uint8），拖动时间轴要流畅；单个后端进程内存 < 4 GB；不要用 GPU 做计算（只用浏览器的 WebGL 渲染）。
- 服务只监听 `127.0.0.1`（不要 0.0.0.0），端口用 8765（避开已占用的 2334、8899、8931）；启动脚本要有明确的停止方法，进程要有寿命（例如空闲 8 小时自动退出），不要留僵尸进程。
- 界面文字用中文。

## 和 Claude 的配合方式
- 接口只有 `audit/` 目录下的文件，版本以 `SCHEMA.md` 为准；Claude 新增数据格式（如 `gpu_npz_v2`、`packed_v1`）时会在 SCHEMA.md 追加说明，你的适配器按版本号读取。
- 你发现数据本身的问题（字段缺失、哈希对不上、标签和三维几何对不上，比如标"接触"的点离地 5 mm），在界面上显示出来，并在你的 README 里列一个"数据问题清单"，Claude 会据此修流水线。
- 不要自己"修"数据或重算标签；看到的就是要审的。可以在界面上临时显示你独立算的几何量（例如某点到地面的距离）用于核对，但要标明"界面现算"，不能写回数据。

## 交付和验收
1. `/home/arenalabs/motion_audit_web/README.md`：怎么启动、停止、快捷键、架构、已知限制。
2. 用现有的踢球片段（G1、GR3、H2）和跳舞片段跑通：三维视图、点云上色、时间轴、物体显示、审计判决写入都要能用；附几张截图。
3. 自动化测试：适配器读取所有现有 npz 不报错；位解包后的接触点数和 npz 里区域标签一致；FK 姿态和 MuJoCo 直接算的结果一致（误差 < 0.1 mm）；判决文件格式校验。
4. 写一个小脚本 `check_decisions.py`，校验 `decisions.jsonl` 每行都符合 SCHEMA.md。
