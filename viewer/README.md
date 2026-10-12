# 动作人工审计工作台

Agent 入口见 [MCP 工具链](docs/MCP.md)：共用数据与几何后端，提供原生图像、指标及后台任务接口。尚未接入的训练等工作流明确显示缺口。

在浏览器中并排审阅人体表面点、真实 MJCF 机器人及交互物体，比较 UMR 原始姿态与严格修正姿态，并将人工决定按 `audit/SCHEMA.md` 追加给数据工厂。这里展示的是运动学回放，不是物理仿真或真机验证。

## 启动、访问和停止

工程在 ArenaLabs 的 `/home/arenalabs/motion_audit_web/`。后端使用工程内独立 `.venv`；只读继承现有 UMR 环境的 NumPy、MuJoCo、trimesh、SciPy，新增包仅安装在本工程。版本快照见 `qa/environment-freeze.txt`。

```bash
cd /home/arenalabs/motion_audit_web
.venv/bin/python scripts/service.py start
.venv/bin/python scripts/service.py status
.venv/bin/python scripts/service.py stop
```

只监听 `127.0.0.1:8765`。在自己的电脑建立 SSH 转发后访问：

```bash
ssh -N -L 127.0.0.1:18765:127.0.0.1:8765 arenalabs
```

浏览器打开 `http://127.0.0.1:18765`。终端 Ctrl+C 关闭该 SSH 转发。服务在连续 8 小时没有请求后自动退出。启动脚本拒绝抢占已有端口；停止前同时核对 PID、Linux boot ID、`/proc` 启动标识、完整命令行与工作目录，只向匹配的本工程进程发送 TERM。详细记录在 `logs/service.json` 和 `logs/server.log`。

测试模式使用相同端口，需要先停止本工程已有实例：

```bash
.venv/bin/python scripts/service.py stop
.venv/bin/python scripts/service.py start --mode test
```

界面明确显示测试模式，判决只写 `qa/fixture_audit/`。完成 QA 后停止测试实例，再按默认方式启动正式模式。正式模式仅在人工提交时追加 `/home/arenalabs/umr_factory_data/audit/decisions.jsonl` 或 `issues.jsonl`。

## 使用与证据边界

左侧按来源、机器人、动作、自动质检指标或审阅状态筛选，优先级来自队列原值。队列尚未包含的扫描条目显示“未入队”。随机浏览只是打乱列表；进度看板的随机抽样置信区间只纳入上游队列明确标记 `sample_kind=random`、哈希仍有效的整条人工判决。未审条目及仅帧段判决不进入合格率分母，不能据此声称全量数据总体无偏。

使用视角按钮、表面点开关、骨架及原始/修正切换核对动作；时间轴与三维同步。I/O 标出问题帧段，采用左闭右开 `[start,end)`；空帧段列表代表整条。提交前填写审阅人、判决、原因和必要备注。快捷键：空格播放/暂停，左右箭头逐帧，A/R/F 选择通过/拒绝/需返修，Ctrl+Enter 提交；I/O 标记半开帧段。“跳至帧”可直接输入精确帧号，“下一条待审”按当前筛选和排序推进。

每次预览绑定当时的 NPZ SHA-256。源文件、伴随人体数据或资产在读取期间变化时拒绝继续使用旧快照；提交还会在持有日志锁后再次读取文件哈希。哈希改变的历史判决保留原记录并标为过期。提交落盘后若检测到并发数据变化，会明确返回已追加但过期，不能自动重试或悄悄覆盖旧行。

点颜色：绿色为文件地面接触，橙色为文件物体接触，红色为界面现算穿透超过 0.5 mm，黄色为人体接触而机器人悬空 1–20 mm，其它灰色。独立几何量、速度箭头标为“界面现算”，不修改任何源标签。UMR 原始姿态下，保存的接触/间隙/flags 仍属于修正后姿态，界面提示这一限制。

## 数据图谱页（转换进度 / GPU 输出 / 人体原始）

顶部新增三个页签，全部只读：「转换进度」（来源漏斗、HiPHI 卷状态、GPU 量产状态、来源×机器人矩阵，后台最多 5 分钟重扫一次、只读文件头）；「GPU 输出」（`gpu_npz_v2`：umr_gpu 评测导出与 `production/v1` 量产文件，人体源 + 机器人点云 + 骨架，部位语义 / 接触二值 / 距离场 / 法向箭头四种显示，13 区时间轴与 flags 位条）；「人体原始」（SMPL-X 片段列表与 CPU 现算点云）。字段口径、数据问题与进度见 [docs/ATLAS_PROGRESS.md](docs/ATLAS_PROGRESS.md)。显示的接触是逐点二值 + 2 cm 内距离场 + 13 区标签，不是力或压力。

## 工程结构

- `backend/adapters.py`：按版本读取工厂 NPZ；CPU 分块、哈希快照、只读内存映射与队列扫描。
- `backend/geometry.py`：真实 MJCF 视觉几何、MuJoCo 正向运动学、点到地面及物体的独立核对量。
- `backend/wire.py`：带形状和类型描述的二进制协议。位置/旋转使用 float32、标签使用 uint8，避免 float16 绝对米坐标的量化误差影响亚毫米判定。
- `backend/audit.py`：严格 schema、半开帧段、追加锁与 fsync、过期判决和 Wilson 95% 区间。
- `backend/app.py`：FastAPI 接口、同源提交检查、串行有界 CPU 工作、静态前端服务。
- `frontend/`：React + Vite + Three.js 工作台；运行时使用源资产，设计概念图不作为机器人替身。
- `check_decisions.py`：只读校验日志每行格式，可选核对当前文件哈希。
- `qa/`：测试、源文件基线、资源用量和浏览器截图；`docs/inputs/` 保存开工时规范副本。

所有解压 NPY 缓存在本工程 `cache/arrays/`，从 NPZ 以最多 1 MiB 缓冲流式解压，读取采用 mmap。每个动态块最多 64 帧，机器人传 geom 位姿与静态局部表面点，浏览器做逐点变换；人体点云按帧块传输。后端禁止 GPU，线程数限制为 1，重计算串行，模型缓存有上限。

## 数据问题清单与已知限制

1. 踢球文件实际约 90 Hz，舞蹈为 30 Hz，搬运约 30 Hz。界面按真实 `fps` 播放，不能把所有文件当作规范声明的 30 Hz。
2. 源 provenance 的接触滞回为进入 1 cm / 离开 2 cm、打滑阈值 3 cm/s，与 DATA_SPEC_v2 的 1/2 mm 和 1 cm/s 不同。严格修正、原始区域标签与界面核对量分别显示，不冒称标签已经符合新规范。
3. SCHEMA.md 示例中的 `penetration` 不在正式原因枚举中；本工程采用明确的 `penetration_ground`、`penetration_object`、`self_collision` 等原因表。
4. 最初 7 条样本存在 923 / 619,593 个区域帧差异，全部为区域已标接触但无打包接触点，反向差异为 0。其中地面 26、物体 897（G1 踢球物体 860）。不能仅凭这些差异判定原因；采样、阈值定义、几何和资产版本均需上游确认。执行期间新增的 G1-23DoF 舞蹈、足球、搬箱已独立读通；固定截至 2026-10-05 07:06:34 UTC 的 10 条快照合计 1,115 / 650,793（地面 185、物体 930）。之后新增数据不计入本次验收快照。逐文件证据见 `docs/data_findings.md` 与 `qa/contact-denominators.json`。
5. `gpu_npz_v2` 和 `packed_v1` 已留适配器接口，遇到时明确报不支持，不能猜测字段。
6. 人体必需的源点云可视化已经纳入设计；SMPL-X 三角网格属于可选扩展。界面现算物体距离存在算法范围限制，详见对应数据诊断说明。
7. 扫描条目的临时 item_id 带 `scan/`；上游后续入队时须使用约定 item_id，界面不会改写队列或把不同 ID 的判决自动合并。

## 播放与显示限制

机器人对照每页最多 6 台，通过分页支持更多源机器人，避免创建 21 个 WebGL 上下文。当前实际核验踢球 3 台和舞蹈 3 台；不为不存在的输出创建虚拟条目。

暂停定位采用单帧请求；播放采用 32 帧块和下一块有界预取，每视图缓存最多 4 块 / 64 MiB。缓冲时保留上一张真实帧，显示请求帧和实际显示帧，不插值伪造源动作。优化后 G1 踢球每 32 帧约 2.30 MB，压缩约 1.96 MB；本机后端计算约 0.033 秒，但当前 SSH 广域链路仍不足以持续传输 90 Hz 全量人体点云。网页不会把缓冲后的播放冒称实时。

## 验证

所有自动写入测试只用工程内隔离样本。执行：

```bash
cd /home/arenalabs/motion_audit_web
.venv/bin/python -m pytest tests --basetemp=qa/pytest-current -q
.venv/bin/python check_decisions.py --help
```

具体已验证结果、浏览器访问地址、截图和保留限制见 `qa/acceptance.md`。软件测试通过与源数据满足训练标准分别记录；这个网页的完成不代表数据已被人工接受，也不代表训练、仿真或真机验证完成。
