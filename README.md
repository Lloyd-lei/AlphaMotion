# surface-retargeting

把人体动作按"表面接触"重定向到人形机器人（UMR 路线）的全部代码，收在一个仓库里。
以前散在 8 个目录、没有 git；2026-10-12 复制进来（旧目录一个没动）。

## 流程和目录

| 步 | 做什么 | 代码 |
|---|---|---|
| ① | 下载数据、把动捕转成 SMPL-X 人体 | `surface_retargeting/ingest/`（CPU 版、GPU 版、GRAIL、HiPHI、UniML3D、BONES） |
| ② | 机器人暂存、4096 个表面点、登记表 | `surface_retargeting/robots/`，配置在 `configs/robots/`、`configs/umr_configs/` |
| ③ | 人体 ↔ 机器人表面对应关系 | `surface_retargeting/correspondence/` |
| ④ | 重定向：CPU 参考版（包一层 UMR）和 GPU 引擎 | `surface_retargeting/retarget/`、`surface_retargeting/gpu/` |
| ⑤ | 严格后处理（不穿透、不滑）和接触标签 | `surface_retargeting/strict/`、`surface_retargeting/labels/` |
| ⑥ | 判官（独立的 CPU 尺子） | `surface_retargeting/qa/` |
| ⑦ | 打包成训练数据、读取 | `surface_retargeting/dataset/` |
| ⑧ | 审计网页 + MCP | `viewer/` |

其他：`scripts/`（按步骤编号的命令）、`production/`（量产调度）、`ops/`（节流守护、下载启动）、
`tests/`、`experiments/`、`archive/`（旧版本只读快照）、`docs/`、`env/`（环境锁文件）。
上游代码用子模块固定版本：`third_party/UMR`@d1cdaed、`third_party/hiphi2smplx`@65e5131。

## 路径配置

所有跟机器有关的路径只在一处：`configs/sr.example.toml`（本机的值）。换机器就复制成 `configs/sr.toml` 改，
或者用环境变量覆盖：`SR_DATA_ROOT`、`SR_WORK`、`SR_UMR_ROOT`、`SR_BODY_MODELS`、`SR_ROBOT_ASSETS_ROOT`、`SR_PYTHON_ENV`。
看当前生效的值：`python surface_retargeting/config.py`。

## 安装和测试

```bash
git submodule update --init          # 拉上游 UMR 和 hiphi2smplx
pip install -e . --no-deps           # 装进 UMR 环境（云上）；本机不往共享环境装，改用 source ops/sr_env.sh
# 本机 CPU 冒烟（不用 GPU）：
python -m venv --system-site-packages .venv   # 用 UMR 环境的 python 建
.venv/bin/pip install --no-deps -e . && .venv/bin/pip install pytest
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests
```

GPU 测试（`tests/gpu`、`tests/convert`）只在云上跑。

## 不在仓库里的东西

数据输出、缓存、虚拟环境、视频、SMPL-X 和 beta_fit 模型（授权不许再分发）、厂商机器人网格。
迁移细节、还写死的路径、测试结果见 `docs/MIGRATION.md`。判官冻结版是标签 `ruler-v0`。
