# 迁移记录（2026-10-12，方案第 1–4 步）

方案：`docs/platform_design/SURFACE_RETARGETING_PLAN.md`（lloyd 已批）。
原则：**只复制，不动旧目录**。旧入口脚本还在原处跑，切换到新仓库是第 5 步以后的事。

## 1. 提交和标签

| 提交 | 内容 |
|---|---|
| `ruler-v0`（标签） | 判官冻结版：`umrf` 的 `.bak_taskB` 版本 + 没改过的模块，放到新目录。和 `~/umr_gpu/eval/factory_ruler_snapshot` 逐字节一致（只改了 import 行）。见 `docs/RULER_V0.md` |
| 第 2 个提交 | 10-11 的全部现状代码 + 环境锁 + 两个子模块。只改了 import 行（`umrf.*` / `umr_gpu.*` / `umr_dataset.*` → `surface_retargeting.*`） |
| 第 3 个提交 | `config.py` + `pyproject.toml`、写死路径改配置、去掉 `sys.path` 补丁、CPU 冒烟测试 |

子模块（`.gitmodules` 用 SSH 地址，本地是从旧 checkout 浅克隆的，没联网）：
- `third_party/UMR` @ `d1cdaed`（上游 `hanyang9/UMR`）
- `third_party/hiphi2smplx` @ `65e5131`（上游 `noitom-robotics/hiphi2smplx`）

## 2. 旧路径 → 新路径

| 旧 | 新 |
|---|---|
| `~/umr_gpu/umr_gpu/*.py` | `surface_retargeting/gpu/`；其中 `strict.py`、`strict_fp.py` → `surface_retargeting/strict/`，`export.py` → `surface_retargeting/export.py` |
| `~/umr_gpu/eval/*.py` | `surface_retargeting/qa/`（`factory_ruler_snapshot` 不收，用标签 `ruler-v0`） |
| `~/umr_gpu/capture/` | `scripts/capture/` |
| `~/umr_gpu/scripts/` | `scripts/gpu/` |
| `~/umr_gpu/production/` | `production/` |
| `~/umr_gpu/tests/` | `tests/gpu/`（GPU 脚本，本机不收集） |
| `~/umr_gpu/README.md` | `docs/gpu_engine.md` + `docs/CHANGELOG.md`（"Progress log" 一节） |
| `~/umr_factory_code/umrf/robots.py`、`surface.py` | `surface_retargeting/robots/` |
| `~/umr_factory_code/umrf/shared_corr.py` | `surface_retargeting/correspondence/` |
| `~/umr_factory_code/umrf/pipeline.py` | `surface_retargeting/retarget/pipeline.py` |
| `~/umr_factory_code/umrf/{postpass,geometry,objects}.py` | `surface_retargeting/strict/` |
| `~/umr_factory_code/umrf/labels.py` | `surface_retargeting/labels/labels.py` |
| `~/umr_factory_code/umrf/paths.py` | `surface_retargeting/paths.py`（现在只是 `config.py` 的薄壳） |
| `~/umr_factory_code/scripts/` | `scripts/00_stage_robots.py`、`01_build_surfaces.py`、`02_build_correspondence.py`、`02b_build_shared_corr.py`、`03_run_clip.py`、`04_qa_summary.py`；两个渲染脚本 → `scripts/render/` |
| `~/umr_factory_code/tests/` | `tests/strict/` |
| `~/umr_factory_code/README.md` | `docs/factory_cpu.md` |
| `/var/tmp/.../umr/tools/{umr_pipe,umr_run}.py` | `surface_retargeting/retarget/umr_wrappers/` |
| `/var/tmp/.../umr/tools/{gr3_tpose.py,gr3_tpose.json,tpose_ratio.py}` | `surface_retargeting/robots/tools/` |
| `/var/tmp/.../umr/tools/` 其余 4 个评测小工具 | `archive/umr_scout_tools/` |
| `/var/tmp/.../umr/configs/defaults_*.json`（10-04 的 5 份对照） | `configs/ablations/*.json`（只留和上游默认值不同的项） |
| `/var/tmp/.../robots/robots.json` | `configs/robots/registry.json`（路径换成 `${SR_*}` 记号） |
| `/var/tmp/.../robots/robots_table.md` | `configs/robots/robots_table.md` |
| `/var/tmp/.../robots/umr_configs/` | `configs/umr_configs/`（30 份 + GR3 的 `humanoid_retarget_fourier_gr3_umrfactory.json`） |
| `/var/tmp/.../robots/{inventory,finalize,render_tpose}.py` | `surface_retargeting/robots/registry/` |
| `~/umr_factory_data/robots/<r>/{staging,umr_robot,umr_shared_beta0}.json` + `pipeline/pipeline.json` | `configs/robots/<r>/` |
| `~/umr_factory_data/robots/{staging,surface}_summary.json` | `configs/robots/` |
| `~/umr_factory_data/robots/<r>/pipeline/*.py,*.sh,*.json`（5 台） | `archive/robots_wf_2026-10/<r>/`（原样快照） |
| `~/umr_dataset/umr_dataset/` | `surface_retargeting/dataset/` |
| `~/umr_dataset/{scripts,tests}` | `scripts/dataset/`、`tests/dataset/` |
| `~/umr_dataset/README.md` | `docs/dataset.md` |
| `~/umr_dataset/qa/*.json`、`_scale/bench_*.json` | `docs/benchmarks/` |
| `~/gpu_convert/{convert,prep,gpu_ik,gpu_beta,gpu_lowest,gpu_smplx,seg_ik,parts_status}.py`、`run_watch.sh` | `surface_retargeting/ingest/convert_gpu/` |
| `~/gpu_convert/{parity,bench_ik,beta_control,beta_objective_check,test_seg}.py` | `tests/convert/` |
| `~/gpu_convert/{convert_v1,gpu_ik_ref}.py` | `archive/gpu_convert_v1/` |
| `~/umr_factory_data/scripts/{bvh2smplx,smplx_qa,convert_omni,convert_hiphi,build_manifest,obj_pen_audit,omni_labels_extract,omni_npz_rest,render_clip}.py` | `surface_retargeting/ingest/convert_cpu/` |
| `~/umr_factory_data/scripts/{dl_hiphi,dl_omni,bones_seed_dl}.py` | `surface_retargeting/ingest/download/` |
| `~/umr_factory_data/scripts/{throttle_guard,start_hiphi_dl,start_bones_dl}.sh` | `ops/` |
| `~/umr_factory_data/scripts/` 其余 5 个一次性 shell | `archive/factory_shell/` |
| `~/umr_factory_data/scripts_grail/*`（不含 `.bak`） | `surface_retargeting/ingest/grail/`；`start_grail_dl.sh` → `ops/` |
| `~/umr_factory_data/scripts_streamc/{hiphi_staged_dl,rank_hiphi_parts}.py` | `surface_retargeting/ingest/hiphi/` |
| `/var/tmp/.../uniml3d_reload/proto/*.py` | `surface_retargeting/ingest/uniml3d/` |
| 研究仓 `alpha_motion2/wbc_expert/scripts/` 的 3 个 BONES 流式下载脚本 | `surface_retargeting/ingest/bones_seed/legacy/` |
| `~/umr_factory_data/experiments/truncation`（代码、PROGRESS.md、final_rules） | `experiments/truncation/`；定稿规则 → `surface_retargeting/curate/final_rules_v{1,2}.json` |
| `~/umr_factory_data/diversity/**.py` | `surface_retargeting/census/` |
| `~/umr_factory_data/diversity` 的 REPORT 和各 summary | `docs/census/` |
| `~/motion_audit_web/{backend,agent_tools,scripts,tests,frontend 源码和 lock,docs(不含 png)}`、`check_decisions.py`、`server.py`、requirements、README | `viewer/` |
| `~/motion_audit_web/qa/{acceptance.md,environment-freeze.txt,mcp-*,source_baseline.json,fixture_audit/}` | `viewer/qa/` |
| `~/motion_audit_web/configs/mcp.json` | `viewer/configs/mcp.json`（去掉了 AlphaMotion 的两个工作流和两个证据目录） |
| `/var/tmp/.../labels`（pcl_contact 原型，代码 + 规格） | `archive/pcl_contact_proto/` |
| `/var/tmp/.../lloyd_gpu_eval/*.py` | `archive/lloyd_gpu_eval/` |
| `~/umr_factory_data/pilot/diag/coll_vs_vis.py` | `tools/diag/` |
| 研究仓 `am2exp_v1/assets/{umr_sampler,umr_reference}.py` + 采样器测试 | `surface_retargeting/robots/sampling_reference/` + `tests/sampling_reference/` |
| 研究仓 `research/hand_surface_case_20261004`（只收代码和 README） | `examples/hand_surface/` |
| `audit/SCHEMA.md`、`CODEX_PROMPT.md`、`DATA_SPEC_v2.md` | `docs/contracts/` |
| `RESOURCE_LIMITS.md`、`platform_design/*.md` | `docs/`、`docs/platform_design/` |
| 会话工作流里提到 UMR 的 9 个 `*.js` + `umr_factory_data/workflows/*.js` | `docs/agent_workflows/` |
| UMR / genmo-gem 两个环境 | `env/`（conda explicit、conda list、pip freeze；GENMO 版本见 `env/README.md`） |

没收的：所有数据输出、缓存、venv、mp4、node_modules、SMPL-X / beta_fit、厂商网格、GR3 网格（40 MB）、`h2s_shim`、`umr_ref/`、
`data/correspondence_*.npz`、`.bak` 文件、`~/motion_audit_web_backups/*.tar.gz`（没比对）。

## 3. 配置怎么用

- 唯一的路径来源：`surface_retargeting/config.py`。读取顺序：`configs/sr.example.toml` → `configs/sr.toml`（不进 git）→ 环境变量 `SR_<KEY>`。
- 主要 6 个：`SR_DATA_ROOT`、`SR_WORK`、`SR_UMR_ROOT`、`SR_BODY_MODELS`、`SR_ROBOT_ASSETS_ROOT`、`SR_PYTHON_ENV`。其余（`SR_FACTORY_ROOT`、`SR_GMR_ROOT`、`SR_GENMO_PYTHON` 等）同样能覆盖。
- 本机默认值 = 迁移前写死的值（`tests/test_config.py` 逐个核对）。例外：代码位置改成指向本仓库（UMR 包装脚本、登记表、umr_configs、hiphi2smplx 子模块，内容一致）。
- `SR_UMR_ROOT` 本机仍指向旧的 `/var/tmp/.../umr/UMR`，因为那里有子模块里没有的运行时文件（GR3 网格、`data/` 对应关系缓存）。
- 配置 JSON 里的绝对路径换成了 `${SR_DATA_ROOT}` 这类记号，读的时候用 `CFG.load_json()` 展开；展开后和旧文件逐项相等（有测试）。
- Python 代码：`from surface_retargeting.config import CFG as SRCFG`，原来的字符串换成 `f'{SRCFG.data_root}/...'`，值完全相同。
- Shell 脚本：开头 `eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"`，然后用 `$SR_*`。交互式 shell 用 `source ops/sr_env.sh`。
- 共享环境（UMR env、genmo-gem）里要能 `import surface_retargeting`：云上 `pip install -e . --no-deps`；本机不往共享环境装，用 `PYTHONPATH`（shell 脚本已自动加）。

## 4. 还写死的路径（以及为什么）

| 位置 | 原因 |
|---|---|
| `configs/sr.example.toml` | 本机默认值本身 |
| `tests/test_config.py` | 用来核对"默认值 = 旧路径" |
| 各模块的 docstring / 用法说明 / 注释（约 20 处：dataset、convert_cpu、convert_gpu、throttle、produce、parity 等） | 只是文字说明，不影响运行 |
| `configs/robots/<r>/pipeline.json`、`registry.json`、`configs/ablations/*.json` 里的 `code` / `source` / 备注字段 | 记录"当时是从哪来的"的出处文字，不是程序读的路径 |
| `viewer/qa/source_baseline.json`、`viewer/frontend/.qa/catalog.json`、`viewer/README.md`、`viewer/qa/acceptance.md` | 10-04/10-07 的验收证据原样保留 |
| `surface_retargeting/census/core_sources/scripts/{run_xwbc,build_summary}.py` | 读 AlphaMotion 的 `topo_exp` / `xwbc_lafan` / `xwbc_full`，部分已删；一次性普查脚本，原样留着当记录 |
| `surface_retargeting/ingest/hiphi/hiphi_staged_dl.py` 第 1 行 shebang 和 zstd 兜底路径 | shebang 不能用变量；zstd 先 `shutil.which` 再兜底 |
| `surface_retargeting/ingest/convert_gpu/prep.py` 的 `GPU_PIPELINE` 字符串 | 写进输出元数据的版本说明，改了会让新旧输出对不上 |
| `experiments/`、`archive/`、`docs/`、`env/`、`examples/*/README.md` | 归档 / 文档，故意不改 |

## 5. 行为上有意变化的地方

- `retarget/pipeline.py` 输出元数据里的 `code` 字段：`/home/arenalabs/umr_factory_code` → 本仓库路径（记录实际跑的代码）。
- `gpu/production.py::engine_version()`：还是对原 `umr_gpu` 那组文件求哈希（现在分在 `gpu/` + `strict/` + `export.py`），但 import 行改了，所以哈希值会和旧的不同。
- `robots/registry/inventory.py`：候选机器人列表改从 `configs/robots/registry.json` 读，不再读 AlphaMotion 的 `sole_audit.json`；
  去掉了 `topo_exp_caches`、`xwbc_lafan_extracted`、`surface_v2_asset`、`sole_audit_*` 几个交叉字段；`am2_role` 只按名单算。
  输出仍写到 `{factory_root}/robots`（和以前一样），不写进仓库。
- `uniml3d` 原型的 `remote/`、`out/` 改成指向 `{factory_root}/uniml3d_reload/`（以前是相对代码目录的 `../remote`）。
- `tests/strict/test_postpass.py` 新增 `SR_TEST_OUT`，测试时写临时目录，不写 `out/tests`。
- `tests/dataset/test_umr_dataset.py` 的临时目录不再写死到某个会话的 scratchpad。
- `viewer` 的 MCP 配置读的时候展开 `${SR_*}`；`mcp_stdio.sh` 在仓库的 `viewer/` 下跑，但 `.mcp-venv` 仍在 `{viewer_home}`。

## 6. 本机 CPU 冒烟结果（2026-10-12，无 GPU）

命令：`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -rsx tests`
（要去掉 ROS 的 `PYTHONPATH`，不然 ROS 的 pytest 插件会报错）。`.venv` 是用 UMR 环境建的 `--system-site-packages` 虚拟环境，只装了本包（`-e`，不带依赖）和 pytest。

| 测试 | 结果 |
|---|---|
| `tests/test_config.py`（默认值 = 旧路径、环境变量覆盖、3.10 兜底 TOML 解析、配置 JSON 展开、登记表和旧 robots.json 逐项相等、shell 导出） | 6 通过 |
| `tests/test_imports.py`（24 个 CPU 库模块能 import；全仓库 .py 能编译） | 25 通过 |
| `tests/labels/`（CPU 工厂版和 GPU 引擎版标签规则在合成数据上一致） | 7 通过 |
| `tests/qa/test_judge_parity.py`（新判官重判 2 段 × 2 条轨迹的前 40 帧，和旧判官存下的逐帧结果相差 ≤ 1e-9） | 2 通过 |
| `tests/strict/test_postpass_smoke.py`（严格后处理合成测试，G1） | 1 通过 |
| `tests/sampling_reference/`（采样器和上游逐位一致） | 3 通过 |
| `tests/dataset/`：编解码、容器 | 2 通过 |
| `tests/dataset/`：打包 + 压缩 + 配对 | 预期失败（xfail）：已知 bug，现在的 `out/` 里有空片段，`write_robot_shards` 取 `seg[0]` 越界；旧代码一模一样地失败。留到第 6 步修 |
| `viewer/tests`（用旧的 `.venv` 跑新代码） | 51 通过，1 失败：`test_all_existing_npz_read_and_report_actual_contact_mismatches` 期望踢球片段 90 fps，现在数据是 30 fps；旧目录跑同一个测试也失败，是原来就有的 |
| `viewer/tests/test_mcp.py`（借 `.mcp-venv` 的包） | 7 通过 |
| `tests/gpu/*`、`tests/convert/*` | 不收集：要 CUDA 或 genmo-gem 的 GPU 转换器，留到云上 |
| 前端 vitest | 没跑（没装 node_modules） |
