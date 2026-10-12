# 环境锁文件（2026-10-12 导出）

| 文件 | 来源 |
|---|---|
| `umr_env.*` | UMR 环境 `/var/tmp/alphamotion/umr_factory/umr/env`（Python 3.12.15；torch 2.14.1、warp-lang 1.18.0、mujoco 3.3.7、clarabel 0.11.1、numpy 1.26.4）。CPU 工厂、GPU 引擎、GRAIL 转换、判官都用它 |
| `genmo-gem.*` | conda 环境 `genmo-gem`（Python 3.10.0）。GPU 人体转换器 `ingest/convert_gpu` 用它的 python |

每个环境三份：`conda-explicit`（`conda list --explicit`，可 `conda create --file` 重建 conda 部分）、
`conda-list`（带 pip 包的总表）、`pip-freeze`。

重建：`conda create -p <dir> --file umr_env.conda-explicit.txt` 然后 `pip install -r umr_env.pip-freeze.txt`
（pip 里的 `-e` / `@ file://` 行要换成实际路径或删掉）。

## genmo-gem 里的可编辑安装

- `gem`（GENMO）：`-e git+https://github.com/NVlabs/GENMO.git@7b6ed3c9f640258df214210aea25032d39f7523c`，
  实际目录是研究仓里的 `GENMO/`。本地有改动（4 个文件改过、17 个新文件，见下），没提交。
- `GMR`：`-e /home/arenalabs/Desktop/Gen2Humanoid/third_party/GMR`
- `smplx`：git vchoutas/smplx@1265df7

**gpu_convert 根本不 import GENMO**：只用了这个环境的 python、torch、smplx，加上 hiphi2smplx（经 sys.path）。
所以重建转换器环境时 GENMO 可以不装；只有 Studio / GEM 那条线才要它。
另外这个环境里没有 h5py，转换器靠 `h2s_shim/h5py.py` 垫片顶替（见 docs/MIGRATION.md）。

GENMO 本地改动（`git status --short`，2026-10-12）：
```
 M gem/gem.py
 M gem/utils/smplx_utils.py
 M scripts/demo/demo_smpl.py
 M scripts/demo/demo_utils.py
?? notebooks/
?? scripts/alpha_motion_synergy_heatmap.py
?? scripts/alpha_motion_synergy_video.py
?? scripts/demo/analyze_temporal_joint_jitter.py
?? scripts/demo/compare_gem_kimodo_motion.py
?? scripts/demo/demo_motion_continuation.py
?? scripts/demo/gem_smpl_to_kimodo_smplx.py
?? scripts/demo/gem_smpl_to_kimodo_soma.py
?? scripts/demo/render_keymodel_continuity_batch.py
?? scripts/demo/render_kimodo_smplx_mesh.py
?? scripts/demo/render_kimodo_soma_mesh.py
?? scripts/demo/run_keymodel_continuity_batch.py
?? scripts/demo/run_keymodel_dense_prefix_batch.py
?? scripts/demo/run_keymodel_targeted_continuity.py
?? scripts/demo/run_mask_reconstruction_batch.py
?? scripts/demo/run_sparse_keyframe_resynthesis.py
?? scripts/demo/run_temporal_mask_batch.py
```
```
 gem/gem.py                 |   3 +
 gem/utils/smplx_utils.py   |  25 +++++++
 scripts/demo/demo_smpl.py  | 162 ++++++++++++++++++++++++++++++++++++++++++++-
 scripts/demo/demo_utils.py |  90 ++++++++++++++++++++-----
 4 files changed, 260 insertions(+), 20 deletions(-)
```
