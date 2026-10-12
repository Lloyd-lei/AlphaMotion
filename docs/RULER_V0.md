# ruler-v0：判官冻结版

这个提交（标签 `ruler-v0`）是 CPU 工厂 `umrf` 在 10-04 晚 Task B 改动之前的样子，
也就是 GPU 引擎拿来当独立判官和 CPU 严格参考的那一版
（原件：`~/umr_factory_code/umrf/*.py.bak_taskB` 加上没改过的文件；
和 `~/umr_gpu/eval/factory_ruler_snapshot/` 逐字节相同，md5 见下）。

搬进来时只做了一件事：把 `from umrf...` / `from .x import` 改成新包路径
`surface_retargeting.<子包>.<模块>`。逻辑一行没动。旧文件 → 新位置：

| 旧 | 新 |
|---|---|
| umrf/robots.py | surface_retargeting/robots/robots.py |
| umrf/surface.py | surface_retargeting/robots/surface.py |
| umrf/shared_corr.py | surface_retargeting/correspondence/shared_corr.py |
| umrf/pipeline.py(.bak_taskB) | surface_retargeting/retarget/pipeline.py |
| umrf/postpass.py(.bak_taskB) | surface_retargeting/strict/postpass.py |
| umrf/geometry.py(.bak_taskB) | surface_retargeting/strict/geometry.py |
| umrf/objects.py(.bak_taskB) | surface_retargeting/strict/objects.py |
| umrf/labels.py | surface_retargeting/labels/labels.py |
| umrf/paths.py | surface_retargeting/paths.py |

原件 md5（和 SNAPSHOT_MD5.txt 一致）：
```
b81330f722a19c46d2a7fa7215522b9a  umrf/geometry.py
a7d346f3d9bdf641f890dbae27565e25  umrf/__init__.py
6322f3056b66f2d64f660b1ecaab20ba  umrf/labels.py
00bede8eb6062551c1c31594d39a25f4  umrf/objects.py
39516fcda765815e32fda971d31a42a6  umrf/paths.py
e21012fed899a190248ffc8a2c4707a5  umrf/pipeline.py
295880ccee363e23658789aadf02775f  umrf/postpass.py
177b49507b9166710ea389031852e14e  umrf/robots.py
ff012e88347166277b7e4e0df879576c  umrf/surface.py
```
