# 数据下载清单（lloyd 拉，0911 起）

规则（所有条目通用）：
1. **只下动作和 metadata，不下视频。** 视频文件（mp4 / mkv / jpeg 序列）留 URL，我们在 source 记录里存 `video_url` + 帧映射。
2. 落盘根目录 `/var/tmp/arena/raw/<dataset>/`（本机还有 280 GB 空闲；每个条目下面写了预计体积）。下完在该目录放一个 `SOURCE.txt`：URL、下载日期、许可证、你的账号名。
3. HF 上 gated 的先在网页上点"同意条款"，再用 `huggingface-cli download <repo> --repo-type dataset --include <pattern> --local-dir <dir>`。带 `--include` 过滤是关键，否则会把视频一起拉下来。
4. 下完告诉我目录名，我这边 `arena source register` 登记 + gauge 实测。

| # | 数据集 | 用途 | 来源 | 只下什么 | 预计体积 | 状态 |
|---|---|---|---|---|---|---|
| 1 | **LAFAN1**（BVH，Ubisoft） | locomotion / bvh 家族 100 条（现在只有重定向缓存 77 段 + 原始 10 条） | https://github.com/ubisoft/ubisoft-laforge-animation-dataset → `lafan1/lafan1.zip` | 全部 `.bvh`（77 个文件） | ~1 GB | 无账号 |
| 2 | **100STYLE**（BVH） | locomotion / bvh 家族补 dance·other | https://www.ianxmason.com/100style/ | `100STYLE.zip` 里的 BVH | ~3 GB | 无账号，需填表 |
| 3 | **AMASS 原始**（SMPL-X N） | 只有当某个家族（roll / ground）在重定向缓存里不够时才需要 | https://amass.is.tue.mpg.de/ （登录） | 先只下 `ACCAD`、`CMU`、`KIT`、`BMLmovi`、`GRAB` 的 `SMPL-X N` 包 | ~15 GB | **可以先不下**，缓存里 13,281 段够 100 条 |
| 4 | **EgoDex 全集**（Apple） | manipulation / egocentric 50 条 → 后面 1 万小时的主力 | https://ml-site.cdn-apple.com/datasets/egodex/ （`test.zip`、`part1.zip` … `part5.zip`） | 解压后**只留 `.hdf5`，删 `.mp4`**（本机 `~/tmp/ee_data/fetch_zip.py` 就是这么做的，可以直接复用） | hdf5 约 40 GB；带 mp4 是 1 TB 级，**不要留视频** | 无账号 |
| 5 | **UMI**（Stanford） | manipulation / umi 50 条 | https://umi-data.github.io/ → `https://real.stanford.edu/umi/data/zarr_datasets/` | 各 `*.zarr.zip`，解压后删 `camera0_rgb` 数组（`umi_media.log` 里有做法） | 去掉图像后每个几十 MB | 无账号 |
| 6 | **AgiBot World Beta** | manipulation / 人形（AgiBot G1 双臂）；后面 1 万小时的另一主力 | HF `agibot-world/AgiBotWorld-Beta`（gated，网页同意条款） | `--include "*/proprio_stats/*" "*/task_info/*" "*.json"`，**排除 `observations/videos`** | 状态文件 ~50 GB（全集含视频 ~2 TB） | 需要你的 HF 账号 |
| 7 | **DROID**（机械臂 Panda） | manipulation / fixed_arm 50 条 | HF `lerobot/droid_1.0.1`（LeRobot v2 格式，不 gated） | `--include "data/*" "meta/*"`，**排除 `videos/`** | parquet ~30 GB | 无账号 |
| 8 | **Unitree G1 开源操作数据** | manipulation / humanoid_g1 50 条（现在只有 5 条 pingpong） | HF `unitreerobotics/`：`G1_ToastedBread_Dataset`、`G1_Dex3_BlockStacking_Dataset`、`G1_Dex3_Pouring_Dataset`、`G1_Dex3_ObjectPlacement_Dataset`、`G1_Dex3_CameraPackaging_Dataset`、`G1_Dex3_GraspSquare_Dataset`、`G1_Dex1_MountCamera_Dataset` | 每个 `--include "data/*" "meta/*"`，排除 `videos/` | 每个 0.5–3 GB | 无账号 |
| 9 | **xwbc-retarget-data**（AMASS/LAFAN 18 台重定向） | 已在盘上（`/var/tmp/alphamotion/xwbc_full`） | HF `TonyZhang1999/xwbc-retarget-data` | — | — | **不用再下** |
| 10 | **HuMI** | manipulation / humi 50 条 | **我没找到公开地址**（HF 搜 "HuMI" 没有对应仓库）。你知道链接的话给我，或者告诉我它是哪家的 | — | — | **等你给链接** |
| 11 | **轮式机器人操作数据** | manipulation / wheeled 50 条 | 你说明里没指定来源。候选：AgiBot World 里的轮式底盘任务（同 #6，不用另下）；或 Open X-Embodiment 里的 Fetch / TidyBot 子集。你定一个 | — | — | **等你定** |
| 12 | **SOMA / BoneSeed 原始** | locomotion / soma 家族 100 条（现在只有 12 段预处理） | SOMA 需要 marker 数据 + 拟合流程，不是现成下载；BoneSeed 我们只有审计产物 | — | — | **本周按 blocked**，先用 12 段 |

**优先级**：1 → 4 → 8 → 6 → 7 → 5 → 2；#3、#12 先不动；#10、#11 等你答复。

**下载命令示例**（#8 一个仓）：
```bash
huggingface-cli download unitreerobotics/G1_ToastedBread_Dataset --repo-type dataset \
  --include "data/*" "meta/*" --local-dir /var/tmp/arena/raw/unitree_g1/G1_ToastedBread_Dataset
```
（#6 同样写法，把 `--include` 换成 `"*/proprio_stats/*" "*/task_info/*" "*.json"`。）
