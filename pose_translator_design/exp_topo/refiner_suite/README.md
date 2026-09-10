# SE(3) 精修器套件（对照臂，0905–0909）

18 台机器人 + agibot X2 / A3 零样本，手 SE(3) 钉点下的全身补全与精度评估。
结果、视频不在仓库里：`~/tmp/refiner_suite/{out,out_manip,vid,vid_manip}`。

- `profiles.py` / `profiles.json`：每台机器人的手/脚/肩/髋关节名、肢体长度、脚底法线、前向轴。
- `refine.py`：批量投影 LM（float64 CPU）。逐轴位置权重 + 方向残差；脚约束四档 point / plane / axis / full。
- `suite_core.py`：模型起点（FZ5c 软补全）/ 零姿态起点 / 两起点择优（`best_*`），零样本目标按臂长、腿长、骨盆高比例从 G1 缩放。
- `eval_suite.py` / `eval_extra.py` / `aggregate.py`：评估与汇总（走路集 `run_all.sh`，manipulation 集 `run_manip.sh`，补条件 `run_extra.sh` / `run_best.sh`）。
- `render_rollout.py`：人体 | GT | 各条件并排视频 + 12 帧网格。

主要结论（15 台有真值，全身旋转误差均值）：只钉手 33.7°；脚点 / 面 19.0 / 17.9°；脚轴 7.9°；脚全 SE(3) 5.9°。
手 SE(3) 单独不够，脚至少要"位置 + 前向轴"。
