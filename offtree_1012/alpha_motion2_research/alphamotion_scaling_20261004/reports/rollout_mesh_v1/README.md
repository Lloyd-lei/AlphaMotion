# AlphaMotion-2：匹配源网格的运动学回放

先打开 `index.html`，选择身体、模型与全景／近景。所有视频都可直接用系统播放器打开。六组的选例规则相同：冻结严格合法参考病例中的字典序首例；不按效果挑片，不丢未达标帧。

## 这批交付

- Adroit 与完整 Franka：每体 A/B × S/M/L，各有全景和近景，共 24 段。每段六栏为记录参考、原生自由姿态、真实关节投影 step 0、P1 关节优化 step 20/100、默认 q 起步优化 step 100。
- G1：参考与六组冻结网络输出的同一人到机器人配对片段，带匹配源 visual mesh。
- Adroit 固定 64 帧，0.64 秒：按官方标称 0.01 秒控制周期回放，无实测墙钟时间。Franka 固定 35 帧，2.8 秒：保留原 MJL 时间戳及 stride 40，编码逐帧 PTS 最大误差小于 0.5 微秒。末帧显示时长取相邻间隔中位数。
- G1 使用已保存评测缓存的时间轴；原始采集速率未在本轮独立复核，因此不宣称原拍摄速度。
- Kuavo 与两只 UniMate 四足原骨架视频仍保留在原交付目录；本轮没有它们的已验证 mesh 绑定，不把骨架版称为已满足网格要求。

## 如何读图

视频时间推进的是记录动作，不是优化迭代。各栏的优化步数固定。原生自由姿态可能不符合关节约束；精修栏只检查源模型已知限位。绿线是该栏末端轨迹；红／紫是两个目标，浅蓝骨架为参考。

全景采用各栏共同的固定相机，保持原位置误差。近景为明确标注的局部形状视角：每栏相机跟随掌心或夹爪，右上保留共同相机的全景；数值始终在原坐标计算，没有重新对齐评测。Franka 开口是指根沿实际滑动轴的间距，不是接触垫间隙。

近景按源模型只显示手掌／手指或夹爪的 visual 几何；Adroit visual group 1、Franka visual group 0，碰撞几何不混入。源材质保留，仅调整照明与背景。没有替换手型或机器人。

## 证据与验证

`bundles/` 保存可离线读取的版本化输入、压缩姿态、匹配模型 MJB 与源元数据。12 个病例的 P1 重放全部 101 步指标及最终归一化 q 与冻结记录逐值相同；未加载神经网络、未训练、未调用 mj_step。step 20 来自真实冻结优化器重放，未做姿态插值。

各视频目录的 `output_manifest.json` 记录输入/模型/代码哈希、checkpoint、病例、帧号、坐标与单位、时间、相机、编码、完整解码和容量检查。独立 mj_forward 对所有保存关节态验证网格变换。`all_frame_mesh_bounds_qa.json` 检查六栏、全部源帧的手指/手掌网格顶点无视口裁切；`tests.txt` 为 7 项契约、真实病例和非均匀时间编码测试。图像视觉抽检另见 `visual_qa.json`，不能把程序检查当作人眼逐帧审阅。

输出均为 1920×1080、H.264、yuv420p、faststart MP4。原始短窗口未人为补长或循环录制；播放器可手动慢放，画面时间仍显示来源时间。

## 离线复现

本工具是已在 ArenaLabs Linux/EGL 环境验证的有限离线原型，不是任意机器人通用渲染平台。依赖 Python 3.11、NumPy、Pillow、MuJoCo、FFmpeg/FFprobe；仅导出输入时需要冻结工程的 PyTorch/CUDA，纯重渲染不依赖训练、Docs 或网络。

从工程根目录运行（路径含空格时务必引号包裹）：

```bash
python rollout_tools/render.py \
  --bundle reports/rollout_mesh_v1/bundles/adroit_pen/A_M_seed0 \
  --view detail --output reports/rollout_mesh_rebuild \
  --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

批量使用同一入口：`python rollout_tools/render.py --batch rollout_tools/batch.json --output reports/rollout_mesh_rebuild --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。`batch.json` 中每项仅含 `bundle` 与 `view`，迁移后改为当地路径。G1 薄适配器复现：`python rollout_tools/g1.py --bundle reports/rollout_mesh_v1/g1_bundle --output reports/g1_mesh_rebuild --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。

同一输入/代码/字体/运行时/视角命中缓存时核验输出哈希后返回；不覆盖旧证据。中断任务在同一输出目录重跑，会核验并复用已有帧。失败保留于隐藏 `.work` 目录的 `failure.json`；完成后原子提交整个目录。输入或代码改变时使用新输出根目录。缓存键不是跨硬件逐像素一致性的保证。

测试：`ROLLOUT_ROOT=reports/rollout_mesh_v1 python rollout_tools/test_rollout.py`。全帧网格边界：`python rollout_tools/qa_mesh_bounds.py --root reports/rollout_mesh_v1`。纯渲染不运行优化器；`prepare.py` 仅在需要重新派生中间状态时使用，它要求原冻结 P1 源码、已完成队列与 GPU 文件锁。

远端每个任务先组合剩余资源下载与当前空间，再保留 15 GiB 余量、1 GiB 开销和 512 MiB 渲染/归档预算。未动下载主进程或队列，未解压数据归档。

所有展示均为**运动学回放 / 非仿真执行**。软件测试、数据可用、训练效果、动力学仿真与真机验证是不同阶段；本视频不证明碰撞、接触、抓握或真机成功。Google Docs 只保留工程入口；本地 MP4 不伪装成在线嵌入视频，也未公开发布。
