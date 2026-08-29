# HANDOFF — 给 demo 工程师（Codex）的完整交接文档

写于 2026-08-07。读者假设：会 python/torch/mujoco，但对本项目零背景。
本文档 = 我们做了什么 + 资产在哪 + 怎么正确地做 demo + 会咬人的坑。
**规矩第一条：给用户（lloyd）看的一切动作视频必须是机器人 mesh 渲染，火柴人只准自己 debug 用。**

---

## 0. 一句话与三层架构

项目：**原生跨具身的动作生成模型 + SE(3) 稀疏约束控制接口**。
同一串离散 token，解码到任意人形机器人（含从没见过的），并接受"某关节某时刻在某处"式的约束。

```
[空间层]  每一帧: 全身姿态 <-> 离散码       (已训好, 冻结使用)
[时序层]  60帧窗口: 动作 <-> 9个FSQ token   (本周主战场, 两臂对决)
[生成层]  prompt/约束 -> token 序列          (未建, 只有设计稿 ARCH_V4)
```

- **空间层**（`model_dual.py` DualTranslator）：逐关节全局 6D 旋转+根相对位置，SigLIP 关节名语义 + 图注意力 → Perceiver 池化 → 双 FSQ（pos/rot）→ 解码到**任意目标骨架**。被掩关节喂 learned mask token（BERT 式输入、全身 loss）。
- **时序层**（`train_v7.py` V7Tokenizer）：只建模 **6 个关键关节**（root/head/双腕/双踝）的轨迹。输入 = 6D 旋转 + 速度 + SG(5,2) 平滑角加速度（**没有 jerk**，信噪比为负，别加回来）。两条臂：
  - `--arm jepa`：8 个无序全局**内容 token** + 1 个**速度控制 token**（编码归一化累计弧长曲线 = "进度条形状"，不是时钟）
  - `--arm maskfsq`：9 个时间局部**内容 token**（60 帧切 7/6/7/7/6/7/7/6/7）。**没有速度控制 token**
  - 两臂比特预算严格相等 62.69 bit/s，`--rate_sweep` 是唯一豁免口。
  - 解码器输出的是**相对"首尾直线插值"的偏移**（直线是架构先验）。
  - **OAT（有序前缀训练制度）已判死刑并删除**（config: `oat=False`），别复活它。
  - **术语纪律**：不要说"寄存器"。瓶颈里的每个槽位就叫 **token**，分两类：
    **内容 token**（动作走哪条路）和 **速度控制 token**（进度条形状）。
    代码里残留的 `n_path`/`n_time`/`path_z`/`time_z` 是历史命名，含义即上面两类。

## 1. 环境与路径（原样复制可用）

```bash
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python     # 唯一正确的解释器
EXP='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
#    ^^^ 目录名里真的带引号字符，bash 里要按上面这样写
R=/var/tmp/alphamotion/topo_exp/runs                    # 所有 checkpoint/verdict
VIZ=.../pose_translator_design/R4_VIZ/mp4               # 所有视频产物
export MUJOCO_GL=egl                                    # 无头渲染
```

数据缓存：`v5_common.AMASS`（cache_xwbc_amass，4000 clips，30fps）。
机器人资产：`/var/tmp/alphamotion/config_harvest/GMR/assets/`（17 家）+
`/var/tmp/alphamotion/robot_models/agibot_x2_urdf/`（X2，零样本明星）。

## 2. 模型资产（demo 直接加载这些）

| checkpoint | 是什么 | 用途 |
|---|---|---|
| `$R/E19_K128_0805` | 冻结空间层（SigLIP 版） | **一切时序 demo 的解码底座** |
| `$R/E27_qwen_0806` | Qwen 语义版空间层 | 后投影质量赢 18/18，但时序层没在它上面训过——**别混搭** |
| `$R/E28_resid_0807` | 空间层 + vis_residual | 被命令关节 0.00cm，SE3 约束 demo 优先用它 |
| `$R/V7_maskfsq_0807` | **当前领先的时序臂** | 大多数 demo 用它 |
| `$R/V7_jepa_0807` | 对决另一臂（带速度控制 token） | 速度控制类 demo 用它 |
| `$R/CTRLa_maskfsq_4body_0807` 等 | 混淆对照 | 别当主角 |
| `$R/RATE_*_0807` | 词表尺寸扫描（可能还在跑） | **别动、别占它们的 GPU 大头** |

加载（照抄即可）：
```python
import v5_common as V
from train_v5 import load_run, to_states
kd  = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
tok, _ = V.load_frozen(dev)                       # 冻结空间层 (E19)
model  = load_run(f"{V.RUNS}/V7_maskfsq_0807", dev)[0]
```

## 3. 数据 API（60 帧窗口，6 关节 × 9 维）

```python
g = np.random.default_rng(seed)
st, cids = kd.sample_starts(n, g, "val", span=60, families=["kick"])
w = kd.window("human_smpl", st, 60, 60)           # [n,60,6,9] rot6d + pos/reach
fam = kd.family[int(cids[0])]                     # 家族标签, 用它, 别自己 regex
```
- 家族：walk/run/jacks/kick/dance/punch/…/other。**"other" 占 77%**——任何按族统计/检索的库必须配平，否则底率碾压一切（我们栽过两次）。
- `"val"` split = 没训过的动作，demo 的"未见动作"全从这取。
- reach 常数：`kd.reach[emb]`（human_smpl 104.2cm）。**位置通道已除过 reach**。

## 4. 时序模型的三个入口

```python
kn = torch.ones(B, device=dev)
e = model.encode(w, w[:,0], w[:,-1], kn)          # -> path_z/time_z/path_q...
s = model.decode(e["path_z"], e["time_z"], cur, tgt, kn, T_out)   # T_out 可 != 60
key = to_states(s, model, cur, tgt, kn)           # -> [B,T,6,9] 关键关节态
line = model.baseline(cur, tgt, kn, T_out)        # 首尾直线先验（对照组必备）
ints, tints = model.token_ints(e)                 # FSQ 整数码（检索/统计用）
```
- **速度控制** = 同一 e，换 T_out（30/60/120）解码。jepa 另可移植速度控制 token（`time_z`）。
- **jepa 的 cond_drop=False**：模型从没见过缺端点。单帧外推/残缺端点是 OOD，能 demo 但必须标注。

## 5. 解码到机器人（唯一被认可的渲染管线）

**每个用户可见视频必须走**：关键关节态 → 冻结空间层补全全身 → **可行性投影进该机器人自己的限位** → 写进它自己的 MJCF qpos → MuJoCo FK 渲染 → 每帧最低点贴地。

现成函数（**别重写，直接用**）：
```python
import v7_bridge_mesh as BM
q, rootR, gp, limstats = BM.robot_stream(tok, src_ent, tgt_ent, key_seq,
                                         kidx, spec, dof_d, rest_d, dev)
# 场景组装/多机并排/标注/写mp4: 抄 carnival.py 的 solo_frames + label_tile
# 或 exam_c_mesh.py 的 MeshCtx.render (单机型多变体并排)
```
机器人注册：`exam_c_mesh.TARGETS`（x2/g1）、`carnival.ROSTER`（8 台已验通过）。
新机器人 = 给 `eval_newbody.get_target` 一个 MJCF 路径，能 build 就能用。
**A3 还没接**（`agibot.CANDIDATES` 里有 4 个 a3 候选，接法同 `night2_eval.build_x2_entry`）——这是待办 demo 素材。

相机纪律：`azimuth=180`（正对机器人排，远近一致），fps 20，尺寸 ~1700×780。

## 6. Demo 配方（每种一个模板文件，抄改即可）

| demo | 模板 | 关键机制 |
|---|---|---|
| 整段重建（codec） | `exam_c_mesh.py` m1 | encode 全窗 → decode |
| 两段拼接桥 | `v7_bridge.py` + `v7_bridge_mesh.py` | **B 段必须 `yaw_align` 到 A 段末帧朝向**；桥 token 用 donor（真实动作的 token）= 天花板，interp = 地板；桥帧全部标红 |
| 速度三档 | `exam_c_mesh.py` m6 | 同 token 变 T_out；做完的面板**定格+标签变灰** |
| 关键帧草图填肉 | `exam_c_mesh.py` m5 | 分段 baseline 拼草图 → encode → decode |
| token 检索 | `exam_c.py` m5 | Hamming 距离 + **配平的族库** |
| 遥控器干预 | `probe_remote.py` | 只换一个 token 再解码 |
| 多机嘉年华 | `carnival.py` | 章节+标题卡+瓦片拼图 |

## 7. 已判定的结论（demo 的口径，别说过头）

- maskfsq 3:2 胜 jepa（能量/零样本/转场 vs 时间不变性）；**词表效率上 jepa 用 78% 比特拿 94% 质量**，率失真扫描在跑，两边都别吹死。
- 砍 OAT 是动作幅度修复的原因（对照已裁决：架构 0.91 vs 数据轴 0.58）。
- **token 是全身旋钮不是按钮**（REMOTE_verdict：换任一码全身动，maskfsq 时间局部性仅 2-3 倍底率）。**demo 别宣称"这个 token 控制手臂"**。
- 桥的物理：模型编的帧**零新增**限位/力矩/自碰撞违规；但全管线滑脚 ~0.5m/s（GT 重定向地板 0.017）——**别在 demo 里宣称"物理上可执行"，说"物理一致性不劣于其重建基线"**。
- 速度不变性验收线 0.95 两臂都没过（0.867/0.542）——别宣称"路径与速度完全解耦"。

## 8. 坑清单（每条都咬过我们）

1. **单位**：`spec.rest_offsets` 已是 cm，`fk_pos` 输出即 cm。**别再乘 reach**（渲染无感、度量爆 104 倍）。
2. **拼接必须对齐朝向**：两段素材各自朝向随机，不 `yaw_align` 的话"桥"=原地转体 180°。
3. **族统计必须配平库**（other 77%）。
4. 任何 shuffle/采样**必须给种子并多抽平均**（content_share 曾经 ±0.1 抛硬币）。
5. 厂商 MJCF 的自由关节可能**没名字**——按 body 前缀找（carnival.py 已修，抄它）。
6. 多面板不同长度用 **max+定格**，不是 min 截断。
7. GIF 交付前转 mp4（`ffmpeg -movflags +faststart -pix_fmt yuv420p`），路径带引号目录名小心转义。
8. 限位数字疑高先怀疑测量：**3 自由度关节的欧拉分支会伪造 60-100° 假超限**（X2 那个 62° 未清白）；9/18 机器人的 ±180° 限位盒已修过一轮。
9. `unitree_h2` 关节名是 body00..31（未修），`pal_talos` 用 `_fixed` spec（已自动加载）。
10. 评估别拿 GT 当唯一尺子（GT 自己会超限/偏 93°）——**视觉关先过，数字第二**，每份产物打印绝对路径让 lloyd 肉眼判。

## 9. 文档地图（要背景读这些）

- `ARCH_V4.md` 三层总设计 + SE3"点亮格子"抽象（**未交付的 m4 demo 就做这个**）
- `ARCH_V5_TEMPORAL.md` 时序层规格 + §六 四维零样本展示要求 + §七 八项统计
- `LIT_JEPA_COLLAPSE.md` JEPA 塌缩文献与我们的对应
- `$R/V7_verdict_0807.txt`、`$R/EXAMB_verdict.txt`、`$R/EXAMA_verdict.json`、`$R/REMOTE_verdict.txt`、`$R/V7_BRIDGE_verdict.txt` — 全部现成结论数字
- 视频产物：`R4_VIZ/mp4/{CARNIVAL.mp4, v7_bridge_mesh/, exam_c_mesh/}`

## 10. 建议你（Codex）做的 demo，按价值排序

1. **m4 / SE3 点亮格子**（欠交付的核心）：给一个关节的整条轨迹当约束（kick 约束脚），
   温层 endpoints 补全 + 空间层可见集只留 {root,该关节} + `manifold_project.project_constrained`
   钉住轨迹 → 与 GT / spatial(GT全钥匙) 三列并排。加**密度收敛曲线**：首末→+1 点→+3 点。
2. **嘉年华 v2**：统一天空盒（各家 xml 自带背景色不齐）、加 A3、加 SE3 章节。
3. **遥控器视频版**：最强内容 token 的某一维扫 5 档，X2 五列并排（数据在 REMOTE_verdict.json）。
4. 长链 rollout：10×2s 段连跳（考卷 C 的"缝合卷"完整版）。

有不明白的机制，先读模板文件顶部的 docstring——每个文件都写了它证明/证伪了什么。
