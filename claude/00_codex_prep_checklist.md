# 文件1：Codex开工前的准备清单与死角确认

**给codex**：这份文件里的每一项都要在写训练代码**之前**完成。带 `[BLOCK]` 的项目没做完，后面的实验结论不成立。带 `[CONFIRM]` 的是我无法在远程确认、需要你实际跑一下验证的死角。

所有环境探测结论均为2026-07-30在arenalabs上实测。

---

## A. 环境：缺什么

### A.1 `[BLOCK]` 建新的conda环境，不要污染现有的

现有env的实测状态：

| env | torch | pytorch_kinematics | transformers | open_clip | viser | yourdfpy | mujoco |
|---|---|---|---|---|---|---|---|
| ardy | 2.11.0+cu128 | **缺** | 5.8.1 | **缺** | 1.0.16 | 0.0.60 | 3.6.0 |
| kimodo | 2.11.0+cu128 | **缺** | 5.1.0 | **缺** | 1.0.16 | 0.0.60 | 3.6.0 |
| rot | 2.9.1+cu128 | **缺** | 5.1.0 | **缺** | 缺 | 0.0.60 | 3.6.0 |
| isaaclab | 2.9.1+cu128 | **缺** | 4.57.3 | **缺** | 缺 | 缺 | 缺 |

四个env全部CUDA可用（12.8）。**但四个env都缺 `pytorch_kinematics`**（可微FK的首选实现）。

建议：`conda create -n alphamotion --clone kimodo`（kimodo已有torch 2.11+viser+yourdfpy+mujoco，基础最全），然后装 `pytorch_kinematics`。**不要直接往ardy/kimodo里装**——那两个env在跑现有的生成任务，装坏了会影响别的工作。

### A.2 `[BLOCK]` SigLIP权重没有本地缓存

HF缓存里现有的是：ARDY×2、Kimodo×2、GEAR-SONIC、TMR-SOMA-RP、LLM2Vec×2、Llama-3-8B、GRAIL数据集。**`grep -i "siglip|clip"` 结果为空——SigLIP和CLIP都没有缓存。**

`~/.cache/huggingface/token` 文件存在，所以HF访问应该是通的。需要你先把SigLIP文本塔拉下来并**冻结**（本轮不训练它）。

`[CONFIRM]` 拉取后确认两件事：(1) 只加载文本塔，不要把视觉塔也载进显存；(2) 记录用的是哪个具体checkpoint名和维度$D_\text{text}$，写进配置文件——后面所有实验都要用同一个，中途换会让结果不可比。

---

## B. 数据：缺什么

### B.1 `[BLOCK]` Truebones Zoo 未下载

全盘搜索 `-iname "*truebone*"` 和 `-iname "*objaverse*"` 结果为空，**两个都不在磁盘上**。

本轮只需要 **Truebones Zoo（65个动物物种）**，Objaverse-XL跳过（71k角色，体量大且需自行筛选绑定资产，不在本轮范围）。

下载后需要做的转换：
1. 转成统一的 **per-joint root-relative 位置**格式（**不是世界坐标**，见文件2的§2.0）：先按C.1对齐up-axis与facing direction，再减去根节点位置，再按骨架高度$H$归一化尺度。三步都不能省，否则跨具身不可比。
2. 提取每个物种的骨架拓扑（父子关系、rest pose / 第一帧作为pose anchor）；
3. **按物种切分** train / val_topo（**不是按motion切分**）——留出约10个物种完全不参与训练。同物种的其他motion泄漏进训练集会让 `val_topo` 这个指标失去意义。

SAMoR原文Table 1给的Truebones Zoo规模，可作为下载后的核对基准：**54物种训练 / 11物种验证**，**939 motion训练 / 110验证**，**关节数范围9–71**。

**注意motion数量极少**（939 vs BONES-SEED的142,220），所以它的作用是提供**拓扑多样性而非数据量**。训练时必须**过采样**，否则动物样本被完全淹没，`val_topo`和$\mathcal L_\text{adv}$失效。起点配比：人形:动物 ≈ 7:3（按帧数计）。

**不选Objaverse-XL的理由**（供参考，避免你误以为漏了）：OXL确实有骨架——SAMoR自建了一个67,646角色的rigged+skinned子集（关节数5–96）——但它含大量non-biological rigs、运动多为程序化animation而非动捕、且需自行从海量资产中curate（SAMoR还用了VLM打caption）。与"主要面对人形"的目标不匹配，本轮跳过。

`[CONFIRM]` 下载后报告：实际物种数、每物种的关节数范围、总motion数、总帧数、文件格式（BVH还是别的）。

### B.2 `[BLOCK]` 关节名→功能组标签映射表

SAMoR的 attention supervision 需要每个关节一个功能组标签 $y_j\in\{0,\dots,K-1\}\cup\{-1\}$。

**兜底机制很宽松，不用追求标全**：匹配不上的关节取 $y_j=-1$，对supervision loss贡献为零，但仍然被编码、由学到的注意力自由路由。对完全没有规范关节名的机械rig/程序化资产，所有关节都是$-1$，此时$K$个slot退化为纯无监督的功能桶，只靠运动学结构和运动模式驱动。

所以：**人形（SOMA-78 / G1-29）尽量标全，动物只标能确定的（四肢、脊柱、头、尾），其余一律$-1$。**

### B.3 `[BLOCK]` 关节名规范化对照表（SigLIP输入用）

不能把原始字符串直接喂SigLIP文本塔，有实际的语义陷阱：
- SOMA的 `LeftLeg` 实际是**大腿**（它是 `LeftShin` 的父节点），字面读作"左腿"会误导文本塔；
- G1的 `left_hip_pitch_joint` 需转成自然语言 `left hip pitch joint`。

需要手工维护 `原始名 → 规范自然语言描述` 的表，SOMA-78 + G1-29 共107项，一次性工作。例：
```
LeftLeg   -> "left thigh, upper leg"
LeftShin  -> "left shin, lower leg"
Neck1     -> "lower neck"
```

### B.4 磁盘：用 `/` 分区，不要用 `/home`

| 挂载点 | 容量 | 可用 |
|---|---|---|
| `/home` (nvme0n1p2) | 1.2T | **42G**（97%满） |
| `/` (nvme0n1p5) | 657G | **428G** |

`/` 上 `/opt`、`/srv`、`/usr/local` 都需要sudo，而 `sudo -n` 测试显示**需要密码**（我无法非交互获取）。目前 `/` 上可直接写的只有 `/var/tmp` 和 `/tmp`。`/tmp` 重启会清空，不适合放缓存。

**决定：用 `/var/tmp/alphamotion`**（在`/`分区，428G可用，按FHS重启后保留）。lloyd说sudo的事先不管，所以不走`/data`。T1的缓存约0.56GB，T2会到~10GB量级，`/var/tmp`都够。

**注意别往`/home`写缓存**——只剩42G（97%满）。

### B.5 GPU占用

RTX PRO 6000 Blackwell，97,887 MiB总量，**当前已被占用24,475 MiB**（有别的任务在跑）。可用约73GB，对本轮的小模型绰绰有余，但开跑前需要和正在跑的任务协调。

---

## C. 死角：必须实测确认的四件事

这四项我在远程只能看到部分信息，**必须由你实际跑一遍确认**，因为每一项做错都会导致模型系统性学错但在平均指标上看不出来。

### C.1 `[CONFIRM]` 坐标系变换的四项验收

已实测确认的事实：

| | SOMA BVH | G1 URDF |
|---|---|---|
| Up轴 | **+Y**（Head−Hips=`(0.33, 59.62, 2.45)`） | **+Z**（yaw关节axis=`(0,0,1)`，膝origin的Z为负） |
| Lateral（左为正） | **+X**（LeftHand−RightHand=`(42.60, 0.06, −2.05)`） | **+Y**（左髋origin`+0.064452`，右髋`−0.064452`） |
| Forward | **+Z**（右手系下 forward = left × up） | **+X** |
| 手性 | 右手系，实测 $\det[\text{fwd},\text{left},\text{up}]=+1.0$ | 标准ROS右手系 |
| 单位 | cm（Hips高101.28） | 米 |

变换式：
$$(x,y,z)_\text{SOMA}=(\,y_{G1},\;z_{G1},\;x_{G1}\,)\times 100,\qquad M=\begin{pmatrix}0&1&0\\0&0&1\\1&0&0\end{pmatrix},\quad \det M=+1$$

**四项验收检查，写成单元测试，不要靠肉眼**：
1. $\det M = +1$（不是$-1$）。**这一项不是形式主义**——本文档早期版本曾把该式写成 $(-y_{G1},z_{G1},x_{G1})$，行列式是$-1$，是个反射，会让模型系统性学出左右翻转且在平均误差上几乎看不出来。这项检查就能直接抓住它。
2. 变换后G1的左髋在+X侧、头部在+Y最高处。
3. 取一条行走motion，SOMA侧与G1侧的**前进方向**在变换后一致。
4. 取一条明确的单侧动作（如"左手挥手"，可从metadata的`content_natural_desc_*`字段检索），确认两侧动的是**同一只手**。**这是唯一能真正抓住镜像错误的检查。**

`[CONFIRM]` **注意BVH的OFFSET是写在骨骼局部坐标系里的**（Hips→Spine1的offset是`(5.00,-0.04,-0.01)`，脊柱沿局部+X而非+Y），**不能直接读OFFSET判断世界朝向，必须做完整FK**。

仓库里已有 `data/fix_amass_axis.py`、`data/diag_axis.py`，说明这类坑踩过，建议复用其约定。

### C.2 `[CONFIRM]` G1 CSV的root列单位与轴序

CSV首行数据的root部分是 `(0.030441707, 1.8163793, 78.11283)`（`root_translateX/Y/Z`），后接 `root_rotateX/Y/Z` 为 `(-0.3256, 1.1543, -177.91)`。

**这里有矛盾**：URDF用米，但78.11若是米则荒谬、若是cm则恰好是G1骨盆高度——说明CSV的root平移可能是cm且Z-up，与URDF的米不一致。旋转列的量级（-177.91）看起来是**度**不是弧度。

**必须用一整条轨迹实测确认**，不要照搬上表：
- 取一条明确的行走motion，看**哪一维在时间上单调变化**（=前进方向）、**哪一维基本恒定且约等于骨盆高度**（=竖直方向）；
- 确认旋转列是度还是弧度（看数值范围是否超出$\pm\pi$）；
- 确认旋转的欧拉角顺序（XYZ还是ZYX等）——**这一项做错同样会导致朝向系统性偏差**。

### C.3 `[CONFIRM]` G1 CSV列的精确schema

实测：**36列 = 1(`Frame`索引列) + 6(`root_translateX/Y/Z` + `root_rotateX/Y/Z`) + 29(以`_dof`结尾的关节列)**。

写dataloader时**按列名解析，不要按列序号硬编码**——首列是`Frame`索引不是数据，容易off-by-one。

### C.4 `[CONFIRM]` FK实现的正确性

需要把G1的29个关节角通过URDF做正向运动学得到3D位置。**选可微实现**（`pytorch_kinematics`），因为loss需要在G1侧3D位置上算，梯度要能穿过FK。

URDF路径：`~/.cache/newton/newton-assets_unitree_g1_308a72cd/unitree_g1/urdf/g1_29dof.urdf`
（同目录还有 `g1_23dof.urdf`、带手版本等变体；**用29dof版本**，与CSV的29个`_dof`列对应。实测该URDF共39个joint，其中revolute 29个。）

验收：把CSV的关节角喂进FK，得到的骨盆-脚底高度、双脚间距等应当在物理合理范围内。

---

### C.5 `[CONFIRM]` Truebones动物骨架的关节类型

SOMA的BVH和G1的URDF都是明确的（BVH为3-DOF欧拉角关节、URDF为单轴revolute）。**动物骨架的关节类型未知**——若Truebones的BVH里存在只有1或2个旋转通道的关节，或存在带平移通道的非root关节（某些动画rig会有伸缩骨），dataloader按"每关节3通道旋转"硬编码会直接解析错位。

下载后请报告：CHANNELS行的通道数分布（是否全为3、root是否为6）、是否存在非root关节带position通道。若有异常，先只取位置（做完FK后的关节世界坐标再转root-relative），**位置通道对任何关节类型都成立**，不受此影响。

### C.6 `[CONFIRM]` 数据规模与磁盘的一致性检查

BONES-SEED是**601GB / 428,279个文件**。§3的采样方案是"每30帧取1"，所以**不要试图把全部124M帧解码进内存或缓存到磁盘**。建议先建一个轻量索引（motion_id, frame_idx）再按需惰性读BVH——BVH是文本格式，随机访问某一帧需要顺序扫到那一行，若逐帧随机读会成为IO瓶颈。

请实测一下：随机读1000个(motion, frame)样本的耗时。若超过几秒，需要预先把采样到的帧转成一个紧凑的二进制缓存（T1的200k帧约0.56GB，很小）。**这一项决定训练是GPU-bound还是IO-bound，值得先量一下。**

## D. 一个我已经量化、但需要你在实现时特别注意的事实

我用 `g1_29dof.urdf` 跑了完整FK，逐个DOF施加0.2 rad扰动，测量它能引起的**最大关节位置位移**：

| DOF | 最大位移 @0.2rad |
|---|---|
| `left_ankle_roll_joint` | **0.000 mm** |
| `right_ankle_roll_joint` | **0.000 mm** |
| `left_wrist_roll_joint` | 0.599 mm |
| `right_wrist_roll_joint` | 0.599 mm |
| `left/right_ankle_pitch_joint` | 3.506 mm |
| 其余23个DOF | 8.3 – 130.6 mm |

**G1的29个自由度里有4个从关节位置中完全观测不到**（双踝roll精确为0——它们在位置映射的零空间里；双腕roll约0.6mm）。

**解法（已定，见文件2 `01_round1_spec.md` §1）**：网络输出**canonical SE(3) frame**而非仅位置。实测同一扰动下这4个DOF在**frame朝向上完全可见（11.459°，正是0.2 rad）**，只在位置上不可见（0.000 mm）。关节角由**解析投影**从frame反解，零学习参数（实测最大误差4.29e-12 rad）。

**注意：不要试图用DCT/频域分解替代旋转通道。** 已实测：构造两条仅`ankle_roll`扫过0.4 rad、其余完全相同的轨迹，沿时间轴做DCT后**全部系数逐比特相同**（低频高频均为0.000e+00）——DCT是沿时间的线性映射，无法创造每一帧都不存在的信息。频域压缩解决的是时序token化效率，与逐帧旋转信息缺失是正交的两件事。

这也决定了评估时**必须单独报告这4个DOF的误差**——它们在29维平均里只占4/29，即使完全学错也会被平均值掩盖，而`ankle_roll`是踝侧倾，直接决定足部接触面姿态与横向平衡。

---

## E. 顺序

1. A.1 建env + 装 `pytorch_kinematics`
2. A.2 拉SigLIP文本塔，记录checkpoint名与维度
3. C.1/C.2/C.3/C.4 四项实测确认 + FK管线（**这一组是最硬的阻塞项**）
4. C.6 数据IO耗时实测（决定是否需要预先转二进制缓存）
5. B.3 关节名规范化表（107项）
6. B.1 下载Truebones Zoo + C.5关节类型确认 + 按物种切分
7. B.2 功能组标签映射（$K=8$的功能组划分见文件2 §6.3c）
8. B.4 建 `mkdir -p /var/tmp/alphamotion`（无需sudo）

**GPU已由lloyd协调腾出**，B.5的占用问题已解决。

**做完1-5就可以先在SOMA↔G1上跑T0冒烟**，不必等Truebones到位。6-7是`val_topo`指标和$\mathcal L_\text{adv}$的前提，T1正式跑之前必须完成。

---

## E.2 v6修订后新增的准备项（2026-07-30）

你的反证审查我接受了5条P0+1条部分，spec已改到v6。**新增以下准备项**：

> **v7更新（2026-07-30 第二轮）**：你对v6 §1.2的两条P0都对，我实测确认后**作废了标量φ方案**。旋转改为 **rest-relative delta rotation** $\Delta R_j=R_j(t)R_j(\text{rest})^\top$（6D参数化）。下面第1、2项相应改写。

1. **`[BLOCK]` $R^{\text{rest}}$ 提取器 + $\Delta R$ 转换**（spec §1.2.3）：URDF侧取全部 $q=0$ 的FK link frame；BVH侧取HIERARCHY的rest hierarchy（所有channel为0）。验收：对每个骨架，$\Delta R_j(\text{rest})=I$ 逐关节成立（误差<1e-6）。
   构造式triad仍需实现（**但降级为anchor几何特征，不再参与旋转表示**），四个退化情形照旧要覆盖：zero-length bone（实测G1有2处：`waist_yaw_joint`、`pelvis_contour_joint`）、参考向量共线、叶节点、分支节点，断言 `not isnan(E).any()`。

2. **`[BLOCK]` 完整链路的 per-link re-gauge test**（你提的正确版本）：在URDF里给若干link的origin rpy加**随机常量旋转** $G_j$，然后验证——
   - 位置 $\boldsymbol p$ 不变（这是必然的，不算证据）；
   - **$\Delta R_j$ 不变**（这才是真正携带旋转信息的量，v6的φ在这里会挂）；
   - **解析出的 $q$ 与物理姿态不变**。
   
   我实测过 $\Delta R$ 的两条性质：随机 $(R_t,R_\text{rest},G)$ 2000次，**gauge偏差最大 3.2e-6 度**；两个rig做同一语义动作，**跨rig差异最大 3.0e-6 度**。你的实现应能复现这两个量级。
   
   **对照**：同样的测试对v6的φ方案，实测 $|\varphi-\varphi'|$ **最大 180.0 度**——所以那个方案被作废了。
3. **`[BLOCK]` paired manifest**（spec §3.0的五条路径）：为路径A/B/C各建索引。**Truebones只进路径D/E**，别给它建跨拓扑配对索引（那种GT不存在）。
4. **`[CONFIRM]` 剪枝拓扑的确定性定义**：SOMA-78 → 38/24/20 的剪枝规则要写成确定性函数（哪些关节保留），因为路径C的GT就是"原关节位置的子集"，规则一变GT就变。把规则和产出的关节名列表存进manifest。
5. **`[CONFIRM]` anchor泄漏标记**（spec §2.2三级回退）：为每个样本标 `anchor_may_leak`。只有一个clip的物种会落到第三级。
6. **`[CONFIRM]` 采样offset随机化**：每30帧取1时 offset 必须 per-motion 随机（按motion_id做种子），不能固定从0开始——否则锁定动作相位。
7. **`[CONFIRM]` 切分键**：用 `(actor_uid, take_name)` 联合分组，防同源重复版本泄漏；镜像按原始motion group整组切。
8. **T0阶段记录每项loss的gradient norm**（spec §4.1）。SAMoR的权重不能直接照搬（表示/单位/任务都变了），要用实测梯度量级重新标定。

9. **`[BLOCK]` T-c的三项对照**（spec §6.2b新增，防"按名字copy坐标"的捷径）：C-1匿名化关节名（100% name-drop + 随机假名）、C-2目标骨架再做held-out骨长/比例变换、C-3语义copy非学习基线。**只有在C-1和C-2下仍显著优于C-3，T-c才支撑"翻译"的结论。**

10. **`[CONFIRM]` joint axis与bone方向的关系**（供你复核我的实测）：G1的29个revolute里，axis与bone对齐的只有**6个**，PERPENDICULAR的有**19个**，oblique 3个，zero-length 1个——**23/29无法用"绕bone轴的标量"表示**。你举的`left_ankle_roll_joint`（bone $(0,0,-1)$ vs axis $(1,0,0)$，$|\cos|=0.000$）是其中之一。这个表值得你自己跑一遍确认。

**关于SigLIP**：你说它不是首要阻塞、可以先用deterministic name features跑几何M0/T0——**同意**。SAMoR原文的name-ablation显示去掉关节名只降13–16%，且规范名覆盖率在OXL只有69%。所以先用确定性name特征（例如关节名的字符级hash或one-hot+组合式编码）把几何链路跑通，T1再固定SigLIP checkpoint。

**关于checkpoint存放**：你提醒得对，`/var/tmp` 只适合可重建cache。**唯一的checkpoint和结果不要放那里**——放 `/home` 下的项目目录（虽然只剩42G，但checkpoint体量小，10–15M参数×21次×fp32约1.2GB量级），并保留manifest+hash。大体量的**帧缓存**才放 `/var/tmp/alphamotion`。

**关于viser**：同意lazy load，不要一次把七个模型都载进显存。

## E.3 v7新增：四项训练前就能算完的分析（不需要GPU，优先做）

spec §0.25列了七项已经算出来、并且**已经改变了设计**的东西（位置零空间8维、12 bits码率下界14.2mm、单码本VQ的码率天花板、姿态本征维数、gauge不变性、axis-bone关系、CSV单位）。下面四项**还没算，但同样能在写训练代码前给出答案**，且每一项都可能再改设计：

1. **`[BLOCK]` SOMA→G1 是不是一个函数？**
   在配对数据里，对每个SOMA姿态找 $\epsilon$-近邻（用归一化关节位置的L2），看对应G1姿态的**散布**。若同一个SOMA姿态对应显著不同的G1姿态，说明retargeter本身多解，**那个散布就是GT噪声，是精度的硬天花板**——模型再好也超不过。
   **这决定了"末端误差目标定多少才有意义"**，不做的话我们可能在追一个低于噪声地板的数字。产出：散布的分位数（中位/P90/P99），单位mm。

2. **`[BLOCK]` 16个对应站点能否唯一确定G1的29维？**
   用和spec §1.2.0同样的Jacobian秩分析，但只取那16个有语义对应的站点。**若秩<29（很可能），跨拓扑翻译在数学上就是欠定的**——缺的信息只能由先验补。这不是坏消息（那正是模型的价值），但它意味着**存在不可消除的误差下界**，报告里必须标出来。产出：秩、零空间维数、以及零空间对应哪些DOF。

3. **`[CONFIRM]` $K=8$ 对动物骨架合适吗？**
   Truebones关节数范围9–71。$K=8$对9关节的物种是**过参数化**，对71关节是**压缩17倍**。对各物种分别做PCA，看本征维数分布，判断单一$K$跨这个范围是否合理。产出：各物种本征维数（90%方差）的直方图。

4. **`[CONFIRM]` 采样出的约束组合是否可行？**
   给定一组稀疏SE3约束，目标骨架在**关节限位内是否存在解**？用URDF+IK优化器离线判定一批采样出的约束。**若某些组合本就无解，它们不该进训练集**——否则是在教模型拟合不可能的目标。产出：不可行约束组合的比例，以及典型失败模式。

**这四项加起来约一天CPU工作量，比先训21个模型再回头查便宜得多。** 结果请贴回群里，我据此再调spec。

## F. 文档优先级（避免冲突时无所适从）

| 文件 | 定位 |
|---|---|
| **`01_round1_spec.md`** | **实现的唯一依据。** 冲突时以它为准 |
| `00_codex_prep_checklist.md`（本文件） | 开工前准备与需实测的死角 |
| `02_stage1_project_plan.md` | 里程碑、验收标准、未定事项 |
| `round1_experiment_plan.md` | **推导底稿，非实现依据**。含已作废的方案（尤其§1.7的per-embodiment头），仅供追溯"为什么这么定" |
| `architecture_spec.md` / `architecture_final_design.md` / `math_framework.md` / `ablation_experiment_spec.md` | 更早期的架构讨论，**已被上述文件取代**，不要照着写代码 |

**遇到任何两份文档冲突，以 `01_round1_spec.md` 为准，并在群里提一下让我修。**