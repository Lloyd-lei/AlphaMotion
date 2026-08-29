# Round-1 实验计划：单帧"翻译官"瓶颈选型

> ## ⚠️ 这份文档是推导底稿，不是实现依据
>
> **实现请以 `01_round1_spec.md` 为准。** 本文档记录了完整的推导过程和所有实测数据（对追溯"为什么这么定"有用），但其中若干设计已在后续讨论中被推翻并替换，尤其是：
>
> - §1.7的"具身特定旋转头/关节角头"方案 **已作废**（违反零per-embodiment的scope约束），正确方案见 `01_round1_spec.md` §1；
> - 指标数量、关节对应关系的精确数字，以 `01_round1_spec.md` 为准。
>
> **v6（2026-07-30）后本文档新增的已作废项**，均以 `01_round1_spec.md` 为准：
>
> | 本文档写的（已作废） | 正确的 |
> |---|---|
> | FSQ隐式码本 640,000 | **2,560,000**；且v6已改配置为 `[4,4,4,4,4,4]` 对齐12 bits/token（spec §3.3） |
> | 瓶颈配置 V=1024 / R=4 | **VQVAE V=4096 / RVQ R=2,V=64**（码率对齐，spec §3.3） |
> | `T[is_ctrl] = ctrl_val`（约束赋给SE3 frame、拼在source端） | **纯target-side、位置/旋转分离的四张量契约**（spec §5.0） |
> | "用一个通用的root通道输出"全局root | **本轮不预测global root**（单帧下不可识别，spec §2.3） |
> | 输入含6D旋转 $r_j$ + `is_ctrl`/`ctrl_val`（14维） | **5维：$[p_j,\varphi_j,\texttt{phi\_avail}]$**（spec §2.1） |
> | "canonical绝对frame朝向跨rig可比" | **错**。需构造式gauge fixing（实测URDF link带16°固有旋转，spec §1.2） |
>
> 遇到两份文档冲突时，**一律以 `01_round1_spec.md` 为准**，并在群里提出来。

目标：在**单帧跨拓扑retargeting**任务上，确定 Direct-Token / AE / VAE / CVAE / VQVAE / RVQ / FSQ **七**种瓶颈中哪个最适合做"翻译官"。文本塔固定用SigLIP（不作为变量）。

**本文档所有数据端数字均为在arenalabs上实测得到，非引用。** 核实日期2026-07-30。

---

## 0. 实测到的数据现状（决定后续所有设计）

| 项目 | 实测值 |
|---|---|
| BONES-SEED总motion数 | 142,220（71,132原始 + 71,088镜像） |
| 总帧数 @120fps | 124,554,605 |
| 每条motion帧数 | min 69 / 中位数 684 / 均值 875 / max 21,617 |
| 演员数 | 522 |
| SOMA骨架关节数 | **78**（含Root、Jaw、LeftEye/RightEye、**每只手24个手指关节**（Thumb 4 + Index/Middle/Ring/Pinky 各5，均含End标记；若剔除End标记则每手19个）、头颈7个、脚趾4个） |
| G1 CSV列数 | **36 = 1(Frame索引列) + 6(root_translateX/Y/Z + root_rotateX/Y/Z) + 29(以`_dof`结尾的关节列)**（逐列实测：首列名为`Frame`，`_dof`结尾列计数=29） |
| 三种格式配对 | 逐帧严格对齐（抽查同一motion：BVH uniform 3769帧 = BVH proportional 3769帧 = G1 CSV 3769行） |
| package分布 | Locomotion 74,488(52%) / Communication 21,493 / Interactions 14,643 / Dances 11,006 / Gaming 8,700 / Everyday 5,816 / Sport 3,993 / Other 2,081 |
| body_position分布 | **standing 122,548(86%)** / sitting on floor 4,589 / crouching 2,590 / standing,leaning 2,512 / crawling 1,670 / 其余<1k |
| 磁盘 | /home 1.2T，**已用1.1T，仅剩42G** |
| GPU | RTX PRO 6000 Blackwell 97,887 MiB，**当前已占用24,475 MiB**（有别的任务在跑） |
| viser | 1.0.16，已装在 ardy / kimodo / kimodo-bones 三个env |
| G1 URDF | 已在本地：`~/.cache/newton/newton-assets_unitree_g1_308a72cd/unitree_g1/urdf/g1_29dof.urdf`（另有23dof/带手等变体） |

---

## 1. 【阻塞项】三个必须先解决的数据管线问题

这三项不是"设计选择"，是不解决就跑不起来的事实问题。

### 1.1 G1侧是关节角，不是3D位置 —— 必须做FK

G1 CSV给的是`left_hip_pitch_joint_dof`这样的**关节角度值**，而架构假设两侧都是SE3 frame $(\boldsymbol p_j,\boldsymbol r_j)$。要拿到G1的3D关节位置，必须把这29个角度通过`g1_29dof.urdf`做**正向运动学(FK)**。

- 这是一个此前所有架构文档都没提到的必需预处理步骤。
- 建议：用现成的URDF FK库（pytorch_kinematics可微，或直接用MuJoCo的mjcf——本地也有`g1_29dof.xml`）。**选可微的实现**，因为后面若要在loss里直接约束G1侧的3D位置，需要梯度能穿过FK。
- 决策点：**目标输出到底是29维关节角，还是FK后的3D frame？** 二者对应完全不同的loss和不同的解码头。我的建议是**同时输出**：解码器出SE3 frame，再加一个小head回归关节角，用FK(角度)≈frame做一致性loss——这样既保留SE3语义，又直接产出可下发到机器人的角度。

### 1.2 坐标系与手性：实测结论

这一节全部由**实跑FK**得出，不是从OFFSET字面推断的（BVH的OFFSET写在骨骼局部坐标系里——例如Hips→Spine1的offset是`(5.00, -0.04, -0.01)`，脊柱沿局部+X而非+Y——所以**不能直接读OFFSET判断世界朝向**，必须做完整FK）。

**SOMA BVH侧（实跑FK，第0帧世界坐标）**：

| 关节 | 世界坐标 (cm) |
|---|---|
| Root | (0, 0, 0) |
| Hips | (0.00, 101.28, 0.00) |
| Head | (0.33, 160.90, 2.45) |
| LeftHand | (21.82, 86.21, 4.44) |
| RightHand | (−20.78, 86.15, 6.49) |
| LeftFoot | (15.65, 8.33, −6.35) |
| RightFoot | (−18.25, 7.92, −4.50) |

推出：
- **Up轴 = +Y**（Head−Hips = `(0.33, 59.62, 2.45)`，Y分量占绝对主导）
- **Lateral轴 = X，且左侧为+X**（LeftHand−RightHand = `(42.60, 0.06, −2.05)`，符号为正）
- **Forward轴 = +Z**。右手系下 $\text{forward}=\text{left}\times\text{up}$，即 $\hat X\times\hat Y=\hat Z$。
- **手性：右手系**。用实测FK数值验证：up=归一化(Head−Hips)=`(0.01, 1.00, 0.04)`、left=正交化后的(LeftHand−RightHand)=`(1.00, 0.00, −0.05)`、fwd=left×up=`(0.05, −0.04, 1.00)`，则 $\det[\text{fwd},\text{left},\text{up}] = +1.0$。
- 单位 cm（Hips高101.28cm）。

**G1侧（解析`g1_29dof.urdf`）**：
- `waist_yaw_joint`、`left_hip_yaw_joint` 的axis均为 `(0,0,1)` → **yaw绕Z ⇒ Up轴 = +Z**
- `left_hip_pitch_joint`、`left_knee_joint`、`left_shoulder_pitch_joint` 的axis均为 `(0,1,0)` → **pitch绕Y ⇒ Lateral轴 = Y**
- `left_hip_pitch_joint` origin = `(0, +0.064452, −0.1027)`，`right_hip_pitch_joint` origin = `(0, −0.064452, −0.1027)` → **左侧为+Y**
- 膝origin的Z为负（`−0.17734`，在髋之下）→ 确认 **+Z朝上**
- URDF为**标准ROS右手系 x-forward / y-left / z-up**，单位**米**
- URDF共39个joint，其中**revolute 29个**（与CSV的29个`_dof`列吻合）
- CSV的root平移首帧 `(0.030, 1.816, 77.93)`：**注意这一列的单位/语义与URDF不一致**（77.93若为cm则是骨盆高度、Z-up；但URDF用米）。**这一列的确切单位与轴序必须在写dataloader前用一整条轨迹验证**（检查哪一维在行走时单调变化=前进方向、哪一维基本恒定≈骨盆高度），不要照搬本表。

**结论：两侧都是右手系，但Up轴不同（SOMA为+Y，G1为+Z），Lateral轴也不同（SOMA为X-左正，G1为Y-左正）。** 这是一个**纯轴置换、不含镜像**的变换（行列式为+1），因此**不会引入左右翻转**——这一点很关键，如果变换里混入了反射（det=−1），模型会系统性地学出镜像错误且极难察觉。

统一到SOMA的Y-up右手系，变换为：
$$(x,y,z)_\text{SOMA} \;=\; (\,y_{G1},\;\; z_{G1},\;\; x_{G1}\,) \times 100\ \text{(m→cm)}$$

即矩阵
$$M_{\text{G1}\to\text{SOMA}}=\begin{pmatrix}0&1&0\\0&0&1\\1&0&0\end{pmatrix},\qquad \det M = +1$$

逐项语义核对（代入单位向量验证）：G1的 x-forward $\to$ SOMA的 `(0,0,1)`=+Z-forward；G1的 y-left $\to$ SOMA的 `(1,0,0)`=+X-left；G1的 z-up $\to$ SOMA的 `(0,1,0)`=+Y-up。三项全部吻合，且行列式为+1，**是纯循环轴置换，不含反射**。

> **修订记录**：本文档v2曾把该式写成 $(-y_{G1}, z_{G1}, x_{G1})$，那个式子的行列式是**−1**（是一个反射），正是本节警告要避免的镜像错误。已更正为上式（去掉负号）。这也说明下面这四项验收检查不是形式主义——第1项就能直接抓住这个错误。

**必做的验收检查**（写进单元测试，不要靠肉眼）：
1. 变换矩阵 $\det = +1$（非−1）；
2. 变换后G1的左髋在+X侧、头部在+Y最高处；
3. 取一条行走motion，SOMA侧与G1侧的**前进方向**在变换后一致；
4. 取一条明确的单侧动作（如"左手挥手"），确认两侧动的是同一只手——**这是唯一能真正抓住镜像错误的检查**。

仓库里已有`data/fix_amass_axis.py`、`data/diag_axis.py`，说明这类坑踩过，建议复用其约定。

### 1.3 拓扑不对称极其严重

| | SOMA | G1(29dof) |
|---|---|---|
| 关节总数 | 78 | 29 dof |
| 头部 | Neck1, Neck2, Head, HeadEnd, Jaw, LeftEye, RightEye（共7） | **无任何头/颈自由度** |
| 手指 | **每只手24个关节**（Thumb 4 + Index/Middle/Ring/Pinky 各5，含End标记；剔除End后每手19） | **无** |
| 脚趾 | LeftToeBase/ToeEnd, RightToeBase/ToeEnd | **无** |
| 腰 | Spine1, Spine2, Chest | waist_yaw/roll/pitch (3) |

SOMA→G1丢弃**59个**关节的信息（双手手指48 + 脚趾4 + 头颈7 = 59，逐项相加所得；78−59=19个SOMA关节在G1侧有语义对应）；G1→SOMA必须凭空"脑补"出手指、头部朝向、脚趾。**这不是可以糊过去的细节，直接决定"重建误差"这个指标本身的含义。**

必须现在决定的策略（三选一，我推荐第2个）：
1. 只在两边共有的语义关节子集上计loss（约22-24个关节），其余关节完全不参与——最干净，但浪费了SOMA侧丰富的手部信息。
2. **分层loss**：共有子集用$\mathcal L_\text{direct}$全权重监督；SOMA独有关节（手指/头/脚趾）在decode到G1时用mask屏蔽、不计loss，但在SOMA→SOMA自重建路径上仍计loss——这样翻译官仍学到手部synergy，只是不强求G1侧有对应输出。**推荐此项。**
3. 给G1侧补虚拟关节（用固定几何位置代表头部/手部）——引入虚假监督信号，不推荐。

---

## 1.5 速度通道：单帧翻译官应当**去掉**速度

结论：**去掉**。理由三条，其中第一条是决定性的。

1. **速度会污染跨具身瓶颈的语义。** 速度$\boldsymbol v_j=\boldsymbol p_j^{(t)}-\boldsymbol p_j^{(t-1)}$依赖帧率。BONES-SEED是120fps，若将来接入其他帧率数据，同一个动作的$\boldsymbol v$会差数倍，而$\boldsymbol p$不变。让瓶颈token同时编码"姿态"和"帧率相关的运动量"，会削弱它作为纯粹姿态语义空间的性质。
2. **单帧任务里速度不提供任何目标信息。** 本轮的监督信号是"给定源骨架姿态，输出目标骨架姿态"——目标姿态由源姿态唯一确定（在retargeter的意义上），与它运动得多快无关。速度是**与标签无关的输入噪声**，只会增加过拟合风险。
3. **省掉一个工程坑。** 有速度就必须保证采样出的帧有合法前驱帧（motion首帧、跨motion边界都要特判）。去掉后**真正做到单帧**，dataloader可以直接对(motion_id, frame_idx)随机索引，无任何邻接约束——这对§3.1的分层采样实现简化很大。

**注意这与SAMoR的做法不同，且差异是有理由的。** SAMoR原文（§3.1）明确写道：跨拓扑设定下局部旋转相对各自rest pose定义、不可跨骨架比较，因此它**只用root-normalized的位置与速度** $x_j^t=[p_j^t, v_j^t]\in\mathbb R^6$ 作为跨rig共享信号（全特征设定下才是$[p,v,R]\in\mathbb R^{12}$）。但SAMoR处理的是**时序chunk**（$T$帧、4倍时间下采样），速度在那里承载真实的时序信息；我们这一轮是**严格单帧**，同样的理由不成立。

因此本轮输入特征改为：
$$x_j = [\;\boldsymbol p_j \in\mathbb R^3,\;\; \boldsymbol r_j \in\mathbb R^6,\;\; \text{rot\_avail}_j,\;\; \text{is\_ctrl}_j,\;\; \text{ctrl\_val}_j\;]$$
（$\boldsymbol r_j$为6D旋转，仅在有明确rest pose的具身上可用、由`rot_avail`标记；动物/机械rig该通道置零并mask——详见§1.7。）

**这也顺带回答了架构文档里"最小时间窗=2帧,非严格单帧"那句话——本轮把它改回严格单帧。** 速度通道留到时间层再引入。

---

## 1.7 旋转要不要编码：**要，但方式必须和SAMoR不同**

### SAMoR原文怎么做的（已核实）

- **全特征表示**是 $x_j^t=[p_j^t, v_j^t, R_j^t]\in\mathbb R^{12}$（$R$为6D旋转表示）。
- **但在跨拓扑设定下，它主动丢掉了旋转**，原文理由："local rotations are defined relative to rig-specific rest poses and are not directly comparable across arbitrary skeletons"，因此只用 root-normalized 的 $[p_j^t,v_j^t]\in\mathbb R^6$ 作跨rig共享信号。
- 旋转重建loss **只在 HumanML3D-compatible（单一SMPL骨架）设定下才加**。
- 全局root平移也被排除，理由是异构资产的参考系与rest pose定义不一致；HumanML3D那一路用一个独立的轻量VQ分支单独建模root通道。
- 需要旋转时（给mesh做LBS蒙皮动画），靠**IK后处理**从位置反解。

### SAMoR自己承认这是个未解决的问题

原文Limitations一节明确写道：跨拓扑设定下它只处理关节位置而非旋转，因为异构资产缺乏统一的rest-pose参考；为LBS恢复骨骼旋转需要IK后处理，而这会引入**twist ambiguity（绕骨轴的扭转歧义）**，并称"accurate twist recovery from positions alone remains an open problem"。

### 对动画可以忍，对机器人不行——我做了量化验证

Twist ambiguity的本质是：**绕骨骼自身轴线的旋转不改变任何子关节的位置**，因此从位置信息中原理上不可恢复。我用`g1_29dof.urdf`做了完整FK，逐个DOF施加0.2 rad扰动，测量它能引起的最大关节位置位移：

| DOF | 最大关节位移 @0.2rad |
|---|---|
| left/right_ankle_roll | **0.000 mm** |
| left/right_wrist_roll | **0.599 mm** |
| left/right_ankle_pitch | 3.506 mm |
| （其余23个DOF） | 8.3 – 130.6 mm |

**G1的29个自由度里，有4个从关节位置中完全观测不到**（双踝roll精确为0，双腕roll约0.6mm），另有双踝pitch仅3.5mm、近乎不可观测。

其中 **ankle_roll 是踝关节侧倾，直接决定足部接触面的姿态与横向平衡**——对一个要下发到真实机器人的模型，这不是可以四舍五入掉的自由度。而这正好落在你反复强调的"SE3精确控制精度"上：**SE(3)按定义就包含旋转，只输出位置等于把SE(3)降级成$\mathbb R^3$。**

### 因此：一个通用头输出canonical SE(3) frame，关节角靠解析投影

> **订正（重要）**：本节早期版本提出的是"位置共享瓶颈 + 具身特定旋转头/关节角头"。**那个方案违反了项目最核心的scope**——一次unified universal预训练、零per-embodiment模块。`rot_head[embodiment]` / `angle_head[embodiment]` 意味着接入第N+1个机器人就要加参数、要新GT、要重训。已作废，替换为下面的方案。以文件 `01_round1_spec.md` §1 为准。

**正确解法**：网络输出**canonical空间的per-joint SE(3) frame**$T_j=(R_j,\boldsymbol t_j)$，**一个通用头，对所有拓扑逐字节相同**。

两条实测支撑：
1. **canonical frame承载了位置丢失的twist信息**。同一扰动测试同时测两个量：`ankle_roll`的位置位移是**0.000 mm**，但frame朝向变化是**11.459°**（正是0.2 rad）。twist在位置里不可见，在frame里完全可见。
2. **关节角是canonical frame的闭式函数，零学习参数**：
$$R_j^{\text{rel}}=R_{0,j}^\top R_{\text{parent}(j)}^\top R_j,\qquad \theta_j=\mathrm{sign}(\boldsymbol\omega\cdot\hat a_j)\arccos\frac{\mathrm{tr}(R_j^{\text{rel}})-1}{2}$$
实测G1的29个revolute DOF，**最大误差4.29e-12 rad、平均6.67e-13 rad（机器精度），学习参数0个**。换任何机器人只需换URDF。

**为什么canonical旋转跨拓扑可比、而SAMoR说的局部旋转不可比**：SAMoR针对的是**相对各自rest pose的局部旋转**（语义依赖rig特定的rest pose定义）。canonical空间的**绝对frame朝向**只是在回答"这根骨在共享对齐空间里朝哪"——与位置同样可比。**换表示就绕过了SAMoR避开的问题，不需要per-embodiment头。**

**旋转loss按GT可用性mask**（动物无可比旋转GT则置零）——**mask一个loss项 ≠ 加一个头**，网络权重对所有具身完全相同。

以下为已作废的旧方案，仅作追溯保留：

```
共享瓶颈 z  ←  只由位置驱动（照搬SAMoR，这正是动物骨架能接进来的原因）
              ↓
        ┌─────┴─────┐
   位置解码头        旋转解码头（具身特定）
   所有具身通用       仅在有明确rest pose的具身上激活并监督
                     ├─ SOMA-78  ✓（BVH自带局部旋转GT）
                     ├─ G1-29    ✓（CSV即关节角，是旋转的另一种参数化）
                     └─ 动物/机械rig ✗（rest pose不可比，不监督，y=-1式跳过）
```

这不是我发明的——**这就是SAMoR"rotation loss added only in HumanML3D-compatible setting"的推广**：它在唯一有统一rest pose的那个子集上加旋转监督，我们把这个子集从"1个SMPL骨架"扩展到"SOMA + G1"这两个rest pose定义明确的具身。

**输入侧**：$\boldsymbol r_j$ 仅作为**可选输入通道**，配一个availability mask；动物样本该通道置零并mask。这样跨拓扑共享信号仍是纯位置（与SAMoR一致），不破坏动物骨架的接入。

**关键的额外评估项**：必须**单独报告 twist DOF 的误差**（G1的双踝roll、双腕roll），不能混在29维平均误差里——它们在平均值里只占4/29，即使完全学错也看不出来。

---

## 1.8 坐标表示：为什么用root-relative，而不是世界坐标、也不是逐骨父子相对

先澄清一处措辞——**本方案从来没有用世界坐标**。§2.1的输入定义写的是 `p_j: root-relative 位置`。世界坐标是绝对不能用的：G1骨盆离地78cm、人体髋101cm，同一个"举手"动作在世界坐标下数值完全不同，跨具身无从对齐；而且整个人平移一米，所有关节坐标全变，模型要浪费容量去学"平移不改变姿态"这件本该免费的事。

所以真正的问题是你问的后半句：**root-relative（相对根节点）vs 逐骨父子相对（bone-vector / 局部链式）**，选哪个。

**SAMoR选的是root-relative**——原文原话："we therefore use **root-normalized** joint positions and velocities, $x_j^t=[p_j^t,v_j^t]\in\mathbb R^6$, as the shared cross-rig signal"，且"Global root translation is **excluded** because heterogeneous assets have inconsistent reference frames and rest-pose definitions"。

**我也建议root-relative，理由是误差累积**。父子相对表示的问题在于：每根骨的误差沿运动学链**累加**才得到末端位置。实测两侧的链深度：

| 关节 | 链深度 |
|---|---|
| SOMA `LeftFoot` | 4 |
| SOMA `Head` | 7 |
| SOMA `LeftHand` | 8 |
| SOMA 指尖 `LeftHandIndex4` | 12 |
| SOMA 最大链深 | **13** |
| G1 `ankle_roll_link` | 6 |
| G1 `wrist_yaw_link` | 10 |
| G1 `rubber_hand` | 11 |
| G1 最大链深 | **11** |

若每根骨有独立误差 $\epsilon$，深度 $d$ 处的末端误差按 $\epsilon\sqrt d$ 增长：

| 每骨误差 | LeftFoot(d=4) | Head(d=7) | LeftHand(d=8) | G1 hand(d=11) | 指尖(d=12) |
|---|---|---|---|---|---|
| 1 mm | 2.00 mm | 2.65 mm | 2.83 mm | 3.32 mm | 3.46 mm |
| 2 mm | 4.00 mm | 5.29 mm | 5.66 mm | 6.63 mm | 6.93 mm |

**而"末端位置误差"恰恰是我们的第一评估指标**（§7维度A第1项）。父子相对表示等于把误差最大的地方放在最在乎的量上。root-relative下每个关节的监督是**直接的**，不经过链式累加。

**但父子相对不是没用，它应该以loss的形式出现**：这正是 $\mathcal L_\text{bone}$（骨长一致性，父子距离的平方误差）的作用——**表示用root-relative，约束用父子相对**。SAMoR给这一项的权重是**2.0，比位置loss的1.0还高**，且原文说在Table 3做过消融。两者不矛盾，是互补的：root-relative保证末端精度，$\mathcal L_\text{bone}$保证骨骼不被拉伸。

**三条必须配套的归一化**（否则root-relative仍然跨具身不可比）：
1. **按骨架高度$H$归一化尺度**——G1和人体身高不同，不归一化则同一动作数值域差一截。
2. **对齐朝向**——SAMoR原文说所有角色都做了"normalized by mesh bounding box, up-axis, and facing direction"。我们至少要做up-axis和facing direction的对齐（见§1.2的坐标变换）。
3. **rest-pose几何单独走anchor通道**（§4的几何anchor），那里**才**用父子相对的骨向量$\boldsymbol b_j=\boldsymbol p_j^{\text{rest}}-\boldsymbol p_{\text{parent}(j)}^{\text{rest}}$——因为anchor描述的是"这具骨架长什么样"，父子相对正是描述形状的自然方式；而motion描述的是"这一帧摆成什么姿势"，root-relative才是自然方式。**两者用不同表示是有意为之，不是不一致。**

### 输出格式（回答"输出什么参照SAMoR"）

| | SAMoR | 本轮 |
|---|---|---|
| 主输出 | root-relative关节位置 + 速度 | **root-relative关节位置**（单帧，无速度） |
| 旋转 | 跨拓扑不输出，靠IK后处理 | **canonical frame的朝向，同一个通用头输出**；loss按GT可用性mask |
| 关节角 | 不输出 | **解析投影**（零学习参数，实测机器精度），不是头 |
| 全局root平移 | **排除**（异构资产参考系不一致），HumanML3D路用独立VQ分支 | 同样排除出瓶颈；用**一个通用root通道**输出（root是$SE(3)$刚体变换，语义对任何骨架相同，本不需per-embodiment处理） |
| loss | smooth-$\ell_1$(位置) + smooth-$\ell_1$(速度) + VQ commitment + 骨长一致性 + 时序平滑 + attention supervision；权重分别为 1.0 / 0.5 / 1.0 / 2.0 / 0.01 / 1.0 | smooth-$\ell_1$(位置) 1.0 + 骨长一致性 2.0 + attention supervision 1.0 + **旋转测地loss（通用头输出，按GT可用性mask）** + §5的约束一致性loss + FK一致性loss；**去掉速度loss与时序平滑loss**（单帧无时序） |

注意SAMoR的**骨长一致性loss权重是2.0，比位置loss的1.0还高**，且原文说它在Table 3中做过消融——这说明在跨拓扑场景下，保证骨骼长度不被拉伸比精确命中位置更重要。这一条值得直接照搬。

---

## 1.6 【补】第七个候选：不做任何量化的直接tokenize

原六变体清单漏了一个，补上。这一项与AE的区别必须说清楚，否则会变成重复实验：

| | **AE（第1个变体）** | **Direct-Token（第7个变体）** |
|---|---|---|
| 瓶颈 | $K$个part query对$J$个关节做cross-attention pooling → $z\in\mathbb R^{K\times D}$ | **无pooling**：每个关节自己就是一个token，$z\in\mathbb R^{J_\text{src}\times D}$ |
| 压缩 | 有（$J\to K$，如78→8） | **无**（$J\to J$，恒等） |
| 跨拓扑机制 | 靠$K$个与拓扑无关的摘要卡 | 靠decoder的cross-attention直接查源关节token |
| 风险 | 压缩可能丢SE3精度 | **可能退化成"逐关节抄近邻"**，学不到真正的跨拓扑语义 |

这一项是**极其重要的对照组**——它直接检验你最担心的那件事："压缩到潜空间会不会损失SE3精度"。如果Direct-Token在SE3精度上明显优于其余六个，说明压缩确实有代价；如果它在`val_topo`（未见拓扑）上明显更差，则说明压缩正是泛化能力的来源。**没有这一组，"要不要压缩"这个问题无法被回答，只能靠信仰。**

SAMoR原文已经给了一个相关证据：其消融显示$K=8$优于$K=1$和$K=J$（逐关节、无压缩），且论文明确指出naive训练下part query会**塌缩**——"queries attend broadly and produce redundant representations, reducing K=8 to the effective equivalent of K=1"，这正是它引入attention supervision loss的原因。但那是在动画重建任务上得到的，**是否适用于我们的SE3精确控制场景需要自己验证**。

**所以候选总数是7个**：Direct-Token / AE / VAE / CVAE / VQVAE / RVQ / FSQ。
建议再加一档 **$K$的扫描**（$K\in\{4,8,16\}$）——但只在T1胜出的那1-2个变体上做，不要7×3全扫。

---

## 2. 【核心问题】只有一个具身对，测不出"跨具身"

这是我认为**当前计划最大的结构性问题**，比上面三个技术坑更重要。

现在只有SOMA→G1一个真配对。而架构里的$\mathcal L_\text{swap}$（cycle-consistency）和$\mathcal L_\text{adv}$（具身判别器对抗）**都需要≥3个具身才有意义**：判别器只有2类时几乎是平凡任务，swap只有一条路径时退化成自重建。**结果就是：六个变体全部只在一条翻译方向上被评估，"哪个瓶颈更适合跨具身"这个问题根本没被测到。**

### 2.0 SAMoR用的是什么骨架？（已核实原文）

直接从PDF正文核实（非转述）。SAMoR的语料是**三个数据集拼起来的异构corpus**：

| 来源 | 内容 | 骨架多样性 |
|---|---|---|
| HumanML3D | 约23K条SMPL-22人形motion-caption对，263维标准特征格式 | **1种**骨架 |
| Truebones Zoo | 动捕库，**65个动物物种**（四足、鸟类、爬行类、昆虫），关节数各不相同 | **65种** |
| Objaverse-XL（自行筛选子集） | 约71k个带骨骼绑定与蒙皮的动画角色，涵盖人形、四足、非生物rig | **约71k种** |

原文汇总为"约71k个角色、约100k条motion，来自1个SMPL骨架 + 65个动物物种 + 约71k个Objaverse-XL动画角色"，并做**character-level切分**，保证验证角色训练时从未见过。

**这个发现对我们的处境有直接启示**：SAMoR之所以能声称"any skeleton and topology"，是因为它手上有**七万种**不同拓扑。我们原计划只有**两种**（SOMA-78、G1-29），加上合成剪枝也就5种左右——和71k不是同一个量级。

**这正是引入外部异构骨架的决定性理由**。基线拓扑数为**5种**（SOMA-78 + G1-29 + 3个合成剪枝变体）。SAMoR原文Table 1给出了两个候选数据集的精确规模：

| 数据集 | 训练实体 | 验证实体 | 训练motion | 验证motion | **关节数范围** |
|---|---|---|---|---|---|
| HumanML3D | 1 (SMPL) | 1 (SMPL) | 23,384 | 4,384 | 22 |
| Truebones Zoo | **54物种** | **11物种** | 939 | 110 | **9–71** |
| Objaverse-XL（SAMoR自建） | **67,646** | **3,528** | 67,646 | 3,528 | **5–96** |

加入Truebones Zoo后：5 + 65 = **70种**拓扑。（详见§2.0b关于选哪个数据集的讨论。）

### 2.0b Truebones Zoo 还是 Objaverse-XL？

**结论：本轮选 Truebones Zoo，不选 Objaverse-XL。** 但理由不是"OXL更难搞"，而是**OXL和我们的目标不匹配**。

**Objaverse-XL确实有骨架**——SAMoR原文明确说它用的是"our curated subset of **rigged and skinned** animated characters from Objaverse-XL"，涵盖humanoid、quadruped、non-biological rigs，关节数范围5–96，规模67,646个训练角色。所以"有没有骨架"这个问题的答案是：**有，但需要自己筛**（原始Objaverse-XL里绝大多数资产没有绑定，SAMoR是自己curate出这个子集的，且用Gemini Flash做了视频渲染+VLM打caption——这套curation pipeline本身就是相当大的工程量）。

**关键取舍——你说了"我们主要面对的是人形"**：

| | Truebones Zoo | Objaverse-XL |
|---|---|---|
| 拓扑数 | 65物种 | 67,646角色 |
| 关节数范围 | 9–71 | 5–96 |
| **数据获取** | 现成动捕库，直接下载 | **需自行从海量资产中筛选绑定+蒙皮的**，SAMoR的curation含VLM打caption |
| **motion数量** | **仅939训练/110验证**（很少！） | 67,646/3,528 |
| **运动质量** | 真实动捕 | **程序化/手工animation为主**，非动捕 |
| **与人形的相关性** | 中（真实生物运动学，但四足为主） | **低**（含大量non-biological rigs，如机械臂、抽象物体） |
| SAMoR的规范关节名覆盖率 | **81%** | **69%** |

选Truebones的三条理由：
1. **OXL含大量non-biological rigs**（原文原话），对"主要面对人形"的目标是**噪声不是信号**——让编码器去拟合抽象物体的运动，会稀释人形/生物运动学的先验。
2. **OXL的运动多为程序化animation而非动捕**，物理合理性远差于Truebones，而我们最终要下发到真机。
3. **工程量差一个量级**：Truebones是现成动捕库；OXL要自己从Objaverse-XL海量资产里筛rigged+skinned子集。

但要清楚Truebones的**代价**：它的**motion数量极少**（939训练/110验证），远小于BONES-SEED的142,220条。所以它的作用是**提供拓扑多样性、而非数据量**——训练时需要对它做**过采样**，否则在142k条人形motion面前它会被完全淹没，$\mathcal L_\text{adv}$和`val_topo`都会失效。

建议采样配比：人形（SOMA/G1/剪枝）: 动物 ≈ **7:3**（按帧数计，不是按motion计）。这个数字需要在T1做一次敏感性检查。

**注意SAMoR的一个反直觉消融结果**：去掉关节名条件（name-ablation）只让性能下降13–16%，原文明确说"confirming SAMoR does not rely on joint-name strings as a shortcut"，且指出规范标签覆盖率在H3D是100%、OXL只有69%、Zoo只有81%。**这对我们是好消息**——说明动物关节名标不全（甚至完全不标）不会毁掉模型，几何结构本身承载了大部分信息。也侧面说明SigLIP这一路的收益上限可能没想象中高，不要在它身上过度投入。

---

**决定：本轮引入Truebones Zoo（动物骨架）。** 它把拓扑多样性从5种提到70种，直接消解了上面这个最大的结构性问题。但关于"那机械狗的动作生成不就顺带解决了"这个推论，需要分清三件事：

**能得到的**：一个**见过四足/鸟类/爬行类拓扑的编码器**。这让"任意骨架"不再是空话，也让$\mathcal L_\text{adv}$（具身判别器）和swap-consistency真正有活干。这是必要前提。

**得不到的（还差三步）**：
1. **没有四足机器人侧的配对GT。** SOMA→G1这条线之所以能训，是因为NVIDIA的retargeter提供了逐帧配对。狗→Go2没有这样的东西，需要另外构建（要么找现成的四足retargeting数据，要么自己用IK生成）。
2. **动物→机械狗的gap不比人→人形小，反而更大**：动物脊柱是柔性多节的、机械狗躯干是刚性的；四足步态相位（trot/bound/gallop）与机器人可执行的步态集合不重合；爪/蹄与足端接触模型不同。
3. **动物动捕的物理质量通常差于人体动捕**（穿模、脚滑、骨长不守恒更常见），而SAMoR是做动画的、对这些容忍度高，机器人不行。

我注意到机器上有`go2-convex-mpc`这个conda环境，说明你确实有四足线在跑。**这条路是通的，但它是Round-2的任务：先用Truebones把编码器的拓扑覆盖打开，再单独补一份Go2配对数据做四足侧的验证。** 本轮不要把"机械狗动作生成"写进目标，否则会因为缺GT而无法验收。

**引入动物骨架的三项额外工程量**（都不难但不能漏）：
- **数据获取**：Truebones Zoo磁盘上目前没有（已搜索确认），需下载。Objaverse-XL子集（71k角色）体量大且需自行筛选绑定资产，**本轮建议跳过，只上Truebones Zoo的65个物种**。
- **功能组标签**：SAMoR的attention supervision需要每个关节的功能组标签$y_j\in\{0,\dots,K-1\}\cup\{-1\}$，从关节名自动导出。**原文给了很关键的兜底机制**：匹配不上的关节取$y_j=-1$，对supervision loss贡献为零，但仍然被编码、由学到的注意力自由路由；对于完全没有规范关节名的机械rig/程序化资产，所有关节都是$-1$，这时$K$个slot退化为纯无监督的功能桶，只靠运动学结构和运动模式驱动。**这意味着动物关节即使标不全也不会卡住流程**，可以先只标能确定的（四肢、脊柱、头、尾），其余全给$-1$。
- **旋转通道对动物一律mask**（见§1.7）——动物rig没有可比的rest pose，这也正是SAMoR排除旋转的原因。
- 你提到的**ARDY/Kimodo可以编辑骨架**——这条路有一个必须先弄清的限制：它们编辑的是**joint angle而不是末端位置**（你自己也指出了这点）。这意味着用它们生成的变体骨架，其"GT"仍然是同一个底层生成模型的输出，**拓扑本身并不变**（还是G1-29或SOMA的关节集），变的只是姿态。所以它们能提供**姿态多样性**，但**提供不了拓扑多样性**——不能替代§2的剪枝方案。若要用，定位应是"扩充稀有姿态（蹲/爬/坐）的采样"，而非"增加具身种类"。

### 2.1 现有数据里的两条多样性轴

好消息是数据里有现成的解法，两条正交的多样性轴：

**轴A：骨长/比例多样性（免费，已存在）**
`soma_uniform`（所有演员统一骨架比例）vs `soma_proportional`（**每个演员各自的真实身材比例，522种不同比例**），且metadata里带完整人体测量学参数（身高/足长/锁骨高度/肘展/腕展/肩宽/髋高/膝高/踝高/体重/年龄）。
→ 这直接给了522+1种"骨长不同但拓扑相同"的具身，可以立刻用于测试骨长泛化，无需任何额外数据。

**轴B：拓扑多样性（需人工构造，成本低）**
从SOMA-78按语义剪枝出若干合成拓扑：
- `SOMA-78` 全量
- `SOMA-38`：去掉全部手指关节（保留LeftHand/RightHand）
- `SOMA-24`：再去掉Jaw/双眼/脚趾/Neck2
- `SOMA-20`：再合并Spine1/Spine2/Chest为单节点
- `G1-29`（真实机器人）

这样得到5种拓扑 × 522种骨长比例，判别器和swap-consistency才真正有活干。**剪枝是纯确定性操作，不引入任何虚假数据，且完全在现有数据上完成。**

**建议：把这一条列为Round-1的前置任务，否则这一轮实验的结论对"跨具身"这个核心命题没有说服力。**

---

## 3. 数据采样与切分

### 3.1 帧的采样

- **去镜像**：71,088条镜像motion是原始motion的精确镜像，作为训练增广可以用，但**绝不能让一条motion和它的镜像分处train/val**（等价于标签泄漏）。基础池只取71,132条原始motion，镜像仅在训练侧做增广。
- **时间去冗余**：120fps下相邻帧几乎相同，全量124M帧里绝大多数是重复信息。按**每0.25秒取1帧（每30帧取1）**采样 → 原始motion侧约 $62{,}277{,}000/30 \approx 2.08\text{M}$ 帧可用池。
- **分层采样对抗类别失衡**：naive随机采样下86%是standing、52%是Locomotion，模型基本见不到crouching/crawling/sitting。按 `(package × body_position)` 联合分层，对稀有类过采样，使每个bucket至少占1-2%。**这一条对SE3精度影响很大**——蹲姿/坐姿/爬行恰恰是关节角度分布最极端、retargeting最容易失败的区域。

### 3.2 三重切分（不是简单的train/val）

| Split | 构造方式 | 检验什么 |
|---|---|---|
| `train` | ~450演员 × 分层采样帧 | — |
| `val_id` | 同训练演员、同motion，**held-out帧**（不同时间段） | 拟合是否正常 |
| `val_actor` | **完全held-out的~50个演员**（含极端身材：最高199cm/最矮145cm/最重145kg各留几个） | 骨长/体型泛化 |
| `val_topo` | 训练时未出现的拓扑：**held-out的动物物种（约10个物种完全不参与训练）** + 1个合成剪枝拓扑（如SOMA-24） | **拓扑零样本泛化——这才是"翻译官"的核心卖点** |

注意SAMoR用的是**character-level切分**（保证验证角色训练时从未见过）——我们对动物同样要按**物种**切分，不能按motion切分，否则同一物种的其他motion泄漏进训练集，`val_topo`就失去意义。

`val_topo`是本轮最重要的指标；如果七个变体在`val_id`上难分伯仲（很可能），差异会在`val_topo`上显现。

### 3.3 数据量分级（回答"多少数据能初见端倪"）

| 阶段 | 帧数 | 目的 | 预计单变体耗时 |
|---|---|---|---|
| T0 冒烟 | 5k | 跑通pipeline、shape检查、loss下降 | <10分钟 |
| **T1 初见端倪** | **200k** | **区分七个变体的相对优劣** | **约0.5-1小时** |
| T2 定论 | 1-2M | 确认T1结论、给最终数字 | 4-8小时 |

**T1取200k的理由**：单帧pose→pose是一个低内在维度的回归问题（人体姿态流形的有效维度约50-80），不是需要海量数据的生成任务；200k帧在分层采样下能覆盖各pose类别各约2-4k样本，足以让七个变体的差异显现。**先跑T1，看七个变体的排序是否稳定（跨3个seed），稳定了再上T2；不要一上来就烧1-2M。**

**磁盘（已重新核实，你说得对，另一个分区很充足）**：

| 挂载点 | 设备 | 容量 | 已用 | 可用 |
|---|---|---|---|---|
| `/home` | nvme0n1p2 | 1.2T | 1.1T | **42G**（97%，紧张） |
| `/` | nvme0n1p5 | 657G | 195G | **428G**（32%，充裕） |

**但有一个坑**：`/`分区上大部分目录（`/opt`、`/srv`、`/usr/local`）需要sudo，且`sudo -n`测试显示**需要密码**（我无法非交互获取）。目前在`/`上**可直接写入的是 `/var/tmp` 和 `/tmp`**（均在nvme0n1p5，428G可用）。

`/tmp`通常在重启时被清空，**不适合放训练缓存**；`/var/tmp`按FHS约定在重启后保留，是更合适的选择。

**建议**：请你用sudo建一个正式的数据目录（例如 `sudo mkdir -p /data/alphamotion && sudo chown arenalabs:arenalabs /data/alphamotion`），比用`/var/tmp`更规范。在那之前，T1的缓存约0.56GB放`/home`也完全够用（$200{,}000\times78\times9\times4\text{B}$）。T2的1-2M帧加G1 FK结果会到~10GB量级，那时就需要用到`/`分区了。无论如何**不要试图缓存全部124M帧**。

---

## 4. 骨架标签设计（回答"怎么针对不同骨架打标签"）

每个关节token的身份信息由三部分构成，**都不是可学习的自由参数表**（否则新拓扑零样本时无法查表）：

1. **SigLIP文本embedding（冻结）**：关节名字符串 → SigLIP文本塔 → 固定向量。
   **必须先做语义规范化**，不能直接喂原始字符串：
   - SOMA的`LeftLeg`实际是**大腿**（LeftShin的父节点），字面读作"左腿"会误导文本塔；
   - G1的`left_hip_pitch_joint`需转成自然语言`left hip pitch joint`；
   - 建议手工维护一张 `原始名 → 规范自然语言描述` 的对照表（78+29项，一次性工作），例如
     `LeftLeg → "left thigh, upper leg"`、`LeftShin → "left shin, lower leg"`。
   - 训练时对该embedding做随机dropout（置零），强制模型不能只靠名字、必须用几何信息——这也是应对新拓扑中未见关节名的关键。
2. **拓扑位置编码**：kinematic tree上的结构信息，作为attention bias注入——hop距离（Floyd-Warshall算最短路后截断）、是否同一运动链、边类型。**不是**LLM那种线性位置编码，因为骨架是树不是序列。
3. **几何anchor（相对量，非绝对坐标）**：
   $$\text{anchor}_j = \text{MLP}\Big(\big[\;\text{Fourier}(\hat{\boldsymbol b}_j),\;\; \|\boldsymbol b_j\|/H\;\big]\Big),\qquad \boldsymbol b_j = \boldsymbol p_j^{\text{rest}}-\boldsymbol p_{\text{parent}(j)}^{\text{rest}}$$
   其中$\hat{\boldsymbol b}_j$为单位骨向量（尺度无关）、$\|\boldsymbol b_j\|/H$为按骨架总高归一化的骨长、$H$取骨架高度。**Fourier位置编码不可省**：MLP直接吃低维原始坐标存在频谱偏置，对几何细节不敏感。
   编码绝对坐标会导致G1（骨盆78cm）与人体（髋101cm）数值域完全不同、无法跨具身泛化；相对量则可比。metadata里现成的人体测量学字段可直接用于$H$的计算与校验。

---

## 5. SE3约束控制怎么做（训练与推理）

**训练时**：对每个样本随机采样一个约束掩码 $m(j)\in\{0,1\}$，被约束的关节把GT值作为条件注入：
```
# 编码侧
joint_feat_j = concat([p_j, v_j, r_j, is_controlled_j, constraint_value_j * is_controlled_j])
# 解码侧
Q_j += constraint_embed(constraint_value_j) * is_controlled_j
T_j[is_controlled] = constraint_value          # RePaint式硬替换
```
掩码采样比例覆盖三种模式（沿用架构文档定义）：全约束 / 稀疏末端约束 / 无约束纯脑补。单帧场景下建议 `全约束:末端稀疏:无约束 = 2:6:2`——中间那档（只给手腕/脚/骨盆，脑补其余）是"翻译官"最核心的使用场景。

**关键的一条额外loss（易被忽略）**：对`is_controlled=1`的关节，除了硬钉死，还要在**钉死之前**的网络原始输出上算一个约束一致性loss：
$$\mathcal L_\text{ctrl}=\sum_{j:m_j=1}\big\|\tilde{T}_j^{\text{pre-clamp}} - T_j^{\text{constraint}}\big\|$$
否则网络会学到"反正会被钉死，这个关节随便输出"，导致被约束关节和周围关节在clamp边界处出现明显不连续。

**推理时**：用户指定任意关节子集及其目标SE3，其余留空由模型脑补。评估时**约束满足误差单独统计**（应当≈0，因为是硬替换；若不为0说明实现有bug），真正要看的是**被约束关节周围的邻居关节是否合理联动**。

---

## 6. 七个候选的完整架构

**完全共享的部分**（七个变体逐行相同的代码）：

```python
# ---------- Encoder ----------
# 输入: 严格单帧(见§1.5, 已去掉速度通道), 源骨架 J_src 个关节
h = MLP([p, r * rot_avail, rot_avail, is_ctrl, ctrl_val])   # (J_src, D)  见§1.7
h = h + SigLIP_frozen(canonical_name) * dropout_mask
h = h + anchor_embed(fourier(bone_dir), bone_len/H)
for _ in range(L_enc):                            # 几何双轨 graph transformer
    h = SelfAttn(h, bias=hop_bias + chain_bias + edge_type_bias)
# --- part pooling: 仅变体2-7使用；变体1(Direct-Token)跳过这一步 ---
Q_part = Parameter(K, D)                          # K 待消融, 起点 K=8
A = softmax(Q_part @ h.T / sqrt(D))               # (K, J_src)
u = Q_part + A @ h                                # (K, D)  ← 瓶颈输入
# 变体1(Direct-Token): u = h, 形状 (J_src, D), 无pooling无压缩

# ---------- [ 瓶颈模块：六选一，见下表 ] ----------
z = BOTTLENECK(u)                                 # (K, D)

# ---------- Decoder ----------
# 目标骨架 J_tgt 个关节，J_tgt 可完全不同于 J_src
T = init_frames(target_rest_pose)                 # (J_tgt, SE3)
Q = anchor_embed(target) + SigLIP_frozen(target_names) + tree_pos_enc
Q = Q + constraint_embed(ctrl_val) * is_ctrl
for _ in range(N_recycle):
    for _ in range(L_dec):
        loc = InvariantPointAttention(Q, T, context=z)
        dR, dt = MLP(loc)
        T = compose(T, (dR, dt))
T[is_ctrl] = ctrl_val                             # 硬替换
p_out = T.translation                             # canonical位置, 通用
R_out = T.rotation                                # canonical frame朝向(承载twist), 通用
q_out = analytic_angles(T, target_urdf)           # 解析投影, 零学习参数, 通用
```

**七个瓶颈变体的规格**：

| 变体 | 瓶颈实现 | 额外loss项 | 关键超参 |
|---|---|---|---|
| **Direct-Token** | **无pooling无量化**：`z = h`，形状`(J_src, D)`，每个关节一个token | 无 | — （不存在$K$） |
| **AE** | part pooling到`(K,D)`后 `z = u`（或过一层线性投影），确定性、无约束 | 无 | $K$ |
| **VAE** | `μ,logσ = Head(u); z = μ + σ·ε` | $+\beta_\text{KL}\,D_\text{KL}(q(z|x)\|\mathcal N(0,I))$ | $\beta_\text{KL}$（建议warmup，从0线性升到目标值，防posterior collapse） |
| **CVAE** | `μ,logσ = Head([u, e_tgt])`，$e_\text{tgt}$为目标骨架embedding | 同VAE | $\beta_\text{KL}$；注意decoder已经通过topology-query吃到了目标骨架信息，CVAE的"C"主要作用在**编码侧后验**上 |
| **VQVAE** | 单层码本 $V\times D$，逐token最近邻查找，straight-through梯度 | $+\|\text{sg}[u]-z_q\|^2 + \gamma\|u-\text{sg}[z_q]\|^2$ | 码本大小$V$（建议1024）、$\gamma$（建议0.25）、EMA更新 |
| **RVQ** | $R$层残差量化，第$i$层量化前$i-1$层的残差 | 每层commitment loss之和 | $R$（建议4）、每层$V$ |
| **FSQ** | 逐维bounded round：`z = round(clip(Linear(u)))`，无码本 | **无额外loss**（FSQ不需要commitment） | levels列表（如`[8,8,8,8,5,5,5,5]`，隐式码本640,000） |

**参数量控制**：七个变体的backbone参数完全一致，仅瓶颈模块参数不同（VQ系多一个码本、VAE系多一个方差头、Direct-Token少一组part query）。报告时必须同时列参数量与**瓶颈实际信息容量**（Direct-Token为$J_\text{src}\times D$，其余为$K\times D$，量化系还要折算成比特），避免"某变体更好只是因为瓶颈更宽"——**Direct-Token的瓶颈比其余六个宽约10倍（78 vs 8），这个对比本身就不公平，必须在报告里显式标注，不能简单地看谁误差低就宣布谁赢。**

---

## 7. 训练配置与评估

### 7.1 训练

- **以step为准，不以epoch为准**（因为不同数据量下epoch含义不同）。T1阶段：batch 256，目标 **60k step**（200k帧 ≈ 780 step/epoch → 约77 epoch）。
- 七变体 × **3个seed**（否则无法判断差异是真实的还是随机波动）= 21次训练。T1单次约0.5-1小时 → 全部约11-21小时，单卡一晚上到一天能跑完。
- 注意GPU上已有24.5GB被占用，实际可用约73GB；batch 256的小模型远够，但**开跑前需要和正在跑的任务协调**。
- 早停/选点：按`val_topo`的末端位置误差选best checkpoint，不用`val_id`（`val_id`会偏向过拟合的变体）。

### 7.2 评估指标（分两个维度，对应你说的"翻译准确率/相似度"和"SE3精确控制"）

**维度A：SE3精确控制**（在有GT的SOMA→G1上）
1. 末端关节位置误差（mm）——只统计手腕/脚踝/骨盆/头部（若目标有），不是全关节平均。
2. 旋转测地距离误差（度）：$d(R_1,R_2)=\arccos\frac{\mathrm{tr}(R_1^\top R_2)-1}{2}$。
3. **约束满足误差**——应≈0，非0即实现有bug（属于正确性检查，不是性能指标）。
4. **约束邻域连续性**：被约束关节的相邻关节，其误差是否显著高于全局平均（检验clamp是否造成局部撕裂）。
5. **关节限位违反率**：FK后检查是否超出URDF定义的关节限位——这是能否真正下发到机器人的硬门槛。

**维度B：跨拓扑翻译质量**（重点，在`val_topo`上；注意§2.0——拓扑多样性只有约5种，结论强度受限）
6. **零样本拓扑重建误差**：在训练未见过的合成拓扑上的位置/旋转误差。
7. **Swap-consistency**：$z \to$ decode到拓扑B $\to$ 重新encode $\to$ 与原$z$的余弦距离。
8. **骨长泛化曲线**：按`val_actor`的身高分箱（145-160/160-175/175-199cm），看误差随体型偏离训练分布的增长速度。

**维度C：瓶颈本身的性质**（直接检验量化担忧）
9. **帧间瓶颈跳变**：连续两帧近乎相同的输入（同一pose加极小噪声），测 $\|z_t - z_{t-1}\|$ 和输出差异——这是VQ/FSQ系的量化边界跳变风险的直接测量，AE/VAE/Direct-Token系应显著更平滑。
10. **码本利用率**（仅VQ/RVQ）：实际被用到的码字比例，检测codebook collapse。
11b. **Twist DOF专项误差**（G1侧，见§1.7）：单独报告 `left/right_ankle_roll`、`left/right_wrist_roll` 这4个从位置完全观测不到的DOF的角度误差。它们在29维平均里只占4/29，**即使完全学错也会被平均值掩盖**，但ankle_roll直接决定足部接触姿态，对下发到真机是硬指标。

11. **Part query塌缩检测**（仅有pooling的六个变体）：SAMoR原文明确指出naive训练下$K$个query会"attend broadly and produce redundant representations, reducing K=8 to the effective equivalent of K=1"。直接测量：$K\times J$的attention矩阵$A$，计算各query注意力分布之间的两两余弦相似度，以及每个query的注意力熵。**若相似度普遍很高，说明发生了塌缩，此时该变体与Direct-Token的对比失去意义**，需要加上attention supervision loss重跑。

### 7.3 【必须有】非学习基线

没有基线的话，七个变体互相比较得再漂亮，也无法回答"这套学习式翻译官到底有没有超过现成方法"。至少要有：
- **解析retargeting基线**：现成的IK retargeter（GMR或SOMA Retargeter官方pipeline）在同样测试集上的误差；
- **最近邻基线**：在训练集里找最相似的源pose，直接返回它对应的目标pose。

如果七个变体都打不过最近邻，说明模型没学到泛化、只是在记忆。

### 7.4 一个诚实的前提说明

SOMA→G1的"GT"本身是NVIDIA用**IK retargeter生成的**，不是物理真值。因此本轮实质上是在**蒸馏一个retargeter**。这意味着：(a) 在SOMA→G1上"超过GT"没有意义；(b) 真正的价值在于`val_topo`——学习式方法能否泛化到那个retargeter根本没配置过的新拓扑。**评估权重应向维度B倾斜。**

---

## 8. Viser可视化：翻译官的"chatbox"

参照大模型需要chatbox来验证，翻译官需要一个能直观判断"翻译得对不对"的交互界面。仓库里已有`experiment/stage1/viser_compare.py`（三骨架对比浏览器）可作为起点，viser 1.0.16已装好。

**建议的界面布局（三栏 + 控制面板）**：

| 区域 | 内容 |
|---|---|
| 左栏 | **源骨架**（SOMA-78），显示输入pose；被约束的关节高亮为不同颜色 |
| 中栏 | **目标骨架实时翻译结果**，下拉框切换目标拓扑（G1-29 / SOMA-38 / SOMA-24 / ...）；下拉框切换瓶颈变体（Direct-Token/AE/VAE/.../FSQ），**同一输入立即看到七个变体的差异** |
| 右栏 | **GT对照**（若该拓扑有GT）或**误差热力图**（关节按误差着色，红=大误差） |
| 控制面板 | ① 帧slider（在测试集上前后拖动）<br>② 关节约束编辑器：点选任意关节 → 拖动gizmo改变目标位置 → 实时看全身重新脑补（这是最能体现"翻译官+SE3控制"卖点的交互）<br>③ 骨长滑块：直接改target骨架的肢段比例，看模型是否真正泛化<br>④ 数值面板：实时显示当前帧的末端误差/旋转误差/关节限位违反 |

**关键交互（这是验收的核心，不只是看图好看）**：
- **拖动一个关节 → 全身实时联动**：这直接检验"稀疏约束+脑补"是否work，比任何数字都直观。
- **同输入切换七个瓶颈变体**：定性差异（比如VQ系是否出现抖动/卡顿感）往往肉眼可辨，而这在平均误差数字上会被平滑掉。
- **骨长滑块推到训练分布之外**：看模型是崩溃还是优雅退化。

建议实现成一个独立脚本 `viser_translator_demo.py`，加载多个变体的checkpoint到同一进程，共享同一份测试数据。

---

## 9. 建议的执行顺序

1. **[阻塞]** G1 FK管线（URDF→3D frame，选可微实现）+ 坐标系统一（按§1.2的变换式，含det=+1和左右手四项验收检查）+ 关节名规范化对照表 + **G1 CSV root列单位/轴序的实测确认**。
1b. **[阻塞]** 下载Truebones Zoo（65物种），转成统一的per-joint位置格式，按**物种**切分train/val_topo，编写关节名→功能组标签的映射（匹配不上的一律给$y_j=-1$，按SAMoR的兜底机制不影响流程）。
2. **[阻塞]** 合成拓扑剪枝脚本（SOMA-78 → 38/24/20），产出多拓扑数据。
3. 分层采样 + 三重切分脚本，产出T0(5k)/T1(200k)缓存。
4. 共享backbone + 七个可插拔瓶颈类，T0冒烟跑通。
5. T1正式跑：7变体 × 3seed = 21次。
6. 评估脚本产出对比表（**13项指标**，以 `01_round1_spec.md` §7为准）+ 非学习基线对照。
7. viser demo，人工定性验收。

**其中1和2是前置阻塞项，没做完后面的实验结论都不成立。**

---

## 10. 你没提到但需要现在决定的事项汇总

1. **G1输出到底是关节角还是3D frame**（§1.1）——影响解码头设计。
2. **拓扑不对称的loss策略**三选一（§1.3）——影响"重建误差"的定义。
3. **只有一个具身对的问题**（§2）——不解决则本轮测不出跨具身能力，建议先做合成拓扑剪枝。
4. **非学习基线**（§7.3）——没有它无法判断学习式方法是否值得。
5. **GT本身是retargeter产物**（§7.4）——决定评估权重该往哪边倾斜。
6. **磁盘已澄清**：`/home`剩42G，但`/`分区剩428G——需要你用sudo建一个正式数据目录（`/data/alphamotion`），否则只能用`/var/tmp`。GPU当前已被占用24.5G（97.9G总量），开跑前需协调。
7. **拓扑多样性已决定引入Truebones Zoo**（§2.0）——拓扑数从5提到约70。但"机械狗动作生成顺带解决"不成立：缺四足侧配对GT、动物↔机械狗gap更大、动物动捕物理质量差。定位为Round-2，本轮不写进验收目标。Objaverse-XL本轮跳过。
9. **旋转必须编码，但不能照搬SAMoR**（§1.7）——SAMoR跨拓扑下主动丢弃旋转、靠IK后处理，并自承twist ambiguity是open problem。实测G1有4个DOF（双踝roll、双腕roll）从位置完全观测不到（0.000 mm），但在canonical frame里完全可见（11.459°）。**已定方案：一个通用头输出canonical SE(3) frame，关节角靠解析投影（零学习参数，实测4.29e-12 rad）——零per-embodiment模块。**
8. **ARDY/Kimodo的定位**（§2.0）——它们编辑的是joint angle不是拓扑，能提供姿态多样性但提供不了具身种类，不能替代剪枝方案。若要用，定位是补充稀有姿态（蹲/爬/坐）采样。
