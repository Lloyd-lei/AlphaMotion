# Greenwich-v2 pilot 解码对比视频 (0831)

给数字判官 (v2_judge) 配的肉眼卷子：同一批固定窗口，把三条消融臂的解码直接播出来，
你自己看动作做没做出来。

## 渲染了什么

- 方向：human_smpl -> unitree_g1_29dof（判官用的留出具身，真零样本方向）。
- 解码路径：完全复用 v2_judge 的加载/编解码（同 v2_windows.json 固定窗口、同 reach 口径、
  同 geom/encode/decode 调用），没有另起炉灶。
- 旋转 -> 关节角：仓库现成的铰链投影 manifold_project.fit_angles(method="global")
  （整链 LM，仓库文档写明模型预测该用这个），槽位 -> MJCF 关节名来自缓存的
  unitree_g1_29dof_dof.npz (slot_joints)，回放模型是 GMR 的 g1_mocap_29dof.xml，纯运动学
  qpos 播放（无物理）。
- 对照臂自检：GT 走同一条投影管线，投影后测地残差 0.08 度（≈零），说明"旋转->qpos->画面"
  这条尺子本身是干净的；解码臂画面上的偏差就是解码本身的偏差。
  （解码臂投影前离流形程度：base 均值 ~11 度 / fk、fkvn ~13 度，最大 137 度——投影会把
  超限部分按真机限位夹掉，画面如实反映"机器人做得出来的那部分"。）

## 每个视频 4 个并排面板

GT（原生 G1 retarget，配对缓存）| base | fk（+FK损失）| fkvn（+FK+var_norm）

每段 9 秒：先原速 3 秒，再 0.5x 慢放一遍（左下角出黄色 "0.5x" 标记）。

## 根轨迹的口径（重要）

四个面板的世界根运动（位置+朝向）全部用 GT 的：这批 pilot 不解码世界根。
所以面板之间的一切差异都是纯关节解码质量，位移/转身一样是喂进去的。

## 三个窗口（按内容挑的，来自 v2_windows.json 的固定卷子）

| 文件 | 片段 | 内容 |
|---|---|---|
| walk | walk3_subject1 @460017 | 平稳步行（根轨迹 1.6 m） |
| ground | aiming1_subject4 @11023 | 全程贴地爬行/趴伏（骨盆 37-42 cm）——val 窗口里没有专门的摔倒片段，这是最接近地面动作的窗 |
| dyn | sprint1_subject4 @402964 | 真冲刺（根轨迹 9.7 m，含大步幅+压低转向） |

## 文件路径

视频（1920x480，30fps，面板标签 GT/base/fk/fkvn）：
- /home/arenalabs/tmp/pilot_decode_walk.mp4
- /home/arenalabs/tmp/pilot_decode_ground.mp4
- /home/arenalabs/tmp/pilot_decode_dyn.mp4

8 帧网格图（每行 = 两个时刻的 4 面板组，从上到下时间推进）：
- /home/arenalabs/tmp/pilot_decode_walk_grid.png
- /home/arenalabs/tmp/pilot_decode_ground_grid.png
- /home/arenalabs/tmp/pilot_decode_dyn_grid.png

生成脚本（可换窗口/换 ckpt 重跑）：
- pose_translator_design/exp_topo/pilot_decode_viz.py

---

# 0831 补考：官方 G1 模型重渲 + GR3 渲染（关节限位问题的验证）

## 背景

lloyd 看出"腿太直"。查到 GMR 的 mocap 模型 (g1_mocap_29dof.xml) 把髋 pitch 卡在 ±90°，
官方谱系模型 (ufo g1_mjlab/g1_29dof.xml) 允许 −145°~165°。这次做两件事验证：
用官方模型重投影+重放同样三个窗口；再渲一版 human→GR3。

## 官方 vs GMR 限位表（度，两个 XML 关节名完全一致，只有 range 不同）

| 关节 | GMR mocap | 官方 (mjlab) |
|---|---|---|
| hip_pitch | −90 ~ 90 | −145 ~ 165 |
| hip_roll (左) | −30 ~ 90 | −30 ~ 170 |
| hip_yaw | ±90 | ±158 |
| knee | −5 ~ 165 | −5 ~ 165（相同）|
| ankle pitch/roll | 相同 | 相同 |
| waist_yaw | ±90 | ±150 |
| waist_roll / waist_pitch | ±30 / ±30 | 相同（都是 ±30）|
| shoulder_pitch | −177 ~ 66 | −177 ~ 153 |
| shoulder_roll (左) | −34 ~ 129 | −91 ~ 129 |
| shoulder_yaw | −80 ~ 115 | ±150 |
| elbow | −60 ~ 97 | −60 ~ 120 |

## 官方模型重渲的结果——诚实说：换播放模型救不了这三个窗口

做法：同样三个窗口、同样的解码管线，投影限位从缓存 dof（逐位等于 GMR XML，已核对）
换成官方 XML 的 range，播放也用官方模型。GT 也用同一把尺重投影（0.08° 误差，忠实）。

数值对照（同一批解码结果，分别用两套限位投影，直接对角度差）：

- **walk / ground：两套限位投出的角度逐帧逐关节完全一样（差 0.0°）**。
- **dyn：解码臂被 GMR 限位卡掉的部分放出来了**——base/fkvn 有 9%、fk 有 4% 的帧
  髋 pitch 超过 90°，最大差 17.7°（left_hip_pitch）。GT 不变。
- ground 窗口真正顶死的关节是：waist_pitch（GT 94~100% 的帧顶在 ±30° 上限，
  **这个限位两个模型一样**，不是 GMR 砍的）和踝（也相同）。髋 pitch 只有 7% 的帧
  刚好停在 −90° 边上，放宽后角度也没变。

为什么 ground 没变化：**GT 的旋转本身就是 GMR retarget 在 ±90° 模型上算出来的**，
数据里根本没有超过 90° 的髋弯；解码器又是在这份数据上训练的，所以它输出的爬行
也基本活在 ±90° 盒子里。换播放/投影模型追不回 retarget 阶段就丢掉的弯曲。
这和"手臂尾巴是数据墙"是同一个结论：杠杆在 GMR retarget（用官方限位重跑 retarget
再重建缓存），不在渲染端。

我自己看过网格图（下面的路径，请 lloyd 也自己看）：ground 窗口新旧姿势肉眼等同
（场景光照不同别被骗），解码臂的腿仍比 GT 直——这部分是解码误差+数据墙，不是
播放限位；dyn 窗口冲刺时解码臂的抬腿比旧版更开。

先前我第一眼觉得"新版弯多了"，量了角度才发现 ground 逐位相同——按纪律以数为准。

## GR3 渲染（human→fourier_gr3，判官方向，无 GT 面板）

GR3（GMR 自带 gr3v2_1_1.xml）的腿部限位是健康的：

| 关节 | GR3 (GMR mjcf) | G1 官方 | 判断 |
|---|---|---|---|
| hip_pitch | ±150 | −145 ~ 165 | 正常，没被砍半 |
| hip_roll (左) | −10 ~ 90 | −30 ~ 170 | 偏窄但够用 |
| hip_yaw (左) | −40 ~ 90 | ±158 | 偏窄 |
| knee | 0 ~ 135 | −5 ~ 165 | 正常 |
| ankle pitch | ±45 | −50 ~ 30 | 正常 |

结论：GR3 的 GMR 模型没有 G1 mocap 那种 ±90° 髋砍半问题。渲染里三个臂的爬行
（ground 窗）髋膝都能收到身体下面，冲刺（dyn 窗）抬腿折膝正常。投影残差也
低得多（mean 4.8~5.7° vs G1 的 10.9~13.4°）——训练内具身 + 健康限位。

## 口径说明

- 官方版四个面板共用一把尺：GT 的原生 retarget 旋转和解码臂一样，
  全部经官方模型重投影（fit_angles global LM + clamp 官方 range）再播放。
- 世界根仍然全部喂 GT 的（这批 pilot 不解码根）。
- GR3 根用缓存里 GR3 自己的 GT 世界根。

## 文件路径

官方 G1 重渲（1920x480，GT(reproj)|base|fk|fkvn）：
- /home/arenalabs/tmp/pilot_decode_walk_OFFICIAL.mp4
- /home/arenalabs/tmp/pilot_decode_ground_OFFICIAL.mp4
- /home/arenalabs/tmp/pilot_decode_dyn_OFFICIAL.mp4
- 网格图：/home/arenalabs/tmp/pilot_decode_{walk,ground,dyn}_OFFICIAL_grid.png

GR3（1440x480，base|fk|fkvn）：
- /home/arenalabs/tmp/pilot_decode_walk_GR3.mp4
- /home/arenalabs/tmp/pilot_decode_ground_GR3.mp4
- /home/arenalabs/tmp/pilot_decode_dyn_GR3.mp4
- 网格图：/home/arenalabs/tmp/pilot_decode_{walk,ground,dyn}_GR3_grid.png

生成脚本：pose_translator_design/exp_topo/pilot_decode_viz2.py（official|gr3 两个模式）

## 下一步建议

1. 真正的修复在 GMR：把 g1_mocap_29dof.xml 的限位换成官方 range 重跑 retarget，
   重建 g1 缓存（爬行/深蹲类数据受益最大）。
2. waist_pitch ±30° 是两个模型共同的硬限位，爬行类动作 GT 都顶死——这是具身
   本身的墙，翻译时腰弯不足要靠髋/踝补，评估时别把它算成解码误差。

## Adam 渲染 + 棋盘格地面（0831 追加）

地面换了：所有新渲染用高对比蓝白棋盘格（0.5m 方格）+ 渐变天空 + 平行光，
灰蒙蒙看不出接触的问题解决了。GR3 三个视频也用新地面重渲了一遍（_GR3v2），
和 Adam 同一块地板好直接对比。

### Adam (pnd_adam_lite) 限位表（缓存 dof 逐位 == GMR XML，已核对；度）

| 关节 | Adam | GR3 (GMR) | G1 官方 | G1 mocap |
|---|---|---|---|---|
| hip_pitch | ±119.7 | ±150 | −145 ~ 165 | ±90 |
| hip_roll (左) | −44.7 ~ 90 | −10 ~ 90 | −30 ~ 170 | 同官方 |
| hip_yaw | ±44.7 | −40 ~ 90 (左) | ±158 | ±90 |
| knee | −5.2 ~ 137.5 | 0 ~ 135 | −5 ~ 165 | 同官方 |
| ankle_pitch | −57.3 ~ **20.1** | ±45 | −50 ~ 30 | 同官方 |
| waist_pitch | **±44.7** | ±30 | ±30 | 同官方 |
| waist_roll / yaw | ±29.8 / ±44.7 | ±30 / ±150 | ±30 / ±150 | roll同 / yaw±90 |

三台身高（缓存内部单位）：Adam 452 < G1 471.6 < GR3 494.3 —— Adam 最矮。

### 数字（三窗、GT 也过同一投影；GT 投影残差 0.08°，解码臂 4.5~4.9°，
和 GR3 的 4.8~5.7° 同级，都是训练内具身的健康水平）

ground（爬行）窗的关键角度，[min,max] 度 + 顶限位帧占比：

| | Adam GT | Adam 解码(base) | GR3 GT | GR3 解码(base) |
|---|---|---|---|---|
| hip_pitch 左 | [−82,−46] 0%顶 | [−89,−58] 0%顶 | [−102,−66] 0%顶 | [−106,−62] 0%顶 |
| knee 左 | [79,125] 0%顶 | [84,122] 0%顶 | [94,135] 4%顶 | [92,131] 0%顶 |
| waist_pitch | [40,45] **86%顶** +44.7 | [32,42] 0% | [19,30] **64%顶** +30 | [16,28] 0% |
| ankle_pitch 左 | [12,20] **82%顶** +20.1 | [1,8] 0% | [−2,28] 0%顶(限45) | [1,7] 0% |

walk / dyn：Adam 髋膝踝都离限位很远（dyn 髋最深 −107，离 −119.7 还有余量），
0% 顶限位；和 GR3 一个模式。

### 我肉眼看网格图（lloyd 请自己再看一遍）

- walk：Adam 走得成立，步态和 GR3 同级，躯干略前倾（Adam GT waist_pitch
  本来就压着 +45 上限）。
- ground：Adam 是"高臀熊爬"，胸口离地比 GR3 高、腿更直；GR3 是贴地爬。
- dyn：两台都能深蹲折腿冲刺，无明显差别。

第一眼我以为 Adam 爬得差是"髋不够弯被砍了"——量完发现**不是**：解码髋 −95、
膝 126，离 Adam 的 ±119.7/137.5 限位都还远，0% 顶限。真正顶死的是
**ankle_pitch 上限 +20.1°（GT 82% 帧顶死；GR3 是 ±45 根本不顶）**和
waist_pitch。而且 Adam 的 GT 本身爬行髋弯就比 GR3 的 GT 浅 20°
（−82 vs −102）——浅爬在 retarget 阶段就定型了，解码只是如实复现（老结论：
数据墙在 GMR，不在解码/渲染）。

注意口径：缓存里没有 Adam 的 root_world，Adam 的根位置沿用 G1 GT 根透传
（ctrl_verify/adam_tracks_lafan.py 的既有做法），所以 Adam 画面里的"身体高度"
有一部分是 G1 的骨盆轨迹给的，别拿贴地程度直接当解码质量；关节角才是这里的尺。

### 判决：身高 vs 限位

**爬行质量跟限位走，不跟身高走。** 排序：GR3（最高 494，hip ±150）爬最好；
Adam（最矮 452，hip ±119.7、踝上限只有 +20.1）中等；G1 mocap（中等身高
471.6，hip ±90）最差。如果身高是主因，最矮的 Adam 应该在两端之一——实际它
卡在中间，位置正好等于它限位的宽窄。而且 Adam 的短板具体到关节：不是髋
（没顶），是踝背屈 +20.1° 和 GT retarget 本身的浅爬。

### 文件路径

Adam（1440x480，base|fk|fkvn，棋盘地面）：
- /home/arenalabs/tmp/pilot_decode_{walk,ground,dyn}_ADAM.mp4
- 网格图：/home/arenalabs/tmp/pilot_decode_{walk,ground,dyn}_ADAM_grid.png

GR3 新地面重渲（同上布局）：
- /home/arenalabs/tmp/pilot_decode_{walk,ground,dyn}_GR3v2.mp4
- 网格图：/home/arenalabs/tmp/pilot_decode_{walk,ground,dyn}_GR3v2_grid.png

脚本：pilot_decode_viz2.py 现在支持 official|gr3|adam 三个模式；顺手修了
xml_ranges 的一个隐性 bug（`range=` 正则会误吃 `actuatorfrcrange`，无 range
的关节会读出 ±366.7° 假限位——只影响打印，不影响投影）。
