# AlphaMotion-2: Unified Skeleton-Surface Motion Mapping Foundation Model

- Document ID: 1gH9WaPSgqUcTdrPYeBCNaLkWrVoG0qzNsiJLPVsniBU
- Revision ID: ANLCKQlsW6S34uVnKnbT29MGfV_MqQIyH5O6C3gOqXUVUwxq9EVjqkipbOQRgoFWm_73PZ-AlobWGDC6GWwGp10XtXcTPMnoOfHkG1RYARk
- Selected tab: t.2w3c93da5d23
- Protected controls: 0
- Opaque controls: 0
- Authoritative dropdowns: 0

Protected-control annotations are preservation instructions. Do not insert their displayed placeholder text to recreate a native control.

## 单帧扩容实验 · Megatron (t.2w3c93da5d23)

[P00001 | 1:26 | HEADING_1]
单帧预实验：Megatron + UniMate

[P00002 | 26:180 | NORMAL_TEXT]
状态：预实验完成 3,000 步（13.3 分钟）。纯 SE(3) 补全相对“无源姿态、无条件先验”的选定末端误差平均下降 40.2%／57.8%。G1 双末端绝对误差仍为 144.94 mm、24.25°；有源动作映射为 63.02 mm、13.50°。此 warmstart 预实验不计入六组从零对照。

[P00003 | 180:188 | HEADING_2]
这轮要回答什么

[P00004 | 188:331 | NORMAL_TEXT]
候选沿用一期单帧双 FSQ：隐藏维度 256、8 头，编码器与各流解码器 6 层。共享主干 25,743,428 参数，训练辅助 judge 266,512 参数，冻结 Qwen 另计。batch 128，最多 3,000 步或训练计时 30 分钟；不足 1,000 步仅作运行检查。

[P00005 | 331:337 | HEADING_2]
保留与新增

[P00006 | 337:456 | NORMAL_TEXT]
保留拓扑 hop 偏置、双 FSQ、共享目标解码和适用的旧单帧损失。SigLIP 换为 Qwen，新增带有效性 mask 的稀疏 SE(3) 条件。原对比、重编码、judge 及接触／穿透项已恢复；没有实际时间差监督的滑步和时序项关闭。

[P00007 | 456:467 | HEADING_2]
单帧如何使用轨迹条件

[P00008 | 467:570 | NORMAL_TEXT]
第一版使用轨迹当前帧的末端位姿，保持单帧主干。单／双末端条件由真实 FK 采样；平移以根节点为原点，旋转保留统一坐标系下的朝向。保留无条件样本与条件 dropout。世界／物体锚定轨迹和时序平滑另行验证。

[P00009 | 570:578 | HEADING_2]
数据与文本准备

[P00010 | 578:716 | NORMAL_TEXT]
采样比例为 XWBC 55%、UniMate 45%：前者真实跨本体配对，后者同体重建或稀疏补全。XWBC 30,322 个共享源帧 × 16 体已完成提取和地面来源验收；UniMate 全帧 FK 后纳入 217 文件、10 体、24,017 帧，另合并 1 个重复别名。

[P00011 | 716:838 | NORMAL_TEXT]
Qwen3-Embedding-0.6B 已为 584 个真实关节名缓存 1024 维向量；文本塔冻结，适配投影参与训练。SigLIP 投影不直接作为等价权重复用。旧模型对照仍使用原 SigLIP 缓存与权重，保持相同数据划分和 FK 指标。

[P00012 | 838:844 | HEADING_2]
运行与验收

[P00013 | 844:950 | NORMAL_TEXT]
最终 checkpoint、旧模型对照、初始化与终评、三类配对图均已保存。定量门槛通过；视觉复核显示四足全身姿态仍有明显偏差，稀疏约束不决定唯一全身解。报告脚本单独修复，未重跑训练；软件验收与模型效果分别记录。

[P00014 | 950:1053 | NORMAL_TEXT]
UniMate 使用 canonical 单位，不能报告厘米精度；当前表示无法覆盖的滑动、局部平移或动态骨长样本会隔离。本轮包含多项改动，不能单独归因，也不代表 surface、灵巧手、动力学或真机验证。

[P00015 | 1053:1062 | HEADING_2]
与正式对照的关系

[P00016 | 1062:1110 | NORMAL_TEXT]
先导试验已完成；下面的六组正式对照从随机初始化独立训练，单独比较数据曝光、计算成本与模型效果。

[P00017 | 1110:1127 | HEADING_1]
六组从零训练：骨架与表面扩容对照

[P00018 | 1127:1189 | NORMAL_TEXT]
六组各完成 6,000 步及固定 P0／P1 评估。关节精修有所改善，留出体映射未呈现稳定扩容收益；原生夹爪开度仍未恢复。

[P00019 | 1189:1273 | NORMAL_TEXT]
A 为骨架模型，B 增加表面 patch、语义与几何损失；共享滑动关节路径和 FK。同为 seed 0、batch 128，两阶段各自的实际采样与固定评估条件均匹配。

[P00020 | 1273:1274 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00021 | 1277:1280 | NORMAL_TEXT | TABLE row=0 col=0]
模型

[P00022 | 1281:1287 | NORMAL_TEXT | TABLE row=0 col=1]
主模型参数

[P00023 | 1288:1298 | NORMAL_TEXT | TABLE row=0 col=2]
6,000 步累计

[P00024 | 1300:1304 | NORMAL_TEXT | TABLE row=1 col=0]
A-S

[P00025 | 1305:1315 | NORMAL_TEXT | TABLE row=1 col=1]
6,567,365

[P00026 | 1316:1330 | NORMAL_TEXT | TABLE row=1 col=2]
已完成 · 25.1 分钟

[P00027 | 1332:1336 | NORMAL_TEXT | TABLE row=2 col=0]
B-S

[P00028 | 1337:1347 | NORMAL_TEXT | TABLE row=2 col=1]
6,817,237

[P00029 | 1348:1362 | NORMAL_TEXT | TABLE row=2 col=2]
已完成 · 34.5 分钟

[P00030 | 1364:1368 | NORMAL_TEXT | TABLE row=3 col=0]
A-M

[P00031 | 1369:1380 | NORMAL_TEXT | TABLE row=3 col=1]
25,813,317

[P00032 | 1381:1395 | NORMAL_TEXT | TABLE row=3 col=2]
已完成 · 26.3 分钟

[P00033 | 1397:1401 | NORMAL_TEXT | TABLE row=4 col=0]
B-M

[P00034 | 1402:1413 | NORMAL_TEXT | TABLE row=4 col=1]
26,542,293

[P00035 | 1414:1428 | NORMAL_TEXT | TABLE row=4 col=2]
已完成 · 36.8 分钟

[P00036 | 1430:1434 | NORMAL_TEXT | TABLE row=5 col=0]
A-L

[P00037 | 1435:1446 | NORMAL_TEXT | TABLE row=5 col=1]
75,489,061

[P00038 | 1447:1461 | NORMAL_TEXT | TABLE row=5 col=2]
已完成 · 36.4 分钟

[P00039 | 1463:1467 | NORMAL_TEXT | TABLE row=6 col=0]
B-L

[P00040 | 1468:1479 | NORMAL_TEXT | TABLE row=6 col=1]
76,926,517

[P00041 | 1480:1494 | NORMAL_TEXT | TABLE row=6 col=2]
已完成 · 51.2 分钟

[P00042 | 1495:1496 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00043 | 1496:1592 | NORMAL_TEXT]
辅助 judge（S/M/L）：134,412／265,484／396,556 参数；冻结 Qwen：595,776,512。时间仅计优化。B-L 含 CPU 探针并发，不用于严格架构测速。

[P00044 | 1592:1600 | HEADING_2]
数据覆盖与缺口

[P00045 | 1600:1707 | NORMAL_TEXT]
固定清单共 28 体，其中 20 体参与训练、8 体整体留出。XWBC 提供人到机器人的真实配对；UniMate、Adroit 手和完整 Franka 臂夹爪提供同体重建／补全。后两者尚不能证明跨末端执行器映射。

[P00046 | 1707:1714 | HEADING_2]
如何比较结果

[P00047 | 1714:1820 | NORMAL_TEXT]
XWBC 配对映射与 UniMate 留出体同体重建分别报告。P0、阈值归一 P1 均保留自由姿态、关节投影、真实关节精修及默认姿态基线，记录 0—100 步；成功要求全部可见目标同时满足 1 cm 与 5°。

[P00048 | 1820:1933 | NORMAL_TEXT]
连续片段保留真实采样时间，原始越界与严格合法参考分组、失败样本保留。源 episode 置信区间不代表训练 seed 置信区间；3 个 UniMate 留出体各仅 1 个源片段，无法估计片段区间。单 seed 仅作早期对照。

[P00049 | 1933:1940 | HEADING_2]
续训如何解释

[P00050 | 1940:2038 | NORMAL_TEXT]
六组恢复各自模型与 Adam，以共同 LR 重启、新采样流及 sigmoid 稳定反传完成续训。前向逐位不变，但 3,000→6,000 的变化不能纯归因于步数；非零梯度也不代表夹爪开度已恢复。

[P00051 | 2038:2113 | NORMAL_TEXT]
六组共享扩展数据混合，不能隔离 UniMate 相对 XWBC-only 的因果收益。独立表面接触标签仍为零；接触、抓取、动力学和真机能力尚未验证。

[P00052 | 2113:2128 | HEADING_1]
计算成本：更多计算带来了什么

[P00053 | 2128:2242 | NORMAL_TEXT]
同一未见动作片段上，已见 G1 六组位置误差均较 3,000 步下降，最终为 88.6—96.0 mm；留出 Kuavo 为 121.5—131.0 mm，仅 B-M 略降。训练时间与估算计算量增加，并未带来一致的留出体收益。

[P00054 | 2242:2244 | NORMAL_TEXT]
[INLINE_OBJECT kix.1jpi071yfwgc]

[P00055 | 2244:2317 | NORMAL_TEXT]
图中实线为 A、虚线为 B；颜色表示 S/M/L。两阶段分段绘制，续训包含共同 LR 重启与数值反传修正。点是实际保存节点，未插值成同预算优势。

[P00056 | 2317:2460 | NORMAL_TEXT]
累计估计 dense PFLOPs（A-S、B-S、A-M、B-M、A-L、B-L）：5.87、9.67、22.51、35.69、63.43、99.45。按线性／注意力实际形状计数，MAC=2，反向约为前向两倍；不含归一化、激活、FK、损失和优化器等，不能当作硬件 FLOPs 测量。

[P00057 | 2460:2478 | HEADING_1]
XWBC：四个留出机器人的配对映射

[P00058 | 2478:2553 | NORMAL_TEXT]
源为人类姿态，目标为整体验证留出的机器人；每体使用固定的 3 个未见源 episode。下图和表格均为根相对全身关节位置误差，单位 mm，越低越好。

[P00059 | 2553:2555 | NORMAL_TEXT]
[INLINE_OBJECT kix.l4li1924vbud]

[P00060 | 2555:2556 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00061 | 2559:2567 | NORMAL_TEXT | TABLE row=0 col=0]
本体 / mm

[P00062 | 2568:2572 | NORMAL_TEXT | TABLE row=0 col=1]
A-S

[P00063 | 2573:2577 | NORMAL_TEXT | TABLE row=0 col=2]
B-S

[P00064 | 2578:2582 | NORMAL_TEXT | TABLE row=0 col=3]
A-M

[P00065 | 2583:2587 | NORMAL_TEXT | TABLE row=0 col=4]
B-M

[P00066 | 2588:2592 | NORMAL_TEXT | TABLE row=0 col=5]
A-L

[P00067 | 2593:2597 | NORMAL_TEXT | TABLE row=0 col=6]
B-L

[P00068 | 2599:2603 | NORMAL_TEXT | TABLE row=1 col=0]
GR3

[P00069 | 2604:2610 | NORMAL_TEXT | TABLE row=1 col=1]
102.6

[P00070 | 2611:2617 | NORMAL_TEXT | TABLE row=1 col=2]
106.3

[P00071 | 2618:2624 | NORMAL_TEXT | TABLE row=1 col=3]
100.0

[P00072 | 2625:2631 | NORMAL_TEXT | TABLE row=1 col=4]
103.6

[P00073 | 2632:2638 | NORMAL_TEXT | TABLE row=1 col=5]
115.1

[P00074 | 2639:2645 | NORMAL_TEXT | TABLE row=1 col=6]
103.6

[P00075 | 2647:2653 | NORMAL_TEXT | TABLE row=2 col=0]
Kuavo

[P00076 | 2654:2660 | NORMAL_TEXT | TABLE row=2 col=1]
126.1

[P00077 | 2661:2667 | NORMAL_TEXT | TABLE row=2 col=2]
126.2

[P00078 | 2668:2674 | NORMAL_TEXT | TABLE row=2 col=3]
125.3

[P00079 | 2675:2681 | NORMAL_TEXT | TABLE row=2 col=4]
121.5

[P00080 | 2682:2688 | NORMAL_TEXT | TABLE row=2 col=5]
131.0

[P00081 | 2689:2695 | NORMAL_TEXT | TABLE row=2 col=6]
129.4

[P00082 | 2697:2700 | NORMAL_TEXT | TABLE row=3 col=0]
H2

[P00083 | 2701:2707 | NORMAL_TEXT | TABLE row=3 col=1]
137.4

[P00084 | 2708:2714 | NORMAL_TEXT | TABLE row=3 col=2]
141.3

[P00085 | 2715:2721 | NORMAL_TEXT | TABLE row=3 col=3]
144.4

[P00086 | 2722:2728 | NORMAL_TEXT | TABLE row=3 col=4]
159.5

[P00087 | 2729:2735 | NORMAL_TEXT | TABLE row=3 col=5]
148.4

[P00088 | 2736:2742 | NORMAL_TEXT | TABLE row=3 col=6]
157.4

[P00089 | 2744:2747 | NORMAL_TEXT | TABLE row=4 col=0]
R1

[P00090 | 2748:2754 | NORMAL_TEXT | TABLE row=4 col=1]
106.5

[P00091 | 2755:2761 | NORMAL_TEXT | TABLE row=4 col=2]
108.8

[P00092 | 2762:2768 | NORMAL_TEXT | TABLE row=4 col=3]
101.1

[P00093 | 2769:2775 | NORMAL_TEXT | TABLE row=4 col=4]
103.8

[P00094 | 2776:2782 | NORMAL_TEXT | TABLE row=4 col=5]
103.5

[P00095 | 2783:2789 | NORMAL_TEXT | TABLE row=4 col=6]
112.3

[P00096 | 2790:2791 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00097 | 2791:2886 | NORMAL_TEXT]
表为 6,000 步结果。相较 3,000 步，24 个模型×本体组合中 8 个均值下降、16 个上升；这只是误差方向计数，既非独立重复，也非统计显著性。未观察到跨任务一致的单调尺寸收益。

[P00098 | 2886:2905 | HEADING_1]
UniMate：四个留出体的同体重建

[P00099 | 2905:2987 | NORMAL_TEXT]
这一组测试模型能否读取并重建未参与训练的骨架，不能当作人到机器人映射成功率。纵轴为位置误差／本体 reach，采用 canonical 单位；不能换称毫米或厘米。

[P00100 | 2987:2989 | NORMAL_TEXT]
[INLINE_OBJECT kix.v7lympo139ow]

[P00101 | 2989:2990 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00102 | 2993:3004 | NORMAL_TEXT | TABLE row=0 col=0]
本体 / reach

[P00103 | 3005:3009 | NORMAL_TEXT | TABLE row=0 col=1]
A-S

[P00104 | 3010:3014 | NORMAL_TEXT | TABLE row=0 col=2]
B-S

[P00105 | 3015:3019 | NORMAL_TEXT | TABLE row=0 col=3]
A-M

[P00106 | 3020:3024 | NORMAL_TEXT | TABLE row=0 col=4]
B-M

[P00107 | 3025:3029 | NORMAL_TEXT | TABLE row=0 col=5]
A-L

[P00108 | 3030:3034 | NORMAL_TEXT | TABLE row=0 col=6]
B-L

[P00109 | 3036:3039 | NORMAL_TEXT | TABLE row=1 col=0]
蛇形

[P00110 | 3040:3046 | NORMAL_TEXT | TABLE row=1 col=1]
0.283

[P00111 | 3047:3053 | NORMAL_TEXT | TABLE row=1 col=2]
0.493

[P00112 | 3054:3060 | NORMAL_TEXT | TABLE row=1 col=3]
0.282

[P00113 | 3061:3067 | NORMAL_TEXT | TABLE row=1 col=4]
0.386

[P00114 | 3068:3074 | NORMAL_TEXT | TABLE row=1 col=5]
0.340

[P00115 | 3075:3081 | NORMAL_TEXT | TABLE row=1 col=6]
0.211

[P00116 | 3083:3087 | NORMAL_TEXT | TABLE row=2 col=0]
动画手

[P00117 | 3088:3094 | NORMAL_TEXT | TABLE row=2 col=1]
0.564

[P00118 | 3095:3101 | NORMAL_TEXT | TABLE row=2 col=2]
0.543

[P00119 | 3102:3108 | NORMAL_TEXT | TABLE row=2 col=3]
0.508

[P00120 | 3109:3115 | NORMAL_TEXT | TABLE row=2 col=4]
0.633

[P00121 | 3116:3122 | NORMAL_TEXT | TABLE row=2 col=5]
0.267

[P00122 | 3123:3129 | NORMAL_TEXT | TABLE row=2 col=6]
0.286

[P00123 | 3131:3134 | NORMAL_TEXT | TABLE row=3 col=0]
短链

[P00124 | 3135:3141 | NORMAL_TEXT | TABLE row=3 col=1]
0.186

[P00125 | 3142:3148 | NORMAL_TEXT | TABLE row=3 col=2]
0.183

[P00126 | 3149:3155 | NORMAL_TEXT | TABLE row=3 col=3]
0.295

[P00127 | 3156:3162 | NORMAL_TEXT | TABLE row=3 col=4]
0.233

[P00128 | 3163:3169 | NORMAL_TEXT | TABLE row=3 col=5]
0.202

[P00129 | 3170:3176 | NORMAL_TEXT | TABLE row=3 col=6]
0.260

[P00130 | 3178:3181 | NORMAL_TEXT | TABLE row=4 col=0]
四足

[P00131 | 3182:3188 | NORMAL_TEXT | TABLE row=4 col=1]
0.115

[P00132 | 3189:3195 | NORMAL_TEXT | TABLE row=4 col=2]
0.100

[P00133 | 3196:3202 | NORMAL_TEXT | TABLE row=4 col=3]
0.116

[P00134 | 3203:3209 | NORMAL_TEXT | TABLE row=4 col=4]
0.105

[P00135 | 3210:3216 | NORMAL_TEXT | TABLE row=4 col=5]
0.108

[P00136 | 3217:3223 | NORMAL_TEXT | TABLE row=4 col=6]
0.123

[P00137 | 3224:3225 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00138 | 3225:3312 | NORMAL_TEXT]
表为 6,000 步结果。蛇形、动画手、短链各只有 1 个源片段，无法估计源片段置信区间；四足有 3 个。不同本体的最佳规模与 A/B 选择不同，不能据此宣称通用扩容规律。

[P00139 | 3312:3337 | HEADING_1]
Manipulation：已见本体的轨迹条件补全

[P00140 | 3337:3453 | NORMAL_TEXT]
Adroit 手和完整 Franka 臂夹爪均参与过训练，这里评估 3 个未见源 episode 的稀疏末端条件补全。P1 从投影后的真实关节 q 开始优化，图示 0—100 步；两末端须同时满足位置 ≤1 cm 且旋转 ≤5°。

[P00141 | 3453:3455 | NORMAL_TEXT]
[INLINE_OBJECT kix.djombhnkzwgt]

[P00142 | 3455:3572 | NORMAL_TEXT]
曲线是各 episode 等权的达标帧比例，保留未达标帧；不是整段轨迹或抓握成功率。P1 精修 0 步两体六组均为 0；20 步 Adroit 最高 18.8%，Franka 仍全为 0。原生模型输出与 100 步精修结果必须分开。

[P00143 | 3572:3668 | NORMAL_TEXT]
P1 用阈值归一的平移／旋转目标及归一化 q；与原 P0 分别保存，未替换旧结果。默认 q 使用同协议优化预算。自由姿态可能违反真实关节约束，不能用其末端达标替代 native-q 的结果。

[P00144 | 3668:3683 | HEADING_1]
精修收益与仍未解决的夹爪开度

[P00145 | 3683:3784 | NORMAL_TEXT]
下表为 6,000 步模型、严格合法参考、真实 q 精修 100 步后的 episode 等权达标帧比例：每格 P0 → P1。各体 3 个未见源 episode；默认 q 的两协议达标比例均为 0。

[P00146 | 3784:3785 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00147 | 3788:3791 | NORMAL_TEXT | TABLE row=0 col=0]
模型

[P00148 | 3792:3807 | NORMAL_TEXT | TABLE row=0 col=1]
Adroit：P0 → P1

[P00149 | 3808:3823 | NORMAL_TEXT | TABLE row=0 col=2]
Franka：P0 → P1

[P00150 | 3825:3829 | NORMAL_TEXT | TABLE row=1 col=0]
A-S

[P00151 | 3830:3843 | NORMAL_TEXT | TABLE row=1 col=1]
2.1% → 46.9%

[P00152 | 3844:3852 | NORMAL_TEXT | TABLE row=1 col=2]
0% → 0%

[P00153 | 3854:3858 | NORMAL_TEXT | TABLE row=2 col=0]
B-S

[P00154 | 3859:3872 | NORMAL_TEXT | TABLE row=2 col=1]
1.0% → 51.6%

[P00155 | 3873:3881 | NORMAL_TEXT | TABLE row=2 col=2]
0% → 0%

[P00156 | 3883:3887 | NORMAL_TEXT | TABLE row=3 col=0]
A-M

[P00157 | 3888:3902 | NORMAL_TEXT | TABLE row=3 col=1]
17.7% → 76.6%

[P00158 | 3903:3911 | NORMAL_TEXT | TABLE row=3 col=2]
0% → 0%

[P00159 | 3913:3917 | NORMAL_TEXT | TABLE row=4 col=0]
B-M

[P00160 | 3918:3929 | NORMAL_TEXT | TABLE row=4 col=1]
0% → 75.5%

[P00161 | 3930:3940 | NORMAL_TEXT | TABLE row=4 col=2]
0% → 3.0%

[P00162 | 3942:3946 | NORMAL_TEXT | TABLE row=5 col=0]
A-L

[P00163 | 3947:3960 | NORMAL_TEXT | TABLE row=5 col=1]
9.9% → 64.6%

[P00164 | 3961:3974 | NORMAL_TEXT | TABLE row=5 col=2]
3.0% → 15.2%

[P00165 | 3976:3980 | NORMAL_TEXT | TABLE row=6 col=0]
B-L

[P00166 | 3981:3994 | NORMAL_TEXT | TABLE row=6 col=1]
2.1% → 40.6%

[P00167 | 3995:4008 | NORMAL_TEXT | TABLE row=6 col=2]
7.1% → 55.6%

[P00168 | 4009:4010 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00169 | 4010:4157 | NORMAL_TEXT]
P1 的 Adroit 六组全帧平均为 5.99—8.84 mm、1.71—2.33°。Franka B-L 为 10.85 mm、10.88°，达标帧比例 55.60%；A-L 为 10.73 mm、11.92°、15.15%。位置均值接近而达标比例不同，不能说 B-L 整体精度普遍领先。

[P00170 | 4157:4291 | NORMAL_TEXT]
B-L 的三个源 episode 所选合法片段分别达标 15/35（42.86%）、14/31（45.16%）、26/33（78.79%）。55.60% 是三比例等权均值；合并计数 55/99 为 55.56%，不是本表指标。评估只覆盖固定片段，不代表完整轨迹成功。

[P00171 | 4291:4440 | NORMAL_TEXT]
网络原生 Franka 开度仍失败：6,000 步 M/L 四组滑动输出精确饱和比例为 100%；S 两组未精确到边界但已近饱和。六组开度误差均约 12.142 mm。3,000／4,500／6,000 的诊断未证明原生开度学习恢复；P1 真实 q 精修后的 B-L 开度误差为 8.24 mm。

[P00172 | 4440:4519 | NORMAL_TEXT]
后续优先解决滑动关节开度和留出体泛化，再以新注册协议复核。当前没有 zero-shot 手↔夹爪、物体抓握、接触、动力学或真机成功证据；本轮不据此增加训练。

[P00173 | 4519:4685 | NORMAL_TEXT]
工程入口：research/alphamotion_scaling_20261004/README.md。逐 episode 计数、固定病例、0/20/100 步指标、完整曲线与配对视频见 evaluation_phase2_v2/reports；checkpoints 与可恢复状态见 continuation_v3/runs。

[P00174 | 4685:4686 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

