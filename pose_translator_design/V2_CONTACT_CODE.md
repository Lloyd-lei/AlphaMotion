# Greenwich-v2 接触层代码 (0831)

按 `GREENWICH_V2_SPEC.md` §7-§9 实现。**每个开关默认关, 全关时与打补丁前逐位相同**
(`p3_smoke_identity.sh` 比 model.pt 的 sha256, 已验证一致)。`contact_label.py` 未改动。

## 一条恒等式撑起全部三项损失

训练缓存里的姿态是**根相对**的, 根本没有世界根, 单靠 FK 拿不到"离地多高"。实测确认:

```
h_j - h0  ==  fk_y_j(根相对, 米) + gap_root        最大偏差 0.7mm (= fp16 存储精度)
```

`gap_root` 是标注文件里根关节那一列 (= 根的世界高 − 地板 h0), 一个标量常量、不回传梯度。
借它就补上了缺的那一维, 于是穿模/接触/滑移三项都能经**可微 FK** 把梯度送回码本 ——
spec §7 的全部机关。实现在 `dual_common.fk_height_m`, 只存 gap 的根那一列 (其余 J-1 列
在训练里冗余)。

GT 上的标定 (LAFAN, 0831 实测, 权重量级都从这来):

| | human_smpl | fourier_gr3 | unitree_g1_29dof |
|---|---|---|---|
| GT 穿地 mean / p95 | 0.01 / 0.00 cm | 0.07 / 0.00 cm | 0.03 / 0.00 cm |
| GT 接触点 \|h\| mean | 1.06 cm | 1.15 cm | 2.14 cm |

接触点 \|h\| 的 1-2cm 就是 `w_contact` 那项的**数据地板**(§8 的 ±3cm 接地标定噪声);
而穿地项没有地板 —— §8.1: `relu(-h)` 是不等式不是回归, 训出的模型可以比数据更干净。

## 逐处改动

### `model_dual.py` — 结构 (+641 参数, +0.018%, spec 预算 +2%)
- `contact_chan=False` 默认无新模块, 老 checkpoint 照旧 strict-load。
- `con_emb: Linear(1,d)` **零初始化** —— 开关刚打开时函数与不开时逐位相同, 但
  `dL/dW = grad_out * x` 非零, 能自己长出来 (tiny 冒烟里 8 步后权重已 9e-4)。
- **顺序是纪律**: `con_emb` 加在 `mask_tok` 替换**之后**。cask 的语义是"这个关节被遮住了,
  但它必须踩地"; 若先加后遮, mask_tok 会把接触位一起抹掉, cask 静默退化成普通遮挡。
  这一行和 `masking.py` 的 cask 必须同时成立, 单独看任何一处都发现不了。
- `_split_con` 让 `encode` 同时吃 9 通道 (老调用点一字不改) 和 10 通道 (v2 契约)。
- `decode(..., ret_con=True)` 才返回接触头 logit, 默认 False => eval_dual / bridge_* /
  v2_judge / amplitude_report 全部无需改动。接触头读 **qr(旋转支路)**: 接触是姿态的
  确定性函数, 旋转支路才是出画的那支, 且 `rot_only` 下 qp 根本不算。

### `masking.py` — 两个新掩码族
- `cgive`: 接触关节可见、其余全遮 ("这几个点在地上, 把身体补出来")。
- `cask`: 接触关节被遮、接触通道置 1 ("这个关节必须踩地, 你自己想办法") —— lloyd 定的
  主方案, 因为**坐标是具身相关的, "踩没踩"这个标签是具身无关的**, cask 给的约束能跨
  具身直接搬运, cgive 给的坐标不能。
- 两族**不消耗任何 RNG** (给定 contact 就是确定的), 且不列进 `--mask_mix` 就一行不走 =>
  与 rand/ee/root/sparse 自由组合, 既往 run 逐位不变。
- 通道值用 soft、族判定用 hard (标注器过完滞回+去毛刺的保守判定), 分两个入口传。

### `dual_common.py` — 数据 + 损失
- `ContactStore`: soft 存 fp16 (PoseStore 同套路), hard **位打包** uint8 (布尔标签本来
  就只有 1 bit, 用 fp16 存要 16 倍内存), gap_root fp32 驻卡。18 具身 x 5.2M 行的生产
  规模下这是必须算的账。
- `load_contact`: 拼接口径与 `load_paired_data` 逐字对齐 (先按 cache 顺序 concat 再
  `max_rows` 截断), 行数对不上直接断言失败 —— 行号错位会把某一帧的接触配到另一段动作上,
  这种错在数字上完全看不出来, 只能拒跑。**缺 npz = `con_ok=False` = 通道喂零 + 该具身
  所有接触项静默失效**。
- `contact_terms` 三项:
  - `pen` = `relu(-h)²` 对**所有**关节, 按全体关节均值归一 (它是全身普适律, 不是接触事件),
    **永远开着**。
  - `cmaint` = `|h|` 只在有标签处, **逐帧按当帧接触数归一** —— 躺姿一帧 6 个接触点、
    站姿 2 个, 不归一的话一帧躺姿抵三帧站姿, 损失会被接触多的那几帧牵着走。
  - `cbce` = 接触头对软标签的 BCE, **永远开着**: 推理时没有输入接触标签, 模型得自己预测,
    跟"稀疏关节推全身"是同一范式。
  - 置信掩码 (§8.3): 权重取 `soft * hard`, 模糊帧权重为 0 = 不给接触监督, 宁可少教不可教错。
- `contact_balance_weights`: 按**接触构型签名**(哪些肢体组在接触的位掩码)分桶, 频率倒数
  配权 + 上限截断, 再按 `--contact_target` 配两池总质量。按**肢体组**不按关节, 因为各家
  一只脚有 1~3 个关节, 按关节分桶等于给关节多的具身更多桶, 桶就不可比。
  实测: LAFAN 地面帧 8.89% -> 采样 10.0% (664 种签名, 单行最高 58x 均值)。

### `train_dual.py` — 接线
- 新开关: `--contact_chan --w_pen --w_contact --w_slip_c --w_cbce --contact_balance
  --contact_target --contact_wcap --contact_ref`, 全默认关/0。
- 接触损失走**独立的 `cex` 字典**, 不碰 `--var_floor` 那个 `ex` —— 免得 `+ 0.0 * x` 混进
  既有路径。
- **no-slip 复用 P1.5 的 slip 副批**: 同一个副批、同一次前向、同一次 FK, 但作用域从
  "脚 + GT 速度门"换成"标注器说这两帧都在接触的**任意**关节"(手掌撑地、膝盖跪地一样算)。
  两项合成**一次** backward (梯度是同一个和); `--w_slip_c 0` 时整段退化成逐位一致的老路径。
- 均衡采样用**自己的 numpy 流** + 累积分布 searchsorted (O(bs log N)), 不碰 torch RNG,
  掩码/初始化那几路随机流与非均衡运行保持同源。
- EVAL 行新增 `c_pen / c_cmaint / c_cbce / c_cframe / slipc_dec_ms`。

### `contact_flip_test.py` (新) — 聋子测试
接触状态是姿态的确定性函数 (§7 纠正四), 模型完全可以从姿态自己推出接触、把输入通道当
装饰品晾着, 而 loss 曲线一切正常。**XW 那条臂当初就是这样, 只有归零测试问出来。**
三条臂 (flip_off / flip_on / **perm 对照臂**: 接触数量不变、位置全错) 看
`delta_cm / bleed_cm / dh_cm 带符号 / sign_rate / code_chg`。
判据跑之前写死: `delta_cm < 0.5` 或 `< 3x bleed_cm` = **FAIL, 通道是装饰品, 该臂的接触
数字全部作废**; `sign_rate < 0.60` = WEAK。
源侧通道 -> 目标侧身体的关节对应经 `contact_label.limb_groups` 映射 (不另立口径)。

### `v2_judge.py` — 新列 + GROUND 卷子
- 每方向新增 `pen_max_cm / pen_p95_cm / foot_pen_* / skate_cms / skate_p95_cms`。
  地板口径与训练**同源** (同一条恒等式), 不另立。`pen_p95` 的 population 写死为
  **逐帧最深穿透的 p95** —— 逐 (帧,关节) 的 p95 恒等于 0 (大多数关节永远不穿地),
  那是个假的好看数字。滑移只在**窗口内**取相邻帧且要求 t 与 t+1 都判接触。
  缺标注时这几列为 `None`, 不猜。
- `--wset GROUND`: 从 ground/fall/crawl 片段抽窗。**必须单开**, 因为 FIXED 那份卷子的
  val 片段是 aiming/dance/fight/sprint/walk —— **一个地面片段都没有**, 手/地面的活儿
  在它上面是盲测的。LAFAN 的 7 个 val 片段里也没有地面族, 所以 GROUND 退回全部片段并把
  `holdout=false` 如实写进结果文件, 不装作是留出的。
- `res["guard"]`: 预注册回归护栏 (`leg_mean_deg` / `amp_all` / `ul_mean_cm` /
  `hand_mean_cm` 不许退化) + 验收线 (`pen_p95_cm` <1cm, `skate_cms` <15cm/s) + 盲区说明。

## 已测到的两个数

同一个 3.67M 基线 (`P0V2_base_0830`), 换卷子就换结论 —— FIXED 一直在替它遮丑:

| 方向 | 卷子 | leg_mean_deg | pen_p95_cm | skate_cms |
|---|---|---|---|---|
| human->g1_29dof | FIXED | 16.96 | 16.18 | 45.18 |
| human->g1_29dof | GROUND | **21.52** | **19.00** | 42.64 |
| human->gr3 | FIXED | 8.52 | 9.41 | 50.79 |
| human->gr3 | GROUND | **11.84** | **20.60** | 44.58 |

穿地量级与 spec §9 预注册的 10.4/14.2/10.7cm 对得上 (独立实现、同一量级 = 尺子可信)。
两条验收线 (1cm / 15cm/s) 现在差 10-20 倍, 靶子很清楚。

## 跑法

```bash
bash p3_smoke_identity.sh          # 先证"全关 = 逐位一致" (CPU, 不占卡)
bash p3_pilots.sh                  # GPU 空了再跑; 4 条阶梯臂, 每臂考 FIXED+GROUND+聋子测试
```

读表顺序**别倒过来**: 先看 flip 判决 (FAIL 则该臂接触数字作废) -> 再看 GROUND 的
pen/skate 有没有往验收线走 -> 最后逐条对护栏。数字看完必须渲一段地面动作视频再下结论。
