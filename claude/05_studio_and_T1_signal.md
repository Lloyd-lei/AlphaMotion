# AlphaMotion Studio（可视化工具）+ T1早期泛化信号

## 一、Studio：数据集浏览器 + 动作翻译软件

`src/any_body/cli/studio.py`，运行在 **http://localhost:8092**（arenalabs本机；远程访问需要SSH端口转发）。

```bash
python -m any_body.cli.studio --checkpoint runs/vae_20bit_seed0/best.pt --variant vae --port 8092
```

### Tab 1：Dataset —— 三个split并排看

- 顶部一张表直接列出每个split的 **帧数 / 演员数 / motion数**，说明各自留出的是什么
- 下拉切split、滑块选帧，源骨架（灰）和目标骨架（橙）并排显示
- **"Overlay all splits"** 打开后三个split同帧横向排开，可以直接肉眼比较分布差异
- 底部显示当前帧的 actor_uid / motion_id / 帧号 / `78J → 29J` 拓扑变换

### Tab 2：Translate —— 动作翻译软件

这是你要的"chatbox"。四个控件 + 可拖拽手柄：

| 控件 | 作用 |
|---|---|
| **Source split / frame** | 选源姿态，可以从train或任一验证集取 |
| **Target skeleton** | `g1_29dof`（有配对GT）/ `source`（同拓扑自重建）/ `pruned-38` / `pruned-24` / `pruned-20` |
| **Constraints** | `end effectors`（末端约束，核心用例）/ `full body` / `none`（纯脑补） |
| **红色拖拽手柄** | 每个被约束关节一个SE(3) handle，**拖动即时重解全身** |

实时显示：**自由关节误差（mm）** 和 **旋转误差（度）**——只统计未被约束的关节，所以数字不会被硬替换的GT稀释。

冒烟测试跑了 5种目标 × 3种约束模式 = 15种组合，全部正常。例如同一帧：

| 目标 | 约束模式 | 约束关节 | 自由关节误差 |
|---|---|---|---|
| g1_29dof | end effectors | 11 / 29 | 33.1 mm |
| g1_29dof | none | 0 / 29 | 39.9 mm |
| pruned-38 | end effectors | 30 / 38 | — |
| pruned-20 | end effectors | 14 / 20 | — |

### 写这个工具时抓到的两个bug

**1. `pruned-38/-24/-20` 三个选项产出的是同一个骨架。** 走 `dataset[index]` 会让 dataset 自己用 `rng.choice((20,24,38))` 再掷一次骰子，**下拉框里解析出的数字被丢掉了**——三个菜单项其实是同一个实验。改成直接调 `prune_skeleton(record, keep)`。修好后确认输出 38J / 24J / 20J 各不相同。

**2. `source (identity)` 同样受影响**，也改成显式构造。

这类bug的共性是**"看起来在配置，实际被下游随机覆盖"**——和之前那批"测一个不敏感的量"是同一族问题，只是方向相反：这次是**设了一个不生效的参数**。

---

## 二、发现并修复：`val_actor` 根本没有留出演员

写Dataset tab时列split统计，发现 `val_actor` 只有6个演员、`train` 只有5个——数字小得可疑。查了manifest：

```python
def stable_split(actor_uid, take_name):
    value = hash(f"{actor_uid}|{take_name}") % 100      # ← 哈希了 actor+take
    return "train" if value < 80 else ("val_actor" if value < 90 else "val_motion")
```

哈希键包含take，所以**同一个演员的不同take被散到三个split里**。全manifest实测：

| | 演员数 | **与train重叠的演员** |
|---|---|---|
| train | 517 | — |
| **val_actor** | 497 | **494 / 497** |
| val_motion | 498 | 496 / 498 |

**`val_actor` 里 99.4% 的演员都在训练集里见过。** 它留出的是take，不是演员——**和 `val_motion` 测的是同一件事**。

改成嵌套split：`val_actor` 只按**演员哈希**决定（整个演员进或出），剩下的演员里再按take哈希切 `val_motion`。修复后：

| | 演员数 | 帧占比 | 与train演员重叠 |
|---|---|---|---|
| train | 459 | 79.9% | — |
| **val_actor** | **61** | 9.2% | **0** ✓ |
| val_motion | 441 | 10.9% | 439（设计如此：同演员、留出take） |

**T1当前跑的还是旧split**，所以下面那组数字里 `val_actor` 应当读作"另一个held-out-take集合"。这个修复要重建缓存后才生效。

### 顺带修了缓存的采样偏差

`build_frame_cache` 按manifest文件顺序取行，所以 **40,000帧只覆盖了5个演员、221个motion**（可用的是522演员 / 142,220 motion）。加了确定性shuffle，让截断后的缓存是代表性抽样而不是文件前缀。

**这解释了为什么 `val_actor`（6演员）和 `train`（5演员）数字这么小**——两个都是各自split的文件前缀。

---

## 三、T1早期信号（VAE，2000步，40k训练帧）

**第一次有真正的held-out数字。** 每100步验证一次：

| step | train loss | val_actor mm | val_motion mm | val_actor rot° | val_motion rot° |
|---|---|---|---|---|---|
| 99 | 24.30 | 292.3 | 282.4 | 91.1 | 88.8 |
| 399 | 14.60 | 117.4 | 112.4 | 72.3 | 68.7 |
| **999** | 11.90 | **77.0** | **72.9** | 57.6 | 51.7 |
| 1499 | 8.73 | 130.7 | 126.7 | 52.0 | 45.5 |
| 1999 | 10.81 | 140.0 | 123.5 | **44.3** | **40.9** |

### 三个值得注意的现象

**1. 位置和旋转的学习行为完全不同。**

- **旋转单调改善**：91.1° → 44.3°，**降51%**，而且几乎没有反弹
- **位置在step 999触底（77.0mm）后开始震荡**：最后5个检查点 134.0 ± 23.9 mm

这个分化很有意思：训练loss还在降（24.3 → 10.8），旋转也还在改善，但**位置泛化误差在1000步后变差了**。像是位置通道开始过拟合而旋转通道还在学。这跟之前测的梯度占比对得上——position占40%、rotation只占16%，位置项先饱和。

**2. `val_actor` 和 `val_motion` 相关系数 0.9928，平均只差9.3mm（7%）。**

这是上面那个split bug的**直接实证**：两个split本来应该测两个独立的泛化轴（未见演员体型 vs 未见动作内容），实际上几乎完全同步。修好split后这两条曲线应该分开——**如果修好后仍然重合，那才是真的说明模型对演员体型不敏感**，那会是个正面结果。

**3. 最好的checkpoint在step 999，不是最后一步。** `best.pt` 现在按held-out误差选，所以保存的是step 999那个（77.0mm）而不是最终的140mm。**这正是为什么模型选择必须用验证集**——按训练loss选会挑到step 1999。

### 与T0的对照

| | T0（5k帧，15.4 epoch，训练集） | T1（40k帧，held-out） |
|---|---|---|
| 自由关节误差 | 145.7 mm | **77.0 mm**（最佳）/ 140 mm（最终） |
| 旋转误差 | 65.7° | **44.3°** |

T1的held-out最佳值（77mm）**优于**T0的训练集数字（145.7mm）——说明T0确实主要在记忆5k帧，而40k帧配上真验证集之后，模型学到的东西更实在。

---

## 四、下一步

**已就绪**：Studio可交互、验证集打通、码本修复已实施（真实利用率待T1量化变体跑完确认）。

**待做**：
1. **用修好的split重建缓存并重跑** —— 当前 `val_actor` 不是演员留出集，两条曲线的分离度是本轮最想看的东西
2. **位置通道在1000步后过拟合** —— 值得试更强的正则化或提高rotation权重
3. **GT噪声地板** 仍未测 —— 77mm距离retargeter自身噪声下限多远，不知道
4. T1剩余6个变体跑完后出完整对照表