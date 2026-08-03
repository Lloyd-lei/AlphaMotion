# 人体旋转/twist 瓶颈 —— 调查结论(文献 + 实验)

日期:2026-08-02 · 承接:上轮发现"人体绝对 per-joint 全局旋转卡 ~90°,机器人正常"

> 问题:为什么共享 FSQ 码重建不出人体的绝对旋转?怎么解?
> 方法:先查文献定位,再用实验逐一证伪各假设。

---

## 一、一句话结论

**这不是"twist 学不到"的问题,是"FK 沿运动链放大 per-joint 误差 + twist 位置不可观测"两件事叠加的结构性问题,单帧紧凑码内无解。** 正确做法:**按下游需求选解码目标**——机器人(产品目标)用全局帧解码,twist 非问题;人体高保真 twist 移出单帧码,交给时序/先验。多个直觉修法已被实验证伪。

---

## 二、被实验证伪的修法(负结果,省得再走)

| 修法 | 假设 | 结果 | 裁决 |
|---|---|---|---|
| twist-aware 头(swing+twist 分开输出+专门监督) | twist 被整体旋转 loss 淹没 | 人体 twist 89.9°(没救),机器人 13°→32°(退化) | ❌ 有害 |
| per-joint 瓶颈(不池化,~2500bit/帧) | 池化把 per-joint 高频丢了 | 人体 global 仍 90.4°,位置反而更好(0.72cm) | ❌ 非池化问题 |
| 残差 FSQ 加深(1→2→4 级) | 比特不够 | 位置单调改善(1.44→0.82cm),人体旋转不动 | 位置有用,旋转无效 |
| 加大容量 latent(32→64)、旋转主导 loss(上轮 v7) | 容量/权重不够 | 人体旋转仍 91° | ❌ 非容量/权重 |
| decode-to-local(编码 global、解码 local 旋转) | global 目标不好学,local 好学 | local 角度 24° 但 FK 位置累积到 20cm+、global 113° | ❌ 换成更糟的位置累积 |

**关键对照:同一份数据,解码 global → 位置 1cm/旋转 90°;解码 local → local 24° 但位置 20cm。两头不可兼得。**

---

## 三、机制(为什么无解)

1. **FK 误差放大**:全局量 = local 旋转沿运动链累乘。深层关节的全局旋转依赖所有祖先,任何 per-joint 小误差被指数放大。紧凑码给不出足够低的 per-joint 误差,所以要么位置好(直接约束 global 位置)、要么 local 角度好,不能同时。
2. **twist 位置不可观测**(文献 HybrIK/NIKI):绕骨轴的 twist 不改变子关节位置,位置监督对它零梯度。机器人 1 自由度铰链没有 twist 自由度 → 旋转随结构确定 → 重建到 13°;人体 3 自由度球关节 twist 真实存在 → 无从位置约束。
3. **跨 rig 无真值**:即便重建救回,人体前臂 twist 在机器人上也没有对应,不该进"跨拓扑共享码"。

---

## 四、文献给的正解方向(与我们数据一致)

- **twist 要额外信号**:HybrIK 从图像特征回归 twist;单帧姿态本身给不了。
- **twist 要时序**:伸直的肢体单帧 twist 天然歧义,需时间窗口消歧(SmoothNet 等)。
- **twist 建成分布**:ProHMR / HuManiFlow / SO(3) flow-matching 把旋转建成分布,按 NLL / best-of-N 评,而非点误差 MPJPE。
- **离散码丢高频细节**:RVQ 深度补细节(MoMask)——对位置成立(我们验证了),对人体 twist 不成立(因为它根本没进码,不是被量化掉)。

---

## 五、落到设计上的决定

1. **产品线(翻译到机器人、实时)**:全局帧解码 + 残差 FSQ。twist 非问题(机器人低自由度),位置是可靠的高精度指标(残差 4 级 1.78cm,随比特改善)。**这条可以直接 scale。**
2. **单帧 tokenizer 只编码 swing + 位置**(可重建、跨拓扑有意义、高精度的部分),**twist 不进共享码**。
3. **人体 twist 高保真**(若将来要 SMPL 蒙皮)是独立子问题:走时序模型 + 每-rig twist 先验/分布,不在单帧码里解。已证实单帧内加结构(twist 头)有害、加容量无效。

---

## 六、复现

```
conda activate ardy && cd pose_translator_design/exp_topo
# 证伪实验(负结果)
python train_global.py --target global_st --w_twist 1 ...   # twist 头(有害)
python train_global.py --pool_mode per_joint ...            # per-joint(无效)
python train_global.py --target glob2loc ...                # decode-to-local(更糟)
# RD 曲线:--fsq_stages 1/2/4
```
