# 源码对齐逐层计算附录

本附录把“实际模型配置”“计算模块”“训练目标”“非神经优化”分开。完全重复的层用次数标明，每种不同神经运算展开；没有把类构造默认值当作当前demo配置。本文是源码审计，不是性能复现实验。

B=batch；Js/Jt=源/目标关节数；T=动作帧数；N=表面点数；Linear(a,b)均含bias，除非条目特别注明；×L为独立参数重复层，共享参数另标。

FSQ属于表示/量化；flow matching与diffusion属于生成目标。二者不构成天然互斥的同层架构选项。

## AlphaMotion-1：当前demo megatron / DualTranslator

当前demo有效config由主审实时读取；/home/arenalabs/Desktop/AlphaMotion/weights_staging/megatron/config.json；原run /var/tmp/alphamotion/topo_exp/runs/L_M20k_0903/config.json。权重SHA256 c9ab598df182d2d6b2fb804282acdd98fa86d4c2da6581c13db2f1e0b51011e5，文件71932184bytes。层表由固定源码独立复核；参数总数为服务读取值，本文未重新load该权重。 本附录另已读取 verified-megatron-config.json 与 verified-megatron-tensors.json，独立核对14个关键权重形状；383个tensor共17,974,798元素，减去两组13维_levels buffer，参数17,974,772。没有重新运行权重推理，形状核对不等于性能验收。

当前主路径：D256，8头，encoder4，position K4 + rotation K128，两路各4组Cross→Self decoder，13维FSQ各1stage、levels32；SigLIP768；DOF19开启；contact输入/输出结构开启；root_pos开启而root_head关闭。训练辅助16维Judge存在。

### A01 冻结关节文本接口

重复：每个不同关节名一次，可缓存。形状：`关节名字符串` → `[J,768]`。

normalize_joint_name → “the {name} joint of a body” → SigLIP text tower pooled feature → L2 normalize。模型外缓存，在线主网络接收已算好的向量。

当前 megatron 用 SigLIP。Qwen3-Embedding-0.6B 是已有替换分支：左 padding，最后一个有效 token，1024 维，再 L2 normalize；不是当前权重的文本塔。冻结文本塔内部的 Hugging Face Transformer 实现不在 exp_topo 中；本文逐层展开本仓库学习模块及其接口，不虚构该外部依赖的内部层。

源码：[AlphaMotion1/semantics.py:23–48](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/semantics.py:23>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/semantics.py`；[AlphaMotion1/semantics.py:72–102](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/semantics.py:72>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/semantics.py`。

### A02 共享 JointFeature

重复：源与目标各计算一次；同一组参数。形状：`[B,J,15+19]；sem [B,J,768]` → `[B,J,256]`。

g=Linear(34,256)([rest_offset3, normalize(offset)3, bone_length1, sin(depth/10^k)4, cos(depth/10^k)4, DOF19]) + Linear(256,256)(GELU(Linear(768,256)(sem)))，k=0..3。

有效 use_dof=true：3个DOF槽各含axis3、limited1、half_range/π、center/π，共18，再加DOF_count/3。use_sem=true，因此无joint-index embedding。几何尺度来自数据spec/reach，须保留原预处理。

源码：[AlphaMotion1/model.py:176–210](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:176>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`；[AlphaMotion1/model_dual.py:67–70](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:67>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A03 姿态嵌入与遮挡

重复：1。形状：`[B,Js,9]` → `[B,Js,256]`。

h=Linear(9,256)([global_rotation6D, root_relative_position/reach]); 若 joint_mask 为 True，h_j←learned mask_token[256]。

这是输入姿态遮挡；attention本身全双向，非因果。关节的几何条件不会被mask token删除。

源码：[AlphaMotion1/model_dual.py:71–74](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:71>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:156–160](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:156>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A04 接触输入与几何相加

重复：1。形状：`h [B,Js,256]；contact [B,Js,1]` → `[B,Js,256]`。

h←h+Linear(1,256)(nan_to_num(contact))*(~isnan(contact)) + g_source。无contact输入则跳过接触项。

必须在pose mask之后注入；NaN标签贡献0。代码只处理NaN，不应将±Inf自动视作合法输入。Linear初始化全0。

源码：[AlphaMotion1/model_dual.py:111–114](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:111>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:161–174](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:161>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A05 拓扑编码 EncBlock

重复：4。形状：`[B,Js,256]` → `[B,Js,256]`。

x←x+HopAttn(LN256(x)); x←x+Linear(1024,256)(GELU(Linear(256,1024)(LN256(x))))。每层独立参数。

LN为PyTorch LayerNorm默认eps=1e-5，有scale/bias；模型代码未设置dropout。

源码：[AlphaMotion1/model.py:239–249](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:239>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`；[AlphaMotion1/model_dual.py:74–74](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:74>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A06 HopAttn 内部

重复：每个 EncBlock 一次；encoder4 + decoder两流各4，共12组。形状：`[B,J,256]；hop [B,J,J]` → `[B,J,256]`。

Q,K,V由独立Linear(256,256)投影；reshape为8头×32维。A=softmax(QKᵀ/√32+E_hop[min(hop,17)])；输出Linear(256,256)(concat_heads(A V))。E_hop为18×8。

hop是无向骨架树BFS距离。类支持padding mask，但DualTranslator当前调用传None；因此跨样本变长需调用层正确组织。

源码：[AlphaMotion1/model.py:19–40](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:19>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`；[AlphaMotion1/model.py:213–236](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:213>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`。

### A07 两组潜变量查询池化

重复：position 1次，rotation 1次。形状：`learned Qp [4,256], Qr [128,256]；source H [B,Js,256]` → `zp_pre [B,4,256]；zr_pre [B,128,256]`。

CrossBlock(Q,H): Q←Q+MHA8(LN256(Q),LN256(H),LN256(H)); Q←Q+Linear(1024,256)(GELU(Linear(256,1024)(LN256(Q))))。

MHA每头32，内部Q/K/V/O线性投影。position/rotation查询与pool参数独立。

源码：[AlphaMotion1/model.py:252–264](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:252>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`；[AlphaMotion1/model_dual.py:83–93](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:83>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:121–141](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:121>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A08 独立 FSQ 瓶颈

重复：position与rotation各1；每流1 stage。形状：`[B,K,256]，K=4或128` → `grid [B,K,13]；decoded latent [B,K,256]`。

f=Linear(256,13)(z); a=(levels−1)/2=15.5; u=tanh(f)⊙a; q=(u+stopgrad(round(u)−u))/a; zq=Linear(13,256)(q)。position与rotation各自拥有Win/Wout。

有效levels=[32]×13，单stage，不是ResidualFSQ。训练可附code_noise均匀扰动，再除a；是否启用须看run。bits字段按Σlog2(levels)记账；偶数levels与饱和/round边界不能仅按参数名推断实际占用整数集合，本文不声称精确码字覆盖数。

源码：[AlphaMotion1/model.py:49–95](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:49>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`；[AlphaMotion1/model_dual.py:91–93](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:91>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A09 目标双流解码

重复：两流各4个 CrossBlock→EncBlock。形状：`target g [B,Jt,256]；zp[B,4,256]；zr[B,128,256]` → `Hp,Hr [B,Jt,256]`。

Hp=Hr=JointFeature(target); 每层：Hp←CrossBlock(Hp,zp); Hp←EncBlock(Hp,hop_t)；rotation同理使用zr与独立参数。CrossBlock和EncBlock各自都含FFN。

合计8个decoder cross blocks + 8个decoder self blocks，不能误画为每流4个普通Transformer层。

源码：[AlphaMotion1/model_dual.py:94–105](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:94>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:209–215](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:209>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A10 旋转输出头

重复：1。形状：`Hr [B,Jt,256]` → `global rot6D [B,Jt,6]`。

rot6D=Linear(256,6)(LN256(Hr))。后续Gram-Schmidt把6D转SO(3)，FK读取目标骨架。

当前rot_head=free。angles模式是另一变体，输出9维并按关节DOF做FK，不应混入本图。

源码：[AlphaMotion1/model_dual.py:98–105](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:98>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:216–232](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:216>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/common.py:23–31](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/common.py:23>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/common.py`；[AlphaMotion1/dual_common.py:294–316](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/dual_common.py:294>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/dual_common.py`。

### A11 位置输出与实际根高监督

重复：1。形状：`Hp [B,Jt,256]` → `raw hpos [B,Jt,3]；root height [B]`。

raw_hpos=Linear(256,3)(LN256(Hp))；有效root_pos路径读取raw_hpos[:,root,0]，SmoothL1监督根离地高（米）。

当前root_head=false，因此没有独立hroot神经层。旧138行注释写height+两速度，中间295行写height+偏移，但实际448–451只监督第0维高度；剩余2维不可标成经训练验证的速度/位移。骨架点位置指标在fk_loss打开时来自旋转FK，与hpos槽不同。

源码：[AlphaMotion1/model_dual.py:98–98](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:98>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:231–245](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:231>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/train_dual.py:438–453](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:438>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_dual.py`。

### A12 接触输出头

重复：结构存在，ret_con=True才输出。形状：`Hr [B,Jt,256]` → `contact logits [B,Jt]`。

logits=squeeze(Linear(256,1)(LN256(Hr)))；soft labels监督由外部loss决定；输出不是自动sigmoid后的概率。

当前config w_cbce=0：此头结构/权重存在，却没有该BCE训练信号；w_cgeo=1为从旋转FK高度计算的几何接触监督，不能将hcon标成已训练接触预测能力。

源码：[AlphaMotion1/model_dual.py:111–114](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:111>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/model_dual.py:246–251](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:246>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`。

### A13 训练辅助 JudgeHead

重复：1。形状：`zp [B,4,256]；zr [B,128,256]` → `[B,16] logits`。

v=concat(mean_K(zp),max_K(zp),mean_K(zr),max_K(zr)) [B,1024]；logits=Linear(256,16)(GELU(Linear(1024,256)(v)))。

当前n_judges=16，judge_mode=grad，BCEWithLogits对本体one-hot标签，权重0.1，梯度流向主干；是辨别本体的辅助支路，不是GRL对抗去本体支路，也不是动作映射必经路径。

源码：[AlphaMotion1/model_dual.py:13–24](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model_dual.py:13>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model_dual.py`；[AlphaMotion1/train_dual.py:633–643](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:633>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_dual.py`。

### 可对照源码的伪代码

```python
# megatron: single frame; source and target may have different J
g_s = JointFeature(geometry34_s, frozen_siglip768_s)
h = Linear9to256(pose9_s)
h = where(pose_hidden, learned_mask_token, h)
if contact is supplied:
    h += Linear1to256(nan_to_num(contact)) * (~isnan(contact))
h += g_s
for block in encoder[0:4]: h = block(h, hop_s)
z_p = FSQ13(PoolCross(query_p[4], h))
z_r = FSQ13(PoolCross(query_r[128], h))
p = r = JointFeature(geometry34_t, frozen_siglip768_t)
for k in range(4):
    p = SelfBlock_p[k](CrossBlock_p[k](p, z_p), hop_t)
    r = SelfBlock_r[k](CrossBlock_r[k](r, z_r), hop_t)
rot6 = Linear256to6(LN(r))
raw_pos = Linear256to3(LN(p))
root_height = raw_pos[:, root_index, 0]  # actual root_pos loss: metres
contact_logits = Linear256to1(LN(r))   # requested with ret_con
# No independent hroot in this checkpoint; JudgeHead is an auxiliary branch.
```

## UniMate：默认GraphAdaLN + flow matching

固定仓库SHA 2c5b384715aa63d8639b1ed7eb74bfe614570c7a 的 configs/uniml3d_60frames_graph_adaln.json；本附录核查的是发布配置/实现，不声称所有公开checkpoint均采用此配置。

配置主路径：60帧+1个Tpose前缀，最多60 joints，12D，D512，10个空间→时间→SwiGLU block，8头。主FFN真实hidden1365；TposePool为4个query/4 heads。caption与关节名采用冻结FLAN-T5-base768。

### U01 动作特征构造（非神经）

重复：1。形状：`global positions/rotations + topology` → `x [B,J,12,T]，T≤60，J≤60`。

[RIFKE_position3 | rotation6D6 | local_velocity3]。RIFKE减root XZ并旋入facing坐标，Y仍绝对高度；velocity是相邻帧位移旋入目的帧facing坐标。rotation先相对Tpose重基，再按HML把父关节rotation存至子slot。

root: pos=[0,height,0]，rot6D=facing quaternion，velocity=局部位移；原代码未除Δt，勿直接称SI m/s。根XZ由累计恢复。原始数据单位和归一化另看data管线，不能统一标米。

源码：[UniMate/unimate/utils/motion_utils.py:36–179](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/utils/motion_utils.py#L36-L179)。

### U02 冻结 FLAN-T5 文本接口

重复：caption与joint names均可预计算。形状：`token ids/mask` → `caption [B,768]；names[B,J,768]`。

T5EncoderModel.last_hidden_state [B,L,768]；pooled=Σ_l(mask_l h_l)/max(Σ_l mask_l,1)。factory设置finetune=False，无梯度。

GraphAdaLN消费masked mean caption，不是逐词cross-attention；FullCrossAttn可消费未pool的tokens。外部FLAN-T5内部层属于Transformers依赖，本附录不以不在仓库的实现冒充本地逐层证据。

源码：[UniMate/unimate/models/text_encoder/t5.py:175–238](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/text_encoder/t5.py#L175-L238)；[UniMate/unimate/models/text_encoder/t5.py:269–291](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/text_encoder/t5.py#L269-L291)；[UniMate/unimate/models/text_encoder/factory.py:184–193](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/text_encoder/factory.py#L184-L193)。

### U03 root/joint动作与Tpose嵌入

重复：4组独立MLP。形状：`root或joint raw12D` → `motion[B,T,J,512]；tpose[B,1,J,512]`。

每组Linear(12,512)→SiLU→Linear(512,512)，分别对应root_x、joint_x、root_tpose、joint_tpose。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:225–292](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L225-L292)。

### U04 父Tpose融合

重复：1组parent MLP + 1组fuse MLP，non-root共享。形状：`parent raw[B,J−1,12]；child_emb[B,J−1,512]` → `tpose_joint[B,J−1,512]`。

p=Linear(512,512)(SiLU(Linear(12,512)(parent_raw))); fused=Linear(512,512)(SiLU(Linear(1024,512)(concat(child_emb,p))))。

root不拼parent。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:242–248](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L242-L248)；[UniMate/unimate/models/denoiser/blocks/common.py:271–280](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L271-L280)。

### U05 时间前缀与关节条件

重复：1。形状：`motion[B,T,J,512]；tpose[B,J,512]` → `H[B,T+1,J,512]`。

H=concat_time(tpose,motion)；H+=Embedding(max_depth+1,512)(depth)+Linear(768,512)(joint_name_emb)，关节条件广播到每帧。

T=60时总61帧，满额3660tokens。use_graph_emb=false，因此无GCN加法，但graph attention bias仍开启。max_depth来自数据不是固定60。dropout=0。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:284–298](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L284-L298)；[UniMate/unimate/models/denoiser/base.py:134–147](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L134-L147)；[UniMate/unimate/models/denoiser/base.py:355–369](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L355-L369)。

### U06 时间与caption驱动

重复：1。形状：`flow time t[B]；caption[B,768]` → `y[B,512]`。

e_t=Linear(512,512)(SiLU(Linear(256,512)([cos(tω_k),sin(tω_k)])))，ω_k=exp(−log10000·k/128), k=0..127；y=e_t+Linear(768,512)(CFGMask(caption))。

训练caption向量以概率0.1整体置零；之后线性层bias仍保留。文本tower本身不随motion模型训练。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:81–106](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L81-L106)；[UniMate/unimate/models/denoiser/base.py:252–266](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L252-L266)；[UniMate/unimate/models/denoiser/base.py:330–336](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L330-L336)；[UniMate/unimate/models/denoiser/base.py:378–389](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L378-L389)。

### U07 Tpose查询池

重复：1。形状：`tpose[B,J,512]；learned queries[1,4,512]` → `pooled[B,512]加到y`。

q←q+MHA4(RMS(q),RMS(tpose),RMS(tpose))；q←q+SwiGLU_1024(RMS(q))；pooled=Linear(512,512)(RMS(mean_4(q)))。

MHA4每头128；SwiGLU隐藏1024（W12:512→2048，W3:1024→512），与主block1365不同。有效joint mask排除padding。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:128–175](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L128-L175)；[UniMate/unimate/models/denoiser/graph_adaln.py:73–82](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/graph_adaln.py#L73-L82)。

### U08 共享谱SignNet→空间RoPE

重复：1个共享参数模块；每个block调用。形状：`Laplacian eigenvectors[B,J,8]` → `angles[B,J,32]；cos/sin[B,J,64]`。

φ:Linear(1,64)→GELU→Linear(64,64)；h_k=φ(v_k)+φ(−v_k)；ρ:Linear(8×64,64)→GELU→Linear(64,32)；θ=ρ(flatten h)。θ复制为64维；Q′=Q cosθ+rotate_half(Q) sinθ，K同理。

可保证特征向量符号翻转不变；不能由此宣称重特征值空间任意基旋转不变。每层与所有帧共用该谱模块参数。

源码：[UniMate/unimate/models/denoiser/rope.py:197–244](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/rope.py#L197-L244)；[UniMate/unimate/models/denoiser/rope.py:324–350](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/rope.py#L324-L350)；[UniMate/unimate/models/denoiser/rope.py:399–414](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/rope.py#L399-L414)；[UniMate/unimate/models/denoiser/graph_adaln.py:38–56](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/graph_adaln.py#L38-L56)。

### U09 GraphAdaLN modulation

重复：10。形状：`y[B,512]` → `9组[B,512]：shift/scale/gate×3`。

m=Linear(512,4608)(SiLU(y))；split成空间、时间、FFN的β、γ、α。每路先RMSNorm，再(1+γ)⊙RMS(x)+β，输出乘α后residual加回。

RMS(x)=w⊙x/sqrt(mean(x²)+1e−6)，统计float32；无减均值。adaLN是模块名，实际norm是RMSNorm。

源码：[UniMate/unimate/models/denoiser/blocks/graph.py:281–287](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L281-L287)；[UniMate/unimate/models/denoiser/blocks/graph.py:299–341](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L299-L341)。

### U10 空间graph attention

重复：10。形状：`[B,61,J,512]` → `同形状`。

每帧独立：Linear512→1536取得QKV；8头×64维，Q/K先RMSNorm64再谱RoPE。A=softmax_fp32(Q′K′ᵀ/8+b_dist+b_relation+mask)，AV合头→Linear512→512。

graph-dist:Embedding(6,128)→Linear128→8×learned_scale；relation同样6×128→8，各scale初值.02；每层自有bias表（share_graph_attn_bias默认false）。表索引来自预处理graph_dist/joint_relations。

源码：[UniMate/unimate/models/denoiser/blocks/graph.py:97–196](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L97-L196)。

### U11 时间attention

重复：10。形状：`[B,61,J,512]` → `同形状`。

每joint独立：reshape[B·J,61,512]，Linear512→1536；8头×64，Q/K RMSNorm64，1D时间RoPE；masked SDPA(Q,K,V)→Linear512→512→reshape。

非因果，mask处理有效帧，含Tpose前缀。默认dropout0，投影和attention无有效随机drop。

源码：[UniMate/unimate/models/denoiser/blocks/graph.py:199–245](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L199-L245)；[UniMate/unimate/models/denoiser/base.py:180–187](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L180-L187)。

### U12 主block SwiGLU

重复：10。形状：`[B,61,J,512]` → `同形状`。

m=int(2/3·2048)=1365；[a,b]=chunk(Linear(512,2730)(x)); f=Linear(1365,512)(SiLU(a)⊙b)。由本block第三组adaLN门控制residual。

2048是配置ff_size，不是实际gated hidden，也不是FFN输出维。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:64–74](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L64-L74)；[UniMate/unimate/models/denoiser/blocks/graph.py:281–286](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L281-L286)；[UniMate/unimate/models/denoiser/blocks/graph.py:337–341](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L337-L341)。

### U13 输出调制

重复：1。形状：`H[B,61,J,512]；y[B,512]` → `Hmod同形状`。

[β,γ]=split(Linear(512,1024)(SiLU(y))); Hmod=(1+γ)⊙RMS512(H)+β。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:469–491](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L469-L491)；[UniMate/unimate/models/denoiser/blocks/common.py:493–511](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L493-L511)。

### U14 root聚合

重复：每帧一次，共享参数。形状：`root[B·61,1,512]；nonroot[B·61,J−1,512]` → `root_agg[B·61,1,512]`。

r←r+MHA4(RMS512(r), RMS512(joints), joints)。Q/K做norm，V用原始joint features；mask剔除padding。

4头×128；不可误写Q/K/V全norm。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:474–479](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L474-L479)；[UniMate/unimate/models/denoiser/blocks/common.py:514–533](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L514-L533)。

### U15 两组velocity输出头

重复：root与nonroot各1组MLP。形状：`root/nonroot 512` → `[B,J,12,T]`。

root_out与joint_out分别Linear512→512→SiLU→Linear512→12，拼joint轴；重排[B,J,12,T+1]并删掉第0个Tpose前缀。

flow配置下输出为整个12维特征空间的速度场vθ，而非只输出局部速度通道，也不是直接clean motion。

源码：[UniMate/unimate/models/denoiser/blocks/common.py:481–487](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L481-L487)；[UniMate/unimate/models/denoiser/blocks/common.py:535–540](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L535-L540)；[UniMate/unimate/models/denoiser/graph_adaln.py:100–105](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/graph_adaln.py#L100-L105)。

### U16 初始化

重复：构造模型时一次。形状：`各可学习模块` → `adaLN-Zero初始状态`。

普通Linear Xavier；time与caption projection权重normal(.02)；10个block调制与最终调制Linear全0；最后12D heads全0；root cross-attn out-proj与Tpose pool最终projection全0。

这是训练初始化，不代表训练后gate/output为0。

源码：[UniMate/unimate/models/denoiser/base.py:201–246](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L201-L246)。

### 可对照源码的伪代码

```python
x0 = randn_like(clean_normalized_motion)
t = uniform(0, 1)
xt = (1-t) * x0 + t * clean_normalized_motion
target_velocity = clean_normalized_motion - x0
y = TimeMLP(sincos256(t)) + CaptionLinear(CFGMask(frozen_t5_mean))
h, tpose_emb, joint_valid = InputLayer(xt, tpose12, parent_tpose12)
y += TposeQueryPool(tpose_emb, joint_valid)
h += DepthEmbedding(depth) + JointNameLinear(frozen_t5_joint_names)
for block in ten_blocks:
    beta, gamma, gate = split_nine(block.modulation(y))
    h += gate.s * SpatialGraphAttention(modulate(RMS(h), beta.s, gamma.s))
    h += gate.t * TemporalAttention(modulate(RMS(h), beta.t, gamma.t))
    h += gate.f * SwiGLU1365(modulate(RMS(h), beta.f, gamma.f))
h = modulate(RMS(h), FinalModulation(y))
h.root += RootCrossAttention(RMS(h.root), RMS(h.joints), h.joints)
velocity = concatenate_joint(RootMLP(h.root), JointMLP(h.joints))
velocity = rearrange_and_remove_tpose_prefix(velocity)
x1_hat = xt + (1-t) * velocity
loss = masked_feature_MSE(velocity, target_velocity)
loss += .5 * masked_rotation_geodesic(denorm(x1_hat), denorm(clean_motion))
loss += .1 * masked_adjacent_velocity_smoothness(denorm(x1_hat))
# Inference: solve dx/dt = CFG(v_theta(x,t,conditions)) from noise at t=0 to t=1.
```

## UMR：静态表面对应网络 + 约束重定向

固定仓库SHA d1cdaedaf67a827198da71c5fa831f48b2e51f91 的 humanoid_retarget_defaults.json。Python pipeline500epochs,batch2；standaloneCLI默认5000/batch7；browser运行参数单独审计，不能混图。

配置主路径：静态表面4096点；PointNet通道3→64→128→512→1024，全局max；MLP1024→512→512→12288；有序模板+残差。之后是独立的slots绑定、MuJoCo几何FK/Jacobian与Clarabel优化。

### R01 PointNet逐点层1

重复：1。形状：`P[B,N,3]→transpose[B,3,N]` → `[B,64,N]`。

Conv1d(3,64,kernel=1,bias=True)→BatchNorm1d64→ReLU。

N=4096为pipeline默认，BN使用PyTorch默认eps=1e−5、momentum=.1、affine与running stats。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:244–261](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L244-L261)。

### R02 PointNet逐点层2

重复：1。形状：`[B,64,N]` → `[B,128,N]`。

Conv1d(64,128,1)→BatchNorm1d128→ReLU。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:250–261](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L250-L261)。

### R03 PointNet逐点层3

重复：1。形状：`[B,128,N]` → `[B,512,N]`。

Conv1d(128,512,1)→BatchNorm1d512→ReLU。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:250–261](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L250-L261)。

### R04 逐点latent投影与全局池化

重复：1。形状：`[B,512,N]` → `z[B,1024]`。

Conv1d(512,1024,1)，随后沿N维逐通道max；此Conv之后无BN或ReLU。

全局max使编码对输入点次序置换不变；输出slot次序由模板约定，不是对输入点逐点回归。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:257–278](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L257-L278)。

### R05 模板残差decoder第一层

重复：1。形状：`z[B,1024]` → `[B,512]`。

Linear(1024,512)→ReLU。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:262–278](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L262-L278)。

### R06 模板残差decoder第二层

重复：1。形状：`[B,512]` → `[B,512]`。

Linear(512,512)→ReLU。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:262–278](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L262-L278)。

### R07 点残差输出与模板相加

重复：1。形状：`[B,512]；固定模板T[N,3]` → `slots S[B,N,3]`。

δ=reshape(Linear(512,3N)(h),[B,N,3]); S=T+δ；N4096时最终Linear512→12288。输出Linear权重/bias默认全0初始化。

T是register_buffer非学习参数；NN无文本塔、attention、时序动作输入。不能把“template residual”误画为Transformer residual block。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:262–278](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L262-L278)。

### R08 静态学习目标

重复：训练每step。形状：`S,δ与target pointcloud` → `scalar loss`。

L=1·L_CD+.002·L_rep+0·L_res+.4·L_edge；CD为双向最近邻平方距离均值之和；rep=mean exp(−d²_kNN/.035²)，k8；res=mean δ²；edge=mean_edges ||δ_i−δ_j||²，模板geodesic k32。

pipeline有效默认500epochs,batch2，AdamW lr.001,wd1e−4，cosine→1e−5；CLI单跑默认5000epochs,batch7及不同损失权重，不可混用。输入按每样本高度归一，训练noise_std默认.002。

源码：[UMR/scripts/train_correspondence_template_residual_ae.py:281–298](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L281-L298)；[UMR/scripts/train_correspondence_template_residual_ae.py:391–397](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L391-L397)；[UMR/scripts/train_correspondence_template_residual_ae.py:733–746](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L733-L746)；[UMR/humanoid_retarget_defaults.json:61–81](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/humanoid_retarget_defaults.json#L61-L81)。

### R09 表面slots绑定与运动学（非神经）

重复：静态绑定一次，每帧更新。形状：`有序source/robot slots；source motion；robot model` → `source slots随动；robot points/normals与Jacobian`。

静态slots绑定到各自表面；source运动生成目标点/法线；MuJoCo根据q做几何FK和点/法线Jacobian。点slot的索引建立跨表面关系。

这是运动学与几何查询，不是动力学rollout、策略网络或神经网络层。

源码：[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:657–712](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L657-L712)；[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:1763–1809](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1763-L1809)。

### R10 逐帧残差构造（非神经）

重复：每solver iteration。形状：`q；target slots/normals/contact；前1/2帧q` → `J[M,nv]，r[M]`。

点项sqrt(wp)(p(q)−p*)；法线sqrt(wn)(n(q)−n*)；关节平滑sqrt(ws)(q−qprev)；二阶平滑sqrt(wa)(q−2qprev+qprev2)；接触/物体项及不等式按配置加入。

self-contact clearance只在robot pair distance<source distance时激活。ground/anchor/object contact把cost直接乘残差/J而非sqrt(cost)，其二次目标权重实际平方；不能统一套sqrt(w)。

源码：[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:1770–1809](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1770-L1809)；[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:1811–1837](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1811-L1837)；[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:1839–1868](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1839-L1868)；[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:1890–1918](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1890-L1918)。

### R11 Clarabel约束步（非神经）

重复：有效默认每帧1次；pose_init15次。形状：`J,r；box/inequality/SOC约束` → `Δq[nv]→新qpos[nq]`。

P=JᵀJ+λI，g=Jᵀr；min_Δq .5ΔqᵀPΔq+gᵀΔq，约束lower≤Δq≤upper、AΔq≤b、可选||Δq_group||₂≤step。软不等式引入非负slack并以.5Σcost·slack²惩罚。随后mj_integratePos(q,Δq,1)并clamp关节范围。

有L2步长锥时是凸锥约束二次优化，不应只称无约束IK。log的cost=mean(r²)是更新前残差统计，并不等于整个目标或新的动力学可行性分数。默认另有LQR轨迹滤波。

源码：[UMR/scripts/smpl_surface_retarget_common.py:1556–1628](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/smpl_surface_retarget_common.py#L1556-L1628)；[UMR/scripts/smpl_surface_retarget_common.py:1659–1735](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/smpl_surface_retarget_common.py#L1659-L1735)；[UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py:2010–2117](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2010-L2117)；[UMR/humanoid_retarget_defaults.json:123–156](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/humanoid_retarget_defaults.json#L123-L156)。

### 可对照源码的伪代码

```python
# Stage A: static correspondence learning; no motion/text input to this NN
points = normalize_each_sample_by_height(static_surface_samples)
h = relu(BN64(Conv1x1_3to64(points.transpose(1,2))))
h = relu(BN128(Conv1x1_64to128(h)))
h = relu(BN512(Conv1x1_128to512(h)))
z = max_over_points(Conv1x1_512to1024(h))
delta = reshape(Linear512to12288(relu(Linear512to512(relu(Linear1024to512(z))))))
slots = ordered_template + delta
optimize(chamfer + .002 * repulsion + .4 * residual_edge_smoothness)
bindings = bind_ordered_slots_to_source_and_robot_meshes(slots)
# Stage B: per-motion-frame geometric optimization, outside the learned network
for source_frame in motion:
    source_points, source_normals, contact = update_source_surface(bindings, source_frame)
    for iteration in configured_iteration_count:
        robot_points, robot_normals, J = geometric_FK_and_Jacobian(q, bindings)
        r, J, box, inequalities, cones = build_residuals_and_constraints(...)
        dq = Clarabel(J.T @ J + damping*I, J.T @ r, box, inequalities, cones)
        q = clamp_joint_ranges(mj_integratePos(q, dq, dt=1))
    save(q)
```

## 配置变体、损失与解释边界

### AlphaMotion-1 可选 ResidualFSQ（当前megatron未启用）

先f=Win(z)，acc=0。stage k使用s_k=abs(stage_scale[k])+1；u_k=tanh((f−acc)·s_k)·a；q_k=STEround(u_k)/(a·max(s_k,1e−3))；acc+=q_k，最后Wout(acc)。每个stage量化剩余误差。FSQ是表示/量化瓶颈；flow matching/diffusion是生成目标，二者可串联并非同层互斥选项。

源码：[AlphaMotion1/model.py:98–164](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/model.py:98>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/model.py`。

### AlphaMotion-1 独立latent diffusion实验

train_latent_diffusion.py使用冻结tokenizer的pre-round features作连续空间；示例cache假设两流相同K，D=2Kf，不可直接把该旧cache形状套到megatron的4+128不等长两流。默认64帧窗=56干净context+8 noisy target。每帧Linear(D,384)+learned位置/segment embedding+时间MLP(1→384→384,SiLU)，然后6个PyTorch pre-LN TransformerEncoderLayer（8heads,head48，FF1536,GELU,dropout.1），Linear384→D只取最后8帧。内部forward没有causal mask；是按chunk的条件生成，不是token级causal self-attention。cosine DDPM默认T1000，代码实际默认v-pred：target=√αbar·ε−√(1−αbar)·x0；eps-pred可选，不能照旧文件顶部注释标默认epsilon。尚未以本次审计验证该实验的生成性能。

源码：[AlphaMotion1/train_latent_diffusion.py:22–70](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_latent_diffusion.py:22>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_latent_diffusion.py`；[AlphaMotion1/train_latent_diffusion.py:77–81](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_latent_diffusion.py:77>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_latent_diffusion.py`；[AlphaMotion1/train_latent_diffusion.py:108–118](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_latent_diffusion.py:108>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_latent_diffusion.py`。

### UniMate FullCrossAttn是另一份配置

full_cross_attn把61×60展为3660tokens做一次全时空self-attention；GraphRoPE给时间与谱joint编码。每层改为SelfAttn→TextCrossAttn→SwiGLU，10层。self+FFN使用6组adaLN的shift/scale/gate，驱动仅time+TposePool；caption经768→512逐词投影供cross K/V。CrossAttention有Q:512→512、KV:512→1024、8heads×64、Q/K RMSNorm、无文本RoPE、输出Linear512，纯residual相加，不乘额外adaLN gate。cross输出projection初始化0；CFG-drop用learned null caption。若推理只传pooled caption，会退成单token memory。默认GraphAdaLN图不能混入该模块；full attention路径没有graph-distance/relation logits bias。

源码：[UniMate/unimate/models/denoiser/full_cross_attn.py:27–78](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/full_cross_attn.py#L27-L78)；[UniMate/unimate/models/denoiser/full_cross_attn.py:80–143](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/full_cross_attn.py#L80-L143)；[UniMate/unimate/models/denoiser/blocks/common.py:419–457](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/common.py#L419-L457)；[UniMate/unimate/models/denoiser/blocks/cross.py:27–78](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/cross.py#L27-L78)；[UniMate/unimate/models/denoiser/base.py:268–328](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/base.py#L268-L328)。

### UniMate flow损失的归一化与采样

每样本Lflow=Σ_valid ||vθ−(x1−x0)||²/(T_valid·J_valid·12)。令xhat1=xt+(1−t)vθ，再按数据mean/std反归一化。Lgeo=Σ_valid acos(clamp((tr(RhatᵀRgt)−1)/2,−1+1e−6,1−1e−6))/(T_valid·J_valid)，是弧度角均值，不平方。Lsmooth只对反归一化xhat1的9:12通道做相邻帧差平方，分母max(T_valid−1,1)·J_valid·3，无GT，代码没有除Δt²。总L=Lflow+.5Lgeo+.1Lsmooth。一般生成调用sample_ode()默认dopri5、50个保存采样点、atol1e−6、rtol1e−3；自适应ODE的50并非50次网络计算。scheduler.diffusion_steps=100不能当flow NFE。inbetween/edit路径使用另一Euler clamp sampler。

源码：[UniMate/unimate/models/flow/transport.py:95–157](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L95-L157)；[UniMate/unimate/models/flow/transport.py:200–287](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L200-L287)；[UniMate/unimate/utils/geo_utils.py:6–22](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/utils/geo_utils.py#L6-L22)；[UniMate/unimate/models/flow/transport.py:476–508](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/flow/transport.py#L476-L508)；[UniMate/unimate/inference/generate.py:86–114](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/inference/generate.py#L86-L114)。

### AlphaMotion-1 loss/物理含义边界

DualTranslator本身不内置训练总目标。dual_common.run_dir返回全局旋转geodesic_deg与位置欧氏范数；位置是按reach归一、未平方，fk_loss=true时从旋转做目标骨架FK再比位置；可叠加var_floor、尾部/末端加权和contact几何项。root_pos只监督根高，Judge BCE为辅助项。精确总权重依run config，不凭类定义填写。由FK得到自洽关节点不能据此称动力学一致、真实接触力可实现或sim2real成功。

源码：[AlphaMotion1/dual_common.py:326–455](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/dual_common.py:326>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/dual_common.py`；[AlphaMotion1/train_dual.py:438–453](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:438>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_dual.py`；[AlphaMotion1/train_dual.py:633–643](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:633>)；远端原件 `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/train_dual.py`。

### UMR canonical slots与pipeline兼容性

point cloud输入无序，但输出索引绑定有序canonical template，因此跨网格对应是同slot索引。pipeline检查缓存的template/robot/training_mode；不兼容则重新执行静态对应训练，再进行运动重定向。UMR这条已公开主路径适配新的机器人不需要先获得其动作监督，但不等于已经实现一套固定NN权重读任意新mesh的zero-shot能力。AlphaMotion-2的局部区域文本语义锚定是新增研究假设，不能反写成UMR现有层。

源码：[UMR/scripts/humanoid_retarget_pipeline.py:420–464](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L420-L464)；[UMR/scripts/humanoid_retarget_pipeline.py:890–942](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L890-L942)；[UMR/scripts/humanoid_retarget_pipeline.py:1034–1056](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L1034-L1056)。

## 本次自查

所有列出的file:line均程序校验落在本地审计快照内，并验证整文件SHA256与source-manifest一致；UniMate/UMR引用固定commit。Tensor维度按forward运算逐项展开，未用假造张量输出冒充模型复现。外部冻结文本塔只审计明确加载版本、输出维数、池化与冻结接口，其依赖内部层未在这份快照中重建。

已将AlphaMotion-1有效config和真实safetensors header进行14项shape交叉核对，参数统计一致。UMR浏览器执行链/可视化测量工具由另一独立审计文档提供。图中的AlphaMotion-2必须标为候选/假设，不可画作上述三个已实现模型的实证结果。
