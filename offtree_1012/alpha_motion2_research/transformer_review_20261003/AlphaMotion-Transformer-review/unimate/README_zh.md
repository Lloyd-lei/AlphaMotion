# UniMate：Transformer 风格重绘

本轮只有两张 UniMate 图，不涉及 UMR 或 AlphaMotion-2。两图均按 510×640 pt 原生尺寸绘制，算子字体至少 9 pt，注释至少 8 pt。提供可编辑 SVG、矢量 PDF、240 dpi PNG、90 dpi 预览和 `draw_unimate.py`。

**主图 `unimate_01_main` 从下向上看。** 最下方 x_tau 是当前生成迭代的带噪状态，不是已完成的源动作。它经 root/joint MLP、T-pose 前缀和 depth/name 相加后，进入 10 个不同参数的时空块。T-pose 的逐关节编码明确分成两路：一路成为时间前缀，一路进入全局 TposPool。右侧条件 c 为生成时间 tau 的 sin/cos 编码及 MLP、FLAN-T5 caption 均值投影、T-pose 池化三者相加。tau 与运动序列的 frame 维度不同。

输出侧先进行 RMS 和条件仿射，再按 root/joint 分开。root 用自己的 RMS 特征作 Q，以 non-root 的 RMS 特征作 K、原始 non-root 特征作 V，经 cross-attention 和残差相加；两路分别投影为 12 维，拼接并去除 T-pose 前缀后得到速度场。最右侧回路是 ODE 更新生成状态，不是神经网络层的重复次数。

**单块图 `unimate_02_block` 也是从下向上看。** 顺序是空间注意力、时间注意力、SwiGLU。每条路径都先 RMSNorm，再用条件 scale/shift，再算 attention 或 FFN，最后乘 gate 后与黑色旁路的原输入相加。圆形加号表示残差加法。Q/K/V 三条箭头都来自当前分支的调制后 token，不是 caption token。紫色虚线仅表示条件，黑色线表示数据与残差。SiLU→Linear(512,4608) 为每个块独立的条件模块，拆出三个分支所需的九个向量。

这两图使用 Graph AdaLN 的实际计算顺序，未套用经典 Transformer 的 post-norm 结构，也未引入另一变体的 text cross-attention 或 AlphaMotion-2 的 SE(3) 任务输入。输入和输出 MLP 框省略 SiLU 文字以保留可读性；其层定义在 source-map 中链接到原实现。空间注意力的 Q/K 归一化、谱 RoPE 和图 bias 保留在图中；完整参数表由主研究文档承接。
