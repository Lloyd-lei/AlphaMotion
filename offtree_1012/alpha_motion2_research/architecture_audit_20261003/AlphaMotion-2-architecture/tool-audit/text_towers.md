# 冻结文本塔：逐层计算补充

固定上游Transformers v4.51.3实现 + ArenaLabs checkpoint缓存config的依赖结构核查；不是已锁定历史runtime版本，也未运行模型加载/数值等价验证。

B为文本批次；L为token长度（不是动作帧数）；W[a→b]表示输入a、输出b的Linear。RMS(x)=gamma*x/sqrt(mean(x²)+1e-6)，不减均值、无beta；LN另减均值且有gamma/beta。所有公式按冻结eval路径写，dropout=identity。

|文本塔|重复层与宽度|注意力|最终文本表示|
|---|---|---|---|
|SigLIP text|12层；768→3072→768；12×64 heads|双向，QKᵀ/√64；learned absolute positions|最后位置经final LN和768→768 head；AlphaMotion再L2|
|FLAN-T5-base encoder|12层；768→2048→768；12×64 heads|双向，**QKᵀ无√d缩放**；32桶relative bias|final RMS；UniMate valid-token mean，或保留tokens|
|Qwen3-Embedding-0.6B|28层；1024→3072→1024；Q16/KV8，head_dim128|因果GQA；QK-RMS→RoPE；QKᵀ/√128|final RMS后最后有效token；AlphaMotion再L2|

## 1. SigLIP text tower

[Checkpoint config](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/config.json)。cacheconfig未显式存layers/activation/epsilon；按v4.51.3 SiglipTextConfig缺省解析为12/gelu_pytorch_tanh/1e-6。projection_size缺省hidden_size。不是声称已验证历史runtime。

|计算单元|形状与精确操作|来源|
|---|---|---|
|Token + learned position embedding|`ids[B,64]→H[B,64,768]`。H=Embed[32000,768](ids)+Pos[64,768](positions)|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py#L318-L356)|
|Pre-LN self-attention ×12|`Q,K,V[B,12,64,64]; A[B,12,64,64]`。U=LN(H); Q=U Wq+bq,K=U Wk+bk,V=U Wv+bv; A=softmax(QK^T/8+M_padding); H←H+concat(A V) Wo+bo; 所有Linear768→768，双向无causal mask|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py#L381-L509)|
|Pre-LN feedforward ×12|`[B,64,768]→[B,64,3072]→[B,64,768]`。H←H+W2 GELU_tanh(W1 LN(H)+b1)+b2；两次Linear均有bias|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py#L453-L509)|
|Final LN, last-position head|`[B,64,768]→[B,768]`。Z=LN(H); e=Linear[768→768](Z[:,-1,:]); tokenizer sticky-EOS假设下取最后位置，非搜索argmax/EOS索引|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py#L775-L839)|
|AlphaMotion normalization + cache|`[B,768]`。get_text_features仅返回上述head输出；调用端e←e/||e||₂，再按joint name缓存。图不应加入SigLIP vision tower。|[实现](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/semantics.py:36)|

```python
H = token(ids) + position(ids)
for layer in layers_12:
    H = H + layer.attn(layer.ln1(H), pad_mask)
    H = H + layer.fc2(gelu_tanh(layer.fc1(layer.ln2(H))))
e = head(final_ln(H)[:, -1])
return normalize(e, dim=-1) # AlphaMotion caller
```

## 2. FLAN-T5-base encoder only

[Checkpoint config](https://huggingface.co/google/flan-t5-base/blob/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/config.json)。

|计算单元|形状与精确操作|来源|
|---|---|---|
|Shared token embedding; no absolute position|`ids[B,L]→H[B,L,768]`。H=Embed[32128,768](ids); embedding dropout在eval为identity；只运行T5EncoderModel，无decoder/LMhead|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/t5/modeling_t5.py#L1997-L2019)|
|Pre-RMS self-attention ×12|`Q,K,V[B,12,L,64]; scores[B,12,L,L]`。U=RMS(H); Q=UWq,K=UWk,V=UWv; A=softmax(QK^T+R_bucket+M_padding); H←H+concat(A V) Wo；Q/K/V/O均768→768且无bias；没有除sqrt(64)|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/t5/modeling_t5.py#L351-L614)|
|Relative position bias once, reused ×12|`Embedding[32,12]→R[1,12,L,L]`。bucket(j−i)：双向各16桶，小距离0..7精确，大距离按max_distance128做对数分桶并截断到该方向的最后一桶；只有第一个block拥有表，其position_bias张量后续层复用|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/t5/modeling_t5.py#L404-L472)|
|Pre-RMS gated-GELU FFN ×12|`two [B,L,768]→[B,L,2048] branches→[B,L,768]`。U=RMS(H); G=GELU_new(U Wi0) ⊙ (U Wi1); H←H+G Wo；Wi0/Wi1:768→2048，Wo:2048→768，全部无bias|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/t5/modeling_t5.py#L304-L348)|
|Final RMS and caller pooling|`[B,L,768]→[B,768] or token sequence`。Z=RMS(H); pooled=sum_i(mask_i Z_i)/max(sum_i mask_i,1)；pool=False输出完整Z及调用方保留mask；空文本mask全0使pooled=0；无额外L2归一化|[实现](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/text_encoder/t5.py#L268-L294)|

```python
H = token(ids)
R = first_layer.relative_bias(bucket(j - i)) + pad_mask
for layer in encoder_layers_12:
    U = rms1(H)
    H = H + Wo(softmax(Q(U) @ K(U).T + R) @ V(U))
    U = rms2(H)
    H = H + Wffo(gelu_new(Wi0(U)) * Wi1(U))
Z = final_rms(H)
return Z if not pool else masked_mean(Z, mask)
```

缓存或cross-attention路径可先pool=False存token，再在adaLN分支求相同masked mean。
第一层bias共享证据：T5Stack第896行仅i==0，1153–1158行传递position_bias。
gated-gelu→gelu_new映射在configuration_t5.py:135–136；不是ReLU FFN，也不是SiLU SwiGLU。

## 3. Qwen3-Embedding-0.6B base model

[Checkpoint config](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json)。

|计算单元|形状与精确操作|来源|
|---|---|---|
|Embedding|`ids[B,L≤64]→H[B,L,1024]`。Embed[151669,1024]；AlphaMotion左padding，所有样本最后位置都要求有效；无learned绝对position embedding|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L467-L488)|
|Pre-RMS Q/K/V and per-head QK-RMS ×28|`Q[B,16,L,128]; K,V[B,8,L,128]`。U=RMS(H); Q=RMS_128(reshape(U Wq)); K=RMS_128(reshape(U Wk)); V=reshape(U Wv)。Wq:1024→2048；Wk,Wv:1024→1024；Q/K专用RMS各有128维gain、跨heads广播。|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L170-L222)|
|RoPE then causal grouped-query attention ×28|`K,V repeat2→[B,16,L,128]; scores[B,16,L,L]`。先对Q/K施加RoPE(theta=1e6)，再重复K/V到16heads；A=softmax(Q_rot K_rot^T/sqrt(128)+M_causal+M_padding)；H←H+concat(A V) Wo，Wo:2048→1024。Q/K/V/O无bias。|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L99-L253)|
|Pre-RMS SwiGLU ×28|`two [B,L,1024]→[B,L,3072] branches→[B,L,1024]`。U=RMS(H); H←H+Wdown(SiLU(Wgate U) ⊙ Wup U)；gate/up:1024→3072，down:3072→1024；均无bias。|[实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L62-L95)|
|Final RMS, last token, caller L2|`[B,L,1024]→[B,1024]`。Z=RMS(H); e=Z[:,-1,:]; e←e/||e||₂。AutoModel返回hidden states，不运行词表LMhead；不使用mean pooling。|[实现](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/semantics.py:52)|

```python
H = token(left_padded_ids)
for layer in decoder_layers_28:
    U = rms1(H)
    Q = rope(q_norm(split16(Wq(U)))) # head_dim=128
    K = rope(k_norm(split8(Wk(U))))
    V = split8(Wv(U))
    A = softmax(Q @ repeat2(K).T / sqrt(128) + causal_pad_mask)
    H = H + Wo(concat(A @ repeat2(V)))
    U = rms2(H)
    H = H + Wdown(silu(Wgate(U)) * Wup(U))
return normalize(final_rms(H)[:, -1], dim=-1)
```

不要用hidden_size/heads=64覆盖checkpoint显式head_dim128；Q总宽度确实大于residual hidden。
QK-RMS发生在head_dim轴、RoPE之前；V不做QK-RMS。
逐层预RMS与最终RMS都是1024维，Q/K专用RMS是128维。

冻结调用边界：[AlphaMotion semantics.py](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/semantics.py:36) 的 SigLIP/Qwen 都 `.eval()` 并 `no_grad`，新计算向量在统一cache写入前L2归一化；[UniMate T5Conditioner](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/text_encoder/t5.py#L225-L239) 在 `finetune=False` 时eval并排除于模型checkpoint。T5配置dropout=0.1，但这条冻结eval计算路径不随机丢弃。

这些文本塔提供语义特征，不等于已经学到局部表面区域的指认。AlphaMotion-2 的region token→surface membership映射及其学习仍属于新模型设计。
