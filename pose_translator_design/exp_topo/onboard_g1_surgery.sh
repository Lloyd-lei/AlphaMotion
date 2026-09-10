#!/bin/bash
# G1 接入手术 (0831): 冻结码本, 只训解码器, 把 G1 纳入训练具身 + 开 EE-FK 加权。
# 依据: 0831 诊断 —— 腿直=G1 是解码器 holdout 的零样本税(训练内 +4.3° vs holdout -17~-20°);
#       手不准=病 A(信息在码里, 85% 梯度在训被丢弃的 hpos 头, fk_loss 拉回 50/50)。
# 两病同一部位 => 一次手术同时治。码本不动 => 所有历史码逐位兼容(0812 前科: 往返 11%->61%)。
# 这也是数据平台 "新机器人 onboarding" 流程的第一次实弹演练。
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
RUNS=/var/tmp/alphamotion/topo_exp/runs
PROD=$RUNS/PT_c1dec_reenc_0812          # 生产 18M, 其 holdout 含 unitree_g1_29dof
EE_W=${EE_W:-1.0}                        # 由 P2 两臂(1.0/0.5)判决后填入胜者
OUT=$RUNS/ONB_g1_eefk_0831

$PY train_dual.py \
  --cache /var/tmp/alphamotion/topo_exp/cache_xwbc_lafan \
  --init $PROD/model.pt --train_scope decoder \
  --steps 20000 --bs 256 --lr 1e-4 \
  --n_latent 128 --fsq_stages 1 --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8 \
  --d_model 256 --n_heads 4 --enc_layers 3 --dec_layers 4 \
  --mask_prob 0.3 --mask_mix rand:0.3,ee:0.25,root:0.15,sparse:0.3 \
  --w_pos 5.0 --w_contrast 0.5 --tau 0.1 \
  --reenc_consistency 1.0 --reenc_prob 0.5 \
  --holdout unitree_g1_23dof,pal_talos \
  --fk_loss --ee_fk_w $EE_W \
  --combo_sample 8 --contrast_sample 6 --use_dof --judge_mode grad --w_judge 0.1 \
  --eval_every 1000 --seed 0 --out $OUT 2>&1 | tee $OUT.log | grep -E "params|trainable|EVAL|DONE" | tail -4

# 判官 + 手部/腿部厘米补充表 + 码不变性证明(冻结码本必须逐位相同)
$PY v2_judge.py --ckpt $OUT --tag onb_g1_eefk --append $RUNS/P0V2_results.json
$PY duel_decoder_pt.py --base PT_c1dec_reenc_0812 --pt $(basename $OUT) 2>&1 | tail -6
echo "手术完成 -> $OUT"
