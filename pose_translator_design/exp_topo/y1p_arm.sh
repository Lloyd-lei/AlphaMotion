#!/bin/bash
# Y1 (0901 lloyd 定): 单真相源配方 —— 码本全给旋转, 其它全是读数。
#   rot_only   码位全存旋转 (0812 已证: 位置误差 11.65->6.73)
#   fk_loss    位置 = FK(旋转), 误差梯度回改旋转
#   w_pen      罚穿地 (P3 四臂唯一赢家)
#   w_cgeo     接触 = sigmoid((thr - FK脚底高)/tau) 的 BCE, 无独立接触头
#   contact_chan + 距离编码通道, cask/cgive 互猜
#   holdout 空 = g1 入训 (lloyd: 不再交零样本税)
set -o pipefail
cd "$(dirname "$0")"
PY=/home/arenalabs/miniconda3/envs/rot/bin/python
RUNS=/var/tmp/alphamotion/topo_exp/runs
O=$RUNS/Y1p_sentinel_0901
mkdir -p $O
$PY train_dual.py \
  --cache /var/tmp/alphamotion/topo_exp/cache_lafan_v3 \
  --steps 8000 --bs 256 --lr 3e-4 --n_latent 128 --fsq_stages 1 \
  --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8 --mask_prob 0.3 --w_pos 5.0 \
  --w_contrast 0.5 --tau 0.1 --holdout "" \
  --combo_sample 8 --contrast_sample 6 --use_dof --judge_mode grad --w_judge 0.1 \
  --eval_every 1000 --seed 0 --d_model 128 --n_heads 4 --enc_layers 3 --dec_layers 3 \
  --rot_only --fk_loss --w_pen 200 --w_cgeo 1.0 \
  --contact_chan --con_dropout 0.3 \
  --mask_mix rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10 \
  --out $O 2>&1 | tee $O.log | grep -E "params|c_cgeo" | tail -2 || { echo "Y1 TRAIN FAIL"; exit 1; }
for W in FIXED GROUND; do
  $PY v2_judge.py --ckpt $O --tag Y1p --wset $W --append $RUNS/PXV2_results.json 2>&1 | tail -1
done
$PY contact_flip_test.py --ckpt $O --append $RUNS/Y1_flip_0901.json 2>&1 | tail -3
for T in unitree_g1_29dof pnd_adam_lite fourier_gr3; do
  $PY p3_contact_viz.py $T GROUND /home/arenalabs/tmp/y1p_contact \
    Greenwich=E24_small_0805 pen=P3V2_pen_0831 Y1=Y1p_sentinel_0901 2>&1 | grep -E "contact\]|WROTE"
done
echo "Y1 CHAIN DONE"
