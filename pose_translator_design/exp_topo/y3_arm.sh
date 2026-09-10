#!/bin/bash
set -o pipefail
cd "$(dirname "$0")"
PY=/home/arenalabs/miniconda3/envs/rot/bin/python
RUNS=/var/tmp/alphamotion/topo_exp/runs
O=$RUNS/Y3_rootslot_0901; mkdir -p $O
$PY train_dual.py \
  --cache /var/tmp/alphamotion/topo_exp/cache_lafan_v3 \
  --steps 8000 --bs 256 --lr 3e-4 --n_latent 128 --n_latent_p 4 --fsq_stages 1 \
  --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8 --mask_prob 0.3 --w_pos 5.0 \
  --w_contrast 0.5 --tau 0.1 --holdout "" \
  --combo_sample 8 --contrast_sample 6 --use_dof --judge_mode grad --w_judge 0.1 \
  --eval_every 1000 --seed 0 --d_model 128 --n_heads 4 --enc_layers 3 --dec_layers 3 \
  --fk_loss --w_pen 200 --w_cgeo 1.0 --root_pos --w_rootpos 1.0 \
  --contact_chan --con_dropout 0.3 \
  --mask_mix rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10 \
  --out $O 2>&1 | tee $O.log | grep -E "params|c_rootpos" | tail -2 || { echo "Y3 TRAIN FAIL"; exit 1; }
for W in FIXED GROUND; do
  $PY v2_judge.py --ckpt $O --tag Y3 --wset $W --append $RUNS/PXV2_results.json 2>&1 | tail -1
done
$PY contact_flip_test.py --ckpt $O --append $RUNS/Y3_flip_0901.json 2>&1 | tail -2
# 视频: 模型own位移 (AM_ODO=pred) 的走路/跳
AM_ODO=pred $PY crossrig_amass.py Y3_rootslot_0901 /home/arenalabs/tmp/y3_odo \
  'HumanEva__S2__Walking_1' 'Male1Walking_c3d__Walk_B16' 'MoSh__50007__jumping_jacks' 2>&1 | grep -E "WROTE"
echo "Y3 CHAIN DONE"
