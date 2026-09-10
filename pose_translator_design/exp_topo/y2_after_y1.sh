#!/bin/bash
# Y2 = Y1 + 根高槽位 (hroot, 目标=gap_root)。等 Y1 整链(含视频)结束再起, 不抢卡。
set -o pipefail
cd "$(dirname "$0")"
PY=/home/arenalabs/miniconda3/envs/rot/bin/python
RUNS=/var/tmp/alphamotion/topo_exp/runs
while pgrep -f "Y1_singletruth_0901" >/dev/null; do sleep 30; done
O=$RUNS/Y2_roothead_0901; mkdir -p $O
$PY train_dual.py \
  --cache /var/tmp/alphamotion/topo_exp/cache_lafan_v3 \
  --steps 8000 --bs 256 --lr 3e-4 --n_latent 128 --fsq_stages 1 \
  --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8 --mask_prob 0.3 --w_pos 5.0 \
  --w_contrast 0.5 --tau 0.1 --holdout "" \
  --combo_sample 8 --contrast_sample 6 --use_dof --judge_mode grad --w_judge 0.1 \
  --eval_every 1000 --seed 0 --d_model 128 --n_heads 4 --enc_layers 3 --dec_layers 3 \
  --rot_only --fk_loss --w_pen 200 --w_cgeo 1.0 \
  --root_head --w_root 1.0 --contact_chan --con_dropout 0.3 \
  --mask_mix rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10 \
  --out $O 2>&1 | tee $O.log | grep -E "params|c_rooth" | tail -2 || { echo "Y2 TRAIN FAIL"; exit 1; }
for W in FIXED GROUND; do
  $PY v2_judge.py --ckpt $O --tag Y2 --wset $W --append $RUNS/PXV2_results.json 2>&1 | tail -1
done
$PY contact_flip_test.py --ckpt $O --append $RUNS/Y2_flip_0901.json 2>&1 | tail -3
for T in unitree_g1_29dof pnd_adam_lite fourier_gr3; do
  $PY p3_contact_viz.py $T GROUND /home/arenalabs/tmp/y2_contact \
    Y1=Y1_singletruth_0901 Y2=Y2_roothead_0901 2>&1 | grep -E "contact\]|WROTE"
done
echo "Y2 CHAIN DONE"
