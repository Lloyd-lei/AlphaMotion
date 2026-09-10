#!/bin/bash
# X 系列 (lloyd 0831 盘点后重开): 拆干净的三条臂 + 距离编码。
# 病史: 旧 chan/bal 臂把"通道"叠在 cmaint 上测, cmaint 的几何拉扯(幅度塌 0.78-0.89)
# 淹没了通道本身; 且旧概率编码 89% 恒 0, 通道嵌入梯度死。两处都换掉:
#   输入通道 = 对数压缩符号距离 (ContactStore.sdist), 考试也喂 (v2_judge 已改)
#   监督端   = 置信度 soft (cbce 目标), 与输入分离
#   X1  base + 通道 + hcon 头            -> lloyd 的设计单独成不成立
#   X2  X1 + pen                          -> 加上已证明有效的一项
#   X3  base + cmaint                     -> 单独确认幅度塌是不是 cmaint 干的
set -o pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
RUNS=/var/tmp/alphamotion/topo_exp/runs
RES=$RUNS/PXV2_results.json
COMMON=(--cache /var/tmp/alphamotion/topo_exp/cache_lafan_v3
  --steps 8000 --bs 256 --lr 3e-4 --n_latent 128 --fsq_stages 1
  --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8 --mask_prob 0.3 --w_pos 5.0
  --w_contrast 0.5 --tau 0.1 --holdout unitree_g1_29dof,unitree_g1_23dof
  --combo_sample 8 --contrast_sample 6 --use_dof --judge_mode grad --w_judge 0.1
  --eval_every 1000 --seed 0 --d_model 128 --n_heads 4 --enc_layers 3 --dec_layers 3)
CMIX=rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10
run(){ n=$1; shift; o=$RUNS/PXV2_${n}_0831
  if [ -f $o.log ] && [ ! -f $o/model.pt ]; then
    echo "== $n: 已有并行进程在训, 等它写出 model.pt"
    while [ ! -f $o/model.pt ] && pgrep -f "out $o" >/dev/null; do sleep 20; done
  fi
  if [ ! -f $o/model.pt ]; then
    echo "== $n: $*"
    $PY train_dual.py "${COMMON[@]}" "$@" --out $o 2>&1 | tee $o.log | grep -E "params|EVAL" | tail -2 || { echo "== $n TRAIN FAIL"; return 1; }
  fi
  for W in FIXED GROUND; do
    $PY v2_judge.py --ckpt $o --tag $n --wset $W --append $RES 2>&1 | tail -1
  done
  if grep -q '"contact_chan": true' $o/config.json; then
    $PY contact_flip_test.py --ckpt $o --append $RUNS/PXV2_flip_0831.json 2>&1 | tail -3
  fi
}
run X1 --contact_chan --w_cbce 0.1 --con_dropout 0.3 --mask_mix "$CMIX"
run X2 --contact_chan --w_cbce 0.1 --con_dropout 0.3 --mask_mix "$CMIX" --w_pen 200
run X3 --w_pen 0 --w_contact 5
echo "PX DONE -> $RES"
