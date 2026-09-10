#!/bin/bash
# Greenwich-v2 P1: 方差归一化消融 (0830 深夜, 接 P0 判决)。
# P0 判决: fk_loss 小胜保留(上肢 6.39 最优), var_floor@0.5 判负(腿角撞线)。
# P1 单变量: 在 fk 配方上加 --var_norm。对照 = P0 的 fk 臂(已训好, 直接引用其判官行)。
# 判据(先立): 上肢 < 6.39cm 且 腿角(警戒) 不高于 17.51°+1° ; 方言逐维一致率不降。
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
RUNS=/var/tmp/alphamotion/topo_exp/runs
STAMP=0830
COMMON=(
  --cache /var/tmp/alphamotion/topo_exp/cache_xwbc_lafan
  --steps 8000 --bs 256 --lr 3e-4
  --n_latent 128 --fsq_stages 1 --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8
  --mask_prob 0.3 --mask_mix rand:0.3,ee:0.25,root:0.15,sparse:0.3
  --w_pos 5.0 --w_contrast 0.5 --tau 0.1
  --holdout unitree_g1_29dof,unitree_g1_23dof
  --combo_sample 8 --contrast_sample 6
  --use_dof --judge_mode grad --w_judge 0.1
  --eval_every 1000 --seed 0
  --d_model 128 --n_heads 4 --enc_layers 3 --dec_layers 3
)
run_arm () {
  local name=$1; shift
  local out=$RUNS/P1V2_${name}_${STAMP}
  if [ -f "$out/model.pt" ]; then echo "== $name 已有, 跳过"; else
    echo "== $name: train_dual $*"
    $PY train_dual.py "${COMMON[@]}" "$@" --out "$out" 2>&1 | tee "$out.log" | grep -E "params|EVAL|DONE" | tail -3
  fi
  $PY v2_judge.py --ckpt "$out" --tag "$name" --append "$RUNS/P0V2_results.json" \
      2>&1 | tee -a "$RUNS/P1V2_judge_${STAMP}.log"
}
run_arm fkvn --fk_loss --var_norm
# 顺手补一臂: slip 副批单独试跑(P1.5 预览, 单变量: 在 fk 上只开 slip)
run_arm fksp --fk_loss --slip_pairs --slip_w 1.0
echo "P1 完成 -> $RUNS/P0V2_results.json"
