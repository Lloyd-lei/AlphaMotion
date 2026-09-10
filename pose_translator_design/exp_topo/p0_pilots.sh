#!/bin/bash
# Greenwich-v2 P0 消融: 4 条串行小臂 + 每臂考一次 v2_judge (spec §6, 0830 定)。
#
#   bash p0_pilots.sh          # GPU 空出来(~05:00 后)再跑; 全程约 4 x E24_small 的钟点
#
# 配方 = E24_small_0805 的 3.67M 配置逐项照抄 (runs/E24_small_0805/config.json 是
# 唯一权威, 本脚本 0830 与它逐 key 核对过)。为什么用它: E 梯队里最便宜的完整配方,
# 参数刻度上 var_floor/fk_loss 的效应若在 3.67M 都看不见, 18M 上更不值得开。
# 每臂单变量: BASE 全关 / VF 只开 var_floor / FK 只开 fk_loss / VFFK 两个都开。
# var_floor 权重 0.5 = N3f_varfloor_0807 的既用值 (仓库里唯一开过它的 run)。
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
RUNS=/var/tmp/alphamotion/topo_exp/runs
STAMP=0830

# E24_small_0805 的公共配方 (勿改单项; 改了就不是和 3.67M 基线同一把尺子)
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

run_arm () {           # $1 = 臂名, 其余 = 该臂独有的开关
  local name=$1; shift
  local out=$RUNS/P0V2_${name}_${STAMP}
  if [ -f "$out/model.pt" ]; then
    echo "== $name 已有 model.pt, 跳过训练"
  else
    echo "== $name: train_dual $*"
    $PY train_dual.py "${COMMON[@]}" "$@" --out "$out" 2>&1 | tee "$out.log" | grep -E "params|EVAL|DONE" | tail -3
  fi
  # 每臂考完立刻记账: 同一份固定窗口卷子, 汇总进 P0V2_results.json (判官内部追加)
  $PY v2_judge.py --ckpt "$out" --tag "$name" --append "$RUNS/P0V2_results.json" \
      2>&1 | tee -a "$RUNS/P0V2_judge_${STAMP}.log"
}

run_arm base
run_arm vf   --var_floor 0.5
run_arm fk   --fk_loss
run_arm vffk --var_floor 0.5 --fk_loss

# 基线参考行: 生产 checkpoint 同卷考一次 (窗口/种子全同, 直接可比)
$PY v2_judge.py --ckpt PT_c1dec_reenc_0812 --tag prod_reenc \
    --append "$RUNS/P0V2_results.json" 2>&1 | tee -a "$RUNS/P0V2_judge_${STAMP}.log"

echo "P0 全部完成 -> $RUNS/P0V2_results.json"
