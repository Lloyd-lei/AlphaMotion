#!/bin/bash
# Greenwich-v2 P2 (0831): 末端-杠杆位置权重 --ee_fk_w 的单变量消融。
#
# 立论 (探针 B 实测, 见 ../HAND_DIAGNOSIS.md): 码里有手的信息 —— 生产 ckpt 上, 从
# 冻结码读出的 MLP 探针把 g1 手部做到 2.85cm, 解码器同一批帧只做到 7.32cm; 而
# 码/原始输入 的信息损失比在 手/上肢/腿 三组几乎一样 (1.9-2.0x), 没有"手臂专属"的
# 瓶颈损失 => 病 A (解码器没交付), 不是病 B (码里没有)。
# 对症下药: pos_j 是 L2 范数再对关节取平均, 每个关节拿到的位置梯度完全相等 ——
# 半径 48cm 的手要达到和半径 10cm 的髋一样的 cm 误差却只买到同样多的梯度。
# --ee_fk_w 按杠杆臂 + 拓扑树叶重新分配这份预算 (无关键词表, 见 dual_common.ee_fk_weights)。
#
# 对照 = P0 的 fk 臂 (已训好, 判官行已在 P0V2_results.json), 单变量只多 --ee_fk_w。
# 判据 (先立): g1 上肢 < 6.39cm 且 gr3 上肢 < 3.67cm; 腿角 (警戒) 不高于 fk 的 17.51+1 度。
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
RUNS=/var/tmp/alphamotion/topo_exp/runs
STAMP=0831
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
  local out=$RUNS/P2V2_${name}_${STAMP}
  if [ -f "$out/model.pt" ]; then echo "== $name 已有 model.pt, 跳过训练"; else
    echo "== $name: train_dual $*"
    $PY train_dual.py "${COMMON[@]}" "$@" --out "$out" 2>&1 | tee "$out.log" | grep -E "params|EVAL|DONE" | tail -3
  fi
  $PY v2_judge.py --ckpt "$out" --tag "$name" --append "$RUNS/P0V2_results.json" \
      2>&1 | tee -a "$RUNS/P2V2_judge_${STAMP}.log"
  # 判官的 UL_KW 把真正的手排除在外 (g1 的 *_rubber_hand 被 FINGER_KW 过滤, gr3 的
  # *_end_effector_link 一个关键词都不匹配), 所以每臂再补一份"手/腿 厘米"补充表,
  # 同一批固定窗口, 拓扑自动认末端。判官那张主表口径一个字不改。
  $PY hand_probe_d.py --ckpt "$out" --tag "$name" 2>&1 | tee -a "$RUNS/P2V2_hand_${STAMP}.log"
}
run_arm eefk1  --fk_loss --ee_fk_w 1.0
run_arm eefk05 --fk_loss --ee_fk_w 0.5
echo "P2 完成 -> $RUNS/P0V2_results.json"
