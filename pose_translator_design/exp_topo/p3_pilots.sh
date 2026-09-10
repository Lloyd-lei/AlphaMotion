#!/bin/bash
# Greenwich-v2 P3 接触消融: 4 条串行小臂 (spec §7/§8, 0831 定)。
#
#   bash p3_pilots.sh          # GPU 空出来再跑 (标注还在占卡时别起); 约 4 x E24_small 钟点
#
# 配方 = p0_pilots.sh 的 COMMON 逐字照抄 (E24_small_0805 的 3.67M), 所以 P0 的 base 臂
# (runs/P0V2_base_0830) 直接就是本轮的对照行, 不必重训。接触层把参数从 3.6715M 抬到
# 3.6721M (+641, +0.018%), spec 的 +2% 预算完全够。
#
# 阶梯 (每级只加一样东西, 七机制抢梯度的警告仍然有效):
#   pen   只加"不许穿模"的不等式 —— 无输入通道、无接触头。§8.1: 这是唯一能超过数据
#         质量的机制 (源数据自穿 2.5-4.3cm), 所以它单独成一级, 也永远开着。
#   cons  再加"该碰必须碰到"+"碰着的不许滑"。到这一级为止全是损失, 码本结构没动,
#         老 checkpoint 仍然逐位可加载。
#   chan  再加第 10 通道 + 接触头 + cask/cgive 掩码族 —— 破坏性的一级 (结构变了)。
#   bal   再加接触构型均衡采样。
#
# 权重量级的依据 (0831 在 LAFAN GT 上实测, 不是拍的):
#   GT 穿地 mean 0.01-0.07cm / p95 0.00cm  -> pen 的目标就是 0, relu(-h)^2 单位是 m^2,
#     10cm 穿透 = 0.01, 几个关节摊到全身均值 ~1e-3 量级 => w_pen 200 让它贡献 ~0.2
#   GT 接触点 |h| mean 1.06-2.14cm         -> cmaint 的数据地板是 0.01-0.02m,
#     w_contact 5 让它贡献 ~0.1, 与 w_pos*pos (~0.25) 同量级而不压过它
#   w_slip_c 沿用 slip_w 的既用值 0.2; w_cbce 沿用 w_judge 的 0.1
# 头 200 步务必看 EVAL 里的 c_pen/c_cmaint/c_cbce 三行, 哪一项压过了 rot/pos 就调它。
set -e
# 0831: 必须加 pipefail。四条臂的训练与判官调用全是 `... | tee | grep | tail` 管道,
# bash 取最后一段的退出码, 于是训练 OOM、判官断言失败都不会触发 set -e, 脚本照样
# 打印"全部完成"并留下一个空的结果文件。
set -o pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
RUNS=/var/tmp/alphamotion/topo_exp/runs
STAMP=0831
RES=$RUNS/P3V2_results.json

# ---- 与 p0_pilots.sh 完全相同的公共配方 (勿改单项) ----
COMMON=(
  --cache /var/tmp/alphamotion/topo_exp/cache_lafan_v3
  --steps 8000 --bs 256 --lr 3e-4
  --n_latent 128 --fsq_stages 1 --fsq_levels 8,8,8,8,8,8,8,8,8,8,8,8,8
  --mask_prob 0.3 --mask_mix rand:0.3,ee:0.25,root:0.15,sparse:0.3
  --w_pos 5.0 --w_contrast 0.5 --tau 0.1
  --holdout unitree_g1_29dof,unitree_g1_23dof   # pal_talos 已从语料整体剔除(lloyd 0831: URDF 有问题),不再作为 holdout
  --combo_sample 8 --contrast_sample 6
  --use_dof --judge_mode grad --w_judge 0.1
  --eval_every 1000 --seed 0
  --d_model 128 --n_heads 4 --enc_layers 3 --dec_layers 3
)
# 接触族掩码: cask 占比高于 cgive, 因为 lloyd 定 cask 为主方案 —— 坐标是具身相关的,
# "踩没踩"这个标签是具身无关的, 所以 cask 给的约束能跨具身直接搬运。原有四族按比例
# 缩到 0.75, 产品面 (稀疏关节约束) 不丢。
CMIX=rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10

run_arm () {           # $1 = 臂名, 其余 = 该臂独有的开关
  local name=$1; shift
  local out=$RUNS/P3V2_${name}_${STAMP}
  mkdir -p "$out"
  if [ -f "$out/model.pt" ]; then
    echo "== $name 已有 model.pt, 跳过训练"
  else
    echo "== $name: train_dual $*"
    $PY train_dual.py "${COMMON[@]}" "$@" --out "$out" 2>&1 | tee "$out.log" | grep -E "params|contact|EVAL|DONE" | tail -4
  fi
  # 两份卷子都考: FIXED 是与 P0 可比的老尺子, GROUND 才是接触活儿的真考场
  # (FIXED 的片段是 aiming/dance/fight/sprint/walk, 地面片段为零 = 盲测)
  $PY v2_judge.py --ckpt "$out" --tag "$name" --append "$RES" \
      2>&1 | tee -a "$RUNS/P3V2_judge_${STAMP}.log"
  $PY v2_judge.py --ckpt "$out" --tag "$name" --wset GROUND --append "$RES" \
      2>&1 | tee -a "$RUNS/P3V2_judge_${STAMP}.log"
  # 有第 10 通道的臂必须过聋子测试: FAIL 就别再解读它的任何接触数字
  if grep -q '"contact_chan": true' "$out/config.json"; then
    $PY contact_flip_test.py --ckpt "$out" --append "$RUNS/P3V2_flip_${STAMP}.json" \
        2>&1 | tee -a "$RUNS/P3V2_judge_${STAMP}.log"
  fi
}

run_arm pen  --w_pen 200
run_arm cons --w_pen 200 --w_contact 5 --slip_pairs --w_slip_c 0.2
run_arm chan --w_pen 200 --w_contact 5 --slip_pairs --w_slip_c 0.2 \
             --contact_chan --w_cbce 0.1 --mask_mix "$CMIX"
run_arm bal  --w_pen 200 --w_contact 5 --slip_pairs --w_slip_c 0.2 \
             --contact_chan --w_cbce 0.1 --mask_mix "$CMIX" \
             --contact_balance --contact_target 0.10

# 对照行: P0 的 base 臂 (接触层全关, 同一配方) 在两份卷子上各考一次
$PY v2_judge.py --ckpt P0V2_base_0830 --tag base --append "$RES" \
    2>&1 | tee -a "$RUNS/P3V2_judge_${STAMP}.log"
$PY v2_judge.py --ckpt P0V2_base_0830 --tag base --wset GROUND --append "$RES" \
    2>&1 | tee -a "$RUNS/P3V2_judge_${STAMP}.log"

echo "P3 全部完成 -> $RES"
echo "读表顺序 (别倒过来): (1) 先看 flip 判决, FAIL 则该臂的接触数字全部作废;"
echo "  (2) 再看 GROUND 卷子的 pen_p95_cm / skate_cms 有没有往验收线 (1cm / 15cm/s) 走;"
echo "  (3) 最后逐条对预注册护栏 leg_mean_deg / amp_all / ul_mean_cm / hand_mean_cm,"
echo "      任一项退化都要先解释再谈收益。数字看完必须渲一段地面动作视频再下结论。"
