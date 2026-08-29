#!/usr/bin/env bash
# V7 duel: conditional JEPA vs per-frame masked FSQ, matched at 62.69 bit/s, no OAT.
# Both arms launch together, the verdict runs when both have finished.
set -u
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
EXP='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
R=/var/tmp/alphamotion/topo_exp/runs
STEPS=${STEPS:-60000}
cd "$EXP" || exit 1

$PY -W ignore train_v7.py --arm jepa    --out $R/V7_jepa_0807    --steps $STEPS \
    > $R/V7_jepa_0807.log 2>&1 &
J=$!
sleep 90                      # stagger the two KeyData loads (16 bodies each)
$PY -W ignore train_v7.py --arm maskfsq --out $R/V7_maskfsq_0807 --steps $STEPS \
    > $R/V7_maskfsq_0807.log 2>&1 &
M=$!
echo "launched jepa=$J maskfsq=$M  steps=$STEPS  $(date)"
wait $J; echo "jepa exit=$? $(date)"
wait $M; echo "maskfsq exit=$? $(date)"

$PY -W ignore v7_eval.py --runs V7_jepa_0807,V7_maskfsq_0807 \
    --extra V6_mae_resid_0807,V5c_0806 --out $R/V7_verdict_0807.txt \
    > $R/V7_eval.log 2>&1
echo "verdict exit=$? -> $R/V7_verdict_0807.txt  $(date)"

$PY -W ignore v7_render.py --runs V7_jepa_0807,V7_maskfsq_0807 --n_each 2 \
    > $R/V7_render.log 2>&1
echo "render exit=$? -> R4_VIZ/mp4/v7/  $(date)"
