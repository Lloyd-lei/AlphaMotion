#!/bin/bash
# 两起点择优条件 best_axis/best_full: 走路集 (out) + manipulation 集 (out_manip), 18 台, 8 进程
cd ~/tmp/refiner_suite
ROBOTS="booster_k1 booster_t1 booster_t1_29dof fourier_gr3 fourier_n1 hightorque_hi kuavo_s45 pnd_adam_lite tienkung unitree_g1_23dof unitree_g1_29dof unitree_h1 unitree_h1_2 unitree_h2 unitree_r1 agibot_x2_ultra agibot_a3_t3d0 agibot_a3_ultra_t3d0"
echo "[run_best] start $(date)"
for O in out_manip out; do
  printf "%s\n" $ROBOTS | xargs -P 8 -I{} bash -c "OUT=$O OMP_NUM_THREADS=2 python3 eval_extra.py {} best_axis,best_full > $O/{}_best.log 2>&1; echo \"[run_best] $O {} rc=\$? \$(date)\""
  python3 aggregate.py $O > $O/aggregate.log 2>&1
done
echo "[run_best] ALL DONE $(date)"
