#!/bin/bash
# 18 台机器人, 8 进程并行, 每台一个 json
cd ~/tmp/refiner_suite
ROBOTS="booster_k1 booster_t1 booster_t1_29dof fourier_gr3 fourier_n1 hightorque_hi kuavo_s45 pnd_adam_lite tienkung unitree_g1_23dof unitree_g1_29dof unitree_h1 unitree_h1_2 unitree_h2 unitree_r1 agibot_x2_ultra agibot_a3_t3d0 agibot_a3_ultra_t3d0"
echo "[run_all] start $(date)"
printf "%s\n" $ROBOTS | xargs -P 8 -I{} bash -c 'python3 eval_suite.py {} out/{}.json > out/{}.log 2>&1; echo "[run_all] {} rc=$? $(date)"'
echo "[run_all] ALL DONE $(date)"
