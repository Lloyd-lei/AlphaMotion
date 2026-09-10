#!/bin/bash
cd ~/tmp/refiner_suite
ROBOTS="booster_k1 booster_t1 booster_t1_29dof fourier_gr3 fourier_n1 hightorque_hi kuavo_s45 pnd_adam_lite tienkung unitree_g1_23dof unitree_g1_29dof unitree_h1 unitree_h1_2 unitree_h2 unitree_r1 agibot_x2_ultra agibot_a3_t3d0 agibot_a3_ultra_t3d0"
echo "[run_extra] start $(date)"
for r in $ROBOTS; do echo $r; done | xargs -P 8 -I{} bash -c 'OMP_NUM_THREADS=2 python eval_extra.py {} feet_axis_ref0,feet_full_ref0 > out/{}_extra.log 2>&1; echo "[run_extra] {} rc=$? $(date)"'
echo "[run_extra] ALL DONE $(date)"
