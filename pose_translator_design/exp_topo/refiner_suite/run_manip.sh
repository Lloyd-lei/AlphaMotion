#!/bin/bash
# manipulation 片段 (GRAB/KIT/HumanEva, 85 段抽 40) 18 台机器人, 8 进程; 主条件 + ref0 补条件
cd ~/tmp/refiner_suite
export AM_SET=~/tmp/refiner_suite/manip_set.json OUT=out_manip
ROBOTS="booster_k1 booster_t1 booster_t1_29dof fourier_gr3 fourier_n1 hightorque_hi kuavo_s45 pnd_adam_lite tienkung unitree_g1_23dof unitree_g1_29dof unitree_h1 unitree_h1_2 unitree_h2 unitree_r1 agibot_x2_ultra agibot_a3_t3d0 agibot_a3_ultra_t3d0"
echo "[run_manip] start $(date)"
printf "%s\n" $ROBOTS | xargs -P 8 -I{} bash -c 'OMP_NUM_THREADS=2 python3 eval_suite.py {} out_manip/{}.json > out_manip/{}.log 2>&1 && OMP_NUM_THREADS=2 python3 eval_extra.py {} feet_axis_ref0,feet_full_ref0 > out_manip/{}_extra.log 2>&1; echo "[run_manip] {} rc=$? $(date)"'
python3 aggregate.py out_manip > out_manip/aggregate.log 2>&1
echo "[run_manip] ALL DONE $(date)"
