#!/bin/bash
# manipulation 片段: 18 台 × {拿水瓶 GRAB s1 waterbottle_pick_all, 喝水 KIT 441 drinking04}, 4 台重点 × {擦桌 KIT 674 wipe_front01 (T250), 搬箱 HumanEva S2 Box_3}
cd ~/tmp/refiner_suite
ROBOTS="booster_k1 booster_t1 booster_t1_29dof fourier_gr3 fourier_n1 hightorque_hi kuavo_s45 pnd_adam_lite tienkung unitree_g1_23dof unitree_g1_29dof unitree_h1 unitree_h1_2 unitree_h2 unitree_r1 agibot_x2_ultra agibot_a3_t3d0 agibot_a3_ultra_t3d0"
CONDS=zero_ik_full,hands,feet_axis,feet_axis_ref0
echo "[render_manip] start $(date)"
{ for r in $ROBOTS; do echo "$r bottle GRAB__s1__waterbottle_pick_all 300"; echo "$r drink KIT__441__drinking04 300"; done
  for r in unitree_g1_29dof agibot_x2_ultra agibot_a3_t3d0 fourier_gr3; do echo "$r wipe KIT__674__wipe_front01 250"; echo "$r box HumanEva__S2__Box_3 300"; done; } | \
  xargs -P 6 -L 1 bash -c 'r=$0; tag=$1; clip=$2; T=$3; OMP_NUM_THREADS=2 python render_rollout.py $r "$clip" vid_manip --T $T --conds '$CONDS' --tag $tag > vid_manip/${r}_${tag}.log 2>&1; echo "[render_manip] $r $tag rc=$? $(date)"'
echo "[render_manip] ALL DONE $(date)"
