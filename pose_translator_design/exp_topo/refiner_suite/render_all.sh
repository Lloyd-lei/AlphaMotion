#!/bin/bash
# 18 台机身 × 2 段语料外片段 (走路 / 踢腿), 4 并行 (每个渲染进程 OMP 2 线程)
cd ~/tmp/refiner_suite
ROBOTS="booster_k1 booster_t1 booster_t1_29dof fourier_gr3 fourier_n1 hightorque_hi kuavo_s45 pnd_adam_lite tienkung unitree_g1_23dof unitree_g1_29dof unitree_h1 unitree_h1_2 unitree_h2 unitree_r1 agibot_x2_ultra agibot_a3_t3d0 agibot_a3_ultra_t3d0"
CONDS=zero_ik_full,hands,feet_plane,feet_axis
echo "[render_all] start $(date)"
for r in $ROBOTS; do echo "$r walk hamada__walk-09-handbag"; echo "$r kick SOMA__soma_subject2__kick_001"; done | \
  xargs -P 6 -L 1 bash -c 'r=$0; tag=$1; clip=$2; OMP_NUM_THREADS=2 python render_rollout.py $r "$clip" vid --T 300 --conds '$CONDS' --tag $tag > vid/${r}_${tag}.log 2>&1; echo "[render_all] $r $tag rc=$? $(date)"'
echo "[render_all] ALL DONE $(date)"
