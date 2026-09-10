#!/bin/bash
# 0903 Chinchilla approach-1 阶梯队列: 等 S(8k)/S0 两条老 ladder 退出 -> 换上 ladder_v2.sh -> S20k -> M20k -> L20k 串行 (一次一个 AMASS 进程)
cd '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
R=/var/tmp/alphamotion/topo_exp/runs
WAIT="${WAIT_PIDS:-2004222 2004313}"
echo "[queue] waiting for $WAIT  $(date)"
for p in $WAIT; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
echo "[queue] old ladders gone $(date)"
[ -f ladder_v2.sh ] && mv ladder_v2.sh ladder.sh
export LADDER_STEPS1=20000 LADDER_STEPS2=8000
for spec in "L_S20k_0903 128 4 3 3" "L_M20k_0903 256 8 4 4" "L_L20k_0903 384 12 6 6"; do
  set -- $spec; n=$1
  echo "[queue] start $n $(date)"; free -g | sed -n 2p
  bash ladder.sh $spec > $R/$n.ladder.out 2>&1
  echo "[queue] end $n rc=$? $(date)"; tail -2 $R/$n.ladder.out
done
echo "[queue] ALL DONE $(date)"
