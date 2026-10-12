#!/bin/bash
# Render all clips of one robot (waits for each parity series).  Usage: render_robot.sh <robot> [skip_clip]
r=$1; skip=$2
cd /home/arenalabs/umr_gpu
for c in lafan1_smplx__dance1_subject2 hiphi_getup__getup0004_neutral_30fps case3_kick_right__20260330000586_1_1775269234; do
  [ "$c" = "$skip" ] && continue
  until [ -f /var/tmp/alphamotion/umr_factory/umr_gpu/results/parity/${c}__$r.series.npz ]; do sleep 30; done
  nice -n 10 timeout 5400 /var/tmp/alphamotion/umr_factory/umr/env/bin/python eval/render_compare.py $c $r > /var/tmp/alphamotion/umr_factory/umr_gpu/logs/render_${c}__$r.log 2>&1
  echo "$c $r rc=$?"
done
