#!/bin/bash
# Render the comparison videos + 12-frame grids for every evaluated (clip, robot).
cd /home/arenalabs/umr_gpu
PY=/var/tmp/alphamotion/umr_factory/umr/env/bin/python
for c in ${CLIPS:-lafan1_smplx__dance1_subject2 hiphi_getup__getup0004_neutral_30fps case3_kick_right__20260330000586_1_1775269234}; do
  for r in unitree_g1_29dof fourier_gr3 unitree_h2 agibot_a3_t3d0; do
    [ -f /var/tmp/alphamotion/umr_factory/umr_gpu/results/parity/${c}__${r}.series.npz ] || continue
    nice -n 10 timeout 3600 $PY eval/render_compare.py $c $r > /var/tmp/alphamotion/umr_factory/umr_gpu/logs/render_${c}__${r}.log 2>&1
    echo "$c $r rc=$?"
  done
done
