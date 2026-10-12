#!/bin/bash
# Render all clips of one robot (waits for each parity series).  Usage: render_robot.sh <robot> [skip_clip]
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
r=$1; skip=$2
cd "$REPO"
for c in lafan1_smplx__dance1_subject2 hiphi_getup__getup0004_neutral_30fps case3_kick_right__20260330000586_1_1775269234; do
  [ "$c" = "$skip" ] && continue
  until [ -f $SR_WORK/results/parity/${c}__$r.series.npz ]; do sleep 30; done
  nice -n 10 timeout 5400 $SR_UMR_PYTHON surface_retargeting/qa/render_compare.py $c $r > $SR_WORK/logs/render_${c}__$r.log 2>&1
  echo "$c $r rc=$?"
done
