#!/bin/bash
# Parity + strictness evaluation of all captured (clip, robot) problems (sequential, each with a timeout).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
PY=$SR_UMR_PYTHON
for c in lafan1_smplx__dance1_subject2 hiphi_getup__getup0004_neutral_30fps case3_kick_right__20260330000586_1_1775269234; do
  for r in unitree_g1_29dof fourier_gr3 unitree_h2 agibot_a3_t3d0; do
    timeout 3600 $PY surface_retargeting/qa/run_parity.py $c $r > $SR_WORK/logs/parity_${c}__${r}.log 2>&1
    echo "$c $r rc=$?"
  done
done
