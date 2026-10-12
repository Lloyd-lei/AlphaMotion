#!/bin/bash
# Capture the 6 parity problems (CPU UMR stock defaults) -- sequential, each with a timeout.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=$SR_UMR_PYTHON
U=$SR_UMR_ROOT/sample_data
cd "$REPO/scripts/capture"
for r in ${ROBOTS:-unitree_g1_29dof fourier_gr3 unitree_h2}; do
  $PY run_capture.py --robot $r --kind smplx --data $U/lafan1_smplx --seq dance1_subject2 --timeout 2400
  $PY run_capture.py --robot $r --kind hoi --data $U/omnicontact/soccer/case3_kick_right --seq 20260330000586_1_1775269234 --timeout 2400
done
