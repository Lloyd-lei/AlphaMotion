#!/bin/bash
# After each new parity run (results newer than $SINCE): continuity plot + full-length [human | GPU UMR raw | GPU strict]
# video + 12-frame grid + a 4x slow-motion window around the frame where the OLD frame-parallel export jumped most.
# Usage: render_fix.sh <since-epoch> "clip robot" ...
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
PY=$SR_UMR_PYTHON
OUT=$SR_WORK
V=$OUT/videos/fix_jump
SINCE=$1; shift
for cr in "$@"; do
  set -- $cr; C=$1; R=$2; J=$OUT/results/parity/${C}__${R}.json
  until [ -f $J ] && [ $(stat -c %Y $J) -gt $SINCE ]; do sleep 30; done
  sleep 20
  python3 surface_retargeting/qa/plot_continuity.py $V ${C}__${R} > $V/continuity_${C}__${R}.txt 2>&1
  mv $V/continuity_summary.json $V/continuity_${C}__${R}.json 2>/dev/null
  W=$($PY - <<PYEOF
import numpy as np, os
f = '$OUT/results/export_old_frameparallel/${C}__${R}.npz'
if not os.path.exists(f):
    f = '$OUT/results/export_old_frameparallel/${C}__unitree_g1_29dof.npz'
if '${C}' == 'lafan1_smplx__dance1_subject2' and '${R}' == 'unitree_g1_29dof':
    print('2190:2290'); raise SystemExit
z = np.load(f); q = z['qpos'].astype(float); fps = float(z['fps'])
st = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
t = int(np.argmax(st)) + 1; h = int(round(50 * fps / 30))
print(f'{max(0, t - h)}:{min(len(q), t + h)}')
PYEOF
)
  OMP_NUM_THREADS=2 nice -n 10 taskset -c 0-15 timeout 3600 $PY surface_retargeting/qa/render_compare.py $C $R --cols umr --outdir $V > $V/render_${C}__${R}.log 2>&1
  OMP_NUM_THREADS=2 nice -n 10 taskset -c 0-15 timeout 3600 $PY surface_retargeting/qa/render_compare.py $C $R --cols umr --frames $W --slow 4 --tag win${W/:/-}_slow4x --outdir $V >> $V/render_${C}__${R}.log 2>&1
  if [ "$C" = lafan1_smplx__dance1_subject2 ] && [ "$R" = unitree_g1_29dof ]; then
    OMP_NUM_THREADS=2 nice -n 10 taskset -c 0-15 timeout 3600 $PY surface_retargeting/qa/render_compare.py $C $R --cols umr --frames 270:420 --slow 4 --tag win270-420_slow4x --outdir $V >> $V/render_${C}__${R}.log 2>&1
  fi
  echo "$(date +%T) rendered $C $R window $W"
done
