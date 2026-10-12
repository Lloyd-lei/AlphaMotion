#!/bin/bash
# Capture the 6 parity problems (CPU UMR stock defaults) -- sequential, each with a timeout.
PY=/var/tmp/alphamotion/umr_factory/umr/env/bin/python
U=/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data
cd /home/arenalabs/umr_gpu/capture
for r in ${ROBOTS:-unitree_g1_29dof fourier_gr3 unitree_h2}; do
  $PY run_capture.py --robot $r --kind smplx --data $U/lafan1_smplx --seq dance1_subject2 --timeout 2400
  $PY run_capture.py --robot $r --kind hoi --data $U/omnicontact/soccer/case3_kick_right --seq 20260330000586_1_1775269234 --timeout 2400
done
