#!/bin/bash
# Parity + export for a list of "clip robot" pairs (one queue; run two queues with UMRG_DUTY to stay < 70 % GPU util).
# Usage: parity_queue.sh <logdir> "clip robot" ["clip robot" ...]
# Env: PARITY_ARGS  extra run_parity.py args (e.g. "--outdir export_slip_v3 --judge strict": new results dir, CPU-light)
#      UMRG_DUTY    GPU duty cycle (default 0.6);  each run holds the exclusive gpu_heavy.lock (RESOURCE_LIMITS.md)
cd /home/arenalabs/umr_gpu
PY=/var/tmp/alphamotion/umr_factory/umr/env/bin/python
LOCK=/home/arenalabs/umr_factory_data/logs/gpu_heavy.lock
LOGD=$1; shift
mkdir -p $LOGD
for cr in "$@"; do
  set -- $cr
  while grep -q "net=down" <(tail -1 /home/arenalabs/umr_factory_data/logs/throttle.log 2>/dev/null); do sleep 30; done
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 UMRG_DUTY=${UMRG_DUTY:-0.6} timeout 5400 flock $LOCK taskset -c 0-15 nice -n 10 \
    $PY eval/run_parity.py $1 $2 $PARITY_ARGS > $LOGD/parity_$1__$2.log 2>&1
  rc=$?
  echo "$(date +%T) $1 $2 rc=$rc"
done
