#!/bin/bash
# usage: runwait.sh <timeout_s> <cmd...>
# waits (max 30 min) for MemAvailable >= 15 GB, then runs cmd throttled per RESOURCE_LIMITS.md
# (cores 0-15, nice 10, 2 BLAS/OMP threads, CPU-only MuJoCo/UMR, EGL for rendering) with a timeout
T=$1; shift
for i in $(seq 1 60); do
  A=$(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)
  NET=$(tail -1 /home/arenalabs/umr_factory_data/logs/throttle.log 2>/dev/null | grep -o 'net=[a-z]*')
  if [ "$A" -ge 15 ] && [ "$NET" != "net=down" ]; then break; fi
  echo "[runwait] MemAvailable ${A} GB ${NET}, waiting"; sleep 30
done
A=$(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)
if [ "$A" -lt 15 ]; then echo "[runwait] gave up: ${A} GB"; exit 3; fi
echo "[runwait] start with ${A} GB available"
cd /home/arenalabs/umr_factory_data/robots/unitree_g1_29dof/pipeline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= timeout "$T" taskset -c 0-15 nice -n 10 "$@"
