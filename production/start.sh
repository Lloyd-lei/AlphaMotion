#!/bin/bash
# Start the detached production runner (survives the shell / agent session).  Resource limits per RESOURCE_LIMITS.md.
OUT=/home/arenalabs/umr_factory_data/production/v1
mkdir -p $OUT/_logs
rm -f $OUT/_STOP
if [ -f $OUT/_logs/runner.pid ] && kill -0 $(cat $OUT/_logs/runner.pid) 2>/dev/null; then echo "already running pid $(cat $OUT/_logs/runner.pid)"; exit 0; fi
cd /home/arenalabs/umr_gpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 UMRG_DUTY=${UMRG_DUTY:-0.6} setsid nohup taskset -c 0-15 nice -n 10 \
  /var/tmp/alphamotion/umr_factory/umr/env/bin/python production/produce.py "$@" >> $OUT/_logs/runner.stdout 2>&1 < /dev/null &
echo $! > $OUT/_logs/runner.pid
echo "started pid $(cat $OUT/_logs/runner.pid)"
