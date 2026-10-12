#!/bin/bash
# v2 metric (max-joint 12 cm) full pass: coverage then head/tail rules; single process at a time
cd /home/arenalabs/umr_factory_data/experiments/truncation
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 METRIC=v2
PY=/var/tmp/alphamotion/umr_factory/umr/env/bin/python
until grep -q DONE headtail.log; do sleep 10; done
timeout 7200 taskset -c 0-15 nice -n 10 $PY cov_all.py > cov_all_v2.log 2>&1
timeout 7200 taskset -c 0-15 nice -n 10 $PY headtail.py > headtail_v2.log 2>&1
echo V2_DONE >> headtail_v2.log
