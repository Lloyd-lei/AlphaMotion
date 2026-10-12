#!/bin/bash
# audit every converted clip lacking obj_pen.json; $1 = parallelism
cd /home/arenalabs/umr_factory_data
PY=/home/arenalabs/miniconda3/envs/genmo-gem/bin/python
for q in smplx/omnicontact/*/*/*/qa.json smplx/hiphi/data/*/*/*/qa.json; do
  d=$(dirname $q); [ -f $d/obj_pen.json ] || echo $d
done | xargs -P ${1:-4} -I{} timeout 900 env OMP_NUM_THREADS=1 $PY scripts/obj_pen_audit.py {} 40 > /dev/null 2>> logs/pen_audit.err
echo "audit sweep done $(date)"
