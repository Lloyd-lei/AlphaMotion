#!/bin/bash
# GPU BVH->SMPL-X watcher for HiPHI parts extracted by the bulk downloader (raw/hiphi/extracted/part_XXXX/.extracted).
# Every 2 min: one converter run over all extracted parts (it skips clips that already have qa.json in smplx/hiphi
# (CPU path) or smplx_gpu/hiphi (this converter); with nothing to do it exits in ~1 s without touching the GPU),
# then refresh parts_status.json / parts_done.txt. One converter at a time, each run capped at 4 h; the watcher
# exits after MAX_H hours (default 12).  Stop: kill <watcher pid> (see smplx_gpu/hiphi/watch.pid).
set -u
MAX_H=${MAX_H:-12}
PY=/home/arenalabs/miniconda3/envs/genmo-gem/bin/python
D=/home/arenalabs/umr_factory_data
OUT=$D/smplx_gpu/hiphi
OUT30=$D/smplx_gpu/hiphi_30fps
mkdir -p $OUT
echo $$ > $OUT/watch.pid
END=$(( $(date +%s) + MAX_H * 3600 ))
echo "watch start $(date) max_h=$MAX_H pid=$$" >> $OUT/watch.log
while [ "$(date +%s)" -lt "$END" ]; do
  flock /home/arenalabs/umr_factory_data/logs/gpu_heavy.lock timeout 4h $PY /home/arenalabs/gpu_convert/convert.py --kind hiphi --out $OUT --fps30 $OUT30 --lanes 256 \
      --prep-workers 6 --io-workers 3 --max-wall 13800 --log $OUT/convert_log.jsonl \
      --summary $OUT/last_run_summary.json 2>&1 | grep -v "Warning\|warn(" >> $OUT/watch.log
  timeout 10m $PY /home/arenalabs/gpu_convert/parts_status.py > /dev/null 2>> $OUT/watch.log
  sleep 120
done
echo "watch end $(date)" >> $OUT/watch.log
