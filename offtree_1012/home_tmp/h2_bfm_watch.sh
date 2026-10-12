#!/bin/bash
# BFM-Zero H2 ground run: every 10 min log steps/emd; at 10M/20M/30M/40M/50M run tracking_inference (videos) on the 4 clips
set -u; cd /home/arenalabs/reference_code/ufo
P=.venv312/bin/python; L=/home/arenalabs/tmp/ufo_train_h2_ground.log; S=/home/arenalabs/tmp/h2_bfm_status.txt; O=/home/arenalabs/tmp/h2_bfm
export PYTHONPATH=$PWD UFO_CACHE_DIR=/home/arenalabs/tmp/ufo_cache TORCHDYNAMO_DISABLE=1; mkdir -p $O
say(){ echo "[watch $(date +%H:%M:%S)] $*" | tee -a $S; }
done_ms=""
while true; do
  sleep 600
  PID=$(awk '{print $2}' /home/arenalabs/tmp/ufo_h2_ground_pid.txt); kill -0 $PID 2>/dev/null || { say "TRAIN_PROCESS_GONE"; break; }
  ST=$($P - <<'PY' 2>/dev/null
import re
s=open('/home/arenalabs/tmp/ufo_train_h2_ground.log',errors='ignore').read()
g=re.findall(r"global_env_steps': ([0-9]+)",s); e=re.findall(r"'emd': np\.float64\(([0-9.]+)\)",s); f=re.findall(r"'FPS': np\.float64\(([0-9.]+)\)",s); tz=re.findall(r"'torso_z_mean': np\.float64\(([0-9.]+)\)",s)
print(g[-1] if g else 0, e[-1] if e else 'na', f[-1] if f else 'na', tz[-1] if tz else 'na')
PY
)
  say "status steps/emd/fps/torso_z = $ST"; N=$(echo $ST | awk '{print $1}')
  for M in 10 20 30 40 50; do
    echo "$done_ms" | grep -q " $M " && continue
    [ "$N" -ge $((M*1000000)) ] || continue
    say "milestone ${M}M: tracking_inference"
    timeout 1800 $P -m humanoidverse.tracking_inference --model-folder runs/ufo_fb_h2_ground --data-manifest configs/data/h2_ground.yaml --dataset h2_ground --device cuda:0 --headless --save-mp4 --disable-dr --motion-list 0 1 2 3 > $O/ms_${M}M.log 2>&1
    for i in 0 1 2 3; do [ -f runs/ufo_fb_h2_ground/tracking_inference/tracking_$i.mp4 ] && cp runs/ufo_fb_h2_ground/tracking_inference/tracking_$i.mp4 $O/H2_${M}M_track$i.mp4; done
    say "milestone ${M}M done: $(ls $O/H2_${M}M_track*.mp4 2>/dev/null | wc -l) videos"; done_ms="$done_ms $M "
  done
done
say WATCH_DONE
