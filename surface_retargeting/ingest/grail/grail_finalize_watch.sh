#!/bin/bash
# Rebuild the GRAIL manifest every 10 min; once all converter loops have finished (done files present and no loop
# process left), run one non-loop collision/metadata pass and a last manifest build, then exit. Lifetime set by caller.
S=/home/arenalabs/umr_factory_data/scripts_grail; G=/home/arenalabs/umr_factory_data/smplx/grail
P=/var/tmp/alphamotion/umr_factory/umr/env/bin/python
cd $S
while true; do
  timeout 600 python3 build_manifest.py > /home/arenalabs/umr_factory_data/logs/grail/manifest_watch.log 2>&1
  if ! pgrep -f "convert_grail.py --loop" > /dev/null && ! pgrep -f "grail_collision.py --loop" > /dev/null; then
    echo "$(date) loops finished; final collision pass" 
    timeout 3h nice -n 10 $P -u grail_collision.py --workers 6 --poll 10 --max-hours 2.8 > /home/arenalabs/umr_factory_data/logs/grail/collision_final.log 2>&1
    timeout 600 python3 build_manifest.py > /home/arenalabs/umr_factory_data/logs/grail/manifest_final.log 2>&1
    echo "$(date) final manifest written"; exit 0
  fi
  sleep 600
done
