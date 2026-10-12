#!/bin/bash
# two-stage teleop probe: baseline BFM (native-trained) x {native refs, Greenwich-decoded refs} x {z window 1, 15}
cd /home/arenalabs/reference_code/ufo; export PYTHONPATH=$PWD UFO_CACHE_DIR=/home/arenalabs/tmp/ufo_cache TORCHDYNAMO_DISABLE=1
P=.venv312/bin/python; O=/home/arenalabs/tmp/teleop2; mkdir -p $O; M=runs/ufo_fb_g1_repl
for spec in "native cache/motion_data/amass_stress/amass_stress_full_ufo.pkl 1" "gw cache/motion_data/amass_gwstress/amass_gwstress_full_ufo.pkl 1" "native cache/motion_data/amass_stress/amass_stress_full_ufo.pkl 15" "gw cache/motion_data/amass_gwstress/amass_gwstress_full_ufo.pkl 15"; do
  set -- $spec; TAG=${1}_w$3
  echo "[teleop2 $(date +%H:%M:%S)] start $TAG"
  timeout 2400 $P /home/arenalabs/tmp/teleop_probe.py --model-folder $M --data-path $2 --tag $TAG --out-dir $O --motion-list 0 1 2 3 4 5 6 --z-window $3 --device cuda:0 > $O/$TAG.log 2>&1
  echo "[teleop2 $(date +%H:%M:%S)] end $TAG rc=$? $(grep -c '\[OK\]' $O/$TAG.log) clips"
done
echo TELEOP2_DONE
