#!/bin/bash
S=/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad/ardy33
cd '/home/arenalabs/Desktop/"be water, robot"/experiment/ardy'
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=api TEXT_ENCODER_URL=http://127.0.0.1:9550/
until curl -s -o /dev/null http://127.0.0.1:9550/; do sleep 5; done
echo "server up $(date)"
~/miniconda3/envs/ardy/bin/python $S/gen33.py $S/gen
echo "exit $?"
