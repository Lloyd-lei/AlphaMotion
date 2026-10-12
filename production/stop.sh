#!/bin/bash
# Graceful stop: the runner exits after the current job (its outputs are complete).  Use --now to kill immediately
# (the interrupted job has no .done marker and is redone on the next start).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
OUT=$SR_DATA_ROOT/production/v1
touch $OUT/_STOP
if [ "$1" = "--now" ] && [ -f $OUT/_logs/runner.pid ]; then kill $(cat $OUT/_logs/runner.pid) && echo killed; fi
