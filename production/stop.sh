#!/bin/bash
# Graceful stop: the runner exits after the current job (its outputs are complete).  Use --now to kill immediately
# (the interrupted job has no .done marker and is redone on the next start).
OUT=/home/arenalabs/umr_factory_data/production/v1
touch $OUT/_STOP
if [ "$1" = "--now" ] && [ -f $OUT/_logs/runner.pid ]; then kill $(cat $OUT/_logs/runner.pid) && echo killed; fi
