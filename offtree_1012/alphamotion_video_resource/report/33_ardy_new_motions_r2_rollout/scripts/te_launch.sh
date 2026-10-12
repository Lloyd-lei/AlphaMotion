#!/bin/bash
cd '/home/arenalabs/Desktop/"be water, robot"/experiment/ardy'
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
exec ~/miniconda3/envs/ardy/bin/python scripts/run_text_encoder_server.py
