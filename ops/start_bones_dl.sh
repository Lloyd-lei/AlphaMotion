#!/bin/bash
# 启动/重启 BONES-SEED 下载（断点续传；寿命 48 小时；关掉 xet 多路并发，免得 USB WiFi 被连接数压垮）。停止：kill $(cat $SR_DATA_ROOT/logs/bones/dl.pid)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd $SR_DATA_ROOT || exit 1
export HF_HUB_DISABLE_XET=1 HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")" HF_HOME=/var/tmp/hf_home_bones
setsid nohup timeout --signal=TERM --kill-after=120 48h taskset -c 0-15 nice -n 10 \
  $SR_UMR_PYTHON -u "$REPO/surface_retargeting/ingest/download/bones_seed_dl.py" >> logs/bones/dl.log 2>&1 < /dev/null &
echo $! > logs/bones/dl.pid
echo "started $(cat logs/bones/dl.pid)"
