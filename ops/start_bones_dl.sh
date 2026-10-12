#!/bin/bash
# 启动/重启 BONES-SEED 下载（断点续传；寿命 48 小时；关掉 xet 多路并发，免得 USB WiFi 被连接数压垮）。停止：kill $(cat /home/arenalabs/umr_factory_data/logs/bones/dl.pid)
cd /home/arenalabs/umr_factory_data || exit 1
export HF_HUB_DISABLE_XET=1 HF_TOKEN="$(cat /home/arenalabs/.cache/huggingface/token)" HF_HOME=/var/tmp/hf_home_bones
setsid nohup timeout --signal=TERM --kill-after=120 48h taskset -c 0-15 nice -n 10 \
  /var/tmp/alphamotion/umr_factory/umr/env/bin/python -u scripts/bones_seed_dl.py >> logs/bones/dl.log 2>&1 < /dev/null &
echo $! > logs/bones/dl.pid
echo "started $(cat logs/bones/dl.pid)"
