#!/bin/bash
# 节流守护（lloyd 1007：USB WiFi 网卡在整机高负载时会掉线）
# 每 20 秒：把本项目的重活进程（umr 环境 / genmo-gem 环境 / aria2c）绑到 CPU 0-15（32 线程里的一半）并降优先级 nice 10；
# 记录 CPU 负载、GPU 利用率/功耗、WiFi 是否在线，用来找“多大负载会掉线”。寿命 48 小时。停止：kill $(cat logs/throttle_guard.pid)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
D=$SR_DATA_ROOT
LOG=$D/logs/throttle.log
echo $$ > $D/logs/throttle_guard.pid
END=$(( $(date +%s) + 48*3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  for p in $(pgrep -u "$USER" -f "$SR_PYTHON_ENV/bin/python|/envs/genmo-gem/bin/python|^aria2c|hiphi_staged_dl|grail_dl_loop"); do
    taskset -a -p -c 0-15 "$p" >/dev/null 2>&1
    renice -n 10 -p "$p" >/dev/null 2>&1
  done
  g=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  w=$(cat /sys/class/net/wlx8c86dd7755c9/operstate 2>/dev/null)
  ping -c1 -W2 1.1.1.1 >/dev/null 2>&1 && net=up || net=down
  echo "$(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg) gpu(util,W,MiB)=$g wifi=$w net=$net" >> $LOG
  sleep 20
done
