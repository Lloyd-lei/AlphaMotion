#!/bin/bash
# 启动/重启 GRAIL 下载循环（断点续传；8 个连接，守 RESOURCE_LIMITS 的 ≤16；寿命 24 小时）。停止：kill $(cat logs/grail/dl_loop.pid)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd $SR_DATA_ROOT || exit 1
setsid nohup timeout --signal=TERM --kill-after=120 25h taskset -c 0-15 nice -n 10 \
  $SR_UMR_PYTHON -u "$REPO/surface_retargeting/ingest/grail/grail_dl_loop.py" --max-hours 24 --conns 8 \
  >> logs/grail/dl_loop_restart_1007.log 2>&1 < /dev/null &
echo $! > logs/grail/dl_loop.pid
echo "started $(cat logs/grail/dl_loop.pid)"
