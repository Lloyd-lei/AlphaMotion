#!/bin/bash
# 启动/重启 HiPHI 分卷下载器（会从断点续传；寿命 120 小时；1011 lloyd 同意继续下：预算 800 GB、/home 至少留 60 GB、8 连接）。停止：kill $(cat $SR_DATA_ROOT/logs/hiphi_dl.pid)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd $SR_DATA_ROOT || exit 1
setsid nohup env LIFETIME_H=120 BUDGET_GB=800 MIN_FREE_GB=60 CONNS=8 taskset -c 0-15 nice -n 10 timeout --signal=TERM --kill-after=300 121h \
  ${SR_BASE_PYTHON}3 -u "$REPO/surface_retargeting/ingest/hiphi/hiphi_staged_dl.py" >> logs/hiphi_streamc.log 2>&1 < /dev/null &
echo $! > logs/hiphi_dl.pid
echo "started $(cat logs/hiphi_dl.pid)"
