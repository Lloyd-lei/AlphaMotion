#!/bin/bash
# usage: memgate.sh <min_avail_GB> <max_wait_s> -- cmd...   (waits for free RAM, then runs cmd)
need=$1; maxw=$2; shift 3
t=0
while true; do
  av=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$av" -ge "$need" ]; then break; fi
  if [ "$t" -ge "$maxw" ]; then echo "[memgate] gave up: avail=${av}G < ${need}G after ${t}s"; exit 99; fi
  sleep 20; t=$((t+20))
done
echo "[memgate] avail=${av}G, starting: $*"
exec "$@"
