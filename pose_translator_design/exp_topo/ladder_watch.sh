#!/bin/bash
# 0903 存档 watcher: 训练每存一次 model.pt (每 eval_every 步) 就自动出中途视频, 问进度时直接有现成的
# 用法: ladder_watch.sh <run_name>   (由 ladder.sh 后台拉起, 训练 DONE 后自己退出)
set -u
R=/var/tmp/alphamotion/topo_exp/runs; n=$1; last=""
cd '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
while true; do
  sleep 60
  [ -f $R/$n/model.pt ] || { grep -q "^DONE\|Error" $R/$n.log 2>/dev/null && exit 0; continue; }
  step=$(grep -c "^EVAL" $R/$n.log 2>/dev/null); step=$((step*1000))
  grep -q "^DONE" $R/$n.log 2>/dev/null && exit 0           # 终版由 ladder.sh 自己渲
  [ "$step" = "$last" ] || [ "$step" = 0 ] && continue
  last=$step; bash ladder_video.sh $n step$step > $R/$n/watch_step$step.log 2>&1
done
