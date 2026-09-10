#!/bin/bash
# 0903 阶梯视频: 人体 | FZ5c(对照, 上排)  vs  人体 | 新模型(下排), 1 条 LAFAN(语料内) + 2 条 AMASS 语料外
# 用法: ladder_video.sh <run_name> <tag>     tag=final 或 step2000 (中途存档)
# 产物: $R/<run>/vid_<tag>/cmp_<clip>.mp4 + grid_<clip>.png (12 帧网格自检)
set -u
cd '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
R=/var/tmp/alphamotion/topo_exp/runs; T=/var/tmp/alphamotion/topo_exp
n=$1; tag=$2; REF=FZ5c_f32_slipw2_hpair01_rp3_0902
export AM_ROWS=1 AM_BODIES=unitree_g1_29dof,pnd_adam_lite,fourier_gr3,x2 AM_T=240 AM_DEV=cpu
laf(){ AM_CACHE=$T/cache_lafan_v3 AM_LBL=$HOME/tmp/cl_v3/lafan AM_SRC="source: LAFAN1 mocap (in corpus)" AM_TITLE="$1" python showcase_render.py $2 $3 $CLIPS_LAF; }
ams(){ AM_CACHE=$T/cache_amass_lconv_tfix AM_LBL=$HOME/tmp/cl_v3/amass_lconv_tfix AM_SRC="source: AMASS held-out (never trained)" AM_TITLE="$1" python showcase_render.py $2 $3 $CLIPS_AMS; }
CLIPS_LAF="dance1_subject1@1971"
CLIPS_AMS="KIT__3__walk_6m_straight_line06 kudo__jump-06-rope_normal_run_fast"
# 快照: 训练进程还在写 model.pt, 复制一份再渲染
if [ "$tag" = final ]; then CK=$R/$n; else CK=$R/$n/snap_$tag; mkdir -p $CK; cp $R/$n/model.pt $R/$n/config.json $CK/; fi
V=$R/$n/vid_$tag; mkdir -p $V $R/ladder_ref
# 对照臂 FZ5c 只渲一次, 大家共用
if [ ! -f $R/ladder_ref/.done ]; then
  laf "FZ5c reference (d128, LAFAN only)" $REF $R/ladder_ref > $R/ladder_ref/log_laf.txt 2>&1
  ams "FZ5c reference (d128, LAFAN only)" $REF $R/ladder_ref > $R/ladder_ref/log_ams.txt 2>&1
  touch $R/ladder_ref/.done
fi
laf "$n  $tag" $CK $V > $V/log_laf.txt 2>&1
ams "$n  $tag" $CK $V > $V/log_ams.txt 2>&1
# 上排 FZ5c / 下排新模型 叠成一个 mp4, 再抽 12 帧网格
for f in $V/showcase_*.mp4; do
  b=$(basename $f .mp4); ref=$R/ladder_ref/$b.mp4
  [ -f "$ref" ] || { echo "[ladder_video] 无对照 $ref"; continue; }
  ffmpeg -y -loglevel error -i $ref -i $f -filter_complex "[0:v][1:v]vstack=inputs=2,scale=1280:-2" -c:v libx264 -crf 23 -pix_fmt yuv420p $V/cmp_$b.mp4
  ffmpeg -y -loglevel error -i $V/cmp_$b.mp4 -vf "select='not(mod(n\,20))',scale=640:-2,tile=4x3" -frames:v 1 $V/grid_$b.png
done
echo "[ladder_video] $n $tag done $(date)"; ls $V/cmp_*.mp4
