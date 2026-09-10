#!/bin/bash
# 0903 FZ5c 配方 scaling 阶梯一条龙: 数据 -> 训练 -> eval_dual -> wbc_ready_audit(600 语料外) -> showcase 视频 -> 图
# 用法:  ladder.sh <name> <d_model> <n_heads> <enc_layers> <dec_layers> [extra train args...]
#        LADDER_DATA=lafan  只用 LAFAN (对照臂 S0);  默认 lafan+amass(tfix)
#        LADDER_PROBE=100   只跑 100 步测 s/步 (不 eval 不渲染)
#        LADDER_STEPS1=20000 LADDER_STEPS2=8000  两段步数 (0903 Chinchilla 版: backbone 扫完一遍数据=20k, 微调段照 FZ5c 8k)
#        LADDER_PLOT="run1 run2 ..."  结束后画图用的 run 列表
set -u
cd '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
R=/var/tmp/alphamotion/topo_exp/runs; T=/var/tmp/alphamotion/topo_exp
n=$1; DM=$2; NH=$3; EL=$4; DL=$5; shift 5
export AM_EXCLUDE_EMB=${AM_EXCLUDE_EMB:-pal_talos,engineai_pm01}
if [ "${LADDER_DATA:-full}" = "lafan" ]; then
  CACHE=$T/cache_lafan_v3; EXC=""
else
  CACHE=$T/cache_lafan_v3,$T/cache_amass_lconv_tfix
  EXC="--exclude_set $T/cache_amass_heldout/sequences.json"
fi
STEPS1=${LADDER_STEPS1:-${LADDER_STEPS:-8000}}; STEPS2=${LADDER_STEPS2:-8000}
# FZ5c 血统是两段 (0903 查明): 第一段 = F32 从零 8000 步 (lr 3e-4, 无脚滑/往返/手对, w_rootpos 1);
# 第二段 = FZ5c 8000 步 (lr 1e-4, --init 第一段, 加 slip_pairs/reenc/hpair/slip_w, w_rootpos 3).
# 只换宽深, 其余一字不改. 两段的 config 差异见 runs/F32_lv32_0902 vs FZ5c_*/config.json.
BASE="--cache $CACHE $EXC --bs 256 --n_latent 128 --fsq_stages 1 --fsq_levels 32,32,32,32,32,32,32,32,32,32,32,32,32 --mask_prob 0.3 --mask_mix rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10 --w_pos 5 --w_contrast 0.5 --combo_sample 8 --contrast_sample 6 --use_dof --judge_mode grad --w_judge 0.1 --fk_loss --contact_chan --w_pen 200 --w_cgeo 1 --n_latent_p 4 --root_pos --con_dropout 0.3 --reenc_prob 0.3 --eval_every 1000 --train_scope all --log_every 50"
STAGE1="$BASE --steps $STEPS1 --lr 3e-4 --w_rootpos 1"
STAGE2="$BASE --steps $STEPS2 --lr 1e-4 --w_rootpos 3 --slip_pairs --slip_frac 0.5 --reenc_consistency 1.0 --w_slip_w 2 --w_hpair 0.1"
SIZE="--d_model $DM --n_heads $NH --enc_layers $EL --dec_layers $DL"
mkdir -p $R/$n $R/${n}_s1
if [ -n "${LADDER_PROBE:-}" ]; then
  python train_dual.py $STAGE2 $SIZE --probe_steps $LADDER_PROBE --out $R/${n}_probe "$@" 2>&1 | grep -v Warning | tee $R/${n}_probe.log | grep -E "PROBE|exclude_set|rows ->|h0 表|Error|error" ; exit
fi
# ---- 第一段 (F32 配方, 从零) ----
if [ ! -f $R/${n}_s1/DONE ]; then
  echo "[ladder] $n stage1 start $(date)"; echo "$STAGE1 $SIZE $@" > $R/${n}_s1/launch_args.txt
  setsid nohup bash ladder_watch.sh ${n}_s1 > $R/${n}_s1/watch.log 2>&1 < /dev/null & WATCH=$!
  python train_dual.py $STAGE1 $SIZE --out $R/${n}_s1 "$@" > $R/${n}_s1.log 2>&1
  kill $WATCH 2>/dev/null
  grep -q "^DONE" $R/${n}_s1.log || { echo "[ladder] $n stage1 没跑到 DONE, 停"; tail -20 $R/${n}_s1.log; exit 1; }
  touch $R/${n}_s1/DONE; echo "[ladder] $n stage1 done $(date)"
fi
# ---- 第二段 (FZ5c 配方, --init 第一段) ----
echo "[ladder] $n stage2 start $(date)"; echo "$STAGE2 $SIZE --init $R/${n}_s1/model.pt $@" > $R/$n/launch_args.txt
setsid nohup bash ladder_watch.sh $n > $R/$n/watch.log 2>&1 < /dev/null & WATCH=$!
python train_dual.py $STAGE2 $SIZE --init $R/${n}_s1/model.pt --out $R/$n "$@" > $R/$n.log 2>&1
kill $WATCH 2>/dev/null
grep -q "^DONE" $R/$n.log || { echo "[ladder] $n stage2 没跑到 DONE, 停"; tail -20 $R/$n.log; exit 1; }
echo "[ladder] $n train done $(date)"
# ---- 判官 1: 分割内 ----
python eval_dual.py --run $R/$n > $R/$n.eval_dual.log 2>&1; echo "[ladder] eval_dual $(tail -1 $R/$n.eval_dual.log)"
# ---- 判官 2: 600 条语料外四张考卷 (与 FZ5c 同尺) ----
AM_CACHE=$T/cache_amass_lconv_tfix AM_LBL=$HOME/tmp/cl_v3/amass_lconv_tfix AM_BS=4096 \
  python wbc_ready_audit.py $n $HOME/tmp/wbcready_$n.json > $HOME/tmp/wbcready_$n.log 2>&1 \
  || AM_CACHE=$T/cache_amass_lconv_tfix AM_LBL=$HOME/tmp/cl_v3/amass_lconv_tfix AM_BS=1024 \
     python wbc_ready_audit.py $n $HOME/tmp/wbcready_$n.json > $HOME/tmp/wbcready_$n.log 2>&1
for t in unitree_g1_29dof pnd_adam_lite fourier_gr3; do
  python wbc_ready_report.py $HOME/tmp/wbcready_$n.json $t > $HOME/tmp/wbcready_${n}_$t.txt 2>&1
done
echo "[ladder] audit done $(date)"
# ---- 判官 3: 眼睛. 1 条 LAFAN + 2 条 AMASS 语料外, 上排 人体|FZ5c 下排 人体|新模型 ----
bash ladder_video.sh $n final > $R/$n/video_final.log 2>&1; tail -4 $R/$n/video_final.log
PL=${LADDER_PLOT:-"L_S_0903 L_S20k_0903 L_M20k_0903 L_L20k_0903 L_S0_lafan_0903"}; python plot_ladder.py $(for r in $PL; do echo $R/$r; done) --out $R/ladder_0903 2>&1 | tail -3
echo "[ladder] $n ALL DONE $(date)"
