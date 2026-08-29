#!/bin/bash
# E25 — DOF masking. Two arms, strictly controlled against E19_K128_0805.
#
# CONTROL: every flag below is copied verbatim from E19_K128_0805/config.json — same
# cache (cache_xwbc_lafan, NOT any rebuilt/bug-fixed cache), same K=128, same single-stage
# f13 codebook, same 8000 steps, same holdout, same mask recipe, same combo/contrast
# sampling, same judge_mode=grad, same use_dof, same 256/8/4/4 backbone, same seed 0.
# The ONLY difference is the DOF flag:
#   arm A  --rot_head angles   decoder emits hinge angles -> FK -> global rotations
#                              (+771 params = the 3 extra output units, +0.0043%)
#   arm B  --dof_loss          free 6D head, loss projected onto the hinge manifold
#                              (+0 params)
#
# GPU ETIQUETTE: waits for E13_DONE — the LAST link of the existing chain
# (E24 param axis -> E23 seeds -> SEEDS_DONE -> E13 probe -> E13_DONE). Waiting on
# SEEDS_DONE would collide with run_e13_after_seeds.sh, which polls that one. On top of
# the sentinel it also waits until the card actually has room, so a straggler can't be
# starved by us.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-/home/arenalabs/miniconda3/envs/ardy/bin/python}
R=/var/tmp/alphamotion/topo_exp/runs
C=${CACHE:-/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan}
SENTINEL=${SENTINEL:-$R/E13_DONE}
MINFREE=${MINFREE:-56000}          # MiB of GPU memory a full-cache run needs
MAX_WAIT_H=${MAX_WAIT_H:-30}
DATE=${DATE:-0805}
STEPS=${STEPS:-8000}
L8=8,8,8,8,8,8,8,8,8,8,8,8,8
BASE="--cache $C --steps $STEPS --w_contrast 0.5 --mask_prob 0.3 \
--mask_mix rand:0.3,ee:0.25,root:0.15,sparse:0.3 --combo_sample 8 --contrast_sample 6 \
--holdout unitree_g1_29dof,unitree_g1_23dof,pal_talos --fsq_stages 1 --fsq_levels $L8 \
--judge_mode grad --use_dof --n_latent 128 --d_model 256 --n_heads 8 --enc_layers 4 \
--dec_layers 4 --seed 0"

log() { echo "[$(date '+%F %T')] $*"; }

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
if [ "${SKIP_WAIT:-0}" != "1" ]; then
  log "waiting for $SENTINEL"
  while [ ! -e "$SENTINEL" ]; do
    [ "$(date +%s)" -ge "$deadline" ] && { log "TIMEOUT waiting for sentinel"; exit 3; }
    sleep 60
  done
fi
wait_gpu() {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( total - used ))
    [ "$free" -ge "$MINFREE" ] && { log "GPU free ${free}MiB >= ${MINFREE}MiB, go"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && { log "TIMEOUT waiting for GPU room"; return 1; }
    log "GPU free ${free}MiB < ${MINFREE}MiB, waiting"
    sleep 120
  done
}

cd "$HERE" || exit 1
run () {   # $1=tag  $2...=extra flags
  tag=$1; shift
  out="$R/E25_${tag}_${DATE}"
  if [ -f "$out/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    log "skip $tag (already trained)"; return 0
  fi
  wait_gpu || return 1
  log "=== E25 $tag start ==="
  echo "=== E25 $tag start $(date +%H:%M:%S) ===" >> "$R/e25_serial.log"
  # shellcheck disable=SC2086
  $PY train_dual.py $BASE "$@" --out "$out" > "$R/E25_${tag}_${DATE}.log" 2>&1
  rc=$?
  echo "=== E25 $tag done $(date +%H:%M:%S) rc=$rc ===" >> "$R/e25_serial.log"
  log "=== E25 $tag done rc=$rc ==="
}

run dofhead --rot_head angles
run dofloss --dof_loss

log "evaluating against E19_K128_0805 (CPU, same yardstick)"
$PY "$HERE/eval_e25.py" --runs E19_K128_0805 "E25_dofhead_${DATE}" "E25_dofloss_${DATE}" \
    --out "$R/E25_verdict.json" 2>&1 | tee "$R/E25_eval_${DATE}.log"
# held-out AMASS, same three models (G1 unseen there too)
$PY "$HERE/eval_e25.py" --runs E19_K128_0805 "E25_dofhead_${DATE}" "E25_dofloss_${DATE}" \
    --cache /var/tmp/alphamotion/topo_exp/cache_amass_heldout --reach_from "$C" \
    --out "$R/E25_verdict_heldout.json" 2>&1 | tee -a "$R/E25_eval_${DATE}.log"
# Last step, unconditionally: E26 (fixed-cache vs original-cache pair) waits on this.
# Dropped even if an eval hiccups — the sentinel means "E25 has released the GPU and
# written what it could", and both model.pt files are on disk by this point.
touch "$R/E25_DONE"
log "E25 DONE — $R/E25_verdict.json  (sentinel $R/E25_DONE dropped for E26)"
