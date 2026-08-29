#!/usr/bin/env bash
# E13 probe — launches itself once the GPU queue is done.
#
# The queue currently runs E22 then four E23 seed runs; that pipeline drops the
# sentinel file below when it finishes. This script blocks on the sentinel, then
# does the whole E13 probe end to end and leaves a verdict json + figures:
#
#   1. tokenize_stream.py   freeze E19_K128_0805, encode contiguous 30 fps AMASS clips
#   2. train_e13_probe.py   the factorized probe + its controls (see RUNS below)
#   3. eval_e13_probe.py    pre-registered criteria, including the CLUSTERING STOP RULE
#
# The controls are the whole point of the bits claim: `frame` spends the SAME number
# of bits per window on a plain per-frame code, so "factorizing wins" is a comparison,
# not an assertion. Each pair (factor_*, frame_*) is bit-matched; the two budget points
# give a quality-vs-bits curve instead of a single unfalsifiable number.
#
# Usage:
#   nohup bash run_e13_after_seeds.sh > /var/tmp/alphamotion/topo_exp/runs/E13_pipeline.log 2>&1 &
# Env overrides:
#   SKIP_WAIT=1      start immediately (only if you know the GPU is free)
#   MAX_WAIT_H=24    give up waiting after this many hours
#   STEPS=30000      training steps per run
#   MAX_CLIPS=0      0 = all 4000 AMASS clips
#   DATE=0805        run-name suffix
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-/home/arenalabs/miniconda3/envs/ardy/bin/python}
ROOT=/var/tmp/alphamotion/topo_exp
SENTINEL=${SENTINEL:-$ROOT/runs/SEEDS_DONE}
TOKENIZER=${TOKENIZER:-$ROOT/runs/E19_K128_0805}
CACHE=${CACHE:-$ROOT/cache_xwbc_amass}
EMB=${EMB:-human_smpl}
STREAM=${STREAM:-$ROOT/e13_stream/E19_K128_amass_${EMB}}
DATE=${DATE:-0805}
STEPS=${STEPS:-30000}
MAX_CLIPS=${MAX_CLIPS:-0}
MAX_WAIT_H=${MAX_WAIT_H:-24}
POLL=${POLL:-60}

log() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- 0. wait for the queue
if [ "${SKIP_WAIT:-0}" != "1" ]; then
  log "waiting for $SENTINEL (poll ${POLL}s, give up after ${MAX_WAIT_H}h)"
  deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  while [ ! -e "$SENTINEL" ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "TIMEOUT: sentinel never appeared, refusing to compete for the GPU"; exit 3
    fi
    sleep "$POLL"
  done
  log "sentinel found; letting the GPU settle for 60s"
  sleep 60
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------- 1. freeze + tokenize
if [ -f "$STREAM/meta.json" ]; then
  log "stream already built at $STREAM (delete it to rebuild)"
else
  log "tokenizing contiguous stream -> $STREAM"
  "$PY" "$HERE/tokenize_stream.py" --run "$TOKENIZER" --cache "$CACHE" --emb "$EMB" \
      --out "$STREAM" --max_clips "$MAX_CLIPS"
fi

# ---------------------------------------------------------------- 2. train
# name|mode|extra args      (LOW = ~78 bits/window, HIGH = ~1266 bits/window)
RUNS=(
  "E13_factor_low_$DATE|factor|--phase_bins 16"
  "E13_frame_low_$DATE|frame|--match_bits 77.9 --frame_levels 5"
  "E13_slow_only_$DATE|slow_only|"
  "E13_phase_only_$DATE|phase_only|--phase_bins 16"
  "E13_factor_high_$DATE|factor|--phase_bins 16 --res_dims 32 --res_levels 5"
  "E13_frame_high_$DATE|frame|--match_bits 1266 --frame_levels 5"
  # anti-collapse fallback: the CPU dry run showed the free phase barely moving
  # (|dtheta| ~ 0.02 rad/frame after 600 steps). --w_phase_spread only penalizes the
  # phases of one window being bunched on the circle; it says nothing about order or
  # direction, so "does it advance monotonically and wrap" stays a real test. Read this
  # run only if the free-phase run's phase collapsed.
  "E13_factor_low_spread_$DATE|factor|--phase_bins 16 --w_phase_spread 0.3"
)
OUTS=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name mode extra <<< "$spec"
  out="$ROOT/runs/$name"; OUTS+=("$out")
  if [ -f "$out/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    log "skip $name (already trained; FORCE=1 to redo)"; continue
  fi
  log "training $name  mode=$mode $extra"
  # shellcheck disable=SC2086
  "$PY" "$HERE/train_e13_probe.py" --stream "$STREAM" --out "$out" --mode "$mode" \
      --steps "$STEPS" $extra 2>&1 | tee "$ROOT/runs/$name.log"
done

# ---------------------------------------------------------------- 3. verdict
MAIN="$ROOT/runs/E13_factor_low_$DATE"
log "evaluating (stop rule + bits + phase + decoded quality)"
"$PY" "$HERE/eval_e13_probe.py" --run "$MAIN" --compare "${OUTS[@]:1}" \
    2>&1 | tee "$ROOT/runs/E13_eval_$DATE.log"
# the high-bit factorization gets its own verdict too (same stop rule, different budget)
"$PY" "$HERE/eval_e13_probe.py" --run "$ROOT/runs/E13_factor_high_$DATE" \
    --compare "$ROOT/runs/E13_frame_high_$DATE" \
    2>&1 | tee -a "$ROOT/runs/E13_eval_$DATE.log"
# and the anti-collapse fallback, so section C has something to say if the free phase died
"$PY" "$HERE/eval_e13_probe.py" --run "$ROOT/runs/E13_factor_low_spread_$DATE" \
    2>&1 | tee -a "$ROOT/runs/E13_eval_$DATE.log"

log "E13 DONE"
log "verdict json : $MAIN/e13_eval.json"
log "figures      : $MAIN/figs/fig_e13_slow.png  $MAIN/figs/fig_e13_phase.png"
log "full log     : $ROOT/runs/E13_eval_$DATE.log"

touch "$ROOT/runs/E13_DONE"
