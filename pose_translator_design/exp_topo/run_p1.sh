#!/usr/bin/env bash
# P1 launcher — temporal tokenizer on top of the FROZEN E19_K128_0805 base.
#
# GATE:  waits for runs/P0_DONE, or accepts runs/P0_verdict.md older than 30 min.
# FLAGS: this script REFUSES to auto-pick the P0-decided flags. The coordinator
#        reads runs/P0_verdict.md and exports:
#          P1_STREAM_REPR=z|codes     (P0-b verdict: which input arm)
#          P1_AUG_TIME=<prob 0..1>    (P0-a verdict: time-scale aug probability)
#          P1_AUG_STRENGTH=<0..1>     (P0-a verdict: 1.0 = full 0.5x..2x)
#        optional: P1_W_NCE (default 0.25), P1_STEPS (default 30000),
#                  P1_RUN_CHUNK=1 (also train the chunk control), P1_TAG.
#
# Does, in order: encode missing multi-embodiment streams (idempotent, ~12 GB),
# train OAT main run, train unordered control (matched bits), run eval_temporal.
set -u
ROOT=/var/tmp/alphamotion/topo_exp
RUNS=$ROOT/runs
STREAMS=$ROOT/e13_stream
EXP='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
E19=$RUNS/E19_K128_0805

# ---------------------------------------------------------------- P0 gate
while true; do
  [ -f "$RUNS/P0_DONE" ] && { echo "P0_DONE found"; break; }
  if [ -f "$RUNS/P0_verdict.md" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$RUNS/P0_verdict.md") ))
    [ "$age" -gt 1800 ] && { echo "P0_verdict.md stable for ${age}s — accepting"; break; }
  fi
  echo "waiting for P0 ($(date +%H:%M:%S))"; sleep 120
done

# ---------------------------------------------------------------- refuse to auto-pick
missing=0
for v in P1_STREAM_REPR P1_AUG_TIME P1_AUG_STRENGTH; do
  if [ -z "${!v:-}" ]; then echo "UNSET: $v"; missing=1; fi
done
if [ "$missing" = 1 ]; then
  echo "EDIT FLAGS PER P0 VERDICT THEN RUN"
  echo "  read $RUNS/P0_verdict.md, then e.g.:"
  echo "  P1_STREAM_REPR=z P1_AUG_TIME=0.3 P1_AUG_STRENGTH=0.5 bash run_p1.sh"
  exit 2
fi
W_NCE=${P1_W_NCE:-0.25}
STEPS=${P1_STEPS:-30000}
TAG=${P1_TAG:-$(date +%m%d)}
echo "flags: repr=$P1_STREAM_REPR aug_time=$P1_AUG_TIME aug_strength=$P1_AUG_STRENGTH w_nce=$W_NCE steps=$STEPS"

# ---------------------------------------------------------------- streams (idempotent)
TRAIN_EMBS="human_smpl,unitree_h1,booster_t1,fourier_gr3,tienkung"   # non-holdout for E19
PROBE_ONLY="pal_talos,unitree_g1_29dof"                              # E19 holdouts: kNN probe only
ALL_EMBS="$TRAIN_EMBS,$PROBE_ONLY"

free_gb=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 15 ]; then
  echo "ABORT: only ${free_gb}G free on /var/tmp — streams need ~12G headroom"; exit 3
fi

encode () {  # cache max_clips set_name
  local cache=$1 mc=$2 sname=$3
  for emb in ${ALL_EMBS//,/ }; do
    local out=$STREAMS/${sname}_${emb}
    if [ -f "$out/meta.json" ]; then echo "stream exists: $out"; continue; fi
    echo "encoding $out"
    $PY "$EXP/tokenize_stream.py" --run "$E19" --cache "$cache" --emb "$emb" \
        --out "$out" --max_clips "$mc" --min_len 60 --seed 0 \
        || { echo "ENCODE FAILED $out"; exit 4; }
  done
}
encode "$ROOT/cache_xwbc_amass" 1000 E19a1k
encode "$ROOT/cache_xwbc_lafan" 24  E19l24
SETS="E19a1k,E19l24"

# ---------------------------------------------------------------- train
OAT=$RUNS/P1_oat_${P1_STREAM_REPR}_${TAG}
UNORD=$RUNS/P1_unord_${P1_STREAM_REPR}_${TAG}

common=( --stream_root "$STREAMS" --sets "$SETS" --embs "$TRAIN_EMBS"
         --stream_repr "$P1_STREAM_REPR" --steps "$STEPS" --in_ram
         --aug_time "$P1_AUG_TIME" --aug_strength "$P1_AUG_STRENGTH"
         --w_nce "$W_NCE" )

$PY "$EXP/train_temporal.py" "${common[@]}" --mode oat --out "$OAT" \
    > "$OAT.log" 2>&1 || { echo "OAT RUN FAILED — see $OAT.log"; exit 5; }
$PY "$EXP/train_temporal.py" "${common[@]}" --mode unordered --out "$UNORD" \
    > "$UNORD.log" 2>&1 || { echo "UNORDERED RUN FAILED — see $UNORD.log"; exit 5; }
if [ "${P1_RUN_CHUNK:-0}" = 1 ]; then
  CHUNK=$RUNS/P1_chunk_${P1_STREAM_REPR}_${TAG}
  $PY "$EXP/train_temporal.py" "${common[@]}" --mode chunk --out "$CHUNK" \
      > "$CHUNK.log" 2>&1 || echo "CHUNK RUN FAILED (control only) — see $CHUNK.log"
fi

# ---------------------------------------------------------------- eval
$PY "$EXP/eval_temporal.py" --run "$OAT" --run_unordered "$UNORD" \
    --sets E19a1k --embs "$TRAIN_EMBS" --probe_embs "human_smpl,$PROBE_ONLY,unitree_h1,booster_t1,fourier_gr3" \
    > "$OAT/eval.log" 2>&1 || { echo "EVAL FAILED — see $OAT/eval.log"; exit 6; }

touch "$RUNS/P1_DONE"
echo "P1 DONE: $OAT (+unordered control). Eval: $OAT/eval_temporal/results.json"
echo "mp4 for the visual verdict: $OAT/eval_temporal/mp4/"
