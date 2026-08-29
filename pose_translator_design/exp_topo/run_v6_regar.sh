#!/usr/bin/env bash
# V6_regar_0807 — V6 frame masking + --reg_ar, the "both fixes" cell of the 2x2.
# Then the authoritative 5-arm eval, assembled into runs/V6_verdict_0807.txt.
#
# NEVER KILLS ANYTHING, and — the lesson from tonight — never launches on a
# startup-memory reading. train_dual arms grow 24.5 -> 46.3 GB at their eval
# checkpoints hours after they start, so a slot has to be measured as SUSTAINED
# free memory, not inferred from a pid disappearing.
set -u
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
EXP='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
RUNS=/var/tmp/alphamotion/topo_exp/runs
cd "$EXP" || exit 1

NEED_MIB=${NEED_MIB:-12000}   # 3.7 GB for this arm + ~8 GB margin for others' eval peaks
POLLS=${POLLS:-5}             # consecutive one-minute polls that must all clear
STEPS=${STEPS:-60000}

free_mib () {
  local u t
  read -r u t < <(nvidia-smi --query-gpu=memory.used,memory.total \
                  --format=csv,noheader,nounits | tr ',' ' ')
  echo $(( t - u ))
}

echo "[regar] waiting for >= ${NEED_MIB} MiB free, sustained over ${POLLS} polls  $(date)"
ok=0
while [ "$ok" -lt "$POLLS" ]; do
  f=$(free_mib)
  if [ "$f" -ge "$NEED_MIB" ]; then
    ok=$(( ok + 1 ))
    echo "[regar] free ${f} MiB  ok ${ok}/${POLLS}  $(date +%H:%M:%S)"
  else
    [ "$ok" -gt 0 ] && echo "[regar] free ${f} MiB  gate reset  $(date +%H:%M:%S)"
    ok=0
  fi
  [ "$ok" -lt "$POLLS" ] && sleep 60
done

echo "[regar] launching V6_regar_0807 $(date)"
$PY -W ignore train_v6.py --out "$RUNS/V6_regar_0807" --steps "$STEPS" --reg_ar \
    > "$RUNS/V6_regar_0807.log" 2>&1
echo "[regar] training done $(date)"

# don't run two evals at once: wait for run_v6.sh's 4-arm eval to finish first
while pgrep -f "n3_eval_temporal.py" > /dev/null 2>&1; do sleep 60; done

ARMS=V5c_0806,N3a_regAR_0807,V6_mae_0807,V6_regar_0807,V6_mae_resid_0807
echo "[regar] authoritative 5-arm eval: $ARMS  $(date)"
$PY -W ignore n3_eval_temporal.py --runs "$ARMS" \
    --out "$RUNS/V6_table_0807" > "$RUNS/V6_eval_full.log" 2>&1
$PY -W ignore v6_append_report.py "$RUNS/V6_appendix_0807.txt" "$ARMS" \
    >> "$RUNS/V6_eval_full.log" 2>&1

# assemble: how-to-read framing -> the table -> appendix -> build record
cat "$RUNS/V6_HEADER_0807.txt" "$RUNS/V6_table_0807.txt" \
    "$RUNS/V6_appendix_0807.txt" "$RUNS/V6_BUILD_0807.txt" \
    > "$RUNS/V6_verdict_0807.txt" 2>/dev/null
echo "[regar] DONE $(date) -> $RUNS/V6_verdict_0807.txt"
