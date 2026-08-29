#!/usr/bin/env bash
# V6 training chain. NEVER kills anything: each arm waits for a specific already-
# running N3 pid to exit before it takes that slot.
#
#   V6_mae_0807        default (--frame_repr key), the isomorphism alone
#   V6_mae_resid_0807  + --residual_in (velocity + SG5 acceleration, I/P anchors)
#
# then n3_eval_temporal against V5c_0806 (control) and N3a_regAR_0807.
set -u
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
EXP='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
RUNS=/var/tmp/alphamotion/topo_exp/runs
cd "$EXP" || exit 1

WAIT_A=${WAIT_A:-0}      # pid whose slot V6_mae takes
WAIT_B=${WAIT_B:-0}      # pid whose slot V6_mae_resid takes
STEPS=${STEPS:-60000}

waitpid () {
  local p=$1
  [ "$p" = "0" ] && return 0
  while kill -0 "$p" 2>/dev/null; do sleep 60; done
  echo "[v6] slot freed (pid $p gone) $(date +%H:%M:%S)"
  sleep 20
}

echo "[v6] chain start $(date)"
waitpid "$WAIT_A"
echo "[v6] launching V6_mae_0807 $(date)"
$PY -W ignore train_v6.py --out "$RUNS/V6_mae_0807" --steps "$STEPS" \
    > "$RUNS/V6_mae_0807.log" 2>&1 &
P1=$!
echo "[v6] V6_mae pid $P1"

waitpid "$WAIT_B"
echo "[v6] launching V6_mae_resid_0807 $(date)"
$PY -W ignore train_v6.py --out "$RUNS/V6_mae_resid_0807" --steps "$STEPS" \
    --residual_in > "$RUNS/V6_mae_resid_0807.log" 2>&1 &
P2=$!
echo "[v6] V6_mae_resid pid $P2"

wait $P1; echo "[v6] V6_mae done $(date)"
wait $P2; echo "[v6] V6_mae_resid done $(date)"

ARMS=V5c_0806,N3a_regAR_0807,V6_mae_0807,V6_mae_resid_0807
echo "[v6] eval $ARMS $(date)"
$PY -W ignore n3_eval_temporal.py --runs "$ARMS" \
    --out "$RUNS/V6_verdict_0807" > "$RUNS/V6_eval.log" 2>&1
$PY -W ignore v6_append_report.py "$RUNS/V6_verdict_0807.txt" "$ARMS" \
    >> "$RUNS/V6_eval.log" 2>&1
# n3_eval_temporal opens the verdict with "w", so the build record is re-appended
# after it rather than being written first and lost.
cat "$RUNS/V6_BUILD_0807.txt" >> "$RUNS/V6_verdict_0807.txt" 2>/dev/null
echo "[v6] DONE $(date) -> $RUNS/V6_verdict_0807.txt"
