#!/usr/bin/env bash
# P1c launcher — REPLACE GLOBAL POOLING. Two arms on the same trainer, everything else
# held constant vs P1/P1b (same streams, same frozen E19_K128_0805, same InfoNCE, same
# OAT prefix idea, same steps/seed/lr/batch).
#
#   ARM C1  --temporal_arch conv   strided 1D conv keeps the TIME AXIS: 60 frames ->
#                                  latent time steps, SNAC-style multi-scale residual
#                                  FSQ (2 tokens @1Hz + 6 tokens @3Hz), decoder
#                                  upsamples back. Ordered prefix = coarse-to-fine over
#                                  SCALE, so the budgets are (2,8) not (1,2,4,8).
#   ARM C2  --temporal_arch freq   register k = the k-th temporal DCT band, fixed basis
#                                  in and out. A static window has only DC, so motion is
#                                  structurally impossible to drop.
#
# WHY (all measured 2026-08-05/06, cited in temporal_common.py's header):
#   1. P1's 8 globally-pooled registers retain 22-73% of GT motion energy; the frame-code
#      floor retains 82-98%. P1b's loss-only fixes did not rescue it (motion-weighted CE
#      null-to-negative; +velocity loss reached .86/.55/.29/.79, still under the bar).
#   2. A zero-training DCT probe retains 95-100% of walk motion with 8 temporal
#      coefficients per dim -> motion is temporally LOW-FREQUENCY; the slots are not the
#      problem, what they attend to is. (Not bit-matched — direction, not a win.)
#   3. LIT_TEMPORAL_FSQ.md: audio codecs universally use STRIDED CONVOLUTIONAL
#      downsampling and never global pooling; SNAC uses multi-scale time resolutions;
#      stable-codec runs OUR exact 6x5 FSQ config at 25 tokens/s with the time axis kept.
#   4. Argmax was cleared as a suspect first (scratchpad p1c_decode_probe.py, 2026-08-06):
#      categorical sampling instead of argmax moved P1's prefix-1 walk energy .733->.755
#      and run .397->.388 while RAISING geodesic error — the collapse is in the
#      representation, not in the readout.
#
# DEFAULT LOSS RECIPE = P1b's ONLY validated positive: --w_vel 0.5, and explicitly NO
# --w_motion (null-to-negative in P1b). --w_multires stays an opt-in extra run so the
# architecture A/B against P1b_motion_vel_0806 changes exactly one thing.
#
# GATE: waits until NO train_temporal.py process is alive, so it never competes for the
# GPU. It never signals or kills anything it did not start.
#
# ENV: P1C_TAG (default MMDD)  P1C_STEPS (30000)  P1C_RATES ("lo hi")
#      P1C_EXTRA=1 (adds the --w_multires run and the 60->20 frame-subsample run)
#      P1C_SKIP_VIDEO=1  P1C_POLL (60 s)
set -u
ROOT=/var/tmp/alphamotion/topo_exp
RUNS=$ROOT/runs
STREAMS=$ROOT/e13_stream
EXP='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
SP='/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad'
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python
TAG=${P1C_TAG:-$(date +%m%d)}
STEPS=${P1C_STEPS:-30000}
RATES=${P1C_RATES:-"lo hi"}
POLL=${P1C_POLL:-60}

TRAIN_EMBS="human_smpl,unitree_h1,booster_t1,fourier_gr3,tienkung"   # non-holdout for E19
PROBE_ONLY="pal_talos,unitree_g1_29dof"                              # E19 holdouts
SETS="E19a1k,E19l24"

# ---------------------------------------------------------------- GPU gate
echo "== P1c gate: waiting for the GPU to be free of train_temporal.py =="
while true; do
  # -a matches the full command line; our own children are launched only after this loop
  if ! pgrep -af "python.*[t]rain_temporal\.py" > /dev/null 2>&1; then
    echo "GPU free at $(date +%H:%M:%S) — starting P1c"; break
  fi
  echo "  busy: $(pgrep -af 'python.*[t]rain_temporal\.py' | sed 's/.*--out //' | tr '\n' ' ')($(date +%H:%M:%S))"
  sleep "$POLL"
done
[ -f "$RUNS/P1B_DONE" ] && echo "(P1B_DONE sentinel present)"

for s in ${SETS//,/ }; do
  for e in ${TRAIN_EMBS//,/ }; do
    [ -f "$STREAMS/${s}_${e}/meta.json" ] || { echo "MISSING STREAM $STREAMS/${s}_${e}"; exit 3; }
  done
done

# ---------------------------------------------------------------- common flags
# byte-identical to the P1b_motion_vel_0806 recipe except --w_motion (dropped: null to
# negative) and the arch flags. Same seed, same steps, same InfoNCE, same time aug.
COMMON=( --stream_root "$STREAMS" --sets "$SETS" --embs "$TRAIN_EMBS"
         --stream_repr z --mode oat --steps "$STEPS" --in_ram
         --aug_time 0.3 --aug_strength 0.5 --w_nce 0.25 --nce_embs 2
         --w_vel 0.5 --seed 0 )

RUN_LIST=()

train () {          # name, then extra flags
  local name=$1; shift
  local out=$RUNS/$name
  if [ -f "$out/model.pt" ] && [ -f "$out/eval_log.json" ]; then
    echo "SKIP $name (already trained)"; RUN_LIST+=("$out"); return 0
  fi
  echo "== TRAIN $name  ($(date +%H:%M:%S)) =="
  $PY "$EXP/train_temporal.py" "${COMMON[@]}" "$@" --out "$out" > "$out.log" 2>&1 \
    || { echo "TRAIN FAILED $name — see $out.log"; tail -20 "$out.log"; return 1; }
  grep -m1 "BIT BUDGET" "$out.log"
  tail -2 "$out.log"
  RUN_LIST+=("$out")
}

# ---------------------------------------------------------------- the arms
# n_content is held at the P1 value inside each rate tier so the two arms are compared
# at IDENTICAL bits: 8 registers x 6 dims x log2(5) = 111.45 bit / 2 s = 55.7 bit/s
# (+13.93 bit for the never-hidden duration seat). The "hi" tier doubles that to
# 111.5 bit/s, inside LIT's 128-192 bit/s rate-distortion estimate for gait phase, so a
# budget failure is not mistaken for an architecture failure.
for rate in $RATES; do
  case $rate in
    lo) NC=8;  SCALES=2,6 ;;
    hi) NC=16; SCALES=4,12 ;;
    *) echo "unknown rate '$rate'"; exit 4 ;;
  esac
  train "P1c_C1_conv_${rate}_${TAG}" --temporal_arch conv --n_content "$NC" \
        --conv_scales "$SCALES"     || exit 5
  train "P1c_C2_freq_${rate}_${TAG}" --temporal_arch freq --n_content "$NC" || exit 5
done

if [ "${P1C_EXTRA:-0}" = 1 ]; then
  # multi-resolution temporal dynamics loss (audio's multi-res STFT loss, ported)
  train "P1c_C1_conv_lo_mres_${TAG}" --temporal_arch conv --n_content 8 \
        --conv_scales 2,6 --w_multires 0.5 || true
  # ProMAG: same 2 s span and the SAME 55.7 bit/s, but 20 frames instead of 60, so the
  # temporal compression drops 7.5x -> 2.5x without spending a single extra bit
  train "P1c_C1_conv_fps10_${TAG}" --temporal_arch conv --n_content 8 \
        --conv_scales 2,6 --T 20 --frame_stride 3 || true
fi

# ---------------------------------------------------------------- eval (SAME harness as P1)
for out in "${RUN_LIST[@]}"; do
  name=$(basename "$out")
  echo "== EVAL $name  ($(date +%H:%M:%S)) =="
  $PY "$EXP/eval_temporal.py" --run "$out" --sets E19a1k --embs "$TRAIN_EMBS" \
      --probe_embs "human_smpl,$PROBE_ONLY,unitree_h1,booster_t1,fourier_gr3" \
      > "$out/eval.log" 2>&1 || echo "  eval_temporal FAILED — see $out/eval.log"
  $PY "$SP/p1c_motion_metrics.py" "$out" > "$out/motion.log" 2>&1 \
      || echo "  motion metrics FAILED — see $out/motion.log"
  sed -n '/== motion energy/,$p' "$out/motion.log"
  if [ "${P1C_SKIP_VIDEO:-0}" != 1 ]; then
    P1C_RUN="$out" P1C_OUT="$out/mp4_fixed" P1C_LABEL="$name" \
      $PY "$SP/p1_fkfix.py" > "$out/fkfix.log" 2>&1 \
      || echo "  video FAILED — see $out/fkfix.log"
    grep '^MP4:' "$out/fkfix.log" || true
  fi
done

# ---------------------------------------------------------------- verdict table
# Re-measure the two comparison rows with the SAME script so every row of the table is
# produced by identical code on identical windows:
#   P1_oat_z_0805        the P1 baseline (global pooling, CE only)
#   P1b_motion_vel_0806  P1b's best LOSS-ONLY fix (global pooling + velocity loss)
# The verdict this table answers: does keeping the time axis beat the best loss-only
# fix, AT MATCHED BITS?
BASELINES=("$RUNS/P1_oat_z_0805" "$RUNS/P1b_motion_vel_0806")
for b in "${BASELINES[@]}"; do
  [ -d "$b" ] || { echo "WARN baseline missing: $b"; continue; }
  $PY "$SP/p1c_motion_metrics.py" "$b" > "$b/motion_p1c.log" 2>&1 \
    || echo "  baseline motion metrics FAILED — see $b/motion_p1c.log"
done
echo
echo "== P1c VERDICT TABLE (P1 baseline + P1b best loss-only fix + the P1c arms) =="
$PY "$SP/p1c_table.py" "${BASELINES[@]}" "${RUN_LIST[@]}" \
    | tee "$RUNS/P1c_verdict_${TAG}.txt"
touch "$RUNS/P1C_DONE"
echo
echo "P1c DONE. Per-run JSON: <run>/eval_temporal/{results,motion_energy}.json"
echo "VIDEOS TO LOOK AT WITH YOUR OWN EYES (数字好但动画飘没价值):"
for out in "${RUN_LIST[@]}"; do echo "  $out/mp4_fixed/"; done
echo "Verdict table: $RUNS/P1c_verdict_${TAG}.txt"
