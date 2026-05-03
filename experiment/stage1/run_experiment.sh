#!/usr/bin/env bash
# Stage 1 full run: train baseline and motionformer in parallel on the same GPU,
# both for the same number of epochs and same hyperparameters.

set -euo pipefail

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-128}"
SEED="${SEED:-0}"
OUTDIR="${OUTDIR:-runs}"

mkdir -p "$OUTDIR"

source /home/arenalabs/miniconda3/etc/profile.d/conda.sh
conda activate rot

# Run both models concurrently — they comfortably fit on a 96GB Blackwell.
python train.py --model baseline      --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED" --outdir "$OUTDIR" \
    > "$OUTDIR/baseline.log" 2>&1 &
PID_BASE=$!
python train.py --model motionformer  --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED" --outdir "$OUTDIR" \
    > "$OUTDIR/motionformer.log" 2>&1 &
PID_MOTION=$!

echo "Launched baseline (pid=$PID_BASE) and motionformer (pid=$PID_MOTION)."
echo "Tail logs:  tail -f $OUTDIR/baseline.log $OUTDIR/motionformer.log"
echo "Waiting for both to finish..."

wait $PID_BASE && echo "[baseline] OK" || echo "[baseline] FAILED"
wait $PID_MOTION && echo "[motionformer] OK" || echo "[motionformer] FAILED"

echo ""
echo "=== Final summary ==="
python - <<'PY'
import json
from pathlib import Path
for name in ("baseline", "motionformer"):
    h = json.load(open(Path("runs") / name / "history.json"))
    last = h[-1]
    print(f"[{name}] final  train={last['train_loss']:.5f}  val={last['val_loss']:.5f}  "
          f"subspace={last.get('subspace_sim', float('nan')):.3f}  "
          f"frob={last.get('frobenius_align', float('nan')):.3f}")
PY
