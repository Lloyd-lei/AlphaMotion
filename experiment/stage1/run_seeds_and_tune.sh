#!/usr/bin/env bash
# Stage 1 extended validation:
#   Part 1: seeds 1 and 2 for all 6 ablation variants (seed 0 already in runs/)
#   Part 2: baseline hyper-parameter sweep (seed 0, three configs)
#
# Output layout:
#   runs/{name}/                             seed 0 (existing)
#   runs/{name}__seed{s}/                    seeds 1, 2
#   runs/baseline__{tune_name}/              baseline tune configs

set -euo pipefail

source /home/arenalabs/miniconda3/etc/profile.d/conda.sh
conda activate rot

cd "$(dirname "$0")"

COMMON_ARGS="--epochs 30 --batch-size 32 --data runs/kimodo_cache.npz --outdir runs"

run_wave() {
    # Args: label, model1, model2  — runs two models in parallel, waits
    local label="$1"; shift
    echo "[wave $label]"
    for cmd in "$@"; do
        bash -c "$cmd" &
    done
    wait
}

echo "============================================================"
echo "PART 1: seeds 1 and 2 for all 6 variants"
echo "============================================================"

for SEED in 1 2; do
    echo ""
    echo "### seed=$SEED"
    # Wave A: baseline + full (baseline is the slow one)
    run_wave "seed${SEED}-A" \
        "python train.py --model baseline --seed $SEED --run-tag seed${SEED} $COMMON_ARGS > runs/seed${SEED}_baseline.log 2>&1" \
        "python train.py --model full     --seed $SEED --run-tag seed${SEED} $COMMON_ARGS > runs/seed${SEED}_full.log 2>&1"
    # Wave B: axial_only + pair_static  (fastest)
    run_wave "seed${SEED}-B" \
        "python train.py --model axial_only  --seed $SEED --run-tag seed${SEED} $COMMON_ARGS > runs/seed${SEED}_axial_only.log 2>&1" \
        "python train.py --model pair_static --seed $SEED --run-tag seed${SEED} $COMMON_ARGS > runs/seed${SEED}_pair_static.log 2>&1"
    # Wave C: opm_only + triangle_only
    run_wave "seed${SEED}-C" \
        "python train.py --model opm_only      --seed $SEED --run-tag seed${SEED} $COMMON_ARGS > runs/seed${SEED}_opm_only.log 2>&1" \
        "python train.py --model triangle_only --seed $SEED --run-tag seed${SEED} $COMMON_ARGS > runs/seed${SEED}_triangle_only.log 2>&1"
done

echo ""
echo "============================================================"
echo "PART 2: baseline hyperparameter sweep (seed 0)"
echo "============================================================"

# Config 1: lower lr
python train.py --model baseline --seed 0 --run-tag lr1e4 --lr 1e-4 $COMMON_ARGS \
    > runs/baseline_lr1e4.log 2>&1 && echo "[baseline lr=1e-4 done]"

# Config 2: lower lr + warmup
python train.py --model baseline --seed 0 --run-tag lr1e4_warmup5 --lr 1e-4 --warmup-epochs 5 $COMMON_ARGS \
    > runs/baseline_lr1e4_warmup5.log 2>&1 && echo "[baseline lr=1e-4 warmup5 done]"

# Config 3: even smaller lr + longer warmup (conservative)
python train.py --model baseline --seed 0 --run-tag lr5e5_warmup5 --lr 5e-5 --warmup-epochs 5 $COMMON_ARGS \
    > runs/baseline_lr5e5_warmup5.log 2>&1 && echo "[baseline lr=5e-5 warmup5 done]"

echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
