"""Aggregate seed-3 ablation + baseline hyperparameter sweep results.

Reads the runs/ directory looking for:
    runs/{variant}/                 seed 0
    runs/{variant}__seed1/          seed 1
    runs/{variant}__seed2/          seed 2
    runs/baseline__{tune_tag}/      baseline tuning configs

For each variant, produces mean ± std over 3 seeds on:
    - final val_loss, val_kc_loss, train_loss
    - steps-to-reach val_loss thresholds

For baseline tuning, produces a separate table of seed-0 results under
different hyperparameters.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RUNS = Path(__file__).parent / "runs"

VARIANTS = ["baseline", "axial_only", "pair_static", "triangle_only", "opm_only", "full"]
SEEDS = [0, 1, 2]
THRESHOLDS = [0.5, 0.3, 0.2, 0.15, 0.1]
TUNE_TAGS = ["lr1e4", "lr1e4_warmup5", "lr5e5_warmup5"]


def run_dir(variant: str, seed: int) -> Path:
    if seed == 0:
        return RUNS / variant
    return RUNS / f"{variant}__seed{seed}"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.load(open(path))


def collect_variant(variant: str) -> dict:
    rows = []
    for s in SEEDS:
        p = run_dir(variant, s) / "history.json"
        data = load(p)
        if data is None:
            continue
        last = data["history"][-1]
        rows.append({
            "seed": s,
            "val": last["val_loss"],
            "val_kc": last["val_kc_loss"],
            "train": last["train_loss"],
            **{f"steps_{t}": data["target_steps"].get(str(t)) for t in THRESHOLDS},
        })
    return {"variant": variant, "rows": rows}


def mean_std(xs: list[float | None]) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def fmt(m: float | None, s: float | None, n: int, width: int = 8, prec: int = 3) -> str:
    if m is None:
        return "—".rjust(width)
    if n < 2:
        return f"{m:>{width}.{prec}f}"
    return f"{m:>{max(1,width-3)}.{prec}f}±{s:.{prec}f}"


def print_seed_table():
    print()
    print("=" * 95)
    print(f"{'variant':<15} {'n_seed':>6}  {'val (ID)':>14}  {'val_kc (OOD)':>14}  {'train':>14}  {'Δ val_kc vs base':>18}")
    print("-" * 95)

    # Collect baseline val_kc first
    base_rows = collect_variant("baseline")["rows"]
    base_vals_kc = [r["val_kc"] for r in base_rows]
    base_mean_kc, _ = mean_std(base_vals_kc)

    for v in VARIANTS:
        pack = collect_variant(v)
        rows = pack["rows"]
        n = len(rows)
        val_m, val_s = mean_std([r["val"] for r in rows])
        kc_m, kc_s = mean_std([r["val_kc"] for r in rows])
        tr_m, tr_s = mean_std([r["train"] for r in rows])
        if kc_m is not None and base_mean_kc is not None and v != "baseline":
            delta = f"{(kc_m - base_mean_kc) / base_mean_kc * 100:+.1f}%"
        else:
            delta = "—"
        print(
            f"{v:<15} {n:>6}  {fmt(val_m, val_s, n, 14, 3)}  "
            f"{fmt(kc_m, kc_s, n, 14, 3)}  {fmt(tr_m, tr_s, n, 14, 3)}  {delta:>18}"
        )
    print("=" * 95)


def print_step_table():
    print()
    print(f"{'variant':<15} "
          + "  ".join([f"{'steps≤'+str(t):>14}" for t in THRESHOLDS]))
    print("-" * (15 + 2 + 16 * len(THRESHOLDS)))
    for v in VARIANTS:
        pack = collect_variant(v)
        rows = pack["rows"]
        n = len(rows)
        cells = []
        for t in THRESHOLDS:
            vals = [r[f"steps_{t}"] for r in rows]
            m, s = mean_std(vals)
            if m is None:
                cells.append(f"{'never':>14}")
            else:
                cells.append(f"{fmt(m, s, n, 14, 0)}")
        print(f"{v:<15} " + "  ".join(cells))


def print_tuning_table():
    print()
    print("=" * 70)
    print("Baseline hyperparameter sweep (seed=0)")
    print("-" * 70)
    # Include seed-0 default baseline as reference
    print(f"{'config':<22} {'val':>10} {'val_kc':>10} {'train':>10}")
    data = load(RUNS / "baseline" / "history.json")
    if data is not None:
        last = data["history"][-1]
        print(f"{'default (lr=3e-4)':<22} "
              f"{last['val_loss']:>10.3f} {last['val_kc_loss']:>10.3f} {last['train_loss']:>10.3f}")
    for tag in TUNE_TAGS:
        p = RUNS / f"baseline__{tag}" / "history.json"
        data = load(p)
        if data is None:
            print(f"{tag:<22} (not found)")
            continue
        last = data["history"][-1]
        print(f"{tag:<22} "
              f"{last['val_loss']:>10.3f} {last['val_kc_loss']:>10.3f} {last['train_loss']:>10.3f}")
    print("=" * 70)


if __name__ == "__main__":
    print_seed_table()
    print_step_table()
    print_tuning_table()
