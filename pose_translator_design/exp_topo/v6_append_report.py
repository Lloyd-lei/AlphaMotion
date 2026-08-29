"""Append the V6-specific sections to runs/V6_verdict_0807.txt.

n3_eval_temporal.py measures every arm in the SAME regime (encoder sees the full
window, decoder sees the endpoints) — that is what makes V6 comparable with V5c at
all. But the whole V6 claim is about what happens at OTHER densities, and no V5-era
harness has a row for that. This script adds two:

  * BIT BUDGET, printed as an equality check rather than a claim.
  * THE MASK LADDER: key error as a function of how much the decoder was allowed to
    see (endpoints / 3 keyframes / random 50% / past half / everything), each next to
    the pure-interpolation floor for the SAME visible set. The gap between the two
    columns is what the 125 bits actually bought at that density.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                            # noqa: E402

ORDER = ("endpoints", "sparse3", "rand50", "prefix_half", "full")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{V.RUNS}/V6_verdict_0807.txt"
    runs = sys.argv[2].split(",") if len(sys.argv) > 2 else []
    L = ["", "=" * 100,
         "V6 APPENDIX — bit budget and the MASK LADDER (rows no V5-era harness has)",
         "=" * 100]
    L.append(f"{'arm':22s}{'bits/s':>10s}{'registers':>11s}{'levels':>12s}"
             f"{'params':>10s}  arch")
    for r in runs:
        rd = r if os.path.isabs(r) else f"{V.RUNS}/{r}"
        try:
            c = json.load(open(f"{rd}/config.json"))
        except Exception:
            continue
        b = c.get("bit_budget", {})
        L.append(f"{os.path.basename(rd):22s}{b.get('total_bits_per_s', 0):>10.2f}"
                 f"{b.get('n_path', 0):>7d}+{b.get('n_time', 0):<3d}"
                 f"{str(c.get('fsq_levels')):>12s}"
                 f"{c.get('n_params', 0)/1e6:>9.2f}M  {c.get('arch', 'v5')}"
                 + ("  residual_in" if c.get("residual_in") else "")
                 + (f"  frame_repr={c.get('frame_repr')}"
                    if c.get("frame_repr", "key") != "key" else ""))
    L.append("  P1 / V5 / V5c reference: 62.69 bit/s. Any arm off that number is not "
             "comparable and the")
    L.append("  trainer asserts on it, so a mismatch cannot reach this table silently.")

    L += ["", "-- MASK LADDER (V6 arms only; val split, 192 windows, full 8 registers) --",
          "   key_pos_cm = decoded error | interp = the SAME visible set interpolated "
          "with ZERO bits",
          "   the bits are worth exactly (interp - key_pos_cm) cm at each density."]
    any6 = False
    for r in runs:
        rd = r if os.path.isabs(r) else f"{V.RUNS}/{r}"
        try:
            c = json.load(open(f"{rd}/config.json"))
            if str(c.get("arch", "v5")).lower() != "v6":
                continue
            log = json.load(open(f"{rd}/eval_log.json"))
            ml = log[-1]["val"]["mask_ladder"]
        except Exception as e:
            continue
        any6 = True
        L.append(f"\n  {os.path.basename(rd)}  (step {log[-1]['step']})")
        L.append(f"    {'visible set':14s}{'key_pos_cm':>12s}{'interp_only':>13s}"
                 f"{'bits buy':>10s}{'energy':>9s}")
        for k in ORDER:
            if k not in ml:
                continue
            row = ml[k]
            L.append(f"    {k:14s}{row['key_pos_cm']:>12.3f}"
                     f"{row['interp_only_cm']:>13.3f}"
                     f"{row['interp_only_cm'] - row['key_pos_cm']:>10.3f}"
                     f"{row['energy_ratio']:>9.3f}")
    if not any6:
        L.append("  (no V6 arm finished)")
    L += ["",
          "READING IT: 'endpoints' is the row every other arm in the table above was",
          "measured in — V5c's only regime. The rows below it are densities V5c was",
          "never trained for and cannot be asked about. If 'bits buy' is ~0 at every",
          "density, the registers are decorative and V6 failed the same way V5c did;",
          "if it grows as the visible set thins, the tokens are carrying the interior.",
          "=" * 100]
    txt = "\n".join(L)
    print(txt)
    with open(out, "a") as f:
        f.write(txt + "\n")
    print(f"\nAPPENDED -> {out}")


if __name__ == "__main__":
    main()
