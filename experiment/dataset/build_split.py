"""Build a 4-way split over the mixed dataset.

    train        : 80% of in-distribution
    val_id       : 10% of in-distribution  (mixed sources, seen subjects)
    val_subject  : all clips from held-out AMASS subjects (body-shape OOD)
    val_source   : all HY Motion clips    (source-distribution OOD)

The remaining 10% is reserved for final reporting (not used until Stage 1.5
results are frozen).

Output:
    splits.json   -> { "train": [int, ...], "val_id": [...], "val_subject": [...], "val_source": [...] }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixed", default="mixed_soma77.npz")
    p.add_argument("--out",   default="splits.json")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--val-id-frac", type=float, default=0.10)
    p.add_argument("--held-out-frac", type=float, default=0.10,
                   help="Reserved, not returned in splits.")
    p.add_argument("--amass-heldout-subjects", type=int, default=3,
                   help="Hold out this many AMASS subjects (by sample_name prefix) per subset.")
    args = p.parse_args()

    root = Path(__file__).parent
    d = np.load(root / args.mixed, allow_pickle=True)
    N = d["source_id"].shape[0]
    source_id = d["source_id"]
    subset = d["subset"]
    sample_name = d["sample_name"]

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------
    # val_source: everything tagged HY Motion (id=2)
    # ------------------------------------------------------------
    val_source = np.where(source_id == 2)[0].tolist()

    # ------------------------------------------------------------
    # val_subject: for each AMASS subset, hold out a few "subjects".
    # AMASS naming: subset folders like "s1", "s2" etc. We treat the
    # `subset` column (within AMASS) as the subject id.
    # ------------------------------------------------------------
    val_subject = []
    amass_mask = (source_id == 0)
    amass_idx = np.where(amass_mask)[0]
    per_subset_subjects = defaultdict(set)
    for i in amass_idx:
        per_subset_subjects[str(subset[i])].add(str(sample_name[i]).split("_")[0])

    held_out_keys = set()
    for subset_name, names in per_subset_subjects.items():
        names = sorted(names)
        if len(names) <= args.amass_heldout_subjects:
            continue
        hold = rng.choice(names, args.amass_heldout_subjects, replace=False).tolist()
        for h in hold:
            held_out_keys.add((subset_name, h))

    for i in amass_idx:
        key = (str(subset[i]), str(sample_name[i]).split("_")[0])
        if key in held_out_keys:
            val_subject.append(int(i))

    # ------------------------------------------------------------
    # train + val_id + held_out over the remaining clips
    # ------------------------------------------------------------
    reserved = set(val_source) | set(val_subject)
    remaining = np.array([i for i in range(N) if i not in reserved])
    rng.shuffle(remaining)

    n_rem = len(remaining)
    n_heldout = int(n_rem * args.held_out_frac)
    n_val_id = int(n_rem * args.val_id_frac)

    held_out = remaining[:n_heldout].tolist()
    val_id   = remaining[n_heldout:n_heldout + n_val_id].tolist()
    train    = remaining[n_heldout + n_val_id:].tolist()

    splits = {
        "train":       sorted(train),
        "val_id":      sorted(val_id),
        "val_subject": sorted(val_subject),
        "val_source":  sorted(val_source),
        "held_out":    sorted(held_out),
    }

    print(f"Total clips: {N}")
    for k, v in splits.items():
        counts = defaultdict(int)
        for i in v:
            counts[f"src{int(source_id[i])}"] += 1
        src_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {k:<12} {len(v):>6}  by source: {src_str}")

    out_path = root / args.out
    with open(out_path, "w") as f:
        json.dump(splits, f)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
