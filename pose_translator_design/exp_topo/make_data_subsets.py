"""Carve clip-count subsets out of the AMASS paired cache for the DATA scaling axis.

Only the number of clips varies — same embodiments, same rig, same specs/dof, same
convention. (LAFAN and AMASS ship two DIFFERENT human rigs, so mixing them would
confound data volume with rest-frame convention; we stay inside the AMASS half.)

Usage: python make_data_subsets.py --sizes 500 1000 2000
(4000 = the full cache; symlinked rather than copied.)
"""
from __future__ import annotations
import os, sys, json, shutil, argparse
import numpy as np, torch

SRC = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass"
OUT = "/var/tmp/alphamotion/topo_exp/cache_amass_%d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000])
    args = ap.parse_args()

    man = json.load(open(f"{SRC}/manifest.json"))
    seqs = json.load(open(f"{SRC}/sequences.json"))
    names = man["embodiments"]
    print(f"source: {len(seqs)} clips, {man['rows']} rows, {len(names)} embodiments")

    # deterministic nested subsets: same rng, prefix-nested so bigger ⊃ smaller
    order = np.random.default_rng(0).permutation(len(seqs))
    pose = {n: torch.load(f"{SRC}/{n}_paired_local6d.pt") for n in names}

    for size in args.sizes:
        out = OUT % size
        os.makedirs(out, exist_ok=True)
        pick = sorted(order[:size].tolist())
        rows = np.concatenate([np.arange(seqs[i]["start"], seqs[i]["start"] + seqs[i]["len"]) for i in pick])
        idx = torch.as_tensor(rows, dtype=torch.long)
        newseq, cur = [], 0
        for i in pick:
            newseq.append({"name": seqs[i]["name"], "start": cur, "len": seqs[i]["len"]})
            cur += seqs[i]["len"]
        for n in names:
            torch.save(pose[n][idx].contiguous(), f"{out}/{n}_paired_local6d.pt")
            for suffix in ("_spec.npz", "_dof.npz"):
                src = f"{SRC}/{n}{suffix}"
                if os.path.exists(src):
                    shutil.copy(src, f"{out}/{n}{suffix}")
        json.dump({**man, "rows": int(cur), "clips": size, "subset_of": SRC},
                  open(f"{out}/manifest.json", "w"), indent=2)
        json.dump(newseq, open(f"{out}/sequences.json", "w"))
        print(f"  {size:>5} clips -> {cur:>9} rows  {out}")

    full = OUT % 4000
    if not os.path.exists(full):
        os.symlink(SRC, full)
        print(f"  {4000:>5} clips -> {man['rows']:>9} rows  {full} (symlink to full cache)")


if __name__ == "__main__":
    main()
