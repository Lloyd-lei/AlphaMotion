#!/usr/bin/env python
"""Merge BABEL frame/sequence labels onto the local raw AMASS sequences.

BABEL is NOT downloadable here (license-gated). Download `babel_v1.0_release`
from https://babel.is.tue.mpg.de/ , unzip to  data/babel/  , then run:

    python data/babel_merge.py --babel-dir data/babel --amass-dir data/amass \
                               --out data/babel_labels.json

Output  data/babel_labels.json  maps each AMASS sequence (relpath under
data/amass) -> {dur, babel_sid, seq_labels[], frame_labels[]} where each
frame label = {start_t, end_t, act_cat[], proc_label}. The frame_labels are
the per-frame, time-aligned action boundaries §3 needs as ground truth.

Also prints coverage per AMASS sub-dataset so we know exactly how much of the
8 local subsets BABEL actually annotates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# BABEL feat_p dataset folders -> our extracted data/amass/ folder names.
# (BABEL uses the official AMASS names; our extracted dirs already match.)
KNOWN_SUBSETS = {
    "BioMotionLab_NTroje", "BMLmovi", "CMU", "DanceDB",
    "EKUT", "KIT", "MPI_HDM05", "TotalCapture",
}


def norm_feat_p(feat_p: str) -> str:
    """BABEL feat_p = '<alias>/<AMASS_dir>/<subject>/.../<stem>_poses.npz'.
    Local key (relpath under data/amass) = '<AMASS_dir>/<subject>/.../stem'.
    So drop the leading BABEL alias component."""
    p = feat_p.replace("\\", "/").lstrip("./")
    parts = p.split("/")
    return "/".join(parts[1:]) if len(parts) > 1 else p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--babel-dir", default="data/babel")
    ap.add_argument("--amass-dir", default="data/amass")
    ap.add_argument("--out", default="data/babel_labels.json")
    args = ap.parse_args()

    babel_dir = Path(args.babel_dir)
    amass_dir = Path(args.amass_dir)

    # index the local AMASS sequences by their 'Dataset/.../stem.npz' key
    amass_index: dict[str, str] = {}
    per_subset_total: Counter = Counter()
    for npz in amass_dir.rglob("*_poses.npz"):
        rel = npz.relative_to(amass_dir).as_posix()
        amass_index[rel] = rel
        per_subset_total[rel.split("/")[0]] += 1
    print(f"local AMASS sequences found: {len(amass_index)}")
    for s, n in sorted(per_subset_total.items()):
        print(f"  {s:22s} {n}")

    if not babel_dir.exists():
        print(f"\n[BABEL not present] put babel_v1.0_release under {babel_dir} "
              f"then re-run. (AMASS side verified above — script is ready.)")
        return 0

    # NOTE: skip test.json — BABEL withholds test-split labels (proc_label /
    # act_cat are null), they'd pollute the merge and are useless for §3.
    json_files = sorted(p for p in babel_dir.rglob("*.json")
                        if "__MACOSX" not in p.parts
                        and not p.name.startswith("._")
                        and p.name != "test.json")
    if not json_files:
        print(f"\n[no .json under {babel_dir}] check the unzip path.")
        return 1

    merged: dict[str, dict] = {}
    matched = 0
    unmatched = 0
    per_subset_labeled: Counter = Counter()
    frame_label_seqs = 0

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  skip {jf.name}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        for sid, ann in data.items():
            if not isinstance(ann, dict) or "feat_p" not in ann:
                continue
            key = norm_feat_p(ann["feat_p"])
            rel = amass_index.get(key)
            if rel is None:
                unmatched += 1
                continue
            matched += 1

            seq_labels = []
            sa = ann.get("seq_ann") or {}
            for lab in (sa.get("labels") or []):
                seq_labels.append({
                    "act_cat": lab.get("act_cat"),
                    "proc_label": lab.get("proc_label") or lab.get("raw_label"),
                })

            frame_labels = []
            fa = ann.get("frame_ann")
            if isinstance(fa, dict):
                for lab in (fa.get("labels") or []):
                    if lab.get("start_t") is None:
                        continue
                    frame_labels.append({
                        "start_t": lab.get("start_t"),
                        "end_t": lab.get("end_t"),
                        "act_cat": lab.get("act_cat"),
                        "proc_label": lab.get("proc_label") or lab.get("raw_label"),
                    })
            cand = {
                "dur": ann.get("dur"),
                "babel_sid": sid,
                "seq_labels": seq_labels,
                "frame_labels": frame_labels,
            }
            # keep the richest entry for this sequence: prefer the one with
            # the most non-null-labelled frame segments (a real train/val
            # entry beats a null/extra one).
            def _quality(e):
                fl = e["frame_labels"]
                named = sum(1 for x in fl if x.get("proc_label"))
                return (named, len(fl),
                        sum(1 for s in e["seq_labels"] if s.get("proc_label")))
            prev = merged.get(rel)
            if prev is None or _quality(cand) > _quality(prev):
                merged[rel] = cand

    Path(args.out).write_text(json.dumps(merged, indent=1, ensure_ascii=False))

    # stats computed from the FINAL merged dict (one entry per sequence)
    seqs = len(merged)
    with_boundaries = sum(1 for e in merged.values() if e["frame_labels"])
    with_named = sum(1 for e in merged.values()
                     if any(x.get("proc_label") for x in e["frame_labels"]))
    for rel, e in merged.items():
        if e["frame_labels"]:
            per_subset_labeled[rel.split("/")[0]] += 1

    print(f"\nBABEL feat_p scanned (raw)            : {matched + unmatched}")
    print(f"  matched to local AMASS             : {matched}")
    print(f"  no local file (subsets we lack)    : {unmatched}")
    print(f"unique local sequences with BABEL     : {seqs}")
    print(f"  with frame-level boundaries        : {with_boundaries}")
    print(f"  with NAMED frame labels            : {with_named}  (§3 truth)")
    print(f"\nframe-boundary coverage per local subset:")
    for s in sorted(per_subset_total):
        tot = per_subset_total[s]
        lab = per_subset_labeled.get(s, 0)
        pct = (100.0 * lab / tot) if tot else 0.0
        print(f"  {s:22s} {lab:5d} / {tot:5d}  ({pct:4.1f}%)")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
