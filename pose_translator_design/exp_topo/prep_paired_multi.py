"""Build an N-embodiment strictly-parallel paired cache by DERIVING variants from an
existing 2-embodiment cache (no BVH reloading, runs in seconds):

  soma77       as-is from --src
  g1_30        as-is from --src
  soma_scaled  same local rotations, bone offsets rescaled (legs shorter, arms longer)
               -> a real new geometry, perfectly paired by construction (HuBE-style)
  soma_nohand  finger/hand joints dropped -> a real new TOPOLOGY (the leave-one-out
               holdout for E5); local rotations of kept joints are unchanged because
               dropped joints form leaf subtrees

Why not true soma_proportional here: its rest offsets differ PER ACTOR, but the
current pipeline conditions on one static spec per embodiment — one actor's spec
with another actor's frames breaks FK. Per-row geometry support is the right fix;
postponed (noted in APPENDIX open risks).

Manifest schema 2: {"schema": 2, "embodiments": [...], "rows": N, ...}
"""
from __future__ import annotations
import os, sys, json, shutil, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT, SkeletonSpec

FINGER_KW = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")
SCALE_RULES = (("Leg", 0.85), ("Foot", 0.85), ("Toe", 0.85),
               ("Arm", 1.15), ("Shoulder", 1.1), ("Spine", 1.05))


def load_spec(path):
    d = np.load(path, allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))


def save_spec(spec, path):
    np.savez(path, name=spec.name, parents=spec.parents, rest_offsets=spec.rest_offsets,
             norm_bonelen=spec.norm_bonelen,
             joint_names=np.array(spec.joint_names, dtype=object), height=spec.height)


def make_scaled(spec, name="soma_scaled"):
    off = spec.rest_offsets.copy()
    for j, jn in enumerate(spec.joint_names):
        for kw, s in SCALE_RULES:
            if kw in jn:
                off[j] = off[j] * s
                break
    return SkeletonSpec(name, spec.parents.copy(), off, list(spec.joint_names))


def make_nohand(spec, local6d, name="soma_nohand"):
    keep = [j for j, jn in enumerate(spec.joint_names) if not any(k in jn for k in FINGER_KW)]
    kset = set(keep)
    for j in keep:                                   # dropped joints must be leaf subtrees
        p = int(spec.parents[j])
        assert p < 0 or p in kset, f"kept joint {spec.joint_names[j]} has dropped parent"
    remap = {j: i for i, j in enumerate(keep)}
    parents = np.array([remap[int(spec.parents[j])] if spec.parents[j] >= 0 else -1 for j in keep])
    sub = SkeletonSpec(name, parents, spec.rest_offsets[keep], [spec.joint_names[j] for j in keep])
    return sub, local6d[:, keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{CACHE_ROOT}/cache_paired_fixed")
    ap.add_argument("--out", default=f"{CACHE_ROOT}/cache_paired_multi4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    man = json.load(open(f"{args.src}/manifest.json"))
    g1name = man["g1_name"]

    soma_spec = load_spec(f"{args.src}/soma77_spec.npz")
    soma6d = torch.load(f"{args.src}/soma77_paired_local6d.pt")

    # pass-through embodiments
    for name in ["soma77", g1name]:
        shutil.copy(f"{args.src}/{name}_spec.npz", f"{args.out}/{name}_spec.npz")
        shutil.copy(f"{args.src}/{name}_paired_local6d.pt", f"{args.out}/{name}_paired_local6d.pt")

    scaled = make_scaled(soma_spec)
    save_spec(scaled, f"{args.out}/soma_scaled_spec.npz")
    torch.save(soma6d, f"{args.out}/soma_scaled_paired_local6d.pt")     # same local rotations

    nohand_spec, nohand6d = make_nohand(soma_spec, soma6d)
    save_spec(nohand_spec, f"{args.out}/soma_nohand_spec.npz")
    torch.save(nohand6d.contiguous(), f"{args.out}/soma_nohand_paired_local6d.pt")

    emb = ["soma77", g1name, "soma_scaled", "soma_nohand"]
    json.dump({"schema": 2, "embodiments": emb, "rows": int(soma6d.shape[0]),
               "src": args.src, "g1_name": g1name},
              open(f"{args.out}/manifest.json", "w"), indent=2)
    if os.path.exists(f"{args.src}/sequences.json"):
        shutil.copy(f"{args.src}/sequences.json", f"{args.out}/sequences.json")
    print(f"SAVED {args.out}: {emb} rows={soma6d.shape[0]} "
          f"(nohand J={nohand_spec.J}, scaled height={scaled.height:.1f} vs {soma_spec.height:.1f})")


if __name__ == "__main__":
    main()
