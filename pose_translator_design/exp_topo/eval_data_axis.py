"""Evaluate the E21 DATA-axis series on the disjoint held-out AMASS cache.

Every E21_data{N} model trains on a nested subset of cache_xwbc_amass's 4000 clips,
so the 4000-clip model has no held-out data inside that cache. cache_amass_heldout is
600 clips drawn (default_rng(1)) from the same 13,281-clip all-embodiment intersection
MINUS those 4000 -- same rig, same specs (bit-identical), same conventions.

Uses dual_common's build_model / load_paired_data / geom -- the same code path as
eval_dual.py -- seed 0, CPU (the GPU is training).

Usage: python eval_data_axis.py [--runs ...] [--n_eval 4096]
"""
from __future__ import annotations
import os, sys, json, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import rot6d_to_matrix, geodesic_deg
from dual_common import load_paired_data, build_model, geom

CACHE = "/var/tmp/alphamotion/topo_exp/cache_amass_heldout"
RUNS = "/var/tmp/alphamotion/topo_exp/runs"
SERIES = [("E21_data500_0805", 500), ("E21_data1000_0805", 1000),
          ("E21_data2000_0805", 2000), ("E21_data4000_0805", 4000)]
# per-part groups over the TARGET skeleton (unitree_g1_29dof), exactly as specified
PARTS = {"legs": ("hip", "knee", "ankle", "toe"),
         "arms": ("shoulder", "elbow"),
         "wrists": ("wrist", "hand")}
DIRS = [("human_smpl", "unitree_g1_29dof"), ("unitree_h1", "unitree_g1_29dof")]


def pin_reach(data, ref_cache, device):
    """load_paired_data derives `reach` as the max root-relative joint distance IN THE
    CACHE, so a different clip set shifts it (~2% here). The models were trained with the
    training cache's value, so re-pin it: rescale the normalized positions and the cm
    conversion. Only the embodiments actually evaluated are touched."""
    import numpy as np
    from common import SkeletonSpec
    from train_global import build_global
    for name in sorted({n for d in DIRS for n in d}):
        d = np.load(f"{ref_cache}/{name}_spec.npz", allow_pickle=True)
        spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
        _, _, reach_ref = build_global(torch.load(f"{ref_cache}/{name}_paired_local6d.pt"),
                                       spec, device)
        old = data[name]["reach"]
        data[name]["pose"][..., 6:9] *= old / reach_ref
        data[name]["reach"] = reach_ref
        print(f"[reach] {name}: {old:.2f} -> {reach_ref:.2f} cm (pinned to {ref_cache})")
    return data


def part_masks(spec, device):
    names = [n.lower() for n in spec.joint_names]
    m = {p: torch.tensor([any(k in n for k in kws) for n in names], device=device)
         for p, kws in PARTS.items()}
    m["other"] = ~(m["legs"] | m["arms"] | m["wrists"])
    return m


@torch.no_grad()
def direction(model, data, src, tgt, idx, masks, chunk=512):
    """Mean geodesic (deg) and position error (cm), overall and per part.
    Also returns the per-frame overall rot error for the clip-block bootstrap."""
    rot_sum = {k: 0.0 for k in ["all"] + list(masks)}
    pos_sum = {k: 0.0 for k in ["all"] + list(masks)}
    per_frame, n = [], 0
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        B = len(sub)
        zp, zr = model.encode(data[src]["pose"][sub], geom(data, src, B))
        prot, ppos = model.decode(zp, zr, geom(data, tgt, B))
        gt = data[tgt]["pose"][sub]
        rot_j = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6]))   # [B,J] deg
        pos_j = (ppos - gt[..., 6:9]).norm(dim=-1) * data[tgt]["reach"]             # [B,J] cm
        rot_sum["all"] += float(rot_j.mean()) * B
        pos_sum["all"] += float(pos_j.mean()) * B
        for p, m in masks.items():
            rot_sum[p] += float(rot_j[:, m].mean()) * B
            pos_sum[p] += float(pos_j[:, m].mean()) * B
        per_frame.append(rot_j.mean(-1).cpu())
        n += B
    out = {k: {"rot_deg": round(rot_sum[k] / n, 2), "pos_cm": round(pos_sum[k] / n, 2)}
           for k in rot_sum}
    return out, torch.cat(per_frame)


def clip_bootstrap(per_frame, clip_of, n_boot=2000, seed=0):
    """95% CI on the mean rot error, resampling CLIPS (frames inside a clip are
    correlated, so a frame-level SEM would be far too optimistic).
    per_frame may be a difference between two models -> PAIRED CI on the delta,
    which is the statistic that matters for ranking models on the same clips."""
    g = torch.Generator().manual_seed(seed)
    uniq = torch.unique(clip_of)
    sums = torch.tensor([float(per_frame[clip_of == c].sum()) for c in uniq])
    cnts = torch.tensor([float((clip_of == c).sum()) for c in uniq])
    C = len(uniq)
    pick = torch.randint(0, C, (n_boot, C), generator=g)
    means = (sums[pick].sum(1) / cnts[pick].sum(1)).sort().values
    return {"ci95_lo": round(float(means[int(0.025 * n_boot)]), 2),
            "ci95_hi": round(float(means[int(0.975 * n_boot)]), 2),
            "n_clips_hit": int(C)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--n_eval", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=f"{RUNS}/E21_data_axis_heldout.json")
    ap.add_argument("--reach_from", default="/var/tmp/alphamotion/topo_exp/cache_amass_500",
                    help="pin `reach` to this cache's value ('' = use the eval cache's own)")
    args = ap.parse_args()
    device = args.device

    man = json.load(open(f"{args.cache}/manifest.json"))
    assert man["disjoint_from"]["overlap"] == 0, "cache is NOT disjoint from training clips"

    torch.manual_seed(0)                                  # split contract: seed -> load -> perm
    data, names, N = load_paired_data(args.cache, device)
    idx = torch.randperm(N)[:args.n_eval]
    print(f"[eval] cache={args.cache} rows={N} clips={man['clips']} n_eval={len(idx)}")
    if args.reach_from:
        data = pin_reach(data, args.reach_from, device)

    # map each eval frame to its source clip, for the block bootstrap
    seqs = json.load(open(f"{args.cache}/sequences.json"))
    bounds = torch.tensor([s["start"] for s in seqs])
    clip_of = torch.bucketize(idx, bounds, right=True) - 1

    masks = {t: part_masks(data[t]["spec"], device) for _, t in DIRS}
    for p, m in masks[DIRS[0][1]].items():
        print(f"  part {p:7s} J={int(m.sum())}")

    out = {"cache": args.cache, "heldout_clips": man["clips"], "rows": N,
           "n_eval": int(len(idx)), "seed": 0, "device": device,
           "reach_pinned_to": args.reach_from or None,
           "reach": {n: round(data[n]["reach"], 2) for d in DIRS for n in d}, "runs": {}}
    per_frame = {}
    for run, nclips in SERIES:
        rd = f"{RUNS}/{run}"
        if not (os.path.exists(f"{rd}/config.json") and os.path.exists(f"{rd}/model.pt")):
            print(f"[skip] {run}: not finished"); continue
        cfg = json.load(open(f"{rd}/config.json"))
        model = build_model(cfg, device)
        sd = torch.load(f"{rd}/model.pt", map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()
        r = {"train_clips": nclips, "train_cache": cfg["cache"],
             "state_dict_mismatch": bool(missing or unexpected)}
        for src, tgt in DIRS:
            res, pf = direction(model, data, src, tgt, idx, masks[tgt])
            res["boot_rot_deg"] = clip_bootstrap(pf, clip_of)
            r[f"{src}->{tgt}"] = res
            if (src, tgt) == DIRS[0]:
                per_frame[run] = pf
        out["runs"][run] = r
        h = r["human_smpl->unitree_g1_29dof"]
        b = h["boot_rot_deg"]
        print(f"[{run}] clips={nclips:>4}  human->g1  rot={h['all']['rot_deg']:.2f}deg "
              f"[{b['ci95_lo']:.2f},{b['ci95_hi']:.2f}]  pos={h['all']['pos_cm']:.2f}cm",
              flush=True)
        del model
    # PAIRED deltas vs the smallest run: same clips, so clip-difficulty variance cancels.
    # This, not the absolute CI, is what says whether the data axis moved anything.
    if len(per_frame) > 1:
        base = min(per_frame, key=lambda k: out["runs"][k]["train_clips"])
        out["paired_delta_vs_smallest"] = {"baseline": base, "direction": "->".join(DIRS[0]),
                                           "note": "negative = better than baseline", "deltas": {}}
        for run, pf in per_frame.items():
            if run == base:
                continue
            d = clip_bootstrap(pf - per_frame[base], clip_of)
            d["mean_delta_deg"] = round(float((pf - per_frame[base]).mean()), 2)
            d["significant"] = bool(d["ci95_lo"] > 0 or d["ci95_hi"] < 0)
            out["paired_delta_vs_smallest"]["deltas"][run] = d
            print(f"[paired] {run} vs {base}: {d['mean_delta_deg']:+.2f} deg "
                  f"[{d['ci95_lo']:+.2f},{d['ci95_hi']:+.2f}] "
                  f"{'SIGNIFICANT' if d['significant'] else 'not significant'}", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
