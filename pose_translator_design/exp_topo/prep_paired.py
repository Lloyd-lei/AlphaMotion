"""Build PAIRED SOMA<->G1 frames (same motion, frame-aligned) for cross-embodiment
translation. SOMA (BVH) and G1 (CSV) of the same filename are the same motion via
GMR retargeting, at possibly different fps/length -> resample both to K frames by
relative time so row i of each cache is the SAME pose in two embodiments.
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT, spec_from_bvh_skeleton, matrix_to_rot6d

BONES = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/wbc_expert/data/raw/bones_seed_full'


def rel_time_idx(T, K):
    if T <= 1:
        return np.zeros(K, dtype=np.int64)
    return np.clip(np.round(np.linspace(0, T - 1, K)).astype(np.int64), 0, T - 1)


def save_spec(spec, path):
    np.savez(path, name=spec.name, parents=spec.parents, rest_offsets=spec.rest_offsets,
             norm_bonelen=spec.norm_bonelen,
             joint_names=np.array(spec.joint_names, dtype=object), height=spec.height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_files", type=int, default=1500)
    ap.add_argument("--frames_per_file", type=int, default=24)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/cache_paired")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from ardy.data_processing.skeleton import (load_bvh_animation, load_csv_g1_animation,
                                               Skeleton, SkeletonG1, _DEFAULT_G1_XML)
    rng = np.random.default_rng(0)

    sessions = sorted(os.listdir(f"{BONES}/soma_uniform/bvh"))
    pairs = []
    for s in sessions:
        soma = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{BONES}/soma_uniform/bvh/{s}/*.bvh")}
        g1 = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{BONES}/g1/csv/{s}/*.csv")}
        for k in soma.keys() & g1.keys():
            pairs.append((soma[k], g1[k]))
    rng.shuffle(pairs)
    pairs = pairs[:args.n_files]
    print(f"[paired] {len(pairs)} matched files")

    g1_skl = SkeletonG1(); g1_skl.load_from_xml(_DEFAULT_G1_XML)
    soma_spec = g1_spec = None
    soma_all, g1_all = [], []
    t0 = time.time()
    for i, (bvh, csv) in enumerate(pairs):
        try:
            skl = Skeleton(); skl.load_from_bvh(bvh)
            rt_s, jr_s = load_bvh_animation(bvh, skl)
            rt_g, jr_g = load_csv_g1_animation(csv, g1_skl)
            jr_s = torch.as_tensor(np.asarray(jr_s), dtype=torch.float32)  # [Ts,Js,3,3]
            jr_g = torch.as_tensor(np.asarray(jr_g), dtype=torch.float32)  # [Tg,Jg,3,3]
            if soma_spec is None:
                soma_spec = spec_from_bvh_skeleton(skl, "soma77"); soma_spec.order_root_first()
                g1_spec = spec_from_bvh_skeleton(g1_skl, "g1_%d" % jr_g.shape[1]); g1_spec.order_root_first()
            if jr_s.shape[1] != soma_spec.J or jr_g.shape[1] != g1_spec.J:
                continue
            K = args.frames_per_file
            si = rel_time_idx(jr_s.shape[0], K)
            gi = rel_time_idx(jr_g.shape[0], K)          # same relative-time grid -> aligned
            l6s = matrix_to_rot6d(jr_s[si])              # [K,Js,6]
            l6g = matrix_to_rot6d(jr_g[gi])
            if torch.isnan(l6s).any() or torch.isnan(l6g).any():
                continue
            soma_all.append(l6s); g1_all.append(l6g)
        except Exception as e:
            if i < 5:
                print("  skip", os.path.basename(bvh), repr(e)[:80])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pairs)} ({time.time()-t0:.0f}s)")
    soma = torch.cat(soma_all, 0); g1 = torch.cat(g1_all, 0)
    assert soma.shape[0] == g1.shape[0], "paired rows must match"
    print(f"[paired] rows={soma.shape[0]}  soma J={soma_spec.J}  g1 J={g1_spec.J}  ({time.time()-t0:.0f}s)")
    torch.save(soma, f"{args.out}/soma77_paired_local6d.pt")
    torch.save(g1, f"{args.out}/{g1_spec.name}_paired_local6d.pt")
    save_spec(soma_spec, f"{args.out}/soma77_spec.npz")
    save_spec(g1_spec, f"{args.out}/{g1_spec.name}_spec.npz")
    json.dump({"rows": int(soma.shape[0]), "soma_J": soma_spec.J, "g1_J": g1_spec.J,
               "g1_name": g1_spec.name}, open(f"{args.out}/manifest.json", "w"), indent=2)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
