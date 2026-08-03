"""Cache single-frame per-joint SE3 for multiple real topologies.

SOMA-77 (human, BVH) and G1 (robot, CSV) -> local_rot6d frames + SkeletonSpec.
Everything cached under /var/tmp (home partition is full).
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE_ROOT, SkeletonSpec, spec_from_bvh_skeleton,
                    spec_from_ardy_skeleton, matrix_to_rot6d, fk_global)

BONES = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/wbc_expert/data/raw/bones_seed_full'


def sample_frames(T, n, rng):
    if T <= n:
        return np.arange(T)
    return np.sort(rng.choice(T, size=n, replace=False))


def save_spec(spec: SkeletonSpec, path):
    np.savez(path, name=spec.name, parents=spec.parents,
             rest_offsets=spec.rest_offsets, norm_bonelen=spec.norm_bonelen,
             joint_names=np.array(spec.joint_names, dtype=object), height=spec.height)


def prep_soma(n_motions, frames_per_motion, out_dir, seed=0, variant="soma_uniform", tag="soma77"):
    from ardy.data_processing.skeleton import load_bvh_animation, Skeleton
    rng = np.random.default_rng(seed)
    sessions = sorted(os.listdir(f"{BONES}/{variant}/bvh"))
    bvhs = []
    for s in sessions:
        bvhs += sorted(glob.glob(f"{BONES}/{variant}/bvh/{s}/*.bvh"))
    rng.shuffle(bvhs)
    bvhs = bvhs[:n_motions]
    print(f"[soma] {len(bvhs)} motions")

    spec = None
    all_local6d = []
    t0 = time.time()
    for i, f in enumerate(bvhs):
        try:
            skl = Skeleton(); skl.load_from_bvh(f)
            if spec is None:
                spec = spec_from_bvh_skeleton(skl, tag)
                spec.order_root_first()
            root_trans, joint_rots = load_bvh_animation(f, skl)   # [T,3],[T,J,3,3]
            jr = torch.as_tensor(np.asarray(joint_rots), dtype=torch.float32)
            if jr.shape[1] != spec.J:
                continue
            idx = sample_frames(jr.shape[0], frames_per_motion, rng)
            local6d = matrix_to_rot6d(jr[idx])                    # [n,J,6]
            all_local6d.append(local6d)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {repr(e)[:80]}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(bvhs)}  ({time.time()-t0:.1f}s)")
    local6d = torch.cat(all_local6d, 0)                           # [N,J,6]
    print(f"[soma] frames={local6d.shape[0]} J={spec.J} time={time.time()-t0:.1f}s")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(local6d, f"{out_dir}/{tag}_local6d.pt")
    save_spec(spec, f"{out_dir}/{tag}_spec.npz")
    return spec, local6d.shape[0]


def prep_g1(n_motions, frames_per_motion, out_dir, seed=0):
    from ardy.data_processing.skeleton import load_csv_g1_animation, Skeleton, SkeletonG1
    from ardy.skeleton.definitions import G1Skeleton34
    rng = np.random.default_rng(seed + 1)
    sessions = sorted(os.listdir(f"{BONES}/g1/csv"))
    csvs = []
    for s in sessions:
        csvs += sorted(glob.glob(f"{BONES}/g1/csv/{s}/*.csv"))
    rng.shuffle(csvs)
    csvs = csvs[:n_motions]
    print(f"[g1] {len(csvs)} motions")

    # Build a Skeleton for the CSV loader from the default G1 xml.
    from ardy.data_processing.skeleton import _DEFAULT_G1_XML
    skl = SkeletonG1(); skl.load_from_xml(_DEFAULT_G1_XML)
    spec_ardy = spec_from_ardy_skeleton(G1Skeleton34(), "g1_34")

    all_local6d = []
    spec = None
    t0 = time.time()
    for i, f in enumerate(csvs):
        try:
            root_trans, joint_rots = load_csv_g1_animation(f, skl)
            jr = torch.as_tensor(np.asarray(joint_rots), dtype=torch.float32)
            if spec is None:
                # build spec from the CSV-loader skeleton to match joint ordering
                spec = spec_from_bvh_skeleton(skl, "g1_%d" % jr.shape[1])
                spec.order_root_first()
            if jr.shape[1] != spec.J:
                continue
            idx = sample_frames(jr.shape[0], frames_per_motion, rng)
            all_local6d.append(matrix_to_rot6d(jr[idx]))
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {repr(e)[:100]}")
            if i > 5 and not all_local6d:
                print("[g1] too many failures early, aborting g1 prep")
                return None, 0
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(csvs)}  ({time.time()-t0:.1f}s)")
    if not all_local6d:
        return None, 0
    local6d = torch.cat(all_local6d, 0)
    print(f"[g1] frames={local6d.shape[0]} J={spec.J} time={time.time()-t0:.1f}s")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(local6d, f"{out_dir}/{spec.name}_local6d.pt")
    save_spec(spec, f"{out_dir}/{spec.name}_spec.npz")
    return spec, local6d.shape[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["soma", "g1", "both"])
    ap.add_argument("--n_motions", type=int, default=300)
    ap.add_argument("--frames_per_motion", type=int, default=30)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    if args.which in ("soma", "both"):
        _, n = prep_soma(args.n_motions, args.frames_per_motion, args.out)
        summary["soma77"] = n
    if args.which in ("g1", "both"):
        _, n = prep_g1(args.n_motions, args.frames_per_motion, args.out)
        summary["g1"] = n
    print("SUMMARY", json.dumps(summary))
