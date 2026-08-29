"""Combine AMASS + Kimodo + HY Motion into a single training npz with
source_id tags. Supports target sample count via uniform sub-sampling per
source.

Output:
    experiment/dataset/mixed_soma77.npz
        joints_world     [N, T, 77, 3]
        local_rot_mats   [N, T, 77, 3, 3]
        global_rot_mats  [N, T, 77, 3, 3]
        root_positions   [N, T, 3]
        source_id        [N]       int    0=amass 1=kimodo 2=hy
        source_name      [N]       str
        subset           [N]       str    e.g. "CMU", "kimodo-prompt-42"
        sample_name      [N]       str
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SOURCE_IDS = {"amass": 0, "kimodo": 1, "hy_motion": 2, "100style": 3}


def sample_indices(total: int, target: int, seed: int = 0) -> np.ndarray:
    """Uniform random indices without replacement; if target >= total return all."""
    if target >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=target, replace=False)


def load_amass_subset(npz_path: Path, target: int, seed: int = 0) -> dict:
    from action_labels import label_row
    d = np.load(npz_path, allow_pickle=True)
    N = d["joints_world"].shape[0]
    idx = sample_indices(N, target, seed)
    meta = d["meta"][idx]
    stem = npz_path.stem  # e.g. "cmu_soma77"
    subsets = np.array([m["subset"] for m in meta], dtype=object)
    names   = np.array([m["sample_name"] for m in meta], dtype=object)
    labels  = np.array(
        [label_row("amass", stem, s, n) for s, n in zip(subsets, names)],
        dtype=object)
    return {
        "joints_world":    d["joints_world"][idx],
        "local_rot_mats":  d["local_rot_mats"][idx],
        "global_rot_mats": d["global_rot_mats"][idx],
        "root_positions":  d["root_positions"][idx],
        "source_name":     np.array(["amass"] * len(idx), dtype=object),
        "source_id":       np.full(len(idx), SOURCE_IDS["amass"], dtype=np.int32),
        "subset":          subsets,
        "sample_name":     names,
        "action_label":    labels,
    }


def load_kimodo(npz_path: Path, target: int, seed: int = 0) -> dict:
    from action_labels import label_row
    d = np.load(npz_path, allow_pickle=True)
    N = d["joints_world"].shape[0]
    idx = sample_indices(N, target, seed)
    prompts = d["prompts"][idx] if "prompts" in d else np.array([""] * len(idx))
    names   = np.array([str(p) for p in prompts], dtype=object)
    labels  = np.array(
        [label_row("kimodo", "kimodo", "kimodo", n) for n in names],
        dtype=object)
    return {
        "joints_world":    d["joints_world"][idx].astype(np.float32),
        "local_rot_mats":  d["local_rot_mats"][idx].astype(np.float32),
        "global_rot_mats": d["global_rot_mats"][idx].astype(np.float32),
        "root_positions":  d.get("root_positions_world",
                                  d["root_positions"])[idx].astype(np.float32),
        "source_name":     np.array(["kimodo"] * len(idx), dtype=object),
        "source_id":       np.full(len(idx), SOURCE_IDS["kimodo"], dtype=np.int32),
        "subset":          np.array(["kimodo"] * len(idx), dtype=object),
        "sample_name":     names,
        "action_label":    labels,
    }


def load_hy_motion(dir_path: Path, target: int, seed: int = 0,
                    clip_len: int = 90, fps: float = 30.0) -> dict | None:
    """HY Motion files use SMPL-H axis-angle but are stored in Y-up natively
    (unlike AMASS, whose raw files are Z-up). So we run FK without any
    coordinate fix — HY Motion trans already sits at Y≈1.1 (standing height)."""
    import sys
    sys.path.insert(0, str(dir_path.parent))
    from prepare_amass import axis_angle_to_matrix, SMPLH_TO_SOMA
    from action_labels import label_row
    sys.path.insert(0, str(dir_path.parent.parent / "stage1"))
    from soma_skeleton import build_parent_index, JOINT_INDEX, compute_bone_offsets

    km = np.load(dir_path.parent / "kimodo" / "kimodo_200.npz", allow_pickle=True)
    bone_offsets = compute_bone_offsets(km["joints_world"], km["global_rot_mats"])
    parents = build_parent_index()

    all_j, all_l, all_g, all_r, all_name = [], [], [], [], []
    for fp in sorted(dir_path.glob("*.npz")):
        d = np.load(fp, allow_pickle=True)
        if "poses" not in d:
            continue
        poses = d["poses"]
        trans = d["trans"]
        T = poses.shape[0]
        if T < clip_len:
            continue
        aa_52 = poses[:, :156].reshape(T, 52, 3)
        R_52 = axis_angle_to_matrix(aa_52)
        J77 = 77
        lr = np.tile(np.eye(3), (T, J77, 1, 1)).astype(np.float32)
        for si, name in SMPLH_TO_SOMA.items():
            lr[:, JOINT_INDEX[name]] = R_52[:, si]
        gr = np.zeros_like(lr); gp = np.zeros((T, J77, 3), dtype=np.float32)
        for j in range(J77):
            p = parents[j]
            if p < 0:
                gr[:, j] = lr[:, j]
                gp[:, j] = trans.astype(np.float32)
            else:
                gr[:, j] = np.einsum("tij,tjk->tik", gr[:, p], lr[:, j])
                gp[:, j] = (gp[:, p]
                            + np.einsum("tij,j->ti", gr[:, p], bone_offsets[j]))

        n_clips = T // clip_len
        for c in range(n_clips):
            s, e = c * clip_len, (c + 1) * clip_len
            all_j.append(gp[s:e]); all_l.append(lr[s:e]); all_g.append(gr[s:e])
            all_r.append(trans[s:e].astype(np.float32))
            all_name.append(f"{fp.stem}_c{c}")

    if not all_j:
        return None
    N = len(all_j)
    idx = sample_indices(N, target, seed)
    names  = np.array([all_name[i] for i in idx], dtype=object)
    labels = np.array(
        [label_row("hy_motion", "hy_motion", "hy_motion", n) for n in names],
        dtype=object)
    return {
        "joints_world":    np.stack([all_j[i] for i in idx]),
        "local_rot_mats":  np.stack([all_l[i] for i in idx]),
        "global_rot_mats": np.stack([all_g[i] for i in idx]),
        "root_positions":  np.stack([all_r[i] for i in idx]),
        "source_name":     np.array(["hy_motion"] * len(idx), dtype=object),
        "source_id":       np.full(len(idx), SOURCE_IDS["hy_motion"], dtype=np.int32),
        "subset":          np.array(["hy_motion"] * len(idx), dtype=object),
        "sample_name":     names,
        "action_label":    labels,
    }


def concat_packs(packs: list[dict]) -> dict:
    out = {}
    for k in packs[0]:
        out[k] = np.concatenate([p[k] for p in packs], axis=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).parent))
    p.add_argument("--out",  default="mixed_soma77.npz")
    p.add_argument("--amass-per-subset", type=int, default=1000,
                   help="Target clips per AMASS subset (uniform subsample).")
    p.add_argument("--kimodo-target", type=int, default=2000)
    p.add_argument("--hy-target",     type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    packs = []

    # AMASS: 8 subsets, each capped at amass-per-subset
    amass_files = sorted((root / "amass").glob("*_soma77.npz"))
    for f in amass_files:
        pack = load_amass_subset(f, args.amass_per_subset, args.seed)
        print(f"  AMASS {f.stem}: {pack['joints_world'].shape[0]} clips")
        packs.append(pack)

    # Kimodo
    kimodo_path = root / "kimodo" / "kimodo_2000.npz"
    if kimodo_path.exists():
        pack = load_kimodo(kimodo_path, args.kimodo_target, args.seed)
        print(f"  Kimodo: {pack['joints_world'].shape[0]} clips")
        packs.append(pack)

    # HY Motion
    hy = load_hy_motion(root / "hy_motion", args.hy_target, args.seed)
    if hy is not None:
        print(f"  HY Motion: {hy['joints_world'].shape[0]} clips")
        packs.append(hy)

    merged = concat_packs(packs)
    N_total = merged["joints_world"].shape[0]
    print(f"\nTotal: {N_total} clips")
    for sid, name in {v: k for k, v in SOURCE_IDS.items()}.items():
        c = int((merged["source_id"] == sid).sum())
        print(f"  {name:<12} id={sid}  count={c}")

    out_path = root / args.out
    np.savez(out_path, **merged)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
