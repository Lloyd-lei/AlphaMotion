"""Comprehensive Stage 1.5 diagnostic suite.

Produces under runs/<model>/diag/ :
    01_training_curves.png        — loss evolution (train vs val splits)
    02_rotation_degrees.png       — geodesic rot in degrees over epochs
    03_pair_heatmap.png           — [77 x 77] learned pair-tensor L2 norm
    04_per_joint_error.png        — per-joint MSE bar chart by body part
    05_time_profile.png           — reconstruction error vs frame index
    06_mask_mode_comparison.png   — error per mask-mode (random/joint/kc/etc)
    07_style_pca.png              — PCA of pair tensors, coloured by source/subset
    08_style_tsne.png             — t-SNE version for cleaner clusters
    09_source_stratified.json     — numerical per-source breakdown of val_id

Run once per checkpoint:
    python diagnose_stage15.py --checkpoint runs/opm_only__mixed_full30/final.pt --model opm_only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import MixedMotionDataset, make_masked_batch, mixed_mode_masking
from motionformer import MotionFormerConfig, MotionFormer
from baseline import BaselineConfig, BaselineTransformer
from soma_skeleton import (
    build_parent_index, JOINT_NAMES, JOINT_GROUPS, rot6d_to_matrix,
    forward_kinematics_torch,
)
from train import MF_VARIANTS, compute_losses


# ----------------------------------------------------------------------------
# 1. Training curves
# ----------------------------------------------------------------------------


def plot_training_curves(model_name: str, history: list, out_dir: Path):
    splits = ["val_id", "val_subject", "val_source"]
    metrics = ["pos", "rot", "fk"]

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, metrics):
        ax.plot([h["epoch"] for h in history],
                [h["train"][metric] for h in history],
                "--", label="train", alpha=0.5, color="black")
        for split in splits:
            if split in history[0]["val"]:
                ax.plot([h["epoch"] for h in history],
                        [h["val"][split][metric] for h in history],
                        label=split, linewidth=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{metric} loss")
        ax.set_title(f"{metric} — {model_name}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "01_training_curves.png", dpi=100)
    plt.close(fig)

    # Rotation in degrees
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in splits:
        if split in history[0]["val"]:
            degs = [math.sqrt(h["val"][split]["rot"]) * 180 / math.pi for h in history]
            ax.plot([h["epoch"] for h in history], degs, label=split, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("rotation error (degrees)")
    ax.set_title(f"Rotation error over epochs — {model_name}")
    ax.axhline(y=5, color="gray", linestyle=":", alpha=0.5, label="~Kimodo quality")
    ax.axhline(y=10, color="gray", linestyle=":", alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "02_rotation_degrees.png", dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. Pair tensor heatmap (for MotionFormer variants only)
# ----------------------------------------------------------------------------


def plot_pair_heatmap(model, ds: MixedMotionDataset, device: str, out_dir: Path, n_samples: int = 16):
    if not hasattr(model, "_last_pair"):
        return
    model.eval()
    # Average pair tensor across n_samples to get stable estimate
    total = None
    count = 0
    with torch.no_grad():
        for idx in np.random.default_rng(0).choice(len(ds), n_samples, replace=False):
            sample = ds[int(idx)]["pos"].unsqueeze(0).to(device)
            zero_mask = torch.zeros(1, 90, 77, dtype=torch.bool, device=device)
            _ = model(sample, zero_mask)
            pair = model._last_pair
            if pair is None:
                return
            norm = pair.norm(dim=-1)[0].cpu().numpy()  # [J, J]
            total = norm if total is None else total + norm
            count += 1
    avg = total / count
    # symmetrize
    avg = 0.5 * (avg + avg.T)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(avg, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Pair tensor L2 norm (averaged over {n_samples} samples)")
    # Annotate key joints
    key_joints = [("Hips", 0), ("Chest", 3), ("Head", 6),
                  ("LeftHand", 14), ("RightHand", 41),
                  ("LeftLeg", 67), ("RightLeg", 72)]
    tick_pos = [i for _, i in key_joints]
    tick_lbl = [n for n, _ in key_joints]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbl, rotation=45, fontsize=8)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lbl, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_pair_heatmap.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Per-joint error heatmap
# ----------------------------------------------------------------------------


def plot_per_joint_error(model, ds, val_indices, device, out_dir, pm, ps, rm, rs, bo, parents):
    model.eval()
    # accumulate per-joint reconstruction error for random mask
    joint_err = np.zeros(77)
    counts = np.zeros(77)
    rng = torch.Generator(device=device).manual_seed(0)

    def collate(batch_list):
        out = {}
        for k in batch_list[0]:
            if isinstance(batch_list[0][k], int):
                out[k] = torch.tensor([b[k] for b in batch_list])
            else:
                out[k] = torch.stack([b[k] for b in batch_list])
        return out

    from torch.utils.data import Subset, DataLoader
    subset = Subset(ds, val_indices[:500])
    dl = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collate, num_workers=0)

    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, mask, _ = mixed_mode_masking(batch["pos"], mask_ratio=0.3, rng=rng)
            masked = batch["pos"].clone(); masked[mask] = 0
            pred = model(masked, mask)
            # per-joint error on masked
            diff_pos = (pred["pos"] - batch["pos"]) ** 2   # [B, T, J, 3]
            mask_f = mask.float()                          # [B, T, J]
            per_joint = (diff_pos.sum(-1) * mask_f).sum(dim=(0, 1)).cpu().numpy()  # [J]
            per_joint_cnt = mask_f.sum(dim=(0, 1)).cpu().numpy()
            joint_err += per_joint
            counts += per_joint_cnt

    avg_err = joint_err / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = []
    for name in JOINT_NAMES:
        if any(name in g for g in [JOINT_GROUPS["left_fingers"], JOINT_GROUPS["right_fingers"]]):
            bar_colors.append("tab:orange")
        elif any(name in g for g in [JOINT_GROUPS["left_leg"], JOINT_GROUPS["right_leg"]]):
            bar_colors.append("tab:green")
        elif any(name in g for g in [JOINT_GROUPS["left_arm"], JOINT_GROUPS["right_arm"]]):
            bar_colors.append("tab:blue")
        else:
            bar_colors.append("tab:gray")
    ax.bar(range(77), avg_err, color=bar_colors)
    ax.set_xticks(range(77))
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    ax.set_ylabel("avg masked position MSE")
    ax.set_title(f"Per-joint reconstruction error (random mask, 500 val_id samples)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="tab:blue",   label="arm"),
        Patch(color="tab:orange", label="fingers"),
        Patch(color="tab:green",  label="legs"),
        Patch(color="tab:gray",   label="spine/head"),
    ], fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "04_per_joint_error.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Time-wise error profile (answers "does error grow over time?")
# ----------------------------------------------------------------------------


def plot_time_profile(model, ds, val_indices, device, out_dir):
    from torch.utils.data import Subset, DataLoader
    def collate(batch_list):
        out = {}
        for k in batch_list[0]:
            if isinstance(batch_list[0][k], int):
                out[k] = torch.tensor([b[k] for b in batch_list])
            else:
                out[k] = torch.stack([b[k] for b in batch_list])
        return out

    subset = Subset(ds, val_indices[:500])
    dl = DataLoader(subset, batch_size=32, collate_fn=collate, num_workers=0)
    model.eval()
    time_err = np.zeros(90)
    counts = np.zeros(90)
    rng = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, mask, _ = mixed_mode_masking(batch["pos"], mask_ratio=0.3, rng=rng)
            masked = batch["pos"].clone(); masked[mask] = 0
            pred = model(masked, mask)
            diff = ((pred["pos"] - batch["pos"]) ** 2).sum(-1)  # [B, T, J]
            mask_f = mask.float()
            per_t = (diff * mask_f).sum(dim=(0, 2)).cpu().numpy()
            per_t_cnt = mask_f.sum(dim=(0, 2)).cpu().numpy()
            time_err += per_t
            counts += per_t_cnt
    avg = time_err / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(90), avg, linewidth=2)
    ax.set_xlabel("frame index (0-89, 90 frames = 3s)")
    ax.set_ylabel("avg masked position MSE")
    ax.set_title("Reconstruction error by frame index")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_time_profile.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Mask-mode comparison
# ----------------------------------------------------------------------------


def plot_mask_mode(model, ds, val_indices, device, out_dir, pm, ps, rm, rs, bo, parents, lambdas):
    from torch.utils.data import Subset, DataLoader
    def collate(batch_list):
        out = {}
        for k in batch_list[0]:
            if isinstance(batch_list[0][k], int):
                out[k] = torch.tensor([b[k] for b in batch_list])
            else:
                out[k] = torch.stack([b[k] for b in batch_list])
        return out
    subset = Subset(ds, val_indices[:300])
    dl = DataLoader(subset, batch_size=32, collate_fn=collate, num_workers=0)
    model.eval()
    modes = ["random", "joint", "time", "keyframe", "kinematic_chain"]
    results = {m: {"pos": 0, "rot": 0, "n": 0} for m in modes}
    with torch.no_grad():
        for mode in modes:
            rng = torch.Generator(device=device).manual_seed(0)
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, mask, _ = make_masked_batch(batch["pos"], mask_ratio=0.3,
                                                 mask_mode=mode, rng=rng)
                masked = batch["pos"].clone(); masked[mask] = 0
                pred = model(masked, mask)
                losses = compute_losses(pred, batch, mask, pm, ps, rm, rs, bo, parents, lambdas)
                results[mode]["pos"] += losses["pos"].item()
                results[mode]["rot"] += losses["rot"].item()
                results[mode]["n"] += 1
    for m in modes:
        n = results[m]["n"]
        results[m]["pos"] /= n
        results[m]["rot"] /= n
        results[m]["rot_deg"] = math.sqrt(results[m]["rot"]) * 180 / math.pi

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(len(modes))
    axes[0].bar(xs, [results[m]["pos"] for m in modes], color="tab:blue")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(modes, rotation=20)
    axes[0].set_ylabel("position MSE on masked")
    axes[0].set_title("Position reconstruction by mask mode")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(xs, [results[m]["rot_deg"] for m in modes], color="tab:orange")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(modes, rotation=20)
    axes[1].set_ylabel("rotation error (degrees)")
    axes[1].set_title("Rotation reconstruction by mask mode")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "06_mask_mode_comparison.png", dpi=110)
    plt.close(fig)
    return results


# ----------------------------------------------------------------------------
# 7 & 8. Style PCA / t-SNE
# ----------------------------------------------------------------------------


def plot_style_clustering(model, ds, device, out_dir, max_samples: int = 600):
    model.eval()
    embs = []
    labels_source = []
    labels_subset = []
    labels_action = []

    # Sample up to max_samples evenly spaced
    n = min(max_samples, len(ds))
    idx = np.linspace(0, len(ds) - 1, n, dtype=int)
    has_action = hasattr(ds, "action_label")
    with torch.no_grad():
        for i in idx:
            sample = ds[int(i)]["pos"].unsqueeze(0).to(device)
            zero_mask = torch.zeros(1, 90, 77, dtype=torch.bool, device=device)
            _ = model(sample, zero_mask)
            pair = model._last_pair
            if pair is None:
                return None
            emb = pair.flatten().cpu().numpy()
            embs.append(emb)
            labels_source.append(str(ds.source_name[i]))
            labels_subset.append(str(ds.subset[i]))
            labels_action.append(str(ds.action_label[i]) if has_action else "unknown")
    embs = np.stack(embs)
    # Mean-center to make cosine distances sensible
    embs_c = embs - embs.mean(axis=0, keepdims=True)

    # PCA to 2D
    pca = PCA(n_components=2)
    pca2 = pca.fit_transform(embs_c)
    # t-SNE to 2D
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, n // 5)),
                 init="pca", learning_rate="auto", random_state=0)
    tsne2 = tsne.fit_transform(embs_c)

    def _plot(coords, method_name, label_list, label_kind, out_name):
        fig, ax = plt.subplots(figsize=(9, 7))
        unique = sorted(set(label_list))
        cmap = plt.get_cmap("tab20" if len(unique) > 10 else "tab10")
        for i, lbl in enumerate(unique):
            m = np.array([l == lbl for l in label_list])
            ax.scatter(coords[m, 0], coords[m, 1], label=f"{lbl} (n={m.sum()})",
                       alpha=0.7, s=30, color=cmap(i % 20))
        ax.set_title(f"{method_name} of pair-tensor embeddings — coloured by {label_kind}")
        ax.legend(fontsize=7, loc="best", markerscale=0.8)
        ax.set_xlabel(f"{method_name} dim 1")
        ax.set_ylabel(f"{method_name} dim 2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=100)
        plt.close(fig)

    _plot(pca2, "PCA", labels_source, "source", "07_style_pca_source.png")
    _plot(pca2, "PCA", labels_subset, "subset", "07_style_pca_subset.png")
    _plot(pca2, "PCA", labels_action, "action", "07_style_pca_action.png")
    _plot(tsne2, "t-SNE", labels_source, "source", "08_style_tsne_source.png")
    _plot(tsne2, "t-SNE", labels_subset, "subset", "08_style_tsne_subset.png")
    _plot(tsne2, "t-SNE", labels_action, "action", "08_style_tsne_action.png")
    # Also a "Kimodo-only" PCA/tSNE coloured by action — cleaner than mixing
    # AMASS's 'other' noise with Kimodo's labelled clusters.
    k_mask = np.array([s == "kimodo" for s in labels_source])
    if k_mask.sum() > 20:
        _plot(pca2[k_mask], "PCA",
              [labels_action[i] for i in range(len(labels_action)) if k_mask[i]],
              "action (kimodo only)", "07_style_pca_action_kimodo.png")
        _plot(tsne2[k_mask], "t-SNE",
              [labels_action[i] for i in range(len(labels_action)) if k_mask[i]],
              "action (kimodo only)", "08_style_tsne_action_kimodo.png")
    return {
        "pca_var_explained": pca.explained_variance_ratio_.tolist(),
        "n_samples_embedded": n,
        "n_kimodo_in_sample": int(k_mask.sum()),
    }


# ----------------------------------------------------------------------------
# 9. Source-stratified eval
# ----------------------------------------------------------------------------


def source_stratified(model, ds, val_indices, device, out_dir, pm, ps, rm, rs, bo, parents, lambdas):
    from collections import defaultdict
    from torch.utils.data import Subset, DataLoader
    def collate(batch_list):
        out = {}
        for k in batch_list[0]:
            if isinstance(batch_list[0][k], int):
                out[k] = torch.tensor([b[k] for b in batch_list])
            else:
                out[k] = torch.stack([b[k] for b in batch_list])
        return out

    by_source = defaultdict(list)
    for i in val_indices:
        by_source[int(ds.source_id[i])].append(i)

    results = {}
    for src, idxs in by_source.items():
        subset = Subset(ds, idxs)
        dl = DataLoader(subset, batch_size=32, collate_fn=collate, num_workers=0)
        tally = {"pos": 0, "rot": 0, "fk": 0, "n": 0}
        rng = torch.Generator(device=device).manual_seed(0)
        model.eval()
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, mask, _ = mixed_mode_masking(batch["pos"], mask_ratio=0.3, rng=rng)
                masked = batch["pos"].clone(); masked[mask] = 0
                pred = model(masked, mask)
                losses = compute_losses(pred, batch, mask, pm, ps, rm, rs, bo, parents, lambdas)
                for k in ("pos", "rot", "fk"):
                    tally[k] += losses[k].item()
                tally["n"] += 1
        tally["pos"] /= tally["n"]
        tally["rot"] /= tally["n"]
        tally["fk"] /= tally["n"]
        tally["rot_deg"] = math.sqrt(tally["rot"]) * 180 / math.pi
        tally["n_clips"] = len(idxs)
        source_name = {0: "amass", 1: "kimodo", 2: "hy_motion", 3: "100style"}.get(src, f"src{src}")
        results[source_name] = tally
    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True,
                   choices=list(MF_VARIANTS.keys()) + ["baseline"])
    p.add_argument("--mixed-data",
                   default='/home/arenalabs/Desktop/"be water, robot"/experiment/dataset/mixed_soma77.npz')
    p.add_argument("--splits",
                   default='/home/arenalabs/Desktop/"be water, robot"/experiment/dataset/splits.json')
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    out_dir = ckpt.parent / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    history = state["history"]

    # 1-2. Training curves
    plot_training_curves(args.model, history, out_dir)
    print(f"[01-02] Training curves saved")

    # Load dataset + model
    ds = MixedMotionDataset(args.mixed_data, crop_frames=90)
    with open(args.splits) as f:
        splits = json.load(f)
    val_ids = splits["val_id"]

    if args.model == "baseline":
        cfg = BaselineConfig(T=90, J=77, C=3, hidden=128, depth=12, heads=4, ffn_mult=4)
        model = BaselineTransformer(cfg)
    else:
        variant_kw = MF_VARIANTS[args.model]
        cfg = MotionFormerConfig(T=90, J=77, C=3, hidden=128, pair_hidden=32,
                                  depth=6, heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16,
                                  **variant_kw)
        model = MotionFormer(cfg)
    model.load_state_dict(state["model_state"])
    model = model.to(device)

    pm = torch.from_numpy(state["pos_mean"][0]).to(device)
    ps = torch.from_numpy(state["pos_std"][0]).to(device)
    rm = torch.from_numpy(state["root_mean"][0]).to(device)
    rs = torch.from_numpy(state["root_std"][0]).to(device)
    bo = torch.from_numpy(state["bone_offsets"]).to(device)
    parents_t = torch.from_numpy(np.array(build_parent_index(), dtype=np.int64)).to(device)
    lambdas = state.get("lambdas", dict(pos=1.0, rot=5.0, root=1.0, fk=0.5))

    # 3. Pair heatmap
    plot_pair_heatmap(model, ds, device, out_dir)
    print(f"[03] Pair heatmap saved")

    # 4. Per-joint error
    plot_per_joint_error(model, ds, val_ids, device, out_dir, pm, ps, rm, rs, bo, parents_t)
    print(f"[04] Per-joint error saved")

    # 5. Time profile
    plot_time_profile(model, ds, val_ids, device, out_dir)
    print(f"[05] Time profile saved")

    # 6. Mask mode
    mask_res = plot_mask_mode(model, ds, val_ids, device, out_dir,
                               pm, ps, rm, rs, bo, parents_t, lambdas)
    print(f"[06] Mask mode saved")

    # 7-8. Style PCA + t-SNE
    pca_info = plot_style_clustering(model, ds, device, out_dir)
    print(f"[07-08] Style clustering saved")

    # 9. Source-stratified
    src_res = source_stratified(model, ds, val_ids, device, out_dir,
                                 pm, ps, rm, rs, bo, parents_t, lambdas)
    print(f"[09] Source stratified:")
    for src, r in src_res.items():
        print(f"   {src:<12} n={r['n_clips']:>5}  pos={r['pos']:.4f}  "
              f"rot={r['rot_deg']:.2f}°  fk={r['fk']:.4f}")

    # Save summary
    summary = {
        "model": args.model,
        "checkpoint": str(ckpt),
        "final_epoch": history[-1]["epoch"],
        "mask_mode": mask_res,
        "source_stratified": src_res,
        "pca": pca_info,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll diagnostics in {out_dir}/")


if __name__ == "__main__":
    main()
