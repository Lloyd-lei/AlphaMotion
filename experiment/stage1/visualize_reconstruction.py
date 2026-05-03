"""Sanity-check reconstruction: does MotionFormer's output look like real motion?

Takes a few held-out Kimodo samples, applies different masks, predicts the
filled-in motion, and renders:
    - ground truth (GT) motion
    - masked input shown to the model
    - model reconstruction

Output:
    runs/reconstruction/
        sample_{i}_{mask_mode}.gif   — 3D animated comparison
        sample_{i}_{mask_mode}.png   — static 4-frame snapshot grid
        sample_{i}_{mask_mode}.npz   — raw data (gt, masked, pred, mask)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from data import KimodoMotionDataset, make_masked_batch
from motionformer import MotionFormerConfig, MotionFormer
from soma_skeleton import build_parent_index, JOINT_NAMES
from train import MF_VARIANTS


def load_model(variant: str, device: str, cfg_args: dict):
    variant_kw = MF_VARIANTS[variant]
    cfg = MotionFormerConfig(**cfg_args, **variant_kw)
    model = MotionFormer(cfg).to(device)
    state = torch.load(Path("runs") / variant / "final.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def plot_3d_skeleton(ax, joints, parents, color="tab:blue", alpha=1.0, marker_size=6):
    """joints: [J, 3]. parents: list of J ints, -1 for root."""
    xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
    ax.scatter(xs, ys, zs, s=marker_size, c=color, alpha=alpha)
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([xs[j], xs[p]], [ys[j], ys[p]], [zs[j], zs[p]],
                    color=color, alpha=alpha * 0.7, linewidth=1)


def set_equal_3d_axes(ax, coords):
    """coords: array of all [X, Y, Z] points to bound the view."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    ctr = 0.5 * (mn + mx)
    span = (mx - mn).max() * 0.6 + 0.1
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


def make_snapshot_grid(gt, masked, pred, mask, parents, out_path, title):
    """Show 4 equally spaced frames as a 3-column (gt/masked/pred) grid."""
    T = gt.shape[0]
    frame_ids = [0, T // 3, 2 * T // 3, T - 1]
    fig = plt.figure(figsize=(15, 13))
    all_coords = np.concatenate([gt, pred], axis=0).reshape(-1, 3)
    for row, frame in enumerate(frame_ids):
        for col, (data, label, color) in enumerate([
            (gt[frame], "ground truth", "tab:green"),
            (masked[frame], "masked input", "tab:orange"),
            (pred[frame], "model reconstruction", "tab:blue"),
        ]):
            ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection="3d")
            if col == 1:
                # masked: colour masked joints red, kept joints grey
                m = mask[frame]
                ax.scatter(data[~m, 0], data[~m, 1], data[~m, 2],
                           s=6, c="grey", alpha=0.8)
                ax.scatter(data[m, 0], data[m, 1], data[m, 2],
                           s=12, c="red", alpha=0.6, marker="x")
                for j, p in enumerate(parents):
                    if p >= 0 and not m[j] and not m[p]:
                        ax.plot([data[j, 0], data[p, 0]],
                                [data[j, 1], data[p, 1]],
                                [data[j, 2], data[p, 2]],
                                color="grey", alpha=0.5, linewidth=1)
            else:
                plot_3d_skeleton(ax, data, parents, color=color)
            set_equal_3d_axes(ax, all_coords)
            ax.set_title(f"t={frame}  {label}", fontsize=9)
            ax.view_init(elev=20, azim=-75)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def make_animation(gt, pred, parents, out_path, title, fps=15):
    """Side-by-side 3D animation of gt vs pred."""
    T = gt.shape[0]
    fig = plt.figure(figsize=(10, 5))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pr = fig.add_subplot(1, 2, 2, projection="3d")
    all_coords = np.concatenate([gt, pred], axis=0).reshape(-1, 3)

    def draw(frame):
        for ax, data, color, label in [
            (ax_gt, gt[frame], "tab:green", "ground truth"),
            (ax_pr, pred[frame], "tab:blue", "reconstruction"),
        ]:
            ax.clear()
            plot_3d_skeleton(ax, data, parents, color=color)
            set_equal_3d_axes(ax, all_coords)
            ax.set_title(f"{label}  t={frame}")
            ax.view_init(elev=20, azim=-75)

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="full", choices=list(MF_VARIANTS.keys()))
    p.add_argument("--data", default="runs/kimodo_cache.npz")
    p.add_argument("--samples", type=int, nargs="*",
                   default=[0, 50, 100, 150],
                   help="Dataset indices to visualise.")
    p.add_argument("--mask-modes", nargs="*",
                   default=["random", "kinematic_chain"])
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--out", default="runs/reconstruction")
    p.add_argument("--gif", action="store_true",
                   help="Also produce GIF animations (slower).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = KimodoMotionDataset(args.data, crop_frames=90)
    prompts = list(np.load(args.data, allow_pickle=True)["prompts"])

    cfg_args = dict(T=ds.T, J=ds.J, C=ds.C, hidden=128, pair_hidden=32,
                    depth=6, heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16)
    model = load_model(args.model, device, cfg_args)
    parents = build_parent_index()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # De-normalisation: every displayed quantity must be mapped back to
    # physical units or the skeleton looks broken.
    mean_jc = ds.mean[0]    # [J, C]
    std_jc = ds.std[0]      # [J, C]

    def denorm(x_nrm: np.ndarray) -> np.ndarray:
        # x_nrm: [..., J, C] in normalised space
        return x_nrm * std_jc + mean_jc

    for idx in args.samples:
        gt_tensor = ds[idx].unsqueeze(0).to(device)  # [1, T, J, C] (normalised)
        for mode in args.mask_modes:
            masked_in, mask, target = make_masked_batch(
                gt_tensor, mask_ratio=args.mask_ratio, mask_mode=mode,
            )
            with torch.no_grad():
                pred_nrm = model(masked_in, mask).cpu().numpy()[0]   # normalised
            gt_nrm = target.cpu().numpy()[0]
            masked_nrm = masked_in.cpu().numpy()[0]
            m = mask.cpu().numpy()[0]    # [T, J]

            # De-normalise everything back to physical (root-relative) coords
            gt = denorm(gt_nrm)
            pred = denorm(pred_nrm)
            # Masked input: unmasked joints come from the real data; masked
            # positions are conceptually zero — keep them at root position so the
            # visualisation shows "this joint is missing, will be filled".
            masked_physical = denorm(gt_nrm.copy())
            masked_physical[m] = 0.0   # zero in normalised space == mean in physical

            composite = gt.copy()
            composite[m] = pred[m]
            masked = masked_physical

            prompt = prompts[idx]
            tag = f"sample{idx}_{mode}"
            title = f"[{args.model}] [{mode}]  sample {idx}: {prompt[:60]}"

            # Static snapshot grid
            make_snapshot_grid(gt, masked, composite, m, parents,
                               out_dir / f"{tag}.png", title)

            # Animation (optional)
            if args.gif:
                make_animation(gt, composite, parents,
                               out_dir / f"{tag}.gif", title)

            # Raw data for later inspection
            np.savez(out_dir / f"{tag}.npz",
                     gt=gt, masked=masked, pred=pred, composite=composite,
                     mask=m, prompt=prompt)
            print(f"[{tag}] saved. prompt='{prompt[:50]}'  "
                  f"masked_frac={m.mean():.2f}  "
                  f"L2_on_masked={np.sqrt(((composite - gt)**2 * m[..., None]).sum() / (m.sum() * 3 + 1e-8)):.4f}")


if __name__ == "__main__":
    main()
