"""Pair tensor sanity check: is the anatomical structure in MotionFormer's
pair tensor genuinely learned, or some confound (sampling, architecture, init,
or OPM alone)?

Produces 6-panel comparison + Pearson r matrix to rule out four confounds:
    A. Reproducibility across batch sampling (seed=0 vs seed=42)
    B. Architecture artifact (untrained MotionFormer)
    C. pair_init weight residual (parameter alone, no forward)
    D. OPM-only learns same thing (no-triangle variant)

Outputs:
    runs/pair_heatmap_sanity.png            — 6-panel comparison figure
    runs/pair_heatmap_sanity.json           — quantitative results
    runs/<variant>__<tag>/diag/03_pair_heatmap.png  — per-variant single heatmap

Usage:
    python pair_heatmap_sanity.py \
        --full-checkpoint runs/full__mixed_fixed30/final.pt \
        --opm-checkpoint  runs/opm_only__mixed_fixed30/final.pt

Companion writeup: note/triangle_sanity_check.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import MixedMotionDataset
from motionformer import MotionFormerConfig, MotionFormer
from train import MF_VARIANTS


def build(variant: str, device: str):
    kw = MF_VARIANTS[variant]
    cfg = MotionFormerConfig(
        T=90, J=77, C=3,
        hidden=128, pair_hidden=32, depth=6, heads=4, pair_heads=4,
        opm_chunk=16, tri_hidden=16, **kw,
    )
    return MotionFormer(cfg).to(device)


def avg_pair_norm(model, ds, n: int, seed: int, device: str) -> np.ndarray:
    """Forward `n` randomly sampled clips, average their final-block pair
    tensor's L2 norm in the channel dim, and symmetrize."""
    model.eval()
    total = None
    cnt = 0
    with torch.no_grad():
        for i in np.random.default_rng(seed).choice(len(ds), n, replace=False):
            s = ds[int(i)]["pos"].unsqueeze(0).to(device)
            zm = torch.zeros(1, 90, 77, dtype=torch.bool, device=device)
            _ = model(s, zm)
            norm = model._last_pair.norm(dim=-1)[0].cpu().numpy()  # [J, J]
            total = norm if total is None else total + norm
            cnt += 1
    avg = total / cnt
    return 0.5 * (avg + avg.T)


def offdiag(M: np.ndarray) -> np.ndarray:
    return M[~np.eye(M.shape[0], dtype=bool)]


def stats_str(M: np.ndarray, label: str) -> str:
    o = offdiag(M)
    return f"{label:42s}  range=[{o.min():.2f}, {o.max():.2f}]  std/mean={o.std()/abs(o.mean()):.3f}"


def corr(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.corrcoef(offdiag(A), offdiag(B))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-checkpoint", default="runs/full__mixed_fixed30/final.pt")
    ap.add_argument("--opm-checkpoint", default="runs/opm_only__mixed_fixed30/final.pt")
    ap.add_argument("--mixed-data", default="../dataset/mixed_soma77.npz")
    ap.add_argument("--n-samples", type=int, default=32,
                    help="Number of dataset clips to forward per heatmap.")
    ap.add_argument("--seed-a1", type=int, default=0)
    ap.add_argument("--seed-a2", type=int, default=42)
    ap.add_argument("--out-dir", default="runs",
                    help="Where to write the 6-panel sanity png + json.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is up.")
    args = ap.parse_args()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    print(f"device={device}")

    print(f"Loading dataset {args.mixed_data} ...")
    ds = MixedMotionDataset(args.mixed_data, crop_frames=90)
    print(f"  N={ds.N}  T={ds.T}  J={ds.J}")

    # ---- A: full TRAINED, two seeds ----
    print("\n[A] full TRAINED — two batch seeds")
    full_trained = build("full", device)
    full_trained.load_state_dict(torch.load(
        args.full_checkpoint, map_location=device, weights_only=False,
    )["model_state"])
    t0 = time.time(); A1 = avg_pair_norm(full_trained, ds, args.n_samples, args.seed_a1, device)
    print(f"  A1 (seed={args.seed_a1}) {time.time()-t0:.1f}s")
    t0 = time.time(); A2 = avg_pair_norm(full_trained, ds, args.n_samples, args.seed_a2, device)
    print(f"  A2 (seed={args.seed_a2}) {time.time()-t0:.1f}s")

    # ---- B: full UNTRAINED ----
    print("\n[B] full UNTRAINED (random init)")
    torch.manual_seed(0)
    full_untrained = build("full", device)
    t0 = time.time(); B = avg_pair_norm(full_untrained, ds, args.n_samples, args.seed_a1, device)
    print(f"  done {time.time()-t0:.1f}s")

    # ---- C: pair_init weights (no forward) ----
    print("\n[C] pair_init weight tensors (no forward)")
    C1 = full_trained.pair_init.detach().norm(dim=-1).cpu().numpy()
    C2 = full_untrained.pair_init.detach().norm(dim=-1).cpu().numpy()
    C1 = 0.5 * (C1 + C1.T)
    C2 = 0.5 * (C2 + C2.T)

    # ---- D: opm_only TRAINED ----
    print("\n[D] opm_only TRAINED (no triangle)")
    opm_trained = build("opm_only", device)
    opm_trained.load_state_dict(torch.load(
        args.opm_checkpoint, map_location=device, weights_only=False,
    )["model_state"])
    t0 = time.time(); D = avg_pair_norm(opm_trained, ds, args.n_samples, args.seed_a1, device)
    print(f"  done {time.time()-t0:.1f}s")

    # ---- Print quantitative summary ----
    print()
    for line in [
        stats_str(A1, "[A1] full TRAINED, sample seed=0"),
        stats_str(A2, "[A2] full TRAINED, sample seed=42"),
        stats_str(B,  "[B]  full UNTRAINED — control"),
        stats_str(C1, "[C1] pair_init param TRAINED"),
        stats_str(C2, "[C2] pair_init param UNTRAINED"),
        stats_str(D,  "[D]  opm_only TRAINED — no triangle"),
    ]:
        print(line)

    delta = float(np.abs(A1 - A2).mean())
    a1_mean = float(A1.mean())
    print(f"\n   |Δ(A1, A2)| / mean = {100*delta/a1_mean:.2f}%")

    correlations = {
        "trained_seed0_vs_trained_seed42":  corr(A1, A2),
        "trained_seed0_vs_untrained":       corr(A1, B),
        "trained_seed0_vs_opm_only":        corr(A1, D),
        "pair_init_TRAINED_vs_UNTRAINED":   corr(C1, C2),
        "pair_init_TRAINED_vs_trained_final": corr(C1, A1),
    }
    print("\nPearson r between heatmaps (off-diagonal):")
    for k, v in correlations.items():
        print(f"  {k:42s} = {v:+.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    panels = [
        (A1, "[A1] full TRAINED — seed=0 (n=32)"),
        (A2, "[A2] full TRAINED — seed=42 (n=32)"),
        (B,  "[B]  full UNTRAINED — control"),
        (C1, "[C1] pair_init weight TRAINED"),
        (C2, "[C2] pair_init weight UNTRAINED randn"),
        (D,  "[D]  opm_only TRAINED — no triangle"),
    ]
    key_joints = [
        ("Hips", 0), ("Chest", 3), ("Head", 6),
        ("L.Sh", 11), ("L.Hd", 14),
        ("R.Sh", 39), ("R.Hd", 42),
        ("L.Knee", 66), ("R.Knee", 71),
    ]
    tp = [i for _, i in key_joints]
    tl = [n for n, _ in key_joints]
    for ax, (M, label) in zip(axes.flat, panels):
        im = ax.imshow(M, cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(tp); ax.set_xticklabels(tl, rotation=45, fontsize=7)
        ax.set_yticks(tp); ax.set_yticklabels(tl, fontsize=7)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "pair_heatmap_sanity.png"
    plt.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"\nsaved {fig_path}")

    # Also write per-variant single heatmap (replaces the broken diag/03 entry)
    for variant, M in [("full", A1), ("opm_only", D)]:
        ck_dir = Path(f"runs/{variant}__mixed_fixed30/diag")
        if ck_dir.exists():
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(M, cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"Pair tensor L2 norm — {variant} (mixed_fixed30, n={args.n_samples})")
            ax.set_xticks(tp); ax.set_xticklabels(tl, rotation=45, fontsize=8)
            ax.set_yticks(tp); ax.set_yticklabels(tl, fontsize=8)
            plt.tight_layout()
            plt.savefig(ck_dir / "03_pair_heatmap.png", dpi=110)
            plt.close(fig)

    json_path = out_dir / "pair_heatmap_sanity.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_samples": args.n_samples,
            "seeds": {"A1": args.seed_a1, "A2": args.seed_a2},
            "checkpoints": {
                "full": str(args.full_checkpoint),
                "opm_only": str(args.opm_checkpoint),
            },
            "stats": {
                "A1": {"min": float(offdiag(A1).min()), "max": float(offdiag(A1).max()),
                       "std_over_mean": float(offdiag(A1).std() / offdiag(A1).mean())},
                "A2": {"min": float(offdiag(A2).min()), "max": float(offdiag(A2).max()),
                       "std_over_mean": float(offdiag(A2).std() / offdiag(A2).mean())},
                "B":  {"min": float(offdiag(B).min()),  "max": float(offdiag(B).max()),
                       "std_over_mean": float(offdiag(B).std() / offdiag(B).mean())},
                "D":  {"min": float(offdiag(D).min()),  "max": float(offdiag(D).max()),
                       "std_over_mean": float(offdiag(D).std() / offdiag(D).mean())},
            },
            "correlations": correlations,
            "delta_seeds_pct_of_mean": 100 * delta / a1_mean,
        }, f, indent=2)
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
