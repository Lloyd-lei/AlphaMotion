"""Pairwise distance probe: test whether motorically similar actions live
closer in the learned pair-tensor embedding than motorically different ones.

Three pair categories:
    - same_prompt        : same text prompt, different Kimodo samples (different noise)
    - similar_motor      : hand-curated pairs with similar force / coordination pattern
    - different_motor    : hand-curated pairs with very different motor strategies

Expectation if style emerges implicitly:
        dist(same_prompt) < dist(similar_motor) < dist(different_motor)

This is a cleaner test than silhouette because it does not depend on a
particular taxonomy — it compares pairs you already agree are similar vs
not in motor terms.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import KimodoMotionDataset
from motionformer import MotionFormerConfig, MotionFormer
from train import MF_VARIANTS


# ----------------------------------------------------------------------------
# Curated pair lists
# ----------------------------------------------------------------------------
# Each pair uses prompt-keyword substrings that match against our prompt bank.


SIMILAR_MOTOR_PAIRS = [
    # Locomotion family (gait variants)
    ("walks forward", "walks backward"),
    ("walks forward", "jogs slowly"),
    ("walks forward", "walks slowly with small steps"),
    ("runs in a straight line", "sprints forward"),
    # Right-arm reach / grasp
    ("picks up a cup with the right hand", "reaches up to a high shelf"),
    ("picks up a cup with the right hand", "drinks from a cup"),
    ("opens a door with the right hand", "picks up a cup with the right hand"),
    # Right-arm explosive
    ("throws a ball with the right hand", "performs a karate punch"),
    ("throws a ball with the right hand", "swings a baseball bat"),
    # Jumping
    ("jumps in place", "hops on one foot"),
    ("jumps in place", "jumps forward"),
    # Sit/stand reciprocal
    ("sits down on a chair", "stands up from a chair"),
    # Overhead reach
    ("stretches both arms above the head", "reaches up to a high shelf"),
    # Forward bending
    ("bows forward", "bends down to tie shoelaces"),
    # Kick family
    ("performs a karate kick", "performs a side kick"),
    ("performs a karate kick", "performs a front kick"),
]


DIFFERENT_MOTOR_PAIRS = [
    # Whole-body dynamic  vs  fine-motor static
    ("walks forward", "types on a keyboard"),
    ("runs in a straight line", "writes on a notepad"),
    ("performs a karate kick", "types on a keyboard"),
    ("sprints forward", "writes on a notepad"),
    # Explosive  vs  static reach
    ("performs a karate kick", "bows forward"),
    ("throws a ball with the right hand", "sits down on a chair"),
    # Locomotion  vs  balance / kneel
    ("runs in a straight line", "stretches both arms above the head"),
    ("jumps forward", "drinks from a cup"),
    ("climbs stairs", "writes on a notepad"),
    # Crawl  vs  balance
    ("crawls on hands and knees", "stands on one foot"),
    # Single-joint fine motor  vs  whole-body dynamic
    ("types on a keyboard", "sprints forward"),
    ("claps hands several times", "crawls on hands and knees"),
    ("writes on a notepad", "jumps in place"),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def match_indices(prompts: list[str], key: str) -> list[int]:
    """Return indices where prompt contains `key` (case-insensitive)."""
    kl = key.lower()
    return [i for i, p in enumerate(prompts) if kl in p.lower()]


def extract_embeddings(model: MotionFormer, dataset: KimodoMotionDataset,
                       device: str, batch_size: int = 16,
                       pool: str = "pair_full_flat",
                       center: bool = True) -> np.ndarray:
    """Pool the pair tensor into a per-sample embedding.

    pool options:
        pair_norm_flat : [J, J] norms -> length J*J (all-positive, cosine compressed)
        pair_mean_flat : [J, J] channel means -> length J*J (signed)
        pair_full_flat : full [J, J, H] flattened (signed, high-dim)

    center : subtract the dataset mean embedding afterwards, removing the
             'everyone lives in the same cone' compression effect that would
             otherwise make cosine distances uninformative.
    """
    model.eval()
    all_z = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            idx = list(range(start, min(start + batch_size, len(dataset))))
            batch = torch.stack([dataset[i] for i in idx]).to(device)
            zero_mask = torch.zeros(batch.shape[:3], dtype=torch.bool, device=device)
            _ = model(batch, zero_mask)
            pair = model._last_pair
            if pair is None:
                raise ValueError("Model has no pair tensor.")
            if pool == "pair_norm_flat":
                z = pair.norm(dim=-1).reshape(pair.shape[0], -1)
            elif pool == "pair_mean_flat":
                z = pair.mean(dim=-1).reshape(pair.shape[0], -1)
            elif pool == "pair_full_flat":
                z = pair.reshape(pair.shape[0], -1)
            else:
                raise ValueError(pool)
            all_z.append(z.cpu().numpy())
    embs = np.concatenate(all_z, axis=0)
    if center:
        embs = embs - embs.mean(axis=0, keepdims=True)
    return embs


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(a @ b / denom)


def collect_same_prompt_pairs(prompts: list[str]) -> list[tuple[int, int]]:
    """Indices of pairs where both entries have the same prompt string."""
    by_prompt: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        by_prompt.setdefault(p.strip(), []).append(i)
    out = []
    for p, idxs in by_prompt.items():
        if len(idxs) < 2:
            continue
        for a, b in combinations(idxs, 2):
            out.append((a, b))
    return out


def collect_named_pairs(
    prompts: list[str], pair_list: list[tuple[str, str]],
) -> list[tuple[int, int]]:
    """Expand keyword pairs into all index pairs that match both sides."""
    out = []
    for k1, k2 in pair_list:
        idx1 = match_indices(prompts, k1)
        idx2 = match_indices(prompts, k2)
        for a in idx1:
            for b in idx2:
                if a != b:
                    out.append((a, b))
    return out


def distances_for_pairs(embs: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([cosine_dist(embs[a], embs[b]) for a, b in pairs], dtype=np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def load_model(variant: str, device: str, cfg_args: dict):
    variant_kw = MF_VARIANTS[variant]
    cfg = MotionFormerConfig(**cfg_args, **variant_kw)
    model = MotionFormer(cfg).to(device)
    state = torch.load(Path("runs") / variant / "final.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    return model


def summarize(label: str, d: np.ndarray) -> str:
    if len(d) == 0:
        return f"{label:<18} n=0 (no pairs found)"
    return (f"{label:<18} n={len(d):>4}  mean={d.mean():.3f}  "
            f"median={np.median(d):.3f}  std={d.std():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="full",
                   choices=["full", "opm_only", "pair_static", "triangle_only"])
    p.add_argument("--data", default="runs/kimodo_cache.npz")
    p.add_argument("--out", default="runs/pair_distance")
    p.add_argument("--pool", default="pair_full_flat",
                   choices=["pair_norm_flat", "pair_mean_flat", "pair_full_flat"])
    p.add_argument("--no-center", action="store_true",
                   help="Disable mean-centering before cosine distance.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = KimodoMotionDataset(args.data, crop_frames=90)
    prompts = list(np.load(args.data, allow_pickle=True)["prompts"])
    prompts = [str(p) for p in prompts]

    cfg_args = dict(T=ds.T, J=ds.J, C=ds.C, hidden=128, pair_hidden=32,
                    depth=6, heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16)
    model = load_model(args.model, device, cfg_args)
    embs = extract_embeddings(model, ds, device, pool=args.pool, center=not args.no_center)
    print(f"Using pool={args.pool}, centered={not args.no_center}")

    pair_same = collect_same_prompt_pairs(prompts)
    pair_sim = collect_named_pairs(prompts, SIMILAR_MOTOR_PAIRS)
    pair_diff = collect_named_pairs(prompts, DIFFERENT_MOTOR_PAIRS)

    d_same = distances_for_pairs(embs, pair_same)
    d_sim = distances_for_pairs(embs, pair_sim)
    d_diff = distances_for_pairs(embs, pair_diff)

    # Random pairs baseline
    rng = np.random.default_rng(0)
    N = len(embs)
    rand_idx = rng.choice(N, size=(min(len(pair_diff) * 2, 200), 2), replace=True)
    rand_pairs = [(int(a), int(b)) for a, b in rand_idx if a != b]
    d_rand = distances_for_pairs(embs, rand_pairs)

    print(f"\nEmbedding: {embs.shape}   model={args.model}\n")
    print(summarize("same_prompt",      d_same))
    print(summarize("similar_motor",    d_sim))
    print(summarize("random_pair",      d_rand))
    print(summarize("different_motor",  d_diff))
    print()

    # Hypothesis expressed as ordered means
    def arrow(a, b):
        if a < b:
            return "✓"
        return "✗"
    order = (
        f"  same({d_same.mean():.3f}) {arrow(d_same.mean(), d_sim.mean())}"
        f" sim({d_sim.mean():.3f}) {arrow(d_sim.mean(), d_rand.mean())}"
        f" rand({d_rand.mean():.3f}) {arrow(d_rand.mean(), d_diff.mean())}"
        f" diff({d_diff.mean():.3f})"
    )
    print("Ordering check (expect all ✓ for full implicit style emergence):")
    print(order)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(d_diff.max() if len(d_diff) else 1,
                              d_rand.max() if len(d_rand) else 1) + 0.02, 40)
    ax.hist(d_same, bins=bins, alpha=0.55, label=f"same_prompt (n={len(d_same)})",
            color="tab:green", density=True)
    ax.hist(d_sim, bins=bins, alpha=0.55, label=f"similar_motor (n={len(d_sim)})",
            color="tab:blue", density=True)
    ax.hist(d_rand, bins=bins, alpha=0.35, label=f"random (n={len(d_rand)})",
            color="gray", density=True)
    ax.hist(d_diff, bins=bins, alpha=0.55, label=f"different_motor (n={len(d_diff)})",
            color="tab:red", density=True)
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("density")
    ax.set_title(f"Pair distance distribution  —  model={args.model}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"pairdist_{args.model}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
