"""Style-clustering diagnostic: does a trained MotionFormer's pair tensor
implicitly organise samples by action family without any supervision?

Procedure:
    1. Load the trained `full` checkpoint.
    2. For every Kimodo sample, run a single forward pass (no mask).
    3. Extract a style embedding by pooling the final pair tensor.
    4. t-SNE to 2-D. Colour by the prompt family (extracted from prompt string).
    5. Report cluster separation score: average silhouette across prompt families.

If action family self-organises in the embedding, style emerges from pair
tensor dynamics + MMM alone, with no contrastive / caption supervision.

Usage:
    python style_cluster.py                  # uses runs/full/final.pt
    python style_cluster.py --model pair_static
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from data import KimodoMotionDataset
from motionformer import MotionFormerConfig, MotionFormer
from train import MF_VARIANTS


# ----------------------------------------------------------------------------
# Prompt family classification (hand-rolled from our prompt_bank)
# ----------------------------------------------------------------------------


FAMILY_PATTERNS = [
    ("locomotion",  r"\b(walk|run|jog|sprint|hop|skip|climb|descend|crawl|stair)"),
    ("manipulation", r"\b(pick up|open|reach|door|box|cup|shelf|throw|catch|swing|bat|drink|type|write|fold|sweep|carry)"),
    ("stance",      r"\b(sit|stand|kneel|cross|bend|tie)"),
    ("jump",        r"\b(jump|leap|hop)"),
    ("balance",     r"\b(balance|one foot|narrow line|lean)"),
    ("dance_stretch", r"\b(dance|stretch|twist|rotate|bow|wave|clap|point)"),
    ("martial",     r"\b(karate|punch|kick|block|spar)"),
]


def classify_prompt(p: str) -> str:
    pl = p.lower()
    for family, patt in FAMILY_PATTERNS:
        if re.search(patt, pl):
            return family
    return "other"


# ----------------------------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------------------------


def extract_style_embeddings(
    model: MotionFormer,
    dataset: KimodoMotionDataset,
    prompts: list[str],
    pool_mode: str = "pair_norm_flat",
    device: str = "cuda",
    batch_size: int = 16,
):
    """Return (embeddings[N, d], families[N], prompts[N])."""
    model.eval()
    all_z = []
    all_fam = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            idx = list(range(start, min(start + batch_size, len(dataset))))
            batch = torch.stack([dataset[i] for i in idx]).to(device)
            zero_mask = torch.zeros(batch.shape[:3], dtype=torch.bool, device=device)
            _ = model(batch, zero_mask)
            pair = model._last_pair                # [B, J, J, H] or None
            if pair is None:
                # Fallback for axial-only variants: use MSA final tensor
                raise ValueError("Model has no pair tensor; try a variant with use_pair=True.")
            if pool_mode == "pair_norm_flat":
                # Compact: L2 norm of each (i, j) vector, flatten
                z = pair.norm(dim=-1).reshape(pair.shape[0], -1)
            elif pool_mode == "pair_mean_flat":
                z = pair.mean(dim=-1).reshape(pair.shape[0], -1)
            elif pool_mode == "pair_full_flat":
                z = pair.reshape(pair.shape[0], -1)
            else:
                raise ValueError(pool_mode)
            all_z.append(z.cpu().numpy())
            all_fam.extend(classify_prompt(prompts[i]) for i in idx)
    embs = np.concatenate(all_z, axis=0)
    return embs, np.array(all_fam), prompts


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def cluster_scores(embs: np.ndarray, families: np.ndarray) -> dict:
    """Silhouette score on raw embeddings (how well family labels separate)."""
    # Need at least 2 families with members
    uniq = np.unique(families)
    if len(uniq) < 2:
        return {"silhouette_raw": None, "silhouette_tsne": None}
    # Raw high-dim silhouette (may be noisy)
    # Convert families to integer codes
    fam_code = {f: i for i, f in enumerate(uniq)}
    y = np.array([fam_code[f] for f in families])
    sil_raw = float(silhouette_score(embs, y, metric="cosine"))
    return {"silhouette_raw_cosine": sil_raw}


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def load_model(variant: str, device: str, cfg_args: dict):
    variant_kw = MF_VARIANTS[variant]
    cfg = MotionFormerConfig(**cfg_args, **variant_kw)
    model = MotionFormer(cfg).to(device)
    ckpt_path = Path("runs") / variant / "final.pt"
    if not ckpt_path.exists():
        # Try seed-tagged fallback
        for alt in (Path("runs") / f"{variant}__seed0", ):
            if alt.exists():
                ckpt_path = alt / "final.pt"
                break
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="full",
                   choices=["full", "opm_only", "pair_static", "triangle_only"],
                   help="Which MotionFormer variant to probe.")
    p.add_argument("--data", default="runs/kimodo_cache.npz")
    p.add_argument("--pool", default="pair_norm_flat",
                   choices=["pair_norm_flat", "pair_mean_flat", "pair_full_flat"])
    p.add_argument("--out", default="runs/style_cluster")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = KimodoMotionDataset(args.data, crop_frames=90)

    # Prompts come from the npz cache
    prompts = list(np.load(args.data, allow_pickle=True)["prompts"])
    assert len(prompts) == len(ds), f"prompts {len(prompts)} != dataset {len(ds)}"

    cfg_args = dict(T=ds.T, J=ds.J, C=ds.C, hidden=128, pair_hidden=32,
                    depth=6, heads=4, pair_heads=4, opm_chunk=16, tri_hidden=16)
    model = load_model(args.model, device, cfg_args)

    embs, fams, _ = extract_style_embeddings(
        model, ds, prompts, pool_mode=args.pool, device=device,
    )
    print(f"Embeddings: {embs.shape}   families: {np.unique(fams, return_counts=True)}")

    scores = cluster_scores(embs, fams)
    print(f"Silhouette scores: {scores}")

    # t-SNE
    perp = min(30, max(5, len(embs) // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca",
                learning_rate="auto")
    pts = tsne.fit_transform(embs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    uniq = sorted(set(fams))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for i, fam in enumerate(uniq):
        m = fams == fam
        ax.scatter(pts[m, 0], pts[m, 1], label=f"{fam} (n={m.sum()})",
                   alpha=0.8, s=35, color=cmap(i % 10))
    ax.legend(fontsize=9, loc="best")
    sil = scores.get("silhouette_raw_cosine")
    sil_str = f"{sil:.3f}" if sil is not None else "N/A"
    ax.set_title(f"Style t-SNE   model={args.model}   pool={args.pool}   "
                 f"silhouette(cosine)={sil_str}")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.3)

    out_path = out_dir / f"tsne_{args.model}_{args.pool}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Dump raw numbers for later analysis
    np.savez(out_dir / f"{args.model}_{args.pool}.npz",
             embeddings=embs, families=fams, tsne_2d=pts)


if __name__ == "__main__":
    main()
