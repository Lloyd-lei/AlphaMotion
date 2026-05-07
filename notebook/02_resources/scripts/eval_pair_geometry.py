"""Pair-tensor geometry analysis (FM regime, Stage-1 v2).

Forward the FM model at **t=1 (data state)** with clean ground-truth
(pos, rot6d, root), hook the last block's pair tensor. This gives the
trunk's pair representation conditioned on real data — the right slice
for Claim 2 (geometry under data state).

Public API (unchanged):
    extract_pair_tensor(ckpt_path, subset, *, n_clips=200) -> np.ndarray [J, J] | None
    modularity_score(heatmap) -> dict
    pair_embedding_pca(ckpt_path, subset, *, n=500) -> Figure
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from _eval_common import load_eval_model, eval_loader, fm_forward_at_t


def _hook_last_block_pair(model):
    """Returns (handle, captured_dict) for grabbing last block's pair tensor.

    Uses **L2 norm over the per-pair feature axis** (matches the original
    diagnose_stage15.py methodology). L2 is positive everywhere so per-clip
    averaging across input-conditioned variants does NOT cancel out — the
    earlier `mean(dim=-1)` choice silently dampened `full`'s magnitude.
    """
    if not hasattr(model, "blocks"):
        raise RuntimeError(f"model {type(model).__name__} has no .blocks")
    captured = {"pair": None}
    last = model.blocks[-1]
    def _h(m, inp, out):
        pair = out[1] if isinstance(out, tuple) and len(out) >= 2 else out
        captured["pair"] = pair.norm(dim=-1).detach().cpu()   # [B, J, J] L2 norm
    h = last.register_forward_hook(_h)
    return h, captured


@torch.no_grad()
def extract_pair_tensor(
    ckpt_path: str,
    subset: Subset,
    *,
    n_clips: int = 200,
    batch_size: int = 32,
    device: str = "cuda",
    t_value: float = 1.0,
) -> Optional[np.ndarray]:
    """Mean pair tensor [J, J] over `n_clips`. None if no pair stream."""
    model, cfg, variant = load_eval_model(ckpt_path, device=device)
    if not getattr(cfg, "use_pair", False):
        print(f"[{variant}] no pair tensor; returning None")
        return None
    dev = next(model.parameters()).device

    h, captured = _hook_last_block_pair(model)
    sub2 = Subset(subset.dataset, subset.indices[:n_clips])
    loader = eval_loader(sub2, batch=batch_size)
    accum, n_seen = None, 0
    try:
        for batch in loader:
            x_pos_1  = batch["pos"].to(dev)
            x_rot_1  = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=t_value)
            pair_bjj = captured["pair"]
            if pair_bjj is None:
                raise RuntimeError("hook didn't capture pair")
            inc = pair_bjj.sum(dim=0).numpy().astype(np.float64)
            accum = inc if accum is None else accum + inc
            n_seen += pair_bjj.shape[0]
    finally:
        h.remove()

    avg = (accum / max(1, n_seen)).astype(np.float32)
    # Symmetrize per the original diagnose_stage15.py methodology
    return 0.5 * (avg + avg.T)


def modularity_score(heatmap: np.ndarray) -> Dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if heatmap is None:
        return {"silhouette": float("nan"), "modularity_coef": float("nan"),
                "hub_score": float("nan"), "n_communities": 0}

    H = np.abs(heatmap.astype(np.float64))
    np.fill_diagonal(H, 0)
    rows = H + H.T
    K = 5
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(rows)
    labels = km.labels_
    try:
        sil = float(silhouette_score(rows, labels))
    except Exception:
        sil = float("nan")
    A = rows; m = A.sum() / 2 + 1e-12; deg = A.sum(axis=1); Q = 0.0
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sub = A[np.ix_(idx, idx)]
        e_c = sub.sum() / (2 * m)
        d_c = deg[idx].sum() / (2 * m)
        Q += e_c - d_c**2
    return {"silhouette": sil, "modularity_coef": float(Q),
            "hub_score": float(deg.max() / (deg.mean() + 1e-12)),
            "n_communities": K}


@torch.no_grad()
def pair_embedding_pca(
    ckpt_path: str, subset: Subset,
    *, n: int = 500, batch_size: int = 32, device: str = "cuda",
    t_value: float = 1.0,
):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    model, cfg, variant = load_eval_model(ckpt_path, device=device)
    if not getattr(cfg, "use_pair", False):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{variant}: no pair tensor (use_pair=False)",
                ha="center", va="center", fontsize=12, style="italic")
        return fig

    dev = next(model.parameters()).device
    h, captured = _hook_last_block_pair(model)
    base_ds = subset.dataset
    labels  = np.asarray([str(l) for l in base_ds.action_label])
    use_idx = subset.indices[:n]
    sub2 = Subset(base_ds, use_idx)
    loader = eval_loader(sub2, batch=batch_size)

    feats, fam = [], []
    try:
        for bi, batch in enumerate(loader):
            x_pos_1  = batch["pos"].to(dev)
            x_rot_1  = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=t_value)
            B = x_pos_1.shape[0]
            f = captured["pair"].reshape(B, -1).numpy()
            feats.append(f)
            for i in range(B):
                fam.append(labels[use_idx[bi*batch_size + i]])
    finally:
        h.remove()

    X = np.concatenate(feats, 0)
    pca = PCA(n_components=2, random_state=0).fit(X)
    XY  = pca.transform(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    families = sorted(set(fam))
    cmap = plt.get_cmap("tab10")
    for k, f in enumerate(families):
        m = np.array([x == f for x in fam])
        ax.scatter(XY[m, 0], XY[m, 1], s=14, alpha=0.7,
                   color=cmap(k % 10), label=f"{f} ({m.sum()})")
    ax.set_xlabel(f"PCA-1   ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PCA-2   ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"{variant}: pair-tensor embedding (FM, t={t_value}) by action family (N={len(X)})")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig
