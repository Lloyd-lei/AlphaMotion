"""Family/style emergence with REAL trained encoder embeddings.

Replaces NB1 §2.2.1.b's placeholder (raw mid-frame joint features which
captured subject identity, not motion style).

Public API:
    extract_pair_embeddings(ckpt_path, subset)
        -> (X[N, D], sample_names[N], action_labels[N])

    cluster_and_visualize(X, names, labels, *, n_clusters=8,
                           method='kmeans', reducer='pca', seed=0) -> Figure

    label_alignment(X, true_labels, n_clusters=12) -> dict
        {ari, nmi, purity, n_evaluated, n_clusters, n_true_labels}

    silhouette_per_variant(ckpts: dict, subset) -> dict
        {variant: silhouette_score}
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from _eval_common import load_eval_model, eval_loader


@torch.no_grad()
def extract_pair_embeddings(
    ckpt_path: str, subset: Subset,
    *, batch_size: int = 32, device: str = "cuda",
    embedding: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-clip embeddings.

    embedding:
        'auto'      — pair-norm if model has pair tensor; else MSA pool
        'pair_norm' — pair tensor L2 norm (anatomical structure; matches §3 heatmap)
        'msa_pool'  — MSA pool over time (encoder embedding; better for motion style)
    """
    model, cfg, variant = load_eval_model(ckpt_path, device=device)
    has_pair = getattr(cfg, "use_pair", False)
    use_msa = (embedding == "msa_pool") or (embedding == "auto" and not has_pair)
    if use_msa:
        print(f"[{variant}] using MSA-pool embedding")
        return _extract_msa_pool(model, subset, batch_size=batch_size)
    if not has_pair:
        raise ValueError(f"[{variant}] use_pair=False but embedding='pair_norm' requested")
    return _extract_pair(model, subset, batch_size=batch_size)


def _extract_pair(model, subset, *, batch_size: int):
    """FM: forward at t=1 with clean data, hook last block's pair tensor (L2 norm)."""
    from _eval_common import fm_forward_at_t
    dev = next(model.parameters()).device
    if not hasattr(model, "blocks"):
        raise RuntimeError(f"model has no .blocks: {type(model).__name__}")
    captured = {"pair": None}
    last = model.blocks[-1]
    def _h(m, inp, out):
        pair = out[1] if isinstance(out, tuple) and len(out) >= 2 else out
        captured["pair"] = pair.norm(dim=-1).detach().cpu()   # ← norm, matches §3
    h = last.register_forward_hook(_h)

    base_ds = subset.dataset
    s_names = np.asarray([str(s) for s in base_ds.sample_name])
    a_labels = np.asarray([str(l) for l in base_ds.action_label])
    loader = eval_loader(subset, batch=batch_size)
    feats, names, labels = [], [], []
    try:
        for bi, batch in enumerate(loader):
            x_pos_1  = batch["pos"].to(dev)
            x_rot_1  = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=1.0)
            B = x_pos_1.shape[0]
            f = captured["pair"].reshape(B, -1).numpy()
            feats.append(f)
            for i in range(B):
                gi = subset.indices[bi*batch_size + i]
                names.append(s_names[gi]); labels.append(a_labels[gi])
    finally:
        h.remove()
    return (np.concatenate(feats, 0).astype(np.float32),
            np.asarray(names), np.asarray(labels))


def _extract_msa_pool(model, subset, *, batch_size: int):
    """FM: forward at t=1, hook output_norm — works for both motionformer
    (has .blocks) AND baseline (uses nn.TransformerEncoder, no .blocks).
    Both expose a final `output_norm` LayerNorm with [B, T, J, H] output."""
    from _eval_common import fm_forward_at_t
    dev = next(model.parameters()).device
    base_ds = subset.dataset
    s_names = np.asarray([str(s) for s in base_ds.sample_name])
    a_labels = np.asarray([str(l) for l in base_ds.action_label])
    loader = eval_loader(subset, batch=batch_size)
    captured = {"msa": None}
    if not hasattr(model, "output_norm"):
        raise RuntimeError(f"model has no output_norm: {type(model).__name__}")
    def _h(m, inp, out):
        captured["msa"] = out.detach().cpu()         # [B, T, J, H]
    h = model.output_norm.register_forward_hook(_h)
    feats, names, labels = [], [], []
    try:
        for bi, batch in enumerate(loader):
            x_pos_1  = batch["pos"].to(dev)
            x_rot_1  = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            _ = fm_forward_at_t(model, x_pos_1, x_rot_1, x_root_1, t_value=1.0)
            msa = captured["msa"].mean(dim=1)        # mean over T → [B, J, H]
            B = x_pos_1.shape[0]
            f = msa.reshape(B, -1).numpy()
            feats.append(f)
            for i in range(B):
                gi = subset.indices[bi*batch_size + i]
                names.append(s_names[gi]); labels.append(a_labels[gi])
    finally:
        h.remove()
    return (np.concatenate(feats, 0).astype(np.float32),
            np.asarray(names), np.asarray(labels))


def cluster_and_visualize(
    X: np.ndarray, names: np.ndarray, labels: np.ndarray,
    *, n_clusters: int = 8, method: str = "kmeans",
    reducer: str = "pca", seed: int = 0,
):
    from collections import Counter
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    n = X.shape[0]
    if reducer == "umap":
        try:
            import umap
            xy = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
            ax_label_x, ax_label_y = "UMAP-1", "UMAP-2"
        except ImportError:
            reducer = "pca"
    if reducer == "pca":
        pca = PCA(n_components=2, random_state=seed).fit(X)
        xy = pca.transform(X)
        v1, v2 = pca.explained_variance_ratio_
        ax_label_x = f"PCA-1   ({v1*100:.1f}% var)"
        ax_label_y = f"PCA-2   ({v2*100:.1f}% var)"

    cidx = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X).labels_

    fig = plt.figure(figsize=(15, 6), facecolor="white")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.6], wspace=0.05)
    ax  = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("tab10")
    ax.scatter(xy[:, 0], xy[:, 1], c=cidx, cmap=cmap, s=18, alpha=0.75, edgecolor="none")
    ax.set_xlabel(ax_label_x); ax.set_ylabel(ax_label_y)
    ax.set_title(
        f"Unsupervised clustering on TRAINED encoder embeddings\n"
        f"N={n}, D={X.shape[1]}, KMeans in feature space, {reducer.upper()} for 2D viz",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    for k in range(n_clusters):
        m = cidx == k
        if m.sum() > 0:
            cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
            ax.text(cx, cy, str(k), ha="center", va="center", fontsize=11,
                    color="black", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=cmap(k % 10),
                              boxstyle="circle,pad=0.25", alpha=0.9))

    axL = fig.add_subplot(gs[0, 1])
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
    axL.text(0.0, 0.98, "Cluster interpretation",
             ha="left", va="top", fontsize=11, fontweight="bold")
    axL.text(0.0, 0.95, "(top action_label + 3 example clip names)",
             ha="left", va="top", fontsize=9, color="#555")
    top, bot = 0.88, 0.02
    line_h = (top - bot) / n_clusters
    for k in range(n_clusters):
        mask = cidx == k
        if not mask.any():
            continue
        examples = names[mask][:3]
        top_label = Counter(labels[mask]).most_common(1)[0][0]
        n_in = int(mask.sum())
        y_top = top - k * line_h
        axL.add_patch(plt.Circle((0.04, y_top - line_h*0.20), 0.022,
                                  facecolor=cmap(k % 10), edgecolor="black", lw=0.5,
                                  transform=axL.transAxes))
        axL.text(0.04, y_top - line_h*0.20, str(k), ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white",
                 transform=axL.transAxes)
        axL.text(0.10, y_top - line_h*0.20,
                 f"cluster {k}   (n={n_in},  top label: {top_label!r})",
                 ha="left", va="center", fontsize=9.5, fontweight="bold", color="#333")
        for j, name in enumerate(examples):
            short = (str(name)[:42] + "…") if len(str(name)) > 42 else str(name)
            axL.text(0.11, y_top - line_h*0.45 - j*line_h*0.20,
                     f"• {short}", ha="left", va="center",
                     fontsize=8, color="#555", family="monospace")
    return fig


def label_alignment(X: np.ndarray, true_labels: np.ndarray,
                     *, n_clusters: int = 12, seed: int = 0) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from collections import Counter
    pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X).labels_
    uniq = sorted(set(true_labels))
    enc = {l: i for i, l in enumerate(uniq)}
    true_int = np.array([enc[l] for l in true_labels])
    purity = sum(Counter(true_labels[pred == k]).most_common(1)[0][1]
                  for k in range(n_clusters) if (pred == k).any()) / len(pred)
    return {"ari": float(adjusted_rand_score(true_int, pred)),
            "nmi": float(normalized_mutual_info_score(true_int, pred)),
            "purity": float(purity),
            "n_evaluated": int(len(pred)),
            "n_clusters": int(n_clusters),
            "n_true_labels": len(uniq)}


def silhouette_per_variant(ckpts: Dict[str, str], subset: Subset, *,
                            seed: int = 0, batch_size: int = 32) -> Dict[str, float]:
    from sklearn.metrics import silhouette_score
    out = {}
    for v, ckpt in ckpts.items():
        try:
            X, _, labels = extract_pair_embeddings(ckpt, subset, batch_size=batch_size)
            if len(set(labels)) < 2:
                out[v] = float("nan"); continue
            uniq = sorted(set(labels))
            enc  = {l: i for i, l in enumerate(uniq)}
            y    = np.array([enc[l] for l in labels])
            out[v] = float(silhouette_score(X, y))
        except Exception as e:
            print(f"[{v}] silhouette failed: {e}")
            out[v] = float("nan")
    return out
