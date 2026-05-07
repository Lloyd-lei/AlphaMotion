"""Visualization for NB1.

  - plot_family_distribution     bar chart with %% highlight
  - plot_other_subset_breakdown  stacked bar of "other" by AMASS subset
  - plot_other_unsupervised      PCA→KMeans scatter on "other" raw-joint features
  - plot_alphamotion_arch        dual-track Evoformer-style diagram (matplotlib auto preview;
                                  paper-grade source: 01_resources/diagrams/alphamotion_arch.drawio)
  - plot_baseline_arch           plain transformer encoder stack
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
import numpy as np


# ===== distributions ========================================================

def plot_family_distribution(
    dist: Dict[str, int],
    *,
    title: str = "Action family distribution",
    highlight: str = "other",
    figsize=(10, 4),
):
    fams   = sorted(dist.keys(), key=lambda k: -dist[k])
    counts = [dist[f] for f in fams]
    total  = sum(counts) or 1
    colors = ["#cc3333" if f == highlight else "#3366aa" for f in fams]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(fams, counts, color=colors)
    ax.set_title(title); ax.set_ylabel("# clips")
    ax.tick_params(axis="x", rotation=30)
    if highlight in dist:
        pct = dist[highlight] / total * 100
        i = fams.index(highlight)
        ax.text(i, dist[highlight], f" {pct:.0f}%", ha="center", va="bottom", color="#cc3333")
    fig.tight_layout()
    return fig


def plot_other_subset_breakdown(
    breakdown: Dict[str, int],
    *,
    title: str = "AMASS 'other' clips by source subset",
    top_k: int = 12,
    figsize=(10, 4),
):
    """Horizontal bar of top-K AMASS subsets contributing to 'other' family."""
    items = sorted(breakdown.items(), key=lambda kv: -kv[1])[:top_k]
    labels = [k for k, _ in items][::-1]
    counts = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, counts, color="#cc3333")
    ax.set_title(title); ax.set_xlabel("# 'other' clips in subset")
    for i, v in enumerate(counts):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_other_unsupervised(
    features:     np.ndarray,    # [N, D] flattened mid-frame joint features
    sample_names: np.ndarray,    # [N,]
    subsets:      np.ndarray,    # [N,]
    *,
    n_clusters: int = 8,
    n_samples:  int = 800,
    seed:       int = 0,
    title:      str = "Unsupervised structure within 'other' (PCA→KMeans preview — PLACEHOLDER)",
    figsize=(15, 6),
):
    """PCA→2D + KMeans preview on raw mid-frame joint features.

    Honest version: shows explained-variance ratio in axis labels, and a side
    panel listing 3 sample names + dominant subset per cluster so the reviewer
    can interpret what each cluster IS (not just colored dots).

    This is NOT trained-model encoder embeddings. Those come from NB02 §5,
    where the same kind of plot is the real answer to "does 'other' have
    discoverable style structure?".
    """
    from collections import Counter
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    if features.shape[0] > n_samples:
        sub = rng.choice(features.shape[0], size=n_samples, replace=False)
        features = features[sub]
        sample_names = sample_names[sub]
        subsets      = subsets[sub]

    pca_obj = PCA(n_components=2, random_state=seed).fit(features)
    pca     = pca_obj.transform(features)
    var1, var2 = pca_obj.explained_variance_ratio_
    km     = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(features)
    labels = km.labels_

    # Layout: scatter (left) + interpretation legend (right)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.6], wspace=0.05)
    ax  = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("tab10")

    sc = ax.scatter(pca[:, 0], pca[:, 1], c=labels, cmap=cmap,
                    s=18, alpha=0.75, edgecolor="none")
    ax.set_xlabel(f"PCA-1   ({var1*100:.1f}% var)")
    ax.set_ylabel(f"PCA-2   ({var2*100:.1f}% var)")
    ax.set_title(
        f"{title}\n"
        f"N={features.shape[0]} 'other' clips, D={features.shape[1]} "
        f"(77 joints × 3, mid-frame, pelvis-centered) → KMeans in {features.shape[1]}-D, "
        f"PCA only for 2-D viz",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

    # cluster centroids in PCA space
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cx, cy = pca[mask, 0].mean(), pca[mask, 1].mean()
            ax.text(cx, cy, str(k), ha="center", va="center", fontsize=11,
                    color="black", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=cmap(k % 10),
                              boxstyle="circle,pad=0.25", alpha=0.9))

    # ---- Interpretation panel ----
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.set_xlim(0, 1); ax_legend.set_ylim(0, 1); ax_legend.axis("off")
    ax_legend.text(0.0, 0.985,
                   "Cluster interpretation",
                   ha="left", va="top", fontsize=11, fontweight="bold")
    ax_legend.text(0.0, 0.95,
                   "3 example clip names + dominant AMASS subject ID",
                   ha="left", va="top", fontsize=9, color="#555")

    # Reserve top 12% for header; rest for clusters
    top, bot = 0.88, 0.02
    line_h = (top - bot) / n_clusters
    for k in range(n_clusters):
        mask = labels == k
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        examples = sample_names[mask][:3]
        sub_top  = Counter(subsets[mask]).most_common(1)[0][0]

        y_top = top - k * line_h        # top of this cluster's row

        # Color circle marker on far left
        ax_legend.add_patch(plt.Circle((0.04, y_top - line_h*0.20), 0.022,
                                        facecolor=cmap(k % 10), edgecolor="black", lw=0.5,
                                        transform=ax_legend.transAxes))
        ax_legend.text(0.04, y_top - line_h*0.20, str(k), ha="center", va="center",
                       fontsize=10, fontweight="bold", color="white",
                       transform=ax_legend.transAxes)

        head = f"cluster {k}   (n={n_in},  top subject: {sub_top!r})"
        ax_legend.text(0.10, y_top - line_h*0.20, head, ha="left", va="center",
                       fontsize=9.5, fontweight="bold", color="#333")
        for j, name in enumerate(examples):
            short = (str(name)[:42] + "…") if len(str(name)) > 42 else str(name)
            ax_legend.text(0.11, y_top - line_h*0.45 - j*line_h*0.20,
                           f"• {short}", ha="left", va="center",
                           fontsize=8, color="#555", family="monospace")

    return fig


# ===== architecture diagrams (matplotlib preview) ============================

C_MSA   = "#1f77b4"
C_PAIR  = "#ff7f0e"
C_GRAY  = "#444444"
C_LIGHT = "#fafafa"
C_FRAME = "#bbbbbb"
HEAD_COLORS = {"pos": "#3a8a3a", "rot6d": "#a13aa1", "root": "#a4673a"}


def _rounded(ax, xy, w, h, label, *, fc, fontsize=10, fontweight="bold", text="white"):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                         fc=fc, ec=fc, lw=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(xy[0]+w/2, xy[1]+h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text, zorder=3)


def _attn(ax, x, y, r, label_above, *, color):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.8, zorder=2))
    ax.text(x, y, "×", ha="center", va="center", fontsize=14, color=color,
            fontweight="bold", zorder=3)
    ax.text(x, y + r + 0.15, label_above, ha="center", va="bottom",
            fontsize=8.5, color=color, fontweight="bold")


def _ffn(ax, x, y, w, h, *, color):
    skew = 0.10
    pts = [(x-w/2, y-h/2), (x+w/2-skew, y-h/2),
           (x+w/2, y+h/2), (x-w/2+skew, y+h/2)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=color, lw=1.8, zorder=2))
    ax.text(x, y, "FFN", ha="center", va="center",
            fontsize=10, color=color, fontweight="bold")


def _opm(ax, x, y, w, h):
    pts = [(x-w/2, y+h/2), (x+w/2, y+h/2),
           (x+w*0.30, y-h/2), (x-w*0.30, y-h/2)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=C_PAIR, lw=1.8, zorder=2))
    ax.text(x, y, "OPM", ha="center", va="center",
            fontsize=10, color=C_PAIR, fontweight="bold")


def _tri(ax, x, y, r, label_above):
    pts = [(x-r, y-r*0.7), (x+r, y-r*0.7), (x, y+r*0.9)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=C_PAIR, lw=1.8, zorder=2))
    ax.text(x, y + r + 0.15, label_above, ha="center", va="bottom",
            fontsize=8.5, color=C_PAIR, fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, *, color="#444", style="-", lw=1.5, label=None, label_offset=(0, 0.15)):
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle="-|>", mutation_scale=12,
                          color=color, lw=lw, linestyle=style, zorder=4)
    ax.add_patch(arr)
    if label:
        lx = (x0 + x1) / 2 + label_offset[0]
        ly = (y0 + y1) / 2 + label_offset[1]
        ax.text(lx, ly, label, ha="center", fontsize=8.5,
                color=color, style="italic")


def plot_alphamotion_arch(
    cfg, T: int, J: int,
    *,
    mesh_inset_paths: List[str] = None,
    title: str = "AlphaMotion (full) — pair-first whole-body controller",
):
    """Dual-track Evoformer-style preview. Paper-grade source = drawio file.

    cfg: MotionFormerConfig (uses cfg.hidden, cfg.pair_hidden, cfg.depth)
    """
    fig = plt.figure(figsize=(17, 8.5), facecolor="white")
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.6, 5.5, 1.6], wspace=0.05)

    # ---- (a) Input panel ----
    axA = fig.add_subplot(gs[0, 0])
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axA.text(0.5, 0.97, "(a) Input motion", ha="center",
             fontsize=13, fontweight="bold")

    # mesh insets (axes-coords children)
    if mesh_inset_paths:
        bbox = axA.get_position()
        n = len(mesh_inset_paths)
        slot_w = 0.95 * bbox.width / n
        for i, p in enumerate(mesh_inset_paths):
            sub = fig.add_axes([
                bbox.x0 + 0.025*bbox.width + i*slot_w,
                bbox.y0 + 0.55*bbox.height,
                slot_w*0.85, 0.32*bbox.height,
            ])
            try:
                sub.imshow(plt.imread(str(p)))
            except Exception:
                sub.text(0.5, 0.5, "(inset)", ha="center", va="center", fontsize=9)
            sub.set_xticks([]); sub.set_yticks([])
            for s in sub.spines.values():
                s.set_color(C_FRAME); s.set_linewidth(0.8)
            sub.set_xlabel(f"$t = {[0, T//3, 2*T//3][i]}$", fontsize=9)

    axA.text(0.5, 0.45,
             r"$\mathbf{x}_{\rm pos} \in \mathbb{R}^{B\times T\times J\times 3}$" + "\n"
             r"$\mathbf{x}_{\rm rot} \in \mathbb{R}^{B\times T\times J\times 6}$" + "\n"
             r"$\mathbf{x}_{\rm root} \in \mathbb{R}^{B\times T\times 3}$",
             ha="center", va="top", fontsize=11, color=C_GRAY, family="serif")
    axA.text(0.5, 0.18,
             f"$T={T}$ frames\n$J={J}$ joints\n(SOMA-77 skeleton)",
             ha="center", va="top", fontsize=9, color=C_GRAY, style="italic")

    # ---- (b) Trunk panel ----
    axB = fig.add_subplot(gs[0, 1])
    axB.set_xlim(0, 12); axB.set_ylim(0, 7.5); axB.axis("off")
    axB.text(6, 7.15, f"(b) Pair-First Trunk  (× depth = {cfg.depth} blocks)",
             ha="center", fontsize=13, fontweight="bold")

    # block frame
    axB.add_patch(Rectangle((0.20, 0.40), 11.6, 6.40, fc=C_LIGHT, ec=C_FRAME,
                            lw=1.0, linestyle="--", zorder=1))
    axB.text(11.65, 0.55, f"× {cfg.depth}", ha="right", va="bottom",
             fontsize=12, color=C_GRAY, fontweight="bold")

    # MSA track ----
    y_msa = 5.4
    axB.text(0.45, y_msa+0.85, "MSA tensor",
             ha="left", fontsize=11, color=C_MSA, fontweight="bold")
    axB.text(0.45, y_msa+0.55, f"[B, T, J, H={cfg.hidden}]",
             ha="left", fontsize=9, color=C_MSA, family="serif")
    _rounded(axB, (0.45, y_msa-0.30), 1.10, 0.60, "Embed", fc=C_MSA, fontsize=10)
    _attn(axB, 3.20, y_msa,  0.35, "Row Attn (J)\n+ pair bias", color=C_MSA)
    _attn(axB, 5.10, y_msa,  0.35, "Col Attn (T)", color=C_MSA)
    _ffn (axB, 6.85, y_msa,  0.95, 0.60, color=C_MSA)
    _rounded(axB, (8.20, y_msa-0.30), 1.20, 0.60, "Decoder", fc=C_GRAY, fontsize=10)

    # forward arrows (MSA)
    _arrow(axB, 1.55, y_msa, 2.85, y_msa)
    _arrow(axB, 3.55, y_msa, 4.75, y_msa)
    _arrow(axB, 5.45, y_msa, 6.38, y_msa)
    _arrow(axB, 7.32, y_msa, 8.20, y_msa)

    # Pair track ----
    y_pair = 1.6
    axB.text(0.45, y_pair-0.50, "Pair tensor",
             ha="left", fontsize=11, color=C_PAIR, fontweight="bold")
    axB.text(0.45, y_pair-0.80, f"[B, J, J, P={cfg.pair_hidden}]",
             ha="left", fontsize=9, color=C_PAIR, family="serif")
    _rounded(axB, (0.45, y_pair-0.30), 1.10, 0.60, "Pair init", fc=C_PAIR, fontsize=10)
    _tri(axB, 5.10, y_pair, 0.40, "Δ-Mult (out)")
    _tri(axB, 6.85, y_pair, 0.40, "Δ-Attn (start)")
    _ffn(axB, 8.55, y_pair, 0.95, 0.60, color=C_PAIR)

    # OPM in middle column
    _opm(axB, 3.20, 3.50, 1.10, 0.75)

    # forward arrows (Pair)
    _arrow(axB, 1.55, y_pair, 4.65, y_pair, color=C_PAIR)
    _arrow(axB, 5.50, y_pair, 6.40, y_pair, color=C_PAIR)
    _arrow(axB, 7.25, y_pair, 8.05, y_pair, color=C_PAIR)

    # cross-track: OPM (MSA → Pair, solid orange)
    _arrow(axB, 3.20, y_msa-0.35, 3.20, 3.85, color=C_PAIR, label="OPM (MSA → Pair)",
           label_offset=(1.4, 0.0))
    _arrow(axB, 3.20, 3.10, 3.20, y_pair+0.40, color=C_PAIR)

    # cross-track: Pair → MSA bias (dashed blue)
    _arrow(axB, 2.85, y_pair+0.30, 2.85, y_msa-0.32, color=C_MSA, style="--",
           label="bias", label_offset=(-0.5, 0.0))

    # Pair feedback loop into next block
    loop = mpatches.FancyArrowPatch(
        (9.05, y_pair-0.32), (1.00, y_pair-0.32),
        connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
        mutation_scale=10, color=C_PAIR, lw=1.0, linestyle=":")
    axB.add_patch(loop)
    axB.text(5.0, y_pair-0.95, "feed next block", ha="center", fontsize=8,
             color=C_PAIR, style="italic")

    # ---- (c) Heads panel ----
    axC = fig.add_subplot(gs[0, 2])
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    axC.text(0.5, 0.97, "(c) Heads", ha="center", fontsize=13, fontweight="bold")

    heads = [("pos head",   r"$\hat{p} \in \mathbb{R}^{B \times T \times J \times 3}$"),
             ("rot6d head", r"$\hat{r} \in \mathbb{R}^{B \times T \times J \times 6}$"),
             ("root head",  r"$\hat{r}_{\rm root} \in \mathbb{R}^{B \times T \times 3}$")]
    for i, (name, sh) in enumerate(heads):
        col = list(HEAD_COLORS.values())[i]
        y = 0.78 - i * 0.20
        axC.add_patch(FancyBboxPatch((0.10, y - 0.06), 0.80, 0.14,
                                     boxstyle="round,pad=0.01,rounding_size=0.04",
                                     fc=col, ec=col, alpha=0.90))
        axC.text(0.50, y + 0.025, name, ha="center", va="center",
                 color="white", fontsize=10, fontweight="bold")
        axC.text(0.50, y - 0.035, sh, ha="center", va="center",
                 color="white", fontsize=8, family="serif")

    axC.text(0.5, 0.06,
             r"$FK\ loss\!: \hat{r}, \hat{r}_{\rm root} \to FK \to \hat{p}$",
             ha="center", fontsize=9, color=C_GRAY, style="italic")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    return fig


def plot_baseline_arch(
    cfg, T: int, J: int,
    *,
    title: str = "Baseline — plain transformer encoder stack",
):
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="white")
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis("off")
    ax.text(0.5, 5.30, title, fontsize=13, fontweight="bold")

    # input
    _rounded(ax, (0.30, 1.85), 1.50, 1.80,
             f"Input\n[B, T={T}, J={J},\n3+6+3]", fc=C_GRAY, fontsize=10)
    _arrow(ax, 1.85, 2.75, 2.50, 2.75)
    _rounded(ax, (2.50, 2.20), 1.20, 1.10, "Linear\nEmbed", fc=C_GRAY, fontsize=10)
    _arrow(ax, 3.75, 2.75, 4.40, 2.75)

    # block frame
    ax.add_patch(Rectangle((4.40, 0.50), 4.30, 4.50, fc=C_LIGHT,
                           ec=C_FRAME, lw=1.0, linestyle="--"))
    ax.text(8.65, 0.65, f"× {cfg.depth}", ha="right",
            fontsize=12, color=C_GRAY, fontweight="bold")
    ax.text(6.55, 4.85, "(one block of " + str(cfg.depth) + ")",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    parts = [("Norm", "#999999"),
             ("Multi-Head\nAttention", C_GRAY),
             ("Norm", "#999999"),
             ("FFN", C_GRAY)]
    y0 = 1.10
    h  = 0.75
    gap = 0.10
    for i, (lab, col) in enumerate(parts):
        _rounded(ax, (4.65, y0 + i*(h+gap)), 3.80, h, lab, fc=col, fontsize=10)

    _arrow(ax, 8.70, 2.75, 9.30, 2.75)
    _rounded(ax, (9.30, 2.20), 1.20, 1.10, "Decoder", fc=C_GRAY, fontsize=10)

    # heads
    for i, (key, lab) in enumerate([("pos", "pos head"),
                                     ("rot6d", "rot6d head"),
                                     ("root", "root head")]):
        col = HEAD_COLORS[key]
        y = 3.40 - i * 0.55
        ax.add_patch(FancyBboxPatch((10.80, y - 0.20), 1.80, 0.40,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=col, ec=col, alpha=0.90))
        ax.text(11.70, y, lab, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        _arrow(ax, 10.50, 2.75, 10.80, y, color="#666", lw=1.2)

    return fig
