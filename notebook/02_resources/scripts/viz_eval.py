"""Plotting helpers for NB02 §1–§5.

All functions return matplotlib Figure (NB cells use show_and_save).

Public API:
    plot_train_loss_overlay_from_tb(ckpt_root, variants) -> Figure
    plot_per_family_mpjpe(mpjpe_table, test_set='A-eval')  -> Figure
    plot_k_vs_a_gap(mpjpe_table)                            -> Figure
    worst_k_gallery(ckpt_path, subset, *, k=5)              -> Figure
    plot_pair_heatmap_4way(heatmaps_dict)                   -> Figure
    annotate_modular_structure(heatmap)                     -> Figure
    plot_damage_sweep(damage_table)                         -> Figure
    damage_reconstruction_compare(ckpts, subset, masked_joints) -> Figure
    plot_silhouette_bars(silh_dict)                         -> Figure
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _eval_common import REPO

# Make 01_resources/scripts importable for skeleton/joint helpers
SCRIPTS_01 = REPO / "notebook/01_resources/scripts"
if str(SCRIPTS_01) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_01))


# Variant-color palette (consistent across plots)
VARIANT_COLORS = {
    "full":          "#1f77b4",   # blue (main)
    "opm_only":      "#2ca02c",   # green
    "triangle_only": "#d62728",   # red
    "pair_static":   "#9467bd",   # purple
    "axial_only":    "#7f7f7f",   # gray
    "baseline":      "#8c564b",   # brown
}


# ---------- §1.4 train loss overlay (reads loss_history.json — paper-grade) ----

def load_loss_history(ckpt_dir: Path) -> dict | None:
    """Read structured loss history written by train_runner.

    Returns None if missing. Falls back to TB if explicitly requested via
    plot_train_loss_overlay_from_tb. JSON is the canonical source for paper.
    """
    import json
    p = Path(ckpt_dir) / "loss_history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def plot_train_loss_overlay_from_tb(ckpt_root: Path, variants: List[str],
                                      *, source: str = "json") -> plt.Figure:
    """Overlay training loss curves for all variants.

    source='json' (default) — loss_history.json (paper-grade, structured)
    source='tb'              — TensorBoard events (fallback)
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n_variants_plotted = 0
    for v in variants:
        steps, loss = None, None
        if source == "json":
            hist = load_loss_history(Path(ckpt_root) / v)
            if hist and hist.get("entries"):
                steps = [e["step"] for e in hist["entries"]]
                loss  = [e["loss"] for e in hist["entries"]]
        if steps is None:
            # Fall back to TB — TB now lives in 02_resources/tensorboard/<variant>/
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            tb_dir = Path(ckpt_root).parent / "tensorboard" / v
            if not tb_dir.exists():
                # Backward compat: legacy nested location
                tb_dir = Path(ckpt_root) / v / "tensorboard"
            if tb_dir.exists():
                ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 4096}); ea.Reload()
                if "train/loss" in ea.Tags().get("scalars", []):
                    events = ea.Scalars("train/loss")
                    steps = [e.step for e in events]
                    loss  = [e.value for e in events]
        if steps:
            ax.plot(steps, loss, label=v, color=VARIANT_COLORS.get(v, None), lw=1.6)
            n_variants_plotted += 1
    ax.set_xlabel("step"); ax.set_ylabel("masked-pos MSE  (loss)")
    ax.set_title(f"Training loss — {n_variants_plotted} variants  (source: {source})")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


# ---------- §2.2 per-family MPJPE bar table ---------------------------------

def plot_per_family_mpjpe(mpjpe_table: Dict, test_set: str = "A-eval") -> plt.Figure:
    """Bar chart: groups = families, bars = variants. mpjpe_table[v][test_set] = mpjpe_eval result."""
    families = sorted({fam for v in mpjpe_table
                       for fam in mpjpe_table[v][test_set]["mpjpe_per_family"]},
                      key=lambda f: -max(
                          mpjpe_table[v][test_set]["n_per_family"].get(f, 0)
                          for v in mpjpe_table
                      ))
    variants = list(mpjpe_table.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(families)*0.9), 4.5))
    x = np.arange(len(families))
    bar_w = 0.8 / max(1, len(variants))
    for i, v in enumerate(variants):
        vals = [mpjpe_table[v][test_set]["mpjpe_per_family"].get(f, 0) for f in families]
        ax.bar(x + i*bar_w - 0.4 + bar_w/2, vals, bar_w,
               color=VARIANT_COLORS.get(v, None), label=v)

    # n_per_family on top of x labels
    n_top = mpjpe_table[variants[0]][test_set]["n_per_family"]
    xtick_labels = [f"{f}\n(n={n_top.get(f, 0)})" for f in families]
    ax.set_xticks(x); ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylabel("MPJPE (m)")
    ax.set_title(f"Per-family MPJPE on {test_set}")
    ax.legend(loc="best", fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------- §2.3 K-A gap monitor --------------------------------------------

def plot_k_vs_a_gap(mpjpe_table: Dict) -> plt.Figure:
    """Scatter: x = K-eval, y = A-eval. Diagonal = no gap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys, vs = [], [], []
    for v, perset in mpjpe_table.items():
        if "K-eval" not in perset or "A-eval" not in perset:
            continue
        xs.append(perset["K-eval"]["mpjpe_overall"])
        ys.append(perset["A-eval"]["mpjpe_overall"])
        vs.append(v)
    if not xs:
        ax.text(0.5, 0.5, "no K-eval/A-eval data", ha="center")
        return fig
    lo = min(xs + ys) * 0.95; hi = max(xs + ys) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="K = A (no gap)")
    for x, y, v in zip(xs, ys, vs):
        ax.scatter([x], [y], s=120, color=VARIANT_COLORS.get(v, "gray"),
                   edgecolor="black", lw=1)
        ax.annotate(v, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("K-eval MPJPE (m)   [synthetic, balanced]")
    ax.set_ylabel("A-eval MPJPE (m)   [real labeled]")
    ax.set_title("K-A gap monitor (above diagonal = synthetic-overfit alarm)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------- §2.4 worst-K reconstruction gallery (mesh) ----------------------

def worst_k_gallery(ckpt_path: str, subset, *, k: int = 5,
                     batch_size: int = 32, device: str = "cuda") -> plt.Figure:
    """Render the K worst-MPJPE clips: gt vs predicted mesh side-by-side.

    Per clip, render middle frame as side-by-side pyrender mesh image.
    """
    import torch
    from torch.utils.data import Subset
    from _eval_common import load_eval_model
    from lbs import load_skin, lbs_apply
    from mesh_render import _look_at
    import pyrender, trimesh

    skin = load_skin(REPO / "notebook/01_resources/assets/skin_standard.npz")
    model, _, variant = load_eval_model(ckpt_path, device=device)
    dev = next(model.parameters()).device
    base_ds = subset.dataset

    pos_mean = torch.from_numpy(base_ds.mean[0]).to(dev)
    pos_std  = torch.from_numpy(base_ds.std[0]).to(dev)

    # Score each clip by MPJPE (FM 1-step denoise)
    from _eval_common import eval_loader, fm_one_step_denoise
    scores, idxs = [], list(subset.indices)
    with torch.no_grad():
        loader = eval_loader(subset, batch=batch_size)
        for bi, batch in enumerate(loader):
            x_pos_1  = batch["pos"].to(dev)
            x_rot_1  = batch["rot6d"].to(dev)
            x_root_1 = batch["root"].to(dev)
            pred, _, _ = fm_one_step_denoise(model, x_pos_1, x_rot_1, x_root_1, t_value=0.95)
            err = ((pred * pos_std + pos_mean - (x_pos_1 * pos_std + pos_mean))
                    .norm(dim=-1).mean(dim=(1, 2))).cpu().numpy()
            scores.extend(err.tolist())
    order = np.argsort(scores)[::-1][:k]   # worst K
    worst_global_idx = [idxs[i] for i in order]

    # Open the source npz to grab GT global rotations + world joints for skinning
    from _eval_common import DATA
    raw = np.load(DATA["amass"]["npz"], allow_pickle=True)

    fig, axes = plt.subplots(k, 2, figsize=(7, 2.6*k), facecolor="white")
    if k == 1:
        axes = axes[None, :]

    cam_dist = 3.0
    cam_pose = _look_at(np.array([0.0, 0.4, cam_dist]), np.array([0.0, 0.4, 0.0]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.0)
    light = pyrender.DirectionalLight(intensity=4.0)
    r = pyrender.OffscreenRenderer(viewport_width=300, viewport_height=300)
    try:
        for row, gi in enumerate(worst_global_idx):
            t = raw["joints_world"].shape[1] // 2
            for col, src in enumerate(["gt", "pred"]):
                if src == "gt":
                    grot = raw["global_rot_mats"][gi][t]
                    jts  = raw["joints_world"][gi][t]
                else:
                    # FM 1-step denoise on this single clip
                    with torch.no_grad():
                        item = base_ds[gi]
                        x_pos_1  = item["pos"].unsqueeze(0).to(dev)
                        x_rot_1  = item["rot6d"].unsqueeze(0).to(dev)
                        x_root_1 = item["root"].unsqueeze(0).to(dev)
                        pred_norm, _, _ = fm_one_step_denoise(
                            model, x_pos_1, x_rot_1, x_root_1, t_value=0.95
                        )
                        pred_norm = pred_norm.squeeze(0)
                        pred_pos = (pred_norm * pos_std + pos_mean).cpu().numpy()  # [T, J, 3]
                    grot = raw["global_rot_mats"][gi][t]
                    jts  = pred_pos[t] - pred_pos[t, 0:1]   # pelvis-aligned
                verts = lbs_apply(skin, grot, jts)
                scene = pyrender.Scene(bg_color=[1,1,1,1], ambient_light=[0.4]*3)
                mesh = trimesh.Trimesh(vertices=verts, faces=skin["faces"], process=False)
                mesh.visual.vertex_colors = ([180,180,220,255] if src == "gt" else [220,180,180,255])
                scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
                scene.add(cam, pose=cam_pose); scene.add(light, pose=cam_pose)
                color, _ = r.render(scene)
                axes[row, col].imshow(color)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title("GT" if src == "gt" else "Pred", fontsize=11)
            axes[row, 0].set_ylabel(f"clip {gi}\nMPJPE={scores[order[row]]:.3f}m",
                                     fontsize=9, rotation=0, ha="right", va="center", labelpad=40)
    finally:
        r.delete()

    fig.suptitle(f"{variant}: worst-{k} reconstructions", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- §3.1 4-way pair heatmap -----------------------------------------

def plot_pair_heatmap_4way(heatmaps: Dict[str, np.ndarray]) -> plt.Figure:
    """4-way pair heatmap of L2-norm pair tensors (matches §9.2 methodology).

    Heatmaps are per-clip L2 norms then averaged across n_clips and symmetrized
    (eval_pair_geometry.extract_pair_tensor). Values are positive → use viridis
    colormap with a SHARED max so magnitude is comparable across variants.
    """
    n = len(heatmaps)
    fig, axes = plt.subplots(1, n, figsize=(3.6*n, 4.0), facecolor="white")
    if n == 1:
        axes = [axes]

    valid = [h for h in heatmaps.values() if h is not None]
    vmax = max(h.max() for h in valid) if valid else 1
    for ax, (variant, H) in zip(axes, heatmaps.items()):
        if H is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{variant}\n(no pair)", ha="center", va="center",
                    fontsize=11, style="italic")
            continue
        im = ax.imshow(H, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(variant, fontsize=11, fontweight="bold",
                      color=VARIANT_COLORS.get(variant, "black"))
        ax.set_xlabel(f"max={H.max():.2f}  mean={H.mean():.2f}",
                      fontsize=8, color="#555")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    fig.suptitle("Pair tensor L2 norm [J×J] — 4-way ablation  (per-clip norm, averaged)",
                  fontsize=12, fontweight="bold")
    return fig


# ---------- §3.4.1 6-way PCA grid (per-variant comparison) ------------------

def plot_pca_6way(
    ckpts: Dict[str, str], subset, *,
    embedding: str = "msa_pool",
    n_clips: int = 300, batch_size: int = 32, seed: int = 0,
) -> plt.Figure:
    """One PCA scatter per variant (2×3 grid), colored by action_label.

    embedding:
        'pair_norm' — anatomical structure (Claim 2 territory; family clusters NOT expected)
        'msa_pool'  — encoder embedding (motion style; family clusters expected if model emerges)

    Useful for "did the trunk emerge action structure?" comparison across variants.
    """
    import sys
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(REPO / "notebook/02_resources/scripts"))
    from eval_style_emerg import extract_pair_embeddings

    n_var = len(ckpts)
    nrows = 2; ncols = (n_var + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), facecolor="white")
    axes = np.atleast_1d(axes).flatten()

    # Use a global label color palette so legends are consistent across panels
    use_idx = subset.indices[:n_clips]
    labels_all = np.asarray([str(subset.dataset.action_label[i]) for i in use_idx])
    families = sorted(set(labels_all))
    cmap = plt.get_cmap("tab10")
    color_of = {f: cmap(k % 10) for k, f in enumerate(families)}

    for i, (variant, ckpt) in enumerate(ckpts.items()):
        ax = axes[i]
        try:
            X, names, labels = extract_pair_embeddings(
                str(ckpt), subset, batch_size=batch_size, embedding=embedding,
            )
            X = X[:n_clips]; labels = labels[:n_clips]
            pca = PCA(n_components=2, random_state=seed).fit(X)
            XY = pca.transform(X)
            v1, v2 = pca.explained_variance_ratio_
            for f in sorted(set(labels)):
                m = labels == f
                ax.scatter(XY[m, 0], XY[m, 1], s=14, alpha=0.7,
                           color=color_of.get(f, "gray"), label=f"{f} ({m.sum()})")
            ax.set_xlabel(f"PCA-1   ({v1*100:.1f}% var)", fontsize=8)
            ax.set_ylabel(f"PCA-2   ({v2*100:.1f}% var)", fontsize=8)
            ax.set_title(f"{variant}", fontsize=11, fontweight="bold",
                          color=VARIANT_COLORS.get(variant, "black"))
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(loc="best", fontsize=7, ncol=2)
        except Exception as e:
            ax.text(0.5, 0.5, f"{variant}\n{type(e).__name__}: {str(e)[:60]}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, style="italic")
            ax.set_xticks([]); ax.set_yticks([])

    for j in range(n_var, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"6-variant PCA — embedding={embedding}  "
                  f"(N={min(n_clips, len(use_idx))} clips, colored by action_label)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- §3.2 annotated modular structure --------------------------------

def annotate_modular_structure(heatmap: np.ndarray) -> plt.Figure:
    """Overlay rectangles for spine + 4 limb clusters + L/R Hand hubs.

    SOMA77 joint groups (approximate, from JOINT_NAMES order):
      spine:    0 (Pelvis), 3 (Spine1), 6 (Spine2), 9 (Spine3), 12 (Neck), 15 (Head)
      L arm:    13 (L_Collar), 16 (L_Shoulder), 18 (L_Elbow), 20 (L_Wrist), 22-36 (L hand)
      R arm:    14 (R_Collar), 17 (R_Shoulder), 19 (R_Elbow), 21 (R_Wrist), 37-51 (R hand)
      L leg:    1 (L_Hip), 4 (L_Knee), 7 (L_Ankle), 10 (L_Foot)
      R leg:    2 (R_Hip), 5 (R_Knee), 8 (R_Ankle), 11 (R_Foot)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if heatmap is None:
        ax.text(0.5, 0.5, "no pair tensor", ha="center", va="center")
        return fig

    ax.imshow(heatmap, cmap="viridis", vmin=0, vmax=heatmap.max(), aspect="equal")

    groups = [
        ("spine",  [0, 3, 6, 9, 12, 15],          "#444444"),
        ("L arm",  [13, 16, 18, 20] + list(range(22, 37)), "#1f77b4"),
        ("R arm",  [14, 17, 19, 21] + list(range(37, 52)), "#ff7f0e"),
        ("L leg",  [1, 4, 7, 10],                 "#2ca02c"),
        ("R leg",  [2, 5, 8, 11],                 "#d62728"),
        ("L Hand hub", [20] + list(range(22, 37)), "#9467bd"),
        ("R Hand hub", [21] + list(range(37, 52)), "#8c564b"),
    ]
    for name, joints, col in groups:
        joints_in_range = [j for j in joints if j < heatmap.shape[0]]
        if not joints_in_range:
            continue
        lo, hi = min(joints_in_range), max(joints_in_range)
        # Diagonal box
        rect = Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                          fill=False, edgecolor=col, lw=1.8)
        ax.add_patch(rect)
        ax.text(lo - 0.5, lo - 1.5, name, color=col, fontsize=8, fontweight="bold")

    ax.set_title("Annotated modular structure (full)", fontsize=11, fontweight="bold")
    ax.set_xlabel("joint j"); ax.set_ylabel("joint i")
    fig.tight_layout()
    return fig


# ---------- §4.2 damage degradation sweep -----------------------------------

def plot_damage_sweep(damage_table: Dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, d in damage_table.items():
        ks = d["k_values"]
        ys = d["mpjpe_per_k"]
        es = d["std_per_k"]
        ax.errorbar(ks, ys, yerr=es, marker="o",
                    color=VARIANT_COLORS.get(v, None), label=v, capsize=3, lw=1.6)
    ax.set_xlabel("K masked joints (random per clip)")
    ax.set_ylabel("MPJPE on masked joints (m)")
    ax.set_title("Motor damage robustness — joint mask sweep")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


# ---------- §4.3 broken-arm reconstruction visual ---------------------------

def damage_reconstruction_compare(
    ckpts: Dict[str, str], subset, masked_joints: List[str],
    *, device: str = "cuda",
) -> plt.Figure:
    """Side-by-side pred-vs-gt mesh under joint-mask damage.

    Picks the first eval clip, masks specified joints by name, runs each ckpt,
    renders meshes side-by-side.
    """
    import torch
    from _eval_common import load_eval_model
    from soma_skeleton import JOINT_NAMES
    from lbs import load_skin, lbs_apply
    from mesh_render import _look_at
    import pyrender, trimesh

    skin = load_skin(REPO / "notebook/01_resources/assets/skin_standard.npz")
    name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    bad_idx = [name_to_idx[n] for n in masked_joints if n in name_to_idx]
    base_ds = subset.dataset

    item = base_ds[subset.indices[0]]
    pos_mean = torch.from_numpy(base_ds.mean[0])
    pos_std  = torch.from_numpy(base_ds.std[0])

    fig, axes = plt.subplots(1, len(ckpts) + 1, figsize=(3.2*(len(ckpts)+1), 3.6),
                              facecolor="white")

    cam_pose = _look_at(np.array([0.0, 0.4, 3.0]), np.array([0.0, 0.4, 0.0]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.0)
    light = pyrender.DirectionalLight(intensity=4.0)
    r = pyrender.OffscreenRenderer(viewport_width=300, viewport_height=300)

    # GT (un-damaged) mesh
    raw_npz = np.load(_safe_data_path(), allow_pickle=True)
    gi = subset.indices[0]
    t = raw_npz["joints_world"].shape[1] // 2
    grot_gt = raw_npz["global_rot_mats"][gi][t]
    jts_gt  = raw_npz["joints_world"][gi][t]
    jts_gt  = jts_gt - jts_gt[0:1]
    verts_gt = lbs_apply(skin, grot_gt, jts_gt)
    axes[0].imshow(_render_mesh(verts_gt, skin["faces"], r, cam, cam_pose, light, color=(180,220,180)))
    axes[0].set_title("GT (un-damaged)", fontsize=10); axes[0].axis("off")

    from _eval_common import fm_sample_conditional_inpaint
    try:
        for col, (variant, ckpt) in enumerate(ckpts.items(), start=1):
            model, _, _ = load_eval_model(ckpt, device=device)
            dev = next(model.parameters()).device
            with torch.no_grad():
                x_pos_1  = item["pos"].unsqueeze(0).to(dev)
                x_rot_1  = item["rot6d"].unsqueeze(0).to(dev)
                x_root_1 = item["root"].unsqueeze(0).to(dev)
                B, T, J, _ = x_pos_1.shape
                bad_bt = torch.zeros(B, T, J, dtype=torch.bool, device=dev)
                bad_bt[:, :, bad_idx] = True
                # FM conditional inpainting: visible joints stay clean GT, bad ones generate from noise
                pred_pos_t, _, _ = fm_sample_conditional_inpaint(
                    model, x_pos_1, x_rot_1, x_root_1, mask=bad_bt, n_steps=16,
                )
                pred_pos = (pred_pos_t.squeeze(0).cpu() * pos_std + pos_mean).numpy()
            jts_pred = pred_pos[t] - pred_pos[t, 0:1]
            verts_p = lbs_apply(skin, grot_gt, jts_pred)
            axes[col].imshow(_render_mesh(verts_p, skin["faces"], r, cam, cam_pose, light,
                                           color=(220,180,180)))
            axes[col].set_title(f"{variant} pred", fontsize=10,
                                 color=VARIANT_COLORS.get(variant, "black"))
            axes[col].axis("off")
    finally:
        r.delete()

    fig.suptitle(f"Joint-mask damage — masked joints: {', '.join(masked_joints)}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def _safe_data_path():
    from _eval_common import DATA
    return DATA["amass"]["npz"]


def _render_mesh(verts, faces, r, cam, cam_pose, light, color=(200,200,220)):
    import pyrender, trimesh
    scene = pyrender.Scene(bg_color=[1,1,1,1], ambient_light=[0.4]*3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = list(color) + [255]
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    scene.add(cam, pose=cam_pose); scene.add(light, pose=cam_pose)
    color_img, _ = r.render(scene)
    return color_img


# ---------- §5.4 silhouette per variant -------------------------------------

def plot_silhouette_bars(silh: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    keys = list(silh.keys())
    vals = [silh[k] for k in keys]
    ax.bar(keys, vals, color=[VARIANT_COLORS.get(k, "gray") for k in keys])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("silhouette score (higher = better cluster separation)")
    ax.set_title("Per-variant silhouette on action_label clusters")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
