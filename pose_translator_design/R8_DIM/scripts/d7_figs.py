"""All R8 figures."""
import json, numpy as np
from d0_common import *

L = lambda f: json.load(open(f"{SP}/{f}"))
san = L("nums_d1_sanity.json")
main = L("nums_d2_main.json")
famf = L("nums_d3b_family_fair.json")
fam4 = L("nums_d3_family.json")
part = L("nums_d3_part.json")
anch = L("nums_d5_anchor.json")
sat = L("nums_d6_sat.json")
satb = L("nums_d6b_sat.json")


# ---------------- fig1: estimator calibration ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
g = [r for r in san if r["case"].startswith("gauss_d") and "_n" not in r["case"] and "pow" not in r["case"]]
td = [r["true_d"] for r in g]
for k, (key, lab) in enumerate([("twonn", "TwoNN"), ("mle_k10", "MLE k=10"),
                                ("corrdim", "correlation dim"), ("pca_pr", "PCA participation ratio"),
                                ("pca_d95", "PCA 95% var dim")]):
    ax[0].plot(td, [r[key] for r in g], "o-", color=SERIES[k], label=lab, ms=4)
ax[0].plot([0, 100], [0, 100], "--", color=MUTED, lw=1, label="truth")
ax[0].set(xlabel="true dimension (isotropic Gaussian, n=12000)", ylabel="estimate",
          title="A. estimators on data of known dimension")
ax[0].legend(fontsize=7, frameon=False)

s = [r for r in san if "_n" in r["case"]]
ns = [r["n"] for r in s]
for k, (key, lab) in enumerate([("twonn", "TwoNN"), ("mle_k10", "MLE k=10"), ("corrdim", "corr dim")]):
    ax[1].plot(ns, [r[key] for r in s], "o-", color=SERIES[k], label=lab, ms=4)
ax[1].axhline(50, ls="--", color=MUTED, lw=1, label="truth = 50")
ax[1].set(xscale="log", xlabel="number of samples", ylabel="estimate",
          title="B. all estimators read LOW when samples are few")
ax[1].legend(fontsize=7, frameon=False)
fig.suptitle("Estimator calibration: below ~15 they are trustworthy, above that they under-read",
             y=1.02, fontsize=11)
savefig(fig, "fig1_calibration.png")


# ---------------- fig2: main result, frames vs windows ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
M = {(r["cache"], r["body"], r["T"], r["neigh"]): r for r in main}
bodies = ["human_smpl", "unitree_g1_29dof", "pal_talos"]
Ts = [1, 16, 32]
w = 0.12
for ci, (cache, cl) in enumerate([("cache_xwbc_amass", "AMASS 4000 clips"),
                                  ("cache_xwbc_lafan", "LAFAN 77 clips")]):
    for bi, b in enumerate(bodies):
        y = [M[(cache, b, T, "xclip")]["twonn"] for T in Ts]
        ax[0].plot([0, 1, 2], y, "o-", color=SERIES[bi], ms=5,
                   ls="-" if ci == 0 else "--", alpha=1 if ci == 0 else 0.65,
                   label=f"{b} / {cl}")
ax[0].set(xticks=[0, 1, 2], xticklabels=["1 frame", "16 frames", "32 frames"],
          ylabel="TwoNN intrinsic dim", ylim=(0, 18),
          title="A. 32x more numbers per sample\nbuys only ~1-4 extra dimensions")
ax[0].legend(fontsize=6.5, frameon=False, ncol=1)

# ambient vs intrinsic
amb = [M[("cache_xwbc_amass", "human_smpl", T, "xclip")]["ambient"] for T in Ts]
for k, (key, lab) in enumerate([("ambient", "ambient (raw numbers)"), ("pca_d99", "PCA 99% var"),
                                ("pca_d95", "PCA 95% var"), ("twonn", "TwoNN intrinsic"),
                                ("mle_k10", "MLE k=10 intrinsic")]):
    ax[1].plot([0, 1, 2], [M[("cache_xwbc_amass", "human_smpl", T, "xclip")][key] for T in Ts],
               "o-", color=SERIES[k], ms=5, label=lab)
ax[1].set(yscale="log", xticks=[0, 1, 2], xticklabels=["1 frame", "16 frames", "32 frames"],
          ylabel="dimensions (log)", title="B. AMASS human: raw size vs real size")
ax[1].legend(fontsize=7, frameon=False)
savefig(fig, "fig2_frames_vs_windows.png")


# ---------------- fig3: by action family ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
order = ["other", "walk", "reach", "run", "sit", "jump", "dance"]
for k, (recs, T, lab) in enumerate([(famf, 1, "single frame"), (famf, 32, "32-frame window")]):
    d = {r["family"]: r for r in recs if r["T"] == T}
    fams = [f for f in order if f in d]
    x = np.arange(len(fams))
    ax[0].bar(x + (k - 0.5) * 0.38, [d[f]["twonn"] for f in fams], 0.36,
              color=SERIES[k], label=lab)
ax[0].set(xticks=np.arange(len(order)), xticklabels=order, ylabel="TwoNN intrinsic dim",
          title="A. by action family (matched n=1500 each)")
ax[0].legend(fontsize=8, frameon=False)
for i, f in enumerate(order):
    r = [x for x in famf if x["family"] == f and x["T"] == 1][0]
    if r["sampling_stride"] < 10:
        ax[0].text(i, 0.6, "thin\ndata", ha="center", fontsize=6.5, color="#e34948")

d1 = {r["family"]: r for r in famf if r["T"] == 1}
d32 = {r["family"]: r for r in famf if r["T"] == 32}
for i, f in enumerate(order):
    ax[1].scatter(d1[f]["twonn"], d32[f]["twonn"], s=60, color=SERIES[i % 8], zorder=3)
    ax[1].annotate(f, (d1[f]["twonn"], d32[f]["twonn"]), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
lim = [3, 18]
ax[1].plot(lim, lim, "--", color=MUTED, lw=1)
ax[1].set(xlabel="pose dim (1 frame)", ylabel="motion dim (32 frames)", xlim=lim, ylim=lim,
          title="B. dance/jump gain the most from time")
savefig(fig, "fig3_action_family.png")


# ---------------- fig4: by body part ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
porder = ["hands_wrists", "arms", "legs", "torso_head", "whole"]
for k, T in enumerate([1, 16]):
    d = {r["part"]: r for r in part if r["T"] == T}
    x = np.arange(len(porder))
    ax[0].bar(x + (k - 0.5) * 0.38, [d[p]["twonn"] for p in porder], 0.36, color=SERIES[k],
              label="1 frame" if T == 1 else "16-frame window")
    ax[1].bar(x + (k - 0.5) * 0.38, [d[p]["twonn"] / d[p]["n_joints"] for p in porder], 0.36,
              color=SERIES[k], label="1 frame" if T == 1 else "16-frame window")
ax[0].set(xticks=np.arange(len(porder)), xticklabels=porder, ylabel="TwoNN intrinsic dim",
          title="A. absolute: every part sits near 5-8")
ax[1].set(xticks=np.arange(len(porder)), xticklabels=porder, ylabel="intrinsic dim per joint",
          title="B. per joint: wrists carry ~3x the legs")
for a in ax:
    a.legend(fontsize=8, frameon=False)
    a.tick_params(axis="x", labelrotation=20)
savefig(fig, "fig4_body_part.png")


# ---------------- fig5: saturation ----------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
# A. PCA variance dimension -- this one is NOT distorted by how densely we sample
for k, (T, lab) in enumerate([(1, "1 frame"), (32, "32-frame window")]):
    A = sorted([r for r in sat if r["arm"] == "A_more_clips" and r["T"] == T],
               key=lambda z: z["n_clips"])
    ax[0].plot([r["n_clips"] for r in A], [r["pca_d95"] for r in A], "o-", color=SERIES[k],
               ms=5, label=f"{lab}: 95% var")
    ax[0].plot([r["n_clips"] for r in A], [r["pca_d99"] for r in A], "o--", color=SERIES[k],
               ms=4, alpha=0.6, label=f"{lab}: 99% var")
ax[0].set(xscale="log", yscale="log", xlabel="clips in the pool (fixed n drawn)",
          ylabel="PCA dimension", ylim=(8, 200),
          title="A. PCA dimension is FLAT from 500 to 4000 clips\n(8x the data, +5% dimensions)")
ax[0].legend(fontsize=7, frameon=False)

# B. the NN estimators at fixed n -- goes DOWN, and here is why
A = [r for r in sat if r["arm"] == "A_more_clips" and r["kind"] == "frame"]
B = [r for r in sat if r["arm"] == "B_more_samples"]
ax[1].plot([r["n_clips"] for r in A], [r["twonn"] for r in A], "o-", color=SERIES[1], ms=5,
           label="TwoNN vs #clips (n fixed = 12000)")
ax[1].plot([r["n_clips"] for r in A], [r["mle_k10"] for r in A], "o-", color=SERIES[0], ms=5,
           label="MLE k=10 vs #clips")
ax2 = ax[1].twiny()
ax2.plot([r["n"] for r in B], [r["twonn"] for r in B], "s--", color=SERIES[3], ms=4,
         label="TwoNN vs #samples (clips fixed = 4000)")
ax2.set_xscale("log"); ax2.set_xlabel("samples drawn", color=SERIES[3])
ax2.grid(False)
ax[1].set(xscale="log", xlabel="clips in the pool", ylabel="intrinsic dim",
          title="B. the NN estimators mostly track\nsampling density, not new content")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False)

# C. matched sample size
for k, T in enumerate([1, 32]):
    for j, nc in enumerate([500, 1000, 2000, 4000]):
        r = sorted([x for x in satb if x["n_clips"] == nc and x["T"] == T], key=lambda z: z["n"])
        ax[2].plot([z["n"] for z in r], [z["twonn"] for z in r], "o-", color=SERIES[j],
                   ms=4, ls="-" if T == 1 else "--",
                   label=f"{nc} clips, {'frame' if T == 1 else '32-frame win'}")
ax[2].set(xscale="log", xlabel="samples drawn (matched across clip counts)",
          ylabel="TwoNN intrinsic dim",
          title="C. at matched n, more clips reads LOWER --\nextra clips widen the cloud, not its dimension")
ax[2].legend(fontsize=6, frameon=False, ncol=2)
savefig(fig, "fig5_saturation.png")


# ---------------- fig6: anchors ----------------
fig, ax = plt.subplots(figsize=(8, 4.2))
items = [("motion: AMASS 1 frame", M[("cache_xwbc_amass", "human_smpl", 1, "xclip")]["twonn"], SERIES[0]),
         ("motion: AMASS 32-frame window", M[("cache_xwbc_amass", "human_smpl", 32, "xclip")]["twonn"], SERIES[0]),
         ("motion: LAFAN 32-frame window", M[("cache_xwbc_lafan", "human_smpl", 32, "xclip")]["twonn"], SERIES[0]),
         ("motion: AMASS dance, 32-frame", d32["dance"]["twonn"], SERIES[0])]
A = {r["case"]: r for r in anch}
items += [("images: sklearn digits 8x8 (n=1797)", A["images_digits_8x8"]["twonn"], SERIES[1]),
          ("images: natural patch 8x8 gray", A["images_natural_patch_8x8"]["twonn"], SERIES[1]),
          ("images: natural patch 16x16 gray", A["images_natural_patch_16x16"]["twonn"], SERIES[1]),
          ("audio PROXY (synthetic log-mel)", A["audio_PROXY_synth_logmel64"]["twonn"], SERIES[3])]
lit = [("[lit] MNIST", 13, SERIES[4]), ("[lit] CIFAR-10", 26, SERIES[4]),
       ("[lit] ImageNet", 40, SERIES[4])]
items += lit
y = np.arange(len(items))[::-1]
ax.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.62)
for yy, (_, v, _) in zip(y, items):
    ax.text(v + 0.4, yy, f"{v:.1f}", va="center", fontsize=8, color=INK2)
ax.set(yticks=y, yticklabels=[n for n, _, _ in items], xlabel="TwoNN intrinsic dimension",
       title="Anchors. Blue = our motion data. Orange = images WE measured here.\n"
             "Gold = a synthetic audio proxy (not a real corpus). Pink = published numbers, not measured here.")
ax.tick_params(axis="y", labelsize=8)
savefig(fig, "fig6_anchors.png")


# ---------------- fig7: PCA spectrum ----------------
E = np.load(f"{SP}/eigs_d2.npz")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, tag in enumerate(["cache_xwbc_amass/human_smpl/frame1",
                         "cache_xwbc_amass/human_smpl/window16",
                         "cache_xwbc_amass/human_smpl/window32",
                         "cache_xwbc_lafan/human_smpl/window32"]):
    ev = E[tag]
    ax[0].plot(np.arange(1, len(ev) + 1), ev / ev.sum(), color=SERIES[k], label=tag.split("/", 1)[1] +
               (" (LAFAN)" if "lafan" in tag else ""), lw=1.3)
    ax[1].plot(np.arange(1, len(ev) + 1), np.cumsum(ev) / ev.sum(), color=SERIES[k], lw=1.3)
ax[0].set(xscale="log", yscale="log", xlabel="principal component", ylabel="variance share",
          title="A. spectrum decays smoothly -- no clean cutoff")
ax[0].legend(fontsize=7, frameon=False)
for lv in [0.9, 0.95, 0.99]:
    ax[1].axhline(lv, ls="--", color=MUTED, lw=0.8)
ax[1].set(xscale="log", xlabel="principal component", ylabel="cumulative variance",
          ylim=(0.3, 1.02), title="B. cumulative variance")
savefig(fig, "fig7_pca_spectrum.png")
print("figs done")
