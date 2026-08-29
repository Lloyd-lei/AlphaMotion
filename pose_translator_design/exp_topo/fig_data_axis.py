"""Figure: held-out error vs training clip count for the E21 DATA axis.

Reads runs/E21_data_axis_heldout.json (written by eval_data_axis.py) and plots
zero-shot human_smpl -> unitree_g1_29dof error on the disjoint held-out AMASS cache:
  A  rotation error, overall + per body part (robot->robot as context)
  B  position error, same series
  C  the PAIRED delta vs the 500-clip model (same clips, so clip difficulty cancels)
     against the +/-1 deg single-seed training noise -- the panel that says whether
     the data axis moved anything real.
"""
from __future__ import annotations
import json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

SRC = "/var/tmp/alphamotion/topo_exp/runs/E21_data_axis_heldout.json"
OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R7_LATENT/FIG_data_axis.png')
HUM, ROB = "human_smpl->unitree_g1_29dof", "unitree_h1->unitree_g1_29dof"
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SERIES = [("all", "overall", "#2a78d6"), ("legs", "legs", "#eb6834"),
          ("arms", "arms", "#1baf7a"), ("wrists", "wrists", "#eda100")]
NOISE_DEG = 1.0     # single-seed run-to-run spread observed on this harness


def style(ax, x, xlabel="training clips (log scale)"):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=9.5, length=3)


def direct_labels(ax, x_end, entries):
    """entries: [(y, text, color)] -- nudged apart so close end-points stay legible."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.068
    entries = sorted(entries, key=lambda t: t[0])
    for i in range(1, len(entries)):
        entries[i] = (max(entries[i][0], entries[i - 1][0] + gap),) + entries[i][1:]
    for ypos, txt, col in entries:
        ax.annotate(txt, (x_end, ypos), xytext=(9, 0), textcoords="offset points",
                    fontsize=9.5, color=col, va="center", ha="left", annotation_clip=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    d = json.load(open(args.src))
    runs = sorted(d["runs"].items(), key=lambda kv: kv[1]["train_clips"])
    x = [r["train_clips"] for _, r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.82]})
    for ax, key, unit, title in [
            (axes[0], "rot_deg", "deg", "A · Rotation error (mean geodesic)"),
            (axes[1], "pos_cm", "cm", "B · Position error")]:
        labels = []
        for part, label, color in SERIES:
            y = [r[HUM][part][key] for _, r in runs]
            ax.plot(x, y, color=color, lw=2, marker="o", ms=6.5, mew=1.6,
                    mfc=SURFACE, mec=color, zorder=3, clip_on=False)
            labels.append((y[-1], f"{label}  {y[-1]:.1f}", INK))
        yr = [r[ROB]["all"][key] for _, r in runs]
        ax.plot(x, yr, color=INK3, lw=1.8, ls=(0, (4, 3)), marker="o", ms=5.5,
                mew=1.4, mfc=SURFACE, mec=INK3, zorder=2, clip_on=False)
        labels.append((yr[-1], f"h1→g1  {yr[-1]:.1f}", INK2))
        style(ax, x)
        ax.set_xlim(x[0] * 0.88, x[-1] * 1.75)
        ax.set_ylabel(f"held-out error ({unit})", fontsize=10, color=INK2)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=9)
        direct_labels(ax, x[-1], labels)

    # ---- C: paired delta vs the smallest model, same clips ----
    ax = axes[2]
    pd = d.get("paired_delta_vs_smallest")
    base_clips = x[0]
    ax.axhspan(-NOISE_DEG, NOISE_DEG, color=INK3, alpha=0.13, lw=0, zorder=0)
    ax.axhline(0, color="#b9b8b3", lw=1.2, zorder=1)
    if pd:
        xs = [d["runs"][r]["train_clips"] for r in pd["deltas"]]
        ys = [v["mean_delta_deg"] for v in pd["deltas"].values()]
        lo = [v["mean_delta_deg"] - v["ci95_lo"] for v in pd["deltas"].values()]
        hi = [v["ci95_hi"] - v["mean_delta_deg"] for v in pd["deltas"].values()]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs, ys = [xs[i] for i in order], [ys[i] for i in order]
        lo, hi = [lo[i] for i in order], [hi[i] for i in order]
        ax.errorbar([base_clips] + xs, [0.0] + ys, yerr=[[0.0] + lo, [0.0] + hi],
                    color="#2a78d6", lw=2, marker="o", ms=6.5, mew=1.6, mfc=SURFACE,
                    mec="#2a78d6", ecolor="#2a78d6", elinewidth=1.4, capsize=4,
                    capthick=1.4, zorder=3, clip_on=False)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:+.2f}", (xi, yi), xytext=(9, -1), textcoords="offset points",
                        fontsize=9.5, color=INK, ha="left", va="center",
                        annotation_clip=False)
    style(ax, x)
    ax.set_xlim(x[0] * 0.88, x[-1] * 1.55)
    ax.margins(y=0.22)
    ax.set_ylabel("Δ rotation error vs 500-clip model (deg)", fontsize=10, color=INK2)
    ax.set_title("C · Paired Δ, same held-out clips", fontsize=11.5, color=INK,
                 loc="left", pad=9)
    ax.annotate(f"±{NOISE_DEG:.0f}° single-seed training noise", (x[0] * 0.9, NOISE_DEG),
                xytext=(2, 4), textcoords="offset points", fontsize=8.5, color=INK2,
                ha="left", va="bottom")
    ax.annotate("bars = 95% CI, clip-block bootstrap (eval sampling only)",
                (0.025, 0.025), xycoords="axes fraction", fontsize=8.5, color=INK2,
                ha="left", va="bottom")

    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=6, mfc=SURFACE,
                          mec=c, mew=1.6, label=f"human→g1  {l}")
               for _, l, c in SERIES]
    handles.append(plt.Line2D([], [], color=INK3, lw=1.8, ls=(0, (4, 3)), marker="o",
                              ms=5.5, mfc=SURFACE, mec=INK3, mew=1.4,
                              label="unitree_h1→g1  overall (context)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("DATA axis: zero-shot retarget error on 600 held-out AMASS clips, "
                 "disjoint from every training subset",
                 fontsize=12.5, color=INK, x=0.045, ha="left", y=0.975)
    fig.text(0.045, 0.912,
             f"target unitree_g1_29dof, an embodiment also held out of training · "
             f"{d['n_eval']} eval frames over {d['heldout_clips']} clips, seed 0 · "
             f"same rig, specs and conventions as the training cache",
             fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0.0, 0.075, 0.995, 0.885))
    fig.savefig(args.out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
