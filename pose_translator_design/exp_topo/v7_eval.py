"""V7 VERDICT — V7_jepa vs V7_maskfsq, matched at 62.69 bit/s, OAT eliminated.

    v7_eval.py --runs V7_jepa_0807,V7_maskfsq_0807 [--extra V6_mae_resid_0807]

Five decision rows, exactly the ones the brief names, plus the machinery they are
borrowed from so the numbers land in the same table as V5c's / V6's:

  (i)   PER-FAMILY MOTION ENERGY   n3_eval_temporal.family_energy (jacks / run /
        kick are where mean regression shows: a jack whose interior is averaged
        away still lands on the right endpoints)
  (ii)  2c WARP INVARIANCE         n3_eval_temporal.exam_2c, warp_exam.py's recipe
        verbatim (seed-13 windows, warps seeded 1000..1005, oracle = centered
        arc-length descriptor). READ THE CAVEAT PRINTED IN THE TABLE: arm 2's code
        stream is TIME-LOCAL by construction and was never asked to be invariant to
        a non-uniform warp, only to a pure frame-rate change.
  (iii) CONTENT SHARE              eval_v5.stat8_ladder's shuffled-vs-zeroed control.
        The one number that says whether the 125 bits carry THIS window's motion or
        just wake the decoder up.
  (iv)  ZERO-SHOT G1               unitree_g1_29dof / _23dof were never sampled in
        training (16 of 18 bodies trained). Reconstruction + family energy + content
        share on the held-out body, plus GT-free foot slip through the frozen
        spatial layer -> MuJoCo (n3_eval_temporal.foot_slip).
  (v)   TRANSITIONS                NEW, lloyd asked for it: splice two clips from
        DIFFERENT families (walk->kick, run->stand), hide the seam, and measure
        whether the model invents a plausible transition or mean-regresses to mush.

Every section is independently guarded; a broken stat never costs the others.
Figures go to runs/V7_figs/ and the paths are printed — lloyd judges with his eyes.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
NIGHT2_LIB = "/var/tmp/alphamotion/topo_exp/night2_lib"
if os.path.isdir(NIGHT2_LIB):
    sys.path.append(NIGHT2_LIB)

import v5_common as V                                              # noqa: E402
from common import rot6d_to_matrix, matrix_to_rot6d                # noqa: E402
from train_v5 import load_run, to_states                           # noqa: E402
from eval_v5 import tokens_of, cos_rows, stat7_leakage, stat8_ladder, R7_BODIES  # noqa: E402
import n3_eval_temporal as N3                                      # noqa: E402

RUNS = V.RUNS
FIGS = f"{RUNS}/V7_figs"
EMB = "human_smpl"
G1 = ["unitree_g1_29dof", "unitree_g1_23dof"]
# the families where mean regression is visible, plus the transition pairs
TRANSITIONS = [("walk", "kick"), ("run", "stand"), ("walk", "jacks")]
GATES = {"warp_2c": 0.90, "content_share": 0.60, "key_cm": 7.5}


# --------------------------------------------------------------- zero-shot holdout
@torch.no_grad()
def zero_shot(model, kd_zs, dev, T, span, n=256, seed=17):
    """Reconstruction + the content control on bodies the arm NEVER saw."""
    g = np.random.default_rng(seed)
    st, _ = kd_zs.sample_starts(n, g, "val", span=span)
    out = {}
    for emb in kd_zs.embs:
        w = kd_zs.window(emb, st, T, span).to(dev).float()
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(w.shape[0], device=dev)
        e = model.encode(w, cur, tgt, kn)
        p = to_states(model.decode(e["path_z"], e["time_z"], cur, tgt, kn, T),
                      model, cur, tgt, kn).float()
        pe, re = V.key_se3_err(p, w, kd_zs.reach[emb])
        p0 = model.decode(torch.zeros_like(e["path_z"]), e["time_z"],
                          cur, tgt, kn, T).float()
        pe0, _ = V.key_se3_err(p0, w, kd_zs.reach[emb])
        perm = torch.randperm(w.shape[0], device=dev)
        ps = model.decode(e["path_z"][perm], e["time_z"], cur, tgt, kn, T).float()
        pes, _ = V.key_se3_err(ps, w, kd_zs.reach[emb])
        bl = model.baseline(cur, tgt, kn, T).float()
        gain, content = pe0 - pe, pes - pe
        out[emb] = {"key_pos_cm": round(pe, 2), "key_rot_deg": round(re, 1),
                    "energy_ratio": round(float(p.std(1).mean() / w.std(1).mean()), 3),
                    "zeroed_cm": round(pe0, 2), "shuffled_cm": round(pes, 2),
                    "straight_line_cm": round(V.key_se3_err(bl, w, kd_zs.reach[emb])[0], 2),
                    "registers_buy_cm": round(gain, 3),
                    "content_share": round(content / gain, 3) if gain > 1e-6 else None}
    return out


# --------------------------------------------------------------- transitions (v)
def splice(kd, emb, famA, famB, n, T, seed=101):
    """Two clips from DIFFERENT families, spliced at T/2 with the SEAM HIDDEN.

    The second half's GLOBAL orientations and root-relative positions are rotated by
    the constant offset that carries clip B's first root frame onto clip A's last, so
    the ROOT HEADING is continuous across the seam and the model is not simply
    detecting a world-frame jump. The BODY CONFIGURATION still jumps — that is the
    transition the model has to invent, and nothing in the input marks where it is.
    """
    g = np.random.default_rng(seed)
    stA, _ = kd.sample_starts(n, g, "val", span=T, families=[famA])
    stB, _ = kd.sample_starts(n, g, "val", span=T, families=[famB])
    A = kd.window(emb, stA, T, T).double()
    B = kd.window(emb, stB, T, T).double()
    h = T // 2
    a, b = A[:, :h], B[:, h:]
    off = (rot6d_to_matrix(a[:, -1, 0, :6])
           @ rot6d_to_matrix(B[:, h, 0, :6]).transpose(-1, -2))          # [n,3,3]
    Rb = off[:, None, None] @ rot6d_to_matrix(b[..., :6])
    pb = torch.einsum("nab,ntkb->ntka", off, b[..., 6:9])
    b2 = torch.cat([matrix_to_rot6d(Rb), pb], -1)
    return torch.cat([a, b2], 1).float(), h


@torch.no_grad()
def transition_test(model, kd, dev, T, span, n=96, seed=101):
    """-> per-pair metrics. The tell for mean-regressed mush is the ENERGY of each
    half collapsing while the endpoint error stays respectable."""
    out = {}
    h = T // 2
    for famA, famB in TRANSITIONS:
        try:
            w, h = splice(kd, EMB, famA, famB, n, T, seed)
        except AssertionError:
            continue
        w = w.to(dev)
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(w.shape[0], device=dev)
        p = to_states(model(w, cur, tgt, kn)[0], model, cur, tgt, kn).float()
        bl = model.baseline(cur, tgt, kn, T).float()
        r = {}
        band = slice(max(h - 5, 0), min(h + 5, T))
        for tag, sl in (("all", slice(0, T)), ("firstHalf", slice(0, h)),
                        ("secondHalf", slice(h, T)), ("seam", band)):
            pe, _ = V.key_se3_err(p[:, sl], w[:, sl], kd.reach[EMB])
            be, _ = V.key_se3_err(bl[:, sl], w[:, sl], kd.reach[EMB])
            gt_std = float(w[:, sl].std(1).mean())
            r[tag] = {"key_pos_cm": round(pe, 2),
                      "interp_only_cm": round(be, 2),
                      "energy_ratio": round(float(p[:, sl].std(1).mean()) / gt_std, 3)}
        # arc length: a plausible transition SPENDS the motion; mush does not
        dsp, dsg = V.se3_step(p), V.se3_step(w)
        r["arclen_ratio"] = round(float(dsp.sum(-1).mean() / dsg.sum(-1).mean()), 3)
        # the artificial splice puts a spike in GT at the seam; how much of it does
        # the model reproduce, and does it smear the rest of the window instead?
        r["seam_step_ratio"] = round(
            float(dsp[:, h - 1:h + 1].mean() / dsg[:, h - 1:h + 1].mean()), 3)
        away = torch.cat([dsp[:, :h - 3], dsp[:, h + 3:]], 1)
        awg = torch.cat([dsg[:, :h - 3], dsg[:, h + 3:]], 1)
        r["offseam_step_ratio"] = round(float(away.mean() / awg.mean()), 3)
        out[f"{famA}->{famB}"] = r
    return out


@torch.no_grad()
def transition_figure(models, kd, dev, T, path, n=3, seed=101):
    """GT vs each arm across a hidden walk->kick seam: the SE3 step profile and the
    left-wrist height. Mean-regressed mush shows up as a flat ds curve."""
    plt = V.fig_style()
    w, h = splice(kd, EMB, "walk", "kick", n, T, seed)
    w = w.to(dev)
    cur, tgt = w[:, 0], w[:, -1]
    kn = torch.ones(w.shape[0], device=dev)
    preds = {}
    for tag, (m, _) in models.items():
        preds[tag] = to_states(m(w, cur, tgt, kn)[0], m, cur, tgt, kn).float().cpu()
    gt = w.float().cpu()
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 6.4), squeeze=False)
    for i in range(n):
        ds_g = V.se3_step(gt[i:i + 1])[0].numpy()
        ax[0][i].plot(ds_g, color="#111", lw=1.8, label="GT (spliced)")
        for j, (tag, p) in enumerate(preds.items()):
            ax[0][i].plot(V.se3_step(p[i:i + 1])[0].numpy(), color=V.SERIES[j],
                          lw=1.4, label=tag)
        ax[0][i].axvline(h - 0.5, color="#e34948", ls="--", lw=1.2)
        ax[0][i].set_title(f"walk->kick #{i}: SE3 step ds_t (seam hidden at {h})")
        ax[0][i].set_xlabel("frame")
        if i == 0:
            ax[0][i].set_ylabel("ds (reach units)"); ax[0][i].legend(fontsize=8)
        # left wrist height (key index 2 = l_wrist), root-relative, in reach units
        ax[1][i].plot(gt[i, :, 2, 7].numpy(), color="#111", lw=1.8)
        for j, (tag, p) in enumerate(preds.items()):
            ax[1][i].plot(p[i, :, 2, 7].numpy(), color=V.SERIES[j], lw=1.4)
        ax[1][i].axvline(h - 0.5, color="#e34948", ls="--", lw=1.2)
        ax[1][i].set_title("left wrist height"); ax[1][i].set_xlabel("frame")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def curve_figure(arms, path):
    """Training curves straight out of each run's eval_log.json, with arm 2's
    l-DeTok mask-warm-up boundary marked. If masking induces the grokking-like
    delayed phase transition the paper describes, it is visible right there."""
    plt = V.fig_style()
    keys = [("key_pos_cm", "key pos [cm] (lower better)"),
            ("energy_ratio", "energy ratio (1.0 = GT)"),
            ("content_share", "content share (>= 0.6 gate)")]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for j, (a, rd) in enumerate(arms):
        try:
            log = json.load(open(f"{rd}/eval_log.json"))
            cfg = json.load(open(f"{rd}/config.json"))
        except Exception:
            continue
        st = [e["step"] for e in log]
        for i, (k, _) in enumerate(keys):
            y = []
            for e in log:
                v = e["val"].get("human_smpl", e["val"]).get(k)
                y.append(np.nan if v is None else v)
            ax[i].plot(st, y, color=V.SERIES[j % len(V.SERIES)], lw=1.6, label=a,
                       marker="o", ms=2.6)
        mw = int(cfg.get("mask_warmup_steps") or 0)
        if cfg.get("arm") == "maskfsq" and mw:
            for i in range(3):
                ax[i].axvline(mw, color=V.SERIES[j % len(V.SERIES)], ls=":", lw=1.2)
    for i, (_, lab) in enumerate(keys):
        ax[i].set_xlabel("step"); ax[i].set_title(lab, fontsize=10)
    ax[1].axhline(1.0, color="#111", lw=0.9, ls="--")
    ax[2].axhline(0.6, color="#111", lw=0.9, ls="--")
    ax[0].legend(fontsize=8)
    fig.suptitle("V7 training curves (dotted = arm 2's l-DeTok mask warm-up ends)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def energy_figure(res, names, path):
    plt = V.fig_style()
    fams = V.V5_FAMILIES
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(fams))
    wdt = 0.8 / max(len(names), 1)
    for j, n_ in enumerate(names):
        y = []
        for f in fams:
            try:
                y.append(res["arms"][n_]["family_energy"][f]["energy_ratio"])
            except Exception:
                y.append(np.nan)
        ax.bar(x + j * wdt, y, wdt, label=n_, color=V.SERIES[j % len(V.SERIES)])
    ax.axhline(1.0, color="#111", lw=1.0, ls="--")
    ax.set_xticks(x + wdt * (len(names) - 1) / 2); ax.set_xticklabels(fams)
    ax.set_ylabel("energy ratio (pred / GT)")
    ax.set_title("per-family motion energy — 1.0 = GT, low = mean regression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807")
    ap.add_argument("--extra", default="", help="reference arms (V6/V5c) for the table")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_ladder", type=int, default=256)
    ap.add_argument("--n_knn", type=int, default=250)
    ap.add_argument("--n_trans", type=int, default=96)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--no_phys", action="store_true")
    ap.add_argument("--out", default=f"{RUNS}/V7_verdict_0807.txt")
    args = ap.parse_args()
    if args.tiny:
        args.n, args.n_ladder, args.n_knn, args.n_trans = 48, 32, 24, 16
    dev, t0 = args.device, time.time()

    arms = []
    for a in [r for r in (args.runs + "," + args.extra).split(",") if r]:
        rd = a if os.path.isabs(a) else f"{RUNS}/{a}"
        if os.path.exists(f"{rd}/model.pt"):
            arms.append((a, rd))
        else:
            print(f"[skip] {a}: no model.pt", flush=True)
    if not arms:
        raise SystemExit("no finished arm found")

    kd = V.KeyData(V.AMASS, list(dict.fromkeys([EMB] + R7_BODIES)), device="cpu")
    kd_zs = V.KeyData(V.AMASS, G1, device="cpu")
    tok = ent = None
    try:
        tok, _ = V.load_frozen(dev)
        ent = V.body_entry(V.AMASS, EMB, dev, kd.reach[EMB])
    except Exception as e:
        print(f"WARN frozen spatial layer unavailable: {e}", flush=True)

    T0 = S0 = V.T_WIN
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "arms": {}, "gates": GATES}
    try:
        base, warps, perm = N3.warp_windows(kd, args.n, T0, S0)
        dA = N3.arclen_descriptor(base)
        res["oracle_2c"] = {
            "control": round(float(cos_rows(dA, dA[perm]).mean()), 4),
            "warp": round(float(np.mean([float(cos_rows(dA, N3.arclen_descriptor(w)).mean())
                                         for w in warps])), 4)}
        print("oracle_2c", json.dumps(res["oracle_2c"]), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        res["oracle_2c"] = {"error": str(e)}

    models = {}
    for a, rd in arms:
        print(f"\n===== {a}", flush=True)
        model, cfg = load_run(rd, dev)
        models[a] = (model, cfg)
        T, span = cfg.get("T", V.T_WIN), cfg.get("span", V.T_WIN)
        out = {"arch": cfg.get("arch"), "arm": cfg.get("arm"),
               "steps": cfg.get("steps"),
               "n_path": int(getattr(model, "n_path", 0)),
               "n_time": int(getattr(model, "n_time", 0)),
               "bit_budget_per_s": (cfg.get("bit_budget") or {}).get("total_bits_per_s"),
               "n_bodies_trained": len(cfg.get("embs_train") or cfg.get("embs", "").split(",")),
               "inv_mode": cfg.get("inv_mode"),
               "mask_warmup_steps": cfg.get("mask_warmup_steps")}
        jobs = [
            ("warp_2c", lambda: N3.exam_2c(model, kd, dev, args.n, T, span)),
            ("ladder", lambda: stat8_ladder(model, kd, EMB, dev, args.n_ladder,
                                            T, span, tok, ent)),
            ("family_energy", lambda: N3.family_energy(model, kd, dev, T, span,
                                                       n=16 if args.tiny else 64)),
            ("leakage", lambda: stat7_leakage(model, kd, dev, args.n_knn, T, span)),
            ("zero_shot", lambda: zero_shot(model, kd_zs, dev, T, span,
                                            n=32 if args.tiny else 256)),
            ("transitions", lambda: transition_test(model, kd, dev, T, span,
                                                    n=args.n_trans)),
        ]
        for name, fn in jobs:
            try:
                out[name] = fn()
                print(f"  {name}: ok", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                out[name] = {"error": f"{type(e).__name__}: {e}"}
        res["arms"][a] = out

    figs = []
    for tag, fn in (("transitions", lambda: transition_figure(
                        models, kd, dev, V.T_WIN, f"{FIGS}/v7_transition.png")),
                    ("family_energy", lambda: energy_figure(
                        res, [a for a, _ in arms], f"{FIGS}/v7_family_energy.png")),
                    ("curves", lambda: curve_figure(arms, f"{FIGS}/v7_curves.png"))):
        try:
            figs.append(fn())
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"WARN figure {tag}: {e}", flush=True)
    res["figures"] = figs

    if not args.no_phys:
        try:
            res["foot_slip"] = N3.foot_slip(models, dev, tiny=args.tiny)
        except Exception as e:
            import traceback; traceback.print_exc()
            res["foot_slip"] = {"error": f"{type(e).__name__}: {e}"}

    # ----------------------------------------------------------------- the table
    names = [a for a, _ in arms]
    W = max([len(n) for n in names] + [14]) + 2
    L, A = [], None
    A = L.append
    A("=" * 110)
    A("V7 VERDICT — conditional JEPA vs per-frame masked FSQ, OAT ELIMINATED, "
      + time.strftime("%Y-%m-%d %H:%M"))
    A("=" * 110)
    A("SETUP (identical in both arms unless stated):")
    A("  * NO OAT. No prefix budgets, no ordered registers, every latent always")
    A("    present. Justification: the prefix ladder measured 0.000-0.031 cm on")
    A("    every arm ever run (V5c 0.017, N3a 0.031, V6_mae 0.000, V6_regar 0.019),")
    A("    while ordered-prefix training spent 25% of every step in a crippled")
    A("    1-register regime. The ladder rows below are an UNTRAINED diagnostic.")
    A("  * BIT BUDGET 62.69 bit/s in BOTH arms, asserted before the first step:")
    A("      jepa    = 8 path latents + 1 time register = 9 x 6x5 FSQ")
    A("      maskfsq = 9 time-local group codes         = 9 x 6x5 FSQ")
    A("    (a literal per-FRAME FSQ would be 418 bit/s = 6.7x arm 1; matching the")
    A("     budget at 60 registers forces 2.09 bits/frame, a degenerate code. 9")
    A("     contiguous groups of 7/6/7/7/6/7/7/6/7 frames is the fairest matched")
    A("     configuration: identical register count, dims and levels, so the only")
    A("     difference is WHERE the bits live.)")
    A("  * Input: 6 key joints' 6D rotations PLUS velocity + SG(5,2) angular")
    A("    acceleration residuals on every frame. NO JERK (R10: -11..-0.3 dB SNR).")
    A("  * Loss: path integral + Riemann per-segment + velocity + multires + frame.")
    A("  * Time-scale augmentation ON. 16 of 18 embodiments trained; both G1s held")
    A("    out for the zero-shot row. pal_talos uses the FIXED alias spec.")
    A("  * THE ONE LOSS ASYMMETRY, architecture-forced: the invariance pair is a")
    A("    non-uniform WARP for jepa (a global path code should survive it) and a")
    A("    pure frame-rate change for maskfsq (a time-local code stream cannot")
    A("    survive a warp that moves content between its own groups; demanding it")
    A("    would only force the codes to discard the content). Row (ii) must be")
    A("    read with that in mind — it measures an architectural property.")
    A("  * arm 1 additionally carries the JEPA latent-prediction term (--w_jepa),")
    A("    which is what makes it a JEPA rather than a conditional VQ-VAE.")
    A("=" * 110)

    def row(label, fn, fmt="{:.3f}"):
        vals = []
        for n_ in names:
            try:
                v = fn(res["arms"][n_])
                vals.append(fmt.format(v) if v is not None else "-")
            except Exception:
                vals.append("-")
        A(f"{label:44s}" + "".join(f"{v:>{W}s}" for v in vals))

    A(f"{'metric':44s}" + "".join(f"{n_:>{W}s}" for n_ in names))
    row("  bit budget [bit/s]", lambda r: r["bit_budget_per_s"], "{:.2f}")
    row("  registers (path)", lambda r: r["n_path"], "{:.0f}")
    row("  registers (time)", lambda r: r["n_time"], "{:.0f}")
    row("  bodies trained", lambda r: r["n_bodies_trained"], "{:.0f}")

    A("-- (i) PER-FAMILY MOTION ENERGY (energy ratio; 1.0 = GT, low = mean regression) --")
    for fam in V.V5_FAMILIES:
        row(f"  {fam}", lambda r, f=fam: r["family_energy"][f]["energy_ratio"])
    for fam in ("jacks", "run", "kick"):
        row(f"  {fam} key pos [cm]",
            lambda r, f=fam: r["family_energy"][f]["key_pos_cm"], "{:.2f}")

    o = res.get("oracle_2c", {})
    A(f"-- (ii) 2c WARP INVARIANCE [gate >= {GATES['warp_2c']}; oracle "
      f"{o.get('warp','?')} / control {o.get('control','?')}] --")
    row("  warp invariance (frac)", lambda r: r["warp_2c"]["warp_frac_mean"])
    row("  control (unrelated)", lambda r: r["warp_2c"]["control_unrelated"])
    # V5/V6 arms have no --inv_mode flag; they all paired against a non-uniform warp,
    # so a missing key means "warp" and must not be printed as "no".
    row("  (trained against warps?)",
        lambda r: 1.0 if (r.get("inv_mode") or "warp") == "warp" else 0.0, "{:.0f}")

    A(f"-- (iii) CONTENT SHARE [gate >= {GATES['content_share']}; key cm <= "
      f"{GATES['key_cm']}] --")
    row("  content share", lambda r: r["ladder"]["_token_content"]["content_share"])
    row("  registers buy [cm]", lambda r: r["ladder"]["_token_content"]["registers_buy_cm"])
    row("  window-specific [cm]",
        lambda r: r["ladder"]["_token_content"]["window_specific_cm"])
    row("  key pos (all latents) [cm]",
        lambda r: r["ladder"][f"prefix{r['n_path']}"]["key_pos_cm"], "{:.2f}")
    row("  key rot (all latents) [deg]",
        lambda r: r["ladder"][f"prefix{r['n_path']}"]["key_rot_deg"], "{:.1f}")
    row("  energy ratio (all latents)",
        lambda r: r["ladder"][f"prefix{r['n_path']}"]["energy_ratio"])
    row("  tokens shuffled [cm]", lambda r: r["ladder"]["tokens_shuffled"]["key_pos_cm"], "{:.2f}")
    row("  tokens zeroed [cm]", lambda r: r["ladder"]["tokens_zeroed"]["key_pos_cm"], "{:.2f}")
    row("  straight line [cm]", lambda r: r["ladder"]["straight_line"]["key_pos_cm"], "{:.2f}")
    A("  [UNTRAINED prefix-ladder diagnostic — V7 never trained a prefix regime]")
    for b in (1, 2, 4):
        row(f"  n{b} latents [cm]",
            lambda r, b=b: r["ladder"][f"prefix{b}"]["key_pos_cm"], "{:.2f}")
    row("  kNN body leakage", lambda r: r["leakage"]["knn_body_acc"])

    A("-- (iv) ZERO-SHOT, HELD-OUT G1 (never sampled in training) --")
    for g in G1:
        row(f"  {g} key pos [cm]",
            lambda r, g=g: r["zero_shot"][g]["key_pos_cm"], "{:.2f}")
        row(f"  {g} energy ratio",
            lambda r, g=g: r["zero_shot"][g]["energy_ratio"])
        row(f"  {g} content share",
            lambda r, g=g: r["zero_shot"][g]["content_share"])
        row(f"  {g} straight line [cm]",
            lambda r, g=g: r["zero_shot"][g]["straight_line_cm"], "{:.2f}")
    fs = res.get("foot_slip", {})
    if fs and "error" not in fs:
        A("   [GT-free physeval, decoded G1, median over "
          f"{','.join(N3.PHYS_CLIPS)}; GT_retarget floor slip50="
          f"{N3.med(fs.get('GT_retarget', {}), 'slip_p50_ms'):.3f}, V5c@0806 0.751]")
        for lab, k_ in (("foot slip p50 [m/s]", "slip_p50_ms"),
                        ("foot slip p95 [m/s]", "slip_p95_ms"),
                        ("torque over limit [%]", "torque_over_pct"),
                        ("jerk p95 [rad/s^3]", "jerk_p95")):
            vals = []
            for n_ in names:
                try:
                    vals.append(f"{N3.med(fs[n_], k_):.3f}")
                except Exception:
                    vals.append("-")
            A(f"  {lab:42s}" + "".join(f"{v:>{W}s}" for v in vals))

    A("-- (v) TRANSITIONS: two families spliced at frame 30, SEAM HIDDEN --")
    A("   [mush = both halves' energy collapses while the endpoints stay fine]")
    for pair in [f"{a}->{b}" for a, b in TRANSITIONS]:
        A(f"  {pair}")
        row("    energy 1st half", lambda r, p=pair: r["transitions"][p]["firstHalf"]["energy_ratio"])
        row("    energy 2nd half", lambda r, p=pair: r["transitions"][p]["secondHalf"]["energy_ratio"])
        row("    key pos all [cm]", lambda r, p=pair: r["transitions"][p]["all"]["key_pos_cm"], "{:.2f}")
        row("    interp-only [cm]", lambda r, p=pair: r["transitions"][p]["all"]["interp_only_cm"], "{:.2f}")
        row("    key pos seam [cm]", lambda r, p=pair: r["transitions"][p]["seam"]["key_pos_cm"], "{:.2f}")
        row("    arclen ratio", lambda r, p=pair: r["transitions"][p]["arclen_ratio"])
        row("    off-seam step ratio", lambda r, p=pair: r["transitions"][p]["offseam_step_ratio"])

    # ------------------------------------------------------- the plain verdict
    A("")
    A("=" * 110)
    A("PLAIN VERDICT — V7_jepa vs V7_maskfsq, same 62.69 bit/s, same data, same losses")
    A("=" * 110)
    primary = [r for r in args.runs.split(",") if r]
    duel = [n_ for n_ in names if n_ in primary][:2]
    if len(duel) == 2:
        a1, a2 = duel
        r1, r2 = res["arms"][a1], res["arms"][a2]

        def cmp(label, get, higher_better, fmt="{:.3f}", tol=0.0):
            try:
                v1, v2 = get(r1), get(r2)
                if v1 is None or v2 is None:
                    raise ValueError
                v1, v2 = float(v1), float(v2)
            except Exception:
                A(f"  {label:26s} (not measurable)")
                return None
            if abs(v1 - v2) <= tol:
                win = "TIE"
            else:
                win = a1 if ((v1 > v2) == higher_better) else a2
            A(f"  {label:26s} {a1} {fmt.format(v1)}   {a2} {fmt.format(v2)}"
              f"   -> {win}")
            return win

        def fam_mean(r, fams):
            v = [r["family_energy"][f]["energy_ratio"] for f in fams]
            return float(np.mean(v))

        wins = {}
        A("(i) PER-FAMILY MOTION ENERGY (jacks/run/kick — where mean regression shows)")
        wins["energy"] = cmp("mean energy jacks/run/kick",
                             lambda r: fam_mean(r, ("jacks", "run", "kick")),
                             True, tol=0.01)
        cmp("mean energy all 6 families", lambda r: fam_mean(r, V.V5_FAMILIES),
            True, tol=0.01)
        A("(ii) 2c WARP INVARIANCE  [see the caveat: only jepa was TRAINED against warps]")
        wins["invariance"] = cmp("warp invariance",
                                 lambda r: r["warp_2c"]["warp_frac_mean"], True,
                                 tol=0.005)
        A("(iii) CONTENT SHARE")
        wins["content"] = cmp("content share",
                              lambda r: r["ladder"]["_token_content"]["content_share"],
                              True, tol=0.02)
        cmp("key pos (all latents) [cm]",
            lambda r: r["ladder"][f"prefix{r['n_path']}"]["key_pos_cm"], False,
            "{:.2f}", tol=0.05)
        A("(iv) ZERO-SHOT, HELD-OUT G1")
        wins["zeroshot"] = cmp("g1_29dof key pos [cm]",
                               lambda r: r["zero_shot"][G1[0]]["key_pos_cm"], False,
                               "{:.2f}", tol=0.05)
        cmp("g1_29dof energy ratio",
            lambda r: r["zero_shot"][G1[0]]["energy_ratio"], True, tol=0.01)
        try:
            s1 = N3.med(res["foot_slip"][a1], "slip_p50_ms")
            s2 = N3.med(res["foot_slip"][a2], "slip_p50_ms")
            w = "TIE" if abs(s1 - s2) <= 0.01 else (a1 if s1 < s2 else a2)
            A(f"  {'g1 foot slip p50 [m/s]':26s} {a1} {s1:.3f}   {a2} {s2:.3f}"
              f"   -> {w}")
        except Exception:
            A(f"  {'g1 foot slip p50 [m/s]':26s} (not measurable)")
        A("(v) TRANSITIONS (two families spliced, seam hidden)")

        def trans_energy(r):
            v = []
            for p in [f"{x}->{y}" for x, y in TRANSITIONS]:
                t = r["transitions"].get(p)
                if t:
                    v += [t["firstHalf"]["energy_ratio"], t["secondHalf"]["energy_ratio"]]
            return float(np.mean(v))
        wins["transitions"] = cmp("mean half-window energy", trans_energy, True,
                                  tol=0.01)
        cmp("mean arclen ratio (1.0 = GT)",
            lambda r: float(np.mean([r["transitions"][f"{x}->{y}"]["arclen_ratio"]
                                     for x, y in TRANSITIONS])), True, tol=0.01)
        cmp("mean key pos over splice [cm]",
            lambda r: float(np.mean([r["transitions"][f"{x}->{y}"]["all"]["key_pos_cm"]
                                     for x, y in TRANSITIONS])), False, "{:.2f}", tol=0.05)
        tally = {a1: 0, a2: 0}
        n_tie = sum(1 for w in wins.values() if w == "TIE")
        for w in wins.values():
            if w in tally:
                tally[w] += 1
        A("")
        A("  ROW WINNERS: " + ", ".join(f"{k_}={v}" for k_, v in wins.items()))
        A(f"  TALLY: {a1} {tally[a1]}/5, {a2} {tally[a2]}/5, ties {n_tie}/5")
        if tally[a1] == tally[a2]:
            best = "UNDECIDED (split) — read rows (i), (iii) and (v) directly"
        else:
            best = a1 if tally[a1] > tally[a2] else a2
        A(f"  OVERALL: {best}")
        A("  Caveat that must travel with row (ii): maskfsq's code stream is")
        A("  time-local and was trained for frame-rate invariance, not warp")
        A("  invariance, because a warp moves content between its own groups.")
        A("  If jepa wins (ii) and loses everything else, (ii) is the price of the")
        A("  factorization, not evidence that jepa is the better tokenizer.")
    A("=" * 110)
    A("")
    A("GATES (2c >= 0.90, content >= 0.6, key <= 7.5 cm):")
    for n_ in names:
        r = res["arms"][n_]
        try:
            w2 = r["warp_2c"]["warp_frac_mean"]
            cs = r["ladder"]["_token_content"]["content_share"]
            kc = r["ladder"][f"prefix{r['n_path']}"]["key_pos_cm"]
            A(f"  {n_:18s} 2c {w2:.3f} {'PASS' if w2 >= GATES['warp_2c'] else 'FAIL'} | "
              f"content {(cs if cs is not None else float('nan')):.3f} "
              f"{'PASS' if (cs or 0) >= GATES['content_share'] else 'FAIL'} | "
              f"key {kc:.2f} {'PASS' if kc <= GATES['key_cm'] else 'FAIL'}")
        except Exception:
            A(f"  {n_:18s} (incomplete)")
    if figs:
        A("")
        A("FIGURES (look at these before believing any number above):")
        for f in figs:
            A(f"  {f}")
    A("=" * 110)
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(txt + "\n")
    json.dump(res, open(args.out.removesuffix(".txt") + ".json", "w"), indent=1,
              default=float)
    print(f"\nSAVED {args.out} (+.json)  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
