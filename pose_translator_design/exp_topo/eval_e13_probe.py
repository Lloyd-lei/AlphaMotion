"""E13 probe verdict — the PRE-REGISTERED criteria, implemented honestly.

Appendix ("备用架:E13 时空分解码本"):
    预注册判据:①同质量bits/块显著更低 ②rollout不劣 ③换动力学token干净变速
    首探针:现成码流上16帧窗小AE,瓶颈=慢token+相位,慢token聚类 vs 动作标签
    —— **聚类混沌则全线止损**

This script answers four things and prints a verdict:

  A. DO SLOW TOKENS CLUSTER BY ACTION?  (the stop rule)
     k-means on the pre-quantization slow vector + purity/NMI vs regex action families,
     a label-shuffle null, the vocabulary statistics of the *discrete* slow token, and a
     clip-split linear probe. Every number is reported next to (i) the majority-class
     baseline and (ii) the same probe run on the raw 16-frame mean code, which is the
     information that WAS available -- so "the slow token clusters" cannot be confused
     with "the raw codes cluster and we merely didn't destroy it".

  B. BITS ACCOUNTING per 16-frame window: the factorized code vs the raw frame codes,
     reported together with reconstruction quality so the compression claim is always
     read at a stated quality, never on its own.

  C. DOES THE PHASE BEHAVE LIKE A PHASE? monotone advance + wrapping on periodic clips
     (walk/run), compared against near-static clips, and against the dominant frequency
     independently estimated from the raw code stream. Plot of phase vs time.

  D. WHAT THE FACTORIZATION COSTS: decoded geodesic/position error through the FROZEN
     tokenizer decoder, vs the same decoder on the original codes (the floor) and vs a
     constant-pose window (the trivial baseline that the R7 free-decode collapsed to),
     plus per-joint temporal std -- the direct test of "is there any temporal prior".

A/B/C run on CPU. D needs the GPU (frozen tokenizer decode) and the source cache;
--skip_decode drops it.

Usage:
  python eval_e13_probe.py --run .../runs/E13_factor_0805 \
      --compare .../runs/E13_frame_0805 .../runs/E13_slow_only_0805
"""
from __future__ import annotations
import os, sys, json, math, re, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize_stream import load_stream                                    # noqa: E402
from train_e13_probe import E13Probe, WindowStream, parse_levels, grid_ints  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

# ---- plot style, copied from the R7 analysis so E13 figures sit next to fig3/fig4
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "font.family": "sans-serif"})

# ---- action families from AMASS clip names. VERBATIM from the R7 analysis
# (r7_common.py, the parser behind ANALYSIS.md fig3) so E13's numbers are directly
# comparable to R7's "purity 82% / probe 81.6%" frame-level result.
FAMILIES = ["dance", "jump", "run", "walk", "sit", "reach"]
_PATS = {
    "dance": r"danc|salsa|cha_?cha|tango|ballet|jazz|samba|swing_danc",
    "jump": r"jump|hop|leap|skip",
    "run": r"run|jog|sprint",
    "walk": r"walk|stroll|march|stride",
    "sit": r"sit|chair|stool",
    "reach": r"reach|pick|grasp|grab|place",
}
PERIODIC = ("walk", "run")
STATIC_PAT = r"stand|a1_-_stand|静|idle|t_?pose|rest"


def action_family(name):
    low = name.lower()
    for f in FAMILIES:
        if re.search(_PATS[f], low):
            return f
    return "other"


# --------------------------------------------------------------------- probe-run loading
def load_probe(run, device):
    cfg = json.load(open(f"{run}/config.json"))
    model = E13Probe(cfg["D"], T=cfg["T"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                     n_enc=cfg["n_enc"], n_dec=cfg["n_dec"], mode=cfg["mode"],
                     slow_levels=parse_levels(cfg["slow_levels"]), phase_bins=cfg["phase_bins"],
                     phase_harm=cfg["phase_harm"], res_dims=cfg["res_dims"],
                     res_levels=parse_levels(cfg["res_levels"]), frame_dims=cfg["frame_dims"],
                     frame_levels=parse_levels(cfg["frame_levels"])).to(device)
    model.load_state_dict(torch.load(f"{run}/model.pt", map_location=device))
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_windows(model, ws, starts, device, bs=256, want=("slow_pre", "slow_q", "theta")):
    """Run the probe encoder over given window starts -> stacked numpy parts."""
    out = {k: [] for k in want}
    for i in range(0, len(starts), bs):
        x = ws.deq(ws.window_codes(starts[i:i + bs]), device)
        p = model.encode(x)
        for k in want:
            if k in p:
                out[k].append(p[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def clip_windows(ws, clips, stride):
    """-> (window starts, clip index per window) for the given clip ids."""
    b = ws.s["bounds"]
    st, cid = [], []
    for c in clips:
        s = np.arange(b[c], b[c + 1] - ws.T + 1, stride)
        st.append(s); cid.append(np.full(len(s), c, np.int64))
    return np.concatenate(st), np.concatenate(cid)


# --------------------------------------------------------------------- A. slow-token semantics
def purity(labels, fam, mask):
    tot = hit = 0
    for k in np.unique(labels[mask]):
        m = mask & (labels == k)
        if m.sum() == 0:
            continue
        _, cts = np.unique(fam[m], return_counts=True)
        hit += cts.max(); tot += m.sum()
    return hit / max(tot, 1)


def linear_probe(Xtr, ytr, Xte, yte, device, epochs=60, lr=1e-2, wd=1e-4, bs=8192):
    """Same one-layer probe as R7 a3_semantics.py (standardize -> logistic regression)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    C = int(max(ytr.max(), yte.max())) + 1
    W = torch.zeros(Xtr.shape[1], C, device=device, requires_grad=True)
    b = torch.zeros(C, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    Xt = torch.from_numpy((Xtr - mu) / sd).float().to(device)
    yt = torch.from_numpy(ytr).long().to(device)
    Xv = torch.from_numpy((Xte - mu) / sd).float().to(device)
    yv = torch.from_numpy(yte).long().to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(Xt[idx] @ W + b, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (Xv @ W + b).argmax(1).cpu().numpy()
        acc = float((pred == yte).mean())
        tr = float(((Xt @ W + b).argmax(1) == yt).float().mean())
    conf = np.zeros((C, C))
    for t, p in zip(yte, pred):
        conf[t, p] += 1
    return acc, tr, conf / conf.sum(1, keepdims=True).clip(min=1)


def analyze_slow(model, ws, cfg, device, args, fam_of_clip, fig_dir):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    res = {}
    if not model.use_slow:
        return {"skipped": "mode has no slow token"}

    tr_st, tr_c = clip_windows(ws, ws.train_clips, args.stride)
    va_st, va_c = clip_windows(ws, ws.val_clips, args.stride)
    if args.max_windows and len(tr_st) > args.max_windows:
        sel = np.linspace(0, len(tr_st) - 1, args.max_windows).astype(np.int64)
        tr_st, tr_c = tr_st[sel], tr_c[sel]
    if args.max_windows and len(va_st) > args.max_windows:
        sel = np.linspace(0, len(va_st) - 1, args.max_windows).astype(np.int64)
        va_st, va_c = va_st[sel], va_c[sel]
    tr = encode_windows(model, ws, tr_st, device, want=("slow_pre", "slow_q"))
    va = encode_windows(model, ws, va_st, device, want=("slow_pre", "slow_q"))
    fam_tr, fam_va = fam_of_clip[tr_c], fam_of_clip[va_c]
    res["n_windows"] = {"train": len(tr_st), "val": len(va_st)}

    # --- A1. is the discrete slow token a reusable VOCABULARY, or singletons again?
    ints = model.slow_fsq.ints(torch.from_numpy(va["slow_pre"]).to(device)).cpu().numpy()
    keys = np.ascontiguousarray(ints.astype(np.int16)).view(
        np.dtype((np.void, ints.shape[1] * 2))).ravel()
    uniq, inv, cts = np.unique(keys, return_inverse=True, return_counts=True)
    p = cts / cts.sum()
    res["vocabulary"] = {
        "n_possible_words": int(model.slow_fsq.n_words),
        "n_used": int(len(uniq)), "n_windows": int(len(keys)),
        "singleton_frac": float((cts == 1).mean()),
        "top10_coverage": float(np.sort(p)[::-1][:10].sum()),
        "top100_coverage": float(np.sort(p)[::-1][:100].sum()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "entropy_bits_max": float(model.slow_fsq.bits_exact),
        "R7_frame_code_singleton_frac": [0.65, 0.59],   # pos/rot streams, ANALYSIS.md fig4
    }
    # NMI of the raw discrete token against families: purity is meaningless when most
    # tokens are singletons (it would be ~1.0 by construction), NMI is not.
    res["discrete_token_nmi"] = float(normalized_mutual_info_score(fam_va, inv))
    common = np.isin(inv, np.where(cts >= args.min_token_count)[0])
    res["discrete_token_purity_common"] = float(purity(inv, fam_va, common))
    res["discrete_token_common_frac"] = float(common.mean())

    # --- A2. k-means on the continuous slow vector (R7 a3 protocol, k=32)
    P = va["slow_pre"] / (np.linalg.norm(va["slow_pre"], axis=1, keepdims=True) + 1e-9)
    K = args.kmeans_k
    lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(P)
    named = fam_va != "other"
    maj_named = max((fam_va[named] == f).mean() for f in FAMILIES) if named.any() else float("nan")
    res["kmeans"] = {
        "k": K,
        "purity_all": float(purity(lab, fam_va, np.ones(len(lab), bool))),
        "purity_named": float(purity(lab, fam_va, named)),
        "majority_named": float(maj_named),
        "nmi": float(normalized_mutual_info_score(fam_va, lab)),
        "ari": float(adjusted_rand_score(fam_va, lab)),
        "named_frac": float(named.mean()),
        "R7_frame_reference": {"purity_named": 0.82, "majority_named": 0.49},
    }
    # label-shuffle null: shuffle the family label BY CLIP (shuffling by window would
    # leak, since windows of one clip share a label)
    rng = np.random.default_rng(0)
    nulls_p, nulls_n = [], []
    clips_u = np.unique(va_c)
    for _ in range(args.null_reps):
        perm_fam = fam_of_clip[clips_u].copy(); rng.shuffle(perm_fam)
        m = dict(zip(clips_u.tolist(), perm_fam.tolist()))
        f2 = np.array([m[c] for c in va_c])
        nulls_p.append(purity(lab, f2, f2 != "other"))
        nulls_n.append(normalized_mutual_info_score(f2, lab))
    res["kmeans"]["null_purity_named_mean"] = float(np.mean(nulls_p))
    res["kmeans"]["null_purity_named_p95"] = float(np.percentile(nulls_p, 95))
    res["kmeans"]["null_nmi_mean"] = float(np.mean(nulls_n))
    res["kmeans"]["null_nmi_p95"] = float(np.percentile(nulls_n, 95))

    # --- A3. linear probe, clip-split, named families only
    def probe_on(Xtr, Xte):
        mtr, mte = fam_tr != "other", fam_va != "other"
        if mtr.sum() < 50 or mte.sum() < 50:
            return None
        ytr = np.array([FAMILIES.index(f) for f in fam_tr[mtr]])
        yte = np.array([FAMILIES.index(f) for f in fam_va[mte]])
        acc, tr_acc, conf = linear_probe(Xtr[mtr], ytr, Xte[mte], yte, device)
        maj = float(np.bincount(yte, minlength=len(FAMILIES)).max() / len(yte))
        return {"acc": acc, "train_acc": tr_acc, "majority": maj, "conf": conf,
                "n_train": int(mtr.sum()), "n_test": int(mte.sum())}

    pr_slow = probe_on(tr["slow_pre"], va["slow_pre"])

    # reference ceiling: the SAME probe on the raw 16-frame mean code. This is the
    # semantics that WAS available in the window; the slow token can only lose it.
    def window_mean_codes(starts):
        return np.concatenate([ws.deq(ws.window_codes(starts[i:i + 512]), "cpu").mean(1).numpy()
                               for i in range(0, len(starts), 512)])
    pr_raw = probe_on(window_mean_codes(tr_st), window_mean_codes(va_st))
    res["probe_slow"] = {k: v for k, v in (pr_slow or {}).items() if k != "conf"}
    res["probe_raw_window_mean"] = {k: v for k, v in (pr_raw or {}).items() if k != "conf"}
    res["probe_slow"]["R7_frame_reference"] = {"acc": 0.816, "majority": 0.389}

    # --- A4. the STOP RULE.
    # The appendix says "聚类混沌则全线止损" without a number. Thresholds fixed here
    # BEFORE looking at any trained model, and recorded in the output so they cannot be
    # moved afterwards:
    #   PASS needs ALL of
    #     (1) k-means purity on named clips >= majority baseline + 0.05
    #     (2) k-means purity > the 95th percentile of the clip-shuffled null
    #     (3) NMI >= 0.05 and NMI > null p95
    #     (4) linear-probe accuracy >= majority + 0.05
    #   WEAK  = (1)-(3) hold but the probe does not clear +0.05, or the slow token keeps
    #           less than half the raw window-mean probe's margin over majority.
    #   CHAOTIC (stop the whole E13 line) = anything less.
    km = res["kmeans"]
    c1 = km["purity_named"] >= km["majority_named"] + 0.05
    c2 = km["purity_named"] > km["null_purity_named_p95"]
    c3 = km["nmi"] >= 0.05 and km["nmi"] > km["null_nmi_p95"]
    c4 = bool(pr_slow) and pr_slow["acc"] >= pr_slow["majority"] + 0.05
    keep = None
    if pr_slow and pr_raw:
        mr = pr_raw["acc"] - pr_raw["majority"]
        # if the raw window-mean ceiling itself does not beat majority, the ratio is
        # meaningless (usually too few named val clips) -> report None, not a fake number
        keep = (pr_slow["acc"] - pr_slow["majority"]) / mr if mr > 0.02 else None
    res["stop_rule"] = {
        "thresholds": {"purity_margin": 0.05, "nmi_min": 0.05, "probe_margin": 0.05,
                       "weak_if_semantics_kept_below": 0.5},
        "checks": {"purity_beats_majority": bool(c1), "purity_beats_shuffle_null": bool(c2),
                   "nmi_ok": bool(c3), "probe_beats_majority": bool(c4)},
        "semantics_kept_vs_raw_window": keep,
        "verdict": ("PASS" if (c1 and c2 and c3 and c4 and (keep is None or keep >= 0.5))
                    else "WEAK" if (c1 and c2 and c3) else "CHAOTIC -> STOP E13"),
    }

    # --- figure (mirrors R7 fig3 so the two are read side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    sizes = np.bincount(lab, minlength=K)
    top = np.argsort(-sizes)[:10]
    fams = FAMILIES + ["other"]
    bot = np.zeros(len(top))
    for fi, f in enumerate(fams):
        v = np.array([np.sum(fam_va[lab == k] == f) for k in top], float)
        col = MUTED if f == "other" else SERIES[fi % 8]
        ax.bar(np.arange(len(top)), v, bottom=bot, color=col, width=0.72, label=f, linewidth=0)
        bot += v
    ax.set_xticks(np.arange(len(top))); ax.set_xticklabels([f"c{k}" for k in top], fontsize=8.5)
    ax.set_xlabel(f"10 biggest k-means clusters (k={K}) on the SLOW token")
    ax.set_ylabel("windows")
    ax.set_title(f"Slow token vs action family — purity {km['purity_named']:.0%} "
                 f"(majority {km['majority_named']:.0%}, shuffled {km['null_purity_named_mean']:.0%})")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[1]
    if pr_slow:
        conf = pr_slow["conf"]
        ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(FAMILIES))); ax.set_xticklabels(FAMILIES, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(FAMILIES))); ax.set_yticklabels(FAMILIES, fontsize=8.5)
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if conf[i, j] > 0.5 else INK)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
        ax.set_title(f"Linear probe slow→family (clip-split)\nacc {pr_slow['acc']:.0%} vs "
                     f"majority {pr_slow['majority']:.0%}"
                     + (f", raw-window ceiling {pr_raw['acc']:.0%}" if pr_raw else ""))
    p = f"{fig_dir}/fig_e13_slow.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    res["figure"] = p
    return res


# --------------------------------------------------------------------- B. bits accounting
def stream_entropy(ws, clips, max_frames=120_000, chunk=8192):
    """Per-dim independent H(code_t) and H(code_t|code_{t-1}) in bits/frame -- the same
    estimator R7 fig4 used. It ignores cross-dim correlation, so both are upper bounds
    and the conditional SAVING is the honest headline (R7: 77.7% at 30 fps).

    Counts are accumulated with one vectorized bincount per frame-chunk over
    D*L*L bins (L = grid values reachable by the FSQ, = nominal levels + 1 for even
    level counts because tanh can saturate; a spare bin is harmless, empty bins add 0)."""
    b = ws.s["bounds"]
    segs, tot = [], 0
    for c in clips:
        segs.append((int(b[c]), int(b[c + 1]))); tot += int(b[c + 1] - b[c])
        if tot >= max_frames:
            break
    D = ws.D
    L = int(max(ws.meta["levels"])) + 2
    joint = np.zeros(D * L * L, np.int64)
    marg = np.zeros(D * L, np.int64)
    dmarg = np.arange(D, dtype=np.int64) * L
    djoint = np.arange(D, dtype=np.int64) * L * L
    n_pairs = 0
    for s, e in segs:
        for i in range(s, e, chunk):                            # marginal
            u = np.asarray(ws.codes[i:min(i + chunk, e)], np.int64)
            marg += np.bincount((u + dmarg).ravel(), minlength=D * L)
        for i in range(s, e - 1, chunk):                        # (t, t+1) pairs
            j = min(i + chunk, e - 1)
            a = np.asarray(ws.codes[i:j], np.int64)
            c2 = np.asarray(ws.codes[i + 1:j + 1], np.int64)
            joint += np.bincount((a * L + c2 + djoint).ravel(), minlength=D * L * L)
            n_pairs += j - i

    def ent(counts, axis=-1):
        tot_ = counts.sum(axis, keepdims=True).clip(min=1)
        p = counts / tot_
        return -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0).sum(axis)
    H1 = float(ent(marg.reshape(D, L)).sum())
    j2 = joint.reshape(D, L, L)
    Hcond = float((ent(j2.reshape(D, -1)) - ent(j2.sum(2))).sum())
    return {"H_frame_bits": H1, "H_cond_bits": Hcond,
            "saving": float(1 - Hcond / max(H1, 1e-9)), "n_frame_pairs": int(n_pairs),
            "n_frames": int(marg.sum() // D),
            "R7_reference": {"H_frame_bits": 7662, "H_cond_bits": 1705, "saving": 0.777}}


# --------------------------------------------------------------------- C. phase
def analyze_phase(model, ws, device, args, fam_of_clip, clip_names, fig_dir):
    if not model.use_phase:
        return {"skipped": "mode has no phase"}
    b = ws.s["bounds"]
    per_fam, traces = {}, []
    rng = np.random.default_rng(0)
    clips = list(ws.val_clips)
    rows = []
    for c in clips:
        n = b[c + 1] - b[c]
        if n < 4 * ws.T:
            continue
        st = np.arange(b[c], b[c + 1] - ws.T + 1)
        if len(st) > args.phase_max_windows:
            st = st[np.linspace(0, len(st) - 1, args.phase_max_windows).astype(np.int64)]
        enc = encode_windows(model, ws, st, device, want=("theta", "theta_raw"))
        th, th_raw = enc["theta"], enc["theta_raw"]                             # [W,T]
        # (i) WITHIN a window: does phase advance smoothly and monotonically?
        d = np.angle(np.exp(1j * np.diff(th, axis=1)))
        mono = float((np.sign(d).mean(1) ** 2).mean())        # 1 = every window monotone
        frac_pos = float((d > 0).mean())
        unw = np.cumsum(np.concatenate([np.zeros((len(th), 1)), d], 1), 1)
        t = np.arange(ws.T) - (ws.T - 1) / 2
        slope = (unw * t).sum(1) / (t ** 2).sum()             # rad/frame per window
        pred = slope[:, None] * t + unw.mean(1, keepdims=True)
        ss = ((unw - pred) ** 2).sum(1)
        r2 = float(np.mean(1 - ss / np.maximum(((unw - unw.mean(1, keepdims=True)) ** 2).sum(1), 1e-9)))
        hz = float(np.median(np.abs(slope)) * ws.meta["fps"] / (2 * np.pi))
        # independent cycle-rate estimate from the RAW code stream (PC1 of q over the clip)
        q = ws.deq(np.asarray(ws.codes[b[c]:b[c + 1]])[None], "cpu")[0].numpy()
        q = q - q.mean(0)
        sub = q[:, ::17]                                      # cheap projection, 196 dims
        u, s, vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
        sig = u[:, 0] * s[0]
        sp = np.abs(np.fft.rfft(sig - sig.mean()))
        fr = np.fft.rfftfreq(len(sig), d=1.0 / ws.meta["fps"])
        ok = (fr > 0.3) & (fr < 4.0)                          # plausible gait band
        hz_raw = float(fr[ok][np.argmax(sp[ok])]) if ok.any() else float("nan")
        fam = fam_of_clip[c]
        rec = {"clip": clip_names[c], "family": fam, "n_windows": len(st),
               "monotone_score": mono, "frac_positive_step": frac_pos,
               "mean_abs_step_rad": float(np.abs(d).mean()),
               "raw_mean_abs_step_rad": float(np.abs(np.angle(np.exp(1j * np.diff(th_raw, axis=1)))).mean()),
               "raw_dispersion": float(1 - np.hypot(np.cos(th_raw).mean(1), np.sin(th_raw).mean(1)).mean()),
               "sub_bin": bool(np.abs(d).mean() < 1e-6 < np.abs(np.diff(th_raw, axis=1)).mean()),
               "linear_r2": r2,
               "phase_hz": hz, "raw_dominant_hz": hz_raw,
               "hz_ratio": float(hz / hz_raw) if hz_raw and not math.isnan(hz_raw) else None}
        rows.append(rec)
        per_fam.setdefault(fam, []).append(rec)
        if re.search(STATIC_PAT, clip_names[c].lower()):
            per_fam.setdefault("static(name)", []).append(rec)

    def agg(rs):
        return {k: float(np.nanmean([r[k] for r in rs])) for k in
                ("monotone_score", "frac_positive_step", "mean_abs_step_rad",
                 "raw_mean_abs_step_rad", "raw_dispersion", "linear_r2", "phase_hz")} \
            | {"n_clips": len(rs),
               "hz_ratio_median": float(np.nanmedian([r["hz_ratio"] for r in rs
                                                      if r["hz_ratio"] is not None])) if rs else None}
    out = {"per_family": {f: agg(rs) for f, rs in per_fam.items() if rs},
           "all": agg(rows) if rows else None, "n_clips": len(rows)}
    out["periodic_vs_other"] = {
        "periodic_monotone": float(np.nanmean([r["monotone_score"] for r in rows
                                               if r["family"] in PERIODIC])) if any(
            r["family"] in PERIODIC for r in rows) else None,
        "other_monotone": float(np.nanmean([r["monotone_score"] for r in rows
                                            if r["family"] not in PERIODIC])) if rows else None}

    # --- figure: phase vs time for a few clips, against the raw stream's PC1
    pick = []
    for f in ("walk", "run", "other"):
        cs = [r for r in rows if r["family"] == f][:2]
        pick += cs
    pick = pick[:6]
    if pick:
        name2c = {clip_names[c]: c for c in clips}
        fig, axes = plt.subplots(len(pick), 1, figsize=(9, 1.7 * len(pick)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, r in zip(axes, pick):
            c = name2c[r["clip"]]
            st = np.arange(b[c], b[c + 1] - ws.T + 1)
            th = encode_windows(model, ws, st, device, want=("theta",))["theta"][:, ws.T // 2]
            tt = np.arange(len(th)) / ws.meta["fps"]
            ax.plot(tt, th, lw=1.2, color=SERIES[0])
            ax.set_ylabel("phase θ", fontsize=8)
            ax.set_ylim(-math.pi - 0.2, math.pi + 0.2)
            ax.set_title(f"{r['clip'].split('__')[-1][:56]}  [{r['family']}]  "
                         f"θ {r['phase_hz']:.2f} Hz vs raw {r['raw_dominant_hz']:.2f} Hz, "
                         f"monotone {r['monotone_score']:.2f}", fontsize=8.5, loc="left")
            ax.tick_params(labelsize=7.5)
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        p = f"{fig_dir}/fig_e13_phase.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        out["figure"] = p
    out["per_clip"] = rows[:60]
    return out


# --------------------------------------------------------------------- D. decoded quality
def unpack_ints(codes, meta, device):
    """[b, D] uint8 -> (ip, ir) each [S,b,K,f] long, as decode_from_ints wants."""
    S, K, f = meta["stages"], meta["K"], meta["f"]
    # codes may already be a CUDA tensor; np.asarray() on one raises, so branch.
    x = (codes.to(device=device, dtype=torch.long) if torch.is_tensor(codes)
         else torch.as_tensor(np.asarray(codes), dtype=torch.long, device=device))
    h = x.shape[-1] // 2
    ip = x[..., :h].reshape(-1, S, K, f).permute(1, 0, 2, 3)
    ir = x[..., h:].reshape(-1, S, K, f).permute(1, 0, 2, 3)
    return ip, ir


@torch.no_grad()
def analyze_decode(model, ws, cfg, args, device):
    """Push original codes, probe reconstructions and a constant-pose baseline through
    the FROZEN tokenizer decoder; report geodesic/pos error vs ground truth and the
    per-joint temporal std that R7's free decode found to be exactly 0.000 cm."""
    from common import rot6d_to_matrix, geodesic_deg
    from tokenize_stream import load_tokenizer, load_body, geom_for
    meta = ws.meta
    tok, _ = load_tokenizer(meta["run"], device)

    st, _ = clip_windows(ws, ws.val_clips, ws.T)               # non-overlapping windows
    sel = st[np.linspace(0, len(st) - 1, min(args.n_decode_windows, len(st))).astype(np.int64)]
    idx = (sel[:, None] + np.arange(ws.T)[None, :])            # [W,T] packed frame index
    src_rows = ws.s["rows"][idx.reshape(-1)]
    entry, pose = load_body(meta["cache"], meta["emb"], device, rows=src_rows, reach=meta["reach"])
    reach = meta["reach"]
    W, T = idx.shape

    codes_orig = np.asarray(ws.codes[idx.reshape(-1)])         # [W*T, D]
    x = ws.deq(codes_orig.reshape(W, T, ws.D), device)
    parts = model.encode(x)
    variants = {"probe": parts}
    if model.use_phase:                                        # ablation: freeze the phase
        p2 = dict(parts); th = parts["theta"]
        p2["theta"] = torch.atan2(torch.sin(th).mean(1, keepdim=True),
                                  torch.cos(th).mean(1, keepdim=True)).expand_as(th)
        variants["probe_phase_frozen"] = p2
    if model.use_slow:                                         # ablation: mean slow token
        p3 = dict(parts); p3["slow_q"] = parts["slow_q"].mean(0, keepdim=True).expand_as(parts["slow_q"])
        variants["probe_slow_meaned"] = p3

    off = ws.off.to(device); half = ws.half.to(device)
    code_sets = {"tokenizer_original": torch.as_tensor(codes_orig, dtype=torch.long, device=device),
                 "const_first_frame": torch.as_tensor(
                     np.repeat(codes_orig.reshape(W, T, ws.D)[:, :1], T, 1).reshape(W * T, ws.D),
                     dtype=torch.long, device=device)}
    for k, p in variants.items():
        pred = model.decode(p, T)
        code_sets[k] = grid_ints(pred, off, half).reshape(W * T, ws.D)

    gt = pose.to(device)
    gtR = rot6d_to_matrix(gt[..., :6]); gtP = gt[..., 6:9] * reach
    out = {"n_windows": int(W), "n_frames": int(W * T), "reach_cm": reach,
           "gt_temporal_std_cm": float(gtP.reshape(W, T, -1, 3).std(1).norm(dim=-1).mean())}
    for name, cs in code_sets.items():
        rot_l, pos_l, tstd_l = [], [], []
        for i in range(0, W * T, args.decode_bs):
            ip, ir = unpack_ints(cs[i:i + args.decode_bs], meta, device)
            pr, pp = tok.decode_from_ints(ip, ir, geom_for(entry, ip.shape[1]))
            rot_l.append(geodesic_deg(rot6d_to_matrix(pr), gtR[i:i + args.decode_bs]).mean(1))
            pos_l.append(((pp - gt[i:i + args.decode_bs, :, 6:9]).norm(dim=-1) * reach).mean(1))
            tstd_l.append(pp * reach)
        P = torch.cat(tstd_l).reshape(W, T, -1, 3)
        out[name] = {"geodesic_deg": round(float(torch.cat(rot_l).mean()), 2),
                     "pos_cm": round(float(torch.cat(pos_l).mean()), 2),
                     "temporal_std_cm": round(float(P.std(1).norm(dim=-1).mean()), 4)}
    return out


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an E13 probe run directory")
    ap.add_argument("--compare", nargs="*", default=[], help="other runs for the bits table")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=8, help="window stride for the slow analysis")
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--kmeans_k", type=int, default=32)
    ap.add_argument("--null_reps", type=int, default=20)
    ap.add_argument("--min_token_count", type=int, default=20)
    ap.add_argument("--phase_max_windows", type=int, default=600)
    ap.add_argument("--n_decode_windows", type=int, default=128)
    ap.add_argument("--decode_bs", type=int, default=512)
    ap.add_argument("--skip_decode", action="store_true")
    ap.add_argument("--skip_entropy", action="store_true")
    ap.add_argument("--fig_dir", default="")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    dev = args.device
    model, cfg = load_probe(args.run, dev)
    stream = load_stream(cfg["stream"])
    ws = WindowStream(stream, T=cfg["T"], val_frac=cfg["val_frac"], seed=cfg["seed"],
                      limit_clips=cfg.get("limit_clips", 0))
    assert list(ws.val_clips) == list(cfg["val_clips"]), "val split drifted from training"
    clip_names = np.array([c["name"] for c in stream["clips"]])
    fam_of_clip = np.array([action_family(n) for n in clip_names])
    fig_dir = args.fig_dir or f"{args.run}/figs"
    os.makedirs(fig_dir, exist_ok=True)
    print(f"probe {args.run} mode={cfg['mode']} params={cfg['n_params']/1e6:.2f}M  "
          f"stream={cfg['stream']}  val clips={len(ws.val_clips)}")
    print("family counts (val clips):",
          {f: int((fam_of_clip[ws.val_clips] == f).sum()) for f in FAMILIES + ["other"]})

    R = {"run": args.run, "mode": cfg["mode"], "config": {k: cfg[k] for k in
         ("stream", "T", "mode", "slow_levels", "phase_bins", "phase_harm", "res_dims",
          "frame_dims", "steps", "bs", "seed", "n_params")}}

    # ---- A. slow-token semantics (THE STOP RULE)
    R["A_slow_semantics"] = analyze_slow(model, ws, cfg, dev, args, fam_of_clip, fig_dir)

    # ---- B. bits accounting
    hist = json.load(open(f"{args.run}/loss_history.json"))
    last = hist[-1]
    table = [{"run": args.run, "mode": cfg["mode"], "bits_per_window": cfg["bits"]["total_per_window"],
              "bits_breakdown": cfg["bits"], "val_mse": last["mse"], "val_level_acc": last["level_acc"]}]
    for r in args.compare:
        try:
            c2 = json.load(open(f"{r}/config.json")); h2 = json.load(open(f"{r}/loss_history.json"))[-1]
            table.append({"run": r, "mode": c2["mode"], "bits_per_window": c2["bits"]["total_per_window"],
                          "bits_breakdown": c2["bits"], "val_mse": h2["mse"], "val_level_acc": h2["level_acc"]})
        except FileNotFoundError as e:
            print("skip compare:", r, e)
    raw = ws.meta["bits_per_frame_nominal"] * cfg["T"]
    R["B_bits"] = {"raw_frame_codes_bits_per_window_nominal": raw, "T": cfg["T"], "table": table}
    for row in table:
        row["compression_vs_raw"] = round(raw / max(row["bits_per_window"], 1e-9), 1)
    if not args.skip_entropy:
        ent = stream_entropy(ws, ws.val_clips)
        R["B_bits"]["stream_entropy"] = ent
        R["B_bits"]["raw_bits_per_window_empirical"] = ent["H_frame_bits"] * cfg["T"]
        R["B_bits"]["ar_achievable_bits_per_window"] = (ent["H_frame_bits"] +
                                                        ent["H_cond_bits"] * (cfg["T"] - 1))

    # ---- C. phase
    R["C_phase"] = analyze_phase(model, ws, dev, args, fam_of_clip, clip_names, fig_dir)

    # ---- D. decoded quality
    if args.skip_decode:
        R["D_decode"] = {"skipped": "--skip_decode"}
    else:
        R["D_decode"] = analyze_decode(model, ws, cfg, args, dev)

    out = args.out_json or f"{args.run}/e13_eval.json"
    json.dump(R, open(out, "w"), indent=1, default=float)

    # ---- plain report
    A = R["A_slow_semantics"]
    print("\n================ E13 PROBE VERDICT ================")
    if "skipped" not in A:
        km, sr = A["kmeans"], A["stop_rule"]
        print(f"A. slow tokens vs action families")
        print(f"   k-means purity (named clips)  {km['purity_named']:.3f}   majority {km['majority_named']:.3f}"
              f"   shuffled-null p95 {km['null_purity_named_p95']:.3f}   [R7 frame-level: 0.82 vs 0.49]")
        print(f"   NMI {km['nmi']:.3f} (null p95 {km['null_nmi_p95']:.3f})   ARI {km['ari']:.3f}")
        if A.get("probe_slow", {}).get("acc") is not None:
            print(f"   linear probe  slow {A['probe_slow']['acc']:.3f} vs majority "
                  f"{A['probe_slow']['majority']:.3f}   raw-window ceiling "
                  f"{A.get('probe_raw_window_mean',{}).get('acc')}")
        v = A["vocabulary"]
        print(f"   slow vocabulary: {v['n_used']}/{v['n_possible_words']} words used on "
              f"{v['n_windows']} windows, singleton {v['singleton_frac']:.1%} "
              f"[R7 frame codes: 65%/59%], top-100 covers {v['top100_coverage']:.1%}, "
              f"H={v['entropy_bits']:.2f}/{v['entropy_bits_max']:.2f} bits")
        print(f"   >>> STOP RULE: {sr['verdict']}   checks={sr['checks']}"
              f"   semantics kept vs raw window = {sr['semantics_kept_vs_raw_window']}")
    else:
        print("A. skipped:", A["skipped"])
    print("\nB. bits per 16-frame window")
    print(f"   raw frame codes (nominal)  {raw:,.0f}")
    if "stream_entropy" in R["B_bits"]:
        e = R["B_bits"]["stream_entropy"]
        print(f"   raw, empirical H(code)     {R['B_bits']['raw_bits_per_window_empirical']:,.0f}"
              f"   AR-achievable {R['B_bits']['ar_achievable_bits_per_window']:,.0f}"
              f"   (temporal saving {e['saving']:.1%}, R7 measured 77.7%)")
    for row in table:
        print(f"   {row['mode']:<11} {row['bits_per_window']:>10.1f} bits "
              f"({row['compression_vs_raw']:>7.0f}x)   val mse {row['val_mse']:.5f}  "
              f"level_acc {row['val_level_acc']:.3f}   {os.path.basename(row['run'])}")
    C = R["C_phase"]
    print("\nC. phase")
    if "skipped" in C:
        print("   skipped:", C["skipped"])
    else:
        for f, a in sorted(C["per_family"].items()):
            print(f"   {f:<14} n={a['n_clips']:<4} monotone {a['monotone_score']:.2f}  "
                  f"|dθ| {a['mean_abs_step_rad']:.3f} rad (raw {a['raw_mean_abs_step_rad']:.3f}, "
                  f"disp {a['raw_dispersion']:.3f})  linear R² {a['linear_r2']:.2f}  "
                  f"{a['phase_hz']:.2f} Hz  hz/raw {a['hz_ratio_median']}")
        print(f"   periodic vs rest monotone: {C['periodic_vs_other']}")
        if "figure" in C:
            print("   figure:", C["figure"])
    D = R["D_decode"]
    print("\nD. decoded quality through the FROZEN tokenizer (val windows)")
    if "skipped" in D:
        print("   skipped:", D["skipped"])
    else:
        print(f"   ground-truth temporal std {D['gt_temporal_std_cm']:.3f} cm")
        print("   (tokenizer_original is the FLOOR: E19's own error on THIS corpus. It is high "
              "because\n    AMASS is unseen data for a LAFAN-trained tokenizer -- ~9 deg on LAFAN, "
              "~24 deg here.\n    E13's cost is probe minus tokenizer_original, not probe alone.)")
        for k, v in D.items():
            if isinstance(v, dict):
                print(f"   {k:<22} geodesic {v['geodesic_deg']:>6.2f} deg   pos {v['pos_cm']:>6.2f} cm"
                      f"   temporal std {v['temporal_std_cm']:>7.4f} cm")
    if "figure" in A:
        print("\nfigures:", A["figure"], C.get("figure", ""))
    print("json:", out)


if __name__ == "__main__":
    main()
