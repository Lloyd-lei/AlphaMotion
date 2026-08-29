"""V0-b — where does the information actually live? NO TRAINING.

Two of the eight required statistics from ARCH_V5 §7, both measurable today with the
frozen spatial layer and no temporal model at all:

STATISTIC #1  信息熵压缩分布 — spatial-code utilization / repetition INSIDE temporal
  windows. E19 emits, per frame, 256 code slots (2 streams x K=128 latents), each a
  13-dim integer tuple. Inside one 2 s window a given slot takes some value over and
  over (a low-information transition pose) or exactly once (a high-information key
  pose). We measure:
    - the RECURRENCE DISTRIBUTION: how many frames share each (slot, value)
    - per-frame NOVELTY: fraction of the 256 slots whose value is a one-off in this
      window — a direct per-instant information score
    - CONCENTRATION: what share of a window's total novelty sits in its top-4 frames
      (the "信息集中在少数时刻" claim, as a number)
    - WHERE the one-off codes fall: novelty aligned to MOTION TURNING POINTS (local
      maxima of key-joint acceleration), plus correlations with speed and |accel|.
      If the claim is right, novelty spikes at turning points; if it is wrong,
      novelty is flat or, worse, tracks raw speed (which would just mean "codes churn
      when things move fast", a very different and much weaker statement).
  The base-9 exact hash (E19 levels are 0..8, 9^13 < 2^63) means "same code" means
  bit-identical, never an approximate match.

STATISTIC #5  内在维度对照 — intrinsic dimension of the KEY-JOINT trajectory vs the
  FULL BODY, per action family, using R8's own estimators (R8_DIM/scripts/d0_common:
  TwoNN, Levina-Bickel MLE, correlation dimension, PCA participation ratio), with
  R8's same-clip exclusion so temporal duplicates cannot fake a low dimension.
  R8's per-joint reference: wrist 2.65, leg 0.84.

Figures -> {RUNS}/V5_figs/v0b_*.png ; numbers -> {RUNS}/V5_v0b_entropy.json

  python v5_probe_entropy.py          (defaults, ~5-15 min)
  python v5_probe_entropy.py --tiny   (CPU smoke)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                        # noqa: E402
from tokenize_stream import load_stream                      # noqa: E402

R8_SCRIPTS = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
              'R8_DIM/scripts')
OUT_JSON = f"{V.RUNS}/V5_v0b_entropy.json"
STREAM = f"{V.STREAM_ROOT}/E19a1k_human_smpl"


# --------------------------------------------------------------------------- #1 codes
def slot_hash(codes, f):
    """codes uint8 [T,D] -> exact int64 hash per (frame, slot) [T,D/f].

    E19's observed level range is 0..8, so base-9 positional encoding over f=13 dims
    is INJECTIVE (9^13 = 2.5e12 < 2^63): two frames hash equal iff their 13-dim code
    tuple is bit-identical. No collisions, no tolerance, no approximate matching."""
    c = codes.astype(np.int64).reshape(codes.shape[0], -1, f)
    p = (9 ** np.arange(f, dtype=np.int64))[None, None, :]
    return (c * p).sum(-1)                                   # [T,S]


def window_recurrence(h):
    """h [T,S] -> (count_per_element [T,S], counts of each distinct (slot,value))."""
    T, S = h.shape
    slot = np.broadcast_to(np.arange(S, dtype=np.int64)[None, :], h.shape)
    pair = np.stack([slot.ravel(), h.ravel()], 1)
    _, inv, cnt = np.unique(pair, axis=0, return_inverse=True, return_counts=True)
    return cnt[inv].reshape(T, S), cnt


def turning_points(key):
    """key [T,nk,9] -> (speed [T-1], accel [T-2], turning-point frame indices).

    Turning points = local maxima of key-joint acceleration magnitude above the
    window's own median+MAD. That is the frame-level definition of "the motion
    changed its mind", which is where the spec claims the information lives."""
    p = np.asarray(key[..., 6:9])
    sp = np.linalg.norm(np.diff(p, axis=0), axis=-1).sum(-1)          # [T-1]
    ac = np.linalg.norm(np.diff(p, n=2, axis=0), axis=-1).sum(-1)     # [T-2]
    thr = np.median(ac) + 1.4826 * np.median(np.abs(ac - np.median(ac)))
    tp = [i for i in range(1, len(ac) - 1)
          if ac[i] >= ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > thr]
    return sp, ac, np.asarray(tp, np.int64) + 1                       # -> frame index


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def stat1(kd, stream, n_win, T, fams, seed=5):
    meta = stream["meta"]; f = meta["f"]
    codes = stream["codes"]; rows_map = stream["rows"]
    bounds = stream["bounds"]; clips = stream["clips"]
    # stream clip -> family, and stream clip -> cache clip name
    fam_of = [V.family_of(c["name"]) for c in clips]
    g = np.random.default_rng(seed)
    picks = []                                       # (clip, start_in_stream, family)
    per_fam = max(1, n_win // max(len(fams), 1))
    for fam in fams:
        cand = [c for c in range(len(clips))
                if fam_of[c] == fam and clips[c]["len"] >= T]
        for c in g.permutation(cand)[:per_fam]:
            s = int(bounds[c]); L = int(bounds[c + 1] - bounds[c])
            picks.append((int(c), s + int(g.integers(0, L - T + 1)), fam))
    # plus a random sample over everything, so the distribution is not family-skewed
    allc = [c for c in range(len(clips)) if clips[c]["len"] >= T]
    for c in g.permutation(allc)[:n_win]:
        s = int(bounds[c]); L = int(bounds[c + 1] - bounds[c])
        picks.append((int(c), s + int(g.integers(0, L - T + 1)), fam_of[c]))

    rec_hist = {}                     # recurrence count -> number of (slot,value)
    nov_by_fam, conc_by_fam = {}, {}
    nov_prof = np.zeros(T); nov_n = 0
    LAG = 8
    tp_prof = np.zeros(2 * LAG + 1); tp_n = 0
    cor_sp, cor_ac, per_win = [], [], []
    for ci, s0, fam in picks:
        w = np.asarray(codes[s0:s0 + T])
        h = slot_hash(w, f)
        cpe, cnt = window_recurrence(h)
        for c, k in zip(*np.unique(cnt, return_counts=True)):
            rec_hist[int(c)] = rec_hist.get(int(c), 0) + int(k)
        nov = (cpe == 1).mean(1)                                  # [T] per-frame novelty
        nov_prof += nov; nov_n += 1
        srt = np.sort(nov)[::-1]
        conc = float(srt[:4].sum() / max(nov.sum(), 1e-9))        # top-4 share
        # motion side: the SAME frames, key-joint trajectory out of the cache
        rows = rows_map[s0:s0 + T]
        key = kd.key["human_smpl"][torch.as_tensor(rows)].numpy()
        sp, ac, tps = turning_points(key)
        cor_sp.append(pearson(nov[1:], sp)); cor_ac.append(pearson(nov[1:-1], ac))
        for t in tps:
            lo, hi = t - LAG, t + LAG + 1
            if lo < 0 or hi > T:
                continue
            m = nov.mean() + 1e-9
            tp_prof += nov[lo:hi] / m; tp_n += 1
        nov_by_fam.setdefault(fam, []).append(float(nov.mean()))
        conc_by_fam.setdefault(fam, []).append(conc)
        per_win.append({"clip": clips[ci]["name"][:60], "family": fam,
                        "novelty_mean": float(nov.mean()), "top4_share": conc,
                        "n_turning_points": int(len(tps))})

    tot = sum(rec_hist.values())
    one_off = rec_hist.get(1, 0) / max(tot, 1)
    # a "recurrent" code is one that fills at least a quarter of the window
    recurrent = sum(v for k, v in rec_hist.items() if k >= T // 4) / max(tot, 1)
    out = {"n_windows": len(picks), "T": T, "n_slots": int(meta["D"] // f),
           "recurrence_hist": {str(k): int(v) for k, v in sorted(rec_hist.items())},
           "frac_codes_one_off": round(float(one_off), 4),
           "frac_codes_recurrent_ge_quarter_window": round(float(recurrent), 4),
           "mean_distinct_values_per_slot": round(float(tot / max(len(picks) *
                                                 (meta["D"] // f), 1)), 3),
           "novelty_profile_over_window": (nov_prof / max(nov_n, 1)).round(4).tolist(),
           "novelty_at_turning_points": (tp_prof / max(tp_n, 1)).round(4).tolist()
           if tp_n else None,
           "turning_point_lag_axis": list(range(-LAG, LAG + 1)),
           "n_turning_points": int(tp_n),
           "corr_novelty_speed": round(float(np.mean(cor_sp)), 4),
           "corr_novelty_accel": round(float(np.mean(cor_ac)), 4),
           "by_family": {k: {"novelty_mean": round(float(np.mean(v)), 4),
                             "top4_share": round(float(np.mean(conc_by_fam[k])), 4),
                             "n": len(v)}
                         for k, v in sorted(nov_by_fam.items())},
           "per_window": per_win}
    if tp_n:
        prof = np.asarray(out["novelty_at_turning_points"])
        out["turning_point_peak_over_mean"] = round(float(prof[LAG]), 4)
        out["turning_point_edge_over_mean"] = round(float(prof[[0, -1]].mean()), 4)
    return out


# --------------------------------------------------------------------------- #5 dim
def stat5(kd, fams, T, n_win, device, kmax=21):
    """Intrinsic dimension: key-joint window trajectories vs whole-body, per family."""
    if R8_SCRIPTS not in sys.path:
        sys.path.insert(0, R8_SCRIPTS)
    import d0_common as D0
    D0.DEV = device
    cache = kd.cache
    body = "human_smpl"
    kidx = kd.kidx[body]
    out = {"estimators": "R8_DIM/scripts/d0_common (TwoNN, Levina-Bickel MLE, "
                         "correlation dim, PCA participation ratio); same-clip "
                         "neighbours excluded, as in R8",
           "T": T, "key_joints": kd.knames[body],
           "r8_reference": {"wrist_per_joint": V.REF["wrist_intrinsic_dim"],
                            "leg_per_joint": V.REF["leg_intrinsic_dim"]},
           "by_family": {}}
    for fam in list(fams) + ["ALL"]:
        subset = None if fam == "ALL" else [c for c in range(len(kd.clips))
                                            if kd.family[c] == fam]
        if subset is not None and not subset:
            continue
        starts, cid = D0.sample_windows(cache, T, n_win, seed=0, clip_subset=subset)
        if len(starts) < 200:
            print(f"  [dim] {fam}: only {len(starts)} windows, skipped", flush=True)
            continue
        grp = torch.as_tensor(cid.astype(np.int64), device=device)
        row = {"n_windows": int(len(starts))}
        for tag, joints in (("key6", kidx), ("full_body", None)):
            X, _ = D0.window_features(cache, body, starts, T, drop_root=True,
                                      joints=joints)
            r, _ = D0.all_estimators(X, group=grp, kmax=kmax, tag=tag)
            row[tag] = r
        row["ratio_key_over_full_twonn"] = round(
            row["key6"]["twonn"] / max(row["full_body"]["twonn"], 1e-6), 3)
        row["ratio_key_over_full_mle10"] = round(
            row["key6"]["mle_k10"] / max(row["full_body"]["mle_k10"], 1e-6), 3)
        out["by_family"][fam] = row
        print(f"  [dim] {fam:6s} key6 TwoNN {row['key6']['twonn']:6.2f} "
              f"MLE10 {row['key6']['mle_k10']:6.2f} | full {row['full_body']['twonn']:6.2f} "
              f"/ {row['full_body']['mle_k10']:6.2f} | ratio "
              f"{row['ratio_key_over_full_twonn']:.2f}", flush=True)
    return out


# --------------------------------------------------------------------------- figures
def figures(s1, s5, figdir):
    plt = V.fig_style()
    os.makedirs(figdir, exist_ok=True)
    paths = []

    # 1. recurrence distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    k = np.array([int(x) for x in s1["recurrence_hist"]])
    v = np.array([s1["recurrence_hist"][str(x)] for x in k], float)
    o = np.argsort(k); k, v = k[o], v[o]
    ax[0].loglog(k, v / v.sum(), "o-", color=V.SERIES[0], ms=4)
    ax[0].set_xlabel("frames in the window sharing this exact spatial code")
    ax[0].set_ylabel("share of (slot, code) pairs")
    ax[0].set_title(f"recurrence: {s1['frac_codes_one_off']*100:.1f}% of codes are "
                    f"one-offs,\n{s1['frac_codes_recurrent_ge_quarter_window']*100:.1f}% "
                    f"fill >= 1/4 of the window")
    ax[1].plot(s1["novelty_profile_over_window"], color=V.SERIES[1])
    ax[1].set_xlabel("frame index in the 2 s window")
    ax[1].set_ylabel("novelty = share of one-off slots")
    ax[1].set_title("per-frame novelty, averaged over windows")
    fig.tight_layout(); p = f"{figdir}/v0b_recurrence.png"; fig.savefig(p); plt.close(fig)
    paths.append(p)

    # 2. novelty at turning points — the actual claim
    if s1.get("novelty_at_turning_points"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        lag = s1["turning_point_lag_axis"]
        ax[0].axhline(1.0, color="#898781", lw=1, ls="--")
        ax[0].plot(lag, s1["novelty_at_turning_points"], "o-", color=V.SERIES[2], ms=4)
        ax[0].axvline(0, color="#c3c2b7", lw=1)
        ax[0].set_xlabel("frames from a motion turning point")
        ax[0].set_ylabel("novelty / window mean")
        ax[0].set_title(f"is information at the turning points?\npeak "
                        f"{s1.get('turning_point_peak_over_mean', float('nan')):.3f}x "
                        f"vs edges "
                        f"{s1.get('turning_point_edge_over_mean', float('nan')):.3f}x "
                        f"({s1['n_turning_points']} points)")
        fams = list(s1["by_family"]); y = [s1["by_family"][f]["top4_share"] for f in fams]
        ax[1].bar(fams, y, color=V.SERIES[3])
        ax[1].axhline(4 / s1["T"], color="#898781", ls="--", lw=1)
        ax[1].set_ylabel("share of window novelty in its top-4 frames")
        ax[1].set_title(f"concentration (dashed = uniform, 4/{s1['T']})")
        ax[1].tick_params(axis="x", rotation=30)
        fig.tight_layout(); p = f"{figdir}/v0b_turning_points.png"
        fig.savefig(p); plt.close(fig); paths.append(p)

    # 3. intrinsic dimension key vs full
    if s5["by_family"]:
        fams = list(s5["by_family"])
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        x = np.arange(len(fams))
        for i, (tag, lbl) in enumerate((("key6", "6 key joints"),
                                        ("full_body", "whole body"))):
            ax.bar(x + (i - .5) * .38, [s5["by_family"][f][tag]["twonn"] for f in fams],
                   .38, label=lbl, color=V.SERIES[i])
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30)
        ax.set_ylabel("intrinsic dimension (TwoNN)")
        ax.set_title(f"key-joint trajectory vs whole body, {s5['T']}-frame windows")
        ax.legend(frameon=False)
        fig.tight_layout(); p = f"{figdir}/v0b_intrinsic_dim.png"
        fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=STREAM)
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--T", type=int, default=V.T_WIN)
    ap.add_argument("--n_win", type=int, default=400)
    ap.add_argument("--n_dim_win", type=int, default=2500)
    ap.add_argument("--families", default=",".join(V.V5_FAMILIES))
    ap.add_argument("--skip_dim", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--figs", default=V.FIGS)
    args = ap.parse_args()
    if args.tiny:
        args.n_win, args.n_dim_win, args.T = 12, 400, 24
        args.families, args.device = "walk,punch", "cpu"
    t0 = time.time()

    print("=== V5 V0-b  information distribution inside temporal windows (NO TRAINING) ===")
    fams = [f for f in args.families.split(",") if f]
    kd = V.KeyData(args.cache, ["human_smpl"], device=args.device)
    st = load_stream(args.stream, mmap=True)
    print(f"stream {args.stream}  frames={st['meta']['n_frames']} D={st['meta']['D']} "
          f"K={st['meta']['K']} f={st['meta']['f']} -> "
          f"{st['meta']['D']//st['meta']['f']} code slots per frame", flush=True)

    print("\n--- statistic #1: spatial-code utilization / repetition ---", flush=True)
    s1 = stat1(kd, st, args.n_win, args.T, fams)
    print(f"  windows {s1['n_windows']}  one-off codes "
          f"{s1['frac_codes_one_off']*100:.1f}%  recurrent(>=T/4) "
          f"{s1['frac_codes_recurrent_ge_quarter_window']*100:.1f}%")
    print(f"  mean distinct values per slot per window: "
          f"{s1['mean_distinct_values_per_slot']:.2f} of {args.T} frames")
    print(f"  novelty correlation with speed {s1['corr_novelty_speed']:+.3f}, "
          f"with |accel| {s1['corr_novelty_accel']:+.3f}")
    if s1.get("turning_point_peak_over_mean") is not None:
        print(f"  novelty at turning points: {s1['turning_point_peak_over_mean']:.3f}x "
              f"the window mean (edges {s1['turning_point_edge_over_mean']:.3f}x, "
              f"{s1['n_turning_points']} turning points)")
    for f_, b in s1["by_family"].items():
        print(f"    {f_:7s} novelty {b['novelty_mean']:.3f}  top-4 share "
              f"{b['top4_share']:.3f}  (uniform would be {4/args.T:.3f})")

    s5 = {"by_family": {}}
    if not args.skip_dim:
        print("\n--- statistic #5: intrinsic dimension, key joints vs whole body ---",
              flush=True)
        try:
            s5 = stat5(kd, fams, args.T, args.n_dim_win, args.device)
        except Exception as e:                       # never block the queue on a figure
            print(f"  DIM FAILED: {type(e).__name__}: {e}", flush=True)
            s5 = {"by_family": {}, "error": f"{type(e).__name__}: {e}"}

    figs = []
    try:
        figs = figures(s1, s5, args.figs)
        for p in figs:
            print("FIG:", p, flush=True)
    except Exception as e:
        print(f"  FIGURES FAILED: {type(e).__name__}: {e}", flush=True)

    # ------------------------------------------------------------------- verdict
    conc = s1["by_family"]
    uniform = 4.0 / args.T
    mean_top4 = float(np.mean([b["top4_share"] for b in conc.values()])) if conc else 0.0
    peak = s1.get("turning_point_peak_over_mean")
    concentrated = mean_top4 > 1.5 * uniform
    # A peak above the WINDOW mean is not evidence of localization: windows that
    # contain turning points are moving windows, and novelty tracks speed, so the
    # whole neighbourhood floats up together. The claim "information sits AT the
    # turning points" requires the peak to stand above its OWN neighbourhood, i.e.
    # above the value 8 frames away. First measurement (2026-08-06): peak 1.146x the
    # window mean but edges 1.147x — a flat plateau, not a peak.
    edge = s1.get("turning_point_edge_over_mean")
    at_turns = (peak is not None and edge is not None
                and peak > 1.05 and peak > 1.05 * edge)
    dims = s5.get("by_family", {})
    dim_lower = None
    if "ALL" in dims:
        dim_lower = dims["ALL"]["ratio_key_over_full_twonn"] < 1.0

    res = {"probe": "V0-b information distribution (spec statistics #1 and #5)",
           "stream": args.stream, "cache": args.cache,
           "stat1_code_utilization": s1, "stat5_intrinsic_dimension": s5,
           "figures": figs,
           "verdict": {"novelty_concentrated_in_few_frames": bool(concentrated),
                       "novelty_peaks_at_turning_points": bool(at_turns),
                       "key_joint_dim_lower_than_full_body": dim_lower,
                       "mean_top4_share": round(mean_top4, 4),
                       "uniform_top4_share": round(uniform, 4)}}

    print("\nVERDICT (plain words)")
    print(f"  information concentration: the top 4 frames of a {args.T}-frame window "
          f"carry {mean_top4*100:.1f}% of its novelty (uniform would be "
          f"{uniform*100:.1f}%) -> "
          f"{'CONCENTRATED' if concentrated else 'NOT concentrated — spread evenly'}")
    if peak is not None:
        print(f"  where: novelty at motion turning points is {peak:.2f}x the window "
              f"mean, but {edge:.2f}x 8 frames away -> "
              f"{'LOCALIZED at the turning points' if at_turns else 'a FLAT PLATEAU, not a peak: the whole moving neighbourhood is up, the turning point itself is not special'}")
    print(f"  what novelty tracks: speed r={s1['corr_novelty_speed']:+.2f}, "
          f"|accel| r={s1['corr_novelty_accel']:+.2f}")
    if dim_lower is not None:
        a = dims["ALL"]
        print(f"  intrinsic dimension: key-joint trajectory "
              f"{a['key6']['twonn']:.2f} vs whole body {a['full_body']['twonn']:.2f} "
              f"(TwoNN) -> {'LOWER as predicted' if dim_lower else 'NOT lower'}")

    msg = []
    if not concentrated:
        msg.append(f"ASSUMPTION NOT SUPPORTED: information is NOT concentrated in a few "
                   f"instants. Top-4 frames hold {mean_top4*100:.1f}% of window novelty "
                   f"vs {uniform*100:.1f}% for a uniform spread. The keyframe premise "
                   f"behind '快动作中间可压' is weaker than ARCH_V5 §1 assumes.")
    if peak is not None and not at_turns:
        msg.append(f"ASSUMPTION NOT SUPPORTED: one-off codes do NOT LOCALIZE at motion "
                   f"turning points — {peak:.2f}x the window mean at the turning point "
                   f"but {edge:.2f}x eight frames away, i.e. a flat plateau. Novelty "
                   f"tracks raw SPEED (r={s1['corr_novelty_speed']:+.2f}) far more than "
                   f"acceleration (r={s1['corr_novelty_accel']:+.2f}). The honest reading "
                   f"is 'codes churn while the body is moving', NOT 'information lives at "
                   f"the moments the motion changes its mind'. Keyframe placement should "
                   f"not be built on turning points until this is re-measured.")
    if dim_lower is False:
        msg.append(f"ASSUMPTION NOT SUPPORTED: the key-joint trajectory is NOT lower "
                   f"dimensional than the whole body "
                   f"(ratio {dims['ALL']['ratio_key_over_full_twonn']:.2f}). The 60x "
                   f"state reduction in ARCH_V5 §2 does not buy a simpler manifold.")
    if msg:
        V.loud(msg)
    else:
        V.loud(["V0-b: both statistics support the design — information is "
                "concentrated, it sits at turning points, and the key-joint "
                "trajectory is the lower-dimensional object."], mark="===")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(f"SAVED {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
