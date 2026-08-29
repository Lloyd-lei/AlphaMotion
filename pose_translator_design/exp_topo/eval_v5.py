"""V5 evaluation — all EIGHT statistics of ARCH_V5 §7, one table plus figures.

  eval_v5.py --runs RUN_A,RUN_B,RUN_C

  ->  /var/tmp/alphamotion/topo_exp/runs/V5_verdict_0806.txt      (the table)
      /var/tmp/alphamotion/topo_exp/runs/V5_figs/*.png            (the figures)
      {each run}/eval_v5.json                                     (the numbers)

The eight, in the spec's order:
  1 information/entropy — spatial-code recurrence inside windows is measured by
    v5_probe_entropy.py (no training needed); here we add the V5 codebooks' OWN
    utilization: how many of the 15,625 words per register are ever used, the
    realized entropy in bits against the 13.93 bit the FSQ grid could carry, and
    how often a path code repeats across different windows.
  2 PATH-TOKEN SPEED INVARIANCE — the non-negotiable one. ACCEPTANCE > 0.95; the
    current architecture measured 0.644 and incoherent motion 0.246. Two recipes:
      (2a) pure resample: the same raw path read out at 30 / 60 / 120 frames. V5's
           path readout only ever sees a canonically-resampled window, so this is
           high BY CONSTRUCTION and is labelled as such — it is a design check, not
           a discovery.
      (2b) lloyd's reference recipe, the one 0.644 came from: resample a 60-frame
           window's path into 30 frames and append the FOLLOWING 30 frames, against
           the original 60. Both windows are 60 frames long and their content really
           differs, so this one is a genuine test and is the number to quote.
  3 TIME-TOKEN DISTRIBUTION — do gears emerge on their own (no hand-set bins
    anywhere), and do they track the actual speed (Spearman against measured
    arc length per frame)?
  4 VOCABULARY SATURATION — distinct path codes vs windows seen. A decaying
    new-token rate is the evidence the vocabulary is being REUSED rather than
    memorized; a straight line means one code per window, i.e. a hash.
  5 INTRINSIC DIMENSION — read from v5_probe_entropy.py's output (R8 estimators).
  6 SPECTRAL COVERAGE — per family, what fraction of key-joint motion energy falls
    inside the token bandwidth. n_path tokens per 2 s window is a sampling rate of
    n_path/2 Hz, so the honest Nyquist limit is n_path/4 Hz. Reference: run was
    66.5%, and 33% of run/jacks energy lives above 2 Hz.
  7 CROSS-EMBODIMENT LEAKAGE — kNN body probe on the temporal tokens over 6 bodies
    (chance 16.7%). P1's was 14.1%, essentially blind; V5 must not regress.
  8 RECONSTRUCTION LADDER — prefix 1/2/4/8 motion energy retention and error, with
    four reference rows ALWAYS present: the straight-line-between-endpoints baseline
    (what you get for zero temporal bits), the FRAME-CODE FLOOR (GT key joints pushed
    through the frozen spatial layer — the ceiling on anything the temporal layer can
    deliver downstream), and the two token ablations that decide whether the codebook
    is real at all: registers ZEROED, and registers SHUFFLED across windows. The last
    one is the important one. A model can look good against the baseline while its
    tokens are only a generic "not zero" wake-up signal, and the prefix ladder alone
    cannot tell the difference — it goes flat either way. The derived content_share
    (how much of the registers' benefit depends on WHICH window's token was used) is
    printed under the ladder in words.

Every section is independently guarded: one broken statistic never costs the other
seven, and what failed is written into the table.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                        # noqa: E402
from train_v5 import load_run, to_states                     # noqa: E402

R7_BODIES = ["human_smpl", "unitree_g1_29dof", "pal_talos", "unitree_h1",
             "booster_t1", "fourier_gr3"]


# --------------------------------------------------------------------------- helpers
@torch.no_grad()
def tokens_of(model, w, dev, bs=256, cur=None, tgt=None, kn=None):
    """-> (path_q [n,n_path,f], time_q [n,n_time,f], path_ints, time_ints) on cpu."""
    pq, tq, pi, ti = [], [], [], []
    for i in range(0, w.shape[0], bs):
        x = w[i:i + bs].to(dev).float()
        c = x[:, 0] if cur is None else cur[i:i + bs].to(dev)
        g = x[:, -1] if tgt is None else tgt[i:i + bs].to(dev)
        k = torch.ones(x.shape[0], device=dev) if kn is None else kn[i:i + bs].to(dev)
        e = model.encode(x, c, g, k)
        a, b = model.token_ints(e)
        pq.append(e["path_q"].float().cpu()); tq.append(e["time_q"].float().cpu())
        pi.append(a.cpu()); ti.append(b.cpu())
    return (torch.cat(pq), torch.cat(tq), torch.cat(pi), torch.cat(ti))


def cos_rows(a, b):
    a = torch.nn.functional.normalize(a.reshape(a.shape[0], -1), dim=-1)
    b = torch.nn.functional.normalize(b.reshape(b.shape[0], -1), dim=-1)
    return (a * b).sum(-1)


def code_keys(ints):
    """[n,R,f] integer codes -> list of n hashable STRING keys (whole register bank).

    Strings, not tuples, and this is load-bearing: np.array(list_of_tuples,
    dtype=object) builds a 2D object array, so np.unique(...) then counts distinct
    ELEMENTS (the 5 FSQ levels) instead of distinct CODES. That bug made the first
    verdict table report "5 distinct path codes / 5 time gears" for every arm — which
    is just the level count wearing a disguise. A 1-D array of joined strings makes
    np.unique count what it is supposed to count.
    """
    return ["|".join(map(str, x))
            for x in ints.reshape(ints.shape[0], -1).numpy().tolist()]


def entropy_bits(keys):
    _, c = np.unique(np.asarray(keys), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- 1 codebook
def stat1_codebook(model, kd, emb, dev, n, T, span, seed=11):
    g = np.random.default_rng(seed)
    st, _ = kd.sample_starts(n, g, "val", span=span)
    w = kd.window(emb, st, T, span)
    pq, tq, pi, ti = tokens_of(model, w, dev)
    kp, kt = code_keys(pi), code_keys(ti)
    up, cp = np.unique(np.array(kp, dtype=object).astype(str), return_counts=True)
    ut, ct = np.unique(np.array(kt, dtype=object).astype(str), return_counts=True)
    # per-register level usage: how much of the 5-level grid each dim actually visits
    lv = model.fsq_path._levels.cpu().numpy()
    used = [(len(np.unique(pi[:, r].numpy(), axis=0))) for r in range(model.n_path)]
    return {"n_windows": int(n),
            "path_distinct_codes": int(len(up)),
            "path_repeat_rate": round(float(1 - len(up) / len(kp)), 4),
            "path_entropy_bits": round(entropy_bits(kp), 3),
            "path_max_bits": round(float(model.n_path * np.log2(lv).sum()), 3),
            "path_distinct_per_register": used,
            "time_distinct_codes": int(len(ut)),
            "time_entropy_bits": round(entropy_bits(kt), 3),
            "time_max_bits": round(float(model.n_time * np.log2(lv).sum()), 3),
            "time_top1_share": round(float(ct.max() / ct.sum()), 4)}


# --------------------------------------------------------------------------- 2 invariance
def stat2_speed(model, kd, emb, dev, n, T, span, seed=13):
    """The acceptance statistic. Two recipes, plus the incoherent-motion control."""
    g = np.random.default_rng(seed)
    st, _ = kd.sample_starts(n, g, "val", span=2 * span + 8)   # room for recipe 2b
    base = kd.window(emb, st, T, span)
    p0 = tokens_of(model, base, dev)[0]
    out = {"acceptance": V.REF["path_speed_cos_accept"],
           "reference_current_arch": V.REF["path_speed_cos_current"],
           "reference_unrelated": V.REF["path_speed_cos_unrelated"]}

    # ---- 2a pure resample: the SAME raw path read out at a different frame count
    for tag, T2 in (("half_30f", T // 2), ("double_120f", 2 * T)):
        p = tokens_of(model, kd.window(emb, st, T2, span), dev)[0]
        out[f"2a_{tag}"] = round(float(cos_rows(p0, p).mean()), 4)
    out["2a_pure_resample"] = round(float(np.mean([out["2a_half_30f"],
                                                   out["2a_double_120f"]])), 4)

    # ---- 2b lloyd's recipe: first `span` frames at 2x + the following span/2 frames
    idx = np.concatenate([V.uniform_idx(span, T // 2),
                          np.arange(span, span + T - T // 2, dtype=np.int64)])
    w2 = kd.window_idx(emb, st, idx)
    p2 = tokens_of(model, w2, dev)[0]
    out["2b_lloyd_recipe"] = round(float(cos_rows(p0, p2).mean()), 4)

    # ---- control: unrelated windows (a shuffled pairing of the same set)
    perm = torch.randperm(len(st))
    out["control_unrelated"] = round(float(cos_rows(p0, p0[perm]).mean()), 4)
    out["pass_2b"] = bool(out["2b_lloyd_recipe"] > V.REF["path_speed_cos_accept"])
    out["note_2a"] = ("V5's path readout sees only a canonically-resampled window, so "
                      "2a is high BY CONSTRUCTION — a design check, not a discovery. "
                      "2b is the number to quote.")
    return out


# --------------------------------------------------------------------------- 3 time
def stat3_time(model, kd, emb, dev, n, T, span, seed=17):
    """Do gears emerge, and do they track real speed?"""
    g = np.random.default_rng(seed)
    st, _ = kd.sample_starts(n, g, "val", span=span)
    keys, speeds, dim0 = [], [], []
    for T2 in (T // 2, int(T * 0.75), T, int(T * 1.5), 2 * T):
        w = kd.window(emb, st, T2, span)
        _, tq, _, ti = tokens_of(model, w, dev)
        keys += code_keys(ti)
        c = V.arclen_curve(w.float())
        speeds += (c[:, -1] / T2).tolist()        # arc length per frame = real speed
        dim0 += tq[:, 0, 0].tolist()
    u, c = np.unique(np.array(keys, dtype=object).astype(str), return_counts=True)
    o = np.argsort(-c)
    return {"n_samples": len(keys), "distinct_time_codes": int(len(u)),
            "entropy_bits": round(entropy_bits(keys), 3),
            "top5_share": [round(float(x / c.sum()), 4) for x in c[o][:5]],
            "gears_covering_80pct": int(np.searchsorted(np.cumsum(c[o]) / c.sum(), .8) + 1),
            "spearman_code_dim0_vs_speed": round(spearman(dim0, speeds), 4),
            "speed_range_arclen_per_frame": [round(float(np.min(speeds)), 4),
                                             round(float(np.max(speeds)), 4)],
            "hist_counts": c[o][:24].tolist()}


# --------------------------------------------------------------------------- 4 saturation
def stat4_saturation(model, kd, emb, dev, n, T, span, seed=19):
    g = np.random.default_rng(seed)
    st, _ = kd.sample_starts(n, g, "train", span=span)
    w = kd.window(emb, st, T, span)
    keys = code_keys(tokens_of(model, w, dev)[2])
    seen, curve = set(), []
    for i, k in enumerate(keys):
        seen.add(k)
        if (i + 1) % max(1, n // 60) == 0:
            curve.append((i + 1, len(seen)))
    xs = np.array([a for a, _ in curve], float); ys = np.array([b for _, b in curve], float)
    tail = max(2, len(curve) // 4)
    rate_end = float((ys[-1] - ys[-tail]) / max(xs[-1] - xs[-tail], 1))
    rate_start = float((ys[tail - 1] - ys[0]) / max(xs[tail - 1] - xs[0], 1))
    sl = float(np.polyfit(np.log(xs), np.log(np.maximum(ys, 1)), 1)[0])
    return {"n_windows": int(n), "distinct_at_end": int(ys[-1]),
            "new_token_rate_first_quarter": round(rate_start, 4),
            "new_token_rate_last_quarter": round(rate_end, 4),
            "decay_ratio": round(rate_end / max(rate_start, 1e-6), 4),
            "loglog_slope": round(sl, 3),
            "saturating": bool(rate_end < 0.8 * rate_start),
            "curve": [[int(a), int(b)] for a, b in curve]}


# --------------------------------------------------------------------------- 6 spectrum
def stat6_spectrum(kd, emb, T, span, n_per_fam, fams, seed=23):
    """Fraction of key-joint motion energy inside the token bandwidth. Independent of
    the trained model — it is a property of the DATA versus the token rate."""
    g = np.random.default_rng(seed)
    out = {}
    for fam in list(fams) + ["ALL"]:
        cl = None if fam == "ALL" else [c for c in range(len(kd.clips))
                                        if kd.family[c] == fam]
        if cl is not None and not cl:
            continue
        st, _ = kd.sample_starts(n_per_fam, g, "val", span=span,
                                 families=None if fam == "ALL" else [fam])
        w = kd.window(emb, st, T, span).numpy()[..., 6:9]      # positions
        x = w - w.mean(1, keepdims=True)
        F_ = np.fft.rfft(x, axis=1)
        P = (np.abs(F_) ** 2).sum((2, 3))                      # [n, T//2+1]
        fr = np.fft.rfftfreq(T, d=1.0 / V.FPS)
        tot = P.sum(1) + 1e-12
        out[fam] = {f"below_{b}Hz": round(float((P[:, fr <= b].sum(1) / tot).mean()), 4)
                    for b in (1.0, 2.0, 4.0)}
        out[fam]["above_2Hz"] = round(1 - out[fam]["below_2.0Hz"], 4)
        out[fam]["n"] = int(len(st))
    return out


# --------------------------------------------------------------------------- 7 leakage
def stat7_leakage(model, kd, dev, n_per, T, span, k=10, seed=29):
    g = np.random.default_rng(seed)
    st, cid = kd.sample_starts(n_per, g, "val", span=span)
    embs = [e for e in R7_BODIES if e in kd.key]
    X, body, clip = [], [], []
    for bi, e in enumerate(embs):
        w = kd.window(e, st, T, span)
        pq, tq, _, _ = tokens_of(model, w, dev)
        X.append(torch.cat([pq.reshape(len(st), -1), tq.reshape(len(st), -1)], -1))
        body += [bi] * len(st); clip += list(cid)
    X = torch.nn.functional.normalize(torch.cat(X), dim=-1).to(dev)
    body = torch.as_tensor(body, device=dev); clip = torch.as_tensor(clip, device=dev)
    sim = X @ X.t()
    sim.fill_diagonal_(-2.0)
    same = (clip[:, None] == clip[None, :]) & (body[:, None] == body[None, :])
    sim = sim.masked_fill(same, -2.0)
    pred = torch.mode(body[sim.topk(k, -1).indices], -1).values
    acc = (pred == body).float()
    return {"bodies": embs, "n_per_body": int(n_per), "chance": round(1 / len(embs), 4),
            "knn_body_acc": round(float(acc.mean()), 4),
            "per_body": {e: round(float(acc[body == i].mean()), 4)
                         for i, e in enumerate(embs)},
            "p1_reference": V.REF["knn_body_leak_p1"],
            "pass_no_regression": bool(float(acc.mean()) <=
                                       V.REF["knn_body_leak_p1"] + 0.05)}


# --------------------------------------------------------------------------- 8 ladder
@torch.no_grad()
def stat8_ladder(model, kd, emb, dev, n, T, span, tok=None, ent=None, seed=31):
    g = np.random.default_rng(seed)
    st, _ = kd.sample_starts(n, g, "val", span=span)
    w = kd.window(emb, st, T, span).to(dev).float()
    cur, tgt = w[:, 0], w[:, -1]
    kn = torch.ones(w.shape[0], device=dev)
    gt_std = float(w.std(1).mean())
    rows = {}
    for b in model.budgets:
        hid = (torch.arange(model.n_path, device=dev)[None] >= b).expand(w.shape[0], -1)
        o, _ = model(w, cur, tgt, kn, hid)
        p = to_states(o, model, cur, tgt, kn).float()
        pe, re = V.key_se3_err(p, w, kd.reach[emb])
        l, _ = V.pathint_loss(p, w)
        rows[f"prefix{b}"] = {"energy_ratio": round(float(p.std(1).mean()) / gt_std, 3),
                              "key_pos_cm": round(pe, 2), "key_rot_deg": round(re, 1),
                              "pathint": round(float(l), 4)}
    # ---- THE CONTROL THAT DECIDES WHETHER THE CODEBOOK IS REAL --------------------
    # Two ablations, and the gap between them is the whole story:
    #   zeroed   : path registers set to 0 — the decoder gets only the endpoints
    #   shuffled : each window is decoded with ANOTHER window's path token
    # If shuffled ~= correct, the tokens are a generic "not zero" wake-up signal and
    # carry almost no window-specific content, no matter how good the headline error
    # looks. Measured on V5_pathint_0806 (60k steps, 2026-08-06): correct 7.78 cm,
    # SHUFFLED 8.33 cm, zeroed 11.08 cm, straight line 9.32 cm — i.e. only 0.55 cm of
    # the 3.30 cm the registers buy is actually about THIS window. A flat prefix
    # ladder alone cannot show that; this row can.
    e = model.encode(w, cur, tgt, kn)
    o = model.decode(torch.zeros_like(e["path_z"]), e["time_z"], cur, tgt, kn, T)
    p = to_states(o, model, cur, tgt, kn).float()
    pe0, re0 = V.key_se3_err(p, w, kd.reach[emb])
    rows["tokens_zeroed"] = {"energy_ratio": round(float(p.std(1).mean()) / gt_std, 3),
                             "key_pos_cm": round(pe0, 2), "key_rot_deg": round(re0, 1),
                             "pathint": round(float(V.pathint_loss(p, w)[0]), 4),
                             "note": "path registers = 0: the endpoints alone"}
    # A SINGLE shuffle draw is far too noisy for the 0.6 gate content_share is judged
    # against: on identical checkpoints the run-to-run spread measured ~0.1, the same
    # size as the gaps between arms (V5c read 0.619 then 0.527 fifty minutes apart;
    # N3a 0.700 then 0.601 against a 0.6 gate). Average n_sh seeded draws and carry
    # the spread so a near-gate number can be recognised as a coin flip.
    n_sh, sh_pe, sh_re, sh_en, sh_pi = 8, [], [], [], []
    for k in range(n_sh):
        gsh = torch.Generator().manual_seed(seed * 1000 + k)
        perm = torch.randperm(w.shape[0], generator=gsh).to(dev)
        o = model.decode(e["path_z"][perm], e["time_z"], cur, tgt, kn, T)
        ps = to_states(o, model, cur, tgt, kn).float()
        a_, b_ = V.key_se3_err(ps, w, kd.reach[emb])
        sh_pe.append(a_); sh_re.append(b_)
        sh_en.append(float(ps.std(1).mean()) / gt_std)
        sh_pi.append(float(V.pathint_loss(ps, w)[0]))
    pes, res = float(np.mean(sh_pe)), float(np.mean(sh_re))
    rows["tokens_shuffled"] = {"energy_ratio": round(float(np.mean(sh_en)), 3),
                               "key_pos_cm": round(pes, 2), "key_rot_deg": round(res, 1),
                               "key_pos_cm_sd": round(float(np.std(sh_pe)), 3),
                               "n_draws": n_sh,
                               "pathint": round(float(np.mean(sh_pi)), 4),
                               "note": f"another window's path token, mean of {n_sh} "
                                       "seeded draws: anything this row already "
                                       "achieves is NOT window-specific content"}
    full = rows[f"prefix{model.budgets[-1]}"]["key_pos_cm"]
    gain = pe0 - full                                    # what the registers buy at all
    content = pes - full                                 # ...of which this is specific
    rows["_token_content"] = {
        "registers_buy_cm": round(gain, 3),
        "window_specific_cm": round(content, 3),
        "content_share": round(content / gain, 3) if gain > 1e-6 else None,
        "content_share_sd": (round(float(np.std(sh_pe)) / gain, 3)
                             if gain > 1e-6 else None),
        "note": "content_share = fraction of the registers' benefit that depends on "
                "WHICH window's token was used. Near 0 means the codebook is close to "
                "decorative; near 1 means the bits are really carrying the motion."}
    bl = model.baseline(cur, tgt, kn, T).float()
    pe, _ = V.key_se3_err(bl, w, kd.reach[emb])
    rows["straight_line"] = {"energy_ratio": round(float(bl.std(1).mean()) / gt_std, 3),
                             "key_pos_cm": round(pe, 2), "key_rot_deg": None,
                             "pathint": round(float(V.pathint_loss(bl, w)[0]), 4),
                             "note": "zero temporal bits: just interpolate the endpoints"}
    rows["GT_key_joints"] = {"energy_ratio": 1.0, "key_pos_cm": 0.0, "key_rot_deg": 0.0,
                             "note": "the target itself"}
    # frame-code floor: GT key joints -> FROZEN spatial layer -> FULL BODY. This is the
    # ceiling on anything the temporal layer can deliver downstream, so it is always
    # printed next to the ladder, in full-body motion-energy terms.
    if tok is not None:
        spec = kd.spec[emb]
        sub = min(16, w.shape[0])
        rot, _ = V.complete_from_key(tok, ent, ent, w[:sub].reshape(-1, w.shape[2], 9),
                                     kd.kidx[emb], dev)
        rot = rot.reshape(sub, T, -1, 6)
        gt_full, _ = V.full_pose_of(kd.cache, emb,
                                    np.asarray(st[:sub])[:, None] + np.arange(T),
                                    dev, kd.reach[emb])
        gt_full = gt_full.reshape(sub, T, -1, 9)
        a = [V.motion_energy(V.fk_pos(rot[i], spec)) for i in range(sub)]
        b = [V.motion_energy(V.fk_pos(gt_full[i, ..., :6], spec)) for i in range(sub)]
        rows["frame_code_floor"] = {
            "energy_ratio": round(float(np.mean(a) / max(np.mean(b), 1e-6)), 3),
            "fullbody_energy_cm": round(float(np.mean(a)), 3),
            "gt_fullbody_energy_cm": round(float(np.mean(b)), 3),
            "note": "GT key joints -> FROZEN spatial layer -> full body: the CEILING "
                    "on any temporal layer (full-body motion energy, not key-joint)"}
    return rows


# --------------------------------------------------------------------------- figures
def figures(res, figdir):
    plt = V.fig_style()
    os.makedirs(figdir, exist_ok=True)
    runs = list(res)
    paths = []

    def ok(r, k):
        return isinstance(res[r].get(k), dict) and "error" not in res[r][k]

    # ---- speed invariance (the acceptance figure)
    rs = [r for r in runs if ok(r, "stat2_speed")]
    if rs:
        fig, ax = plt.subplots(figsize=(8.5, 4))
        x = np.arange(len(rs)); wdt = 0.26
        for i, (k, lbl) in enumerate((("2a_pure_resample", "2a pure resample (structural)"),
                                      ("2b_lloyd_recipe", "2b reference recipe"),
                                      ("control_unrelated", "control: unrelated motion"))):
            ax.bar(x + (i - 1) * wdt, [res[r]["stat2_speed"][k] for r in rs], wdt,
                   label=lbl, color=V.SERIES[i])
        ax.axhline(V.REF["path_speed_cos_accept"], color="#1baf7a", ls="--", lw=1.4)
        ax.axhline(V.REF["path_speed_cos_current"], color="#e34948", ls=":", lw=1.4)
        ax.text(len(rs) - .5, V.REF["path_speed_cos_accept"] + .01, "accept 0.95",
                ha="right", fontsize=8, color="#1baf7a")
        ax.text(len(rs) - .5, V.REF["path_speed_cos_current"] + .01,
                "current arch 0.644", ha="right", fontsize=8, color="#e34948")
        ax.set_xticks(x); ax.set_xticklabels([os.path.basename(r) for r in rs],
                                             rotation=15, fontsize=8)
        ax.set_ylabel("path-token cosine similarity"); ax.set_ylim(0, 1.05)
        ax.set_title("statistic 2 — is the PATH token invariant to speed?")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); p = f"{figdir}/v5_stat2_speed_invariance.png"
        fig.savefig(p); plt.close(fig); paths.append(p)

    # ---- prefix ladder
    rs = [r for r in runs if ok(r, "stat8_ladder")]
    if rs:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, r in enumerate(rs):
            L = res[r]["stat8_ladder"]
            ks = [k for k in L if k.startswith("prefix")]
            ks.sort(key=lambda s: int(s[6:]))
            xs = [int(k[6:]) for k in ks]
            ax[0].plot(xs, [L[k]["energy_ratio"] for k in ks], "o-",
                       color=V.SERIES[i % len(V.SERIES)], label=os.path.basename(r))
            ax[1].plot(xs, [L[k]["key_pos_cm"] for k in ks], "o-",
                       color=V.SERIES[i % len(V.SERIES)], label=os.path.basename(r))
            if i == 0:
                for a, key, lbl, col in ((ax[0], "energy_ratio", "straight line", "#898781"),
                                         (ax[1], "key_pos_cm", "straight line", "#898781")):
                    a.axhline(L["straight_line"][key], color=col, ls="--", lw=1.2)
        ax[0].axhline(1.0, color="#1baf7a", ls=":", lw=1.2)
        ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2)
        ax[0].set_xlabel("OAT prefix (path registers kept)")
        ax[1].set_xlabel("OAT prefix (path registers kept)")
        ax[0].set_ylabel("motion energy / GT"); ax[1].set_ylabel("key-joint error, cm")
        ax[0].set_title("statistic 8 — motion energy ladder\n(dashed = zero temporal bits)")
        ax[1].set_title("statistic 8 — key-joint error ladder")
        ax[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); p = f"{figdir}/v5_stat8_ladder.png"
        fig.savefig(p); plt.close(fig); paths.append(p)

    # ---- saturation + time distribution
    rs = [r for r in runs if ok(r, "stat4_saturation")]
    if rs:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, r in enumerate(rs):
            c = np.array(res[r]["stat4_saturation"]["curve"], float)
            ax[0].plot(c[:, 0], c[:, 1], "-", color=V.SERIES[i % len(V.SERIES)],
                       label=os.path.basename(r))
            if ok(r, "stat3_time"):
                h = res[r]["stat3_time"]["hist_counts"]
                ax[1].bar(np.arange(len(h)) + i * 0.3, np.array(h) / sum(h), 0.3,
                          color=V.SERIES[i % len(V.SERIES)], label=os.path.basename(r))
        lim = ax[0].get_xlim()
        ax[0].plot([0, lim[1]], [0, lim[1]], ":", color="#898781", lw=1)
        ax[0].text(lim[1], lim[1], " one code per window\n (no reuse)", fontsize=7,
                   color="#898781", va="top", ha="right")
        ax[0].set_xlabel("windows seen"); ax[0].set_ylabel("distinct path codes")
        ax[0].set_title("statistic 4 — vocabulary saturation")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].set_xlabel("time code, rank-ordered"); ax[1].set_ylabel("share of windows")
        ax[1].set_title("statistic 3 — do gears emerge?")
        ax[1].legend(frameon=False, fontsize=8)
        fig.tight_layout(); p = f"{figdir}/v5_stat34_vocab_time.png"
        fig.savefig(p); plt.close(fig); paths.append(p)

    # ---- spectrum + leakage
    rs = [r for r in runs if ok(r, "stat6_spectrum")]
    if rs:
        r = rs[0]
        S = res[r]["stat6_spectrum"]
        fams = [f for f in S if f != "ALL"] + ["ALL"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(fams, [S[f]["below_2.0Hz"] for f in fams], color=V.SERIES[0])
        ax[0].axhline(V.REF["spectral_cover_run"], color="#e34948", ls="--", lw=1.2)
        ax[0].text(len(fams) - .5, V.REF["spectral_cover_run"] + .01,
                   "run reference 0.665", ha="right", fontsize=8, color="#e34948")
        ax[0].set_ylabel("energy below 2 Hz"); ax[0].tick_params(axis="x", rotation=30)
        ax[0].set_title("statistic 6 — spectral coverage of the token bandwidth")
        lk = [r_ for r_ in runs if ok(r_, "stat7_leakage")]
        if lk:
            ax[1].bar([os.path.basename(x) for x in lk],
                      [res[x]["stat7_leakage"]["knn_body_acc"] for x in lk],
                      color=V.SERIES[2])
            ax[1].axhline(V.REF["knn_body_leak_p1"], color="#e34948", ls="--", lw=1.2)
            ax[1].axhline(res[lk[0]]["stat7_leakage"]["chance"], color="#898781",
                          ls=":", lw=1.2)
            ax[1].text(len(lk) - .5, V.REF["knn_body_leak_p1"] + .005, "P1 0.141",
                       ha="right", fontsize=8, color="#e34948")
            ax[1].set_ylabel("kNN body identifiability")
            ax[1].set_title("statistic 7 — cross-embodiment leakage\n(dotted = chance)")
            ax[1].tick_params(axis="x", rotation=15, labelsize=8)
        fig.tight_layout(); p = f"{figdir}/v5_stat67_spectrum_leakage.png"
        fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


# --------------------------------------------------------------------------- table
def write_table(res, probe_b, figs, path):
    L = []
    A = L.append
    runs = list(res)
    A("=" * 108)
    A("ARCH V5 TEMPORAL LAYER — VERDICT TABLE (spec ARCH_V5_TEMPORAL.md §7, all eight)")
    A(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A("=" * 108)
    A("")
    A("RUNS")
    for r in runs:
        c = res[r].get("config", {})
        A(f"  {os.path.basename(r):28s} arm={c.get('arm','?'):16s} "
          f"params={c.get('n_params',0)/1e6:.2f}M  "
          f"bits/s={c.get('bit_budget',{}).get('total_bits_per_s','?')}  "
          f"steps={c.get('steps','?')}")
    A("")

    A("-" * 108)
    A("2. PATH-TOKEN SPEED INVARIANCE   [acceptance > 0.95; current architecture 0.644; "
      "unrelated motion 0.246]")
    A(f"  {'run':28s} {'2a resample':>12s} {'2b REFERENCE':>13s} {'control':>9s} "
      f"{'verdict':>10s}")
    for r in runs:
        s = res[r].get("stat2_speed", {})
        if "error" in s or not s:
            A(f"  {os.path.basename(r):28s}  FAILED: {s.get('error','missing')}"); continue
        A(f"  {os.path.basename(r):28s} {s['2a_pure_resample']:12.3f} "
          f"{s['2b_lloyd_recipe']:13.3f} {s['control_unrelated']:9.3f} "
          f"{'PASS' if s['pass_2b'] else 'FAIL':>10s}")
    A("  2a is high by construction (the path readout never sees the frame count); "
      "2b is the real test.")
    A("")

    A("-" * 108)
    A("3. TIME-TOKEN DISTRIBUTION   [do gears emerge on their own, do they track speed]")
    A(f"  {'run':28s} {'distinct':>9s} {'bits':>6s} {'gears@80%':>10s} "
      f"{'top1 share':>11s} {'spearman vs speed':>18s}")
    for r in runs:
        s = res[r].get("stat3_time", {})
        if not s or "error" in s:
            A(f"  {os.path.basename(r):28s}  FAILED: {s.get('error','missing')}"); continue
        A(f"  {os.path.basename(r):28s} {s['distinct_time_codes']:9d} "
          f"{s['entropy_bits']:6.2f} {s['gears_covering_80pct']:10d} "
          f"{s['top5_share'][0]:11.3f} {s['spearman_code_dim0_vs_speed']:18.3f}")
    A("")

    A("-" * 108)
    A("4. VOCABULARY SATURATION   [decaying new-token rate = the vocabulary is reused]")
    A(f"  {'run':28s} {'distinct':>9s} {'rate first':>11s} {'rate last':>10s} "
      f"{'decay':>7s} {'log-log slope':>14s} {'saturating':>11s}")
    for r in runs:
        s = res[r].get("stat4_saturation", {})
        if not s or "error" in s:
            A(f"  {os.path.basename(r):28s}  FAILED: {s.get('error','missing')}"); continue
        A(f"  {os.path.basename(r):28s} {s['distinct_at_end']:9d} "
          f"{s['new_token_rate_first_quarter']:11.3f} "
          f"{s['new_token_rate_last_quarter']:10.3f} {s['decay_ratio']:7.3f} "
          f"{s['loglog_slope']:14.3f} {str(s['saturating']):>11s}")
    A("")

    A("-" * 108)
    A("1. CODEBOOK UTILIZATION (V5's own tokens)")
    A(f"  {'run':28s} {'path codes':>11s} {'repeat':>7s} {'path bits':>10s} "
      f"{'of max':>7s} {'time codes':>11s} {'time bits':>10s}")
    for r in runs:
        s = res[r].get("stat1_codebook", {})
        if not s or "error" in s:
            A(f"  {os.path.basename(r):28s}  FAILED: {s.get('error','missing')}"); continue
        A(f"  {os.path.basename(r):28s} {s['path_distinct_codes']:11d} "
          f"{s['path_repeat_rate']:7.3f} {s['path_entropy_bits']:10.2f} "
          f"{s['path_max_bits']:7.1f} {s['time_distinct_codes']:11d} "
          f"{s['time_entropy_bits']:10.2f}")
    A("")

    A("-" * 108)
    A("7. CROSS-EMBODIMENT LEAKAGE   [P1 = 0.141, chance = 0.167; must not regress]")
    A(f"  {'run':28s} {'kNN body acc':>13s} {'chance':>7s} {'vs P1':>7s} {'verdict':>9s}")
    for r in runs:
        s = res[r].get("stat7_leakage", {})
        if not s or "error" in s:
            A(f"  {os.path.basename(r):28s}  FAILED: {s.get('error','missing')}"); continue
        A(f"  {os.path.basename(r):28s} {s['knn_body_acc']:13.3f} {s['chance']:7.3f} "
          f"{s['knn_body_acc']-s['p1_reference']:+7.3f} "
          f"{'OK' if s['pass_no_regression'] else 'REGRESSED':>9s}")
    A("")

    A("-" * 108)
    A("8. RECONSTRUCTION LADDER   [motion energy retention / key-joint error by OAT prefix]")
    for r in runs:
        s = res[r].get("stat8_ladder", {})
        if not s or "error" in s:
            A(f"  {os.path.basename(r):28s}  FAILED: {s.get('error','missing')}"); continue
        A(f"  {os.path.basename(r)}")
        A(f"    {'row':18s} {'energy/GT':>10s} {'key pos cm':>11s} {'key rot deg':>12s} "
          f"{'pathint':>9s}")
        order = [k for k in s if k.startswith("prefix")]
        order.sort(key=lambda x: int(x[6:]))
        for k in order + ["tokens_shuffled", "tokens_zeroed", "straight_line",
                          "GT_key_joints", "frame_code_floor"]:
            if k not in s:
                continue
            v = s[k]
            A(f"    {k:18s} {v.get('energy_ratio', float('nan')):10.3f} "
              f"{(v.get('key_pos_cm') if v.get('key_pos_cm') is not None else float('nan')):11.2f} "
              f"{(v.get('key_rot_deg') if v.get('key_rot_deg') is not None else float('nan')):12.1f} "
              f"{(v.get('pathint') if v.get('pathint') is not None else float('nan')):9.4f}"
              + (f"   [{v['note']}]" if "note" in v else ""))
        if "frame_code_floor" in s:
            A(f"    frame-code floor full-body energy: "
              f"{s['frame_code_floor']['fullbody_energy_cm']} cm "
              f"(ceiling on any temporal layer)")
        tc = s.get("_token_content")
        if tc and tc.get("content_share") is not None:
            share = tc["content_share"]
            A(f"    >> IS THE CODEBOOK REAL? the path registers buy "
              f"{tc['registers_buy_cm']:.2f} cm over having none, but only "
              f"{tc['window_specific_cm']:.2f} cm of that depends on WHICH window's "
              f"token was used")
            A(f"    >> content share = {share:.2f}  "
              + ("<-- the tokens are largely a generic 'not zero' signal; the codebook "
                 "is close to DECORATIVE and the headline error is mostly the endpoint "
                 "conditioning" if share < 0.35 else
                 ("partial: a real but minority share of the benefit is window-specific"
                  if share < 0.7 else
                  "the bits really are carrying this window's motion")))
        A("")

    A("-" * 108)
    A("6. SPECTRAL COVERAGE   [energy inside the token bandwidth; run reference 0.665]")
    s = next((res[r].get("stat6_spectrum") for r in runs
              if isinstance(res[r].get("stat6_spectrum"), dict)), None)
    if s:
        A(f"  {'family':10s} {'<1Hz':>7s} {'<2Hz':>7s} {'<4Hz':>7s} {'>2Hz':>7s} {'n':>5s}")
        for f_, v in s.items():
            A(f"  {f_:10s} {v['below_1.0Hz']:7.3f} {v['below_2.0Hz']:7.3f} "
              f"{v['below_4.0Hz']:7.3f} {v['above_2Hz']:7.3f} {v['n']:5d}")
    else:
        A("  FAILED / missing")
    A("")

    A("-" * 108)
    A("5. INTRINSIC DIMENSION   [from v5_probe_entropy.py; R8 estimators]")
    d = (probe_b or {}).get("stat5_intrinsic_dimension", {}).get("by_family", {})
    if d:
        A(f"  {'family':10s} {'key6 TwoNN':>11s} {'full TwoNN':>11s} {'ratio':>7s} "
          f"{'key MLE10':>10s} {'full MLE10':>11s}")
        for f_, v in d.items():
            A(f"  {f_:10s} {v['key6']['twonn']:11.2f} {v['full_body']['twonn']:11.2f} "
              f"{v['ratio_key_over_full_twonn']:7.2f} {v['key6']['mle_k10']:10.2f} "
              f"{v['full_body']['mle_k10']:11.2f}")
    else:
        A("  not available — run v5_probe_entropy.py first")
    A("")

    A("-" * 108)
    A("1b. SPATIAL-CODE RECURRENCE INSIDE WINDOWS   [from v5_probe_entropy.py]")
    s1 = (probe_b or {}).get("stat1_code_utilization")
    if s1:
        A(f"  one-off codes {s1['frac_codes_one_off']*100:.1f}% | recurrent (>= T/4) "
          f"{s1['frac_codes_recurrent_ge_quarter_window']*100:.1f}% | "
          f"novelty~speed r={s1['corr_novelty_speed']:+.2f} | "
          f"novelty~|accel| r={s1['corr_novelty_accel']:+.2f}")
        if s1.get("turning_point_peak_over_mean") is not None:
            A(f"  novelty at motion turning points: "
              f"{s1['turning_point_peak_over_mean']:.2f}x the window mean")
    else:
        A("  not available — run v5_probe_entropy.py first")
    A("")

    A("-" * 108)
    A("FIGURES")
    for p in figs:
        A(f"  {p}")
    A("=" * 108)
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(txt + "\n")
    return txt


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of V5 run dirs")
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--emb", default="human_smpl")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--n_knn", type=int, default=250)
    ap.add_argument("--n_sat", type=int, default=4000)
    ap.add_argument("--n_spec", type=int, default=120)
    ap.add_argument("--no_floor", action="store_true", help="skip the frozen-layer row")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=V.VERDICT)
    ap.add_argument("--figs", default=V.FIGS)
    V.add_root_vel_flag(ap)
    args = ap.parse_args()
    if args.tiny:
        args.n, args.n_knn, args.n_sat, args.n_spec = 48, 24, 120, 16
        args.device = "cpu"
    t0, dev = time.time(), args.device
    runs = [r.rstrip("/") for r in args.runs.split(",") if r]
    runs = [r for r in runs if os.path.exists(f"{r}/model.pt")]
    if not runs:
        raise SystemExit(f"no run in --runs has a model.pt: {args.runs}")

    embs = list(dict.fromkeys([args.emb] + R7_BODIES))
    kd = V.KeyData(args.cache, embs, device=dev, root_vel=args.root_vel)
    tok = ent = None
    if not args.no_floor:
        try:
            tok, _ = V.load_frozen(dev)
            ent = V.body_entry(args.cache, args.emb, dev, kd.reach[args.emb])
        except Exception as e:
            print(f"WARN frozen spatial layer unavailable: {e}", flush=True)

    res = {}
    for r in runs:
        print(f"\n===== {r}", flush=True)
        model, cfg = load_run(r, dev)
        T, span = cfg.get("T", V.T_WIN), cfg.get("span", cfg.get("T", V.T_WIN))
        out = {"config": cfg}
        jobs = [
            ("stat1_codebook", lambda: stat1_codebook(model, kd, args.emb, dev,
                                                      args.n, T, span)),
            ("stat2_speed", lambda: stat2_speed(model, kd, args.emb, dev,
                                                min(args.n, 512), T, span)),
            ("stat3_time", lambda: stat3_time(model, kd, args.emb, dev,
                                              min(args.n, 512), T, span)),
            ("stat4_saturation", lambda: stat4_saturation(model, kd, args.emb, dev,
                                                          args.n_sat, T, span)),
            ("stat6_spectrum", lambda: stat6_spectrum(kd, args.emb, T, span,
                                                      args.n_spec, V.V5_FAMILIES)),
            ("stat7_leakage", lambda: stat7_leakage(model, kd, dev, args.n_knn, T, span)),
            ("stat8_ladder", lambda: stat8_ladder(model, kd, args.emb, dev,
                                                  min(args.n, 256), T, span, tok, ent)),
        ]
        for name, fn in jobs:
            try:
                out[name] = fn()
                print(f"  {name}: ok", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                out[name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {name}: FAILED {e}", flush=True)
        json.dump(out, open(f"{r}/eval_v5.json", "w"), indent=1, default=float)
        res[r] = out

    probe_b = None
    pb = f"{V.RUNS}/V5_v0b_entropy.json"
    if os.path.exists(pb):
        try:
            probe_b = json.load(open(pb))
        except Exception:
            pass

    figs = []
    try:
        figs = figures(res, args.figs)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FIGURES FAILED: {e}", flush=True)
    txt = write_table(res, probe_b, figs, args.out)
    print("\n" + txt)
    print(f"\nSAVED {args.out}  ({time.time()-t0:.0f}s)", flush=True)
    for p in figs:
        print("FIG:", p)


if __name__ == "__main__":
    main()
