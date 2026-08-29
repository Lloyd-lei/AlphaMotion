"""THE ARCHITECTURE DECIDER — JEPA vs MSBA under PARTIAL OBSERVATION.

    exam_partial.py --runs V7_jepa_0807,V7_maskfsq_0807,V8_arclen_0807

lloyd, 2026-08-08, correcting me: "jepa只能学会往后预测动作，但是masked
self-bidirectional attn应该能够补全未来和过去的动作". The first half of that is
something I said and it was WRONG — grep for causal/triu/tril in train_v7.py returns
nothing. NEITHER arm has any causal masking; both are fully bidirectional. So
"forward-only vs both-directions" is not the axis that separates them.

The axis that DOES separate them is CODE LOCALITY under partial observation:

  jepa    8 unordered GLOBAL latents, each a Perceiver query cross-attending to the
          whole window. Hide half the frames and every latent is computed from a
          corrupted set — the damage is global.
  MSBA    codes are tied to a contiguous slice (of the clock for maskfsq, of the PATH
          for arclen). Hide half the frames and the codes over the OBSERVED half
          should still be valid; only the hidden slices' codes degrade.

If that story is true, MSBA degrades gracefully and jepa falls off a cliff, and that
is a functional reason to pick MSBA that has nothing to do with reconstruction error.
If it is false, the two are interchangeable and the choice should be made on the
speed-token axis instead.

FOUR OBSERVATION PATTERNS (what the ENCODER sees; the decoder's endpoint
conditioning is left alone, because that is the deployed interface):
  prefix   first k frames      -> "I have the past, complete the future"
  suffix   last  k frames      -> "I have the future, complete the past"
  middle   central k frames    -> complete both ends
  sparse   k random frames     -> the keyframe case

Scored ONLY on the frames the encoder never saw. Baselines on every row:
  full     the same arm with nothing hidden (its own ceiling)
  line     the straight line between the endpoints (zero tokens)

CONFOUND, stated up front: maskfsq and arclen TRAIN with encoder frame masking
(enc_mix rand/sparse/prefix, warm-up 6k); jepa never does. That is intrinsic to the
designs rather than a tuning choice — masking is what makes a local code stream
learnable — but it means this exam measures design+training together, not
architecture in a vacuum.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                             # noqa: E402
from train_v5 import load_run, to_states                          # noqa: E402

EMB = "human_smpl"
FAMS = ("kick", "walk", "dance")
KS = [8, 15, 30, 45]


def pattern_mask(kind, k, T, rng, dev):
    """-> bool [T]  True = HIDDEN from the encoder."""
    m = np.ones(T, bool)
    if kind == "prefix":
        m[:k] = False
    elif kind == "suffix":
        m[T - k:] = False
    elif kind == "middle":
        s = (T - k) // 2
        m[s:s + k] = False
    elif kind == "sparse":
        m[rng.choice(T, size=k, replace=False)] = False
    return torch.as_tensor(m, device=dev)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807,V8_arclen_0807")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    reach = kd.reach[EMB]
    models = {a.replace("_0807", ""): load_run(f"{V.RUNS}/{a}", dev)[0]
              for a in args.runs.split(",") if a}
    rows = {}

    def cm(x, y, sel):
        return float((x[:, sel, :, 6:9] - y[:, sel, :, 6:9]).norm(dim=-1).mean() * reach)

    for fam in FAMS:
        g = np.random.default_rng(808)
        st, _ = kd.sample_starts(args.n, g, "val", span=T, families=[fam])
        w = kd.window(EMB, st, T, T).to(dev).float()
        kn = torch.ones(w.shape[0], device=dev)
        line = next(iter(models.values())).baseline(w[:, 0], w[:, -1], kn, T).float()

        for nm, m in models.items():
            e_full = m.encode(w, w[:, 0], w[:, -1], kn)
            full = to_states(m.decode(e_full["path_z"], e_full["time_z"],
                                      w[:, 0], w[:, -1], kn, T),
                             m, w[:, 0], w[:, -1], kn).float()
            for kind in ("prefix", "suffix", "middle", "sparse"):
                for k in KS:
                    hid = pattern_mask(kind, k, T, np.random.default_rng(k), dev)
                    em = hid[None].expand(w.shape[0], -1)
                    e = m.encode(w, w[:, 0], w[:, -1], kn, enc_mask=em)
                    o = to_states(m.decode(e["path_z"], e["time_z"],
                                           w[:, 0], w[:, -1], kn, T),
                                  m, w[:, 0], w[:, -1], kn).float()
                    sel = hid.cpu().numpy()          # score ONLY the unseen frames
                    rows[f"{fam}|{kind}|seen={T-int(hid.sum())}|{nm}"] = {
                        "unseen_cm": round(cm(o, w, sel), 2),
                        "own_ceiling_cm": round(cm(full, w, sel), 2),
                        "straight_line_cm": round(cm(line, w, sel), 2),
                        "degradation_vs_own": round(cm(o, w, sel) - cm(full, w, sel), 2)}

    # ---- the summary that actually decides it ---------------------------------
    L = ["=" * 104,
         "PARTIAL OBSERVATION — the JEPA vs MSBA decider",
         "=" * 104,
         "degradation = error on UNSEEN frames minus that arm's own no-masking error.",
         "Lower = the code survives partial observation. This is the number to compare;",
         "absolute error still reflects each arm's baseline quality.",
         "CONFOUND: maskfsq/arclen train with encoder masking, jepa does not.",
         "-" * 104]
    arms = list(models)
    for kind in ("prefix", "suffix", "middle", "sparse"):
        L.append(f"\n--- {kind} ---")
        L.append(f"{'seen frames':<14}" + "".join(f"{a:>26}" for a in arms))
        for k in KS:
            cells = []
            for a in arms:
                ds = [rows[f"{f}|{kind}|seen={k}|{a}"]["degradation_vs_own"]
                      for f in FAMS if f"{f}|{kind}|seen={k}|{a}" in rows]
                us = [rows[f"{f}|{kind}|seen={k}|{a}"]["unseen_cm"]
                      for f in FAMS if f"{f}|{kind}|seen={k}|{a}" in rows]
                cells.append(f"{np.mean(us):>17.2f} (+{np.mean(ds):.2f})")
            L.append(f"{k:<14}" + "".join(f"{c:>26}" for c in cells))
    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{V.RUNS}/PARTIAL_verdict.txt", "w").write(txt + "\n")
    json.dump(rows, open(f"{V.RUNS}/PARTIAL_verdict.json", "w"), indent=1)
    print(f"\nverdict: {V.RUNS}/PARTIAL_verdict.txt", flush=True)


if __name__ == "__main__":
    main()
