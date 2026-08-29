"""EXAM B — the UPSTREAM exam: is the V7 codebook a vocabulary a generator can learn?

    exam_b.py --runs V7_jepa_0807,V7_maskfsq_0807

No rendering here on purpose — this exam is about the tokens themselves, the one
place numbers ARE the product. Everything a future AR generator cares about:

  (1) USAGE      how much of the 125-bit budget is actually spent (per-digit entropy
                 vs capacity; distinct register codes vs the 15625-word vocabulary)
  (2) STRUCTURE  is the next token predictable from the previous one? Adjacent-
                 register mutual information + conditional digit entropy. High MI =
                 an AR model has something to learn; zero MI = the stream is white
                 noise and AR generation is hopeless; MI ~ H = degenerate copying.
                 (No small-LM training — lloyd already ruled it redundant, R9:
                 conditional entropy gives the same information in minutes.)
  (3) SATURATION new-code rate vs windows seen. Decaying curve = the vocabulary is
                 REUSED across motions (a real language); flat = every clip mints
                 fresh codes (a hash).
  (4) SEMANTICS  kNN family accuracy in token space (exam_c m5 showed retrieval
                 works; this is the same property as one number over many windows).

Caveat printed with the results: maskfsq's registers are TIME-LOCAL (group k = a
fixed ~0.22 s slice), so its "adjacent register" axis is literally time — the AR
reading is direct. jepa's 8 path registers are unordered; for them MI between
registers measures redundancy, not temporal structure, and the honest AR axis is
(path register | time register). Both are reported.
"""
from __future__ import annotations
import os, sys, json, argparse, math
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import v5_common as V                                             # noqa: E402
from train_v5 import load_run                                     # noqa: E402

EMB = "human_smpl"


def H(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def digit_entropies(D, base=5):
    """D [N, R, 6] ints -> per-register summed digit entropy [R] (bits)."""
    N, R, K = D.shape
    out = np.zeros(R)
    for r in range(R):
        for k in range(K):
            out[r] += H(np.bincount(D[:, r, k], minlength=base))
    return out


def pair_mi(a, b, base=5):
    """Mean mutual information between corresponding digits of two registers."""
    K = a.shape[1]
    mi = 0.0
    for k in range(K):
        j = np.zeros((base, base))
        np.add.at(j, (a[:, k], b[:, k]), 1)
        mi += H(j.sum(1)) + H(j.sum(0)) - H(j.reshape(-1))
    return mi / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    T = V.T_WIN
    rows = {}

    @torch.no_grad()
    def tokenize(m, split, n, seed):
        g = np.random.default_rng(seed)
        st, cids = kd.sample_starts(n, g, split, span=T)
        fams = [kd.family[int(c)] for c in cids]
        out = []
        for i in range(0, n, 256):
            w = kd.window(EMB, st[i:i + 256], T, T).to(dev).float()
            kn = torch.ones(w.shape[0], device=dev)
            e = m.encode(w, w[:, 0], w[:, -1], kn)
            pi, ti = m.token_ints(e)
            t = pi if ti.numel() == 0 else torch.cat(
                [pi, ti.reshape(ti.shape[0], -1, ti.shape[-1] if ti.dim() > 2
                                else ti.shape[-1])], 1) if ti.dim() == pi.dim() else pi
            out.append(pi.cpu())
        return torch.cat(out).numpy(), fams

    for run in [r for r in args.runs.split(",") if r]:
        nm = run.replace("_0807", "")
        m = load_run(f"{V.RUNS}/{run}", dev)[0]
        Dtr, Ftr = tokenize(m, "train", args.n_train, 11)
        Dva, Fva = tokenize(m, "val", args.n_val, 12)
        N, R, K = Dtr.shape
        cap = K * math.log2(5)                                  # 13.93 bit/register

        # (1) usage
        ent = digit_entropies(Dtr)
        tuples = [set(map(tuple, Dtr[:, r])) for r in range(R)]
        distinct = [len(s) for s in tuples]

        # (2) structure: adjacent-register MI + conditional entropy
        mi_adj = [pair_mi(Dtr[:, r], Dtr[:, r + 1]) for r in range(R - 1)]
        # unconditional per-digit entropy of register r+1, for the "of which" line
        h1 = [digit_entropies(Dtr[:, r + 1:r + 2])[0] / K for r in range(R - 1)]

        # (3) saturation: new register-tuples per 1000 windows, first vs last chunk
        def new_rate(D, chunk=1000):
            seen, rates = set(), []
            for i in range(0, len(D), chunk):
                block = set(map(tuple, D[i:i + chunk].reshape(-1, K)))
                fresh = len(block - seen)
                rates.append(fresh / max(len(block), 1))
                seen |= block
            return rates
        sat = new_rate(Dtr.reshape(N, R, K).transpose(0, 1, 2).reshape(N, R, K)
                       .reshape(-1, R, K)[:, 0] if False else Dtr[:, 0])
        sat_all = new_rate(Dtr.reshape(-1, K))                  # all registers pooled

        # (4) semantics: 5-NN family vote in Hamming space, val queried against train.
        # BANK RESTRICTED TO THE 6 NAMED FAMILIES — a uniform train bank is 77%
        # "other" (AMASS skew) and the majority vote degenerates to always-other;
        # the first run of this exam scored 0.15 for exactly that reason.
        FAM6 = ("walk", "run", "jacks", "kick", "dance", "punch")
        bidx = [i for i, f in enumerate(Ftr) if f in FAM6]
        b = Dtr.reshape(N, -1)[bidx]
        bf = [Ftr[i] for i in bidx]
        q = Dva.reshape(len(Dva), -1)
        hits = 0
        idx = [i for i, f in enumerate(Fva) if f in FAM6][:400]
        for i in idx:
            d = (b != q[i][None]).mean(1)
            top = np.argsort(d)[:5]
            vote = [bf[t] for t in top]
            hits += max(set(vote), key=vote.count) == Fva[i]
        knn = hits / max(len(idx), 1)
        base = max(np.bincount([FAM6.index(f) for f in bf])) / max(len(bf), 1)

        rows[nm] = {
            "bits_used_per_register": [round(e, 2) for e in ent],
            "bits_capacity_per_register": round(cap, 2),
            "bits_used_total": round(float(ent.sum()), 1),
            "bits_capacity_total": round(cap * R, 1),
            "distinct_codes_per_register (vocab 15625)": distinct,
            "adjacent_register_MI_bits_per_digit": [round(x, 3) for x in mi_adj],
            "next_register_digit_entropy": [round(x, 3) for x in h1],
            "new_code_rate_first_vs_last_1k": [round(sat_all[0], 3),
                                               round(sat_all[-1], 3)],
            "knn5_family_acc (bank = 6 named families only)": round(knn, 3),
            "majority_class_baseline": round(float(base), 3),
            "n_train_windows": N}
        del m; torch.cuda.empty_cache()

    hdr = ["=" * 100,
           "EXAM B — codebook statistics for the future generator",
           "READ ME FIRST: maskfsq registers are time slices -> adjacent-register MI",
           "IS temporal predictability. jepa registers are unordered -> its MI is",
           "redundancy between codes, not an AR axis.",
           "=" * 100]
    body = [f"\n[{k}]\n" + json.dumps(v, indent=2) for k, v in rows.items()]
    txt = "\n".join(hdr + body)
    print(txt, flush=True)
    open(f"{V.RUNS}/EXAMB_verdict.txt", "w").write(txt + "\n")
    json.dump(rows, open(f"{V.RUNS}/EXAMB_verdict.json", "w"), indent=1)
    print(f"\nverdict: {V.RUNS}/EXAMB_verdict.txt", flush=True)


if __name__ == "__main__":
    main()
