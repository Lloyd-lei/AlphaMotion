"""RAINBOW CODES — share-level spectrum of the token vocabulary.

    rainbow_codes.py

lloyd's definition: a code carried by more than two shared paths is a RAINBOW
code.  Measured at two levels, because "connectivity" means two different
things:

  WITHIN-BRIDGE   among K samples of ONE endpoint pair, how many paths carry
                  the same (position, code)?  High share level = a hub in that
                  bridge's path graph.
  CROSS-BRIDGE    across DIFFERENT endpoint pairs, is the same code reused?
                  This is the one that decides the data question: if bridges
                  already share a vocabulary, more data enriches a common
                  lattice; if every bridge uses private codes, more data just
                  adds more private codes.

Plus the decisive containment check: what fraction of generated codes already
occurs in the CORPUS vocabulary (real motion tokenised by the same A3)?  All
inside = the generator interpolates within the observed code set; genuinely new
codes = it leaves it.

Figures: png/rainbow_codes.png
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
from bridge_a import CodeData, load_bridge_a                         # noqa: E402
from bridge_b import BridgeB, tokens_of                              # noqa: E402
from homecoming_probe import sample_nll                              # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
FAMS = ["walk", "run", "kick", "punch", "dance", "sit", "jump", "jacks"]
V5 = 5 ** 6


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--n_bridge", type=int, default=14)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n_corpus", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, N = args.device, args.n

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    # ---------- corpus vocabulary ----------
    g = np.random.default_rng(0)
    pool = [(a, b) for spl in ("train", "val") for a, b in data.pools[spl]
            if b - a >= N + 2]
    cst = [int(g.integers(a, b - N)) for a, b in
           (pool[i] for i in g.choice(len(pool), args.n_corpus, replace=True))]
    corpus_pos, corpus_any = set(), set()
    corpus_counts = Counter()
    for i in range(0, len(cst), 64):
        blk = cst[i:i + 64]
        c = torch.stack([codes_at(s, N) for s in blk]).to(dev)
        tok, _ = tokens_of(ma, c)
        t = tok.cpu().numpy()
        for row in t:
            for pos, cd in enumerate(row):
                corpus_pos.add((pos, int(cd))); corpus_any.add(int(cd))
                corpus_counts[(pos, int(cd))] += 1
    print(f"[corpus] {len(cst)} windows -> {len(corpus_any)} distinct codes "
          f"({100*len(corpus_any)/V5:.2f}% of the {V5} lattice), "
          f"{len(corpus_pos)} distinct (pos,code)", flush=True)

    # ---------- generated paths ----------
    gg = np.random.default_rng(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    per_bridge, bridge_names = [], []
    for bi in range(args.n_bridge):
        fa, fb = rng.choice(FAMS, 2, replace=False)
        try:
            sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[fa])
            sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[fb])
        except Exception:
            continue
        ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
        ep = ma.endpoints(torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev))
        nf = torch.full((1,), float(N), device=dev)
        toks = []
        for s_ in range(args.K):
            tk, _ = sample_nll(mb, ep, nf, dev, temp=args.tau,
                               seed=bi * 1000 + s_)
            toks.append(tk.cpu().numpy())
        per_bridge.append(np.stack(toks))                 # [K,32]
        bridge_names.append(f"{fa}->{fb}")
        print(f"[bridge {bi}] {fa}->{fb}", flush=True)

    # ---------- within-bridge share spectrum ----------
    share_hist = Counter()
    rainbow_frac = []
    for T in per_bridge:
        cnt = Counter()
        for row in T:
            for pos, cd in enumerate(row):
                cnt[(pos, int(cd))] += 1
        for v in cnt.values():
            share_hist[v] += 1
        tot = sum(cnt.values())
        rb = sum(v for v in cnt.values() if v >= 3)
        rainbow_frac.append(rb / tot)
    print(f"\n[within-bridge] rainbow (share>=3) slot fraction: "
          f"{np.mean(rainbow_frac)*100:.1f}% +- {np.std(rainbow_frac)*100:.1f}")

    # ---------- cross-bridge sharing ----------
    seen_in = defaultdict(set)
    for bi, T in enumerate(per_bridge):
        for row in T:
            for pos, cd in enumerate(row):
                seen_in[(pos, int(cd))].add(bi)
    xb = Counter(len(v) for v in seen_in.values())
    nb = len(per_bridge)
    multi = sum(c for k, c in xb.items() if k >= 3)
    print(f"[cross-bridge] (pos,code) seen in >=3 of {nb} bridges: "
          f"{multi}/{len(seen_in)} = {100*multi/len(seen_in):.1f}%")
    # code-level (ignoring position)
    seen_code = defaultdict(set)
    for bi, T in enumerate(per_bridge):
        for row in T:
            for cd in row:
                seen_code[int(cd)].add(bi)
    xc = Counter(len(v) for v in seen_code.values())
    multic = sum(c for k, c in xc.items() if k >= 3)
    print(f"[cross-bridge] CODE (any position) in >=3 bridges: "
          f"{multic}/{len(seen_code)} = {100*multic/len(seen_code):.1f}%")

    # ---------- containment in the corpus vocabulary ----------
    gen_any = set()
    inv_pos = inv_any = tot_slots = 0
    for T in per_bridge:
        for row in T:
            for pos, cd in enumerate(row):
                gen_any.add(int(cd)); tot_slots += 1
                inv_pos += (pos, int(cd)) in corpus_pos
                inv_any += int(cd) in corpus_any
    print(f"[containment] generated slots whose (pos,code) occurs in corpus: "
          f"{100*inv_pos/tot_slots:.1f}%")
    print(f"[containment] generated slots whose CODE occurs anywhere in corpus: "
          f"{100*inv_any/tot_slots:.1f}%")
    print(f"[vocab] generated distinct codes {len(gen_any)}, corpus "
          f"{len(corpus_any)}, novel {len(gen_any - corpus_any)}")

    # ---------- figures ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ks = sorted(share_hist)
    ax[0].loglog(ks, [share_hist[k] for k in ks], "o-", color="#c33")
    ax[0].axvline(3, color="k", ls="--", lw=1)
    ax[0].text(3.1, max(share_hist.values()) * .5, "rainbow threshold", fontsize=8)
    ax[0].set_xlabel(f"share level  (how many of {args.K} paths carry this "
                     "(pos,code))")
    ax[0].set_ylabel("number of (pos,code) entries")
    ax[0].set_title("WITHIN-BRIDGE share spectrum\n"
                    f"rainbow slot fraction {np.mean(rainbow_frac)*100:.0f}%")
    ax[0].grid(alpha=0.3, which="both")
    kk = sorted(xb)
    ax[1].bar([k for k in kk], [xb[k] for k in kk], color="#468",
              label="(pos,code)")
    kk2 = sorted(xc)
    ax[1].bar([k + .0 for k in kk2], [xc[k] for k in kk2], color="#c83",
              alpha=.55, label="code only")
    ax[1].set_yscale("log")
    ax[1].axvline(2.5, color="k", ls="--", lw=1)
    ax[1].set_xlabel(f"number of the {nb} bridges that use it")
    ax[1].set_ylabel("count (log)")
    ax[1].set_title("CROSS-BRIDGE reuse\n"
                    f"code-level >=3 bridges: {100*multic/len(seen_code):.0f}%")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    ax[2].bar(["(pos,code)\nin corpus", "code\nin corpus", "novel codes"],
              [100 * inv_pos / tot_slots, 100 * inv_any / tot_slots,
               100 * len(gen_any - corpus_any) / max(len(gen_any), 1)],
              color=["#3a6", "#38c", "#c33"])
    ax[2].set_ylabel("% ")
    ax[2].set_title("containment in the corpus vocabulary\n"
                    f"corpus covers {100*len(corpus_any)/V5:.1f}% of the lattice")
    ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("RAINBOW CODES — share-level spectrum of the token vocabulary",
                 fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/rainbow_codes.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"rainbow_frac": float(np.mean(rainbow_frac)),
               "within_share_hist": {str(k): v for k, v in share_hist.items()},
               "cross_bridge_poscode": {str(k): v for k, v in xb.items()},
               "cross_bridge_code": {str(k): v for k, v in xc.items()},
               "inv_pos_pct": 100 * inv_pos / tot_slots,
               "inv_any_pct": 100 * inv_any / tot_slots,
               "corpus_codes": len(corpus_any), "gen_codes": len(gen_any),
               "novel_codes": len(gen_any - corpus_any)},
              open(f"{RUNS}/RAINBOW_CODES_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
