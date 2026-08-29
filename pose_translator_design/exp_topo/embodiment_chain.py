"""EMBODIMENT CHAIN — does the code survive being passed body to body?

    embodiment_chain.py

lloyd's test: human -> robot1 -> robot2 -> ... , each hop going back out to
OBSERVATION space and being re-encoded.  If the representation is genuinely
embodiment-invariant, every hop should emit the SAME tokens.  Whatever decay we
measure is the honest amount of invariance the stack actually has, and it needs
no ground truth at any step — every hop is compared with hop 0.

Measured per hop:
  code agree      fraction of the 256x20 integer levels identical to hop 0
  level drift     mean |level - level_hop0| on the same lattice
  token agree     fraction of the 32 Equator tokens identical to hop 0
  prior NLL       B3's masked CE on that hop's tokens (does the temporal prior
                  notice the degradation?)
Controls, both required to read the numbers:
  CEILING  re-encoding the SAME pose without translating (should be 1.00)
  FLOOR    a different random motion's codes (what chance agreement looks like)

Figure: png/embodiment_chain.png
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
from dual_common import load_paired_data, split_indices, build_model, geom  # noqa: E402
from bridge_a import load_bridge_a                                   # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--chain", default="human_smpl,booster_t1,unitree_h1,"
                                       "fourier_gr3,human_smpl,booster_t1")
    ap.add_argument("--n_win", type=int, default=24)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, N = args.device, args.n

    cfg = json.load(open(f"{RUNS}/{args.spatial}/config.json"))
    torch.manual_seed(0)
    data, names, NR = load_paired_data(cfg["cache"], dev,
                                       sem_encoder=cfg["sem_encoder"])
    val_idx, _ = split_indices(NR)
    sp = build_model(cfg, dev)
    sp.load_state_dict(torch.load(f"{RUNS}/{args.spatial}/model.pt",
                                  map_location=dev)); sp.eval()
    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt",
                                  map_location=dev)); mb.eval()

    chain = args.chain.split(",")
    for c in chain:
        assert c in names, f"{c} not in {names}"
    W = 5 ** torch.arange(6, device=dev)

    def ints_of(p9, name):
        g = geom(data, name, p9.shape[0])
        zp, zr = sp.encode(p9, g)
        ip = sp.fsqp.ints(sp.fsqp.features(zp))[0]
        ir = sp.fsqr.ints(sp.fsqr.features(zr))[0]
        return torch.cat([ip, ir], 1), (zp, zr)

    def hop(zp, zr, name):
        g = geom(data, name, zp.shape[0])
        prot, ppos = sp.decode(zp, zr, g)
        return torch.cat([prot, ppos], -1)

    def tokens_nll(c_ints):
        """[N,256,20] ints -> A3 tokens + B3 masked CE."""
        c = c_ints[None]
        h = ma.embed_frames(c)
        B = h.shape[0]
        seq = torch.cat([ma.q[None].expand(B, -1, -1), h], 1)
        mask = None
        if ma.time_local:
            K, T = ma.n_content, h.shape[1]
            L = K + T
            mask = torch.zeros(L, L, dtype=torch.bool, device=dev)
            b = np.round(np.linspace(0, T, K + 1)).astype(int)
            for i in range(K):
                mask[i, K:] = True
                mask[i, K + b[i]:K + max(b[i + 1], b[i] + 1)] = False
        out = ma.enc(seq, mask=mask)[:, :ma.n_content]
        ints = ma.fsq.ints(ma.fsq.features(out))[0]
        tok = (ints * W).sum(-1)
        ep = ma.endpoints(c)
        nf = torch.full((1,), float(c.shape[1]), device=dev)
        mk = torch.ones_like(tok, dtype=torch.bool)
        ce = float(F.cross_entropy(mb(tok, mk, ep, nf)[mk], tok[mk]))
        return tok[0], ce

    g = np.random.default_rng(3)
    starts = g.choice(len(val_idx) - N, args.n_win, replace=False)
    rows = []
    for w, s0 in enumerate(starts):
        idx = val_idx[int(s0):int(s0) + N]
        p9 = data[chain[0]]["pose"][idx]
        c0, (zp, zr) = ints_of(p9, chain[0])
        tok0, nll0 = tokens_nll(c0)
        rec = [{"hop": 0, "emb": chain[0], "code": 1.0, "drift": 0.0,
                "tok": 1.0, "nll": nll0}]
        # ceiling control: re-encode the same pose, no translation
        c_re, _ = ints_of(p9, chain[0])
        ceil = float((c_re == c0).float().mean())
        # floor control: a different motion
        idx2 = val_idx[int(g.integers(0, len(val_idx) - N)):][:N]
        cF, _ = ints_of(data[chain[0]]["pose"][idx2], chain[0])
        floor = float((cF == c0).float().mean())
        cur_zp, cur_zr = zp, zr
        for k in range(1, len(chain)):
            p9n = hop(cur_zp, cur_zr, chain[k])
            ck, (cur_zp, cur_zr) = ints_of(p9n, chain[k])
            tokk, nllk = tokens_nll(ck)
            rec.append({"hop": k, "emb": chain[k],
                        "code": float((ck == c0).float().mean()),
                        "drift": float((ck - c0).abs().float().mean()),
                        "tok": float((tokk == tok0).float().mean()),
                        "nll": nllk})
        rows.append({"rec": rec, "ceil": ceil, "floor": floor})
        if w == 0:
            for r in rec:
                print(f"  hop{r['hop']} {r['emb']:14s} code-agree "
                      f"{r['code']:.3f}  drift {r['drift']:.3f}  "
                      f"token-agree {r['tok']:.3f}  nll {r['nll']:.3f}",
                      flush=True)
            print(f"  [control] ceiling (re-encode, no hop) {ceil:.3f}   "
                  f"floor (different motion) {floor:.3f}", flush=True)

    H = len(chain)
    code = np.array([[r["rec"][k]["code"] for k in range(H)] for r in rows])
    drift = np.array([[r["rec"][k]["drift"] for k in range(H)] for r in rows])
    tok = np.array([[r["rec"][k]["tok"] for k in range(H)] for r in rows])
    nll = np.array([[r["rec"][k]["nll"] for k in range(H)] for r in rows])
    ceil = np.mean([r["ceil"] for r in rows])
    floor = np.mean([r["floor"] for r in rows])
    print(f"\n{'hop':>3} {'embodiment':14s} {'code-agree':>11s} {'drift':>7s} "
          f"{'token-agree':>12s} {'prior NLL':>10s}")
    for k in range(H):
        print(f"{k:3d} {chain[k]:14s} {code[:,k].mean():11.3f} "
              f"{drift[:,k].mean():7.3f} {tok[:,k].mean():12.3f} "
              f"{nll[:,k].mean():10.3f}")
    print(f"\ncontrols: ceiling {ceil:.3f} (same pose re-encoded), "
          f"floor {floor:.3f} (unrelated motion)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(H)
    ax[0].plot(x, code.mean(0), "o-", lw=2, color="#c33", label="code agreement")
    ax[0].fill_between(x, np.quantile(code, .25, 0), np.quantile(code, .75, 0),
                       color="#c33", alpha=.18)
    ax[0].plot(x, tok.mean(0), "s-", lw=2, color="#38c", label="Equator token agreement")
    ax[0].axhline(ceil, color="k", ls="--", lw=1, label=f"ceiling {ceil:.2f}")
    ax[0].axhline(floor, color="0.5", ls=":", lw=1.5, label=f"floor {floor:.2f}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(chain, rotation=30, ha="right",
                                               fontsize=8)
    ax[0].set_ylabel("fraction identical to hop 0")
    ax[0].set_title("does the code survive body-to-body translation?")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(x, drift.mean(0), "o-", lw=2, color="#a3a")
    ax[1].set_xticks(x); ax[1].set_xticklabels(chain, rotation=30, ha="right",
                                               fontsize=8)
    ax[1].set_ylabel("mean |level - level_hop0|")
    ax[1].set_title("lattice drift (0 = identical, ~2.6 = random)")
    ax[1].grid(alpha=0.3)
    ax[2].plot(x, nll.mean(0), "o-", lw=2, color="#3a6")
    ax[2].fill_between(x, np.quantile(nll, .25, 0), np.quantile(nll, .75, 0),
                       color="#3a6", alpha=.18)
    ax[2].set_xticks(x); ax[2].set_xticklabels(chain, rotation=30, ha="right",
                                               fontsize=8)
    ax[2].set_ylabel("B3 masked CE (nats/token)")
    ax[2].set_title("does the temporal prior notice the damage?")
    ax[2].grid(alpha=0.3)
    fig.suptitle("EMBODIMENT CHAIN — human -> body -> body -> ..., "
                 "re-encoded at every hop, compared with hop 0", fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/embodiment_chain.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"chain": chain, "code": code.mean(0).tolist(),
               "drift": drift.mean(0).tolist(), "token": tok.mean(0).tolist(),
               "nll": nll.mean(0).tolist(), "ceiling": ceil, "floor": floor},
              open(f"{RUNS}/EMBODIMENT_CHAIN_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
