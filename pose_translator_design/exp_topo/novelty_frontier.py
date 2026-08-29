"""NOVELTY FRONTIER — does the model produce motion the CORPUS never contained?

    novelty_frontier.py

The decisive test for "we found a way to reach outside the training set":
sample bridge paths across a temperature ladder, and for each one measure

  novelty      distance to the NEAREST real corpus window, in rendered joint
               space (the corpus index is built from GT human motion, so this
               is a comparison against the data itself, not against the model)
  implausible  the model's own chain NLL (its opinion of the path)
  jerk         second-difference magnitude vs the corpus distribution — a
               physics-side sanity number that does not consult the model

Genuine extrapolation = a region with novelty ABOVE the corpus's own
nearest-neighbour spacing while jerk stays inside the corpus band.  If high
novelty only ever comes with high jerk, the "new" motion is just noise, and the
honest conclusion is that the model interpolates.

Baseline that makes the number meaningful: the corpus's own leave-one-out
nearest-neighbour distance.  A generated path is only "outside" if it is
farther from the corpus than real clips typically are from each other.

Figure: png/novelty_frontier.png
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
import eval_newbody as E                                             # noqa: E402
from bridge_a import CodeData, load_bridge_a, FULL                   # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from homecoming_probe import sample_nll                              # noqa: E402
from train_global import build_global                                # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
TAUS = [3.0, 2.0, 1.3, 0.9, 0.6, 0.35]
NSUB = 12                                                  # frames kept per window


def feat(fk):
    """[T,J,3] cm -> compact window descriptor (root-relative, time-subsampled)."""
    idx = torch.linspace(0, fk.shape[0] - 1, NSUB).long()
    x = fk[idx]
    x = x - x[:, :1]
    return x.reshape(-1)


def jerk_of(fk):
    return float((fk[2:] - 2 * fk[1:-1] + fk[:-2]).norm(dim=-1).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_corpus", type=int, default=6000)
    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, N = args.device, args.n

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=spcfg.get("sem_encoder"))
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5

    # ---------- corpus index (REAL human motion, no model involved) ----------
    g = np.random.default_rng(0)
    pool = [(a, b) for spl in ("train", "val") for a, b in data.pools[spl]
            if b - a >= N + 2]
    starts = [int(g.integers(a, b - N)) for a, b in
              (pool[i] for i in g.choice(len(pool), args.n_corpus, replace=True))]
    feats, jerks = [], []
    for i in range(0, len(starts), 256):
        blk = starts[i:i + 256]
        rows = torch.as_tensor(np.concatenate([np.arange(s, s + N) for s in blk]))
        g6, _, _ = build_global(l6h[rows], sspec, dev)
        fk = torch.as_tensor(V.fk_pos(g6, sspec)).float().reshape(len(blk), N, -1, 3)
        for k in range(len(blk)):
            feats.append(feat(fk[k]))
            jerks.append(jerk_of(fk[k]))
    Cf = torch.stack(feats)
    Cj = np.array(jerks)
    print(f"[corpus] {len(Cf)} windows, jerk median {np.median(Cj):.3f}",
          flush=True)

    # leave-one-out spacing = how far real clips are from each other
    sub = Cf[torch.randperm(len(Cf))[:1500]]
    Dl = torch.cdist(sub, Cf)
    Dl[Dl < 1e-6] = float("inf")
    loo = (Dl.min(1).values / (NSUB * (Cf.shape[1] // (NSUB * 3))) ** 0.5).numpy()
    loo_med, loo_p90 = float(np.median(loo)), float(np.quantile(loo, 0.9))
    print(f"[corpus] leave-one-out NN distance: median {loo_med:.2f} "
          f"p90 {loo_p90:.2f}", flush=True)

    # ---------- generated paths across temperature ----------
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    gg = np.random.default_rng(5)
    scenes = []
    for fa, fb in [("walk", "kick"), ("dance", "sit"), ("run", "punch")]:
        sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[fa])
        sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[fb])
        ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
        scenes.append(torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev))

    # ---------- CONTROL: real motion through the SAME pipeline ----------
    # Without this, "novelty" is uninterpretable: generated paths go through
    # tokens -> A3 -> C1 -> FK, while the corpus index is raw GT FK, so codec
    # distortion alone buys novelty. This is the floor.
    from bridge_b import tokens_of
    ctl_nov, ctl_jerk = [], []
    ctl_starts = [int(g.integers(a, b - N)) for a, b in
                  (pool[i] for i in g.choice(len(pool), 48, replace=False))]
    for s0 in ctl_starts:
        c = codes_at(s0, N)[None].to(dev)
        tok_r, ep_r = tokens_of(ma, c)
        z = ma.fsq.from_ints(((tok_r[0][:, None] // W) % 5)[None].float())[None]
        q = (F.softmax(ma.decode(z, ep_r, N), -1)[0] * lv).sum(-1)
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        fk = torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()
        d = torch.cdist(feat(fk)[None], Cf)
        ctl_nov.append(float(d.min()) / (NSUB * (Cf.shape[1] // (NSUB * 3))) ** 0.5)
        ctl_jerk.append(jerk_of(fk))
    ctl_nov_m, ctl_jerk_m = float(np.median(ctl_nov)), float(np.median(ctl_jerk))
    print(f"[CONTROL round-trip of REAL motion] novelty {ctl_nov_m:.2f} "
          f"jerk {ctl_jerk_m:.3f}  <- the pipeline floor", flush=True)

    rec = []
    for tau in TAUS:
        for si, ep_c in enumerate(scenes):
            ep = ma.endpoints(ep_c)
            nf = torch.full((1,), float(N), device=dev)
            for s_ in range(args.n_gen):
                tok, nll = sample_nll(mb, ep, nf, dev, temp=tau,
                                      seed=si * 1000 + s_)
                z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
                q = (F.softmax(ma.decode(z, ep, N), -1)[0] * lv).sum(-1)
                zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
                rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
                fk = torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()
                d = torch.cdist(feat(fk)[None], Cf)
                nov = float(d.min()) / (NSUB * (Cf.shape[1] // (NSUB * 3))) ** 0.5
                rec.append({"tau": tau, "scene": si, "novelty": nov,
                            "nll": nll, "jerk": jerk_of(fk)})
        m = [r for r in rec if r["tau"] == tau]
        print(f"[tau {tau:4.2f}] novelty {np.mean([r['novelty'] for r in m]):6.2f} "
              f"jerk {np.mean([r['jerk'] for r in m]):5.3f} "
              f"nll {np.mean([r['nll'] for r in m]):5.2f}", flush=True)

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cm = plt.get_cmap("plasma")
    for i, tau in enumerate(TAUS):
        m = [r for r in rec if r["tau"] == tau]
        ax[0].scatter([r["novelty"] for r in m], [r["jerk"] for r in m],
                      s=34, alpha=0.8, color=cm(i / (len(TAUS) - 1)),
                      label=f"tau={tau}")
    ax[0].axvline(loo_med, color="k", ls="--", lw=1)
    ax[0].axvline(loo_p90, color="k", ls=":", lw=1)
    ax[0].scatter(ctl_nov, ctl_jerk, s=60, marker="X", color="#0a0",
                  edgecolor="k", zorder=5,
                  label="CONTROL: real motion round-tripped")
    ax[0].text(loo_med, ax[0].get_ylim()[1] * 0.96, " corpus NN median",
               fontsize=8, rotation=90, va="top")
    ax[0].axhspan(float(np.quantile(Cj, 0.05)), float(np.quantile(Cj, 0.95)),
                  color="#8c8", alpha=0.18)
    ax[0].text(ax[0].get_xlim()[1] * 0.55, float(np.median(Cj)),
               "corpus jerk band (5-95%)", fontsize=8, color="#262")
    ax[0].set_xlabel("novelty  = distance to nearest REAL corpus window (cm)")
    ax[0].set_ylabel("jerk (cm/frame^2)")
    ax[0].set_title("Is new motion new, or just rough?")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    for i, tau in enumerate(TAUS):
        m = [r for r in rec if r["tau"] == tau]
        ax[1].scatter([r["novelty"] for r in m], [r["nll"] for r in m],
                      s=34, alpha=0.8, color=cm(i / (len(TAUS) - 1)))
    ax[1].axvline(loo_med, color="k", ls="--", lw=1)
    ax[1].set_xlabel("novelty (cm)")
    ax[1].set_ylabel("model's own NLL of the path")
    ax[1].set_title("Does the model itself believe the novel ones?")
    ax[1].grid(alpha=0.3)
    fig.suptitle("NOVELTY FRONTIER — generated paths vs the real corpus; "
                 f"corpus self-spacing median {loo_med:.2f} cm", fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/novelty_frontier.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"loo_median": loo_med, "loo_p90": loo_p90,
               "control_novelty": ctl_nov_m, "control_jerk": ctl_jerk_m,
               "corpus_jerk": [float(np.quantile(Cj, q)) for q in (.05, .5, .95)],
               "rec": rec}, open(f"{RUNS}/NOVELTY_FRONTIER_0811.json", "w"))


if __name__ == "__main__":
    main()
