"""GAUGE INVARIANCE — is the dead sector the SAME set of coordinates everywhere?

    gauge_invariance.py

response_spectrum found ~100 of 192 lattice coordinates to be near-null at any
operating point: the code moves, the body does not.  If that gauge sector is
the same coordinates across bridges, it is an INVARIANT of the representation —
a stronger and statistically far better-powered claim than aligning the ~5
active modes, and directly actionable (wasted codebook capacity).

Why per-coordinate sensitivity rather than subspace angles: in 192 dims two
~100-dim subspaces must intersect by dimension counting, so principal angles
are dominated by that arithmetic.  The column norms of the Jacobian give a
192-vector per bridge -- "how much does the body move when I push coordinate
i?" -- and those profiles can be compared directly.

Controls, both necessary:
  shuffle   correlate profile_A with a coordinate-permuted profile_B: kills any
            shared structure while preserving the marginal distribution.
  token-mean the profile could be trivially shared because whole TOKEN
            POSITIONS (time segments) differ in activity.  So the test is
            repeated after removing the per-token mean, isolating the
            within-token (per-FSQ-dimension) pattern.

Figure: png/gauge_invariance.png
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

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
SCENES = [("walk", "kick"), ("run", "sit"), ("dance", "walk"), ("punch", "jump"),
          ("jacks", "run"), ("sit", "dance"), ("kick", "punch"), ("jump", "walk")]
NSUB = 12


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=3)
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
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=spcfg.get("sem_encoder"))
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    tsub = torch.linspace(0, N - 1, NSUB).long()

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def render_batch(lat_b, ep):
        ints = lat_b.round().clamp(0, 4)
        outs = []
        for i in range(0, ints.shape[0], 8):
            blk = ints[i:i + 8]
            z = ma.fsq.from_ints(blk[None])
            q = (F.softmax(ma.decode(z, ep.expand(len(blk), -1, -1), N), -1)
                 * lv).sum(-1)
            qf = q.reshape(-1, q.shape[2], q.shape[3])
            zp = sp.fsqp.proj_out(qf[:, :128]); zr = sp.fsqr.proj_out(qf[:, 128:])
            rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, qf.shape[0]))
            fk = torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()
            fk = fk.reshape(len(blk), N, -1, 3)[:, tsub]
            fk = fk - fk[:, :, :1]
            outs.append(fk.reshape(len(blk), -1))
        return torch.cat(outs)

    gg = np.random.default_rng(args.seed)
    profiles, names = [], []
    for fa, fb in SCENES:
        sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[fa])
        sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[fb])
        ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
        ep = ma.endpoints(torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev))
        nf = torch.full((1,), float(N), device=dev)
        tok, _ = sample_nll(mb, ep, nf, dev, temp=args.tau, seed=args.seed)
        base = ((tok[:, None] // W) % 5).float().reshape(-1)
        d = base.numel()
        pert = []
        for i in range(d):
            for sgn in (+1.0, -1.0):
                v = base.clone(); v[i] = (v[i] + sgn).clamp(0, 4)
                pert.append(v.reshape(32, 6))
        Fp = render_batch(torch.stack(pert), ep)
        J = ((Fp[0::2] - Fp[1::2]) / 2.0).numpy()            # [d, feat]
        s = np.linalg.norm(J, axis=1)                        # per-coordinate
        profiles.append(s); names.append(f"{fa}->{fb}")
        dead = int((s < 0.05 * s.max()).sum())
        print(f"[{fa}->{fb}] dead coords {dead}/{d}  "
              f"sensitivity max {s.max():.2f} median {np.median(s):.3f}",
              flush=True)
    P = np.stack(profiles)                                   # [S,192]
    S_ = len(P)

    def corr_matrix(M):
        Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
        return (Z @ Z.T) / M.shape[1]

    C = corr_matrix(P)
    off = C[~np.eye(S_, dtype=bool)]
    rng = np.random.default_rng(0)
    shuf = []
    for _ in range(200):
        i, j = rng.choice(S_, 2, replace=False)
        a = P[i]; b = P[j][rng.permutation(P.shape[1])]
        shuf.append(float(np.corrcoef(a, b)[0, 1]))
    # within-token pattern: remove the per-token mean (token = time segment)
    Pt = P.reshape(S_, 32, 6)
    Pw = (Pt - Pt.mean(2, keepdims=True)).reshape(S_, -1)
    Cw = corr_matrix(Pw)
    offw = Cw[~np.eye(S_, dtype=bool)]
    print(f"\ncross-bridge sensitivity-profile correlation: {off.mean():.3f}"
          f"   shuffled control {np.mean(shuf):.3f} +- {np.std(shuf):.3f}")
    print(f"within-token (per-FSQ-dim) correlation:       {offw.mean():.3f}")
    # consistently dead coordinates
    thr = np.array([0.05 * p.max() for p in P])[:, None]
    deadmask = P < thr
    always = int((deadmask.all(0)).sum())
    frac = deadmask.mean()
    expect = float(frac ** S_ * P.shape[1])
    print(f"coords dead in ALL {S_} bridges: {always}/192  "
          f"(chance if independent: {expect:.1f})")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].imshow(P.mean(0).reshape(32, 6).T, aspect="auto", cmap="magma",
                      origin="lower")
    ax[0].set_xlabel("token position (time segment)")
    ax[0].set_ylabel("FSQ dimension")
    ax[0].set_title("mean sensitivity  |dBody / dCoord|\n(dark = gauge: code "
                    "moves, body does not)")
    plt.colorbar(im, ax=ax[0])
    im2 = ax[1].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(S_)); ax[1].set_yticks(range(S_))
    ax[1].set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_title(f"cross-bridge profile correlation\nmean off-diagonal "
                    f"{off.mean():.2f}")
    plt.colorbar(im2, ax=ax[1])
    ax[2].hist(shuf, bins=30, color="0.7", label="coordinate-shuffled control")
    ax[2].axvline(off.mean(), color="#c33", lw=2.5,
                  label=f"cross-bridge {off.mean():.2f}")
    ax[2].axvline(offw.mean(), color="#38c", lw=2.5, ls="--",
                  label=f"within-token only {offw.mean():.2f}")
    ax[2].set_xlabel("correlation"); ax[2].set_ylabel("count")
    ax[2].set_title("is the gauge sector an invariant?")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle("GAUGE INVARIANCE — do all bridges agree on which coordinates "
                 "are dead?", fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/gauge_invariance.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"names": names, "corr_mean": float(off.mean()),
               "corr_within_token": float(offw.mean()),
               "shuffle_mean": float(np.mean(shuf)),
               "shuffle_std": float(np.std(shuf)),
               "always_dead": always, "chance_dead": expect,
               "profile_mean": P.mean(0).tolist()},
              open(f"{RUNS}/GAUGE_INVARIANCE_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
