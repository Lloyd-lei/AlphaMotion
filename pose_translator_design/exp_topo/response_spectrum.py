"""RESPONSE SPECTRUM — normal modes of the latent, and whether different
bridges share a basis.

    response_spectrum.py

Physics framing, stated precisely so the analogy is not oversold: there is no
Hamiltonian and no unitary evolution here, so "eigenstate" in the QM sense does
not apply.  What DOES carry over is LINEAR RESPONSE.  Treat

    F : lattice coords (192)  ->  rendered motion (subsampled joint positions)

and estimate its Jacobian J by central differences of one lattice cell.  Then

  E1 NORMAL MODES     SVD(J).  Singular values = how strongly each latent
     direction moves the body: the mode spectrum.  Near-zero singular values
     are GAUGE directions — code changes that leave the motion invariant.
     Participation ratio gives the effective number of active modes, i.e. how
     many degrees of freedom the 192-coord latent actually exposes here.

  E2 SHARED BASIS     the earlier transfer test compared single vectors and
     failed.  The right question for a shared structure is subspace alignment:
     PRINCIPAL ANGLES between the top-k right-singular subspaces of different
     bridges, against a random-subspace control of the same dimension.  Local
     modes can differ while spanning a common subspace — as phonon modes do.

  E3 DENSITY OF STATES  histogram of per-token NLL over sampled paths: bands
     and gaps in the "energy" axis, the cheap spectral summary.

Figures: png/response_modes.png, png/response_subspace.png
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
from common import rot6d_to_matrix                                   # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
SCENES = [("walk", "kick"), ("run", "sit"), ("dance", "walk"),
          ("punch", "jump"), ("jacks", "run"), ("sit", "dance")]
NSUB = 12


def principal_angles(A, B):
    """A [d,k], B [d,k] orthonormal column bases -> angles in degrees."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=8)
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
        """lat_b [B,32,6] float -> features [B, NSUB*J*3] (root-relative)."""
        ints = lat_b.round().clamp(0, 4)
        outs = []
        for i in range(0, ints.shape[0], 8):
            blk = ints[i:i + 8]
            z = ma.fsq.from_ints(blk[None])                  # [b,32,d]
            q = (F.softmax(ma.decode(z, ep.expand(len(blk), -1, -1), N), -1)
                 * lv).sum(-1)                               # [b,N,slots,dims]
            qf = q.reshape(-1, q.shape[2], q.shape[3])
            zp = sp.fsqp.proj_out(qf[:, :128]); zr = sp.fsqr.proj_out(qf[:, 128:])
            rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, qf.shape[0]))
            fk = torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()
            fk = fk.reshape(len(blk), N, -1, 3)[:, tsub]
            fk = fk - fk[:, :, :1]
            outs.append(fk.reshape(len(blk), -1))
        return torch.cat(outs)

    gg = np.random.default_rng(args.seed)
    gsrc = torch.Generator(device=dev).manual_seed(args.seed)
    results = []
    for fa, fb in SCENES:
        sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[fa])
        sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[fb])
        ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
        ep_c = torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev)
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(N), device=dev)
        tok, _ = sample_nll(mb, ep, nf, dev, temp=args.tau, seed=args.seed)
        L0 = ((tok[:, None] // W) % 5).float()               # [32,6]
        base = L0.reshape(-1)                                # [192]
        d = base.numel()

        # central differences, one lattice cell
        pert = []
        for i in range(d):
            for sgn in (+1.0, -1.0):
                v = base.clone(); v[i] = (v[i] + sgn).clamp(0, 4)
                pert.append(v.reshape(32, 6))
        Fp = render_batch(torch.stack(pert), ep)             # [2d, feat]
        J = ((Fp[0::2] - Fp[1::2]) / 2.0).numpy().T          # [feat, d]
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        p = S ** 2 / (S ** 2).sum()
        pr = float(1.0 / (p ** 2).sum())                     # participation ratio
        gauge = int((S < 0.02 * S.max()).sum())
        results.append({"name": f"{fa}->{fb}", "S": S, "Vt": Vt, "pr": pr,
                        "gauge": gauge})
        print(f"[{fa}->{fb}] effective modes (participation ratio) {pr:.1f}/{d}"
              f"   near-null (gauge) dirs {gauge}"
              f"   sigma1/sigma10 {S[0]/S[9]:.1f}", flush=True)

    # ---------- Fig 1: mode spectra ----------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    cm = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        s = r["S"] / r["S"][0]
        ax[0].semilogy(range(1, len(s) + 1), s, lw=1.8,
                       color=cm(i / (len(results) - 1)), label=r["name"])
    ax[0].set_xlabel("mode index"); ax[0].set_ylabel("singular value / sigma_1")
    ax[0].set_title("NORMAL-MODE SPECTRUM of d(motion)/d(latent)\n"
                    "flat tail = gauge directions (code moves, body does not)")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
    ax[1].bar(range(len(results)), [r["pr"] for r in results], color="#468")
    ax[1].set_xticks(range(len(results)))
    ax[1].set_xticklabels([r["name"] for r in results], rotation=25,
                          ha="right", fontsize=8)
    ax[1].set_ylabel("participation ratio (effective # of modes)")
    ax[1].axhline(192, color="r", ls="--", lw=1, label="192 latent coords")
    ax[1].set_title("how many of the 192 coords actually move the body")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = f"{VIZ}/png/response_modes.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print("->", p1, flush=True)

    # ---------- Fig 2: subspace alignment vs random control ----------
    k = args.k
    S_ = len(results)
    M = np.zeros((S_, S_))
    for i in range(S_):
        for j in range(S_):
            a = principal_angles(results[i]["Vt"][:k].T, results[j]["Vt"][:k].T)
            M[i, j] = float(a.mean())
    rng = np.random.default_rng(0)
    rand = []
    for _ in range(200):
        A = rng.normal(size=(192, k)); B = rng.normal(size=(192, k))
        rand.append(float(principal_angles(A, B).mean()))
    rmean, rstd = float(np.mean(rand)), float(np.std(rand))
    off = M[~np.eye(S_, dtype=bool)]
    print(f"[subspace] cross-bridge mean principal angle {off.mean():.1f} deg "
          f"(random control {rmean:.1f} +- {rstd:.1f})", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    im = ax[0].imshow(M, cmap="viridis_r", vmin=0, vmax=90)
    ax[0].set_xticks(range(S_)); ax[0].set_yticks(range(S_))
    ax[0].set_xticklabels([r["name"] for r in results], rotation=30, ha="right",
                          fontsize=8)
    ax[0].set_yticklabels([r["name"] for r in results], fontsize=8)
    for i in range(S_):
        for j in range(S_):
            ax[0].text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                       fontsize=8, color="w" if M[i, j] > 45 else "k")
    ax[0].set_title(f"mean principal angle between top-{k} response subspaces\n"
                    "(0 deg = same subspace, 90 = orthogonal)")
    plt.colorbar(im, ax=ax[0], label="degrees")
    ax[1].hist(rand, bins=30, color="0.7", label="random subspaces")
    ax[1].axvline(off.mean(), color="#c33", lw=2.5,
                  label=f"cross-bridge mean {off.mean():.1f}")
    ax[1].set_xlabel("mean principal angle (deg)")
    ax[1].set_ylabel("count")
    ax[1].set_title("is the alignment better than chance?")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.suptitle("SHARED BASIS TEST — single eigenvectors did not transfer; "
                 "do the SUBSPACES still coincide?", fontsize=12)
    fig.tight_layout()
    p2 = f"{VIZ}/png/response_subspace.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    print("->", p2, flush=True)
    json.dump({"scenes": [r["name"] for r in results],
               "participation_ratio": [r["pr"] for r in results],
               "gauge_dirs": [r["gauge"] for r in results],
               "angle_matrix": M.tolist(),
               "random_mean": rmean, "random_std": rstd},
              open(f"{RUNS}/RESPONSE_SPECTRUM_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
