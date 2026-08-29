"""EIGEN TRANSFER — is psi_1 the SAME axis across different endpoint pairs?

    eigen_transfer.py

The open question left by rainbow_eigen: those latent directions were fitted on
ONE endpoint pair.  If w_1 found on (walk->kick) also steers (run->sit), the
axis is a property of the model, not of that bridge — i.e. a general control
handle.  If not, each bridge has its own private axes and the handle is local.

Protocol, per scene S (an endpoint pair):
  fit w_1(S)      as in rainbow_eigen (cloud -> psi_1 -> ridge on lattice PCs)
  self-steer      push scene S along w_1(S)          -> rho_self
  cross-steer     push scene S along w_1(S') for S'!=S -> rho_cross
  random control  push along a random unit direction  -> rho_rand
Transfer is real only if rho_cross is close to rho_self AND clearly above
rho_rand.  The random arm is the honest floor: with 192 lattice coords, ANY
large push changes the motion, so "it moved" proves nothing by itself.

Figure: png/eigen_transfer.png  (matrix of rho, scenes x directions)
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
from rainbow_bridge import gumbel_paths                              # noqa: E402
from rainbow_eigen import diffusion_eigs                             # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
SCENES = [("walk", "kick"), ("run", "sit"), ("dance", "walk"), ("punch", "jump")]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, K, N = args.device, args.K, args.n

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

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    gsrc = torch.Generator(device=dev).manual_seed(args.seed)
    U = torch.rand(K, 32, 5 ** 6, device=dev, generator=gsrc).clamp_(1e-9, 1 - 1e-9)
    G = -torch.log(-torch.log(U)); del U

    scenes = []
    gg = np.random.default_rng(args.seed)
    for fa, fb in SCENES:
        sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[fa])
        sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[fb])
        ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
        ep_c = torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev)
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(N), device=dev)

        def fk_of(q6, ep=ep):
            z = ma.fsq.from_ints(q6.round().clamp(0, 4)[None])[None]
            q = (F.softmax(ma.decode(z, ep, N), -1)[0] * lv).sum(-1)
            zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
            rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
            return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

        tok = gumbel_paths(mb, ep, nf, G, args.tau, dev)
        lat = ((tok[..., None] // W) % 5).float()
        fks = torch.stack([fk_of(lat[i]) for i in range(K)])
        f = fks.reshape(K, -1)
        D = torch.cdist(f, f).numpy() / (N * fks.shape[2]) ** 0.5
        _, psi = diffusion_eigs(D)
        X = lat.reshape(K, -1).cpu().numpy()
        Xc = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:10]
        Z = Xc @ P.T
        b = np.linalg.solve(Z.T @ Z + 1e-2 * K * np.eye(10), Z.T @ psi[:, 0])
        w = P.T @ b
        w = w / (np.abs(w).max() + 1e-9)
        med = int(np.argmin(np.abs(psi[:, 0] - np.median(psi[:, 0]))))
        scenes.append({"name": f"{fa}->{fb}", "psi": psi, "fks": fks,
                       "base": X[med].copy(), "w": w, "fk_of": fk_of, "D": D})
        print(f"[fit] {fa}->{fb}", flush=True)

    def rho_of(sc, w, seed=None):
        """push scene sc along w; correlation of alpha with realised psi_1."""
        if seed is not None:
            rg = np.random.default_rng(seed)
            w = rg.normal(size=w.shape)
            w = w / (np.abs(w).max() + 1e-9)
        alphas = np.linspace(-args.alpha, args.alpha, 7)
        real = []
        psi, fks = sc["psi"], sc["fks"]
        for a in alphas:
            q6 = torch.as_tensor((sc["base"] + a * w).reshape(32, 6),
                                 dtype=torch.float32, device="cuda")
            fa_ = sc["fk_of"](q6)
            d = (fks.reshape(len(fks), -1) - fa_.reshape(1, -1)).norm(dim=1).numpy()
            d = d / (N * fks.shape[2]) ** 0.5
            nb = np.argsort(d)[:8]
            wg = np.exp(-(d[nb] / (np.median(d) + 1e-9)) ** 2)
            real.append(float((psi[nb, 0] * wg).sum() / (wg.sum() + 1e-9)))
        return float(np.corrcoef(alphas, real)[0, 1])

    S = len(scenes)
    M = np.zeros((S, S + 1))
    for i, sc in enumerate(scenes):
        for j, src in enumerate(scenes):
            M[i, j] = rho_of(sc, src["w"])
        M[i, S] = float(np.mean([rho_of(sc, sc["w"], seed=100 + r)
                                 for r in range(3)]))
        print(f"[rho] {sc['name']}: self {M[i,i]:.2f}  "
              f"cross {np.mean([M[i,j] for j in range(S) if j!=i]):.2f}  "
              f"random {M[i,S]:.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(S + 1))
    ax.set_xticklabels([s["name"] for s in scenes] + ["RANDOM"], rotation=25,
                       ha="right", fontsize=9)
    ax.set_yticks(range(S))
    ax.set_yticklabels([s["name"] for s in scenes], fontsize=9)
    for i in range(S):
        for j in range(S + 1):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="k" if abs(M[i, j]) < 0.6 else "w")
    ax.set_xlabel("direction w_1 fitted on ...   (last column = random control)")
    ax.set_ylabel("scene being steered")
    ax.set_title("EIGEN AXIS TRANSFER\ndiagonal = self-steer; off-diagonal = "
                 "does another bridge's axis work here?")
    plt.colorbar(im, label="rho(alpha, realised psi_1)")
    fig.tight_layout()
    p = f"{VIZ}/png/eigen_transfer.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"scenes": [s["name"] for s in scenes], "rho": M.tolist()},
              open(f"{RUNS}/EIGEN_TRANSFER_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
