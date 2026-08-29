"""RAINBOW BRIDGE — temperature-annealed spectroscopy of P(path | endpoints).

    rainbow_bridge.py

lloyd's "温度采样扩散—拉普拉斯妖" algorithm, as built:

  WHITE LIGHT   fix (start, goal, n).  Draw K paths with Gumbel-max sampling
                at high tau: logits/tau + G, with G FROZEN PER SEED (common
                random numbers).  High tau => G dominates => chaotic cloud.
  PRISM         walk tau down a ladder, reusing the same G per seed, so every
                seed is a trackable trajectory through temperature.  Modes
                coalesce as tau falls; each merge is a phase transition.
  SPECTRUM      at each rung: pairwise distance between paths measured in
                RENDERED joint space (physically meaningful, not token
                Hamming), build an affinity graph, take the normalized graph
                LAPLACIAN, and read the eigengap => how many distinct modes
                survive at this temperature.  (The other Laplace.)
  DEMON         at low tau the seed -> mode map is (near-)deterministic: the
                genealogy strip shows which basin every seed falls into, and
                merges are readable backwards.

Outputs (all under R4_VIZ):
  png/rainbow_cascade.png    modes & spread vs temperature (+ eigenvalue bars)
  png/rainbow_spectrum.png   MDS embedding of the path cloud at 3 rungs
  png/rainbow_genealogy.png  seed x tau strip: which mode each seed lands in
  mp4/.../rainbow_modes_x2.mp4   the surviving modes rendered side by side
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
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
import render_newbody as RN                                          # noqa: E402
import greenwich_carnival as GC                                      # noqa: E402
from bridge_a import CodeData, load_bridge_a, FULL                   # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from spatial5 import pad_band, font                                  # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
TAUS = [4.0, 2.5, 1.6, 1.0, 0.6, 0.35, 0.2]


def gumbel_paths(mb, ep, nf, G, tau, dev):
    """Gumbel-max sampling with FROZEN noise G [K,32,V] -> tokens [K,32].

    pick = argmax(logits/tau + G) == a draw from softmax(logits/tau), but the
    randomness is pinned to the seed, so the same seed is one trackable
    trajectory across the whole temperature ladder.
    """
    K = G.shape[0]
    tok = torch.zeros(K, 32, dtype=torch.long, device=dev)
    mask = torch.ones(K, 32, dtype=torch.bool, device=dev)
    ar = torch.arange(K, device=dev)
    for _ in range(32):
        logits = mb(tok, mask, ep.expand(K, -1, -1), nf.expand(K))
        scored = logits / tau + G
        conf = scored.max(-1).values.masked_fill(~mask, -1e9)
        pos = conf.argmax(-1)
        tok[ar, pos] = scored[ar, pos].argmax(-1)
        mask[ar, pos] = False
    return tok


def spectral_modes(D, knn=None):
    """kNN affinity with self-tuning scale -> normalized Laplacian ->
    (n_modes, labels, eigs).

    A FULLY connected Gaussian graph always has exactly one zero eigenvalue, so
    it can never report more than one mode: separated clusters must be made
    weakly connected first.  Hence a mutual-kNN graph with Zelnik-Manor local
    scaling (sigma_i = distance to the k-th neighbour)."""
    K = D.shape[0]
    knn = knn or max(3, K // 8)
    idx = np.argsort(D, 1)
    sig = D[np.arange(K), idx[:, knn]] + 1e-6              # local scale
    A = np.exp(-(D ** 2) / (sig[:, None] * sig[None, :]))
    nb = np.zeros_like(A, bool)
    for i in range(K):
        nb[i, idx[i, 1:knn + 1]] = True
    A = A * (nb | nb.T)                                    # union-kNN sparsify
    np.fill_diagonal(A, 0.0)
    d = A.sum(1) + 1e-9
    L = np.eye(K) - (A / np.sqrt(d[:, None] * d[None, :]))
    w, v = np.linalg.eigh(L)
    gaps = np.diff(w[:min(8, K)])
    k = int(np.argmax(gaps[:6])) + 1                      # eigengap heuristic
    X = v[:, :k]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    # k-means on the spectral embedding (few points, plain Lloyd iterations)
    rng = np.random.default_rng(0)
    C = X[rng.choice(K, k, replace=False)]
    lab = np.zeros(K, int)
    for _ in range(60):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return k, lab, w[:8]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--fam_a", default="walk")
    ap.add_argument("--fam_b", default="kick")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, K, N = args.device, args.K, args.n
    os.makedirs(f"{VIZ}/png", exist_ok=True)
    os.makedirs(f"{VIZ}/mp4/bridge_demo", exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData(args.spatial, use_root=True)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    # endpoints: a real start pose -> a real goal pose from a different family
    gg = np.random.default_rng(args.seed)
    sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[args.fam_a])
    sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[args.fam_b])
    ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
    ep_c = torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev)
    ep = ma.endpoints(ep_c)
    nf = torch.full((1,), float(N), device=dev)

    # frozen Gumbel noise: one "initial condition" per seed
    gsrc = torch.Generator(device=dev).manual_seed(args.seed)
    U = torch.rand(K, 32, 5 ** 6, device=dev, generator=gsrc).clamp_(1e-9, 1 - 1e-9)
    G = -torch.log(-torch.log(U))
    del U

    def fk_of(tok):
        out = []
        for i in range(tok.shape[0]):
            z = ma.fsq.from_ints(((tok[i][:, None] // W) % 5)[None].float())[None]
            q = (F.softmax(ma.decode(z, ep, N), -1)[0] * lv).sum(-1)
            zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
            rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
            out.append(torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float())
        return torch.stack(out)                                # [K,N,J,3]

    cascade, labels_by_tau, eigs_by_tau, fks_by_tau = [], [], [], {}
    for tau in TAUS:
        tok = gumbel_paths(mb, ep, nf, G, tau, dev)
        fk = fk_of(tok)
        f = fk.reshape(K, -1)
        D = torch.cdist(f, f).numpy() / (N * fk.shape[2]) ** 0.5
        k, lab, w = spectral_modes(D)
        spread = float((fk - fk.mean(0, keepdim=True)).norm(dim=-1).mean())
        uniq = len({tuple(t.tolist()) for t in tok.cpu()})
        cascade.append({"tau": tau, "modes": k, "spread_cm": spread,
                        "unique_token_paths": uniq,
                        "mean_pair_dist": float(D[np.triu_indices(K, 1)].mean())})
        labels_by_tau.append(lab); eigs_by_tau.append(w)
        fks_by_tau[tau] = (fk, D, tok)
        print(f"[tau {tau:4.2f}] modes {k}  spread {spread:5.2f}cm  "
              f"unique paths {uniq}/{K}", flush=True)

    # ---------- Fig 1: coalescence cascade ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    t = [c["tau"] for c in cascade]
    ax[0].plot(t, [c["modes"] for c in cascade], "o-", color="#c33", lw=2,
               label="distinct modes (eigengap)")
    ax[0].set_xscale("log"); ax[0].invert_xaxis()
    ax[0].set_xlabel("temperature  tau  (annealing ->)")
    ax[0].set_ylabel("number of modes", color="#c33")
    ax2 = ax[0].twinx()
    ax2.plot(t, [c["spread_cm"] for c in cascade], "s--", color="#36c", lw=2,
             label="path spread (cm)")
    ax2.set_ylabel("path spread (cm)", color="#36c")
    ax[0].set_title("DISPERSION CASCADE — modes RESOLVE as the prism cools,\n"
                    "then collapse when it is too cold")
    ax[0].grid(alpha=0.3)
    for i, tau in enumerate(TAUS):
        w = eigs_by_tau[i]
        ax[1].plot(range(1, len(w) + 1), w, "o-", ms=4,
                   label=f"tau={tau}", alpha=0.85)
    ax[1].set_xlabel("eigenvalue index of the normalized graph Laplacian")
    ax[1].set_ylabel("eigenvalue")
    ax[1].set_title("LAPLACIAN SPECTRUM — the gap counts the colours")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p1 = f"{VIZ}/png/rainbow_cascade.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print("->", p1, flush=True)

    # ---------- Fig 2: MDS of the path cloud at three rungs ----------
    show = [TAUS[0], TAUS[len(TAUS) // 2], TAUS[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax_, tau in zip(axes, show):
        _, D, _ = fks_by_tau[tau]
        J = np.eye(K) - 1.0 / K
        B = -0.5 * J @ (D ** 2) @ J
        w_, v_ = np.linalg.eigh(B)
        Y = v_[:, -2:] * np.sqrt(np.clip(w_[-2:], 0, None))
        lab = labels_by_tau[TAUS.index(tau)]
        for j in sorted(set(lab)):
            m = lab == j
            ax_.scatter(Y[m, 0], Y[m, 1], s=44, alpha=0.85, label=f"mode {j}")
        ax_.set_title(f"tau = {tau}   ({len(set(lab))} modes)")
        ax_.set_xlabel("MDS-1"); ax_.grid(alpha=0.3); ax_.legend(fontsize=7)
    axes[0].set_ylabel("MDS-2")
    fig.suptitle("THE RAINBOW — path cloud in rendered-joint distance, "
                 "coloured by spectral mode", fontsize=12)
    fig.tight_layout()
    p2 = f"{VIZ}/png/rainbow_spectrum.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    print("->", p2, flush=True)

    # ---------- Fig 3: demon's genealogy ----------
    M = np.stack(labels_by_tau)                                # [T,K]
    order = np.lexsort(M[::-1])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(M[:, order], aspect="auto", interpolation="nearest",
              cmap="tab10")
    ax.set_yticks(range(len(TAUS)))
    ax.set_yticklabels([f"{t}" for t in TAUS])
    ax.set_ylabel("temperature tau")
    ax.set_xlabel("seed (frozen Gumbel noise), sorted by final basin")
    ax.set_title("LAPLACE'S DEMON — every seed's basin at each temperature; "
                 "fixed noise, so each column is one trackable trajectory")
    fig.tight_layout()
    p3 = f"{VIZ}/png/rainbow_genealogy.png"
    fig.savefig(p3, dpi=150); plt.close(fig)
    print("->", p3, flush=True)

    # ---------- Video: the surviving modes, side by side ----------
    # the rainbow band: the rung that resolves the most modes
    tau_v = max(cascade, key=lambda c: (c["modes"], -c["tau"]))["tau"]
    fk, D, tok = fks_by_tau[tau_v]
    lab = labels_by_tau[TAUS.index(tau_v)]
    reps = []
    for j in sorted(set(lab))[:4]:
        idx = np.where(lab == j)[0]
        sub = D[np.ix_(idx, idx)].mean(1)
        reps.append((j, int(idx[int(np.argmin(sub))]), len(idx)))
    tiles, labs = [], []
    for j, i, cnt in reps:
        z = ma.fsq.from_ints(((tok[i][:, None] // W) % 5)[None].float())[None]
        q = (F.softmax(ma.decode(z, ep, N)[0], -1) * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5
        q[-1] = (ep_c[0, 2].float() - 4.0) / 3.5
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d, root_R=Rg[:, 0])
        tiles.append(GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                                    gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                                    qj.shape[0], status="nodataset"))
        labs.append(f"mode {j}  ({cnt}/{K} seeds)")
    frames = []
    for t_ in range(N):
        row = np.concatenate([pad_band(ti[min(t_, len(ti) - 1)])
                              for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k_, lb in enumerate(labs):
            dr.text((k_ * GC.TILE_W + 8, 6), lb, font=font(13),
                    fill=(255, 235, 170))
        dr.text((8, GC.TILE_H + 24),
                f"RAINBOW BRIDGE  same start ({args.fam_a}) -> same goal "
                f"({args.fam_b}), n={N}, tau={tau_v}: distinct ways across",
                font=font(12, False), fill=(212, 212, 218))
        frames.append(np.asarray(im))
    p4 = f"{VIZ}/mp4/bridge_demo/rainbow_modes_x2.mp4"
    RN.write_mp4(p4, np.stack(frames), 20)
    print("->", p4, flush=True)
    json.dump({"cascade": cascade, "taus": TAUS,
               "fam": [args.fam_a, args.fam_b], "K": K, "n": N},
              open(f"{RUNS}/RAINBOW_BRIDGE_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
