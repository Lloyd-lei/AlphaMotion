"""RAINBOW EIGEN — diffusion eigenvectors of the path cloud, mapped back into
the latent, then pushed to see whether they STEER.

    rainbow_eigen.py

lloyd's addition to the Laplace's-demon algorithm: don't stop at counting modes
with eigenvalues — decompose with the eigenVECTORS.

  1 CLOUD      K paths, fixed endpoints, Gumbel-max at the rainbow temperature.
  2 COORDINATE diffusion-map eigenvectors psi_1..psi_4 of the rendered-joint
               affinity: continuous choice axes, not discrete labels.
  3 MEANING    for each axis, the [time x joint] difference between its high
               and low quartiles: WHERE and WHEN the axis acts.
  4 PULL BACK  ridge-regress psi_k on the A3 lattice coordinates -> a latent
               direction w_k.
  5 STEER      move the latent along +-alpha*w_k, re-quantize, decode, and test
               whether psi_k actually moves as predicted.  A correlational axis
               only becomes a knob if this test passes; the plot shows the
               predicted-vs-realised line, including failures.

Figures: png/eigen_axes.png, png/eigen_meaning.png, png/eigen_steer.png
Video:   mp4/bridge_demo/eigen_axis_extremes_x2.mp4
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
from rainbow_bridge import gumbel_paths                              # noqa: E402
from spatial5 import pad_band, font                                  # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")


def diffusion_eigs(D, knn=None, n_eig=5):
    """rendered-joint distances -> diffusion-map eigenvectors psi_1..psi_k."""
    K = D.shape[0]
    knn = knn or max(4, K // 8)
    idx = np.argsort(D, 1)
    sig = D[np.arange(K), idx[:, knn]] + 1e-6
    A = np.exp(-(D ** 2) / (sig[:, None] * sig[None, :]))
    np.fill_diagonal(A, 0.0)
    d = A.sum(1) + 1e-9
    L = np.eye(K) - A / np.sqrt(d[:, None] * d[None, :])
    w, v = np.linalg.eigh(L)
    psi = v[:, 1:1 + n_eig] / np.sqrt(d[:, None])          # skip trivial psi_0
    psi = psi / (np.linalg.norm(psi, axis=0, keepdims=True) + 1e-9)
    return w[:1 + n_eig], psi


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--K", type=int, default=96)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--fam_a", default="walk")
    ap.add_argument("--fam_b", default="kick")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, K, N = args.device, args.K, args.n

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
    jnames = list(sspec.joint_names)

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    gg = np.random.default_rng(args.seed)
    sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[args.fam_a])
    sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[args.fam_b])
    ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
    ep_c = torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev)
    ep = ma.endpoints(ep_c)
    nf = torch.full((1,), float(N), device=dev)

    gsrc = torch.Generator(device=dev).manual_seed(args.seed)
    U = torch.rand(K, 32, 5 ** 6, device=dev, generator=gsrc).clamp_(1e-9, 1 - 1e-9)
    G = -torch.log(-torch.log(U)); del U

    def lattice_of(tok):
        """token ids [.,32] -> lattice coords [.,32,6] float (0..4)."""
        return ((tok[..., None] // W) % 5).float()

    def fk_from_lattice(q6):
        """[32,6] lattice (float, will be rounded) -> human FK [N,J,3]."""
        ints = q6.round().clamp(0, 4)
        z = ma.fsq.from_ints(ints[None])[None]
        q = (F.softmax(ma.decode(z, ep, N), -1)[0] * lv).sum(-1)
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    print(f"[cloud] sampling K={K} at tau={args.tau}", flush=True)
    tok = gumbel_paths(mb, ep, nf, G, args.tau, dev)
    lat = lattice_of(tok)                                   # [K,32,6]
    fks = torch.stack([fk_from_lattice(lat[i]) for i in range(K)])
    f = fks.reshape(K, -1)
    D = torch.cdist(f, f).numpy() / (N * fks.shape[2]) ** 0.5
    w, psi = diffusion_eigs(D)
    print(f"[eig] eigenvalues {np.round(w, 4)}", flush=True)

    # ---------- Fig 1: axes ----------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].plot(range(len(w)), w, "o-", color="#333")
    ax[0].set_xlabel("index"); ax[0].set_ylabel("Laplacian eigenvalue")
    ax[0].set_title("spectrum (psi_0 trivial)"); ax[0].grid(alpha=0.3)
    s = ax[1].scatter(psi[:, 0], psi[:, 1], c=psi[:, 0], cmap="coolwarm", s=42)
    ax[1].set_xlabel("psi_1  (primary choice axis)"); ax[1].set_ylabel("psi_2")
    ax[1].set_title("the cloud in its own diffusion coordinates")
    ax[1].grid(alpha=0.3); plt.colorbar(s, ax=ax[1])
    ax[2].scatter(psi[:, 1], psi[:, 2], c=psi[:, 0], cmap="coolwarm", s=42)
    ax[2].set_xlabel("psi_2"); ax[2].set_ylabel("psi_3")
    ax[2].set_title("secondary axes"); ax[2].grid(alpha=0.3)
    fig.suptitle(f"EIGEN AXES of P(path | fixed endpoints), tau={args.tau}, "
                 f"K={K}: continuous choice coordinates, not class labels")
    fig.tight_layout()
    p1 = f"{VIZ}/png/eigen_axes.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print("->", p1, flush=True)

    # ---------- Fig 2: what each axis MEANS ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, ax_ in enumerate(axes):
        o = np.argsort(psi[:, k])
        lo, hi = o[:K // 4], o[-K // 4:]
        dif = (fks[hi].mean(0) - fks[lo].mean(0)).norm(dim=-1).numpy().T  # [J,N]
        im = ax_.imshow(dif, aspect="auto", cmap="magma", origin="lower")
        ax_.set_title(f"psi_{k+1}: high-quartile minus low-quartile")
        ax_.set_xlabel("frame")
        if k == 0:
            step = max(1, len(jnames) // 12)
            ax_.set_yticks(range(0, len(jnames), step))
            ax_.set_yticklabels([jnames[i] for i in range(0, len(jnames), step)],
                                fontsize=7)
            ax_.set_ylabel("joint")
        else:
            ax_.set_yticks([])
        plt.colorbar(im, ax=ax_, label="cm")
    fig.suptitle("WHAT EACH AXIS DOES — where and when the two extremes differ")
    fig.tight_layout()
    p2 = f"{VIZ}/png/eigen_meaning.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    print("->", p2, flush=True)

    # ---------- pull back to the latent, then STEER ----------
    X = lat.reshape(K, -1).cpu().numpy()                    # [K,192]
    Xc = X - X.mean(0, keepdims=True)
    # 192 latent dims vs K samples: an unregularised in-sample fit is a
    # tautology (it returned r=1.000). Reduce to a few PCs and report the
    # 5-fold OUT-OF-SAMPLE correlation instead.
    U_, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    npc = min(10, K // 4)
    P = Vt[:npc]                                            # [npc,192]
    Z = Xc @ P.T                                            # [K,npc]
    lam = 1e-2 * K
    steer = {}
    folds = np.arange(K) % 5
    for k in range(3):
        y = psi[:, k]
        pred = np.zeros(K)
        for fo in range(5):
            tr, te = folds != fo, folds == fo
            A_ = Z[tr].T @ Z[tr] + lam * np.eye(npc)
            b = np.linalg.solve(A_, Z[tr].T @ y[tr])
            pred[te] = Z[te] @ b
        r_cv = float(np.corrcoef(pred, y)[0, 1])
        A_ = Z.T @ Z + lam * np.eye(npc)
        bk = np.linalg.solve(A_, Z.T @ y)
        wk = P.T @ bk                                       # back to 192-d
        wk = wk / (np.abs(wk).max() + 1e-9)                 # unit = 1 lattice cell
        steer[k] = (wk, r_cv)
        print(f"[pullback] psi_{k+1}: 5-fold out-of-sample r={r_cv:.3f}", flush=True)

    # causal test: push the median path along w_k, re-quantize, re-embed
    med = int(np.argmin(np.abs(psi[:, 0] - np.median(psi[:, 0]))))
    alphas = np.linspace(-args.alpha, args.alpha, 7)   # units: lattice cells
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    steer_fk = {}
    for k, ax_ in enumerate(axes):
        wk, r = steer[k]
        realised, pushed = [], []
        base = lat[med].reshape(-1).cpu().numpy()
        fk_list = []
        for a in alphas:
            v = base + a * wk                       # wk max-normalised: a = cells
            q6 = torch.as_tensor(v.reshape(32, 6), dtype=torch.float32, device=dev)
            fk_a = fk_from_lattice(q6)
            fk_list.append(fk_a)
            # Nystrom-style: project the new path onto psi_k via its neighbours
            d = (fks.reshape(K, -1) - fk_a.reshape(1, -1)).norm(dim=1).numpy()
            d = d / (N * fks.shape[2]) ** 0.5
            nb = np.argsort(d)[:8]
            wgt = np.exp(-(d[nb] / (np.median(d) + 1e-9)) ** 2)
            realised.append(float((psi[nb, k] * wgt).sum() / (wgt.sum() + 1e-9)))
            pushed.append(float(a))
        nchg = int((np.abs((base + alphas[-1] * wk).round()
                           - base.round()) > 0).sum())
        print(f"[steer] psi_{k+1}: {nchg}/{len(base)} lattice coords move at "
              f"alpha={alphas[-1]:.1f}", flush=True)
        steer_fk[k] = fk_list
        ax_.plot(pushed, realised, "o-", lw=2, color="#c33")
        ax_.axhline(psi[med, k], color="k", ls=":", lw=1,
                    label="unsteered path")
        rho = float(np.corrcoef(pushed, realised)[0, 1])
        ax_.set_title(f"psi_{k+1}   fit r={r:.2f}   steer rho={rho:.2f}")
        ax_.set_xlabel("alpha (push along latent direction w)")
        ax_.grid(alpha=0.3); ax_.legend(fontsize=8)
        print(f"[steer] psi_{k+1}: rho(alpha, realised)={rho:.3f}", flush=True)
    axes[0].set_ylabel("realised psi_k of the steered path")
    fig.suptitle("DOES THE AXIS STEER?  push the latent along w_k and measure "
                 "where the decoded motion lands (monotone => a real knob)")
    fig.tight_layout()
    p3 = f"{VIZ}/png/eigen_steer.png"
    fig.savefig(p3, dpi=150); plt.close(fig)
    print("->", p3, flush=True)

    # ---------- Video: psi_1 extremes + the steered sweep ----------
    def render_x2_from_lattice(q6):
        ints = q6.round().clamp(0, 4)
        z = ma.fsq.from_ints(ints[None])[None]
        q = (F.softmax(ma.decode(z, ep, N), -1)[0] * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5
        q[-1] = (ep_c[0, 2].float() - 4.0) / 3.5
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d, root_R=Rg[:, 0])
        return GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                              gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                              qj.shape[0], status="nodataset")

    o = np.argsort(psi[:, 0])
    picks = [(f"psi_1 LOW (sampled)", lat[o[0]]),
             (f"psi_1 HIGH (sampled)", lat[o[-1]]),
             (f"steered  a=-{args.alpha:g} cells",
              torch.as_tensor((lat[med].reshape(-1).cpu().numpy()
                               - args.alpha * steer[0][0]).reshape(32, 6),
                              dtype=torch.float32, device=dev)),
             (f"steered  a=+{args.alpha:g} cells",
              torch.as_tensor((lat[med].reshape(-1).cpu().numpy()
                               + args.alpha * steer[0][0]).reshape(32, 6),
                              dtype=torch.float32, device=dev))]
    tiles = [render_x2_from_lattice(q6) for _, q6 in picks]
    frames = []
    for t_ in range(N):
        row = np.concatenate([pad_band(ti[min(t_, len(ti) - 1)]) for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for j, (lb, _) in enumerate(picks):
            dr.text((j * GC.TILE_W + 8, 6), lb, font=font(13),
                    fill=(255, 235, 170) if j < 2 else (150, 235, 255))
        dr.text((8, GC.TILE_H + 24),
                "LEFT PAIR: the two extremes the sampler itself produced along "
                "psi_1.  RIGHT PAIR: one path pushed along the latent direction "
                "fitted to psi_1 (unsupervised knob).",
                font=font(12, False), fill=(212, 212, 218))
        frames.append(np.asarray(im))
    p4 = f"{VIZ}/mp4/bridge_demo/eigen_axis_extremes_x2.mp4"
    RN.write_mp4(p4, np.stack(frames), 20)
    print("->", p4, flush=True)
    json.dump({"eigenvalues": w.tolist(),
               "pullback_r": {f"psi_{k+1}": steer[k][1] for k in steer},
               "tau": args.tau, "K": K, "n": N},
              open(f"{RUNS}/RAINBOW_EIGEN_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
