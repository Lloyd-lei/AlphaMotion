"""SOLVER CERTIFICATE — can GT-free instance statistics predict output quality?

    solver_certificate.py

The claim worth testing (and the honest version of "solving hard subproblems"):
we cannot beat worst-case complexity, but we may be able to say, per instance
and WITHOUT ground truth, how much to trust this particular solve.

For each bridging instance (start pose, goal pose, n) compute four certificates,
all GT-free and all available online:

  C1 mode count      distinct paths among m samples (spectral, tau fixed)
  C2 spread          mean deviation of the sample cloud, cm
  C3 mean NLL        the prior's own opinion of its samples
  C4 eff modes       participation ratio of SVD(d motion / d latent), the local
                     search dimension (cheap variant: 24 random probe
                     directions instead of the full 192-coord Jacobian)

and one OUTCOME that does need a reference, used only to fit and to score:

  Y  endpoint error  distance from the decoded path's final pose to the actual
                     goal pose, in rendered joint cm — the thing a caller cares
                     about and cannot see at solve time.

Fit Y from C1..C4 with 5-fold cross-validation; report out-of-sample R^2 and a
CALIBRATION curve of the predicted 80% interval (does 80% of held-out truth
actually fall inside?).  An uncalibrated predictor is worse than none, so the
coverage number, not R^2, is the headline.

Figure: png/solver_certificate.png
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
from rainbow_eigen import diffusion_eigs                             # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
FAMS = ["walk", "run", "kick", "punch", "dance", "sit", "jump", "jacks"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--n_inst", type=int, default=60)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
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

    def fk_of_lat(lat, ep, n):
        z = ma.fsq.from_ints(lat.round().clamp(0, 4)[None])[None]
        q = (F.softmax(ma.decode(z, ep, n), -1)[0] * lv).sum(-1)
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    def fk_of_codes(c):
        zp = sp.fsqp.proj_out(((c[:, :128].float().to(dev) - 4.0) / 3.5))
        zr = sp.fsqr.proj_out(((c[:, 128:].float().to(dev) - 4.0) / 3.5))
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, c.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    gg = np.random.default_rng(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for it in range(args.n_inst):
        fa, fb = rng.choice(FAMS, 2, replace=False)
        n = int(rng.integers(30, 110))
        try:
            sa, _ = kd.sample_starts(1, gg, "val", span=4, families=[fa])
            sb, _ = kd.sample_starts(1, gg, "val", span=4, families=[fb])
        except Exception:
            continue
        ca, cb = codes_at(int(sa[0]), 2), codes_at(int(sb[0]), 2)
        ep_c = torch.stack([ca[0], ca[1], cb[0], cb[1]])[None].to(dev)
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(n), device=dev)

        lats, fks, nlls = [], [], []
        for s_ in range(args.m):
            tok, nll = sample_nll(mb, ep, nf, dev, temp=0.9, seed=it * 50 + s_)
            lat = ((tok[:, None] // W) % 5).float()
            lats.append(lat); nlls.append(nll)
            fks.append(fk_of_lat(lat, ep, n))
        fk = torch.stack(fks)
        f = fk.reshape(args.m, -1)
        D = torch.cdist(f, f).numpy() / (n * fk.shape[2]) ** 0.5
        try:
            _, psi = diffusion_eigs(D, knn=2, n_eig=2)
            c1 = float(np.std(psi[:, 0]) / (np.std(psi[:, 1]) + 1e-9))
        except Exception:
            c1 = 1.0
        c2 = float((fk - fk.mean(0, keepdim=True)).norm(dim=-1).mean())
        c3 = float(np.mean(nlls))

        # cheap effective-mode probe: random directions instead of full Jacobian
        base = lats[0].reshape(-1)
        resp = []
        for _ in range(args.probes):
            v = torch.as_tensor(rng.choice([-1.0, 0.0, 1.0], base.numel()),
                                dtype=torch.float32, device=dev)
            fkp = fk_of_lat((base + v).reshape(32, 6), ep, n)
            resp.append((fkp - fks[0]).reshape(-1).numpy())
        Rm = np.stack(resp)
        sv = np.linalg.svd(Rm, compute_uv=False)
        p = sv ** 2 / (sv ** 2).sum()
        c4 = float(1.0 / (p ** 2).sum())

        goal_fk = fk_of_codes(cb[:1])
        y = float((fks[0][-1] - goal_fk[0]).norm(dim=-1).mean())
        rows.append({"fam": f"{fa}->{fb}", "n": n, "c1": c1, "c2": c2,
                     "c3": c3, "c4": c4, "y": y})
        if it % 10 == 0:
            print(f"[{it}] {fa}->{fb} n={n} spread {c2:.2f} nll {c3:.2f} "
                  f"effmodes {c4:.1f} -> endpoint err {y:.2f}cm", flush=True)

    X = np.array([[r["c1"], r["c2"], r["c3"], r["c4"], r["n"]] for r in rows])
    Y = np.array([r["y"] for r in rows])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    folds = np.arange(len(X)) % 5
    pred, resid = np.zeros(len(X)), np.zeros(len(X))
    for fo in range(5):
        tr, te = folds != fo, folds == fo
        b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]),
                            X[tr].T @ Y[tr])
        pred[te] = X[te] @ b
        resid[tr] = Y[tr] - X[tr] @ b
    ss = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    sig = float(np.std(Y - pred))
    lo, hi = pred - 1.2816 * sig, pred + 1.2816 * sig      # nominal 80%
    cover = float(((Y >= lo) & (Y <= hi)).mean())
    # baseline: predict the mean (no certificates)
    ss0 = 1 - ((Y - Y.mean()) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    corr = {k: float(np.corrcoef([r[k] for r in rows], Y)[0, 1])
            for k in ("c1", "c2", "c3", "c4")}
    print(f"\nout-of-sample R^2 {ss:.3f} (mean-predictor {ss0:.3f})")
    print(f"nominal-80% interval coverage {cover*100:.0f}%  (sigma {sig:.2f}cm)")
    print("single-certificate correlations with endpoint error:", corr)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].scatter(pred, Y, s=40, alpha=0.8, color="#36c")
    lim = [min(pred.min(), Y.min()), max(pred.max(), Y.max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].fill_between(lim, [l - 1.2816 * sig for l in lim],
                       [l + 1.2816 * sig for l in lim], color="#36c", alpha=0.15)
    ax[0].set_xlabel("predicted endpoint error (cm), certificates only")
    ax[0].set_ylabel("actual endpoint error (cm)")
    ax[0].set_title(f"out-of-sample R^2 = {ss:.2f}\n80% band covers "
                    f"{cover*100:.0f}% of held-out cases")
    ax[0].grid(alpha=0.3)
    ks = list(corr)
    names = {"c1": "C1 mode ratio", "c2": "C2 spread", "c3": "C3 NLL",
             "c4": "C4 eff modes"}
    ax[1].barh([names[k] for k in ks], [corr[k] for k in ks], color="#468")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("correlation with actual error")
    ax[1].set_title("which certificate carries the signal?")
    ax[1].grid(alpha=0.3, axis="x")
    q = np.argsort(pred)
    k = max(1, len(q) // 3)
    for lab, sel, col in (("most-confident third", q[:k], "#3a3"),
                          ("least-confident third", q[-k:], "#c33")):
        ax[2].hist(Y[sel], bins=12, alpha=0.6, color=col, label=lab)
    ax[2].set_xlabel("actual endpoint error (cm)")
    ax[2].set_title("does the certificate separate good from bad solves?")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)
    fig.suptitle("SOLVER CERTIFICATE — GT-free instance statistics predicting "
                 "this solve's error", fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/solver_certificate.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"r2": ss, "coverage80": cover, "sigma": sig, "corr": corr,
               "rows": rows}, open(f"{RUNS}/SOLVER_CERT_0811.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
