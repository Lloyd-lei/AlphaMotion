"""COMPOSITION TEST — can the stack represent a limb combination the corpus
never contains?

    composition_test.py

Correction that motivates this script: Greenwich's 128 code slots are LEARNED
QUERIES, not per-joint slots, so "splice the arm slots" is not an available
operation.  The well-posed version is to compose in OBSERVATION space and ask
what the representation does with it.

  build     take a walk clip and a wave/punch clip; keep the walker's lower
            body and root, graft the other clip's arm-chain LOCAL rotations.
            The result is a motion that (by construction) no corpus clip
            contains, while every joint trajectory individually comes from real
            data — so it is novel in COMBINATION, not in content.
  Q1 spatial   encode->decode through frozen Greenwich: reconstruction error on
               the composite vs on its two real parents.  If the composite is
               reconstructed as well as real motion, the spatial code is
               COMPOSITIONAL — it spans combinations it never saw.
  Q2 temporal  A3 tokens -> B3 masked NLL of the composite vs the parents.
               Higher NLL = the temporal prior treats the combination as
               unlikely, which is what would block generating it.
  Q3 novelty   nearest-neighbour distance to a real-motion index, with the same
               pipeline-floor control as novelty_frontier.

Figure: png/composition_test.png
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
from bridge_b import BridgeB, tokens_of                              # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
ARM_KW = ("shoulder", "elbow", "wrist", "hand", "collar", "clavicle")
N = 60


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="walk:punch,walk:dance,sit:punch")
    ap.add_argument("--n_rep", type=int, default=8)
    ap.add_argument("--n_corpus", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=spcfg.get("sem_encoder"))
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    jn = [n.lower() for n in sspec.joint_names]
    par = list(sspec.parents)

    def is_arm(i):
        j, seen = i, 0
        while j >= 0 and seen < 12:
            if any(k in jn[j] for k in ARM_KW):
                return True
            j = par[j]; seen += 1
        return False
    arm_idx = [i for i in range(len(jn)) if is_arm(i)]
    print(f"[arm chain] {len(arm_idx)} joints: "
          f"{[sspec.joint_names[i] for i in arm_idx][:8]}", flush=True)

    # the encoder's input is EXACTLY dual_common's: cat([gr6, gp / reach]) from
    # build_global, with reach measured over the whole corpus. Hand-rolling it
    # from fk_pos/height gave 23 deg self-reconstruction where the known value
    # is 0.1 deg — that broken run is discarded.
    _ref = l6h[torch.as_tensor(np.arange(0, 200000))]
    _, _gp, REACH = build_global(_ref, sspec, dev)
    del _ref, _gp
    print(f"[reach] reference {REACH:.4f}", flush=True)

    def fk_from_local(l6):
        g6, gp, _ = build_global(l6, sspec, dev)
        return torch.as_tensor(V.fk_pos(g6, sspec)).float(), (g6, gp)

    def pose9(gg6):
        g6, gp = gg6
        return torch.cat([g6.float(), gp.float() / REACH], -1).to(dev)

    def recon_err(p9):
        zp, zr = sp.encode(p9, E.geom_batch(g_h, p9.shape[0]))
        prot, _ = sp.decode(zp, zr, E.geom_batch(g_h, p9.shape[0]))
        a = rot6d_to_matrix(prot); b = rot6d_to_matrix(p9[..., :6])
        cos = ((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        return float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).mean())

    def nll_of(p9):
        zp, zr = sp.encode(p9, E.geom_batch(g_h, p9.shape[0]))
        ip = sp.fsqp.ints(sp.fsqp.features(sp.poolp(
            sp.qp[None].expand(1, -1, -1), sp._trunk(p9[None], E.geom_batch(g_h, p9.shape[0]), None)))) \
            if False else None
        return None

    # corpus index for novelty (raw GT)
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    g = np.random.default_rng(0)
    pool = [(a, b) for spl in ("train", "val") for a, b in data.pools[spl]
            if b - a >= N + 2]
    cst = [int(g.integers(a, b - N)) for a, b in
           (pool[i] for i in g.choice(len(pool), args.n_corpus, replace=True))]
    tsub = torch.linspace(0, N - 1, 12).long()

    def feat(fk):
        x = fk[tsub]
        return (x - x[:, :1]).reshape(-1)
    Cf = []
    for i in range(0, len(cst), 256):
        blk = cst[i:i + 256]
        rows = torch.as_tensor(np.concatenate([np.arange(s, s + N) for s in blk]))
        fk, _ = fk_from_local(l6h[rows])
        fk = fk.reshape(len(blk), N, -1, 3)
        Cf += [feat(fk[k]) for k in range(len(blk))]
    Cf = torch.stack(Cf)
    print(f"[corpus] {len(Cf)} windows", flush=True)

    def novelty(fk):
        return float(torch.cdist(feat(fk)[None], Cf).min()) / (12 * 3) ** 0.5

    def jerk(fk):
        return float((fk[2:] - 2 * fk[1:-1] + fk[:-2]).norm(dim=-1).mean())

    gg = np.random.default_rng(args.seed)
    rows_out = []
    for spec_pair in args.pairs.split(","):
        fa, fb = spec_pair.split(":")
        for rep in range(args.n_rep):
            try:
                sa, _ = kd.sample_starts(1, gg, "val", span=N + 2, families=[fa])
                sb, _ = kd.sample_starts(1, gg, "val", span=N + 2, families=[fb])
            except Exception:
                continue
            ra, rb = int(sa[0]), int(sb[0])
            la = l6h[torch.as_tensor(np.arange(ra, ra + N))].clone()
            lb = l6h[torch.as_tensor(np.arange(rb, rb + N))].clone()
            lc = la.clone()
            lc[:, arm_idx] = lb[:, arm_idx]                # graft the arm chain
            out = {}
            for tag, l6 in (("A_" + fa, la), ("B_" + fb, lb), ("COMPOSITE", lc)):
                fk, g6 = fk_from_local(l6)
                p9 = pose9(g6)
                out[tag] = {"recon_deg": recon_err(p9), "novelty": novelty(fk),
                            "jerk": jerk(fk)}
            # temporal prior opinion (needs codes -> A3 tokens -> B3 CE)
            for tag, l6 in (("A_" + fa, la), ("B_" + fb, lb), ("COMPOSITE", lc)):
                fk, g6 = fk_from_local(l6)
                p9 = pose9(g6)
                zp, zr = sp.encode(p9, E.geom_batch(g_h, p9.shape[0]))
                ipp = sp.fsqp.ints(sp.fsqp.features(zp))[0]
                irr = sp.fsqr.ints(sp.fsqr.features(zr))[0]
                c = torch.cat([ipp, irr], 1)[None]
                tok, ep = tokens_of(ma, c)
                nf = torch.full((1,), float(N), device=dev)
                mk = torch.ones_like(tok, dtype=torch.bool)
                out[tag]["nll"] = float(F.cross_entropy(
                    mb(tok, mk, ep, nf)[mk], tok[mk]))
            rows_out.append({"pair": spec_pair, **{f"{k}|{m}": v[m]
                             for k, v in out.items() for m in v}})
            if rep == 0:
                print(f"[{spec_pair}] recon A/B/C "
                      f"{out['A_'+fa]['recon_deg']:.1f}/"
                      f"{out['B_'+fb]['recon_deg']:.1f}/"
                      f"{out['COMPOSITE']['recon_deg']:.1f} deg   "
                      f"nll {out['A_'+fa]['nll']:.2f}/{out['B_'+fb]['nll']:.2f}/"
                      f"{out['COMPOSITE']['nll']:.2f}   "
                      f"novelty {out['A_'+fa]['novelty']:.1f}/"
                      f"{out['COMPOSITE']['novelty']:.1f}", flush=True)

    def col(kind, metric):
        vals = []
        for r in rows_out:
            for k in r:
                if k.startswith(kind) and k.endswith("|" + metric):
                    vals.append(r[k])
        return np.array(vals)
    parents_r = np.concatenate([col("A_", "recon_deg"), col("B_", "recon_deg")])
    comp_r = col("COMPOSITE", "recon_deg")
    parents_n = np.concatenate([col("A_", "nll"), col("B_", "nll")])
    comp_n = col("COMPOSITE", "nll")
    parents_v = np.concatenate([col("A_", "novelty"), col("B_", "novelty")])
    comp_v = col("COMPOSITE", "novelty")
    print(f"\nrecon deg   parents {parents_r.mean():.2f}  composite {comp_r.mean():.2f}")
    print(f"prior NLL   parents {parents_n.mean():.3f}  composite {comp_n.mean():.3f}")
    print(f"novelty cm  parents {parents_v.mean():.2f}  composite {comp_v.mean():.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for a_, (pv, cv, name, unit) in zip(ax, [
            (parents_r, comp_r, "Q1 spatial reconstruction", "geodesic deg"),
            (parents_n, comp_n, "Q2 temporal prior NLL", "nats/token"),
            (parents_v, comp_v, "Q3 novelty vs real corpus", "cm")]):
        a_.boxplot([pv, cv], labels=["real parents", "COMPOSITE"])
        a_.scatter(np.ones(len(pv)) + np.random.normal(0, .04, len(pv)), pv,
                   s=18, alpha=.6, color="#468")
        a_.scatter(2 + np.random.normal(0, .04, len(cv)), cv, s=18, alpha=.6,
                   color="#c33")
        a_.set_title(f"{name}\nparents {pv.mean():.2f} -> composite {cv.mean():.2f}")
        a_.set_ylabel(unit); a_.grid(alpha=0.3, axis="y")
    fig.suptitle("COMPOSITION TEST — walker's legs + another clip's arms: a "
                 "combination no corpus clip contains", fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/composition_test.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({"recon": [float(parents_r.mean()), float(comp_r.mean())],
               "nll": [float(parents_n.mean()), float(comp_n.mean())],
               "novelty": [float(parents_v.mean()), float(comp_v.mean())]},
              open(f"{RUNS}/COMPOSITION_TEST_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
