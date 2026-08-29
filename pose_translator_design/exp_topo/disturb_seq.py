"""DISTURBANCE AS A SEGMENT — lloyd's formulation.

    disturb_seq.py

A shove is a MOTION, not a pose. The correct structure is

    A_curr --bridge1--> B_init [ real disturbance segment ] B_target --bridge2--> A_target

Motion A is what we were doing; motion B is what the disturbance makes us do.
The model builds BOTH seams itself, so seam continuity is a real measurement
(the previous version pinned a single pose and was retracted for exactly this).

Arms:
  CLEAN      A_curr -> A_target directly, n = n1 + |B| + n2      (reference)
  SEQ-given  the two bridges with supplied budgets
  SEQ-free   the two bridges with budgets the prior picks itself (NLL sweep)
Metrics, all measurable without ground truth for the generated parts:
  seam-in / seam-out  max single-frame joint step at each junction, against the
                      largest step the CLEAN trajectory itself takes
  arrive              distance from the final produced pose to A_target
  jerk                over the whole composed sequence
Figure: png/disturb_seq.png    Video: mp4/bridge_demo/disturb_seq_x2.mp4
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
from homecoming_probe import sample_nll                              # noqa: E402
from spatial5 import pad_band, font                                  # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
NGRID = [20, 30, 45, 60, 80, 100]
DIST_FAMS = ["jump", "kick", "dance"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--n1", type=int, default=30)
    ap.add_argument("--nB", type=int, default=45)
    ap.add_argument("--n2", type=int, default=45)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    n1, nB, n2 = args.n1, args.nB, args.n2

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def q_of_codes(c):
        return (c.float().to(dev) - 4.0) / 3.5

    def bridge(c4, n, seed, pin_exit=True):
        """pin_exit: the endpoint pass-through. Correct AT A SEAM (we are
        joining to a real segment), but it must be OFF on the final leg or
        "did it arrive" is 0.00 by construction — which is how the first run of
        this script reported a meaningless 0.00 for every arm."""
        ep = ma.endpoints(c4[None].to(dev))
        nf = torch.full((1,), float(n), device=dev)
        tok, nll = sample_nll(mb, ep, nf, dev, temp=0.9, seed=seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        q = (F.softmax(ma.decode(z, ep, n), -1)[0] * lv).sum(-1)
        q[0] = q_of_codes(c4[1])
        if pin_exit:
            q[-1] = q_of_codes(c4[2])
        return q, nll

    def pick_n(c4, seed):
        best = None
        for n_ in NGRID:
            ep = ma.endpoints(c4[None].to(dev))
            nf = torch.full((1,), float(n_), device=dev)
            _, nll = sample_nll(mb, ep, nf, dev, temp=0.9, seed=seed)
            if best is None or nll < best[1]:
                best = (n_, nll)
        return best[0]

    def fk(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    step = lambda f: (f[1:] - f[:-1]).norm(dim=-1).mean(-1)
    jerk = lambda f: float((f[2:] - 2 * f[1:-1] + f[:-2]).norm(dim=-1).mean())

    gg = np.random.default_rng(args.seed)
    rows, vid = [], None
    for rep in range(args.n_rep):
        fam = DIST_FAMS[rep % len(DIST_FAMS)]
        sa, _ = kd.sample_starts(1, gg, "val", span=4, families=["walk"])
        sg, _ = kd.sample_starts(1, gg, "val", span=4, families=["walk"])
        sb, _ = kd.sample_starts(1, gg, "val", span=nB + 2, families=[fam])
        cA = codes_at(int(sa[0]), 2)              # A_curr
        cG = codes_at(int(sg[0]), 2)              # A_target
        cB = codes_at(int(sb[0]), nB + 2)         # the real disturbance segment
        # CLEAN reference over the same total length
        Ntot = n1 + nB + n2
        qC, _ = bridge(torch.stack([cA[0], cA[1], cG[0], cG[1]]), Ntot, rep,
                       pin_exit=False)
        fC = fk(qC)
        # SEQ: A_curr -> B_init, real B, B_target -> A_target
        q1, _ = bridge(torch.stack([cA[0], cA[1], cB[0], cB[1]]), n1, rep)
        qB = q_of_codes(cB[:nB])
        q2, _ = bridge(torch.stack([cB[nB - 2], cB[nB - 1], cG[0], cG[1]]),
                       n2, rep + 77, pin_exit=False)
        qS = torch.cat([q1, qB, q2], 0)
        fS = fk(qS)
        # SEQ with the prior choosing both budgets
        m1 = pick_n(torch.stack([cA[0], cA[1], cB[0], cB[1]]), rep)
        m2 = pick_n(torch.stack([cB[nB - 2], cB[nB - 1], cG[0], cG[1]]), rep + 77)
        q1f, _ = bridge(torch.stack([cA[0], cA[1], cB[0], cB[1]]), m1, rep)
        q2f, _ = bridge(torch.stack([cB[nB - 2], cB[nB - 1], cG[0], cG[1]]),
                        m2, rep + 77, pin_exit=False)
        qF = torch.cat([q1f, qB, q2f], 0)
        fF = fk(qF)
        gA = fk(q_of_codes(cG[:1]))[0]
        sC = step(fC)
        sS = step(fS)
        rows.append({
            "clean_arrive": float((fC[-1] - gA).norm(dim=-1).mean()),
            "seq_arrive": float((fS[-1] - gA).norm(dim=-1).mean()),
            "seqF_arrive": float((fF[-1] - gA).norm(dim=-1).mean()),
            "clean_maxstep": float(sC.max()),
            "seam_in": float(sS[n1 - 2:n1 + 2].max()),
            "seam_out": float(sS[n1 + nB - 2:n1 + nB + 2].max()),
            "seq_maxstep": float(sS.max()),
            "clean_jerk": jerk(fC), "seq_jerk": jerk(fS), "seqF_jerk": jerk(fF),
            "m1": m1, "m2": m2, "fam": fam})
        print(f"[rep {rep}] {fam:6s} arrive clean {rows[-1]['clean_arrive']:.2f} "
              f"seq {rows[-1]['seq_arrive']:.2f} seqF {rows[-1]['seqF_arrive']:.2f} | "
              f"seam-in {rows[-1]['seam_in']:.2f} seam-out "
              f"{rows[-1]['seam_out']:.2f} vs clean-max "
              f"{rows[-1]['clean_maxstep']:.2f} | n picked {m1}/{m2}", flush=True)
        if rep == 0 and args.video:
            vid = (qC, qS, qF, n1, nB, m1, fam)

    def M(k):
        return float(np.mean([r[k] for r in rows]))
    print(f"\n{'metric':16s} {'value':>8s}")
    for k in ("clean_arrive", "seq_arrive", "seqF_arrive", "clean_maxstep",
              "seam_in", "seam_out", "seq_maxstep", "clean_jerk", "seq_jerk",
              "seqF_jerk"):
        print(f"{k:16s} {M(k):8.2f}")
    print("n picked (in/out):", [(r["m1"], r["m2"]) for r in rows])

    if vid:
        qC, qS, qF, n1_, nB_, m1_, fam = vid
        spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
        g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
        dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
        rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
        tiles = []
        for q in (qC, qS, qF):
            zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
            rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
            Rg = rot6d_to_matrix(rot).double()
            _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
            gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                       root_R=Rg[:, 0])
            tiles.append(GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                                        gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                                        qj.shape[0], status="nodataset"))
        labs = ["CLEAN  A_curr -> A_target",
                f"SEQ  A->B_init | REAL {fam} | B_target->A  (n given)",
                f"SEQ  same, budgets chosen by the model (n={m1_})"]
        L = max(len(t) for t in tiles)
        frames = []
        for t_ in range(L):
            row = np.concatenate([pad_band(ti[min(t_, len(ti) - 1)])
                                  for ti in tiles], 1)
            im = Image.fromarray(row).convert("RGB")
            dr = ImageDraw.Draw(im)
            for k_, lb in enumerate(labs):
                dr.text((k_ * GC.TILE_W + 8, 6), lb, font=font(11),
                        fill=(255, 235, 170))
            for k_, (a, b) in enumerate([(0, L), (n1_, n1_ + nB_),
                                         (m1_, m1_ + nB_)]):
                col = (120, 200, 255) if not (a <= t_ < b) else (255, 120, 120)
                if k_:
                    dr.rectangle([k_ * GC.TILE_W, 0, (k_ + 1) * GC.TILE_W, 5],
                                 fill=col)
            dr.text((8, GC.TILE_H + 24),
                    "red bar = the REAL disturbance segment; the model builds "
                    "both seams itself and must still reach A_target",
                    font=font(12, False), fill=(210, 210, 215))
            frames.append(np.asarray(im))
        p = f"{VIZ}/mp4/bridge_demo/disturb_seq_x2.mp4"
        RN.write_mp4(p, np.stack(frames), 20)
        print("->", p, flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].bar(["clean", "A-B-A\n(n given)", "A-B-A\n(n chosen)"],
              [M("clean_arrive"), M("seq_arrive"), M("seqF_arrive")],
              color=["#3a6", "#c33", "#83c"])
    ax[0].set_ylabel("distance to A_target (cm)")
    ax[0].set_title("does it still get back to the original goal?")
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(["clean\nmax step", "seam IN", "seam OUT"],
              [M("clean_maxstep"), M("seam_in"), M("seam_out")],
              color=["#3a6", "#c33", "#c83"])
    ax[1].axhline(M("clean_maxstep"), color="k", ls="--", lw=1,
                  label="what a clean motion already does")
    ax[1].set_ylabel("max single-frame joint step (cm)")
    ax[1].set_title("are the seams continuous?")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    ax[2].bar(["clean", "A-B-A\n(n given)", "A-B-A\n(n chosen)"],
              [M("clean_jerk"), M("seq_jerk"), M("seqF_jerk")],
              color=["#3a6", "#c33", "#83c"])
    ax[2].set_ylabel("jerk (cm/frame^2)")
    ax[2].set_title("smoothness of the whole composed motion")
    ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("DISTURBANCE AS A SEGMENT — A_curr -> B_init [real B] "
                 "B_target -> A_target", fontsize=12)
    fig.tight_layout()
    pp = f"{VIZ}/png/disturb_seq.png"
    fig.savefig(pp, dpi=150); plt.close(fig)
    print("->", pp, flush=True)
    json.dump({k: M(k) for k in rows[0] if k not in ("fam",)} |
              {"picked": [(r["m1"], r["m2"]) for r in rows]},
              open(f"{RUNS}/DISTURB_SEQ_0812.json", "w"), indent=2)


if __name__ == "__main__":
    main()
