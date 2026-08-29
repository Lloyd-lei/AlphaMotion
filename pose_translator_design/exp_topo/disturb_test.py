"""DISTURBANCE TEST — shove the motion mid-flight and see whether it recovers.

    disturb_test.py

lloyd: external feedback does not need a simulator. A push, a stumble, a change
of mind can all be injected KINEMATICALLY as a hard mid-course constraint, and
the interesting question is what the model does next. Four scenarios, each with
the frame budget GIVEN and INFERRED:

  S1 clean      A -> B in n frames, no disturbance (reference)
  S2 shoved     A -> B, but the mid pose is pinned to a FOREIGN pose P (from a
                different action family). The goal is unchanged: the model must
                still arrive at B. This is re-planning under disturbance.
  S3 re-plan    after the shove, re-bridge from P to B as a fresh problem
  S4 goal swap  at the mid point the target itself changes B -> C (lloyd's
                adversarial moving-target idea, kinematic version)

n GIVEN    the remaining budget is supplied, as today.
n INFERRED the model picks its own budget: sweep n over a grid and keep the one
           its own prior likes best. That is an n-free interface built on top of
           the existing model, no retraining — and it tests lloyd's point that
           the target is the identity while n is only pacing.

Metrics (all GT-free except the S1 reference):
  arrive    distance from the produced final pose to the intended goal pose
  detour    peak deviation from the clean S1 trajectory
  recover   deviation at the end vs at the shove: does it come back?
  jerk      smoothness, against the corpus band
Figure: png/disturb_test.png   Video: mp4/bridge_demo/disturb_shove_x2.mp4
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
NGRID = [30, 45, 60, 75, 90, 105, 120]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, N = args.device, args.n
    H = N // 2

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

    def sample_pinned(ep, nf, seed, pin_tok=None, temp=0.9):
        """confidence-ordered sampling with some tokens PRE-SET and unmasked —
        the sparse-pin mode B was trained on. This is how a disturbance must be
        injected: as a constraint the model plans around, not as a pose spliced
        into its output afterwards (which is what the first version of this
        script did, and why its trajectories jumped at the seam)."""
        g_ = torch.Generator(device=dev).manual_seed(seed)
        tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
        mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
        if pin_tok:
            for j, v in pin_tok.items():
                tok[0, j] = int(v); mask[0, j] = False
        nll = 0.0
        for _ in range(int(mask.sum())):
            logits = mb(tok, mask, ep, nf)
            probs = F.softmax(logits[0] / temp, -1)
            conf = probs.max(-1).values.masked_fill(~mask[0], -1)
            pos = int(conf.argmax())
            pick = torch.multinomial(probs[pos], 1, generator=g_)
            nll -= float(F.log_softmax(logits[0, pos], -1)[pick])
            tok[0, pos] = pick; mask[0, pos] = False
        return tok[0], nll / 32.0

    def bridge(c4, n, seed, pins=None, pin_exit=False, pin_tok=None):
        """c4 [4,S,D] endpoints -> (q [n,S,D], nll). pins: {idx: code}.

        pin_exit defaults to FALSE here: the endpoint pass-through would make
        "did it arrive" trivially 0.00 (it did, by construction). Arrival is
        only a measurement if the last frame is the model's own."""
        ep = ma.endpoints(c4[None].to(dev))
        nf = torch.full((1,), float(n), device=dev)
        tok, nll = sample_pinned(ep, nf, seed, pin_tok=pin_tok)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        q = (F.softmax(ma.decode(z, ep, n), -1)[0] * lv).sum(-1)
        q[0] = (c4[1].float().to(dev) - 4.0) / 3.5
        if pin_exit:
            q[-1] = (c4[2].float().to(dev) - 4.0) / 3.5
        if pins:
            for i, code in pins.items():
                q[i] = (code.float().to(dev) - 4.0) / 3.5
        return q, nll

    def pick_n(c4, seed):
        """n-FREE: let the prior choose its own budget by chain NLL."""
        best = None
        for n_ in NGRID:
            ep = ma.endpoints(c4[None].to(dev))
            nf = torch.full((1,), float(n_), device=dev)
            _, nll = sample_nll(mb, ep, nf, dev, temp=0.9, seed=seed)
            if best is None or nll < best[1]:
                best = (n_, nll)
        return best

    def fk(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    def jerk(f):
        return float((f[2:] - 2 * f[1:-1] + f[:-2]).norm(dim=-1).mean())

    gg = np.random.default_rng(args.seed)
    rows = []
    picks = []
    for rep in range(args.n_rep):
        sa, _ = kd.sample_starts(1, gg, "val", span=N + 4, families=["walk"])
        sb, _ = kd.sample_starts(1, gg, "val", span=4, families=["walk"])
        sp_, _ = kd.sample_starts(1, gg, "val", span=4,
                                  families=["dance", "jump", "kick"][rep % 3:][:1])
        sc, _ = kd.sample_starts(1, gg, "val", span=4, families=["sit"])
        ca = codes_at(int(sa[0]), 2)
        cb = codes_at(int(sb[0]), 2)
        cp_ = codes_at(int(sp_[0]), 2)                     # the foreign shove
        cc = codes_at(int(sc[0]), 2)                       # the swapped goal
        c4 = torch.stack([ca[0], ca[1], cb[0], cb[1]])
        # S1 clean
        q1, nll1 = bridge(c4, N, seed=rep)
        f1 = fk(q1)
        # S2 shoved: the disturbance is a PINNED TOKEN in the conditioning, so
        # the model must plan around it. Token j covers timeline segment j.
        from bridge_b import tokens_of as _tok_of
        cf = codes_at(int(sp_[0]), N)
        tokF, _ = _tok_of(ma, cf[None].to(dev))
        jmid = int(round(H / N * 32))
        pin_tok = {jmid: int(tokF[0, jmid]),
                   min(31, jmid + 1): int(tokF[0, min(31, jmid + 1)])}
        q2, nll2 = bridge(c4, N, seed=rep, pin_tok=pin_tok)
        f2 = fk(q2)
        jump2 = float((f2[1:] - f2[:-1]).norm(dim=-1).mean(-1).max())
        jump1 = float((f1[1:] - f1[:-1]).norm(dim=-1).mean(-1).max())
        # S3 re-plan: fresh bridge from the shove pose to the same goal
        c4b = torch.stack([cp_[0], cp_[1], cb[0], cb[1]])
        q3, nll3 = bridge(c4b, N - H, seed=rep)
        f3 = fk(torch.cat([q2[:H], q3], 0))
        n_free, nll3f = pick_n(c4b, seed=rep)
        q3f, _ = bridge(c4b, n_free, seed=rep)
        f3f = fk(torch.cat([q2[:H], q3f], 0))
        # S4 goal swap
        c4c = torch.stack([q2[H - 1:H].round().long()[0] if False else cp_[0],
                           cp_[1], cc[0], cc[1]])
        q4, nll4 = bridge(c4c, N - H, seed=rep)
        f4 = fk(torch.cat([q2[:H], q4], 0))
        gB = fk((cb[:1].float().to(dev) - 4.0) / 3.5)[0]
        gC = fk((cc[:1].float().to(dev) - 4.0) / 3.5)[0]
        rows.append({
            "clean_arrive": float((f1[-1] - gB).norm(dim=-1).mean()),
            "shove_arrive": float((f2[-1] - gB).norm(dim=-1).mean()),
            "replan_arrive": float((f3[-1] - gB).norm(dim=-1).mean()),
            "replanF_arrive": float((f3f[-1] - gB).norm(dim=-1).mean()),
            "swap_arrive": float((f4[-1] - gC).norm(dim=-1).mean()),
            "clean_jerk": jerk(f1), "shove_jerk": jerk(f2),
            "replan_jerk": jerk(f3), "replanF_jerk": jerk(f3f),
            "detour": float((f2 - f1).norm(dim=-1).mean(-1).max()),
            "max_step_clean": jump1, "max_step_shoved": jump2,
            "end_dev": float((f2[-1] - f1[-1]).norm(dim=-1).mean()),
            "n_free": n_free})
        picks.append(n_free)
        if rep == 0 and args.video:
            picks_v = [("CLEAN  A->B", q1), ("SHOVED mid-pin, same goal", q2),
                       (f"RE-PLAN n={N-H} given", torch.cat([q2[:H], q3], 0)),
                       (f"RE-PLAN n={n_free} inferred",
                        torch.cat([q2[:H], q3f], 0))]
            spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
            g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
            dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
            rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
            tiles = []
            for _, q in picks_v:
                zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
                rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
                Rg = rot6d_to_matrix(rot).double()
                _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d,
                                      method="global")
                gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                           root_R=Rg[:, 0])
                tiles.append(GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                                            gR[:, 0].cpu().numpy(),
                                            gp.cpu().numpy(), qj.shape[0],
                                            status="nodataset"))
            L = max(len(t) for t in tiles)
            frames = []
            for t_ in range(L):
                row = np.concatenate([pad_band(ti[min(t_, len(ti) - 1)])
                                      for ti in tiles], 1)
                im = Image.fromarray(row).convert("RGB")
                dr = ImageDraw.Draw(im)
                for k_, (lb, _) in enumerate(picks_v):
                    dr.text((k_ * GC.TILE_W + 8, 6), lb, font=font(12),
                            fill=(255, 235, 170))
                    if t_ >= H:
                        dr.rectangle([k_ * GC.TILE_W, 0,
                                      (k_ + 1) * GC.TILE_W, 4],
                                     fill=(255, 90, 90) if k_ else (90, 200, 120))
                dr.text((8, GC.TILE_H + 24),
                        f"a foreign pose is pinned at frame {H} (red bar = after "
                        "the shove). goal unchanged; does it still arrive?",
                        font=font(12, False), fill=(210, 210, 215))
                frames.append(np.asarray(im))
            pv = f"{VIZ}/mp4/bridge_demo/disturb_shove_x2.mp4"
            RN.write_mp4(pv, np.stack(frames), 20)
            print("->", pv, flush=True)
        print(f"[rep {rep}] arrive clean {rows[-1]['clean_arrive']:.1f} "
              f"shove {rows[-1]['shove_arrive']:.1f} "
              f"replan {rows[-1]['replan_arrive']:.1f} "
              f"replan-nfree(n={n_free}) {rows[-1]['replanF_arrive']:.1f} "
              f"swap {rows[-1]['swap_arrive']:.1f}", flush=True)

    def M(k):
        return float(np.mean([r[k] for r in rows]))
    print(f"\n{'metric':22s} {'value':>8s}")
    for k in ("clean_arrive", "shove_arrive", "replan_arrive", "replanF_arrive",
              "swap_arrive", "clean_jerk", "shove_jerk", "replan_jerk",
              "replanF_jerk", "detour", "end_dev", "max_step_clean",
              "max_step_shoved"):
        print(f"{k:22s} {M(k):8.2f}")
    print(f"inferred n: {np.bincount(picks, minlength=130)[NGRID]} over {NGRID}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    labs = ["clean", "shoved\n(same goal)", "re-plan\n(n given)",
            "re-plan\n(n inferred)", "goal swap"]
    vals = [M("clean_arrive"), M("shove_arrive"), M("replan_arrive"),
            M("replanF_arrive"), M("swap_arrive")]
    ax[0].bar(labs, vals, color=["#3a6", "#c33", "#38c", "#83c", "#c83"])
    ax[0].set_ylabel("distance to the intended goal (cm)")
    ax[0].set_title("does it still arrive after being shoved?")
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(labs[:4], [M("clean_jerk"), M("shove_jerk"), M("replan_jerk"),
                         M("replanF_jerk")],
              color=["#3a6", "#c33", "#38c", "#83c"])
    ax[1].set_ylabel("jerk (cm/frame^2)")
    ax[1].set_title("is the recovery smooth?")
    ax[1].grid(alpha=0.3, axis="y")
    ax[2].hist(picks, bins=np.array(NGRID + [135]) - 7.5, color="#83c")
    ax[2].set_xlabel("n chosen by the prior itself")
    ax[2].set_ylabel("count")
    ax[2].set_title(f"n-FREE: the model's own pacing (given n was {N-H})")
    ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("DISTURBANCE — a foreign pose pinned mid-flight, goal unchanged",
                 fontsize=12)
    fig.tight_layout()
    p = f"{VIZ}/png/disturb_test.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)
    json.dump({k: M(k) for k in rows[0] if k != "n_free"} |
              {"inferred_n": picks},
              open(f"{RUNS}/DISTURB_TEST_0812.json", "w"), indent=2)


if __name__ == "__main__":
    main()
