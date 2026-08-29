"""PHASE SPACE — treat the bridge as a FLOW, not as a static latent geometry.

    phase_space.py

lloyd, correctly: the object is a dynamical system, and (q, qdot) is already
there.  Two measurements that only make sense in phase space:

E1 PORTRAIT + POINCARE RETURN MAP
   Roll the model forward autonomously (goal DROPPED — B was trained with 10%
   goal dropout, so forward mode is a real autonomous flow, not interpolation),
   chunk after chunk.  Plot (x, xdot) for a joint coordinate; take the section
   xdot=0 with x increasing; the successive crossings x_n -> x_{n+1} are the
   return map.  |slope| < 1 at the fixed point = a STABLE limit cycle, i.e. the
   learned flow contracts back onto the gait.  Corpus trajectories go through
   the identical measurement as the reference.
   (This is what the homecoming experiment was already doing without saying so:
   a first-return map.  The freedom ratio was its spread.)

E2 FINITE-TIME LYAPUNOV EXPONENT
   Perturb the initial state by one lattice cell in random directions, roll
   forward with the SAME sampling seed, and track the separation.  The slope of
   log|delta| against time is lambda: positive = the flow amplifies error
   (ballistic / open-loop), negative = it rejects error (closed-loop).  This is
   the test that distinguishes the two synergies, and unlike a spread statistic
   it cannot be faked by data sparsity.

Figures: png/phase_portrait.png, png/phase_lyapunov.png
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
from train_global import build_global                                # noqa: E402

RUNS = V.RUNS
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")


def sample_fwd(mb, ep, nf, dev, temp=0.9, seed=0):
    """goal-DROPPED sampling = autonomous forward flow."""
    g = torch.Generator(device=dev).manual_seed(seed)
    gd = torch.ones(1, dtype=torch.bool, device=dev)
    tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
    mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
    for _ in range(32):
        logits = mb(tok, mask, ep, nf, goal_drop=gd)
        probs = F.softmax(logits[0] / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask[0], -1)
        pos = int(conf.argmax())
        tok[0, pos] = torch.multinomial(probs[pos], 1, generator=g)
        mask[0, pos] = False
    return tok[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=45)
    ap.add_argument("--n_chunk", type=int, default=4)
    ap.add_argument("--n_pert", type=int, default=10)
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, C = args.device, args.chunk

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    print(f"[stage_b] {args.stage_b}", flush=True)
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    print(f"[spatial] {args.spatial}", flush=True)
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=spcfg.get("sem_encoder"))
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    jn = list(sspec.joint_names)
    JIDX = jn.index("R_Ankle")                       # gait coordinate

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def decode(tok, ep, n):
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        q = (F.softmax(ma.decode(z, ep, n), -1)[0] * lv).sum(-1)
        return q

    def q_to_fk(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    def rollout(c2, seed, n_chunk=None):
        """autonomous forward flow from a 2-frame state -> FK [T,J,3]."""
        cur = c2.clone()
        out = []
        for k in range(n_chunk or args.n_chunk):
            ep = ma.endpoints(torch.stack([cur[0], cur[1], cur[0], cur[1]]
                                          )[None].to(dev))
            nf = torch.full((1,), float(C), device=dev)
            tok = sample_fwd(mb, ep, nf, dev, seed=seed * 100 + k)
            q = decode(tok, ep, C)
            out.append(q_to_fk(q))
            cur = (q[-2:] * 3.5 + 4.0).round().clamp(0, 8).long().cpu()
        return torch.cat(out, 0)

    # ---------------- E1: portrait + return map ----------------
    gg = np.random.default_rng(args.seed)
    s, _ = kd.sample_starts(1, gg, "val", span=260, families=["walk"])
    r0 = int(s[0])
    c2 = codes_at(r0, 2)
    fk_model = rollout(c2, seed=7)
    g6, _, _ = build_global(l6h[torch.as_tensor(np.arange(r0, r0 + 240))],
                            sspec, dev)
    fk_real = torch.as_tensor(V.fk_pos(g6, sspec)).float()

    def portrait(fk):
        x = (fk[:, JIDX] - fk[:, 0])[:, 1].numpy()       # fore-aft ankle
        v = np.gradient(x)
        return x, v

    def return_map(x, v):
        """crossings of v=0 with v going + -> -, i.e. local maxima of x."""
        idx = [i for i in range(1, len(v) - 1) if v[i - 1] > 0 >= v[i]]
        xs = [x[i] for i in idx]
        return np.array(xs[:-1]), np.array(xs[1:])

    xr, vr = portrait(fk_real)
    xm, vm = portrait(fk_model)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(xr, vr, lw=1.2, color="#356", label="real corpus walk")
    ax[0].set_xlabel("ankle fore-aft x (cm)"); ax[0].set_ylabel("dx/dt")
    ax[0].set_title("phase portrait — corpus"); ax[0].grid(alpha=0.3)
    ax[1].plot(xm, vm, lw=1.2, color="#c33", label="model, goal dropped")
    ax[1].set_xlabel("ankle fore-aft x (cm)")
    ax[1].set_title(f"phase portrait — autonomous rollout "
                    f"({args.n_chunk}x{C} frames)")
    ax[1].grid(alpha=0.3)
    slopes = {}
    for lab, (x, v), col in (("corpus", (xr, vr), "#356"),
                             ("model", (xm, vm), "#c33")):
        a, b = return_map(x, v)
        if len(a) >= 3:
            k_, c_ = np.polyfit(a, b, 1)
            slopes[lab] = float(k_)
            xs = np.linspace(a.min(), a.max(), 10)
            ax[2].plot(xs, k_ * xs + c_, color=col, lw=1.5)
        ax[2].scatter(a, b, s=46, color=col, label=f"{lab} (n={len(a)})")
    lim = ax[2].get_xlim()
    ax[2].plot(lim, lim, "k:", lw=1, label="identity")
    ax[2].set_xlabel("x_n at section"); ax[2].set_ylabel("x_{n+1}")
    ax[2].set_title("POINCARE RETURN MAP\n|slope|<1 = stable limit cycle")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
    fig.suptitle("PHASE SPACE — the bridge run as an autonomous flow "
                 f"(return-map slope: {slopes})", fontsize=11)
    fig.tight_layout()
    p1 = f"{VIZ}/png/phase_portrait{args.tag}.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"[E1] return-map slopes {slopes}", flush=True)
    print("->", p1, flush=True)

    # ---------------- E2: finite-time Lyapunov ----------------
    fams = ["walk", "jacks", "sit"]
    curves = {}
    for fam in fams:
        s, _ = kd.sample_starts(1, gg, "val", span=260, families=[fam])
        base_c = codes_at(int(s[0]), 2)
        ref = rollout(base_c, seed=11, n_chunk=3)
        seps = []
        rng = np.random.default_rng(0)
        for p in range(args.n_pert):
            c = base_c.clone()
            flat = c.reshape(-1)
            pick = rng.choice(len(flat), 6, replace=False)
            flat[pick] = (flat[pick] + torch.as_tensor(
                rng.choice([-1, 1], 6))).clamp(0, 8)
            pert = rollout(flat.reshape(c.shape), seed=11, n_chunk=3)
            T = min(len(ref), len(pert))
            seps.append((ref[:T] - pert[:T]).norm(dim=-1).mean(-1).numpy())
        S = np.stack(seps)
        curves[fam] = S
        t = np.arange(S.shape[1])
        m = S.mean(0)
        lo, hi = 5, min(80, len(m) - 1)
        lam = float(np.polyfit(t[lo:hi], np.log(m[lo:hi] + 1e-6), 1)[0])
        print(f"[E2] {fam}: lambda={lam:+.4f}/frame  sep0={m[0]:.2f} "
              f"sep_end={m[-1]:.2f} cm", flush=True)
        curves[fam + "_lam"] = lam

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    cols = {"walk": "#356", "jacks": "#c33", "sit": "#3a3"}
    for fam in fams:
        S = curves[fam]
        m = S.mean(0)
        t = np.arange(len(m))
        ax[0].plot(t, m, lw=2, color=cols[fam],
                   label=f"{fam}  lambda={curves[fam+'_lam']:+.3f}")
        ax[0].fill_between(t, np.quantile(S, .25, 0), np.quantile(S, .75, 0),
                           color=cols[fam], alpha=0.18)
        ax[1].semilogy(t, m, lw=2, color=cols[fam])
    ax[0].set_xlabel("frame after a 1-cell perturbation of the initial state")
    ax[0].set_ylabel("separation (cm)")
    ax[0].set_title("does the learned flow reject or amplify error?")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("separation (cm, log)")
    ax[1].set_title("log scale: slope = finite-time Lyapunov exponent")
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("FINITE-TIME LYAPUNOV — lambda>0 ballistic/open-loop, "
                 "lambda<0 error-rejecting/closed-loop", fontsize=11)
    fig.tight_layout()
    p2 = f"{VIZ}/png/phase_lyapunov{args.tag}.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    print("->", p2, flush=True)
    json.dump({"return_map_slopes": slopes,
               "lambda": {f: curves[f + "_lam"] for f in fams}},
              open(f"{RUNS}/PHASE_SPACE_0811{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
