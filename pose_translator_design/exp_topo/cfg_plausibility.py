"""Is the CFG=8 'composition' a real motion or a guidance artefact?

    cfg_plausibility.py

At w=8 the composite column lit up both kinematic signatures, but the retrieval
judge disagreed — and two checks disagreeing is the project's standing bug
signal.  The obvious way for the kinematic judge to be fooled is guidance
pushing the sample somewhere extreme: arms flung overhead read as "hands close
together, high, oscillating" without any clapping happening.

So measure plausibility, not semantics, against two references that need no GT:
  * jerk    third difference of FK joint positions -> smoothness
  * limits  fraction of joint angles at their mechanical stop
  * REAL    the same numbers on real retargeted corpus windows = the ceiling
If w=8 sits near the corpus, the composition survives; if it is far out, the
kinematic verdict at w=8 is measuring distortion.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
from bridge_a import CodeData, load_bridge_a                         # noqa: E402
from manifold_project import _limits, fit_angles                     # noqa: E402
from bridge_b5 import BridgeB5, Critic                               # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from composition_demo import SETS, encode_prompts, sample_tokens     # noqa: E402

RUNS = V.RUNS
X2 = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
dev, T, NS = "cuda", 90, 12


@torch.no_grad()
def main():
    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    mb = BridgeB5(ep_dim=ma.d).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B5_xattn_0814/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    spec, dof, rest, qn, xml = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec, dof, dev, sem_encoder=spcfg.get("sem_encoder"))
    dof_d = torch.as_tensor(dof, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)
    PROMPTS, LABELS = SETS["runclap"]
    seqs, lens, pooled = encode_prompts(PROMPTS, dev)

    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    s0, _ = kd.sample_starts(1, np.random.default_rng(0), "val", span=4,
                             families=["walk"])
    r0 = int(s0[0])
    cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + 2])).long()
    pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + 2])).long()
    rr = torch.empty_like(cp)
    rr[..., 0::2] = pk & 0x0F; rr[..., 1::2] = (pk >> 4) & 0x0F
    c01 = torch.cat([cp, rr], 1)
    ep_c = torch.stack([c01[0], c01[1], c01[0], c01[1]])[None].to(dev)
    ep = ma.endpoints(ep_c)
    nf = torch.full((1,), float(T), device=dev)
    gd1 = torch.zeros(1, dtype=torch.bool, device=dev)      # goal kept
    lo_, hi_, limited_ = _limits(dof_d, 1.0)   # limit_saturation.py's contract
    span_ = (hi_ - lo_).clamp(min=1e-6)

    def metrics(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        qa, _ = fit_angles(Rg, spec, dof_d, rest=rest_d, clamp=True,
                           method="greedy")
        _, gp = MP.fk_from_angles(qa, spec, dof_d, rest=rest_d, root_R=Rg[:, 0])
        p = gp.cpu().numpy()
        jerk = np.abs(np.diff(p, 3, axis=0)).mean()
        near = (((qa - lo_[None]) / span_[None] < 0.02) |
                ((hi_[None] - qa) / span_[None] < 0.02)) & limited_[None]
        sat = float(near[:, limited_].float().mean()) if limited_.any() else 0.0
        return jerk, sat

    def q_of(tk):
        z = ma.fsq.from_ints(((tk[:, None] // W) % 5)[None].float())[None]
        prob = F.softmax(ma.decode(z, ep, T)[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (prob * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5
        return q

    def fn(i, w):
        td0 = torch.ones(1, dtype=torch.bool, device=dev)
        tdT = torch.zeros(1, dtype=torch.bool, device=dev)
        f_t = lambda tk, mk: mb(tk, mk, ep, nf, seqs[i:i+1], lens[i:i+1],
                                goal_drop=gd1, text_drop=tdT)
        if w == 1.0:
            return f_t
        f_0 = lambda tk, mk: mb(tk, mk, ep, nf, seqs[i:i+1], lens[i:i+1],
                                goal_drop=gd1, text_drop=td0)
        return lambda tk, mk: f_0(tk, mk) + w * (f_t(tk, mk) - f_0(tk, mk))

    # reference ceiling: real retargeted corpus windows
    g = np.random.default_rng(3)
    rj, rs = [], []
    for _ in range(NS):
        c, _n = data.sample(1, g, "val")
        a, b = metrics((c[0, :T].float().to(dev) - 4.0) / 3.5)
        rj.append(a); rs.append(b)
    print(f"{'':<22}{'jerk':>10}{'限位命中':>12}")
    print(f"{'真实语料(参考上限)':<22}{np.mean(rj):10.4f}{np.mean(rs)*100:11.1f}%")
    out = {"corpus": [float(np.mean(rj)), float(np.mean(rs))]}
    for w in (1.0, 2.0, 4.0, 8.0, 12.0):
        js, ss = [], []
        for i in (0, 1, 2):                    # run / clap / composite
            for s in range(NS // 3):
                tk = sample_tokens(fn(i, w), dev, temp=0.9, seed=91 * i + s)
                a, b = metrics(q_of(tk))
                js.append(a); ss.append(b)
        out[f"w{w}"] = [float(np.mean(js)), float(np.mean(ss))]
        flag = "  <-- 超出语料" if np.mean(js) > 1.5 * np.mean(rj) else ""
        print(f"{'CFG w=' + str(w):<22}{np.mean(js):10.4f}"
              f"{np.mean(ss)*100:11.1f}%{flag}")
    json.dump(out, open(f"{RUNS}/CFG_PLAUSIBILITY_0814.json", "w"), indent=2)
    print(f"-> {RUNS}/CFG_PLAUSIBILITY_0814.json")


if __name__ == "__main__":
    main()
