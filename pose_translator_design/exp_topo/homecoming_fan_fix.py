"""Fan chart, corrected: the freedom ratio is RELATIVE, so each panel must
show BOTH fans — return paths (color) against open-loop paths (grey) from the
identical pose.  The ratio is the width contrast between them; a chart of
return paths alone cannot show it (and read backwards in the first attempt).

    homecoming_fan_fix.py
"""
from __future__ import annotations
import os, sys, json
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

RUNS = V.RUNS
PNG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "png")
PAIRS = [("jacks", "LOCKED"), ("dance", "LOCKED"), ("walk", "FREE"),
         ("kick", "FREE")]
N, NS = 45, 8


@torch.no_grad()
def main():
    dev = "cuda"
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
    jn = list(sspec.joint_names)
    JW = jn.index("R_Wrist")

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def fan(ep, seed0):
        hs, fks = [], []
        for s_ in range(NS):
            tk, _ = sample_nll(mb, ep, torch.full((1,), float(N), device=dev),
                               dev, seed=seed0 + s_)
            z = ma.fsq.from_ints(((tk[:, None] // W) % 5)[None].float())[None]
            q = (F.softmax(ma.decode(z, ep, N)[0], -1) * lv).sum(-1)
            zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
            rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
            f = torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()
            fks.append(f)
            hs.append(f[:, JW, 2].numpy())
        f = torch.stack(fks)
        spread = float((f - f.mean(0, keepdim=True)).norm(dim=-1).mean())
        return np.stack(hs), spread

    gg = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.1 * len(PAIRS), 4.4),
                             sharex=True)
    for ax, (fam, tag) in zip(axes, PAIRS):
        s, _ = kd.sample_starts(1, gg, "val", span=130, families=[fam])
        r0 = int(s[0])
        c = codes_at(r0, 130)
        eph = ma.endpoints(torch.stack([c[0], c[1], c[0], c[1]])[None].to(dev))
        epo = ma.endpoints(torch.stack([c[0], c[1], c[120], c[121]])[None].to(dev))
        ho, sh = fan(eph, 7)
        op, so = fan(epo, 7)
        for h in op:
            ax.plot(h, color="0.68", lw=1.2, zorder=1)
        for h in ho:
            ax.plot(h, color="#d33", lw=1.5, alpha=0.9, zorder=2)
        ax.set_title(f"{fam}  ratio={sh/max(so,1e-6):.2f}  ({tag})\n"
                     f"return spread {sh:.2f} vs open {so:.2f} cm", fontsize=10)
        ax.set_xlabel(f"frame (n={N})"); ax.grid(alpha=0.3)
        print(f"[{fam}] home {sh:.2f} open {so:.2f} ratio {sh/so:.2f}", flush=True)
    axes[0].set_ylabel("right-wrist height (cm)")
    fig.suptitle("RED = 8 return paths (curr->curr)   GREY = 8 open-loop paths "
                 "(same start, free goal).  Locked = red fan collapses INSIDE "
                 "a wide grey fan.", fontsize=11)
    fig.tight_layout()
    p = f"{PNG}/homecoming_fan_return_vs_open.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
