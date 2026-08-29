"""The two levers, measured: B-temperature/top-p (diversity) x A-sharpening (energy).

    bridge_knob_sweep.py

Grid: tau_B in {0.9, 1.1, 1.3} x tau_A in {1.0, 0.7, 0.5}, plus top-p 0.92 at the
best tau_B. All zero-training. Clean G1 ruler, 16 val windows x 3 samples each.
Output: runs/KNOB_SWEEP_0810.txt — pick on the diversity/energy/jerk pareto.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
import eval_newbody as E
from bridge_a import BridgeA, CodeData, load_bridge_a, FULL
from bridge_b import BridgeB
from train_global import build_global

RUNS = V.RUNS


def sample_topp(mb, ep, nf, dev, temp, top_p, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
    mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
    for _ in range(32):
        logits = mb(tok, mask, ep, nf)
        probs = F.softmax(logits[0] / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask[0], -1)
        pos = int(conf.argmax())
        p = probs[pos]
        if top_p < 1.0:
            sp_, idx = torch.sort(p, descending=True)
            keep = torch.cumsum(sp_, 0) <= top_p
            keep[0] = True
            p = torch.zeros_like(p).scatter_(0, idx[keep], sp_[keep])
            p = p / p.sum()
        tok[0, pos] = torch.multinomial(p, 1, generator=g)
        mask[0, pos] = False
    return tok[0]


@torch.no_grad()
def main():
    dev = "cuda"
    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    G1 = "unitree_g1_29dof"
    spec_g = V.load_spec(FULL, G1)
    gdof = np.load(f"{FULL}/{G1}_dof.npz")["dof"]
    gg = E.target_geom(spec_g, gdof, dev, sem_encoder=enc)
    l6g = torch.load(f"{FULL}/{G1}_paired_local6d.pt", mmap=True)
    names_g = [n.lower() for n in spec_g.joint_names]
    anc = [i for i, n in enumerate(names_g)
           if any(k in n for k in ("wrist", "ankle_roll", "head"))]
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    T = 60

    def win(g):
        pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]
        a0, b0 = pool[int(g.integers(len(pool)))]
        s0 = int(g.integers(a0, b0 - T + 1))
        cp = torch.from_numpy(np.ascontiguousarray(data.c[s0:s0 + T])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[s0:s0 + T])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)[None].to(dev), np.arange(s0, s0 + T)

    def gen(c, tb, ta, pp, seed):
        ep = ma.endpoints(c)
        nf = torch.full((1,), float(T), device=dev)
        tok = sample_topp(mb, ep, nf, dev, tb, pp, seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, T)
        p = F.softmax(logits[0] / ta, -1)
        q = (p * lv).sum(-1)
        q[0] = (c[0, 0].float() - 4.0) / 3.5
        q[-1] = (c[0, -1].float() - 4.0) / 3.5
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(gg, T))
        return V.fk_pos(rot.cpu(), spec_g)

    def vel(p): return np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)
    def jerk(p):
        return np.linalg.norm(p[3:] - 3*p[2:-1] + 3*p[1:-2] - p[:-3], axis=-1).mean()

    CONFIGS = [(1.6, 1.0, 1.0), (2.0, 1.0, 1.0), (1.6, 1.0, 0.95)]
    L = ["=" * 88,
         "KNOB SWEEP — tau_B (B sampling) x tau_A (A sharpening) x top-p | G1 ruler",
         f"{'tauB':>5} {'tauA':>5} {'topP':>5} | {'divers':>7} {'energy':>7} "
         f"{'jerk':>6} {'ancFK':>7}", "=" * 88]
    for tb, ta, pp in CONFIGS:
        g = np.random.default_rng(5)
        div, en, jk, anc_e = [], [], [], []
        for wi in range(16):
            c, rows = win(g)
            gr6g, _, _ = build_global(l6g[torch.as_tensor(rows)], spec_g, dev)
            gt = V.fk_pos(gr6g, spec_g)
            fks = [gen(c, tb, ta, pp, 100 * wi + sd) for sd in range(3)]
            for fk in fks[:2]:
                anc_e.append(float(np.linalg.norm(fk[:, anc] - gt[:, anc],
                                                  axis=-1).mean()))
                en.append(float(vel(fk[:, anc]).mean() /
                                (vel(gt[:, anc]).mean() + 1e-9)))
                jk.append(float(jerk(fk[:, anc]) / (jerk(gt[:, anc]) + 1e-9)))
            pw = [float(np.linalg.norm(fks[i][:, anc] - fks[j][:, anc],
                                       axis=-1).mean())
                  for i in range(3) for j in range(i + 1, 3)]
            div.append(float(np.mean(pw)))
        L.append(f"{tb:>5} {ta:>5} {pp:>5} | {np.mean(div):>7.2f} "
                 f"{np.mean(en):>7.2f} {np.mean(jk):>6.2f} {np.mean(anc_e):>7.2f}")
        print(L[-1], flush=True)
    txt = "\n".join(L)
    open(f"{RUNS}/KNOB_SWEEP_0810.txt", "w").write(txt + "\n")
    print(f"verdict: {RUNS}/KNOB_SWEEP_0810.txt", flush=True)


if __name__ == "__main__":
    main()
