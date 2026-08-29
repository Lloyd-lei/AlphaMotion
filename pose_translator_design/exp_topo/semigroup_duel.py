"""SEMIGROUP DUEL — B3 vs B5 on ONE yardstick, both metrics GT-free.

    semigroup_duel.py

B5's own training log is not comparable to B3's (fixed n=60 vs variable n), so
both models are re-measured here on identical batches:

  gap        one-hop P(60) vs the model's own two-hop P(30)oP(30), in rendered
             joint cm — the quantity B5 was trained to shrink
  noise      two independent one-hop samples of the SAME bridge — the floor
             any gap must be read against
  ratio      gap / noise   (B3 baseline measured 0811: 0.69)
  bridge CE  ordinary masked CE on real tokens, same batches — the thing that
             must NOT degrade
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
from bridge_a import CodeData, load_bridge_a, FULL                   # noqa: E402
from bridge_b import BridgeB, tokens_of                              # noqa: E402
from homecoming_probe import sample_nll                              # noqa: E402

RUNS = V.RUNS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="BRIDGE_B3_prod_0810,EQ_B5_semigroup_0811")
    ap.add_argument("--n_win", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    N, H = 60, 30

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
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

    def dec(tok, ep, n):
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        q = (F.softmax(ma.decode(z, ep, n), -1)[0] * lv).sum(-1)
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float(), q

    g = np.random.default_rng(11)
    pool = [(a, b) for a, b in data.pools["val"] if b - a >= N + 4]
    rows = [int(g.integers(a, b - N - 2)) for a, b in
            (pool[i] for i in g.choice(len(pool), args.n_win, replace=False))]

    print(f"{'run':28s} {'gap_cm':>8s} {'noise_cm':>9s} {'ratio':>7s} "
          f"{'bridgeCE':>9s} {'ppl':>7s}")
    out = {}
    for run in args.runs.split(","):
        cfg = json.load(open(f"{RUNS}/{run}/config.json"))
        m = BridgeB(ep_dim=cfg.get("ep_dim")).to(dev)
        m.load_state_dict(torch.load(f"{RUNS}/{run}/model.pt", map_location=dev))
        m.eval()
        gaps, noise, ces = [], [], []
        for i, r0 in enumerate(rows):
            c = codes_at(r0, N + 2)
            ep = ma.endpoints(torch.stack([c[0], c[1], c[N], c[N + 1]])[None].to(dev))
            nf = torch.full((1,), float(N), device=dev)
            ta, _ = sample_nll(m, ep, nf, dev, seed=i)
            tb, _ = sample_nll(m, ep, nf, dev, seed=5000 + i)
            fa, qa = dec(ta, ep, N)
            fb, _ = dec(tb, ep, N)
            noise.append(float((fa - fb).norm(dim=-1).mean()))
            midc = (qa[H - 1:H + 1] * 3.5 + 4.0).round().clamp(0, 8).long().cpu()
            ep1 = ma.endpoints(torch.stack([c[0], c[1], midc[0], midc[1]])[None].to(dev))
            ep2 = ma.endpoints(torch.stack([midc[0], midc[1], c[N], c[N + 1]])[None].to(dev))
            nfh = torch.full((1,), float(H), device=dev)
            t1, _ = sample_nll(m, ep1, nfh, dev, seed=i)
            t2, _ = sample_nll(m, ep2, nfh, dev, seed=i)
            f2 = torch.cat([dec(t1, ep1, H)[0], dec(t2, ep2, H)[0]], 0)
            gaps.append(float((fa - f2).norm(dim=-1).mean()))
            tok_r, ep_r = tokens_of(ma, c[None, :N].to(dev))
            mk = torch.ones_like(tok_r, dtype=torch.bool)
            ces.append(float(F.cross_entropy(m(tok_r, mk, ep_r, nf)[mk], tok_r[mk])))
        gp, nz, ce = np.mean(gaps), np.mean(noise), np.mean(ces)
        out[run] = {"gap": gp, "noise": nz, "ratio": gp / nz, "ce": ce}
        print(f"{run:28s} {gp:8.2f} {nz:9.2f} {gp/nz:7.2f} {ce:9.4f} "
              f"{np.exp(ce):7.1f}")
    json.dump(out, open(f"{RUNS}/SEMIGROUP_DUEL_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
