"""E2: are A2's tokens really TIME-LOCAL? Swap token i with a donor window's
token i and measure WHERE the decode changes. Diagonal-dominant damage matrix =
locality holds = Stage B's fwd/bwd masks mean what we think they mean.

    bridge_locality_probe.py --run BRIDGE_A2_full_0810
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_a import BridgeA, CodeData, load_bridge_a                               # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="BRIDGE_A2_full_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--n_win", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, 64

    m, cfg = load_bridge_a(f"{RUNS}/{args.run}", dev)
    data = CodeData(args.spatial, use_root=bool(cfg.get("use_root", False)))
    g = np.random.default_rng(9)
    pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]

    K = 32
    seg = np.round(np.linspace(0, T, K + 1)).astype(int)
    dmg = np.zeros((K, K))                               # [swapped token, segment]
    for w in range(args.n_win):
        def win():
            a0, b0 = pool[int(g.integers(len(pool)))]
            s0 = int(g.integers(a0, b0 - T + 1))
            cp = torch.from_numpy(np.ascontiguousarray(
                data.c[s0:s0 + T])).long()
            if data.use_root:
                pk = torch.from_numpy(np.ascontiguousarray(
                    data.cr[s0:s0 + T])).long()
                r = torch.empty_like(cp)
                r[..., 0::2] = pk & 0x0F
                r[..., 1::2] = (pk >> 4) & 0x0F
                cp = torch.cat([cp, r], 1)
            return cp[None].to(dev)
        c, cd = win(), win()
        z = m.encode(c); zd = m.encode(cd)
        ep = m.endpoints(c)
        base = m.decode(z, ep, T).argmax(-1)[0]
        base_acc = (base == c[0]).float().mean(dim=(1, 2))        # [T]
        for i in range(K):
            zi = z.clone(); zi[:, i] = zd[:, i]
            pred = m.decode(zi, ep, T).argmax(-1)[0]
            acc = (pred == c[0]).float().mean(dim=(1, 2))
            drop = (base_acc - acc).cpu().numpy()                 # [T]
            for s in range(K):
                dmg[i, s] += drop[seg[s]:max(seg[s + 1], seg[s] + 1)].mean()
    dmg /= args.n_win
    diag = np.trace(dmg) / K
    off = (dmg.sum() - np.trace(dmg)) / (K * K - K)
    # band: within +-1 segment
    band = np.mean([dmg[i, j] for i in range(K) for j in range(K)
                    if abs(i - j) <= 1])
    print(f"[{args.run}] token-swap damage: diag {diag:.4f}  band(+-1) {band:.4f}  "
          f"off-diag {off:.4f}  ratio diag/off {diag / max(off, 1e-9):.1f}x")
    np.save(f"{RUNS}/LOCALITY_{args.run}.npy", dmg)
    print(f"matrix -> {RUNS}/LOCALITY_{args.run}.npy")


if __name__ == "__main__":
    main()
