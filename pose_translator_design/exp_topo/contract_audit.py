"""CONTRACT AUDIT — A3 was trained with a time-local mask; B3's tokens were not.

    contract_audit.py

bridge_a.encode() applies the time-local attention mask (token i sees only clock
segment i). bridge_b.tokens_of() calls ma.enc(seq) with NO mask. So the tokens
B3 was trained on are NOT the tokens A3's own encoder produces, and A3's decoder
was trained on z from the masked path. Measure the damage before deciding what
to retrain.
"""
import os, sys, json
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from bridge_a import CodeData, load_bridge_a
from bridge_b import BridgeB, tokens_of

RUNS, dev = V.RUNS, "cuda"
ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt", map_location=dev)); mb.eval()
data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
W = 5 ** torch.arange(6, device=dev)

def masked_tokens(c):
    """what A3.encode() actually produces (its training contract)."""
    z = ma.encode(c)                       # applies time_local mask internally
    ints = ma.fsq.ints(ma.fsq.features(z))[0] if False else None
    # encode() returns post-FSQ z; get ints from the same path A3 uses
    h = ma.embed_frames(c); B, T = h.shape[0], h.shape[1]
    seq = torch.cat([ma.q[None].expand(B, -1, -1), h], 1)
    mask = None
    if ma.time_local:
        K, L = ma.n_content, ma.n_content + T
        mask = torch.zeros(L, L, dtype=torch.bool, device=dev)
        b = np.round(np.linspace(0, T, K + 1)).astype(int)
        for i in range(K):
            mask[i, K:] = True
            mask[i, K + b[i]:K + max(b[i+1], b[i]+1)] = False
            mask[i, :K] = False
    out = ma.enc(seq, mask=mask)[:, :ma.n_content]
    ints = ma.fsq.ints(ma.fsq.features(out))[0]
    return (ints * W).sum(-1)

g = np.random.default_rng(0)
pool = [(a, b) for sp in ("val",) for a, b in data.pools[sp] if b - a >= 62]
rows = [int(g.integers(a, b - 60)) for a, b in (pool[i] for i in g.choice(len(pool), 64, replace=False))]

agree, nll_m, nll_u, ints_l1 = [], [], [], []
with torch.no_grad():
    for r0 in rows:
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0+60])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0+60])).long()
        rr = torch.empty_like(cp); rr[...,0::2] = pk & 15; rr[...,1::2] = (pk >> 4) & 15
        c = torch.cat([cp, rr], 1)[None].to(dev)
        t_u, ep = tokens_of(ma, c)                   # B3's contract (no mask)
        t_m = masked_tokens(c)                       # A3's contract (masked)
        agree.append(float((t_u == t_m).float().mean()))
        # per-dimension L1 on the 6 FSQ levels (a 1-level shift is not a "miss")
        du = ((t_u[..., None] // W) % 5).float(); dm = ((t_m[..., None] // W) % 5).float()
        ints_l1.append(float((du - dm).abs().mean()))
        nf = torch.full((1,), 60.0, device=dev)
        for t, acc in ((t_u, nll_u), (t_m, nll_m)):
            mk = torch.ones_like(t, dtype=torch.bool)
            acc.append(float(F.cross_entropy(mb(t, mk, ep, nf)[mk], t[mk])))
print(f"token agreement  masked vs unmasked : {np.mean(agree):.4f}  (1.0 = same contract)")
print(f"mean |level diff| per FSQ dim       : {np.mean(ints_l1):.3f}  (levels are 0..4)")
print(f"B3 masked-CE on ITS OWN tokens      : {np.mean(nll_u):.3f} nats")
print(f"B3 masked-CE on A3-contract tokens  : {np.mean(nll_m):.3f} nats")
print(f"penalty for feeding the wrong ones  : {np.mean(nll_m)-np.mean(nll_u):+.3f} nats")
json.dump({"agree": float(np.mean(agree)), "l1": float(np.mean(ints_l1)),
           "nll_unmasked": float(np.mean(nll_u)), "nll_masked": float(np.mean(nll_m))},
          open(f"{RUNS}/CONTRACT_AUDIT_0812.json", "w"), indent=2)
