"""ENCODER OR DECODER? — split the blame for the embodiment-chain collapse.

    blame_split.py

The chain test showed code agreement 0.605 after one hop (floor 0.567). Two
candidate causes, separated here using the PAIRED ground truth we already have
(the same motion recorded on every body):

  A encoder invariance   ints(GT human)  vs  ints(GT body X)
        if low, the encoder never produced an embodiment-invariant code and the
        shared-latent premise is weaker than assumed.
  B decoder fidelity     ints(GT body X) vs  ints(decode(z_human -> X))
        if low, the decoder emits poses that are off-manifold for the encoder.
  C the chain number     ints(GT human)  vs  ints(decode(z_human -> X))
Controls: ceiling = same pose encoded twice; floor = an unrelated motion.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from dual_common import load_paired_data, split_indices, build_model, geom

RUNS = V.RUNS
dev = "cuda"
RUN = os.environ.get("SPATIAL_RUN", "PROD_c1_91k_dim20_tail_0810")
cfg = json.load(open(f"{RUNS}/{RUN}/config.json"))
torch.manual_seed(0)
data, names, NR = load_paired_data(cfg["cache"], dev, sem_encoder=cfg["sem_encoder"])
val_idx, _ = split_indices(NR)
sp = build_model(cfg, dev)
sp.load_state_dict(torch.load(f"{RUNS}/{RUN}/model.pt",
                              map_location=dev)); sp.eval()

def ints_of(p9, name):
    """BUGFIX 0812: the previous version did fsqp.ints(fsqp.features(zp)) where
    zp was already encode()'s output, i.e. proj_in(proj_out(q)) — two stacked
    projections, not the code. proj_out sits on the DECODER side, so a
    decoder-only finetune silently changed that quantity while the real codes
    stayed bit-identical, and the two checks contradicted each other. The codes
    are ints_from_features(encode_features(.)), which is what the tokenizer
    cache uses."""
    g = geom(data, name, p9.shape[0])
    ip, ir = sp.ints_from_features(*sp.encode_features(p9, g))
    zp, zr = sp.encode(p9, g)
    return torch.cat([ip[0], ir[0]], 1), (zp, zr)

g = np.random.default_rng(3)
idx = val_idx[torch.as_tensor(g.choice(len(val_idx), 2048, replace=False))]
idx2 = val_idx[torch.as_tensor(g.choice(len(val_idx), 2048, replace=False))]
TARGETS = ["booster_t1", "unitree_h1", "fourier_gr3", "kuavo_s45"]
with torch.no_grad():
    cH, (zpH, zrH) = ints_of(data["human_smpl"]["pose"][idx], "human_smpl")
    cH2, _ = ints_of(data["human_smpl"]["pose"][idx], "human_smpl")
    cF, _ = ints_of(data["human_smpl"]["pose"][idx2], "human_smpl")
    print(f"ceiling (same pose twice) {float((cH2==cH).float().mean()):.3f}   "
          f"floor (unrelated motion) {float((cF==cH).float().mean()):.3f}\n")
    print(f"{'target':14s} {'A enc-invar':>12s} {'B dec-fidel':>12s} {'C chain':>9s}")
    out = {}
    for t in TARGETS:
        cR, _ = ints_of(data[t]["pose"][idx], t)               # real body-X pose
        gt = geom(data, t, len(idx))
        prot, ppos = sp.decode(zpH, zrH, gt)
        cD, _ = ints_of(torch.cat([prot, ppos], -1), t)        # decoded to X
        A = float((cR == cH).float().mean())
        B = float((cD == cR).float().mean())
        C = float((cD == cH).float().mean())
        out[t] = (A, B, C)
        print(f"{t:14s} {A:12.3f} {B:12.3f} {C:9.3f}")
print("run:", RUN)
json.dump(out, open(f"{RUNS}/BLAME_{RUN}.json", "w"), indent=2)
