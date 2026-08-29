"""JOINT-LIMIT SATURATION — do the hard families sit on the robots' limits?

    limit_saturation.py

The last untested explanation for the arm tail. If the paired GT pose for a
family lands on the joint stops, the label is a CLAMPED approximation of a
motion the body cannot perform, and no model and no extra data can recover what
was clamped away. Uses manifold_project's own _limits() so the margins are
exactly the ones the projection uses.
Reference: the easy families, on the same bodies, same windows.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
import manifold_project as MP
from manifold_project import _limits, fit_angles
from dual_common import load_paired_data
from common import rot6d_to_matrix
RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
cfg = json.load(open(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg["cache"], dev, sem_encoder=cfg["sem_encoder"])
kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
ROBOTS = ["unitree_g1_29dof", "unitree_h1", "booster_t1", "hightorque_hi", "kuavo_s45"]
FAMS = ["run", "dance", "walk", "punch", "jacks", "kick"]
T, NW = 60, 6
g = np.random.default_rng(0)
starts = {f: [int(x) for x in kd.sample_starts(NW, g, "val", span=T+2, families=[f])[0].tolist()]
          for f in FAMS}
print(f"{'family':>8s} " + " ".join(f"{r.split('_')[-1][:7]:>8s}" for r in ROBOTS) + f" {'mean':>7s}")
tab = {}
with torch.no_grad():
    for f in FAMS:
        row = []
        for tgt in ROBOTS:
            spec = data[tgt]["spec"]
            dofn = np.load(f"{os.environ['V5_AMASS']}/{tgt}_dof.npz", allow_pickle=True)["dof"]
            dof = torch.as_tensor(dofn, device=dev, dtype=torch.float64)
            rest = torch.as_tensor(np.load(f"{os.environ['V5_AMASS']}/{tgt}_dof.npz",
                                   allow_pickle=True)["rest"], device=dev, dtype=torch.float64)
            lo, hi, limited = _limits(dof, 1.0)
            fr = []
            for r0 in starts[f]:
                rows = torch.arange(r0, r0+T)
                Rg = rot6d_to_matrix(data[tgt]["pose"][rows][..., :6]).double()
                q, _ = fit_angles(Rg, spec, dof, rest=rest, clamp=True, method="greedy")
                span = (hi - lo).clamp(min=1e-6)
                near = (((q - lo[None]) / span[None] < 0.02) |
                        ((hi[None] - q) / span[None] < 0.02)) & limited[None]
                fr.append(float(near[:, limited].float().mean()) if limited.any() else 0.0)
            row.append(float(np.mean(fr)) * 100)
        tab[f] = row
        print(f"{f:>8s} " + " ".join(f"{v:8.2f}" for v in row) + f" {np.mean(row):7.2f}")
hard, easy = ["kick", "jacks", "punch"], ["run", "dance", "walk"]
h = np.mean([np.mean(tab[f]) for f in hard]); e = np.mean([np.mean(tab[f]) for f in easy])
print(f"\n== % of limited DOFs sitting within 2% of a joint stop ==")
print(f"   hard families {h:.2f}%   easy families {e:.2f}%   ratio {h/max(e,1e-9):.2f}")
print(f"   jacks on g1_29dof {tab['jacks'][0]:.2f}%  vs on hightorque_hi "
      f"{tab['jacks'][3]:.2f}%  (the 16x error cell)")
json.dump(tab, open(f"{RUNS}/LIMIT_SATURATION_0813.json", "w"), indent=2)
