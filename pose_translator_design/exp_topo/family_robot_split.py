"""RARITY OR INFEASIBILITY? — separate the two explanations for the hard families.

    family_robot_split.py

corr(family frequency, arm error) = -0.684 says rare families are worse, but
rare families are rare BECAUSE they are extreme, and extreme is also what a
robot's joints cannot reach. The two explanations are confounded in the data.

Separation: run the SAME families through several robots. A DATA problem is a
property of the family and should hurt every body roughly equally; an
INFEASIBILITY problem is a property of the body-family pair and should vary
with how much reach/limit headroom each robot has.
Statistic: for each family, the spread of its error ACROSS robots, normalised
by that family's mean error. Low spread = the family is uniformly hard (data);
high spread = some bodies cope and others do not (infeasibility).
Control: the same statistic for the easy families.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from dual_common import load_paired_data, build_model, geom, fk_pos_torch
RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
cfg = json.load(open(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg["cache"], dev, sem_encoder=cfg["sem_encoder"])
m = build_model(cfg, dev)
m.load_state_dict(torch.load(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/model.pt",
                             map_location=dev)); m.eval()
kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
ROBOTS = ["unitree_g1_29dof", "unitree_h1", "booster_t1", "fourier_gr3",
          "kuavo_s45", "hightorque_hi"]
FAMS = ["run", "sit", "dance", "walk", "jump", "punch", "jacks", "kick"]
ARM = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle")
T, NW = 90, 12
g = np.random.default_rng(0)
starts = {}
for f in FAMS:
    s, _ = kd.sample_starts(NW, g, "val", span=T + 2, families=[f])
    starts[f] = [int(x) for x in s.tolist()]
tab = {}
print(f"{'family':>8s} " + " ".join(f"{r.split('_')[-1][:7]:>8s}" for r in ROBOTS)
      + f" {'mean':>7s} {'CV':>6s}")
with torch.no_grad():
    for f in FAMS:
        row = []
        for tgt in ROBOTS:
            spec = data[tgt]["spec"]
            am = torch.tensor([any(k in x.lower() for k in ARM)
                               for x in spec.joint_names], device=dev)
            es = []
            for r0 in starts[f]:
                rows = torch.arange(r0, r0 + T)
                prot, _ = m.decode(*m.encode(data["human_smpl"]["pose"][rows],
                                             geom(data, "human_smpl", T)),
                                   geom(data, tgt, T))
                e = (fk_pos_torch(prot, spec) -
                     fk_pos_torch(data[tgt]["pose"][rows][..., :6], spec)).norm(dim=-1)
                # scale-fair: report in units of that robot's reach
                es.append(float(e[:, am].mean()) / data[tgt]["reach"] * 100)
            row.append(float(np.mean(es)))
        cv = float(np.std(row) / (np.mean(row) + 1e-9))
        tab[f] = {"per_robot": row, "mean": float(np.mean(row)), "cv": cv}
        print(f"{f:>8s} " + " ".join(f"{v:8.2f}" for v in row)
              + f" {np.mean(row):7.2f} {cv:6.3f}")
hard = ["kick", "jacks", "punch", "jump"]
easy = ["run", "sit", "dance", "walk"]
ch = np.mean([tab[f]["cv"] for f in hard]); ce = np.mean([tab[f]["cv"] for f in easy])
print(f"\n== errors are in % of each robot's reach (scale-fair) ==")
print(f"   hard families  mean CV across robots {ch:.3f}")
print(f"   easy families  mean CV across robots {ce:.3f}")
print(f"   ratio {ch/max(ce,1e-9):.2f}   (>1.3 => body-specific => INFEASIBILITY;"
      f" ~1 => uniform => DATA)")
json.dump(tab, open(f"{RUNS}/FAMILY_ROBOT_SPLIT_0813.json", "w"), indent=2)
