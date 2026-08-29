"""SEW DECOMPOSITION — how much of the arm error is DIRECTION and how much is
ROLL about the limb axis?

    sew_decompose.py

SEW-Mimic (arXiv 2602.01632) argues that matching an end effector's POSITION
leaves the arm's configuration under-determined: many elbow placements give the
same hand position, and the solver picks one. If our paired labels were made
that way, part of their arm configuration is solver choice, not human motion —
unpredictable in principle, and a model that is scored on reproducing it is
being scored partly on noise.

Decomposition per limb segment (upper arm = shoulder->elbow, forearm =
elbow->wrist):
  DIRECTION error  angle between the predicted and target segment vectors.
                   Determined by the human's pose; a model should get this.
  ROLL error       rotation about the segment axis after the direction is
                   aligned. Invisible to segment positions, and exactly the
                   component an end-effector-position solver picks arbitrarily.
Measured twice: LABEL vs HUMAN (is the label itself diverging?) and MODEL vs
LABEL (what is our model actually being penalised for?).
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from dual_common import load_paired_data, build_model, geom, fk_pos_torch
from common import rot6d_to_matrix
RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
cfg = json.load(open(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg["cache"], dev, sem_encoder=cfg["sem_encoder"])
m = build_model(cfg, dev)
m.load_state_dict(torch.load(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/model.pt", map_location=dev)); m.eval()
kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")

def seg_idx(spec):
    """(shoulder, elbow, wrist) index triples per side, by tree order."""
    jn = [x.lower() for x in spec.joint_names]
    par = list(spec.parents)
    out = []
    for side in ("left", "right", "l_", "r_"):
        ch = [i for i, n in enumerate(jn)
              if side in n and any(k in n for k in
                                   ("shoulder", "elbow", "wrist", "arm", "hand"))]
        if len(ch) >= 3:
            d = []
            for i in ch:
                k, j = 0, i
                while par[j] >= 0:
                    j = par[j]; k += 1
                d.append((k, i))
            d.sort()
            out.append((d[0][1], d[len(d)//2][1], d[-1][1]))
    return out[:2]

def decomp(fk, R, tri):
    """returns (upper-arm dir vec, forearm dir vec, roll frame) per triple."""
    s, e, w = tri
    ua = fk[:, e] - fk[:, s]
    fa = fk[:, w] - fk[:, e]
    return ua, fa, R[:, e]

def ang(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.rad2deg(torch.arccos((a*b).sum(-1).clamp(-1+1e-6, 1-1e-6)))

def roll_err(Ra, Rb, axis):
    """rotation about `axis` remaining after aligning: angle of Ra^T Rb projected
    on the axis."""
    D = Ra.transpose(-1, -2) @ Rb
    ax = torch.nn.functional.normalize(axis, dim=-1)
    tr = D.diagonal(dim1=-2, dim2=-1).sum(-1)
    th = torch.arccos(((tr - 1) / 2).clamp(-1+1e-6, 1-1e-6))
    v = torch.stack([D[..., 2, 1]-D[..., 1, 2], D[..., 0, 2]-D[..., 2, 0],
                     D[..., 1, 0]-D[..., 0, 1]], -1)
    v = torch.nn.functional.normalize(v, dim=-1)
    return torch.rad2deg((th * (v*ax).sum(-1)).abs())

FAMS = ["run", "dance", "walk", "punch", "jacks", "kick"]
tgt = "unitree_g1_29dof"; spec = data[tgt]["spec"]; tris = seg_idx(spec)
hspec = V.load_spec(os.environ["V5_AMASS"], "human_smpl"); htris = seg_idx(hspec)
T, NW = 60, 8
g = np.random.default_rng(0)
print(f"== 手臂误差分解 (人->G1, C1) — 单位:度 ==")
print(f"{'family':>8s} | {'MODEL vs LABEL':>26s} | {'LABEL vs HUMAN':>22s}")
print(f"{'':>8s} | {'上臂方向':>8s} {'前臂方向':>8s} {'滚转':>7s} | {'上臂方向':>10s} {'前臂方向':>10s}")
rows = {}
with torch.no_grad():
    for f in FAMS:
        s, _ = kd.sample_starts(NW, g, "val", span=T+2, families=[f])
        acc = {k: [] for k in ("ua","fa","roll","hua","hfa")}
        for r0 in s.tolist():
            rows_ = torch.arange(int(r0), int(r0)+T)
            p9 = data["human_smpl"]["pose"][rows_]
            prot, _ = m.decode(*m.encode(p9, geom(data,"human_smpl",T)), geom(data,tgt,T))
            gtr = data[tgt]["pose"][rows_][..., :6]
            fp, fg = fk_pos_torch(prot, spec), fk_pos_torch(gtr, spec)
            Rp, Rg = rot6d_to_matrix(prot), rot6d_to_matrix(gtr)
            fh = fk_pos_torch(p9[..., :6], hspec)
            for ti, tri in enumerate(tris):
                uap, fap, Rep = decomp(fp, Rp, tri)
                uag, fag, Reg = decomp(fg, Rg, tri)
                acc["ua"].append(float(ang(uap, uag).mean()))
                acc["fa"].append(float(ang(fap, fag).mean()))
                acc["roll"].append(float(roll_err(Reg, Rep,
                    torch.nn.functional.normalize(fag, dim=-1)).mean()))
                if ti < len(htris):
                    uah, fah, _ = decomp(fh, rot6d_to_matrix(p9[..., :6]), htris[ti])
                    acc["hua"].append(float(ang(uag, uah).mean()))
                    acc["hfa"].append(float(ang(fag, fah).mean()))
        rows[f] = {k: float(np.mean(v)) for k, v in acc.items() if v}
        r = rows[f]
        print(f"{f:>8s} | {r['ua']:8.1f} {r['fa']:8.1f} {r['roll']:7.1f} | "
              f"{r.get('hua',float('nan')):10.1f} {r.get('hfa',float('nan')):10.1f}")
hard = ["kick","jacks","punch"]; easy = ["run","dance","walk"]
for nm, grp in (("hard", hard), ("easy", easy)):
    d = np.mean([rows[f]['ua']+rows[f]['fa'] for f in grp])/2
    rl = np.mean([rows[f]['roll'] for f in grp])
    hh = np.mean([rows[f].get('hua',0)+rows[f].get('hfa',0) for f in grp])/2
    print(f"== {nm:>4s}: 模型方向误差 {d:5.1f}deg  滚转误差 {rl:5.1f}deg  "
          f"滚转占比 {100*rl/(rl+d):4.0f}% | 标签vs人方向差 {hh:5.1f}deg")
json.dump(rows, open(f"{RUNS}/SEW_DECOMPOSE_0813.json", "w"), indent=2)
