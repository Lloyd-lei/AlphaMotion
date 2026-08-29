"""POSITION-STREAM AUDIT — three questions lloyd raised (0812).

    pos_stream_audit.py

Q1 CONSISTENCY GAP. The decoder has two heads: hpos emits joint positions
   directly, hrot emits rotations from which FK gives positions. FK positions
   are bone-rigid by construction; hpos is unconstrained ("rubber bones", which
   is why rendering never uses it). Since both describe the same thing, their
   disagreement is a free, ground-truth-free self-check. Measure it per joint
   group and per embodiment.

Q2 IS `gp / reach` HURTING ARMS? reach = max over the corpus of ||joint - root||,
   a single scalar per body. If that maximum is set by an extended arm, the
   normalisation is calibrated on the arms' extreme and everything else lives in
   a fraction of the range. Report which joint actually attains the max, and
   what fraction of the range each group occupies.

Q3 DOES SIZE PREDICT ERROR? lloyd recalls one very small robot always being bad.
   Correlate per-embodiment error with reach.

All measurements use the FK yardstick for the "true" positions, per the V5 rule.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from dual_common import load_paired_data, split_indices, build_model, geom
from common import rot6d_to_matrix

RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
ARM = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle")
LEG = ("hip", "knee", "ankle", "foot", "leg", "toe", "shank")
cfg = json.load(open(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg["cache"], dev, sem_encoder=cfg["sem_encoder"])
val_idx, _ = split_indices(N)
sp = build_model(cfg, dev)
sp.load_state_dict(torch.load(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/model.pt",
                              map_location=dev)); sp.eval()
g = np.random.default_rng(1)
val = val_idx[torch.as_tensor(g.choice(len(val_idx), 2048, replace=False))]

print(f"{'embodiment':20s} {'reach':>7s} {'max-joint':>16s} "
      f"{'gap arm':>8s} {'gap leg':>8s} {'FKerr arm':>10s} {'FKerr leg':>10s}")
rows = []
with torch.no_grad():
    for nm in names:
        spec = data[nm]["spec"]
        jn = [x.lower() for x in spec.joint_names]
        am = torch.tensor([any(k in x for k in ARM) for x in jn], device=dev)
        lm = torch.tensor([any(k in x for k in LEG) for x in jn], device=dev)
        if not am.any() or not lm.any():
            continue
        reach = data[nm]["reach"]
        pose = data[nm]["pose"][val]                       # [B,J,9] gt
        # Q2: which joint attains the reach, and range use per group
        gt_pos = pose[..., 6:9] * reach                    # cm, root-relative
        dist = gt_pos.norm(dim=-1)                         # [B,J]
        jmax = int(dist.mean(0).argmax())
        frac_arm = float(dist[:, am].max() / dist.max())
        frac_leg = float(dist[:, lm].max() / dist.max())
        # decode
        zp, zr = sp.encode(pose, geom(data, nm, len(val)))
        prot, ppos = sp.decode(zp, zr, geom(data, nm, len(val)))
        fk = torch.as_tensor(V.fk_pos(prot.float().cpu(), spec)).float().to(dev)
        fk = fk - fk[:, :1]                                # root-relative cm
        pos_cm = ppos * reach
        gap = (pos_cm - fk).norm(dim=-1)                   # Q1 consistency gap
        fk_err = (fk - gt_pos).norm(dim=-1)                # FK vs GT (the real ruler)
        rows.append({"emb": nm, "reach": float(reach), "jmax": spec.joint_names[jmax],
                     "gap_arm": float(gap[:, am].mean()), "gap_leg": float(gap[:, lm].mean()),
                     "fk_arm": float(fk_err[:, am].mean()), "fk_leg": float(fk_err[:, lm].mean()),
                     "fk_arm_p95": float(fk_err[:, am].quantile(0.95)),
                     "frac_arm": frac_arm, "frac_leg": frac_leg})
        r = rows[-1]
        print(f"{nm:20s} {r['reach']:7.1f} {r['jmax'][:16]:>16s} "
              f"{r['gap_arm']:8.2f} {r['gap_leg']:8.2f} "
              f"{r['fk_arm']:10.2f} {r['fk_leg']:10.2f}")

print(f"\nQ2 range use: arms reach {np.mean([r['frac_arm'] for r in rows])*100:.0f}% "
      f"of the max, legs {np.mean([r['frac_leg'] for r in rows])*100:.0f}%")
from collections import Counter
print("   joint attaining the max:", Counter(r["jmax"] for r in rows).most_common(4))
R = np.array([r["reach"] for r in rows])
for k in ("fk_arm", "fk_arm_p95", "fk_leg", "gap_arm"):
    v = np.array([r[k] for r in rows])
    print(f"Q3 corr(reach, {k:11s}) = {np.corrcoef(R, v)[0,1]:+.3f}")
json.dump(rows, open(f"{RUNS}/POS_STREAM_AUDIT_0812.json", "w"), indent=2)
