"""FK-YARDSTICK DUEL — the three 0812 arms against the matched baseline.

    fk_duel.py

Every model is measured the same way: encode the source pose, decode to the
target body, run FK on the PREDICTED rotations, compare with FK on the GT
rotations. That is the quantity every reported V5 position comes through, and
it is the only one that is meaningful for rot_only models (their hpos head is
gone, so the eval's own pos_cm column measures nothing).
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from dual_common import load_paired_data, split_indices, build_model, geom, fk_pos_torch

RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
ARM = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle")
LEG = ("hip", "knee", "ankle", "foot", "leg", "toe", "shank")
MODELS = ["PROD_dim16_13k_0809", "SMALL_dim16_reenc_0812",
          "FK1_rot128_0812", "FK2_rot256_0812"]
DIRS = [("human_smpl", "unitree_g1_29dof"), ("human_smpl", "fourier_gr3"),
        ("human_smpl", "booster_t1"), ("human_smpl", "kuavo_s45")]
cfg0 = json.load(open(f"{RUNS}/{MODELS[0]}/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg0["cache"], dev, sem_encoder=cfg0["sem_encoder"])
val_idx, _ = split_indices(N)
g = np.random.default_rng(1)
val = val_idx[torch.as_tensor(g.choice(len(val_idx), 2048, replace=False))]

print(f"{'model':24s} {'slots':>6s} {'bits/f':>7s} | "
      f"{'arm mean':>9s} {'arm p95':>8s} {'leg mean':>9s} {'leg p95':>8s} "
      f"{'wrist p95':>10s}")
out = {}
for mn in MODELS:
    cfg = json.load(open(f"{RUNS}/{mn}/config.json"))
    m = build_model(cfg, dev)
    m.load_state_dict(torch.load(f"{RUNS}/{mn}/model.pt", map_location=dev))
    m.eval()
    agg = {k: [] for k in ("arm", "leg", "wrist")}
    p95 = {k: [] for k in ("arm", "leg", "wrist")}
    with torch.no_grad():
        for src, tgt in DIRS:
            spec = data[tgt]["spec"]
            nl = [x.lower() for x in spec.joint_names]
            am = torch.tensor([any(k in x for k in ARM) for x in nl], device=dev)
            lm = torch.tensor([any(k in x for k in LEG) for x in nl], device=dev)
            wm = torch.tensor(["wrist" in x for x in nl], device=dev)
            zp, zr = m.encode(data[src]["pose"][val], geom(data, src, len(val)))
            prot, _ = m.decode(zp, zr, geom(data, tgt, len(val)))
            gtr = data[tgt]["pose"][val][..., :6]
            fp = fk_pos_torch(prot, spec)                    # cm, root-relative
            fg = fk_pos_torch(gtr, spec)
            err = (fp - fg).norm(dim=-1)
            for k, msk in (("arm", am), ("leg", lm), ("wrist", wm)):
                if msk.any():
                    e = err[:, msk].reshape(-1)
                    agg[k].append(float(e.mean())); p95[k].append(float(e.quantile(.95)))
    slots = cfg["n_latent"] * (1 if cfg.get("rot_only") else 2)
    eff = cfg["n_latent"] if cfg.get("rot_only") else cfg["n_latent"]
    bits = slots * len(cfg["fsq_levels"].split(",")) * 3
    r = {k: float(np.mean(agg[k])) for k in agg} | \
        {k + "_p95": float(np.mean(p95[k])) for k in p95}
    out[mn] = r | {"slots": slots, "rot_only": bool(cfg.get("rot_only")),
                   "fk_loss": bool(cfg.get("fk_loss"))}
    print(f"{mn:24s} {slots:6d} {bits:7d} | {r['arm']:9.2f} {r['arm_p95']:8.2f} "
          f"{r['leg']:9.2f} {r['leg_p95']:8.2f} {r['wrist_p95']:10.2f}")
b = out[MODELS[0]]
print(f"\n{'vs baseline':24s} {'arm mean':>9s} {'arm p95':>8s} {'leg mean':>9s} {'wrist p95':>10s}")
for mn in MODELS[1:]:
    r = out[mn]
    print(f"{mn:24s} {100*(r['arm']/b['arm']-1):+8.1f}% {100*(r['arm_p95']/b['arm_p95']-1):+7.1f}% "
          f"{100*(r['leg']/b['leg']-1):+8.1f}% {100*(r['wrist_p95']/b['wrist_p95']-1):+9.1f}%")
json.dump(out, open(f"{RUNS}/FK_DUEL_0813.json", "w"), indent=2)
