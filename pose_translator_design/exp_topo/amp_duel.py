"""AMPLITUDE DUEL — the new metric across every codec we have.

    amp_duel.py

24 contiguous 90-frame val windows, 4 cross-body directions, FK positions.
Reported alongside the error each model was actually selected on, because the
point is that the two rulers disagree.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from dual_common import (load_paired_data, split_indices, build_model, geom,
                         fk_pos_torch, amplitude_report)
from bridge_a import CodeData
RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
MODELS = ["PROD_c1_91k_dim20_tail_0810", "PROD_dim16_13k_0809",
          "SMALL_dim16_reenc_0812", "FK1_rot128_0812", "FK2_rot256_0812",
          "PT_c1dec_reenc_0812"]
DIRS = [("human_smpl", "unitree_g1_29dof"), ("human_smpl", "fourier_gr3"),
        ("human_smpl", "booster_t1"), ("human_smpl", "kuavo_s45")]
cfg0 = json.load(open(f"{RUNS}/{MODELS[0]}/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg0["cache"], dev, sem_encoder=cfg0["sem_encoder"])
val_idx, _ = split_indices(N)
pools = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True).pools["val"]
sel = val_idx[torch.randperm(len(val_idx))[:2048]]
ARM = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle")
print(f"{'model':26s} {'arm err':>8s} {'arm p95':>8s} | {'amp all':>8s} "
      f"{'amp arm':>8s} {'amp leg':>8s}")
res = {}
for mn in MODELS:
    cfg = json.load(open(f"{RUNS}/{mn}/config.json"))
    m = build_model(cfg, dev)
    m.load_state_dict(torch.load(f"{RUNS}/{mn}/model.pt", map_location=dev)); m.eval()
    with torch.no_grad():
        es, ps = [], []
        for src, tgt in DIRS:
            spec = data[tgt]["spec"]
            am = torch.tensor([any(k in x.lower() for k in ARM)
                               for x in spec.joint_names], device=dev)
            prot, _ = m.decode(*m.encode(data[src]["pose"][sel], geom(data, src, len(sel))),
                               geom(data, tgt, len(sel)))
            e = (fk_pos_torch(prot, spec) -
                 fk_pos_torch(data[tgt]["pose"][sel][..., :6], spec)).norm(dim=-1)
            es.append(float(e[:, am].mean())); ps.append(float(e[:, am].quantile(.95)))
    amp = amplitude_report(m, data, DIRS, pools, n_win=24, T=90)
    A = {k: float(np.mean([v[k] for v in amp.values()])) for k in ("all", "arm", "leg")}
    res[mn] = {"arm_err": float(np.mean(es)), "arm_p95": float(np.mean(ps))} | A
    print(f"{mn:26s} {np.mean(es):8.2f} {np.mean(ps):8.2f} | {A['all']:8.3f} "
          f"{A['arm']:8.3f} {A['leg']:8.3f}")
    del m; torch.cuda.empty_cache()
json.dump(res, open(f"{RUNS}/AMP_DUEL_0813.json", "w"), indent=2)
