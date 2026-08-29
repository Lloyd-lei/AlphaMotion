"""Main: intrinsic dimension of RAW motion -- frames and windows, AMASS + LAFAN, several bodies."""
import json, numpy as np, torch
from d0_common import *

N_FR = 15000
N_WIN = 12000
rows_out = []
eigs = {}


def run(cache, body, kind, T=None, n=None, tag=None):
    if kind == "frame":
        rws, cid, _ = sample_frames(cache, n)
        X, jn = frame_features(cache, body, rws)
    else:
        st, cid = sample_windows(cache, T, n)
        X, jn = window_features(cache, body, st, T)
    g = torch.as_tensor(cid, device=DEV)
    out = {}
    for excl, lbl in [(True, "xclip"), (False, "raw")]:
        r, ev = all_estimators(X, group=g, exclude_same_group=excl)
        r.update(cache=os.path.basename(cache), body=body, kind=kind, T=T or 1,
                 neigh=lbl, tag=tag or f"{os.path.basename(cache)}/{body}/{kind}{T or 1}")
        rows_out.append(r); print(r)
        if excl:
            eigs[r["tag"]] = ev
    return X


for cache, cname in [(AMASS, "amass"), (LAFAN, "lafan")]:
    for body in ["human_smpl", "unitree_g1_29dof", "pal_talos"]:
        run(cache, body, "frame", n=N_FR)
        for T in [16, 32]:
            run(cache, body, "window", T=T, n=N_WIN)

json.dump(rows_out, open(f"{SP}/nums_d2_main.json", "w"), indent=1)
np.savez(f"{SP}/eigs_d2.npz", **{k: v[:600] for k, v in eigs.items()})
print("done")
