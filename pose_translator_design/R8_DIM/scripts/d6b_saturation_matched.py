"""Saturation, done properly.

d6 arm A held the SAMPLE SIZE fixed while the clip pool grew, so the sampling density per
clip collapsed as clips were added -- and the NN estimators are strongly density-sensitive
(d6 arm B: at 4000 clips, TwoNN goes 4.9 -> 8.5 as n goes 1k -> 24k). The two effects were
confounded and the raw arm-A curve even went DOWN.

Fix: compare clip counts at MATCHED sample size n. If the ID(n) curve for 500 clips lies on
top of the curve for 4000 clips, then the 8x extra clips contributed no new directions -- they
only gave more samples of the same manifold. That is exactly the claim under test.
"""
import json, numpy as np, torch
from d0_common import *

CACHE, BODY = AMASS, "human_smpl"
seqs = seq_table(CACHE)
g = np.random.default_rng(0)
perm = g.permutation(len(seqs))
rows = []
for nclip in [500, 1000, 2000, 4000]:
    clips = perm[:nclip].tolist()
    pool = sum(seqs[i]["len"] for i in clips)
    for n in [2000, 5000, 10000]:
        for kind, T in [("frame", 1), ("window", 32)]:
            if kind == "frame":
                rws, cid, _ = sample_frames(CACHE, n, clip_subset=clips)
                X, _ = frame_features(CACHE, BODY, rws)
            else:
                st, cid = sample_windows(CACHE, T, n, clip_subset=clips)
                X, _ = window_features(CACHE, BODY, st, T)
            r, _ = all_estimators(X, group=torch.as_tensor(cid, device=DEV), exclude_same_group=True)
            r.update(n_clips=nclip, pool_frames=int(pool), kind=kind, T=T,
                     stride=round(pool / len(cid), 1))
            rows.append(r); print(r)
json.dump(rows, open(f"{SP}/nums_d6b_sat.json", "w"), indent=1)
print("done")
