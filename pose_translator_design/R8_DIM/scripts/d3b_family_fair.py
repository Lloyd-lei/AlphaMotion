"""Family comparison at a FAIR sample size.

d3 drew 4000 samples per family. For 'sit' (7.7k frames) and 'run' (8.5k frames) that means
sampling almost every frame, which breaks the local-uniform-density assumption TwoNN needs
(it read 26-35 while MLE read 9-10 -- a blow-up, not a signal). Here every family is sampled
at n=1500 with stride >= 4 frames, which every family can supply honestly.
"""
import json, numpy as np, torch
from d0_common import *

CACHE, BODY = AMASS, "human_smpl"
seqs = seq_table(CACHE)
names = [s["name"] for s in seqs]
fam = np.array([action_family(n) for n in names])
N = 1500
rows = []
for f in sorted(set(fam)):
    clips = np.where(fam == f)[0].tolist()
    pool = sum(seqs[i]["len"] for i in clips)
    for kind, T in [("frame", 1), ("window", 32)]:
        if kind == "frame":
            rws, cid, _ = sample_frames(CACHE, N, clip_subset=clips)
            X, _ = frame_features(CACHE, BODY, rws)
        else:
            st, cid = sample_windows(CACHE, T, N, clip_subset=clips)
            X, _ = window_features(CACHE, BODY, st, T)
        r, _ = all_estimators(X, group=torch.as_tensor(cid, device=DEV), exclude_same_group=True)
        r.update(family=f, n_clips=len(clips), pool_frames=int(pool), kind=kind, T=T,
                 sampling_stride=round(pool / max(len(cid), 1), 1))
        rows.append(r); print(r)
json.dump(rows, open(f"{SP}/nums_d3b_family_fair.json", "w"), indent=1)
print("done")
