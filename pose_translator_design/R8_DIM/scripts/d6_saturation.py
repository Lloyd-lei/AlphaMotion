"""Saturation: does intrinsic dimension grow as we add clips?

Two different questions, both answered:
  A) MORE CLIPS, SAME SAMPLE SIZE  -- nested subsets of 100/500/1000/2000/4000 AMASS clips,
     always drawing the SAME number of frames/windows. If ID is flat, the extra clips
     carried no new directions. This is the one that maps onto the data-axis null result.
  B) MORE SAMPLES FROM THE SAME POOL -- control for the known small-N downward bias of the
     NN estimators (see d1 sanity: at true d=50, TwoNN reads 33.5 at n=1k and 35.9 at n=30k).
"""
import json, numpy as np, torch
from d0_common import *

CACHE, BODY = AMASS, "human_smpl"
seqs = seq_table(CACHE)
g = np.random.default_rng(0)
perm = g.permutation(len(seqs))          # nested subsets: first N of a fixed permutation

rows = []
for nclip in [100, 500, 1000, 2000, 4000]:
    clips = perm[:nclip].tolist()
    nfr = sum(seqs[i]["len"] for i in clips)
    for kind, T, n in [("frame", 1, 12000), ("window", 16, 10000), ("window", 32, 10000)]:
        if kind == "frame":
            rws, cid, _ = sample_frames(CACHE, n, clip_subset=clips)
            X, _ = frame_features(CACHE, BODY, rws)
        else:
            st, cid = sample_windows(CACHE, T, n, clip_subset=clips)
            X, _ = window_features(CACHE, BODY, st, T)
        r, _ = all_estimators(X, group=torch.as_tensor(cid, device=DEV), exclude_same_group=True)
        r.update(n_clips=nclip, n_frames_pool=int(nfr), kind=kind, T=T, arm="A_more_clips")
        rows.append(r); print(r)

# --- B) sample-size control on the FULL 4000-clip pool ---
for n in [1000, 3000, 6000, 12000, 24000]:
    rws, cid, _ = sample_frames(CACHE, n)
    X, _ = frame_features(CACHE, BODY, rws)
    r, _ = all_estimators(X, group=torch.as_tensor(cid, device=DEV), exclude_same_group=True)
    r.update(n_clips=4000, kind="frame", T=1, arm="B_more_samples")
    rows.append(r); print(r)

json.dump(rows, open(f"{SP}/nums_d6_sat.json", "w"), indent=1)
print("done")
