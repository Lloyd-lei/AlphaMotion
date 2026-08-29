"""Breakdowns: (a) by action family, (b) by body part. AMASS human_smpl."""
import json, numpy as np, torch
from d0_common import *

CACHE, BODY = AMASS, "human_smpl"
seqs = seq_table(CACHE)
names = [s["name"] for s in seqs]
fam = np.array([action_family(n) for n in names])
print({f: int((fam == f).sum()) for f in set(fam)})

# ---------- (a) by action family ----------
# Matched sample size: every family gets the SAME number of frames/windows, so any
# difference in ID cannot come from having more data.
N_MATCH = 4000
fam_rows = []
for f in sorted(set(fam)):
    clips = np.where(fam == f)[0].tolist()
    nfr = sum(seqs[i]["len"] for i in clips)
    for kind, T in [("frame", 1), ("window", 16), ("window", 32)]:
        if kind == "frame":
            rws, cid, _ = sample_frames(CACHE, N_MATCH, clip_subset=clips)
            X, _ = frame_features(CACHE, BODY, rws)
        else:
            st, cid = sample_windows(CACHE, T, N_MATCH, clip_subset=clips)
            if len(st) < 500:
                continue
            X, _ = window_features(CACHE, BODY, st, T)
        r, _ = all_estimators(X, group=torch.as_tensor(cid, device=DEV), exclude_same_group=True)
        r.update(family=f, n_clips=len(clips), n_frames_total=nfr, kind=kind, T=T)
        fam_rows.append(r); print(r)
json.dump(fam_rows, open(f"{SP}/nums_d3_family.json", "w"), indent=1)

# ---------- (b) by body part ----------
rws, cid, _ = sample_frames(CACHE, 15000)
_, _, jn = fk_positions(CACHE, BODY, rws[:2])
parts = part_indices(jn)
part_rows = []
g = torch.as_tensor(cid, device=DEV)
for pname, idxs in parts.items():
    X, _ = frame_features(CACHE, BODY, rws, joints=idxs)
    r, _ = all_estimators(X, group=g, exclude_same_group=True)
    r.update(part=pname, n_joints=len(idxs), kind="frame", T=1,
             per_joint=round(r["twonn"] / len(idxs), 3))
    part_rows.append(r); print(r)

st, cid_w = sample_windows(CACHE, 16, 12000)
gw = torch.as_tensor(cid_w, device=DEV)
for pname, idxs in parts.items():
    X, _ = window_features(CACHE, BODY, st, 16, joints=idxs)
    r, _ = all_estimators(X, group=gw, exclude_same_group=True)
    r.update(part=pname, n_joints=len(idxs), kind="window", T=16,
             per_joint=round(r["twonn"] / len(idxs), 3))
    part_rows.append(r); print(r)
json.dump(part_rows, open(f"{SP}/nums_d3_part.json", "w"), indent=1)
print("done")
