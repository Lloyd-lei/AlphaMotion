"""Robustness-first scoring of the BFM-Zero stress rollouts (lloyd 0831)."""
import glob, json, os
import numpy as np

DT = 0.02  # env control dt (50 Hz)
import sys
OUT = sys.argv[1]
IPELVIS, ISHL, ISHR = 0, 16, 23   # torso_link origin sits at the waist, so use the shoulder midpoint


def upright(bpos):
    v = 0.5 * (bpos[:, ISHL] + bpos[:, ISHR]) - bpos[:, IPELVIS]
    n = np.linalg.norm(v, axis=-1) + 1e-9
    return v[:, 2] / n


def episodes(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None: s = i
        if not m and s is not None: out.append((s, i)); s = None
    if s is not None: out.append((s, len(mask)))
    return out


def score(f):
    d = np.load(f, allow_pickle=True)
    bp, dp, dv = d["bpos"], d["dpos"], d["dvel"]
    T = len(bp)
    rbp = d["ref_bpos"][1:1 + T][:, :bp.shape[1]]; rdp = d["ref_dpos"][1:1 + T]
    n = min(T, len(rbp)); bp, dp, dv, rbp, rdp = bp[:n], dp[:n], dv[:n], rbp[:n], rdp[:n]
    u, ru = upright(bp), upright(rbp)
    torso_z = 0.5 * (bp[:, ISHL, 2] + bp[:, ISHR, 2])
    ref_torso_z = 0.5 * (rbp[:, ISHL, 2] + rbp[:, ISHR, 2])
    fallen = (u < 0.5) | (torso_z < 0.50)
    ref_fallen = (ru < 0.5) | (ref_torso_z < 0.50)
    fail = fallen & ~ref_fallen                      # down while the reference is up
    idx = np.flatnonzero(fail)
    first = int(idx[0]) if idx.size else None
    eps = episodes(fail)
    recs = []
    for s, e in eps:
        # recovered = upright again for >= 0.5 s after the episode ends
        if e < len(u) and (e + 25) <= len(u) and (u[e:e + 25] > 0.7).all():
            recs.append((e - s) * DT)
    m = dict(
        motion=str(d["motion"]), tag=str(d["tag"]), steps=int(n), dur_s=round(n * DT, 2),
        survival_frac=round(1.0 if first is None else first / n, 3),
        survival_s=round(n * DT if first is None else first * DT, 2),
        fail_frac=round(float(fail.mean()), 3),
        n_fall_episodes=len(eps), n_recovered=len(recs),
        recover_time_s=[round(r, 2) for r in recs],
        chest_z_mean=round(float(torso_z.mean()),3), upright_at_end=round(float(u[-1]), 3), fallen_at_end=bool(fail[-1]),
        upright_mean=round(float(u.mean()), 3), upright_p05=round(float(np.percentile(u, 5)), 3),
        env_terminated_step=(int(np.flatnonzero(d["term"])[0]) if d["term"].any() else None),
        dvel_absmax=round(float(np.abs(dv).max()), 1),
        dvel_p999=round(float(np.percentile(np.abs(dv), 99.9)), 1),
        dvel_spike_frac_gt20=round(float((np.abs(dv) > 20).mean()), 5),
        min_body_z_cm=round(float(bp[..., 2].min() * 100), 1),
        penetration_frac_below_m2cm=round(float((bp[..., 2] < -0.02).any(1).mean()), 3),
        nonfinite=int((~np.isfinite(bp)).sum() + (~np.isfinite(dp)).sum()),
        # --- secondary: tracking accuracy
        mpjpe_l=round(float((np.linalg.norm(dp - rdp, axis=-1) * 1000).mean()), 1),
        body_pos_err_cm=round(float(np.linalg.norm(bp - rbp, axis=-1).mean() * 100), 1),
        pelvis_drift_cm=round(float(np.linalg.norm(bp[:, 0, :2] - rbp[:, 0, :2], axis=-1).mean() * 100), 1),
        ref_fallen_frac=round(float(ref_fallen.mean()), 3),
    )
    try:
        import ot, torch
        a = dp.astype(np.float64); b = rdp.astype(np.float64)
        C = np.sqrt(np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0))
        m["emd"] = round(float(ot.emd2(np.ones(len(a)) / len(a), np.ones(len(b)) / len(b), C, numItermax=100000)), 3)
    except Exception as e:
        m["emd"] = None; m["emd_err"] = str(e)
    return m


rows = [score(f) for f in sorted(glob.glob(f"{OUT}/*.npz"))]
json.dump(rows, open(OUT + "/scores.json", "w"), indent=1)
cols = ["tag", "motion", "dur_s", "survival_frac", "survival_s", "fail_frac", "n_fall_episodes", "n_recovered",
        "fallen_at_end", "upright_mean", "env_terminated_step", "dvel_absmax", "dvel_spike_frac_gt20",
        "min_body_z_cm", "mpjpe_l", "body_pos_err_cm", "pelvis_drift_cm", "emd", "ref_fallen_frac"]
print("\t".join(cols))
for r in sorted(rows, key=lambda r: (r["motion"], r["tag"])):
    print("\t".join(str(r.get(c)) for c in cols))
