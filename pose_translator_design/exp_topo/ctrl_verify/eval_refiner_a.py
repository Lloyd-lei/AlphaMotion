"""Refiner-A evaluation: RAW decode (A) vs REFINED (R), native retarget (B) as the
ruler.  B is EVAL-ONLY -- the refiner itself never sees it.

metrics per track:
  penetration : per-frame max floor penetration depth (exact MuJoCo collision)
  skate_self  : foot speed on frames the track itself calls planted (low+slow+hyst rule)
  skate@Bstance: foot speed during the NATIVE track's planted windows (the honest ruler:
                 "when the foot should be planted, how fast does it move")
  jerk        : dof jerk RMS (3rd difference * fps^3)
vs-B metrics (A and R only):
  joint_abs_err (deg, 29 dof), body_pos_err (m, mean over bodies)

usage: eval_refiner_a.py --xml <xml> --dir <tracks dir> --refined_dir <dir> [--fps 30]
"""
import argparse, glob, json, os, sys
import numpy as np
import mujoco as mj
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refiner_a import (build_model, find_feet, window_speed, stance_segments,
                       FOOT_V_ON, FOOT_V_OFF, H_LOW, H_HIGH, MIN_SEG_SEC)

SLIP = 0.15  # m/s physics slip line

_G = {}
def _init(xml, fps):
    m, fl = build_model(xml)
    _G.update(m=m, d=mj.MjData(m), floor=fl, feet=find_feet(m, "ankle_roll|foot|toe"),
              fps=fps)

def fk_full(q):
    m, d, fl, feet = _G["m"], _G["d"], _G["floor"], _G["feet"]
    T = len(q); nb = m.nbody
    pen = np.zeros(T); xpos = np.zeros((T, nb, 3))
    anch = np.zeros((T, len(feet), 3)); zbot = np.zeros((T, len(feet)))
    for t in range(T):
        d.qpos[:] = q[t]; mj.mj_kinematics(m, d); mj.mj_collision(m, d)
        p = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == fl or c.geom2 == fl) and c.dist < -p: p = -c.dist
        pen[t] = p
        xpos[t] = d.xpos
        for k, f in enumerate(feet):
            anch[t, k] = d.xpos[f["body"]]
            zb = 1e9
            for g in f["geoms"]:
                r = m.geom_size[g][0] if m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE \
                    else m.geom_rbound[g]
                zb = min(zb, d.geom_xpos[g][2] - r)
            zbot[t, k] = zb
    return pen, xpos, anch, zbot

def stance_mask(anch, zbot, fps, rel=True):
    T = len(anch); mask = np.zeros((T, anch.shape[1]), bool)
    speeds = np.zeros((T, anch.shape[1]))
    for k in range(anch.shape[1]):
        v = window_speed(anch[:, k, :2], fps)
        zb = zbot[:, k]
        if rel:  # measure height above the track's own robust ground estimate
            zb = zb - max(0.0, float(np.percentile(zb, 5)))
        for (s, e) in stance_segments(zb, v, fps):
            mask[s:e, k] = True
        speeds[:, k] = v
    return mask, speeds

def _work(name):
    dirA, dirR, fps = _G["dirA"], _G["dirR"], _G["fps"]
    qA = np.load(os.path.join(dirA, name + "__qposA.npy"))
    qB = np.load(os.path.join(dirA, name + "__qposB.npy"))
    qR = np.load(os.path.join(dirR, name + "__qposR.npy"))
    T = min(len(qA), len(qB), len(qR))
    qA, qB, qR = qA[:T], qB[:T], qR[:T]
    out = dict(name=name, frames=T)
    fk = {}
    for tag, q in (("A", qA), ("B", qB), ("R", qR)):
        pen, xpos, anch, zbot = fk_full(q)
        mask, spd = stance_mask(anch, zbot, fps)
        fk[tag] = (pen, xpos, anch, zbot, mask, spd)
        dq = q[:, 7:]
        jerk = np.diff(dq, 3, axis=0) * fps**3
        out[tag] = dict(pen=pen.astype(np.float32),
                        skate_self=spd[mask].astype(np.float32),
                        jerk_ss=float((jerk**2).sum()), jerk_n=int(jerk.size))
    mB = fk["B"][4]
    for tag in ("A", "B", "R"):
        out[tag]["skate_atB"] = fk[tag][5][mB].astype(np.float32)
    for tag in ("A", "R"):
        q = qA if tag == "A" else qR
        jerr = np.abs(q[:, 7:] - qB[:, 7:])
        out[tag]["joint_err"] = np.degrees(jerr[::3]).astype(np.float32).ravel()
        out[tag]["body_err"] = np.linalg.norm(
            fk[tag][1] - fk["B"][1], axis=2).mean(axis=1).astype(np.float32)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--refined_dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    names = sorted(os.path.basename(f)[:-len("__qposA.npy")]
                   for f in glob.glob(os.path.join(a.dir, "*__qposA.npy")))
    import multiprocessing as mp
    def initw():
        _init(a.xml, a.fps); _G.update(dirA=a.dir, dirR=a.refined_dir)
    with mp.get_context("fork").Pool(a.workers, initw) as pool:
        res = pool.map(_work, names)

    agg = {}
    for tag in ("A", "R", "B"):
        pen = np.concatenate([r[tag]["pen"] for r in res])
        sk_self = np.concatenate([r[tag]["skate_self"] for r in res])
        sk_atB = np.concatenate([r[tag]["skate_atB"] for r in res])
        jss = sum(r[tag]["jerk_ss"] for r in res); jn = sum(r[tag]["jerk_n"] for r in res)
        agg[tag] = dict(
            pen_frac_1mm=float((pen > 0.001).mean()),
            pen_p95_mm=float(np.percentile(pen, 95) * 1000),
            pen_max_mm=float(pen.max() * 1000),
            skate_self_mean_cms=float(sk_self.mean() * 100) if len(sk_self) else None,
            skate_self_frames=int(len(sk_self)),
            skate_self_frac_slip=float((sk_self > SLIP).mean()) if len(sk_self) else None,
            skate_atB_mean_cms=float(sk_atB.mean() * 100),
            skate_atB_p95_cms=float(np.percentile(sk_atB, 95) * 100),
            skate_atB_frac_slip=float((sk_atB > SLIP).mean()),
            jerk_rms=float(np.sqrt(jss / jn)))
    for tag in ("A", "R"):
        je = np.concatenate([r[tag]["joint_err"] for r in res])
        be = np.concatenate([r[tag]["body_err"] for r in res])
        agg[tag].update(joint_err_mean_deg=float(je.mean()),
                        joint_err_p95_deg=float(np.percentile(je, 95)),
                        body_err_mean_cm=float(be.mean() * 100),
                        body_err_p95_cm=float(np.percentile(be, 95) * 100))
    frames = sum(r["frames"] for r in res)
    print(json.dumps(agg, indent=1))
    print("clips", len(res), "frames", frames)
    if a.out_json:
        json.dump(dict(agg=agg, clips=len(res), frames=frames),
                  open(a.out_json, "w"), indent=1)

if __name__ == "__main__":
    main()
