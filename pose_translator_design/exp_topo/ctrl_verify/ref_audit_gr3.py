"""Offline reference-quality audit on Fourier GR3: GMR retarget vs Greenwich decode.

Question: is a Greenwich-decoded reference DYNAMICALLY worse than the true GMR
retarget on the same robot and the same motions?  No training, no policy — this
is pure kinematics -> inverse dynamics on the GMR GR3 MJCF.

Two references, frame-for-frame paired, for the same AMASS clips:
  A = GMR      fourier_gr3_paired_local6d.pt  (that IS the xwbc/GMR retarget)
               -> build_global -> hinge-angle fit -> qpos
               (validated: recovers the raw GMR npz joint_pos to 1.4e-6 deg)
  B = Greenwich human_smpl_paired_local6d.pt -> Greenwich.encode
               -> decode_full on the GR3 descriptor -> MP.project into GR3's
               limit box -> qpos
Root trajectory (position AND orientation) is GMR passthrough for BOTH tracks —
that is what amctrl's production reference builder does (ROOT_ORI_PASSTHROUGH,
gr3_tracks.py), and it isolates the question to the joint angles.

Metrics (identical operator on A and B):
 1. joint-limit violation, before and after the manifold projection
 2. inverse-dynamics torque as a fraction of each joint's actuatorfrcrange
    - "free"    : contacts disabled, tau = M qacc + C + G. Deterministic, no
                  contact-model artefacts.  Exact for arms/waist/head and for
                  swing legs; an UNDER-estimate for stance legs (no GRF).
                  Also reports the 6-DOF base residual = the net external
                  wrench the ground would have to supply.
    - "contact" : the base residual is redistributed to the feet through a
                  friction-pyramid NNLS (unilateral, mu=0.8) over the foot
                  points that are within CONTACT_BAND of the per-clip floor;
                  tau_joint = qfrc_inverse - J_c^T f.  Reported with the
                  caveat that these references are NOT ground-consistent (even
                  the GMR ground truth's lowest foot point moves ~9 cm within a
                  clip), so stance detection is a heuristic.
 3. foot behaviour: ankle-roll body height, steps/s, planted-foot skate speed
 4. per-frame A-vs-B difference: joint RMSE (deg), body position RMSE (cm)

Read-only w.r.t. amctrl/ and all training configs.

  MUJOCO_GL=egl PYTHONPATH=/home/arenalabs/alphamotion_ctrl \
    /home/arenalabs/miniconda3/envs/ardy/bin/python ref_audit_gr3.py
"""
from __future__ import annotations
import os, sys, json, re, argparse, collections
import numpy as np

sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
torch.set_grad_enabled(False)
import mujoco as mj
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import nnls
from scipy.signal import savgol_filter

from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine import constraints as MP
from alphamotion.engine.nets.rotations import rot6d_to_matrix

CACHE = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"
XML = "/var/tmp/alphamotion/config_harvest/GMR/assets/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml"
OUT = "/home/arenalabs/tmp/ref_audit_gr3.json"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])      # Z-up -> Y-up (prep_gmr_cache v2)
FPS = 30.0
NFRAMES = 150
MU = 0.8                 # friction coefficient for the contact-wrench pyramid
CONTACT_BAND = 0.03      # m above the per-clip floor that counts as planted
FLOOR_PCT = 2.0          # per-clip floor = this percentile of A's per-frame min foot z

# priority order: first match wins (so "walk_to_run" -> run).  The negative
# lookbehinds matter: "tranSITions" and "cHOPping" are real clip names in AMASS.
FAMILIES = [
    ("kick",  r"kick"),
    ("throw", r"throw|toss"),
    ("sit",   r"(?<![a-z])sit|siting|chair|stool"),
    ("dance", r"danc|salsa|waltz|chacha|tango|karsilamas|zeibekiko"),
    ("jump",  r"jump|(?<![a-z])hop|(?<![a-z])leap"),
    ("run",   r"(?<![a-z])run|jog|sprint"),
    ("walk",  r"walk"),
]
PER_FAMILY = 3


# --------------------------------------------------------------------- model
def build_model():
    spec = mj.MjSpec.from_file(XML)
    if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in spec.joints):
        spec.bodies[1].add_freejoint(name="floating_base_joint")   # GMR ships GR3 fixed-base
    m = spec.compile()
    assert m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
    return m


class Rig:
    def __init__(self):
        self.m = build_model()
        self.m.opt.disableflags |= mj.mjtDisableBit.mjDSBL_CONTACT   # ID is done analytically
        self.d = mj.MjData(self.m)
        m = self.m
        self.hn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
        self.hadr = np.array([m.jnt_qposadr[j] for j in range(1, m.njnt)])
        self.hdof = np.array([m.jnt_dofadr[j] for j in range(1, m.njnt)])
        self.jrange = m.jnt_range[1:].copy()                     # [nj,2] rad
        self.effort = m.jnt_actfrcrange[1:, 1].copy()            # [nj] Nm, from the MJCF
        assert (self.effort > 0).all(), "MJCF is missing actuatorfrcrange on some joint"
        self.bn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        # foot collision geoms.  NOTE an MJCF asymmetry: the right sole cylinders hang
        # off right_foot_pitch_link, the left off left_foot_roll_link.  Contact wrenches
        # are applied to the *_foot_roll_link body on both sides (the physical sole), so
        # the ankle-roll joint sees ground reaction symmetrically.
        self.fgeom = {s: [g for g in range(m.ngeom)
                          if m.geom_contype[g] and m.geom_bodyid[g] > 0
                          and self.bn[m.geom_bodyid[g]].startswith(s)
                          and "foot" in self.bn[m.geom_bodyid[g]]] for s in ("left", "right")}
        self.sole = {s: self.bn.index(f"{s}_foot_roll_link") for s in ("left", "right")}
        self.mass = float(m.body_mass.sum())
        self.weight = self.mass * 9.81
        t1, t2, n = np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])
        self.edges = np.array([n + MU * t1, n - MU * t1, n + MU * t2, n - MU * t2])

    def qpos(self, root_p, root_q_wxyz, q_by_name):
        T = root_p.shape[0]
        qp = np.tile(self.m.qpos0, (T, 1))
        qp[:, 0:3] = root_p
        qp[:, 3:7] = root_q_wxyz
        for a, nm in zip(self.hadr, self.hn):
            qp[:, a] = q_by_name[nm]
        return qp

    def kin(self, qpt):
        """-> (foot points [2,4,3] world, sole body xpos [2,3], all body xpos [nbody,3])."""
        self.d.qpos[:] = qpt
        mj.mj_kinematics(self.m, self.d)
        pts = {}
        for s, gs in self.fgeom.items():
            P = []
            for g in gs:
                R = self.d.geom_xmat[g].reshape(3, 3)
                c = self.d.geom_xpos[g]
                ax = R[:, 2]                                     # cylinder axis
                hl, r = self.m.geom_size[g][1], self.m.geom_size[g][0]
                for sgn in (-1, 1):
                    P.append(c + sgn * hl * ax - np.array([0., 0., r]))
            pts[s] = np.array(P)
        sole = np.array([self.d.xpos[self.sole[s]] for s in ("left", "right")])
        return pts, sole, self.d.xpos.copy()


# ------------------------------------------------------------- derivatives
def derivatives(qp, rig, dt, smooth=0):
    """qpos [T,nq] -> (qvel [T,nv], qacc [T,nv]).  smooth>0 = Savitzky-Golay
    (window `smooth`, poly 2) applied to the root xyz / rotvec / joint angles
    before differencing, to separate per-frame jitter from structural load."""
    T = qp.shape[0]
    nv = rig.m.nv
    p = qp[:, 0:3].copy()
    R = Rot.from_quat(qp[:, 3:7][:, [1, 2, 3, 0]])
    jq = qp[:, rig.hadr].copy()
    if smooth and T > smooth:
        p = savgol_filter(p, smooth, 2, axis=0)
        jq = savgol_filter(jq, smooth, 2, axis=0)
        # quaternions, not rotvecs: rotvec wraps at pi and the wrap would be smoothed
        # into a fake several-hundred-rad/s^2 base acceleration.
        qq = qp[:, 3:7].copy()
        for t in range(1, T):
            if qq[t] @ qq[t - 1] < 0:
                qq[t] *= -1
        qq = savgol_filter(qq, smooth, 2, axis=0)
        qq /= np.linalg.norm(qq, axis=1, keepdims=True)
        R = Rot.from_quat(qq[:, [1, 2, 3, 0]])
        qp = qp.copy()
        qp[:, 0:3] = p
        qp[:, rig.hadr] = jq
        qp[:, 3:7] = R.as_quat()[:, [3, 0, 1, 2]]
    v = np.zeros((T, nv))
    v[:, 0:3] = np.gradient(p, dt, axis=0)                       # world linear vel (MuJoCo free joint)
    rel = np.array([(R[t].inv() * R[t + 1]).as_rotvec() / dt for t in range(T - 1)])
    w = np.zeros((T, 3))
    w[1:-1] = 0.5 * (rel[:-1] + rel[1:]); w[0] = rel[0]; w[-1] = rel[-1]
    v[:, 3:6] = w                                                # body-frame angular vel
    v[:, rig.hdof] = np.gradient(jq, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    return qp, v, a


# ------------------------------------------------------------ inverse dyn
def inverse_dynamics(rig, qp, qv, qa, floor, contact):
    """-> (tau [T,nj] Nm, base_res [T,6], ncontact [T], unmatched [T] bool)."""
    T = qp.shape[0]
    tau = np.zeros((T, len(rig.hn)))
    base = np.zeros((T, 6))
    ncp = np.zeros(T, dtype=int)
    unmatched = np.zeros(T, dtype=bool)
    jp_ = np.zeros((3, rig.m.nv)); jr_ = np.zeros((3, rig.m.nv))
    for t in range(T):
        rig.d.qpos[:] = qp[t]; rig.d.qvel[:] = qv[t]; rig.d.qacc[:] = qa[t]
        mj.mj_inverse(rig.m, rig.d)
        f = rig.d.qfrc_inverse.copy()
        if not contact:
            tau[t] = f[rig.hdof]; base[t] = f[0:6]
            continue
        pts, _, _ = rig.kin(qp[t])
        cols = []
        for s in ("left", "right"):
            for p in pts[s]:
                if p[2] - floor < CONTACT_BAND:
                    mj.mj_jac(rig.m, rig.d, jp_, jr_, p, rig.sole[s])
                    J = jp_.copy()
                    for e in rig.edges:
                        cols.append(J.T @ e)
        ncp[t] = len(cols) // 4
        if cols:
            A = np.stack(cols, 1)
            lam, _ = nnls(np.vstack([A[0:6], 1e-4 * np.eye(A.shape[1])]),
                          np.concatenate([f[0:6], np.zeros(A.shape[1])]))
            fc = A @ lam
            base[t] = f[0:6] - fc[0:6]
            tau[t] = (f - fc)[rig.hdof]
        else:
            base[t] = f[0:6]; tau[t] = f[rig.hdof]
        # "unmatched": the ground could not supply the required wrench (flight phase
        # with a non-zero demand, or a demand outside the friction cone)
        unmatched[t] = np.linalg.norm(base[t, 0:3]) > 0.15 * rig.weight
    return tau, base, ncp, unmatched


def tau_stats(tau, rig):
    r = np.abs(tau) / rig.effort
    return {
        "tau_mean_frac": float(r.mean()),
        "tau_p95_frac": float(np.percentile(r, 95)),
        "tau_max_frac": float(r.max()),
        "over_limit_frac": float((r > 1.0).mean()),
        "frames_any_over": float((r > 1.0).any(1).mean()),
        "per_joint_p95_frac": {rig.hn[i]: round(float(np.percentile(r[:, i], 95)), 4)
                               for i in range(len(rig.hn))},
        "per_joint_over_frac": {rig.hn[i]: round(float((r[:, i] > 1).mean()), 4)
                                for i in range(len(rig.hn))},
    }


# ------------------------------------------------------------------- feet
def foot_metrics(rig, qp, floor):
    T = qp.shape[0]
    low = np.zeros((T, 2)); sole_z = np.zeros((T, 2)); sole_xy = np.zeros((T, 2, 2))
    for t in range(T):
        pts, sole, _ = rig.kin(qp[t])
        low[t] = [pts["left"][:, 2].min(), pts["right"][:, 2].min()]
        sole_z[t] = sole[:, 2]; sole_xy[t] = sole[:, :2]
    low = low - floor
    sole_z = sole_z - floor
    planted = low < CONTACT_BAND
    v = np.linalg.norm(np.diff(sole_xy, axis=0), axis=-1) * FPS      # [T-1,2] m/s
    pm = planted[1:]
    skate = v[pm] if pm.any() else np.array([0.0])
    onsets = int((planted[1:] & ~planted[:-1]).sum())
    return {
        "ankle_roll_z_mean_cm": float(sole_z.mean() * 100),
        "ankle_roll_z_p05_cm": float(np.percentile(sole_z, 5) * 100),
        "ankle_roll_z_p95_cm": float(np.percentile(sole_z, 95) * 100),
        "foot_low_z_p50_cm": float(np.median(low) * 100),
        "foot_low_z_min_cm": float(low.min() * 100),
        "foot_low_z_max_cm": float(low.max() * 100),
        "planted_frac": float(planted.mean()),
        "steps_per_s": float(onsets / (T / FPS)),
        "skate_mean_cm_s": float(skate.mean() * 100),
        "skate_p95_cm_s": float(np.percentile(skate, 95) * 100),
    }


# ------------------------------------------------------------------ clips
def pick_clips(seqs, per_family=PER_FAMILY):
    buckets = collections.defaultdict(list)
    for s in seqs:
        if s["len"] < NFRAMES + 10:
            continue
        n = s["name"].lower()
        for fam, pat in FAMILIES:
            if re.search(pat, n):
                buckets[fam].append(s["name"]); break
    rng = np.random.default_rng(0)
    out = []
    for fam, _ in FAMILIES:
        names = sorted(buckets[fam])
        if not names:
            print(f"!! no clips for family {fam}"); continue
        idx = rng.choice(len(names), size=min(per_family, len(names)), replace=False)
        for i in sorted(idx):
            out.append((fam, names[i]))
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_family", type=int, default=PER_FAMILY)
    ap.add_argument("--frames", type=int, default=NFRAMES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smooth", type=int, default=9, help="Savitzky-Golay window for the smoothed ID variant")
    args = ap.parse_args()

    rig = Rig()
    print(f"GR3: nq={rig.m.nq} nv={rig.m.nv} hinges={len(rig.hn)} mass={rig.mass:.1f} kg "
          f"weight={rig.weight:.0f} N")

    def load_spec(p):
        d = np.load(p, allow_pickle=True)
        return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                            [str(n) for n in d["joint_names"]])

    hs = load_spec(CACHE + "human_smpl_spec.npz")
    hd = np.load(CACHE + "human_smpl_dof.npz", allow_pickle=True)["dof"]
    gspec, gdof, grest, gqn, _ = build_from_mjcf(XML, "fourier_gr3")
    cspec = load_spec(CACHE + "fourier_gr3_spec.npz")
    assert list(gspec.joint_names) == list(cspec.joint_names)
    dof_t = torch.as_tensor(gdof, dtype=torch.float64)
    rest_t = torch.as_tensor(grest, dtype=torch.float64)
    root_j = int(np.where(np.asarray(gspec.parents) < 0)[0][0])

    HL = torch.load(CACHE + "human_smpl_paired_local6d.pt", mmap=True)
    GL = torch.load(CACHE + "fourier_gr3_paired_local6d.pt", mmap=True)
    RW = torch.load(CACHE + "fourier_gr3_root_world.pt", mmap=True)
    seqs = json.load(open(CACHE + "sequences.json"))
    clips = pick_clips(seqs, args.per_family)
    byname = {s["name"]: s for s in seqs}
    print(f"{len(clips)} clips: " + ", ".join(f"{f}/{n[:40]}" for f, n in clips))

    G = Greenwich.load()
    dt = 1.0 / FPS

    def angles_from_R(Rg, clamp):
        """global rotmats [T,J,3,3] -> {mjcf joint name: [T] rad}."""
        if clamp:
            _, _, q = MP.project(Rg, gspec, dof_t, rest=rest_t, method="global", lm_iters=20)
        else:
            q, _ = MP.fit_angles(Rg, gspec, dof_t, rest=rest_t, clamp=False,
                                 method="global", lm_iters=20)
        q = np.asarray(q)
        out = {}
        for j, names3 in enumerate(gqn):
            for k, nm in enumerate(names3):
                if nm:
                    out[nm] = q[:, j, k]
        missing = [h for h in rig.hn if h not in out]
        assert not missing, missing
        return out

    def limit_stats(qmap):
        A = np.stack([qmap[nm] for nm in rig.hn], 1)
        over = np.maximum(A - rig.jrange[:, 1], 0) + np.maximum(rig.jrange[:, 0] - A, 0)
        bad = over > 1e-6
        worst = {}
        if bad.any():
            per = np.degrees(over.max(0))
            worst = {rig.hn[i]: round(float(per[i]), 2)
                     for i in np.argsort(-per)[:5] if per[i] > 1e-3}
        return {"violation_frac": float(bad.mean()),
                "frames_any": float(bad.any(1).mean()),
                "worst_overshoot_deg": float(np.degrees(over.max())),
                "worst_joints_deg": worst}

    report = {"config": {"frames": args.frames, "fps": FPS, "mu": MU,
                         "contact_band_m": CONTACT_BAND, "floor_pct": FLOOR_PCT,
                         "smooth_window": args.smooth, "mass_kg": rig.mass,
                         "root": "GMR passthrough (position + orientation) for BOTH tracks",
                         "effort_limits_Nm": {rig.hn[i]: float(rig.effort[i])
                                              for i in range(len(rig.hn))}},
              "clips": {}}

    for fam, name in clips:
        s = byname[name]
        st, n = s["start"], min(s["len"], args.frames)
        # ---- A: GMR retarget, straight out of the cache
        gA6, _, _ = build_global(GL[st:st + n].float(), cspec, "cpu")
        RgA = rot6d_to_matrix(gA6).double()
        # ---- B: Greenwich (human -> codes -> GR3 descriptor)
        h6, hp, reach = build_global(HL[st:st + n].float(), hs, "cpu")
        codes = G.encode(torch.cat([h6, hp / reach], -1).to(G.device), hs, hd)
        rotB, _ = G.decode_full(codes, gspec, gdof)
        RgB = rot6d_to_matrix(rotB).double().cpu()
        del codes, rotB
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        qA_pre, qB_pre = angles_from_R(RgA, False), angles_from_R(RgB, False)
        qA, qB = angles_from_R(RgA, True), angles_from_R(RgB, True)

        # root: GMR passthrough for both
        root_p = (RW[st:st + n].numpy().astype(np.float64) @ AX) / 100.0    # cm Y-up -> m Z-up
        root_q = Rot.from_matrix(AX.T @ RgA[:, root_j].numpy() @ AX).as_quat()[:, [3, 0, 1, 2]]
        qpA = rig.qpos(root_p, root_q, qA)
        qpB = rig.qpos(root_p, root_q, qB)

        # per-clip floor from A (the ground truth), applied to BOTH
        lowA = np.array([min(rig.kin(qpA[t])[0][s2][:, 2].min() for s2 in ("left", "right"))
                         for t in range(n)])
        floor = float(np.percentile(lowA, FLOOR_PCT))

        entry = {"family": fam, "frames": int(n), "floor_m": floor}
        for tag, qp in (("A_gmr", qpA), ("B_greenwich", qpB)):
            e = {}
            for var, sm in (("raw", 0), ("smooth", args.smooth)):
                qps, qv, qa = derivatives(qp, rig, dt, smooth=sm)
                tf, bf, _, _ = inverse_dynamics(rig, qps, qv, qa, floor, contact=False)
                tc, bc, ncp, unm = inverse_dynamics(rig, qps, qv, qa, floor, contact=True)
                e[f"id_free_{var}"] = tau_stats(tf, rig)
                e[f"id_free_{var}"]["base_resid_force_over_weight_mean"] = float(
                    np.linalg.norm(bf[:, 0:3], axis=1).mean() / rig.weight)
                e[f"id_contact_{var}"] = tau_stats(tc, rig)
                e[f"id_contact_{var}"].update({
                    "contact_points_mean": float(ncp.mean()),
                    "unmatched_wrench_frac": float(unm.mean()),
                    "base_resid_force_over_weight_mean": float(
                        np.linalg.norm(bc[:, 0:3], axis=1).mean() / rig.weight)})
                if var == "raw":
                    e["joint_vel_p95_rad_s"] = float(np.percentile(np.abs(qv[:, rig.hdof]), 95))
                    e["joint_acc_p95_rad_s2"] = float(np.percentile(np.abs(qa[:, rig.hdof]), 95))
            e["limits_pre_projection"] = limit_stats(qA_pre if tag == "A_gmr" else qB_pre)
            e["limits_post_projection"] = limit_stats(qA if tag == "A_gmr" else qB)
            e["feet"] = foot_metrics(rig, qp, floor)
            entry[tag] = e

        # ---- A vs B per-frame difference
        jA = np.stack([qA[nm] for nm in rig.hn], 1)
        jB = np.stack([qB[nm] for nm in rig.hn], 1)
        dj = np.degrees(jA - jB)
        bpA = np.array([rig.kin(qpA[t])[2] for t in range(n)])
        bpB = np.array([rig.kin(qpB[t])[2] for t in range(n)])
        per_j = np.sqrt((dj ** 2).mean(0))
        entry["A_vs_B"] = {
            "joint_rmse_deg": float(np.sqrt((dj ** 2).mean())),
            "joint_rmse_p95_deg": float(np.percentile(np.abs(dj), 95)),
            "body_pos_rmse_cm": float(np.sqrt((np.linalg.norm(bpA - bpB, axis=-1) ** 2).mean()) * 100),
            "worst_joints_rmse_deg": {rig.hn[i]: round(float(per_j[i]), 2)
                                      for i in np.argsort(-per_j)[:6]},
        }
        report["clips"][name] = entry

        a, b = entry["A_gmr"], entry["B_greenwich"]
        print(f"\n== [{fam}] {name} ({n} f)")
        print(f"   limits pre  A {a['limits_pre_projection']['violation_frac']:.4f} "
              f"(max {a['limits_pre_projection']['worst_overshoot_deg']:.1f} deg)   "
              f"B {b['limits_pre_projection']['violation_frac']:.4f} "
              f"(max {b['limits_pre_projection']['worst_overshoot_deg']:.1f} deg)")
        for k in ("id_free_raw", "id_free_smooth", "id_contact_raw"):
            print(f"   {k:16s} p95|tau|/lim  A {a[k]['tau_p95_frac']:6.3f}  B {b[k]['tau_p95_frac']:6.3f}"
                  f"   over-limit  A {a[k]['over_limit_frac']:.4f}  B {b[k]['over_limit_frac']:.4f}")
        print(f"   A vs B  joint RMSE {entry['A_vs_B']['joint_rmse_deg']:.2f} deg   "
              f"body RMSE {entry['A_vs_B']['body_pos_rmse_cm']:.2f} cm")

    # ------------------------------------------------------------ aggregate
    def agg(names):
        o = {}
        for tag in ("A_gmr", "B_greenwich"):
            o[tag] = {}
            for k in ("id_free_raw", "id_free_smooth", "id_contact_raw", "id_contact_smooth"):
                for f in ("tau_mean_frac", "tau_p95_frac", "over_limit_frac", "frames_any_over"):
                    o[tag][f"{k}.{f}"] = float(np.mean([report["clips"][n][tag][k][f] for n in names]))
            for k in ("limits_pre_projection", "limits_post_projection"):
                for f in ("violation_frac", "worst_overshoot_deg"):
                    o[tag][f"{k}.{f}"] = float(np.mean([report["clips"][n][tag][k][f] for n in names]))
            for f in ("planted_frac", "steps_per_s", "skate_mean_cm_s", "skate_p95_cm_s",
                      "ankle_roll_z_mean_cm", "foot_low_z_min_cm", "foot_low_z_max_cm"):
                o[tag][f"feet.{f}"] = float(np.mean([report["clips"][n][tag]["feet"][f] for n in names]))
            o[tag]["joint_vel_p95_rad_s"] = float(np.mean([report["clips"][n][tag]["joint_vel_p95_rad_s"] for n in names]))
            o[tag]["joint_acc_p95_rad_s2"] = float(np.mean([report["clips"][n][tag]["joint_acc_p95_rad_s2"] for n in names]))
        for f in ("joint_rmse_deg", "body_pos_rmse_cm"):
            o[f"A_vs_B.{f}"] = float(np.mean([report["clips"][n]["A_vs_B"][f] for n in names]))
        o["n_clips"] = len(names)
        return o

    fams = collections.defaultdict(list)
    for n, e in report["clips"].items():
        fams[e["family"]].append(n)
    report["by_family"] = {f: agg(ns) for f, ns in fams.items()}
    report["overall"] = agg(list(report["clips"]))

    # per-joint aggregate (overall) for the free-ID raw variant
    pj = {}
    for tag in ("A_gmr", "B_greenwich"):
        pj[tag] = {j: float(np.mean([report["clips"][n][tag]["id_free_raw"]["per_joint_p95_frac"][j]
                                     for n in report["clips"]])) for j in rig.hn}
    report["overall"]["per_joint_p95_frac_id_free_raw"] = pj

    # clips where B is catastrophically worse
    flags = []
    for n, e in report["clips"].items():
        a, b = e["A_gmr"], e["B_greenwich"]
        ratio = (b["id_free_raw"]["tau_p95_frac"] + 1e-9) / (a["id_free_raw"]["tau_p95_frac"] + 1e-9)
        if ratio > 2.0 or b["id_free_raw"]["over_limit_frac"] > 0.02 or e["A_vs_B"]["joint_rmse_deg"] > 25:
            flags.append({"clip": n, "family": e["family"], "p95_ratio_B_over_A": round(float(ratio), 2),
                          "B_over_limit_frac": b["id_free_raw"]["over_limit_frac"],
                          "joint_rmse_deg": e["A_vs_B"]["joint_rmse_deg"]})
    report["flagged"] = sorted(flags, key=lambda x: -x["p95_ratio_B_over_A"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("\nsaved", args.out)

    o = report["overall"]
    print("\n================ OVERALL ================")
    print(f"{'metric':44s} {'A=GMR':>10s} {'B=Greenwich':>12s}")
    for k in ("id_free_raw.tau_p95_frac", "id_free_raw.over_limit_frac",
              "id_free_smooth.tau_p95_frac", "id_free_smooth.over_limit_frac",
              "id_contact_raw.tau_p95_frac", "id_contact_raw.over_limit_frac",
              "limits_pre_projection.violation_frac", "limits_pre_projection.worst_overshoot_deg",
              "limits_post_projection.violation_frac",
              "joint_vel_p95_rad_s", "joint_acc_p95_rad_s2",
              "feet.planted_frac", "feet.steps_per_s", "feet.skate_mean_cm_s",
              "feet.ankle_roll_z_mean_cm", "feet.foot_low_z_max_cm"):
        print(f"{k:44s} {o['A_gmr'][k]:10.4f} {o['B_greenwich'][k]:12.4f}")
    print(f"{'A_vs_B.joint_rmse_deg':44s} {o['A_vs_B.joint_rmse_deg']:23.4f}")
    print(f"{'A_vs_B.body_pos_rmse_cm':44s} {o['A_vs_B.body_pos_rmse_cm']:23.4f}")
    print("\nflagged clips:", json.dumps(report["flagged"], indent=1))


if __name__ == "__main__":
    main()
