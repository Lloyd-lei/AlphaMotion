"""GT-FREE PHYSICAL CONSISTENCY EVALUATION.

Every number in this file is computed from a joint-angle trajectory and the
robot's own MJCF.  Nothing is compared against a reference motion.  That is the
whole point: our ground truth is a retargeter's output and we have measured it
to be wrong (xwbc GT breaks the real G1's joint limits on 1-5% of frames, BONES
GT on 27.5%, the pnd_adam walk reference is 93 deg off at the root).  A metric
that needs GT inherits GT's errors.  These metrics do not.

INPUT
    qpos : [T, nq] float, MuJoCo convention for the given model.  If the model
           has a floating base the first 7 entries are (x,y,z, qw,qx,qy,qz) in
           WORLD frame -- feed the root_world side channel, not root-relative
           data, or the contact/balance/slip metrics are meaningless.
    fps  : sampling rate of the trajectory.

WHAT IS MEASURED (all per-frame + summarised)

  0. joint_limits   -- qpos vs model.jnt_range.  Free, and it is the metric that
                       started this whole audit.
  1. actuator       -- inverse dynamics.  qvel/qacc by finite difference on a
                       zero-phase low-pass filtered qpos, then mj_inverse WITH THE
                       CONSTRAINT SOLVER DISABLED, giving the pure rigid-body
                       requirement  tau_req = M(q)qacc + C(q,qvel) - passive.
                       Contact forces are then solved for explicitly: find the
                       friction-cone-feasible contact forces at the detected
                       contact points that best cancel the 6-dof ROOT RESIDUAL
                       (the base is unactuated, so whatever is left there is
                       physically unexplainable).  The joint torques the motors
                       must supply are what remains after those contact forces,
                       compared against the vendor's own torque limits.

                       Why not just let mj_inverse use its constraint inverse:
                       MuJoCo's soft contact model turns a 5 mm ground-offset
                       error into a 20x-bodyweight normal force, so that number
                       measures your z-calibration, not the motion.  Measured:
                       a G1 standing still, sunk 5 mm, reported a 23-bodyweight
                       root residual and 21% of actuators over limit.  The
                       explicit force solve is flat over the same offset range.
  2. contact        -- foot geoms vs floor.  Foot slip (world tangential speed
                       of the material point at the contact), airborne fraction,
                       floor penetration depth.
  3. balance        -- CoM from the model's own mass properties, projected to
                       the ground, vs the convex hull of active contact points.
                       Static margin AND the instantaneous capture point
                       (CoM_xy + v_xy*sqrt(z/g)), because a walking robot is
                       SUPPOSED to leave its static support polygon and scoring
                       that as failure would be the same mistake as trusting GT.
  4. self_collision -- contacts not involving the floor.  MuJoCo already filters
                       parent-child pairs, so what is left is real.  A vendor
                       baseline (the MJCF's own rest/key pose) is reported
                       alongside, since e.g. the agibot X2 ships a rest pose that
                       already self-penetrates ~7 mm; our trajectories should be
                       judged against that floor, not against zero.
  5. smoothness     -- joint-space jerk percentiles and step discontinuities.

CONVENTIONS / CAVEATS, stated once so the numbers can be read honestly:
  * Filter: 4th-order Butterworth, zero-phase (filtfilt), default cutoff 6 Hz,
    applied to qpos before differencing.  Raw differencing of a 30 fps decode
    produces jerk numbers dominated by quantisation, which would flatter nobody
    and inform nobody.  The UNFILTERED jerk is also reported (smoothness metric)
    because there the noise IS the signal.
  * Quaternion: filtered componentwise then renormalised.  Fine for the small
    per-frame rotations here; noted because it is an approximation.
  * Contact candidates are detected with an inflated floor margin (default 2 cm)
    so a foot hovering a few mm above the plane still counts as a support point
    for the force solve.  None of these models predict root height, so without
    that tolerance the feasibility numbers would be an artefact of grounding.
    The CONTACT metrics (penetration, slip) still use dist <= 0, i.e. real touch.
  * A frame with no contact candidate is FLIGHT.  There the residual is the
    deviation from a ballistic arc, which is the correct test for the airborne
    phase of a run, and it is reported separately from stance.
  * Torque limits: forcerange if the actuator is force-limited, else
    gear*ctrlrange if ctrl-limited, else unknown (excluded, and counted).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

G = 9.81

FOOT_TOKENS = ("foot", "ankle", "sole", "toe", "heel", "feet")
FLOOR_TOKENS = ("floor", "ground", "plane", "world")


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_model(xml_path: str, add_floor: bool = True):
    """Load an MJCF; guarantee a floor plane exists so contact metrics mean
    something.  Returns (model, floor_geom_id)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    fid = _find_floor(model)
    if fid is not None or not add_floor:
        return model, fid

    # No floor in this XML -> re-load through MjSpec and append one.
    spec = mujoco.MjSpec.from_file(xml_path)
    g = spec.worldbody.add_geom()
    g.name = "physeval_floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0.0, 0.0, 0.05]
    g.pos = [0.0, 0.0, 0.0]
    g.condim = 3
    model = spec.compile()
    return model, _find_floor(model)


def _find_floor(model):
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            return i
    for i in range(model.ngeom):
        n = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        if any(t in n for t in FLOOR_TOKENS):
            return i
    return None


def foot_geoms(model, floor_gid):
    """Geoms belonging to a body whose name (or an ancestor's) looks like a foot.

    Falls back to 'the two lowest-in-the-tree leaf bodies of the legs' only if
    the name heuristic finds nothing, in which case we say so in the report.
    """
    names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
             for i in range(model.nbody)]
    is_foot_body = np.zeros(model.nbody, bool)
    for b in range(model.nbody):
        n = names[b]
        if any(t in n for t in FOOT_TOKENS):
            is_foot_body[b] = True
    # inherit downward: children of a foot body are part of the foot
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if is_foot_body[p]:
            is_foot_body[b] = True

    out = {}
    for gi in range(model.ngeom):
        if gi == floor_gid:
            continue
        if model.geom_contype[gi] == 0 and model.geom_conaffinity[gi] == 0:
            continue                                    # visual-only geom, never collides
        b = model.geom_bodyid[gi]
        if is_foot_body[b]:
            # side: left / right by body name, else by rest-y sign
            n = names[b]
            side = "left" if ("left" in n or n.startswith("l_") or "_l_" in n) else (
                "right" if ("right" in n or n.startswith("r_") or "_r_" in n) else "unknown")
            out[gi] = (b, side)
    return out, bool(len(out))


def _collidable_geom_verts(model):
    """Local-frame vertices (or a bounding proxy) for every collidable geom.

    Cached per model so the grounding sweep is one matmul per frame.
    """
    key = "_physeval_verts"
    if hasattr(model, key):
        return getattr(model, key)
    out = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue                          # the floor is the reference, not a surface
        if model.geom_bodyid[g] == 0:
            continue                          # anything welded to the world
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            a, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + n].reshape(-1, 3).astype(np.float64)
            if n > 400:                       # subsample big meshes; the hull is what matters
                v = v[np.linspace(0, n - 1, 400).astype(int)]
            out.append((g, v, 0.0))
        else:
            out.append((g, np.zeros((1, 3)), float(model.geom_rbound[g])))
    try:
        setattr(model, key, out)
    except AttributeError:
        pass
    return out


def lowest_surface_z(model, qpos, data=None):
    """Per-frame lowest point of any collidable surface, world z.  [T]"""
    q = np.atleast_2d(np.asarray(qpos, np.float64))
    data = data or mujoco.MjData(model)
    geoms = _collidable_geom_verts(model)
    lo = np.empty(q.shape[0])
    for t in range(q.shape[0]):
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        m = np.inf
        for g, v, r in geoms:
            R = data.geom_xmat[g].reshape(3, 3)
            p = data.geom_xpos[g]
            m = min(m, float((v @ R.T)[:, 2].min() + p[2] - r))
        lo[t] = m
    return lo


def ground_offset(model, qpos, pct=50.0):
    """Constant vertical shift so the pct-th percentile of the per-frame lowest
    surface point lands on the floor.

    NONE of the methods compared here predict root height -- the decoders emit
    joint angles only -- so without a stated, identical grounding rule the contact
    and balance metrics would measure the calibration instead of the motion.

    The default is the MEDIAN, not the minimum and not a low percentile.  Reasons,
    both learned the hard way on this data:
      * pinning the minimum to zero makes penetration identically zero by
        construction and destroys the metric;
      * pinning a low percentile (10) makes the robot hover: the LAFAN GT retarget
        of a G1 walk has its lowest foot point wandering between 0.3 and 2.6 cm
        above any fixed plane, never actually landing, so a p10 calibration
        reported 88% of a WALK as airborne.
    Median calibration means roughly half the frames have the lowest point at or
    below the plane, which is what a gait actually looks like.  The residual
    quantity of interest is then FOOT CLEARANCE SPREAD (reported under `contact`):
    how far the foot wanders vertically relative to a fixed floor.  That number is
    calibration-invariant and is the honest version of "does this motion respect
    the ground".
    """
    lo = lowest_surface_z(model, qpos)
    return float(-np.percentile(lo, pct)), lo


def actuator_torque_limits(model):
    """Per-actuator torque limit AT THE JOINT, plus the dof it drives.

    Source priority, most-to-least trustworthy:
      1. jnt_actfrcrange  -- the joint's <actuatorfrcrange>.  This is where the
         vendors actually put the motor spec (G1: 88/139/50 Nm, X2: 120/60/36).
      2. actuator_forcerange.
      3. gear*ctrlrange, but ONLY for a plain motor (biastype == none).  For a
         position servo ctrlrange is an ANGLE and multiplying it by gear yields a
         meaningless "limit" -- the g1_mocap XML is exactly that trap (it would
         hand you a 1.0 Nm limit on the knee and every frame would look saturated).
      4. unknown -> excluded from the over-limit rate, and counted in the report.

    Returns list of dicts: {aid, name, joint, dof, gear, limit(nan), source}.
    """
    rows = []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}"
        trntype = model.actuator_trntype[a]
        gear = float(model.actuator_gear[a, 0])
        dof, jid = None, None
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[a, 0])
            dof = int(model.jnt_dofadr[jid])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"jnt{jid}"
        else:
            jname = f"<trn{int(trntype)}>"

        lim, src = float("nan"), "unknown"
        if jid is not None and model.jnt_actfrclimited[jid]:
            lo, hi = model.jnt_actfrcrange[jid]
            if max(abs(lo), abs(hi)) > 0:
                lim, src = float(max(abs(lo), abs(hi))), "jnt_actuatorfrcrange"
        if not np.isfinite(lim) and model.actuator_forcelimited[a]:
            lo, hi = model.actuator_forcerange[a]
            if max(abs(lo), abs(hi)) > 0:
                lim, src = float(max(abs(lo), abs(hi))), "actuator_forcerange"
        if (not np.isfinite(lim) and model.actuator_ctrllimited[a]
                and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE):
            lo, hi = model.actuator_ctrlrange[a]
            if max(abs(lo), abs(hi)) > 0:
                lim = float(max(abs(lo), abs(hi)) * abs(gear))
                src = "gear*ctrlrange(motor)"
        rows.append(dict(aid=a, name=name, joint=jname, dof=dof, gear=gear,
                         limit=lim, source=src))
    return rows


# --------------------------------------------------------------------------- #
# kinematics: filtered derivatives
# --------------------------------------------------------------------------- #
def filter_qpos(model, qpos, fps, cutoff_hz=6.0, order=4):
    """Zero-phase low-pass on qpos.  Quaternion blocks renormalised after."""
    q = np.asarray(qpos, np.float64).copy()
    T = q.shape[0]
    nyq = 0.5 * fps
    if cutoff_hz is None or cutoff_hz >= nyq or T < 3 * (order + 1) * 2:
        return q, False
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    qf = filtfilt(b, a, q, axis=0, padlen=min(3 * max(len(a), len(b)), T - 1))
    # renormalise quaternion blocks (free joints and ball joints)
    for j in range(model.njnt):
        t = model.jnt_type[j]
        adr = int(model.jnt_qposadr[j])
        if t == mujoco.mjtJoint.mjJNT_FREE:
            s = slice(adr + 3, adr + 7)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            s = slice(adr, adr + 4)
        else:
            continue
        blk = qf[:, s]
        n = np.linalg.norm(blk, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        qf[:, s] = blk / n
    return qf, True


def derivatives(model, q, fps):
    """qvel [T,nv] and qacc [T,nv] from qpos, quaternion-aware.

    Central differences in the interior, one-sided at the ends.  qvel is
    computed with mj_differentiatePos (correct on the SO(3) parts); qacc is a
    plain central difference of qvel, which lives in the tangent space and so is
    a vector space -- that is legitimate.
    """
    T, dt = q.shape[0], 1.0 / fps
    nv = model.nv
    vel = np.zeros((T, nv))
    tmp = np.zeros(nv)
    for t in range(T):
        i0, i1 = max(t - 1, 0), min(t + 1, T - 1)
        h = (i1 - i0) * dt
        if h <= 0:
            continue
        mujoco.mj_differentiatePos(model, tmp, h, q[i0], q[i1])
        vel[t] = tmp
    acc = np.zeros((T, nv))
    acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt)
    if T > 1:
        acc[0] = (vel[1] - vel[0]) / dt
        acc[-1] = (vel[-1] - vel[-2]) / dt
    return vel, acc


def _pct(x, ps=(50, 90, 95, 99)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps} | {"max": float("nan"), "mean": float("nan")}
    d = {f"p{p}": float(np.percentile(x, p)) for p in ps}
    d["max"] = float(x.max())
    d["mean"] = float(x.mean())
    return d


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
@dataclass
class PhysEvalConfig:
    fps: float = 30.0
    cutoff_hz: float = 6.0
    filter_order: int = 4
    contact_pen_tol: float = 1e-4      # m; deeper than this counts as penetration
    slip_speed_tol: float = 0.02       # m/s; contact point moving faster = slip
    jerk_disc_k: float = 8.0           # step > k * median |dq| counts as a discontinuity
    settle_frames: int = 0
    floor_margin: float = 0.02         # m; contact-candidate tolerance for the force solve
    torque_ref_len: float = 0.5        # m; scales root torque rows to force units in the solve


def evaluate(model, qpos, cfg: PhysEvalConfig, floor_gid=None, label="",
             baseline_pairs=None):
    """Full GT-free physical-consistency report for one trajectory.

    baseline_pairs: {(bodyA, bodyB): depth_m} from vendor_baseline().  Those pairs
    are ALSO reported net of the vendor's own rest-pose penetration, because a
    robot whose shipped rest pose already has the head 7.1 mm inside the torso
    (agibot X2) would otherwise score 100% self-colliding on every trajectory
    anyone ever gives it, and that says nothing about the trajectory.
    """
    q_raw = np.asarray(qpos, np.float64)
    assert q_raw.ndim == 2 and q_raw.shape[1] == model.nq, \
        f"qpos {q_raw.shape} != model.nq {model.nq}"
    T = q_raw.shape[0]
    data = mujoco.MjData(model)
    if floor_gid is None:
        floor_gid = _find_floor(model)
    feet, feet_named = foot_geoms(model, floor_gid)
    q, filtered = filter_qpos(model, q_raw, cfg.fps, cfg.cutoff_hz, cfg.filter_order)
    vel, acc = derivatives(model, q, cfg.fps)

    # -- contact candidates need a tolerance; the dynamics must not use MuJoCo's
    #    soft-constraint inverse (see module docstring).  Both are restored after.
    old_margin = model.geom_margin.copy()
    old_gap = model.geom_gap.copy()
    if floor_gid is not None:
        # MuJoCo's plane broadphase only widens the search if BOTH geoms carry the
        # margin, so it goes on the floor and on the feet -- and nowhere else, so
        # the self-collision pass is not polluted with near-misses.
        for gi in [floor_gid] + list(feet):
            model.geom_margin[gi] = max(float(old_margin[gi]), cfg.floor_margin)

    has_free = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    nroot = 6 if has_free else 0

    rep = {
        "label": label,
        "n_frames": int(T),
        "fps": cfg.fps,
        "filter": (f"butterworth order {cfg.filter_order}, zero-phase, "
                   f"cutoff {cfg.cutoff_hz} Hz" if filtered else "none (clip too short / cutoff>=nyquist)"),
        "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
                  "nbody": int(model.nbody), "floating_base": bool(has_free),
                  "total_mass": float(model.body_mass.sum()),
                  "floor_geom": None if floor_gid is None else
                  (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, floor_gid) or f"geom{floor_gid}"),
                  "n_foot_geoms": len(feet), "foot_geoms_from_names": feet_named},
    }

    # ---------------- 0. joint limits -------------------------------------- #
    rep["joint_limits"] = _joint_limits(model, q_raw)

    # ---------------- per-frame sweep -------------------------------------- #
    torque = np.full((T, model.nv), np.nan)
    root_res = np.full((T, 6), np.nan)
    n_footc = np.zeros(T, int)
    slip = []           # (t, geom, speed)
    pen = []            # (t, geom, depth)
    com = np.zeros((T, 3))
    com_margin = np.full(T, np.nan)
    cap_margin = np.full(T, np.nan)
    has_support = np.zeros(T, bool)
    selfcol_depth = np.zeros(T)
    selfcol_excess = np.zeros(T)          # net of the vendor's own rest-pose penetration
    selfcol_pairs = {}
    base = {tuple(sorted(k)): v for k, v in (baseline_pairs or {}).items()}

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    n_cand = np.zeros(T, int)

    Ma = np.zeros(model.nv)
    try:
        for t in range(T):
            data.qpos[:] = q[t]
            data.qvel[:] = vel[t]
            data.qacc[:] = acc[t]
            mujoco.mj_forward(model, data)      # kinematics, collision, bias, passive
            mujoco.mj_subtreeVel(model, data)   # fills subtree_linvel for the CoM velocity
            # Rigid-body requirement WITHOUT MuJoCo's constraint inverse.  Built by
            # hand rather than by mj_inverse because disabling the constraint solver
            # also disables collision detection, and we need the contacts.  This is
            # exactly what mj_inverse would return with mjDSBL_CONSTRAINT set.
            mujoco.mj_mulM(model, data, Ma, acc[t])
            tau_req = Ma + data.qfrc_bias - data.qfrc_passive
            com[t] = data.subtree_com[0]

            cpts, cand_B, cand_dirs = [], [], []
            for ci in range(data.ncon):
                c = data.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 == floor_gid) or (g2 == floor_gid):
                    other = g2 if g1 == floor_gid else g1
                    if other not in feet:
                        continue                  # a non-foot body near the floor
                    dist = float(c.dist)
                    pos = np.array(c.pos)
                    bid = int(model.geom_bodyid[other])
                    mujoco.mj_jac(model, data, jacp, jacr, pos, bid)
                    # support candidate for the force solve (uses the margin)
                    n_cand[t] += 1
                    frame = np.array(c.frame).reshape(3, 3)
                    nrm, t1, t2 = frame[0], frame[1], frame[2]
                    mu = float(c.friction[0])
                    for sx in (+1, -1):
                        for ty in (t1, t2):
                            d = nrm + mu * sx * ty
                            cand_dirs.append(d)
                            cand_B.append(jacp.T @ d)      # [nv] generalised force / unit
                    if dist <= 0:                          # REAL touch
                        n_footc[t] += 1
                        cpts.append(pos)
                        if -dist > cfg.contact_pen_tol:
                            pen.append((t, other, -dist))
                        v = jacp @ vel[t]
                        vt = v - np.dot(v, nrm) * nrm
                        slip.append((t, other, float(np.linalg.norm(vt))))
                elif float(c.dist) < 0:
                    d = -float(c.dist)
                    selfcol_depth[t] = max(selfcol_depth[t], d)
                    key = tuple(sorted((
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(model.geom_bodyid[g1])) or str(g1),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(model.geom_bodyid[g2])) or str(g2))))
                    selfcol_pairs[key] = max(selfcol_pairs.get(key, 0.0), d)
                    selfcol_excess[t] = max(selfcol_excess[t], d - base.get(key, 0.0))

            # ---- solve for the friction-feasible contact forces ---------------
            if has_free and cand_B:
                Bm = np.stack(cand_B, 1)                   # [nv, K]
                a = _solve_contact(Bm[:6], tau_req[:6], cfg.torque_ref_len)
                gen = Bm @ a
                torque[t] = tau_req - gen
                root_res[t] = tau_req[:6] - gen[:6]
            else:
                torque[t] = tau_req
                if has_free:
                    root_res[t] = tau_req[:6]

            if cpts:
                has_support[t] = True
                P3 = np.asarray(cpts)
                P = P3[:, :2]                              # support polygon, ground plane
                com_margin[t] = _hull_margin(P, com[t, :2])
                zc = max(com[t, 2] - float(P3[:, 2].mean()), 1e-3)
                vcom = _com_velocity(model, data)
                icp = com[t, :2] + vcom[:2] * np.sqrt(zc / G)
                cap_margin[t] = _hull_margin(P, icp)
    finally:
        model.geom_margin[:] = old_margin
        model.geom_gap[:] = old_gap

    rep["actuator"] = _actuator_report(model, torque, root_res, has_free, n_cand > 0)
    rep["contact"] = _contact_report(model, T, n_footc, slip, pen, feet, cfg,
                                     foot_clearance(model, q, feet))
    rep["balance"] = _balance_report(com, com_margin, cap_margin, has_support)
    rep["self_collision"] = _selfcol_report(T, selfcol_depth, selfcol_pairs, cfg,
                                            selfcol_excess, base)
    rep["smoothness"] = _smoothness_report(model, q_raw, q, cfg)
    rep["headline"] = _headline(rep)
    return rep


def _solve_contact(B6, r6, Lref, ridge=1e-6):
    """Non-negative friction-pyramid contact forces that best cancel the root residual.

    B6 : [6, K] generalised force on the 6 base dofs per unit of each pyramid edge.
    r6 : [6]    the base-dof force the rigid-body dynamics demands.
    Returns a [K] >= 0 (edge magnitudes).  Unilateral + inside the friction cone by
    construction, because every column is already an edge of the cone.

    The rotational rows are scaled by 1/Lref so N and Nm are commensurate; without
    that the solve would silently trade a 1 Nm torque error against a 1 N force error.
    """
    from scipy.optimize import nnls
    w = np.ones(6)
    w[3:] = 1.0 / max(Lref, 1e-3)
    A = B6 * w[:, None]
    b = r6 * w
    K = A.shape[1]
    A_r = np.vstack([A, ridge * np.eye(K)])       # tiny ridge -> unique min-norm solution
    b_r = np.concatenate([b, np.zeros(K)])
    try:
        a, _ = nnls(A_r, b_r, maxiter=200 * K)
    except Exception:
        a = np.zeros(K)
    return a


def _com_velocity(model, data):
    """CoM linear velocity, world frame, from subtree_linvel of the world body."""
    return np.array(data.subtree_linvel[0])


def _hull_margin(P, pt):
    """Signed distance from pt to the convex hull of P in 2D.  + = inside."""
    P = np.asarray(P, float)
    if P.shape[0] == 1:
        return -float(np.linalg.norm(pt - P[0]))
    if P.shape[0] == 2 or np.linalg.matrix_rank(P - P.mean(0), tol=1e-6) < 2:
        a, b = P[0], P[-1]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-9:
            return -float(np.linalg.norm(pt - a))
        s = np.clip(np.dot(pt - a, ab) / (L * L), 0, 1)
        return -float(np.linalg.norm(pt - (a + s * ab)))
    try:
        h = ConvexHull(P)
    except QhullError:
        return -float(np.min(np.linalg.norm(P - pt, axis=1)))
    # hull equations: A.x + b <= 0 inside
    d = h.equations[:, :2] @ pt + h.equations[:, 2]
    return float(-d.max())


# --------------------------------------------------------------------------- #
# sub-reports
# --------------------------------------------------------------------------- #
def _joint_limits(model, q):
    """% of (joint, frame) outside the MJCF's own jnt_range, worst overshoot."""
    rows, viol_frames = [], np.zeros(q.shape[0], bool)
    tot = 0
    over = 0
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        adr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        x = q[:, adr]
        d = np.maximum(lo - x, x - hi)
        bad = d > 1e-6
        tot += x.size
        over += int(bad.sum())
        viol_frames |= bad
        if bad.any():
            rows.append(dict(
                joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}",
                pct_frames=100.0 * bad.mean(),
                worst_overshoot_deg=float(np.degrees(d.max())),
                range_deg=[float(np.degrees(lo)), float(np.degrees(hi))]))
    rows.sort(key=lambda r: -r["worst_overshoot_deg"])
    return dict(
        definition="qpos vs model.jnt_range for limited hinge/slide joints; overshoot in deg",
        pct_joint_frames_over=100.0 * over / max(tot, 1),
        pct_frames_any_violation=100.0 * float(viol_frames.mean()) if q.shape[0] else 0.0,
        worst_overshoot_deg=max([r["worst_overshoot_deg"] for r in rows], default=0.0),
        n_limited_joints=int(model.jnt_limited.sum()),
        offenders=rows[:10])


def _actuator_report(model, torque, root_res, has_free, stance):
    lims = actuator_torque_limits(model)
    T = torque.shape[0]
    per_act, n_pairs, n_over, worst = [], 0, 0, 0.0
    ratios_all = []
    for r in lims:
        if r["dof"] is None or not np.isfinite(r["limit"]) or r["limit"] <= 0:
            continue
        tau = torque[:, r["dof"]]
        ratio = np.abs(tau) / r["limit"]
        ratios_all.append(ratio)
        n_pairs += T
        n_over += int((ratio > 1.0).sum())
        worst = max(worst, float(np.nanmax(ratio)) if np.isfinite(ratio).any() else 0.0)
        per_act.append(dict(actuator=r["name"], joint=r["joint"], limit_Nm=r["limit"],
                            limit_source=r["source"],
                            pct_frames_over=100.0 * float((ratio > 1.0).mean()),
                            peak_ratio=float(np.nanmax(ratio)),
                            p99_ratio=float(np.nanpercentile(ratio, 99))))
    per_act.sort(key=lambda d: -d["peak_ratio"])
    unknown = [r["name"] for r in lims if not np.isfinite(r["limit"])]

    out = dict(
        definition=("mj_inverse (constraint solver OFF) on filtered finite-difference "
                    "qvel/qacc gives M*qacc+C; friction-cone contact forces are then solved "
                    "to cancel the unactuated base residual, and what is left at each joint "
                    "is the torque the motor must supply. Ratio = |that| / vendor torque "
                    "limit. >1 = the motor physically cannot do it."),
        n_actuators_scored=len(per_act),
        n_actuators_unknown_limit=len(unknown),
        pct_actuator_frames_over_limit=100.0 * n_over / max(n_pairs, 1),
        worst_overshoot_ratio=worst,
        saturating_actuators=[d for d in per_act if d["pct_frames_over"] > 0][:12],
        top_loaded=per_act[:6],
    )
    if ratios_all:
        R = np.stack(ratios_all, 1)
        out["ratio_stats"] = _pct(R.ravel())
        out["pct_frames_any_actuator_over"] = 100.0 * float((R > 1.0).any(1).mean())
    if has_free:
        f = np.linalg.norm(root_res[:, :3], axis=1)
        m = np.linalg.norm(root_res[:, 3:], axis=1)
        mass = float(model.body_mass.sum())
        bw = mass * G
        st, fl = np.asarray(stance, bool), ~np.asarray(stance, bool)
        out["root_residual"] = dict(
            definition=("6-dof wrench on the floating base still unaccounted for after the "
                        "best friction-feasible contact forces. No actuator can produce it, "
                        "so it is the part of the motion physics had to supply and could not. "
                        "Normalised by body weight. STANCE frames: is the ground reaction "
                        "consistent with the motion. FLIGHT frames: is the body on a "
                        "ballistic arc (a real run's flight phase should be ~0 here too)."),
            body_weight_N=bw,
            stance_fraction=float(st.mean()),
            force_N=_pct(f), torque_Nm=_pct(m),
            force_over_bodyweight=_pct(f / bw),
            stance_force_over_bodyweight=_pct(f[st] / bw),
            flight_force_over_bodyweight=_pct(f[fl] / bw),
            torque_over_bwL=_pct(m / (bw * 0.5)))
    return out


def foot_clearance(model, qpos, feet):
    """Per-frame lowest point of any FOOT surface, world z.  [T]

    The one contact number that survives any grounding choice is its SPREAD: a
    constant vertical offset shifts the whole trace and cannot change p95 - p5.
    A motion that respects the ground plants the foot at the same height every
    step, so the spread is small; a motion that skates or sinks has a large one.
    """
    q = np.atleast_2d(np.asarray(qpos, np.float64))
    data = mujoco.MjData(model)
    allg = {g: (v, r) for g, v, r in _collidable_geom_verts(model)}
    fg = [(g, *allg[g]) for g in feet if g in allg]
    if not fg:
        return np.full(q.shape[0], np.nan)
    lo = np.empty(q.shape[0])
    for t in range(q.shape[0]):
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        m = np.inf
        for g, v, r in fg:
            R = data.geom_xmat[g].reshape(3, 3)
            m = min(m, float((v @ R.T)[:, 2].min() + data.geom_xpos[g][2] - r))
        lo[t] = m
    return lo


def _contact_report(model, T, n_footc, slip, pen, feet, cfg, clear=None):
    airborne = float((n_footc == 0).mean())
    slip_sp = np.array([s[2] for s in slip]) if slip else np.zeros(0)
    pen_d = np.array([p[2] for p in pen]) if pen else np.zeros(0)
    # per-frame max slip over contacts in that frame
    per_frame_slip = np.zeros(T)
    for t, _, s in slip:
        per_frame_slip[t] = max(per_frame_slip[t], s)
    per_frame_pen = np.zeros(T)
    for t, _, d in pen:
        per_frame_pen[t] = max(per_frame_pen[t], d)
    contact_frames = n_footc > 0
    cl = {}
    if clear is not None and np.isfinite(clear).any():
        c = clear[np.isfinite(clear)]
        cl = dict(
            foot_clearance_m=_pct(c, (5, 50, 95)),
            foot_clearance_spread_mm=float((np.percentile(c, 95) - np.percentile(c, 5)) * 1000),
            foot_clearance_std_mm=float(c.std() * 1000))
    return dict(**cl, **dict(
        definition=("foot-geom vs floor contacts. slip = world tangential speed of the material "
                    "point at the contact while in contact (m/s); penetration = -contact.dist."),
        airborne_fraction=airborne,
        mean_contacts_per_frame=float(n_footc.mean()),
        n_contact_events=int(len(slip)),
        slip_m_per_s=_pct(slip_sp),
        pct_contacts_slipping=100.0 * float((slip_sp > cfg.slip_speed_tol).mean()) if slip_sp.size else 0.0,
        slip_per_contact_frame_mean=float(per_frame_slip[contact_frames].mean()) if contact_frames.any() else float("nan"),
        penetration_m=_pct(pen_d),
        pct_frames_penetrating=100.0 * float((per_frame_pen > cfg.contact_pen_tol).mean()),
        max_penetration_mm=float(pen_d.max() * 1000) if pen_d.size else 0.0,
    ))


def _balance_report(com, com_margin, cap_margin, has_support):
    n = com.shape[0]
    sup = has_support
    def frac_out(m):
        mm = m[sup]
        mm = mm[np.isfinite(mm)]
        return float((mm < 0).mean()) if mm.size else float("nan")
    return dict(
        definition=("CoM from the model's mass properties, projected to z=0, vs the convex hull "
                    "of active foot contact points. margin>0 = inside (metres). ICP = "
                    "instantaneous capture point CoM_xy + v_xy*sqrt(h/g): the fair criterion for "
                    "motion that is supposed to be dynamic."),
        frames_with_support=float(sup.mean()),
        static_frac_outside_support=frac_out(com_margin),
        static_margin_m=_pct(com_margin[sup & np.isfinite(com_margin)]),
        icp_frac_outside_support=frac_out(cap_margin),
        icp_margin_m=_pct(cap_margin[sup & np.isfinite(cap_margin)]),
        com_height_m=_pct(com[:, 2]),
    )


def _selfcol_report(T, depth, pairs, cfg, excess=None, base=None):
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
    out = dict(
        definition=("mj_collision contacts not involving the floor with dist<0. MuJoCo already "
                    "excludes parent-child/welded pairs, so these are genuine self-penetrations. "
                    "The *_excess_vendor fields subtract the depth the SAME body pair already "
                    "has in the vendor's own rest pose, so a robot that ships self-penetrating "
                    "(agibot X2: head_pitch_link/torso_link, 7.11 mm) is comparable to one "
                    "that does not."),
        pct_frames_penetrating=100.0 * float((depth > cfg.contact_pen_tol).mean()),
        worst_depth_mm=float(depth.max() * 1000),
        depth_mm=_pct(depth * 1000),
        worst_pairs=[{"bodies": list(k), "depth_mm": v * 1000,
                      "vendor_rest_mm": (base or {}).get(k, 0.0) * 1000} for k, v in top],
    )
    if excess is not None:
        out["pct_frames_penetrating_excess_vendor"] = \
            100.0 * float((excess > cfg.contact_pen_tol).mean())
        out["worst_depth_excess_vendor_mm"] = float(max(excess.max(), 0.0) * 1000)
        out["vendor_pairs_discounted"] = {"/".join(k): v * 1000 for k, v in (base or {}).items()}
    return out


def _smoothness_report(model, q_raw, q_filt, cfg):
    dt = 1.0 / cfg.fps
    # hinge/slide dofs only (angle space); exclude free-joint quaternion
    idx = []
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            idx.append(int(model.jnt_qposadr[j]))
    idx = np.array(idx, int)
    out = {"definition": ("third difference of joint angles / dt^3 (rad/s^3) on hinge/slide dofs. "
                          "Reported on RAW qpos (filtering would hide exactly what this measures). "
                          "Discontinuity = |dq| step > k * median |dq| of that joint.")}
    if idx.size == 0 or q_raw.shape[0] < 4:
        out["jerk_rad_s3"] = _pct(np.zeros(0))
        return out
    X = q_raw[:, idx]
    d1 = np.diff(X, axis=0)
    jerk = np.diff(X, n=3, axis=0) / dt ** 3
    out["jerk_rad_s3"] = _pct(np.abs(jerk).ravel())
    out["jerk_rad_s3_filtered"] = _pct(np.abs(np.diff(q_filt[:, idx], n=3, axis=0) / dt ** 3).ravel())
    med = np.median(np.abs(d1), axis=0)
    med[med < 1e-6] = 1e-6
    disc = np.abs(d1) > cfg.jerk_disc_k * med
    out["pct_frames_with_discontinuity"] = 100.0 * float(disc.any(1).mean())
    out["n_discontinuities"] = int(disc.sum())
    out["max_step_deg"] = float(np.degrees(np.abs(d1).max()))
    out["max_joint_speed_deg_s"] = float(np.degrees(np.abs(d1).max() / dt))
    return out


def _headline(rep):
    a, c, b, s, sm, jl = (rep["actuator"], rep["contact"], rep["balance"],
                          rep["self_collision"], rep["smoothness"], rep["joint_limits"])
    h = dict(
        jointlim_pct=jl["pct_joint_frames_over"],
        torque_over_pct=a["pct_actuator_frames_over_limit"],
        torque_worst_ratio=a["worst_overshoot_ratio"],
        slip_p50_ms=c["slip_m_per_s"]["p50"],
        slip_p95_ms=c["slip_m_per_s"]["p95"],
        airborne_frac=c["airborne_fraction"],
        penetration_max_mm=c["max_penetration_mm"],
        foot_spread_mm=c.get("foot_clearance_spread_mm", float("nan")),
        icp_out_frac=b["icp_frac_outside_support"],
        selfcol_pct=s["pct_frames_penetrating"],
        selfcol_worst_mm=s["worst_depth_mm"],
        selfcol_pct_x=s.get("pct_frames_penetrating_excess_vendor", float("nan")),
        selfcol_worst_x_mm=s.get("worst_depth_excess_vendor_mm", float("nan")),
        jerk_p95=sm["jerk_rad_s3"]["p95"],
    )
    if "root_residual" in a:
        rr = a["root_residual"]
        h["root_resid_bw_p50"] = rr["force_over_bodyweight"]["p50"]
        h["root_resid_bw_p95"] = rr["force_over_bodyweight"]["p95"]
        h["root_resid_bw_stance_p50"] = rr["stance_force_over_bodyweight"]["p50"]
        h["stance_frac"] = rr["stance_fraction"]
    return h


# --------------------------------------------------------------------------- #
def vendor_baseline(model, cfg: PhysEvalConfig, floor_gid=None, n=4):
    """Physical consistency of the MJCF's OWN rest / keyframe poses.

    Held still.  This is the floor below which no trajectory on this robot can
    go, and it is not zero: some vendors ship a rest pose that already
    self-penetrates.  Reporting it separately is what makes our numbers readable.
    """
    if floor_gid is None:
        floor_gid = _find_floor(model)
    poses = {}
    q0 = np.array(model.qpos0, float)
    poses["qpos0"] = q0
    for k in range(model.nkey):
        kn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}"
        poses[kn] = np.array(model.key_qpos[k], float)
    out = {}
    for name, qp in poses.items():
        Q = np.repeat(qp[None], max(n, 8), 0)
        if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            dz, _ = ground_offset(model, Q[:1], pct=0.0)      # a static pose: just seat it
            Q = Q.copy()
            Q[:, 2] += dz
        r = evaluate(model, Q, cfg, floor_gid, label=f"vendor:{name}")
        out[name] = {"headline": r["headline"],
                     "self_collision": r["self_collision"],
                     "joint_limits": {k: r["joint_limits"][k] for k in
                                      ("pct_joint_frames_over", "worst_overshoot_deg")},
                     "root_height_m": float(qp[2]) if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else None}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml", required=True)
    ap.add_argument("--qpos", help="npz with key 'qpos', or .npy, or .pt [T,nq]")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vendor_only", action="store_true")
    args = ap.parse_args()

    model, fgid = load_model(args.xml)
    cfg = PhysEvalConfig(fps=args.fps, cutoff_hz=args.cutoff)
    res = {"xml": args.xml, "vendor_baseline": vendor_baseline(model, cfg, fgid)}
    if not args.vendor_only and args.qpos:
        if args.qpos.endswith(".npz"):
            q = np.load(args.qpos)["qpos"]
        elif args.qpos.endswith(".npy"):
            q = np.load(args.qpos)
        else:
            import torch
            q = torch.load(args.qpos, map_location="cpu").numpy()
        res["trajectory"] = evaluate(model, q, cfg, fgid, label=os.path.basename(args.qpos))
    txt = json.dumps(res, indent=2, default=float)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
        print("wrote", args.out)
    print(json.dumps(res.get("trajectory", res)["headline"]
                     if "trajectory" in res else res["vendor_baseline"], indent=2, default=float))


if __name__ == "__main__":
    main()
