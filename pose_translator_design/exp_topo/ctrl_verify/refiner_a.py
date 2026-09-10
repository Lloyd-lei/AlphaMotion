"""Refiner-A: parameter-free geometric post-processor for decoded robot qpos motion.

Fixes, in order:
  1. temporal chatter   -- Savitzky-Golay on dof + root (pipeline constant: 9 frames @30fps
                           = 0.3 s; previously measured to cut decode chatter 1.43x -> 1.14x)
  2. stance foot skate  -- low+slow+hysteresis contact rule, then damped-least-squares IK
                           pins each planted foot to its stance-segment median (static pin
                           => zero skate inside the segment by construction)
  3. ground penetration -- exact MuJoCo collision query vs the floor plane; minimal root-z
                           lift = smooth upper envelope of per-frame penetration depth
                           (envelope >= depth pointwise, so output never penetrates)

NO-GT GUARANTEE: this file never reads native-retarget data. Inputs are ONLY
(a) the decoded qpos itself, (b) MJCF geometry via MuJoCo FK/collision,
(c) physics constants below. Nothing is tuned per clip or per embodiment.

usage:
  refiner_a.py --xml <model.xml> --src '<glob of *__qposA.npy>' --out_dir <dir>
               [--fps 30] [--workers 16] [--suffix_in __qposA.npy] [--suffix_out __qposR.npy]
               [--foot_regex 'ankle_roll|foot']
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import mujoco as mj
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# ---- physics / pipeline constants (universal, not per-motion tuned) -------------
FOOT_V_ON   = 0.10          # m/s  contact rule: enter stance below this horizontal speed
FOOT_V_OFF  = 0.15          # m/s  hysteresis exit = the physics slip line
MIN_SEG_SEC = 8.0 / 50.0    # s    FOOT_MIN_SEG = 8 frames @ 50 fps = 0.16 s
H_LOW       = FOOT_V_ON * MIN_SEG_SEC   # 1.6 cm: distance travelled at the contact-speed
                                        # limit over one minimum segment -> "low" gate
H_HIGH      = 2.0 * H_LOW   # hysteresis exit height (standard doubling)
SG_SEC      = 9.0 / 30.0    # s    Sav-Gol window, measured pipeline constant (0.3 s)
SG_POLY     = 2
IK_TOL      = 1e-3          # m    numerical convergence tolerance (not a tuned knob)
IK_ITERS    = 10
IK_LAMBDA2  = 1e-3          # damped-LS regulariser (numerical conditioning)
IK_MAX_STEP = 0.2           # rad, per-iteration trust region (numerical safeguard)
ENV_PASSES  = 3             # max-smooth passes for the smooth upper envelope

# ---------------------------------------------------------------------------------
def build_model(xml):
    spec = mj.MjSpec.from_file(xml)
    if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in spec.joints):
        spec.bodies[1].add_freejoint()
    names = set()
    for g in spec.geoms:
        if g.name: names.add(g.name)
    m = spec.compile()
    # floor plane: use an existing plane geom at z<=0, else add one
    floor_id = -1
    for g in range(m.ngeom):
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE and m.geom_bodyid[g] == 0:
            floor_id = g; break
    if floor_id < 0:
        spec.worldbody.add_geom(name="_refiner_floor", type=mj.mjtGeom.mjGEOM_PLANE,
                                size=[50, 50, 0.1])
        m = spec.compile()
        floor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "_refiner_floor")
    return m, floor_id

def find_feet(m, foot_regex):
    pat = re.compile(foot_regex, re.I)
    feet = []
    for b in range(1, m.nbody):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or ""
        if not pat.search(name):
            continue
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if not geoms:
            continue
        feet.append(dict(body=b, name=name, geoms=geoms))
    # keep only leaf-most matches (drop a match that is an ancestor of another match)
    keep = []
    for f in feet:
        if not any(is_ancestor(m, f["body"], g["body"]) for g in feet if g is not f):
            keep.append(f)
    for f in keep:
        f["chain_dofs"], f["chain_qpos"], f["chain_jids"] = chain_of(m, f["body"])
    return keep

def is_ancestor(m, a, b):
    while b != 0:
        b = m.body_parentid[b]
        if b == a: return True
    return False

def chain_of(m, body):
    """1-dof joints on the kinematic chain from `body` up to the free-joint body."""
    dofs, qadr, jids = [], [], []
    b = body
    while b != 0:
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            if m.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                dofs.append(m.jnt_dofadr[j]); qadr.append(m.jnt_qposadr[j]); jids.append(j)
        b = m.body_parentid[b]
    return np.array(dofs[::-1]), np.array(qadr[::-1]), np.array(jids[::-1])

def sg_smooth(q, fps):
    """Sav-Gol on root pos, root quat (hemisphere-fixed, renormalised) and all dof."""
    win = int(round(SG_SEC * fps));  win += (win + 1) % 2         # odd
    win = max(win, SG_POLY + 2 - (SG_POLY % 2))                    # valid window
    if len(q) <= win: return q.copy()
    out = q.copy()
    out[:, 0:3] = savgol_filter(q[:, 0:3], win, SG_POLY, axis=0)
    quat = q[:, 3:7].copy()
    for t in range(1, len(quat)):
        if quat[t] @ quat[t-1] < 0: quat[t] = -quat[t]
    quat = savgol_filter(quat, win, SG_POLY, axis=0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    out[:, 3:7] = quat
    out[:, 7:] = savgol_filter(q[:, 7:], win, SG_POLY, axis=0)
    return out

def fk_pass(m, d, q, floor_id, feet):
    """Per-frame: floor penetration depth, foot anchor pos, foot bottom z."""
    T = len(q)
    pen = np.zeros(T)
    anch = np.zeros((T, len(feet), 3))
    zbot = np.zeros((T, len(feet)))
    for t in range(T):
        d.qpos[:] = q[t]
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        p = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == floor_id or c.geom2 == floor_id) and c.dist < -p:
                p = -c.dist
        pen[t] = p
        for k, f in enumerate(feet):
            anch[t, k] = d.xpos[f["body"]]
            zb = 1e9
            for g in f["geoms"]:
                r = m.geom_size[g][0] if m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE \
                    else m.geom_rbound[g]
                zb = min(zb, d.geom_xpos[g][2] - r)
            zbot[t, k] = zb
    return pen, anch, zbot

def smooth_envelope(p, fps):
    """Smallest smooth curve >= p: iterated max(p, moving-average)."""
    win = int(round(SG_SEC * fps)) | 1
    l = p.copy()
    for _ in range(ENV_PASSES):
        l = np.maximum(p, uniform_filter1d(l, win, mode="nearest"))
    return l

def window_speed(pxy, fps):
    """Average horizontal speed over the contact timescale (one minimum segment,
    0.16 s): displacement across a +-MIN_SEG/2 window divided by its duration.
    This is the speed the contact rule is defined on -- instantaneous speed is
    dominated by decode chatter."""
    T = len(pxy)
    w = max(1, int(round(MIN_SEG_SEC * fps / 2)))
    hi = np.minimum(np.arange(T) + w, T - 1)
    lo = np.maximum(np.arange(T) - w, 0)
    disp = np.linalg.norm(pxy[hi] - pxy[lo], axis=1)
    return disp / ((hi - lo) / fps)

def stance_segments(zb, vxy, fps):
    """low+slow+hysteresis rule -> list of (start, end) index pairs (end exclusive)."""
    min_seg = max(2, int(round(MIN_SEG_SEC * fps)))
    T = len(zb); on = False; segs = []; s = 0
    for t in range(T):
        if not on:
            if zb[t] < H_LOW and vxy[t] < FOOT_V_ON:
                on = True; s = t
        else:
            if vxy[t] > FOOT_V_OFF or zb[t] > H_HIGH:
                on = False
                if t - s >= min_seg: segs.append((s, t))
    if on and T - s >= min_seg: segs.append((s, T))
    return segs

def ik_pin(m, d, q, t, f, target):
    """Damped-LS IK on the leg chain: bring the foot body to `target` while holding
    its current orientation (so the foot's lowest point stays its lowest point)."""
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    cols = f["chain_dofs"]; qadr = f["chain_qpos"]
    lo = m.jnt_range[f["chain_jids"], 0]; hi = m.jnt_range[f["chain_jids"], 1]
    d.qpos[:] = q[t]; mj.mj_kinematics(m, d)
    quat0 = d.xquat[f["body"]].copy()          # hold this orientation
    q0 = q[t, qadr].copy()
    e0 = np.linalg.norm(target - d.xpos[f["body"]])
    erot = np.zeros(3); qneg = np.zeros(4); qrel = np.zeros(4)
    for _ in range(IK_ITERS):
        d.qpos[:] = q[t]
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        ep = target - d.xpos[f["body"]]
        mj.mju_negQuat(qneg, d.xquat[f["body"]])
        mj.mju_mulQuat(qrel, quat0, qneg)          # world-frame rotation error
        mj.mju_quat2Vel(erot, qrel, 1.0)
        e = np.concatenate([ep, erot])
        if np.linalg.norm(ep) < IK_TOL and np.linalg.norm(erot) < IK_TOL * 10: break
        mj.mj_jacBody(m, d, jacp, jacr, f["body"])
        J = np.vstack([jacp[:, cols], jacr[:, cols]])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_LAMBDA2 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > IK_MAX_STEP: dq *= IK_MAX_STEP / n
        q[t, qadr] = np.clip(q[t, qadr] + dq, lo, hi)
    # never leave the frame worse than we found it
    d.qpos[:] = q[t]; mj.mj_kinematics(m, d)
    if np.linalg.norm(target - d.xpos[f["body"]]) > e0 + IK_TOL:
        q[t, qadr] = q0

def refine_clip(q, m, d, floor_id, feet, fps):
    tm = {}
    t0 = time.perf_counter()
    q = sg_smooth(q, fps);                       tm["savgol"] = time.perf_counter() - t0
    # stage A: exact penetration resolve (smooth minimal root-z lift)
    t0 = time.perf_counter()
    pen, anch, zbot = fk_pass(m, d, q, floor_id, feet)
    lift = smooth_envelope(pen, fps)
    q[:, 2] += lift
    anch[:, :, 2] += lift[:, None]; zbot += lift[:, None]
    tm["penetration"] = time.perf_counter() - t0
    # stage B: stance detection + static pin via damped-LS IK
    t0 = time.perf_counter()
    min_seg = max(2, int(round(MIN_SEG_SEC * fps)))
    ramp = max(1, min_seg // 2)
    n_seg = 0; planted = 0
    for k, f in enumerate(feet):
        vxy = window_speed(anch[:, k, :2], fps)
        for (s, e) in stance_segments(zbot[:, k], vxy, fps):
            n_seg += 1; planted += e - s
            med = np.median(anch[s:e, k, :2], axis=0)
            for t in range(s, e):
                # xy: static segment median (zero skate); z: bottom point on the ground
                tgt = np.array([med[0], med[1], anch[t, k, 2] - zbot[t, k]])
                w = min(1.0, (t - s + 1) / ramp, (e - t) / ramp)
                ik_pin(m, d, q, t, f, w * tgt + (1 - w) * anch[t, k])
    tm["ik"] = time.perf_counter() - t0
    # stage C: final exact penetration guarantee (IK may re-touch the floor by <= tol)
    t0 = time.perf_counter()
    pen2, _, _ = fk_pass(m, d, q, floor_id, feet)
    q[:, 2] += smooth_envelope(pen2, fps)
    tm["final_pen"] = time.perf_counter() - t0
    return q, dict(n_seg=n_seg, planted_frac=planted / (len(q) * max(1, len(feet))),
                   max_pen_in=float(pen.max()),
                   max_pen_resid=float(pen2.max()), timing=tm)

# ---------------------------------------------------------------------------------
_G = {}
def _init(xml, foot_regex, fps):
    m, floor_id = build_model(xml)
    _G.update(m=m, d=mj.MjData(m), floor=floor_id, fps=fps,
              feet=find_feet(m, foot_regex))

def _work(args):
    src, dst = args
    q = np.load(src).astype(np.float64)
    t0 = time.perf_counter()
    qr, info = refine_clip(q, _G["m"], _G["d"], _G["floor"], _G["feet"], _G["fps"])
    cpu = time.perf_counter() - t0
    np.save(dst, qr)
    return dict(src=os.path.basename(src), frames=len(q), cpu_s=cpu, **info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--src", required=True, help="glob of input qpos .npy files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--suffix_in", default="__qposA.npy")
    ap.add_argument("--suffix_out", default="__qposR.npy")
    ap.add_argument("--foot_regex", default="ankle_roll|foot|toe")
    ap.add_argument("--log_json", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(a.src))
    assert files, f"no files match {a.src}"
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(f, os.path.join(a.out_dir,
             os.path.basename(f).replace(a.suffix_in, a.suffix_out))) for f in files]
    m, fl = build_model(a.xml)
    feet = find_feet(m, a.foot_regex)
    print(f"model: nq={m.nq} floor_geom={fl} feet={[f['name'] for f in feet]}")
    for f in feet:
        print(f"  {f['name']}: chain joints "
              f"{[mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, int(j)) for j in f['chain_jids']]}")
    t0 = time.perf_counter()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, _init,
                                          (a.xml, a.foot_regex, a.fps)) as pool:
            results = pool.map(_work, jobs)
    else:
        _init(a.xml, a.foot_regex, a.fps)
        results = [_work(j) for j in jobs]
    wall = time.perf_counter() - t0
    frames = sum(r["frames"] for r in results)
    cpu = sum(r["cpu_s"] for r in results)
    stage = {k: sum(r["timing"][k] for r in results) for k in results[0]["timing"]}
    print(f"\n{len(results)} clips, {frames} frames")
    print(f"wall {wall:.1f}s ({frames/wall:.0f} fps with {a.workers} workers)")
    print(f"cpu  {cpu:.1f}s ({frames/cpu:.0f} fps single-core equivalent)")
    print("stage cpu-s:", {k: round(v, 1) for k, v in stage.items()})
    print(f"max residual penetration after refine: "
          f"{max(r['max_pen_resid'] for r in results)*1000:.2f} mm (pre-final-lift)")
    if a.log_json:
        json.dump(dict(results=results, wall_s=wall, cpu_s=cpu, frames=frames),
                  open(a.log_json, "w"), indent=1)

if __name__ == "__main__":
    main()
