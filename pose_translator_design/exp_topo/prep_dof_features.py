"""Per-joint DOF descriptors (rotation axes + limits) for every cached embodiment.

Writes {cache}/{emb}_dof.npz  ->  float32 `dof` [J,19]:
  slot k in 0..2 : [ax, ay, az, limited, half_range/pi, center/pi]   (18 values)
  then           : [dof_count / 3]
plus float32 `rest` [J,3,3] (and `rest6d` [J,6]): the STATIC rest rotation R_rest.

Axis convention: LOCAL (parent-kept-frame) and Y-up harmonized, identical to the
cache's local rot6d, i.e.
    R_local(q) = rot(A_1,q_1) @ ... @ rot(A_n,q_n) @ R_rest
    A_k = AX @ (B_1...B_k) @ jnt_axis_k        (AX = MuJoCo Z-up -> ardy Y-up)
    R_rest = AX @ (B_1...B_n) @ AX^T           (SAME Y-up frame as the axes)
with B_i the fixed MJCF <body quat> along the merged chain, hinges at 0.
Without R_rest a consumer cannot go angles -> rotations; it is stored so that the
hinge-angle <-> rotation map is exact (round trip to ~1e-7, float32 noise).

R_rest for the ROOT joint is the identity by convention: the root's "local"
rotation is its global orientation, which comes from the free joint / the data and
is not reconstructible from hinge angles.  FK consumers must pass the root's global
rotation through (see manifold_project.fk_from_angles(root_R=...)).
"""
from __future__ import annotations
import os, sys, json
import numpy as np

EXP = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
sys.path.insert(0, EXP)
import prep_gmr_cache as PG
import prep_xwbc_cache as PX
import mujoco
from scipy.spatial.transform import Rotation as Rot

AX = PG.AX
CACHE_X = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan"
CACHE_P = "/var/tmp/alphamotion/topo_exp/cache_paired_fixed"
EXT = PX.EXT

REPORT = {}


def qmat(q):
    return Rot.from_quat(np.asarray(q, float), scalar_first=True).as_matrix()


def pack(slots, dof_count):
    """slots: list of (axis3, limited, half, center) -> 19 floats."""
    v = np.zeros(19, np.float32)
    for k, (a, lim, half, cen) in enumerate(slots[:3]):
        v[6 * k:6 * k + 3] = a
        v[6 * k + 3] = lim
        v[6 * k + 4] = half
        v[6 * k + 5] = cen
    v[18] = dof_count / 3.0
    return v


def ball_dof(J):
    """3-DOF ball joints (SMPL / BVH): orthonormal basis, unlimited.
    R_rest = I: the stored local rotation IS the joint rotation, and
    rot(x,q0) rot(y,q1) rot(z,q2) is an intrinsic-XYZ Euler chart covering SO(3)."""
    D = np.zeros((J, 19), np.float32)
    for k, a in enumerate(np.eye(3)):
        D[:, 6 * k:6 * k + 3] = a
    D[:, 18] = 1.0
    return D, np.tile(np.eye(3, dtype=np.float32), (J, 1, 1))


def rest6d(Rst):
    """[J,3,3] -> [J,6] (first two columns), the rot6d convention used everywhere."""
    return np.ascontiguousarray(np.asarray(Rst, np.float32)[:, :, :2].transpose(0, 2, 1)
                                ).reshape(-1, 6)


# --------------------------------------------------------------- MJCF path ---

def chain_from(model, b_top_excl, b_bot_incl):
    """body ids strictly below b_top_excl down to b_bot_incl, parent-most first."""
    out = []
    c = int(b_bot_incl)
    while c != b_top_excl:
        out.append(c)
        c = int(model.body_parentid[c])
        assert c >= 0, "not an ancestor"
    return out[::-1]


def body_hinges(model, b):
    a, n = int(model.body_jntadr[b]), int(model.body_jntnum[b])
    return [j for j in range(a, a + n)] if a >= 0 else []


def mjcf_dof(emb, xml, spec, ranges=None):
    """`ranges`: optional {mjcf_joint_name: (lo, hi)} in radians that OVERRIDES the
    MJCF's own jnt_range when encoding the limit features (axes, rest rotations and
    the tree still come from the MJCF).  Used for robots whose kinematic MJCF is a
    third party's retargeting model with hand-narrowed ranges — G1, where GMR's
    g1_mocap_29dof.xml clips 15 of 29 joints; see prep_xwbc_cache.LIMIT_URDF."""
    model = mujoco.MjModel.from_xml_path(xml)
    keep, frame = PG.build_merge(model)
    assert len(keep) == spec_J(spec), (emb, len(keep), spec_J(spec))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert [names[b] for b in keep] == list(spec["joint_names"]), emb
    kset = set(keep)

    D = np.zeros((len(keep), 19), np.float32)
    slot_names = []
    nonhinge = []
    rest = {}
    axes_raw = {}
    for k, b in enumerate(keep):
        if k == 0:
            chain = chain_from(model, 0, frame[b])        # world -> frame[root]
        else:
            p = int(model.body_parentid[b])
            while p >= 1 and p not in kset:
                p = int(model.body_parentid[p])
            chain = chain_from(model, frame[p], frame[b])
        B = np.eye(3)
        slots, snames = [], []
        for c in chain:
            B = B @ qmat(model.body_quat[c])
            for j in body_hinges(model, c):
                jt = model.jnt_type[j]
                jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
                if jt != mujoco.mjtJoint.mjJNT_HINGE:
                    nonhinge.append((jn, int(jt)))
                    continue
                a = AX @ (B @ model.jnt_axis[j])
                a = a / np.linalg.norm(a)
                lim = float(model.jnt_limited[j])
                lo, hi = model.jnt_range[j]
                if ranges and jn in ranges:
                    lo, hi = ranges[jn]
                    lim = 1.0
                half = (hi - lo) / 2 / np.pi if lim else 0.0
                cen = (hi + lo) / 2 / np.pi if lim else 0.0
                slots.append((a, lim, half, cen))
                snames.append(jn)
        rest[k] = B
        axes_raw[k] = [AX.T @ s[0] for s in slots]
        assert len(slots) <= 3, (emb, spec["joint_names"][k], len(slots))
        D[k] = pack(slots, len(slots))
        slot_names.append(",".join(snames))
    # Y-up rest rotations, same conjugation as the cache's globals (R' = AX R AX^T).
    # Root: identity by convention (its rotation is the free joint's, from data).
    Rst = np.stack([np.eye(3) if k == 0 else AX @ rest[k] @ AX.T
                    for k in range(len(keep))]).astype(np.float32)
    return D, slot_names, model, keep, frame, rest, axes_raw, nonhinge, Rst


def spec_J(spec):
    return len(spec["parents"])


def verify_composition(emb, model, keep, frame, rest, axes_raw, D, seed=0):
    """Random hinge angles -> MuJoCo FK local rotation vs predicted composition."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    kset = set(keep)
    worst = 0.0
    for trial in range(4):
        q = np.zeros(model.nq)
        q[3] = 1.0                                       # free-joint quat w (if any)
        ang = {}
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = model.jnt_range[j]
                v = rng.uniform(max(lo, -1.0), min(hi, 1.0)) if model.jnt_limited[j] \
                    else rng.uniform(-1.0, 1.0)
                q[model.jnt_qposadr[j]] = v
                ang[j] = v
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        R = data.xmat.reshape(-1, 3, 3)
        for k, b in enumerate(keep):
            if k == 0:
                continue
            p = int(model.body_parentid[b])
            while p >= 1 and p not in kset:
                p = int(model.body_parentid[p])
            L = R[frame[p]].T @ R[frame[b]]
            n = int(round(D[k, 18] * 3))
            P = np.eye(3)
            # angles in chain order: recover from the slot joints
            chain = chain_from(model, frame[p], frame[b])
            js = [j for c in chain for j in body_hinges(model, c)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
            for a_raw, j in zip(axes_raw[k], js):
                P = P @ Rot.from_rotvec(a_raw * ang[j]).as_matrix()
            P = P @ rest[k]
            worst = max(worst, float(np.abs(P - L).max()))
    return worst


# ----------------------------------------------------- data-tree (h2/talos) ---

def datatree_dof(emb, spec, clip):
    """Recover per-joint hinge axis directly from the stored body kinematics
    (each npz body has exactly one hinge), then borrow limits from the MJCF by
    matching the recovered angle track against joint_pos columns."""
    R, P = PX.data_tree_globals(emb, f"{EXT}/{emb}/{clip}.npz", 4000)   # Y-up already
    R = R.astype(np.float64)
    d = np.load(f"{EXT}/{emb}/{clip}.npz")
    jp = d["joint_pos"][:R.shape[0]].astype(np.float64)
    jnames = [str(x) for x in d["joint_names"]]
    parents = spec["parents"]
    J = len(parents)
    T = R.shape[0]
    sub = np.linspace(0, T - 1, min(600, T)).astype(int)

    model = mujoco.MjModel.from_xml_path(PX.resolve_xml(emb))
    lim_by_name = {}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lim_by_name[n] = (float(model.jnt_limited[j]), model.jnt_range[j].copy())

    D = np.zeros((J, 19), np.float32)
    Rst = np.tile(np.eye(3), (J, 1, 1))
    slot_names = []
    worst_resid = 0.0
    worst_rest = 0.0
    unmatched = []
    for k in range(J):
        if parents[k] < 0:
            slot_names.append("")
            continue
        L = np.einsum("tab,tbc->tac", R[sub, parents[k]].transpose(0, 2, 1), R[sub, k])
        Rel = np.einsum("tab,cb->tac", L, L[0])            # L(t) @ L(0)^T = rot(a, dtheta)
        rv = Rot.from_matrix(Rel).as_rotvec()              # [t,3], all along a
        u, s, vt = np.linalg.svd(rv - 0.0, full_matrices=False)
        a = vt[0] / np.linalg.norm(vt[0])
        th = rv @ a
        # residual of the 1-DOF assumption
        resid = np.abs(rv - th[:, None] * a).max()
        worst_resid = max(worst_resid, float(resid))
        # match to a joint_pos column (dtheta must equal +-(q(t)-q(0)))
        dq = jp[sub] - jp[sub][0]
        err = np.abs(dq - th[:, None]).max(0)
        errn = np.abs(dq + th[:, None]).max(0)
        c, cn = int(err.argmin()), int(errn.argmin())
        if err[c] <= errn[cn]:
            col, sign, e = c, 1.0, err[c]
        else:
            col, sign, e = cn, -1.0, errn[cn]
        if e > 1e-3:
            unmatched.append((k, float(e)))
            name, lim, half, cen = "", 0.0, 0.0, 0.0
        else:
            a = a * sign                                   # +axis = +joint_pos
            name = jnames[col]
            lm = lim_by_name.get(name)
            if lm is None:
                lim, half, cen = 0.0, 0.0, 0.0
                name += "(no-mjcf)"
            else:
                lim = lm[0]
                lo, hi = lm[1]
                half = (hi - lo) / 2 / np.pi if lim else 0.0
                cen = (hi + lo) / 2 / np.pi if lim else 0.0
            # L(t) = rot(a, q(t)) R_rest  ->  R_rest = rot(a, -q(t)) L(t), constant in t.
            # Average over the sampled frames (SVD-projected) instead of trusting one.
            qs = jp[sub, col]
            Rr = np.einsum("tab,tbc->tac",
                           Rot.from_rotvec(-qs[:, None] * a[None, :]).as_matrix(), L)
            u, _, vt = np.linalg.svd(Rr.mean(0))
            M = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
            worst_rest = max(worst_rest, float(np.abs(Rr - M).max()))
            Rst[k] = M
        D[k] = pack([(a, lim, half, cen)], 1)
        slot_names.append(name)
    return D, slot_names, worst_resid, unmatched, Rst.astype(np.float32), worst_rest


# ------------------------------------------------------------------- g1_30 ---

def g1_30_dof(spec):
    model = mujoco.MjModel.from_xml_path(PG.ARDY_G1_XML)
    J = spec_J(spec)
    D = np.zeros((J, 19), np.float32)
    Rst = np.tile(np.eye(3), (J, 1, 1))
    slot_names = []
    for j, name in enumerate(spec["joint_names"]):
        if spec["parents"][j] < 0:
            slot_names.append("")
            continue
        jn = str(name).replace("_skel", "_joint")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, jn
        bid = model.jnt_bodyid[jid]
        Q = qmat(model.body_quat[bid])
        Rst[j] = AX @ Q @ AX.T                 # local = Q @ rot(axis,q) -> rest = AX Q AX^T
        a = AX @ (Q @ model.jnt_axis[jid])
        a = a / np.linalg.norm(a)
        lim = float(model.jnt_limited[jid])
        lo, hi = model.jnt_range[jid]
        half = (hi - lo) / 2 / np.pi if lim else 0.0
        cen = (hi + lo) / 2 / np.pi if lim else 0.0
        D[j] = pack([(a, lim, half, cen)], 1)
        slot_names.append(jn)
    return D, slot_names, Rst.astype(np.float32)


# --------------------------------------------------------------------- main ---

def load_spec(cache, emb):
    d = np.load(f"{cache}/{emb}_spec.npz", allow_pickle=True)
    return {"parents": d["parents"], "joint_names": [str(x) for x in d["joint_names"]]}


def check(emb, D):
    assert not np.isnan(D).any(), emb
    n = np.linalg.norm(D.reshape(-1, 19)[:, :18].reshape(-1, 3, 6)[:, :, :3], axis=-1)
    bad = ((n > 1e-5) & (np.abs(n - 1) > 1e-5)).sum()
    counts = np.bincount(np.round(D[:, 18] * 3).astype(int), minlength=4)
    return int(bad), counts.tolist(), int(round(D[:, 18].sum() * 3))


def main():
    clips = PX.clip_names()
    ref = clips[0]
    out = {}

    # ---- cache_xwbc_lafan ----
    man = json.load(open(f"{CACHE_X}/manifest.json"))
    embs = man["embodiments"]
    for emb in embs:
        spec = load_spec(CACHE_X, emb)
        J = spec_J(spec)
        note = ""
        if emb == "human_smpl":
            D, Rst = ball_dof(J)
            snames = ["ball(x,y,z)"] * J
            note = "SMPL ball joints; R_rest = I"
        elif emb in PX.DATA_TREE:
            D, snames, resid, unmatched, Rst, wrest = datatree_dof(emb, spec, ref)
            note = (f"data-tree axis recovery: max 1-DOF residual {resid:.2e} rad; "
                    f"R_rest constancy max|err|={wrest:.2e}; unmatched joints {unmatched}")
        else:
            xml = PX.resolve_xml(emb)
            rng = PX.limit_override(emb)
            D, snames, model, keep, frame, rest, axes_raw, nonhinge, Rst = \
                mjcf_dof(emb, xml, spec, ranges=rng)
            w = verify_composition(emb, model, keep, frame, rest, axes_raw, D)
            note = f"MJCF {os.path.basename(xml)}; composition max|err|={w:.2e}"
            if rng:
                note += (f"; limits from vendor URDF "
                         f"{os.path.basename(PX.LIMIT_URDF[emb])}")
            if nonhinge:
                note += f"; non-hinge joints skipped: {sorted(set(nonhinge))}"
        bad, counts, tot = check(emb, D)
        assert Rst.shape == (J, 3, 3), (emb, Rst.shape)
        np.savez(f"{CACHE_X}/{emb}_dof.npz", dof=D, rest=Rst.astype(np.float32),
                 rest6d=rest6d(Rst),
                 joint_names=np.array(spec["joint_names"], dtype=object),
                 slot_joints=np.array(snames, dtype=object))
        out[emb] = {"cache": "cache_xwbc_lafan", "J": J, "dof_hist_0123": counts,
                    "total_dof": tot, "nonunit_axes": bad, "note": note}
        print(f"[{emb}] J={J} hist(0/1/2/3)={counts} total={tot} bad_axes={bad} :: {note}",
              flush=True)

    # ---- cache_paired_fixed ----
    for emb in ["soma77", "g1_30"]:
        spec = load_spec(CACHE_P, emb)
        J = spec_J(spec)
        if emb == "soma77":
            D, Rst = ball_dof(J)
            snames = ["ball(x,y,z)"] * J
            note = "BVH ball joints; R_rest = I"
        else:
            D, snames, Rst = g1_30_dof(spec)
            note = "ardy g1skel34/xml/g1.xml, one hinge per spec joint"
        bad, counts, tot = check(emb, D)
        assert Rst.shape == (J, 3, 3), (emb, Rst.shape)
        np.savez(f"{CACHE_P}/{emb}_dof.npz", dof=D, rest=Rst.astype(np.float32),
                 rest6d=rest6d(Rst),
                 joint_names=np.array(spec["joint_names"], dtype=object),
                 slot_joints=np.array(snames, dtype=object))
        out[emb] = {"cache": "cache_paired_fixed", "J": J, "dof_hist_0123": counts,
                    "total_dof": tot, "nonunit_axes": bad, "note": note}
        print(f"[{emb}] J={J} hist(0/1/2/3)={counts} total={tot} bad_axes={bad} :: {note}",
              flush=True)

    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "dof_summary.json"), "w"), indent=2)

    # ---- (a) leg-chain sanity print ----
    print("\n=== leg chain axes (Y-up: X=lateral/roll-ish, Y=up/yaw, Z=fwd/pitch-ish) ===")
    for emb in ["unitree_g1_29dof", "unitree_h1", "fourier_n1"]:
        d = np.load(f"{CACHE_X}/{emb}_dof.npz", allow_pickle=True)
        D, jn, sn = d["dof"], list(d["joint_names"]), list(d["slot_joints"])
        print(f"\n-- {emb}")
        for j, n in enumerate(jn):
            if not (n.lower().startswith(("left", "l_")) and
                    any(t in n.lower() for t in ("hip", "knee", "ankle", "thigh",
                                                 "shin", "calf", "foot", "toe"))):
                continue
            nd = int(round(D[j, 18] * 3))
            ax = [np.round(D[j, 6 * k:6 * k + 3], 3).tolist() for k in range(nd)]
            rg = [(round(float(D[j, 6 * k + 4] * 180), 1), round(float(D[j, 6 * k + 5] * 180), 1))
                  for k in range(nd)]
            print(f"   {n:28s} dof={nd} axes={ax} half/center_deg={rg} [{sn[j]}]")
        # hip triple orthogonality
        idx = [j for j, n in enumerate(jn) if "left" in n.lower() and
               ("hip" in n.lower() or "thigh" in n.lower())]
        vs = []
        for j in idx:
            nd = int(round(D[j, 18] * 3))
            vs += [D[j, 6 * k:6 * k + 3] for k in range(nd)]
        if len(vs) >= 2:
            G = np.abs(np.array(vs) @ np.array(vs).T)
            np.fill_diagonal(G, 0)
            print(f"   hip-group max |dot| between distinct axes = {G.max():.4f} "
                  f"({len(vs)} axes)")


if __name__ == "__main__":
    main()
