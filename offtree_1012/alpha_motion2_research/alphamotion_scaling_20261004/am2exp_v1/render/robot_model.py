"""Vendor-MJCF robot geometry bound rigidly to AM2 joint frames (read-only use of the MJCFs).

Convention (same as surface/build_verified_patches.py and rollout_tools/g1.py, verified per body):
  canonical joint rotation  R_j = AX @ xmat[frame_j](q) @ AX.T      (own-joint global, Y-up)
  canonical joint position  P_j = AX @ (xpos[keep_j](q) - xpos[root](q)) * 100   (cm, root-relative)
Each geom g gets a fixed (anchor j, localR, localP) with
  AX @ geom_xmat @ AX.T = R_j @ localR ,  AX @ (geom_xpos - xpos[root]) * 100 = R_j @ localP + P_j
verified over qpos0 + 7 seeded random q (rigid, < 1e-6 / 1e-4 cm).  Geoms with no rigid cached frame
(intermediate links of merged co-located chains) get the best approximate anchor and are flagged.

The cache skeleton (names, parents, non-root offsets) is rebuilt from the MJCF with the same merge rule
that produced the XWBC cache (exp_topo/prep_gmr_cache.build_merge, re-implemented here verbatim so no
foreign ``common`` module is imported) and must equal the frozen static spec exactly.

unitree_h2 is a DATA_TREE body (cache skeleton body00..31 inferred from stored kinematics): its nodes are
mapped to vendor MJCF bodies by the explicit node map of judge J1 (judge/pose_mjcf.H2_NODE_MAP, report-32
decoder_gauge approach; copied here and asserted equal in tests); binding is then rigid w.r.t. those MJCF body
frames.  (An earlier nearest-rest-offset Hungarian map put node 9 on imu_in_pelvis with a 19 cm residual,
because the true node, torso_link, is dropped by the merge rule.)

Regions: every visual and collision geom gets the region judge J1 gives the MJCF body that owns it
(judge/regions.region_of: name rules + per-robot overrides + root-link trunk/pelvis split by height), so the
colours in the video are the judge's regions.  The structural tree rule of render/regions.py is only the
fallback (judge absent, or a body name the judge rule does not cover; counted in audit['region_fallback']).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import _env
from ._env import AX
from .regions import robot_joint_regions

import mujoco  # noqa: E402  (MUJOCO_GL set by _env)

UNITREE_ROS = '/var/tmp/alphamotion/robot_models/unitree_ros/robots'
GMR_PARAMS = _env.ROOT.parent.parent / 'external' / 'GMR' / 'general_motion_retargeting' / 'params.py'
XML_KEY = {  # exp_topo/prep_xwbc_cache.XML_KEY (copied; '@' = absolute path)
    'unitree_g1_29dof': 'unitree_g1', 'unitree_g1_23dof': 'unitree_g1', 'unitree_h1': 'unitree_h1',
    'unitree_h1_2': 'unitree_h1_2', 'unitree_h2': f'@{UNITREE_ROS}/h2_description/unitree_h2_mjcf.xml',
    'unitree_r1': f'@{UNITREE_ROS}/r1_description/unitree_r1_mjcf.xml', 'booster_k1': 'booster_k1',
    'booster_t1': 'booster_t1', 'booster_t1_29dof': 'booster_t1_29dof', 'engineai_pm01': 'engineai_pm01',
    'fourier_gr3': 'fourier_gr3', 'fourier_n1': 'fourier_n1', 'hightorque_hi': 'hightorque_hi',
    'kuavo_s45': 'kuavo_s45', 'pal_talos': 'pal_talos', 'pnd_adam_lite': 'pnd_adam_lite', 'tienkung': 'tienkung'}
DATA_TREE = {'unitree_h2'}
# copy of judge/pose_mjcf.H2_NODE_MAP (AM2 unitree_h2 node -> vendor MJCF body)
H2_NODE_MAP = {0: 'pelvis', 1: 'left_hip_pitch_link', 2: 'right_hip_pitch_link', 3: 'waist_yaw_link',
               4: 'left_hip_roll_link', 5: 'right_hip_roll_link', 6: 'waist_roll_link',
               7: 'left_hip_yaw_link', 8: 'right_hip_yaw_link', 9: 'torso_link', 10: 'left_knee_link',
               11: 'right_knee_link', 12: 'head_pitch_link', 13: 'left_shoulder_pitch_link',
               14: 'right_shoulder_pitch_link', 15: 'left_ankle_roll_link', 16: 'right_ankle_roll_link',
               17: 'head_yaw_link', 18: 'left_shoulder_roll_link', 19: 'right_shoulder_roll_link',
               20: 'left_ankle_pitch_link', 21: 'right_ankle_pitch_link', 22: 'left_shoulder_yaw_link',
               23: 'right_shoulder_yaw_link', 24: 'left_elbow_link', 25: 'right_elbow_link',
               26: 'left_wrist_roll_link', 27: 'right_wrist_roll_link', 28: 'left_wrist_pitch_link',
               29: 'right_wrist_pitch_link', 30: 'left_wrist_yaw_link', 31: 'right_wrist_yaw_link'}

VISUAL_DEFAULT_RGBA = np.array([0.80, 0.82, 0.85, 1.0], np.float32)


@lru_cache(maxsize=1)
def _gmr_xml_dict():
    spec = importlib.util.spec_from_file_location('_am2render_gmr_params', str(GMR_PARAMS))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # params.py imports only pathlib
    return {k: str(v) for k, v in mod.ROBOT_XML_DICT.items()}


def resolve_xml(body: str) -> str:
    k = XML_KEY[body]
    return k[1:] if k.startswith('@') else _gmr_xml_dict()[k]


def build_merge(model, eps=1e-3):
    """Verbatim re-implementation of exp_topo/prep_gmr_cache.build_merge (keep, frame-by-body dict)."""
    nb = model.nbody
    parent = model.body_parentid
    pos = model.body_pos
    drop = {i for i in range(2, nb) if np.linalg.norm(pos[i]) < eps}

    def compute(drop):
        keep = [i for i in range(1, nb) if i not in drop]
        kset = set(keep)

        def nka(b):
            p = int(parent[b])
            while p >= 1 and p not in kset:
                p = int(parent[p])
            return p
        attach = {b: set() for b in keep}
        for c in keep:
            a = nka(c)
            if a >= 1:
                attach[a].add(int(parent[c]))
        conflicts = {b: s for b, s in attach.items() if len(s) > 1}
        return keep, attach, conflicts

    keep, attach, conflicts = compute(drop)
    while conflicts:
        for b, s in conflicts.items():
            for x in s:
                drop.discard(x)
        keep, attach, conflicts = compute(drop)
    dropped_children = {}
    for d in sorted(drop):
        dropped_children.setdefault(int(parent[d]), []).append(d)

    def chain_end(b):
        cur = b
        while len(dropped_children.get(cur, [])) == 1:
            cur = dropped_children[cur][0]
        return cur
    frame = {}
    for b in keep:
        s = attach[b]
        frame[b] = next(iter(s)) if len(s) == 1 else chain_end(b)
    return keep, frame


def rebuilt_spec(model, keep):
    """(names, parents, offsets_cm canonical) like prep_gmr_cache.build_spec, root offset left 0."""
    parent = model.body_parentid
    pos = model.body_pos
    kset = set(keep)
    remap = {b: k for k, b in enumerate(keep)}
    parents, offsets, names = [], [], []
    for b in keep:
        p = int(parent[b])
        while p >= 1 and p not in kset:
            p = int(parent[p])
        parents.append(remap[p] if p in kset else -1)
        acc = pos[b].copy()
        cur = int(parent[b])
        while cur >= 1 and cur not in kset:
            acc = acc + pos[cur]
            cur = int(parent[cur])
        offsets.append(acc * 100.0)
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f'body{b}')
    off = np.asarray(offsets, np.float64) @ AX.T
    off[0] = 0.0
    return names, np.asarray(parents, np.int64), off


def random_states(model, count=7, seed=20261004):
    """Same sampler as surface/build_verified_patches.random_states (copied, not imported)."""
    rng = np.random.default_rng(seed)
    states = [model.qpos0.copy()]
    for _ in range(count):
        q = model.qpos0.copy()
        for j in range(model.njnt):
            adr = int(model.jnt_qposadr[j])
            typ = int(model.jnt_type[j])
            if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                if model.jnt_limited[j]:
                    lo, hi = model.jnt_range[j]
                    q[adr] = rng.uniform(lo + .08 * (hi - lo), hi - .08 * (hi - lo))
                else:
                    q[adr] = rng.uniform(-.9, .9)
            elif typ == int(mujoco.mjtJoint.mjJNT_SLIDE):
                lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-.03, .03)
                q[adr] = rng.uniform(lo, hi)
            elif typ == int(mujoco.mjtJoint.mjJNT_FREE):
                q[adr:adr + 3] += rng.normal(0, .1, 3)
                quat = rng.normal(size=4)
                q[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
            elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
                quat = rng.normal(size=4)
                q[adr:adr + 4] = quat / np.linalg.norm(quat)
        states.append(q)
    return states


def _hull_vertices(v):
    if len(v) <= 16:
        return v
    try:
        from scipy.spatial import ConvexHull
        return v[np.unique(ConvexHull(v).vertices)]
    except Exception:
        return v


def _quat_mat(q):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, np.float64))
    return out.reshape(3, 3)


def _rodrigues(axis, th):
    """axis [3] unit, th [...] -> [...,3,3]."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = np.asarray(th, np.float64)[..., None, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


@dataclass
class Chain:
    """MJCF bodies strictly below frame[parent] down to frame[joint] (inclusive); their rotations are
    recovered from the data's relative rotation R_parent^T R_joint by best-fit hinge angles."""
    joint: int
    parent: int
    bodies: list               # MJCF body ids, parent side -> child side
    fixed: list                # per body: canonical body_quat rotation (3x3)
    bpos: list                 # per body: canonical body_pos (cm) in its MJCF parent frame
    hinges: list               # per body: list of canonical hinge axes (unit)
    q0: list                   # per body: list of qpos0 angles of those hinges
    limits: list = None        # per body: list of (lo, hi) per hinge (+-inf when unlimited)

    @property
    def n_hinge(self):
        return sum(len(h) for h in self.hinges)

    def rotations(self, th):
        """th [T,n] -> list of per-body relative rotations (w.r.t. frame[parent]) [T,3,3]."""
        T = th.shape[0]
        Rr = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        out = []
        k = 0
        for Qc, axes in zip(self.fixed, self.hinges):
            Rr = Rr @ Qc
            for a in axes:
                Rr = Rr @ _rodrigues(a, th[:, k])
                k += 1
            out.append(Rr)
        return out

    def solve(self, Mrel, iters=40):
        """Mrel [T,3,3] target relative rotation -> (th [T,n], residual_deg [T]).
        Frobenius least squares, multi-start; among (near-)exact solutions the one inside the hinge limits and
        closest to the previous frame (first frame: closest to 0) wins (3-hinge chains have two exact solutions
        that differ only in the intermediate links)."""
        T = Mrel.shape[0]
        n = self.n_hinge
        if n == 0:
            F = self.rotations(np.zeros((T, 0)))[-1]
            return np.zeros((T, 0)), _angle_deg(np.swapaxes(F, -1, -2) @ Mrel)

        def resid(th):
            return (self.rotations(th)[-1] - Mrel).reshape(T, 9)

        def gn(th):
            for _ in range(iters):
                r = resid(th)
                Jm = np.empty((T, 9, n))
                for k in range(n):
                    d = np.zeros(n)
                    d[k] = 1e-6
                    Jm[:, :, k] = (resid(th + d) - resid(th - d)) / 2e-6
                JTJ = np.swapaxes(Jm, 1, 2) @ Jm + 1e-4 * np.eye(n)
                step = np.linalg.solve(JTJ, (np.swapaxes(Jm, 1, 2) @ r[..., None]))[..., 0]
                th = th - np.clip(step, -0.6, 0.6)
                if np.max(np.abs(step)) < 1e-10:
                    break
            return np.arctan2(np.sin(th), np.cos(th))
        starts = [0.0, np.pi / 2, -np.pi / 2, np.pi]
        rng = np.random.default_rng(0)
        inits = [np.full((T, n), v) for v in starts] + [np.broadcast_to(rng.uniform(-np.pi, np.pi, n), (T, n)).copy() for _ in range(4)]
        cands = np.stack([gn(x) for x in inits])            # [S,T,n]
        errs = np.stack([_angle_deg(np.swapaxes(self.rotations(c)[-1], -1, -2) @ Mrel) for c in cands])  # [S,T]
        lo = np.array([l for ls in self.limits for l in [x[0] for x in ls]])
        hi = np.array([h for ls in self.limits for h in [x[1] for x in ls]])
        viol = (np.maximum(lo - cands, 0) + np.maximum(cands - hi, 0)).sum(-1)  # [S,T]
        best_err = errs.min(0)
        th = np.empty((T, n))
        err = np.empty(T)
        prev = np.zeros(n)
        for t in range(T):
            ok = errs[:, t] <= best_err[t] + 1e-3
            cost = np.where(ok, 0.0, 1e9) + 10.0 * viol[:, t] + np.abs(np.arctan2(np.sin(cands[:, t] - prev), np.cos(cands[:, t] - prev))).sum(-1)
            s = int(np.argmin(cost))
            th[t] = cands[s, t]
            err[t] = errs[s, t]
            prev = th[t]
        return th, err


def _angle_deg(A):
    c = np.clip((np.trace(A, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


@dataclass
class GeomSet:
    """Geoms of one kind ('visual' or 'collision').
    mode 0: rigid binding to AM2 joint `anchor` (exact over qpos0 + 7 random q).
    mode 1: geom on an MJCF body inside the merged chain of AM2 joint `chain` (pose via best-fit hinge angles);
            local transform relative to that body (canonical).
    mode 2: geom on a side body whose joints the AM2 skeleton does not carry -> rigid to `anchor` at qpos0."""
    gid: np.ndarray
    mode: np.ndarray
    anchor: np.ndarray         # AM2 joint whose frame carries the geom (mode 0/2) or the chain joint (mode 1)
    body: np.ndarray           # MJCF body id (mode 1)
    localR: np.ndarray         # [G,3,3] canonical
    localP: np.ndarray         # [G,3] cm canonical
    gtype: np.ndarray
    size_cm: np.ndarray
    hull: list
    region: np.ndarray

    def __len__(self):
        return len(self.gid)


@dataclass
class RobotModel:
    body: str
    xml: str
    model: object
    names: list
    parents: np.ndarray
    offsets: np.ndarray
    reach: float
    keep: np.ndarray
    frame: np.ndarray
    mode: str
    chains: dict = field(default_factory=dict)   # joint -> Chain (only chains with hinges that carry geoms)
    visual: GeomSet = None
    collision: GeomSet = None
    joint_region: np.ndarray = None
    region_info: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ posing
    def chain_poses(self, R, P, joints=None):
        """-> {body: (Rb [T,3,3], pb [T,3])}, {joint: residual_deg [T]} for the chain bodies."""
        poses, resid = {}, {}
        for j, ch in self.chains.items():
            if joints is not None and j not in joints:
                continue
            p = ch.parent
            Mrel = np.swapaxes(R[:, p], -1, -2) @ R[:, j]
            th, err = ch.solve(Mrel)
            rel = ch.rotations(th)
            Rprev = R[:, p]
            pprev = P[:, p]
            for b, Rr, bp in zip(ch.bodies, rel, ch.bpos):
                pb = pprev + np.einsum('tab,b->ta', Rprev, bp)
                Rb = R[:, p] @ Rr
                poses[b] = (Rb, pb)
                Rprev, pprev = Rb, pb
            resid[j] = err
        return poses, resid

    def geom_poses(self, gs: GeomSet, R, P, chain_cache=None):
        """R [T,J,3,3], P [T,J,3] canonical cm (root-relative) -> Gc [T,G,3,3], c [T,G,3] cm."""
        R = np.asarray(R, np.float64)
        P = np.asarray(P, np.float64)
        T, G = R.shape[0], len(gs)
        Gc = np.empty((T, G, 3, 3))
        c = np.empty((T, G, 3))
        m02 = gs.mode != 1
        if m02.any():
            Ra = R[:, gs.anchor[m02]]
            Gc[:, m02] = Ra @ gs.localR[m02][None]
            c[:, m02] = np.einsum('tgab,gb->tga', Ra, gs.localP[m02]) + P[:, gs.anchor[m02]]
        if (~m02).any():
            if chain_cache is None:
                chain_cache = self.chain_poses(R, P, set(gs.anchor[~m02].tolist()))[0]
            for k in np.flatnonzero(~m02):
                Rb, pb = chain_cache[int(gs.body[k])]
                Gc[:, k] = Rb @ gs.localR[k]
                c[:, k] = np.einsum('tab,b->ta', Rb, gs.localP[k]) + pb
        return Gc, c

    def lowest_y(self, gs: GeomSet, R, P, chain_cache=None):
        """Exact lowest canonical y (cm, root-relative) of every geom: [T,G]."""
        Gc, c = self.geom_poses(gs, R, P, chain_cache)
        M = Gc @ AX[None, None]  # MuJoCo geom-local vector (cm) -> canonical
        T, G = c.shape[:2]
        out = np.empty((T, G))
        for k in range(G):
            t = int(gs.gtype[k])
            s = gs.size_cm[k]
            m1 = M[:, k, 1, :]
            if t == int(mujoco.mjtGeom.mjGEOM_MESH):
                out[:, k] = (m1 @ gs.hull[k].T).min(-1)
            elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                out[:, k] = -s[0]
            elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                out[:, k] = -np.abs(m1[:, 2]) * s[1] - s[0]
            elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                a = np.clip(np.abs(m1[:, 2]), 0, 1)
                out[:, k] = -a * s[1] - s[0] * np.sqrt(1 - a ** 2)
            elif t == int(mujoco.mjtGeom.mjGEOM_BOX):
                out[:, k] = -(np.abs(m1) * s[None]).sum(-1)
            elif t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                out[:, k] = -np.sqrt(((m1 * s[None]) ** 2).sum(-1))
            else:
                out[:, k] = -s.max()
            out[:, k] += c[:, k, 1]
        return out

    def region_min_y(self, gs: GeomSet, R, P, chain_cache=None):
        """[T,15] minimum canonical y (cm, root-relative) per region (NaN when the region has no geom)."""
        low = self.lowest_y(gs, R, P, chain_cache)
        out = np.full((low.shape[0], 15), np.nan)
        for r in range(15):
            m = gs.region == r
            if m.any():
                out[:, r] = low[:, m].min(-1)
        return out


def _snapshots(model, keep_ids, frame_ids, root_id, rebase=None):
    data = mujoco.MjData(model)
    snaps = []
    for q in random_states(model):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        r = AX[None] @ data.xmat[frame_ids].reshape(-1, 3, 3) @ AX.T[None]
        if rebase is not None:
            r = r @ np.swapaxes(rebase, -1, -2)
        p = (data.xpos[keep_ids] - data.xpos[root_id]) @ AX.T * 100.0
        gp = (data.geom_xpos - data.xpos[root_id]) @ AX.T * 100.0
        gr = AX[None] @ data.geom_xmat.reshape(-1, 3, 3) @ AX.T[None]
        bp = (data.xpos - data.xpos[root_id]) @ AX.T * 100.0
        br = AX[None] @ data.xmat.reshape(-1, 3, 3) @ AX.T[None]
        snaps.append(dict(r=r, p=p, gp=gp, gr=gr, bp=bp, br=br))
    return snaps


def _build_chains(model, parents, keep, frame):
    """Chain (bodies between frame[parent] and frame[joint]) for every non-root AM2 joint."""
    chains = {}
    for j, p in enumerate(parents):
        if p < 0:
            continue
        path = []
        x = int(frame[j])
        while x > 0 and x != int(frame[p]):
            path.append(x)
            x = int(model.body_parentid[x])
        if x != int(frame[p]):
            continue  # frame[p] not an ancestor (should not happen for XWBC merges)
        path = path[::-1]
        fixed, bpos, hinges, q0, lims = [], [], [], [], []
        for b in path:
            fixed.append(AX @ _quat_mat(model.body_quat[b]) @ AX.T)
            bpos.append(AX @ model.body_pos[b] * 100.0)
            axes, angs, lim = [], [], []
            for jid in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                    a = AX @ model.jnt_axis[jid]
                    axes.append(a / np.linalg.norm(a))
                    angs.append(float(model.qpos0[model.jnt_qposadr[jid]]))
                    lim.append(tuple(model.jnt_range[jid]) if model.jnt_limited[jid] else (-np.inf, np.inf))
            hinges.append(axes)
            q0.append(angs)
            lims.append(lim)
        chains[j] = Chain(joint=j, parent=int(p), bodies=path, fixed=fixed, bpos=bpos, hinges=hinges, q0=q0, limits=lims)
    return chains


def _bind(model, gids, snaps, keep, frame, joint_region, chains, allow_chain=True):
    J = len(keep)
    chain_of_body = {}
    if allow_chain:
        for j, ch in chains.items():
            for b in ch.bodies:
                if b not in set(int(f) for f in frame):
                    chain_of_body[b] = j
    frame_of_body = {int(f): j for j, f in enumerate(frame)}
    rows = []
    s0 = snaps[0]
    for gid in gids:
        cands = []
        for j in range(J):
            mats = [s['r'][j].T @ s['gr'][gid] for s in snaps]
            trans = [s['r'][j].T @ (s['gp'][gid] - s['p'][j]) for s in snaps]
            er = max(float(np.max(np.abs(m - mats[0]))) for m in mats)
            ep = max(float(np.linalg.norm(t - trans[0])) for t in trans)
            if er < 1e-6 and ep < 1e-4:
                cands.append((0 if keep[j] == model.geom_bodyid[gid] else 1, float(np.linalg.norm(trans[0])), j, mats[0], trans[0]))
        b = int(model.geom_bodyid[gid])
        if cands:
            cands.sort(key=lambda t: (t[0], t[1]))
            _, _, j, mat, tr = cands[0]
            row = (0, j, -1, mat, tr)
        elif b in chain_of_body:
            j = chain_of_body[b]
            mat = s0['br'][b].T @ s0['gr'][gid]
            tr = s0['br'][b].T @ (s0['gp'][gid] - s0['bp'][b])
            row = (1, j, b, mat, tr)
        else:
            # side body: nearest ancestor that is an AM2 frame, rigid at qpos0
            x = b
            while x > 0 and x not in frame_of_body:
                x = int(model.body_parentid[x])
            j = frame_of_body.get(x, 0)
            mat = s0['r'][j].T @ s0['gr'][gid]
            tr = s0['r'][j].T @ (s0['gp'][gid] - s0['p'][j])
            row = (2, j, -1, mat, tr)
        t = int(model.geom_type[gid])
        hull = None
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[gid])
            a, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            hull = _hull_vertices(model.mesh_vert[a:a + n].astype(np.float64) * 100.0)
        rows.append((gid,) + row + (t, model.geom_size[gid] * 100.0, hull, int(joint_region[row[1]])))
    return GeomSet(gid=np.array([r[0] for r in rows], np.int64), mode=np.array([r[1] for r in rows], np.int64),
                   anchor=np.array([r[2] for r in rows], np.int64), body=np.array([r[3] for r in rows], np.int64),
                   localR=np.array([r[4] for r in rows], np.float64).reshape(-1, 3, 3),
                   localP=np.array([r[5] for r in rows], np.float64).reshape(-1, 3),
                   gtype=np.array([r[6] for r in rows], np.int64), size_cm=np.array([r[7] for r in rows], np.float64).reshape(-1, 3),
                   hull=[r[8] for r in rows], region=np.array([r[9] for r in rows], np.int64))


def _judge_geom_regions(body, model, gs, s0, root_body):
    """Region of each geom = judge J1's region of its MJCF body (regions.region_of); structural fallback per geom."""
    jr = _env.judge_module('regions')
    if jr is None:
        return gs.region, {'rule': 'render structural tree rule (judge regions unavailable)', 'fallback': len(gs)}
    out = gs.region.copy()
    fb = []
    for k, gid in enumerate(gs.gid):
        b = int(model.geom_bodyid[gid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f'body{b}'
        try:
            out[k] = int(jr.region_of(body, name, b == int(root_body), float(s0['gp'][gid][1])))
        except Exception:
            fb.append(name)
    return out, {'rule': 'judge regions.region_of (MJCF body of the geom)', 'fallback': len(fb),
                 'fallback_bodies': sorted(set(fb))}


@lru_cache(maxsize=32)
def load_robot(body: str) -> RobotModel:
    names, parents, off, reach = _env.static_spec(body)
    xml = resolve_xml(body)
    model = mujoco.MjModel.from_xml_path(xml)
    keep_all, frame_all = build_merge(model)
    rn, rp, roff = rebuilt_spec(model, keep_all)
    audit = {'xml': xml}
    rebase = None
    if body not in DATA_TREE:
        if rn != names or not np.array_equal(rp, parents):
            raise ValueError(f'{body}: MJCF merged skeleton differs from frozen static spec')
        err = float(np.max(np.abs(roff[1:] - off[1:])))
        if err > 1e-3:
            raise ValueError(f'{body}: rebuilt non-root offsets differ from static spec by {err} cm')
        keep = np.asarray(keep_all, np.int64)
        frame = np.asarray([frame_all[b] for b in keep_all], np.int64)
        mode = 'xwbc_merged_frame'
        audit['spec_offset_max_err_cm'] = err
    else:
        # DATA_TREE (H2): cache skeleton (body00..31) was inferred from stored body kinematics; node -> vendor body by
        # the judge's explicit map.  The data rotations ARE the vendor body frames (GT hinge-axis residual ~0.04 deg,
        # see hinge_axis_residual_deg), so no rest rebase; the cache bone offsets differ from this MJCF by a few cm
        # (known H2 data wall), visible as small gaps between links.
        nm = H2_NODE_MAP
        jm = _env.judge_module('pose_mjcf')
        if jm is not None and getattr(jm, 'H2_NODE_MAP', None) is not None:
            nm = jm.H2_NODE_MAP
        node = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm[j]) for j in range(len(names))], np.int64)
        if (node < 0).any():
            raise ValueError('H2 node map names missing in vendor MJCF')
        keep = node
        frame = node.copy()
        mode = 'data_tree_explicit_node_map'
        s0 = _snapshots(model, keep, frame, int(keep[0]))[0]
        fk = np.zeros((len(names), 3))
        for j, p in enumerate(parents):
            if p >= 0:
                fk[j] = fk[p] + off[j]
        audit['node_map_names'] = [nm[j] for j in range(len(names))]
        audit['node_map_source'] = 'judge/pose_mjcf.H2_NODE_MAP' if nm is not H2_NODE_MAP else 'render copy of judge H2_NODE_MAP'
        audit['node_map_rest_residual_cm'] = [float(np.linalg.norm(fk[i] - s0['p'][i])) for i in range(len(names))]
        audit['spec_offset_max_err_cm'] = float(max(audit['node_map_rest_residual_cm']))
    root = int(keep[0])
    snaps = _snapshots(model, keep, frame, root, rebase)
    fk_err = 0.0
    for s in snaps:
        fk = np.zeros_like(s['p'])
        for j, p in enumerate(parents):
            if p >= 0:
                fk[j] = fk[p] + s['r'][p] @ off[j]
        fk_err = max(fk_err, float(np.max(np.linalg.norm(fk - s['p'], axis=-1))))
    audit['random_q_fk_max_err_cm'] = fk_err
    if body not in DATA_TREE and fk_err > 0.5:
        # GR3 shows 0.05 cm: build_spec accumulates body_pos through dropped bodies ignoring their quats.
        raise ValueError(f'{body}: random-q FK mismatch {fk_err} cm')
    rest_pos = snaps[0]['p']
    tag_names = names if body not in DATA_TREE else audit['node_map_names']
    joint_region, rinfo = robot_joint_regions(parents, rest_pos, off, tag_names)
    sub = set()
    for b in range(model.nbody):
        x = b
        while x > 0 and x != root:
            x = int(model.body_parentid[x])
        if x == root:
            sub.add(b)
    robot_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub
                 and int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
    vis_ids = [g for g in robot_ids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    col_ids = [g for g in robot_ids if not (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0)]
    if not vis_ids:
        vis_ids = robot_ids
    chains = _build_chains(model, parents, keep, frame)
    vis = _bind(model, vis_ids, snaps, keep, frame, joint_region, chains)
    col = _bind(model, col_ids, snaps, keep, frame, joint_region, chains) if col_ids else None
    region_rule = {}
    for kind, gs in (('visual', vis), ('collision', col)):
        if gs is not None:
            gs.region, region_rule[kind] = _judge_geom_regions(body, model, gs, snaps[0], root)
    audit['region_rule'] = region_rule
    used = set(vis.anchor[vis.mode == 1].tolist()) | (set(col.anchor[col.mode == 1].tolist()) if col is not None else set())
    chains = {j: ch for j, ch in chains.items() if j in used}
    rm = RobotModel(body=body, xml=xml, model=model, names=list(names), parents=parents, offsets=off, reach=reach,
                    keep=keep, frame=frame, mode=mode, chains=chains, visual=vis, collision=col,
                    joint_region=joint_region, region_info=rinfo, audit=audit)
    # verify mode-1/2 geoms against MuJoCo at the random states (exactness of the chain solve / side rigidity)
    errs = {'visual': 0.0, 'collision': 0.0}
    if True:
        Rs = np.stack([s['r'] for s in snaps])
        Ps = np.stack([s['p'] for s in snaps])
        for kind, gs in (('visual', vis), ('collision', col)):
            if gs is None or not (gs.mode == 1).any():
                continue
            Gc, c = rm.geom_poses(gs, Rs, Ps)
            ref = np.stack([s['gp'][gs.gid] for s in snaps])
            m1 = gs.mode == 1
            errs[kind] = float(np.max(np.linalg.norm(c[:, m1] - ref[:, m1], axis=-1)))
    audit.update({'n_visual': len(vis), 'n_visual_chain': int((vis.mode == 1).sum()), 'n_visual_side': int((vis.mode == 2).sum()),
                  'n_collision': 0 if col is None else len(col),
                  'n_collision_chain': 0 if col is None else int((col.mode == 1).sum()),
                  'n_collision_side': 0 if col is None else int((col.mode == 2).sum()),
                  'chain_geom_random_q_max_err_cm': errs, 'mode': mode,
                  'chains': {rm.names[j]: [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ch.bodies] for j, ch in chains.items()}})
    return rm


def hinge_axis_residual_deg(rm: RobotModel, R) -> dict:
    """Independent data-convention check: residual (deg) of the best-fit hinge angles for EVERY merged chain with
    hinges (not only the ones carrying geoms): the data relative rotation R_parent^T R_joint must be reachable by the
    MJCF hinges between the two frames.  GT XWBC data should give ~0; free model predictions may not."""
    R = np.asarray(R, np.float64)
    allch = _build_chains(rm.model, rm.parents, rm.keep, rm.frame)
    errs = []
    n = 0
    for j, ch in allch.items():
        if ch.n_hinge == 0:
            continue
        Mrel = np.swapaxes(R[:, ch.parent], -1, -2) @ R[:, j]
        _, e = ch.solve(Mrel)
        errs.append(e)
        n += 1
    if not errs:
        return {'n_checked': 0, 'max_deg': float('nan'), 'median_deg': float('nan')}
    e = np.concatenate(errs)
    return {'n_checked': n, 'max_deg': float(e.max()), 'median_deg': float(np.median(e)), 'p99_deg': float(np.percentile(e, 99))}
