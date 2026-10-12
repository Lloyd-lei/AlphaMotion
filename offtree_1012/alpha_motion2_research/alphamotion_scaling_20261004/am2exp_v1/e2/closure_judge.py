"""E2 closure judge J2 (independent of every training loss).

Never imports e2.xbody_loss, e2.opposition or any model/loss code.  Geometry = MuJoCo COLLISION
capsules of the Adroit hand (C_*distal) and the dataset object's collision geom, placed in a COPY of
the pen MJCF built with MjSpec (a mocap body carries the object; data/models is never edited).

Hand grasp label from dataset hand+object poses (never enters training):
  g = 1  thumb distal capsule AND at least one of ff/mf/rf distal capsules each < 5 mm from the object,
         on opposite sides of the object axis (radial vectors of their closest points: r_th . r_f < 0)
  g = 0  every distal capsule (th, ff, mf, rf, lf) > 20 mm from the object
  g = -1 otherwise (excluded)
Predicted gripper state from the predicted Franka pad clearance c_F (mm), object diameter d (mm):
  hold  if d-3 <= c_F <= d+8 ; open if c_F > d+8 ; crush if c_F < d-3
BA = 0.5 * (P(hold | g=1) + P(open | g=0)).
F->A hand-closure test (decoded hand, Franka source clearance c_F = cylinder diameter): the grasp-label rule
turned around.  Pass iff SOME placement of a cylinder of diameter c_F (axis = palm lateral, 18-axis grid) has the
thumb distal capsule and one of the ff/mf/rf distal capsules within 5 mm of its surface on opposite sides of
its axis while no hand collision geom penetrates it by more than 5 mm (placement search, see closure_at_frame).
(Rev 2: the first version fixed the cylinder at the Tch-site midpoint; on real hammer grasps that passed only
~2% of frames at the true handle size, so it could not validate anything.)
"""
from __future__ import annotations
import numpy as np

from ._env import PEN_XML

AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
DIGITS = ['th', 'ff', 'mf', 'rf', 'lf']
AM2_DISTAL = {'th': 24, 'ff': 6, 'mf': 10, 'rf': 14, 'lf': 19}
HOLD_LOW_MM, HOLD_HIGH_MM = 3.0, 8.0
GRASP_NEAR_MM, FREE_FAR_MM = 5.0, 20.0
CLOSURE_TOL_MM = 5.0          # = GRASP_NEAR_MM: same contact tolerance as the grasp labels (rev 2)
HOLD, OPEN, CRUSH = 1, 0, -1


def _quat_from_mat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, np.float64).reshape(9))
    return q


def _frame_with_z(axis):
    z = np.asarray(axis, np.float64)
    z = z / np.linalg.norm(z)
    ref = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


class JudgeScene:
    """Pen-hand MJCF copy (MjSpec) + mocap objects: handle capsule, ball, pen cylinder, closure cylinder."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        spec = mujoco.MjSpec.from_file(str(PEN_XML))
        body = spec.worldbody.add_body()
        body.name = 'e2_object'
        body.mocap = True
        body.pos = [0, 0, -5]
        shapes = {'capsule': (mujoco.mjtGeom.mjGEOM_CAPSULE, [0.025, 0.05, 0]), 'sphere': (mujoco.mjtGeom.mjGEOM_SPHERE, [0.035, 0, 0]),
                  'cylinder': (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.015, 0.065, 0])}
        for name, (gt, size) in shapes.items():
            g = body.add_geom()
            g.name = 'e2_' + name
            g.type = gt
            g.size = size
            g.contype = 0
            g.conaffinity = 0
        cb = spec.worldbody.add_body()
        cb.name = 'e2_closure'
        cb.mocap = True
        cb.pos = [0, 0, -6]
        g = cb.add_geom()
        g.name = 'e2_closure_cyl'
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [0.025, 0.06, 0]
        g.contype = 0
        g.conaffinity = 0
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        self.obj_body = nid(mujoco.mjtObj.mjOBJ_BODY, 'e2_object')
        self.obj_mocap = int(m.body_mocapid[self.obj_body])
        self.clo_body = nid(mujoco.mjtObj.mjOBJ_BODY, 'e2_closure')
        self.clo_mocap = int(m.body_mocapid[self.clo_body])
        self.obj_geoms = {k: nid(mujoco.mjtObj.mjOBJ_GEOM, 'e2_' + k) for k in shapes}
        self.clo_geom = nid(mujoco.mjtObj.mjOBJ_GEOM, 'e2_closure_cyl')
        self.caps = {k: nid(mujoco.mjtObj.mjOBJ_GEOM, f'C_{k}distal') for k in DIGITS}
        self.sites = {k: nid(mujoco.mjtObj.mjOBJ_SITE, f'Tch_{k}tip') for k in DIGITS}
        self.fa = nid(mujoco.mjtObj.mjOBJ_BODY, 'forearm')
        self.palm = nid(mujoco.mjtObj.mjOBJ_BODY, 'palm')
        self.d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, self.d)
        self.P_fa = self.d.xpos[self.fa].copy()
        # rest frames of the 25 AM2 bodies (pen adapter order: forearm subtree in body-id order)
        root = self.fa
        self.body_ids = [b for b in range(root, m.nbody) if self._under(b, root)]
        self.rest_R = self.d.xmat[self.body_ids].reshape(-1, 3, 3).copy()
        self.body_index = {b: i for i, b in enumerate(self.body_ids)}

    def _under(self, b, root):
        m = self.m
        if b in (self.obj_body, getattr(self, 'clo_body', -1)):
            return False
        p = b
        while p > 0 and p != root:
            p = int(m.body_parentid[p])
        return p == root

    # -- placement -----------------------------------------------------------------------------
    def can_to_world(self, p_cm):
        return AX.T @ (np.asarray(p_cm, np.float64) / 100.0) + self.P_fa

    def set_object(self, obj_type, centre_cm, geom_R_can=None, axis_can=None):
        R_can = geom_R_can if geom_R_can is not None else (_frame_with_z(axis_can) if axis_can is not None else np.eye(3))
        Rw = AX.T @ np.asarray(R_can, np.float64)
        self.d.mocap_pos[self.obj_mocap] = self.can_to_world(centre_cm)
        self.d.mocap_quat[self.obj_mocap] = _quat_from_mat(Rw)
        return self.obj_geoms[obj_type]

    def set_object_size(self, obj_type, radius_cm, half_cm):
        g = self.obj_geoms[obj_type]
        self.m.geom_size[g][0] = radius_cm / 100.0
        if obj_type != 'sphere':
            self.m.geom_size[g][1] = half_cm / 100.0
        self.m.geom_rbound[g] = (radius_cm + half_cm) / 100.0

    def set_hand_qpos(self, q24):
        self.d.qpos[:24] = q24

    def kinematics(self):
        self.mj.mj_kinematics(self.m, self.d)

    def distance(self, g1, g2, distmax=0.1):
        ft = np.zeros(6)
        dist = self.mj.mj_geomDistance(self.m, self.d, g1, g2, distmax, ft)
        return dist, ft[:3].copy(), ft[3:].copy()

    def repose_hand(self, P_cm, G):
        """Rigidly re-pose hand bodies' collision geoms and sites from AM2 rotations/positions (one frame).

        R_body = AX^T G AX R_rest ; p_body = AX^T P/100 + P_forearm.  Writes geom_xpos/xmat and site_xpos.
        """
        m, d = self.m, self.d
        Rw = np.einsum('ij,njk,kl,nlm->nim', AX.T, np.asarray(G, np.float64), AX, self.rest_R)
        Pw = np.asarray(P_cm, np.float64) / 100.0 @ AX + self.P_fa
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            i = self.body_index.get(b)
            if i is None:
                continue
            gR = np.zeros(9)
            self.mj.mju_quat2Mat(gR, m.geom_quat[g])
            d.geom_xmat[g] = (Rw[i] @ gR.reshape(3, 3)).reshape(9)
            d.geom_xpos[g] = Pw[i] + Rw[i] @ m.geom_pos[g]
        for s in range(m.nsite):
            b = int(m.site_bodyid[s])
            i = self.body_index.get(b)
            if i is None:
                continue
            d.site_xpos[s] = Pw[i] + Rw[i] @ m.site_pos[s]
        self._palm_R = Rw[self.body_index[self.palm]]


# ---------------------------------------------------------------------------------------------
# Grasp labels from dataset hand + object
# ---------------------------------------------------------------------------------------------
def _radial(x, obj_type, centre_w, axis_w, half_m):
    if obj_type == 'sphere':
        return x - centre_w
    t = np.clip(np.dot(x - centre_w, axis_w), -half_m, half_m)
    return x - (centre_w + t * axis_w)


def grasp_labels(data, scene=None, near_mm=GRASP_NEAR_MM, far_mm=FREE_FAR_MM):
    """data: dict from hand_data.load_hand()/load_pen(). Returns labels [N] in {1,0,-1} and distances [N,5] mm."""
    scene = scene or JudgeScene()
    obj_type = str(data['obj_type'])
    radius = float(data['obj_radius_cm'])
    half = float(data['obj_half_length_cm'])
    scene.set_object_size(obj_type, radius, half)
    N = len(data['hand_qpos'])
    dist = np.zeros((N, 5))
    opposite = np.zeros((N, 3), bool)   # th vs ff, mf, rf
    if 'obj_geom_R_can' in data:
        Rs = data['obj_geom_R_can']
    else:
        Rs = [_frame_with_z(a) for a in _pen_axes(data)]
    for i in range(N):
        scene.set_hand_qpos(data['hand_qpos'][i])
        g = scene.set_object(obj_type, data['obj_centre_cm'][i], geom_R_can=Rs[i])
        scene.kinematics()
        centre_w = scene.d.geom_xpos[g].copy()
        axis_w = scene.d.geom_xmat[g].reshape(3, 3)[:, 2].copy()
        r = {}
        for k_i, k in enumerate(DIGITS):
            dd, p_hand, _ = scene.distance(scene.caps[k], g, distmax=0.1)
            dist[i, k_i] = dd * 1000
            r[k] = _radial(p_hand, obj_type, centre_w, axis_w, half / 100.0)
        for j, k in enumerate(['ff', 'mf', 'rf']):
            opposite[i, j] = float(np.dot(r['th'], r[k])) < 0
    near = dist < near_mm
    g1 = near[:, 0] & np.any(near[:, 1:4] & opposite, axis=1)
    g0 = np.all(dist > far_mm, axis=1)
    labels = np.full(N, -1, np.int8)
    labels[g1] = 1
    labels[g0 & ~g1] = 0
    return labels, dist, opposite


def _pen_axes(data):
    seg = np.asarray(data['obj_segment_cm'], np.float64)
    a = seg[:, 1] - seg[:, 0]
    return a / np.linalg.norm(a, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------
# Gripper state + balanced accuracy
# ---------------------------------------------------------------------------------------------
def gripper_state(c_F_mm, d_mm):
    c = np.asarray(c_F_mm, np.float64)
    s = np.full(c.shape, HOLD, np.int8)
    s[c > d_mm + HOLD_HIGH_MM] = OPEN
    s[c < d_mm - HOLD_LOW_MM] = CRUSH
    return s


def balanced_accuracy(state, label):
    state = np.asarray(state)
    label = np.asarray(label)
    pos = label == 1
    neg = label == 0
    tpr = float((state[pos] == HOLD).mean()) if pos.any() else float('nan')
    tnr = float((state[neg] == OPEN).mean()) if neg.any() else float('nan')
    return {'BA': 0.5 * (tpr + tnr) if pos.any() and neg.any() else float('nan'), 'P_hold_given_grasp': tpr, 'P_open_given_free': tnr,
            'crush_rate_given_grasp': float((state[pos] == CRUSH).mean()) if pos.any() else float('nan'),
            'crush_rate_given_free': float((state[neg] == CRUSH).mean()) if neg.any() else float('nan'),
            'n_grasp': int(pos.sum()), 'n_free': int(neg.sum())}


def episode_bootstrap(state, label, episode, n_boot=1000, seed=0):
    """Percentile CI of BA by resampling episodes (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    idx = {e: np.flatnonzero(episode == e) for e in eps}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(eps, size=len(eps), replace=True)
        rows = np.concatenate([idx[e] for e in pick])
        v = balanced_accuracy(state[rows], label[rows])['BA']
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {'lo': float('nan'), 'hi': float('nan'), 'n': 0}
    return {'lo': float(np.percentile(vals, 2.5)), 'hi': float(np.percentile(vals, 97.5)), 'n': len(vals)}


# ---------------------------------------------------------------------------------------------
# F->A hand-closure test on decoded hands (placement search over MJCF collision geoms)
# ---------------------------------------------------------------------------------------------
THUMB_GEOM = 'C_thdistal'
FINGER_GEOMS = ['C_ffdistal', 'C_mfdistal', 'C_rfdistal']
# cylinder axis candidates: palm lateral (palm x) rotated about palm y by YAW then about palm z by TILT (deg).
# Range set on hammer DEV grasp frames (true handle axis in the palm frame: ~+10 deg about y, ~-5 deg about z).
CLOSURE_AXES_YAW_DEG = (-20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
CLOSURE_AXES_TILT_DEG = (-10.0, 0.0, 10.0)
_EXCLUDE_FROM_PENETRATION = ('C_forearm1', 'C_wrist')


def _sample_geom_local(m, g, step_m=0.003, n_ang=16):
    """Surface points of a capsule/box collision geom in its own frame (metres)."""
    import mujoco
    t = int(m.geom_type[g])
    size = m.geom_size[g]
    pts = []
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = float(size[0]), float(size[1])
        ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        for z in np.linspace(-h, h, max(2, int(np.ceil(2 * h / step_m)) + 1)):
            pts.append(np.c_[ring, np.full(n_ang, z)])
        for sgn in (-1.0, 1.0):
            for el in np.linspace(0, np.pi / 2, 5)[1:]:
                k = max(4, int(n_ang * np.cos(el)))
                a = np.linspace(0, 2 * np.pi, k, endpoint=False)
                pts.append(np.c_[r * np.cos(el) * np.cos(a), r * np.cos(el) * np.sin(a), np.full(k, sgn * (h + r * np.sin(el)))])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        step_m = max(step_m, 0.005)
        hx = np.asarray(size[:3], np.float64)
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            u = np.linspace(-hx[o[0]], hx[o[0]], max(2, int(np.ceil(2 * hx[o[0]] / step_m)) + 1))
            v = np.linspace(-hx[o[1]], hx[o[1]], max(2, int(np.ceil(2 * hx[o[1]] / step_m)) + 1))
            U, V = np.meshgrid(u, v, indexing='ij')
            for sgn in (-1.0, 1.0):
                q = np.zeros((U.size, 3))
                q[:, o[0]], q[:, o[1]], q[:, ax] = U.ravel(), V.ravel(), sgn * hx[ax]
                pts.append(q)
    else:
        return np.zeros((0, 3))
    return np.concatenate(pts)


def _closure_geometry(scene):
    cache = getattr(scene, '_closure_geom_cache', None)
    if cache is not None:
        return cache
    import mujoco
    m = scene.m
    names, locs = [], []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in scene.body_index or int(m.geom_group[g]) != 4:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if not nm.startswith('C_') or nm in _EXCLUDE_FROM_PENETRATION:
            continue
        loc = _sample_geom_local(m, g)
        if len(loc):
            names.append((g, nm))
            locs.append(loc)
    n = np.array([len(l) for l in locs])
    starts = np.r_[0, np.cumsum(n)[:-1]]
    cache = {'geoms': names, 'local': locs, 'gid': np.concatenate([np.full(len(l), i) for i, l in enumerate(locs)]),
             'starts': starts, 'slices': [(int(s0), int(s0 + k)) for s0, k in zip(starts, n)]}
    scene._closure_geom_cache = cache
    return cache


def _hand_points_world(scene, geo):
    d = scene.d
    out = []
    for (g, _), loc in zip(geo['geoms'], geo['local']):
        R = d.geom_xmat[g].reshape(3, 3)
        out.append(loc @ R.T + d.geom_xpos[g])
    return np.concatenate(out)


def _rot_about(axis, deg):
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    th = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def closure_axes(palm_R):
    out = []
    for yaw in CLOSURE_AXES_YAW_DEG:
        for tilt in CLOSURE_AXES_TILT_DEG:
            out.append(_rot_about(palm_R[:, 2], tilt) @ _rot_about(palm_R[:, 1], yaw) @ palm_R[:, 0])
    return out


def closure_at_frame(scene, D_m, tol_m=CLOSURE_TOL_MM / 1000, half_len_m=0.05, n_anchor=12, n_dir=24, pen_tol_m=None, axes=None):
    """Can a cylinder of diameter D_m be held by the CURRENTLY POSED hand (scene.repose_hand called)?

    Hold = the judge's own grasp-label rule turned around: some placement of the cylinder (axis = palm
    lateral rotated by the closure_axes grid) has the thumb distal capsule AND one of the
    ff/mf/rf distal capsules within +-tol of its surface, on opposite sides of its axis, while NO hand
    collision geom (palm, all phalanges; forearm/wrist excluded) penetrates it by more than pen_tol (= tol).
    Candidates: cylinder tangent to a sampled surface point of the thumb or ff/mf/rf distal capsule
    (n_anchor points each x n_dir directions in the cross-section plane) for every axis of closure_axes();
    cylinder length 2*half_len_m centred on the anchor.  Pruned in three stages (thumb touch -> opposite finger
    touch -> penetration).  Geometry = sampled collision-geom surfaces (capsules: 3 mm rings x 16; boxes: 5 mm
    grid).  Returns (pass, smallest violation found in m (diagnostic), finger index 0 ff/1 mf/2 rf or -1).
    """
    pen_tol_m = tol_m if pen_tol_m is None else pen_tol_m
    geo = _closure_geometry(scene)
    X = _hand_points_world(scene, geo)
    sl = geo['slices']
    names = [n for _, n in geo['geoms']]
    th = names.index(THUMB_GEOM)
    fing = [names.index(n) for n in FINGER_GEOMS]
    r = 0.5 * float(D_m)
    palm_R = scene._palm_R
    best_v, best_f = np.inf, -1
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    circ = np.stack([np.cos(ang), np.sin(ang)], 1)
    T = []
    for gi in [th] + fing:                       # tangent candidates anchored on the thumb AND on each finger capsule
        Ti = np.arange(*sl[gi])
        T.append(Ti[np.linspace(0, len(Ti) - 1, min(n_anchor, len(Ti))).astype(int)])
    T = np.concatenate(T)
    if axes is None:
        axes = closure_axes(palm_R)
    for a in axes:
        a = np.asarray(a, np.float64) / np.linalg.norm(a)
        e1 = _frame_with_z(a)[:, 0]
        e2 = np.cross(a, e1)
        Bm = np.stack([e1, e2], 1)
        X2 = (X @ Bm).astype(np.float32)                                          # [P,2] cross-section coords
        xa = (X @ a).astype(np.float32)
        C2 = (X2[T][:, None] + np.float32(r) * circ[None].astype(np.float32)).reshape(-1, 2)   # candidate centres [Nc,2]
        ca = np.repeat(xa[T], n_dir)

        def gaps(idx_slice, cand):
            V = X2[idx_slice][None] - C2[cand][:, None]
            g = np.sqrt((V ** 2).sum(-1)) - r
            g[np.abs(xa[idx_slice][None] - ca[cand][:, None]) > half_len_m] = np.inf
            return g, V

        cand = np.arange(len(C2))
        # stage 1: thumb distal capsule touches (|gap| <= tol)
        g, V = gaps(slice(*sl[th]), cand)
        jt = g.argmin(1)
        g_th = g[np.arange(len(cand)), jt]
        v1 = np.maximum(np.abs(g_th) - tol_m, 0)
        keep = v1 <= 0
        if not keep.any():
            best_v = min(best_v, float(v1.min()))
            continue
        cand, vt = cand[keep], V[np.arange(len(keep)), jt][keep]
        for fi, f in enumerate(fing):
            # stage 2: finger capsule touches on the opposite side
            g, V = gaps(slice(*sl[f]), cand)
            j = g.argmin(1)
            g_f = g[np.arange(len(cand)), j]
            vf = V[np.arange(len(cand)), j]
            opp = (vt * vf).sum(1) < 0
            v2 = np.maximum(np.abs(g_f) - tol_m, 0) + np.where(opp, 0.0, 1.0)
            ok = v2 <= 0
            if not ok.any():
                if v2.min() < best_v:
                    best_v, best_f = float(v2.min()), fi
                continue
            # stage 3: nothing penetrates by more than pen_tol
            c3 = cand[ok]
            g, _ = gaps(slice(None), c3)
            v3 = np.maximum(-g.min(1) - pen_tol_m, 0)
            if (v3 <= 0).any():
                return True, 0.0, fi
            if v3.min() < best_v:
                best_v, best_f = float(v3.min()), fi
    return False, best_v, best_f


def hand_closure_test(P_cm, G, c_F_cm, scene=None, tol_mm=CLOSURE_TOL_MM, half_len_cm=5.0):
    """P_cm [N,25,3], G [N,25,3,3] decoded Adroit pose; c_F_cm [N] Franka source clearance (cylinder diameter).

    Returns dict: pass [N] bool (see closure_at_frame), violation_mm [N] (0 when passed; else the smallest
    constraint violation found), finger [N] (1 ff, 2 mf, 3 rf, 0 none), tch_gap_mm [N] (thumb-to-nearest ff/mf
    Tch site distance, a descriptive aperture only).
    """
    scene = scene or JudgeScene()
    N = len(P_cm)
    passed = np.zeros(N, bool)
    viol = np.zeros(N)
    which = np.zeros(N, np.int8)
    tch_gap = np.zeros(N)
    for i in range(N):
        scene.repose_hand(P_cm[i], G[i])
        s = {k: scene.d.site_xpos[scene.sites[k]].copy() for k in DIGITS}
        tch_gap[i] = min(np.linalg.norm(s[k] - s['th']) for k in ('ff', 'mf')) * 1000
        D = max(float(c_F_cm[i]), 0.0) / 100.0
        ok, v, fi = closure_at_frame(scene, D, tol_mm / 1000, half_len_cm / 100.0)
        passed[i] = ok
        viol[i] = v * 1000
        which[i] = fi + 1 if fi >= 0 else 0
    return {'pass': passed, 'violation_mm': viol, 'finger': which, 'tch_gap_mm': tch_gap}


FIT_MARGIN_CM = 3.0


def hand_fit_test(P_cm, G, c_F_cm, scene=None, margin_cm=FIT_MARGIN_CM, tol_mm=CLOSURE_TOL_MM):
    """Two-sided F->A test: the decoded hand holds a cylinder of diameter c_F but NOT one margin_cm larger.

    One-sided 'hold' cannot tell an open hand from a correct one (an open hand also pinches small objects
    somewhere), so the F->A score uses fit = hold(c_F) & ~hold(c_F + 3 cm).  On real hammer grasps (50 mm)
    hold(50) ~ 0.97 and hold(80) ~ 0.06.
    """
    scene = scene or JudgeScene()
    c = np.asarray(c_F_cm, np.float64)
    a = hand_closure_test(P_cm, G, c, scene, tol_mm)
    b = hand_closure_test(P_cm, G, c + margin_cm, scene, tol_mm)
    return {'hold': a['pass'], 'hold_plus_margin': b['pass'], 'fit': a['pass'] & ~b['pass'], 'tch_gap_mm': a['tch_gap_mm'],
            'violation_mm': a['violation_mm'], 'margin_cm': float(margin_cm)}
