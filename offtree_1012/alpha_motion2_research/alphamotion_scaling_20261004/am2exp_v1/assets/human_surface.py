"""human_smpl dense surface: position-only "bone revolution" binding built from SMPL-X neutral.

Why position-only: the AM2 human (LAFAN dialect) rest pose is not standing and the twist of every human bone is
unobservable in that dialect, so human ROTATIONS are never used.  Every surface point is stored as
(bone j, t, radial, theta) in a bone frame built from joint POSITIONS only (see surface_v2.bone_frames):

    u_j = unit(P[child_j] - P[j])           (leaf j: unit(P[j] - P[parent_j]))
    v_j = normalize( sum_k w_jk perp_u(unit(ref_jk)) + 1e-3 perp_u(fallback_j) )
    w_j = u_j x v_j
    p   = P[j] + t * L_j * u_j + radial * (cos(theta) v_j + sin(theta) w_j)     (leaf: L_j -> leaf_len)

Per-bone reference rule (FRAME_RULES below, frozen; written into the manifest):
  * pelvis / lower trunk (Pelvis, Torso):   lateral of the hips  LAT_P = P[L_Hip]-P[R_Hip]   (+0.3 forward FWD_P)
  * Spine:                                  0.5 LAT_P + 0.5 LAT_C (thorax lateral) (+0.2 FWD_C)
  * Chest, Neck, Head:                      LAT_C = P[L_Thorax]-P[R_Thorax]           (+0.3 FWD_C)
  * clavicles (L/R_Thorax):                 UP_C = P[Neck]-P[Chest]                   (+0.3 FWD_C)
  * thighs, shins (Hip, Knee bones):        LAT_P                                     (+0.3 FWD_P)
      -> flexion (the dominant leg DOF) keeps the anatomical faces right: standing w = forward, sitting w = up.
  * upper arm, forearm, hand (Shoulder, Elbow, Wrist): FWD_C = LAT_C x UP_C          (+0.3 UP_C)
      -> abduction (dominant arm DOF in the frontal plane) keeps the front of the arm in front; at 90 deg
         forward flexion the UP term takes over (front of the arm faces up, as in the real body).
  * feet (Ankle, Toe): the knee-ankle-toe plane: ref = P[Knee]-P[Ankle] (+0.1 LAT_P); u = toe direction, v = up
    inside the leg's sagittal plane, w lateral.  The sole normal n_s is a FIXED combination a*u + b*v (+c*w) of
    that frame (the calibration ground normal expressed in it; the ankle->toe line is tilted ~25 deg so n_s != v);
    sole points (outward normal . n_s < -0.5) are flattened onto one plane (2nd percentile level), normal = -n_s.
  FWD_P = LAT_P x (P[Torso]-P[Pelvis]); FWD_C = LAT_C x UP_C  (left x up = forward for a right-handed Y-up frame;
  the AM2 human L/R chirality equals SMPL-X's: checked on payload rows, cos 0.99).
Hands are ellipsoids fitted to the SMPL-X hand vertices in the wrist frame (u along the forearm).

Shape: SMPL-X betas (first 16) are fitted so that the 21 SMPL-X bone lengths match the AM2 human skeleton
(orchestration/scaling_data human_smpl offsets), so posing on AM2 joint positions barely stretches anything.
SMPL-X joints 0..21 have exactly the AM2 human names/parents (Torso=spine1, Spine=spine2, Chest=spine3,
L_Toe=left_foot, L_Thorax=left_collar).  Vertex -> bone by argmax LBS weight (jaw/eyes -> Head, fingers -> Wrist).
"""
from __future__ import annotations

import numpy as np

from ._env import RID, SMPLX_NPZ

NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
         'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
         'R_Wrist']
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
MAIN_CHILD = {0: 3, 9: 12}
HUMAN_BONE_REGION = {'Pelvis': 'pelvis', 'L_Hip': 'L_thigh', 'R_Hip': 'R_thigh', 'Torso': 'trunk', 'Spine': 'trunk',
                     'Chest': 'trunk', 'L_Thorax': 'trunk', 'R_Thorax': 'trunk', 'Neck': 'head', 'Head': 'head',
                     'L_Knee': 'L_shin', 'R_Knee': 'R_shin', 'L_Ankle': 'L_foot', 'R_Ankle': 'R_foot',
                     'L_Toe': 'L_foot', 'R_Toe': 'R_foot', 'L_Shoulder': 'L_upper_arm', 'R_Shoulder': 'R_upper_arm',
                     'L_Elbow': 'L_forearm', 'R_Elbow': 'R_forearm', 'L_Wrist': 'L_hand', 'R_Wrist': 'R_hand'}
FOOT_BONES = (7, 8, 10, 11)
HAND_BONES = (20, 21)

# reference vectors: ('d', a, b) = P[a]-P[b] ; ('x', a, b, c, d) = (P[a]-P[b]) x (P[c]-P[d])
LAT_P = ('d', 1, 2, 0, 0)
FWD_P = ('x', 1, 2, 3, 0)
LAT_C = ('d', 13, 14, 0, 0)
UP_C = ('d', 12, 9, 0, 0)
FWD_C = ('x', 13, 14, 12, 9)
FRAME_RULES = {
    0: [(LAT_P, 1.0), (FWD_P, 0.3)], 3: [(LAT_P, 1.0), (FWD_P, 0.3)],
    6: [(LAT_P, 0.5), (LAT_C, 0.5), (FWD_C, 0.2)],
    9: [(LAT_C, 1.0), (FWD_C, 0.3)], 12: [(LAT_C, 1.0), (FWD_C, 0.3)], 15: [(LAT_C, 1.0), (FWD_C, 0.3)],
    13: [(UP_C, 1.0), (FWD_C, 0.3)], 14: [(UP_C, 1.0), (FWD_C, 0.3)],
    1: [(LAT_P, 1.0), (FWD_P, 0.3)], 2: [(LAT_P, 1.0), (FWD_P, 0.3)],
    4: [(LAT_P, 1.0), (FWD_P, 0.3)], 5: [(LAT_P, 1.0), (FWD_P, 0.3)],
    7: [(('d', 4, 7, 0, 0), 1.0), (LAT_P, 0.1)], 10: [(('d', 4, 7, 0, 0), 1.0), (LAT_P, 0.1)],
    8: [(('d', 5, 8, 0, 0), 1.0), (LAT_P, 0.1)], 11: [(('d', 5, 8, 0, 0), 1.0), (LAT_P, 0.1)],
    16: [(FWD_C, 1.0), (UP_C, 0.3)], 17: [(FWD_C, 1.0), (UP_C, 0.3)],
    18: [(FWD_C, 1.0), (UP_C, 0.3)], 19: [(FWD_C, 1.0), (UP_C, 0.3)],
    20: [(FWD_C, 1.0), (UP_C, 0.3)], 21: [(FWD_C, 1.0), (UP_C, 0.3)],
}
N_REF = 3


def joint_child():
    ch = np.full(22, -1, np.int64)
    for j, p in enumerate(PARENTS):
        if p >= 0 and (ch[p] < 0 or MAIN_CHILD.get(p) == j):
            ch[p] = j if p not in MAIN_CHILD else MAIN_CHILD[p]
    return ch


def frame_tables(fallback):
    """frame_ref_type [J,R] (0 diff, 1 cross, -1 unused), frame_ref_idx [J,R,4], frame_ref_w [J,R], fallback [J,3]."""
    rt = np.full((22, N_REF), -1, np.int64); ri = np.zeros((22, N_REF, 4), np.int64); rw = np.zeros((22, N_REF))
    for j, rules in FRAME_RULES.items():
        for k, (ref, w) in enumerate(rules):
            rt[j, k] = 0 if ref[0] == 'd' else 1
            ri[j, k] = ref[1:]
            rw[j, k] = w
    return rt, ri, rw, np.asarray(fallback, np.float64)


def _perp_unit(vec, u):
    vn = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-8)
    return vn - (vn * u).sum(-1, keepdims=True) * u


def bone_frames_np(P, rt, ri, rw, fb):
    """Numpy twin of surface_v2.bone_frames for one pose P [22,3] -> u, v, w [22,3], L [22]."""
    ch = joint_child(); par = np.asarray(PARENTS)
    leaf = ch < 0
    d_child = P[np.where(leaf, 0, ch)] - P
    d_leaf = P - P[np.where(par < 0, 0, par)]
    d = np.where(leaf[:, None], d_leaf, d_child)
    L = np.linalg.norm(d_child, axis=-1)
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)
    v = 1e-3 * _perp_unit(fb, u)
    for r in range(rt.shape[1]):
        a, b, c, e = ri[:, r].T
        diff = P[a] - P[b]
        cross = np.cross(P[a] - P[b], P[c] - P[e])
        vec = np.where((rt[:, r] == 1)[:, None], cross, diff)
        wgt = np.where(rt[:, r] >= 0, rw[:, r], 0.0)
        v = v + wgt[:, None] * _perp_unit(vec, u)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    w = np.cross(u, v)
    return u, v, w, L


# ---------------------------------------------------------------------------------------------------------------
def load_smplx():
    z = np.load(SMPLX_NPZ, allow_pickle=True)
    return dict(v_template=z['v_template'].astype(np.float64), shapedirs=z['shapedirs'][:, :, :300].astype(np.float64),
                J_regressor=z['J_regressor'].astype(np.float64), weights=z['weights'].astype(np.float64),
                faces=z['f'].astype(np.int64), kintree=z['kintree_table'].astype(np.int64))


def fit_betas(sm, target_len_cm, n_betas=16, reg=1e-3):
    """Least-squares SMPL-X betas so that bone lengths (cm) of joints 1..21 match target_len_cm [22]."""
    from scipy.optimize import least_squares
    Jr = sm['J_regressor'][:22]
    JT = Jr @ sm['v_template']                                      # [22,3] m
    JS = np.einsum('jv,vcb->jcb', Jr, sm['shapedirs'][:, :, :n_betas])  # [22,3,B]
    par = np.asarray(PARENTS)

    def lengths(b):
        J = JT + JS @ b
        return np.linalg.norm(J[1:] - J[par[1:]], axis=-1) * 100.0

    def resid(b):
        return np.concatenate([lengths(b) - target_len_cm[1:], np.sqrt(reg) * b])
    sol = least_squares(resid, np.zeros(n_betas), method='lm')
    b = sol.x
    return b, lengths(np.zeros(n_betas)), lengths(b)


def smplx_tpose(sm, betas):
    nb = len(betas)
    V = sm['v_template'] + sm['shapedirs'][:, :, :nb] @ betas
    J = sm['J_regressor'] @ V
    return V * 100.0, J * 100.0   # cm


def vertex_bones(sm):
    w = sm['weights']
    b = np.argmax(w, axis=1)
    b = np.where((b >= 22) & (b <= 24), 15, b)        # jaw, eyes -> Head
    b = np.where((b >= 25) & (b <= 39), 20, b)        # left hand fingers -> L_Wrist
    b = np.where(b >= 40, 21, b)                      # right hand fingers -> R_Wrist
    return b


def _icosphere(subdiv=3):
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    return np.asarray(s.vertices, np.float64), np.asarray(s.faces, np.int64)


def build_body_mesh(V, Jp, vb, faces, frames):
    """SMPL-X mesh with the two hands replaced by fitted ellipsoids.  Returns (V2, F2, face_bone, ellipsoids)."""
    u, v, w, L = frames
    fb = vb[faces]
    face_bone = np.array([np.bincount(r, minlength=22).argmax() for r in fb])
    keep = ~np.isin(face_bone, HAND_BONES)
    Vs, Fs, FB = [V], [faces[keep]], [face_bone[keep]]
    off = len(V)
    ell = {}
    sv, sf = _icosphere(3)
    for hb in HAND_BONES:
        sel = vb == hb
        X = V[sel] - Jp[hb]
        B = np.stack([u[hb], v[hb], w[hb]], 1)          # columns
        c = X @ B                                       # coords in (u,v,w)
        lo, hi = np.percentile(c, 2, axis=0), np.percentile(c, 98, axis=0)
        centre = 0.5 * (lo + hi); semi = 0.5 * (hi - lo)
        # keep the ellipsoid attached to the wrist: its proximal end at the wrist joint at most
        pts = (sv * semi + centre) @ B.T + Jp[hb]
        Vs.append(pts); Fs.append(sf + off); FB.append(np.full(len(sf), hb)); off += len(pts)
        ell[NAMES[hb]] = {'centre_uvw_cm': centre.tolist(), 'semi_axes_uvw_cm': semi.tolist(),
                          'n_hand_vertices_replaced': int(sel.sum())}
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(FB), ell


def encode_points(points, normals, bones, Jp, frames, leaf_len):
    """(t, radial, theta, normal_uvw) of each point in its bone frame at the calibration joint positions Jp."""
    u, v, w, L = frames
    ch = joint_child()
    d = points - Jp[bones]
    ax = (d * u[bones]).sum(-1)
    Lb = np.where(ch[bones] < 0, leaf_len[bones], L[bones])
    t = ax / Lb
    r = d - ax[:, None] * u[bones]
    radial = np.linalg.norm(r, axis=-1)
    theta = np.arctan2((r * w[bones]).sum(-1), (r * v[bones]).sum(-1))
    nuvw = np.stack([(normals * u[bones]).sum(-1), (normals * v[bones]).sum(-1), (normals * w[bones]).sum(-1)], -1)
    return t, radial, theta, nuvw


def decode_points(t, radial, theta, nuvw, bones, Jp, frames, leaf_len):
    u, v, w, L = frames
    ch = joint_child()
    Lb = np.where(ch[bones] < 0, leaf_len[bones], L[bones])
    rd = np.cos(theta)[:, None] * v[bones] + np.sin(theta)[:, None] * w[bones]
    p = Jp[bones] + (t * Lb)[:, None] * u[bones] + radial[:, None] * rd
    n = nuvw[:, 0:1] * u[bones] + nuvw[:, 1:2] * v[bones] + nuvw[:, 2:3] * w[bones]
    return p, n


def flatten_soles(points, normals, bones, Jp, frames, up=(0.0, 1.0, 0.0)):
    """Flatten the sole: foot points whose outward normal faces the ground are moved onto one plane.

    The plane normal is the calibration ground normal (+Y, SMPL-X standing T-pose) expressed in the foot frame
    (u = ankle->toe, v = up inside the knee-ankle-toe plane, w = lateral), i.e. a FIXED direction n_s =
    a*u + b*v (+ c*w) of the knee-ankle-toe frame; the stored (t, radial, theta) of the flattened points therefore
    keep the sole flat and moving with the knee-ankle-toe plane in every pose.  Level = 2nd percentile of the foot
    points along n_s (relative to the ankle).  Selection: n . n_s < -0.5."""
    u, v, w, L = frames
    up = np.asarray(up, np.float64)
    info = {}
    points = points.copy(); normals = normals.copy()
    for side, bset in (('L', (7, 10)), ('R', (8, 11))):
        a = bset[0]
        coef = np.array([up @ u[a], up @ v[a], up @ w[a]])
        ns = coef[0] * u[a] + coef[1] * v[a] + coef[2] * w[a]
        ns /= np.linalg.norm(ns)
        foot = np.isin(bones, bset)
        h = (points[foot] - Jp[a]) @ ns
        level = float(np.percentile(h, 2))
        down = foot & ((normals * ns).sum(-1) < -0.5)
        hd = (points[down] - Jp[a]) @ ns
        points[down] = points[down] + (level - hd)[:, None] * ns
        normals[down] = -ns
        info[side] = {'sole_normal_in_foot_frame_uvw': coef.tolist(), 'sole_level_below_ankle_cm': -level,
                      'n_flattened': int(down.sum()),
                      'max_shift_cm': float(np.abs(level - hd).max()) if down.any() else 0.0,
                      'ankle_toe_frame_agreement': float(v[bset[0]] @ v[bset[1]])}
    return points, normals, info


# ---------------------------------------------------------------------------------------------------------------
def lbs_pose(sm, betas, rotvecs):
    """Plain LBS (no pose blendshapes) of the shaped SMPL-X for per-joint local axis-angle rotvecs {joint: [3]}.
    Returns posed vertices (cm), posed joints (cm, 55)."""
    from scipy.spatial.transform import Rotation as Rot
    V0, J0 = smplx_tpose(sm, betas)
    par = sm['kintree'][0].copy(); par[0] = -1
    nJ = len(par)
    G = np.tile(np.eye(4), (nJ, 1, 1))
    for j in range(nJ):
        Rl = Rot.from_rotvec(rotvecs.get(j, np.zeros(3))).as_matrix()
        T = np.eye(4); T[:3, :3] = Rl
        T[:3, 3] = J0[j] - (J0[par[j]] if par[j] >= 0 else 0)
        G[j] = T if par[j] < 0 else G[par[j]] @ T
    Jp = G[:, :3, 3].copy()
    A = G.copy()
    A[:, :3, 3] = G[:, :3, 3] - np.einsum('jab,jb->ja', G[:, :3, :3], J0)
    W = sm['weights']
    M = np.einsum('vj,jab->vab', W, A)
    Vp = np.einsum('vab,vb->va', M[:, :3, :3], V0) + M[:, :3, 3]
    return Vp, Jp


def point_to_mesh(points, V, F):
    import trimesh
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def region_ids(bones):
    return np.array([RID[HUMAN_BONE_REGION[NAMES[b]]] for b in bones], np.int64)
