"""Shared helpers for the HiPHI truncation (prefix-coverage) experiment.

Feature = SMPL-X FK body joints (fitted_joints == smplx FK, verified 4.8e-7 m),
root-relative, heading-canonicalised (yaw from hip+shoulder axis), at 10 Hz.
Root height = pelvis y - floor_y_m (qa.json).
"""
import csv, glob, json, os
import numpy as np

ROOT = '/home/arenalabs/umr_factory_data'
OUT = ROOT + '/experiments/truncation'
HZ = 10  # analysis rate

THROW = {'cast', 'chuck', 'pitch', 'toss', 'throw', 'hurl', 'fling', 'lob', 'sling', 'heave', 'bowl'}
PUSHPULL = {'push', 'pull', 'move', 'shove', 'force', 'thrust', 'propel', 'drag', 'slide', 'nudge',
            'tug', 'yank', 'jerk', 'press', 'roll', 'wheel', 'haul', 'tow', 'draw'}
LIFTPLACE_CM = {'lift', 'drop', 'raise', 'hoist', 'lower'}

GROUP_ZH = {
    'carry': '走路搬运', 'push_pull': '推拉挪', 'lift_place': '抬放拿取', 'sit': '坐椅',
    'ground': '地面', 'kick_hit': '高动态踢打', 'upper_manip': '上肢操作物体', 'throw': '抛掷',
    'step': '迈步/踩凳', 'whole_other': '全身物体-其他', 'locomotion': '纯移动步态',
    'other_other': '其他身体动作', 'other_gesture_social': '手势社交', 'other_idle_posture': '站姿拉伸',
}


def group_of(r):
    p, fr, lu = r['primary'], r['frame'], r['lu']
    if p == 'ground_contact':
        return 'ground'
    if p == 'sitting_on_object':
        return 'sit'
    if p == 'dynamic':
        return 'kick_hit'
    if p == 'locomotion':
        return 'locomotion'
    if fr == 'Self_motion' and lu == 'step':
        return 'step'
    if lu in THROW:
        return 'throw'
    if p == 'whole_body_object':
        if fr == 'Bringing':
            return 'carry'
        if lu in PUSHPULL and fr in ('Cause_motion', 'Manipulation'):
            return 'push_pull'
        if fr in ('Placing', 'Taking') or (fr == 'Cause_motion' and lu in LIFTPLACE_CM):
            return 'lift_place'
        return 'whole_other'
    if p == 'upper_body_manipulation':
        return 'upper_manip'
    return 'other_' + p


def load_labels():
    return {r['motion_id']: r for r in csv.DictReader(open(ROOT + '/diversity/hiphi/hiphi_clips_labeled.csv'))}


def clip_index():
    """motion_id -> dict(path_npz, step (frames to skip to reach 10 Hz), dir)."""
    idx = {}
    for p in glob.glob(ROOT + '/smplx_gpu/hiphi_30fps/data/*/*/*/motion_actor_smplx.npz'):
        mid = p.split('/')[-2]
        idx[mid] = dict(npz=p, step=3, dir=os.path.dirname(p), src='gpu30')
    for p in glob.glob(ROOT + '/smplx/hiphi/data/*/*/*/motion_actor_smplx.npz'):
        mid = p.split('/')[-2]
        if mid not in idx:
            idx[mid] = dict(npz=p, step=9, dir=os.path.dirname(p), src='cpu90')
    return idx


def floor_y(d):
    try:
        return json.load(open(d + '/qa.json'))['floor']['floor_y_m']
    except Exception:
        return None


def canon_features(J):
    """J: (T,22,3) world, y-up. Returns (T,22,3) root-relative heading-canonical, root height raw y."""
    pel = J[:, 0]
    across = (J[:, 1] - J[:, 2]) + (J[:, 16] - J[:, 17])  # L - R (hips + shoulders)
    ang = np.arctan2(across[:, 2], across[:, 0])  # angle of across in xz plane
    c, s = np.cos(-ang), np.sin(-ang)
    X = J - pel[:, None]
    x, z = X[..., 0], X[..., 2]
    # rotate about y so that 'across' maps to +x
    xr = c[:, None] * x - s[:, None] * z
    zr = s[:, None] * x + c[:, None] * z
    F = np.stack([xr, X[..., 1], zr], -1)
    return F.astype(np.float32), pel[:, 1].astype(np.float32)


def dist_mats(FA, hA, FB, hB, chunk=200):
    """D[i,j] = mean_joint ||FA_i - FB_j||, H[i,j] = |hA_i - hB_j|."""
    TA, TB = len(FA), len(FB)
    D = np.empty((TA, TB), np.float32)
    for i in range(0, TA, chunk):
        a = FA[i:i + chunk, None]  # (c,1,22,3)
        D[i:i + chunk] = np.sqrt(((a - FB[None]) ** 2).sum(-1)).mean(-1)
    H = np.abs(hA[:, None] - hB[None, :]).astype(np.float32)
    return D, H


def window_mats(D, H, L):
    """windowed (length L frames, aligned diagonals) mean of D and H. Index = window start."""
    TA, TB = D.shape
    na, nb = TA - L + 1, TB - L + 1
    Dw = np.zeros((na, nb), np.float32)
    Hw = np.zeros((na, nb), np.float32)
    for k in range(L):
        Dw += D[k:k + na, k:k + nb]
        Hw += H[k:k + na, k:k + nb]
    return Dw / L, Hw / L


def earliest_match(M):
    """M bool (T,T'): index of first True per row, or large if none."""
    any_ = M.any(1)
    em = np.argmax(M, 1)
    em[~any_] = 10 ** 7
    return em


def load_clip(info, lab=None):
    d = np.load(info['npz'])
    J = d['fitted_joints'][::info['step']].astype(np.float64)
    fl = floor_y(info['dir'])
    if fl is None:
        fl = float(np.percentile(J[:, [10, 11], 1].min(1), 2)) - 0.03
    F, h = canon_features(J)
    return J, F, h - fl, fl


# ---------------- object interaction events ----------------
_MESH = {}


def mesh_kd(mesh_id):
    if mesh_id not in _MESH:
        import trimesh
        from scipy.spatial import cKDTree
        m = trimesh.load(ROOT + '/raw/hiphi/object_meshes/%s.obj' % mesh_id, force='mesh', process=False)
        pts = np.asarray(m.vertices) * 0.01
        try:
            sp, _ = trimesh.sample.sample_surface(m, 8000, seed=0)
            pts = np.concatenate([pts, np.asarray(sp) * 0.01])
        except Exception:
            pass
        _MESH[mesh_id] = (cKDTree(pts), pts)
    return _MESH[mesh_id]


def load_track(info, obj, T):
    tr = np.loadtxt(info['dir'] + '/' + obj['trajectory_path'], delimiter=',', skiprows=1)
    tr = tr[::info['step']][:T]
    return tr[:, 2:5], tr[:, 5:9]


def runs(b, fill_gap=0, min_len=1):
    """bool series -> list of (start, end_exclusive) after gap filling / min length."""
    b = np.asarray(b, bool).copy()
    idx = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
    rr = list(zip(idx[::2], idx[1::2]))
    merged = []
    for s, e in rr:
        if merged and s - merged[-1][1] <= fill_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


def object_events(info, meta, J):
    """Contact (hands/pelvis/feet) and object-motion episodes at HZ. Returns dict or None."""
    from scipy.spatial.transform import Rotation as R
    objs = meta.get('objects') or []
    if not objs:
        return None
    obj = objs[0]
    T = len(J)
    pos, quat = load_track(info, obj, T)
    T = min(T, len(pos))
    J = J[:T]
    kd, _ = mesh_kd(obj['mesh_id'])
    Rm = R.from_quat(quat)
    # body points
    fore_l = J[:, 20] - J[:, 18]
    fore_r = J[:, 21] - J[:, 19]
    palm_l = J[:, 20] + 0.08 * fore_l / (np.linalg.norm(fore_l, axis=1, keepdims=True) + 1e-9)
    palm_r = J[:, 21] + 0.08 * fore_r / (np.linalg.norm(fore_r, axis=1, keepdims=True) + 1e-9)
    pts = {'lw': J[:, 20], 'rw': J[:, 21], 'lp': palm_l, 'rp': palm_r, 'pel': J[:, 0], 'lf': J[:, 10], 'rf': J[:, 11]}
    dist = {}
    for k, P in pts.items():
        loc = Rm.inv().apply(P - pos[:T])
        dist[k], _ = kd.query(loc)
    hand = np.minimum.reduce([dist['lw'], dist['rw'], dist['lp'], dist['rp']])
    hand_c = hand < 0.08
    pel_c = dist['pel'] < 0.15
    foot_c = np.minimum(dist['lf'], dist['rf']) < 0.06
    contact = hand_c | pel_c | foot_c
    # object motion
    v = np.r_[0, np.linalg.norm(np.diff(pos[:T], axis=0), axis=1) * HZ]
    rel = (Rm[1:] * Rm[:-1].inv()).magnitude() if T > 1 else np.zeros(0)
    w = np.r_[0, np.degrees(rel) * HZ]
    k = np.ones(3) / 3
    v = np.convolve(v, k, 'same'); w = np.convolve(w, k, 'same')
    moving = (v > 0.05) | (w > 15)
    g, m = int(0.5 * HZ), int(0.3 * HZ)
    ce = runs(contact, g, m)
    me = runs(moving, g, m)
    he = runs(hand_c, g, m)
    return dict(T=T, contact=contact, hand=hand_c, pel=pel_c, foot=foot_c, moving=moving,
                contact_ep=ce, move_ep=me, hand_ep=he, hand_dist=hand, obj_disp=float(np.ptp(pos[:T], 0).max()),
                cat=obj.get('object_category'), mesh_id=obj['mesh_id'], pos=pos[:T], quat=quat[:T])


def cycle_stats(ev, N_list):
    """Complete interaction cycle = contact episode fully inside clip that starts after 0.2 s
    (approach -> touch -> release), or an object-motion episode (object moved then came to rest)."""
    T = ev['T']
    cyc_c = [(s, e) for s, e in ev['contact_ep'] if s > 0.2 * HZ and e < T]
    cyc_m = [(s, e) for s, e in ev['move_ep'] if e < T]
    ends = sorted([e for _, e in cyc_c] + [e for _, e in cyc_m])
    out = dict(n_contact_cycles=len(cyc_c), n_move_cycles=len(cyc_m),
               first_contact_cycle_end_s=(cyc_c[0][1] / HZ if cyc_c else None),
               first_move_cycle_end_s=(cyc_m[0][1] / HZ if cyc_m else None),
               first_cycle_end_s=(ends[0] / HZ if ends else None),
               contact_frac=float(ev['contact'].mean()), moving_frac=float(ev['moving'].mean()),
               obj_disp_m=ev['obj_disp'])
    if len(cyc_m) >= 3:
        st = np.array([s for s, _ in cyc_m]) / HZ
        out['move_period_s'] = float(np.median(np.diff(st)))
    if len(cyc_c) >= 3:
        st = np.array([s for s, _ in cyc_c]) / HZ
        out['contact_period_s'] = float(np.median(np.diff(st)))
    for N in N_list:
        out['cyc_in_%d' % N] = bool(ends and ends[0] <= N * HZ)
        out['ncyc_contact_in_%d' % N] = sum(e <= N * HZ for _, e in cyc_c)
        out['ncyc_move_in_%d' % N] = sum(e <= N * HZ for _, e in cyc_m)
    return out


# ---------------- v2 features: max-joint body distance + object state ----------------
def object_state(info, meta, J, F_unused=None):
    """per-frame object centroid in the body heading frame (relative to pelvis) and object tilt (deg). None if no object."""
    from scipy.spatial.transform import Rotation as R
    objs = meta.get('objects') or []
    if not objs:
        return None
    obj = objs[0]
    T = len(J)
    pos, quat = load_track(info, obj, T)
    T2 = min(T, len(pos))
    _, pts = mesh_kd(obj['mesh_id'])
    c_local = pts.mean(0)
    Rm = R.from_quat(quat[:T2])
    cw = Rm.apply(c_local) + pos[:T2]
    up = Rm.apply(np.array([0, 1.0, 0]))
    tilt = np.degrees(np.arccos(np.clip(up[:, 1], -1, 1)))
    Jt = J[:T2]
    across = (Jt[:, 1] - Jt[:, 2]) + (Jt[:, 16] - Jt[:, 17])
    ang = np.arctan2(across[:, 2], across[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    X = cw - Jt[:, 0]
    xr = c * X[:, 0] - s * X[:, 2]
    zr = s * X[:, 0] + c * X[:, 2]
    rel = np.stack([xr, X[:, 1], zr], -1)
    if T2 < T:  # pad
        rel = np.concatenate([rel, np.repeat(rel[-1:], T - T2, 0)])
        tilt = np.concatenate([tilt, np.repeat(tilt[-1:], T - T2)])
    return rel.astype(np.float32), tilt.astype(np.float32), obj.get('object_category')


def dist_mats_v2(FA, hA, FB, hB, OA=None, OB=None, chunk=200):
    """Dmax[i,j] = max_joint ||FA_i - FB_j||; H root height diff; Dobj = object centroid diff (m) or 0;
    Tilt = |tilt diff| deg or 0."""
    TA, TB = len(FA), len(FB)
    Dm = np.empty((TA, TB), np.float32)
    for i in range(0, TA, chunk):
        a = FA[i:i + chunk, None]
        Dm[i:i + chunk] = np.sqrt(((a - FB[None]) ** 2).sum(-1)).max(-1)
    H = np.abs(hA[:, None] - hB[None, :]).astype(np.float32)
    if OA is None:
        return Dm, H, None, None
    relA, tA, _ = OA; relB, tB, _ = OB
    Do = np.sqrt(((relA[:, None] - relB[None]) ** 2).sum(-1)).astype(np.float32)
    Dt = np.abs(tA[:, None] - tB[None, :]).astype(np.float32)
    return Dm, H, Do, Dt


# ---------------- metric switch ----------------
# v1: mean-over-joints distance, primary threshold 5 cm (f5/w5), alt 3/8 cm
# v2: max-over-joints distance (catches single-limb novelty, e.g. chair-on-shoulder), primary 12 cm, alt 10/15 cm
METRIC = os.environ.get('METRIC', 'v1')
SUFFIX = '' if METRIC == 'v1' else '_' + METRIC
PRIMARY_THR = 0.05 if METRIC == 'v1' else 0.12


def dist_mats_m(FA, hA, FB, hB, chunk=200):
    if METRIC == 'v1':
        return dist_mats(FA, hA, FB, hB, chunk)
    Dm, H, _, _ = dist_mats_v2(FA, hA, FB, hB, chunk=chunk)
    return Dm, H
