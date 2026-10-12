"""Point-cloud contact / penetration / slip labeller (spec v0.1).  Pure numpy (+trimesh for object SDF).

Inputs are WORLD-frame posed dense surface points: X [T,P,3] cm, Y-up, one row per frame at `fps`.
Every point is a MATERIAL point (rigidly bound to a link, or bone_revolution-bound), so the finite difference of
X[:,p] is that material point's velocity.  That is what makes the slip label correct under rolling contact
(heel-to-toe): the material points that touch the ground have ~zero velocity even though the contact patch moves.

Partners
  ground : plane {x : n.x = c}  (n unit, c cm).  d = n.x - c  (cm, + above, - below).
  object : trimesh mesh in object frame + per-frame pose T_obj [T,4,4] (cm).  d = outside-positive signed distance.

Labels (all thresholds in cm, cm/s, frames; defaults in DEFAULTS)
  point contact   c_p[t]   : hysteresis on d: enter d < enter_cm, leave d > leave_cm (state kept in between).
  link contact    c_l[t]   : hysteresis on the link clearance  dl[t] = k-th smallest d over the link's points
                               (k = link_k, robust to a single stray point), then despeckle
                               (fill gaps < min_gap_frames, then drop contacts < min_on_frames).
                               Final point flags = c_p AND c_l[anchor].   Same for regions.
  penetration     pen_p    = max(0, -d);  link/region/frame maxima.
  slip            v_t      = tangential (w.r.t. the partner normal) material velocity; for objects the velocity is
                               RELATIVE to the object (finite difference of the point expressed in the object frame).
                   link slip speed = p25 of v_t over the link's SUPPORT LAYER: points in contact at t-1, t and t+1
                   whose d is within support_tol_cm of the link's lowest point (a rolling foot's support points
                   have ~0 material velocity; points higher in the 1-2 cm hysteresis band do not); slip flag = c_l & speed > slip_cms, despeckled with min_slip_frames.
                   slip distance per contact segment = sum(speed)/fps over its frames.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = dict(
    enter_cm=1.0,          # point/link enters contact below this clearance
    leave_cm=2.0,          # ... and leaves above this
    link_k=3,              # link clearance = k-th smallest point distance (k=1 is the raw min)
    min_on_frames=3,       # drop contact runs shorter than this (0.1 s @30fps)
    min_gap_frames=2,      # fill non-contact gaps shorter than this
    slip_cms=4.0,          # tangential material speed threshold while in contact
    support_tol_cm=0.5,    # slip is measured on the SUPPORT LAYER: stable contact points with d <= group min d + tol
    slip_stat_pct=25.0,    # group slip speed = this percentile of the support-layer speeds (rolling-invariant, noise-robust)
    min_slip_frames=2,     # slip runs shorter than this are dropped
    trim_frames=3,         # first/last frames of every contact segment = touch-down/lift-off TRANSITION (slip reported separately)
    pen_flag_cm=1.0,       # penetration counted as a violation beyond this depth
    smooth_window=5,       # Savitzky-Golay window for positions before differentiating (0/1 = off)
    smooth_order=2,
)


# ----------------------------------------------------------------------------------------------- temporal machinery
def hysteresis(d, enter, leave):
    """d [T,...] -> bool [T,...]: state = 1 if d<enter, 0 if d>leave, else previous (frame 0: d<enter).
    NaN = undefined -> leave."""
    d = np.asarray(d)
    T = d.shape[0]
    flat = d.reshape(T, -1)
    ev = np.where(flat < enter, 1, np.where((flat > leave) | ~np.isfinite(flat), 0, -1)).astype(np.int8)
    ev[0] = np.where(flat[0] < enter, 1, 0)
    idx = np.where(ev >= 0, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return (np.take_along_axis(ev, idx, axis=0) == 1).reshape(d.shape)


def _runs(col):
    """Run-length segments of a 1-D bool array -> (starts, ends_exclusive, values)."""
    T = len(col)
    if T == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    chg = np.flatnonzero(np.diff(col.astype(np.int8))) + 1
    st = np.concatenate([[0], chg]); en = np.concatenate([chg, [T]])
    return st, en, col[st]


def despeckle(c, min_on, min_gap):
    """c [T,K] bool. Fill interior False runs shorter than min_gap, then drop True runs shorter than min_on.
    Runs touching the clip ends are filled/dropped only if short as well (no special casing)."""
    c = np.array(c, bool, copy=True)
    squeeze = c.ndim == 1
    if squeeze:
        c = c[:, None]
    for k in range(c.shape[1]):
        col = c[:, k]
        st, en, v = _runs(col)
        for a, b, val in zip(st, en, v):
            if not val and (b - a) < min_gap and a > 0 and b < len(col):
                col[a:b] = True
        st, en, v = _runs(col)
        for a, b, val in zip(st, en, v):
            if val and (b - a) < min_on:
                col[a:b] = False
    return c[:, 0] if squeeze else c


def segments(c):
    """bool [T] -> list of (start, end_exclusive) True runs."""
    st, en, v = _runs(np.asarray(c, bool))
    return [(int(a), int(b)) for a, b, val in zip(st, en, v) if val]


def smooth_positions(X, window, order):
    if window is None or window < 3 or X.shape[0] < window:
        return X
    from scipy.signal import savgol_filter
    return savgol_filter(X, window, order, axis=0, mode='interp')


def velocity(X, fps):
    """Central finite difference [T,...,3] cm -> cm/s (one-sided at the ends)."""
    return np.gradient(X, 1.0 / fps, axis=0)


# --------------------------------------------------------------------------------------------- distances to partners
def plane_distance(X, normal=(0.0, 1.0, 0.0), offset_cm=0.0):
    n = np.asarray(normal, np.float64); n = n / np.linalg.norm(n)
    return X @ n - offset_cm


def object_signed_distance(X, mesh, T_obj, cull_cm=15.0):
    """X [T,P,3] world cm; mesh: trimesh.Trimesh in object frame (cm, watertight preferred); T_obj [T,4,4] world<-object.
    Returns (d [T,P] outside-positive cm, NaN where culled = farther than cull_cm from the bounding sphere;
             q [T,P,3] points in the object frame; nrm [T,P,3] object-frame outward normal at the closest point)."""
    import trimesh
    T, P, _ = X.shape
    Rinv = np.transpose(T_obj[:, :3, :3], (0, 2, 1))
    q = np.einsum('tij,tpj->tpi', Rinv, X - T_obj[:, None, :3, 3])
    centre = mesh.bounding_sphere.primitive.center; rad = mesh.bounding_sphere.primitive.radius
    near = np.linalg.norm(q - centre, axis=-1) < rad + cull_cm
    d = np.full((T, P), np.nan); nrm = np.full((T, P, 3), np.nan)
    if near.any():
        pq = trimesh.proximity.ProximityQuery(mesh)
        pts = q[near]
        closest, dist, tri = pq.on_surface(pts)
        inside = mesh.contains(pts) if mesh.is_watertight else np.zeros(len(pts), bool)
        sd = np.where(inside, -dist, dist)
        fn = mesh.face_normals[tri]
        d[near] = sd; nrm[near] = fn
    return d, q, nrm


# ------------------------------------------------------------------------------------------------------ labeller
def _kth_smallest(d, idx, k):
    sub = d[:, idx]
    k = min(k, sub.shape[1])
    return np.partition(sub, k - 1, axis=1)[:, k - 1]


def label_partner(d, vt, groups, area=None, cfg=None):
    """Generic labeller for one partner.
    d  [T,P] signed distance cm (NaN = unknown/far);  vt [T,P] tangential speed cm/s (relative to the partner);
    groups: dict name -> {'kind': 'link'|'region', 'idx': int array of point indices}.
    Returns dict of arrays (see README)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    T, P = d.shape
    dd = np.where(np.isfinite(d), d, np.inf)
    c_pt_raw = hysteresis(dd, cfg['enter_cm'], cfg['leave_cm'])
    names = list(groups)
    G = len(names)
    clear = np.stack([_kth_smallest(dd, groups[n]['idx'], cfg['link_k']) for n in names], 1) if G else np.zeros((T, 0))
    c_g_raw = hysteresis(clear, cfg['enter_cm'], cfg['leave_cm'])
    c_g = despeckle(c_g_raw, cfg['min_on_frames'], cfg['min_gap_frames'])
    # point flags gated by their LINK group state
    link_names = [n for n in names if groups[n]['kind'] == 'link']
    owner = np.full(P, -1)
    for n in link_names:
        owner[groups[n]['idx']] = names.index(n)
    gate = np.where(owner[None] >= 0, c_g[:, np.maximum(owner, 0)], False)
    c_pt = c_pt_raw & gate
    pen = np.maximum(0.0, -np.where(np.isfinite(d), d, 0.0))
    # stable contact points (in contact t-1, t, t+1) for slip
    stable = c_pt.copy(); stable[1:] &= c_pt[:-1]; stable[:-1] &= c_pt[1:]
    slip_speed = np.full((T, G), np.nan); n_stable = np.zeros((T, G), np.int32)
    pen_g = np.zeros((T, G)); area_g = np.zeros((T, G))
    a = np.ones(P) if area is None else np.asarray(area, np.float64)
    for gi, n in enumerate(names):
        idx = groups[n]['idx']
        dg = dd[:, idx]
        s = stable[:, idx] & (dg <= dg.min(1, keepdims=True) + cfg['support_tol_cm'])
        v = np.where(s, vt[:, idx], np.nan)
        n_stable[:, gi] = s.sum(1)
        with np.errstate(all='ignore'):
            ok = n_stable[:, gi] > 0
            if ok.any():
                slip_speed[ok, gi] = np.nanpercentile(v[ok], cfg['slip_stat_pct'], axis=1)
        pen_g[:, gi] = pen[:, idx].max(1)
        area_g[:, gi] = (c_pt[:, idx] * a[idx]).sum(1)
    # slip is JUDGED only while the group truly touches (clearance < enter_cm) and has stable support points;
    # frames in the enter/leave band (touch-down / lift-off) carry contact=1 but slip=NaN ("not judged").
    judged = c_g & (clear < cfg['enter_cm']) & (n_stable > 0)
    slip_speed = np.where(judged, slip_speed, np.nan)
    slip_raw = c_g & (np.nan_to_num(slip_speed, nan=0.0) > cfg['slip_cms'])
    slip = despeckle(slip_raw, cfg['min_slip_frames'], 1) & c_g
    interior = np.zeros_like(c_g)
    for gi in range(G):
        for a0, b0 in segments(c_g[:, gi]):
            tr = cfg['trim_frames']
            if b0 - a0 > 2 * tr:
                interior[a0 + tr:b0 - tr, gi] = True
    slip_sustained = slip & interior
    seg = {}
    for gi, n in enumerate(names):
        rows = []
        for a0, b0 in segments(c_g[:, gi]):
            sp = np.nan_to_num(slip_speed[a0:b0, gi])
            spi = np.where(interior[a0:b0, gi], sp, 0.0)
            rows.append(dict(start=a0, end=b0, frames=b0 - a0, slip_dist_cm=float(sp.sum() / cfg['fps']),
                             sustained_slip_dist_cm=float(spi.sum() / cfg['fps']),
                             slip_frames=int(slip[a0:b0, gi].sum()), sustained_slip_frames=int(slip_sustained[a0:b0, gi].sum()), max_pen_cm=float(pen_g[a0:b0, gi].max()),
                             p50_speed_cms=float(np.median(sp))))
        seg[n] = rows
    return dict(group_names=names, group_kind=[groups[n]['kind'] for n in names], clearance=clear,
                contact_raw=c_g_raw, contact=c_g, slip_judged=judged, slip_speed=slip_speed, slip=slip, interior=interior, slip_sustained=slip_sustained, n_stable=n_stable,
                pen=pen_g, contact_area=area_g, point_contact=c_pt, point_pen=pen.astype(np.float32),
                segments=seg, cfg=cfg)


def ground_labels(Xw, fps, groups, area=None, floor_cm=0.0, normal=(0.0, 1.0, 0.0), cfg=None):
    """World points Xw [T,P,3] cm -> labels against the plane n.x = floor_cm."""
    cfg = {**DEFAULTS, **(cfg or {}), 'fps': fps}
    n = np.asarray(normal, np.float64); n /= np.linalg.norm(n)
    d = plane_distance(Xw, n, floor_cm)
    Xs = smooth_positions(Xw, cfg['smooth_window'], cfg['smooth_order'])
    V = velocity(Xs, fps)
    vt = np.linalg.norm(V - (V @ n)[..., None] * n, axis=-1)
    out = label_partner(d, vt, groups, area, cfg)
    out['d'] = d.astype(np.float32); out['vt'] = vt.astype(np.float32)
    return out


def object_labels(Xw, fps, groups, mesh, T_obj, area=None, cfg=None, cull_cm=15.0):
    """Hand (or any) points vs a posed rigid object.  Slip = tangential RELATIVE velocity (object frame)."""
    cfg = {**DEFAULTS, **(cfg or {}), 'fps': fps}
    d, q, nrm = object_signed_distance(Xw, mesh, T_obj, cull_cm)
    qs = smooth_positions(q, cfg['smooth_window'], cfg['smooth_order'])
    Vq = velocity(qs, fps)                                  # relative velocity, object frame
    nn = np.nan_to_num(nrm)
    vt = np.linalg.norm(Vq - (Vq * nn).sum(-1, keepdims=True) * nn, axis=-1)
    out = label_partner(d, vt, groups, area, cfg)
    out['d'] = d.astype(np.float32); out['vt'] = vt.astype(np.float32)
    return out


def groups_from_asset(anchor, region, joint_names, region_names, min_points=1):
    """link groups (one per anchor joint; every point belongs to exactly one link) + region groups."""
    g = {}
    anchor = np.asarray(anchor); region = np.asarray(region)
    for j in np.unique(anchor):
        idx = np.flatnonzero(anchor == j)
        if len(idx) >= min_points:
            g[f'link:{joint_names[j]}'] = dict(kind='link', idx=idx)
    for r in np.unique(region):
        idx = np.flatnonzero(region == r)
        if len(idx) >= min_points:
            g[f'region:{region_names[r]}'] = dict(kind='region', idx=idx)
    return g
