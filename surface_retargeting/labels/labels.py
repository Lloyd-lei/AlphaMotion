"""Point-cloud contact / slip / penetration labels (same rules for robot and human).

Rules (spec, all in the retarget frame, metres; reported in cm / cm/s / mm):
  region contact  : hysteresis on the region clearance c[t,r] (exact lowest point for the robot, lowest dense
                    point for the human): enter c < 1 cm, leave c > 2 cm; contact runs shorter than 3 frames dropped.
  point contact   : point hysteresis (enter d < 1 cm, leave d > 2 cm) AND its region in contact.
  penetration     : max(0, -c)  per region and frame.
  slip            : tangential (w.r.t. partner normal) MATERIAL speed of the region's support points (in contact at
                    t-1, t, t+1 and within 3 mm of the region's lowest point); region speed = median.  slip flag =
                    contact & speed > 3 cm/s sustained >= 3 frames.  For objects the velocity is relative to the
                    object (finite difference in the object frame).
Hysteresis / despeckle / segment helpers adapted from umr_factory/labels/pcl_contact/core.py (scout prototype).
"""
from __future__ import annotations

import numpy as np

CFG = dict(enter=0.01, leave=0.02, min_on=3, slip_speed=0.03, slip_frames=3, support_tol=0.003, smooth_window=5)


def hysteresis(d, enter, leave):
    d = np.asarray(d)
    T = d.shape[0]
    flat = d.reshape(T, -1)
    ev = np.where(flat < enter, 1, np.where((flat > leave) | ~np.isfinite(flat), 0, -1)).astype(np.int8)
    ev[0] = np.where(flat[0] < enter, 1, 0)
    idx = np.where(ev >= 0, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return (np.take_along_axis(ev, idx, axis=0) == 1).reshape(d.shape)


def _runs(col):
    T = len(col)
    if T == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    chg = np.flatnonzero(np.diff(col.astype(np.int8))) + 1
    st = np.concatenate([[0], chg]); en = np.concatenate([chg, [T]])
    return st, en, col[st]


def drop_short(c, min_on):
    c = np.array(c, bool, copy=True)
    c2 = c.reshape(c.shape[0], -1)
    for k in range(c2.shape[1]):
        st, en, v = _runs(c2[:, k])
        for a, b, val in zip(st, en, v):
            if val and (b - a) < min_on:
                c2[a:b, k] = False
    return c2.reshape(c.shape)


def segments(c):
    st, en, v = _runs(np.asarray(c, bool))
    return [(int(a), int(b)) for a, b, val in zip(st, en, v) if val]


def fps_window(fps, cfg=CFG):
    """Savitzky-Golay window: cfg frames at 30 fps, same duration (~0.17 s) at other rates, odd."""
    w = int(round(cfg['smooth_window'] * fps / 30.0))
    return max(5, w | 1)


def smooth(X, w):
    if w is None or w < 3 or X.shape[0] < w:
        return X
    from scipy.signal import savgol_filter
    return savgol_filter(X, w, 2, axis=0, mode='interp')


def region_contact(clear, cfg=CFG):
    return drop_short(hysteresis(clear, cfg['enter'], cfg['leave']), cfg['min_on'])


def label(d_pts, vt_pts, reg_of_pt, clear, cfg=CFG, nreg=13, contact=None):
    """d_pts [T,P] point distances (inf = far); vt_pts [T,P] tangential speeds (m/s); reg_of_pt [P];
    clear [T,R] region clearance; contact [T,R] optional externally fixed region contact.  Returns dict."""
    T, P = d_pts.shape
    c_reg = region_contact(clear, cfg) if contact is None else np.asarray(contact, bool)
    c_pt = hysteresis(np.where(np.isfinite(d_pts), d_pts, np.inf), cfg['enter'], cfg['leave'])
    c_pt &= c_reg[:, reg_of_pt]
    stable = c_pt.copy(); stable[1:] &= c_pt[:-1]; stable[:-1] &= c_pt[1:]
    speed = np.full((T, nreg), np.nan)
    for r in range(nreg):
        idx = np.flatnonzero(reg_of_pt == r)
        if not len(idx):
            continue
        dr = np.where(np.isfinite(d_pts[:, idx]), d_pts[:, idx], np.inf)
        sup = stable[:, idx] & (dr <= dr.min(1, keepdims=True) + cfg['support_tol'])
        v = np.where(sup, vt_pts[:, idx], np.nan)
        ok = sup.any(1)
        if ok.any():
            speed[ok, r] = np.nanmedian(v[ok], axis=1)
    judged = c_reg & np.isfinite(speed)
    slip_raw = judged & (np.nan_to_num(speed) > cfg['slip_speed'])
    slip = drop_short(slip_raw, cfg['slip_frames'])
    pen = np.maximum(0.0, -np.where(np.isfinite(clear), clear, 0.0))
    return dict(contact=c_reg, point_contact=c_pt, slip_speed=speed, slip=slip, slip_judged=judged, pen=pen,
                clearance=clear)


def ground_inputs(X, floor, reg_of_pt, fps, cfg=CFG, nreg=13, reg_clear=None):
    """X [T,P,3] dense world points -> (d, vt, clear)."""
    d = X[..., 2] - floor
    V = np.gradient(smooth(X, fps_window(fps, cfg)), 1.0 / fps, axis=0)
    vt = np.linalg.norm(V[..., :2], axis=-1)
    if reg_clear is None:
        reg_clear = np.full((X.shape[0], nreg), np.inf)
        for r in range(nreg):
            idx = np.flatnonzero(reg_of_pt == r)
            if len(idx):
                reg_clear[:, r] = d[:, idx].min(1)
    return d, vt, reg_clear


def object_inputs(X, obj, reg_of_pt, fps, cfg=CFG, nreg=13, cull=0.05):
    """Signed distance + tangential speed relative to the object for dense points X [T,P,3]."""
    T, P, _ = X.shape
    d = np.full((T, P), np.inf); N = np.zeros((T, P, 3)); Ql = np.zeros((T, P, 3))
    for t in range(T):
        dt, nt, qt = obj.sdf(X[t], t, cull=cull)
        d[t] = dt; N[t] = nt; Ql[t] = qt
    Vl = np.gradient(smooth(Ql, fps_window(fps, cfg)), 1.0 / fps, axis=0)       # object-frame velocity
    Vw = np.einsum('tij,tpj->tpi', obj.rot[:T], Vl)                             # express in world
    vn = (Vw * N).sum(-1, keepdims=True)
    vt = np.linalg.norm(Vw - vn * N, axis=-1)
    clear = np.full((T, nreg), np.inf)
    for r in range(nreg):
        idx = np.flatnonzero(reg_of_pt == r)
        if len(idx):
            clear[:, r] = d[:, idx].min(1)
    return d, vt, clear


def slip_summary(lab, regions, fps, which=None):
    """Per region: contact frames, slip frames, judged-frame speed p50/p95 (cm/s), slip distance (cm)."""
    out = {}
    for r, name in enumerate(regions):
        if which is not None and name not in which:
            continue
        c = lab['contact'][:, r]
        if not c.any():
            continue
        sp = lab['slip_speed'][lab['slip_judged'][:, r], r]
        out[name] = dict(contact_frames=int(c.sum()), slip_frames=int(lab['slip'][:, r].sum()),
                         slip_frac=round(float(lab['slip'][:, r].sum() / max(1, c.sum())), 4),
                         speed_cms_p50=round(float(np.median(sp) * 100), 2) if len(sp) else None,
                         speed_cms_p95=round(float(np.percentile(sp, 95) * 100), 2) if len(sp) else None,
                         slip_dist_cm=round(float(np.nansum(np.where(lab['slip'][:, r], lab['slip_speed'][:, r], 0)) / fps * 100), 2))
    return out


def floor_from_points(X, reg_of_pt, foot_regions, fps, static_speed=0.05, low_tol=0.005):
    """Session floor: median lowest foot point over quasi-static stance frames (foot median speed < 5 cm/s and the
    foot's lowest point is the body's lowest within 5 mm).  Returns (floor, n_samples, method)."""
    zmin_all = X[..., 2].min(1)
    samples = []
    for r in foot_regions:
        idx = np.flatnonzero(reg_of_pt == r)
        if not len(idx):
            continue
        V = np.gradient(X[:, idx], 1.0 / fps, axis=0)
        sp = np.median(np.linalg.norm(V, axis=-1), axis=1)
        fmin = X[:, idx, 2].min(1)
        ok = (sp < static_speed) & (fmin <= zmin_all + low_tol)
        samples.append(fmin[ok])
    s = np.concatenate(samples) if samples else np.zeros(0)
    if len(s) >= 10:
        return float(np.median(s)), int(len(s)), 'median lowest foot point, quasi-static stance frames'
    allfoot = np.concatenate([X[:, np.flatnonzero(reg_of_pt == r), 2].min(1) for r in foot_regions])
    return float(np.percentile(allfoot, 5)), int(len(s)), 'FALLBACK p5 of lowest foot point (too few static frames)'


def region_support_speed(d_pts, vt_pts, reg_of_pt, nreg=13, tol=0.003):
    """Median tangential speed of each region's lowest points (within tol of the region minimum), no contact gating."""
    T = d_pts.shape[0]
    out = np.full((T, nreg), np.inf)
    for r in range(nreg):
        idx = np.flatnonzero(reg_of_pt == r)
        if not len(idx):
            continue
        dr = np.where(np.isfinite(d_pts[:, idx]), d_pts[:, idx], np.inf)
        sup = dr <= dr.min(1, keepdims=True) + tol
        out[:, r] = np.nanmedian(np.where(sup, vt_pts[:, idx], np.nan), axis=1)
    return out


def trim_band_ends(c, clear, enter, min_on):
    """Drop touch-down / lift-off frames at both ends of every contact run while the region is still in the
    hysteresis band (clearance >= enter, i.e. 1-2 cm up).  Those frames are approach/departure, not touch: snapping
    them to the floor would drag a swinging foot along the ground.  Interior band frames stay (no chatter)."""
    c = np.array(c, bool, copy=True)
    for r in range(c.shape[1]):
        for a, b in segments(c[:, r]):
            while a < b and clear[a, r] >= enter:
                c[a, r] = False; a += 1
            while b > a and clear[b - 1, r] >= enter:
                c[b - 1, r] = False; b -= 1
    return drop_short(c, min_on)
