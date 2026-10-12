"""Contact labels (same rules as the factory's umrf/labels.py, re-implemented vectorised; numpy, per clip).

  region contact : hysteresis on region clearance (enter < 1 cm, leave > 2 cm), runs shorter than 3 frames dropped
  slip           : contact & median tangential speed of the region's support points > 3 cm/s for >= 3 frames
  floor          : median lowest foot point over quasi-static stance frames (factory floor_from_points)
"""
from __future__ import annotations

import numpy as np

REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
RID = {r: i for i, r in enumerate(REGIONS)}
NR = len(REGIONS)
CFG = dict(enter=0.01, leave=0.02, min_on=3, slip_speed=0.03, slip_frames=3, support_tol=0.003, smooth_window=5)
# UMR body-segment schema smplx_55_to_19_upper_arm_v1 part id -> region
SEG19_REGION = {1: 'R_hand', 2: 'R_thigh', 3: 'L_upper_arm', 4: 'trunk', 5: 'L_shin', 6: 'L_foot', 7: 'trunk',
                8: 'R_foot', 9: 'R_upper_arm', 10: 'L_hand', 11: 'R_shin', 12: 'L_forearm', 13: 'R_forearm',
                14: 'L_thigh', 15: 'trunk', 16: 'trunk', 17: 'trunk', 18: 'L_upper_arm', 19: 'R_upper_arm'}


def part_to_region(part_ids):
    return np.array([RID[SEG19_REGION.get(int(p), 'trunk')] for p in part_ids], np.int64)


def hysteresis(d, enter, leave):
    d = np.asarray(d)
    T = d.shape[0]
    flat = d.reshape(T, -1)
    ev = np.where(flat < enter, 1, np.where((flat > leave) | ~np.isfinite(flat), 0, -1)).astype(np.int8)
    ev[0] = np.where(flat[0] < enter, 1, 0)
    idx = np.where(ev >= 0, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return (np.take_along_axis(ev, idx, axis=0) == 1).reshape(d.shape)


def drop_short(c, min_on):
    """Vectorised: remove True-runs shorter than min_on along axis 0."""
    c = np.asarray(c, bool)
    T = c.shape[0]
    f = c.reshape(T, -1)
    # run start index for each True element
    start = np.where(f & ~np.vstack([np.zeros((1, f.shape[1]), bool), f[:-1]]), np.arange(T)[:, None], 0)
    np.maximum.accumulate(start, axis=0, out=start)
    end = np.where(f & ~np.vstack([f[1:], np.zeros((1, f.shape[1]), bool)]), np.arange(T)[:, None], T)
    end = np.minimum.accumulate(end[::-1], axis=0)[::-1]
    length = end - start + 1
    return (f & (length >= min_on)).reshape(c.shape)


def smooth(X, w):
    if w is None or w < 3 or X.shape[0] < w:
        return X
    from scipy.signal import savgol_filter
    return savgol_filter(X, w, 2, axis=0, mode='interp')


def region_min(d, reg, nreg=NR):
    out = np.full(d.shape[:-1] + (nreg,), np.inf)
    for r in range(nreg):
        idx = np.flatnonzero(reg == r)
        if len(idx):
            out[..., r] = d[..., idx].min(-1)
    return out


def label(d_pts, vt_pts, reg, clear, cfg=CFG, nreg=NR):
    T, P = d_pts.shape
    c_reg = drop_short(hysteresis(clear, cfg['enter'], cfg['leave']), cfg['min_on'])
    c_pt = hysteresis(np.where(np.isfinite(d_pts), d_pts, np.inf), cfg['enter'], cfg['leave'])
    c_pt &= c_reg[:, reg]
    stable = c_pt.copy(); stable[1:] &= c_pt[:-1]; stable[:-1] &= c_pt[1:]
    speed = np.full((T, nreg), np.nan)
    for r in range(nreg):
        idx = np.flatnonzero(reg == r)
        if not len(idx):
            continue
        dr = np.where(np.isfinite(d_pts[:, idx]), d_pts[:, idx], np.inf)
        sup = stable[:, idx] & (dr <= dr.min(1, keepdims=True) + cfg['support_tol'])
        v = np.where(sup, vt_pts[:, idx], np.nan)
        ok = sup.any(1)
        if ok.any():
            speed[ok, r] = np.nanmedian(v[ok], axis=1)
    judged = c_reg & np.isfinite(speed)
    slip = drop_short(judged & (np.nan_to_num(speed) > cfg['slip_speed']), cfg['slip_frames'])
    return dict(contact=c_reg, point_contact=c_pt, slip_speed=speed, slip=slip, slip_judged=judged, clearance=clear)


def ground_inputs(X, floor, reg, fps, cfg=CFG):
    d = X[..., 2] - floor
    V = np.gradient(smooth(X, cfg['smooth_window']), 1.0 / fps, axis=0)
    vt = np.linalg.norm(V[..., :2], axis=-1)
    return d, vt, region_min(d, reg)


def object_inputs(d, Ql, rot, nrm_w, reg, fps, cfg=CFG):
    """d [T,P] signed distance (inf far), Ql [T,P,3] object-frame positions, rot [T,3,3], nrm_w [T,P,3]."""
    Vl = np.gradient(smooth(Ql, cfg['smooth_window']), 1.0 / fps, axis=0)
    Vw = np.einsum('tij,tpj->tpi', rot, Vl)
    vn = (Vw * nrm_w).sum(-1, keepdims=True)
    vt = np.linalg.norm(Vw - vn * nrm_w, axis=-1)
    return d, vt, region_min(d, reg)


def floor_from_points(X, reg, foot_regions, fps, static_speed=0.05, low_tol=0.005):
    zmin_all = X[..., 2].min(1)
    samples = []
    for r in foot_regions:
        idx = np.flatnonzero(reg == r)
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
    allfoot = np.concatenate([X[:, np.flatnonzero(reg == r), 2].min(1) for r in foot_regions])
    return float(np.percentile(allfoot, 5)), int(len(s)), 'FALLBACK p5 of lowest foot point'


def human_planted(sole_z, sole_xy, floor, contact_feet, fps, piece=None, z_tol=0.015, v_tol=0.05):
    """Human foot planted flat (torch): median sole-slot height above the floor < z_tol, sole-centroid horizontal
    speed < v_tol (central difference inside a piece), and the foot region in contact.
    sole_z [B,T,2], sole_xy [B,T,2,2], floor [B,T], contact_feet [B,T,2] -> [B,T,2] bool."""
    import torch
    B, T = sole_z.shape[:2]
    if piece is None:
        piece = torch.zeros(B, T, dtype=torch.long, device=sole_z.device)
    low = (sole_z - floor[..., None]) < z_tol
    v = torch.zeros(B, T, 2, device=sole_z.device)
    if T > 2:
        same = (piece[:, 2:] == piece[:, :-2])[..., None]
        d = (sole_xy[:, 2:] - sole_xy[:, :-2]).norm(dim=-1) * (fps / 2.0)
        v[:, 1:-1] = torch.where(same, d, torch.full_like(d, 1e9))
        v[:, 0] = v[:, 1]; v[:, -1] = v[:, -2]
    return low & (v < v_tol) & contact_feet
