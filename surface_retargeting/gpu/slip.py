"""Contact slip ruler shared by the strict pass, the export and the evaluation (DATA_SPEC_v2 "no slip", lloyd 10-11
relaxation: a robot region may slide when the same HUMAN region slides).

Slip speed of a contact region between frames t-1 and t (backward difference, tangential = xy for the floor):
    the MINIMUM tangential speed over the region's points that are in contact at BOTH frames.
  * robot: in contact = dense point within [-0.5, 1] mm of the floor (the judge's contact band);
  * human: in contact = SMPL-X slot of a region labelled in contact (1 cm / 2 cm hysteresis) whose lowest slot is
    ON the floor (<= 1 cm, the label's enter threshold: a region lifting off / swinging 1-2 cm above the floor is
    not sliding) and that is within 3 mm of the region's lowest slot (the 'support' points of labels.label),
    positions smoothed with the label's Savitzky-Golay window (mocap jitter).
The minimum is the speed of the region's most stationary contact point = the contact (latch / pivot) point:
rolling heel->toe or pivoting about the contact point leaves one point (the rolling line / the pivot) at rest, so
neither counts as slip; a true slide moves every contact point.  (The previous gauges took the region's lowest point
or the median of all contact points, which a pivot about any other point inflates.)

Rule per (frame, region), robot region in contact at t-1 and t with robot slip speed rs > V_SLIP (1 cm/s):
  * human region in contact and itself sliding  hs > V_HUMAN  and  rs <= max(V_ABS_TOL, K_REL * hs)
        -> LEGAL, flag bit 256 'follow_human_slip' (informational, frame kept)
  * otherwise -> bit 32 'slip' (robot skates while the human does not, or slides much faster than the human).
"""
from __future__ import annotations

import numpy as np
import torch

from surface_retargeting.gpu import labels as L

V_SLIP = 0.01          # robot slip threshold (spec: 1 cm/s)
V_HUMAN = 0.02         # human region counts as sliding above this (m/s), see README "slip rule (10-11)"
V_ABS_TOL = 0.02       # robot may slide up to max(V_ABS_TOL, K_REL x human speed) while following
K_REL = 1.5
BAND_LO, BAND_HI = -0.0005, 0.001
FLAG_SLIP = 32
FLAG_FOLLOW = 256      # informational: robot region slides together with the human region (frame KEPT)
DROP_MASK = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128   # a frame is excluded iff flags & DROP_MASK != 0 (bit 256 is not in it)


def region_min_speed(Xa, Xb, za, zb, reg, fps, nreg=L.NR, lo=BAND_LO, hi=BAND_HI, ok_a=None, ok_b=None):
    """torch. Xa, Xb [N,P,3] positions at t-1 / t; za, zb [N,P] clearances; reg [P] region id.
    Returns [N,nreg] min tangential speed (m/s) over points in contact at both frames (nan if none)."""
    inb = lambda z: (z >= lo) & (z <= hi)
    both = inb(za) & inb(zb)
    if ok_a is not None:
        both &= ok_a & ok_b
    v = (Xb[..., :2] - Xa[..., :2]).norm(dim=-1) * float(fps)
    v = torch.where(both, v, torch.full_like(v, float('inf')))
    out = torch.full((v.shape[0], nreg), float('inf'), device=v.device, dtype=v.dtype)
    out = out.scatter_reduce(1, reg[None].expand(v.shape[0], -1), v, reduce='amin')
    return torch.where(torch.isfinite(out), out, torch.full_like(out, float('nan')))


def robot_slip_speed(proj, Q, floor, fps, valid_pair=None, chunk=2048):
    """Q [T,nq] (raw world, floor scalar or [T]).  Returns [T,R] robot region slip speed for (t-1 -> t), nan where
    the region has no point in contact at both frames or t is the first frame / not a valid pair."""
    kt = proj.kt; nd = proj.pts.ndense; reg = proj.pts.region[:nd]
    T = Q.shape[0]; dev = Q.device
    fl = torch.as_tensor(floor, device=dev, dtype=torch.float32).reshape(-1).expand(T) if torch.as_tensor(floor).numel() == 1 \
        else torch.as_tensor(floor, device=dev, dtype=torch.float32)
    out = torch.full((T, L.NR), float('nan'), device=dev)
    for s in range(1, T, chunk):
        e = min(T, s + chunk)
        X = proj.pts.world(kt.fk(Q[s - 1:e]))[:, :nd]
        z = X[..., 2] - fl[s - 1:e, None]
        out[s:e] = region_min_speed(X[:-1], X[1:], z[:-1], z[1:], reg, fps)
        del X, z
    if valid_pair is not None:
        out = torch.where(valid_pair[:, None], out, torch.full_like(out, float('nan')))
    return out


def human_slip_speed(H, floor, hreg, contact, fps, support_tol=L.CFG['support_tol'], smooth_w=L.CFG['smooth_window'], chunk=1024,
                     on_floor=L.CFG['enter'], return_vec=False):
    """numpy in/out. H [T,P,3] human SMPL-X slots (raw world), floor float, hreg [P], contact [T,R] human region
    contact labels.  Returns [T,R] human region slip speed (t-1 -> t), nan where not in contact at both frames.
    return_vec: also [T,R,2] slip velocity (m/s): direction of the support points' mean displacement, magnitude =
    the slip speed (a pivot / roll has ~0 magnitude) -- the strict pass moves a robot latch along it."""
    Hs = L.smooth(np.asarray(H, np.float64), smooth_w)
    T = len(Hs)
    d = Hs[..., 2] - floor
    rmin = L.region_min(d, hreg)                                   # [T,R]
    on = contact & (rmin <= on_floor)                              # [T,R] region in contact and on the floor
    sup = (d <= rmin[:, hreg] + support_tol) & on[:, hreg]        # [T,P]
    out = np.full((T, L.NR), np.nan)
    vec = np.zeros((T, L.NR, 2))
    for s in range(1, T, chunk):
        e = min(T, s + chunk)
        dxy = (Hs[s:e, :, :2] - Hs[s - 1:e - 1, :, :2]) * fps
        v = np.linalg.norm(dxy, axis=-1)
        ok = sup[s:e] & sup[s - 1:e - 1]
        v = np.where(ok, v, np.inf)
        for r in range(L.NR):
            k = np.flatnonzero(hreg == r)
            if len(k):
                m = v[:, k].min(1)
                out[s:e, r] = np.where(np.isfinite(m), m, np.nan)
                if return_vec:
                    okk = ok[:, k]
                    mv = (dxy[:, k] * okk[..., None]).sum(1) / np.maximum(okk.sum(1), 1)[:, None]
                    nrm = np.linalg.norm(mv, axis=-1)
                    vec[s:e, r] = np.where((nrm > 1e-9)[:, None] & np.isfinite(m)[:, None],
                                           mv / np.maximum(nrm, 1e-9)[:, None] * np.where(np.isfinite(m), m, 0.0)[:, None], 0.0)
    return (out, vec) if return_vec else out


def slip_rule(rs, hs, robot_contact=None, v_slip=V_SLIP, v_human=V_HUMAN, v_abs=V_ABS_TOL, k_rel=K_REL):
    """rs, hs [...,T,R] (torch or numpy, nan = not in contact).  robot_contact: optional extra mask [...,T,R].
    Returns (slip_region, follow_region) bool [...,T,R]."""
    tch = torch.is_tensor(rs)
    nz = (lambda x: torch.nan_to_num(x, nan=0.0)) if tch else (lambda x: np.nan_to_num(x, nan=0.0))
    r = nz(rs); h = nz(hs)
    moving = r > v_slip
    if robot_contact is not None:
        moving = moving & robot_contact
    tol = (torch.clamp(k_rel * h, min=v_abs) if tch else np.maximum(k_rel * h, v_abs))
    legal = (h > v_human) & (r <= tol)
    return moving & ~legal, moving & legal
