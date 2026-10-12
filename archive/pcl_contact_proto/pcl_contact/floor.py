"""Floor (ground plane height) estimation from posed dense surface points, per capture SESSION.

floor = median, over quasi-static stance frames of all clips in the session, of the lowest foot-region point height.
quasi-static stance frame (per foot): median speed of that foot's points < STATIC_CMS in all three axes combined, and
the foot's lowest point is the lowest point of the whole body within LOW_TOL_CM (so kneeling/lying frames are excluded).
Pooling over a session of different motions keeps the estimate non-circular (cf. legacy contact_label.py 0831 note).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from . import sources as SR

STATIC_CMS = 5.0
LOW_TOL_CM = 0.5
FOOT_REGIONS = (13, 14)


def session_of(group_id):
    clip = group_id.split('::', 1)[-1]
    if '__' in clip:
        p = clip.split('__'); return '__'.join(p[:2]) if len(p) >= 3 else p[0]
    m = re.search(r'(subject\w+)', clip, re.I)
    return m.group(1) if m else clip


def clip_floor_samples(group_id, body, hand_fix='rot_axis', fps=30.0):
    d = SR.load_traj(group_id, body)
    w = SR.world_points(d, body, hand_fix=hand_fix)
    X = w['X']; reg = w['asset'].region.numpy()
    y = X[..., 1]; ymin_all = y.min(1)
    out = []
    for r in FOOT_REGIONS:
        idx = np.flatnonzero(reg == r)
        V = np.gradient(X[:, idx], 1.0 / fps, axis=0)
        sp = np.median(np.linalg.norm(V, axis=-1), axis=1)
        fmin = y[:, idx].min(1)
        ok = (sp < STATIC_CMS) & (fmin <= ymin_all + LOW_TOL_CM)
        out.append(fmin[ok])
    s = np.concatenate(out)
    return s, len(d['frame_index'])


def session_floor(group_ids, body, **kw):
    per_clip = {}; pooled = []
    for g in group_ids:
        s, T = clip_floor_samples(g, body, **kw)
        per_clip[g] = dict(n_static=int(len(s)), frames=int(T), median=float(np.median(s)) if len(s) else None,
                           p10=float(np.percentile(s, 10)) if len(s) else None, p90=float(np.percentile(s, 90)) if len(s) else None)
        pooled.append(s)
    s = np.concatenate(pooled)
    return dict(body=body, floor_cm=float(np.median(s)), n_static=int(len(s)),
                p10=float(np.percentile(s, 10)), p90=float(np.percentile(s, 90)), per_clip=per_clip,
                rule=f'median lowest foot point on quasi-static stance frames (speed<{STATIC_CMS} cm/s, foot lowest within {LOW_TOL_CM} cm)')
