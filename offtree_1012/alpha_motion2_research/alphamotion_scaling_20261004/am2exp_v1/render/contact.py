"""Per-region ground clearance for colouring and the min-height trace.

Robots: CODE/judge/ground_judge.judge_case (independent judge, collision geoms) when importable, else a local
exact lowest point over the vendor MJCF COLLISION geoms (mesh convex hulls / analytic primitives), re-posed with
the same rigid bindings as the rendered visual meshes.  Both are fed the root height EXACTLY as rendered
(root_height_m - root_offset_cm/100), so colours, trace and picture always agree.

Human: dense surface points (CODE/assets/surface_v2.py + OUT/assets/surface_v2/human_smpl.npz) when present,
else the human's own contact labels (joint-centre ruler from the frozen contact cache), else joint centres minus a
nominal body half-thickness (documented approximation).
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

import numpy as np

from . import _env
from .regions import REGIONS, human_bone_regions, human_joint_regions

TOUCH_CM = 2.5
PEN_CM = 1.0

# nominal half-thickness (cm) around SMPL joint centres -- fallback only
HUMAN_RADIUS_CM = {'Pelvis': 10, 'L_Hip': 8, 'R_Hip': 8, 'Torso': 10, 'Spine': 10, 'Chest': 10, 'Neck': 6, 'Head': 10,
                   'L_Thorax': 7, 'R_Thorax': 7, 'L_Shoulder': 5, 'R_Shoulder': 5, 'L_Elbow': 4, 'R_Elbow': 4,
                   'L_Wrist': 3, 'R_Wrist': 3, 'L_Knee': 5, 'R_Knee': 5, 'L_Ankle': 6, 'R_Ankle': 6, 'L_Toe': 0, 'R_Toe': 0}


def states(region_h_cm):
    """[T,15] heights -> int8 [T,15]: 0 free, 1 touching (< 2.5 cm), 2 penetrating (< -1 cm); NaN -> 0."""
    h = np.asarray(region_h_cm, np.float64)
    s = np.zeros(h.shape, np.int8)
    with np.errstate(invalid='ignore'):
        s[h < TOUCH_CM] = 1
        s[h < -PEN_CM] = 2
    return s


def _judge_module():
    """CODE/judge/ground_judge (imported the judge's own way, ``am2exp_v1.judge.ground_judge``) or None."""
    mod = _env.judge_module('ground_judge')
    return mod if (mod is not None and hasattr(mod, 'judge_case')) else None


def robot_region_heights(rm, R, P, root_y_cm, source='auto'):
    """-> (region_h [T,15] cm above floor, zmin [T] cm, used_source str)."""
    R = np.asarray(R, np.float64)
    P = np.asarray(P, np.float64)
    root_y_cm = np.asarray(root_y_cm, np.float64)
    if source in ('auto', 'judge'):
        mod = _judge_module()
        if mod is not None:
            try:
                out = mod.judge_case(rm.body, R, P, root_y_cm / 100.0)
                rh = np.asarray(out['region_min_height_cm'], np.float64)
                zmin = np.asarray(out.get('zmin_cm', np.nanmin(rh, -1)), np.float64)
                return rh, zmin, 'judge (CODE/judge/ground_judge.judge_case, collision geoms)'
            except Exception as e:
                if source == 'judge':
                    raise
                print(f'[render.contact] judge failed on {rm.body} ({e!r}); local fallback', file=sys.stderr)
        elif source == 'judge':
            raise RuntimeError('CODE/judge/ground_judge.py not available')
    gs = rm.collision if (rm.collision is not None and len(rm.collision)) else rm.visual
    kind = 'collision' if gs is rm.collision else 'visual (no collision geoms)'
    rh = rm.region_min_y(gs, R, P) + root_y_cm[:, None]
    low = np.nanmin(rh, -1)
    return rh, low, f'local lowest point, vendor MJCF {kind} geoms'


@lru_cache(maxsize=1)
def _surface_module():
    p = _env.CODE / 'assets' / 'surface_v2.py'
    a = _env.ASSETS_SURFACE_V2 / 'human_smpl.npz'
    if not (p.is_file() and a.is_file()):
        return None
    try:
        key = '_am2render_surface_v2'
        spec = importlib.util.spec_from_file_location(key, str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def human_dense_points(R, P):
    """Posed dense human surface [T,4096,3] canonical cm root-relative and region [4096], or None."""
    mod = _surface_module()
    if mod is None:
        return None
    try:
        import torch
        asset = mod.load_surface('human_smpl', 'cpu')
        with torch.no_grad():
            pts, _ = mod.pose_surface(asset, torch.as_tensor(np.asarray(R), dtype=torch.float32),
                                      torch.as_tensor(np.asarray(P), dtype=torch.float32))
        reg = np.asarray(getattr(asset, 'region').cpu().numpy() if hasattr(asset, 'region') else asset['region'])
        return pts.cpu().numpy().astype(np.float64), reg.astype(np.int64)
    except Exception:
        return None


def human_region_heights(win, root_y_cm, dense=None):
    """win: case_io.Window of the human (names/parents/P, optional contact labels).
    -> (region_h [T,15] cm, zmin [T] cm, state override [T,15] or None, used_source)."""
    P = np.asarray(win.P, np.float64)
    T, J = P.shape[:2]
    root_y_cm = np.asarray(root_y_cm, np.float64)
    if dense is not None:
        pts, reg = dense
        h = pts[..., 1] + root_y_cm[:, None]
        rh = np.full((T, 15), np.nan)
        for r in range(15):
            m = reg == r
            if m.any():
                rh[:, r] = h[:, m].min(-1)
        # [integration] the dense human hand is an ellipsoid along the forearm (wrist bend is unobservable from
        # positions) and sinks ~20 cm when the palm is planted (WP-ASSETS README) -> hand regions use the wrist
        # joint centre minus its nominal half-thickness, like the joint-centre fallback.
        names = list(win.names)
        for r, wn in ((REGIONS.index('L_hand'), 'L_Wrist'), (REGIONS.index('R_hand'), 'R_Wrist')):
            if wn in names:
                rh[:, r] = P[:, names.index(wn), 1] + root_y_cm - HUMAN_RADIUS_CM[wn]
        return rh, np.nanmin(rh, -1), None, 'dense human surface points (assets/surface_v2); hands = wrist centre - 3 cm'
    bone = human_bone_regions(win.names)
    att = human_joint_regions(win.names)
    if win.contact_hard is not None and win.contact_gap_m is not None:
        gap_cm = np.asarray(win.contact_gap_m, np.float64) * 100.0
        hard = np.asarray(win.contact_hard, bool)
        rh = np.full((T, 15), np.nan)
        st = np.zeros((T, 15), np.int8)
        for r in range(15):
            js = [j for j in range(J) if bone[j] == r or att[j] == r]
            if not js:
                continue
            rh[:, r] = np.nanmin(gap_cm[:, js], -1)
            touch = hard[:, js].any(-1)
            pen = (gap_cm[:, js] < -PEN_CM).any(-1)
            st[:, r] = np.where(pen, 2, np.where(touch, 1, 0))
        return rh, np.nanmin(gap_cm, -1), st, 'human contact labels (joint-centre ruler, frozen cache)'
    rad = np.array([HUMAN_RADIUS_CM.get(n, 5) for n in win.names], np.float64)
    hj = P[..., 1] + root_y_cm[:, None] - rad[None]
    rh = np.full((T, 15), np.nan)
    for r in range(15):
        js = [j for j in range(J) if bone[j] == r or att[j] == r]
        if js:
            rh[:, r] = hj[:, js].min(-1)
    return rh, np.nanmin(hj, -1), None, 'joint centres minus nominal body half-thickness (approximate)'


__all__ = ['REGIONS', 'TOUCH_CM', 'PEN_CM', 'states', 'robot_region_heights', 'human_region_heights', 'human_dense_points']
