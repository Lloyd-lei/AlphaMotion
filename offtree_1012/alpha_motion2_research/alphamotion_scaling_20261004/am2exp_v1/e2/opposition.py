"""E2 opposition descriptor (torch, differentiable, batched, fp32).  Canonical Y-up, cm.

Adroit (adroit_pen J25, AM2 order; palm 2, distal ff 6, mf 10, rf 14, lf 19, th 24):
    pad points  p_k = P_k + G_k c_k        (c_k = pad surface point in the distal joint's canonical rest frame)
    pad normals n_k = G_k nhat_k           (outward palmar normals)
    virtual finger (variant, frozen on dev): p_vf = sum_k w_k p_k,  w = softmax(-|p_k - p_th| / tau) over a finger set
    clearance  c_A = |p_vf - p_th|         (normal-gap variant: (p_vf - p_th) . n_th)
    facing     f   = -n_th . normalize(sum_k w_k n_k)
Franka (relay_franka_full_v2 J11; leftfinger 8, rightfinger 9):
    c_F = q_L + q_R - 0.20 cm   (slide_native, cm; verified against mj_geomDistance of the pad capsules)

Second family ("thumb opposition to any facing hand surface", added in Stage-0 after the planned
pad-to-pad variants failed on the enveloping hammer/relocate grasps): surface pads on every phalanx
(capsule surface point along the palmar normal) + the palm face.  For thumb pads t in {thmiddle, thdistal}
and every other pad o (12 finger phalanges + palm), the pair "faces" when n_t.u > thr, -n_o.u > thr and
-n_t.n_o > thr (u = unit(p_o - p_t)); c_A = min over facing pairs of |p_o - p_t| (HARD gate and min: the
frozen definition used for targets, AN and eval; gradients flow through the selected pair's distance and
normals, a max-pool-like subgradient), f = -n_t.n_o of the selected pair.  No facing pair => c_A = 20 cm +
min distance => clipped to the open gripper.  (A smooth-gate/soft-min variant exists, hard=False, but it
does not reproduce the hard descriptor near the gate threshold and is NOT used.)

Pad geometry is DERIVED FROM THE MJCF (capsule C_*distal surface point radially outward of the Tch_*tip
touch site), i.e. geometry-only; ``pad_table(source='surface_v2')`` uses the dense-asset pad tags of
OUT/assets/surface_v2/adroit_pen.npz when that file exists (cross-checked within 3 mm by tests/eval).
The descriptor actually used by the loss is fixed in ``opposition_frozen.json`` (written once on dev).
"""
from __future__ import annotations
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ._env import PEN_XML, FRANKA_XML, SURFACE_V2, ROOT

AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
DIGITS = ['th', 'ff', 'mf', 'rf', 'lf']            # pad ids 0..4 (surface_v2 'pad' convention)
ADROIT_DISTAL = {'th': 24, 'ff': 6, 'mf': 10, 'rf': 14, 'lf': 19}
ADROIT_PALM = 2
ADROIT_J = 25
FRANKA_LEFT, FRANKA_RIGHT, FRANKA_HAND, FRANKA_EE = 8, 9, 7, 10
FRANKA_J = 11
FRANKA_CLEARANCE_OFFSET_CM = 0.20
FRANKA_MAX_CLEARANCE_CM = 7.8                       # 2 x 4.0 cm stroke - 0.2 cm
FROZEN_PATH = Path(__file__).resolve().parent / 'opposition_frozen.json'

VARIANTS = {
    # name: (finger set, mode)            -- planned family (pad-to-pad, virtual finger)
    'vf_ffmf': (('ff', 'mf'), 'distance'),
    'vf_ffmfrf': (('ff', 'mf', 'rf'), 'distance'),
    'vf_all4': (('ff', 'mf', 'rf', 'lf'), 'distance'),
    'gap_ffmf': (('ff', 'mf'), 'normal_gap'),
    # name: (facing threshold, mode)      -- Stage-0 family (thumb vs any facing hand surface)
    'thface_050': (0.5, 'thumb_facing'),
    'thface_060': (0.6, 'thumb_facing'),
    'thface_070': (0.7, 'thumb_facing'),
}
PLANNED_VARIANTS = ['vf_ffmf', 'vf_ffmfrf', 'vf_all4', 'gap_ffmf']
FACING_VARIANTS = ['thface_050', 'thface_060', 'thface_070']
FACING_GATE_SHARPNESS = 0.05
FACING_SOFTMIN_TAU_CM = 0.2
FACING_PENALTY_CM = 20.0
# surface pads of the facing family: (name, AM2 joint index, capsule/box geom, palmar normal in body frame)
SURFACE_PADS = ([(f + seg, j, f'C_{f}{seg}', (0., -1., 0.)) for f, base in [('ff', 4), ('mf', 8), ('rf', 12), ('lf', 17)]
                 for seg, j in zip(['proximal', 'middle', 'distal'], [base, base + 1, base + 2])]
                + [('palm', ADROIT_PALM, 'C_palm0', (0., -1., 0.)), ('thmiddle', 23, 'C_thmiddle', (-1., 0., 0.)),
                   ('thdistal', 24, 'C_thdistal', (-1., 0., 0.))])
SURFACE_THUMB = ['thmiddle', 'thdistal']
DEFAULT_VARIANT = 'vf_ffmf'
DEFAULT_TAU_CM = 1.0

# Reference constants (cm, canonical rest) — recomputed from the MJCF by pad_table(); a test asserts equality.
ADROIT_PAD_CONST = {
    'ff': ((1.8, 0.705, 0.0), (0.0, 1.0, 0.0)),
    'th': ((2.063, 0.0, 0.764), (0.7071, 0.0, -0.7071)),
}


# ------------------------------------------------------------------------------------------------
# Geometry tables (numpy, from the MJCF)
# ------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _mjcf_adroit_table():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(PEN_XML))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    out = {}
    for k in DIGITS:
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{k}distal')
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'Tch_{k}tip')
        geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'C_{k}distal')
        b = d.xpos[body]
        s = d.site_xpos[site]
        c = d.geom_xpos[geom]
        axis = d.geom_xmat[geom].reshape(3, 3)[:, 2]
        r = float(m.geom_size[geom][0])
        t = float(np.dot(s - c, axis))
        foot = c + t * axis
        radial = s - foot
        depth_inside = r - float(np.linalg.norm(radial))
        n = radial / np.linalg.norm(radial)
        pad = foot + r * n
        out[k] = {'offset_cm': AX @ (pad - b) * 100, 'normal': AX @ n, 'tch_site_cm': AX @ (s - b) * 100,
                  'tch_depth_below_surface_mm': depth_inside * 1000, 'capsule_radius_cm': r * 100,
                  'capsule_centre_cm': AX @ (c - b) * 100, 'capsule_axis': AX @ axis, 'capsule_half_cm': float(m.geom_size[geom][1]) * 100}
    return out


@lru_cache(maxsize=2)
def surface_pad_table():
    """Canonical rest offsets/normals of the facing-family pads (capsule/box surface along the palmar normal)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(PEN_XML))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    names, joints, offs, nrms = [], [], [], []
    for name, j, geom, n_local in SURFACE_PADS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
        b = int(m.geom_bodyid[g])
        Rb = d.xmat[b].reshape(3, 3)
        n_w = Rb @ np.asarray(n_local)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            depth = float(m.geom_size[g][1])            # palm box: half-thickness along body y
        else:
            depth = float(m.geom_size[g][0])            # capsule radius
        x = d.geom_xpos[g] + depth * n_w
        names.append(name)
        joints.append(j)
        offs.append(AX @ (x - d.xpos[b]) * 100)
        nrms.append(AX @ n_w)
    return {'names': names, 'joints': np.array(joints), 'offset': np.array(offs), 'normal': np.array(nrms)}


def surface_pads(P, G):
    """Facing-family pads [B,15,3] cm and normals [B,15,3]."""
    t = surface_pad_table()
    idx = torch.as_tensor(t['joints'], device=P.device)
    off = torch.as_tensor(t['offset'], device=P.device, dtype=torch.float32)
    nrm = torch.as_tensor(t['normal'], device=P.device, dtype=torch.float32)
    Gd = G.float()[:, idx]
    return P.float()[:, idx] + torch.einsum('bkij,kj->bki', Gd, off), torch.einsum('bkij,kj->bki', Gd, nrm)


def thumb_facing_descriptor(P, G, thr, hard=True, tau_cm=FACING_SOFTMIN_TAU_CM, s=FACING_GATE_SHARPNESS):
    t = surface_pad_table()
    pads, normals = surface_pads(P, G)
    ti = [t['names'].index(k) for k in SURFACE_THUMB]
    oi = [i for i, k in enumerate(t['names']) if k not in SURFACE_THUMB]
    pt, nt = pads[:, ti], normals[:, ti]                      # [B,2,3]
    po, no = pads[:, oi], normals[:, oi]                      # [B,13,3]
    v = po[:, None] - pt[:, :, None]                          # [B,2,13,3]
    L = v.norm(dim=-1).clamp_min(1e-6)
    u = v / L[..., None]
    a_t = (nt[:, :, None] * u).sum(-1)
    a_o = -(no[:, None] * u).sum(-1)
    fac = -(nt[:, :, None] * no[:, None]).sum(-1)
    if hard:
        gate = ((a_t > thr) & (a_o > thr) & (fac > thr)).float()
    else:
        gate = torch.sigmoid((a_t - thr) / s) * torch.sigmoid((a_o - thr) / s) * torch.sigmoid((fac - thr) / s)
    eff = (L + (1 - gate) * FACING_PENALTY_CM).flatten(1)
    if hard:
        c, k = eff.min(dim=1)
        w = torch.nn.functional.one_hot(k, eff.shape[1]).float()
    else:
        w = torch.softmax(-eff / tau_cm, dim=1)
        c = -tau_cm * torch.logsumexp(-eff / tau_cm, dim=1)
    f = (w * fac.flatten(1)).sum(1)
    k = w.argmax(1)
    p_a = pt[torch.arange(len(P)), k // len(oi)]
    p_b = po[torch.arange(len(P)), k % len(oi)]
    o = torch.nn.functional.normalize(p_b - p_a, dim=-1)
    return {'c_A': c, 'facing': f, 'p_th': p_a, 'p_vf': p_b, 'o': o, 'centre': 0.5 * (p_a + p_b), 'weights': w,
            'pads': pads, 'normals': normals, 'gate_max': gate.flatten(1).max(1).values}


def _surface_v2_adroit_table(path=None):
    """Area-weighted pad-tag centroid / mean normal per digit from the dense asset (anchor-local cm)."""
    path = Path(path) if path is not None else SURFACE_V2 / 'adroit_pen.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        pts, nrm, pad, anchor, area = z['points_local'], z['normals_local'], z['pad'], z['anchor'], z['area']
    out = {}
    for i, k in enumerate(DIGITS):
        sel = pad == i
        if not sel.any():
            return None
        if not np.all(anchor[sel] == ADROIT_DISTAL[k]):
            # pad points anchored elsewhere would need re-expression; refuse rather than guess
            return None
        w = area[sel] / area[sel].sum()
        n = (nrm[sel] * w[:, None]).sum(0)
        out[k] = {'offset_cm': (pts[sel] * w[:, None]).sum(0), 'normal': n / np.linalg.norm(n), 'count': int(sel.sum())}
    return out


def pad_table(source='mjcf', surface_path=None):
    """Return {'offset': [5,3] cm, 'normal': [5,3]} in DIGITS order for the requested source."""
    if source == 'mjcf':
        t = _mjcf_adroit_table()
    elif source == 'surface_v2':
        t = _surface_v2_adroit_table(surface_path)
        if t is None:
            raise FileNotFoundError('surface_v2 adroit pad tags unavailable')
    elif source == 'constants':
        t = {k: {'offset_cm': np.array(ADROIT_PAD_CONST['th' if k == 'th' else 'ff'][0]),
                 'normal': np.array(ADROIT_PAD_CONST['th' if k == 'th' else 'ff'][1])} for k in DIGITS}
    else:
        raise ValueError(source)
    return {'offset': np.stack([np.asarray(t[k]['offset_cm'], np.float64) for k in DIGITS]),
            'normal': np.stack([np.asarray(t[k]['normal'], np.float64) for k in DIGITS]), 'source': source}


def surface_vs_mjcf_check(surface_path=None, tol_mm=3.0):
    """Cross-check dense-asset pad tags against the MJCF pad surface points and Tch sites.

    Two readings per digit: the area-weighted CENTROID of the tagged patch vs the MJCF pad point (strict; a
    curved ~12 mm patch has its centroid ~1-1.5 mm below the surface) and the NEAREST tagged point to the MJCF pad
    point (does the patch cover the pad?).  'passed' = centroid rule (as planned); 'passed_nearest' = cover rule.
    """
    s = _surface_v2_adroit_table(surface_path)
    if s is None:
        return {'available': False}
    m = _mjcf_adroit_table()
    path = Path(surface_path) if surface_path is not None else SURFACE_V2 / 'adroit_pen.npz'
    with np.load(path, allow_pickle=False) as z:
        pts_all, pad_all = z['points_local'], z['pad']
    rows = {}
    for i, k in enumerate(DIGITS):
        d_pad = float(np.linalg.norm(s[k]['offset_cm'] - m[k]['offset_cm']) * 10)
        d_site = float(np.linalg.norm(s[k]['offset_cm'] - m[k]['tch_site_cm']) * 10)
        near = float(np.linalg.norm(pts_all[pad_all == i] - m[k]['offset_cm'], axis=1).min() * 10)
        rows[k] = {'pad_surface_mm': d_pad, 'nearest_tag_point_mm': near, 'tch_site_mm': d_site,
                   'tch_depth_mm': m[k]['tch_depth_below_surface_mm'], 'normal_cos': float(np.dot(s[k]['normal'], m[k]['normal'])),
                   'count': s[k]['count']}
    ok = all(r['pad_surface_mm'] <= tol_mm for r in rows.values())
    ok_near = all(r['nearest_tag_point_mm'] <= tol_mm for r in rows.values())
    return {'available': True, 'passed': ok, 'passed_nearest': ok_near, 'tol_mm': tol_mm, 'digits': rows}


@lru_cache(maxsize=2)
def franka_pad_table():
    """Franka inner-pad centre per finger (canonical rest offset from the finger joint) from the capsules."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(FRANKA_XML))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    out = {}
    for side, name, joint in [('left', 'panda0_leftfinger', 'panda0_finger_joint1'), ('right', 'panda0_rightfinger', 'panda0_finger_joint2')]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        opening = d.xmat[b].reshape(3, 3) @ m.jnt_axis[j]          # finger moves outward along +axis
        inner_n = -opening / np.linalg.norm(opening)
        caps = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        surf = np.stack([d.geom_xpos[g] + m.geom_size[g][0] * inner_n for g in caps])   # inner surface point per capsule
        out[side] = {'capsule_ids': caps, 'centre_offset_cm': AX @ (surf.mean(0) - d.xpos[b]) * 100,
                     'axis_centre_offset_cm': AX @ (d.geom_xpos[caps].mean(0) - d.xpos[b]) * 100, 'inner_normal_can': AX @ inner_n}
    return out


# ------------------------------------------------------------------------------------------------
# Frozen choice
# ------------------------------------------------------------------------------------------------
def load_frozen(path=None):
    path = Path(path) if path is not None else FROZEN_PATH
    if path.exists():
        cfg = json.loads(path.read_text())
        cfg['frozen'] = True
        return cfg
    return {'variant': DEFAULT_VARIANT, 'tau_cm': DEFAULT_TAU_CM, 'pad_source': 'mjcf', 'frozen': False}


def freeze(choice, evidence, path=None, overwrite=False):
    path = Path(path) if path is not None else FROZEN_PATH
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already frozen; refusing to overwrite')
    rec = {'variant': choice['variant'], 'tau_cm': float(choice['tau_cm']), 'pad_source': choice['pad_source'],
           'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'evidence': evidence}
    path.write_text(json.dumps(rec, indent=2) + '\n')
    return rec


# ------------------------------------------------------------------------------------------------
# Torch descriptor
# ------------------------------------------------------------------------------------------------
def _table_tensors(table, device, dtype=torch.float32):
    return (torch.as_tensor(table['offset'], device=device, dtype=dtype), torch.as_tensor(table['normal'], device=device, dtype=dtype))


def adroit_pads(P, G, table=None):
    """P [B,25,3] cm, G [B,25,3,3] -> pads [B,5,3] cm, normals [B,5,3] (DIGITS order th,ff,mf,rf,lf)."""
    table = table if table is not None else pad_table('mjcf')
    off, nrm = _table_tensors(table, P.device)
    idx = torch.tensor([ADROIT_DISTAL[k] for k in DIGITS], device=P.device)
    Pd = P.float()[:, idx]
    Gd = G.float()[:, idx]
    pads = Pd + torch.einsum('bkij,kj->bki', Gd, off)
    normals = torch.einsum('bkij,kj->bki', Gd, nrm)
    return pads, normals


def adroit_descriptor(P, G, variant=None, tau_cm=None, table=None):
    """Opposition descriptor of an Adroit pose.  Returns dict of [B] / [B,3] tensors."""
    if variant is None or tau_cm is None:
        fz = load_frozen()
        variant = variant or fz['variant']
        tau_cm = tau_cm if tau_cm is not None else fz['tau_cm']
    spec, mode = VARIANTS[variant]
    if mode == 'thumb_facing':
        out = thumb_facing_descriptor(P, G, float(spec), hard=True)
        out.update(variant=variant, tau_cm=float(tau_cm))
        return out
    fingers = spec
    pads, normals = adroit_pads(P, G, table)
    p_th, n_th = pads[:, 0], normals[:, 0]
    fi = [DIGITS.index(k) for k in fingers]
    pf, nf = pads[:, fi], normals[:, fi]
    dist = (pf - p_th[:, None]).norm(dim=-1)
    w = torch.softmax(-dist / float(tau_cm), dim=-1)
    p_vf = (w[..., None] * pf).sum(1)
    n_vf = torch.nn.functional.normalize((w[..., None] * nf).sum(1), dim=-1)
    v = p_vf - p_th
    dist_vf = v.norm(dim=-1)
    if mode == 'normal_gap':
        c = (v * n_th).sum(-1)
    else:
        c = dist_vf
    o = v / dist_vf.clamp_min(1e-6)[:, None]
    f = -(n_th * n_vf).sum(-1)
    return {'c_A': c, 'facing': f, 'p_th': p_th, 'p_vf': p_vf, 'o': o, 'centre': 0.5 * (p_th + p_vf), 'weights': w,
            'pads': pads, 'normals': normals, 'variant': variant, 'tau_cm': float(tau_cm)}


def franka_clearance(slide_native):
    """slide_native [B,11] cm (or [B,2] = (qL,qR)) -> pad clearance c_F [B] cm."""
    s = slide_native.float()
    if s.shape[-1] == FRANKA_J:
        return s[..., FRANKA_LEFT] + s[..., FRANKA_RIGHT] - FRANKA_CLEARANCE_OFFSET_CM
    if s.shape[-1] == 2:
        return s[..., 0] + s[..., 1] - FRANKA_CLEARANCE_OFFSET_CM
    raise ValueError('franka_clearance expects [...,11] or [...,2]')


def franka_descriptor(P, G, slide_native):
    """Franka opposition frame: c_F, opening axis u (left slide axis in world), centre between pads."""
    c = franka_clearance(slide_native)
    pt = franka_pad_table()
    off_l = torch.as_tensor(pt['left']['centre_offset_cm'], device=P.device, dtype=torch.float32)
    off_r = torch.as_tensor(pt['right']['centre_offset_cm'], device=P.device, dtype=torch.float32)
    # finger joints carry the hand rotation (prismatic: derived rotation = parent @ fixed)
    pl = P[:, FRANKA_LEFT].float() + torch.einsum('bij,j->bi', G[:, FRANKA_LEFT].float(), off_l)
    pr = P[:, FRANKA_RIGHT].float() + torch.einsum('bij,j->bi', G[:, FRANKA_RIGHT].float(), off_r)
    u = torch.nn.functional.normalize(pl - pr, dim=-1)
    return {'c_F': c, 'o': u, 'centre': 0.5 * (pl + pr), 'p_left': pl, 'p_right': pr}


def pose9_to_PG(pose9, reach_cm):
    """pose9 [B,J,9] (rot6 + position/reach) -> P [B,J,3] cm, G [B,J,3,3]."""
    from ._sealed_math import rot6d_to_matrix
    G = rot6d_to_matrix(pose9[..., :6].float())
    P = pose9[..., 6:].float() * float(reach_cm)
    return P, G
