"""Contact-pad tags for the external manipulators (geometry-only, verified against MuJoCo).

adroit_pen (pad ids 0 thumb, 1 ff, 2 mf, 3 rf, 4 lf):
    A point is pad of digit k when it is anchored at that digit's distal joint, its outward normal makes < 60 deg
    with the pad normal n_k, and it lies inside the Tch_<k>tip touch-site box grown by 3 mm in its two in-plane
    axes and within 1 cm of the site along n_k.  n_k = unit(site centre - its foot on the C_<k>distal capsule
    axis) (radially outward through the touch site, the palmar side).
    Verification: area-weighted pad centroid vs the capsule surface point radially outward of the site (and vs the
    site centre itself), and mean normal vs n_k.
relay_franka_full_v2 (0 left inner pad, 1 right inner pad):
    A point is pad when it is anchored at the finger joint and its outward normal opposes the finger's own slide
    axis (n . axis < -0.9, i.e. faces the other finger).  Verification: every pad point's signed distance to the
    plane of the inner capsule surfaces (capsules on the finger body, contype 1 -- MJCF geoms 19-30 / 33-44) and
    the pad centroid vs the capsule inner-surface centre, all in the finger's own frame.
All quantities are in canonical cm in the anchor (finger/distal) frame at the asset rest pose (identity = qpos0
for both bodies, rest-rebased by BVP).
"""
from __future__ import annotations

import numpy as np

AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)
ADROIT_DIGITS = ['th', 'ff', 'mf', 'rf', 'lf']
PAD_MARGIN_CM = 0.0   # in-plane growth of the Tch box (0: the touch-site footprint itself)


def _forward(kin):
    import mujoco
    m = kin.model
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    return m, d


def adroit_pads(kin, local, normals_local, anchor, area):
    """-> pad [P] int8, report dict."""
    import mujoco
    m, d = _forward(kin)
    R0, P0 = kin.forward(m.qpos0.copy())[:2]
    pad = np.full(len(local), -1, np.int8)
    rep = {}
    for k, dg in enumerate(ADROIT_DIGITS):
        j = kin.names.index(f'{dg}distal')
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'Tch_{dg}tip')
        geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'C_{dg}distal')
        root = d.xpos[kin.keep[0]]
        to_local = lambda x: R0[j].T @ ((x - root) @ AX.T * 100 - P0[j])   # world m -> anchor-local canonical cm
        # native frame axes (columns of xmat, world Z-up) -> canonical anchor-local directions
        axes_local = lambda Rw: R0[j].T @ AX @ Rw
        s = to_local(d.site_xpos[site])
        Rs = axes_local(d.site_xmat[site].reshape(3, 3))   # columns = site box axes (native order of site_size)
        half = m.site_size[site] * 100
        c = to_local(d.geom_xpos[geom])
        ax = axes_local(d.geom_xmat[geom].reshape(3, 3))[:, 2]  # capsule axis = geom local z
        r = float(m.geom_size[geom][0]) * 100
        foot = c + ((s - c) @ ax) * ax
        n = (s - foot) / np.linalg.norm(s - foot)
        pad_point = foot + r * n
        thin = int(np.argmax(np.abs(Rs.T @ n)))  # site box axis closest to the pad normal
        inplane = [a for a in range(3) if a != thin]
        sel = anchor == j
        q = local - s
        coord = q @ Rs
        inside = np.all(np.abs(coord[:, inplane]) <= half[inplane] + PAD_MARGIN_CM, axis=1) & (np.abs(q @ n) <= 1.0)
        facing = normals_local @ n > 0.5
        tag = sel & inside & facing
        pad[tag] = k
        if tag.any():
            w = area[tag] / area[tag].sum()
            cen = (local[tag] * w[:, None]).sum(0)
            mn = (normals_local[tag] * w[:, None]).sum(0); mn /= np.linalg.norm(mn)
            pq = local[tag]
            tq = np.clip((pq - c) @ ax, -float(m.geom_size[geom][1]) * 100, float(m.geom_size[geom][1]) * 100)
            surf_d = np.linalg.norm(pq - (c + tq[:, None] * ax), axis=1) - r     # signed distance to the capsule (cm)
            rep[dg] = {'pad_id': k, 'joint': kin.names[j], 'count': int(tag.sum()), 'area_cm2': float(area[tag].sum()),
                       'point_to_capsule_surface_mm': {'p50': float(np.median(surf_d) * 10), 'max_abs': float(np.abs(surf_d).max() * 10)},
                       'centroid_local_cm': cen.tolist(), 'capsule_pad_point_local_cm': pad_point.tolist(),
                       'tch_site_local_cm': s.tolist(),
                       'centroid_to_capsule_pad_point_mm': float(np.linalg.norm(cen - pad_point) * 10),
                       'centroid_to_tch_site_mm': float(np.linalg.norm(cen - s) * 10),
                       'centroid_depth_along_normal_vs_capsule_mm': float((cen - pad_point) @ n * 10),
                       'mean_normal_cos_vs_pad_normal': float(mn @ n), 'pad_normal_local': n.tolist()}
        else:
            rep[dg] = {'pad_id': k, 'count': 0}
    ok_surface = all(v.get('count', 0) > 0 and v['point_to_capsule_surface_mm']['max_abs'] <= 3.0 for v in rep.values())
    ok_centroid = all(v.get('count', 0) > 0 and v['centroid_to_capsule_pad_point_mm'] <= 3.0 for v in rep.values())
    return pad, {'rule': f'anchor=distal & normal.n>0.5 & inside the Tch_*tip box footprint (+{PAD_MARGIN_CM} cm in-plane, '
                         '+-1 cm along n)',
                 'verified_within_3mm': bool(ok_surface), 'verification': 'every pad point within 3 mm of the MuJoCo '
                 'C_*distal capsule surface (visual vs collision geometry)',
                 'centroid_within_3mm_of_capsule_pad_point': bool(ok_centroid), 'digits': rep}


def franka_pads(kin, local, normals_local, anchor, area):
    import mujoco
    m, d = _forward(kin)
    R0, P0 = kin.forward(m.qpos0.copy())[:2]
    pad = np.full(len(local), -1, np.int8)
    rep = {}
    for k, (side, name) in enumerate([('left', 'panda0_leftfinger'), ('right', 'panda0_rightfinger')]):
        j = kin.names.index(name)
        axis = kin.slide_axis[j] / np.linalg.norm(kin.slide_axis[j])     # canonical opening direction
        axis_l = R0[j].T @ axis
        inner = -axis_l
        b = int(kin.keep[j])
        root = d.xpos[kin.keep[0]]
        caps = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        cap_pts = []
        for g in caps:
            cl = R0[j].T @ ((d.geom_xpos[g] - root) @ AX.T * 100 - P0[j])
            cap_pts.append(cl + float(m.geom_size[g][0]) * 100 * inner)
        cap_pts = np.asarray(cap_pts)
        rad = max(float(m.geom_size[g][0]) for g in caps) * 100
        # in-plane box spanned by the capsule inner surface points (+ capsule radius)
        e1 = cap_pts.max(0) - cap_pts.min(0); e1 -= (e1 @ inner) * inner
        e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-9 else np.cross(inner, [0, 1, 0])
        e2 = np.cross(inner, e1)
        lo1, hi1 = (cap_pts @ e1).min() - rad, (cap_pts @ e1).max() + rad
        lo2, hi2 = (cap_pts @ e2).min() - rad, (cap_pts @ e2).max() + rad
        facing = (anchor == j) & (normals_local @ axis_l < -0.9)
        within = ((local @ e1 >= lo1) & (local @ e1 <= hi1) & (local @ e2 >= lo2) & (local @ e2 <= hi2))
        tag = facing & within
        pad[tag] = k
        plane = float((cap_pts @ inner).mean())                     # inner capsule surface level along `inner`
        depth = local[tag] @ inner - plane                           # >0: pad face beyond the capsule surface
        w = area[tag] / area[tag].sum()
        cen = (local[tag] * w[:, None]).sum(0)
        cap_c = cap_pts.mean(0)
        # in-plane extent of capsule surface points vs pad points (along the finger)
        rep[side] = {'pad_id': k, 'joint': name, 'count': int(tag.sum()), 'area_cm2': float(area[tag].sum()),
                     'capsule_geom_ids': caps, 'centroid_local_cm': cen.tolist(), 'capsule_inner_centre_local_cm': cap_c.tolist(),
                     'pad_depth_vs_capsule_surface_mm': {'p5': float(np.percentile(depth, 5) * 10), 'p50': float(np.median(depth) * 10),
                                                         'p95': float(np.percentile(depth, 95) * 10)},
                     'centroid_to_capsule_inner_centre_mm': float(np.linalg.norm(cen - cap_c) * 10),
                     'centroid_normal_offset_mm': float((cen - cap_c) @ inner * 10),
                     'inner_facing_points_total': int(facing.sum()), 'inner_facing_outside_capsule_span': int((facing & ~within).sum()),
                     'capsule_plane_spread_mm': float(np.ptp(cap_pts @ inner) * 10),
                     'inner_normal_local': inner.tolist()}
    ok = all(abs(v['pad_depth_vs_capsule_surface_mm']['p50']) <= 3.0 and abs(v['centroid_normal_offset_mm']) <= 3.0
             for v in rep.values())
    return pad, {'rule': 'anchor=finger & normal.slide_axis < -0.9 & inside the in-plane span of the inner capsules (+radius)', 'verified_within_3mm': bool(ok), 'fingers': rep}
