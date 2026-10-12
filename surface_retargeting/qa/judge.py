"""Independent CPU judge (MuJoCo + numpy, no GPU code): the same ruler for CPU UMR, GPU UMR and GPU strict outputs.

Per frame (floor-relative qpos, calibrated floor = z 0):
  ground  : dense points (4096) + visual-hull vertices: min z; region min over DENSE points (contact ruler)
            + MuJoCo collision geometry vs a floor plane (mj_geomDistance)        -> colground
  self    : every collision-geom pair MuJoCo's filter admits, mj_geomDistance (exact for primitives / convex hulls)
  object  : MuJoCo collision geoms vs the object geom (mj_geomDistance) + dense/hull points vs the object SDF
Summary: penetration counts (> 0.5 mm), human->robot contact consistency (human region in contact -> robot region
dense min in [-0.5, 1] mm), lift consistency (human region > 2 cm off -> robot region > 5 mm), contact-point slip
(dense points within 1 mm at t and t+1: tangential speed; > 1 cm/s counts as slip).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, '/home/arenalabs/umr_gpu')
from surface_retargeting.gpu import labels as L                                    # noqa: E402  (pure numpy rules)
from surface_retargeting.gpu.geometry import mujoco_collision_pairs, robot_collision_geoms    # noqa: E402  (pure python)

PEN = 0.0005


def floor_model(xml):
    """Robot model + a floor plane at z = 0.  Returns (model, floor geom id, map original geom id -> new id)."""
    m0 = mujoco.MjModel.from_xml_path(str(xml))
    spec = mujoco.MjSpec.from_file(str(xml))
    g = spec.worldbody.add_geom()
    g.name = 'judge_floor'; g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [50, 50, 0.1]
    g.contype = 0; g.conaffinity = 0
    m = spec.compile()
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'judge_floor')
    gmap = np.array([gi if gi < fid else gi + 1 for gi in range(m0.ngeom)])
    assert all(m.geom_bodyid[gmap[i]] == m0.geom_bodyid[i] and m.geom_type[gmap[i]] == m0.geom_type[i] for i in range(m0.ngeom))
    return m, fid, gmap


class Judge:
    def __init__(self, robot_xml, surface_npz, objpen_mjb=None, object_geoms=None, object_qadr=None):
        self.m, self.floor_gid, gmap = floor_model(robot_xml)
        self.d = mujoco.MjData(self.m)
        s = np.load(surface_npz)
        self.pl = s['points_local'].astype(np.float64); self.pg = gmap[s['geom'].astype(np.int64)]; self.preg = s['region'].astype(np.int64)
        self.hl = s['hull_local'].astype(np.float64); self.hg = gmap[s['hull_geom'].astype(np.int64)]
        self.gmap = gmap
        self.nd = len(self.pl)
        self.col = robot_collision_geoms(self.m)
        self.col = self.col[self.col != self.floor_gid]
        inv = {int(v): i for i, v in enumerate(gmap)}
        self.col0 = [inv[int(g)] for g in self.col]                 # ids in the floor-less (and object) models
        self.pairs = mujoco_collision_pairs(self.m, list(self.col))
        self.obj = None
        if objpen_mjb is not None:
            self.om = mujoco.MjModel.from_binary_path(str(objpen_mjb)); self.od = mujoco.MjData(self.om)
            self.obj = True
            self.og = [int(g) for g in object_geoms]; self.oq = int(object_qadr)

    def points(self):
        d = self.d
        R = d.geom_xmat[self.pg].reshape(-1, 3, 3); X = np.einsum('nij,nj->ni', R, self.pl) + d.geom_xpos[self.pg]
        Rh = d.geom_xmat[self.hg].reshape(-1, 3, 3); H = np.einsum('nij,nj->ni', Rh, self.hl) + d.geom_xpos[self.hg]
        return X, H

    def frame(self, q, obj_pose=None, obj_sdf=None, self_thr=0.02):
        m, d = self.m, self.d
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        X, H = self.points()
        out = dict(dense_z=X[:, 2], hull_zmin=float(H[:, 2].min()), X=X)
        ft = np.zeros(6)
        cg = [mujoco.mj_geomDistance(m, d, int(g), self.floor_gid, 0.05, ft) for g in self.col]
        out['colground_min'] = float(min(cg)) if cg else np.inf
        # self (pairs within self_thr via bounding spheres, then exact distance)
        smin = np.inf
        for g1, g2 in self.pairs:
            if np.linalg.norm(d.geom_xpos[g1] - d.geom_xpos[g2]) - m.geom_rbound[g1] - m.geom_rbound[g2] > self_thr:
                continue
            smin = min(smin, mujoco.mj_geomDistance(m, d, g1, g2, self_thr, ft))
        out['self_min'] = float(smin)
        if self.obj is not None and obj_pose is not None:
            om, od = self.om, self.od
            od.qpos[:] = 0; od.qpos[:m.nq] = q; od.qpos[self.oq:self.oq + 7] = obj_pose
            mujoco.mj_kinematics(om, od)
            omin = np.inf
            for g in self.col0:
                for go in self.og:
                    omin = min(omin, mujoco.mj_geomDistance(om, od, int(g), go, 0.05, ft))
            out['obj_col_min'] = float(omin)
            if obj_sdf is not None:
                dd = obj_sdf(np.concatenate([X, H]), obj_pose)
                out['obj_dense_d'] = dd[:self.nd]; out['obj_pts_min'] = float(dd.min())
        return out


def sphere_sdf(center, radius):
    def f(P, pose):
        Rm = Rot.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
        loc = (P - pose[:3]) @ Rm
        return np.linalg.norm(loc - center, axis=1) - radius
    return f


def judge_clip(J, Q, fps, human_contact, human_clear, obj_pose=None, obj_sdf=None, human_obj_contact=None, step=1, keep=None):
    """Q floor-relative [T,nq].  Returns (per-frame series dict, summary dict)."""
    T = len(Q)
    idx = np.arange(0, T, step)
    R = L.NR
    reg = J.preg
    S = dict(ground_min=[], colground_min=[], self_min=[], region_min=[], obj_min=[], obj_region_min=[])
    Xs = []
    for t in idx:
        o = J.frame(Q[t], None if obj_pose is None else obj_pose[t], obj_sdf)
        z = o['dense_z']
        S['ground_min'].append(min(z.min(), o['hull_zmin'])); S['colground_min'].append(o['colground_min']); S['self_min'].append(o['self_min'])
        rm = np.full(R, np.inf)
        np.minimum.at(rm, reg, z)
        S['region_min'].append(rm)
        Xs.append(o['X'])
        if 'obj_col_min' in o:
            S['obj_min'].append(min(o['obj_col_min'], o.get('obj_pts_min', np.inf)))
            orm = np.full(R, np.inf)
            np.minimum.at(orm, reg, o['obj_dense_d'])
            S['obj_region_min'].append(orm)
    S = {k: np.asarray(v) for k, v in S.items() if len(v)}
    X = np.stack(Xs)
    mm = lambda a: a * 1000
    summ = dict(frames=int(len(idx)))
    summ['ground_pen_frames_gt_0p5mm'] = int((S['ground_min'] < -PEN).sum())
    summ['ground_pen_max_mm'] = round(float(-mm(S['ground_min'].min())), 3)
    summ['colground_pen_frames_gt_0p5mm'] = int((S['colground_min'] < -PEN).sum())
    summ['self_pen_frames_gt_0p5mm'] = int((S['self_min'] < -PEN).sum())
    summ['self_pen_max_mm'] = round(float(max(0.0, -mm(S['self_min'].min()))), 3)
    if 'obj_min' in S:
        summ['obj_pen_frames_gt_0p5mm'] = int((S['obj_min'] < -PEN).sum())
        summ['obj_pen_max_mm'] = round(float(max(0.0, -mm(S['obj_min'].min()))), 3)
    rm = S['region_min']
    hc = np.asarray(human_contact)[idx]
    solid = (rm >= -PEN) & (rm <= 0.001)
    summ['human_contact_region_frames'] = int(hc.sum())
    summ['human_contact_robot_solid_frac'] = round(float((hc & solid).sum() / max(hc.sum(), 1)), 4)
    hcl = np.asarray(human_clear)[idx]
    off = hcl > 0.02
    summ['human_off_robot_lifted_frac'] = round(float((off & (rm > 0.005)).sum() / max(off.sum(), 1)), 4)
    # contact-point slip on the dense ruler (frames are consecutive when step == 1)
    zc = X[..., 2]
    inc = (zc >= -PEN) & (zc <= 0.001)
    both = inc[:-1] & inc[1:]
    v = np.linalg.norm(X[1:, :, :2] - X[:-1, :, :2], axis=-1) * fps / step
    sp = np.where(both, v, np.nan)
    with np.errstate(all='ignore'):
        fr_speed = np.nanmedian(sp, axis=1)
    ok = np.isfinite(fr_speed)
    summ['contact_frames'] = int(ok.sum())
    summ['contact_speed_cms_p50'] = round(float(np.median(fr_speed[ok]) * 100), 2) if ok.any() else None
    summ['contact_speed_cms_p95'] = round(float(np.percentile(fr_speed[ok], 95) * 100), 2) if ok.any() else None
    summ['contact_slip_frac_gt_1cms'] = round(float((fr_speed[ok] > 0.01).mean()), 4) if ok.any() else None
    if keep is not None:
        kk = np.asarray(keep)[idx]
        okk = ok & kk[:-1] & kk[1:]
        summ['kept_contact_frames'] = int(okk.sum())
        summ['kept_contact_speed_cms_p50'] = round(float(np.median(fr_speed[okk]) * 100), 2) if okk.any() else None
        summ['kept_contact_speed_cms_p95'] = round(float(np.percentile(fr_speed[okk], 95) * 100), 2) if okk.any() else None
        summ['kept_contact_slip_frac_gt_1cms'] = round(float((fr_speed[okk] > 0.01).mean()), 4) if okk.any() else None
        # the spec quantity: speed of each LATCHABLE contact point = region-wise lowest dense point that stays in
        # contact at t and t+1 (per region, not the patch median)
        sp_r = []
        zc_ = X[..., 2]
        for r in range(R):
            k = np.flatnonzero(reg == r)
            if not len(k):
                continue
            i1 = k[np.argmin(zc_[:-1, k], axis=1)]
            z1 = zc_[np.arange(len(i1)), i1]; z2 = zc_[np.arange(1, len(i1) + 1), i1]
            okr = (z1 >= -PEN) & (z1 <= 0.001) & (z2 >= -PEN) & (z2 <= 0.001) & kk[:-1] & kk[1:]
            vv = np.linalg.norm(X[1:, :, :2][np.arange(len(i1)), i1] - X[:-1, :, :2][np.arange(len(i1)), i1], axis=-1) * fps / step
            sp_r.append(vv[okr])
        spr = np.concatenate(sp_r) if sp_r else np.zeros(0)
        summ['kept_region_lowest_point_speed_cms_p50'] = round(float(np.median(spr) * 100), 3) if len(spr) else None
        summ['kept_region_lowest_point_speed_cms_p95'] = round(float(np.percentile(spr, 95) * 100), 3) if len(spr) else None
        summ['kept_region_lowest_point_slip_frac_gt_1cms'] = round(float((spr > 0.01).mean()), 4) if len(spr) else None
    if 'obj_region_min' in S and human_obj_contact is not None:
        ho = np.asarray(human_obj_contact)[idx]
        orm = S['obj_region_min']
        osolid = (orm >= -PEN) & (orm <= 0.001)
        summ['human_obj_contact_region_frames'] = int(ho.sum())
        summ['human_obj_contact_robot_solid_frac'] = round(float((ho & osolid).sum() / max(ho.sum(), 1)), 4)
    S['dense_min'] = S['region_min'].min(1)
    return S, summ
