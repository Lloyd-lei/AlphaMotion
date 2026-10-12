"""lloyd's strict constraints, solved on the GPU in the same streaming solver (batched port of the factory post-pass).

Gauges (metres, + separated):
  ground : z - floor over the robot's dense exterior points (4096) U convex-hull vertices of every visual geom
           (minimum over a mesh = minimum over its hull vertices -> exact);
  object : object SDF at the same points;
  self   : MuJoCo-filtered collision-geom pairs, point-vs-SDF (geometry.SelfCollision).
Per frame (sequential, warm-started from the previous frame's correction, batched over sequences):
  verify(q): ground >= -0.5 mm, object >= -0.5 mm, self >= -0.5 mm, every REQUIRED contact region's clearance in
             [-0.5 mm, 1 mm];   pass at the reference -> reference kept bit-exact.
  else iterate a QP:  min |W(q+dq - q_ref)|^2 + w_t|q+dq - q_prev|^2 + w_a|xy anchors|^2 + damp|dq|^2
                      s.t. linearised clearances >= 0.1 mm (rows within 3 cm), required-contact lowest point
                      <= 0.8 mm, joint limits, |dq|_inf <= step;  line search {1, .5, .25}, trust-region shrink;
  frames still failing after max_it are FLAGGED (bitmask), never silently passed.
Afterwards the corrections are Gaussian-blended (sigma 2 frames) into neighbours; a blended frame is kept only if it
verifies, otherwise the per-frame solution stays.
Required contacts: robot region in contact (UMR output, hysteresis) OR (human region in contact AND robot region
within 3 cm (ground) / 5 cm (object)), runs < 3 frames dropped (factory rule).  Slip latch: in a required ground
stance that the human does not slip in, the lowest support points keep their previous-frame xy (soft anchor).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu.geometry import jac_entries
from surface_retargeting.gpu.kin import KinTree
from surface_retargeting.gpu import ipm

PEN_TOL = 0.0005
CONTACT_TOL = 0.001
TARGET_LO = 0.0001
TARGET_HI = 0.0008
ACT = 0.03
SELF_VERIFY = 0.0005    # GPU self gauge is a deepest-point estimate (<= MuJoCo EPA depth): require >= +0.5 mm (conservative)
SELF_TARGET = 0.0015    # self rows aim at +1.5 mm
FLAG_GROUND_PEN, FLAG_OBJ_PEN, FLAG_SELF_PEN, FLAG_GROUND_CONTACT, FLAG_OBJ_CONTACT = 1, 2, 4, 8, 16
FLAG_SLIP = 32          # latched contact point moved > 0.15 mm from its segment target (tangential speed spec)
FLAG_LIFT = 64          # human region > 2 cm off but robot region < 5 mm off
LATCH_BOX = 0.0001      # QP box around the latch target (m)
LATCH_VERIFY = 0.00015  # verification tolerance (frame-to-frame motion <= 0.3 mm -> < 1 cm/s at 30 fps)
LIFT_MIN = 0.005
YAW_BOX = 0.001          # rad, QP box around the segment heading of the latched body
YAW_VERIFY = 0.0015     # rad (3 mrad/frame max change -> < 1 cm/s at 10 cm from the latch point)
LIFT_TARGET = 0.0055


class RobotPoints:
    """Factory dense surface (surface.npz) + hull vertices, as body-frame points."""

    def __init__(self, kt: KinTree, surface_npz, device='cuda'):
        s = np.load(surface_npz)
        m = kt.m
        dl, dg, dr = s['points_local'].astype(np.float64), s['geom'].astype(np.int64), s['region'].astype(np.int64)
        hl, hg = s['hull_local'].astype(np.float64), s['hull_geom'].astype(np.int64)
        breg = s['body_region'].astype(np.int64)
        hr = breg[m.geom_bodyid[hg]]
        self.ndense = len(dl)
        loc = np.concatenate([dl, hl]); geom = np.concatenate([dg, hg]); reg = np.concatenate([dr, hr])
        from scipy.spatial.transform import Rotation as Rot
        gq = m.geom_quat[geom]
        Rg = Rot.from_quat(np.stack([gq[:, 1], gq[:, 2], gq[:, 3], gq[:, 0]], 1)).as_matrix()
        bl = np.einsum('nij,nj->ni', Rg, loc) + m.geom_pos[geom]
        f = dict(device=device, dtype=torch.float32)
        self.blocal = torch.as_tensor(bl, **f)
        self.body = torch.as_tensor(m.geom_bodyid[geom].astype(np.int64), device=device)
        self.region = torch.as_tensor(reg, device=device)
        self.region_np = reg
        self.n = len(bl)
        self.is_hull = torch.zeros(self.n, dtype=torch.bool, device=device); self.is_hull[self.ndense:] = True
        self.has_reg = torch.as_tensor(np.bincount(reg, minlength=L.NR) > 0, device=device)
        self.dense_region_np = dr

    def build_soles(self, kt, jq, lim_lo, lim_hi, n_sole=12, tol=0.002):
        """Sole patch per foot: dense points of the foot region's main body that are within `tol` of the lowest
        foot point at the robot's zero pose (joints 0, clamped; flat feet), FPS-subsampled to n_sole points."""
        from surface_retargeting.gpu.geometry import fps
        nq = kt.nq
        q = torch.zeros(1, nq, dtype=torch.float64, device=self.blocal.device)
        if kt.free_body >= 0:
            q[0, 3] = 1.0
        q[0, jq] = torch.minimum(torch.maximum(q[0, jq], lim_lo), lim_hi)
        K = kt.fk(q)
        X = self.world(K)[0, :self.ndense].cpu().numpy()
        reg = self.region_np[:self.ndense]; body = self.body[:self.ndense].cpu().numpy()
        idx_all, ok = [], []
        for r in (L.RID['L_foot'], L.RID['R_foot']):
            k = np.flatnonzero(reg == r)
            if len(k) == 0:
                idx_all.append(np.zeros(n_sole, np.int64)); ok.append(False); continue
            z = X[k, 2]
            cand = k[z <= z.min() + tol]
            bmain = np.bincount(body[cand]).argmax()
            cand = cand[body[cand] == bmain]
            sel = cand[fps(X[cand], n_sole)] if len(cand) > n_sole else cand
            sel = np.concatenate([sel, np.repeat(sel[:1], n_sole - len(sel))]) if len(sel) < n_sole else sel
            idx_all.append(sel); ok.append(len(cand) >= 3)
        self.sole_idx = torch.as_tensor(np.stack(idx_all), device=self.blocal.device)        # [2,Ks]
        self.sole_ok = torch.as_tensor(ok, device=self.blocal.device)
        return self.sole_idx

    def world(self, K, idx=None):
        body = self.body if idx is None else self.body[idx]
        bl = self.blocal if idx is None else self.blocal[idx]
        if body.dim() == 1:
            R = K['xmat'][:, body]; x = K['xpos'][:, body]
            return (R @ bl[..., None])[..., 0] + x
        B = body.shape[0]
        ar = torch.arange(B, device=body.device)[:, None]
        return (K['xmat'][ar, body] @ bl[..., None])[..., 0] + K['xpos'][ar, body]


def seg_min(v, reg, nreg=L.NR):
    """v [B,N] -> per-region min [B,nreg] and argmin index into N."""
    B = v.shape[0]
    out = torch.full((B, nreg), float('inf'), device=v.device, dtype=v.dtype)
    out = out.scatter_reduce(1, reg[None].expand(B, -1), v, reduce='amin')
    # argmin: first point attaining the min
    hit = v == out.gather(1, reg[None].expand(B, -1))
    big = torch.full_like(v, 2 ** 30, dtype=torch.float32)
    idxf = torch.where(hit, torch.arange(v.shape[1], device=v.device, dtype=torch.float32)[None], big)
    am = torch.full((B, nreg), 2 ** 30, device=v.device, dtype=torch.float32).scatter_reduce(
        1, reg[None].expand(B, -1), idxf, reduce='amin')
    return out, am.long().clamp(max=v.shape[1] - 1)


class StrictProjector:
    """Gauges, flags and one linearised QP step of lloyd's strict spec (DATA_SPEC_v2), batched over frames.

    contact   : required region -> min over its DENSE points of clearance in [-0.5, 1.0] mm (QP aims [0.1, 0.8])
    no-pen    : every point (dense U visual-hull vertices U analytic collision primitives) >= -0.5 mm; object SDF
                likewise; self (collision geoms, point-vs-SDF) >= 0 (conservative)
    no-slip   : latched contact point stays within +-0.15 mm (xy for ground, xyz in the object frame) of its segment
                target (QP box +-0.1 mm)
    lift      : region whose human counterpart is > 2 cm off -> region dense min > 5 mm (QP aims 5.5 mm)
    """

    def __init__(self, kt, pts: RobotPoints, selfcol=None, obj=None, w_root_pos=3.0, w_root_rot=1.0, w_joint=1.0,
                 w_temporal=0.3, damping=1e-4, step=0.15, k_ground=256, k_obj=128, k_self=32,
                 lim_lo=None, lim_hi=None, jq=None, jd=None, colground=None):
        self.kt, self.pts, self.sc, self.obj = kt, pts, selfcol, obj
        nv = kt.nv; dev = kt.device
        w = torch.full((nv,), w_joint, dtype=torch.float64, device=dev)
        if kt.free_body >= 0:
            f0 = kt.free_dof0; w[f0:f0 + 3] = w_root_pos; w[f0 + 3:f0 + 6] = w_root_rot
        self.w2 = w * w
        self.w_t, self.damp = w_temporal, damping
        self.step0 = step
        self.kg, self.ko, self.ks = k_ground, k_obj, k_self
        self.jq, self.jd, self.lo, self.hi = jq, jd, lim_lo, lim_hi
        self.qp_dtype = torch.float64; self.qp_eps = 1e-9; self.qp_iters = 40; self.self_act = ACT
        self.cg = colground
        self.latch_verify = LATCH_VERIFY; self.latch_box = LATCH_BOX
        self.nd = pts.ndense
        self.dreg = pts.region[:pts.ndense]
        try:
            from surface_retargeting.gpu.narrow_warp import PointGauges
            self.gauge = PointGauges(pts, L.NR)
        except Exception as e:                     # pragma: no cover
            print('[strict] warp gauges unavailable, torch fallback:', e)
            self.gauge = None

    # ------------------------------------------------------------------ gauges
    def measure(self, q, floor, pose, req_g, req_o, self_act=None, latch=None, lift_g=None, lift_o=None, yaw=None, **_):
        """latch: (idx [B,A] dense point ids, target world [B,A,3], valid [B,A]); lift_*: [B,R] bool."""
        self_act = self.self_act if self_act is None else self_act
        kt = self.kt
        K = kt.fk(q)
        B = q.shape[0]
        dev = q.device
        if self.gauge is not None:
            z, _, _ = self.gauge.ground(K, floor)
        else:
            z = self.pts.world(K)[..., 2] - floor[:, None].float()
        greg, gam = seg_min(z[:, :self.nd], self.dreg)                    # contact: dense points only
        gmin = z.amin(-1)                                                  # penetration: all points
        out = dict(K=K, z=z, greg=greg, gam=gam, pose=pose)
        if self.cg is not None and self.cg.n:
            cl, cP = self.cg.lowest(K, floor)
            out.update(cg_cl=cl, cg_P=cP)
            gmin = torch.minimum(gmin, cl.amin(-1))
        out['gmin'] = gmin
        inf = torch.full((B,), float('inf'), device=dev)
        if self.obj is not None:
            if self.gauge is not None and self.obj.kind in ('sphere', 'box'):
                d, _, _ = self.gauge.object(K, self.obj, pose)
            else:
                d, _ = self.obj.sdf(self.pts.world(K), pose)
            oreg, oam = seg_min(d[:, :self.nd], self.dreg)
            out.update(od=d, oreg=oreg, oam=oam, omin=d.amin(-1))
        else:
            out.update(omin=inf, oreg=torch.full((B, L.NR), float('inf'), device=dev))
        if self.sc is not None and self_act > 0:
            smin, ent = self.sc.min_dist(K, self_act)
            out.update(smin=smin, sent=ent)
        else:
            out.update(smin=inf)
        relu = torch.relu
        f = torch.zeros(B, dtype=torch.int32, device=dev)
        f |= (out['gmin'] < -PEN_TOL).int() * FLAG_GROUND_PEN
        f |= (out['omin'] < -PEN_TOL).int() * FLAG_OBJ_PEN
        f |= (out['smin'] < SELF_VERIFY).int() * FLAG_SELF_PEN
        gbad = req_g & ~((greg >= -PEN_TOL) & (greg <= CONTACT_TOL))
        f |= gbad.any(-1).int() * FLAG_GROUND_CONTACT
        oreg = out['oreg']
        obad = req_o & ~((oreg >= -PEN_TOL) & (oreg <= CONTACT_TOL))
        f |= obad.any(-1).int() * FLAG_OBJ_CONTACT
        v = relu(-out['gmin'] - PEN_TOL)
        v = v + torch.where(torch.isfinite(out['omin']), relu(-out['omin'] - PEN_TOL), torch.zeros_like(v))
        v = v + torch.where(torch.isfinite(out['smin']), relu(SELF_VERIFY - out['smin']), torch.zeros_like(v))
        v = v + ((relu(greg - CONTACT_TOL) + relu(-greg - PEN_TOL)) * req_g).nan_to_num(0, 0, 0).sum(-1)
        v = v + torch.where(req_o & torch.isfinite(oreg), relu(oreg - CONTACT_TOL) + relu(-oreg - PEN_TOL), torch.zeros_like(oreg)).sum(-1)
        if latch is not None:
            li, lt, lv = latch
            P = self.pts.world(K, li)
            dev_ = (P.double() - lt).abs()                                 # [B,A,3]
            tol = self._latch_tol(dev)
            bad = lv & (dev_ > tol).any(-1)
            f |= bad.any(-1).int() * FLAG_SLIP
            v = v + (relu(dev_ - tol).sum(-1) * lv).sum(-1).float()
            out['latch'] = latch
        if yaw is not None:
            yb, yt, yv = yaw
            Rm = K['xmat'][torch.arange(B, device=dev)[:, None], yb]
            ye = torch.remainder(torch.atan2(Rm[..., 1, 0], Rm[..., 0, 0]).double() - yt + torch.pi, 2 * torch.pi) - torch.pi
            bad = yv & (ye.abs() > YAW_VERIFY)
            f |= bad.any(-1).int() * FLAG_SLIP
            v = v + (relu(ye.abs() - YAW_VERIFY) * yv).sum(-1).float() * 0.1
            out['yaw_err'] = ye
        if lift_g is not None:
            bad = lift_g & (greg < LIFT_MIN)
            f |= bad.any(-1).int() * FLAG_LIFT
            v = v + (relu(LIFT_MIN - greg).nan_to_num(0, 0, 0) * lift_g).sum(-1)
            out['lift_g'] = lift_g
        if lift_o is not None and self.obj is not None:
            bad = lift_o & (oreg < LIFT_MIN)
            f |= bad.any(-1).int() * FLAG_LIFT
            v = v + (torch.where(torch.isfinite(oreg), relu(LIFT_MIN - oreg), torch.zeros_like(oreg)) * lift_o).sum(-1)
            out['lift_o'] = lift_o
        out['flags'] = f; out['viol'] = v
        return out

    # ------------------------------------------------------------------ one QP step
    def qp_step(self, q, q_ref, q_prev, has_prev, mm, req_g, req_o, anchors, step, latch=None, lift_g=None, lift_o=None, yaw=None,
                box=None, **_):
        kt = self.kt; nv = kt.nv; B = q.shape[0]; dev = q.device; f64 = torch.float64
        K = mm['K']
        Gs, hs = [], []
        z = mm['z']
        INF = torch.full((), float('inf'), dtype=f64, device=dev)
        Xsel = lambda ix: self.pts.world(K, ix)

        def zrow(ix):
            P = Xsel(ix)
            return kt.jac(K, P, self.pts.body[ix])[:, :, 2, :].to(f64), torch.gather(z, 1, ix).to(f64)
        # ground penetration: the k lowest of all points within ACT
        key = torch.where(z < ACT, z, torch.full_like(z, float('inf')))
        zz, gi = torch.topk(key, min(self.kg, key.shape[1]), dim=1, largest=False)
        gv = torch.isfinite(zz)
        Jz, _ = zrow(gi)
        Gs.append(-Jz * gv[..., None]); hs.append(torch.where(gv, zz.to(f64) - TARGET_LO, INF))
        if 'cg_cl' in mm:
            cl, cP = mm['cg_cl'], mm['cg_P']
            cv = cl < ACT
            Jc = kt.jac(K, cP, self.cg.body)[:, :, 2, :].to(f64)
            Gs.append(-Jc * cv[..., None]); hs.append(torch.where(cv, cl.to(f64) - TARGET_LO, INF))
        # required ground contact: region's lowest dense point <= HI (latched regions use the latch rows instead)
        ri = mm['gam']
        Jr, zr = zrow(ri)
        rv = req_g & self.pts.has_reg[None]
        Gs.append(Jr * rv[..., None]); hs.append(torch.where(rv, TARGET_HI - zr, INF))
        if lift_g is not None:
            lv = lift_g & self.pts.has_reg[None]
            Gs.append(-Jr * lv[..., None]); hs.append(torch.where(lv, zr - LIFT_TARGET, INF))
        if latch is not None:
            li, lt, lvv = latch
            P = Xsel(li).to(f64)
            J = kt.jac(K, P.float(), self.pts.body[li]).to(f64)               # [B,A,3,nv]
            m3 = lvv[..., None].expand(-1, -1, 3)
            for sgn in (1.0, -1.0):
                Gs.append((sgn * J * m3[..., None]).reshape(B, -1, nv))
                hs.append(torch.where(m3, self.latch_box - sgn * (P - lt), INF).reshape(B, -1))
        if yaw is not None:
            yb, yt, yv = yaw
            ye = mm['yaw_err'] if 'yaw_err' in mm else None
            if ye is None:
                Rm = K['xmat'][torch.arange(B, device=dev)[:, None], yb]
                ye = torch.remainder(torch.atan2(Rm[..., 1, 0], Rm[..., 0, 0]).double() - yt + torch.pi, 2 * torch.pi) - torch.pi
            Pb = K['xpos'][torch.arange(B, device=dev)[:, None], yb]
            _, Jr = kt.jac(K, Pb, yb, rot=True)
            Jw = Jr[:, :, 2, :].to(f64)                                       # world yaw rate rows
            for sgn in (1.0, -1.0):
                Gs.append(sgn * Jw * yv[..., None]); hs.append(torch.where(yv, YAW_BOX - sgn * ye, INF))
        if self.obj is not None:
            od = mm['od']
            key = torch.where(od < ACT, od, torch.full_like(od, float('inf')))
            dd, oi = torch.topk(key, min(self.ko, key.shape[1]), dim=1, largest=False)
            ov = torch.isfinite(dd)
            P = Xsel(oi); _, n = self.obj.sdf(P, mm['pose'])
            J = kt.jac(K, P, self.pts.body[oi])
            Jn = (n[..., None, :] @ J)[..., 0, :].to(f64)
            Gs.append(-Jn * ov[..., None]); hs.append(torch.where(ov, dd.to(f64) - TARGET_LO, INF))
            ri = mm['oam']
            P = Xsel(ri); _, n = self.obj.sdf(P, mm['pose'])
            J = kt.jac(K, P, self.pts.body[ri])
            Jn = (n[..., None, :] @ J)[..., 0, :].to(f64)
            dr = torch.gather(od, 1, ri).to(f64)
            rv = req_o & torch.isfinite(dr) & self.pts.has_reg[None]
            Gs.append(Jn * rv[..., None]); hs.append(torch.where(rv, TARGET_HI - dr, INF))
            if lift_o is not None:
                lv = lift_o & torch.isfinite(dr) & self.pts.has_reg[None]
                Gs.append(-Jn * lv[..., None]); hs.append(torch.where(lv, dr - LIFT_TARGET, INF))
        if self.sc is not None and 'sent' in mm and mm['sent']['b'].numel():
            e = mm['sent']
            sel = e['d'] < ACT
            b = e['b'][sel]
            if b.numel():
                Jp = jac_entries(kt, K, b, e['p'][sel], e['body_p'][sel])
                Jq = jac_entries(kt, K, b, e['q'][sel], e['body_q'][sel])
                row = (e['n'][sel][:, None, :].to(Jp.dtype) @ (Jp - Jq))[:, 0, :].to(f64)
                d = e['d'][sel].to(f64)
                order = torch.argsort(d)
                b, row, d = b[order], row[order], d[order]
                order = torch.argsort(b, stable=True)
                b, row, d = b[order], row[order], d[order]
                cnt = torch.bincount(b, minlength=B)
                first = torch.cumsum(cnt, 0) - cnt
                rank = torch.arange(b.numel(), device=dev) - first[b]
                keep = rank < self.ks
                Gsf = torch.zeros(B, self.ks, nv, dtype=f64, device=dev)
                hsf = torch.full((B, self.ks), float('inf'), dtype=f64, device=dev)
                Gsf[b[keep], rank[keep]] = -row[keep]
                hsf[b[keep], rank[keep]] = d[keep] - SELF_TARGET
                Gs.append(Gsf); hs.append(hsf)
        G = torch.cat(Gs, 1); h = torch.cat(hs, 1)
        r0 = kt.difference(q_ref, q)
        H = torch.diag_embed(self.w2.expand(B, -1) + self.damp).clone()
        g = self.w2 * r0
        if q_prev is not None:
            rp = kt.difference(q_prev, q)
            hp = has_prev.to(f64)[:, None]
            H = H + torch.diag_embed(self.w_t * hp.expand(-1, nv))
            g = g + self.w_t * hp * rp
        lo = torch.full((B, nv), -1.0, dtype=f64, device=dev) * step[:, None]
        hi = -lo
        jlo = (self.lo - q[:, self.jq]); jhi = (self.hi - q[:, self.jq])
        lo[:, self.jd] = torch.maximum(lo[:, self.jd], jlo); hi[:, self.jd] = torch.minimum(hi[:, self.jd], jhi)
        if box is not None:                                    # extra per-dof bounds (temporal continuity)
            lo = torch.maximum(lo, box[0]); hi = torch.minimum(hi, box[1])
            bad = lo > hi
            mid = 0.5 * (lo + hi)
            lo = torch.where(bad, mid, lo); hi = torch.where(bad, mid, hi)
        res = ipm.solve_qp(H, g, lo, hi, G, h, None, None, None, max_iter=self.qp_iters, dtype=self.qp_dtype, eps=self.qp_eps)
        return torch.nan_to_num(res.x)

    def _latch_tol(self, dev):
        # xy (tangential for the floor) 0.15 mm; z 0.55 mm around the 0.45 mm target -> clearance in [-0.1, 1.0] mm
        return torch.tensor([self.latch_verify, self.latch_verify, 0.00055], dtype=torch.float64, device=dev)

    def set_fps(self, fps):
        """Latch tolerances from the 1 cm/s spec: the latched point may deviate <= min(0.15 mm, 0.5 cm/s / fps)
        from its fixed segment target (frame-to-frame motion <= 2x that -> < 1 cm/s)."""
        self.latch_verify = min(LATCH_VERIFY, 0.005 / float(fps))
        self.latch_box = 0.67 * self.latch_verify

    def _clamp(self, q):
        q = q.clone()
        q[:, self.jq] = torch.minimum(torch.maximum(q[:, self.jq], self.lo), self.hi)
        if self.kt.free_body >= 0:
            q[:, 3:7] = q[:, 3:7] / q[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return q


# ---------------------------------------------------------------------------------------------- asynchronous driver
def _pick(mm, idx, nsrc):
    """Select rows idx [B] (indices into a measurement of nsrc elements) -> measurement of B elements."""
    out = {}
    for k, v in mm.items():
        if k == 'K':
            out['K'] = {kk: vv[idx] for kk, vv in v.items()}
        elif k == 'sent':
            remap = torch.full((nsrc,), -1, dtype=torch.long, device=idx.device)
            remap[idx] = torch.arange(idx.numel(), device=idx.device)
            keep = remap[v['b']] >= 0
            e = {kk: vv[keep] for kk, vv in v.items()}
            e['b'] = remap[v['b'][keep]]
            out['sent'] = e
        elif torch.is_tensor(v) and v.dim() and v.shape[0] == nsrc:
            out[k] = v[idx]
        elif v is None:
            out[k] = None
        else:
            out[k] = v
    return out


def _merge(mm_old, mm_new, take):
    """Per element: take mm_new where take [B] else mm_old (same B)."""
    B = take.shape[0]
    out = {}
    for k, v in mm_old.items():
        w = mm_new[k]
        if k == 'K':
            out['K'] = {kk: torch.where(take.reshape(-1, *([1] * (vv.dim() - 1))), w[kk], vv) for kk, vv in v.items()}
        elif k == 'sent':
            ko = ~take[v['b']]; kn = take[w['b']]
            out['sent'] = {kk: torch.cat([v[kk][ko], w[kk][kn]]) for kk in v}
        elif torch.is_tensor(v) and v.dim() and v.shape[0] == B:
            out[k] = torch.where(take.reshape(-1, *([1] * (v.dim() - 1))), w, v)
        else:
            out[k] = v
    return out


