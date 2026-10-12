"""Warp narrow-phase kernel for robot self-collision (one thread per (pair, direction)).

For entry (b, g1, g2) the thread finds the deepest / closest point of geom g1 with respect to the signed distance
function of geom g2 (geom-local SDFs: sphere, capsule, cylinder, box analytic; mesh = outer polytope of the convex hull):
  1. coarse pass over g1's surface samples with g2's coarse plane subset (a lower bound of the fine SDF);
  2. fine SDF only for samples whose coarse value beats the best fine value so far (exact pruning);
  3. projected-gradient refinement  p <- Proj_g1(p - step * grad sdf_g2(p))  (convex problem: min sdf_g2 over g1).
"""
from __future__ import annotations

import torch
import warp as wp

wp.init()


@wp.func
def sdf_geom(t: int, prm: wp.vec3, planes: wp.array2d(dtype=wp.vec4), g: int, npl: int, p: wp.vec3):
    eps = 1.0e-12
    if t == 0:
        r = wp.length(p)
        n = p / wp.max(r, eps)
        return wp.vec4(r - prm[0], n[0], n[1], n[2])
    if t == 1:
        c = wp.vec3(0.0, 0.0, wp.clamp(p[2], -prm[1], prm[1]))
        v = p - c
        l = wp.length(v)
        n = v / wp.max(l, eps)
        return wp.vec4(l - prm[0], n[0], n[1], n[2])
    if t == 2:
        rho = wp.sqrt(p[0] * p[0] + p[1] * p[1])
        dx = rho - prm[0]
        dz = wp.abs(p[2]) - prm[1]
        sz = wp.sign(p[2])
        rx = p[0] / wp.max(rho, eps)
        ry = p[1] / wp.max(rho, eps)
        ox = wp.max(dx, 0.0)
        oz = wp.max(dz, 0.0)
        out = wp.sqrt(ox * ox + oz * oz)
        if out > 0.0:
            n = wp.vec3(rx * ox, ry * ox, sz * oz) / out
            return wp.vec4(out, n[0], n[1], n[2])
        if dx > dz:
            return wp.vec4(dx, rx, ry, 0.0)
        return wp.vec4(dz, 0.0, 0.0, sz)
    if t == 3:
        o = wp.vec3(wp.abs(p[0]) - prm[0], wp.abs(p[1]) - prm[1], wp.abs(p[2]) - prm[2])
        oc = wp.vec3(wp.max(o[0], 0.0), wp.max(o[1], 0.0), wp.max(o[2], 0.0))
        out = wp.length(oc)
        if out > 0.0:
            n = wp.vec3(wp.sign(p[0]) * oc[0], wp.sign(p[1]) * oc[1], wp.sign(p[2]) * oc[2]) / out
            return wp.vec4(out, n[0], n[1], n[2])
        if o[0] >= o[1] and o[0] >= o[2]:
            return wp.vec4(o[0], wp.sign(p[0]), 0.0, 0.0)
        if o[1] >= o[2]:
            return wp.vec4(o[1], 0.0, wp.sign(p[1]), 0.0)
        return wp.vec4(o[2], 0.0, 0.0, wp.sign(p[2]))
    best = float(-1.0e30)
    bn = wp.vec3(0.0, 0.0, 1.0)
    for i in range(npl):
        pl = planes[g, i]
        s = pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3]
        if s > best:
            best = s
            bn = wp.vec3(pl[0], pl[1], pl[2])
    return wp.vec4(best, bn[0], bn[1], bn[2])


@wp.kernel
def sample_sdf_kernel(eb: wp.array(dtype=wp.int32), e1: wp.array(dtype=wp.int32), e2: wp.array(dtype=wp.int32),
                      gpos: wp.array2d(dtype=wp.vec3), grot: wp.array2d(dtype=wp.mat33),
                      typ: wp.array(dtype=wp.int32), prm: wp.array(dtype=wp.vec3), samples: wp.array2d(dtype=wp.vec3),
                      planes: wp.array2d(dtype=wp.vec4), npl: wp.array(dtype=wp.int32), ncoarse: int,
                      thr: wp.array(dtype=float), use_thr: int, out: wp.array2d(dtype=float)):
    """out[i, s] = sdf_g2(sample s of g1); coarse planes if use_thr == 0, else fine planes where coarse < thr[i]."""
    i, s = wp.tid()
    b = eb[i]
    g1 = e1[i]
    g2 = e2[i]
    pw = grot[b, g1] * samples[g1, s] + gpos[b, g1]
    pl = wp.transpose(grot[b, g2]) * (pw - gpos[b, g2])
    t2 = typ[g2]
    nf2 = npl[g2]
    nc2 = wp.min(ncoarse, nf2)
    r = sdf_geom(t2, prm[g2], planes, g2, nc2, pl)
    if use_thr == 0:
        out[i, s] = r[0]
    else:
        if t2 == 4 and r[0] < thr[i]:
            rf = sdf_geom(t2, prm[g2], planes, g2, nf2, pl)
            out[i, s] = rf[0]
        else:
            out[i, s] = 1.0e30


@wp.kernel
def refine_kernel(eb: wp.array(dtype=wp.int32), e1: wp.array(dtype=wp.int32), e2: wp.array(dtype=wp.int32),
                  gpos: wp.array2d(dtype=wp.vec3), grot: wp.array2d(dtype=wp.mat33),
                  typ: wp.array(dtype=wp.int32), prm: wp.array(dtype=wp.vec3), samples: wp.array2d(dtype=wp.vec3),
                  planes: wp.array2d(dtype=wp.vec4), npl: wp.array(dtype=wp.int32),
                  sidx: wp.array(dtype=wp.int32), steps: wp.array(dtype=float), nsteps: int,
                  od: wp.array(dtype=float), op: wp.array(dtype=wp.vec3), on: wp.array(dtype=wp.vec3)):
    """Fine SDF at the chosen sample + projected-gradient refinement; also used to get the fine value at the
    coarse argmin (nsteps = 0)."""
    i = wp.tid()
    b = eb[i]
    g1 = e1[i]
    g2 = e2[i]
    R1 = grot[b, g1]
    x1 = gpos[b, g1]
    R2 = grot[b, g2]
    x2 = gpos[b, g2]
    R2t = wp.transpose(R2)
    R1t = wp.transpose(R1)
    t1 = typ[g1]
    t2 = typ[g2]
    nf2 = npl[g2]
    bp = R1 * samples[g1, sidx[i]] + x1
    r = sdf_geom(t2, prm[g2], planes, g2, nf2, R2t * (bp - x2))
    fbest = r[0]
    bn = R2 * wp.vec3(r[1], r[2], r[3])
    pc = bp
    for k in range(nsteps):
        r = sdf_geom(t2, prm[g2], planes, g2, nf2, R2t * (pc - x2))
        pn = pc - steps[k] * (R2 * wp.vec3(r[1], r[2], r[3]))
        l1 = R1t * (pn - x1)
        r1 = sdf_geom(t1, prm[g1], planes, g1, npl[g1], l1)
        if r1[0] > 0.0:
            l1 = l1 - r1[0] * wp.vec3(r1[1], r1[2], r1[3])
        pc = R1 * l1 + x1
    if nsteps > 0:
        r = sdf_geom(t2, prm[g2], planes, g2, nf2, R2t * (pc - x2))
        if r[0] < fbest:
            fbest = r[0]
            bp = pc
            bn = R2 * wp.vec3(r[1], r[2], r[3])
    od[i] = fbest
    op[i] = bp
    on[i] = bn


class WarpNarrow:
    def __init__(self, shapes, steps):
        dev = shapes.samples.device
        self.typ = wp.from_torch(shapes.type.to(torch.int32).contiguous())
        self.prm = wp.from_torch(shapes.prm.contiguous(), dtype=wp.vec3)
        self.samples = wp.from_torch(shapes.samples.contiguous(), dtype=wp.vec3)
        self.ns = int(shapes.samples.shape[1])
        self.planes = wp.from_torch(shapes.planes.contiguous(), dtype=wp.vec4)
        npl = (shapes.planes[..., 3] > -1e8).sum(-1).to(torch.int32)
        self.npl = wp.from_torch(npl.contiguous())
        self.ncoarse = int(shapes.n_coarse)
        self._steps_t = torch.as_tensor(list(steps), dtype=torch.float32, device=dev)
        self.steps = wp.from_torch(self._steps_t)
        self.nsteps = len(steps)

    def __call__(self, eb, e1, e2, gpos, grot):
        n = eb.numel()
        dev = gpos.device
        od = torch.empty(n, dtype=torch.float32, device=dev)
        op = torch.empty(n, 3, dtype=torch.float32, device=dev)
        on = torch.empty(n, 3, dtype=torch.float32, device=dev)
        if n == 0:
            return od, op, on
        st = wp.stream_from_torch(torch.cuda.current_stream())
        wd = 'cuda:%d' % (dev.index or 0)
        E = [wp.from_torch(x.to(torch.int32).contiguous()) for x in (eb, e1, e2)]
        G = [wp.from_torch(gpos.float().contiguous(), dtype=wp.vec3), wp.from_torch(grot.float().contiguous(), dtype=wp.mat33)]
        S = [self.typ, self.prm, self.samples, self.planes, self.npl]
        vals = torch.empty(n, self.ns, dtype=torch.float32, device=dev)
        thr = torch.zeros(n, dtype=torch.float32, device=dev)
        wp.launch(sample_sdf_kernel, dim=(n, self.ns), inputs=E + G + S + [self.ncoarse, wp.from_torch(thr), 0, wp.from_torch(vals)],
                  device=wd, stream=st)
        am = vals.argmin(1).to(torch.int32)
        none = wp.from_torch(self._steps_t[:0])
        wp.launch(refine_kernel, dim=n, inputs=E + G + S + [wp.from_torch(am), none, 0, wp.from_torch(od),
                  wp.from_torch(op, dtype=wp.vec3), wp.from_torch(on, dtype=wp.vec3)], device=wd, stream=st)
        thr.copy_(od)
        wp.launch(sample_sdf_kernel, dim=(n, self.ns), inputs=E + G + S + [self.ncoarse, wp.from_torch(thr), 1, wp.from_torch(vals)],
                  device=wd, stream=st)
        mv, am2 = vals.min(1)
        am = torch.where(mv < thr, am2.to(torch.int32), am)
        wp.launch(refine_kernel, dim=n, inputs=E + G + S + [wp.from_torch(am), self.steps, self.nsteps, wp.from_torch(od),
                  wp.from_torch(op, dtype=wp.vec3), wp.from_torch(on, dtype=wp.vec3)], device=wd, stream=st)
        return od, op, on


# ------------------------------------------------------------------------------------------ fused point gauges
@wp.kernel
def ground_points_kernel(blocal: wp.array(dtype=wp.vec3), body: wp.array(dtype=wp.int32), region: wp.array(dtype=wp.int32),
                         xmat: wp.array2d(dtype=wp.mat33), xpos: wp.array2d(dtype=wp.vec3), floor: wp.array(dtype=float),
                         z_out: wp.array2d(dtype=float), rmin: wp.array2d(dtype=float)):
    b, i = wp.tid()
    bd = body[i]
    p = xmat[b, bd] * blocal[i] + xpos[b, bd]
    z = p[2] - floor[b]
    z_out[b, i] = z
    wp.atomic_min(rmin, b, region[i], z)


@wp.kernel
def region_argmin_kernel(v: wp.array2d(dtype=float), region: wp.array(dtype=wp.int32), rmin: wp.array2d(dtype=float),
                         ramin: wp.array2d(dtype=wp.int32)):
    b, i = wp.tid()
    r = region[i]
    if v[b, i] == rmin[b, r]:
        wp.atomic_min(ramin, b, r, i)


@wp.kernel
def object_points_kernel(blocal: wp.array(dtype=wp.vec3), body: wp.array(dtype=wp.int32), region: wp.array(dtype=wp.int32),
                         xmat: wp.array2d(dtype=wp.mat33), xpos: wp.array2d(dtype=wp.vec3),
                         orot: wp.array(dtype=wp.mat33), opos: wp.array(dtype=wp.vec3), oscale: wp.array(dtype=float), kind: int,
                         c: wp.vec3, rad: float, boxR: wp.mat33, boxc: wp.vec3, half: wp.vec3,
                         d_out: wp.array2d(dtype=float), rmin: wp.array2d(dtype=float)):
    """kind 0 sphere (center c, radius rad), 1 box (boxR, boxc, half); object frame = (orot[b], opos[b])."""
    b, i = wp.tid()
    bd = body[i]
    p = xmat[b, bd] * blocal[i] + xpos[b, bd]
    sc = oscale[b]
    pl = wp.transpose(orot[b]) * (p - opos[b]) / sc
    d = float(0.0)
    if kind == 0:
        d = wp.length(pl - c) - rad
    else:
        q = wp.transpose(boxR) * (pl - boxc)
        o = wp.vec3(wp.abs(q[0]) - half[0], wp.abs(q[1]) - half[1], wp.abs(q[2]) - half[2])
        oc = wp.vec3(wp.max(o[0], 0.0), wp.max(o[1], 0.0), wp.max(o[2], 0.0))
        d = wp.length(oc) + wp.min(wp.max(o[0], wp.max(o[1], o[2])), 0.0)
    d = d * sc
    d_out[b, i] = d
    wp.atomic_min(rmin, b, region[i], d)


class PointGauges:
    """Fused GPU gauges over a robot point set: per-point clearance + per-region min/argmin (ground, object)."""

    def __init__(self, pts, nreg):
        self.pts = pts
        self.nreg = nreg
        self.bl = wp.from_torch(pts.blocal.float().contiguous(), dtype=wp.vec3)
        self.body = wp.from_torch(pts.body.to(torch.int32).contiguous())
        self.reg = wp.from_torch(pts.region.to(torch.int32).contiguous())

    def _st(self, dev):
        return wp.stream_from_torch(torch.cuda.current_stream()), 'cuda:%d' % (dev.index or 0)

    def ground(self, K, floor):
        xm, xp = K['xmat'].float().contiguous(), K['xpos'].float().contiguous()
        B = xm.shape[0]; dev = xm.device; N = self.pts.n
        z = torch.empty(B, N, dtype=torch.float32, device=dev)
        rmin = torch.full((B, self.nreg), 1e30, dtype=torch.float32, device=dev)
        ram = torch.full((B, self.nreg), 2 ** 30, dtype=torch.int32, device=dev)
        st, wd = self._st(dev)
        wp.launch(ground_points_kernel, dim=(B, N), inputs=[self.bl, self.body, self.reg, wp.from_torch(xm, dtype=wp.mat33),
                  wp.from_torch(xp, dtype=wp.vec3), wp.from_torch(floor.float().contiguous()), wp.from_torch(z), wp.from_torch(rmin)],
                  device=wd, stream=st)
        wp.launch(region_argmin_kernel, dim=(B, N), inputs=[wp.from_torch(z), self.reg, wp.from_torch(rmin), wp.from_torch(ram)],
                  device=wd, stream=st)
        rmin = torch.where(rmin > 1e29, torch.full_like(rmin, float('inf')), rmin)
        return z, rmin, ram.long().clamp(max=N - 1)

    def object(self, K, obj, pose):
        from surface_retargeting.gpu.kin import quat_to_mat
        xm, xp = K['xmat'].float().contiguous(), K['xpos'].float().contiguous()
        B = xm.shape[0]; dev = xm.device; N = self.pts.n
        d = torch.empty(B, N, dtype=torch.float32, device=dev)
        rmin = torch.full((B, self.nreg), 1e30, dtype=torch.float32, device=dev)
        ram = torch.full((B, self.nreg), 2 ** 30, dtype=torch.int32, device=dev)
        R = quat_to_mat(pose[:, 3:7].float()).contiguous(); x = pose[:, :3].float().contiguous()
        sc = (pose[:, 7].float() if pose.shape[-1] > 7 else torch.ones(B, device=dev)).contiguous()
        st, wd = self._st(dev)
        if obj.kind == 'sphere':
            args = [0, wp.vec3(*obj.center.tolist()), float(obj.radius), wp.mat33(*([0.0] * 9)), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)]
        else:
            args = [1, wp.vec3(0.0, 0.0, 0.0), 0.0, wp.mat33(*obj.boxR.flatten().tolist()), wp.vec3(*obj.boxc.tolist()), wp.vec3(*obj.half.tolist())]
        wp.launch(object_points_kernel, dim=(B, N), inputs=[self.bl, self.body, self.reg, wp.from_torch(xm, dtype=wp.mat33),
                  wp.from_torch(xp, dtype=wp.vec3), wp.from_torch(R, dtype=wp.mat33), wp.from_torch(x, dtype=wp.vec3),
                  wp.from_torch(sc)] + args
                  + [wp.from_torch(d), wp.from_torch(rmin)], device=wd, stream=st)
        wp.launch(region_argmin_kernel, dim=(B, N), inputs=[wp.from_torch(d), self.reg, wp.from_torch(rmin), wp.from_torch(ram)],
                  device=wd, stream=st)
        rmin = torch.where(rmin > 1e29, torch.full_like(rmin, float('inf')), rmin)
        return d, rmin, ram.long().clamp(max=N - 1)
