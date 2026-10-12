"""GPU distance kernels (torch): robot collision geoms (point-vs-SDF), ground plane, posed rigid objects.

Collision geoms (what MuJoCo collides, contype|conaffinity != 0) get a geom-local signed distance function
  sphere / capsule / cylinder / box : analytic, exact;
  mesh                              : MuJoCo collides the convex hull -> outer polytope approximation of the hull
                                      (support planes along 256 Fibonacci directions + the largest hull faces);
                                      exact inside, conservative (never misses a penetration) by <= `gap` (reported).
and a set of surface samples (farthest-point sampled on the hull / primitive surface).
Pair distance(A,B) ~= min( min_{p in S_A} sdf_B(p), min_{p in S_B} sdf_A(p) ) with the witness point and normal; the
MuJoCo pair filter (contype/conaffinity, same weld body, parent-child, <exclude>) is reproduced.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

SPHERE, CAPSULE, CYLINDER, BOX, POLY = 0, 1, 2, 3, 4


def fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def fps(points, k, seed=0):
    points = np.asarray(points, np.float64)
    if len(points) <= k:
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(points)))]
    d = np.linalg.norm(points - points[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax()); sel.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.asarray(sel)


def sample_triangles(V, F, n, rng):
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(F), size=n, p=area / area.sum())
    u = rng.random((n, 2)); m = u.sum(1) > 1; u[m] = 1 - u[m]
    t = tri[pick]
    return t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])


def mesh_verts(m, g):
    mi = int(m.geom_dataid[g])
    v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(np.float64)
    f = m.mesh_face[m.mesh_faceadr[mi]:m.mesh_faceadr[mi] + m.mesh_facenum[mi]].astype(np.int64)
    return v, f


def polytope_planes(V, n_dirs=256, n_faces=128, n_dirs_c=64, n_faces_c=32):
    """Outer approximation of conv(V): planes n.x + d <= 0.  Returns planes [P,4] whose first n_dirs_c+n_faces_c rows
    are a coarse subset (coarse polytope contains the fine one -> coarse sdf <= fine sdf), hull (V,F), fine gap."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    HV = V[h.vertices]
    eq = h.equations                                   # n.x + d <= 0, unit n
    # merge coplanar facets, rank by area
    tri = V[h.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    key = np.round(eq, 4)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    farea = np.bincount(inv, weights=area)
    top = np.argsort(-farea)[:n_faces]
    face_n = np.stack([eq[inv == k][0, :3] for k in top]) if len(top) else np.zeros((0, 3))
    fc = face_n[:n_faces_c]
    fc = np.concatenate([fc, np.repeat(fibonacci_dirs(1), n_faces_c - len(fc), 0)]) if len(fc) < n_faces_c else fc
    dirs = np.concatenate([fibonacci_dirs(n_dirs_c), fc, fibonacci_dirs(n_dirs), face_n[n_faces_c:]], 0)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    off = (HV @ dirs.T).max(0)
    planes = np.concatenate([dirs, -off[:, None]], 1)
    # gap: approx sdf of true-hull surface samples (<= 0), most negative = conservative overshoot
    rng = np.random.default_rng(0)
    S = sample_triangles(V, h.simplices, 4000, rng)
    dapp = (S @ planes[:, :3].T + planes[:, 3]).max(1)
    return planes, (V, h.simplices), float(-dapp.min())


class ShapeSet:
    """Geom-local SDFs + surface samples for a list of MuJoCo geoms (robot collision geoms or object geoms)."""

    def __init__(self, m, gids, device='cuda', n_samples=256, n_dirs=256, n_faces=128, seed=0):
        self.m = m
        self.gids = np.asarray(gids, np.int64)
        G = len(self.gids)
        self.dev = torch.device(device)
        typ = np.zeros(G, np.int64); prm = np.zeros((G, 3))
        planes, samples, gaps, boxes = [], [], [], []
        rng = np.random.default_rng(seed)
        for i, g in enumerate(self.gids):
            t = int(m.geom_type[g]); s = m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                typ[i] = SPHERE; prm[i] = [s[0], 0, 0]
                P = fibonacci_dirs(4 * n_samples) * s[0]; pl = None; gap = 0.0
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                typ[i] = CAPSULE; prm[i] = [s[0], s[1], 0]
                d = fibonacci_dirs(4 * n_samples) * s[0]
                P = np.concatenate([d + [0, 0, s[1]] * (d[:, 2:] >= 0), d - [0, 0, s[1]] * (d[:, 2:] < 0)])
                ang = rng.random(2 * n_samples) * 2 * np.pi; z = rng.uniform(-s[1], s[1], 2 * n_samples)
                P = np.concatenate([P, np.stack([s[0] * np.cos(ang), s[0] * np.sin(ang), z], 1)])
                pl = None; gap = 0.0
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                typ[i] = CYLINDER; prm[i] = [s[0], s[1], 0]
                n = 6 * n_samples
                ang = rng.random(n) * 2 * np.pi; z = rng.uniform(-s[1], s[1], n)
                side = np.stack([s[0] * np.cos(ang), s[0] * np.sin(ang), z], 1)
                rr = s[0] * np.sqrt(rng.random(n)); a2 = rng.random(n) * 2 * np.pi
                caps = np.stack([rr * np.cos(a2), rr * np.sin(a2), np.where(rng.random(n) < .5, s[1], -s[1])], 1)
                a3 = np.linspace(0, 2 * np.pi, 64, endpoint=False)
                rims = np.concatenate([np.stack([s[0] * np.cos(a3), s[0] * np.sin(a3), np.full(64, zz)], 1) for zz in (s[1], -s[1])])
                P = np.concatenate([side, caps, rims]); pl = None; gap = 0.0
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                typ[i] = BOX; prm[i] = s[:3]
                n = 6 * n_samples
                P = rng.uniform(-1, 1, (n, 3)) * s[:3]
                ax = rng.integers(3, size=n); sg = np.where(rng.random(n) < .5, -1.0, 1.0)
                P[np.arange(n), ax] = sg * s[ax]
                corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * s[:3]
                P = np.concatenate([corners, P]); pl = None; gap = 0.0
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                typ[i] = POLY
                V, _ = mesh_verts(m, g)
                pl, (HV, HF), gap = polytope_planes(V, n_dirs, n_faces)
                P = np.concatenate([sample_triangles(HV, HF, 8 * n_samples, rng), HV[np.unique(HF)]])
            elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                raise NotImplementedError('ellipsoid collision geom')
            else:
                raise NotImplementedError(f'geom type {t}')
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                e = np.array([s[0]] * 3)
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                e = np.array([s[0], s[0], s[1] + s[0]])
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                e = np.array([s[0], s[0], s[1]])
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                e = np.asarray(s[:3], np.float64)
            else:
                e = None
            if e is not None:
                boxes.append((np.zeros(3), e))
            else:
                boxes.append((0.5 * (V.max(0) + V.min(0)), 0.5 * (V.max(0) - V.min(0))))
            sel = fps(P, n_samples, seed)
            # keep extreme points (support in +-xyz) so plane contact is never missed
            ext = np.unique(np.concatenate([np.argmax(P, 0), np.argmin(P, 0)]))
            sel = np.unique(np.concatenate([sel, ext]))[:n_samples + 6]
            samples.append(P[sel]); planes.append(pl); gaps.append(gap)
        Ns = max(len(s) for s in samples)
        smp = np.zeros((G, Ns, 3)); sval = np.zeros((G, Ns), bool)
        for i, s in enumerate(samples):
            smp[i, :len(s)] = s; sval[i, :len(s)] = True
            smp[i, len(s):] = s[0]                      # duplicate (harmless for min)
        Pm = max([len(p) for p in planes if p is not None] + [1])
        pls = np.zeros((G, Pm, 4)); pls[..., 3] = -1e9
        for i, p in enumerate(planes):
            if p is not None:
                pls[i, :len(p)] = p
        f = dict(device=self.dev, dtype=torch.float32)
        self.type = torch.as_tensor(typ, device=self.dev)
        self.prm = torch.as_tensor(prm, **f)
        self.samples = torch.as_tensor(smp, **f)
        self.planes = torch.as_tensor(pls, **f)
        self.n_coarse = 64 + 32
        self.gap = np.asarray(gaps)
        self.rbound = torch.as_tensor(m.geom_rbound[self.gids], **f)
        self.box_c = torch.as_tensor(np.stack([b[0] for b in boxes]), **f)
        self.box_e = torch.as_tensor(np.stack([b[1] for b in boxes]), **f)
        self.has_poly = bool((typ == POLY).any())

    def sdf(self, gi, p, coarse=False):
        """gi [E] (index into this set), p [E,N,3] geom-local -> (d [E,N], grad [E,N,3] unit).
        coarse=True: polytopes use only their coarse plane subset (a lower bound of the fine sdf)."""
        t = self.type[gi][:, None]
        prm = self.prm[gi][:, None, :]
        r = prm[..., 0]; hh = prm[..., 1]
        eps = 1e-12
        # sphere
        nrm = p.norm(dim=-1)
        d = torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
        gr = torch.zeros_like(p)
        m = t == SPHERE
        if bool(m.any()):
            d = torch.where(m, nrm - r, d)
            gr = torch.where(m[..., None], p / nrm.clamp_min(eps)[..., None], gr)
        m = t == CAPSULE
        if bool(m.any()):
            c = torch.stack([torch.zeros_like(r.expand_as(nrm)), torch.zeros_like(nrm), torch.maximum(torch.minimum(p[..., 2], hh), -hh)], -1)
            v = p - c; vn = v.norm(dim=-1)
            d = torch.where(m, vn - r, d)
            gr = torch.where(m[..., None], v / vn.clamp_min(eps)[..., None], gr)
        m = t == CYLINDER
        if bool(m.any()):
            rho = p[..., :2].norm(dim=-1)
            dx = rho - r; dz = p[..., 2].abs() - hh
            out = torch.sqrt(dx.clamp_min(0) ** 2 + dz.clamp_min(0) ** 2)
            ins = torch.minimum(torch.maximum(dx, dz), torch.zeros_like(dx))
            radial = torch.cat([p[..., :2] / rho.clamp_min(eps)[..., None], torch.zeros_like(rho)[..., None]], -1)
            axial = torch.stack([torch.zeros_like(rho), torch.zeros_like(rho), torch.sign(p[..., 2])], -1)
            go = radial * dx.clamp_min(0)[..., None] + axial * dz.clamp_min(0)[..., None]
            go = go / go.norm(dim=-1, keepdim=True).clamp_min(eps)
            gi_ = torch.where((dx > dz)[..., None], radial, axial)
            d = torch.where(m, out + ins, d)
            gr = torch.where(m[..., None], torch.where((out > 0)[..., None], go, gi_), gr)
        m = t == BOX
        if bool(m.any()):
            a = prm.expand_as(p)
            o = p.abs() - a
            out = o.clamp_min(0).norm(dim=-1)
            mx, am = o.max(-1)
            ins = mx.clamp_max(0)
            go = torch.sign(p) * o.clamp_min(0); go = go / go.norm(dim=-1, keepdim=True).clamp_min(eps)
            gi_ = torch.zeros_like(p).scatter(-1, am[..., None], 1.0) * torch.sign(p)
            d = torch.where(m, out + ins, d)
            gr = torch.where(m[..., None], torch.where((out > 0)[..., None], go, gi_), gr)
        m = t == POLY
        if bool(m.any()):
            pl = self.planes[gi][:, :self.n_coarse] if coarse else self.planes[gi]     # [E,P,4]
            s = p @ pl[..., :3].transpose(1, 2) + pl[..., 3][:, None, :]   # [E,N,P]
            mx, am = s.max(-1)
            n = torch.gather(pl[..., :3], 1, am[..., None].expand(-1, -1, 3))
            d = torch.where(m, mx, d)
            gr = torch.where(m[..., None], n, gr)
        return d, gr


def mujoco_collision_pairs(m, gids):
    """Geom pairs among gids that MuJoCo's collision filter admits (contype/conaffinity, weld body, parent-child,
    explicit <exclude>)."""
    filterparent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    excl = set()
    for i in range(m.nexclude):
        sig = int(m.exclude_signature[i])
        excl.add((sig >> 16, sig & 0xFFFF))
    weld = m.body_weldid; par = m.body_parentid
    out = []
    for i, g1 in enumerate(gids):
        for g2 in gids[i + 1:]:
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            w1, w2 = int(weld[b1]), int(weld[b2])
            if w1 == w2:
                continue
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            if filterparent and w1 and w2 and (w1 == int(weld[par[w2]]) or w2 == int(weld[par[w1]])):
                continue
            if (min(b1, b2), max(b1, b2)) in excl:
                continue
            out.append((int(g1), int(g2)))
    return out


def robot_collision_geoms(m):
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == 0 or int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        name = ((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '') + ' ' +
                (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '')).lower()
        if 'floor' in name or 'ground' in name:
            continue
        if int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
            continue
        out.append(g)
    return np.asarray(out, np.int64)


def jac_entries(kt, K, bidx, P, body):
    """Point Jacobians for scattered entries: bidx [E] batch ids, P [E,3] world, body [E] -> [E,3,nv]."""
    axis = K['axis'][bidx]; anchor = K['anchor'][bidx]           # [E,nv,3]
    r = P[:, None, :].to(axis.dtype) - anchor
    cr = torch.linalg.cross(axis, r, dim=-1)
    J = torch.where(kt.dof_rot[None, :, None] > 0, cr, axis) * kt.anc[body][..., None]
    return J.transpose(1, 2)


class SelfCollision:
    def __init__(self, kt, device='cuda', n_samples=int(__import__('os').environ.get('UMRG_SELF_SAMPLES', 160)), extra_exclude=(), n_dirs=1024, n_faces=512,
                 refine_steps=(0.004, 0.002, 0.001, 0.0005, 0.00025, 0.0001), use_warp=True):
        m = kt.m
        self.kt = kt
        gids = robot_collision_geoms(m)
        self.shapes = ShapeSet(m, gids, device, n_samples=n_samples, n_dirs=n_dirs, n_faces=n_faces)
        self.refine_steps = refine_steps
        self.topk = 16
        self.warp = None
        if use_warp:
            from surface_retargeting.gpu.narrow_warp import WarpNarrow
            self.warp = WarpNarrow(self.shapes, refine_steps)
        pos = {int(g): i for i, g in enumerate(gids)}
        pairs = [p for p in mujoco_collision_pairs(m, gids) if p not in set(extra_exclude)]
        self.pairs_geom = np.asarray(pairs, np.int64).reshape(-1, 2)
        self.pa = torch.as_tensor([pos[a] for a, _ in pairs], device=device)
        self.pb = torch.as_tensor([pos[b] for _, b in pairs], device=device)
        self.gids_t = torch.as_tensor(gids, device=device)
        self.body = kt.geom_body[self.gids_t]
        # geom-local OBB (AABB in the geom frame) of the collision shape, for the separating-axis broadphase
        self.obb_c = self.shapes.box_c; self.obb_e = self.shapes.box_e + 1e-4     # exact local AABB of the shape
        self.use_obb = True

    def obb_overlap(self, gp, gR, act):
        """Separating-axis test of the pair OBBs inflated by act -> mask [B,P] of pairs that may be within act."""
        pa, pb = self.pa, self.pb
        Ra, Rb = gR[:, pa], gR[:, pb]                               # [B,P,3,3] columns = box axes
        ca = (Ra @ self.obb_c[pa][..., None])[..., 0] + gp[:, pa]
        cb = (Rb @ self.obb_c[pb][..., None])[..., 0] + gp[:, pb]
        ea, eb = self.obb_e[pa], self.obb_e[pb]
        Tw = cb - ca
        C = Ra.transpose(-1, -2) @ Rb                              # [B,P,3,3]
        Ta = (Ra.transpose(-1, -2) @ Tw[..., None])[..., 0]        # T in A frame
        absC = C.abs() + 1e-6
        sep = torch.zeros(Tw.shape[:-1], dtype=torch.bool, device=Tw.device)
        # A axes
        ra = ea; rb = (absC * eb[..., None, :]).sum(-1)
        sep |= ((Ta.abs() - ra - rb) > act).any(-1)
        # B axes
        Tb = (C.transpose(-1, -2) @ Ta[..., None])[..., 0]
        ra = (absC.transpose(-1, -2) * ea[..., None, :]).sum(-1); rb = eb
        sep |= ((Tb.abs() - ra - rb) > act).any(-1)
        # cross axes A_i x B_j
        for i in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            for j in range(3):
                j1, j2 = (j + 1) % 3, (j + 2) % 3
                t = (Ta[..., i2] * C[..., i1, j] - Ta[..., i1] * C[..., i2, j]).abs()
                ra = ea[:, i1] * absC[..., i2, j] + ea[:, i2] * absC[..., i1, j]
                rb = eb[:, j1] * absC[..., i, j2] + eb[:, j2] * absC[..., i, j1]
                L = torch.sqrt((C[..., i1, j] ** 2 + C[..., i2, j] ** 2).clamp_min(1e-12))     # |A_i x B_j|
                sep |= (t - ra - rb) > act * L
        return ~sep

    def query(self, K, act, chunk=1024):
        """All (batch, pair) within `act` (bounding spheres), exact-ish distance via point-vs-SDF both ways.
        Returns dict of entries: b [E], pair [E], d [E], p (world point on the penetrating/closest geom) [E,3],
        q (closest point on the other geom) [E,3], n (world normal, other geom outward) [E,3], body_p, body_q."""
        gp, gR = self.kt.geom_pose(K, self.gids_t)                 # [B,G,3], [B,G,3,3]
        ca, cb = gp[:, self.pa], gp[:, self.pb]
        rb = self.shapes.rbound
        broad = (ca - cb).norm(dim=-1) - rb[self.pa] - rb[self.pb] < act
        if self.use_obb:
            broad = broad & self.obb_overlap(gp, gR, act)
        bi, pi = torch.nonzero(broad, as_tuple=True)
        E = bi.numel()
        res = dict(b=bi, pair=pi)
        if E == 0:
            z = torch.zeros(0, device=gp.device)
            res.update(d=z, p=z.reshape(0, 3), q=z.reshape(0, 3), n=z.reshape(0, 3),
                       body_p=bi, body_q=bi)
            return res
        if self.warp is not None:
            ga, gb = self.pa[pi], self.pb[pi]
            od, op, on = self.warp(torch.cat([bi, bi]), torch.cat([ga, gb]), torch.cat([gb, ga]), gp, gR)
            take2 = od[E:] < od[:E]
            sel = lambda x: torch.where(take2.reshape(-1, *([1] * (x.dim() - 1))), x[E:], x[:E])
            d = sel(od); p = sel(op); n = sel(on)
            bp = torch.where(take2, self.body[gb], self.body[ga]); bq = torch.where(take2, self.body[ga], self.body[gb])
            res.update(d=d, p=p, q=p - d[:, None] * n, n=n, body_p=bp, body_q=bq)
            return res
        out_d, out_p, out_q, out_n, out_bp, out_bq = [], [], [], [], [], []
        for s in range(0, E, chunk):
            b = bi[s:s + chunk]; p = pi[s:s + chunk]
            ga, gb = self.pa[p], self.pb[p]
            best = None
            ar = torch.arange(len(b), device=gp.device)
            for (g1, g2) in ((ga, gb), (gb, ga)):              # deepest point of g1 inside sdf of g2
                R1, x1 = gR[b, g1], gp[b, g1]
                R2, x2 = gR[b, g2], gp[b, g2]
                pw = (R1[:, None] @ self.shapes.samples[g1][..., None])[..., 0] + x1[:, None]
                pl = ((pw - x2[:, None])[..., None, :] @ R2[:, None])[..., 0, :]     # R2^T (pw - x2)
                dc, _ = self.shapes.sdf(g2, pl, coarse=True)
                kk = min(self.topk, dc.shape[1])
                _, ci = torch.topk(dc, kk, dim=1, largest=False)
                plk = torch.gather(pl, 1, ci[..., None].expand(-1, -1, 3))
                pwk = torch.gather(pw, 1, ci[..., None].expand(-1, -1, 3))
                d, gr = self.shapes.sdf(g2, plk)
                dm, am = d.min(-1)
                pw = pwk
                # projected-gradient refinement of the best sample: min sdf_2(p) s.t. p in geom 1 (convex problem)
                pc = pw[ar, am]
                for step in self.refine_steps:
                    l2 = ((pc - x2)[:, None, :] @ R2)[:, 0, :]
                    d2, n2 = self.shapes.sdf(g2, l2[:, None])
                    pn = pc - step * (R2 @ n2[:, 0, :, None])[..., 0]
                    l1 = ((pn - x1)[:, None, :] @ R1)[:, 0, :]
                    d1, n1 = self.shapes.sdf(g1, l1[:, None])
                    l1 = l1 - d1[:, 0, None].clamp_min(0) * n1[:, 0]
                    pc = (R1 @ l1[..., None])[..., 0] + x1
                l2 = ((pc - x2)[:, None, :] @ R2)[:, 0, :]
                d2, n2 = self.shapes.sdf(g2, l2[:, None])
                use_ref = d2[:, 0] < dm
                dm = torch.where(use_ref, d2[:, 0], dm)
                pwm = torch.where(use_ref[:, None], pc, pw[ar, am])
                nl = torch.where(use_ref[:, None], n2[:, 0], gr[ar, am])
                nw = (R2 @ nl[..., None])[..., 0]
                cand = (dm, pwm, pwm - dm[:, None] * nw, nw, self.body[g1], self.body[g2])
                if best is None:
                    best = cand
                else:
                    take = cand[0] < best[0]
                    best = tuple(torch.where(take.reshape(-1, *([1] * (c.dim() - 1))), c, o) for c, o in zip(cand, best))
            out_d.append(best[0]); out_p.append(best[1]); out_q.append(best[2]); out_n.append(best[3])
            out_bp.append(best[4]); out_bq.append(best[5])
        res.update(d=torch.cat(out_d), p=torch.cat(out_p), q=torch.cat(out_q), n=torch.cat(out_n),
                   body_p=torch.cat(out_bp), body_q=torch.cat(out_bq))
        return res

    def min_dist(self, K, act=0.02):
        """Per batch element min pair distance (inf if no pair within act)."""
        r = self.query(K, act)
        B = K['xpos'].shape[0]
        out = torch.full((B,), float('inf'), device=K['xpos'].device)
        if r['b'].numel():
            out = out.scatter_reduce(0, r['b'], r['d'].to(out.dtype), reduce='amin')
        return out, r


# ---------------------------------------------------------------------- objects
class ObjectSDF:
    """Rigid object in the retarget frame: analytic sphere / box if the mesh is one (factory rule: radial or OBB face
    residual < 1.5 mm), else a dense SDF grid (trimesh closest point + containment) with trilinear lookup.
    Mesh is given in the object BODY frame; per-frame body pose (pos, quat wxyz) is supplied at query time."""

    def __init__(self, V, F, device='cuda', tol=0.0015, grid=96, pad=0.06):
        import trimesh
        self.dev = torch.device(device)
        V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
        mesh = trimesh.Trimesh(V, F, process=True)
        self.mesh = mesh
        v = np.asarray(mesh.vertices)
        c = v.mean(0); r = np.linalg.norm(v - c, axis=1)
        self.kind = 'mesh'
        f = dict(device=self.dev, dtype=torch.float32)
        if r.std() < tol and (r.max() - r.min()) < 4 * tol:
            self.kind = 'sphere'; self.center = torch.as_tensor(c, **f); self.radius = float(r.mean())
        else:
            obb = mesh.bounding_box_oriented
            T = np.asarray(obb.primitive.transform); ext = np.asarray(obb.primitive.extents) / 2
            q = (v - T[:3, 3]) @ T[:3, :3]
            if np.min(np.abs(np.abs(q) - ext[None]), axis=1).max() < tol:
                self.kind = 'box'; self.boxR = torch.as_tensor(T[:3, :3], **f); self.boxc = torch.as_tensor(T[:3, 3], **f)
                self.half = torch.as_tensor(ext, **f)
            else:
                lo = v.min(0) - pad; hi = v.max(0) + pad
                n = grid
                ax = [np.linspace(lo[i], hi[i], n) for i in range(3)]
                X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
                pq = trimesh.proximity.ProximityQuery(mesh)
                _, dist, _ = pq.on_surface(X)
                inside = mesh.contains(X) if mesh.is_watertight else np.zeros(len(X), bool)
                sd = np.where(inside, -dist, dist).reshape(n, n, n)
                self.grid = torch.as_tensor(sd, **f)[None, None]            # [1,1,n,n,n] (x,y,z)
                self.glo = torch.as_tensor(lo, **f); self.ghi = torch.as_tensor(hi, **f)
        self.bound_c = torch.as_tensor(v.mean(0), **f)
        self.bound_r = float(np.linalg.norm(v - v.mean(0), axis=1).max())

    def sdf(self, Pw, pose):
        """Pw [B,N,3] world, pose [B,7] (pos, quat wxyz of the object body) or [B,8] (+ uniform object scale s:
        the object is s x this asset) -> d [B,N], n_world [B,N,3]."""
        from surface_retargeting.gpu.kin import quat_to_mat
        R = quat_to_mat(pose[:, 3:7].float()); x = pose[:, :3].float()
        pl = ((Pw.float() - x[:, None])[..., None, :] @ R[:, None])[..., 0, :]
        sc = pose[:, 7].float()[:, None] if pose.shape[-1] > 7 else None
        if sc is not None:
            pl = pl / sc[..., None]
        eps = 1e-12
        if self.kind == 'sphere':
            v = pl - self.center; r = v.norm(dim=-1)
            d = r - self.radius; n = v / r.clamp_min(eps)[..., None]
        elif self.kind == 'box':
            p = (pl - self.boxc) @ self.boxR
            o = p.abs() - self.half
            out = o.clamp_min(0).norm(dim=-1); mx, am = o.max(-1)
            d = out + mx.clamp_max(0)
            go = torch.sign(p) * o.clamp_min(0); go = go / go.norm(dim=-1, keepdim=True).clamp_min(eps)
            gi = torch.zeros_like(p).scatter(-1, am[..., None], 1.0) * torch.sign(p)
            n = torch.where((out > 0)[..., None], go, gi) @ self.boxR.T
        else:
            u = (pl - self.glo) / (self.ghi - self.glo) * 2 - 1                  # [-1,1]
            uu = u.clamp(-1, 1)
            # grid_sample expects (x->W, y->H, z->D) order reversed: grid [N,D,H,W,3] with (x=W)
            g = uu[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)
            d = torch.nn.functional.grid_sample(self.grid, g, align_corners=True).reshape(pl.shape[:-1])
            outside = (u - uu).norm(dim=-1) * 0.5 * (self.ghi - self.glo).max()
            d = d + outside
            h = 1e-3
            n = torch.zeros_like(pl)
            for k in range(3):
                e = torch.zeros(3, device=pl.device); e[k] = h
                up = ((pl + e - self.glo) / (self.ghi - self.glo) * 2 - 1).clamp(-1, 1)[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)
                dp = torch.nn.functional.grid_sample(self.grid, up, align_corners=True).reshape(pl.shape[:-1])
                n[..., k] = (dp - d) / h
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(eps)
        nw = (R[:, None] @ n[..., None])[..., 0]
        if sc is not None:
            d = d * sc
        return d, nw


def object_from_objpen(mjb_path, object_geoms):
    """Object mesh (body frame) from UMR's robot+object penetration model (object geoms appended to the robot)."""
    m = mujoco.MjModel.from_binary_path(mjb_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    V, F, off = [], [], 0
    body = int(m.geom_bodyid[int(object_geoms[0])])
    for g in object_geoms:
        g = int(g)
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        v, f = mesh_verts(m, g)
        # geom frame -> body frame
        from scipy.spatial.transform import Rotation as Rot
        q = m.geom_quat[g]; Rg = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        V.append(v @ Rg.T + m.geom_pos[g]); F.append(f + off); off += len(v)
    return np.concatenate(V), np.concatenate(F), body


class UMRGeometry:
    """Extra UMR stage-A rows needing collision geometry: robot-object penetration (UMR robot_object_hard_constraint,
    mj_geomDistance per (robot collision geom, object geom) pair -> here min over the robot geom's surface samples
    of the object SDF; one row per robot geom, the 128 closest within the activation distance)."""

    def __init__(self, solver_like_kt, bank, device='cuda', n_samples=192):
        kt = solver_like_kt
        self.kt = kt
        self.bank = bank
        a = bank.args
        self.obj = None
        P0 = bank.problems[0]
        if bank.has_obj and 'objpen_mjb' in P0.z.files and (bool(a.get('robot_object_hard_constraint')) or
                                                           float(a.get('robot_object_penetration_soft_cost', 0)) > 0):
            V, F, _ = object_from_objpen(str(P0['objpen_mjb']), P0['objpen_object_geoms'])
            self.obj = ObjectSDF(V, F, device)
            gids = np.asarray(P0['objpen_robot_geoms'], np.int64)
            self.shapes = ShapeSet(kt.m, gids, device, n_samples=n_samples)
            self.gids_t = torch.as_tensor(gids, device=device)
            self.body = kt.geom_body[self.gids_t]

    @classmethod
    def from_sdf(cls, kt, bank, obj_sdf, device='cuda', n_samples=192):
        """Robot-object rows for an arbitrary (per-row scaled) object SDF; robot geoms = UMR's
        robot_object_collision_geom_ids (collision geoms, no planes / floor)."""
        self = cls.__new__(cls)
        self.kt = kt; self.bank = bank; self.obj = obj_sdf
        gids = robot_collision_geoms(kt.m)
        self.shapes = ShapeSet(kt.m, gids, device, n_samples=n_samples)
        self.gids_t = torch.as_tensor(gids, device=device)
        self.body = kt.geom_body[self.gids_t]
        return self

    def umr_rows(self, sv, K, q, inp, st):
        a = sv.a
        if self.obj is None or 'obj_pose' not in inp:
            return []
        kt = self.kt
        gp, gR = kt.geom_pose(K, self.gids_t)
        B, G = gp.shape[:2]
        S = self.shapes.samples                                             # [G,Ns,3]
        pw = (gR[:, :, None] @ S[None, ..., None])[..., 0] + gp[:, :, None]  # [B,G,Ns,3]
        d, n = self.obj.sdf(pw.reshape(B, -1, 3), inp['obj_pose'])
        d = d.reshape(B, G, -1); n = n.reshape(B, G, -1, 3)
        dm, am = d.min(-1)
        ar = torch.arange(B, device=d.device)[:, None]; ag = torch.arange(G, device=d.device)[None]
        pm = pw[ar, ag, am]; nm = n[ar, ag, am]
        margin = float(a['robot_object_margin']); thr = float(a['robot_object_threshold'])
        act = dm <= max(thr, margin, 0.0)
        k = int(a['robot_object_max_pairs'])
        if 0 < k < G:
            act = act & topk_mask_static(dm, act, k)
        J = kt.jac(K, pm, self.body)                                        # [B,G,3,nv]
        rows = -(nm[..., None, :].to(J.dtype) @ J)[..., 0, :]
        Grow = torch.where(act[..., None], rows, torch.zeros_like(rows)).to(torch.float64)
        h = torch.where(act, dm.to(torch.float64) - margin, torch.full_like(dm, float('inf'), dtype=torch.float64))
        cost = float(a['robot_object_hard_slack_cost']) if bool(a['robot_object_hard_slack']) else 0.0
        return [(Grow, h, cost)]


def topk_mask_static(key, valid, k):
    big = torch.where(valid, key, torch.full_like(key, float('inf')))
    _, idx = torch.topk(big, k, dim=1, largest=False)
    m = torch.zeros_like(valid); m.scatter_(1, idx, True)
    return m


class CollisionGround:
    """Lowest point of every robot COLLISION geom w.r.t. the floor (MuJoCo collision geometry cross-check of the
    ground gauge): analytic for sphere / capsule / cylinder / box, support vertices (256 directions of the convex
    hull) for meshes.  Returns per geom clearance and the witness point (world) for constraint rows."""

    def __init__(self, kt, device='cuda', n_dirs=None, covered=()):
        """covered: geom ids whose exact hull vertices are already in the point gauge (collision mesh == visual
        mesh on the same body/pose); they are skipped here."""
        from scipy.spatial import ConvexHull
        import os
        n_dirs = n_dirs or int(os.environ.get('UMRG_COLGROUND_DIRS', 8192))
        m = kt.m
        self.kt = kt
        cov = set(int(g) for g in covered)
        gids = np.asarray([g for g in robot_collision_geoms(m) if not (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and int(g) in cov)],
                          np.int64)
        self.n = len(gids)
        if self.n == 0:
            return
        self.gids = torch.as_tensor(gids, device=device)
        typ, prm, verts = [], [], []
        for g in gids:
            t = int(m.geom_type[g]); s = m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                V, _ = mesh_verts(m, g)
                HV = V[ConvexHull(V).vertices]
                d = fibonacci_dirs(n_dirs)
                sup = np.unique(np.argmax(HV @ d.T, axis=0))
                verts.append(HV[sup]); typ.append(POLY); prm.append([0, 0, 0])
            else:
                verts.append(np.zeros((1, 3)))
                typ.append({mujoco.mjtGeom.mjGEOM_SPHERE: SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE: CAPSULE,
                            mujoco.mjtGeom.mjGEOM_CYLINDER: CYLINDER, mujoco.mjtGeom.mjGEOM_BOX: BOX}[t])
                prm.append(list(s[:3]))
        Vm = max(len(v) for v in verts)
        Vp = np.zeros((len(gids), Vm, 3)); Vv = np.zeros((len(gids), Vm), bool)
        for i, v in enumerate(verts):
            Vp[i, :len(v)] = v; Vp[i, len(v):] = v[0]; Vv[i, :len(v)] = True
        f = dict(device=device, dtype=torch.float32)
        self.typ = torch.as_tensor(typ, device=device); self.prm = torch.as_tensor(np.asarray(prm), **f)
        self.V = torch.as_tensor(Vp, **f)
        self.body = kt.geom_body[self.gids]
        self.is_mesh = self.typ == POLY

    def lowest(self, K, floor):
        """-> clearance [B,G], witness world point [B,G,3]."""
        if self.n == 0:
            B = K['xpos'].shape[0]
            return torch.full((B, 0), float('inf'), device=K['xpos'].device), torch.zeros(B, 0, 3, device=K['xpos'].device)
        gp, gR = self.kt.geom_pose(K, self.gids)                       # [B,G,3], [B,G,3,3]
        gp = gp.float(); gR = gR.float()
        B, G = gp.shape[:2]
        ez = gR[..., 2, :]                                             # world z of each local axis: R[2,:]
        r = self.prm[:, 0]; h = self.prm[:, 1]
        # analytic lowest points (local coordinates)
        sgn = -torch.sign(ez); sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        box = sgn * self.prm                                           # box corner minimising z
        az = ez[..., 2]                                                # local z-axis world-z component
        cap = torch.zeros(B, G, 3, device=gp.device)
        cap[..., 2] = torch.where(az > 0, -h, h)
        down = -ez / ez.norm(dim=-1, keepdim=True).clamp_min(1e-12)    # world -z expressed in local frame
        sph = down * r[None, :, None]
        capsule = cap + sph
        radial = down.clone(); radial[..., 2] = 0
        radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-12) * r[None, :, None]
        cyl = cap + radial
        t = self.typ[None, :, None]
        loc = torch.where(t == SPHERE, sph, torch.where(t == CAPSULE, capsule, torch.where(t == CYLINDER, cyl, box)))
        # meshes: support vertices
        zv = (self.V[None] @ ez[..., None])[..., 0]                    # [B,G,Vm] local->world z offsets
        vi = zv.argmin(-1)
        vm = self.V[torch.arange(G, device=gp.device)[None].expand(B, -1), vi]
        loc = torch.where(self.is_mesh[None, :, None], vm, loc)
        P = (gR @ loc[..., None])[..., 0] + gp
        return P[..., 2] - floor[:, None].float(), P
