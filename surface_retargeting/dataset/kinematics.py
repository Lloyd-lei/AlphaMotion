"""Robot static spec (quantisation ranges, joint names, asset hashes) + a batched torch forward kinematics that
reproduces MuJoCo's mj_kinematics for the staged UMR-factory MJCFs (free root + hinge/slide joints), including the
4096-point dense surface and the convex-hull ground gauge.  Lets the trainer recompute per-point / per-region
clearance on the GPU instead of storing it (see README "what is dropped").

Export (needs mujoco, run once per robot):  build_spec(robot) -> <root>/robots/<robot>/{spec.json, kin.npz}
Use (numpy + torch only):                   fk = TorchFK.load(<root>/robots/<robot>/kin.npz, device='cuda')
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import hashlib
import json
from pathlib import Path

import numpy as np

STAGED = Path(f'{SRCFG.data_root}/robots')
JOINT_MARGIN = 1e-3            # rad (m for slides) added outside the MJCF limits
UNLIMITED_HALF_RANGE = 2 * np.pi


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_spec(robot: str, root: Path, staged: Path = STAGED) -> dict:
    import mujoco
    st = json.load(open(staged / robot / 'staging.json'))
    xml = Path(st['staged_mjcf'])
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_qposadr[0] == 0, robot
    names, lo, hi, limited = [], [], [], []
    for j in range(1, m.njnt):
        t = m.jnt_type[j]
        assert t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE), (robot, j, t)
        assert m.jnt_qposadr[j] == 7 + len(names), (robot, j)
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        if m.jnt_limited[j]:
            lo.append(m.jnt_range[j, 0] - JOINT_MARGIN); hi.append(m.jnt_range[j, 1] + JOINT_MARGIN); limited.append(True)
        else:
            lo.append(-UNLIMITED_HALF_RANGE); hi.append(UNLIMITED_HALF_RANGE); limited.append(False)
    lo, hi = np.array(lo), np.array(hi)
    step = (hi - lo) / 65535.0
    surf = staged / robot / 'surface.npz'
    spec = dict(robot=robot, nq=int(m.nq), nj=len(names), free_joint=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, 0),
                joint_names=names, joint_limited=limited, q_lo=lo.tolist(), q_step=step.tolist(),
                max_quant_err_rad=float(step.max() / 2), staged_mjcf=str(xml), mjcf_sha256=_sha(xml),
                surface_npz=str(surf), surface_sha256=_sha(surf) if surf.exists() else None,
                facing_plus_x=st.get('facing_plus_x'), nbody=int(m.nbody), ngeom=int(m.ngeom))
    d = root / 'robots' / robot
    d.mkdir(parents=True, exist_ok=True)
    export_kin(m, surf, d / 'kin.npz')
    spec['kin_sha256'] = _sha(d / 'kin.npz')
    json.dump(spec, open(d / 'spec.json', 'w'), indent=1)
    return spec


def export_kin(m, surface_npz: Path, out: Path):
    """Everything TorchFK needs, as plain arrays."""
    import mujoco
    jb = -np.ones(m.nbody, np.int64)            # first non-free joint of each body (UMR staged robots: <= 1 per body)
    for j in range(1, m.njnt):
        b = m.jnt_bodyid[j]
        assert jb[b] < 0, 'more than one joint on a body is not supported'
        jb[b] = j
    arr = dict(body_parent=m.body_parentid.astype(np.int64), body_pos=m.body_pos.astype(np.float64),
               body_quat=m.body_quat.astype(np.float64), body_joint=jb,
               jnt_type=m.jnt_type.astype(np.int64), jnt_qposadr=m.jnt_qposadr.astype(np.int64),
               jnt_axis=m.jnt_axis.astype(np.float64), jnt_pos=m.jnt_pos.astype(np.float64),
               qpos0=m.qpos0.astype(np.float64), root_body=np.int64(m.jnt_bodyid[0]),
               geom_body=m.geom_bodyid.astype(np.int64), geom_pos=m.geom_pos.astype(np.float64),
               geom_quat=m.geom_quat.astype(np.float64))
    if surface_npz.exists():
        s = np.load(surface_npz)
        arr.update(pts_geom=s['geom'].astype(np.int64), pts_local=s['points_local'].astype(np.float64),
                   pts_normal=s['normals_local'].astype(np.float64), pts_region=s['region'].astype(np.int64),
                   hull_geom=s['hull_geom'].astype(np.int64), hull_local=s['hull_local'].astype(np.float64),
                   body_region=s['body_region'].astype(np.int64))
    np.savez(out, **arr)


# ---------------------------------------------------------------------------------------------------- torch FK
def _qmul(a, b):
    import torch
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _qmat(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _np_qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class TorchFK:
    """Batched MuJoCo-equivalent FK. qpos [..., nq] (wxyz root quat) -> body xpos [...,nbody,3], xmat [...,nbody,3,3]."""

    def __init__(self, kin: dict, device='cpu', dtype=None):
        import torch
        self.dtype = dtype or torch.float32
        self.device = torch.device(device)
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), device=self.device, dtype=dt or self.dtype)
        self.k = kin
        self.nbody = len(kin['body_parent'])
        self.parent = [int(p) for p in kin['body_parent']]
        self.body_joint = [int(j) for j in kin['body_joint']]
        self.root_body = int(kin['root_body'])
        self.body_pos, self.body_quat = t(kin['body_pos']), t(kin['body_quat'])
        self.jnt_axis, self.jnt_pos, self.jnt_type = t(kin['jnt_axis']), t(kin['jnt_pos']), [int(x) for x in kin['jnt_type']]
        self.jnt_qposadr = [int(x) for x in kin['jnt_qposadr']]
        self.qpos0 = t(kin['qpos0'])
        self.geom_body = torch.as_tensor(kin['geom_body'], device=self.device)
        self.geom_pos, self.geom_quat = t(kin['geom_pos']), t(kin['geom_quat'])
        if 'pts_geom' in kin:
            self.pts_geom = torch.as_tensor(kin['pts_geom'], device=self.device)
            self.pts_local, self.pts_normal = t(kin['pts_local']), t(kin['pts_normal'])
            self.pts_region = torch.as_tensor(kin['pts_region'], device=self.device)
            self.hull_geom = torch.as_tensor(kin['hull_geom'], device=self.device)
            self.hull_local = t(kin['hull_local'])

    @classmethod
    def load(cls, path, device='cpu', dtype=None):
        with np.load(path) as z:
            kin = {k: z[k] for k in z.files}
        return cls(kin, device, dtype)

    def _prep_levels(self):
        """Group bodies by tree depth so each depth is one batched rigid-transform step (few kernel launches)."""
        import torch
        if hasattr(self, '_lv'):
            return self._lv
        depth = [0] * self.nbody
        for b in range(1, self.nbody):
            depth[b] = depth[self.parent[b]] + 1
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), device=self.device, dtype=dt or self.dtype)
        bq = self.k['body_quat']
        levels = []
        for dep in range(1, max(depth) + 1):
            L = [b for b in range(1, self.nbody) if depth[b] == dep and b != self.root_body]
            if not L:
                continue
            P = [self.parent[b] for b in L]
            adr, hj, sl, K, ax, jp = [], [], [], [], [], []
            for b in L:
                j = self.body_joint[b]
                if j >= 0:
                    a = self.k['jnt_axis'][j]
                    adr.append(self.jnt_qposadr[j]); hj.append(self.jnt_type[j] == 3); sl.append(self.jnt_type[j] == 2)
                    ax.append(a); jp.append(self.k['jnt_pos'][j])
                    K.append([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                else:
                    adr.append(0); hj.append(False); sl.append(False); ax.append([0, 0, 0]); jp.append([0, 0, 0])
                    K.append(np.zeros((3, 3)))
            K = np.asarray(K, np.float64)
            Rq = np.stack([_np_qmat(bq[b]) for b in L])
            levels.append(dict(L=torch.as_tensor(L, device=self.device), P=torch.as_tensor(P, device=self.device),
                               bpos=t(self.k['body_pos'][L]), Rq=t(Rq), adr=torch.as_tensor(adr, device=self.device),
                               hj=t(hj), sl=t(sl), K=t(K), K2=t(K @ K), ax=t(ax), jp=t(jp),
                               q0=t(self.k['qpos0'][adr])))
        self._lv = levels
        return levels

    def bodies(self, qpos):
        """Body world poses: xpos [...,nbody,3], xmat [...,nbody,3,3] (MuJoCo mj_kinematics semantics)."""
        import torch
        qpos = qpos.to(self.device, self.dtype)
        sh = qpos.shape[:-1]
        qf = qpos.reshape(-1, qpos.shape[-1]); B = qf.shape[0]
        X = torch.zeros(B, self.nbody, 3, device=self.device, dtype=self.dtype)
        R = torch.eye(3, device=self.device, dtype=self.dtype).repeat(B, self.nbody, 1, 1)
        X[:, self.root_body] = qf[:, 0:3]
        R[:, self.root_body] = _qmat(qf[:, 3:7] / qf[:, 3:7].norm(dim=-1, keepdim=True))
        for lv in self._prep_levels():
            Rp, Xp = R[:, lv['P']], X[:, lv['P']]                              # [B,n,3,3], [B,n,3]
            pos = Xp + (Rp @ lv['bpos'].unsqueeze(-1)).squeeze(-1)
            R0 = Rp @ lv['Rq']
            th = qf[:, lv['adr']] - lv['q0']                                    # [B,n] (0 for fixed bodies via masks)
            s = torch.sin(th * lv['hj'])[..., None, None]; c = torch.cos(th * lv['hj'])[..., None, None]
            Rj = torch.eye(3, device=self.device, dtype=self.dtype) + s * lv['K'] + (1 - c) * lv['K2']
            Rb = R0 @ Rj
            jp = lv['jp'].unsqueeze(-1)
            pos = pos + (R0 @ jp).squeeze(-1) - (Rb @ jp).squeeze(-1)           # rotate about the joint anchor
            pos = pos + (R0 @ lv['ax'].unsqueeze(-1)).squeeze(-1) * (th * lv['sl']).unsqueeze(-1)   # slide joints
            X[:, lv['L']] = pos
            R[:, lv['L']] = Rb
        return X.reshape(sh + (self.nbody, 3)), R.reshape(sh + (self.nbody, 3, 3))

    def geoms(self, qpos):
        X, R = self.bodies(qpos)
        Rb, Xb = R[..., self.geom_body, :, :], X[..., self.geom_body, :]
        gx = Xb + (Rb @ self.geom_pos.unsqueeze(-1)).squeeze(-1)
        if not hasattr(self, '_gRq'):
            self._gRq = _qmat(self.geom_quat)
        gR = Rb @ self._gRq
        return gx, gR

    def surface(self, qpos, normals=False):
        """Dense surface points [...,4096,3] (and world normals)."""
        gx, gR = self.geoms(qpos)
        R = gR[..., self.pts_geom, :, :]
        P = gx[..., self.pts_geom, :] + (R @ self.pts_local.unsqueeze(-1)).squeeze(-1)
        if normals:
            return P, (R @ self.pts_normal.unsqueeze(-1)).squeeze(-1)
        return P

    def hull(self, qpos):
        gx, gR = self.geoms(qpos)
        return gx[..., self.hull_geom, :] + (gR[..., self.hull_geom, :, :] @ self.hull_local.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------ cheap gauges used by the converter guard
    def _prep_light(self, nreg=13):
        import torch
        if hasattr(self, '_light'):
            return self._light
        hreg = torch.as_tensor(self.k['body_region'], device=self.device)[self.geom_body[self.hull_geom]]
        g_all = torch.cat([self.pts_geom, self.hull_geom])
        l_all = torch.cat([self.pts_local, self.hull_local])
        r_all = torch.cat([self.pts_region, hreg])
        ng = len(self.geom_pos)
        # per-geom bounding sphere of every gauge point (geom-local): centre = mean, radius = max distance
        cnt = torch.zeros(ng, device=self.device, dtype=self.dtype).index_add_(0, g_all, torch.ones_like(l_all[:, 0]))
        cen = torch.zeros(ng, 3, device=self.device, dtype=self.dtype).index_add_(0, g_all, l_all) / cnt.clamp(min=1)[:, None]
        rad = torch.zeros(ng, device=self.device, dtype=self.dtype).scatter_reduce(0, g_all, (l_all - cen[g_all]).norm(dim=-1), 'amax')
        used = cnt > 0
        self._light = dict(g=g_all, l=l_all, r=r_all, cen=cen, rad=rad, used=used, nreg=nreg)
        return self._light

    def clearance_from_geoms(self, gx, gR, nreg=13):
        """Per-region lowest gauge point z from precomputed geom poses.  One region at a time (peak memory
        ~ batch x largest-region points x 4 floats), z-only: z = t_z + R[2,:] . p."""
        import torch
        L = self._prep_light(nreg)
        if 'blocks' not in L:
            L['blocks'] = [(r, L['g'][L['r'] == r], L['l'][L['r'] == r]) for r in range(nreg)]
        Rz = gR[..., 2, :]                                            # [...,ngeom,3]
        tz = gx[..., 2]
        cols = []
        for r, g, l in L['blocks']:
            if len(g) == 0:
                cols.append(torch.full(tz.shape[:-1], float('inf'), device=self.device, dtype=tz.dtype))
                continue
            cols.append((tz[..., g] + (Rz[..., g, :] * l).sum(-1)).amin(-1))
        return torch.stack(cols, -1)

    def disp_bound(self, gxa, gRa, gxb, gRb):
        """Upper bound of max point displacement between two poses: per geom |dt + dR c| + ||dR||_F r. [...]"""
        L = self._prep_light()
        dR = gRa - gRb
        dc = (gxa - gxb) + (dR @ L['cen'].unsqueeze(-1)).squeeze(-1)
        b = dc.norm(dim=-1) + dR.flatten(-2).norm(dim=-1) * L['rad']
        return b[..., L['used']].amax(-1)

    def disp_bound_geoms(self, gxa, gRa, gxb, gRb, rbound):
        """Upper bound of the displacement of any point inside any geom's bounding sphere (centre = geom frame origin,
        radius rbound [ngeom], MuJoCo geom_rbound): |dx| + ||dR||_F r.  Covers collision geoms (self penetration). [...]"""
        import torch
        rb = torch.as_tensor(np.asarray(rbound), device=self.device, dtype=gxa.dtype)
        b = (gxa - gxb).norm(dim=-1) + (gRa - gRb).flatten(-2).norm(dim=-1) * rb
        return b.amax(-1)

    def region_ground_clearance(self, qpos, nreg=13):
        """Per-region lowest point height above the floor (z=0), the factory gauge: min over hull vertices (region of
        their body) U dense points (their own region).  [..., 13], inf for empty regions."""
        import torch
        gx, gR = self.geoms(qpos)
        H = (gx[..., self.hull_geom, :] + (gR[..., self.hull_geom, :, :] @ self.hull_local.unsqueeze(-1)).squeeze(-1))[..., 2]
        P = (gx[..., self.pts_geom, :] + (gR[..., self.pts_geom, :, :] @ self.pts_local.unsqueeze(-1)).squeeze(-1))[..., 2]
        if not hasattr(self, '_all_reg'):
            hreg = torch.as_tensor(self.k['body_region'], device=self.device)[self.geom_body[self.hull_geom]]
            self._all_reg = torch.cat([self.pts_region, hreg])
        Z = torch.cat([P, H], -1)
        out = torch.full(Z.shape[:-1] + (nreg,), float('inf'), device=self.device, dtype=Z.dtype)
        return out.scatter_reduce(-1, self._all_reg.expand(Z.shape), Z, reduce='amin')


# ---------------------------------------------------------------------------------------------------- static bound
ROOT_ANG_ERR = 1e-5            # rad, >= smallest-three u64 root-quaternion decode error (measured max 2.5e-6)
JOINT_EXTRA = 1e-6             # rad, float32 rounding of a decoded joint (|q| <= 2 pi) + slack


def static_disp_bound(robot: str, root: Path, staged: Path = STAGED, refresh=False) -> dict:
    """Configuration-independent upper bound of how far any gauge point (dense surface points, hull vertices) or any
    collision / visual geom can move between the source qpos and its decoded qpos (root position stored exactly,
    root rotation error <= ROOT_ANG_ERR, joint error <= step/2 + JOINT_EXTRA).

    Rotating by angle e about an axis through anchor a moves a point at distance r by <= e r.  Perturbations are
    applied joint by joint from the root outward; the distance from joint a's anchor to a point p on descendant body b
    is bounded independently of the configuration by the chain  h_a -> h_c1 -> ... -> h_b -> p  where h_c is body c's
    joint anchor (fixed in both c and its parent; body origin for jointless bodies / the free root), each link rigid in
    one body.  Result cached in <root>/robots/<robot>/bounds.json:
        region_m [13]  bound over the gauge points of each region (ground / object region clearance gauges)
        all_m          bound over every gauge point and every geom bounding sphere (self / object penetration)
    """
    out_p = Path(root) / 'robots' / robot / 'bounds.json'
    spec = json.load(open(Path(root) / 'robots' / robot / 'spec.json'))
    if out_p.exists() and not refresh:
        b = json.load(open(out_p))
        if b.get('version') == 2 and b.get('mjcf_sha256') == spec['mjcf_sha256'] and b.get('q_step_max') == float(np.max(spec['q_step'])):
            return b
    import mujoco
    m = mujoco.MjModel.from_xml_path(spec['staged_mjcf'])
    with np.load(Path(root) / 'robots' / robot / 'kin.npz') as z:
        kin = {k: z[k] for k in z.files}
    nb = m.nbody
    par = m.body_parentid
    rootb = int(kin['root_body'])
    step = np.asarray(spec['q_step'])
    h = np.zeros((nb, 3)); delta = np.zeros(nb)
    for b in range(nb):
        j = int(kin['body_joint'][b])
        if j >= 0:
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, 'slide joints: add their travel to the link length'
            h[b] = m.jnt_pos[j]
            delta[b] = step[m.jnt_qposadr[j] - 7] / 2 + JOINT_EXTRA
    delta[rootb] = ROOT_ANG_ERR
    ell = np.zeros(nb)
    for b in range(1, nb):
        if b == rootb:
            continue
        p = par[b]
        ell[b] = np.linalg.norm(m.body_pos[b] + _np_qmat(m.body_quat[b]) @ h[b] - h[p])
    A = np.zeros(nb); B = np.zeros(nb)                       # bound(p on b) = A_b + B_b * |p - h_b|
    for b in range(nb):
        chain = []
        c = b
        while True:
            chain.append(c)
            if c == rootb or c == 0:
                break
            c = par[c]
        if chain[-1] != rootb:                               # world-attached (not part of the robot)
            continue
        # chain = [b, ..., root]; S(b, a) = sum of ell over bodies strictly below a down to b
        s = 0.0
        for c in chain:                                      # walk upward: s = sum ell of bodies below c (excl. c)
            A[b] += delta[c] * s
            B[b] += delta[c]
            s += ell[c]
    def pts_bound(gid, loc):
        bb = m.geom_bodyid[gid]
        P = m.geom_pos[gid] + np.einsum('nij,nj->ni', np.stack([_np_qmat(q) for q in m.geom_quat[gid]]), loc)
        return A[bb] + B[bb] * np.linalg.norm(P - h[bb], axis=1)
    bp = pts_bound(kin['pts_geom'], kin['pts_local'])
    bh = pts_bound(kin['hull_geom'], kin['hull_local'])
    hreg = kin['body_region'][m.geom_bodyid[kin['hull_geom']]]
    gb = m.geom_bodyid
    robot_geom = np.array([A[gb[g]] > 0 or B[gb[g]] > 0 for g in range(m.ngeom)])
    gbound = A[gb] + B[gb] * (np.linalg.norm(m.geom_pos - h[gb], axis=1) + m.geom_rbound)
    greg = kin['body_region'][gb]
    reg = np.zeros(13)
    for r in range(13):
        # region gauge = dense points U visual hull vertices U (factory >= 2026-10-04 22:47) the lowest support point of
        # every collision geom of the region: bound every robot geom of the region by its bounding sphere as well
        v = np.concatenate([bp[kin['pts_region'] == r], bh[hreg == r], gbound[robot_geom & (greg == r)]])
        reg[r] = v.max() if len(v) else 0.0
    allm = float(max(bp.max(), bh.max(), gbound[robot_geom].max()))
    out = dict(version=2, robot=robot, region_m=reg.tolist(), all_m=allm, geom_rbound=np.where(robot_geom, m.geom_rbound, 0.0).tolist(), root_ang_err=ROOT_ANG_ERR, joint_extra=JOINT_EXTRA,
               q_step_max=float(step.max()), mjcf_sha256=spec['mjcf_sha256'])
    tmp = out_p.with_name(out_p.name + '.tmp')
    json.dump(out, open(tmp, 'w'), indent=1)
    import os
    os.replace(tmp, out_p)
    return out
