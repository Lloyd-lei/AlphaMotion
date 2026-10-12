"""Step 3b -- INDEPENDENT measurement of one unitree_r1 output clip against DATA_SPEC_v2.

Robot side (own code, no umrf.geometry / umrf.labels): dense surface points posed by MuJoCo FK + convex-hull vertices
of the visual geoms; ground = z (floor already 0 in the output); object = own signed distance on the object's solid
(analytic sphere/box or union of convex pieces); MuJoCo cross-check = mj_geomDistance of every robot collision geom vs
a z=0 plane, vs the object's solid geoms, and every non parent/child self pair.
Human side: UMR's 4096 source slots (retarget frame, scaled), shifted by the same floor; slot regions from the
correspondence (factory mapping, asset only); human contact = same ruler as the factory (enter < 1 cm / leave > 2 cm,
>= 3 frames).  Object geometry + pose come from the output npz / object loader (asset, not a gauge).
Writes qa/unitree_r1/<qa_id>/measure.json + state.npz (per point/frame colour state for the renderer).
Usage: timeout 1800 <umr env python> measure_r1.py <qa_id>"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

import r1_common as C

PEN = 0.0005; CT = 0.001; ENTER = 0.001; LEAVE = 0.002; MIN_ON = 3; SLIP = 0.01
H_ENTER, H_LEAVE = 0.01, 0.02; H_OFF = 0.02; R_OFF = 0.005; HOVER = (0.001, 0.02)
NR = len(C.REGIONS)


# ----------------------------------------------------------------------------------------- helpers
def runs(col):
    col = np.asarray(col, bool)
    if not len(col):
        return []
    chg = np.flatnonzero(np.diff(col.astype(np.int8))) + 1
    st = np.concatenate([[0], chg]); en = np.concatenate([chg, [len(col)]])
    return [(int(a), int(b)) for a, b in zip(st, en) if col[a]]


def hyst(x, enter, leave):
    """1-D hysteresis on a clearance series (inf/nan = far)."""
    out = np.zeros(len(x), bool); on = False
    for i, v in enumerate(x):
        if not np.isfinite(v):
            on = False
        elif on:
            on = not (v > leave)
        else:
            on = v < enter
        out[i] = on
    return out


def drop_short(c, n):
    c = c.copy()
    for a, b in runs(c):
        if b - a < n:
            c[a:b] = False
    return c


def pct(v, q):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return round(float(np.percentile(v, q)), 3) if len(v) else None


class Solid:
    """Object solid in its body frame: sphere / box / union of convex pieces.  sdf(Xlocal) -> (d, n_local)."""

    def __init__(self, obj, col=None):
        # The factory classifier (umrf.objects.PosedObject) calls ANY 8-vertex box a 'sphere' (all 8 corners of a box are
        # equidistant from its centre) -> never trust obj.kind for low-vertex meshes: rebuild from the collision geoms.
        self.kind = obj.kind
        self.factory_kind = obj.kind
        if col is not None and obj.kind in ('sphere', 'box', 'mesh'):
            nv = sum(len(np.unique(np.round(v, 7), axis=0)) for v, f in col)
            if not (obj.kind == 'sphere' and nv > 50):
                self.kind = 'convex_union'
                obj = type('O', (), dict(pieces=[dict(verts=v) for v, f in col]))()
        if self.kind == 'sphere':
            self.c, self.r = np.asarray(obj.center, float), float(obj.radius)
            self.mj = [dict(type='sphere', size=[self.r, 0, 0], pos=self.c, mat=np.eye(3))]
            self.bc, self.br = self.c, self.r
        elif self.kind == 'box':
            self.T = np.asarray(obj.box_T, float); self.h = np.asarray(obj.box_half, float)
            self.mj = [dict(type='box', size=list(self.h), pos=self.T[:3, 3], mat=self.T[:3, :3])]
            self.bc, self.br = self.T[:3, 3], float(np.linalg.norm(self.h))
        elif self.kind == 'convex_union':
            self.pieces = []
            for p in obj.pieces:
                v = np.asarray(p['verts'], float); h = ConvexHull(v)
                self.pieces.append(dict(A=h.equations[:, :3], b=h.equations[:, 3], mesh=trimesh.convex.convex_hull(v),
                                        c=v.mean(0), r=float(np.linalg.norm(v - v.mean(0), axis=1).max()), v=v))
            self.mj = [dict(type='mesh', verts=p['v'], pos=np.zeros(3), mat=np.eye(3)) for p in self.pieces]
            allv = np.concatenate([p['v'] for p in self.pieces])
            self.bc = allv.mean(0); self.br = float(np.linalg.norm(allv - self.bc, axis=1).max())
        else:
            raise ValueError(self.kind)

    def surface_points(self, n=1500, seed=0):
        if self.kind == 'sphere':
            u = np.random.default_rng(seed).normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            return self.c + self.r * u
        if self.kind == 'box':
            m = trimesh.creation.box(extents=2 * self.h, transform=self.T)
        else:
            m = trimesh.util.concatenate([p['mesh'] for p in self.pieces])
        return trimesh.sample.sample_surface_even(m, n, seed=seed)[0]

    def sdf(self, q, cull=0.03):
        d = np.full(len(q), np.inf)
        near = np.linalg.norm(q - self.bc, axis=1) < self.br + cull
        if not near.any():
            return d
        x = q[near]
        if self.kind == 'sphere':
            dn = np.linalg.norm(x - self.c, axis=1) - self.r
        elif self.kind == 'box':
            p = (x - self.T[:3, 3]) @ self.T[:3, :3]
            o = np.abs(p) - self.h
            dn = np.linalg.norm(np.maximum(o, 0), axis=1) + np.minimum(o.max(1), 0)
        else:
            dn = np.full(len(x), np.inf)
            for pc in self.pieces:
                k = np.flatnonzero(np.linalg.norm(x - pc['c'], axis=1) < pc['r'] + cull)
                if not len(k):
                    continue
                s = (x[k] @ pc['A'].T + pc['b']).max(1)          # <0 inside: exact depth for a convex polytope
                ins = s < 0
                val = np.where(ins, s, np.inf)
                o = np.flatnonzero(~ins & (s < cull))
                if len(o):
                    _, dist, _ = trimesh.proximity.closest_point(pc['mesh'], x[k[o]])
                    val[o] = dist
                dn[k] = np.minimum(dn[k], val)
        d[near] = dn
        return d


# ----------------------------------------------------------------------------------------- main
def main(qa_id):
    t_all = time.time()
    man = [json.loads(l) for l in open(C.PDIR / 'out' / 'manifest.jsonl')]
    row = [r for r in man if r['qa_id'] == qa_id and r['status'].startswith('ok')][-1]
    o = np.load(row['npz'], allow_pickle=True)
    prov = json.loads(str(o['provenance']))
    Q = o['qpos'].astype(np.float64); T = len(Q); fps = float(o['fps'])
    floor = float(o['floor_offset_removed'])
    m = C.load_model(); surf = C.load_surface()
    patches = dict(np.load(C.PDIR / 'patches.npz'))
    reg = surf['region'].astype(int)
    has_obj = 'object_pos' in o.files
    solid = None
    if has_obj:
        from umrf.objects import load_object_strict
        U = np.load(prov['umr_out'], allow_pickle=True)
        up = str(U['noitom_output_up']) if 'noitom_output_up' in U.files else 'y'
        obj, oinfo = load_object_strict(str(U['source_object_dir']), U['frame_ids'], float(U['smpl_scale'][0]), up)
        from umrf.objects import _mjcf_geoms_local
        _, col = _mjcf_geoms_local(oinfo['xml'], float(U['smpl_scale'][0]))
        solid = Solid(obj, col)
        opos = o['object_pos'].astype(float); orot = o['object_rot'].astype(float)
        pose_check_mm = float(np.abs(obj.pos - np.array([0, 0, floor]) - opos).max() * 1000)
    # ---- own MuJoCo model: robot + plane z=0 + object mocap body
    spec = mujoco.MjSpec.from_file(str(C.XML))
    fb = spec.worldbody.add_body(); fb.name = '__r1_floor_body'        # appended LAST: robot body/geom ids unchanged
    pl = fb.add_geom(); pl.name = '__r1_floor'; pl.type = mujoco.mjtGeom.mjGEOM_PLANE; pl.size = [50, 50, 0.1]
    if has_obj:
        b = spec.worldbody.add_body(); b.name = '__r1_obj'; b.mocap = True
        for i, gd in enumerate(solid.mj):
            g = b.add_geom(); g.name = f'__r1_obj{i}'
            q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, np.asarray(gd['mat'], float).reshape(-1)); g.pos = np.asarray(gd['pos'], float); g.quat = q4
            if gd['type'] == 'sphere':
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.size = np.asarray(gd['size'], float)
            elif gd['type'] == 'box':
                g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = np.asarray(gd['size'], float)
            else:
                ms = spec.add_mesh(); ms.name = f'__r1_objm{i}'; ms.uservert = np.asarray(gd['verts'], np.float32).reshape(-1)
                g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = ms.name
    mm = spec.compile(); dd = mujoco.MjData(mm)
    assert mm.nq == m.nq
    for gi in range(m.ngeom):                                       # ids must match the staged model exactly
        assert mm.geom_bodyid[gi] == m.geom_bodyid[gi] and mm.geom_type[gi] == m.geom_type[gi] and \
            mm.geom_dataid[gi] == m.geom_dataid[gi], 'geom ids moved'
    plane = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, '__r1_floor')
    ogeoms = [g for g in range(mm.ngeom) if (mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('__r1_obj')]
    mocap = int(mm.body_mocapid[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, '__r1_obj')]) if has_obj else -1
    rcg = C.collision_geoms(m)                               # robot collision geoms (ids identical in mm)
    st = json.load(open(C.RDIR / 'staging.json'))
    pairs = C.self_pairs_all(m, [tuple(e['bodies']) if isinstance(e, dict) else tuple(e) for e in st.get('exclusions', [])])
    breg = surf['body_region'].astype(int)
    cg_reg = np.array([breg[m.geom_bodyid[g]] for g in rcg])
    # ---- posing
    pl_loc = surf['points_local'].astype(float); pg = surf['geom'].astype(int)
    hl = surf['hull_local'].astype(float); hg = surf['hull_geom'].astype(int); hreg_ = breg[m.geom_bodyid[hg]]
    groups = [(g, np.flatnonzero(pg == g)) for g in np.unique(pg)]
    hgroups = [(g, np.flatnonzero(hg == g)) for g in np.unique(hg)]
    P = len(pl_loc)
    X = np.zeros((T, P, 3), np.float32)
    gclr_dense = np.zeros((T, NR)); gclr_hull = np.zeros((T, NR))
    gcoll = np.full((T, NR), np.inf); gcoll_min = np.zeros(T)
    oclr_dense = np.full((T, P), np.inf, np.float32); oreg_dense = np.full((T, NR), np.inf)
    ocoll = np.full((T, NR), np.inf); ocoll_min = np.full(T, np.inf)
    self_min = np.full(T, np.inf); self_bad = {}
    Xl_obj = np.zeros((T, P, 3), np.float32) if has_obj else None
    ft = np.zeros(6)
    fz = dict(ground=0, self=0, object=0, object_lt_1mm_if_trusted=0)

    def gdist(a, b, dmax, kind):
        """mj_geomDistance with a guard: MuJoCo 3.3.7 returns a FALSE 0.0 for some mesh-mesh pairs whose true
        distance is below distmax (verified: 36 mm apart -> 0.0 with distmax >= 0.04, 35.0 with distmax 0.035).
        Exact zeros are resolved by bisection on distmax (a query returns distmax when the true distance exceeds it)."""
        r = mujoco.mj_geomDistance(mm, dd, a, b, dmax, ft)
        if r != 0.0:
            return r
        r1 = mujoco.mj_geomDistance(mm, dd, a, b, 1e-3, ft)
        if r1 < 1e-3:
            return r1                                            # genuine contact / sub-mm (or negative)
        lo, hi = 1e-3, dmax
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            rm = mujoco.mj_geomDistance(mm, dd, a, b, mid, ft)
            if rm >= mid - 1e-12:
                lo = mid
            elif rm == 0.0:
                hi = mid
            else:
                lo = hi = rm; break
        fz[kind] += 1
        return 0.5 * (lo + hi)

    for t in range(T):
        dd.qpos[:] = Q[t]
        if has_obj:
            q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, orot[t].reshape(-1))
            dd.mocap_pos[mocap] = opos[t]; dd.mocap_quat[mocap] = q4
        mujoco.mj_kinematics(mm, dd)
        for g, k in groups:
            X[t, k] = pl_loc[k] @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g]
        H = np.empty((len(hl), 3))
        for g, k in hgroups:
            H[k] = hl[k] @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g]
        z = X[t, :, 2].astype(float)
        for r in range(NR):
            gclr_dense[t, r] = z[reg == r].min()
            gclr_hull[t, r] = min(gclr_dense[t, r], H[hreg_ == r, 2].min())
        # MuJoCo: plane vs every robot collision geom
        for i, g in enumerate(rcg):
            dist = gdist(plane, g, 0.05, 'ground')
            gcoll[t, cg_reg[i]] = min(gcoll[t, cg_reg[i]], dist)
        gcoll_min[t] = gcoll[t].min()
        # self pairs
        for a, b in pairs:
            dist = gdist(a, b, 0.02, 'self')
            if dist < self_min[t]:
                self_min[t] = dist
            if dist < -PEN:
                key = f'{C.bname(m, m.geom_bodyid[a])}|{C.bname(m, m.geom_bodyid[b])}'
                self_bad.setdefault(key, []).append(round(-dist * 1000, 2))
        if has_obj:
            ql = (X[t].astype(float) - opos[t]) @ orot[t]          # object frame
            Xl_obj[t] = ql
            od = solid.sdf(ql)
            oclr_dense[t] = od
            for r in range(NR):
                oreg_dense[t, r] = od[reg == r].min()
            for i, g in enumerate(rcg):
                if np.linalg.norm(dd.geom_xpos[g] - (opos[t] + orot[t] @ solid.bc)) > solid.br + m.geom_rbound[g] + 0.05:
                    continue
                for og in ogeoms:
                    n0 = fz['object']
                    dist = gdist(g, og, 0.05, 'object')
                    if fz['object'] > n0 and oreg_dense[t, cg_reg[i]] > CT:
                        fz['object_lt_1mm_if_trusted'] += 1
                    ocoll[t, cg_reg[i]] = min(ocoll[t, cg_reg[i]], dist)
            ocoll_min[t] = ocoll[t].min()
    t_pose = time.time() - t_all
    fz_out = dict(fz)
    # ---- human
    from umrf import pipeline as UP
    Hs = np.load(prov['human_slots'])['source_slots']
    U = np.load(prov['umr_out'], allow_pickle=True)
    if len(Hs) != T:
        Hs = Hs[U['frame_ids']]
    Hs = (Hs - np.array([0, 0, floor], np.float32)).astype(np.float32)
    hreg, hinfo = UP.human_slot_regions(C.ROBOT, prov['source_kind'], prov['source_data'], prov['seq_key'],
                                        str(Path(prov['umr_out']).with_suffix('.umr.log')), shared=prov['source_kind'] == 'hoi')
    h_g = np.full((T, NR), np.inf); h_o = np.full((T, NR), np.inf)
    h_od = np.full((T, Hs.shape[1]), np.inf, np.float32) if has_obj else None
    for t in range(T):
        for r in range(NR):
            if (hreg == r).any():
                h_g[t, r] = Hs[t, hreg == r, 2].min()
        if has_obj:
            hd = solid.sdf((Hs[t].astype(float) - opos[t]) @ orot[t], cull=0.05)
            h_od[t] = hd
            for r in range(NR):
                if (hreg == r).any():
                    h_o[t, r] = hd[hreg == r].min()
    hc_g = np.stack([drop_short(hyst(h_g[:, r], H_ENTER, H_LEAVE), MIN_ON) for r in range(NR)], 1)
    hc_o = np.stack([drop_short(hyst(h_o[:, r], H_ENTER, H_LEAVE), MIN_ON) for r in range(NR)], 1) if has_obj else np.zeros((T, NR), bool)
    # ---- robot contact (spec v2)
    out = dict(qa_id=qa_id, clip=row['clip'], frames=T, fps=fps, floor_offset_removed_m=floor, npz=row['npz'],
               human_region_map=hinfo, seconds_pose=round(t_pose, 1))
    if has_obj:
        out['object'] = dict(name=str(o['object_name']), kind=solid.kind, factory_kind=solid.factory_kind, info=oinfo,
                             pose_vs_loader_max_mm=round(pose_check_mm, 3),
                             n_collision_pieces=len(col), collision_vertex_counts=[int(len(v)) for v, f in col])

    def contact_block(clr_dense, clr_hull, coll, hc, kind):
        """clr_* [T,NR] clearances; coll [T,NR] MuJoCo; hc [T,NR] human contact."""
        res = {}
        solid_d = (clr_dense >= -PEN) & (clr_dense <= CT)              # >= 1 dense point within [-0.5, 1] mm
        solid_strict = (clr_dense >= 0) & (clr_dense <= CT)
        solid_h = (clr_hull >= -PEN) & (clr_hull <= CT)
        hover = (clr_dense > HOVER[0]) & (clr_dense <= HOVER[1])
        for r, rn in enumerate(C.REGIONS):
            if not hc[:, r].any() and not solid_d[:, r].any():
                continue
            hcr = hc[:, r]; n = int(hcr.sum())
            e = dict(human_contact_frames=n)
            if n:
                e.update(robot_solid_frac=round(float(solid_d[hcr, r].mean()), 4),
                         robot_solid_strict_0_1mm_frac=round(float(solid_strict[hcr, r].mean()), 4),
                         robot_solid_incl_hull_frac=round(float(solid_h[hcr, r].mean()), 4),
                         hover_1_20mm_frames=int(hover[hcr, r].sum()),
                         far_gt20mm_frames=int((clr_dense[hcr, r] > HOVER[1]).sum()),
                         pen_frames=int((clr_dense[hcr, r] < -PEN).sum()),
                         robot_clearance_mm_p50_p95=[pct(clr_dense[hcr, r] * 1000, 50), pct(clr_dense[hcr, r] * 1000, 95)])
            # human off (> 2 cm) -> robot must be off (> 5 mm)
            hoff = (h_g if kind == 'ground' else h_o)[:, r] > H_OFF
            e['human_off_frames'] = int(hoff.sum())
            e['robot_not_off_while_human_off_frames'] = int((hoff & (clr_dense[:, r] <= R_OFF)).sum())
            e['robot_solid_frames_total'] = int(solid_d[:, r].sum())
            # flicker: raw solid runs and spec hysteresis runs
            raw = solid_d[:, r]
            hy = hyst(clr_dense[:, r], ENTER, LEAVE)
            e['raw_contact_runs'] = len(runs(raw)); e['raw_runs_lt3'] = sum(1 for a, b in runs(raw) if b - a < MIN_ON)
            e['hyst_runs'] = len(runs(hy)); e['hyst_runs_lt3'] = sum(1 for a, b in runs(hy) if b - a < MIN_ON)
            e['mujoco_coll_min_mm'] = pct(coll[:, r] * 1000, 0)
            res[rn] = e
        return res, solid_d, hover

    def slip_block(clr_pts, pos, solid_reg, kind):
        """clr_pts [T,P] clearance per dense point; pos [T,P,3] positions in the contact partner frame;
        solid_reg [T,NR].  Contact points = region points within [-0.5, 1] mm at t-1, t, t+1 (material points)."""
        V = np.zeros_like(pos, dtype=np.float64)
        V[1:-1] = (pos[2:].astype(float) - pos[:-2].astype(float)) * fps / 2
        if kind == 'ground':
            sp = np.linalg.norm(V[..., :2], axis=-1)
        else:
            sp = np.linalg.norm(V, axis=-1)                        # full relative speed in the object frame (upper bound)
        cp = (clr_pts >= -PEN) & (clr_pts <= CT)
        cp3 = cp.copy(); cp3[1:] &= cp[:-1]; cp3[:-1] &= cp[1:]; cp3[0] = False; cp3[-1] = False
        res = {}; slip_frames = np.zeros((T, NR), bool)
        allmin, allmed = [], []
        for r, rn in enumerate(C.REGIONS):
            idx = np.flatnonzero(reg == r)
            vmin = np.full(T, np.nan); vmed = np.full(T, np.nan)
            for t in range(1, T - 1):
                k = idx[cp3[t, idx]]
                if solid_reg[t, r] and len(k):
                    vmin[t] = sp[t, k].min(); vmed[t] = np.median(sp[t, k])
            ok = np.isfinite(vmin)
            if not ok.any():
                continue
            slip_frames[:, r] = ok & (vmin >= SLIP)
            allmin.append(vmin[ok]); allmed.append(vmed[ok])
            res[rn] = dict(judged_frames=int(ok.sum()),
                           best_point_speed_cms_p50=pct(vmin[ok] * 100, 50), best_point_speed_cms_p95=pct(vmin[ok] * 100, 95),
                           median_point_speed_cms_p50=pct(vmed[ok] * 100, 50), median_point_speed_cms_p95=pct(vmed[ok] * 100, 95),
                           frames_best_point_ge_1cms=int((vmin[ok] >= SLIP).sum()),
                           frac_best_point_ge_1cms=round(float((vmin[ok] >= SLIP).mean()), 4),
                           frac_median_point_ge_1cms=round(float((vmed[ok] >= SLIP).mean()), 4))
        if allmin:
            a = np.concatenate(allmin) * 100; b = np.concatenate(allmed) * 100
            res['_all'] = dict(judged_region_frames=int(len(a)), best_point_speed_cms_p50=pct(a, 50), best_point_speed_cms_p95=pct(a, 95),
                               median_point_speed_cms_p50=pct(b, 50), median_point_speed_cms_p95=pct(b, 95),
                               frac_best_point_ge_1cms=round(float((a >= 1).mean()), 4), frac_median_point_ge_1cms=round(float((b >= 1).mean()), 4))
        return res, slip_frames

    gres, g_solid, g_hover = contact_block(gclr_dense, gclr_hull, gcoll, hc_g, 'ground')
    zpts = X[..., 2].astype(np.float32)
    gslip, g_slipf = slip_block(zpts, X, g_solid, 'ground')
    out['ground'] = dict(regions=gres, slip=gslip)
    if has_obj:
        ores, o_solid, o_hover = contact_block(oreg_dense, oreg_dense, ocoll, hc_o, 'object')
        oslip, o_slipf = slip_block(oclr_dense, Xl_obj, o_solid, 'object')
        out['object_contact'] = dict(regions=ores, slip=oslip)
    # ---- penetration
    out['penetration'] = dict(
        ground_dense_max_mm=round(float(max(0, -gclr_dense.min()) * 1000), 3),
        ground_hull_max_mm=round(float(max(0, -gclr_hull.min()) * 1000), 3),
        ground_mujoco_coll_max_mm=round(float(max(0, -gcoll_min.min()) * 1000), 3),
        ground_frames_gt_0p5mm=int(((np.minimum(gclr_hull.min(1), gcoll_min)) < -PEN).sum()),
        self_mujoco_max_mm=round(float(max(0, -self_min.min()) * 1000), 3),
        self_frames_gt_0p5mm=int((self_min < -PEN).sum()),
        self_pairs_gt_0p5mm={k: dict(frames=len(v), max_mm=max(v)) for k, v in sorted(self_bad.items(), key=lambda x: -len(x[1]))})
    if has_obj:
        out['penetration'].update(
            object_dense_max_mm=round(float(max(0, -np.nanmin(np.where(np.isfinite(oreg_dense), oreg_dense, np.inf))) * 1000), 3),
            object_mujoco_coll_max_mm=round(float(max(0, -ocoll_min.min()) * 1000), 3) if np.isfinite(ocoll_min).any() else 0.0,
            object_frames_gt_0p5mm=int(((np.minimum(oreg_dense.min(1), ocoll_min)) < -PEN).sum()))
    # ---- per-frame rule failures (contact-capable regions = all 13 for R1)
    fail = {}
    pen_frame = (np.minimum(gclr_hull.min(1), gcoll_min) < -PEN) | (self_min < -PEN)
    if has_obj:
        pen_frame |= np.minimum(oreg_dense.min(1), ocoll_min) < -PEN
    fail['penetration'] = pen_frame
    nsolid = (hc_g & ~g_solid).any(1)
    if has_obj:
        nsolid |= (hc_o & ~o_solid).any(1)
    fail['human_contact_not_solid'] = nsolid
    off = ((h_g > H_OFF) & (gclr_dense <= R_OFF)).any(1)
    if has_obj:
        off |= ((h_o > H_OFF) & (oreg_dense <= R_OFF)).any(1)
    fail['robot_contact_while_human_off'] = off
    sl = g_slipf.any(1)
    if has_obj:
        sl |= o_slipf.any(1)
    fail['slip_ge_1cms'] = sl
    fl = np.zeros(T, bool)
    for r in range(NR):
        for a, b in runs(hyst(gclr_dense[:, r], ENTER, LEAVE)):
            if b - a < MIN_ON:
                fl[a:b] = True
        if has_obj:
            for a, b in runs(hyst(oreg_dense[:, r], ENTER, LEAVE)):
                if b - a < MIN_ON:
                    fl[a:b] = True
    fail['flicker_lt3'] = fl
    anyf = np.zeros(T, bool)
    for v in fail.values():
        anyf |= v
    flags = o['flags'].astype(int)
    out['frames_failing'] = {k: int(v.sum()) for k, v in fail.items()}
    out['frames_failing_any'] = int(anyf.sum())
    out['frames_failing_any_excl_slip'] = int((anyf & ~0).sum() if False else (fail['penetration'] | fail['human_contact_not_solid'] |
                                                                                 fail['robot_contact_while_human_off'] | fail['flicker_lt3']).sum())
    out['factory_flags'] = dict(frames_flagged=int((flags > 0).sum()),
                                hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16)})
    out['failing_frames_not_flagged_by_factory'] = {k: int((v & (flags == 0)).sum()) for k, v in fail.items()}
    # ---- spec format checks on the factory output
    pc = np.unpackbits(o['ground_point_contact'], axis=1)[:, :P].astype(bool)
    zc = zpts[pc]
    out['format'] = dict(
        floor_height_is_0=bool(float(o['floor_height']) == 0.0), floor_offset_stored='floor_offset_removed' in o.files,
        fps=fps, ground_point_contact_points=int(pc.sum()),
        ground_point_contact_frac_gt_1mm=round(float((zc > CT).mean()), 4) if len(zc) else None,
        ground_point_contact_clearance_mm_p50_p95=[pct(zc * 1000, 50), pct(zc * 1000, 95)],
        has_quantized_distance_lt_2cm=any(k for k in o.files if 'dist' in k and 'quant' in k),
        has_human_smplx_params=any(k in o.files for k in ('human_poses', 'smplx_poses', 'betas')),
        has_object_6d_pose=bool(has_obj and 'object_rot' in o.files), keys=list(o.files))
    # solidity of the factory's OWN output contact label (spec: >= 1 dense point within [0, 1] mm)
    def label_solidity(lab, clr_dense, clr_hull):
        lab = np.asarray(lab, bool); n = int(lab.sum())
        if not n:
            return dict(label_region_frames=0)
        cd = clr_dense[lab]; ch = clr_hull[lab]
        return dict(label_region_frames=n,
                    dense_solid_frac=round(float(((cd >= -PEN) & (cd <= CT)).mean()), 4),
                    dense_strict_0_1mm_frac=round(float(((cd >= 0) & (cd <= CT)).mean()), 4),
                    incl_hull_solid_frac=round(float(((ch >= -PEN) & (ch <= CT)).mean()), 4),
                    dense_clearance_mm_p50_p95_max=[pct(cd * 1000, 50), pct(cd * 1000, 95), pct(cd * 1000, 100)])
    out['factory_label_solidity'] = dict(ground=label_solidity(o['ground_contact'], gclr_dense, gclr_hull))
    if has_obj:
        out['factory_label_solidity']['object'] = label_solidity(o['object_contact'], oreg_dense, oreg_dense)
    # human floor sanity: human foot lowest point during human foot contact
    hf = np.concatenate([h_g[hc_g[:, C.RID[f]], C.RID[f]] for f in ('L_foot', 'R_foot')])
    out['human_floor_check_mm'] = dict(foot_contact_lowest_p05_p50_p95=[pct(hf * 1000, 5), pct(hf * 1000, 50), pct(hf * 1000, 95)])
    out['human_contact_frames'] = dict(ground={rn: int(hc_g[:, r].sum()) for r, rn in enumerate(C.REGIONS) if hc_g[:, r].any()},
                                       object={rn: int(hc_o[:, r].sum()) for r, rn in enumerate(C.REGIONS) if hc_o[:, r].any()})
    # patch-level summary (knee / elbow / pelvis / hand / foot) ground contact frames for the robot
    pt = {}
    for k, idx in patches.items():
        zmin = zpts[:, idx].min(1)
        pt[k] = dict(robot_ground_solid_frames=int(((zmin >= -PEN) & (zmin <= CT)).sum()), robot_min_clearance_mm=round(float(zmin.min() * 1000), 2))
        if has_obj:
            om = oclr_dense[:, idx].min(1)
            pt[k]['robot_object_solid_frames'] = int(((om >= -PEN) & (om <= CT)).sum())
    out['patches'] = pt
    out['mujoco_false_zero_resolved'] = fz_out
    out['seconds_total'] = round(time.time() - t_all, 1)
    qd = C.QADIR / qa_id; qd.mkdir(parents=True, exist_ok=True)
    C.dump(qd / 'measure.json', out)
    # ---- per point state for the renderer: 0 none, 1 contact (green), 2 penetration (red), 3 hover while human in contact (orange)
    state = np.zeros((T, P), np.uint8)
    hov_reg = (hc_g & g_hover & ~g_solid)
    clr_pt = zpts.copy()
    if has_obj:
        hov_reg_o = hc_o & o_hover & ~o_solid
    for t in range(T):
        zz = zpts[t]
        st_ = np.zeros(P, np.uint8)
        hv = hov_reg[t][reg] & (zz > HOVER[0]) & (zz <= HOVER[1])
        if has_obj:
            od = oclr_dense[t]
            hv |= hov_reg_o[t][reg] & (od > HOVER[0]) & (od <= HOVER[1])
        st_[hv] = 3
        st_[(zz >= -PEN) & (zz <= CT)] = 1
        st_[zz < -PEN] = 2
        if has_obj:
            st_[(od >= -PEN) & (od <= CT)] = 1
            st_[od < -PEN] = 2
        state[t] = st_
    hstate = np.zeros((T, Hs.shape[1]), np.uint8)
    for t in range(T):
        hz = Hs[t, :, 2]
        hstate[t][(hz <= 0.01) & hc_g[t][hreg]] = 1
        if has_obj:
            hstate[t][(h_od[t] <= 0.01) & hc_o[t][hreg]] = 1
    extra = {}
    if has_obj:
        extra = dict(obj_pts_local=solid.surface_points(1500).astype(np.float32), opos=opos.astype(np.float32), orot=orot.astype(np.float32))
    np.savez_compressed(qd / 'state.npz', state=state, hstate=hstate, hreg=hreg.astype(np.int8),
                        fail_any=anyf, fail_pen=fail['penetration'], fail_nsolid=fail['human_contact_not_solid'],
                        fail_slip=fail['slip_ge_1cms'], hc_g=hc_g, hc_o=hc_o, g_solid=g_solid,
                        gclr=gclr_dense.astype(np.float32), gclr_hull=gclr_hull.astype(np.float32),
                        oclr=(oreg_dense if has_obj else np.full((T, NR), np.inf)).astype(np.float32),
                        self_min=self_min.astype(np.float32), **extra)
    print(json.dumps(C.jsonable({k: out[k] for k in ('clip', 'penetration', 'frames_failing', 'frames_failing_any',
                                                       'factory_flags', 'failing_frames_not_flagged_by_factory', 'seconds_total')}), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
