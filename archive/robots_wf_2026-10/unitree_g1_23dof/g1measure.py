"""Independent DATA_SPEC_v2 measurement for unitree_g1_23dof outputs (does not import the factory's geometry / labels /
postpass code; only reads the output npz, the staged MJCF, surface.npz and the raw UMR files referenced in provenance).

Gauges (metres, + separated, - penetrating), all floor-relative (output qpos has the calibrated floor at z = 0):
  ground  : dense points (4096) and convex-hull vertices of every visual geom posed by MuJoCo FK (z), plus
            mj_geomDistance(floor plane, every robot collision geom)                                 [MuJoCo cross-check]
  object  : object rebuilt from its MJCF (mesh scale x smpl_scale) as convex pieces; signed distance of dense points
            (inside = max face-plane value, outside = exact closest point on the piece hull), plus
            mj_geomDistance(robot collision geom, object mesh geom) with the object as a mocap body  [MuJoCo cross-check]
  self    : mj_geomDistance for every robot collision-geom pair MuJoCo itself would check (same-weld / parent-child /
            <exclude> / contype filters applied, i.e. only adjacent pairs skipped)
Human: UMR's 4096 captured source slots (retarget frame, robot scale) minus the same floor; regions from the
correspondence (nearest target point -> SMPL-X face -> part -> region).
"""
from __future__ import annotations

import json, pickle, re, sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1common as C

UMR_ROOT = Path('/var/tmp/alphamotion/umr_factory/umr/UMR')
SPEC = dict(contact_max=0.001, contact_min_strict=0.0, pen_tol=0.0005, slip_max=0.01, min_seg=3,
            hyst_enter=0.001, hyst_leave=0.002, human_enter=0.01, human_leave=0.02, human_off=0.02, robot_off=0.005,
            hover_max=0.02)
SMPLX_PART_REGION = {0: 'trunk', 3: 'trunk', 6: 'trunk', 9: 'trunk', 12: 'trunk', 13: 'trunk', 14: 'trunk', 15: 'trunk',
                     22: 'trunk', 23: 'trunk', 24: 'trunk', 1: 'L_thigh', 2: 'R_thigh', 4: 'L_shin', 5: 'R_shin',
                     7: 'L_foot', 10: 'L_foot', 8: 'R_foot', 11: 'R_foot', 16: 'L_upper_arm', 17: 'R_upper_arm',
                     18: 'L_forearm', 19: 'R_forearm', 20: 'L_hand', 21: 'R_hand'}
SMPLX_PART_REGION.update({j: 'L_hand' for j in range(25, 40)}); SMPLX_PART_REGION.update({j: 'R_hand' for j in range(40, 55)})
NR = len(C.REGIONS)


# ------------------------------------------------------------------------------------------------ helpers
def hysteresis(x, enter, leave):
    """x [T,...] clearance; on when x < enter, off when x > leave (or not finite)."""
    T = x.shape[0]; f = x.reshape(T, -1)
    out = np.zeros_like(f, bool); state = f[0] < enter
    out[0] = state
    for t in range(1, T):
        on = f[t] < enter; off = (f[t] > leave) | ~np.isfinite(f[t])
        state = np.where(on, True, np.where(off, False, state))
        out[t] = state
    return out.reshape(x.shape)


def runs(col):
    col = np.asarray(col, bool)
    if not len(col):
        return []
    chg = np.flatnonzero(np.diff(col.astype(np.int8))) + 1
    st = np.concatenate([[0], chg]); en = np.concatenate([chg, [len(col)]])
    return [(int(a), int(b)) for a, b in zip(st, en) if col[a]]


def drop_short(c, n):
    c = np.array(c, bool, copy=True); c2 = c.reshape(c.shape[0], -1)
    for k in range(c2.shape[1]):
        for a, b in runs(c2[:, k]):
            if b - a < n:
                c2[a:b, k] = False
    return c2.reshape(c.shape)


# ------------------------------------------------------------------------------------------------ object
class ConvexObject:
    """Union of convex pieces in the object body frame + per-frame pose (world <- object)."""

    def __init__(self, xml, scale, pos, rot):
        spec = mujoco.MjSpec.from_file(str(xml))
        for mm in spec.meshes:
            mm.scale = np.asarray(mm.scale, np.float64) * float(scale)
        m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        coll = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if not coll:
            coll = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.pieces = []
        for g in coll:
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(np.float64)
            body = int(m.geom_bodyid[g])
            # geom -> body frame (object body = the moving prop body)
            Rg = d.geom_xmat[g].reshape(3, 3); pg = d.geom_xpos[g]
            Rb = d.xmat[body].reshape(3, 3); pb = d.xpos[body]
            vb = (v @ Rg.T + pg - pb) @ Rb
            h = ConvexHull(vb)
            self.pieces.append(dict(v=vb[h.vertices], A=h.equations[:, :3], c=h.equations[:, 3],
                                    hull=trimesh.convex.convex_hull(vb), center=vb[h.vertices].mean(0)))
        allv = np.concatenate([p['v'] for p in self.pieces])
        self.center = allv.mean(0); self.radius = float(np.linalg.norm(allv - self.center, axis=1).max())
        self.pos = np.asarray(pos, np.float64); self.rot = np.asarray(rot, np.float64)
        self.n_faces = sum(len(p['c']) for p in self.pieces)
        ext = allv.max(0) - allv.min(0)
        self.extent = ext

    def local(self, X, t):
        return (X - self.pos[t]) @ self.rot[t]

    def sdf_local(self, q, band=0.05, normals=False):
        """signed distance (object frame) [N]; with normals=True also outward unit normals [N,3] (object frame)."""
        d = np.full(len(q), np.inf); nrm = np.zeros((len(q), 3))
        near = np.linalg.norm(q - self.center, axis=1) < self.radius + band
        if not near.any():
            return (d, nrm) if normals else d
        qn = q[near]; best = np.full(len(qn), np.inf); bn = np.zeros((len(qn), 3))
        for p in self.pieces:
            S_ = qn @ p['A'].T + p['c']
            j = S_.argmax(1); s = S_[np.arange(len(qn)), j]   # <0 inside (exact depth for a convex piece)
            dd = s.copy(); nn = p['A'][j].copy()
            out = (s > 0) & (s < band)
            if out.any():
                cp, dist, _ = trimesh.proximity.closest_point(p['hull'], qn[out])
                dd[out] = dist
                v = qn[out] - cp; ln = np.linalg.norm(v, axis=1, keepdims=True)
                nn[out] = np.where(ln > 1e-9, v / np.maximum(ln, 1e-12), nn[out])
            dd[s >= band] = np.inf
            better = dd < best
            best[better] = dd[better]; bn[better] = nn[better]
        d[near] = best; nrm[near] = bn
        return (d, nrm) if normals else d


def add_object_to_spec(spec, obj):
    b = spec.worldbody.add_body(); b.name = '__qa_obj'; b.mocap = True
    for i, p in enumerate(obj.pieces):
        me = spec.add_mesh(); me.name = f'__qa_obj_m{i}'; me.uservert = p['v'].astype(np.float32).reshape(-1)
        g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = me.name
        g.contype = 0; g.conaffinity = 0; g.group = 3
    return b


# ------------------------------------------------------------------------------------------------ human regions
def human_regions(prov, log):
    txt = Path(log).read_text(errors='replace') if log and Path(log).exists() else ''
    dsm = re.findall(r'correspondence dataset: (\S+\.npz)', txt); slm = re.findall(r'correspondence slots: (\S+\.npz)', txt)
    if dsm and slm:
        dsp, slp = Path(dsm[-1]), Path(slm[-1])
    else:
        tag = 'beta0' if prov.get('shared_corr') else 'smplx'
        base = Path('/home/arenalabs/umr_factory_data/corr') / C.ROBOT / tag
        dsp, slp = base / 'correspondence_dataset.npz', base / 'train' / 'correspondence_slots_final.npz'
    ds = np.load(dsp, allow_pickle=True); sl = np.load(slp, allow_pickle=True)
    segm = pickle.load(open(UMR_ROOT / 'assets' / 'smplx_parts_segm.pkl', 'rb'), encoding='latin1')['segm']
    from scipy.spatial import cKDTree
    pts = ds['points'][0]; face = ds['sample_face_ids_0']
    dist, nn = cKDTree(pts).query(sl['reconstructed_slots'][0])
    part = segm[face[nn]]
    reg = np.array([C.RID[SMPLX_PART_REGION.get(int(p), 'trunk')] for p in part], np.int64)
    return reg, dict(dataset=str(dsp), slots=str(slp), nn_p95_mm=round(float(np.percentile(dist, 95) * 1000), 2))


# ------------------------------------------------------------------------------------------------ measurement
def measure(npz_path, want_points=False, log_every=0):
    o = np.load(npz_path, allow_pickle=True)
    prov = json.loads(str(o['provenance']))
    Q = o['qpos'].astype(np.float64); T = len(Q); fps = float(o['fps'])
    floor = float(o['floor_offset_removed'])
    surf = C.load_surface()
    reg = surf['region'].astype(np.int64)
    has_obj = 'object_pos' in o.files
    spec = mujoco.MjSpec.from_file(str(C.XML))
    obj = None
    if has_obj:
        scale = float(o['human_scale'])
        obj = ConvexObject(str(o['object_xml']), scale, o['object_pos'], o['object_rot'])
        add_object_to_spec(spec, obj)
    m = spec.compile(); d = mujoco.MjData(m)
    assert m.nq == Q.shape[1]
    floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor_g < 0:
        floor_g = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE][0]
    cg = C.robot_collision_geoms(m)
    cg = [g for g in cg if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) != '__qa_obj']
    breg = surf['body_region']
    cg_reg = np.array([breg[m.geom_bodyid[g]] for g in cg])
    pairs = sorted(C.mujoco_considered_pairs(m))
    pairs = [(a, b) for a, b in pairs if a in cg and b in cg]
    og = [g for g in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == '__qa_obj']
    if has_obj:
        mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, '__qa_obj')])
        oq = np.zeros((T, 4))
        for t in range(T):
            mujoco.mju_mat2Quat(oq[t], obj.rot[t].reshape(-1))
    # poser for dense + hull points
    pl = np.concatenate([surf['points_local'], surf['hull_local']]).astype(np.float64)
    pgm = np.concatenate([surf['geom'], surf['hull_geom']]).astype(np.int64)
    groups = [(int(g), np.flatnonzero(pgm == g)) for g in np.unique(pgm)]
    ND = len(reg)
    hreg_hull = breg[m.geom_bodyid[surf['hull_geom']]]
    X = np.zeros((T, ND, 3), np.float32)
    g_dense = np.zeros((T, NR)); g_hull = np.zeros((T, NR)); g_coll = np.full((T, NR), np.inf)
    o_dense = np.full((T, NR), np.inf); o_coll = np.full((T, NR), np.inf)
    self_min = np.full(T, np.inf); self_worst = [None] * T
    Dobj = np.full((T, ND), np.inf, np.float32) if has_obj else None
    Lobj = np.zeros((T, ND, 3), np.float32) if has_obj else None
    Nobj = np.zeros((T, ND, 3), np.float32) if has_obj else None
    rb = m.geom_rbound
    for t in range(T):
        d.qpos[:] = Q[t]
        if has_obj:
            d.mocap_pos[mocap] = obj.pos[t]; d.mocap_quat[mocap] = oq[t]
        mujoco.mj_kinematics(m, d)
        P = np.empty((len(pl), 3))
        for g, k in groups:
            P[k] = pl[k] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        Xd = P[:ND]; Xh = P[ND:]
        X[t] = Xd
        z = Xd[:, 2]
        g_dense[t] = np.full(NR, np.inf); np.minimum.at(g_dense[t], reg, z)
        g_hull[t] = np.full(NR, np.inf); np.minimum.at(g_hull[t], hreg_hull, Xh[:, 2])
        for i, g in enumerate(cg):
            dist = mujoco.mj_geomDistance(m, d, floor_g, g, 0.05, None)
            g_coll[t, cg_reg[i]] = min(g_coll[t, cg_reg[i]], dist)
        smin = np.inf; worst = None
        for a, b in pairs:
            if np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) > rb[a] + rb[b] + 0.01:
                continue
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.01, None)
            if dist < smin:
                smin = dist; worst = (a, b)
        self_min[t] = smin; self_worst[t] = worst
        if has_obj:
            ql = obj.local(Xd, t)
            dd, nn = obj.sdf_local(ql, normals=True)
            Dobj[t] = dd; Lobj[t] = ql; Nobj[t] = nn
            o_dense[t] = np.full(NR, np.inf); np.minimum.at(o_dense[t], reg, dd)
            for i, g in enumerate(cg):
                if np.linalg.norm(d.geom_xpos[g] - obj.pos[t]) > rb[g] + obj.radius + np.linalg.norm(obj.center) + 0.05:
                    continue
                for g2 in og:
                    dist = mujoco.mj_geomDistance(m, d, g, g2, 0.05, None)
                    o_coll[t, cg_reg[i]] = min(o_coll[t, cg_reg[i]], dist)
        if log_every and t % log_every == 0:
            print(f'  measure t={t}/{T}', flush=True)
    res = dict(T=T, fps=fps, floor=floor, X=X, reg=reg, g_dense=g_dense, g_hull=g_hull, g_coll=g_coll,
               self_min=self_min, self_worst=self_worst, has_obj=has_obj, o_dense=o_dense, o_coll=o_coll,
               Dobj=Dobj, Lobj=Lobj, Nobj=Nobj, obj=obj, m=m, prov=prov, npz=o, pairs_checked=len(pairs), cg=cg)
    # ---------------- human
    Hs = np.load(prov['human_slots'])['source_slots']
    umr = np.load(prov['umr_out'], allow_pickle=True)
    if len(Hs) != T:
        Hs = Hs[umr['frame_ids']]
    Hs = Hs.astype(np.float64) - np.array([0, 0, floor])
    hreg, hinfo = human_regions(dict(shared_corr='beta0' in str(prov.get('human_slots', '')) or 'qa_v2__dance' not in str(npz_path)),
                                str(Path(prov['umr_out']).with_suffix('.umr.log')))
    h_g = np.full((T, NR), np.inf)
    for r in range(NR):
        k = hreg == r
        if k.any():
            h_g[:, r] = Hs[:, k, 2].min(1)
    h_o = np.full((T, NR), np.inf)
    Dh = None
    if has_obj:
        Dh = np.full((T, len(hreg)), np.inf, np.float32)
        for t in range(T):
            Dh[t] = obj.sdf_local(obj.local(Hs[t], t))
            h_o[t] = np.full(NR, np.inf); np.minimum.at(h_o[t], hreg, Dh[t])
    res.update(H=Hs.astype(np.float32), hreg=hreg, hinfo=hinfo, h_g=h_g, h_o=h_o, Dh=Dh)
    return res


def contact_stats(res, kind):
    """kind 'ground' | 'object'.  Region-level DATA_SPEC_v2 checks + point slip."""
    S = SPEC; T = res['T']; fps = res['fps']; reg = res['reg']
    if kind == 'ground':
        rc = res['g_dense']                     # robot region clearance (dense points)
        hc = res['h_g']
        Dp = res['X'][..., 2].astype(np.float64)          # point clearance
    else:
        rc = res['o_dense']; hc = res['h_o']; Dp = res['Dobj'].astype(np.float64)
    hcon = drop_short(hysteresis(np.where(np.isfinite(hc), hc, np.inf), S['human_enter'], S['human_leave']), S['min_seg'])
    hoff = hc > S['human_off']
    solid = (rc >= -S['pen_tol']) & (rc <= S['contact_max'])
    solid_strict = (rc >= S['contact_min_strict']) & (rc <= S['contact_max'])
    hover = (rc > S['contact_max']) & (rc <= S['hover_max'])
    far = rc > S['hover_max']
    pen = rc < -S['pen_tol']
    phantom = hoff & (rc < S['robot_off'])
    out = dict(per_region={}, totals={})
    tot = dict(human_contact_region_frames=0, solid=0, solid_strict=0, hover=0, far=0, pen=0)
    for r in range(NR):
        n = int(hcon[:, r].sum())
        if n == 0 and not (rc[:, r] <= S['contact_max']).any():
            continue
        e = dict(human_contact_frames=n,
                 robot_solid_when_human=int((hcon[:, r] & solid[:, r]).sum()),
                 robot_solid_strict_when_human=int((hcon[:, r] & solid_strict[:, r]).sum()),
                 robot_hover_1_20mm_when_human=int((hcon[:, r] & hover[:, r]).sum()),
                 robot_far_gt20mm_when_human=int((hcon[:, r] & far[:, r]).sum()),
                 robot_pen_when_human=int((hcon[:, r] & pen[:, r]).sum()),
                 robot_solid_frames_total=int(solid[:, r].sum()),
                 phantom_contact_frames=int(phantom[:, r].sum()),
                 human_off_frames=int(hoff[:, r].sum()))
        e['solid_frac_when_human'] = round(e['robot_solid_when_human'] / n, 4) if n else None
        if n:
            hv = rc[hcon[:, r], r]
            e['robot_clearance_when_human_mm_p50_p95'] = [round(float(np.percentile(hv, 50) * 1000), 2),
                                                          round(float(np.percentile(hv, 95) * 1000), 2)]
        out['per_region'][C.REGIONS[r]] = e
        tot['human_contact_region_frames'] += n
        tot['solid'] += e['robot_solid_when_human']; tot['solid_strict'] += e['robot_solid_strict_when_human']
        tot['hover'] += e['robot_hover_1_20mm_when_human']; tot['far'] += e['robot_far_gt20mm_when_human']
        tot['pen'] += e['robot_pen_when_human']
    n = max(1, tot['human_contact_region_frames'])
    out['totals'] = dict(tot, solid_frac=round(tot['solid'] / n, 4), solid_strict_frac=round(tot['solid_strict'] / n, 4),
                         hover_frac=round(tot['hover'] / n, 4), far_frac=round(tot['far'] / n, 4),
                         phantom_region_frames=int(phantom.sum()))
    # ---------------- robot contact segments + flicker (hysteresis enter <= 1 mm, leave > 2 mm)
    rcon_raw = solid
    rcon_h = hysteresis(np.where(np.isfinite(rc), rc, np.inf), S['hyst_enter'] + 1e-9, S['hyst_leave'])
    seg_raw = [b - a for r in range(NR) for a, b in runs(rcon_raw[:, r])]
    seg_h = [b - a for r in range(NR) for a, b in runs(rcon_h[:, r])]
    out['robot_contact_segments'] = dict(
        raw_1mm=dict(n=len(seg_raw), short_lt3=int(sum(1 for s in seg_raw if s < 3))),
        hysteresis_1_2mm=dict(n=len(seg_h), short_lt3=int(sum(1 for s in seg_h if s < 3))),
        toggles_per_region_second=round(float(np.abs(np.diff(rcon_raw.astype(int), axis=0)).sum() / max(1, NR) / (T / fps)), 3))
    # ---------------- slip of contact points (point clearance in [-0.5, 1] mm at t-1, t, t+1)
    cp = (Dp >= -S['pen_tol']) & (Dp <= S['contact_max'])
    st = cp.copy(); st[1:] &= cp[:-1]; st[:-1] &= cp[1:]; st[0] = False; st[-1] = False
    if kind == 'ground':
        Xw = res['X'].astype(np.float64)
        V = np.zeros_like(Xw); V[1:-1] = (Xw[2:] - Xw[:-2]) * fps / 2
        vt = np.linalg.norm(V[..., :2], axis=-1)
    else:
        L = res['Lobj'].astype(np.float64)                 # object-frame coordinates
        V = np.zeros_like(L); V[1:-1] = (L[2:] - L[:-2]) * fps / 2
        Nn = res['Nobj'].astype(np.float64)                # outward normal (object frame) at the contact point
        vt = np.linalg.norm(V - (V * Nn).sum(-1, keepdims=True) * Nn, axis=-1)   # tangential relative speed
    sp = vt[st]
    slip = dict(contact_point_frames=int(st.sum()))
    if len(sp):
        slip.update(p50_cms=round(float(np.percentile(sp, 50) * 100), 3), p95_cms=round(float(np.percentile(sp, 95) * 100), 3),
                    max_cms=round(float(sp.max() * 100), 2), frac_gt_1cms=round(float((sp > S['slip_max']).mean()), 4))
    # per frame: any contact point sliding > 1 cm/s
    fr_slip = (st & (vt > S['slip_max'])).any(1)
    slip['frames_with_slipping_contact_point'] = int(fr_slip.sum())
    per_reg = {}
    for r in range(NR):
        k = reg == r
        s = vt[:, k][st[:, k]]
        if len(s):
            per_reg[C.REGIONS[r]] = dict(n=int(len(s)), p50_cms=round(float(np.percentile(s, 50) * 100), 3),
                                         p95_cms=round(float(np.percentile(s, 95) * 100), 3),
                                         frac_gt_1cms=round(float((s > S['slip_max']).mean()), 4))
    slip['per_region'] = per_reg
    out['slip'] = slip
    # frame masks for exclusion accounting
    out['_masks'] = dict(human_contact_not_solid=(hcon & ~solid).any(1), phantom=phantom.any(1), slip=fr_slip,
                         pen=pen.any(1), hcon=hcon, solid=solid, hover=hover, phantom_rf=phantom)
    return out
