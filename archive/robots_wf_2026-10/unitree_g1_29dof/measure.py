"""unitree_g1_29dof -- INDEPENDENT QA measurement of an engine output against DATA_SPEC_v2.

Does not use the engine's gauges (umrf.geometry / labels / postpass). Own implementation:
  robot dense points  : surface.npz points_local posed with MuJoCo FK (geom_xpos/xmat of the staged MJCF)
  exact ground        : every visual-mesh vertex (min of a plane over a mesh = a vertex) + MuJoCo mj_geomDistance
                        between the model's floor plane (z = 0 = calibrated floor; outputs are floor-shifted) and
                        every robot collision geom
  self                : MuJoCo mj_collision (margin 2 cm) + mj_geomDistance on robot collision geoms (parent-child
                        filter only; staged exclusions = none)
  object              : object MJCF loaded with UMR's own mesh-path/scale helpers, scaled by human_scale, posed by
                        the npz object_pos/object_rot; (a) robot collision geoms vs object collision geoms via
                        mj_geomDistance (object = mocap body), (b) dense points vs convex pieces (half-space SDF inside,
                        exact closest point outside within 3 cm)
  human               : UMR's captured 4096 source slots (retarget frame, floor-shifted), region per slot from the
                        correspondence (the only engine helper used: umrf.pipeline.human_slot_regions = data mapping)
Rules (DATA_SPEC_v2): solid contact = region dense min in [-0.5, 1.0] mm (>=1 dense point); penetration <= 0.5 mm
(dense, mesh vertices, collision); slip = contact-point tangential speed < 1 cm/s (points touching at t and t+1,
raw finite difference, world frame / object frame); flicker = robot hysteresis (enter <= 1 mm, leave > 2 mm) segments
< 3 frames; human consistency = human region in contact (enter 1 cm / leave 2 cm, >= 3 frames) -> robot solid, human
region > 2 cm -> robot region > 5 mm.
Usage: <umr env python> measure.py <engine npz> <qa_clip_id>
Writes qa/unitree_g1_29dof/measure_<clip>.json and pipeline/out/<clip>_status.npz (per-point colours for the renderer).
"""
import json, sys, time, xml.etree.ElementTree as ET, tempfile
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

ROBOT = 'unitree_g1_29dof'
DATA = Path('/home/arenalabs/umr_factory_data')
PD = DATA / 'robots' / ROBOT / 'pipeline'
QA = DATA / 'qa' / ROBOT; QA.mkdir(parents=True, exist_ok=True)
CFG = json.load(open(PD / 'pipeline.json'))
TH = CFG['thresholds']
PEN, CON = TH['penetration_max_m'], TH['contact_solid_m'][1]
T_ = mujoco.mjtGeom

npz_path, clip = sys.argv[1], sys.argv[2]
RAW = '--raw' in sys.argv[3:]          # control arm: measure UMR's raw output (qpos_umr) with the same ruler
o = np.load(npz_path, allow_pickle=True)
prov = json.loads(str(o['provenance']))
S = np.load(DATA / 'robots' / ROBOT / 'surface.npz')
REG = [str(r) for r in S['regions']]; NR = len(REG)
preg = S['region'].astype(int); pl = S['points_local'].astype(float); pg = S['geom'].astype(int)
patches = np.load(PD / 'assets' / 'patches.npz')
Q = (o['qpos_umr'] if RAW else o['qpos']).astype(float); T = len(Q)
fps = float(o['fps'])
if RAW:
    clip = clip + '_umrraw'
has_obj = 'object_pos' in o.files
cap_g = set(CFG['contact_capable']['ground']); cap_o = set(CFG['contact_capable']['object'])

# ------------------------------------------------------------------ model (+ object as mocap body)
xml = CFG['assets']['mjcf']
spec = mujoco.MjSpec.from_file(xml)
obj_pieces = []                                       # (verts_local [n,3]) convex collision pieces, object frame
if has_obj:
    sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
    import retarget_smpl_to_humanoid_surface_vector as rs
    oxml = str(o['object_xml'])
    root = ET.parse(oxml).getroot()
    rs.make_mjcf_mesh_paths_absolute(root, Path(oxml))
    rs.scale_mjcf_mesh_assets(root, float(o['human_scale']))
    with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False, dir=str(PD / 'out')) as f:
        f.write(ET.tostring(root, encoding='unicode')); tmp = f.name
    om = mujoco.MjModel.from_xml_path(tmp); Path(tmp).unlink()
    od = mujoco.MjData(om); mujoco.mj_forward(om, od)
    allp, colp = [], []
    for g in range(om.ngeom):
        if om.geom_type[g] != T_.mjGEOM_MESH:
            continue
        mi = om.geom_dataid[g]
        v = om.mesh_vert[om.mesh_vertadr[mi]:om.mesh_vertadr[mi] + om.mesh_vertnum[mi]].astype(float)
        v = v @ od.geom_xmat[g].reshape(3, 3).T + od.geom_xpos[g]
        allp.append(v)
        if om.geom_contype[g] or om.geom_conaffinity[g]:
            colp.append(v)
    obj_pieces = colp if colp else allp
    b = spec.worldbody.add_body(); b.name = 'qa_object'; b.mocap = True
    for i, v in enumerate(obj_pieces):
        ms = spec.add_mesh(); ms.name = f'qa_obj_m{i}'; ms.uservert = v.astype(np.float32).ravel()
        g = b.add_geom(); g.type = T_.mjGEOM_MESH; g.meshname = ms.name; g.contype = 0; g.conaffinity = 0
m = spec.compile(); d = mujoco.MjData(m)
assert m.nq == Q.shape[1]
FLOOR = 0
obody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'qa_object')
ogeoms = [g for g in range(m.ngeom) if obody >= 0 and m.geom_bodyid[g] == obody]
rcol = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, obody) and (m.geom_contype[g] or m.geom_conaffinity[g])]
rvis = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, obody) and m.geom_group[g] == 1 and m.geom_type[g] == T_.mjGEOM_MESH]
breg = S['body_region']
rcol_reg = np.array([breg[m.geom_bodyid[g]] for g in rcol])
vverts = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(float) for g in rvis}
vreg = {g: int(breg[m.geom_bodyid[g]]) for g in rvis}
hull_eq = [ConvexHull(v).equations for v in obj_pieces]
hull_mesh = [trimesh.convex.convex_hull(v) for v in obj_pieces]
if has_obj:
    opos = o['object_pos'].astype(float); orot = o['object_rot'].astype(float)
    mocap = int(m.body_mocapid[obody])


def obj_sdf(Pw, t):
    """Signed distance of world points to the union of convex pieces (exact inside; exact outside within 3 cm)."""
    pl_ = (Pw - opos[t]) @ orot[t]                              # R^T (p - x)
    best = np.full(len(Pw), np.inf)
    for eq, hm in zip(hull_eq, hull_mesh):
        s = (pl_ @ eq[:, :3].T + eq[:, 3]).max(1)               # <= true distance outside, exact inside
        near = (s > 0) & (s < 0.03)
        if near.any():
            cp, dist, _ = trimesh.proximity.closest_point(hm, pl_[near])
            s[near] = dist
        best = np.minimum(best, s)
    return best, pl_


# ------------------------------------------------------------------ per-frame measurement
X = np.zeros((T, len(pl), 3), np.float32)
g_dense = np.full((T, NR), np.inf); g_exact = np.full((T, NR), np.inf)
g_vis_min = np.zeros(T); g_col_min = np.zeros(T); g_dense_min = np.zeros(T)
self_min = np.full(T, np.inf); self_pairs = {}
o_dense = np.full((T, NR), np.inf); o_col_reg = np.full((T, NR), np.inf); o_min_dense = np.full(T, np.inf); o_min_col = np.full(T, np.inf)
OD = np.full((T, len(pl)), np.inf, np.float32); OL = np.zeros((T, len(pl), 3), np.float32)
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
ft = np.zeros(6)
t0 = time.time()
for t in range(T):
    d.qpos[:] = Q[t]
    if has_obj:
        q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, orot[t].ravel())
        d.mocap_pos[mocap] = opos[t]; d.mocap_quat[mocap] = q4
    mujoco.mj_kinematics(m, d)
    Rm = d.geom_xmat[pg].reshape(-1, 3, 3)
    x = np.einsum('nij,nj->ni', Rm, pl) + d.geom_xpos[pg]
    X[t] = x
    z = x[:, 2]
    np.minimum.at(g_dense[t], preg, z)
    g_dense_min[t] = z.min()
    ge = g_dense[t].copy()
    vmin = np.inf
    for g in rvis:
        vz = (vverts[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()
        ge[vreg[g]] = min(ge[vreg[g]], vz); vmin = min(vmin, vz)
    g_vis_min[t] = vmin
    cmin = np.inf
    for g, r in zip(rcol, rcol_reg):
        dist = mujoco.mj_geomDistance(m, d, FLOOR, g, 0.05, ft)
        ge[r] = min(ge[r], dist); cmin = min(cmin, dist)
    g_col_min[t] = cmin; g_exact[t] = ge
    # self + object collision pairs
    saved = m.geom_margin.copy(); m.geom_margin[:] = 0.02
    mujoco.mj_collision(m, d); m.geom_margin[:] = saved
    rset = set(rcol); oset = set(ogeoms)
    for i in range(d.ncon):
        c = d.contact[i]; a, b_ = int(c.geom1), int(c.geom2)
        if a in rset and b_ in rset:
            dist = mujoco.mj_geomDistance(m, d, a, b_, 0.02, ft)
            if dist < self_min[t]:
                self_min[t] = dist
            if dist < -PEN:
                k = tuple(sorted((bn(int(m.geom_bodyid[a])), bn(int(m.geom_bodyid[b_])))))
                self_pairs[k] = self_pairs.get(k, 0) + 1
    if has_obj:
        for g, r in zip(rcol, rcol_reg):
            for og in ogeoms:
                dist = mujoco.mj_geomDistance(m, d, g, og, 0.05, ft)
                o_col_reg[t, r] = min(o_col_reg[t, r], dist); o_min_col[t] = min(o_min_col[t], dist)
        # dense points near the object only
        near = np.linalg.norm(x - opos[t], axis=1) < 1.0
        sd = np.full(len(x), np.inf); lp = np.zeros_like(x)
        if near.any():
            sd[near], lp[near] = obj_sdf(x[near], t)
        OD[t] = sd; OL[t] = lp
        np.minimum.at(o_dense[t], preg, sd)
        o_min_dense[t] = sd.min()
print(f'[measure] {clip}: FK+gauges {time.time() - t0:.0f} s', flush=True)

# ------------------------------------------------------------------ human
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != T:
    Hs = Hs[umr['frame_ids']]
Hs = Hs.astype(float) - np.array([0, 0, float(o['floor_offset_removed'])])
hreg, _ = P.human_slot_regions(ROBOT, prov['source_kind'], prov['source_data'], prov['seq_key'],
                               str(Path(prov['umr_out']).with_suffix('.umr.log')), 'beta0' in str(prov.get('umr_out', '')) or
                               prov['source_kind'] == 'hoi')
h_g = np.full((T, NR), np.inf); np.minimum.at(h_g.T, hreg, Hs[..., 2].T) if False else None
for r in range(NR):
    k = hreg == r
    if k.any():
        h_g[:, r] = Hs[:, k, 2].min(1)
h_o = np.full((T, NR), np.inf); HOD = np.full((T, Hs.shape[1]), np.inf)
if has_obj:
    for t in range(T):
        near = np.linalg.norm(Hs[t] - opos[t], axis=1) < 1.0
        if near.any():
            HOD[t, near] = obj_sdf(Hs[t][near], t)[0]
        for r in range(NR):
            k = hreg == r
            h_o[t, r] = HOD[t, k].min()


def hyst(c, enter, leave):
    out = np.zeros_like(c, bool); on = np.zeros(c.shape[1], bool)
    for t in range(len(c)):
        on = np.where(c[t] < enter, True, np.where(c[t] > leave, False, on)); out[t] = on
    return out


def runs(col):
    segs = []; s = None
    for t, v in enumerate(col):
        if v and s is None: s = t
        if not v and s is not None: segs.append((s, t)); s = None
    if s is not None: segs.append((s, len(col)))
    return segs


def drop_short(c, n):
    c = c.copy()
    for r in range(c.shape[1]):
        for a, b in runs(c[:, r]):
            if b - a < n: c[a:b, r] = False
    return c


# human contact label (engine-equivalent ruler for the human: enter 1 cm / leave 2 cm, >= 3 frames)
hc_g = drop_short(hyst(h_g, TH['human_contact_enter_m'], TH['human_contact_leave_m']), 3)
hc_o = drop_short(hyst(h_o, TH['human_contact_enter_m'], TH['human_contact_leave_m']), 3) if has_obj else np.zeros_like(hc_g)


# ------------------------------------------------------------------ rules
def solid(c):
    return (c >= -PEN) & (c <= CON)


def contact_stats(rob, hum_c, hum_clear, cap, partner):
    cols = [r for r in range(NR) if REG[r] in cap]
    out = dict(per_region={})
    tot = dict(human_contact=0, solid=0, hover_1_20mm=0, far_gt20mm=0, pen=0, release_viol=0, human_released=0)
    for r in cols:
        hc = hum_c[:, r]
        n = int(hc.sum())
        rel = hum_clear[:, r] > TH['human_release_m']
        rv = rel & (rob[:, r] < TH['robot_release_required_m'])
        rec = dict(human_contact_frames=n,
                   solid=int((hc & solid(rob[:, r])).sum()),
                   hover_1_20mm=int((hc & (rob[:, r] > CON) & (rob[:, r] <= 0.02)).sum()),
                   far_gt20mm=int((hc & (rob[:, r] > 0.02)).sum()),
                   pen=int((hc & (rob[:, r] < -PEN)).sum()),
                   human_released_frames=int(rel.sum()), release_violation=int(rv.sum()),
                   robot_solid_frames_total=int(solid(rob[:, r]).sum()))
        if n:
            rec['solid_frac'] = round(rec['solid'] / n, 4)
            gap = rob[hc, r]
            rec['robot_clearance_mm_when_human_contact_p50_p95'] = [round(float(np.percentile(gap, 50) * 1000), 2),
                                                                    round(float(np.percentile(gap, 95) * 1000), 2)]
        if n or rec['robot_solid_frames_total'] or rec['release_violation']:
            out['per_region'][REG[r]] = rec
        tot['human_contact'] += n; tot['solid'] += rec['solid']; tot['hover_1_20mm'] += rec['hover_1_20mm']
        tot['far_gt20mm'] += rec['far_gt20mm']; tot['pen'] += rec['pen']
        tot['release_viol'] += rec['release_violation']; tot['human_released'] += rec['human_released_frames']
    tot['solid_frac'] = round(tot['solid'] / max(1, tot['human_contact']), 4)
    tot['hover_frac'] = round(tot['hover_1_20mm'] / max(1, tot['human_contact']), 4)
    out['total'] = tot
    return out


def slip_stats(rob_clear, dist_pts, pos_pts, cap):
    """Contact-point speed: dense points of a solid-contact region touching (|d| in [-0.5, 1] mm) at t and t+1."""
    sp = np.full((T - 1, NR), np.nan)
    for r in range(NR):
        if REG[r] not in cap:
            continue
        k = np.flatnonzero(preg == r)
        dd = dist_pts[:, k]
        touch = (dd >= -PEN) & (dd <= CON)
        both = touch[:-1] & touch[1:]
        v = np.linalg.norm(np.diff(pos_pts[:, k], axis=0), axis=-1) * fps
        ok = both.any(1)
        if ok.any():
            sp[ok, r] = np.nanmedian(np.where(both[ok], v[ok], np.nan), axis=1)
    vals = sp[np.isfinite(sp)]
    per = {}
    for r in range(NR):
        s = sp[:, r][np.isfinite(sp[:, r])]
        if len(s):
            per[REG[r]] = dict(frames=int(len(s)), p50_cms=round(float(np.median(s) * 100), 2),
                               p95_cms=round(float(np.percentile(s, 95) * 100), 2),
                               frac_gt_1cms=round(float((s > TH['slip_max_mps']).mean()), 4))
    tot = dict(region_frames=int(len(vals)))
    if len(vals):
        tot.update(p50_cms=round(float(np.median(vals) * 100), 2), p95_cms=round(float(np.percentile(vals, 95) * 100), 2),
                   max_cms=round(float(vals.max() * 100), 2), frac_gt_1cms=round(float((vals > TH['slip_max_mps']).mean()), 4))
    return dict(total=tot, per_region=per), sp


def flicker_stats(rob, cap):
    c = hyst(rob, TH['hysteresis_enter_m'] + 1e-12, TH['hysteresis_leave_m'])
    c &= rob >= -1.0
    per = {}; short = 0; segs = 0
    for r in range(NR):
        if REG[r] not in cap:
            continue
        rr = runs(c[:, r]); s_ = [b - a for a, b in rr if b - a < TH['min_segment_frames']]
        if rr:
            per[REG[r]] = dict(segments=len(rr), short_lt3=len(s_), toggles_per_s=round(2 * len(rr) / (T / fps), 2))
        short += len(s_); segs += len(rr)
    return dict(segments=segs, short_lt3=short, per_region=per), c


res = dict(robot=ROBOT, clip=clip, measured='qpos_umr (raw UMR, control arm)' if RAW else 'qpos (engine post-pass output)', engine_npz=npz_path, frames=T, fps=fps, has_object=bool(has_obj),
           floor=dict(floor_height=float(o['floor_height']), floor_offset_removed=float(o['floor_offset_removed']),
                      robot_dense_min_over_clip_mm=round(float(g_dense_min.min() * 1000), 3)))
# penetration
gpen = np.maximum(0, -np.minimum(np.minimum(g_dense_min, g_vis_min), g_col_min))
res['penetration'] = dict(
    ground_max_mm=round(float(gpen.max() * 1000), 3), ground_frames_gt_0p5mm=int((gpen > PEN).sum()),
    ground_dense_min_mm=round(float(g_dense_min.min() * 1000), 3), ground_mesh_vertex_min_mm=round(float(g_vis_min.min() * 1000), 3),
    ground_collision_min_mm=round(float(g_col_min.min() * 1000), 3),
    self_max_mm=round(float(max(0, -self_min.min()) * 1000), 3) if np.isfinite(self_min).any() else 0.0,
    self_frames_gt_0p5mm=int((self_min < -PEN).sum()),
    self_pairs_gt_0p5mm=sorted([[*k, v] for k, v in self_pairs.items()], key=lambda r: -r[2])[:8])
if has_obj:
    omin = np.minimum(o_min_dense, o_min_col)
    res['penetration'].update(object_max_mm=round(float(max(0, -omin.min()) * 1000), 3),
                              object_frames_gt_0p5mm=int((omin < -PEN).sum()),
                              object_dense_min_mm=round(float(o_min_dense.min() * 1000), 3),
                              object_collision_min_mm=round(float(o_min_col.min() * 1000), 3))
# ground contact vs human
res['ground_contact'] = contact_stats(g_dense, hc_g, h_g, cap_g, 'ground')
res['ground_contact_exact_gauge'] = contact_stats(g_exact, hc_g, h_g, cap_g, 'ground')['total']
res['ground_slip'], sp_g = slip_stats(g_dense, X[..., 2], X[..., :2], cap_g)
res['ground_flicker'], rc_g = flicker_stats(g_dense, cap_g)
# foot contact on disallowed points (instep / ankle link)
fu = np.concatenate([patches['L_foot_upper'], patches['R_foot_upper']])
res['foot_upper_points_touching_frames'] = int(((X[:, fu, 2] <= CON)).any(1).sum())
# dense vs exact gap on solid robot ground contacts
gap = (g_dense - g_exact)[solid(g_exact)]
res['dense_minus_exact_mm_on_contacts_p50_p95_max'] = [round(float(np.percentile(gap, q) * 1000), 2) for q in (50, 95, 100)] if len(gap) else None
if has_obj:
    oc = np.minimum(o_dense, np.where(np.isfinite(o_col_reg), o_col_reg, np.inf))
    res['object_contact'] = contact_stats(o_dense, hc_o, h_o, cap_o, 'object')
    res['object_contact_with_collision_geoms'] = contact_stats(oc, hc_o, h_o, cap_o, 'object')['total']
    res['object_slip'], sp_o = slip_stats(o_dense, OD, OL, cap_o)
    res['object_flicker'], rc_o = flicker_stats(o_dense, cap_o)
# engine labels vs independent
eg = o['ground_contact']
res['engine_vs_independent'] = dict(
    engine_flagged_frames=int((o['flags'] > 0).sum()),
    engine_flags_hist={str(b): int(((o['flags'] & b) > 0).sum()) for b in (1, 2, 4, 8, 16)},
    engine_ground_contact_region_frames=int(eg.sum()),
    engine_contact_frames_not_solid_independent=int((eg & ~solid(g_dense)).sum()),
    engine_contact_frames_not_solid_exact=int((eg & ~solid(g_exact)).sum()),
    human_label_agreement_engine_vs_mine=round(float((o['human_ground_contact'] == hc_g).mean()), 4))
# per-frame v2 exclusion mask
bad = np.zeros(T, bool)
why = {}
masks = {}
def add(name, mask):
    global bad
    why[name] = int(mask.sum()); bad |= mask; masks[name] = mask.copy()
add('ground_pen', gpen > PEN)
add('self_pen', self_min < -PEN)
if has_obj:
    add('object_pen', np.minimum(o_min_dense, o_min_col) < -PEN)
capg = np.array([REG[r] in cap_g for r in range(NR)])
add('human_ground_contact_not_solid', (hc_g & ~solid(g_dense) & capg).any(1))
add('ground_release_violation', ((h_g > TH['human_release_m']) & (g_dense < TH['robot_release_required_m']) & capg).any(1))
slipm = np.zeros(T, bool); slipm[:-1] |= np.nan_to_num(sp_g, nan=0).max(1) > TH['slip_max_mps']
add('ground_slip_gt_1cms', slipm)
if has_obj:
    capo = np.array([REG[r] in cap_o for r in range(NR)])
    add('human_object_contact_not_solid', (hc_o & ~solid(o_dense) & capo).any(1))
    so = np.zeros(T, bool); so[:-1] |= np.nan_to_num(sp_o, nan=0).max(1) > TH['slip_max_mps']
    add('object_slip_gt_1cms', so)
eflag = o['flags'] > 0
res['v2_exclusion'] = dict(frames_failing_any_rule=int(bad.sum()), frac=round(float(bad.mean()), 4), by_rule=why,
                           frames_passing_all=int((~bad).sum()),
                           engine_flagged=int(eflag.sum()),
                           failing_but_not_flagged_by_engine=int((bad & ~eflag).sum()),
                           failing_but_not_flagged_by_rule={k: int((v & ~eflag).sum()) for k, v in masks.items()},
                           flagged_by_engine_but_passing=int((eflag & ~bad).sum()))
res['seconds'] = round(time.time() - t0, 1)
json.dump(res, open(QA / f'measure_{clip}.json', 'w'), indent=1, default=float)

# per-point status for the renderer: 0 none, 1 ground contact, 2 object contact, 3 penetration, 4 hover (human in contact)
stt = np.zeros((T, len(pl)), np.uint8)
z = X[..., 2]
stt[(z >= -PEN) & (z <= CON)] = 1
if has_obj:
    stt[(OD >= -PEN) & (OD <= CON)] = 2
hov = hc_g & (g_dense > CON) & (g_dense <= 0.02)
stt[(hov[:, preg]) & (z <= 0.02) & (stt == 0)] = 4
if has_obj:
    hovo = hc_o & (o_dense > CON) & (o_dense <= 0.02)
    stt[(hovo[:, preg]) & (OD <= 0.02) & (stt == 0)] = 4
stt[z < -PEN] = 3
if has_obj:
    stt[OD < -PEN] = 3
hst = np.zeros((T, Hs.shape[1]), np.uint8)
hst[hc_g[:, hreg] & (Hs[..., 2] <= 0.02)] = 1
if has_obj:
    hst[hc_o[:, hreg] & (HOD <= 0.02)] = 2
np.savez_compressed(PD / 'out' / f'{clip}_status.npz', robot=stt, human=hst, hreg=hreg, bad=bad, hc_g=hc_g, hc_o=hc_o,
                    **{'rule_' + k: v for k, v in masks.items()})
print(json.dumps({k: res[k] for k in ('penetration', 'v2_exclusion')}, indent=1))
print(json.dumps(res['ground_contact']['total']), json.dumps(res['ground_slip']['total']), json.dumps(res['ground_flicker']['segments']))
if has_obj:
    print(json.dumps(res['object_contact']['total']), json.dumps(res['object_slip']['total']))
