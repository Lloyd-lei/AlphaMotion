"""unitree_h2 dedicated pipeline -- INDEPENDENT measurement of one QA output against DATA_SPEC_v2.

Does not use the engine's gauges/labels: robot geometry = surface.npz dense points posed by MuJoCo FK of the output
qpos (+ every visual-mesh vertex for penetration) and MuJoCo collision geoms (mj_geomDistance vs a z=0 plane, the
object's collision pieces, and every non-parent-child robot pair).  Object = its own MJCF re-read here (mesh scaled
by the clip's smpl_scale), signed distance = convex pieces (plane equations inside, exact closest point outside).
Human = captured UMR source slots (same floor), regions from the engine's slot->SMPL-X part map (a map, not a gauge).
Contact patches from pipeline/contact_patches.npz.
Usage: <umr python> measure_h2.py <cid>    -> qa/unitree_h2/<cid>.measure.json (+ .perframe.npz)"""
import json, os, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import mujoco, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2common as H2

ROBOT = 'unitree_h2'
DATA = Path('/home/arenalabs/umr_factory_data')
RD = DATA / 'robots' / ROBOT; PD = RD / 'pipeline'; QD = DATA / 'qa' / ROBOT
QD.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(PD / 'pipeline.json')); TH = cfg['thresholds']
CT, PT, SLIP = TH['contact_max_m'], TH['penetration_tol_m'], TH['slip_speed_max_m_s']
cid = sys.argv[1]
RAW = '--raw' in sys.argv          # control arm: measure the raw UMR pose (qpos_umr, same floor shift) instead of the post-pass output
SUF = '.raw' if RAW else ''
o = np.load(PD / 'out' / cid / f'{ROBOT}.npz', allow_pickle=True)
prov = json.loads(str(o['provenance']))
surf = dict(np.load(RD / 'surface.npz', allow_pickle=True))
cp = dict(np.load(PD / 'contact_patches.npz', allow_pickle=True))
REG = [str(r) for r in surf['regions']]; NR = len(REG)
preg = cp['patch_region'].astype(int)                     # contact patch region per dense point (-1 = never contact)
sreg = surf['region'].astype(int)                         # surface region (engine definition) per point
Q = o['qpos_umr' if RAW else 'qpos'].astype(np.float64); T = len(Q); fps = float(o['fps'])
floor = float(o['floor_offset_removed']); hscale = float(o['human_scale'])
has_obj = 'object_pos' in o.files

# ------------------------------------------------------------------ model: robot + floor plane (+ object pieces, mocap)
spec = mujoco.MjSpec.from_file(str(RD / f'{ROBOT}.xml'))
fl = spec.worldbody.add_geom(); fl.name = '__floor'; fl.type = mujoco.mjtGeom.mjGEOM_PLANE; fl.size = [0, 0, 1]
fl.contype = 0; fl.conaffinity = 0
m0 = mujoco.MjModel.from_xml_path(str(RD / f'{ROBOT}.xml'))
# diagnostic visual hulls for collision-less links and the knee (cylinder collision is thinner than the visual shin)
DIAG_BODIES = ['left_hip_pitch_link', 'right_hip_pitch_link', 'left_wrist_roll_link', 'right_wrist_roll_link',
               'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_knee_link', 'right_knee_link']
for b in spec.bodies:
    if b.name in DIAG_BODIES:
        for g in list(b.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.contype == 0 and g.conaffinity == 0:
                ng = b.add_geom(); ng.name = f'__diag_{b.name}'; ng.type = mujoco.mjtGeom.mjGEOM_MESH
                ng.meshname = g.meshname; ng.pos = g.pos; ng.quat = g.quat; ng.contype = 0; ng.conaffinity = 0; ng.group = 5
                break
obj_pieces = []
if has_obj:
    obj_pieces, _ = H2.load_object_pieces(str(o['object_xml']), hscale)
    ob = spec.worldbody.add_body(); ob.name = '__obj'; ob.mocap = True
    for i, p in enumerate(obj_pieces):
        me = spec.add_mesh(); me.name = f'__objm{i}'; me.uservert = p['v'].astype(np.float32).reshape(-1)
        g = ob.add_geom(); g.name = f'__objg{i}'; g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = me.name
        g.contype = 0; g.conaffinity = 0
    OPOS = o['object_pos'].astype(np.float64); OROT = o['object_rot'].astype(np.float64)
m = spec.compile(); d = mujoco.MjData(m)
assert m.nq == Q.shape[1]
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, '__floor')
robot_body = lambda b: 0 < b < m0.nbody
# all ids from the compiled model m (the floor geom added to the worldbody shifts every geom id by one vs m0)
rgeoms = [g for g in range(m.ngeom) if robot_body(m.geom_bodyid[g]) and not gname(g).startswith('__')]
rcoll = [g for g in rgeoms if m.geom_contype[g] or m.geom_conaffinity[g]]
diag = [g for g in range(m.ngeom) if gname(g).startswith('__diag_')]
objg = [g for g in range(m.ngeom) if gname(g).startswith('__objg')]
vis_mesh = [g for g in rgeoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and not (m.geom_contype[g] or m.geom_conaffinity[g])]
vis_v = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(np.float64)
         for g in vis_mesh}
assert len(rcoll) == 23 and len(vis_mesh) == 34, (len(rcoll), len(vis_mesh))


def adjacent(b1, b2):
    return b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1


self_pairs = [(g1, g2) for i, g1 in enumerate(rcoll) for g2 in rcoll[i + 1:]
              if not adjacent(m.geom_bodyid[g1], m.geom_bodyid[g2])]
diag_pairs = [(g1, g2) for g1 in diag for g2 in rcoll + diag
              if g1 != g2 and not adjacent(m.geom_bodyid[g1], m.geom_bodyid[g2])]
diag_pairs = list({tuple(sorted(p)) for p in diag_pairs})
pl = surf['points_local'].astype(np.float64)
g0 = surf['geom'].astype(int)
# map m0 geom id -> m geom id (same body, same order within body)
m0_by_body = {}
for g in range(m0.ngeom):
    m0_by_body.setdefault(int(m0.geom_bodyid[g]), []).append(g)
m_by_body = {}
for g in rgeoms:
    m_by_body.setdefault(int(m.geom_bodyid[g]), []).append(g)
gmap = {g: m_by_body[b][i] for b, gs in m0_by_body.items() if b > 0 for i, g in enumerate(gs)}
pg = np.array([gmap[g] for g in g0])
assert all(m.geom_type[gmap[g]] == m0.geom_type[g] and m.geom_dataid[gmap[g]] == m0.geom_dataid[g] for g in gmap)
ft = np.zeros(6)


def obj_sdf(Xw, t, cull=0.03):
    return H2.obj_sdf(obj_pieces, OPOS[t], OROT[t], Xw, cull)


# ------------------------------------------------------------------ per-frame robot measurement
X = np.zeros((T, len(pl), 3), np.float32)
g_vis_min = np.zeros(T); g_coll_min = np.zeros(T); g_coll_arg = np.zeros(T, int)
self_min = np.full(T, np.inf); self_arg = [None] * T
diag_min = np.full(T, np.inf); diag_arg = [None] * T
o_pt = None
if has_obj:
    o_pt = np.full((T, len(pl)), np.inf, np.float32); o_nrm = np.zeros((T, len(pl), 3), np.float32)
    o_loc = np.zeros((T, len(pl), 3), np.float32); o_vis_min = np.full(T, np.inf); o_coll_min = np.full(T, np.inf)
t0 = time.time()
for t in range(T):
    d.qpos[:] = Q[t]
    if has_obj:
        d.mocap_pos[0] = OPOS[t]; qq = np.zeros(4); mujoco.mju_mat2Quat(qq, OROT[t].reshape(-1)); d.mocap_quat[0] = qq
    mujoco.mj_kinematics(m, d)
    Xt = np.einsum('nij,nj->ni', d.geom_xmat[pg].reshape(-1, 3, 3), pl) + d.geom_xpos[pg]
    X[t] = Xt
    vz = min(float((vis_v[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()) for g in vis_mesh)
    g_vis_min[t] = min(vz, float(Xt[:, 2].min()))
    cd = [mujoco.mj_geomDistance(m, d, floor_g, g, 0.05, ft) for g in rcoll]
    g_coll_min[t] = min(cd); g_coll_arg[t] = rcoll[int(np.argmin(cd))]
    for g1, g2 in self_pairs:
        dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.01, ft)
        if dist < self_min[t]:
            self_min[t] = dist; self_arg[t] = (g1, g2)
    if has_obj:
        dd, nn, qq_ = obj_sdf(Xt, t, cull=0.15)            # wide band: report how far a missing contact is
        o_pt[t] = dd; o_nrm[t] = nn; o_loc[t] = qq_
        vv = np.concatenate([vis_v[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in vis_mesh])
        o_vis_min[t] = min(float(np.min(obj_sdf(vv, t)[0])), float(dd.min()))
        o_coll_min[t] = min(mujoco.mj_geomDistance(m, d, g1, g2, 0.05, ft) for g1 in rcoll for g2 in objg)
t_meas = time.time() - t0
# diag baseline: pairs already overlapping at rest / T-pose are construction overlaps, not motion penetration
d.qpos[:] = m0.qpos0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_kinematics(m, d)
diag_rest = {p: mujoco.mj_geomDistance(m, d, p[0], p[1], 0.01, ft) for p in diag_pairs}


# ------------------------------------------------------------------ region clearances
def reg_min(D, regof):
    out = np.full((T, NR), np.inf)
    for r in range(NR):
        idx = np.flatnonzero(regof == r)
        if len(idx):
            out[:, r] = D[:, idx].min(1)
    return out


zr = X[..., 2].astype(np.float64)
cg_patch = reg_min(zr, preg)                     # ground clearance per contact patch
cg_surf = reg_min(zr, sreg)                      # same with the engine's surface regions
co_patch = reg_min(o_pt.astype(np.float64), preg) if has_obj else None
co_surf = reg_min(o_pt.astype(np.float64), sreg) if has_obj else None
# ------------------------------------------------------------------ human
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P                  # only for the slot -> SMPL-X part region map
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != T:
    fid = umr['frame_ids']; Hs = Hs[fid - (fid[0] if fid.max() >= len(Hs) else 0)]
Hs = Hs.astype(np.float64) - np.array([0, 0, floor])
hreg, hinfo = P.human_slot_regions(ROBOT, prov['source_kind'], prov['source_data'], prov['seq_key'],
                                   str(Path(prov['umr_out']).with_suffix('.umr.log')),
                                   shared=('beta0' in open(Path(prov['umr_out']).with_suffix('.umr.log')).read()))
hg = reg_min(Hs[..., 2], hreg)
ho = None
if has_obj:
    ho_pt = np.stack([obj_sdf(Hs[t], t, cull=0.05)[0] for t in range(T)])
    ho = reg_min(ho_pt, hreg)


def hyst(c, enter, leave, min_on=3):
    on = np.zeros_like(c, bool)
    for r in range(c.shape[1]):
        s = False
        for t in range(c.shape[0]):
            s = (c[t, r] < enter) if not s else not (c[t, r] > leave)
            on[t, r] = s
    return drop_short(on, min_on)


def segs(col):
    col = np.asarray(col, bool); out = []; t = 0
    while t < len(col):
        if col[t]:
            a = t
            while t < len(col) and col[t]:
                t += 1
            out.append((a, t))
        else:
            t += 1
    return out


def drop_short(on, min_on):
    on = on.copy()
    for r in range(on.shape[1]):
        for a, b in segs(on[:, r]):
            if b - a < min_on:
                on[a:b, r] = False
    return on


H_g = hyst(hg, TH['human_contact_enter_m'], TH['human_contact_leave_m'])
H_o = hyst(ho, TH['human_contact_enter_m'], TH['human_contact_leave_m']) if has_obj else None


# ------------------------------------------------------------------ rules
def contact_eval(c, Hc, hc, kind):
    """c robot patch clearance [T,R]; Hc human contact [T,R]; hc human clearance [T,R]."""
    solid = (c >= 0.0) & (c <= CT)
    solid_tol = (c >= -PT) & (c <= CT)
    hover = (c > CT) & (c <= 0.02)
    res = dict(human_contact_region_frames=int(Hc.sum()),
               solid_strict_0_1mm=int((Hc & solid).sum()), solid_tol_m0p5_1mm=int((Hc & solid_tol).sum()),
               hover_1_20mm=int((Hc & hover).sum()), far_gt_20mm=int((Hc & (c > 0.02)).sum()),
               below_m0p5mm=int((Hc & (c < -PT)).sum()))
    n = max(1, res['human_contact_region_frames'])
    res['solid_rate_strict'] = round(res['solid_strict_0_1mm'] / n, 4) if Hc.any() else None
    res['solid_rate_tol'] = round(res['solid_tol_m0p5_1mm'] / n, 4) if Hc.any() else None
    res['hover_rate'] = round(res['hover_1_20mm'] / n, 4) if Hc.any() else None
    res['hover_frames'] = int((Hc & hover).any(1).sum())
    sep = (hc > TH['human_separated_m'])
    res['separation_violations_region_frames'] = int((sep & (c <= TH['robot_must_separate_m'])).sum())
    res['separation_violation_frames'] = int((sep & (c <= TH['robot_must_separate_m'])).any(1).sum())
    per = {}
    for r in range(NR):
        if Hc[:, r].any() or solid_tol[:, r].any():
            hcnt = int(Hc[:, r].sum())
            per[REG[r]] = dict(human=hcnt, robot_solid_tol=int(solid_tol[:, r].sum()),
                               solid_when_human=int((Hc[:, r] & solid_tol[:, r]).sum()),
                               hover_when_human=int((Hc[:, r] & hover[:, r]).sum()),
                               far_when_human=int((Hc[:, r] & (c[:, r] > 0.02)).sum()),
                               human_clear_robot_le5mm=int((sep[:, r] & (c[:, r] <= TH['robot_must_separate_m'])).sum()),
                               robot_clear_p50_when_human_mm=round(float(np.median(c[Hc[:, r], r]) * 1000), 2) if hcnt else None)
    res['per_region'] = per
    viol = Hc & ~solid_tol
    return res, viol, solid_tol, sep & (c <= TH['robot_must_separate_m'])


def slip_eval(solid, frame_pts_xyz, dpt, regof, obj_normals=None):
    """Material speed of contact points (<= 1 mm at t and t+1) of regions in solid contact at t and t+1."""
    sp_med, sp_max, rf = [], [], []
    for r in range(NR):
        idx = np.flatnonzero(regof == r)
        if not len(idx):
            continue
        for t in range(T - 1):
            if not (solid[t, r] and solid[t + 1, r]):
                continue
            k = idx[(dpt[t, idx] <= CT) & (dpt[t + 1, idx] <= CT)]
            if not len(k):
                continue
            dv = frame_pts_xyz[t + 1, k] - frame_pts_xyz[t, k]
            if obj_normals is None:
                v = np.linalg.norm(dv[:, :2], axis=1) * fps
            else:
                n = obj_normals[t, k]; v = np.linalg.norm(dv - (dv * n).sum(1, keepdims=True) * n, axis=1) * fps
            sp_med.append(np.median(v)); sp_max.append(v.max()); rf.append((t, r))
    sp_med = np.array(sp_med); sp_max = np.array(sp_max)
    if not len(sp_med):
        return dict(region_frames=0), np.zeros((T, NR), bool)
    bad = np.zeros((T, NR), bool)
    for (t, r), s in zip(rf, sp_max):
        if s >= SLIP:
            bad[t, r] = True
    return dict(region_frames=int(len(sp_med)), median_point_speed_cms_p50=round(float(np.percentile(sp_med, 50) * 100), 2),
                median_point_speed_cms_p95=round(float(np.percentile(sp_med, 95) * 100), 2),
                max_point_speed_cms_p50=round(float(np.percentile(sp_max, 50) * 100), 2),
                max_point_speed_cms_p95=round(float(np.percentile(sp_max, 95) * 100), 2),
                frac_region_frames_median_ge_1cms=round(float((sp_med >= SLIP).mean()), 4),
                frac_region_frames_any_point_ge_1cms=round(float((sp_max >= SLIP).mean()), 4),
                per_region={REG[r]: dict(n=int(sum(1 for (_, rr) in rf if rr == r)),
                                         p50_cms=round(float(np.median([s for (_, rr), s in zip(rf, sp_med) if rr == r]) * 100), 2))
                            for r in sorted({rr for _, rr in rf})}), bad


def flicker_eval(c):
    raw = (c >= -PT) & (c <= CT)
    hy = np.zeros_like(raw)
    for r in range(NR):
        s = False
        for t in range(T):
            s = (c[t, r] < CT) if not s else not (c[t, r] > 2 * CT)
            hy[t, r] = s
    out = dict(raw_segments=0, raw_segments_lt3=0, hyst_segments=0, hyst_segments_lt3=0, toggles_per_s_raw=0.0)
    bad = np.zeros_like(raw)
    tog = 0
    for r in range(NR):
        sr = segs(raw[:, r]); sh = segs(hy[:, r])
        out['raw_segments'] += len(sr); out['raw_segments_lt3'] += sum(1 for a, b in sr if b - a < 3)
        out['hyst_segments'] += len(sh); out['hyst_segments_lt3'] += sum(1 for a, b in sh if b - a < 3)
        for a, b in sh:
            if b - a < 3:
                bad[a:b, r] = True
        tog += int(np.abs(np.diff(raw[:, r].astype(int))).sum())
    out['toggles_per_s_raw'] = round(tog / (T / fps), 2)
    return out, bad


res = dict(cid=cid, pose='qpos_umr (raw UMR, control arm)' if RAW else 'qpos (post-pass output)', robot=ROBOT, frames=T, fps=fps, floor_offset_removed_m=floor, human_scale=hscale,
           measure_seconds=round(t_meas, 1), human_region_map=hinfo)
# floor-relative check
res['floor'] = dict(npz_floor_height=float(o['floor_height']), floor_offset_removed=floor,
                    robot_min_z_p01_mm=round(float(np.percentile(zr.min(1), 1) * 1000), 2),
                    human_feet_min_z_p05_mm=round(float(np.percentile(np.minimum(hg[:, REG.index('L_foot')], hg[:, REG.index('R_foot')]), 5) * 1000), 2),
                    human_feet_min_z_p50_mm=round(float(np.percentile(np.minimum(hg[:, REG.index('L_foot')], hg[:, REG.index('R_foot')]), 50) * 1000), 2))
# penetration
pen = dict(ground_dense_max_mm=round(float(max(0, -zr.min()) * 1000), 3),
           ground_visual_verts_max_mm=round(float(max(0, -g_vis_min.min()) * 1000), 3),
           ground_mujoco_collision_max_mm=round(float(max(0, -g_coll_min.min()) * 1000), 3),
           ground_frames_gt0p5mm=int(((np.minimum(g_vis_min, g_coll_min)) < -PT).sum()),
           self_mujoco_max_mm=round(float(max(0, -self_min.min()) * 1000), 3),
           self_frames_gt0p5mm=int((self_min < -PT).sum()),
           self_worst_pair=(lambda t: f'{bname(m.geom_bodyid[self_arg[t][0]])}|{bname(m.geom_bodyid[self_arg[t][1]])}' if self_arg[t] else None)(int(np.argmin(self_min))))
# diagnostic visual-hull self check (links without collision geoms + knee visual hull), excluding construction overlaps
dpen = np.full(T, np.inf); dpair = {}
d2 = mujoco.MjData(m)
for t in range(T):
    d2.qpos[:] = Q[t]; mujoco.mj_kinematics(m, d2)
    for p in diag_pairs:
        if diag_rest[p] < -1e-4:
            continue
        dist = mujoco.mj_geomDistance(m, d2, p[0], p[1], 0.01, ft)
        if dist < -PT:
            k = f'{bname(m.geom_bodyid[p[0]])}|{bname(m.geom_bodyid[p[1]])}'
            dpair[k] = dpair.get(k, 0) + 1
        dpen[t] = min(dpen[t], dist)
pen['diag_visual_hull_self_max_mm'] = round(float(max(0, -dpen.min()) * 1000), 2)
pen['diag_visual_hull_self_frames_gt0p5mm'] = int((dpen < -PT).sum())
pen['diag_visual_hull_pairs'] = dict(sorted(dpair.items(), key=lambda x: -x[1])[:6])
pen['diag_construction_overlaps_skipped'] = int(sum(1 for v in diag_rest.values() if v < -1e-4))
viol_frames = (np.minimum(g_vis_min, g_coll_min) < -PT) | (self_min < -PT)
if has_obj:
    pen.update(object_dense_max_mm=round(float(max(0, -np.nanmin(o_pt)) * 1000), 3),
               object_visual_verts_max_mm=round(float(max(0, -o_vis_min.min()) * 1000), 3),
               object_mujoco_collision_max_mm=round(float(max(0, -o_coll_min.min()) * 1000), 3),
               object_frames_gt0p5mm=int((np.minimum(o_vis_min, o_coll_min) < -PT).sum()))
    viol_frames |= np.minimum(o_vis_min, o_coll_min) < -PT
res['penetration'] = pen
# ground contact, consistency, slip, flicker
g_res, g_viol, g_solid, g_sepv = contact_eval(cg_patch, H_g, hg, 'ground')
g_res_surf, _, _, _ = contact_eval(cg_surf, H_g, hg, 'ground')
res['ground_contact'] = g_res
res['ground_contact_surface_regions'] = {k: g_res_surf[k] for k in ('solid_rate_tol', 'hover_rate', 'hover_frames')}
dpt_g = np.where(preg[None] >= 0, zr, np.inf)
res['ground_slip'], g_slipbad = slip_eval(g_solid, X.astype(np.float64), dpt_g, preg)
res['ground_flicker'], g_flick = flicker_eval(cg_patch)
viol_frames |= g_viol.any(1) | g_sepv.any(1) | g_slipbad.any(1) | g_flick.any(1)
rule_frames = dict(penetration=int(((np.minimum(g_vis_min, g_coll_min) < -PT) | (self_min < -PT) | ((np.minimum(o_vis_min, o_coll_min) < -PT) if has_obj else False)).sum()),
                   ground_touch=int(g_viol.any(1).sum()), ground_separation=int(g_sepv.any(1).sum()),
                   ground_slip=int(g_slipbad.any(1).sum()), ground_flicker=int(g_flick.any(1).sum()))
if has_obj:
    o_res, o_viol, o_solid, o_sepv = contact_eval(co_patch, H_o, ho, 'object')
    o_res_surf, _, _, _ = contact_eval(co_surf, H_o, ho, 'object')
    res['object_contact'] = o_res
    res['object_contact_surface_regions'] = {k: o_res_surf[k] for k in ('solid_rate_tol', 'hover_rate', 'hover_frames')}
    dpt_o = np.where(preg[None] >= 0, o_pt, np.inf)
    res['object_slip'], o_slipbad = slip_eval(o_solid, o_loc.astype(np.float64), dpt_o, preg, obj_normals=o_nrm.astype(np.float64))
    res['object_flicker'], o_flick = flicker_eval(co_patch)
    viol_frames |= o_viol.any(1) | o_sepv.any(1) | o_slipbad.any(1) | o_flick.any(1)
    rule_frames.update(object_touch=int(o_viol.any(1).sum()), object_separation=int(o_sepv.any(1).sum()),
                       object_slip=int(o_slipbad.any(1).sum()), object_flicker=int(o_flick.any(1).sum()))
    # object geometry used by the engine vs the real object (box/sphere detection audit)
    res['object_geometry'] = dict(xml=str(o['object_xml']), n_pieces=len(obj_pieces),
                                  piece_vertex_counts=[int(len(p['v'])) for p in obj_pieces],
                                  extent_m=np.round(np.ptp(np.concatenate([p['v'] for p in obj_pieces]), 0), 4).tolist())
flags = o['flags'].astype(int)
res['frames_violating_any_rule'] = int(viol_frames.sum())
res['frames_violating_by_rule'] = rule_frames
res['engine_flagged_frames'] = int((flags > 0).sum())
res['violating_but_not_flagged'] = int((viol_frames & (flags == 0)).sum())
res['human_contact_frames'] = dict(ground={REG[r]: int(H_g[:, r].sum()) for r in range(NR) if H_g[:, r].any()},
                                   object=({REG[r]: int(H_o[:, r].sum()) for r in range(NR) if H_o[:, r].any()} if has_obj else None))
json.dump(res, open(QD / f'{cid}{SUF}.measure.json', 'w'), indent=1, default=str)
np.savez_compressed(QD / f'{cid}{SUF}.perframe.npz', cg_patch=cg_patch.astype(np.float32), hg=hg.astype(np.float32), H_g=H_g,
                    g_vis_min=g_vis_min, g_coll_min=g_coll_min, self_min=self_min, viol=viol_frames, flags=flags,
                    **(dict(co_patch=co_patch.astype(np.float32), ho=ho.astype(np.float32), H_o=H_o, o_vis_min=o_vis_min,
                            o_coll_min=o_coll_min) if has_obj else {}))
print(json.dumps({k: v for k, v in res.items() if k not in ('human_region_map',)}, indent=1, default=str))
