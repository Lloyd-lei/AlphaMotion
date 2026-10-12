"""unitree_h1 dedicated pipeline -- step 3b: INDEPENDENT measurement of a factory output against DATA_SPEC_v2.
Own FK (MuJoCo mj_kinematics on the output qpos), own gauges:
  ground : dense points z (surface.npz posed per geom)  +  MuJoCo mj_geomDistance(floor plane, every robot collision geom)
  object : own SDF on the TRUE object geometry (trimesh exact signed distance on the collision mesh pieces / convex hull,
           no sphere/box auto-detection)  +  MuJoCo mj_geomDistance(robot collision geom, object convex pieces)
  self   : every collision-geom pair of different, non parent/child bodies (no other exclusions), mj_geomDistance
Human labels recomputed from UMR's captured source slots (same floor offset as the output).
Usage: <umr py> measure_independent.py <clip_id> [<clip_id> ...]   -> qa/unitree_h1/measure_<clip_id>.json + .npz"""
import json, os, sys, time
from pathlib import Path
import numpy as np, mujoco, trimesh

sys.path.insert(0, '/home/arenalabs/umr_factory_code')
ROBOT = 'unitree_h1'
D = Path('/home/arenalabs/umr_factory_data'); RD = D / 'robots' / ROBOT; PD = RD / 'pipeline'; QA = D / 'qa' / ROBOT
cfg = json.load(open(PD / 'pipeline.json'))
TH = cfg['thresholds_DATA_SPEC_v2']
PEN = TH['penetration_tol_m']; C_HI = TH['contact_clearance_m'][1]; SLIP = TH['slip_speed_max_m_s']
S = np.load(RD / 'surface.npz'); REG = [str(x) for x in S['regions']]; NR = len(REG)
preg = S['region'].astype(int); pgeom = S['geom'].astype(int); ploc = S['points_local'].astype(float)
hloc = S['hull_local'].astype(float); hgeom = S['hull_geom'].astype(int)
HUM_ENTER, HUM_LEAVE, MIN_ON = 0.01, 0.02, 3        # human (soft body / mocap) contact hysteresis, same as engine


def runs(col):
    col = np.asarray(col, bool); T = len(col)
    if T == 0:
        return []
    ch = np.flatnonzero(np.diff(col.astype(np.int8))) + 1
    st = np.r_[0, ch]; en = np.r_[ch, T]
    return [(int(a), int(b)) for a, b in zip(st, en) if col[a]]


def hyst(c, enter, leave, min_on=MIN_ON):
    T = len(c); out = np.zeros(T, bool); on = False
    for t in range(T):
        if not np.isfinite(c[t]):
            on = False
        elif on and c[t] > leave:
            on = False
        elif not on and c[t] < enter:
            on = True
        out[t] = on
    for a, b in runs(out):
        if b - a < min_on:
            out[a:b] = False
    return out


def pctl(v, q):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return round(float(np.percentile(v, q)), 3) if len(v) else None


class TrueObject:
    """Object solid = union of the MJCF collision geoms (each treated by its convex hull, as MuJoCo does)."""

    def __init__(self, xml, scale, rot, pos):
        from umrf.objects import _mjcf_geoms_local          # file loader only (mesh paths + scale as UMR)
        vis, col = _mjcf_geoms_local(xml, scale)
        self.pieces = [trimesh.convex.convex_hull(np.asarray(v)) for v, f in col]
        self.rot, self.pos = np.asarray(rot, float), np.asarray(pos, float)
        allv = np.concatenate([p.vertices for p in self.pieces]); self.c = allv.mean(0)
        self.R = float(np.linalg.norm(allv - self.c, axis=1).max())
        self.n_vertices = [len(p.vertices) for p in self.pieces]
        self.extents = [np.round(p.extents, 4).tolist() for p in self.pieces]

    def local(self, X, t):
        return (X - self.pos[t]) @ self.rot[t]

    def sdf(self, X, t, cull=0.05):
        q = self.local(X, t); d = np.full(len(X), np.inf)
        near = np.linalg.norm(q - self.c, axis=1) < self.R + cull
        if near.any():
            dd = np.full(near.sum(), np.inf)
            for p in self.pieces:
                dd = np.minimum(dd, -trimesh.proximity.signed_distance(p, q[near]))   # trimesh: inside +
            d[near] = dd
        return d


def build_model(obj):
    spec = mujoco.MjSpec.from_file(cfg['mjcf']['path'])
    if obj is not None:
        b = spec.worldbody.add_body(); b.name = '__obj'; b.mocap = True
        for i, p in enumerate(obj.pieces):
            mesh = spec.add_mesh(); mesh.name = f'__objm{i}'; mesh.uservert = np.asarray(p.vertices, np.float32).reshape(-1)
            g = b.add_geom(); g.name = f'__objg{i}'; g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = mesh.name
            g.contype = 0; g.conaffinity = 0
    return spec.compile()


def measure(cid):
    t_start = time.time()
    o = np.load(PD / 'out' / cid / f'{ROBOT}.npz', allow_pickle=True)
    prov = json.loads(str(o['provenance']))
    Q = o['qpos'].astype(float); T = len(Q); fps = float(o['fps'])
    floor = float(o['floor_offset_removed'])
    has_obj = 'object_pos' in o.files
    obj = TrueObject(str(o['object_xml']), float(o['human_scale']), o['object_rot'], o['object_pos']) if has_obj else None
    m = build_model(obj); d = mujoco.MjData(m)
    bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
    floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    obj_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, '__obj')
    rob_cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, obj_b) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    obj_g = [g for g in range(m.ngeom) if obj_b >= 0 and m.geom_bodyid[g] == obj_b]
    breg = S['body_region']; cg_reg = np.array([breg[m.geom_bodyid[g]] for g in rob_cg])
    pairs = []
    for i, g1 in enumerate(rob_cg):
        for g2 in rob_cg[i + 1:]:
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 != b2 and m.body_parentid[b1] != b2 and m.body_parentid[b2] != b1:
                pairs.append((g1, g2))
    # ------------------------------------------------ robot pass
    X = np.zeros((T, len(ploc), 3))
    g_reg = np.full((T, NR), np.inf); g_reg_col = np.full((T, NR), np.inf)
    o_reg = np.full((T, NR), np.inf); o_reg_col = np.full((T, NR), np.inf)
    o_pt = np.full((T, len(ploc)), np.inf)
    hreg_h = np.array([breg[m.geom_bodyid[g]] for g in hgeom]).astype(int)
    g_reg_hull = np.full((T, NR), np.inf); o_reg_hull = np.full((T, NR), np.inf)
    self_min = np.full(T, np.inf); self_pair_cnt = {}
    ft = np.zeros(6)
    for t in range(T):
        d.qpos[:] = Q[t]
        if obj is not None:
            d.mocap_pos[0] = obj.pos[t]; q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, obj.rot[t].reshape(-1)); d.mocap_quat[0] = q4
        mujoco.mj_kinematics(m, d)
        Xt = np.einsum('nij,nj->ni', d.geom_xmat[pgeom].reshape(-1, 3, 3), ploc) + d.geom_xpos[pgeom]
        X[t] = Xt
        np.minimum.at(g_reg[t], preg, Xt[:, 2])
        Xh = np.einsum('nij,nj->ni', d.geom_xmat[hgeom].reshape(-1, 3, 3), hloc) + d.geom_xpos[hgeom]
        np.minimum.at(g_reg_hull[t], hreg_h, Xh[:, 2])
        for k, g in enumerate(rob_cg):
            dist = mujoco.mj_geomDistance(m, d, floor_g, g, 0.05, ft)
            g_reg_col[t, cg_reg[k]] = min(g_reg_col[t, cg_reg[k]], dist)
            for og in obj_g:
                dist = mujoco.mj_geomDistance(m, d, g, og, 0.05, ft)
                o_reg_col[t, cg_reg[k]] = min(o_reg_col[t, cg_reg[k]], dist)
        for g1, g2 in pairs:
            dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.01, ft)
            if dist < self_min[t]:
                self_min[t] = dist
            if dist < -PEN:
                k = f'{bn(m.geom_bodyid[g1])}|{bn(m.geom_bodyid[g2])}'
                self_pair_cnt[k] = self_pair_cnt.get(k, 0) + 1
        if obj is not None:
            o_pt[t] = obj.sdf(Xt, t)
            np.minimum.at(o_reg[t], preg, o_pt[t])
            np.minimum.at(o_reg_hull[t], hreg_h, obj.sdf(Xh, t))
    # ------------------------------------------------ human pass (UMR source slots, same floor shift)
    from umrf import pipeline as P
    Hs = np.load(prov['human_slots'])['source_slots']
    umr = np.load(prov['umr_out'], allow_pickle=True)
    if len(Hs) != T:
        Hs = Hs[umr['frame_ids']]
    Hs = Hs.astype(float) - [0, 0, floor]
    hreg, _ = P.human_slot_regions(ROBOT, prov['source_kind'], prov['source_data'], prov['seq_key'],
                                   str(Path(prov['umr_out']).with_suffix('.umr.log')), prov['source_kind'] == 'hoi')
    hg = np.full((T, NR), np.inf); ho = np.full((T, NR), np.inf)
    for t in range(T):
        np.minimum.at(hg[t], hreg, Hs[t][:, 2])
        if obj is not None:
            np.minimum.at(ho[t], hreg, obj.sdf(Hs[t], t))
    h_g = np.stack([hyst(hg[:, r], HUM_ENTER, HUM_LEAVE) for r in range(NR)], 1)
    h_o = np.stack([hyst(ho[:, r], HUM_ENTER, HUM_LEAVE) for r in range(NR)], 1) if obj is not None else np.zeros((T, NR), bool)
    # ------------------------------------------------ rule evaluation
    flags = o['flags'].astype(int); flagged = flags > 0
    res = dict(clip=cid, robot=ROBOT, frames=T, fps=fps, floor_offset_removed_m=floor,
               object_geometry=None if obj is None else dict(xml=str(o['object_xml']), pieces=len(obj.pieces), piece_vertices=obj.n_vertices,
                                                     piece_extents_m=obj.extents, engine_solid_kind=json.load(open(PD / 'out' / cid / f'{ROBOT}.qa.json')).get('object', {}).get('kind')))
    pen_ground_dense = np.maximum(0, -X[..., 2].min(1)); pen_ground_col = np.maximum(0, -np.min(g_reg_col, 1))
    pen_self = np.maximum(0, -self_min)
    pen = dict(ground_dense=pen_ground_dense, ground_mujoco=pen_ground_col, self_mujoco=pen_self)
    if obj is not None:
        pen['object_dense_true_sdf'] = np.maximum(0, -np.min(o_pt, 1)); pen['object_mujoco'] = np.maximum(0, -np.min(o_reg_col, 1))
    res['penetration'] = {k: dict(max_mm=round(float(v.max() * 1000), 3), frames_gt_0p5mm=int((v > PEN).sum()),
                                  frames_gt_0p5mm_unflagged=int(((v > PEN) & ~flagged).sum())) for k, v in pen.items()}
    res['self_pen_pairs_gt_0p5mm'] = dict(sorted(self_pair_cnt.items(), key=lambda x: -x[1])[:8])
    pen_frame = np.zeros(T, bool)
    for v in pen.values():
        pen_frame |= v > PEN

    def contact_eval(kind, rc_dense, rc_col, hcon, hclear_val, partner_local, rc_hull):
        """rc_dense [T,R] robot region min clearance over dense points; rc_col [T,R] MuJoCo collision-geom clearance."""
        solid = (rc_dense >= -PEN) & (rc_dense <= C_HI) & ~(rc_col < -PEN)
        solid_strict = (rc_dense >= 0) & (rc_dense <= C_HI)
        rc_any = np.minimum(np.minimum(rc_dense, rc_hull), rc_col)
        solid_any = (rc_any >= -PEN) & (rc_any <= C_HI)
        hov = (rc_dense > C_HI) & (rc_dense <= 0.02)
        hc = hcon
        n_h = int(hc.sum())
        miss = hc & ~solid
        out = dict(human_contact_region_frames=n_h,
                   human_contact_by_region={REG[r]: int(hc[:, r].sum()) for r in range(NR) if hc[:, r].any()},
                   robot_solid_when_human=round(float((hc & solid).sum() / max(n_h, 1)), 4),
                   robot_solid_strict_0_1mm_when_human=round(float((hc & solid_strict).sum() / max(n_h, 1)), 4),
                   robot_solid_any_point_when_human=round(float((hc & solid_any).sum() / max(n_h, 1)), 4),
                   dense_gap_when_any_solid_mm_p50=pctl((rc_dense - rc_any)[hc & solid_any] * 1000, 50),
                   dense_gap_when_any_solid_mm_p95=pctl((rc_dense - rc_any)[hc & solid_any] * 1000, 95),
                   hover_any_point_1_20mm_when_human=int((hc & (rc_any > C_HI) & (rc_any <= 0.02)).sum()),
                   hover_1_20mm_when_human=int((hc & hov).sum()), far_gt_20mm_when_human=int((hc & (rc_dense > 0.02)).sum()),
                   penetrating_when_human=int((hc & (rc_dense < -PEN)).sum()),
                   by_region={REG[r]: dict(human=int(hc[:, r].sum()), solid=int((hc[:, r] & solid[:, r]).sum()),
                                           hover=int((hc[:, r] & hov[:, r]).sum()),
                                           robot_clear_p50_mm_when_human=pctl(rc_dense[hc[:, r], r] * 1000, 50),
                                           robot_clear_p95_mm_when_human=pctl(rc_dense[hc[:, r], r] * 1000, 95))
                              for r in range(NR) if hc[:, r].any()},
                   viol_frames=int(miss.any(1).sum()), viol_frames_unflagged=int((miss.any(1) & ~flagged).sum()))
        # consistency B: human clearly off (> 2 cm) -> robot must be off (> 5 mm)
        hoff = hclear_val > TH['human_clear_m']
        vb = hoff & (rc_dense < TH['robot_clear_when_human_clear_m'])
        out['human_off_robot_within_5mm_region_frames'] = int(vb.sum())
        out['human_off_robot_within_5mm_by_region'] = {REG[r]: int(vb[:, r].sum()) for r in range(NR) if vb[:, r].any()}
        out['human_off_viol_frames_unflagged'] = int((vb.any(1) & ~flagged).sum())
        # robot-side contact segments (spec hysteresis 1 mm / 2 mm), flicker
        seg_short, seg_n, toggles = 0, 0, 0
        raw_short = 0
        rhy = np.zeros((T, NR), bool)
        for r in range(NR):
            c = rc_dense[:, r]
            raw = (c >= -PEN) & (c <= C_HI)
            rr = runs(raw); raw_short += sum(1 for a, b in rr if b - a < MIN_ON)
            hh = hyst(c, TH['hysteresis_enter_m'], TH['hysteresis_leave_m'], min_on=1)
            rh = runs(hh); seg_n += len(rh); seg_short += sum(1 for a, b in rh if b - a < MIN_ON)
            toggles += int(np.abs(np.diff(raw.astype(int))).sum())
            rhy[:, r] = hh
        out['flicker'] = dict(raw_solid_segments_lt3=raw_short, hysteresis_segments=seg_n, hysteresis_segments_lt3=seg_short,
                              raw_toggles_per_s=round(toggles / (T / fps), 2))
        # slip: points within 1 mm of the partner at t and t+1 while the region is solid at t and t+1
        sp_med, sp_min, sp_reg = [], [], []
        switched = 0
        for r in range(NR):
            idx = np.flatnonzero(preg == r)
            if not len(idx):
                continue
            for t in range(T - 1):
                if not (solid[t, r] and solid[t + 1, r]):
                    continue
                if kind == 'ground':
                    c0, c1 = X[t, idx, 2], X[t + 1, idx, 2]
                    P0, P1 = X[t, idx, :2], X[t + 1, idx, :2]
                else:
                    c0, c1 = o_pt[t, idx], o_pt[t + 1, idx]
                    P0, P1 = partner_local(X[t, idx], t), partner_local(X[t + 1, idx], t + 1)
                k = (c0 <= C_HI) & (c1 <= C_HI) & (c0 >= -0.002) & (c1 >= -0.002)
                if not k.any():
                    switched += 1; continue
                v = np.linalg.norm(P1[k] - P0[k], axis=1) * fps
                sp_med.append(np.median(v)); sp_min.append(v.min()); sp_reg.append(r)
        sp_med, sp_min, sp_reg = np.array(sp_med), np.array(sp_min), np.array(sp_reg, int)
        out['slip'] = dict(region_frame_pairs=int(len(sp_med)), support_switch_no_common_point=switched,
                           median_point_speed_cm_s_p50=pctl(sp_med * 100, 50), median_point_speed_cm_s_p95=pctl(sp_med * 100, 95),
                           best_point_speed_cm_s_p50=pctl(sp_min * 100, 50), best_point_speed_cm_s_p95=pctl(sp_min * 100, 95),
                           frac_gt_1cm_s_median_point=round(float((sp_med > SLIP).mean()), 4) if len(sp_med) else None,
                           frac_gt_1cm_s_best_point=round(float((sp_min > SLIP).mean()), 4) if len(sp_min) else None,
                           by_region={REG[r]: dict(n=int((sp_reg == r).sum()), p50_cm_s=pctl(sp_med[sp_reg == r] * 100, 50),
                                                   p95_cm_s=pctl(sp_med[sp_reg == r] * 100, 95))
                                      for r in np.unique(sp_reg)})
        return out, solid, hov, rhy

    res['ground'], solid_g, hov_g, rhy_g = contact_eval('ground', g_reg, g_reg_col, h_g, hg, None, g_reg_hull)
    res['ground']['human_label_agreement_vs_engine'] = round(float((h_g == o['human_ground_contact']).mean()), 4)
    if obj is not None:
        res['object'], solid_o, hov_o, rhy_o = contact_eval('object', o_reg, o_reg_col, h_o, ho, obj.local, o_reg_hull)
        res['object']['human_label_agreement_vs_engine'] = round(float((h_o == o['human_object_contact']).mean()), 4)
        eng_o = o['object_clearance'].astype(float)
        res['object']['engine_vs_true_clearance_mm_when_engine_contact'] = dict(
            true_clear_p50=pctl((o_reg[o['object_contact']]) * 1000, 50), true_clear_p95=pctl((o_reg[o['object_contact']]) * 1000, 95),
            engine_clear_p50=pctl(eng_o[o['object_contact']] * 1000, 50))
    # engine output label flicker (stored labels)
    lab_short = 0
    for key in ('ground_contact', 'object_contact'):
        if key in o.files:
            for r in range(NR):
                lab_short += sum(1 for a, b in runs(o[key][:, r]) if b - a < MIN_ON)
    res['engine_label_segments_lt3'] = lab_short
    res['engine_flags'] = dict(frames_flagged=int(flagged.sum()), hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16)})
    # engine's own contact labels: are they solid by our gauge?
    eg = o['ground_contact']
    res['engine_ground_label_solid_by_our_gauge'] = round(float((eg & solid_g).sum() / max(eg.sum(), 1)), 4)
    # floor checks
    feet = [REG.index('L_foot'), REG.index('R_foot')]
    stance = solid_g[:, feet]
    res['floor'] = dict(floor_height=float(o['floor_height']), floor_offset_removed_m=floor,
                        human_foot_min_p50_mm_when_contact=pctl(hg[:, feet][h_g[:, feet]] * 1000, 50),
                        robot_foot_min_p50_mm_when_solid=pctl(g_reg[:, feet][stance] * 1000, 50))
    # per-frame exclusion set (our gauge): any rule violated
    viol = pen_frame.copy()
    viol |= (h_g & ~solid_g).any(1)
    if obj is not None:
        viol |= (h_o & ~solid_o).any(1)
    res['frames_violating_any_geometric_rule'] = int(viol.sum())
    res['frames_violating_unflagged'] = int((viol & ~flagged).sum())
    res['seconds'] = round(time.time() - t_start, 1)
    QA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(QA / f'measure_{cid}.npz', g_reg=g_reg.astype(np.float32), g_reg_col=g_reg_col.astype(np.float32),
                        g_reg_hull=g_reg_hull.astype(np.float32), o_reg_hull=o_reg_hull.astype(np.float32), o_reg_col=o_reg_col.astype(np.float32),
                        o_reg=o_reg.astype(np.float32), hg=hg.astype(np.float32), ho=ho.astype(np.float32), h_g=h_g, h_o=h_o,
                        solid_g=solid_g, hov_g=hov_g, flags=flags, self_min=self_min.astype(np.float32))
    json.dump(res, open(QA / f'measure_{cid}.json', 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    return res


if __name__ == '__main__':
    for cid in sys.argv[1:]:
        r = measure(cid)
        print(json.dumps({k: r[k] for k in ('clip', 'penetration', 'frames_violating_any_geometric_rule', 'frames_violating_unflagged', 'seconds')}))
