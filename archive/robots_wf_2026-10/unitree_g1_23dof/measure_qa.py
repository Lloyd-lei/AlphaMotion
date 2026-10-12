"""Step 3b -- independent DATA_SPEC_v2 audit of the 4 QA outputs (g1measure; no factory gauge code).
Writes QADIR/metrics/<clip>.json.  Usage: python measure_qa.py [dance soccer box getup]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import g1common as C
import g1measure as M
from run_qa import CLIPS, cid_of

OUT = Path('/home/arenalabs/umr_factory_data/out')
S = M.SPEC
names = sys.argv[1:] or list(CLIPS)
(C.QADIR / 'metrics').mkdir(parents=True, exist_ok=True)


def mm(x):
    return None if x is None or not np.isfinite(x) else round(float(x) * 1000, 3)


for nm in names:
    c = CLIPS[nm]; cid = cid_of(nm, c)
    npz = OUT / cid / f'{C.ROBOT}.npz'
    if not npz.exists():
        print(nm, 'missing', npz); continue
    t0 = time.time()
    res = M.measure(str(npz))
    o = res['npz']; T = res['T']
    # ---------------- penetration
    g_all = np.minimum(np.minimum(res['g_dense'].min(1), res['g_hull'].min(1)), res['g_coll'].min(1))
    pen = dict(
        ground_dense_max_mm=mm(max(0, -res['g_dense'].min())), ground_hull_max_mm=mm(max(0, -res['g_hull'].min())),
        ground_mujoco_collision_max_mm=mm(max(0, -res['g_coll'].min())),
        ground_frames_gt_0p5mm=int((g_all < -S['pen_tol']).sum()),
        self_mujoco_max_mm=mm(max(0, -res['self_min'].min())), self_frames_gt_0p5mm=int((res['self_min'] < -S['pen_tol']).sum()),
        self_pairs_checked=res['pairs_checked'])
    wp = {}
    for t in np.flatnonzero(res['self_min'] < -S['pen_tol']):
        a, b = res['self_worst'][t]
        k = '|'.join(sorted(C.name(res['m'], 1, res['m'].geom_bodyid[g]) for g in (a, b)))
        wp[k] = wp.get(k, 0) + 1
    pen['self_pen_pairs_frames'] = wp
    o_all = None
    if res['has_obj']:
        o_all = np.minimum(res['o_dense'].min(1), res['o_coll'].min(1))
        pen.update(object_dense_max_mm=mm(max(0, -res['o_dense'].min())),
                   object_mujoco_collision_max_mm=mm(max(0, -res['o_coll'].min())),
                   object_frames_gt_0p5mm=int((o_all < -S['pen_tol']).sum()))
    # ---------------- contact / consistency / slip / flicker
    cg = M.contact_stats(res, 'ground'); mg = cg.pop('_masks')
    # same checks with the mesh-exact gauge (dense points U visual hull vertices): separates sampling resolution
    # from real hover
    res_h = dict(res); res_h['g_dense'] = np.minimum(res['g_dense'], res['g_hull'])
    cg_h = M.contact_stats(res_h, 'ground'); cg_h.pop('_masks')
    lab_g = o['ground_contact'].astype(bool); rc_h = res_h['g_dense']
    ground_meshexact = dict(totals=cg_h['totals'],
                            label_truth_solid=int((lab_g & (rc_h >= -S['pen_tol']) & (rc_h <= S['contact_max'])).sum()),
                            labelled=int(lab_g.sum()))
    rc_c = res['g_coll']
    ground_meshexact['label_truth_solid_by_mujoco_collision_geoms'] = int((lab_g & (rc_c >= -S['pen_tol']) & (rc_c <= S['contact_max'])).sum())
    co = mo = None
    if res['has_obj']:
        co = M.contact_stats(res, 'object'); mo = co.pop('_masks')
    # label truthfulness: frames the OUTPUT npz labels as in contact must be solid by our gauge
    lt = {}
    for key, rc in (('ground_contact', res['g_dense']),) + ((('object_contact', res['o_dense']),) if res['has_obj'] else ()):
        lab = o[key].astype(bool)
        ok = (rc >= -S['pen_tol']) & (rc <= S['contact_max'])
        lt[key] = dict(labelled_region_frames=int(lab.sum()), solid_by_independent_gauge=int((lab & ok).sum()),
                       not_solid=int((lab & ~ok).sum()),
                       not_solid_clearance_mm_max=mm(np.max(rc[lab & ~ok])) if (lab & ~ok).any() else None,
                       touching_but_unlabelled_region_frames=int((~lab & ok).sum()))
    # ---------------- per-frame exclusion accounting (v2)
    hs_seg = np.zeros(T, bool)
    for rc in [res['g_dense']] + ([res['o_dense']] if res['has_obj'] else []):
        h = M.hysteresis(np.where(np.isfinite(rc), rc, np.inf), S['hyst_enter'] + 1e-9, S['hyst_leave'])
        for r in range(rc.shape[1]):
            for a, b in M.runs(h[:, r]):
                if b - a < S['min_seg']:
                    hs_seg[a:b] = True
    masks = dict(pen_ground=g_all < -S['pen_tol'], pen_self=res['self_min'] < -S['pen_tol'],
                 human_ground_contact_not_solid=mg['human_contact_not_solid'], phantom_ground=mg['phantom'],
                 slip_ground=mg['slip'], short_contact_segment=hs_seg)
    if res['has_obj']:
        masks.update(pen_object=o_all < -S['pen_tol'], human_object_contact_not_solid=mo['human_contact_not_solid'],
                     phantom_object=mo['phantom'], slip_object=mo['slip'])
    union = np.zeros(T, bool)
    for v in masks.values():
        union |= v
    no_slip = np.zeros(T, bool)
    for k, v in masks.items():
        if not k.startswith('slip'):
            no_slip |= v
    excl = dict({k: int(v.sum()) for k, v in masks.items()}, union_all_rules=int(union.sum()),
                union_without_slip=int(no_slip.sum()), kept_frames=int(T - union.sum()),
                engine_flagged_frames=int((o['flags'] > 0).sum()),
                engine_flagged_but_not_ours=int(((o['flags'] > 0) & ~union).sum()),
                ours_but_not_engine_flagged=int((union & ~(o['flags'] > 0)).sum()))
    # ---------------- floor convention
    fl = dict(floor_height=float(o['floor_height']), floor_offset_removed_m=float(o['floor_offset_removed']),
              human_floor_method=json.load(open(str(npz).replace('.npz', '.qa.json')))['floor']['human_method'])
    # stance-frame foot clearance of the human (should be ~0 if floor calibration right)
    hf = res['h_g'][:, [C.RID['L_foot'], C.RID['R_foot']]].min(1)
    fl['human_lowest_foot_point_p05_p50_mm'] = [mm(np.percentile(hf, 5)), mm(np.percentile(hf, 50))]
    rf = res['g_dense'][:, [C.RID['L_foot'], C.RID['R_foot']]].min(1)
    fl['robot_lowest_foot_point_p05_p50_mm'] = [mm(np.percentile(rf, 5)), mm(np.percentile(rf, 50))]
    # ---------------- rules
    def verdict(n):
        return 'PASS' if n == 0 else 'FAIL'
    tot_g = cg['totals']
    rules = {
        'R1_solid_contact_label_truth(labelled contact -> min clearance in [-0.5,1]mm)':
            verdict(sum(v['not_solid'] for v in lt.values())),
        'R2_no_penetration_ground(points+hull+MuJoCo coll <=0.5mm)': verdict(pen['ground_frames_gt_0p5mm']),
        'R2_no_penetration_self(MuJoCo non-adjacent <=0.5mm)': verdict(pen['self_frames_gt_0p5mm']),
        'R3_no_slip(contact point tangential < 1 cm/s)':
            verdict(cg['slip'].get('frames_with_slipping_contact_point', 0) + (co['slip'].get('frames_with_slipping_contact_point', 0) if co else 0)),
        'R4_no_flicker(contact segments >= 3 frames, 1/2 mm hysteresis)': verdict(int(hs_seg.sum())),
        'R5_human_consistency_on(human contact -> robot solid)':
            verdict(tot_g['human_contact_region_frames'] - tot_g['solid'] + ((co['totals']['human_contact_region_frames'] - co['totals']['solid']) if co else 0)),
        'R5_human_consistency_off(human >2cm -> robot >5mm)':
            verdict(tot_g['phantom_region_frames'] + (co['totals']['phantom_region_frames'] if co else 0)),
        'R6_floor_relative(z=0 output, offset stored)': 'PASS' if fl['floor_height'] == 0.0 else 'FAIL',
        'R7_failing_frames_flagged(engine flags cover every v2 failure)': verdict(excl['ours_but_not_engine_flagged']),
    }
    if res['has_obj']:
        rules['R2_no_penetration_object(points+MuJoCo coll <=0.5mm)'] = verdict(pen['object_frames_gt_0p5mm'])
    out = dict(clip=nm, cid=cid, npz=str(npz), frames=T, fps=res['fps'], window=dict(start=c['start'], end=c['end'], stride=c['stride']),
               source=dict(kind=c['kind'], data=c['data'], seq=c['seq'], note=c['note']),
               human_region_map=res['hinfo'], human_scale=float(o['human_scale']),
               penetration=pen, ground=cg, ground_meshexact_gauge=ground_meshexact, object=co, label_truth=lt, exclusion=excl, floor=fl, rules=rules,
               seconds=round(time.time() - t0, 1))
    if res['has_obj']:
        ob = res['obj']
        out['object_geometry'] = dict(n_pieces=len(ob.pieces), n_faces=ob.n_faces, extent_m=np.round(ob.extent, 4).tolist(),
                                      factory_solid=json.load(open(str(npz).replace('.npz', '.qa.json')))['object'].get('solid'))
        # factory object gauge vs ours on the labelled object-contact region frames
        fo = o['object_clearance'].astype(np.float64)
        lab = o['object_contact'].astype(bool)
        if lab.any():
            diff = res['o_dense'][lab] - fo[lab]
            out['object_geometry']['ours_minus_factory_clearance_on_labelled_mm_p50_p95_max'] = [
                mm(np.percentile(diff, 50)), mm(np.percentile(diff, 95)), mm(diff.max())]
    json.dump(out, open(C.QADIR / 'metrics' / f'{nm}.json', 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(nm, json.dumps(rules, indent=0))
    print(nm, 'pen', json.dumps(pen)); print(nm, 'excl', json.dumps(excl))
    print(nm, 'ground totals', json.dumps(cg['totals']), 'slip', json.dumps({k: v for k, v in cg['slip'].items() if k != 'per_region'}))
    if co:
        print(nm, 'object totals', json.dumps(co['totals']), 'slip', json.dumps({k: v for k, v in co['slip'].items() if k != 'per_region'}))
        print(nm, 'object geom', json.dumps(out['object_geometry']))
    print(nm, 'meshexact', json.dumps(ground_meshexact))
    print(nm, 'label truth', json.dumps(lt), 'seconds', out['seconds'], flush=True)
