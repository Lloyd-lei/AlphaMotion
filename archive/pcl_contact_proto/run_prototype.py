"""Prototype: point-cloud contact/penetration/slip labels on GMR-retargeted G1 (and human_smpl source) for two LAFAN
clips, compared against the legacy joint-centre labels (pose_translator_design/exp_topo/contact_label.py output
shipped in cache_lafan_v3/<body>_contact.npz, read through data/trajectory_reader.py).

Labels are computed on the FULL clip (no window edge effects for hysteresis or velocities), then reported on the
full clip and on the requested window.
    python run_prototype.py [--bodies unitree_g1_29dof human_smpl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pcl_contact import core as C  # noqa: E402
from pcl_contact import sources as SR  # noqa: E402

OUT = HERE / 'out'
CASES = [('lafan::fallAndGetUp1_subject4', (4136, 4264)), ('lafan::walk1_subject5', (1000, 1300))]
# capture-session floor (cm, world y) from run_floor.py -> out/session_floor.json
REG = SR.S.REGIONS
# comparison groups: PCL regions <-> legacy limb groups (contact_label.limb_groups header field)
CMP = {
    'L_foot': (['L_foot'], ['L_foot']), 'R_foot': (['R_foot'], ['R_foot']),
    'L_knee/shin': (['L_shin'], ['L_knee']), 'R_knee/shin': (['R_shin'], ['R_knee']),
    'L_hip/thigh': (['L_thigh'], ['L_hip']), 'R_hip/thigh': (['R_thigh'], ['R_hip']),
    'L_hand': (['L_hand'], ['L_hand']), 'R_hand': (['R_hand'], ['R_hand']),
    'L_elbow/forearm': (['L_forearm'], ['L_elbow']), 'R_elbow/forearm': (['R_forearm'], ['R_elbow']),
    'L_shoulder/upperarm': (['L_upper_arm'], ['L_shoulder']), 'R_shoulder/upperarm': (['R_upper_arm'], ['R_shoulder']),
    'trunk+pelvis': (['pelvis', 'trunk'], ['trunk']), 'head': (['head'], ['head']),
}
HUMAN_LIMB = {  # legacy header has no limb_groups for human in some caches; name-based fallback
    'Pelvis': 'trunk', 'Torso': 'trunk', 'Spine': 'trunk', 'Chest': 'trunk', 'Neck': 'head', 'Head': 'head',
    'L_Hip': 'L_hip', 'R_Hip': 'R_hip', 'L_Knee': 'L_knee', 'R_Knee': 'R_knee', 'L_Ankle': 'L_foot', 'R_Ankle': 'R_foot',
    'L_Toe': 'L_foot', 'R_Toe': 'R_foot', 'L_Thorax': 'L_shoulder', 'R_Thorax': 'R_shoulder', 'L_Shoulder': 'L_shoulder',
    'R_Shoulder': 'R_shoulder', 'L_Elbow': 'L_elbow', 'R_Elbow': 'R_elbow', 'L_Wrist': 'L_hand', 'R_Wrist': 'R_hand'}


def legacy_source(body, gid):
    m = json.loads(Path(SR.MANIFEST).read_text())['xwbc']
    b = next(b for b in m['bodies'] if b['body'] == body)
    g = next(g for g in m['groups'] if g['group_id'] == gid)
    return b['caches'][g['cache_index']]['contact_npz'], g['start']


def legacy_limb_groups(body, names):
    with np.load(legacy_source(body, CASES[0][0])[0]) as z:
        h = json.loads(str(z['header']))
    lg = h.get('limb_groups')
    if lg is None or len(lg) != len(names):
        lg = [HUMAN_LIMB.get(n, 'other') for n in names]
    return lg, h


def kappa(a, b):
    a = a.astype(bool); b = b.astype(bool); n = len(a)
    po = (a == b).mean(); pa = a.mean(); pb = b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def onset_offsets(pcl, leg, max_lag=10):
    """For every PCL contact segment, signed frame offsets of the nearest legacy onset/offset (legacy - pcl)."""
    ps = C.segments(pcl); ls = C.segments(leg)
    on, off = [], []
    for a, b in ps:
        cand = [(la - a) for la, lb in ls if abs(la - a) <= max_lag]
        if cand: on.append(min(cand, key=abs))
        cand = [(lb - b) for la, lb in ls if abs(lb - b) <= max_lag]
        if cand: off.append(min(cand, key=abs))
    return on, off


def pct(x, q):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return [round(float(v), 2) for v in np.percentile(x, q)] if len(x) else None


def run_case(body, gid, window, floor_cm, cfg=None, hand_fix='rot_axis'):
    t0 = time.time()
    d = SR.load_traj(gid, body)
    w = SR.world_points(d, body, hand_fix=hand_fix)
    X = w['X']; a = w['asset']; names = w['names']
    anchor = a.anchor.numpy(); region = a.region.numpy(); area = a.area.numpy()
    mask = SR.point_mask(a)
    groups = C.groups_from_asset(anchor, region, names, REG)
    for cname, (regs, _) in CMP.items():
        idx = np.flatnonzero(np.isin(region, [REG.index(r) for r in regs]) & mask)
        if len(idx):
            groups[f'cmp:{cname}'] = dict(kind='region', idx=idx)
    for g in groups.values():
        g['idx'] = g['idx'][mask[g['idx']]] if g['kind'] == 'link' else g['idx']
    fps = float(d['fps'])
    lab = C.ground_labels(X, fps, groups, area=area, floor_cm=floor_cm, cfg=cfg)
    lab_w0 = C.ground_labels(X, fps, {k: v for k, v in groups.items() if k.startswith('cmp:')}, area=area, floor_cm=0.0, cfg=cfg)
    T = X.shape[0]
    # ---------------------------------------------------------------- legacy
    lg, hdr = legacy_limb_groups(body, names)
    cache_npz, start = legacy_source(body, gid)
    leg_hard = d['contact_hard']
    leg_low = SR.TR.npz_rows(cache_npz, 'contact_low', start, T).astype(bool)
    leg_gap = d['contact_gap_m'] * 100.0
    # legacy joint horizontal speed (cm/s) for the slip comparison
    Wp = C.smooth_positions(d['world_positions'], 5, 2)
    Vj = C.velocity(Wp, fps); vj_h = np.linalg.norm(Vj[..., [0, 2]], axis=-1)
    gi = {n: i for i, n in enumerate(lab['group_names'])}
    gw = {n: i for i, n in enumerate(lab_w0['group_names'])}
    sl = slice(*window)
    rep = dict(body=body, group=gid, window=list(window), frames=T, fps=fps, floor_cm=floor_cm, hand_fix=hand_fix if a.binding == 'bone_revolution' else None,
               n_points=int(a.num_points), n_points_masked=int((~mask).sum()), cfg={k: v for k, v in lab['cfg'].items()},
               legacy=dict(source=cache_npz, h0_m=hdr.get('h0_m'), thr_m=hdr.get('thr_m'), v_thr_ms=hdr.get('v_thr_ms'),
                           rule=hdr.get('rule'), sole_applied=hdr.get('sole_applied')), cmp={}, frames_summary={})
    for scope, s in (('clip', slice(0, T)), ('window', sl)):
        ffs = {}
        anyc = lab['contact'][s][:, [gi[k] for k in gi if k.startswith('region:')]].any(1)
        pen_frame = lab['pen'][s][:, [gi[k] for k in gi if k.startswith('region:')]].max(1)
        ffs['frames'] = int(anyc.shape[0])
        ffs['frac_frames_any_contact'] = round(float(anyc.mean()), 4)
        ffs['frac_frames_pen_gt_1cm'] = round(float((pen_frame > 1).mean()), 4)
        ffs['frac_frames_pen_gt_3cm'] = round(float((pen_frame > 3).mean()), 4)
        ffs['pen_frame_max_cm_p50_p95_max'] = pct(pen_frame, [50, 95, 100])
        ffs['min_clearance_cm_p1_p50'] = pct(lab['d'][s].min(1), [1, 50])
        reg_contact = {k[7:]: round(float(lab['contact'][s][:, gi[k]].mean()), 4) for k in gi if k.startswith('region:')}
        ffs['region_contact_frac'] = {k: v for k, v in reg_contact.items() if v > 0}
        ffs['region_pen_max_cm'] = {k[7:]: round(float(lab['pen'][s][:, gi[k]].max()), 2) for k in gi if k.startswith('region:') and lab['pen'][s][:, gi[k]].max() > 0.5}
        link_contact = {k[5:]: round(float(lab['contact'][s][:, gi[k]].mean()), 4) for k in gi if k.startswith('link:')}
        ffs['link_contact_frac'] = {k: v for k, v in link_contact.items() if v > 0}
        rep['frames_summary'][scope] = ffs
        cm = {}
        for cname, (_, lgroups) in CMP.items():
            k = f'cmp:{cname}'
            if k not in gi:
                continue
            jsel = [j for j, g in enumerate(lg) if g in lgroups]
            if not jsel:
                continue
            p = lab['contact'][s][:, gi[k]]
            p0 = lab_w0['contact'][s][:, gw[k]]
            lh = leg_hard[s][:, jsel].any(1); ll = leg_low[s][:, jsel].any(1)
            clear = lab['clearance'][s][:, gi[k]]
            lgap = np.nanmin(leg_gap[s][:, jsel], axis=1)
            sp = lab['slip_speed'][s][:, gi[k]]
            row = dict(pcl_frac=round(float(p.mean()), 4), pcl_world0_frac=round(float(p0.mean()), 4),
                       legacy_frac=round(float(lh.mean()), 4), legacy_low_frac=round(float(ll.mean()), 4))
            if p.any() or lh.any() or ll.any():
                for lname, L in (('legacy', lh), ('legacy_low', ll)):
                    row[f'vs_{lname}'] = dict(agree=round(float((p == L).mean()), 4), kappa=round(kappa(p, L), 3),
                                              both=int((p & L).sum()), pcl_only=int((p & ~L).sum()), legacy_only=int((~p & L).sum()))
                row['legacy_only_pcl_clearance_cm_p10_50_90'] = pct(clear[~p & lh], [10, 50, 90])
                row['pcl_only_legacy_gap_cm_p10_50_90'] = pct(lgap[p & ~lh], [10, 50, 90])
                on, off = onset_offsets(p, lh)
                row['onset_offset_legacy_minus_pcl_frames'] = dict(n=len(on), p50=pct(on, [50]), mean=round(float(np.mean(on)), 2) if on else None)
                row['liftoff_offset_legacy_minus_pcl_frames'] = dict(n=len(off), p50=pct(off, [50]), mean=round(float(np.mean(off)), 2) if off else None)
                jg = lab['slip_judged'][s][:, gi[k]]
                row['pcl_slip_judged_frac_of_contact'] = round(float(jg[p].mean()), 4) if p.any() else None
                row['pcl_slip_frac_of_judged'] = round(float(lab['slip'][s][:, gi[k]][jg].mean()), 4) if jg.any() else None
                row['pcl_slip_speed_cms_p50_p90_p99'] = pct(sp[jg], [50, 90, 99])
                inn = lab['interior'][s][:, gi[k]] & jg
                row['sustained'] = dict(judged_interior_frames=int(inn.sum()),
                                        slip_frac=round(float(lab['slip_sustained'][s][:, gi[k]][inn].mean()), 4) if inn.any() else None,
                                        speed_cms_p50_p90_p99=pct(sp[inn], [50, 90, 99]),
                                        legacy_joint_hspeed_cms_p50_p90=pct(vj_h[s][:, jsel].min(1)[inn], [50, 90]))
                row['legacy_joint_hspeed_cms_on_judged_frames_p50_p90'] = pct(vj_h[s][:, jsel].min(1)[jg], [50, 90])
                row['pen_max_cm'] = round(float(lab['pen'][s][:, gi[k]].max()), 2)
            cm[cname] = row
        rep['cmp'][scope] = cm
    # segments (window only) for feet
    rep['segments_window'] = {}
    for k in ('cmp:L_foot', 'cmp:R_foot', 'cmp:L_knee/shin', 'cmp:R_knee/shin', 'cmp:L_hand', 'cmp:R_hand', 'cmp:trunk+pelvis'):
        if k in lab['segments']:
            rep['segments_window'][k[4:]] = [r for r in lab['segments'][k] if r['end'] > window[0] and r['start'] < window[1]]
    rep['seconds'] = round(time.time() - t0, 1)
    tag = f"{body}__{gid.split('::')[1]}"
    np.savez_compressed(OUT / f'labels_{tag}_window.npz',
                        X=X[sl].astype(np.float16), d=lab['d'][sl].astype(np.float16), vt=lab['vt'][sl].astype(np.float16),
                        point_contact=lab['point_contact'][sl], point_pen=lab['point_pen'][sl].astype(np.float16),
                        group_names=np.array(lab['group_names']), contact=lab['contact'][sl], slip=lab['slip'][sl], slip_judged=lab['slip_judged'][sl], slip_sustained=lab['slip_sustained'][sl], interior=lab['interior'][sl],
                        slip_speed=lab['slip_speed'][sl].astype(np.float32), pen=lab['pen'][sl].astype(np.float32),
                        clearance=lab['clearance'][sl].astype(np.float32), legacy_contact=leg_hard[sl], legacy_low=leg_low[sl],
                        legacy_gap_cm=leg_gap[sl].astype(np.float32), joint_world=d['world_positions'][sl].astype(np.float32),
                        region=region, anchor=anchor, floor_cm=np.float32(floor_cm), window=np.array(window), legacy_limb=np.array(lg))
    (OUT / f'report_{tag}.json').write_text(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bodies', nargs='*', default=['unitree_g1_29dof', 'human_smpl'])
    ap.add_argument('--floor', choices=['session', 'world0'], default='session')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    fl = json.loads((OUT / 'session_floor.json').read_text())
    from pcl_contact.floor import session_of
    for body in args.bodies:
        for gid, win in CASES:
            floor = fl[f'{session_of(gid)}/{body}']['floor_cm'] if args.floor == 'session' else 0.0
            r = run_case(body, gid, win, floor)
            print(body, gid, 'floor %.2f' % floor, r['seconds'], 's', flush=True)


if __name__ == '__main__':
    main()
