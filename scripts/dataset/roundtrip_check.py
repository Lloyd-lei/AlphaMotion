"""Decode every live entry and compare with the factory source npz (quantisation error + label fidelity).

    python scripts/roundtrip_check.py --root data [--device cuda] [--smplx-model .../SMPLX_NEUTRAL.npz]

Robot: root pos / root rot / joint errors; FK displacement of the 4096 dense surface points and all hull vertices
(torch FK == MuJoCo to 1e-15); per-region ground clearance recomputed on the DECODED pose vs the stored exact value;
strict-label re-check on the decoded pose (pen <= 0.5 mm, contact regions within [-0.5, 1] mm); bit-exact labels.
Human: SMPL-X param error + vertex displacement (smplx package, neutral model).  Objects: robot-frame round trip.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_retargeting.dataset import schema as S                      # noqa: E402
from surface_retargeting.dataset.dataset import Reader                    # noqa: E402
from surface_retargeting.dataset.kinematics import TorchFK                # noqa: E402
from surface_retargeting.dataset.catalog import _shard_entries            # noqa: E402
import surface_retargeting.dataset.catalog as CAT                        # noqa: E402

PEN, CTOL = 5e-4, 1e-3


def quat_angle(a, b):
    """Rotation angle between unit quaternions, chord form (well conditioned near 0, float64)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    s = np.sign((a * b).sum(-1, keepdims=True)); s[s == 0] = 1
    return 4 * np.arcsin(np.clip(np.linalg.norm(a - s * b, axis=-1) / 2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/home/arenalabs/umr_dataset/data'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--smplx-model', default='/var/tmp/alphamotion/umr_factory/umr/UMR/smpl/SMPLX_NEUTRAL.npz')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    rd = Reader(a.root); cat = rd.cat
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    rep = dict(robot=[], human=[], created=time.time())
    fks = {}
    man = json.load(open(a.root / 'manifest.json'))
    for e in range(len(cat.e_clip)):
        c, b = int(cat.e_clip[e]), int(cat.e_body[e])
        if b == 0:
            continue
        sh = rd.shard(int(cat.e_shard[e]))
        rel = cat.shards[int(cat.e_shard[e])]
        ent = next(x for x in _shard_entries(a.root, rel, man['shards'][rel]['sha256'])['entries']
                   if x['clip'] == cat.clips[c] and x['body'] == cat.bodies[b] and x['row0'] == int(cat.e_row0[e]))
        k0, n = int(cat.e_k0[e]), int(cat.e_n[e])
        dec = rd.robot(c, b, k0, n, fields=('labels', 'clearance', 'exact', 'points'))
        z = np.load(ent['src']['path'])
        stride = ent['stride']
        src_rows = CAT.src_rows(ent, k0, n)                       # factory npz rows behind stored frames k0..k0+n-1
        q = z['qpos'][src_rows].astype(np.float64)
        qd = dec['qpos'].astype(np.float64)
        r = dict(clip=cat.clips[c], robot=cat.bodies[b], frames=n, stride=stride, tier=sh.meta['tier'])
        r['root_pos_err_m'] = float(np.abs(qd[:, :3] - q[:, :3]).max())
        r['root_rot_err_rad'] = float(quat_angle(qd[:, 3:7], q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)).max())
        je = np.abs(qd[:, 7:] - q[:, 7:])
        r['joint_err_rad_max'] = float(je.max()); r['joint_err_rad_mean'] = float(je.mean())
        r['joint_step_rad_max'] = float(np.max(sh.meta['q_step']))
        # FK displacement
        if cat.bodies[b] not in fks:
            fks[cat.bodies[b]] = TorchFK.load(a.root / 'robots' / cat.bodies[b] / 'kin.npz', device=dev, dtype=torch.float64)
        fk = fks[cat.bodies[b]]
        dmax_pts = dmax_hull = 0.0; d99 = []
        clr_dec, clr_src = [], []
        for s0 in range(0, n, 512):
            Q1 = torch.tensor(q[s0:s0 + 512], device=dev); Q2 = torch.tensor(qd[s0:s0 + 512], device=dev)
            P1, P2 = fk.surface(Q1), fk.surface(Q2)
            dp = (P1 - P2).norm(dim=-1)
            dmax_pts = max(dmax_pts, float(dp.max())); d99.append(dp.flatten().cpu().numpy()[::7])
            H1, H2 = fk.hull(Q1), fk.hull(Q2)
            dmax_hull = max(dmax_hull, float((H1 - H2).norm(dim=-1).max()))
            clr_dec.append(fk.region_ground_clearance(Q2).cpu().numpy()); clr_src.append(fk.region_ground_clearance(Q1).cpu().numpy())
        clr_dec = np.concatenate(clr_dec); clr_src = np.concatenate(clr_src)
        r['fk_surface_disp_mm_max'] = dmax_pts * 1e3
        r['fk_surface_disp_mm_p99'] = float(np.percentile(np.concatenate(d99), 99) * 1e3)
        r['fk_hull_disp_mm_max'] = dmax_hull * 1e3
        bnd = json.load(open(a.root / 'robots' / cat.bodies[b] / 'bounds.json'))
        r['static_bound_mm'] = bnd['all_m'] * 1e3
        r['static_bound_holds'] = bool(max(dmax_pts, dmax_hull) <= bnd['all_m'])
        fin = np.isfinite(clr_src)
        r['region_clearance_change_mm_max'] = float(np.abs(clr_dec - clr_src)[fin].max() * 1e3)
        stored = z['ground_clearance'][src_rows]
        r['region_clearance_recompute_vs_factory_mm_max'] = float(np.abs(clr_src - stored)[fin].max() * 1e3)
        # strict-label re-check on decoded pose (only frames valid in the source)
        valid = dec['flags'] == 0
        gc = dec['contact']
        lowest = np.where(np.isfinite(clr_dec), clr_dec, np.inf).min(1)
        pen_bad = valid & (lowest < -PEN)
        cbad = valid & (gc & ~((clr_dec >= -PEN) & (clr_dec <= CTOL))).any(1)
        lowest0 = np.where(np.isfinite(clr_src), clr_src, np.inf).min(1)
        r['strict_violations_after_decode'] = dict(ground_pen_frames=int(pen_bad.sum()), contact_band_frames=int(cbad.sum()),
                                                   worst_pen_mm_decoded=float(max(0, -lowest[valid].min()) * 1e3),
                                                   worst_pen_mm_source=float(max(0, -lowest0[valid].min()) * 1e3))
        # labels bit-exact
        eq = {}
        for k_dec, k_src in [('contact', 'ground_contact_required' if 'ground_contact_required' in z.files else 'ground_contact'), ('slip', 'ground_slip'), ('human_contact', 'human_ground_contact'),
                             ('human_slip', 'human_ground_slip'), ('object_contact', 'object_contact_required' if 'object_contact_required' in z.files else 'object_contact'), ('object_slip', 'object_slip'),
                             ('human_object_contact', 'human_object_contact'), ('human_object_slip', 'human_object_slip')]:
            if k_dec in dec and k_src in z.files:
                eq[k_dec] = bool((dec[k_dec] == z[k_src][src_rows]).all())
        eq['factory_flags_bits_0_5'] = bool(((dec['flags'] & 63) == z['flags'][src_rows]).all())
        r['decode_guard_flagged_frames'] = int(((dec['flags'] & 64) > 0).sum())
        r['frames_valid_after_guard'] = int((dec['flags'] == 0).sum())
        if 'point_contact' in dec:
            eq['point_contact'] = bool((dec['point_contact'] == np.unpackbits(z['ground_point_contact'][src_rows], axis=1).astype(bool)).all())
        r['labels_bit_exact'] = eq
        if 'clearance' in dec:
            g = z['ground_clearance'][src_rows]; d = dec['clearance']
            band = np.isfinite(g) & (np.abs(g) < 0.02)
            far = np.isfinite(g) & (np.abs(g) >= 0.02) & (np.abs(g) < 3)
            r['sdist_u8_err'] = dict(band_lt_2cm_mm_max=float(np.abs(d - g)[band].max() * 1e3) if band.any() else None,
                                     rel_err_far_max=float((np.abs(d - g)[far] / np.abs(g[far])).max()) if far.any() else None)
            sp, spd = z['ground_slip_speed'][src_rows], dec['slip_speed']
            m = np.isfinite(sp) & (sp > 1e-3)
            r['speed_u8_rel_err_max'] = float((np.abs(spd - sp)[m] / sp[m]).max()) if m.any() else None
            r['speed_nan_pattern_equal'] = bool((np.isnan(spd) == np.isnan(sp)).all())
        if 'object_pos' in z.files:
            hum = rd.human(c, k0, n)
            p = hum['object_pos'][:, 0].astype(np.float64) * ent['scale']; p[:, 2] -= ent['floor_offset']
            op = z['object_pos'][src_rows].astype(np.float64)
            r['object_robot_frame_pos_err_mm'] = float(np.abs(p - op).max() * 1e3)
            R = S.quat_to_rotmat(hum['object_quat'][:, 0])
            r['object_rot_err_rad'] = float(quat_angle(hum['object_quat'][:, 0], S.rotmat_to_quat(z['object_rot'][src_rows].astype(np.float64))).max())
        rep['robot'].append(r)
        print(json.dumps(r))
    # ------------------------------------------------------------------ human
    try:
        import smplx
        model = smplx.create(a.smplx_model, model_type='smplx', gender='neutral', use_pca=False, num_betas=16,
                             flat_hand_mean=True, batch_size=1).to(dev)
    except Exception as ex:
        model = None; print('smplx unavailable:', ex)
    import surface_retargeting.dataset.sources as SRC
    for e in range(len(cat.e_clip)):
        if int(cat.e_body[e]) != 0:
            continue
        c = int(cat.e_clip[e]); n = int(cat.e_n[e]); rel = cat.shards[int(cat.e_shard[e])]
        ent = next(x for x in _shard_entries(a.root, rel, man['shards'][rel]['sha256'])['entries'] if x['clip'] == cat.clips[c])
        hd = rd.human(c, 0, n)
        # the source again, through the same resolver the converter used
        rob = next(x for x in range(len(cat.e_clip)) if int(cat.e_clip[x]) == c and int(cat.e_body[x]) != 0)
        rrel = cat.shards[int(cat.e_shard[rob])]
        rent = next(x for x in _shard_entries(a.root, rrel, man['shards'][rrel]['sha256'])['entries'] if x['clip'] == cat.clips[c])
        prov = json.loads(str(np.load(rent['src']['path'])['provenance']))
        h = SRC.load_human(prov)
        kk = np.arange(0, len(h['poses']), ent['stride'])
        P = h['poses'][kk]
        dec_full = np.concatenate([hd['global_orient'], hd['body_pose'].reshape(n, -1), hd['jaw_pose'], hd['leye_pose'],
                                   hd['reye_pose'], hd['left_hand_pose'].reshape(n, -1), hd['right_hand_pose'].reshape(n, -1)], 1)
        r = dict(clip=cat.clips[c], frames=n, has_extra=ent['has_extra'],
                 pose_err_rad_max=float(np.abs(dec_full[:, :P.shape[1]] - P).max()),
                 transl_err_m=float(np.abs(hd['transl'] - h['trans'][kk]).max()))
        if model is not None:
            bt = torch.zeros(1, 16, device=dev); b0 = np.asarray(h['betas'])[:16]; bt[0, :len(b0)] = torch.tensor(b0)
            vmax = jmax = 0.0
            for s0 in range(0, n, 256):
                s1 = min(n, s0 + 256); B = s1 - s0
                def run(pp, tr):
                    pp = torch.tensor(pp, device=dev, dtype=torch.float32)
                    return model(betas=bt.expand(B, -1), global_orient=pp[:, :3], body_pose=pp[:, 3:66], jaw_pose=pp[:, 66:69],
                                 leye_pose=pp[:, 69:72], reye_pose=pp[:, 72:75], left_hand_pose=pp[:, 75:120],
                                 right_hand_pose=pp[:, 120:165], transl=torch.tensor(np.ascontiguousarray(tr), device=dev, dtype=torch.float32),
                                 expression=torch.zeros(B, 10, device=dev))
                Pf = np.zeros((B, 165), np.float32); Pf[:, :P.shape[1]] = P[s0:s1]
                o1 = run(Pf, h['trans'][kk][s0:s1]); o2 = run(dec_full[s0:s1], hd['transl'][s0:s1])
                vmax = max(vmax, float((o1.vertices - o2.vertices).norm(dim=-1).max()))
                jmax = max(jmax, float((o1.joints - o2.joints).norm(dim=-1).max()))
            r['smplx_vertex_disp_mm_max'] = vmax * 1e3; r['smplx_joint_disp_mm_max'] = jmax * 1e3
        rep['human'].append(r)
        print(json.dumps(r))
    summ = dict(
        robot_entries=len(rep['robot']),
        joint_err_rad_max=max(x['joint_err_rad_max'] for x in rep['robot']),
        root_rot_err_rad_max=max(x['root_rot_err_rad'] for x in rep['robot']),
        root_pos_err_m_max=max(x['root_pos_err_m'] for x in rep['robot']),
        fk_surface_disp_mm_max=max(x['fk_surface_disp_mm_max'] for x in rep['robot']),
        fk_hull_disp_mm_max=max(x['fk_hull_disp_mm_max'] for x in rep['robot']),
        region_clearance_change_mm_max=max(x['region_clearance_change_mm_max'] for x in rep['robot']),
        strict_violations_after_decode=sum(x['strict_violations_after_decode']['ground_pen_frames'] +
                                           x['strict_violations_after_decode']['contact_band_frames'] for x in rep['robot']),
        labels_all_bit_exact=all(all(x['labels_bit_exact'].values()) for x in rep['robot']),
        static_bound_holds_all=all(x['static_bound_holds'] for x in rep['robot']),
        decode_guard_flagged_frames=sum(x['decode_guard_flagged_frames'] for x in rep['robot']),
        frames_valid_after_guard=sum(x['frames_valid_after_guard'] for x in rep['robot']),
        robot_frames=sum(x['frames'] for x in rep['robot']),
        sdist_band_lt_2cm_mm_max=max((x['sdist_u8_err']['band_lt_2cm_mm_max'] or 0) for x in rep['robot'] if 'sdist_u8_err' in x),
        human_transl_err_m_max=max((x['transl_err_m'] for x in rep['human']), default=None),
        human_joint_disp_mm_max=max((x.get('smplx_joint_disp_mm_max', -1) for x in rep['human']), default=None),
        object_rot_err_rad_max=max((x.get('object_rot_err_rad', 0) for x in rep['robot']), default=None),
        human_pose_err_rad_max=max((x['pose_err_rad_max'] for x in rep['human']), default=None),
        human_vertex_disp_mm_max=max((x.get('smplx_vertex_disp_mm_max', -1) for x in rep['human']), default=None),
        object_pos_err_mm_max=max((x.get('object_robot_frame_pos_err_mm', 0) for x in rep['robot']), default=None))
    rep['summary'] = summ
    print('SUMMARY', json.dumps(summ, indent=1))
    out = a.out or (a.root / 'qa' / 'roundtrip.json')
    out.parent.mkdir(exist_ok=True)
    json.dump(rep, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
