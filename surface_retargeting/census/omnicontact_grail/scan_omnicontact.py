# Read-only diversity scan of OmniContact converted SMPL-X captures + raw BVH headers (no GPU).
import json, glob, os, re, hashlib, csv, time
import numpy as np

S = '/home/arenalabs/umr_factory_data/smplx/omnicontact'
R = '/home/arenalabs/umr_factory_data/raw/omnicontact'
OUT = '/home/arenalabs/umr_factory_data/diversity/omnicontact_grail/omnicontact_clip_stats.jsonl'


def rodrigues(aa):
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    k = aa / np.maximum(th, 1e-8)
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[:, :, None]
    return np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def bvh_skeleton(path):
    """Parse only the HIERARCHY header: rest-pose joint positions -> skeleton hash + stature estimate (cm)."""
    lines = []
    with open(path, 'r', errors='ignore') as fh:
        for ln in fh:
            if ln.strip().startswith('MOTION'):
                break
            lines.append(ln.strip())
    stack, pos, names, cur = [], [], [], np.zeros(3)
    pending = None
    offs = []
    for ln in lines:
        tok = ln.split()
        if not tok:
            continue
        if tok[0] in ('ROOT', 'JOINT'):
            pending = tok[1]
        elif tok[0] == 'End':
            pending = (stack[-1][0] if stack else 'root') + '_end'
        elif tok[0] == '{':
            stack.append((pending, None))
        elif tok[0] == 'OFFSET':
            o = np.array([float(x) for x in tok[1:4]])
            offs.append(o)
            parent = stack[-2][1] if len(stack) >= 2 else np.zeros(3)
            p = parent + o
            stack[-1] = (stack[-1][0], p)
            pos.append(p)
            names.append(stack[-1][0])
        elif tok[0] == '}':
            stack.pop()
    P = np.array(pos)
    h = hashlib.md5(np.round(np.array(offs), 2).tobytes()).hexdigest()[:12]
    return {'skel_hash': h, 'skel_stature_cm': float(P[:, 1].max() - P[:, 1].min()), 'n_joints': len(names)}


def main():
    splits = {r['capture_id']: r['split'] for r in csv.DictReader(open(f'{R}/metadata/splits.csv'))}
    g1 = json.load(open(f'{R}/g1_labels/summary.json'))
    rows = []
    t0 = time.time()
    for d in sorted(glob.glob(f'{S}/*/*/*/')):
        d = d.rstrip('/')
        dom, case, cid = d.split('/')[-3:]
        qa = json.load(open(f'{d}/qa.json')) if os.path.exists(f'{d}/qa.json') else {}
        op = json.load(open(f'{d}/obj_pen.json')) if os.path.exists(f'{d}/obj_pen.json') else {}
        r = {'clip': f'{dom}/{case}/{cid}', 'domain': dom, 'case': case, 'capture_id': cid,
             'split': splits.get(cid), 'ok': qa.get('ok'), 'problems': qa.get('problems'),
             'frames': qa.get('frames'), 'fps': qa.get('fps'), 'seconds': qa.get('seconds'),
             'ik_residual_cm_mean': (qa.get('ik_residual_cm') or {}).get('mean'),
             'objects': [o.get('object_id') for o in qa.get('objects', [])],
             'obj_resting_frac': [o.get('obj_resting_frac') for o in qa.get('objects', [])],
             'betas': qa.get('betas')}
        try:
            fps = float(qa.get('fps') or np.load(f'{d}/mocap_framerate.npy'))
            t = np.load(f'{d}/transl.npy').astype(np.float64)  # y-up
            aa = np.load(f'{d}/poses.npy', mmap_mode='r')[:, :3].astype(np.float64)
            Rm = rodrigues(aa)
            fwd = Rm @ np.array([0, 0, 1.0])
            yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
            up = Rm @ np.array([0, 1.0, 0])
            T = len(t)
            y0 = float(np.median(t[:, 1]))
            dxz = np.linalg.norm(np.diff(t[:, [0, 2]], axis=0), axis=1)
            k = max(1, int(round(fps / 5)))
            sp = np.linalg.norm(t[k:, [0, 2]] - t[:-k, [0, 2]], axis=1) * fps / k
            r.update({'T': T, 'seconds_T': T / fps,
                      'pelvis_drop_from_median_m': float(y0 - t[:, 1].min()),
                      'frac_pelvis_drop_gt_0p30': float(np.mean(y0 - t[:, 1] > 0.30)),
                      'frac_pelvis_drop_gt_0p45': float(np.mean(y0 - t[:, 1] > 0.45)),
                      'path_xz_m': float(dxz.sum()),
                      'speed_p50': float(np.median(sp)), 'speed_p95': float(np.percentile(sp, 95)),
                      'frac_speed_gt_0p3': float(np.mean(sp > 0.3)), 'frac_speed_gt_2p0': float(np.mean(sp > 2.0)),
                      'yaw_total_deg': float(np.degrees(np.abs(np.diff(yaw)).sum())),
                      'torso_tilt_p95_deg': float(np.percentile(np.degrees(np.arccos(np.clip(up[:, 1], -1, 1))), 95)),
                      'torso_tilt_max_deg': float(np.degrees(np.arccos(np.clip(up[:, 1], -1, 1))).max())})
        except Exception as e:
            r['motion_err'] = repr(e)[:200]
        objs = op.get('objects') or []
        if objs:
            o = objs[0]
            r['obj_within_2cm_frac_sampled'] = o['frames_within_2cm'] / max(1, op.get('frames_sampled', 1))
            r['obj_pen_max_cm'] = {k: o[k]['pen_max_cm'] for k in ('hand', 'forearm', 'torso', 'leg') if k in o}
        g = g1.get(f'{dom}/{case}/{cid}')
        if g is None and os.path.exists(f'{R}/g1_labels/{dom}/{case}/{cid}.npz'):
            with np.load(f'{R}/g1_labels/{dom}/{case}/{cid}.npz') as z:
                ci = z['contact_info'].reshape(len(z['contact_info']), -1)
                g = {'contact_frac_lA_rA_lW_rW': [round(float(x), 3) for x in ci.mean(0)]}
        r['g1_obj_contact_frac_lA_rA_lW_rW'] = g.get('contact_frac_lA_rA_lW_rW') if g else None
        bvh = f'{R}/raw_mocap/{dom}/{case}/{cid}/motion_actor.bvh'
        if os.path.exists(bvh):
            r.update(bvh_skeleton(bvh))
        rows.append(r)
    with open(OUT, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print('rows', len(rows), f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
