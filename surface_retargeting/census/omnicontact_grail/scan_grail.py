from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
# Read-only diversity scan of GRAIL converted SMPL-X clips (no GPU). Writes per-clip stats JSONL.
import json, glob, os, sys, time, re
import numpy as np, joblib

ROOT = f'{SRCFG.data_root}/smplx/grail'
OUT = f'{SRCFG.data_root}/diversity/omnicontact_grail/grail_clip_stats.jsonl'
CATS = ['curb', 'slope', 'sitting', 'stair_p1', 'stair_p2', 'pickup_table', 'pickup_ground']
FPS = 24.0


def rodrigues(aa):
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    k = aa / np.maximum(th, 1e-8)
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[:, :, None]
    return np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def stats(path, cat):
    d = joblib.load(path)
    h = d['human_data']
    t = np.asarray(h['trans'], np.float64)
    T = len(t)
    p = np.asarray(h['poses'], np.float64)
    R = rodrigues(p[:, :3])
    fwd = R @ np.array([0, 0, 1.0])  # SMPL body forward (+z local)
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    up_body = R @ np.array([0, 1.0, 0])  # SMPL body up (+y local); z-up world
    n0 = max(1, min(12, T // 10))
    z0 = float(np.median(t[:max(n0, 24) if T > 48 else n0, 2]))
    dxy = np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1)
    sp = dxy * FPS
    sp_s = np.convolve(sp, np.ones(6) / 6, mode='same') if len(sp) > 6 else sp
    fcp = h.get('foot_contact_probs')
    uf = d.get('umrf_frame') or {}
    sv = uf.get('src_valid')
    b = np.asarray(h['betas']).reshape(-1)[:10]
    return {
        'clip': os.path.basename(path)[:-4], 'category': cat, 'frames': T, 'seconds': T / FPS,
        'dz_end_start_m': float(np.mean(t[-n0:, 2]) - np.mean(t[:n0, 2])),
        'z_range_m': float(t[:, 2].max() - t[:, 2].min()),
        'pelvis_drop_m': float(z0 - t[:, 2].min()),
        'path_xy_m': float(dxy.sum()),
        'net_xy_m': float(np.linalg.norm(t[-1, :2] - t[0, :2])),
        'speed_p50': float(np.median(sp_s)) if len(sp_s) else 0.0,
        'speed_p95': float(np.percentile(sp_s, 95)) if len(sp_s) else 0.0,
        'yaw_net_deg': float(np.degrees(yaw[-1] - yaw[0])),
        'yaw_total_deg': float(np.degrees(np.abs(np.diff(yaw)).sum())),
        'torso_tilt_max_deg': float(np.degrees(np.arccos(np.clip(up_body[:, 2], -1, 1))).max()),
        'betas': [round(float(x), 4) for x in b],
        'pred_height_m': float(h.get('predicted_body_height', np.nan)),
        'foot_contact_mean': None if fcp is None else [round(float(x), 3) for x in np.asarray(fcp).mean(0)],
        'src_valid_frac': None if sv is None else float(np.mean(np.asarray(sv))),
    }


def main():
    t0 = time.time()
    n = 0
    with open(OUT + '.tmp', 'w') as fh:
        for c in CATS:
            fs = sorted(glob.glob(f'{ROOT}/{c}/recon/*.pkl'))
            for f in fs:
                try:
                    r = stats(f, c)
                except Exception as e:
                    r = {'clip': os.path.basename(f)[:-4], 'category': c, 'error': repr(e)[:200]}
                fh.write(json.dumps(r) + '\n')
                n += 1
            print(c, len(fs), f'{time.time()-t0:.0f}s', flush=True)
    os.replace(OUT + '.tmp', OUT)
    print('done', n)


if __name__ == '__main__':
    main()
