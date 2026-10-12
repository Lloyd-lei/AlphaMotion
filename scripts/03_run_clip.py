"""Retarget one clip onto one or more robots with UMR, run the strict post-pass + labels, write outputs.
Usage:
  python scripts/run_clip.py --kind smplx --data <dir or npz> --seq <key> --robots unitree_g1_29dof,fourier_gr3
  python scripts/run_clip.py --kind hoi   --data <seq parent dir> --seq <key> --robots ...
Each robot runs in sequence (one GPU correspondence job at a time). Updates OUT/manifest.jsonl.
"""
import argparse, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_retargeting.retarget import pipeline as P
from surface_retargeting.robots import robots as R
from surface_retargeting.paths import OUT

ap = argparse.ArgumentParser()
ap.add_argument('--kind', choices=['smplx', 'hoi'], required=True)
ap.add_argument('--data', required=True); ap.add_argument('--seq', required=True)
ap.add_argument('--robots', required=True); ap.add_argument('--force-umr', action='store_true')
ap.add_argument('--umr-timeout', type=int, default=3600)
a = ap.parse_args()
inv = {r['name']: r for r in R.load_inventory()}
for robot in a.robots.split(','):
    t0 = time.time()
    row = dict(robot=robot, kind=a.kind, data=a.data, seq=a.seq, started=time.strftime('%F %T'))
    try:
        up = P.run_umr(robot, a.kind, a.data, a.seq, timeout=a.umr_timeout, force=a.force_umr)
        print(f'[run_clip] {robot}: UMR done ({up.get("umr_seconds", "cached")} s)', flush=True)
        qa = P.postprocess(robot, a.kind, a.data, a.seq, up, inv[robot])
        row.update(status='ok' if qa['frames_flagged_final'] == 0 else 'ok_with_flagged_frames', clip=qa['clip'],
                   npz=qa['npz'], frames=qa['frames'], flagged=qa['frames_flagged_final'],
                   pen_after=qa['penetration_after'], seconds=round(time.time() - t0, 1))
        print(json.dumps({k: qa[k] for k in ('clip', 'robot', 'frames', 'penetration_before', 'penetration_after',
                                             'frames_flagged_final', 'flags_final_hist', 'seconds')}, default=str), flush=True)
    except Exception as e:
        row.update(status='error', error=repr(e), tb=traceback.format_exc()[-2000:])
        print('[run_clip] ERROR', robot, repr(e), flush=True)
        traceback.print_exc()
    with open(OUT / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
