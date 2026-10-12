"""Build + train (cache) the UMR point-cloud correspondence for every core robot against the neutral-shape SMPL-X
template used by plain SMPL-X sources (LAFAN1 sample).  ~10 s CPU build + ~12-20 s GPU train per robot.
HOI / HiPHI sources with their own body shape get their own cache on first use (pipeline.run_umr).
Usage: python scripts/build_correspondence.py [robot1,robot2]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_retargeting.retarget import pipeline as P
from surface_retargeting.robots import robots as R
from surface_retargeting.paths import CORR, UMR_ROOT
only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
rep = {}
for rec in R.load_inventory():
    n = rec['name']
    if only and n not in only:
        continue
    t = time.time()
    try:
        P.run_umr(n, 'smplx', str(UMR_ROOT / 'sample_data/lafan1_smplx'), 'dance1_subject2', timeout=900, stage='train')
        ok = (CORR / n / 'smplx' / 'train' / 'correspondence_slots_final.npz').exists()
        rep[n] = dict(ok=ok, seconds=round(time.time() - t, 1))
    except Exception as e:
        rep[n] = dict(ok=False, error=repr(e), seconds=round(time.time() - t, 1))
    print(n, rep[n], flush=True)
p = CORR / 'correspondence_summary.json'
old = json.load(open(p)) if p.exists() else {}
old.update(rep); json.dump(old, open(p, 'w'), indent=1)
