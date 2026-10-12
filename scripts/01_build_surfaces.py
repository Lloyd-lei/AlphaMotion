"""Build DATA/robots/<name>/surface.npz for all core robots + a region-coloured check image per robot."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from surface_retargeting.robots import robots as R, surface as S
from surface_retargeting.paths import ROBOTS
only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
rep = {}
for rec in R.load_inventory():
    if only and rec['name'] not in only: continue
    t = time.time(); o = S.build_surface(rec['name'], rec)
    cnt = {r: int((o['region'] == i).sum()) for i, r in enumerate(S.REGIONS)}
    rep[rec['name']] = dict(n=len(o['geom']), hull_vertices=len(o['hull_geom']), region_counts=cnt, s=round(time.time()-t,1))
    print(rec['name'], rep[rec['name']], flush=True)
p = ROBOTS / 'surface_summary.json'
old = json.load(open(p)) if p.exists() else {}
old.update(rep); json.dump(old, open(p, 'w'), indent=1)
