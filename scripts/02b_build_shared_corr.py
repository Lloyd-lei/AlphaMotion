"""Build + train the beta-zero shared correspondence for each robot (sequential; one GPU job at a time).
Usage: python scripts/build_shared_corr.py [robot1,robot2]"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from surface_retargeting.robots import robots as R
from surface_retargeting.correspondence import shared_corr as SC
from surface_retargeting.retarget import pipeline as P
from surface_retargeting.paths import UMR_ROOT, UMR_PY, UMR_TOOLS, CORR

only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
rep = {}
for rec in R.load_inventory():
    n = rec['name']
    if only and n not in only:
        continue
    p = SC.paths(n); p['dir'].mkdir(parents=True, exist_ok=True)
    t = time.time()
    rcfg = SC.robot_cfg_for_training(n)
    json.dump(rcfg, open(p['train_cfg'], 'w'), indent=1)
    dflt = P._absolutize(json.load(open(UMR_ROOT / 'humanoid_retarget_defaults.json')), UMR_ROOT)
    dflt['smpl_template'] = rcfg['smpl_template']
    dflt['motion'].update(data=str(UMR_ROOT / 'sample_data/lafan1_smplx'), seq_key='dance1_subject2')
    dflt.setdefault('view', {})['enabled'] = False
    dpath = p['dir'] / 'defaults.json'; json.dump(dflt, open(dpath, 'w'), indent=1)
    log = p['dir'] / 'train.log'
    cmd = ['timeout', '900', UMR_PY, str(UMR_TOOLS / 'umr_pipe.py'), 'humanoid_retarget_pipeline', '--config',
           str(p['train_cfg']), '--defaults', str(dpath), '--skip-view', '--stage', 'train']
    with open(log, 'w') as lf:
        rc = subprocess.run(cmd, cwd=str(UMR_ROOT), stdout=lf, stderr=subprocess.STDOUT,
                            env=dict(os.environ, MUJOCO_GL='egl')).returncode
    ok = p['slots'].exists()
    names = [str(x) for x in np.load(p['slots'], allow_pickle=True)['names']] if ok else []
    rep[n] = dict(ok=ok and SC.BETA0_NAME in names, rc=rc, names=names, seconds=round(time.time() - t, 1))
    print(n, rep[n], flush=True)
s = CORR / 'shared_beta0_summary.json'
old = json.load(open(s)) if s.exists() else {}
old.update(rep); json.dump(old, open(s, 'w'), indent=1)
