"""unitree_h2 dedicated pipeline -- step 3: run one QA clip through the shared CPU reference engine
(umrf.pipeline.run_umr -> umrf.pipeline.postprocess, i.e. what scripts/run_clip.py calls) with:
  * UMR on CPU (CUDA_VISIBLE_DEVICES=''), frame window from pipeline.json
  * outputs redirected to robots/unitree_h2/pipeline/{runs,out}/ (shared runs/ out/ manifest untouched)
  * HOI clips use the shared beta0 correspondence (no per-clip training)
Usage: <umr python> run_qa.py <clip_id>      (clip ids in pipeline.json qa_clips)"""
import json, os, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MUJOCO_GL', 'egl')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '8')
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P, robots as R
from umrf.paths import UMR_ROOT

ROBOT = 'unitree_h2'
PD = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT / 'pipeline'
cfg = json.load(open(PD / 'pipeline.json'))
clip = next(c for c in cfg['qa_clips'] if c['id'] == sys.argv[1])
P.RUNS = PD / 'runs'; P.OUT = PD / 'out'
P.RUNS.mkdir(exist_ok=True); P.OUT.mkdir(exist_ok=True)
cid = f"{ROBOT}_qa_{clip['id']}__f{clip['start']}-{clip['end']}s{clip['stride']}"
if clip['kind'] == 'smplx':
    # per-robot defaults: UMR smplx defaults + window + SMPL-X forward on CPU
    dflt = P._absolutize(json.load(open(P.DEFAULTS['smplx'])), UMR_ROOT)
    dflt['motion'].update(start=clip['start'], end=clip['end'], stride=clip['stride'])
    dflt['retarget']['smplx_device'] = 'cpu'
    dp = PD / 'umr_defaults_smplx_window.json'
    json.dump(dflt, open(dp, 'w'), indent=1)
    P.DEFAULTS['smplx'] = dp
inv = {r['name']: r for r in R.load_inventory()}
t0 = time.time()
up = P.run_umr(ROBOT, clip['kind'], clip['data'], clip['seq'], timeout=3600, shared_corr=clip.get('shared_corr', False),
               start=clip['start'] if clip['kind'] == 'hoi' else None, end=clip['end'] if clip['kind'] == 'hoi' else None,
               stride=clip['stride'] if clip['kind'] == 'hoi' else None, cid=cid)
print(f'[run_qa] UMR done {up.get("umr_seconds")} s', flush=True)
qa = P.postprocess(ROBOT, clip['kind'], clip['data'], clip['seq'], up, inv[ROBOT], log_every=200)
row = dict(clip_id=clip['id'], cid=cid, npz=qa['npz'], frames=qa['frames'], flagged=qa['frames_flagged_final'],
           flags_hist=qa['flags_final_hist'], pen_after=qa['penetration_after'], seconds=round(time.time() - t0, 1),
           umr=up)
with open(PD / 'runs_manifest.jsonl', 'a') as f:
    f.write(json.dumps(row, default=str) + '\n')
print(json.dumps(row, default=str, indent=1))
