"""unitree_h1 dedicated pipeline -- step 3a: run the fixed QA clips through the CPU reference factory
(umrf.pipeline.run_umr + postprocess, i.e. what scripts/run_clip.py does) with per-robot windows and per-robot output
dirs.  Shared modules are not edited: RUNS/OUT/DEFAULTS are rebound at runtime only.  GPU is hidden
(CUDA_VISIBLE_DEVICES='') so UMR's SMPL-X forward runs on CPU.
Usage: <umr py> run_qa_clips.py dance[,kick,box_carry,kneel]"""
import json, os, sys, time, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P, robots as R
from umrf.paths import UMR_ROOT

ROBOT = 'unitree_h1'
PD = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT / 'pipeline'
cfg = json.load(open(PD / 'pipeline.json'))
P.RUNS = PD / 'runs'; P.OUT = PD / 'out'
P.RUNS.mkdir(exist_ok=True); P.OUT.mkdir(exist_ok=True)
inv = {r['name']: r for r in R.load_inventory()}
want = sys.argv[1].split(',')
for c in cfg['qa_clips']:
    if c['id'] not in want:
        continue
    t0 = time.time(); row = dict(robot=ROBOT, **c, started=time.strftime('%F %T'))
    try:
        base = json.load(open(UMR_ROOT / ('humanoid_retarget_defaults.json' if c['kind'] == 'smplx'
                                          else 'humanoid_retarget_defaults_hsi_hoi_standard.json')))
        base.setdefault('retarget', {})['smplx_device'] = 'cpu'
        if c['kind'] == 'smplx':
            base['motion'].update(start=c['start'], end=c['end'], stride=c['stride'])
        dp = PD / f"defaults_{c['id']}.json"; json.dump(base, open(dp, 'w'), indent=1)
        P.DEFAULTS = dict(P.DEFAULTS, **{c['kind']: dp})
        kw = dict(cid=c['cid'])
        if c['kind'] == 'hoi':
            kw.update(shared_corr=True, start=c['start'], end=c['end'], stride=c['stride'])
        up = P.run_umr(ROBOT, c['kind'], c['data'], c['seq'], timeout=5400, **kw)
        print(f'[h1] {c["id"]}: UMR done ({up.get("umr_seconds", "cached")} s)', flush=True)
        qa = P.postprocess(ROBOT, c['kind'], c['data'], c['seq'], up, inv[ROBOT])
        row.update(status='ok' if qa['frames_flagged_final'] == 0 else 'ok_with_flagged_frames', npz=qa['npz'],
                   frames=qa['frames'], flagged=qa['frames_flagged_final'], flags_hist=qa['flags_final_hist'],
                   pen_after=qa['penetration_after'], umr_seconds=up.get('umr_seconds'), seconds=round(time.time() - t0, 1))
        print(json.dumps({k: qa[k] for k in ('clip', 'frames', 'penetration_after', 'frames_flagged_final', 'flags_final_hist', 'seconds')}, default=str), flush=True)
    except Exception as e:
        row.update(status='error', error=repr(e), tb=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))
        print('[h1] ERROR', c['id'], repr(e), flush=True); traceback.print_exc()
    with open(PD / 'runs_manifest.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
