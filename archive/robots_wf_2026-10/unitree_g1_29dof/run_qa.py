"""unitree_g1_29dof dedicated pipeline -- run the fixed QA clip set through the shared CPU reference engine.

Same body as scripts/run_clip.py (umrf.pipeline.run_umr + umrf.pipeline.postprocess), plus:
  * frame window per clip (HOI: UMR --start/--end/--stride; SMPL-X: sliced copy of the source npz in pipeline/src),
  * RUNS/OUT redirected into pipeline/runs and pipeline/out (nothing written to the shared factory out/),
  * UMR forced to CPU (CUDA_VISIBLE_DEVICES=''),
  * waits for >= 12 GB available RAM before each clip (one heavy process at a time).
Usage: <umr env python> run_qa.py qa_dance [qa_kick ...]
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P, robots as R

ROBOT = 'unitree_g1_29dof'
PD = Path('/home/arenalabs/umr_factory_data/robots') / ROBOT / 'pipeline'
CFG = json.load(open(PD / 'pipeline.json'))
P.RUNS = PD / 'runs'; P.OUT = PD / 'out'
P.RUNS.mkdir(exist_ok=True); P.OUT.mkdir(exist_ok=True)


def avail_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable'):
            return int(ln.split()[1]) / 1e6
    return 0.0


def wait_mem(need=12.0, timeout=1800):
    t0 = time.time()
    while avail_gb() < need:
        if time.time() - t0 > timeout:
            raise RuntimeError(f'RAM available {avail_gb():.1f} GB < {need} GB for {timeout} s')
        print(f'[run_qa] waiting for RAM: {avail_gb():.1f} GB available', flush=True)
        time.sleep(30)


def prepare(c):
    """Return (data, seq, start, end, stride) for the engine."""
    a, b, s = c['window']
    if c['kind'] == 'smplx':
        src = Path(c['data']) / f"{c['seq']}.npz"
        dst_dir = PD / 'src' / f"lafan1_smplx_f{a}-{b}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{c['seq']}.npz"
        if not dst.exists():
            z = np.load(src, allow_pickle=True)
            out = {k: z[k] for k in z.files}
            for k in ('trans', 'poses'):
                out[k] = z[k][a:b:s]
            np.savez(dst, **out)
        return str(dst_dir), c['seq'], None, None, None
    if c['id'] == 'qa_ground':                       # private copy of the HiPHI sequence + object mesh
        src_root = Path('/home/arenalabs/umr_factory_data/smplx/hiphi')
        seq_src = src_root / 'data/Cause_motion/push' / c['seq']
        seq_dst = Path(c['data']) / c['seq']
        if not (seq_dst / 'metadata.json').exists():
            shutil.copytree(seq_src, seq_dst, dirs_exist_ok=True)
            meta = json.load(open(seq_dst / 'metadata.json'))
            om = PD / 'src/hiphi/object_meshes'; om.mkdir(parents=True, exist_ok=True)
            for o in meta['objects']:
                shutil.copy2(src_root / 'object_meshes' / f"{o['mesh_id']}.obj", om / f"{o['mesh_id']}.obj")
    return c['data'], c['seq'], a, b, s


inv = {r['name']: r for r in R.load_inventory()}
clips = {c['id']: c for c in CFG['qa_clips']}
for cid in sys.argv[1:]:
    c = clips[cid]
    row = dict(robot=ROBOT, clip=cid, kind=c['kind'], started=time.strftime('%F %T'))
    t0 = time.time()
    try:
        data, seq, a, b, s = prepare(c)
        wait_mem()
        tag = f"{cid}__{seq}__f{c['window'][0]}-{c['window'][1]}s{c['window'][2]}"
        up = P.run_umr(ROBOT, c['kind'], data, seq, timeout=3000, shared_corr=c.get('shared_corr', False),
                       start=a, end=b, stride=s, cid=tag)
        print(f'[run_qa] {cid}: UMR done ({up.get("umr_seconds", "cached")} s)', flush=True)
        wait_mem()
        qa = P.postprocess(ROBOT, c['kind'], data, seq, up, inv[ROBOT])
        row.update(status='ok' if qa['frames_flagged_final'] == 0 else 'ok_with_flagged_frames', engine_clip=tag,
                   npz=qa['npz'], frames=qa['frames'], flagged=qa['frames_flagged_final'],
                   flags_hist=qa['flags_final_hist'], pen_after=qa['penetration_after'],
                   umr_seconds=up.get('umr_seconds'), seconds=round(time.time() - t0, 1))
        print(json.dumps({k: row[k] for k in ('clip', 'frames', 'flagged', 'flags_hist', 'seconds')}), flush=True)
    except Exception as e:
        row.update(status='error', error=repr(e), tb=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))
        print('[run_qa] ERROR', cid, repr(e), flush=True); traceback.print_exc()
    with open(PD / 'out' / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
