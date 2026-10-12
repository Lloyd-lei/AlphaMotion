"""Step 3a -- run the fixed QA clip set through the CPU reference engine (umrf: UMR + strict post-pass + labels) for
unitree_r1 ONLY.  Same engine calls as scripts/run_clip.py (pipeline.run_umr + pipeline.postprocess) plus the frame
windows run_clip.py does not expose; all outputs are redirected under robots/unitree_r1/pipeline/{runs,out}.
CUDA is hidden (UMR's SMPL-X forward falls back to CPU) -- this robot run uses no GPU compute.
Usage: timeout 5400 <umr env python> run_qa_clips.py lafan_dance|soccer_kick|box_carry|hiphi_kneel [--force]"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MUJOCO_GL', 'egl')
import json
import shutil
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
import numpy as np
import r1_common as C
from umrf import pipeline as P, robots as R

P.RUNS = C.PDIR / 'runs'
P.OUT = C.PDIR / 'out'
P.RUNS.mkdir(parents=True, exist_ok=True); P.OUT.mkdir(parents=True, exist_ok=True)
SRC = C.PDIR / 'src'
cfg = json.load(open(C.PDIR / 'pipeline.json'))
clips = {c['id']: c for c in cfg['qa_clips']}


def mem_avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def wait_mem(need=12.0, timeout_s=1800):
    t0 = time.time()
    while mem_avail_gb() < need:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f'MemAvailable stayed < {need} GB for {timeout_s} s')
        print(f'[r1] waiting for memory: {mem_avail_gb():.1f} GB available', flush=True)
        time.sleep(30)
    print(f'[r1] MemAvailable {mem_avail_gb():.1f} GB', flush=True)


def prep_lafan(c):
    a, b = c['window_frames']
    d = SRC / f'lafan1_dance1_f{a}-{b}'
    d.mkdir(parents=True, exist_ok=True)
    f = d / 'dance1_subject2.npz'
    if not f.exists():
        z = np.load(c['source'], allow_pickle=True)
        np.savez(f, trans=z['trans'][a:b], poses=z['poses'][a:b], betas=z['betas'], gender=z['gender'],
                 mocap_frame_rate=z['mocap_frame_rate'])
    return str(d), 'dance1_subject2', f'lafan1_dance1_subject2__f{a}-{b}'


def prep_hiphi(c):
    root = SRC / 'hiphi'
    seq_src = Path(c['data']) / c['seq']
    seq_dst = root / 'data' / Path(c['data']).relative_to('/home/arenalabs/umr_factory_data/smplx/hiphi/data') / c['seq']
    if not (seq_dst / 'metadata.json').exists():
        shutil.copytree(seq_src, seq_dst, dirs_exist_ok=True)
    meta = json.load(open(seq_dst / 'metadata.json'))
    (root / 'object_meshes').mkdir(parents=True, exist_ok=True)
    for o in meta['objects']:
        s = Path('/home/arenalabs/umr_factory_data/smplx/hiphi') / o['mesh_path']
        dst = root / o['mesh_path']
        if not dst.exists():
            shutil.copy2(s, dst)                       # real copy: UMR resolves symlinks and would write next to the source
    return str(seq_dst.parent), c['seq']


def main():
    cid_arg = sys.argv[1]
    force = '--force' in sys.argv
    c = clips[cid_arg]
    inv = {r['name']: r for r in R.load_inventory()}
    t0 = time.time()
    row = dict(robot=C.ROBOT, qa_id=cid_arg, kind=c['kind'], started=time.strftime('%F %T'))
    try:
        wait_mem()
        if c['id'] == 'lafan_dance':
            data, seq, cid = prep_lafan(c)
            up = P.run_umr(C.ROBOT, 'smplx', data, seq, timeout=1800, force=force, cid=cid)
        else:
            if c['id'] == 'hiphi_kneel':
                data, seq = prep_hiphi(c)
            else:
                data, seq = c['data'], c['seq']
            a, b = c['window_src_frames']; s = c['stride']
            cid = f"{c['id']}__{seq}__f{a}-{b}s{s}"
            up = P.run_umr(C.ROBOT, 'hoi', data, seq, timeout=1800, force=force, shared_corr=c.get('shared_corr', False),
                           start=a, end=b, stride=s, cid=cid)
        print(f'[r1] UMR done: {up.get("umr_seconds", "cached")} s', flush=True)
        qa = P.postprocess(C.ROBOT, c['kind'], data, seq, up, inv[C.ROBOT])
        row.update(status='ok' if qa['frames_flagged_final'] == 0 else 'ok_with_flagged_frames', clip=qa['clip'],
                   data=data, seq=seq, npz=qa['npz'], frames=qa['frames'], flagged=qa['frames_flagged_final'],
                   flags_hist=qa['flags_final_hist'], pen_after=qa['penetration_after'],
                   umr_paths=up, seconds=round(time.time() - t0, 1))
        print(json.dumps({k: qa[k] for k in ('clip', 'frames', 'penetration_after', 'frames_flagged_final',
                                             'flags_final_hist', 'seconds')}, default=str), flush=True)
    except Exception as e:
        row.update(status='error', error=repr(e), tb=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))
        traceback.print_exc()
    with open(P.OUT / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    print('[r1] status', row['status'], flush=True)


if __name__ == '__main__':
    main()
