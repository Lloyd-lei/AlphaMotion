"""Step 3a -- run the fixed 4-clip QA set through the CPU reference pipeline for unitree_g1_23dof only.

Same engine calls as umr_factory_code/scripts/run_clip.py (umrf.pipeline.run_umr -> umrf.pipeline.postprocess), plus
what run_clip.py lacks: a frame window (start/end/stride), CPU-only SMPL-X (CUDA hidden from the UMR subprocess, GPU is
reserved), and the beta-zero shared correspondence for HOI / HiPHI clips (no per-clip GPU training).  The shared
modules are imported, never edited.  Usage: python run_qa.py <dance|soccer|box|getup> [--force]
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''            # inherited by the UMR subprocess: CPU only
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P, robots as R
from umrf.paths import UMR_ROOT, UMR_PY, UMR_TOOLS, CORR, ROBOTS, RUNS, OUT

ROBOT = 'unitree_g1_23dof'
PDIR = ROBOTS / ROBOT / 'pipeline'
UMRS = UMR_ROOT / 'sample_data'
CLIPS = {
    'dance': dict(kind='smplx', data=str(UMRS / 'lafan1_smplx'), seq='dance1_subject2', start=1900, end=2500, stride=1,
                  note='LAFAN1 dance1_subject2 (30 fps), frames 1900-2500'),
    'soccer': dict(kind='hoi', data=str(UMRS / 'omnicontact/soccer/case3_kick_right'), seq='20260330000586_1_1775269234',
                   start=600, end=1800, stride=3, shared=True,
                   note='OmniContact soccer case3_kick_right (90 fps -> 30 fps), approach + first kicks'),
    'box': dict(kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/omnicontact/box/case1_carry',
                seq='20260330000166_1_1775007076', start=450, end=1950, stride=3, shared=True,
                note='OmniContact box case1_carry (90 fps -> 30 fps): lift, carry, put down, lift again'),
    'getup': dict(kind='hoi', post_kind='smplx', data=str(UMRS / 'hiphi/data/Getting_up/get_up'), seq='Getting_up-get_up_0004',
                  start=1000, end=2500, stride=3, shared=True,
                  note='HiPHI Getting_up-get_up_0004 (90 fps -> 30 fps, no object): go down, lie prone, push up '
                       'with hands/knees, stand, go down again. HiPHI clip with hand/knee/torso/hip ground contact '
                       '(none of the converted /home/.../smplx/hiphi clips is a pure ground-contact clip; the kneel '
                       'clips there also need CoACD chair assets written into the shared HiPHI tree)'),
}


def cid_of(name, c):
    return f"qa_v2__{name}__{Path(c['data']).name}__{c['seq']}__f{c['start']}-{c['end']}s{c['stride']}"


def run_umr_smplx_window(robot, data, seq, start, end, stride, cid, timeout=1800, force=False):
    """umrf.pipeline.run_umr for kind='smplx' + motion window + CPU SMPL-X (copy of its smplx branch)."""
    rdir = RUNS / cid; rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / f'{robot}.npz'; cap = rdir / f'{robot}.src_slots.npz'; stats = rdir / f'{robot}.qpstats.json'
    log = rdir / f'{robot}.umr.log'
    res = dict(umr_out=str(out), src_slots=str(cap), qp_stats=str(stats), log=str(log), clip=cid, shared_corr=False)
    if out.exists() and cap.exists() and not force:
        res['cached'] = True
        return res
    dflt = P._absolutize(json.load(open(P.DEFAULTS['smplx'])), UMR_ROOT)
    dflt.setdefault('solver', {}).update(P.SOLVER_OVERRIDES)
    dflt.setdefault('view', {})['enabled'] = False
    dflt['motion'].update({'data': str(data), 'seq_key': seq, 'start': int(start), 'end': int(end), 'stride': int(stride)})
    dflt['retarget']['out'] = str(out)
    dflt['retarget']['smplx_device'] = 'cpu'
    dpath = rdir / f'{robot}.defaults.json'; json.dump(dflt, open(dpath, 'w'), indent=1)
    rcfg = json.load(open(ROBOTS / robot / 'umr_robot.json'))
    cdir = CORR / robot / 'smplx'
    rcfg['correspondence'].setdefault('dataset', {})['out'] = str(cdir / 'correspondence_dataset.npz')
    rcfg['correspondence'].setdefault('train', {})['out_dir'] = str(cdir / 'train')
    cpath = rdir / f'{robot}.robot.json'; json.dump(rcfg, open(cpath, 'w'), indent=1)
    cmd = ['timeout', str(timeout), UMR_PY, str(UMR_TOOLS / 'umr_pipe.py'), P.PIPE_MODULE['smplx'], '--config', str(cpath),
           '--defaults', str(dpath), '--skip-view', '--stage', 'all', '--force-retarget']
    env = dict(os.environ, UMR_CAPTURE=str(cap), UMR_QP_FALLBACK='1', UMR_STATS=str(stats), MUJOCO_GL='egl')
    t0 = time.time()
    with open(log, 'w') as lf:
        p = subprocess.run(cmd, cwd=str(UMR_ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
    res['umr_seconds'] = round(time.time() - t0, 1); res['returncode'] = p.returncode
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f'UMR failed rc={p.returncode}, see {log}')
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('clip', choices=list(CLIPS)); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    c = CLIPS[a.clip]; cid = cid_of(a.clip, c)
    inv = {r['name']: r for r in R.load_inventory()}
    t0 = time.time()
    row = dict(robot=ROBOT, qa_clip=a.clip, cid=cid, **{k: c[k] for k in ('kind', 'data', 'seq', 'start', 'end', 'stride')},
               started=time.strftime('%F %T'), runner=str(Path(__file__).resolve()))
    try:
        if c['kind'] == 'smplx':
            up = run_umr_smplx_window(ROBOT, c['data'], c['seq'], c['start'], c['end'], c['stride'], cid, force=a.force)
        else:
            up = P.run_umr(ROBOT, 'hoi', c['data'], c['seq'], timeout=1800, force=a.force, shared_corr=c.get('shared', False),
                           start=c['start'], end=c['end'], stride=c['stride'], cid=cid)
        print(f'[run_qa] UMR done ({up.get("umr_seconds", "cached")} s)', flush=True)
        qa = P.postprocess(ROBOT, c.get('post_kind', c['kind']), c['data'], c['seq'], up, inv[ROBOT], log_every=100)
        row.update(status='ok' if qa['frames_flagged_final'] == 0 else 'ok_with_flagged_frames', npz=qa['npz'],
                   frames=qa['frames'], flagged=qa['frames_flagged_final'], pen_after=qa['penetration_after'],
                   umr_seconds=up.get('umr_seconds'), seconds=round(time.time() - t0, 1))
        print(json.dumps({k: qa[k] for k in ('clip', 'robot', 'frames', 'penetration_after', 'frames_flagged_final',
                                             'flags_final_hist', 'seconds')}, default=str), flush=True)
    except Exception as e:
        import traceback
        row.update(status='error', error=repr(e), tb=traceback.format_exc()[-2000:], seconds=round(time.time() - t0, 1))
        print('[run_qa] ERROR', repr(e), flush=True); traceback.print_exc()
    with open(PDIR / 'qa_runs.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    with open(OUT / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps(dict(row, clip=cid), default=str) + '\n')


if __name__ == '__main__':
    main()
