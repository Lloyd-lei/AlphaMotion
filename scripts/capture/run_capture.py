"""Build UMR configs for (robot, clip) and run UMR through the capture wrapper.

python run_capture.py --robot unitree_g1_29dof --kind smplx --data <UMR>/sample_data/lafan1_smplx --seq dance1_subject2
python run_capture.py --robot unitree_g1_29dof --kind hoi --data <UMR>/sample_data/omnicontact/soccer/case3_kick_right \
       --seq 20260330000586_1_1775269234
Stock UMR defaults are used (no solver overrides) unless --overrides json is given.  Robot model/config = the factory's
staged MJCF + umr_robot.json (copied to OUT/robots/<robot>).  Correspondence caches go to OUT/corr (never the factory's).
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

UMR_ROOT = Path(f'{SRCFG.umr_root}')
UMR_PY = f'{SRCFG.umr_python}'
OUT = Path(f'{SRCFG.work}')
HERE = Path(__file__).resolve().parent
DEFAULTS = {'smplx': UMR_ROOT / 'humanoid_retarget_defaults.json',
            'hoi': UMR_ROOT / 'humanoid_retarget_defaults_hsi_hoi_standard.json'}
PIPE = {'smplx': 'humanoid_retarget_pipeline', 'hoi': 'humanoid_retarget_pipeline_hsi_hoi'}


def _abs(obj, base):
    if isinstance(obj, dict):
        return {k: _abs(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_abs(v, base) for v in obj]
    if isinstance(obj, str) and obj and not obj.startswith('/') and ('/' in obj or obj == 'smpl'):
        p = base / obj
        if p.exists():
            return str(p.resolve())
    return obj


def clip_id(kind, data, seq):
    return f"{Path(str(data)).stem if kind == 'smplx' else Path(str(data)).name}__{seq}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', required=True); ap.add_argument('--kind', choices=['smplx', 'hoi'], required=True)
    ap.add_argument('--data', required=True); ap.add_argument('--seq', required=True)
    ap.add_argument('--mode', default='cpu', choices=['cpu', 'skip'])
    ap.add_argument('--overrides', default='{}'); ap.add_argument('--tag', default='stock')
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--stride', type=int, default=1, help='temporal stride (e.g. 3: 90 fps source -> 30 fps)')
    ap.add_argument('--corr-tag', default='', help='correspondence cache tag (default: smplx / per-HOI-sequence hash)')
    a = ap.parse_args()
    cid = clip_id(a.kind, a.data, a.seq) + (f'_s{a.stride}' if a.stride > 1 else '')
    rdir = OUT / 'runs' / cid / a.tag; rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / f'{a.robot}.npz'
    cap = OUT / 'problems' / cid / f'{a.robot}.{a.tag}.{a.mode}.npz'; cap.parent.mkdir(parents=True, exist_ok=True)
    dflt = _abs(json.load(open(DEFAULTS[a.kind])), UMR_ROOT)
    dflt.setdefault('solver', {}).update(json.loads(a.overrides))
    dflt.setdefault('view', {})['enabled'] = False
    if a.kind == 'smplx':
        dflt['motion'].update({'data': str(a.data), 'seq_key': a.seq})
        dflt['retarget']['out'] = str(out)
    dpath = rdir / f'{a.robot}.defaults.json'; json.dump(dflt, open(dpath, 'w'), indent=1)
    rr = OUT / 'robots' / a.robot
    rcfg = json.load(open(rr / 'umr_robot.json'))
    rcfg['robot']['xml'] = str(rr / f'{a.robot}.xml')
    tag = hashlib.sha1(f'{a.kind}:{a.data}:{a.seq}'.encode()).hexdigest()[:10] if a.kind == 'hoi' else 'smplx'
    tag = a.corr_tag or tag
    cdir = OUT / 'corr' / a.robot / tag; cdir.mkdir(parents=True, exist_ok=True)
    rcfg['correspondence'].setdefault('dataset', {})['out'] = str(cdir / 'correspondence_dataset.npz')
    rcfg['correspondence'].setdefault('train', {})['out_dir'] = str(cdir / 'train')
    cpath = rdir / f'{a.robot}.robot.json'; json.dump(rcfg, open(cpath, 'w'), indent=1)
    cmd = ['timeout', str(a.timeout), UMR_PY, str(HERE / 'umr_capture_pipe.py'), PIPE[a.kind], '--config', str(cpath),
           '--defaults', str(dpath), '--skip-view', '--force-retarget']
    if a.kind == 'hoi':
        cmd += ['--data', str(a.data), '--seq-key', a.seq, '--out', str(out)]
        if a.stride > 1:
            cmd += ['--stride', str(a.stride)]
    elif a.stride > 1:
        dflt['motion']['stride'] = a.stride
        json.dump(dflt, open(dpath, 'w'), indent=1)
    env = dict(os.environ, UMRG_CAPTURE=str(cap), UMRG_MODE=a.mode, MUJOCO_GL='egl')
    log = OUT / 'logs' / f'{cid}.{a.robot}.{a.tag}.{a.mode}.log'
    t0 = time.time()
    with open(log, 'w') as lf:
        p = subprocess.run(cmd, cwd=str(UMR_ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
    print(json.dumps(dict(rc=p.returncode, seconds=round(time.time() - t0, 1), out=str(out), capture=str(cap), log=str(log))))


if __name__ == '__main__':
    main()
