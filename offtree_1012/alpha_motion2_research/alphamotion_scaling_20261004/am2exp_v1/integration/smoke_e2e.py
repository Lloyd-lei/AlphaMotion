"""WP-INTEGRATION end-to-end CPU smoke (no training, no GPU).

  cd ROOT; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
      $PY -m am2exp_v1.integration.smoke_e2e all            # or one stage: dryrun | untrained | e1 | render | index

Stages (outputs under OUT/integration/):
  dryrun     train_arm.py --dry-run-cpu --max-steps 5 for SF, S, O, I, S_wrong with the REAL dense assets
             (OUT/assets/surface_v2); then an independent re-check of every step's sampling_sha256 against
             ROOT/runs/A_S_seed0/train.jsonl (on top of the trainer's own in-loop assert).
  untrained  writes an untrained (step 0, seed-0 init exactly as train_arm builds it) S checkpoint in the format
             train_arm writes, so eval/run_e1_eval.py can load it through the am2exp registry path.
  e1         run_e1_eval (sensor_on) on 2 primary cases (human -> G1 and GR3, fallAndGetUp1_s4@4136) for A0@3000 and
             the untrained S; score both with judge J1 (eval/score_e1.py).
  render     one [human | A0 | S-untrained] triptych + 12-frame grid from those npz (render/triptych_e1.py).
  index      writes OUT/stage0_index.md (paths of every Stage-0 artifact that exists).
Every child runs with a timeout; nothing here takes the GPU lease.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from am2exp_v1.common.paths import ROOT, CODE, OUT, PY  # noqa: E402

IOUT = OUT / 'integration'
ARMS = ['SF', 'S', 'O', 'I', 'S_wrong']
REF = ROOT / 'runs' / 'A_S_seed0' / 'train.jsonl'
A0_CKPT = ROOT / 'runs' / 'A_S_seed0' / 'checkpoint_step3000.pt'
S0_CKPT = IOUT / 'ckpt' / 'S_untrained' / 'checkpoint_step0000.pt'
E1_CASES = ['human_smpl__unitree_g1_29dof__fallAndGetUp1_subject4@4136', 'human_smpl__fourier_gr3__fallAndGetUp1_subject4@4136']


def child_env():
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}  # ROS humble PYTHONPATH breaks children
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT),
               MUJOCO_GL='egl', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return env


def run(cmd, log, timeout, cwd=ROOT, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, 'w') as f:
        f.write('$ ' + ' '.join(map(str, cmd)) + '\n'); f.flush()
        try:
            p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env or child_env(), stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    return {'cmd': [str(c) for c in cmd], 'log': str(log), 'returncode': rc, 'seconds': round(time.time() - t0, 1)}


def reference_hashes(path=REF):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if 'sampling_sha256' in r:
                out[int(r['step'])] = (r.get('domain'), r['sampling_sha256'])
    return out


def check_hashes(run_dir, ref):
    rows = [json.loads(l) for l in (Path(run_dir) / 'train_slim.jsonl').read_text().splitlines() if l.strip()]
    per = []
    for r in rows:
        rd, rh = ref.get(int(r['step']), (None, None))
        verifiable = r['domain'] != 'xwbc'  # XWBC hashes include device-side torch.rand (CUDA reference)
        per.append({'step': r['step'], 'domain': r['domain'], 'domain_match': r['domain'] == rd,
                    'hash_match': (r['sampling_sha256'] == rh) if verifiable else None, 'total': r.get('total'),
                    'reference_total': r.get('reference_total')})
    ok = bool(per) and all(p['domain_match'] and p['hash_match'] is not False for p in per)
    return {'steps': len(per), 'all_ok': ok, 'per_step': per}


def stage_dryrun(args):
    ref = reference_hashes()
    res = {}
    for arm in args.arms.split(','):
        d = IOUT / 'dryrun' / arm
        if (d / 'result.json').exists() and json.loads((d / 'result.json').read_text()).get('status') == 'completed':
            r = {'returncode': 0, 'reused': True}
        else:
            if d.exists():
                import shutil
                shutil.rmtree(d)  # own folder only (OUT/integration/dryrun/<arm>)
            r = run([PY, CODE / 'train' / 'train_arm.py', '--arm', arm, '--dry-run-cpu', '--max-steps', args.steps,
                     '--out-dir', d, '--lifetime-seconds', 1800, '--threads', 4, '--rss-limit-gib', 19.5],
                    IOUT / 'logs' / f'dryrun_{arm}.log', timeout=2400)
        if (d / 'result.json').exists():
            rj = json.loads((d / 'result.json').read_text())
            r.update(status=rj.get('status'), sampling=rj.get('sampling'), loss_vs_reference=rj.get('loss_vs_reference'),
                     rss_peak_gib=(rj.get('rss') or {}).get('rss_peak_gib'),
                     seconds_per_step_mean=rj.get('seconds_per_step_mean'))
        if (d / 'train_slim.jsonl').exists():
            r['independent_hash_check'] = check_hashes(d, ref)
        if (d / 'exit_status.json').exists():
            r['exit_status'] = json.loads((d / 'exit_status.json').read_text())
        res[arm] = r
        print(arm, json.dumps({k: r.get(k) for k in ('returncode', 'status', 'rss_peak_gib')}), flush=True)
    (IOUT / 'dryrun_summary.json').write_text(json.dumps(res, indent=1, default=str))
    return res


def build_untrained(arm_name='S', out=S0_CKPT):
    """Step-0 checkpoint built exactly like train_arm._train builds the model (seeding before construction)."""
    import numpy as np
    import torch
    from am2exp_v1.train import train_arm as TA
    arm = TA.resolve_arm(arm_name)
    cfg = TA.build_config(arm)
    ex = arm['extra']
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    cls = TA.load_symbol(arm['model_class'])
    model = cls(**cfg['model'], **(ex.get('model_kwargs') or {}))
    model.set_judge_body_order(cfg['judge_body_order'])
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    obj = {'model': model.state_dict(), 'step': 0, 'config': cfg, 'data_manifest_sha256': cfg['data_manifest_sha256'],
           'optimizer_included': False, 'final': False, 'note': 'am2exp integration: untrained seed-0 init (pipeline test only)'}
    torch.save(obj, out)
    (out.parent / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    return {'path': str(out), 'class': arm['model_class'], 'params': sum(p.numel() for p in model.parameters())}


def stage_untrained(args):
    r = build_untrained()
    print(json.dumps(r), flush=True)
    return r


def stage_e1(args):
    res = {}
    targets = 'unitree_g1_29dof,fourier_gr3'
    for name, ck in (('A0', A0_CKPT), ('S_untrained', S0_CKPT)):
        d = IOUT / 'e1' / name / 'sensor_on'
        r = run([PY, '-m', 'am2exp_v1.eval.run_e1_eval', '--arm', name, '--checkpoint', ck, '--mode', 'sensor_on',
                 '--out', d, '--sources', 'human_smpl', '--targets', targets, '--only-primary', '--limit', 2, '--threads', 4],
                IOUT / 'logs' / f'e1_{name}.log', timeout=1800)
        r['npz'] = sorted(str(p) for p in d.glob('*.npz'))
        s = run([PY, '-m', 'am2exp_v1.eval.score_e1', '--dir', d, '--jobs', 2], IOUT / 'logs' / f'score_{name}.log', timeout=1800)
        r['score'] = s
        sc = d / 'scores_j1.jsonl'
        if sc.exists():
            r['scores'] = [json.loads(l) for l in sc.read_text().splitlines() if l.strip()]
        res[name] = r
    (IOUT / 'e1_summary.json').write_text(json.dumps(res, indent=1, default=str))
    return res


def stage_render(args):
    res = {}
    for case in E1_CASES:
        ctrl = IOUT / 'e1' / 'A0' / 'sensor_on' / f'{case}.npz'
        test = IOUT / 'e1' / 'S_untrained' / 'sensor_on' / f'{case}.npz'
        if not (ctrl.exists() and test.exists()):
            res[case] = {'skipped': 'npz missing'}; continue
        out = IOUT / 'videos' / case.replace('@', '_')
        r = run([PY, '-m', 'render.triptych_e1', '--control', ctrl, '--test', test, '--out', out,
                 '--labels', 'A0 (step 3000),S (untrained)', '--overwrite', '--max-seconds', 1500],
                IOUT / 'logs' / f'render_{case.replace("@", "_")}.log', timeout=1800, cwd=CODE)
        r['files'] = sorted(str(p) for p in out.glob('*')) if out.exists() else []
        res[case] = r
        if args.one_render:
            break
    (IOUT / 'render_summary.json').write_text(json.dumps(res, indent=1, default=str))
    return res


def _ls(p, pattern='*'):
    p = Path(p)
    return sorted(p.glob(pattern)) if p.exists() else []


def stage_index(args):
    from am2exp_v1.integration.index import write_index
    path = write_index()
    print(path)
    return {'index': str(path)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['dryrun', 'untrained', 'e1', 'render', 'index', 'all'])
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--one-render', action='store_true', default=True)
    args = ap.parse_args(argv)
    IOUT.mkdir(parents=True, exist_ok=True)
    stages = ['dryrun', 'untrained', 'e1', 'render', 'index'] if args.stage == 'all' else [args.stage]
    for st in stages:
        globals()['stage_' + st](args)


if __name__ == '__main__':
    main()
