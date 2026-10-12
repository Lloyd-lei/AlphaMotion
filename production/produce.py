"""Resumable production runner (detached).  Reads _jobs/jobs.jsonl, runs every job without a done marker:
  * takes an exclusive flock on /home/arenalabs/umr_factory_data/logs/gpu_heavy.lock per job (released between
    jobs so the GPU BVH->SMPL-X converter can interleave);
  * duty-cycle throttle (UMRG_DUTY, default 0.6) + pause while throttle.log reports net=down;
  * per-clip npz written atomically to <out>/<source>/<clip_id>/<robot>.npz; at job end: _jobs/<job>.done (summary +
    per-clip QA), append manifest.jsonl (one row per clip x robot), append _jobs/progress.jsonl;
  * failures: _jobs/<job>.failed (traceback); retried up to 2 times on later passes;
  * stop: touch <out>/_STOP (checked between jobs).
"""
import argparse
import fcntl
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, '/home/arenalabs/umr_gpu')
OUT = Path('/home/arenalabs/umr_factory_data/production/v1')
LOCK = '/home/arenalabs/umr_factory_data/logs/gpu_heavy.lock'


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(OUT / '_logs' / 'runner.log', 'a') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default=str(OUT / '_jobs' / 'jobs.jsonl'))
    ap.add_argument('--max-jobs', type=int, default=0)
    ap.add_argument('--B', type=int, default=1024)
    ap.add_argument('--max-piece', type=int, default=2500)
    a = ap.parse_args()
    (OUT / '_logs').mkdir(parents=True, exist_ok=True)
    import torch
    from surface_retargeting.gpu import production as PR
    from surface_retargeting.gpu import throttle
    throttle.set_duty(float(os.environ.get('UMRG_DUTY', '0.6')))
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    log(f'runner start pid={os.getpid()} jobs={len(jobs)} engine={PR.engine_version()} duty={throttle.DUTY}')
    done_n = 0
    for job in jobs:
        jd = OUT / '_jobs'
        if (jd / f"{job['job_id']}.done").exists():
            continue
        fails = len(list(jd.glob(f"{job['job_id']}.failed*")))
        if fails >= 3:
            continue
        if (OUT / '_STOP').exists():
            log('stop file found -> exit'); break
        while throttle.net_down():
            log('network down -> waiting'); time.sleep(60)
        r = job['robot']
        asset_dir = OUT / '_assets' / r
        if not (asset_dir / 'surface.npz').exists():
            asset_dir.mkdir(parents=True, exist_ok=True)
            src = Path('/var/tmp/alphamotion/umr_factory/umr_gpu/robots') / r
            shutil.copy(src / 'surface.npz', asset_dir / 'surface.npz'); shutil.copy(src / f'{r}.xml', asset_dir / f'{r}.xml')
        with open(LOCK, 'a') as lf:
            log(f"waiting for GPU lock: {job['job_id']}")
            fcntl.flock(lf, fcntl.LOCK_EX)
            t0 = time.time()
            log(f"start {job['job_id']} clips={len(job['clips'])}")
            try:
                summary, rows, qa = PR.run_job(job, OUT, B_max=a.B, max_piece=a.max_piece)
                ok = True
            except Exception:
                ok = False
                tb = traceback.format_exc()
            finally:
                torch.cuda.empty_cache()
                fcntl.flock(lf, fcntl.LOCK_UN)
        if not ok:
            (jd / f"{job['job_id']}.failed{fails}").write_text(tb)
            log(f"FAILED {job['job_id']}: {tb.splitlines()[-1]}")
            continue
        with open(OUT / 'manifest.jsonl', 'a') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        summary['wall_s'] = round(time.time() - t0, 1)
        (jd / f"{job['job_id']}.done").write_text(json.dumps(dict(summary=summary, qa=qa), indent=1))
        with open(jd / 'progress.jsonl', 'a') as f:
            f.write(json.dumps(dict(summary, finished=time.strftime('%Y-%m-%d %H:%M:%S'))) + '\n')
        log(f"done {job['job_id']} robot_h={summary['robot_hours']} kept={summary['kept_frac']} "
            f"rf/s={summary['robot_frames_per_s']} wall={summary['wall_s']}s")
        done_n += 1
        if a.max_jobs and done_n >= a.max_jobs:
            break
    log('runner exit')


if __name__ == '__main__':
    main()
