"""Single sequential GPU queue owner for am2exp.

    python queue.py --plan PLAN.json            # run / resume the plan
    python queue.py --plan PLAN.json --status   # print state, run nothing

One process owns ROOT/gpu_queue.lock for the whole queue (pre-checked in /proc/locks, never
recreated) plus a per-plan launcher lock (no duplicate owners). Jobs run one after another as
child processes that INHERIT the lease fd (subprocess pass_fds; the child re-asserts LOCK_EX on
the same open file description), so the lease never drops between jobs and there is no
probe-release-spawn race. Every job has a wall-clock lifetime: SIGTERM to its process group,
then SIGKILL after a grace period.

Plan JSON (ordered):
  {"schema": "am2exp_queue_plan_v1", "name": "e1_core", "max_queue_seconds": 43200,
   "out_allocation_bytes": 4000000000,
   "jobs": [
     {"id": "Arep", "arm": "Arep"},                                   # train/train_arm.py --arm Arep
     {"id": "S_wrong", "arm": "S_wrong"},                             # auto_gate/stage from the registry
     {"id": "eval_SF", "argv": ["{PY}", "-u", "{CODE}/eval/run_e1_eval.py", "..."],
      "stage": "eval_e1", "lifetime_seconds": 900}]}
Gates before a job starts (in this order; any failure stops the queue, nothing is skipped silently):
  1. approval: OUT/logs/approvals/<stage>.ok must exist (lloyd's OK for the GPU time);
  2. auto_gate (twins): rule(job, params, ctx) -> run | skip | blocked (ops/gates.py);
  3. capacity guard (/home downloader reserve + / 20 GiB);
A failed or timed-out job stops the queue (no implicit restart); rerunning the queue resumes at
the first job that is not completed/skipped. Arm jobs whose run folder already holds a completed
result.json are treated as completed.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if _sys.path and _sys.path[0] and _Path(_sys.path[0]).resolve() == _Path(__file__).resolve().parent:
    _sys.path.pop(0)  # running ops/<x>.py as a script: ops/queue.py must not shadow the stdlib 'queue'
import sys
sys.dont_write_bytecode = True
import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from am2exp_v1.common.paths import CODE, OUT, PY, APPROVALS, GPU_LOCK, GPU_LOCK_INODE, BASE_ENV  # noqa: E402
from am2exp_v1.ops import receipts as R  # noqa: E402
from am2exp_v1.ops import guard as G  # noqa: E402
from am2exp_v1.ops.lease import GpuLease  # noqa: E402

TRAIN_ARM = CODE / 'train' / 'train_arm.py'
KILL_GRACE_SECONDS = 60
CHILD_LIFETIME_MARGIN = 90  # the child's own graceful lifetime ends this much before the hard kill
TERMINAL_OK = ('completed', 'skipped_gate', 'already_completed')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_symbol(spec):
    mod, _, attr = spec.partition(':')
    obj = importlib.import_module(mod)
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj


class Queue:
    def __init__(self, plan_path, *, state_dir=None, lease_path=GPU_LOCK, expected_inode=GPU_LOCK_INODE,
                 approvals_dir=APPROVALS, decisions_dir=None, python=PY, registry=None, guard_fn=None,
                 kill_grace=KILL_GRACE_SECONDS, poll_seconds=1.0):
        self.plan_path = Path(plan_path)
        self.plan = json.loads(self.plan_path.read_text())
        if self.plan.get('schema') != 'am2exp_queue_plan_v1':
            raise ValueError('plan schema must be am2exp_queue_plan_v1')
        ids = [j['id'] for j in self.plan['jobs']]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate job ids')
        self.plan_sha = _sha(self.plan_path)
        self.name = self.plan['name']
        self.state_dir = Path(state_dir or OUT / 'logs' / 'queue' / self.name)
        self.lease_path = Path(lease_path)
        self.expected_inode = expected_inode
        self.approvals_dir = Path(approvals_dir)
        self.decisions_dir = decisions_dir
        self.python = python
        self.registry = registry
        self.guard_fn = guard_fn
        self.kill_grace = kill_grace
        self.poll_seconds = poll_seconds
        self.child = None
        self.jobs_path = self.state_dir / 'jobs.json'

    # ---------------------------------------------------------------- state
    def load_jobs_state(self):
        if self.jobs_path.exists():
            return json.loads(self.jobs_path.read_text())
        return {}

    def save_jobs_state(self, jobs):
        R.write_json(self.jobs_path, jobs)

    def arm_spec(self, arm):
        add = sys.modules.get('am2exp_v1.train.train_arm')
        if add is None:
            from am2exp_v1.train import train_arm as add  # light import: no torch at module level
        return add.resolve_arm(arm, add.load_registry(self.registry) if self.registry else None)

    def describe_job(self, job):
        out = dict(job)
        if 'arm' in job:
            spec = self.arm_spec(job['arm'])
            ex = spec['extra']
            out.setdefault('stage', ex.get('stage'))
            out.setdefault('auto_gate', ex.get('auto_gate'))
            out.setdefault('lifetime_seconds', ex.get('lifetime_seconds'))
            out['output_dir'] = spec['output_dir']
        out.setdefault('lifetime_seconds', self.plan.get('default_lifetime_seconds', 3600))
        if not out.get('stage'):
            raise ValueError(f'job {job["id"]} has no stage (approval gate key)')
        return out

    def arm_completed(self, desc):
        if 'arm' not in desc:
            return False
        res = Path(desc['output_dir']) / 'result.json'
        return res.exists() and json.loads(res.read_text()).get('status') == 'completed'

    # ---------------------------------------------------------------- gates
    def approval(self, stage):
        ok = self.approvals_dir / f'{stage}.ok'
        if not ok.exists():
            return None
        return {'path': str(ok), 'sha256': _sha(ok), 'mtime': ok.stat().st_mtime}

    def auto_gate(self, desc):
        gate = desc.get('auto_gate')
        if not gate:
            return 'run', 'no auto gate'
        fn = load_symbol(gate['rule'])
        return fn(desc, gate.get('params') or {}, {'decisions_dir': self.decisions_dir, 'queue': self.name})

    def guard(self):
        if self.guard_fn is not None:
            return self.guard_fn()
        return G.check(out_allocation_bytes=int(self.plan.get('out_allocation_bytes', 0)))

    # ---------------------------------------------------------------- child
    def command(self, desc, fd):
        life = float(desc['lifetime_seconds'])
        if 'arm' in desc:
            cmd = [self.python, '-u', str(TRAIN_ARM), '--arm', desc['arm'], '--lease-fd', str(fd),
                   '--lifetime-seconds', str(max(60.0, life - CHILD_LIFETIME_MARGIN)),
                   '--approval-stage', str(desc['stage'])]
            if self.registry:
                cmd += ['--registry', str(self.registry)]
            return cmd + [str(a) for a in desc.get('args', [])]
        subst = {'{PY}': self.python, '{CODE}': str(CODE), '{OUT}': str(OUT), '{LEASE_FD}': str(fd)}
        cmd = []
        for a in desc['argv']:
            a = str(a)
            for k, v in subst.items():
                a = a.replace(k, v)
            cmd.append(a)
        return cmd

    def run_child(self, desc, fd, log_path):
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}  # ROS humble paths must not leak in
        env.update(BASE_ENV, AM2_GPU_LEASE_FD=str(fd), AM2_QUEUE=self.name, AM2_JOB=desc['id'])
        env.update({k: str(v) for k, v in (desc.get('env') or {}).items()})
        cmd = self.command(desc, fd)
        life = float(desc['lifetime_seconds'])
        started = time.monotonic()
        with open(log_path, 'ab') as log:
            log.write(f'\n=== {R.utc()} job {desc["id"]} cmd {json.dumps(cmd)}\n'.encode())
            log.flush()
            self.child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(fd,),
                                          start_new_session=True, cwd=str(CODE))
            ident = None
            try:
                ident = R.identity(self.child.pid)
            except Exception:
                pass
            self.on_child_started(desc, cmd, ident)
            status = None
            try:
                rc = self.child.wait(timeout=life)
                status = 'completed' if rc == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status = 'timeout'
                rc = self.kill_child()
            finally:
                if self.child is not None and self.child.poll() is None:  # queue interrupted
                    rc = self.kill_child()
                    status = status or 'killed_with_queue'
                self.child = None
        return {'status': status, 'returncode': rc, 'cmd': cmd, 'identity': ident,
                'wall_seconds': time.monotonic() - started, 'lifetime_seconds': life, 'log': str(log_path)}

    def kill_child(self):
        child = self.child
        if child is None:
            return None
        for sig, wait in ((signal.SIGTERM, self.kill_grace), (signal.SIGKILL, 30)):
            if child.poll() is not None:
                break
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                break
            try:
                child.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        return child.poll()

    def on_child_started(self, desc, cmd, ident):  # hook for tests / state
        jobs = self.load_jobs_state()
        jobs.setdefault(desc['id'], {}).update(status='running', child_identity=ident, cmd=cmd, started_utc=R.utc())
        self.save_jobs_state(jobs)

    # ---------------------------------------------------------------- main loop
    def status(self):
        return {'plan': str(self.plan_path), 'plan_sha256': self.plan_sha, 'jobs': self.load_jobs_state()}

    def run(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lpath = self.state_dir / 'launcher.lock'
        launcher = open(lpath, 'a+')  # our own file (never the GPU lock)
        try:
            fcntl.flock(launcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            launcher.close()
            raise BlockingIOError(f'another queue owner holds {lpath}')
        deadline = time.monotonic() + float(self.plan.get('max_queue_seconds', 86400))
        outcome = 'failed'
        with R.ProcessReceipts(self.state_dir, 'queue', extra={'plan': str(self.plan_path), 'plan_sha256': self.plan_sha,
                                                                'queue': self.name},
                               pids_fn=lambda: [os.getpid()] + ([self.child.pid] if self.child else [])) as rec:
            lease = GpuLease(self.lease_path, self.expected_inode)
            try:
                lease.acquire()
                R.write_json(self.state_dir / 'lease.json', lease.receipt)
                jobs = self.load_jobs_state()
                outcome = 'all_jobs_done'
                for job in self.plan['jobs']:
                    jid = job['id']
                    desc = self.describe_job(job)
                    jr = jobs.setdefault(jid, {})
                    jr.update(job=desc, plan_sha256=self.plan_sha)
                    if jr.get('status') in TERMINAL_OK:
                        continue
                    if self.arm_completed(desc):
                        jr.update(status='already_completed', checked_utc=R.utc())
                        self.save_jobs_state(jobs)
                        continue
                    rec.update(phase='gates', job=jid)
                    if time.monotonic() > deadline:
                        jr.update(status='blocked_queue_lifetime', utc=R.utc())
                        outcome = 'stopped_queue_lifetime'
                        break
                    appr = self.approval(desc['stage'])
                    if appr is None:
                        jr.update(status='blocked_approval', stage=desc['stage'], utc=R.utc(),
                                  needs=str(self.approvals_dir / f'{desc["stage"]}.ok'))
                        outcome = f'stopped_awaiting_approval:{desc["stage"]}'
                        break
                    jr['approval'] = appr
                    decision, reason = self.auto_gate(desc)
                    jr['auto_gate_decision'] = {'decision': decision, 'reason': reason, 'utc': R.utc()}
                    if decision == 'skip':
                        jr.update(status='skipped_gate')
                        self.save_jobs_state(jobs)
                        continue
                    if decision != 'run':
                        jr.update(status='blocked_gate', utc=R.utc())
                        outcome = f'stopped_gate_undecided:{jid}'
                        break
                    try:
                        jr['guard'] = self.guard()
                    except G.CapacityError as exc:
                        jr.update(status='blocked_capacity', error=str(exc), utc=R.utc())
                        outcome = 'stopped_capacity'
                        break
                    jr['free_before'] = R.free_snapshot()
                    self.save_jobs_state(jobs)
                    rec.update(phase='running', job=jid)
                    result = self.run_child(desc, lease.handoff_fd(), self.state_dir / f'{jid}.log')
                    jobs = self.load_jobs_state()
                    jr = jobs[jid]
                    jr.update(result, finished_utc=R.utc(), free_after=R.free_snapshot())
                    self.save_jobs_state(jobs)
                    if result['status'] != 'completed':
                        outcome = f'stopped_job_{result["status"]}:{jid}'
                        break
                self.save_jobs_state(jobs)
                rec.update(phase='done', outcome=outcome)
            finally:
                if self.child is not None:
                    self.kill_child()
                lease.release()
                launcher.close()
        R.write_json(self.state_dir / 'outcome.json', {'utc': R.utc(), 'outcome': outcome, 'plan_sha256': self.plan_sha})
        return outcome


def main():
    ap = argparse.ArgumentParser(description='am2exp single-owner GPU queue')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--registry', default=None)
    args = ap.parse_args()
    q = Queue(args.plan, registry=args.registry)
    if args.status:
        print(json.dumps(q.status(), indent=2))
        return
    outcome = q.run()
    print(json.dumps({'queue': q.name, 'outcome': outcome}), flush=True)
    sys.exit(0 if outcome == 'all_jobs_done' else 2)


if __name__ == '__main__':
    main()
