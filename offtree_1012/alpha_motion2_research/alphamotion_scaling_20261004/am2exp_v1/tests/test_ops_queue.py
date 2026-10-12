"""WP-OPS queue tests: ordered jobs, approval gate, auto-gated twins, lifetimes, inherited lease, resume.

Uses a private lock file (never the real ROOT/gpu_queue.lock) and tiny python child jobs.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from am2exp_v1.ops import gates  # noqa: E402
from am2exp_v1.ops import lease as L  # noqa: E402
from am2exp_v1.ops.queue import Queue  # noqa: E402

CHILD_OK = ("import json, os, sys; sys.path.insert(0, {root!r}); "
            "from am2exp_v1.ops.lease import GpuLease, precheck; "
            "l = GpuLease.inherit(int(os.environ['AM2_GPU_LEASE_FD']), path={lock!r}, expected_inode=None); "
            "open({mark!r}, 'a').write(os.environ['AM2_JOB'] + ' ' + str(precheck({lock!r}, None)['held']) + "
            "' ' + str('PYTHONPATH' in os.environ) + '\\n')")


@pytest.fixture()
def env(tmp_path):
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    appr = tmp_path / 'approvals'
    appr.mkdir()
    dec = tmp_path / 'decisions'
    dec.mkdir()
    return {'tmp': tmp_path, 'lock': lock, 'appr': appr, 'dec': dec, 'mark': tmp_path / 'marks.txt'}


def make_queue(env, jobs, name='t', **kw):
    plan = {'schema': 'am2exp_queue_plan_v1', 'name': name, 'max_queue_seconds': 600, 'jobs': jobs}
    p = env['tmp'] / f'{name}.json'
    p.write_text(json.dumps(plan))
    guard_calls = []

    def guard():
        guard_calls.append(1)
        return {'passed': True}
    q = Queue(p, state_dir=env['tmp'] / 'state' / name, lease_path=env['lock'], expected_inode=None,
              approvals_dir=env['appr'], decisions_dir=env['dec'], python=sys.executable, guard_fn=guard,
              kill_grace=2, **kw)
    q.guard_calls = guard_calls
    return q


def child(env, extra=''):
    return ['{PY}', '-c', CHILD_OK.format(root=str(ROOT), lock=str(env['lock']), mark=str(env['mark'])) + extra]


def jobs_state(q):
    return json.loads(q.jobs_path.read_text())


def test_approval_gate_order_and_resume(env):
    jobs = [{'id': 'a', 'argv': child(env), 'stage': 's1', 'lifetime_seconds': 60},
            {'id': 'b', 'argv': child(env), 'stage': 's2', 'lifetime_seconds': 60}]
    q = make_queue(env, jobs)
    assert q.run() == 'stopped_awaiting_approval:s1'
    assert not env['mark'].exists() and jobs_state(q)['a']['status'] == 'blocked_approval'
    (env['appr'] / 's1.ok').write_text('lloyd ok')
    assert q.run() == 'stopped_awaiting_approval:s2'
    (env['appr'] / 's2.ok').write_text('lloyd ok')
    assert q.run() == 'all_jobs_done'
    lines = env['mark'].read_text().split()
    # each job ran exactly once, in order, with the lease held (inherited) and without PYTHONPATH
    assert lines == ['a', 'True', 'False', 'b', 'True', 'False']
    st = jobs_state(q)
    assert st['a']['status'] == st['b']['status'] == 'completed' and st['a']['approval']['sha256']
    assert st['a']['child_identity']['start_ticks']
    assert q.run() == 'all_jobs_done' and env['mark'].read_text().split() == lines  # resume: nothing re-ran
    assert not L.precheck(env['lock'], None)['held']  # lease released after the queue
    ex = json.loads((q.state_dir / 'exit_status.json').read_text())
    assert ex['status'] == 'completed' and len(q.guard_calls) == 2


def test_lease_held_elsewhere_blocks_queue(env):
    (env['appr'] / 's.ok').write_text('ok')
    q = make_queue(env, [{'id': 'a', 'argv': child(env), 'stage': 's'}])
    with L.GpuLease(env['lock'], expected_inode=None):
        with pytest.raises(BlockingIOError):
            q.run()
    assert not env['mark'].exists()


def test_second_queue_owner_refused(env):
    (env['appr'] / 's.ok').write_text('ok')
    q = make_queue(env, [{'id': 'a', 'argv': child(env), 'stage': 's'}])
    import fcntl
    q.state_dir.mkdir(parents=True)
    with open(q.state_dir / 'launcher.lock', 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError, match='another queue owner'):
            q.run()


def test_auto_gate_twins(env):
    (env['appr'] / 'twins.ok').write_text('ok')
    gate = lambda t, p: {'rule': 'am2exp_v1.ops.gates:decision_file', 'params': {'test': t, 'parent': p}}  # noqa: E731
    jobs = [{'id': 'S_wrong', 'argv': child(env), 'stage': 'twins', 'auto_gate': gate('S', 'SF')},
            {'id': 'O_roll', 'argv': child(env), 'stage': 'twins', 'auto_gate': gate('O', 'S')},
            {'id': 'I_roll', 'argv': child(env), 'stage': 'twins', 'auto_gate': gate('I', 'S')}]
    q = make_queue(env, jobs)
    assert q.run() == 'stopped_gate_undecided:S_wrong'  # no frozen decision -> nothing runs
    assert not env['mark'].exists()
    (env['dec'] / 'S_vs_SF.json').write_text(json.dumps({'test': 'S', 'parent': 'SF', 'beats_parent': True}))
    (env['dec'] / 'O_vs_S.json').write_text(json.dumps({'test': 'O', 'parent': 'S', 'beats_parent': False}))
    (env['dec'] / 'I_vs_S.json').write_text(json.dumps({'test': 'I', 'parent': 'S', 'beats_parent': True,
                                                         'lloyd_agrees': False}))
    assert q.run() == 'all_jobs_done'
    st = jobs_state(q)
    assert st['S_wrong']['status'] == 'completed'
    assert st['O_roll']['status'] == st['I_roll']['status'] == 'skipped_gate'
    assert 'lloyd overruled' in st['I_roll']['auto_gate_decision']['reason']
    assert env['mark'].read_text().split()[0] == 'S_wrong' and len(env['mark'].read_text().splitlines()) == 1


def test_decision_file_validation(env):
    (env['dec'] / 'X_vs_Y.json').write_text(json.dumps({'test': 'X', 'parent': 'Y', 'beats_parent': 'yes'}))
    d, _ = gates.decision_file({}, {'test': 'X', 'parent': 'Y'}, {'decisions_dir': env['dec']})
    assert d == 'blocked'
    (env['dec'] / 'X_vs_Y.json').write_text(json.dumps({'test': 'Z', 'parent': 'Y', 'beats_parent': True}))
    d, _ = gates.decision_file({}, {'test': 'X', 'parent': 'Y'}, {'decisions_dir': env['dec']})
    assert d == 'blocked'


def test_job_lifetime_kills_and_stops_queue(env):
    (env['appr'] / 's.ok').write_text('ok')
    jobs = [{'id': 'slow', 'argv': ['{PY}', '-c', 'import time; time.sleep(120)'], 'stage': 's', 'lifetime_seconds': 2},
            {'id': 'after', 'argv': child(env), 'stage': 's'}]
    q = make_queue(env, jobs)
    assert q.run() == 'stopped_job_timeout:slow'
    st = jobs_state(q)
    assert st['slow']['status'] == 'timeout' and st['slow']['returncode'] in (-15, -9)
    assert st['slow']['wall_seconds'] < 30
    assert not env['mark'].exists()  # queue stops after a failed job; no implicit restart
    assert 'after' not in st or st['after'].get('status') is None


def test_failed_job_stops_queue(env):
    (env['appr'] / 's.ok').write_text('ok')
    jobs = [{'id': 'bad', 'argv': ['{PY}', '-c', 'raise SystemExit(3)'], 'stage': 's'},
            {'id': 'after', 'argv': child(env), 'stage': 's'}]
    q = make_queue(env, jobs)
    assert q.run() == 'stopped_job_failed:bad'
    assert jobs_state(q)['bad']['returncode'] == 3 and not env['mark'].exists()


def test_capacity_failure_blocks(env, tmp_path):
    from am2exp_v1.ops import guard as G
    (env['appr'] / 's.ok').write_text('ok')
    q = make_queue(env, [{'id': 'a', 'argv': child(env), 'stage': 's'}])

    def bad():
        raise G.CapacityError('/home free too small')
    q.guard_fn = bad
    assert q.run() == 'stopped_capacity'
    assert not env['mark'].exists()


def test_arm_job_description_from_registry(env):
    """An arm job takes stage / auto_gate / lifetime / output_dir from train/arms.json."""
    q = make_queue(env, [{'id': 'S_wrong', 'arm': 'S_wrong'}, {'id': 'SF', 'arm': 'SF'}])
    d = q.describe_job({'id': 'S_wrong', 'arm': 'S_wrong'})
    assert d['stage'] == 'twins_e1' and d['auto_gate']['params'] == {'test': 'S', 'parent': 'SF'}
    assert d['output_dir'] == '/var/tmp/alphamotion/am2exp/runs/S_wrong'
    cmd = q.command(q.describe_job({'id': 'SF', 'arm': 'SF'}), 9)
    assert cmd[cmd.index('--arm') + 1] == 'SF' and cmd[cmd.index('--lease-fd') + 1] == '9'
    assert float(cmd[cmd.index('--lifetime-seconds') + 1]) < d['lifetime_seconds'] + 1e-9


def test_core_plan_describes(env):
    """The shipped plan is valid: every job resolves to a registry arm with a stage and lifetime."""
    from am2exp_v1.common.paths import CODE
    q = Queue(CODE / 'ops' / 'plans' / 'core_v1.json', state_dir=env['tmp'] / 'state' / 'core', lease_path=env['lock'],
              expected_inode=None, approvals_dir=env['appr'], python=sys.executable, guard_fn=lambda: {'passed': True})
    descs = [q.describe_job(j) for j in q.plan['jobs']]
    assert [d['id'] for d in descs][:5] == ['Arep', 'V1_old', 'V1_new', 'SF_M', 'SF']
    assert all(d['stage'] and d['lifetime_seconds'] > 0 for d in descs)
    assert q.run() == 'stopped_awaiting_approval:pipeline'  # nothing runs without lloyd's approval file
