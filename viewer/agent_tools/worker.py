"""Detached single-job worker; never reuses or signals an unrelated PID."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from .common import file_lock, identity, now, read_json, write_json
from .jobs import Jobs


def main():
    config, job_id = sys.argv[1:]
    settings = read_json(config); jobs = Jobs(settings, config)
    folder = jobs.root / 'jobs' / job_id
    plan = read_json(folder / 'plan.json'); recipe = plan['recipe']
    state = {'job_id': job_id, 'status': 'waiting_for_slot', 'worker_identity': identity(os.getpid()),
             'started_unix': now(), 'workflow': plan['workflow'], 'evidence_boundary': plan['evidence_boundary']}
    write_json(folder / 'state.json', state)
    try:
        # Busy means failure, not an unbounded hidden queue. The caller can inspect
        # the owner and intentionally create a fresh plan later.
        with file_lock(jobs.root / 'execution.lock', blocking=False):
            with file_lock(recipe.get('lock_path', str(jobs.root / 'cpu.lock')), blocking=False):
                state['capacity'] = jobs.validate(plan)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                env.update(recipe.get('env', {}))
                if recipe.get('device', 'cpu') == 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                child = subprocess.Popen(plan['argv'], cwd=plan['cwd'], env=env,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                state.update(status='running', child_identity=identity(child.pid))
                write_json(folder / 'state.json', state)
                try:
                    code = child.wait(timeout=recipe['timeout_seconds'])
                    state.update(status='completed' if code == 0 else 'failed', exit_code=code)
                except subprocess.TimeoutExpired:
                    # Direct child is still owned and unreaped; kill only its new
                    # process group. It cannot have been recycled while unreaped.
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    state.update(status='timed_out', exit_code=child.returncode)
                jobs.artifacts(job_id)
    except Exception as exc:
        state.update(status='failed', error=repr(exc)); traceback.print_exc()
    finally:
        state['finished_unix'] = now()
        write_json(folder / 'state.json', state)


if __name__ == '__main__':
    main()
