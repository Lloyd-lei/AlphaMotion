"""Persistent, idempotent jobs for explicit server-side workflow recipes.

No shell tool, no download launcher, no implicit retry/resume. A plan records the
exact recipe and input files; submit rechecks these before starting a worker.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import jsonschema

from .common import checked_id, digest, file_lock, identity, inside, now, read_json, source_stamp, write_json


class Jobs:
    def __init__(self, settings, config_path):
        self.settings = settings
        self.config_path = Path(config_path).resolve()
        self.root = Path(settings['state_root']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def workflows(self):
        result = []
        for key, recipe in self.settings.get('workflows', {}).items():
            missing = [p for p in recipe.get('code_files', []) if not Path(p).is_file()]
            enabled = bool(recipe.get('enabled')) and not missing
            result.append({'id': key, 'stage': recipe['stage'], 'description': recipe['description'],
                           'enabled': enabled, 'reason': recipe.get('disabled_reason') if not enabled else None,
                           'missing_files': missing, 'parameters': recipe.get('parameters', {}),
                           'timeout_seconds': recipe.get('timeout_seconds'),
                           'evidence_boundary': recipe.get('evidence_boundary')})
        return result

    def plan(self, workflow, parameters):
        recipe = self.settings.get('workflows', {}).get(workflow)
        if not recipe or not recipe.get('enabled'):
            raise ValueError('Workflow unavailable: ' + (recipe or {}).get('disabled_reason', workflow))
        jsonschema.validate(parameters, recipe['parameters'])
        params = dict(parameters)
        stamps = [source_stamp(p) for p in recipe.get('code_files', [])]
        for name, root_name in recipe.get('input_files', {}).items():
            p = inside(self.settings['evidence_roots'][root_name], params[name])
            if not p.is_file():
                raise ValueError(f'Input file missing: {name}')
            stamps.append(source_stamp(p)); params[name] = str(p)
        plan_id = uuid.uuid4().hex
        output = self.root / 'jobs' / plan_id / 'artifacts'
        params['output'] = str(output)
        argv = [part.format_map(params) for part in recipe['argv']]
        if not Path(argv[0]).is_file() or not Path(recipe['cwd']).is_dir():
            raise ValueError('Workflow executable or working directory missing')
        plan = {'schema': 'alphamotion-job-plan-v1', 'plan_id': plan_id, 'workflow': workflow,
                'parameters': parameters, 'argv': argv, 'cwd': recipe['cwd'], 'output': str(output),
                'recipe': recipe, 'source_files': stamps, 'created_unix': now(),
                'config_sha256': digest(self.config_path), 'evidence_boundary': recipe.get('evidence_boundary')}
        self.capacity(recipe)
        write_json(self.root / 'plans' / (plan_id + '.json'), plan)
        return plan

    def capacity(self, recipe):
        # Reserve the *entire* still-planned download total conservatively. This
        # never treats stale transfer counters as available capacity.
        free = shutil.disk_usage(self.root).free
        remaining_bound = int(self.settings.get('download_remaining_upper_bound_bytes', 0))
        required = remaining_bound + int(self.settings.get('reserve_bytes', 16 * 1024**3)) + int(recipe.get('output_budget_bytes', 0))
        if free < required:
            raise ValueError(f'Insufficient capacity: free={free}, required={required}')
        return {'free_bytes': free, 'required_bytes': required, 'download_remaining_upper_bound_bytes': remaining_bound}

    def validate(self, plan):
        if digest(self.config_path) != plan['config_sha256']:
            raise ValueError('Workflow configuration changed; create a fresh plan')
        if now() - plan['created_unix'] > 3600:
            raise ValueError('Plan is older than one hour; create a fresh plan')
        for item in plan['source_files']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('Source or input changed; create a fresh plan: ' + item['path'])
        return self.capacity(plan['recipe'])

    def submit(self, plan_id, idempotency_key):
        checked_id(plan_id); checked_id(idempotency_key)
        key = hashlib.sha256(idempotency_key.encode()).hexdigest()
        with file_lock(self.root / 'submit.lock'):
            prior = self.root / 'requests' / (key + '.json')
            if prior.exists():
                r = read_json(prior)
                if r['plan_id'] != plan_id:
                    raise ValueError('Idempotency key already belongs to a different plan')
                return self.status(plan_id)
            path = self.root / 'jobs' / plan_id
            if path.exists():
                write_json(prior, {'plan_id': plan_id})
                return self.status(plan_id)
            plan = read_json(self.root / 'plans' / (plan_id + '.json'))
            cap = self.validate(plan)
            path.mkdir(parents=True); (path / 'artifacts').mkdir()
            write_json(path / 'plan.json', plan)
            write_json(path / 'state.json', {'job_id': plan_id, 'status': 'submitted', 'created_unix': now(), 'capacity': cap})
            write_json(prior, {'plan_id': plan_id})
            try:
                with (path / 'worker.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen([sys.executable, '-m', 'agent_tools.worker', str(self.config_path), plan_id],
                        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                        stdout=log, stderr=log, start_new_session=True)
                write_json(path / 'worker.json', {'pid': child.pid, 'identity': identity(child.pid)})
            except Exception as exc:
                write_json(path / 'state.json', {'job_id': plan_id, 'status': 'failed', 'error': repr(exc), 'exit_code': None})
                raise
        return self.status(plan_id)

    def status(self, job_id):
        path = self.root / 'jobs' / checked_id(job_id)
        state = read_json(path / 'state.json')
        ident = state.get('worker_identity')
        if state['status'] in ('running', 'waiting_for_slot') and ident and identity(ident['pid']) != ident:
            state = dict(state, status='interrupted', error='Worker identity absent or changed; no automatic restart')
        if state['status'] == 'submitted' and now() - state.get('created_unix', now()) > 30:
            state = dict(state, status='startup_unconfirmed', error='Inspect worker.log; no automatic restart')
        return state

    def logs(self, job_id, offset=0, max_bytes=16000):
        if offset < 0 or not 1 <= max_bytes <= 64000:
            raise ValueError('Log bounds invalid')
        path = self.root / 'jobs' / checked_id(job_id) / 'worker.log'
        with path.open('rb') as f:
            f.seek(offset); raw = f.read(max_bytes)
        return {'text': raw.decode(errors='replace'), 'next_offset': offset + len(raw), 'bytes_total': path.stat().st_size}

    def artifacts(self, job_id):
        root = self.root / 'jobs' / checked_id(job_id) / 'artifacts'
        files = []
        for p in sorted(root.rglob('*')):
            if len(files) >= 500:
                raise ValueError('More than 500 artifacts; use a scoped evidence directory')
            if p.is_file() and p.resolve().is_relative_to(root.resolve()):
                files.append({'path': str(p.relative_to(self.root)), 'bytes': p.stat().st_size})
        return {'job_id': job_id, 'scope': 'agent_state', 'artifacts': files}
