"""Write an explicit host recipe registry; never start training/downloads."""
from pathlib import Path
import argparse
import json
import sys


def configure(app, alpha, python):
    foundation = alpha / 'pretraining_foundation'
    scaling = alpha / 'research/alphamotion_scaling_20261004'
    state = app / 'agent_state'
    roots = {'foundation': str(foundation), 'scaling': str(scaling), 'workbench': str(app / 'qa')}
    no_parameters = {'type': 'object', 'properties': {}, 'additionalProperties': False}
    smoke = {'enabled': True, 'stage': 'validate', 'description': 'Real foundation schema/cache/FK smoke; no training',
             'argv': [str(python), '-m', 'pretraining_foundation', 'smoke', '--config', str(foundation / 'configs/smoke.json'),
                      '--output', '{output}/smoke_report.json'], 'cwd': str(alpha),
             'code_files': [str(p) for p in sorted(foundation.glob('*.py'))] + [str(foundation / 'configs/smoke.json')],
             'parameters': no_parameters, 'device': 'cpu', 'timeout_seconds': 300,
             'output_budget_bytes': 2 * 1024**3,
             'evidence_boundary': 'Software, file/schema/cache and kinematic checks only; no model training or simulation'}
    tests = {'enabled': True, 'stage': 'test', 'description': 'Foundation regression tests in an isolated pytest directory',
             'argv': [str(python), '-m', 'pytest', str(foundation / 'tests'), '-q', '-p', 'no:cacheprovider',
                      '--basetemp', '{output}/pytest', '--junitxml', '{output}/pytest.xml'],
             'cwd': str(alpha), 'parameters': no_parameters, 'device': 'cpu',
             'code_files': [str(p) for p in sorted(foundation.rglob('test_*.py'))] + smoke['code_files'],
             'timeout_seconds': 300, 'output_budget_bytes': 1024**3,
             'env': {'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}, 'evidence_boundary': 'Software regression checks'}
    # Historical six-run orchestration mutates its own fixed configs and receipts.
    # Do not repackage it as a supposedly isolated new training job.
    workflows = {'foundation_smoke': smoke, 'foundation_tests': tests}
    for name, stage, reason in [
        ('retarget', 'retarget', 'Existing UMR pipelines write to their own outputs; isolated input/output adapter still required'),
        ('train', 'train', 'Next-round configuration and isolated training adapter required; historical six-run queue is not reused'),
        ('evaluate', 'evaluate', 'Next-round checkpoint/data selection contract required; historical evaluation evidence is readable now'),
        ('simulate', 'simulate', 'No dynamics-validation workflow registered; kinematic replay does not satisfy this stage')]:
        workflows[name] = {'enabled': False, 'stage': stage, 'description': name,
                           'disabled_reason': reason, 'parameters': no_parameters}
    return {'schema': 'alphamotion-mcp-config-v1', 'data_root': '/home/arenalabs/umr_factory_data',
            'cache_root': str(app / 'cache/arrays'), 'state_root': str(state), 'evidence_roots': roots,
            'reserve_bytes': 16 * 1024**3, 'download_remaining_upper_bound_bytes': 200505865539,
            'workflows': workflows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--python', type=Path, default=Path('/home/arenalabs/miniconda3/bin/python'))
    p.add_argument('--alpha-root', type=Path, default=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2'))
    a = p.parse_args()
    if a.output.exists():
        sys.exit('Config exists; inspect and edit deliberately, never overwrite a live registry')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(configure(Path(__file__).resolve().parents[1], a.alpha_root, a.python), indent=2) + '\n')
    print(a.output)
