"""Strict post-pass smoke: run the factory's synthetic post-pass script (CPU, MuJoCo + Clarabel) on a staged robot.
Writes its json into a temp dir (SR_TEST_OUT), never into the factory data tree."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from surface_retargeting.config import CFG
from conftest import need

SCRIPT = Path(__file__).with_name('test_postpass.py')


@pytest.mark.data
@pytest.mark.parametrize('robot', ['unitree_g1_29dof'])
def test_postpass_script(robot, tmp_path):
    need(f'{CFG.data_root}/robots/{robot}/staging.json', f'staged robot {robot}')
    need(f'{CFG.data_root}/robots/{robot}/surface.npz', f'surface points {robot}')
    env = dict(os.environ, SR_TEST_OUT=str(tmp_path), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    r = subprocess.run([sys.executable, str(SCRIPT), robot], capture_output=True, text=True, env=env, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert 'ALL TESTS PASSED' in r.stdout
    assert (tmp_path / f'test_postpass_{robot}.json').exists()
