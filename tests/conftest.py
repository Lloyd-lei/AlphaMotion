"""Shared pytest setup.  CPU-only on the workstation: the GPU engine / GPU converter scripts are never collected here."""
import os
from pathlib import Path

import pytest

# tests/gpu/*: CUDA scripts (module-level code); tests/convert/*: genmo-gem GPU converter parity/bench scripts;
# tests/strict/test_postpass.py: a script, run through tests/strict/test_postpass_smoke.py instead.
collect_ignore_glob = ['gpu/*', 'convert/*', 'strict/test_postpass.py']

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


@pytest.fixture
def tmp(tmp_path):
    """The dataset tests were written against a `tmp` Path argument."""
    return tmp_path


def need(path, what=None):
    """Skip (not fail) when factory data that a test reads is not on this machine."""
    if not Path(path).exists():
        pytest.skip(f'needs data: {what or path}')
