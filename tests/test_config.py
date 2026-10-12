"""Config: local defaults equal the pre-migration literals; env overrides; config JSON tokens round-trip."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from surface_retargeting import config as C

OLD = {  # the literals that were hard-coded before 2026-10-12
    'data_root': '/home/arenalabs/umr_factory_data',
    'work': '/var/tmp/alphamotion/umr_factory/umr_gpu',
    'umr_root': '/var/tmp/alphamotion/umr_factory/umr/UMR',
    'python_env': '/var/tmp/alphamotion/umr_factory/umr/env',
    'robot_assets_root': '/var/tmp/alphamotion/robot_models',
    'smplx_neutral': '/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz',
    'beta_fit': '/var/tmp/alphamotion/umr_factory/beta_fit',
}


def test_local_defaults_unchanged():
    if C.LOCAL.exists() or any(k.startswith('SR_') for k in os.environ):
        pytest.skip('configs/sr.toml or SR_* env present: not the stock local config')
    cfg = C.Config(env={})
    for k, v in OLD.items():
        assert getattr(cfg, k) == v, k
    assert cfg.umr_python == OLD['python_env'] + '/bin/python'


def test_env_override_and_interpolation():
    cfg = C.Config(env={'SR_DATA_ROOT': '/data/umr', 'SR_BODY_MODELS': '/models'})
    assert cfg.data_root == '/data/umr'
    assert cfg.smplx_neutral == '/models/SMPLX_NEUTRAL.npz'


def test_minimal_toml_parser_matches_tomllib():
    text = C.EXAMPLE.read_text()
    tomllib = pytest.importorskip('tomllib')
    import builtins
    real_import = builtins.__import__

    def no_tomllib(name, *a, **k):
        if name == 'tomllib':
            raise ModuleNotFoundError(name)
        return real_import(name, *a, **k)
    builtins.__import__ = no_tomllib
    try:
        fallback = C._parse_toml(text)
    finally:
        builtins.__import__ = real_import
    assert fallback == tomllib.loads(text)


def test_config_jsons_expand_cleanly():
    cfg = C.Config(env={})
    n = 0
    for f in sorted((C.REPO / 'configs').rglob('*.json')):
        d = cfg.load_json(f)
        assert not re.search(r'\$\{SR_[A-Z_]+\}', json.dumps(d)), f
        n += 1
    assert n > 90


def test_registry_roundtrip_against_old_file():
    old = Path(OLD['work']).parent / 'robots' / 'robots.json'
    if not old.exists():
        pytest.skip('old registry not on this machine')
    cfg = C.Config(env={})
    assert cfg.load_json(cfg.registry_json) == json.load(open(old))


def test_shell_export():
    out = subprocess.run([sys.executable, str(C.REPO / 'surface_retargeting' / 'config.py'), '--export'],
                         capture_output=True, text=True, check=True).stdout
    assert 'export SR_DATA_ROOT=' in out and 'export SR_UMR_PYTHON=' in out
