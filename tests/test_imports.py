"""Import smoke for the CPU library modules (no GPU init) + byte-compile every Python file in the repo."""
import importlib
import py_compile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CPU_MODULES = [
    'surface_retargeting.config', 'surface_retargeting.paths',
    'surface_retargeting.robots.robots', 'surface_retargeting.robots.surface',
    'surface_retargeting.robots.sampling_reference.umr_sampler', 'surface_retargeting.robots.sampling_reference.umr_reference',
    'surface_retargeting.correspondence.shared_corr', 'surface_retargeting.retarget.pipeline',
    'surface_retargeting.strict.postpass', 'surface_retargeting.strict.geometry', 'surface_retargeting.strict.objects',
    'surface_retargeting.labels.labels', 'surface_retargeting.gpu.labels', 'surface_retargeting.gpu.slip',
    'surface_retargeting.qa.judge',
    'surface_retargeting.dataset.schema', 'surface_retargeting.dataset.container', 'surface_retargeting.dataset.catalog',
    'surface_retargeting.dataset.compact', 'surface_retargeting.dataset.convert', 'surface_retargeting.dataset.dataset',
    'surface_retargeting.dataset.kinematics', 'surface_retargeting.dataset.sources', 'surface_retargeting.dataset.verify',
]


@pytest.mark.parametrize('mod', CPU_MODULES)
def test_import(mod):
    importlib.import_module(mod)


def test_compile_all(tmp_path):
    bad = []
    for f in sorted(REPO.rglob('*.py')):
        rel = f.relative_to(REPO)
        if rel.parts[0] in ('third_party', '.venv', 'archive') or 'node_modules' in rel.parts:
            continue
        try:
            py_compile.compile(str(f), cfile=str(tmp_path / 'x.pyc'), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f'{rel}: {e.msg}')
    assert not bad, bad
