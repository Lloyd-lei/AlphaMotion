"""Run an upstream UMR pipeline module, routing its retarget subprocess through umr_capture_run.py.
Usage: umr_capture_pipe.py <pipeline_module_name> [pipeline args...]   (UMR env python)
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import importlib
import sys
from pathlib import Path

UMR = f'{SRCFG.umr_root}'
WRAP = str(Path(__file__).resolve().parent / 'umr_capture_run.py')
sys.path.insert(0, UMR + '/scripts'); sys.path.insert(0, UMR)
base = importlib.import_module('humanoid_retarget_pipeline')
orig = base.run_command
mod = importlib.import_module(sys.argv[1])


def rc(cmd, dry_run=False, *a, **k):
    cmd = list(cmd)
    if len(cmd) > 1 and str(cmd[1]).endswith('.py') and Path(str(cmd[1])).name.startswith('retarget_'):
        cmd = [cmd[0], WRAP] + cmd[1:]
    return orig(cmd, dry_run, *a, **k)


base.run_command = rc
if hasattr(mod, 'run_command'):
    mod.run_command = rc
sys.argv = [sys.argv[1]] + sys.argv[2:]
mod.main()
