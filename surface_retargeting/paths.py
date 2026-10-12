"""Fixed locations. Upstream UMR is used read-only; everything we write goes under DATA (on /home)."""
from pathlib import Path
import sys

UMR_ROOT = Path('/var/tmp/alphamotion/umr_factory/umr/UMR')
UMR_SCRIPTS = UMR_ROOT / 'scripts'
UMR_PY = '/var/tmp/alphamotion/umr_factory/umr/env/bin/python'
UMR_TOOLS = Path('/var/tmp/alphamotion/umr_factory/umr/tools')        # scout wrappers (umr_pipe.py / umr_run.py)
INVENTORY = Path('/var/tmp/alphamotion/umr_factory/robots/robots.json')
DRAFT_CONFIGS = Path('/var/tmp/alphamotion/umr_factory/robots/umr_configs')
SURFACE_V2 = Path('/var/tmp/alphamotion/am2exp/assets/surface_v2')

CODE = Path('/home/arenalabs/umr_factory_code')
DATA = Path('/home/arenalabs/umr_factory_data')
ROBOTS = DATA / 'robots'          # staged MJCF + UMR config + dense surface per robot
CORR = DATA / 'corr'              # correspondence caches (dataset npz + trained slots)
RUNS = DATA / 'runs'              # raw UMR outputs + captured human slots
OUT = DATA / 'out'                # final per-(clip,robot) npz + manifest + QA
LOGS = DATA / 'logs'


def add_umr_to_path():
    for p in (str(UMR_SCRIPTS), str(UMR_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
