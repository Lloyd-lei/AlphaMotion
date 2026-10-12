"""Fixed locations (CPU factory). All machine paths come from surface_retargeting.config (configs/sr.toml + SR_* env).
Upstream UMR is used read-only; everything we write goes under DATA."""
from pathlib import Path

from surface_retargeting.config import CFG as SRCFG, REPO

UMR_ROOT = SRCFG.path('umr_root')
UMR_SCRIPTS = UMR_ROOT / 'scripts'
UMR_PY = SRCFG.umr_python
UMR_TOOLS = Path(SRCFG.umr_tools)          # UMR wrappers (umr_pipe.py / umr_run.py), in this repo
INVENTORY = Path(SRCFG.registry_json)      # ${SR_*} tokens: read with SRCFG.load_json
DRAFT_CONFIGS = Path(SRCFG.draft_configs)  # ${SR_*} tokens: read with SRCFG.load_json
SURFACE_V2 = SRCFG.path('am2exp_root') / 'assets' / 'surface_v2'

CODE = REPO
DATA = SRCFG.path('data_root')
ROBOTS = DATA / 'robots'          # staged MJCF + UMR config + dense surface per robot
CORR = DATA / 'corr'              # correspondence caches (dataset npz + trained slots)
RUNS = DATA / 'runs'              # raw UMR outputs + captured human slots
OUT = DATA / 'out'                # final per-(clip,robot) npz + manifest + QA
LOGS = DATA / 'logs'


def add_umr_to_path():
    SRCFG.add_umr_to_path()
