# source ops/sr_env.sh  -- export SR_* paths (configs/sr.toml) and put this repo on PYTHONPATH for the current shell.
# Use when a shared env (UMR env, genmo-gem) must import surface_retargeting without `pip install -e .` into it.
_SR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
eval "$(python3 "$_SR_REPO/surface_retargeting/config.py" --export)"
export PYTHONPATH="$_SR_REPO${PYTHONPATH:+:$PYTHONPATH}"
unset _SR_REPO
