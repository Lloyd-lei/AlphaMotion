#!/bin/sh
# Executed on ArenaLabs. Stdout is exclusively the MCP transport.
set -eu
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
eval "$(python3 "$REPO/surface_retargeting/config.py" --export)"   # SR_* paths from configs/sr.toml
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/viewer"
exec "$SR_VIEWER_HOME/.mcp-venv/bin/python" -m agent_tools.server --config configs/mcp.json
