#!/bin/sh
# Executed on ArenaLabs. Stdout is exclusively the MCP transport.
set -eu
cd /home/arenalabs/motion_audit_web
exec .mcp-venv/bin/python -m agent_tools.server --config configs/mcp.json
