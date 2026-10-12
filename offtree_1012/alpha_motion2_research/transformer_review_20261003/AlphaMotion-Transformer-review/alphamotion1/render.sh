#!/bin/sh
set -eu
cd "$(dirname "$0")"
REDRAW_PYTHON=${REDRAW_PYTHON:-/Users/yudonglei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}
"$REDRAW_PYTHON" build.py
for name in alphamotion1_path alphamotion1_blocks; do
  pdftoppm -r 240 -singlefile -png "$name.pdf" "$name"
  pdftoppm -r 105 -singlefile -png "$name.pdf" "${name}_preview"
done
