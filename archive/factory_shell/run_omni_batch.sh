#!/bin/bash
# Convert every fully-downloaded OmniContact capture not yet converted. 14 parallel workers, 30 min cap each.
cd /home/arenalabs/umr_factory_data
PY=/home/arenalabs/miniconda3/envs/genmo-gem/bin/python
LIST=logs/omni_todo.txt
$PY - <<'PYEOF' > $LIST
import json, pathlib
raw = pathlib.Path('raw/omnicontact/raw_mocap'); out = pathlib.Path('smplx/omnicontact')
for m in sorted(raw.glob('*/*/*/capture_meta.json')):
    d = m.parent; meta = json.loads(m.read_text())
    need = [meta['human_motion_file'], meta['human_timestamps_file']] + [o['source_file'] for o in meta.get('objects', [])]
    if not all((d / f).exists() for f in need): continue
    if (out / meta['domain'] / meta['normalized_case'] / meta['capture_id'] / 'qa.json').exists(): continue
    print(d.resolve())
PYEOF
echo "todo $(wc -l < $LIST)"
cat $LIST | xargs -P 14 -I{} timeout 1800 $PY scripts/convert_omni.py {} >> logs/omni_convert.jsonl 2>> logs/omni_convert.err
echo "batch done"
