#!/bin/bash
# Progress: jobs done / total, robot-hours produced, kept fraction, throughput, last log lines.
OUT=/home/arenalabs/umr_factory_data/production/v1
/var/tmp/alphamotion/umr_factory/umr/env/bin/python - <<PY
import json, os, glob, time
out = '$OUT'
jobs = [json.loads(l) for l in open(out + '/_jobs/jobs.jsonl')]
prog = [json.loads(l) for l in open(out + '/_jobs/progress.jsonl')] if os.path.exists(out + '/_jobs/progress.jsonl') else []
done = {p['job_id'] for p in prog}
fails = glob.glob(out + '/_jobs/*.failed*')
rh = sum(p['robot_hours'] for p in prog); kept = sum(p['kept_frames'] for p in prog); fr = sum(p['frames'] for p in prog)
wall = sum(p.get('wall_s', 0) for p in prog)
print(f'jobs done {len(done)}/{len(jobs)}  failed markers {len(fails)}')
print(f'robot-hours produced {rh:.2f} (kept {kept / max(fr, 1):.3f})  job wall {wall / 3600:.2f} h  -> {rh / max(wall / 3600, 1e-9):.2f} robot-h per wall-h')
by = {}
for p in prog:
    b = by.setdefault(p['robot'], [0, 0, 0]); b[0] += p['robot_hours']; b[1] += p['kept_frames']; b[2] += p['frames']
for r, (h, k, f) in by.items():
    print(f'  {r}: {h:.2f} robot-h, kept {k / max(f, 1):.3f}')
PY
pid=$(cat $OUT/_logs/runner.pid 2>/dev/null); if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then echo "runner running pid $pid"; else echo "runner NOT running"; fi
tail -n 5 $OUT/_logs/runner.log
