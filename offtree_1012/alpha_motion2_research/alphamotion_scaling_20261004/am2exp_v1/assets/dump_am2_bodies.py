"""Subprocess helper: dump the exact AM2 body definitions (orchestration/scaling_data.load_scaling_bodies, config
A_S_seed0) for the 18 surface bodies to JSON.  Runs in its own process because the sealed base/legacy/common.py
and pose_translator_design/exp_topo/common.py cannot both be imported as ``common`` in one interpreter.

Usage: python -m am2exp_v1.assets.dump_am2_bodies OUT.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True


def main(out_path):
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))  # CODE (for assets._env)
    from assets import _env as E  # noqa: E402  (loads common/paths.py by location)
    del sys.path[0]
    for p in reversed([E.ROOT / 'orchestration', E.ROOT / 'base', E.ROOT / 'base' / 'legacy', E.ROOT, E.ROOT / 'prismatic']):
        sys.path.insert(0, str(p))
    import numpy as np
    import scaling_data  # sealed, read-only
    cfg = json.loads(E.CONFIG_PATH.read_text())
    t0 = time.time()
    bodies, audit = scaling_data.load_scaling_bodies(E.ROOT, cfg, 'cpu', include_surface=False)
    out = {'schema': 'am2_bodies_dump_v1', 'config': str(E.CONFIG_PATH), 'config_sha256': E.sha256_file(E.CONFIG_PATH),
           'load_seconds': time.time() - t0, 'bodies': {}}
    for b in bodies:
        if b.name not in E.ALL_BODIES:
            continue
        a = b.arrays
        rec = dict(name=b.name, domain=b.domain, J=len(b.parents), names=list(b.names), parents=[int(x) for x in b.parents],
                   offsets_cm=np.asarray(b.offsets, np.float64).tolist(), reach_cm=float(b.reach),
                   endpoint=b.endpoint, metric=bool(b.metric), n_rows=int(len(a['pose9'])),
                   split_counts={str(s): int((a['split'] == s).sum()) for s in (0, 1, 2)})
        ts = getattr(b, '_typed_static', {})
        for k, v in ts.items():
            rec['typed_' + k] = np.asarray(v).tolist()
        if b.domain == 'xwbc':
            pl = np.load(E.PILOT_XWBC / 'payloads' / f'{b.name}.npz', allow_pickle=False)
            pos = a['pose9'][..., 6:].astype(np.float64) * b.reach
            rec['payload_position_cm_max_abs_diff'] = float(np.abs(pos - pl['position_cm']).max())
            rec['payload_path'] = str(E.PILOT_XWBC / 'payloads' / f'{b.name}.npz')
            rec['ground_static'] = b.metadata.get('ground_static', {})
        out['bodies'][b.name] = rec
    missing = sorted(set(E.ALL_BODIES) - set(out['bodies']))
    if missing:
        raise SystemExit('missing bodies ' + str(missing))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({'written': str(out_path), 'bodies': len(out['bodies']), 'seconds': time.time() - t0}))


if __name__ == '__main__':
    main(sys.argv[1])
