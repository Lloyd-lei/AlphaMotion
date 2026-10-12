"""Integrity check of a dataset root: every manifest shard exists, whole-file sha256 == manifest, every blob sha256 ==
its header, catalog references resolve.

    python -m surface_retargeting.dataset.verify --root /home/arenalabs/umr_dataset/data [--quick]   (--quick: size + header only)
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import argparse
import json
import sys
import time
from pathlib import Path

from .container import Shard, file_sha256, read_header


def verify(root: Path, quick=False, log=print) -> dict:
    t0 = time.time()
    man = json.load(open(root / 'manifest.json'))
    bad, nbytes = [], 0
    for rel, info in sorted(man['shards'].items()):
        p = root / rel
        if not p.exists():
            bad.append((rel, 'missing')); continue
        if p.stat().st_size != info['nbytes']:
            bad.append((rel, 'size')); continue
        nbytes += info['nbytes']
        if quick:
            read_header(p); continue
        if file_sha256(p) != info['sha256']:
            bad.append((rel, 'file sha256')); continue
        blobs = Shard(p).verify()
        if not all(blobs.values()):
            bad.append((rel, 'blob sha256 ' + ','.join(k for k, v in blobs.items() if not v)))
    cm = json.load(open(root / 'catalog_meta.json'))
    for rel in cm['shards']:
        if rel not in man['shards']:
            bad.append((rel, 'catalog shard not in manifest'))
    dt = time.time() - t0
    log(f'[verify] {len(man["shards"])} shards, {nbytes / 1e9:.3f} GB, {len(bad)} problems, {dt:.1f}s '
        f'({nbytes / 1e6 / max(dt, 1e-9):.0f} MB/s)')
    for b in bad:
        log('[verify] BAD', *b)
    return dict(shards=len(man['shards']), bytes=nbytes, problems=bad, seconds=dt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(f'{SRCFG.dataset_root}'))
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(argv)
    r = verify(a.root, a.quick)
    sys.exit(1 if r['problems'] else 0)


if __name__ == '__main__':
    main()
