"""Loader throughput: samples/s, frames/s, decoded MB/s for DataLoader worker counts; optional cold page cache.

    python scripts/bench_loader.py --root data --window 64 --batch 256 --workers 0,4,8,16 --seconds 20 [--cold]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_retargeting.dataset.dataset import PairedWindows, collate        # noqa: E402


def drop_cache(root: Path):
    n = 0
    for p in list(root.rglob('*.umrs')):
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
        finally:
            os.close(fd)
    return n


def nbytes(batch):
    t = 0
    for v in batch.values():
        if isinstance(v, dict):
            t += nbytes(v)
        elif torch.is_tensor(v):
            t += v.numel() * v.element_size()
    return t


def run(root, mode, window, batch, workers, seconds, cold, seed, fields):
    ds = PairedWindows(root, window=window, pair_mode=mode, n_bodies=3 if mode == 'multi' else 2, seed=seed,
                       samples_per_epoch=10_000_000, fields=fields, val_permille=0)
    if cold:
        drop_cache(Path(root))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate,
                                     persistent_workers=workers > 0, prefetch_factor=4 if workers else None,
                                     pin_memory=True)
    it = iter(dl)
    first = next(it)                                   # warm-up (worker start)
    t0 = time.time(); nb = 0; by = 0; ns = 0
    while time.time() - t0 < seconds:
        b = next(it)
        nb += 1; ns += len(b['clip']); by += nbytes(b)
    dt = time.time() - t0
    del it, dl
    return dict(mode=mode, workers=workers, window=window, batch=batch, cold=cold, fields=list(fields),
                samples_per_s=round(ns / dt, 1), frames_per_s=round(ns * window / dt, 1),
                body_frames_per_s=round(ns * window * (3 if mode == 'multi' else 2) / dt, 1),
                decoded_MB_per_s=round(by / dt / 1e6, 1), batches=nb, seconds=round(dt, 2),
                eligible_clips=len(ds.clips))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/arenalabs/umr_dataset/data')
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', default='0,4,8,16')
    ap.add_argument('--modes', default='human2robot,robot2robot,multi')
    ap.add_argument('--seconds', type=float, default=15)
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--fields', default='labels')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = []
    for mode in a.modes.split(','):
        for w in [int(x) for x in a.workers.split(',')]:
            r = run(a.root, mode, a.window, a.batch, w, a.seconds, a.cold, 0, tuple(a.fields.split(',')))
            print(json.dumps(r), flush=True); res.append(r)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
