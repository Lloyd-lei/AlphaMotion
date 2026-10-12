"""Shard compaction: drop dead shards, merge small / mostly-dead robot shards into large ones (byte-exact row copy;
robot quantisation ranges are per-robot constants, so no re-encoding).  Human shards carry a per-shard axis-angle
scale and are only dropped when dead, never rewritten.

    python -m umr_dataset.compact --root /home/arenalabs/umr_dataset/data [--target-rows 2000000] [--grace-hours 6]

Takes the converter lock.  Replaced files stay at their path (listed in <root>/pending_delete.json) and are deleted by a later
run once older than --grace-hours, so a training job that still holds the previous catalog keeps reading (reload the
catalog per epoch).  --grace-hours 0 deletes immediately.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np

from . import catalog as CAT
from .container import Shard, write_shard


def _pending(root: Path) -> dict:
    p = root / 'pending_delete.json'
    return json.load(open(p)) if p.exists() else {}


def _save_pending(root: Path, d: dict):
    tmp = root / 'pending_delete.json.tmp'
    json.dump(d, open(tmp, 'w'), indent=1)
    os.replace(tmp, root / 'pending_delete.json')


def _empty_trash(root: Path, grace_s: float, log):
    """Delete replaced shard files once older than the grace period (they stay at their path until then)."""
    d = _pending(root)
    man = CAT.load_manifest(root)
    n = 0
    for rel, t in list(d.items()):
        if rel in man['shards']:                      # re-registered meanwhile: keep
            d.pop(rel); continue
        if time.time() - t > grace_s:
            p = root / rel
            if p.exists():
                p.unlink()
            d.pop(rel); n += 1
    _save_pending(root, d)
    if n:
        log(f'[compact] deleted {n} replaced shard files older than {grace_s / 3600:.1f} h')
    return n


def compact(root: Path, target_rows=2_000_000, min_live_frac=0.6, grace_hours=6.0, log=print) -> dict:
    root = Path(root)
    lockf = open(root / '.convert.lock', 'a+')
    fcntl.flock(lockf, fcntl.LOCK_EX)
    t0 = time.time()
    try:
        _empty_trash(root, grace_hours * 3600, log)
        man = CAT.load_manifest(root)
        live, shard_meta = CAT.live_entries(root, man)
        live_by_shard = {}
        for (clip, body), ((_, rel), ents) in live.items():
            live_by_shard.setdefault(rel, []).extend(ents)
        dead = [rel for rel in man['shards'] if rel not in live_by_shard]
        # robot merge candidates
        groups = {}
        for rel, ents in live_by_shard.items():
            m = shard_meta[rel]
            if 'robot' not in m:
                continue
            total = sum(e['n'] for e in CAT._shard_entries(root, rel, man['shards'][rel]['sha256'])['entries'])
            nlive = sum(e['n'] for e in ents)
            small = total < target_rows // 4
            if small or nlive < min_live_frac * total:
                key = (m['robot'], m['tier'], bool(m['has_object']), json.dumps(m['q_lo']), json.dumps(m['q_step']))
                groups.setdefault(key, []).append((rel, ents))
        new_infos, replaced = [], []
        import uuid
        batch = 'c' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + f'-{os.getpid()}-{uuid.uuid4().hex[:6]}'
        for key, lst in sorted(groups.items()):
            if len(lst) < 2 and all(sum(e['n'] for e in ents) >= min_live_frac * sum(
                    x['n'] for x in CAT._shard_entries(root, rel, man['shards'][rel]['sha256'])['entries']) for rel, ents in lst):
                continue                                       # a single small, fully live shard: nothing to gain
            robot = key[0]
            # pack entries (keep each (clip, body) together; a clip's segments live in one shard)
            items = []
            for rel, ents in lst:
                by_cb = {}
                for e in ents:
                    by_cb.setdefault((e['clip'], e['body']), []).append(e)
                items += [(rel, v) for v in by_cb.values()]
            packs, cur, rows = [], [], 0
            for rel, ents in items:
                n = sum(e['n'] for e in ents)
                if cur and rows + n > target_rows:
                    packs.append(cur); cur, rows = [], 0
                cur.append((rel, ents)); rows += n
            if cur:
                packs.append(cur)
            for pi, pk in enumerate(packs):
                N = sum(e['n'] for _, ents in pk for e in ents)
                first = Shard(root / pk[0][0])
                blob_names = list(first.header['blobs'])
                outb = {}
                for bn in blob_names:
                    b0 = first.blob(bn)
                    outb[bn] = np.empty((N,) + b0.shape[1:], b0.dtype)
                entries, row = [], 0
                shards = {}
                for rel, ents in pk:
                    sh = shards.get(rel) or shards.setdefault(rel, Shard(root / rel))
                    for e in sorted(ents, key=lambda x: x['k0']):
                        for bn in blob_names:
                            outb[bn][row:row + e['n']] = sh.blob(bn)[e['row0']:e['row0'] + e['n']]
                        e2 = dict(e); e2['row0'] = row; e2['compacted_from'] = rel
                        entries.append(e2); row += e['n']
                meta = {k: v for k, v in shard_meta[pk[0][0]].items()}
                meta.update(entries=entries, created=time.time(), batch=batch)
                name = f"{robot}-{'hoi' if key[2] else 'body'}-{batch}-{pi:03d}.umrs"
                info = write_shard(root / 'robots' / robot / name, outb, meta, kind='robot')
                info['rel'] = str(Path('robots') / robot / name)
                new_infos.append(info)
            replaced += [rel for rel, _ in lst]
        CAT.register_shards(root, new_infos)
        drop = sorted(set(dead) | set(replaced))
        CAT.unregister_shards(root, drop)
        CAT.build(root, log=log)
        freed = 0
        pend = _pending(root)
        for rel in drop:
            p = root / rel
            if p.exists():
                freed += p.stat().st_size
                pend.setdefault(rel, time.time())
        _save_pending(root, pend)
        if grace_hours <= 0:
            _empty_trash(root, -1, log)
        res = dict(dead=len(dead), merged=len(replaced), new=len(new_infos), pending_delete_bytes=freed, seconds=time.time() - t0)
        log('[compact] ' + json.dumps(res))
        return res
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/home/arenalabs/umr_dataset/data'))
    ap.add_argument('--target-rows', type=int, default=2_000_000)
    ap.add_argument('--min-live-frac', type=float, default=0.6)
    ap.add_argument('--grace-hours', type=float, default=6.0)
    a = ap.parse_args(argv)
    compact(a.root, a.target_rows, a.min_live_frac, a.grace_hours)


if __name__ == '__main__':
    main()
