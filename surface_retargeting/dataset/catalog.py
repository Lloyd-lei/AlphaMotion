"""Manifest (shard list + whole-file sha256) and catalog (live (clip, body) entries as flat numpy arrays).

Files under <root>:
  manifest.json      {"format": "umr_dataset", "shards": {rel: {"sha256", "nbytes", "kind", "created"}}, ...}
  catalog.npz        flat arrays used by the loader (loaded in < 0.1 s; see build())
  catalog_meta.json  body / clip / shard names, per-robot specs summary, totals
  clips.jsonl        one line per clip: human meta (betas, gender, up, fps, text annotation, objects, ...)
  ingest_index.npz   (clip, body) -> source npz size / mtime / sha256 (converter only)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .container import read_header

ESSENTIAL = ('clip', 'body', 'row0', 'n', 'k0', 'valid_runs', 'src', 'stride', 'fps', 'fps_src', 'has_object',
             'ingested_at', 'n_flagged', 'scale', 'floor_offset', 'n_src')


def _atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_manifest(root: Path) -> dict:
    p = root / 'manifest.json'
    if p.exists():
        return json.load(open(p))
    return dict(format='umr_dataset', version=1, shards={})


def register_shards(root: Path, infos: list):
    man = load_manifest(root)
    for i in infos:
        man['shards'][i['rel']] = dict(sha256=i['sha256'], nbytes=i['nbytes'], created=time.time())
    man['updated'] = time.time()
    _atomic_json(root / 'manifest.json', man)


def _shard_entries(root: Path, rel: str, sha: str):
    """Parsed header (cached by shard sha256: shards are immutable)."""
    cache = root / 'cache' / f'{sha[:32]}.pkl'
    if cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)
    h = read_header(root / rel)
    meta = h['meta']
    ents = meta.pop('entries')
    out = dict(kind=h['kind'], meta=meta, entries=ents)
    cache.parent.mkdir(exist_ok=True)
    tmp = cache.with_name(cache.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, cache)
    return out


def unregister_shards(root: Path, rels: list):
    man = load_manifest(root)
    for r in rels:
        man['shards'].pop(r, None)
    man['updated'] = time.time()
    _atomic_json(root / 'manifest.json', man)


def live_entries(root: Path, man: dict = None):
    """(live, shard_meta): live[(clip, body)] = [(ingested_at, shard_rel), [segment entries]] (newest ingest wins)."""
    man = man or load_manifest(root)
    groups = {}                                        # (clip, body) -> {(shard_rel, src_sha): [entries]}
    shard_meta = {}
    for rel, info in man['shards'].items():
        if not (root / rel).exists():
            continue
        s = _shard_entries(root, rel, info['sha256'])
        shard_meta[rel] = s['meta']
        for e in s['entries']:
            g = groups.setdefault((e['clip'], e['body']), {})
            g.setdefault((rel, (e.get('src') or {}).get('sha256') or e.get('file_sha256') or ''), []).append(e)
    live = {}
    for key, g in groups.items():
        best = max(g.items(), key=lambda kv: (max(x.get('ingested_at', 0) for x in kv[1]), kv[0][0]))
        live[key] = [(max(x.get('ingested_at', 0) for x in best[1]), best[0][0]), best[1]]
    return live, shard_meta


def build(root: Path, log=print) -> dict:
    """Rebuild catalog.npz / catalog_meta.json / clips.jsonl / ingest_index.npz from all manifest shards."""
    t0 = time.time()
    man = load_manifest(root)
    live, shard_meta = live_entries(root, man)
    bodies = ['human'] + sorted({b for (_, b) in live if b != 'human'})
    clips = sorted({c for (c, _) in live})
    shards = sorted({v[0][1] for v in live.values()})
    bi = {b: i for i, b in enumerate(bodies)}; ci = {c: i for i, c in enumerate(clips)}; si = {s: i for i, s in enumerate(shards)}
    rows = []
    for (c, b), (_, ents) in live.items():
        for e in ents:
            rows.append((ci[c], bi[b], e))
    rows.sort(key=lambda r: (r[0], r[1], r[2]['k0']))
    E = len(rows)
    e_clip = np.array([r[0] for r in rows], np.int32); e_body = np.array([r[1] for r in rows], np.int16)
    e_shard = np.array([si[_rel_of(live, clips[r[0]], bodies[r[1]])] for r in rows], np.int32)
    e_row0 = np.array([r[2]['row0'] for r in rows], np.int64); e_n = np.array([r[2]['n'] for r in rows], np.int32)
    e_k0 = np.array([r[2]['k0'] for r in rows], np.int32)
    e_obj = np.array([bool(r[2].get('has_object') or r[2].get('n_obj')) for r in rows], bool)
    e_nflag = np.array([r[2].get('n_flagged', 0) for r in rows], np.int32)
    e_scale = np.array([r[2].get('scale', 1.0) if r[1] else 1.0 for r in rows], np.float32)
    e_floor = np.array([r[2].get('floor_offset', 0.0) if r[1] else r[2].get('human_floor_z', 0.0) for r in rows], np.float32)
    ptr = np.zeros(E + 1, np.int64)
    runs = []
    for i, r in enumerate(rows):
        k0 = r[2]['k0']
        rr = [[k0 + a, k0 + b] for a, b in r[2]['valid_runs']]
        runs += rr; ptr[i + 1] = ptr[i] + len(rr)
    runs = np.array(runs, np.int32).reshape(-1, 2)
    # clip tables
    C = len(clips)
    c_ptr = np.zeros(C + 1, np.int64)
    np.add.at(c_ptr, e_clip + 1, 1); c_ptr = np.cumsum(c_ptr)
    c_fps = np.zeros(C, np.float32); c_stride = np.zeros(C, np.int16); c_nk = np.zeros(C, np.int32)
    c_hash = np.array([int(hashlib.sha1(c.encode()).hexdigest()[:8], 16) % 1000 for c in clips], np.int16)
    c_dataset = []
    clip_lines = []
    c_betas = np.zeros((C, 16), np.float32); c_nbetas = np.zeros(C, np.int16); c_up = np.zeros(C, np.int8)
    c_gender = np.zeros(C, np.int8); c_floor_h = np.zeros(C, np.float32)
    hum = {(c): ents[0] for (c, b), (_, ents) in live.items() if b == 'human'}
    for i, r in enumerate(rows):
        c = r[0]
        c_fps[c] = r[2].get('fps', 0); c_stride[c] = r[2].get('stride', 0)
        c_nk[c] = max(c_nk[c], r[2]['k0'] + r[2]['n'])
    for c in clips:
        h = hum.get(c)
        if h:
            i = ci[c]; bt = np.asarray(h.get('betas') or [], np.float32)[:16]
            c_betas[i, :len(bt)] = bt; c_nbetas[i] = len(bt); c_up[i] = 1 if h.get('up') == 'y' else 0
            c_gender[i] = {'neutral': 0, 'male': 1, 'female': 2}.get(h.get('gender'), 0); c_floor_h[i] = h.get('human_floor_z', 0.0)
        c_dataset.append(h.get('dataset', 'other') if h else _dataset_guess(c, live))
        if h:
            clip_lines.append({k: h.get(k) for k in ('clip', 'dataset', 'fps', 'fps_src', 'stride', 'n_src', 'betas', 'gender', 'up',
                                                     'human_floor_z', 'floor_from_robot', 'has_extra', 'n_obj', 'objects',
                                                     'metadata', 'layout', 'file', 'file_sha256', 'human_scale_src')})
        else:
            clip_lines.append(dict(clip=c, human=None))
    datasets = sorted(set(c_dataset)); di = {d: i for i, d in enumerate(datasets)}
    np.savez(root / 'catalog.tmp.npz', e_clip=e_clip, e_body=e_body, e_shard=e_shard, e_row0=e_row0, e_n=e_n, e_k0=e_k0,
             e_obj=e_obj, e_nflag=e_nflag, e_scale=e_scale, e_floor=e_floor, e_runs_ptr=ptr,
             c_betas=c_betas, c_nbetas=c_nbetas, c_up=c_up, c_gender=c_gender, c_floor_h=c_floor_h, runs=runs, c_ptr=c_ptr, c_fps=c_fps, c_stride=c_stride, c_nk=c_nk,
             c_hash=c_hash, c_dataset=np.array([di[d] for d in c_dataset], np.int16))
    os.replace(root / 'catalog.tmp.npz', root / 'catalog.npz')
    # converter-only source index
    np.savez(root / 'ingest_index.tmp.npz',
             clip=np.array([c for (c, b) in live], dtype=object).astype(str) if live else np.zeros(0, 'U1'),
             body=np.array([b for (c, b) in live], dtype=object).astype(str) if live else np.zeros(0, 'U1'),
             size=np.array([v[1][0].get('src', {}).get('size', -1) for v in live.values()], np.int64),
             mtime_ns=np.array([v[1][0].get('src', {}).get('mtime_ns', -1) for v in live.values()], np.int64),
             sha256=np.array([v[1][0].get('src', {}).get('sha256') or '' for v in live.values()], 'U64'),
             stride=np.array([v[1][0].get('stride', 0) for v in live.values()], np.int32))
    os.replace(root / 'ingest_index.tmp.npz', root / 'ingest_index.npz')
    with open(root / 'clips.tmp.jsonl', 'w') as f:
        for l in clip_lines:
            f.write(json.dumps(l) + '\n')
    os.replace(root / 'clips.tmp.jsonl', root / 'clips.jsonl')
    robots = {}
    for rel in shards:
        m = shard_meta[rel]
        if 'robot' in m:
            robots[m['robot']] = dict(nj=m['nj'], joint_names=m['joint_names'], mjcf_sha256=m['mjcf_sha256'],
                                      kin_sha256=m.get('spec_kin_sha256'))
    nbytes = {rel: man['shards'][rel]['nbytes'] for rel in shards}
    frames_body = {b: int(e_n[e_body == i].sum()) for i, b in enumerate(bodies)}
    cm = dict(format='umr_dataset_catalog', version=1, built=time.time(), bodies=bodies, clips=clips, shards=shards,
              datasets=datasets, robots=robots, shard_bytes=nbytes, total_bytes=int(sum(nbytes.values())),
              frames_per_body=frames_body, n_entries=E, n_clips=C)
    _atomic_json(root / 'catalog_meta.json', cm)
    log(f'[catalog] {C} clips, {len(bodies)} bodies, {E} entries, {len(shards)} live shards, '
        f'{cm["total_bytes"] / 1e6:.1f} MB, built in {time.time() - t0:.2f}s')
    return cm


def _rel_of(live, clip, body):
    return live[(clip, body)][0][1]


def _dataset_guess(clip, live):
    for (c, b), (_, ents) in live.items():
        if c == clip:
            sd = (ents[0].get('prov') or {}).get('source_data', '') or ''
            for k in ('hiphi', 'omnicontact', 'grail', 'lafan'):
                if k in sd.lower():
                    return 'lafan1' if k == 'lafan' else k
    return 'other'


def load_live_entries(root: Path) -> dict:
    """(clip, body) -> {'src': {...}, 'stride': int} from ingest_index.npz (converter use)."""
    p = root / 'ingest_index.npz'
    if not p.exists():
        return {}
    z = np.load(p)
    return {(c, b): dict(src=dict(size=int(s), mtime_ns=int(m), sha256=str(h)), stride=int(st))
            for c, b, s, m, h, st in zip(z['clip'], z['body'], z['size'], z['mtime_ns'], z['sha256'], z['stride'])}


class Catalog:
    """Loader-side view: arrays + names. Cheap to pickle (arrays are small)."""

    def __init__(self, root):
        self.root = Path(root)
        z = np.load(self.root / 'catalog.npz')
        for k in z.files:
            setattr(self, k, z[k])
        self.meta = json.load(open(self.root / 'catalog_meta.json'))
        self.bodies = self.meta['bodies']; self.clips = self.meta['clips']; self.shards = self.meta['shards']
        self.body_index = {b: i for i, b in enumerate(self.bodies)}

    def entries_of_clip(self, c):
        return range(int(self.c_ptr[c]), int(self.c_ptr[c + 1]))

    def runs_of(self, e):
        return self.runs[self.e_runs_ptr[e]:self.e_runs_ptr[e + 1]]


def src_rows(entry: dict, k0: int, n: int) -> np.ndarray:
    """Rows of the factory npz holding stored frames k0 .. k0+n-1 of this (segment) entry."""
    i = np.arange(k0 - entry['k0'], k0 - entry['k0'] + n)
    if 'src_rows' in entry:
        return np.asarray(entry['src_rows'])[i]
    if 'src_row0' in entry:
        return entry['src_row0'] + i * entry['src_row_step']
    return (entry['k0'] + i) * entry['stride']            # entries written before 2026-10-04 23:10 (contiguous frame_ids)
