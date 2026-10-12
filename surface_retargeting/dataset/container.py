"""UMRS v1 shard container: one immutable, self-describing, page-aligned file per shard.

Layout (little endian)::

    [0, 8)          magic  b"UMRS0001"
    [8, 16)         uint64 header length H
    [16, 16+H)      UTF-8 JSON header
    pad to 4096
    blob 0          (4096-aligned, C-contiguous)
    blob 1 ...

Header::

    {"format": "umrs", "version": 1, "kind": "robot"|"human", "meta": {...},
     "blobs": {name: {"dtype": <descr>, "shape": [...], "offset": int (from data start),
                      "nbytes": int, "sha256": hex}}}

``offset`` is relative to the data start ``align(16 + H)`` so the header size never depends on the offsets.
Shards are written to ``<path>.tmp`` then hard-linked into place (never overwriting), so readers never see a
partial file and an existing shard can never be replaced.
A reader maps the file once (``np.memmap`` uint8) lazily per process; pickling a Shard ships only the path
(safe for DataLoader workers with fork or spawn).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

MAGIC = b"UMRS0001"
ALIGN = 4096
VERSION = 1


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _bytes(a: np.ndarray) -> memoryview:
    """Raw bytes of a C-contiguous array (works for structured dtypes)."""
    a = np.ascontiguousarray(a)
    return memoryview(a.reshape(-1).view(np.uint8))


def dtype_to_json(dt: np.dtype):
    dt = np.dtype(dt)
    if dt.names is None:
        return dt.str
    out = []
    for name in dt.names:
        sub, _off = dt.fields[name][:2]
        if sub.subdtype is not None:
            base, shape = sub.subdtype
            out.append([name, base.str, list(shape)])
        else:
            out.append([name, sub.str, []])
    return out


def dtype_from_json(js) -> np.dtype:
    if isinstance(js, str):
        return np.dtype(js)
    return np.dtype([(n, t, tuple(s)) if s else (n, t) for n, t, s in js])


def write_shard(path, blobs: dict, meta: dict, kind: str) -> dict:
    """Write blobs (name -> np.ndarray) + meta atomically. Returns {"path", "nbytes", "sha256"} (whole file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'{path}: shards are immutable, refusing to overwrite')
    arrs = {k: np.ascontiguousarray(v) for k, v in blobs.items()}
    desc, off = {}, 0
    for k, a in arrs.items():
        desc[k] = dict(dtype=dtype_to_json(a.dtype), shape=list(a.shape), offset=off, nbytes=int(a.nbytes),
                       sha256=hashlib.sha256(_bytes(a)).hexdigest())
        off = _align(off + a.nbytes)
    header = dict(format="umrs", version=VERSION, kind=kind, meta=meta, blobs=desc)
    hb = json.dumps(header, separators=(",", ":"), default=_json_default).encode()
    data_start = _align(16 + len(hb))
    tmp = path.with_name(path.name + ".tmp")
    h = hashlib.sha256()
    with open(tmp, "wb") as f:
        def put(b):
            f.write(b); h.update(b)
        put(MAGIC); put(struct.pack("<Q", len(hb))); put(hb)
        put(b"\0" * (data_start - 16 - len(hb)))
        pos = 0
        for k, a in arrs.items():
            assert pos == desc[k]["offset"]
            put(_bytes(a))
            pos += a.nbytes
            pad = _align(pos) - pos
            if pad:
                put(b"\0" * pad); pos += pad
        f.flush(); os.fsync(f.fileno())
    os.link(tmp, path)                       # atomic and fails if the name was taken meanwhile (never overwrite)
    os.unlink(tmp)
    return dict(path=str(path), nbytes=path.stat().st_size, sha256=h.hexdigest())


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_header(path):
    with open(path, "rb") as f:
        mg = f.read(8)
        if mg != MAGIC:
            raise ValueError(f"{path}: bad magic {mg!r}")
        (hl,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hl))
    header["_data_start"] = _align(16 + hl)
    return header


class Shard:
    """Lazy, fork/spawn-safe read-only view of a UMRS file."""

    def __init__(self, path, header=None):
        self.path = str(path)
        self.header = header if header is not None else read_header(self.path)
        self._mm = None
        self._views = {}

    def __getstate__(self):
        return dict(path=self.path, header=self.header)

    def __setstate__(self, st):
        self.path, self.header = st["path"], st["header"]
        self._mm, self._views = None, {}

    @property
    def meta(self):
        return self.header["meta"]

    @property
    def kind(self):
        return self.header["kind"]

    def has(self, name):
        return name in self.header["blobs"]

    def blob(self, name) -> np.ndarray:
        v = self._views.get(name)
        if v is None:
            if self._mm is None:
                self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
            d = self.header["blobs"][name]
            s = self.header["_data_start"] + d["offset"]
            dt = dtype_from_json(d["dtype"])
            v = self._mm[s:s + d["nbytes"]].view(dt).reshape(d["shape"])
            self._views[name] = v
        return v

    def verify(self) -> dict:
        """Recompute every blob sha256 against the header. Returns {name: ok}."""
        out = {}
        for k, d in self.header["blobs"].items():
            out[k] = hashlib.sha256(_bytes(self.blob(k))).hexdigest() == d["sha256"]
        return out


def file_sha256(path, bufsize=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
