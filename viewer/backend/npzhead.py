"""Cheap, read-only NPZ inspection helpers (no full decompression, no pickle).

* ``member_header``: shape/dtype of one member, decompressing only the first
  bytes of its stream (enough for the .npy header).
* ``small_member``: load a small numeric/string member fully (size-capped).
* ``stored_memmap``: memory-map an uncompressed (ZIP_STORED) member in place,
  so large captured arrays are paged in lazily instead of being expanded.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

MAX_SMALL = 64 * 1024 * 1024


def _read_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    return tuple(int(x) for x in shape), bool(fortran), dtype


def member_names(path):
    with zipfile.ZipFile(path) as z:
        return [i.filename[:-4] for i in z.infolist() if i.filename.endswith('.npy')]


def member_header(path, name, zf=None):
    """Return (shape, dtype) of ``name`` (without .npy) in an npz."""
    own = zf is None
    z = zipfile.ZipFile(path) if own else zf
    try:
        with z.open(name + '.npy') as f:
            shape, _, dtype = _read_header(f)
        return shape, dtype
    finally:
        if own:
            z.close()


def small_member(path, name, zf=None, max_bytes=MAX_SMALL):
    """Load one member fully; refuses object arrays and oversize members."""
    own = zf is None
    z = zipfile.ZipFile(path) if own else zf
    try:
        info = z.getinfo(name + '.npy')
        if info.file_size > max_bytes:
            raise ValueError(f'{name} 超过 {max_bytes} 字节读取上限')
        with z.open(info) as f:
            shape, fortran, dtype = _read_header(f)
            if dtype.hasobject:
                raise ValueError(f'{name} 是 object 数组，拒绝反序列化')
            data = f.read()
        a = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape, dtype=np.int64)))
        return a.reshape(shape, order='F' if fortran else 'C').copy()
    finally:
        if own:
            z.close()


def stored_memmap(path, name):
    """Memory-map an uncompressed npz member directly from the archive file."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f'{name} 是压缩成员，不能原地 mmap')
        offset = info.header_offset
    with open(path, 'rb') as f:
        f.seek(offset)
        local = f.read(30)
        if local[:4] != b'PK\x03\x04':
            raise ValueError('ZIP 本地文件头无效')
        name_len, extra_len = struct.unpack('<HH', local[26:30])
        data_start = offset + 30 + name_len + extra_len
        f.seek(data_start)
        shape, fortran, dtype = _read_header(f)
        if dtype.hasobject:
            raise ValueError('object 数组不能 mmap')
        array_start = f.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=array_start, shape=shape,
                     order='F' if fortran else 'C')
