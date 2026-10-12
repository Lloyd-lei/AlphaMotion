"""Versioned binary wire format. Positions are float32, never float16."""
from __future__ import annotations

import json
import struct
import numpy as np

DTYPES = {'float32': np.dtype('<f4'), 'uint32': np.dtype('<u4'),
          'int32': np.dtype('<i4'), 'uint8': np.dtype('u1')}


def pack(meta, arrays):
    descriptors, payload = {}, bytearray()
    for name, value in arrays.items():
        a = np.asarray(value)
        if a.dtype.kind == 'f':
            dtype = 'float32'
        elif a.dtype.kind == 'b' or a.dtype == np.dtype('uint8'):
            dtype = 'uint8'
        elif a.dtype.kind == 'u':
            dtype = 'uint32'
        elif a.dtype.kind == 'i':
            dtype = 'int32'
        else:
            raise ValueError(f'unsupported binary dtype: {name}: {a.dtype}')
        a = np.ascontiguousarray(a, dtype=DTYPES[dtype])
        payload.extend(b'\0' * ((-len(payload)) % 4))
        descriptors[name] = {'dtype': dtype, 'shape': list(a.shape), 'offset': len(payload), 'nbytes': a.nbytes}
        payload.extend(a.tobytes())
    header = json.dumps({'schema': 'motion-audit-binary-v1', 'meta': meta, 'arrays': descriptors},
                        ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    return struct.pack('<I', len(header)) + header + b'\0' * ((-len(header)) % 4) + payload


def unpack(data):
    if len(data) < 4:
        raise ValueError('truncated binary header')
    n = struct.unpack_from('<I', data)[0]
    if n > len(data) - 4:
        raise ValueError('truncated JSON header')
    h = json.loads(data[4:4+n])
    if h.get('schema') != 'motion-audit-binary-v1':
        raise ValueError('unsupported binary schema')
    base = 4 + (n + 3) // 4 * 4
    arrays = {}
    for name, desc in h['arrays'].items():
        dtype = DTYPES[desc['dtype']]
        count = int(np.prod(desc['shape'], dtype=np.int64))
        offset = base + desc['offset']
        if offset < base or offset + count * dtype.itemsize > len(data):
            raise ValueError('truncated array payload')
        arrays[name] = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(desc['shape'])
    return h['meta'], arrays
