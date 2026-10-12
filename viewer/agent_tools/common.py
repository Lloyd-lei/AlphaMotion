from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


def digest(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed while hashing: {path}')
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def inside(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Path must remain inside the named evidence root')
    return p


def checked_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', value):
        raise ValueError('Invalid identifier')
    return value


@contextlib.contextmanager
def file_lock(path, blocking=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'argv': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                'cwd': os.readlink(p / 'cwd')}
    except (OSError, IndexError):
        return None


def source_stamp(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def now():
    return time.time()
