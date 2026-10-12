"""GPU lease on ROOT/gpu_queue.lock (flock LOCK_EX|LOCK_NB), never recreating the file.

Protocol (same as every existing owner in the tree, plus three safety checks):
  1. pre-check /proc/locks for any holder of the lock file's (device, inode); refuse if held;
  2. open the EXISTING file without O_CREAT (a missing file is an error, never recreated) and
     verify its inode is the expected one (55451429 for the real lock);
  3. flock(LOCK_EX|LOCK_NB): a second taker gets BlockingIOError immediately.
The lock lives as long as the open file description. A queue owner can hand the lease to a
child job by passing the fd (subprocess pass_fds); the child calls ``GpuLease.inherit(fd)``,
which re-asserts LOCK_EX on the same open file description (a no-op when inherited, never a
release) and verifies the inode.  The child never unlocks: that would release the parent too.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if _sys.path and _sys.path[0] and _Path(_sys.path[0]).resolve() == _Path(__file__).resolve().parent:
    _sys.path.pop(0)  # running ops/<x>.py as a script: ops/queue.py must not shadow the stdlib 'queue'
import errno
import fcntl
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from am2exp_v1.common.paths import GPU_LOCK, GPU_LOCK_INODE  # noqa: E402


class LeaseError(RuntimeError):
    pass


def _dev_inode(path):
    st = os.stat(path)
    return os.major(st.st_dev), os.minor(st.st_dev), st.st_ino


def lock_holders(path=GPU_LOCK):
    """Entries of /proc/locks on this file's (device, inode). Waiters ('->' lines) included."""
    major, minor, ino = _dev_inode(path)
    out = []
    with open('/proc/locks') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            waiter = len(parts) > 1 and parts[1] == '->'
            if waiter:
                parts = [parts[0]] + parts[2:]
            # id: TYPE ADVISORY|MANDATORY MODE PID MAJ:MIN:INODE START END
            if len(parts) < 6:
                continue
            try:
                maj, mnr, inode = parts[5].split(':')
                maj, mnr, inode = int(maj, 16), int(mnr, 16), int(inode)
            except ValueError:
                continue
            if (maj, mnr, inode) == (major, minor, ino):
                out.append({'type': parts[1], 'mode': parts[3], 'pid': int(parts[4]) if parts[4].isdigit() else parts[4],
                            'waiter': waiter, 'raw': line.strip()})
    return out


def precheck(path=GPU_LOCK, expected_inode=GPU_LOCK_INODE):
    """Read-only status of the lease. Never opens the file for writing, never creates it."""
    path = Path(path)
    if not path.exists():
        raise LeaseError(f'lock file missing: {path} (never recreate it; flock is bound to the inode)')
    major, minor, ino = _dev_inode(path)
    if expected_inode is not None and ino != expected_inode:
        raise LeaseError(f'lock file inode {ino} != expected {expected_inode}; it was recreated, refuse')
    holders = [h for h in lock_holders(path) if not h['waiter']]
    return {'path': str(path), 'device': f'{major}:{minor}', 'inode': ino, 'expected_inode': expected_inode,
            'held': bool(holders), 'holders': holders, 'checked_unix': time.time()}


class GpuLease:
    """Exclusive non-blocking lease. Use as a context manager or acquire()/release()."""

    def __init__(self, path=GPU_LOCK, expected_inode=GPU_LOCK_INODE):
        self.path = Path(path)
        self.expected_inode = expected_inode
        self.fd = None
        self.inherited = False
        self.receipt = None

    # -- owner side --------------------------------------------------------------------------
    def acquire(self):
        if self.fd is not None:
            raise LeaseError('lease already held by this object')
        pre = precheck(self.path, self.expected_inode)
        if pre['held']:
            pids = [h['pid'] for h in pre['holders']]
            raise BlockingIOError(errno.EWOULDBLOCK, f'GPU lease held by pid(s) {pids} (pre-check /proc/locks)')
        fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)  # no O_CREAT: never recreate
        try:
            if os.fstat(fd).st_ino != pre['inode']:
                raise LeaseError('lock file replaced between pre-check and open')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # BlockingIOError if someone races us
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        self.receipt = {**pre, 'acquired_unix': time.time(), 'owner_pid': os.getpid(), 'inherited': False}
        return self

    def release(self):
        if self.fd is not None and not self.inherited:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
        self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False

    def handoff_fd(self):
        """fd to pass to a child (subprocess pass_fds); the child must call GpuLease.inherit(fd)."""
        if self.fd is None:
            raise LeaseError('no lease held')
        return self.fd

    # -- child side --------------------------------------------------------------------------
    @classmethod
    def inherit(cls, fd, path=GPU_LOCK, expected_inode=GPU_LOCK_INODE):
        fd = int(fd)
        st = os.fstat(fd)
        ino = os.stat(path).st_ino
        if st.st_ino != ino or (expected_inode is not None and ino != expected_inode):
            raise LeaseError(f'inherited fd {fd} is not the GPU lock (inode {st.st_ino} vs {ino})')
        # Same open file description: LOCK_EX succeeds as a no-op if the parent holds it; if the
        # lease was free it is taken now; if another process holds it -> BlockingIOError.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = cls(path, expected_inode)
        lease.fd = fd
        lease.inherited = True
        lease.receipt = {'path': str(path), 'inode': ino, 'inherited_fd': fd, 'owner_pid': os.getppid(),
                         'child_pid': os.getpid(), 'inherited': True, 'verified_unix': time.time(),
                         'holders': lock_holders(path)}
        return lease


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Read-only GPU lease status (never takes the lock).')
    ap.add_argument('--path', default=str(GPU_LOCK))
    args = ap.parse_args()
    print(json.dumps(precheck(args.path, GPU_LOCK_INODE if args.path == str(GPU_LOCK) else None), indent=2))


if __name__ == '__main__':
    main()
