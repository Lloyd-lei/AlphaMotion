"""Process receipts for every am2exp process: identity, state.json, exit_status.json, RSS, free -g.

identity  = pid + start_ticks (/proc/<pid>/stat field 22) + boot_id + argv (+cwd, host, exe).
            "Never restart from a PID alone": (pid, start_ticks, boot_id) is the identity.
state.json        rewritten atomically on every phase/heartbeat update.
exit_status.json  ALWAYS written in ``finally`` (completed / failed / terminated / interrupted),
                  including SIGTERM (converted to SystemExit so finally blocks run).
RssSampler        background thread sampling own VmRSS/VmHWM (+ optional children) every N s;
                  ``exceeded`` flips when RSS passes the limit (20 GiB default; swap is 0 here).
"""
from __future__ import annotations
import datetime
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

GiB = 1024 ** 3
DEFAULT_RSS_LIMIT = 20 * GiB


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    """Atomic JSON write (tmp + replace), NaN rejected like the sealed write_json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n')
    tmp.replace(path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def start_ticks(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    return stat.rsplit(')', 1)[1].split()[19]  # field 22 overall (starttime)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    p = Path('/proc') / str(pid)
    raw = (p / 'cmdline').read_bytes()
    out = {'pid': pid, 'start_ticks': start_ticks(pid), 'boot_id': boot_id(),
           'argv': [x.decode(errors='replace') for x in raw.split(b'\0') if x],
           'hostname': platform.node(), 'utc': utc()}
    try:
        out['cwd'] = os.readlink(p / 'cwd')
        out['exe'] = os.readlink(p / 'exe')
        out['ppid'] = int((p / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    except OSError:
        pass
    return out


def same_process(record):
    """True iff the recorded identity is still the live process (pid AND start_ticks AND boot)."""
    try:
        return (start_ticks(record['pid']) == str(record['start_ticks']) and boot_id() == record['boot_id'])
    except (FileNotFoundError, ProcessLookupError, KeyError, IndexError):
        return False


def _status_kb(pid, key):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(key + ':'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def rss_bytes(pid=None):
    return _status_kb(os.getpid() if pid is None else pid, 'VmRSS')


def free_snapshot():
    """`free -b` parsed + `free -g` text (the human-readable receipt lloyd asked for)."""
    out = {'utc': utc()}
    try:
        text_g = subprocess.check_output(['free', '-g'], text=True, timeout=10)
        text_b = subprocess.check_output(['free', '-b'], text=True, timeout=10)
        out['free_g_text'] = text_g
        for line in text_b.splitlines():
            parts = line.split()
            if parts and parts[0] in ('Mem:', 'Swap:'):
                keys = ['total', 'used', 'free', 'shared', 'buff_cache', 'available'] if parts[0] == 'Mem:' else ['total', 'used', 'free']
                out[parts[0][:-1].lower()] = {k: int(v) for k, v in zip(keys, parts[1:])}
    except Exception as exc:  # receipts must never kill a run
        out['error'] = repr(exc)
    return out


class RssSampler(threading.Thread):
    def __init__(self, interval=5.0, limit_bytes=DEFAULT_RSS_LIMIT, pids_fn=None, log_path=None):
        super().__init__(daemon=True, name='am2exp-rss')
        self.interval = float(interval)
        self.limit_bytes = int(limit_bytes) if limit_bytes else None
        self.pids_fn = pids_fn or (lambda: [os.getpid()])
        self.log_path = Path(log_path) if log_path else None
        self.peak = 0
        self.last = 0
        self.samples = 0
        self.exceeded = False
        self._stop_evt = threading.Event()

    def sample(self):
        total = 0
        for pid in self.pids_fn():
            v = rss_bytes(pid)
            total += v or 0
        self.last = total
        self.peak = max(self.peak, total)
        self.samples += 1
        if self.limit_bytes and total > self.limit_bytes:
            self.exceeded = True
        if self.log_path is not None:
            with self.log_path.open('a') as f:
                f.write(json.dumps({'t': time.time(), 'rss': total}) + '\n')
        return total

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.sample()
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()

    def report(self):
        return {'rss_last_bytes': self.last, 'rss_peak_bytes': self.peak, 'rss_peak_gib': round(self.peak / GiB, 3),
                'vmhwm_bytes': _status_kb(os.getpid(), 'VmHWM'), 'samples': self.samples,
                'limit_bytes': self.limit_bytes, 'exceeded': self.exceeded}


class Terminated(SystemExit):
    pass


def install_sigterm():
    """SIGTERM -> SystemExit(143) so ``finally`` writes exit_status.json. Returns old handler."""
    def handler(signum, _frame):
        raise Terminated(128 + signum)
    old = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handler)
    return old


class ProcessReceipts:
    """Context manager owning a folder's identity/state/exit receipts.

    with ProcessReceipts(folder, kind='train_arm', extra={...}) as rec:
        rec.update(phase='training', step=10)
    exit_status.json is written in __exit__ (a finally) whatever happens.
    """

    def __init__(self, folder, kind, extra=None, rss_interval=5.0, rss_limit=DEFAULT_RSS_LIMIT, pids_fn=None,
                 sigterm=True):
        self.folder = Path(folder)
        self.kind = kind
        self.extra = dict(extra or {})
        self.state = {}
        self.started = time.monotonic()
        self.sampler = RssSampler(rss_interval, rss_limit, pids_fn) if rss_interval else None
        self.sigterm = sigterm
        self._old_handler = None
        self.status = 'failed'
        self.error = None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.identity = identity()
        write_json(self.folder / 'process_identity.json', {**self.identity, 'kind': self.kind, **self.extra})
        write_json(self.folder / 'free_start.json', free_snapshot())
        if self.sigterm:
            self._old_handler = install_sigterm()
        if self.sampler:
            self.sampler.start()
        self.update(status='running', phase='start')
        return self

    def update(self, **fields):
        self.state.update(fields)
        rec = {'utc': utc(), 'kind': self.kind, 'identity': self.identity, **self.extra, **self.state,
               'elapsed_wall_seconds': time.monotonic() - self.started}
        if self.sampler:
            rec['rss'] = self.sampler.report()
        write_json(self.folder / 'state.json', rec)
        return rec

    def mark_completed(self):
        self.status = 'completed'

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            status = self.status if self.status != 'failed' else 'completed'
        elif issubclass(exc_type, Terminated):
            status = 'terminated'
        elif issubclass(exc_type, KeyboardInterrupt):
            status = 'interrupted'
        else:
            status = 'failed'
        error = None
        if exc_type is not None:
            error = {'type': exc_type.__name__, 'error': repr(exc), 'traceback': ''.join(traceback.format_exception(exc_type, exc, tb))}
        try:
            if self.sampler:
                self.sampler.stop()
            self.update(status=status, phase=self.state.get('phase'))
            write_json(self.folder / 'exit_status.json', {
                'utc': utc(), 'status': status, 'kind': self.kind, 'phase': self.state.get('phase'),
                'step': self.state.get('step'), 'identity': self.identity, 'error': error,
                'rss': self.sampler.report() if self.sampler else None, 'free_end': free_snapshot(),
                'elapsed_wall_seconds': time.monotonic() - self.started, **self.extra})
        finally:
            if self.sigterm and self._old_handler is not None and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, self._old_handler)
        return False
