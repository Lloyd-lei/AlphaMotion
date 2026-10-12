#!/usr/bin/env python3
"""Start/status/stop this service only, with Linux boot/start/argv identity checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / 'logs' / 'service.json'


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'argv': (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                'cwd': os.readlink(proc / 'cwd')}
    except (OSError, ValueError, IndexError):
        return None


def saved():
    if not PID_FILE.exists():
        return None
    value = json.loads(PID_FILE.read_text())
    return value


def matches(record):
    return bool(record and identity(record['identity']['pid']) == record['identity'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('start', 'status', 'stop'))
    parser.add_argument('--mode', choices=('production', 'test'), default='production')
    args = parser.parse_args()
    (ROOT / 'logs').mkdir(exist_ok=True)
    with (ROOT / 'logs' / 'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = saved()
        alive = matches(record)
        if args.command == 'status':
            print(json.dumps({'running': alive, 'record': record}, ensure_ascii=False))
            return
        if args.command == 'stop':
            if not alive:
                print('No matching owned process; no signal sent.')
                return
            os.kill(record['identity']['pid'], signal.SIGTERM)
            for _ in range(100):
                if not matches(record):
                    PID_FILE.unlink(missing_ok=True)
                    print('Owned service stopped.')
                    return
                time.sleep(.1)
            raise SystemExit('Owned service is shutting down; inspect logs before another action.')
        if alive:
            if record['mode'] != args.mode:
                raise SystemExit('Owned service already running in another mode. Explicitly stop it first.')
            print(json.dumps({'already_running': True, 'record': record}))
            return
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', 8765))
            except OSError:
                raise SystemExit('Port 8765 is occupied; no process touched or started.')
        python = ROOT / '.venv' / 'bin' / 'python'
        argv = [str(python), str(ROOT / 'server.py'), '--mode', args.mode]
        with (ROOT / 'logs' / 'server.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=log,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(40):
            current = identity(process.pid)
            if current and current['argv'] == argv:
                break
            if process.poll() is not None:
                raise SystemExit('Service exited; inspect logs/server.log.')
            time.sleep(.1)
        else:
            raise SystemExit('Could not confirm launched process identity; inspect logs.')
        record = {'identity': current, 'mode': args.mode, 'bind': '127.0.0.1:8765',
                  'started_at_unix': time.time(), 'idle_timeout_seconds': 28800}
        temp = PID_FILE.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(PID_FILE)
        for _ in range(100):
            if process.poll() is not None:
                raise SystemExit('Service exited; inspect logs/server.log.')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=1) as response:
                    health = json.load(response)
                print(json.dumps({'started': record, 'health': health}, ensure_ascii=False))
                return
            except OSError:
                time.sleep(.1)
        raise SystemExit('Process started; HTTP startup not confirmed. Inspect service status and log.')


if __name__ == '__main__':
    main()
