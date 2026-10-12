"""Duty-cycle throttle + network guard (RESOURCE_LIMITS.md: GPU util < ~70 %, power < ~350 W, pause when the
network is down).  Call tick() between GPU work items: it synchronises, then sleeps so that the busy fraction since
the previous tick is <= DUTY, and blocks while /home/arenalabs/umr_factory_data/logs/throttle.log reports net=down."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os
import time

import torch

DUTY = float(os.environ.get('UMRG_DUTY', '1.0'))
LOG = f'{SRCFG.data_root}/logs/throttle.log'
_last = [time.time()]
_slept = [0.0]
_net_check = [0.0]
_net_slept = [0.0]


def set_duty(d):
    global DUTY
    DUTY = float(d)


def net_down():
    try:
        with open(LOG, 'rb') as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 300))
            last = f.read().decode(errors='ignore').strip().splitlines()[-1]
        return 'net=down' in last
    except Exception:
        return False


def tick():
    now = time.time()
    if DUTY < 1.0:
        torch.cuda.synchronize()
        now = time.time()
        busy = now - _last[0]
        s = busy * (1.0 / DUTY - 1.0)
        if s > 0:
            time.sleep(min(s, 5.0)); _slept[0] += min(s, 5.0)
    if now - _net_check[0] > 10:
        _net_check[0] = now
        while net_down():
            print('[throttle] network down -> pausing', flush=True)
            time.sleep(30); _slept[0] += 30; _net_slept[0] += 30
    _last[0] = time.time()


def slept():
    return _slept[0]


def net_slept():
    return _net_slept[0]
