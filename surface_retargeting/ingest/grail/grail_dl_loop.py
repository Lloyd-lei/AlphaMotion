"""GRAIL resumable downloader driver (Stream B, restart after the 22:15 DNS outage).

Phases (semicolon separated) are downloaded in order; inside a phase files are queued round-robin by clip across
categories so every category progresses. Only recon/*.pkl + object_usd/*.usd + meta/*.pkl of clips that have BOTH a
recon and a USD (no textures/videos/robot/objects/bps). Pinned to the revision in raw/grail/_tree.json.
aria2c reads the list (incl. the auth header) from stdin, so the token never hits disk or argv.
Each aria2c round is bounded (--round-min); on network failure the driver sleeps and re-lists. Lifetime --max-hours.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = '/home/arenalabs/umr_factory_data/raw/grail'
REPO_URL = 'https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/resolve'

ap = argparse.ArgumentParser()
ap.add_argument('--phases', default='pickup_ground,sitting,slope,curb;pickup_table,stair_p1,stair_p2')
ap.add_argument('--conns', type=int, default=48)
ap.add_argument('--max-hours', type=float, default=10.0)
ap.add_argument('--round-min', type=float, default=40.0, help='restart aria2c every N minutes (re-list, re-prioritise)')
ap.add_argument('--max-list', type=int, default=6000, help='files per aria2c round')
a = ap.parse_args()

tree = json.load(open(f'{ROOT}/_tree.json'))
SHA = tree['_sha']
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()


def clips(c):
    rec = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/recon']['files']}
    usd = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/object_usd']['files'] if p.endswith('.usd')}
    meta = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/meta']['files']}
    keys = sorted(k for k in rec if k in usd)
    return [[rec[k], usd[k]] + ([meta[k]] if k in meta else []) for k in keys]


def done(p, s):
    dst = f'{ROOT}/{p}'
    return os.path.exists(dst) and os.path.getsize(dst) == s and not os.path.exists(dst + '.aria2')


def todo_list():
    out = []
    for ph in a.phases.split(';'):
        cats = ph.split(',')
        per = {c: clips(c) for c in cats}
        i = 0
        while any(i < len(v) for v in per.values()):
            for c in cats:
                if i < len(per[c]):
                    out += [(p, s) for p, s in per[c][i] if not done(p, s)]
            i += 1
    return out


t_end = time.time() + a.max_hours * 3600
rnd = 0
while time.time() < t_end:
    todo = todo_list()
    if not todo:
        print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
        open(f'{ROOT}/_dl_complete', 'w').write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        break
    batch = todo[:a.max_list]
    gb = sum(s for _, s in batch) / 1e9
    rem = sum(s for _, s in todo) / 1e9
    rnd += 1
    print(f'{time.strftime("%H:%M:%S")} round {rnd}: remaining files {len(todo)} {rem:.2f} GB; this round {len(batch)} files {gb:.2f} GB', flush=True)
    lines = []
    for p, s in batch:
        dst = f'{ROOT}/{p}'
        lines.append(f'{REPO_URL}/{SHA}/{p}\n  header=Authorization: Bearer {tok}\n  dir={os.path.dirname(dst)}\n  out={os.path.basename(dst)}\n')
    budget = max(60, min(a.round_min * 60, t_end - time.time()))
    cmd = ['timeout', '--signal=INT', '--kill-after=60', str(int(budget)), 'aria2c', '-i', '-',
           f'-j{a.conns}', '-x1', '-s1', '--continue=true', '--auto-file-renaming=false', '--allow-overwrite=true',
           '--file-allocation=none', '--max-tries=8', '--retry-wait=5', '--timeout=60', '--connect-timeout=20',
           '--console-log-level=error', '--summary-interval=0', '--download-result=hide', '--show-console-readout=false',
           '--async-dns=false', '--lowest-speed-limit=1K']
    t0 = time.time()
    b0 = sum(os.path.getsize(f'{ROOT}/{p}') for p, s in batch if os.path.exists(f'{ROOT}/{p}'))
    pr = subprocess.run(cmd, input=''.join(lines), text=True, stdout=sys.stdout, stderr=sys.stdout)
    dt = time.time() - t0
    b1 = sum(os.path.getsize(f'{ROOT}/{p}') for p, s in batch if os.path.exists(f'{ROOT}/{p}'))
    nd = sum(done(p, s) for p, s in batch)
    print(f'{time.strftime("%H:%M:%S")} round {rnd} rc={pr.returncode} {dt/60:.1f} min, +{(b1-b0)/1e9:.2f} GB '
          f'({(b1-b0)/1e6/max(dt,1):.2f} MB/s), files done {nd}/{len(batch)}', flush=True)
    if pr.returncode not in (0, 124) and (b1 - b0) < 1e6:
        time.sleep(60)  # network down: back off then re-list
print(time.strftime('%H:%M:%S'), 'driver exit', flush=True)
