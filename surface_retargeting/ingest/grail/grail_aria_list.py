"""Emit an aria2c input list (stdout) for GRAIL recon/meta/object_usd (no textures/videos/robot/objects),
round-robin across categories by clip so every category progresses; files already present with the
listed size are skipped. The auth header is only written to the pipe (never to disk/argv)."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import json, os, sys
ROOT = f'{SRCFG.data_root}/raw/grail'
tree = json.load(open(f'{ROOT}/_tree.json'))
SHA = tree['_sha']
cats = sys.argv[1].split(',') if len(sys.argv) > 1 else ['pickup_ground', 'sitting', 'slope', 'curb', 'pickup_table', 'stair_p1', 'stair_p2']
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
per = {}
for c in cats:
    rec = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/recon']['files']}
    usd = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/object_usd']['files'] if p.endswith('.usd')}
    meta = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/meta']['files']}
    keys = sorted(k for k in rec if k in usd)
    per[c] = [[rec[k], usd[k]] + ([meta[k]] if k in meta else []) for k in keys]
n = 0; b = 0
i = 0
while any(i < len(v) for v in per.values()):
    for c in cats:
        if i >= len(per[c]):
            continue
        for p, s in per[c][i]:
            dst = f'{ROOT}/{p}'
            if os.path.exists(dst) and os.path.getsize(dst) == s and not os.path.exists(dst + '.aria2'):
                continue
            sys.stdout.write(f'https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/resolve/{SHA}/{p}\n'
                             f'  header=Authorization: Bearer {tok}\n  dir={os.path.dirname(dst)}\n  out={os.path.basename(dst)}\n')
            n += 1; b += s
    i += 1
sys.stderr.write(f'queued files {n} GB {b/1e9:.2f}\n')
