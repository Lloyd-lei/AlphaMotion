"""GRAIL selective downloader (Stream B).

Downloads recon/meta/object_usd (no textures, no video/robot/objects/bps) for the categories in the given order,
clip by clip, into raw/grail (HF local_dir layout data/<cat>/<sub>/<file>). Pinned to the revision recorded in
raw/grail/_tree.json. Skips clips without both a recon pkl and a USD. Resumable (skips files with matching size).
Per-clip completion is visible as: recon + usd present with the listed size.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault('HF_HOME', '/home/arenalabs/umr_factory_data/hf_cache/grail')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = '/home/arenalabs/umr_factory_data/raw/grail'
REPO = 'nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'
ap = argparse.ArgumentParser()
ap.add_argument('--cats', default='pickup_table,pickup_ground,sitting,stair_p1,stair_p2,slope,curb')
ap.add_argument('--limit', type=int, default=0, help='max clips per category (0 = all)')
ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--interleave', type=int, default=0, help='first N clips of every category before the rest')
ap.add_argument('--tag', default='dl')
args = ap.parse_args()
tree = json.load(open(f'{ROOT}/_tree.json'))
SHA = tree['_sha']
token = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
cats = args.cats.split(',')
jobs = {}
for c in cats:
    rec = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/recon']['files']}
    usd = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/object_usd']['files'] if p.endswith('.usd')}
    meta = {os.path.basename(p)[:-4]: (p, s) for p, s in tree[f'{c}/meta']['files']}
    keys = sorted(k for k in rec if k in usd)
    if args.limit:
        keys = keys[:args.limit]
    jobs[c] = [[rec[k], usd[k]] + ([meta[k]] if k in meta else []) for k in keys]
    print(c, 'clips with recon+usd', len(keys), 'recon without usd', len(set(rec) - set(usd)),
          'usd without recon', len(set(usd) - set(rec)), flush=True)
order = []
if args.interleave:
    for c in cats:
        order += jobs[c][:args.interleave]
    for c in cats:
        order += jobs[c][args.interleave:]
else:
    for c in cats:
        order += jobs[c]
files = [f for clip in order for f in clip]
tot = sum(s for _, s in files)
print('files', len(files), 'GB', round(tot / 1e9, 2), flush=True)
done_b = 0
done_n = 0
fail = []
t0 = time.time()


def get(p, s):
    dst = f'{ROOT}/{p}'
    if os.path.exists(dst) and os.path.getsize(dst) == s:
        return s, 'skip'
    err = ''
    for att in range(8):
        try:
            hf_hub_download(REPO, p, repo_type='dataset', revision=SHA, local_dir=ROOT, token=token)
            return s, 'ok'
        except Exception as e:  # retry with backoff (429 etc.)
            err = repr(e)[:300]
            time.sleep(min(60, 2 ** att))
    raise RuntimeError(f'{p}: {err}')


# bounded submission keeps the queue in priority order (as_completed over a full submit is also ordered by the
# executor's FIFO, so a plain submit-all is fine)
with ThreadPoolExecutor(args.workers) as ex:
    futs = {ex.submit(get, p, s): p for p, s in files}
    for f in as_completed(futs):
        try:
            s, st = f.result()
            done_b += s
        except Exception as e:
            fail.append(str(e))
            print('FAIL', e, flush=True)
        done_n += 1
        if done_n % 300 == 0 or done_n == len(files) or done_n in (10, 50, 100):
            dt = time.time() - t0
            print(f'{time.strftime("%H:%M:%S")} {done_n}/{len(files)} {done_b/1e9:.2f}/{tot/1e9:.2f} GB '
                  f'{done_b/1e6/dt:.2f} MB/s {dt/60:.1f} min fails={len(fail)}', flush=True)
json.dump(fail, open(f'{ROOT}/_dl_failures_{args.tag}.json', 'w'))
print('DONE fails', len(fail), flush=True)
