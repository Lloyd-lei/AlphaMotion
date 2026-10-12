"""Exact on-disk size of the UMRS format per robot-hour and for the planned corpus, per tier.

    python scripts/storage_estimate.py --root data

Record sizes come from schema.robot_dtype / human_dtype with each staged robot's real joint count; per-entry header
overhead is measured on the existing shards.  Factory npz sizes are measured on /home/arenalabs/umr_factory_data/out.
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from surface_retargeting.dataset import schema as S                               # noqa: E402
from surface_retargeting.dataset.container import read_header                    # noqa: E402

# corpus plan (hours, fraction with objects, stored fps). HiPHI numbers from raw/hiphi/metadata/hiphi_metadata.csv.
CORPUS = {
    'hiphi_all_617h (incl. mirrors)': dict(hours=617.5, hoi_hours=245.7, fps=30.0003, clip_s=63.1),
    'hiphi_originals_309h (factory excludes __mirror)': dict(hours=308.7, hoi_hours=122.8, fps=30.0003, clip_s=63.1),
    'grail_62h (25 Hz, kept native, all with objects)': dict(hours=62.0, hoi_hours=62.0, fps=25.0, clip_s=30.0),
    'omnicontact_10h (90 -> 30 Hz, all with objects)': dict(hours=10.0, hoi_hours=10.0, fps=30.0003, clip_s=3.9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(f'{SRCFG.dataset_root}'))
    ap.add_argument('--robots-staged', type=Path, default=Path(f'{SRCFG.data_root}/robots'))
    a = ap.parse_args()
    specs = {p.parent.name: json.load(open(p)) for p in sorted((a.root / 'robots').glob('*/spec.json'))}
    assert len(specs) == 21, len(specs)
    # measured per-entry header bytes
    hb, ne = 0, 0
    for p in glob.glob(str(a.root / '**/*.umrs'), recursive=True):
        h = read_header(p)
        hb += sum(len(json.dumps(e, separators=(',', ':'))) for e in h['meta']['entries']); ne += len(h['meta']['entries'])
    hdr_per_entry = hb / max(ne, 1) * 2          # x2: shard JSON header + the catalog cache pickle
    rows = []
    out = dict(per_robot_hour_MB={}, corpus={}, header_bytes_per_entry=round(hdr_per_entry))
    print(f'measured header bytes per entry: {hdr_per_entry:.0f}')
    print(f"{'robot':20s} nj  " + '  '.join(f'{t:>14s}' for t in ('core', 'core+obj', 'standard', 'std+obj', 'full', 'full+obj')) + '   (bytes/frame)')
    frame_bytes = {}
    for r, sp in specs.items():
        fb = {}
        for t in S.TIERS:
            for ho in (False, True):
                b = S.robot_dtype(sp['nj'], t, ho).itemsize
                if t == 'full':
                    b += 512 + (512 if ho else 0)                  # per-point packbits blobs
                fb[(t, ho)] = b
        frame_bytes[r] = fb
        print(f"{r:20s} {sp['nj']:2d}  " + '  '.join(f'{fb[(t, ho)]:14d}' for t in S.TIERS for ho in (False, True)))
    hum = {('body', False): S.human_dtype(False, 0).itemsize, ('hands', False): S.human_dtype(True, 0).itemsize,
           ('body', True): S.human_dtype(False, 1).itemsize, ('hands', True): S.human_dtype(True, 1).itemsize}
    print('human record bytes/frame:', {f'{k[0]}{"+obj" if k[1] else ""}': v for k, v in hum.items()})
    tot = {t: {ho: sum(fb[(t, ho)] for fb in frame_bytes.values()) for ho in (False, True)} for t in S.TIERS}
    print('sum over 21 robots, bytes per source frame:', {t: tot[t] for t in S.TIERS})
    for t in S.TIERS:
        out['per_robot_hour_MB'][t] = {r: round(fb[(t, False)] * 30 * 3600 / 1e6, 2) for r, fb in frame_bytes.items()}
        out['per_robot_hour_MB'][t + '+obj'] = {r: round(fb[(t, True)] * 30 * 3600 / 1e6, 2) for r, fb in frame_bytes.items()}
    mean = {t: np.mean(list(out['per_robot_hour_MB'][t].values())) for t in out['per_robot_hour_MB']}
    print('mean MB per robot-hour at 30 fps:', {k: round(v, 2) for k, v in mean.items()})
    # factory npz for comparison
    sizes = []
    for f in glob.glob(f'{SRCFG.data_root}/out/*/*.npz'):
        z = np.load(f)
        T = len(z['qpos']); fps = float(z['fps'])
        sizes.append(Path(f).stat().st_size / (T / fps / 3600) / 1e6)
    if sizes:
        print(f'factory npz (compressed, native fps) MB per robot-hour: {np.min(sizes):.0f}-{np.max(sizes):.0f}')
        out['factory_npz_MB_per_robot_hour'] = [round(float(np.min(sizes))), round(float(np.max(sizes)))]
    # corpus
    print('\ncorpus totals (GB = 1e9 B), 21 robots + human, incl. headers:')
    for name, c in CORPUS.items():
        fr_all = c['hours'] * 3600 * c['fps']; fr_hoi = c['hoi_hours'] * 3600 * c['fps']; fr_body = fr_all - fr_hoi
        nclips = c['hours'] * 3600 / c['clip_s']
        res = {}
        for t in S.TIERS:
            rob = fr_body * tot[t][False] + fr_hoi * tot[t][True]
            hu = fr_body * hum[('hands', False)] + fr_hoi * hum[('hands', True)]           # upper bound: hands stored
            hdr = nclips * 22 * hdr_per_entry
            res[t] = round((rob + hu + hdr) / 1e9, 1)
        res['human_only'] = round((fr_body * hum[('hands', False)] + fr_hoi * hum[('hands', True)]) / 1e9, 2)
        res['frames_stored_M'] = round(fr_all / 1e6, 1)
        out['corpus'][name] = res
        print(f'  {name:52s} ' + '  '.join(f'{k}={v}' for k, v in res.items()))
    plan_a = ['hiphi_all_617h (incl. mirrors)', 'grail_62h (25 Hz, kept native, all with objects)', 'omnicontact_10h (90 -> 30 Hz, all with objects)']
    plan_b = ['hiphi_originals_309h (factory excludes __mirror)'] + plan_a[1:]
    for nm, plan in (('617h HiPHI + GRAIL + OmniContact', plan_a), ('309h HiPHI originals + GRAIL + OmniContact', plan_b)):
        s = {t: round(sum(out['corpus'][p][t] for p in plan), 1) for t in S.TIERS}
        out['corpus']['TOTAL ' + nm] = s
        print(f'  TOTAL {nm:46s} ' + '  '.join(f'{k}={v} GB' for k, v in s.items()))
    (a.root / 'qa').mkdir(exist_ok=True)
    json.dump(out, open(a.root / 'qa' / 'storage_estimate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
