"""Build the production job list (jobs.jsonl): one job = (source, object asset, robot, <= N clips).
Order: source by source (T0 = OmniContact), clip chunks interleaved over robots (G1, GR3, H2, A3 first).
Only sources the GPU engine supports are listed: flat floor + rigid object with a prop trajectory (MJCF)."""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ROBOTS4 = ['unitree_g1_29dof', 'fourier_gr3', 'unitree_h2', 'agibot_a3_t3d0']


def omnicontact(root='/home/arenalabs/umr_factory_data/smplx/omnicontact'):
    out = []
    for f in sorted(glob.glob(f'{root}/*/*/*/poses.npy')):
        d = os.path.dirname(f)
        xmls = sorted(glob.glob(d + '/*.xml'))
        if len(xmls) != 1:
            continue
        stem = Path(xmls[0]).stem
        prop = d + f'/prop_{stem}.csv'
        if not os.path.exists(prop):
            continue
        rel = os.path.relpath(d, root)
        if 'mirror' in rel.lower():
            continue
        out.append(dict(clip_id=rel.replace('/', '__'), dir=d, layout='flat', object_xml=xmls[0], object_prop=prop, object_name=stem))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/arenalabs/umr_factory_data/production/v1/_jobs/jobs.jsonl')
    ap.add_argument('--clips-per-job', type=int, default=64)
    ap.add_argument('--robots', default=','.join(ROBOTS4))
    a = ap.parse_args()
    robots = a.robots.split(',')
    jobs = []
    srcs = [('omnicontact', omnicontact())]
    for src, clips in srcs:
        by_obj = defaultdict(list)
        for c in clips:
            by_obj[(c['object_name'], c['object_xml'] if False else '')].append(c)
        # one asset per object name: use the first clip's xml (identical meshes per object name)
        chunks = []
        for (name, _), cl in sorted(by_obj.items()):
            for i in range(0, len(cl), a.clips_per_job):
                chunks.append((name, cl[0]['object_xml'], cl[i:i + a.clips_per_job], i // a.clips_per_job))
        for name, xml, cl, k in chunks:
            for r in robots:
                jobs.append(dict(job_id=f'{src}__{name}__c{k:03d}__{r}', source=src, robot=r,
                                 object=dict(name=name, xml=xml), clips=cl))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        for j in jobs:
            f.write(json.dumps(j) + '\n')
    print(f'{len(jobs)} jobs -> {a.out}; clips: ' + ', '.join(f'{s}={len(c)}' for s, c in srcs))


if __name__ == '__main__':
    main()
