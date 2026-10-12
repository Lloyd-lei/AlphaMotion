"""Re-run only the CPU judge on the trajectories stored in results/parity/<clip>__<robot>.series.npz (no GPU)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/arenalabs/umr_gpu'); sys.path.insert(0, '/home/arenalabs/umr_gpu/eval')
from surface_retargeting.gpu.bank import Problem
from judge import Judge, judge_clip, sphere_sdf
from surface_retargeting.gpu.geometry import object_from_objpen
OUT = Path('/var/tmp/alphamotion/umr_factory/umr_gpu')
clip, robot = sys.argv[1], sys.argv[2]
S = dict(np.load(OUT / 'results' / 'parity' / f'{clip}__{robot}.series.npz', allow_pickle=True))
res = json.load(open(OUT / 'results' / 'parity' / f'{clip}__{robot}.json'))
P = Problem(OUT / 'problems' / clip / f'{robot}.stock.cpu.npz')
floor = float(S['floor']); fps = float(S['fps']); flags = S['flags']
pose = S['obj_pose'] if S['obj_pose'].size else None
has_obj = pose is not None
J = Judge(OUT / 'robots' / robot / f'{robot}.xml', OUT / 'robots' / robot / 'surface.npz', P['objpen_mjb'] if has_obj else None,
          P['objpen_object_geoms'] if has_obj else None, int(P['objpen_object_qadr']) if has_obj else None)
osdf = None
if has_obj:
    V, F, _ = object_from_objpen(str(P['objpen_mjb']), P['objpen_object_geoms'])
    c = V.mean(0); r = float(np.linalg.norm(V - c, axis=1).mean()); osdf = sphere_sdf(c, r)
# human object contact labels (same rule as the pipeline) if available
ho = None
if has_obj:
    import torch
    from surface_retargeting.gpu.bank import Bank
    from surface_retargeting.gpu.pipeline import Retargeter
    Rt = Retargeter(Bank([P]), str(OUT / 'robots' / robot / 'surface.npz'), strict=False)
    ho = Rt.human(0).get('o_contact')
cands = {'cpu_umr': S['q_cpu_umr'], 'gpu_umr': S['q_gpu_umr'], 'gpu_strict': S['q_gpu_strict']}
if 'q_cpu_factory_strict' in S:
    cands['cpu_factory_strict'] = S['q_cpu_factory_strict']
for name, Qw in cands.items():
    Qr = Qw.copy(); Qr[:, 2] -= floor
    Sj, summ = judge_clip(J, Qr, fps, S['human_contact'], S['human_clear'], pose, osdf, ho, keep=(flags == 0) if name == 'gpu_strict' else None)
    if name == 'gpu_strict':
        keep = flags == 0
        S2 = {k: v[keep] for k, v in Sj.items() if len(v) == len(keep)}
        summ['flagged_excluded'] = int((~keep).sum())
        for k, kk in (('ground_min', 'kept_ground_pen_frames_gt_0p5mm'), ('colground_min', 'kept_colground_pen_frames_gt_0p5mm'),
                      ('self_min', 'kept_self_pen_frames_gt_0p5mm'), ('obj_min', 'kept_obj_pen_frames_gt_0p5mm')):
            if k in S2:
                summ[kk] = int((S2[k] < -0.0005).sum())
        rm = S2['region_min']; hck = S['human_contact'][keep]
        summ['kept_human_contact_robot_solid_frac'] = round(float((hck & (rm >= -0.0005) & (rm <= 0.001)).sum() / max(hck.sum(), 1)), 4)
        if 'obj_region_min' in S2 and ho is not None:
            hok = ho[keep]; orm = S2['obj_region_min']
            summ['kept_human_obj_contact_robot_solid_frac'] = round(float((hok & (orm >= -0.0005) & (orm <= 0.001)).sum() / max(hok.sum(), 1)), 4)
    res[name] = summ
    for k, v in Sj.items():
        S[f'{name}__{k}'] = v
json.dump(res, open(OUT / 'results' / 'parity' / f'{clip}__{robot}.json', 'w'), indent=1)
np.savez_compressed(OUT / 'results' / 'parity' / f'{clip}__{robot}.series.npz', **S)
print(json.dumps({k: res[k] for k in cands}, indent=1))
