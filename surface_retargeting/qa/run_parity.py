"""Parity + strictness evaluation for one (clip, robot):  GPU pipeline vs CPU UMR (and the CPU factory strict output
when available), all judged by eval/judge.py (independent CPU MuJoCo ruler).  Writes
  results/parity/<clip>__<robot>.json        summary
  results/parity/<clip>__<robot>.series.npz  per-frame judge series (used by the video renderer)
  results/export/<clip>__<robot>.npz         GPU output in the DATA_SPEC_v2 layout
With --outdir NAME: results/NAME/<clip>__<robot>.npz (export) and results/NAME/parity/ (json + series) instead, so a
re-run never overwrites the reference exports.  --judge strict judges only the GPU strict output (CPU-light);
the other candidates' trajectories are still stored in the series for the videos.
Kept frames = flags & DROP_MASK == 0 (bit 256 follow_human_slip is informational).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/arenalabs/umr_gpu'); sys.path.insert(0, '/home/arenalabs/umr_gpu/eval')
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.pipeline import Retargeter
from surface_retargeting.export import export_clip
from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu.slip import DROP_MASK, FLAG_FOLLOW
from judge import Judge, judge_clip, sphere_sdf

OUT = Path('/var/tmp/alphamotion/umr_factory/umr_gpu')
FACTORY = {('lafan1_smplx__dance1_subject2', 'unitree_g1_29dof'): '/var/tmp/alphamotion/umr_factory/lloyd_gpu_eval/cpu_unitree_g1_29dof.npz',
           ('lafan1_smplx__dance1_subject2', 'fourier_gr3'): '/var/tmp/alphamotion/umr_factory/lloyd_gpu_eval/cpu_fourier_gr3.npz'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('clip'); ap.add_argument('robot'); ap.add_argument('--qp', default='fp32')
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--outdir', default='', help='results/<outdir>/ for export + parity (default: results/export, results/parity)')
    ap.add_argument('--judge', default='all', choices=['all', 'strict'])
    a = ap.parse_args()
    prob = OUT / 'problems' / a.clip / f'{a.robot}.stock.cpu.npz'
    P = Problem(prob); bank = Bank([P]); dev = bank.device
    R = Retargeter(bank, str(OUT / 'robots' / a.robot / 'surface.npz'))
    R.strict_iters = 30
    if a.qp == 'fp32':
        R.sv.qp_dtype = torch.float32; R.sv.qp_eps = 1e-6
        R.proj.qp_dtype = torch.float32; R.proj.qp_eps = 1e-6; R.proj.qp_iters = 24
    T = P.T
    tm = {}
    torch.cuda.synchronize(); t0 = time.time()
    out = R.run(torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev), torch.tensor([T], device=dev), timings=tm)
    torch.cuda.synchronize(); tm['total'] = time.time() - t0
    h = R.human(0)
    floor = float(h['floor']); fps = float(h['fps'])
    q_umr_gpu = out['q_umr'][0].cpu().numpy(); q_strict = out['q_strict_world'][0].cpu().numpy()
    flags = out['flags'][0].cpu().numpy()
    keep_all = (flags & DROP_MASK) == 0
    RES = OUT / 'results' / a.outdir if a.outdir else OUT / 'results'
    PAR = RES / 'parity'
    EXP = RES if a.outdir else OUT / 'results' / 'export'
    cpu_out = np.load(P.args['out'], allow_pickle=True)['qpos'].astype(np.float64)
    res = dict(clip=a.clip, robot=a.robot, frames=T, fps=fps, floor_m=floor, timings_s={k: round(v, 2) for k, v in tm.items()},
               gpu_fps_b1=round(T / tm['total'], 1), cpu_umr_solve_s=round(float(P['t_solve'].sum()), 1),
               cpu_umr_fps=round(T / float(P['t_solve'].sum()), 1), qp=a.qp)
    # ---- parity of the UMR stage (GPU vs CPU, both after the LQR filter)
    jq = R.sv.jq.cpu().numpy()
    dj = np.abs(q_umr_gpu[:, jq] - cpu_out[:, jq]).max(1)
    K1 = R.kt.fk(torch.as_tensor(q_umr_gpu, device=dev)); K2 = R.kt.fk(torch.as_tensor(cpu_out, device=dev))
    X1 = R.pts.world(K1)[:, :R.pts.ndense]; X2 = R.pts.world(K2)[:, :R.pts.ndense]
    dp = (X1 - X2).norm(dim=-1).cpu().numpy() * 1000
    res['parity_umr_stage'] = dict(joint_maxabs_rad_p50=float(np.median(dj)), p99=float(np.percentile(dj, 99)), max=float(dj.max()),
                                   root_pos_max_mm=float(np.linalg.norm(q_umr_gpu[:, :3] - cpu_out[:, :3], axis=1).max() * 1000),
                                   surface_point_mm_mean=float(dp.mean()), surface_point_mm_p99_of_frame_max=float(np.percentile(dp.max(1), 99)),
                                   surface_point_mm_max=float(dp.max()))
    # ---- judge
    pose = P['obj_pose'].copy() if bank.has_obj else None
    J = Judge(OUT / 'robots' / a.robot / f'{a.robot}.xml', OUT / 'robots' / a.robot / 'surface.npz',
              P['objpen_mjb'] if bank.has_obj else None, P['objpen_object_geoms'] if bank.has_obj else None,
              int(P['objpen_object_qadr']) if bank.has_obj else None)
    osdf = None
    if R.obj is not None and R.obj.kind == 'sphere':
        osdf = sphere_sdf(R.obj.center.cpu().numpy(), R.obj.radius)
    if pose is not None:
        pose[:, 2] -= floor
    hreg_any = lambda arr: arr                                      # [T,R]
    hc = h['g_contact']; hcl = h['g_clear']; hoc = h.get('o_contact')
    cands = {'cpu_umr': cpu_out, 'gpu_umr': q_umr_gpu, 'gpu_strict': q_strict}
    fac = FACTORY.get((a.clip, a.robot))
    if fac and Path(fac).exists():
        fz = np.load(fac, allow_pickle=True)
        cands['cpu_factory_strict'] = fz['qpos'].astype(np.float64) + np.eye(fz['qpos'].shape[1])[2] * float(fz['floor_offset_removed'])
        res['cpu_factory_flagged'] = int((fz['flags'] > 0).sum())
    series = {}
    for name, Qw in cands.items():
        if a.judge == 'strict' and name != 'gpu_strict':
            continue
        t1 = time.time()
        Qr = Qw.copy(); Qr[:, 2] -= floor
        S, summ = judge_clip(J, Qr, fps, hc, hcl, pose, osdf, hoc, step=a.step, keep=keep_all if name == 'gpu_strict' else None)
        if name == 'gpu_strict':
            keep = keep_all[::a.step]
            S2 = {k: v[keep] for k, v in S.items() if len(v) == len(keep)}
            summ['flagged_excluded'] = int((~keep).sum())
            summ['kept_ground_pen_frames_gt_0p5mm'] = int((S2['ground_min'] < -0.0005).sum())
            summ['kept_colground_pen_frames_gt_0p5mm'] = int((S2['colground_min'] < -0.0005).sum())
            summ['kept_self_pen_frames_gt_0p5mm'] = int((S2['self_min'] < -0.0005).sum())
            if 'obj_min' in S2:
                summ['kept_obj_pen_frames_gt_0p5mm'] = int((S2['obj_min'] < -0.0005).sum())
            rm = S2['region_min']; hck = hc[::a.step][keep]
            solid = (rm >= -0.0005) & (rm <= 0.001)
            summ['kept_human_contact_robot_solid_frac'] = round(float((hck & solid).sum() / max(hck.sum(), 1)), 4)
        summ['judge_s'] = round(time.time() - t1, 1)
        res[name] = summ
        for k, v in S.items():
            series[f'{name}__{k}'] = v
    res['gpu_flags_hist'] = {str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)}
    res['kept_frac'] = round(float(keep_all.mean()), 4)
    res['follow_human_slip_frac'] = round(float(((flags & FLAG_FOLLOW) > 0).mean()), 4)
    # temporal continuity of the root (cm per frame, scaled to 30 fps) for every variant
    def root_step(Q):
        dd = np.linalg.norm(np.diff(Q[:, :3], axis=0), axis=1) * (fps / 30.0) * 100
        return dict(p99_cm=round(float(np.percentile(dd, 99)), 2), max_cm=round(float(dd.max()), 2), frames_gt_5cm=int((dd > 5).sum()))
    res['root_step'] = {k: root_step(v) for k, v in cands.items()}
    kk = keep_all[1:]
    dd = np.linalg.norm(np.diff(q_strict[:, :3], axis=0), axis=1) * (fps / 30.0) * 100
    res['root_step']['gpu_strict_kept'] = dict(max_cm=round(float(dd[kk].max()), 2) if kk.any() else None, frames_gt_5cm=int((dd[kk] > 5).sum()))
    res['gpu_flagged'] = int((~keep_all).sum())
    hs_any = h['g_slip'].any(1); hc_any = h['g_contact'].any(1)
    res['gpu_flagged_while_human_slips'] = int((~keep_all & hs_any).sum())
    res['human_slip_frames'] = int(hs_any.sum()); res['human_contact_frames'] = int(hc_any.sum())
    PAR.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(PAR / f'{a.clip}__{a.robot}.json', 'w'), indent=1)
    np.savez_compressed(PAR / f'{a.clip}__{a.robot}.series.npz', human_slip_speed=h['g_slip_speed'], floor=floor, fps=fps, flags=flags,
                        q_cpu_umr=cpu_out, q_gpu_umr=q_umr_gpu, q_gpu_strict=q_strict, human_contact=hc, human_clear=hcl,
                        obj_pose=pose if pose is not None else np.zeros(0), **series,
                        **({'q_cpu_factory_strict': cands['cpu_factory_strict']} if 'cpu_factory_strict' in cands else {}))
    # ---- spec-v2 export of the GPU output
    hum = dict(ground_contact=hc, ground_clearance_mm=(hcl * 1000).astype(np.float16), ground_slip_speed=h['g_slip_speed'])
    if hoc is not None:
        hum.update(object_contact=hoc, object_clearance_mm=(np.clip(h['o_clear'], -1, 10) * 1000).astype(np.float16))
    pose_w = None if not bank.has_obj else torch.as_tensor(P['obj_pose'], device=dev)
    ex, qc = export_clip(R.proj, out['q_strict_world'][0], floor, fps, flags, pose_w, hum, qref=out['q_umr'][0])
    if bank.has_obj:
        ex['object_pose'] = pose.astype(np.float32)        # floor-relative pos + quat wxyz
    EXP.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXP / f'{a.clip}__{a.robot}.npz', **ex,
                        surface_asset=str(OUT / 'robots' / a.robot / 'surface.npz'), robot_xml=str(OUT / 'robots' / a.robot / f'{a.robot}.xml'))
    res['export_qc'] = qc
    json.dump(res, open(PAR / f'{a.clip}__{a.robot}.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
