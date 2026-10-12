"""Throughput-mode GPU retarget factory: many clips x several robots on one GPU, end to end.

  SMPL-X forward (shared by all robots) -> per robot: GPU source preprocessing -> UMR solve (CUDA graph, chains) ->
  LQR (banded, per piece) -> labels + frame-parallel strict pass -> per-clip outputs.

Benchmark corpus: --synthetic N_clips windows of random length/start cut from the UMR sample clip (same subject, so
one static prep per robot).  Reports robot-frames/s and a stage breakdown; --save writes per (clip, robot) npz.
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
from surface_retargeting.gpu.source_gpu import SMPLXForward, SourcePrep, load_smplx_sequence
from surface_retargeting.gpu.chains import build_schedule, ChainRunner
from surface_retargeting.gpu.lqr_band import lqr_chains
from surface_retargeting.strict.strict import RobotPoints, StrictProjector
from surface_retargeting.gpu.geometry import SelfCollision, CollisionGround
from surface_retargeting.export import export_clip
from surface_retargeting.strict.strict_fp import strict_chains, hysteresis_t, drop_short_t
from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu import throttle

OUT = Path(f'{SRCFG.work}')
PROB = {'agibot_a3_t3d0': 'lafan1_smplx__dance1_subject2/agibot_a3_t3d0.stock.cpu.npz',
        'unitree_g1_29dof': 'lafan1_smplx__dance1_subject2/unitree_g1_29dof.stock.cpu.npz',
        'fourier_gr3': 'lafan1_smplx__dance1_subject2/fourier_gr3.stock.cpu.npz',
        'unitree_h2': 'lafan1_smplx__dance1_subject2/unitree_h2.stock.cpu.npz'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robots', default='unitree_g1_29dof,fourier_gr3,unitree_h2')
    ap.add_argument('--B', type=int, default=1024)
    ap.add_argument('--Tb', type=int, default=16)
    ap.add_argument('--synthetic', type=int, default=0, help='number of synthetic clips (0: the sample clip once)')
    ap.add_argument('--clip-min', type=int, default=300); ap.add_argument('--clip-max', type=int, default=1800)
    ap.add_argument('--max-piece', type=int, default=1800); ap.add_argument('--warmup', type=int, default=192)
    ap.add_argument('--qp', default='fp32'); ap.add_argument('--strict', type=int, default=1)
    ap.add_argument('--strict-iters', type=int, default=30); ap.add_argument('--strict-chunk', type=int, default=16384)
    ap.add_argument('--save', default=''); ap.add_argument('--tag', default='bench'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dmon', type=int, default=1)
    ap.add_argument('--export', default=''); ap.add_argument('--duty', type=float, default=1.0); ap.add_argument('--repeat', type=int, default=1); ap.add_argument('--write', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    robots = a.robots.split(',')
    t_start = time.time()
    tm = {}
    # ---------------- corpus
    P0 = Problem(OUT / 'problems' / PROB[robots[0]])
    seq = load_smplx_sequence(P0)
    pose_src = np.asarray(seq['pose_aa'], np.float32); tr_src = np.asarray(seq['trans_orig'], np.float32)
    Tsrc = len(pose_src)
    rng = np.random.default_rng(a.seed)
    if a.synthetic > 0:
        lens = rng.integers(a.clip_min, a.clip_max + 1, a.synthetic)
        lens = np.minimum(lens, Tsrc)
        starts = rng.integers(0, Tsrc - lens + 1)
    else:
        lens = np.array([Tsrc]); starts = np.array([0])
    t0 = time.time()
    pose_all = torch.as_tensor(np.concatenate([pose_src[s:s + l] for s, l in zip(starts, lens)]), device=dev)
    tr_all = torch.as_tensor(np.concatenate([tr_src[s:s + l] for s, l in zip(starts, lens)]), device=dev)
    tm['load_h2d'] = time.time() - t0
    S = build_schedule(lens, a.B, a.max_piece, a.warmup)
    sched = {k: torch.as_tensor(v, device=dev) for k, v in S.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    B, Tch = sched['gframe'].shape
    print(json.dumps(dict(clips=len(lens), useful_frames=S['useful_frames'], solved_frames=S['solved_frames'],
                          pieces=len(S['pieces']), B=B, Tch=Tch, chain_occupancy=round(S['solved_frames'] / (B * Tch), 3))), flush=True)
    # ---------------- robots
    R = []
    for r in robots:
        P = Problem(OUT / 'problems' / PROB[r]); bank = Bank([P]); sv = UMRSolver(bank)
        if a.qp == 'fp32':
            sv.qp_dtype = torch.float32; sv.qp_eps = 1e-6
        prep = SourcePrep(P, dev)
        run = ChainRunner(sv, prep, B, a.Tb, dev)
        pts = RobotPoints(sv.kt, str(OUT / 'robots' / r / 'surface.npz'), dev)
        sc = SelfCollision(sv.kt, dev)
        cov = np.unique(np.load(OUT / 'robots' / r / 'surface.npz')['hull_geom'])
        proj = StrictProjector(sv.kt, pts, sc, None, lim_lo=sv.j_lo, lim_hi=sv.j_hi, jq=sv.jq, jd=sv.jd,
                               colground=CollisionGround(sv.kt, dev, covered=cov))
        proj.qp_dtype = torch.float32 if a.qp == 'fp32' else torch.float64; proj.qp_eps = 1e-6 if a.qp == 'fp32' else 1e-9; proj.qp_iters = 24
        R.append(dict(name=r, sv=sv, prep=prep, run=run, proj=proj, out=torch.zeros(B, Tch, bank.q_init.shape[1], dtype=torch.float64, device=dev),
                      hcl=torch.full((B, Tch, L.NR), float('inf'), device=dev), foot=torch.zeros(B, Tch, 2, 3, device=dev)))
    smplx = SMPLXForward(seq.get('gender', 'neutral'), seq.get('beta'), batch=8192, device=dev)
    # ---------------- warm-up (torch.compile / CUDA-graph capture / Warp JIT), excluded from the timing
    tw = time.time()
    Vw, Jw = smplx(pose_all[:8], tr_all[:8])
    for r in R:
        r['run'].init_pieces(Vw, Jw)
        bk = r['sv'].bank
        Qw = r['sv'].bank.problems[0]['q_cpu'][:96]
        Qw = torch.as_tensor(Qw, device=dev)[None].repeat(2, 1, 1)
        pw_ = torch.zeros(2, 96, dtype=torch.long, device=dev); pw_[1] = 1
        tlw = torch.arange(96, device=dev)[None].expand(2, -1).contiguous()
        hgw = torch.zeros(2, 96, L.NR, dtype=torch.bool, device=dev); hgw[..., L.RID['L_foot']] = True; hgw[..., L.RID['R_foot']] = True
        strict_chains(r['proj'], Qw, pw_, tlw, torch.zeros(2, 96, device=dev), hgw, iters=a.strict_iters,
                      human_clear_g=torch.where(hgw, 0.0, 0.5).float())
    for r in R:
        r['run'].timing = dict(prep=0.0, solve=0.0, init=0.0)
    torch.cuda.synchronize(); tm['warmup_excluded'] = round(time.time() - tw, 1)
    throttle.set_duty(a.duty)
    dmon = None
    if a.dmon:
        dmon_path = OUT / 'logs' / f'dmon_{a.tag}.txt'
        dmon = subprocess.Popen(['timeout', '7200', 'nvidia-smi', 'dmon', '-s', 'pu', '-d', '1', '-f', str(dmon_path)])
    torch.cuda.synchronize(); t_run0 = time.time()
    # ---------------- init solves (frame 0 of every piece)
    pieces = S['pieces']
    gstart = torch.as_tensor(S['offs'][pieces[:, 0]] + pieces[:, 1], device=dev)
    t0 = time.time(); V, J = smplx(pose_all[gstart], tr_all[gstart]); torch.cuda.synchronize(); tm['smplx'] = time.time() - t0
    for r in R:
        r['run'].init_pieces(V, J)
    del V, J
    # ---------------- streaming blocks
    feet = [L.RID['L_foot'], L.RID['R_foot']]
    for t0b in range(0, Tch, a.Tb):
        nb = min(a.Tb, Tch - t0b)
        gfr = sched['gframe'][:, t0b:t0b + a.Tb]
        if gfr.shape[1] < a.Tb:
            gfr = torch.cat([gfr, torch.full((B, a.Tb - gfr.shape[1]), -1, dtype=gfr.dtype, device=dev)], 1)
        ts = time.time()
        V, J = smplx(pose_all[gfr.reshape(-1).clamp_min(0)], tr_all[gfr.reshape(-1).clamp_min(0)])
        torch.cuda.synchronize(); tm['smplx'] = tm.get('smplx', 0) + time.time() - ts
        throttle.tick()
        for r in R:
            hcl = r['run'].fill_block(V, J)
            r['hcl'][:, t0b:t0b + nb] = hcl.reshape(B, a.Tb, -1)[:, :nb]
            r['run'].steps(sched, t0b, nb, r['out'], None)
        del V, J
        if t0b % (a.Tb * 32) == 0:
            el = time.time() - t_run0
            done = int((sched['gframe'][:, :t0b + nb] >= 0).sum())
            print(f'[factory] step {t0b + nb}/{Tch}  solved robot-frames/s so far {done * len(R) / el:.0f}', flush=True)
    torch.cuda.synchronize(); tm['stream_total'] = time.time() - t_run0
    for r in R:
        for k, v in r['run'].timing.items():
            tm[f'{r["name"]}:{k}'] = round(v, 2)
    # ---------------- LQR + floor + labels + strict
    piece = sched['piece']; tl = sched['tl']
    plen = torch.as_tensor(pieces[:, 2] - pieces[:, 1], device=dev)
    rem = torch.where(piece >= 0, plen[piece.clamp_min(0)] - tl, torch.zeros_like(tl))
    valid = piece >= 0
    results = {}
    for r in R:
        sv = r['sv']
        ts = time.time()
        Qf = lqr_chains(r['out'], piece, tl, rem, sv.a, sv.jq.tolist(), sv.clamp)
        torch.cuda.synchronize(); tm[f'{r["name"]}:lqr'] = round(time.time() - ts, 2)
        # human floor per clip (factory rule on the foot-region lowest points; quasi-static frames)
        ts = time.time()
        hz = r['hcl']                                      # [B,T,R] (z of region lowest human slot)
        footmin = hz[..., feet]                            # [B,T,2]
        allmin = hz.amin(-1, keepdim=True)
        vel = torch.zeros_like(footmin)
        dz = (footmin[:, 1:] - footmin[:, :-1]).abs() * 30.0
        vel[:, 1:] = torch.where((piece[:, 1:] == piece[:, :-1])[..., None], dz, torch.full_like(dz, 1e9))
        ok = valid[..., None] & (vel < 0.05) & (footmin <= allmin + 0.005)
        clip = sched['clip']
        floor_clip = torch.zeros(len(lens), device=dev)
        cl = clip[..., None].expand_as(footmin)[ok]; fv = footmin[ok]
        if cl.numel():
            order = torch.argsort(cl * 10.0 + fv.clamp(-5, 5) / 100.0)          # group by clip, then value
            cl_s, fv_s = cl[order], fv[order]
            cnt = torch.bincount(cl_s, minlength=len(lens)); first = torch.cumsum(cnt, 0) - cnt
            med = fv_s[(first + cnt // 2).clamp(max=len(fv_s) - 1)]
            floor_clip = torch.where(cnt >= 10, med, torch.zeros_like(med))
        floor_bt = floor_clip[clip.clamp_min(0)]
        hcl = hz - floor_bt[..., None]
        reset = valid & (tl == 0)
        human_g = drop_short_t(hysteresis_t(torch.where(valid[..., None], hcl, torch.full_like(hcl, float('inf'))), 0.01, 0.02, reset), 3, reset)
        tm[f'{r["name"]}:floor_labels'] = round(time.time() - ts, 2)
        if a.strict:
            ts = time.time()
            Qs, flags, f0, req_g, req_o, info = strict_chains(r['proj'], Qf, piece, tl, floor_bt, human_g, chunk=a.strict_chunk,
                                                              iters=a.strict_iters, human_clear_g=hcl)
            for k_ in ('extras', 'lift_g', 'lift_o'):
                info.pop(k_, None)
            torch.cuda.synchronize(); tm[f'{r["name"]}:strict'] = round(time.time() - ts, 2)
            tm[f'{r["name"]}:strict_info'] = json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
            print('[factory] strict', r['name'], tm[f'{r["name"]}:strict_info'], flush=True)
        else:
            Qs, flags, req_g = Qf, torch.zeros_like(piece, dtype=torch.int32), human_g
        r['final'] = Qs; r['flags'] = flags; r['req_g'] = req_g; r['human_g'] = human_g; r['floor_bt'] = floor_bt; r['raw_lqr'] = Qf
        if a.export:
            ts = time.time()
            keepm = valid & sched['keep']
            clip_t = sched['clip']; cfr_t = sched['cframe']
            ed = Path(a.export) / r['name']; ed.mkdir(parents=True, exist_ok=True)
            for c in range(len(lens)):
                m = keepm & (clip_t == c)
                bb, tt_ = torch.nonzero(m, as_tuple=True)
                o = torch.argsort(cfr_t[bb, tt_])
                bb, tt_ = bb[o], tt_[o]
                exd, qc = export_clip(r['proj'], Qs[bb, tt_], float(floor_bt[bb[0], tt_[0]]), 30.0, flags[bb, tt_], None,
                                      dict(ground_contact=human_g[bb, tt_].cpu().numpy(), ground_clearance_mm=(hcl[bb, tt_] * 1000).half().cpu().numpy()))
                if a.write:
                    np.savez(ed / f'clip{c:05d}.npz', **exd)
            torch.cuda.synchronize(); tm[f'{r["name"]}:export'] = round(time.time() - ts, 2)
    torch.cuda.synchronize()
    t_total = time.time() - t_run0
    if dmon is not None:
        dmon.terminate()
    useful = S['useful_frames'] * len(R)
    res = dict(tag=a.tag, B=B, Tb=a.Tb, robots=robots, clips=len(lens), useful_robot_frames=useful,
               solved_robot_frames=S['solved_frames'] * len(R), seconds_total=round(t_total, 2),
               robot_frames_per_s=round(useful / t_total), stage_seconds={k: (round(v, 2) if isinstance(v, float) else v) for k, v in tm.items()},
               projected_hours_50h_x_20robots=round(50 * 3600 * 30 * 20 / (useful / t_total) / 3600, 2),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), qp=a.qp, strict=a.strict,
               chain_occupancy=round(S['solved_frames'] / (B * Tch), 3), duty=a.duty, throttle_sleep_s=round(throttle.slept(), 1),
               network_pause_s=round(throttle.net_slept(), 1),
               robot_frames_per_s_excl_network_pauses=round(useful / max(t_total - throttle.net_slept(), 1e-9)))
    if dmon is not None:
        try:
            lines = [l for l in open(OUT / 'logs' / f'dmon_{a.tag}.txt')]
            hdr = [l for l in lines if l.startswith('# gpu')][0].split()[1:]
            rows = [l.split() for l in lines if l.strip() and not l.startswith('#')]
            col = lambda name: np.array([float(x[hdr.index(name)]) for x in rows if len(x) == len(hdr) and x[hdr.index(name)].replace('.', '').isdigit()])
            sm = col('sm'); pw = col('pwr')
            res['sm_util_mean_pct'] = round(float(sm.mean()), 1) if len(sm) else None
            res['sm_util_p50_pct'] = round(float(np.median(sm)), 1) if len(sm) else None
            res['gpu_power_mean_W'] = round(float(pw.mean()), 1) if len(pw) else None
            res['gpu_power_p95_W'] = round(float(np.percentile(pw, 95)), 1) if len(pw) else None
            if len(pw):
                res['robot_frames_per_joule'] = round(useful / t_total / float(pw.mean()), 2)
            res['note_gpu_shared'] = 'util/power are whole-GPU numbers (a BVH->SMPL-X converter shares the GPU)'
        except Exception as e:
            res['sm_util_error'] = str(e)
    for r in R:
        fl = r['flags'][valid & sched['keep']]
        res[f'{r["name"]}:flagged_frac'] = round(float((fl != 0).float().mean()), 5)
    print(json.dumps(res, indent=1), flush=True)
    (OUT / 'results' / 'bench').mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'results' / 'bench' / f'{a.tag}.json', 'w'), indent=1)
    if a.save:
        sd = Path(a.save); sd.mkdir(parents=True, exist_ok=True)
        keep = (valid & sched['keep']).cpu().numpy()
        clip = sched['clip'].cpu().numpy(); cfr = sched['cframe'].cpu().numpy()
        for r in R:
            Qn = r['final'].cpu().numpy(); Ql = r['raw_lqr'].cpu().numpy(); Qr = r['out'].cpu().numpy()
            fl = r['flags'].cpu().numpy(); fb = r['floor_bt'].cpu().numpy(); rg = r['req_g'].cpu().numpy(); hg = r['human_g'].cpu().numpy()
            for c in range(len(lens)):
                m = keep & (clip == c)
                order = np.argsort(cfr[m])
                qo = Qn[m][order].copy(); qo[:, 2] -= fb[m][order]
                np.savez_compressed(sd / f'clip{c:05d}__{r["name"]}.npz', qpos=qo.astype(np.float32), qpos_strict_unshifted=Qn[m][order],
                                    qpos_umr_lqr=Ql[m][order], qpos_umr_raw=Qr[m][order], flags=fl[m][order], floor=fb[m][order],
                                    ground_contact_required=rg[m][order], human_ground_contact=hg[m][order],
                                    clip_start_in_source=np.int64(starts[c]), clip_len=np.int64(lens[c]))


if __name__ == '__main__':
    main()
