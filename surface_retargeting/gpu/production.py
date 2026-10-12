"""Production: one job = (source, object asset, robot, list of clips) -> per-clip DATA_SPEC_v2 files.

Pipeline per job (all on the GPU, one process):
  static robot prep (beta-zero shared correspondence capture) -> per clip: betas -> smpl_scale + template frames;
  SMPL-X forward per block (betas per frame) -> UMR inputs (+ object contact maps) -> UMR solve (CUDA graph, chains)
  -> LQR per piece -> human labels (same ruler) -> strict pass (spec v2) -> export per clip (atomic writes).
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu import throttle
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.chains import build_schedule, ChainRunner
from surface_retargeting.export import export_clip
from surface_retargeting.gpu.geometry import SelfCollision, CollisionGround
from surface_retargeting.gpu.hoi import ObjectAsset, object_contact_maps
from surface_retargeting.gpu.lqr_band import lqr_chains
from surface_retargeting.gpu.solver import UMRSolver, SeqState
from surface_retargeting.gpu.source_gpu import SMPLXForward, SourcePrep
from surface_retargeting.strict.strict import RobotPoints, StrictProjector, seg_min
from surface_retargeting.strict.strict_fp import strict_seq_segmented, hysteresis_t, drop_short_t
from surface_retargeting.gpu import slip as SL

UMR = Path(f'{SRCFG.umr_root}')
GPUOUT = Path(f'{SRCFG.work}')
SPEC_VERSION = 'DATA_SPEC_v2 (2026-10-04)'
FEET = [L.RID['L_foot'], L.RID['R_foot']]


def engine_version():
    h = hashlib.sha1()
    # the old umr_gpu package = gpu/*.py + strict/strict.py + strict/strict_fp.py + export.py (hashed in basename order)
    pkg = Path(__file__).resolve().parents[1]
    files = list((pkg / 'gpu').glob('*.py')) + [pkg / 'strict' / 'strict.py', pkg / 'strict' / 'strict_fp.py', pkg / 'export.py']
    for f in sorted(files, key=lambda p: p.name):
        h.update(f.read_bytes())
    return 'umr_gpu-' + h.hexdigest()[:12]


def sha_files(paths):
    h = hashlib.sha1()
    for p in paths:
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- clip loading
def load_clip(clip):
    """clip dict -> (pose [T,165] f32, transl [T,3], betas [10], fps, up, frame_ids) at 30 fps (stride from source fps)."""
    d = Path(clip['dir'])
    if clip['layout'] == 'flat':
        poses = np.load(d / 'poses.npy').reshape(-1, 165).astype(np.float32)
        tr = np.load(d / ('transl.npy' if (d / 'transl.npy').exists() else 'trans.npy')).astype(np.float32)
        betas = np.load(d / 'betas.npy').reshape(-1).astype(np.float32)
        fps = float(np.load(d / 'mocap_framerate.npy')) if (d / 'mocap_framerate.npy').exists() else 30.0
        up = str(np.load(d / 'output_up.npy')) if (d / 'output_up.npy').exists() else 'z'
        files = [d / 'poses.npy', d / 'betas.npy']
    else:
        z = np.load(d / clip['npz'], allow_pickle=True)
        poses = np.asarray(z['poses']).reshape(len(z['poses']), -1).astype(np.float32)
        tr = np.asarray(z['trans'] if 'trans' in z.files else z['transl']).astype(np.float32)
        betas = np.asarray(z['betas']).reshape(-1).astype(np.float32)
        fps = float(z['fps']) if 'fps' in z.files else 30.0
        up = str(z['output_up']) if 'output_up' in z.files else 'z'
        files = [d / clip['npz']]
    stride = max(1, int(round(fps / 30.0)))
    fids = np.arange(0, len(poses), stride)
    b10 = np.zeros(10, np.float32); b10[:min(10, len(betas))] = betas[:10]
    return poses[fids], tr[fids], b10, fps / stride, up.lower(), fids, files, fps


# ---------------------------------------------------------------------------------------------- robot statics
class RobotStatic:
    def __init__(self, robot, hoi, device='cuda', qp='fp32'):
        prob = GPUOUT / 'problems' / 'betazero__static90' / f'{robot}.betazero.cpu.npz'
        self.P = Problem(prob)
        self.robot = robot
        self.bank = Bank([self.P], device)
        overrides = {}
        if hoi:
            d = json.load(open(UMR / 'humanoid_retarget_defaults_hsi_hoi_standard.json'))['solver']
            overrides = {k: v for k, v in d.items() if k in self.P.args and not isinstance(v, (dict, list))}
            # HOI body-segment module: same sampled slots, arm/shoulder normal costs are 0
            for p in (str(UMR / 'scripts'), str(UMR)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import retarget_body_segment_surface_hoi_hsi as hb
            part = self.P['part_ids']; sel = self.P['selected']
            nc = hb.surface_slot_costs_from_segments(len(part), part, 'normal_cost', 'SurfaceNormal', log_prefix='umr_gpu')
            pc = hb.surface_slot_costs_from_segments(len(part), part, 'point_cost', 'SurfacePoint', log_prefix='umr_gpu')
            self.bank.normal_cost[0, :len(sel)] = torch.as_tensor(np.asarray(nc)[sel], device=device, dtype=torch.float64)
            self.bank.point_cost[0, :len(sel)] = torch.as_tensor(np.asarray(pc)[sel], device=device, dtype=torch.float64)
            self.bank.args = dict(self.P.args, **overrides)
        self.overrides = overrides
        self.hoi = hoi
        self.qp = qp


# ---------------------------------------------------------------------------------------------- job
def run_job(job, out_root, B_max=1024, Tb=16, max_piece=1800, warmup=192, strict_iters=30, strict_chunk=8192,
            device='cuda', log=print, write=True, seg_len=256, seg_overlap=90):
    t_job = time.time()
    dev = torch.device(device)
    robot = job['robot']
    clips = job['clips']
    hoi = job.get('object') is not None
    tm = {}
    # ---- clips
    t0 = time.time()
    data = [load_clip(c) for c in clips]
    lens = np.array([len(d[0]) for d in data])
    pose_all = torch.as_tensor(np.concatenate([d[0] for d in data]), device=dev)
    tr_all = torch.as_tensor(np.concatenate([d[1] for d in data]), device=dev)
    betas = torch.as_tensor(np.stack([d[2] for d in data]), device=dev)
    fps_c = np.array([d[3] for d in data]); ups = [d[4] for d in data]
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    clip_of_frame = torch.as_tensor(np.repeat(np.arange(len(clips)), lens), device=dev)
    tm['load'] = time.time() - t0
    # ---- robot statics
    RS = RobotStatic(robot, hoi, device)
    smplx = SMPLXForward('neutral', None, batch=4096, device=dev)
    prep = SourcePrep(RS.P, dev)
    scale_c = prep.set_clips(smplx, betas, ups)                                     # [C]
    asset = None
    if hoi:
        asset = ObjectAsset(job['object']['xml'], dev)
    bank = RS.bank
    sv = UMRSolver(bank, geometry=(asset.sdf if hoi else None))
    if RS.qp == 'fp32':
        sv.qp_dtype = torch.float32; sv.qp_eps = 1e-6
    # object poses per clip (scaled), pose8 = pos, quat, scale
    obj_pose_all = None
    if hoi:
        pp = []
        for c, d in zip(clips, data):
            mo = ObjectAsset.motion(c['object_prop'], d[5], float(scale_c[len(pp)]), d[4])
            pp.append(np.concatenate([mo, np.full((len(mo), 1), float(scale_c[len(pp)]))], 1))
        obj_pose_all = torch.as_tensor(np.concatenate(pp), device=dev)
    # ---- schedule
    S = build_schedule(lens, max(1, min(B_max, len(lens))), max_piece, warmup)
    sched = {k: torch.as_tensor(v, device=dev) for k, v in S.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    B, Tch = sched['gframe'].shape
    pieces = S['pieces']
    run = ChainRunner(sv, prep, B, Tb, dev)
    n = B * Tb
    if hoi:
        bank.has_obj = True
        bank.obj_pose = torch.zeros(n, 8, dtype=torch.float64, device=dev)
        K = 128
        bank.oc_slot = torch.full((n, K), -1, dtype=torch.long, device=dev)
        bank.oc_d = torch.zeros(n, K, dtype=torch.float64, device=dev)
        bank.oc_vec = torch.zeros(n, K, 3, dtype=torch.float64, device=dev)
        bank.oc_pt = torch.zeros(n, K, 3, dtype=torch.float64, device=dev)
    pts = RobotPoints(sv.kt, str(GPUOUT / 'robots' / robot / 'surface.npz'), dev)
    cov = np.unique(np.load(GPUOUT / 'robots' / robot / 'surface.npz')['hull_geom'])
    proj = StrictProjector(sv.kt, pts, SelfCollision(sv.kt, dev), asset.sdf if hoi else None, lim_lo=sv.j_lo, lim_hi=sv.j_hi,
                           jq=sv.jq, jd=sv.jd, colground=CollisionGround(sv.kt, dev, covered=cov))
    proj.qp_dtype = torch.float32; proj.qp_eps = 1e-6; proj.qp_iters = 24
    proj.set_fps(30.0)
    hreg = torch.as_tensor(L.part_to_region(RS.P['part_ids']), device=dev)
    # ---- piece initial frames: inputs + UMR frame-0 solve
    t0 = time.time()
    g0 = torch.as_tensor(S['offs'][pieces[:, 0]] + pieces[:, 1], device=dev)
    V, J = smplx(pose_all[g0], tr_all[g0], betas[clip_of_frame[g0]])
    o = prep.frames(V, J, clip=clip_of_frame[g0])
    data_ns = SimpleNamespace(src_sel=o['src_sel'], ntgt=o['ntgt'], gd=o['gd'], gw=o['gw'], has_ground=True,
                              has_sc='sc_pairs' in o, sc_pairs=o.get('sc_pairs'), sc_d=o.get('sc_d'), sc_w=o.get('sc_w'),
                              sc_dir=o.get('sc_dir'), has_obj=hoi)
    if hoi:
        po = obj_pose_all[g0]
        oc = object_contact_maps(asset, o['src'], po, sv.a)
        data_ns.obj_pose = po; data_ns.oc_slot = oc['oc_slot']; data_ns.oc_d = oc['oc_d']; data_ns.oc_vec = oc['oc_vec']; data_ns.oc_pt = oc['oc_pt']
    Np = len(pieces)
    q0s, at0, ao0 = [], [], []
    for s in range(0, Np, 4096):
        sl = slice(s, s + 4096)
        sub = SimpleNamespace(**{k: (v[sl] if torch.is_tensor(v) else v) for k, v in vars(data_ns).items()})
        nn_ = sub.src_sel.shape[0]
        pz = torch.zeros(nn_, dtype=torch.long, device=dev)
        stc = SeqState(pz, pz, pz + 1, bank.S, dev)
        inp = sv.frame_inputs(stc, torch.arange(nn_, device=dev), sub)
        q = o['q_init'][sl].clone()
        sv.update_anchors(stc, q, inp)
        fz = torch.zeros(nn_, dtype=torch.bool, device=dev); live = torch.ones(nn_, dtype=torch.bool, device=dev)
        for k in range(int(sv.a['pose_init_iters'])):
            dq, _ = sv.gn_step(stc, q, q, q, fz, fz, inp, live)
            q = sv.clamp(sv.kt.integrate(q, dq))
        q0s.append(q); at0.append(stc.anchor_tgt.clone()); ao0.append(stc.anchor_on.clone())
    run.q0 = torch.cat(q0s); run.a_tgt0 = torch.cat(at0); run.a_on0 = torch.cat(ao0)
    del V, J, o, data_ns
    tm['init'] = time.time() - t0
    # ---- streaming
    t0 = time.time()
    out = torch.zeros(B, Tch, bank.q_init.shape[1], dtype=torch.float64, device=dev)
    hz = torch.full((B, Tch, L.NR), float('inf'), device=dev)          # human region min z (raw, before floor)
    hsp = torch.full((B, Tch, L.NR), float('nan'), device=dev)        # human slip speed per region (m/s, slip.py ruler)
    hsv = torch.zeros((B, Tch, L.NR, 2), device=dev)                  # human slip velocity (m/s; mean direction, min magnitude)
    prev_src = None; prev_supp = None
    hoc = torch.full((B, Tch, L.NR), float('inf'), device=dev)         # human region min object distance
    for t0b in range(0, Tch, Tb):
        nb = min(Tb, Tch - t0b)
        gfr = sched['gframe'][:, t0b:t0b + Tb]
        if gfr.shape[1] < Tb:
            gfr = torch.cat([gfr, torch.full((B, Tb - gfr.shape[1]), -1, dtype=gfr.dtype, device=dev)], 1)
        gf = gfr.reshape(-1).clamp_min(0)
        V, J = smplx(pose_all[gf], tr_all[gf], betas[clip_of_frame[gf]])
        ob = prep.frames(V, J, chunk=4096, clip=clip_of_frame[gf])
        bank.src_sel.copy_(ob['src_sel']); bank.ntgt.copy_(ob['ntgt']); bank.gd.copy_(ob['gd']); bank.gw.copy_(ob['gw'])
        if bank.has_sc:
            bank.sc_pairs.copy_(ob['sc_pairs']); bank.sc_d.copy_(ob['sc_d']); bank.sc_w.copy_(ob['sc_w']); bank.sc_dir.copy_(ob['sc_dir'])
        zsrc = ob['src'][..., 2]
        hz[:, t0b:t0b + nb] = torch.full((zsrc.shape[0], L.NR), float('inf'), device=dev).scatter_reduce(
            1, hreg[None].expand(zsrc.shape[0], -1), zsrc, reduce='amin').reshape(B, Tb, -1)[:, :nb]
        # human slip speed (umr_gpu/slip.py ruler, unsmoothed here because the source is streamed): min tangential
        # speed (backward difference, 30 fps) over the region's points within 3 mm of the region minimum at both t-1
        # and t (masked to human contact frames after the labels below)
        Xs = ob['src'].reshape(B, Tb, -1, 3)
        Xp = torch.cat([(prev_src if prev_src is not None else Xs[:, 0])[:, None], Xs[:, :-1]], 1)
        pc = sched['piece'][:, t0b:t0b + Tb]
        pprev = torch.cat([sched['piece'][:, t0b - 1:t0b] if t0b > 0 else torch.full_like(pc[:, :1], -2), pc[:, :-1]], 1)
        okv = (pc >= 0) & (pc == pprev)
        if pc.shape[1] < Tb:
            okv = torch.cat([okv, torch.zeros(B, Tb - okv.shape[1], dtype=torch.bool, device=dev)], 1)
        rmin = hz[:, t0b:t0b + Tb] if nb == Tb else torch.cat([hz[:, t0b:t0b + nb], torch.full((B, Tb - nb, L.NR), float('inf'), device=dev)], 1)
        supp = Xs[..., 2] <= rmin[..., hreg] + 0.003
        suppp = torch.cat([(prev_supp if prev_supp is not None else supp[:, 0])[:, None], supp[:, :-1]], 1)
        vt = (Xs[..., :2] - Xp[..., :2]).norm(dim=-1) * 30.0
        idxr = hreg[None, None].expand(B, Tb, -1)
        both = supp & suppp
        mn = torch.full((B, Tb, L.NR), float('inf'), device=dev).scatter_reduce_(2, idxr, torch.where(both, vt, torch.full_like(vt, float('inf'))), reduce='amin')
        spd = torch.where(torch.isfinite(mn) & okv[..., None], mn, torch.full_like(mn, float('nan')))
        hsp[:, t0b:t0b + nb] = spd[:, :nb]
        dxy = (Xs[..., :2] - Xp[..., :2]) * 30.0 * both[..., None]
        sv = torch.zeros(B, Tb, L.NR, 2, device=dev).index_add_(2, hreg, dxy)
        cn = torch.zeros(B, Tb, L.NR, device=dev).scatter_add_(2, idxr, both.float())
        sv = sv / cn.clamp_min(1)[..., None]
        sv = sv / sv.norm(dim=-1, keepdim=True).clamp_min(1e-9) * torch.nan_to_num(spd, nan=0.0)[..., None]
        hsv[:, t0b:t0b + nb] = sv[:, :nb]
        prev_src = Xs[:, nb - 1].clone(); prev_supp = supp[:, nb - 1].clone()
        del Xs, Xp, supp, suppp, both, vt, dxy, sv
        if hoi:
            po = obj_pose_all[gf]
            oc = object_contact_maps(asset, ob['src'], po, sv.a)
            bank.obj_pose.copy_(po); bank.oc_slot.copy_(oc['oc_slot']); bank.oc_d.copy_(oc['oc_d'])
            bank.oc_vec.copy_(oc['oc_vec']); bank.oc_pt.copy_(oc['oc_pt'])
            dsrc, _ = asset.sdf.sdf(ob['src'], po)
            hoc[:, t0b:t0b + nb] = torch.full((dsrc.shape[0], L.NR), float('inf'), device=dev).scatter_reduce(
                1, hreg[None].expand(dsrc.shape[0], -1), dsrc, reduce='amin').reshape(B, Tb, -1)[:, :nb]
        del V, J, ob
        throttle.tick()
        run.steps(sched, t0b, nb, out, None)
    torch.cuda.synchronize(); tm['stream'] = time.time() - t0
    # ---- LQR, floor, labels
    t0 = time.time()
    piece = sched['piece']; tl = sched['tl']
    plen = torch.as_tensor(pieces[:, 2] - pieces[:, 1], device=dev)
    rem = torch.where(piece >= 0, plen[piece.clamp_min(0)] - tl, torch.zeros_like(tl))
    valid = piece >= 0
    Qf = lqr_chains(out, piece, tl, rem, sv.a, sv.jq.tolist(), sv.clamp)
    clip = sched['clip']
    footmin = hz[..., FEET]; allmin = hz.amin(-1, keepdim=True)
    vel = torch.full_like(footmin, 1e9)
    dz = (footmin[:, 1:] - footmin[:, :-1]).abs() * 30.0
    vel[:, 1:] = torch.where((piece[:, 1:] == piece[:, :-1])[..., None], dz, torch.full_like(dz, 1e9))
    ok = valid[..., None] & (vel < 0.05) & (footmin <= allmin + 0.005)
    floor_clip = torch.zeros(len(clips), device=dev)
    for c in range(len(clips)):
        v_ = footmin[(clip == c)[..., None].expand_as(footmin) & ok]
        if v_.numel() >= 10:
            floor_clip[c] = v_.median()
        else:
            allf = footmin[(clip == c)[..., None].expand_as(footmin)]
            floor_clip[c] = torch.quantile(allf[torch.isfinite(allf)].float(), 0.05) if allf.numel() else 0.0
    floor_bt = floor_clip[clip.clamp_min(0)]
    hcl = hz - floor_bt[..., None]
    reset = valid & (tl == 0)
    inf = torch.full_like(hcl, float('inf'))
    human_g = drop_short_t(hysteresis_t(torch.where(valid[..., None], hcl, inf), 0.01, 0.02, reset), 3, reset)
    human_o = None
    if hoi:
        human_o = drop_short_t(hysteresis_t(torch.where(valid[..., None], hoc, inf), 0.01, 0.02, reset), 3, reset)
    pose_bt = obj_pose_all[sched['gframe'].clamp_min(0)] if hoi else None
    tm['lqr_labels'] = time.time() - t0
    # ---- strict: sequential in time over whole clips (clip-major layout, all clips in lock-step) so corrections are
    # temporally continuous across UMR pieces; the continuity gate (flag 128) checks every frame against UMR's step
    t0 = time.time()
    keepm = valid & sched['keep']
    cfr = sched['cframe']
    C = len(clips); Tmax = int(lens.max())
    ib = torch.zeros(C, Tmax, dtype=torch.long, device=dev); it = torch.zeros(C, Tmax, dtype=torch.long, device=dev)
    vC = torch.zeros(C, Tmax, dtype=torch.bool, device=dev)
    for c in range(C):
        bb, tt_ = torch.nonzero(keepm & (clip == c), as_tuple=True)
        o_ = torch.argsort(cfr[bb, tt_]); bb, tt_ = bb[o_], tt_[o_]
        n_ = len(bb)
        ib[c, :n_] = bb; it[c, :n_] = tt_; vC[c, :n_] = True
    QC_ = Qf[ib, it]; flC = floor_bt[ib, it]; hclC = torch.where(vC[..., None], hcl[ib, it], torch.full_like(hcl[ib, it], float('inf')))
    hspC = hsp[ib, it]; hsvC = hsv[ib, it].double()
    pieceC = torch.where(vC, torch.arange(C, device=dev)[:, None].expand(C, Tmax), torch.full((C, Tmax), -1, dtype=torch.long, device=dev))
    tlC = torch.arange(Tmax, device=dev)[None].expand(C, Tmax).contiguous()
    resetC = vC & (tlC == 0)
    hgC = drop_short_t(hysteresis_t(hclC, 0.01, 0.02, resetC), 3, resetC) & vC[..., None]
    hon = hgC & (hclC <= 0.01)                                     # in contact and on the floor (slip.py ruler)
    hon_prev = torch.zeros_like(hon); hon_prev[:, 1:] = hon[:, :-1]
    hspC = torch.where(hon & hon_prev, hspC, torch.full_like(hspC, float('nan')))   # human region on the floor at t-1 and t
    hocC = hoC = poseC = None
    if hoi:
        hocC = torch.where(vC[..., None], hoc[ib, it], torch.full_like(hoc[ib, it], float('inf')))
        hoC = drop_short_t(hysteresis_t(hocC, 0.01, 0.02, resetC), 3, resetC) & vC[..., None]
        poseC = pose_bt[ib, it]
    Qs, flags, req_g, req_o, info = strict_seq_segmented(proj, QC_, vC, flC, hgC, hoC, poseC, hclC, hocC, hspC, fps=30.0, hsvC=hsvC,
                                                         seg_len=seg_len, overlap=seg_overlap, iters=strict_iters)
    for k_ in ('extras', 'lift_g', 'lift_o'):
        info.pop(k_, None)
    # final independent verification of every output frame (no latches: those are online state, their slip flags and
    # the continuity flags are carried over from the solve)
    fin = torch.zeros(C, Tmax, dtype=torch.int32, device=dev)
    Qflat = Qs.reshape(C * Tmax, -1)
    for s0 in range(0, C * Tmax, 8192):
        sl = slice(s0, s0 + 8192)
        mm_ = proj.measure(Qflat[sl], flC.reshape(-1)[sl], None if poseC is None else poseC.reshape(C * Tmax, -1)[sl],
                           req_g.reshape(C * Tmax, -1)[sl], req_o.reshape(C * Tmax, -1)[sl])
        fin.view(-1)[sl] = mm_['flags']
        del mm_
    flags = torch.where(vC, fin | (flags.to(torch.int32) & (32 | 64 | 128 | SL.FLAG_FOLLOW)), torch.zeros_like(fin))
    info['flagged_final'] = int(((flags & SL.DROP_MASK) != 0).sum())
    torch.cuda.synchronize(); tm['strict'] = time.time() - t0
    # ---- export per clip
    t0 = time.time()
    rows = []
    qa_all = []
    env = dict(engine=engine_version(), spec=SPEC_VERSION, robot=robot, robot_xml=str(GPUOUT / 'robots' / robot / f'{robot}.xml'),
               surface_asset=str(Path(out_root) / '_assets' / robot / 'surface.npz'), static_prep=str(RS.P.path),
               correspondence='UMR beta-zero shared SMPL-X correspondence (reuse_across_smplx_betas)',
               umr_commit='d1cdaed', hoi_overrides=RS.overrides if hoi else {}, qp='fp32 IPM 24 it',
               strict='sequential strict_seq_segmented (256-frame segments, 90-frame overlap, seam at best agreement): continuity box vs UMR step, leash, UMR floor shift, latches at UMR touch-down, slip gate (slip.py ruler; robot slip with a sliding human region = bit 256, kept) + continuity gate (flag 128)')
    for c in range(len(clips)):
        n_ = int(vC[c].sum())
        fl = float(flC[c, 0])
        hum = dict(ground_contact=hgC[c, :n_].cpu().numpy(), ground_clearance_mm=(hclC[c, :n_] * 1000).clamp(-1e4, 1e4).half().cpu().numpy(),
                   ground_slip_speed=hspC[c, :n_].float().cpu().numpy())
        pose_c = None
        if hoi:
            hum.update(object_contact=hoC[c, :n_].cpu().numpy(),
                       object_clearance_mm=((hocC[c, :n_]).clamp(-10, 10) * 1000).half().cpu().numpy())
            pose_c = poseC[c, :n_]
        ex, qc = export_clip(proj, Qs[c, :n_], fl, 30.0, flags[c, :n_], pose_c, hum, qref=QC_[c, :n_])
        ex['human_scale'] = np.array(float(scale_c[c]), np.float32)
        ex['frame_ids_in_source'] = data[c][5].astype(np.int32)
        ex['source_fps'] = np.array(float(data[c][7]), np.float32)
        if hoi:
            op = pose_c.cpu().numpy().copy(); op[:, 2] -= fl
            ex['object_pose'] = op[:, :7].astype(np.float32)                 # floor-relative pos + quat wxyz
            ex['object_scale'] = np.array(float(scale_c[c]), np.float32)
            ex['object_id'] = np.array(job['object']['name'])
            ex['object_xml'] = np.array(job['object']['xml'])
        in_sha = sha_files(data[c][6] + ([Path(clips[c]['object_prop'])] if hoi else []))
        prov = dict(env, clip_id=clips[c]['clip_id'], source=job['source'], source_dir=clips[c]['dir'], input_sha1=in_sha,
                    created=time.strftime('%Y-%m-%d %H:%M:%S'), job_id=job['job_id'])
        ex['provenance'] = np.array(json.dumps(prov))
        qc.update(clip_id=clips[c]['clip_id'], robot=robot, source=job['source'], kept_frames=int(((ex['flags'] & SL.DROP_MASK) == 0).sum()),
                  robot_seconds=round(len(ex['qpos']) / 30.0, 2), kept_robot_seconds=round(int(((ex['flags'] & SL.DROP_MASK) == 0).sum()) / 30.0, 2))
        qa_all.append(qc)
        if write:
            od = Path(out_root) / job['source'] / clips[c]['clip_id']
            od.mkdir(parents=True, exist_ok=True)
            tmp = od / f'.{robot}.tmp.npz'
            np.savez_compressed(tmp, **ex, qc_json_full=np.array(json.dumps(qc)))
            os.replace(tmp, od / f'{robot}.npz')
            rows.append(dict(qc, path=str(od / f'{robot}.npz'), engine=env['engine'], spec=SPEC_VERSION, input_sha1=in_sha))
    torch.cuda.synchronize(); tm['export'] = time.time() - t0
    tm['total'] = time.time() - t_job
    frames = int(lens.sum())
    summary = dict(job_id=job['job_id'], robot=robot, source=job['source'], clips=len(clips), frames=frames,
                   robot_hours=round(frames / 30 / 3600, 4), kept_frames=int(sum(q['kept_frames'] for q in qa_all)),
                   seconds={k: round(v, 1) for k, v in tm.items()}, strict_info=info, B=B, Tch=Tch,
                   robot_frames_per_s=round(frames / tm['total'], 1))
    summary['kept_frac'] = round(summary['kept_frames'] / max(frames, 1), 4)
    return summary, rows, qa_all
