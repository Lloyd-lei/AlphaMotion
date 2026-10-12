"""End-to-end GPU retarget of many (clip, robot) sequences: UMR solve (CUDA graph) -> LQR -> strict pass -> flags.

    from surface_retargeting.gpu.pipeline import Retargeter
    R = Retargeter(bank, surface_npz)
    out = R.run(prob, start, length)          # tensors on the GPU
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu.geometry import SelfCollision, ObjectSDF, object_from_objpen
from surface_retargeting.gpu.solver import UMRSolver, lqr_filter
from surface_retargeting.strict.strict import RobotPoints, StrictProjector, PEN_TOL, CONTACT_TOL
from surface_retargeting.gpu.geometry import CollisionGround
from surface_retargeting.strict.strict_fp import strict_chains, strict_seq
from surface_retargeting.gpu import slip as SL

FEET = [L.RID['L_foot'], L.RID['R_foot']]


def problem_fps(problem):
    a = problem.args
    try:
        z = np.load(a['out'], allow_pickle=True)
        return float(z['fps'][0])
    except Exception:
        return 30.0


class Retargeter:
    def __init__(self, bank, surface_npz, strict=True, self_collision=True, device='cuda', strict_kw=None):
        self.bank = bank
        self.sv = UMRSolver(bank)
        kt = self.sv.kt
        self.kt = kt
        self.pts = RobotPoints(kt, surface_npz, device)
        self.obj = None
        P0 = bank.problems[0]
        if bank.has_obj and 'objpen_mjb' in P0.z.files:
            V, F, _ = object_from_objpen(str(P0['objpen_mjb']), P0['objpen_object_geoms'])
            self.obj = ObjectSDF(V, F, device)
        self.sc = SelfCollision(kt, device) if self_collision else None
        self.strict = strict
        cov = np.unique(np.load(surface_npz)['hull_geom'])
        self.proj = StrictProjector(kt, self.pts, self.sc, self.obj, lim_lo=self.sv.j_lo, lim_hi=self.sv.j_hi,
                                    jq=self.sv.jq, jd=self.sv.jd, colground=CollisionGround(kt, device, covered=cov),
                                    **(strict_kw or {}))
        self._human = {}
        self.async_strict = True
        self.strict_iters = 12
        self.strict_mode = 'seq'

    # ------------------------------------------------------------------ human-side labels (per problem, cached)
    def human(self, pi):
        if pi in self._human:
            return self._human[pi]
        P = self.bank.problems[pi]
        t0 = time.time()
        H = P['src'].astype(np.float64)                                  # [T,4096,3]
        hreg = L.part_to_region(P['part_ids'])
        fps = problem_fps(P)
        floor, n_static, how = L.floor_from_points(H, hreg, FEET, fps)
        dg, vg, cg = L.ground_inputs(H, floor, hreg, fps)
        lab = L.label(dg, vg, hreg, cg)
        out = dict(floor=floor, floor_samples=n_static, floor_method=how, fps=fps, g_contact=lab['contact'],
                   g_slip=lab['slip'], g_clear=cg, g_speed=lab['slip_speed'],
                   )
        out['g_slip_speed'], out['g_slip_vec'] = SL.human_slip_speed(H, floor, hreg, lab['contact'], fps, return_vec=True)  # slip.py ruler (t-1 -> t)
        if self.obj is not None:
            pose = torch.as_tensor(P['obj_pose'], device=self.bank.device)
            T = len(H)
            d = np.zeros(H.shape[:2]); n = np.zeros(H.shape); Ql = np.zeros(H.shape)
            from surface_retargeting.gpu.kin import quat_to_mat
            for s in range(0, T, 256):
                Pw = torch.as_tensor(H[s:s + 256], device=self.bank.device, dtype=torch.float32)
                dd, nn = self.obj.sdf(Pw, pose[s:s + 256])
                d[s:s + 256] = dd.cpu().numpy(); n[s:s + 256] = nn.cpu().numpy()
                R = quat_to_mat(pose[s:s + 256, 3:7].float())
                Ql[s:s + 256] = ((Pw - pose[s:s + 256, None, :3].float())[..., None, :] @ R[:, None])[..., 0, :].cpu().numpy()
            rot = quat_to_mat(pose[:, 3:7].float()).cpu().numpy().astype(np.float64)
            do, vo, co = L.object_inputs(np.where(d < 0.05 + 0.2, d, np.inf), Ql, rot, n, hreg, fps)
            labo = L.label(do, vo, hreg, co)
            out.update(o_contact=labo['contact'], o_slip=labo['slip'], o_clear=co)
        out['seconds'] = time.time() - t0
        self._human[pi] = out
        return out

    # ------------------------------------------------------------------ full pipeline
    def run(self, prob, start, length, graph=True, log_every=0, timings=None):
        b = self.bank; sv = self.sv
        dev = b.device
        B = prob.shape[0]
        tm = {} if timings is None else timings
        torch.cuda.synchronize(); t0 = time.time()
        if graph:
            Q, st = sv.run_graph(prob, start, length, log_every=log_every)
        else:
            Q, st = sv.run(prob, start, length, log_every=log_every)
        torch.cuda.synchronize(); tm['umr_solve'] = time.time() - t0
        t1 = time.time()
        Qf = lqr_filter(Q, length, sv.jq.tolist(), sv.a, sv.clamp)
        torch.cuda.synchronize(); tm['lqr'] = time.time() - t1
        out = dict(q_umr_raw=Q, q_umr=Qf)
        if not self.strict:
            return out
        t2 = time.time()
        T = Qf.shape[1]
        # windows of per-problem human labels
        pr = prob.cpu().numpy(); stt = start.cpu().numpy(); ln = length.cpu().numpy()
        NR = L.NR
        hg = np.zeros((B, T, NR), bool); hs = np.zeros((B, T, NR), bool); ho = np.zeros((B, T, NR), bool)
        floor = np.zeros(B)
        for i in range(B):
            h = self.human(int(pr[i]))
            sl = slice(stt[i], stt[i] + ln[i])
            hg[i, :ln[i]] = h['g_contact'][sl]; hs[i, :ln[i]] = h['g_slip'][sl]
            if 'o_contact' in h:
                ho[i, :ln[i]] = h['o_contact'][sl]
            floor[i] = h['floor']
        tm['human_labels'] = time.time() - t2
        floor_t = torch.as_tensor(floor, device=dev)
        fi = (b.off[prob][:, None] + torch.minimum(start[:, None] + torch.arange(T, device=dev)[None], b.T[prob][:, None] - 1))
        poses = b.obj_pose[fi] if (self.obj is not None) else None
        valid = np.arange(T)[None] < ln[:, None]
        tt = lambda a: torch.as_tensor(a, device=dev)
        t4 = time.time()
        # frame-parallel strict pass (batched over all frames of all sequences)
        piece = torch.where(torch.arange(T, device=dev)[None] < length[:, None], torch.arange(B, device=dev)[:, None].expand(B, T),
                            torch.full((B, T), -1, dtype=torch.long, device=dev))
        tl = torch.arange(T, device=dev)[None].expand(B, T).contiguous()
        hcg = np.full((B, T, NR), np.inf); hco = np.full((B, T, NR), np.inf); hsp = np.full((B, T, NR), np.nan)
        hsv = np.zeros((B, T, NR, 2))
        for i in range(B):
            h = self.human(int(pr[i]))
            hcg[i, :ln[i]] = h['g_clear'][stt[i]:stt[i] + ln[i]] - 0.0      # g_clear is already floor-relative
            hsp[i, :ln[i]] = h['g_slip_speed'][stt[i]:stt[i] + ln[i]]
            hsv[i, :ln[i]] = h['g_slip_vec'][stt[i]:stt[i] + ln[i]]
            if 'o_clear' in h:
                hco[i, :ln[i]] = h['o_clear'][stt[i]:stt[i] + ln[i]]
        self.proj.set_fps(self.human(int(pr[0]))['fps'])
        Q1, flags, flags0, req_g_t, req_o_t, info = (strict_seq if self.strict_mode == 'seq' else strict_chains)(
            self.proj, Qf, piece, tl, floor_t[:, None].expand(B, T).contiguous(), tt(hg), tt(ho) if self.obj is not None else None,
            pose=poses, iters=self.strict_iters, human_clear_g=tt(hcg).float(), human_clear_o=tt(hco).float() if self.obj is not None else None,
            fps=self.human(int(pr[0]))['fps'], human_slip=tt(hs), human_speed=tt(hsp).float(), human_slip_vec=tt(hsv).double())
        ex = info.pop('extras'); lift_g = info.pop('lift_g'); lift_o = info.pop('lift_o')
        req_g = req_g_t.cpu().numpy(); req_o = req_o_t.cpu().numpy()
        torch.cuda.synchronize(); tm['strict'] = time.time() - t4
        # final independent GPU verification of the output (all gauges incl. self collision)
        t5 = time.time()
        nq = Q1.shape[-1]
        Qflat = Q1.reshape(B * T, nq); fl_flat = floor_t[:, None].expand(B, T).reshape(-1)
        rgf = tt(req_g).reshape(B * T, -1); rof = tt(req_o).reshape(B * T, -1)
        psf = None if poses is None else poses.reshape(B * T, -1)
        fin = torch.zeros(B * T, dtype=torch.int32, device=dev)
        CH = 16384
        for c in range(0, B * T, CH):
            sl = slice(c, c + CH)
            exs = {k: (None if v is None else (tuple(t[sl] for t in v) if isinstance(v, tuple) else v[sl])) for k, v in ex.items()}
            mm = self.proj.measure(Qflat[sl], fl_flat[sl], None if psf is None else psf[sl], rgf[sl], rof[sl], **exs)
            fin[sl] = mm['flags']
        fin = fin.reshape(B, T) | (flags.to(fin.dtype) & 128)          # keep the temporal-continuity gate flag
        if not ex:                                                     # sequential mode: latches are online state
            fin = fin | (flags.to(fin.dtype) & (32 | 64 | SL.FLAG_FOLLOW))
        fin = torch.where(tt(valid), fin, torch.zeros_like(fin))
        iters = None
        tm['verify'] = time.time() - t5
        Qout = Q1.clone()
        Qout[..., 2] -= floor_t[:, None]
        out.update(q_strict_world=Q1, qpos=Qout, flags=fin, flags_solver=flags, flags_initial=flags0,
                   req_g=req_g, req_o=req_o, lift_g=lift_g, latch=ex.get('latch'), floor=floor, strict_info=info)
        return out
