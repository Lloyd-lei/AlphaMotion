"""Problem bank: captured UMR per-frame problems (see capture/umr_capture_run.py) packed into GPU tensors.

One bank holds P problems (clips) for ONE robot model.  Batch element b of a solve reads problem prob[b] at frame
start[b] + t; nothing is copied per element, everything is gathered on the GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch


def _pad_stack(arrs, fill, dtype):
    n = max(a.shape[1] if a.ndim > 1 else 0 for a in arrs) if arrs else 0
    out = []
    for a in arrs:
        if a.shape[1] < n:
            pad = np.full((a.shape[0], n - a.shape[1]) + a.shape[2:], fill, a.dtype)
            a = np.concatenate([a, pad], 1)
        out.append(a)
    return np.concatenate(out, 0).astype(dtype)


class Problem:
    """Host-side view of one captured npz."""

    def __init__(self, path):
        self.path = str(path)
        z = np.load(path, allow_pickle=False)
        self.z = z
        self.args = json.loads(str(z['args_json']))
        self.T = int(z['q_init'].shape[0])
        self.mjb = str(z['mjb'])
        self.has_obj = 'obj_pose' in z.files

    def __getitem__(self, k):
        return self.z[k]


class Bank:
    def __init__(self, problems, device='cuda', fdt=torch.float32):
        self.problems = problems
        self.device = torch.device(device)
        P0 = problems[0]
        self.model = mujoco.MjModel.from_binary_path(P0.mjb)
        self.args = P0.args
        for p in problems[1:]:
            m = mujoco.MjModel.from_binary_path(p.mjb)
            assert m.nq == self.model.nq and m.ngeom == self.model.ngeom, 'bank must hold one robot model'
        f = dict(device=self.device, dtype=fdt)
        d64 = dict(device=self.device, dtype=torch.float64)
        i64 = dict(device=self.device, dtype=torch.long)
        self.nprob = len(problems)
        self.T = torch.as_tensor([p.T for p in problems], **i64)
        self.off = torch.as_tensor(np.concatenate([[0], np.cumsum([p.T for p in problems])[:-1]]), **i64)
        # ---- static per problem
        S = max(len(p['selected']) for p in problems)
        self.S = S
        sel = np.full((self.nprob, S), -1, np.int64)
        for i, p in enumerate(problems):
            sel[i, :len(p['selected'])] = p['selected']
        self.sel = torch.as_tensor(sel, **i64)
        self.sel_valid = self.sel >= 0
        self.tmpl_geom = torch.as_tensor(np.stack([p['tmpl_geom'] for p in problems]).astype(np.int64), **i64)
        self.tmpl_local = torch.as_tensor(np.stack([p['tmpl_local_pos'] for p in problems]), **f)
        self.tmpl_normal = torch.as_tensor(np.stack([p['tmpl_local_normal'] for p in problems]), **f)
        self.nslot = self.tmpl_geom.shape[1]
        sc = np.zeros((self.nprob, S)); nc = np.zeros((self.nprob, S))
        for i, p in enumerate(problems):
            s = p['selected']; sc[i, :len(s)] = p['point_cost'][s]; nc[i, :len(s)] = p['normal_cost'][s]
        self.point_cost = torch.as_tensor(sc, **d64); self.normal_cost = torch.as_tensor(nc, **d64)
        self.lim_lo = torch.as_tensor(P0['lim_lo'], **d64); self.lim_hi = torch.as_tensor(P0['lim_hi'], **d64)
        self.part_ids = torch.as_tensor(np.stack([p['part_ids'] for p in problems]).astype(np.int64), **i64)
        # ---- per frame (concatenated over problems)
        cat = lambda k, dt=None: np.concatenate([p[k] for p in problems], 0) if dt is None else np.concatenate([p[k] for p in problems], 0).astype(dt)
        src_sel = []
        for i, p in enumerate(problems):
            src_sel.append(p['src'][:, p['selected']])
        self.src_sel = torch.as_tensor(_pad_stack(src_sel, 0.0, np.float32), **f)
        self.ntgt = torch.as_tensor(_pad_stack([p['ntgt'] for p in problems], 0.0, np.float32), **f)
        self.q_init = torch.as_tensor(cat('q_init'), **d64)
        self.iters = torch.as_tensor(cat('iters'), **i64)
        self.has_prev = torch.as_tensor(cat('has_prev'), device=self.device)
        self.q_cpu = torch.as_tensor(cat('q_cpu'), **d64)
        self.has_ground = all('gd' in p.z.files for p in problems)
        if self.has_ground:
            self.gd = torch.as_tensor(_pad_stack([p['gd'] for p in problems], np.inf, np.float64), **d64)
            self.gw = torch.as_tensor(_pad_stack([p['gw'] for p in problems], np.inf, np.float64), **d64)
        # self-contact maps -> positions inside the selected list
        self.has_sc = all('sc_pairs' in p.z.files for p in problems)
        if self.has_sc:
            pairs, dd, ww, dirs = [], [], [], []
            for i, p in enumerate(problems):
                s = p['selected']; pos = np.full(self.nslot, -1, np.int64); pos[s] = np.arange(len(s))
                pr = p['sc_pairs'].astype(np.int64)
                valid = pr[..., 0] >= 0
                pp = np.where(valid[..., None], pos[np.maximum(pr, 0)], -1)
                assert (pp[valid] >= 0).all(), 'self-contact slot outside selected set'
                src = p['src']
                a = np.take_along_axis(src, np.maximum(pr[..., 0], 0)[..., None].repeat(3, -1), 1)
                b = np.take_along_axis(src, np.maximum(pr[..., 1], 0)[..., None].repeat(3, -1), 1)
                dv = a - b
                dv = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-12)
                pairs.append(pp); dd.append(p['sc_d']); ww.append(p['sc_w']); dirs.append(dv)
            self.sc_pairs = torch.as_tensor(_pad_stack(pairs, -1, np.int64), **i64)
            self.sc_d = torch.as_tensor(_pad_stack(dd, 0.0, np.float64), **d64)
            self.sc_w = torch.as_tensor(_pad_stack(ww, 0.0, np.float64), **d64)
            self.sc_dir = torch.as_tensor(_pad_stack(dirs, 0.0, np.float32), **f)
        self.has_obj = all(p.has_obj for p in problems)
        if self.has_obj:
            self.obj_pose = torch.as_tensor(cat('obj_pose'), **d64)
            self.oc_slot = torch.as_tensor(_pad_stack([p['oc_slot'].astype(np.int64) for p in problems], -1, np.int64), **i64)
            self.oc_d = torch.as_tensor(_pad_stack([p['oc_d'] for p in problems], 0.0, np.float64), **d64)
            self.oc_vec = torch.as_tensor(_pad_stack([p['oc_vec'] for p in problems], 0.0, np.float64), **d64)
            self.oc_pt = torch.as_tensor(_pad_stack([p['oc_pt'] for p in problems], 0.0, np.float64), **d64)

    def frame_index(self, prob, frame):
        return self.off[prob] + frame
