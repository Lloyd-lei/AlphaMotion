"""GPU source preprocessing: SMPL-X forward + every per-frame UMR solver input, batched over frames.

Static data per (subject shape, robot) comes from a UMR static capture (capture/umr_capture_run.py with the x_*
extras: slot binding to the SMPL-X template mesh, robot T-pose normals in the SMPL frame, selected slots, part ids,
costs, args).  Per frame, on the GPU (identical formulas to UMR):
  vertices      SMPL-X forward (UMR's own smplx layer; hands zeroed as zero_source_finger_pose), x smpl_scale
  source slots  barycentric interpolation on the bound faces              (dynamic_surface_template_to_world)
  normal target triangle-frame transport of the robot T-pose normal       (transport_tpose_robot_normals)
  ground        z, snapped >=0 distances, weight distances = z - min_z     (compute_source_slot_ground_contact)
  self contact  threshold_global_topk (<= 0.1 m, 256 closest) U source_clearance_body_pair_topk (closest pair per
                non-adjacent segment pair, rank weights 1 -> 0.1), merged with max weight (merge_self_contact_maps)
  q_init        pelvis position + hip heading on the zero-joint pose        (initial_qpos_for_frame)
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import json
import sys
from pathlib import Path

import numpy as np
import torch

UMR = Path(f'{SRCFG.umr_root}')


def _umr_path():
    for p in (str(UMR / 'scripts'), str(UMR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def normalize(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def triangle_frames(tri):
    """tri [...,3,3] (vertex, xyz) -> [...,3,3] with columns e1, e2, n (UMR triangle_frames)."""
    e1 = normalize(tri[..., 1, :] - tri[..., 0, :])
    n = normalize(torch.linalg.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :], dim=-1))
    e2 = normalize(torch.linalg.cross(n, e1, dim=-1))
    return torch.stack([e1, e2, n], -1)


class SMPLXForward:
    """Batched SMPL-X forward with UMR's model loader (one fixed batch size; inputs padded)."""

    def __init__(self, gender='neutral', betas=None, batch=4096, device='cuda', model_dir=None):
        _umr_path()
        import smpl_surface_retarget_common as common
        self.model_dir = Path(model_dir or (UMR / 'smpl'))
        self.batch = batch
        self.dev = torch.device(device)
        self._common = common; self.gender = gender
        self.models = {}
        self.model = self._get(batch)
        b = np.zeros(10, np.float32)
        if betas is not None:
            bb = np.asarray(betas, np.float32).reshape(-1)[:10]; b[:len(bb)] = bb
        self.betas = torch.as_tensor(b, device=self.dev)[None].expand(batch, -1).contiguous()
        self.faces = torch.as_tensor(np.asarray(self.model.faces, np.int64), device=self.dev)

    def _get(self, n):
        if n not in self.models:
            self.models[n] = self._common.build_smplx_model(self.model_dir, self.gender, n, device=self.dev)
        return self.models[n]

    @torch.no_grad()
    def __call__(self, pose_aa, transl, betas=None):
        """pose_aa [F,>=66] axis-angle, transl [F,3] (torch on device) -> vertices [F,V,3], joints [F,J,3].
        All SMPL-X inputs are passed explicitly (zeros for hands/jaw/eyes/expression, as UMR with
        zero_source_finger_pose), so any batch size <= self.batch works without padding."""
        F = pose_aa.shape[0]
        Vs, Js = [], []
        for s in range(0, F, self.batch):
            e = min(F, s + self.batch); n0 = e - s
            n = min(self.batch, 1 << (n0 - 1).bit_length())          # model batch: next power of two
            pa = torch.zeros(n, pose_aa.shape[1], device=self.dev); pa[:n0] = pose_aa[s:e]
            tr = torch.zeros(n, 3, device=self.dev); tr[:n0] = transl[s:e]
            model = self._get(n)
            z3 = torch.zeros(n, 3, device=self.dev)
            zh = torch.zeros(n, 45, device=self.dev)
            ne = int(getattr(self.model, 'num_expression_coeffs', 10))
            if betas is not None:
                bt = torch.zeros(n, 10, device=self.dev); bt[:n0] = betas[s:e].float()
            else:
                bt = self.betas[:n]
            out = model(global_orient=pa[:, 0:3], body_pose=pa[:, 3:66], betas=bt, transl=tr,
                             left_hand_pose=zh, right_hand_pose=zh, jaw_pose=z3, leye_pose=z3, reye_pose=z3,
                             expression=torch.zeros(n, ne, device=self.dev), return_verts=True)
            Vs.append(out.vertices[:n0]); Js.append(out.joints[:n0, :25])
        return torch.cat(Vs), torch.cat(Js)


class SourcePrep:
    def __init__(self, problem, device='cuda'):
        _umr_path()
        z = problem.z
        dev = torch.device(device)
        self.dev = dev
        self.args = problem.args
        a = self.args
        f = dict(device=dev, dtype=torch.float32)
        self.faces = torch.as_tensor(z['x_template_faces'].astype(np.int64), device=dev)
        self.face_ids = torch.as_tensor(z['x_bind_face_ids'].astype(np.int64), device=dev)
        self.bary = torch.as_tensor(z['x_bind_bary'], **f)
        self.slot_faces = self.faces[self.face_ids]                                     # [N,3]
        sel = z['selected'].astype(np.int64)
        self.sel = torch.as_tensor(sel, device=dev)
        tv = torch.as_tensor(z['x_template_vertices'], **f)
        self.template_basis = triangle_frames(tv[self.slot_faces[self.sel]])             # [S,3,3]
        rn = torch.as_tensor(z['x_robot_tpose_normals_smpl'], **f)
        self.robot_n = normalize(rn[self.sel])                                          # [S,3]
        import smpl_surface_retarget_common as common
        self.up = str(problem.args.get('_source_up', 'z'))
        try:
            self.up = str(load_smplx_sequence(problem).get('output_up', 'z')).lower()
        except Exception:
            pass
        self.up_mat = None
        if self.up == 'y':
            self.up_mat = torch.as_tensor(np.asarray(common.Y_UP_TO_Z_UP_MATRIX, np.float32), **f)
        v0 = z['x_vertices_world0'].astype(np.float64); v1 = z['x_vertices_scaled0'].astype(np.float64)
        if self.up == 'y':
            v0 = v0 @ np.asarray(common.Y_UP_TO_Z_UP_MATRIX, np.float64).T
        self.scale = float(np.median(v1[0][np.abs(v0[0]) > 1e-3] / v0[0][np.abs(v0[0]) > 1e-3]))
        self.snap = float(a['ground_contact_map_snap_threshold'])
        # human sole slots (feet regions, lowest 2 cm of each foot in the SMPL-X template; template is y-up)
        from surface_retargeting.gpu import labels as L
        reg = L.part_to_region(z['part_ids'])
        tvn = z['x_template_vertices'].astype(np.float64); tf = z['x_template_faces'].astype(np.int64)
        tslot = (z['x_bind_bary'][..., None] * tvn[tf[z['x_bind_face_ids']]]).sum(1)
        soles = []
        for r in (L.RID['L_foot'], L.RID['R_foot']):
            k = np.flatnonzero(reg == r)
            soles.append(k[tslot[k, 1] <= tslot[k, 1].min() + 0.02])
        ns = max(len(x) for x in soles)
        sp = np.stack([np.concatenate([x, np.repeat(x[:1], ns - len(x))]) for x in soles])
        self.sole_slots = torch.as_tensor(sp, device=dev)                     # [2,Ns] (padded by repetition)
        self.sole_count = [len(x) for x in soles]
        # zero-joint pose for q_init (joints of UMR's initial_qpos_for_frame are q_body_zero)
        self.q_body_zero = torch.as_tensor(z['q_init'][0], device=dev, dtype=torch.float64)
        # self-contact static structure (UMR functions on the captured part ids)
        self.sc_on = float(a['self_contact_map_cost']) > 0
        if self.sc_on:
            import retarget_body_segment_surface as bss
            import retarget_smpl_to_humanoid_surface_vector as M
            modes = M.parse_self_contact_map_modes(a['self_contact_map_mode'])
            assert set(modes) <= {'threshold_global_topk', 'source_clearance_body_pair_topk'}
            self.modes = modes
            part = z['part_ids'].astype(np.int64)
            mask = bss.non_adjacent_segment_pair_mask(part, sel)
            ui, uj = np.where(mask)
            self.ui = torch.as_tensor(ui, device=dev); self.uj = torch.as_tensor(uj, device=dev)
            pp = np.sort(np.stack([part[sel[ui]], part[sel[uj]]], 1), 1)
            ub, gid = np.unique(pp, axis=0, return_inverse=True)
            topk = bss.parse_body_topk_config(a.get('self_contact_map_body_topk')) if hasattr(bss, 'parse_body_topk_config') else {}
            dflt = topk.get('default', topk.get('*', 1))
            keep = [max(int(topk.get(name_of(bss, p0), dflt)), int(topk.get(name_of(bss, p1), dflt))) for p0, p1 in ub]
            assert all(k == 1 for k in keep), 'only body top-1 implemented'
            self.gid = torch.as_tensor(gid.reshape(-1), device=dev)
            self.ngroup = len(ub)
            self.thr = float(a['self_contact_map_threshold']); self.maxp = int(a['self_contact_map_max_pairs'])
            self.K = (self.maxp if 'threshold_global_topk' in modes else 0) + (self.ngroup if 'source_clearance_body_pair_topk' in modes else 0)
            self.scw = None
            try:
                from surface_retargeting.gpu.selfcontact_warp import SelfContactMaps
                self.scw = SelfContactMaps(ui, uj, gid.reshape(-1), self.thr, self.maxp, device=dev)
            except Exception as e:                       # pragma: no cover
                print('[source_gpu] warp self-contact unavailable, torch path:', e)

    @torch.no_grad()
    def set_clips(self, smplx, betas, ups):
        """Shared beta-zero correspondence (UMR reuse_across_smplx_betas): the slot binding (faces + bary) is fixed;
        per clip only the body shape changes -> smpl_scale = robot_height / template height, and the template
        triangle frames used to transport the robot T-pose normals.  betas [C,10] (tensor), ups: list of 'y'/'z'."""
        import smpl_surface_retarget_common as common
        C = betas.shape[0]
        z55 = torch.zeros(C, 165, device=self.dev); zt = torch.zeros(C, 3, device=self.dev)
        V0, _ = smplx(z55, zt, betas)
        hh = V0[..., 1].amax(1) - V0[..., 1].amin(1)
        self.clip_scale = float(self.args['robot_height']) / hh                      # [C]
        self.clip_basis = triangle_frames(V0[:, self.slot_faces[self.sel]])             # [C,S,3,3]
        Y = torch.as_tensor(np.asarray(common.Y_UP_TO_Z_UP_MATRIX, np.float32), device=self.dev)
        I = torch.eye(3, device=self.dev)
        self.clip_up = torch.stack([Y if str(u).lower() == 'y' else I for u in ups])  # [C,3,3]
        return self.clip_scale

    @torch.no_grad()
    def frames(self, V, J, chunk=2048, clip=None):
        """V [F,Nv,3] unscaled world vertices (retarget frame), J [F,Nj,3] -> dict of solver inputs (fp32/fp64)."""
        F = V.shape[0]
        out = {k: [] for k in ('src', 'src_sel', 'ntgt', 'gd', 'gw', 'sc_pairs', 'sc_d', 'sc_w', 'sc_dir', 'q_init', 'sole_z', 'sole_xy')}
        for s in range(0, F, chunk):
            Vs = V[s:s + chunk]; Js = J[s:s + chunk]
            if clip is not None:
                c = clip[s:s + chunk]
                Um = self.clip_up[c]; sc = self.clip_scale[c][:, None, None]
                Vs = (Vs @ Um.transpose(1, 2)) * sc; Js = (Js @ Um.transpose(1, 2)) * sc
                basis = self.clip_basis[c]
            else:
                if self.up_mat is not None:
                    Vs = Vs @ self.up_mat.T; Js = Js @ self.up_mat.T
                Vs = Vs * self.scale; Js = Js * self.scale
                basis = self.template_basis[None]
            tri = Vs[:, self.slot_faces]                                                # [f,N,3,3]
            src = (self.bary[None, :, :, None] * tri).sum(2)                            # [f,N,3]
            out['src'].append(src)
            sole = src[:, self.sole_slots.reshape(-1)].reshape(src.shape[0], 2, -1, 3)
            out['sole_z'].append(sole[..., 2].median(-1).values)
            out['sole_xy'].append(sole[..., :2].mean(-2))
            src_sel = src[:, self.sel]
            out['src_sel'].append(src_sel)
            fb = triangle_frames(tri[:, self.sel])
            rot = fb @ basis.transpose(-1, -2)
            out['ntgt'].append(normalize((rot @ self.robot_n[None, ..., None])[..., 0]))
            raw = src[..., 2]
            gw = raw - raw.min(1, keepdim=True).values
            gd = raw.clamp_min(0)
            gd = torch.where(gd < self.snap, torch.zeros_like(gd), gd)
            out['gd'].append(gd[:, self.sel].double()); out['gw'].append(gw[:, self.sel].double())
            # q_init
            q = self.q_body_zero[None].repeat(len(Vs), 1)
            q[:, 0:3] = Js[:, 0].double()
            left = (Js[:, 1] - Js[:, 2]).double(); left[:, 2] = 0
            ln = left.norm(dim=-1, keepdim=True)
            left = torch.where(ln > 1e-6, left / ln.clamp_min(1e-12), torch.tensor([0.0, 1.0, 0.0], dtype=left.dtype, device=left.device))
            up = torch.tensor([0.0, 0.0, 1.0], dtype=left.dtype, device=left.device).expand_as(left)
            fwd = torch.linalg.cross(left, up, dim=-1); fwd = normalize(fwd)
            left = torch.linalg.cross(up, fwd, dim=-1)
            R = torch.stack([fwd, left, up], -1)
            q[:, 3:7] = mat_to_quat_wxyz(R)
            out['q_init'].append(q)
            if self.sc_on and self.scw is not None:
                P = src_sel
                r = self.scw(P)
                nf = P.shape[0]
                pid, dd, ww = [], [], []
                if 'threshold_global_topk' in self.modes:
                    ok = r['t_pair'] >= 0
                    pid.append(r['t_pair']); dd.append(r['t_d']); ww.append(ok.float())
                    in_thr = (r['g_pair'][:, :, None] == r['t_pair'][:, None, :]).any(-1)       # [f,G]
                else:
                    in_thr = torch.zeros(nf, self.ngroup, dtype=torch.bool, device=P.device)
                if 'source_clearance_body_pair_topk' in self.modes:
                    gmin = r['g_min']; gam = r['g_pair']
                    rank = torch.argsort(torch.argsort(gmin, dim=1, stable=True), dim=1).float()
                    G = self.ngroup
                    w = 1.0 - 0.9 * rank / max(G - 1, 1) if G > 1 else torch.ones_like(rank)
                    pid.append(torch.where(in_thr, torch.full_like(gam, -1), gam)); dd.append(torch.where(in_thr, torch.zeros_like(gmin), gmin))
                    ww.append(torch.where(in_thr, torch.zeros_like(w), w))
                pid = torch.cat(pid, 1); dd = torch.cat(dd, 1); ww = torch.cat(ww, 1)
                valid = pid >= 0
                pc = pid.clamp_min(0)
                pairs = torch.stack([self.ui[pc], self.uj[pc]], -1)
                pairs = torch.where(valid[..., None], pairs, torch.full_like(pairs, -1))
                a_ = torch.gather(P, 1, pairs[..., 0].clamp_min(0)[..., None].expand(-1, -1, 3))
                b_ = torch.gather(P, 1, pairs[..., 1].clamp_min(0)[..., None].expand(-1, -1, 3))
                out['sc_pairs'].append(pairs); out['sc_d'].append(dd.double()); out['sc_w'].append(ww.double())
                out['sc_dir'].append(normalize(a_ - b_))
            elif self.sc_on:
                P = src_sel
                d = (P[:, self.ui] - P[:, self.uj]).norm(dim=-1)                          # [f,Np]
                nf = d.shape[0]
                pid, dd, ww = [], [], []
                if 'threshold_global_topk' in self.modes:
                    key = torch.where(d <= self.thr, d, torch.full_like(d, float('inf')))
                    k = min(self.maxp, key.shape[1])
                    vv, ii = torch.topk(key, k, dim=1, largest=False)
                    ok = torch.isfinite(vv)
                    pid.append(torch.where(ok, ii, torch.full_like(ii, -1))); dd.append(torch.where(ok, vv, torch.zeros_like(vv)))
                    ww.append(ok.float())
                    in_thr = torch.zeros_like(d, dtype=torch.bool).scatter_(1, torch.where(ok, ii, torch.zeros_like(ii)), ok)
                else:
                    in_thr = torch.zeros_like(d, dtype=torch.bool)
                if 'source_clearance_body_pair_topk' in self.modes:
                    gmin = torch.full((nf, self.ngroup), float('inf'), device=d.device).scatter_reduce(
                        1, self.gid[None].expand(nf, -1), d, reduce='amin')
                    hit = d == gmin.gather(1, self.gid[None].expand(nf, -1))
                    idxf = torch.where(hit, torch.arange(d.shape[1], device=d.device, dtype=torch.float32)[None],
                                       torch.full_like(d, 3e38))
                    gam = torch.full((nf, self.ngroup), 3e38, device=d.device).scatter_reduce(
                        1, self.gid[None].expand(nf, -1), idxf, reduce='amin').long()
                    rank = torch.argsort(torch.argsort(gmin, dim=1), dim=1).float()
                    G = self.ngroup
                    w = 1.0 - 0.9 * rank / max(G - 1, 1) if G > 1 else torch.ones_like(rank)
                    dup = in_thr.gather(1, gam)
                    pid.append(torch.where(dup, torch.full_like(gam, -1), gam)); dd.append(torch.where(dup, torch.zeros_like(gmin), gmin))
                    ww.append(torch.where(dup, torch.zeros_like(w), w))
                pid = torch.cat(pid, 1); dd = torch.cat(dd, 1); ww = torch.cat(ww, 1)
                valid = pid >= 0
                pc = pid.clamp_min(0)
                pairs = torch.stack([self.ui[pc], self.uj[pc]], -1)                     # positions within selected
                pairs = torch.where(valid[..., None], pairs, torch.full_like(pairs, -1))
                a_ = torch.gather(P, 1, pairs[..., 0].clamp_min(0)[..., None].expand(-1, -1, 3))
                b_ = torch.gather(P, 1, pairs[..., 1].clamp_min(0)[..., None].expand(-1, -1, 3))
                out['sc_pairs'].append(pairs); out['sc_d'].append(dd.double()); out['sc_w'].append(ww.double())
                out['sc_dir'].append(normalize(a_ - b_))
        return {k: torch.cat(v) for k, v in out.items() if v}


def name_of(bss, part_id):
    for name, pid in bss.SMPLX_PART_IDS.items():
        if int(pid) == int(part_id):
            return name
    return ''


def mat_to_quat_wxyz(R):
    """Rotation matrices [...,3,3] -> unit quaternions wxyz (scipy-compatible branch selection, sign w >= 0)."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qw = torch.sqrt((1 + t).clamp_min(0)) / 2
    qx = torch.sqrt((1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp_min(0)) / 2
    qy = torch.sqrt((1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]).clamp_min(0)) / 2
    qz = torch.sqrt((1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]).clamp_min(0)) / 2
    qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    q = torch.stack([qw, qx, qy, qz], -1)
    return q / q.norm(dim=-1, keepdim=True)


def load_smplx_sequence(problem):
    """Motion of the problem's clip (same loader UMR uses)."""
    _umr_path()
    import smpl_surface_retarget_common as common
    a = problem.args
    data, _ = common.load_motion_collection(Path(a['data']))
    key, seq = common.select_sequence(data, a.get('seq_key'), a.get('seq_index', 0))
    return seq
