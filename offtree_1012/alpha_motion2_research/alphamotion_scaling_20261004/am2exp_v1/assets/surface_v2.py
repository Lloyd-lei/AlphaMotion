"""Dense surface assets (surface_v2) runtime API — interfaces.md contract.  Pure torch; no sealed imports.

    load_surface(body_name, device)                  -> SurfaceAsset
    pose_surface(asset, rot [B,J,3,3], pos [B,J,3])  -> (points [B,P,3] cm, normals [B,P,3])
    pose_surface_subset(asset, rot, pos, idx)        -> same for idx (LongTensor [n])
    ground_clearance(points_cm, root_height_m [B], beta_vis_cm) -> h_cm [B,P] = y + 100*root_height_m - beta_vis_cm

Conventions: AM2 canonical frame (Y-up, cm, root-relative).  `rot` are OWN-JOINT GLOBAL rotations (identity at the
AM2 rest pose), `pos` root-relative joint positions in cm (for relay_franka_full_v2 these must already include the
finger slides, i.e. prismatic/kinematics.mixed_forward_kinematics joint_position).

binding 'rigid' (all robots, adroit_pen, relay_franka_full_v2):
    p = R[anchor] @ points_local + pos[anchor];   n = R[anchor] @ normals_local
binding 'bone_revolution' (human_smpl; joint POSITIONS only, `rot` is ignored — LAFAN-dialect twist is
unobservable):
    per joint j a bone frame from positions: u = unit(pos[child_j] - pos[j]) (leaf: unit(pos[j] - pos[parent_j])),
    v = normalize(sum_k w_jk * perp_u(unit(ref_jk)) + 1e-3 * perp_u(fallback_j)), w = u x v, where each ref_jk is
    pos[a]-pos[b] (type 0) or (pos[a]-pos[b]) x (pos[c]-pos[d]) (type 1) per the frozen table frame_ref_* stored in
    the npz (documented in the json manifest);
    p = pos[bone_parent] + t*L*u + radial*(cos(theta) v + sin(theta) w), L = |pos[child]-pos[parent]| or leaf_len;
    n = normal_uvw[0] u + normal_uvw[1] v + normal_uvw[2] w.
Everything is float32 by default (inputs are cast to the asset dtype), batched and differentiable w.r.t. rot/pos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

ASSET_DIR = Path('/var/tmp/alphamotion/am2exp/assets/surface_v2')
REGIONS = ['pelvis', 'trunk', 'head', 'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand',
           'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot', 'digit', 'base_or_link', 'palm']
N_REGIONS = len(REGIONS)
_EPS = 1e-8

_LONG_KEYS = ('anchor', 'region', 'pad', 'group_centre', 'group_members', 'group_anchor', 'group_region', 'sub1024',
              'group128_centre', 'group128_members', 'group128_anchor', 'group128_region', 'bone_parent', 'bone_child',
              'joint_parents', 'joint_child', 'frame_ref_type', 'frame_ref_idx', 'point_geom')
_BOOL_KEYS = ('allowed_anchor_pairs', 'leaf_joint')
_SCALAR_KEYS = ('beta_vis_cm', 'delta_col_cm')


@dataclass
class SurfaceAsset:
    body: str
    binding: str                       # 'rigid' | 'bone_revolution'
    points_local: torch.Tensor         # [P,3] cm (anchor rest frame; human: calibration pose minus anchor)
    normals_local: torch.Tensor        # [P,3]
    anchor: torch.Tensor               # [P] long
    region: torch.Tensor               # [P] long
    pad: torch.Tensor                  # [P] long (-1 none)
    area: torch.Tensor                 # [P] cm^2
    group_centre: torch.Tensor         # [G]
    group_members: torch.Tensor        # [G,K]
    group_anchor: torch.Tensor         # [G]
    group_region: torch.Tensor         # [G]
    group128_centre: torch.Tensor
    group128_members: torch.Tensor
    group128_anchor: torch.Tensor
    group128_region: torch.Tensor
    sub1024: torch.Tensor              # [1024]
    allowed_anchor_pairs: torch.Tensor  # [J,J] bool
    rest_points: torch.Tensor          # [P,3] cm
    rest_normals: torch.Tensor         # [P,3]
    beta_vis_cm: float
    delta_col_cm: float
    num_joints: int
    # bone_revolution only (None for rigid)
    bone_parent: torch.Tensor | None = None
    bone_child: torch.Tensor | None = None
    t: torch.Tensor | None = None
    radial: torch.Tensor | None = None
    theta: torch.Tensor | None = None
    leaf_len: torch.Tensor | None = None
    normal_uvw: torch.Tensor | None = None
    joint_parents: torch.Tensor | None = None   # [J]
    joint_child: torch.Tensor | None = None     # [J] main child, -1 leaf
    frame_ref_type: torch.Tensor | None = None  # [J,R] 0 diff, 1 cross, -1 unused
    frame_ref_idx: torch.Tensor | None = None   # [J,R,4]
    frame_ref_w: torch.Tensor | None = None     # [J,R]
    frame_fallback: torch.Tensor | None = None  # [J,3]
    extra: dict = field(default_factory=dict)

    @property
    def num_points(self):
        return int(self.points_local.shape[0])

    def to(self, device=None, dtype=None):
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                if dtype is not None and v.is_floating_point():
                    v = v.to(dtype)
                v = v.to(device) if device is not None else v
            kw[f.name] = v
        return SurfaceAsset(**kw)


def asset_path(body_name, asset_dir=None) -> Path:
    return Path(asset_dir or ASSET_DIR) / f'{body_name}.npz'


def load_surface(body_name, device='cpu', asset_dir=None, dtype=torch.float32) -> SurfaceAsset:
    """Load OUT/assets/surface_v2/<body>.npz into tensors on `device`."""
    path = asset_path(body_name, asset_dir)
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    kw = {'body': body_name, 'binding': str(a['binding'].reshape(-1)[0]), 'num_joints': int(a['allowed_anchor_pairs'].shape[0])}
    names = {f.name for f in fields(SurfaceAsset)}
    for k, v in a.items():
        if k in ('binding',) or k not in names:
            continue
        if k in _SCALAR_KEYS:
            kw[k] = float(np.asarray(v).reshape(-1)[0])
        elif k in _BOOL_KEYS:
            kw[k] = torch.as_tensor(v.astype(bool), device=device)
        elif k in _LONG_KEYS:
            kw[k] = torch.as_tensor(v.astype(np.int64), device=device)
        else:
            kw[k] = torch.as_tensor(v.astype(np.float64), device=device).to(dtype)
    kw['extra'] = {k: v for k, v in a.items() if k not in names}
    return SurfaceAsset(**kw)


def _perp_unit(vec, u):
    """Component of unit(vec) orthogonal to unit axis u (not renormalised)."""
    vn = vec / vec.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return vn - (vn * u).sum(-1, keepdim=True) * u


def bone_frames(asset: SurfaceAsset, pos: torch.Tensor):
    """Per-joint bone frames from joint positions only.  pos [B,J,3] -> (u, v, w [B,J,3], L [B,J])."""
    jp = asset.joint_parents; jc = asset.joint_child
    leaf = jc < 0
    child = torch.where(leaf, torch.zeros_like(jc), jc)
    par = torch.where(jp < 0, torch.zeros_like(jp), jp)
    d_child = pos[:, child] - pos            # j -> child
    d_leaf = pos - pos[:, par]               # parent -> j (leaf direction)
    d = torch.where(leaf[None, :, None], d_leaf, d_child)
    L = d_child.norm(dim=-1)
    u = d / d.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    rt, ri, rw = asset.frame_ref_type, asset.frame_ref_idx, asset.frame_ref_w
    B, J = pos.shape[:2]
    v_raw = 1e-3 * _perp_unit(asset.frame_fallback.to(pos.dtype)[None].expand(B, J, 3), u)
    for r in range(rt.shape[1]):
        a, b, c, e = ri[:, r, 0], ri[:, r, 1], ri[:, r, 2], ri[:, r, 3]
        diff = pos[:, a] - pos[:, b]
        cross = torch.cross(pos[:, a] - pos[:, b], pos[:, c] - pos[:, e], dim=-1)
        vec = torch.where((rt[:, r] == 1)[None, :, None], cross, diff)
        wgt = torch.where(rt[:, r] >= 0, rw[:, r], torch.zeros_like(rw[:, r])).to(pos.dtype)
        v_raw = v_raw + wgt[None, :, None] * _perp_unit(vec, u)
    v = v_raw / v_raw.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    w = torch.cross(u, v, dim=-1)
    return u, v, w, L


def _pose_subset(asset: SurfaceAsset, rot, pos, idx):
    dt = asset.points_local.dtype
    pos = pos.to(dt)
    if asset.binding == 'rigid':
        rot = rot.to(dt)
        anc = asset.anchor[idx]
        R = rot[:, anc]
        p = torch.einsum('bnij,nj->bni', R, asset.points_local[idx]) + pos[:, anc]
        n = torch.einsum('bnij,nj->bni', R, asset.normals_local[idx])
        return p, n
    if asset.binding != 'bone_revolution':
        raise ValueError('unknown binding ' + asset.binding)
    u, v, w, L = bone_frames(asset, pos)
    bp = asset.bone_parent[idx]
    leaf = asset.bone_child[idx] < 0
    Lp = torch.where(leaf[None], asset.leaf_len[idx][None].to(dt), L[:, bp])
    ub, vb, wb = u[:, bp], v[:, bp], w[:, bp]
    th = asset.theta[idx]
    rad = asset.radial[idx]
    radial_dir = torch.cos(th)[None, :, None] * vb + torch.sin(th)[None, :, None] * wb
    p = pos[:, bp] + (asset.t[idx][None] * Lp)[..., None] * ub + rad[None, :, None] * radial_dir
    nuvw = asset.normal_uvw[idx]
    n = nuvw[None, :, 0:1] * ub + nuvw[None, :, 1:2] * vb + nuvw[None, :, 2:3] * wb
    return p, n


def pose_surface(asset: SurfaceAsset, rot: torch.Tensor, pos: torch.Tensor):
    """(points [B,P,3] cm, normals [B,P,3]).  rot [B,J,3,3] (ignored for bone_revolution), pos [B,J,3] cm."""
    idx = torch.arange(asset.num_points, device=asset.points_local.device)
    return _pose_subset(asset, rot, pos, idx)


def pose_surface_subset(asset: SurfaceAsset, rot: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor):
    """Same as pose_surface for asset point indices idx (LongTensor [n]); e.g. asset.sub1024."""
    return _pose_subset(asset, rot, pos, idx.to(asset.points_local.device))


def ground_clearance(points_cm: torch.Tensor, root_height_m: torch.Tensor, beta_vis_cm) -> torch.Tensor:
    """h_cm [B,P] = points_y + 100*root_height_m - beta_vis_cm (root_height_m [B] in metres)."""
    return points_cm[..., 1] + 100.0 * root_height_m.to(points_cm.dtype)[:, None] - float(beta_vis_cm)


def rest_joint_positions(asset_or_offsets, parents=None):
    """Identity-rotation FK helper (cm): P_child = P_parent + offset_child."""
    off = np.asarray(asset_or_offsets, np.float64)
    P = np.zeros_like(off)
    for j, p in enumerate(parents):
        if p >= 0:
            P[j] = P[p] + off[j]
    return P


def manifest(body_name, asset_dir=None) -> dict:
    return json.loads((Path(asset_dir or ASSET_DIR) / f'{body_name}.json').read_text())
