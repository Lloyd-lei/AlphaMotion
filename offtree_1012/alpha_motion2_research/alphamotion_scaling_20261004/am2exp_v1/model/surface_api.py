"""Adapter to the WP-ASSETS dense surface API (CODE/assets/surface_v2.py).

Uses ``am2exp_v1.assets.surface_v2`` (load_surface / pose_surface / pose_surface_subset /
ground_clearance) when it imports; otherwise a minimal fallback that follows the
interfaces.md schema exactly for RIGID bindings (enough for synthetic test assets and for
reading rigid robot npz files).  ``bone_revolution`` posing (human_smpl) is only available
through the real WP-ASSETS module; the fallback raises a clear error.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
import math
import numpy as np
import torch
from ._sealed import OUT

ASSET_DIR = OUT / 'assets' / 'surface_v2'
REGIONS = ['pelvis', 'trunk', 'head', 'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand',
           'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot', 'digit', 'base_or_link', 'palm']
N_REGIONS = len(REGIONS)
BODY_REGIONS = list(range(15))  # whole-body regions used by the relational objective

_REAL = None
_REAL_ERROR = None


def real_module():
    """The WP-ASSETS module, or None (reason in import_error())."""
    global _REAL, _REAL_ERROR
    if _REAL is None and _REAL_ERROR is None:
        try:
            import am2exp_v1.assets.surface_v2 as m  # noqa: F401
            for name in ('load_surface', 'pose_surface'):
                if not hasattr(m, name): raise ImportError('assets.surface_v2 lacks ' + name)
            _REAL = m
        except Exception as exc:  # module absent or still being written by WP-ASSETS
            _REAL_ERROR = repr(exc)
    return _REAL


def import_error():
    real_module()
    return _REAL_ERROR


def asset_path(body_name, asset_dir=None):
    return Path(asset_dir or ASSET_DIR) / f'{body_name}.npz'


def real_asset_available(body_name, asset_dir=None):
    return asset_path(body_name, asset_dir).exists()


def field_(asset, name, default=None):
    """Read one schema field from a dataclass/object or a dict."""
    if isinstance(asset, dict):
        return asset.get(name, default)
    return getattr(asset, name, default)


@dataclass
class FallbackAsset:
    """interfaces.md schema (tensors). Only rigid posing is implemented here."""
    body: str
    points_local: torch.Tensor
    normals_local: torch.Tensor
    anchor: torch.Tensor
    region: torch.Tensor
    pad: torch.Tensor
    area: torch.Tensor
    group_centre: torch.Tensor
    group_members: torch.Tensor
    group_anchor: torch.Tensor
    group_region: torch.Tensor
    sub1024: torch.Tensor
    allowed_anchor_pairs: torch.Tensor
    rest_points: torch.Tensor
    binding: str = 'rigid'
    beta_vis_cm: float = 0.0
    delta_col_cm: float = float('nan')
    group128_centre: torch.Tensor | None = None
    group128_members: torch.Tensor | None = None
    group128_anchor: torch.Tensor | None = None
    group128_region: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)

    def to(self, device):
        kw = {}
        for f in dc_fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.to(device) if isinstance(v, torch.Tensor) else v
        return FallbackAsset(**kw)


def _binding_str(value):
    if isinstance(value, (bytes, np.bytes_)): value = value.decode()
    if isinstance(value, np.ndarray): value = value.reshape(-1)[0]
    if isinstance(value, (list, tuple)): value = value[0]
    if isinstance(value, (bytes, np.bytes_)): value = value.decode()
    return str(value)


def load_fallback_npz(body_name, device='cpu', asset_dir=None):
    path = asset_path(body_name, asset_dir)
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    ints = {'anchor', 'region', 'pad', 'group_centre', 'group_members', 'group_anchor', 'group_region', 'sub1024',
            'group128_centre', 'group128_members', 'group128_anchor', 'group128_region'}
    kw = {'body': body_name}
    for f in dc_fields(FallbackAsset):
        if f.name in ('body', 'extra'): continue
        if f.name not in a: continue
        v = a[f.name]
        if f.name == 'binding':
            kw[f.name] = _binding_str(v)
        elif f.name in ('beta_vis_cm', 'delta_col_cm'):
            kw[f.name] = float(np.asarray(v).reshape(-1)[0])
        elif f.name == 'allowed_anchor_pairs':
            kw[f.name] = torch.as_tensor(v.astype(bool), device=device)
        elif f.name in ints:
            kw[f.name] = torch.as_tensor(v.astype(np.int64), device=device)
        else:
            kw[f.name] = torch.as_tensor(v.astype(np.float32), device=device)
    kw['extra'] = {k: v for k, v in a.items() if k not in kw}
    return FallbackAsset(**kw)


def load_surface(body_name, device='cpu', asset_dir=None):
    m = real_module()
    if m is not None:  # real loader (supports bone_revolution); asset_dir=None -> OUT/assets/surface_v2
        return m.load_surface(body_name, device, asset_dir) if asset_dir is not None else m.load_surface(body_name, device)
    return load_fallback_npz(body_name, device, asset_dir)


def binding_of(asset):
    return _binding_str(field_(asset, 'binding', 'rigid'))


def _pose_rigid_subset(asset, rot, pos, idx):
    local = asset.points_local[idx].float(); nloc = asset.normals_local[idx].float(); anc = asset.anchor[idx]
    R = rot.float()[:, anc]  # [B,n,3,3]
    p = torch.einsum('bnij,nj->bni', R, local) + pos.float()[:, anc]
    n = torch.einsum('bnij,nj->bni', R, nloc)
    return p, n


def pose_surface_subset(asset, rot, pos, idx):
    """(points [B,n,3] cm, normals [B,n,3]) for asset point indices idx (LongTensor [n])."""
    if isinstance(asset, FallbackAsset):
        if asset.binding != 'rigid':
            raise NotImplementedError('fallback surface posing supports rigid bindings only (got %s)' % asset.binding)
        return _pose_rigid_subset(asset, rot, pos, idx)
    m = real_module()
    if m is None: raise RuntimeError('real surface asset without assets.surface_v2: ' + str(import_error()))
    if hasattr(m, 'pose_surface_subset'):
        return m.pose_surface_subset(asset, rot, pos, idx)
    p, n = m.pose_surface(asset, rot, pos)
    return p[:, idx], n[:, idx]


def pose_surface(asset, rot, pos):
    if isinstance(asset, FallbackAsset):
        idx = torch.arange(asset.points_local.shape[0], device=asset.points_local.device)
        return pose_surface_subset(asset, rot, pos, idx)
    m = real_module()
    if m is None: raise RuntimeError('real surface asset without assets.surface_v2: ' + str(import_error()))
    return m.pose_surface(asset, rot, pos)


def ground_clearance(points_cm, root_height_m, beta_vis_cm):
    """h_cm [B,P] = points_y + 100*root_height_m - beta_vis_cm (interfaces.md)."""
    return points_cm[..., 1] + 100.0 * root_height_m[:, None] - float(beta_vis_cm)


# ----------------------------------------------------------------------------------------
# synthetic schema-conformant assets (tests / benchmarks only; never used for training)
# ----------------------------------------------------------------------------------------

def rest_joint_positions(parents, offsets):
    """Identity-rotation FK: P_child = P_parent + offset_child (cm, root-relative)."""
    P = np.zeros((len(parents), 3), np.float64)
    for j, p in enumerate(parents):
        if p >= 0: P[j] = P[p] + np.asarray(offsets[j], np.float64)
    return P


def hop_matrix(parents):
    J = len(parents); D = np.full((J, J), 10 ** 6, np.int64)
    adj = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0: adj[j].append(int(p)); adj[int(p)].append(j)
    for s in range(J):
        D[s, s] = 0; frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if D[s, v] > D[s, u] + 1: D[s, v] = D[s, u] + 1; nxt.append(v)
            frontier = nxt
    return D


def farthest_point_sampling(points, count, seed=0):
    pts = np.asarray(points, np.float64); n = len(pts)
    rng = np.random.default_rng(seed); first = int(rng.integers(n))
    chosen = [first]; dist = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(1, count):
        k = int(dist.argmax()); chosen.append(k); dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
    return np.asarray(chosen, np.int64)


def _region_for(name):
    n = name.lower()
    side = 'L_' if (n.startswith('l_') or 'left' in n) else ('R_' if (n.startswith('r_') or 'right' in n) else '')
    if any(k in n for k in ('toe', 'foot', 'ankle', 'sole')): base = 'foot'
    elif any(k in n for k in ('knee', 'shin', 'calf')): base = 'shin'
    elif any(k in n for k in ('hip', 'thigh')): base = 'thigh'
    elif any(k in n for k in ('wrist', 'hand', 'palm')): base = 'hand'
    elif any(k in n for k in ('elbow', 'forearm')): base = 'forearm'
    elif any(k in n for k in ('shoulder', 'thorax', 'upper_arm', 'arm')): base = 'upper_arm'
    elif any(k in n for k in ('head', 'neck')): return REGIONS.index('head')
    elif any(k in n for k in ('pelvis', 'root', 'base')): return REGIONS.index('pelvis')
    elif any(k in n for k in ('torso', 'spine', 'chest', 'waist', 'imu')): return REGIONS.index('trunk')
    else: return REGIONS.index('base_or_link')
    key = side + base
    return REGIONS.index(key) if key in REGIONS else REGIONS.index('trunk')


def build_synthetic_asset(name, parents, offsets, joint_names, *, points=4096, groups=256, members=16,
                          groups_alt=128, members_alt=32, sub=1024, radius_frac=0.035, seed=0, device='cpu'):
    """Rigid capsule-per-bone surface following the interfaces.md schema.

    Bone parent->child is rigid in the parent's frame, so its points anchor to the parent.
    Leaf joints get a sphere anchored to themselves.  Rest pose = identity rotations.
    """
    parents = np.asarray(parents, np.int64); offsets = np.asarray(offsets, np.float64)
    J = len(parents); Prest = rest_joint_positions(parents, offsets)
    reach = max(float(np.linalg.norm(Prest, axis=1).max()), 1e-3)
    radius = max(radius_frac * reach, 0.5)
    rng = np.random.default_rng(seed)
    children = [[c for c in range(J) if parents[c] == j] for j in range(J)]
    segs = []  # (anchor, a, b, region)
    for j in range(1, J):
        p = int(parents[j]); L = float(np.linalg.norm(Prest[j] - Prest[p]))
        if L > 1e-3: segs.append((p, Prest[p], Prest[j], _region_for(joint_names[j])))
    for j in range(J):
        if not children[j]: segs.append((j, Prest[j], Prest[j], _region_for(joint_names[j])))
    lengths = np.array([max(np.linalg.norm(b - a), radius) for _, a, b, _ in segs])
    alloc = np.maximum(1, np.floor(points * lengths / lengths.sum()).astype(int))
    while alloc.sum() > points: alloc[alloc.argmax()] -= 1
    while alloc.sum() < points: alloc[alloc.argmin()] += 1
    pts = []; nrm = []; anc = []; reg = []
    for (a_j, a, b, r), cnt in zip(segs, alloc):
        axis = b - a; L = np.linalg.norm(axis)
        if L < 1e-6:  # sphere
            d = rng.normal(size=(cnt, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            pts.append(a + radius * d); nrm.append(d)
        else:
            u = axis / L; tmp = np.array([1., 0, 0]) if abs(u[0]) < .9 else np.array([0, 1., 0])
            v = np.cross(u, tmp); v /= np.linalg.norm(v); w = np.cross(u, v)
            t = rng.random(cnt); th = rng.random(cnt) * 2 * math.pi
            radial = np.cos(th)[:, None] * v + np.sin(th)[:, None] * w
            pts.append(a + t[:, None] * axis + radius * radial); nrm.append(radial)
        anc.append(np.full(cnt, a_j)); reg.append(np.full(cnt, r))
    rest = np.concatenate(pts); normals = np.concatenate(nrm); anchor = np.concatenate(anc); region = np.concatenate(reg)
    local = rest - Prest[anchor]

    def grouping(G, K):
        centre = farthest_point_sampling(rest, G, seed)
        d = np.linalg.norm(rest[centre][:, None] - rest[None], axis=-1)
        mem = np.argsort(d, axis=1, kind='stable')[:, :K]
        mem[:, 0] = centre
        ganc = np.array([np.bincount(anchor[m], minlength=J).argmax() for m in mem])
        greg = region[centre]
        return centre, mem, ganc, greg

    c, m, ga, gr = grouping(groups, members)
    c2, m2, ga2, gr2 = grouping(groups_alt, members_alt)
    subset = farthest_point_sampling(rest, sub, seed + 1)
    hop = hop_matrix(parents); allowed = hop >= 3
    T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
    L_ = torch.long
    return FallbackAsset(
        body=name, points_local=T(local), normals_local=T(normals), anchor=T(anchor, L_), region=T(region, L_),
        pad=T(np.full(len(rest), -1), L_), area=T(np.full(len(rest), 1.0)), group_centre=T(c, L_),
        group_members=T(m, L_), group_anchor=T(ga, L_), group_region=T(gr, L_), sub1024=T(subset, L_),
        allowed_anchor_pairs=T(allowed, torch.bool), rest_points=T(rest), binding='rigid', beta_vis_cm=0.0,
        group128_centre=T(c2, L_), group128_members=T(m2, L_), group128_anchor=T(ga2, L_), group128_region=T(gr2, L_),
        extra={'synthetic': True, 'radius_cm': radius})
