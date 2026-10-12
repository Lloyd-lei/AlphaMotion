"""Attach dense surfaces to the sealed Body.geom() dicts (wraps scaling_data.attach_geometry).

``attach_dense_geometry(bodies, device, ...)`` keeps whatever ``attach_geometry`` installed
(g['surface'] 96-patch B asset etc.) and adds ``g['dense_surface']`` for every body that has
an asset: static per-body tensors expanded to the batch plus the asset object.  Bodies without
an asset get no key, so DenseSurfaceMegatron falls back to SFMegatron exactly.

S_wrong twin: ``derangement={body: donor}`` gives ``body`` the donor's surface SHAPE re-bound to
``body``'s own skeleton (rest points scaled by reach_body/reach_donor, each point/group anchored
to the nearest own rest joint).  The deranged dict is static-shape only (no posing).
"""
from __future__ import annotations
import types
import numpy as np
import torch
from torch.nn import functional as F
from . import surface_api as sa

_STATIC_TENSORS = ('rest_points', 'normals', 'region', 'anchor', 'group_members', 'group_centre', 'group_anchor',
                   'group_region', 'sub', 'allowed', 'member_feat', 'centre_feat', 'point_dep', 'self_allowed',
                   'member_flat', 'group_valid')
LOCAL_SCALE_FRAC = 0.05  # PointNet input (p - centre)/(0.05*reach): O(1) values (plan: '/reach', x20)


def _t(asset, name, device, dtype=None):
    v = sa.field_(asset, name)
    if v is None: return None
    v = torch.as_tensor(v, device=device)
    return v if dtype is None else v.to(dtype)


def _grouping_fields(asset, grouping):
    pre = {'g256': 'group', 'g128': 'group128'}[grouping]
    return pre + '_centre', pre + '_members', pre + '_anchor', pre + '_region'


@torch.no_grad()
def point_dependency(asset, joints, device, trials=2, seed=0):
    """bool [P,J]: which joints' pose9 (rotation or position) move each surface point.

    Rigid: one-hot anchor.  Other bindings (bone_revolution): measured numerically by
    perturbing one joint at a time at random poses through the real pose_surface.
    """
    anchor = _t(asset, 'anchor', device, torch.long)
    P = anchor.shape[0]
    if sa.binding_of(asset) == 'rigid':
        return F.one_hot(anchor, joints).bool()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    rest = _t(asset, 'rest_points', device, torch.float32)
    dep = torch.zeros(P, joints, dtype=torch.bool, device=device)
    scale = float(rest.abs().max().clamp_min(1.0))
    for _ in range(trials):
        pos0 = torch.randn(1, joints, 3, generator=gen).to(device) * 0.3 * scale
        rot0 = torch.linalg.qr(torch.randn(1, joints, 3, 3, generator=gen))[0].to(device)
        rot0 = rot0 * torch.sign(torch.linalg.det(rot0))[..., None, None]
        rot = rot0.repeat(joints + 1, 1, 1, 1); pos = pos0.repeat(joints + 1, 1, 1)
        for j in range(joints):
            pos[j + 1, j] += torch.randn(3, generator=gen).to(device) * 0.1 * scale
            dR = torch.linalg.matrix_exp(torch.tensor([[0., -.3, .2], [.3, 0., -.1], [-.2, .1, 0.]], device=device))
            rot[j + 1, j] = dR @ rot[j + 1, j]
        pts, _ = sa.pose_surface(asset, rot, pos)
        moved = (pts[1:] - pts[:1]).norm(dim=-1) > 1e-4 * scale  # [J,P]
        dep |= moved.T
    # a point always depends on its anchor
    dep |= F.one_hot(anchor, joints).bool()
    return dep


POINT_TRUST_MAX_CM = 3.0  # [integration] WP-ASSETS README: mask points with data residual p95 > 3 cm


def point_trust_mask(asset, P, device, max_cm=POINT_TRUST_MAX_CM):
    """[integration] bool [P]: True unless the asset's per-point data-binding residual (p95, cm) exceeds max_cm.
    The field lives in SurfaceAsset.extra (assets.surface_v2) or directly on dict/fallback assets; NaN (no data
    check, e.g. adroit/franka) and absent fields count as trusted."""
    res = sa.field_(asset, 'point_data_residual_p95_cm')
    if res is None:
        extra = sa.field_(asset, 'extra') or {}
        res = extra.get('point_data_residual_p95_cm') if isinstance(extra, dict) else None
    if res is None:
        return torch.ones(P, dtype=torch.bool, device=device)
    r = torch.as_tensor(np.asarray(res, np.float32), device=device).reshape(-1)
    if r.numel() != P:
        raise ValueError(f'point_data_residual_p95_cm has {r.numel()} entries, expected {P}')
    return ~(torch.isfinite(r) & (r > max_cm))


def build_dense_static(asset, body, device, grouping='g256', *, donor_asset=None, donor_body=None):
    """Per-body static tensors used by the model (no batch dimension)."""
    J = len(body.parents); reach = float(body.reach)
    src = asset if donor_asset is None else donor_asset
    c_name, m_name, a_name, r_name = _grouping_fields(src, grouping)
    rest = _t(src, 'rest_points', device, torch.float32)
    normals = F.normalize(_t(src, 'normals_local', device, torch.float32), dim=-1)
    region = _t(src, 'region', device, torch.long)
    members = _t(src, m_name, device, torch.long); centre = _t(src, c_name, device, torch.long)
    gregion = _t(src, r_name, device, torch.long)
    if members is None or centre is None:
        raise ValueError(f'{body.name}: asset lacks grouping {grouping}')
    sub = _t(src, 'sub1024', device, torch.long)
    if donor_asset is None:
        anchor = _t(asset, 'anchor', device, torch.long); ganchor = _t(asset, a_name, device, torch.long)
        point_dep = point_dependency(asset, J, device)
        deranged = False
    else:
        # wrong-body twin: donor shape scaled to this body's reach, anchored to own nearest rest joint
        s = reach / float(donor_body.reach)
        rest = rest * s
        Pj = torch.as_tensor(sa.rest_joint_positions(body.parents, body.offsets), device=device, dtype=torch.float32)
        anchor = torch.cdist(rest, Pj).argmin(-1)
        ganchor = torch.stack([torch.bincount(anchor[m], minlength=J).argmax() for m in members])
        point_dep = F.one_hot(anchor, J).bool()
        deranged = True
    allowed = _t(asset, 'allowed_anchor_pairs', device, torch.bool)  # always the OWN body's kinematic table
    if allowed is None or allowed.shape != (J, J):
        raise ValueError(f'{body.name}: allowed_anchor_pairs must be [J,J]')
    if int(anchor.max()) >= J or int(ganchor.max()) >= J:
        raise ValueError(f'{body.name}: surface anchors outside skeleton (J={J})')
    G, K = members.shape
    loc = (rest[members] - rest[centre][:, None]) / (LOCAL_SCALE_FRAC * reach)
    onehot = F.one_hot(region[members].clamp(0, sa.N_REGIONS - 1), sa.N_REGIONS).float()
    member_feat = torch.cat([loc, normals[members], onehot], -1)  # [G,K,3+3+18]
    centre_feat = rest[centre] / reach  # [G,3]
    member_flat = members.reshape(-1)
    self_allowed = allowed[anchor[member_flat]][:, anchor[sub]]  # [G*K, S]
    beta = float(sa.field_(asset, 'beta_vis_cm', 0.0) or 0.0)
    if not np.isfinite(beta): beta = 0.0
    # [integration] per-point trust from WP-ASSETS: points whose rigid binding misses the data by > 3 cm (p95;
    # kuavo_s45 457 pts, unitree_h1_2 219 pts: housings spanning two merged MuJoCo links) are excluded from the
    # posed-surface uses (arm O losses, arm I source features). Static group tokens still see the rest shape.
    point_trusted = point_trust_mask(asset, rest.shape[0], device) if donor_asset is None else \
        torch.ones(rest.shape[0], dtype=torch.bool, device=device)
    return {
        'body': body.name, 'donor': donor_body.name if donor_body is not None else body.name, 'deranged': deranged,
        'asset': None if deranged else asset, 'binding': 'deranged_static_only' if deranged else sa.binding_of(asset),
        'grouping': grouping, 'reach': reach, 'beta_vis_cm': beta,
        'rest_points': rest, 'normals': normals, 'region': region, 'anchor': anchor, 'group_members': members,
        'group_centre': centre, 'group_anchor': ganchor, 'group_region': gregion, 'sub': sub, 'allowed': allowed,
        'member_feat': member_feat, 'centre_feat': centre_feat, 'point_dep': point_dep,
        'self_allowed': self_allowed, 'member_flat': member_flat,
        'group_valid': torch.ones(G, dtype=torch.bool, device=device),
        'point_trusted': point_trusted,  # [integration] [P] bool, see point_trust_mask
    }


def expand_dense(static, batch):
    out = {}
    for k, v in static.items():
        out[k] = v[None].expand(batch, *v.shape) if (k in _STATIC_TENSORS and isinstance(v, torch.Tensor)) else v
    out['static'] = static  # un-expanded view for the model (row-invariant tensors)
    return out


def _install(body, static):
    if not hasattr(body, '_geom_without_dense'):
        body._geom_without_dense = body.geom  # the closure installed by scaling_data.attach_geometry
    body._dense_surface_static = static
    previous = body._geom_without_dense

    def geom(self, batch):
        g = previous(batch)
        if getattr(self, '_dense_surface_static', None) is not None:
            g['dense_surface'] = expand_dense(self._dense_surface_static, batch)
        return g
    body.geom = types.MethodType(geom, body)


def default_derangement(names):
    """Fixed derangement: cyclic shift within {XWBC humanoids incl. human} and swap within externals."""
    names = list(names)
    ext = sorted(n for n in names if n in ('adroit_pen', 'relay_franka_full_v2'))
    hum = sorted(n for n in names if n not in ext)
    out = {}
    for group in (hum, ext):
        if len(group) >= 2:
            for i, n in enumerate(group): out[n] = group[(i + 1) % len(group)]
    return out


def attach_dense_geometry(bodies, device, *, assets=None, asset_dir=None, grouping='g256', derangement=None,
                          require=None):
    """Attach dense surfaces in place; returns a JSON-safe report.

    assets: optional {body_name: asset} (synthetic tests); otherwise load_surface() from
            OUT/assets/surface_v2 for every body whose npz exists.
    derangement: {body: donor} (S_wrong twin) or 'default'.
    require: iterable of body names that MUST receive a surface (raises otherwise).
    """
    by = {b.name: b for b in bodies}
    loaded = {}
    for b in bodies:
        if assets is not None:
            if b.name in assets: loaded[b.name] = assets[b.name]
        elif sa.real_asset_available(b.name, asset_dir):
            loaded[b.name] = sa.load_surface(b.name, device, asset_dir)
    if derangement == 'default': derangement = default_derangement(loaded)
    derangement = dict(derangement or {})
    for n, donor in derangement.items():
        if n == donor: raise ValueError('derangement maps a body to itself: ' + n)
        if n not in loaded or donor not in loaded: raise ValueError(f'derangement needs both assets: {n}<-{donor}')
    report = {'grouping': grouping, 'bodies': {}, 'derangement': derangement,
              'surface_api': 'assets.surface_v2' if sa.real_module() is not None else 'fallback:' + str(sa.import_error())}
    for b in bodies:
        if b.name not in loaded:
            if require and b.name in set(require): raise ValueError('required dense surface missing: ' + b.name)
            if hasattr(b, '_geom_without_dense'): _install(b, None)
            continue
        donor = derangement.get(b.name)
        static = build_dense_static(loaded[b.name], b, device, grouping,
                                    donor_asset=loaded[donor] if donor else None, donor_body=by[donor] if donor else None)
        _install(b, static)
        report['bodies'][b.name] = {'donor': static['donor'], 'groups': int(static['group_members'].shape[0]),
                                    'members': int(static['group_members'].shape[1]),
                                    'points': int(static['rest_points'].shape[0]), 'subset': int(static['sub'].shape[0]),
                                    'binding': static['binding'], 'beta_vis_cm': static['beta_vis_cm']}
    return report


def detach_dense_geometry(bodies):
    for b in bodies:
        if hasattr(b, '_geom_without_dense'):
            b.geom = b._geom_without_dense; del b._geom_without_dense
        if hasattr(b, '_dense_surface_static'): del b._dense_surface_static
