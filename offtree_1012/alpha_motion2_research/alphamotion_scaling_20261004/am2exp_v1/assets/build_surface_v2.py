"""Build the UMR-density dense surface assets (surface_v2) for the 18 AM2 bodies.

    PY -m assets.build_surface_v2 --all            (cwd = CODE; or run this file directly)
    PY -m assets.build_surface_v2 --body unitree_g1_29dof [--body ...]

Outputs (OUT/assets/surface_v2): <body>.npz + <body>.json (interfaces.md schema) and manifest.json (frozen config,
sha256 of every file + content hash, per-body summary).  CPU only; ~10-60 s per body; RAM < 3 GB.

Pipeline per robot (adroit_pen / relay_franka_full_v2 included):
  1. AM2 Body dump (orchestration/scaling_data.load_scaling_bodies, config A_S_seed0) in a subprocess.
  2. Kinematic adapter (surface/build_verified_patches.source_adapter, read-only; H2 by data joint names) and
     visual-geom -> AM2-joint binding verified over qpos0 + 7 seeded random q (robot_binding.bind_geoms), plus the
     residual on real data joint states (xwbc extracted LAFAN trajectories) where available.
  3. Whole-body visual mesh assembled at the vendor neutral pose qpos0 (see README: AM2 identity != qpos0 for 10
     robots) -> UMR first-hit exterior sampler (26 views, >=2 visible, 4096 x 8 x 12 candidates, FPS seed 0).
  4. Per point: anchor (from the part of the face), anchor-local point/normal (normals outward via the view
     directions that saw the face), rest_points at AM2 identity rotations, area (Voronoi share of exterior
     candidates), region (regions.py), pads (pads.py).
  5. Groups (FPS centres + kNN on rest_points), sub1024, allowed_anchor_pairs (tree hop >= 3 minus quasi-rigid
     anchor pairs measured on 128 data rows), beta_vis_cm, delta_col_cm.
human_smpl: human_surface.py (SMPL-X bone revolution), same sampler/grouping/pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.dont_write_bytecode = True

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'assets'

import numpy as np  # noqa: E402

from . import _env as E  # noqa: E402
from . import umr_sampler as U  # noqa: E402

CONFIG = dict(schema='surface_v2', num_points=4096, oversample_ratio=8, candidate_multiplier=12, first_hit_directions=26,
              min_visible_views=2, ray_offset=1e-4, ray_distance=0.0, chunk_size=4096, sample_seed=0, fps_seed=0,
              sampler=U.SAMPLER_VERSION, groups=[256, 16], groups_alt=[128, 32], sub_selfcontact=1024, hop_min=3,
              quasi_rigid_mean_cm=8.0, quasi_rigid_std_cm=1.0, quasi_rigid_rows=128, quasi_rigid_points_per_anchor=64,
              verify_random_states=7, verify_seed=20261004, human_n_betas=10, human_beta_reg=4.0,
              sample_pose='robots: MJCF qpos0 (vendor neutral) assembled through the verified AM2 binding; '
                          'human: shaped SMPL-X T-pose; adroit/franka: qpos0 (= AM2 identity)')
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
def ensure_body_dump():
    p = E.CACHE_DIR / 'am2_bodies.json'
    if not p.exists():
        E.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([E.PY, str(Path(__file__).with_name('dump_am2_bodies.py')), str(p)], check=True, env=E.cpu_env(),
                       timeout=1200)
    return E.read_json(p)


def sample(V, F):
    return U.sample_first_hit_surface(V, F, CONFIG['num_points'], CONFIG['sample_seed'],
                                      oversample_ratio=CONFIG['oversample_ratio'],
                                      candidate_multiplier=CONFIG['candidate_multiplier'], ray_offset=CONFIG['ray_offset'],
                                      ray_distance=CONFIG['ray_distance'], min_visible_views=CONFIG['min_visible_views'],
                                      chunk_size=CONFIG['chunk_size'])


def oriented_normals(V, F, s):
    tri = V[F[s.face_ids]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    sgn = np.sign((n * s.visible_dir_sum).sum(1))
    flipped = sgn < 0
    n[flipped] *= -1
    return n, float(flipped.mean())


def point_areas(s):
    """cm^2 per kept point: each exterior candidate represents total_area / candidate_count (area-weighted draws);
    it is credited to its nearest kept point (Voronoi share)."""
    from scipy.spatial import cKDTree
    _, nn = cKDTree(s.points).query(s.exterior_points)
    cnt = np.bincount(nn, minlength=len(s.points))
    return cnt * (s.total_area / s.candidate_count)


def hop_matrix(parents):
    J = len(parents)
    adj = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0:
            adj[j].append(p); adj[p].append(j)
    H = np.full((J, J), 10 ** 6, np.int64)
    for s in range(J):
        H[s, s] = 0; q = [s]
        while q:
            a = q.pop(0)
            for b in adj[a]:
                if H[s, b] > H[s, a] + 1:
                    H[s, b] = H[s, a] + 1; q.append(b)
    return H


def make_groups(rest, anchor, region, G, K, seed):
    from scipy.spatial import cKDTree
    centre = U.farthest_point_indices(rest, G, seed).astype(np.int64)
    _, mem = cKDTree(rest).query(rest[centre], k=K)
    mem = np.asarray(mem, np.int64)
    assert (mem[:, 0] == centre).all() or np.allclose(rest[mem[:, 0]], rest[centre])
    mem[:, 0] = centre
    ga = np.array([np.bincount(anchor[m]).argmax() for m in mem])
    gr = np.array([np.bincount(region[m]).argmax() for m in mem])
    return centre, mem, ga, gr


def rot6_to_matrix(x):
    a = x[..., :3]; b = x[..., 3:]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def data_rows(body, n_rows, seed=0, mask=None):
    """(R [n,J,3,3], P [n,J,3] cm, aux dict) from the AM2 training payload of `body` (evenly spaced rows)."""
    if body in E.BODIES_EXTERNAL:
        path = E.ROOT / 'data' / 'payloads' / f'{body}.npz'
        with np.load(path) as z:
            N = len(z['global_rot6'])
            idx = np.linspace(0, N - 1, n_rows).astype(int)
            return rot6_to_matrix(z['global_rot6'][idx].astype(np.float64)), z['position_native'][idx].astype(np.float64) * 100, {'rows': idx, 'path': str(path)}
    path = E.PILOT_XWBC / 'payloads' / f'{body}.npz'
    with np.load(path) as z:
        ok = np.ones(len(z['split']), bool) if mask is None else mask(z)
        cand = np.flatnonzero(ok)
        idx = cand[np.linspace(0, len(cand) - 1, min(n_rows, len(cand))).astype(int)]
        aux = {k: z[k][idx] for k in ('root_height_m', 'root_height_valid', 'ground_valid', 'contact_hard', 'group_index',
                                      'contact_trust')}
        aux['rows'] = idx; aux['path'] = str(path)
        return rot6_to_matrix(z['global_rot6'][idx].astype(np.float64)), z['position_cm'][idx].astype(np.float64), aux


def pose_np(arrays, binding, R, P):
    """Pose with the runtime torch implementation (surface_v2.pose_surface) for numpy inputs."""
    import torch
    from . import surface_v2 as S
    asset = asset_from_arrays(arrays, binding)
    with torch.no_grad():
        p, n = S.pose_surface(asset, torch.as_tensor(R, dtype=torch.float32), torch.as_tensor(P, dtype=torch.float32))
    return p.numpy().astype(np.float64), n.numpy().astype(np.float64)


def asset_from_arrays(arrays, binding, body='tmp'):
    import tempfile
    from . import surface_v2 as S
    with tempfile.TemporaryDirectory() as td:
        write_npz(Path(td) / f'{body}.npz', arrays)
        return S.load_surface(body, 'cpu', asset_dir=td)


def quasi_rigid_pairs(arrays, binding, parents, R, P):
    """Anchor pairs whose closest surface distance is nearly constant and small over the data rows."""
    J = len(parents)
    anchor = arrays['anchor']
    rng = np.random.default_rng(CONFIG['fps_seed'])
    sel = []
    for a in range(J):
        idx = np.flatnonzero(anchor == a)
        if len(idx) > CONFIG['quasi_rigid_points_per_anchor']:
            idx = np.sort(rng.choice(idx, CONFIG['quasi_rigid_points_per_anchor'], replace=False))
        sel.append(idx)
    flat = np.concatenate(sel)
    pts, _ = pose_np(arrays, binding, R, P)
    pts = pts[:, flat]
    lab = anchor[flat]
    import torch
    lab_t = torch.as_tensor(lab, dtype=torch.int64)
    M = len(lab)
    dmin = np.full((len(R), J, J), np.nan)
    present = np.zeros(J, bool); present[np.unique(lab)] = True
    for r in range(len(R)):
        X = torch.as_tensor(pts[r], dtype=torch.float32)
        D = torch.cdist(X, X)                                                       # [M,M]
        col = torch.full((M, J), float('inf')).scatter_reduce(1, lab_t[None].expand(M, M), D, 'amin')   # [M,J]
        aa = torch.full((J, J), float('inf')).scatter_reduce(0, lab_t[:, None].expand(M, J), col, 'amin')  # [J,J]
        dmin[r] = aa.numpy()
    dmin[:, ~present] = np.nan; dmin[:, :, ~present] = np.nan
    idx = np.arange(J); dmin[:, idx, idx] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mean = np.nanmean(dmin, 0)
        std = np.nanstd(dmin, 0)
    qr = (mean < CONFIG['quasi_rigid_mean_cm']) & (std < CONFIG['quasi_rigid_std_cm'])
    qr = np.where(np.isnan(mean), False, qr)
    return qr, mean, std


def finish_asset(body, parents, names, rest, rest_n, local, local_n, anchor, region, pad, area, extra, binding,
                 R_rows, P_rows, beta, delta, man):
    """Groups / sub1024 / allowed pairs, assemble arrays + manifest."""
    J = len(parents)
    G, K = CONFIG['groups']; G2, K2 = CONFIG['groups_alt']
    gc, gm, ga, gr = make_groups(rest, anchor, region, G, K, CONFIG['fps_seed'])
    gc2, gm2, ga2, gr2 = make_groups(rest, anchor, region, G2, K2, CONFIG['fps_seed'])
    sub = U.farthest_point_indices(rest, CONFIG['sub_selfcontact'], CONFIG['fps_seed']).astype(np.int64)
    arrays = dict(points_local=local.astype(np.float32), normals_local=local_n.astype(np.float32),
                  anchor=anchor.astype(np.int32), region=region.astype(np.int16), pad=pad.astype(np.int8),
                  area=area.astype(np.float32), group_centre=gc.astype(np.int32), group_members=gm.astype(np.int32),
                  group_anchor=ga.astype(np.int32), group_region=gr.astype(np.int16),
                  group128_centre=gc2.astype(np.int32), group128_members=gm2.astype(np.int32),
                  group128_anchor=ga2.astype(np.int32), group128_region=gr2.astype(np.int16),
                  sub1024=sub.astype(np.int32), rest_points=rest.astype(np.float32), rest_normals=rest_n.astype(np.float32),
                  binding=np.array([binding]), beta_vis_cm=np.float32(beta), delta_col_cm=np.float32(delta),
                  joint_parents=np.asarray(parents, np.int32))
    arrays.update(extra)
    hop = hop_matrix(parents)
    arrays['allowed_anchor_pairs'] = hop >= CONFIG['hop_min']  # provisional (needed by pose for J)
    qr, mean, std = quasi_rigid_pairs(arrays, binding, parents, R_rows, P_rows)
    allowed = (hop >= CONFIG['hop_min']) & ~qr
    arrays['allowed_anchor_pairs'] = allowed
    qr_list = [{'a': names[a], 'b': names[b], 'hop': int(hop[a, b]), 'mean_cm': float(mean[a, b]), 'std_cm': float(std[a, b])}
               for a in range(J) for b in range(a + 1, J) if qr[a, b] and hop[a, b] >= CONFIG['hop_min']]
    man['self_contact'] = {'hop_min': CONFIG['hop_min'], 'n_pairs_hop_ge_min': int(np.triu(hop >= CONFIG['hop_min'], 1).sum()),
                           'quasi_rigid_removed': qr_list, 'n_allowed_pairs': int(np.triu(allowed, 1).sum()),
                           'rows': int(len(R_rows)), 'rule': f"mean closest-surface distance < {CONFIG['quasi_rigid_mean_cm']} cm "
                                                             f"and std < {CONFIG['quasi_rigid_std_cm']} cm over the rows"}
    from scipy.spatial import cKDTree
    nn = cKDTree(rest).query(rest, k=2)[0][:, 1]
    reg_counts = {E.REGIONS[r]: int((region == r).sum()) for r in range(len(E.REGIONS)) if (region == r).any()}
    man['coverage'].update(point_spacing_nn_cm={'p50': float(np.median(nn)), 'p95': float(np.percentile(nn, 95)),
                                                'max': float(nn.max())},
                           region_counts=reg_counts, region_area_cm2={E.REGIONS[r]: float(area[region == r].sum()) for r in range(len(E.REGIONS)) if (region == r).any()},
                           anchors_with_points=int(len(np.unique(anchor))), J=J,
                           group_size_cm={'p50_radius': float(np.median(np.linalg.norm(rest[gm] - rest[gc][:, None], axis=-1).max(1)))})
    man['beta_vis_cm'] = float(beta); man['delta_col_cm'] = None if not np.isfinite(delta) else float(delta)
    return arrays, man


def write_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


def content_sha(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
def build_robot(body, am2):
    from . import robot_binding as RB
    from . import regions as RG
    t0 = time.time()
    kin = RB.load_kinematics(body)
    parts, audit = RB.bind_geoms(kin, n_random=CONFIG['verify_random_states'])
    q0 = kin.model.qpos0.copy()
    if body == 'relay_franka_full_v2':
        # fingers closed at qpos0 hide the inner pads from every view: sample with the gripper fully open
        q0 = kin.open_slides_qpos()
    R0, P0, _, gp0, gr0 = kin.forward(q0)
    V, F, face_part = RB.assembled_mesh(kin, parts, R0, P0)
    s = sample(V, F)
    n_world, flip_frac = oriented_normals(V, F, s)
    part_idx = face_part[s.face_ids]
    anchor = np.array([parts[i]['anchor'] for i in part_idx], np.int64)
    local = np.einsum('pji,pj->pi', R0[anchor], s.points - P0[anchor])          # R0^T (x - P0)
    local_n = np.einsum('pji,pj->pi', R0[anchor], n_world)
    Prest = kin.rest_positions()
    rest = local + Prest[anchor]
    area = point_areas(s)
    # geom-frame coordinate of each point (canonical cm) consistent with its binding: v = mat^T (local - trans)
    geom_v = np.stack([parts[i]['mat'].T @ (l - parts[i]['trans']) for i, l in zip(part_idx, local)])
    pt_geom = np.array([parts[i]['geom'] for i in part_idx], np.int32)
    pt_rigid = np.array([parts[i]['rigid'] for i in part_idx], bool)
    pt_res = np.array([parts[i]['residual_cm'] for i in part_idx], np.float32)
    names = list(kin.names); parents = [int(x) for x in kin.parents]
    # regions (q0 pose: vendor neutral, upright)
    man = {'body': body, 'binding': 'rigid', 'J': kin.J, 'names': names, 'parents': parents}
    if body == 'adroit_pen':
        region, rtable = RG.adroit_regions(names, anchor)
    elif body == 'relay_franka_full_v2':
        region, rtable = RG.franka_regions(names, anchor, [parts[i]['mesh_name'] for i in part_idx])
    else:
        region, rtable = RG.humanoid_regions(names, parents, P0, s.points, anchor)
    pad = np.full(len(local), -1, np.int8); pad_rep = None
    if body == 'adroit_pen':
        from .pads import adroit_pads
        pad, pad_rep = adroit_pads(kin, local, local_n, anchor, area)
    elif body == 'relay_franka_full_v2':
        from .pads import franka_pads
        pad, pad_rep = franka_pads(kin, local, local_n, anchor, area)
    # data-state verification (xwbc)
    st, msg = RB.data_states(kin) if body in E.BODIES_XWBC_TRAIN + E.BODIES_XWBC_HELDOUT else ([], 'n/a (fixed-base payload; binding is exact)')
    dres = RB.data_residual(kin, parts, st)
    pt_data_res = (np.array([dres['per_part_p95_cm'][i] for i in part_idx], np.float32) if dres
                   else np.full(len(local), np.nan, np.float32))
    if dres:
        dres['points_with_data_p95_gt_1cm'] = int((pt_data_res > 1.0).sum())
        dres['points_with_data_p95_gt_3cm'] = int((pt_data_res > 3.0).sum())
        dres['approx_parts'] = [{'body': parts[i]['body_name'], 'anchor': names_i, 'rule': parts[i].get('fit_rule'),
                                 'points': int((part_idx == i).sum()), 'data_p95_cm': dres['per_part_p95_cm'][i],
                                 'data_max_cm': dres['per_part_max_cm'][i]}
                                for i, names_i in ((i, kin.names[parts[i]['anchor']]) for i in range(len(parts)))
                                if not parts[i]['rigid']]
    if body == 'unitree_h2':
        audit['h2_data_frame_verification'] = RB.h2_data_verification(kin, parts)
    # beta_vis / delta_col
    sa = E.read_json(E.SOLE_AUDIT).get(body, {})
    beta, beta_info = beta_vis_robot(kin, V, F, face_part, parts, P0, sa, names, parents)
    delta = sa.get('predicted_bias_cm')
    delta = float('nan') if delta is None else float(delta)
    # payload rows for quasi-rigid pairs
    Rr, Pr, aux = data_rows(body, CONFIG['quasi_rigid_rows'])
    man.update(dict(
        source=dict(mjcf=kin.xml, mjcf_sha256=E.sha256_file(kin.xml), sole_audit_mjcf=sa.get('mjcf'),
                    meshes=mesh_sources(kin, parts), adapter_kind=kin.kind, adapter_notes=kin.notes),
        sampling=dict(CONFIG, sample_pose=('MJCF qpos0 with the gripper fully open (slides at upper limit)' if body == 'relay_franka_full_v2' else 'MJCF qpos0') + ' assembled through the AM2 binding', sample_qpos=q0.tolist(),
                      candidate_count=s.candidate_count, exterior_count=s.exterior_count,
                      normals_flipped_vs_winding_frac=flip_frac),
        binding_verification=dict({k: v for k, v in audit.items()}, data_states=msg, data_residual=dres,
                                  am2_rest_vs_qpos0_max_pos_cm=float(np.abs(Prest - P0).max())),
        coverage=dict(total_visual_area_cm2=s.total_area, exterior_area_fraction=s.exterior_count / s.candidate_count,
                      represented_area_cm2=float(area.sum()), points=int(len(local))),
        regions_table=rtable, pads=pad_rep, beta_vis_info=beta_info,
        quasi_rigid_rows_source=aux.get('path')))
    extra = dict(point_geom=pt_geom, point_geom_vertex_cm=geom_v.astype(np.float32), point_rigid=pt_rigid,
                 point_residual_cm=pt_res, point_data_residual_p95_cm=pt_data_res, sample_pose_rot=R0.astype(np.float32), sample_pose_pos=P0.astype(np.float32))
    arrays, man = finish_asset(body, parents, names, rest, local_n, local, local_n, anchor, region, pad, area, extra,
                               'rigid', Rr, Pr, beta, delta, man)
    if body in E.BODIES_XWBC_TRAIN + E.BODIES_XWBC_HELDOUT:
        man['beta_data_check'] = beta_data_check(body, arrays, region)
    man['build_seconds'] = time.time() - t0
    return arrays, man


def mesh_sources(kin, parts):
    out = {}
    try:
        recs, _ = RB_bvp().mesh_sources(kin.xml)
    except Exception as e:  # pragma: no cover
        return {'error': repr(e)}
    for p in parts:
        r = recs.get(p['mesh_name'])
        if r is not None:
            out[p['mesh_name']] = r
    return out


def RB_bvp():
    from . import robot_binding as RB
    return RB.bvp()


def beta_vis_robot(kin, V, F, face_part, parts, P0, sa, names, parents):
    """beta = label sole drop - visual lowest-point drop, geometry only, at the vendor neutral pose qpos0."""
    from .regions import children_of, subtree
    ch = children_of(parents)
    part_anchor = np.array([p['anchor'] for p in parts])
    vert_anchor = np.full(len(V), -1)
    vert_anchor[F.reshape(-1)] = np.repeat(part_anchor[face_part], 3)
    if kin.body in E.BODIES_EXTERNAL:
        return 0.0, {'rule': 'fixed-base manipulator: no ground label (root_height_valid=False); beta = 0. Support planes: '
                             'Adroit table at canonical Y=-20 cm, Franka floor at Y=-180 cm (surface audit)'}
    feet = sa.get('label_feet') or []
    drops = sa.get('label_sole_drop_cm') or []
    info = {'pose': 'qpos0', 'geometry': 'visual mesh (NOT collision; report-32 drops are collision-based)'}
    if feet and all(f in names for f in feet):
        per = {}
        for f, ld in zip(feet, drops):
            j = names.index(f)
            sub = subtree(ch, j)
            yv = V[np.isin(vert_anchor, sub), 1]
            dv = float(P0[j, 1] - yv.min())
            per[f] = {'label_drop_cm': float(ld), 'visual_drop_cm': dv, 'beta_cm': float(ld) - dv,
                      'collision_true_drop_cm': (sa.get('true_drop_cm') or {}).get(f)}
        beta = float(np.mean([v['beta_cm'] for v in per.values()]))
        info.update(rule='label_sole_drop - (foot joint y - lowest visual vertex y of the foot subtree)', feet=per)
        return beta, info
    # joint-centre labels (no sole table): ground = lowest joint centre at stance
    jl = int(np.argmin(P0[:, 1]))
    dv = float(P0[jl, 1] - V[:, 1].min())
    info.update(rule='no sole table: label ground = lowest joint centre; beta = -(lowest joint centre y - lowest visual vertex y)',
                lowest_joint=names[jl], visual_drop_cm=dv, sole_audit_j_floor=sa.get('j_floor_name'))
    return -dv, info


def beta_data_check(body, arrays, region):
    """Median over stance payload rows of the lowest foot-surface height with the GT root (uncorrected).
    Should equal beta_vis_cm if the label convention and the geometry agree (independent check, not used)."""
    feet = np.isin(region, [E.RID['L_foot'], E.RID['R_foot']])
    if not feet.any():
        return None

    def mask(z):
        return z['root_height_valid'] & z['ground_valid'] & z['contact_trust'] & z['contact_hard'].any(1)
    R, P, aux = data_rows(body, 1500, mask=mask)
    pts, _ = pose_np(arrays, str(arrays['binding'][0]), R, P)
    h_foot = pts[:, feet, 1].min(1) + 100 * aux['root_height_m']
    h_all = pts[:, :, 1].min(1) + 100 * aux['root_height_m']
    low_is_foot = (pts[:, feet, 1].min(1) <= pts[:, :, 1].min(1) + 1e-6)
    # stance = the lowest surface point is on a foot (in contact by labels)
    return {'rows': int(len(R)), 'median_lowest_foot_height_cm': float(np.median(h_foot)),
            'p25_p75': [float(np.percentile(h_foot, 25)), float(np.percentile(h_foot, 75))],
            'median_lowest_point_height_cm': float(np.median(h_all)),
            'lowest_point_is_foot_frac': float(low_is_foot.mean()),
            'beta_vis_cm': float(arrays['beta_vis_cm']),
            'residual_after_beta_cm': float(np.median(h_foot) - float(arrays['beta_vis_cm'])),
            'note': 'rows with any contact_hard; geometry-only beta is NOT fitted to this'}


# ---------------------------------------------------------------------------------------------------------------
def build_human(am2):
    from . import human_surface as H
    t0 = time.time()
    if am2['names'] != H.NAMES or am2['parents'] != H.PARENTS:
        raise ValueError('AM2 human tree differs from the SMPL-X 22-joint tree')
    target_len = np.linalg.norm(np.asarray(am2['offsets_cm']), axis=1)
    sm = H.load_smplx()
    betas, len0, len1 = H.fit_betas(sm, target_len, CONFIG['human_n_betas'], CONFIG['human_beta_reg'])
    Vb, Jall = H.smplx_tpose(sm, betas)
    root = Jall[0].copy()
    Vb = Vb - root; Jall = Jall - root
    Jp = Jall[:22]
    vb = H.vertex_bones(sm)
    fallback = np.tile([0.0, 0.0, 1.0], (22, 1))
    # vertical bones fall back to the forward axis, horizontal ones to up (SMPL-X T-pose: Y up, Z forward)
    rt, ri, rw, _ = H.frame_tables(fallback)
    u0 = H.bone_frames_np(Jp, rt, ri, rw, fallback)[0]
    fallback = np.where(np.abs(u0[:, 1:2]) > 0.7, np.array([[0, 0, 1.0]]), np.array([[0, 1.0, 0]]))
    frames = H.bone_frames_np(Jp, rt, ri, rw, fallback)
    V2, F2, face_bone, ell = H.build_body_mesh(Vb, Jp, vb, sm['faces'], frames)
    s = sample(V2, F2)
    n_world, flip_frac = oriented_normals(V2, F2, s)
    bones = face_bone[s.face_ids].astype(np.int64)
    pts, nrm, sole_info = H.flatten_soles(s.points, n_world, bones, Jp, frames)
    par = np.asarray(H.PARENTS)
    leaf_len = np.where(par >= 0, np.linalg.norm(Jp - Jp[np.where(par < 0, 0, par)], axis=1), 0.0)
    t, radial, theta, nuvw = H.encode_points(pts, nrm, bones, Jp, frames, leaf_len)
    rest, rest_n = H.decode_points(t, radial, theta, nuvw, bones, Jp, frames, leaf_len)
    enc_err = float(np.abs(rest - pts).max())
    region = H.region_ids(bones)
    area = point_areas(s)
    ch = H.joint_child()
    local = rest - Jp[bones]
    extra = dict(bone_parent=bones.astype(np.int32), bone_child=ch[bones].astype(np.int32), t=t.astype(np.float32),
                 radial=radial.astype(np.float32), theta=theta.astype(np.float32),
                 leaf_len=leaf_len[bones].astype(np.float32), normal_uvw=nuvw.astype(np.float32),
                 joint_child=ch.astype(np.int32), frame_ref_type=rt.astype(np.int32), frame_ref_idx=ri.astype(np.int32),
                 frame_ref_w=rw.astype(np.float32), frame_fallback=fallback.astype(np.float32),
                 calib_joint_positions=Jp.astype(np.float32), smplx_betas=betas.astype(np.float32))
    # beta: joint-centre labels -> ground = lowest joint centre at stance; calibration pose is standing
    jl = int(np.argmin(Jp[:, 1]))
    dv = float(Jp[jl, 1] - rest[:, 1].min())
    beta = -dv
    beta_info = {'rule': 'no sole table (human labels are joint centres): beta = -(lowest joint centre - lowest surface point) '
                         'in the standing calibration pose', 'lowest_joint': H.NAMES[jl], 'visual_drop_cm': dv}
    Rr, Pr, aux = data_rows('human_smpl', CONFIG['quasi_rigid_rows'])
    rules = {H.NAMES[j]: [{'ref': list(map(str, r[0])), 'w': r[1]} for r in rl] for j, rl in H.FRAME_RULES.items()}
    man = {'body': 'human_smpl', 'binding': 'bone_revolution', 'J': 22, 'names': H.NAMES, 'parents': H.PARENTS,
           'source': {'smplx_npz': str(E.SMPLX_NPZ), 'smplx_sha256': E.sha256_file(E.SMPLX_NPZ),
                      'betas': betas.tolist(), 'bone_length_target_cm': target_len[1:].tolist(),
                      'bone_length_smplx_neutral_cm': len0.tolist(), 'bone_length_fitted_cm': len1.tolist(),
                      'height_cm': float(Vb[:, 1].max() - Vb[:, 1].min()), 'hands': ell,
                      'vertex_bone_rule': 'argmax LBS weight; jaw/eyes->Head; fingers->Wrist'},
           'sampling': dict(CONFIG, sample_pose='shaped SMPL-X T-pose (hands replaced by ellipsoids)',
                            candidate_count=s.candidate_count, exterior_count=s.exterior_count,
                            normals_flipped_vs_winding_frac=flip_frac),
           'bone_revolution': {'frame_rules': rules, 'main_child': {H.NAMES[k]: H.NAMES[v] for k, v in H.MAIN_CHILD.items()},
                               'fallback': fallback.tolist(), 'encode_decode_max_err_cm': enc_err, 'soles': sole_info,
                               'note': 'rot is ignored; posing uses joint positions only. t is unbounded (pelvis/heel '
                                       'points lie behind their bone origin, t<0).'},
           'binding_verification': {'kind': 'position-only bone revolution; no MJCF', 'encode_decode_max_err_cm': enc_err},
           'coverage': dict(total_visual_area_cm2=s.total_area, exterior_area_fraction=s.exterior_count / s.candidate_count,
                            represented_area_cm2=float(area.sum()), points=int(len(pts))),
           'regions_table': {'rule': 'region of the bone', 'bone_region': H.HUMAN_BONE_REGION},
           'beta_vis_info': beta_info, 'quasi_rigid_rows_source': aux.get('path')}
    arrays, man = finish_asset('human_smpl', H.PARENTS, H.NAMES, rest, rest_n, local, rest_n, bones, region,
                               np.full(len(pts), -1, np.int8), area, extra, 'bone_revolution', Rr, Pr, beta, float('nan'), man)
    man['beta_data_check'] = beta_data_check('human_smpl', arrays, region)
    man['build_seconds'] = time.time() - t0
    return arrays, man


# ---------------------------------------------------------------------------------------------------------------
def build(body, dump):
    am2 = dump['bodies'][body]
    arrays, man = build_human(am2) if body == 'human_smpl' else build_robot(body, am2)
    # schema guards
    P = arrays['points_local'].shape[0]
    assert P == CONFIG['num_points'], P
    assert arrays['allowed_anchor_pairs'].shape == (am2['J'], am2['J'])
    assert man['names'] == am2['names'] and [int(x) for x in man['parents']] == [int(x) for x in am2['parents']]
    man['am2_body'] = {'config': dump['config'], 'config_sha256': dump['config_sha256'], 'reach_cm': am2['reach_cm'],
                       'domain': am2['domain']}
    man['config_sha256'] = CONFIG_SHA
    man['content_sha256'] = content_sha(arrays)
    npz = E.ASSET_DIR / f'{body}.npz'
    write_npz(npz, arrays)
    man['npz_sha256'] = E.sha256_file(npz)
    E.write_json(E.ASSET_DIR / f'{body}.json', man)
    return man


def write_manifest():
    files, bodies = {}, {}
    for b in E.ALL_BODIES:
        for ext in ('npz', 'json'):
            p = E.ASSET_DIR / f'{b}.{ext}'
            if p.exists():
                files[p.name] = E.sha256_file(p)
        j = E.ASSET_DIR / f'{b}.json'
        if j.exists():
            m = E.read_json(j)
            bv = m.get('binding_verification', {})
            bodies[b] = {'binding': m['binding'], 'content_sha256': m['content_sha256'], 'beta_vis_cm': m['beta_vis_cm'],
                         'delta_col_cm': m['delta_col_cm'], 'exterior_area_fraction': m['coverage']['exterior_area_fraction'],
                         'point_spacing_p50_cm': m['coverage']['point_spacing_nn_cm']['p50'],
                         'binding_max_residual_cm': bv.get('max_residual_cm'),
                         'binding_max_residual_rigid_cm': bv.get('max_residual_rigid_cm'),
                         'data_residual_p95_cm': (bv.get('data_residual') or {}).get('p95_of_frame_max_cm')}
    man = {'schema': 'surface_v2_manifest', 'config': CONFIG, 'config_sha256': CONFIG_SHA, 'bodies': bodies, 'files': files,
           'missing': [b for b in E.ALL_BODIES if b not in bodies], 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'code_sha256': {p.name: E.sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    E.write_json(E.ASSET_DIR / 'manifest.json', man)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--body', action='append', default=[])
    a = ap.parse_args(argv)
    bodies = E.ALL_BODIES if a.all else a.body
    if not bodies:
        ap.error('--all or --body')
    dump = ensure_body_dump()
    for b in bodies:
        t = time.time()
        m = build(b, dump)
        bv = m['binding_verification']
        print(json.dumps({'body': b, 'seconds': round(time.time() - t, 1), 'beta_vis_cm': round(m['beta_vis_cm'], 2),
                          'ext_frac': round(m['coverage']['exterior_area_fraction'], 3),
                          'spacing_p50': round(m['coverage']['point_spacing_nn_cm']['p50'], 2),
                          'max_res_cm': bv.get('max_residual_cm'), 'regions': m['coverage']['region_counts']}), flush=True)
    write_manifest()


if __name__ == '__main__':
    main()
