"""WP-ASSETS: built surface_v2 assets + runtime API (CPU).  Needs OUT/assets/surface_v2 from
`PY assets/build_surface_v2.py --all` (tests skip otherwise)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets import _env as E  # noqa: E402
from assets import surface_v2 as S  # noqa: E402

built = pytest.mark.skipif(not all((E.ASSET_DIR / f'{b}.npz').exists() for b in E.ALL_BODIES),
                           reason='run assets/build_surface_v2.py --all first')
DUMP = E.CACHE_DIR / 'am2_bodies.json'


def _dump():
    return E.read_json(DUMP)['bodies']


# ------------------------------------------------------------------------------------------------ runtime API (synthetic)
def _synthetic_rigid(tmp_path):
    rng = np.random.default_rng(0)
    P, J = 4096, 3
    a = dict(points_local=rng.normal(size=(P, 3)).astype(np.float32), normals_local=np.tile([[0, 1, 0]], (P, 1)).astype(np.float32),
             anchor=rng.integers(0, J, P).astype(np.int32), region=np.zeros(P, np.int16), pad=np.full(P, -1, np.int8),
             area=np.ones(P, np.float32), group_centre=np.arange(256, dtype=np.int32),
             group_members=np.tile(np.arange(16, dtype=np.int32), (256, 1)), group_anchor=np.zeros(256, np.int32),
             group_region=np.zeros(256, np.int16), group128_centre=np.arange(128, dtype=np.int32),
             group128_members=np.tile(np.arange(32, dtype=np.int32), (128, 1)), group128_anchor=np.zeros(128, np.int32),
             group128_region=np.zeros(128, np.int16), sub1024=np.arange(1024, dtype=np.int32),
             allowed_anchor_pairs=np.zeros((J, J), bool), rest_points=np.zeros((P, 3), np.float32),
             rest_normals=np.zeros((P, 3), np.float32), binding=np.array(['rigid']), beta_vis_cm=np.float32(1.5),
             delta_col_cm=np.float32(np.nan))
    np.savez(tmp_path / 'syn.npz', **a)
    return S.load_surface('syn', asset_dir=tmp_path)


def test_rigid_pose_formula_batched_differentiable(tmp_path):
    asset = _synthetic_rigid(tmp_path)
    assert asset.points_local.dtype == torch.float32 and asset.anchor.dtype == torch.int64
    B, J = 4, 3
    rot = torch.linalg.qr(torch.randn(B, J, 3, 3))[0].requires_grad_(True)
    pos = torch.randn(B, J, 3, requires_grad=True)
    p, n = S.pose_surface(asset, rot, pos)
    a = asset.anchor
    ref = torch.einsum('bpij,pj->bpi', rot[:, a], asset.points_local) + pos[:, a]
    assert p.shape == (B, 4096, 3) and torch.allclose(p, ref, atol=1e-5)
    (p.sum() + n.sum()).backward()
    assert torch.isfinite(rot.grad).all() and pos.grad.abs().sum() > 0
    idx = torch.tensor([5, 9, 4000])
    ps, ns = S.pose_surface_subset(asset, rot.detach(), pos.detach(), idx)
    assert torch.allclose(ps, p.detach()[:, idx]) and torch.allclose(ns, n.detach()[:, idx])
    h = S.ground_clearance(p.detach(), torch.tensor([0.9, 1.0, 1.1, 0.5]), asset.beta_vis_cm)
    assert torch.allclose(h, p.detach()[..., 1] + torch.tensor([90., 100., 110., 50.])[:, None] - 1.5)


# ------------------------------------------------------------------------------------------------ built assets
@built
def test_schema_all_bodies():
    from assets.validate_assets import schema_all
    r = schema_all()
    assert r['passed'], r['errors']


@built
def test_names_parents_match_am2_and_manifest():
    d = _dump()
    man = E.read_json(E.ASSET_DIR / 'manifest.json')
    assert not man['missing'] and set(man['bodies']) == set(E.ALL_BODIES)
    for b in E.ALL_BODIES:
        m = E.read_json(E.ASSET_DIR / f'{b}.json')
        assert m['names'] == d[b]['names'] and [int(x) for x in m['parents']] == d[b]['parents']
        assert man['files'][f'{b}.npz'] == E.sha256_file(E.ASSET_DIR / f'{b}.npz')
        assert m['sampling']['num_points'] == 4096 and m['sampling']['oversample_ratio'] == 8
        assert m['sampling']['candidate_multiplier'] == 12 and m['sampling']['min_visible_views'] == 2


@built
@pytest.mark.parametrize('body', E.ROBOTS)
def test_rigid_identity_pose_equals_rest_points(body):
    asset = S.load_surface(body)
    assert asset.binding == 'rigid'
    J = asset.num_joints
    d = _dump()[body]
    off = np.asarray(d['offsets_cm'])
    P = np.zeros((J, 3))
    for j, p in enumerate(d['parents']):
        if p >= 0:
            P[j] = P[p] + off[j]
    if body == 'relay_franka_full_v2':   # rest = slides at their reference (already in the AM2 offsets)
        pass
    rot = torch.eye(3).expand(1, J, 3, 3)
    p, n = S.pose_surface(asset, rot, torch.as_tensor(P[None], dtype=torch.float32))
    assert (p[0] - asset.rest_points).abs().max() < 1e-3
    assert (n[0] - asset.rest_normals).abs().max() < 1e-5


@built
@pytest.mark.parametrize('body', E.ROBOTS)
def test_fk_consistency_vs_mujoco(body):
    """3 fresh random q: rigid points within 0.5 mm of MuJoCo; approximate points within 1.5x the body's documented
    max residual (verification states)."""
    from assets.validate_assets import robot_fk
    r = robot_fk(body, n_q=3, seed=11)
    assert r['passed_rigid_0.5mm'], r
    if r['n_approx_points']:
        doc = E.read_json(E.ASSET_DIR / f'{body}.json')['binding_verification']['max_residual_cm']
        assert r['approx_max_err_cm'] <= 1.5 * doc + 0.1, (r, doc)   # body-level documented bound


@built
def test_human_bone_revolution_positions_only():
    asset = S.load_surface('human_smpl')
    assert asset.binding == 'bone_revolution'
    J = asset.num_joints
    Jp = torch.as_tensor(asset.extra['calib_joint_positions'])[None]
    p0, n0 = S.pose_surface(asset, torch.eye(3).expand(1, J, 3, 3), Jp)
    assert (p0[0] - asset.rest_points).abs().max() < 2e-3
    # rotations are ignored (LAFAN twist unobservable)
    p1, _ = S.pose_surface(asset, torch.linalg.qr(torch.randn(1, J, 3, 3))[0], Jp)
    assert torch.equal(p0, p1)
    # rigid motion of the whole skeleton moves the surface rigidly
    Rz = torch.tensor([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    t = torch.tensor([3., -2, 5])
    p2, n2 = S.pose_surface(asset, torch.eye(3).expand(1, J, 3, 3), Jp @ Rz.T + t)
    assert (p2[0] - (p0[0] @ Rz.T + t)).abs().max() < 0.05   # fixed 1e-3 fallback term is not rotated
    assert (n2[0] - n0[0] @ Rz.T).abs().max() < 5e-3
    # differentiable wrt joint positions, batched
    pos = Jp.repeat(3, 1, 1).clone().requires_grad_(True)
    p, _ = S.pose_surface(asset, torch.eye(3).expand(3, J, 3, 3), pos)
    p[..., 1].min().backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0


@built
def test_human_acceptance_numbers():
    v = E.read_json(E.ASSET_DIR / 'validation' / 'human.json')
    for pose in ('tpose_calibration', 'standing_apose'):
        c = v['chamfer'][pose]
        assert c['points_to_mesh_cm']['median'] < 1.5 and c['mesh_vertices_to_points_cm']['median'] < 1.5
    st = v['stance']
    assert st['lowest_point_is_foot_frac'] > 0.9
    assert abs(st['lowest_sole_height_after_beta_cm']['median']) < 2.5


@built
def test_regions_cover_body_parts():
    for b in E.BODIES_XWBC_TRAIN + E.BODIES_XWBC_HELDOUT:
        reg = np.load(E.ASSET_DIR / f'{b}.npz')['region']
        present = set(np.unique(reg).tolist())
        assert set(range(15)) <= present, (b, sorted(set(range(15)) - present))
    reg = np.load(E.ASSET_DIR / 'adroit_pen.npz')['region']
    assert {E.RID['digit'], E.RID['palm'], E.RID['base_or_link']} <= set(np.unique(reg).tolist())


@built
def test_pads():
    z = np.load(E.ASSET_DIR / 'adroit_pen.npz')
    names = _dump()['adroit_pen']['names']
    for k, dg in enumerate(['th', 'ff', 'mf', 'rf', 'lf']):
        sel = z['pad'] == k
        assert sel.sum() >= 5 and np.all(z['anchor'][sel] == names.index(f'{dg}distal'))
    m = E.read_json(E.ASSET_DIR / 'adroit_pen.json')['pads']
    assert m['verified_within_3mm']
    z = np.load(E.ASSET_DIR / 'relay_franka_full_v2.npz')
    names = _dump()['relay_franka_full_v2']['names']
    for k, n in enumerate(['panda0_leftfinger', 'panda0_rightfinger']):
        sel = z['pad'] == k
        assert sel.sum() >= 5 and np.all(z['anchor'][sel] == names.index(n))
    assert E.read_json(E.ASSET_DIR / 'relay_franka_full_v2.json')['pads']['verified_within_3mm']


@built
def test_groups_pairs_beta():
    from assets.build_surface_v2 import hop_matrix
    d = _dump()
    for b in E.ALL_BODIES:
        z = np.load(E.ASSET_DIR / f'{b}.npz')
        hop = hop_matrix(d[b]['parents'])
        allowed = z['allowed_anchor_pairs']
        assert not np.any(allowed & (hop < 3))
        assert len(np.unique(z['group_centre'])) == 256 and len(np.unique(z['group128_centre'])) == 128
        m = E.read_json(E.ASSET_DIR / f'{b}.json')
        if b in E.BODIES_EXTERNAL:
            assert float(z['beta_vis_cm']) == 0.0
        else:
            c = m['beta_data_check']
            assert abs(c['median_lowest_foot_height_cm'] - float(z['beta_vis_cm'])) < 2.0, (b, c)


@built
def test_load_surface_device_and_dtype():
    a = S.load_surface('unitree_h2', device='cpu')
    assert a.points_local.shape == (4096, 3) and a.group_members.shape == (256, 16)
    assert a.allowed_anchor_pairs.dtype == torch.bool and a.sub1024.dtype == torch.int64
    if torch.cuda.is_available():
        b = a.to('cuda')
        assert b.points_local.is_cuda
