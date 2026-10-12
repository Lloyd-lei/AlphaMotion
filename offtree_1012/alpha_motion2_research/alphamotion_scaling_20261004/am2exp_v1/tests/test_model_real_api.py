"""WP-MODEL x WP-ASSETS integration through the REAL loader/poser (am2exp_v1.assets.surface_v2).

Schema-conformant npz files are written to a tmp dir (synthetic geometry, real file format):
  * unitree_g1_29dof : binding 'rigid'
  * human_smpl       : binding 'bone_revolution' (positions-only posing; frame refs use the hip axis,
                       so every point depends on >2 joints -> exercises the numeric dependency mask)
They are loaded with assets.surface_v2.load_surface(asset_dir=tmp) via model.surface_api and pushed
through S / I / O.  The real OUT/assets/surface_v2 acceptance tests live in test_model_relational.py.
"""
import numpy as np
import pytest
import torch
from ._model_helpers import M, CPU, sa, build, loaded_bodies, inputs, fk, outputs_equal, copy_shared, perturb

ALL_KEYS = ('rot6', 'raw_hpos', 'contact_logits', 'slide_normalized', 'slide_native', 'joint_position', 'joint_rotation')
real = pytest.importorskip('am2exp_v1.assets.surface_v2')


def _grouping(rest, anchor, J, G, K, seed=0):
    centre = sa.farthest_point_sampling(rest, G, seed)
    d = np.linalg.norm(rest[centre][:, None] - rest[None], axis=-1)
    mem = np.argsort(d, axis=1, kind='stable')[:, :K]; mem[:, 0] = centre
    ganc = np.array([np.bincount(anchor[m], minlength=J).argmax() for m in mem])
    return centre, mem, ganc


def _npz_from_fallback(fa):
    keys = ['points_local', 'normals_local', 'anchor', 'region', 'pad', 'area', 'group_centre', 'group_members',
            'group_anchor', 'group_region', 'group128_centre', 'group128_members', 'group128_anchor', 'group128_region',
            'sub1024', 'allowed_anchor_pairs', 'rest_points']
    out = {k: getattr(fa, k).cpu().numpy() for k in keys}
    out['rest_normals'] = out['normals_local'].copy()
    out['binding'] = np.array(['rigid']); out['beta_vis_cm'] = np.float32(0.0); out['delta_col_cm'] = np.float32(np.nan)
    return out


def _bone_revolution_npz(body, P=2048, seed=0):
    rng = np.random.default_rng(seed)
    parents = np.asarray(body.parents); J = len(parents)
    Prest = sa.rest_joint_positions(body.parents, body.offsets)
    child = np.full(J, -1)
    for j in range(J - 1, 0, -1): child[parents[j]] = j  # main child = lowest-index child
    joint = rng.integers(0, J, P)
    t = rng.random(P).astype(np.float32); th = (rng.random(P) * 2 * np.pi).astype(np.float32)
    radial = np.full(P, 4.0, np.float32); leaf_len = np.full(P, 8.0, np.float32)
    nuvw = np.stack([np.zeros(P), np.cos(th), np.sin(th)], 1).astype(np.float32)
    names = list(body.names); lh, rh = names.index('L_Hip'), names.index('R_Hip')
    ref_type = np.zeros((J, 1), np.int64); ref_idx = np.zeros((J, 1, 4), np.int64); ref_idx[:, 0, 0] = lh; ref_idx[:, 0, 1] = rh
    a = {'joint_parents': parents.astype(np.int64), 'joint_child': child, 'frame_ref_type': ref_type, 'frame_ref_idx': ref_idx,
         'frame_ref_w': np.ones((J, 1), np.float32), 'frame_fallback': np.tile(np.array([0, 0, 1.], np.float32), (J, 1)),
         'bone_parent': joint.astype(np.int64), 'bone_child': child[joint], 't': t, 'radial': radial, 'theta': th,
         'leaf_len': leaf_len, 'normal_uvw': nuvw}
    # pose at rest with the real poser to get consistent rest points
    tmp = real.SurfaceAsset(body=body.name, binding='bone_revolution', points_local=torch.zeros(P, 3), normals_local=torch.zeros(P, 3),
                            anchor=torch.as_tensor(joint), region=torch.zeros(P, dtype=torch.long), pad=torch.full((P,), -1),
                            area=torch.ones(P), group_centre=None, group_members=None, group_anchor=None, group_region=None,
                            group128_centre=None, group128_members=None, group128_anchor=None, group128_region=None,
                            sub1024=None, allowed_anchor_pairs=None, rest_points=None, rest_normals=None, beta_vis_cm=0.0,
                            delta_col_cm=float('nan'), num_joints=J,
                            **{k: torch.as_tensor(v) for k, v in a.items()})
    pts, nrm = real.pose_surface(tmp, torch.eye(3).expand(1, J, 3, 3), torch.as_tensor(Prest, dtype=torch.float32)[None])
    rest = pts[0].numpy(); normals = nrm[0].numpy()
    region = np.array([sa._region_for(names[j]) for j in joint], np.int64)
    c, m, ga = _grouping(rest, joint, J, 256, 16); c2, m2, ga2 = _grouping(rest, joint, J, 128, 32)
    a.update({'points_local': (rest - Prest[joint]).astype(np.float32), 'normals_local': normals, 'rest_normals': normals,
              'anchor': joint, 'region': region, 'pad': np.full(P, -1), 'area': np.ones(P, np.float32),
              'group_centre': c, 'group_members': m, 'group_anchor': ga, 'group_region': region[c],
              'group128_centre': c2, 'group128_members': m2, 'group128_anchor': ga2, 'group128_region': region[c2],
              'sub1024': sa.farthest_point_sampling(rest, 512, 1), 'allowed_anchor_pairs': sa.hop_matrix(parents) >= 3,
              'rest_points': rest.astype(np.float32), 'binding': np.array(['bone_revolution']),
              'beta_vis_cm': np.float32(0.0), 'delta_col_cm': np.float32(np.nan)})
    return a


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp('surface_v2')
    bodies, _ = loaded_bodies(); by = {b.name: b for b in bodies}
    g1 = by['unitree_g1_29dof']; hu = by['human_smpl']
    np.savez(d / 'unitree_g1_29dof.npz', **_npz_from_fallback(sa.build_synthetic_asset(g1.name, g1.parents, g1.offsets, g1.names)))
    np.savez(d / 'human_smpl.npz', **_bone_revolution_npz(hu))
    M.detach_dense_geometry(bodies)
    rep = M.attach_dense_geometry(bodies, CPU, asset_dir=d, require=['unitree_g1_29dof', 'human_smpl'])
    sf = build('SFMegatron'); s = build('DenseSurfaceMegatron'); i = build('RelationalInputMegatron')
    yield by, rep, sf, s, i
    from ._model_helpers import bodies_with_synthetic_dense
    bodies_with_synthetic_dense()  # restore the default synthetic attachment for later test modules


def test_real_loader_used_and_bindings(env):
    by, rep, *_ = env
    assert rep['surface_api'] == 'assets.surface_v2'
    assert set(rep['bodies']) == {'unitree_g1_29dof', 'human_smpl'}
    st = by['human_smpl'].geom(1)['dense_surface']['static']
    assert isinstance(st['asset'], real.SurfaceAsset) and st['binding'] == 'bone_revolution'
    dep = st['point_dep']
    lh, rh = list(by['human_smpl'].names).index('L_Hip'), list(by['human_smpl'].names).index('R_Hip')
    # every bone-revolution point depends on its bone joint and on both hip joints (frame reference)
    assert bool(dep[:, lh].all()) and bool(dep[:, rh].all()) and int(dep.sum(1).min()) >= 2


@pytest.mark.parametrize('pair', [('human_smpl', 'unitree_g1_29dof'), ('unitree_g1_29dof', 'human_smpl'), ('human_smpl', 'human_smpl')])
@pytest.mark.parametrize('mode', ['full', 'masked'])
def test_identity_at_init_real_api(env, pair, mode):
    by, _, sf, s, i = env
    src, tgt = by[pair[0]], by[pair[1]]
    rows, inp, _ = inputs(src, tgt, B=10, mode=mode, seed=31)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], CPU)
    with torch.no_grad():
        a = fk(sf, src, tgt, inp); b = fk(s, src, tgt, inp); c = fk(i, src, tgt, inp, source_root_height_m=rh)
    assert all(outputs_equal(a, b, ALL_KEYS).values()) and all(outputs_equal(b, c, ALL_KEYS).values())


def test_bone_revolution_hidden_joint_invariance(env):
    """Hidden human joints (NaN or garbage) cannot leak through bone-revolution relational features."""
    by, _, _, s, _ = env
    m = build('RelationalInputMegatron'); copy_shared(s, m); perturb(m, 0.02, 41)
    src, tgt = by['human_smpl'], by['unitree_g1_29dof']
    for seed in (42, 43, 44):
        rows, inp, _ = inputs(src, tgt, B=12, mode='masked', seed=seed)
        rh = torch.as_tensor(src.arrays['root_height_m'][rows]).float()
        hid = inp['source_hidden']
        with torch.no_grad():
            y = fk(m, src, tgt, inp, source_root_height_m=rh)
            for fill in (float('nan'), None):
                inp2 = dict(inp)
                junk = torch.full_like(inp['source'], fill) if fill is not None else torch.randn_like(inp['source']) * 40
                inp2['source'] = torch.where(hid[..., None], junk, inp['source'])
                z = fk(m, src, tgt, inp2, source_root_height_m=rh)
                assert all(outputs_equal(y, z, ALL_KEYS).values()), seed
    g = src.geom(4); st = g['dense_surface']['static']
    x = src.tensor('pose9', np.arange(4), CPU).float()
    feat, valid = M.relational_features(x, None, g, st, torch.as_tensor(src.arrays['root_height_m'][:4]).float())
    assert torch.isfinite(feat).all() and bool(valid.all())


def test_relational_objective_bone_revolution_target(env):
    from ._model_helpers import cfg_with_order
    rot6d_to_matrix = M._sealed.rot6d_to_matrix
    by, *_ = env
    tgt = by['human_smpl']; w = cfg_with_order()['loss']
    ok = np.flatnonzero(M.row_ground_ok(tgt, np.arange(len(tgt.arrays['pose9'])), CPU).numpy())[::1500][:12]
    target = tgt.tensor('pose9', ok, CPU).float()
    pos = (target[..., 6:] * tgt.reach).clone().requires_grad_(True)
    out = {'rot6': target[..., :6], 'joint_rotation': rot6d_to_matrix(target[..., :6]), 'joint_position': pos,
           'contact_logits': torch.zeros(len(ok), len(tgt.parents), requires_grad=True)}
    r = M.relational_objective(out, target, tgt, ok, w)
    assert int(r['count_relational_rows']) == len(ok) > 0
    assert float(r['relational_pen']) == 0.0 and float(r['relational_clear']) < 0.05
    sunk = dict(out); sunk['joint_position'] = (pos.detach() - torch.tensor([0., 4., 0.])).requires_grad_(True)
    r2 = M.relational_objective(sunk, target, tgt, ok, w)
    r2['total'].backward()
    assert float(r2['relational_pen']) > 0 and float(sunk['joint_position'].grad[..., 1].sum()) < 0
