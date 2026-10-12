"""WP-MODEL arm S / I / O identity, masking and leakage tests (CPU, synthetic dense assets)."""
import numpy as np
import pytest
import torch
from ._model_helpers import (M, CPU, build, bodies_with_synthetic_dense, loaded_bodies, inputs, fk, outputs_equal,
                             copy_shared, perturb, cfg_with_order)

PAIRS = [('human_smpl', 'unitree_g1_29dof'), ('unitree_g1_29dof', 'human_smpl'), ('booster_t1', 'fourier_gr3'),
         ('relay_franka_full_v2', 'relay_franka_full_v2'), ('adroit_pen', 'adroit_pen')]
ALL_KEYS = ('rot6', 'raw_hpos', 'contact_logits', 'slide_normalized', 'slide_native', 'joint_position', 'joint_rotation')


@pytest.fixture(scope='module')
def env():
    by, audit, report = bodies_with_synthetic_dense()
    sf = build('SFMegatron'); s = build('DenseSurfaceMegatron'); i = build('RelationalInputMegatron')
    return by, sf, s, i


def active(cls, s, seed, roll=False):
    """Copy of S with the zero-initialised gates opened (as after the first optimizer steps)."""
    m = build(cls, **({'roll_relations': True} if roll else {})); copy_shared(s, m)
    with torch.no_grad():
        for z in m.group_to_joint: z.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(seed))
        m.target_dense_attachment.out.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(seed + 1))
    if hasattr(m, 'relational_mlp'): perturb(m.relational_mlp, 0.2, seed + 2)
    return m


def test_parent_state_dicts_bit_equal(env):
    _, sf, s, i = env
    for parent, child in [(sf, s), (s, i)]:
        a, b = parent.state_dict(), child.state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert set(b) > set(a)


@pytest.mark.parametrize('mode', ['full', 'masked'])
@pytest.mark.parametrize('pair', PAIRS)
def test_forward_bit_equal_at_init_S_vs_SF_and_I_vs_S(env, pair, mode):
    by, sf, s, i = env
    src, tgt = by[pair[0]], by[pair[1]]
    assert src.geom(2).get('dense_surface') is not None and tgt.geom(2).get('dense_surface') is not None
    rows, inp, _ = inputs(src, tgt, B=12, mode=mode, seed=7)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], CPU)
    with torch.no_grad():
        o_sf = fk(sf, src, tgt, inp); o_s = fk(s, src, tgt, inp)
        o_i = fk(i, src, tgt, inp, source_root_height_m=rh)
        o_i_roll = fk(i, src, tgt, inp, source_root_height_m=rh, roll_relations=True)
    for a, b in [(o_sf, o_s), (o_s, o_i), (o_s, o_i_roll)]:
        eq = outputs_equal(a, b, ALL_KEYS)
        assert all(eq.values()), eq


def test_O_arm_model_is_S_and_objective_runs(env):
    import train_scaling as original
    by, sf, s, _ = env
    cfg = cfg_with_order()
    src, tgt = by['human_smpl'], by['unitree_g1_29dof']
    rows, inp, target = inputs(src, tgt, B=16, seed=8)
    o_s = build('DenseSurfaceMegatron'); o_s.train()  # arm O builds the very same class as S
    out = fk(o_s, src, tgt, inp)
    obj = M.make_objective(original.loss_function(cfg), slide_fix=True, relational=True)
    loss, terms = obj(out, target, tgt, rows, inp['condition'], cfg['loss'])
    assert torch.isfinite(loss)
    for k in ('relational_pen', 'relational_clear', 'relational_hcon', 'count_relational_rows'):
        assert k in terms
    assert int(terms['count_relational_rows']) > 0
    loss.backward()
    assert float(o_s.hcon[1].weight.grad.abs().sum()) > 0  # hcon now supervised


def test_bodies_without_dense_surface_equal_SF(env):
    by, sf, s, i = env
    src, tgt = by['unitree_h1'], by['booster_k1']  # no assets attached
    assert src.geom(2).get('dense_surface') is None and tgt.geom(2).get('dense_surface') is None
    s2 = build('DenseSurfaceMegatron'); i2 = build('RelationalInputMegatron')
    perturb(s2, 0.05, 1); copy_shared(sf, s2)  # perturb only the S-only modules
    perturb(i2, 0.05, 2); copy_shared(sf, i2)
    rows, inp, _ = inputs(src, tgt, B=10, mode='masked', seed=9)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], CPU)
    with torch.no_grad():
        a = fk(sf, src, tgt, inp); b = fk(s2, src, tgt, inp); c = fk(i2, src, tgt, inp, source_root_height_m=rh)
    assert all(outputs_equal(a, b, ALL_KEYS).values()) and all(outputs_equal(a, c, ALL_KEYS).values())


def test_mixed_dense_and_plain(env):
    by, sf, s, _ = env
    for sn, tn in [('human_smpl', 'unitree_h1'), ('unitree_h1', 'unitree_g1_29dof')]:
        src, tgt = by[sn], by[tn]
        _, inp, _ = inputs(src, tgt, B=6, seed=10)
        with torch.no_grad():
            assert all(outputs_equal(fk(sf, src, tgt, inp), fk(s, src, tgt, inp), ALL_KEYS).values())


def test_bare_encode_decode_work_and_never_use_relations(env):
    by, sf, s, i = env
    s2 = active('DenseSurfaceMegatron', s, 3)
    i2 = build('RelationalInputMegatron'); copy_shared(s2, i2); perturb(i2.relational_mlp, 0.2, 3)
    src, tgt = by['unitree_g1_29dof'], by['human_smpl']
    x = src.tensor('pose9', np.arange(8), CPU).float()
    with torch.no_grad():
        zp, zr = i2.encode(x, src.geom(8)); zp2, zr2 = s2.encode(x, src.geom(8))
        assert torch.equal(zp, zp2) and torch.equal(zr, zr2)
        rot, pos = i2.decode(zp, zr, tgt.geom(8))
        assert rot.shape == (8, len(tgt.parents), 6) and pos.shape == (8, len(tgt.parents), 3)
        # cycle path: re-encode [rot, raw_hpos] bare
        zp3, zr3 = i2.encode(torch.cat([rot, pos], -1), tgt.geom(8))
        assert torch.isfinite(zp3).all() and torch.isfinite(zr3).all()
        f1 = i2.encode_features(x, src.geom(8)); assert torch.isfinite(f1[0]).all()
        # forward with relations differs once the relational MLP is non-zero
        rows = np.arange(8)
        rh = torch.as_tensor(src.arrays['root_height_m'][rows]).float()
        o_rel = i2.forward_kinematic(x, src.geom(8), tgt.geom(8), source_root_height_m=rh)
        o_s = s2.forward_kinematic(x, src.geom(8), tgt.geom(8))
        assert not torch.equal(o_rel['rot6'], o_s['rot6'])


@pytest.mark.parametrize('cls', ['DenseSurfaceMegatron', 'RelationalInputMegatron'])
def test_nan_fill_hidden_source_and_root_height_sparse_only(env, cls):
    by, sf, s, _ = env
    m = build(cls); copy_shared(s, m); perturb(m, 0.02, 4)  # all branches active
    for sn, tn in [('unitree_g1_29dof', 'unitree_g1_29dof'), ('human_smpl', 'human_smpl'), ('relay_franka_full_v2', 'relay_franka_full_v2')]:
        src, tgt = by[sn], by[tn]
        rows, inp, _ = inputs(src, tgt, B=12, mode='sparse_only', seed=11, condition_count=min(2, len(tgt.endpoint['indices'])))
        rh = torch.as_tensor(src.arrays['root_height_m'][rows]).float()
        with torch.no_grad():
            y = fk(m, src, tgt, inp, source_root_height_m=rh)
            inp2 = dict(inp); inp2['source'] = torch.full_like(inp['source'], float('nan'))
            if 'source_slide_normalized' in inp2: inp2['source_slide_normalized'] = torch.full_like(inp2['source_slide_normalized'], float('nan'))
            z = fk(m, src, tgt, inp2, source_root_height_m=torch.full_like(rh, float('nan')))
        assert all(outputs_equal(y, z, ALL_KEYS).values()), (sn, outputs_equal(y, z, ALL_KEYS))


def test_nan_fill_hidden_joints_partial_mask_I(env):
    """Hidden joints (masked mode) carry no information, also through relational features."""
    by, _, s, _ = env
    m = active('RelationalInputMegatron', s, 5)
    src, tgt = by['unitree_g1_29dof'], by['human_smpl']
    rows, inp, _ = inputs(src, tgt, B=16, mode='masked', seed=12)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], CPU)
    assert bool(inp['source_hidden'].any()) and not bool(inp['source_hidden'].all())
    with torch.no_grad():
        y = fk(m, src, tgt, inp, source_root_height_m=rh)
        inp2 = dict(inp)
        inp2['source'] = torch.where(inp['source_hidden'][..., None], torch.full_like(inp['source'], float('nan')), inp['source'])
        z = fk(m, src, tgt, inp2, source_root_height_m=rh)
        # and garbage (finite) instead of NaN at hidden joints
        inp3 = dict(inp)
        inp3['source'] = torch.where(inp['source_hidden'][..., None], torch.randn_like(inp['source']) * 50, inp['source'])
        w = fk(m, src, tgt, inp3, source_root_height_m=rh)
    assert all(outputs_equal(y, z, ALL_KEYS).values()) and all(outputs_equal(y, w, ALL_KEYS).values())
    assert float(m.last_relational['valid_fraction']) > 0


def test_root_height_changes_output_only_when_sensor_valid(env):
    by, _, s, _ = env
    m = active('RelationalInputMegatron', s, 6)
    src, tgt = by['unitree_g1_29dof'], by['unitree_g1_29dof']
    rows, inp, _ = inputs(src, tgt, B=8, mode='full', seed=13)
    rh = torch.as_tensor(src.arrays['root_height_m'][rows]).float()
    with torch.no_grad():
        a = fk(m, src, tgt, inp, source_root_height_m=rh)
        b = fk(m, src, tgt, inp, source_root_height_m=rh + 0.2)
        c = fk(m, src, tgt, inp, source_root_height_m=None)
        d = fk(m, src, tgt, inp, source_root_height_m=torch.full_like(rh, float('nan')))
    assert not torch.equal(a['rot6'], b['rot6'])
    assert torch.equal(c['rot6'], d['rot6'])


def test_roll_mode_deterministic_and_information_free(env):
    by, _, s, _ = env
    m = active('RelationalInputMegatron', s, 7)
    src, tgt = by['unitree_g1_29dof'], by['human_smpl']
    for B in (8, 7):
        rows, inp, _ = inputs(src, tgt, B=B, mode='full', seed=14)
        rh = torch.as_tensor(src.arrays['root_height_m'][rows]).float()
        with torch.no_grad():
            r1 = fk(m, src, tgt, inp, source_root_height_m=rh, roll_relations=True)
            r2 = fk(m, src, tgt, inp, source_root_height_m=rh, roll_relations=True)
            n1 = fk(m, src, tgt, inp, source_root_height_m=rh, roll_relations=False)
        assert torch.equal(r1['rot6'], r2['rot6'])
        assert not torch.equal(r1['rot6'], n1['rot6'])
    # features: row i (roll) == row (i+B//2)%B (no roll) where both rows are fully valid
    g = src.geom(8); st = g['dense_surface']['static']
    rh8 = torch.as_tensor(src.arrays['root_height_m'][np.arange(8)]).float()
    assert bool(torch.isfinite(rh8).all())
    x = src.tensor('pose9', np.arange(8), CPU).float()
    f0, v0 = M.relational_features(x, None, g, st, rh8, roll=False)
    f1, v1 = M.relational_features(x, None, g, st, rh8, roll=True)
    assert torch.equal(v0, v1)
    assert torch.equal(f1[0], f0[4]) and torch.equal(f1[5], f0[1])
    # I_roll arm = constructor default
    ir = build('RelationalInputMegatron', roll_relations=True); copy_shared(m, ir)
    with torch.no_grad():
        a = fk(ir, src, tgt, inp, source_root_height_m=rh)
    assert torch.equal(a['rot6'], r1['rot6'])


def _grad_sums(m, by, seed=15):
    import train_scaling as original
    cfg = cfg_with_order()
    m.train(); m.zero_grad(set_to_none=True)
    src, tgt = by['human_smpl'], by['unitree_g1_29dof']
    rows, inp, target = inputs(src, tgt, B=8, mode='full', seed=seed)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], CPU)
    out = fk(m, src, tgt, inp, source_root_height_m=rh)
    loss, _ = M.make_objective(original.loss_function(cfg))(out, target, tgt, rows, inp['condition'], cfg['loss'])
    loss.backward()
    return {n: (0.0 if p.grad is None else float(p.grad.abs().sum())) for n, p in m.named_parameters()}


def test_new_branches_receive_gradient(env):
    by, _, s, _ = env
    m = build('RelationalInputMegatron'); copy_shared(s, m)
    g0 = _grad_sums(m, by)
    # step-1 gradient reaches the zero-initialised gates (encoder Z_l, decoder attachment output)
    for name in ['group_to_joint.0.weight', 'group_to_joint.5.weight', 'target_dense_attachment.out.weight']:
        assert g0[name] > 0, name
    # behind a zero gate the upstream group stream is exactly gradient-free at init (Flamingo-style)
    assert g0['relational_mlp.2.weight'] == 0 and g0['group_point_mlp.0.weight'] == 0
    # once the gates are non-zero (after the first optimizer step) every new module learns
    with torch.no_grad():
        for z in m.group_to_joint: z.weight.normal_(0, 1e-3)
        m.target_dense_attachment.out.weight.normal_(0, 1e-3)
    g1 = _grad_sums(m, by)
    for name in ['relational_mlp.2.weight', 'group_point_mlp.0.weight', 'group_centre_mlp.0.weight',
                 'group_region_emb.weight', 'group_type', 'target_dense_attachment.q.weight']:
        assert g1[name] > 0, name


def test_alt_grouping_and_join_attention_variants():
    by, _, _ = bodies_with_synthetic_dense(grouping='g128')
    try:
        src, tgt = by['human_smpl'], by['unitree_g1_29dof']
        st = src.geom(2)['dense_surface']['static']
        assert tuple(st['group_members'].shape) == (128, 32)
        sf = build('SFMegatron')
        for kw in ({}, {'dense_group_attention': 'joints_only'}):
            s = build('DenseSurfaceMegatron', **kw)
            _, inp, _ = inputs(src, tgt, B=6, seed=16)
            with torch.no_grad():
                assert all(outputs_equal(fk(sf, src, tgt, inp), fk(s, src, tgt, inp), ALL_KEYS).values())
    finally:
        bodies_with_synthetic_dense()  # restore g256 for other tests


def test_hop_construction_matches_B_convention(env):
    by, _, s, _ = env
    src = by['unitree_g1_29dof']; g = src.geom(1); st = g['dense_surface']['static']
    hop = g['hop'][0]; ga = st['group_anchor']
    # recompute B's surface_model.py:108-113 construction for [J;P] nodes
    J = hop.shape[0]; G = ga.shape[0]
    node = torch.cat([torch.arange(J), ga]); full = hop[node][:, node]
    extra = torch.cat([torch.zeros(J, dtype=torch.long), torch.ones(G, dtype=torch.long)])
    full = full + extra[:, None] + extra[None]; full.fill_diagonal_(0)
    assert torch.equal(full[:J, J:], hop[:, ga] + 1) and torch.equal(full[J:, :J], hop[ga, :] + 1)
    pp = hop[ga][:, ga] + 2; pp.fill_diagonal_(0)
    assert torch.equal(full[J:, J:], pp)


def test_derangement_S_wrong():
    by, _, report = bodies_with_synthetic_dense(derangement='default')
    try:
        assert report['derangement'] and all(k != v for k, v in report['derangement'].items())
        g1 = by['unitree_g1_29dof']; st = g1.geom(1)['dense_surface']['static']
        assert st['deranged'] and st['donor'] != 'unitree_g1_29dof' and st['asset'] is None
        assert int(st['anchor'].max()) < len(g1.parents)
        sf = build('SFMegatron'); s = build('DenseSurfaceMegatron')
        src, tgt = by['human_smpl'], g1
        _, inp, _ = inputs(src, tgt, B=6, seed=17)
        with torch.no_grad():
            assert all(outputs_equal(fk(sf, src, tgt, inp), fk(s, src, tgt, inp), ALL_KEYS).values())
        i = build('RelationalInputMegatron')
        with pytest.raises(ValueError):
            fk(i, src, tgt, inp, source_root_height_m=torch.zeros(6))
    finally:
        bodies_with_synthetic_dense()


def test_parameter_counts_per_arm(env):
    _, sf, s, i = env
    a = build('ConditionalMegatron')
    counts = {n: M.arm_parameter_counts(m) for n, m in [('A', a), ('SF', sf), ('S', s), ('I', i)]}
    assert counts['SF']['total'] - counts['A']['total'] == 128 * 4 + 2 * 128 + 1 - (128 + 1)
    assert counts['S']['total'] > counts['SF']['total'] and counts['I']['total'] > counts['S']['total']
    assert counts['I']['total'] - counts['S']['total'] == counts['I']['added_by_component']['relational_input']


@pytest.mark.parametrize('cls', ['SFMegatron', 'DenseSurfaceMegatron', 'RelationalInputMegatron'])
def test_checkpoint_round_trip_strict(env, cls, tmp_path):
    m = build(cls); perturb(m, 0.01, 30)
    path = tmp_path / 'ckpt.pt'
    torch.save({'model': m.state_dict()}, path)
    m2 = build(cls); m2.load_state_dict(torch.load(path, weights_only=True)['model'], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_bench_runs_small():
    from am2exp_v1.model.bench import run
    rep = run(batch=4, reps=1, threads=4)
    assert set(rep['arms']) >= {'A', 'SF', 'S_g256', 'S_g128', 'I_g256', 'O_g256'}
    assert all(r['saved_activation_bytes'] > 0 for r in rep['arms'].values())
