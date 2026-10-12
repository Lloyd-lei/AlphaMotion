"""WP-MODEL relational features (arm I), relational objective (arm O) and ground hygiene tests."""
import numpy as np
import pytest
import torch
from ._model_helpers import (M, CPU, sa, build, bodies_with_synthetic_dense, loaded_bodies, inputs, fk, cfg_with_order,
                             real_assets_ready)
from am2exp_v1.model._sealed import rot6d_to_matrix

FALL_GROUPS = [13, 14, 15, 16, 17, 18, 54, 55]  # LAFAN fallAndGetUp*/pushAndFall* (data_manifest_v2 group order)


@pytest.fixture(scope='module')
def by():
    b, _, _ = bodies_with_synthetic_dense()
    return b


def _unslog(v):
    return torch.sign(v) * torch.expm1(v.abs())


def _posed(st, src, rows):
    x = src.tensor('pose9', rows, CPU).float()
    R = rot6d_to_matrix(x[..., :6]); pos = x[..., 6:] * float(src.reach)
    return x, R, pos


def test_features_match_brute_force(by):
    src = by['unitree_g1_29dof']; rows = np.arange(0, 600, 75)
    g = src.geom(len(rows)); st = g['dense_surface']['static']
    x, R, pos = _posed(st, src, rows)
    rh = torch.as_tensor(src.arrays['root_height_m'][rows]).float()
    feat, valid = M.relational_features(x, None, g, st, rh)
    assert feat.shape == (len(rows), st['group_members'].shape[0], 28) and bool(valid.all())
    pts, nrm = sa.pose_surface(st['asset'], R, pos)
    h = pts[..., 1] + 100 * rh[:, None] - st['beta_vis_cm']
    allowed = st['allowed']; anchor = st['anchor']; sub = st['sub']
    for gi in [0, 17, 100, 255]:
        mem = st['group_members'][gi]
        assert torch.allclose(_unslog(feat[:, gi, 0]), h[:, mem].min(1).values, atol=1e-3)
        assert torch.allclose(feat[:, gi, 1] * 50, h[:, mem].mean(1), atol=1e-3)
        ok = allowed[anchor[mem]][:, anchor[sub]]  # [K,S]
        d = torch.cdist(pts[:, mem], pts[:, sub])
        d = torch.where(ok[None], d, torch.full_like(d, 1e4)).flatten(1).min(1).values
        dfeat = torch.log1p(d.clamp(max=100) / 2) / np.log(26)
        has = d < 5e3
        assert torch.allclose(feat[has, gi, 4], dfeat[has], atol=1e-4)
        assert torch.allclose(feat[:, gi, 5:8].norm(dim=-1)[has], torch.ones(int(has.sum())), atol=1e-4)
        assert torch.equal(feat[:, gi, 27] > 0, has)


def test_hidden_anchor_and_missing_sensor_masking(by):
    src = by['human_smpl']; rows = np.arange(8)
    g = src.geom(8); st = g['dense_surface']['static']
    x, _, _ = _posed(st, src, rows)
    hidden = torch.zeros(8, len(src.parents), dtype=torch.bool); hidden[:, [4, 7, 10]] = True  # L_Knee, L_Ankle, L_Toe
    xm = torch.where(hidden[..., None], torch.zeros_like(x), x)
    feat, valid = M.relational_features(xm, hidden, g, st, None)
    ga = st['group_anchor']
    hid_groups = hidden[0, ga]
    assert not bool(valid[:, hid_groups].any()) and bool((feat[:, hid_groups] == 0).all())
    assert bool((feat[..., :4] == 0).all()) and bool((feat[..., 26] == 0).all())  # no sensor -> no ground block
    assert bool(feat[:, ~hid_groups, 27].any())  # self block still active on visible groups
    # partners never sit on hidden points: re-run with hidden joints moved far away gives the same features
    xg = torch.where(hidden[..., None], torch.randn_like(x) * 30, x)
    feat2, _ = M.relational_features(xg, hidden, g, st, None)
    assert torch.equal(feat, feat2)


def test_sensor_helper_rules():
    bodies, _ = loaded_bodies(); by_ = {b.name: b for b in bodies}
    ex = M.excluded_ground_groups_from_bodies(bodies)
    assert len(ex) == 21  # 14 DFaust + 7 PosePrior human trust=0
    M.set_excluded_ground_groups(ex)
    assert M.excluded_ground_groups() == ex
    import am2exp_v1.model.ground as ground
    grp, trust = ground._load_human_trust()
    assert ground.excluded_ground_groups((grp, trust)) == ex
    hum = by_['human_smpl']
    dfaust = np.flatnonzero(np.isin(hum.arrays['group'], sorted(ex)))[:4]
    normal = np.flatnonzero(hum.arrays['ground_valid'] & ~np.isin(hum.arrays['group'], sorted(ex)))[:4]
    rows = np.concatenate([dfaust, normal])
    con = torch.zeros(len(rows), len(hum.parents))
    rh = M.source_root_height_sensor(hum, rows, con, CPU)
    assert bool(torch.isnan(rh[:4]).all()) and bool(torch.isfinite(rh[4:]).all())
    assert torch.allclose(rh[4:], torch.as_tensor(hum.arrays['root_height_m'][normal]).float())
    con_nan = con.clone(); con_nan[4:] = float('nan')
    assert bool(torch.isnan(M.source_root_height_sensor(hum, rows, con_nan, CPU)).all())
    assert bool(torch.isnan(M.source_root_height_sensor(hum, rows, None, CPU)).all())
    fr = by_['relay_franka_full_v2']
    assert bool(torch.isnan(M.source_root_height_sensor(fr, np.arange(4), torch.zeros(4, 11), CPU)).all())


def _gt_outputs(tgt, rows, target, logit_scale=0.0):
    x = target.float()
    R = rot6d_to_matrix(x[..., :6]); pos = x[..., 6:] * float(tgt.reach)
    J = len(tgt.parents)
    return {'rot6': x[..., :6].clone().requires_grad_(True), 'joint_rotation': R, 'joint_position': pos.clone().requires_grad_(True),
            'contact_logits': torch.zeros(len(rows), J, requires_grad=True)}


def test_relational_objective_on_ground_truth_and_perturbed(by):
    tgt = by['unitree_g1_29dof']; src = by['human_smpl']
    rows, inp, target = inputs(src, tgt, B=16, seed=21)
    w = cfg_with_order()['loss']
    out = _gt_outputs(tgt, rows, target)
    r = M.relational_objective(out, target, tgt, rows, w)
    assert float(r['relational_pen'].detach()) == 0.0
    assert float(r['relational_clear'].detach()) < 0.05
    assert int(r['count_relational_rows']) == int(M.row_ground_ok(tgt, rows, CPU).sum()) > 0
    r['total'].backward()
    assert float(out['contact_logits'].grad.abs().sum()) > 0
    sunk = dict(out); sunk['joint_position'] = out['joint_position'].detach() - torch.tensor([0., 5., 0.])
    sunk['joint_position'].requires_grad_(True)
    r2 = M.relational_objective(sunk, target, tgt, rows, w)
    assert float(r2['relational_pen'].detach()) > 0.01 and float(r2['relational_clear'].detach()) > float(r['relational_clear'].detach())
    r2['total'].backward()
    assert float(sunk['joint_position'].grad[:, :, 1].sum()) < 0  # gradient pushes points up


def test_relational_objective_masks_and_roll(by):
    tgt = by['unitree_g1_29dof']; w = cfg_with_order()['loss']
    ex = M.excluded_ground_groups_from_bodies(loaded_bodies()[0])
    bad = np.flatnonzero(np.isin(tgt.arrays['group'], sorted(ex)))[:8]
    target = tgt.tensor('pose9', bad, CPU).float()
    r = M.relational_objective(_gt_outputs(tgt, bad, target), target, tgt, bad, w)
    assert int(r['count_relational_rows']) == 0 and float(r['total']) == 0.0
    good = np.flatnonzero(tgt.arrays['ground_valid'] & ~np.isin(tgt.arrays['group'], sorted(ex)))[::997][:9]
    rows = np.concatenate([good, bad[:3]])
    target = tgt.tensor('pose9', rows, CPU).float()
    out = _gt_outputs(tgt, rows, target)
    r0 = M.relational_objective(out, target, tgt, rows, w)
    r1 = M.relational_objective(out, target, tgt, rows, w, roll=True)
    r1b = M.relational_objective(out, target, tgt, rows, w, roll=True)
    assert int(r0['count_relational_rows']) == 9
    ok = np.ones(12, bool); ok[9:] = False
    assert int(r1['count_relational_rows']) == int((ok & np.roll(ok, -6)).sum())
    assert float(r1['total']) == float(r1b['total'])
    # rolled targets are another frame's: GT prediction no longer has zero penetration-beyond-GT in general
    assert float(r1['relational_pen']) >= 0.0
    fr = by['relay_franka_full_v2']
    rr = np.arange(4); t = fr.tensor('pose9', rr, CPU).float()
    o = {'rot6': t[..., :6], 'joint_rotation': rot6d_to_matrix(t[..., :6]), 'joint_position': t[..., 6:] * fr.reach,
         'contact_logits': torch.zeros(4, 11)}
    assert float(M.relational_objective(o, t, fr, rr, w)['total']) == 0.0


HUMAN_FEET = ([7, 10], [8, 11])  # (L_Ankle, L_Toe), (R_Ankle, R_Toe)


@pytest.mark.parametrize('body', ['unitree_g1_29dof', 'human_smpl'])
def test_real_asset_standing_soles_and_fall_torso(body):
    """Acceptance (final_plan WP-MODEL): standing sole groups min h in [-1,3] cm; fall-frame torso groups < 5 cm."""
    if not real_assets_ready(body):
        pytest.skip('real WP-ASSETS surface_v2 not available yet for ' + body)
    bodies, _ = loaded_bodies()
    M.detach_dense_geometry(bodies)
    M.attach_dense_geometry(bodies, CPU)
    try:
        b = next(x for x in bodies if x.name == body)
        a = b.arrays; ex = M.excluded_ground_groups_from_bodies(bodies)
        ok = a['ground_valid'] & a['root_height_valid'] & a['contact_trust'] & ~np.isin(a['group'], sorted(ex))
        if body == 'human_smpl':
            both = a['contact_hard'][:, HUMAN_FEET[0]].any(1) & a['contact_hard'][:, HUMAN_FEET[1]].any(1)
            tall = a['root_height_m'] > 0.8
        else:
            both = a['contact_hard'][:, b.metadata['ground_static']['sole_j']].all(1)
            tall = a['root_height_m'] > 0.6
        stand = np.flatnonzero(ok & both & tall)[::7][:128]
        assert len(stand) >= 16
        g = b.geom(len(stand)); st = g['dense_surface']['static']
        x = b.tensor('pose9', stand, CPU).float(); rh = torch.as_tensor(a['root_height_m'][stand]).float()
        feat, _ = M.relational_features(x, None, g, st, rh)
        hmin = _unslog(feat[..., 0])
        per_foot = torch.stack([torch.where(st['group_region'] == r, hmin, torch.full_like(hmin, 1e4)).amin(1) for r in (13, 14)], 1)
        p50 = float(per_foot.median())
        print(body, 'standing per-foot lowest group min-h p50 cm', p50, 'p10/p90', float(per_foot.quantile(.1)), float(per_foot.quantile(.9)))
        assert -1.0 <= p50 <= 3.0, p50
        fall = np.flatnonzero(ok & np.isin(a['group'], FALL_GROUPS) & (a['root_height_m'] < (0.3 if body != 'human_smpl' else 0.35)))
        assert len(fall) > 0
        g = b.geom(len(fall)); x = b.tensor('pose9', fall, CPU).float(); rh = torch.as_tensor(a['root_height_m'][fall]).float()
        feat, _ = M.relational_features(x, None, g, st, rh)
        torso = torch.isin(st['group_region'], torch.tensor([0, 1]))
        tmin = _unslog(feat[..., 0])[:, torso].amin(1)
        print(body, 'fall torso min-h median cm', float(tmin.median()), 'n', len(fall))
        assert float(tmin.median()) < 5.0, float(tmin.median())
    finally:
        bodies_with_synthetic_dense()
