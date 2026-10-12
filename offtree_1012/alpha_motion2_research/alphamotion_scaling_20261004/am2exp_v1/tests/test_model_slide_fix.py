"""WP-MODEL V0 slide-fix tests (CPU)."""
import math
import pytest
import torch
from ._model_helpers import M, CPU, build, cfg_with_order, loaded_bodies, inputs, fk, outputs_equal, copy_shared

MARGIN = M.slide_fix.SLIDE_MARGIN if hasattr(M, 'slide_fix') else 0.05


def _head(d=128, seed=0):
    torch.manual_seed(seed)
    return M.SlideFixHead(d)


def test_parent_state_dict_bit_equal_A_vs_SF():
    a = build('ConditionalMegatron'); sf = build('SFMegatron')
    sa_, ss = a.state_dict(), sf.state_dict()
    shared = [k for k in sa_ if not k.startswith('slide_head.')]
    assert shared and all(torch.equal(sa_[k], ss[k]) for k in shared)
    assert set(ss) - set(sa_) == {'slide_head.norm_p.weight', 'slide_head.norm_p.bias', 'slide_head.norm_r.weight',
                                   'slide_head.norm_r.bias', 'slide_head.linear.weight', 'slide_head.linear.bias'}


def test_init_bias_and_q():
    h = _head()
    expected = math.log(((0.968 + 0.05) / 1.1) / (1 - (0.968 + 0.05) / 1.1))
    assert abs(float(h.linear.bias.detach()) - expected) < 1e-6
    q = -0.05 + 1.1 * torch.sigmoid(h.linear.bias)
    assert abs(float(q) - 0.968) < 1e-6
    assert abs(float(h.linear.weight.std()) - 0.02) < 0.004


def test_scale_invariance_x100():
    h = _head(); g = torch.Generator().manual_seed(1)
    qp = torch.randn(8, 11, 128, generator=g); qr = torch.randn(8, 11, 128, generator=g)
    with torch.no_grad():
        d = (h(qp, qr) - h(qp * 100, qr * 100)).abs().max()
    assert float(d) < 1e-4


def test_reachable_range_with_bounded_logit():
    m = 0.05
    for target in (0.0, 1.0, 1.0185):
        x = M.slide_fix.margin_logit(target, m)
        assert abs(x) < 4.0, (target, x)
        assert abs((-m + (1 + 2 * m) * torch.sigmoid(torch.tensor(x))).item() - target) < 1e-6
    xs = torch.linspace(-4, 4, 801)
    q = -m + (1 + 2 * m) * torch.sigmoid(xs)
    assert float(q.min()) < 0.0 and float(q.max()) > 1.0185


@pytest.mark.parametrize('target', [0.0, 0.5, 1.0, 1.0185])
def test_nonzero_gradient_at_targets(target):
    h = _head(); g = torch.Generator().manual_seed(2)
    qp = torch.randn(4, 11, 128, generator=g); qr = torch.randn(4, 11, 128, generator=g)
    logits = h(qp, qr)
    sv = torch.zeros(4, 11, dtype=torch.bool); sv[:, 8:10] = True
    q = torch.where(sv, -0.05 + 1.1 * torch.sigmoid(logits), torch.zeros_like(logits))
    out = {'slide_valid': sv, 'slide_normalized': q, 'aux': {'slide_logit': logits}}
    loss, terms = M.slide_fix_loss(out, torch.full((4, 11), target))
    loss.backward()
    assert float(h.linear.weight.grad.abs().sum()) > 0 and float(h.linear.bias.grad.abs()) > 0
    assert torch.isfinite(loss)


def test_gradient_survives_at_saturated_old_regime():
    """Even at logit 8 (old saturation), the margin head still has a usable gradient."""
    x = torch.tensor([8.0], requires_grad=True)
    q = -0.05 + 1.1 * torch.sigmoid(x)
    out = {'slide_valid': torch.ones(1, 1, dtype=torch.bool), 'slide_normalized': q[None], 'aux': {'slide_logit': x[None]}}
    loss, _ = M.slide_fix_loss(out, torch.tensor([[1.0185]]))
    loss.backward()
    assert float(x.grad.abs()) > 1e-4


def test_sf_vs_a_forward_non_slide_outputs_identical():
    bodies, _ = loaded_bodies(); by = {b.name: b for b in bodies}
    a = build('ConditionalMegatron'); sf = build('SFMegatron')
    for s, t in [('human_smpl', 'unitree_g1_29dof'), ('relay_franka_full_v2', 'relay_franka_full_v2')]:
        src, tgt = by[s], by[t]
        _, inp, _ = inputs(src, tgt, B=8, mode='full', seed=3)
        with torch.no_grad():
            oa = fk(a, src, tgt, inp); osf = fk(sf, src, tgt, inp)
        eq = outputs_equal(oa, osf, ('rot6', 'raw_hpos', 'contact_logits'))
        assert all(eq.values()), eq
        if t == 'relay_franka_full_v2':
            assert bool(osf['slide_valid'].any())
            q = osf['slide_normalized'][osf['slide_valid']]
            assert float(q.min()) > -0.05 and float(q.max()) < 1.05
            assert 'slide_logit' in osf['aux']


def test_slide_fix_objective_replaces_l1_and_restores_count():
    import train_scaling as original
    bodies, _ = loaded_bodies(); fr = next(b for b in bodies if b.name == 'relay_franka_full_v2')
    cfg = cfg_with_order()
    base = original.loss_function(cfg)
    obj = M.make_slide_fix_objective(base)
    sf = build('SFMegatron'); sf.train()
    rows, inp, target = inputs(fr, fr, B=8, mode='full', seed=4)
    out = fk(sf, fr, fr, inp)
    loss, terms = obj(out, target, fr, rows, inp['condition'], cfg['loss'])
    assert torch.isfinite(loss)
    assert 'slide_fix_smoothl1' in terms and 'slide_normalized_l1' in terms
    assert int(terms['count_slide_joint_frames']) == 16
    # base objective with slide_valid masked contains no slide term
    l2, t2 = base({**out, 'slide_valid': torch.zeros_like(out['slide_valid'])}, target, fr, rows, inp['condition'], cfg['loss'])
    assert 'slide_normalized_l1' not in t2
    expected = l2 + cfg['loss']['slide_normalized'] * (terms['slide_fix_smoothl1'] + 1e-3 * terms['slide_fix_logit_sq'])
    assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-7)
    loss.backward()
    assert float(sf.slide_head.linear.weight.grad.abs().sum()) > 0


def test_clamped_native_within_limits():
    bodies, _ = loaded_bodies(); fr = next(b for b in bodies if b.name == 'relay_franka_full_v2')
    sf = build('SFMegatron')
    with torch.no_grad():
        sf.slide_head.linear.bias.fill_(6.0)  # q01 ~ 1.047 > 1 -> raw native beyond upper limit
        _, inp, _ = inputs(fr, fr, B=4, seed=5)
        out = fk(sf, fr, fr, inp)
    g = fr.geom(4)
    clamped = M.clamp_slide_native(out, g)
    sv = out['slide_valid']
    up = torch.as_tensor(g['slide_upper'], dtype=torch.float32).expand_as(out['slide_native'])
    assert bool((out['slide_native'][sv] > up[sv]).all())
    assert bool((clamped[sv] <= up[sv] + 1e-6).all())
