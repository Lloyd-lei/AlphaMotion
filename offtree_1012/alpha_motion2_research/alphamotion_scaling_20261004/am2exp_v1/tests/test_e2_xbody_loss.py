"""WP-E2 cross-body loss tests (CPU, real adroit_pen / relay_franka_full_v2 bodies, A_S checkpoint weights)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from e2 import xbody_loss as xl  # noqa: E402
from e2 import opposition as op  # noqa: E402
from e2._env import ROOT  # noqa: E402

CKPT = ROOT / 'runs/A_S_seed0/checkpoint_step3000.pt'
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason='needs runs/A_S_seed0')


@pytest.fixture(scope='module')
def mb():
    from e2 import bodies as eb
    torch.set_num_threads(4)
    model, cfg, _ = eb.load_model(CKPT)
    model.train()
    return model, eb.load_external_bodies(cfg), cfg


def _rng_state(rng):
    return repr(rng.bit_generator.state)


def test_both_argument_orders_and_result_contract(mb):
    model, bodies, _ = mb
    torch.manual_seed(0)
    r1 = xl.xbody_loss(model, bodies, np.random.default_rng(7), {'w_xb': 0.2, 'batch_size': 8})
    torch.manual_seed(0)
    r2 = xl.xbody_loss(model, bodies, {'w_xb': 0.2, 'batch': 8}, np.random.default_rng(7))
    assert set(r1) == {'total', 'terms', 'sampling'}
    assert r1['total'].requires_grad and r1['total'].dim() == 0
    assert torch.equal(r1['total'].detach(), r2['total'].detach())
    assert r1['sampling'] == r2['sampling'] and r1['sampling']['batch'] == 8
    assert float(r1['terms']['count_xb_rows_A']) == 8 and float(r1['terms']['count_xb_rows_F']) == 8
    with pytest.raises(KeyError):
        xl.xbody_loss(model, bodies, np.random.default_rng(0), {'batch_size': 8})
    with pytest.raises(TypeError):
        xl.xbody_loss(model, bodies, {'w_xb': 0.2}, {'w_xb': 0.2})


def test_trainer_accepts_result(mb):
    """Same parsing as train/train_arm._as_loss_terms + hand_gripper_step requirements."""
    model, bodies, _ = mb
    res = xl.xbody_loss(model, bodies, np.random.default_rng(1), {'w_xb': 0.0, 'batch_size': 8, 'fn': 'x', 'step': 3,
                                                                  'device': torch.device('cpu'), 'autocast': lambda d: torch.autocast('cpu', enabled=False)})
    assert isinstance(res, dict) and 'total' in res
    total = res['total']
    assert torch.is_tensor(total) and total.requires_grad
    for k, v in res['terms'].items():
        float(v)


def test_rng_consumption_independent_of_weight_and_roll(mb):
    model, bodies, _ = mb
    states, samplings = [], []
    for w, roll, dirs in [(0.0, 0, 'both'), (0.2, 0, 'both'), (0.2, 4, 'both'), (0.2, 0, 'AF'), (0.2, 0, 'FA')]:
        rng = np.random.default_rng(11)
        info = {}
        res = xl.xbody_loss(model, bodies, rng, {'w_xb': w, 'roll': roll, 'batch_size': 8, 'directions': dirs, 'info': info})
        states.append(_rng_state(rng))
        samplings.append(res['sampling'])
        assert len(info['rows_A']) == 8 and len(info['rows_F']) == 8
    assert len(set(states)) == 1
    assert all(s == samplings[0] for s in samplings)


def test_rows_are_train_split_only(mb):
    model, bodies, _ = mb
    info = {}
    xl.xbody_loss(model, bodies, np.random.default_rng(3), {'w_xb': 0.0, 'batch_size': 64, 'info': info})
    A, F = bodies['adroit_pen'], bodies['relay_franka_full_v2']
    assert (A.arrays['split'][info['rows_A']] == 0).all()
    assert (F.arrays['split'][info['rows_F']] == 0).all()


def test_zero_weight_control_is_graph_connected_with_zero_grad(mb):
    model, bodies, _ = mb
    model.zero_grad(set_to_none=True)
    res = xl.xbody_loss(model, bodies, np.random.default_rng(2), {'w_xb': 0.0, 'batch_size': 8})
    assert res['total'].requires_grad and float(res['total']) == 0.0
    assert float(res['terms']['xb_total_unweighted']) > 0
    res['total'].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, 'w_xb=0 must still backpropagate (trainer finite_backward needs grads)'
    assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_positive_weight_reaches_slide_head_and_hand_decoder(mb):
    model, bodies, _ = mb
    model.zero_grad(set_to_none=True)
    res = xl.xbody_loss(model, bodies, np.random.default_rng(2), {'w_xb': 0.2, 'batch_size': 16})
    res['total'].backward()
    named = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith('slide_head') and float(g.abs().sum()) > 0 for n, g in named.items())
    assert sum(float(g.abs().sum()) for n, g in named.items() if n.startswith('enc')) > 0


def test_loss_value_matches_definition_and_roll(mb):
    model, bodies, _ = mb
    model.eval()
    A, F = bodies['adroit_pen'], bodies['relay_franka_full_v2']
    try:
        for roll in (0, 5):
            info = {}
            res = xl.xbody_loss(model, bodies, np.random.default_rng(9), {'w_xb': 1.0, 'roll': roll, 'batch_size': 16, 'info': info})
            ra, rf = info['rows_A'], info['rows_F']
            with torch.no_grad():
                xA = A.tensor('pose9', ra, 'cpu').float()
                outF = model.forward_kinematic(xA, A.geom(16), F.geom(16))
                pred = outF['slide_native'][:, 8] + outF['slide_native'][:, 9] - 0.2
                P, G = xl.pose9_PG(xA, A.reach)
                tgt = op.adroit_descriptor(P, G)['c_A'].clamp(0, 7.8)
                tgt = torch.roll(tgt, roll) if roll else tgt
                L_AF = torch.nn.functional.huber_loss(pred, tgt, delta=0.5)
                xF = F.tensor('pose9', rf, 'cpu').float()
                outA = model.forward_kinematic(xF, F.geom(16), A.geom(16), source_slide_normalized=F.tensor('slide_normalized', rf, 'cpu').float())
                d = op.adroit_descriptor(outA['joint_position'], outA['joint_rotation'])
                sn = F.tensor('slide_native', rf, 'cpu').float()
                ftgt = sn[:, 8] + sn[:, 9] - 0.2
                ftgt = torch.roll(ftgt, roll) if roll else ftgt
                L_FA = torch.nn.functional.huber_loss(d['c_A'], ftgt, delta=0.5) + 0.5 * torch.relu(0.5 - d['facing']).mean()
            assert float(res['total']) == pytest.approx(float(L_AF + L_FA), rel=1e-5, abs=1e-6)
            assert float(res['terms']['xb_AF_huber']) == pytest.approx(float(L_AF), rel=1e-5, abs=1e-6)
    finally:
        model.train()


def test_never_conditions_on_franka_fingers(monkeypatch, mb):
    model, bodies, _ = mb
    calls = []
    orig = model.forward_kinematic

    def spy(*a, **kw):
        calls.append(kw)
        assert kw.get('condition') is None and kw.get('source_hidden') is None and kw.get('source_contact') is None
        return orig(*a, **kw)
    monkeypatch.setattr(model, 'forward_kinematic', spy)
    xl.xbody_loss(model, bodies, np.random.default_rng(0), {'w_xb': 0.2, 'batch_size': 4})
    assert len(calls) == 2 and 'source_slide_normalized' in calls[1] and 'source_slide_normalized' not in calls[0]


def test_slide_fix_model_if_available(mb):
    """WP-MODEL's SFMegatron (arms G0/G1): loss reaches the new slide head through the raw native slide."""
    _, bodies, cfg = mb
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from e2._env import sealed
        sealed()
        import importlib
        SF = importlib.import_module('am2exp_v1.model.slide_fix').SFMegatron
    except Exception as exc:  # pragma: no cover - depends on WP-MODEL
        pytest.skip(f'SFMegatron unavailable: {exc}')
    torch.manual_seed(0)
    m = SF(**cfg['model'])
    m.train()
    res = xl.xbody_loss(m, bodies, np.random.default_rng(0), {'w_xb': 0.2, 'batch_size': 8})
    res['total'].backward()
    g = [p.grad for n, p in m.named_parameters() if n.startswith('slide_head') and p.grad is not None]
    assert g and sum(float(x.abs().sum()) for x in g) > 0
