"""End-to-end CPU step through the sealed recipe (backward_training_step) for every WP-MODEL arm.

Exercises: 8 XWBC directions + contrast/judge/cycle (bare encode on dense geometry) + sparse
extra, a Franka unimate step (slide fix), the I-arm root-height sensor wrapper (the pattern
WP-OPS installs: rebinding recipe_training.make_reference_inputs, no numpy RNG consumed), the O
objective, and finite gradients.  Batch 16 (config batch 128 is a GPU setting).
"""
import numpy as np
import pytest
import torch
from ._model_helpers import M, CPU, sa, loaded_bodies, cfg_with_order, synthetic_asset

TRAIN_DENSE = ['booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi', 'human_smpl', 'pnd_adam_lite',
               'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2', 'adroit_pen', 'relay_franka_full_v2']


@pytest.fixture(scope='module')
def setup():
    bodies, audit = loaded_bodies()
    M.detach_dense_geometry(bodies)
    M.attach_dense_geometry(bodies, CPU, assets={n: synthetic_asset(n) for n in TRAIN_DENSE})
    cfg = cfg_with_order(); cfg['batch_size'] = 16
    yield bodies, cfg
    from ._model_helpers import bodies_with_synthetic_dense
    bodies_with_synthetic_dense()


def _install_sensor_wrapper():
    import recipe_training
    original = recipe_training.make_reference_inputs

    def wrapped(src, tgt, rows, rng, device, config):
        inputs, target = original(src, tgt, rows, rng, device, config)
        inputs['source_root_height_m'] = M.source_root_height_sensor(src, rows, inputs['source_contact'], device)
        return inputs, target
    recipe_training.make_reference_inputs = wrapped
    return lambda: setattr(recipe_training, 'make_reference_inputs', original)


@pytest.mark.parametrize('arm', ['SF', 'S', 'O', 'I', 'I_roll', 'O_roll', 'S_wrong'])
def test_full_recipe_step(setup, arm):
    import train_scaling as original
    from recipe_training import backward_training_step
    from train_pilot import autocast
    bodies, cfg = setup
    cls, extra, objflags, geomflags = M.ARMS[arm]
    if geomflags.get('derangement'):
        M.detach_dense_geometry(bodies)
        M.attach_dense_geometry(bodies, CPU, assets={n: synthetic_asset(n) for n in TRAIN_DENSE}, derangement='default')
    restore = _install_sensor_wrapper() if geomflags.get('root_height_sensor') else (lambda: None)
    try:
        M.set_excluded_ground_groups(M.excluded_ground_groups_from_bodies(bodies))
        model = M.build_arm(cfg, CPU, arm); model.train()
        obj = M.make_objective(original.loss_function(cfg), **objflags)
        reports = []
        for domain, seed in [('xwbc', 3), ('unimate', 11)]:
            model.zero_grad(set_to_none=True)
            rep = backward_training_step(model, bodies, cfg, np.random.default_rng(seed), np.random.default_rng(4),
                                         np.random.default_rng(77), device=CPU, domain=domain, forward_fn=original.forward,
                                         objective_fn=obj, autocast_fn=autocast, force_all_losses=True)
            gn = original.finite_backward(model)
            assert np.isfinite(gn) and gn > 0
            reports.append(rep)
        xw = reports[0]
        assert len(xw['directions']) == 9 and xw['auxiliary']['reenc_active']
        if objflags.get('relational'):
            assert any(d['valid_loss_samples'].get('relational_rows', 0) > 0 for d in xw['directions'])
            assert 'relational_pen' in xw['terms']
        if objflags.get('slide_fix'):
            uni = reports[1]['directions'][0]
            if uni['target'] == 'relay_franka_full_v2':
                assert 'slide_fix_smoothl1' in reports[1]['terms']
    finally:
        restore()
        if geomflags.get('derangement'):
            M.detach_dense_geometry(bodies)
            M.attach_dense_geometry(bodies, CPU, assets={n: synthetic_asset(n) for n in TRAIN_DENSE})


def test_sensor_wrapper_consumes_no_rng(setup):
    """The I-arm input wrapper must leave the numpy data stream untouched (sampling_sha256 identity)."""
    import recipe_training
    from pilot_data import sample_rows
    bodies, cfg = setup
    by = {b.name: b for b in bodies}
    src, tgt = by['human_smpl'], by['unitree_g1_29dof']
    outs = []
    for wrap in (False, True):
        restore = _install_sensor_wrapper() if wrap else (lambda: None)
        try:
            rng = np.random.default_rng(5); rows = sample_rows(src, rng, 16)
            inp, _ = recipe_training.make_reference_inputs(src, tgt, rows, rng, CPU, cfg)
            outs.append((inp['source_hidden'].clone(), torch.isfinite(inp['source_contact']).clone(), rng.random()))
        finally:
            restore()
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]) and outs[0][2] == outs[1][2]
