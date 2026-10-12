"""Tiny CUDA checks (B<=16, < 2 min): bf16-autocast bit identity at init + small-batch timing ratio.

Takes ROOT/gpu_queue.lock with LOCK_EX|LOCK_NB for the duration of the module and releases it;
skips when CUDA is absent/hidden or the lock is held by someone else.  Never trains.
"""
import fcntl
import json
import time
import numpy as np
import pytest
import torch
from ._model_helpers import M, ROOT, bodies_with_synthetic_dense, cfg_with_order, copy_shared, synthetic_asset

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not visible (CPU-only run)')


@pytest.fixture(scope='module')
def gpu():
    f = (ROOT / 'gpu_queue.lock').open('a+')  # existing file; never recreated
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close(); pytest.skip('gpu_queue.lock held by another process')
    try:
        yield torch.device('cuda')
    finally:
        fcntl.flock(f, fcntl.LOCK_UN); f.close(); torch.cuda.empty_cache()


def _setup(dev):
    import train_scaling as original
    from pilot_data import make_inputs, sample_rows
    by, _, _ = bodies_with_synthetic_dense()
    # re-home the two bodies we need on the GPU (static tensors) without touching the CPU originals
    import copy
    out = {}
    for n in ('human_smpl', 'unitree_g1_29dof'):
        b = copy.copy(by[n]); b.static = {k: v.to(dev) for k, v in by[n].static.items()}
        from scaling_data import attach_geometry
        attach_geometry(b, None)
        if hasattr(b, '_geom_without_dense'): del b._geom_without_dense
        M.attach_dense_geometry([b], dev, assets={n: synthetic_asset(n).to(dev)})
        out[n] = b
    return out


def _models(dev):
    cfg = cfg_with_order()
    sf = M.build_model(cfg, dev, 'SFMegatron').eval()
    s = M.build_model(cfg, dev, 'DenseSurfaceMegatron').eval()
    i = M.build_model(cfg, dev, 'RelationalInputMegatron').eval()
    return sf, s, i


def test_bf16_autocast_bit_identity_at_init(gpu):
    from pilot_data import make_inputs, sample_rows
    from train_pilot import autocast
    by = _setup(gpu); sf, s, i = _models(gpu)
    src, tgt = by['human_smpl'], by['unitree_g1_29dof']
    rng = np.random.default_rng(0); rows = sample_rows(src, rng, 8)
    inp, _ = make_inputs(src, tgt, rows, rng, gpu, 'masked', contact_dropout=0.0)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], gpu)
    with torch.no_grad(), autocast(gpu):
        a = sf.forward_kinematic(**inp, source_geom=src.geom(8), target_geom=tgt.geom(8))
        b = s.forward_kinematic(**inp, source_geom=src.geom(8), target_geom=tgt.geom(8))
        c = i.forward_kinematic(**inp, source_geom=src.geom(8), target_geom=tgt.geom(8), source_root_height_m=rh)
    for k in ('rot6', 'raw_hpos', 'contact_logits', 'joint_position'):
        assert torch.equal(a[k], b[k]) and torch.equal(b[k], c[k]), k


def test_small_batch_gpu_timing_ratio(gpu):
    """Report-only: fwd+bwd of one human->G1 direction, B=16, bf16 autocast."""
    import train_scaling as original
    from pilot_data import make_inputs, sample_rows
    from train_pilot import autocast
    by = _setup(gpu); sf, s, i = _models(gpu)
    cfg = cfg_with_order(); base = original.loss_function(cfg)
    src, tgt = by['human_smpl'], by['unitree_g1_29dof']
    rng = np.random.default_rng(1); rows = sample_rows(src, rng, 16)
    inp, target = make_inputs(src, tgt, rows, rng, gpu, 'full', contact_dropout=0.0)
    rh = M.source_root_height_sensor(src, rows, inp['source_contact'], gpu)
    res = {}
    for name, m, kw in [('SF', sf, {}), ('S', s, {}), ('I', i, {'source_root_height_m': rh})]:
        m.train(); obj = M.make_objective(base)
        ts = []
        for r in range(6):
            m.zero_grad(set_to_none=True); torch.cuda.synchronize(); t0 = time.perf_counter()
            with autocast(gpu):
                out = m.forward_kinematic(**inp, source_geom=src.geom(16), target_geom=tgt.geom(16), **kw)
            loss, _ = obj(out, target, tgt, rows, inp['condition'], cfg['loss'])
            loss.backward(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        res[name] = float(np.median(ts[1:]))
    res['S_over_SF'] = res['S'] / res['SF']; res['I_over_SF'] = res['I'] / res['SF']
    out = M._sealed.OUT / 'model'; out.mkdir(parents=True, exist_ok=True)
    (out / 'gpu_timing_B16.json').write_text(json.dumps(res, indent=2))
    assert all(np.isfinite(v) for v in res.values())
