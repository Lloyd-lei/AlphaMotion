"""Shared helpers for WP-MODEL tests (CPU).  Bodies are loaded once per pytest session."""
from __future__ import annotations
import functools
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import am2exp_v1.model as M  # noqa: E402
from am2exp_v1.model import surface_api as sa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CPU = torch.device('cpu')
REAL_ASSET_DIR = sa.ASSET_DIR
DENSE_TEST_BODIES = ('unitree_g1_29dof', 'human_smpl', 'booster_t1', 'relay_franka_full_v2', 'adroit_pen', 'fourier_gr3')


def config():
    cfg = json.loads((ROOT / 'orchestration/configs/A_S_seed0.json').read_text())
    return cfg


@functools.lru_cache(maxsize=1)
def loaded_bodies():
    import train_scaling as original
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cfg = config()
    bodies, audit = original.load_scaling_bodies(ROOT, cfg, CPU)
    return bodies, audit


def bodies_with_synthetic_dense(names=DENSE_TEST_BODIES, grouping='g256', derangement=None, points=4096):
    bodies, audit = loaded_bodies()
    by = {b.name: b for b in bodies}
    assets = {n: synthetic_asset(n, points) for n in names}
    M.detach_dense_geometry(bodies)
    report = M.attach_dense_geometry(bodies, CPU, assets=assets, grouping=grouping, derangement=derangement)
    return by, audit, report


@functools.lru_cache(maxsize=None)
def synthetic_asset(name, points=4096):
    bodies, _ = loaded_bodies()
    b = next(x for x in bodies if x.name == name)
    return sa.build_synthetic_asset(name, b.parents, b.offsets, b.names, points=points)


def cfg_with_order():
    cfg = config()
    _, audit = loaded_bodies()
    cfg['judge_body_order'] = audit['judge_body_order']
    return cfg


def build(cls, **extra):
    m = M.build_model(cfg_with_order(), CPU, cls, extra)
    m.eval()
    return m


def copy_shared(src, dst):
    """Load src's parameters into dst for every shared key (dst's extra keys untouched)."""
    sd = dst.state_dict()
    sd.update({k: v for k, v in src.state_dict().items() if k in sd and sd[k].shape == v.shape})
    dst.load_state_dict(sd)


def perturb(module, scale=0.05, seed=123):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)


def inputs(src, tgt, B=16, mode='full', seed=0, condition_count=None, contact=True):
    from pilot_data import make_inputs, sample_rows
    rng = np.random.default_rng(seed)
    rows = sample_rows(src, rng, B)
    inp, target = make_inputs(src, tgt, rows, rng, CPU, mode, condition_count, contact_dropout=0.0 if contact else 1.0)
    return rows, inp, target


def fk(model, src, tgt, inp, **kw):
    return model.forward_kinematic(**inp, source_geom=src.geom(len(inp['source'])), target_geom=tgt.geom(len(inp['source'])), **kw)


def outputs_equal(a, b, keys=('rot6', 'raw_hpos', 'contact_logits', 'slide_normalized', 'slide_native', 'joint_position', 'joint_rotation')):
    return {k: bool(torch.equal(a[k], b[k])) for k in keys if a.get(k) is not None}


def real_assets_ready(*names):
    return sa.real_module() is not None and all(sa.real_asset_available(n) for n in names)
