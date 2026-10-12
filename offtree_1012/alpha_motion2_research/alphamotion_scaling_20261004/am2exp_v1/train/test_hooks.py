"""TEST-ONLY geometry hook: attach WP-MODEL's synthetic capsule surfaces (no real assets needed).

Used through ``train_arm.py --override`` in CPU dry runs of S/O/I/S_wrong before WP-ASSETS has
written OUT/assets/surface_v2. Never referenced by train/arms.json.
"""
from __future__ import annotations


def attach_synthetic_dense(bodies, device, *, names=None, derangement=None, grouping='g256', require=None):
    from am2exp_v1.model import surface_api as sa
    from am2exp_v1.model.geometry import attach_dense_geometry
    wanted = set(names) if names else {b.name for b in bodies if b.domain == 'xwbc' or b.name in ('adroit_pen', 'relay_franka_full_v2')}
    assets = {b.name: sa.build_synthetic_asset(b.name, b.parents, b.offsets, b.names, device=str(device))
              for b in bodies if b.name in wanted}
    report = attach_dense_geometry(bodies, device, assets=assets, grouping=grouping, derangement=derangement, require=require)
    report['synthetic_test_assets'] = sorted(assets)
    return report


def stub_xbody_loss(model, bodies, batch_rng_state, cfg):
    """TEST-ONLY stand-in honouring the xbody_loss contract (see train/README.md).

    Draws Adroit and Franka training rows from the run's data RNG (same draws whatever w_xb),
    maps A->F and F->A through the model, returns a graph-connected total = w_xb * L.
    """
    import hashlib
    import numpy as np
    import torch
    A = bodies['adroit_pen']
    F = bodies['relay_franka_full_v2']
    B = int(cfg.get('batch_size', 8))
    dev = cfg['device']
    ra = batch_rng_state.choice(A.eligible(True), B)
    rf = batch_rng_state.choice(F.eligible(True), B)
    with cfg['autocast'](dev):
        out_af = model.forward_kinematic(A.tensor('pose9', ra, dev).float(), A.geom(B), F.geom(B))
        out_fa = model.forward_kinematic(F.tensor('pose9', rf, dev).float(), F.geom(B), A.geom(B),
                                         source_slide_normalized=F.tensor('slide_normalized', rf, dev).float())
    loss = out_af['slide_native'].float().abs().mean() + out_fa['joint_position'].float().norm(dim=-1).mean() / A.reach
    total = float(cfg.get('w_xb', 0.0)) * loss
    sha = hashlib.sha256(np.concatenate([ra, rf]).astype(np.int64).tobytes()).hexdigest()
    return {'total': total, 'terms': {'stub_loss': loss.detach()}, 'sampling': {'rows_sha256': sha, 'batch': B}}
