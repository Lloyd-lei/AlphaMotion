"""WP-INTEGRATION: cross-package contract tests on the REAL Stage-0 artifacts (CPU).

* WP-ASSETS per-point trust (data residual p95 > 3 cm) reaches WP-MODEL: arm O losses and arm I features are exactly
  invariant to where the untrusted points are (kuavo_s45 / unitree_h1_2 housings), and do change without the mask.
* WP-RENDER uses the dense human surface but not its sinking hand ellipsoid.
* The untrained S checkpoint written by integration/smoke_e2e loads through eval/run_e1_eval's am2exp registry path.
* Smoke outputs (if present): every dry-run arm completed with matching sampling hashes.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ._model_helpers import M, CPU, sa, loaded_bodies
from am2exp_v1.model._sealed import rot6d_to_matrix
from am2exp_v1.model import geometry as GEO
from am2exp_v1.model import losses_relational as LR

OUT = Path('/var/tmp/alphamotion/am2exp')
ASSETS = OUT / 'assets' / 'surface_v2'
need_assets = pytest.mark.skipif(not (ASSETS / 'unitree_h1_2.npz').exists(), reason='real dense assets not built')


@need_assets
def test_trust_mask_counts_on_real_assets():
    want = {'kuavo_s45': 457, 'unitree_h1_2': 219, 'unitree_g1_29dof': 0, 'adroit_pen': 0, 'human_smpl': 0}
    for body, n_bad in want.items():
        a = sa.load_surface(body, CPU)
        m = GEO.point_trust_mask(a, a.points_local.shape[0], CPU)
        assert int((~m).sum()) == n_bad, body


def _moved_untrusted(asset, by_cm=50.0):
    bad = ~GEO.point_trust_mask(asset, asset.points_local.shape[0], CPU)
    pl = asset.points_local.clone(); pl[bad, 1] += by_cm
    return dataclasses.replace(asset, points_local=pl), bad


@pytest.fixture(scope='module')
def h12():
    bodies, _ = loaded_bodies()
    by = {b.name: b for b in bodies}
    yield by['unitree_h1_2'], bodies
    M.detach_dense_geometry(bodies)


def _attach(bodies, asset):
    M.detach_dense_geometry(bodies)
    M.attach_dense_geometry(bodies, CPU, assets={'unitree_h1_2': asset})


def _rows(body):
    from am2exp_v1.model.ground import row_ground_ok
    rows = np.arange(0, len(body.arrays['root_height_m']), 97)[:48]
    ok = row_ground_ok(body, rows, CPU, None)
    assert bool(ok.any())
    return rows


def _objective(body, rows, seed=0):
    x = body.tensor('pose9', rows, CPU).float()
    R = rot6d_to_matrix(x[..., :6]); pos = x[..., 6:] * float(body.reach)
    g = torch.Generator().manual_seed(seed)
    pos = pos + torch.randn(pos.shape, generator=g) * 3.0
    pos[:, :, 1] -= 4.0  # push the prediction into the floor so L_pen is active
    out = {'rot6': x[..., :6], 'joint_rotation': R, 'joint_position': pos,
           'contact_logits': torch.randn(len(rows), len(body.parents), generator=g)}
    r = LR.relational_objective(out, x, body, rows, {})
    return {k: float(v) for k, v in r.items()}


@need_assets
def test_relational_objective_ignores_untrusted_points(h12):
    body, bodies = h12
    real = sa.load_surface('unitree_h1_2', CPU)
    moved, bad = _moved_untrusted(real)
    rows = _rows(body)
    _attach(bodies, real); a = _objective(body, rows)
    _attach(bodies, moved); b = _objective(body, rows)
    assert a['relational_pen'] > 0
    assert a['count_relational_points'] == a['count_relational_rows'] * int((~bad).sum())
    for k in ('relational_pen', 'relational_clear', 'relational_hcon', 'total'):
        assert a[k] == b[k], k
    # without the mask the moved points DO change the objective (the mask is what makes it invariant)
    body._dense_surface_static['point_trusted'] = torch.ones_like(body._dense_surface_static['point_trusted'])
    c = _objective(body, rows)
    assert c['total'] != b['total']


@need_assets
def test_relational_features_ignore_untrusted_points(h12):
    body, bodies = h12
    real = sa.load_surface('unitree_h1_2', CPU)
    moved, _ = _moved_untrusted(real)
    rows = _rows(body)[:16]
    x = body.tensor('pose9', rows, CPU).float()
    rh = torch.as_tensor(body.arrays['root_height_m'][rows]).float()
    feats = []
    for asset in (real, moved):
        _attach(bodies, asset)
        g = body.geom(len(rows)); st = g['dense_surface']['static']
        f, v = M.relational_features(x, None, g, st, rh)
        feats.append((f, v))
    assert torch.equal(feats[0][0], feats[1][0]) and torch.equal(feats[0][1], feats[1][1])


@need_assets
def test_render_human_hands_use_wrist_centre():
    from am2exp_v1.render import contact as C
    from am2exp_v1.render.regions import REGIONS
    names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine', 'L_Knee', 'R_Knee', 'Chest', 'L_Ankle', 'R_Ankle', 'Torso', 'L_Toe',
             'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
             'L_Wrist', 'R_Wrist']
    T, J = 3, len(names)
    P = np.zeros((T, J, 3)); P[:, :, 1] = 50.0
    P[:, names.index('L_Wrist'), 1] = 10.0
    reg = np.full(8, REGIONS.index('trunk')); reg[:4] = REGIONS.index('L_hand')
    pts = np.zeros((T, 8, 3)); pts[:, :4, 1] = -20.0; pts[:, 4:, 1] = 30.0  # sunk hand ellipsoid
    win = type('W', (), {'names': names, 'P': P, 'parents': None, 'contact_hard': None, 'contact_gap_m': None})()
    rh, zmin, st, used = C.human_region_heights(win, np.zeros(T), dense=(pts, reg))
    assert np.allclose(rh[:, REGIONS.index('L_hand')], 10.0 - 3.0)
    assert np.allclose(zmin, 7.0) and 'wrist' in used


def test_untrained_s_checkpoint_loads_through_e1_registry():
    ck = OUT / 'integration' / 'ckpt' / 'S_untrained' / 'checkpoint_step0000.pt'
    if not ck.exists():
        pytest.skip('run integration/smoke_e2e.py untrained first')
    from am2exp_v1.eval.run_e1_eval import build_model
    model, cfg, info = build_model(ck, torch.device('cpu'))
    assert info['variant'] == 'am2exp' and info['model_class'].endswith('DenseSurfaceMegatron')
    assert info['geometry']['hook'].endswith('attach_dense_geometry') and info['step'] == 0


def test_smoke_dryruns_matched_reference():
    p = OUT / 'integration' / 'dryrun_summary.json'
    if not p.exists():
        pytest.skip('run integration/smoke_e2e.py dryrun first')
    d = json.loads(p.read_text())
    for arm in ('SF', 'S', 'O', 'I', 'S_wrong'):
        r = d[arm]
        assert r['status'] == 'completed', arm
        assert r['sampling']['mismatches'] == 0 and r['sampling']['all_checked_matched'], arm
        assert r['independent_hash_check']['all_ok'], arm
