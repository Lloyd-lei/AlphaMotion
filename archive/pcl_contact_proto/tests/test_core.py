"""Synthetic acceptance tests for pcl_contact.core (CPU, < 30 s).
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= /home/arenalabs/miniconda3/envs/ardy/bin/python -m pytest -q -p no:cacheprovider tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcl_contact import core as C  # noqa: E402

FPS = 30.0


def ring(radius, n=400):
    """points on a cylinder (axis z, length 10 cm) surface, centred at origin."""
    th = np.linspace(0, 2 * np.pi, n // 10, endpoint=False)
    z = np.linspace(-5, 5, 10)
    T, Z = np.meshgrid(th, z)
    return np.stack([radius * np.cos(T).ravel(), radius * np.sin(T).ravel(), Z.ravel()], 1), T.ravel()


def roll(T, v_cms, slip_frac, radius=5.0):
    """Cylinder rolling along +x on the floor y=0 with centre speed v and angular speed (1-slip_frac)*v/r."""
    P0, _ = ring(radius)
    t = np.arange(T) / FPS
    X = np.empty((T, len(P0), 3))
    for i, ti in enumerate(t):
        ang = -(1 - slip_frac) * v_cms * ti / radius
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        X[i] = P0 @ R.T + np.array([v_cms * ti, radius, 0.0])
    return X


def one_group(P):
    return {'link:cyl': dict(kind='link', idx=np.arange(P))}


def test_hysteresis_and_despeckle():
    d = np.array([3, 1.5, 0.5, 1.5, 1.9, 2.5, 1.5, 0.2, 3, 0.2, 0.2, 0.2, 0.2, 3.0])[:, None]
    h = C.hysteresis(d, 1.0, 2.0)[:, 0]
    assert h.tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0]
    c = C.despeckle(h[:, None], 3, 2)[:, 0]
    assert c.tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0] or c.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]


def test_rolling_without_slip_is_not_slip():
    X = roll(60, 30.0, 0.0)
    lab = C.ground_labels(X, FPS, one_group(X.shape[1]), floor_cm=0.0)
    c = lab['contact'][:, 0]
    assert c.all()
    sp = lab['slip_speed'][5:-5, 0]
    assert np.nanmedian(sp) < 2.0, np.nanmedian(sp)        # material contact points are ~at rest
    assert lab['slip'][5:-5, 0].mean() < 0.1
    # the centre (a joint-centre label) moves at 30 cm/s: a joint-speed slip rule would flag this as sliding


def test_pure_sliding_is_slip():
    X = roll(60, 20.0, 1.0)                                  # no rotation: block sliding at 20 cm/s
    lab = C.ground_labels(X, FPS, one_group(X.shape[1]), floor_cm=0.0)
    sp = lab['slip_speed'][5:-5, 0]
    assert abs(np.nanmedian(sp) - 20.0) < 0.5
    assert lab['slip'][5:-5, 0].all() and lab['slip_sustained'][5:-5, 0].all()


def test_penetration_and_lift():
    X = roll(60, 0.0, 0.0)
    X[:20, :, 1] -= 2.0                                      # sunk 2 cm
    X[40:, :, 1] += 6.0                                      # lifted 6 cm
    lab = C.ground_labels(X, FPS, one_group(X.shape[1]), floor_cm=0.0)
    assert abs(lab['pen'][:20, 0].max() - 2.0) < 1e-3
    assert lab['contact'][:38, 0].all() and not lab['contact'][41:, 0].any()


def test_floor_offset_and_tilted_plane():
    X = roll(30, 0.0, 0.0)
    X[..., 1] += 2.0
    assert not C.ground_labels(X, FPS, one_group(X.shape[1]), floor_cm=0.0)['contact'].any()
    assert C.ground_labels(X, FPS, one_group(X.shape[1]), floor_cm=2.0)['contact'].all()


def test_object_relative_slip():
    import trimesh
    box = trimesh.creation.box(extents=(20.0, 10.0, 20.0))  # top face at y=+5 in the object frame
    T = 90
    t = np.arange(T) / FPS
    Tobj = np.tile(np.eye(4), (T, 1, 1))
    for i, ti in enumerate(t):                              # object translates at 15 cm/s and yaws at 0.5 rad/s
        a = 0.5 * ti
        Tobj[i, :3, :3] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Tobj[i, :3, 3] = [15 * ti, 0, 0]
    gx, gz = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    hand_obj = np.stack([gx.ravel(), np.full(gx.size, 5.0), gz.ravel()], 1)   # pads lying on the top face
    q = np.repeat(hand_obj[None], T, 0).copy()
    q[30:60, :, 0] += 10.0 * (t[30:60, None] - t[30])       # frames 30-59: slide 10 cm/s across the top (object frame)
    q[30:60, :, 0] -= 5.0
    q[60:, :, 0] += 10.0 * (t[59] - t[30]) - 5.0
    q[60:, :, 1] += 5.0                                      # frames 60+: lifted 5 cm
    X = np.einsum('tij,tpj->tpi', Tobj[:, :3, :3], q) + Tobj[:, None, :3, 3]
    lab = C.object_labels(X, FPS, {'link:hand': dict(kind='link', idx=np.arange(len(hand_obj)))}, box, Tobj)
    c = lab['contact'][:, 0]; sp = lab['slip_speed'][:, 0]
    assert c[:58].all() and not c[62:].any()
    assert np.nanmax(sp[5:27]) < 1.0                         # carried by the moving, rotating object: no relative slip
    assert abs(np.nanmedian(sp[34:56]) - 10.0) < 0.5        # relative slide measured in the object frame
    assert lab['slip_sustained'][34:56, 0].all() and not lab['slip'][5:27, 0].any()
