"""Load XWBC trajectories (cache_lafan_v3 / AMASS caches via data/trajectory_reader.py, read-only) and place the
surface_v2 dense point cloud in the WORLD frame.

World frame (verified on G1 + human, LAFAN): trajectory_reader returns world_root [T,3] cm, Y-up, and
world_root_y == 100*root_height_m + h0_legacy (G1 +1.25 cm, human +1.44 cm, constant to +-0.03 cm), i.e. the source
capture floor is world y = 0 and the legacy contact floor h0 sits 1.25/1.44 cm above it.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import torch

ROOT = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004'
MANIFEST = ROOT + '/data/data_manifest_v2.json'


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


TR = _load('pclc_trajectory_reader', ROOT + '/data/trajectory_reader.py')
S = _load('pclc_surface_v2', ROOT + '/am2exp_v1/assets/surface_v2.py')


def load_traj(group_id, body, window=None):
    d = TR.load_xwbc_trajectory(MANIFEST, group_id, body, with_contact=True)
    if window is not None:
        s0, s1 = window
        T = len(d['frame_index'])
        d = {k: (v[s0:s1] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v) for k, v in d.items()}
    return d


# ---------------------------------------------------------------------------------------------------- human hand fix
def hand_dirs_from_rotation(d, asset_names, side):
    """LAFAN-dialect human_smpl: bones point along the parent's local +Z, so the hand direction is
    R_wrist @ unit(rest_offset[wrist]) (rest offset of the wrist = elbow->wrist = (0,0,25.2) cm).
    Verified: R_elbow @ unit(off_wrist) reproduces the measured forearm to 0.02 deg p99."""
    j = asset_names.index(f'{side}_Wrist')
    off = d['rest_offsets'][j]
    return np.einsum('tij,j->ti', d['global_rotation_matrices'][:, j], off / np.linalg.norm(off)), j


def pose_human_handfix(asset, rot, pos, hand_u, hand_v=None):
    """bone_revolution posing, but points on the two leaf wrist bones use the measured hand axis u (from the
    wrist ROTATION) instead of the forearm direction.  v is the old position-based v re-orthogonalised to the new u.
    hand_u: dict joint_index -> [B,3] unit."""
    p, n = S.pose_surface(asset, rot, pos)
    u, v, w, L = S.bone_frames(asset, pos.to(asset.points_local.dtype))
    for j, uh in hand_u.items():
        uh = torch.as_tensor(uh, dtype=u.dtype)
        uh = uh / uh.norm(dim=-1, keepdim=True)
        vref = v[:, j] if hand_v is None else torch.as_tensor(hand_v[j], dtype=u.dtype)
        vh = vref - (vref * uh).sum(-1, keepdim=True) * uh
        bad = vh.norm(dim=-1, keepdim=True) < 1e-3
        vh = torch.where(bad, w[:, j] - (w[:, j] * uh).sum(-1, keepdim=True) * uh, vh)
        vh = vh / vh.norm(dim=-1, keepdim=True)
        wh = torch.cross(uh, vh, dim=-1)
        idx = torch.nonzero(asset.bone_parent == j).flatten()
        idx = idx[asset.bone_child[idx] < 0]
        th = asset.theta[idx]; rad = asset.radial[idx]
        rd = torch.cos(th)[None, :, None] * vh[:, None] + torch.sin(th)[None, :, None] * wh[:, None]
        p[:, idx] = pos[:, j][:, None] + (asset.t[idx] * asset.leaf_len[idx])[None, :, None] * uh[:, None] + rad[None, :, None] * rd
        nu = asset.normal_uvw[idx]
        n[:, idx] = nu[None, :, 0:1] * uh[:, None] + nu[None, :, 1:2] * vh[:, None] + nu[None, :, 2:3] * wh[:, None]
    return p, n


# --------------------------------------------------------------------------------------------------- world placement
HAND_CALIB = __import__('pathlib').Path(__file__).with_name('hand_frame_calib.json')


def world_points(d, body, hand_fix='rot_axis', chunk=512):
    """-> dict(X [T,P,3] world cm float32, N normals, asset, names).
    human hand_fix: False/None = original asset (hand ellipsoid along the forearm);
                    'rot_axis'  = hand axis u from the wrist rotation, twist v from the position rule;
                    'rot_frame' = full hand frame from the wrist rotation (u and v calibrated once in the wrist-local
                                  frame on lafan::walk1_subject5, hand_frame_calib.json)."""
    asset = S.load_surface(body)
    names = [str(x) for x in d['joint_names']]
    R = torch.as_tensor(d['global_rotation_matrices'], dtype=torch.float32)
    P = torch.as_tensor(d['position_native'], dtype=torch.float32)
    hu, hv = {}, None
    if asset.binding == 'bone_revolution' and hand_fix:
        import json
        cal = json.loads(HAND_CALIB.read_text()) if hand_fix == 'rot_frame' else None
        hv = {} if cal else None
        for side in 'LR':
            u, j = hand_dirs_from_rotation(d, names, side)
            hu[j] = u
            if cal:
                hv[j] = np.einsum('tij,j->ti', d['global_rotation_matrices'][:, j], np.asarray(cal[side]['v_local']))
    Xs, Ns = [], []
    with torch.no_grad():
        for i in range(0, len(R), chunk):
            sl = slice(i, i + chunk)
            if hu:
                p, n = pose_human_handfix(asset, R[sl], P[sl], {j: u[sl] for j, u in hu.items()},
                                          None if hv is None else {j: v[sl] for j, v in hv.items()})
            else:
                p, n = S.pose_surface(asset, R[sl], P[sl])
            Xs.append(p.numpy()); Ns.append(n.numpy())
    X = np.concatenate(Xs) + d['world_root'][:, None, :].astype(np.float32)
    return dict(X=X.astype(np.float32), N=np.concatenate(Ns), asset=asset, names=names)


def point_mask(asset, max_residual_cm=3.0):
    """Mask unreliable approximate-binding points (assets README deviation 3)."""
    r = asset.extra.get('point_data_residual_p95_cm')
    if r is None:
        return np.ones(asset.num_points, bool)
    r = np.asarray(r, np.float64)
    return ~(np.isfinite(r) & (r > max_residual_cm))
