"""UMR's LQR trajectory filter for chains of independent pieces, on the GPU (batched banded Cholesky).

For every piece:  (d I + v D1'D1 + a D2'D2 + j D3'D3) x = d x_raw  with the first/last `anchor` frames fixed
(exactly UMR's lqr_smooth_qpos_sequence); difference rows never straddle two pieces.  All chains share no matrix, so
the band (diagonal + 3 super-diagonals) is built per chain and factorised by a sequential-in-time, batched-over-chain
Cholesky (2*T small kernels).
"""
from __future__ import annotations

import torch

COEF = {1: (-1.0, 1.0), 2: (1.0, -2.0, 1.0), 3: (-1.0, 3.0, -3.0, 1.0)}


def band_system(piece, tl, rem, d, costs, s_anchor=1, e_anchor=1):
    """piece [B,T] (long, -1 idle) -> band [B,T,4] (M[i,i+o]), fixed mask [B,T]."""
    B, T = piece.shape
    dt = torch.float64
    band = torch.zeros(B, T, 4, dtype=dt, device=piece.device)
    valid = piece >= 0
    band[..., 0] = d * valid
    for k, c in costs.items():
        if c <= 0 or T <= k:
            continue
        co = COEF[k]
        # row r covers frames r..r+k; valid iff all in the same piece
        same = valid[:, :T - k].clone()
        for o in range(1, k + 1):
            same &= piece[:, o:T - k + o] == piece[:, :T - k]
        w = c * same.to(dt)                                         # [B, T-k]
        for i in range(k + 1):
            for jj in range(i, k + 1):
                o = jj - i
                band[:, i:T - k + i, o] += w * co[i] * co[jj]
    # fixed frames: first s_anchor and last e_anchor frames of every piece (tl = index in piece, rem = frames left)
    fixed = valid & ((tl < s_anchor) | (rem <= e_anchor))
    fixed |= ~valid
    return band, fixed


def banded_solve(band, fixed, X, rhs):
    """Solve M x = rhs with fixed rows (x = X there).  band [B,T,4] SPD band, X/rhs [B,T,C]."""
    B, T, _ = band.shape
    p = 3
    M = band.clone()
    rhs = rhs.clone()
    fx = fixed.to(M.dtype)
    # move fixed columns to the rhs: rhs[i] -= M[i,j] X[j] for j fixed, |i-j|<=p
    for o in range(1, p + 1):
        # upper: (i, i+o) with i+o fixed
        rhs[:, :T - o] -= (M[:, :T - o, o] * fx[:, o:])[..., None] * X[:, o:]
        # lower: (i+o, i) with i fixed
        rhs[:, o:] -= (M[:, :T - o, o] * fx[:, :T - o])[..., None] * X[:, :T - o]
        keep = (1 - fx[:, :T - o]) * (1 - fx[:, o:])
        M[:, :T - o, o] *= keep
    M[..., 0] = torch.where(fixed, torch.ones_like(M[..., 0]), M[..., 0])
    rhs = torch.where(fixed[..., None], X, rhs)
    # banded Cholesky  M = L L^T, L lower with bandwidth p; store L[i, i-o] at Lb[i, o]
    Lb = torch.zeros_like(M)
    for i in range(T):
        for o in range(p, 0, -1):
            j = i - o
            if j < 0:
                continue
            s = M[:, j, o].clone()
            for k in range(1, p - o + 1):
                if j - k >= 0:
                    s -= Lb[:, i, o + k] * Lb[:, j, k]
            Lb[:, i, o] = s / Lb[:, j, 0]
        s = M[:, i, 0].clone()
        for o in range(1, p + 1):
            if i - o >= 0:
                s -= Lb[:, i, o] ** 2
        Lb[:, i, 0] = torch.sqrt(s.clamp_min(1e-300))
    y = torch.zeros_like(rhs)
    for i in range(T):
        s = rhs[:, i].clone()
        for o in range(1, p + 1):
            if i - o >= 0:
                s -= Lb[:, i, o, None] * y[:, i - o]
        y[:, i] = s / Lb[:, i, 0, None]
    x = torch.zeros_like(rhs)
    for i in range(T - 1, -1, -1):
        s = y[:, i].clone()
        for o in range(1, p + 1):
            if i + o < T:
                s -= Lb[:, i + o, o, None] * x[:, i + o]
        x[:, i] = s / Lb[:, i, 0, None]
    return x


def lqr_chains(Q, piece, tl, rem, a, cols, clamp_fn):
    """Q [B,T,nq] raw chain trajectories, piece [B,T] -> filtered (UMR LQR per piece)."""
    if str(a['trajectory_filter_mode']).lower() != 'lqr':
        return Q
    d = max(float(a['trajectory_filter_data_cost']), 1e-12)
    costs = {1: max(float(a['trajectory_filter_velocity_cost']), 0.0), 2: max(float(a['trajectory_filter_acceleration_cost']), 0.0),
             3: max(float(a['trajectory_filter_jerk_cost']), 0.0)}
    if all(c <= 0 for c in costs.values()):
        return Q
    cols = sorted(set(([0, 1, 2] if bool(a['trajectory_filter_root_translation']) else []) + [int(c) for c in cols]))
    ci = torch.as_tensor(cols, device=Q.device)
    band, fixed = band_system(piece, tl, rem, d, costs, int(a['trajectory_filter_anchor_start_frames']), int(a['trajectory_filter_anchor_end_frames']))
    X = Q[:, :, ci]
    x = banded_solve(band, fixed, X, d * X)
    out = Q.clone()
    out[:, :, ci] = x
    B, T, nq = out.shape
    return clamp_fn(out.reshape(B * T, nq)).reshape(B, T, nq)
