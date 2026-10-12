"""Per-clip floor calibration for GRAIL (z-up world).

Support surface = max(plane z=0, scene-object top surface) for terrain categories (slope/curb/stairs), plane z=0
otherwise. For every frame and foot: lowest foot vertex (G1-SMPL-X, GRAIL overlay) and its horizontal speed.
Stance = horizontal speed of the foot's 8 lowest vertices' centroid < 0.10 m/s AND gap to support < 0.10 m
AND GRAIL foot_contact_prob (max of heel/toe of that foot) > 0.5. Floor offset dz = median gap over stance
frames (both feet). The human is shifted by -dz (objects per category policy, see convert)."""
import numpy as np
import torch

TERRAIN = {'slope', 'curb', 'stair_p1', 'stair_p2'}


def object_world_verts(V_local, R, t):
    return V_local @ R.T + t


def support_height(xy, mesh_world):
    """Top-surface height of a static mesh under xy (rays cast downwards); 0 where no hit; returns max(0, hit)."""
    if mesh_world is None:
        return np.zeros(len(xy))
    zmax = float(mesh_world.vertices[:, 2].max()) + 1.0
    origins = np.c_[xy, np.full(len(xy), zmax)]
    dirs = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
    loc, idx_ray, _ = mesh_world.ray.intersects_location(origins, dirs, multiple_hits=False)
    h = np.zeros(len(xy))
    if len(idx_ray):
        h[idx_ray] = loc[:, 2]
    return np.maximum(h, 0.0)


def foot_tracks(body, verts):
    """verts torch [T,V,3] -> dict foot -> (low_xyz [T,3], centroid8 [T,3]) numpy."""
    out = {}
    for f, idx in body.foot_idx.items():
        fv = verts[:, torch.as_tensor(idx, device=verts.device)]  # [T,n,3]
        z = fv[..., 2]
        k = torch.topk(-z, 8, dim=1).indices  # 8 lowest
        low8 = torch.gather(fv, 1, k[..., None].expand(-1, -1, 3))
        out[f] = (low8[:, 0].cpu().numpy().astype(np.float64), low8.mean(1).cpu().numpy().astype(np.float64))
    return out


def calibrate(tracks, fps, contact_probs=None, mesh_world=None, speed_thr=0.10, gap_thr=0.10, upright=None):
    """upright: optional bool[T]; when given (object categories) only upright stance frames calibrate the floor,
    because GRAIL video recon drifts vertically in kneel/crouch phases (verified on pickup_ground)."""
    gaps, stance = {}, {}
    for f, (low, cen) in tracks.items():
        T = len(low)
        sp = np.zeros(T)
        if T > 1:
            d = np.linalg.norm(np.diff(cen[:, :2], axis=0), axis=1) * fps
            sp[1:] = d
            sp[0] = d[0]
        sup = support_height(low[:, :2], mesh_world)
        g = low[:, 2] - sup
        st = (sp < speed_thr) & (g < gap_thr)
        if contact_probs is not None:
            cp = contact_probs[:, [0, 1]] if f == 'L' else contact_probs[:, [2, 3]]
            st &= cp.max(1) > 0.5
        gaps[f], stance[f] = g, st
    flag = []
    if upright is not None:
        su = {f: stance[f] & upright for f in stance}
        if sum(int(v.sum()) for v in su.values()) >= 10:
            stance = su
        else:
            flag.append('few_upright_stance_frames_used_all_stance')
    allg = np.concatenate([gaps[f][stance[f]] for f in gaps])
    n = int(len(allg))
    if n < 10:
        allg = np.concatenate([np.sort(gaps[f])[: max(3, len(gaps[f]) // 10)] for f in gaps])
        flag.append('few_stance_frames_fallback_low10pct')
    dz = float(np.median(allg))
    rel = np.concatenate([gaps[f][stance[f]] for f in gaps]) - dz if n else allg - dz
    lowest_all = np.minimum(gaps['L'], gaps['R']) - dz
    return {
        'dz_m': dz,
        'stance_frames_L': int(stance['L'].sum()), 'stance_frames_R': int(stance['R'].sum()),
        'stance_gap_spread_p05_p95_mm': [float(np.percentile(rel, 5) * 1000), float(np.percentile(rel, 95) * 1000)],
        'foot_lowest_after_min_mm': float(lowest_all.min() * 1000),
        'foot_sink_gt10mm_frac_after': float((lowest_all < -0.01).mean()),
        'flags': flag,
    }, gaps, stance
