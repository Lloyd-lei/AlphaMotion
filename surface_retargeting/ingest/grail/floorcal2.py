"""GRAIL floor/support calibration v2 (Stream B).

Why v2 (measured on converted v1 clips, see report):
  * v1 probed the support with a ray at ONE vertex (the lowest foot vertex). On stairs/curbs that vertex often hangs over a
    step edge, so the probe read the lower step: e.g. stair clip f0 right foot +83 mm in v1 vs -3 mm true min clearance.
  * GRAIL is video reconstruction: the body drifts vertically within a clip (stance-foot gap p05..p95 ~ -20..+25 mm on
    terrain, frame-to-frame change only ~1 mm => slow drift, not jitter). A single constant offset cannot remove it.

v2:
  * Per-vertex clearance against the real 3D scene (floor z=0 + static scene mesh: terrain / chair / table box):
    a vertex is INSIDE when the first hit of an upward ray is an up-facing face (exits through a top surface); its
    penetration = min(vertical depth to that top, 3D distance to the surface). Otherwise its clearance = vertical distance
    to the first up-facing surface below it (or the floor). Works under chair seats / stair lips, unlike a heightfield.
  * Only vertices that are not obviously far above the scene (z <= top-surface heightfield + 0.12 m) are ray-tested.
  * Anchor a(t) = min clearance over SLOW (3D speed < 0.15 m/s) candidate vertices within 0.10 m of the support,
    head/face excluded. Anchors far from the clip median (> 60 mm) are rejected as outliers (pose errors, not drift).
  * dz(t) = Gaussian kernel regression of the anchors (sigma = 3 frames = 0.125 s, robust 2-pass with 20 mm residual
    cut); frames without anchors take the wider-kernel (sigma 1 s) estimate. The human is shifted by -dz(t).
  * Per-frame quality after correction: min clearance of all tested vertices (negative = penetration depth), anchor
    residual, and a validity mask (penetration <= PEN_TOL and lowest point within SUPPORT_TOL of a support).
"""
import numpy as np
import torch
import trimesh

HEAD_JOINTS = [12, 15, 22, 23, 24]  # neck, head, jaw, eyes
ARM_HAND_JOINTS = [18, 19, 20, 21] + list(range(25, 55))  # elbows, wrists, fingers: object contact, not support
SPEED_THR = 0.15
NEAR_THR = 0.10
OUTLIER_MM = 60.0
SIGMA_F = 3.0
SIGMA_WIDE_S = 1.0
PEN_TOL = 0.010     # source-human penetration tolerated in the validity mask (m)
SUPPORT_TOL = 0.020  # lowest body point must be within this of a support (m) unless seated/supported elsewhere


class Scene:
    """Static support scene: floor z=0 plus optional world-frame mesh (None => floor only)."""

    def __init__(self, mesh_world=None, xy_lo=None, xy_hi=None, res=0.01):
        self.m = mesh_world
        self.hf = None
        if mesh_world is not None:
            self.fn = np.asarray(mesh_world.face_normals)
            lo = np.asarray(xy_lo) - 0.2
            hi = np.asarray(xy_hi) + 0.2
            xs = np.arange(lo[0], hi[0] + res, res)
            ys = np.arange(lo[1], hi[1] + res, res)
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            xy = np.c_[X.ravel(), Y.ravel()]
            ztop = float(mesh_world.vertices[:, 2].max()) + 1.0
            loc, ir, _ = mesh_world.ray.intersects_location(np.c_[xy, np.full(len(xy), ztop)],
                                                            np.tile([0.0, 0.0, -1.0], (len(xy), 1)), multiple_hits=False)
            H = np.zeros(len(xy))
            H[ir] = loc[:, 2]
            H = np.maximum(H, 0.0).reshape(X.shape)
            # dilate by one cell (max filter) so the prefilter never misses an edge
            Hp = np.pad(H, 1, mode='edge')
            Hd = np.max(np.stack([Hp[i:i + H.shape[0], j:j + H.shape[1]] for i in range(3) for j in range(3)]), 0)
            self.hf = (Hd, lo, res)

    def top(self, xy):
        if self.hf is None:
            return np.zeros(len(xy))
        H, lo, res = self.hf
        i = np.clip(np.round((xy[:, 0] - lo[0]) / res).astype(int), 0, H.shape[0] - 1)
        j = np.clip(np.round((xy[:, 1] - lo[1]) / res).astype(int), 0, H.shape[1] - 1)
        return H[i, j]

    def clearance(self, P):
        """P [n,3] -> clearance [n] (m): >=0 gap to the support below, <0 penetration depth."""
        c = P[:, 2].copy()  # floor
        if self.m is None or len(P) == 0:
            return c
        eps = 1e-5
        n = len(P)
        up = np.tile([0.0, 0.0, 1.0], (n, 1))
        dn = -up
        inside = np.zeros(n, bool)
        dep_v = np.full(n, np.inf)
        loc, ir, it = self.m.ray.intersects_location(P - up * eps, up, multiple_hits=False)
        if len(ir):
            exit_top = self.fn[it, 2] > 0
            inside[ir[exit_top]] = True
            dep_v[ir[exit_top]] = loc[exit_top, 2] - P[ir[exit_top], 2]
        out = ~inside
        if out.any():
            oi = np.where(out)[0]
            loc, ir, it = self.m.ray.intersects_location(P[oi] + up[oi] * eps, dn[oi], multiple_hits=False)
            if len(ir):
                ok = self.fn[it, 2] > 0  # landing on an up-facing surface
                g = P[oi[ir[ok]], 2] - loc[ok, 2]
                c[oi[ir[ok]]] = np.minimum(c[oi[ir[ok]]], g)
        if inside.any():
            ii = np.where(inside)[0]
            _, d3, _ = trimesh.proximity.closest_point(self.m, P[ii])
            depth = np.minimum(dep_v[ii], d3)
            c[ii] = -depth
        c = np.minimum(c, P[:, 2])  # below the floor is penetration too
        return c


def _kernel_fit(t_a, a, T, sigma):
    t = np.arange(T, dtype=np.float64)[:, None]
    w = np.exp(-0.5 * ((t - t_a[None]) / sigma) ** 2)
    sw = w.sum(1)
    return (w * a[None]).sum(1) / np.maximum(sw, 1e-12), sw


def calibrate_v2(body, verts, fps, scene, upright=None, fit=True):
    """verts: torch [T,V,3] on device (already in the clip's world frame, before any shift).
    Returns dict with dz [T] (m), per-frame arrays, summary; arrays are float32."""
    T, V, _ = verts.shape
    dev = verts.device
    keep = ~np.isin(body.dom, HEAD_JOINTS)
    kidx = torch.as_tensor(np.where(keep)[0], device=dev)
    is_hand = np.isin(body.dom[np.where(keep)[0]], ARM_HAND_JOINTS)  # [K]
    P = verts[:, kidx]  # [T,K,3]
    vel = torch.zeros(T, P.shape[1], device=dev)
    if T > 2:
        d = (P[2:] - P[:-2]).norm(dim=-1) * fps / 2.0
        vel[1:-1] = d
        vel[0] = d[0]
        vel[-1] = d[-1]
    Pn = P.cpu().numpy().astype(np.float64)
    veln = vel.cpu().numpy()
    K = Pn.shape[1]
    flat = Pn.reshape(-1, 3)
    top = scene.top(flat[:, :2]).reshape(T, K)
    cand = Pn[..., 2] <= top + 0.12
    clr = np.full((T, K), np.inf)
    ci = np.where(cand.ravel())[0]
    if len(ci):
        clr.ravel()[ci] = scene.clearance(flat[ci])
    slow = veln < SPEED_THR
    near = clr < NEAR_THR
    a_mask = slow & near & ~is_hand[None]
    a = np.where(a_mask, clr, np.inf).min(1)
    has = np.isfinite(a)
    if upright is not None and (has & upright).sum() >= 10:
        has_fit = has & upright
    else:
        has_fit = has
    flags = []
    if fit and has_fit.sum() >= 5:
        med = float(np.median(a[has_fit]))
        good = has_fit & (np.abs(a - med) * 1000 <= OUTLIER_MM)
        ta = np.where(good)[0].astype(np.float64)
        dz, sw = _kernel_fit(ta, a[good], T, SIGMA_F)
        res = a - dz
        good2 = good & (np.abs(res) <= 0.020)
        if good2.sum() >= 5:
            ta = np.where(good2)[0].astype(np.float64)
            good = good2
        dz, sw = _kernel_fit(ta, a[good], T, SIGMA_F)
        dzw, _ = _kernel_fit(ta, a[good], T, SIGMA_WIDE_S * fps)
        far = sw < 0.05  # no anchor within ~2.4 sigma
        dz[far] = dzw[far]
        n_anchor = int(good.sum())
    elif fit:
        lowest = np.sort(np.where(np.isfinite(clr), clr, np.inf)[:, ~is_hand].min(1))
        dz = np.full(T, float(np.median(lowest[: max(3, T // 10)])) if np.isfinite(lowest).any() else 0.0)
        flags.append('few_anchor_frames_constant_low10pct')
        n_anchor = int(has_fit.sum())
        good = has_fit
    else:
        n_anchor = int(has_fit.sum())
        good = has_fit
    if not fit:
        dz = np.zeros(T)
    # quality after shifting the human by -dz(t)
    clr_after = clr - dz[:, None]
    low_after = np.where(np.isfinite(clr_after), clr_after, np.inf)[:, ~is_hand].min(1)  # lowest body (non-arm) clearance
    hand_low = np.where(np.isfinite(clr_after), clr_after, np.inf)[:, is_hand].min(1)
    hand_pen = np.maximum(0.0, -hand_low)
    anchor_res = np.where(has, a - dz, np.nan)
    pen = np.maximum(0.0, -low_after)
    valid = (pen <= PEN_TOL) & (low_after <= SUPPORT_TOL)
    out = {
        'dz': dz.astype(np.float32),
        'anchor': np.where(has, a, np.nan).astype(np.float32),
        'anchor_used': good,
        'lowest_clearance_after': np.where(np.isfinite(low_after), low_after, np.nan).astype(np.float32),
        'valid': valid,
        'hand_pen': hand_pen.astype(np.float32),
        'summary': {
            'hand_scene_pen_gt10mm_frac': round(float((hand_pen > 0.010).mean()), 4),
            'dz_median_mm': round(float(np.median(dz)) * 1000, 2),
            'dz_p05_p95_mm': [round(float(np.percentile(dz, 5)) * 1000, 2), round(float(np.percentile(dz, 95)) * 1000, 2)],
            'dz_range_mm': round(float(dz.max() - dz.min()) * 1000, 2),
            'anchor_frames': int(has.sum()), 'anchor_frames_used': n_anchor,
            'anchor_residual_p05_p50_p95_mm': [round(float(np.nanpercentile(anchor_res[good], q)) * 1000, 2) for q in (5, 50, 95)] if good.any() else None,
            'pen_gt1mm_frac': round(float((pen > 0.001).mean()), 4),
            'pen_gt10mm_frac': round(float((pen > 0.010).mean()), 4),
            'pen_p95_max_mm': [round(float(np.percentile(pen, 95)) * 1000, 2), round(float(pen.max()) * 1000, 2)],
            'lowest_gt20mm_frac': round(float((low_after > SUPPORT_TOL).mean()), 4),
            'valid_frac': round(float(valid.mean()), 4),
            'flags': flags,
        },
    }
    return out


# ---------------------------------------------------------------------------------------------------------------
# Horizontal scene registration (terrain only): GRAIL places the generated terrain mesh and the video-reconstructed
# human independently; on stairs the feet sink into risers (pen>10 mm in ~25% frames). We search a rigid xy offset of
# the static terrain (2.5D top-surface heightfield, GPU) maximising the fraction of frames whose human source is
# physically valid after the dz(t) fit (no penetration > PEN_TOL, lowest point within SUPPORT_TOL of support).

def _hf_top_exact(mesh_world, lo, hi, res):
    xs = np.arange(lo[0], hi[0] + res, res)
    ys = np.arange(lo[1], hi[1] + res, res)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xy = np.c_[X.ravel(), Y.ravel()]
    ztop = float(mesh_world.vertices[:, 2].max()) + 1.0
    loc, ir, _ = mesh_world.ray.intersects_location(np.c_[xy, np.full(len(xy), ztop)],
                                                    np.tile([0.0, 0.0, -1.0], (len(xy), 1)), multiple_hits=False)
    H = np.zeros(len(xy))
    H[ir] = loc[:, 2]
    return np.maximum(H, 0.0).reshape(X.shape)


@torch.no_grad()
def register_xy(body, verts, fps, mesh_world, radius=0.20, step=0.01, res=0.01):
    """Returns (best_offset_xy [2], score_best, score_zero, table) where the terrain should be moved by +offset."""
    dev = verts.device
    keep = ~np.isin(body.dom, HEAD_JOINTS + ARM_HAND_JOINTS)
    P = verts[:, torch.as_tensor(np.where(keep)[0], device=dev)][:, ::2]  # [T,K,3], half the body (non-arm) vertices
    T = P.shape[0]
    vel = torch.zeros(T, P.shape[1], device=dev)
    if T > 2:
        d = (P[2:] - P[:-2]).norm(dim=-1) * fps / 2.0
        vel[1:-1] = d; vel[0] = d[0]; vel[-1] = d[-1]
    slow = vel < SPEED_THR
    xy = P[..., :2].reshape(-1, 2)
    lo = xy.min(0).values.cpu().numpy() - radius - 0.05
    hi = xy.max(0).values.cpu().numpy() + radius + 0.05
    H = torch.as_tensor(_hf_top_exact(mesh_world, lo, hi, res), device=dev, dtype=torch.float32)
    lo_t = torch.as_tensor(lo, device=dev, dtype=torch.float32)
    offs = np.arange(-radius, radius + 1e-9, step)
    grid = np.array([(a, b) for a in offs for b in offs])
    tt = torch.arange(T, device=dev, dtype=torch.float32)
    W = torch.exp(-0.5 * ((tt[:, None] - tt[None]) / SIGMA_F) ** 2)  # [T,T]
    Ww = torch.exp(-0.5 * ((tt[:, None] - tt[None]) / (SIGMA_WIDE_S * fps)) ** 2)
    scores = np.zeros(len(grid))
    z = P[..., 2]
    for i0 in range(0, len(grid), 16):
        g = torch.as_tensor(grid[i0:i0 + 16], device=dev, dtype=torch.float32)  # [B,2]
        q = (P[None, ..., :2] - g[:, None, None, :] - lo_t) / res  # [B,T,K,2]
        ii = q[..., 0].round().long().clamp(0, H.shape[0] - 1)
        jj = q[..., 1].round().long().clamp(0, H.shape[1] - 1)
        c = z[None] - H[ii, jj]  # [B,T,K]
        cand = slow[None] & (c < NEAR_THR)
        a = torch.where(cand, c, torch.full_like(c, float('inf'))).min(-1).values  # [B,T]
        has = torch.isfinite(a)
        med = torch.stack([a[b][has[b]].median() if has[b].any() else torch.tensor(0.0, device=dev) for b in range(len(g))])
        good = has & ((a - med[:, None]).abs() <= OUTLIER_MM / 1000)
        av = torch.where(good, a, torch.zeros_like(a))
        gw = good.float()
        num = av @ W.T; den = gw @ W.T
        numw = av @ Ww.T; denw = gw @ Ww.T
        dz = torch.where(den >= 0.05, num / den.clamp_min(1e-9), numw / denw.clamp_min(1e-9))
        low = c.min(-1).values - dz
        valid = (low >= -PEN_TOL) & (low <= SUPPORT_TOL) & (denw > 1e-6)
        scores[i0:i0 + len(g)] = valid.float().mean(-1).cpu().numpy()
    i_best = int(np.argmax(scores - 1e-4 * np.linalg.norm(grid, axis=1)))  # prefer small offsets on ties
    i_zero = int(np.argmin(np.linalg.norm(grid, axis=1)))
    return grid[i_best], float(scores[i_best]), float(scores[i_zero]), (grid, scores)
