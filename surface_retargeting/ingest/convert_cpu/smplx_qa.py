"""SMPL-X mesh QA for converted clips: per-clip floor estimate, sink/float stats,
object-on-floor check, and a 12-frame mesh+object render grid.

Conventions (verified against HiPHI docs/data_format.md and OmniContact raw files):
  world = Y-up, metres; object OBJ in centimetres, object-local;
  v_world = R(q_xyzw) @ (0.01 * v_cm) + p.
SMPL-X: neutral, flat_hand_mean=True, use_pca=False, 10 betas (same as UMR's loader).
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation

MODEL = f'{SRCFG.smplx_neutral}'
_MODEL = {}


def body_model(batch):
    key = batch
    if key not in _MODEL:
        _MODEL[key] = smplx.create(MODEL, model_type="smplx", gender="neutral", use_pca=False,
                                   flat_hand_mean=True, num_betas=10, batch_size=batch)
    return _MODEL[key]


@torch.no_grad()
def smplx_verts(poses, trans, betas, frames=None, chunk=256, reduce=None):
    """poses [T,165] axis-angle, trans [T,3], betas [10]. Returns verts [len(frames),V,3]
    or, with reduce=fn, a list of fn(verts_chunk) results (memory-light)."""
    poses = np.asarray(poses, np.float32).reshape(len(poses), -1)
    trans = np.asarray(trans, np.float32)
    idx = np.arange(len(poses)) if frames is None else np.asarray(frames)
    out = []
    faces = None
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        n = len(ii)
        m = body_model(n)
        p = torch.from_numpy(poses[ii])
        b = torch.from_numpy(np.asarray(betas, np.float32).reshape(1, -1)[:, :10]).expand(n, -1)
        o = m(betas=b, global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69],
              leye_pose=p[:, 69:72], reye_pose=p[:, 72:75], left_hand_pose=p[:, 75:120],
              right_hand_pose=p[:, 120:165], transl=torch.from_numpy(trans[ii]),
              expression=torch.zeros(n, 10))
        v = o.vertices.numpy()
        faces = m.faces
        out.append(reduce(v) if reduce else v)
    if reduce:
        return out
    return np.concatenate(out, 0), faces


def load_obj_cm(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, np.float64) * 0.01, np.asarray(m.faces)


def pose_obj(v_local_m, pq):
    """pq [7] = px,py,pz,qx,qy,qz,qw -> world verts."""
    R = Rotation.from_quat(pq[3:7]).as_matrix()
    return v_local_m @ R.T + pq[:3]


def floor_stats(min_y, low_xyz, fps, up_tol=0.01):
    """min_y [T]: lowest body vertex height (Y-up); low_xyz [T,3]: that vertex position.
    Floor = median lowest-vertex height over quasi-static support frames (lowest vertex
    horizontal speed < 10 cm/s), i.e. frames where something rests on the ground."""
    T = len(min_y)
    v = np.zeros(T)
    if T > 1:
        d = np.linalg.norm(np.diff(low_xyz[:, [0, 2]], axis=0), axis=1) * fps
        v[1:] = d
        v[0] = d[0]
    static = v < 0.10
    if static.sum() < 10:
        static = np.ones(T, bool)
    floor = float(np.median(min_y[static]))
    rel = min_y - floor
    return {
        "floor_y_m": round(floor, 4),
        "static_frac": round(float(static.mean()), 3),
        "static_lowest_p05_p95_cm": [round(float(np.percentile(rel[static], 5) * 100), 2),
                                     round(float(np.percentile(rel[static], 95) * 100), 2)],
        "static_lowest_p10_rel_median_cm": round(float(np.percentile(rel[static], 10) * 100), 2),
        "sink_gt1cm_frac": round(float((rel < -up_tol).mean()), 4),
        "sink_max_cm": round(float(-rel.min() * 100), 2),
        "above_gt3cm_frac": round(float((rel > 0.03).mean()), 4),
        "raw_lowest_p01_p50_p99_m": [round(float(np.percentile(min_y, q)), 4) for q in (1, 50, 99)],
    }


def lowest_reduce(v):
    i = v[:, :, 1].argmin(1)
    low = v[np.arange(len(v)), i]
    return low


def body_floor(poses, trans, betas, fps):
    lows = smplx_verts(poses, trans, betas, reduce=lowest_reduce)
    low = np.concatenate(lows, 0)
    return floor_stats(low[:, 1], low, fps), low


def object_floor_stats(obj_v_m, track, floor_y, fps):
    """track [T,7]. Lowest object vertex height rel. to body floor, on static object frames."""
    T = len(track)
    step = max(1, T // 600)
    ii = np.arange(0, T, step)
    mins = np.array([pose_obj(obj_v_m, track[i])[:, 1].min() for i in ii])
    sp = np.linalg.norm(np.diff(track[:, :3], axis=0), axis=1) * fps
    sp = np.concatenate([sp[:1], sp])[ii]
    rest = (sp < 0.02) & (mins - floor_y < 0.05)  # resting on/near the floor
    rel = (mins - floor_y) * 100
    out = {"obj_lowest_rel_floor_cm_p05_p50": [round(float(np.percentile(rel, 5)), 2), round(float(np.percentile(rel, 50)), 2)]}
    if rest.sum() >= 3:
        out["obj_resting_rel_floor_cm_median"] = round(float(np.median(rel[rest])), 2)
        out["obj_resting_frac"] = round(float(rest.mean()), 3)
    return out


def render_grid(out_png, poses, trans, betas, fps, floor_y, objects=(), title="", n=12, extra_rows=None):
    """objects: list of (name, v_local_m, faces, track[T,7])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    T = len(poses)
    frames = np.linspace(0, T - 1, n).round().astype(int)
    V, F = smplx_verts(poses, trans, betas, frames=frames)
    Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # (x,y,z)_yup -> (x,-z,y)_zup
    fig = plt.figure(figsize=(4 * 4, 3 * 4.2))
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    rows = []
    for k, t in enumerate(frames):
        ax = fig.add_subplot(3, 4, k + 1, projection="3d")
        v = V[k] @ Y2Z.T
        c = v.mean(0)
        tri = v[F]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        sh = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[sh * 0.55, sh * 0.65, sh * 0.85], linewidths=0))
        pen_info = []
        lo = v.min(0); hi = v.max(0)
        for name, ov, of, tr in objects:
            w = pose_obj(ov, tr[t])
            # penetration of body verts into object (closed meshes only; best-effort)
            try:
                om = trimesh.Trimesh(w, of, process=True)
                tag = ""
                if not om.is_watertight:
                    om = om.convex_hull
                    tag = "(hull)"
                near = np.linalg.norm(V[k] - w.mean(0), axis=1) < np.linalg.norm(w - w.mean(0), axis=1).max() + 0.05
                if near.any():
                    sd = trimesh.proximity.signed_distance(om, V[k][near])
                    pen_info.append(f"{name[:10]}{tag} pen {max(0.0, sd.max()) * 100:.1f}cm")
            except Exception:
                pass
            wz = w @ Y2Z.T
            ax.add_collection3d(Poly3DCollection(wz[of], facecolors=(0.95, 0.6, 0.2, 0.85), linewidths=0))
            lo = np.minimum(lo, wz.min(0)); hi = np.maximum(hi, wz.max(0))
        r = max(0.6, 0.5 * float((hi - lo)[:2].max()) + 0.1)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        fz = floor_y  # Y-up height becomes Z
        g = np.array([[cx - r, cy - r, fz], [cx + r, cy - r, fz], [cx + r, cy + r, fz], [cx - r, cy + r, fz]])
        ax.add_collection3d(Poly3DCollection([g], facecolors=(0.5, 0.5, 0.5, 0.25), edgecolors=(0.3, 0.3, 0.3, 0.6)))
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_zlim(fz - 0.1, fz - 0.1 + 2 * r)
        ax.view_init(elev=6, azim=-60)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        low = float(V[k][:, 1].min() - floor_y) * 100
        txt = f"f{t} t={t / fps:.1f}s low={low:+.1f}cm" + ("\n" + "; ".join(pen_info) if pen_info else "")
        ax.set_title(txt, fontsize=8)
        rows.append({"frame": int(t), "lowest_rel_floor_cm": round(low, 2), "obj": pen_info})
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=70)
    plt.close(fig)
    return rows
