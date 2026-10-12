"""Acceptance renders for surface_v2 (matplotlib, Agg, CPU).

    PY assets/render_assets.py [--bodies ...] [--grids]

png/<body>.png         : 4096 points coloured by region over the skeleton (sample pose front + side, AM2 identity
                         rest), sole close-up from below (feet), pad close-ups (adroit / franka).
png/grid_<body>.png    : 12 frames evenly over a real 128-frame LAFAN window (trajectory_reader, read-only), points
                         posed by surface_v2.pose_surface from the data (R, P) and lifted by the GT root height minus
                         beta_vis; points with clearance < 2.5 cm drawn red.  human/G1/H2/GR3.
png/human_overlay.png  : posed bone-revolution points over the LBS-posed SMPL-X mesh (standing A-pose).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'assets'

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import _env as E  # noqa: E402
from . import surface_v2 as S  # noqa: E402

CMAP = plt.get_cmap('tab20')
REG_COLORS = np.array([CMAP(i % 20) for i in range(len(E.REGIONS))])
GRID_WINDOW = ('lafan::fallAndGetUp1_subject4', 4136, 4264)   # plan's unseen-motion split-1 window (contact rich)
GRID_BODIES = ['human_smpl', 'unitree_g1_29dof', 'unitree_h2', 'fourier_gr3']


def _scatter(ax, pts, col, h, v, depth, s=2.0, alpha=1.0):
    o = np.argsort(depth)
    ax.scatter(pts[o, h], pts[o, v], c=col[o], s=s, lw=0, alpha=alpha)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def _skeleton(ax, P, parents, h, v, color='k'):
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([P[p, h], P[j, h]], [P[p, v], P[j, v]], color=color, lw=0.8, alpha=0.8)
    ax.scatter(P[:, h], P[:, v], c=color, s=4)


def _legend(fig, regions):
    from matplotlib.patches import Patch
    hs = [Patch(color=REG_COLORS[r], label=E.REGIONS[r]) for r in regions]
    fig.legend(handles=hs, loc='lower center', ncol=min(9, len(hs)), fontsize=7, frameon=False)


def sample_pose(body, z):
    if body == 'human_smpl':
        return z['rest_points'].astype(np.float64), z['calib_joint_positions'].astype(np.float64)
    R = z['sample_pose_rot'].astype(np.float64); P = z['sample_pose_pos'].astype(np.float64)
    a = z['anchor']
    return np.einsum('pij,pj->pi', R[a], z['points_local']) + P[a], P


def rest_joints(body, z, man):
    if body == 'human_smpl':
        return z['calib_joint_positions'].astype(np.float64)
    a = z['anchor']
    P = np.zeros((len(man['parents']), 3))
    # identity-rotation FK reconstructed from rest_points - points_local (exact for every anchor with points)
    for j in range(len(P)):
        m = a == j
        if m.any():
            P[j] = (z['rest_points'][m] - z['points_local'][m]).mean(0)
    return P


def body_png(body):
    z = np.load(E.ASSET_DIR / f'{body}.npz')
    man = E.read_json(E.ASSET_DIR / f'{body}.json')
    parents = man['parents']
    reg = z['region'].astype(int)
    col = REG_COLORS[reg]
    pts, P = sample_pose(body, z)
    extra = body in ('adroit_pen', 'relay_franka_full_v2')
    fig, axs = plt.subplots(1, 5, figsize=(22, 6.2))
    _scatter(axs[0], pts, col, 0, 1, pts[:, 2]); _skeleton(axs[0], P, parents, 0, 1)
    axs[0].set_title('sample pose, front (x | y)')
    _scatter(axs[1], pts, col, 2, 1, -pts[:, 0]); _skeleton(axs[1], P, parents, 2, 1)
    axs[1].set_title('sample pose, side (z | y)')
    rp = z['rest_points'].astype(np.float64); RP = rest_joints(body, z, man)
    _scatter(axs[2], rp, col, 0, 1, rp[:, 2]); _skeleton(axs[2], RP, parents, 0, 1)
    axs[2].set_title('AM2 rest (identity rotations) = rest_points')
    # close-ups
    nrm = None
    if body == 'human_smpl':
        nrm = z['rest_normals'].astype(np.float64)
    else:
        R = z['sample_pose_rot'].astype(np.float64)
        nrm = np.einsum('pij,pj->pi', R[z['anchor']], z['normals_local'])
    if not extra:
        feet = np.isin(reg, [E.RID['L_foot'], E.RID['R_foot']])
        down = feet & (nrm[:, 1] < -0.7)
        fp = pts[feet]
        sc = axs[3].scatter(fp[:, 0], fp[:, 2], c=fp[:, 1] - fp[:, 1].min(), s=10, cmap='viridis', lw=0)
        axs[3].scatter(pts[down, 0], pts[down, 2], facecolors='none', edgecolors='r', s=22, lw=0.6)
        axs[3].set_aspect('equal'); axs[3].set_title(f'feet from below: colour = height above lowest (cm)\n'
                                                     f'red rings: sole normals (n_y<-0.7) {int(down.sum())}/{int(feet.sum())} pts')
        fig.colorbar(sc, ax=axs[3], fraction=0.04)
        side = feet
        _scatter(axs[4], pts[side], col[side], 2, 1, -pts[side, 0], s=8)
        ymin = pts[feet, 1].min()
        axs[4].axhline(ymin, color='r', lw=0.6)
        axs[4].set_title(f"feet side view; lowest visual y={ymin:.1f} cm\nbeta_vis={man['beta_vis_cm']:.2f} cm "
                         f"(beta data check {((man.get('beta_data_check') or {}).get('median_lowest_foot_height_cm', float('nan'))):.2f})")
    else:
        padv = z['pad'].astype(int)
        hand = reg != E.RID['base_or_link']
        if body == 'relay_franka_full_v2':
            hand = np.isin(z['anchor'], [man['names'].index(n) for n in ('panda0_link7', 'panda0_leftfinger', 'panda0_rightfinger')])
        hp = pts[hand]
        pc = np.where(padv[hand] >= 0, 1.0, 0.0)
        pal = plt.get_cmap('Set1')
        cc = np.where(pc[:, None] > 0, np.array([pal(int(k) % 9) for k in padv[hand].clip(0)]), np.array([[0.75, 0.75, 0.75, 1]]))
        for ax, (h, v, d, t) in zip(axs[3:], [(0, 2, hp[:, 1], 'top view (x | z)'), (2, 1, -hp[:, 0], 'side view (z | y)')]):
            o = np.argsort(d)
            ax.scatter(hp[o, h], hp[o, v], c=cc[o], s=np.where(pc[o] > 0, 18, 4), lw=0)
            ax.set_aspect('equal'); ax.set_title('pads (coloured by pad id) ' + t)
        info = man.get('pads') or {}
        rows = info.get('digits') or info.get('fingers') or {}
        txt = '; '.join(f"{k}:{v.get('count')}pts" for k, v in rows.items())
        axs[4].set_xlabel(txt + f"\nverified_within_3mm={info.get('verified_within_3mm')}")
    cov = man['coverage']; bv = man.get('binding_verification', {})
    res = bv.get('max_residual_cm'); dres = (bv.get('data_residual') or {}).get('p95_of_frame_max_cm')
    fig.suptitle(f"{body}  |  {cov['points']} exterior points (UMR first-hit, 26 views, >=2 visible, FPS)  |  NN spacing p50 "
                 f"{cov['point_spacing_nn_cm']['p50']:.2f} cm  |  exterior area {cov['exterior_area_fraction'] * 100:.0f}% of "
                 f"{cov['total_visual_area_cm2']:.0f} cm2  |  binding {man['binding']}"
                 + (f", max residual {res:.2f} cm (random q), data p95 {dres:.2f} cm" if res is not None and dres is not None else
                    (f", max residual {res:.3f} cm" if res is not None else '')), fontsize=10)
    _legend(fig, sorted(set(reg.tolist())))
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = E.PNG_DIR / f'{body}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------------------------
def _trajectory_reader():
    import importlib.util
    spec = importlib.util.spec_from_file_location('am2_assets_trajectory_reader', str(E.ROOT / 'data' / 'trajectory_reader.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid_png(body, window=GRID_WINDOW, n=12):
    tr = _trajectory_reader()
    gid, s0, s1 = window
    d = tr.load_xwbc_trajectory(E.ROOT / 'data' / 'data_manifest_v2.json', gid, body, with_contact=True)
    sl = slice(s0, s1)
    R = d['global_rotation_matrices'][sl].astype(np.float32); P = d['position_native'][sl].astype(np.float32)
    rh = d['root_height_m'][sl].astype(np.float32)
    asset = S.load_surface(body)
    names = [str(x) for x in d['joint_names']]
    man = E.read_json(E.ASSET_DIR / f'{body}.json')
    assert names == man['names'], 'trajectory joint order differs from the asset'
    idx = np.linspace(0, s1 - s0 - 1, n).astype(int)
    with torch.no_grad():
        pts, _ = S.pose_surface(asset, torch.as_tensor(R[idx]), torch.as_tensor(P[idx]))
        h = S.ground_clearance(pts, torch.as_tensor(np.nan_to_num(rh[idx])), asset.beta_vis_cm).numpy()
    pts = pts.numpy()
    reg = asset.region.numpy()
    col = REG_COLORS[reg]
    fig, axs = plt.subplots(2, 6, figsize=(24, 9.5))
    for k, ax in enumerate(axs.ravel()):
        p = pts[k].copy(); p[:, 1] = h[k]                       # vertical = clearance above ground (cm)
        Pj = P[idx[k]].astype(np.float64).copy(); Pj[:, 1] += 100 * rh[idx[k]] - float(asset.beta_vis_cm)
        c = col.copy(); low = h[k] < 2.5
        c[low] = (0.85, 0.05, 0.05, 1)
        o = np.argsort(p[:, 2])
        ax.scatter(p[o, 0], p[o, 1], c=c[o], s=1.6, lw=0)
        _skeleton(ax, Pj, man['parents'], 0, 1)
        ax.axhline(0, color='saddlebrown', lw=1)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(-20, 200 if body != 'human_smpl' else 210); ax.set_xlim(-110, 110)
        lowreg = sorted({E.REGIONS[r] for r in reg[low]})
        ax.set_title(f"frame {s0 + idx[k]}  t={idx[k] / 30:.1f}s  min h={h[k].min():.1f} cm\n"
                     f"<2.5cm: {', '.join(lowreg) if lowreg else '-'}", fontsize=8)
    fig.suptitle(f"{body}: dense surface posed on real data ({gid} [{s0},{s1}) 128 frames, 12 evenly spaced)  |  "
                 f"pose_surface(R, P) + GT root height - beta_vis ({float(asset.beta_vis_cm):.2f} cm)  |  red = clearance < 2.5 cm  "
                 f"|  binding {asset.binding}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = E.PNG_DIR / f'grid_{body}.png'
    fig.savefig(out, dpi=100); plt.close(fig)
    return out


def human_overlay_png():
    f = E.ASSET_DIR / 'validation' / 'human_apose_overlay.npz'
    if not f.exists():
        return None
    z = np.load(f)
    asset = np.load(E.ASSET_DIR / 'human_smpl.npz')
    reg = asset['region'].astype(int)
    V = z['mesh_vertices']; pts = z['points']
    fig, axs = plt.subplots(1, 2, figsize=(14, 9))
    for ax, (h, v, dd, t) in zip(axs, [(0, 1, 2, 'front'), (2, 1, 0, 'side')]):
        ax.scatter(V[:, h], V[:, v], c='0.8', s=0.5, lw=0)
        o = np.argsort(pts[:, dd] if t == 'front' else -pts[:, dd])
        ax.scatter(pts[o, h], pts[o, v], c=REG_COLORS[reg[o]], s=3, lw=0)
        _skeleton(ax, z['joints'].astype(np.float64), E.read_json(E.ASSET_DIR / 'human_smpl.json')['parents'], h, v)
        ax.set_aspect('equal'); ax.set_title(f'standing A-pose {t}: grey = LBS-posed SMPL-X vertices, colour = bone-revolution '
                                             'points posed from joint positions only', fontsize=9)
    val = E.read_json(E.ASSET_DIR / 'validation' / 'human.json')['chamfer']['standing_apose']
    fig.suptitle(f"human_smpl overlay | points->mesh median {val['points_to_mesh_cm']['median']:.3f} cm, p95 "
                 f"{val['points_to_mesh_cm']['p95']:.2f} cm | mesh->points median {val['mesh_vertices_to_points_cm']['median']:.2f} cm")
    fig.tight_layout()
    out = E.PNG_DIR / 'human_overlay.png'
    fig.savefig(out, dpi=100); plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bodies', nargs='*', default=None)
    ap.add_argument('--grids', action='store_true')
    ap.add_argument('--no-bodies', action='store_true')
    a = ap.parse_args(argv)
    E.PNG_DIR.mkdir(parents=True, exist_ok=True)
    if not a.no_bodies:
        for b in (a.bodies or E.ALL_BODIES):
            print(body_png(b), flush=True)
        print(human_overlay_png())
    if a.grids:
        for b in GRID_BODIES:
            print(grid_png(b), flush=True)


if __name__ == '__main__':
    main()
