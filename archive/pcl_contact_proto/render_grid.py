"""12-frame grid (evenly spaced over the window) of the labelled dense point cloud, side view.
grey = surface; red = ground-contact point; orange = contact point on a group in SUSTAINED slip; blue = penetration > 1 cm;
black rings = legacy joint-centre contact (contact_hard) at the joint centres; brown line = floor used by the labeller.
    python render_grid.py   (renders every out/labels_*_window.npz to png/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REG = ['pelvis', 'trunk', 'head', 'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand',
       'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot', 'digit', 'base_or_link', 'palm']


def _wrap(items, per=4):
    if not items:
        return '-'
    return '\n  '.join(', '.join(items[i:i + per]) for i in range(0, len(items), per))


def render(npz_path, n=12):
    z = np.load(npz_path)
    X = z['X'].astype(np.float32); pc = z['point_contact']; pp = z['point_pen'].astype(np.float32)
    gn = [str(g) for g in z['group_names']]; region = z['region']; anchor = z['anchor']
    floor = float(z['floor_cm']); w0, w1 = [int(v) for v in z['window']]
    J = z['joint_world']; lc = z['legacy_contact']
    slip_s = z['slip_sustained']; cont = z['contact']; sps = z['slip_speed']
    T = X.shape[0]
    idx = np.linspace(0, T - 1, n).astype(int)
    root = J[:, 0, [0, 2]]
    disp = root - root.mean(0)
    if np.linalg.norm(root[-1] - root[0]) > 50 or np.ptp(disp, 0).max() > 50:
        ax_h = np.linalg.svd(disp, full_matrices=False)[2][0]
    else:
        P = X[:, ::8][..., [0, 2]].reshape(-1, 2); ax_h = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    depth_ax = np.array([-ax_h[1], ax_h[0]])
    # sustained-slip point mask: contact point whose REGION group is in sustained slip
    reg_g = {r: gn.index(f'region:{REG[r]}') for r in np.unique(region) if f'region:{REG[r]}' in gn}
    tag = Path(npz_path).stem.replace('labels_', '').replace('_window', '')
    top = float(np.percentile((X[idx][..., 1] - floor).max(1), 95)) + 10
    fig, axs = plt.subplots(2, 6, figsize=(30, 12))
    for k, ax in enumerate(axs.ravel()):
        t = idx[k]
        c0 = root[t]
        h = (X[t][:, [0, 2]] - c0) @ ax_h; v = X[t][:, 1] - floor; dep = (X[t][:, [0, 2]] - c0) @ depth_ax
        col = np.tile(np.array([0.72, 0.72, 0.72, 0.35]), (X.shape[1], 1))
        con = pc[t]
        slipm = np.zeros_like(con)
        for r, g in reg_g.items():
            if slip_s[t, g]:
                slipm |= (region == r) & con
        col[con] = (0.85, 0.05, 0.05, 1.0)
        col[slipm] = (1.0, 0.55, 0.0, 1.0)
        pen = pp[t] > 1.0
        col[pen] = (0.1, 0.25, 0.95, 1.0)
        o = np.argsort(-dep)
        ax.scatter(h[o], v[o], c=col[o], s=1.5, lw=0)
        jh = (J[t][:, [0, 2]] - c0) @ ax_h; jv = J[t][:, 1] - floor
        lm = lc[t]
        ax.scatter(jh[lm], jv[lm], s=60, facecolors='none', edgecolors='k', lw=1.2)
        ax.axhline(0, color='saddlebrown', lw=1.2)
        ax.set_aspect('equal'); ax.set_xlim(-100, 100); ax.set_ylim(-20, max(80.0, top))
        ax.set_xticks([]); ax.set_yticks([])
        regs = [gname[7:] for gi, gname in enumerate(gn) if gname.startswith('region:') and cont[t, gi]]
        slips = [f"{gname[7:]}:{sps[t, gi]:.0f}" for gi, gname in enumerate(gn) if gname.startswith('region:') and slip_s[t, gi]]
        nleg = int(lm.sum())
        ax.set_title(f"frame {w0 + t}  min d={v.min():.1f} cm  pen={pp[t].max():.1f} cm\n"
                     f"PCL: {_wrap(regs)}\nslip cm/s: {_wrap(slips)}\nlegacy contact joints: {nleg}",
                     fontsize=7, loc='left')
    fig.suptitle(f"{tag}  frames [{w0},{w1})  floor={floor:.2f} cm (session estimate)  |  red = point contact (enter<1 / leave>2 cm, "
                 f"link-gated)  orange = contact on a region in sustained slip (>4 cm/s)  blue = penetration>1 cm  black rings = legacy "
                 f"joint-centre contact", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / 'png' / f'grid_{tag}.png'
    fig.savefig(out, dpi=90); plt.close(fig)
    return out


if __name__ == '__main__':
    for f in sorted((HERE / 'out').glob('labels_*_window.npz')):
        print(render(f))
