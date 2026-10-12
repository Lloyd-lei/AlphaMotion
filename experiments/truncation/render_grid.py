"""12-frame grid (uniform over whole clip) + prefix boundary + coverage/contact timeline.

usage: python render_grid.py <motion_id> [N_boundary=20] [out.png]
Human = SMPL-X mesh (smplx FK), object = OBJ mesh posed by object_tracks.
"""
import json, os, sys, textwrap
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import gridspec
from scipy.spatial.transform import Rotation as R
import trimesh
import torch, smplx
torch.set_num_threads(2)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SMPLX_PATH = '/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz'
LIGHT = np.array([0.3, 0.5, 0.8]); LIGHT /= np.linalg.norm(LIGHT)


def to_plot(P):  # world y-up -> plot z-up (proper rotation)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def shaded(V, F, base):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lam = 0.45 + 0.55 * np.abs(n @ LIGHT)
    col = np.clip(np.asarray(base)[None, :3] * lam[:, None], 0, 1)
    return tri, col


def smplx_verts(npz, frames):
    d = np.load(npz)
    p = torch.tensor(d['poses'][frames], dtype=torch.float32)
    T = len(frames)
    m = smplx.create(SMPLX_PATH, model_type='smplx', gender='neutral', use_pca=False, num_betas=10,
                     flat_hand_mean=True, batch_size=T)
    out = m(betas=torch.tensor(d['betas'][:10], dtype=torch.float32)[None].repeat(T, 1),
            global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72],
            reye_pose=p[:, 72:75], left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
            transl=torch.tensor(d['trans'][frames], dtype=torch.float32))
    return out.vertices.detach().numpy(), m.faces.astype(np.int64)


def main(mid, NB=20, out=None, title_extra=''):
    labs = load_labels(); idx = clip_index()
    r = labs[mid]; info = idx[mid]
    J, Fe, h, fl = load_clip(info)
    T = len(J)
    meta = json.load(open(info['dir'] + '/metadata.json'))
    ev = object_events(info, meta, J) if meta.get('objects') else None
    em = np.load(OUT + '/em%s/%s.npz' % (SUFFIX, mid))
    # 12 uniform frames over whole clip (10 Hz index), map to npz rate
    k12 = np.linspace(0, T - 1, 12).round().astype(int)
    n = NB * HZ
    D, H = dist_mats_m(Fe[n:], h[n:], Fe[:n], h[:n])
    Dp = np.where(H < 0.05, D, 1.0)
    nn = Dp.min(1); nn_arg = Dp.argmin(1)
    nnd = np.full(T, np.nan); nnd[n:] = nn
    # two most novel frames (>= 3 s apart) and their nearest prefix frames
    order = np.argsort(-nn); nov = []
    for i in order:
        if all(abs(i - j) >= 3 * HZ for j in nov):
            nov.append(i)
        if len(nov) == 2:
            break
    pairs = []
    for i in nov:
        pairs += [i + n, nn_arg[i]]
    kall = np.r_[k12, np.array(pairs, int)]
    V, Fsm = smplx_verts(info['npz'], kall * info['step'])
    if ev is not None:
        om = trimesh.load(ROOT + '/raw/hiphi/object_meshes/%s.obj' % ev['mesh_id'], force='mesh', process=False)
        if len(om.faces) > 12000:
            sel = np.random.default_rng(0).choice(len(om.faces), 12000, replace=False)
            oF = np.asarray(om.faces)[sel]
        else:
            oF = np.asarray(om.faces)
        oV = np.asarray(om.vertices) * 0.01
    # azimuth from median facing
    across = (J[:, 1] - J[:, 2]) + (J[:, 16] - J[:, 17])
    fwd = np.cross(across, [0, 1, 0])
    fp = to_plot(fwd)
    ang = np.arctan2(fp[:, 1], fp[:, 0])
    az = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 35

    fig = plt.figure(figsize=(16, 21))
    gs = gridspec.GridSpec(6, 4, height_ratios=[1, 1, 1, 1, 0.35, 0.62], hspace=0.10, wspace=0.0)
    for p in range(len(kall)):
        ax = fig.add_subplot(gs[p // 4, p % 4], projection='3d')
        t = kall[p]
        inside = t < NB * HZ
        Vp = to_plot(V[p])
        tri, col = shaded(Vp, Fsm, (0.55, 0.62, 0.85) if inside else (0.85, 0.6, 0.5))
        polys = [Poly3DCollection(tri, facecolors=col, edgecolors='none', linewidths=0)]
        pel = to_plot(J[t, 0])
        cx, cy = pel[0], pel[1]
        if ev is not None:
            Rm = R.from_quat(ev['quat'][t])
            ow = to_plot(Rm.apply(oV) + ev['pos'][t])
            otri, ocol = shaded(ow, oF, (0.75, 0.68, 0.45))
            polys.append(Poly3DCollection(otri, facecolors=ocol, edgecolors='none', linewidths=0))
            oc = ow.mean(0)
            if np.hypot(oc[0] - cx, oc[1] - cy) < 1.6:
                cx, cy = 0.6 * cx + 0.4 * oc[0], 0.6 * cy + 0.4 * oc[1]
        for pc in polys:
            ax.add_collection3d(pc)
        hw = 0.95
        fz = fl
        # floor grid
        gx = np.arange(np.floor((cx - hw) * 2) / 2, cx + hw + 1e-6, 0.5)
        gy = np.arange(np.floor((cy - hw) * 2) / 2, cy + hw + 1e-6, 0.5)
        for x in gx:
            if cx - hw <= x <= cx + hw:
                ax.plot([x, x], [cy - hw, cy + hw], [fz, fz], color='0.8', lw=0.6)
        for y in gy:
            if cy - hw <= y <= cy + hw:
                ax.plot([cx - hw, cx + hw], [y, y], [fz, fz], color='0.8', lw=0.6)
        ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw); ax.set_zlim(fz, fz + 2 * hw)
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
        ax.view_init(elev=14, azim=az)
        ax.set_axis_off()
        if inside:
            cov = 'kept'
        else:
            cov = ('covered' if em['f5'][t] < NB * HZ else 'NEW') + ' (NN %.1fcm)' % (100 * nnd[t])
        ttl = 't=%.1fs  %s' % (t / HZ, cov)
        if p >= 12:
            q = (p - 12) // 2
            ttl = ('NOVEL#%d t=%.1fs NN %.1fcm' % (q + 1, t / HZ, 100 * nnd[t])) if (p - 12) % 2 == 0 else \
                  '<- nearest in first %ds: t=%.1fs' % (NB, t / HZ)
        ax.set_title(ttl, fontsize=11,
                     color=('#1f4e9c' if inside else ('#555555' if cov.startswith('covered') else '#c0392b')), pad=-6)
    # NN distance trace
    axn = fig.add_subplot(gs[4, :])
    YMAX = 20 if METRIC == 'v1' else 40
    LINES = ((3, '#2e86c1'), (5, '#c0392b'), (8, '#d68910')) if METRIC == 'v1' else ((10, '#2e86c1'), (12, '#c0392b'), (15, '#d68910'))
    axn.plot(np.arange(T) / HZ, 100 * np.clip(nnd, 0, YMAX / 100), color='k', lw=0.8)
    for y, c in LINES:
        axn.axhline(y, color=c, lw=0.8, ls='--')
        axn.text(T / HZ, y, ' %dcm' % y, color=c, fontsize=8, va='center')
    axn.set_xlim(0, T / HZ); axn.set_ylim(0, YMAX)
    axn.set_ylabel('NN %s-joint dist\nto first %ds (cm)' % ('mean' if METRIC == 'v1' else 'max', NB), fontsize=9)
    axn.tick_params(labelbottom=False)
    axn.axvline(NB, color='k', ls='--', lw=1.2)
    # timeline
    axt = fig.add_subplot(gs[5, :])
    tt = np.arange(T) / HZ
    rows = []
    for N in (15, 20, 30):
        c = np.where(np.arange(T) < N * HZ, 2, (em['f5'] < N * HZ).astype(int))
        rows.append(('frame cover by first %ds' % N, c))
    emw = em['w5']
    cw = np.full(T, 0)
    cw[:len(emw)] = (emw <= NB * HZ - HZ)
    cw[:NB * HZ] = 2
    rows.append(('1s-window cover, first %ds' % NB, cw))
    cmap = {0: '#e74c3c', 1: '#7fbf7f', 2: '#9db4e0'}
    for i, (name, c) in enumerate(rows):
        for s, e in runs(c == 0):
            axt.add_patch(plt.Rectangle((s / HZ, i), (e - s) / HZ, 0.8, color=cmap[0], lw=0))
        for s, e in runs(c == 1):
            axt.add_patch(plt.Rectangle((s / HZ, i), (e - s) / HZ, 0.8, color=cmap[1], lw=0))
        for s, e in runs(c == 2):
            axt.add_patch(plt.Rectangle((s / HZ, i), (e - s) / HZ, 0.8, color=cmap[2], lw=0))
    labels = [x[0] for x in rows]
    if ev is not None:
        i = len(rows)
        for s, e in ev['move_ep']:
            axt.add_patch(plt.Rectangle((s / HZ, i), (e - s) / HZ, 0.8, color='#d4a017', lw=0))
        labels.append('object moving')
        i += 1
        for name, key, col in (('hand contact', 'hand', '#8e44ad'), ('pelvis/foot contact', None, '#16a085')):
            b = ev['hand'] if key else (ev['pel'] | ev['foot'])
            for s, e in runs(b, int(0.5 * HZ), int(0.3 * HZ)):
                axt.add_patch(plt.Rectangle((s / HZ, i), (e - s) / HZ, 0.8, color=col, lw=0))
            labels.append(name)
            i += 1
    # root height trace (scaled) on top
    axt.set_yticks(np.arange(len(labels)) + 0.4); axt.set_yticklabels(labels, fontsize=9)
    axt.set_xlim(0, T / HZ); axt.set_ylim(-0.2, len(labels))
    for N, ls in ((15, ':'), (20, '--'), (30, '-.')):
        axt.axvline(N, color='k', ls=ls, lw=1.2)
        axt.text(N, len(labels) - 0.05, ' %ds' % N, fontsize=9, va='bottom')
    for t in k12:
        axt.plot([t / HZ], [-0.12], marker='^', color='k', ms=5)
    axt.set_xlabel('time (s)   [blue = kept prefix, green = covered by prefix (%s-joint < %d cm), red = not covered]' % ('mean' if METRIC == 'v1' else 'max', round(PRIMARY_THR * 100)), fontsize=10)
    axt.invert_yaxis()
    # header
    import json as _j
    row = None
    for line in open(OUT + '/cov_all%s.jsonl' % SUFFIX):
        if line.startswith('{"motion_id": "%s"' % mid):
            row = _j.loads(line); break
    covs = ''
    if row:
        covs = ('cover(rest|first N, %s-joint<%dcm): 15s' % ('mean' if METRIC == 'v1' else 'max', round(PRIMARY_THR * 100))) + '  %.0f%%  20s %.0f%%  30s %.0f%%   | 1s-window 20s %.0f%%' % (
            100 * (row.get('cov_f5_15') or 0), 100 * (row.get('cov_f5_20') or 0), 100 * (row.get('cov_f5_30') or 0),
            100 * (row.get('cov_w5_20') or 0))
        if row.get('first_cycle_end_s') is not None:
            covs += '   | first full interaction cycle ends %.1fs' % row['first_cycle_end_s']
    head = '%s   [%s / %s / obj=%s]   %.1fs   %s' % (mid, group_of(r), GROUP_ZH.get(group_of(r), ''),
                                                   r['object_categories'] or '-', T / HZ, title_extra)
    fig.suptitle(head + '\n' + '\n'.join(textwrap.wrap(r['text_annotation'], 150)) + '\n' + covs,
                 fontsize=12, y=0.995, va='top')
    out = out or OUT + '/renders%s/%s_N%d.png' % (SUFFIX, mid, NB)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=72, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    mid = sys.argv[1]
    NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(mid, NB, sys.argv[3] if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else '')
