from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, glob, json, os
import numpy as np
from surface_retargeting.ingest.uniml3d.reload import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = f'{SRCFG.factory_root}/uniml3d_reload/remote/'  # data, not code
OUT = f'{SRCFG.factory_root}/uniml3d_reload/proto/out'; os.makedirs(OUT, exist_ok=True)
FPS = 30.0
CASES = {
    'objaverse_quadruped_e457402d_Walk': (R + 'glb/e457402d4b2e4681b5a5f88b747aea4a.glb', glob.glob(R + 'export/objaverse/motions/*/e457402d*-Walk_Object_4.npz')[0]),
    'objaverse_biped_669a82d1_walk000': (R + 'glb/669a82d157bbceb71e80ada5_glb.glb', glob.glob(R + 'export/objaverse/motions/*/669a82d1*walk_000*.npz')[0]),
    'objaverse_object_669a76c5c6_Take001': (R + 'glb/669a76c557bbceb71e7106c6_fbx.glb', glob.glob(R + 'export/objaverse/motions/*/669a76c5*Take_001*.npz')[0]),
    'mixamo_YBot_Big_Jump': (R + 'glb/Y_Bot.glb', R + 'export/mixamo/motions/Big_Jump.npz'),
    'mixamo_YBot_180_Turn_W_Briefcase': (R + 'glb/Y_Bot.glb', R + 'export/mixamo/motions/180_Turn_W_Briefcase.npz'),
}
CONTACT_M = 0.01          # strict contact: signed height above ground < 1 cm (penetration counts as contact)
SLIP_MPS = [0.05, 0.10, 0.20]


def unit_to_m(name, height_units):
    # Mixamo export verified cm (Y Bot = 1.80 m). Objaverse: author units unknown; heuristic power of 10
    if name.startswith('mixamo'):
        return 0.01, 'verified: Mixamo FBX authored in cm, Y Bot rest height 180.5 units'
    for f in [1.0, 0.01, 0.001, 0.1, 10.0]:
        if 0.05 <= height_units * f <= 5.0:
            return f, f'ASSUMED: power-of-10 heuristic putting extent in [0.05,5] m (x{f})'
    return 1.0, 'ASSUMED: none fits'


summary = {}
for name, (g, e) in CASES.items():
    m = glb_skinned_mesh(g); d = np.load(e, allow_pickle=True)
    names = list(d['names']); G, Gr = export_fk(d)
    common = [n for n in m['names'] if n in names]
    pb = np.array([np.linalg.inv(m['ibm'][m['names'].index(n)])[:3, 3] for n in common])
    pe = np.array([Gr[names.index(n)][:3, 3] for n in common])
    sc, Rr, t = umeyama(pb, pe)
    bind_res = np.linalg.norm((sc * (Rr @ pb.T)).T + t - pe, axis=1)
    Ve = (sc * (Rr @ m['V'].T)).T + t
    # weights onto export joint order (skin joints absent from export carry ~0 weight, checked)
    We = np.zeros((len(Ve), len(names)))
    for k, n in enumerate(m['names']):
        if n in names: We[:, names.index(n)] += m['W'][:, k]
    lost = 1 - We.sum(1)
    We /= np.clip(We.sum(1, keepdims=True), 1e-9, None)
    P0, Wp = sample_surface(Ve, m['F'], We, n_cand=30000, n=4096)
    P = lbs(P0, Wp, G, Gr)                                    # (T,4096,3) export units, Y-up
    ext = np.ptp(Ve, 0); u2m, u_note = unit_to_m(name, ext.max())
    Pm = P * u2m; T = len(Pm)
    rest_min_y = Ve[:, 1].min() * u2m
    perframe_min = Pm[:, :, 1].min(1)
    # ground hypotheses: (A) authored floor y=0 of the export frame; (B) clip-lowest point (p2 of per-frame minima)
    gA = 0.0; gB = float(np.percentile(perframe_min, 2))
    res = {'T': T, 'V_mesh': int(len(m['V'])), 'faces': int(len(m['F'])), 'skin_joints': len(m['names']), 'export_joints': len(names),
           'export_skin_matrix_V': int(d['skin_matrix'].shape[0]), 'bind_vs_export_rest_residual_units_max': float(bind_res.max()),
           'bind_to_export_scale': float(sc), 'weight_mass_lost_max': float(lost.max()),
           'rest_extent_units': ext.round(3).tolist(), 'unit_to_m': u2m, 'unit_note': u_note,
           'rest_min_y_m': float(rest_min_y), 'perframe_min_y_m': [float(perframe_min.min()), float(np.median(perframe_min)), float(perframe_min.max())],
           'ground_B_m': gB}
    lab = {}
    for gname, g0 in [('A_authored_y0', gA), ('B_clip_lowest_p2', gB)]:
        h = Pm[:, :, 1] - g0                                   # signed height (T,N)
        con = h < CONTACT_M
        vel = np.zeros_like(Pm); vel[1:-1] = (Pm[2:] - Pm[:-2]) * FPS / 2; vel[0] = (Pm[1] - Pm[0]) * FPS; vel[-1] = (Pm[-1] - Pm[-2]) * FPS
        vt = np.linalg.norm(vel[..., [0, 2]], axis=-1)        # tangential speed (ground plane = XZ)
        fr_con = con.any(1)
        slip_frame = {}
        for s in SLIP_MPS:
            # frame slips if the median tangential speed of its contacting points exceeds s
            med = np.array([np.median(vt[i][con[i]]) if con[i].any() else 0.0 for i in range(T)])
            slip_frame[f'{s:.2f}'] = float((fr_con & (med > s)).sum() / max(fr_con.sum(), 1))
        pen = np.clip(-h, 0, None).max(1)
        lab[gname] = {'ground_y_m': g0, 'frames_with_contact': float(fr_con.mean()),
                      'contact_points_per_contact_frame_median': float(np.median(con.sum(1)[fr_con])) if fr_con.any() else 0,
                      'slip_frac_of_contact_frames': slip_frame,
                      'penetration_m_max': float(pen.max()), 'penetration_m_p95': float(np.percentile(pen, 95)),
                      'airborne_frames': float((~fr_con).mean())}
        if gname == 'A_authored_y0':
            keep = dict(con=con, vt=vt, h=h)
    res['labels'] = lab; summary[name] = res
    np.savez_compressed(f'{OUT}/{name}.npz', points=Pm.astype(np.float32), rest_points=(P0 * u2m).astype(np.float32),
                        weights=Wp.astype(np.float32), contact=keep['con'], tangential_speed=keep['vt'].astype(np.float32),
                        signed_height=keep['h'].astype(np.float32), unit_to_m=u2m)
    # 12-frame grid, side view (X horizontal, Y up) coloured by contact/slip, ground line
    idx = np.linspace(0, T - 1, 12).astype(int)
    fig, axs = plt.subplots(2, 6, figsize=(24, 8.5))
    lo = Pm.reshape(-1, 3).min(0); hi = Pm.reshape(-1, 3).max(0); pad = 0.05 * (hi - lo).max()
    for ax, i in zip(axs.ravel(), idx):
        c = np.full(len(Pm[i]), '#9aa4b1', dtype=object)
        con = keep['con'][i]; sl = con & (keep['vt'][i] > 0.10)
        c[con] = '#1f77b4'; c[sl] = '#d62728'
        order = np.argsort(con.astype(int) + sl.astype(int))
        ax.scatter(Pm[i, order, 0], Pm[i, order, 1], s=1.2, c=list(c[order]))
        ax.axhline(0.0, color='k', lw=0.8); ax.axhline(CONTACT_M, color='k', lw=0.4, ls=':')
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(min(lo[1], 0) - pad, hi[1] + pad); ax.set_aspect('equal')
        ax.set_title(f'f{i}  contact pts {con.sum()}  slip>{0.10}m/s {sl.sum()}', fontsize=9)
    fig.suptitle(f'{name}  ({u_note}; ground = authored y=0; blue = |h|<1cm contact, red = contact & tangential speed>0.10 m/s)', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/{name}_grid12.png', dpi=80); plt.close(fig)
    print(name, json.dumps(res, indent=None)[:1500])
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
