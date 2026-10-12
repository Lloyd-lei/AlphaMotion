"""GRAIL -> UMR GRAIL layout converter (Stream B).

raw/grail/data/<cat>/{recon,object_usd,meta}/<key>.*  ->  smplx/grail/<cat>/
    recon/<key>.pkl          slim, floor-calibrated, fps-fixed GRAIL recon (joblib, UMR load_grail_pickle compatible)
    object_usd/<key>.usd     hard link to the raw USD (UMR only checks existence for a GRAIL root)
    object_mjcf/<safe_name(key)>/   UMR asset_version 3 layout: grail_object.{obj,xml}, prop_grail_object.csv,
                             grail_object_collision_<i>.obj, metadata.json (written last, by grail_collision.py)
    object_mjcf/<..>/scene/  pickup_table only: static table box (not auto-discovered by UMR; opt-in)

Fixes vs. feeding the raw GRAIL files to UMR (all verified on data, see report):
  * fps: source pkl says mocap_frame_rate=30, true rate is 24 (README; frame counts 121/241/364 = 5/10/15 s at 24 fps;
    root track matches the 25 Hz robot track best at 24 fps).
  * object scale: pickup_table/pickup_ground USDs are already at final size (mesh bottom == table top to 0.1 mm);
    UMR's exporter multiplies by obj_scale (~0.23) again -> 4x too small. We write obj_scale_effective = 1 there.
    sitting needs obj_scale (0.8 puts chair legs on z=0); terrain has obj_scale=1.
  * floor: per-clip constant offset dz from stance feet (see floorcal.py); terrain cats shift the human only
    (terrain + plane are the support), object cats shift the whole scene rigidly (human-object relations preserved).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')

RAW = Path('/home/arenalabs/umr_factory_data/raw/grail/data')
TREE = Path('/home/arenalabs/umr_factory_data/raw/grail/_tree.json')
OUT = Path('/home/arenalabs/umr_factory_data/smplx/grail')
MESH_CACHE = OUT / '_meshes'
USD_INDEX = OUT / '_meshes' / '_usd_index'
CATS = ['pickup_ground', 'sitting', 'slope', 'curb', 'pickup_table', 'stair_p1', 'stair_p2']
TERRAIN = {'slope', 'curb', 'stair_p1', 'stair_p2'}
PRESCALED = {'pickup_table', 'pickup_ground'}  # USD already at final size; do not apply obj_scale again
FPS_TRUE = 24.0
CONVERTER_VERSION = 3  # v3: pickup_* human dz(t) + attached-object shift; v2: 3D support clearance, time-varying dz(t) (terrain+sitting), terrain xy registration, per-frame src mask
REG_RADIUS = 0.10      # max terrain xy registration offset (m)
REG_MIN_GAIN = 0.03    # apply the offset only if the valid-frame fraction improves by >= 3 points
LICENSE = 'CC BY-NC 4.0 (GRAIL motion/recon/meta); object USD: RoboCasa-derived CC BY 4.0 (pickup_*), Hunyuan3D-2.1 output CC BY-NC 4.0 (sitting, stairs), procedural CC BY-NC 4.0 (curb, slope, stairs)'
REPO = 'nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'


def usd_mesh(usd_path):
    """All UsdGeom.Mesh prims, local-to-world transformed (metres, z-up), fan-triangulated. Returns V[n,3], F[m,3]."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(usd_path))
    if st is None:
        raise ValueError(f'cannot open USD {usd_path}')
    xc = UsdGeom.XformCache()
    Vs, Fs, off = [], [], 0
    for pr in st.Traverse():
        if not pr.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(pr)
        pts = np.asarray(m.GetPointsAttr().Get(), np.float64).reshape(-1, 3)
        M = np.asarray(xc.GetLocalToWorldTransform(pr), np.float64)
        pts = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3]
        cnt = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int64)
        ind = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64)
        ntri = cnt - 2
        ok = ntri > 0
        offs = np.cumsum(cnt) - cnt
        cnt, ntri, offs = cnt[ok], ntri[ok], offs[ok]
        pid = np.repeat(np.arange(len(cnt)), ntri)
        k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
        base = offs[pid]
        tri = np.stack([ind[base], ind[base + k], ind[base + k + 1]], 1)
        Vs.append(pts)
        Fs.append(tri + off)
        off += len(pts)
    if not Vs:
        raise ValueError(f'no mesh in USD {usd_path}')
    return np.concatenate(Vs), np.concatenate(Fs)


def effective_scale(cat, x):
    s = np.asarray(x['obj_data'].get('obj_scale', np.ones(3)), np.float64).reshape(-1)
    s = np.repeat(s, 3) if s.size == 1 else s[:3]
    return np.ones(3) if cat in PRESCALED else s


def mesh_digest(V, F):
    h = hashlib.sha256()
    h.update(np.round(V, 6).astype(np.float64).tobytes())
    h.update(np.asarray(F, np.int64).tobytes())
    return h.hexdigest()[:20]


def canonical_mesh(usd_path, scale):
    """Processed (merged, outward-wound) visual mesh, cached by USD bytes + scale. Returns (digest, obj_path)."""
    import trimesh
    raw = Path(usd_path).read_bytes()
    ukey = hashlib.sha1(raw + np.round(scale, 8).tobytes()).hexdigest()
    idx = USD_INDEX / f'{ukey}.txt'
    if idx.exists():
        dg = idx.read_text().strip()
        p = MESH_CACHE / f'{dg}.obj'
        if p.exists():
            return dg, p
    V, F = usd_mesh(usd_path)
    V = V * scale[None]
    m = trimesh.Trimesh(V, F, process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m, multibody=True)
    Vr = np.round(np.asarray(m.vertices), 6)
    Fr = np.asarray(m.faces, np.int64)
    dg = mesh_digest(Vr, Fr)
    p = MESH_CACHE / f'{dg}.obj'
    if not p.exists():
        tmp = p.with_suffix(f'.tmp{os.getpid()}')
        lines = ['# GRAIL visual mesh (metres, object-local, z-up); canonicalised by umr_factory grail converter']
        lines += [f'v {a:.6f} {b:.6f} {c:.6f}' for a, b, c in Vr]
        lines += [f'f {a + 1} {b + 1} {c + 1}' for a, b, c in Fr]
        tmp.write_text('\n'.join(lines) + '\n')
        os.replace(tmp, p)
        (MESH_CACHE / f'{dg}.json').write_text(json.dumps({
            'vertices': int(len(Vr)), 'faces': int(len(Fr)), 'watertight': bool(m.is_watertight),
            'extent_m': np.round(m.extents, 4).tolist(), 'bounds_m': np.round(m.bounds, 4).tolist()}))
    tmpi = idx.with_suffix(f'.tmp{os.getpid()}')
    tmpi.write_text(dg + '\n')
    os.replace(tmpi, idx)
    return dg, p


def tree_sizes():
    t = json.loads(TREE.read_text())
    sizes = {}
    for k, v in t.items():
        if k.startswith('_'):
            continue
        for p, s in v['files']:
            sizes[p] = s
    return sizes


def raw_complete(cat, key, sizes):
    for sub, ext in (('recon', 'pkl'), ('object_usd', 'usd')):
        rel = f'data/{cat}/{sub}/{key}.{ext}'
        p = RAW / cat / sub / f'{key}.{ext}'
        if not p.exists() or p.stat().st_size != sizes.get(rel, -1) or Path(str(p) + '.aria2').exists():
            return False
    return True


def link(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


# ----------------------------------------------------------------------------------------------------------------
_BODY = None


def _body():
    global _BODY
    if _BODY is None:
        from g1smplx import G1SMPLX
        _BODY = G1SMPLX('cuda')
    return _BODY


def convert_clip(cat, key):
    import joblib
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as Rot
    from humanoid_retarget_pipeline_hsi_hoi import load_grail_pickle, safe_name
    from floorcal import calibrate, foot_tracks
    from floorcal2 import Scene, calibrate_v2, register_xy
    t0 = time.time()
    row = {'clip': key, 'category': cat, 'status': 'ok', 'flags': [], 'converter_version': CONVERTER_VERSION}
    rp = RAW / cat / 'recon' / f'{key}.pkl'
    up = RAW / cat / 'object_usd' / f'{key}.usd'
    x = load_grail_pickle(rp)
    h, o = x['human_data'], x['obj_data']
    poses = np.asarray(h['poses'], np.float32).reshape(-1, 165)
    T = len(poses)
    trans = np.asarray(h['trans'], np.float64).reshape(T, 3)
    R = np.asarray(o['obj_R'], np.float64).reshape(-1, 3, 3)
    t = np.asarray(o['obj_t'], np.float64).reshape(-1, 3)
    row.update(frames=T, fps=FPS_TRUE, seconds=round(T / FPS_TRUE, 4), fps_in_source_pkl=float(h.get('mocap_frame_rate', -1)))
    if len(R) != T or len(t) != T:
        row.update(status='dropped', reason=f'object track length {len(R)} != human {T}')
        return row
    if not (np.isfinite(poses).all() and np.isfinite(trans).all() and np.isfinite(R).all() and np.isfinite(t).all()):
        row.update(status='dropped', reason='non-finite values in recon')
        return row
    if T < 24:
        row.update(status='dropped', reason=f'too short ({T} frames)')
        return row
    if str(h.get('model', 'smplx')).lower() != 'smplx':
        row.update(status='dropped', reason=f"model {h.get('model')}")
        return row
    s_eff = effective_scale(cat, x)
    s_src = np.asarray(o.get('obj_scale', np.ones(3)), np.float64).reshape(-1)
    dg, objp = canonical_mesh(up, s_eff)
    mm = trimesh.load(objp, force='mesh', process=False)
    OV, OF = np.asarray(mm.vertices), np.asarray(mm.faces)
    row['object'] = {'name': key.split('__')[1] if '__' in key else key, 'mesh_digest': dg,
                     'mesh_vertices': int(len(OV)), 'mesh_faces': int(len(OF)),
                     'extent_m': np.round(OV.max(0) - OV.min(0), 4).tolist(),
                     'obj_scale_source': np.round(s_src, 5).tolist(), 'obj_scale_effective': np.round(s_eff, 5).tolist()}
    static_obj = bool(np.abs(R - R[:1]).max() < 1e-5 and np.abs(t - t[:1]).max() < 1e-5)
    row['object']['static'] = static_obj
    # ---------------- body on GPU
    import torch
    body = _body()
    v, j = body.verts(poses, trans)
    tracks = foot_tracks(body, v)
    hand_ids = np.where(np.isin(body.dom, [20, 21] + list(range(25, 55))))[0]
    hand = v[:, torch.as_tensor(hand_ids, device=v.device)].cpu().numpy().astype(np.float64)
    pel = j[:, 0].cpu().numpy().astype(np.float64)
    terrain = cat in TERRAIN
    OVw0 = OV @ R[0].T + t[0]  # object/terrain in world at frame 0 (static for terrain/sitting)
    if terrain and not static_obj:
        row['flags'].append('terrain_object_moves')
    if cat == 'sitting' and not static_obj:
        row['flags'].append('chair_moves')
    cp = np.asarray(h.get('foot_contact_probs'), np.float64) if h.get('foot_contact_probs') is not None else None
    upright = None
    if not terrain:  # pelvis within 85% of its upright height (top-20% median) -> standing phases only
        pz = pel[:, 2]
        up_h = float(np.median(np.sort(pz)[-max(1, T // 5):]))
        upright = pz >= 0.85 * up_h
    # v1 constant calibration (lowest stance foot vertex vs plane / terrain probe) kept for reference + pickup policy
    cal, gaps, stance = calibrate(tracks, FPS_TRUE, cp, trimesh.Trimesh(OVw0, OF, process=False) if terrain else None,
                                  upright=upright)
    dz_v1 = float(cal['dz_m'])
    xy_all = v[..., :2].reshape(-1, 2)
    xy_lo = xy_all.min(0).values.cpu().numpy().astype(np.float64)
    xy_hi = xy_all.max(0).values.cpu().numpy().astype(np.float64)
    off = np.zeros(2)
    reg = None
    table = None
    sd = x.get('scene_data')
    if isinstance(sd, dict) and isinstance(sd.get('table'), dict):
        table = {'pos': np.asarray(sd['table']['pos'], np.float64).tolist(), 'size': np.asarray(sd['table']['size'], np.float64).tolist()}
    if terrain:
        mw_raw = trimesh.Trimesh(OVw0, OF, process=False)
        off_b, s_b, s_0, _ = register_xy(body, v, FPS_TRUE, mw_raw, radius=REG_RADIUS)
        applied = bool(s_b - s_0 >= REG_MIN_GAIN)
        off = np.asarray(off_b, np.float64) if applied else np.zeros(2)
        reg = {'best_offset_m': np.round(off_b, 3).tolist(), 'valid_frac_hf_zero': round(s_0, 4),
               'valid_frac_hf_best': round(s_b, 4), 'applied': applied, 'offset_applied_m': np.round(off, 3).tolist(),
               'radius_m': REG_RADIUS, 'min_gain': REG_MIN_GAIN}
        scene = Scene(trimesh.Trimesh(OVw0 + np.r_[off, 0.0], OF, process=False), xy_lo, xy_hi)
        fc = calibrate_v2(body, v, FPS_TRUE, scene)
        dz_t = fc['dz'].astype(np.float64)
        policy = 'human_only_dz_t+terrain_xy_reg'
    elif cat == 'sitting':
        scene = Scene(trimesh.Trimesh(OVw0, OF, process=False) if static_obj else None, xy_lo, xy_hi)
        fc = calibrate_v2(body, v, FPS_TRUE, scene)
        dz_t = fc['dz'].astype(np.float64)
        policy = 'human_only_dz_t'
    else:  # pickup_* (v3): human dz(t) from floor anchors; object ATTACHED: shift = dz(clamp(t, t_grasp, t_release))
        # so the hand-object relation is preserved exactly while the object moves, the object stays at rest while
        # static, and the table (static) moves with the object's rest phase -> object stays on the table.
        fc0 = calibrate_v2(body, v, FPS_TRUE, Scene(None, xy_lo, xy_hi))
        dz_t = fc0['dz'].astype(np.float64)
        rot = np.degrees(np.arccos(np.clip((np.einsum('tij,ij->t', R, R[0]) - 1) / 2, -1, 1)))
        moving = (np.linalg.norm(t - t[:1], axis=1) > 0.002) | (rot > 1.0)
        if moving.any():
            tg, tr = int(np.argmax(moving)), int(T - 1 - np.argmax(moving[::-1]))
        else:
            tg = tr = int(min(max(int((x.get('meta') or {}).get('inter_start_idx', 0) or 0), 0), T - 1))
        obj_shift = dz_t[np.clip(np.arange(T), tg, tr)]
        row['object_motion_frames'] = [tg, tr] if moving.any() else None
        if table is not None:
            table['pos'][2] -= float(dz_t[tg])
            tb = trimesh.creation.box(extents=table['size'])
            tb.apply_translation(table['pos'])
        else:
            tb = None
        vs = v.clone()
        vs[..., 2] -= torch.as_tensor(dz_t, device=vs.device, dtype=vs.dtype)[:, None]
        fc = calibrate_v2(body, vs, FPS_TRUE, Scene(tb, xy_lo, xy_hi), fit=False)
        del vs
        fc['anchor'] = fc['anchor'] + dz_t.astype(np.float32)  # back to the unshifted frame (written as anchor - dz_t)
        for k_ in ('dz_median_mm', 'dz_p05_p95_mm', 'dz_range_mm', 'anchor_frames_used', 'anchor_residual_p05_p50_p95_mm'):
            fc['summary'][k_] = fc0['summary'][k_]
        fc['summary']['flags'] = fc0['summary']['flags']
        policy = 'human_dz_t+object_attached'
    del v, j
    torch.cuda.empty_cache()
    dz = float(np.median(dz_t))
    trans_n = trans.copy(); trans_n[:, 2] -= dz_t
    t_n = t.copy()
    if policy == 'human_dz_t+object_attached':
        t_n[:, 2] -= obj_shift
    if terrain:
        t_n[:, :2] += off[None]
    cal['dz_v1_const_m'] = dz_v1
    cal['dz_median_m'] = dz  # v2: median of the applied dz(t); per-frame values in umrf_frame['dz_m']
    cal['v2'] = fc['summary']
    if reg is not None:
        cal['xy_registration'] = reg
    # ---------------- QA after calibration (v2 per-frame source quality; see floorcal2.py)
    s2 = fc['summary']
    valid = np.asarray(fc['valid'], bool)
    qa = {
        'floor_dz_mm': round(dz * 1000, 2), 'floor_dz_v1_const_mm': round(dz_v1 * 1000, 2),
        'dz_t_range_mm': s2['dz_range_mm'], 'dz_t_p05_p95_mm': s2['dz_p05_p95_mm'],
        'anchor_frames': s2['anchor_frames'], 'anchor_residual_p05_p50_p95_mm': s2['anchor_residual_p05_p50_p95_mm'],
        'src_pen_gt1mm_frac': s2['pen_gt1mm_frac'], 'src_pen_gt10mm_frac': s2['pen_gt10mm_frac'],
        'src_pen_p95_max_mm': s2['pen_p95_max_mm'], 'src_lowest_gt20mm_frac': s2['lowest_gt20mm_frac'],
        'src_valid_frac': s2['valid_frac'], 'hand_scene_pen_gt10mm_frac': s2['hand_scene_pen_gt10mm_frac'],
    }
    # object resting gap (frame 0), both scale hypotheses for audit
    def obj_min_z(scale_ratio, f=0):
        return float(((OV * scale_ratio) @ R[f].T + t_n[f])[:, 2].min())
    alt_ratio = (s_src[:3] if s_src.size >= 3 else np.repeat(s_src, 3)) if cat in PRESCALED else 1.0 / np.where(s_eff == 0, 1, s_eff)
    if cat == 'pickup_table' and table is not None:
        top = table['pos'][2] + table['size'][2] / 2.0
        qa['object_bottom_minus_table_top_f0_mm'] = round((obj_min_z(1.0) - top) * 1000, 1)
        qa['object_bottom_minus_table_top_f0_mm_if_obj_scale_applied'] = round((obj_min_z(alt_ratio) - top) * 1000, 1)
    else:
        qa['object_min_z_f0_mm'] = round(obj_min_z(1.0) * 1000, 1)
        qa['object_min_z_f0_mm_alt_scale'] = round(obj_min_z(alt_ratio) * 1000, 1)
    # hand-object distance on interaction frames (object categories)
    m_ = x.get('meta') or {}
    i0, i1 = int(m_.get('inter_start_idx', 0) or 0), int(m_.get('inter_end_idx', T) or T)
    i0, i1 = max(0, min(i0, T - 1)), max(1, min(i1, T))
    row['interaction_frames'] = [i0, i1]
    if not terrain:
        surf = trimesh.Trimesh(OV, OF, process=False).sample(30000, seed=0) if hasattr(trimesh.Trimesh, 'sample') else OV
        kd = cKDTree(surf)
        fr = np.arange(i0, i1)
        if len(fr):
            hand_n = hand.copy(); hand_n[..., 2] -= dz_t[:, None]  # hands after the human shift
            loc = np.einsum('fnk,fkj->fnj', hand_n[fr] - (t_n[fr])[:, None, :], R[fr])
            d, _ = kd.query(loc.reshape(-1, 3))
            dmin = d.reshape(len(fr), -1).min(1)
            qa['hand_obj_dist_interaction_p10_p50_p90_mm'] = [round(float(np.percentile(dmin, q) * 1000), 1) for q in (10, 50, 90)]
            qa['hand_obj_within_20mm_frac'] = round(float((dmin < 0.02).mean()), 4)
    ev = x.get('eval_data')
    if isinstance(ev, dict):
        row['grail_eval'] = {k_: round(float(v_), 5) for k_, v_ in ev.items() if np.size(v_) == 1}
    row['qa'] = qa
    row['flags'] += cal['flags'] + s2['flags']
    if s2['valid_frac'] < 0.5:
        row['flags'].append('src_valid_lt_50pct')
    if s2['pen_gt10mm_frac'] > 0.2:
        row['flags'].append('src_pen_gt10mm_over_20pct_frames')
    if cat != 'pickup_table' and not json.loads((MESH_CACHE / f'{dg}.json').read_text()).get('watertight', True):
        row['flags'].append('scene_mesh_not_watertight')
    if cat == 'pickup_table' and table is not None and abs(qa['object_bottom_minus_table_top_f0_mm']) > 20:
        row['flags'].append('object_not_on_table_f0')
    if cat in ('sitting', 'pickup_ground') and qa['object_min_z_f0_mm'] > 20:
        row['flags'].append('object_floats_above_floor_f0')
    if 'hand_obj_within_20mm_frac' in qa and cat in PRESCALED and qa['hand_obj_within_20mm_frac'] < 0.5:
        row['flags'].append('hand_far_from_object_in_interaction')
    # ---------------- write outputs
    cdir = OUT / cat
    (cdir / 'recon').mkdir(parents=True, exist_ok=True)
    (cdir / 'object_usd').mkdir(parents=True, exist_ok=True)
    od = cdir / 'object_mjcf' / safe_name(key)
    od.mkdir(parents=True, exist_ok=True)
    umrf = {
        'converter': 'umr_factory_data/scripts_grail/convert_grail.py',
        'source': {'repo': REPO, 'revision': json.loads(TREE.read_text())['_sha'], 'recon': f'data/{cat}/recon/{key}.pkl',
                   'usd': f'data/{cat}/object_usd/{key}.usd', 'license': LICENSE},
        'fps_fix': {'source_mocap_frame_rate': row['fps_in_source_pkl'], 'used': FPS_TRUE},
        'floor': cal, 'shift_policy': policy,
        'obj_scale_source': s_src.tolist(), 'obj_scale_effective': s_eff.tolist(),
        'mesh_digest': dg, 'qa': qa, 'flags': row['flags'],
    }
    slim = {
        'human_data': {
            'poses': poses.astype(np.float32), 'trans': trans_n.astype(np.float32),
            'betas': np.asarray(h.get('betas', np.zeros((1, 10))), np.float32).reshape(1, -1),
            'gender': 'neutral', 'model': 'smplx', 'mocap_frame_rate': FPS_TRUE,
            'scale': float(np.asarray(h.get('scale', 1.0)).reshape(-1)[0]),
            'foot_contact_probs': None if cp is None else cp.astype(np.float32),
            'predicted_body_height': float(h.get('predicted_body_height', np.nan) or np.nan),
        },
        'obj_data': {'obj_R': R.astype(np.float32), 'obj_t': t_n.astype(np.float32),
                     'obj_scale': np.asarray(s_eff, np.float32).reshape(3, 1)},
        'meta': dict(m_) if isinstance(m_, dict) else {},
        'scene_data': {'table': {k_: np.asarray(v_, np.float32) for k_, v_ in table.items()}} if table else None,
        'object_path': x.get('object_path'),
        'umrf': umrf,
        'umrf_frame': {  # per-frame human-source quality AFTER calibration (metres), for frame exclusion downstream
            'dz_m': dz_t.astype(np.float32),  # applied vertical shift: trans_z_new = trans_z_src - dz_m
            'lowest_clearance_m': fc['lowest_clearance_after'],  # min over non-head vertices of clearance to support; <0 = penetration
            'anchor_clearance_m': (fc['anchor'] - dz_t).astype(np.float32),  # slow near-support vertices (contact proxy); NaN = none
            'src_valid': valid,  # penetration <= 10 mm and lowest point within 20 mm of a support
        },
    }
    tmp = cdir / 'recon' / f'.{key}.pkl.tmp'
    joblib.dump(slim, tmp, compress=('zlib', 3))
    os.replace(tmp, cdir / 'recon' / f'{key}.pkl')
    link(up, cdir / 'object_usd' / f'{key}.usd')
    link(objp, od / 'grail_object.obj')
    q = Rot.from_matrix(R).as_quat()  # xyzw (UMR prop csv convention)
    with open(od / 'prop_grail_object.csv', 'w') as fh:
        fh.write('px,py,pz,qx,qy,qz,qw\n')
        for p_, q_ in zip(t_n, q):
            fh.write(','.join(f'{v_:.9g}' for v_ in (*p_, *q_)) + '\n')
    if table is not None:
        sdir = od / 'scene'
        sdir.mkdir(exist_ok=True)
        (sdir / 'table.json').write_text(json.dumps({**table, 'note': 'static table box (full size, metres, z-up), '
                                                     'same frame as prop_grail_object.csv; not auto-discovered by UMR'}))
        hx, hy, hz = (np.asarray(table['size']) / 2).tolist()
        (sdir / 'table.xml').write_text(
            f'<mujoco model="grail_table"><worldbody><body name="grail_table" pos="{table["pos"][0]} {table["pos"][1]} {table["pos"][2]}">'
            f'<geom name="grail_table_box" type="box" size="{hx} {hy} {hz}" contype="1" conaffinity="1" rgba="0.5 0.5 0.5 1"/>'
            f'</body></worldbody></mujoco>\n')
    if (od / 'metadata.json').exists():  # re-conversion: keep collision metadata, refresh the umrf calibration fields
        try:
            md = json.loads((od / 'metadata.json').read_text())
            md.setdefault('umrf', {}).update({'floor_dz_m': round(dz, 5), 'shift_policy': policy,
                                              'converter_version': CONVERTER_VERSION,
                                              'terrain_xy_offset_m': np.round(off, 4).tolist()})
            tmpm = od / f'.metadata.json.tmp{os.getpid()}'
            tmpm.write_text(json.dumps(md, indent=2, sort_keys=True) + '\n')
            os.replace(tmpm, od / 'metadata.json')
        except Exception as e:  # noqa: BLE001
            row['flags'].append(f'metadata_patch_failed:{e!r}'[:120])
    row['hours'] = round(T / FPS_TRUE / 3600.0, 6)
    row['src_valid_frac'] = round(float(valid.mean()), 4)
    row['src_valid_hours'] = round(float(valid.sum()) / FPS_TRUE / 3600.0, 6)
    row['terrain_xy_offset_m'] = np.round(off, 4).tolist()
    if reg is not None:
        row['xy_registration'] = reg
    row['floor_dz_m'] = round(dz, 5)
    row['shift_policy'] = policy
    row['license'] = LICENSE
    row['paths'] = {'recon': str(cdir / 'recon' / f'{key}.pkl'), 'object_mjcf': str(od)}
    if table is not None:
        row['table'] = table
    row['sec'] = round(time.time() - t0, 3)
    return row


def _worker(args):
    cat, key = args
    try:
        return convert_clip(cat, key)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {'clip': key, 'category': cat, 'status': 'dropped', 'reason': f'converter error: {e!r}'[:400],
                'trace': traceback.format_exc()[-1500:]}


def _read_new_lines(path, state):
    """Incrementally read complete JSON lines appended to path since the last call (byte offset in state)."""
    out = []
    if not path.exists():
        return out
    with open(path, 'rb') as f:
        f.seek(state.get('pos', 0))
        data = f.read()
    if not data:
        return out
    cut = data.rfind(b'\n')
    if cut < 0:
        return out
    state['pos'] = state.get('pos', 0) + cut + 1
    for ln in data[:cut].split(b'\n'):
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def main():
    """v2 loop. Writes rows to _convert_rows_v2.jsonl. While the v1 loop (--v1-pid) is alive, a clip is (re)converted
    only after its v1 row exists, so v2 always overwrites v1 outputs (never the other way round); once v1 has exited,
    raw-complete clips are converted directly."""
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--cats', default=','.join(CATS))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--loop', action='store_true', help='keep polling for newly downloaded clips')
    ap.add_argument('--poll', type=float, default=60)
    ap.add_argument('--max-hours', type=float, default=10)
    ap.add_argument('--v1-pid', type=int, default=0, help='gating loop pid: while alive, only clips with a row in --gate-rows')
    ap.add_argument('--gate-rows', default='_convert_rows.jsonl')
    ap.add_argument('--rows-out', default='_convert_rows_v2.jsonl')
    ap.add_argument('--done-file', default='_converter_v2_done')
    a = ap.parse_args()
    MESH_CACHE.mkdir(parents=True, exist_ok=True)
    USD_INDEX.mkdir(parents=True, exist_ok=True)
    rows_v1, rows_v2 = OUT / a.gate_rows, OUT / a.rows_out
    done = set()
    for r in _read_new_lines(rows_v2, {}):
        if r.get('status') == 'ok' or 'converter error' not in r.get('reason', ''):
            done.add((r['category'], r['clip']))
    v1_state, v1_seen = {}, set()
    tree = json.loads(TREE.read_text())
    cats = a.cats.split(',')
    allkeys = {c: sorted(os.path.basename(p)[:-4] for p, _ in tree[f'{c}/recon']['files']) for c in cats}
    usdkeys = {c: {os.path.basename(p)[:-4] for p, _ in tree[f'{c}/object_usd']['files'] if p.endswith('.usd')} for c in cats}
    kidx = {c: {k: i for i, k in enumerate(allkeys[c])} for c in cats}
    t_end = time.time() + a.max_hours * 3600
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers) as pool, open(rows_v2, 'a') as fh:
        while True:
            for r in _read_new_lines(rows_v1, v1_state):
                v1_seen.add((r['category'], r['clip']))
            v1_alive = bool(a.v1_pid) and os.path.exists(f'/proc/{a.v1_pid}')
            sizes = tree_sizes()
            todo, pending = [], 0
            for c in cats:
                n = 0
                for k in allkeys[c]:
                    if (c, k) in done:
                        continue
                    if k not in usdkeys[c]:
                        r = {'clip': k, 'category': c, 'status': 'dropped', 'reason': 'no object USD in release',
                             'converter_version': CONVERTER_VERSION}
                        fh.write(json.dumps(r) + '\n'); done.add((c, k))
                        continue
                    if raw_complete(c, k, sizes) and ((c, k) in v1_seen or not v1_alive):
                        todo.append((c, k)); n += 1
                        if a.limit and n >= a.limit:
                            break
                    else:
                        pending += 1
            todo.sort(key=lambda ck: (kidx[ck[0]][ck[1]], cats.index(ck[0])))
            if todo:
                t0 = time.time()
                for i, r in enumerate(pool.imap_unordered(_worker, todo, chunksize=2)):
                    fh.write(json.dumps(r, default=float) + '\n'); fh.flush()
                    done.add((r['category'], r['clip']))
                    if (i + 1) % 200 == 0:
                        print(f'{time.strftime("%H:%M:%S")} v2 converted {i + 1}/{len(todo)} '
                              f'{(i + 1) / (time.time() - t0):.2f} clips/s', flush=True)
                    if time.time() > t_end:
                        break
                print(f'{time.strftime("%H:%M:%S")} v2 batch done {len(todo)} clips; pending {pending} (v1 alive {v1_alive})', flush=True)
            if not a.loop or a.limit or time.time() > t_end or (pending == 0 and not todo):
                break
            if not todo:
                time.sleep(a.poll)
    if a.loop and pending == 0:
        (OUT / a.done_file).write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    print('converter v2 exit', flush=True)


if __name__ == '__main__':
    main()
