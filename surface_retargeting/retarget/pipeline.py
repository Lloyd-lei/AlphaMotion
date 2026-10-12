"""Batch runner: UMR retarget (wrapped) -> strict post-pass -> labels -> npz + manifest row."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from surface_retargeting.strict import geometry as G, postpass as PP
from surface_retargeting.labels import labels as L
from surface_retargeting.robots import surface as S
from surface_retargeting.paths import CORR, ROBOTS, RUNS, OUT, LOGS, UMR_ROOT, UMR_PY, UMR_TOOLS, add_umr_to_path

DEFAULTS = {'smplx': UMR_ROOT / 'humanoid_retarget_defaults.json',
            'hoi': UMR_ROOT / 'humanoid_retarget_defaults_hsi_hoi_standard.json'}
PIPE_MODULE = {'smplx': 'humanoid_retarget_pipeline', 'hoi': 'humanoid_retarget_pipeline_hsi_hoi'}

# constraint switches we force on every run
SOLVER_OVERRIDES = {
    'ground_penetration_hard_constraint': True,
    'robot_self_penetration_hard_constraint': True,       # stock UMR crashes; umr_run.py QP fallback handles it
    'robot_self_penetration_hard_slack': True,
    'robot_self_penetration_hard_slack_cost': 1000.0,
    'robot_self_penetration_margin': -0.001,
}
LATCH_PIN = 0.0
HOI_OVERRIDES = {'robot_object_hard_constraint': True}


def _absolutize(obj, base: Path):
    if isinstance(obj, dict):
        return {k: _absolutize(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_absolutize(v, base) for v in obj]
    if isinstance(obj, str) and obj and not obj.startswith('/') and '/' in obj or obj in ('smpl',):
        p = base / obj
        if p.exists():
            return str(p.resolve())
    return obj


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def clip_id(kind, data, seq_key):
    return f"{Path(str(data)).stem if kind == 'smplx' else Path(str(data)).name}__{seq_key}"


def run_umr(robot, kind, data, seq_key, timeout=3600, force=False, stage='all', shared_corr=False, start=None,
            end=None, stride=None, cid=None):
    """Run UMR (build/train correspondence if needed, retarget) with our overrides. Returns paths dict.
    shared_corr (HOI only): reuse the robot's beta-zero correspondence across body shapes (UMR's HiPHI mechanism,
    see umrf/shared_corr.py) instead of training one per sequence.  start/end/stride: UMR frame window (source fps
    / stride = output fps).  cid: clip id override (pilot uses '<source>__<seq>__f<start>-<end>s<stride>')."""
    cid = cid or clip_id(kind, data, seq_key)
    rdir = RUNS / cid; rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / f'{robot}.npz'; cap = rdir / f'{robot}.src_slots.npz'; stats = rdir / f'{robot}.qpstats.json'
    log = rdir / f'{robot}.umr.log'
    res = dict(umr_out=str(out), src_slots=str(cap), qp_stats=str(stats), log=str(log), clip=cid,
               shared_corr=bool(kind == 'hoi' and shared_corr))
    if stage == 'all' and out.exists() and cap.exists() and not force:
        res['cached'] = True
        return res
    dflt = json.load(open(DEFAULTS[kind]))
    dflt = _absolutize(dflt, UMR_ROOT)
    if kind == 'hoi' and shared_corr:
        # HiPHI-style defaults: standard HSI/HOI solver + HiPHI non-interaction overrides; no streaming chunks so the
        # captured source slots cover the whole window (windows are short)
        hip = _absolutize(json.load(open(UMR_ROOT / 'humanoid_retarget_defaults_hiphi.json')), UMR_ROOT)
        hip.pop('extends', None); hip.pop('correspondence', None); hip.pop('motion', None)
        dflt = _deep_merge(dflt, hip)
        dflt.setdefault('retarget', {})['stream_chunk_frames'] = 0
    dflt.setdefault('solver', {}).update(SOLVER_OVERRIDES)
    if kind == 'hoi':
        dflt['solver'].update(HOI_OVERRIDES)
    dflt.setdefault('view', {})['enabled'] = False
    if kind == 'smplx':
        dflt['motion'].update({'data': str(data), 'seq_key': seq_key})
        dflt['retarget']['out'] = str(out)
    dpath = rdir / f'{robot}.defaults.json'
    json.dump(dflt, open(dpath, 'w'), indent=1)
    rcfg = json.load(open(ROBOTS / robot / 'umr_robot.json'))
    if kind == 'hoi' and shared_corr:
        from surface_retargeting.correspondence import shared_corr as SC
        if not SC.paths(robot)['slots'].exists():
            raise RuntimeError(f'shared beta0 correspondence missing for {robot}; run scripts/build_shared_corr.py')
        rcfg['correspondence'] = _deep_merge(rcfg['correspondence'], SC.runtime_correspondence_section(robot))
        res['shared_corr'] = True
        cpath = rdir / f'{robot}.robot.json'
        json.dump(rcfg, open(cpath, 'w'), indent=1)
    tag = hashlib.sha1(f'{kind}:{data}:{seq_key}'.encode()).hexdigest()[:10] if kind == 'hoi' else 'smplx'
    cdir = CORR / robot / tag
    if not res.get('shared_corr'):
        cdir.mkdir(parents=True, exist_ok=True)
        rcfg['correspondence'].setdefault('dataset', {})['out'] = str(cdir / 'correspondence_dataset.npz')
        rcfg['correspondence'].setdefault('train', {})['out_dir'] = str(cdir / 'train')
        cpath = rdir / f'{robot}.robot.json'
        json.dump(rcfg, open(cpath, 'w'), indent=1)
    cmd = ['timeout', str(timeout), UMR_PY, str(UMR_TOOLS / 'umr_pipe.py'), PIPE_MODULE[kind], '--config', str(cpath),
           '--defaults', str(dpath), '--skip-view', '--stage', stage]
    if stage == 'all':
        cmd.append('--force-retarget')
    if kind == 'hoi':
        cmd += ['--data', str(data), '--seq-key', seq_key, '--out', str(out),
                '--work-dir', str(rdir / f'{robot}.work')]                    # not /tmp
        for k, v in (('--start', start), ('--end', end), ('--stride', stride)):
            if v is not None:
                cmd += [k, str(int(v))]
    env = dict(os.environ, UMR_CAPTURE=str(cap), UMR_QP_FALLBACK='1', UMR_STATS=str(stats), MUJOCO_GL='egl')
    t0 = time.time()
    with open(log, 'w') as lf:
        p = subprocess.run(cmd, cwd=str(UMR_ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
    res['umr_seconds'] = round(time.time() - t0, 1); res['returncode'] = p.returncode
    import shutil
    shutil.rmtree(rdir / f'{robot}.work', ignore_errors=True)
    if p.returncode != 0 or (stage == 'all' and not out.exists()):
        raise RuntimeError(f'UMR failed rc={p.returncode}, see {log}')
    return res


# --------------------------------------------------------------------------------------------- human regions
def correspondence_files(robot, kind, data, seq_key, log=None, shared=False):
    """(dataset npz, slots npz) actually used by the UMR run: parsed from its log (the HOI pipeline ignores our
    correspondence out paths and uses UMR/data + UMR/output), else our CORR layout."""
    import re
    if shared:
        from surface_retargeting.correspondence import shared_corr as SC
        sp = SC.paths(robot)
        return sp['dataset'], sp['slots']
    if log and Path(log).exists():
        txt = Path(log).read_text(errors='replace')
        dsm = re.findall(r'correspondence dataset: (\S+\.npz)', txt)
        slm = re.findall(r'correspondence slots: (\S+\.npz)', txt)
        if dsm and slm:
            return Path(dsm[-1]), Path(slm[-1])
    tag = hashlib.sha1(f'{kind}:{data}:{seq_key}'.encode()).hexdigest()[:10] if kind == 'hoi' else 'smplx'
    cdir = CORR / robot / tag
    return cdir / 'correspondence_dataset.npz', cdir / 'train' / 'correspondence_slots_final.npz'


def human_slot_regions(robot, kind, data, seq_key, log=None, shared=False):
    """Region per human slot: nearest correspondence target point -> SMPL-X face -> segm part -> region."""
    import pickle
    from scipy.spatial import cKDTree
    dsp, slp = correspondence_files(robot, kind, data, seq_key, log, shared)
    ds = np.load(dsp, allow_pickle=True)
    sl = np.load(slp, allow_pickle=True)
    segm = pickle.load(open(UMR_ROOT / 'assets' / 'smplx_parts_segm.pkl', 'rb'), encoding='latin1')['segm']
    face = ds['sample_face_ids_0']
    pts = ds['points'][0]
    tree = cKDTree(pts)
    dist, nn = tree.query(sl['reconstructed_slots'][0])
    part = segm[face[nn]]
    reg = np.array([S.RID[S.SMPLX_PART_REGION.get(int(p), 'trunk')] for p in part], np.int64)
    return reg, dict(nn_dist_mm_p95=float(np.percentile(dist, 95) * 1000), target_points_match=bool(
        np.allclose(sl['target_points'][0], pts, atol=1e-5)))


# --------------------------------------------------------------------------------------------- measuring
def measure_all(geo, Q):
    T = len(Q)
    P = geo.ndense
    X = np.zeros((T, P, 3), np.float32)
    gmin = np.zeros(T); omin = np.full(T, np.inf); smin = np.full(T, np.inf)
    gvis = np.zeros(T); gcol = np.full(T, np.inf); ovis = np.full(T, np.inf); ocol = np.full(T, np.inf)
    greg = np.zeros((T, len(S.REGIONS))); oreg = np.full((T, len(S.REGIONS)), np.inf)
    greg_vis = np.zeros((T, len(S.REGIONS))); oreg_vis = np.full((T, len(S.REGIONS)), np.inf)
    self_pairs = {}
    for t in range(T):
        mm = geo.measure(Q[t], t)
        X[t] = mm['X'][:P]
        gmin[t] = mm['ground_min']; greg[t] = mm['ground_reg_min']
        omin[t] = mm['obj_min']; oreg[t] = mm['obj_reg_min']; smin[t] = mm['self_min']
        gvis[t] = mm['ground_min_visual']; gcol[t] = mm['ground_min_coll']
        ovis[t] = mm['obj_min_visual']; ocol[t] = mm['obj_min_coll']
        greg_vis[t] = mm['ground_reg_min_visual']; oreg_vis[t] = mm['obj_reg_min_visual']
        for g1, g2, dd, _ in mm['self_pairs']:
            if dd < -PP.PEN_TOL:
                b = (int(geo.m.geom_bodyid[g1]), int(geo.m.geom_bodyid[g2]))
                self_pairs[b] = self_pairs.get(b, 0) + 1
    return dict(X=X, ground_min=gmin, ground_reg=greg, obj_min=omin, obj_reg=oreg, self_min=smin, self_pairs=self_pairs,
                ground_min_visual=gvis, ground_min_coll=gcol, obj_min_visual=ovis, obj_min_coll=ocol,
                ground_reg_visual=greg_vis, obj_reg_visual=oreg_vis)


def pen_stats(meas, has_obj):
    def st(v):
        p = np.maximum(0.0, -v[np.isfinite(v)])
        if not len(p):
            return None
        return dict(max_mm=round(float(p.max() * 1000), 3), p95_mm=round(float(np.percentile(p, 95) * 1000), 3),
                    frames_gt_0p5mm=int((p > PP.PEN_TOL).sum()), frames_gt_1cm=int((p > 0.01).sum()), frames=int(len(v)))
    out = dict(ground=st(meas['ground_min']), self=st(meas['self_min']),
               ground_visual=st(meas['ground_min_visual']), ground_collision=st(meas['ground_min_coll']))
    if has_obj:
        out['object'] = st(meas['obj_min'])
        out['object_visual_points'] = st(meas['obj_min_visual']); out['object_collision_geoms'] = st(meas['obj_min_coll'])
    return out


# --------------------------------------------------------------------------------------------- post-process
def postprocess(robot, kind, data, seq_key, umr_paths, inv, log_every=500):
    from surface_retargeting.strict.objects import load_umr_hoi_object
    t_all = time.time()
    R = np.load(umr_paths['umr_out'], allow_pickle=True)
    Q0 = R['qpos'].astype(np.float64)
    fps = float(R['fps'][0]); T = len(Q0)
    st = json.load(open(ROBOTS / robot / 'staging.json'))
    obj = None; obj_info = {}
    if kind == 'hoi':
        from surface_retargeting.strict.objects import load_object_strict
        up = str(R['noitom_output_up']) if 'noitom_output_up' in R.files else 'y'
        obj, obj_info = load_object_strict(str(R['source_object_dir']), R['frame_ids'], float(R['smpl_scale'][0]), up)
        obj_info['kind'] = obj.kind
    m = G.build_model(st['staged_mjcf'], obj)              # object appended as a mocap body (robot ids unchanged)
    assert m.nq == Q0.shape[1], (m.nq, Q0.shape)
    surf = S.load_surface(robot)
    NR = len(S.REGIONS); FEET = [S.RID['L_foot'], S.RID['R_foot']]
    # human slots
    Hs = np.load(umr_paths['src_slots'])['source_slots']
    if len(Hs) != T:
        Hs = Hs[R['frame_ids']]
    hreg, hinfo = human_slot_regions(robot, kind, data, seq_key, umr_paths.get('log'), umr_paths.get('shared_corr', False))
    # ---------------- floor calibration (session = clip here; human source is the reference)
    floor_h, n_h, how_h = L.floor_from_points(Hs, hreg, FEET, fps)
    geo = G.RobotGeometry(m, surf, floor=floor_h, obj=obj)
    t0 = time.time()
    before = measure_all(geo, Q0)
    t_meas = time.time() - t0
    floor_r, n_r, how_r = L.floor_from_points(before['X'], surf['region'].astype(int), FEET, fps)
    # ---------------- labels BEFORE (robot = UMR output) and HUMAN
    rreg = surf['region'].astype(np.int64)
    dg, vg, _ = L.ground_inputs(before['X'], floor_h, rreg, fps)
    lab_r0 = L.label(dg, vg, rreg, before['ground_reg'])
    dh, vh, ch = L.ground_inputs(Hs, floor_h, hreg, fps)
    lab_h = L.label(dh, vh, hreg, ch)
    lab_ro0 = lab_ho = None
    if obj is not None:
        do_, vo_, _ = L.object_inputs(before['X'], obj, rreg, fps)
        lab_ro0 = L.label(do_, vo_, rreg, before['obj_reg'])
        dho, vho, cho = L.object_inputs(Hs, obj, hreg, fps)
        lab_ho = L.label(dho, vho, hreg, cho)
    # ---------------- required contacts: robot label OR (human label AND robot region within reach)
    # human-driven additions only where the robot region is near (< 3 cm) AND not moving tangentially (< slip speed):
    # snapping a sliding robot foot onto the floor would manufacture slip.
    rsp_g = L.region_support_speed(dg, vg, rreg)
    # band trimming applies to the robot's own hysteresis runs only; human-driven frames are touching frames
    req_g = L.trim_band_ends(lab_r0['contact'], before['ground_reg'], L.CFG['enter'], 1) | \
        (lab_h['contact'] & (before['ground_reg'] < 0.03) & (rsp_g < L.CFG['slip_speed']))
    req_g = L.drop_short(req_g, L.CFG['min_on'])
    req_o = np.zeros((T, NR), bool)
    if obj is not None:
        rsp_o = L.region_support_speed(do_, vo_, rreg)
        # objects: no speed gate -- if the human touches the object and the robot region is within 5 cm, the robot
        # must touch it too (UMR hovers 1-2 cm off objects); any resulting relative motion is labelled as slip.
        req_o = L.trim_band_ends(lab_ro0['contact'], before['obj_reg'], L.CFG['enter'], 1) | \
            (lab_ho['contact'] & (before['obj_reg'] < 0.05))
        req_o = L.drop_short(req_o, L.CFG['min_on'])
    # never require a region that has no points (e.g. hand-less robot): nothing to snap
    has_pts = np.array([(rreg == r).any() for r in range(NR)])
    req_g &= has_pts[None]; req_o &= has_pts[None]
    latch = req_g                                          # every required stance region is latched ...
    # s=0: support points keep UMR's own tangential velocity (the projection adds no slip, removes none).
    # s=1 (pin) was tried on dance1: it removes slip locally but over the full clip it raised slip distance
    # by 30-60 % through release spikes, so pinning is opt-in (LATCH_PIN) and only where the human does not slip.
    lam = LATCH_PIN * (~lab_h['slip']).astype(float)
    # ---------------- projection
    proj = PP.Projector(geo)
    Q1, flags, flags0, pinfo = PP.run_projection(proj, Q0, req_g, req_o, latch_regions=latch, latch_strength=lam, smooth_sigma=0.067 * fps, fps=fps,
                                                 log_every=log_every)
    after = measure_all(geo, Q1)
    dg1, vg1, _ = L.ground_inputs(after['X'], floor_h, rreg, fps)
    lab_r1 = L.label(dg1, vg1, rreg, after['ground_reg'], contact=req_g)        # output label = enforced set
    lab_r1_hyst = L.label(dg1, vg1, rreg, after['ground_reg'])                 # plain hysteresis on the output
    lab_ro1 = None
    if obj is not None:
        do1, vo1, _ = L.object_inputs(after['X'], obj, rreg, fps)
        lab_ro1 = L.label(do1, vo1, rreg, after['obj_reg'], contact=req_o)
        lab_ro1_hyst = L.label(do1, vo1, rreg, after['obj_reg'])
    # ---------------- strict verification of the OUTPUT (independent of the solver's own flags)
    final_flags = np.zeros(T, np.int32)
    final_flags |= np.where(after['ground_min'] < -PP.PEN_TOL, PP.FLAG_GROUND_PEN, 0).astype(np.int32)
    final_flags |= np.where(after['self_min'] < -PP.PEN_TOL, PP.FLAG_SELF_PEN, 0).astype(np.int32)
    if obj is not None:
        final_flags |= np.where(after['obj_min'] < -PP.PEN_TOL, PP.FLAG_OBJ_PEN, 0).astype(np.int32)
    gbad = req_g & ~((after['ground_reg'] >= -PP.PEN_TOL) & (after['ground_reg'] <= PP.CONTACT_TOL))
    final_flags |= np.where(gbad.any(1), PP.FLAG_GROUND_CONTACT, 0).astype(np.int32)
    if obj is not None:
        obad = req_o & ~((after['obj_reg'] >= -PP.PEN_TOL) & (after['obj_reg'] <= PP.CONTACT_TOL))
        final_flags |= np.where(obad.any(1), PP.FLAG_OBJ_CONTACT, 0).astype(np.int32)
    # ---------------- shift so the calibrated floor is z = 0
    Qout = Q1.copy()
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        Qout[:, 2] -= floor_h
    names = S.REGIONS

    def agree(a, b):
        return {names[r]: round(float((a[:, r] == b[:, r]).mean()), 4) for r in range(NR) if a[:, r].any() or b[:, r].any()}
    qa = dict(
        clip=umr_paths['clip'], robot=robot, frames=T, fps=fps,
        floor=dict(human_floor_m=floor_h, human_static_samples=n_h, human_method=how_h, robot_umr_floor_m=floor_r,
                   robot_static_samples=n_r, applied_shift_m=-floor_h),
        human_region_map=hinfo, object=obj_info,
        penetration_before=pen_stats(before, obj is not None), penetration_after=pen_stats(after, obj is not None),
        projection=pinfo, frames_flagged_final=int((final_flags > 0).sum()),
        flags_final_hist={str(b): int(((final_flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16)},
        required_ground_region_frames=int(req_g.sum()), required_object_region_frames=int(req_o.sum()),
        ground_slip_before=L.slip_summary(lab_r0, names, fps), ground_slip_after=L.slip_summary(lab_r1, names, fps),
        ground_slip_human=L.slip_summary(lab_h, names, fps),
        ground_contact_agreement_human_vs_robot_before=agree(lab_h['contact'], lab_r0['contact']),
        ground_contact_agreement_human_vs_robot_after=agree(lab_h['contact'], lab_r1['contact']),
        ground_slip_after_hysteresis_label=L.slip_summary(lab_r1_hyst, names, fps),
        ground_label_consistency=dict(
            hyst_contact_not_enforced_region_frames=int((lab_r1_hyst['contact'] & ~req_g).sum()),
            enforced_not_hyst_region_frames=int((req_g & ~lab_r1_hyst['contact']).sum()),
            nonrequired_touching_le_1mm_region_frames=int((~req_g & (after['ground_reg'] <= PP.CONTACT_TOL)).sum())),
        top_self_pen_pairs_before=sorted([(R_bn(m, k), v) for k, v in before['self_pairs'].items()], key=lambda x: -x[1])[:5],
        top_self_pen_pairs_after=sorted([(R_bn(m, k), v) for k, v in after['self_pairs'].items()], key=lambda x: -x[1])[:5],
        seconds=dict(measure_before=round(t_meas, 1), projection=round(pinfo['seconds_total'], 1),
                     total_postprocess=round(time.time() - t_all, 1), umr=umr_paths.get('umr_seconds')),
    )
    if obj is not None:
        qa.update(object_slip_before=L.slip_summary(lab_ro0, names, fps), object_slip_after=L.slip_summary(lab_ro1, names, fps),
                  object_slip_human=L.slip_summary(lab_ho, names, fps),
                  object_contact_agreement_human_vs_robot_before=agree(lab_ho['contact'], lab_ro0['contact']),
                  object_contact_agreement_human_vs_robot_after=agree(lab_ho['contact'], lab_ro1['contact']),
                  object_contact_frames=dict(human={names[r]: int(lab_ho['contact'][:, r].sum()) for r in range(NR) if lab_ho['contact'][:, r].any()},
                                             robot_before={names[r]: int(lab_ro0['contact'][:, r].sum()) for r in range(NR) if lab_ro0['contact'][:, r].any()},
                                             robot_after={names[r]: int(lab_ro1['contact'][:, r].sum()) for r in range(NR) if lab_ro1['contact'][:, r].any()}))
    # ---------------- write npz
    od = OUT / umr_paths['clip']; od.mkdir(parents=True, exist_ok=True)
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    arr = dict(
        qpos=Qout.astype(np.float32), qpos_umr=(Q0 - np.eye(Q0.shape[1])[2] * floor_h).astype(np.float32),
        root_pos=Qout[:, :3].astype(np.float32), root_quat_wxyz=Qout[:, 3:7].astype(np.float32),
        fps=np.float32(fps), floor_height=np.float32(0.0), floor_offset_removed=np.float32(floor_h),
        joint_names=np.array(jn), regions=np.array(names), robot_xml=np.array(st['staged_mjcf']),
        ground_contact=lab_r1['contact'], ground_clearance=after['ground_reg'].astype(np.float32),
        ground_clearance_visual=after['ground_reg_visual'].astype(np.float32),
        ground_min_visual_m=after['ground_min_visual'].astype(np.float32),
        ground_min_collision_m=np.where(np.isfinite(after['ground_min_coll']), after['ground_min_coll'], 9.9).astype(np.float32),
        ground_slip=lab_r1['slip'], ground_slip_speed=lab_r1['slip_speed'].astype(np.float32),
        ground_contact_required=req_g, ground_point_contact=np.packbits(lab_r1['point_contact'], axis=1),
        n_points=np.int32(rreg.size),
        pen_ground_m=np.maximum(0, -after['ground_min']).astype(np.float32),
        pen_self_m=np.maximum(0, -np.where(np.isfinite(after['self_min']), after['self_min'], 1)).astype(np.float32),
        flags=final_flags, flags_initial=flags0,
        human_ground_contact=lab_h['contact'], human_ground_slip=lab_h['slip'],
        human_ground_clearance=lab_h['clearance'].astype(np.float32),
        human_scale=np.float32(float(R['smpl_scale'][0])),
    )
    if obj is not None:
        arr.update(object_contact=lab_ro1['contact'], object_clearance=np.where(np.isfinite(after['obj_reg']), after['obj_reg'], 9.9).astype(np.float32),
                   object_slip=lab_ro1['slip'], object_slip_speed=lab_ro1['slip_speed'].astype(np.float32),
                   object_contact_required=req_o, object_point_contact=np.packbits(lab_ro1['point_contact'], axis=1),
                   object_clearance_visual=np.where(np.isfinite(after['obj_reg_visual']), after['obj_reg_visual'], 9.9).astype(np.float32),
                   object_min_visual_m=np.where(np.isfinite(after['obj_min_visual']), after['obj_min_visual'], 9.9).astype(np.float32),
                   object_min_collision_m=np.where(np.isfinite(after['obj_min_coll']), after['obj_min_coll'], 9.9).astype(np.float32),
                   pen_object_m=np.maximum(0, -np.where(np.isfinite(after['obj_min']), after['obj_min'], 1)).astype(np.float32),
                   object_pos=(obj.pos - np.array([0, 0, floor_h])).astype(np.float32), object_rot=obj.rot.astype(np.float32),
                   object_name=np.array(obj.name), object_xml=np.array(obj_info.get('xml', '')),
                   human_object_contact=lab_ho['contact'], human_object_slip=lab_ho['slip'])
    prov = dict(source_kind=kind, source_data=str(data), seq_key=seq_key, umr_commit='d1cdaed', umr_out=umr_paths['umr_out'],
                human_slots=umr_paths['src_slots'], staged_mjcf=st['staged_mjcf'], exclusions=st['exclusions'],
                solver_overrides=SOLVER_OVERRIDES, label_cfg=L.CFG, pen_tol_m=PP.PEN_TOL, contact_tol_m=PP.CONTACT_TOL,
                code='/home/arenalabs/umr_factory_code', created=time.strftime('%Y-%m-%d %H:%M:%S'))
    arr['provenance'] = np.array(json.dumps(prov))
    npz = od / f'{robot}.npz'
    np.savez_compressed(npz, **arr)
    qa['npz'] = str(npz)
    json.dump(qa, open(od / f'{robot}.qa.json', 'w'), indent=1, default=_js)
    return qa


def R_bn(m, k):
    return '|'.join(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b) for b in k)


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
