"""Capture wrapper around UMR's per-frame retarget script (upstream code untouched, runs in the UMR env).

Usage (normally injected by umr_capture_pipe.py):
    umr_capture_run.py <UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py> [args...]

It imports the upstream script as a module, replaces ``solve_frame_body_segment_qp`` by a recorder and calls the
upstream ``main()``.  Every per-frame call is recorded *exactly as UMR passes it* (source slots, transported normal
targets, ground-contact distances, self-contact maps, object-contact rows, warm start, iteration count, ...), plus the
static robot data (compiled MjModel, robot slot template, costs, joint limits, solver args).

Env:
  UMRG_CAPTURE=/path/problem.npz   output problem file (required)
  UMRG_MODE=cpu|skip               cpu (default): also run the original Clarabel solve and record its qpos + timing;
                                   skip: return the warm start (problem capture only, no CPU solve)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

UMR = Path('/var/tmp/alphamotion/umr_factory/umr/UMR')
sys.path.insert(0, str(UMR / 'scripts'))
sys.path.insert(0, str(UMR))

import mujoco  # noqa: E402

script = Path(sys.argv[1])
modname = script.stem
import importlib  # noqa: E402

M = importlib.import_module(modname)
ORIG = M.solve_frame_body_segment_qp
SIG = inspect.signature(ORIG)
OUT = Path(os.environ['UMRG_CAPTURE'])
MODE = os.environ.get('UMRG_MODE', 'cpu')

REC = dict(frames=[], static=None, t_solve=[], t_start=time.time())


def _args_dict(args):
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
            out[k] = list(v)
        elif isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, dict) and k == 'dof_max_dq_box_by_dof':
            out[k] = {str(kk): float(vv['max_dq']) for kk, vv in v.items()}
    return out


def _static(b):
    model = b['model']
    mjb = OUT.with_suffix('.mjb')
    mujoco.mj_saveModel(model, str(mjb), None)
    args = b['args']
    jl = b['joint_limits_by_qpos'] or {}
    lim_lo = np.full(model.nq, -np.inf); lim_hi = np.full(model.nq, np.inf)
    for qadr, info in jl.items():
        lim_lo[int(qadr)] = float(info['lower']); lim_hi[int(qadr)] = float(info['upper'])
    st = dict(
        mjb=str(mjb), robot_xml=str(getattr(args, 'robot_xml', '')),
        tmpl_geom=np.asarray(b['robot_template']['geom_ids'], np.int32),
        tmpl_local_pos=np.asarray(b['robot_template']['local_pos'], np.float32),
        tmpl_local_normal=np.asarray(b['robot_template']['local_normals'], np.float32),
        selected=np.asarray(b['selected_slot_ids'], np.int32),
        part_ids=np.asarray(b['source_slot_part_ids'], np.int32),
        point_cost=np.asarray(b['surface_point_slot_costs'], np.float64),
        normal_cost=np.asarray(b['surface_normal_slot_costs'], np.float64),
        joint_qpos_addrs=np.asarray(b['joint_qpos_addrs'], np.int32),
        joint_dof_addrs=np.asarray(b['joint_dof_addrs'], np.int32),
        lim_lo=lim_lo, lim_hi=lim_hi,
        args_json=json.dumps(_args_dict(args)),
    )
    oc = b['robot_object_penetration_cache']
    if oc is not None:
        pm = oc['model']
        pmjb = OUT.with_name(OUT.stem + '.objpen.mjb')
        mujoco.mj_saveModel(pm, str(pmjb), None)
        st.update(objpen_mjb=str(pmjb), objpen_object_qadr=int(oc['object_qadr']),
                  objpen_robot_geoms=np.asarray(oc['robot_geom_ids'], np.int32),
                  objpen_object_geoms=np.asarray(oc['object_geom_ids'], np.int32))
    return st


def recorder(*a, **kw):
    b = SIG.bind(*a, **kw)
    b.apply_defaults()
    b = b.arguments
    args = b['args']
    if REC['static'] is None:
        REC['static'] = _static(b)
    sel = np.asarray(b['selected_slot_ids'], np.int32)
    fr = {}
    fr['q_init'] = np.asarray(b['qpos_init'], np.float64).copy()
    fr['has_prev'] = b['qpos_prev'] is not None
    fr['has_prev2'] = b['qpos_prev2'] is not None
    fr['iters'] = int(args.iters) if b['iters'] is None else int(b['iters'])
    fr['src'] = np.asarray(b['source_slots'], np.float32).copy()
    nt = b['source_surface_normal_targets']
    if nt is None:
        nt = b['source_slot_normals']
    fr['ntgt'] = np.asarray(nt, np.float32)[sel].copy()
    gd = b['source_ground_contact_distances']
    if gd is not None:
        gw = b['source_ground_contact_weight_distances']
        gw = gd if gw is None else gw
        fr['gd'] = np.asarray(gd, np.float64)[sel].copy(); fr['gw'] = np.asarray(gw, np.float64)[sel].copy()
    sc = b['source_self_contact_map']
    if sc is not None and float(args.self_contact_map_cost) > 0:
        fr['sc_pairs'] = np.asarray(sc['slot_pairs'], np.int32).reshape(-1, 2).copy()
        fr['sc_d'] = np.asarray(sc['distances'], np.float64).reshape(-1).copy()
        fr['sc_w'] = np.asarray(sc.get('weights', np.ones(len(fr['sc_d']))), np.float64).reshape(-1).copy()
    of = b['object_contact_frame']
    if of is not None:
        fr['obj_pose'] = np.concatenate([np.asarray(of['object_position'], np.float64).reshape(3),
                                         np.asarray(of['object_quat_wxyz'], np.float64).reshape(4)])
        thr = float(args.object_contact_map_threshold); maxp = int(args.object_contact_map_max_points)
        if float(args.object_contact_map_cost) > 0 and thr >= 0:
            assert str(of.get('mode', 'world_points')) == 'world_points', of.get('mode')
            td = np.asarray(of['distances'], np.float64).reshape(-1)
            oid = np.asarray(of['object_ids'], np.int32).reshape(-1)
            tv = np.asarray(of['pair_vectors'], np.float64).reshape(-1, 3)
            op = np.asarray(of['object_points'], np.float64).reshape(-1, 3)
            # identical selection to UMR solve_frame_body_segment_qp (same numpy calls -> same tie-breaking)
            active = np.where(td <= thr)[0].astype(np.int32)
            if active.size > 0 and len(op) > 0:
                if maxp > 0 and active.size > maxp:
                    active = active[np.argsort(td[active])[:maxp]]
                pid = oid[active]
                valid = (pid >= 0) & (pid < len(op))
                active = active[valid]; pid = pid[valid]
            else:
                active = np.zeros(0, np.int32); pid = np.zeros(0, np.int32)
            fr['oc_slot'] = active.copy(); fr['oc_d'] = td[active].copy(); fr['oc_vec'] = tv[active].copy()
            fr['oc_pt'] = op[pid].copy() if len(pid) else np.zeros((0, 3))
    t0 = time.perf_counter()
    if MODE == 'cpu':
        q, cost = ORIG(*a, **kw)
    else:
        q, cost = np.asarray(b['qpos_init'], np.float64).copy(), 0.0
    REC['t_solve'].append(time.perf_counter() - t0)
    fr['q_cpu'] = np.asarray(q, np.float64).copy(); fr['cost_cpu'] = float(cost)
    REC['frames'].append(fr)
    return q, cost


def _pack(key, frames, width, dtype, fill, tail=()):
    T = len(frames)
    out = np.full((T, width) + tail, fill, dtype)
    cnt = np.zeros(T, np.int32)
    for t, f in enumerate(frames):
        v = f.get(key)
        if v is None:
            continue
        n = len(v); cnt[t] = n
        out[t, :n] = v
    return out, cnt


def save():
    fr = REC['frames']
    if not fr:
        print('[umr_capture] no frames recorded'); return
    st = REC['static']
    T = len(fr)
    d = dict(st)
    d['q_init'] = np.stack([f['q_init'] for f in fr])
    d['has_prev'] = np.array([f['has_prev'] for f in fr]); d['iters'] = np.array([f['iters'] for f in fr], np.int32)
    d['src'] = np.stack([f['src'] for f in fr]).astype(np.float32)
    d['ntgt'] = np.stack([f['ntgt'] for f in fr]).astype(np.float32)
    if 'gd' in fr[0]:
        d['gd'] = np.stack([f['gd'] for f in fr]); d['gw'] = np.stack([f['gw'] for f in fr])
    if any('sc_pairs' in f for f in fr):
        K = max(len(f.get('sc_d', [])) for f in fr)
        d['sc_pairs'], d['sc_n'] = _pack('sc_pairs', fr, K, np.int32, -1, (2,))
        d['sc_d'], _ = _pack('sc_d', fr, K, np.float64, 0.0)
        d['sc_w'], _ = _pack('sc_w', fr, K, np.float64, 0.0)
    if any('obj_pose' in f for f in fr):
        d['obj_pose'] = np.stack([f['obj_pose'] for f in fr])
        K = max(1, max(len(f.get('oc_slot', [])) for f in fr))
        d['oc_slot'], d['oc_n'] = _pack('oc_slot', fr, K, np.int32, -1)
        d['oc_d'], _ = _pack('oc_d', fr, K, np.float64, 0.0)
        d['oc_vec'], _ = _pack('oc_vec', fr, K, np.float64, 0.0, (3,))
        d['oc_pt'], _ = _pack('oc_pt', fr, K, np.float64, 0.0, (3,))
    d['q_cpu'] = np.stack([f['q_cpu'] for f in fr]); d['cost_cpu'] = np.array([f['cost_cpu'] for f in fr])
    d['t_solve'] = np.array(REC['t_solve'])
    d['mode'] = np.array(MODE)
    d['umr_out'] = np.array(str(getattr(M, '_LAST_OUT', '')))
    for k, v in EXTRA.items():
        d['x_' + k] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **d)
    ts = np.array(REC['t_solve'])
    print(f'[umr_capture] saved {OUT} frames={T} mode={MODE} solve_total={ts.sum():.2f}s '
          f'fps_solve={T / max(ts.sum(), 1e-9):.1f} wall={time.time() - REC["t_start"]:.1f}s')


# ---- static-prep extras: source binding, robot T-pose normals, scale (recorded for GPU preprocessing)
EXTRA = {}
import smpl_surface_retarget_common as _common
_orig_smv = _common.source_motion_vertices_joints


def _smv(*a, **kw):
    v, j, f = _orig_smv(*a, **kw)
    EXTRA['vertices_world0'] = np.asarray(v[:2], np.float32).copy(); EXTRA['joints_world0'] = np.asarray(j[:2], np.float32).copy()
    EXTRA['faces'] = np.asarray(f, np.int32)
    return v, j, f


_common.source_motion_vertices_joints = _smv
for _mn in ('retarget_body_segment_surface', 'retarget_body_segment_surface_hoi_hsi'):
    try:
        _bm = importlib.import_module(_mn)
    except Exception:
        continue
    def _wrap_bind(orig):
        def f(slot_points, template_vertices, template_faces, motion_vertices, *a, **kw):
            res = orig(slot_points, template_vertices, template_faces, motion_vertices, *a, **kw)
            EXTRA['bind_face_ids'] = np.asarray(res[4]['face_ids'], np.int32); EXTRA['bind_bary'] = np.asarray(res[4]['bary'], np.float32)
            EXTRA['template_vertices'] = np.asarray(template_vertices, np.float32); EXTRA['template_faces'] = np.asarray(template_faces, np.int32)
            EXTRA['vertices_scaled0'] = np.asarray(motion_vertices[:2], np.float32).copy()
            EXTRA['tpose_slot_normals'] = np.asarray(res[3], np.float32)
            return res
        return f
    def _wrap_transport(orig):
        def f(robot_tpose_normals_smpl, source_binding, template_vertices, template_faces, motion_vertices, *a, **kw):
            EXTRA['robot_tpose_normals_smpl'] = np.asarray(robot_tpose_normals_smpl, np.float32)
            return orig(robot_tpose_normals_smpl, source_binding, template_vertices, template_faces, motion_vertices, *a, **kw)
        return f
    if hasattr(_bm, 'bind_source_slots_with_normals'):
        _bm.bind_source_slots_with_normals = _wrap_bind(_bm.bind_source_slots_with_normals)
    if hasattr(_bm, 'transport_tpose_robot_normals'):
        _bm.transport_tpose_robot_normals = _wrap_transport(_bm.transport_tpose_robot_normals)
M.load_body_segment_module({})

M.solve_frame_body_segment_qp = recorder
sys.argv = [str(script)] + sys.argv[2:]
try:
    M.main()
finally:
    save()
