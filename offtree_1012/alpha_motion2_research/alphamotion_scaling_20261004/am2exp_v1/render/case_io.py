"""Inputs for the renderers: per-case npz files (interfaces.md "Eval output") and GT windows read from the
frozen trajectory caches through ROOT/data/trajectory_reader.py (read-only).

Per-case npz keys consumed (E1, written by CODE/eval/run_e1_eval.py / WP-JUDGE):
  pred_rot6 [T,J,6]  pred_joint_position [T,J,3] cm  pred_root_height_m [T]  (pred_contact_logits [T,J] unused)
  gt_rot6 [T,J,6]    gt_joint_position [T,J,3] cm    gt_root_height_m [T]
  source_pose9 [T,Js,9]  source_body  target_body  (+ window meta)
Optional keys this renderer understands (never required):
  meta / meta_json (JSON string)  -- or a sidecar <npz>.json -- with group_id, clip, start, end|length, fps, arm, mode
  source_world_root_cm [T,3]      -- source horizontal trajectory for the follow camera
  source_root_height_m [T]        -- source vertical placement (else re-read from the GT cache by meta)
  pred_root_offset_cm, gt_root_offset_cm (scalars) -- root convention used by the judge (subtracted, cm)
  timestamps_seconds [T]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _env
from ._env import rot6_to_matrix


@dataclass
class Window:
    """One body's motion on one window (all canonical Y-up, cm, root-relative positions)."""
    body: str
    R: np.ndarray | None            # [T,J,3,3] own-joint global rotations (None for human stick figure)
    P: np.ndarray                   # [T,J,3] cm
    root_h_m: np.ndarray | None     # [T] metres above the label floor (None -> unknown)
    world_root_cm: np.ndarray | None = None  # [T,3] canonical world root (x,z used for the follow camera)
    contact_hard: np.ndarray | None = None   # [T,J] label contacts (human source colouring fallback)
    contact_gap_m: np.ndarray | None = None  # [T,J]
    names: list = field(default_factory=list)
    parents: np.ndarray | None = None
    reach: float = 0.0


def _meta_from(npz_path: Path, z) -> dict:
    for k in ('meta', 'meta_json', 'window_meta', 'window_meta_json'):
        if k in z.files:
            v = z[k]
            v = v.item() if v.shape == () else v
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            if isinstance(v, dict):
                return v
    for side in (npz_path.with_suffix('.json'), npz_path.parent / (npz_path.stem + '_meta.json'), npz_path.parent / 'meta.json'):
        if side.is_file():
            try:
                return json.loads(side.read_text())
            except Exception:
                pass
    return {}


def _str(z, k, default=None):
    if k not in z.files:
        return default
    v = z[k]
    v = v.item() if v.shape == () else v
    return v.decode() if isinstance(v, bytes) else str(v)


def load_case(npz_path) -> dict:
    """-> dict(meta, source_body, target_body, source=Window, pred=Window, gt=Window|None, times[T])."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        meta = _meta_from(npz_path, z)
        sb = _str(z, 'source_body', meta.get('source_body'))
        tb = _str(z, 'target_body', meta.get('target_body'))
        arr = {k: z[k] for k in z.files if z[k].dtype != object and z[k].dtype.kind in 'fiub'}
    if sb is None or tb is None:
        raise ValueError(f'{npz_path}: source_body/target_body missing')
    T = arr['pred_rot6'].shape[0]
    fps = float(meta.get('fps', 30.0))
    times = arr.get('timestamps_seconds', np.arange(T) / fps).astype(np.float64)
    tn, tp, toff, treach = _env.static_spec(tb)
    pred = Window(body=tb, R=rot6_to_matrix(arr['pred_rot6']), P=arr['pred_joint_position'].astype(np.float64),
                  root_h_m=arr.get('pred_root_height_m'), names=tn, parents=tp, reach=treach)
    gt = None
    if 'gt_rot6' in arr:
        gt = Window(body=tb, R=rot6_to_matrix(arr['gt_rot6']), P=arr['gt_joint_position'].astype(np.float64),
                    root_h_m=arr.get('gt_root_height_m'), names=tn, parents=tp, reach=treach)
    sn, sp, soff, sreach = _env.static_spec(sb)
    s9 = arr['source_pose9'].astype(np.float64)
    src = Window(body=sb, R=rot6_to_matrix(s9[..., :6]), P=s9[..., 6:] * sreach,
                 root_h_m=arr.get('source_root_height_m'), world_root_cm=arr.get('source_world_root_cm'),
                 names=sn, parents=sp, reach=sreach)
    # fill source placement / labels from the frozen cache when meta identifies the window
    if (src.root_h_m is None or src.world_root_cm is None) and meta.get('group_id') is not None and meta.get('start') is not None:
        try:
            g = load_gt_window(meta['group_id'], sb, int(meta['start']), int(meta['start']) + T, with_contact=True)
            if src.root_h_m is None:
                src.root_h_m = g.root_h_m
            if src.world_root_cm is None:
                src.world_root_cm = g.world_root_cm
            src.contact_hard, src.contact_gap_m = g.contact_hard, g.contact_gap_m
        except Exception as e:  # pragma: no cover - depends on caches
            meta.setdefault('render_warnings', []).append(f'source cache re-read failed: {e!r}')
    offsets = {}
    for k in ('pred_root_offset_cm', 'gt_root_offset_cm'):
        if k in arr:
            offsets[k] = float(np.asarray(arr[k]).ravel()[0])
    return dict(meta=meta, source_body=sb, target_body=tb, source=src, pred=pred, gt=gt, times=times,
                offsets=offsets, path=str(npz_path))


def load_gt_window(group_id: str, body: str, start: int, end: int, with_contact: bool = True) -> Window:
    """GT window [start,end) (frame indices inside the clip) from the frozen caches."""
    tr = _env.trajectory_reader()
    d = tr.load_xwbc_trajectory(_env.data_manifest(), group_id, body, with_contact=with_contact)
    n = len(d['frame_index'])
    if not (0 <= start < end <= n):
        raise IndexError(f'window [{start},{end}) outside clip of {n} frames')
    sl = slice(start, end)
    w = Window(body=body, R=d['global_rotation_matrices'][sl].astype(np.float64),
               P=d['position_native'][sl].astype(np.float64),
               root_h_m=d['root_height_m'][sl].astype(np.float64) if 'root_height_m' in d else None,
               world_root_cm=d['world_root'][sl].astype(np.float64),
               contact_hard=d.get('contact_hard', None)[sl] if 'contact_hard' in d else None,
               contact_gap_m=d.get('contact_gap_m', None)[sl].astype(np.float64) if 'contact_gap_m' in d else None,
               names=[str(x) for x in d['joint_names']], parents=np.asarray(d['parents'], np.int64), reach=float(d['reach']))
    w.times = (np.arange(end - start) / float(d['fps'])).astype(np.float64)
    w.meta = dict(d['metadata'], start=start, end=end, fps=float(d['fps']))
    return w


def window_to_case_npz(path, source: Window, target_gt: Window, pred: Window | None = None, meta: dict | None = None):
    """Write an interfaces.md-style per-case npz (used by tests and the GT self-render)."""
    def r6(R):
        return np.concatenate([R[..., :, 0], R[..., :, 1]], -1).astype(np.float32)
    pred = pred or target_gt
    meta = dict(meta or {})
    np.savez(path, pred_rot6=r6(pred.R), pred_joint_position=pred.P.astype(np.float32),
             pred_root_height_m=np.asarray(pred.root_h_m, np.float32),
             pred_contact_logits=np.zeros(pred.P.shape[:2], np.float32),
             gt_rot6=r6(target_gt.R), gt_joint_position=target_gt.P.astype(np.float32),
             gt_root_height_m=np.asarray(target_gt.root_h_m, np.float32),
             source_pose9=np.concatenate([r6(source.R), source.P / source.reach], -1).astype(np.float32),
             source_body=np.array(source.body), target_body=np.array(target_gt.body),
             meta=np.array(json.dumps(meta)))
