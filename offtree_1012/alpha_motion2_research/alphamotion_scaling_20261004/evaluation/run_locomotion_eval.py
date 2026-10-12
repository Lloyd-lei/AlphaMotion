#!/usr/bin/env python3
"""Fixed continuous XWBC mapping and UniMate own-body evaluation; no training.

Example (run only under the caller's resource lock):
  python evaluation/run_locomotion_eval.py --root ROOT --run RUN_A --run RUN_B \
      --output ROOT/evaluations/locomotion --checkpoints all --device cpu

Reads sealed data/trajectory_reader.py and the actual scaling loader. Defaults to
three source clips in EACH XWBC source-exposure partition, all 15 robot targets,
and up to three heldout native sequences per UniMate body. A center window uses
consecutive ORIGINAL frames, never the sparse pilot rows. A shared selection JSON
and exact numeric case digest are checked for every checkpoint/run. --self-test
uses tiny synthetic CPU fixtures, and provides no real checkpoint evidence.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in [ROOT, ROOT/'orchestration', ROOT/'base', ROOT/'base/legacy', ROOT/'data', HERE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import numpy as np
import torch
from run_scaling_eval import (model_factory, model_variant, discover_checkpoints,
                              load_evaluation_checkpoint, json_write, file_sha)
from paired_storage import save_paired_arrays,load_paired_arrays
from eval_engine import (TrajectoryCase, PosePrediction, evaluate_case, temporal_metrics,
                         metric_summary, rotation_error_degrees, map_independent_frame_batches)
from curve_statistics import (evaluation_set_digest, summarize_cases, bootstrap_clips,
                              paired_clip_difference)
from surface_metrics import evaluate_fixed_surface
from trajectory_reader import load_xwbc_trajectory, load_unpaired_trajectory

MODES = ('full_source', 'masked_source', 'pure_se3')
TIME_FIELDS = {'pose9', 'global_rot6', 'global_rotation_matrices', 'position_native',
    'world_root', 'world_positions', 'world_rotation_matrices', 'frame_index',
    'source_row_index', 'timestamp_seconds', 'contact_hard', 'contact_soft',
    'contact_valid', 'contact_gap_m', 'contact_sdist', 'contact_trust', 'root_height_m',
    'root_height_valid', 'ground_valid', 'world_sole_positions_cm',
    'slide_normalized', 'slide_valid', 'slide_position', 'aperture_cm'}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest_arrays(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def actual_group_split(body, rows):
    """Original source partition survives a later whole-body holdout override."""
    a = body.arrays
    if not len(rows):
        raise ValueError(f'{body.name}: source group absent from actual loader')
    out = {}
    for key in ('split', 'source_clip_split'):
        if key not in a:
            raise ValueError(f'{body.name}: loader must preserve {key}')
        values = np.unique(a[key][rows])
        if len(values) != 1 or int(values[0]) not in (0, 1, 2):
            raise ValueError(f'{body.name}: physical source group crosses {key}')
        out[key] = int(values[0])
    return out


def xwbc_rows(body, group):
    a = body.arrays
    lo, hi = int(group['start']), int(group['start'] + group['length'])
    rows = np.flatnonzero((a['source_cache_index'] == group['cache_index']) &
                         (a['source_row_index'] >= lo) & (a['source_row_index'] < hi))
    if not len(rows) or len(np.unique(a['group'][rows])) != 1:
        raise ValueError('cache range must match exactly one actual loader source group')
    if not np.array_equal(a['frame_index'][rows], a['source_row_index'][rows] - lo):
        raise ValueError('loader/source-cache frame identity disagreement')
    return rows


def endpoints(body, count=2):
    ids = [int(i) for i in body.endpoint['indices']]
    if len(set(ids)) != len(ids) or any(not 0 <= i < len(body.names) for i in ids):
        raise ValueError('invalid source-verified endpoint policy')
    # The loader already applies arm/side evidence or honest generic-leaf labels.
    # Do not infer human hands from names of feet, wings, quadrupeds or tools here.
    return ids[:count]


def build_selection(manifest, bodies, *, clips=3, max_frames=64, seed=0,
                    targets=None, unimate_targets=None, source_contact=False, condition_count=2):
    if clips < 3 or not 2 <= max_frames <= 512:
        raise ValueError('need >=3 XWBC physical clips per source split; window length2..512')
    if condition_count not in (1, 2):
        raise ValueError('one or two source-verified endpoints only')
    by = {b.name: b for b in bodies}
    if 'human_smpl' not in by or by['human_smpl'].domain != 'xwbc':
        raise ValueError('real human_smpl source required')
    source = by['human_smpl']
    target_names = sorted(targets if targets is not None else
                          [b.name for b in bodies if b.domain == 'xwbc' and b.name != 'human_smpl'])
    if not target_names or any(n not in by or by[n].domain != 'xwbc' or n == 'human_smpl' for n in target_names):
        raise ValueError('explicit XWBC robot targets only')
    exposure = {n: ('seen_body' if len(by[n].eligible(True)) else 'heldout_body') for n in target_names}
    if set(exposure.values()) != {'seen_body', 'heldout_body'}:
        raise ValueError('evaluation must include seen and entire-body-heldout XWBC targets')
    groups = manifest['xwbc']['groups']
    candidates = defaultdict(list)
    for g in groups:
        sp = actual_group_split(source, xwbc_rows(source, g))
        if sp['source_clip_split'] != int(g['split']):
            raise ValueError('trajectory manifest split contradicts actual source loader')
        if int(g['length']) >= 2:
            candidates[sp['source_clip_split']].append(g)
    selected = []
    for split in (0, 1):
        pool = sorted(candidates[split], key=lambda g: digest_json([seed, g['group_id']]))
        if len(pool) < clips:
            raise ValueError(f'only {len(pool)} actual XWBC source clips in split{split}')
        selected.extend(pool[:clips])
    entries, coverage, unsupported = [], [], []
    body_receipts = {}

    def receipt(b):
        g = b.geom(1); surface = g.get('surface')
        body_receipts[b.name] = {'payload_sha256': b.metadata['payload_sha256'],
            'joint_names': list(b.names), 'parents': np.asarray(b.parents).tolist(),
            'offset_sha256': digest_arrays(np.asarray(b.offsets)), 'reach': float(b.reach),
            'metric': bool(b.metric), 'body_exposure': 'seen_body' if len(b.eligible(True)) else 'heldout_body',
            'surface_asset_sha256': surface['asset_sha256'] if surface is not None else None,
            'endpoint_policy': b.endpoint}

    receipt(source)
    for name in target_names:
        b = by[name]; receipt(b)
        for g in selected:
            rows = xwbc_rows(b, g); sp = actual_group_split(b, rows)
            if sp['source_clip_split'] != g['split']:
                raise ValueError('same XWBC source group must keep source split across bodies')
            a, sa = b.arrays, source.arrays; sr = xwbc_rows(source, g)
            if not np.array_equal(a['source_row_index'][rows], sa['source_row_index'][sr]):
                raise ValueError('cross-body paired source rows differ')
            size = min(max_frames, int(g['length'])); start = (int(g['length']) - size)//2
            common = dict(dataset='XWBC', target=name, group_id=g['group_id'],
                group_index=int(a['group'][rows[0]]), clip_id=g['group_id'], original_clip_name=g['clip'],
                source_dataset=g['source_dataset'], start=start, frames=size, source_frames=int(g['length']),
                source_clip_split=sp['source_clip_split'], target_effective_split=sp['split'],
                body_exposure=exposure[name], clip_exposure='train_clip' if g['split'] == 0 else 'heldout_clip')
            entries.append(dict(common, input_mode='full_source', source='human_smpl', task_kind='paired_mapping'))
            if endpoints(b, condition_count):
                entries.append(dict(common, input_mode='pure_se3', source=name, task_kind='completion'))
            else:
                unsupported.append(dict(target=name, group_id=g['group_id'], input_mode='pure_se3', reason='no source-verified endpoints'))
        coverage.append(dict(dataset='XWBC', body=name, body_exposure=exposure[name],
                             train_clips=clips, heldout_clips=clips))
    # Exact source-group identity, not filename or caption, determines any training exposure.
    uni_names = {b['object_type'] for b in manifest['unimate']['bodies']}
    if unimate_targets is not None:
        if not set(unimate_targets) <= uni_names:
            raise ValueError('requested UniMate body absent from sealed native manifest')
        uni_names &= set(unimate_targets)
    training_uni_groups = set()
    for b in bodies:
        if b.name in {v['object_type'] for v in manifest['unimate']['bodies']}:
            training_uni_groups.update(int(x) for x in b.arrays['group'][b.eligible(True)])
    for ub in sorted(manifest['unimate']['bodies'], key=lambda v: v['object_type']):
        name = ub['object_type']
        if name not in uni_names: continue
        if name not in by: raise ValueError('sealed UniMate body missing in actual loader')
        b = by[name]; receipt(b); pool = []
        for seq in ub['source_sequences']:
            if seq['status'] != 'passed': continue
            rows = np.flatnonzero(b.arrays['group'] == seq['source_group_index'])
            sp = actual_group_split(b, rows)
            if sp['source_clip_split'] != int(seq['split']):
                raise ValueError('UniMate native source split disagrees with loader')
            if sp['split'] == 0: continue
            if int(seq['frames']) < 2: continue
            pool.append((seq, sp))
        pool.sort(key=lambda pair: digest_json([seed, pair[0]['source_group_id']]))
        unique = [v[0]['source_group_id'] for v in pool]
        if len(set(unique)) != len(unique):
            raise ValueError('duplicate source aliases must not become independent trajectories')
        chosen = pool[:clips]
        coverage.append(dict(dataset='UniMate', body=name, available_heldout_groups=len(pool),
            selected_groups=len(chosen), minimum_clips_for_CI=3,
            independent_clip_CI_status='eligible' if len(chosen) >= 3 else 'insufficient_independent_clips',
            body_exposure=body_receipts[name]['body_exposure'],
            topology_category=ub['upstream_category'], metric=False, cross_body_mapping_gt=False))
        for seq, sp in chosen:
            n = int(seq['frames']); size = min(max_frames, n)
            common = dict(dataset='UniMate', target=name, source=name,
                group_index=int(seq['source_group_index']), group_id=seq['source_group_id'], clip_id=seq['source_group_id'],
                source_dataset='UniML3D', start=(n-size)//2, frames=size, source_frames=n,
                source_clip_split=sp['source_clip_split'], target_effective_split=sp['split'],
                body_exposure=body_receipts[name]['body_exposure'],
                clip_exposure='train_clip' if int(seq['source_group_index']) in training_uni_groups else 'heldout_clip',
                morphology=ub['upstream_category'], raw_source_sha256=seq['source_sha256'])
            entries.append(dict(common, input_mode='full_source', task_kind='reconstruction'))
            if endpoints(b, condition_count):
                for mode in ('masked_source', 'pure_se3'):
                    entries.append(dict(common, input_mode=mode, task_kind='completion'))
            else:
                unsupported.append(dict(target=name, group_id=seq['source_group_id'], input_mode='sparse_completion', reason='no source-verified endpoints'))
    for e in entries:
        e['case_id'] = e['dataset']+'__'+e['target']+'__'+digest_json([e['group_id'], e['start'], e['frames']])[:16]+'__'+e['input_mode']
    return {'schema': 'fixed_continuous_locomotion_selection_v1', 'seed': seed,
        'clips_per_XWBC_source_partition': clips, 'max_consecutive_native_frames': max_frames,
        'window_rule': 'center contiguous window by source length; no prediction/quality filtering',
        'source_contact_enabled': source_contact, 'condition_count': condition_count,
        'manifest_sha256': digest_json(manifest), 'bodies': body_receipts,
        'entries': entries, 'coverage': coverage, 'unsupported': unsupported,
        'ancestry_scope': 'Declared source groups and exact-content aliases only; cross-source/cross-dataset ancestry unknown',
        'normalization': 'actual loader static-geometry maximum root-to-joint path; no heldout motion fit',
        'world_prediction_policy': 'No reference world root or heading injected into prediction; root-relative metrics only'}


def load_window(manifest, entry, body, *, source=False, with_contact=False):
    """Read a complete physical clip, validate identity, then take native contiguous rows."""
    if entry['dataset'] == 'XWBC':
        full = load_xwbc_trajectory(manifest, entry['group_id'], body.name, with_contact=with_contact)
        if with_contact:
            with np.load(full['metadata']['contact_source'], allow_pickle=False) as z:
                header = json.loads(str(z['header']))
            threshold, distance = float(header['thr_m']), float(header['D_used_m'])
            if not np.isfinite([threshold, distance]).all() or min(threshold, distance) <= 0:
                raise ValueError('legacy contact feature scales missing')
            score = threshold - full['contact_gap_m']
            score = (np.sign(score)*np.log1p(np.abs(score)/threshold)/np.log1p(distance/threshold)).astype(np.float16).astype(np.float32)
            full['contact_valid'] &= np.isfinite(score)
            full['contact_sdist'] = np.where(full['contact_valid'], score, np.nan)
    else:
        full = load_unpaired_trajectory(manifest, body.name, entry['group_index'])
    n = entry['source_frames']; start = entry['start']; stop = start + entry['frames']
    if len(full['frame_index']) != n or not np.array_equal(full['frame_index'], np.arange(n)):
        raise ValueError('trajectory reader must return every original frame before window selection')
    if list(full['joint_names'].astype(str)) != list(body.names) or not np.array_equal(full['parents'], body.parents):
        raise ValueError('reader/loader joint order or topology disagreement')
    if not np.allclose(full['rest_offsets'], body.offsets, atol=1e-5, rtol=1e-6):
        raise ValueError('reader/loader native-unit geometry disagreement')
    if bool(full['metric_length_valid']) != bool(body.metric):
        raise ValueError('reader/loader physical units disagree')
    ts = np.asarray(full['timestamp_seconds'])
    if not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0):
        raise ValueError('actual source timestamps missing or non-monotonic')
    # Do not use the trajectory reader legacy stored reach for UniMate.
    full['pose9'] = np.concatenate([full['global_rot6'], full['position_native']/body.reach], axis=-1).astype(np.float32)
    a = body.arrays
    if entry['dataset'] == 'XWBC':
        g = next(g for g in manifest['xwbc']['groups'] if g['group_id'] == entry['group_id'])
        rows = xwbc_rows(body, g)
    else:
        rows = np.flatnonzero(a['group'] == entry['group_index'])
    fr = np.asarray(a['frame_index'][rows], dtype=np.int64)
    if np.any(fr < 0) or np.any(fr >= n) or len(np.unique(fr)) != len(fr):
        raise ValueError('loader source frame identity outside physical sequence')
    pe = np.linalg.norm(full['pose9'][fr, :, 6:] - a['pose9'][rows, :, 6:], axis=-1)
    re = np.max(np.abs(full['global_rot6'][fr] - a['global_rot6'][rows]))
    if float(pe.max()) > 2e-5 or float(re) > 2e-5:
        raise ValueError(f'reader versus sealed loader mismatch: normalized position {pe.max()}, rot6 {re}')
    if with_contact:
        if not np.array_equal(full['contact_valid'][fr], a['contact_valid'][rows]) or not np.array_equal(full['contact_sdist'][fr], a['contact_sdist'][rows], equal_nan=True):
            raise ValueError('actual source signed-log contact feature disagrees with sealed loader')
    out = {k: (v[start:stop].copy() if k in TIME_FIELDS else v) for k, v in full.items()}
    if not np.all(np.diff(out['frame_index']) == 1): raise ValueError('not native consecutive frames')
    out['validation'] = {'sealed_loader_overlap_frames': len(rows),
        'reader_loader_max_position_over_reach': float(pe.max()), 'reader_loader_max_rot6_abs': float(re),
        'normalizer': float(body.reach), 'source_native_frame_stride': 1,
        'min_timestamp_delta_seconds': float(np.diff(out['timestamp_seconds']).min()),
        'max_timestamp_delta_seconds': float(np.diff(out['timestamp_seconds']).max()),
        'declared_fps': float(full['fps']), 'whole_source_checksum_verified': False,
        'checksum_scope': 'sealed sparse payload/spec and selected continuous arrays; no full original cache numeric hash scan'}
    return out


def make_case(entry, source, target, sw, tw, *, condition_count=2, source_contact=False):
    T, J = tw['pose9'].shape[:2]; mode = entry['input_mode']
    if not np.array_equal(sw['frame_index'], tw['frame_index']) or not np.array_equal(sw['timestamp_seconds'], tw['timestamp_seconds']):
        raise ValueError('paired source/target must keep exact original frame and time identity')
    ids = endpoints(target, condition_count) if mode != 'full_source' else []
    if mode != 'full_source' and not ids: raise ValueError('sparse task lacks verified endpoints')
    role = {'declared_endpoint_policy': endpoints(target, condition_count)}
    values = tw['pose9'][:, ids].copy() if mode == 'pure_se3' else None
    condition_sha = digest_arrays(np.asarray(ids, np.int64), values) if values is not None else None
    fullbody_diagnostic = mode != 'full_source'
    source_hash = digest_arrays(sw['global_rot6'], sw['position_native'], *([sw['contact_sdist']] if source_contact and 'contact_sdist' in sw else [])) if mode == 'full_source' else None
    reference_hash = digest_arrays(tw['global_rot6'], tw['position_native'])
    body_exposure = 'seen_body' if len(target.eligible(True)) else 'heldout_body'
    if body_exposure != entry['body_exposure']: raise ValueError('actual training-body exposure changed')
    provenance = {'dataset': entry['dataset'], 'input_mode': mode, 'source_group_id': entry['group_id'],
        'source_clip_split': entry['source_clip_split'], 'target_effective_split': entry['target_effective_split'],
        'source_payload_sha256': source.metadata['payload_sha256'], 'target_payload_sha256': target.metadata['payload_sha256'],
        'source_window_numeric_sha256': source_hash, 'reference_window_numeric_sha256': reference_hash,
        'source_reader_validation': sw['validation'], 'target_reader_validation': tw['validation'],
        'source_frame_count': entry['source_frames'], 'selected_start': entry['start'], 'source_native_frame_stride': 1,
        'source_inputs': ('all true human/own-body pose joints; no target pose or target SE3' if mode == 'full_source' else
                          'only named own-body endpoint pose slots' if mode == 'masked_source' else
                          'constant-zero source; every source joint hidden; only declared target SE3'),
        'source_contact': 'actual trusted source gap converted by cache-specific signed-log feature; fp16 rounded as training' if source_contact and mode == 'full_source' and entry['dataset'] == 'XWBC' else None,
        'endpoint_policy': target.endpoint, 'selected_endpoint_names': [target.names[i] for i in ids],
        'fullbody_metric_role': 'diagnostic: sparse SE3 does not uniquely specify full posture' if fullbody_diagnostic else 'paired mapping error' if entry['dataset'] == 'XWBC' else 'same-body reconstruction error',
        'paired_cross_body_ground_truth': entry['task_kind'] == 'paired_mapping',
        'units': 'cm, explicit metric cache' if target.metric else 'canonical body units; no metric claim',
        'normalization': 'static geometry only, actual loader reach', 'position_frame': 'root relative canonical Y-up; original heading retained',
        'world_metrics': 'unsupported: no separately predicted world root/heading; GT anchors not copied',
        'ancestry': 'within-domain declared source identity/content aliases; cross-source ancestry unknown',
        'DOF_or_contact_feasibility': 'not certified by free global rotations', 'physical_execution': False}
    case = TrajectoryCase(entry['case_id'], entry['clip_id'], source.name, target.name, entry['task_kind'],
        body_exposure, entry['clip_exposure'], tw['frame_index'], tw['timestamp_seconds'], list(target.names),
        bool(target.metric), float(target.reach), tw['position_native'], tw['global_rotation_matrices'],
        complete_clip=entry['start'] == 0 and T == entry['source_frames'], source_frame_count=entry['source_frames'],
        reference_kind='paired_target' if entry['task_kind'] == 'paired_mapping' else 'same_body',
        role_endpoints=role, provenance=provenance, condition_set_sha256=condition_sha,
        target_morphology='humanoid' if entry['dataset'] == 'XWBC' else entry['morphology'])
    case.validate()
    return case, ids


def predict_window(model, source, target, sw, tw, mode, ids, *, batch_size=32, source_contact=False):
    device = next(model.parameters()).device; T = len(tw['frame_index']); J = len(target.names)
    saved = defaultdict(list); gpu_events = []
    def predict(indices):
        B = len(indices)
        x = torch.as_tensor(sw['pose9'][indices], device=device, dtype=torch.float32)
        hidden = torch.zeros(x.shape[:2], device=device, dtype=torch.bool)
        condition = None; contact = None; source_slide = None
        if mode == 'pure_se3':
            x = torch.zeros((B, J, 9), device=device); hidden = torch.ones((B, J), device=device, dtype=torch.bool)
            condition = {'joint_indices': torch.tensor(ids, device=device)[None].expand(B, -1),
                'values': torch.as_tensor(tw['pose9'][indices][:, ids], device=device, dtype=torch.float32),
                'valid': torch.ones((B, len(ids)), device=device, dtype=torch.bool)}
        elif mode == 'masked_source':
            if source.name != target.name: raise ValueError('masked same-body task only')
            hidden[:] = True; hidden[:, ids] = False
            x = torch.where(hidden[..., None], torch.zeros_like(x), x)
        elif mode != 'full_source': raise ValueError('explicit input mode required')
        if mode == 'full_source' and source_contact and 'contact_sdist' in sw:
            c = sw['contact_sdist'][indices]; valid = sw['contact_valid'][indices]
            contact = torch.as_tensor(np.where(valid, c, np.nan), device=device, dtype=torch.float32)
        if mode != 'pure_se3' and 'slide_normalized' in sw:
            source_slide = torch.as_tensor(sw['slide_normalized'][indices], device=device, dtype=torch.float32)
            source_slide = torch.where(hidden, torch.full_like(source_slide, float('nan')), source_slide)
        sg, tg = source.geom(B), target.geom(B)
        begin = end = None
        if device.type == 'cuda':
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        with torch.inference_mode():
            result = model.forward_kinematic(x, sg, tg,
                source_hidden=hidden, source_contact=contact, condition=condition,
                source_slide_normalized=source_slide)
        if end is not None:
            end.record(); gpu_events.append((begin, end))
        required = {'rot6': (B,J,6), 'slide_native': (B,J), 'joint_position': (B,J,3),
                    'joint_rotation': (B,J,3,3), 'raw_hpos': (B,J,3)}
        if not isinstance(result, dict): raise ValueError('explicit typed forward_kinematic dict required')
        for key, shape in required.items():
            if key not in result or tuple(result[key].shape) != shape: raise ValueError('typed model output '+key)
            saved[key].append(result[key].detach().float().cpu().numpy())
        return saved['joint_position'][-1], saved['joint_rotation'][-1]
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    pred, timing = map_independent_frame_batches(T, batch_size, predict, synchronize=sync)
    timing.update(gpu_forward_event_seconds=sum(a.elapsed_time(b) for a,b in gpu_events)/1000 if gpu_events else None,
                  synchronized_inference_wall_seconds=timing['wall_seconds'],
                  gpu_event_scope='forward_kinematic only; synchronized wall additionally includes CPU input/geometry/output transfer',
                  temporal_context=False, source_contact_supplied=source_contact and mode == 'full_source' and 'contact_sdist' in sw,
                  slides='model output only; GT slide values never fill target predictions', raw_hpos_used_as_XYZ=False)
    return pred, {k: np.concatenate(v) for k, v in saved.items()}, timing


def stable_surface(target, tw, outputs, device):
    T = len(tw['frame_index'])
    def move(value):
        if isinstance(value, torch.Tensor): return value.to(device)
        if isinstance(value, dict): return {k:move(v) for k,v in value.items()}
        return value
    g = move(target.geom(T)); s = g.get('surface')
    if s is None: return {'status': 'NA', 'reason': 'no verified mesh-to-skeleton binding'}, {}
    from surface.surface_geometry import predict_bound_surface
    kwargs = {'device': device, 'dtype': torch.float32}
    pr = torch.as_tensor(outputs['rot6'], **kwargs); rr = torch.as_tensor(tw['global_rot6'], **kwargs)
    ps = torch.as_tensor(outputs['slide_native'], **kwargs)
    if bool((torch.as_tensor(g.get('joint_type', [])) == 3).any()) and 'slide_position' not in tw:
        raise ValueError('actual prismatic reference positions required for surface comparison')
    rs = torch.as_tensor(tw.get('slide_position', np.zeros((T,len(target.names)))), **kwargs)
    result = evaluate_fixed_surface(pr, rr, g, bind_fn=predict_bound_surface,
        expected_patch_sha256=s['asset_sha256'], predicted_slide_native=ps, reference_slide_native=rs)
    with torch.inference_mode():
        p = predict_bound_surface(pr, g, slide_native=ps)['surface']
        r = predict_bound_surface(rr, g, slide_native=rs)['surface']
    arrays = {'surface_patch_ids': np.asarray(s['patch_ids']), 'surface_valid': s['valid'].cpu().numpy(),
        'surface_area_native_squared': s['area'].cpu().numpy(),
        'predicted_surface_position_native': p['position'].cpu().numpy(),
        'reference_surface_position_native': r['position'].cpu().numpy(),
        'predicted_surface_normal': p['normal'].cpu().numpy(), 'reference_surface_normal': r['normal'].cpu().numpy()}
    if target.metric and result['status'] == 'measured':
        mm = dict(result['position_area_mean'])
        for key in ('mean', 'finite_only_mean', 'p95'):
            if mm[key] is not None: mm[key] *= 10
        mm['unit'] = 'mm'; result['position_area_mean_mm'] = mm
    return result, arrays


def summarize(records, samples=1000):
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['input_mode'])].append(r)
    reports = []
    for (dataset, mode), rows in sorted(groups.items()):
        strata = defaultdict(list)
        for r in rows:
            strata[(r['task_kind'],r['target_morphology'],r['body_exposure'],r['clip_exposure'],r['metric_length_valid'])].append(r)
        reports.append({'dataset': dataset, 'input_mode': mode,
            'fullbody_metric_role': rows[0]['provenance']['fullbody_metric_role'],
            'position': summarize_cases(rows, 'metrics.position'), 'position_mm': summarize_cases([r for r in rows if r['metric_length_valid']], 'metrics.position_mm') if any(r['metric_length_valid'] for r in rows) else [],
            'rotation_degrees': summarize_cases(rows, 'metrics.rotation'),
            'position_over_reach': summarize_cases(rows),
            'surface': summarize_cases([r for r in rows if r['surface'].get('patch_set_sha256')], 'surface.position_over_reach_area_mean') if any(r['surface'].get('patch_set_sha256') for r in rows) else [],
            'surface_unsupported_case_ids': [r['case_id'] for r in rows if r['surface']['status'] != 'measured'],
            'bootstrap': [{'stratum': list(key), **bootstrap_clips(rs, samples=samples, minimum_clips=3)} for key, rs in sorted(strata.items(), key=lambda x:repr(x[0]))]})
    return reports


def evaluate_entry(model, entry, source, target, sw, tw, directory, *, batch_size=32,
                   source_contact=False, condition_count=2, reference_store=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    case, ids = make_case(entry, source, target, sw, tw, condition_count=condition_count, source_contact=source_contact)
    T,J = len(case.frame_ids), len(case.joint_names); extras = {}; timing = None
    try:
        pred, outputs, timing = predict_window(model, source, target, sw, tw, entry['input_mode'], ids,
                                               batch_size=batch_size, source_contact=source_contact)
        report_start = time.perf_counter()
        surface, extras = stable_surface(target, tw, outputs, torch.device('cpu'))
        failure = None
    except Exception as exc:
        report_start = time.perf_counter()
        failure = repr(exc)
        pred = PosePrediction(np.full((T,J,3), np.nan), np.full((T,J,3,3), np.nan), failure_reason=failure)
        outputs = {'rot6': np.full((T,J,6), np.nan), 'slide_native': np.full((T,J), np.nan), 'raw_hpos': np.full((T,J,3), np.nan)}
        surface = {'status':'failed_case_retained', 'reason':failure}
        bound = target.geom(1).get('surface')
        if bound is not None:
            surface.update(patch_set_sha256=bound['asset_sha256'], position_over_reach_area_mean=metric_summary(np.full(T,np.nan),'reach_fraction'))
        extras['failure_traceback'] = np.array(traceback.format_exc())
    result = evaluate_case(case, pred, timing=timing)
    pe = np.linalg.norm(pred.positions - case.reference_positions, axis=-1)
    re = rotation_error_degrees(pred.rotations, case.reference_rotations)
    if case.metric: result['metrics']['position_mm'] = metric_summary(pe*10, 'mm')
    unit = 'cm' if case.metric else 'canonical_body_units'
    result.update(dataset=entry['dataset'], input_mode=entry['input_mode'], surface=surface,
        continuity={'native_frame_stride':1, 'frame_ids_consecutive':bool(np.all(np.diff(case.frame_ids)==1)),
            'true_timestamps':True, 'selected_frames':T, 'original_source_frames':entry['source_frames'],
            'source_clip_split':entry['source_clip_split'], 'effective_target_split':entry['target_effective_split'],
            'evaluation_is_subclip':not case.complete_clip},
        continuous_window_temporal=temporal_metrics(pred.positions, case.timestamps_seconds, unit),
        reference_window_temporal=temporal_metrics(case.reference_positions, case.timestamps_seconds, unit),
        error_window_temporal=temporal_metrics(pred.positions-case.reference_positions, case.timestamps_seconds, unit))
    if ids:
        result['visible_SE3'] = {'indices':ids, 'names':[target.names[i] for i in ids],
            'position':metric_summary(pe[:,ids],unit), 'rotation':metric_summary(re[:,ids],'degree'),
            'position_over_reach':metric_summary(pe[:,ids]/target.reach,'reach_fraction'),
            'condition_kind': 'target SE3 condition' if entry['input_mode']=='pure_se3' else 'visible source pose slots'}
    arrays = {'frame_ids':case.frame_ids, 'timestamps_seconds':case.timestamps_seconds,
        'source_joint_names':np.asarray(source.names), 'target_joint_names':np.asarray(target.names),
        'source_pose9_reference_not_necessarily_input':sw['pose9'],
        'reference_global_rot6':tw['global_rot6'], 'reference_position_native':case.reference_positions,
        'reference_global_rotation':case.reference_rotations, 'predicted_position_native':pred.positions,
        'predicted_global_rotation':pred.rotations, 'predicted_global_rot6':outputs['rot6'],
        'predicted_slide_native':outputs['slide_native'], 'raw_hpos_NOT_cartesian':outputs['raw_hpos'],
        'position_error_native':pe, 'position_error_over_reach':pe/target.reach, 'rotation_error_degrees':re,
        'observed_endpoint_indices':np.asarray(ids,np.int64), 'target_parents':np.asarray(target.parents),
        'target_rest_offsets_native':np.asarray(target.offsets), 'target_static_reach':np.array(target.reach), **extras}
    if entry['input_mode']=='pure_se3': arrays['condition_values_pose9']=tw['pose9'][:,ids]
    if 'world_positions' in tw: arrays['reference_world_positions_only']=tw['world_positions']
    if case.metric: arrays['position_error_mm']=pe*10
    result['arrays_storage']=save_paired_arrays(directory,arrays,reference_store=reference_store)
    result['arrays_sha256'] = file_sha(directory/'paired_arrays.npz')
    with (directory/'per_frame_curve.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['frame_id','timestamp_seconds','mean_position_'+unit,'mean_position_over_reach','mean_rotation_degree','mean_position_mm' if case.metric else 'metric_mm_unavailable'])
        for i in range(T): w.writerow([int(case.frame_ids[i]),float(case.timestamps_seconds[i]),float(pe[i].mean()),float(pe[i].mean()/target.reach),float(re[i].mean()),float(pe[i].mean()*10) if case.metric else 'NA'])
    result['post_inference_cpu_reporting_wall_seconds'] = time.perf_counter()-report_start
    result['reporting_timing_scope'] = 'CPU metrics, CPU fixed-surface binding (includes geometry transfer), compression and CSV; separate from synchronized inference'
    json_write(directory/'case_result.json', result)
    return case, result


def plot_checkpoint_curves(output, evaluations):
    """Readable checkpoint curves, stratified by task, morphology and exposure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for item in evaluations:
        for report in item['strata']:
            for s in report['position_over_reach']:
                key=(report['dataset'],report['input_mode'],s['target_morphology'],s['body_exposure'],s['clip_exposure'])
                groups[key].append((item['run_id'],item['step'],s['body_macro']))
    for key, rows in groups.items():
        fig,ax=plt.subplots(figsize=(8,4))
        for run in sorted({r[0] for r in rows}):
            points=sorted((s,v) for r,s,v in rows if r==run)
            ax.plot([v[0] for v in points],[np.nan if v[1] is None else v[1] for v in points],marker='o',label=run)
        ax.set(xlabel='Checkpoint optimizer step (not matched wall compute)',ylabel='Position error / static reach',title=' | '.join(key))
        ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(Path(output)/('checkpoint_curve_'+digest_json(key)[:12]+'.png'),dpi=140);plt.close(fig)


def paired_comparisons(evaluations, samples):
    matched=defaultdict(dict);out=[]
    for e in evaluations:
        key=(e['size'],e['seed'],e['step'])
        if e['variant'] in matched[key]: raise ValueError('multiple same-size/seed/step runs need separate output comparison cohorts')
        matched[key][e['variant']]=e
    for key,pair in sorted(matched.items(),key=lambda x:repr(x[0])):
        if set(pair)!={'A','B'}:continue
        if pair['A']['evaluation_set_sha256']!=pair['B']['evaluation_set_sha256']:raise ValueError('A/B numeric case set differs')
        ar={r['case_id']:r for r in pair['A']['records']};br={r['case_id']:r for r in pair['B']['records']}
        strata=defaultdict(lambda:([],[]))
        for cid,a in ar.items():
            b=br[cid]
            if a['surface'].get('patch_set_sha256')!=b['surface'].get('patch_set_sha256') and a['surface']['status']==b['surface']['status']=='measured':
                raise ValueError('A/B surface points changed')
            sk=(a['dataset'],a['input_mode'],a['target_morphology'],a['body_exposure'],a['clip_exposure'],a['metric_length_valid'])
            strata[sk][0].append(a);strata[sk][1].append(b)
        for sk,(a,b) in sorted(strata.items(),key=lambda x:repr(x[0])):
            row={'size_seed_step':list(key),'stratum':list(sk),'direction':'A minus B','position_over_reach':paired_clip_difference(a,b,samples=samples)}
            valid=[i for i,(x,y) in enumerate(zip(a,b)) if x['surface'].get('patch_set_sha256') and x['surface'].get('patch_set_sha256')==y['surface'].get('patch_set_sha256')]
            row['surface_coverage']={'matched_bound_cases_including_failures':len(valid),'all_cases':len(a),'unsupported_or_failed_case_ids':[x['case_id'] for i,x in enumerate(a) if i not in valid]}
            if valid:row['surface_position_over_reach']=paired_clip_difference([a[i] for i in valid],[b[i] for i in valid],metric_path='surface.position_over_reach_area_mean',samples=samples)
            out.append(row)
    return out


def source_snapshot():
    paths = {Path(__file__).resolve(), ROOT/'data/trajectory_reader.py'}
    for directory in ('evaluation','base','base/legacy','surface','prismatic','orchestration'):
        paths.update((ROOT/directory).glob('*.py'))
    return {str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths)}


def resume_report(directory, identity):
    """Skip only a successful exact identity with every declared output still intact."""
    directory=Path(directory);path=directory/'evaluation_summary.json'
    if not directory.exists() or not any(directory.iterdir()):return None
    if not path.is_file():raise FileExistsError('partial evaluation exists; use a new output directory: '+str(directory))
    report=json.loads(path.read_text())
    if report.get('resume_identity')!=identity or report.get('completion_status')!='complete' or report.get('failure_case_ids'):
        raise FileExistsError('existing evaluation is failed, incomplete or identity changed; refusing blind skip/overwrite: '+str(directory))
    files=report.get('output_files_sha256')
    if not isinstance(files,dict) or len(files)!=report['case_count']*3:
        raise ValueError('successful evaluation output seal missing')
    for relative, expected in files.items():
        f=directory/relative
        if f.resolve().parent.parent!=directory.resolve() or not f.is_file() or file_sha(f)!=expected:
            raise ValueError('saved evaluation output hash changed: '+str(f))
        if f.name=='paired_arrays.npz':load_paired_arrays(f)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--manifest',type=Path)
    p.add_argument('--run',type=Path,action='append');p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path,action='append',help='Explicit completed weight node; repeatable')
    p.add_argument('--checkpoints',choices=['all','final'],default='final')
    p.add_argument('--selection',type=Path,help='Existing shared selection must match; otherwise created once')
    p.add_argument('--device',default='cpu');p.add_argument('--clips-per-split',type=int,default=3)
    p.add_argument('--max-frames',type=int,default=64);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--target',action='append');p.add_argument('--unimate-target',action='append')
    p.add_argument('--source-contact',action='store_true',help='Optional verified signed-log contact feature; default pose-only')
    p.add_argument('--condition-count',type=int,choices=[1,2],default=2)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--bootstrap-samples',type=int,default=1000)
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:return self_test()
    if args.root is None or not args.run or args.output is None:p.error('--root, --run and --output required')
    if len(set(p.resolve() for p in args.run))!=len(args.run):p.error('duplicate run directory')
    if args.checkpoint and len(args.run)>1 and any(cp.resolve().parent not in {r.resolve() for r in args.run} for cp in args.checkpoint):
        p.error('with multiple runs each explicit checkpoint must reside in one specified run directory')
    torch.set_num_threads(1);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
    manifest_path=args.manifest or args.root/'data/data_manifest_v2.json'
    manifest=json.loads(manifest_path.read_text());args.output.mkdir(parents=True,exist_ok=True)
    selection_path=args.selection or args.output/'fixed_selection.json'
    locked=json.loads(selection_path.read_text()) if selection_path.exists() else None
    fixed_hash=None;evaluations=[];sources=source_snapshot()
    from scaling_data import load_scaling_bodies
    for run in args.run:
        seen_steps={}
        checkpoints=([cp for cp in args.checkpoint if len(args.run)==1 or cp.resolve().parent==run.resolve()]
                     if args.checkpoint else discover_checkpoints(run,args.checkpoints))
        if not checkpoints:raise ValueError('no explicit checkpoint assigned to run '+str(run))
        for checkpoint in checkpoints:
            payload=load_evaluation_checkpoint(checkpoint)
            config=payload['config'] or json.loads((run/'config.json').read_text())
            variant=model_variant(config);run_id=config.get('run_id',run.name);step=payload['step'];cp_sha=file_sha(checkpoint)
            if step in seen_steps:
                if seen_steps[step]==cp_sha:continue
                raise ValueError('conflicting explicit checkpoints at the same step')
            seen_steps[step]=cp_sha
            bodies,audit=load_scaling_bodies(args.root,config,device,include_surface=True);by={b.name:b for b in bodies}
            selection=build_selection(manifest,bodies,clips=args.clips_per_split,max_frames=args.max_frames,seed=args.seed,
                targets=args.target,unimate_targets=args.unimate_target,source_contact=args.source_contact,condition_count=args.condition_count)
            if locked is not None and locked!=selection:raise ValueError('frozen selection/geometry/source exposure changed across checkpoints or runs')
            if locked is None:locked=selection;json_write(selection_path,selection)
            out=args.output/run_id/f'checkpoint_step{step:04d}';prepared=[];cases=[];last={}
            for entry in selection['entries']:
                source,target=by[entry['source']],by[entry['target']]
                def read(b,contact):
                    k=(entry['dataset'],entry['group_id'],entry['start'],entry['frames'],b.name,contact)
                    if k not in last:
                        if len(last)>=4:last.pop(next(iter(last)))
                        last[k]=load_window(manifest,entry,b,with_contact=contact)
                    return last[k]
                tw=read(target,False)
                sw=read(source,args.source_contact and entry['input_mode']=='full_source' and entry['dataset']=='XWBC') if entry['input_mode']=='full_source' else tw
                case,_=make_case(entry,source,target,sw,tw,condition_count=args.condition_count,source_contact=args.source_contact)
                cases.append(case);prepared.append((entry,source,target,sw,tw))
            current_hash=evaluation_set_digest(cases)
            if fixed_hash is not None and fixed_hash!=current_hash:raise ValueError('actual selected numeric input/reference cases changed')
            fixed_hash=current_hash
            identity={'checkpoint_sha256':cp_sha,'runner_sha256':file_sha(__file__),'source_sha256':sources,
                'selection_sha256':digest_json(selection),'evaluation_set_sha256':current_hash,
                'execution_device':str(device),'batch_size':args.batch_size,'bootstrap_samples':args.bootstrap_samples,
                'model_config_sha256':digest_json(config)}
            previous=resume_report(out,identity)
            if previous is not None:
                evaluations.append(previous)
                print(json.dumps({'status':'verified_resume_skip','run_id':run_id,'step':step,'checkpoint_sha256':cp_sha}),flush=True)
                del bodies,by,last,prepared,payload
                continue
            out.mkdir(parents=True,exist_ok=True)
            model=model_factory(config,variant);model.load_state_dict(payload['model'],strict=True);model.to(device).eval();records=[]
            for entry,source,target,sw,tw in prepared:
                case,result=evaluate_entry(model,entry,source,target,sw,tw,out/entry['case_id'],batch_size=args.batch_size,
                    source_contact=args.source_contact,condition_count=args.condition_count,reference_store=args.output/'_references')
                records.append(result)
            outputs={str(f.relative_to(out)):file_sha(f) for e in selection['entries'] for f in (out/e['case_id']).iterdir() if f.name in {'case_result.json','paired_arrays.npz','per_frame_curve.csv'}}
            failures=[r['case_id'] for r in records if r['status']!='measured']
            report={'schema':'continuous_locomotion_evaluation_v1','run_id':run_id,'variant':variant,'size':config.get('size','unspecified'),
                'seed':config.get('seed',0),'step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':cp_sha,
                'checkpoint_load_policy':payload['checkpoint_load_policy'],'selection_sha256':digest_json(selection),
                'evaluation_set_sha256':current_hash,'case_count':len(cases),'physical_source_clip_count':len({c.clip_id for c in cases}),
                'coverage':selection['coverage'],'strata':summarize(records,args.bootstrap_samples),'records':records,
                'data_audit':audit,'source_manifest_sha256':file_sha(manifest_path),'runner_sha256':file_sha(__file__),
                'resume_identity':identity,'output_files_sha256':outputs,'completion_status':'complete' if not failures else 'failed_cases_retained',
                'training_updates':0,'execution_device':str(device),'world_root_from_reference_used':False,
                'failure_case_ids':failures,
                'inference_synchronized_wall_seconds':sum(r['timing']['synchronized_inference_wall_seconds'] for r in records if r.get('timing')),
                'gpu_forward_event_seconds':sum(r['timing']['gpu_forward_event_seconds'] or 0 for r in records if r.get('timing')) if device.type=='cuda' else None,
                'post_inference_cpu_reporting_wall_seconds':sum(r['post_inference_cpu_reporting_wall_seconds'] for r in records),
                'claim_boundary':'Paired XWBC human-to-humanoid mapping; separate UniMate same-body reconstruction/completion. Sparse-SE3 fullbody errors diagnostic. No physical execution, certified joint feasibility, cross-morphology paired truth or exhaustive cross-source dedup claim.'}
            json_write(out/'evaluation_summary.json',report);evaluations.append(report)
            print(json.dumps({'status':report['completion_status'],'run_id':run_id,'step':step,'cases':len(records)}),flush=True)
            del model,bodies,payload,by,last,prepared
            if device.type=='cuda':torch.cuda.empty_cache()
    # Retain prior successful nodes when a later invocation requests only early steps.
    current={(e['run_id'],e['step']):e for e in evaluations}
    for path in sorted(args.output.glob('*/checkpoint_step*/evaluation_summary.json')):
        previous=json.loads(path.read_text());key=(previous['run_id'],previous['step'])
        if key in current:continue
        identity=previous.get('resume_identity',{})
        if identity.get('runner_sha256')!=file_sha(__file__) or identity.get('source_sha256')!=sources or previous.get('evaluation_set_sha256')!=fixed_hash:
            raise ValueError('existing earlier/later report is not comparable under current code and fixed set')
        checked=resume_report(path.parent,identity)
        current[key]=checked
    evaluations=sorted(current.values(),key=lambda e:(e['run_id'],e['step']))
    json_write(args.output/'evaluation_index.json',{'fixed_evaluation_set_sha256':fixed_hash,
        'evaluations':[{k:e[k] for k in ('run_id','variant','size','seed','step','checkpoint_sha256','failure_case_ids')} for e in evaluations],
        'actual_execution':True,'training_updates':0})
    json_write(args.output/'paired_A_B.json',{'comparison_unit':'same size/seed/checkpoint step, same selected numeric cases; not equal wall compute',
        'comparisons':paired_comparisons(evaluations,args.bootstrap_samples)})
    plot_checkpoint_curves(args.output,evaluations)
    print(json.dumps({'output':str(args.output),'evaluated_or_verified_skipped_checkpoints':len(evaluations),'fixed_evaluation_set_sha256':fixed_hash}))
    if any(e['failure_case_ids'] for e in evaluations):raise SystemExit(1)


def self_test():
    """CPU-only fixtures exercise actual A/B factories; never real-data evidence."""
    import tempfile
    from types import SimpleNamespace
    from prismatic.kinematics import mixed_forward_kinematics
    torch.set_num_threads(1)
    class ToyBody:
        def __init__(self,name,heldout=False):
            self.name=name;self.names=['root','left_wrist','right_wrist'];self.parents=np.array([-1,0,0]);self.offsets=np.array([[0,0,0],[10,0,0],[-10,0,0]],np.float32)
            self.reach=10.;self.metric=True;self.domain='xwbc';self.heldout=heldout
            self.endpoint={'indices':[1,2],'names':self.names[1:],'kind':'SYNTHETIC_TEST_ONLY'}
            self.metadata={'payload_sha256':'0'*64}
        def eligible(self,train):return np.array([],np.int64) if self.heldout else np.array([0])
        def geom(self,B):
            off=torch.from_numpy(self.offsets);J=3
            return dict(parents=torch.tensor(self.parents),kinematic_rest_off=off,rest_off=off[None].expand(B,-1,-1),bonelen=off.norm(dim=-1)[None].expand(B,-1)/10,
                depth=torch.tensor([0,1,1])[None].expand(B,-1),hop=torch.tensor([[0,1,1],[1,0,2],[1,2,0]])[None].expand(B,-1,-1),
                sem=torch.zeros(B,J,1024),dof=torch.zeros(B,J,19),metadata_missing=torch.zeros(B,J,2),pose_scale=torch.full((B,),10.),
                root_fixed=True,surface=None)
    src=ToyBody('human_smpl');tgt=ToyBody('synthetic_heldout',True);seen=ToyBody('synthetic_seen')
    G=np.tile(np.array([1,0,0,0,1,0],np.float32),(6,3,1));fk=mixed_forward_kinematics(torch.from_numpy(G),tgt.geom(6))
    P=fk['joint_position'].numpy();R=fk['joint_rotation'].numpy()
    window=dict(global_rot6=G,global_rotation_matrices=R,position_native=P,pose9=np.concatenate([G,P/10],-1),frame_index=np.arange(1,7),timestamp_seconds=np.arange(1,7)/30,validation={'synthetic':True})
    base=dict(dataset='XWBC',target=tgt.name,group_id='synthetic_clip',clip_id='synthetic_clip',source_clip_split=1,target_effective_split=2,
        body_exposure='heldout_body',clip_exposure='heldout_clip',source_frames=8,start=1,frames=6,source='human_smpl')
    cfg={'model':dict(d_model=32,n_heads=4,enc_layers=1,dec_layers=1,n_latent=8,n_latent_p=4,fsq_stages=1,fsq_levels=[32]*13,sem_dim=1024,n_judges=12)}
    checks=[]
    with tempfile.TemporaryDirectory(prefix='locomotion_eval_synthetic_') as td:
        hashes={}
        for variant in ('A','B'):
            torch.manual_seed(0);model=model_factory(cfg,variant).eval();cases=[];records=[]
            for mode in ('full_source','masked_source','pure_se3'):
                entry=dict(base,input_mode=mode,task_kind='paired_mapping' if mode=='full_source' else 'completion',case_id='SYNTHETIC__'+mode)
                s=src if mode=='full_source' else tgt
                case,result=evaluate_entry(model,entry,s,tgt,window,window,Path(td)/variant/mode,batch_size=2)
                assert result['status']=='measured',result
                assert result['metrics']['position_mm']['unit']=='mm'
                assert not case.complete_clip and result['continuity']['native_frame_stride']==1
                assert np.isfinite(result['continuous_window_temporal']['speed']['mean'])
                assert all(p.grad is None for p in model.parameters())
                z=load_paired_arrays(Path(td)/variant/mode/'paired_arrays.npz')
                assert z['predicted_position_native'].shape==(6,3,3)
                assert np.max(np.abs(z['predicted_global_rotation'][:,0]-np.eye(3)))<1e-6
                cases.append(case);records.append(result)
            hashes[variant]=evaluation_set_digest(cases)
            assert len(summarize(records,5))==3
            checks.append(variant+' real factory typed dict / full / masked / pure SE3 / fixed root / outputs')
        assert hashes['A']==hashes['B'];checks.append('identical selected numeric case digest across A/B')
        # Freeze-source split cannot be replaced by the whole-body target split.
        groups=[]
        for i in range(6):groups.append(dict(group_id='clip'+str(i),clip='clip'+str(i),source_dataset='SYNTHETIC',cache_index=0,start=i*8,length=8,split=0 if i<3 else 1))
        for b in (src,tgt,seen):
            b.arrays=dict(source_cache_index=np.zeros(12,int),source_row_index=np.concatenate([np.array([0,7])+i*8 for i in range(6)]),
                frame_index=np.tile([0,7],6),group=np.repeat(np.arange(6),2),split=np.full(12,2) if b.heldout else np.repeat([0,0,0,1,1,1],2),
                source_clip_split=np.repeat([0,0,0,1,1,1],2))
        manifest={'xwbc':{'groups':groups},'unimate':{'bodies':[]}}
        sel=build_selection(manifest,[src,tgt,seen],max_frames=4)
        assert len(sel['entries'])==24 and len({e['clip_id'] for e in sel['entries']})==6
        assert len([e for e in sel['entries'] if e['body_exposure']=='heldout_body' and e['clip_exposure']=='train_clip'])==6
        tgt.arrays['source_clip_split'][0]=1
        try:build_selection(manifest,[src,tgt,seen],max_frames=4)
        except ValueError:pass
        else:raise AssertionError('split leakage accepted')
        checks.append('six native source clips / seen-heldout crossed exposure / reject cross-group split leakage')
        tgt.arrays['source_clip_split'][0]=0
        # Exercise the unchanged real trajectory reader on tiny synthetic files.
        cache=Path(td)/'native';cache.mkdir()
        n=48;local=np.tile(np.array([1,0,0,0,1,0],np.float32),(n,3,1))
        torch.save(torch.from_numpy(local),cache/'local.pt');torch.save(torch.zeros(n,3),cache/'root.pt')
        spec=cache/'spec.npz';np.savez(spec,parents=src.parents,rest_offsets=src.offsets,joint_names=np.array(src.names))
        ground=cache/'ground.json';ground.write_text(json.dumps({'sole_j':[],'sole_off_m':[]}))
        manifest['xwbc'].update(fps=30,bodies=[dict(body=src.name,static_spec=str(spec),static_spec_sha256=file_sha(spec),
            static_ground=str(ground),heldout_body=False,caches=[dict(local_rot6_tensor=str(cache/'local.pt'),root_world_cm_tensor=str(cache/'root.pt'))])])
        src.arrays['global_rot6']=local[src.arrays['source_row_index']]
        src.arrays['pose9']=np.concatenate([src.arrays['global_rot6'],np.tile(src.offsets,(12,1,1))/src.reach],-1)
        sample=dict(sel['entries'][0],group_id='clip0',start=2,frames=4,source_frames=8)
        native=load_window(manifest,sample,src)
        assert native['frame_index'].tolist()==[2,3,4,5] and native['validation']['sealed_loader_overlap_frames']==2
        src.arrays['pose9'][0,1,6]+=1
        try:load_window(manifest,sample,src)
        except ValueError:pass
        else:raise AssertionError('reader/loader mismatch accepted')
        checks.append('unchanged real trajectory reader on synthetic cache: original contiguous window and mismatch rejection')
        # Resume needs exact identity AND every declared output byte hash.
        resume_dir=Path(td)/'resume';case_dir=resume_dir/'case';case_dir.mkdir(parents=True)
        for name in ('per_frame_curve.csv','case_result.json'):(case_dir/name).write_text('SYNTHETIC')
        np.savez(case_dir/'paired_arrays.npz',synthetic_fixture=np.zeros(1))
        identity={'checkpoint_sha256':'0'*64,'runner_sha256':'1'*64,'evaluation_set_sha256':'2'*64}
        report=dict(resume_identity=identity,completion_status='complete',failure_case_ids=[],case_count=1,
            output_files_sha256={str(f.relative_to(resume_dir)):file_sha(f) for f in case_dir.iterdir()})
        json_write(resume_dir/'evaluation_summary.json',report)
        assert resume_report(resume_dir,identity)==report
        try:resume_report(resume_dir,dict(identity,checkpoint_sha256='f'*64))
        except FileExistsError:pass
        else:raise AssertionError('changed checkpoint accepted by resume')
        (case_dir/'paired_arrays.npz').write_text('tampered')
        try:resume_report(resume_dir,identity)
        except ValueError:pass
        else:raise AssertionError('changed output accepted by resume')
        checks.append('strict resume: exact success accepted; changed checkpoint or output bytes rejected')

        # Real mixed-model path must predict sliders and inherit fixed-root rotation.
        from test_run_scaling_eval import metadata, MixedJointTree, FakeBody
        p=Path(td)/'mixed.json';metadata(p);tree=MixedJointTree(p);mixed=FakeBody(tree)
        base_geom=mixed.geom
        def geom(B):
            g=base_geom(B);J=len(mixed.names);off=tree.offsets.float()[None].expand(B,-1,-1)
            g.update(rest_off=off,bonelen=off.norm(dim=-1)/mixed.reach,depth=torch.tensor([0,1,2,2,2])[None].expand(B,-1),
                hop=torch.ones(B,J,J,dtype=torch.long),sem=torch.zeros(B,J,1024),dof=torch.zeros(B,J,19),
                metadata_missing=torch.zeros(B,J,2),pose_scale=torch.full((B,),mixed.reach))
            return g
        mixed.geom=geom;mw={'pose9':mixed.arrays['pose9'][:3],'frame_index':np.arange(3)}
        for variant in ('A','B'):
            model=model_factory(cfg,variant).eval()
            pred,outputs,_=predict_window(model,mixed,mixed,mw,mw,'pure_se3',[2,3],batch_size=2)
            assert np.all((outputs['slide_native'][:,2:4]>0)&(outputs['slide_native'][:,2:4]<4))
            assert np.allclose(pred.rotations[:,0],np.eye(3))
        checks.append('A/B real mixed slider prediction: no target q passed; bounds and fixed root honored')
    print(json.dumps({'status':'passed','execution':'CPU synthetic fixtures only; no real checkpoint/data read','checks':checks},indent=2))


if __name__=='__main__':main()
