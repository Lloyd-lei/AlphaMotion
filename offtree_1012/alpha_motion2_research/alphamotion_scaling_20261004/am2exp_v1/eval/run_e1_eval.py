"""E1 inference runner: frozen case list -> one npz per case (interfaces.md "Eval output").

  python -m am2exp_v1.eval.run_e1_eval --arm A0 --checkpoint ROOT/runs/A_S_seed0/checkpoint_step3000.pt \
      --cases OUT/eval/cases_e1_v1.json --mode sensor_on --out OUT/eval/e1/A0_s3000/sensor_on [--shard 0/4] [--shuffle-source]

Model registry (no training, CPU by default):
  * A0 / B0 / any checkpoint whose config has group A|B and no am2exp block -> the ORIGINAL classes through the sealed
    evaluation/run_scaling_eval.model_factory (ConditionalMegatron / SurfaceMegatron(max_patches=96)).
  * am2exp arms (checkpoint config carries cfg['am2exp']['registry_entry'] written by CODE/train/train_arm.py) ->
    the class named by cfg['model_kind'] ('module:attr', e.g. am2exp_v1.model.dense_surface:DenseSurfaceMegatron), the
    registry entry's model_kwargs and geometry hook (dense surfaces), judge body order from the config.
  * --model-class NAME overrides with a class from CODE/model/registry.MODEL_CLASSES (by class name).

Inputs per case are formed exactly like evaluation/run_locomotion_eval.predict_window(full_source): source pose9 of the
window (all joints visible), no condition, and
  sensor_on : source_contact = contact_sdist recomputed from the window's gap with the cache header thr/D and fp16
              rounding (run_locomotion_eval.load_window, which also verifies it bit-exactly against the sealed payload
              rows); models that accept `source_root_height_m` (arm I) get the source root height where the sensor is
              visible (trusted, finite root height and any finite contact), NaN elsewhere.
  sensor_off: source_contact=None, source_root_height_m all NaN.
--shuffle-source replaces every case's source window by the window of a DIFFERENT clip (fixed cyclic rule within the
same window family) while keeping the case's GT: the shuffled-source floor.
Saved per case: pred_rot6, pred_joint_position (cm), pred_joint_rotation, pred_root_height_m (raw_hpos[:,0,0]),
pred_contact_logits, gt_* (target window), source_pose9, source labels, the exact sensor inputs fed, meta json.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from am2exp_v1.eval.cases import load_cases, CASES_FILE  # noqa: E402

SCHEMA = 'am2exp_e1_case_v1'
WINDOW_FAMILIES = {'unseen_motion_listed': 'motion', 'unseen_motion_mined': 'motion', 'guard': 'motion',
                   'unseen_body': 'body', 'rootbias_train': 'rootbias'}


def _sealed():
    sys.dont_write_bytecode = True
    for p in [ROOT / 'evaluation', ROOT / 'data', ROOT / 'prismatic', ROOT / 'base' / 'legacy', ROOT / 'base', ROOT / 'orchestration']:
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    import run_locomotion_eval as RLE  # noqa: E402  (sealed, read-only)
    import run_scaling_eval as RSE  # noqa: E402
    return RLE, RSE


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def load_symbol(spec):
    import importlib
    mod, _, attr = spec.partition(':')
    return getattr(importlib.import_module(mod), attr)


def build_model(checkpoint, device, model_class=None):
    """-> (model, config, info). See module docstring for the registry rules."""
    import torch
    RLE, RSE = _sealed()
    payload = RSE.load_evaluation_checkpoint(checkpoint)
    run = Path(checkpoint).resolve().parent
    config = payload['config'] or json.loads((run / 'config.json').read_text())
    am2 = config.get('am2exp')
    geometry_record = None
    if model_class is None and am2 is None:
        variant = RSE.model_variant(config)
        model = RSE.model_factory(config, variant)
        cls_name = {'A': 'pilot_model:ConditionalMegatron', 'B': 'surface.surface_model:SurfaceMegatron(max_patches=96)'}[variant]
        extra_kwargs, geo = {}, None
    else:
        from am2exp_v1.model import _sealed as _ms  # noqa: F401  (installs sealed paths for the package)
        from am2exp_v1.model.registry import MODEL_CLASSES
        entry = (am2 or {}).get('registry_entry') or {}
        ex = entry.get('extra') or {}
        if model_class is not None:
            cls = MODEL_CLASSES[model_class]; cls_name = model_class
        else:
            cls_name = config['model_kind']; cls = load_symbol(cls_name)
        extra_kwargs = dict(ex.get('model_kwargs') or {})
        model = cls(**config['model'], **extra_kwargs)
        geo = ex.get('geometry')
        variant = 'am2exp'
    if config.get('judge_body_order') is not None and hasattr(model, 'set_judge_body_order'):
        model.set_judge_body_order(config['judge_body_order'])
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    info = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint), step=payload['step'], variant=variant,
                model_class=cls_name, model_kwargs_extra=extra_kwargs, geometry=geo, run_id=config.get('run_id', run.name))
    return model, config, info


def load_bodies(config, device, geometry=None):
    _sealed()
    from scaling_data import load_scaling_bodies
    bodies, audit = load_scaling_bodies(ROOT, config, device, include_surface=True)
    record = None
    if geometry:
        from am2exp_v1.model import _sealed as _ms  # noqa: F401
        hook = load_symbol(geometry['hook'])
        record = hook(bodies, device=device, **(geometry.get('params') or {}))
    return {b.name: b for b in bodies}, audit, record


def window_entry(w):
    return {'dataset': 'XWBC', 'group_id': w['group_id'], 'source_frames': int(w['source_frames']),
            'start': int(w['start']), 'frames': int(w['frames'])}


def shuffle_map(windows):
    """window_id -> donor window_id from a different clip, cyclic within the window family (frozen rule)."""
    out = {}
    fam = {}
    for w in windows:
        fam.setdefault(WINDOW_FAMILIES[w['kind']], []).append(w)
    for ws in fam.values():
        n = len(ws)
        for i, w in enumerate(ws):
            for k in range(1, n):
                d = ws[(i + k) % n]
                if d['clip'] != w['clip']:
                    out[w['window_id']] = d['window_id']; break
            else:
                raise ValueError('cannot shuffle a family with a single clip')
    return out


def accepts(model, name):
    return name in inspect.signature(model.forward_kinematic).parameters


def predict_case(model, source, target, sw, mode, device):
    """One forward over the whole window (B = T)."""
    import torch
    T = len(sw['frame_index'])
    x = torch.as_tensor(sw['pose9'], device=device, dtype=torch.float32)
    hidden = torch.zeros(x.shape[:2], device=device, dtype=torch.bool)
    contact = None
    rh = np.full(T, np.nan, np.float32)
    kwargs = {}
    if mode == 'sensor_on':
        c = np.where(sw['contact_valid'], sw['contact_sdist'], np.nan).astype(np.float32)
        contact = torch.as_tensor(c, device=device)
        ok = sw['root_height_valid'] & sw['ground_valid'] & np.isfinite(sw['root_height_m']) & np.isfinite(c).any(1)
        rh = np.where(ok, sw['root_height_m'], np.nan).astype(np.float32)
    elif mode != 'sensor_off':
        raise ValueError(mode)
    if accepts(model, 'source_root_height_m'):
        kwargs['source_root_height_m'] = torch.as_tensor(rh, device=device)
    with torch.inference_mode():
        out = model.forward_kinematic(x, source.geom(T), target.geom(T), source_hidden=hidden, source_contact=contact,
                                      condition=None, source_slide_normalized=None, **kwargs)
    res = {k: out[k].detach().float().cpu().numpy() for k in ('rot6', 'joint_position', 'joint_rotation', 'raw_hpos', 'contact_logits')}
    fed = dict(source_contact_input=np.full(x.shape[:2], np.nan, np.float32) if contact is None else contact.cpu().numpy(),
               source_root_height_input_m=rh, source_root_height_supplied=np.array('source_root_height_m' in kwargs))
    return res, fed


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    RLE, RSE = _sealed()
    cases_doc = load_cases(args.cases)
    windows = {w['window_id']: w for w in cases_doc['windows']}
    cases = cases_doc['cases']
    if args.only_primary:
        cases = [c for c in cases if c['primary']]
    if args.sources:
        cases = [c for c in cases if c['source'] in args.sources.split(',')]
    if args.targets:
        cases = [c for c in cases if c['target'] in args.targets.split(',')]
    k, n = (int(x) for x in args.shard.split('/'))
    cases = [c for i, c in enumerate(cases) if i % n == k]
    if args.limit:
        cases = cases[:args.limit]
    donor = shuffle_map(cases_doc['windows']) if args.shuffle_source else {}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, config, info = build_model(args.checkpoint, device, args.model_class)
    by, audit, georec = load_bodies(config, device, info['geometry'])
    manifest = json.loads((ROOT / 'data/data_manifest_v2.json').read_text())
    load_s = time.time() - t0
    cache = {}

    def read(window_id, body):
        key = (window_id, body)
        if key not in cache:
            if len(cache) > 48:
                cache.pop(next(iter(cache)))
            cache[key] = RLE.load_window(manifest, window_entry(windows[window_id]), by[body], with_contact=True)
        return cache[key]

    done = 0; skipped = 0; t_inf = 0.0
    # process cases window-by-window to reuse reads
    cases = sorted(cases, key=lambda c: (c['window_id'], c['source'], c['target']))
    for c in cases:
        path = out / f"{c['case_id']}.npz"
        if path.exists() and not args.overwrite:
            skipped += 1; continue
        src_w = donor.get(c['window_id'], c['window_id'])
        sw = read(src_w, c['source']); tw = read(c['window_id'], c['target'])
        hw = read(c['window_id'], c['source']) if src_w != c['window_id'] else sw  # labels of the case's own source window
        ti = time.time()
        res, fed = predict_case(model, by[c['source']], by[c['target']], sw, args.mode, device)
        t_inf += time.time() - ti
        meta = dict(schema=SCHEMA, case=c, window=windows[c['window_id']], mode=args.mode, shuffled_source=bool(args.shuffle_source),
                    source_window_id=src_w, arm=args.arm, model=info, cases_sha256=cases_doc['sha256'],
                    source_body=c['source'], target_body=c['target'], device=str(device),
                    target_joint_names=list(by[c['target']].names), source_joint_names=list(by[c['source']].names),
                    root_height_unit='m', position_unit='cm', rotation='AM2 own-joint global (data convention)')
        arrays = dict(
            pred_rot6=res['rot6'], pred_joint_position=res['joint_position'], pred_joint_rotation=res['joint_rotation'],
            pred_root_height_m=res['raw_hpos'][:, 0, 0], pred_contact_logits=res['contact_logits'].reshape(len(res['rot6']), -1),
            gt_rot6=tw['global_rot6'].astype(np.float32), gt_joint_position=tw['position_native'].astype(np.float32),
            gt_joint_rotation=tw['global_rotation_matrices'].astype(np.float32),
            gt_root_height_m=np.asarray(tw['root_height_m'], np.float32), gt_root_height_valid=np.asarray(tw['root_height_valid']),
            gt_contact_hard=tw['contact_hard'], gt_contact_valid=tw['contact_valid'],
            source_pose9=sw['pose9'].astype(np.float32), source_root_height_m=np.asarray(sw['root_height_m'], np.float32),
            source_label_contact_hard=hw['contact_hard'], source_label_contact_valid=hw['contact_valid'],
            frame_index=tw['frame_index'], source_frame_index=sw['frame_index'], meta=np.array(json.dumps(meta)), **fed)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        done += 1
    summary = dict(arm=args.arm, mode=args.mode, shard=args.shard, cases=len(cases), written=done, skipped_existing=skipped,
                   shuffle_source=bool(args.shuffle_source), model=info, cases_file=str(args.cases), cases_sha256=cases_doc['sha256'],
                   load_seconds=load_s, inference_seconds=t_inf, wall_seconds=time.time() - t0, threads=args.threads,
                   geometry_record=georec if isinstance(georec, (dict, list, type(None))) else str(georec))
    (out / f"run_manifest_shard{k}of{n}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k2: v for k2, v in summary.items() if k2 != 'geometry_record'}, default=str), flush=True)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cases', default=str(CASES_FILE)); ap.add_argument('--mode', choices=['sensor_on', 'sensor_off'], required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shuffle-source', action='store_true'); ap.add_argument('--model-class')
    ap.add_argument('--only-primary', action='store_true'); ap.add_argument('--sources'); ap.add_argument('--targets')
    ap.add_argument('--overwrite', action='store_true')
    return ap


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
