"""am2exp arm trainer: a clone of the orchestration/train_scaling.py main loop for ONE registered arm.

    python train_arm.py --arm NAME [--max-steps N] [--dry-run-cpu] [--lease-fd FD] ...

Never calls ``original.main()`` or ``original.guard()``. ``import train_scaling as original`` and
reuses load_scaling_bodies, backward_training_step, finite_backward, ComputeCounter,
effective_tokens, checkpoint, evaluate, loss_function, forward, autocast, digest exactly.

What differs from the original loop (all declared in the arm registry, train/arms.json):
  * model class by name (sealed ``pilot_model:ConditionalMegatron`` or ``am2exp_v1.model.*``),
    optional geometry hook (dense surfaces), optional init checkpoint (E2 fine-tunes);
  * objective wrapper: slide-fix loss replaces the slide L1 (old L1 still logged at weight 0),
    relational objective added on XWBC targets; corrections are target-side only;
  * input wrappers (arm I): rebind recipe_training.make_inputs / make_reference_inputs and
    original.make_inputs; they ADD keys only and provably consume no numpy RNG (state asserted);
  * domain mixtures: original (xwbc/unimate), E2 (xwbc/unimate/franka_extra/hand_gripper),
    franka_only (V1); optional side batches on a private numpy RNG under torch.random.fork_rng;
  * per-step sampling_sha256 compared with a reference log BEFORE the optimizer update; the run
    aborts on the first mismatch;
  * lean logs (full record every 10 steps + slim per-step hash/time line), weights-only
    checkpoints, final checkpoint without Adam, receipts (identity/state/exit_status/RSS/free -g),
    two-partition guard, GPU lease (own or inherited from the queue), wall-clock lifetime.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if '--dry-run-cpu' in sys.argv and __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # before torch is imported anywhere

from am2exp_v1.common.paths import ROOT, CODE, OUT, APPROVALS, add_sealed_paths  # noqa: E402
from am2exp_v1.ops import receipts as R  # noqa: E402
from am2exp_v1.ops import guard as G  # noqa: E402
from am2exp_v1.ops.lease import GpuLease  # noqa: E402

REGISTRY = CODE / 'train' / 'arms.json'
SMOKE_FREE_STEPS = 12          # GPU runs above this many steps need an approval file
MIX_ORDER = ('xwbc', 'unimate', 'franka_extra', 'hand_gripper')


class SamplingMismatch(RuntimeError):
    pass


class ApprovalRequired(RuntimeError):
    pass


class LifetimeExceeded(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------
def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _subst(v, ctx):
    if isinstance(v, str):
        for k, val in ctx.items():
            v = v.replace('${' + k + '}', str(val))
        return v
    if isinstance(v, list):
        return [_subst(x, ctx) for x in v]
    if isinstance(v, dict):
        return {k: _subst(x, ctx) for k, x in v.items()}
    return v


def load_registry(path=None):
    return json.loads(Path(path or REGISTRY).read_text())


def resolve_arm(name, registry=None, overrides=None):
    reg = registry if registry is not None else load_registry()
    if name not in reg['arms']:
        raise KeyError(f'unknown arm {name!r}; registered: {sorted(reg["arms"])}')
    spec = deep_merge(reg.get('defaults', {}), reg['arms'][name])
    if overrides:
        spec = deep_merge(spec, overrides)
    spec['name'] = name
    return _subst(spec, {'OUT': OUT, 'ROOT': ROOT, 'CODE': CODE, 'ARM': name})


def load_symbol(spec):
    """'module:attr' -> object. Sealed modules by bare name (after add_sealed_paths());
    am2exp_v1 code by its prefixed package name, e.g. 'am2exp_v1.model.slide_fix:SFMegatron'."""
    mod_name, _, attr = spec.partition(':')
    if not attr:
        raise ValueError(f'symbol spec needs module:attr, got {spec!r}')
    add_sealed_paths()
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        owner = ('WP-MODEL (CODE/model/*.py)' if '.model.' in mod_name else
                 'WP-E2 (CODE/e2/xbody_loss.py: xbody_loss(model, bodies, batch_rng_state, cfg))' if '.e2.' in mod_name else 'its owner')
        raise ImportError(f'arm registry symbol {spec!r}: cannot import {mod_name!r} ({exc}). '
                          f'This module is provided by {owner}; it is not present yet or failed to import. '
                          f'Import am2exp code with the am2exp_v1. prefix (CODE/common and CODE/model clash with '
                          f'base/legacy/common.py and model.py).') from exc
    obj = mod
    for part in attr.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ImportError(f'arm registry symbol {spec!r}: {mod_name} has no attribute {part!r}') from exc
    return obj


# --------------------------------------------------------------------------------------------
# sealed modules (read-only)
# --------------------------------------------------------------------------------------------
_SEALED = {}


def sealed():
    if not _SEALED:
        add_sealed_paths()
        import train_scaling as original  # noqa: WPS433 (sealed, read-only)
        import recipe_training
        import pilot_data
        _SEALED.update(original=original, recipe_training=recipe_training, pilot_data=pilot_data)
    return _SEALED['original'], _SEALED['recipe_training'], _SEALED['pilot_data']


# --------------------------------------------------------------------------------------------
# config, data identity, references
# --------------------------------------------------------------------------------------------
def build_config(arm):
    ex = arm['extra']
    base = json.loads((ROOT / ex['base_config']).read_text())
    cfg = copy.deepcopy(base)
    cfg['seed'] = int(arm['seed'])
    cfg['steps'] = int(arm['steps'])
    cfg['checkpoint_steps'] = sorted(int(s) for s in ex.get('checkpoints') or [])
    cfg['optimizer'] = {**cfg['optimizer'], 'lr': float(ex.get('lr', cfg['optimizer']['lr']))}
    if ex.get('max_training_seconds'):
        cfg['max_training_seconds'] = float(ex['max_training_seconds'])
    for dotted, value in (ex.get('config_overrides') or {}).items():
        node = cfg
        keys = dotted.split('.')
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    cfg['run_id'] = arm['name']
    cfg['size'] = arm['size']
    cfg['model_kind'] = arm['model_class']
    cfg['train_allowed'] = 'am2exp_v1 arm registry; approval file gates long GPU runs'
    cfg['am2exp'] = {'arm': arm['name'], 'parent': arm.get('parent'), 't_max': int(ex['t_max']),
                     'base_config': ex['base_config'], 'registry_entry': arm,
                     'never_called': ['train_scaling.main', 'train_scaling.guard']}
    if cfg['group'] != 'A':
        raise ValueError('am2exp arms are built on the A recipe (group A); B surface losses are not cloned here')
    return cfg


def _strip_fk_audit(audit):
    a = copy.deepcopy(audit)
    for b in a.get('bodies', []):
        b.pop('fk_max_error_over_reach', None)
    for b in a.get('upstream_data_audit', {}).get('body_audit', []):
        b.pop('fk_max_error_over_reach', None)
    return a


def data_identity(original, audit, cfg):
    sha = original.digest(audit)
    if sha == cfg['data_manifest_sha256']:
        return {'status': 'exact', 'data_manifest_sha256': sha}
    recorded = json.loads((ROOT / 'data_manifest.json').read_text())
    if original.digest(recorded) != cfg['data_manifest_sha256']:
        raise RuntimeError('ROOT/data_manifest.json no longer matches the config data_manifest_sha256')
    if original.digest(_strip_fk_audit(audit)) != original.digest(_strip_fk_audit(recorded)):
        raise RuntimeError(f'loaded data differs from the sealed data manifest ({sha})')
    worst = max(float(b['fk_max_error_over_reach']) for b in audit['bodies'])
    if worst > 2e-5:
        raise RuntimeError('FK audit error above the sealed 2e-5 tolerance')
    return {'status': 'equal_except_device_dependent_fk_audit_float', 'data_manifest_sha256': sha,
            'recorded_sha256': cfg['data_manifest_sha256'], 'max_fk_error_over_reach': worst,
            'note': 'only fk_max_error_over_reach differs (CPU vs CUDA float); all inputs/hashes identical'}


def load_reference(path):
    """step -> {'sampling_sha256', 'domain', 'total', 'device'} from a train.jsonl (full) or *_slim.jsonl.

    Sealed runs (ROOT/runs/*/train.jsonl) were all CUDA runs and carry no device field -> 'cuda'.
    am2exp slim logs record the device per line.  A reference line written at an abort (match False)
    is never a valid reference: such a file is rejected.
    """
    path = Path(path)
    if not path.exists():
        return None
    ref = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('match') is False:
                raise RuntimeError(f'reference {path} contains a sampling-mismatch abort line (step {r.get("step")})')
            ref[int(r['step'])] = {'sampling_sha256': r['sampling_sha256'], 'domain': r.get('domain'),
                                   'total': (r.get('terms') or {}).get('total', r.get('total')),
                                   'device': r.get('device', 'cuda')}
    return ref


def compare_sampling(step, domain, sampling, ref, device_type):
    """-> (match, reason). match is True/False, or None when the hash is not comparable.

    XWBC reference-mask sampling draws torch.rand/multinomial ON THE TRAINING DEVICE
    (recipe_training.make_reference_inputs), and CPU and CUDA generators differ, so an XWBC step
    hash is only comparable between runs on the same device type.  Every other domain (unimate,
    franka_*, hand_gripper) hashes numpy draws only and is always comparable.  The domain itself
    comes from the numpy stream and must always agree.
    """
    if ref is None:
        return None, 'no_reference_line'
    if ref.get('domain') is not None and ref['domain'] != domain:
        return False, f'domain {domain} != reference domain {ref["domain"]}'
    if domain == 'xwbc' and ref.get('device', 'cuda') != device_type:
        return None, f'xwbc_unverifiable_{device_type}_vs_{ref.get("device", "cuda")}'
    if sampling == ref['sampling_sha256']:
        return True, 'match'
    return False, 'sampling_sha256 differs'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def code_hashes():
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        # only real absolute files (some extension modules report bare names like '_ops.py',
        # which would resolve against the cwd)
        if (f and os.path.isabs(f) and f.endswith('.py') and os.path.isfile(f)
                and str(Path(f).resolve()).startswith(str(CODE) + os.sep)):
            out[str(Path(f).resolve().relative_to(CODE))] = file_sha256(f)
    reg = CODE / 'train' / 'arms.json'
    out['train/arms.json'] = file_sha256(reg)
    return dict(sorted(out.items()))


def rng_state_sha(rng):
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# input wrappers (arm I and twins): add keys only, consume no numpy RNG
# --------------------------------------------------------------------------------------------
def _rng_snapshot(rng):
    return copy.deepcopy(rng.bit_generator.state)


@contextlib.contextmanager
def input_wrappers(arm, cfg):
    """Rebind recipe_training.make_inputs/make_reference_inputs and original.make_inputs."""
    original, recipe_training, _pd = sealed()
    inp = arm['extra'].get('inputs') or {}
    if not inp.get('relational_source'):
        yield {'installed': False}
        return
    import torch
    roll = bool(inp.get('roll_relations', False))
    # ONE definition of 'sensor visible' shared with the O-arm target masks (WP-MODEL ground.py):
    # root_height_valid & ground_valid & contact_trust & not DFaust/PosePrior-trust0 & any finite
    # source_contact (ties the sensor to the existing 0.3 contact dropout; consumes no RNG).
    sensor_fn = load_symbol(inp.get('sensor_fn', 'am2exp_v1.model.ground:source_root_height_sensor'))
    orig = {'rt_make_inputs': recipe_training.make_inputs, 'rt_make_reference_inputs': recipe_training.make_reference_inputs,
            'original_make_inputs': original.make_inputs}
    calls = {'make_inputs': 0, 'make_reference_inputs': 0, 'reference_rows': 0, 'reference_sensor_visible_rows': 0}

    def make_inputs(src, tgt, rows, rng, device, *args, **kwargs):
        inputs, target = orig['rt_make_inputs'](src, tgt, rows, rng, device, *args, **kwargs)
        before = _rng_snapshot(rng)
        # sparse / unimate / eval paths: no ground sensor (NaN), declared in final_plan.md
        inputs['source_root_height_m'] = torch.full((len(rows),), float('nan'), device=device)
        if roll:
            inputs['roll_relations'] = True
        if rng.bit_generator.state != before:
            raise RuntimeError('input wrapper consumed numpy RNG')
        calls['make_inputs'] += 1
        return inputs, target

    def make_reference_inputs(src, tgt, rows, rng, device, config):
        inputs, target = orig['rt_make_reference_inputs'](src, tgt, rows, rng, device, config)
        before = _rng_snapshot(rng)
        sensor = sensor_fn(src, rows, inputs.get('source_contact'), device)
        if sensor.shape != (len(rows),):
            raise ValueError('source_root_height_m must be [B]')
        inputs['source_root_height_m'] = sensor
        if roll:
            inputs['roll_relations'] = True
        if rng.bit_generator.state != before:
            raise RuntimeError('reference input wrapper consumed numpy RNG')
        calls['make_reference_inputs'] += 1
        calls['reference_rows'] += len(rows)
        calls['reference_sensor_visible_rows'] += int(torch.isfinite(sensor).sum())
        return inputs, target

    recipe_training.make_inputs = make_inputs
    recipe_training.make_reference_inputs = make_reference_inputs
    original.make_inputs = make_inputs
    try:
        yield {'installed': True, 'roll_relations': roll, 'calls': calls}
    finally:
        recipe_training.make_inputs = orig['rt_make_inputs']
        recipe_training.make_reference_inputs = orig['rt_make_reference_inputs']
        original.make_inputs = orig['original_make_inputs']


# --------------------------------------------------------------------------------------------
# objective wrapper (target side only)
# --------------------------------------------------------------------------------------------
class SlideLogitHook:
    """Captures the slide head output (logit) of the most recent forward for drift diagnostics."""

    def __init__(self, model):
        self.last = None
        mod = getattr(model, 'slide_head', None)
        self.handle = mod.register_forward_hook(self._hook) if mod is not None else None

    def _hook(self, _m, _inp, out):
        self.last = out[0] if isinstance(out, (tuple, list)) else out

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def _as_loss_terms(res, name):
    """Accept (loss, terms) tuples or dicts with 'total'/'loss' (+ other tensor terms)."""
    if isinstance(res, tuple) and len(res) == 2:
        return res[0], dict(res[1])
    if isinstance(res, dict):
        key = 'total' if 'total' in res else 'loss' if 'loss' in res else None
        if key is None:
            raise KeyError(f'{name} must return (loss, terms) or a dict with total/loss')
        return res[key], {k: v for k, v in res.items() if k not in (key, 'terms', 'diagnostics')} | dict(res.get('terms') or {})
    if hasattr(res, 'backward'):
        return res, {}
    raise TypeError(f'{name} returned {type(res)}')


def _call_slide_fix(fn, outputs, slide_target, slide_valid, tgt, rows):
    import inspect
    params = list(inspect.signature(fn).parameters)
    kw = {}
    pool = {'outputs': outputs, 'target': slide_target, 'slide_target': slide_target, 'targets': slide_target,
            'valid': slide_valid, 'slide_valid': slide_valid, 'tgt': tgt, 'body': tgt, 'rows': rows}
    positional = []
    for p in params:
        if p in pool:
            kw[p] = pool[p]
    if not kw:
        positional = [outputs, slide_target]
    return fn(*positional, **kw)


def make_objective(arm, cfg, diag, hook, model=None):
    import torch
    original, _rt, _pd = sealed()
    base = original.loss_function(cfg)
    oc = arm['extra'].get('objective') or {}
    slide_fix = bool(oc.get('slide_fix'))
    slide_fn = load_symbol(oc['slide_fix_loss']) if slide_fix else None
    rel = oc.get('relational')
    rel_fn = load_symbol(rel['fn']) if rel else None
    weight = float(oc.get('scale', 1.0))
    model_ref = [model] if model is not None else []

    def objective(outputs, target, tgt, rows, condition, weights):
        w = dict(weights)
        if slide_fix:
            w['slide_normalized'] = 0.0  # old L1 still computed and logged, contributes nothing
        loss, terms = base(outputs, target, tgt, rows, condition, w)
        sv = outputs['slide_valid']
        if bool(sv.any()):
            logits = outputs.get('slide_logit')
            if logits is None and isinstance(outputs.get('aux'), dict):
                logits = outputs['aux'].get('slide_logit')
            if logits is None:
                logits = hook.last
            with torch.no_grad():
                rec = {'target': tgt.name}
                if logits is not None:
                    lg = logits.detach().float().reshape(sv.shape)[sv]
                    s = torch.sigmoid(lg)
                    rec.update(logit_absmax=float(lg.abs().max()), logit_mean=float(lg.mean()),
                               exact_saturation_frac=float(((s == 1) | (s == 0)).float().mean()))
                pred = outputs['slide_normalized'].detach().float()[sv]
                m = float(getattr(model_ref[0], 'slide_margin', 0.0) or 0.0) if model_ref else 0.0
                lo, hi = -m, 1.0 + m
                rec['pred_at_bound_frac'] = float(((pred <= lo) | (pred >= hi)).float().mean())
                rec['pred_bounds'] = [lo, hi]
                st = tgt.tensor('slide_normalized', rows, pred.device).float()[sv]
                rec.update(pred_min=float(pred.min()), pred_max=float(pred.max()), target_max=float(st.max()),
                           l1=float((pred - st).abs().mean()))
            diag['slide'].append(rec)
            if slide_fix:
                st_full = tgt.tensor('slide_normalized', rows, sv.device).float()
                sl, sterms = _as_loss_terms(_call_slide_fix(slide_fn, outputs, st_full, sv, tgt, rows), 'slide_fix_loss')
                loss = loss + weights['slide_normalized'] * sl
                terms['slide_fix_loss'] = sl.detach() if hasattr(sl, 'detach') else loss.new_tensor(float(sl))
                for k, v in sterms.items():
                    key = k if (k.startswith('count_') or k not in terms) else 'slide_fix_' + k
                    terms[key] = v if torch.is_tensor(v) else loss.new_tensor(float(v))
        if rel_fn is not None and tgt.domain == 'xwbc':
            res = rel_fn(outputs, target, tgt, rows, rel.get('weights') or {}, roll=bool(rel.get('roll', False)))
            rl, rterms = _as_loss_terms(res, 'relational_objective')
            loss = loss + rl
            for k, v in rterms.items():
                key = k if (k.startswith('count_') or k not in terms) else 'rel_' + k
                terms[key] = v if torch.is_tensor(v) else loss.new_tensor(float(v))
        if weight != 1.0:
            loss = loss * weight
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite combined loss (am2exp objective)')
        return loss, terms
    return objective


# --------------------------------------------------------------------------------------------
# domain steps
# --------------------------------------------------------------------------------------------
def draw_domain(rng, cfg, mixture):
    if not mixture:
        return 'xwbc' if rng.random() < cfg['mixture']['xwbc'] else 'unimate'
    if mixture.get('franka_only'):
        return 'franka_only'  # V1: no domain draw at all (both V1 arms share the stream)
    probs = [(k, float(mixture[k])) for k in MIX_ORDER if mixture.get(k)]
    if abs(sum(p for _k, p in probs) - 1.0) > 1e-9:
        raise ValueError(f'mixture must sum to 1: {mixture}')
    u = rng.random()
    acc = 0.
    for k, p in probs:
        acc += p
        if u < acc:
            return k
    return probs[-1][0]


_RNG_PARAM_NAMES = ('rng', 'batch_rng_state', 'batch_rng', 'data_rng', 'generator')
_CFG_PARAM_NAMES = ('cfg', 'config', 'xcfg', 'e2cfg')


def call_xbody(fn, model, bodies_by_name, rng, xcfg):
    """Call WP-E2's cross-body loss whatever its argument order.

    interfaces/WP-OPS contract: ``xbody_loss(model, bodies, batch_rng_state, cfg)``; the WP-E2
    implementation (e2/xbody_loss.py) ships ``xbody_loss(model, bodies, cfg, rng)``.  Both are
    accepted: the rng and cfg arguments are bound by parameter NAME (rng/batch_rng_state/... and
    cfg/config), falling back to the contract order.  The rng is the run's main data stream
    (numpy Generator), consumed identically whatever w_xb/roll are (G0 == G1 == G1_roll sampling).
    """
    import inspect
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []
    rest = params[2:4]
    if len(rest) == 2 and rest[0] in _CFG_PARAM_NAMES and rest[1] in _RNG_PARAM_NAMES:
        return fn(model, bodies_by_name, xcfg, rng)
    return fn(model, bodies_by_name, rng, xcfg)


def _rows_sha(rows):
    import numpy as np
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def hand_gripper_step(model, bodies_by_name, rng, xb_fn, e2cfg, device, step):
    """One E2 'hand_gripper' domain step: backprop w_xb * L_xb (graph-connected also at w_xb = 0).

    The hashed auxiliary record contains ONLY sampling facts (rows drawn + data-RNG state after the
    draw) so G0 (w_xb 0), G1 (w_xb .2) and G1_roll (roll 32) produce identical per-step hashes;
    w_xb/roll are logged in report['e2'], which is not hashed.
    """
    import torch
    original, _rt, _pd = sealed()
    info = {}
    bs = int(e2cfg.get('batch', e2cfg.get('batch_size', 64)))
    xcfg = {k: v for k, v in e2cfg.items() if k not in ('fn',)}
    xcfg.update(batch=bs, batch_size=bs, device=device, autocast_fn=original.autocast, autocast=original.autocast,
                step=step, info=info)
    res = call_xbody(xb_fn, model, bodies_by_name, rng, xcfg)
    total, terms = _as_loss_terms(res, 'xbody_loss')
    if not torch.is_tensor(total) or not total.requires_grad:
        raise RuntimeError('xbody_loss total must be a graph-connected tensor (also when w_xb == 0)')
    total.backward()
    sampling = res.get('sampling') if isinstance(res, dict) else None
    metrics = {'total': float(total.detach())}
    for k, v in terms.items():
        if k == 'sampling' or k.startswith('count_'):
            continue
        try:
            key = k if k.startswith('xb_') else 'xb_' + k
            metrics[key] = float(v.detach()) if torch.is_tensor(v) else float(v)
        except (TypeError, ValueError, RuntimeError):
            pass
    rows = {}
    for key in ('rows_A', 'rows_F'):
        if info.get(key) is not None:
            rows[key + '_sha256'] = _rows_sha(info[key])
            rows[key + '_n'] = int(len(info[key]))
    aux = {'domain': 'hand_gripper', 'xbody_rows': rows or None, 'xbody_sampling': sampling,
           'data_rng_state_sha256': rng_state_sha(rng)}
    e2rec = {'w_xb': float(e2cfg.get('w_xb', 0.0)), 'roll': int(e2cfg.get('roll', 0)), 'batch': bs,
             'variant': info.get('variant'), 'tau_cm': info.get('tau_cm'),
             'counts': {k: float(v) for k, v in terms.items() if k.startswith('count_')}}
    return {'domain': 'hand_gripper', 'terms': metrics, 'directions': [], 'auxiliary': aux, 'e2': e2rec}


def weighted_objective(objective, w):
    def f(*a, **kw):
        loss, terms = objective(*a, **kw)
        return loss * w, terms
    return f


# --------------------------------------------------------------------------------------------
# main runner
# --------------------------------------------------------------------------------------------
def gpu_snapshot_safe(original):
    try:
        return original.gpu_snapshot()
    except Exception as exc:  # nvidia-smi missing in CPU tests
        return {'error': repr(exc)}


def write_final_checkpoint(model, step, run, cfg, audit_sha):
    import torch
    path = run / f'checkpoint_step{step:04d}.pt'
    tmp = path.with_suffix('.tmp.pt')
    obj = {'model': model.state_dict(), 'step': step, 'config': cfg, 'data_manifest_sha256': audit_sha,
           'torch_rng': torch.get_rng_state(), 'optimizer_included': False, 'final': True}
    torch.save(obj, tmp)
    tmp.replace(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha256(path), 'step': step,
            'optimizer_included': False, 'final': True}


def default_out_dir(arm, opts):
    if opts.out_dir:
        return Path(opts.out_dir)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if opts.dry_run_cpu:
        return OUT / 'runs' / '_dryrun' / f'{arm["name"]}_{stamp}'
    if opts.max_steps is not None and opts.max_steps < int(arm['steps']):
        return OUT / 'runs' / '_smoke' / f'{arm["name"]}_{opts.max_steps}steps_{stamp}'
    return Path(arm['output_dir'])


def run_arm(arm, opts):
    """Run one arm. ``opts`` is an argparse-like namespace (see parse_args)."""
    import numpy as np
    import torch
    original, recipe_training, _pd = sealed()
    cfg = build_config(arm)
    ex = arm['extra']
    steps_total = int(cfg['steps'])
    steps = steps_total if opts.max_steps is None else min(steps_total, int(opts.max_steps))
    device = torch.device('cpu' if opts.dry_run_cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for a real run (use --dry-run-cpu for the CPU pipeline check)')
    stage = ex.get('stage') or 'unstaged'
    approval_stage = getattr(opts, 'approval_stage', None) or stage
    approval = None
    if device.type == 'cuda' and steps > SMOKE_FREE_STEPS:
        ok = APPROVALS / f'{approval_stage}.ok'
        if not ok.exists():
            raise ApprovalRequired(f'{steps} GPU steps of arm {arm["name"]} need lloyd approval file {ok}')
        approval = {'path': str(ok), 'sha256': file_sha256(ok)}
    run = default_out_dir(arm, opts)
    if run.exists() and any(run.iterdir()):
        res = run / 'result.json'
        if res.exists() and json.loads(res.read_text()).get('status') == 'completed':
            print(json.dumps({'arm': arm['name'], 'status': 'already_completed', 'run': str(run)}), flush=True)
            return {'status': 'already_completed', 'run': str(run)}
        raise RuntimeError(f'unfinished existing run folder requires identity/exit review; no implicit restart: {run}')
    run.mkdir(parents=True, exist_ok=True)
    lifetime = float(opts.lifetime_seconds or ex.get('lifetime_seconds') or 0) or None
    deadline = time.monotonic() + lifetime if lifetime else None
    torch.set_num_threads(int(opts.threads))

    with R.ProcessReceipts(run, 'train_arm', extra={'arm': arm['name'], 'stage': stage, 'device': device.type,
                                                     'steps_planned': steps, 'steps_registered': steps_total,
                                                     'approval_stage': approval_stage, 'approval': approval},
                           rss_limit=int(opts.rss_limit_gib * R.GiB)) as rec:
        lease = None
        if device.type == 'cuda':
            lease = GpuLease.inherit(opts.lease_fd) if opts.lease_fd is not None else GpuLease().acquire()
            R.write_json(run / 'lease.json', lease.receipt)
        try:
            return _train(arm, opts, cfg, ex, run, rec, device, steps, deadline, original, recipe_training, np, torch)
        finally:
            if lease is not None:
                lease.release()


def _train(arm, opts, cfg, ex, run, rec, device, steps, deadline, original, recipe_training, np, torch):
    cuda = device.type == 'cuda'

    def sync():
        if cuda:
            torch.cuda.synchronize()

    rec.update(phase='guard')
    try:
        cap = G.check(out_allocation_bytes=int(ex.get('out_allocation_bytes', 0)))
    except G.CapacityError:
        if not opts.dry_run_cpu:
            raise
        cap = G.check(raise_on_fail=False)
    print(G.describe(cap), flush=True)
    R.write_json(run / 'capacity_start.json', cap)
    R.write_json(run / 'arm.json', arm)
    rec.update(phase='load_data')
    t0 = time.monotonic()
    bodies, audit = original.load_scaling_bodies(ROOT, cfg, device)
    ident = data_identity(original, audit, cfg)
    audit_sha = cfg['data_manifest_sha256']
    if audit['judge_body_order'] != cfg['judge_body_order']:
        raise RuntimeError('judge body order changed')
    by_name = {b.name: b for b in bodies}
    load_seconds = time.monotonic() - t0
    geo = ex.get('geometry')
    geometry_record = None
    if geo:
        hook_fn = load_symbol(geo['hook'])
        geometry_record = hook_fn(bodies, device=device, **(geo.get('params') or {}))
    # references
    ref_path = arm.get('reference_train_jsonl')
    assert_mode = ex.get('sampling_assert', True)
    reference = load_reference(ref_path) if ref_path else None
    if ref_path and assert_mode == 'required' and reference is None:
        raise RuntimeError(f'required sampling reference missing: {ref_path}')
    if reference is not None and assert_mode and max(reference) < steps:
        raise RuntimeError(f'reference {ref_path} covers {max(reference)} steps < {steps}')
    assert_on = bool(reference is not None and assert_mode)
    # model
    rec.update(phase='build_model')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    cls = load_symbol(arm['model_class'])
    model = cls(**cfg['model'], **(ex.get('model_kwargs') or {})).to(device)
    model.set_judge_body_order(cfg['judge_body_order'])
    init_record = None
    if ex.get('init_checkpoint'):
        payload = torch.load(ex['init_checkpoint'], map_location='cpu', weights_only=True)
        if payload.get('data_manifest_sha256') != audit_sha:
            raise RuntimeError('init checkpoint has a different data lineage')
        if ex.get('init_step') is not None and int(payload['step']) != int(ex['init_step']):
            raise RuntimeError(f'init checkpoint step {payload["step"]} != {ex["init_step"]}')
        model.load_state_dict(payload['model'], strict=True)
        init_record = {'path': ex['init_checkpoint'], 'sha256': file_sha256(ex['init_checkpoint']), 'step': int(payload['step'])}
        del payload
    hook = SlideLogitHook(model)
    diag = {'slide': []}
    objective = make_objective(arm, cfg, diag, hook, model)
    counter = original.ComputeCounter(model)
    # streams (identical to the original for seed-0 A-stream arms)
    st = ex.get('streams') or {}
    seeds = {'data': int(st.get('data', cfg['seed'])), 'combo': int(st.get('combo', 4)),
             'reenc': int(st.get('reenc', cfg['seed'] * 1000 + 77))}
    rng = np.random.default_rng(seeds['data'])
    combo = np.random.default_rng(seeds['combo'])
    reenc = np.random.default_rng(seeds['reenc'])
    sb = ex.get('side_batch')
    side_rngs = None
    if sb:
        side_rngs = [np.random.default_rng(int(sb['seed']) + k) for k in (0, 1, 2)]
    e2cfg = ex.get('e2') or {}
    xb_fn = load_symbol(e2cfg['fn']) if (ex.get('mixture') or {}).get('hand_gripper') else None
    franka = by_name.get('relay_franka_full_v2')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(ex['t_max']))
    sealed_code = original.source_hashes()
    with input_wrappers(arm, cfg) as wrap_info:
        own_code = code_hashes()
        R.write_json(run / 'config.json', cfg)
        R.write_json(run / 'source_snapshot.json', {'sealed': sealed_code, 'am2exp_v1': own_code})
        R.write_json(run / 'gpu_start.json', gpu_snapshot_safe(original) if cuda else {'device': 'cpu'})
        R.write_json(run / 'setup.json', {'data_identity': ident, 'load_seconds': load_seconds, 'streams': seeds,
                                          'reference': ref_path, 'reference_steps': max(reference) if reference else 0,
                                          'sampling_assert': assert_on, 'init': init_record, 'geometry': geometry_record,
                                          'input_wrappers': {k: v for k, v in wrap_info.items() if k != 'calls'},
                                          'model_class': arm['model_class'], 'steps': steps, 't_max': int(ex['t_max']),
                                          'parameter_count': sum(p.numel() for p in model.parameters())})
        fwd = original.forward

        def domain_step(domain):
            if domain in ('xwbc', 'unimate'):
                return original.backward_training_step(model, bodies, cfg, rng, combo, reenc, device=device, domain=domain,
                                                       forward_fn=fwd, objective_fn=objective, autocast_fn=original.autocast)
            if domain in ('franka_extra', 'franka_only'):
                r = original.backward_training_step(model, [franka], cfg, rng, combo, reenc, device=device, domain='unimate',
                                                    forward_fn=fwd, objective_fn=objective, autocast_fn=original.autocast)
                r['domain'] = domain
                return r
            if domain == 'hand_gripper':
                return hand_gripper_step(model, by_name, rng, xb_fn, {'batch_size': cfg['batch_size'], **e2cfg}, device, step)
            raise ValueError(domain)

        def side_step(step):
            if not sb or step % int(sb['every']):
                return None
            body = by_name[sb['body']]
            scfg = deep_merge(cfg, {'batch_size': int(sb['rows'])})
            if sb.get('modes'):
                scfg['mixture']['modes'] = sb['modes']
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(sb['seed']) * 1000 + step)
                r = original.backward_training_step(model, [body], scfg, *side_rngs, device=device, domain='unimate',
                                                    forward_fn=fwd, objective_fn=weighted_objective(objective, float(sb.get('weight', 1.0))),
                                                    autocast_fn=original.autocast)
            sha = original.digest([{k: d[k] for k in ['source', 'target', 'mode', 'selected_row_sha256', 'sampling_sha256']} for d in r['directions']])
            return {'sampling_sha256': sha, 'total': r['terms'].get('total'), 'mode': r['directions'][0]['mode']}

        started = time.monotonic()
        train_seconds = 0.
        flops = 0
        tokens = {}
        ckpts = []
        hashes = []
        step_times = {}
        mismatches = 0
        checked = 0
        unverifiable = 0
        checked_domains = {}
        compare = []
        peak_gpu = 0
        phase = 'training'
        step = 0
        rec.update(phase=phase, step=0)
        full_log = (run / 'train.jsonl').open('a')
        slim_log = (run / 'train_slim.jsonl').open('a')
        try:
            if not opts.no_eval and ex.get('initial_eval'):
                original.evaluate(model, bodies, cfg, run / 'eval', 'initial', quick=True)
            for step in range(1, steps + 1):
                if train_seconds > cfg['max_training_seconds']:
                    raise RuntimeError('bounded run time exhausted before fixed exposure')
                if deadline is not None and time.monotonic() > deadline:
                    raise LifetimeExceeded(f'wall-clock lifetime exceeded at step {step}')
                if rec.sampler is not None and rec.sampler.exceeded:
                    raise MemoryError(f'RSS above limit: {rec.sampler.report()}')
                if step % 100 == 0 and not opts.dry_run_cpu:
                    G.check(out_allocation_bytes=int(ex.get('out_allocation_bytes', 0)))
                if step == 1 or step % 1000 == 0:
                    if original.source_hashes() != sealed_code:
                        raise RuntimeError('sealed source changed during the run; no silent continuation')
                    changed = [k for k, v in own_code.items() if file_sha256(CODE / k) != v]
                    if changed:
                        raise RuntimeError(f'am2exp_v1 code changed during the run ({changed}); no silent continuation')
                if cuda and step % 1000 == 0:
                    R.write_json(run / f'gpu_step{step}.json', gpu_snapshot_safe(original))
                domain = draw_domain(rng, cfg, ex.get('mixture'))
                if opts.perturb_rng_at_step is not None and step == int(opts.perturb_rng_at_step):
                    rng.random()  # TEST HOOK: deliberately desynchronise the data stream
                torch.manual_seed(cfg['seed'] * 1000000 + step)
                opt.zero_grad(set_to_none=True)
                counter.reset()
                diag['slide'] = []
                sync()
                tick = time.monotonic()
                report = domain_step(domain)
                side = side_step(step)
                sync()
                fb_seconds = time.monotonic() - tick
                sampling = original.digest([{k: d[k] for k in ['source', 'target', 'mode', 'selected_row_sha256', 'sampling_sha256']}
                                            for d in report['directions']] + [report['auxiliary']])
                hashes.append(sampling)
                ref = reference.get(step) if reference else None
                match, match_reason = compare_sampling(step, report['domain'], sampling, ref, device.type)
                if match is not None:
                    checked += 1
                    checked_domains[report['domain']] = checked_domains.get(report['domain'], 0) + 1
                elif ref is not None:
                    unverifiable += 1
                if match is False:
                    mismatches += 1
                slide_rec = None
                if diag['slide']:
                    gn = [p.grad.detach().float().square().sum() for n, p in model.named_parameters()
                          if n.startswith('slide_head.') and p.grad is not None]
                    slide_rec = {'directions': diag['slide'],
                                 'slide_head_grad_norm': float(torch.stack(gn).sum().sqrt()) if gn else 0.0}
                if assert_on and match is False:
                    slim_log.write(json.dumps({'step': step, 'domain': report['domain'], 'device': device.type,
                                               'sampling_sha256': sampling,
                                               'reference_sampling_sha256': ref['sampling_sha256'], 'reference_domain': ref['domain'],
                                               'match': False, 'reason': match_reason}) + '\n')
                    slim_log.flush()
                    raise SamplingMismatch(f'sampling_sha256 mismatch at step {step}: {sampling} != reference {ref["sampling_sha256"]} '
                                           f'({match_reason}; domain {report["domain"]} vs reference {ref["domain"]}; reference '
                                           f'{ref_path}); aborting before the optimizer update')
                tick = time.monotonic()
                grad = original.finite_backward(model)
                opt.step()
                schedule.step()
                sync()
                elapsed = fb_seconds + time.monotonic() - tick
                train_seconds += elapsed
                step_times.setdefault(report['domain'], []).append(elapsed)
                compute = counter.report()
                flops += compute['estimated_dense_training_flops']
                actual = original.effective_tokens(report, by_name)
                if cfg['group'] == 'A':
                    for key in list(actual):
                        if 'surface' in key:
                            actual[key] = 0
                for key, value in actual.items():
                    if isinstance(value, int):
                        tokens[key] = tokens.get(key, 0) + value
                total = report['terms'].get('total')
                slim = {'step': step, 'domain': report['domain'], 'device': device.type, 'sampling_sha256': sampling,
                        'match': match,
                        'total': total, 'gradient_norm_before_clip': grad, 'lr': opt.param_groups[0]['lr'],
                        'step_train_seconds': elapsed, 'cumulative_train_seconds': train_seconds,
                        'cumulative_wall_seconds': time.monotonic() - started}
                if ref is not None and ref.get('total') is not None and total is not None:
                    slim['reference_total'] = ref['total']
                    slim['total_abs_diff'] = abs(total - ref['total'])
                    compare.append((step, total, ref['total']))
                if match is None and ref is not None:
                    slim['match_reason'] = match_reason
                if slide_rec:
                    slim['slide'] = slide_rec
                if side:
                    slim['side'] = side
                if report.get('e2'):
                    slim['e2'] = report['e2']
                    slim['xb'] = {k: v for k, v in report['terms'].items() if k.startswith('xb_')}
                slim_log.write(json.dumps(slim, allow_nan=False) + '\n')
                if step == 1 or step % 10 == 0 or step == steps:
                    record = {'step': step, **report, 'sampling_sha256': sampling, 'reference_match': match,
                              'gradient_norm_before_clip': grad, 'lr': opt.param_groups[0]['lr'],
                              'step_train_seconds': elapsed, 'cumulative_train_seconds': train_seconds,
                              'cumulative_wall_seconds': time.monotonic() - started, 'compute': compute,
                              'cumulative_estimated_training_flops': flops, 'effective_tokens': actual,
                              'cumulative_tokens': dict(tokens), 'slide': slide_rec, 'side': side}
                    full_log.write(json.dumps(record, allow_nan=False) + '\n')
                    full_log.flush()
                    slim_log.flush()
                if step == 1 or step % 100 == 0:
                    if cuda:
                        peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated())
                    rec.update(phase='training', step=step, train_seconds=train_seconds, peak_cuda_bytes=peak_gpu,
                               sampling_checked=checked, sampling_mismatches=mismatches,
                               sampling_unverifiable=unverifiable)
                    print(json.dumps({'arm': arm['name'], 'step': step, 'domain': report['domain'], 'total': total,
                                      'match': match, 'train_seconds': round(train_seconds, 2)}), flush=True)
                if step in cfg['checkpoint_steps'] and step != steps and not opts.no_checkpoints:
                    ckpts.append(original.checkpoint(model, opt, step, run, cfg, audit_sha))
                    if not opts.no_eval and ex.get('quick_eval_at_checkpoints', True):
                        original.evaluate(model, bodies, cfg, run / 'eval', f'step{step}', quick=True)
            phase = 'final_checkpoint'
            rec.update(phase=phase, step=step)
            final = None if opts.no_checkpoints else write_final_checkpoint(model, step, run, cfg, audit_sha)
            if cuda:
                peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated())
            fe = ex.get('final_eval', 'full')
            if not opts.no_eval and fe in ('full', 'quick'):
                if deadline is not None and time.monotonic() > deadline:
                    raise LifetimeExceeded('lifetime exceeded before final evaluation')
                phase = 'final_eval'
                rec.update(phase=phase, step=step)
                original.evaluate(model, bodies, cfg, run / 'eval', 'final', quick=(fe == 'quick'))
            chain = hashlib.sha256('\n'.join(hashes).encode()).hexdigest()
            rel = [abs(a - b) / max(abs(b), 1e-8) for _s, a, b in compare]
            result = {
                'status': 'completed', 'arm': arm['name'], 'parent': arm.get('parent'), 'steps': step,
                'steps_registered': int(cfg['steps']), 'complete_exposure': step == int(cfg['steps']),
                'train_seconds': train_seconds, 'total_wall_seconds': time.monotonic() - started,
                'seconds_per_step_mean': train_seconds / max(step, 1),
                'seconds_per_step_median_by_domain': {k: statistics.median(v) for k, v in step_times.items()},
                'steps_by_domain': {k: len(v) for k, v in step_times.items()},
                'checkpoint': final, 'intermediate_checkpoints': ckpts,
                'sampling': {'chain_sha256': chain, 'reference': ref_path, 'assert_enabled': assert_on,
                             'steps_checked': checked, 'mismatches': mismatches,
                             'steps_unverifiable_cross_device_xwbc': unverifiable,
                             'all_checked_matched': checked > 0 and mismatches == 0,
                             'checked_by_domain': dict(checked_domains)},
                'loss_vs_reference': None if not compare else {
                    'steps': len(compare), 'max_abs_diff': max(abs(a - b) for _s, a, b in compare),
                    'max_rel_diff': max(rel), 'median_rel_diff': statistics.median(rel),
                    'first_step_abs_diff': abs(compare[0][1] - compare[0][2])},
                'peak_cuda_allocated_bytes': peak_gpu, 'rss': rec.sampler.report() if rec.sampler else None,
                'cumulative_estimated_training_flops': flops, 'cumulative_tokens': tokens,
                'data_manifest_sha256': audit_sha, 'data_identity': ident['status'], 'device': device.type,
                'input_wrappers': dict(wrap_info.get('calls') or {}) if wrap_info.get('installed') else None,
                'parameter_counts': {'main': sum(p.numel() for n, p in model.named_parameters() if not n.startswith('judge.')),
                                     'auxiliary': sum(p.numel() for n, p in model.named_parameters() if n.startswith('judge.')),
                                     'frozen_qwen': cfg['frozen_qwen_parameters']},
                'timing_scope': 'synchronized forward/backward + optimizer; sampling comparison, I/O and evaluation excluded',
                'physical_robot_or_dynamics_validation': False}
            R.write_json(run / 'result.json', result)
            phase = 'done'
            rec.update(phase=phase, step=step)
            rec.mark_completed()
            print(json.dumps({'arm': arm['name'], 'status': 'completed', 'steps': step, 'run': str(run),
                              'sampling': result['sampling'], 'loss_vs_reference': result['loss_vs_reference']}), flush=True)
            return result
        except BaseException as exc:
            R.write_json(run / 'failure.json', {'phase': phase, 'step': step, 'error': repr(exc), 'type': type(exc).__name__,
                                                'traceback': traceback.format_exc(), 'utc': R.utc()})
            rec.update(phase=phase, step=step)
            raise
        finally:
            full_log.close()
            slim_log.close()
            counter.close()
            hook.close()
            del model, opt, schedule
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', required=True)
    ap.add_argument('--registry', default=str(REGISTRY))
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--dry-run-cpu', action='store_true', help='CPU pipeline check: no lease, no GPU, no eval, own folder')
    ap.add_argument('--lease-fd', type=int, default=None, help='GPU lease fd inherited from ops/queue.py')
    ap.add_argument('--approval-stage', default=None,
                    help='approval file key checked for > 12 GPU steps (default: the arm stage; the queue passes the job stage)')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--no-checkpoints', action='store_true', help='write no checkpoint files (default for --dry-run-cpu)')
    ap.add_argument('--keep-checkpoints', action='store_true', help='with --dry-run-cpu: still write checkpoints')
    ap.add_argument('--lifetime-seconds', type=float, default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--rss-limit-gib', type=float, default=20.0)
    ap.add_argument('--perturb-rng-at-step', type=int, default=None, help='TEST ONLY: desynchronise the data RNG')
    ap.add_argument('--override', default=None, help='JSON deep-merged into the registry entry (tests/probes)')
    args = ap.parse_args(argv)
    if args.dry_run_cpu:
        args.no_eval = True
        args.no_checkpoints = not args.keep_checkpoints
        if args.max_steps is None:
            args.max_steps = 5
    return args


def main(argv=None):
    args = parse_args(argv)
    arm = resolve_arm(args.arm, load_registry(args.registry), json.loads(args.override) if args.override else None)
    run_arm(arm, args)


if __name__ == '__main__':
    main()
