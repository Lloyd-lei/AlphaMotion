"""E2 cross-body opposition loss for the trainer (WP-OPS 'hand_gripper' domain).

Signature (stable contract; the trainer's order is the canonical one)::

    res = xbody_loss(model, bodies, rng, cfg)      # train/train_arm.py hand_gripper_step order
    res = xbody_loss(model, bodies, cfg, rng)      # also accepted (detected by type: rng is a numpy Generator)
    res['total']     graph-connected scalar tensor = w_xb * L_xb  (also when w_xb == 0)
    res['terms']     dict of detached scalar tensors (xb_* and count_xb_rows_*)
    res['sampling']  {'rows_A_sha256', 'rows_F_sha256', 'batch', 'roll', 'variant'}  -- identical for G0/G1
    loss, terms = res['total'], res['terms']        (train_arm._as_loss_terms accepts this dict as is)

model   ConditionalMegatron-compatible module: forward_kinematic(source, source_geom, target_geom,
        source_hidden=None, source_contact=None, condition=None, source_slide_normalized=None) returning
        at least 'slide_native' (cm), 'joint_position' (cm) and 'joint_rotation' (global, rest-relative).
bodies  dict name -> pilot_data.Body (or an iterable of Body); needs 'adroit_pen' and 'relay_franka_full_v2'
        exactly as orchestration/scaling_data.load_scaling_bodies builds them.  Only TRAIN rows are drawn.
rng     numpy.random.Generator owned by the caller.  Consumption is fixed by the batch size only
        (pilot_data.sample_rows(A, rng, B) then sample_rows(F, rng, B)), never by w_xb or roll.
cfg     dict (all optional except w_xb):
          w_xb (float, REQUIRED; 0.0 for the G0 control -> rows drawn, zero-weight loss still in the graph)
          batch_size | batch (int rows per direction, default 64)
          roll (int, default 0; 32 for the G1_roll twin: targets rolled by `roll` rows inside the batch)
          huber_delta_cm 0.5, clip_cm 7.8, facing_margin 0.5, facing_weight 0.5
          variant / tau_cm (default: e2/opposition_frozen.json)
          directions: 'both' | 'AF' | 'FA'  (default 'both')
          autocast | autocast_fn: callable(device) -> context manager around the two model forwards
          device: torch device (default: model's device)
          info: optional dict; receives 'rows_A'/'rows_F' (numpy int64 rows drawn)
          (other keys such as 'fn', 'step' are ignored)

Loss  L_xb = Huber_d(c_F(T_F(x_A)) - clip(c_A(x_A), 0, 7.8))
             + Huber_d(c_A(T_A(x_F)) - c_F(x_F)) + 0.5 * relu(0.5 - f(T_A(x_F)))
      total = w_xb * L_xb
A->F: Adroit source pose9, no condition, no hidden joints, no contact.  F->A: Franka source pose9 + its visible
slide_normalized (the model requires it for visible prismatic sources).  Never conditions on Franka joints 8/9.
Targets come from the SOURCE ground truth through the frozen descriptor; no paired data.  c_F uses the RAW
(unclamped) slide_native, so the margin-sigmoid head is never cut off from gradient.
"""
from __future__ import annotations
import contextlib
import hashlib

import numpy as np
import torch

from . import opposition as op

ADROIT = 'adroit_pen'
FRANKA = 'relay_franka_full_v2'
DEFAULTS = {'batch_size': 64, 'roll': 0, 'huber_delta_cm': 0.5, 'clip_cm': op.FRANKA_MAX_CLEARANCE_CM, 'facing_margin': 0.5,
            'facing_weight': 0.5, 'directions': 'both'}


def _bodies(bodies):
    if isinstance(bodies, dict):
        return bodies
    return {b.name: b for b in bodies}


def _sample_rows(body, rng, n):
    """base/pilot_data.sample_rows (group-uniform over the TRAIN split), imported read-only."""
    from ._env import sealed
    sealed()
    from pilot_data import sample_rows
    return sample_rows(body, rng, n, train=True)


def pose9_PG(pose9, reach):
    from ._sealed_math import rot6d_to_matrix
    G = rot6d_to_matrix(pose9[..., :6].float())
    P = pose9[..., 6:].float() * float(reach)
    return P, G


def huber(x, delta):
    return torch.nn.functional.huber_loss(x, torch.zeros_like(x), delta=float(delta), reduction='mean')


def _corr(a, b):
    a = a.detach().float()
    b = b.detach().float()
    a = a - a.mean()
    b = b - b.mean()
    den = (a.norm() * b.norm()).clamp_min(1e-9)
    return (a * b).sum() / den


def _sha(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def _split_args(a, b):
    """Accept (rng, cfg) [trainer order] or (cfg, rng) [legacy order]."""
    if isinstance(a, np.random.Generator) and (b is None or isinstance(b, dict)):
        return b, a
    if isinstance(b, np.random.Generator) and (a is None or isinstance(a, dict)):
        return a, b
    raise TypeError('xbody_loss(model, bodies, rng, cfg): rng must be a numpy.random.Generator and cfg a dict')


def xbody_loss(model, bodies, arg3, arg4):
    user, rng = _split_args(arg3, arg4)
    user = dict(user or {})
    if 'batch' in user and 'batch_size' not in user:
        user['batch_size'] = user['batch']
    cfg = {**DEFAULTS, **user}
    if 'w_xb' not in cfg:
        raise KeyError('cfg.w_xb is required (0.0 for the G0 control)')
    by = _bodies(bodies)
    A, F = by[ADROIT], by[FRANKA]
    device = cfg.get('device') or next(model.parameters()).device
    device = torch.device(device)
    B = int(cfg['batch_size'])
    roll = int(cfg['roll'])
    if roll and not 0 < roll < B:
        raise ValueError('roll must be in (0, batch)')
    frozen = op.load_frozen()
    variant = cfg.get('variant') or frozen['variant']
    tau = cfg.get('tau_cm', frozen['tau_cm'])
    ac = cfg.get('autocast_fn') or cfg.get('autocast')
    ctx = (lambda: ac(device)) if ac is not None else contextlib.nullcontext
    # Row draws: always both directions, in this order, whatever w_xb / roll / directions are.
    rows_A = _sample_rows(A, rng, B)
    rows_F = _sample_rows(F, rng, B)
    terms = {}
    total = None
    if cfg['directions'] in ('both', 'AF'):
        xA = A.tensor('pose9', rows_A, device).float()
        with ctx():
            outF = model.forward_kinematic(xA, A.geom(B), F.geom(B))
        c_F_pred = op.franka_clearance(outF['slide_native'].float())
        with torch.no_grad():
            P, G = pose9_PG(xA, A.reach)
            c_src = op.adroit_descriptor(P, G, variant=variant, tau_cm=tau)['c_A'].clamp(0.0, float(cfg['clip_cm']))
            target = torch.roll(c_src, shifts=roll, dims=0) if roll else c_src
        L_AF = huber(c_F_pred - target, cfg['huber_delta_cm'])
        total = L_AF
        terms.update(xb_AF_huber=L_AF.detach(), xb_AF_corr=_corr(c_F_pred, target), xb_AF_cF_pred_mean=c_F_pred.detach().mean(),
                     xb_AF_target_mean=target.mean())
    if cfg['directions'] in ('both', 'FA'):
        xF = F.tensor('pose9', rows_F, device).float()
        sF = F.tensor('slide_normalized', rows_F, device).float()
        with ctx():
            outA = model.forward_kinematic(xF, F.geom(B), A.geom(B), source_slide_normalized=sF)
        desc = op.adroit_descriptor(outA['joint_position'].float(), outA['joint_rotation'].float(), variant=variant, tau_cm=tau)
        with torch.no_grad():
            c_src = op.franka_clearance(F.tensor('slide_native', rows_F, device).float())
            target = torch.roll(c_src, shifts=roll, dims=0) if roll else c_src
        L_FA = huber(desc['c_A'] - target, cfg['huber_delta_cm'])
        hinge = torch.relu(float(cfg['facing_margin']) - desc['facing']).mean()
        part = L_FA + float(cfg['facing_weight']) * hinge
        total = part if total is None else total + part
        terms.update(xb_FA_huber=L_FA.detach(), xb_FA_facing_hinge=hinge.detach(), xb_FA_corr=_corr(desc['c_A'], target),
                     xb_FA_cA_pred_mean=desc['c_A'].detach().mean(), xb_FA_target_mean=target.mean())
    if total is None:
        raise ValueError("cfg.directions must be 'both', 'AF' or 'FA'")
    loss = float(cfg['w_xb']) * total
    if not torch.isfinite(loss) or not torch.isfinite(total):
        raise FloatingPointError('nonfinite xbody loss')
    terms.update(xb_total_unweighted=total.detach(), xb_weighted=loss.detach(),
                 count_xb_rows_A=torch.tensor(float(B if cfg['directions'] in ('both', 'AF') else 0)),
                 count_xb_rows_F=torch.tensor(float(B if cfg['directions'] in ('both', 'FA') else 0)))
    # sampling record: rows only (NOT w_xb / roll), so G0, G1 and G1_roll log the same value
    sampling = {'rows_A_sha256': _sha(rows_A), 'rows_F_sha256': _sha(rows_F), 'batch': B}
    if isinstance(cfg.get('info'), dict):
        cfg['info'].update(rows_A=rows_A, rows_F=rows_F, variant=variant, tau_cm=tau)
    return {'total': loss, 'terms': terms, 'sampling': sampling}
