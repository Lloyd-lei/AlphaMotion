"""Slide fix (arm SF): LayerNorm readout of [qp; qr] + margin sigmoid + SmoothL1 loss.

final_plan.md 'Slide fix':
  logit = Linear(2d,1)([LN(qp); LN(qr)]), weight N(0,0.02), bias = logit((0.968+m)/(1+2m)), m=0.05
  q01   = -m + (1+2m) * sigmoid(logit)          (reaches [-m, 1+m]; data max 1.0185)
  native = lower + q01*(upper-lower)            (UNCLAMPED; eval reports raw and clamped)
  loss  = SmoothL1(beta=0.02)(q01 - clip(t,-m,1+m)) + 1e-3*mean(logit^2)   (replaces the L1 term)

The head is created AFTER ConditionalMegatron.__init__ under the same attribute name
``slide_head`` so every inherited parameter is bit-identical to an A build at the same seed,
and validate_model's 'slide_head.' gradient-prefix check still applies.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ._sealed import ConditionalMegatron, SurfaceMegatron, joint_metadata, derive_joint_rotations, geometry_float32

SLIDE_MARGIN = 0.05
SLIDE_INIT_Q = 0.968
SLIDE_BETA = 0.02
SLIDE_LOGIT_REG = 1e-3


def margin_logit(q01, margin=SLIDE_MARGIN):
    s = (q01 + margin) / (1 + 2 * margin)
    return math.log(s / (1 - s))


class SlideFixHead(nn.Module):
    """logit = Linear(2d,1)([LN(qp); LN(qr)]); scale-invariant in qp, qr."""

    def __init__(self, d, margin=SLIDE_MARGIN, init_q=SLIDE_INIT_Q):
        super().__init__()
        self.margin = float(margin)
        self.norm_p = nn.LayerNorm(d)
        self.norm_r = nn.LayerNorm(d)
        self.linear = nn.Linear(2 * d, 1)
        nn.init.normal_(self.linear.weight, std=0.02)
        with torch.no_grad():
            self.linear.bias.fill_(margin_logit(init_q, margin))

    def forward(self, qp, qr):
        return self.linear(torch.cat([self.norm_p(qp), self.norm_r(qr)], -1)).squeeze(-1)


@geometry_float32
def decode_slide_margin(logits, g, metadata=None, margin=SLIDE_MARGIN):
    """Margin sigmoid; returns (q01, native_unclamped, slide_valid). q01 is 0 on non-slide joints."""
    b, j = logits.shape
    m = metadata or joint_metadata(g, b, j, logits.device)
    q = -margin + (1 + 2 * margin) * torch.sigmoid(logits.float())
    q01 = torch.where(m['slide_valid'], q, torch.zeros_like(q))
    native = torch.where(m['slide_valid'], m['lower'] + q01 * (m['upper'] - m['lower']), torch.zeros_like(q01))
    return q01, native, m['slide_valid']


def clamp_slide_native(outputs, g):
    """Limit-clamped native slide for evaluation (raw stays in outputs['slide_native'])."""
    native = outputs['slide_native']
    b, j = native.shape
    m = joint_metadata(g, b, j, native.device)
    clamped = torch.minimum(torch.maximum(native, m['lower']), m['upper'])
    return torch.where(m['slide_valid'], clamped, torch.zeros_like(native))


class SlideFixMixin:
    """Mix in BEFORE a ConditionalMegatron subclass: class SF(SlideFixMixin, ConditionalMegatron)."""

    def __init__(self, *args, slide_margin=SLIDE_MARGIN, slide_init_q=SLIDE_INIT_Q, **kwargs):
        super().__init__(*args, **kwargs)
        d = kwargs['d_model']
        self.slide_margin = float(slide_margin)
        # replaced AFTER the parent constructor: parent RNG stream and parameters unchanged
        self.slide_head = SlideFixHead(d, slide_margin, slide_init_q)

    # verbatim copy of ConditionalMegatron._decode_kinematic_heads (pilot_model.py:113-142)
    # except the slide readout; also exposes qp/qr through self._last_queries for subclasses.
    def _decode_kinematic_heads(self, zp, zr, g, condition=None):
        qp = self.descriptor(g); qr = qp.clone(); kp = kr = None
        if condition is not None:
            ix = condition['joint_indices']; v = condition['values']; valid = condition['valid']
            if ix.ndim != 2 or v.shape != (*ix.shape, 9) or valid.shape != ix.shape:
                raise ValueError('sparse condition shape contract violated')
            if ix.shape[0] != qp.shape[0] or ix.shape[1] > 2 or valid.dtype != torch.bool:
                raise ValueError('only zero, one or two declared SE3 conditions are accepted')
            if ix.dtype != torch.long: raise TypeError('condition joint indices must be int64')
            if bool(((ix[valid] < 0) | (ix[valid] >= qp.shape[1])).any()):
                raise ValueError('valid condition index outside target joint tree')
            if not bool(torch.isfinite(v[valid]).all()): raise ValueError('valid condition values must be finite')
            ix = torch.where(valid, ix, torch.zeros_like(ix))
            v = torch.where(valid[..., None], v, torch.zeros_like(v))
            descriptor = torch.gather(qp, 1, ix[..., None].expand(-1, -1, qp.shape[-1]))
            tokens = self.condition_projection(v) + descriptor + self.condition_type
            tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
            kp = torch.cat([torch.zeros(zp.shape[:2], device=zp.device, dtype=torch.bool), ~valid], 1)
            kr = torch.cat([torch.zeros(zr.shape[:2], device=zr.device, dtype=torch.bool), ~valid], 1)
            zp = torch.cat([zp, tokens], 1); zr = torch.cat([zr, tokens], 1)
        for cb, sb in zip(self.decp, self.selfp): qp = sb(cb(qp, zp, kp), g['hop'], None)
        for cb, sb in zip(self.decr, self.selfr): qr = sb(cb(qr, zr, kr), g['hop'], None)
        raw_rot = self.hrot(qr); pos = self.hpos(qp)
        self.last_pos = pos
        con = self.hcon(qr).squeeze(-1) if self.contact_chan else None
        m = joint_metadata(g, *raw_rot.shape[:2], raw_rot.device)
        rot = derive_joint_rotations(raw_rot, g, m)
        logits = self.slide_head(qp, qr)
        normalized, native, valid = decode_slide_margin(logits, g, m, self.slide_margin)
        return {'rot6': rot, 'raw_hpos': pos, 'contact_logits': con, 'slide_normalized': normalized,
                'slide_native': native, 'slide_valid': valid, 'aux': {'slide_logit': logits}}

    def forward_kinematic(self, source, source_geom, target_geom, source_hidden=None, source_contact=None,
                          condition=None, source_slide_normalized=None, source_root_height_m=None,
                          roll_relations=False):
        # extra kwargs are accepted (and ignored) so one input wrapper can feed every arm
        return super().forward_kinematic(source, source_geom, target_geom, source_hidden, source_contact,
                                         condition, source_slide_normalized)


class SFMegatron(SlideFixMixin, ConditionalMegatron):
    """Arm SF = A recipe + slide fix."""


class SFSurfaceMegatron(SlideFixMixin, SurfaceMegatron):
    """B + slide fix (not a planned arm; provided for completeness / B-side probes)."""


def slide_fix_loss(outputs, slide_target, margin=SLIDE_MARGIN, beta=SLIDE_BETA, logit_reg=SLIDE_LOGIT_REG):
    """Replaces train_scaling's |sigmoid - t| L1.  Returns (loss, terms) on valid slide joints.

    outputs: model dict with slide_normalized (q01), slide_valid, aux.slide_logit.
    slide_target: [B,J] payload slide_normalized (may exceed 1; max 1.0185 in Franka train rows).
    """
    sv = outputs['slide_valid']
    q = outputs['slide_normalized'].float()
    zero = q.reshape(-1)[:0].sum()
    if not bool(sv.any()):
        return zero, {}
    t = torch.as_tensor(slide_target, device=q.device, dtype=torch.float32)
    if t.shape != q.shape: raise ValueError('slide target shape')
    if not bool(torch.isfinite(t[sv]).all()): raise ValueError('valid slide target nonfinite')
    tc = t.clamp(-margin, 1 + margin)
    smooth = F.smooth_l1_loss(q[sv], tc[sv], beta=beta)
    logits = outputs.get('aux', {}).get('slide_logit')
    if logits is None: raise ValueError('slide fix loss requires aux.slide_logit (SF model)')
    reg = logits.float()[sv].square().mean()
    loss = smooth + logit_reg * reg
    terms = {'slide_fix_smoothl1': smooth, 'slide_fix_logit_sq': reg,
             'slide_normalized_l1': (q[sv] - t[sv]).abs().mean().detach(),
             'slide_logit_abs_max': logits.float()[sv].abs().max().detach()}
    return loss, terms


def make_slide_fix_objective(base_objective, weight_key='slide_normalized'):
    """Wrap train_scaling.loss_function(cfg) so its slide L1 is replaced by slide_fix_loss.

    The base objective sees slide_valid all-False (it skips its own slide term and nothing
    else reads slide_valid), the fixed term is added with weights[weight_key], and the
    count_slide_joint_frames entry is restored.
    """
    def objective(outputs, target, tgt, rows, condition, weights):
        sv = outputs['slide_valid']
        if not bool(sv.any()):
            return base_objective(outputs, target, tgt, rows, condition, weights)
        masked = {**outputs, 'slide_valid': torch.zeros_like(sv)}
        loss, terms = base_objective(masked, target, tgt, rows, condition, weights)
        slide_target = tgt.tensor('slide_normalized', rows, sv.device).float()
        sl, sterms = slide_fix_loss(outputs, slide_target)
        loss = loss + weights.get(weight_key, 1.0) * sl
        terms.update(sterms)
        terms['count_slide_joint_frames'] = sl.new_tensor(float(sv.sum()), dtype=torch.float32)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite combined loss')
        return loss, terms
    return objective
