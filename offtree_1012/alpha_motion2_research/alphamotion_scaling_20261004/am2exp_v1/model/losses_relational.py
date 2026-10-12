"""Arm O: target-side relational objective on the dense surface (+ objective wrappers).

For XWBC target rows with trusted ground (ground_valid & root_height_valid & contact_trust),
excluding the DFaust groups and the PosePrior trust=0 clips (ground.excluded_ground_groups):
  h_gt   = y(GT posed points)   + 100*rh_gt - beta_vis_cm       (GT root corrected by beta_vis)
  h_pred = y(pred posed points) + 100*rh_gt - beta_vis_cm       (same GT root; root head untouched)
  L_pen   = mean_points relu(-h_pred - relu(-h_gt))^2 / (2 cm)^2             (penetration beyond GT)
  L_clear = Huber_1((softmin_0.5cm(h_pred | region) - min(h_gt | region)) / 2 cm)
            over body regions 0..14 with GT clearance < 20 cm; softmin = softmax(-h/0.5)-weighted
            mean (>= min, bias << 0.5 cm; a -tau*logsumexp softmin would sit tau*ln(n) ~ 3 cm low)
  L_hcon  = BCE(contact_logits_j, 1[min over points anchored to j of h_gt < 2.5 cm])
weights (config['loss'] keys, defaults from final_plan): relational_penetration .5,
relational_clearance .2, relational_hcon .2.  The per-direction 1/8 is applied by the trainer.
All corrections are target-side inside the objective (Body.ground_batch is never touched).
O_roll twin: every GT-derived quantity (h_gt, rh_gt, labels) of row i comes from row
(i+B//2) % B; rows count only if both rows are valid.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from ._sealed import rot6d_to_matrix
from . import surface_api as sa
from .ground import row_ground_ok
from .slide_fix import make_slide_fix_objective

PEN_SCALE_CM = 2.0
CLEAR_SCALE_CM = 2.0
CLEAR_MAX_GT_CM = 20.0
SOFTMIN_TAU_CM = 0.5
HCON_THRESHOLD_CM = 2.5
DEFAULT_WEIGHTS = {'relational_penetration': 0.5, 'relational_clearance': 0.2, 'relational_hcon': 0.2}
COUNT_KEYS = ('relational_rows', 'relational_points', 'relational_regions', 'relational_hcon_joint_frames')


def _roll(x, B):
    return torch.roll(x, -(B // 2), 0) if B > 1 else x


def relational_objective(outputs, target_batch, tgt_body, rows, weights, roll=False, excluded=None):
    """Returns {'total': scalar, 'relational_*': unweighted terms, 'count_*': counts}."""
    rot6 = outputs['rot6']
    zero = rot6.reshape(-1)[:0].float().sum()
    out = {'total': zero}
    out.update({'count_' + k: zero.detach().clone() for k in COUNT_KEYS})
    st = getattr(tgt_body, '_dense_surface_static', None)
    if st is None or st.get('asset') is None or tgt_body.domain != 'xwbc':
        return out
    dev = rot6.device; B = len(rows)
    ok = row_ground_ok(tgt_body, rows, dev, excluded)
    if roll: ok = ok & _roll(ok, B)
    if not bool(ok.any()):
        return out
    w = {k: float(weights.get(k, v)) if hasattr(weights, 'get') else v for k, v in DEFAULT_WEIGHTS.items()}
    asset = st['asset']; beta = st['beta_vis_cm']; reach = float(tgt_body.reach)
    idx = ok.nonzero().squeeze(1)
    with torch.autocast(device_type=dev.type, enabled=False):
        rh = tgt_body.tensor('root_height_m', rows, dev).float()
        rh = torch.where(torch.isfinite(rh), rh, torch.zeros_like(rh))
        with torch.no_grad():
            tgt = target_batch.float()
            gt_pts, _ = sa.pose_surface(asset, rot6d_to_matrix(tgt[..., :6]), tgt[..., 6:] * reach)
            h_gt_all = sa.ground_clearance(gt_pts.float(), rh, beta)  # [B,P]
        if roll:
            h_gt_all = _roll(h_gt_all, B); rh = _roll(rh, B)
        h_gt = h_gt_all[idx]
        R = outputs['joint_rotation'].float()[idx]; pos = outputs['joint_position'].float()[idx]
        pred_pts, _ = sa.pose_surface(asset, R, pos)
        h_pred = sa.ground_clearance(pred_pts.float(), rh[idx], beta)  # [b,P]
        b, P = h_pred.shape
        # [integration] untrusted points (bad rigid binding, WP-ASSETS residual p95 > 3 cm) take no part
        trusted = st.get('point_trusted')
        if trusted is None: trusted = torch.ones(P, dtype=torch.bool, device=dev)
        # --- penetration beyond GT
        pen_pp = torch.relu(-h_pred - torch.relu(-h_gt)).square()
        pen = (pen_pp * trusted.float()).sum() / (b * trusted.sum().clamp_min(1)) / PEN_SCALE_CM ** 2
        # --- region clearance
        region = torch.where(trusted, st['region'], torch.full_like(st['region'], -1))
        regs = torch.arange(15, device=dev)
        rmask = region[None, :] == regs[:, None]  # [R,P]
        has = rmask.any(1)
        gt_min = torch.where(rmask[None], h_gt[:, None, :], torch.full_like(h_gt[:, None, :].expand(-1, 15, -1), 1e4)).amin(-1)
        logits = torch.where(rmask[None], -h_pred[:, None, :] / SOFTMIN_TAU_CM, torch.full((1, 1, 1), -torch.inf, device=dev))
        wts = torch.softmax(torch.where(has[None, :, None], logits, torch.zeros_like(logits)), -1)
        soft = (wts * h_pred[:, None, :]).sum(-1)  # [b,R]
        rvalid = has[None] & (gt_min < CLEAR_MAX_GT_CM)
        if bool(rvalid.any()):
            diff = (soft - gt_min) / CLEAR_SCALE_CM
            clear = F.smooth_l1_loss(diff[rvalid], torch.zeros_like(diff[rvalid]), beta=1.0)
        else:
            clear = zero
        # --- contact head (hcon) supervision from the dense-surface GT contact flag
        anchor = torch.where(trusted, st['anchor'], torch.full_like(st['anchor'], -1)); J = len(tgt_body.parents)
        amask = anchor[None, :] == torch.arange(J, device=dev)[:, None]  # [J,P]
        jhas = amask.any(1)
        gt_jmin = torch.where(amask[None], h_gt[:, None, :], torch.full((1, 1, 1), 1e4, device=dev)).amin(-1)  # [b,J]
        label = (gt_jmin < HCON_THRESHOLD_CM).float()
        con = outputs.get('contact_logits')
        if con is not None and bool(jhas.any()):
            cl = con.float()[idx]
            hcon = F.binary_cross_entropy_with_logits(cl[:, jhas], label[:, jhas])
        else:
            hcon = zero
    total = w['relational_penetration'] * pen + w['relational_clearance'] * clear + w['relational_hcon'] * hcon
    out.update({'total': total, 'relational_pen': pen, 'relational_clear': clear, 'relational_hcon': hcon,
                'relational_hcon_positive_rate': label[:, jhas].mean().detach() if bool(jhas.any()) else zero.detach(),
                'count_relational_rows': zero.new_tensor(float(b)),
                'count_relational_points': zero.new_tensor(float(b * int(trusted.sum()))),
                'count_relational_regions': zero.new_tensor(float(rvalid.sum())),
                'count_relational_hcon_joint_frames': zero.new_tensor(float(b * int(jhas.sum())))})
    return out


def make_objective(base_objective, *, slide_fix=True, relational=False, relational_roll=False, excluded=None):
    """Compose train_scaling.loss_function(cfg) with the slide fix and (arm O) the relational terms."""
    obj = make_slide_fix_objective(base_objective) if slide_fix else base_objective
    if not relational:
        return obj

    def objective(outputs, target, tgt, rows, condition, weights):
        loss, terms = obj(outputs, target, tgt, rows, condition, weights)
        rel = relational_objective(outputs, target, tgt, rows, weights, roll=relational_roll, excluded=excluded)
        loss = loss + rel.pop('total')
        terms.update(rel)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite combined loss')
        return loss, terms
    return objective
