"""Judge J1: ground / contact / self-penetration judge on vendor MJCF collision geometry.

judge_case(body, rot [T,J,3,3], pos [T,J,3] cm, root_height_m [T]) -> per-frame dict
    region_min_height_cm [T,15]  (NaN where the body has no collision geom in that region)
    zmin_cm [T]                   lowest collision point of the whole body above the ground plane
    selfpen_cm [T]                deepest penetration between allowed non-adjacent geoms (0 = none)

Everything here is independent of training: inputs are AM2 global rotations/positions plus a root
height; the geometry is the vendor MJCF collision set bound in pose_mjcf.  No dense visual surface,
no sole table, no training loss and never the model's contact head ("hcon") enter any metric.

Root-height conventions (final_plan.md "Root convention"), applied by `corrected_root`:
  * GT:                 root_height_m - delta_col_cm/100 for every body (report-32 collision bias)
  * prediction, seen:   predicted root - delta_col_cm/100 (the model learned the biased label)
  * prediction, heldout: predicted root - c_b/100, c_b measured from A0 stance frames on TRAIN-split
    windows (OUT/judge/heldout_root_bias_v1.json), applied identically to every arm
  * 'raw' (no correction) and per-frame ground-snapped variants are reported next to it.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from am2exp_v1.judge.pose_mjcf import load_binding, geom_lowest_cm, selfpen_depth_cm, OUT
from am2exp_v1.judge.regions import (N_REGIONS, NONFOOT_REGIONS, FOOT_REGIONS, REGIONS, ROBOTS, ROBOTS_HELDOUT,
                                     HUMAN_JOINT_REGION, R as REGION_INDEX)

CONTACT_THR_CM = 2.5      # region in contact if its lowest collision point is below this height
PEN_THR_CM = -1.0         # PenRate: zmin below -1 cm
FLOAT_THR_CM = 3.0        # FloatRate: zmin above 3 cm while the GT has a region in contact
SELFPEN_THR_CM = 1.0      # SelfPen: deepest allowed-pair penetration beyond 1 cm
REPORT32 = Path('/home/arenalabs/Desktop/alphamotion video/resource/report/32_ground_gauge_audit')
HELDOUT_BIAS_FILE = OUT / 'judge' / 'heldout_root_bias_v1.json'

# report-32 results.json `predicted_bias_cm` (label sole drop - vendor collision drop), frozen copy;
# `delta_col_table()` re-reads the file and refuses to run if it differs.
DELTA_COL_CM_REPORT32 = {
    'booster_k1': 6.4, 'booster_t1': 7.149961179822281, 'booster_t1_29dof': 9.65, 'fourier_gr3': 9.99044620482585,
    'fourier_n1': 0.016692899036159226, 'hightorque_hi': -7.762576181846725e-08, 'kuavo_s45': 4.7,
    'pnd_adam_lite': 9.398823840557583, 'tienkung': 9.999951209626197, 'unitree_g1_23dof': 0.04000000000000048,
    'unitree_g1_29dof': 0.04000000000000048, 'unitree_h1': -9.614582054950915e-07, 'unitree_h1_2': 1.7941505170426808e-08,
    'unitree_h2': -5.43, 'unitree_r1': -0.4992017187320008}
# J1 ruler finding (Stage-0, ruler_v1/unitree_r1.json): report 32 measured R1's "true drop" (0.50 cm) inside the
# ankle_constraint_A/B subtrees only and missed the actual sole (left/right_ankle_roll_link, merged into the
# ankle_pitch node), which reaches 4.38 cm below the label-floor joint centre at qpos0.  For every other body the
# J1 static drop equals report 32's true drop to <0.02 cm, so the corrected table differs only for R1.
R1_J1_STATIC_DROP_CM = 4.384469233317674
DELTA_COL_CM = dict(DELTA_COL_CM_REPORT32, unitree_r1=0.0 - R1_J1_STATIC_DROP_CM)


def delta_col_table(verify=True, corrected=True):
    if verify:
        rows = json.loads((REPORT32 / 'results.json').read_text())['rows']
        live = {r['embodiment']: float(r['predicted_bias_cm']) for r in rows}
        for k, v in DELTA_COL_CM_REPORT32.items():
            if abs(live[k] - v) > 1e-6:
                raise ValueError(f'report-32 delta_col changed for {k}: {live[k]} vs {v}')
    return dict(DELTA_COL_CM if corrected else DELTA_COL_CM_REPORT32)


def report32_sha256():
    return {f: hashlib.sha256((REPORT32 / f).read_bytes()).hexdigest() for f in ('results.json', 'sole_audit.json', 'cache_gauge.json')}


def load_heldout_bias(path=HELDOUT_BIAS_FILE):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {k: float(v['applied_c_cm']) for k, v in d['bodies'].items()}


def corrected_root(body, root_height_m, kind, heldout_bias=None):
    """kind: 'gt' | 'pred' | 'raw'.  Returns root height in metres under the frozen convention."""
    r = np.asarray(root_height_m, np.float64)
    if kind == 'raw':
        return r
    if kind == 'gt' or body not in ROBOTS_HELDOUT:
        return r - DELTA_COL_CM[body] / 100.0
    if kind == 'pred':
        if heldout_bias is None or body not in heldout_bias:
            raise ValueError(f'heldout root bias for {body} not measured yet (Stage-0 item 2)')
        return r - heldout_bias[body] / 100.0
    raise ValueError(kind)


def judge_case(body, rot, pos, root_height_m, selfpen=True, binding_dir=None):
    """Per-frame J1 quantities.  rot [T,J,3,3] AM2 global rotations, pos [T,J,3] cm root-relative,
    root_height_m [T] (any root convention; NaN frames propagate)."""
    b = load_binding(body) if binding_dir is None else load_binding(body, str(binding_dir))
    rot = np.asarray(rot, np.float64); pos = np.asarray(pos, np.float64)
    if rot.shape[1] != b.J or pos.shape[1] != b.J:
        raise ValueError(f'{body}: expected J={b.J}, got {rot.shape}')
    low, gr, gp = geom_lowest_cm(b, rot, pos)
    low = low + 100.0 * np.asarray(root_height_m, np.float64)[:, None]
    T = len(low)
    region = np.full((T, N_REGIONS), np.nan)
    for r in range(N_REGIONS):
        if b.regions_present[r]:
            region[:, r] = low[:, b.region_mask[r]].min(1)
    out = dict(region_min_height_cm=region, zmin_cm=low.min(1), geom_low_cm=low,
               regions_present=b.regions_present.copy())
    out['selfpen_cm'] = selfpen_depth_cm(b, gr, gp) if selfpen else np.full(T, np.nan)
    return out


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------
def region_contact(region_min_height_cm, thr=CONTACT_THR_CM):
    z = np.asarray(region_min_height_cm)
    return np.where(np.isfinite(z), z < thr, False)


def snapped(region_min_height_cm):
    """Per-frame ground snap: shift every frame so its lowest region touches 0 (root-free pattern)."""
    z = np.asarray(region_min_height_cm)
    return z - np.nanmin(z, axis=1, keepdims=True)


def f1_score(pred, gt, mask=None):
    pred = np.asarray(pred, bool); gt = np.asarray(gt, bool)
    if mask is not None:
        pred = pred & mask; gt = gt & mask
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)


def human_region_contact(contact_hard, contact_valid, joint_names):
    """[T,22] joint-centre labels -> [T,15] region contact, [T,15] validity (joint-centre undercount!)."""
    T = len(contact_hard)
    c = np.zeros((T, N_REGIONS), bool); v = np.zeros((T, N_REGIONS), bool)
    for j, n in enumerate(joint_names):
        r = REGION_INDEX[HUMAN_JOINT_REGION[str(n)]]
        c[:, r] |= np.asarray(contact_hard[:, j], bool) & np.asarray(contact_valid[:, j], bool)
        v[:, r] |= np.asarray(contact_valid[:, j], bool)
    return c, v


def _rate(x):
    x = np.asarray(x, float)
    return float(np.mean(x)) if x.size else float('nan')


def case_metrics(pred, gt, *, human_contact=None, human_valid=None, position_error_mm=None):
    """pred/gt: judge_case dicts (already root-corrected).  Frames where either zmin is NaN are dropped."""
    zp, zg = pred['zmin_cm'], gt['zmin_cm']
    ok = np.isfinite(zp) & np.isfinite(zg)
    if not ok.any():  # e.g. untrusted GT clip (no root height): the case carries no ground evidence
        out = {k: float('nan') for k in ('PenRate', 'PenP95_cm', 'FloatRate', 'SelfPen', 'CSE_GT', 'CSE_GT_nonfoot', 'CSE_snap',
                                         'CSE_snap_nonfoot', 'zmin_p50_cm', 'gt_zmin_p50_cm', 'gt_PenRate', 'gt_PenP95_cm',
                                         'gt_SelfPen', 'gt_nonfoot_contact_frac', 'pred_nonfoot_contact_frac')}
        out['frames'] = 0
        if human_contact is not None:
            out.update(CSE_human=float('nan'), CSE_human_nonfoot=float('nan'), gt_CSE_human=float('nan'))
        if position_error_mm is not None:
            out['pos_err_mm'] = float(np.nanmean(position_error_mm))
        return out
    rp, rg = pred['region_min_height_cm'][ok], gt['region_min_height_cm'][ok]
    zp, zg = zp[ok], zg[ok]
    present = np.isfinite(rp).any(0) & np.isfinite(rg).any(0)
    cp, cg = region_contact(rp), region_contact(rg)
    mask = np.broadcast_to(present, cp.shape)
    nf = np.zeros(N_REGIONS, bool); nf[list(NONFOOT_REGIONS)] = True
    sp, sg = region_contact(snapped(rp)), region_contact(snapped(rg))
    gt_any = cg[:, present].any(1)
    out = {
        'frames': int(ok.sum()),
        'PenRate': _rate(zp < PEN_THR_CM), 'PenP95_cm': float(np.percentile(np.maximum(0, -zp), 95)) if len(zp) else float('nan'),
        'FloatRate': _rate(zp[gt_any] > FLOAT_THR_CM) if gt_any.any() else float('nan'),
        'SelfPen': _rate(pred['selfpen_cm'][ok] > SELFPEN_THR_CM) if np.isfinite(pred['selfpen_cm']).all() else float('nan'),
        'CSE_GT': 1 - f1_score(cp, cg, mask), 'CSE_GT_nonfoot': 1 - f1_score(cp, cg, mask & nf),
        'CSE_snap': 1 - f1_score(sp, sg, mask), 'CSE_snap_nonfoot': 1 - f1_score(sp, sg, mask & nf),
        'zmin_p50_cm': float(np.median(zp)) if len(zp) else float('nan'),
        'gt_zmin_p50_cm': float(np.median(zg)) if len(zg) else float('nan'),
        'gt_PenRate': _rate(zg < PEN_THR_CM), 'gt_PenP95_cm': float(np.percentile(np.maximum(0, -zg), 95)) if len(zg) else float('nan'),
        'gt_SelfPen': _rate(gt['selfpen_cm'][ok] > SELFPEN_THR_CM) if np.isfinite(gt['selfpen_cm']).all() else float('nan'),
        'gt_nonfoot_contact_frac': _rate(cg[:, list(NONFOOT_REGIONS)][:, present[list(NONFOOT_REGIONS)]].any(1)) if len(cg) else float('nan'),
        'pred_nonfoot_contact_frac': _rate(cp[:, list(NONFOOT_REGIONS)][:, present[list(NONFOOT_REGIONS)]].any(1)) if len(cp) else float('nan'),
    }
    if human_contact is not None:
        hc, hv = human_contact[ok], human_valid[ok]
        m = hv & present[None]
        out['CSE_human'] = 1 - f1_score(cp, hc, m)
        out['CSE_human_nonfoot'] = 1 - f1_score(cp, hc, m & nf)
        out['gt_CSE_human'] = 1 - f1_score(cg, hc, m)
    if position_error_mm is not None:
        out['pos_err_mm'] = float(np.nanmean(position_error_mm))
    return out
