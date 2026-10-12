"""Score E1 case npz files with judge J1 (CPU, bounded process pool).

  python -m am2exp_v1.eval.score_e1 --dir OUT/eval/e1/A0_s3000/sensor_on [--jobs 8]
writes <dir>/scores_j1.jsonl (one line per case) and <dir>/scores_j1_meta.json.

Per case (pred vs GT on the same ruler, GT root convention = root - delta_col for every body):
  * frozen convention ('conv'): pred root - delta_col (seen robots) / - c_b (heldout, OUT/judge/heldout_root_bias_v1.json)
  * 'raw' variant: predicted root height unchanged (prefix raw_)
  * snapped variants CSE_snap* (per-frame lowest region shifted to 0, root-free)
  * CSE_human (human sources): pred region contact vs human joint-centre labels (documented under-count)
  * pos_err_mm: mean joint position error (root-relative), the mm guard
The judge never reads the model's contact head; pred_contact_logits is ignored here on purpose.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from am2exp_v1.judge import ground_judge as GJ  # noqa: E402
from am2exp_v1.judge.regions import ROBOTS_HELDOUT  # noqa: E402

_BIAS = None


def _init(bias):
    global _BIAS
    _BIAS = bias
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def score_file(path, heldout_bias=None, selfpen=True):
    bias = heldout_bias if heldout_bias is not None else _BIAS
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    body = meta['target_body']
    gt = GJ.judge_case(body, z['gt_joint_rotation'], z['gt_joint_position'],
                       GJ.corrected_root(body, z['gt_root_height_m'], 'gt'), selfpen=selfpen)
    raw_root = z['pred_root_height_m'].astype(np.float64)
    pr_raw = GJ.judge_case(body, z['pred_joint_rotation'], z['pred_joint_position'], raw_root, selfpen=selfpen)
    if body in ROBOTS_HELDOUT and (bias is None or body not in bias):
        shift_cm = np.nan
    else:
        shift_cm = 100.0 * float(np.nanmedian(raw_root - GJ.corrected_root(body, raw_root, 'pred', bias)))
    # the root shift is a per-body constant, so the corrected judge = raw judge shifted (no second FK pass)
    pr = {k: (v - shift_cm if k in ('region_min_height_cm', 'zmin_cm', 'geom_low_cm') else v) for k, v in pr_raw.items()}
    human = None
    if meta['source_body'] == 'human_smpl':
        hc, hv = GJ.human_region_contact(z['source_label_contact_hard'], z['source_label_contact_valid'], meta['source_joint_names'])
        human = (hc, hv)
    err = np.linalg.norm(z['pred_joint_position'] - z['gt_joint_position'], axis=-1) * 10.0
    rec = dict(case_id=meta['case']['case_id'], window_id=meta['case']['window_id'], source=meta['source_body'], target=body,
               kind=meta['case']['kind'], primary=meta['case']['primary'], heldout=body in ROBOTS_HELDOUT,
               mode=meta['mode'], shuffled=meta['shuffled_source'], root_shift_cm=shift_cm,
               pred_root_p50_m=float(np.nanmedian(raw_root)), gt_root_p50_m=float(np.nanmedian(z['gt_root_height_m'])))
    if np.isfinite(shift_cm):
        m = GJ.case_metrics(pr, gt, human_contact=None if human is None else human[0],
                            human_valid=None if human is None else human[1], position_error_mm=err)
        rec.update(m)
    else:
        rec['pos_err_mm'] = float(np.nanmean(err))
    mr = GJ.case_metrics(pr_raw, gt, human_contact=None if human is None else human[0],
                         human_valid=None if human is None else human[1])
    rec.update({'raw_' + k: v for k, v in mr.items() if k in ('PenRate', 'PenP95_cm', 'FloatRate', 'CSE_GT', 'CSE_GT_nonfoot',
                                                           'zmin_p50_cm', 'CSE_human', 'CSE_human_nonfoot')})
    rec['stance_dz_cm'] = stance_dz(pr_raw, gt)
    return rec


def stance_dz(pred_raw, gt):
    """Calibration helper: median over GT stance frames (a foot region in contact, no non-foot region in contact,
    GT root convention) of zmin(pred, raw root) - zmin(GT).  Used only for the heldout root bias (TRAIN windows)."""
    from am2exp_v1.judge.regions import FOOT_REGIONS, NONFOOT_REGIONS
    rg = gt['region_min_height_cm']
    c = np.where(np.isfinite(rg), rg < GJ.CONTACT_THR_CM, False)
    st = c[:, list(FOOT_REGIONS)].any(1) & ~c[:, list(NONFOOT_REGIONS)].any(1)
    st &= np.isfinite(pred_raw['zmin_cm']) & np.isfinite(gt['zmin_cm'])
    if not st.any():
        return dict(n=0, median=None, values=[])
    d = pred_raw['zmin_cm'][st] - gt['zmin_cm'][st]
    return dict(n=int(st.sum()), median=float(np.median(d)), p25=float(np.percentile(d, 25)), p75=float(np.percentile(d, 75)))


def score_dir(directory, jobs=8, heldout_bias='auto', selfpen=True, timeout_s=3600):
    directory = Path(directory)
    files = sorted(directory.glob('*.npz'))
    files = [f for f in files if not f.name.endswith('.tmp.npz')]
    bias = GJ.load_heldout_bias() if heldout_bias == 'auto' else heldout_bias
    t0 = time.time()
    recs = []
    if jobs <= 1:
        _init(bias)
        recs = [score_file(f, bias, selfpen) for f in files]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, 8), initializer=_init, initargs=(bias,)) as ex:
            futs = [ex.submit(score_file, str(f), None, selfpen) for f in files]
            for f in futs:
                recs.append(f.result(timeout=max(1.0, timeout_s - (time.time() - t0))))
    with open(directory / 'scores_j1.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r, default=float) + '\n')
    meta = dict(n=len(recs), heldout_bias=bias, seconds=time.time() - t0, delta_col_cm=GJ.delta_col_table(verify=True),
                report32_sha256=GJ.report32_sha256(), selfpen=selfpen)
    (directory / 'scores_j1_meta.json').write_text(json.dumps(meta, indent=1))
    return recs


def load_scores(directory):
    return [json.loads(l) for l in open(Path(directory) / 'scores_j1.jsonl')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True, action='append'); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--no-selfpen', action='store_true')
    a = ap.parse_args()
    for d in a.dir:
        recs = score_dir(d, a.jobs, selfpen=not a.no_selfpen)
        print(d, len(recs), flush=True)


if __name__ == '__main__':
    main()
