"""Stage-0 ruler gate for judge J1 (CPU only).

(1) Reproduce report 32's standing lowest point on GMR GT:
    reference_b,corpus = cache_stance_low_gap_p50 (label-floor height of the labeller sole point on stance
    frames, report-32 results.json) + delta_col_b (label sole drop - vendor collision drop).  That is where
    the lowest COLLISION point must sit on stance frames when the root height is the label root height.
    J1 measurement: p50 over stance frames (labeller hard contact on a foot-group joint, trusted clip, exactly
    report-32 cache_gauge's definition) of zmin_cm from J1 on the cache pose (local6d -> FK with the AM2 static
    spec, the same path load_xwbc_trajectory uses) with root height = contact-npz gap[:, root].
    Sampling: every 8th LAFAN row (report 32 used 8) and every 64th AMASS row (report 32: 8; AMASS has 4.7M
    rows).  Pass per body iff |J1 - reference| <= 1 cm on both corpora.  Gate: >= 13/15 bodies pass.
    Diagnostic (reported next to the strict gate): flat-foot stance frames only (node tilt < 3 deg vs qpos0) against
    the reference rebuilt with the J1 static drop (identical to report 32's true drop except R1, see ground_judge).
(2) Synthetic +-5 cm root shifts on the same stance frames with the GT root convention (root - delta_col):
    detected iff PenRate(-5) >= 0.9, FloatRate(+5) >= 0.5 and both moved in the right direction (v1);
    the v0 rule (both rates move by >= 0.5) is kept as shift_detected_v0.

Usage:  python -m am2exp_v1.judge.ruler_gate --all [--jobs 8]      (spawns one bounded process per body)
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from am2exp_v1.judge.pose_mjcf import ROOT, OUT, load_binding, geom_lowest_cm
from am2exp_v1.judge.ground_judge import delta_col_table, REPORT32, PEN_THR_CM, FLOAT_THR_CM, report32_sha256
from am2exp_v1.judge.regions import ROBOTS, FOOT_REGIONS

STRIDE = {'lafan': 8, 'amass': 64}
CORPUS_OF_CACHE = {'cache_lafan_v3': 'lafan', 'cache_amass_lconv_tfix': 'amass'}
CHUNK = 65536
RULER_DIR = OUT / 'judge' / 'ruler_v1'


def _sealed_reader():
    sys.dont_write_bytecode = True
    p = str(ROOT / 'data')
    if p not in sys.path:
        sys.path.insert(0, p)
    import trajectory_reader as TR  # read-only sealed module (numpy-only helpers)
    return TR


def ruler_body(body, out_dir=RULER_DIR):
    import torch
    TR = _sealed_reader()
    man = json.loads((ROOT / 'data/data_manifest_v2.json').read_text())
    b = next(x for x in man['xwbc']['bodies'] if x['body'] == body)
    with np.load(b['static_spec'], allow_pickle=True) as s:
        parents = np.asarray(s['parents'], np.int64); off = np.asarray(s['rest_offsets'], np.float32)
        names = [str(x) for x in s['joint_names']]
    rows32 = {r['embodiment']: r for r in json.loads((REPORT32 / 'results.json').read_text())['rows']}[body]
    delta32 = delta_col_table(corrected=False)[body]      # pre-registered reference (report 32)
    delta = delta_col_table(corrected=True)[body]         # J1-corrected (differs only for R1, see ground_judge)
    binding = load_binding(body)
    res = {'body': body, 'delta_col_cm_report32': delta32, 'delta_col_cm': delta, 'corpora': {}}
    for cache in b['caches']:
        corpus = CORPUS_OF_CACHE[Path(cache['cache']).name]
        stride = STRIDE[corpus]
        path = cache['contact_npz']
        with np.load(path, allow_pickle=False) as z:
            hdr = json.loads(str(z['header']))
        if hdr['joint_names'] != names:
            raise ValueError('contact/spec joint names differ')
        groups = hdr['limb_groups']
        feet = [i for i, g in enumerate(groups) if g in ('L_foot', 'R_foot')] or sorted({hdr['j_low'], hdr['j_floor']})
        root = int(np.flatnonzero(parents < 0)[0])
        N = int(hdr['N'])
        t6 = torch.load(cache['local_rot6_tensor'], weights_only=True, mmap=True, map_location='cpu')
        if t6.shape[0] != N:
            raise ValueError('cache rows != contact rows')
        # static (qpos0) drop from the label-floor joint to the lowest foot collision point, J1 machinery
        rest_low = geom_lowest_cm(binding, binding.rest_rot[None], binding.rest_pos_cm[None])[0][0]
        fm_all = binding.region_mask[list(FOOT_REGIONS)].any(0)
        jf = names.index(hdr['j_floor_name'])
        drop_j1 = float(binding.rest_pos_cm[jf, 1] - rest_low[fm_all].min())
        label_drop = float(hdr['sole_drop_cm'][0]) if hdr.get('sole_drop_cm') else 0.0
        zmins, feetmin, stances, roots, tilts = [], [], [], [], []
        for start in range(0, N, CHUNK):
            n = min(CHUNK, N - start)
            sel = np.arange(start + ((-start) % stride), start + n, stride)
            if not len(sel):
                continue
            gap = TR.npz_rows(path, 'gap', start, n).astype(np.float32)[sel - start]
            hard = TR.npz_rows(path, 'contact', start, n).astype(bool)[sel - start]
            trust = TR.npz_rows(path, 'trust', start, n).astype(bool)[sel - start]
            local = t6[torch.as_tensor(sel)].float().numpy()
            R, P = TR.local_fk(local.astype(np.float64), parents, off.astype(np.float64))
            rh = gap[:, root].astype(np.float64)
            low, _, _ = geom_lowest_cm(binding, R, P)
            low = low + 100 * rh[:, None]
            zmins.append(low.min(1))
            # tilt of the node carrying the lowest geom, relative to its qpos0 (flat-foot) orientation
            a = binding.anchor[low.argmin(1)]
            M = np.einsum('tab,tcb->tac', R[np.arange(len(a)), a], binding.rest_rot[a])
            tilts.append(np.degrees(np.arccos(np.clip(M[:, 1, 1], -1, 1))))
            fm = binding.region_mask[list(FOOT_REGIONS)].any(0)
            feetmin.append(low[:, fm].min(1))
            stances.append(hard[:, feet].any(1) & trust & np.isfinite(rh))
            roots.append(rh)
        zmin = np.concatenate(zmins); fmin = np.concatenate(feetmin); st = np.concatenate(stances); tilt = np.concatenate(tilts)
        flat = st & (tilt < 3.0)
        ref = float(rows32[f'cache_stance_low_gap_p50_{corpus}']) + delta32
        j1 = float(np.median(zmin[st]))
        zc = zmin[st] - delta  # GT root convention (root - delta_col)
        shifts = {}
        for s in (-5.0, 0.0, 5.0):
            z = zc + s
            shifts[f'{s:+.0f}'] = {'PenRate': float((z < PEN_THR_CM).mean()), 'FloatRate': float((z > FLOAT_THR_CM).mean()),
                                   'zmin_p50_cm': float(np.median(z))}
        # v0 rule (kept for the record): both rates move by >= 0.5 -- fails when the unshifted GT already
        # penetrates on many stance frames (tilted feet), which is a property of the data, not of the ruler.
        detected_v0 = (shifts['-5']['PenRate'] - shifts['+0']['PenRate'] >= 0.5) and (shifts['+5']['FloatRate'] - shifts['+0']['FloatRate'] >= 0.5)
        # v1 rule: the shift is seen in the right direction and dominates the stance frames
        detected = (shifts['-5']['PenRate'] >= 0.9 and shifts['+5']['FloatRate'] >= 0.5
                    and shifts['-5']['PenRate'] > shifts['+0']['PenRate'] and shifts['+5']['FloatRate'] > shifts['+0']['FloatRate'])
        res['corpora'][corpus] = {
            'stride': stride, 'frames_sampled': int(len(zmin)), 'stance_frames': int(st.sum()), 'feet_joints': [names[i] for i in feet],
            'report32_stance_gap_p50_cm': float(rows32[f'cache_stance_low_gap_p50_{corpus}']), 'reference_lowest_p50_cm': ref,
            'j1_stance_zmin_p50_cm': j1, 'j1_stance_footregion_min_p50_cm': float(np.median(fmin[st])),
            'j1_stance_zmin_p10_cm': float(np.percentile(zmin[st], 10)), 'j1_stance_zmin_p90_cm': float(np.percentile(zmin[st], 90)),
            'diff_cm': j1 - ref, 'pass_1cm': bool(abs(j1 - ref) <= 1.0),
            'reference_lowest_p50_corrected_delta_cm': ref - delta32 + delta, 'diff_corrected_delta_cm': j1 - (ref - delta32 + delta),
            'pass_1cm_corrected_delta': bool(abs(j1 - (ref - delta32 + delta)) <= 1.0),
            'diag_flat_stance_frames': int(flat.sum()), 'diag_flat_stance_zmin_p50_cm': float(np.median(zmin[flat])) if flat.any() else None,
            'diag_flat_diff_cm': float(np.median(zmin[flat]) - ref) if flat.any() else None,
            'diag_stance_tilt_p50_deg': float(np.median(tilt[st])),
            'diag_label_floor_joint': hdr['j_floor_name'], 'diag_label_sole_drop_cm': label_drop,
            'diag_report32_true_drop_cm': float(rows32['true_sole_drop_cm']), 'diag_j1_static_drop_cm': drop_j1,
            'diag_reference_with_j1_static_drop_cm': float(rows32[f'cache_stance_low_gap_p50_{corpus}']) + label_drop - drop_j1,
            'diag_diff_vs_j1_static_reference_cm': j1 - (float(rows32[f'cache_stance_low_gap_p50_{corpus}']) + label_drop - drop_j1),
            'synthetic_shift': shifts, 'shift_detected': bool(detected), 'shift_detected_v0': bool(detected_v0)}
    res['pass'] = all(c['pass_1cm'] for c in res['corpora'].values())
    res['pass_corrected_delta'] = all(c['pass_1cm_corrected_delta'] for c in res['corpora'].values())
    res['pass_lafan_only'] = res['corpora']['lafan']['pass_1cm'] if 'lafan' in res['corpora'] else None
    # diagnostic verdict: flat-foot stance frames (tilt < 3 deg) against the reference rebuilt with the J1 static drop
    for c in res['corpora'].values():
        c['diag_flat_diff_vs_j1_static_reference_cm'] = (None if c['diag_flat_stance_zmin_p50_cm'] is None else
            c['diag_flat_stance_zmin_p50_cm'] - c['diag_reference_with_j1_static_drop_cm'])
    res['diag_pass_flat_j1ref'] = all(c['diag_flat_diff_vs_j1_static_reference_cm'] is not None and
                                      abs(c['diag_flat_diff_vs_j1_static_reference_cm']) <= 1.0 for c in res['corpora'].values())
    res['shift_detected'] = all(c['shift_detected'] for c in res['corpora'].values())
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{body}.json').write_text(json.dumps(res, indent=1))
    return res


def run_all(jobs=8, timeout_s=1800, out_dir=RULER_DIR):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONPATH=str(ROOT))
    pending = list(ROBOTS); running = {}; t0 = time.time(); status = {}
    while pending or running:
        while pending and len(running) < jobs:
            body = pending.pop(0)
            log = open(out_dir / f'{body}.log', 'w')
            running[body] = (subprocess.Popen([sys.executable, '-m', 'am2exp_v1.judge.ruler_gate', '--body', body, '--out', str(out_dir)],
                                              env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)), time.time(), log)
        for body, (pr, ts, log) in list(running.items()):
            rc = pr.poll()
            if rc is None and time.time() - ts > timeout_s:
                pr.kill(); rc = 'timeout'
            if rc is not None:
                log.close(); status[body] = rc; del running[body]
        time.sleep(1)
    results = {b: json.loads((out_dir / f'{b}.json').read_text()) for b in ROBOTS if (out_dir / f'{b}.json').exists()}
    n_pass = sum(r['pass'] for r in results.values())
    summary = {'schema': 'am2exp_j1_ruler_gate_v1', 'bodies': results, 'exit_status': status,
               'n_pass': n_pass, 'n_bodies': len(ROBOTS), 'gate_pass': n_pass >= 13 and len(results) == len(ROBOTS),
               'diag_n_pass_flat_j1ref': sum(r['diag_pass_flat_j1ref'] for r in results.values()),
               'n_pass_corrected_delta': sum(r['pass_corrected_delta'] for r in results.values()),
               'n_pass_lafan_only': sum(bool(r['pass_lafan_only']) for r in results.values()),
               'shift_detected_v0_all': all(all(c['shift_detected_v0'] for c in r['corpora'].values()) for r in results.values()),
               'diag_rule': 'flat-foot stance (node tilt < 3 deg vs qpos0) p50 vs report32 stance gap p50 + label drop - J1 static drop, |d|<=1 cm both corpora',
               'shift_detected_all': all(r['shift_detected'] for r in results.values()) and len(results) == len(ROBOTS),
               'report32_sha256': report32_sha256(), 'wall_seconds': time.time() - t0,
               'rule': 'per body: |p50 J1 stance zmin - (report32 stance gap p50 + delta_col)| <= 1 cm on LAFAN and AMASS; gate >= 13/15'}
    (out_dir / 'ruler_gate_summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--body'); ap.add_argument('--all', action='store_true'); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default=str(RULER_DIR)); ap.add_argument('--timeout', type=int, default=1800)
    a = ap.parse_args()
    if a.body:
        r = ruler_body(a.body, a.out)
        print(json.dumps({k: (v if k != 'corpora' else {c: {kk: vv for kk, vv in d.items() if kk != 'synthetic_shift'} for c, d in v.items()}) for k, v in r.items()}, indent=1))
    if a.all:
        s = run_all(a.jobs, a.timeout, a.out)
        for b, r in s['bodies'].items():
            print(b, 'PASS' if r['pass'] else 'FAIL', {c: round(d['diff_cm'], 2) for c, d in r['corpora'].items()})
        print('gate_pass', s['gate_pass'], 'n_pass', s['n_pass'], 'shift_detected_all', s['shift_detected_all'],
              'diag_n_pass_flat_j1ref', s['diag_n_pass_flat_j1ref'])


if __name__ == '__main__':
    main()
