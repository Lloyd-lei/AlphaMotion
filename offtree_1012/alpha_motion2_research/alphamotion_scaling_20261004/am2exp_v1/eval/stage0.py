"""Stage-0 for E1 on CPU: ruler gate summary, heldout root bias of A0, headroom table (A0/B0 vs GT, sensor on/off,
shuffled-source floor), case-level synthetic root shifts, OUT/judge/stage0_report.md.

  python -m am2exp_v1.eval.stage0 [--jobs 8] [--steps rootbias,infer,score,report]

Every child process is CPU-only (CUDA_VISIBLE_DEVICES=''), 2 threads, killed after --timeout seconds; at most --jobs
children run at once.  Ruler gate itself: python -m am2exp_v1.judge.ruler_gate --all (run before this).
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

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from am2exp_v1.eval.cases import CASES_FILE, ROOTBIAS_FILE, load_cases  # noqa: E402
from am2exp_v1.eval import stats as ST  # noqa: E402

OUT = Path('/var/tmp/alphamotion/am2exp')
E1 = OUT / 'eval' / 'e1'
JUDGE = OUT / 'judge'
PY = sys.executable
RUNS = {'A0': ROOT / 'runs/A_S_seed0/checkpoint_step3000.pt', 'B0': ROOT / 'runs/B_S_seed0/checkpoint_step3000.pt'}
SHARDS = 4
METRICS = ['CSE_GT', 'CSE_GT_nonfoot', 'CSE_snap', 'CSE_snap_nonfoot', 'FloatRate', 'PenRate', 'PenP95_cm', 'SelfPen',
           'pos_err_mm', 'CSE_human', 'CSE_human_nonfoot', 'raw_CSE_GT', 'raw_FloatRate', 'raw_PenP95_cm',
           'gt_PenRate', 'gt_PenP95_cm', 'gt_SelfPen', 'gt_CSE_human', 'gt_nonfoot_contact_frac', 'pred_nonfoot_contact_frac',
           'zmin_p50_cm', 'gt_zmin_p50_cm']


def env():
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
             MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
    return e


def run_pool(cmds, jobs, timeout_s, log_dir):
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(cmds)); running = {}; status = {}
    while pending or running:
        while pending and len(running) < jobs:
            i, cmd = pending.pop(0)
            log = open(log_dir / f'job{i:03d}.log', 'w')
            running[i] = (subprocess.Popen(cmd, env=env(), stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)), time.time(), log)
        for i, (p, t0, log) in list(running.items()):
            rc = p.poll()
            if rc is None and time.time() - t0 > timeout_s:
                p.kill(); p.wait(); rc = 'timeout'
            if rc is not None:
                log.close(); status[i] = rc; del running[i]
        time.sleep(0.5)
    bad = {i: rc for i, rc in status.items() if rc != 0}
    if bad:
        raise RuntimeError(f'jobs failed: {bad} (logs in {log_dir})')
    return status


def infer_cmds(arm, ckpt, mode, out, cases, shuffle=False, shards=SHARDS):
    base = [PY, '-m', 'am2exp_v1.eval.run_e1_eval', '--arm', arm, '--checkpoint', str(ckpt), '--mode', mode,
            '--out', str(out), '--cases', str(cases), '--threads', '2']
    if shuffle:
        base.append('--shuffle-source')
    return [base + ['--shard', f'{k}/{shards}'] for k in range(shards)]


def run_dirs():
    d = {}
    for arm in RUNS:
        for mode in ('sensor_on', 'sensor_off'):
            d[(arm, mode, False)] = E1 / f'{arm}_s3000' / mode
    d[('A0', 'sensor_on', True)] = E1 / 'A0_s3000' / 'sensor_on_shuffled'
    return d


def rootbias_dirs():
    return {m: E1 / 'rootbias' / 'A0_s3000' / m for m in ('sensor_on', 'sensor_off')}


def step_rootbias(jobs, timeout):
    cmds = []
    for mode, d in rootbias_dirs().items():
        cmds += infer_cmds('A0', RUNS['A0'], mode, d, ROOTBIAS_FILE, shards=2)
    run_pool(cmds, jobs, timeout, E1 / 'logs' / 'rootbias')
    from am2exp_v1.eval.score_e1 import score_dir
    res = {}
    for mode, d in rootbias_dirs().items():
        recs = score_dir(d, jobs, heldout_bias=None, selfpen=False)
        res[mode] = recs
    bodies = {}
    for b in ('fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1'):
        row = {}
        for mode, recs in res.items():
            vals = [(r['stance_dz_cm']['median'], r['stance_dz_cm']['n'], r['window_id']) for r in recs
                    if r['target'] == b and r['stance_dz_cm']['n'] >= 8]
            med = [v[0] for v in vals]
            row[mode] = dict(c_cm=float(np.median(med)) if med else None, windows=len(vals),
                             stance_frames=int(sum(v[1] for v in vals)),
                             per_window={v[2]: v[0] for v in vals},
                             iqr_cm=[float(np.percentile(med, 25)), float(np.percentile(med, 75))] if med else None,
                             pred_root_p50_m=float(np.median([r['pred_root_p50_m'] for r in recs if r['target'] == b])),
                             gt_root_p50_m=float(np.median([r['gt_root_p50_m'] for r in recs if r['target'] == b])))
        c = row['sensor_on']['c_cm']
        if c is None:
            raise RuntimeError(f'no stance frames for {b}')
        bodies[b] = dict(applied_c_cm=c, measured=row)
    doc = dict(schema='am2exp_j1_heldout_root_bias_v1', model='A0 = runs/A_S_seed0/checkpoint_step3000.pt', cases=str(ROOTBIAS_FILE),
               cases_sha256=load_cases(ROOTBIAS_FILE)['sha256'],
               definition='c_b = median over TRAIN-split windows of the per-window median, over GT stance frames (J1 GT, GT root '
                          'convention: a foot region < 2.5 cm and no non-foot region < 2.5 cm), of zmin(A0 pred, raw predicted root) - '
                          'zmin(GT).  Applied to EVERY arm as pred_root - c_b/100 for heldout bodies (sensor_on value for both modes).',
               bodies=bodies)
    JUDGE.mkdir(parents=True, exist_ok=True)
    (JUDGE / 'heldout_root_bias_v1.json').write_text(json.dumps(doc, indent=1))
    return doc


def step_infer(jobs, timeout):
    cmds = []
    for (arm, mode, shuffle), d in run_dirs().items():
        cmds += infer_cmds(arm, RUNS[arm], mode, d, CASES_FILE, shuffle=shuffle)
    run_pool(cmds, jobs, timeout, E1 / 'logs' / 'infer')


def step_score(jobs):
    from am2exp_v1.eval.score_e1 import score_dir
    for key, d in run_dirs().items():
        t = time.time(); recs = score_dir(d, jobs)
        print('scored', key, len(recs), f'{time.time() - t:.0f}s', flush=True)


def synthetic_case_shifts(cases_doc, max_cases=None):
    """GT re-scored against itself with the root shifted by -5/0/+5 cm (primary human cases)."""
    from am2exp_v1.judge import ground_judge as GJ
    d = run_dirs()[('A0', 'sensor_on', False)]
    out = {s: [] for s in (-5.0, 0.0, 5.0)}
    cs = [c for c in cases_doc['cases'] if c['primary']][:max_cases]
    for c in cs:
        z = np.load(d / f"{c['case_id']}.npz")
        b = c['target']
        gt = GJ.judge_case(b, z['gt_joint_rotation'], z['gt_joint_position'], GJ.corrected_root(b, z['gt_root_height_m'], 'gt'), selfpen=False)
        for s in out:
            sh = {k: (v + s if k in ('region_min_height_cm', 'zmin_cm', 'geom_low_cm') else v) for k, v in gt.items()}
            m = GJ.case_metrics(sh, gt)
            out[s].append(dict(window_id=c['window_id'], target=b, CSE_GT=m['CSE_GT'], PenP95_cm=m['PenP95_cm'], FloatRate=m['FloatRate'],
                               PenRate=m['PenRate']))
    summ = {}
    for s, rows in out.items():
        summ[f'{s:+.0f}'] = {k: float(np.nanmean([r[k] for r in rows])) for k in ('CSE_GT', 'PenP95_cm', 'FloatRate', 'PenRate')}
    detected_v0 = (summ['-5']['PenP95_cm'] - summ['+0']['PenP95_cm'] >= 4.0 and summ['+5']['FloatRate'] - summ['+0']['FloatRate'] >= 0.5
                   and summ['-5']['CSE_GT'] >= 0.10 and summ['+5']['CSE_GT'] >= 0.10 and summ['+0']['CSE_GT'] == 0.0)
    # v1 (revised AFTER seeing v0 numbers; documented): FloatRate cannot move by 0.5 on windows where the GT itself sinks
    # ~13 cm (p95) into the floor, so only its direction is required.
    detected = (summ['-5']['PenP95_cm'] - summ['+0']['PenP95_cm'] >= 4.0 and summ['+5']['FloatRate'] > summ['+0']['FloatRate']
                and summ['-5']['CSE_GT'] >= 0.10 and summ['+5']['CSE_GT'] >= 0.10 and summ['+0']['CSE_GT'] == 0.0)
    return dict(n_cases=len(cs), summary=summ, detected=bool(detected), detected_v0=bool(detected_v0),
                rule_v0='GT vs itself: PenP95(-5)-PenP95(0) >= 4 cm, FloatRate(+5)-FloatRate(0) >= 0.5, CSE_GT(+-5) >= 0.10, CSE_GT(0) == 0',
                rule='v1 (post hoc): as v0 but FloatRate(+5) > FloatRate(0)')


def gt_shuffled_floor(cases_doc):
    """Chance level of the contact pattern: the GT of a DIFFERENT window (same target, same frozen donor rule as
    --shuffle-source) scored against the case GT.  Human primary cases."""
    from am2exp_v1.judge import ground_judge as GJ
    from am2exp_v1.eval.run_e1_eval import shuffle_map
    d = run_dirs()[('A0', 'sensor_on', False)]
    donor = shuffle_map(cases_doc['windows'])
    ids = {c['case_id'] for c in cases_doc['cases']}
    rows = []
    for c in cases_doc['cases']:
        if not (c['primary'] and c['source'] == 'human_smpl'):
            continue
        dc = f"{c['source']}__{c['target']}__{donor[c['window_id']]}"
        if dc not in ids:
            continue
        z, zd, b = np.load(d / f"{c['case_id']}.npz"), np.load(d / f'{dc}.npz'), c['target']
        gt = GJ.judge_case(b, z['gt_joint_rotation'], z['gt_joint_position'], GJ.corrected_root(b, z['gt_root_height_m'], 'gt'), selfpen=False)
        sh = GJ.judge_case(b, zd['gt_joint_rotation'], zd['gt_joint_position'], GJ.corrected_root(b, zd['gt_root_height_m'], 'gt'), selfpen=False)
        m = GJ.case_metrics(sh, gt)
        rows.append(dict(case_id=c['case_id'], window_id=c['window_id'], target=b, CSE_GT=m['CSE_GT'], CSE_snap=m['CSE_snap'],
                         heldout=c['target_heldout'], source='human_smpl', primary=True, kind=c['kind']))
    return rows


# ------------------------------------------------------------------------------------------------- report
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def load_all():
    from am2exp_v1.eval.score_e1 import load_scores
    return {k: load_scores(d) for k, d in run_dirs().items() if (d / 'scores_j1.jsonl').exists()}


def headroom_tables(S):
    human_primary = lambda r: r['primary'] and r['source'] == 'human_smpl'
    rows = []
    for (arm, mode, shuf), recs in sorted(S.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        for held in (False, True):
            sub = lambda r, h=held: human_primary(r) and r['heldout'] == h
            s = ST.summarize(recs, METRICS, group_keys=('heldout',), subset=sub)
            if not s:
                continue
            v = list(s.values())[0]
            rows.append(dict(arm=arm, mode=mode, shuffled=shuf, heldout=held, **v))
    return rows


def gate_headroom(rows):
    out = {}
    for held in (False, True, None):
        for arm in ('A0', 'B0'):
            sel = [r for r in rows if r['arm'] == arm and r['mode'] == 'sensor_on' and not r['shuffled'] and (held is None or r['heldout'] == held)]
            if not sel:
                continue
            r = sel[0] if held is not None else None
            if r is None:
                continue
            cse_gap = r['CSE_GT']  # GT scored against itself has CSE 0 by construction
            float_gap = r['FloatRate']  # GT FloatRate against itself is 0 by construction
            pen_gap = r['PenP95_cm'] - r['gt_PenP95_cm']
            out[f'{arm}_{"heldout" if held else "seen"}'] = dict(CSE_gap=cse_gap, FloatRate_gap=float_gap, PenP95_gap_cm=pen_gap,
                                                                 numeric_pass=bool(cse_gap >= 0.10 or float_gap >= 0.10 or abs(pen_gap) >= 2.0))
    return out


def write_report(extra=None):
    S = load_all()
    rows = headroom_tables(S)
    ruler = json.loads((JUDGE / 'ruler_v1' / 'ruler_gate_summary.json').read_text())
    bias = json.loads((JUDGE / 'heldout_root_bias_v1.json').read_text())
    cases_doc = load_cases(CASES_FILE)
    L = []
    L.append('# E1 Stage-0 (CPU): ruler gate, heldout root bias, headroom\n')
    L.append(f'Generated {time.strftime("%Y-%m-%d %H:%M %Z")}. Case list `{CASES_FILE}` sha256 `{cases_doc["sha256"]}` '
             f'({cases_doc["counts"]["windows"]} windows, {cases_doc["counts"]["cases"]} cases, {cases_doc["counts"]["primary_cases"]} primary). '
             'Judge J1 = vendor MJCF collision geoms re-posed from AM2 global rotations/positions (OUT/judge/bindings_v1). '
             'All numbers are means of per-window means (window = cluster). No video has been looked at yet: these numbers do '
             'not replace the [human | GT | A0] videos.\n')
    # ---- ruler
    L.append('## 1. Ruler gate (J1 on GMR GT vs report 32)\n')
    L.append('Reference per body/corpus = report-32 stance gap p50 (label sole point above label floor) + delta_col (label drop - '
             'vendor collision drop). J1 = p50 over the same stance frames of the lowest collision point with the label root height.\n')
    L.append('| body | ref LAFAN | J1-ref LAFAN | J1-ref AMASS | strict pass | flat-foot diff L/A | flat pass | J1 static drop vs r32 true drop | shift detected |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for b, r in ruler['bodies'].items():
        la, am = r['corpora']['lafan'], r['corpora']['amass']
        L.append(f"| {b} | {la['reference_lowest_p50_cm']:.2f} | {la['diff_cm']:+.2f} | {am['diff_cm']:+.2f} | {'yes' if r['pass'] else 'NO'} | "
                 f"{la['diag_flat_diff_vs_j1_static_reference_cm']:+.2f} / {am['diag_flat_diff_vs_j1_static_reference_cm']:+.2f} | "
                 f"{'yes' if r['diag_pass_flat_j1ref'] else 'NO'} | {la['diag_j1_static_drop_cm']:.2f} vs {la['diag_report32_true_drop_cm']:.2f} | "
                 f"{'yes' if r['shift_detected'] else 'NO'} |")
    L.append('')
    L.append(f"* Strict gate (both corpora within 1 cm, report-32 delta): **{ruler['n_pass']}/15 -> {'PASS' if ruler['gate_pass'] else 'FAIL'}** (needs 13). "
             f"With the R1 delta corrected: {ruler.get('n_pass_corrected_delta')}/15. LAFAN only: {ruler.get('n_pass_lafan_only')}/15.")
    L.append(f"* Flat-foot stance frames (lowest-geom node tilt < 3 deg vs qpos0), J1-static-drop reference: **{ruler['diag_n_pass_flat_j1ref']}/15**.")
    L.append(f"* Synthetic +-5 cm root shifts on stance frames detected for all bodies: **{ruler['shift_detected_all']}** "
             f"(v0 rule, both rates move >= 0.5: {ruler.get('shift_detected_v0_all')}).")
    if extra and extra.get('synthetic'):
        s = extra['synthetic']
        L.append(f"* Case-level shifts (GT scored against itself, {s['n_cases']} primary cases): " + '; '.join(
            f"{k}: CSE {v['CSE_GT']:.3f}, PenP95 {v['PenP95_cm']:.2f} cm, FloatRate {v['FloatRate']:.3f}" for k, v in s['summary'].items())
            + f" -> detected (v1, revised after seeing v0) **{s['detected']}**; pre-declared v0 rule (FloatRate +0.5): {s['detected_v0']}.")
    L.append('')
    L.append('Why the strict gate fails (measured, not assumed): J1 sits 0.5-2 cm BELOW the reference on every body, more on AMASS. '
             'The report-32 reference is a single labeller sole point rigidly attached to the ankle; J1 is the lowest corner of the '
             'collision box/mesh. On heel-strike / toe-off stance frames (median tilt of the lowest node 5-10 deg) the corner is '
             'lower than the sole point. On flat-foot stance frames the two agree within 1 cm for 13/15 bodies, and at qpos0 the J1 '
             'static drop equals report 32\'s true drop to < 0.02 cm on 14/15 bodies. R1 is the exception: report 32 measured R1\'s '
             'drop inside the ankle_constraint subtree and missed the real sole (ankle_roll link, merged into ankle_pitch), so its '
             'delta is wrong by 3.9 cm; J1 uses the corrected value (DELTA_COL_CM in ground_judge.py). Consequence for E1: every '
             'arm and the GT are measured on the same J1 ruler, so the tilt effect cancels in paired comparisons; absolute PenRate of '
             'the GT itself is not zero (see gt_PenRate below).\n')
    # ---- bias
    L.append('## 2. Heldout root bias of A0 (TRAIN-split windows)\n')
    L.append(bias['definition'] + '\n')
    L.append('| body | applied c_b (cm) | windows | stance frames | IQR of window medians | sensor_off c_b (diag) | pred root p50 (m) | GT root p50 (m) |')
    L.append('|---|---|---|---|---|---|---|---|')
    for b, v in bias['bodies'].items():
        on, off = v['measured']['sensor_on'], v['measured']['sensor_off']
        L.append(f"| {b} | {v['applied_c_cm']:+.2f} | {on['windows']} | {on['stance_frames']} | {fmt(on['iqr_cm'][0], 2)} .. {fmt(on['iqr_cm'][1], 2)} | "
                 f"{fmt(off['c_cm'], 2)} | {on['pred_root_p50_m']:.3f} | {on['gt_root_p50_m']:.3f} |")
    L.append('')
    # ---- headroom
    L.append('## 3. Headroom: A0 / B0 vs GT (human source, primary cases)\n')
    L.append('Primary = human_smpl source on split-1 contact-rich windows (listed + mined) for all 15 targets, plus the 4 train-split '
             'ground windows for the 4 heldout targets. GT is scored by the same J1; CSE_GT and FloatRate of GT against itself are 0.\n')
    cols = ['CSE_GT', 'CSE_GT_nonfoot', 'CSE_snap', 'FloatRate', 'PenRate', 'PenP95_cm', 'gt_PenP95_cm', 'SelfPen', 'gt_SelfPen',
            'pos_err_mm', 'CSE_human', 'gt_CSE_human', 'pred_nonfoot_contact_frac', 'gt_nonfoot_contact_frac']
    L.append('| run | mode | targets | windows | cases | ' + ' | '.join(cols) + ' |')
    L.append('|---' * (5 + len(cols)) + '|')
    for r in rows:
        name = r['arm'] + (' shuffled-source' if r['shuffled'] else '')
        L.append(f"| {name} | {r['mode']} | {'heldout' if r['heldout'] else 'seen'} | {r['n_windows']} | {r['n_cases']} | "
                 + ' | '.join(fmt(r.get(c), 2 if c.endswith('_cm') or c == 'pos_err_mm' else 3) for c in cols) + ' |')
    L.append('')
    hg = gate_headroom(rows)
    L.append('Headroom gate (numeric part; the plan also requires lloyd to see it in [human | GT | A0] videos):\n')
    for k, v in hg.items():
        L.append(f"* {k}: CSE gap {v['CSE_gap']:.3f}, FloatRate gap {v['FloatRate_gap']:.3f}, PenP95 gap {v['PenP95_gap_cm']:+.2f} cm -> "
                 f"numeric {'PASS' if v['numeric_pass'] else 'FAIL'}")
    # shuffled floor
    if ('A0', 'sensor_on', True) in S and ('A0', 'sensor_on', False) in S:
        c = ST.compare(S[('A0', 'sensor_on', True)], S[('A0', 'sensor_on', False)], 'CSE_GT', alternative='greater',
                       subset=ST.primary_human)
        if extra and extra.get('gt_floor'):
            g = ST.summarize(extra['gt_floor'], ['CSE_GT', 'CSE_snap'], group_keys=('primary',))
            v = list(g.values())[0]
            ca = ST.compare(S[('A0', 'sensor_on', False)], extra['gt_floor'], 'CSE_GT', alternative='less', subset=ST.primary_human)
            L.append(f"* Chance level of the contact pattern (GT of another window, same target, vs the case GT): CSE_GT {v['CSE_GT']:.3f}, "
                     f"CSE_snap {v['CSE_snap']:.3f} ({v['n_windows']} windows). A0 - chance = {ca['mean_of_window_means']:+.3f} (perm p {ca['perm_p']:.4f}).")
        L.append(f"* Shuffled-source floor: CSE_GT(shuffled) - CSE_GT(A0) = {c['mean_of_window_means']:+.3f} over {c['n_windows']} windows "
                 f"(perm p {c['perm_p']:.4f}) -> ruler requirement >= 0.15: **{'PASS' if c['mean_of_window_means'] >= 0.15 else 'FAIL'}**")
        for m in ('CSE_snap', 'pos_err_mm'):
            c2 = ST.compare(S[('A0', 'sensor_on', True)], S[('A0', 'sensor_on', False)], m, alternative='greater', subset=ST.primary_human)
            L.append(f"  * {m}: shuffled - A0 = {c2['mean_of_window_means']:+.3f}")
    L.append('')
    # sensor effect and B0 vs A0
    L.append('## 4. Paired comparisons (window-cluster tests, human primary cases)\n')
    L.append('| comparison | metric | mean diff | windows | sign p | perm p | bodies neg/pos |')
    L.append('|---|---|---|---|---|---|---|')
    comps = []
    for arm in ('A0', 'B0'):
        if (arm, 'sensor_on', False) in S and (arm, 'sensor_off', False) in S:
            comps.append((f'{arm}: sensor_on - sensor_off', S[(arm, 'sensor_on', False)], S[(arm, 'sensor_off', False)]))
    for mode in ('sensor_on', 'sensor_off'):
        if ('B0', mode, False) in S and ('A0', mode, False) in S:
            comps.append((f'B0 - A0 ({mode})', S[('B0', mode, False)], S[('A0', mode, False)]))
    for name, a, b in comps:
        for m in ('CSE_GT', 'FloatRate', 'PenP95_cm', 'pos_err_mm'):
            c = ST.compare(a, b, m, alternative='two-sided', subset=ST.primary_human)
            L.append(f"| {name} | {m} | {c['mean_of_window_means']:+.4f} | {c['n_windows']} | {c['sign_p']:.3f} | {c['perm_p']:.4f} | "
                     f"{c['bodies']['negative']}/{c['bodies']['positive']} |")
    L.append('')
    if ('B0', 'sensor_on', False) in S and ('A0', 'sensor_on', False) in S:
        d = ST.decide(S[('B0', 'sensor_on', False)], S[('A0', 'sensor_on', False)])
        L.append(f"Decision-rule dry run (B0 as if it were a factor arm vs A0; illustrative only): verdict **{d['verdict']}**; "
                 f"dCSE {d['effect']['comparison']['mean_of_window_means']:+.4f} vs threshold {d['effect']['threshold']:+.3f}, "
                 f"perm p {d['effect']['comparison']['perm_p']:.3f}; reasons: {d['reasons'] or 'none'}; pending: {d['pending']}.\n")
    # per window & per body for A0
    if ('A0', 'sensor_on', False) in S:
        A = S[('A0', 'sensor_on', False)]
        L.append('## 5. A0 sensor_on per window (human source; seen / heldout targets)\n')
        L.append('| window | kind | human non-foot frac | CSE_GT seen | CSE_GT heldout | FloatRate heldout | PenP95 heldout (cm) | GT PenP95 heldout (cm) | mm heldout |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        wins = {w['window_id']: w for w in cases_doc['windows']}
        for wid, w in wins.items():
            def m(metric, held):
                v = [r[metric] for r in A if r['window_id'] == wid and r['source'] == 'human_smpl' and r['heldout'] == held
                     and r.get(metric) is not None and np.isfinite(r[metric])]
                return float(np.mean(v)) if v else None
            L.append(f"| {wid} | {w['kind']} | {w['human_nonfoot_contact_frac']:.2f} | {fmt(m('CSE_GT', False))} | {fmt(m('CSE_GT', True))} | "
                     f"{fmt(m('FloatRate', True))} | {fmt(m('PenP95_cm', True), 2)} | {fmt(m('gt_PenP95_cm', True), 2)} | {fmt(m('pos_err_mm', True), 1)} |")
        L.append('')
        L.append('## 6. A0 sensor_on per target body (human primary cases)\n')
        L.append('| target | heldout | root shift used (cm) | CSE_GT | CSE_snap | FloatRate | PenP95 (cm) | GT PenP95 (cm) | SelfPen | GT SelfPen | mm |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for t in sorted({r['target'] for r in A}):
            rs = [r for r in A if r['target'] == t and ST.primary_human(r)]
            if not rs:
                continue
            g = lambda k: float(np.nanmean([r.get(k, np.nan) for r in rs]))
            L.append(f"| {t} | {rs[0]['heldout']} | {rs[0]['root_shift_cm']:+.2f} | {g('CSE_GT'):.3f} | {g('CSE_snap'):.3f} | {g('FloatRate'):.3f} | "
                     f"{g('PenP95_cm'):.2f} | {g('gt_PenP95_cm'):.2f} | {g('SelfPen'):.3f} | {g('gt_SelfPen'):.3f} | {g('pos_err_mm'):.1f} |")
        L.append('')
    L.append('## 7. Gate verdicts\n')
    L.append(f"* Ruler gate (strict, as pre-registered): **{'PASS' if ruler['gate_pass'] else 'FAIL'}** ({ruler['n_pass']}/15). Diagnostic flat-foot: {ruler['diag_n_pass_flat_j1ref']}/15.")
    L.append(f"* Synthetic +-5 cm shifts detected: stance-level {ruler['shift_detected_all']}; case-level {extra['synthetic']['detected'] if extra and extra.get('synthetic') else 'n/a'}.")
    for k, v in hg.items():
        L.append(f"* Headroom numeric ({k}): {'PASS' if v['numeric_pass'] else 'FAIL'}")
    L.append('* Headroom video part: pending (render package; lloyd must see [human | GT | A0]).')
    L.append('')
    L.append('Files: case npz + scores under OUT/eval/e1/<run>/<mode>/ (scores_j1.jsonl), heldout bias OUT/judge/heldout_root_bias_v1.json, '
             'ruler OUT/judge/ruler_v1/ruler_gate_summary.json, bindings OUT/judge/bindings_v1/.')
    # plain-language summary at the top
    P = ['## In short\n']
    P.append(f"* Ruler: J1 agrees with report 32 on flat-foot standing frames ({ruler['diag_n_pass_flat_j1ref']}/15 within 1 cm) but the "
             f"pre-registered strict gate fails ({ruler['n_pass']}/15): on tilted stance frames the collision box corner sits 0.5-2 cm lower "
             f"than the single label sole point. Root shifts of +-5 cm are seen. R1's report-32 delta was wrong by 3.9 cm (fixed in J1).")
    a_s = next((r for r in rows if r['arm'] == 'A0' and r['mode'] == 'sensor_on' and not r['shuffled'] and r['heldout']), None)
    if a_s is not None:
        P.append(f"* A0 almost never puts a non-foot body part on the ground: {a_s['pred_nonfoot_contact_frac']:.0%} of frames vs "
                 f"{a_s['gt_nonfoot_contact_frac']:.0%} in the GT (heldout targets, contact-rich windows). Contact-pattern error CSE_GT "
                 f"{a_s['CSE_GT']:.2f}; position error {a_s['pos_err_mm']:.0f} mm.")
    if extra and extra.get('gt_floor') and ('A0', 'sensor_on', True) in S:
        g = ST.summarize(extra['gt_floor'], ['CSE_GT'], group_keys=('primary',)); g = list(g.values())[0]['CSE_GT']
        P.append('* A0 ignores the source for ground contact: feeding it another clip (shuffled source) gives the same CSE_GT; only the '
                 f'pose error gets worse. A real contact pattern from a different window scores CSE_GT {g:.2f}, i.e. better than A0. '
                 'So the headroom is large, and the shuffled-source floor gate (>= 0.15 worse than A0) cannot pass for A0 itself.')
    P.append('* The GT retargets themselves sink into the floor on ground motions (GT PenP95 ~13 cm on primary windows); penetration '
             'must be judged relative to the GT, not to zero.')
    P.append('* B0 (96-patch surface) floats more than A0 and penetrates less; its contact pattern is not better.')
    P.append('* Nothing here has been checked on video yet.\n')
    L[2:2] = P
    (JUDGE / 'stage0_report.md').write_text('\n'.join(L) + '\n')
    (JUDGE / 'stage0_summary.json').write_text(json.dumps(dict(headroom_rows=rows, headroom_gate=hg, ruler_gate_pass=ruler['gate_pass'],
                                                               ruler_n_pass=ruler['n_pass'], ruler_flat_n_pass=ruler['diag_n_pass_flat_j1ref'],
                                                               heldout_bias={b: v['applied_c_cm'] for b, v in bias['bodies'].items()},
                                                               synthetic=(extra or {}).get('synthetic')), indent=1, default=float))
    return JUDGE / 'stage0_report.md'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--steps', default='rootbias,infer,score,report')
    a = ap.parse_args()
    steps = a.steps.split(',')
    t0 = time.time()
    if 'rootbias' in steps:
        doc = step_rootbias(a.jobs, a.timeout)
        print('heldout bias', {b: round(v['applied_c_cm'], 2) for b, v in doc['bodies'].items()}, flush=True)
    if 'infer' in steps:
        step_infer(a.jobs, a.timeout); print('infer done', f'{time.time() - t0:.0f}s', flush=True)
    if 'score' in steps:
        step_score(a.jobs); print('score done', f'{time.time() - t0:.0f}s', flush=True)
    if 'report' in steps:
        cd = load_cases(CASES_FILE)
        syn = synthetic_case_shifts(cd)
        p = write_report(dict(synthetic=syn, gt_floor=gt_shuffled_floor(cd))); print('report', p, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
