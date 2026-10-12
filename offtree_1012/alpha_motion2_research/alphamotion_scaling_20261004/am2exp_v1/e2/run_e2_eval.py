"""E2 evaluation runner (CPU).  Stage-0: labels, descriptor freeze, floors, AN, A0/A_M direct mapping, OOD.

Usage:
  PY -m e2.run_e2_eval stage0 [--out OUT/eval/e2] [--threads 8]
  PY -m e2.run_e2_eval arm --name G1 --checkpoint PATH [--model-class module:Class] [--config run/config.json]

Arms are scored by e2.closure_judge only (independent of training-loss code).  A->F rows: Adroit source
pose9 (hand_<ds>.npz / pen payload eval rows), no condition, no hidden joints; c_F from slide_native[8,9].
Never conditions on Franka joints 8/9.  F->A: Franka eval episodes {9,12,15,17} (by episode index), stride-40
rows as in training, squeeze proxy from recorded_ctrl at payload_row_index (unstrided payload).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'e2'  # noqa: A001

from ._env import E2_EVAL, E2_DATA, ROOT, FRANKA_PAYLOAD, FRANKA_PAYLOAD_JSON  # noqa: E402
from . import hand_data as hd  # noqa: E402
from . import closure_judge as cj  # noqa: E402
from . import opposition as op  # noqa: E402

DATASETS = ['hammer', 'relocate', 'pen']
SPLITS = {'hammer': ['dev', 'test'], 'relocate': ['dev', 'test'], 'pen': ['eval']}
FRANKA_EVAL_EPISODES = [9, 12, 15, 17]
STAGE0_CHECKPOINTS = {
    'A0_S@1000': 'runs/A_S_seed0/checkpoint_step1000.pt',
    'A0_S@3000': 'runs/A_S_seed0/checkpoint_step3000.pt',
    'A_M@1000': 'runs/A_M_seed0/checkpoint_step1000.pt',
    'A_M@3000': 'runs/A_M_seed0/checkpoint_step3000.pt',
}
SELECTION_CRITERION = 'mean over {hammer, relocate} of median |c_A - d_obj| (mm) on DEV grasp frames (g=1), raw c_A'
G0_GATE_BA = 0.75
OPEN_MM = op.FRANKA_MAX_CLEARANCE_CM * 10        # 78 mm
CLOSED_MM = -op.FRANKA_CLEARANCE_OFFSET_CM * 10  # q=0 -> -2 mm


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


# ---------------------------------------------------------------------------------------------
# data + labels
# ---------------------------------------------------------------------------------------------
def load_sets():
    sets = {'hammer': hd.load_hand('hammer'), 'relocate': hd.load_hand('relocate'), 'pen': hd.load_pen()}
    for ds, D in sets.items():
        D['episode'] = np.asarray(D['episode_index'])
    return sets


def labels_for(ds, D, out_dir, scene=None):
    src = E2_DATA / ('pen_object.npz' if ds == 'pen' else f'hand_{ds}.npz')
    key = hashlib.sha256((sha_file(src) + sha_file(Path(cj.__file__))).encode()).hexdigest()
    cache = Path(out_dir) / 'labels' / f'labels_{ds}.npz'
    if cache.exists():
        with np.load(cache) as z:
            if str(z['key']) == key:
                return {k: z[k] for k in z.files}
    lab, dist, opp = cj.grasp_labels(D, scene)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, labels=lab, dist_mm=dist, opposite=opp, key=np.array(key))
    return {'labels': lab, 'dist_mm': dist, 'opposite': opp, 'key': np.array(key)}


def label_counts(sets, labels):
    out = {}
    for ds in DATASETS:
        D, L = sets[ds], labels[ds]['labels']
        for s in sorted(set(D['split'].tolist())):
            m = D['split'] == s
            out[f'{ds}/{s}'] = {'frames': int(m.sum()), 'episodes': int(len(np.unique(D['episode'][m]))), 'grasp_g1': int((L[m] == 1).sum()),
                                'free_g0': int((L[m] == 0).sum()), 'excluded': int((L[m] == -1).sum())}
    return out


def PG_from(D, chunk=None):
    from ._sealed_math import rot6d_to_matrix
    P = torch.as_tensor(np.asarray(D['position_cm']), dtype=torch.float32)
    G = rot6d_to_matrix(torch.as_tensor(np.asarray(D['global_rot6']), dtype=torch.float32))
    return P, G


def descriptor_mm(D, variant, tau=None):
    P, G = PG_from(D)
    tau = tau if tau is not None else (op.DEFAULT_TAU_CM if op.VARIANTS[variant][1] != 'thumb_facing' else op.FACING_SOFTMIN_TAU_CM)
    out = []
    with torch.no_grad():
        for s in range(0, len(P), 4096):
            out.append(op.adroit_descriptor(P[s:s + 4096], G[s:s + 4096], variant=variant, tau_cm=tau)['c_A'].numpy() * 10)
    return np.concatenate(out), tau


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).statistic)


def score(c_F_mm, D, L, split, ref_cA_mm=None, boot=1000):
    m = D['split'] == split
    c = np.asarray(c_F_mm)[m]
    lab = L[m]
    d = float(D['obj_diameter_mm'])
    st = cj.gripper_state(c, d)
    r = cj.balanced_accuracy(st, lab)
    pos, neg = lab == 1, lab == 0
    r['BA_nothold'] = 0.5 * (float((st[pos] == cj.HOLD).mean()) + float((st[neg] != cj.HOLD).mean())) if pos.any() and neg.any() else float('nan')
    r['ci95_episode_bootstrap'] = cj.episode_bootstrap(st, lab, D['episode'][m], n_boot=boot, seed=0)
    r['median_cF_mm_grasp'] = float(np.median(c[pos])) if pos.any() else float('nan')
    r['median_cF_mm_free'] = float(np.median(c[neg])) if neg.any() else float('nan')
    r['object_diameter_mm'] = d
    r['hold_band_mm'] = [d - cj.HOLD_LOW_MM, d + cj.HOLD_HIGH_MM]
    if ref_cA_mm is not None:
        r['spearman_cF_vs_cA'] = spearman(c, np.asarray(ref_cA_mm)[m])
    return r


def shuffle_within_episode(x, episode, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(x, copy=True)
    for e in np.unique(episode):
        idx = np.flatnonzero(episode == e)
        y[idx] = np.asarray(x)[rng.permutation(idx)]
    return y


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def model_AF(model, bodies, pose9, batch=512):
    A, F = bodies['adroit_pen'], bodies['relay_franka_full_v2']
    q = []
    for s in range(0, len(pose9), batch):
        x = torch.as_tensor(pose9[s:s + batch], dtype=torch.float32)
        n = len(x)
        out = model.forward_kinematic(x, A.geom(n), F.geom(n))
        q.append(out['slide_native'][:, [op.FRANKA_LEFT, op.FRANKA_RIGHT]].float().numpy())
    q = np.concatenate(q)
    raw = (q.sum(1) - op.FRANKA_CLEARANCE_OFFSET_CM) * 10
    clamped = (np.clip(q, 0, 4).sum(1) - op.FRANKA_CLEARANCE_OFFSET_CM) * 10
    return {'q_cm': q, 'c_F_mm_raw': raw, 'c_F_mm_clamped': clamped}


@torch.no_grad()
def model_recon(model, bodies, pose9, batch=512):
    from ._sealed_math import rot6d_to_matrix
    A = bodies['adroit_pen']
    pe, re = [], []
    for s in range(0, len(pose9), batch):
        x = torch.as_tensor(pose9[s:s + batch], dtype=torch.float32)
        n = len(x)
        out = model.forward_kinematic(x, A.geom(n), A.geom(n))
        gt = x[..., 6:] * A.reach
        pe.append((out['joint_position'].float() - gt).norm(dim=-1).mean(1).numpy())
        Rp = rot6d_to_matrix(out['rot6'].float())
        Rg = rot6d_to_matrix(x[..., :6])
        cos = (((Rp.transpose(-1, -2) @ Rg).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        re.append(torch.rad2deg(torch.acos(cos)).mean(1).numpy())
    return np.concatenate(pe), np.concatenate(re)


def franka_eval_rows(F):
    meta = json.loads(Path(FRANKA_PAYLOAD_JSON).read_text())
    eps = meta['episodes']
    pr = np.asarray(F.arrays['payload_row_index'])
    sel, ep_of = [], []
    for e in FRANKA_EVAL_EPISODES:
        a, b = eps[e]['row_range']
        if eps[e]['split'] != 1:
            raise RuntimeError(f'Franka episode {e} is not an eval episode')
        idx = np.flatnonzero((pr >= a) & (pr < b))
        sel.append(idx)
        ep_of.append(np.full(len(idx), e))
    rows = np.concatenate(sel)
    with np.load(FRANKA_PAYLOAD) as z:
        q = z['recorded_robot_qpos'][pr[rows]]
        ctrl = z['recorded_ctrl'][pr[rows]]
    squeeze = (ctrl[:, 7] < q[:, 7] - 0.002) & (ctrl[:, 8] < q[:, 8] - 0.002)
    return rows, np.concatenate(ep_of), squeeze


@torch.no_grad()
def model_FA(model, bodies, rows, batch=256):
    A, F = bodies['adroit_pen'], bodies['relay_franka_full_v2']
    Ps, Gs = [], []
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        n = len(r)
        x = F.tensor('pose9', r, 'cpu').float()
        sF = F.tensor('slide_normalized', r, 'cpu').float()
        out = model.forward_kinematic(x, F.geom(n), A.geom(n), source_slide_normalized=sF)
        Ps.append(out['joint_position'].float().numpy())
        Gs.append(out['joint_rotation'].float().numpy())
    return np.concatenate(Ps), np.concatenate(Gs)


def closure_summary(res, squeeze):
    """res from cj.hand_fit_test.  closure_BA_fit = 0.5 (P(fit | squeeze) + P(fit | not squeeze))."""
    def rate(x, m):
        return float(x[m].mean()) if m.any() else float('nan')
    hold, fit = res['hold'], res['fit']
    out = {'hold_rate_squeeze': rate(hold, squeeze), 'hold_rate_nonsqueeze': rate(hold, ~squeeze),
           'fit_rate_squeeze': rate(fit, squeeze), 'fit_rate_nonsqueeze': rate(fit, ~squeeze),
           'hold_plus_margin_rate_squeeze': rate(res['hold_plus_margin'], squeeze),
           'n_squeeze': int(squeeze.sum()), 'n_nonsqueeze': int((~squeeze).sum()), 'margin_cm': res['margin_cm'],
           'median_tch_gap_mm_squeeze': float(np.median(res['tch_gap_mm'][squeeze])) if squeeze.any() else float('nan')}
    out['closure_BA_fit'] = 0.5 * (out['fit_rate_squeeze'] + out['fit_rate_nonsqueeze'])
    # backwards-compatible names (one-sided hold)
    out['pass_rate_squeeze'], out['pass_rate_nonsqueeze'] = out['hold_rate_squeeze'], out['hold_rate_nonsqueeze']
    return out


# ---------------------------------------------------------------------------------------------
# descriptor selection / freeze
# ---------------------------------------------------------------------------------------------
def select_descriptor(sets, labels, refreeze=False):
    table = {}
    for v in op.VARIANTS:
        row = {'family': 'planned' if v in op.PLANNED_VARIANTS else 'thumb_facing'}
        meds = []
        for ds in ['hammer', 'relocate']:
            D, L = sets[ds], labels[ds]['labels']
            c, tau = descriptor_mm(D, v)
            g = (D['split'] == 'dev') & (L == 1)
            med = float(np.median(np.abs(c[g] - float(D['obj_diameter_mm']))))
            meds.append(med)
            row[f'{ds}_dev_median_abs_err_mm'] = med
            row[f'{ds}_dev_median_cA_grasp_mm'] = float(np.median(c[g]))
            st = cj.gripper_state(np.clip(c, 0, OPEN_MM), float(D['obj_diameter_mm']))
            m = D['split'] == 'dev'
            ba = cj.balanced_accuracy(st[m], L[m])
            row[f'{ds}_dev_AN_BA'] = ba['BA']
            row[f'{ds}_dev_AN_TPR'] = ba['P_hold_given_grasp']
            row['tau_cm'] = tau
            # post-freeze DIAGNOSTIC (never used by the selection rule below): same number on TEST
            mt = D['split'] == 'test'
            row[f'{ds}_test_AN_BA_diagnostic'] = cj.balanced_accuracy(st[mt], L[mt])['BA']
        row['criterion_mm'] = float(np.mean(meds))
        table[v] = row
    best = min(table, key=lambda k: table[k]['criterion_mm'])
    best_planned = min(op.PLANNED_VARIANTS, key=lambda k: table[k]['criterion_mm'])
    choice = {'variant': best, 'tau_cm': table[best]['tau_cm'], 'pad_source': 'mjcf'}
    evidence = {'criterion': SELECTION_CRITERION, 'table': table, 'best_planned_variant': best_planned,
                'data_sha256': {ds: sha_file(E2_DATA / f'hand_{ds}.npz') for ds in ['hammer', 'relocate']},
                'note': ('Planned virtual-finger variants were evaluated first; all were far from the object size on these '
                         'enveloping grasps, so the thumb-facing family (thumb pads vs any facing phalanx/palm pad) was added '
                         'in Stage-0 and the whole table re-ranked by the same criterion on DEV only.  Test untouched by selection.')}
    if op.FROZEN_PATH.exists() and not refreeze:
        frozen = op.load_frozen()
        status = 'already_frozen_' + ('consistent' if frozen['variant'] == best else 'DIFFERS_from_recomputed_choice')
    else:
        frozen = op.freeze(choice, evidence, overwrite=refreeze)
        status = 'frozen_now'
    return {'choice': choice, 'frozen': {k: frozen[k] for k in ['variant', 'tau_cm', 'pad_source']}, 'status': status, 'table': table,
            'best_planned_variant': best_planned}


# ---------------------------------------------------------------------------------------------
# stage 0
# ---------------------------------------------------------------------------------------------
def franka_train_median_mm():
    with np.load(FRANKA_PAYLOAD) as z:
        fi, sp, sl = z['frame_index'], z['split'], z['slide_position_m']
    rows = (fi % 40 == 0) & (sp == 0)
    return float(np.median((sl[rows, op.FRANKA_LEFT] + sl[rows, op.FRANKA_RIGHT]) * 1000 - 2.0))


def stage0(out_dir=None, threads=8, boot=1000, refreeze=False, checkpoints=None):
    t0 = time.time()
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    out_dir = Path(out_dir) if out_dir is not None else E2_EVAL
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = load_sets()
    scene = cj.JudgeScene()
    labels = {ds: labels_for(ds, sets[ds], out_dir, scene) for ds in DATASETS}
    report = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'label_counts': label_counts(sets, labels),
              'hand_data_manifest': json.loads((E2_DATA / 'hand_data_manifest.json').read_text())}
    # pen opposition re-measure (audit: thumb pad AND finger pad within 5 mm of the pen ~ 0%)
    P, G = PG_from(sets['pen'])
    pads, _ = op.adroit_pads(P, G)
    seg = torch.as_tensor(sets['pen']['obj_segment_cm'], dtype=torch.float32)
    a, b = seg[:, 0], seg[:, 1]
    ab = b - a
    t = (((pads - a[:, None]) * ab[:, None]).sum(-1) / (ab * ab).sum(-1)[:, None]).clamp(0, 1)
    dd = ((pads - (a[:, None] + t[..., None] * ab[:, None])).norm(dim=-1) * 10 - float(sets['pen']['obj_radius_cm']) * 10).numpy()
    near = dd < 5
    report['pen_opposition_remeasure'] = {'thumb_and_any_finger_pad_within_5mm_frac': float((near[:, 0] & near[:, 1:].any(1)).mean()),
                                          'per_digit_within_5mm_frac(th,ff,mf,rf,lf)': near.mean(0).tolist(),
                                          'judge_g1_frac_all_pen_frames': float((labels['pen']['labels'] == 1).mean())}
    sel = select_descriptor(sets, labels, refreeze=refreeze)
    report['descriptor_selection'] = sel
    frozen = op.load_frozen()
    cA = {ds: descriptor_mm(sets[ds], frozen['variant'], frozen['tau_cm'])[0] for ds in DATASETS}
    arms = {}
    median_mm = franka_train_median_mm()
    report['franka_train_median_clearance_mm'] = median_mm

    def add(name, cmap, kind, extra=None):
        arms[name] = {'kind': kind, 'scores': {}, **(extra or {})}
        for ds in DATASETS:
            for s in SPLITS[ds]:
                arms[name]['scores'][f'{ds}/{s}'] = score(cmap[ds], sets[ds], labels[ds]['labels'], s, ref_cA_mm=cA[ds], boot=boot)
            arm_dir = out_dir / 'arms' / name.replace('@', '_at_')
            arm_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(arm_dir / f'{ds}.npz', c_F_mm=np.asarray(cmap[ds], np.float32), labels=labels[ds]['labels'],
                                episode=sets[ds]['episode'], split=sets[ds]['split'], frame_index=sets[ds]['frame_index'])

    add('constant_open', {ds: np.full(len(sets[ds]['split']), OPEN_MM) for ds in DATASETS}, 'floor')
    add('constant_closed', {ds: np.full(len(sets[ds]['split']), CLOSED_MM) for ds in DATASETS}, 'floor')
    add('constant_median', {ds: np.full(len(sets[ds]['split']), median_mm) for ds in DATASETS}, 'floor')
    an = {ds: np.clip(cA[ds], 0, OPEN_MM) for ds in DATASETS}
    add('AN', an, 'reference', {'descriptor': {k: frozen[k] for k in ['variant', 'tau_cm', 'pad_source']}})
    add('AN_shuffled', {ds: shuffle_within_episode(an[ds], sets[ds]['episode']) for ds in DATASETS}, 'floor')
    bp = sel['best_planned_variant']
    an_p = {ds: np.clip(descriptor_mm(sets[ds], bp)[0], 0, OPEN_MM) for ds in DATASETS}
    add(f'AN_planned_{bp}', an_p, 'diagnostic')
    # exact floor checks
    for s in ['dev', 'test']:
        ba = arms['constant_open']['scores'][f'hammer/{s}']['BA']
        if ba != 0.5:
            raise AssertionError(f'constant-open BA on hammer/{s} = {ba} != 0.5')
    # model arms (CPU)
    from . import bodies as eb
    checkpoints = checkpoints if checkpoints is not None else STAGE0_CHECKPOINTS
    bodies = None
    rows_F, ep_F, squeeze = None, None, None
    fa = {}
    ood = {}
    for name, rel in checkpoints.items():
        path = ROOT / rel
        model, cfg, step = eb.load_model(path)
        if bodies is None:
            bodies = eb.load_external_bodies(cfg)
            rows_F, ep_F, squeeze = franka_eval_rows(bodies['relay_franka_full_v2'])
            # sanity: pose9 of the pen payload eval rows equals the Body arrays
            pen_b = bodies['adroit_pen'].arrays['pose9']
            if np.abs(pen_b - sets['pen']['pose9']).max() > 1e-4:
                raise RuntimeError('pen pose9 from load_pen differs from the training Body')
        af = {ds: model_AF(model, bodies, sets[ds]['pose9']) for ds in DATASETS}
        add(name, {ds: af[ds]['c_F_mm_raw'] for ds in DATASETS}, 'model_direct',
            {'checkpoint': str(path), 'step': step, 'clamped_equals_raw': bool(all(np.allclose(af[ds]['c_F_mm_raw'], af[ds]['c_F_mm_clamped']) for ds in DATASETS)),
             'median_q01': {ds: float(np.median(af[ds]['q_cm'] / 4.0)) for ds in DATASETS},
             'cF_mm_range': {ds: [float(af[ds]['c_F_mm_raw'].min()), float(af[ds]['c_F_mm_raw'].max())] for ds in DATASETS}})
        add(name + '_shuffled', {ds: shuffle_within_episode(af[ds]['c_F_mm_raw'], sets[ds]['episode']) for ds in DATASETS}, 'floor')
        # OOD: same-body reconstruction
        o = {}
        for ds, s in [('pen', 'eval'), ('hammer', 'test'), ('relocate', 'test'), ('hammer', 'dev')]:
            m = sets[ds]['split'] == s
            pe, re = model_recon(model, bodies, sets[ds]['pose9'][m])
            o[f'{ds}/{s}'] = {'pos_err_cm_mean': float(pe.mean()), 'pos_err_cm_median': float(np.median(pe)), 'rot_err_deg_mean': float(re.mean())}
        o['ratio_hammer_test_over_pen_eval_pos'] = o['hammer/test']['pos_err_cm_mean'] / o['pen/eval']['pos_err_cm_mean']
        o['ratio_relocate_test_over_pen_eval_pos'] = o['relocate/test']['pos_err_cm_mean'] / o['pen/eval']['pos_err_cm_mean']
        o['ood_gate_hammer_le_2x'] = bool(o['ratio_hammer_test_over_pen_eval_pos'] <= 2.0)
        ood[name] = o
        # F->A closure on decoded hands
        Pd, Gd = model_FA(model, bodies, rows_F)
        cF_src = (bodies['relay_franka_full_v2'].arrays['slide_native'][rows_F][:, [op.FRANKA_LEFT, op.FRANKA_RIGHT]].sum(1) - op.FRANKA_CLEARANCE_OFFSET_CM)
        res = cj.hand_fit_test(Pd, Gd, cF_src, scene)
        fa[name] = closure_summary(res, squeeze)
        del model
    # F->A floors: rest hand and random pen-train hands
    if rows_F is not None:
        F = bodies['relay_franka_full_v2']
        cF_src = (F.arrays['slide_native'][rows_F][:, [op.FRANKA_LEFT, op.FRANKA_RIGHT]].sum(1) - op.FRANKA_CLEARANCE_OFFSET_CM)
        A = bodies['adroit_pen']
        rest_P = np.zeros((len(A.parents), 3))
        for j, p in enumerate(A.parents):
            if p >= 0:
                rest_P[j] = rest_P[p] + A.offsets[j]
        n = len(rows_F)
        res = cj.hand_fit_test(np.repeat(rest_P[None], n, 0), np.repeat(np.eye(3)[None, None], n, 0).repeat(len(A.parents), 1), cF_src, scene)
        fa['floor_rest_hand'] = closure_summary(res, squeeze)
        rng = np.random.default_rng(0)
        pen = sets['pen']
        tr = np.flatnonzero(pen['split'] == 'train')
        pick = rng.choice(tr, size=n)
        Pp, Gp = PG_from({'position_cm': pen['position_cm'][pick], 'global_rot6': pen['global_rot6'][pick]})
        res = cj.hand_fit_test(Pp.numpy(), Gp.numpy(), cF_src, scene)
        fa['floor_random_pen_train_hand'] = closure_summary(res, squeeze)
        report['franka_FA_rows'] = {'episodes': FRANKA_EVAL_EPISODES, 'rows': int(n), 'squeeze_frac': float(squeeze.mean()),
                                    'source_clearance_mm_median_squeeze': float(np.median(cF_src[squeeze]) * 10) if squeeze.any() else float('nan')}
    # closure-test validity on GT hammer grasps (judge sanity, not an arm)
    report['closure_test_validity'] = closure_validity(sets['hammer'], labels['hammer']['labels'], scene)
    report['arms'] = arms
    report['ood'] = ood
    report['FA_closure'] = fa
    hdev = arms['AN']['scores']['hammer/dev']['BA']
    report['G0_gate'] = {'AN_hammer_dev_BA': hdev, 'threshold': G0_GATE_BA, 'AN_pass': bool(hdev >= G0_GATE_BA),
                         'constant_open_hammer_dev_BA': arms['constant_open']['scores']['hammer/dev']['BA'],
                         'constant_open_hammer_test_BA': arms['constant_open']['scores']['hammer/test']['BA'],
                         'constant_open_exact_half': bool(arms['constant_open']['scores']['hammer/dev']['BA'] == 0.5 and arms['constant_open']['scores']['hammer/test']['BA'] == 0.5),
                         'passed': bool(hdev >= G0_GATE_BA and arms['constant_open']['scores']['hammer/dev']['BA'] == 0.5),
                         'caveat': 'descriptor family extended and selected on the same DEV episodes as this gate (planned family failed); hammer TEST BA of AN is the unbiased number'}
    report['surface_pad_check'] = op.surface_vs_mjcf_check()
    report['wall_seconds'] = time.time() - t0
    jdump(out_dir / 'stage0_summary.json', report)
    (out_dir / 'stage0_report.md').write_text(render_report(report))
    return report


def closure_validity(D, L, scene, n=200, seed=0):
    """Judge sanity on real hammer frames (hammer TEST split; dev was used to set the axis grid):
    grasp frames should hold a cylinder of the true handle diameter and not one 30 mm larger;
    free frames should mostly not hold the true diameter."""
    from ._sealed_math import rot6d_to_matrix
    rng = np.random.default_rng(seed)
    d_cm = float(D['obj_diameter_mm']) / 10
    out = {}
    for lv, tag in [(1, 'grasp'), (0, 'free')]:
        idx = np.flatnonzero((D['split'] == 'test') & (L == lv))
        idx = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
        P = D['position_cm'][idx]
        G = rot6d_to_matrix(torch.as_tensor(D['global_rot6'][idx])).numpy()
        sizes = [('true_diameter', d_cm), ('plus_30mm', d_cm + 3.0), ('minus_30mm', d_cm - 3.0)] if lv == 1 else [('true_diameter', d_cm), ('plus_30mm', d_cm + 3.0)]
        for name, c in sizes:
            r = cj.hand_closure_test(P, G, np.full(len(idx), c), scene)
            out[f'{tag}_{name}'] = float(r['pass'].mean())
        out[f'n_{tag}_frames'] = int(len(idx))
    return out


def fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        return 'nan' if not np.isfinite(x) else f'{x:.{nd}f}'
    return str(x)


def render_report(r):
    L = []
    g = r['G0_gate']
    sel = r['descriptor_selection']
    L.append('# E2 Stage-0 report (CPU only, no training)\n')
    L.append(f"Generated {r['utc']}. All numbers come from `e2/run_e2_eval.py stage0`; the judge is `e2/closure_judge.py` "
             '(MuJoCo collision capsules + dataset object pose; it never calls training-loss code).\n')
    L.append('## Short version\n')
    L.append(f"- **G-0 gate: {'PASS' if g['passed'] else 'FAIL'}.** Analytic retarget (AN) on hammer dev: balanced accuracy "
             f"{fmt(g['AN_hammer_dev_BA'])} (needs >= {g['threshold']}). Constant-open gives exactly {fmt(g['constant_open_hammer_dev_BA'], 4)} (dev) and "
             f"{fmt(g['constant_open_hammer_test_BA'], 4)} (test), as it must.")
    an_t = r['arms']['AN']['scores']['hammer/test']
    an_d = r['arms']['AN']['scores']['hammer/dev']
    L.append(f"- AN on hammer **test** (never used for any choice): BA {fmt(an_t['BA'])} "
             f"[{fmt(an_t['ci95_episode_bootstrap']['lo'])}, {fmt(an_t['ci95_episode_bootstrap']['hi'])}] (episode bootstrap 95%); "
             f"P(hold | grasp) falls from {fmt(an_d['P_hold_given_grasp'])} (dev) to {fmt(an_t['P_hold_given_grasp'])} (test). "
             + ('**The gate passes on dev but the analytic reference does not carry over to test: treat the G-0 pass as fragile.**'
                if an_t['BA'] < G0_GATE_BA else 'The test number also clears the gate.'))
    bp = sel['best_planned_variant']
    tp = sel['table'][bp]
    L.append(f"- The planned descriptor (thumb pad to virtual-finger pad, best planned variant `{bp}`) does **not** measure object size on these "
             f"grasps: on hammer dev grasps it reads {fmt(tp['hammer_dev_median_cA_grasp_mm'], 1)} mm for a 50 mm handle, AN BA {fmt(tp['hammer_dev_AN_BA'])}. "
             'Hammer and relocate are wrap (power) grasps: the palm, finger middle/proximal segments and the thumb side touch the object, '
             'the fingertip pads mostly do not face the thumb tip.')
    L.append(f"- Frozen descriptor: `{sel['frozen']['variant']}` (status: {sel['status']}). It measures the gap between a thumb pad and the nearest "
             'hand pad that faces it (any finger segment or the palm). It was chosen on dev only, by the pre-set rule '
             f"\"{SELECTION_CRITERION}\". Because the family was widened after the planned one failed, the dev gate is optimistic; read the test number.")
    for name in ['A0_S@1000', 'A0_S@3000', 'A_M@1000', 'A_M@3000']:
        if name in r['arms']:
            a = r['arms'][name]
            s = a['scores']['hammer/test']
            q = a.get('median_q01', {}).get('hammer', float('nan'))
            rng_ = a.get('cF_mm_range', {}).get('hammer', [float('nan')] * 2)
            L.append(f"- Existing model `{name}` mapping hand -> gripper directly: hammer test BA {fmt(s['BA'])}; median gripper clearance "
                     f"{fmt(s['median_cF_mm_grasp'], 1)} mm on grasp frames, {fmt(s['median_cF_mm_free'], 1)} mm on free frames "
                     f"(all hammer frames span {fmt(rng_[0], 1)}-{fmt(rng_[1], 1)} mm; median slide fraction {fmt(q, 3)}).")
    if r.get('FA_closure'):
        parts = [f"`{k}` {fmt(v['closure_BA_fit'])}" for k, v in r['FA_closure'].items()]
        L.append('- Gripper -> hand closure, balanced fit rate over squeeze / other frames: ' + '; '.join(parts) + '. '
                 'Read against the random-hand floor: a hand that ignores the gripper can still score on one side '
                 '(a fixed open hand fits the open frames), so only a clear margin above `floor_random_pen_train_hand` and above '
                 'a fixed-hand score means the hand follows the gripper gap. None of the existing models shows that.')
    for name, o in r.get('ood', {}).items():
        L.append(f"- OOD check `{name}`: same-body hand reconstruction error hammer test / pen eval = {fmt(o['ratio_hammer_test_over_pen_eval_pos'], 2)}x "
                 f"({'within' if o['ood_gate_hammer_le_2x'] else 'OVER'} the 2x limit).")
    L.append('\n## Grasp / non-grasp frame counts (closure judge labels)\n')
    L.append('| set | episodes | frames | grasp g=1 | free g=0 | excluded |')
    L.append('|---|---|---|---|---|---|')
    for k, v in r['label_counts'].items():
        L.append(f"| {k} | {v['episodes']} | {v['frames']} | {v['grasp_g1']} | {v['free_g0']} | {v['excluded']} |")
    pr = r['pen_opposition_remeasure']
    L.append(f"\nPen re-measure (audit said ~0%): thumb pad AND a finger pad within 5 mm of the pen = {fmt(pr['thumb_and_any_finger_pad_within_5mm_frac'] * 100, 2)}% of frames "
             f"(per digit th/ff/mf/rf/lf: {', '.join(fmt(x * 100, 1) + '%' for x in pr['per_digit_within_5mm_frac(th,ff,mf,rf,lf)'])}). "
             f"With capsules + opposite-side rule the judge finds g=1 on {fmt(pr['judge_g1_frac_all_pen_frames'] * 100, 1)}% of pen frames.\n")
    L.append('## Descriptor candidates (DEV only)\n')
    L.append('| variant | family | hammer median c_A on grasps (mm, d=50) | hammer median abs err | relocate median c_A (d=70) | relocate median abs err | rule value | AN hammer dev BA | AN hammer test BA (diagnostic, after freeze) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for v, t in sel['table'].items():
        L.append(f"| {v} | {t['family']} | {fmt(t['hammer_dev_median_cA_grasp_mm'], 1)} | {fmt(t['hammer_dev_median_abs_err_mm'], 1)} | "
                 f"{fmt(t['relocate_dev_median_cA_grasp_mm'], 1)} | {fmt(t['relocate_dev_median_abs_err_mm'], 1)} | {fmt(t['criterion_mm'], 2)} | {fmt(t['hammer_dev_AN_BA'])} | "
                 f"{fmt(t.get('hammer_test_AN_BA_diagnostic', float('nan')))} |")
    L.append('\nThe last column was computed after the choice was frozen and is shown only to make the dev-to-test drop visible; the frozen choice is not changed by it.')
    L.append('\n## Arms on hammer (primary set; hold band 47-58 mm)\n')
    L.append('| arm | kind | dev BA | test BA [95% CI] | test P(hold given grasp) | test P(open given free) | test crush given grasp | test median c_F on grasps (mm) |')
    L.append('|---|---|---|---|---|---|---|---|')
    for name, a in r['arms'].items():
        d, t = a['scores']['hammer/dev'], a['scores']['hammer/test']
        ci = t['ci95_episode_bootstrap']
        L.append(f"| {name} | {a['kind']} | {fmt(d['BA'])} | {fmt(t['BA'])} [{fmt(ci['lo'])}, {fmt(ci['hi'])}] | {fmt(t['P_hold_given_grasp'])} | "
                 f"{fmt(t['P_open_given_free'])} | {fmt(t['crush_rate_given_grasp'])} | {fmt(t['median_cF_mm_grasp'], 1)} |")
    L.append('\nBA follows the plan: 0.5 x (P(hold | grasp) + P(open | free)). Under it a constant-closed gripper scores 0 (it crushes and is never open); '
             'the alternative "not-hold" form (free frames only need not-hold) gives constant-open = constant-closed = 0.5; it is in the JSON as BA_nothold.\n')
    L.append('## Relocate (secondary; ball 70 mm sits at the top of the 78 mm stroke, so only grasp hold-rate and crush-rate mean something)\n')
    L.append('| arm | test P(hold given grasp) | test crush given grasp | test median c_F on grasps (mm) |')
    L.append('|---|---|---|---|')
    for name, a in r['arms'].items():
        t = a['scores']['relocate/test']
        L.append(f"| {name} | {fmt(t['P_hold_given_grasp'])} | {fmt(t['crush_rate_given_grasp'])} | {fmt(t['median_cF_mm_grasp'], 1)} |")
    L.append('\n## Pen eval (no opposition grasps; open side + rank agreement with the source descriptor)\n')
    L.append('| arm | free frames | P(open given free) | Spearman(c_F, c_A source) |')
    L.append('|---|---|---|---|')
    for name, a in r['arms'].items():
        t = a['scores']['pen/eval']
        L.append(f"| {name} | {t['n_free']} | {fmt(t['P_open_given_free'])} | {fmt(t.get('spearman_cF_vs_cA', float('nan')))} |")
    if r.get('ood'):
        L.append('\n## OOD check (same-body Adroit reconstruction, mean joint position error)\n')
        L.append('| model | pen eval (cm) | hammer test (cm) | relocate test (cm) | hammer/pen | gate (<= 2x) |')
        L.append('|---|---|---|---|---|---|')
        for name, o in r['ood'].items():
            L.append(f"| {name} | {fmt(o['pen/eval']['pos_err_cm_mean'], 2)} | {fmt(o['hammer/test']['pos_err_cm_mean'], 2)} | "
                     f"{fmt(o['relocate/test']['pos_err_cm_mean'], 2)} | {fmt(o['ratio_hammer_test_over_pen_eval_pos'], 2)} | {'pass' if o['ood_gate_hammer_le_2x'] else 'FAIL'} |")
    if r.get('FA_closure'):
        fr = r.get('franka_FA_rows', {})
        L.append(f"\n## Gripper -> hand closure test (Franka eval episodes {fr.get('episodes')}, {fr.get('rows')} rows, squeeze share {fmt(fr.get('squeeze_frac', float('nan')))})\n")
        L.append('A Franka source frame gives a cylinder of diameter c_F (the gripper pad gap). The decoded hand "holds" it if the closure rule below passes; '
                 'it "fits" it if it holds c_F but not c_F + 30 mm (one-sided hold cannot tell an open hand from a right one). '
                 'Squeeze frames: median c_F ' + fmt(fr.get('source_clearance_mm_median_squeeze', float('nan')), 1) + ' mm; other frames are mostly fully open (~77 mm).\n')
        L.append('| hand source | hold, squeeze | hold, other | fit, squeeze | fit, other | closure BA (fit) |')
        L.append('|---|---|---|---|---|---|')
        for name, f in r['FA_closure'].items():
            L.append(f"| {name} | {fmt(f['hold_rate_squeeze'])} | {fmt(f['hold_rate_nonsqueeze'])} | {fmt(f['fit_rate_squeeze'])} | "
                     f"{fmt(f['fit_rate_nonsqueeze'])} | {fmt(f['closure_BA_fit'])} |")
        cv = r['closure_test_validity']
        L.append(f"\nClosure-test sanity on real hammer TEST frames ({cv['n_grasp_frames']} grasp, {cv['n_free_frames']} free): grasping hands hold a cylinder of the real "
                 f"handle size in {fmt(cv['grasp_true_diameter'])} of frames, one 30 mm larger in {fmt(cv['grasp_plus_30mm'])}, one 30 mm smaller in {fmt(cv['grasp_minus_30mm'])} "
                 f"(the test cannot say a closed hand is too closed: a smaller object can always be pinched somewhere). Free hands hold the real size in "
                 f"{fmt(cv['free_true_diameter'])} and +30 mm in {fmt(cv['free_plus_30mm'])}. Pass rule: some placement of a cylinder of diameter c_F has the thumb tip "
                 f"capsule and an ff/mf/rf tip capsule within {cj.CLOSURE_TOL_MM:.0f} mm of its surface on opposite sides, with no hand capsule pushed in deeper "
                 f"than {cj.CLOSURE_TOL_MM:.0f} mm (same 5 mm tolerance as the grasp labels; real D4RL grasps sink 1-4 mm into the handle).\n")
    pc = r.get('surface_pad_check')
    if pc and pc.get('available'):
        L.append('\n## Pad cross-check: dense-asset pad tags (WP-ASSETS surface_v2) vs MJCF pads (descriptor uses the MJCF pads)\n')
        L.append('| digit | tagged points | patch centroid to MJCF pad (mm) | nearest tagged point to MJCF pad (mm) | normal cos |')
        L.append('|---|---|---|---|---|')
        for k, d in pc['digits'].items():
            L.append(f"| {k} | {d.get('count', '')} | {fmt(d['pad_surface_mm'], 2)} | {fmt(d.get('nearest_tag_point_mm', float('nan')), 2)} | {fmt(d['normal_cos'], 3)} |")
        L.append(f"\nCentroid rule (<= {pc['tol_mm']} mm): {'pass' if pc['passed'] else 'FAIL'}; cover rule (nearest tagged point <= {pc['tol_mm']} mm): "
                 f"{'pass' if pc.get('passed_nearest') else 'FAIL'}. The tagged patches cover the MJCF pads; their centroids sit 1-1.5 mm under the "
                 'surface (curved patch) and ~2.5 mm toward the knuckle on th/ff/mf, so two centroids miss 3 mm by a fraction of a millimetre. '
                 'The frozen descriptor keeps the MJCF pads, so this does not change any E2 number.')
    L.append('\n## Notes\n')
    L.append('- Time alignment: D4RL observations hold qpos after the step but body/site positions from one 2 ms physics sub-step earlier. '
             'Hammer/relocate frames are therefore taken at observation time (hand qpos moved back 0.2 frame); with that, relocate palm-minus-ball '
             'matches obs[30:33] within '
             f"{fmt(r['hand_data_manifest']['datasets']['relocate']['checks']['palm_minus_obj_vs_obs_max_mm'], 2)} mm (max), hammer palm within "
             f"{fmt(r['hand_data_manifest']['datasets']['hammer']['checks']['palm_site_vs_obs_max_mm'], 2)} mm. Pen hand rows stay as trained; the pen pose is moved forward 0.2 frame instead.")
    L.append('- Hammer orientation uses the gymnasium-robotics euler convention (= scipy intrinsic XYZ); checked by the hammer head landing on the nail at impact frames (median 3.5 mm; other conventions 218-450 mm).')
    L.append('- Videos for Stage-0 (hand+object | analytic gripper) belong to the render package; per-frame inputs are in `arms/<arm>/<set>.npz`.')
    L.append(f"\nWall time {fmt(r['wall_seconds'], 1)} s.\n")
    return '\n'.join(L) + '\n'


def eval_arm(name, checkpoint, model_class=None, config=None, out_dir=None, threads=8, boot=1000):
    """Score one trained arm (e.g. G0/G1) with the frozen descriptor; writes OUT/eval/e2/arms/<name>/summary.json."""
    torch.set_num_threads(threads)
    from . import bodies as eb
    out_dir = Path(out_dir) if out_dir is not None else E2_EVAL
    sets = load_sets()
    scene = cj.JudgeScene()
    labels = {ds: labels_for(ds, sets[ds], out_dir, scene) for ds in DATASETS}
    frozen = op.load_frozen()
    if not frozen.get('frozen'):
        raise RuntimeError('descriptor not frozen; run stage0 first')
    cfg = eb.load_config(config) if config else None
    model, cfg, step = eb.load_model(checkpoint, cfg, model_class)
    bodies = eb.load_external_bodies(cfg)
    cA = {ds: descriptor_mm(sets[ds], frozen['variant'], frozen['tau_cm'])[0] for ds in DATASETS}
    af = {ds: model_AF(model, bodies, sets[ds]['pose9']) for ds in DATASETS}
    res = {'arm': name, 'checkpoint': str(checkpoint), 'step': step, 'descriptor': frozen, 'scores': {}, 'scores_clamped': {}}
    for ds in DATASETS:
        for s in SPLITS[ds]:
            res['scores'][f'{ds}/{s}'] = score(af[ds]['c_F_mm_raw'], sets[ds], labels[ds]['labels'], s, cA[ds], boot)
            res['scores_clamped'][f'{ds}/{s}'] = score(af[ds]['c_F_mm_clamped'], sets[ds], labels[ds]['labels'], s, cA[ds], boot)
            res['scores'][f'{ds}/{s}_shuffled'] = score(shuffle_within_episode(af[ds]['c_F_mm_raw'], sets[ds]['episode']), sets[ds], labels[ds]['labels'], s, cA[ds], boot)
    rows_F, ep_F, squeeze = franka_eval_rows(bodies['relay_franka_full_v2'])
    Pd, Gd = model_FA(model, bodies, rows_F)
    cF_src = bodies['relay_franka_full_v2'].arrays['slide_native'][rows_F][:, [op.FRANKA_LEFT, op.FRANKA_RIGHT]].sum(1) - op.FRANKA_CLEARANCE_OFFSET_CM
    res['FA_closure'] = closure_summary(cj.hand_fit_test(Pd, Gd, cF_src, scene), squeeze)
    o = {}
    for ds, s in [('pen', 'eval'), ('hammer', 'test'), ('relocate', 'test')]:
        m = sets[ds]['split'] == s
        pe, re = model_recon(model, bodies, sets[ds]['pose9'][m])
        o[f'{ds}/{s}'] = {'pos_err_cm_mean': float(pe.mean()), 'rot_err_deg_mean': float(re.mean())}
    o['ratio_hammer_test_over_pen_eval_pos'] = o['hammer/test']['pos_err_cm_mean'] / o['pen/eval']['pos_err_cm_mean']
    res['ood'] = o
    d = out_dir / 'arms' / name.replace('@', '_at_')
    d.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        np.savez_compressed(d / f'{ds}.npz', c_F_mm=af[ds]['c_F_mm_raw'].astype(np.float32), labels=labels[ds]['labels'], episode=sets[ds]['episode'], split=sets[ds]['split'])
    jdump(d / 'summary.json', res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description='E2 evaluation (CPU)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('stage0')
    a.add_argument('--out', type=Path, default=E2_EVAL)
    a.add_argument('--threads', type=int, default=8)
    a.add_argument('--boot', type=int, default=1000)
    a.add_argument('--refreeze', action='store_true')
    c = sub.add_parser('rerender', help='re-render stage0_report.md from stage0_summary.json (adds the pad cross-check if missing)')
    c.add_argument('--out', type=Path, default=E2_EVAL)
    b = sub.add_parser('arm')
    b.add_argument('--name', required=True)
    b.add_argument('--checkpoint', type=Path, required=True)
    b.add_argument('--model-class', default=None)
    b.add_argument('--config', type=Path, default=None)
    b.add_argument('--out', type=Path, default=E2_EVAL)
    b.add_argument('--threads', type=int, default=8)
    args = ap.parse_args(argv)
    if args.cmd == 'stage0':
        r = stage0(args.out, args.threads, args.boot, args.refreeze)
        print(json.dumps(r['G0_gate'], indent=1))
        print(str(Path(args.out) / 'stage0_report.md'))
    elif args.cmd == 'rerender':
        sp = Path(args.out) / 'stage0_summary.json'
        r = json.loads(sp.read_text())
        if 'surface_pad_check' not in r:
            r['surface_pad_check'] = op.surface_vs_mjcf_check()
            jdump(sp, r)
        (Path(args.out) / 'stage0_report.md').write_text(render_report(r))
        print(str(Path(args.out) / 'stage0_report.md'))
    else:
        r = eval_arm(args.name, args.checkpoint, args.model_class, args.config, args.out, args.threads)
        print(json.dumps({k: v['BA'] for k, v in r['scores'].items() if 'hammer' in k}, indent=1))


if __name__ == '__main__':
    main()
