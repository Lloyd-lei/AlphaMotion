"""WP-JUDGE: E1 case list, runner, scorer and statistics (CPU)."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True

from am2exp_v1.eval import cases as C  # noqa: E402
from am2exp_v1.eval import stats as ST  # noqa: E402


# ------------------------------------------------------------------ cases
def test_mining_rule_synthetic():
    nf = np.zeros(1000, bool); nf[100:400] = True; nf[600:700] = True
    tr = np.ones(1000, bool)
    picks = C.mine_clip(nf, tr, blocked=[(0, 64)])
    starts = [s for s, _ in picks]
    assert len(picks) <= 4 and all(f >= 0.5 for _, f in picks)
    for i, a in enumerate(starts):
        for b in starts[i + 1:]:
            assert abs(a - b) >= C.WINDOW
    assert all(s % 8 == 0 for s in starts) and all(s >= 64 for s in starts)
    tr2 = tr.copy(); tr2[:] = False
    assert C.mine_clip(nf, tr2, []) == []


@pytest.fixture(scope='module')
def frozen():
    if not C.CASES_FILE.exists():
        pytest.skip('case list not frozen yet')
    return C.load_cases(C.CASES_FILE, verify=True)


def test_frozen_case_list(frozen):
    man = C.load_manifest(); by = C.group_by_clip(man)
    ws = frozen['windows']
    kinds = {w['kind'] for w in ws}
    assert kinds == {'unseen_motion_listed', 'unseen_motion_mined', 'guard', 'unseen_body'}
    listed = {(w['clip'], w['start']) for w in ws if w['kind'] == 'unseen_motion_listed'}
    assert listed == {(c, s) for c, s in C.LISTED}
    per_clip = {}
    for w in ws:
        g = by[w['clip']]
        assert w['frames'] == 128 and 0 <= w['start'] <= g['length'] - 128
        if w['kind'] != 'unseen_body':
            assert g['split'] == 1
        else:
            assert g['split'] == 0
        per_clip.setdefault(w['clip'], []).append(w)
        if w['kind'] == 'unseen_motion_mined':
            assert C.excluded(g) is None and w['human_nonfoot_contact_frac'] >= 0.5
    for clip, items in per_clip.items():
        assert sum(w['kind'] == 'unseen_motion_mined' for w in items) <= 4
        s = sorted((w['start'], w['start'] + 128) for w in items)
        assert all(a1 <= b0 for (a0, a1), (b0, b1) in zip(s, s[1:])), clip
    for c in frozen['cases']:
        assert c['source'] != c['target']
        if c['kind'] == 'unseen_body':
            assert c['target_heldout']
    # rebuilding gives the same sha (frozen rule is deterministic)
    w2, _ = C.build_windows(man)
    obj_cases = C.build_cases(w2)
    assert obj_cases == frozen['cases']


def test_shuffle_map(frozen):
    from am2exp_v1.eval.run_e1_eval import shuffle_map
    wins = {w['window_id']: w for w in frozen['windows']}
    m = shuffle_map(frozen['windows'])
    assert set(m) == set(wins)
    for a, b in m.items():
        assert wins[a]['clip'] != wins[b]['clip']
        assert (wins[a]['kind'] == 'unseen_body') == (wins[b]['kind'] == 'unseen_body')


# ------------------------------------------------------------------ stats
def test_permutation_and_sign_tests():
    x = np.array([-1.0, -2.0, -0.5, -0.1])
    assert ST.permutation_test(x, 'less') == pytest.approx(1 / 16)
    assert ST.sign_test(x, 'less') == pytest.approx(1 / 16)
    assert ST.permutation_test(-x, 'less') == 1.0
    y = np.array([-1.0, 1.0])
    assert ST.permutation_test(y, 'two-sided') == 1.0
    big = -np.ones(30)
    assert ST.permutation_test(big, 'less') < 1e-3


def _records(delta_cse=0.0, guard=0.0, windows=10, bodies=('fourier_gr3', 'kuavo_s45', 'unitree_h2'), seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for w in range(windows):
        for t in bodies:
            out.append(dict(case_id=f'h__{t}__w{w}', window_id=f'w{w}', source='human_smpl', target=t, heldout=True, primary=True,
                            kind='unseen_motion_mined', CSE_GT=0.5 + delta_cse + noise * rng.normal(), PenP95_cm=2.0 + guard,
                            FloatRate=0.1, SelfPen=0.0, pos_err_mm=100.0))
    return out


def test_decide_rule():
    parent = _records()
    better = _records(-0.06)
    d = ST.decide(better, parent)
    assert d['effect']['effect'] and d['verdict'] == 'pending' and not d['reasons']
    d2 = ST.decide(better, parent, sf_records=parent, sf_s1_records=_records(0.05), twin_records=parent, lloyd_video_ok=True)
    assert d2['seed_floor'] == pytest.approx(0.05) and d2['effect']['threshold'] == pytest.approx(-0.10)
    assert d2['verdict'] == 'fail'  # -0.06 does not beat 2 x floor
    d3 = ST.decide(better, parent, sf_records=parent, sf_s1_records=_records(0.01), twin_records=parent, lloyd_video_ok=True)
    assert d3['verdict'] == 'pass'
    d4 = ST.decide(_records(-0.06, guard=1.5), parent, sf_records=parent, sf_s1_records=_records(0.01), twin_records=parent, lloyd_video_ok=True)
    assert d4['verdict'] == 'fail' and d4['guards']['PenP95_cm']['regressed']
    d5 = ST.decide(better, parent, sf_records=parent, sf_s1_records=_records(0.01), twin_records=_records(-0.06), lloyd_video_ok=True)
    assert d5['verdict'] == 'fail' and d5['twin']['twin_effect']
    assert ST.replication_ok(-0.06, -0.035) and not ST.replication_ok(-0.06, -0.02) and not ST.replication_ok(-0.06, 0.04)


def test_summarize_mean_of_window_means():
    r = [dict(window_id='a', heldout=True, m=1.0), dict(window_id='a', heldout=True, m=3.0), dict(window_id='b', heldout=True, m=0.0)]
    s = ST.summarize(r, ['m'])
    assert s[(True,)]['m'] == pytest.approx(1.0)


# ------------------------------------------------------------------ runner + scorer (real A0 checkpoint, 1 window x 2 cases)
@pytest.fixture(scope='module')
def a0_cases(tmp_path_factory, frozen):
    from am2exp_v1.eval import run_e1_eval as RE
    out = tmp_path_factory.mktemp('e1')
    ck = ROOT / 'runs/A_S_seed0/checkpoint_step3000.pt'
    for mode in ('sensor_on', 'sensor_off'):
        RE.main(['--arm', 'A0', '--checkpoint', str(ck), '--mode', mode, '--out', str(out / mode), '--targets', 'unitree_h2,unitree_g1_29dof',
                 '--sources', 'human_smpl', '--limit', '2', '--threads', '4'])
    return out


def test_runner_outputs(a0_cases, frozen):
    on = sorted((a0_cases / 'sensor_on').glob('*.npz')); off = sorted((a0_cases / 'sensor_off').glob('*.npz'))
    assert len(on) == 2 and [p.name for p in on] == [p.name for p in off]
    for p_on, p_off in zip(on, off):
        a, b = np.load(p_on), np.load(p_off)
        meta = json.loads(str(a['meta']))
        T = 128; J = len(meta['target_joint_names'])
        for k, shape in dict(pred_rot6=(T, J, 6), pred_joint_position=(T, J, 3), pred_root_height_m=(T,), pred_contact_logits=(T, J),
                             gt_rot6=(T, J, 6), gt_joint_position=(T, J, 3), gt_root_height_m=(T,)).items():
            assert a[k].shape == shape, k
        assert a['source_pose9'].shape[-1] == 9 and meta['cases_sha256'] == frozen['sha256']
        assert np.isnan(b['source_contact_input']).all() and np.isfinite(a['source_contact_input']).any()
        # sensor_on input is the training-payload signed-log feature log1p(|thr-gap|/thr)/log1p(D/thr): |v| ~ 1 at |thr-gap| = D
        # (slightly below -1 for joints higher than D above the floor), positive below threshold
        v = a['source_contact_input'][np.isfinite(a['source_contact_input'])]
        assert v.max() <= 1.0 + 1e-3 and v.min() > -1.5 and (v < 0).mean() > 0.5
        # the sensor changes the prediction (A0 was trained with contact input + 0.3 dropout)
        assert not np.allclose(a['pred_joint_position'], b['pred_joint_position'])
        assert np.isfinite(a['pred_root_height_m']).all()


def test_sensor_matches_sealed_payload_rows():
    """contact_sdist recomputed from the window gap equals the sealed payload rows where they overlap
    (run_locomotion_eval.load_window raises otherwise); here we re-check one clip explicitly."""
    from am2exp_v1.eval import run_e1_eval as RE
    import torch
    RLE, _ = RE._sealed()
    cfg = json.loads((ROOT / 'runs/A_S_seed0/config.json').read_text())
    by, _, _ = RE.load_bodies(cfg, torch.device('cpu'))
    man = json.loads((ROOT / 'data/data_manifest_v2.json').read_text())
    g = next(x for x in man['xwbc']['groups'] if x['clip'] == 'push1_subject2')
    entry = dict(dataset='XWBC', group_id=g['group_id'], source_frames=g['length'], start=0, frames=g['length'])
    full = RLE.load_window(man, entry, by['human_smpl'], with_contact=True)
    rows = RLE.xwbc_rows(by['human_smpl'], g)
    fr = by['human_smpl'].arrays['frame_index'][rows]
    assert np.array_equal(full['contact_sdist'][fr], by['human_smpl'].arrays['contact_sdist'][rows], equal_nan=True)


def test_scorer_on_runner_output(a0_cases):
    from am2exp_v1.eval.score_e1 import score_file
    f = sorted((a0_cases / 'sensor_on').glob('*unitree_g1_29dof*.npz'))[0]
    r = score_file(f, heldout_bias={})
    for k in ('CSE_GT', 'FloatRate', 'PenP95_cm', 'SelfPen', 'pos_err_mm', 'CSE_human', 'raw_CSE_GT', 'CSE_snap'):
        assert k in r and np.isfinite(r[k]), k
    assert abs(r['root_shift_cm'] - 0.04) < 1e-6  # G1 delta_col
    h = sorted((a0_cases / 'sensor_on').glob('*unitree_h2*.npz'))
    if h:
        r2 = score_file(h[0], heldout_bias={})
        assert not np.isfinite(r2['root_shift_cm']) and 'CSE_GT' not in r2
        r3 = score_file(h[0], heldout_bias={'unitree_h2': 1.5})
        assert abs(r3['root_shift_cm'] - 1.5) < 1e-6 and np.isfinite(r3['CSE_GT'])


def test_registry_path_new_arm_checkpoint(tmp_path, frozen):
    """A train_arm-style checkpoint (config carries am2exp.registry_entry + model_kind) is rebuilt with the new class."""
    import torch
    from am2exp_v1.eval import run_e1_eval as RE
    from am2exp_v1.model.registry import build_model
    cfg = json.loads((ROOT / 'runs/A_S_seed0/config.json').read_text())
    cfg = dict(cfg, model_kind='am2exp_v1.model.slide_fix:SFMegatron', run_id='SF_test',
               am2exp={'arm': 'SF_test', 'registry_entry': {'name': 'SF_test', 'extra': {'model_kwargs': {}, 'geometry': None}}})
    model = build_model(cfg, torch.device('cpu'), 'SFMegatron')
    ck = tmp_path / 'checkpoint_step0001.pt'
    torch.save({'model': model.state_dict(), 'step': 1, 'config': cfg}, ck)
    m2, cfg2, info = RE.build_model(ck, torch.device('cpu'))
    assert type(m2).__name__ == 'SFMegatron' and info['variant'] == 'am2exp'
    sd1, sd2 = model.state_dict(), m2.state_dict()
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    RE.main(['--arm', 'SF_test', '--checkpoint', str(ck), '--mode', 'sensor_on', '--out', str(tmp_path / 'o'), '--limit', '1',
             '--sources', 'human_smpl', '--targets', 'unitree_g1_29dof'])
    z = np.load(next((tmp_path / 'o').glob('*.npz')))
    assert json.loads(str(z['meta']))['model']['model_class'] == 'am2exp_v1.model.slide_fix:SFMegatron'
