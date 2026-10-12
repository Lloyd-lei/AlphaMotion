"""WP-OPS trainer tests.

Fast unit tests (registry, comparison rules, mixture, xbody binding) plus CPU dry runs of the real
loop as subprocesses (each ~45 s, ~5 GB RSS, CUDA hidden):
  * Arep 5 steps: the first 5 seed-0 steps are UniMate steps and reproduce runs/A_S_seed0 hashes;
  * Arep with a deliberately perturbed data RNG at step 3 aborts before the optimizer update;
  * SF 7 steps with and without the arm-I input wrappers: identical hashes incl. the first XWBC
    step (wrappers consume no numpy RNG on the make_inputs AND make_reference_inputs paths);
  * G0 vs G1 (real e2 xbody_loss, init checkpoint disabled) 10 steps: identical hash sequences.
Set AM2_SKIP_DRYRUN=1 to skip the subprocess runs.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from am2exp_v1.common.paths import CODE, OUT, PY  # noqa: E402
from am2exp_v1.train import train_arm as T  # noqa: E402

REF = ROOT / 'runs' / 'A_S_seed0' / 'train.jsonl'
REQUIRED_ARMS = ['Arep', 'SF', 'SF_s1', 'S', 'O', 'I', 'S_wrong', 'O_roll', 'I_roll', 'V1_old', 'V1_new', 'SF_M',
                 'G0', 'G1', 'G1_roll']
XWBC = ['human_smpl', 'booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi', 'pnd_adam_lite',
        'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2', 'fourier_gr3', 'kuavo_s45',
        'unitree_h2', 'unitree_r1']


def ref_lines(n):
    out = []
    with REF.open() as f:
        for line in f:
            r = json.loads(line)
            out.append((r['step'], r['domain'], r['sampling_sha256']))
            if len(out) == n:
                break
    return out


# ------------------------------------------------------------------ registry
def test_registry_resolves_all_arms():
    reg = T.load_registry()
    for name in REQUIRED_ARMS:
        assert name in reg['arms'], name
    for name in reg['arms']:
        a = T.resolve_arm(name, reg)
        assert '${' not in json.dumps(a), name
        assert a['output_dir'] == str(OUT / 'runs' / name)
        assert a['extra']['lifetime_seconds'] > 0 and a['extra']['stage']
        T.load_symbol(a['model_class'])  # every model class importable (WP-MODEL present)
        obj = a['extra']['objective']
        if obj.get('slide_fix'):
            T.load_symbol(obj['slide_fix_loss'])
        if obj.get('relational'):
            T.load_symbol(obj['relational']['fn'])
        if a['extra'].get('e2'):
            T.load_symbol(a['extra']['e2']['fn'])
        if a['extra'].get('geometry'):
            T.load_symbol(a['extra']['geometry']['hook'])
        if a['extra'].get('auto_gate'):
            T.load_symbol(a['extra']['auto_gate']['rule'])


def test_registry_recipe_facts():
    r = lambda n: T.resolve_arm(n)  # noqa: E731
    sf = r('SF')
    assert (sf['steps'], sf['extra']['t_max'], sf['extra']['lr'], sf['seed']) == (3000, 3000, 1e-4, 0)
    assert sf['reference_train_jsonl'] == str(REF) and sf['extra']['sampling_assert'] == 'required'
    assert sorted(sf['extra']['checkpoints']) == [1000, 3000]
    assert r('Arep')['steps'] == 200 and r('Arep')['model_class'] == 'pilot_model:ConditionalMegatron'
    m = r('SF_M')
    assert (m['steps'], m['extra']['t_max'], m['size']) == (1000, 3000, 'M')
    assert m['reference_train_jsonl'] == str(ROOT / 'runs' / 'A_M_seed0' / 'train.jsonl')
    assert m['extra']['base_config'] == 'orchestration/configs/A_M_seed0.json'
    assert r('SF_s1')['seed'] == 1 and r('SF_s1')['extra']['sampling_assert'] is False
    g = {n: r(n) for n in ('G0', 'G1', 'G1_roll')}
    for n, a in g.items():
        assert a['steps'] == 1500 and a['extra']['t_max'] == 1500
        assert a['extra']['init_checkpoint'] == str(OUT / 'runs' / 'SF' / 'checkpoint_step3000.pt')
        assert a['extra']['mixture'] == {'xwbc': 0.40, 'unimate': 0.25, 'franka_extra': 0.10, 'hand_gripper': 0.25}
        assert a['extra']['streams'] == g['G0']['extra']['streams']
    assert [g[n]['extra']['e2']['w_xb'] for n in g] == [0.0, 0.2, 0.2]
    assert [g[n]['extra']['e2']['roll'] for n in g] == [0, 0, 32]
    assert g['G1']['reference_train_jsonl'] == str(OUT / 'runs' / 'G0' / 'train_slim.jsonl')
    der = r('S_wrong')['extra']['geometry']['params']['derangement']
    assert sorted(der) == sorted(der.values()) == sorted(XWBC + ['adroit_pen', 'relay_franka_full_v2'])
    assert all(k != v for k, v in der.items())
    assert r('I_roll')['extra']['inputs']['roll_relations'] and r('O_roll')['extra']['objective']['relational']['roll']
    for twin, (t, p) in {'S_wrong': ('S', 'SF'), 'O_roll': ('O', 'S'), 'I_roll': ('I', 'S'), 'G1_roll': ('G1', 'G0')}.items():
        assert r(twin)['extra']['auto_gate']['params'] == {'test': t, 'parent': p}
    v_old, v_new = r('V1_old'), r('V1_new')
    assert v_old['model_class'] == 'pilot_model:ConditionalMegatron' and v_new['model_class'].endswith(':SFMegatron')
    assert v_old['steps'] == v_new['steps'] == 1500 and v_old['extra']['mixture'] == {'franka_only': True}


def test_build_config_matches_original_recipe():
    cfg = T.build_config(T.resolve_arm('SF'))
    base = json.loads((ROOT / 'orchestration/configs/A_S_seed0.json').read_text())
    for k in ('model', 'batch_size', 'mixture', 'weights', 'data_manifest_sha256', 'judge_body_order', 'group'):
        if k in base:
            assert cfg[k] == base[k], k
    assert cfg['optimizer']['lr'] == base['optimizer']['lr'] == 1e-4
    assert cfg['checkpoint_steps'] == [1000, 3000]


def test_missing_model_module_gives_clear_import_error():
    with pytest.raises(ImportError, match='WP-MODEL'):
        T.load_symbol('am2exp_v1.model.does_not_exist:Foo')


# ------------------------------------------------------------------ comparison rules
def test_compare_sampling_rules(tmp_path):
    ref = {'sampling_sha256': 'aa', 'domain': 'xwbc', 'device': 'cuda'}
    assert T.compare_sampling(1, 'xwbc', 'aa', ref, 'cuda') == (True, 'match')
    assert T.compare_sampling(1, 'xwbc', 'bb', ref, 'cuda')[0] is False
    assert T.compare_sampling(1, 'xwbc', 'bb', ref, 'cpu')[0] is None       # device-dependent torch.rand
    assert T.compare_sampling(1, 'unimate', 'bb', ref, 'cpu')[0] is False    # domain must always agree
    uref = {'sampling_sha256': 'cc', 'domain': 'unimate', 'device': 'cuda'}
    assert T.compare_sampling(1, 'unimate', 'cc', uref, 'cpu') == (True, 'match')
    assert T.compare_sampling(1, 'unimate', 'cc', None, 'cpu')[0] is None
    p = tmp_path / 'r.jsonl'
    p.write_text(json.dumps({'step': 1, 'domain': 'unimate', 'sampling_sha256': 'x', 'match': False}) + '\n')
    with pytest.raises(RuntimeError, match='abort'):
        T.load_reference(p)
    ref = T.load_reference(REF)
    assert len(ref) == 3000 and ref[1]['device'] == 'cuda' and ref[1]['domain'] == 'unimate'


def test_draw_domain_consumes_one_uniform_like_original():
    cfg = {'mixture': {'xwbc': 0.55}}
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(50):
        assert T.draw_domain(a, cfg, None) == ('xwbc' if b.random() < 0.55 else 'unimate')
    mix = {'xwbc': .4, 'unimate': .25, 'franka_extra': .1, 'hand_gripper': .25}
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    seen = {T.draw_domain(a, cfg, mix) for _ in range(400)}
    assert seen == set(mix)
    [b.random() for _ in range(400)]
    assert a.bit_generator.state == b.bit_generator.state
    assert T.draw_domain(a, cfg, {'franka_only': True}) == 'franka_only'
    with pytest.raises(ValueError):
        T.draw_domain(a, cfg, {'xwbc': .5, 'unimate': .4})


def test_call_xbody_binds_both_signatures():
    rng = np.random.default_rng(0)
    seen = []
    def contract(model, bodies, batch_rng_state, cfg):
        seen.append((batch_rng_state is rng, cfg == {'w': 1}))
    def legacy(model, bodies, cfg, rng):
        seen.append((rng is globals()['np'].random.default_rng or True, isinstance(cfg, dict)))
    T.call_xbody(contract, None, {}, rng, {'w': 1})
    T.call_xbody(legacy, None, {}, rng, {'w': 1})
    assert seen == [(True, True), (True, True)]


# ------------------------------------------------------------------ CPU dry runs (subprocess)
DRY = pytest.mark.skipif(os.environ.get('AM2_SKIP_DRYRUN') == '1', reason='AM2_SKIP_DRYRUN=1')


def dry_run(tmp_path, name, arm, steps, *extra, override=None):
    out = OUT / 'runs' / '_dryrun' / f'pytest_{name}_{os.getpid()}'
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    cmd = [PY, str(CODE / 'train' / 'train_arm.py'), '--arm', arm, '--dry-run-cpu', '--max-steps', str(steps),
           '--out-dir', str(out), '--lifetime-seconds', '420', *extra]
    if override:
        cmd += ['--override', json.dumps(override)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    slim = [json.loads(x) for x in (out / 'train_slim.jsonl').read_text().splitlines()] if (out / 'train_slim.jsonl').exists() else []
    res = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else None
    ex = json.loads((out / 'exit_status.json').read_text()) if (out / 'exit_status.json').exists() else None
    return {'proc': p, 'slim': slim, 'result': res, 'exit': ex, 'dir': out}


@DRY
def test_dryrun_first5_reproduce_A_S_seed0(tmp_path):
    r = dry_run(tmp_path, 'Arep5', 'Arep', 5)
    assert r['proc'].returncode == 0, r['proc'].stderr[-3000:]
    ref = ref_lines(5)
    assert [x[1] for x in ref] == ['unimate'] * 5
    assert [(s['step'], s['domain'], s['sampling_sha256']) for s in r['slim']] == ref
    assert all(s['match'] is True for s in r['slim'])
    samp = r['result']['sampling']
    assert samp['assert_enabled'] and samp['steps_checked'] == 5 and samp['mismatches'] == 0
    # CPU fp32 vs the CUDA bf16 reference: same data, same init -> losses agree within bf16-level noise
    assert r['result']['loss_vs_reference']['max_rel_diff'] < 0.01
    assert r['exit']['status'] == 'completed'
    for f in ('process_identity.json', 'state.json', 'free_start.json', 'capacity_start.json', 'setup.json',
              'source_snapshot.json', 'train.jsonl'):
        assert (r['dir'] / f).exists(), f
    print('Arep CPU dry run:', json.dumps(r['result']['loss_vs_reference']))


@DRY
def test_dryrun_perturbed_rng_aborts(tmp_path):
    r = dry_run(tmp_path, 'Arep_perturb', 'Arep', 5, '--perturb-rng-at-step', '3')
    assert r['proc'].returncode != 0
    assert 'SamplingMismatch' in r['proc'].stderr and 'step 3' in r['proc'].stderr
    assert [s['match'] for s in r['slim']] == [True, True, False]
    assert r['exit']['status'] == 'failed' and r['exit']['step'] == 3
    assert r['result'] is None
    assert json.loads((r['dir'] / 'failure.json').read_text())['type'] == 'SamplingMismatch'


@DRY
def test_dryrun_input_wrappers_consume_no_rng(tmp_path):
    plain = dry_run(tmp_path, 'SF7_plain', 'SF', 7)
    wrapped = dry_run(tmp_path, 'SF7_wrapped', 'SF', 7,
                      override={'extra': {'inputs': {'relational_source': True}}})
    for r in (plain, wrapped):
        assert r['proc'].returncode == 0, r['proc'].stderr[-3000:]
    hp = [(s['domain'], s['sampling_sha256']) for s in plain['slim']]
    hw = [(s['domain'], s['sampling_sha256']) for s in wrapped['slim']]
    assert hp == hw and 'xwbc' in [d for d, _ in hp]
    calls = wrapped['result']['input_wrappers']
    assert calls['make_reference_inputs'] >= 1 and plain['result']['input_wrappers'] is None
    assert [s['match'] for s in plain['slim'][:5]] == [True] * 5


@DRY
def test_dryrun_G0_G1_identical_sampling(tmp_path):
    ov = {'extra': {'init_checkpoint': None, 'init_step': None}}
    g0 = dry_run(tmp_path, 'G0', 'G0', 10, override=ov)
    assert g0['proc'].returncode == 0, g0['proc'].stderr[-3000:]
    ov1 = {'reference_train_jsonl': str(g0['dir'] / 'train_slim.jsonl'), **ov}
    g1 = dry_run(tmp_path, 'G1', 'G1', 10, override=ov1)
    assert g1['proc'].returncode == 0, g1['proc'].stderr[-3000:]
    h0 = [s['sampling_sha256'] for s in g0['slim']]
    h1 = [s['sampling_sha256'] for s in g1['slim']]
    assert h0 == h1 and len(h0) == 10
    doms = [s['domain'] for s in g0['slim']]
    assert 'hand_gripper' in doms
    assert g1['result']['sampling']['steps_checked'] == 10 and g1['result']['sampling']['mismatches'] == 0
    hg0 = [s for s in g0['slim'] if s['domain'] == 'hand_gripper']
    hg1 = [s for s in g1['slim'] if s['domain'] == 'hand_gripper']
    assert all(s['e2']['w_xb'] == 0.0 for s in hg0) and all(s['e2']['w_xb'] == 0.2 for s in hg1)
    # G0 still backpropagates its zero-weight loss (finite_backward saw gradients, all zero)
    assert all(s['total'] == 0.0 and s['gradient_norm_before_clip'] == 0.0 for s in hg0)
    assert all(s['gradient_norm_before_clip'] > 0 for s in hg1)
    print('G0/G1 chain', g0['result']['sampling']['chain_sha256'], g1['result']['sampling']['chain_sha256'], doms)
