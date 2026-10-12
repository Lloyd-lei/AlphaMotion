"""Actual-run recipe audit: immutable config, source evidence, current call graph.

This is a read-only source inspection. Presence of a function or config weight is
not a numerical validation or online training authorization. Default output is
recipe_review.json only; documentation is maintained separately.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
from recipe_gate import reference_config, training_recipe_status, REFERENCE_CONFIG_SHA256

HERE = Path(__file__).resolve().parent
ORIGINAL_SNAPSHOT = Path('/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_root():
    choices = [Path(os.environ['PILOT_RECIPE_SOURCE'])] if os.environ.get('PILOT_RECIPE_SOURCE') else []
    choices += [HERE / 'reference_recipe' / 'source', HERE / 'legacy', ORIGINAL_SNAPSHOT]
    return next((p for p in choices if all((p / n).is_file() for n in ['train_dual.py', 'dual_common.py', 'masking.py'])), None)


def symbol(path, name):
    """Return exact current source evidence, not a claim that its behavior passes."""
    path = Path(path)
    if not path.is_file():
        return {'file': path.name, 'symbol': name, 'present': False, 'reason': 'file absent'}
    raw = path.read_bytes()
    try:
        tree = ast.parse(raw, filename=str(path))
    except SyntaxError as e:
        return {'file': path.name, 'symbol': name, 'present': False, 'reason': str(e)}
    target = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if target is None:
        return {'file': path.name, 'symbol': name, 'present': False, 'file_sha256': hashlib.sha256(raw).hexdigest()}
    code = ast.get_source_segment(raw.decode(), target)
    calls = sorted({ast.unparse(n.func) for n in ast.walk(target) if isinstance(n, ast.Call)})
    return {'file': path.name, 'symbol': name, 'present': True, 'file_sha256': hashlib.sha256(raw).hexdigest(),
            'start_line': target.lineno, 'end_line': target.end_lineno,
            'symbol_sha256': hashlib.sha256(code.encode()).hexdigest(), 'calls': calls}


def build(config):
    old = reference_config()
    source_root = _source_root()
    sources = {}
    if source_root:
        for name in ['train_dual.py', 'dual_common.py', 'masking.py', 'model_dual.py', 'common.py']:
            p = source_root / name
            if p.is_file():
                sources[name] = {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}
    inventory = {}
    targets = {
        'train_pilot.py': ['objective', 'preflight', 'main'],
        'recipe_training.py': ['make_reference_inputs', 'xwbc_combos', 'backward_training_step'],
        'recipe_auxiliary.py': ['contrast_loss', 'judge_loss', 'cycle_consistency', 'auxiliary_losses', '_paired_bodies'],
        'recipe_ground.py': ['ground_losses'],
        'pilot_model.py': ['load_reference_weights', 'set_judge_body_order', 'judge_logits'],
        'recipe_gate.py': ['reference_config', 'training_recipe_status', 'assert_training_recipe_ready'],
        'xwbc_contract.py': ['validate_xwbc'],
    }
    for filename, names in targets.items():
        for name in names:
            inventory[filename + ':' + name] = symbol(HERE / filename, name)

    def hook(filename, name, calls=()):
        key = filename + ':' + name
        ev = inventory[key]
        missing = sorted(set(calls) - set(ev.get('calls', [])))
        return {'symbol': key, 'present': ev['present'], 'expected_calls': list(calls),
                'missing_expected_calls': missing, 'connected': ev['present'] and not missing}

    def row(name, keys, semantics, current, hooks=(), differences=(), evidence=()):
        return {'term': name, 'actual_run_flags': {k: old[k] for k in keys},
                'actual_source_semantics': semantics, 'current_configuration': current,
                'current_source_connections': list(hooks),
                'all_expected_connections_present': all(h['connected'] for h in hooks) if hooks else None,
                'explicit_differences': list(differences), 'legacy_source_evidence': list(evidence),
                'assessment_scope': 'source-level inspection; exact current file/function hashes above; not execution proof'}

    recipe = config.get('recipe', {})
    loss = config.get('loss', {})
    rows = [
        row('rotation and FK position', ['w_pos', 'fk_loss', 'rot_head', 'rot_only'],
            'Global geodesic degrees mean over B*J, divided by30; Euclidean FK(predrot)-FK(GTrot) / target reach mean over B*J, multiplied by5. Neither position MSE nor raw-hpos Cartesian loss.',
            {'position_weight': loss.get('position')},
            [hook('train_pilot.py', 'objective', ['fk', 'angle_degrees'])],
            evidence=['dual_common.py:349-455', 'train_dual.py:426']),
        row('root height', ['root_pos', 'w_rootpos', 'root_head', 'w_root'],
            'SmoothL1 on raw hpos[:,root_index,0] against metric GT gap_root; weighted3. Raw hpos channels are not XYZ. Legacy root-pos branch did not filter trust itself.',
            {'weight': loss.get('root_height')},
            [hook('train_pilot.py', 'objective'), hook('recipe_ground.py', 'ground_losses')],
            ['Explicit valid/finite selection is retained; no supervision on unknown UniMate ground.'],
            ['train_dual.py:438-452']),
        row('InfoNCE', ['w_contrast', 'tau', 'contrast_sample'],
            'Six distinct XWBC bodies; full unconditional pose encodes on the same source rows. Flatten and concatenate both FSQ latents, L2 normalize, symmetric row-diagonal CE at tau .1 averaged over all15 unordered body pairs; weight .5. Original row sampling used replacement.',
            {k: recipe.get(k) for k in ['w_contrast', 'tau', 'contrast_sample']},
            [hook('recipe_auxiliary.py', 'contrast_loss', ['_stable_unique_indices', 'F.cross_entropy']),
             hook('recipe_auxiliary.py', 'auxiliary_losses', ['contrast_loss']),
             hook('recipe_training.py', 'backward_training_step', ['auxiliary_losses'])],
            ['Stable deduplication of verified original source-row identity only for InfoNCE fixes repeated-row false negatives; judge and cycle retain original batch weighting.', 'UniMate contributes no cross-body positives.'],
            ['train_dual.py:371', 'train_dual.py:609-633']),
        row('unconditional reencoding', ['reenc_consistency', 'reenc_prob'],
            'Probability .3; independent direction from full body-product including self; max(8,B//2) first rows. Full9 pose, no contact/mask/target conditions. Decode raw hpos then encode concatenated predicted rot+raw hpos; sum mean squared errors of both latents, no detach, weight1, not divided by8 or event probability.',
            {k: recipe.get(k) for k in ['reenc_consistency', 'reenc_prob']},
            [hook('recipe_auxiliary.py', 'cycle_consistency', ['model.encode', 'model.decode']),
             hook('recipe_auxiliary.py', 'auxiliary_losses', ['cycle_consistency'])],
            ['Half-batch count is capped only for explicit tiny synthetic test batches.'], ['train_dual.py:502-513']),
        row('auxiliary16-class judge', ['judge_mode', 'n_judges', 'w_judge'],
            'Same six unconditional body encodes used for contrast. Independent sigmoid BCEWithLogits against16-dimensional one-hot targets, mean over B*16 then mean over body encodes; grad mode does not detach; weight .1. Original class identity depended on cache-manifest body order and launch exclusion environment.',
            {'model_n_judges': config.get('model', {}).get('n_judges'), **{k: recipe.get(k) for k in ['judge_mode', 'w_judge']}},
            [hook('recipe_auxiliary.py', 'judge_loss', ['F.binary_cross_entropy_with_logits', 'model.judge_logits']),
             hook('pilot_model.py', 'load_reference_weights')],
            ['Authorized entire-judge reinitialization if original label mapping is unproven; use and record verified current chosen_split.body_names. Do not infer mapping from output shape or permute old columns silently.', 'No UniMate judge classes.'],
            ['train_dual.py:207-209', 'train_dual.py:634-644']),
        row('ground penetration and cgeo', ['w_pen', 'w_cgeo', 'w_contact', 'w_cbce'],
            'Trusted target rows only. h=fk_cm.Y/100 + GT root gap + (R_global @ sole_offset_m).Y at mapped soles. pen=mean(relu(-h)^2) over all trusted rows*joints, weight200. cgeo=BCEWithLogits((thr-h)/(thr/3), soft) over trusted rows*joints, weight1. cgeo uses soft without hard weighting. cmaint/contact-count normalization and hcon BCE were inactive.',
            {'penetration': loss.get('penetration'), 'contact_geometry': loss.get('contact_geometry')},
            [hook('recipe_ground.py', 'ground_losses', ['F.binary_cross_entropy_with_logits']),
             hook('train_pilot.py', 'objective', ['ground_losses'])],
            ['Explicit measurement validity allows unknown values to remain NaN; select valid fields before arithmetic.', 'UniMate nonmetric lengths and unknown ground cannot supply these losses.'],
            ['dual_common.py:46-60', 'dual_common.py:737-794']),
        row('source masks and contact observation', ['mask_prob', 'mask_mix', 'con_dropout', 'contact_chan'],
            'Six family weights rand.25/ee.15/root.10/sparse.25/cask.15/cgive.10; family mask enabled per-row with probability .3; root always visible. Sparse samples1..3 from whole joint set plus root, not leaf-only. Contact input is actual signed distance (unless historical AM_OLDCHAN override); sampler con_over is discarded. Any row with a hidden joint bypasses .3 contact dropout; hidden pose joints retain declared contact observations.',
            {k: recipe.get(k) for k in ['mask_prob', 'mask_mix']} | {'contact_dropout': config.get('contact_dropout')},
            [hook('recipe_training.py', 'make_reference_inputs', ['sample_joint_mask', 'parse_mix']),
             hook('recipe_training.py', 'backward_training_step', ['make_reference_inputs'])],
            ['Pure target-SE3 tasks separately hide every source pose/root and omit contact.', 'New target-SE3 tokens and condition losses are explicit added inputs/objectives.'],
            ['masking.py:28-115', 'train_dual.py:376-411']),
        row('main direction mean', ['combo_sample', 'self_only'],
            'Draw8 distinct ordered directions from training-body Cartesian product including self; reuse one row batch across every direction; backward main losses scaled1/8. Auxiliary losses are additional and not divided by8. Clip accumulated gradients once.',
            {'combo_sample': recipe.get('combo_sample'), 'sparse_extra': config.get('xwbc_sparse_extra')},
            [hook('recipe_training.py', 'xwbc_combos', ['itertools.product']),
             hook('recipe_training.py', 'backward_training_step', ['xwbc_combos']),
             hook('train_pilot.py', 'main', ['backward_training_step'])],
            ['Additional pure-SE3 same-body direction with probability.25 and weight1 never replaces original eight directions.', 'UniMate full/masked/sparse tasks remain same-body only.'],
            ['train_dual.py:247', 'train_dual.py:371-375', 'train_dual.py:500-501']),
        row('temporal world-slip and height derivative', ['slip_pairs', 'w_slip_w', 'w_hpair', 'w_slip_c', 'slip_w', 'tdec'],
            'Old single-frame model (tdec0) nevertheless used adjacent-frame auxiliary pairs: world foot slip weight2 and root-height velocity MSE weight.1 at30Hz. Requires actual adjacent rows, reliable sequence boundaries, world-root/ground and sole data.',
            {'policy': recipe.get('temporal_slip_hpair')},
            differences=['Explicit temporal exception: disabled for sparsely selected nonadjacent single frames; do not fabricate adjacent pairs or apply hardcoded30Hz to row neighbors.'],
            evidence=['train_dual.py:515-608']),
        row('optimizer and bounded scale', ['lr', 'bs', 'steps', 'enc_layers', 'dec_layers'],
            'Original AdamW lr1e-4 weight_decay1e-4; CosineAnnealingLR over8000 steps; no warmup, one gradient clip1 after all main and auxiliary backwards.',
            {'optimizer': config.get('optimizer'), 'batch_size': config.get('batch_size'), 'steps': config.get('steps'), 'model': config.get('model')},
            [hook('train_pilot.py', 'main', ['torch.optim.AdamW', 'torch.optim.lr_scheduler.CosineAnnealingLR', 'torch.nn.utils.clip_grad_norm_'])],
            ['Authorized bounded batch256->128 and steps8000->3000/time budget; depth4->6; frozen SigLIP768->pinned frozen Qwen1024 with fresh projection; source-group sampling/splits and UniMate mixture.', 'Mixture55% XWBC applies original XWBC objectives on those steps; its global update frequency differs from the old all-XWBC run.'],
            ['train_dual.py:241-242', 'train_dual.py:660-661']),
        row('inactive original objectives', ['w_align', 'dof_loss', 'var_floor', 'var_norm', 'arm_boost', 'ee_boost', 'ee_fk_w', 'w_root'],
            'Do not add originally inactive objectives from CLI defaults.', {'old_values_retained_in_reference': True}),
    ]
    gate = training_recipe_status(config)
    connected = [r for r in rows if r['all_expected_connections_present'] is not None]
    file_hashes = {p.name: digest(p) for p in HERE.glob('*.py') if p.name in targets or p.name in ['audit_recipe.py', 'test_recipe_gate.py']}
    return {'schema': 'actual-run-recipe-source-audit-v2', 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'reference_config_sha256': REFERENCE_CONFIG_SHA256, 'reference_config_source': 'hash-matched actual L_M20k_0903 config bytes, not CLI defaults',
            'reference_sources': sources, 'reference_source_snapshot_available': source_root is not None,
            'current_source_inventory': inventory, 'current_source_sha256': file_hashes, 'rows': rows,
            'source_connection_summary': {'complete_rows': sum(r['all_expected_connections_present'] for r in connected), 'total_rows': len(connected)},
            'training_gate': gate, 'numerical_tests': {'executed_by_this_build': False, 'independent_test_file': 'test_recipe_gate.py'},
            'current_real_training': False, 'online_verification_performed': False,
            'limitations': ['Static presence/call checks are necessary evidence, not proof of correct reductions or numerical behavior.',
                           'Real source hashes, complete ground/static seal, FK, environment and all-loss real-batch forward/backward must pass an independent hash-bound validation receipt before training.',
                           'Offline synthetic tests do not prove real-data correctness or trained quality.'],
            'explicit_recipe_departures': config.get('explicit_recipe_departures', [])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=HERE / 'candidate_config.json')
    p.add_argument('--output', type=Path, default=HERE / 'recipe_review.json')
    args = p.parse_args()
    report = build(json.loads(args.config.read_text()))
    output = args.output / 'recipe_review.json' if args.output.is_dir() else args.output
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'connections': report['source_connection_summary'], 'train_allowed': report['training_gate']['train_allowed'], 'real_training': False}))


if __name__ == '__main__':
    main()
