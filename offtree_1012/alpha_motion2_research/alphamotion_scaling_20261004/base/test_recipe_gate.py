"""Offline recipe-gate rejection tests; all receipts/data below are synthetic.

No remote connection, model/checkpoint loading, CUDA execution, or real training.
These tests assert that incomplete/tampered evidence cannot authorize training.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get('PILOT_LEGACY_SOURCE', str(HERE / 'legacy')))
import recipe_gate as gate


def config():
    return json.loads((HERE / 'candidate_config.json').read_text())


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class RecipeGateTests(unittest.TestCase):
    def test_reference_is_actual_pinned_bytes_not_cli_defaults(self):
        raw = gate.REFERENCE_CONFIG.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '0e61ed5b012420d5fa186bc09537e9cac32ed14fa4533182c6d44805f5e7ee58')
        actual = gate.reference_config()
        self.assertEqual((actual['combo_sample'], actual['contrast_sample'], actual['n_judges']), (8, 6, 16))
        self.assertEqual((actual['w_pen'], actual['w_cgeo'], actual['w_hpair'], actual['w_slip_w']), (200., 1., .1, 2.))
        self.assertEqual(actual['mask_mix'], 'rand:0.25,ee:0.15,root:0.10,sparse:0.25,cask:0.15,cgive:0.10')

    def test_mutated_reference_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fake_reference.json'
            path.write_bytes(gate.REFERENCE_CONFIG.read_bytes() + b' ')
            with patch.object(gate, 'REFERENCE_CONFIG', path):
                with self.assertRaises(RuntimeError):
                    gate.reference_config()
                with self.assertRaises(RuntimeError):
                    gate.training_recipe_status(config())

    def test_no_receipt_is_blocked_even_with_config_ready_flags(self):
        c = config()
        c.update(train_allowed=True, recipe_status='ready', validation_passed=True)
        self.assertFalse(gate.training_recipe_status(c)['train_allowed'])
        with self.assertRaisesRegex(RuntimeError, 'TRAINING_RECIPE_BLOCKED'):
            gate.assert_training_recipe_ready(c)

    def test_missing_or_invalid_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'validation'; run.mkdir()
            root = Path(d) / 'root'; root.mkdir()
            for value in [None, {}, {'status': 'passed'}, {'status': 'failed'}, {'status': 'passed', 'implementation_revision': 'wrong'}]:
                if value is not None:
                    put(run / 'recipe_validation_receipt.json', value)
                with self.subTest(receipt=value):
                    self.assertFalse(gate.training_recipe_status(config(), run, root)['train_allowed'])

    def test_candidate_recipe_mismatches_are_not_config_overridable(self):
        base = config()
        self.assertEqual(gate.config_blockers(base), [])
        modifications = [(['reference_config_sha256'], '0' * 64),
                         (['model', 'n_judges'], 0), (['optimizer', 'lr'], 3e-4),
                         (['optimizer', 'weight_decay'], .01)]
        modifications += [(['recipe', key], (0 if isinstance(base['recipe'][key], (int, float)) else 'disabled'))
                          for key in ['w_contrast', 'tau', 'contrast_sample', 'reenc_consistency', 'reenc_prob',
                                      'n_judges', 'judge_mode', 'w_judge', 'w_pen', 'w_cgeo', 'mask_prob', 'mask_mix', 'combo_sample']]
        for keys, value in modifications:
            c = copy.deepcopy(base); target = c
            for key in keys[:-1]: target = target[key]
            target[keys[-1]] = value
            with self.subTest(key=keys):
                self.assertTrue(gate.config_blockers(c))

    def test_effective_objective_weights_contact_dropout_and_clip_are_pinned(self):
        base = config()
        for keys in [('loss', 'position'), ('loss', 'root_height'), ('loss', 'penetration'),
                     ('loss', 'contact_geometry'), ('optimizer', 'clip_norm'), ('contact_dropout',)]:
            c = copy.deepcopy(base); target = c
            for key in keys[:-1]: target = target[key]
            target[keys[-1]] = 0.
            with self.subTest(key=keys):
                self.assertTrue(gate.config_blockers(c), 'effective training setting was silently changed: ' + '/'.join(keys))

    def _minimal_untrusted_receipt(self, directory):
        """Deliberately incomplete receipt; never represented as a real validation."""
        c = config(); root = Path(directory) / 'root'; run = Path(directory) / 'validation'
        names = ['synthetic_body_' + str(i) for i in range(16)]
        put(root / 'data_xwbc' / 'chosen_split.json', {'body_names': names})
        fake = root / 'synthetic_file.txt'; fake.write_text('NOT REAL DATA OR CHECKPOINT')
        put(run / 'exit_status.json', {'status': 'recipe_validated_no_training'})
        put(run / 'real_batch_preflight.json', {'full_reference_recipe_verified': True, 'optimizer_steps': 0, 'gpu': 'cuda:0'})
        receipt = {'status': 'passed', 'implementation_revision': gate.IMPLEMENTATION_REVISION,
                   'configuration_sha256': gate.configuration_digest(c), 'code_sha256': gate.code_hashes(),
                   'root': str(root.resolve()), 'judge_body_order': names,
                   'live_files_sha256': {str(fake.resolve()): gate.sha256(fake)}, 'validation_artifacts': {}}
        put(run / 'recipe_validation_receipt.json', receipt)
        return c, root, run, receipt

    def test_minimal_receipt_with_empty_artifact_bindings_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as d:
            c, root, run, _ = self._minimal_untrusted_receipt(d)
            self.assertFalse(gate.training_recipe_status(c, run, root)['train_allowed'])
            with self.assertRaisesRegex(RuntimeError, 'TRAINING_RECIPE_BLOCKED'):
                gate.assert_training_recipe_ready(c, run, root)

    def test_stale_config_code_root_and_source_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c, root, run, receipt = self._minimal_untrusted_receipt(d)
            mutations = [dict(configuration_sha256='wrong'), dict(code_sha256={}), dict(root='/wrong'),
                         dict(live_files_sha256={}), dict(implementation_revision='old'),
                         dict(judge_body_order=['same'] * 16)]
            for change in mutations:
                bad = dict(receipt, **change); put(run / 'recipe_validation_receipt.json', bad)
                with self.subTest(change=change):
                    self.assertFalse(gate.training_recipe_status(c, run, root)['train_allowed'])
            put(run / 'recipe_validation_receipt.json', receipt)
            Path(next(iter(receipt['live_files_sha256']))).write_text('changed synthetic bytes')
            self.assertFalse(gate.training_recipe_status(c, run, root)['train_allowed'])

    def test_cpu_or_partial_preflight_cannot_write_validation_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            for report in [{}, {'full_reference_recipe_verified': False, 'optimizer_steps': 0, 'gpu': 'cuda:0'},
                           {'full_reference_recipe_verified': True, 'optimizer_steps': 0, 'gpu': 'cpu'},
                           {'full_reference_recipe_verified': True, 'optimizer_steps': 1, 'gpu': 'cuda:0'}]:
                with self.subTest(report=report), self.assertRaises((RuntimeError, ValueError)):
                    gate.write_validation_receipt(config(), Path(d) / 'root', Path(d) / 'run', report)
            self.assertFalse((Path(d) / 'run' / 'recipe_validation_receipt.json').exists())

    def test_cli_train_rejects_before_cuda_capacity_lock_or_data(self):
        import train_pilot
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / 'config.json'; c = config(); c['train_allowed'] = True; put(cfg, c)
            argv = ['train_pilot.py', '--config', str(cfg), '--action', 'train']
            with patch.object(sys, 'argv', argv), \
                 patch.object(train_pilot.torch.cuda, 'is_available', side_effect=AssertionError('CUDA inspected')) as cuda, \
                 patch.object(train_pilot, 'capacity_guard', side_effect=AssertionError('capacity touched')) as capacity, \
                 patch.object(train_pilot, 'load_bodies', side_effect=AssertionError('data loaded')) as loader, \
                 patch.object(train_pilot.fcntl, 'flock', side_effect=AssertionError('lock touched')) as lock:
                with self.assertRaisesRegex(RuntimeError, 'TRAINING_RECIPE_BLOCKED'):
                    train_pilot.main()
                for mocked in [cuda, capacity, loader, lock]: mocked.assert_not_called()
            self.assertFalse((Path(d) / 'runs').exists())

    @staticmethod
    def synthetic_report_shape():
        # Deliberately synthetic parser fixture; never persisted as validation evidence.
        directions = [dict(source='synthetic_' + str(i), target='synthetic_' + str((i + 1) % 16),
                           mode='reference_mask_mix', rows=128, valid_conditions=16, weight=.125) for i in range(8)]
        directions += [dict(source='synthetic_0', target='synthetic_0', mode='sparse_only_extra',
                            rows=128, valid_conditions=16, weight=1.)]
        terms = {key: .25 for key in ['total', 'pen', 'cgeo', 'root_height_smooth_l1_m',
                                     'rotation_deg', 'position_over_reach', 'contrast', 'judge', 'reenc']}
        return dict(status='passed', full_reference_recipe_verified=True, optimizer_steps=0, gpu='cuda:0',
                    hidden_source_perturbation_invariant=True, full_recipe_profiles=[
            dict(domain='xwbc', optimizer_steps=0, gradient_norm=1., forward_backward_seconds=.5,
                 terms=terms, directions=directions, auxiliary=dict(reenc_active=True,
                 contrast_bodies=['synthetic_' + str(i) for i in range(6)],
                 contrast=dict(active=True, unique_rows=128, pairs=15, duplicate_rows=0))),
            dict(domain='unimate', optimizer_steps=0, gradient_norm=1., forward_backward_seconds=.5,
                 terms=dict(total=.2, rotation_deg=.1, position_over_reach=.1),
                 directions=[dict(source='synthetic_native', target='synthetic_native', mode='full',
                                  rows=128, valid_conditions=0, weight=1.)],
                 auxiliary=dict(cross_body_auxiliary_applied=False))])

    def test_complete_synthetic_report_schema_only_is_accepted(self):
        # Acceptance here is only of report structure, not a receipt or a CUDA run.
        self.assertTrue(gate.check_full_report(self.synthetic_report_shape()))

    def test_report_nonfinite_gradient_or_loss_is_rejected(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            for field in ['gradient_norm', 'loss']:
                r = self.synthetic_report_shape()
                if field == 'gradient_norm': r['full_recipe_profiles'][0]['gradient_norm'] = value
                else: r['full_recipe_profiles'][0]['terms']['pen'] = value
                with self.subTest(value=value, field=field), self.assertRaises((ValueError, TypeError)):
                    gate.check_full_report(r)

    def test_report_requires_active_losses_cycle_unique_contrast_and_no_unimate_aux(self):
        reports = []
        r = self.synthetic_report_shape(); del r['full_recipe_profiles'][0]['terms']['pen']; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['auxiliary']['reenc_active'] = False; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['auxiliary']['contrast']['active'] = False; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['auxiliary']['contrast_bodies'] = ['same'] * 6; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][1]['auxiliary']['cross_body_auxiliary_applied'] = True; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'] = r['full_recipe_profiles'][:1]; reports.append(r)
        for i, r in enumerate(reports):
            with self.subTest(case=i), self.assertRaises((ValueError, TypeError)):
                gate.check_full_report(r)

    def test_report_requires_eight_unique_mean_reduced_directions_plus_extra(self):
        reports = []
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['directions'][0]['weight'] = 1.; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['directions'][-1]['weight'] = .125; reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['directions'][1] = copy.deepcopy(r['full_recipe_profiles'][0]['directions'][0]); reports.append(r)
        r = self.synthetic_report_shape(); r['full_recipe_profiles'][0]['directions'].pop(); reports.append(r)
        for i, r in enumerate(reports):
            with self.subTest(case=i), self.assertRaises((ValueError, TypeError)):
                gate.check_full_report(r)

    def test_temporal_exception_does_not_erase_single_frame_requirements(self):
        status = gate.training_recipe_status(config())
        self.assertTrue(status['temporal_exceptions'])
        flags = status['verified_active_reference_flags']
        for name in ['w_contrast', 'reenc_consistency', 'w_judge', 'w_pen', 'w_cgeo']:
            self.assertGreater(flags[name], 0)
        self.assertFalse(status['train_allowed'])

    def test_source_audit_distinguishes_static_connections_and_execution(self):
        import audit_recipe
        report = audit_recipe.build(config())
        self.assertFalse(report['current_real_training'])
        self.assertFalse(report['online_verification_performed'])
        self.assertFalse(report['numerical_tests']['executed_by_this_build'])
        self.assertEqual(report['reference_config_sha256'], gate.REFERENCE_CONFIG_SHA256)
        self.assertTrue(report['reference_source_snapshot_available'])
        self.assertTrue(any('false negatives' in ' '.join(r['explicit_differences']) for r in report['rows']))
        self.assertTrue(all('not execution proof' in r['assessment_scope'] for r in report['rows']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
