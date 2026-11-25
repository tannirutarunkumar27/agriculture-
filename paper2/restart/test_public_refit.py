"""Regression for the legacy audit gap, strict provenance and train-only refit."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper2.restart import control_search_integrity as integrity
from paper2.restart import public_refit_protocol as protocol
from paper2.restart import public_refit_training as training
from paper2.restart import verify_control_search as legacy_audit
from paper2.restart.control_convergence import candidate
from paper2.restart.test_control_search_runner import SearchTests
from paper2.restart.validation_search import ROOT, identity, read, sha, atomic_json
from paper2.restart.validation_search import runtime


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SearchTests('test_dry_run_and_caps')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.run_batch(cap=14)
        self.study, self.plan = self.fixture.output, self.fixture.plan
        self.binding = identity(self.fixture.manifest)

    def verify(self):
        return integrity.verify_attempts(self.study, self.plan, self.binding)

    def test_legacy_gap_reproduced_and_new_guard_rejects_missing_journal(self):
        path = self.study / 'runs' / self.plan[0]['id'] / 'attempts.json'
        path.unlink()  # generated temporary fixture only; real journals remain untouched
        with patch.object(legacy_audit, 'load_inner', return_value=(self.fixture.fold, self.fixture.arrays)):
            result = legacy_audit.audit(self.fixture.root / 'package', self.study, self.fixture.root / 'legacy_missing_journal')
        self.assertEqual(result['status'], 'PASS_COMPLETE_INNER_SEARCH')
        with self.assertRaisesRegex(ValueError, 'Missing attempt'): self.verify()

    def test_journal_identity_sequence_status_and_receipt_corruption(self):
        path = self.study / 'runs' / self.plan[0]['id'] / 'attempts.json'
        original = read(path)
        for key, value in (('identity', 'wrong'), ('attempt', 7), ('status', 'STARTED'),
                           ('completion_sha256', 'wrong'), ('started_utc', '2026-10-14T00:00:00')):
            bad = copy.deepcopy(original); bad[-1][key] = value; atomic_json(path, bad)
            with self.assertRaises(ValueError): self.verify()
        atomic_json(path, original)
        self.assertEqual(len(self.verify()), 14)

    def test_failed_history_preserved_and_cache_verified_interrupt_allowed(self):
        path = self.study / 'runs' / self.plan[0]['id'] / 'attempts.json'
        original = read(path)[0]
        failed = dict(original, status='FAILED', traceback='synthetic retained failure')
        final = dict(original, attempt=2)
        atomic_json(path, [failed, final])
        self.assertEqual(self.verify()[0]['failed_attempts'], 1)
        failed.pop('traceback'); atomic_json(path, [failed, final])
        with self.assertRaisesRegex(ValueError, 'traceback'): self.verify()
        final = dict(original, status='CACHE_VERIFIED_AFTER_INTERRUPTION')
        final.pop('completion_sha256'); atomic_json(path, [final])
        self.assertEqual(len(self.verify()), 14)

    def test_strict_metric_receipt_and_selection_binding(self):
        package = self.fixture.root / 'package'; package.mkdir()
        atomic_json(package / 'manifest.json', self.fixture.manifest)
        path = self.fixture.root / 'metric_audit'
        with patch.object(legacy_audit, 'load_inner', return_value=(self.fixture.fold, self.fixture.arrays)):
            legacy_audit.audit(self.fixture.root / 'package', self.study, path)
        def checked():
            return integrity.verify_search(self.fixture.root / 'package', self.study, path / 'verification.json')
        self.assertEqual(checked()[0]['verified'], 14)
        report = read(path / 'verification.json'); report['rows'][0]['completion_sha256'] = 'wrong'
        atomic_json(path / 'verification.json', report)
        with self.assertRaisesRegex(ValueError, 'binding'): checked()


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.selected = {'status': 'COMPLETE_INNER_SELECTION', 'candidates': 224,
                         'outer_model_inference': False, 'privacy_claim': False,
                         'selections': [{'outer_fold': f'outer_{i}', 'selected': {'target_scale': 1., 'learning_rate': .01},
                                         'refit_rounds': 106 + i} for i in range(8)]}
        self.data = {'assets': {f'outer_{i}': {} for i in range(8)}}

    def test_forty_unique_runs_preserve_selected_budget_and_units(self):
        plan = protocol.make_plan(self.selected, self.data, protocol.SEEDS)
        self.assertEqual(len(plan), 40)
        self.assertEqual(len({r['id'] for r in plan}), 40)
        self.assertEqual(plan[0]['rounds'], 106)
        self.assertEqual(plan[-1]['rounds'], 113)
        self.assertTrue(all(r['target_scale'] == 1. and r['learning_rate'] == .01 for r in plan))

    def test_bad_seeds_and_incomplete_selection_rejected(self):
        for seeds in ([], [42], [1, 1], [True], [-1]):
            with self.assertRaises(ValueError): protocol.make_plan(self.selected, self.data, seeds)
        self.selected['selections'].pop()
        with self.assertRaises(ValueError): protocol.make_plan(self.selected, self.data, protocol.SEEDS)

    def test_test_based_selection_invalid_scale_and_budget_rejected(self):
        altered = copy.deepcopy(self.selected); altered['outer_model_inference'] = True
        with self.assertRaises(ValueError): protocol.make_plan(altered, self.data, protocol.SEEDS)
        for field, value in (('refit_rounds', 241), ('refit_rounds', True)):
            altered = copy.deepcopy(self.selected); altered['selections'][0][field] = value
            with self.assertRaises(ValueError): protocol.make_plan(altered, self.data, protocol.SEEDS)
        altered = copy.deepcopy(self.selected); altered['selections'][0]['selected']['target_scale'] = 0.
        with self.assertRaises(ValueError): protocol.make_plan(altered, self.data, protocol.SEEDS)

    def copied_package(self, root):
        data = root / 'data'; data.mkdir(); atomic_json(data / 'manifest.json', self.data)
        package = root / 'package'; package.mkdir(); (package / 'source_snapshot').mkdir()
        verified = {'execution_manifest_sha256': 'synthetic-execution', 'metric_audit_sha256': 'synthetic-metrics'}
        artifacts = {'config.json': protocol.make_config(), 'plan.json': protocol.make_plan(self.selected, self.data, protocol.SEEDS),
                     'selected.json': self.selected, 'search_integrity.json': verified}
        for name, value in artifacts.items(): atomic_json(package / name, value)
        for source in protocol.CODE: (package / 'source_snapshot' / source.name).write_bytes(source.read_bytes())
        atomic_json(package / 'manifest.json', {'runtime': runtime(), 'planned': 40, 'data': str(data),
                    'data_manifest_sha256': sha(data / 'manifest.json'), 'metric_audit': 'synthetic-metrics-path',
                    'metric_audit_sha256': verified['metric_audit_sha256'],
                    'parent_execution_manifest_sha256': verified['execution_manifest_sha256'],
                    'artifacts': {n: sha(package / n) for n in artifacts},
                    'source_sha256': {s.name: sha(s) for s in protocol.CODE}})
        return package, verified, data

    def test_copied_protocol_artifacts_and_semantic_overrides_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'paper2/results') as tmp:
            package, verified, data = self.copied_package(Path(tmp))
            with patch.object(protocol.integrity, 'verify_search', return_value=(verified, self.selected, data)):
                self.assertEqual(protocol.verify(package)['planned'], 40)
                config = read(package / 'config.json'); config['privacy_claim'] = True
                atomic_json(package / 'config.json', config)
                with self.assertRaisesRegex(ValueError, 'package'): protocol.verify(package)
                # Rehashing altered artifacts must not authorize a different scientific contract.
                m = read(package / 'manifest.json'); m['artifacts']['config.json'] = sha(package / 'config.json')
                atomic_json(package / 'manifest.json', m)
                with self.assertRaisesRegex(ValueError, 'configuration'): protocol.verify(package)

    def test_copied_protocol_source_and_parent_binding_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'paper2/results') as tmp:
            package, verified, data = self.copied_package(Path(tmp))
            with patch.object(protocol.integrity, 'verify_search', return_value=(dict(verified, metric_audit_sha256='changed'), self.selected, data)):
                with self.assertRaisesRegex(ValueError, 'binding'): protocol.verify(package)
            (package / 'source_snapshot' / protocol.CODE[0].name).write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source'): protocol.verify(package)


class RefitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'paper2/results'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = {'id': 'synthetic', 'fold': 'outer_0', 'method': 'fixed_public_equal_client_control',
                    'seed': 123, 'rounds': 3, 'learning_rate': .01, 'target_scale': 10.}
        self.fold = {'kind': 'outer_refit', 'train': [0, 1, 2, 3], 'validation': [], 'test': [4, 5],
                     'clients': {'a': [0], 'b': [1, 2, 3]}, 'preprocessing_identity': 'synthetic'}
        self.x = np.arange(8, dtype=np.float32).reshape(4, 2) / 8
        self.y = np.array([2, 3, 4, 5], dtype=np.float32)

    def fit(self, folder, **kwargs):
        return training.fit(folder, self.run, training.SETTINGS, self.fold, self.x, self.y, 'token', **kwargs)

    def test_matches_frozen_inner_engine_at_same_budget(self):
        self.fit(self.root / 'refit')
        inner = dict(self.run, report_rounds=[3])
        a = {'x_train': self.x, 'y_train': self.y, 'x_validation': self.x[:2], 'y_validation': self.y[:2]}
        candidate(self.root / 'inner', inner, self.fold, a, 'inner-token')
        r = torch.load(self.root / 'refit/model.pt', weights_only=True)
        i = torch.load(self.root / 'inner/checkpoint.pt', weights_only=True)['state']
        self.assertTrue(all(torch.equal(r[k], i[k]) for k in r))

    def test_exact_resume_and_completed_cache_do_not_train(self):
        self.fit(self.root / 'a'); self.fit(self.root / 'b', stop_after=1); self.fit(self.root / 'b')
        a = torch.load(self.root / 'a/model.pt', weights_only=True)
        b = torch.load(self.root / 'b/model.pt', weights_only=True)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        with patch.object(training, 'train_regression_client', side_effect=AssertionError('retrain')):
            self.fit(self.root / 'b')

    def test_fixed_completed_round_checkpoint_finalizes_without_training(self):
        self.fit(self.root / 'a')
        (self.root / 'a/fit.json').unlink()
        with patch.object(training, 'train_regression_client', side_effect=AssertionError('retrain')):
            self.fit(self.root / 'a')

    def test_changed_training_data_or_tokens_reject_recovery(self):
        self.fit(self.root / 'a', stop_after=1)
        self.y = self.y + 1
        with self.assertRaisesRegex(ValueError, 'identity'): self.fit(self.root / 'a')
        self.y = self.y - 1
        with self.assertRaisesRegex(ValueError, 'identity'):
            training.fit(self.root / 'a', self.run, training.SETTINGS, self.fold, self.x, self.y, 'new-token')

    def test_bad_client_isolation_and_settings_rejected(self):
        self.fold['clients']['a'] = [4]
        with self.assertRaises(ValueError): self.fit(self.root / 'bad')
        self.fold['clients']['a'] = [0]; self.fold['validation'] = [1]
        with self.assertRaises(ValueError): self.fit(self.root / 'bad')
        self.fold['validation'] = []
        with self.assertRaises(ValueError):
            training.fit(self.root / 'bad', self.run, dict(training.SETTINGS, local_steps=5), self.fold, self.x, self.y, 'token')

    def test_partial_state_tampering_blocks_resume(self):
        self.fit(self.root / 'a', stop_after=1)
        path = self.root / 'a/checkpoint.pt'; saved = torch.load(path, weights_only=True)
        for v in saved['state'].values(): v.add_(100.)
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'metrics'): self.fit(self.root / 'a')

    def test_unit_restoration_once_and_invalid_scale(self):
        model = training.make_regressor(2, 32).eval()
        with torch.no_grad(): normalized = model(torch.from_numpy(self.x))[:, 0].numpy()
        np.testing.assert_array_equal(training.predict_raw(model, self.x, 10.), normalized * 10)
        for s in (0, True, float('nan')):
            with self.assertRaises(ValueError): training.predict_raw(model, self.x, s)

    def test_completed_fit_unit_metadata_tampering_rejected(self):
        self.fit(self.root / 'a')
        path = self.root / 'a/fit.json'; receipt = read(path)
        receipt['model_representation'] = 'already raw; do not scale'
        atomic_json(path, receipt)
        with self.assertRaisesRegex(ValueError, 'Completed refit'): self.fit(self.root / 'a')

    def test_training_loader_never_reads_test_arrays(self):
        data = self.root / 'data'; folder = data / 'folds/outer_0'; folder.mkdir(parents=True)
        atomic_json(folder / 'partition.json', self.fold)
        atomic_json(folder / 'preprocessing.json', {'identity': 'synthetic'})
        # Object test arrays would raise under allow_pickle=False if fetched.
        np.savez(folder / 'arrays.npz', x_train=self.x, y_train=self.y, ids_train=self.fold['train'],
                 x_test=np.array([object()], dtype=object), y_test=np.array([object()], dtype=object), ids_test=[4, 5])
        atomic_json(data / 'manifest.json', {'assets': {'outer_0': {n: sha(folder / n) for n in ('arrays.npz', 'partition.json', 'preprocessing.json')}}})
        _, arrays = training.load_training_fold(data, self.run)
        self.assertEqual(set(arrays), {'x_train', 'y_train', 'ids_train'})
        np.testing.assert_array_equal(arrays['x_train'], self.x)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (IntegrityTests, PlanTests, RefitTests))
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
                'scope': 'synthetic audit-gap reproduction, journal regression, planner and train-only fixed refit'})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
