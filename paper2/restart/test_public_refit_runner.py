"""Acceptance for once-only raw-unit inference and the separate capped runner."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper2.restart import public_refit_runner as runner
from paper2.restart import public_refit_inference as inference
from paper2.restart import public_refit_training as training
from paper2.restart import verify_public_refit as auditor
from paper2.restart.validation_search import ROOT, read, sha, identity, atomic_json


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'paper2/results'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.data = self.root / 'data'
        self.output = self.root / 'study'; self.package = self.root / 'execution'; self.package.mkdir()
        self.m = {'fixture': 'synthetic'}; atomic_json(self.package / 'manifest.json', self.m)
        self.config = {'neural_settings': training.SETTINGS}
        self.plan = [{'id': f'outer_0_equal_public_seed{s}', 'fold': 'outer_0',
                      'method': 'fixed_public_equal_client_control', 'seed': s, 'rounds': 3,
                      'learning_rate': .01, 'target_scale': 10.} for s in (123, 2024)]
        self.fold = {'kind': 'outer_refit', 'train': [0, 1, 2, 3], 'validation': [], 'test': [4, 5],
                     'clients': {'a': [0], 'b': [1, 2, 3]}, 'preprocessing_identity': 'synthetic',
                     'test_group': 'synthetic', 'support': {'test': [
                     {'all_species_seen': True, 'composition_seen': False},
                     {'all_species_seen': False, 'composition_seen': True}]}}
        self.train = {'x_train': np.arange(8, dtype=np.float32).reshape(4, 2) / 8,
                      'y_train': np.array([2, 3, 4, 5], np.float32), 'ids_train': np.arange(4, dtype=np.int64)}
        self.tests = {'x_test': np.ones((2, 2), np.float32), 'y_test': np.array([3, 4], np.float32),
                      'ids_test': np.array([4, 5], np.int64)}
        folder = self.data / 'folds/outer_0'; folder.mkdir(parents=True)
        atomic_json(folder / 'partition.json', self.fold)
        atomic_json(folder / 'preprocessing.json', {'identity': 'synthetic'})
        np.savez(folder / 'arrays.npz', **self.train, **self.tests)
        atomic_json(self.data / 'manifest.json', {'assets': {'outer_0': {n: sha(folder / n) for n in ('arrays.npz', 'partition.json', 'preprocessing.json')}}})
        mocked = patch.object(runner, 'verify_execution', return_value=(self.m, self.config, self.plan, self.data))
        mocked.start(); self.addCleanup(mocked.stop)

    def execute(self, **kwargs):
        return runner.execute(self.package, self.output, dry_run=False, **kwargs)

    def folder(self, index=0):
        return self.output / 'runs' / self.plan[index]['id']

    def token(self, index=0):
        return identity({'binding': identity(self.m), 'run': self.plan[index]})

    def test_dry_run_and_positive_caps(self):
        self.assertEqual(runner.execute(self.package, self.output)['status'], 'DRY_RUN_NO_TRAINING_OR_INFERENCE')
        self.assertFalse(self.output.exists())
        for cap in (0, True, -1, 1.5):
            with self.assertRaises(ValueError): self.execute(cap=cap)

    def test_capped_resume_complete_cache_and_independent_audit(self):
        self.assertEqual(self.execute()['completed'], 1)
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(self.execute(resume=True)['completed'], 2)
        hashes = {r['id']: sha(self.output / 'runs' / r['id'] / 'attempts.json') for r in self.plan}
        with patch.object(training, 'fit', side_effect=AssertionError('retrain')), patch.object(training, 'predict_raw', side_effect=AssertionError('reinference')):
            self.assertEqual(self.execute(resume=True)['executed'], 0)
            self.assertEqual(auditor.audit(self.package, self.output, self.root / 'audit')['status'], 'PASS_COMPLETE_REFIT')
        self.assertEqual(hashes, {r['id']: sha(self.output / 'runs' / r['id'] / 'attempts.json') for r in self.plan})

    def test_partial_audit(self):
        self.execute()
        result = auditor.audit(self.package, self.output, self.root / 'partial')
        self.assertEqual(result['verified'], 1); self.assertEqual(result['pending'], 1)

    def test_raw_units_restored_once_and_only_one_test_prediction(self):
        original = inference.evaluate_once; predictions = []
        def counted(*args, **kwargs):
            with patch.object(training, 'predict_raw', wraps=training.predict_raw) as measured:
                value = original(*args, **kwargs)
                predictions.append(measured.call_count)
                return value
        with patch.object(inference, 'evaluate_once', side_effect=counted): self.execute()
        self.assertEqual(predictions, [1])
        run = self.plan[0]; folder = self.folder(); token = self.token()
        state = torch.load(folder / 'model.pt', weights_only=True)
        model = training.make_regressor(2, 32).eval(); model.load_state_dict(state)
        with torch.no_grad(): expected = model(torch.from_numpy(self.tests['x_test']))[:, 0].numpy() * 10.
        with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['prediction'], expected)
        with patch.object(training, 'predict_raw', side_effect=AssertionError('repeat')):
            inference.evaluate_once(folder, run, self.fold, self.train, self.tests, token)

    def test_ambiguous_inference_stops_even_with_retry(self):
        def failed(*args, **kwargs): raise RuntimeError('synthetic inference interruption')
        original = inference.evaluate_once
        def interrupted(*args, **kwargs):
            with patch.object(training, 'predict_raw', side_effect=failed): return original(*args, **kwargs)
        with patch.object(inference, 'evaluate_once', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interruption'): self.execute()
        self.assertTrue((self.folder() / 'inference_started.json').exists())
        for retry in (False, True):
            with self.assertRaisesRegex(RuntimeError, 'Ambiguous'): self.execute(resume=True, retry_failed=retry)
        self.assertEqual(len(read(self.folder() / 'attempts.json')), 1)

    def test_training_failure_journal_and_explicit_checkpoint_recovery(self):
        original = training.fit
        def interrupted(*args, **kwargs):
            original(*args, **dict(kwargs, stop_after=1))
            raise RuntimeError('synthetic training interruption')
        with patch.object(training, 'fit', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interruption'): self.execute()
        self.assertIn('synthetic training interruption', read(self.folder() / 'attempts.json')[0]['traceback'])
        with self.assertRaisesRegex(RuntimeError, 'retry-failed'): self.execute(resume=True)
        self.execute(resume=True, retry_failed=True)
        self.assertEqual([a['status'] for a in read(self.folder() / 'attempts.json')], ['FAILED', 'COMPLETE'])
        original(self.root / 'uninterrupted', self.plan[0], training.SETTINGS, self.fold,
                 self.train['x_train'], self.train['y_train'], self.token())
        state_a = torch.load(self.folder() / 'model.pt', weights_only=True)
        state_b = torch.load(self.root / 'uninterrupted/model.pt', weights_only=True)
        self.assertTrue(all(torch.equal(state_a[k], state_b[k]) for k in state_a))

    def test_completion_after_crash_recovered_without_inference(self):
        self.execute()
        path = self.folder() / 'attempts.json'; attempts = read(path)
        attempts[-1]['status'] = 'STARTED'; attempts[-1].pop('completion_sha256')
        atomic_json(path, attempts)
        with patch.object(training, 'predict_raw', side_effect=AssertionError('repeat')):
            self.plan[:] = self.plan[:1]
            self.assertEqual(self.execute(resume=True)['executed'], 0)
        self.assertEqual(read(path)[0]['status'], 'CACHE_VERIFIED_AFTER_INTERRUPTION')

    def test_failed_postcommit_attempt_retained_on_cache_recovery(self):
        self.execute(); self.plan[:] = self.plan[:1]
        path = self.folder() / 'attempts.json'; attempts = read(path)
        attempts[-1].update(status='FAILED', traceback='synthetic postcommit failure'); atomic_json(path, attempts)
        self.execute(resume=True)
        self.assertEqual([a['status'] for a in read(path)], ['FAILED', 'CACHE_VERIFIED_AFTER_FAILURE'])
        audit = auditor.audit(self.package, self.output, self.root / 'recovery-audit')
        self.assertEqual(len(audit['failed_attempts']), 1)

    def test_wrong_binding_lock_and_unbound_output(self):
        self.execute()
        with runner.writer_lock(self.output):
            with self.assertRaises((RuntimeError, OSError)): self.execute(resume=True)
        atomic_json(self.output / 'binding.json', {'identity': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'binding'): self.execute(resume=True)
        unbound = self.root / 'unbound'; unbound.mkdir()
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            runner.execute(self.package, unbound, resume=True, dry_run=False)

    def test_protected_path_rejected(self):
        for p in (self.package, self.data / 'child', ROOT / 'paper2'):
            with self.assertRaises(ValueError): runner.execute(self.package, p, dry_run=False)

    def test_journal_missing_identity_and_receipt_tampering_rejected(self):
        self.execute(); path = self.folder() / 'attempts.json'; original = read(path)
        for key, value in (('identity', 'wrong'), ('attempt', 7), ('completion_sha256', 'wrong')):
            bad = copy.deepcopy(original); bad[-1][key] = value; atomic_json(path, bad)
            with self.assertRaises(ValueError): self.execute(resume=True)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Unjournaled'): self.execute(resume=True)

    def test_result_asset_and_panel_tampering_rejected(self):
        self.execute(); folder = self.folder()
        result = read(folder / 'result.json'); result['test']['rmse'] += 1
        atomic_json(folder / 'result.json', result)
        with self.assertRaisesRegex(ValueError, 'assets'): self.execute(resume=True)
        receipt = read(folder / 'completion.json'); receipt['assets']['result.json'] = sha(folder / 'result.json')
        atomic_json(folder / 'completion.json', receipt)
        with self.assertRaisesRegex(ValueError, 'metrics'): self.execute(resume=True)

    def test_test_ids_and_support_context_binding(self):
        self.execute(); bad = dict(self.tests, ids_test=np.array([5, 4], np.int64))
        with self.assertRaises(ValueError): inference.verify_completed(self.folder(), self.plan[0], self.fold, self.train, bad, self.token())
        fold = copy.deepcopy(self.fold); fold['support']['test'][0]['all_species_seen'] = False
        with self.assertRaisesRegex(ValueError, 'signature|binding'): inference.verify_completed(self.folder(), self.plan[0], fold, self.train, self.tests, self.token())

    def test_checkpoint_dtype_corruption_rejected(self):
        self.execute(); folder = self.folder()
        saved = torch.load(folder / 'checkpoint.pt', weights_only=True)
        saved['state'] = {k: v.double() for k, v in saved['state'].items()}
        torch.save(saved, folder / 'checkpoint.pt')
        receipt = read(folder / 'fit.json'); receipt['assets']['checkpoint.pt'] = sha(folder / 'checkpoint.pt')
        atomic_json(folder / 'fit.json', receipt)
        with self.assertRaisesRegex(ValueError, 'checkpoint tensor'):
            inference.verify_fit(folder, self.plan[0], self.fold, self.train, self.token())

    def test_partial_checkpoint_dtype_rejected_before_training(self):
        original = training.fit
        def interrupted(*args, **kwargs):
            original(*args, **dict(kwargs, stop_after=1))
            raise RuntimeError('synthetic training interruption')
        with patch.object(training, 'fit', side_effect=interrupted):
            with self.assertRaises(RuntimeError): self.execute()
        path = self.folder() / 'checkpoint.pt'; saved = torch.load(path, weights_only=True)
        saved['state'] = {k: v.double() for k, v in saved['state'].items()}; torch.save(saved, path)
        with patch.object(training, 'fit', side_effect=AssertionError('Must reject before refit')):
            with self.assertRaisesRegex(ValueError, 'checkpoint tensor'):
                self.execute(resume=True, retry_failed=True)

    def test_independent_undefined_r2_and_empty_panels(self):
        self.assertIsNone(auditor.independent_scores([2.], [3.])['r2'])
        self.assertIsNone(auditor.independent_scores([2., 2.], [3., 3.])['r2'])
        context = {'support': [{'all_species_seen': True, 'composition_seen': True}]}
        panels = inference.measured_panels(np.array([2.]), np.array([3.]), context)
        self.assertIsNone(panels['all_species_seen']['unseen'])


class ExecutionBindingTests(unittest.TestCase):
    def test_source_and_protocol_manifest_mutations(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'paper2/results') as tmp:
            package = Path(tmp); (package / 'source_snapshot').mkdir()
            for p in runner.CODE: (package / 'source_snapshot' / p.name).write_bytes(p.read_bytes())
            record = {'protocol': str(runner.PROTOCOL), 'protocol_manifest_sha256': sha(runner.PROTOCOL / 'manifest.json'),
                      'runtime': runner.runtime(), 'planned': 40, 'privacy_claim': False,
                      'source_sha256': {p.name: sha(p) for p in runner.CODE}}
            atomic_json(package / 'manifest.json', dict(record, protocol_manifest_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'protocol'): runner.verify_execution(package)
            atomic_json(package / 'manifest.json', record)
            (package / 'source_snapshot' / runner.CODE[0].name).write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source'): runner.verify_execution(package)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (RunnerTests, ExecutionBindingTests))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
