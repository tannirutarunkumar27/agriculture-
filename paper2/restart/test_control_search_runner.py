"""Synthetic recovery, immutable bindings and complete-selection acceptance."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper2.restart import control_search_runner as runner
from paper2.restart import control_convergence as engine
from paper2.restart.verify_control_search import recompute_metrics
from paper2.restart import verify_control_search as auditor
from paper2.restart.test_control_convergence import ConvergenceTests
from paper2.restart.validation_search import ROOT, read, atomic_json, identity, sha


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'paper2/results')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'study'
        self.config = {'target_scales': [1., 10.], 'learning_rates': [.003], 'rounds': 3}
        self.plan = [{'id': f'f{f}_s{s}', 'fold': f'outer_0_inner_{f}', 'target_scale': s,
                      'learning_rate': .003, 'rounds': 3, 'report_rounds': [1, 3], 'seed': 42}
                     for f in range(7) for s in (1., 10.)]
        self.manifest = {'synthetic': True}
        self.fold = {'train': [0, 1, 2, 3], 'clients': {'a': [0, 1], 'b': [2, 3]},
                     'preprocessing_identity': 'synthetic'}
        self.arrays = {'x_train': np.ones((4, 2), np.float32), 'y_train': np.array([2, 3, 4, 5], np.float32),
                       'x_validation': np.ones((2, 2), np.float32), 'y_validation': np.array([3, 4], np.float32)}
        self.check = patch.object(runner, 'verify_execution', return_value=(self.manifest, self.config, self.plan, self.root / 'data'))
        self.load = patch.object(runner, 'load_inner', return_value=(self.fold, self.arrays))
        self.check.start(); self.load.start()
        self.addCleanup(self.check.stop); self.addCleanup(self.load.stop)

    def run_batch(self, **kwargs):
        return runner.execute(self.root / 'package', self.output, dry_run=False, **kwargs)

    def test_dry_run_and_caps(self):
        result = runner.execute(self.root / 'package', self.output)
        self.assertEqual(result['status'], 'DRY_RUN')
        self.assertFalse(self.output.exists())
        for cap in (0, True, -1, 1.5):
            with self.assertRaises(ValueError): self.run_batch(cap=cap)

    def test_cap_explicit_resume_cache_and_complete_selection(self):
        self.assertEqual(self.run_batch()['completed'], 1)
        self.assertFalse((self.output / 'selection.json').exists())
        with self.assertRaises(ValueError): self.run_batch()
        with patch.object(runner, 'candidate', wraps=engine.candidate) as fitted:
            self.assertEqual(self.run_batch(resume=True)['completed'], 2)
            self.assertEqual(fitted.call_count, 1)
        self.assertEqual(self.run_batch(cap=12, resume=True)['completed'], 14)
        with patch.object(runner, 'candidate', side_effect=AssertionError('Must not retrain')):
            self.assertEqual(self.run_batch(resume=True)['executed'], 0)
        self.assertTrue((self.output / 'selection.json').exists())
        for run in self.plan:
            self.assertEqual(len(read(self.output / 'runs' / run['id'] / 'attempts.json')), 1)

    def test_failure_journal_explicit_retry_and_exact_checkpoint_resume(self):
        def interrupted(folder, run, fold, arrays, token):
            engine.candidate(folder, run, fold, arrays, token, stop_after=1)
            raise RuntimeError('synthetic interruption')
        with patch.object(runner, 'candidate', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'): self.run_batch()
        folder = self.output / 'runs' / self.plan[0]['id']
        recorded = read(folder / 'attempts.json')[0]
        self.assertEqual(recorded['status'], 'FAILED')
        self.assertIn('synthetic interruption', recorded['traceback'])
        with self.assertRaisesRegex(RuntimeError, 'retry-failed'): self.run_batch(resume=True)
        self.run_batch(resume=True, retry_failed=True)
        token = identity({'binding': identity(self.manifest), 'run': self.plan[0]})
        full = engine.candidate(self.root / 'full', self.plan[0], self.fold, self.arrays, token)
        self.assertEqual(read(folder / 'result.json'), full)
        self.assertEqual([a['status'] for a in read(folder / 'attempts.json')], ['FAILED', 'COMPLETE'])

    def test_completed_checkpoint_finalizes_without_training(self):
        def interrupted(folder, run, fold, arrays, token):
            engine.candidate(folder, run, fold, arrays, token)
            (folder / 'completion.json').unlink()  # synthetic temporary crash before receipt commit
            raise RuntimeError('receipt interruption')
        with patch.object(runner, 'candidate', side_effect=interrupted):
            with self.assertRaises(RuntimeError): self.run_batch()
        with patch.object(engine, 'train_regression_client', side_effect=AssertionError('No repeated training')):
            self.run_batch(resume=True, retry_failed=True)

    def test_binding_cache_and_unbound_rejection(self):
        self.run_batch()
        atomic_json(self.output / 'binding.json', {'identity': 'changed'})
        with self.assertRaisesRegex(ValueError, 'binding'): self.run_batch(resume=True)
        atomic_json(self.output / 'binding.json', {'identity': identity(self.manifest)})
        folder = self.output / 'runs' / self.plan[0]['id']
        atomic_json(folder / 'result.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'assets'): self.run_batch(resume=True)
        empty = self.root / 'unbound'; empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            runner.execute(self.root / 'package', empty, resume=True, dry_run=False)

    def test_lock_and_protected_paths(self):
        self.run_batch()
        with runner.writer_lock(self.output):
            with self.assertRaises((RuntimeError, OSError)): self.run_batch(resume=True)
        for output in (self.root / 'package', self.root / 'data' / 'nested', ROOT / 'paper2'):
            with self.assertRaises(ValueError):
                runner.execute(self.root / 'package', output, dry_run=False)

    def test_corrupt_checkpoint_and_scope_rejected(self):
        run = self.plan[0]; token = 'synthetic'
        folder = self.root / 'candidate'
        result = engine.candidate(folder, run, self.fold, self.arrays, token)
        saved = torch.load(folder / 'checkpoint.pt', weights_only=True)
        for field, value in (('identity', 'changed'), ('done', 4), ('best_round', 99), ('snapshots', {})):
            altered = dict(saved, **{field: value})
            with self.assertRaises(ValueError): runner.validate_checkpoint(altered, run, token, True)
        result['outer_model_inference'] = True
        atomic_json(folder / 'result.json', result)
        atomic_json(folder / 'completion.json', {'identity': token, 'assets': {n: sha(folder / n) for n in ('result.json', 'checkpoint.pt')}})
        with self.assertRaisesRegex(ValueError, 'result'): runner.completed(folder, run, token)

    def test_independent_metrics_and_corrupt_best_model(self):
        run = self.plan[0]; folder = self.root / 'candidate'
        engine.candidate(folder, run, self.fold, self.arrays, 'token')
        saved = torch.load(folder / 'checkpoint.pt', weights_only=True)
        self.assertEqual(recompute_metrics(saved, run, self.fold, self.arrays)['best_round'], saved['best_round'])
        saved = copy.deepcopy(saved)
        for value in saved['best_state'].values(): value.add_(100.)
        with self.assertRaisesRegex(ValueError, 'Best validation'): recompute_metrics(saved, run, self.fold, self.arrays)

    def test_partial_and_complete_audit(self):
        self.run_batch()
        with patch.object(auditor, 'load_inner', return_value=(self.fold, self.arrays)):
            report = auditor.audit(self.root / 'package', self.output, self.root / 'partial_audit')
            self.assertEqual(report['verified'], 1)
            self.assertEqual(report['pending'], 13)
            self.run_batch(cap=13, resume=True)
            report = auditor.audit(self.root / 'package', self.output, self.root / 'complete_audit')
            self.assertEqual(report['status'], 'PASS_COMPLETE_INNER_SEARCH')
            selected = read(self.output / 'selection.json')
            selected['selections'][0]['refit_rounds'] = 99
            atomic_json(self.output / 'selection.json', selected)
            with self.assertRaisesRegex(ValueError, 'selection'):
                auditor.audit(self.root / 'package', self.output, self.root / 'corrupt_audit')

    def test_partial_checkpoint_model_tampering_blocks_resume(self):
        def interrupted(folder, run, fold, arrays, token):
            engine.candidate(folder, run, fold, arrays, token, stop_after=1)
            raise RuntimeError('interruption')
        with patch.object(runner, 'candidate', side_effect=interrupted):
            with self.assertRaises(RuntimeError): self.run_batch()
        path = self.output / 'runs' / self.plan[0]['id'] / 'checkpoint.pt'
        saved = torch.load(path, weights_only=True)
        for v in saved['state'].values(): v.add_(100.)
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'Final validation'):
            self.run_batch(resume=True, retry_failed=True)

    def synthetic_results(self):
        return {r['id']: {'run': r, 'history': [{'round': n, 'validation': {'rmse': 1., 'samples': 1}}
                    for n in range(1, 4)], 'outer_model_inference': False, 'privacy_claim': False} for r in self.plan}

    def test_incomplete_selection_and_first_tie_median(self):
        results = self.synthetic_results()
        with self.assertRaisesRegex(ValueError, 'Complete'): runner.select_complete(self.plan, {}, self.config)
        selected = runner.select_complete(self.plan, results, self.config)['selections'][0]
        self.assertEqual(selected['selected']['target_scale'], 1.)
        self.assertEqual(selected['refit_rounds'], 1)
        for i, r in enumerate(self.plan):
            if r['target_scale'] == 10.:
                at = [1, 1, 2, 2, 2, 3, 3][i // 2]
                results[r['id']]['history'][at - 1]['validation']['rmse'] = .5
        selected = runner.select_complete(self.plan, results, self.config)['selections'][0]
        self.assertEqual(selected['selected']['target_scale'], 10.)
        self.assertEqual(selected['refit_rounds'], 2)

    def test_equal_groups_not_sample_weighted(self):
        results = self.synthetic_results()
        for r in self.plan:
            score, count = (0., 1) if r['target_scale'] == 1. and r['fold'].endswith('_0') else (2., 1000)
            if r['target_scale'] == 10.: score, count = 1.8, 1000
            for h in results[r['id']]['history']: h['validation'].update(rmse=score, samples=count)
        # Unweighted mean 12/7 < 1.8, despite sample-weighted mean approximately 2.
        selected = runner.select_complete(self.plan, results, self.config)['selections'][0]['selected']
        self.assertEqual(selected['target_scale'], 1.)
        self.assertAlmostEqual(selected['equal_inner_group_mean_best_validation_rmse'], 12 / 7)

    def test_protocol_artifact_mutation(self):
        p = self.root / 'mutated'; p.mkdir()
        for name in ('manifest.json', 'config.json', 'plan.json'):
            (p / name).write_bytes((runner.PROTOCOL / name).read_bytes())
        atomic_json(p / 'plan.json', [])
        with self.assertRaisesRegex(ValueError, 'artifact'): runner.verify_protocol(p)

    def test_execution_source_mutation(self):
        p = self.root / 'execution'; p.mkdir(); (p / 'source_snapshot').mkdir()
        for source in runner.CODE: (p / 'source_snapshot' / source.name).write_bytes(source.read_bytes())
        atomic_json(p / 'manifest.json', {'protocol': str(runner.PROTOCOL),
                    'protocol_manifest_sha256': sha(runner.PROTOCOL / 'manifest.json'),
                    'source_sha256': {s.name: sha(s) for s in runner.CODE}, 'runtime': runner.runtime()})
        (p / 'source_snapshot' / runner.CODE[0].name).write_text('changed', encoding='utf-8')
        # Stop the fixture's mocked entry point so the real integrity guard is tested.
        self.check.stop()
        with self.assertRaisesRegex(ValueError, 'source'): runner.verify_execution(p)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (ConvergenceTests, SearchTests))
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
