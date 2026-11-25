"""Bounded-coordinator tests on synthetic seven-client data; no real inference."""
import argparse
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from paper2.restart import matched_noise_runner as runner
from paper2.restart import verify_matched_noise as audit
from paper2.restart.test_matched_noise_training import TrainingTests
from paper2.restart.test_matched_noise_protocol import fixture
from paper2.restart.validation_search import read, sha, identity, atomic_json, writer_lock


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TrainingTests('test_all_variants_smoke_exact_events_and_work'); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups); self.root = self.fixture.root
        self.plan = self.fixture.runs
        self.fold = dict(self.fixture.fold, test_group='held-out', support={'test': [
            {'all_species_seen': True, 'composition_seen': True},
            {'all_species_seen': False, 'composition_seen': False}]})
        self.arrays = {'x_train': self.fixture.x, 'y_train': self.fixture.y, 'ids_train': np.arange(14, dtype=np.int64)}
        self.test = {'x_test': np.array([[0., 1.], [1., 0.]], np.float32),
                     'y_test': np.array([1., 2.], np.float32), 'ids_test': np.array([14, 15], np.int64)}
        self.package = self.root / 'package'; self.package.mkdir()
        self.protocol = self.root / 'protocol'; self.protocol.mkdir()
        self.study = self.root / 'study'; self.data = self.root / 'data'
        self.manifest = {'fixture': 'synthetic'}; atomic_json(self.package / 'manifest.json', self.manifest)
        self.config = runner.protocol.config(); self.prior = fixture()[2]
        atomic_json(self.protocol / 'prior_ledger.json', self.prior)
        for name, value in [('config.json', self.config), ('plan.json', self.plan), ('parents.json', {'data': str(self.data)}), ('manifest.json', {'fixture': True})]:
            atomic_json(self.protocol / name, value)
        self.patches = [patch.object(runner.protocol, 'PROTOCOL', self.protocol),
                        patch.object(runner, 'verify_execution', return_value=(self.manifest, self.config, self.plan, self.data)),
                        patch.object(runner.training.refit, 'load_training_fold', return_value=(self.fold, self.arrays)),
                        patch.object(runner.inference, 'load_test', return_value=self.test)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)

    def execute(self, **kwargs):
        return runner.execute(self.package, self.study, dry_run=False, **kwargs)

    def test_dry_run_caps_explicit_resume_and_complete_cached_audit(self):
        self.assertEqual(runner.execute(self.package, self.study)['status'], 'DRY_RUN')
        self.assertFalse(self.study.exists())
        for cap in (0, -1, True):
            with self.assertRaises(ValueError): self.execute(cap=cap)
        first = self.execute(cap=1); self.assertEqual(first['completed'], 1)
        with self.assertRaisesRegex(ValueError, 'resume'): self.execute(cap=1)
        self.assertEqual(self.execute(cap=3, resume=True)['completed'], 4)
        self.assertEqual(read(self.study / 'shared_ledger.json')['event_count'], 8573)
        before = {r['id']: sha(self.study / 'runs' / r['id'] / 'attempts.json') for r in self.plan}
        with patch.object(runner.training.refit, 'predict_raw', side_effect=AssertionError('no reinference')):
            self.assertEqual(self.execute(resume=True)['executed'], 0)
            report = audit.audit(self.package, self.study, self.root / 'audit')
        self.assertEqual(report['verified'], 4); self.assertEqual(report['failed_attempts'], [])
        self.assertEqual(before, {r['id']: sha(self.study / 'runs' / r['id'] / 'attempts.json') for r in self.plan})

    def test_training_failure_checkpoint_recovery_requires_explicit_retry(self):
        original = runner.training.fit
        def stop(*args, **kwargs):
            original(*args, **kwargs, stop_after=1)
            raise RuntimeError('synthetic stop after committed round')
        with patch.object(runner.training, 'fit', side_effect=stop):
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop'): self.execute()
        with self.assertRaisesRegex(RuntimeError, 'retry'): self.execute(resume=True)
        self.assertEqual(self.execute(resume=True, retry_failed=True)['completed'], 1)
        journal = read(self.study / 'runs' / self.plan[0]['id'] / 'attempts.json')
        self.assertEqual([r['status'] for r in journal], ['FAILED', 'COMPLETE'])
        self.assertIn('synthetic stop', journal[0]['traceback'])

    def test_interrupted_noisy_round_is_retained_once(self):
        self.execute(cap=1); original = runner.training.fit
        def stop(*args, **kwargs):
            original(*args, **kwargs, stop_after=1)
            raise RuntimeError('noise round interruption')
        with patch.object(runner.training, 'fit', side_effect=stop):
            with self.assertRaises(RuntimeError): self.execute(resume=True)
        self.assertEqual(read(self.study / 'shared_ledger.json')['event_count'], 8565)
        self.execute(resume=True, retry_failed=True)
        ledger = read(self.study / 'shared_ledger.json')
        self.assertEqual(ledger['event_count'], 8567)
        self.assertEqual(ledger['events'][:8564], self.prior['events'])

    def test_ambiguous_test_inference_cannot_repeat_even_with_retry(self):
        with patch.object(runner.inference.common, 'scores', side_effect=RuntimeError('after forward')):
            with self.assertRaisesRegex(RuntimeError, 'after forward'): self.execute()
        folder = self.study / 'runs' / self.plan[0]['id']
        self.assertTrue((folder / 'inference_started.json').exists())
        with patch.object(runner.training, 'fit', side_effect=AssertionError('no repeat')):
            with self.assertRaisesRegex(RuntimeError, 'Ambiguous'): self.execute(resume=True, retry_failed=True)

    def test_completed_attempt_interruption_repaired_without_forward(self):
        self.execute(); path = self.study / 'runs' / self.plan[0]['id'] / 'attempts.json'
        journal = read(path); journal[-1]['status'] = 'STARTED'; journal[-1].pop('completion_sha256'); atomic_json(path, journal)
        with patch.object(runner.training.refit, 'predict_raw', side_effect=AssertionError('no repeat')):
            # cap applies to new pending runs; use a truncated completed plan in this synthetic fixture.
            with patch.object(runner, 'verify_execution', return_value=(self.manifest, self.config, self.plan[:1], self.data)):
                self.assertEqual(self.execute(resume=True)['executed'], 0)
        self.assertEqual(read(path)[-1]['status'], 'CACHE_VERIFIED_AFTER_INTERRUPTION')

    def test_missing_journal_and_tampered_completion_rejected(self):
        self.execute(); folder = self.study / 'runs' / self.plan[0]['id']
        path = folder / 'attempts.json'; journal = read(path); path.unlink()
        with self.assertRaisesRegex(ValueError, 'Unjournaled'): self.execute(resume=True)
        atomic_json(path, journal)
        receipt = read(folder / 'completion.json'); receipt['assets']['fit.json'] = 'wrong'
        atomic_json(folder / 'completion.json', receipt)
        with self.assertRaises(ValueError): self.execute(resume=True)

    def test_historical_ledger_removal_and_rewrite_rejected(self):
        self.execute(); path = self.study / 'shared_ledger.json'; original = read(path)
        for mode in ('remove', 'rewrite', 'privacy'):
            bad = copy.deepcopy(original)
            if mode == 'remove': bad['events'].pop(0)
            if mode == 'rewrite': bad['events'][0]['event_id'] = 'rewritten'
            if mode == 'privacy': bad['application_privacy_certified'] = True
            atomic_json(path, bad)
            with self.assertRaises(ValueError): self.execute(resume=True)

    def test_orphan_assets_foreign_binding_and_active_writer_rejected(self):
        self.study.mkdir()
        with writer_lock(self.study):
            with self.assertRaises((RuntimeError, OSError)): self.execute(resume=True)
        atomic_json(self.study / 'binding.json', {'identity': 'foreign'})
        with self.assertRaisesRegex(ValueError, 'binding'): self.execute(resume=True)
        atomic_json(self.study / 'binding.json', {'identity': identity(self.manifest)})
        orphan = self.study / 'runs' / self.plan[0]['id']; orphan.mkdir(parents=True)
        atomic_json(orphan / 'fit.json', {'orphan': True})
        with self.assertRaisesRegex(ValueError, 'Unjournaled'): self.execute(resume=True)

    def test_execution_source_runtime_and_protocol_mutations_rejected(self):
        # Temporarily undo only the execution verifier mock; parent protocol is a synthetic fixture.
        self.patches[1].stop()
        with patch.object(runner.protocol, 'verify', return_value={'status': 'synthetic'}):
            destination = self.root / 'frozen'
            runner.freeze(destination)
            self.assertEqual(runner.verify_execution(destination)[2], self.plan)
            path = destination / 'manifest.json'; original = read(path)
            for key, value in [('planned', 3), ('runtime', {}), ('protocol_manifest_sha256', 'bad')]:
                bad = copy.deepcopy(original); bad[key] = value; atomic_json(path, bad)
                with self.assertRaises(ValueError): runner.verify_execution(destination)
            atomic_json(path, original)
            (destination / 'source_snapshot' / 'matched_noise_training.py').write_text('bad fixture', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source'): runner.verify_execution(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests))
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
