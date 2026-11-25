"""Synthetic R14 training/resume smoke and unchanged mechanism dependencies."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper2.restart import matched_noise_protocol as protocol
from paper2.restart import matched_noise_training as training
from paper2.restart import test_privacy_accounting, test_private_primitives
from paper2.restart.test_matched_noise_protocol import fixture
from paper2.restart.validation_search import read, sha, atomic_json


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r14_training_test_', dir=protocol.RESULTS)
        self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        parents, slots, _ = fixture()
        self.runs = protocol.make_plan(parents, slots)[:4]
        for run in self.runs: run['rounds'] = 3
        self.run = self.runs[1]
        self.x = np.random.default_rng(9).normal(size=(14, 2)).astype(np.float32)
        self.y = np.array([1, 2] * 7, dtype=np.float32)
        self.fold = {'id': 'outer_0', 'kind': 'outer_refit', 'validation': [], 'train': list(range(14)),
                     'test': [14, 15], 'preprocessing_identity': 'synthetic',
                     'clients': {s: [i * 2, i * 2 + 1] for i, s in enumerate(self.run['public_slots'])}}

    def fit(self, folder, run=None, stop_after=None):
        return training.fit(self.root / folder, run or self.run, protocol.config(), self.fold, self.x, self.y,
                            'synthetic-token', stop_after=stop_after)

    def test_all_variants_smoke_exact_events_and_work(self):
        for run in self.runs:
            self.fit(run['variant'], run)
            checkpoint = torch.load(self.root / run['variant'] / 'checkpoint.pt', weights_only=True)
            self.assertEqual(checkpoint['done'], 3)
            self.assertEqual(checkpoint['events'], protocol.planned_events(run))
            self.assertTrue(all(h['local_steps'] == 28 for h in checkpoint['history']))
            self.assertFalse(read(self.root / run['variant'] / 'fit.json')['application_privacy_certified'])

    def test_exact_resume_every_variant_and_no_cache_retraining(self):
        for run in self.runs:
            name = run['variant']; self.fit(name + '_full', run)
            self.assertIsNone(self.fit(name + '_resumed', run, stop_after=1))
            self.fit(name + '_resumed', run)
            full = torch.load(self.root / (name + '_full') / 'checkpoint.pt', weights_only=True)
            resumed = torch.load(self.root / (name + '_resumed') / 'checkpoint.pt', weights_only=True)
            for key in ('signature', 'done', 'history', 'events'): self.assertEqual(full[key], resumed[key])
            for key in full['state']: self.assertTrue(torch.equal(full['state'][key], resumed['state'][key]))
            before = sha(self.root / (name + '_resumed') / 'checkpoint.pt')
            with (patch.object(training.refit, 'train_regression_client', side_effect=AssertionError('no retraining')),
                  patch.object(training.refit, 'predict_raw', side_effect=AssertionError('no reinference'))):
                self.fit(name + '_resumed', run)
            self.assertEqual(sha(self.root / (name + '_resumed') / 'checkpoint.pt'), before)

    def test_above_legacy_round_cap_is_supported(self):
        run = dict(self.run, rounds=81)
        self.fit('long', run)
        checkpoint = torch.load(self.root / 'long/checkpoint.pt', weights_only=True)
        self.assertEqual(checkpoint['done'], 81)
        self.assertEqual(len(checkpoint['events']), 81)

    def test_checkpoint_identity_dtype_state_metrics_and_events_rejected(self):
        self.fit('partial', stop_after=1)
        path = self.root / 'partial/checkpoint.pt'; original = torch.load(path, weights_only=True)
        for mode in ('identity', 'dtype', 'shape', 'nan', 'state', 'events', 'steps'):
            bad = copy.deepcopy(original)
            if mode == 'identity': bad['signature'] = 'bad'
            if mode == 'dtype': bad['state']['0.weight'] = bad['state']['0.weight'].double()
            if mode == 'shape': bad['state']['0.weight'] = torch.zeros(32, 3)
            if mode == 'nan': bad['state']['0.weight'][0, 0] = float('nan')
            if mode == 'state': bad['state']['2.bias'].add_(100)
            if mode == 'events': bad['events'] = []
            if mode == 'steps': bad['history'][0]['local_steps'] = 29
            torch.save(bad, path)
            with self.assertRaises(ValueError): self.fit('partial')
        torch.save(original, path); self.fit('partial')

    def test_changed_training_input_or_roster_rejected(self):
        self.fit('partial', stop_after=1)
        self.y[0] += 1
        with self.assertRaisesRegex(ValueError, 'identity'): self.fit('partial')
        self.y[0] -= 1; self.fold['clients'][self.run['public_slots'][0]] = [14]
        with self.assertRaisesRegex(ValueError, 'partition'): self.fit('bad')

    def test_production_and_retuned_settings_rejected_before_training(self):
        settings = protocol.config(); settings['simulation_only'] = False
        with self.assertRaises(ValueError):
            training.fit(self.root / 'private', self.run, settings, self.fold, self.x, self.y, 't')
        for field, value in [('clip_norm', .4), ('target_scale', 10), ('noise_multiplier', 1), ('rounds', 241), ('seed', 42)]:
            with self.assertRaises(ValueError): self.fit('bad', dict(self.run, **{field: value}))
        self.assertFalse((self.root / 'private').exists())

    def test_noise_stream_separation_and_determinism(self):
        a = training.noise_rng(self.run, 1).normal(size=10)
        np.testing.assert_array_equal(a, training.noise_rng(self.run, 1).normal(size=10))
        self.assertFalse(np.array_equal(a, training.noise_rng(self.run, 2).normal(size=10)))
        self.assertFalse(np.array_equal(a, training.noise_rng(self.runs[2], 1).normal(size=10)))

    def test_final_accounting_and_model_changes_rejected(self):
        self.fit('complete'); folder = self.root / 'complete'
        original = read(folder / 'accounting.json'); bad = copy.deepcopy(original); bad['events'].pop()
        atomic_json(folder / 'accounting.json', bad)
        with self.assertRaisesRegex(ValueError, 'receipt'): self.fit('complete')
        atomic_json(folder / 'accounting.json', original)
        saved = torch.load(folder / 'model.pt', weights_only=True); saved['2.bias'].add_(.1)
        torch.save(saved, folder / 'model.pt')
        with self.assertRaisesRegex(ValueError, 'receipt'): self.fit('complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO(); loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([loader.loadTestsFromTestCase(TrainingTests),
                               loader.loadTestsFromModule(test_privacy_accounting), loader.loadTestsFromModule(test_private_primitives)])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
