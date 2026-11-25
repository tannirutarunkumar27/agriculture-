"""Full-coverage and support-panel regression; old aggregation remains unchanged."""
import argparse
import copy
import io
from pathlib import Path
import unittest

import numpy as np

from paper2.restart.summarize_public_refit import summarize_records
from paper2.restart.test_outer_summary import SummaryTests as ExistingSummaryTests
from paper2.restart.validation_search import atomic_json


class PublicSummaryTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for seed in (1, 2):
            for group, ids, error, seen in (('small', [0], 1., True), ('large', [1, 2, 3], 3., False)):
                self.records.append({'seed': seed, 'group': group, 'ids': np.array(ids), 'y': np.zeros(len(ids)),
                    'p': np.full(len(ids), error), 'support': [{'all_species_seen': seen, 'composition_seen': True} for _ in ids],
                    'training_fit': {'worse_than_matched_constant': False}})

    def summarize(self, rows=None):
        return summarize_records(self.records if rows is None else rows, [1, 2], ['small', 'large'], range(4))

    def test_equal_group_not_pooled_and_seed_sd_descriptive(self):
        result = self.summarize(); metrics = result['summary'][0]['metrics']
        self.assertEqual(metrics['equal_group_rmse']['mean'], 2.)
        self.assertAlmostEqual(metrics['pooled_rmse']['mean'], np.sqrt(7))
        self.assertEqual(metrics['equal_group_rmse']['seed_sd'], 0.)
        self.assertEqual(result['unique_systems_per_seed'], 4)
        self.assertIsNone(metrics['pooled_r2'])

    def test_incomplete_or_duplicate_matrix_rejected(self):
        for rows in (self.records[:-1], self.records + [self.records[0]]):
            with self.assertRaisesRegex(ValueError, 'matrix'): self.summarize(rows)

    def test_overlap_and_missing_system_coverage_rejected(self):
        rows = copy.deepcopy(self.records); rows[1]['ids'][0] = 0
        with self.assertRaisesRegex(ValueError, 'coverage'): self.summarize(rows)

    def test_seen_unseen_and_empty_panels(self):
        result = self.summarize(); panels = {r['panel']: r for r in result['summary']}
        self.assertEqual(panels['species_seen']['samples_per_seed'], 1)
        self.assertEqual(panels['species_unseen']['samples_per_seed'], 3)
        self.assertEqual(panels['species_seen']['metrics']['equal_group_rmse']['mean'], 1.)
        self.assertEqual(panels['species_unseen']['metrics']['equal_group_rmse']['mean'], 3.)
        self.assertIsNone(panels['composition_unseen']['metrics'])

    def test_support_and_nonfinite_outputs_rejected(self):
        rows = copy.deepcopy(self.records); rows[0]['support'][0]['composition_seen'] = 'true'
        with self.assertRaisesRegex(ValueError, 'Boolean'): self.summarize(rows)
        rows = copy.deepcopy(self.records); rows[0]['p'][0] = np.nan
        with self.assertRaises(ValueError): self.summarize(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (ExistingSummaryTests, PublicSummaryTests))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
