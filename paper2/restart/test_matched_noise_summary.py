"""Matched comparison tests; existing aggregation suite retained unchanged."""
import argparse
import copy
import io
from pathlib import Path
import unittest
import numpy as np
from paper2.restart import summarize_matched_noise as summary
from paper2.restart.test_public_refit_summary import ExistingSummaryTests, PublicSummaryTests
from paper2.restart.validation_search import atomic_json


class MatchedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for variant_index, variant in enumerate(summary.VARIANTS):
            for seed in (1, 2):
                for group, ids in [('small', [0]), ('large', [1, 2, 3])]:
                    self.records.append({'variant': variant, 'seed': seed, 'group': group,
                        'ids': np.array(ids), 'y': np.zeros(len(ids)),
                        'p': np.full(len(ids), (1. if group == 'small' else 3.) + variant_index),
                        'support': [{'all_species_seen': group == 'small', 'composition_seen': True} for _ in ids],
                        'training_fit': {'worse_than_matched_constant': False}})

    def summarize(self, rows=None):
        return summary.summarize_variants(self.records if rows is None else rows, [1, 2], ['small', 'large'], range(4))

    def test_known_matched_differences_equal_group_and_pooled(self):
        result = self.summarize(); paired = result['paired_differences_from_cached_reference'][0]
        self.assertEqual(paired['all_equal_group_rmse_difference'], {'mean': 1., 'seed_sd': 0.})
        self.assertAlmostEqual(paired['all_pooled_rmse_difference']['mean'], np.sqrt(13) - np.sqrt(7))
        self.assertIsNone(result['variants'][summary.REFERENCE]['summary'][-1]['metrics'])

    def test_incomplete_or_duplicate_variant_matrix_rejected(self):
        for rows in (self.records[:-1], self.records + [self.records[0]], [r for r in self.records if r['variant'] != 'noisy_sigma8']):
            with self.assertRaises(ValueError): self.summarize(rows)

    def test_cross_variant_target_and_support_mismatch_rejected(self):
        for mode in ('targets', 'support'):
            rows = copy.deepcopy(self.records)
            if mode == 'targets': rows[-1]['y'][0] = 1
            else: rows[-1]['support'][0]['all_species_seen'] = True
            with self.assertRaises(ValueError): self.summarize(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in
                               (MatchedSummaryTests, ExistingSummaryTests, PublicSummaryTests))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
