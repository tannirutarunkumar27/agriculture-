"""Plot extraction guards and rendering smoke; never changes source analysis."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paper2.restart import publication_figures as figures
from paper2.restart.validation_search import ROOT, read, atomic_json


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.analysis = read(ROOT / 'paper2/results/restart_public_refit_analysis_v1/analysis.json')

    def test_exact_seed_values_and_support_preserved(self):
        inputs = figures.extract(self.analysis, 'refit'); row = inputs['variants'][0]
        self.assertEqual(row['overview'][0]['samples'], 607)
        self.assertEqual(row['overview'][0]['groups'], 8)
        self.assertEqual(row['overview'][0]['mean'], self.analysis['summary'][0]['metrics']['equal_group_rmse']['mean'])
        unseen = next(r for r in row['support'] if r['panel'] == 'species_unseen')
        self.assertEqual((unseen['samples'], unseen['groups']), (265, 1))

    def test_partial_or_inconsistent_analysis_rejected(self):
        for mode in ('partial', 'seeds', 'mean', 'sd', 'groups'):
            bad = copy.deepcopy(self.analysis)
            if mode == 'partial': bad['status'] = 'PARTIAL'
            if mode == 'seeds': bad['study_seeds'] = 4
            if mode == 'mean': bad['summary'][0]['metrics']['equal_group_rmse']['mean'] += 1
            if mode == 'sd': bad['summary'][0]['metrics']['equal_group_rmse']['seed_sd'] = -1
            if mode == 'groups': bad['by_group'][0]['samples_per_seed'] += 1
            with self.assertRaises(ValueError): figures.extract(bad, 'refit')

    def test_matched_scope_and_inventory_rejected(self):
        fixture = {'status': 'VERIFIED_COMPLETE_MATCHED_ANALYSIS', 'application_privacy_certified': False,
                   'variants': {name: copy.deepcopy(self.analysis) for name in figures.LABELS}}
        self.assertEqual(len(figures.extract(fixture, 'matched')['variants']), 5)
        fixture['application_privacy_certified'] = True
        with self.assertRaises(ValueError): figures.extract(fixture, 'matched')
        fixture['application_privacy_certified'] = False; fixture['variants'].pop('noisy_sigma8')
        with self.assertRaises(ValueError): figures.extract(fixture, 'matched')

    def test_three_formats_render_without_figure_leaks(self):
        inputs = figures.extract(self.analysis, 'refit')
        with tempfile.TemporaryDirectory(prefix='plot_test_', dir=ROOT / 'paper2/results') as temp:
            names, captions = figures.draw(inputs, temp)
            self.assertEqual(len(names), 9); self.assertEqual(len(captions), 3)
            self.assertTrue(all((Path(temp) / n).stat().st_size > 500 for n in names))
            self.assertFalse(figures.plt.get_fignums())

    def test_all_seed_markers_visible_and_panel_scales_match(self):
        inputs = figures.extract(self.analysis, 'refit'); captured = []
        with patch.object(figures.plt.Figure, 'savefig', lambda fig, *a, **kw: captured.append(fig)):
            figures.draw(inputs, ROOT / 'paper2/results/unused_synthetic_render_path')
        overview, support = captured[0], captured[6]
        self.assertEqual(overview.axes[0].get_xlim(), overview.axes[1].get_xlim())
        for ax, row in zip(overview.axes, inputs['variants'][0]['overview']):
            lo, hi = ax.get_xlim()
            self.assertTrue(all(lo < v < hi for v in row['seed_values']))
            bottom, top = sorted(ax.get_ylim())
            self.assertLess(bottom, -.1); self.assertGreater(top, .1)
        self.assertEqual(len({ax.get_ylim() for ax in support.axes}), 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FigureTests))
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
