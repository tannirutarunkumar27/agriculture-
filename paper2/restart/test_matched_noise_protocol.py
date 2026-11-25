"""Synthetic prospective-plan, historical-accounting and freeze mutation tests."""
import argparse
import copy
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper2.restart import matched_noise_protocol as protocol
from paper2.restart.validation_search import atomic_json, read, sha


def fixture():
    groups = [f'group{i}' for i in range(8)]
    rounds = [133, 129, 114, 149, 106, 109, 106, 128]
    parents = [{'id': f'outer_{i}_equal_public_seed{s}', 'fold': f'outer_{i}',
                'seed': s, 'method': 'fixed_public_equal_client_control',
                'target_scale': 1., 'learning_rate': .01, 'rounds': rounds[i]}
               for i in range(8) for s in protocol.refit.protocol.SEEDS]
    slots = {f'outer_{i}': [g for g in groups if g != groups[i]] for i in range(8)}
    events = [{'event_id': f'old:{i}', 'slots': slots[f'outer_{i % 8}'],
               'event': {'sensitivity': .1, 'noise_std': .4, 'count': 1}} for i in range(8564)]
    ledger = {'event_count': len(events), 'events': events, 'per_slot': protocol.per_slot(events),
              'mode': 'SIMULATION_ACCOUNTING', 'excludes_unnoised_controls': True,
              'application_privacy_certified': False}
    return parents, slots, ledger


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parents, self.slots, self.ledger = fixture()
        self.plan = protocol.make_plan(self.parents, self.slots)

    def test_matched_matrix_carries_every_parent_setting(self):
        self.assertEqual(len(self.plan), 160)
        self.assertEqual(len({r['id'] for r in self.plan}), 160)
        for parent in self.parents:
            matches = [r for r in self.plan if r['reference_id'] == parent['id']]
            self.assertEqual(len(matches), 4)
            for r in matches:
                for key in ('fold', 'seed', 'rounds', 'target_scale', 'learning_rate'):
                    self.assertEqual(r[key], parent[key])
                self.assertEqual(r['clip_norm'], .04)
        self.assertEqual(sum(r['noise_multiplier'] is not None for r in self.plan), 120)

    def test_incomplete_duplicate_or_retuned_references_rejected(self):
        for rows in (self.parents[:-1], self.parents + [self.parents[0]], [self.parents[0]] * 40):
            with self.assertRaises(ValueError): protocol.make_plan(rows, self.slots)
        for key, value in [('target_scale', 10), ('learning_rate', .003), ('rounds', 241), ('seed', True)]:
            bad = copy.deepcopy(self.parents); bad[0][key] = value
            with self.assertRaises(ValueError): protocol.make_plan(bad, self.slots)

    def test_roster_duplicates_and_wrong_fold_exclusion_rejected(self):
        for mode in ('duplicate', 'same', 'missing'):
            slots = copy.deepcopy(self.slots)
            if mode == 'duplicate': slots['outer_0'][0] = slots['outer_0'][1]
            if mode == 'same': slots['outer_0'] = slots['outer_1']
            if mode == 'missing': slots.pop('outer_0')
            with self.assertRaises(ValueError): protocol.make_plan(self.parents, slots)

    def test_accounting_independent_formula_and_historical_rows_preserved(self):
        original = copy.deepcopy(self.ledger)
        result = protocol.accounting(self.plan, self.ledger)
        self.assertEqual(self.ledger, original)
        self.assertEqual(result['new_events'], 14610)
        self.assertEqual(result['combined_events'], 23174)
        for row in result['per_slot']:
            old_count = sum(row['slot'] in r['slots'] for r in self.ledger['events'])
            rho = old_count / 32 + sum(r['rounds'] / (2 * r['noise_multiplier'] ** 2)
                    for r in self.plan if r['noise_multiplier'] is not None and row['slot'] in r['public_slots'])
            self.assertAlmostEqual(row['rho'], rho)
            self.assertAlmostEqual(row['epsilon_upper_bound'], rho + 2 * math.sqrt(rho * math.log(1e5)))
        self.assertEqual(sum(r['conditional_budget'] is None for r in result['per_run']), 40)
        self.assertFalse(result['study_dp_claim_allowed'])

    def test_noise_units_round_counts_and_stream_namespace(self):
        for run in self.plan[:4]:
            events = protocol.planned_events(run)
            if run['noise_multiplier'] is None:
                self.assertEqual(events, []); continue
            self.assertEqual(len(events), 133)
            self.assertEqual(len({e['event_id'] for e in events}), 133)
            self.assertTrue(all(e['event_id'].startswith(protocol.NAMESPACE) for e in events))
            self.assertAlmostEqual(events[0]['event']['sensitivity'], .08 / 7)
            self.assertAlmostEqual(events[0]['event']['noise_std'], .08 / 7 * run['noise_multiplier'])

    def test_ledger_tampering_duplicate_bad_schema_and_collisions_rejected(self):
        for mode in ('removed', 'duplicate', 'schema', 'count', 'slot', 'budget', 'collision'):
            bad = copy.deepcopy(self.ledger)
            if mode == 'removed': bad['events'].pop()
            if mode == 'duplicate': bad['events'][1] = bad['events'][0]
            if mode == 'schema': bad['events'][0]['event']['subsampling'] = .1
            if mode == 'count': bad['events'][0]['event']['count'] = 2
            if mode == 'slot': bad['events'][0]['slots'][0] = bad['events'][0]['slots'][1]
            if mode == 'budget': bad['per_slot'][0]['rho'] = 0.
            if mode == 'collision': bad['events'][0]['event_id'] = protocol.NAMESPACE + 'x'
            with self.assertRaises(ValueError): protocol.check_prior(bad)

    def test_production_privacy_and_release_policy_mutation_rejected(self):
        protocol.require_simulation(protocol.config())
        for field in ('application_privacy_certified', 'study_dp_claim_allowed', 'nonprivate_selection_covered',
                      'private_metric_release_covered', 'unnoised_control_release_covered', 'noise_seed_secrecy'):
            value = protocol.config(); value[field] = True
            with self.assertRaises(ValueError): protocol.require_simulation(value)
        value = protocol.config(); value['simulation_only'] = False
        with self.assertRaises(ValueError): protocol.require_simulation(value)

    def test_package_mutations_and_complete_semantic_recomputation(self):
        with tempfile.TemporaryDirectory(prefix='r14_test_', dir=protocol.RESULTS) as tmp:
            output = Path(tmp) / 'protocol'
            supplied = (self.parents, self.slots, self.ledger, [{'test_fixture': True}], {'test_fixture': True})
            with patch.object(protocol, 'collect_parents', return_value=supplied):
                protocol.freeze(output)
                self.assertEqual(protocol.verify(output)['planned'], 160)
                with self.assertRaises(FileExistsError): protocol.freeze(output)
                manifest = read(output / 'manifest.json')
                original = read(output / 'plan.json'); bad = copy.deepcopy(original); bad[0]['rounds'] += 1
                atomic_json(output / 'plan.json', bad)
                with self.assertRaisesRegex(ValueError, 'artifact'): protocol.verify(output)
                changed = copy.deepcopy(manifest); changed['artifacts']['plan.json'] = sha(output / 'plan.json')
                atomic_json(output / 'manifest.json', changed)
                with self.assertRaisesRegex(ValueError, 'semantic'): protocol.verify(output)
                atomic_json(output / 'plan.json', original); atomic_json(output / 'manifest.json', manifest)
                snap = output / 'source_snapshot' / protocol.CODE[0].name
                snap.write_text('corrupted synthetic snapshot', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'source'): protocol.verify(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output / 'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    atomic_json(args.output / 'verification.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
