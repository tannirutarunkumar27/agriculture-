"""Synthetic software fixtures ONLY; bounds/approvals are not agronomic evidence."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch

from paper2.federated import aggregate, clone_state, train_client
from paper2.knowledge import KnowledgeGuidedMLP, RuleBank
from paper2.knowledge_training import (guidance_schema, load_guided_weights,
                                      save_guided_weights, train_knowledge_client)


def fixture():
    contract = {"dataset": "unit_test_only", "source_sha256": "1"*64,
                "features": [{"name": "toy", "unit": "toy_units", "measurement": "toy",
                              "period": "toy", "verified": True, "evidence": "synthetic test declaration"}]}
    rule = {"id": "synthetic_rule", "crop": "a", "dataset": "unit_test_only", "feature": "toy",
            "unit": "toy_units", "measurement": "toy", "period": "toy", "bounds": [0, 2, 4, 6], "weight": 1,
            "source": {"url": "https://example.org/synthetic-test-only", "version": "test",
                       "locator": "test", "scientific_name": "toy", "bounds_evidence": "synthetic test",
                       "artifact_sha256": "2"*64, "independent_of_labels": True},
            "scope": {"region": "toy_region", "season": "*", "stage": "*", "water_regime": "*"},
            "approval": {"status": "approved", "reviewer": "UNIT_TEST_ONLY", "checked_date": "2026-10-13",
                         "rationale": "test only", "exceptions": "test only", "scope_evidence": "test only"}}
    return {"schema_version": 1, "rules": [rule]}, contract


class TestKnowledge(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.registry, self.contract = fixture()
        self.bank = RuleBank(self.registry, self.contract, ["a", "b"])
        self.context = {"region": "toy_region"}

    def score(self, values, mask=None, contexts=None, bank=None):
        raw = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
        observed = torch.ones_like(raw, dtype=torch.bool) if mask is None else torch.tensor(mask).reshape_as(raw)
        return (bank or self.bank).suitability(raw, observed, contexts or [self.context]*len(raw))

    def test_trapezoid_and_neutral_unruled_class(self):
        q, covered = self.score([-1, 0, 1, 2, 3, 4, 5, 6, 7])
        torch.testing.assert_close(q[:, 0], torch.tensor([0., 0., .5, 1., 1., 1., .5, 0., 0.]))
        self.assertTrue(covered[:, 0].all())
        self.assertFalse(covered[:, 1].any())
        self.assertTrue((q[:, 1] == .5).all())

    def test_missing_and_out_of_scope_neutral(self):
        for context in ({}, {"region": "different"}):
            q, mask = self.score([3], contexts=[context])
            self.assertTrue((q == .5).all())
            self.assertFalse(mask.any())
        q, mask = self.score([float("nan")], [False])
        self.assertTrue((q == .5).all())
        self.assertFalse(mask.any())

    def test_imputed_value_does_not_create_knowledge(self):
        q, covered = self.score([3], [False])
        self.assertFalse(covered.any())
        self.assertTrue((q == .5).all())

    def test_class_permutation_is_explicit(self):
        bank = RuleBank(self.registry, self.contract, ["b", "a"])
        q, _ = self.score([3], bank=bank)
        torch.testing.assert_close(q, torch.tensor([[.5, 1.]]))
        self.assertNotEqual(bank.identity, self.bank.identity)

    def test_weighted_rules(self):
        extra = copy.deepcopy(self.registry["rules"][0])
        extra.update(id="other", bounds=[10, 12, 14, 16], weight=.5)
        self.registry["rules"].append(extra)
        q, _ = self.score([3], bank=RuleBank(self.registry, self.contract, ["a", "b"]))
        self.assertAlmostEqual(float(q[0, 0]), 2/3)

    def test_metadata_cannot_mutate_bank(self):
        self.registry["rules"][0]["bounds"][0] = -999
        data = self.bank.metadata()
        data["rules"][0]["crop"] = "b"
        self.assertEqual(self.bank.metadata()["rules"][0]["crop"], "a")
        self.assertEqual(self.bank.metadata()["rules"][0]["bounds"][0], 0)

    def test_evidence_and_scope_rejections(self):
        mutations = [lambda r, c: r["rules"][0].update(crop="unknown"),
                     lambda r, c: r["rules"][0].update(unit="wrong"),
                     lambda r, c: r["rules"][0].update(period="annual"),
                     lambda r, c: c["features"][0].update(verified=False),
                     lambda r, c: r["rules"][0]["source"].update(independent_of_labels=False),
                     lambda r, c: r["rules"][0]["source"].update(artifact_sha256=""),
                     lambda r, c: r["rules"][0]["approval"].update(status="pending"),
                     lambda r, c: r["rules"][0]["approval"].update(checked_date="bad"),
                     lambda r, c: r["rules"][0].update(bounds=[0, 0, 1, 2]),
                     lambda r, c: r["rules"][0].update(weight=0),
                     lambda r, c: r["rules"][0].update(scope={}),
                     lambda r, c: r["rules"].append(copy.deepcopy(r["rules"][0]))]
        for mutate in mutations:
            r, c = fixture()
            mutate(r, c)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                RuleBank(r, c, ["a", "b"])

    def test_bad_inputs_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.score([value])
        with self.assertRaises(ValueError):
            self.bank.suitability(torch.ones(2, 1), torch.ones(2, 1), [self.context]*2)
        with self.assertRaises(ValueError):
            self.score([3], contexts=[{"region": "toy_region", "unknown": "x"}])

    def test_no_rules_gate(self):
        bank = RuleBank({"schema_version": 1, "rules": []}, self.contract, ["a", "b"])
        with self.assertRaises(ValueError):
            KnowledgeGuidedMLP(1, 4, bank, 1)
        model = KnowledgeGuidedMLP(1, 4, bank, 0)
        self.assertFalse(model.manifest()["guidance_enabled"])

    def test_alpha_zero_and_class_aligned_bias(self):
        model = KnowledgeGuidedMLP(1, 4, self.bank, 2)
        x = torch.tensor([[3.]])
        observed = torch.ones_like(x, dtype=torch.bool)
        base = model.model(x)
        torch.testing.assert_close(model(x, x, observed, [self.context])-base, torch.tensor([[1., 0.]]))
        control = KnowledgeGuidedMLP(1, 4, self.bank, 0)
        torch.testing.assert_close(control(x, x, observed, [self.context]), control.model(x))

    def test_alpha_zero_local_training_matches_baseline(self):
        model = KnowledgeGuidedMLP(1, 4, self.bank, 0)
        x = torch.arange(8, dtype=torch.float32).reshape(-1, 1)/4
        y = torch.arange(8)%2
        contexts = [self.context]*8
        mask = torch.ones_like(x, dtype=torch.bool)
        guided = train_knowledge_client(model, x, x, mask, contexts, y, "a", 1, "prep", 123, batch_size=3)
        baseline = train_client(model.model, x, y, "a", 1, "prep", 123, batch_size=3)
        self.assertEqual(guided.steps, baseline.steps)
        for key, value in baseline.state.items():
            torch.testing.assert_close(guided.state["model."+key], value, rtol=0, atol=0)

    def test_guided_gradients_determinism_and_aggregation(self):
        torch.manual_seed(123)
        model = KnowledgeGuidedMLP(1, 4, self.bank, 1)
        original = clone_state(model)
        x = torch.tensor([[0.], [1.], [2.], [3.], [4.]])
        y = torch.tensor([0, 1, 0, 1, 0])
        mask = torch.ones_like(x, dtype=torch.bool)
        updates = [train_knowledge_client(model, x[:n], x[:n], mask[:n], [self.context]*n,
                                         y[:n], name, 1, "prep", 123, batch_size=2, mu=.01)
                   for name, n in [("a", 3), ("b", 5)]]
        repeat = train_knowledge_client(model, x[:3], x[:3], mask[:3], [self.context]*3,
                                       y[:3], "a", 1, "prep", 123, batch_size=2, mu=.01)
        for key in repeat.state:
            torch.testing.assert_close(repeat.state[key], updates[0].state[key], rtol=0, atol=0)
            torch.testing.assert_close(model.state_dict()[key], original[key], rtol=0, atol=0)
        for algorithm in ("fedavg", "fedprox", "fednova"):
            state = aggregate(original, updates, algorithm, 1, guidance_schema(model, "prep"), {"a": 3, "b": 5})
            self.assertTrue(all(torch.isfinite(v).all() for v in state.values()))
        changed = KnowledgeGuidedMLP(1, 4, self.bank, 2)
        with self.assertRaises(ValueError):
            aggregate(original, updates, "fedavg", 1, guidance_schema(changed, "prep"), {"a": 3, "b": 5})

    def test_weights_roundtrip_and_identity_gate(self):
        model = KnowledgeGuidedMLP(1, 4, self.bank, 1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"weights"
            save_guided_weights(model, folder, "prep")
            other = KnowledgeGuidedMLP(1, 4, self.bank, 1)
            load_guided_weights(other, folder, "prep")
            for key in model.state_dict():
                torch.testing.assert_close(model.state_dict()[key], other.state_dict()[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_guided_weights(other, folder, "changed_prep")
            changed = KnowledgeGuidedMLP(1, 4, self.bank, 2)
            with self.assertRaises(ValueError):
                load_guided_weights(changed, folder, "prep")
            with self.assertRaises(FileExistsError):
                save_guided_weights(model, folder, "prep")
            with (folder/"weights.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                load_guided_weights(other, folder, "prep")

    def test_guidance_changes_supervised_update(self):
        torch.manual_seed(1)
        guided = KnowledgeGuidedMLP(1, 4, self.bank, 1)
        control = KnowledgeGuidedMLP(1, 4, self.bank, 0)
        control.load_state_dict(guided.state_dict())
        x = torch.tensor([[3.], [3.]])
        mask = torch.ones_like(x, dtype=torch.bool)
        y = torch.tensor([0, 1])
        a = train_knowledge_client(guided, x, x, mask, [self.context]*2, y, "a", 1, "prep", 1)
        b = train_knowledge_client(control, x, x, mask, [self.context]*2, y, "a", 1, "prep", 1)
        self.assertTrue(any(not torch.equal(a.state[k], b.state[k]) for k in a.state))

    def test_two_round_federated_smoke_and_saved_predictions(self):
        # Synthetic engineering smoke only; no research accuracy reported.
        torch.manual_seed(12)
        initial = KnowledgeGuidedMLP(1, 4, self.bank, 1)
        x = torch.tensor([[0.], [1.], [2.], [3.], [4.]])
        y = torch.tensor([0, 1, 0, 1, 0])
        mask = torch.ones_like(x, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as tmp:
            for algorithm in ("fedavg", "fedprox", "fednova"):
                model = copy.deepcopy(initial)
                schema = guidance_schema(model, "prep")
                for round_number in (1, 2):
                    updates = [train_knowledge_client(model, x[:n], x[:n], mask[:n], [self.context]*n,
                                                      y[:n], name, round_number, "prep", 42+round_number,
                                                      batch_size=2, mu=.01 if algorithm == "fedprox" else 0)
                               for name, n in [("a", 3), ("b", 5)]]
                    model.load_state_dict(aggregate(clone_state(model), updates, algorithm,
                                                    round_number, schema, {"a": 3, "b": 5}))
                before = model(x, x, mask, [self.context]*5).detach()
                folder = Path(tmp)/algorithm
                save_guided_weights(model, folder, "prep")
                restored = copy.deepcopy(initial)
                load_guided_weights(restored, folder, "prep")
                torch.testing.assert_close(before, restored(x, x, mask, [self.context]*5), rtol=0, atol=0)

    def test_local_invalid_inputs_and_mu_zero_equivalence(self):
        model = KnowledgeGuidedMLP(1, 4, self.bank, 1)
        x = torch.ones(2, 1)
        mask = torch.ones_like(x, dtype=torch.bool)
        y = torch.tensor([0, 1])
        args = (model, x, x, mask, [self.context]*2, y, "a", 1, "prep", 1)
        a = train_knowledge_client(*args)
        b = train_knowledge_client(*args, mu=0)
        for key in a.state:
            torch.testing.assert_close(a.state[key], b.state[key], rtol=0, atol=0)
        for kwargs in ({"mu": -1}, {"learning_rate": float("nan")}, {"batch_size": 0}):
            with self.assertRaises(ValueError):
                train_knowledge_client(*args, **kwargs)
        with self.assertRaises(ValueError):
            train_knowledge_client(model, x, x, mask, [self.context]*2, torch.tensor([0, 2]), "a", 1, "prep", 1)
