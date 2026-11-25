import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from paper2.data import file_hash, load_dataset, make_partition
from paper2.federated import Update, aggregate, clone_state, make_model, proximal_loss, state_hash, train_client
from paper2.train import checked_config, run


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.base = {"w": torch.tensor([10.0])}
        self.updates = [Update("a", 1, "schema", state_hash(self.base), 1, 1, {"w": torch.tensor([8.0])}),
                        Update("b", 1, "schema", state_hash(self.base), 3, 2, {"w": torch.tensor([6.0])})]

    def aggregate(self, algorithm="fedavg", updates=None):
        return aggregate(self.base, updates or self.updates, algorithm, 1, "schema", {"a":1, "b":3})

    def test_sample_weighted_fedavg(self):
        self.assertEqual(float(self.aggregate()["w"]), 6.5)

    def test_unequal_step_fednova(self):
        # p=(.25,.75), normalized deltas=(2,2), tau_eff=1.75 => 10-3.5.
        self.assertEqual(float(self.aggregate("fednova")["w"]), 6.5)
        self.updates[1].state["w"] = torch.tensor([4.0])
        self.assertAlmostEqual(float(self.aggregate("fednova")["w"]), 5.1875)
        self.assertNotEqual(float(self.aggregate()["w"]), 5.1875)

    def test_equal_steps_fednova_matches_fedavg(self):
        self.updates[0].steps = self.updates[1].steps = 3
        torch.testing.assert_close(self.aggregate("fednova")["w"], self.aggregate()["w"])

    def test_reject_malformed_updates(self):
        edits = [lambda u: setattr(u, "round", 0),
                 lambda u: setattr(u, "schema_hash", "wrong"),
                 lambda u: setattr(u, "base_hash", "wrong"),
                 lambda u: setattr(u, "samples", 99),
                 lambda u: setattr(u, "steps", 0),
                 lambda u: u.state.update(w=torch.tensor([float("nan")])),
                 lambda u: u.state.update(w=torch.tensor([1.0, 2.0])),
                 lambda u: u.state.update(w=torch.tensor([1.0], dtype=torch.float64))]
        for edit in edits:
            with self.subTest(edit=edit):
                updates = copy.deepcopy(self.updates)
                edit(updates[0])
                with self.assertRaises(ValueError):
                    self.aggregate(updates=updates)

    def test_reject_duplicate_and_missing_clients(self):
        for updates in [[self.updates[0]], [self.updates[0], self.updates[0]]]:
            with self.assertRaises(ValueError):
                self.aggregate(updates=updates)


class TestLocalTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.model = make_model(2, 2, 8)
        self.x = torch.randn(12,2)
        self.y = torch.arange(12) % 2

    def train(self, mu=0):
        return train_client(self.model, self.x, self.y, "a", 1, "schema", 42,
                            epochs=2, batch_size=5, learning_rate=0.1, mu=mu)

    def test_local_training_actual_steps_and_no_global_mutation(self):
        before = state_hash(clone_state(self.model))
        update = self.train()
        self.assertEqual(update.steps, 6)
        self.assertEqual(update.samples, 12)
        self.assertEqual(before, state_hash(clone_state(self.model)))
        self.assertNotEqual(before, state_hash(update.state))

    def test_mu_zero_equivalence_and_determinism(self):
        a, b = self.train(0), self.train(0)
        for name in a.state:
            torch.testing.assert_close(a.state[name], b.state[name], rtol=0, atol=0)

    def test_proximal_term_gradient(self):
        base = clone_state(self.model)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(1)
        self.model.zero_grad()
        loss = proximal_loss(self.model, base, 0.2)
        loss.backward()
        for p in self.model.parameters():
            torch.testing.assert_close(p.grad, torch.full_like(p, 0.2))

    def test_mu_positive_changes_update(self):
        self.assertNotEqual(state_hash(self.train(0).state), state_hash(self.train(1).state))

    def test_bad_configuration(self):
        for config in [{"momentum":0.9}, {"learning_rate":float("nan")}, {"rounds":True}]:
            with self.assertRaises(ValueError):
                checked_config(config)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        frame, schema, source = load_dataset("canonical")
        partition = make_partition(frame, schema, file_hash(source), "canonical", clients=3)
        self.path = self.root / "partition.json"
        self.path.write_text(json.dumps(partition), encoding="utf-8")
        self.config = {"rounds": 2, "hidden": 8, "batch_size":128}

    def tearDown(self):
        self.directory.cleanup()

    def test_all_methods_end_to_end(self):
        for algorithm in ["centralized", "fedavg", "fedprox", "fednova"]:
            with self.subTest(algorithm=algorithm):
                output = self.root / algorithm
                result = run(self.path, output, algorithm, self.config)
                self.assertEqual(result["test"]["samples"], 330)
                self.assertTrue((output / "checkpoint.pt").exists())
                self.assertTrue((output / "test_predictions.npz").exists())
                self.assertEqual(len(json.loads((output / "rounds.json").read_text())), 2)

    def test_resume_exact_matches_uninterrupted(self):
        a, b = self.root / "whole", self.root / "resumed"
        run(self.path, a, "fednova", self.config)
        run(self.path, b, "fednova", self.config | {"rounds":1})
        run(self.path, b, "fednova", self.config, resume=True)
        ca = torch.load(a / "checkpoint.pt", weights_only=True)
        cb = torch.load(b / "checkpoint.pt", weights_only=True)
        self.assertEqual(state_hash(ca["model"]), state_hash(cb["model"]))
        np.testing.assert_array_equal(np.load(a / "test_predictions.npz")["probabilities"],
                                      np.load(b / "test_predictions.npz")["probabilities"])

    def test_resume_changed_settings_rejected(self):
        output = self.root / "resume"
        run(self.path, output, "fedavg", self.config | {"rounds":1})
        with self.assertRaises(ValueError):
            run(self.path, output, "fedavg", self.config | {"hidden":16}, resume=True)

    def test_mu_zero_fedprox_matches_fedavg(self):
        a, b = self.root / "avg", self.root / "prox"
        run(self.path, a, "fedavg", self.config)
        run(self.path, b, "fedprox", self.config | {"mu":0})
        ca = torch.load(a / "checkpoint.pt", weights_only=True)
        cb = torch.load(b / "checkpoint.pt", weights_only=True)
        self.assertEqual(state_hash(ca["model"]), state_hash(cb["model"]))

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            run(self.path, self.root, "fedavg", self.config)

    def test_partial_participation_and_payload_accounting(self):
        output = self.root / "partial"
        run(self.path, output, "fedavg", self.config | {"participation":0.4})
        manifest = json.loads((output / "manifest.json").read_text())
        records = json.loads((output / "rounds.json").read_text())
        for record in records:
            self.assertEqual(len(record["selected_clients"]), 2)
            self.assertEqual(record["upload_payload_bytes"], 2*manifest["model_payload_bytes"])

    def test_tampered_source_hash_rejected(self):
        partition = json.loads(self.path.read_text())
        partition["source_sha256"] = "wrong"
        self.path.write_text(json.dumps(partition), encoding="utf-8")
        with self.assertRaises(ValueError):
            run(self.path, self.root / "invalid", "fedavg", self.config)

    def test_knowledge_guidance_not_silently_enabled(self):
        registry = json.loads((Path(__file__).resolve().parents[1] / "research/agronomic_registry.json").read_text())
        self.assertFalse(registry["guidance_enabled"])
        self.assertEqual(registry["approved_rules"], [])
        with self.assertRaises(ValueError):
            checked_config({"knowledge_alpha":1.0})
