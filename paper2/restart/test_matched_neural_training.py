"""Synthetic tests for RR06 local work and aggregation primitives."""
import unittest

import torch

from paper2.federated import clone_state
from paper2.restart import matched_neural_training as training


def run(algorithm="fedavg", optimizer="sgd", objective="sample_weighted"):
    return {"id": "fixture", "fold": "outer_0_inner_1", "variant": "fixture", "algorithm": algorithm,
            "aggregation_objective": objective, "outer_test_inference": False, "optimizer": optimizer,
            "schedule": "constant" if optimizer == "sgd" else "cosine", "hidden": 4 if optimizer == "sgd" else 6,
            "learning_rate": 0.01, "weight_decay": 0.0 if optimizer == "sgd" else 0.0001, "rounds": 3, "local_steps": 4}


class MatchedNeuralTrainingTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [2., 1.], [2., 2.]], dtype=torch.float32)
        self.y = torch.tensor([0., 1., 2., 2., 3.], dtype=torch.float32)

    def test_client_has_exact_fixed_work_and_is_deterministic(self):
        candidate = run(); model = training.make_regressor(2, 4)
        first = training.train_client(model, self.x, self.y, "a", 1, "schema", 17, candidate, batch_size=2, fedprox_mu=.01)
        second = training.train_client(model, self.x, self.y, "a", 1, "schema", 17, candidate, batch_size=2, fedprox_mu=.01)
        self.assertEqual(first.steps, 4)
        self.assertEqual(first.state.keys(), second.state.keys())
        self.assertTrue(all(torch.equal(first.state[key], second.state[key]) for key in first.state))

    def test_centralized_step_override_is_recorded_and_checked(self):
        candidate = run(algorithm="centralized"); model = training.make_regressor(2, 4)
        update = training.train_client(model, self.x, self.y, "central", 1, "schema", 17, candidate,
                                       batch_size=2, fedprox_mu=.01, steps=8)
        self.assertEqual(update.steps, 8)
        self.assertGreater(update.examples_processed, 0)
        state = training.aggregate_round(clone_state(model), [update], candidate, 1, "schema", {"central": 5}, expected_steps=8)
        self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))

    def test_fixed_step_example_budget(self):
        self.assertEqual(training.fixed_step_examples(5, 3, 4), 10)
        self.assertEqual(training.fixed_step_examples(2, 32, 4), 8)
        candidate = run(algorithm="centralized"); model = training.make_regressor(2, 4)
        update = training.train_client(model, self.x, self.y, "central", 1, "schema", 17, candidate,
                                       batch_size=2, fedprox_mu=.01, steps=8, example_budget=13)
        self.assertEqual(update.steps, 8)
        self.assertEqual(update.examples_processed, 13)

    def test_adamw_optimizer_state_can_persist(self):
        candidate = run(algorithm="centralized", optimizer="adamw"); model = training.make_regressor(2, 6)
        first = training.train_client(model, self.x, self.y, "central", 1, "schema", 17, candidate,
                                      batch_size=2, fedprox_mu=.01, steps=4)
        self.assertTrue(first.optimizer_state["state"])
        model.load_state_dict(first.state)
        second = training.train_client(model, self.x, self.y, "central", 2, "schema", 18, candidate,
                                       batch_size=2, fedprox_mu=.01, steps=4, optimizer_state=first.optimizer_state)
        self.assertTrue(second.optimizer_state["state"])

    def test_unavailable_cuda_is_rejected(self):
        candidate = run()
        model = training.make_regressor(2, 4)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            training.train_client(model, self.x, self.y, "a", 1, "schema", 17, candidate,
                                  batch_size=2, fedprox_mu=.01, device="cuda")

    def test_adamw_cosine_and_restricted_fednova_guard(self):
        candidate = run(optimizer="adamw")
        self.assertLess(training.learning_rate(candidate, 3), candidate["learning_rate"])
        with self.assertRaisesRegex(ValueError, "FedNova"):
            training.validate_run(run("fednova_restricted", "adamw"), fedprox_mu=.01)

    def test_equal_client_average_is_not_sample_weighted(self):
        candidate = run(objective="equal_client")
        model = training.make_regressor(2, 4); base = clone_state(model)
        left = training.train_client(model, self.x[:1], self.y[:1], "a", 1, "schema", 1, candidate, batch_size=1, fedprox_mu=.01)
        right = training.train_client(model, self.x[1:], self.y[1:], "b", 1, "schema", 2, candidate, batch_size=2, fedprox_mu=.01)
        state = training.aggregate_round(base, [left, right], candidate, 1, "schema", {"a": 1, "b": 4})
        self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))


if __name__ == "__main__":
    unittest.main()
