"""No-test-array and local-model checks for the Flower pilot helpers."""
import unittest
import numpy as np

from paper2.secagg_pilot import task


class SecAggPilotTaskTests(unittest.TestCase):
    def test_parameter_roundtrip_and_bad_shape_rejection(self):
        model = task.initial_parameters(3, hidden=4)
        values = task.state_to_ndarrays(model)
        task.set_ndarrays(model, values)
        bad = list(values); bad[0] = np.zeros((1,), dtype=np.float32)
        with self.assertRaises(ValueError): task.set_ndarrays(model, bad)

    def test_local_fit_returns_finite_model_arrays(self):
        model = task.initial_parameters(3, hidden=4)
        x = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
        y = np.arange(6, dtype=np.float32) / 10
        values = task.local_fit(model, x, y, "synthetic_client", 1)
        self.assertTrue(all(np.isfinite(value).all() for value in values))

    def test_invalid_round_is_rejected(self):
        model = task.initial_parameters(2, hidden=4)
        with self.assertRaises(ValueError):
            task.local_fit(model, np.zeros((2, 2), np.float32), np.zeros(2, np.float32), "c", 0)

    def test_fit_config_sends_the_validated_server_round(self):
        self.assertEqual(task.fit_config(3), {"server_round": 3})
        with self.assertRaises(ValueError):
            task.fit_config(0)


if __name__ == "__main__":
    unittest.main()
