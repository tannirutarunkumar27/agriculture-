import unittest

from paper2.restart.agronomic_guidance_runner import soft_prior


class AgronomicGuidanceTests(unittest.TestCase):
    def test_bounded_interval_prior(self):
        self.assertEqual(soft_prior(6.5, 6.0, 7.0), 1.0)
        self.assertEqual(soft_prior(5.0, 6.0, 7.0), 0.5)
        self.assertEqual(soft_prior(8.0, 6.0, 7.0), 0.5)

    def test_bad_interval_rejected(self):
        with self.assertRaises(ValueError):
            soft_prior(6.0, 7.0, 6.0)


if __name__ == "__main__":
    unittest.main()
