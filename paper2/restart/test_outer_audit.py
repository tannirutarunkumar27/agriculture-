import unittest
import numpy as np
from paper2.restart.verify_outer_study import scores


class AuditMetricTests(unittest.TestCase):
    def test_known_values(self):
        result=scores(np.array([1.,3.]),np.array([-1.,3.]))
        self.assertAlmostEqual(result['rmse'],np.sqrt(2))
        self.assertEqual(result['mae'],1.)
        self.assertEqual(result['r2'],-1.)
        self.assertEqual(result['negative_predictions'],1)

    def test_undefined_r2(self):
        self.assertIsNone(scores(np.array([1.]),np.array([2.]))['r2'])
        self.assertIsNone(scores(np.array([1.,1.]),np.array([2.,2.]))['r2'])

    def test_bad_arrays(self):
        for y,p in (([],[]),([1.],[float('nan')]),([1.,2.],[1.])):
            with self.assertRaises(ValueError):
                scores(np.array(y),np.array(p))
