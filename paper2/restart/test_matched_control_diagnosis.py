import unittest
import numpy as np
from paper2.restart.diagnose_public_control_fit import matched_fit

class MatchedDiagnosticTests(unittest.TestCase):
    def test_unequal_clients_do_not_use_pooled_mean(self):
        y=np.array([0.]+[10.]*9);p=np.full(10,5.)
        result=matched_fit(y,p,[[0],list(range(1,10))])
        self.assertEqual(result['equal_client_optimal_constant'],5.)
        self.assertEqual(result['equal_client_constant_mse'],25.)
        self.assertFalse(result['worse_than_matched_constant'])
        self.assertGreater(np.mean((p-y)**2),np.mean((y.mean()-y)**2))
    def test_underfit_detected(self):
        result=matched_fit(np.array([4.,6.]),np.zeros(2),[[0],[1]])
        self.assertTrue(result['worse_than_matched_constant'])
