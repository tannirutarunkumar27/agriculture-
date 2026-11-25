import math
import unittest
import numpy as np
from paper2.restart.privacy_accounting import clip_vector,replacement_sensitivity,gaussian_budget


class PrivacyTests(unittest.TestCase):
    def test_clip_known_direction_and_no_mutation(self):
        x=np.array([3.,4.]);copy=x.copy()
        y=clip_vector(x,2.)
        np.testing.assert_allclose(y,[1.2,1.6],rtol=1e-14)
        self.assertLessEqual(np.linalg.norm(y),2.)
        np.testing.assert_array_equal(x,copy)
        np.testing.assert_array_equal(clip_vector(x,6.),x)

    def test_large_vector_stable(self):
        y=clip_vector(np.array([1e300,-1e300]),1.)
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(np.linalg.norm(y),1.)
        self.assertGreater(np.linalg.norm(y),.99)

    def test_replacement_factor_two(self):
        sensitivity=replacement_sensitivity(2.,4)
        self.assertEqual(sensitivity,1.)
        # Opposite clipped one-slot contributions attain this sensitivity.
        self.assertEqual(abs((2.-(-2.))/4),sensitivity)

    def test_gaussian_closed_form(self):
        result=gaussian_budget([{'sensitivity':1.,'noise_std':2.,'count':8}],1e-5)
        self.assertEqual(result['rho'],1.)
        self.assertAlmostEqual(result['epsilon_upper_bound'],1+2*math.sqrt(math.log(1e5)))

    def test_composition_and_monotonicity(self):
        one={'sensitivity':2.,'noise_std':4.,'count':10}
        a=gaussian_budget([one,one],1e-5)
        b=gaussian_budget([dict(one,count=20)],1e-5)
        self.assertEqual(a,b)
        self.assertGreater(a['epsilon_upper_bound'],gaussian_budget([one],1e-5)['epsilon_upper_bound'])

    def test_empty_ledger(self):
        self.assertEqual(gaussian_budget([],1e-5)['epsilon_upper_bound'],0.)

    def test_cost_underflow_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_budget([{'sensitivity':1e-300,'noise_std':1e300,'count':1}],1e-5)

    def test_bad_events_and_false_amplification(self):
        base={'sensitivity':1.,'noise_std':1.,'count':1}
        for event in (dict(base,count=True),dict(base,noise_std=0),dict(base,sensitivity=float('nan')),dict(base,sampling_probability=.1)):
            with self.assertRaises(ValueError):gaussian_budget([event],1e-5)
        for delta in (0,1,float('inf'),True):
            with self.assertRaises(ValueError):gaussian_budget([base],delta)
        with self.assertRaises(ValueError):replacement_sensitivity(1,True)
        with self.assertRaises(ValueError):clip_vector(np.array([float('nan')]),1)
