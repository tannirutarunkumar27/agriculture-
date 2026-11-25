import unittest
import numpy as np
from paper2.restart.summarize_outer import aggregate

class SummaryTests(unittest.TestCase):
    def test_unequal_groups(self):
        groups=[{'ids':np.array([0]),'y':np.array([0.]),'p':np.array([4.])},
                {'ids':np.array([1,2,3]),'y':np.zeros(3),'p':np.zeros(3)}]
        result=aggregate(groups)
        self.assertEqual(result['equal_group_rmse'],2.)
        self.assertEqual(result['equal_group_mae'],2.)
        self.assertEqual(result['pooled']['rmse'],2.)
        self.assertEqual(result['pooled']['mae'],1.)
        self.assertIsNone(result['pooled']['r2'])
    def test_overlap_rejected(self):
        g={'ids':np.array([0]),'y':np.array([0.]),'p':np.array([1.])}
        with self.assertRaises(ValueError):aggregate([g,g])
    def test_empty(self):
        self.assertIsNone(aggregate([]))
