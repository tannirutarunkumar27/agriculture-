import unittest
import tempfile
from pathlib import Path
import shutil
import numpy as np
import torch
from paper2.restart.support_guard import fit_gate,apply_gate

class GateTests(unittest.TestCase):
    def test_train_unchanged(self):
        x=np.array([[0,1,1],[0,2,1]],np.float32);mask=fit_gate(x)
        np.testing.assert_array_equal(mask,[True,False,False])
        np.testing.assert_array_equal(apply_gate(x,mask),x)
    def test_unseen_values_cannot_activate(self):
        train=np.array([[0,1],[0,2]],np.float32)
        heldout=np.array([[263,3]],np.float32)
        result=apply_gate(heldout,fit_gate(train))
        np.testing.assert_array_equal(result,[[0,3]])
        self.assertEqual(heldout[0,0],263)
    def test_heldout_does_not_set_mask(self):
        train=np.array([[0,0],[0,1]],np.float32)
        np.testing.assert_array_equal(fit_gate(train),[True,False])
    def test_zero_gradients(self):
        x=torch.tensor([[0.,1.],[0.,2.]])
        model=torch.nn.Linear(2,1)
        model(x).square().mean().backward()
        self.assertEqual(float(model.weight.grad[0,0]),0.)
    def test_bad_arrays(self):
        for x in (np.empty((0,2),np.float32),np.ones((2,2),np.float64),np.array([[np.nan]],np.float32)):
            with self.assertRaises(ValueError):fit_gate(x)
        with self.assertRaises(ValueError):apply_gate(np.ones((2,2),np.float32),np.array([True]))
    def test_frozen_gate_tamper_rejected(self):
        from paper2.restart.guard_sensitivity import verify
        from paper2.restart.validation_search import ROOT
        original=ROOT/'paper2/results/restart_guard_protocol_v1'
        with tempfile.TemporaryDirectory() as temp:
            package=Path(temp)/'package';shutil.copytree(original,package)
            (package/'gates.json').write_text('{}')
            with self.assertRaises(ValueError):verify(package)
    def test_frozen_snapshot_tamper_rejected(self):
        from paper2.restart.guard_sensitivity import verify
        from paper2.restart.validation_search import ROOT
        original=ROOT/'paper2/results/restart_guard_protocol_v1'
        with tempfile.TemporaryDirectory() as temp:
            package=Path(temp)/'package';shutil.copytree(original,package)
            (package/'source_snapshot/support_guard.py').write_text('changed')
            with self.assertRaises(ValueError):verify(package)
