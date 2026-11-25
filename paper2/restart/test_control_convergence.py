import argparse
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from paper2.restart.control_convergence import normalized_targets,candidate,load_inner
from paper2.restart.validation_search import atomic_json

class ConvergenceTests(unittest.TestCase):
    def test_target_units_and_invalid_scale(self):
        y=np.array([2.,4.],np.float32)
        np.testing.assert_allclose(normalized_targets(y,10.)*10,y)
        for s in (0,True,float('nan')):
            with self.assertRaises(ValueError):normalized_targets(y,s)
    def test_test_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'folds/fold';folder.mkdir(parents=True)
            atomic_json(folder/'partition.json',{'kind':'inner_selection'})
            np.savez(folder/'arrays.npz',x_test=np.ones(1))
            with self.assertRaises(ValueError):load_inner(Path(temp),'fold')
    def test_exact_resume_normalized_control(self):
        run={'id':'synthetic','target_scale':10.,'learning_rate':.003,'rounds':3,'report_rounds':[1,3],'seed':42}
        fold={'train':[0,1,2,3],'clients':{'a':[0,1],'b':[2,3]},'preprocessing_identity':'synthetic'}
        a={'x_train':np.ones((4,2),np.float32),'y_train':np.array([2,3,4,5],np.float32),
            'x_validation':np.ones((2,2),np.float32),'y_validation':np.array([3,4],np.float32)}
        with tempfile.TemporaryDirectory() as temp:
            pa,pb=Path(temp)/'a',Path(temp)/'b'
            full=candidate(pa,run,fold,a,'token')
            candidate(pb,run,fold,a,'token',stop_after=1);resumed=candidate(pb,run,fold,a,'token')
            self.assertEqual(full,resumed)
            wa=torch.load(pa/'checkpoint.pt',weights_only=True)['state'];wb=torch.load(pb/'checkpoint.pt',weights_only=True)['state']
            self.assertTrue(all(torch.equal(wa[k],wb[k]) for k in wa))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConvergenceTests))
    (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures)})
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
