import argparse
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from paper2.restart.stable_simulation_training import control_average,fit_simulation
from paper2.restart.stable_privacy_simulation_runner import execute,shared_ledger
from paper2.restart.privacy_study_protocol import VARIANTS
from paper2.restart.validation_search import ROOT,atomic_json,sha,read,identity


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'paper2/results');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.x=np.arange(12,dtype=np.float32).reshape(6,2)/12;self.y=np.arange(6,dtype=np.float32)/6
        self.fold={'kind':'outer_refit','train':list(range(6)),'test':[6,7],'validation':[],
            'clients':{'a':[0,1],'b':[2,3,4,5]},'preprocessing_identity':'synthetic',
            'support':{'test':[{},{}]},'test_group':'synthetic'}
        self.settings={'hidden':4,'batch_size':2,'mu':.01}
        self.run={'id':'synthetic','fold':'outer_0','seed':123,'rounds':3,'learning_rate':.001,'variant':VARIANTS[2]}
    def test_equal_and_global_clip_controls(self):
        base={'w':torch.zeros(2,dtype=torch.float64)};clients={'a':{'w':torch.tensor([3.,4.],dtype=torch.float64)},'b':base}
        np.testing.assert_array_equal(control_average(base,clients,('a','b'),None)['w'],[1.5,2.])
        np.testing.assert_allclose(control_average(base,clients,('a','b'),2.)['w'],[.6,.8],rtol=1e-14)
    def test_exact_resume_every_variant(self):
        for v in VARIANTS:
            run=dict(self.run,variant=v,id=v['name']);a=self.root/(v['name']+'a');b=self.root/(v['name']+'b')
            fit_simulation(a,run,self.settings,self.fold,self.x,self.y,'token')
            fit_simulation(b,run,self.settings,self.fold,self.x,self.y,'token',stop_after=1)
            fit_simulation(b,run,self.settings,self.fold,self.x,self.y,'token')
            wa=torch.load(a/'model.pt',weights_only=True);wb=torch.load(b/'model.pt',weights_only=True)
            self.assertTrue(all(torch.equal(wa[k],wb[k]) for k in wa))
            self.assertEqual(read(a/'accounting.json'),read(b/'accounting.json'))
    def test_ledger_tampering_rejected(self):
        folder=self.root/'model';fit_simulation(folder,self.run,self.settings,self.fold,self.x,self.y,'token',stop_after=1)
        ck=torch.load(folder/'checkpoint.pt',weights_only=True);ck['events'].append(ck['events'][0]);torch.save(ck,folder/'checkpoint.pt')
        with self.assertRaises(ValueError):fit_simulation(folder,self.run,self.settings,self.fold,self.x,self.y,'token')
    def test_completed_fit_no_retraining(self):
        folder=self.root/'model';fit_simulation(folder,self.run,self.settings,self.fold,self.x,self.y,'token')
        with patch('paper2.restart.stable_simulation_training.train_regression_client',side_effect=AssertionError('retrain')):
            fit_simulation(folder,self.run,self.settings,self.fold,self.x,self.y,'token')
    def test_accounting_control_is_not_zero_epsilon_claim(self):
        run=dict(self.run,variant=VARIANTS[0]);fit_simulation(self.root,run,self.settings,self.fold,self.x,self.y,'token')
        self.assertIsNone(read(self.root/'accounting.json')['conditional_gaussian_budget'])
    def test_bad_clients(self):
        with self.assertRaises(ValueError):control_average({'w':torch.zeros(2)}, {'a':{'w':torch.zeros(2)}},('a','b'),None)
    def test_large_noisy_state_has_bounded_local_step(self):
        from paper2.restart.stable_regression_training import make_regressor,train_regression_client
        model=make_regressor(2,4)
        with torch.no_grad():
            for parameter in model.parameters():parameter.fill_(30.)
        base={k:v.clone() for k,v in model.state_dict().items()}
        update=train_regression_client(model,torch.from_numpy(self.x),torch.from_numpy(self.y),
            'a',1,'synthetic',123,learning_rate=.001,batch_size=3)
        norm=torch.sqrt(sum(((update.state[k].double()-base[k].double())**2).sum() for k in base)).item()
        self.assertLessEqual(norm,.001*update.steps+1e-5)
        self.assertTrue(all(torch.isfinite(v).all() for v in update.state.values()))
    def test_capped_integration_and_cached_resume(self):
        data=self.root/'data';folder=data/'folds/outer_0';folder.mkdir(parents=True)
        atomic_json(folder/'partition.json',self.fold);atomic_json(folder/'preprocessing.json',{'identity':'synthetic'})
        np.savez(folder/'arrays.npz',x_train=self.x,y_train=self.y,ids_train=np.arange(6),
            x_test=np.zeros((2,2),np.float32),y_test=np.array([1.,2.],np.float32),ids_test=np.array([6,7]))
        atomic_json(data/'manifest.json',{'assets':{'outer_0':{n:sha(folder/n) for n in ('partition.json','preprocessing.json','arrays.npz')}}})
        plan=[dict(self.run,id=v['name'],variant=v) for v in VARIANTS]
        config={'neural_settings':self.settings,'public_slots':{'outer_0':['a','b']}}
        record={'sim':'synthetic'};output=self.root/'out';package=self.root/'package'
        with patch('paper2.restart.stable_privacy_simulation_runner.verify_execution',return_value=(record,config,plan,data)):
            self.assertEqual(execute(package,output,2,dry_run=False)['completed'],2)
            self.assertEqual(execute(package,output,3,resume=True,dry_run=False)['committed_noisy_events'],9)
            with patch('paper2.restart.stable_simulation_training.train_regression_client',side_effect=AssertionError('retrain')):
                self.assertEqual(execute(package,output,1,resume=True,dry_run=False)['executed'],0)
        ledger=read(output/'shared_ledger.json');self.assertEqual(ledger['event_count'],9)
        self.assertEqual(ledger['per_slot'][0]['releases'],9)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);stream=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ('paper2.restart.test_stable_simulation','paper2.restart.test_privacy_accounting','paper2.restart.test_private_primitives'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures)})
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
