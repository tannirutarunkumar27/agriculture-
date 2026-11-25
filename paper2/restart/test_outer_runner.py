import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from paper2.restart.outer_runner import coordinate, execute
from paper2.restart.validation_search import ROOT, atomic_json, sha, read, writer_lock


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        # The coordinator's public execution contract writes beneath
        # paper2/results. The sandbox denies atomic renames in its protected
        # system-temp folder, so exercise the same writable location used by
        # real study outputs.
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "paper2/results")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.output = self.root/'data', self.root/'out'
        self.output.mkdir()
        self.plan = [{'id': 'test_'+str(i), 'fold': 'outer_0', 'algorithm': 'training_mean', 'seed': i,
                      'configuration': {'model': 'training_mean'}} for i in range(2)]
        self.config = {'neural_settings': {'hidden': 4, 'batch_size': 2, 'mu': .01}}
        folder = self.data/'folds/outer_0'
        folder.mkdir(parents=True)
        fold = {'kind':'outer_refit', 'train':[0,1,2], 'test':[3,4], 'validation':[],
                'test_group':'synthetic', 'preprocessing_identity':'test',
                'clients':{'a':[0,1,2]}, 'support':{'test':[{},{}]}}
        atomic_json(folder/'partition.json', fold)
        atomic_json(folder/'preprocessing.json', {'identity':'test'})
        np.savez(folder/'arrays.npz', x_train=np.ones((3,2),np.float32), y_train=np.arange(3,dtype=np.float32),
                 ids_train=np.array([0,1,2]), x_test=np.ones((2,2),np.float32), y_test=np.array([1,2],np.float32), ids_test=np.array([3,4]))
        atomic_json(self.data/'manifest.json', {'assets':{'outer_0':{n:sha(folder/n) for n in ('partition.json','preprocessing.json','arrays.npz')}}})

    def test_cap_and_cached_resume(self):
        result = coordinate(self.plan,self.config,self.data,self.output,'binding',1)
        self.assertEqual(result['completed'],1)
        result = coordinate(self.plan,self.config,self.data,self.output,'binding',1)
        self.assertEqual(result['completed'],2)
        with patch('paper2.restart.outer_runner.fit',side_effect=AssertionError('retrain')):
            self.assertEqual(coordinate(self.plan,self.config,self.data,self.output,'binding',1)['executed_this_invocation'],0)

    def test_failure_stop_and_retry(self):
        with patch('paper2.restart.outer_runner.fit',side_effect=RuntimeError('diagnostic')):
            with self.assertRaises(RuntimeError):
                coordinate(self.plan,self.config,self.data,self.output,'binding',1)
        self.assertEqual(read(self.output/'progress.json')['status'],'FAILED')
        with self.assertRaises(RuntimeError):
            coordinate(self.plan,self.config,self.data,self.output,'binding',1)
        self.assertEqual(coordinate(self.plan,self.config,self.data,self.output,'binding',1,True)['completed'],1)
        self.assertEqual(len(read(self.output/'runs/test_0/attempts.json')),2)

    def test_bad_asset_journal(self):
        (self.data/'folds/outer_0/preprocessing.json').write_text('{}')
        with self.assertRaises(ValueError):
            coordinate(self.plan,self.config,self.data,self.output,'binding',1)
        self.assertTrue((self.output/'runs/test_0/failure.json').exists())

    def test_context_tamper(self):
        coordinate(self.plan,self.config,self.data,self.output,'binding',1)
        (self.output/'runs/test_0/partition_context.json').write_text('{}')
        with self.assertRaises(ValueError):
            coordinate(self.plan,self.config,self.data,self.output,'binding',1)

    def test_ambiguous_retry_blocked(self):
        folder=self.output/'runs/test_0'
        folder.mkdir(parents=True)
        atomic_json(folder/'inference_started.json',{})
        with self.assertRaises(RuntimeError):
            coordinate(self.plan,self.config,self.data,self.output,'binding',1,True)

    def test_writer_exclusion(self):
        with writer_lock(self.output):
            with self.assertRaises((RuntimeError,BlockingIOError)):
                with writer_lock(self.output):
                    self.fail('second writer admitted')

    def test_dry_run_and_cap(self):
        record={'plan_package':str(self.root/'plan')}
        with patch('paper2.restart.outer_runner.verify_runner',return_value=(record,{'data':str(self.data)},self.config,self.plan)):
            output=Path(__file__).resolve().parents[2]/'paper2/results/not_created_outer_test'
            self.assertEqual(execute(self.root/'package',output)['status'],'DRY_RUN_NO_INFERENCE')
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            execute(self.root,self.output,max_runs=0)


if __name__=='__main__':
    unittest.main()
