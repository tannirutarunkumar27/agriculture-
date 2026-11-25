import unittest
import argparse
import io
from pathlib import Path
from paper2.restart.privacy_study_protocol import make_plan,VARIANTS


class MatchedPlanTests(unittest.TestCase):
    def setUp(self):
        self.parent=[{'algorithm':'fedavg','fold':'outer_0','seed':123,'rounds':22,'learning_rate':.003}]
    def test_matched_settings_and_controls(self):
        plan=make_plan(self.parent,VARIANTS)
        self.assertEqual(len(plan),5)
        self.assertTrue(all(r['rounds']==22 and r['learning_rate']==.003 and r['seed']==123 for r in plan))
        self.assertEqual(plan[0]['variant']['clip_norm'],None)
        self.assertEqual(plan[1]['variant']['noise_multiplier'],None)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):make_plan(self.parent,[VARIANTS[0],VARIANTS[0]])
    def test_noise_without_clip_rejected(self):
        with self.assertRaises(ValueError):make_plan(self.parent,[{'name':'bad','clip_norm':None,'noise_multiplier':2}])
    def test_bad_rounds_and_missing_parent(self):
        with self.assertRaises(ValueError):make_plan([dict(self.parent[0],rounds=0)],VARIANTS)
        with self.assertRaises(ValueError):make_plan([],VARIANTS)


if __name__=='__main__':
    from paper2.restart.validation_search import atomic_json
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);stream=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ('paper2.restart.test_public_study','paper2.restart.test_private_primitives','paper2.restart.test_privacy_accounting'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)})
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
