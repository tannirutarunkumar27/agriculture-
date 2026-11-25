"""Persist synthetic accounting evidence and illustrative full-participation budgets."""
import argparse
import io
from pathlib import Path
import unittest
from paper2.restart.privacy_accounting import replacement_sensitivity,gaussian_budget
from paper2.restart.validation_search import atomic_json,sha


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('paper2.restart.test_privacy_accounting')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'implementation_sha256':sha(Path(__file__).with_name('privacy_accounting.py')),
        'private_training_performed':False})
    if not result.wasSuccessful():raise SystemExit(1)
    sensitivity=replacement_sensitivity(1.,7)
    rows=[]
    for multiplier in (1.,2.,4.,8.):
        for rounds in (80,1600):
            rows.append({'public_slots':7,'clip_norm':1.,'std_over_sensitivity':multiplier,
                'illustration': 'single 80-release run' if rounds==80 else '20 runs of 80 releases sharing a protected client',
                **gaussian_budget([{'sensitivity':sensitivity,'noise_std':multiplier*sensitivity,'count':rounds}],1e-5)})
    atomic_json(args.output/'illustrative_budgets.json',{'scope':'Mathematical planning only; no applied privacy guarantee',
        'adjacency':'replace whole fixed client slot','budgets':rows})
    print(stream.getvalue())
