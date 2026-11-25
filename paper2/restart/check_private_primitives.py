"""Synthetic integration and persistent tests; never reads real datasets."""
import argparse
import io
from pathlib import Path
import unittest
import numpy as np
import torch
from paper2.restart.public_preprocessing import schema,transform_records,SPECIES
from paper2.restart.noisy_aggregation import aggregate_simulation
from paper2.restart.privacy_accounting import gaussian_budget
from paper2.restart.regression_training import make_regressor,train_regression_client
from paper2.federated import clone_state
from paper2.restart.validation_search import atomic_json,identity,sha


def smoke():
    torch.set_num_threads(1);torch.manual_seed(123)
    records=[]
    for i in range(12):
        row={name:float(i) for name in schema()['numeric_scales']}
        row.update({'contains_'+s:int(s=='pea') for s in SPECIES})
        row.update(is_intercrop=0,previous_crop='synthetic',crop_season='spring',pest_control=None)
        records.append(row)
    x=torch.from_numpy(transform_records(records));y=torch.arange(12,dtype=torch.float32)/12
    model=make_regressor(x.shape[1],4);events=[];public_meta=[]
    rng=np.random.default_rng(456)  # Reproducible simulation only; never a production DP seed.
    for round_number in range(1,4):
        updates={}
        for slot,indices in (('a',slice(0,6)),('b',slice(6,12))):
            update=train_regression_client(model,x[indices],y[indices],slot,round_number,'synthetic',
                1000+round_number,learning_rate=.001,batch_size=3)
            updates[slot]=update.state
        state,event,meta=aggregate_simulation(clone_state(model),updates,('a','b'),1.,2.,rng,simulation_only=True)
        model.load_state_dict(state);events.append(event);public_meta.append(meta)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Smoke produced invalid model')
    return {'status':'PASS_SYNTHETIC_3_ROUNDS','records':12,'public_slots':2,'rounds':3,
            'transform_identity':identity(schema()),'conditional_mathematical_budget':gaussian_budget(events,1e-5),
            'aggregation_metadata':public_meta,'real_dataset_training':False,'application_privacy_certified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);stream=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ('paper2.restart.test_privacy_accounting','paper2.restart.test_private_primitives'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('public_preprocessing.py','noisy_aggregation.py','privacy_accounting.py')}})
    if not result.wasSuccessful():print(stream.getvalue());raise SystemExit(1)
    atomic_json(args.output/'public_transform.json',schema())
    atomic_json(args.output/'smoke.json',smoke())
    print(stream.getvalue());print('PASS: synthetic three-round integration')
