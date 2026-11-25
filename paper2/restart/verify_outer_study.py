"""Independent cached-output audit. No model loading or test reinference."""
import argparse
import io
import json
from pathlib import Path
import unittest
import numpy as np
from paper2.restart.validation_search import ROOT, read, sha, identity, atomic_json, writer_lock
from paper2.restart.outer_runner import verify_runner


def scores(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or not len(y) or y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Invalid cached regression values')
    squared = (prediction-y)**2
    total = float(np.sum((y-y.mean())**2))
    return {'samples':len(y), 'rmse':float(np.sqrt(squared.mean())), 'mae':float(np.abs(prediction-y).mean()),
            'negative_predictions':int((prediction<0).sum()), 'r2':None if len(y)<2 or total==0 else 1-float(squared.sum())/total}


def audit(package, study, output):
    output, study = Path(output).resolve(), Path(study).resolve()
    record, manifest, config, plan = verify_runner(package)
    if output.exists():
        raise FileExistsError(output)
    if output == study or study in output.parents or output in study.parents:
        raise ValueError('Audit must be separate from study')
    binding = identity(record)
    if read(study/'binding.json')['identity'] != binding:
        raise ValueError('Study binding mismatch')
    data = Path(manifest['data'])
    rows, pending = [], []
    with writer_lock(study):
        for run in plan:
            folder = study/'runs'/run['id']
            if not (folder/'completion.json').exists():
                pending.append(run['id'])
                continue
            token = identity({'binding':binding,'run':run})
            receipt, result, fitted = (read(folder/name) for name in ('completion.json','result.json','fit.json'))
            if any(value['identity'] != token for value in (receipt,result,fitted)) or result['run'] != run:
                raise ValueError('Receipt identity mismatch')
            expected_names = {'fit.json',fitted['model'],'test_cache.npz','result.json','inference_started.json'}
            if set(receipt['assets']) != expected_names or any(sha(folder/name)!=value for name,value in receipt['assets'].items()):
                raise ValueError('Receipt assets mismatch')
            if sha(folder/fitted['model']) != fitted['model_sha256']:
                raise ValueError('Fitted model binding mismatch')
            started = read(folder/'inference_started.json')
            if started != {'identity':token,'model_sha256':fitted['model_sha256']}:
                raise ValueError('Inference intent mismatch')
            fold = read(data/'folds'/run['fold']/'partition.json')
            context = {'fold':run['fold'],'test_group':fold['test_group'],'ids_test':fold['test'],
                       'support':fold['support']['test'],'preprocessing_identity':fold['preprocessing_identity']}
            if read(folder/'partition_context.json') != context:
                raise ValueError('Support context mismatch')
            with np.load(folder/'test_cache.npz',allow_pickle=False) as cache, np.load(data/'folds'/run['fold']/'arrays.npz',allow_pickle=False) as source:
                if set(cache.files) != {'y','prediction','ids'}:
                    raise ValueError('Unexpected cache fields')
                np.testing.assert_array_equal(cache['ids'],source['ids_test'])
                np.testing.assert_array_equal(cache['y'],source['y_test'])
                measured = scores(cache['y'],cache['prediction'])
                for key,value in result['test'].items():
                    if key not in measured or not np.isclose(value,measured[key],rtol=1e-12,atol=1e-12):
                        raise ValueError('Independent metric mismatch')
                panels = {}
                for name in ('all_species_seen','composition_seen'):
                    mask = np.array([entry[name] for entry in context['support']],dtype=bool)
                    panels[name] = scores(cache['y'][mask],cache['prediction'][mask]) if mask.any() else None
            rows.append({'id':run['id'],'algorithm':run['algorithm'],'seed':run['seed'],'group':fold['test_group'],
                         'metrics':measured,'supported_panels':panels,'completion_sha256':sha(folder/'completion.json')})
        output.mkdir(parents=True)
        report = {'status':'PASS_COMPLETE' if not pending else 'PASS_PARTIAL','verified':len(rows),
                  'planned':len(plan),'pending':len(pending),'rows':rows,'model_reinference':False,
                  'interpretation':'partial folds/seeds are not final comparative evidence'}
        atomic_json(output/'verification.json',report)
    return {k:report[k] for k in ('status','verified','planned','pending','model_reinference')}


def test_evidence(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
        ('paper2.restart.test_outer_protocol','paper2.restart.test_outer_training','paper2.restart.test_outer_runner','paper2.restart.test_outer_audit'))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'fixtures':'synthetic_only'})
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError('Outer verification tests failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=ROOT/'paper2/results/restart_outer_execution_v1')
    parser.add_argument('--study',type=Path,default=ROOT/'paper2/results/restart_outer_study_v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tests',action='store_true')
    args=parser.parse_args()
    if args.tests:
        test_evidence(args.output)
    else:
        print(json.dumps(audit(args.package,args.study,args.output),indent=2))
