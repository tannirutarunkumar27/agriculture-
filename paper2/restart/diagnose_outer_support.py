"""Read-only feature support and saved-weight diagnosis; no model inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from paper2.restart.validation_search import ROOT, read, sha, atomic_json
from paper2.restart.outer_runner import verify_runner
from paper2.restart.regression_training import make_regressor


def diagnose(output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    _,manifest,config,plan=verify_runner(ROOT/'paper2/results/restart_outer_execution_v1')
    data=Path(manifest['data']); study=ROOT/'paper2/results/restart_outer_study_v1'
    features=[]; weights=[]; errors=[]
    for fold_name in sorted({r['fold'] for r in plan}):
        folder=data/'folds'/fold_name; state=read(folder/'preprocessing.json'); fold=read(folder/'partition.json')
        with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
            train,test=arrays['x_train'],arrays['x_test']
            indices=[]
            for i,name in enumerate(state['feature_names']):
                if np.all(train[:,i]==0) and np.any(test[:,i]!=0):
                    indices.append(i)
                    features.append({'fold':fold_name,'group':fold['test_group'],'feature':name,
                        'train_min':0.,'train_max':0.,'test_min':float(test[:,i].min()),'test_max':float(test[:,i].max()),
                        'test_nonzero_rows':int(np.count_nonzero(test[:,i])), 'scaler':state['numeric'].get(name)})
            dimensions=train.shape[1]
        for run in (r for r in plan if r['fold']==fold_name):
            run_dir=study/'runs'/run['id']; receipt=read(run_dir/'completion.json')
            if any(sha(run_dir/n)!=h for n,h in receipt['assets'].items()): raise ValueError('Run assets changed')
            with np.load(run_dir/'test_cache.npz',allow_pickle=False) as cache:
                errors.append({'id':run['id'],'group':fold['test_group'],
                    'prediction_min':float(cache['prediction'].min()),'prediction_max':float(cache['prediction'].max()),
                    'target_min':float(cache['y'].min()),'target_max':float(cache['y'].max()),
                    'rmse':float(np.sqrt(np.mean((cache['prediction'].astype(float)-cache['y'].astype(float))**2)))})
            if run['algorithm'] not in ('centralized','fedavg','fedprox','fednova'): continue
            torch.manual_seed(run['seed'])
            initial=make_regressor(dimensions,config['neural_settings']['hidden']).state_dict()['0.weight']
            trained=torch.load(run_dir/'model.pt',map_location='cpu',weights_only=True)['0.weight']
            for i in indices:
                weights.append({'id':run['id'],'feature':state['feature_names'][i],
                    'max_initial_to_trained_weight_change':float((trained[:,i]-initial[:,i]).abs().max()),
                    'max_abs_trained_weight':float(trained[:,i].abs().max())})
    output.mkdir(parents=True)
    atomic_json(output/'diagnosis.json',{'status':'COMPLETED_READ_ONLY_DIAGNOSIS','zero_train_nonzero_test':features,
        'incoming_weights':weights,'cached_prediction_ranges':errors,'model_inference':False,'training':False,
        'mechanism':'Zero training inputs give zero supervised gradients for corresponding first-layer weights; held-out nonzero values activate unsupported connections.',
        'limits':'Mechanism and scale mismatch established; quantitative causal contribution not measured without a separately labeled post-hoc sensitivity experiment.'})
    return {'features':len(features),'weight_checks':len(weights),'cached_runs':len(errors)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(diagnose(p.parse_args().output))
