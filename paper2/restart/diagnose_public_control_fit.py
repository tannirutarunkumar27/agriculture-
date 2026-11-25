"""Training-only fit diagnostic for public v3 controls; no held-out inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from paper2.restart.work_scaled_privacy_simulation_runner import verify_execution
from paper2.restart.validation_search import ROOT,read,sha,atomic_json
from paper2.restart.regression_training import make_regressor
from paper2.restart.verify_outer_study import scores


def matched_fit(y,prediction,client_indices):
    groups=[np.array(ids,dtype=int) for ids in client_indices]
    if not groups or any(not len(g) for g in groups):raise ValueError('Nonempty client groups required')
    constant=float(np.mean([y[g].astype(float).mean() for g in groups]))
    model_mse=float(np.mean([np.mean((prediction[g].astype(float)-y[g].astype(float))**2) for g in groups]))
    reference_mse=float(np.mean([np.mean((constant-y[g].astype(float))**2) for g in groups]))
    return {'equal_client_model_mse':model_mse,'equal_client_constant_mse':reference_mse,
            'equal_client_optimal_constant':constant,'worse_than_matched_constant':model_mse>reference_mse}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    _,config,plan,data=verify_execution(ROOT/'paper2/results/restart_privacy_execution_v3')
    study=ROOT/'paper2/results/restart_privacy_simulation_v3';torch.set_num_threads(1);rows=[]
    for run in plan:
        if run['variant']['name']!='equal_average':continue
        folder=study/'runs'/run['id'];receipt=read(folder/'completion.json')
        if any(sha(folder/n)!=h for n,h in receipt['assets'].items()):raise ValueError('Control model assets changed')
        with np.load(data/'folds'/run['fold']/'arrays.npz',allow_pickle=False) as a:
            x,y=a['x_train'],a['y_train']
        model=make_regressor(x.shape[1],config['neural_settings']['hidden'])
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():prediction=model(torch.from_numpy(x))[:,0].numpy()
        model_scores=scores(y,prediction);mean_scores=scores(y,np.full(y.shape,float(y.astype(float).mean())))
        fold=read(data/'folds'/run['fold']/'partition.json');lookup={row:i for i,row in enumerate(fold['train'])}
        matched=matched_fit(y,prediction,[[lookup[i] for i in rows] for rows in fold['clients'].values()])
        rows.append({'id':run['id'],'training_model':model_scores,'training_mean_reference':mean_scores,
            'matched_equal_client_fit':matched,'prediction_mean':float(prediction.mean()),'target_mean':float(y.mean())})
    args.output.mkdir(parents=True)
    result={'status':'COMPLETED_PUBLIC_TRAINING_ONLY_DIAGNOSIS','controls':len(rows),'rows':rows,
        'model_worse_than_training_mean':sum(r['training_model']['rmse']>r['training_mean_reference']['rmse'] for r in rows),
        'model_worse_than_equal_client_constant':sum(r['matched_equal_client_fit']['worse_than_matched_constant'] for r in rows),
        'heldout_model_inference':False,'private_release_claim':False}
    atomic_json(args.output/'diagnosis.json',result)
    print({k:v for k,v in result.items() if k!='rows'})
