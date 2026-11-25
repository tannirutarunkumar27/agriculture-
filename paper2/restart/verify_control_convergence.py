"""Independent saved-curve and inner-model audit; outer inference is absent."""
import argparse
from pathlib import Path
import math
import numpy as np
import torch
from paper2.restart.validation_search import ROOT,read,sha,atomic_json,runtime,identity
from paper2.restart.prepare_public_data import verify as verify_data
from paper2.restart.control_convergence import load_inner,CODE
from paper2.restart.regression_training import make_regressor


def verify(package,study,output):
    package,study,output=Path(package),Path(study),Path(output)
    if output.exists():raise FileExistsError(output)
    m=read(package/'manifest.json')
    if runtime()!=m['runtime'] or sha(package/'plan.json')!=m['plan_sha256']:raise ValueError('Protocol changed')
    for p in CODE:
        if sha(p)!=m['source_sha256'][p.name] or sha(package/'source_snapshot'/p.name)!=sha(p):raise ValueError('Code changed')
    data=Path(m['data'])
    if verify_data(data)['manifest_sha256']!=m['data_manifest_sha256']:raise ValueError('Data changed')
    torch.set_num_threads(1);rows=[]
    for run in read(package/'plan.json'):
        folder=study/'runs'/run['id'];receipt=read(folder/'completion.json');token=identity({'protocol':m,'run':run})
        if receipt['identity']!=token or any(sha(folder/n)!=v for n,v in receipt['assets'].items()):raise ValueError('Run assets changed')
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);result=read(folder/'result.json')
        history=result['history']
        if checkpoint['identity']!=token or result['identity']!=token or result['run']!=run or checkpoint['done']!=run['rounds'] or history!=checkpoint['history']:
            raise ValueError('Checkpoint/run mismatch')
        if [h['round'] for h in history]!=list(range(1,run['rounds']+1)):raise ValueError('Round sequence changed')
        expected=min(history,key=lambda h:h['validation']['rmse'])
        if checkpoint['best_round']!=expected['round'] or checkpoint['best']!=expected['validation']['rmse']:raise ValueError('Minimum/first-tie rule differs')
        fold,a=load_inner(data,run['fold']);model=make_regressor(a['x_train'].shape[1],32)
        model.load_state_dict(checkpoint['state']);model.eval()
        with torch.no_grad():pt=model(torch.from_numpy(a['x_train']))[:,0].numpy()*run['target_scale'];pv=model(torch.from_numpy(a['x_validation']))[:,0].numpy()*run['target_scale']
        last_rmse=float(np.sqrt(np.mean((pv.astype(float)-a['y_validation'].astype(float))**2)))
        lookup={i:n for n,i in enumerate(fold['train'])};groups=[[lookup[i] for i in ids] for ids in fold['clients'].values()]
        training_mse=float(np.mean([np.mean((pt[g].astype(float)-a['y_train'][g].astype(float))**2) for g in groups]))
        if not math.isclose(last_rmse,history[-1]['validation']['rmse'],rel_tol=1e-7,abs_tol=1e-7) or not math.isclose(training_mse,history[-1]['training_equal_client_fit']['equal_client_model_mse'],rel_tol=1e-7,abs_tol=1e-7):raise ValueError('Final metrics do not reproduce')
        model.load_state_dict(checkpoint['best_state'])
        with torch.no_grad():pb=model(torch.from_numpy(a['x_validation']))[:,0].numpy()*run['target_scale']
        best_rmse=float(np.sqrt(np.mean((pb.astype(float)-a['y_validation'].astype(float))**2)))
        if not math.isclose(best_rmse,checkpoint['best'],rel_tol=1e-7,abs_tol=1e-7):raise ValueError('Selected inner model metrics differ')
        for budget in run['report_rounds']:
            prefix=history[:budget];selected=min(prefix,key=lambda h:h['validation']['rmse']);s=result['snapshots'][str(budget)]
            if s['last']!=prefix[-1] or s['best_round']!=selected['round'] or s['best_validation_rmse']!=selected['validation']['rmse']:raise ValueError('Reported budget snapshot differs')
        rows.append({'id':run['id'],'final_train_equal_client_mse':training_mse,'best_validation_rmse':best_rmse,
            'best_round':checkpoint['best_round'],'completion_sha256':sha(folder/'completion.json')})
    output.mkdir(parents=True);report={'status':'PASS_COMPLETE_INNER_PILOT','verified':len(rows),'rows':rows,
        'outer_model_inference':False,'inner_reinference':'final and best checkpoints only; verification, no new selection'}
    atomic_json(output/'verification.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path,default=ROOT/'paper2/results/restart_control_pilot_protocol_v1')
    p.add_argument('--study',type=Path,default=ROOT/'paper2/results/restart_control_pilot_v1');args=p.parse_args()
    print(verify(args.package,args.study,args.output))
