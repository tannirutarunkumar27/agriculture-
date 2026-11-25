"""Versioned inner-only conditioning/convergence diagnostic. No outer inference."""
import argparse
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
from paper2.restart.validation_search import ROOT,read,sha,identity,runtime,atomic_json,atomic_checkpoint,writer_lock
from paper2.restart.prepare_public_data import verify as verify_data,new_output
from paper2.restart.work_scaled_regression_training import make_regressor,train_regression_client
from paper2.restart.work_scaled_simulation_training import control_average
from paper2.federated import clone_state
from paper2.restart.verify_outer_study import scores
from paper2.restart.diagnose_public_control_fit import matched_fit

CODE=[Path(__file__),Path(__file__).with_name('work_scaled_regression_training.py'),
      Path(__file__).with_name('work_scaled_simulation_training.py'),
      Path(__file__).with_name('diagnose_public_control_fit.py'),Path(__file__).with_name('verify_outer_study.py')]


def normalized_targets(y,scale):
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not np.isfinite(scale) or scale<=0:
        raise ValueError('Positive public target scale required')
    if y.dtype!=np.float32 or y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError('Finite float32 targets required')
    value=(y/scale).astype(np.float32)
    if not np.isfinite(value).all():raise ValueError('Normalized target overflow')
    return value


def load_inner(data,fold_name):
    path=Path(data)/'folds'/fold_name;fold=read(path/'partition.json')
    if fold['kind']!='inner_selection':raise ValueError('Inner fold only')
    with np.load(path/'arrays.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'x_train','y_train','ids_train','x_validation','y_validation','ids_validation'}:
            raise ValueError('Unexpected arrays or outer test exposure')
        arrays={n:saved[n].copy() for n in saved.files}
    for split in ('train','validation'):
        np.testing.assert_array_equal(arrays['ids_'+split],fold[split])
    return fold,arrays


def candidate(folder,run,fold,arrays,token,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():
        receipt=read(folder/'completion.json')
        if receipt['identity']!=token or any(sha(folder/n)!=h for n,h in receipt['assets'].items()):raise ValueError('Completed pilot changed')
        return read(folder/'result.json')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(run['seed'])
    x,y=arrays['x_train'],arrays['y_train'];scale=run['target_scale']
    tx,ty=torch.from_numpy(x),torch.from_numpy(normalized_targets(y,scale))
    vx=torch.from_numpy(arrays['x_validation']);model=make_regressor(x.shape[1],32)
    done=0;history=[];best=float('inf');best_state=None;best_round=None;snapshots={}
    if (folder/'checkpoint.pt').exists():
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        if checkpoint['identity']!=token:raise ValueError('Checkpoint identity changed')
        done=checkpoint['done'];history=checkpoint['history'];best=checkpoint['best'];best_state=checkpoint['best_state'];best_round=checkpoint['best_round'];snapshots=checkpoint['snapshots']
        if not 0<=done<=run['rounds'] or len(history)!=done:raise ValueError('Invalid checkpoint progress')
        model.load_state_dict(checkpoint['state'])
    lookup={row:i for i,row in enumerate(fold['train'])};slots=tuple(sorted(fold['clients']))
    groups=[[lookup[i] for i in fold['clients'][slot]] for slot in slots]
    if sorted(i for g in groups for i in g)!=list(range(len(y))):raise ValueError('Client partition changed')
    for r in range(done+1,run['rounds']+1):
        clients={}
        for number,(slot,ids) in enumerate(zip(slots,groups)):
            clients[slot]=train_regression_client(model,tx[ids],ty[ids],slot,r,fold['preprocessing_identity'],
                run['seed']+1009*r+number,learning_rate=run['learning_rate'],batch_size=32).state
        model.load_state_dict(control_average(clone_state(model),clients,slots,None));model.eval()
        with torch.no_grad():p_train=model(tx)[:,0].numpy()*scale;p_val=model(vx)[:,0].numpy()*scale
        measured=scores(arrays['y_validation'],p_val);fit=matched_fit(y,p_train,groups)
        history.append({'round':r,'validation':measured,'training_equal_client_fit':fit})
        if measured['rmse']<best:best=measured['rmse'];best_state=clone_state(model);best_round=r
        if r in run['report_rounds']:snapshots[str(r)]={'last':history[-1],'best_round':best_round,'best_validation_rmse':best}
        atomic_checkpoint(folder/'checkpoint.pt',{'identity':token,'done':r,'state':clone_state(model),'history':history,
            'best':best,'best_state':best_state,'best_round':best_round,'snapshots':snapshots})
        if r==stop_after and r<run['rounds']:return None
    result={'identity':token,'run':run,'snapshots':snapshots,'history':history,'outer_model_inference':False,'privacy_claim':False}
    atomic_json(folder/'result.json',result)
    atomic_json(folder/'completion.json',{'identity':token,'assets':{n:sha(folder/n) for n in ('result.json','checkpoint.pt')}})
    return result


def freeze(output):
    data=ROOT/'paper2/results/restart_public_data_v1';verified=verify_data(data);output=new_output(output,data)
    # First two folds in fixed manifest order, no score-based choice.
    names=sorted(n for n in read(data/'manifest.json')['assets'] if '_inner_' in n)[:2]
    plan=[{'id':name+'_scale'+str(scale),'fold':name,'target_scale':scale,'learning_rate':.003,
        'rounds':240,'report_rounds':[80,240],'seed':42} for name in names for scale in (1.,10.)]
    output.mkdir(parents=True);(output/'source_snapshot').mkdir()
    for source in CODE:shutil.copyfile(source,output/'source_snapshot'/source.name)
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'manifest.json',{'status':'FROZEN_DIAGNOSTIC_NO_TRAINING','data':str(data),
        'data_manifest_sha256':verified['manifest_sha256'],'plan_sha256':sha(output/'plan.json'),
        'source_sha256':{p.name:sha(p) for p in CODE},'runtime':runtime(),'planned':len(plan),
        'scope':'Exploratory control pilot, training/inner validation only; all prior development exposure disclosed',
        'selection':'No final study hyperparameters selected from this two-fold pilot'})
    return {'status':'FROZEN_DIAGNOSTIC_NO_TRAINING','planned':len(plan)}


def execute(package,output):
    package=Path(package);m=read(package/'manifest.json');data=Path(m['data'])
    if any(sha(p)!=m['source_sha256'][p.name] or sha(package/'source_snapshot'/p.name)!=sha(p) for p in CODE) or runtime()!=m['runtime'] or sha(package/'plan.json')!=m['plan_sha256']:
        raise ValueError('Frozen pilot code/runtime changed')
    if verify_data(data)['manifest_sha256']!=m['data_manifest_sha256']:raise ValueError('Public data changed')
    output=new_output(output,data)
    if output==package.resolve() or package.resolve() in output.parents:raise ValueError('Separate pilot output required')
    output.mkdir(parents=True);results=[]
    with writer_lock(output):
        for run in read(package/'plan.json'):
            folder=output/'runs'/run['id'];fold,arrays=load_inner(data,run['fold'])
            try:results.append(candidate(folder,run,fold,arrays,identity({'protocol':m,'run':run})))
            except Exception:
                folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'failure.json',{'traceback':traceback.format_exc()});raise
        atomic_json(output/'summary.json',{'status':'COMPLETE_INNER_ONLY_PILOT','completed':len(results),
            'outer_model_inference':False,'runs':[{'id':r['run']['id'],'snapshots':r['snapshots']} for r in results],
            'interpretation':'Two-fold pilot does not establish final settings or independent confirmation'})
    return [{'id':r['run']['id'],'snapshots':r['snapshots']} for r in results]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','execute']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path,default=ROOT/'paper2/results/restart_control_pilot_protocol_v1');args=p.parse_args()
    print(freeze(args.output) if args.action=='freeze' else execute(args.package,args.output))
