"""Fixed-round matched controls and simulated Gaussian training with atomic events."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from paper2.federated import clone_state
from paper2.restart.work_scaled_regression_training import make_regressor,train_regression_client
from paper2.restart.noisy_aggregation import aggregate_simulation
from paper2.restart.privacy_accounting import clip_vector,replacement_sensitivity,gaussian_budget
from paper2.restart.validation_search import read,sha,atomic_json,atomic_checkpoint


def control_average(reference,clients,slots,clip_norm):
    if not slots or len(set(slots))!=len(slots) or set(clients)!=set(slots):raise ValueError('Fixed slot coverage required')
    keys=sorted(reference);base=[];deltas=[]
    for k in keys:
        v=reference[k]
        if v.device.type!='cpu' or v.dtype not in (torch.float32,torch.float64) or not torch.isfinite(v).all():raise ValueError('Invalid reference')
        base.extend(v.detach().numpy().astype(np.float64).ravel())
    base=np.array(base)
    for slot in slots:
        state=clients[slot]
        if set(state)!=set(keys):raise ValueError('Parameter keys changed')
        for k in keys:
            if state[k].shape!=reference[k].shape or state[k].dtype!=reference[k].dtype or state[k].device.type!='cpu' or not torch.isfinite(state[k]).all():raise ValueError('Invalid client state')
        delta=np.concatenate([state[k].detach().numpy().astype(np.float64).ravel() for k in keys])-base
        deltas.append(delta if clip_norm is None else clip_vector(delta,clip_norm))
    vector=base+sum(d/len(slots) for d in deltas);offset=0;result={}
    for k in keys:
        n=reference[k].numel();result[k]=torch.from_numpy(vector[offset:offset+n].copy()).reshape(reference[k].shape).to(reference[k].dtype);offset+=n
        if not torch.isfinite(result[k]).all():raise FloatingPointError('Control aggregate overflow')
    return result


def expected_events(run,slots,done):
    v=run['variant']
    if v['noise_multiplier'] is None:return []
    sensitivity=replacement_sensitivity(v['clip_norm'],len(slots))
    return [{'round':r,'event':{'sensitivity':sensitivity,'noise_std':sensitivity*v['noise_multiplier'],'count':1}} for r in range(1,done+1)]


def fit_simulation(folder,run,settings,fold,x,y,token,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if x.dtype!=np.float32 or y.dtype!=np.float32 or x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid training arrays')
    if type(run['rounds']) is not int or not 1<=run['rounds']<=80:raise ValueError('Invalid fixed budget')
    slots=tuple(sorted(fold['clients']));lookup={row:i for i,row in enumerate(fold['train'])}
    membership=[i for ids in fold['clients'].values() for i in ids]
    if len(lookup)!=len(y) or sorted(membership)!=sorted(lookup) or any(not ids for ids in fold['clients'].values()):raise ValueError('Client coverage mismatch')
    if (folder/'fit.json').exists():
        receipt=read(folder/'fit.json')
        if receipt['identity']!=token or sha(folder/'model.pt')!=receipt['model_sha256'] or sha(folder/'accounting.json')!=receipt['accounting_sha256']:raise ValueError('Fit assets changed')
        return receipt
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(run['seed'])
    model=make_regressor(x.shape[1],settings['hidden']);done=0;events=[]
    checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['identity']!=token or not 0<=saved['done']<=run['rounds']:raise ValueError('Checkpoint identity changed')
        done=saved['done'];events=saved['events']
        if events!=expected_events(run,slots,done):raise ValueError('Checkpoint event ledger changed')
        model.load_state_dict(saved['state'])
    tx,ty=torch.from_numpy(x),torch.from_numpy(y)
    # Independent deterministic simulation stream for each run/round. Not private randomness.
    run_stream=int.from_bytes(hashlib.sha256(('work-v3:'+run['id']).encode()).digest()[:8],'big')
    for r in range(done+1,run['rounds']+1):
        clients={}
        for number,slot in enumerate(slots):
            ids=[lookup[i] for i in fold['clients'][slot]]
            clients[slot]=train_regression_client(model,tx[ids],ty[ids],slot,r,fold['preprocessing_identity'],
                run['seed']+1009*r+number,learning_rate=run['learning_rate'],batch_size=settings['batch_size']).state
        v=run['variant']
        if v['noise_multiplier'] is None:
            state=control_average(clone_state(model),clients,slots,v['clip_norm'])
        else:
            rng=np.random.default_rng(np.random.SeedSequence([run['seed'],run_stream,r]))
            state,event,_=aggregate_simulation(clone_state(model),clients,slots,v['clip_norm'],v['noise_multiplier'],rng,simulation_only=True)
            events.append({'round':r,'event':event})
        model.load_state_dict(state)
        # State and events are committed together; recovery cannot count a round twice.
        atomic_checkpoint(checkpoint,{'identity':token,'done':r,'state':clone_state(model),'events':events})
        if stop_after==r and r<run['rounds']:return None
    atomic_checkpoint(folder/'model.pt',clone_state(model))
    atomic_json(folder/'accounting.json',{'identity':token,'events':events,'rounds':run['rounds'],
        'conditional_gaussian_budget':gaussian_budget([e['event'] for e in events],1e-5) if events else None,
        'unnoised_control':not events,'mode':'SIMULATION_ONLY','application_privacy_certified':False})
    receipt={'identity':token,'model':'model.pt','model_sha256':sha(folder/'model.pt'),
        'accounting_sha256':sha(folder/'accounting.json'),'history':[{'round':r} for r in range(1,run['rounds']+1)]}
    atomic_json(folder/'fit.json',receipt);return receipt
