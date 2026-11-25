"""Cached metrics, model checkpoint and shared Gaussian-event ledger audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from paper2.restart.stable_privacy_simulation_runner import verify_execution,PROTOCOL
from paper2.restart.stable_simulation_training import expected_events
from paper2.restart.validation_search import ROOT,read,sha,identity,atomic_json,writer_lock
from paper2.restart.outer_training import verify_evaluation
from paper2.restart.verify_outer_study import scores
from paper2.restart.privacy_accounting import gaussian_budget


def audit(package,study,output):
    study,output=Path(study).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if study in output.parents or output in study.parents or output==study:raise ValueError('Separate audit output required')
    m,config,plan,data=verify_execution(package);binding=identity(m)
    if read(study/'binding.json')['identity']!=binding:raise ValueError('Study binding changed')
    rows=[];events=[];by_slot={}
    if config.get('prior_ledger_file'):
        prior=read(PROTOCOL/config['prior_ledger_file'])
        for entry in prior['events']:
            row=dict(entry,event_id='v1:'+entry['event_id']);events.append(row)
            for slot in row['slots']:by_slot.setdefault(slot,[]).append(row['event'])
    with writer_lock(study):
        for run in plan:
            folder=study/'runs'/run['id'];token=identity({'binding':binding,'run':run});slots=config['public_slots'][run['fold']]
            if (folder/'checkpoint.pt').exists():
                checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                if checkpoint['identity']!=token or type(checkpoint['done']) is not int or not 0<=checkpoint['done']<=run['rounds'] or checkpoint['events']!=expected_events(run,slots,checkpoint['done']):raise ValueError('Round checkpoint ledger changed')
                for entry in checkpoint['events']:
                    row={'event_id':'v2:'+run['id']+':'+str(entry['round']),'slots':slots,'event':entry['event']};events.append(row)
                    for slot in slots:by_slot.setdefault(slot,[]).append(entry['event'])
            if not (folder/'completion.json').exists():continue
            if not (folder/'checkpoint.pt').exists():raise ValueError('Completed run missing checkpoint')
            result=verify_evaluation(folder,token);fitted=read(folder/'fit.json');account=read(folder/'accounting.json')
            if result['run']!=dict(run,algorithm='fedavg') or sha(folder/'accounting.json')!=fitted['accounting_sha256']:raise ValueError('Run/accounting binding changed')
            expected=expected_events(run,slots,run['rounds'])
            if account['identity']!=token or account['rounds']!=run['rounds'] or account['events']!=expected:raise ValueError('Final event count changed')
            budget=gaussian_budget([e['event'] for e in expected],1e-5) if expected else None
            if account['conditional_gaussian_budget']!=budget or account['unnoised_control']!= (not expected):raise ValueError('Accountant/control mismatch')
            if checkpoint['done']!=run['rounds']:raise ValueError('Incomplete selected model')
            model=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            if any(not torch.equal(v,checkpoint['state'][k]) for k,v in model.items()):raise ValueError('Model differs from final checkpoint')
            fold=read(data/'folds'/run['fold']/'partition.json')
            with np.load(folder/'test_cache.npz',allow_pickle=False) as cache,np.load(data/'folds'/run['fold']/'arrays.npz',allow_pickle=False) as source:
                np.testing.assert_array_equal(cache['ids'],source['ids_test']);np.testing.assert_array_equal(cache['y'],source['y_test'])
                measured=scores(cache['y'],cache['prediction'])
                if any(not np.isclose(measured[k],v,rtol=1e-12,atol=1e-12) for k,v in result['test'].items()):raise ValueError('Independent metric mismatch')
                panels={}
                for name in ('all_species_seen','composition_seen'):
                    mask=np.array([r[name] for r in fold['support']['test']],bool)
                    panels[name]=scores(cache['y'][mask],cache['prediction'][mask]) if mask.any() else None
            rows.append({'id':run['id'],'variant':run['variant']['name'],'seed':run['seed'],'group':fold['test_group'],
                'metrics':measured,'support_panels':panels,'conditional_budget':budget,'completion_sha256':sha(folder/'completion.json')})
        ledger=read(study/'shared_ledger.json')
        budgets=[{'slot':s,**gaussian_budget(e,1e-5)} for s,e in sorted(by_slot.items())]
        if ledger['events']!=events or ledger['event_count']!=len(events) or ledger['per_slot']!=budgets:raise ValueError('Shared ledger mismatch')
        if len({r['event_id'] for r in events})!=len(events):raise ValueError('Duplicate events')
        output.mkdir(parents=True)
        report={'status':'PASS_COMPLETE' if len(rows)==len(plan) else 'PASS_PARTIAL','verified':len(rows),'planned':len(plan),
            'pending':len(plan)-len(rows),'noisy_events':len(events),'rows':rows,'per_slot_noisy_only_budgets':budgets,
            'model_reinference':False,'application_privacy_certified':False,'mode':'PUBLIC_DATA_SIMULATION'}
        atomic_json(output/'verification.json',report)
    return {k:report[k] for k in ('status','verified','planned','pending','noisy_events','model_reinference')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path,default=ROOT/'paper2/results/restart_privacy_execution_v2')
    p.add_argument('--study',type=Path,default=ROOT/'paper2/results/restart_privacy_simulation_v2');args=p.parse_args()
    print(audit(args.package,args.study,args.output))
