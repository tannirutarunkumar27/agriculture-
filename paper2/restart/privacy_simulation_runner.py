"""Capped simulation runner; immutable round ledger and once-only test inference."""
import argparse
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
from paper2.restart.validation_search import ROOT,read,sha,identity,atomic_json,writer_lock,runtime
from paper2.restart.prepare_public_data import verify as verify_data,BASE,new_output
from paper2.restart.outer_runner import verify_runner,load_fold
from paper2.restart.outer_training import evaluate_once,verify_evaluation
from paper2.restart.simulation_training import fit_simulation,expected_events
from paper2.restart.privacy_accounting import gaussian_budget

PROTOCOL=ROOT/'paper2/results/restart_privacy_study_protocol_v1'
CODE=[Path(__file__),Path(__file__).with_name('simulation_training.py'),Path(__file__).with_name('outer_training.py')]


def verify_protocol():
    m=read(PROTOCOL/'manifest.json')
    if runtime()!=m['runtime'] or sha(BASE/'manifest.json')!=m['parent_execution_manifest_sha256']:raise ValueError('Protocol parent/runtime changed')
    for n,h in m['artifacts'].items():
        if sha(PROTOCOL/n)!=h:raise ValueError('Protocol artifact changed')
    for n,h in m['source_sha256'].items():
        if sha(Path(__file__).with_name(n))!=h or sha(PROTOCOL/'source_snapshot'/n)!=h:raise ValueError('Protocol code changed')
    data=Path(m['data']);verified=verify_data(data)
    if verified['manifest_sha256']!=m['data_manifest_sha256']:raise ValueError('Data binding changed')
    return m,read(PROTOCOL/'config.json'),read(PROTOCOL/'plan.json'),data


def freeze(output):
    _,_,plan,data=verify_protocol();output=new_output(output,data)
    output.mkdir(parents=True);(output/'source_snapshot').mkdir()
    for source in CODE:shutil.copyfile(source,output/'source_snapshot'/source.name)
    atomic_json(output/'manifest.json',{'protocol_manifest_sha256':sha(PROTOCOL/'manifest.json'),
        'source_sha256':{s.name:sha(s) for s in CODE},'runtime':runtime(),'planned':len(plan),'mode':'SIMULATION_ONLY'})
    return {'status':'FROZEN_EXECUTION_NO_TRAINING','planned':len(plan)}


def verify_execution(package):
    package=Path(package);m=read(package/'manifest.json')
    if runtime()!=m['runtime'] or sha(PROTOCOL/'manifest.json')!=m['protocol_manifest_sha256']:raise ValueError('Execution binding changed')
    for p in CODE:
        if sha(p)!=m['source_sha256'][p.name] or sha(package/'source_snapshot'/p.name)!=sha(p):raise ValueError('Execution source changed')
    _,config,plan,data=verify_protocol();return m,config,plan,data


def shared_ledger(output,plan,config,binding):
    rows=[];by_slot={}
    for run in plan:
        path=output/'runs'/run['id']/'checkpoint.pt'
        if not path.exists():continue
        saved=torch.load(path,map_location='cpu',weights_only=True);token=identity({'binding':binding,'run':run})
        slots=tuple(config['public_slots'][run['fold']])
        if saved['identity']!=token or type(saved['done']) is not int or not 0<=saved['done']<=run['rounds'] or saved['events']!=expected_events(run,slots,saved['done']):raise ValueError('Checkpoint ledger mismatch')
        for entry in saved['events']:
            rows.append({'event_id':run['id']+':'+str(entry['round']),'slots':list(slots),'event':entry['event']})
            for slot in slots:by_slot.setdefault(slot,[]).append(entry['event'])
    if len({r['event_id'] for r in rows})!=len(rows):raise ValueError('Duplicate accounting event')
    path=output/'shared_ledger.json'
    if path.exists():
        old=read(path);new={r['event_id']:r for r in rows}
        if any(new.get(r['event_id'])!=r for r in old['events']):raise ValueError('Ledger event removed/changed')
    ledger={'events':rows,'event_count':len(rows),'per_slot':[{'slot':s,**gaussian_budget(e,1e-5)} for s,e in sorted(by_slot.items())],
        'mode':'SIMULATION_ACCOUNTING','excludes_unnoised_controls':True,'application_privacy_certified':False}
    atomic_json(path,ledger);return len(rows)


def execute(package,output,cap=1,resume=False,dry_run=True):
    if type(cap) is not int or cap<1:raise ValueError('Positive cap required')
    package,output=Path(package).resolve(),Path(output).resolve();m,config,plan,data=verify_execution(package)
    if ROOT/'paper2/results' not in output.parents or any(output==p or p in output.parents or output in p.parents for p in (package,PROTOCOL,data)):
        raise ValueError('Unsafe output path')
    if dry_run:return {'status':'DRY_RUN','planned':len(plan),'cap':cap}
    if output.exists() and not resume:raise ValueError('Explicit resume required')
    if not output.exists():output.mkdir(parents=True)
    binding=identity(m);done=0;executed=0
    with writer_lock(output):
        if (output/'binding.json').exists():
            if read(output/'binding.json')['identity']!=binding:raise ValueError('Output binding changed')
        elif resume or any(p.name!='writer.lock' for p in output.iterdir()):raise ValueError('Unbound output')
        else:atomic_json(output/'binding.json',{'identity':binding})
        shared_ledger(output,plan,config,binding)
        for run in plan:
            folder=output/'runs'/run['id'];token=identity({'binding':binding,'run':run})
            if (folder/'completion.json').exists():
                result=verify_evaluation(folder,token)
                if result['run']!=dict(run,algorithm='fedavg'):raise ValueError('Completed run mismatch')
                fitted=read(folder/'fit.json')
                if sha(folder/'accounting.json')!=fitted['accounting_sha256']:raise ValueError('Completed accounting changed')
                done+=1;continue
            if executed>=cap:continue
            if (folder/'failure.json').exists() or (folder/'inference_started.json').exists():raise RuntimeError('Inspect failed/ambiguous simulation before recovery')
            folder.mkdir(parents=True,exist_ok=True)
            try:
                fold,a=load_fold(data,run)
                if sorted(fold['clients'])!=config['public_slots'][run['fold']]:raise ValueError('Roster changed')
                fit_simulation(folder,run,config['neural_settings'],fold,a['x_train'],a['y_train'],token)
                shared_ledger(output,plan,config,binding)
                evaluate_once(folder,dict(run,algorithm='fedavg'),config['neural_settings'],a['x_test'],a['y_test'],a['ids_test'],token)
                done+=1;executed+=1
            except Exception:
                atomic_json(folder/'failure.json',{'traceback':traceback.format_exc()})
                shared_ledger(output,plan,config,binding)
                raise
        events=shared_ledger(output,plan,config,binding)
        result={'status':'COMPLETE' if done==len(plan) else 'PARTIAL','completed':done,'planned':len(plan),
            'pending':len(plan)-done,'executed':executed,'committed_noisy_events':events,'mode':'SIMULATION_ONLY'}
        atomic_json(output/'progress.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path,default=ROOT/'paper2/results/restart_privacy_execution_v1')
    p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--cap',type=int,default=1);args=p.parse_args()
    print(freeze(args.output) if args.freeze else execute(args.package,args.output,args.cap,args.resume,not args.execute))
