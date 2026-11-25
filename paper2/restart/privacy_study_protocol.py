"""Freeze a matched simulation utility study and hypothetical transcript budgets."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from paper2.restart.validation_search import ROOT,read,sha,atomic_json,runtime
from paper2.restart.outer_runner import verify_runner
from paper2.restart.prepare_public_data import verify,BASE,new_output
from paper2.restart.privacy_accounting import gaussian_budget,replacement_sensitivity

VARIANTS=[{'name':'equal_average','clip_norm':None,'noise_multiplier':None},
          {'name':'clipped_average','clip_norm':1.,'noise_multiplier':None},
          *[{'name':'noisy_sigma'+str(s),'clip_norm':1.,'noise_multiplier':float(s)} for s in (2,4,8)]]
CODE=['privacy_study_protocol.py','prepare_public_data.py','public_preprocessing.py','noisy_aggregation.py','privacy_accounting.py']


def make_plan(parent_plan,variants):
    if len({v['name'] for v in variants})!=len(variants):raise ValueError('Duplicate variants')
    plan=[]
    for parent in parent_plan:
        if parent['algorithm']!='fedavg':continue
        if type(parent['rounds']) is not int or not 1<=parent['rounds']<=80:raise ValueError('Bad matched round budget')
        for variant in variants:
            if variant['noise_multiplier'] is not None and (variant['clip_norm'] is None or variant['noise_multiplier']<=0):
                raise ValueError('Noise requires clipping and positive multiplier')
            plan.append({'id':parent['fold']+'_'+variant['name']+'_seed'+str(parent['seed']),
                'fold':parent['fold'],'seed':parent['seed'],'rounds':parent['rounds'],
                'learning_rate':parent['learning_rate'],'variant':dict(variant)})
    if not plan or len({r['id'] for r in plan})!=len(plan):raise ValueError('Invalid plan coverage')
    return plan


def freeze(data,output):
    data=Path(data).resolve();verified=verify(data);output=new_output(output,data)
    record,_,config,parent_plan=verify_runner(BASE);plan=make_plan(parent_plan,VARIANTS)
    public_slots={r['fold']:tuple(sorted(read(data/'folds'/r['fold']/'partition.json')['clients'])) for r in plan}
    if any(len(v)!=7 for v in public_slots.values()):raise ValueError('Expected seven fixed public slots per outer fold')
    # This roster comes from a PUBLIC benchmark; future private deployment must
    # supply an independently public roster and transcript policy.
    groups=sorted(set().union(*map(set,public_slots.values())));budgets=[]
    for group in groups:
        events=[]
        for run in plan:
            v=run['variant']
            if v['noise_multiplier'] is not None and group in public_slots[run['fold']]:
                sensitivity=replacement_sensitivity(v['clip_norm'],len(public_slots[run['fold']]))
                events.append({'sensitivity':sensitivity,'noise_std':sensitivity*v['noise_multiplier'],'count':run['rounds']})
        budgets.append({'protected_slot':group,**gaussian_budget(events,1e-5)})
    output.mkdir(parents=True);(output/'source_snapshot').mkdir()
    for n in CODE:shutil.copyfile(Path(__file__).with_name(n),output/'source_snapshot'/n)
    contract={'version':'public-transform-utility-simulation-v1','seeds':config['seeds'],'variants':VARIANTS,
        'neural_settings':config['neural_settings'],'public_slots':public_slots,
        'selection':'Carry original FedAvg LR/rounds unchanged; no new tuning; same settings across matched variants.',
        'controls':'equal slot average without clipping/noise; clipped equal average without noise',
        'scope':'PUBLIC_DATA_DEVELOPMENT_SIMULATION; existing held-out outcomes already exposed',
        'noise_mode':'NumPy simulation; distinct RNG stream from client shuffling; never a production privacy release',
        'initialization':'Same fresh data-independent seed initialization per fold/seed for all variants.',
        'checkpoint':'last fixed round; no early stopping or test selection','participation':1.,
        'accounting':'Every noisy round counted; same clients across folds/seeds/configs compose.',
        'private_controls_warning':'Releasing unnoised controls on private data invalidates a joint DP transcript; no full-experiment guarantee.',
        'release_ledger_required':True,'runner_implemented':False,'application_privacy_certified':False}
    atomic_json(output/'config.json',contract);atomic_json(output/'plan.json',plan)
    atomic_json(output/'hypothetical_noisy_transcript_budgets.json',budgets)
    atomic_json(output/'manifest.json',{'status':'FROZEN_PLAN_ONLY_NO_TRAINING','planned':len(plan),'data':str(data),
        'data_manifest_sha256':verified['manifest_sha256'],'parent_execution_manifest_sha256':sha(BASE/'manifest.json'),
        'runtime':runtime(),'artifacts':{n:sha(output/n) for n in ('config.json','plan.json','hypothetical_noisy_transcript_budgets.json')},
        'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in CODE},
        'runner_status':'Pending matched controls, checkpoint/release-ledger implementation and execution-code freeze'})
    return {'status':'FROZEN_PLAN_ONLY_NO_TRAINING','planned':len(plan),'variants':len(VARIANTS),'protected_slots':len(groups)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'paper2/results/restart_public_data_v1');args=p.parse_args()
    print(freeze(args.data,args.output))
