"""Freeze public-work sensitivity calibration; no fitting to client norms/errors."""
import argparse
from pathlib import Path
import shutil
from paper2.restart.stable_privacy_simulation_runner import verify_protocol,PROTOCOL
from paper2.restart.validation_search import ROOT,read,sha,atomic_json,writer_lock
from paper2.restart.prepare_public_data import new_output
from paper2.restart.privacy_accounting import replacement_sensitivity,gaussian_budget


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old,config,plan,data=verify_protocol();output=new_output(args.output,data)
    local_steps=4
    for run in plan:
        if run['variant']['clip_norm'] is not None:run['variant']['clip_norm']=run['learning_rate']*local_steps
    config.update(version='public-transform-utility-simulation-v3',public_local_steps=local_steps,
        local_gradient_policy='Fixed global gradient norm1, exactly four SGD minibatch steps/client/round',
        update_clip_policy='C=learning_rate*4; a public optimizer bound, not a fitted norm statistic',
        noise_stream='work-v3 run/round stream; distinct from prior versions',
        amendment='Align delta clipping/noise units with public gradient-clipped optimizer work; controls use same work')
    config['neural_settings']['local_epochs']=None
    config['neural_settings']['public_local_steps']=4
    for v in config['variants']:
        if v['clip_norm'] is not None:v['clip_norm']='learning_rate*4'
    legacy=ROOT/'paper2/results/restart_privacy_simulation_v2'
    with writer_lock(legacy):
        prior=read(legacy/'shared_ledger.json')
        output.mkdir(parents=True);(output/'source_snapshot').mkdir()
        for name in old['source_sha256']:shutil.copyfile(PROTOCOL/'source_snapshot'/name,output/'source_snapshot'/name)
        source=dict(old['source_sha256'])
        for name in ('work_scaled_regression_training.py','work_scaled_simulation_training.py','work_scaled_privacy_simulation_runner.py','amend_privacy_work_scale.py'):
            path=Path(__file__).with_name(name);source[name]=sha(path);shutil.copyfile(path,output/'source_snapshot'/name)
        atomic_json(output/'config.json',config);atomic_json(output/'plan.json',plan)
        atomic_json(output/'prior_ledger.json',prior)
        groups=sorted(set().union(*map(set,config['public_slots'].values())));budgets=[]
        for group in groups:
            events=[e['event'] for e in prior['events'] if group in e['slots']]
            for run in plan:
                v=run['variant'];slots=config['public_slots'][run['fold']]
                if v['noise_multiplier'] is not None and group in slots:
                    sensitivity=replacement_sensitivity(v['clip_norm'],len(slots))
                    events.append({'sensitivity':sensitivity,'noise_std':sensitivity*v['noise_multiplier'],'count':run['rounds']})
            budgets.append({'protected_slot':group,**gaussian_budget(events,1e-5)})
        atomic_json(output/'hypothetical_noisy_transcript_budgets.json',budgets)
        manifest=dict(old,status='FROZEN_V3_PUBLIC_WORK_CALIBRATION_NO_TRAINING',source_sha256=source,
            parent_v2_protocol_sha256=sha(PROTOCOL/'manifest.json'),
            artifacts={n:sha(output/n) for n in ('config.json','plan.json','hypothetical_noisy_transcript_budgets.json','prior_ledger.json')})
        atomic_json(output/'manifest.json',manifest)
    print({'status':manifest['status'],'planned':len(plan),'prior_committed_noisy_events':prior['event_count']})
