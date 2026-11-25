"""Complete matched v3 utility summary; no tuning, training or reinference."""
import argparse
from pathlib import Path
import numpy as np
from paper2.restart.validation_search import ROOT,read,sha,atomic_json,writer_lock
from paper2.restart.verify_work_scaled_privacy_simulation import audit
from paper2.restart.work_scaled_privacy_simulation_runner import verify_execution
from paper2.restart.summarize_outer import aggregate


def summarize(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if ROOT/'paper2/results' not in output.parents:raise ValueError('New results directory required')
    package=ROOT/'paper2/results/restart_privacy_execution_v3'
    study=ROOT/'paper2/results/restart_privacy_simulation_v3'
    _,config,plan,data=verify_execution(package)
    checked=audit(package,study,output/'audit')
    if checked['status']!='PASS_COMPLETE':raise ValueError('Complete matched matrix required')
    report=read(output/'audit/verification.json');buckets={}
    with writer_lock(study):
        for row in report['rows']:
            run=next(r for r in plan if r['id']==row['id']);folder=study/'runs'/run['id']
            if sha(folder/'completion.json')!=row['completion_sha256']:raise ValueError('Receipt changed')
            receipt=read(folder/'completion.json')
            if any(sha(folder/n)!=v for n,v in receipt['assets'].items()):raise ValueError('Prediction assets changed')
            fold=read(data/'folds'/run['fold']/'partition.json')
            with np.load(folder/'test_cache.npz',allow_pickle=False) as cached:
                species=np.array([s['all_species_seen'] for s in fold['support']['test']],bool)
                composition=np.array([s['composition_seen'] for s in fold['support']['test']],bool)
                for panel,mask in (('all',np.ones(len(cached['y']),bool)),('species_seen',species),('species_unseen',~species),
                                   ('composition_seen',composition),('composition_unseen',~composition)):
                    if mask.any():
                        buckets.setdefault((row['variant'],run['seed'],panel),[]).append(
                            {'ids':cached['ids'][mask].copy(),'y':cached['y'][mask].copy(),'p':cached['prediction'][mask].copy()})
    per_seed=[];summary=[]
    for variant in config['variants']:
        name=variant['name']
        for panel in ('all','species_seen','species_unseen','composition_seen','composition_unseen'):
            values=[]
            for seed in config['seeds']:
                groups=buckets.get((name,seed,panel),[])
                if panel=='all' and len(groups)!=8:raise ValueError('Incomplete regional coverage')
                measured=aggregate(groups)
                if measured is None or panel=='all' and measured['pooled']['samples']!=607:raise ValueError('Incomplete sample coverage')
                values.append(measured);per_seed.append({'variant':name,'seed':seed,'panel':panel,'metrics':measured})
            row={'variant':name,'panel':panel,'groups':values[0]['groups'],'samples_per_seed':values[0]['pooled']['samples']}
            for metric in ('equal_group_rmse','equal_group_mae','pooled_rmse','pooled_mae','pooled_r2'):
                numbers=[v[metric] if metric.startswith('equal') else v['pooled'][metric.removeprefix('pooled_')] for v in values]
                row[metric]=None if any(n is None for n in numbers) else {'mean':float(np.mean(numbers)),'seed_sd':float(np.std(numbers,ddof=1))}
            summary.append(row)
    atomic_json(output/'analysis.json',{'status':'PASS_COMPLETE_MATCHED_SIMULATION','runs':len(plan),
        'summary':summary,'per_seed':per_seed,'per_region':report['rows'],
        'shared_noisy_events':report['noisy_events'],'conditional_noisy_only_budgets':report['per_slot_noisy_only_budgets'],
        'interpretation':'Development, failure-informed amendment; initialization SD is not regional confidence. No application privacy certification.',
        'audit_sha256':sha(output/'audit/verification.json'),'source_sha256':sha(__file__)})
    return [r for r in summary if r['panel']=='all']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(summarize(args.output))
